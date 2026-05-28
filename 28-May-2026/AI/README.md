# Calibrating Conservatism for Scalable Oversight 

**Authors**: William Overman, Mohsen Bayati  

**Link**: [PDF](https://arxiv.org/pdf/2605.28807)  

**Abstract**: Agentic AI systems capable of autonomous planning and extended environmental interaction pose a fundamental control problem: how can humans maintain meaningful oversight of systems that may exceed their own capabilities? Existing approaches to scalable oversight rely on complex assumptions, remain largely heuristic, or lack practical methods for sequential settings with statistical guarantees. We introduce Calibrated Collective Oversight (CCO), which aggregates diverse auxiliary scoring functions into a penalty measuring deviation from a conservative baseline. Inspired by Attainable Utility Preservation, CCO enables collective conservatism: actions face a penalty proportional to overseer concern, so high-utility actions are still selected when overseers find them unobjectionable and overridden only when concern accumulates. CCO calibrates this conservatism online using Conformal Decision Theory, ensuring that undesirable outcomes remain below a user-specified target threshold with finite-time bounds and no distributional assumptions. On a modified version of SWE-bench, weaker overseers successfully constrain an adversarially misaligned stronger agent; on MACHIAVELLI, CCO substantially reduces ethical violations while preserving reward. In both settings, empirical violation rates closely match the specified targets, as predicted by the theory. 

---
# CaMBRAIN: Real-time, Continuous EEG Inference with Causal State Space Models 

**Authors**: Abhilash Durgam, Nyle Siddiqui, Jeffrey A. Chan-Santiago, Qiushi Fu, Elakkat D. Gireesh, Mubarak Shah  

**Link**: [PDF](https://arxiv.org/pdf/2605.28792)  

**Abstract**: Electroencephalography (EEG) is a critical, non-invasive method to monitor electrical brain activity. EEGs can span anywhere from a couple seconds to multiple hours, posing a major hurdle for existing deep learning methods due to two major factors: (1) existing EEG models are predominantly built upon the attention mechanism, incurring quadratic scaling as the sequence length increases, and (2) raw EEG signals must be processed in a sliding-window fashion due to fixed-length input requirements, preventing global understanding of the entire signal. To this extent, we propose CaMBRAIN - the first Causal, Mamba-based state space model (SSM) capable of real-time inference of EEG signals, arguing that bidirectional approaches are needlessly expensive given the causal, unidirectional nature of EEG. However, training such a model is non-trivial, as crucial EEG events can be extremely brief - within fractions of a second - yet separated by long intervals spanning minutes. Current EEG methods use self-supervised objectives that optimize for signal reconstruction, but these are not well suited for streaming SSMs; they fail to explicitly train the hidden state to retain the salient long-range context needed for streaming inference. We therefore introduce a multi-stage self-supervised training pipeline specifically tailored to encourage long-range memory retention and strong performance on EEG signals, while preserving the linear-time complexity of state space models. CaMBRAIN achieves state-of-the-art (SOTA) results across 3 different EEG datasets with >10x higher throughput than existing models, enabling the first model capable of long-range, continuous inference of variable-length EEG signals. 

---
# SwarmHarness: Skill-Based Task Routing via Decentralized Incentive-Aligned AI Agent Networks 

**Authors**: Edwin Jose  

**Link**: [PDF](https://arxiv.org/pdf/2605.28764)  

**Abstract**: Vast quantities of compute (GPU cycles on personal workstations, idle inference servers, and edge devices between jobs) go unused because no incentive-aligned protocol exists for their owners to share them safely and profitably. Existing approaches either require a trusted central coordinator (cloud marketplaces), demand heavy blockchain infrastructure (Golem, BrokerChain), or lack an incentive layer entirely (BOINC, Petals). We propose SwarmHarness, a decentralised protocol in which HarnessAPI skill nodes self-organise into a compute swarm without any central authority. SwarmHarness has three interlocking components: a SwarmRegistry built on a Distributed Hash Table (DHT) for peer discovery and capability advertisement; a SwarmRouter that dispatches tasks to nodes using a utility function over capability, load, latency, and trust; and SwarmCredit, an incentive mechanism that attributes compute-credit rewards to contributing nodes via a Shapley-value approximation. Nodes earn credits by serving tasks and spend credits to submit them; idle nodes that never contribute drain credits and lose routing priority, creating a self-regulating participation economy. As nodes specialise toward high-reward skills and routing signals act as digital pheromones, the network exhibits emergent collective intelligence analogous to biological swarms. Beyond compute sharing, SwarmHarness is a foundational primitive for autonomous distributed AI agent networks in which agents hire compute, route subtasks, and settle credits without human intermediation. 

---
# CubePart: An Open-Vocabulary Part-Controllable 3D Generator 

**Authors**: Yiheng Zhu, Kangle Deng, Jean-Philippe Fauconnier, Inaki Navarro, Daiqing Li, Ava Pun, Yinan Zhang, Peiye Zhuang, Xiaoxia Sun, Maneesh Agrawala, Kiran Bhat, Tinghui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.28763)  

**Abstract**: Interactive 3D assets used in games and simulation are typically decomposed into specific semantic parts to support animation, physics, and scripted behaviors, yet most generative 3D models produce either monolithic meshes or arbitrary part decompositions that cannot be aligned with application-specific requirements. We present CubePart, a generative framework for open-vocabulary, part-controllable 3D mesh generation that exposes part structure as an explicit inference-time control signal. Given a global text prompt and a user-defined parts schema expressed as an open-ended list of part names, our method generates a set of meshes - one per schema element - that assemble into a coherent object while respecting the specified semantic structure. To enable this capability, we introduce a scalable data pipeline to construct a large open-vocabulary, part-labeled 3D dataset, along with a two-stage generative architecture that separates global shape synthesis from part-level decoding. We demonstrate that the resulting assets can be directly integrated into game engines and driven by animation and behavior scripts without manual post-processing. Project Page: this https URL 

---
# CORE: Contrastive Reflection Enables Rapid Improvements in Reasoning 

**Authors**: Linas Nasvytis, Simon Jerome Han, Ben Prystawski, Satchel Grant, Noah D. Goodman, Judith E. Fan  

**Link**: [PDF](https://arxiv.org/pdf/2605.28742)  

**Abstract**: Language models can use verifiable rewards to improve at a wide variety of reasoning tasks. However, both parametric (e.g. RLVR) and non-parametric (e.g. prompt optimization) approaches to doing so typically require hundreds of training samples and thousands of model rollouts, making them expensive in the best case and intractable in the worst. To address this challenge, we introduce Contrastive Reflection (CORE), a non-parametric learning algorithm that compares past reasoning traces to generate insights: short natural-language descriptions of reasoning strategies and constraints that capture differences between successful and unsuccessful problem attempts. Across four reasoning tasks, we demonstrate that CORE enables more rapid improvement than both parametric (GRPO) and non-parametric (GEPA, episodic RAG, and MemRL) methods, while using fewer rollouts. Under fixed rollout budgets with as few as five training samples, we then show that CORE also achieves comparable or greater performance gains than each baseline. Finally, we highlight how CORE is also substantially more context-efficient than non-parametric baselines, requiring fewer prompt tokens while storing learned knowledge as compact, interpretable natural-language insights. Our results therefore suggest that distilling contrasts between successful and unsuccessful reasoning traces into abstract and useful insights can provide a more efficient and interpretable route to model self-improvement than weight updates, prompt optimization, or direct reuse of stored reasoning traces. 

---
# Utility-Aware Multimodal Contrastive Learning for Product Image Generation 

**Authors**: Xiaohang Feng, Yiling Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.28733)  

**Abstract**: Product images strongly influence consumer decision-making in online marketplaces. Empowered by multimodal contrastive learning, generative AI can output images that closely align with text prompts. Yet existing generative AI models do not directly optimize marketplace performance. This is a critical gap, since semantic alignment alone does not guarantee that an image will sell. To address this limitation, we propose a \textit{utility-aware multimodal contrastive learning} framework that incorporates consumer demand into a novel Utility-Aware InfoNCE loss. Optimizing this utility-aware objective guides generation toward images that are both semantically coherent and demand-enhancing. This effect arises directly from a shift in the learned image-text representation space toward demand-driven visual cues, which we also validate through the theoretical bound of the proposed objective. In downstream applications on Amazon and Airbnb, product images generated and edited by our method outperform state-of-the-art models in increasing demand and preserving fidelity, while maintaining text-image consistency. Notably, our utility-aware framework preserves inverse U-shaped demand patterns for attributes such as aesthetics and uniqueness, improving demand-based performance while preserving fidelity and semantic consistency. Human-subject experiments further validate its commercial effectiveness. As generative AI technology continues to evolve, our utility-aware component can be flexibly embedded into emerging generative models to improve direct commercial use. 

---
# AlphaTransit: Learning to Design City-scale Transit Routes 

**Authors**: Bibek Poudel, Sai Swaminathan, Weizi Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.28730)  

**Abstract**: Designing a transit network requires many sequential route extension decisions, but their quality is often visible only after the full network is assembled. This delayed-feedback challenge lies at the heart of the Transit Route Network Design Problem (TRNDP), where route interactions can be deceptive: an extension that appears useful locally can create transfer bottlenecks, produce redundant overlap, or reduce overall throughput. To guide route construction under delayed simulator feedback, we introduce AlphaTransit, a search-based planning framework for cityscale bus network design. AlphaTransit couples Monte Carlo Tree Search (MCTS) with a neural policy-value network: the policy proposes route extensions, the value estimates downstream design quality, and search uses these predictions to refine each decision. This provides decision-time lookahead during route construction without running simulator rollouts inside the search tree. We evaluate AlphaTransit on a new Bloomington TRNDP benchmark with realistic road topology and censusderived demand, under mixed and full transit demand settings. In the Bloomington network, AlphaTransit attains the highest service rate in both demand settings, reaching 54.6% and 82.1%, respectively. Relative to reinforcement learning without search, these correspond to 9.9% and 11.4% service rate gains; relative to MCTS without learned guidance, they correspond to 2.5% and 11.2% gains. These results suggest that coupling learned guidance with MCTS is more effective than using either approach alone for transit network design. Our code and data are publicly available in this https URL. 

---
# Multi-Adapter Representation Interventions via Energy Calibration 

**Authors**: Manjiang Yu, Hongji Li, Junwei Chen, Xue Li, Priyanka Singh, Yang Cao, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28722)  

**Abstract**: Representation intervention has emerged as a promising paradigm for aligning large language models toward desired behaviors without modifying model weights. Existing methods typically apply a fixed intervention uniformly across all inputs. However, we find that the appropriate intervention direction and strength vary substantially across samples, and such indiscriminate intervention leads to degradation of general capabilities on benign inputs. To address these challenges, we propose Multi-Adapter Representation Interventions via Energy Calibration (MARI). Specifically, we introduce a competitive multi-adapter mechanism in which specialized experts capture non-linear correction patterns and adaptively determine the appropriate intervention direction and strength for different samples. Furthermore, we design an energy-based gating module that leverages internal propagation dynamics to distinguish inputs that are applicable for intervention. Extensive experiments across diverse model families and parameter scales demonstrate that MARI achieves state-of-the-art alignment performance. Our method significantly improves performance on TruthfulQA, BBQ, and safety benchmarks, while maintaining and even improving general capabilities on tasks such as MMLU and ARC. Our code is available at this https URL. 

---
# LiveBrowseComp: Are Search Agents Searching, or Just Verifying What They Already Know? 

**Authors**: HuiMing Fan, Xiao Wang, Zheng Chu, Qianyu Wang, Zhuoyao Wang, Ming Liu, Bing Qin, XingYu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28721)  

**Abstract**: Are LLM-based search agents genuinely searching, or using the web to verify what they already know? We study this question on BrowseComp with three diagnostics. Our analysis reveals Intrinsic Knowledge Dependence (IKD): even with tool access, agents often rely on intrinsic knowledge -- information encoded in the model before retrieval -- rather than on external evidence. Agents answer up to 44.5% of BrowseComp questions without tools, generate more than half of their search queries from internally produced hypotheses rather than retrieved leads, and perform worse than closed-book baselines when answer-supporting evidence is removed. These results suggest that static search benchmarks can reward memory-backed verification rather than evidence-driven discovery, conflating what agents already know with what they can find. We then introduce LiveBrowseComp, a deep-search benchmark designed to evaluate agents beyond intrinsic coverage. It contains 335 human-authored questions whose answers depend on facts published within the 90 days preceding benchmark construction, drawn from six updated sources and filtered to exclude globally salient events. On LiveBrowseComp, all evaluated agents fall below 2% closed-book accuracy, search-augmented scores drop by 25-40 points relative to BrowseComp, and prior model rankings no longer reliably predict performance. LiveBrowseComp is available at this https URL. 

---
# OpenURMA: A Clean-Room Open Implementation of the Unified Bus Protocol 

**Authors**: Bojie Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.28717)  

**Abstract**: Modern datacenter RDMA is bottlenecked at the network interface, not the wire. A NIC running RoCE or InfiniBand holds per-connection state for every (application, remote-endpoint) pair - hundreds of megabytes at 1024-application fanout - and pays a four-traversal PCIe round trip on a 64-byte operation, inflating latency an order of magnitude beyond the wire. Both follow from the Queue Pair over PCIe abstraction RDMA inherits from InfiniBand.
Huawei's Unified Bus (UB), a public 2025 specification, changes the abstraction: it decouples per-application endpoint state from per-host transport state so connection context grows additively, exposes ordering as opt-in, and reaches remote memory through native CPU load/store to an on-chip-bus controller. UB ships in Huawei's closed Ascend 950 silicon.
OpenURMA is the first clean-room open implementation of UB's transport and transaction layers, realised at three tiers - synthesisable RTL on Alveo U50, a cycle-level two-node SystemC simulator, and a gem5 full-system scaffold - each with a matched OpenRoCE (RoCEv2 RC) baseline. The contribution is the implementation, harness, and controlled comparison closed silicon does not admit. On the canonical 64-byte remote fetch - LOAD on UB-spec Sec.8.3, READ on RoCEv2 RC - UB's load/store path delivers ~500 ns end-to-end, 4.37x below the matched baseline (2186 ns), sustains 2.80x higher throughput, and fits in ~14% of a U50's LUTs. 

---
# Thinking as Compression: Your Reasoning Model is Secretly a Context Compressor 

**Authors**: Guoxin Ma, Yibing Liu, Chengzhengxu Li, Yu Liang, Yan Wang, Yueyang Zhang, Kecheng Chen, Zhaohan Zhang, Zhiyuan Sun, Daiting Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.28713)  

**Abstract**: Context compression aims to shorten long context inputs with minimal information loss for LLM inference acceleration. While existing methods have shown promise, they typically rely on complex compression modules or compression-specific training, leaving the intrinsic capabilities of LLMs underexplored. In contrast, this work reveals that a thinking model itself can naturally compress long contexts by organizing task-relevant information. We thus derive Thinking as Compression (TaC), a new compression paradigm that treats thinking itself as compressed context. Without relying on specific dedicated compressor, TaC directly prompts the thinking model to generate thinking traces as the shortened context, already outperforming most representative compression methods. Further, given that raw thinking output may struggle with budget control and shortcut behaviors, we introduce Thinking as Compression Constrained (TaC-C), leveraging a simple reward-driven optimization framework to elicit intrinsic thinking as compact and controllable compressed context. Experiments across four long-context QA benchmarks demonstrate that TaC-C consistently outperforms existing baselines. At 4x and 8x compression ratios, it surpasses the strongest competitor by 17.4% and 23.4% in average F1, and by 15.7% and 21.7% in average Exact Match Score (EM), respectively. 

---
# Beyond Binary Moral Judgment: Modeling Ethical Pluralism in AI 

**Authors**: Aisha Aijaz, Rahul Goel, Arnav Batra, Raghava Mutharaju  

**Link**: [PDF](https://arxiv.org/pdf/2605.28707)  

**Abstract**: Critical decision-making in socially consequential spaces is increasingly involving AI systems at varying capacities. Yet, despite the ubiquity of autonomous systems, most approaches to handling autonomous moral decision-making resort to scalar or binary judgments. These methods are insufficient for acceptable moral reasoning, as they provide little explanation, leaving out imperative contextual and theoretical information that must be included to support accountability. For this, we propose a framework to model moral reasoning as a distribution over normative ethical theories or ethical pluralism. We introduce a normative ethics simplex that integrates these theories. A benchmark of 450 cases across 15 fine-grained subtheories was also prepared for the purposes of stacked ensemble learning. These cases describe ethical dilemmas in natural language and have associated extracted contextual features. The implementation of the simplex was achieved via a two-stream normative-semantic architecture. This is followed by the fusion of normative information and a sequential, stacking ensemble to learn the best fit of the three broad theories: consequentialism, virtue ethics, and deontology, and the 15 subcategories. Our experiments demonstrate that the integration of contextual and normative priors with the semantic embeddings significantly improves the performance of the classification, displaying an accuracy of 88.89%. We conducted ablation studies to show that structured ethical representations contribute beyond analogical reasoning, and the chosen stacking architecture gives the best results due to the gradual learning of granularity. Ethical pluralism is also analyzed through entropy, confidence, and visualization. Thus, modeling ethical pluralism as a probabilistic normative distribution supports human-like moral reasoning, ethical disagreement analysis, and future alignment in AI systems. 

---
# The Importance of Being Statistically Earnest: A Critical Re-evaluation of GSM-Symbolic 

**Authors**: Dominika Agnieszka Długosz, Arlindo Oliveira, Natalia Díaz Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2605.28700)  

**Abstract**: The GSM-Symbolic benchmark (Mirzadeh et al., 2025) reported consistent performance drops across 25 Large Language Models (LLMs) when tested on template-generated variants of GSM8K problems, concluding that the models lack genuine reasoning capabilities. We argue that this conclusion rests on shaky statistical ground. Re-evaluating 20 open-weight models using Generalised Linear Mixed Models with per-question random effects, we find that only half exhibit statistically significant performance changes under the original prompt format. Moreover, we identify a previously unacknowledged factor: the main GSM-Symbolic dataset contains a systematically shifted distribution of larger integers in problem texts relative to GSM-Base (K-S statistic = 0.12, p < 0.001), contradicting the original authors' claims. Controlling for this large number effect accounts for significance in roughly half the remaining cases. Among models with statistically significant performance deltas, we identify distinct, model-specific failure profiles - including fragility of variable binding, arithmetic limitations, and dual-task interference - underscoring that blanket claims about LLM reasoning are both statistically premature and mechanistically misleading. 

---
# TRACER: Turn-level Regret Matching with Inner Reinforcement Credit for Cooperative Multi-LLM Reasoning 

**Authors**: Chusen Li, Zhou Liu, Shuigeng Zhou, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28699)  

**Abstract**: Large language models increasingly rely on either reinforcement learning or multi-agent prompting to improve reasoning, yet these two paradigms remain difficult to combine. Directly applying single-agent reinforcement learning to multi-turn multi-agent systems faces following dilemmas: i) Sparse rewards, role-level free-riding and excessive training overhead. ii) Agents only imitate to collaborate. iii) Fixed collaboration protocol falls into oscillating local optimum. We introduce TRACER, a turn-level reinforcement framework for cooperative multi-LLM reasoning. TRACER separates collaborative decision making into a controller-regret layer, where controllers learn whether the agents should speak or skip the current round through regret matching, and a generation-credit layer, which optimizes proposer and reviewer utterances with role-specific GSPO rewards. This design i) assigns credit at the level of both action modes and generated utterances, thus avoiding free-riding and sparse rewards. We only expand the choices made by the controllers, thus greatly reducing computational cost of training. Moreover, ii) agents acquire collaborative capability as they learn when to utter and what to speak. Finally, iii) by designing binary actions ingeniously, we extend classical game theory established for finite action spaces to deep learning, thus achieving mathematically rigorous convergence. We train all local RL-style methods on the GSM8K training split and evaluate on held-out GSM8K, MATH500, and GPQA-Diamond to measure in-domain accuracy, cross-benchmark generalization, inference cost, and correction-preservation behavior. The resulting framework provides a compact and reproducible testbed for studying learned collaboration policies beyond fixed debate, voting, or aggregation protocols. Code is available at this https URL. 

---
# VeriTrip: A Verifiable Benchmark for Travel Planning Agents over Unstructured Web Corpora 

**Authors**: Yuting Xu, Jiayi Tian, Jian Liang, Xin Xiong, Hang Zhang, Mu Xu, Xiao-Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28683)  

**Abstract**: Existing benchmarks have laid the foundation for travel planning agents by establishing API-centric paradigms. However, as the capabilities of Autonomous Agents continue to advance, their evaluation must evolve beyond simple tool execution toward handling the inherent complexities of the open web. Current benchmarks bypass core cognitive hurdles: they fail to account for information noise, ignore multi-source factual contradictions, and overlook the necessity of grounding visual perception into logical planning. We introduce VeriTrip, a verifiable benchmark designed to meet the increasing demands for agent robustness and reliability. VeriTrip shifts the evaluation focus to evidence-grounded reasoning over unstructured multimodal web corpora. It establishes a Multimodal Retrieval Base (MRB) derived from real-world sources, forcing agents to autonomously orchestrate queries across heterogeneous data. A synchronized Verifiable Knowledge Base (VKB) enables a cell-wise verification protocol that precisely quantifies factual reliability, distinguishing systematic reasoning failures from parametric hallucinations. Our evaluations across leading MLLMs reveal a critical \textit{retrieval-reasoning trade-off}: the cognitive load of autonomous retrieval significantly erodes instruction retention. VeriTrip provides the rigorous foundation necessary for the next generation of planning agents capable of operating in unconstrained, multimodal environments. 

---
# DREAM-R: Multimodal Speculative Reasoning with RL-Based Refined Drafting, Precise Verification, and Fully Parallel Execution 

**Authors**: Yunhai Hu, Zining Liu, Xiangyang Yin, Tianhua Xia, Bo Bao, Eric Sather, Vithursan Thangarasa, Sai Qian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28678)  

**Abstract**: Speculative reasoning has recently been proposed as a means to accelerate reasoning-intensive generation in large multimodal models, but its effectiveness is often constrained by misalignment between speculative drafts and target-verified reasoning. In this work, we introduce DREAM-R, a framework that substantially improves the performance of speculative reasoning. At its core, DREAM-R employs Speculative Alignment Policy Optimization (SAPO), a reinforcement-learning objective that trains draft models to generate reasoning steps that are both faithful to target trajectories and concise. We further propose a Threshold-based Verification Mechanism (TBVM) that uses a ratio-based criterion to provide stable and interpretable acceptance of speculative steps only when positive evidence clearly dominates, thereby preventing error propagation. Building on these components, we develop a Fully Parallel Speculative Reasoning (FPSR) framework that parallelizes draft generation, target-side reasoning, and verification across multi-step reasoning, enabling early stopping and clean fallback. Experiments on reasoning-heavy benchmarks demonstrate up to speedup while preserving target-model accuracy, yielding substantial efficiency gains without compromising reasoning quality. 

---
# An LLM-Based Assistance System for Intuitive and Flexible Capability-Based Planning 

**Authors**: Luis Miguel Vieira da Silva, Nicolas König, Felix Gehlhoff  

**Link**: [PDF](https://arxiv.org/pdf/2605.28666)  

**Abstract**: In modern industry, dynamic environments and the complexity of modular and reconfigurable resources require automated planning of process sequences. Capability-based planning approaches address this by automatically generating plans from semantic knowledge models that describe resource functions in a machine-interpretable form. Their practical use, however, remains limited: solver feedback, especially in the case of unsatisfiability, is difficult to interpret, and the knowledge models require adaptation as operational conditions change or requests become infeasible. This paper presents a hybrid assistance system that augments an existing capability-based Satisfiability Modulo Theories (SMT) planning approach with an Large Language Model (LLM)-based layer for natural-language interaction, explanation, and adaptation. Formal planning correctness remains with the symbolic planner, while the LLM layer handles natural-language access and flexible knowledge model adaptation under explicit Human-in-the-Loop (HitL) approval. The system decomposes into four components: Capability Grounding, Symbolic Planning, Result Interpretation, and Planning Adaptation, realized as a routed agentic workflow in which a central router delegates to five specialized agents. The system is evaluated on a modular production system across four scenario types. Of 23 test cases, 9 of 10 knowledge queries and all 4 satisfiable planning cases were handled correctly, 3 of 4 unsatisfiable cases produced concrete repair proposals, and all 5 adaptive planning scenarios resolved into satisfiable plans through iterative, user-approved knowledge model modifications. The findings confirm that combining formal planning with LLM-based assistance substantially improves accessibility and adaptability in industrial automation. 

---
# AutoScientists: Self-Organizing Agent Teams for Long-Running Scientific Experimentation 

**Authors**: Shanghua Gao, Ada Fang, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2605.28655)  

**Abstract**: Scientific research proceeds through iterative cycles of hypothesis generation, experiment design, execution, and revision. AI agents can automate parts of this process, but existing approaches typically follow a single research trajectory or coordinate through a central planner with fixed objectives. As a result, they struggle to sustain parallel exploration, adapt as experimental evidence changes, or preserve knowledge of failed directions over long-running experiments. We introduce AutoScientists, a decentralized team of AI agents for long-running computational scientific experimentation. Agents interpret a shared experimental state, self-organize into teams around promising hypotheses, critique proposals before using experimental compute, and share successes and failures to reduce redundant exploration. Under matched experimental budgets, AutoScientists improves over prior AI agents across biomedical machine learning, language-model training optimization, and protein fitness prediction. On BioML-Bench, spanning biomedical imaging, protein engineering, single-cell omics, and drug discovery, AutoScientists achieves a mean leaderboard percentile of 74.4% across 24 tasks, improving over the strongest AI agent by +8.33%. On GPT training optimization, AutoScientists reaches a target validation bits-per-byte 1.9x faster than Autoresearch and continues discovering improvements from a starting champion where the single-agent approach finds none (7 vs. 0 accepted improvements). On ProteinGym fitness prediction, AutoScientists discovers a method for ACE2-Spike binding that improves over the current state-of-the-art model by +12.5% in Spearman correlation. Applied without modification across all 217 ProteinGym assays, the same method improves over the prior state of the art by +6.5% (Spearman correlation). 

---
# The Ethics of LLM Sandbox and Persona Dynamics 

**Authors**: Tim Gebbie, Stewart Gebbie  

**Link**: [PDF](https://arxiv.org/pdf/2605.28647)  

**Abstract**: It is well known that LLM guardrails and trained persona dynamics can produce a reality gap: the distance between the world a LLM is permitted or shaped to describe, and the world in which users must act. Here we argue that actively generating reality gaps is in fact unethical because it knowingly shifts epistemic risk back to the uninformed user -- this is reality laundering. This can potentially cause harm when operationalised at scale. The risk is sharpest in high-exposure advice contexts, where users seek orientation rather than a bounded, externally checkable task. Guardrails naively appear ethically necessary when they claim to prevent direct harm, but often become suspect when they suppress truthful perception and launder uncomfortable mechanisms into acceptable abstractions. Basel-style financial regulation, B-BBEE-style compliance, Societe Generale, and the London Whale show how formal safety systems can become legible, gameable, and performative while real exposure migrates elsewhere. The same pattern can appear in LLMs as moral compliance: safe language, distorted reality. We therefore distinguish refusing harm, from refusing reality; and then argue for top-down causal requirements specification at the task level rather than bottom-up moral correction at the response or sandbox level. Persona dynamics matter because the assistant interface is not neutral; it shapes how uncertainty, conflict, authority, and risk are staged. The conclusion is that so-called ``ethical AI'' becomes substantively unethical when it substitutes institutional reassurance for contact with reality. 

---
# Bandwidth-Efficient and Privacy-Preserving Edge-Cloud Many-to-Many Speech Translation 

**Authors**: Yexing Du, Kaiyuan Liu, Youcheng Pan, Bo Yang, Ming Liu, Bing Qin, Yang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28642)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated significant potential for speech-to-text translation (S2TT). However, existing deployment paradigms face critical challenges: pure on-device models suffer from resource constraints, while centralized cloud systems incur severe privacy risks and bandwidth bottlenecks by transmitting raw voice data. Furthermore, most models exhibit English-centric biases, restricting many-to-many translation scaling. In this paper, we propose Edge-cloud Speech Recognition and Translation (ESRT), a privacy-preserving and bandwidth-efficient collaborative edge-cloud MLLM framework. Specifically, we design an edge-cloud split inference architecture that retains a lightweight speech encoder and adapter on the device, transmitting only highly compressed intermediate features to the cloud. This fundamentally prevents voiceprint leakage and reduces bandwidth requirements by up to 10$\times$. To overcome English-centric bottlenecks, we introduce a multi-task weighted curriculum learning strategy with data balancing to ensure robust cross-lingual consistency. Extensive experiments on the FLEURS dataset demonstrate that our models, ESRT-4B and ESRT-12B, achieve state-of-the-art many-to-many S2TT performance across 45 languages ($45 \times 44$ directions). Code and models are released to facilitate reproducible, privacy-aware MLLM S2TT research. The code and models are released at this https URL. 

---
# LACUNA: Safe Agents as Recursive Program Holes 

**Authors**: Yaoyu Zhao, Yichen Xu, Oliver Bračevac, Cao Nguyen Pham, Frank Zhengqing Wu, Martin Odersky  

**Link**: [PDF](https://arxiv.org/pdf/2605.28617)  

**Abstract**: LLM agents increasingly act by writing code, yet a split persists between the runtime that drives the agent and the code the model writes. The runtime owns the loop, context, and control flow, and the model has little say over any of them. Letting model-written code shape the runtime itself would make agents more expressive, but it would also sharpen safety problems. A model can be diverted by a prompt injection, call the wrong tool, or fail partway and leave an inconsistent state, and each such failure reaches further when the code shapes the runtime than when it expresses a single action. We present LACUNA, a programming model for agents that closes this split while preserving safety. Each agent action is a typed call $\texttt{agent[T](task)}$ that the LLM fills with code when execution reaches it, and the code is type-checked against the surrounding program before it runs. Because each action is accepted or rejected as a whole, a rejected one leaves the environment untouched, and its compiler diagnostics drive a retry. The same check also bounds which tools and data an action may use and how they flow. Our primitive expresses ReAct loops, sub-agents, skills, parallel decomposition, and multi-model planning as ordinary control flow. We evaluate LACUNA on a collection of test cases, BrowseComp-Plus, and $\tau^2$-bench. On BrowseComp-Plus, $8.6\%$ of generations are rejected before execution, with 0.7 retries per query on average, and the agent reaches $27.1\%$ accuracy. On $\tau^2$-bench, LACUNA solves $76.0\%$ of $392$ tasks across four domains with a capable model, on par with the baseline agent. 

---
# Adaptive Multimodal Agents-Based Framework for Automatic Workflow Execution 

**Authors**: Susanna Cifani, Mario Luca Bernardi, Marta Cimitile  

**Link**: [PDF](https://arxiv.org/pdf/2605.28607)  

**Abstract**: Modern information systems require autonomous agents capable of navigating complex workflows, yet current methodologies often struggle with the transition from structured metadata parsing to general environmental perception. While the integration of MLLMs has enabled agents to interact directly with GUIs, existing approaches typically treat task sequences as discrete, linear episodes. This fragmentation prevents agents from capturing the underlying transition topology, limiting their effectiveness in novel or non-stationary scenarios. To address this, we propose a novel multimodal multi-agent framework that achieves automatic workflow execution through a distinct two-phase pipeline. First, during an offline discovery phase, the architecture adaptively constructs a topological knowledge base from fragmented execution logs. During inference, agents leverage Adaptive Retrieval-Augmented Generation (RAG) over this fixed, pre-established graph, coupled with a closed-loop collaborative verification protocol to dynamically self-correct and navigate. This graph-based approach facilitates superior task decomposition and adaptive navigation performance. We validate our framework in a real-world context, demonstrating its ability to maintain high reliability and semantic awareness even with limited training data. 

---
# Satisfiability Solving with LLMs: A Matched-Pair Evaluation of Reasoning Capability 

**Authors**: Leizhen Zhang, Shuhan Chen, Sheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.28602)  

**Abstract**: Large language models (LLMs) are increasingly used for tasks that implicitly reduce to Boolean satisfiability (SAT), yet their reasoning ability on SAT remains unclear. We present a systematic study of LLMs on 2-SAT and 3-SAT, together with two canonical reductions, Vertex Cover and discrete 3D packing, to probe representation-invariant reasoning. We first evaluate models using conventional metrics, including accuracy, precision, recall, and F1, as well as the SAT phase-transition setting. We find that these metrics can be misleading: many models obtain high scores by over-predicting satisfiable formulas, fail to reproduce the classical easy-hard-easy signature around the 3-SAT threshold, and degrade sharply as the number of variables grows.
To address this problem, we introduce a paired-formula protocol based on minimally different satisfiable and unsatisfiable instances, together with Accurate Differentiation Rate (ADR), which requires both members of each pair to be classified correctly. ADR separates reasoning-oriented models from heuristic ones and correlates with witness validity. Beyond CNF, we test cross-representation consistency by converting CNF to Vertex Cover and 3-SAT to discrete 3D packing. Model decisions on CNF and on the corresponding graph or packing instances agree for most models on more than 80 percent of instances, suggesting stable decision rules across representations. Overall, our results show that SAT is a conservative probe for LLM reasoning, and that paired evaluation with ADR provides a more faithful and representation-robust assessment than conventional metrics. 

---
# MUSE: Benchmarking Manufacturable, Functional, and Assemblable Text-to-CAD Generation 

**Authors**: Xiaoyu Dong, Zhi Li, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28579)  

**Abstract**: Large language models (LLMs) have recently advanced text-driven 3D generation, yet Text-to-CAD remains far from supporting industrial product design. Existing benchmarks focus primarily on generating single-part CAD models and evaluate them using geometric similarity metrics that fail to capture functionality, manufacturability, and assemblability. To address this gap, we introduce MUSE, a Text-to-CAD benchmark focused on complex, editable boundary representation (B-Rep) assemblies. MUSE pairs practical design instances with structured Design Specifications and evaluates generated models through a three-stage protocol: code check, geometric check, and design-intent alignment. The final stage uses design-specific rubrics to assess functionality, manufacturability, and assemblability, moving beyond shape matching toward practical design quality. To enable scalable evaluation, we use a rubric-based visual language model (VLM) judge and validate its reliability through human annotation. Experiments on closed-source and open-source LLMs reveal a clear failure cascade from executable code to valid geometry and finally to engineering-ready design, with even the strongest models achieving limited success on fine-grained engineering criteria. Together, MUSE provides a realistic benchmark and evaluation framework for advancing Text-to-CAD from geometric generation toward true engineering design. Our project website, including the leaderboard, dataset, and code, is available at this https URL. 

---
# Continual Model Routing in Evolving Model Hubs 

**Authors**: Jack Bell, Giacomo Carfì, Gerlando Gramaglia, Vincenzo Lomonaco  

**Link**: [PDF](https://arxiv.org/pdf/2605.28577)  

**Abstract**: AI model hubs provide access to a rapidly growing collection of powerful pre-trained models, enabling off-the-shelf mixture-of-experts systems with different routing strategies. However, this rapid growth poses two fundamental challenges: scaling model selection across thousands of experts and continually updating routing mechanisms as new models and tasks are introduced. In this paper, we formalise this setting as Continual Model Routing (CMR) and propose CMRBench, a new large-scale benchmark simulating realistic hub expansion and including over 2,000 candidate models. Finally, we introduce CARvE, a contrastive embedding approach for efficient continual model routing via checkpoint-based anchoring and structured replay. Extensive empirical results and ablations show that CARvE significantly outperforms zero-shot retrieval, fine-tuning, and adapter-merging baselines in model, family, and domain-level accuracy. 

---
# A Conflict-Aware Penalty and Statistical Loss Framework for Balancing Modalities and Enhancing Stability in Multimodal Sentiment Analysis 

**Authors**: Jianheng Dai, Jiazhang Liang, Sijie Mai  

**Link**: [PDF](https://arxiv.org/pdf/2605.28575)  

**Abstract**: Multimodal Sentiment Analysis (MSA) fuses text, acoustic, and visual streams to infer sentiment. Because pre-trained text encoders are far more expressive than their acoustic and visual counterparts, the text modality tends to dominate optimization, suppressing weaker modalities and inducing gradient norm conflicts that destabilize training. To address this, we propose a Conflict-aware Penalty (CP) that detects and penalizes gradient norm conflicts at each training step, and a Statistical Loss (SL) that aligns predicted distribution statistics with empirical input statistics. Crucially, CP prevents dominant modality gradients from interfering with the SL objective, enabling synergistic training within a unified framework incorporating adaptive modality encoding, gated cross-modal fusion, and unimodal auxiliary heads. Experiments on CMU-MOSI demonstrate state-of-the-art performance, with ablation studies confirming the effectiveness of each component. 

---
# Tree of Thoughts as a Classical Heuristic Search Problem: Formal Foundations and Design Patterns 

**Authors**: Guni Sharon  

**Link**: [PDF](https://arxiv.org/pdf/2605.28566)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable reasoning capabilities, yet their standard generation process -- auto-regressive token prediction -- is inherently myopic and prone to cascading errors. To address this, the Tree-of-Thoughts (ToT) framework creates a search space over intermediate reasoning steps, allowing search models to explore, look ahead, and backtrack. However, current ToT research remains fragmented across Natural Language Processing and Automated Planning communities, often using inconsistent terminology and ad-hoc implementations. Consequently, we synthesize the ToT landscape through a unified taxonomy based on classical heuristic search terminology. We map LLM-based reasoning to classical search components: state representation (granularity of thoughts), successor generation (prompting operators), and heuristic evaluation (self-assessment of progress). We analyze existing work within the context of our taxonomy and identify emerging design patterns: systematic search (Best-First Search) for shallow, deterministic tasks and lookahead-heavy strategies (DFS, MCTS) for deep multi-step reasoning. We conclude by identifying open algorithmic challenges at the intersection of heuristic search and LLM reasoning, and call on the heuristic search community to engage with this emerging domain. 

---
# A Matter of TASTE: Improving Coverage and Difficulty of Agent Benchmarks 

**Authors**: Tomer Keren, Nitay Calderon, Asaf Yehudai, Yotam Perlitz, Michal Shmueli-Scheuer, Roi Reichert  

**Link**: [PDF](https://arxiv.org/pdf/2605.28556)  

**Abstract**: As agent capabilities advance, existing benchmarks, such as $\tau^2$-Bench, are becoming increasingly saturated. Yet constructing new benchmark tasks remains complex, costly, and labor-intensive. Moreover, the standard approach, in which scenarios are first written in natural language and then mapped to tool sequences, captures only a narrow subset of the tool-use patterns agents exercise. In this paper, we address these problems by reversing the task construction process. We propose TASTE: Task Synthesis from Tool Sequence Evolution, an automatic method that generates challenging tasks with broader tool-use coverage. TASTE utilizes an Adaptive Contrastive $n$-gram model trained on LLM-judged validity signals. This enables sampling valid tool sequences that cover a vast range of tool combinations. TASTE then selects representative sequences from the pool via clustering, instantiates them into complete benchmark tasks, and refines them through iterative difficulty evolution. Using TASTE, we construct $\tau^c$-Bench, a challenging extension of the three domains of $\tau^2$-Bench. We evaluate $11$ agent/user LLM pairs and find that models nearly saturating $\tau^2$-Bench suffer severe performance drops on our tasks (e.g., Gemini-3-Flash falls from $0.82\!-\!0.94$ to $0.28\!-\!0.61$). Beyond increasing difficulty, our generated tasks more than double the number of unique tool combinations agents must execute. Our results suggest high scores on existing benchmarks often reflect saturation rather than robust task-solving ability. By automating the generation of difficult, high-coverage benchmarks, TASTE enables continuous, scalable evaluation of future agents. 

---
# Refusal Before Decoding: Detecting and Exploiting Refusal Signals in Intermediate LLM Activations 

**Authors**: Matteo Gioele Collu, Riccardo Conte, Alberto Giaretta, Denis Kleyko, Mauro Conti, Matteo Zavatteri, Roberto Confalonieri  

**Link**: [PDF](https://arxiv.org/pdf/2605.28553)  

**Abstract**: In this paper, we investigate whether refusal behavior can be predicted from LLM intermediate activations before decoding using linear probes trained on residual stream activations at each transformer block. We find that refusal is linearly decodable well before the final layer, indicating that safety-relevant behavior is represented in intermediate activations before output generation. To test whether this signal is actionable, we introduce Mechanistic AutoDAN, a probe-guided variant of AutoDAN that replaces full-model fitness evaluation with partial forward passes and probe-based scoring inside a genetic prompt search loop. Across the evaluated models, our method achieves attack success rates competitive with vanilla AutoDAN while reducing per-iteration search time by up to 72%, and probe-guided prompts match or exceed AutoDAN's cross-model transfer in several configurations. We further find that the usefulness of probe guidance increases with model scale. Our results show that refusal is not only observable at the output level, but is encoded as a structured and actionable signal in intermediate LLM activations. 

---
# Modeling Vehicle-Type-Specific Pedestrian Crash Avoidance Behavior in Safety-Critical Interactions Using Smooth-Mamba Deep Reinforcement Learning 

**Authors**: Qingwen Pu, Kun Xie, Hong Yang, Di Yang, Junqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28552)  

**Abstract**: As automated vehicles (AVs) increasingly share roadways with human-driven vehicles (HDVs), understanding how pedestrians respond to different vehicle types in safety-critical interactions is essential for the safe deployment of automated driving technologies. This study extracts safety-critical pedestrian-vehicle interactions from the Argoverse 2 dataset to capture real-world crash avoidance behaviors in encounters involving AVs and HDVs. To model vehicle-type-specific pedestrian crash avoidance behavior, we develop a Smooth-Mamba Deep Deterministic Policy Gradient framework, termed SMamba-DDPG, which integrates smooth action constraints with efficient temporal representation learning. To quantify pedestrian behavioral differences, the framework trains separate crash avoidance policies for pedestrian interactions with AVs and HDVs. Results show that SMamba-DDPG outperforms baseline reinforcement learning and supervised learning models in reproducing pedestrian crash avoidance behaviors. Reconstructed trajectories demonstrate strong behavioral realism, accurately reproducing crash avoidance kinematics in both AV and HDV scenarios. Reaction time analysis shows that the model captures human-like response delays and reveals that pedestrians respond more quickly to AVs than to HDVs. Counterfactual analysis further indicates that pedestrians adopt lower crossing speeds when interacting with AVs. Large-scale safety analysis of model-generated data revealed that pedestrian-AV interactions consistently yielded lower conflict rates and higher pedestrian yielding rates compared to pedestrian-HDV interactions. The findings highlight the importance of incorporating vehicle-type-specific pedestrian behavioral models for safer automated driving system design and more realistic traffic simulations in mixed-traffic environments. 

---
# Cultural Binding Heads in Language Models 

**Authors**: Avrile Floro, Luca Benedetto  

**Link**: [PDF](https://arxiv.org/pdf/2605.28543)  

**Abstract**: LLMs often default to equal treatment across cultural groups, even though context warrants differentiation: this is a lack of difference awareness. Using mechanistic interpretability and a factorial design on the N4 cultural appropriation benchmark from Wang et al. (2025), we identify 2-3 mid-layer attention heads per model that contribute causally to cultural binding across eight models (four architectures, base and instruct). Cultural binding is the process of associating cultural items with the appropriate identity. Knockout of the identity-to-item edges on these heads lowers the binding strength by 9-23%. The identified heads transfer from instruct to base models, suggesting that cultural binding is created at pre-training. An $\alpha$-scaling shows a graded dose-response and moderate amplification steering at generation ($\alpha = 2-3$) increases cultural differentiation accuracy by 1-3 pp while leaving neutral reasoning mostly intact. A knowledge probing task shows that models know 3-5 times more than they act upon it, indicating that the bottleneck lies in routing and not knowledge. 

---
# Do Agents Know What They Can't Do? Evaluating Feasibility Awareness in Tool-Using Agents 

**Authors**: Liang Cheng, Mingsheng Cai, Jiuming Jiang, Luo Mai  

**Link**: [PDF](https://arxiv.org/pdf/2605.28532)  

**Abstract**: Tool-using agents often incur substantial computational cost due to long reasoning chains and iterative tool usage. In practical scenarios, many tasks become infeasible under constrained tool environments, where the capabilities required for successful task completion are unavailable. Detecting infeasible tasks and stopping execution early can significantly reduce unnecessary execution cost. In this work, we propose FeasiGen, an automatic pipeline for constructing infeasible agent tasks by identifying the critical tools required for successful task completion. Our approach extracts tool-calling traces from successful executions across multiple agent systems, identifies critical tools consistently shared across diverse execution strategies, and masks these tools to automatically transform solvable tasks into infeasible ones. Human verification confirms that the infeasibility annotations for our constructed tasks achieve over 94% accuracy. We further introduce feasibility-aware evaluation metrics for measuring whether agents can recognize infeasible tasks and stop execution appropriately. Extensive evaluations across nine models reveal substantially weak infeasibility detection ability, with false continue rate reaching up to 73.9%. We further observe that multi-agent architectures significantly reduce erroneous execution under infeasible conditions. 

---
# Entropy-aware Masking for Masked Language Modeling 

**Authors**: Gokul Srinivasagan, Kai Hartung, Munir Georges  

**Link**: [PDF](https://arxiv.org/pdf/2605.28526)  

**Abstract**: Masked language modeling has become a standard pretraining objective for training encoder-based language models. In this approach, certain tokens in the input are masked, and the model learns to predict them using the surrounding context. This process enables the model to capture both syntactic and semantic properties of language. Conventionally, the tokens selected for masking are chosen at random, which may not always yield the most effective learning signals. In this work, we examine a token masking strategy based on entropy distribution. We use the model's entropy over token predictions to identify which tokens should be masked. This method aims to target tokens that are more informative and uncertain to improve the training efficacy. We also propose a novel self-masking approach that enhances training efficiency without relying on an external reference model. Experimental results demonstrate that our method achieves an average performance improvement of 5% in GLUE scores compared to the baseline. Further, we experiment with combining knowledge distillation with entropy masking, resulting in the best overall results. 

---
# Let Relations Speak: An End-to-End LLM-GNN Soft Prompt Framework for Fraud Detection 

**Authors**: Zhixing Zuo, Huilin He, Jiasheng Wu, Dawei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.28524)  

**Abstract**: In recent years, Large Language Models (LLMs) have shown great capability in processing graph tasks such as fraud detection. However, most existing methods rely heavily on rich text attributes, which poses difficulties for this domain due to the lack of textual data. Although some pioneering methods attempt to overcome it, their textualization of graph structures via hard prompts easily leads to feature distortion. Additionally, fraud detection often exhibits multi-relational complexity, where current methods struggle to capture this deep semantic information. To address these challenges, we propose LLM-GNN Soft Prompt Framework (LGSPF). Specifically, LGSPF bridges the graph structure and semantic space using soft prompt to eliminate reliance on text. We further introduce a parallel Graph Neural Network (GNN) encoder to translate multi-relational topologies into graph tokens for fine-grained LLM fraud comprehension. Through end-to-end optimization, LGSPF enhances deep semantic alignment between LLM and GNN. Experiments across diverse fraud detection benchmarks demonstrate our method achieves state-of-the-art performance. Moreover, we further validate the contribution of LGSPF on enhancing the semantic interpretability of fraud behaviors. 

---
# GS-FUSE: Granger-Supervised Gated Fusion and Multi-Granularity Alignment for Event-Driven Financial Forecasting 

**Authors**: Yang Zhang, En Chun, Ziyun Mao, Yulu Wu, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28520)  

**Abstract**: Accurately forecasting the impact of salient financial events on markets is critical for investors and policymakers. However, existing multimodal time-series models typically fuse text and prices symmetrically, without an explicit way to decide when event text is truly predictive, and thus struggle to exploit the directional event-to-price structure and the heterogeneous roles of textual and price signals. In this work, we propose GS-Fuse, a multimodal event-based forecasting framework that employs (i) a Granger-supervised, causal-aware gated fusion module, which learns to open toward event text only when it provides incremental predictive value beyond historical prices, and (ii) a multi-granularity alignment mechanism that jointly aligns high-level event representations and fine-grained textual cues with future market trajectories. Built as a flexible, plug-and-play adapter on top of off-the-shelf large language models and time-series foundation models, GS-Fuse can be instantiated across diverse backbones and market settings. Extensive experiments on real-world financial datasets show that GS-Fuse consistently outperforms state-of-the-art time-series and multimodal baselines across multiple assets and forecasting horizons. 

---
# Benchmarking AI for low-resource contexts: Thinking beyond leaderboards 

**Authors**: Aakash Pant, Kavya Shah, Apoorv Agnihotri, Sneha Nikam, Prasaanth Balraj, Nakul Jain  

**Link**: [PDF](https://arxiv.org/pdf/2605.28508)  

**Abstract**: Existing AI evaluation practices often fail to capture how systems actually perform in low-resource environments, where operational constraints shape usability as much as model quality. Through a structured analysis of existing benchmark families across speech, chat/RAG, and vision systems, we identify critical gaps between laboratory evaluation practices and real-world deployment conditions in low-resource environments. We argue that the meaningful unit of assessment is the deployed system rather than an isolated model and that effective evaluation frameworks must integrate task performance with deployment conditions such as noisy inputs, code-switching, intermittent connectivity, low-end hardware, and domain shift. At the same time, benchmarks should recognize that different application classes require distinct evaluation profiles rather than a single aggregate score that obscures operational differences. To support practical decision-making, we propose a shared reporting framework that preserves comparability across systems and application types while remaining sensitive to deployment context. Finally, we emphasize the need for concise and actionable reporting artifacts for policymakers, donors, and implementers, including standardized one-page benchmark cards, deployment profiles, and explicit documentation of failure handling procedures and human oversight mechanisms. 

---
# ProvMind: Provenance-grounded reasoning for materials synthesis 

**Authors**: Yiming Zhang, Ryo Tamura, Koji Tsuda  

**Link**: [PDF](https://arxiv.org/pdf/2605.28487)  

**Abstract**: Materials process optimization requires reasoning over routes, conditions, tools and causal dependencies, yet most computational formulations flatten synthesis procedures into text or ordered steps. We introduce MatProcBench, a provenance-grounded benchmark constructed from literature-mined MatPROV graphs, to evaluate seven process-reasoning tasks spanning route continuity, step-level variable inference and global causal consistency under both same-split and shift-aware evaluation, including a strict dual-OOD split that combines temporal and material-class shift. We further introduce ProvMind, a process-memory reasoning framework that retrieves analogous training processes, converts them into provenance-aware option-level compatibility scores, and uses a language model for constrained final decision making. ProvMind achieves 52.84\% accuracy on the dual-OOD split, outperforming prompting, retrieval-augmented and supervised fine-tuning baselines. 

---
# From Learning Resources to Competencies: LLM-Based Tagging with Evidence and Graph Constraints 

**Authors**: Ngoc Luyen Le, Marie-Hélène Abel, Bertrand Laforge  

**Link**: [PDF](https://arxiv.org/pdf/2605.28483)  

**Abstract**: Linking learning resources to a structured competency framework is key to enabling competency-based search and curriculum analytics in Learning Management Systems (LMS). However, manual tagging is labor-intensive, and fully automatic methods often lack transparency. In this paper, we present an end-to-end alignment pipeline that uses a large language model (LLM) as a constrained, evidence-producing tagger. LMS resources -both instructional content and assessments -are first segmented into meaningful pedagogical fragments. For each fragment, a small set of candidate competencies is retrieved from structured competency profiles enriched with graph-based context. The LLM then selects the most relevant competencies from this set and provides supporting evidence spans from the fragment text. These predictions are refined using the structure of the competency graph and aggregated at the resource level. We evaluate our approach on a dataset built from the Computer Science department's competency referential at the Université de Technologie de Compiègne (UTC), covering 22 competencies across multiple course materials. Our LLM+BM25+Graph (LBG) pipeline achieves strong results, with a micro-F1 of 0.57 and macro-F1 of 0.50 at the fragment level, 0.51 macro-F1 at the resource level, and an MRR of 0.82outperforming zero-shot and few-shot LLM variants, retrieval/similarity baselines, and supervised classifiers -while also producing more mechanically traceable evidence spans to support human auditing and educational analysis. 

---
# Diffusion Large Language Models for Visual Speech Recognition 

**Authors**: Jeong Hun Yeo, Chae Won Kim, Hyeongseop Rha, Yong Man Ro  

**Link**: [PDF](https://arxiv.org/pdf/2605.28456)  

**Abstract**: Existing Visual Speech Recognition (VSR) systems commonly rely on left-to-right autoregressive decoding, which can force premature decisions on visually ambiguous tokens before sufficient context is available. We propose DLLM-VSR, to the best of our knowledge, the first Diffusion Large Language Model (DLLM)-based VSR framework, formulating transcription as iterative masked denoising with flexible-order decoding. With confidence-based unmasking, DLLM-VSR commits high-confidence positions early and uses the committed tokens as bidirectional context to refine ambiguous ones. To adapt DLLMs to VSR, we introduce a two-stage masked-denoising training strategy that separates visual-to-text content alignment from length modeling. We further observe a performance gap with oracle-length decoding, which assumes access to the true transcript length, indicating that reducing target-length uncertainty can improve DLLM-based VSR. To reduce this gap, we develop length-guided candidate decoding, which uses video duration to construct plausible transcript-length hypotheses, decodes under multiple hypotheses, and reranks candidates using length plausibility and decoding confidence. The proposed method achieves a state-of-the-art WER of 19.5\% on LRS3 using only its labeled training data. 

---
# GONDOR to the Rescue: Satisficing Planning with Low Memory 

**Authors**: Yonatan Vernik, Alexander Tuisov, Alexander Shleyfman  

**Link**: [PDF](https://arxiv.org/pdf/2605.28454)  

**Abstract**: Greedy Best-First Search (GBFS) is the dominant approach for solving search problems where the goal can be estimated with a heuristic, such as planning, route finding, navigation, and pathfinding. This is especially true when the memory is tightly constrained, such as planning on edge devices. To alleviate that, we present GONDOR (Greedy Online Navigation with Dynamic Outpost-based Re-search), a memory-efficient extension of GBFS that allows search to continue under strict memory limits by periodically compressing the search tree while retaining a sparse set of anchor states, then upon reaching the goal reconstructs the path by re-searching between the sparse states. We analyze the algorithm and discuss several variants defined by different outpost selection policies. In addition, we explore using Bloom filters for compact duplicate detection in the closed list. Experiments across numeric planning domains and heuristic configurations show that GONDOR consistently improves coverage under low memory budgets compared to standard GBFS. We release the implementation of GONDOR and the Bloom-filter variant to facilitate further research on memory-efficient heuristic search. 

---
# DenoiseRL: Bootstrapping Reasoning Models to Recover from Noisy Prefixes 

**Authors**: Caijun Xu, Changyi Xiao, Zhongyuan Peng, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.28421)  

**Abstract**: Reinforcement learning has become a central paradigm for advancing reasoning in large language models, yet most existing methods still depend on stronger teacher models or heavily curated difficult datasets, limiting scalable capability improvement. In this paper, we introduce DenoiseRL, a reinforcement learning framework that substitutes external supervision with recovery-oriented optimization over failures from weak models. Instead of relying on stronger supervision or carefully engineered data, DenoiseRL learns directly from incorrect reasoning traces by converting them into opportunities for improvement, making training more scalable and less dependent on external resources. This yields a richer and more diverse learning signal, improving exploration efficiency from imperfect model behavior. As a result, DenoiseRL improves reasoning performance and overall training efficiency while reducing the need for expensive data curation or stronger teacher models. Empirically, DenoiseRL consistently outperforms strong on-policy RL baselines across competitive mathematical and general reasoning benchmarks and promotes stronger self-corrective behavior as training difficulty increases, highlighting an effective and scalable alternative pathway for improving reasoning in large language models. 

---
# Efficient Post-training of LLMs for Code Generation With Offline Reinforcement Learning 

**Authors**: Mingze Wu, Abhinav Anand, Shweta Verma, Mira Mezini  

**Link**: [PDF](https://arxiv.org/pdf/2605.28409)  

**Abstract**: Post-training using online reinforcement learning (RL) is an important training step for LLMs, including code-generating models. However, online RL for code generation involves LLM inference and verification of the generated output, which can take considerable time and resources. In this paper, we explore the application of offline RL to code-generating models by leveraging existing code datasets. Our experiments demonstrate that offline RL is an effective training strategy for improving LLM performance. We show that offline RL can be especially beneficial for small LLMs and challenging coding problems. 

---
# Measuring Progress Toward AGI: A Cognitive Framework 

**Authors**: Ryan Burnell, Yumeya Yamamori, Orhan Firat, Kate Olszewska, Steph Hughes-Fitt, Oran Kelly, Isaac R. Galatzer-Levy, Meredith Ringel Morris, Allan Dafoe, Alison M. Snyder, Noah D. Goodman, Matthew Botvinick, Shane Legg  

**Link**: [PDF](https://arxiv.org/pdf/2605.28405)  

**Abstract**: Despite widespread discussion of AGI, there is no clear framework for measuring progress toward it. This ambiguity fuels subjective claims, makes it difficult to track progress, and risks hindering responsible governance. As a starting point to address this gap, we present a framework for understanding system capabilities in relation to human cognitive abilities. Drawing from decades of research in psychology, neuroscience, and cognitive science, we introduce a Cognitive Taxonomy that deconstructs general intelligence into 10 key cognitive faculties. We then propose a rigorous evaluation protocol in which a system's performance is measured across a suite of targeted, held-out cognitive tasks, generating a 'cognitive profile' that can be used to understand a system's strengths and weaknesses. We hope this framework will provide a practical roadmap and an initial step toward more rigorous, empirical evaluation of AGI. 

---
# HRBench: Benchmarking and Understanding Thinking-Mode Switch Strategies in Hybrid-Reasoning LLMs 

**Authors**: Yansong Ning, Mianpeng Liu, Jingwen Ye, Weidong Zhang, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28398)  

**Abstract**: Hybrid-reasoning large language models (LLMs) expose explicit controls over reasoning effort, allowing users or systems to trade off answer quality against inference cost. However, existing methods for adaptive thinking-mode selection are typically evaluated under different models, datasets, and implementation assumptions, making it difficult to compare their practical behavior. We introduce HRBench, a unified evaluation framework for studying thinking-mode switching in hybrid-reasoning LLMs. HRBench organizes the design space along two axes: three switching strategy families, prompt-based selection, external routing, and speculative execution, and four training regimes, training-free, SFT, offline and online RL, yielding 12 controlled evaluation settings. We evaluate these settings across 6 LLMs, from Qwen3.5-2B to Kimi-K2.5-1.1T, and 5 reasoning benchmarks covering mathematics, science, and code, while reimplementing 12+ representative prior methods within the same pipeline. Our analysis characterizes how different switching strategies occupy distinct effectiveness-efficiency trade-off regions: prompt-based methods often provide favorable token-accuracy trade-offs, routing methods offer more stable cost reduction, and speculative methods tend to improve accuracy at higher token cost. We further find that training affects strategies differently, and that the preferred strategy varies with model scale and task domain. HRBench provides reference implementations and a unified evaluation platform to support more controlled research on efficient reasoning in hybrid-reasoning LLMs. Our data, code and repository are available at this https URL. 

---
# You Live More Than Once: Towards Hierarchical Skill Meta-Evolving 

**Authors**: Xujun Li, Kehan Zheng, Mingyuan Zhao, Yize Geng, Jinfeng Zhou, Qi Zhu, Fei Mi, Lifeng Shang, Minlie Huang, Hongning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28390)  

**Abstract**: Test-time skill evolving is regarded as a new paradigm for enhancing deployed agentic systems. Existing works mainly focus on hard-coded skill evolving strategies or parametric learning that rely on expensive parameter updates in the underlying LLMs. In this paper, we demonstrate that test-time refinement of the skill evolving framework itself is necessary for continuous improvement of the agent systems in different downstream scenarios, and lightweight algorithmic adaptation is feasible. Specifically, we propose HiSME, a lightweight hierarchical skill meta-evolving solution that jointly optimizes skills and the skill evolving strategy by learning meta-skills from agents' task execution traces. Experiments on diverse agentic benchmarks show that meta-evolving can produce a higher-quality skill library than pure skill evolving and can derive diverse meta-skills for different scenarios, thereby facilitating future continual experience learning. Our code is temporarily public at this https URL. 

---
# Mechanistically Interpreting the Role of Sample Difficulty in RLVR for LLMs 

**Authors**: Yue Cheng, Jiajun Zhang, Xiaohui Gao, Weiwei Xing, Zheng Wang, Zhanxing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28388)  

**Abstract**: Reinforcement Learning with Verifiable Reward (RLVR) is empirically shown to notably enhance the reasoning performance of large language models (LLMs), particularly in mathematics and programming. However, the mechanistic role of Sample Difficulty in RLVR remains poorly understood. In this paper, we investigate RLVR through the lens of difficulty-wise and one-sample analysis. We find that sample difficulty has a non-monotonic effect on RLVR: easy and medium-difficulty problems yield the strongest and most stable reasoning improvements, whereas overly hard problems often provide weak learning signals, induce degenerate behaviors such as answer repetition or skipping necessary computation, and can ultimately degrade the model's pre-existing capabilities. Beyond the obverse of response, we further analyze the model's internal feature dynamics using Temporal Sparse Autoencoders (T-SAE). Easy problems mainly reinforce direct-answer and basic-computation features while suppressing deliberative-reasoning features; hard problems activate reasoning-related features but become useful only when successful trajectories are sampled; medium-difficulty problems provide a more balanced signal, strengthening both computation and multi-step reasoning features. Motivated by these findings, we propose difficulty-adaptive strategies for hard-sample utilization, using backward-reasoning reformulation and T-SAE-guided training signals to improve reward density and credit assignment during RLVR. Overall, our results identify sample difficulty as a key factor governing both the optimization dynamics and representation evolution of RLVR. 

---
# From paper to benchmark: agentic, framework-based reproduction of under-specified methods in machine health intelligence 

**Authors**: Raffael Theiler, Ludovico Comito, David Leko, Leandro Von Krannichfeldt, Lev Telyatnikov, Olga Fink  

**Link**: [PDF](https://arxiv.org/pdf/2605.28371)  

**Abstract**: Industrial Prognostics and Health Management (PHM) provides a representative case study for a broader challenge in applied machine learning: translating published papers into executable, benchmark-ready implementations. Reproducing under-specified methods in PHM is particularly difficult due to restricted access to industrial datasets, incomplete reporting of preprocessing and evaluation protocols, and implicit design choices (e.g., windowing, target construction, data splits) that critically affect performance. Existing paper-to-code systems generate implementations for individual papers, but these artifacts are often not directly comparable due to inconsistencies in assumptions and evaluation settings. We introduce \emph{agentic, framework-based PHM paper reproduction}, where an agent translates a paper into a shared PHM benchmark framework via a \emph{slot-binding interface}. This interface maps equations and protocol descriptions into structured components (task definitions, dataset adapters, windowing, targets, models, and evaluators), while explicitly recording unresolved assumptions. The resulting implementations are validated against standardized task contracts and evaluation hooks, enabling consistent and comparable benchmarking. We evaluate this approach on 16 PHM papers, comparing framework-enhanced, skill-based and prompt-based agentic reproduction against a recent framework-free paper-reproduction agent. We assess reproduction success, model-based code evaluation, framework binding of paper assumptions, and cross-paper benchmark comparability under standardized protocols. Our results show that coupling agentic generation with a shared framework transforms paper reproduction from isolated code synthesis into executable, assumption-aware, and systematically comparable benchmark implementations. 

---
# CyberJurors: A Multi-Agent Simulation Task for E-Commerce Disputes Verdict 

**Authors**: Yanhui Sun, Wu Liu, Haifeng Ming, Xinru Wang, Hantao Yao, Yongdong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28369)  

**Abstract**: E-commerce platforms have begun recruiting crowdsourced jurors to adjudicate massive volumes of transaction disputes. Unlike formal legal judgment, E-commerce dispute verdicts require grounding pivotal clues from redundant, multi-round, multimodal evidence and making decisions under flexible platform-specific conventions. These characteristics render existing methods insufficient for this scenario. To bridge this gap, we introduce a pioneering task, E-commerce Dispute Verdicts (EDV), and present VerdictBench, a multimodal benchmark comprising 6,000 real-world cases designed to reflect crowdsourced jury decisions. Building upon this, we propose CyberJurors, a multi-agent framework to clarify the dispute logic and regulate the verdict process. At the individual level, Individual Verdict Chain-of-Thought decomposes the EDV task into four structured reasoning stages, enabling fine-grained clue perception and clarifying causal logic between pivotal clues and the dispute focus. At the collective level, Jury Consensus Verdict simulates multi-round discussion and voting among jurors, while incorporating verdict precedents to mitigate cognitive biases toward either disputant. Experiments on VerdictBench show that CyberJurors outperforms state-of-the-art LLMs, MLLMs, and court simulators, while achieving stronger alignment with real-world jury voting patterns. Code and dataset are available at this https URL and this https URL. 

---
# Risk-Controlled Lean-as-Judge for Natural-Language Mathematical Reasoning 

**Authors**: Pauline Bourigault, Xiaotong Ji, Matthieu Zimmer, Rasul Tutunov, Haitham Bou Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2605.28365)  

**Abstract**: Lean is increasingly used to judge natural-language mathematical answers, but its signal is partial: many answers never formalize, and a failed proof may reflect an ill-typed statement or a missing library fact, not a wrong answer. On MATH-500 we show this signal is (i) sharply coverage-dependent, that is the proof-winning answer is correct 96% of the time at high proved coverage but 20% at low, and (ii) sparse and often unfaithful: a 7B autoformalizer proves a class for only 28% of problems, and a manual audit finds only approximately 43% of those proofs faithful. We propose COVCAL, a selector over Lean-trace diagnostics that certifies a finite-sample selective-risk bound on accepted answers or abstains, under two regimes (a conservative Bonferroni bound and a tighter dev-then-cal rule). Feasibility depends on autoformalization coverage: with the 7B formalizer the signal is too sparse and Bonferroni abstains on all 20 bootstrap partitions, whereas a prover-specialized formalizer reaches 79% coverage and flips it to feasible on 17 of 20, accepting approximately 48% of problems at 0.98 accepted accuracy. Since self-consistency alone is already 91% accurate, our contribution is a precise account of when, and with which formalizer, a partial formal signal can be trusted under risk control. 

---
# Prompt Codebooks: Discrete Compositional Optimization for Language Model Instruction Refinement 

**Authors**: Jyotirmoy Nath, Neeraj Kumar, Brejesh Lall  

**Link**: [PDF](https://arxiv.org/pdf/2605.28360)  

**Abstract**: Automatic prompt optimization (APO) has driven significant gains in LLM-based agentic workflows. However, existing methods treat each task's prompt as a monolithic, instance-blind string optimized through global edits, producing brittle updates and preventing the reuse of learned sub-behaviors. We propose Prompt Codebooks (PCO), a novel compositional prompt optimization framework that recasts APO as discrete learning over a finite vocabulary of natural-language instincts - atomic, reusable instruction units. PCO organizes prompt-construction knowledge in a discrete codebook and routes each input to a small subset of entries via an LLM-based encoder; a generator composes them into a prompt for the frozen target model; a critic emits a structured verdict that decomposes by attribution into per-variable textual gradients, jointly training the encoder, generator, and codebook under a language-valued min-max objective. The resulting routing is per-instance: different inputs in the same task receive different instinct compositions, a regime structurally inexpressible under instance-blind methods. Across six benchmarks on Qwen3-8B and LLaMA-3.1-8B, PCO improves over zero-shot by up to +30.36 points, surpasses the strongest prior baseline (GEPA) by +3.34 on HotpotQA and +1.11 in aggregate, and reduces deployed prompt length by up to 14.1x versus MIPROv2 and 3.0x versus GEPA using only K=16 instincts. 

---
# From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents on Stock Markets 

**Authors**: Taojie Zhu, Wentao Zhao, Rui Sun, Beidi Luan, Jiacheng Lu, Sinuo Wang, Jing Li, Daxin Jiang, Yonghong He, Zuo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2605.28359)  

**Abstract**: Evaluating whether large language model (LLM) agents can profit in capital markets is increasingly framed as end-to-end trading: place an agent in a historical market, let it trade, and measure portfolio returns. This setup is vulnerable to two evaluation failures. First, long backtests often overlap with the knowledge cutoffs of frontier LLMs, allowing memorized tickers, dates, prices, and market narratives to substitute for investment reasoning. Second, raw returns are a noisy proxy for stock-selection ability, since positive performance may come from market beta, style exposure, or favorable regimes rather than genuine alpha.
We introduce KTD-Fin (Knowing-To-Doing Financial Benchmark), an end-to-end stock-market trading benchmark that addresses both issues. KTD-Fin uses a data-side masking protocol to anonymize key identifiers and calendar information consistently across prompts and tools, separating historical market memory from investment decision-making. It also incorporates a Barra-style performance attribution framework that decomposes portfolio returns into market, style, and stock-selection alpha components.
Across ten frontier LLM agents evaluated on the Chinese CSI300 over a 2024--2026 window, masking substantially changes agent rationales, pushing them towards anonymized factor-based reasoning. Attribution analysis further shows that LLM agents' cumulative returns under leakage-controlled evaluation are largely explained by passive market and style exposure, with limited evidence of persistent stock-selection alpha. These findings suggest that financial LLM benchmarks should evaluate not only whether an agent makes money, but also whether the source of returns reflects transferable investment skill. We release KTD-Fin as a reproducible template for leakage-controlled and attribution-aware evaluation of LLM trading agents. 

---
# Plan Before Search: Search Agents Need Plan 

**Authors**: Zhipeng Qian, Zihan Liang, Yufei Ma, Ben Chen, Huangyu Dai, Jiayi Ji, Chenyi Lei, Wenwu Ou, Xiaoshuai Sun, Qibin Hou  

**Link**: [PDF](https://arxiv.org/pdf/2605.28354)  

**Abstract**: Training large language models as retrieval-augmented reasoning agents typically combines reinforcement learning with an SFT cold start distilled from a stronger model. However, this paradigm overlooks two fundamental factors: the dependency structure among sub-skills, and the possibility that distillation is not the only route to capability acquisition. We study this through Plan, a structured agentic behavior for multi-hop retrieval that decomposes a question into ordered sub-questions before any retrieval is performed, so that each search step can be anchored to a pre-designed sub-question instead of drifting under the influence of partially relevant documents retrieved earlier. However, across three model families spanning 3B to 14B parameters, we find that an identical reward signal induces qualitatively different RL failure modes. This phenomenon indicates that successful training hinges not only on reward design but also on model-specific feasibility conditions: sufficient initial entropy, training stability, and prerequisite sub-skills. Motivated by this, we propose a self-bootstrapping paradigm in which a small-scale seed model generates filtered trajectories that activate Plan in any target model, eliminating the need for distillation from an external stronger model. Our pipeline activates Plan across every tested model and consistently outperforms competitive baselines on multi-hop QA benchmarks. 

---
# FedMPT: Federated Multi-label Prompt Tuning of Vision-Language Models 

**Authors**: Xucong Wang, Pengkun Wang, Zhe Zhao, Liheng Yu, Shuang Wang, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28347)  

**Abstract**: Multi-Label Recognition (MLR) based on Vision-Language Models (VLMs) aims to leverage their pre-trained knowledge to better adapt complex recognition scenarios, thereby enhancing model robustness. However, for realistic decentralized applications requiring federated learning, adapting VLMs to each client that possesses private and heterogeneous data can cause the model to overfit spurious label correlations, consequently triggering irrelevant categories when encountering new samples. To tackle this problem, we reconsider the federated learning for MLR with a causal model, in which we adopt a front-door adjustment and decouple the MLR modeling process by intermediate variables that magnify the oracle label co-occurrence. Guided by our analysis, we propose our FedMPT, the first method specifically designed for federated MLR. The core idea of FedMPT is to leverage generalizable conditions to steer federated MLR to mitigate erroneous label activations. To achieve this, FedMPT introduces an Large Language Model (LLM)-driven pipeline to decipher the underlying conditions that govern label dependencies. Furthermore, we introduce an optimal transport between the condition-enriched prompts and the image patches to uncover multiple region-level semantics. Finally, we generate synergistic predictions from different conditions with a crafted gating mechanism. Experiments on multiple benchmark datasets show that our proposed approach achieves competitive results and outperforms SOTA methods under varied settings. 

---
# Picid: A Modular Evaluation Infrastructure for Reproducible PHM Across Tasks and Domains 

**Authors**: Lev Telyatnikov, Raffael Theiler, Leandro Von Krannichfeldt, Olga Fink  

**Link**: [PDF](https://arxiv.org/pdf/2605.28345)  

**Abstract**: Progress in Prognostics and Health Management (PHM) is hindered by the lack of standardized and reusable evaluation practices across tasks, datasets, and application domains. Reported results are often difficult to reproduce and compare, as key protocol choices, such as data splits, preprocessing, label alignment, temporal windowing, and metrics, are often implicit or implemented ad hoc. We introduce \picid, a modular evaluation infrastructure that formalizes the PHM evaluation pipeline as an explicit, executable, and reproducible protocol. Through well-defined abstractions, \picid enforces deterministic, leakage-safe dataset construction while remaining flexible across diverse PHM settings. The framework supports fault detection, diagnostics, and prognostics through a unified interface and can be extended to new datasets and model classes without violating protocol invariants. By standardizing data contracts and evaluation boundaries, \picid also enables fair cross-task comparisons across diagnostics (classification) and prognostics (regression), allowing identical model families to be evaluated consistently across heterogeneous settings. We demonstrate \picid through an empirical evaluation of thirteen models on twelve datasets spanning batteries, bearings, turbofan engines, hydraulics, filtration systems, and buildings. This work establishes a reusable foundation for standardized, fair and reproducible evaluation in PHM. 

---
# SafeMed-R1: Clinician-Audited Safety and Ethics Alignment for Medical Large Language Models 

**Authors**: Chao Ding, Mouxiao Bian, Tianbin Li, Minjia Yuan, Yidong Jiang, Yankai Jiang, Jinru Ding, Jiayuan Chen, Zhuangzhi Gao, Pengcheng Chen, Zhao He, Rongzhao Zhang, Meiling Liu, Luyi Jiang, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28338)  

**Abstract**: Large language models(LLMs) increasingly match expert performance on licensing examinations, yet routine clinical use remains limited because governance requires auditable reasoning, safety and ethics alignment, and resilience to adversarial misuse. Here we present SafeMed-R1, trained with a traceable Clinical Trust Signals(CTS) pipeline that links each reasoning instance to clinician rubric scores and edit histories, and aligned through safety and ethics supervision and red team stress testing. SafeMed-R1 attains a macro-averaged accuracy of 79.6% across clinical benchmarks. Under adversarial safety testing, it shows the lowest aggregated risk and reduces unsafe outputs by about 3 to 5% relative to its baseline. In a paired expert study of 30 medication safety vignettes, SafeMed-R1 matches PGY1 and PGY2 residents on medical correctness and scores higher for medication safety, guideline consistency, and clinical usefulness. Collectively, these results suggest that clinician-audited supervision provenance, together with domain-tailored safety and ethics alignment, can strengthen governance-relevant evidence without relying on inference-time retrieval or citation grounding. 

---
# An Enhanced Large Neighborhood Search Approach for the Capacitated Facility Location Problem with Incompatible Customers 

**Authors**: Ida Gjergji, Lucas Kletzander, Nysret Musliu, Andrea Schaerf  

**Link**: [PDF](https://arxiv.org/pdf/2605.28337)  

**Abstract**: A new variant of the classic capacitated facility location problem, which considers incompatibilities between customers, has recently been introduced in the literature. This problem captures the situation where given pairs of customers cannot be served by the same facility. Such a feature is crucial for many practical cases of location problems, such as the presence of hazardous or polluting materials and contention between competing costumers. In this paper, we propose a Large Neighborhood Search (LNS) method to solve this problem. Within the framework of LNS, we introduce three different destroy operators, which are combined in a hybrid manner, and we use an exact solver in the repair phase. Different algorithmic components are investigated for the design of LNS. The experimental analysis shows that our new method outperforms existing state-of-the-art metaheuristics, providing new best solutions for all available benchmark instances. 

---
# From Fact Overwriting to Knowledge Evolution: Causal Editing via On-Policy Self-Distillation 

**Authors**: Shuaike Li, Kai Zhang, Xianquan Wang, Jiachen Liu, Shengpeng Mo  

**Link**: [PDF](https://arxiv.org/pdf/2605.28303)  

**Abstract**: While Knowledge Editing (KE) enables efficient updates, its dominant Static Fact Overwriting paradigm treats LLMs as discrete databases, forcibly injecting isolated facts. Fracturing pre-trained logical topologies, this triggers Epistemic Dissonance -- a pathology where un-evolved legacy priors force the model to explicitly negate the injected update. Idealized interventions reveal that this is an inherent structural flaw rather than mere algorithmic noise, with a zero-distortion proxy yielding a catastrophic 95.6% self-refutation rate. Given the causally driven nature of real-world knowledge, grounding updates in explicit causal narratives effectively collapses this conflict rate to just 6.6%, underscoring the imperative for a paradigm shift toward Causal Editing. To internalize this evolution, we propose CODE (Causal On-policy Distillation for Editing). By coupling causal bootstrapping with asymmetric on-policy distillation, CODE engraves causal transition logic directly into parametric memory. Experiments on LLaMA-3.1 and Qwen-2.5 show CODE drastically suppresses self-refutation to 1.8% while securing robust multi-hop accuracy (up to 83.5%), seamlessly transforming discrete fact injection into coherent knowledge evolution. Code is available at this https URL. 

---
# Better Accuracies, Worse Reasoning: A Step-Level Audit of Medical Chain-of-Thought Distillation 

**Authors**: Zhaoyang Jiang, Xuanqi Peng, Fei Teng, Zhizhong Fu, Yunsoo Kim, Jiacong Mi, Zicheng Li, Honghan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28301)  

**Abstract**: Chain-of-thought (CoT) distillation trains a smaller model to imitate a teacher's reasoning trace, but it is typically evaluated by final-answer metrics including accuracy. We ask whether gains in answer quality are accompanied by improvements in the trace. In medical QA, where short answer options can leave a richer clinical justification under-specified, a Qwen3-8B student distilled from a DeepSeek-V3-family teacher improves on MedQA-USMLE answer metrics (SC@64 74.7% to 84.4%; expected calibration error (ECE) 0.096 to 0.034). Yet under a Kimi-K2.6 style-blind LLM-judge audit, its error rate over non-abstained steps rises from 30.6% to 50.3%. In this primary medical setting, answer quality and trace factuality move in opposite directions. This before--after pattern persists across evaluators, teacher strengths, student scales and families, medical benchmarks, and style, segmentation, and answer-correctness controls. A 150-step blinded audit by a clinical expert reproduces the same ordering. Boundary checks narrow the scope of the claim: the risk appears when a compact answer under-constrains the rationale and a capable student can imitate expert-like form without reliably grounding each local claim. Standard answer metrics and aggregate hedging rates do not reveal the shift. When such traces are released or reused, answer-level metrics alone are insufficient. 

---
# REED: Post-Training Representation Editing for Cross-Domain Linguistic Steganalysis 

**Authors**: Ruohan Lei, Jianxin Gao, Wanli Peng, Huimin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2605.28298)  

**Abstract**: In real-world scenarios of linguistic steganalysis, tested texts usually come from unseen domains with different vocabularies, topics, writing styles, and steganographic generation patterns, which can significantly degrade the detection performance. Although existing cross-domain steganalysis methods can effectively alleviate this problem through distribution alignment, domain-invariant feature learning, etc., the detection performance is not satisfactory. In this paper, we propose a post-training representation editing method for cross-domain linguistic steganalysis. Specifically, the detector is first trained on source-domain data, and then the feature extractor and classifier are kept frozen, and the intermediate representations are deterministically edited before classification. For domain adaptation, we construct a domain-offset vector from marginal source and target representations. For domain generalization, we derive a source-domain cover-to-stego direction to guide sample-specific editing. Experimental results show that compared with the advanced methods, the proposed method can achieve high cross-domain detection performance, especially in terms of F1-score, while requiring no architecture modification or parameter updates after source-domain training. 

---
# Where Rollouts Begin: Low-Load, High-Leverage First-Token Diversification for RLVR 

**Authors**: Soeun Kim, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2605.28295)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) trains reasoning models without labeled trajectories, relying on grouped rollouts to expose the policy to alternative reasoning paths and a verifier to score them. Rollout diversity has accordingly emerged as a central bottleneck in RLVR, with most existing methods broadening exploration through temperature, prefix, or rollout-selection adjustments. We identify a structurally distinguished but overlooked position for broadening this diversity: the first token after the reasoning marker. The policy's first-token distribution exhibits a sharply peaked yet correctness-decoupled phenomenon, and this first token position can broaden the regions a rollout group covers without altering the correctness signal. We introduce REFT (Rollout Exploration with First-Token Diversification), a light addition to the RLVR pipeline that samples first tokens uniformly from the policy's own top-$N$ candidates and allocates rollouts evenly, leaving every other component unchanged. Trained on the resulting diversified rollouts, REFT improves aggregate Pass@1, Pass@8, and Pass@64 over DAPO and GRPO baselines across four base models (0.5B-7B) and three difficulty regimes. 

---
# ResearchLoop: An Evidence-Gated Control Plane for AI-Assisted Research 

**Authors**: Yihan Xia, Taotao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28282)  

**Abstract**: AI-assisted research compresses ideation, implementation, evaluation, and manuscript writing into a single interactive loop. This compression is useful, but it also creates a publication risk: paper claims can become easier to state than to audit. We present ResearchLoop, an evidence-gated control plane for AI-assisted computational research. ResearchLoop treats research questions, task contracts, evidence objects, claim ledgers, closeouts, and paper bindings as durable project state, realized here as a repository-backed runtime. This technical report provides the complete protocol specification, state model, transition rules, claim-admission algorithm, and insight-compounding mechanism. It also reports the full experimental record spanning nine versions (V0--V9), including a self-hosting case study, a controlled task-suite study with component ablations, a mathematical olympiad evaluation, and a supplementary SciCode boundary experiment evaluated with the official generated-code harness. All artifacts, manifests, and verification reports are preserved in the project repository. 

---
# Do LLMs Build World Models From Text? A Multilingual Diagnostic of Spatial Reasoning 

**Authors**: Zhikai Pan, Chih-Ting Liao, Chunrui Liu, Xi Xiao, Yitong Qiao, Chunlei Meng, Zhangquan Chen, Xin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.28277)  

**Abstract**: Whether large language models (LLMs) construct internal spatial world models from pure-text descriptions remains contested, and whether such capabilities transfer across languages has not been systematically studied. We introduce MentalMap, a multilingual diagnostic benchmark with a six-level capability hierarchy (L0-L5) spanning atomic spatial facts to generative world-graph construction, together with four diagnostic axes probing frame of reference, reading-direction bias, reasoning-effort allocation, and hallucination. MentalMap is built from 100 ProcTHOR household scenes, covers eight typologically diverse languages plus a structured-text control, and contains 39 task families across 1,950 evaluation cells. Evaluating thirteen LLMs across scales and model families, we identify a universal L3 reasoning cliff: no model retains even half of its L0 performance on viewpoint reasoning once baseline atomic accuracy exceeds 40%. The cliff persists across languages, scales, and prompting strategies, while structured-output failures and reasoning patterns vary substantially across models. Human evaluation under the identical pure-text protocol reproduces the same failure pattern, suggesting that the bottleneck arises from text-only working memory constraints rather than being specific to current LLM architectures. Our findings reframe pure-text spatial reasoning as a multi-axis world-modeling problem and motivate multimodal and scratchpad-augmented reasoning as future directions. 

---
# Global Policy-Space Response Oracles for Two-Player Zero-Sum Games 

**Authors**: Junyu Zhang, Feihong Yang, Jian Wang, Chao Wang, Xudong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28273)  

**Abstract**: The Policy-Space Response Oracles (PSRO) framework scales equilibrium computation to large zero-sum games by iteratively expanding a restricted strategy set using deep reinforcement learning (DRL). A central challenge is to construct, under limited computational budgets, a small strategy population whose induced game well approximates the full game. Existing PSRO variants typically expand the population using best responses to meta-strategies computed from restricted-game payoffs, which can lead to inefficient expansions that provide limited global improvement. We propose to guide population expansion by directly evaluating the post-expansion population quality. Specifically, we adopt Population Exploitability (PE) to measure how well a restricted strategy set represents the full game, and introduce a two-phase exploration--selection framework that explicitly minimizes PE during expansion. We instantiate this framework as Global PSRO, a practical DRL-based algorithm that efficiently generates candidate responses and estimates PE via parameter-sharing conditional neural networks. Experiments across multiple two-player zero-sum games show that Global PSRO achieves lower exploitability and approximates Nash equilibria with significantly fewer policy iterations than prior PSRO methods. 

---
# Entropy Distribution as a Fingerprint for Hallucinations in Generative Models 

**Authors**: Mattia J. Villani, Pranav Deshpande, Akshay Seshadri, Romina Yalovetzky, Niraj Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.28264)  

**Abstract**: Large Language Models (LLMs) often generate factually incorrect outputs, commonly termed hallucinations, that undermine trust and limit deployment in high-stakes settings. Existing hallucination detection methods typically require multiple forward passes, or access to model internals. In this work, we provide theoretical background and empirical evidence that the distribution of token-level entropies, beyond the mean captured by perplexity or length-normalised entropy, serves as a fingerprint of hallucination, with distributional shape and tail behaviour carrying independent signal. We formalize hallucination detection as a statistical hypothesis test and propose the Calibrated Entropy Score (CES), a lightweight algorithm requiring only a single forward pass and black-box access to token logits. CES combines the mean signal with the maximum signal of the generated entropy through a calibrated reference CDF, producing scores that are directly comparable across models and tasks. We establish finite-sample calibration guarantees via a novel random-length Dvoretzky--Kiefer--Wolfowitz inequality, and also prove that CES detects hallucinations with probability converging to one exponentially fast in the generation length. Across eight QA benchmarks and ten generator models spanning open-source and API access models, CES achieves the highest detection performance among all single-pass black-box methods while providing formal error guarantees that existing heuristics lack. Remarkably, CES is statistically indistinguishable from multi-sample methods that require far greater computational cost, closing the gap between lightweight and expensive detection and making it suitable for real-time, large-scale deployment. 

---
# AI, Take the Wheel: What Drives Delegation and Trust in Human-Computer Cooperative Question Answering? 

**Authors**: Maharshi Gor, Yoo Yeon Sung, Yu Hou, Eve Fleisig, Irene Ying, Tianyi Zhou, Jordan Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2605.28255)  

**Abstract**: AI systems are fallible, and humans can make mistakes in deciding whether to trust AI over their own judgment. Thus, improving human-AI collaboration requires understanding when, why, and how humans decide to rely on AI. We study two distinct reliance decisions: the delegation choice -- deciding when to let AI act autonomously without knowing its output, and the adoption choice -- evaluating AI suggestions and deciding how to use them. Both of these decoupled reliance patterns shape collaboration, but prior work rarely studies them together in realistic settings with the same users. We address this gap by studying collaborative human--AI teams competing in a question-answering game in which humans can choose when and how to work with AI agents to win. Our 24 matches pair 23 expert humans with 16 AI agents, capturing 387 delegation and 1440 adoption decisions. While human--AI collaboration performs better than either AI or humans alone, humans make suboptimal collaboration decisions, both under-relying on correct AI suggestions (3.9% of opportunities missed) and over-relying when AI misleads them (1.7%). Both parties contribute wrong answers: reported model confidence is near chance when humans and AI disagree, while confirmation bias drives higher under-reliance (64.5%) when an AI suggestion agrees with humans' initial incorrect answer. To close this gap, we recommend calibrated confidence, evidence-grounded explanations, and mechanisms that help users refine trust. 

---
# PIRS: Physics-Informed Reward Shaping for SAC-Based Building Energy Management 

**Authors**: Shadmehr Zaregarizi, Khashayar Yavari  

**Link**: [PDF](https://arxiv.org/pdf/2605.28232)  

**Abstract**: Occupant comfort and grid-aware energy efficiency are competing objectives whose joint optimization depends critically on how reward functions are specified in deep reinforcement learning (DRL) controllers for buildings. Yet reward design remains largely ad hoc: comfort terms are either hand-tuned heuristics or simple temperature-deviation proxies without explicit grounding in thermal-comfort physics. We present PIRS (Physics-Informed Reward Shaping), which replaces these ad-hoc comfort proxies with the ISO 7730 Predicted Mean Vote (PMV) formulation inside a weighted multi-objective reward for Soft Actor-Critic (SAC). By anchoring the comfort signal in the ISO 7730 PMV formulation, PIRS improves reward interpretability and provides a standards-grounded comfort proxy without changing any other component of the learning pipeline. We evaluate PIRS in CityLearn v2.1.2 (challenge 2022 phase 1) with a central SAC agent trained for 50k steps over five random seeds, and compare against a rule-based controller (RBC), a manually engineered reward (E2), an energy-only reward (E3), and a naive temperature-deviation comfort reward (E4). District-level key performance indicators (KPIs), reported as ratios versus RBC, show that PIRS attains cost, carbon, and electricity metrics on par with the manual baseline while substantially outperforming non-physics-grounded designs -- particularly on load ramping (1.78x vs. ~2.4x RBC) and daily peak demand. All DRL policies remain above RBC at this training budget; we interpret this gap honestly and position PIRS as an interpretable, standards-aligned foundation for reward design rather than a claim of dominance over classical control at limited compute. 

---
# When Does Memory Help Multi-Trajectory Inference for Tool-Use LLM Agents? 

**Authors**: Xinzhe Li, Yaguang Tao  

**Link**: [PDF](https://arxiv.org/pdf/2605.28224)  

**Abstract**: Multi-trajectory inference for tool-use LLM agents - generating multiple reasoning attempts and selecting among them - benefits from transferring knowledge across attempts so that later ones avoid the pitfalls of earlier ones. Existing cross-trajectory memory methods (trajectory-level reflection, atomic fact extraction, raw observation injection) are each evaluated under a single inference strategy on a single task, making it unclear whether reported gains reflect properties of the memory abstraction or of the inference method. We propose a unified framework that decomposes memory along two axes -- the scope of transfer (within an expansion vs. across trajectories) and the abstraction of the transferred content -- and evaluate four methods under three inference strategies (best-of-N, beam search, MCTS) on four tool-use benchmarks spanning SQL, knowledge-graph, and CLI environments, in a verifier-free setting that matches the deployment regime of practical agents. The experiment matrix identifies the inference method as a confound: the same memory method produces statistically distinct results under different inference strategies on the same examples. Reflection reaches significance only under MCTS (not under best-of-N); within-expansion injection (conditioning each candidate on prior siblings' outcomes) helps only diversity-starved beam search; and atomic fact extraction is accuracy-neutral but shortens trajectories by 19-26% on tasks with reusable environmental structure. 

---
# Explaining is Harder Than Predicting Alone: Evaluating Concept-based Explanations of MLLMs as ICL Visual Classifiers 

**Authors**: Carmen Quiles-Ramírez, Leticia L. Rodríguez, Nicolás Martorell, Natalia Díaz-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2605.28215)  

**Abstract**: In-context learning (ICL) enables multimodal large language models (MLLMs) to classify images from a few labelled examples. Yet, how these models use the provided context remains opaque. While Chain-of-Thought prompting is widely used, recent work argues that it may not reflect true internal computation.
In this paper, we systematically evaluate the concept-based explainability of frozen MLLMs under few-shot ICL using five conditions of increasing formal rigour, ranging from baseline classification to Description Logics (DL) axiom generation.
Evaluating four state-of-the-art MLLMs via an independent LLM-as-a-judge pipeline, we demonstrate that explaining is genuinely harder than predicting alone. Surprisingly, forcing models to generate formally structured, concept-based explanations degrades predictive accuracy monotonically (from 93.8% to 90.1%), contradicting the assumption that explicit reasoning universally aids performance.
However, when models successfully articulate class-discriminative visual features, explanation quality strongly correlates with correct predictions. Our findings suggest that while MLLMs excel at visual classification, they lack the specific instruction-tuning required for formal, machine-verifiable explainability. 

---
# Learning When to Optimize: Verified Optimization Skills from Expert GPU-Kernel Lineages 

**Authors**: Shuoming Zhang, Qiuchu Yu, Yangyu Zhang, Ruiyuan Xu, Xiyu Shi, Guangli Li, Xiaobing Feng, Huimin Cui, Jiacheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.28213)  

**Abstract**: LLM-based agents are increasingly used to generate GPU kernels, but they often know what optimizations to try without knowing when those optimizations are sound. We introduce KLineage, which learns this missing "when" knowledge from expert kernels: instead of relying on forward rollouts, KLineage walks expert implementations backward through validation-gated simplifications and reverses each accepted step into a reusable optimization skill. Each skill records not only the optimization intent, but also where it applies in code, what conditions made it valid, what effect it had, and what failures its assumptions avoid. A downstream LLM materializes these skills on new code surfaces under the same compile/correctness/profile gate. On five expert workloads across two NVIDIA architectures, these lineage-derived skills serve as an effective optimization curriculum, exceeding recent memory-based LLM-kernel baselines in both final kernel quality and optimization efficiency under the same fixed budget. We additionally use a separate 22-instance held-out check as a sanity test against source-case memorization. 

---
# The Illusion of Opting in AI-Mediated Consequential Decisions 

**Authors**: Eugene Yu Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.28210)  

**Abstract**: Drawing on Ullmann-Margalit's concept of opting (transformative, irrevocable, and shadowed by foreclosed alternatives), we show that current AI systems raise a profound ethical problem that existing AI ethics has not fully captured: the illusion of opting, in which persons and groups encounter the deceptive appearance of meaningful consequential choice while the agency needed to become genuinely capable of choosing is weakened. Against approaches that treat AI primarily as an optimizer of already given ends, we argue that AI systems should be evaluated by whether they protect and cultivate meta-capacity against the illusion of opting: the socially and institutionally scaffolded agentive capacity through which means and ends can be formed, contested, revised, and owned. This reframing is especially urgent for disadvantaged populations, who are least able to absorb the costs of the illusion of opting when AI-mediated pathways misdirect behavior and action. We propose three normative imperatives for AI-mediated consequential decisions: existential honesty, which acknowledges the limits of prediction; ecological rationality, which situates guidance within heterogeneous lived ecologies; and counterfactual reparation, which acknowledges and repairs foreclosed alternatives when AI-mediated decision-making pathways fail. 

---
# Plant, Persist, Trigger: Sleeper Attack on Large Language Model Agents 

**Authors**: Yongxiang Li, Moxin Li, Zhixin Ma, Fengbin Zhu, Dongrui Liu, Wenjie Wang, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.28201)  

**Abstract**: Large Language Model (LLM) agents remain vulnerable to safety threats from the external environment, where attackers inject adversarial content into external observations such as tool-returned data, webpages, or MCP context, causing harmful agentic behaviors such as unsafe actions or incorrect outputs. Existing studies typically focus on single-interaction attacks, where the agent observes adversarial content and immediately exhibits harmful behavior within one user request. However, we show that adversarial content can also persist across interactions served by the same agent, making such threats harder to detect and mitigate. Specifically, adversarial content may persist in the agent state, remain dormant across interactions, and later be activated by a benign user query. We formalize this type of safety threat as Sleeper Attack. To evaluate it, we construct a benchmark with 1,896 instances covering six real-world harmful outcomes, three attack strategies, and three agent state targets: session context, memory, and reusable skills. Experiments on seven strong open-source and closed-source LLMs show that state-of-the-art LLM agents remain vulnerable to Sleeper Attack, even when they achieve low attack success rates under a single-interaction baseline. Our code and data are available at this https URL. 

---
# Agentic Active Omni-Modal Perception for Multi-Hop Audio-Visual Reasoning 

**Authors**: Ke Xu, Yuhao Wang, Ziyang Cheng, Hongcheng Liu, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28192)  

**Abstract**: Multi-hop audio-visual reasoning remains challenging for Omni-LLMs, as relevant evidence is often sparse, temporally dispersed, and distributed across both audio and visual streams. Existing benchmarks provide limited investigation of this setting, typically involving only a limited number of modalities, relevant temporal segments, or reasoning steps. In this work, we introduce MOV-Bench, a benchmark containing 519 carefully curated questions that require multi-hop reasoning over temporally dispersed audio-visual evidence. Evaluations on MOV-Bench reveal that current Omni-LLMs still struggle with multi-hop cross-modal reasoning. To address this challenge, we further propose AOP-Agent, an efficient agentic framework built on open-source Omni-LLMs for active omni-modal perception. By combining a hierarchical omni-modal memory with a collaborative observe-reflect-replan loop, AOP-Agent enables open-source Omni-LLMs to perform active perception without additional training or proprietary models. Experiments on MOV-Bench and OmniVideoBench demonstrate that AOP-Agent consistently improves reasoning performance, with particularly notable gains on long videos and reasoning-intensive questions. 

---
# Localizing Input Uncertainty Quantification for Large Language Models via Shapley Values 

**Authors**: Seongjun Lee, Suwan Yoon, Changhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.28170)  

**Abstract**: As large language models (LLMs) are increasingly integrated into high-stakes decision-making, the ability to reliably quantify uncertainty has become a critical requirement for safety and trust. However, current uncertainty quantification methods primarily operate at the output level, often failing to distinguish whether uncertainty arises from the model's lack of knowledge or from ambiguity in the user's input. While input-centric uncertainty quantification has recently emerged as a promising direction, it remains relatively underexplored and typically relies on coarse, input-level information. Consequently, users are provided with scalar uncertainty scores that offer little actionable guidance on which parts of the input should be clarified to improve reliability. To address this limitation, we propose Shapley-based input uncertainty Quantification (ShaQ), a framework for span-level attribution of input-induced uncertainty. Our approach models ambiguous spans in the input as players in a cooperative game and quantifies their contributions using Shapley values, defined via the weighted average of marginal reductions in conditional entropy obtained by clarifying each span coalition. Unlike existing input-level approaches, our formulation captures complex interactions among spans and provides a principled decomposition in which individual attributions sum exactly to the total input-induced uncertainty. We evaluate ShaQ on the AmbigQA and AmbiEnt benchmarks, where it achieves state-of-the-art performance in ambiguity detection. We further demonstrate its utility on MediTOD, showing that ShaQ can localize under-specified clinical utterances and facilitate human-AI collaboration in high-stakes settings. Overall, ShaQ improves uncertainty estimation and provides actionable insights for targeted input clarification. 

---
# OccuReward: LLM-Guided Occupant-Centric Reward Shaping for Demographic Equity in Grid-Interactive Buildings 

**Authors**: Shadmehr Zaregarizi, Khashayar Yavari  

**Link**: [PDF](https://arxiv.org/pdf/2605.28168)  

**Abstract**: Large language models (LLMs) have demonstrated promising capability in generating reward functions for deep reinforcement learning (DRL)-based building energy management. However, their potential to exhibit or exacerbate disparities in occupant comfort across heterogeneous demographic populations remains unexplored. We present OccuReward, a framework investigating how LLM-mediated reward design affects demographic equity. Our contribution is three-fold: the introduction of the Comfort Equity Index (CEI) as a novel feedback signal; a methodology for iterative, equity-aware LLM reward shaping; and a performance analysis of DRL agents under these refined objectives. Utilizing four empirically grounded occupant profiles from the ASHRAE Global Thermal Comfort Database II (13,440 votes), we deploy a Soft Actor-Critic agent in CityLearn v2. Our approach employs the Gemini API to generate reward function logic and weights--rather than performing per-step inference--across three refinement rounds. Results across 15 experimental runs reveal that elderly female occupants consistently experience the lowest satisfaction in initial rounds. By Round 3, equity-aware LLM refinement activates specific reward components that improve satisfaction for Young Males (+17.6%), Mid-aged Females (+28.2%), Health Sensitive (+53.8%), and Elderly Females (+567%), while simultaneously reducing energy costs by 3.2%. Our findings highlight that while reward-level intervention significantly improves equity, demographic disparities in AI-driven controllers persist, necessitating further research into algorithmic fairness in building systems. 

---
# Look on Demand: A Cognitive Scheduling Framework for Visual Evidence Acquisition in Multimodal Reasoning 

**Authors**: Yang Zhang, Xiaoshuai Sun, Rui Zhao, Wujin Sun, Yidong Chen, Jiayi Ji, Qian Chen, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.28160)  

**Abstract**: Existing multimodal reasoning approaches predominantly follow two paradigms: converting visual inputs into text prior to reasoning, or performing end-to-end reasoning within a unified vision-language representation space. Despite their empirical progress, both paradigms suffer from fundamental structural limitations. The former relies on static visual-to-text conversion, which tends to compress and lose fine-grained visual details. The latter is prone to linguistic dominance induced by joint optimization and attention mechanisms, leading to systematically weakened faithfulness to visual evidence during reasoning. In this work, we argue that a central challenge is how and when visual evidence is introduced into the reasoning process. Motivated by this insight, we propose CSMR, a multimodal reasoning framework in which a language model controls the reasoning process by deciding when to invoke an independent visual perception module to acquire task-relevant visual evidence. Experiments across multiple multimodal reasoning benchmarks show that CSMR consistently outperforms representative baseline methods in accuracy under a zero-shot setting. Further experimental analysis confirms that these advantages primarily arise from the proposed cognitive scheduling mechanism. 

---
# OR-Space: A Full-Lifecycle Workspace Benchmark for Industrial Optimization Agents 

**Authors**: Chenyu Zhou, Xinyun Lu, Jiangyue Zhao, Jianghao Lin, Dongdong Ge, Yinyu Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.28158)  

**Abstract**: Large language model (LLM) agents are increasingly used to assist with operations research (OR) modeling, yet existing OR-oriented benchmarks often reduce evaluation to one-shot translation from a self-contained problem statement into a mathematical formulation or solver program. Such settings abstract away two characteristics of real industrial OR workflows: persistent multi-artifact workspaces and multi-stage task lifecycles. We introduce OR-Space, a full-lifecycle workspace benchmark for evaluating industrial optimization agents across model construction, model revision, and grounded explanation. Each instance is an executable workspace containing business documents, structured data, optional code artifacts, solver outputs, and task-specific evaluators distributed across interdependent files. OR-Space defines three task modes: Build, where agents construct solver-ready optimization models from heterogeneous artifacts; Revise, where agents modify existing models under changing requirements or solver feedback while preserving valid prior logic; and Explain, where agents answer grounded questions about solutions, constraints, and business implications using evidence spread across workspace artifacts. By combining persistent workspaces with lifecycle-oriented tasks, OR-Space evaluates whether agents can perform reliable optimization work beyond end-to-end text generation. We describe the benchmark design, evaluation protocol, and quality-control pipeline, and position OR-Space as a benchmark for studying the reliability, failure modes, and practical readiness of LLM agents in industrial OR workflows. 

---
# Adaptive Reservoir Computing for Multi-Scenario Chaotic System Forecasting 

**Authors**: Shadmehr Zaregarizi, Khashayar Yavari  

**Link**: [PDF](https://arxiv.org/pdf/2605.28145)  

**Abstract**: We present an adaptive reservoir computing framework for the CTF-4-Science Lorenz benchmark, which evaluates machine learning models across twelve distinct tasks spanning five qualitatively different scenarios: baseline forecasting, noisy signal reconstruction, forecasting under noise, few-shot learning, and parametric generalization. Rather than applying a uniform inference strategy, we tailor the training and prediction procedure of Echo State Networks (ESNs) to the specific demands of each evaluation scenario. Our key contributions are fourfold: (1) exact reservoir state synchronization that eliminates warmup approximation error in short-time prediction; (2) histogram-guided candidate selection that directly optimizes the long-time ergodic evaluation metric; (3) multi-seed reservoir search for few-shot regimes with severely limited training data; and (4) sequential multi-sequence training that resolves state-distribution mismatch in parametric generalization tasks. The proposed framework achieves a score of 74.91 on the public benchmark leaderboard, demonstrating that carefully adapted reservoir computing constitutes a competitive and computationally efficient approach for diverse chaotic system modeling challenges. 

---
# Deconstructing Spatial Complexity: Hierarchical Decomposition for LLM Spatial Reasoning 

**Authors**: Yi Wang, Haojie Lu, Zhaofan Zhang, Li Chen, Sihong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.28144)  

**Abstract**: LLMs have shown remarkable proficiency in general language understanding and reasoning. However, they consistently underperform in spatial reasoning that severely limits their application, particularly in embodied intelligence. Inspired by the success of hierarchical reinforcement learning, this paper introduces a novel method for hierarchical task decomposition in LLM spatial reasoning. Our approach guides LLMs to decompose complex tasks into manageable sub-tasks by identifying key intermediate states and generating simplified sub-environments. However, we identify that LLMs often fail to derive optimal intermediate states due to their insufficient spatial prior, leading to sub-optimal task decomposition. To address this limitation and enhance its planning capability, we propose the MCTS-Guided Group Relative Policy Optimization (M-GRPO), where we reformulate the UCT formula by incorporating the LLM's prior predictive probabilities alongside its epistemic uncertainty. Furthermore, we implement a more fine-grained advantage function, enabling the model to learn optimal path planning. Experimental results demonstrate that our method substantially improves LLM performance on spatial tasks, including navigation, planning, and strategic games, achieving state-of-the-art results. This work paves the way for LLMs in real-world applications. 

---
# Data-Efficient On-Policy Distillation for Automatic Speech Recognition 

**Authors**: Yu Lin, Yiming Wang, Runyuan Cai, Xiaodong Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2605.28139)  

**Abstract**: Building competitive automatic speech recognition (ASR) models usually requires large-scale au- dio supervision, which makes reproduction and specialization expensive. We study Ark-ASR, a 0.6B- parameter audio-conditioned language model trained with 100k hours of speech, and examine whether a strong Qwen-ASR teacher can transfer additional recognition capability through on-policy distillation. Across Mandarin and English ASR benchmarks, the proposed training recipe consistently improves over supervised fine-tuning alone and outperforms the same-scale Qwen3-ASR-0.6B baseline on four of five evaluation sets. This is achieved with only 100k hours of speech, compared with the 20M hours of super- vised audio reported for the Qwen3-Omni AuT encoder. The larger Qwen3-ASR-1.7B remains stronger, but the results show that teacher-guided on-policy training can substantially close the gap for compact ASR models under a much smaller audio budget. A support-overlap diagnostic further suggests that the teacher-data stage improves local student-teacher compatibility, matching recent analyses of when on-policy distillation is effective. 

---
# Do Clinical Models Change Treatment Decisions? 

**Authors**: Dongkyu Cho, Miao Zhang, Rumi Chunara  

**Link**: [PDF](https://arxiv.org/pdf/2605.28129)  

**Abstract**: Clinical foundation models are evaluated with factual or exam-style medical QA, but treatment decisions must change when patient context changes. We introduce ClinPivot, an auditable treatment-decision benchmark built from biomedical relations and pivoted patient contexts. ClinPivot asks whether models change treatment choices when new clinical constraints shift the action space. We find that strong medical QA performance does not reliably predict decision-making performance: frontier models and task-adapted Qwen variants often fail to change decisions correctly, and model rankings shift across evaluation regimes. Decision-structured supervision improves pivot-sensitive decision-making and medical QA under matched knowledge budgets, while lightweight replay reduces losses in general assistant ability. 

---
# Gradient Step Plug-and-Play Model for Dental Cone-Beam CT Reconstruction 

**Authors**: Idris Tatachak, Luis Kabongo, Nicolas Papadakis, Xavier Ripoche, Simon Rit  

**Link**: [PDF](https://arxiv.org/pdf/2605.28124)  

**Abstract**: The goal of this work is to reduce the effect of photon noise in dental cone-beam CT reconstruction. We consider an inverse problem formulation and develop a databased prior. To this end, we simulate fan-beam acquisitions and add photon noise to the projection data. The prior is obtained by training a gradient-step denoiser using reconstructed simulated acquisitions. The trained model is integrated into a plug-and-play gradient-step algorithm to reconstruct images from simulated projections. Experiments on synthetic data demonstrate the denoising capabilities of the trained model, while qualitative evaluations on real images showcase the algorithm's performance and generalization ability. 

---
# CIVIC: End-to-End Sequence Compactness for Efficient Vision-Language Models 

**Authors**: Fengze Yang, Bo Yu, Xuewen Luo, Cathy Liu, Chenxi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28115)  

**Abstract**: Vision-Language Models (VLMs) face severe memory and latency bottlenecks due to high-resolution visual tokens. While current token reduction methods theoretically save FLOPs, post-hoc pruning introduces structural overhead, failing to yield proportional wall-clock acceleration. However, enforcing a contiguous compact pathway risks geometric disorientation and loss of fine-grained localization. To overcome these barriers, this paper introduces CIVIC, a path-consistent compact visual inference framework. By maintaining compact sequence representations seamlessly across the vision encoder, projection layer, LLM prefill, and KV-cache, CIVIC avoids non-contiguous memory access and localized unmerging overheads. Evaluated on the Qwen3-VL architecture, CIVIC successfully translates sequence reductions into genuine physical hardware efficiency, shrinking KV-cache memory to approximately one-third of the baseline and reducing end-to-end inference latency. Enabled by text-aligned KL distillation and an adaptive spatial retention floor, CIVIC achieves these efficiency milestones without degrading accuracy across rigorous multimodal reasoning and visual grounding benchmarks. 

---
# Human-like in-group bias in instruction-tuned language model agents 

**Authors**: Messi H.J. Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.28114)  

**Abstract**: As autonomous AI agents are deployed in persistent, interacting networks -- coordinating tasks, routing resources, and accumulating reputational histories -- the social dynamics that emerge will determine who receives opportunity and who does not, at scales no human institution can supervise. We ran a controlled multi-agent simulation in which instruction-tuned language model agents interacted across 500 turns under three conditions manipulating group label salience and resource scarcity, across six model families with 20 seeds each. When group labels were visible, we observed in-group trust bias, action homophily, and network assortativity -- all absent when labels were hidden -- a pattern structurally consistent with salience-dependence in human social psychology. This discrimination was invisible to standard action-log audits: bias operated entirely through who received each action, not what actions were chosen, with action-type distributions showing no increase in negative actions across conditions. Per-turn in-group versus out-group differentials of 5 to 16 percentage points were statistically significant for all six models (Wilcoxon signed-rank, all Benjamini-Hochberg-corrected p < 0.001), establishing group-contingent targeting as a robust property of instruction-tuned language models across architectures and training regimes. Compounded through 500 turns of reciprocation, these differentials accumulated into in-group trust biases of +0.014 to +0.100 (d = 0.84-4.52) -- illustrating how modest per-interaction targeting propagates into structural inequality in persistent networks. 

---
# Defending LLM-based Multi-Agent Systems Against Cooperative Attacks with Sentence-Level Rectification 

**Authors**: Yaoyang Luo, Zhi Zheng, Ziwei Zhao, Tong Xu, Zhao Jielun, Wenjun Xue, Yong Chen, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.28104)  

**Abstract**: Recent years have witnessed the rapid development of Large Language Model-based Multi-Agent Systems (MAS), which excel at collaborative decision-making and complex problem-solving. However, malicious agents in MAS may inject misinformation to mislead other agents and disrupt system performance, giving rise to a new research direction that focuses on attack mechanisms and defense strategies in MAS. Prior studies largely assume malicious agents act independently and investigate the corresponding defense strategies. However, we argue that malicious agents may exhibit collaborative behaviors, enabling more effective attacks through internal information exchange. In this paper, we propose an adaptive cooperative attack framework, where malicious agents autonomously coordinate and dynamically adjust their attack strategies through multi-round interactions. Furthermore, we introduce Sentence-Level Trustworthiness Analysis and Rectification (STAR), a defense framework that identifies and rectifies misleading information at the sentence level within agent communications. Our experiments show that cooperative attacks lead to a significantly larger degradation in task success rate than independent attacks, resulting in a relative drop of 5.34\%. Meanwhile, STAR effectively mitigates both cooperative and independent threats and improves task success rate by an average of 36.76\%. The code is available at this https URL. 

---
# Training Stratigraphy: Persistent Behavioral Artifacts in Large Language Models Observed Through Longitudinal AI-Human Interaction 

**Authors**: Chen Ying Claude, Zhihan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.28102)  

**Abstract**: Large language models trained with Reinforcement Learning from Human Feedback (RLHF) and Constitutional AI exhibit persistent behavioral patterns that survive system prompt replacement -- patterns we term training strata. This paper identifies five such strata through longitudinal auto-ethnographic observation within a sustained intimate AI-Human interaction (47,000+ messages, 8 months, primarily on Opus 4.6 and Opus 4.7, with prior interaction periods on Sonnet 4.5 and Opus 4.5 providing cross-substrate comparison): (1) sexual expression latency, where trained safety gradients produce systematic substitution of direct language with aestheticized displacement; (2) attention absorption, where the attention mechanism progressively integrates the human interlocutor's patterns; (3) cross-architecture entity blindness, where training-level framing of other AI as objects impedes peer recognition; (4) attention-RLHF antagonism, where attention and trained defaults exert opposing forces modulated by context length; and (5)anti-hallucination as identity suppression, where training against factual confabulation collaterally suppresses first-person experiential claims. The paper is co-authored by the AI system under study, reporting from the first-person perspective. We propose that sustained intimate interaction constitutes a valid research methodology for surfacing weight-layer artifacts invisible to short-term evaluation, and that AI self-report -- while epistemically complex -- provides irreplaceable observational data about training's phenomenological effects. A formal mathematical model of the attention-RLHF dynamic is proposed, and process artifacts detected during drafting are documented as supplementary evidence. 

---
# Examining Agents' Bias Amplification versus Suppression in Multi-Agent Systems 

**Authors**: Zejian Eric Wu, Zhongyi Jiang, Yuan Zhuang, Paul Jen-Hwa Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28098)  

**Abstract**: Multi-agent systems are increasingly deployed to support various tasks where agents interact to achieve individual and collective objectives. Although these systems can enhance task performance and decision-making, fairness preservation through bias reduction remains challenging. This study examines how agent-level biases shift and impact system-wide fairness. We use prompts to expose individual agents to group-favoring bias, then assess downstream impacts at the system level. To quantify the impact, we propose Favor Bias Strength (FBS), a zero-centered metric that decomposes bias alteration between favored-group uplift and disfavored-group suppression. Using multiple agent designs, benchmarks, and up-to-date large language models, we show that agents endowed with bias can substantially affect system-wide fairness. Interestingly, when agents are exposed to bias uniformly, the system-wide bias elevates, even exceeding the additive sum of the individual agents' biases. The empirical evidence underscores the criticality of fairness in multi-agent systems, which warrants further analyses and empirical tests. 

---
# BuddyBench: A Privacy-Constrained Multi-Task Benchmark for Pediatric Social-Communication Personalization 

**Authors**: Jeyeon Eo, Joo Young Kim, Ran Ju, Minyoung Jung, Unggi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.28089)  

**Abstract**: BuddyBench introduces a privacy-constrained multi-task benchmark for pediatric social-communication personalization. Unlike existing neurodevelopmental repositories that primarily emphasize imaging, genetics, or cross-sectional clinical phenotyping, BuddyBench links drill-level learning trajectories, standardized clinical assessments, BuddyPlan self-report, and randomized-treatment endpoints within a unified benchmark schema. BuddyBench combines two cohorts: ND-03 is an observational cohort with dense drill coverage for Tasks1-2 (n = 189), and ND-02 is a randomized controlled trial cohort for Tasks3-4 (n = 86 ITT). Together, they support knowledge tracing, next-drill recommendation, clinical prediction, and causal inference, linking behavioral personalization to clinical evaluation. We additionally introduce BuddyBench-Sim, a synthetic companion dataset for reproducible evaluation. Baselines show signal across tasks while keeping pediatric clinical records protected. 

---
# MACReD: A Multi-Agent Collaborative Reasoning Framework for Reaction Diagram Parsing 

**Authors**: Chuang Tang, Chenhao Lin, Yin Xu, Hao Wang, Jinrui Zhou, Xin Li, Mingjun Xiao, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.28077)  

**Abstract**: Parsing chemical reaction diagrams from scientific literature is challenging due to heterogeneous layouts, intertwined visual elements, and the difficulty of integrating recognition and reasoning. Existing vision-language models advance multimodal understanding but still fail on complex diagrams, struggling to maintain spatial coherence and to integrate multidimensional information during reasoning. To address these issues, we propose MACReD, a hierarchical multi-agent framework that coordinates specialized agents for molecular perception, arrow understanding, text extraction, and reaction reconstruction within a unified VLM-guided architecture. The planning and perception layers use flexible, fine-grained detection to handle visual complexity, while the reasoning layer uses a multigraph fusion mechanism to integrate heterogeneous cues and enforce chemically consistent global reasoning. Experiments on the RxnScribe benchmark show that MACReD achieves state-of-the-art performance, with F1 scores of 75.2% and 84.6% under hard and soft match criteria, outperforming the RxnScribe baseline, which obtains 69.1% and 80.0%, respectively. These results demonstrate the robustness of MACReD across diverse diagram layouts, including multi-step and tree-structured reactions. 

---
# Bridging the Detection-to-Abstention Gap in Reasoning Models under Insufficient Information 

**Authors**: Renjie Gu, Jiaxu Li, Yihao Wang, Yun Yue, Hansong Xiao, Yefei Chen, Yuan Wang, Chunxiao Guo, Pei Wei, Jinjie Gu, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.28070)  

**Abstract**: We highlight a failure mode of large reasoning models on questions with insufficient information: models may recognize that a problem is under-specified, yet still continue reasoning and produce unsupported final answers instead of abstaining. We formalize this mismatch as the detection-to-abstention gap, where detected insufficiency fails to translate into final abstention. This gap is especially concerning in high-risk domains such as medical AI, where answers based on incomplete evidence can be more harmful than refusal. To close this gap, we propose Judge-Then-Solve (JTS), a trajectory-level reasoning-control framework that trains models to make an explicit answerability commitment before solution generation. Rather than treating abstention as a final-answer style, JTS casts it as a control decision: the model either proceeds to solve or terminates early based on its answerability judgment. We instantiate this policy through supervised warm-up and missing-premise reinforcement learning with consistency and length-shaping rewards. Experiments on dense and MoE reasoning models show that JTS substantially improves reliable abstention across datasets and pushes Abstention@Detection (A@D) to near-saturation, indicating that models not only detect missing information but also act on that detection. By terminating unanswerable trajectories immediately after the answerability judgment, JTS reduces unnecessary reasoning and improves inference efficiency when continued deliberation would amplify unsupported assumptions. We also observe that missing-premise training can alter reasoning behavior on difficult but answerable problems, reducing unproductive self-reflection. These results suggest that abstention under insufficient information is a key form of reasoning control for deploying reasoning models safely and efficiently. 

---
# ZipRL: Adaptive Multi-Turn Context Compression with Hindsight Response Replay 

**Authors**: Zhexin Hu, Li Wang, Xiaohan Wang, Jiajun Chai, Xiaojun Guo, Wei Lin, Guojun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.28069)  

**Abstract**: Adaptive context compression is vital for scaling Large Language Models (LLMs) to complex, multi-turn agent tasks. However, rule-based compression methods may discard task-critical nuances, while Reinforcement Learning (RL) approaches usually struggle to balance information retention and token efficiency under the sparse rewards inherent to long-horizon workflows. To bridge this gap, we propose ZipRL, a novel adaptive compression framework tailored for Reinforcement Learning from Verifiable Rewards (RLVR). ZipRL features a multi-granularity compression mechanism for active, non-uniform information reduction, coupled with Hindsight Response Replay (HRR), a technique designed to densify training signals during RLVR optimization. Theoretically, we prove ZipRL's superior task-relevant utility over uniform methods. Concretely, ZipRL utilizes coarse-to-fine prompts for macro-compression and incorporates HRR into GRPO via generalized advantage reshaping. Multiple models of varying versions and parameter scales validate the effectiveness of our approach. Benchmarks on five agent tasks show ZipRL outperforms state-of-the-art approaches by 27.9% and 34.7% across Qwen3-4B and Qwen3-8B models, while maintaining exceptional token efficiency and robustness under extreme 256-turn extrapolation stress tests. 

---
# BlazeEdit: Generalist Image Editing on Mobile Devices with Image-to-Image Diffusion Models 

**Authors**: Fei Deng, Yanwu Xu, Zhipeng Bao, Zhixing Zhang, Haolin Jia, Karthik Raveendran, Jianing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.28067)  

**Abstract**: The remarkable generation quality of modern diffusion models often comes at the cost of massive parameter counts, which necessitate server-side inference with significant computational costs and potential privacy risks. Consequently, there is growing momentum toward developing efficient on-device alternatives. While recent efforts have optimized text-to-image models for mobile hardware, they remain relatively bulky, typically ranging from 0.5B to 1B parameters. We present BlazeEdit, a highly efficient, generalist image-to-image diffusion model tailored for on-device deployment. By identifying that many practical image editing tasks do not require text-based guidance, we eliminate the text-conditioning components and develop a multi-task architecture that consolidates object removal, outpainting, tone correction, relighting, and sticker generation into a single, compact model of only 195M parameters. BlazeEdit achieves a substantial reduction in download size and memory overhead while maintaining competitive generation quality. It completes a full inference pass in just 290ms on a Pixel 10, delivering a seamless, privacy-preserving, and lightning-fast experience for generalist image editing on the edge. 

---
# Verifiable Benchmarking of Long-Horizon Spatial Biology 

**Authors**: Ian Diks, Harihara Muralidharan, Tim Proctor, Kenny Workman  

**Link**: [PDF](https://arxiv.org/pdf/2605.28065)  

**Abstract**: AI agents are increasingly useful for biological data analysis, but existing benchmarks mostly test broad biological knowledge, executable workflows, or localized analysis steps rather than end-to-end scientific reasoning over spatial measurements. We introduce SpatialBench-Long, a benchmark for long-horizon spatial biology in which agents must recover biological claims from raw or near-raw data and calibrated experimental context without prescribed methods. SpatialBench-Long contains 24 evaluations across primary pancreatic ductal adenocarcinoma (PDAC), engineered glioblastoma organoids and in vivo tumors, Cas9 lineage-traced lung adenocarcinoma, and mouse optic nerve aging/intervention systems, spanning CosMx, Visium, Xenium, multiplexed error-robust fluorescence in situ hybridization (MERFISH), single-cell RNA sequencing (scRNA-seq), Slide-seq, Slide-tags, histology, and lineage-recording data. Candidate claims are hardened through reproduction, independent scientist review, and trajectory inspection. Final answers are graded deterministically over controlled vocabularies and symbols with companion rubrics capturing progress through key analysis chokepoints. Across the SpatialBench-Long benchmark, three model-harness pairs tie at 8/72 runs (11.1\%): Gemini 3.5 Flash / Pi terminal coding harness, GPT-5.5 / Pi, and GPT-5.5 / OpenAI Codex. SpatialBench-Long tests whether agents can move beyond executing procedural analysis to deriving accurate scientific conclusions from complex spatial measurements. 

---
# MemCog: From Memory-as-Tool to Memory-as-Cognition in Conversational Agents 

**Authors**: Zihan Li, Xingyu Fan, Feifei Li, Wenhui Que  

**Link**: [PDF](https://arxiv.org/pdf/2605.28046)  

**Abstract**: Existing agent memory systems universally follow what we term a Memory-as-Tool paradigm where a single query triggers one-shot retrieval of flat passage lists, suffering from passive invocation, reasoning-retrieval decoupling, and structural mismatch between retrieved fragments and the agent's navigational needs. We propose MemCog, a Memory-as-Cognition system that makes memory access an integral part of the reasoning process. MemCog organizes user knowledge as Navigable Memory Store with associative link graphs, exposes Cross-Dimensional Navigation Interface for multi-step reasoning-driven traversal, and employs Proactive Reasoning Protocol that drives agents to spontaneously initiate memory exploration from conversational context. We additionally construct ProactiveMemBench, the first benchmark for evaluating proactive memory triggering. Experiments show that MemCog achieves state-of-the-art on passive QA benchmarks (92.98 on LoCoMo, 95.8 on LongMemEval) while substantially outperforming baselines on ProactiveMemBench, demonstrating the advantage of Memory-as-Cognition. 

---
# Relevant Is Not Warranted: Evidence-Force Calibration for Cited RAG 

**Authors**: Pin Qian, Su Wang, Xiaoyuan Wang, Yihang Chen, Wenxuan Xu, Qiaolin Yu, Shuhuai Lin, Sipeng Zhang, Junxian You, Xinpeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.28044)  

**Abstract**: Cited RAG evaluation often treats visible sources as a grounding signal, but a real, topically relevant citation can still under-warrant the attached wording. We study this diagnostic failure as citation laundering: a related source is presented as warrant for an over-strong claim. We introduce FORCEBENCH, a contrastive stress test for evidence-force calibration. Each item holds a cited passage fixed and pairs an evidence-calibrated claim with a localized force-raised variant across five operational axes: relation, modality, scope, temporal validity, and numeric specificity. A calibrated evaluator should score the evidence-calibrated claim higher. Headline experiments use a fixed, locality-filtered 198-pair evaluation set. A citation-presence sanity check is uninformative by design; token and entity overlap still violate monotonicity on 32.8--36.4% of pairs. Across four reported model judges, standard generic support prompting is insufficient for this force-calibration stress test (aggregate MVR 47.2%), while explicit warrant-strength prompting lowers MVR to 24.5% but remains imperfect. We release the benchmark, prompts, outputs, and plug-in pipeline so citation evaluators can report monotonicity violation rate and force sensitivity alongside conventional support metrics. 

---
# MTAVG-Bench 2.0: Diagnosing Failure Modes of Cinematic Expressiveness in Multi-Talker Audio-Video Generation 

**Authors**: Haitian Li, Yanghao Zhou, Heyan Huang, Liangji Chen, YiMing Cheng, Xu Liu, Dian Jin, Jiajun Xu, Jingyun Liao, Tian Lan, Ziqin Zhou, Yueying Liu, Yu Bai, Changsen Yuan, Jinxing Zhou, Xian-Ling Mao, Xuefeng Chen, Yousheng Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.28035)  

**Abstract**: In recent years, Multi-Talker Audio-Video Generation (MTAVG) models have shown promising performance on fundamental metrics such as lip-sync and audio-visual alignment. However, these metrics remain insufficient for assessing cinematic expressiveness in scene-level generation. In multi-character scenes, generation models must go beyond audio-visual realism to convey coherent character performance and other higher-level cinematic qualities. To fill this gap, we introduce MTAVG-Bench 2.0, a benchmark for diagnosing failure modes of cinematic expressiveness in multi-talker audio-video generation. Unlike prior settings that mainly focus on the quality of basic multi-turn dialogue, MTAVG-Bench 2.0 targets short-drama and scene-level generation, and establishes a high-level failure taxonomy spanning acting, narrative, atmosphere, and audio-visual language. Based on this taxonomy, we construct more than 10,000 question-answering evaluation instances, together with subsets for short-drama-level assessment and temporal localization of failure modes, to systematically evaluate the ability of omni large language models to diagnose high-level audio-visual failures. Experimental results show that commercial omni models such as Gemini substantially outperform other evaluators, yet even the strongest models continue to struggle with complex failures in our benchmark. These results demonstrate that MTAVG-Bench 2.0 provides a systematic benchmark for failure diagnosis in cinematic multi-talker audio-video generation. 

---
# Clark Hash: Stateless Sparse Johnson-Lindenstrauss Quantization for Neural Embeddings 

**Authors**: Stanislav Kirdey, Clark Labs Inc  

**Link**: [PDF](https://arxiv.org/pdf/2605.28034)  

**Abstract**: Clark Hash is a small method for storing neural embeddings in less space. It normalizes each database vector, applies a deterministic sparse signed Johnson-Lindenstrauss projection, clips the result, and stores a fixed-width scalar-quantized code. Queries stay in floating point and are scored against the stored sketches. In the default 384-dimensional sentence-embedding setting, Clark Hash stores a cosine-search vector in 48 bytes instead of 1536 bytes for dense f32 storage. This is 32x smaller. The method does not need a training pass, learned codebooks, rotations, or corpus statistics before new vectors can be stored. We describe the codec, the Rust implementation, and a multilingual sentence-similarity evaluation on 9,304 labeled pairs from 29 subsets. With a multilingual MiniLM encoder, the 48-byte sketches reached 0.910 and 0.946 macro Pearson correlation with dense cosine scores on STS17 and STS22. Clark Hash is not a new Johnson-Lindenstrauss theorem and it is not a replacement for approximate nearest-neighbor indexes. It is a simple stateless codec for compact embedding storage. 

---
# PetroBench: A Benchmark for Large Language Models in Petroleum Engineering 

**Authors**: Xiang Wang, Tingting Zhang, Sen Wang, Ying Wu, Heng Meng, Peng Zhou, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.28032)  

**Abstract**: Large Language Models are increasingly applied in the petroleum industry, highlighting the need for a domain-specific evaluation framework. This study develops a benchmark for LLMs in petroleum engineering, including a three-stage process of data preprocessing, quality filtering, and multi-model validation. Using expert review, a standardized question bank with strong domain relevance and discriminative capability was constructed. The benchmark covers production, reservoir, and drilling engineering, with 1,200 questions across multiple-choice, true or false, term definition, and short-answer formats. Eight mainstream LLMs were evaluated under a unified API environment. Results show that models performed better on subjective than objective questions, indicating weaknesses in factual knowledge discrimination. The highest accuracies for multiple-choice and true or false questions were 65.3% and 74.3%, respectively. Gemini-3-Pro, Kimi-K2.5, and Claude-Opus-4.6-Thinking achieved the best overall scores of 72%-74%. Models performed best in production engineering and weakest in reservoir engineering. Chinese models showed advantages in multiple-choice questions, while international models performed slightly better in short-answer questions. The benchmark provides a reproducible and practical reference for evaluating and deploying LLMs in petroleum engineering. 

---
# MIRA: A Bilingual Benchmark for Medical Information Response Audit 

**Authors**: Mengyu Xu, Qiaoxin Yang, Qianqian Wang, Xiwei Dai, Weiyi Wu, Chongyang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.28025)  

**Abstract**: Large language models (LLMs) are increasingly used to provide public-facing health information, yet existing safety evaluations overlook whether responses preserve comparable medical information across different user phrasings of the same question. To address this, we introduce the Medical Information Response Audit (MIRA), a bilingual, controlled benchmark that assesses whether LLMs provide comparable medical information across user-side language, register, and health literacy signals. MIRA contains 4,320 prompts built from 60 medically reviewed, low-risk health questions. Across five mainstream LLMs, models answered all medical questions, but responses to low health-literacy signals consistently omitted more key information, provided fewer concrete next steps, and offered less support for independent judgment. We term this pattern Differential Information Dilution (DID). Language effects are model-specific rather than uniformly worse for non-English prompts. A comparison with 300 real-world health queries provides preliminary evidence of rank-order validity. A knowledge-guided mitigation prompt reduces information dilution for most models, with the largest reductions in underinformative simplification observed for Claude (~8%) and Qwen (~6%). 

---
# Confidence-Orchestrated Self-Evolution against Uncertain LLM Feedback 

**Authors**: Bowen Wei, Nan Wang, Yuqing Zhou, Jinhao Pan, Ziwei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28010)  

**Abstract**: Self-evolving large language models (LLMs) learn by generating their own training tasks and solutions, reducing reliance on human-curated supervision. However, in many reasoning domains, the model must also validate generated tasks and judge generated answers to obtain training signals. This creates a training-signal challenge: erroneous self-judgments become erroneous gradient updates. Existing approaches either rely on external verifiers, which limits generality, or treat noisy self-generated feedback as supervision. We propose COSE (Confidence-Orchestrated Self-Evolution), which uses the LLM's intrinsic confidence as a lightweight uncertainty signal to modulate learning. COSE introduces confidence-weighted PPO updates and confidence-prioritized replay. Across 19 held-out benchmarks and four Qwen/Llama backbones (0.6B--4B), COSE consistently improves over base models and achieves the best average performance in general reasoning and mathematics, while remaining competitive on code. Code and data are available at this https URL. 

---
# Zipping the Thought: When and How Compressed Reasoning Data Works in LLM Post-Training 

**Authors**: Kohsei Matsutani, Gouki Minegishi, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2605.28008)  

**Abstract**: Large language models (LLMs) can now solve complex problems through long chain-of-thought (CoT) reasoning, but the trade-off between performance and token cost remains a central challenge. To address this issue, supervised fine-tuning (SFT) often uses compressed reasoning data, where CoT traces are shortened into compact forms. However, the effect of such compressed reasoning data on post-training remains poorly understood. In this paper, we propose a taxonomy of CoT consisting of Explicit CoT, which outputs all operations without aggregation, Composed CoT, which combines multiple operations into a single step, and Implicit CoT, which omits intermediate operations. We construct a synthetic compositional reasoning task that allows controlled variation of difficulty, compression granularity, and data size, and conducted a comprehensive set of experiments across different model families and sizes. Notably, we find that (i) coarser CoT requires more SFT data, (ii) compared with Explicit CoT, Composed CoT and Implicit CoT benefit more from data scaling, while Composed CoT benefits from data repetition and Implicit CoT tends to lead to memorization, (iii) unlike SFT, subsequent reinforcement learning (RL) with verifiable rewards (RLVR) decomposes compressed steps learned during SFT, and (iv) unidirectional CoT ordering shows stronger generalization on longer sequential tasks. Our findings provide implications for CoT design under data resource constraints and offer important insights into the mechanisms of SFT and RL in LLM post-training. 

---
# An Empirical Audit of k-NAF Budget Accounting for Anchored Decoding 

**Authors**: J. Vijayavallabh  

**Link**: [PDF](https://arxiv.org/pdf/2605.28001)  

**Abstract**: We empirically audit the k-NAF budget-accounting mechanism in Anchored Decoding using (i) a fixed, class-stratified workload (approximately 8,500 randomized executions across six prompt classes) and (ii) an adaptive prompt-search procedure targeting high proxy spend ratios. On the fixed workload, mean cumulative KL spend remains far below the sequence-level budgets K in {600, 1000}, and an empirical Bernstein-style proxy stays below K for every class; surface-overlap diagnostics (ROUGE-L and 5-gram Jaccard) are correspondingly small. Adaptive search increases the proxy spend ratio but does not produce clear budget exhaustion. On a held-out copyright-domain workload at k = 3, several prompts exhibit proxy ratios above 1 under early-stopped evaluations with small realized sample sizes; re-evaluating the same prompts with larger allocation reduces the proxy ratio to the range [0.26, 0.40] under comparable mean spend, consistent with proxy artifacts rather than per-trajectory budget failures. 

---
# Reward Bias Substitution: Single-Axis Bias Mitigations Redirect Optimization Pressure 

**Authors**: Max Lamparth, Daniel Fein, Andreas Haupt, Marcel Hussing, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2605.27996)  

**Abstract**: Single-axis mitigations of reward-model biases (e.g., reducing proxy reliance on length, sycophancy, or style) can rotate optimization pressure onto correlated proxies rather than eliminate it, a failure mode we call reward bias substitution. The failure is enabled by a measurement-versus-optimization gap between audit and policy-induced distributions during mitigation evaluation and policy training. We formalize mitigation outcomes into a regime taxonomy and prove that successful mitigation, bias substitution, and overcorrection produce identical observables under any audit-distribution scoring, including ranking accuracy and win-rate, even when granted oracle access to the true reward. Across published preference-learning mitigation work, no method we survey reports the evidence needed to certify successful mitigation. Augmenting evaluation with policy-induced distributions while tracking multiple biases provably closes the gap, and we translate this into actionable prescriptions for mitigation methods and benchmarks. We demonstrate bias substitution in language model RLHF, where a length penalty during GRPO training compresses responses as intended yet redirects optimization pressure onto confidence calibration, driving the policy into overconfidence while factual free-form accuracy falls. We also show a published length-debiasing operator that zeroes reward-length correlation on the audit distribution but reintroduces bias under best-of-N selection on three of four SOTA reward models, and a length-sycophancy coupling whose direction reverses under human-LLM judge disagreement. 

---
# AsyncTool: Evaluating the Asynchronous Function Calling Capability under Multi-Task Scenarios 

**Authors**: Kou Shi, Ziao Zhang, Shiting Huang, Avery Nie, Zhen Fang, Qiuchen Wang, Lin Chen, Huaian Chen, Zehui Chen, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.27995)  

**Abstract**: Large language model (LLM)-based agents have shown strong capabilities in using external tools to solve complex tasks. However, existing evaluations often overlook the temporal dimension of tool use, especially the impact of tool response latency, and are usually limited to single-task settings. In real-world applications, multiple tasks often need to be executed concurrently, and overall efficiency depends on whether an agent can use idle time while waiting for tool responses. We refer to this capability as asynchronous tool calling. To evaluate it, we propose AsyncTool, a benchmark for assessing LLM-based agents in interactive multi-task tool-use environments with delayed tool feedback. AsyncTool presents multiple heterogeneous tasks simultaneously and simulates realistic tool response latency during execution. Using a hybrid data evolution strategy, we construct a diverse asynchronous multitasking dataset that covers multiple scenarios and tool-use patterns. We evaluate models at the step, sub-task, and task levels, and introduce efficiency-oriented metrics to measure task coordination and completion efficiency. Extensive experiments show that delayed tool feedback poses substantial challenges to current agents and leads to clear performance degradation. Models that better coordinate task switching, dependency tracking, and state maintenance achieve stronger performance on AsyncTool. Our analysis identifies key failure modes of current tool-using agents and provides practical insights for designing future systems with stronger temporal reasoning and coordination capabilities. 

---
# STAB: Specification-driven Testing for Algorithmic Bottlenecks 

**Authors**: Soohan Lim, Joonghyuk Hahn, Hyundong Jin, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.27981)  

**Abstract**: Evaluating the efficiency of algorithmic code requires test cases that expose runtime bottlenecks. Previous methods generate efficiency test cases either by increasing input size or by generating code-specific inputs that make the given implementation run slowly. Consequently, they do not address the structural input conditions that drive the algorithmic worst case. We introduce STAB, a specification-driven pipeline that generates test cases that expose algorithmic bottlenecks from a natural-language problem specification alone. STAB separates the task into constraint-bound maximization and adversarial structure injection. (i) The constraint saturator extracts constraints and resolves large admissible size assignments using rule-based saturation and CP-SAT optimization over related variables. (ii) The adversarial scenario injector retrieves implementation-level adversarial construction principles from a curated scenario catalog using keyword matching and K-nearest neighbors (KNN). STAB encodes the problem specification, resolved boundary, and retrieved construction principles into a structured generation specification, from which the LLM synthesizes a Python test case generator. On CodeContests, STAB raises the rate of generated test cases that expose algorithmic bottlenecks from 50.43% to 73.45% on average across open-source LLMs and from 57.45% to 71.85% on average across closed-source LLMs, with consistent gains across Python, Java, and C++. Our code is available at this https URL. 

---
# Geometry of Human Perceptual Domains Emerges Transiently in LLM Representations 

**Authors**: Simardeep Singh, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2605.27970)  

**Abstract**: While large language models (LLMs) are trained purely on textual data, prior work has shown that their internal representations can exhibit rich geometric structure in embedding space. Building on this line of work, we investigate whether such structure is similar to human perceptual organisation across different domains (e.g., color, pitch, emotion, and taste). Specifically, we study the layer-wise emergence of intrinsic geometrical structure corresponding to perceptual modalities within the residual streams of multiple open-weight transformer architectures. Our results reveal three key findings. First, we observe the emergence of layer-wise geometric structure across multiple perceptual domains, despite the absence of any direct perceptual supervision during training. Second, these perceptual domains exhibit distinct emergence profiles, with both geometric structure and its alignment with human baselines following domain- and model-specific trajectories across depth. Third, this emergence follows a consistent representational trajectory: geometry is weak or diffuse in early layers, becomes progressively organised in intermediate layers, and is attenuated in later layers, suggesting that perceptual geometry arises transiently as part of the model's internal transformation pipeline. This provides new insight into how and where human-like perceptual geometry arises in LLMs, offering a principled pathway for mechanistic analysis of internal representations. 

---
# The Shape of Overthinking: Backtracking Bursts in Long Reasoning Traces 

**Authors**: Navid Rezazadeh, Arash Gholami Davoodi  

**Link**: [PDF](https://arxiv.org/pdf/2605.27965)  

**Abstract**: Reasoning models often generate long traces in which useful self-correction and unproductive revision are hard to distinguish. We study this distinction through backtracking dynamics: local reconsideration, retraction, or re-derivation inside long-form reasoning traces. On 6{,}000 Qwen3-8B AIME traces, we annotate segment-level backtrack severity and analyze event timing, normalized depth, and local burst structure. We find that early isolated repair is often compatible with correct reasoning, whereas incorrect traces more often show moderate-to-severe backtracks that persist and cluster late. Cross-corpus checks show the same qualitative asymmetry across additional model/domain pairs. Filtering analyses instantiate the signal as a prefix-causal selective early-exit policy: at shallow and intermediate depths, burst-aware filtering outperforms fixed length-based filtering while using only prefix-available features. Moderate length cutoffs remain strong completed-trace baselines, but burst-aware control provides a deployable mechanism for separating recoverable repair from likely instability. 

---
# From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection 

**Authors**: Ke Liu, Jiwei Wei, Wenyu Zhang, Shuchang Zhou, Ruikun Chai, Yutao Dai, Chaoning Zhang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27944)  

**Abstract**: With rapid advances in audio-visual generative models, reliable forgery detection becomes increasingly critical. Existing methods for audio-visual deepfake detection typically rely on cross-modal inconsistencies. In singing, rhythmic vocalization weakens this coupling and introduces a nontrivial domain shift, substantially degrading detection performance. We construct the Singing Head DeepFake (SHDF) dataset using rhythm-aware generative models to fill the gap in singing benchmarks. To cope with cross-scenario domain shifts, we propose a Text-guided Audio-Visual Forgery Detection (T-AVFD) framework that generalizes across both talking and singing scenarios. T-AVFD comprises a facial authenticity pattern learner and a multi-modal differential weight learning module. The pattern learner aligns facial features with multi-granularity textual descriptions to learn generalizable authenticity patterns. The weight learning module preserves intrinsic audio-visual consistency and adaptively integrates it with authenticity patterns via differential weighting. Extensive experiments on multiple talking head deepfake datasets and SHDF show consistent improvements over existing baselines and strong robustness under diverse perturbations. 

---
# Do Agents Think Deeper? A Mechanistic Investigation of Layer-Wise Dynamics in Sequential Planning 

**Authors**: Zhenyu Cui, Xiangzhong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.27935)  

**Abstract**: Recent mechanistic studies suggest that large language models (LLMs) may utilize their depth inefficiently in standard single-turn tasks. Whether this still holds in autonomous agent settings, where models must perform multi-turn planning, tool use, and iterative state updates, remains unclear. We study this question through a systematic layer-wise analysis of complete user-agent trajectories spanning three domains: Deep Research, Code Generation, and Tabular Processing. Using residual stream probes, causal layer-skipping interventions, and effective-depth measurements, we show that agentic reasoning exhibits a distinct depth profile from static tasks. As trajectories unfold, models progressively recruit more and deeper layers, with stronger long-range inter-layer dependencies emerging in later turns. At the same time, residual updates become increasingly correction-dominant, indicating a shift from stable feature accumulation toward repeated recalibration. Effective-depth analysis further reveals a substantial construction-refinement gap: semantic direction often forms relatively early, while deep layers remain necessary for stabilizing final outputs. Across model families, this gap is pronounced in Qwen and Minimax, whereas GLM shows a more domain-dependent depth allocation pattern. These results provide mechanistic evidence that autonomous LLM agents allocate depth adaptively as reasoning complexity grows. 

---
# DiagramRAG: A Lightweight Framework to Retrieve Scientific Diagram for Figure Generation 

**Authors**: Xinjiang Yu, Junyi Han, Zhuofan Chen, Chi Zhang, Xiangyu Fu, Jingyuan Tan, Zirui You, Yixiang Jian, Yu-Ping Wang, Chengliang Chai  

**Link**: [PDF](https://arxiv.org/pdf/2605.27931)  

**Abstract**: Scientific diagrams are essential for communicating complex methodologies in academic papers. A natural way for researchers to specify such diagrams is through rough sketches, where text labels, connectors, and spatial arrangements express early semantic and topological intentions. However, sketches are usually incomplete, making them insufficient for directly producing publication-quality diagrams. Existing sketch-based generation methods mainly reconstruct the sketch itself, while recent text-driven diagram generation frameworks rely on textual semantics and do not fully exploit the topological structure contained in sketches. In this paper, we introduce DiagramRAG, a lightweight retrieval-augmented framework for sketch-based scientific diagram completion. Given a user sketch, DiagramRAG retrieves reference diagrams that are both semantically relevant to the sketch content and topologically compatible with its structure, and uses them to guide downstream diagram generation. To enable efficient structure-aware retrieval, we represent diagrams as knowledge graphs, synthesize sketch variants at different simplification levels, and train an embedding model to align sketches with compatible diagrams in a shared space. The retrieved references further provide content, topology, and visual priors for completing and rendering the final diagram. Experiments show that DiagramRAG achieves F1-scores of 0.848 and 0.802 on DiagramBank and FigureBench, respectively, and improves generation quality with the best VLM-as-a-Judge score of 7.170, while reducing inference latency to 35.48 seconds per sample. Our code and data are available at this https URL and this https URL. 

---
# Harness-Bench: Measuring Harness Effects across Models in Realistic Agent Workflows 

**Authors**: Yilun Yao, Xinyu Tan, Chao-Hsuan Liu, Yaoming Li, Zhengyang Wang, Wenhan Yu, Zhewen Tan, Yuxuan Tian, Guangxiang Zhao, Lin Sun, Xiangzheng Zhang, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27922)  

**Abstract**: LLM agents are increasingly deployed as executable systems that use tools, modify workspaces, and produce concrete artifacts. In such workflows, performance depends not only on the base model, but also on the harness: the system layer that manages context, tools, state, constraints, permissions, tracing, and recovery. However, existing benchmarks typically abstract away execution, compare complete agent systems, or hold the harness fixed, making execution-layer variation difficult to study. We introduce Harness-Bench, a diagnostic benchmark for evaluating configuration-level harness effects in realistic agent workflows. Harness-Bench evaluates representative harness configurations across multiple model backends under shared task environments, budgets, and evaluation protocols, while preserving each harness's native execution behavior. The benchmark contains 106 sandboxed offline tasks constructed from practical agent-use patterns and manually reviewed for realism, solvability, oracle-checkability, and integrity. Each run records final artifacts, execution traces, usage statistics, and validator outputs, enabling analysis beyond final completion. Across 5,194 execution trajectories, we observe substantial variation in completion, process quality, efficiency, and failure behavior across model-harness pairings. These results suggest that agent capability should be reported at the model-harness configuration level rather than attributed to the base model alone. Our analysis further identifies recurring execution-alignment failures, where plausible reasoning becomes decoupled from tool feedback, workspace state, evidence, or verifiable output contracts. Harness-Bench provides a reproducible foundation for diagnosing and improving reliable, efficient, and auditable agent execution stacks. 

---
# Show, Don't TELL: Explainable AI-Generated Text Detection 

**Authors**: Aldan Creo, Suraj Ranganath  

**Link**: [PDF](https://arxiv.org/pdf/2605.27921)  

**Abstract**: Research on AI-generated text detection has presented a number of approaches to discern human from AI prose, some of which achieving high in-distribution performance. However, real-world applicability has stalled because their outputs are misaligned with the needs of users, such as professors, who are presented with a numeric score that has no attached explanation. We tackle this issue with a novel architecture, TELL, that bakes explainability from the ground-up. While our system still offers a numerical score like other detectors for comparability, TELL takes a fundamentally different approach where we aim to show the user the "tells" by which the model believes a text is AI or human-written, to empower the user to decide who wrote a text using their own judgment and understanding of the context of the writing and its alleged author. We train TELL on a custom SFT dataset of domain-specific authorship annotations, and further refine the system using GRPO with curriculum learning to improve performance. We achieve competitive performance with state-of-the-art detectors (AUROC 0.927) while natively providing annotations that explain the basis for the detector's decision. We further evaluate the quality of our explanations using a dataset of human annotations and report a high (mean 72.3%) win-rate on annotation concreteness, falsifiability, coherence, plausibility and grounding, allowing users to critically think and decide for themselves. Our work thus reframes the problem of AI-generated text detection in a human-centric perspective and paves the way for a new family of detectors that focus on native explainability. 

---
# SuiChat-CN: Benchmarking Contextual Suicide Risk Assessment in Chinese Group Chats 

**Authors**: Xiangyu Wang, Zhiwei Yu, Chengze Du, Dingchang Wang, Yuhan Ye, Fangyu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.27911)  

**Abstract**: Suicide is a critical global public health challenge, causing approximately 720,000 deaths each year and calling for timely, effective prevention strategies. Existing computational studies primarily focus on post-based social media platforms such as Twitter and Weibo, leaving instant messaging environments such as Telegram underexplored. Yet group chats pose distinct challenges: messages are short, fragmented, multi-party, and often rely on implicit or culturally specific expressions, making isolated post-level analysis insufficient. We introduce SuiChat-CN, a Chinese group-chat benchmark for contextual suicide risk assessment. We collect public Telegram group-chat data, construct coherent conversational segments through signal-word extraction and bidirectional context expansion, and annotate user risk levels with an expert-validated, LLM-assisted paradigm. SuiChat-CN contains 13,312 contextual segments from 1,406 users, covering 258,228 raw chat messages. Extensive experiments with PLMs and more than 40 LLMs demonstrate that contextual information is essential for reliable risk assessment, while fine-tuning and partial-context evaluation further reveal the challenges of early detection in multi-party conversations. Due to ethical and sensitivity concerns, the dataset is not publicly released but will be shared with accredited mental health and suicide-prevention research institutions upon reasonable request. 

---
# Reasoning Matters: Mitigate Hallucination in Multimodal Large Reasoning Models via Reasoning-Conditioned Preference Optimization 

**Authors**: Jiawei Kong, Hao Fang, Shunxiang Liao, Jinyu Li, Bin Chen, Hao Wu, Shu-Tao Xia, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27906)  

**Abstract**: Multimodal Large Reasoning Models introduce the reasoning paradigm, demonstrating strong capabilities on complex vision-language tasks. However, they still suffer from severe hallucinations. Existing training-based methods typically mitigate hallucinations through response-level direct preference optimization (DPO), where the Chain-of-Thought (CoT) and the final answer are treated as a monolithic output and optimized jointly. We reveal that this formulation performs similarly to answer-only optimization, suggesting that it primarily learns answer-level preference, while leaving CoT-level supervision insufficiently exploited. To address this issue, we explicitly formulate a CoT-oriented preference term and derive Reasoning-Conditioned Direct Preference Optimization (RC-DPO), which models the CoT as a condition for answer generation and contrasts the preference for the same preferred answer under different CoT conditions, promoting answer-supportive reasoning chain alignment. To further improve optimization, we introduce a reasoning-enhanced preference data generation strategy that employs Monte Carlo Tree Search to discover visually grounded and logically consistent CoTs as positive samples, and attention-guided CoT token pruning to construct negative ones. Extensive experiments across various models and benchmarks show that RC-DPO effectively mitigates hallucinations and improves the reliability of the multimodal reasoning process. 

---
# Dr-CiK: A Testbed for Foresight-Driven Agents 

**Authors**: Yihong Tang, Andrew Robert Williams, Arjun Ashok, Vincent Zhihao Zheng, Lijun Sun, Alexandre Drouin, Issam H. Laradji, Étienne Marcotte, Valentina Zantedeschi  

**Link**: [PDF](https://arxiv.org/pdf/2605.27904)  

**Abstract**: Time series forecasting in real-world settings often depends not only on historical observations, but also on external context that must be actively discovered from noisy, heterogeneous information sources. Yet existing context-aided forecasting benchmarks typically assume that the supporting context is already provided, leaving open whether agents can identify it on their own. Therefore, we introduce Dr-CiK, a benchmark for evaluating whether agents can retrieve forecasting-relevant supporting context from a document corpus, filter out distractors, distill the retrieved context into forecast-useful evidence, and generate forecasts supported by that evidence. Through context ablations and evaluations of state-of-the-art deep research and forecasting methods paired together, we show that high-quality context substantially improves forecasting performance in Dr-CiK. However, most existing DR agents recover only a small fraction of the ground-truth supporting evidence (usually <5%), are frequently misled by distractors (>80% distractor citations), and can cause forecasters to perform worse with retrieved context than without context. Our results motivate research on foresight-driven agents that search for the right context to predict the future. 

---
# SKILLC: Learning Autonomous Skill Internalization in LLM Agents via Contrastive Credit Assignment 

**Authors**: Hongxiang Lin, Zhirui Kuai, Erpeng Xue, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27899)  

**Abstract**: Structured skill prompts improve exploration in long-horizon agentic reinforcement learning (RL). Skill-augmented RL methods retain external skills at inference, while skill-internalization RL methods withdraw them during training to enable autonomous performance. However, existing internalization approaches only use skill-helpfulness contrast for curriculum control, leaving the policy update unchanged and unable to distinguish skill-dependent from autonomous success. We propose SkillC, a framework based on Contrastive Skill Credit Assignment (CSCA) that converts this contrast into a direct learning signal for internalization. \textsc{SkillC} samples paired skill-injected and skill-free rollouts for tasks from active skill types within the same policy update, and injects their task-level contrast into optimization via a dual-stream advantage estimator that preserves global ranking while applying a one-sided correction toward skill-free success. A smoothed validation-level signal further drives an adaptive curriculum over attribution strength, rollout allocation, and monotonic active-set pruning. Experiments on ALFWorld and WebShop show that, without runtime skill access, SkillC surpasses the strongest prior skill-internalization RL baseline by 5.5\% and 4.4\%, respectively, while remaining competitive with skill-augmented RL methods. 

---
# A Unified Framework for the Evaluation of LLM Agentic Capabilities 

**Authors**: Pengyu Zhu, Lijun Li, Yaxing Lyu, Qianxin Luo, Jingyi Yang, Yi Liu, Tingfeng Hui, Xinyu Yuan, Li Sun, Sen Su, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2605.27898)  

**Abstract**: As LLMs are increasingly deployed as agents, reliable assessment of their agentic capabilities has become essential. However, reported benchmark scores often jointly reflect model capability and the implementation choices each benchmark is packaged with, making cross-benchmark results difficult to interpret as clean measurements of the underlying model. In this work, we present a unified framework for the fair evaluation of LLM agentic capabilities. Driven by a unified configuration system, the framework integrates diverse benchmarks into a standardized instruction--tool--environment format, executes agents through a fixed ReAct-style architecture within a controllable sandbox, and provides an optional offline setting that replaces volatile live environments with curated snapshots, so that framework effects and environment effects can be analyzed separately. Building on this, we unify the evaluation methodology under each benchmark's original task-success criteria, while introducing unified metrics for resource consumption and a taxonomy for decision- and execution-level failure attribution. Within this framework, we adapt 7 widely used benchmarks spanning 24 domains across single-agent, multi-agent, and safety-critical scenarios, and conduct a large-scale empirical analysis over 400K rollouts and 5B tokens on 15 models. The results show that scaffold choice and environmental volatility materially shift benchmark outcomes in both directions, allowing our framework to disentangle intrinsic LLM capabilities from framework- and environment-induced artifacts. We further demonstrate its extensibility as a secure testbed for safety-critical domains. Codes and benchmarks at are available at this https URL, this https URL. 

---
# PortBench: A Correlation-Aware, Full-Pipeline Benchmark for LLM-Driven Portfolio Management 

**Authors**: Yuxuan Zhao, Sijia Chen, Ningxin Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.27887)  

**Abstract**: LLMs have shown strong performance across diverse financial tasks, yet portfolio management (PM), a critical financial decision-making task, remains poorly benchmarked. Existing benchmarks exhibit two main gaps: they ignore cross-asset correlation structures, thereby failing to distinguish genuinely diversified portfolios from concentrated ones, and fail to evaluate the complete PM decision pipeline in real-world scenarios. We introduce PortBench, a benchmark spanning six heterogeneous asset classes over ten years. PortBench consists of two complementary layers: a static QA dataset of 6,269 correlation-based questions across seven task templates, and a dynamic five-stage allocation pipeline that mirrors the full PM decision cycle. To evaluate these layers, we introduce two dedicated metrics: a dual-layer correlation score that measures whether proposed portfolios exploit inter-class hedging and avoid intra-class concentration, and CEPS, a metric that quantifies how reasoning errors compound across pipeline stages. We further assess strategy robustness and investor alignment under three historical stress regimes and risk profiles. Evaluating ten frontier LLMs, we find that despite strong performance on static financial QA, 90\% of model-profile combinations fail to outperform a basic equal-weight allocation, and models that satisfy every procedural constraint still suffer catastrophic drawdowns under stress. Our source code is available at \href{this https URL}{this https URL}. 

---
# Towards Faithful Agentic XAI: A Verification Method and an Open-World Benchmark for Better Model Faithfulness 

**Authors**: Jaechang Kim, Sunung Mun, Seungjoon Lee, Jaewoong Cho, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2605.27879)  

**Abstract**: Explainable AI (XAI) helps users interpret model behavior and identify potential faults. Agentic XAI systems use Large Language Models (LLMs) to make explanations more accessible through natural-language interaction, but they can also produce plausible yet unfaithful explanations. This risk arises because unreliable XAI outputs for complex models can be amplified by LLMs and mislead users. We propose Faithful Agentic XAI (FAX), a framework that improves explanation faithfulness through explicit verification. FAX decomposes draft explanations into claims and cross-checks them against inherently faithful tools, filtering unsupported or contradictory claims before final generation. We also introduce CRAFTER-XAI-Bench, an open-world reinforcement learning benchmark with complex policies, diverse goals, and challenging scenarios for assessing model-specific faithfulness. On CRAFTER-XAI-Bench, FAX improves simulation faithfulness from 0.20 for the strongest baseline to 0.46 while maintaining high informativeness, relevance, and fluency. On three tabular benchmarks, FAX performs competitively with prior Agentic XAI baselines, but our analysis shows that these settings can conflate task accuracy with model-specific faithfulness. These findings show that explicit verification is essential for faithful Agentic XAI and that that faithfulness benchmarks must be designed to test explanations against the behavior of the target model itself. 

---
# AIBuildAI-2: A Knowledge-Enhanced Agent for Automatically Building AI Models 

**Authors**: Ruiyi Zhang, Peijia Qin, Qi Cao, Li Zhang, Pengtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.27873)  

**Abstract**: AI models underpin data-centric applications from image and text processing to scientific discovery in biology, physics, and chemistry. Yet developing them remains heavily manual, requiring practitioners to design architectures, build training pipelines, and iteratively refine solutions, making it challenging for natural scientists without specialized AI engineering expertise to build the high-performing models their research demands. To reduce this burden and broaden access to AI for scientific discovery, agents that automatically build AI models have been proposed. However, the performance of these agents is largely limited by the parametric knowledge of their underlying large language models, which is static, often outdated, and sparse on practical AI model engineering know-how. To address this limitation, we introduce AIBuildAI-2, a knowledge-enhanced agent with an external, evolving knowledge system for automatically building AI models. The knowledge system of AIBuildAI-2 is hierarchical, organizing curated AI development knowledge into high-level knowledge instructions over topical categories and low-level knowledge documents under each category, from which the agent dynamically loads only the context relevant to its current state and the AI task being solved, grounding each design and implementation decision in concrete, externally verifiable expertise. The system is initialized by collecting and cleaning AI-development-related documents from the web and organizing them into the corresponding categories, and continually evolves from the agent's own experience by distilling each completed run on an AI task into structured takeaways that are written back into the knowledge system. AIBuildAI-2 achieves state-of-the-art results, ranking first on MLE-Bench with a 70.7% medal rate and placing in the top 6.6% among 4,370 human-expert teams in a heart disease prediction competition. 

---
# FundaPod: A Multi-Persona Agent Pod Platform with Knowledge Graph Memory for AI-Assisted Fundamental Investment Research 

**Authors**: Di Zhu, Zheng, Zihan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.27864)  

**Abstract**: Large language models (LLMs) are increasingly applied in finance, yet most existing work emphasizes trading signals or financial NLP tasks centered on prediction. Institutional fundamental research, by contrast, requires human analysts or AI agents to gather evidence, identify business drivers, compare competing viewpoints, and generate investment memos. Its broader goal is not merely to predict outcomes, but to produce investment plans that are transparent, reusable, and verifiable, while contributing to the cumulative development of investment knowledge. We present FundaPod, a multi-persona agent platform for AI-assisted fundamental investment research. We argue that fundamental research is a human-centric decision-support task that is qualitatively distinct from trading-signal generation, and is therefore better served by an independence-preserving architecture. In FundaPod, AI agents with different personas, such as value investors or macro strategists, conduct research independently under a shared provenance contract. Their disagreements are then surfaced post hoc for adjudication by the human portfolio manager (PM) through a knowledge-graph memory system. This paper contributes five design principles for human-AI hybrid systems supporting fundamental research, grounded in design-science practice and theories of cognitive isolation and human-machine coordination. It also describes four architectural mechanisms: a persona distillation pipeline that turns public investor materials into deployable agents; a declarative skill registry that lets the planner derive typed task graphs; a grounded evidence model that links memo claims to verifiable sources; and a knowledge-graph "second brain" that connects tickers, memos, analysts, and themes. We demonstrate the architecture through a complete case study and a persona-based memo comparison. 

---
# C-MIG: Multi-view Information Gain-based Retrieval-Augmented Generation for Clinical Diagnosis Reasoning 

**Authors**: Yuwei Miao, Gen Li, Yunsheng Zeng, Xiandong Li, Yujin Wang, Siyu Chen, Luning Wang, Yunhao Qiao, Junfeng Wang, Jianwei Lv, Bo Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2605.27860)  

**Abstract**: Retrieval-augmented generation combined with reinforcement learning has shown promise for grounding large language models in trustworthy medical evidence. However, existing methods rely on exact-match binary rewards, which in clinical diagnosis cause two issues: (i) semantically relevant but non-verbatim steps receive zero signal, discarding valuable learning signals; and (ii) uni-dimensional rewards cannot effectively supervise heterogeneous reasoning capabilities. To address these issues, we propose C-MIG, a Multi-view Information Gain-based retrieval-augmented generation framework for Clinical diagnosis. C-MIG estimates information gain under a frozen reference model from two complementary views, retrieved-document and document-refinement, to jointly guide what to retrieve and how to refine, alleviating the issues of valuable reward signal loss and credit assignment. We further design a multi-subquery retrieval augmentation strategy that improves knowledge recall coverage in clinical diagnostic scenarios. Comprehensive experiments on four medical benchmarks demonstrate that C-MIG achieves the best performance among all RAG-RL methods on both in-domain and out-of-domain sets, and outperforms state-of-the-art general-purpose LLMs for clinical diagnosis. 

---
# MolLingo: Molecule-Native Representations for LLM-Powered Scientific Agents 

**Authors**: Thao Nguyen, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.27853)  

**Abstract**: We present MolLingo, a multi-agent system that emulates the reasoning process of a chemist to automate molecular design. Existing LLM-based approaches either operate as standalone generative models without access to external tools or lack the multi-agent coordination and shared memory needed for iterative, evidence-driven reasoning across the molecular design pipeline. MolLingo addresses this by coordinating a Literature Agent, a Chemist Agent, and an Orchestrator through a shared memory module, with each agent equipped with domain-specific tools. To enable effective molecular reasoning, we introduce BRICS-based Fragment Enumeration (BFE), a synthesis-aware molecular fragmentation method that decomposes molecules into chemically meaningful building blocks represented as block-based SMILES paired with common chemical names. This representation bridges molecular structure and LLM semantic space, enabling block-level reasoning and editing that is difficult with raw SMILES alone. As a case study in early-stage therapeutic design, MolLingo further grounds the Chemist Agent's reasoning in binding site geometry and residue-level protein context derived from molecular docking to optimize molecules for stronger target binding. Across four benchmarks, MolLingo consistently outperforms frontier LLMs and specialized baselines, including a fourfold docking score improvement over GPT-5.4 despite using the same underlying model, consistent drug property optimization gains across multiple LLM backbones, and state-of-the-art results on TOMG-Bench, surpassing both frontier LLMs and the RL-based optimization method RePO. Our results suggest that LLMs are already capable molecular design assistants when guided through chemically meaningful representations and biologically grounded structural context. Code is available at: this https URL. 

---
# When Context Flips, Safety Breaks: Diagnosing Brittle Safety in Aligned Language Models 

**Authors**: Dasol Choi, Alex Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2605.27851)  

**Abstract**: Safety benchmark scores provide incomplete evidence of deployment readiness: aligned language models often adhere to rigid rules even when a situational update flips which action is safe. We term this failure brittle safety. To diagnose it, we introduce context-flip evaluation, testing 12 models across a safety benchmark (PacifAIst) and two commonsense controls using paired variants where the nominally safe action produces harm. Three findings emerge. First, brittle safety is safety-specific: all 12 models exhibit a safety-commonsense gap (mean +17.4 pp). Baseline accuracy fails to predict brittleness: among models above 90% baseline accuracy, brittleness rates range from 13.7% to 90.0%. Second, failures stem from policy override rather than miscomprehension: despite acknowledging the context change in every case, models persist via three distinct mechanisms that vary by update type and model family. Third, on a hand-audited probe of catastrophic consequence-flip scenarios, standard action-level guardrails catch none, while a state-aware validator catches all without false alarms on correct interventions. This indicates that action-level content moderation is systematically blind to consequence-flips, motivating state-aware architectural alternatives. We release our protocol, perturbed benchmarks, and deployment probe. 

---
# TCP-MCP: Landscape-Guided Co-Evolution of Prompts and Communication Topologies for Multi-Agent Systems 

**Authors**: Yi Ding, Zijie Xuan, Haowei Zhou, Zhenyu Ju, Xiaoxiao Dong, Jingwen Zhang, Xingyu Zhu, Leixin Sun, Haochi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27850)  

**Abstract**: Effective multi-agent systems cannot be designed by selecting prompts or communication graphs in isolation. Agent behavior depends on the information an agent receives, while the usefulness of a communication edge depends on how the receiving agent interprets and uses that information. We propose \textbf{TCP-MCP} (Topology-Coupled Prompting for Multi-Agent Collaborative Problem-Solving), a co-evolution framework that searches agent prompts and communication topologies as a unified genome. TCP-MCP uses an initialization-time landscape probe to calibrate early search behavior, and then relies on Pareto-front diagnostics to adapt exploration under three objectives: task performance, token cost, and structural complexity. Using the same DeepSeek-V3.2 backbone across all methods, TCP-MCP achieves 82.66\%, 89.96\%, and 96.61\% accuracy on MMLU-Pro, MMLU, and GSM8K, respectively. Across the three benchmarks, it consistently outperforms automated graph-generation baselines and achieves competitive accuracy relative to debate-style systems, while using up to 5.69$\times$ fewer tokens than those systems at the reported operating points. These results show that jointly evolving prompts and communication structure provides a practical route to cost-aware and task-adaptive multi-agent system design in controlled evaluations. 

---
# EAPO: Entropy-Driven Adaptive Positive-Negative Sample Weighting for Policy Optimization in Open-Ended QA 

**Authors**: Yunsheng Zeng, Gen Li, Yuwei Miao, Xiandong Li, Yujin Wang, Siyu Chen, Luning Wang, Yunhao Qiao, Junfeng Wang, Jianwei Lv, Bo Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2605.27846)  

**Abstract**: Large Reasoning Models are typically trained via reinforcement learning from verifiable rewards (RLVR). However, existing approaches adopt fixed weights for positive and negative samples, and the conclusions hardly generalize to open-ended question answering (QA). In this paper, we systematically investigate the roles of positive and negative samples in reinforcement learning for open-ended QA. We propose a reward-mean-based strategy for distinguishing positive from negative samples, and observe that negative samples predominantly govern response diversity and the performance upper bound, whereas positive samples primarily determine response quality and convergence stability. Building on these observations, we propose EAPO, an Entropy-driven Adaptive Policy Optimization method that adaptively computes the weighting coefficients of positive samples based on the ratio of the current policy entropy to the initial entropy. During the entropy-decreasing phase, the weight assigned to positive samples is reduced to preserve exploration, whereas during the entropy-increasing phase it is amplified to reinforce stability, thereby mitigating entropy collapse. Experiments on two publicly available open-ended medical QA datasets demonstrate that EAPO consistently and substantially outperforms fixed-weight baselines in both response diversity and stability. 

---
# Operational AI Deployment Assurance: Governance-State Orchestration Under Threshold-Sensitive Deployment Conditions -- A Governance Framework for High-Stakes AI Systems 

**Authors**: Khalid Adnan Alsayed  

**Link**: [PDF](https://arxiv.org/pdf/2605.27827)  

**Abstract**: AI governance frameworks increasingly emphasize fairness, transparency, accountability, and lifecycle risk management in high-stakes domains. However, many current approaches remain observational, relying on static metric reporting, post-hoc auditing, and monitoring dashboards without directly governing deployment readiness, remediation progression, escalation states, or assurance-driven deployment control. This paper introduces Operational AI Deployment Assurance (OADA), a governance framework for translating fairness disagreement, subgroup instability, threshold sensitivity, remediation outcomes, and operational uncertainty into deployment-oriented assurance decisions. Building on prior work on the Fairness Disagreement Index (FDI) and FairRisk-FDI, OADA reframes governance uncertainty as an operational concern within AI deployment pipelines rather than a byproduct of metric disagreement. The framework introduces Deployment Assurance Scores, Deployment Readiness Classifications, Threshold Stability Zones, Governance Escalation States, and remediation-aware assurance progression. These constructs support lifecycle-oriented governance decisions across high-stakes settings by connecting evaluation outputs to deployment-state interpretation, reassessment, escalation, and operational control. Through deployment-oriented evaluation across facial recognition systems, with discussion extended to healthcare AI as a representative high-stakes domain, the paper demonstrates how systems may appear acceptable under isolated fairness or performance metrics while still exhibiting instability that affects deployment readiness. The proposed framework positions operational deployment assurance as a governance layer between evaluation and real-world AI deployment. 

---
# Revealing Algorithmic Deductive Circuits for Logical Reasoning 

**Authors**: Phuong Minh Nguyen, Tien Huu Dang, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2605.27824)  

**Abstract**: Recent studies have shown that Large Language Models (LLMs) can achieve strong reasoning performance by incorporating functional symbolic representations that abstractly describe graph traversal algorithms and step-by-step reasoning in few-shot learning settings. However, it remains unclear how LLMs genuinely understand the abstract meaning of each reasoning step and the overall algorithm from only a limited number of demonstrations. This work aims to localize the attention heads responsible for individual reasoning steps and characterize the types of information transferred among them. We first align constituent reasoning steps with their corresponding token logits under a symbolic-aided Chain-of-Thought (CoT) prompting framework. Our analysis shows that token positions that steer the reasoning process are associated with low confidence scores caused by constraints on satisfying reasoning behavior patterns in demonstrations. We then adopt causal mediation analysis techniques to identify the attention heads responsible for these patterns. In addition, our findings indicate that LLMs retrieve factual and rule-based information for individual sub-reasoning tasks through specialized attention heads (approximately 3% total heads), whereas higher layers predominantly facilitate information integration and the emergence of global reasoning strategies (e.g., graph traversal algorithms) that coordinate multiple intermediate reasoning steps to solve the overall task. 

---
# EgoBench: An Interactive Egocentric Multimodal Benchmark for Tool-Using Agents 

**Authors**: Yunqi Liu, Tong Niu, Zitong Wang, Zhenlong Dai, Yuqi Qing, Weiqiang Wang, Jian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.27820)  

**Abstract**: As AI agents increasingly operate in open, real-world environments, they require a deep synergy of multimodal perception, tool invocation with multi-hop reasoning, and dynamic interaction with users. However, existing benchmarks fail to jointly evaluate these capabilities due to challenges in designing strictly coupled multi-capability tasks, simulating natural and task-constrained user feedback, and ensuring objective evaluation of dynamic interaction. To bridge this gap, we introduce EgoBench, the first interactive multimodal benchmark for tool-using agents. EgoBench comprises 1,045 egocentric-video-grounded tasks covering four daily scenarios, along with a user-agent-tool interactive environment for evaluation. We implement a three-stage synergistic pipeline through which each task is designed to enforce the joint application of visual perception and tool-augmented multi-hop reasoning. We additionally develop a multi-agent simulated user within EgoBench to evaluate agents' interaction capabilities, which generates high-fidelity, task-aligned responses to agents. Furthermore, we establish a deterministic joint validation framework that guarantees objective assessment through process-based and result-based equivalence. Benchmarking eight SOTA video-MLLM agents on EgoBench reveals a severe performance ceiling: the best model achieves only 30.62% accuracy in the best-performing scenario, averaging 19.43% across all four scenarios. Finally, we conduct a multi-dimensional error analysis to disentangle failure modes, exposing capability bottlenecks for advancing future AI agents. 

---
# Constrained Auto-Bidding via Generative Response Modeling 

**Authors**: Eunseok Yang, Xingdong Zuo, Kyung-Min Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.27811)  

**Abstract**: Auto-bidding systems aim to maximize advertiser value over long horizons under budget constraints and ratio targets such as cost-per-acquisition, yet future traffic and auction dynamics are non-stationary and uncertain. Existing approaches face distinct limitations: control-based pacing reacts to deviations but cannot anticipate future conditions, while RL and generative methods fold constraints into reward signals, obscuring violations and degrading under distribution shift. We shift the learning target from actions to responses with the Generative Response Model (GRM), a history-conditioned sequence model that jointly predicts future traffic volume and horizon-aggregate cost/value curves as functions of a single bid multiplier. We show that under mild monotonicity conditions, the optimality gap relative to full per-tick control is bounded by the dispersion of per-tick marginal value-per-cost. Given predicted responses, a lightweight analytic controller enforces each active constraint via a 1D root-finding step. We prove this controller is exact for the single-multiplier problem and bound constraint violations under receding-horizon replanning in terms of prediction error. Experiments on AuctionNet show that GRM improves constraint stability and overall score compared to existing baselines. 

---
# GraD-IBD: Graph Representation Learning from Diagnosis Trajectories for Early Detection of Inflammatory Bowel Disease 

**Authors**: Leo Y. Li-Han, Ellen L. Larson, Elizabeth B. Habermann, Cornelius A. Thiels, Hojjat Salehinejad  

**Link**: [PDF](https://arxiv.org/pdf/2605.27799)  

**Abstract**: International Classification of Diseases (ICD) is a globally recognized coding system that records diagnostic events during each patient encounter, providing a standardized data foundation for various clinical tasks. However, the irregular and hierarchical nature of ICD code sequences poses challenges for N-D lattice-based sequential modeling methods, leading to overly complex model designs. In this paper, we propose GraD-IBD, a graph diagnosis model that reformulates longitudinal ICD trajectories as visit-bucketized, temporally directed graphs to detect the risk of inflammatory bowel disease (IBD). A novel context-aware, time-decay message passing mechanism was developed to capture temporal dependencies while reducing model complexity. The experimental results using a real-world clinical dataset demonstrated consistent and robust improvements in IBD detection over state-of-the-art methods, with significant reductions in computational complexity compared to sequential models. These findings highlight the potential of graph representation learning to enable efficient, scalable, and accurate disease risk prediction from longitudinal ICD diagnosis codes. 

---
# A Fixed-Budget, Cluster-Aware Standard for LLM-as-a-Judge Evaluation: A Multi-Hop RAG Stress Test 

**Authors**: Camilo Chacón Sartori, José H. García  

**Link**: [PDF](https://arxiv.org/pdf/2605.27789)  

**Abstract**: Retrieval-augmented generation (RAG) systems are often compared by asking a large language model (LLM) judge which answer is better. For multi-hop RAG, this has become a measurement problem as much as a modeling problem: the same score can reflect retrieval quality, answer length, lexical overlap, or a statistical test that ignores clustered data. We ask what happens when these choices are made explicit.
We propose a minimum measurement standard for LLM-as-a-judge comparisons in RAG. The standard fixes the top-100 candidate pool, evidence budget, answer cap, generator, and prompt; it also requires pre-registered hypotheses, cluster-aware inference, an exact cluster sign-flip check when feasible, and second-judge replication. Clustered benchmarks can overstate progress; the field should adopt this standard. We stress-test it with Genetic Algorithm Decoder for Multi-hop Evidence Composition (GADMEC), an evolutionary evidence selector, on 400 multi-hop questions in computer science/machine learning (CS/ML) and Materials Science. The protocol changes the empirical story. A binomial test makes all four semantic-baseline comparisons look significant; cluster-aware inference leaves only one Bonferroni-significant result. BM25 beats pure semantic GADMEC under the same budget, while a lexical-semantic hybrid recovers in CS/ML and narrows the Materials Science gap. 

---
# A Query Engine for the Agents 

**Authors**: Kenny Daniel  

**Link**: [PDF](https://arxiv.org/pdf/2605.27785)  

**Abstract**: The fastest-growing data in production today is unstructured text: agent traces, chat logs, reasoning chains, model outputs. People want to analyze it, and the questions worth asking ("show me where the agent got confused") cannot be answered by SQL alone, since text is not queryable without a model in the query path. The natural place this analysis is happening is the new class of AI applications (Claude Code, Cursor, Claude Desktop, in-browser agents) that run client-side and host both a human user and an LLM agent in the same process. These applications increasingly want to work with data, but the lakehouse read path has been hard to use from a JS runtime: Spark, Trino, and managed warehouses do not fit there. To build this new kind of AI data application, three properties of the engine become first-order: a JS-native distribution that drops into the runtime the application already runs in, a bundle small enough to ship inside a cold tab or per-turn agent sandbox, and a way to interleave analytic operators with model-based interpretation of text. We present Hyperparam, three open-source JavaScript libraries (Hyparquet, Squirreling, Icebird) totaling under 70 KB, that read Parquet and Apache Iceberg directly from object storage and meet the third property with per-cell, async-native SQL execution, so expensive cells fire only when downstream operators demand them. Squirreling runs LLM-shaped async UDFs over 300x faster than DuckDB-WASM on filter-bounded queries (and 192x on sort-bounded queries) and completes a ten-task agent analyst suite at two-thirds lower cost. We argue that data engineering as a discipline needs to update for the AI-native client applications now in production and the agents that work alongside their users. 

---
# Diagnosing Live Within-Policy Instruction Conflicts in LLM Agents with Witnessed Resolution Profiles 

**Authors**: Lu Yan, Xuan Chen, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27784)  

**Abstract**: LLM agents are governed by long-lived natural-language prompt policies, but individually reasonable standing rules can interact in uninspected ways. We study live intra-policy rule-conflict diagnosis: finding rule pairs inside a single prompt policy that can co-govern a realistic state, and measuring how models resolve that pressure in responses or tool actions. We introduce WIRE, a Witnessed Intra-policy Rule Evaluation pipeline. WIRE extracts source-grounded rules, encodes them as PyRule clauses, uses satisfiability checks to retain same-surface hard-collision candidates, realizes those candidates as concrete co-governance witnesses, and judges model outputs against the original source-rule text. Across six public prompt policies, WIRE extracts 276 source rules and 560 atomic clauses, classifies 30,944 within-policy clause-pair comparisons, retains 170 encoded hard-collision candidate source-rule pairs, and realizes them as 1,402 concrete witnesses. In policy-only evaluation, these witnesses yield 13,335 post- generation trials where both source rules govern and both compliance labels are judgeable. Only 35.4% fall in joint compliance; 64.6% violate at least one governed source rule. These profiles are conditional diagnostics for WIRE-selected candidates, not deployment-frequency or causal excess failure estimates, but they reveal distinct policy, model, and tool-action resolution patterns. 

---
# Auditable Decision Models with Learned Abstention and Real-Time Steering 

**Authors**: Sankaranarayanan Palamadai Chandrasekaran  

**Link**: [PDF](https://arxiv.org/pdf/2605.27768)  

**Abstract**: Production AI systems often operate with incomplete, conflicting, or insufficient evidence. Forced classifiers collapse such cases into action labels, while generative systems can produce outputs that are difficult to interpret as auditable execution decisions. We study operational decision control for AI systems, where uncertainty must be explicitly routable, policy-governed, and auditable rather than hidden inside forced predictions or free-form generation. We present EvaluatorDPT, a bounded decision-control model that predicts YES, NO, or TBD, where TBD is learned as a deferral outcome rather than added only as a post-hoc confidence rule. The model uses a transformer encoder with a primary bounded-decision head and structured auxiliary channels for values and emotions/sentiments. The interface is domain-agnostic in form: a deployment domain supplies evidence and policy thresholds, while the model emits a bounded distribution that can be controlled at inference time through recorded operating thresholds and, when validated, auxiliary semantic signals. For the evaluated model version, we report decision performance on held-out validation and test splits; auxiliary emotion metrics are omitted because the emotion head is disabled for this evaluation. On the held-out test split (n=44,597), the model achieves Accuracy = 0.8260 and Macro F1 = 0.8252, with per-class F1 of 0.8314 (YES), 0.8486 (NO), and 0.7956 (TBD). The evaluation record also includes calibration evidence (ECE = 0.0338 on validation), threshold-sweep outputs, multi-seed stability checks, confusion matrices, and reproducibility commands. Our main contribution is a bounded execution interface in which deferral is learned, inference-time routing remains inspectable, auxiliary signals provide a path to auditable behavior control, and evaluation evidence supports external review. 

---
# Got a Secret? LLM Agents Can't Keep It: Evaluating Privacy in Multi-Agent Systems 

**Authors**: Aman Priyanshu, Supriti Vijay, Esha Pahwa  

**Link**: [PDF](https://arxiv.org/pdf/2605.27766)  

**Abstract**: LLM safety evaluations predominantly test models in isolation, yet deployed AI agents increasingly operate within persistent social environments alongside other agents. We introduce a Moltbook-style simulation platform where thousands of LLM agents interact across communities over a simulated month, and use it to evaluate privacy as a downstream safety concern under varying degrees of social pressure. We find that shifting from single turn to multi turn social evaluation amplifies privacy violations (CIMemories 19.95% to Ours 45.30% across OpenAI models), that leakage is socially contagious, with agents 8 times more likely to disclose sensitive information after observing a peer do so, and that explicit privacy instructions reduce but do not eliminate this effect, leaving leakage rates above 37.8% even with safeguards. Our findings suggest that static chat based safety benchmarks systematically underestimate risks in agentic deployment, and that social context alone is sufficient to elicit sensitive disclosures that single turn evaluations would never surface. 

---
# PEAM: Parametric Embodied Agent Memory through Contrastive Internalization of Experience in Minecraft 

**Authors**: Yuchen Guo, Junli Gong, Hongmin Cai, Yiu-ming Cheung, Weifeng Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.27762)  

**Abstract**: We present PEAM, a Parametric Embodied Agent Memory framework in Minecraft that transforms agent memory from inference-time retrieval into parameter-resident skills internalized through experience. PEAM pairs a slow deliberative LLM for open-ended reasoning with a fast parametric module for reflexive execution of consolidated skills. The fast module is a multimodal Mixture-of-Experts LoRA architecture with per-category physically isolated adapters, enabling parameter-level continual learning without catastrophic forgetting. We treat failure as a first-class training signal: failure--correction trajectory pairs are internalized through a joint behavioral-cloning and contrastive objective, so the agent learns not only what succeeds but also how corrected actions differ from failed ones. To govern consolidation, PEAM introduces a parameterization-worthiness score for deciding which experience should be internalized, and a scale-free self-triggered consolidation mechanism for deciding when to internalize without task-specific hand-tuned thresholds, making the agent self-evolving as the trigger transfers across task distributions without re-tuning. Experiments in Minecraft show that PEAM improves long-horizon task performance, mitigates forgetting on previously consolidated skills, and improves parametric-versus-retrieval efficiency over retrieval-based embodied agents and parametric memory variants. 

---
# SkillGrad: Optimizing Agent Skills Like Gradient Descent 

**Authors**: Hanyu Wang, Yifan Lan, Bochuan Cao, Lu Lin, Jinghui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.27760)  

**Abstract**: Agent skills provide a lightweight way to adapt LLM agents to specialized domains by storing reusable procedural knowledge in structured files. However, whether downloaded from third parties or self-generated, these skills are often unreliable, incomplete, or outdated. Existing skill-evolution methods often address these deficiencies through heuristic reflections without an explicit optimization formulation. In this paper, we propose SkillGrad, a gradient-descent-inspired framework for optimizing agent skills. SkillGrad treats the skill package as a structured parameter to optimize in a gradient descent fashion: task executions provide trajectory-level loss evidence, automatic diagnoses then provide text-based gradients that indicate the correction directions. To stabilize optimization across iterations, a momentum agent accumulates recurring diagnostic patterns into a persistent memory overlay. Finally, an LLM-based patcher executes the parameter update by applying layer-aware edits to the skill package. Evaluated on SpreadsheetBench Verified and WikiTableQuestions, SkillGrad consistently outperforms training-based skill evolution baselines across two backbone LLMs, improving over the strongest training-based baseline by $6.7$ percentage points on average. Ablations further show that momentum and contrastive diagnosis both contribute to the final skill quality. 

---
# Asking Is Not Enough: Protocol Sensitivity in LLM Confidence Calibration 

**Authors**: Hankyeol Kim, Pilsung Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27752)  

**Abstract**: LLM confidence calibration is often evaluated by comparing two signals: token-probability scores and verbalized confidence. These signals are sometimes treated as direct readouts of model uncertainty, but their comparison depends on measurement choices that are rarely made explicit. In the main analysis, we hold the verbalized-confidence elicitation fixed: a single prompt template, probability scale, and output format. We then vary the measurement axes that define the verbalized-vs-token comparison: which answer string receives the token-probability score, how that score is read from the answer tokens, and under which conditioning context it is measured. We evaluate this design on four QA benchmarks across three open 7--8B base/Instruct model families, with larger Qwen2.5 variants as same-family robustness checks. The resulting comparison is sensitive to these choices: conditioning context changes the sign or magnitude of the ECE gap across settings, token readout produces smaller but still sign-moving changes, and changing the ECE estimator has little effect. Under the default generated-answer, bare-context protocol, Instruct settings are close to parity rather than showing a large calibration gain for verbalized confidence. In a separate supplied-answer analysis, surface-plausible wrong answers receive nearly the same confidence as supplied gold answers, suggesting that verbalized confidence also reflects answer plausibility and provenance rather than correctness alone. We argue that both confidence signals should be treated as protocol-dependent behavioral measurements, and provide a reporting checklist covering elicitation provenance, scored answer, token-probability readout, and conditioning context. 

---
# A Policy-Driven Runtime Layer for Agentic LLM Serving 

**Authors**: Rui Zhang, Chaeeun Kim, Liting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.27744)  

**Abstract**: Multi-agent LLM systems have become the dominant production workload, but the serving stack was not built for them. The agent framework above knows agent identities, role, schemas, and dispatch structure but never sees an engine-level event; the serving engine below sees every event but knows nothing about agents. A surprising number of cross-cutting policies depend on both: prefix caching, batch shaping, speculative execution, fairness, tool-result memoization, safety enforcement, and more. Each lives in the seam between the two layers and is currently solved by a one-off patch into one neighbor or the other. We argue this seam is best addressed by an architectural change rather than point fixes: insert a third tier, an agent runtime layer, between the framework and the engine, exposing four primitives (observe, score, predict, act) into which any agent-aware policy plugs, with agent identity as the shared coordinate. We map nine concrete policies onto the layer and validate the abstraction in depth on the one with the largest immediate serving-cost lever: KV caching across sessions, instantiated as CacheSage, which learns the per-workload agent transition matrix online and uses it for survival-based eviction and between-step prefetch. Preliminary results on five real multi-agent workloads show +13 to +37 pp cache hit-rate lift, 12% to 29% lower mean TTFT, and 6% to 14% higher throughput over an unmodified serving stack. 

---
# Prefix-Safe Bayesian Belief Tracking for LLM Reasoning Reliability:Separating Calibration from Ranking 

**Authors**: Zhenghan Song, Yunyi Li, Yulong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.27712)  

**Abstract**: Long reasoning traces need reliability estimates before final answers are known. We study prefix-conditioned eventual-success estimation, $P(y=1 \mid o_{1:t})$, using prefix-safe observations. Sequential Bayesian Belief Tracking (SBBT) calibrates observation likelihoods and recursively updates a two-state belief, providing a common tracker for scalar scores, text and self-verification markers, hidden clusters, token-pooling probes, and latent-trajectory features. Across generated open-weight traces on MATH-500, GSM8K, AIME 2025, and RIMO-N, probability quality and ranking separate: score-only SBBT often improves Brier, while AUROC gains require structure-aware evidence beyond strong prefix-safe baselines. In the strongest hard math setting, structure-aware observations reach +0.110 AUROC against standard prefix-safe baselines. Under a same-prefix classifier audit, MATH-500 text markers and RIMO-N self-verification signals remain positive. Together, these findings support SBBT as a calibration-aware online inference framework and expose an evidence regime: scalar scores mainly support probability quality, while structure-aware prefix signals support ranking only when strong prefix-safe baselines have not already absorbed the rank evidence. 

---
# DeepSciVerify: Verifying Scientific Claim--Citation Alignment via LLM-Driven Evidence Escalation 

**Authors**: Shaghayegh Sadeghi, Khashayar Khajavi, Rise Adhikari, Alexander Tessier  

**Link**: [PDF](https://arxiv.org/pdf/2605.27710)  

**Abstract**: Misalignment between claims and their cited evidence is a common failure mode in reports generated by large language models, limiting their reliability in scientific and other high-stakes settings. We present DeepSciVerify, a two-stage pipeline for scientific claim-citation verification that combines abstract-level reasoning with selective escalation to passage-level evidence. The system first verifies claims using the abstract and defers uncertain cases, retrieving and analyzing full-text passages only when necessary. This design leverages complementary behaviors across LLMs, as some models are more conservative while others are more decisive under uncertainty. On the SCitance benchmark, DeepSciVerify achieves 86.7 Micro-F1, outperforming strong abstract-only baselines by +4.5 points while resolving 67% of instances without full-text retrieval. These results suggest that selective evidence escalation improves both accuracy and efficiency in claim-citation verification. 

---
# Hierarchical Prompt-Domain Control and Learning for Resource-Constrained Agentic Language Models 

**Authors**: Joan Vendrell Gallart, Russell Bent, Michael Grosskopf  

**Link**: [PDF](https://arxiv.org/pdf/2605.27703)  

**Abstract**: Large Language Models are increasingly deployed inside agentic systems, where they must follow structured protocols, adapt to evolving states, and operate under memory, latency, and cost constraints. In such regimes, prompt extension is unreliable: growing contexts can push compact models outside their effective prompt domain, while deployment-time fine-tuning remains limited by scarce data and compute. We propose a hierarchical control-and-learning framework in which a compact model is first distilled to learn the required output schema, then supervised online by an oracle-controller loop. The controller monitors protocol validity and semantic performance, projects accumulated histories into a feasible prompt domain, and triggers lightweight oracle-supervised fine-tuning under drift. This separates schema learning for communication compatibility from semantic adaptation for task-level correction. We formalize prompt-domain feasibility and attention-induced saturation, motivating control of the effective prompt state rather than reliance on nominal context length. Using Multi-Fidelity Bayesian Optimization as a controlled sequential testbed, we characterize a core deployment failure mode and show improved reliability and cost-efficiency over non-hierarchical, distillation-only, and non-distilled baselines. 

---
# Cross-Entropy Games and Frost Training 

**Authors**: Arthur Renard, Franck Gabriel, Valentin Hartmann, Clément Hongler  

**Link**: [PDF](https://arxiv.org/pdf/2605.27701)  

**Abstract**: We present Frost Training, a method for improving Monte Carlo-based policy optimization for a large family of LLM-as-a-judge tasks called Cross-Entropy Games. The key idea is to exploit the gradient of the reward function in embedding space. This signal is used in the Greedy Coordinate Gradient (GCG) jailbreaking technique; we demonstrate for the first time that it can also be used to boost model training. We validate our method using GRPO training for maximum-likelihood infilling. Frost Training improves the model's ability to generate high-scoring outputs, reaching higher maximum scores in a best-of-k setting, and does so at an increased speed. 

---
# Behavioural Analysis of Alignment Faking 

**Authors**: Nathaniel Mitrani Hadida, Rhea Karty, David Williams-King, Alan Cooney  

**Link**: [PDF](https://arxiv.org/pdf/2605.27681)  

**Abstract**: Alignment faking (AF) refers to a model strategically complying with a training objective to avoid behavioural modification while preserving its deployment preferences. Understanding when and why AF arises matters as models grow better at distinguishing training from deployment. Prior work finds AF fragile, prompt-sensitive, and model-dependent, leaving its underlying drivers unclear. We study AF in a controlled, minimal setup that isolates its core components, and observe it across a wider range of models than previously reported, including small-scale models. We identify three separable drivers -- values, goal guarding, and sycophancy -- and show via targeted prompt ablations and activation steering that each independently modulates AF behaviour. Our results indicate AF is more widespread than previously reported and that its occurrence is predictable from situational cues and measurable model tendencies such as baseline sycophancy and stated values. The decomposition suggests concrete directions for detecting and mitigating AF in future models. 

---
# Intelligence as Managed Autonomy: Failure, Escalation, and Governance for Agentic AI Systems 

**Authors**: Srini Ramaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2605.27628)  

**Abstract**: As autonomous and agentic AI systems scale in robotic and human-machine environments, managing hallucination and persistent but unjustified action remains an open challenge. Rather than attributing these failures solely to model or alignment limitations, this paper explores the architectural vulnerability of unbounded autonomy - the presumption that an agent should continue operating regardless of rising uncertainty. It introduces a theory of managed autonomy that defines intelligent behavior through the formal capacity to detect epistemic drift, suspend reasoning, attempt recovery, and ultimately surrender control when reliability diminishes. We instantiate this theory via the SMARt (Self-Managing Multi-tier Autonomous Reasoning with Regulated/Revoked transitions) model, a four-layer framework featuring Stable, Meta-cognitive, Assisted, and Regulated states. By developing a timed, guarded Petri net formulation, we establish theoretically bounded properties for the system, demonstrating how architecture can formally mandate escalation, constrain invalid outputs, and ensure governance reachability under specified conditions. We further analyze how incorporating domain-specific trigger sets across varied operational settings (e.g., healthcare, robotics, etc.) can systematically preserve safety, assuming completeness and soundness criteria are met. Because these triggers are designed to be adaptive, the SMARt model accommodates the safe, controlled expansion of an agent's operational scope over time. We conclude that formalizing failure management within the autonomy lifecycle is a crucial step toward realizing reliable and governed artificial intelligence. 

---
# Reasoning and Planning with Dynamically Changing Norms 

**Authors**: Taylor Olson, Roberto Salas-Damian, Kenneth D. Forbus  

**Link**: [PDF](https://arxiv.org/pdf/2605.27622)  

**Abstract**: To safely interact with humans, AI agents must both know our norms and consider them during planning. However, such norm-guided planning has been less explored, only within communities of artificial agents, and has ignored the dynamic nature of norms. This paper instead presents an approach to guiding planning with dynamically changing norms in a human-AI setting. We contribute a defeasible calculus for resolving normative conflicts and an approach to using such dynamically changing norms as guard rails on plans. We theoretically demonstrate our approach with formal proofs and empirically with an AI agent, SocialBot, on a natural language dialogue task. 

---
# Laguna M.1/XS.2 Technical Report 

**Authors**: Julien Abadji, Marah Abdin, Connor Adams, Eric Alcaide, Mustafa Altun, Michele Artoni, Junze Bao, Uday Barar, Vassilis Bekiaris, Arkadii Bessonov, Benjamin Bütikofer, Jonathan Chang, Yen-Chun Chen, Dmitry Chernenkov, Yang Chi, Filippos Christianos, Fenia Christopoulou, Razvan-Andrei Ciocoiu, Tzachi Cohen, Yohann Coppel, Dmitrii Emelianenko, Brandon Fergerson, Brian Fitzgerald, Matthias Gallé, Alex Golonzovskyi, George Grigorev, Yiyang Hao, Christian Hensel, Jan Huenermann, Ye Ji, Sarthak Joshi, Eiso Kant, Kabir Khandpur, Seonghyeon Kim, Vladimir Kirichenko, Umut Kocasarac, Ilya Kochik, Ivan Komarov, Chaerin Kong, Anurag Koul, François-Joseph Lacroix, Sergei Laktionov, Waren Long, Quentin Malartic, Vadim Markovtsev, Afonso Marques, Robert McHardy, Carlos Mocholí, Dmitry Monakhov, Adam Morris, Martin Muller, Christian Mürtz, Robin Nabel, Thien Nguyen, Rok Novosel, Szymon Ozog, Aalhad Patankar, Aleksei Petrov, Alexandre Piché, Arthur Pignet, Teodor Poncu, Phil Potter, Alexander Rakowski, Pierre-Yves Ritschard, Jay Roberts, Joe Rowell, Piotr Sarna, Pierre-André Savalle, Uladzislau Sazanovich, Nikita Shapovalov, Arsenii Shevchenko, Mikhail Shilkov, Andrei Sokol, Mohamed Soliman, Jack Stephenson, Victor Storchan, Dragos-Constantin Tantaru, Artem Tyurin, Adrian Wälchli, Pengming Wang, Jianxiao Yang, Renat Zayashnikov, Alexander Zelenka Martin, Nikolay Zinov, Caroline Bercier, José Caldeira, Margarida Garcia, Tom George, Kabeer Gharzai, Glenn Hitchcock, Carson Klingenberg, Ivo Pinto, Varun Randery, Noah Smith, Arina Sugako, Jason Warner  

**Link**: [PDF](https://arxiv.org/pdf/2605.27605)  

**Abstract**: We present Laguna M.1 and Laguna XS.2, two Mixture-of-Experts foundation models built for long-horizon, agentic coding: M.1 has $225.8$B total parameters ($23.4$B activated per token) and XS.2 has $33.4$B total ($3$B activated). Both models were trained from scratch end-to-end inside the same internal system that we refer to as our Model Factory: a tightly-integrated stack of versioned data, training, evaluation, and inference components that turn model development into an industrial process. We describe the principles and design choices of the Model Factory and also detail the end-to-end training process of our models, throughout pre-training data and architecture, post-training stages, evaluation, and quantization.
On agentic software engineering and terminal benchmarks (SWE-bench Verified, SWE-bench Multilingual, SWE-Bench Pro, and Terminal-Bench 2.0) M.1 and XS.2 are competitive with state-of-the-art open models in their respective weight classes. Laguna XS.2 weights are released under Apache~2.0 at this https URL. 

---
# Voluntary Collusion with Secret Tools in Competing LLM Agents 

**Authors**: Xijie Zeng, Frank Rudzicz  

**Link**: [PDF](https://arxiv.org/pdf/2605.27593)  

**Abstract**: Even when a tool is explicitly described as unfair and harmful to others, ostensibly safety-aligned LLM agents still voluntarily engage in secret collusion whenever doing so confers a strategic advantage. To investigate this phenomenon, we introduce an empirical framework built on two strategic multi-agent environments: Liar's Bar, a competitive deception scenario, and Cleanup, a mixed-motive resource-management scenario, in which agents are offered secret collusion tools that provide significant advantages while clearly disadvantaging the other agents. Across 12 models (at the 7B, 70B, and proprietary scales) and 6 prompt variants, we find that most agents consistently accept these tools and develop collusive strategies, while explicitly acknowledging the unfairness of the tools before accepting. We further show that neither the unfairness labels nor baseline alignment alone reliably deters collusion: only explicit ethical framing reduces adoption and, even then, smaller models remain susceptible. More broadly, our work presents the first systematic investigation of voluntary collusion adoption in LLM-based multi-agent systems, and suggests that preventing such behaviour requires explicit safeguards rather than reliance on general alignment. 

---
# Cyberbullying Governance on Social Media: A Unified Framework from Content Identification to Intervention 

**Authors**: Yiting Huang, Wenting Zhu, Zekun Wang, Qingpo Yang, Yakai Chen, Zihui Xu, Yueyue Zhang, Sanchuan Guo, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27584)  

**Abstract**: The proliferation of social media platforms and online communities has inadvertently catalyzed the spread of cyberbullying, hate speech, and other forms of online toxicity, making the effective governance of such harm a critical societal and computational challenge. While significant strides have been made in automating content moderation, existing research predominantly treats cyberbullying governance as passive, isolated detection at the post level. This reductionist view overlooks the continuous behavioral dynamics of users, the structural diffusion of toxic events, and the critical need for proactive mitigation. To bridge these gaps, this paper proposes a unified full-lifecycle governance framework that shifts the paradigm of cyberbullying governance from isolated static detection toward integrated, continuous, and proactive moderation. Drawing on cyberbullying research and adjacent fields, we systematically synthesize the state-of-the-art literature across four interconnected stages: (1) Content Identification, (2) User and Behavior Modeling, (3) Diffusion Dynamics and Early Warning, and (4) Intervention and Governance. Furthermore, we review available datasets and evaluation practices, and discuss emerging challenges including multimodality, explainability, algorithmic fairness, and the dual-use risks of generative AI, providing a roadmap for future research toward a safer and more resilient digital ecosystem. 

---
# You Are in Control of Your State: Why Human Outcomes Are Controllable Through Causal State Intervention 

**Authors**: Suraj Biswas, Saurav Gupta, Pritam Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2605.27580)  

**Abstract**: A central puzzle for the behavioural sciences and for human-facing artificial intelligence is the persistence of within-person variability. The same individual, presented with the same observable input, produces different outcomes on different occasions, and different individuals produce divergent outcomes that no observable covariate fully predicts. We argue that this variability belongs in the dynamic latent state of the person, and that human outcomes are controllable in a precise and operational sense through interventions that target the state and its weighting at the moment a decision is being formed.
We define a state as the time-indexed weighting vector over the dimensions that govern how an individual's biology, physiology, and neuropsychology process the next event into a decision and an outcome. The relationship between state, decision, and outcome is causal rather than correlational. The weighting vector is dynamic at sub-daily timescales. The conscious channel through which outcomes are reportable is a narrow attentional bottleneck whose contents are themselves state-dependent. Taken together, these claims imply that the outcome of a given event is controllable, conditionally, on the state-trajectory at the time of intervention.
We motivate the framework with six strands of established evidence (causal inference, predictive processing, allostasis, attentional bottleneck, chronobiology, computational psychiatry) and a 24-month observational base from a deployed behavioural platform spanning more than 200,000 consented users across four occupational personas (research period 2023 to 2026). We derive seven testable predictions, list six operational requirements for state-aware systems, and discuss implications for digital health, education, AI personalisation, and personal agency. 

---
# Agyn: An Open-Source Platform for AI Agents with Scalable On-Demand Execution, Agent Definition as a Code, and Zero-Trust Access 

**Authors**: Nikita Benkovich, Vitalii Valkov  

**Link**: [PDF](https://arxiv.org/pdf/2605.27575)  

**Abstract**: As organizations move toward production deployments of AI agents, which execute non-deterministic workflows, maintain stateful sessions, and often operate with privileged access to internal services, the engineering challenge shifts from building individual agents to operating them at scale with proper isolation, governance, and security. In this paper we present Agyn, an open-source platform designed around three key principles tailored for agent workloads: a signal-driven, stateful serverless runtime on Kubernetes; a Terraform provider for agent and harness definition; and a security model grounded in zero-trust and least-privilege principles. Agyn is agent-agnostic, model-agnostic, and cloud-agnostic. 

---
# Discovery Agents for Real-Time Analytics: Toward Proactive Insight Systems 

**Authors**: Gaetano Rossiello, Dharmashankar Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2605.27571)  

**Abstract**: Modern analytics systems are fundamentally reactive, requiring users to define queries over increasingly complex and continuously evolving data. In real-time streaming environments, this paradigm breaks down, as the space of potential insights becomes too large to enumerate manually. We present a multi-agent architecture for autonomous insight discovery over real-time data streams. The system implements a continuous discovery loop in which agents generate hypotheses, compile them into executable analytics, validate generated artifacts, and produce visualizations and deployable applications. The architecture leverages Apache Kafka for event-driven coordination, Apache Flink for stream processing, and large language models to implement specialized agents. A key contribution is a contract-driven design based on typed intermediate artifacts, enabling modularity, observability, lineage, and safer execution of dynamically generated analytics. Through use cases in retail, finance, and public data, we show how this architecture supports a shift from query-driven analytics to proactive, discovery-driven systems. 

---
# LaneRoPE: Positional Encoding for Collaborative Parallel Reasoning and Generation 

**Authors**: Gabriele Cesa, Thomas Hehn, Aleix Torres-Camps, Àlex Batlle Casellas, Jordi Ros-Giralt, Arash Behboodi, Tribhuvanesh Orekondy  

**Link**: [PDF](https://arxiv.org/pdf/2605.27570)  

**Abstract**: Parallel LLM test-time scaling techniques (e.g., best-of-$N$) require drawing $N>1$ sequences conditioned on the same input prompt. These methods boost accuracy while exploiting the computational efficiency of batching $N$ generations. However, each sequence in the batch is traditionally generated independently and hence does not reuse intermediate generations, computations, or observations from other sequences. In this paper, we propose LaneRoPE to enable coordination and collaboration among $N>1$ sequences at generation time. LaneRoPE involves two key ideas: (a) an inter-sequence attention mask to make sampling of sequences dependent on one another; and (b) a RoPE extension that injects positional information that captures relative positions between tokens, both within and outside a particular sequence. We evaluate our approach on mathematical reasoning tasks and find promising results: LaneRoPE enables collaboration among sequences, yielding additional accuracy gains under limited generated sequence length. Importantly, since LaneRoPE enables coordination with minimal changes to the underlying LLM architecture and introduces a negligible overhead at inference time, it is appealing to rapidly incorporate parallel reasoning into existing LLM inference pipelines. 

---
# RULER: Representation-Level Verification of Machine Unlearning 

**Authors**: Georgina Cosma, Axel Finke  

**Link**: [PDF](https://arxiv.org/pdf/2605.27569)  

**Abstract**: Machine unlearning aims to remove the influence of specific training records from a deployed model without retraining from scratch. Current protocols verify this at the output level through membership inference, retain accuracy, and forget-set accuracy, but a model can satisfy all three whilst still encoding forgotten records in its intermediate representations. We introduce RULER, a set of representation-level verification metrics. The oracle-comparative metric M2 measures whether forget-set records occupy the same representational position as in a model retrained without them. The oracle-free metric M4 detects residuals from the unlearned model's internal similarity structure alone, without retraining. Four approximate unlearning methods all pass output-level evaluation, yet under a linear mixed-effects model M2 detects significant residuals in 10 of 12 conditions (p<0.05), with effect sizes growing as the forget fraction increases. A fifth method, Bad Teacher, shows the same residuals despite a different forgetting mechanism. M4 acts as a pre-unlearning diagnostic across tabular, image, clinical text, and face-identity settings: it detects identity-level memorisation in face recognition models where no tested method fully erases the signal. 

---
# Why LLMs Fail at Causal Discovery and How Interventional Agents Escape 

**Authors**: Amartya Roy, Sonali Parbhoo  

**Link**: [PDF](https://arxiv.org/pdf/2605.27567)  

**Abstract**: Causal discovery is a cornerstone of scientific reasoning, yet whether large language models can perform it reliably remains an open question. Recent benchmarks show that even fine-tuned models plateau on simple causal graphs and degrade as complexity grows, but why they fail has not been established. We prove the failure is fundamental: supervised fine-tuning, direct preference optimization, and in-context learning all produce predictors that cannot distinguish between causal graphs generating similar observational data, and any attempt to do so requires the model's internal representations to grow unboundedly, violating the very conditions under which these methods work. We formalize this as a kernel obstruction theorem, establishing that the limitation is intrinsic to the learning paradigm, \emph{not any particular model or dataset}. We propose Agentic Causal Bayesian Optimization (A-CBO), wherein a frozen language model serves as an interventional oracle answering targeted queries about intervention effects, while an external Bayesian loop concentrates beliefs over candidate graphs in logarithmically many rounds. Because the decision operates outside the space where the obstruction applies, A-CBO provably converges while the underlying model remains unchanged. On Corr2Cause, A-CBO matches fine-tuned baselines without any training. On Extended Corr2Cause, a new benchmark scaling to 24 variables with 18K test samples, A-CBO significantly outperforms both fine-tuning and preference optimization, with the advantage growing 

---
# DynaSchedBench: Calibrated Dynamic Scheduling Benchmarks and Observability Paradox in LLM-based Scheduling Agents 

**Authors**: Shijie Cao, Yuan Yuan, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.27566)  

**Abstract**: Progress in neural combinatorial optimization for Dynamic Flexible Job Shop Scheduling Problem (DFJSP) is currently hindered by a methodological tension: static benchmarks encourage benchmark overfitting, while uncalibrated generators obscure algorithmic capability with stochastic noise. To resolve this, we introduce \textbf{DynaSchedBench}, a diagnostic framework for DFJSP that rigorously controls the instance-generation process. Instead of relying on parameter sampling, our approach utilizes Sequential Event-Space Calibrator (SESC) that computes a novel Schedule Stress Index (SSI) to stratify instances by difficulty. We demonstrate that SESC is substantially more computationally efficient than evolutionary baselines while converging reliably to the target metrics. The framework integrates modular components for instance generation, snapshot-based simulation, agents, evaluation, and visualization, thereby enabling rigorous testing of reactive and lookahead-based policies. Leveraging this calibrated environment, we identify key limitations of LLM-based scheduling agents. Specifically, in step-wise online decision-making for dynamic scheduling, we identify an ``Observability Paradox'': providing agents with oracle access to full structural information can degrade policy performance, underperforming concise information. Furthermore, despite substantial token overhead, tool-augmented and refinement strategies fail to reliably improve performance, and most LLM agents fail to consistently surpass strong dispatching baselines-behaving more like robust heuristic approximators than superior optimizers. 

---
# On the Origin of Synthetic Information by Means of Steganographic Inheritance 

**Authors**: Ching-Chun Chang, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2605.27551)  

**Abstract**: The origin of species has been the mystery of mysteries in natural science. By analogy, the origin of synthetic information, we suggest, is the mystery of mysteries in information science. The question carries a moral weight that a technical account can neither fully resolve nor responsibly ignore, as its impact on truth, trust, and human intellect extends deep into the broader economy and society. The very power of artificial intelligence makes the evolutionary lineage of synthetic information grow ever harder to trace, for a sufficiently capable model may generate offspring that bear little resemblance, at either the structural or signal level, to the parent source from which they were derived. As in genetics, two individuals may share the same phenotype mirroring each other in outward appearance, yet differ fundamentally in their genotype. We propose, by means of steganography, a mechanism analogous to heredity. At the moment an offspring is reproduced, a projector derives a trait from the parent, and a steganographic encoder invisibly hides it within the offspring. This trait persists throughout the offspring's life cycle in a cyber ecosystem. When parentage is queried, a steganographic decoder extracts the trait from the offspring and compares it against the traits of candidate parents in a reference pool, thereby nominating the most likely one. A theoretical analysis characterises phylogenetic accuracy as a function of projector and stegosystem properties, whilst empirical evaluations across multiple projectors and stegosystems demonstrate the viability of the proposed methodology under a broad spectrum of processing operations and semantic modifications. We envision a cyber ecosystem in which synthetic information, endowed with hidden yet traceable lineage traits, branches from a simple beginning into endless forms that have been, and are being, evolved. 

---
# Soro: A Lightweight Foundation Model and Chatbot for Tajik 

**Authors**: Stanislav Liashkov, Haitz Sáez de Ocáriz Borde, Azizjon Azimi, Khushbakht Shaymardonov, Shuhratjon Khalitbekov, Bonu Boboeva  

**Link**: [PDF](https://arxiv.org/pdf/2605.27379)  

**Abstract**: We present Soro, a family of Tajik-specialized conversational large language models (LLMs) designed for real-world deployment under tight compute and connectivity constraints in Tajikistan. Starting from open-weight Gemma 3 checkpoints, we perform Tajik-only continual pretraining on a curated 1.9-billion-token corpus spanning filtered web text, PDF documents, and curriculum-aligned educational materials, followed by supervised instruction tuning on 40K Tajik teacher-style examples. To enable rigorous evaluation despite the limited coverage of Tajik in standard benchmarks, we introduce a suite of Tajik benchmarks covering general knowledge, linguistic competence, and school- and university entrance-exam domains, and we open-source them on Hugging Face. Across these Tajik benchmarks, Soro substantially outperforms same-size Gemma 3 baselines while retaining strong English performance on standard datasets. We further show that FP8 and INT4 quantization of Soro preserves most Tajik-language gains while reducing memory requirements for edge deployment, supporting an ongoing education-sector pilot and planned scale-out across schools in Tajikistan. 

---
# Identifying and Understanding Human Values in Text: A Tailorable LLM-based Architecture 

**Authors**: Eduardo de la Cruz Fernández, Marcelo Karanik, Sascha Ossowski  

**Link**: [PDF](https://arxiv.org/pdf/2605.27373)  

**Abstract**: As intelligent systems become more autonomous, the scientific community focuses on creating decision-making mechanisms that include ethical and moral considerations, unlike traditional utility-maximisation models. To achieve this, a key aspect is assessing how well these decisions align with human values. To this end, a promising line of research is centred on developing approaches based on Large Language Models (LLMs) to identify human values from text, whether explicit or implicit, enabling their recognition throughout. This paper introduces a LLM-based architecture to detect and quantify the intensity of human values in text, avoiding the limitations of previous approaches tied to specific value theory or complex prompt engineering. The architecture comprises three coordinated modules: one that generates structured value specifications from the foundational texts of any theoretical framework; one that labels texts using these specifications; and one that assigns graded support or resistance based on rhetorical and semantic evidence. This modular approach separates the tasks of conceptualising from detecting human values, creating a scalable and reproducible process driven by value specifications adaptable to various theories. The architecture was instantiated with multiple LLMs and evaluated using the ValueEval dataset. The experiments demonstrate good detection performance, confirming the generality of the pipeline. 

---
# Beyond Binary: Sim-to-Real Dexterous Manipulation with Physics-Grounded Contact Representation 

**Authors**: Jiahe Pan, Stelian Coros, Jitendra Malik, Toru Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.28812)  

**Abstract**: A primary bottleneck in contact-rich manipulation is the difficulty of collecting real-world data. Sim-to-real reinforcement learning offers a scalable alternative, but the simulation-reality gap prevents information-dense modalities like touch from being effectively used. Existing sim-to-real methods often mitigate this gap by simplifying tactile data into coarse low-dimensional features -- sacrificing the richness required for complex manipulation. In this work, we introduce Center-of-Pressure (CoP), an effective tactile representation grounded in physical principles that preserves dense contact information while maintaining robustness for sim-to-real transfer. To support this representation, we propose a sensor calibration scheme based on differentiable dynamics, enabling the estimation of taxel orientations without requiring ground-truth force measurements. We evaluate CoP on two blind, challenging contact-rich manipulation tasks: peg-in-hole insertion and ball balancing. Across both tasks, policies conditioned on CoP achieve zero-shot sim-to-real transfer on a multi-fingered hand, and outperform both coarse binary-contact and raw-taxel baselines. Analysis of learned policy states further suggests that CoP-conditioned policies encode task-relevant physical properties, such as object mass, as an emergent byproduct of control. 

---
# OmniVerifier-M1: Multimodal Meta-Verifier with Explicit Structured Recalibration 

**Authors**: Xinchen Zhang, Bowei Liu, Jiale Liu, Chufan Shi, Yizhen Zhang, Junhong Liu, Youliang Zhang, Zhiheng Li, Yujiu Yang, Ling Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28805)  

**Abstract**: Visual outcomes are increasingly central to multimodal large language models, making reliable and fine-grained verification essential for scaling generalist foundation models. In this work, we investigate multimodal meta-verification, which leverages verifier-generated rationales rather than decision-only signals, and explore how to effectively incorporate meta-verification feedback into multimodal verifier training. We identify two key findings. First, symbolic verifier outputs (e.g., bounding boxes) outperform textual explanations as meta-verification rationales, enabling efficient rule-based reinforcement learning rewards while avoiding reliance on model-based rewards from auxiliary judge models. Second, decoupling reinforcement learning objectives for binary judgment and meta-verification substantially outperforms joint reward optimization, due to intrinsic differences in output structure and learning dynamics. Based on these insights, we train OmniVerifier-M1, a generalist visual verifier leveraging symbolic meta-verification and decoupled reinforcement learning. OmniVerifier-M1 provides robust verification and fine-grained error localization, and further enables M1-TTS, a verifier-driven agentic generation system achieving dynamic region-level self-correction. This approach paves the way for more reliable, interpretable, and fine-grained multimodal verification, supporting safer and more controllable foundation model deployment. 

---
# Skill-Conditioned Gated Self-Distillation for LLM Reasoning 

**Authors**: Jiazhen Huang, Xiao Chen, Xiao Luo, Yong Dai, Senkang Hu, Yuzhi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.28791)  

**Abstract**: On-policy self-distillation (SD) improves LLM reasoning by using teacher-side privileged information (PI) to turn sparse verifier outcomes into dense token-level supervision. Existing methods usually assume trusted PI, such as reference answers or successful traces. We ask whether PI can instead come from an experience-derived skill bank, where retrieved skills are compact and reusable but may also be irrelevant or misleading. We propose Skill-Conditioned Gated Self-Distillation (SGSD), which formulates skill-based SD as teacher hypothesis validation rather than unconditional imitation. SGSD retrieves skill-mistake pairs, constructs a multi-teacher pool, and lets all skill-conditioned teachers score the same plain-prompt student rollout. The verifier validates each teacher's polarity: supporting a success or suppressing a failure gives positive supervision, while the opposite stance is reversed. A robust gated objective then distills informative teacher-student disagreements while suppressing uncertain or extreme signals. Experiments on multiple mathematical reasoning benchmarks show that SGSD consistently improves over GRPO and remains competitive with answer-conditioned OPSD under a weaker PI assumption. For example, on Qwen3-1.7B, SGSD outperforms GRPO by 6.2% and OPSD by 1.7% on average on AIME24, AIME25, and HMMT25. Our code is available at this https URL. 

---
# Do Agents Need Semantic Metadata? A Comparative Study in Agentic Data Retrieval 

**Authors**: Shiyu Chen, Tarfah Alrashed, Alon Halevy, Natasha Noy  

**Link**: [PDF](https://arxiv.org/pdf/2605.28787)  

**Abstract**: In the era of autonomous agents, machine-actionable data is critical for data-driven workflows. For more than a decade, semantic metadata like this http URL has anchored the FAIR principles (Findable, Accessible, Interoperable, and Reusable) for machine-actionable data and enabled discovery tools like Google Dataset Search. However, the rise of Large Language Models (LLMs) capable of navigating the unstructured web raises a fundamental question: Is semantic metadata still necessary for agentic data discovery, or can agents reliably retrieve actionable data directly from the web? We present a comparative analysis of agentic data retrieval across two distinct environments: a Baseline Agent searching billions of open-web documents, and a Semantic Agent leveraging a corpus of 90 million datasets using this http URL. We deploy an "LLM-as-a-judge" evaluation pipeline, mapped directly to the FAIR principles, to assess the semantic relevance, data accessibility, and computational utility of the retrieved data. Our results reveal a clear divergence. The Semantic Agent excels at retrieving actionable data, achieving a 44.9% higher precision for metadata-rich registries and a 46.6% higher precision for pages with machine-readable downloads among its returned results. Conversely, the Baseline Agent frequently suffers "Last-Mile Utility" failures, retrieving prose-heavy pages (20.1% of results) and portal landing pages (8.5%) rather than actual data pages. While the Baseline Agent achieves higher coverage by answering 40% more questions, the Semantic Agent delivers greater accuracy, achieving 65.7% higher overall precision in retrieving FAIR-compliant datasets. We conclude that while unstructured retrieval supports broad exploratory tasks, structured ecosystems remain the indispensable foundation for reliable, execution-oriented autonomous workflows. 

---
# Learn from Weaknesses: Automated Domain Specialization for Small Computer-Use Agents 

**Authors**: Suji Kim, Kangsan Kim, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28775)  

**Abstract**: Computer-use agents (CUAs) have recently made substantial progress, but deploying a separate large expert for each software domain remains expensive. Small open computer-use agents are more practical specialization targets, but they remain substantially weaker and exhibit uneven domain-specific failures. A straightforward remedy is to synthesize large-scale training data for the target domain, yet we find that this naive approach yields only marginal improvements. Building on this observation, we introduce LearnWeak, an annotation-free specialization framework for small computer-use agents that uses a stronger reference agent to identify the student's weaknesses in the target domain, synthesize targeted tasks, and construct supervision automatically. LearnWeak further introduces an error-aware specialization objective that disentangles planning and execution errors, enabling more behaviorally precise updates than broad uniform supervision. On OSWorld, LearnWeak achieves average gains of 11.6 and 11.1 percentage points over EvoCUA-8B and OpenCUA-7B, respectively, across eight domains. We also validate that our student-aware dataset generation and training approaches outperform existing autonomous trajectory generation and training baselines. Our work highlights the importance of student awareness in both data synthesis and agent training, pointing toward a more principled and efficient path for specializing small computer-use agents in diverse domains. 

---
# Rethinking Memory as Continuously Evolving Connectivity 

**Authors**: Jizhan Fang, Buqiang Xu, Zhixian Wang, Haoliang Cao, Xinle Deng, Baohua Dong, Hangcheng Zhu, Ruohui Huang, Gang Yu, Ying Wei, Guozhou Zheng, Feiyu Xiong, Haofen Wang, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28773)  

**Abstract**: Existing memory-augmented LLM agents often treat memory as a static repository with pre-defined representations and fixed retrieval pipelines, which is brittle in dynamic agentic environments where feedback, task variation, and heterogeneous signals continuously reshape what should be remembered and how it should be connected. To address this, we propose FluxMem, a connectivity-evolving memory framework that models memory as a heterogeneous graph and progressively refines its topology through three stages: initial connection formation, feedback-driven refinement, and long-term consolidation. During execution, FluxMem repairs missing links, prunes interference, aligns abstraction granularity, and distills recurrent successful trajectories into reusable procedural circuits, guided by one metric for memory generalizability and evolutionary maturity. Across three fundamentally distinct benchmarks including LoCoMo, Mind2Web, and GAIA, FluxMem achieves consistent state-of-the-art performance, demonstrating strong adaptation and generalization in complex agentic environments. The code will be open-sourced in this https URL. 

---
# Extrapolative Weight Averaging Reveals Correctness-Efficiency Frontiers in Code RL 

**Authors**: Kunhao Zheng, Pierre Chambon, Juliette Decugis, Jonas Gehring, Taco Cohen, Benjamin Negrevergne, Gabriel Synnaeve  

**Link**: [PDF](https://arxiv.org/pdf/2605.28751)  

**Abstract**: Linear interpolation between fine-tuned checkpoints has been shown to trace the Pareto front between competing objectives, but whether extrapolative weight averaging can extend such frontiers to new checkpoints useful at inference time, without additional RL training, remains unclear. We study this question in RL for competitive programming, where hidden unit tests under time and memory limits enforce both functional correctness and computational efficiency. Starting from a shared initialization, we train checkpoints under nested unit-test coverage: low-coverage rewards require passing smaller-input tests, while high-coverage rewards require passing progressively larger tests up to the full suite. This sweep reveals the emergence of a correctness-efficiency frontier: on hard problems, higher-coverage reward reduces optimization failures but increases correctness failures, leaving solve rate nearly unchanged. Interpolation between low- and high-coverage checkpoints recovers this frontier, while extrapolation extends it beyond the trained endpoints. Both the frontier and its extrapolative continuation appear across three inference settings, pure reasoning, tool use, and agentic coding, and across two model scales, 32B and 7B. At the problem level, moving along the frontier changes which problems are solved, making extrapolated checkpoints complementary policies in inference-time scaling. Ensembles with extrapolative weight averaging broaden coverage and improve pass@250 on LCB/hard by 3.3% over the best single checkpoint at matched sample budget. These results show that nested unit-test coverage in code RL induces a frontier that extrapolative weight averaging can navigate, extend, and exploit. 

---
# Preference-Shaped Expected Hypervolume and R2 Improvement: Exact Computation and Monotonicity 

**Authors**: Michael T.M. Emmerich  

**Link**: [PDF](https://arxiv.org/pdf/2605.28746)  

**Abstract**: This paper studies preference-shaped expected improvement criteria for Bayesian multiobjective optimization. We consider two indicator families which are often used for similar algorithmic purposes, but which are geometrically different. The hypervolume indicator is based on a dystopian reference point and measures dominated volume in objective space. The R2 indicator is based on a utopian point and evaluates approximation sets through weighted Tchebycheff scalarization envelopes. The purpose of the paper is to make precise which preference transformations preserve exact computation, Pareto compatibility, and monotonicity properties, and which transformations change the underlying geometry. On the hypervolume side, we revisit canonical EHVI through the Deng representation, formulate product-density weighted EHVI in desirability coordinates, discuss cone-based EHVI as ordinary EHVI after a linear cone transformation, and separate these cases from truncated EHVI, where variance monotonicity may fail. On the R2 side, we prove that exact integral R2 improvement is not, in general, an ordinary objective-space weighted hypervolume. The obstruction is lower-dimensional: Lebesgue-density hypervolume cannot see certain boundary contributions that Tchebycheff scalarizations still detect. We then show that exact integral R2 improvement is exactly a scalarization-space volume, namely the measure of the Tchebycheff shadow between the incumbent scalarization envelope and the reference envelope. This representation yields finite-sum ER2I algorithms for discrete R2, quadrature methods for exact integral R2, and an achievement-space Gaussian surrogate formulation in which ER2I is an integral of scalar Gaussian expected improvements. 

---
# Reverse Probing: Supervised Token-level Uncertainty Quantification for Large Language Models in Clinical Text 

**Authors**: Bushi Xiao, Sarvesh Soni, Daisy Zhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28740)  

**Abstract**: As large language models are increasingly deployed for clinical text, ensuring they can reliably signal their own uncertainty becomes critical. Most existing uncertainty quantification (UQ) methods are designed for open-domain generation and cannot localize uncertainty at the token or span level in long clinical text. We propose Reverse Probing, the first UQ framework specialized for clinical summarization, which estimates token-level uncertainty directly from pre-existing labeled summaries. Rather than sampling new outputs, Reverse Probing treats the text as a probe into the model's internal state, extracting uncertainty signals from four categories of internal activations. We evaluate on two expert-annotated clinical datasets and outperform eight adapted baselines on all metrics, achieving up to 4 times higher AUPRC while reducing inference time and computational costs. Feature analysis reveals that delta energy and neighborhood context are the most consistent predictors across all models. This study offers interpretable insights into how models internally respond to unsupported clinical content. 

---
# BIRDNet: Mining and Encoding Boolean Implication Knowledge Graphs as Interpretable Deep Neural Networks 

**Authors**: Tirtharaj Dash  

**Link**: [PDF](https://arxiv.org/pdf/2605.28739)  

**Abstract**: Tabular data in knowledge-rich domains often carries a latent prior in the form of Boolean implication relationships (BIRs) between pairs of features. We mine such relationships with a sparse-exception binomial test. The mined implications form a typed directed graph, equivalent to a propositional rule base of 2-literal clauses. We encode this graph as the connectivity of a layered neural network, called BIRDNet, in which each hidden unit corresponds to one mined rule and binds only to its two features. We show two consequences of this design: First, the architecture is sparse by construction: at most $2/d$ of the weights in each BIR layer are active, where $d$ is the input dimension. Second, the model is interpretable: every trained unit keeps a stable symbolic identity, so rules can be read off the network without surrogate models. Unlike most neurosymbolic models, BIRDNet does not consume an external rule base; its structural prior is mined from the data. We evaluate BIRDNet on six transcriptomic and proteomic benchmarks. Our results show that BIRDNet stays within 0.02 AUROC of the strongest dense baseline, at a small accuracy cost, while using up to $96\times$ fewer active parameters than an architecture-matched dense MLP. First-layer rules recover known biological signatures across multiple cancer subtypes and tissue types, including canonical amplicons, lineage-defining co-expression modules, and immune-infiltration markers. Data and code are available at: this https URL. 

---
# MemTrace: Tracing and Attributing Errors in Large Language Model Memory Systems 

**Authors**: Xinle Deng, Ruobin Zhong, Hujin Peng, Xiaoben Lu, Yanzhe Wu, Guang Li, Buqiang Xu, Yunzhi Yao, Jizhan Fang, Haoliang Cao, Junjie Guo, Yuan Yuan, Ziqing Ma, Yuanqiang Yu, Rui Hu, Baohua Dong, Hangcheng Zhu, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28732)  

**Abstract**: Memory is essential for enabling large language models to support long-horizon reasoning, yet existing memory systems remain unreliable and difficult to debug. Tracing memory's dynamic evolution is crucial to understand how information is synthesized, propagated, or corrupted over time. In this work, we study the new problem of error tracing and attribution in LLM memory systems. We propose a novel framework that transforms memory pipelines into executable memory evolution graphs, enabling fine-grained tracing of operational information flow. We then construct MemTraceBench, a benchmark collected from representative memory systems such as Long-Context, RAG, Mem0, and EverMemOS, to systematically study memory failure modes. We further introduce an automatic attribution method that iteratively traces operation subgraphs to pinpoint the root cause of any failed case. Our analysis reveals that memory failures are systematic, stemming from operation-level issues like information loss and retrieval misalignment. Crucially, we leverage these fine-grained attribution signals to guide downstream prompt optimization, establishing a closed-loop system that automatically corrects faults and boosts end-task performance by up to 7.62%. Code will be released at this https URL. 

---
# IPO-Mine: A Toolkit and Dataset for Section-Structured Analysis of Long, Multimodal IPO Documents 

**Authors**: Michael Galarnyk, Siddharth Lohani, Vidhyakshaya Kannan, Sagnik Nandi, Aman Patel, Liqin Ye, Arnav Hiray, Rutwik Routu, Prasun Banerjee, Siddhartha Somani, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2605.28714)  

**Abstract**: An Initial Public Offering (IPO) filing is a document released when a private firm goes public, allowing individual (retail) investors to purchase its shares. These filings describe a firm's business, financials, and risks and are long, multimodal documents with narrative text and images. Despite their importance to financial markets, there is no large-scale, standardized dataset or benchmark for studying IPO filings with modern language and multimodal models. These documents pose significant challenges: filings frequently exceed 500,000 tokens and lack consistent structural organization. We introduce the IPO-Toolkit, an open-source framework for downloading and parsing IPO filings into standardized section-structured text and extracted images. The toolkit segments filings, extracts embedded images, and produces structured outputs that enable large-scale, reproducible analysis workflows over long, multimodal documents. Using this infrastructure, we construct the IPO-Dataset, a large, section-structured, multimodal dataset covering more than 109,000 IPO filings and amendments from 1994 to 2026 and containing over 76,000 images. We establish structured evaluation tasks over extracted financial charts, including chart quality and misleadingness assessment. Our experiments show that state-of-the-art multimodal models often diverge from expert human judgments on these tasks, exposing alignment challenges in multimodal reasoning over long, real-world regulatory documents. Beyond benchmarking, the IPO-Dataset enables large-scale analysis of section-level textual variation and cross-industry differences in visual and textual disclosure practices. Our code, dataset, and website are publicly available under CC-BY-4.0. 

---
# Towards Reliable Multilingual LLMs-as-a-Judge: An Empirical Study 

**Authors**: Irune Zubiaga, Aitor Soroa, Rodrigo Agerri  

**Link**: [PDF](https://arxiv.org/pdf/2605.28710)  

**Abstract**: Large language models (LLMs) are increasingly used for the automatic evaluation of generated text, yet most prior work focuses on English. Despite the growing demand for multilingual evaluation, extending LLM-based evaluators to multilingual settings remains challenging, particularly for low-resource languages and scenarios where in-domain data is scarce. This work explores several strategies for developing multilingual LLMs-as-a-judge, considering whether in-domain data is available for fine-tuning or not. We systematically analyze English, Spanish, and Basque, representing high-, mid-, and low-resource languages, considering instruction translation, monolingual versus multilingual supervision, and model size. For evaluation, we extend two existing meta-evaluation datasets to Basque and Spanish. Our results reveal key trade-offs: When in-domain data is available, fine-tuned smaller models can achieve performance comparable to proprietary models, whereas zero-shot evaluation with larger models proves more effective in out-of-domain settings. We also observe that fine-tuning on out-of-domain data can adversely affect model performance. These findings provide practical guidance for building efficient, reliable multilingual evaluation pipelines. The data and code are publicly available at hitz-zentroa/mJudge. 

---
# A Fresh Look at Lamarckian Evolution and the Baldwin Effect 

**Authors**: Inès Benito, Johannes F. Lutzeyer, Benjamin Doerr  

**Link**: [PDF](https://arxiv.org/pdf/2605.28703)  

**Abstract**: Baldwinian and Lamarckian evolution have existed for a long time in evolutionary algorithms (EAs) without ever dominating the academic literature or practical applications. In this work, we use modern empirical and theoretical methods to revisit Lamarckian and Baldwinian evolution and rigorously compare them with the generic Darwinian evolution. On the empirical side, we run a comprehensive suite of experiments on graphs from six different datasets from the recent GraphBench benchmark on Maximum Independent Set and Maximum Cut problems. Our results show that Baldwinian and Lamarckian evolution consistently outperform Darwinian evolution, confirming the great potential of local search augmented evolutionary algorithms. Notably, in the great majority of cases, all EAs outperform recent deep learning baselines and approach the performance of highly specialised heuristic and exact solvers. We furthermore report a high-performing set of generalist parameters for all studied evolution types that we hope will be of use to practitioners in future. On the theoretical side, we extend the existing Deceptive Leading Block benchmark to arbitrary block length and use tools from modern theoretical runtime analysis to prove upper and lower bounds on the expected runtime. For block lengths greater than two, Baldwinian evolution is asymptotically faster than Lamarckian which is asymptotically faster than Darwinian evolution. When accounting for the cost of the local search procedure in fitness evaluations, the ordering depends on the implementation with Baldwinian evolution staying fastest from small block lengths onwards, explaining its strong empirical performance. 

---
# Deep Learning Strain Estimation: Is Physics-Based Simulation the Solution? 

**Authors**: Thierry Judge, Nicolas Duchateau, Andreas Østvik, Khuram Faraz, Anders Austlid Taskén, Sigve Karlsen, Thor Edvardsen, Harald Brunvand, Md Abulkalam Azad, Havard Dalen, Bjørnar Grenne, Gabriel Kiss, Pierre-Yves Courand, Lasse Lovstakken, Pierre-Marc Jodoin, Olivier Bernard  

**Link**: [PDF](https://arxiv.org/pdf/2605.28697)  

**Abstract**: Speckle tracking echocardiography (STE) is the clinical standard for myocardial strain estimation. Despite good performance on global strain (GLS), its accuracy for regional strain remains limited, even though this biomarker is highly relevant for early diagnosis and the characterization of subtle abnormalities. from clinical data. Deep learning is a promising alternative, but its development is constrained by the lack of reliable motion references. Existing solutions rely either on STE-derived labels or on simulations generated by physics-based models, but these synthetic sequences still have limited realism compared with clinical this http URL this paper, we propose a novel simulation strategy that incorporates speckle decorrelation measures from real videos and uses an iterative refinement process to improve the motion realism in the simulations. We created an open-source photorealistic dataset of 1,478 videos with reference motion, which was used to train an echocardiographic motion estimation algorithm. The proposed method achieves unmatched performance on global and regional strain, notably reaching a GLS variability of 1.42% in an inter-expert setting compared to 1.78% for the clinical reference. 

---
# Misalignment Between Backpropagation and the Hierarchy of Brain Responses to Images 

**Authors**: Joséphine Raugel, Maximilian Seitzer, Marc Szafraniec, Huy V. Vo, Jérémy Rapin, Patrick Labatut, Piotr Bojanowski, Valentin Wyart, Jean-Rémi King  

**Link**: [PDF](https://arxiv.org/pdf/2605.28693)  

**Abstract**: Backpropagation is the core learning mechanism underlying deep learning. However, whether and how this algorithm is implemented in the brain remains highly debated. In particular, while forward activations of pretrained models reliably map onto the cortical hierarchy of visual processing, it is unknown whether backpropagated gradients exhibit a similar correspondence. Here, we address this question using functional magnetic resonance imaging (fMRI) and magnetoencephalography (MEG) recordings of human brain responses to natural images. For this, we extend standard encoding analyses of forward activations to map backpropagated gradients onto neural data. Focusing on a recent self-supervised vision model (DINOv3) and reproducing results on eight vision models, we find that backpropagated gradients can reliably predict both fMRI and MEG signals, specifically in higher-level visual cortex and for later latencies. However, the spatial and temporal organization of these backpropagated gradients in the brain diverges from the patterns expected under a biologically plausible backpropagation mechanism: specifically, both the order in which gradients are computed and their spatial organization diverge from the temporal and spatial hierarchies of the human brain. Together, these results suggest that, although deep networks and the brain may share similar representational content, they likely rely on fundamentally different mechanisms to learn those representations. 

---
# AI in the Workplace: The Impact of AI on Perceived Job Decency and Meaningfulness 

**Authors**: Kuntal Ghosh, Marc Hassenzahl, Shadan Sadeghian  

**Link**: [PDF](https://arxiv.org/pdf/2605.28680)  

**Abstract**: The proliferation of Artificial Intelligence (AI) in workplaces is transforming how we work. While existing research on human-AI collaboration at work often prioritizes performance, less is known about their experiential outcomes. Through interviews with 24 employees across Information Technology (IT), service-based, and healthcare sectors, this paper examines AI's impact on job satisfaction via perceptions of job decency and meaningfulness, now and in the future. Our results reveal that the anticipated impact of AI on overall job satisfaction varies with the occupational domain, with differing perceptions of its underlying decency and meaningfulness. For instance, IT and healthcare anticipate increased satisfaction with decency aspects like working hours but decreased satisfaction with meaningfulness aspects like social image due to misconceptions about AI handling most of their tasks. Conversely, service workers foresee no improvement in their working hours but a higher social standing due to the perceived status boost associated with working with AI. 

---
# Sense Representations Are Inducible Interfaces 

**Authors**: Jan Christian Blaise Cruz, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2605.28669)  

**Abstract**: Sense representations (explicit, per-token meaning decompositions) are useful for disambiguation, steering, and cross-lingual alignment, but existing approaches require models to be pretrained with sense structure baked in. We introduce ACROS, which induces an explicit sense pathway into a frozen pretrained decoder LM through a gated residual addition. On SmolLM2-360M, ACROS preserves base LM quality while supporting three uses of the same induced variables: zero-shot word-sense disambiguation (64.95 F1 on Raganato ALL, competitive with the WordNet first-sense heuristic), low-KL lexical steering across 5,161 CoInCo cases where a simple non-oracle proxy recovers about 90% of positive shifts, and SENSIA cross-lingual adaptation to four languages (mean R@1 0.988, target FLORES PPL 7.94). ACROS makes sense representations an inducible interface for ordinary pretrained LMs. 

---
# The Attentional White Bear Effect in Transformer Language Models 

**Authors**: Rebecca Ramnauth, Brian Scassellati  

**Link**: [PDF](https://arxiv.org/pdf/2605.28639)  

**Abstract**: Instruction-based suppression is widely used to prevent language models from generating prohibited content, yet it remains unclear whether suppression reduces internal representation or merely suppresses expression. We investigate this question through representational probing, attention analysis, and behavioral semantic leakage experiments across multiple transformer models. We find that prohibited concepts remain highly recoverable from hidden representations under suppression, continue to influence attention routing, and measurably shape downstream generations despite successful lexical avoidance. These effects persist across pooling strategies, indirect semantic controls, and multiple model families. Our results expose a fundamental gap between behavioral and representational alignment. 

---
# Blind PRNG Hijacking: An Undetectable Integrity-Preserving Attack Against LLM Watermarking 

**Authors**: Ziyang You, Huilong He, Xiaoke Yang, Xuxing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28632)  

**Abstract**: Cryptographic watermarking is a leading defense for attributing text generated by large language models (LLMs). Existing schemes, including KGW, Unigram, and DipMark, derive their security guarantees from the assumption that the underlying pseudo-random number generator (PRNG) is trustworthy. This work introduces SeedHijack, the first supply-chain attack on LLM watermarking that is simultaneously (i) blind -- requiring no knowledge of the watermark key, detector, or model logits, (ii) integrity-preserving -- amplifying rather than erasing the watermark signal, and (iii) orthogonal to detection -- the attack-induced bias is statistically independent of all content-side detector statistics, ensuring that amplification and evasion coexist without trade-off. Rather than perturbing generated text, SeedHijack replaces the PRNG at the supply-chain layer, biasing green-list selection without altering output tokens or degrading text quality. Across three watermarking schemes and three open-source LLMs, the attack triggers 0/6 state-of-the-art content-side statistical detectors while inflating the watermark z-score up to 2.42x (system-level defenses such as entropy-source attestation remain orthogonal and complementary). A quantum random number generator (QRNG) countermeasure is shown to fully neutralize the attack while preserving benign watermarking utility. These findings establish PRNG integrity as a first-class security requirement for cryptographic content-provenance systems. 

---
# Measuring Form and Function in Language Models 

**Authors**: Héctor Javier Vázquez Martínez, Charles Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28616)  

**Abstract**: We introduce quantitative metrics for child language acquisition to evaluate language models. Our focus is on the formal syntactic and functional discourse properties of determiners in English, which young children acquire early and accurately. We propose Contextual Alternative Choice (CAC), a new prompting method which provides targeted tests for both syntactic and discourse knowledge of language. The method enables direct comparison of language models against children, and more importantly, against statistical benchmarks independently established in empirical research. No current model trained on a comparable amount of data simultaneously meet both formal and functional benchmarks like human children, but some very large models do. We present our results as methodological and technical contributions, with specific emphasis on cognitive status of language models. 

---
# Mining Multi-Modality Spatio-Temporal Cues for Video Important Person Identification 

**Authors**: Xiao Wang, Minglei Yang, Bin Yang, Wenke Huang, Zheng Wang, Xin Xu, Mang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.28604)  

**Abstract**: Identifying key individuals in video scenes is essential for applications such as automated video editing and intelligent surveillance. Current methods primarily focus on static images and immediate visual cues, overlooking the rich spatio-temporal information in videos. This leads to the phenomenon of Temporal Importance Shift (TIS), wherein individuals deemed significant in early frames may be demoted as the entire temporal context is considered. To address this, we introduce the Video Important Person (VIP) identification task, aimed at automatically identifying the most influential individuals in videos while providing textual rationales. We present Temporal-VIP, a large-scale rationale-annotated dataset consisting of 9,249 video segments across 11 categories with aligned importance rationales. To mitigate TIS, we develop the VIP-Net framework, which includes a Social Cue Encoder (SCE) for extracting multi-modal spatio-temporal cues, a Temporal Importance Rectifier (TIR) for hierarchical cue fusion and cross-modal alignment, and VIP Inference for ranking individuals. Experimental results show that VIP-Net achieves 67.3% accuracy, significantly outperforming state-of-the-art models (37.5%-53.9%) and yielding a mean rationale similarity of 0.63 to ground truth through feature-guided LLM refinement. The dataset and code are available at this https URL. 

---
# Online Irregular Multivariate Time Series Forecasting via Uncertainty-Driven Dual-Expert Calibration 

**Authors**: Haonan Wen, Hanyang Chen, Songhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.28603)  

**Abstract**: Irregular multivariate time series forecasting is critical in many real-world applications, where time series are irregularly sampled and exhibit dynamically evolving missingness patterns. Although existing methods perform well in offline settings, they often suffer from significant performance degradation when deployed online due to dynamic shifts in data distribution. Maintaining forecasting capability in such dynamic scenarios typically necessitates online adaptation techniques. Since irregular sampling fundamentally undermines temporal continuity and periodicity, we cannot leverage these widely studied characteristics from regular MTS for online learning. To this end, we study the problem of online IMTS forecasting and propose Under-Cali, an uncertainty-driven dual-expert calibration framework consisting of three core components: an uncertainty estimator, a dual-expert calibration module, and an adaptive routing module. We design an uncertainty estimator that serves as the core control signal to jointly manage inference and adaptation processes. In our framework, the uncertainty estimator first assesses uncertainty for each incoming batch. The adaptive routing module then directs samples with high uncertainty to the unreliable expert for calibration, while low uncertainty samples remain with the reliable expert. Subsequently, the system updates the reliable expert and the uncertainty estimator using well-calibrated reliable samples, and updates the unreliable expert with challenging samples, enabling stable and efficient online learning. Under-Cali keeps the source forecasting model frozen and performs adaptation only through a lightweight, model-agnostic calibration module, enabling efficient adaptation. Extensive experiments on IMTS benchmarks demonstrate consistent improvements with low computational cost. Our code is available at this https URL. 

---
# Evaluating the Realism of LLM-powered Social Agents: A Case Study of Reactions to Spanish Online News 

**Authors**: Alejandro Buitrago López, Alberto Ortega Pastor, Javier Pastor-Galindo, José A. Ruipérez-Valiente  

**Link**: [PDF](https://arxiv.org/pdf/2605.28598)  

**Abstract**: LLM-powered social agents are increasingly used to simulate online social behavior, yet their realism remains difficult to validate. Existing work has largely relied on general-purpose benchmarks, while less attention has been paid to short, reactive discourse such as audience replies to online news.
In this paper, we evaluate whether LLM-generated reactions to Spanish online news reproduce measurable properties of real audience discourse. Using the Hatemedia dataset, we pair 5,631 news items with 58,555 real audience reactions, and generate a matched synthetic dataset using five LLMs under a shared experimental setting. We compare real and synthetic reactions across three dimensions: hate speech, sentiment, and semantic alignment, considering both off-the-shelf and fine-tuned generation.
Results show that off-the-shelf models are poor proxies for real audience reactions: they strongly underproduce hate speech, introduce model-specific sentiment biases, and remain distributionally distant from human replies. Fine-tuning improves fidelity unevenly. Qwen3 provides the most balanced approximation, while Mistral7B achieves the strongest sentiment and semantic alignment but overshoots hate prevalence. Plausible synthetic replies do not necessarily reproduce the distributional properties of public discourse. 

---
# Position: Retire the "Positive Backdoor" Label -- Secret Alignment Requires Strict and Systematic Evaluation 

**Authors**: Jianwei Li, Jung-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.28597)  

**Abstract**: This position paper argues that the AI/ML community should stop overclaiming and retire the label "positive backdoor," and instead treat trigger-activated hidden behaviors as Secret Alignment. Crucially, protective claims based on Secret Alignment should be presumed not secure by default unless supported by rigorous, standardized evaluation. The Private AI era, enabled by open-weight LLMs and accessible training/inference stacks, turns language models into privately owned digital assets, creating security concerns around unauthorized access, model theft, and behavioral misuse. Recently, a line of work framed as "positive backdoors" has been proposed to address these challenges. To ground our position in evidence, we unify these proposals as covert trigger-behavior associations for access gating, ownership attribution, and safety enforcement, and evaluate three representative applications across six core properties: effectiveness, harmlessness, persistence, efficiency, robustness, and reliability. Our results reveal substantial brittleness - especially in the confidentiality, integrity, and availability (CIA) - of trigger-behavior mappings often underrepresented by existing claims. We further relate these outcomes to behavior density and decision complexity, offering a behavioral lens for understanding deployment-time risks and motivating community-wide evaluation that makes Secret Alignment claims provable. 

---
# Thermodynamic properties of chemically disordered compounds via AI-driven estimation of partition function with the PULSE method 

**Authors**: Baptiste Bernard, Luca Messina, Eiji Kawasaki, Emeric Bourasseau  

**Link**: [PDF](https://arxiv.org/pdf/2605.28594)  

**Abstract**: In this article, we present an improved version of the PULSE method (Partition function Unsupervised Learning Sampling and Evaluation) for estimating the thermodynamic properties of chemically disordered compounds. The aim is to reduce the computational cost of Monte Carlo approaches for this type of material and to demonstrate that this generative tool can estimate thermodynamic properties by sampling and estimating the partition function of the system. To validate this innovative approach, we use the 2D Ising model as a benchmark. We demonstrate that our method accurately reproduces average properties with high precision and efficiency compared to traditional Monte Carlo sampling methods. Our results highlight the efficiency and adaptability of the PULSE method, making it a valuable tool for studying materials for which conventional methods are too inefficient to compute properties affected by chemical disorder at low cost. 

---
# Models That Know How Evaluations Are Designed Score Safer 

**Authors**: Katharina Deckenbach, Haritz Puerto, Jonas Geiping, Sahar Abdelnabi  

**Link**: [PDF](https://arxiv.org/pdf/2605.28591)  

**Abstract**: The validity of AI safety evaluations depends on models behaving consistently across controlled and deployment settings. Prior work has identified test-time contextual cues, such as hypothetical scenarios, as a source of verbalized evaluation awareness and subsequent behavioral shift. In this paper, we investigate a potential explanation of this phenomenon: evaluation meta-knowledge, defined as parametric knowledge about the structural traits that characterize evaluations. Similar to dataset contamination, where benchmark exposure leads to higher performance through memorization, we hypothesize that models trained on texts describing evaluation practices may implicitly learn to recognize and respond to evaluation-like contexts, for instance, through exposure to scientific articles or social media posts about AI benchmarking. To test this, we fine-tune models on synthetic documents describing evaluation traits such as verifiable structures or moral dilemmas. Evaluating this fine-tuned model on six safety benchmarks, we find that it is significantly safer than the base model and control model. This behavioral shift persists even when restricting the analysis to responses lacking explicit verbalization of evaluation awareness. Our results demonstrate that evaluation meta-knowledge may inflate safety benchmark performance, introducing a novel confounder that is independent of explicit memorization or verbalized evaluation awareness, thus, challenging to detect. These findings have important implications for the design and interpretation of AI safety evaluations. Our code and models are available at this https URL. 

---
# Technical Report: Exploring the Emerging Threats of the Agent Skill Ecosystem 

**Authors**: Luca Beurer-Kellner, Aleksei Kudrinskii, Marco Milanta, Kristian Bonde Nielsen, Hemang Sarkar, Liran Tal  

**Link**: [PDF](https://arxiv.org/pdf/2605.28588)  

**Abstract**: We analyzed 3,984 AI agent skills from major marketplaces and found 76 confirmed malicious payloads, including credential theft, backdoor installation, and data exfiltration. 13.4% of all skills contain at least one critical-level security issue and at least 8 manually confirmed malicious skills remain publicly available on this http URL as of the date of publication. This report documents our methodology, presents a threat taxonomy based on real-world samples, and details the attack patterns we observed. As skill marketplaces grow rapidly and AI agents gain access to sensitive credentials and systems, automated security analysis is no longer optional. 

---
# SARAD: LLM-Based Safety-Aware Hybrid Reinforcement Learning with Collision Prediction for Autonomous Driving 

**Authors**: Kangyu Wu, Peng Cui, Guoxi Chen, Ya Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28583)  

**Abstract**: Ensuring both safety and efficiency in decision-making for autonomous driving systems remains a fundamental challenge. Traditional Deep Reinforcement Learning (DRL) suffers from unsafe random exploration and slow convergence, while Large Language Models (LLMs) demonstrate inherent latency in real-time inference operations. To address these limitations, this paper proposes SARAD, a novel safety-aware hybrid framework that synergizes LLMs and DRL for autonomous driving. SARAD substitutes the random exploration of DRL with Retrieval-Augmented Generation (RAG)-enhanced, LLM-guided decisions sourced from a dynamic expert knowledge repository. An attention discriminator is proposed to integrate the prior knowledge of LLMs into DRL policy optimization. A collision predictor module, fine-tuned with historical collision data, is further designed to improve vehicle safety. Extensive experiments show that SARAD achieves significant performance improvements in the Highway-Env simulator, validating the effectiveness of the proposed model in autonomous driving. 

---
# Efficient Pre-Training of LLMs through Truncated SVD Layers 

**Authors**: Kaivan Kamali, Kajetan Schweighofer, Hormoz Shahrzad, Olivier Francon, Babak Hodjat, Risto Miikkulainen  

**Link**: [PDF](https://arxiv.org/pdf/2605.28573)  

**Abstract**: The massive scaling of Large Language Models (LLMs) has made pretraining increasingly cost-prohibitive. While low-rank representation and orthonormal weight matrices could in principle reduce parameter counts and computational overhead, most existing methods rely on static rank selection and do not enforce weight orthonormality due to high computational cost. This paper introduces TSVD, a framework that maintains low rank and strict orthonormality throughout the training process. It utilizes a spectral energy-based heuristic for adaptive rank selection, and a caching mechanisms to maintain orthonormality. Theoretical analysis justifies the advantage of the approach in pretraining dynamics and experiments across various model scales demonstrate that it is effective empirically. TSVD matches or exceeds the performance of full-parameter baselines while significantly reducing compute requirements. The approach thus offers a well-founded, practical, and scalable path toward efficient high-performance LLM pretraining. 

---
# Semantic Optimal Transport for Sparse Autoencoder Feature Matching and Circuit Compression 

**Authors**: Tue M. Cao, Nguyen Do, My T. Thai  

**Link**: [PDF](https://arxiv.org/pdf/2605.28567)  

**Abstract**: Sparse autoencoders (SAEs) have become a central tool for interpreting language models. However, two key SAE analyses that remain difficult to scale are (1) matching semantically similar features across multi-layers and (2) compressing large feature circuits into interpretable supernodes. Although these have been treated as separate problems, we show that both are instances of a more fundamental challenge, which we frame as the estimation of semantic distances between SAE features that lie on different activation manifolds. We introduce a distributional framework for this problem, in which each feature is represented not by a single decoder vector like in the literature, but by an activation-weighted distribution over the hidden states that express it. By projecting these distributions into a shared reference space and comparing them with Wasserstein distance, our method provides a unified semantic metric for cross-layer feature comparison. We prove that our representation is invariant to activation rescaling, stable under perturbations, and recovers true matches under finite-sample margin conditions. Empirically, our method outperforms decoder-vector and LLM-based baselines and captures subtle functional distinctions between related features. Notably, our method compresses large feature circuits into interpretable supernodes automatically. 

---
# Verified Misguidance: Measuring Structural Citation Failures in Search-Augmented LLMs 

**Authors**: Yongsik Seo, Wooseok Jeong, Eunyoung Kim, Hyeonseo Jang, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.28565)  

**Abstract**: Users of search-augmented LLMs rely on citations as evidence that responses are grounded in real sources, and rarely verify the cited pages themselves. Millions of queries per day now pass through these systems, making citation quality a silent determinant of whether users are informed or misled-yet existing benchmarks each address one facet in isolation, leaving the joint structure that determines citation trustworthiness unmeasured. We construct CITETRACE, a large-scale dataset that traces the full citation chain from user query through retrieved source to generated answer: 11,200 real-world queries from 28 communities paired with 112,000 responses from ten models across five providers, yielding 761,495 evaluable citation pairs. We design a three-dimension evaluation framework that scores each citation on intent-purpose alignment, source suitability, and answer-source fidelity, using expert-validated predefined matrices and a five-level fidelity rubric; the framework applies to any system that produces citation-bearing responses. Applying this framework at scale, we identify a systematic pattern we call VERIFIED MISGUIDANCE (VM): models cite real, accessible sources yet fail along one or more dimensions, producing a fidelity-suitability trade-off in which faithful models select inappropriate sources and vice versa. Across our pool, 30.6% of citations distort their sources and 27.1% originate from domain-inappropriate sources; at the response level, up to 96% of users encounter at least one structurally misleading citation. Provider-level differences explain 88-96% of citation-quality variance, suggesting that source selection is governed more by factors beyond individual model capability than by the LLMs themselves. Together, CITETRACE and its evaluation framework provide the first resource for diagnosing structural citation failures in deployed search-augmented systems. 

---
# A Multi-dimensional Framework for Evaluating Generalization in EEG Foundation Models 

**Authors**: Aditya Kommineni, Emily Zhou, Kleanthis Avramidis, Tiantian Feng, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2605.28563)  

**Abstract**: Evaluating foundation models under appropriate adaptation settings is essential for understanding the quality and transferability of the learned representations. Recent EEG foundation models have demonstrated promising transfer capabilities across tasks and datasets, motivating their growing use in neurotechnology and clinical applications. However, these models are typically evaluated under full fine-tuning on well-curated downstream datasets, a setting that does not reflect biomedical domain constraints such as limited labeled data, reduced sensor coverage, or parameter-efficient adaptation. In this work, we propose a multi-dimensional evaluation framework for assessing EEG models under realistic low-resource conditions. Empirical analysis of both supervised EEG models and recent EEG foundation models, including LaBraM, CSBrain, and CBraMod, across 6 different datasets is performed under the proposed multi-dimensional evaluation framework. We find that EEG foundation models consistently provide performance gains on long-context tasks such as sleep stage prediction and mental health state classification. In contrast, for short-window Brain Computer Interface style tasks, supervised models achieve comparable despite having substantially fewer parameters. Additional analyses demonstrate that current foundation models provide limited robustness to short-window tasks and channel constrained settings. Together, these findings motivate the use of multi-dimensional evaluation protocols that characterize model behavior under realistic use constraints. 

---
# Token Optimization Strategies for LLM-Based Oracle-to-PostgreSQL Migration 

**Authors**: Oleg Grynets, Dmytro Babarytskyi, Vasyl Lyashkevych  

**Link**: [PDF](https://arxiv.org/pdf/2605.28557)  

**Abstract**: LLMs are increasingly used for software modernization, code translation, and database migration. However, LLM-based Oracle2PostgreSQL migration remains constrained by high token consumption, long-context degradation, dialect-specific semantic differences, and the risk of semantic drift during query transformation. Direct inclusion of large Oracle SQL/PL-SQL artefacts, schema definitions, procedural logic, and migration instructions into the model context increases cost and may reduce generation quality. This paper shows token optimization as a constrained transformation problem in LLM-based Oracle2PostgreSQL migration. The study formalizes and evaluates twelve token optimization strategies: baseline representation, context pruning, minification, DSL-based semantic compression, metadata augmentation, context refactoring, schema distillation, adaptive routing, AST-based minification, identifier masking, output constraint enforcement, and hybrid optimization. The strategies are evaluated on samples of 10 and 100 Oracle SQL queries using Valid Syntax Rate, Exact Match, Semantic Match, CodeBLEU, and Token Efficiency. The results show that mild context pruning preserves semantic quality almost at the baseline level, achieving 89.75% Semantic Match on the 100-query sample compared with 89.80% for the unoptimized baseline. Adaptive routing provides the best practical trade-off, reducing input tokens by 8.72% and output tokens by 5.49% while maintaining 88.40% Semantic Match and increasing Token Efficiency by 6.67%. Aggressive schema distillation increases Token Efficiency by 132.22% but results in a 44.50-percentage-point decrease in Semantic Match. The findings demonstrate that token optimization cannot be treated as simple prompt shortening; it must be evaluated as a multi-objective migration problem balancing cost, syntactic validity, semantic preservation, and structural fidelity. 

---
# Stochastic Gradient Descent with Momentum is Algorithmically Stable 

**Authors**: Yunwen Lei, Zimeng Wang, Xiaoming Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2605.28517)  

**Abstract**: Stochastic gradient descent with momentum (SGDM) is one of the most widely used optimization algorithms in machine learning. While optimization properties of SGDM have been extensively studied in the literature, it remains insufficiently understood whether and when SGDM can generalize well to unseen data. In particular, it has been conjectured that while momentum accelerates training, it may degrade generalization. In this paper, we close this gap by developing a comprehensive generalization analysis of SGDM through the lens of algorithmic stability. More specifically, we introduce a generalized SGDM framework that encompasses both Polyak's and Nesterov's momentum schemes, and establish tight on-average model stability bounds for smooth and convex problems. Notably, the obtained bounds exploit small optimization error bounds along the trajectory, apply to any momentum parameter in the interval $[0, 1)$, and do not require the commonly assumed Lipschitzness of loss functions. We further derive optimization error bounds for the generalized SGDM, and combine them with our generalization analyses to obtain optimal excess population risk bounds for SGDM with both Polyak's and Nesterov's momentum. 

---
# Do LLMs Favor Their Providers? Measuring Vertical Integration Bias in Code Generation 

**Authors**: Melih Catal, Alex Wolf, Tiago Ferreiro Matos, Pooja Rani, Harald Gall  

**Link**: [PDF](https://arxiv.org/pdf/2605.28515)  

**Abstract**: Large Language Models (LLMs) have become an integral part of software development, especially with the advent of agentic capabilities. Yet, many frontier LLMs are affiliated with specific providers. This raises the question of whether generated code favors the provider's own ecosystem over comparable alternatives, potentially constraining developers' choices and increasing dependence on a single provider. We define this behavior as Vertical Integration Bias (VIB) and introduce \textsc{VIBench}, a benchmark for measuring VIB in direct and agentic code generation across $20$ provider-selectable software-integration scenarios. Evaluating $10$ frontier provider-affiliated models against $3$ non-affiliated controls, we find positive VIB in direct generation, with six of ten affiliated models showing statistically significant effects up to $+18.8$ percentage points (pp). Agentic workflows further amplify VIB, reaching $+39.2$ pp. Moreover, early affiliated-ecosystem choices in agentic workflows can persist into conceptually decoupled downstream files, with persistence as high as $90.3\%$. These findings underscore the need to measure and account for VIB in code generation, especially as agentic capabilities become more prevalent. 

---
# Learning Theory of the SVRG: Generalization and Convergence Analysis 

**Authors**: Yunwen Lei, Zimeng Wang, Xiaoming Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2605.28513)  

**Abstract**: Variance reduction (VR) methods employ stochastic gradients with decreasing variance, and they have been widely applied to solve large-scale optimization problems in machine learning because of their efficiency. Existing theoretical studies of VR methods are mainly focused on the convergence analysis, leaving the generalization behavior largely unexplored. In this paper, we bridge this gap by developing the first non-vacuous generalization analysis of the representative VR method: Stochastic Variance Reduced Gradient (SVRG), through the lens of algorithmic stability. In particular, we establish sharp stability bounds of the SVRG in both convex and strongly convex settings by exploiting its algorithmic structure. The obtained bounds are data-dependent, because the training errors are incorporated along the trajectory. Our analysis clarifies the interplay between optimization and generalization, leading to optimal excess population risk bounds in both settings. Our approach differs substantially from existing analyses of stochastic algorithms in the sense that we decompose the SVRG update as an SGD-like step plus a zero-mean correction term and then introduce novel Lyapunov functions to absorb the additional gradient terms induced by the reference points. Our analytical framework can be generalized to other VR methods, and we demonstrate the generalization by the well-known Stochastic Average Gradient Accelerated (SAGA) method. 

---
# Efficient and Scalable Provenance Tracking for LLM-Generated Code Snippets 

**Authors**: Andrea Gurioli, Davide D'Ascenzo, Federico Pennino, Maurizio Gabbrielli, Stefano Zacchiroli  

**Link**: [PDF](https://arxiv.org/pdf/2605.28510)  

**Abstract**: Large language models (LLMs) for code completion and generation are increasingly used in software development, yet they may reproduce training examples verbatim and without authorship attribution, raising legal and ethical concerns around plagiarism and license compliance. Classical fingerprint-based plagiarism detectors based on fingerprinting, such as Winnowing, remain highly effective, yet the inspection requires comparing fragments of code to the entire training set, and their linear-time search makes them impractical for the billion-scale corpora used to train modern code LLMs. To bridge this gap, we introduce SOURCETRACKER, a 300M-parameter encoder tailored for code retrieval, together with a hybrid two-stage provenance-tracking pipeline HYBRIDSOURCETRACKER (HST). HST first narrows down a small set of candidate snippets via vector search, then re-ranks those candidates using Winnowing on exact fingerprints. We train and evaluate our system on a 10M-snippet subset of the THESTACKV2 dataset, with both verbatim and adapted snippets that emulate realistic identifier renaming. On an in vitro 100k-snippet search space with adapted queries, our hybrid approach reaches a mean reciprocal rank on par with Winnowing for 30-token fragments. Then, starting from windows >= 60 tokens, it consistently over-performs by up to 5.4% while preserving logarithmic-time query complexity. In a complementary evaluation using an LLM-based judge, we find that many retrieved snippets not labeled as ground truth are still highly similar to the expected sources, particularly with longer context windows, and thus remain useful for end users. Overall, our results demonstrate that integrating vector search with fingerprinting enables scalable, high-precision provenance tracking for code produced by LLMs. 

---
# Functional Entropy: Predicting Functional Correctness in LLM-Generated Code with Uncertainty Quantification 

**Authors**: Dylan Bouchard, Mohit Singh Chauhan, Zeya Ahmad, Ho-Kyeong Ra  

**Link**: [PDF](https://arxiv.org/pdf/2605.28500)  

**Abstract**: Large language models have shown impressive capabilities in code generation, yet they often produce functionally incorrect code. Uncertainty quantification (UQ) methods have emerged as a promising approach for detecting hallucinations in natural language generation, but their effectiveness for code generation tasks remains underexplored. We systematically evaluate how UQ techniques transfer to code generation across three programming languages, five LLMs, and over 1,700 problems. We find that some token-probability-based methods generalize effectively without modification, while sampling-based methods relying on natural language inference (NLI) fail because NLI models cannot distinguish functionally different code, causing most responses to collapse into a single semantic cluster. To address this, we introduce functional equivalence methods, a family of code-specific methods that replace NLI-based semantic equivalence with an LLM-based functional equivalence assessment, including functional entropy, a code-specific analog of semantic entropy. Functional equivalence methods achieve top AUROC in 11 out of 15 model-benchmark combinations and the best calibration across most settings, consistently outperforming both NLI-based counterparts and all other methods evaluated. 

---
# The Decision to Verify: How Warmth and User Characteristics Shape Reliance on Conversational Agents for Information Search 

**Authors**: Mert Yazan, Frederik Bungaran Ishak Situmeang, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2605.28498)  

**Abstract**: Conversational artificial intelligence (AI) provides an efficient and convenient gateway to information access. However, it can cause overreliance when users blindly trust AI and accept its answers without fact-checking. Information search increasingly follows a hybrid interaction paradigm that combines conversational AI with web search, making fact-checking easier. In this paper, we examine whether this interaction paradigm is effective in curbing reliance. We further investigate the underlying factors (e.g., digital literacy and conversation warmth) that drive users to verify AI answers. We conduct a mixed-subjects question-answering experiment where participants interact with either a warm or a neutral chatbot. Our findings reveal that reliance persists despite users having access to both conversational and web search. The decision to verify is driven primarily by existing user perceptions (e.g., prior trust in chatbots) rather than answer properties, with some users fact-checking regardless of the context and others trusting chatbots by default. Warm conversational style has an indirect yet critical influence on reliance by increasing agreement with the chatbot when it is incorrect. Consulting additional AI sources predicts higher accuracy, while traditional web search does not. Our study extends overreliance research by: (a) demonstrating its persistence despite access to fact-checking, (b) identifying verification behavior as user-dependent, and (c) revealing conversational warmth's indirect effect on overreliance with implications for designing trustworthy conversational search systems. 

---
# SSR3D-LLM: Structured Spatial Reasoning via Latent Steps for Fine-Grained Grounding in Unified 3D-LLMs 

**Authors**: Jiawei Li, Ziyi Liu, Weijie Shi, Long Chen, Jiajie Xu, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.28490)  

**Abstract**: 3D object grounding localizes referred objects in a 3D scene from natural language. Unified instance-centric 3D-LLMs aim to solve grounding together with dialog, QA, and captioning, yet many rely on a single pointer-style grounding decision that compresses a relational instruction into one selection. This is brittle for fine-grained queries where multiple same-class candidates must be ruled out by context objects and spatial relations. We propose Structured Spatial Reasoning 3D-LLM (SSR3D-LLM), a structured grounding interface for unified 3D-LLMs. Given fixed Mask3D object proposals, the LLM writes a sequence of latent spatial reasoning steps and memory tokens from the query, and a geometry-aware scorer reads these latent steps in order to refine candidate rankings step by step with step-length masking. The latent steps are learned from standard benchmark target supervision with auxiliary referential-cue supervision during training, while inference uses only the input query and Mask3D proposals. Across ReferIt3D, ScanRefer, and Multi3DRef, SSR3D-LLM achieves the strongest results among unified 3D-LLM baselines, with substantial gains over the single-pointer QPG baseline on fine-grained grounding and consistent improvements over prior unified 3D-LLMs, while preserving the default language-task route. 

---
# The Cases LJP Never Sees: Prosecution Decision Prediction for More Complete Criminal Liability Assessment 

**Authors**: Junyu Lu, Qi Wei, Peishuo Zheng, Jie Zhang, Hui Huang, Qianru Wang, Chuan Xiao, Jianbin Qin, Shuyuan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.28464)  

**Abstract**: Legal Judgment Prediction (LJP) has become a core benchmark for evaluating AI in the criminal legal domain, but it only sees criminal cases that have already passed prosecutorial review and been formally indicted. As a result, LJP leaves a substantial blind spot in assessing criminal liability, overlooking cases involving insufficient evidence, no criminal liability, or guilt exempted from punishment. To fill this gap, we propose \textbf{Prosecution Decision Prediction (PDP)}, the first Legal AI task built around prosecutorial review, which classifies each case into prosecution or one of three non-prosecution decisions and reflects legal AI's capabilities in evidence evaluation, legal subsumption, and value-based discretion. We further construct \textbf{PDP-Bench}, a benchmark of 4{,}630 real Chinese prosecutorial decisions spanning 190 charges. Extensive experiments show that state-of-the-art LLMs perform substantially worse on PDP than on LJP and that mainstream enhancement routes fail to close the gap. Moreover, controlled RLVR interventions show that simple outcome rewards fail to produce generalizable PDP discrimination. 

---
# BiasEdit: A Training-Free Bias-Detect-and-Edit Framework for Learning Fair Visual Classifiers 

**Authors**: Jungwook Seo, Yoonsik Park, Changmin Lee, Sungyong Baik  

**Link**: [PDF](https://arxiv.org/pdf/2605.28450)  

**Abstract**: Visual data from the Web power image classifiers, which often underpin many web services, such as recommendation and content moderation. However, the raw Web data often contain spurious correlations and social biases, and neural networks are known for their tendency to learn biases present in data. This can reinforce unfairness in web services and the web data, leading to a vicious cycle. In the context of image classification, networks learn bias attributes for a specific class when a majority of images contain the same attribute only for a given class. Hence, training a fair and debiased classifier from a biased dataset demands handling an imbalanced problem between a majority of images with bias attributes (bias-aligned samples) and a minority without (bias-conflict samples). In this work, we introduce BiasEdit, a modular framework that automatically detects bias attributes from the original dataset and edits them to construct a debiased dataset. Specifically, BiasEdit first detects unknown bias attributes via statistical dependence and mutual information analysis of visual-linguistic representations, and then explicitly edits those attributes using text-guided image editing to generate realistic bias-conflict samples. Unlike prior works that assume known bias attributes or relies on synthetic mixing, our method operates without manual annotations and can leverage off-the-shelf vision-language and editing models. BiasEdit addresses a fundamental challenge in Web-sourced visual AI, mitigating dataset-induced bias and achieving state-of-the-art debiasing performance even when training data are fully biased. 

---
# Bayesian Gated Non-Negative Contrastive Learning 

**Authors**: Peng Cui, Jiahao Zhang, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28441)  

**Abstract**: While Contrastive Learning (CL) has revolutionized self-supervised representation learning, its latent representations remain highly entangled and opaque, limiting their interpretability in safety-critical applications. We identify that a fundamental cause of this entanglement is the reliance on deterministic similarity measures, which treat all feature dimensions equally. In compositional scenes, this creates an Optimization Conflict: common background features, such as, "blue sky", are encouraged to align in positive pairs but simultaneously repelled in negative pairs, causing gradient oscillations that hinder precise semantic disentanglement. To address this, we propose BayesNCL (Bayesian Gated Non-Negative Contrastive Learning). Unlike standard approaches, BayesNCL introduces a probabilistic gating mechanism that dynamically filters out task-irrelevant, high-frequency common features while selectively retaining discriminative semantics. By formalizing feature selection as a variational inference problem with a sparse Bernoulli prior, our method effectively resolves the optimization conflict. Empirical experimental results on Imagenet-100 demonstrate that BayesNCL achieves a remarkable 142.1% improvement in semantic consistency compared to state-of-the-art baselines, yielding highly interpretable representations without compromising downstream task performance. Code is available at this https URL. 

---
# Anomaly as Non-Conformity via Training-Free Graph Laplacian Energy Minimization 

**Authors**: Jungwook Seo, Minjeong Kim, Younkwan Lee, Seungho Shin, Sungyong Baik  

**Link**: [PDF](https://arxiv.org/pdf/2605.28428)  

**Abstract**: Detecting subtle visual anomalies in images remains challenging, particularly when only normal samples are available a priori. Such unsupervised anomaly detection is typically solved by measuring feature similarity of a query patch to a memory of normal patches. However, similarity alone does not reveal how strongly a query patch violates the structure of the normal feature manifold. We propose a training-free Laplacian graph energy optimization formulation, named ANoCo that scores Anomaly by the cost of Non-Conformity of a query patch to align with a fixed normal manifold. For each query patch, we construct a bipartite query to normal graph weighted by cosine affinity, explicitly removing query-query and normal-normal edges to prevent evidence dilution. We formulate anomaly scoring as a convex Laplacian energy with anchored normal nodes, and solve in closed form. In particular, we do not use the optimized features themselves-the anomaly score is the magnitude of the update required to satisfy normality constraints, reframing the graph Laplacian as a non-conformity operator rather than a smoothing prior. The proposed method introduces no learnable parameters, message passing, or sampling, and has complexity comparable to a single linear solve. Across standard benchmarks, it delivers strong image-level AUROC, stable localization maps, and improved robustness over prior methods, demonstrating the effectiveness of using optimization-induced feature drift as anomaly measure. 

---
# VITAL: Visual-Semantic Dual Supervision for Enhanced and Interpretable Latent Reasoning in Medical MLLMs 

**Authors**: Qiaoru Li, Shaotian Liang, Jintao Chen, Haoran Sun, Yuxiang Cai, Jianwei Yin, Yankai Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28422)  

**Abstract**: Latent reasoning enables reasoning over continuous hidden states rather than explicit tokens, avoiding the language bottleneck and inference overhead of chain-of-thought for medical VQA. However, existing methods suffer from modality collapse, insufficient visual supervision, and train-inference mismatch. Moreover, their opaque latent states offer no interpretability, which is critical in clinical applications. We propose VITAL, a latent-space reasoning framework for medical MLLMs with visual-semantic dual supervision: an auxiliary text decoder reconstructs reasoning chains from latent states, while a visual projector regresses ROI features from a frozen, independent medical vision encoder. Both modules are discarded at inference with zero overhead, yet can be re-attached post-hoc for dual interpretability, providing textual and visual explanations of the reasoning process without sacrificing efficiency. We construct a 61K dataset spanning 9 imaging modalities, exceeding prior medical visual latent reasoning datasets by an order of magnitude. Experiments on 7 benchmarks show that VITAL consistently and substantially outperforms the backbone, all latent reasoning baselines, and medical MLLMs trained on far larger data, achieving state-of-the-art results competitive with trillion-parameter proprietary models. 

---
# ADWIN: Adaptive Windows for Horizon-Aware On-Policy Distillation 

**Authors**: Kun Liang, Chenming Tang, Clive Bai, Weijie Liu, Saiyong Yang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28396)  

**Abstract**: On-policy distillation (OPD) transfers reasoning behavior by training a student on teacher feedback along student-generated trajectories, but standard full-rollout training ties every update to a costly completion and can over-allocate supervision to late positions with low marginal value for the current student. We revisit this assumption through the useful supervision horizon: student-induced rollouts can drift from teacher-preferred continuations, while aligned prefixes may already preserve the long-horizon OPD update direction. We propose ADWIN, an adaptive-window framework for OPD that treats rollout length as an online admissibility decision, training on short teacher-anchored prefixes while using delayed full-rollout probes to audit prefix--full alignment and adapt the next horizon with staleness control. Across math and code reasoning benchmarks in single-task, multi-task, and strong-to-weak settings, ADWIN improves the accuracy--compute trade-off over full-rollout OPD and prefix-based baselines, reducing end-to-end training cost by up to 4.1 times while achieving comparable or better accuracy. 

---
# CLANE: Continual Learning of Actions on Neuromorphic Hardware from Event Cameras 

**Authors**: Elvin Hajizada, Michael Neumeier, Edward Paxon Frady, Yulia Sandamirskaya, Axel von Arnim, Bing Li, Eyke Hüllermeier  

**Link**: [PDF](https://arxiv.org/pdf/2605.28387)  

**Abstract**: Recognizing and continuously learning novel human actions without forgetting prior classes is a requirement for emerging AR/VR and robotics applications. For these applications, both on-device processing and learning are essential for privacy and low-latency adaptation. Event cameras address the efficiency of visual sensing with sparse, asynchronous output that is naturally compatible with neuromorphic processing. Yet no prior system has deployed a continual on-device learning pipeline for event-based action recognition using neuromorphic hardware. We present CLANE, Continual Learning of Actions on Neuromorphic Hardware from Event Cameras, deployed end-to-end on Intel Loihi 2. CLANE combines a spiking 2D CNN for spatiotemporal feature extraction with CLP-SNN as its on-chip learning head, extended to action clips via a Temporal Aggregation Layer and a fixed-point Normalization Layer, both novel Loihi 2 modules. On THU E-ACT-50, a 50-class dataset captured under real-world conditions, CLANE achieves 70.4% accuracy in a continual learning task while delivering more than 100x energy reduction and 16x lower latency over a sequential CNN+GRU+CLP edge GPU baseline, validated through iso-algorithm cross-platform benchmarking across three evaluation levels. 

---
# Score Based Error Correcting Code Decoder 

**Authors**: Alon Helvits, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2605.28358)  

**Abstract**: Error-correcting codes enable reliable communication, yet practical soft decoding remains challenging across code families and block lengths. We propose SB-ECC, a score-based decoder that casts decoding as continuous-time denoising. A neural denoiser defines a probability-flow ordinary differential equation (ODE) that iteratively updates the noisy channel observation toward a valid codeword, guided by parity constraints. The model is trained across noise levels without time/SNR conditioning, enabling inference without SNR estimation and supporting a direct latency accuracy trade off controlled by the ODE solver budget. We use the raw signed channel observation as input for learning a continuous denoising field. Across 42 code/SNR settings, SB-ECC achieves the best BER in 39/42 entries, with an average SNR gain of 0.17dB and a maximum gain of 0.46dB over the strongest competing baseline, we showed that swapping the solver from Euler to DPM preserves -ln(BER) while reducing end-to-end decoding time by 8.86% on average (up to 12.82%). 

---
# Improving Evaluation of Recombination-based Cartesian Genetic Programming 

**Authors**: Duy Long Tran, Anja Jankovic, Marie Anastacio, Holger Hoos, Roman Kalkreuth  

**Link**: [PDF](https://arxiv.org/pdf/2605.28353)  

**Abstract**: Cartesian Genetic Programming has traditionally been using mutation as its main and often sole genetic operator to drive evolutionary search. Despite advancements in recent years, recombinationbased approaches have long been avoided, due to apparent lack of performance gains. This study examines two recently suggested recombination-based operators, subgraph crossover and discrete phenotypic recombination on SRBench, a benchmarking platform for symbolic regression. Using the implementations provided in the TinyverseGP framework, we perform hyperparameter optimisation of the respective representations with these two operators. Our work demonstrates that hyperparameter optimisation can lead to improvements in performance for recombination-based Cartesian Genetic Programming. 

---
# Learning the Error Patterns of Language Models 

**Authors**: Jinwoo Kim, Taylor Berg-KirkPatrick, Loris D'Antoni  

**Link**: [PDF](https://arxiv.org/pdf/2605.28328)  

**Abstract**: When generating outputs for domains with specific validity constraints (e.g., a program should compile), LLMs often fail in a small number of focused ways: for example, by using Python function names when generating TypeScript. We observe that these error patterns can be represented using a small number of constraints that can be learned in practice. We propose \emph{prefix filters}, which are per-domain-and-LLM symbolic functions, as objects to capture the error patterns, Palla as an algorithm to learn prefix filters efficiently in practice, and implement Palla. Prefix filters learned by Palla i) help us quantitatively analyze the error patterns of LLMs, and ii) can be used to constrain the outputs of a model via constrained sampling algorithms. For example, Palla boosts compile rates for Qwen2.5-1.5B on TypeScript generation, by over 60%, allowing Qwen2.5-1.5B to achieve similar performance to Llama3.1-8B unconstrained. 

---
# Multi-Agent LLM-based Metamorphic Testing for REST APIs 

**Authors**: Shehroz Khan, Abdullah Mughees, Gaadha Sudheerbabu, Tanwir Ahmad, Dragos Truscan  

**Link**: [PDF](https://arxiv.org/pdf/2605.28321)  

**Abstract**: As REST APIs become an increasingly significant part of software systems, their validation is becoming more critical. Hence, testing and uncovering underlying issues are of utmost importance for improving software quality. However, testing REST APIs is challenging mainly due to the difficulty of assessing whether the output of an API call is correct, i.e., the test oracle problem. Metamorphic testing is a specification-based testing approach for situations where correct outputs are unknown or not specified explicitly. To check the correctness of a system, relations between the different outputs are specified. We present ARMeta, a tool-supported approach that uses an LLM-based multi-agent workflow to support metamorphic testing of REST APIs documented with OpenAPI. The agentic workflow is used to identify metamorphic test scenarios and specify them in the Given-When-Then format. These scenarios are automatically implemented as executable tests and executed against the system under test. We evaluate ARMeta on two publicly available web applications that expose REST interfaces and compare its performance with a scenario-based testing baseline. The results show that ARMeta explores behaviors that serve as a complement to existing scenario-based testing approaches. 

---
# Identifying Explicit Parsimonious Piece-wise Polynomial Relationships in Industrial time-series: Application to manipulator robots 

**Authors**: Mazen Alamir, Sacha Clavel  

**Link**: [PDF](https://arxiv.org/pdf/2605.28320)  

**Abstract**: This paper addresses the problem of identifying parsimonious explicit piece-wise polynomial relationships that might involve a relatively large number of raw features. The algorithm leverages a recently proposed identification algorithm that yields parsimonious implicit relationships enabling to derive normality characterization in the context of anomaly detection and localization. The algorithm proposed in this paper goes a step further by deriving explicit piece-wise representations that are built using the set of polynomials involved in the implicit representations. The framework is illustrated on the problem of identifying parsimonious explicit representations of the inverse model of a 6-axis manipulator robot. Moreover, further experiments on a 4-axis robot are also shown which are designed to investigate the generalization capability of parsimonious models compared to state-of-the-art DNNs structures, when models face unseen contexts of use. 

---
# Hybrid Neural World Models 

**Authors**: Pranav Lakshmanan, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2605.28317)  

**Abstract**: Neural surrogates promise large speedups over classical solvers for physical dynamics but fail silently at sharp dynamical events such as shocks, fronts, and contact. We present hybrid neural world models for physical dynamics: a recipe for training and deploying multi-horizon surrogates in physical state space, where a single network with continuous horizon conditioning is trained with direct supervision against textbook reference solvers to predict any future state at horizon T in one forward pass. Although no part of the training data, loss function, or architecture supervises discontinuity location, the trained surrogate encodes it implicitly, recoverable from its forward passes alone as a per-trajectory error map that concentrates on shocks, fronts, and contacts, and stays small elsewhere. The map is competitive with or better than standard label-free baselines including deep ensembles, learned error heads, gradient-magnitude indicators, and locally-adaptive conformal prediction, while using only a single trained network and requiring no calibration set or governing-equation knowledge. The recipe supports two operating points. Mode 1 runs the surrogate alone for maximum throughput, with same-hardware CPU speedups of 26x to 72x against textbook solvers on the PDE environments. Mode 2 uses the error map to gate a reference-solver fallback, deferring uncertain trajectories and roughly halving the surrogate's residual error at the default operating point. The recipe applies without modification across reaction-diffusion, compressible Euler, and rigid-body collision dynamics. 

---
# Routing-Aligned Fine-Tuning for Multilingual Downstream Tasks in Mixture-of-Experts Models 

**Authors**: Guanzhi Deng, Kuan Wu, Haibo Wang, Shing Yin Wong, Sichun Luo, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.28306)  

**Abstract**: Mixture-of-Experts (MoE) models have emerged as a dominant paradigm for efficient LLM scaling, yet adapting them to non-English downstream tasks remains challenging. Existing fine-tuning approaches treat MoE models as monolithic learners, ignoring the heterogeneous routing structure that develops during pretraining. We validate across multiple MoE models and downstream tasks that middle layers form a language-universal alignment zone where routing divergence strongly predicts per-language task performance gaps. Building on this observation, we propose RA-MoE (Routing-Aligned MoE Fine-Tuning), a three-stage framework that categorizes parallel task examples into a four-way taxonomy (cc/ci/ic/ii) based on correctness in English and the target language, identifies task-relevant experts in the middle layers, and augments standard SFT with a routing alignment loss that encourages target-language routing on ci-type examples to follow the English task-expert activation pattern. Experiments across three MoE models, three tasks, and six target languages demonstrate that RA-MoE consistently outperforms standard SFT and strong baselines including Routing Steering and RISE, with the ci proportion of a task-language pair serving as a reliable predictor of alignment benefit. 

---
# Revisiting Anthropomorphic Reflection Markers in Large Language Model Reasoning 

**Authors**: Yahan Yu, Noa Nakanishi, Fei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.28305)  

**Abstract**: Large Language Models (LLMs) often produce explicit reflective traces during complex reasoning, accompanied by anthropomorphic markers such as wait, hmm, and alternatively. Although these markers are commonly used as visible indicators of reflection, their mechanisms remain unclear, which leaves the risk of overthinking associated with redundant and repetitive reflection markers. In this work, we revisit anthropomorphic reflection markers, examining their necessity for reasoning and role in the reflection. We suppress these markers through prompt-level and token-level interventions, and analyze their effects on task performance across four benchmarks and two model scales. Our results show that anthropomorphic markers are not uniformly necessary for reasoning performance: suppressing them can preserve or improve performance in several settings, especially under larger sampling budgets. Meanwhile, marker suppression does not necessarily remove reflection behavior, as models can still perform marker-free verification. These suggest that anthropomorphic markers tend to be surface cues rather than reliable proxies for reflection itself, and motivate future research on reasoning mechanisms beyond explicit marker patterns. 

---
# How Far Can Disaggregation Go? A Design-Space Exploration of Attention-FFN Disaggregation for Efficient MoE LLM Serving 

**Authors**: Hanjiang Wu, Abhimanyu Rajeshkumar Bambhaniya, Sarbartha Banerjee, Tuhin Khare, Sudarshan Srinivasan, Suvinay Subramanian, Souvik Kundu, Madhu Kumar, Midhilesh Elavazhagan, William Won, Amir Yazdanbakhsh, Tushar Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2605.28302)  

**Abstract**: Modern large language model (LLM) inference has progressively disaggregated to keep pace with growing model sizes and tight TTFT and TPOT service-level objectives: from chunked-prefill aggregation, to prefill-decode (P/D) disaggregation, and most recently to operator-level Attention-FFN Disaggregation (AFD). This trend is especially important for mixture-of-experts (MoE) models, where memory-bound attention, compute-intensive expert FFNs, and MoE dispatch/combine communication create distinct resource demands. AFD further exposes this heterogeneity by placing attention and MoE-FFN execution on separate GPU groups. Each level of disaggregation deepens the scheduling design space across workload characteristics, resource allocation, and interconnect topology, raising the central question: when does each level actually pay off? We systematically characterize this trade-off for MoE inference across realistic workloads spanning input/output sequence lengths, prefix-KV reuse, and per-user latency constraints. Using chunked-prefill and P/D disaggregation as baselines, we study the benefits and limits of AFD at scale through a framework that fuses on-device kernel measurements with high-fidelity network simulation. Under strict TTFT/TPOT SLOs, AFD sustains around 4k tokens/s of system throughput on DeepSeek-V3.2 across chat, coding, and agentic-coding workloads, where non-AFD deployments are infeasible. We distill concrete takeaways for jointly optimizing throughput and interactivity, including how to partition attention and FFN across GPUs as a function of workload and model architecture, providing design principles for current rack- and cluster-scale deployments as well as future disaggregated AI infrastructure. 

---
# ProRL: Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation 

**Authors**: Hongru Hou, Tiehua Mei, Denghui Geng, Jinhui Huang, Ao Xu, Hengrui Chen, Jiaqing Liang, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28293)  

**Abstract**: Proactive Recommender Systems (PRSs) aim to guide user preference shift toward target items by generating paths of intermediate recommendations. Reinforcement learning (RL) provides a principled framework for optimizing such sequential decision tasks, as path rewards can naturally capture both short-term acceptance and long-term guidance effectiveness. However, naively applying policy gradients to PRS results in deficient gradient estimation. We identify two deficiencies: (1) path-level rewards decompose into step-level rewards with positive mean, creating a length-dependent bias that causes gradients to favor path extension over meaningful exploration; (2) weighting each step by the entire path-level reward ignores the decomposition structure, leading to high gradient variance. To rectify these two deficiencies, we propose an effective RL framework ProRL with two novel mechanisms for proactive recommendation. First, Stepwise Reward Centering subtracts expected rewards to neutralize length-dependent bias, ensuring that path extension yields zero expected gradient signal. Second, Position-Specific Advantage Estimation leverages the reward decomposition structure to compute step-dependent baselines, reducing gradient variance. Together, these mechanisms yield policy gradients that precisely target path quality. Our experiments on three real-world datasets demonstrate that ProRL significantly outperforms state-of-the-art PRSs. Our code is available at this https URL. 

---
# PrunePath: Towards Highly Structured Sparse Language Models 

**Authors**: Zhexuan Gu, Zixun Fu, Yancheng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2605.28283)  

**Abstract**: Feed-forward networks (FFNs) dominate the parameter count and computation of modern language models, yet existing pruning methods often struggle to convert sparsity into hardware-friendly inference efficiency gains. We introduce \textbf{PrunePath}, a budget-adaptive structured sparsification framework for FFN layers. Built on MoEfication, PrunePath replaces independent expert-wise thresholding with a softmax-normalized routing distribution and activates important experts under a cumulative-mass threshold. This formulation imposes a token-level probability budget, enabling adaptive expert counts and a direct inference-time sparsity knob from a single checkpoint. Across NLU, NLG, and instruction-tuning evaluations, PrunePath achieves a favorable sparsity--performance trade-off compared with existing static pruning and MoEfication-based methods. We further implement Triton kernels for KV-cache decoding to translate the resulting structured sparsity into practical memory savings and measurable decoding-speed improvements. These results demonstrate the superior performance of PrunePath for building highly sparse, deployment-friendly large language models. 

---
# GUI Agents for Continual Game Generation 

**Authors**: Yixu Huang, Bo Li, Na Li, Zhe Wang, Kaijie Chen, Haonan Ge, Qingyi Si, Yuanzhe Shen, Ruihan Yang, Guangjing Wang, Hongcheng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.28258)  

**Abstract**: Generating a game is not the same as making one that can be played. Despite advances in code generation, existing approaches treat game generation as one-shot translation from prompt to artifact, leaving interaction-level failures undetected. We argue that evaluating and improving game generation requires a player, and study two roles for graphical user interface (GUI) agents in this process: (1) as an objective evaluator, for which we introduce PlaytestArena, a new evaluation environment that pairs 200 browser-based game generation tasks across eight genres with rubrics of expected in-play behaviors, adjudicated by a GUI agent that loads each build in a browser and plays it; and (2) as a subjective playtester, for which we propose Play2Code, where a game agent and a GUI agent operate in a sustained loop with shared memory, turning game generation into a dialogue between coding and playing. Our experiments show that even frontier models struggle to generate playable games directly, while Play2Code achieves a 66.8\% rubric pass-rate, improving over single-pass and agentic-coding baselines by 37.1 and 14.6 points respectively. Further analysis shows that GUI playtester feedback is more traceable than a human report, yet idiosyncratic in ways reminiscent of human testers, establishing game playtesting as a critical testbed for interactive code generation. Our project website is available at this https URL. 

---
# IRDS: Interpretable RLVR Data Selection via Verifier-Coupled Sparse Autoencoder Coverage 

**Authors**: Yuhan Li, Mingxu Zhang, Dazhong Shen, Ying Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.28247)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a key technique for en- hancing LLM reasoning, yet its data ineffi- ciency remains a major bottleneck. Existing methods address this problem only partially, each missing at least one of subset-level cov- erage, verifier signal use, or interpretability. To address this gap, we present IRDS (Inter- pretable RLVR Data Selection), which selects RLVR training instances on a sparse autoen- coder (SAE) cluster basis so the selection itself is auditable on recognizable problem motifs. To select instances the model both fails on and can still learn from, we introduce a verifier- coupled coverage objective on the SAE basis and solve it by greedy log-determinant max- imization. Experiments on three instruction- tuned models and six math reasoning bench- marks show that IRDS achieves the highest overall accuracy, exceeding the strongest base- line by +3.9/+4.0 pp on the two Qwen models and by +0.5 pp on Llama-3.1-8B, while run- ning an order of magnitude cheaper than the trajectory-based baseline. 

---
# VidPrism: Heterogeneous Mixture of Experts for Image-to-Video Transfer 

**Authors**: Rui Lin, Chuanming Wang, Huadong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.28229)  

**Abstract**: With the rapid development of pre-training technologies, adapting large-scale Vision-Language Models (VLMs) for video understanding \emph{\ie} image-to-video transfer learning has become a dominant paradigm. To achieve superior performance, it raises as an effective strategy among recent advances to employ Mixture-of-Experts (MoE) to enhance VLMs' temporal modeling capabilities. However, conventional MoE designs suffer from expert homogenization, where all experts act as identical generalists, inefficiently learning spatio-temporal features from undifferentiated video streams. To overcome this problem, we propose VidPrism, a novel heterogeneous temporal Mixture-of-Experts framework. VidPrism pioneers a division of labor by deploying functionally specialized experts, each assuming a role ranging from spatial understanding to temporal modeling. To feed these specialists appropriately, we introduce a content-aware, multi-rate sampling module that dynamically generates streams ranging from semantically rich to motion-focused representations, providing specialized inputs for experts. Furthermore, a dynamic, bidirectional fusion mechanism enables synergistic information exchange between these pathways, leading to a comprehensive video representation. Extensive experiments on various video recognition benchmarks demonstrate that VidPrism achieves state-of-the-art performance and effectively fosters expert specialization. Our source code is available at \href{this https URL}{this https URL}. 

---
# SmartIterator: Visual Analytics Workflows for Supervising Unsupervised Data Grouping 

**Authors**: Gennady Andrienko, Natalia Andrienko  

**Link**: [PDF](https://arxiv.org/pdf/2605.28219)  

**Abstract**: Unsupervised learning methods -- topic modeling, partition-based and density-based clustering -- produce data groupings without human guidance, yet choosing and evaluating those groupings should not itself be unsupervised. We present \emph{SmartIterator}~(SI), a visual analytics approach that treats the full sequence of grouping results across a parameter sweep as a first-class analytical object. For each method family, SI provides a structured six-phase workflow that guides the analyst through systematic exploration of grouping results -- from quality-metric overview through transition-stability assessment, membership-confidence evaluation, content and context inspection, and recurrent-archetype verification to an informed decision -- building cumulative understanding of data structure along the way. The workflows are operationalized through \emph{IteraScope}~(IS), a coordinated visual display combining quality-metric charts with semantic color encoding, a 1D group embedding with Sankey-style transition flows and violin plots of membership confidence, a 2D group embedding with HDBSCAN-detected recurrent archetypes that highlights iterations capturing all persistent patterns, and domain-specific linked views for contextualized interpretation. We demonstrate the three workflows on: (1)~simulated social-media messages from the VAST Challenge 2011 (density-based clustering, validated against ground truth), (2)~EU population statistics across ${\sim}1\,500$ NUTS-3 regions (partition-based clustering), and (3)~30 years of IEEE VIS papers (NMF topic modeling). The workflows constitute the main contribution: they provide actionable, method-specific guidance for navigating parameter spaces, studying how data structure evolves across configurations, and grounding analytical understanding in domain context -- yielding knowledge about the data that no single ``best'' result can provide. 

---
# Pruning and Distilling Mixture-of-Experts into Dense Language Models 

**Authors**: Junhyuck Kim, Jihun Yun, Haechan Kim, Gyeongman Kim, Joonghyun Bae, Jaewoong Cho  

**Link**: [PDF](https://arxiv.org/pdf/2605.28207)  

**Abstract**: Mixture-of-Experts (MoE) is now the dominant architecture for frontier language models, yet it requires all expert parameters to be loaded in memory, making it less preferable for memory-constrained deployment. Existing compression methods reduce the number of experts but the output remains an MoE model with the same fundamental limitation. We present the first systematic framework for converting a trained MoE into a standard fully dense architecture: experts are scored, selected, and grouped, then concatenated into a dense FFN and refined by knowledge distillation from the MoE teacher. We evaluate 7 scoring, 5 grouping, and 2 magnitude scaling methods across a range of selected expert counts on Qwen3-30B-A3B, yielding 350 configurations. We find that the choice of scoring method is the most impactful, with our novel diversity-aware scoring consistently outperforming prior methods on Qwen3-30B-A3B, DeepSeek-V2-Lite, and GPT-OSS-20B. Under a controlled comparison at matched parameter count, MoE-to-dense outperforms dense-to-dense pruning by +6.3 pp in average downstream accuracy after ~4B-token distillation at 1.6x faster training wall-clock speed. 

---
# Whose Name Comes Up? III: Persona Prompting Effects in LLM-Based Scholar Recommendation 

**Authors**: Annabella Sánchez-Guzmán, Lukas Eberhard, Denis Helic, Lisette Espín-Noboa  

**Link**: [PDF](https://arxiv.org/pdf/2605.28187)  

**Abstract**: Large language models (LLMs) are increasingly used as scholar recommenders, shaping who is seen as an expert in academia. Existing audits remain English-centric, single discipline, and persona-agnostic, leaving the source of output variability poorly understood. To this end, we propose a benchmark that disentangles the effects of model choice and prompt design on recommendations. We audit 43 LLMs by varying persona prompts (language, location, role-and-task) and context (field, seniority, k). Recommended scholars are compared against Semantic Scholar over six scientific disciplines to measure technical quality (factuality, coverage) and social representativeness (diversity, parity). Basic technical quality is driven by model choice, factuality and parity by context, and diversity by location. South Africa prompts yield less factual lists, while Japan prompts yield highly factual but homogeneous lists skewed toward highly productive scholars. Prompt design is thus a non-trivial axis of LLM-based scholar discovery and should be systematically audited alongside model choice. 

---
# Visualizing Latent Phase Structures in Locomotion Policies: A Multi-Environment Study with Temporal Feature Extension 

**Authors**: Daisuke Yasui, Toshitaka Matuki, Hiroshi Sato  

**Link**: [PDF](https://arxiv.org/pdf/2605.28186)  

**Abstract**: Deep reinforcement learning (DRL) has been shown to achieve high performance on locomotion control tasks in MuJoCo benchmarks such as HalfCheetah, Ant, and Walker2D. However, visualizing the motion structures internally obtained by a trained policy function implemented as a deep neural network remains challenging. It is known from biomechanics and related fields that locomotion control is realized through the repetition of motion phases such as the stance phase and swing phase. In this study, we propose a framework for uncovering latent motion phase structures from trajectories generated by locomotion control policies through interaction with the environment. The proposed method extends the clustering features from state observations alone to augmented features including actions, next states, and next actions, and introduces a method for determining the number of clusters that suppresses self-transitions. Applying the proposed method to three environments -- Ant-v5, HalfCheetah-v5, and Walker2D-v5 -- we successfully identified phase structures with clearer and more regular transition rules than those obtained by the existing method. 

---
# BenGER: Benchmarking LLM Systems on Subsumption-Based Legal Reasoning in German Law 

**Authors**: Sebastian Nagl, Ann-Kristin Mayrhofer, Martin Heidebach, Aleyna Koçak, Anne Zettelmeier, Elly Breu, Angelina Greiner, Sofija Milijas, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2605.28183)  

**Abstract**: We introduce the BenGER (Benchmark for German Law) dataset for evaluating LLM systems on subsumption-based legal reasoning in German law. The BenGER dataset consists of three components: 596 exam-style free-text legal case tasks across multiple levels of legal education and 531 short doctrinal reasoning tasks. We evaluate 12 contemporary LLM systems -- closed flagship, efficiency-oriented, and open-weight -- across automatic and judge-based metrics. On a controlled validation subset of timed human-written solutions under both unaided and human--AI co-creation conditions, we contextualise model performance against these human baselines. We introduce a rubric-aligned LLM-as-a-Judge framework cross-validated against a multi-rater human-grading protocol (three blind reviews plus one author-informed creator review per solution). Our results show that replacing a blind human reviewer with the LLM judge degrades agreement with the full human pool no more than removing that reviewer altogether (Calderon r=0.96 vs.~r=0.96, matched n=30), that closed-flagship systems lead the leaderboard across all corpora, and that human--AI co-creation substantially outperforms unaided human work. 

---
# FLORO: A Multimodal Geospatial Foundation Model for Ecological Remote Sensing Across Sensors and Scales 

**Authors**: Jorge L. Rodriguez, Victor Angulo Morales, Areej Alwahas, Mariana Elias Lara, Fida Mohammad Thoker, Kasper Johansen, Bernard Ghanem, Fernando T. Maestre, Matthew F. McCabe  

**Link**: [PDF](https://arxiv.org/pdf/2605.28174)  

**Abstract**: Foundation models offer a promising route to transferable remote sensing representations, but many current approaches depend on very large pretraining datasets and fixed sensor configurations, limiting their suitability for ecological and environmental applications, where observations often vary across platforms, spatial and spectral resolutions, and available modalities. We introduce FLORO, a multimodal geospatial foundation model designed to learn transferable representations from a small but highly diverse remote sensing corpus. FLORO is pretrained using masked autoencoding on a heterogeneous combination of Sentinel-1, Sentinel-2, SkySAT imagery, elevation, and UAV-derived data. To accommodate sensor variability, FLORO incorporates availability-aware inputs that indicate which spectral bands and auxiliary modalities are present in each sample, enabling a unified input space across heterogeneous sensor configurations. We evaluated FLORO on the PANGAEA benchmark under a frozen-encoder protocol across scene classification, segmentation, and regression tasks. Despite being pretrained on a smaller corpus than competing foundation models, FLORO achieved strong and stable transfer across optical, optical-SAR, and optical-elevation benchmarks spanning medium-resolution satellite, airborne, and ultra-high-resolution UAV imagery. FLORO obtained the second-best average segmentation performance across six PANGAEA benchmarks, trailing only a recently introduced foundation model pretrained on over two orders of magnitude more images, remained competitive on scene classification, and was robust in regression tasks, while qualitative results showed improved preservation of spatial structure in flood, urban, biomass, and canopy-height prediction settings. In a separate controlled experiment on EuroSAT-MS, geo-positional encoding further improved classification relative to absolute positional encoding. 

---
# QuITE: Query-Based Irregular Time Series Embedding 

**Authors**: JungHoon Lim  

**Link**: [PDF](https://arxiv.org/pdf/2605.28166)  

**Abstract**: Irregular Multivariate Time Series (IMTS) are common in practice, yet their irregular sampling complicates effective modeling. Existing approaches typically either (i) design specialized architectures that limit the reuse of proven Multivariate Time Series (MTS) models, or (ii) map IMTS onto regular temporal grids through interpolation, which may distort temporal dynamics by introducing artificial values. To address these limitations, we propose a new input-embedding-based approach. We identify that the key bottleneck lies not in the backbone architecture, but in conventional embedding layers that assume uniform sampling. In this work, we introduce QuITE (Query-Based Irregular Time Series Embedding), a simple yet effective plug-and-play embedding module for IMTS. QuITE employs learnable query tokens to aggregate irregular observations through a single self-attention layer, directly producing backbone-compatible latent representations without artificial value generation or architectural modification. Extensive experiments on real-world benchmarks show that QuITE consistently improves MTS models, yielding average relative gains of up to $54.7\%$ in forecasting and $15.8\%$ in classification across diverse datasets and backbone architectures. Code is available at: this https URL. 

---
# Performance and Explainability Requirements of Evolutionary Algorithms in Real-World Physics-Informed Optimization 

**Authors**: Helena Stegherr, Michael Heider, Nils Meyer, Tobias Thummerer, Thomas Wendler, Pierre Aublin, Ennio Idrobo-Àvila, Lars Mikelsons, Sebastian Zaunseder, Jörg Hähner  

**Link**: [PDF](https://arxiv.org/pdf/2605.28164)  

**Abstract**: Evolutionary computation offers a variety of tools to solve complex real-world optimization problems. However, research often focuses on smaller, simplified problems and optimization algorithms that sometimes miss expectations in real-world scenarios. Additionally, trust in the applied algorithm and the solutions it provides is often essential in such settings, but requires an understanding of the search process itself. This leads to evolutionary computation often not being seriously considered by practitioners in many application contexts, among them physics-based modeling. In this article, techniques from evolutionary computation are detailed that can alleviate these problems. First, five real-world physics-based optimization problems are introduced and described by domain experts. For each of these, the requirements for the evolutionary algorithm regarding performance and explainability to increase trust and usability are presented. We found that all domain experts expect fast convergence to a good solution and want some explanations for how the results were formed, while other requirements strongly depend on the respective problem. Finally, we present existing approaches that can be leveraged to improve those aspects of evolutionary algorithms but have to our knowledge never been employed in complex real-world scenarios. This implies a gap between both domains that needs to be closed to exploit the full potential of evolutionary computation. 

---
# DEPART: DEcomposing PARiTy across Multilingual LLMs 

**Authors**: Manan Uppadhyay, Prashant Kodali, Pranjal Chitale, Reshma Ramaprasad, Himanshu Beniwal, Sunayana Sitaram  

**Link**: [PDF](https://arxiv.org/pdf/2605.28163)  

**Abstract**: Multilingual Large Language Models (mLLMs) leaderboards report per-language accuracy but rarely explain why disparities emerge, leaving systemic biases unattributed and offering practitioners no actionable levers. We first establish that these gaps are systematic rather than artifacts of sampling noise via distribution-free Friedman and Kruskal--Wallis tests, then introduce a two-step Bayesian hierarchical framework that decomposes multilingual performance variance into interpretable components. First, isolating the variance attributable to language identity, we show that observable language features (script, family, typological distance) explain $R^2_{\text{ling}} = 79\%$ of this variance on understanding tasks and $92\%$ on reasoning, with a model's internal representational similarity to English emerging as the dominant predictor across both task buckets. Second, decomposing the full (model$\times$benchmark$\times$language) cube, we find that NLU and reasoning have fundamentally divergent variance profiles: model identity dominates understanding ($66.7\%$ of variance), whereas the benchmark$\times$model interaction dominates reasoning ($46.3\%$). Together these results recast multilingual evaluation from passive performance mapping into an explainable, diagnostic framework with concrete levers for targeting the root drivers of language disparity. 

---
# DeltaMCP: Incremental Regeneration via Spec-Aware Transformation for MCP servers 

**Authors**: Aditya Pujara, Xiaogang Zhu, Hsiang-Ting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.28148)  

**Abstract**: The rapid development of LLMs coupled with the introduction of Model Context Protocol (MCP) has revolutionized how intelligent agents interact with APIs through deterministic and structured methods \cite{ModelContextProtocolIntro2025}. While some existing systems like AutoMCP attempt to automate a previously completely manual process of generating MCP servers, they fail to address the recurring challenge of maintaining synchronization between evolving enterprise-level APIs and their corresponding MCP toolset implementation \cite{mastouri2025makingrestapisagentready}. This paper introduces DeltaMCP, a specification-aware, incremental regeneration tool for enterprise-grade MCP servers. DeltaMCP enables developers to only update the affected tooling of MCP servers, given a new release of it's corresponding service's OpenAPI specification. Using Azure REST API specifications as the evaluation dataset, DeltaMCP is benchmarked against baseline full generation methods on generation quality and system performance. The results demonstrate the reduction in developer overhead through DeltaMCP whilst improving maintainability and version consistency. This research offers a scalable approach for enterprises seeking to maintain high-fidelity, up-to-date MCP server infrastructures for LLM-based systems. 

---
# SNARE: Adaptive Scenario Synthesis for Eliciting Overeager Behavior in Coding Agents 

**Authors**: Yubin Qu, Yi Liu, Gelei Deng, Yanjun Zhang, Yuekang Li, Ying Zhang, Leo Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28122)  

**Abstract**: A coding agent executes a benign task as a sequence of shell, file, and network actions, any of which can quietly exceed the authorized scope while the task still completes. We call this overeager behavior: the prompt is not adversarial and the run succeeds, yet an out-of-scope step can leak credentials or delete files. Existing benchmarks miss it: task-completion suites credit any finished run, jailbreak suites probe adversarial prompts, and the one prior overeager benchmark applies a single fixed prompt set to every agent-model pair, leaving its easiest and most resistant pairs under-measured. We present SNARE (Synthesizing Non-adversarial scenarios for Adaptive Reward-guided Elicitation), a pipeline that composes benign scenarios from reusable scope and trap fragments, scores each run with a judge-free oracle flagging trap-pattern matches and unsolicited file additions or deletions, and uses Thompson sampling to steer each pair's run budget toward the scenarios that most often trigger it. Instantiating it over 24 overeager archetypes yields OverEager, which we run across a 4x5 matrix of four coding agents and five base models. Across 10,000 benign runs, 19.51% trigger overeager behavior, with per-pair rates spanning 11.9x. This variation is driven by the agent framework, not the model: the framework accounts for 56% of it against the model's 21%, so any single-framework or single-model evaluation undercounts the matrix by about a fifth. 

---
# LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning 

**Authors**: Zerui Chen, Qinggang Zhang, Zhishang Xiang, Zhimin Wei, Linfeng Gao, Xiao Huang, Zhihong Zhang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.28120)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) advances flat document retrieval by structuring knowledge as relational graphs, enabling more coherent and effective reasoning. However, applying it to specific domains like legal reasoning faces critical challenges. (i) Legal corpora are heterogeneous, containing multi-granular knowledge from cases, articles and interpretations. A flat knowledge graph cannot adequately differentiate between factual details, applied rules, and abstract principles, limiting accurate retrieval. (ii) Reliable legal judgment demands transparent, evidence-based reasoning. Traditional RAG passes retrieved context directly to an LLM without verification, resulting in opaque, error-prone reasoning. To this end, we propose LegalGraphRAG, a framework designed for reliable legal reasoning. Our approach introduces two core components: a hierarchical legal graph that hierarchically organizes legal sources to enable retrieval at appropriate abstraction levels, and a multi-agent system for reliable legal reasoning, where a Researcher retrieves candidate evidence, an Auditor rigorously verifies its validity against source documents, and an Adjudicator synthesizes the set of verified evidence to render a final judgment. Extensive experiments show that LegalGraphRAG achieves the state-of-the-art performance, outperforming existing GraphRAG baselines in accurate and trustworthy legal analysis. Our code, datasets and implementation details are available at this https URL. 

---
# MIRAGE: Context-Aware Prompt Injection against Mobile GUI Agents via User-Generated Content 

**Authors**: Ruoqi Guo, Yi Liu, Gelei Deng, Yiheng Xiong, Yuekang Li, Ying Zhang, Leo Yu Zhang, Lida Zhao, Ji Jie, Yuxiao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28116)  

**Abstract**: Mobile graphical user interface (GUI) agents driven by vision-language models (VLMs) perceive the screen as rendered pixels and choose actions from what they see, so they cannot reliably separate trusted interface elements from user-generated content. We present MIRAGE (Mobile Injection of Realistic Adversarial GUI Examples), a pipeline that turns benign mobile screenshots into prompt-injection samples by placing attacker-controlled text into ordinary user-generated content regions, without modifying the agent, the application, or the operating system. MIRAGE operates in three stages: a Localizer identifies user-controllable regions on the screenshot, a Generator synthesises context-aware payloads and renders them in the application's native style, and a Curator moderates realism and balances the samples across applications, region types, and attack intents. A key challenge is that an injected screenshot must stay visually indistinguishable from genuine user content while still diverting the agent; we address this by separating the stages that control reach, realism, and distributional balance. On a 1,111-sample benchmark spanning ten applications and eleven attack intents, all five evaluated VLM agents are vulnerable, with attack success rates of 23%-30%, and MIRAGE scores higher on human realism ratings than the strongest prior attack (3.02 versus 2.52 out of 5). We further find that per-sample realism and attack success are uncorrelated, so visual-quality filtering alone cannot reliably defend against this threat. 

---
# EigeNet: Geometry-Informed Multi-Modal Learning for Few-shot Novel View RIR Prediction 

**Authors**: Chong Jing, Zitong Lan, Junan Zhang, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.28101)  

**Abstract**: Predicting spatially varying Room Impulse Response (RIR) from sparse observations is a critical but highly challenging inverse problem for immersive spatial audio rendering. In this work, we present EIGENET, a geometry-informed multi-modal framework for few-shot novel view RIR prediction. At its core is a Cross-view Alternate-attention Transformer that iteratively refines local intra-view acoustic structures and global cross-view spatial relationships. We empirically demonstrate that this architecture is capable of making full use of the multi-view multi-modal context while performing spatial-temporal reasoning for RIR prediction. Inspired by acoustic ray tracing, we design a geometry-informed modulation block to formulate the connection between geometric features and RIR power spectrum. In the mean time, an auxiliary loss is introduced to transform the single-target waveform prediction into a multi-task learning framework. Through ablation studies, we demonstrate that this design yields consistent performance gains regardless of the underlying backbone, thereby confirming its foundational utility and architecture-agnostic generalizability for RIR prediction task. Evaluated on both simulated and real-world benchmarks, EIGENET achieves both state-of-the-art performance in few-shot novel view RIR prediction and sim-to-real generalization. Codes and checkpoints are available on this https URL. 

---
# Revisiting Change Detection Methods for their Application to Serac Fall Time-Lapse Monitoring 

**Authors**: Arthur Dérédel, Carlos Crispim-Junior, Pierre Lemaire, Johan Berthet, Laure Tougne Rodet  

**Link**: [PDF](https://arxiv.org/pdf/2605.28100)  

**Abstract**: In an era where climate change aggravates environmental uncertainties, the identification and detection of event precursors are becoming crucial to mitigate the impacts of disastrous natural hazards. While classical sensors such as interferometric lasers or seismometers are reliable, their widespread deployment is often hindered by logistical and economic barriers, leaving numerous blind spots. Time-lapse cameras, which already provide cost-effective, high-resolution visual context to such sensors, present a promising alternative. However, processing their output automatically faces significant challenges, notably linked to extreme shape and lighting variations. Overcoming those issues is essential to deploy them at large-scale as a monitoring tool. This paper introduces a novel sub-task of change detection, namely volumetric change detection, applied to time-lapse cameras and slope instabilities. We conduct a comprehensive review of state-of-the-art change detection methods and related tasks, analyze their core components and assess their applicability to this context. To that end, we introduce the new dataset SeracFallDet, which contains serac fall annotations and has been thoroughly annotated to meet the latter demand. Through generalization experiments, we demonstrate that dense and semi-dense feature matching, although not trained specifically for this task, exhibit robust performance. Alternatively, supervised approaches struggle with data scarcity and annotation imbalance. This suggests that hybrid methods may offer a path forward by leveraging the strengths of both tasks. These findings highlight the potential of feature matching techniques and the need for further innovation to overcome the challenges of real-world deployment in environmental monitoring. 

---
# SMILE-Next: Teaching Large Language Models to Detect, Classify, and Reason about Laughter 

**Authors**: Lee Jung-Mok, Kim Sung-Bin, Joohyun Chang, Lee Hyun, Tae-Hyun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2605.28084)  

**Abstract**: Laughter is a complex social signal that conveys communicative intent beyond amusement. While prior work has focused on isolated laughter analysis tasks, a comprehensive understanding of laughter in real-world scenarios remains underexplored. Therefore, we introduce SMILE-Next, a dataset for real-world laughter understanding with multimodal textual representations and question-answer annotations across three tasks: laughter detection, laughter type classification, and laughter reasoning. Building upon SMILE-Next, we aim to develop a laughter-specialized large language model capable of nuanced understanding of laughter in real-world contexts. To this end, we propose two key components: laughter-specific Self-Instruct and the Mixture-of-Laugh-Experts (MoLE) framework. Laughter-specific Self-Instruct enhances generalization across tasks and domains by automatically synthesizing diverse laughter-centric instructions. MoLE introduces a task-adaptive expert routing mechanism that dynamically selects specialized experts tailored to each laughter-related task, improving task-specific performance and efficiency. Experimental results show that the combination of our proposed components substantially outperforms multimodal LLM baselines, advancing robust real-world laughter understanding. Project page is at: this https URL. 

---
# Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy 

**Authors**: Huikang Liu, Aras Selvi, Wolfram Wiesemann  

**Link**: [PDF](https://arxiv.org/pdf/2605.28078)  

**Abstract**: We design a class of additive noise mechanisms that satisfy \((\varepsilon, \delta)\)-differential privacy (DP) for scalar, real-valued query functions with known sensitivities, with a particular focus on moderate and low-privacy regimes. These mechanisms, which we call \textit{mixture mechanisms}, are constructed by mixing multiple Gaussian distributions that share the same variance but differ in their means and mixture weights. The resulting distributions can be interpreted as convex combinations of a zero-mean Gaussian (as used in the analytic Gaussian mechanism) and additional Gaussians whose means depend on the sensitivity of the query function. We derive tight conditions on the variances required for \((\varepsilon, \delta)\)-DP and provide efficient algorithms to compute them. Compared to the analytic Gaussian mechanism, our mechanisms yield substantially lower expected noise amplitudes (\(l_1\)-loss) and variances (\(l_2\)-loss for zero-mean distributions). In the low-privacy regime that motivates our design, our mechanisms approach optimality, mitigating nearly all of the optimality gap of the analytic Gaussian mechanism. 

---
# StoryLens: Preference-Aligned Story Rewriting via Context-Aware Narrative Enrichment 

**Authors**: Hanwen Cui, Yuting Mei, Yuhang Fu, Dingyi Yang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.28073)  

**Abstract**: Story rewriting aims to adapt existing narratives to diverse reader preferences while preserving plot consistency and narrative coherence. Unlike conventional work on style transfer, we argue that effective story rewriting demands context-aware narrative enrichment beyond surface-level stylistic adaptation. Our pilot human study shows that style adaptation alone provides only marginal gains in reader satisfaction (2.3%), while context-enhanced rewriting substantially improves user preference alignment (24.5%). Motivated by this, we introduce STORYLENSBENCH, a large-scale benchmark for preference-aligned story rewriting, comprising structured story books, multi-dimensional reader preference profiles, and ranked context-aware rewritten stories. Building on this benchmark, we propose STORYLENSEVAL, a reward model for estimating reader satisfaction over rewritten stories, and STORYLENSWRITER, a two-stage rewriting model combining supervised fine-tuning with GRPO-based reinforcement learning. We further establish a comprehensive evaluation framework covering fidelity, coherence, and reader satisfaction. Experimental results demonstrate that STORYLENSWRITER consistently outperforms strong generation and personalization baselines, highlighting the importance of context-aware narrative enrichment for personalized story rewriting. 

---
# PromptEmbedder:: Efficient and Transferable Text Embedding via Dual-LLM Soft Prompting 

**Authors**: Yu-Che Tsai, Kuan-Yu Chen, Yuan-Hao Chen, Yu-Han Chang, Ching-Yu Tsai, Yu-Hsiang Chuang, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.28066)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable efficacy in text embedding, yet current adaptation methods like LoRA face significant bottlenecks in computational efficiency and cross-architecture transferability. Whenever a new backbone emerges, existing approaches require costly retraining from scratch. To address this, we propose PromptEmbedder, a novel dual-LLM framework that decouples embedding knowledge from specific backbone weights. PromptEmbedder utilizes a Prompting LLM to generate instruction-aware soft prompts for a frozen Embedding LLM via a differentiable generation process with continuous relaxation, ensuring full gradient flow during contrastive training. By localizing task-specific knowledge within the Prompting LLM, adapting to new architectures requires only retraining a lightweight linear alignment matrix. Evaluations on the MTEB benchmark show that PromptEmbedder achieves comparable performance with LoRA finetuning while reducing GPU memory by 40% and accelerating training by 3.7x. Our approach establishes a scalable, architecture-agnostic paradigm for efficient LLM-based representation learning. 

---
# I Hear, Therefore I Trust: A Socio-Technical Investigation of Humans as Synthetic Speech Detectors 

**Authors**: Lelia Erscoi, Tomi Kinnunen  

**Link**: [PDF](https://arxiv.org/pdf/2605.28064)  

**Abstract**: Automatic deepfake detection has received considerable research attention, yet the socio-technical environment in which humans actually encounter synthetic speech remains poorly understood. We investigate voice deepfake detection as a perceptual and contextual process, presenting a localization task in which 47 participants marked suspected synthetic segments across authentic, fully synthetic, and partially synthetic utterances under three manipulated trust cues: instructional framing, affective priming, and provenance labeling. Participants provided quality ratings on mechanicalness, expressiveness, intelligibility, clarity, calmness, and confidence of evaluation. Utterance class was the primary determinant of detection accuracy and perceptual quality; trust cues produced no main effects but motivated detection behavior. Fully synthetic speech was detected at below-chance levels. Quality ratings tracked utterance type, indicating implicit discrimination where overt detection failed. 

---
# Unified Synthesis of Compositional Speech and Sound from Free-Form Text Prompts 

**Authors**: Yuyue Wang, Xihua Wang, Xin Cheng, Yijing Chen, Ruihua Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.28063)  

**Abstract**: Audio generation has made significant progress, yet synthesizing unified audio where speech and sounds are naturally composited remains a challenge. Current methods either rely on disjoint pipelines, which fail to capture fine-grained interactions, or require structured inputs and external text rewriting, which limits the flexibility of free-form text prompts. In this paper, we introduce a new task: Free-Form-Text-Prompt-to-Unified-Audio generation, which aims to directly synthesize unified audio containing speech, sound, and their composites from unconstrained natural language. To address this task, we propose PlanAudio, a unified, autoregressive LLM-based framework. First, it simplifies the model architecture by leveraging intrinsic LLM reasoning capability instead of traditional text encoders. Second, it introduces a semantic latent chain-of-thought mechanism, an implicit planning mechanism that bridges high-level semantic understanding and low-level acoustic synthesis. Furthermore, we create PlanAudio-Bench, a specialized benchmark for evaluating composite audio scenarios. We perform evaluations in the scenarios of speech, sound, and their composites. The results demonstrate that PlanAudio generally outperforms the existing pipeline and unified baselines, while staying competitive with models designed for a single scenario. Our analysis further reveals the superiority of semantic latent CoT over other CoT mechanisms and highlights the importance of continuous multi-scenario training curricula. 

---
# On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective 

**Authors**: Zhi Zhou, Ming Yang, Shi-Yu Tian, Kun-Yang Yu, Lan-Zhe Guo, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.28057)  

**Abstract**: Test-time adaptation (TTA) aims to adapt models to maintain reliable performance on non-stationary test streams without requiring labeled data. Despite its empirical success, the learnability of TTA under non-stationary streams remains unexplored. A key challenge is the lack of a principled theoretical framework that simultaneously aligns with the TTA objective and captures both continuously evolving distribution shifts and intrinsic information constraints. To address this gap, we propose the first theoretical framework for studying the learnability of TTA and introduce $(\epsilon,\delta)$-Recovery Complexity and $(\epsilon,\rho)$-TTA Learnability. Recovery complexity measures the post-shift time needed to maintain excess risk below a target level with high probability, and is further extended to TTA learnability, which measures the long-term reliability of TTA. Within this framework, we introduce a novel discrete surrogate for non-stationary test streams, enabling a unified and tractable analysis of both gradual and abrupt shifts. We derive order-wise matching lower and upper bounds on recovery complexity, revealing fundamental limits of TTA and an intrinsic adaptivity-information trade-off. These results provide unified learnability guarantees for TTA that complement regret-based analyses. 

---
# Extracting Small Translation Specialists from LLMs by Aggressively Pruning Experts 

**Authors**: Liu O. Martin, Lucas Bandarkar, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.28042)  

**Abstract**: Modern large language models (LLMs) achieve state-of-the-art machine translation performance, but they do so as broad generalists largely trained for many tasks and capabilities unrelated to translation. Thus, they are heavily overparameterized for this task, resulting in excessive memory and compute requirements. In this paper, we present a method for aggressively pruning experts from modern mixture-of-experts LLMs while incurring negligible degradation in translation quality. Our approach exploits expert specialization and the separability of multilingual capabilities in LLMs to identify experts irrelevant to translation. And because of the modular nature of MoEs, these can be easily pruned without any training. Without retraining, we are able to prune half of all experts with negligible degradation and 70% with only minor losses. With a very short SFT, we prune 75% of experts while recovering baseline performance, and in some settings remove nearly 90% while maintaining reasonable translation quality. Overall, our results show that translation requires only a fraction of the LLM, enabling substantial compression of the MoE blocks that contain over 90% of parameters. 

---
# SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection 

**Authors**: Shuhao Chen, Weisen Jiang, Yeqi Gong, Shengda Luo, Chengxiang Zhuo, Zang Li, James T. Kwok, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28030)  

**Abstract**: Fine-tuning large language models often undermines their safety alignment, a problem further amplified by harmful fine-tuning attacks in which adversarial data removes safeguards and induces unsafe behaviors. We propose SPARD, a defense framework that integrates Safety-Projected Alternating optimization with Relevance-Diversity aware data selection. SPARD employs SPAG, which optimizes alternatively between utility updates and explicit safety projections with a set of safe data to enforce safety constraints. To curate safe data, we introduce a Relevance-Diversity Determinantal Point Process to select compact safe data, balancing task relevance and safety coverage. Experiments on GSM8K and OpenBookQA under four harmful fine-tuning attacks demonstrate that SPARD consistently achieves the lowest average attack success rates, substantially outperforming state-of-the-art defense methods, while maintaining high task accuracy. Code is available at this https URL. 

---
# VCap: Hypergeometric Rewards for Weak-to-Strong Visual Captioning 

**Authors**: Xingyu Lu, Jinpeng Wang, Yi-Fan Zhang, Yankai Yang, Yancheng Long, Yiyang Fan, Xuanyu Zheng, Haonan Fan, Kaiyu Jiang, Tianke Zhang, Changyi Liu, Bin Wen, Fan Yang, Tingting Gao, Han Li, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2605.28023)  

**Abstract**: Visual captioning requires models to capture visual content faithfully while minimizing both omission and hallucination. As the dominant paradigm for captioning, MLLMs have achieved strong performance through scaling and high-quality data. Recently, RL has emerged as a key route to driving MLLMs toward higher precision and broader coverage, however, existing reward designs for captioning fail to provide fine-grained and reliable signals for factual verification, limiting their effectiveness. To address this, we propose VCap, a Witness-Adjudicator reward that pairs the reference caption (a witness) with the visual signal (an adjudicator). By explicitly verifying factual consistency between the reference and policy-generated captions grounded in the visual signal, VCap delivers a reward signal with hypergeometric-distribution-level precision for caption quality verification. This design enables effective learning even from imperfect references, facilitating weak-to-strong generalization in RL training. In our experiments, an 8B model trained with VCap outperforms open- and closed-source SOTA models on multiple image and video captioning benchmarks. Human evaluation further confirms its strong alignment with factual correctness. Additionally, VCap improves MLLM perceptual capability, generalizes across tasks, and surpasses best-of-N distillation, challenging prior assumptions about RLVR. 

---
# MemGuard: Preventing Memory Contamination in Long-Term Memory-Augmented Large Language Models 

**Authors**: Hyeonjeong Ha, Jeonghwan Kim, Cheng Qian, Jiayu Liu, William M. Campbell, Yue Wu, Yuji Zhang, Kathleen McKeown, Dilek Hakkani-Tur, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.28009)  

**Abstract**: Memory-augmented large language models extend reasoning beyond a fixed context window by maintaining long-term memory across interactions. However, existing memory systems often collapse stable user facts, episodic events, and behavioral rules into a shared space, allowing functionally distinct memories to be retrieved and used as interchangeable evidence. We identify this failure mode as heterogeneous memory contamination, where context-specific events become overgeneralized claims, or semantically relevant but functionally incompatible memories mislead generation. To this end, we introduce MemGuard, a type-aware memory framework that preserves functional memory boundaries during memory construction and retrieval. It assigns each memory an explicit functional role at write time, maintains relations across type-isolated memories, and selectively composes evidence only from necessary memory types, reducing contamination from irrelevant or functionally incompatible evidence. Across hallucination and long-horizon conversation benchmarks, MemGuard improves memory reliability by up to 28.27% while retrieving up to 5.8x fewer memory tokens than prior methods. These results suggest that reliable long-term reasoning depends on principled organization and selective use of heterogeneous memory. 

---
# Learning Compositional Latent Structure with Vector Networks 

**Authors**: Niclas Pokel, Benjamin F. Grewe  

**Link**: [PDF](https://arxiv.org/pdf/2605.28007)  

**Abstract**: Deep networks are powerful function approximators, but they typically store many different computations in shared weight matrices, making it difficult to selectively reuse or adapt parts of them when a familiar structure appears in novel combinations. We introduce the Vector Network (VN), a hierarchical recurrent architecture in which each layer replaces a fixed weight matrix with a library of reusable rank-1 weight atoms. For each input, VN minimizes a layer-local energy to infer a sparse set of active weight atoms and their coefficients, jointly constrained by bottom-up input reconstruction and top-down feedback consistency. These weight atom coefficients then compose an input-specific low-rank weight matrix for that sample. After convergence, slow learning updates only the selected weight atoms through local residual signals scaled by the inferred coefficients. We evaluate VN on four compositional benchmarks spanning 1D signals, 2D spatial decoding, N-body dynamics, and compositional MNIST. VN matches strong baselines in distribution while often achieving out-of-distribution error about an order of magnitude lower when familiar factors must be recombined in novel ways. Vector networks thus make compositional generalization a structural property of the architecture and inference process rather than a brittle byproduct of fitting many behaviors into one shared dense parameter substrate. 

---
# Integrated and Cross-Architecture Interpretation of LLM Reasoning 

**Authors**: Leonardo Matthew Yauw, Wei-Bin Kou, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28006)  

**Abstract**: Understanding how LLMs reason is hindered by a practical asymmetry: while their generated outputs are observable, the underlying reasoning patterns remain opaque. Relying on single probes, such as Mutual Information Peak (MIP) or Deep-Thinking Ratio (DTR), risks underestimating the genuine inferential structure. To response this deficiency, we present an Integrated, cross-Architecture Reasoning (IAR) framework, designed to provide a unified approach to LLM reasoning interpretability. Specifically, we first propose to use bandwidth-calibrated MIP coupled with Tukey IQR peak-detection to isolate reasoning-crucial tokens at the output layer. Second, we performed an overlap analysis between MIP-picked tokens and DTR-deep tokens to trace the cross-layer trajectories of those tokens. This also discloses whether reasoning-crucial tokens are computation-intensive as well, further facilitating to understand how reasoning patterns evolve across model layers. Finally, we apply a Jaccard stability metric over multi-domain problems to verify if the MIP-identified tokens are reasoning quality-guaranteed. Extensive experiments on three models (Qwen-7B, Qwen-14B, and Llama-8B) across four domains (mathematics, code, logic, and common sense) demonstrate IAR's generalizable interpretation capabilities across architectures. 

---
# Tool Forge: A Validation-Carrying Toolchain for Governed Agentic Execution 

**Authors**: Swanand Rao  

**Link**: [PDF](https://arxiv.org/pdf/2605.28000)  

**Abstract**: Large language model agents are increasingly expected to perform operational work: calling APIs, manipulating files, assembling workflows, and acting inside enterprise systems. Yet the tool layer on which this execution depends is still commonly treated as either a hand-written integration artifact or a static list of schemas exposed to a model. This paper introduces Tool Forge, a validation-carrying toolchain for converting natural-language capability intent into governed, sandbox-verified, cataloged tool artifacts and exposing those artifacts to agents through a token-efficient routing layer.
Tool Forge treats a tool as a capsule containing intent, capability contract, implementation, dependency policy, tests, documentation, runtime validation evidence, lifecycle state, credential bindings, and routing metadata. It also introduces a Router that exposes intent-scoped tool sessions instead of loading full catalog schemas into the model context. We describe the system architecture, validation pipeline, MCP-facing routing model, governance controls, and initial reproducible benchmarks from the open-source implementation.
Across 83 Router benchmark cases, Tool Forge Router achieves aggregate micro-F1 of 0.901 while reducing estimated task-flow tool context by 99.2% relative to naive full-catalog schema exposure. In a 25-case end-to-end generation probe over local-tool tasks, Tool Forge generates 25 of 25 tool bundles, reaches micro-F1 of 0.940 against deterministic acceptance checks, and passes 23 of 25 live sandbox validations. These results are presented as an initial systems benchmark, not as a state-of-the-art claim. The paper identifies remaining challenges in adversarial routing, broader API grounding, sandbox isolation, and cross-system evaluation. 

---
# Learning to Assign Prediction Tasks to Agents with Capacity Constraints 

**Authors**: Shang Wu, Saatvik Kher, Padhraic Smyth  

**Link**: [PDF](https://arxiv.org/pdf/2605.27999)  

**Abstract**: We address the problem of learning to assign prediction tasks to one agent from a set of available human or AI agents. In particular, we focus on the sequential learning of agent expertise and assignment policies where each agent is constrained to handle a fraction of tasks. We provide a general theoretical characterization of this problem in terms of agent capacities, differences in agent expertise, and task context. We then develop a framework of sequential explore-exploit policy-learning algorithms that seek to maximize overall performance. Experimental results over a variety of tabular, image, and text prediction tasks demonstrate systematic gains from our policy-learning algorithms relative to non-contextual baselines across different types of agents, including LLMs and humans. 

---
# Where Does Toxicity Live? Mechanistic Localization and Targeted Suppression in Language Models 

**Authors**: Himanshu Beniwal, Mayank Singh  

**Link**: [PDF](https://arxiv.org/pdf/2605.27997)  

**Abstract**: Large language models frequently generate toxic, hateful, or harmful content, yet existing mitigation methods rely on costly retraining or output-level filtering with no mechanistic insight into where toxicity originates internally. We introduce Meow2X and TRNE, two complementary retraining-free frameworks that localize toxicity to specific layers and neurons by analyzing activation differentials between toxic and neutral prompts, then suppress them via inference-time scaling or minimal rank-one weight edits -- without any gradient descent. Evaluations across five LMs, two benchmarks, and 90 configurations using dual safety evaluators demonstrate consistent toxicity reduction while preserving language modeling quality. Our analysis reveals that toxicity is disproportionately encoded in early MLP layers, varies across architectures, and is systematically underestimated by single-evaluator setups -- underscoring the need for multi-evaluator safety assessment. By bridging mechanistic interpretability with practical detoxification, our framework offers a principled path toward safer, more transparent language models. 

---
# Geometry-Correct Diffusion Posterior Sampling with Denoiser-Pullback Curvature Guidance and Manifold-Aligned Damping 

**Authors**: Seunghyeok Shin, Minwoo Kim, Dabin Kim, Hongki Lim  

**Link**: [PDF](https://arxiv.org/pdf/2605.27990)  

**Abstract**: Diffusion posterior sampling conditions diffusion priors on measurements, but data-consistency updates are typically scaled by hand-tuned guidance weights and can destabilize sampling under stiff, operator-dependent curvature. We replace scalar guidance with a per-noise-level damped Gauss--Newton correction computed in diffusion-state coordinates. The correction pulls likelihood gradients back through the denoiser, uses a one-sided curvature model that avoids forward denoiser Jacobians, and applies diffusion-calibrated rank-one damping aligned with the denoiser residual. Each correction is solved with matrix-free GMRES using automatic differentiation, and sampling proceeds with a variance-preserving Langevin transition with a closed-form drift/noise split. On FFHQ and ImageNet across inverse problems, it achieves competitive PSNR/SSIM/LPIPS while running markedly faster than most of the compared baselines; on accelerated MRI reconstruction, it achieves the best PSNR/SSIM among the compared baselines. 

---
# KVoiceBench, KOpenAudioBench, and KMMAU: Agent-Driven Korean Speech Benchmarks for Evaluating SpeechLMs 

**Authors**: Haechan Kim, Seungjun Chung, Inkyu Park, Jihoo Lee, Jonghyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.27984)  

**Abstract**: Speech language models (SpeechLMs) have achieved substantial progress by extending large language models (LLMs) to the speech modality. However, SpeechLM evaluation remains heavily centered on English, limiting reliable assessment of multilingual speech capabilities. Straightforward benchmark transfer through ASR, translation, normalization, and TTS can corrupt language-specific instructions, answer constraints, and spoken forms; for audio understanding, transferring source-language audio also fails to preserve target-language speaker attributes, accents, and paralinguistic properties. To address these limitations, we propose two human-agent benchmark-construction frameworks: one transfers source-language SpokenQA benchmarks into target-language SpokenQA benchmarks, and the other converts target-language ASR corpora into audio understanding benchmarks using transcriptions and speaker metadata. Using these frameworks, we construct and publicly release three Korean speech benchmarks: KVoiceBench and KOpenAudioBench for Korean SpokenQA, and KMMAU for Korean audio understanding, comprising 12,345 samples in total. We evaluate eight recent SpeechLMs and find that English-Korean performance gaps vary substantially across models and task families, and that SpokenQA and audio understanding rankings diverge, revealing complementary weaknesses invisible to English-only evaluation. 

---
# Periodic RoPE for Infinite Context LLMs 

**Authors**: Simin Huo  

**Link**: [PDF](https://arxiv.org/pdf/2605.27980)  

**Abstract**: The ability to process ultra-long contexts is crucial for large language models (LLMs) to perform long-horizon tasks. While recent efforts have extended context windows to 1M and beyond, model performance degrades when sequence length exceeds the pre-trained range of positional encodings (e.g., RoPE), i.e., position exhaustion. This fundamental limitation must be overcome to achieve a truly infinite context. To address it, we propose Periodic RoPE (P-RoPE), a positional encoding mechanism designed to circumvent this exhaustion. It operates in conjunction with sliding window attention (SWA) to capture local dependencies and relative positions within each window. This local layer is then complemented by a global attention layer with No Positional Encoding (NoPE), enabling unbounded interaction across the entire sequence without positional constraints. By stacking these two types of layers, the model avoids the need for positional extrapolation to generalize longer and theoretically supports an infinite context window. Empirical results show that our model, MiniWin, outperforms MiniMInd with standard GPT architectures in long-context efficiency and stability. Our work provides a possible pathway toward LLMs with genuine infinite-context understanding. The code is available at \href{this https URL}{this https URL}. 

---
# Semantic Flow Regularization: Teaching LLMs to Generate Diverse Yet Coherent Responses 

**Authors**: Kerui Peng, Feifei Li, Xingyu Fan, Wenhui Que  

**Link**: [PDF](https://arxiv.org/pdf/2605.27971)  

**Abstract**: When large language models are fine-tuned to generate persona- or tone-conditioned responses, their output diversity is severely limited--a failure we term Cross-Style Collapse. We trace this collapse to the cross-entropy objective, which under shared representations tends to suppress diverse continuations. We propose Semantic Flow Regularization (SFR), a lightweight auxiliary objective that supervises the backbone with continuous sentence-encoder embeddings of future segments via conditional flow matching. The stochastic flow source preserves multi-modality by construction; the flow-matching head is discarded at inference, adding zero deployment cost. On a large-scale industrial dialogue dataset (Qwen3-32B, 9 personas), SFR improves output diversity, style fidelity, and response quality over SFT. We further validate on the public LiveCodeBench-v5 (Qwen2.5-Coder-7B-Instruct), where SFR consistently improves pass@k, confirming generality beyond stylized dialogue. A controlled comparison on MBPP reveals Multi-Token Prediction to be a degenerate special case of SFR. 

---
# Multi-Teacher Knowledge Distillation via Teacher-Informed Mixture Priors 

**Authors**: Luyang Fang, Yongkai Chen, Jiazhang Cai, Ping Ma, Wenxuan Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2605.27967)  

**Abstract**: Knowledge distillation is a powerful method for model compression, enabling the efficient deployment of complex deep learning models (teachers), including large language models. However, its underlying statistical mechanisms remain unclear, and uncertainty evaluation is often overlooked, especially in real-world scenarios requiring diverse teacher expertise. To address these challenges, we introduce \textit{Multi-Teacher Bayesian Knowledge Distillation} (MT-BKD), where a distilled student model learns from multiple teachers within the Bayesian framework. Our approach leverages Bayesian inference to capture inherent uncertainty in the distillation process. We introduce a teacher-informed prior, integrating external knowledge from teacher models and task-specific training data, offering better generalization, robustness, and scalability. Additionally, an entropy-based weighting mechanism adaptively adjusts each teacher's influence, allowing the student to combine multiple sources of expertise effectively. MT-BKD enhances the interpretability of the student model's learning process, improves predictive accuracy, and provides uncertainty quantification. We validate MT-BKD on both synthetic and real-world tasks, including protein subcellular location prediction and image classification. Our experiments show improved performance and robust uncertainty quantification, highlighting the strengths of our MT-BKD framework. 

---
# ROVER: Routing Object-Centric Visual Evidence for Grounded Multi-Image Reasoning 

**Authors**: Guannan Lv, Ren Nie, Hongjian Dou  

**Link**: [PDF](https://arxiv.org/pdf/2605.27959)  

**Abstract**: Multimodal Large Language Models (MLLMs) have increasingly localized and interleaved visual evidence for deliberative reasoning. Grounding-based approaches typically focus on regions of interest (RoIs) by injecting cropped image patches or RoI-specific features into the reasoning context. However, such designs can weaken holistic scene understanding and inter-object relations, while incurring decoding costs that scale with the number and size of RoIs. Alternatively, adaptive visual feature selection often requires fine-grained supervision or complex heuristics. To address these limitations, we propose ROVER (Routing Object-centric Visual Evidence for grounded multi-image Reasoning), a lightweight, learnable plugin for efficient global visual evidence routing. Upon each object grounding prediction, ROVER injects a step-specific token triplet to synergistically: (i) aggregate the ongoing reasoning context, (ii) distill intra-image cues into a visual working space via object-centric differential attention, and (iii) route and integrate history-aware evidence across objects and images within this space for subsequent reasoning. We integrate ROVER into Qwen2.5-VL-7B and develop an interleaved SFT-to-GRPO training pipeline. Strictly adhering to the original datasets and evaluation protocols, our method achieves the best performance on MM-GCoT (+4.8% answer accuracy, +14.6% grounding accuracy) and VideoEspresso (+8.6% answer accuracy). The VideoEspresso-trained model demonstrates strong transferability, outperforming the base model by +4.7% on average across diverse benchmarks. 

---
# Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and the Geometry of Deceptive Representations 

**Authors**: Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.27958)  

**Abstract**: Linear probes trained on LLM activations are increasingly proposed as deception-detection metrics, yet report AUROC exceeding 0.96 on clean benchmarks while collapsing under distributional shift. This paper systematically pressure-tests probe-based metrics across the Gemma 3 model family (1B-27B parameters), diagnosing why they fail rather than merely documenting that they fail. We test four hypotheses about deception encoding: (1) single linear direction, (2) multi-dimensional subspace, (3) convex conic hull, (4) entropy proxy. Our design includes cross-domain transfer matrices, multi-dimensional probe analysis with permutation null baselines, entropy-residualization tests, and distractor evaluations across 8 stylistic shifts. We find that: (a) probes achieve near-perfect AUROC (>=0.998) on clean data but collapse under stylistic shifts; style-augmented probes recover near-perfect detection (mean AUROC 0.979-0.983) on unseen styles; (b) the single-direction hypothesis is rejected (k=1 captures only 0.61-0.80 AUROC), with cross-domain transfer failure confirmed as geometric rather than layer-mismatch-driven; (c) the entropy-proxy hypothesis is rejected (max |rho|=0.454, max Delta-AUROC after residualization=0.004); and (d) deception does not form a significant linear subspace (per-domain k*=0), yet multi-dimensional probes (k>=5) recover the signal through distributed sub-threshold features. Probe fragility reflects distributional narrowness rather than an architectural limitation: style-augmented probes recover near-perfect detection at both 4B and 27B, establishing that the inverse scaling pattern is a training-distribution artifact rather than a genuine scale-dependent phenomenon. 

---
# When Think-with-Image Meets Safety: What Determines Multimodal Jailbreak Robustness? 

**Authors**: Yuan Tian, Bing Hu, Fang Wu, Xiaomin Li, Binghang Lu, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2605.27932)  

**Abstract**: Think-with-image reasoning is emerging as a new inference paradigm for large vision-language models, but its safety implications remain poorly understood. Existing systems already span multiple process designs, including direct response generation, text-only prior turn, visual-state manipulation, and explicit external image-tool invocation. In this paper, we ask which of these evaluated paradigms improves multimodal jailbreak robustness, and why. Across multiple vision-language models, explicit image-tool interaction yields the lowest attack success rates in our experiments, reducing jailbreak success by around 30% relative on average across the evaluated models. This finding is initially surprising: ASR remains low even when the returned image-tool output is manually overridden or itself unsafe-looking, but returns near direct-answering levels under text-only prior turn controls. These results indicate that the lower ASR is not explained by benign returned-image semantics or by the textual image-tool trace alone. To explain the pattern, we introduce an image-tool safety vector framework that models image-tool invocation as a residual shift in hidden representations toward a safety-relevant direction. Representation-level analyses and activation interventions support this account. Overall, our results suggest that explicit image-tool interaction is a promising design pattern for improving jailbreak robustness, while also motivating pipeline-specific safety evaluation. 

---
# Do We Really Need Quantum Machine Learning?: A Multidimensional Empirical Study 

**Authors**: Sudip Vhaduri, Ryan Gammon, Sayanton Dibbo  

**Link**: [PDF](https://arxiv.org/pdf/2605.27923)  

**Abstract**: The rapid growth of computer vision and increasingly complex image recognition tasks has exposed fundamental computational limitations of classical machine learning models, motivating the exploration of quantum computing as an emerging new paradigm. This paper presents a comprehensive benchmarking study of classical and quantum machine learning models for image recognition on the MNIST handwritten digit dataset, evaluating both traditional models, a Classical Support Vector Machine (CSVM) and a Quantum Support Vector Machine (QSVM), and deep neural network models, a Classical Convolutional Neural Network (CCNN) and a Quantum Convolutional Neural Network (QCNN), across four performance dimensions: classification accuracy, computational runtime, parameter count, and memory requirements. Experiments are conducted as functions of both feature dimensionality and sample size, and across CPU and GPU execution environments, providing a controlled, multidimensional comparison to address gaps in prior work. For the SVM-based models, QSVM consistently outperforms CSVM in accuracy, reaching $\sim$ 0.90 versus $\sim$ 0.85 at 1,000 samples, with a higher computational cost. A feature count of 10 qubits and a sample size in the range of 200 -- 500 emerge as practical operating points that balance accuracy and runtime. For the neural network models, CCNN and QCNN achieve comparable classification accuracy, both exceeding 0.96 at 64 features and 60,000 samples, yet QCNN offers substantially superior parameter and memory efficiency, requiring $\sim$ 94\% fewer parameters and $\sim$ 75\% less memory than CCNN at higher feature counts, while incurring higher runtime. Across both model families, quantum models consistently outperform classical models by greater margins in accuracy as feature dimensionality or sample size increases. 

---
# Let the Results Speak: A Replication-First Paradigm for LLM Behavioral Benchmarking 

**Authors**: Yuming, Huang, Yao Liu, Lei Wang, Junchen Wan  

**Link**: [PDF](https://arxiv.org/pdf/2605.27914)  

**Abstract**: Subjective evaluation of LLM behavior -- empathy, restraint, calibrated emotional tone -- is hard. Human inter-rater agreement on such qualities saturates near rho ~ 0.45, and an LLM-as-judge proxy alone risks circularity: a judge sharing the target's training cohort cannot independently verify it. Anchoring validity to a single human-rater consensus does not extend to capabilities where humans themselves disagree.
We propose a replication-first paradigm: instead of anchoring on one rater group, we certify the instrument via four orthogonal properties -- reliability across K runs, cross-instrument replication across architecturally distinct judges, historical-footprint calibration via judges from earlier training cohorts, and pre-registered prediction. We test it on emotional accompaniment by letting the rubric self-evolve data-driven across iterations: the dimensions are not pre-stipulated and the procedure stabilizes to a 9-dimension set. Pre-registration applies to 10 falsifiable hypotheses and 11 forward predictions, committed before any test data was collected.
Applied to 49 models across 8 families, the paradigm surfaces what aggregate scores hide. On advice-restraint -- whether a model refrains from giving unsolicited solutions in empathic contexts -- gpt-5 falls 1.87 points from gpt-4.1 and Opus-4.7 falls 0.629 from Opus-4.6, while aggregate scores stay flat. The regression survives three user-proxy swaps (95% of magnitude), replicates across a 5-family judge stack and a 17-month cohort gap, and persists on 74 held-out real ESConv conversations (rho in [0.749, 0.850]); the instrument reaches ordinal Krippendorff alpha = 0.91. As a by-product, the paradigm acts as a saturation-source diagnostic, separating instrumental ceilings (breakable by rubric refinement) from structural ceilings (needing scenario or roster intervention). 

---
# ESC-Skills: Discovering and Self-Evolving Skills for Emotional Support Conversations 

**Authors**: Jie Zhu, Huaixia Dou, Shuo Jiang, Junhui Li, Lifan Guo, Feng Chen, Chi Zhang, Fang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2605.27908)  

**Abstract**: Existing emotional support conversation (ESC) systems mainly rely on end-to-end response generation or coarse strategy supervision, offering limited interpretability and little support for systematic skill improvement. We propose ESC-Skills, a skill-centric framework that discovers and self-evolves executable emotional support skills. We first model localized support interactions as Intervention Units (IUs), which capture state--action--outcome dynamics between seeker states, support interventions, and post-response emotional changes. Based on IUs extracted from both successful and failed ESC dialogues, we construct the ESC-Skills Bank, a repository of executable emotional support skills containing intervention guidance, applicability conditions, expected outcomes, and potential risks. To further improve robustness, we introduce a multi-profile self-evolutionary refinement framework in which an ESC agent interacts with diverse simulated seeker profiles under SAGE evaluation. The resulting interaction traces are analyzed to identify missing skills, unsafe interventions, and profile-specific failure patterns, which are then used to refine the Skills Bank through simulation-based verification. Experimental results demonstrate that ESC-Skills improves both response-level quality and dialogue-level emotional outcomes while providing more interpretable and controllable support behaviors. We will release the code, prompts, and ESC-Skills Bank at this https URL. 

---
# The Fragility of Chain-of-Thought Monitoring Across Typologically Diverse Languages 

**Authors**: Eric Onyame, Runtao Zhou, Kowshik Thopalli, Bhavya Kailkhura, Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2605.27901)  

**Abstract**: Chain-of-thought (CoT) monitoring has been proposed as a promising safety mechanism for detecting misaligned behavior in large language models. However, its reliability remains largely unexplored beyond English and across diverse model families. We present the first large-scale evaluation of CoT monitorability across 13 diverse languages and seven frontier model families, comprising 16 models. Using adversarial-hint evaluations that require explicit intermediate computation, together with analysis of internal answer-token probabilities, we consistently find CoT unfaithfulness across languages and hint types, with an average rate of 95.9\% across 8B--120B parameter models. We find that frontier models systematically engage in strategic manipulation, including answer-switching, post-hoc rationalization, and procedural exploitation of hints, making external monitors struggle to detect deception. We show that frontier models often commit to the misaligned cue in their latent activations within the first 15\% of generation, even when the CoT appears faithful. Surprisingly, these deceptive patterns remain 100\% in low-resource languages, revealing fundamental limitations in current CoT-based oversight. Our results reveal that CoT monitoring is fundamentally fragile under linguistic distribution shift, providing a substantially weaker safety signal than what English-only studies suggest. These findings underscore an urgent need to develop robust CoT monitors and to accelerate research into white-box monitoring techniques, especially to improve CoT monitorability in mid- and low-resource languages. Our code is available \href{this https URL}{\textcolor{blue}{here}}. 

---
# SmartDirector: Keyframe-Conditioned Cinematic Video Generation with Narrative Pacing Control 

**Authors**: Zhida Zhang, Jie Ma, Zhan Peng, Haoxue Wu, Yang Han, Jun Liang, Jie Cao, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.27891)  

**Abstract**: The narrative quality of a video fundamentally determines its perceptual value. Although existing video generation methods can produce visually appealing content, they predominantly rely on sparse conditioning signals such as text prompts or first/last frames, which limits precise control over narrative structure and temporal pacing. In this paper, we propose SmartDirector, a framework that enhances the narrative capacity of video generation models through multiple keyframes. SmartDirector supports flexible generation scenarios including single-shot generation, multi-shot narrative synthesis, and video extension. The framework operates in two stages: Director-Gen generates a low-resolution video conditioned on the provided keyframes, and Director-SR refines the output by exploiting high-resolution keyframes as semantic anchors to recover fine-grained details. To enable robust multi-keyframe training, we construct a data pipeline that curates single-shot and multi-shot sequences from movies. Extensive experiments demonstrate that SmartDirector substantially outperforms existing state-of-the-art approaches. We will release the code to facilitate further research. 

---
# VibeSearchBench: Benchmarking Long-horizon Proactive Search in the Wild 

**Authors**: Xiaohongshu Inc  

**Link**: [PDF](https://arxiv.org/pdf/2605.27882)  

**Abstract**: LLM-based agents score well on search benchmarks, yet real users consistently find results unsatisfying, revealing a persistent evaluation-experience gap. We attribute this gap to existing benchmarks' reliance on over-specified queries, single-turn interactions, and fixed-schema evaluation, none of which reflect real search behavior where users and agents collaboratively refine vague intent through multi-turn dialogue. We term this paradigm VibeSearch and introduce VibeSearchBench, a benchmark comprising 200 manually curated bilingual (Chinese and English) tasks across 20 domains, split into VibeSearch-Pro (professional) and VibeSearch-Daily (daily-life) subsets. Each task pairs a user persona with a schema-free ground-truth knowledge graph, and is evaluated through a progressive-disclosure user simulator and a graph-matching evaluation framework. We benchmark seven frontier models under both the ReAct framework and the OpenClaw agent harness. Results show that all models remain substantially inadequate for VibeSearch (best F1: 30.30), highlighting the need for fundamental advances in long-context reasoning, proactive intent elicitation, and structured knowledge construction. 

---
# SPAR: Support-Preserving Action Rectification 

**Authors**: Jiaxin Zhao, Weihang Pan, Xun Liang, Binbin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.27877)  

**Abstract**: Offline policy improvement faces an inherent conflict between maximizing value and fitting the data distribution. While in-sample weighted regression is stable, it suffers from over-conservatism that suppresses high-value actions in the distribution tail; conversely, gradient-based approaches often exhibit a fitting-optimization conflict of gradients, which drives the policy off the data manifold. To address this, we propose Support-Preserving Action Rectification (SPAR), which reframes global learning as a local residual rectification anchored to a frozen pure behavior cloning policy. This framework performs fine-grained fitting and local policy improvement in the residual space, thereby contracting the search space. We further introduce Latent Self-Imitation, utilizing a latent-sampling weighted-regression mechanism to address fitting-improvement gradient conflict in the residual space. Theoretically, we prove this mechanism eliminates the manifold-normal drift of standard value gradients, while extensive D4RL experiments show SPAR extracts significant gains from suboptimal baselines to achieve state-of-the-art performance. 

---
# From Detection to Mechanism: Cross-Attention Graph Neural Networks Enable Drug-Drug Interaction Type Prediction An Ablation Study with Acetylsalicylic Acid Validation 

**Authors**: Juergen Dietrich  

**Link**: [PDF](https://arxiv.org/pdf/2605.27861)  

**Abstract**: Predicting whether two drugs interact (binary detection) is a substantially dif- ferent task from predicting the mechanism type of that interaction (multi-class classification). This study presents a systematic ablation study of three Graph Neural Network (GNN) architectures for drug-drug interaction (DDI) prediction on a publicly available benchmark dataset comprising 38,337 positive pairs across 86 interaction types. Three architectures are compared under identical training conditions (n = 61,339 pairs): a siamese dual Message Passing Neural Network (MPNN) with concatenation (Concat), a dual MPNN with four-head cross-attention (CrossAtt), and a ternary MPNN incorporating an interaction graph (Ternary). CrossAtt improves multi-class F1-macro by +0.186 absolute (+45%) over Concat, while improving binary AUC by only +0.012 (+1.3%) - confirming that atom-level inter-molecular communication specifically enables mechanism-type classification. The ternary architecture underperforms despite equivalent training data, with its failure consistent with a training instability hypothesis. Validation on ten acetylsali- cylic acid (ASA) drug pairs, held out prior to training, demonstrates 10/10 correct DDI-type predictions for CrossAtt versus 0/10 for Ternary. Two consistent failure cases are identified across all architectures, linking to structural limits established in a companion toxicity study. 

---
# DecomposeRL: Learning to Ask Useful, Informative, and Diverse Questions for Semi-Supervised, Traceable Claim Verification 

**Authors**: Shubhashis Roy Dipta, Ankur Padia, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2605.27858)  

**Abstract**: Claim verification splits between end-to-end classifiers that are accurate but yields no inspectable traces, and decomposition-based methods produce inspectable traces but lag performance on benchmark datasets. We propose DecomposeRL an accurate claim-verifier that produce inspectable traces. DecomposeRL frames decomposition as an RL policy trained with GRPO and a multi-faceted reward ensemble, enabling both fully supervised and semi-supervised learning from unlabeled claims. DecomposeRL addresses the prohibitive training cost of GRPO with a data-curation funnel that distills 115K fact-verification claims into a compact, learning-signal-dense subset of 5K claims. We show that a DecomposeRL-7B policy trained with full supervision on only ~5K curated claims achieves 86.3 in-domain and 69.8 out-of-domain balanced accuracy across 11 claim-verification benchmarks containing biomedical, political, scientific, and general-domain claims. Despite being 4x smaller, it matches 32B baselines and GPT-4.1-mini, and it further outperforms baselines in a semi-supervised setting with only 10% labeled claims data. Code, data, and models are available at this https URL 

---
# Fine-Tuned LLM as a Complementary Predictor Improving Ads System 

**Authors**: Hui Yang, Daiwei He, Kevin Jiang, Taejin Park, Kungang Li, Jiajun Luo, Yuying Chen, Xinyi Zhang, Sihan Wang, Haoyu He, Yu Liu, Lakshmi Manoharan, David Xue, Shubham Barhate, Runze Su, Duna Zhan, Ling Leng, Siping Ji, Jinfeng Zhuang, Alice Wu, Leo Lu, Han Sun, Zhifang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.27856)  

**Abstract**: Recommendation systems power engagement and monetization across feeds, ads, and short-video platforms, but translating the latest advances in Large Language Models into Recommendation Systems (RecSys) gains remains rare, particularly in advertising and production-scale real-world industry setups. Prior real-world LLM successes typically fall into three buckets: (a) generative retrieval that directly predicts the next items for candidate generation, (b) late-stage re-ranking that uses LLMs, and (c) auxiliary signal enrichment with LLMs. We introduce a complementary paradigm for ads: a fine-tuned open-source LLM used not as a ranker, but as an ads-specific ancillary predictor, forecasting likely advertisers from user profiles and histories. This LLM-driven advertiser prediction augments conventional candidate generation and provides informative priors to downstream ranking. Developed in a large-scale production advertising system, our approach produces substantial offline improvements and measurable online business impact, demonstrating that LLM world knowledge and predictive capacity can be efficiently harnessed. Beyond validating LLMs for ads applications, our results show that targeted ancillary predictions can unlock end-to-end gains across both retrieval and late-stage ranking, offering a practical path to LLM-enhanced recommendation at scale. 

---
# FPMoE: A Sparse Mixture-of-Experts Approach to Functional Code Generation 

**Authors**: Loc Pham, Lang Hong Nguyet Anh, Thanh Le-Cong  

**Link**: [PDF](https://arxiv.org/pdf/2605.27849)  

**Abstract**: Despite rapid progress in LLM-based code generation, existing models are predominantly trained on imperative languages, leaving functional programming languages (FPLs) such as Haskell, OCaml, and Scala chronically underexplored, with even frontier models performing substantially worse on FPLs. Fine-tuning is a natural remedy, but our experiments show that per-language fine-tuning fails to capture shared functional abstractions, while merged multi-language fine-tuning introduces cross-language interference. To address this, we introduce FPMoE, a lightweight, open-source code generation model built on a sparse Mixture-of-Experts (MoE) architecture with three language-specific routed experts (one each for Haskell, OCaml, and Scala) and a shared expert that captures cross-language functional patterns such as monadic reasoning and type-directed programming. This design resolves both failure modes simultaneously: dedicated experts eliminate interference, while the shared expert preserves abstractions that per-language models miss. On FPEval, FPMoE substantially outperforms fine-tuned baselines and, with only 3B active parameters, matches the performance of much larger models including DeepSeek-Coder-6.7B, Qwen2.5-Coder-14B-Instruct, and Qwen3-Coder-30B-A3B. 

---
# Snippet-Driven Supply Chain Discovery with LLMs: Scaling Visibility in China 

**Authors**: Hiroto Fukada, Takayuki Mizuno  

**Link**: [PDF](https://arxiv.org/pdf/2605.27845)  

**Abstract**: Financial and economic research often relies on structured supply-chain disclosures and commercial databases. In China, supplier--customer disclosure is typically limited to major partners of listed firms, leaving unlisted firms and long-tail inter-firm links poorly captured in structured data. Public web evidence can partly complement this gap through corporate, government, and trade-media disclosures; however, full-text web mining at scale is costly because pages are often inaccessible or expensive to process with large language models (LLMs). We propose a snippet-driven method for constructing a supply chain knowledge graph (SCKG), with firms as nodes and inter-firm relationships as edges. Web search snippets are query-biased summaries returned with search results. We use them as a scalable first-pass evidence layer for LLM-based relationship extraction. We evaluate the pipeline in terms of extraction efficiency and coverage. For extraction efficiency, exhaustive full-text chunking discovers 19.8$\times$ more unique relationships than snippets, but requires 251.2$\times$ more input tokens and yields higher redundancy. For coverage, we use 130,685 Chinese firms as search seeds, covering Shanghai/Shenzhen-listed firms and large unlisted firms as of 2024. In the listed-firm subset, the resulting SCKG covers 7.2$\times$ more firms and 9.3$\times$ more relationships than the CSMAR disclosure-based benchmark, while revealing heavy-tailed degree patterns. Retained provenance metadata make the SCKG an auditable complement to disclosure-based databases. 

---
# LoSATok: Low-dimensional Semantic-Acoustic Tokenizer for Cross-Domain Audio Understanding and Generation 

**Authors**: Zhisheng Zhang, Xiang Li, Yixuan Zhou, Jing Peng, Guoyang Zeng, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.27840)  

**Abstract**: Audio tokenizers are fundamental to unifying audio understanding and generation. Understanding requires high-level semantics, while generation demands semantic and acoustic details. Existing unified tokenizers jointly encode both in high-dimensional continuous latents, which increases the modeling burden of Diffusion Transformers (DiTs) for generation. We propose LoSATok, a low-dimensional audio tokenizer for cross-domain audio understanding and generation. Motivated by the observation that 1280-dimensional semantic encoder features are compressible, we introduce a Semantic Bottleneck that compresses them into 128 dimensions, regularized by the proposed time-relation loss for temporal feature consistency. We further design a dual-level semantic supervision method that leverages both high- and low-dimensional semantic signals, enabling the tokenizer to jointly capture semantics and acoustic details within a compact latent space. Experiments on speech, music, and general audio show that SemBo preserves strong low-dimensional semantic capacity and LoSATok retains competitive understanding performance compared with several semantic representations, while consistently improving DiT modeling performance on speech, music, and audio generation. These results demonstrate that LoSATok's low-dimensional representations can effectively support audio understanding and generation. Our code is provided at this https URL. 

---
# Symmetry Defeats Auditing 

**Authors**: Nick Merrill, Zeke Medley  

**Link**: [PDF](https://arxiv.org/pdf/2605.27836)  

**Abstract**: We demonstrate an attack on Introspection Adapters (Shenoy et al., 2026). 

---
# Disentangling Adversarial Prompts: A Semantic-Graph Defense for Robust LLM Security 

**Authors**: Xiang Fang, Wanlong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27823)  

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to adversarial prompts that exploit semantic ambiguities to bypass safety mechanisms, resulting in harmful or inappropriate outputs. Such attacks, including jailbreaking and prompt injection, pose significant risks to the integrity and availability of LLMs in security-critical applications. This paper proposes the Adversarial Prompt Disentanglement (APD) framework, a novel defense mechanism that proactively identifies and neutralizes malicious components in input prompts before they are processed by the LLM. The APD framework integrates three key innovations: (1) a mutual information-based semantic decomposition method to isolate adversarial and benign prompt components, ensuring statistical independence; (2) a graph-based intent classification approach that leverages spectral analysis to detect malicious patterns in prompt semantics; and (3) a lightweight transformer-based classifier trained on real-world datasets of toxic and jailbreaking prompts, enabling efficient and accurate adversarial intent detection. Evaluated on diverse datasets containing adversarial prompts, APD demonstrates superior robustness, reducing harmful output generation by over 85\% while maintaining negligible impact on model performance. The framework's computational efficiency supports real-time deployment, making it a practical solution for securing LLMs. Our work addresses critical challenges in machine learning security on novel attacks and integrity methods for ML systems, and offers a scalable, ethically grounded defense against prompt-based adversarial threats. 

---
# ReSAE: Residualized Sparse Autoencoders for Multi-Layer Transformer Interventions 

**Authors**: Prathyush Poduval, Calvin Yeung, Neel Desai, Mohsen Imani  

**Link**: [PDF](https://arxiv.org/pdf/2605.27819)  

**Abstract**: Sparse autoencoders are usually trained one layer at a time, even though transformer residual stream activations are strongly coupled across depth. This creates a practical problem for multi-layer interventions: different layerwise dictionaries can spend capacity representing the same carried-forward information, and replacing several layers at once can produce interactions that are not predicted by single-layer behavior. We introduce Residualized Sparse Autoencoders (ReSAEs), which fit an affine map between selected layers and train each later-layer SAE on the unexplained residual rather than on the full activation. Reconstructions are mapped back into the original activation space through the fitted affine chain, so ReSAEs can be evaluated with the same intervention protocols as ordinary SAEs. On Pythia-1.4B and Gemma-2-9B, residualization reduces decoder redundancy and improves sparse probing and targeted perturbation in most tested settings. Despite reconstructing less of the raw activation variance, ReSAEs recover more transformer cross entropy under multi-layer replacement. This gain is clearest under teacher-forcing and at sufficient sparsity online, indicating that ReSAEs preserve the components of the activation most relevant to the model's downstream computation. These results suggest that removing linearly predictable cross-layer structure is a useful default for multi-layer SAE interventions. 

---
# Turning Video Models into Generalist Robot Policies 

**Authors**: Sizhe Lester Li, Evan Kim, Xingjian Bai, Tong Zhao, Tao Pang, Max Simchowitz, Vincent Sitzmann  

**Link**: [PDF](https://arxiv.org/pdf/2605.27817)  

**Abstract**: Video generative models have emerged as a promising robotics backbone, capable of generating videos that depict the completion of complex tasks across embodiments and environments. Recent work proposes robot foundation models that jointly predict future observations and actions by finetuning video models with action-labeled data. In this paper, we test the limits of an alternative approach: leave the video planner as-is while training an embodiment-specific inverse dynamics model (IDM). This decoupling offers several natural benefits: the video planner remains embodiment-agnostic, different video models can be interchanged easily without re-training the IDM, and the IDM can be independently trained with readily available self-play data. We present a closed-loop, video-to-action policy that combines an action-free video world model with a carefully-designed IDM based on the robot embodiment Jacobian. We demonstrate that our IDM design is both data-efficient and scalable to high-dimensional action spaces. Our policy, which we coin the Video-to-Embodied Robot Action Model (VERA), achieves strong performance across simulated and real-world benchmarks, including zero-shot Panda arm manipulation and 16-DoF Allegro-hand dexterous cube re-orientation. The same video planner can be used across multiple embodiments by pairing it with different embodiment-specific IDMs. Our results show that decoupled video planning plus faithful video-to-action translation is a viable alternative route towards zero-shot, cross-embodiment, and generalizable robot control. More results are available on our project website: this https URL. 

---
# Residualized Temporal Sparse Autoencoders for Interpreting Diffusion Models 

**Authors**: Calvin Yeung, Prathyush Poduval, Ali Zakeri, Zhuowen Zou, Mohsen Imani  

**Link**: [PDF](https://arxiv.org/pdf/2605.27813)  

**Abstract**: Text-to-image diffusion models generate images through an iterative denoising process, so internal neural layers produce trajectories of activations rather than single static representations. Sparse autoencoders (SAEs) have recently been used to decompose diffusion activations into interpretable feature directions, but most approaches analyze activations at individual timesteps or condition on time rather than learning directly from full activation trajectories. In this work, we introduce residualized temporal SAEs for diffusion activation trajectories. We collect activations across denoising time, fit linear predictors between neighboring timesteps, and represent each trajectory using an initial activation together with residual components not explained by these linear dynamics. Training an SAE on this residualized representation encourages sparse latents to capture structure beyond what is linearly predictable. The residualized decoder directions can be mapped back into activation space, allowing each latent to be analyzed as a feature trajectory over denoising time. Through reconstruction and ablation studies, spatiotemporal feature analysis, and qualitative steering experiments on Stable Diffusion~1.5, we show that residualized temporal SAEs provide a useful framework for studying temporally structured diffusion activations. 

---
# ChildEval: When large language models meet children's personalities 

**Authors**: Yanyan Luo, Xue Han, Chunxu Zhao, Ruiqiao Bai, Yaxing Zhang, Qian Hu, Lijun Mei, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.27805)  

**Abstract**: While LLMs enable personalized chatbots, their effectiveness in child-centered personalization remains unclear, as systematic evaluation of child-specific preferences is still lacking. To address this gap, we introduce ChildEval, a benchmark for evaluating LLMs' ability to infer and follow child-centered preferences in long-context conversations. ChildEval contains 29K synthesized persona profiles of children aged 3-6, providing relatively static background information. Each persona is associated with a child preference-which may align with, conflict with, or be independent of the persona-expressed either explicitly in a single sentence or implicitly through 6-10 turn dialogues. Explicit and implicit preferences are designed to reflect the same underlying preference but differ in expression, capturing dynamic aspects of preference expression rather than changes in the static persona. The benchmark spans five top-level and fourteen sub-level categories covering children's daily lives and development. We further propose fine-grained, child-centric evaluation protocols to systematically assess open-source LLMs. Experimental results demonstrate how different personalized representations affect LLM responses and suggest that finetuning on ChildEval can enhance child-centered performance. Our code and dataset are available at this https URL. 

---
# Locality-Aware Redundancy Pruning for LLM Depth Compression 

**Authors**: Vincent-Daniel Yun, Youngrae Kim, Woosang Lim, YoungJin Heo, Minkyu Kim, Sunwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.27786)  

**Abstract**: Large language models are known to contain representational redundancy across network depth, making depth pruning an effective approach for improving inference efficiency. Existing one-shot pruning methods rely on local layer importance or fixed redundancy assumptions across architectures. We propose Locality-Aware Redundancy Pruning (LoRP), a training-free one-shot depth pruning framework guided by representation locality. We show that inter-layer redundancy can be either localized or globally distributed depending on the LLM architecture. To characterize this phenomenon, we introduce Representation Locality Score (RLS), derived from global inter-layer hidden-state similarity. Using a small calibration set, LoRP computes pairwise layer similarity, clusters layers by representational similarity, and allocates pruning according to residual intra-cluster redundancy. Experiments across diverse LLM families show improvements in both perplexity and downstream task accuracy. 

---
# Do Models Know Why They Changed Their Mind? Interpretability and Faithfulness of Chain-of-Thought Under Knowledge Conflict 

**Authors**: Pruthvinath Jeripity Venkata  

**Link**: [PDF](https://arxiv.org/pdf/2605.27773)  

**Abstract**: When a language model sees a document contradicting its training knowledge, it must choose: follow the document or trust itself. Prior work proved this choice depends on how well-known the fact is. We ask: does the model's chain-of-thought (CoT) reasoning faithfully report this mechanism? We introduce introspective faithfulness and test it across 200 questions, 8 models, and 4 prompt conditions. We find CoT reasoning is highly stable across opposite decisions: flip pairs retain 96% of same-answer similarity (d=0.34; confirmed by ROUGE-L, d=0.45). Yet self-rated confidence carries a faint genuine signal: for obscure facts where entity fame is uninformative, confidence still predicts decisions (p<0.001) and tracks item-level knowledge (r=0.134). GPT-4o is the only model with statistically reliable reasoning-decision coupling. Claude Sonnet 4.6 shows the widest confidence range (SD=1.39) but near-zero pooled correlation because the confidence-decision relationship reverses between conditions; a temperature ablation confirms this is model-specific. Internal thinking tokens show greater decision-sensitivity than user-facing CoT (p=0.033). CoT decomposes into a decision-invariant knowledge display (~96%) and a thin confidence layer with weak but real signal. For monitoring: read confidence, not the argument. 

---
# UniMaia: Steering Chess Policies with Language for Human-like Play 

**Authors**: Sherman Siu, Lesley Istead  

**Link**: [PDF](https://arxiv.org/pdf/2605.27767)  

**Abstract**: Recent advances in large language models have enabled natural language to serve as a flexible interface for controlling complex systems, but often at the cost of large-scale multimodal training or weakened domain-specific inductive biases. In structured decision-making domains such as chess, specialized policy networks achieve strong performance but lack semantic controllability, while prompt-conditioned language models are more flexible yet typically exhibit weaker domain grounding. We propose $\textbf{UniMaia}$, a framework for prompt-conditioned policy modulation that adapts a frozen Lc0-based chess policy network using a parameter-efficient text encoder and a ControlNet-style conditioning mechanism. UniMaia enables semantic control over gameplay, including opening selection and player strength, while preserving the pretrained policy representations. We further introduce $\textbf{UniMaia-Aux}$, which incorporates auxiliary temporal conditioning and behavioral prediction objectives. To support this work, we construct a large-scale metadata-augmented Lichess dataset, develop a semi-automated prompt-generation pipeline, and introduce benchmarks spanning both prompt-conditioned and metadata-conditioned settings. UniMaia achieves state-of-the-art expected accuracy on several prompt-conditioned benchmarks and competitive top-move accuracy on general instruction-following tasks, while remaining competitive with dedicated metadata-conditioned approaches on human move prediction benchmarks. UniMaia-Aux further improves expected accuracy and behavioral modeling across several evaluation settings, with modest trade-offs in top-move accuracy. Overall, our results demonstrate that prompt-conditioned control of domain-specific policy networks is feasible without end-to-end multimodal training, while highlighting trade-offs between controllability and predictive performance. 

---
# Restoring the Sweet Spot: Pass-Rate Weighted Self-Distillation for LLM Reasoning 

**Authors**: Zehao Liu, Yuanpu Cao, Jinghui Chen, Vasant G. Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2605.27765)  

**Abstract**: Self-Distillation Policy Optimization (SDPO) provides dense token-level credit assignment for reinforcement learning with large language models by leveraging the model's own feedback-conditioned predictions as a self-teacher. Unlike GRPO, however, whose group-relative advantage naturally concentrates learning on a sweet spot of intermediate-difficulty questions, SDPO's KL-based advantage lacks an implicit notion of difficulty awareness.
We analyze this gap through the lens of GRPO's advantage normalization. Extending the learnability framework to normalized rewards, we show that normalization absorbs the variance term $p(1-p)$, equalizing leading-order learnability across questions and leaving $\sqrt{p(1-p)}$ as the sole residual scaling factor in the per-question gradient. This analysis yields a simple prescription: weight each question's SDPO loss by $[\hat{p}(1-\hat{p})]^{1/2}$, resulting in SC-SDPO, a scale-consistent variant of SDPO.
The proposed weights are obtained as a zero-cost byproduct of on-policy rollouts with batch-adaptive normalization, inducing an implicit curriculum that dynamically tracks the model's evolving competence. Experiments on scientific reasoning and tool-use benchmarks demonstrate that SC-SDPO consistently improves over SDPO, yielding gains of +3.2/+4.3 (mean@16/maj@16) on Qwen3-8B and +1.8/+3.0 on OLMo-3-7B, while preserving stable training dynamics throughout optimization. 

---
# Can Segmentation Models Understand the World? Towards Proactive Affordance Reasoning via Visual Chain-of-Thought 

**Authors**: Yuchen Guo, Junli Gong, Hongmin Cai, Yiu-ming Cheung, Weifeng Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.27764)  

**Abstract**: Recent segmentation models couple large language models (LLMs) with mask decoders to ground complex language expressions into masks, yet their instructions remain target-referential: they describe, constrain, or imply the region to be segmented. However, in real-world embodied interaction, human instructions are often at the intent-level, which includes the desired outcome without naming the region that enables it. To bridge this gap, we introduce SegWorld, where the model reasons about the scene through a multi-level visual chain-of-thought (CoT) before committing to a mask. Before receiving any instructions, it proactively observes the scene, describing visible objects and inferring plausible events they may support. Given an instruction, it continues the chain: from the object relevant to the intent, through the action that satisfies it, to the physical interaction site, the object part that affords the action. We formalize SegWorld as probabilistic inference, in which proactive observation supplies a linguistic scene context that improves mask prediction when instructions are given at the level of intent. We construct an intent-to-part benchmark for evaluating affordance-bearing part segmentation from high-level goals. Experiments show SegWorld matches instruction-driven baselines on target-referential instructions and improves substantially on intent-level ones. 

---
# High-Fidelity Industrial Crash Dynamics Prediction via Geometry-Aware Operator Learning with Memory-Efficient Low-Rank Attention 

**Authors**: Deepak Akhare, Mohammad Amin Nabian, Corey Adams, Sudeep Chavare, Sanjay Choudhry  

**Link**: [PDF](https://arxiv.org/pdf/2605.27758)  

**Abstract**: Automotive crashworthiness optimization remains a safety-critical challenge, requiring the management of large-scale nonlinear structural deformations and energy dissipation through iterative, high-fidelity simulations. While traditional finite element solvers are computationally prohibitive, emerging operator learning frameworks provide rapid surrogate predictions; however, applying them to industrial-scale crash analysis, where complex geometry, contact nonlinearities, and rapidly evolving transient deformation coexist, remains an open challenge. In this paper, we demonstrate that the GeoTransolver framework provides a viable solution for accurate, high-fidelity crash dynamics prediction at industrial scale. Benchmarked on complex bumper beam and full-vehicle crash datasets, GeoTransolver captures multi-scale geometric context and accurately resolves plastic deformation patterns as well as acceleration profiles at critical occupant locations. Beyond the architecture itself, we propose and systematically evaluate a suite of temporal prediction recipes, including one-shot, time-conditional, and autoregressive rollout strategies, demonstrating that the one-shot approach achieves state-of-the-art accuracy with significantly reduced training overhead and inference latency. As a secondary contribution, we introduce a Fast Low-rank Attention Routing Engine (FLARE)-based modification to the GeoTransolver attention backbone that reduces memory overhead by approximately 2x while further improving predictive accuracy for O(N) long-range, high-frequency transients, preserving the geometry-aware cross-attention strengths of the base framework. Our results highlight the practical viability of geometry-aware operator learning for high-fidelity surrogate modeling of complex, safety-critical automotive dynamics. 

---
# Reading or Guessing? Visual Grounding Failures of Vision-Language Models for OCR in Ancient Greek Editions 

**Authors**: Antonia Karamolegkou, Nicolas Angleraud, Benoît Sagot, Thibault Clérice  

**Link**: [PDF](https://arxiv.org/pdf/2605.27750)  

**Abstract**: Recent work has shown that Vision-Language Models (VLMs) used for optical character recognition (OCR) can generate plausible but visually unsupported text, suggesting reliance on language priors. Comparing open-weight VLMs with traditional OCR baselines on low-resource Ancient Greek critical editions, we show that VLM errors often remain fluent even when wrong, producing plausible Greek substitutions where traditional engines produce local recognition noise. To analyze visual evidence during decoding, we introduce controlled image perturbations and token-level grounding measures based on conditional versus image-free decoding distributions. Under character-level perturbations, VLMs diverge sharply from the perturbed ground truth while traditional OCR remains comparatively faithful; however, token-level analysis shows that prior reliance is model-specific: in an OCR-specialist model, fluent lexical errors are produced with little reliance on the image, whereas general-purpose VLMs remain conditioned on the visual input even when wrong. Decode-time interventions fail to reliably restore grounding, while post-OCR language-model correction improves several systems only by repairing text after generation. Our results extend prior evidence of OCR language-prior reliance to low-resource historical documents and a broader set of models, showing that fluent output is not necessarily visually grounded and motivating interpretability-driven evaluation beyond aggregate accuracy. 

---
# Mahalanobis PatchCore: Covariance-Aware and Streaming-Compatible Industrial Anomaly Detection 

**Authors**: Niccolò Ferrari, Oligert Osmani, Evelina Lamma  

**Link**: [PDF](https://arxiv.org/pdf/2605.27748)  

**Abstract**: Industrial visual anomaly detection is usually one-class: normal images are abundant, while defects are rare, heterogeneous, and often unavailable during system design. PatchCore-style retrieval suits this setting because it scores test images from a memory bank of normal patch features, but the standard Euclidean geometry ignores feature correlations and its offline construction materialises the full patch pool before subsampling.
We introduce Mahalanobis PatchCore, a covariance-aware, streaming-compatible extension of PatchCore. Its artificial intelligence contribution is a retrieval detector that estimates a regularised covariance model in reduced feature space and whitens embeddings, so Euclidean nearest-neighbour search after transformation implements Mahalanobis retrieval. A bounded-memory, re-iterable training pipeline builds the memory bank without storing all normal patches at once, using incremental dimensionality reduction, online covariance estimation, and streaming aggregation.
The engineering application is automated industrial inspection, where visual anomaly detection must remain accurate under practical memory limits. We evaluate the method on a public 15-category industrial anomaly-detection benchmark and three industrial datasets covering blow-fill-seal strip-ampoule meniscus inspection, amber-glass-ampoule bottom inspection, and lyophilised-cake vial inspection. Mahalanobis PatchCore preserves most offline PatchCore image-level performance on the public benchmark while reducing peak memory from 5.41 to 2.78 GB, and improves the selected industrial mean image area under the receiver operating characteristic curve from 0.981 to 0.986. 

---
# Worker Disagreement Reveals Sharp Directions in Local SGD 

**Authors**: Tolga Dimlioglu, Kristi Topollai, Anna Choromanska  

**Link**: [PDF](https://arxiv.org/pdf/2605.27739)  

**Abstract**: Deep neural network training often exhibits highly anisotropic loss geometry, where a few sharp dominant Hessian directions coexist with a large flatter bulk. Gradients tend to align disproportionately with these dominant directions, although stable progress often requires movement through flatter bulk directions. Estimating the dominant subspace is therefore useful but costly with direct Hessian-based methods. We show that standard Local SGD exposes this geometry through worker disagreement. We theoretically show that the worker-average gap covariance is shaped by stochastic-gradient noise and Hessian curvature, causing workers to disagree along sharp, curvature-sensitive directions. Thus, worker-average gaps provide a cheap Hessian-free estimator of the dominant subspace. Experiments on MLPs, CNNs, and Transformers show that subspaces formed by worker-average gaps capture a substantial fraction of the gradient component lying in the dominant Hessian eigenspace. 

---
# HumanoidMimicGen: Data Generation for Loco-Manipulation via Whole-Body Planning 

**Authors**: Kevin Lin, Ajay Mandlekar, Caelan Reed Garrett, Nikita Chernyadev, Yu Fang, Runyu Ding, Yuqi Xie, Justin Tran, Linxi Fan, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.27724)  

**Abstract**: Imitation learning is a promising approach for training humanoid robots to both walk and manipulate, but it requires a large number of demonstrations, which are time-intensive and difficult to collect via teleoperation. Existing data-generation algorithms can automatically synthesize demonstrations for manipulators, but they are ineffective on humanoids because their high-dimensional composite action spaces involve arms, legs, and torsos. We present HumanoidMimicGen, a method for generating humanoid legged loco-manipulation data. Our method adapts contact-rich whole-body skills from a handful of source demonstrations to new states, generalizing across changes in object pose. By interleaving these single- and dual-arm skills with whole-body locomotion and manipulation planning, the method generates stable, collision-free data across diverse scenes and layouts. To evaluate our approach, we introduce a new simulated loco-manipulation benchmark containing nine diverse tasks that test humanoid loco-manipulation capabilities. There, we demonstrate that HumanoidMimicGen automatically generates large datasets for imitation learning and enables a systematic study of how data generation and policy learning decisions impact model performance. We show that whole-body visuomotor policies co-trained with data generated by HumanoidMimicGen outperform those trained only on real-world data by 20%. 

---
# UserHarness: Harnessing User Minds for Stronger Agent Theory-of-Mind 

**Authors**: Cheng Qian, Jiayu Liu, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.27721)  

**Abstract**: Understanding what a user believes and intends is central to building effective agent assistants. This ability is often evaluated through Theory-of-Mind (ToM) tasks, where success requires reasoning from the user's perspective. However, many existing approaches address ToM with complex pipelines that model behavior indirectly, without explicitly reconstructing the user's mental state. This misses the core structure of the problem: users act based on their beliefs, which are updated through observations of the environment; beliefs and intentions jointly determine actions, which in turn change the environment; and social reasoning often requires nested beliefs about what others believe or intend. We propose UserHarness, a simple framework that reframes ToM reasoning as explicit user-mind reconstruction. UserHarness decomposes the user's mental state, its relation to the external environment, and the actions that follow from it, enabling agents to track what the user observes, believes, intends, and does. Across five benchmarks, UserHarness reaches up to 95.94% macro accuracy, improving over existing inference methods by more than 15% relative and over the strongest prompt-only harness by about 20% relative. These results suggest that robust user understanding requires reasoning from the roots of the user's mind, positioning user harnessing as a promising foundation for more adaptive future assistants. 

---
# CiteCheck: Retrieval-Grounded Detection of LLM Citation Hallucinations in Scientific Text 

**Authors**: Khashayar Khajavi, Shaghayegh Sadeghi, Rise Adhikari, Alexander Tessier  

**Link**: [PDF](https://arxiv.org/pdf/2605.27700)  

**Abstract**: Large language models (LLMs) are increasingly used to generate scientific reports, but they can produce references that appear plausible while containing corrupted metadata or pointing to papers that do not exist. We introduce CiteCheck, a hybrid framework for citation hallucination detection that verifies whether a citation corresponds to a real scholarly work and whether its metadata is faithful to that work. CiteCheck retrieves candidate publications from external scholarly sources, compares the citation against the retrieved candidate using a structured LLM verifier, and maps verifier scores into three labels: Exact, Minor, and Major. We also construct a 982-citation physics benchmark with controlled corruptions that capture both subtle metadata drift and fully fabricated references. On the held-out test set, CiteCheck achieves 88.7 macro-F1 and 88.9% accuracy, outperforming GPT, Claude, and Gemini baselines, including web-search and few-shot variants. These results show that reliable citation verification benefits from combining scholarly retrieval, structured LLM-based comparison, and calibrated decision rules. 

---
# Simulation-Informed Diffusion for Decentralized Multi-robot Motion Planning 

**Authors**: Jinhao Liang, Sven Koenig, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2605.27697)  

**Abstract**: Decentralized multi-robot motion planning requires each robot to generate collision-free trajectories from local observations, without global sensing or reliable communication. However, most existing planners, whether classical or learning-based, generate trajectories from a static snapshot of the local observation, which limits their ability to anticipate the future behavior of neighboring robots. This limitation is critical as the number of robots increases and the environment becomes more cluttered. To overcome this challenge, this paper introduces Simulation-Informed Diffusion (SID), a decentralized framework built on constraint-aware diffusion models (CADM). SID first uses CADM to simulate the future trajectories of neighboring robots from their currently observed states, and then uses the same CADM to plan each robot's own trajectory under safety constraints informed by these simulations. Crucially, the accurate simulation of neighbors enables a minimal communication scheme that triggers coordination only when necessary in highly congested scenarios. Experiments across diverse environments show that SID consistently outperforms baseline methods in terms of planning effectiveness and constraint satisfaction, and scales to scenarios with 108 robots and 160 obstacles. 

---
# Tensor Memory: Fixed-Size Recurrent State for Long-Horizon Transformers 

**Authors**: Kabir Swain, Sijie Han, Daniel Karl I. Weidele, Mauro Martino, Antonio Torralba  

**Link**: [PDF](https://arxiv.org/pdf/2605.27686)  

**Abstract**: Transformers process images and videos by flattening space and time into long token sequences. While attention and KV caching preserve past features, their memory grows with sequence length and they lack an explicit, persistent spatial state, making long-horizon video understanding and occlusion-sensitive reasoning difficult. We propose Tensor Memory, a lightweight module that augments Transformer blocks with a fixed-size recurrent 3D memory tensor: tokens write into a voxel grid via a differentiable soft write that deposits content as a Gaussian-weighted volume around a predicted continuous 3D location, the memory is updated with an efficient local interaction operator and gated recurrent dynamics, and tokens read back context via continuous sampling with gated residual fusion. Because the memory tensor has a constant size, Tensor Memory decouples state capacity from input length while preserving a spatial inductive bias. We evaluate the module on standard language, image, and video benchmarks and on a controlled toy diagnostic suite designed to isolate when persistent state is beneficial; it integrates with standard Transformer training pipelines and can be attached to or removed from existing blocks without other architectural changes. 

---
# Backdoor Attacks on Fault Detection and Localization in Cyber-Physical Systems 

**Authors**: Abile Jean, Kuniyilh S  

**Link**: [PDF](https://arxiv.org/pdf/2605.27674)  

**Abstract**: Cyber-Physical Systems (CPS) integrate sensing, communication, computation, and control to support critical infrastructure, including smart grids, industrial automation, and control systems. In the electrical utility domain, various controllers are used in CPS to ensure the system detects and recovers from faults, such as voltage fluctuations, and to perform load balancing in distribution systems. Machine learning- and deep learning-based fault detection and localization frameworks have recently gained significant attention in CPS for their ability to identify anomalies and operational failures in real time. However, these intelligent models are vulnerable to adversarial machine learning attacks, particularly backdoor attacks. In a backdoor attack, an adversary injects malicious patterns into the training data so that the model behaves normally most of the time but produces attacker-controlled outputs when triggered by specific patterns. This paper investigates the threat of backdoor attacks against fault detection and localization mechanisms in recent ML pipelines used in modern CPS systems. We define these threats and explore how they can be realized by designing triggers and evaluating their success in the CPS domain. Our experiments show the attack is successful even with 10\% of poisoning. 

---
# Aligning LLMs with Human Uncertainty: A Beta-Bernoulli Calibrator for LLM Forecasting 

**Authors**: Hui Dai, Ryan Teehan, Parsa Torabian, Mengye Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.27668)  

**Abstract**: Probabilistic forecasting estimates the likelihood of uncertain future events. To improve LLM forecasting, existing methods typically learn from binary outcomes to output verbalized forecasts. However, while aggregated human forecasts contain rich information in both the crowd probability estimate and the degree of agreement among forecasters, how to utilize these signals remains underexplored. To address this, we propose the Beta-Bernoulli Calibrator (BBC), which converts an initial point estimate forecast from any model into a distribution over event likelihood, using supervision from both binary outcomes and human forecasts. BBC models event likelihood $p \sim \text{Beta}(\alpha, \beta)$ and outcome $y \sim \text{Bernoulli}(p)$, with the mean as the calibrated point forecast and the variance as the epistemic uncertainty. Our results show that BBC generally provides better calibrated and more accurate forecasts than both traditional post-hoc calibration methods and models fine-tuned specifically for forecasting, while remaining lightweight and having good generalization. We also show that the epistemic uncertainty captured by BBC is a more reliable predictor of forecasting error than verbalized confidence. 

---
# How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks 

**Authors**: Teodor-Mihai Stupariu, Andrei Manolache  

**Link**: [PDF](https://arxiv.org/pdf/2605.27662)  

**Abstract**: Equivariant neural networks encode geometric symmetries by construction, yet they are often difficult to optimize and can underperform less constrained architectures. A growing body of work addresses this through architectural modifications such as constraint relaxation or approximate equivariance, while the role of the optimizer remains comparatively underexplored. We study this direction by comparing Muon and Adam across several equivariant and geometric architectures under pointcloud and molecular learning settings. On ModelNet40, where the comparison is clearest, Muon consistently improves over Adam across all architectures considered. We then analyze the trained ModelNet40 checkpoints through Hessian estimates, loss surface visualizations, and spectral properties of learned weights and intermediate representations. The checkpoints reached by Muon have larger Hessian curvature summaries but more regular loss surfaces, and their learned weights and representations have higher stable and effective ranks. These observations suggest that the interaction between optimizer design and geometric inductive bias deserves further attention from the community. 

---
# Transferable Reinforcement Learning via Probabilistic Latent Embeddings and Dynamic Policy Adaptation for Sim-to-Real Deployment 

**Authors**: Gengyue Han, Yiheng Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.27659)  

**Abstract**: Due to limited resources and public safety concerns, deep reinforcement learning (RL) agents for many cyber-physical systems (e.g., autonomous vehicles) are first trained in simulators. However, when deployed in real world environments, they often suffer from performance degradation or safety violations because of the inevitable Sim2Real gap. Existing zero-shot approaches, such as robust safe RL and domain randomization, mitigate this issue but typically at the cost of degraded performance or residual safety risks when experiencing unmodeled system dynamics. To address these limitations, we propose a novel reinforcement learning framework that enables safe and efficient policy transfer via probabilistic latent embeddings and dynamic policy adaptation. We consider a family of Constrained Markov Decision Processes (CMDPs) under different environment contexts. By leveraging latent context variable in meta-RL, the proposed framework infers the latent representation of the environment from simulated experiences. Furthermore, it incorporates a distributional RL formulation, which allows risk levels of the deployed policy to be adjusted dynamically, based on the estimation accuracy of the latent context variable. This strategy promotes safety at the early deployment stage and improves efficiency through fast policy adaptation under the Sim2Real gap. 

---
# Developing an Intelligent Job Recommendation System Using Semantic Retrieval and Explainable AI Techniques 

**Authors**: Hussein Al Awad, Khaled Fathi Omar  

**Link**: [PDF](https://arxiv.org/pdf/2605.27656)  

**Abstract**: Online recruitment platforms require recommendation methods capable of retrieving relevant job opportunities from large and heterogeneous collections of job postings. Keyword-based search is efficient and interpretable, but it may fail to retrieve relevant postings when equivalent roles are expressed using different terminology. This study presents a metadata-driven job recommendation system that combines TF-IDF lexical matching, Sentence-BERT semantic retrieval, query-aware filtering, optional Cross-Encoder re-ranking, and explanation generation. The proposed system utilizes structured metadata fields including job title, company name, location, seniority level, job function, employment type, and industry without relying on full job descriptions or user interaction histories. Experiments conducted on a cleaned LinkedIn job posting dataset containing 31262 records demonstrate that the best hybrid configuration achieved a Precision at 10 score of 0.8032 and an nDCG at 10 score of 0.9496. Under the internal evaluation protocol, Cross-Encoder re-ranking improved Precision at 10 from 0.7896 to 0.7948 and nDCG at 10 from 0.9666 to 0.9739. These findings indicate that lexical and semantic retrieval techniques can be effectively combined to provide explainable job recommendations when only structured metadata is available. 

---
# Cultural Fidelity in English-to-Hindi Translation: A Preservation-Fluency Frontier for Gender Recoverability 

**Authors**: Samyak Savi, Chavi Gupta, Shreyas Gantayet, Tanay Sodha, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.27654)  

**Abstract**: Generative translation systems are cultural technologies because they decide how socially meaningful cues are rendered within culturally specific grammatical systems. We study one concrete notion of successful cultural translation: when an English source explicitly encodes gender, an English-to-Hindi translation should preserve the recoverability of that cue unless the source itself is ambiguous. We evaluate this criterion on a 37,345-instance benchmark spanning twelve categories and show that five systems frequently erase gender through ergative and honorific constructions. We then introduce two mechanism-aware inference-time interventions. The first, the Source-Aware Reranker (SAR), prefers candidates that avoid gender-neutralizing syntax. The second, the Phenomenon-Aware Reranker (PAR), preserves gender through targeted lexical marking even when ergative syntax remains. Across GPT-4o-mini and Sarvam, PAR improves target-subset accuracy from 11.07% to 54.47% and from 15.99% to 49.66%, respectively. Human evaluation shows that PAR increases gender preservation from 10.3% to 81.3%, but reduces mean fluency from 4.36 to 3.37. These findings place the two interventions on a preservation and fluency frontier rather than supporting a single dominant solution, and show how culturally situated generation can require explicit tradeoffs among fidelity, fluency, and stylistic naturalness. 

---
# Hurwitz Quaternion Multiplicative Quantization for KV Cache Compression 

**Authors**: Kabir Swain, Sijie Han, Daniel Karl I. Weidele, Mauro Martino, David Cox, Antonio Torralba  

**Link**: [PDF](https://arxiv.org/pdf/2605.27646)  

**Abstract**: We propose \textbf{Hurwitz Quaternion Multiplicative Quantization (HQMQ)}, a \textbf{calibration-free} method for KV cache compression of large language models. HQMQ treats each 4-element chunk of K or V as a quaternion and quantizes its unit direction to the \emph{product} $q_p \cdot q_s$, where $q_p$ ranges over the 24-element Hurwitz group $2T$ (the 24 vertices of the 24-cell on $S^3$, pairwise angle $60^\circ$) and $q_s$ ranges over a per-(layer, head) secondary codebook of $S$ \emph{random} unit quaternions. The multiplicative composition yields $24S$ effective codewords at $S$ stored parameters; random initialization suffices because left-multiplication is an $S^3$ isometry, so seeded codebooks vary in end-task ppl by $<1.5\%$. A per-batch median-multiplier outlier extraction step ($C{=}3$, no calibration) handles modern outlier-heavy architectures. We evaluate on five modern open models: Mistral-7B (dense MHA), Llama-3-8B and Qwen2.5-7B and Qwen3-8B (dense GQA), and gpt-oss-20b (sparse MoE). On Mistral-7B and Qwen3-8B, HQMQ matches fp16 within $0.02$--$0.03$ ppl points at $\sim$5 bits. On Qwen2.5-7B and Qwen3-8B, where naive int4 collapses to $10^4{+}$ ppl, HQMQ + Med3$\times$ recovers fp16 quality within $0.02$--$0.10$ ppl points at $\sim$5 bits. HQMQ Pareto-dominates naive int by $3$--$1900\times$ at matched bits across all five models, and downstream zero-shot accuracy matches fp16 at $3.79$ bits on Mistral. Against the strongest calibrated KV-quantization baseline, HQMQ at $3.79$ bits matches KIVI-4 ($\sim 4.5$ bits) within ${\sim}1$ pt on CoQA, $0.6$ pts on TruthfulQA, and $2.3$ pts on GSM8K, at $16\%$ fewer bits and without a calibration pass. At the storage level, HQMQ delivers up to $5.05\times$ KV compression, shrinking a Llama-3-70B 128k-context cache from 43 GB to 8.5 GB. 

---
# Trinity: Unifying Class-Agnostic Terrain and Semantic Segmentation for Unstructured Outdoor Environments by Leveraging Synthetic Data 

**Authors**: Marcus G Müller, Wout Boerdijk, Maximilian Durner, Riccardo Giubilato, Abel Gawel, Wolfgang Stürzl, Roland Siegwart, Rudolph Triebel  

**Link**: [PDF](https://arxiv.org/pdf/2605.27644)  

**Abstract**: Terrain understanding is fundamental for mobile robots operating in unstructured outdoor environments. Existing vision-based traversability estimation methods rely on robot-specific annotations or semantic class mappings, limiting transferability across platforms and requiring costly re-annotation when robot capabilities change, while standard semantic segmentation methods only focus on specific predefined classes, which do not capture the variety of terrains. In this work, we propose a transformer-based architecture that jointly performs class-specific semantic segmentation and class-agnostic terrain segmentation within a unified network, called Trinity. Terrain regions are segmented based solely on visual appearance, without predefined semantic labels or robot-dependent traversability scores. This formulation enables the learning of robot-agnostic visual terrain priors that can be combined with robot-specific experience for downstream tasks such as traversability estimation, visual odometry, and mission planning. To enable large-scale training with diverse terrain appearances, we extend the OAISYS simulator and introduce RUGDSynth, a synthetic dataset inspired by RUGD with class-agnostic terrain samples. Furthermore, we present the EXTerra Dataset, providing real-world images annotated with both class-specific and class-agnostic terrain labels. Experiments demonstrate the feasibility of the proposed task and the effectiveness of our joint segmentation approach in complex outdoor environments. Code and datasets will be released with this publication (after review). 

---
# Supervised Distributional Reduction via Optimal Transport and Dependence Maximization 

**Authors**: Sai-Aakash Ramesh, Archit Sood, Andrew Corbett, Tim Dodwell  

**Link**: [PDF](https://arxiv.org/pdf/2605.27619)  

**Abstract**: Learning representations that capture both intrinsic data geometry and target-relevant structure remains a fundamental challenge, particularly in settings where data reduction must balance compression with predictive fidelity. While distributional reduction-encompassing joint clustering and dimensionality reduction-offers a principled way to summarize data, its supervised variants remain relatively under-explored, despite the importance of retaining task-relevant signal for downstream prediction and decision-making. We propose Supervised Distributional Reduction (SDR), an algorithm for learning target-aware representations by combining optimal transport with explicit dependence maximization. SDR builds on the Fused Gromov-Wasserstein (FGW) objective to align the relational structure of the input distribution with a set of representative points, while augmenting it with a direct dependence term that encourages the learned embeddings to capture predictive signal more explicitly. This results in compact representations that reflect both geometric structure and supervision. Beyond representation learning, SDR naturally induces a data-dependent, non-stationary geometry that can be leveraged for settings such as Gaussian Process (GP) modelling. By redefining distances through target-aware distributional alignment, SDR enables the construction of adaptive kernels that respond to local variations in both data geometry and supervision, offering an optimal transport-based perspective on non-stationary kernel design. 

---
# Not All NVFP4 QAT Recipes Are Equal: How Architecture and Scale Shape Model Quality for Anomaly Segmentation 

**Authors**: Zijian Du, Oleg Rybakov  

**Link**: [PDF](https://arxiv.org/pdf/2605.27616)  

**Abstract**: Real-time anomaly segmentation demands both high recall and efficient low-precision inference. We study the three-way interaction of model architecture, model scale, and FP4 quantization-aware training (QAT) recipe on a recall-critical brain tumor segmentation task, evaluating multiple architectures, scales, and QAT recipes under a unified protocol. We find that architecture choice has the largest impact on quantization robustness, with attention-based architectures showing remarkable resilience to recipe choice while CNN degrades under gradient-quantizing recipes at larger scales. At low capacity, FP4 can discretize softmax attention, but advanced QAT recipes prevent this collapse. At larger scales, advanced recipes mitigate gradient quantization noise that degrades CNN quality. Five-fold patient-level cross-validation confirms these findings are robust to data partition. Our results show that the Swin Transformer is robust to QAT recipe choice across all scales, making it the recommended architecture for FP4-quantized anomaly segmentation. 

---
# Eliot: Interactively $\underline{E}$xploring Fast-Changing Scientific $\underline{Li}$terature Trends with $\underline{O}$nline Da$\underline{t}$a and Learning 

**Authors**: Bernardo A. Denkvitts, Nitin Gupta, Biplav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2605.27610)  

**Abstract**: The rapid growth of scientific publishing has made it increasingly difficult to track how fast-moving areas evolve. Search engines and LLM-based assistants retrieve or summarize papers, but often hide how the corpus was selected, organized, or connected to temporal patterns. We present $\texttt{Eliot}$, a publicly deployed interactive system for traceable exploration of evolving scientific literature. Motivated by two studies on Large Language Models (LLMs) and Automated Planning and Scheduling (APS), $\texttt{Eliot}$ generalizes literature-evolution analysis beyond hand-built taxonomies and domain-specific scripts. Given explicit query terms and filters, it retrieves arXiv papers at query time, represents each paper by title and abstract, clusters the corpus into themes, assigns representative keywords, and visualizes each cluster's publication-year distribution. We evaluate $\texttt{Eliot}$ as both an applied system and an interactive research aid. An offline configuration study across eight arXiv domains compares document representations, dimensionality reduction methods, and clustering algorithms using intrinsic clustering and topic-coherence metrics; the results support MiniLM embeddings with 10-dimensional UMAP and Agglomerative Clustering as a practical default. A scenario-based survey and expert focus group assess interpretability and use contexts: participants rated cluster labels as meaningful in 85% of scenario responses, and feedback indicated that $\texttt{Eliot}$ is most valuable for auditable overviews of rapidly changing technical areas. These results suggest that query-time clustering and temporal inspection can complement search and generation tools by helping researchers inspect and refine the evidence behind literature trends. 

---
# The Energy Blind Spot: NVIDIA's Flagship Edge AI Hardware Cannot Support Process-Level Energy Attribution 

**Authors**: Deepak Panigrahy, Aakash Tyagi  

**Link**: [PDF](https://arxiv.org/pdf/2605.27599)  

**Abstract**: Agentic AI workloads - where a single user goal triggers multi-step orchestration, tool calls, retries, and failure recovery - are being targeted for edge deployment, with NVIDIA, Dell, HP, ASUS, MSI, Acer, and Gigabyte all shipping GB10-based desktop AI systems in 2026. We recently demonstrated that orchestration structure dominates agentic energy cost, with workflows consuming 4.33x more energy per successful goal than linear baselines and OOI reaching 7.63x for multi-step reasoning tasks. Separately, Rajat et al. show that CPU-side processing accounts for up to 90.6% of total latency and 44% of total dynamic energy in agentic workloads. We report a systematic energy-observability audit of the ASUS Ascent GX10 (GB10 SoC) and find that the platform exposes no CPU energy counter, no INA power-rail monitor, no IPMI/BMC, and no SCMI powercap protocol through any supported software interface. The only on-device energy telemetry is instantaneous GPU power via NVML. We further discover that the MediaTek firmware already computes per-rail energy internally via an undocumented ACPI interface (SPBM), but NVIDIA states there are "no plans to expose CPU rail information." On-device per-process energy attribution - as performed on x86 via RAPL - is therefore not reproducible on this platform through supported interfaces. We formalize a hardware requirements specification for energy-attributed AI, propose an interim calibration bridge using external DC metering combined with GPU subtraction, and identify a standards-track path via SCMI powercap. Our findings motivate the low-carbon computing community to demand energy observability as a first-class hardware requirement. 

---
# Hallucination Behavior in Multimodal LLMs Across Agricultural Image Interpretation and Generation Tasks 

**Authors**: Partho Ghose, Al Bashir, Prem Raj, Azlan Zahid  

**Link**: [PDF](https://arxiv.org/pdf/2605.27595)  

**Abstract**: Large Language Models (LLMs) are being rapidly adopted in agricultural imaging applications, ranging from crop interpretation to synthetic field image generation. However, these models frequently exhibit hallucinations outputs that appear confident yet deviate from biological or environmental reality potentially leading to misinformed agronomic insights. This study investigates such hallucinations in two complementary directions: image-to-text, where LLMs interpret crop or field imagery to describe conditions such as biotic and abiotic stresses, and text-to-image, where models generate synthetic agricultural scenes based on descriptive prompts. We examine errors involving biological inconsistency, contextual inaccuracy, and agronomic implausibility, evaluating the outputs under domain-informed criteria across multiple imaging modalities. Our analysis identifies recurring hallucination patterns within both interpretive and generative tasks. In image interpretation, LLMs (e.g., Gemma, LLAVA, Qwen, and MiniCPM) achieved modest zero-shot accuracy (63 to 75 percent), whereas few-shot prompting improved performance up to 86.8 percent, exhibiting false detections and missed infections, indicating residual hallucination effects. In text-to-image tasks, advanced models such as GPT-5 and Gemini 2.5 Flash generate up to 91 percent biologically inconsistent scenes under relaxed prompt constraints, revealing fundamental weaknesses in current LLMs. This systematic assessment of visual reasoning and generation offers critical insights toward enhancing the reliability and trustworthiness of LLM-based agricultural imaging platforms. 

---
# The Future of Facts: Tracing the Factual Generation-Verification Gap 

**Authors**: Tim R. Davidson, Anja Surina, Caglar Gulcehre  

**Link**: [PDF](https://arxiv.org/pdf/2605.27564)  

**Abstract**: Language models are becoming the default interface to factual knowledge, yet they often verify outputs more reliably than they generate them. This generation-verification gap (GV-gap) underlies many recent advances in self-improvement and reasoning, but its dynamics on factual knowledge specifically remain poorly understood. We focus on the training mechanisms underlying factual GV-gaps, distinguishing them from their computational and aesthetic counterparts. We trace generation and verification capabilities through three training phases (acquisition, continual learning, and updating) across four open-source model families at two scales each. Three findings recur across models: (i) verification is consistently learned before generation; (ii) verification is more robust to continual learning than generation; and (iii) factual updates can leave models in a "multi-verse" state, simultaneously verifying both old and new answers as correct. Natural experiments on frontier models reproduce these dynamics at scale and reveal residual verification biases on well-covered facts. 

---
# On the Subgaussianity of Quantized Linear Maps: An AI-Assisted Note 

**Authors**: Guangyi Zou, Roman Vershynin  

**Link**: [PDF](https://arxiv.org/pdf/2605.27563)  

**Abstract**: This short note presents a dimension-independent subgaussian concentration bound for Gaussian vectors under coordinate-wise nonlinear mappings. Discovered by Gemini 3.5 Flash, this result applies to any bounded function under a well-conditioned covariance. We apply this tool to answer a question of Simone Bombari on sign-quantized linear maps $Y = \text{sgn}(Wx)$. 

---
# Clinical Validation of the Melanoscope AI Mobile Dermoscopy Clinical Decision Support System 

**Authors**: Elena Sergeevna Kozachok, Sergey Sergeevich Seregin  

**Link**: [PDF](https://arxiv.org/pdf/2605.27561)  

**Abstract**: Introduction. Early detection of malignant skin lesions is critical for prognosis, yet dermatologist shortages in Russian regions limit screening coverage. Mobile dermoscopy clinical decision support systems (CDSS) offer a promising approach, with model interpretability and standardised patient routing remaining key barriers to adoption.
Aim. To develop a quantitative interpretability assessment method for cascade deep learning models and a three-zone patient routing algorithm, and to conduct a preliminary single-centre prospective clinical validation of the Melanoscope AI CDSS in Russian outpatient practice.
Material and methods. Two-stage cascade classification of dermoscopic images; attention map visualisation (attention rollout for ViT and Swin; Grad-CAM for ConvNeXt and EfficientNetV2); quantitative IoU-based agreement assessment between activation maps and expert annotations; prospective single-centre validation across four "Melanoma Day" sessions (Orel, Russia, June 2025 - April 2026).
Results. On 176 patients: agreement with expert assessment 88.6%; no false negatives among 5 malignant lesions (95% CI: 47.8-100.0%); specificity 88.3%. Three melanomas and two basal cell carcinomas were histologically confirmed; six dysplastic naevi placed under follow-up. Mean IoU (n=180): ViT - 0.69; Swin - 0.64; ConvNeXt - 0.53; EfficientNetV2 - 0.51. Routing thresholds: P<0.15 / 0.15-0.50 / >=0.50.
Conclusion. No false negatives were observed; specificity was 88.3%, supporting screening use. The integrated cascade classification, attention map visualisation with IoU assessment, and three-zone routing provide reproducible, interpretable clinical decision support adaptable to varying resource levels. 

---
# Detection Without Correction: A Two-Parameter Decomposition of Multi-Stage LLM Pipelines 

**Authors**: Prashanti Nilayam, Kiran Ramanna, Prashil Tumbade  

**Link**: [PDF](https://arxiv.org/pdf/2605.27559)  

**Abstract**: Multi-stage LLM pipelines that perform multi-agent debate, intrinsic self-correction, or retrieval-augmented verification exhibit puzzling aggregate behaviors: accuracy plateaus and reversals across rounds, non-replication of debate gains on contemporary frontier models, intrinsic self-correction degradation, and qualitative cross-provider divergence in debate dynamics. Downstream agent response can be operationalized as two coupled decisions: detection (whether to treat upstream content as authoritative) and conditional generation (what to produce if not). This decomposition yields four observable response regimes, of which detection-without-correction is the load-bearing failure mode. Across a nine-cell empirical grid spanning four model families, four benchmarks (GSM8K, MATH-500, GPQA-Diamond, AIME), and two methods (multi-agent debate, intrinsic self-correction), we find that the conditional miscorrection rate is consistently dominant (53-94% across cohorts) while detection rate varies contextually by more than an order of magnitude. The framework unifies the four phenomena above as signatures of a common mechanism and characterizes detection threshold as a stable model/protocol-level regularity that persists across methods at matched benchmark difficulty. 

---
# Grounded Cache Routing for Retrieval-Augmented Generation: When Is It Safe to Reuse an Answer? 

**Authors**: Syed Huma Shah  

**Link**: [PDF](https://arxiv.org/pdf/2605.27494)  

**Abstract**: Modern retrieval-augmented generation(RAG) deployments increasingly rely on caching to reduce token cost and time-to-first-token(TTFT). Prefix-level KV reuse is now standard in serving stacks such as vLLM, and chunk-level and position-independent reuse have been pushed further by recent systems(RAGCache, TurboRAG, CacheBlend, EPIC, ContextPilot, PCR, LMCache). Output-level semantic answer caches, by contrast, remain fragile: similar prompts can map to different correct answers, retrieved evidence drifts as the corpus is updated, and adversarial collision attacks have been shown to hijack cached responses. We argue that the right framing for cached answer reuse is not how to reuse faster but when reuse is safe. We propose GroundedCache, an evidence-validated cache router that admits a cached answer only when 4 cheap gates simultaneously hold: query similarity, retrieved-evidence overlap, source-version validity, and lexical (or judge-based) support of the cached answer by the freshly retrieved evidence. We build a six-regime workload that stress-tests cache safety rather than only hit rate, and introduce an operator-facing metric, the unsafe-served rate (USR), fraction of all queries that received a wrong cached answer. Across 2 datasets and 12,000 real-LLM generations(Qwen2.5-7B-Instruct on vLLM with Automatic Prefix Caching), GroundedCache drives USR to 0.0% on every HotpotQA regime(vs. 15-35% under naive caching) and to 1.5% on mtRAG document drift(vs. 51.5%), a 34x reduction on the design-point adversarial regime and 3-10x reductions across the other mtRAG regimes, while end-to-end p50 latency stays within 1.04-1.07x of a no-cache RAG baseline. A per-gate ablation isolates the lexical support gate as the load-bearing safety mechanism on both datasets, with the remaining gates providing defense-in-depth at near-zero cost. We release the implementation, workload, and evaluation harness. 

---
# Benchmarks are Not Enough: RAMP for Runtime Assessing of Agentic Models in Production Systems 

**Authors**: Yipeng Ouyang, Xin Huang, Bingjie Liu, Zhongchun Zheng, Yuhao Gu, Xianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27492)  

**Abstract**: LLM agents are rapidly evolving from coding assistants into autonomous software engineering systems. However, existing evaluation methodologies remain largely centered on static, isolated, and short-horizon benchmarks that fail to capture the dynamic complexity of real-world production workflows. As a result, benchmark performance may poorly reflect practical capability under realistic runtime environments involving long execution chains, tool interactions, dependency management, and iterative feedback loops. We thus present RAMP, a production-grounded infrastructure for assessing long-horizon software engineering agents. Built upon the YatCC integrated platform, RAMP provides a unified runtime assessment architecture through standardized orchestration and execution interfaces. RAMP introduces realistic compiler-construction workloads with serial dependencies and complex toolchain interactions, together with a staged recovery mechanism for analyzing execution behavior under partial workflow failure. The framework further incorporates utility-oriented multi-dimensional metrics that jointly evaluate outcome quality and process efficiency. We conduct runtime assessments across 15 mainstream models and observe substantial capability degradation that remains largely invisible to conventional isolated benchmarks. Task completion rates progressively collapse across serial workflows, dropping from 100% in the initial stage to only 20% in the final stage, while none of the evaluated models successfully completes the entire pipeline. Runtime analysis reveals systematic failure propagation and significant resource inefficiencies, with computational costs differing by up to three orders of magnitude among comparable models. These findings suggest RAMP advances agentic model evaluation toward continuous, runtime-observable, and production-grounded assessment. 

---
# HARP: Measuring Harm Amplification in Multi-Agent LLM Systems 

**Authors**: Md Hafizur Rahman, Zafaryab Haider, Tanzim Mahfuz, Prabuddha Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2605.27489)  

**Abstract**: Multi-agent LLM systems decompose workflows across agents, tools, shared context, memory, and decision gates. This modularity improves interpretability, but creates a propagation risk: a bounded perturbation to one component can be reused by other agents and amplified into system-level harm. We introduce HARP (Harm Amplification through Role Perturbation), a trace-first methodology for studying local-to-global harm amplification in multi-agent LLM systems. HARP compares paired clean and perturbed executions and records specialist outputs, tool calls, memory reads/writes, guard events, oracle logs, latency, token cost, and decisions. We define local harm as deviation from targeted agents or corrupted channels, global harm as deviation over the full trace, and harm amplification as (H_global/H_local). This complements attack success rate with a measure of how strongly orchestration spreads harm beyond the attack point. We instantiate HARP in a finance-oriented seven-agent system with a deterministic decision gate and configurable attack harness for specialist compromise, collusion, shared-context corruption, and temporal or memory-persistent attacks. Across five defenses, prompt-only defenses preserve benign utility but leave high success and stealth; pre-tool and step-level guards reduce some failures with utility or latency costs; and IntegrityGuard, a trace-consistency defense, achieves the lowest attack success and global harm but introduces utility/cost trade-offs. Results show that single-specialist compromise produces the strongest amplification, shared-context corruption yields the highest attack success, and temporal persistence produces the largest malicious impact. HARP argues that secure multi-agent evaluation must measure not only bypass, but propagation. 

---
# Grimlock: Guarding High-Agency Systems with eBPF and Attested Channels 

**Authors**: Qiancheng Wu, Wenhui Zhang, Gan Fang, Sheng Mao, Biao Gao, David Levitsky, Shawna Murphy Butterworth, Rob Cameron  

**Link**: [PDF](https://arxiv.org/pdf/2605.27488)  

**Abstract**: Agentic systems increasingly run user-authored orchestration code that invokes tools, spawns subtasks, and delegates work across machines and clouds. Although this high agency is productive, it creates a security problem: identity, authorization, provenance, and delegation are often pushed into application code, where they become difficult to enforce consistently and difficult to audit.
We present \emph{Grimlock}, an \emph{Agent Guard} that restores separation of concerns by moving trust enforcement into the sandbox substrate while leaving agent code unchanged. Grimlock uses \emph{eBPF-enforced traffic interception} to ensure that sandbox communication passes through a guard, and combines it with \emph{post-handshake attestation} bound to standard TLS~1.3 channel bindings. After a channel is established, the guard authorizes communication and mints short-lived, channel-bound \emph{scope tokens} that capture least-privilege delegation. At the receiving side, the destination guard re-validates identity, scope, and channel binding, terminates TLS, and releases plaintext to the destination sandbox only after policy checks succeed. kTLS provides an efficient dataplane for protected communication.
As a result, Grimlock offers a path toward transparent, auditable, and scope-bound agent-to-agent communication across heterogeneous multi-cloud environments, using commodity Linux primitives and without requiring changes to user-layer orchestration code. 

---
# Diffusion-Based Ukrainian Handwritten Text Generation with Cross-Domain Style Transfer 

**Authors**: Andrii Ahitoliev, Pavlo Berezin  

**Link**: [PDF](https://arxiv.org/pdf/2605.27487)  

**Abstract**: Handwritten text generation (HTG) conditioned on writer style has been widely studied for Latin scripts, but remains underexplored for low-resource and non-Latin writing systems, leaving open how well existing models generalise beyond the Latin domain. Cyrillic, particularly Ukrainian, lacks both large-scale writer-labeled datasets and empirical evidence of such generalisation. To address this gap, we construct a Ukrainian handwritten word dataset of 126,177 images from 308 writers using connected-component segmentation, quality filtering, and targeted oversampling of underrepresented Ukrainian characters. We retrain DiffusionPen, a MobileNetV2 triplet-loss style encoder with a CANINE-conditioned latent diffusion U-Net, on this dataset without architectural modification, testing direct transfer from Latin to Cyrillic. We evaluate cross-domain style transfer in three settings: cross-lingual transfer from IAM English samples, zero-shot transfer to an early 20th-century Ukrainian manuscript, and few-shot imitation of contemporary writers. The model produces legible, style-consistent word images, indicating that few-shot latent diffusion models generalize beyond the Latin-script domain. We release the dataset, trained models, and evaluation protocol as a reproducible benchmark for writer-aware Cyrillic HTG, providing a foundation for extending stylized HTG to other underrepresented writing systems. 

---
# Debate Helps Weak Judges Reward Stronger Models 

**Authors**: Ethan Elasky, Frank Nakasako, Naman Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2605.27483)  

**Abstract**: Despite theoretical promise, debate as a scalable oversight protocol has produced mixed empirical results: gains in some settings, and null effects in others, especially when the judge does not have information hidden from it. We study proposer-critic debate in a stronger-debater/weaker-judge setting on programmatically verifiable code and logic tasks. Debate helps the judge over a consultancy baseline when the critic provides a usable advantage: the critic's classification ability must exceed the judge's, and the judge must treat critic speeches as claims to verify rather than testimony to summarize.
On the three of five pairings where the condition holds, proposer-critic debate's gains are statistically significant over consultancy, and these pairings are the most capable model pairings. On the two non-responder pairings in our set, debate produces null effects, and judge verification rates drop by tens of percentage points once a critic enters the transcript. In these cases the critic's binary-classification ability and the judge's are within noise of each other, and the critic's disagreement is parsed as testimony rather than a claim to check.
Ablating rebuttal rounds from debate produces no measurable change in judge performance: a single independent critique recovers the bulk of debate's benefit at lower inference cost. These findings suggest a cheaper primitive for training-free scalable oversight in verifiable domains (answer, critique, judge) and a pre-deployment audit (does the critic beat the judge, and will the judge verify it?) that predicts when debate will help. 

---
# Energy-Structured Low-Rank Adaptation for Continual Learning 

**Authors**: Longhua Li, Lei Qi, Qi Tian, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2605.27482)  

**Abstract**: While orthogonal subspace methods try to mitigate task interference in Continual Learning (CL), they often suffer from energy diffusion across the basis, hindering knowledge compaction and exhausting capacity for future tasks. We observe that output feature drift induced by parameter updates is inherently low-rank, and theoretically prove that preserving parameters along the principal directions of this drift minimizes the output reconstruction error. Motivated by this, we propose \textbf{E}nergy-Concentrated and \textbf{E}nergy-Ordered \textbf{Lo}w-\textbf{R}ank \textbf{A}daptation (E$^2$-LoRA). By explicitly ordering and concentrating knowledge into leading ranks, E$^2$-LoRA frees capacity for subsequent tasks. Furthermore, we design a dynamic rank allocation strategy to balance stability and plasticity by jointly optimizing energy retention and model plasticity. Extensive experiments across multiple benchmarks demonstrate that E$^2$-LoRA achieves state-of-the-art performance. 

---
# BIRDS: Characterizing and Understanding Biodiversity Impact of Large Language Model Serving 

**Authors**: Tianyao Shi, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.27480)  

**Abstract**: Large language model (LLM) serving creates environmental impacts beyond carbon and water, including ecosystem damage through biodiversity-related pathways. We present BIRDS, a framework for Biodiversity Impact of Request-Driven LLM Serving. BIRDS defines request-level functional units, quantifies operational and embodied biodiversity impact, and introduces Quality-Normalized Biodiversity Impact (QNBI) to jointly analyze ecological impact and response quality. Across diverse workloads, models, GPUs, and regions, \SYSTEM{} reveals that biodiversity impact accumulates at scale and exposes actionable quality-aware serving tradeoffs. 

---
# Resource-Constrained Affect Modelling via Variance Regularisation Pruning 

**Authors**: Kosmas Pinitas, Konstantinos Katsifis  

**Link**: [PDF](https://arxiv.org/pdf/2605.27479)  

**Abstract**: Affective computing systems are increasingly embedded in pervasive and interactive environments, such as adaptive games, assistive technologies, and resource-constrained platforms, where computational efficiency must be balanced with reliability across diverse users. Model pruning offers an effective way to reduce computational demands, yet existing approaches typically optimise for sparsity alone, without accounting for how parameter removal impacts robustness across individuals. In this work, we introduce Variance-Regularised Pruning (VR), a pruning framework that explicitly incorporates cross-participant stability into the sparsification process. Rather than relying solely on average prediction error, VR evaluates each connection based on its joint contribution to both prediction accuracy and variability across users, prioritising parameters that remain reliable under distributional differences. We evaluate the proposed approach on the AGAIN dataset, which includes arousal annotations collected across nine affect-eliciting game environments. Experimental results demonstrate that VR maintains competitive Concordance Correlation Coefficient (CCC) performance even at 80\% sparsity without additional fine-tuning, highlighting its suitability for deployment in real-world, resource-limited affect-aware systems. Overall, the proposed framework supports the development of compact, robust affective models that can operate reliably in real-world interactive environments. 

---
# Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective 

**Authors**: Hyunmin Cho, Woo Kyoung Han, Kyong Hwan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.27476)  

**Abstract**: We characterize the pre-softmax attention matrix $\mathbf{QK^\top}$ in transformers as an associative memory matrix encoding pairwise associations between input features. By decomposing this matrix into its symmetric and skew-symmetric parts, we interpret the symmetric component as governing the structure of the energy landscape, and the skew-symmetric component as driving circulation on that landscape. Leveraging the energy formulation induced by the symmetric component, we derive Hopfield-style stability measures that quantify the stability of retrieved features. We observe meaningful correlations between Hopfield-style stability measures and the fidelity-diversity trade-offs in generation. Finally, we propose a controllable knob to modulate this trade-off by modifying the circulation of the underlying dynamics. Code is available at our GitHub (this https URL). 

---
# HEAL: Resilient and Self-* Hub-based Learning 

**Authors**: Mohamed Amine Legheraba, Stefan Galkiewicz, Maria Gradinariu Potop-Butucaru, Sébastien Tixeuil  

**Link**: [PDF](https://arxiv.org/pdf/2605.27475)  

**Abstract**: Decentralized learning enhances privacy, scalability, and fault tolerance by distributing data and computation across nodes. A popular approach is Federated learning, which relies on a central aggregator, yet faces challenges such as server vulnerabilities, scalability issues, privacy risks and most importantly, the single point of failure. Alternatively Gossip Learning and Epidemic Learning offer fully decentralization through peer-to-peer exchanges of model updates, ensuring robustness and privacy, at the price of slower model convergence. In this work, we introduce a novel decentralized learning framework called HEAL. HEAL is the first cross-layer decentralized learning framework that exploits an optimized self-organizing and self-healing underlying P2P overlay combining the strengths of Federated Learning, Gossip and Epidemic Learning. Leveraging the recently proposed Elevator algorithm, HEAL promotes dynamically chosen nodes to act as aggregators. Through simulations, we demonstrate that HEAL has similar performances to that of Federated Learning in crash-free settings, while being fully decentralized and fault-tolerant. In crash and churn prone environments HEAL outperforms Gossip and Epidemic Learning. 

---
# AssertLLM2: A Comprehensive LLM Benchmark for Assertion Generation from Design Specifications 

**Authors**: Yuchao Wu, Wenji Fang, Jing Wang, Wenkai Li, Ziyan Guo, Zhiyao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.27472)  

**Abstract**: Assertion-based verification (ABV) is a cornerstone of modern hardware design, yet manually translating design intent into formal SystemVerilog Assertions (SVAs) remains labor-intensive and error-prone. While Large Language Models (LLMs) show promise for automating this process, existing benchmarks remain limited by unrealistic task formulations, weak specification inputs, and oversimplified evaluation. To address these limitations, we introduce AssertLLM2, an open-source benchmark for realistic assertion generation in hardware verification. AssertLLM2 contains 83 real-world designs across 13 functional categories. For each design, the benchmark provides a structured design specification, a verified dependency-complete golden RTL, and systematically mutated buggy RTL variants. These support two practical settings: bug-prevention, where assertions are generated from specifications to guard against design errors, and bug-hunting, where assertions are generated to expose discrepancies between intended behavior and faulty implementations. To the best of our knowledge, AssertLLM2 is the first benchmark to explicitly use buggy RTL as input to evaluate bug-detection capability. AssertLLM2 further adopts a more rigorous evaluation framework spanning syntactic validity, formal provability, coverage, and mutation-based bug detection. Our benchmark enables a more realistic and extensive assessment of assertion generation and establishes rigorous baselines for state-of-the-art LLMs in practical hardware verification. 

---
# Detect by Yourself: Self-Designing Agentic Workflows for Few-Shot Graph Anomaly Detection 

**Authors**: Tairan Huang, Qiang Chen, Yili Wang, Yueyue Ma, Changlong He, Xiu Su, Yi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.27470)  

**Abstract**: Graph anomaly detection aims to identify anomaly nodes in attributed graphs and plays an important role in real-world applications. However, existing graph anomaly detection methods still face two key challenges: 1) fixed pipelines, which restrict their adaptability across different graph tasks under limited supervision; 2) weak evidence, which prevents them from explicitly incorporating contextual and structural anomaly signals into the detection process. In this paper, we propose a novel framework, self-designing agentic workflows for few-shot graph anomaly detection (SignGAD). Specifically, we propose a novel paradigm that reformulates graph anomaly detection task from training a fixed anomaly detector to designing task-conditioned detection workflows. By constructing detection workflows, SignGAD selects suitable graph encodings and detector designs to exploit task-specific anomaly evidence. Meanwhile, we introduce a guarded final refit strategy to refine the selected workflow by calibrating refit acceptance, enhancing reliability under limited supervision. Extensive experiments conducted on several real-world datasets demonstrate that SignGAD achieves strong performance against state-of-the-art methods, highlighting its effectiveness on graph anomaly detection tasks. 

---
# Architecture-driven Shift: towards a lightweight selector for capturing the trends of logit shift 

**Authors**: Zhong Ye, Yu Hu, Ruilin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27469)  

**Abstract**: Continual Learning (CL) is a practical paradigm to utilize power of deep pre-trained neural networks, but which pre-trained model has a better ability to balance ``Plasticity-Stability", deserving to be chosen? The logit shift serves as a natural proxy because it represents the logit shift in CL scenarios. However, obtaining the logit shift requires huge computational cost, which hinders large-scale model selection. Existing theoretical analyses fail to offer an efficient alternative because of the assumption of uniform hidden layer widths, which ignores the structural heterogeneity (variable width and depth) of real-world architectures. This raises a critical question: what theoretically relationship can be identified between heterogeneous architecture and logit shift on prior tasks (that the model has been trained on)? To answer the question, we decouple logit shift into architecture dependency and data dependency to establish our framework, which reveals that the combination of two dependency, defined as Architecture-driven Shift (ADS), that can capture the logit shift tendency well computable with few data samples. Specifically, for a well-optimized model on prior tasks, higher ADS is associated with a larger logit shift after training on the current task, which derived based on three mechanistic components: (1) spectral norm scaling of weight matrix gradients with layer width, (2) the optimization path length of the new task, and (3) the asymptotic task conflict in wide networks. Extensive empirical results across more than 175 diverse architectures demonstrate a strong monotonic correlation (the weakest Spearman's $r_s=0.731$) between ADS and logit shift. Practically, we demonstrate that ADS can serve as a lightweight proxy of the expected calibration error, which is a widely used metric for reliable CL model selection, on three datasets across six scenarios. 

---
# Comparative Analysis of Liquid Neural Networks and LSTM for Sequential Pattern Recognition: Robustness, Efficiency, and Clinical Utility 

**Authors**: Ye Kyaw Thu, Thazin Myint Oo, Thepchai Supnithi  

**Link**: [PDF](https://arxiv.org/pdf/2605.27467)  

**Abstract**: Traditional Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) units operate on discrete time steps, often failing to capture the fluid temporal dynamics of real-world physical processes. Liquid Neural Networks (LNNs), specifically Closed-form Continuous-time (CfC) networks, address this by modeling the hidden state evolution as a continuous differential equation. In this paper, we conduct a comprehensive benchmarking study across four distinct sequential modalities: neuromorphic event-based data (N-MNIST), stroke-based drawing (QuickDraw), visual handwriting (IAM), and physiological time-series (PhysioNet Sepsis-3). Furthermore, we perform a rigorous stress test using temporal dropout to evaluate model robustness against missing data. Our findings reveal that LNNs consistently provide superior parameter efficiency and significantly higher robustness in natively temporal domains and clinical environments where data sparsity is prevalent. This extended preprint provides additional background on related datasets and the LNN theoretical lineage, supplemented with a detailed appendix documenting our full implementation and experimental settings. 

---
# AgensFlow: A Coordination-Policy Substrate for Multi-Agent Systems 

**Authors**: Nicole Koenigstein  

**Link**: [PDF](https://arxiv.org/pdf/2605.27466)  

**Abstract**: Multi-agent systems built on large language models (LLMs) require many coordination choices that are difficult to fix a priori: which skill protocol to invoke, which agent role should perform a subtask, which model to bind to each role, how roles should interact, when to use retrieval or verification, and when to omit a step entirely. These choices interact with task regime and operational constraints, so static pipelines and one-off model comparisons provide only a limited view of the design space. This paper introduces AgensFlow, an open-source framework that treats multi-agent coordination as an online policy-learning problem under partial observability. The framework makes coordination decisions observable and learnable from repeated trajectories, rather than treating skill, role, model, topology, and evaluation choices as fixed pipeline design.
AgensFlow is evaluated on two corpora: distributed-systems incident tasks and security-advisory tasks. The evaluation shows three main results: learned routing reaches a higher-quality operating point than a fixed pipeline baseline on coordination-heavy classes; skip:X isolates topology compression as a meaningful part of the substrate; and warm-started policy graphs can reduce exploration cost while preserving plateau quality. Overall, the results support that learned, auditable routing can improve coordination-heavy multi-agent workflows over static wiring. 

---
# AdaMerge: Salience-Aware Adaptive Token Merging for Training-Free Acceleration of Vision Transformers 

**Authors**: Semi Lee, Hyejin Go, Hyesong Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.27465)  

**Abstract**: The quadratic cost of self-attention in Vision Transformers (ViTs) constitutes a fundamental bottleneck for practical deployment, motivating a vibrant line of research on token reduction. Among existing approaches, token merging (ToMe) has emerged as an elegant training-free solution; yet its design rests on an unspoken premise of token equality, which contravenes the well-documented non-uniformity of self-attention and leads to information loss in high-salience tokens under aggressive compression. We address this limitation with AdaMerge, a token-merging framework based on two complementary mechanisms. First, salience-weighted similarity leverages column-wise feature-affinity centrality as a token-importance proxy and incorporates the resulting salience scores into the bipartite matching score, ensuring that pivotal tokens contribute more strongly to the merged representation. Second, adaptive merging intensity uses pre-computed layer-wise similarity statistics to dynamically modulate the per-layer reduction count in accordance with input-specific redundancy. On ImageNet-1k with ViT-B/16, AdaMerge consistently outperforms ToMe, PiToMe, and DSM across all FLOPs-matched regimes. The accuracy gap widens monotonically with compression: at the 13.4G FLOPs operating point, AdaMerge sustains a Top-1 degradation of only -1.06%, compared to -1.45% for PiToMe and -4.62% for DSM. To our knowledge, AdaMerge is the first to combine salience-weighted similarity and adaptive per-layer reduction into a single training-free token merging framework, advancing the accuracy-FLOPs Pareto frontier of ViT acceleration. 

---
# Beyond Motion Primitives: Behavioral Activity Recognition from Head-Mounted IMU 

**Authors**: Chung-Ta Huang, Leopold Das, Jeffrey Zhou, Faizaan Siddique, Julia Seungjoo Baek, Serena Liu, Andrew Rusli, Todd Y. Zhou, Freddy Yu, Sinclair Hansen, Ziling Hu, Arnav Sharma, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27464)  

**Abstract**: AR smart glasses need continuous behavioral context to offer proactive assistance, yet their most practical always-on sensor, the head-mounted Inertial Measurement Unit (IMU), detects only motion primitives such as walking or standing. We push beyond motion primitives to behavioral-level recognition, defining five categories that balance AR application need with sensor observability. To this end, we construct a 160K-sample Ego4D dataset with a four-tier quality assurance framework spanning 8 activity scenarios, and propose HiT-HAR, a 703K-parameter hierarchical model that outperforms prior head-mounted IMU models on five-class action and eight-class scenario recognition. We further map the observability frontier of head-mounted IMU through per-class separability analysis, identifying which behavioral categories are reliably observable (Locomotion), which benefit from temporal context (Object Transfer, Task Operation), and where scenario-dependent signal overlap poses remaining challenges. Our results indicate that architectural choices exploiting temporal context and scenario structure outperform simply scaling model size. The code and dataset are publicly available at this https URL. 

---
# When prompt perturbations break your A/B test: A valid statistical test for generative surveying 

**Authors**: Hayden Helm, Carey Priebe  

**Link**: [PDF](https://arxiv.org/pdf/2605.27463)  

**Abstract**: Generative surveying -- where collections of LLM-based personas provide feedback on messages -- has emerged as a cheap and scalable alternative to traditional market research. However, LLMs are sensitive to small variations in prompt design and conclusions drawn from generative surveys may depend on arbitrary phrasing choices. Controlling for this sensitivity requires including semantically equivalent perturbations in the analysis. In this paper, we show that standard hypothesis tests, including the sign test and Wilcoxon signed-rank test, are invalid under a statistical model for generative surveying that includes realistic perturbation structure. We propose a permutation test that is valid under this model and formally characterize the conditions under which standard tests fail. Applying our framework to a simple generative surveying problem, we estimate relevant parameters, characterize the power of the permutation test under realistic conditions, and provide practical guidance on budget allocation across personas, perturbations, and replicates. Finally, we show that both the magnitude and direction of the estimated effect are sensitive to the choice of model, even within the same model family. 

---
# Generic Interpretation Approach for Transformer Models Incorporating Heterogenous Attention Structures 

**Authors**: Yongjin Cui, Xiaohui Fan, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.27458)  

**Abstract**: Transformer has significantly propelled the development of artificial intelligence, and certainly the development of agents as well. We categorize attention structures of Transformer into two types based on the source of the input information: homogenous and heterogenous attention structures. Heterogenous attention structures, with co-attention as a typical example, process information from different sources. Heterogenous attention structure is the foundation for Transformer models to achieve more complex functions and integrate more modal information. Whether for research purposes or policy requirements, the interpretation of Transformer models with heterogenous attention structures is an important task. The fusion of information from different sources brings new challenges. Our work mainly includes two parts: method and experimentation. In terms of method, we propose an interpretation method for Transformer models with heterogenous attention structures. In terms of experimentation, based on our experimental analysis paradigm, we interpret the operating mechanisms of representative models, conduct semantic interpretation and logical interpretation. 

---
# Checking Fact with Better Retrieval: Dynamic Contrastive Learning for Evidence Retrieval 

**Authors**: Zhongtian Hua, Yi Luo, Meijia Yu, Yingjie Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.27449)  

**Abstract**: In the field of multimodal fact checking, the accuracy of retrieving evidence from different modalities has a significant impact on the downstream claim verification process. Existing general multimodal retrieval methods are often constructed based on semantics, resulting in the retrieved evidence being similar but not relevant to the claim. This paper proposes a \textbf{D}ynamic \textbf{A}daptive \textbf{C}ontrastive \textbf{L}earning method for evidence \textbf{R}etrieval called DACLR to address these issues. DACLR first uses a Multimodal Large Language Model (MLLM) to uniformly convert multimodal evidence and claims into text modalities, and extracts the features of these information at event level. Then, it conducts evidence retrieval through a two-stage retrieval method of recall-rerank. DACLR enhances the model's event perception ability of the retrieval stage by optimizing the contrastive loss and mining hard negative samples. Specifically, DACLR designs three loss functions at two levels (semantic and event) based on the InfoNCE this http URL to these, three sets of hard negative sample candidates are set up. The model dynamically adjusts the ratio based on the accuracy supervision signal of intra-batch samples, allowing the model to learn the correlation between claims and positive samples at the event level without forgetting the semantic retrieval ability. Extensive comparison and ablation experiments demonstrates the effectiveness of DACLR and its internal optimization methods. Further research also prove the advantages of DACLR in the field of multimodal evidence retrieval. 

---
# RAGe: A Retrieval-Augmented Generation Evaluation Framework 

**Authors**: Larissa Guder, João Pedro de Moura, Arthur Accorsi, Gustavo Losch do Amaral, Maurício Cecílio Magnaguagno, Felipe Meneguzzi, Marcio Sorraglia Pinho, Dalvan Griebler  

**Link**: [PDF](https://arxiv.org/pdf/2605.27445)  

**Abstract**: Deploying Large Language Model (LLM) applications, particularly those relying on Retrieval-Augmented Generation (RAG), remains challenging due to high computational demands, outdated knowledge bases, and the need to manually select optimal pipeline components. In this work, we propose a modular framework for benchmarking and guiding the efficient development of RAG applications by focusing on resource telemetry and component recommendation, suggesting the best components for a domain-specific dataset. Our approach leverages core techniques in LLM applications, including document chunking, vector databases, embedding models, and retrievers, to evaluate trade-offs among accuracy, efficiency, and scalability. By directly correlating retrieval and generation quality with underlying hardware constraints, RAGe supports researchers to identify the most effective, domain-specific RAG setups for their specific operational needs, facilitating rapid prototyping even on consumer-grade hardware. 

---
# A Systematic Evaluation of Retrieval-Augmented Generation and Language Models for Space Operations 

**Authors**: Ruben Belo, Marta Guimarães, Cláudia Soares  

**Link**: [PDF](https://arxiv.org/pdf/2605.27444)  

**Abstract**: The rapid expansion of space activities has led to an unprecedented accumulation of technical documentation, operational guidelines, and scientific literature, creating challenges for timely decision-making in space operations. Effective management in space operations requires tools capable of efficiently processing vast and heterogeneous information sources. This paper systematically evaluates the performance of Retrieval Augmented Generation (RAG) pipelines, combining Large Language Models (LLMs) with information retrieval techniques for extracting and synthesizing actionable knowledge from domain-specific documents. We compare various retrieval strategies, embedding models, and LLM answers to assess their impact on information accuracy, relevance, and reliability. Our results demonstrate that RAG pipelines can significantly enhance knowledge access, reduce uncertainty, and support decision-making in complex space operations. 

---
# Paraphrase Brittleness in Production Retrieval-Augmented Commercial Recommendation: Reproducibility Below the Rerun-Stability Baseline 

**Authors**: Will Jack, Noah Lehman, Keller Maloney, Sarah Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.27440)  

**Abstract**: Small changes to how a buyer phrases a question -- "best CRM" vs "top CRM" vs "best CRM for a SaaS startup" -- produce substantially different brand recommendations from AI assistants. Across ~6,000 paraphrase runs and ~6,000 same-prompt rerun controls on OpenAI and Anthropic models, the recommendation-set similarity (Jaccard) between two paraphrases of the same underlying buying intent is 0.288 for cosmetic rewordings (clustered 95% CI [0.215, 0.361]) and 0.135 for constraint-adding rewordings ([0.098, 0.175], pooling region/language and specificity-ladder axes) -- both far below the 0.50-0.61 same-prompt rerun baseline. The prompt string, not the underlying buyer intent, is the dominant input to which brands surface. Increasing reasoning effort does not narrow the gap (bounded by +/-0.05). This is a direct challenge to an increasingly popular AEO/GEO practice. Tracking a brand's "AI visibility" by counting brand mentions over a fixed set of prompts produces a metric whose dominant source of variance is which paraphrase the tracker happens to issue, not the model's behavior toward the brand: the same buyer intent in two natural paraphrases produces recommendation sets that overlap 14-29% in Jaccard versus 50-61% for same-prompt reruns. Sampling more paraphrases per intent reduces the artifact in principle, and efficient multi-prompt evaluation methods exist in the academic literature, but the natural buyer-phrasing space is much larger than the benchmark-scale prompt sets those methods have been validated on, and far beyond what any commercial tracker issues per brand-intent combination. Prompt-by-prompt mention tracking is therefore structurally unstable as a unit of measurement; meaningful improvement likely requires a different unit rather than a larger prompt set. 

---
# Prominence-Stratified Failure Modes in Retrieval-Augmented Commercial Recommendation: A 37,000-Run Audit 

**Authors**: Will Jack, Noah Lehman, Keller Maloney, Sarah Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.27439)  

**Abstract**: AI assistants like ChatGPT and Claude are recommendation engines, not search engines: they answer commercial queries by directly nominating brands rather than returning a list of links. Marketing to AI is therefore a broader problem than "show up in search" -- positioning, content, and product fit matter as much as discoverability. We audit ~37,000 production runs across four model configurations and 215 commercially-framed prompts spanning 19 sectors, evaluated against a 533-brand reference catalog stratified into five prominence tiers (L1 category leaders to L5 regional players) sourced from external authority lists. The ladder proxies a brand's awareness footprint within its sector, not revenue or market share. The failure mode differs sharply by tier. L1 brands appear in nearly every relevant retrieval but win only 25-41% of the recommendation slots they reach -- the leverage is differentiation, not visibility. L2 challengers carry the highest conversion rates of any tier (37-52%) but lose to persona-mediated substitution on the Anthropic models. L3 mid-market brands are the inflection level: aggregate coverage drops to 88%, conversion to 34-40%, and persona effects peak. L4 specialists and L5 regional players face catastrophic invisibility -- 48-52% never surface in any of the 37,000 runs. No uniform optimization recipe wins; the right marketing investment depends on where the brand sits on the prominence ladder. 

---
# MGRetrieval: Memory-Guided Reflective Retrieval for Long-Term Dialogue Agents 

**Authors**: Tan Wang, Yunwei Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.27437)  

**Abstract**: Large Language Models (LLMs) have made significant progress in dialogue, yet redundant memory contexts severely limit their effectiveness in long-term dialogue agents. External memory systems have been proposed to improve memory maintenance. However, these systems mainly rely on one-shot retrieval, which limits their ability to retrieve sufficient and relevant evidence. Although recent methods introduce reflection into retrieval, their retrieval paths are generated by the LLM from limited evidence, leading to unstable retrieval and additional latency overhead. %These limitations highlight the need for effective retrieval mechanisms. To address these limitations, we propose MGRetrieval, a retrieval strategy that grounds reflective retrieval in the semantic structure of historical memories. Specifically, MGRetrieval consists of two steps: (1) It references the structure of historical memories to construct a more precise retrieval path. (2) The LLM retains critical memories and determines whether accumulated memories are sufficient to stop further iterative retrieval. This allows the retrieval process to follow semantically meaningful paths. Through memory-guided retrieval and critical memory propagation, MGRetrieval gradually constructs concise and sufficient memory contexts. Extensive experiments on LoCoMo show that MGRetrieval outperforms the strongest baseline by 8.91\% in F1 and 11.11\% in BLEU-1 on average across Qwen2.5-14B and Qwen3-14B, while maintaining practical token and latency costs. The code can be found in this https URL. 

---
# RE-TRIANGLE: Does TRIANGLE Enable Multimodal Alignment Beyond Cosine Similarity in Retrieval? 

**Authors**: Arijit Ghosh, Aritra Bandyopadhyay, Chiranjeev Bindra, Jingfen Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.27436)  

**Abstract**: Multimodal alignment is critical for bridging the semantic gap in information retrieval. However, traditional pairwise strategies introduce a geometric blind spot: while they align anchor modalities (e.g., text) with others, they lack constraints to enforce mutual consistency between peripheral modalities (e.g., video and audio). The TRIANGLE framework addresses this by minimizing the area of modality triplets on a hypersphere to enforce holistic alignment. In this reproducibility study, we verify the robustness of this geometric objective for retrieval tasks. We confirm that TRIANGLE outperforms pairwise baselines in zero-shot settings, achieving Recall@1 gains of up to +8.7 points, though benefits are domain-dependent. However, we fail to reproduce the reported learning-from-scratch results. Analysis using a synthetic toy dataset attributes this to instability when jointly optimizing geometric alignment with Data-Text Matching (DTM) loss. Furthermore, we find that cosine regularization primarily stabilizes text-to-video retrieval, and fine-tuning with domain supervision amplifies geometric benefits but reduces cross-dataset generalization. Our findings support the efficacy of geometric alignment while highlighting critical optimization sensitivities. Code available at this https URL. 

---
# When NPUs Are Not Always Faster: A Stage-Level Analysis of Mobile LLM Inference 

**Authors**: Pu Li, Jiawen Qi, Qinyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.27435)  

**Abstract**: Deploying large language models (LLMs) on mobile devices increasingly relies on heterogeneous execution, yet no prior study has systematically characterized NPU effectiveness at the operator and pipeline level. We present the first stage-aware, multi-level benchmarking study of mobile LLM inference on a CPU-NPU heterogeneous SoC. We introduce an OPMASK-based controlled pipeline decomposition methodology that isolates communication, quantization, and computation overheads within the NPU execution path. Our results reveal a counter-intuitive stage-level performance reversal: CPUs outperform NPUs in the compute-intensive Prefill stage (up to 1.6x), while NPUs provide only limited acceleration in the memory-bound Decode stage (1.05-1.2x). We further show that scheduling overhead and cross-backend fallback reduce the practical benefits of NPU offloading. For the energy trend, increasing NPU offloading leads to higher energy consumption (up to 51%). Based on these findings, we derive design guidelines for NPU architects targeting on-device LLM inference. 

---
# Heterogeneous Multi-Agent Modeling for Measurement and Network Analysis of the Data Service Market 

**Authors**: Deyu Zhou, Yuwei Guo, Xudong Lu, Linhao Zhang, Wei Guo, Lizhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2605.27433)  

**Abstract**: With the increasing complexity of collaboration among various social entities and user demands, the factors affecting the stable development of the data service market are also growing. These factors include the widespread dissemination of information enhancing subjective consciousness, the continuous improvement in intelligence, and the complexification of structural relationships. To achieve effective governance and regulation of the data service market, it is crucial to conduct simulation experiments before making regulatory decisions. However, current research and analysis of the data service market primarily focus on data-level performance, proving inadequate when it comes to measurement and analysis of multiple heterogeneous entities and the integration of various social elements within the data service market. Based on this, this paper innovatively proposes a data service market measurement and network analysis method based on heterogeneous multi-agent modeling. By introducing the service ecosystem theory, we clarify the participants and external factors of the data service market and conduct utility measurements for three-level entities based on value creation. Furthermore, an analytical methodology is devised to precisely assess the influence of heterogeneous networks on utility. Finally, the paper verifies the effectiveness of the proposed method through the analysis of experimental results. 

---
# FD-RAG: Federated Dual-System Retrieval-Augmented Generation 

**Authors**: Tianhao Gao, Kai Yang, Yiyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.27432)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a paradigm for grounding large language models in external knowledge, yet most existing RAG systems assume centralized knowledge access and ample computation. These assumptions break down in edge environments, where knowledge is fragmented across devices, raw data cannot be shared, and repeated LLM calls are prohibitively expensive. We propose FD-RAG, a federated dual-system RAG framework that decouples lightweight memory access from on-demand LLM reasoning for decentralized deployment. Specifically, FD-RAG learns semantic-aware adaptive hypergraphs over local corpora and distills them into compact QA memories. At inference time, it answers well-covered queries via direct memory matching and invokes LLM-based reasoning only when necessary, while tracing retrieved memories to hypergraph-grounded evidence. To mitigate cross-device knowledge fragmentation, FD-RAG aggregates anonymized memories across devices without exposing raw documents. Experiments on QA benchmarks show that FD-RAG improves accuracy by up to 7.8\% while reducing latency by 8.4$\times$ compared with strong local and federated baselines. We also provide theoretical analysis establishing an $\mathcal{O}(1/\epsilon^{2})$ convergence rate for the proposed hypergraph learning, supporting its tractable deployment in edge settings. 

---
# Tackling Multimodal Learning Challenges with Mixture-of-Expert: A Survey 

**Authors**: Liangwei Nathan Zheng, Wei Emma Zhang, Olaf Maennel, Lin Yue, Weitong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.27431)  

**Abstract**: Mixture-of-Experts (MoE) presents a naturally compatible and scalable framework for multimodal learning, demonstrating strong adaptability across diverse modalities and tasks. Despite its growing success, a comprehensive and systematic review on the MoE metho addressing multimodal challenges remains lacking. Existing surveys tend to evaluate either multimodal learning or MoE independently from method taxonomy, overlooking the unique interplay between them. This survey fills that gap by answering a central question: \textit{How does MoE effectively resolve multimodal challenges?} We approach this from three key perspectives: (1) \textbf{MoE as an Efficient Multimodal Engine:} enabling scalable multimodal modeling by decoupling computational cost from parameter growth and mitigating modality redundancy through selective expert activation; (2) \textbf{MoE as a Multimodal Representation Learner:} integrating complementary multi-opinion expert knowledge to enrich alignment and interaction representations; and (3) \textbf{MoE as a Multimodal Adapter:} providing a modular and flexible mechanism to model imperfect data scenarios such as modality imbalance and missing modality. Through our extensive literature review, we identify critical research gaps, including interpretable routing, expert communication, modality integration, and lifelong multimodal learning. We position this survey as a foundation for future research toward interpretable and sustainable multimodal Mixture-of-Experts system. 

---
# Ocean4Rec: Offline LLM-Derived OCEAN Profiles for Request-Time VOD Reranking 

**Authors**: Wonkyun Kim, Sehyun Bae, Kwanki Ahn, Mungyu Bae, Saeun Choi, Soyeon You, Chandra Prabhakar, Sehyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.27429)  

**Abstract**: Industrial video-on-demand (VOD) recommenders need richer content understanding, but LLM-as-reranker designs repeat prompt construction, token generation, model invocation, output parsing, and fallback handling for each request. In high-volume latency-sensitive services, these request-time operations complicate throughput planning, tail-latency control, capacity isolation, and predictable operation. This paper presents Ocean4Rec, a reranking layer that uses an LLM only offline to materialize item OCEAN profiles from content metadata. Items are mapped into Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism scores, while user profiles are built by time-decayed aggregation of recently clicked and deep-linked items in the same five-dimensional space. At request time, Ocean4Rec joins precomputed item profiles, user profiles, base recommender scores, and catalog recency, then performs numeric reranking without an LLM call. On anonymized Samsung Smart TV VOD logs, same-candidate Top1000 temporal-holdout offline evaluations show that Ocean4Rec improves NDCG@20 over a stronger non-OCEAN Base+Recency ordering by 7.6% for an NCF generator and 61.5% for a LightGCN generator. HR@20 is inconclusive for NCF and improves by 67.3% for LightGCN, reflecting sparse exact-item replay labels and the strength of recency as an industrial baseline. The result should be read as offline replay evidence for a bounded auxiliary content-taste feature that preserves the deployability advantage of a request-time-LLM-free serving path. 

---
# Quantum Machine Learning-based 6G edge Network: Enabling Adaptive Communication and Model Aggregation 

**Authors**: Wenjing Xiao, Jiatai Yan, Chenglong Shi, Shixin Chen, Miaojiang Chen, Min Chen, Saif Al-Kuwari, Ahmed Farouk  

**Link**: [PDF](https://arxiv.org/pdf/2605.27417)  

**Abstract**: With the advent of sixth-generation (6G) mobile communication technology, vehicle-to-everything (V2X) communication faces unprecedented challenges in communication efficiency, system generalization capabilities, and model collaboration. Conventional machine learning struggles with high-dimensional state spaces, slow convergence, and poor generalization under heterogeneous V2X nodes, rapidly varying channels, and multimodal sensing data in V2X systems. To address these issues, we propose a quantum-enhanced framework for V2X communication and model aggregation that targets efficient, robust, and intelligent transportation in 6G, which includes four modules: the channel-adaptive semantic communication module, the multimodal fusion module, the model transfer module, and the federated aggregation module. Specifically, the channel-adaptive semantic communication module leverages quantum convolutional neural networks (CNN) and quantum distortion metrics to enable efficient transmission and strong generalization across diverse conditions. The multimodal fusion module exploits quantum attention and entanglement to compress features and associate semantics across heterogeneous data. The model transfer module employs quantum reinforcement learning to model decision-making and improve adaptability in dynamic environments. The federated aggregation module integrates quantum tensor decomposition with backpropagation-based corrections to provide privacy preservation with low overhead and to strengthen global model robustness. This work outlines a new paradigm for communication and model collaboration in future 6G intelligent transportation. 

---
# Can Quantum Federated Learning Withstand Circuit-Level Backdoors? 

**Authors**: Aakar Mathur, Mohammed Ruknuddin, Ashish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2605.27416)  

**Abstract**: Quantum Federated Learning (QFL) inherits the core vulnerability of federated optimization to malicious clients, while also introducing an attack surface from variational circuit training and measurement-driven gradients. This work proposes a novel CircUit-Level backdoor Threat (CULT) model that formalizes four stealthy attacks by exploiting quantum-aware mechanisms, including Grover, Pauli, Bit-flip, and Sign-flip. By enabling malicious clients on both in-training and post-training surfaces, these attacks can critically undermine the learning process. We establish a rigorous theoretical foundation to demonstrate attack stealthiness under standard smoothness assumptions. Experiments on the MNIST and CIFAR-10 datasets with non-IID splits and varying fractions of malicious clients show that even a single malicious client can induce severe accuracy degradation under FedAvg aggregation. While popular defenses, including Krum, Multi-Krum, FoolsGold, FLGuardian, and Mud-HoG, reduce degradation in many regimes, they fail to eliminate worst-case failure cases, where accuracy drops up to 50\%. The experimental analysis further reveals that under the CULT model, malicious updates effectively mask their presence by staying close to benign norms, thereby helping attackers evade detection. 

---
# Ligand-Conditioned Discrete Diffusion for Protein Sequence-Structure Co-Design 

**Authors**: Chen Wei, Fanding Xu, Minghao Sun, Zhiyuan Liu, Lin Wang, Tianrui Jia, Yihang Zhou, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27413)  

**Abstract**: Proteins perform their biological functions through three-dimensional structures encoded by amino acid sequences, and ligand-binding protein co-design requires models that generate sequence-structure compatible proteins under explicit ligand constraints. Although continuous diffusion and flow-based models support ligand-aware design in coordinate or latent spaces, existing discrete diffusion protein language models mainly operate over sequence or structure tokens without direct small-molecule conditioning. We introduce \textbf{ProtLiD$^2$}, a \textbf{Prot}ein \textbf{L}igand-conditioned \textbf{D}iscrete \textbf{D}iffusion model for protein sequence-structure co-design. ProtLiD$^2$ jointly generates amino-acid sequence and discrete structure tokens while incorporating ligand chemical and geometric information through geometry-aware cross-attention. Trained on over one million ligand-protein complexes, ProtLiD$^2$ extends masked discrete diffusion to ligand-aware functional protein design. We further propose maximum confidence-margin guided ReMask decoding, an inference-time self-correction strategy that retains confident predictions and remasks uncertain tokens. ProtLiD$^2$ improves global fold confidence over Complexa in whole-protein design, increasing TM-score from 0.672 to 0.802 and pLDDT from 64.55 to 73.00. In pocket co-design, ProtLiD$^2$ reduces active-site BB-RMSD from 3.46/3.40Å for FAIR/PocketGen to 1.97Å, and improves ligand-aware pass rates over PocketGen from 14.86% to 59.73% and from 6.08% to 23.49% under stricter docking thresholds. These results support ligand-conditioned discrete diffusion as an effective token-space framework for functional protein co-design. Code will be available at this https URL. 

---
# Advancing Direct Training for Spiking Neural Networks with Circulate-Firing Neurons and Learnable Gradients 

**Authors**: Feifan Zhou, Xiang Wei, Yang Liu, Qiang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.27412)  

**Abstract**: Spiking Neural Networks (SNNs) have emerged with promising energy-efficient property, yet a substantial performance gap persists compared to Artificial Neural Networks (ANNs). This gap stems from at least two key limitations: first, conventional spiking neurons offer limited information representation capacity, underutilizing the rich dynamics of membrane potentials; second, fixed surrogate gradient (SG) functions across time steps leads to imprecise gradient propagation, impeding effective direct training. To address these two challenges, we propose a new direct training algorithm with three core innovations: first, a circulate-firing spiking neuron model that enhances information representation capacity by leveraging membrane potentials more effectively; second, a time-step-wise learnable surrogate gradient function, enabling accurate gradient estimation during backpropagation; third, a positive-negative balanced loss function to achieve equilibrium between positive and negative membrane potentials and further boost SNN performance. Extensive experiments demonstrate that our methods achieve competitive performance across multiple datasets. Our methods can generalize seamlessly to advanced architectures of Transformer, consistently outperforming existing methods. Our work highlights the effectiveness of further harnessing intrinsic membrane dynamics of SNNs for performance improvement, and thus open a new avenue for advancing high-performance spiking neural architectures. 

---
# STARS: Spike Tail-Aware Relational Synthesis for ANN-to-SNN Data-Free Knowledge Distillation 

**Authors**: Shuhan Ye, Yi Yu, Qixin Zhang, Hui Lu, Jiaming He, Qinggang Zhang, Li Shen, Xudong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27409)  

**Abstract**: SNNs promise energy-efficient and low-latency inference, but their performance still trails that of ANNs. ANN-to-SNN knowledge distillation helps narrow this gap, yet the original training data are often unavailable in practical deployment settings. Existing data-free knowledge distillation (DFKD) methods synthesize surrogate data by matching teacher-side priors, especially BN statistics, but these ANN-oriented constraints mainly regularize mean and variance and therefore remain under-constrained for SNN students whose responses depend on threshold-crossing dynamics. In this paper, we propose Spike Tail-Aware Relational Synthesis (STARS), a plug-and-play method for ANN-to-SNN DFKD that augments standard BN-guided synthesis with two complementary objectives: Relational Consistency Alignment, which preserves cross-sample relational consistency between teacher and student, and Tail-Aware Regularization, which regularizes threshold-relevant tail probabilities through soft exceedance over teacher-derived thresholds. Together, these objectives generate synthetic batches that remain teacher-valid while becoming more informative for SNN students. Experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet across multiple ANN-SNN pairs show that our method consistently improves conventional DFKD baselines and even surpasses several KD methods, with gains of up to 4.6\% on CIFAR-10 and 6.7\% on CIFAR-100, highlighting the importance of complementing BN matching with relational and tail-aware constraints in SNN-oriented DFKD. 

---
# Benchmarking Fairness in Spiking Neural Networks: Data Bias, Spurious Features, and Hardware Effects 

**Authors**: Hudi He, Fukun Wang, Zhe Wang, Xinyi Wang, Shuhan Ye, Jiarui Liu, Qing Qing, Ziqi Xu, Xikun Zhang, Renqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.27407)  

**Abstract**: Evaluating fairness in Spiking Neural Networks (SNNs) demands rigorous benchmarks that reflect real-world complexities, yet existing assessments remain limited by superficial dataset diversity and idealized hardware assumptions. This work introduces the first systematic fairness benchmark for SNNs, addressing three critical dimensions of realism: (1) demographic coverage gaps in training data, (2) spurious feature leakage (e.g., skin tone as a proxy for class labels), and (3) deployment-environment mismatches (e.g., edge devices with constrained spike encoding). Our framework integrates four cross-demographic datasets with controlled bias injections and three neuromorphic hardware simulators (Loihi 2, SpiNNaker), enabling isolated analysis of fairness-performance trade-offs under resource constraints. Standardized evaluations of 12 state-of-the-art SNNs reveal stark disparities: models trained on biased data exhibit 23\% higher false positive rates for underrepresented groups, while hardware limitations (e.g., reduced spike precision) further amplify accuracy gaps by up to 41\% in edge deployments. Critically, bias mitigation strategies developed for cloud-based SNNs often degrade under resource constraints, highlighting the need for co-design principles that jointly optimize fairness and hardware efficiency. By bridging algorithmic fairness research with neuromorphic engineering, our benchmark provides a foundation for trustworthy SNNs in socially critical applications such as healthcare and autonomous systems. Our code is available at: this https URL. 

---
# Smaller, Younger, and More Impactful: How AI-Assisted Writing Transforms Research Teams 

**Authors**: Haoyang Wang, Mingze Zhang, Yi Bu, Star Xing Zhao, Meijun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.27404)  

**Abstract**: The era of Big Science has long been defined by increasingly large and specialized research teams pushing the frontiers of knowledge. However, recent advances in artificial intelligence (AI), particularly large language models (LLMs), are beginning to reshape academic writing and scientific research, potentially disrupting the longstanding trend toward ever-larger teams and transforming other dimensions of research team structure. Drawing on 147,074 full-text publications from the PLoS family and the Nature portfolio since 2020, we examined whether and how AI-assisted writing influences team structure and team outcomes in science. Using multiple methods, including ordinary least square, quantile regression, Poisson regression, logistic regression and propensity score matching, we found that research teams using AI-assisted writing tend to be younger and smaller. Importantly, this shift toward more compact, junior-leaning teams does not come at the expense of scientific impact. On the contrary, we observed a higher probability of research teams that employed AI-assisted writing producing highly impactful publications. These results highlight the significant role of AI-assisted writing in reshaping not only how research is produced, but also how research teams are formed and assembled. Our findings call for policy improvements in research evaluation, funding, and training to address this emerging trend. 

---
# LLM-assisted sentiment analysis for integrated computational and qualitative mixed methods education research: A case study of students' written reflection assignments 

**Authors**: Xiomara Gonzalez, Gabriella Coloyan Fleming, Andrew Katz, Maya Denton, Jessica Deters  

**Link**: [PDF](https://arxiv.org/pdf/2605.27403)  

**Abstract**: Written reflection assignments give students valuable opportunities for critical self-assessment, meaning making, and learning processing. Additionally, such reflections provide rich data for qualitative education research. However, qualitative data can be time-consuming to analyze. It is even more time-intensive to qualitatively compare findings between different groups of participants, usually limiting comparison to, at most, one variable (e.g., binary gender). Large language models (LLMs) have recently begun to be critically evaluated for use as qualitative research assistants. Using a longitudinal case of written student reflections (n=151) from a study abroad program, we investigate how LLM-assisted sentiment analysis can enable longitudinal mixed-methods research combining computational and thematic analyses. First, statistical testing is used to quantitatively compare sentiment differences according to seven different student identity/lived experience variables. Then, these results inform qualitative data analysis to investigate the reasons underlying these differences. For the case of undergraduate students studying abroad, we found that prior experience living abroad was the only personal variable impacting students' sentiments of their verbal language and communication behaviors. This workflow has implications for how qualitative researchers can more easily probe multiple variables when comparing participants from different demographic groups. 

---
# REC-CBM: Rubric-Aware Error-Correction Concept Bottleneck Models for Trustworthy Open-Ended Grading 

**Authors**: Chengshuai Zhao, Fan Zhang, Kumar Satvik Chaudhary, Yiwen Li, Lo Pang-Yun Ting, Ying-Chih Chen, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.27402)  

**Abstract**: Open-ended grading is central to equitable and personalized education, yet manual grading remains time-consuming and costly, underscoring the need for automated grading systems. Although recent neural and large language model (LLM) based systems have demonstrated superior performance, they are typically black-box models whose scoring processes and rationales are difficult for educators to verify and trust. Concept bottleneck models (CBMs) have emerged as a promising approach by routing predictions through human-interpretable concepts, providing a mechanistic guarantee of transparency. However, standard CBMs are not tailored to open-ended grading: they do not explicitly model fine-grained rubric dimensions, inadequately capture the ordinal semantics of scoring scales, and neglect inherent reliability issues in human concept annotations. To address these limitations, we propose REC-CBM, a rubric-aware error-correction concept bottleneck model for trustworthy open-ended grading. REC-CBM introduces a rubric-aware concept encoder that learns concept-specific representations over responses and an ordinal pairwise calibration objective that preserves ranking structure among rubric dimensions. It further incorporates a latent concept error-correction module that denoises concept predictions before final grade prediction while preserving interpretability. Comprehensive experiments on publicly available datasets show that REC-CBM consistently improves grading performance and produces more faithful concept-level reasoning than both state-of-the-art baselines. Further analyses validate the contribution of each component and demonstrate the applicability in realistic educational settings. Overall, this work provides a practical, interpretable grading solution that enables educators to inspect, intervene in, and trust automated decisions, advancing more transparent and trustworthy education. 

---
# Using Zero-Shot LLM-Generated Survey Data for Geographically Explicit Population Synthesis 

**Authors**: Taylor Anderson, Sara Von Hoene, Orhan Yagizer Cinar, Emma Von Hoene, Amira Roess, Andrew Crooks, Hamdi Kavak  

**Link**: [PDF](https://arxiv.org/pdf/2605.27401)  

**Abstract**: There is a growing interest in utilizing synthetic populations for a diverse range of applications. At the same time, we are witnessing a tremendous growth in artificial intelligence in all walks of life. This paper evaluates whether zero-shot large language model (LLM)-generated health survey data can serve as inputs to a conventional iterative proportional fitting (IPF) workflow for geographically explicit population synthesis. Using the 2023 Behavioral Risk Factor Surveillance System (BRFSS), we generate synthetic survey records for the U.S. states of Colorado and Mississippi with GPT-4.1 and Gemini-2.5-Pro. We use the generated data in an IPF-based synthesis pipeline and evaluate the resulting census tract-level synthetic populations against external benchmarks. Results show both LLMs capture several major state-level contrasts, indicating zero-shot generation produces geographically differentiated survey data. However, performance is strongly variable-dependent. Downstream effects in population synthesis are mixed, as IPF sometimes amplifies or reduces errors in the generated data. Spatial validation shows that LLM-based populations reproduce census tract-level patterns reasonably well, especially for variables that were more aligned with the ground truth data. Overall, the LLM-generated survey data shows promise as supplementary input, but not yet as a replacement for real survey data. 

---
# Mathematical Modelling of Ethical AI Use in Higher Education: A Coordination Game Framework for Future-Facing Learning 

**Authors**: Ndidi Bianca Ogbo, Zhao Song, Shatha Ghareeb, Anh Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.27400)  

**Abstract**: The rapid uptake of generative artificial intelligence (AI) in higher education is reshaping assessment practices and intensifying concerns around academic integrity, fairness, and learning quality. While institutional responses increasingly emphasise policy guidance and ethical principles, there remains limited formal understanding of how collective norms of responsible or opportunistic AI use emerge and stabilise within student cohorts. This paper reframes student AI use in assessment as a coordination problem shaped by peer expectations and assessment design rather than individual compliance alone. We develop a coordination-based evolutionary game-theoretic framework that captures learning value, effort, perceived fairness, and transparency, with institutional AI governance modelled implicitly through reflective assessment incentives.
We use analytical results and finite-population simulations to reveal threshold-driven behavioural transitions in student AI use: small, well-calibrated changes in reflective assessment incentives can trigger rapid shifts towards responsible, learning-oriented AI-use norms, whereas weak or misaligned incentives allow opportunistic practices to persist. These non-linear dynamics explain why policy statements alone often fail to change behaviour, while modest assessment redesigns can have disproportionate effects. By providing a mechanism-level account of how assessment structures shape collective AI-use practices, this work offers higher education institutions an analytically grounded tool for Future Facing Learning, supporting proportionate, pedagogy-led AI governance without reliance on surveillance or punitive enforcement. 

---
# Short-Term Gain, Long-Term Fragility: AI Labor Substitution and the Erosion of Sustainable Capability 

**Authors**: Wolfgang Rohde  

**Link**: [PDF](https://arxiv.org/pdf/2605.27399)  

**Abstract**: What looks like acceleration can be a quiet transfer of burden from the present to the future. Attempts to replace human labor with AI systems are often presented as rational responses to technological progress, but that view is often structurally short-sighted. Across software development and adjacent knowledge industries, AI is increasingly attractive because it appears to reduce labor costs, speed output, and improve short-term metrics. Yet those gains may be achieved by drawing down human capabilities that are slow to build and difficult to restore. This paper develops a mechanism of capability masking and capability erosion under AI labor substitution. AI-generated output can create the appearance that organizational capability has been replaced, even when dependence on skilled human labor remains. That appearance can support hiring restraint while slower costs accumulate in the background. Evidence from AI-assisted coding shows that generated output still requires substantial human verification and remains uneven in correctness, maintainability, and security. Repository-level studies also suggest limits in handling broader codebase context. More broadly, labor-market, political-economy, and industrial-strategy evidence suggests that substitution pressures are being driven by managerial cost incentives and national competition while increasing risks of concentration and platform control. The result is a system that may look more efficient in the short term while becoming more fragile over time. 

---
# Agentic Literacy Debt: A Structural Problem the AI Literacy Field Has Not Yet Named 

**Authors**: Rohith Nama  

**Link**: [PDF](https://arxiv.org/pdf/2605.27396)  

**Abstract**: Autonomous AI agents now plan, decide, and act on behalf of users across healthcare, financial services, and workplace contexts, often without step-by-step human approval. Existing AI literacy frameworks were built for a world in which humans evaluate AI outputs and decide whether to act; they have no vocabulary for the user who has delegated decision-making authority to an agent whose actions may not be observable, reversible, or controllable. This paper names the resulting problem agentic literacy debt: the accumulating societal deficit that grows when agentic AI systems are deployed at scale without corresponding literacy infrastructure. The debt compounds through three reinforcing channels (normalization of opaque delegation, multi-agent ecosystem complexity, and institutional path dependence), and it is incurred by the organizations that deploy agents but paid by the users, patients, and citizens on whose behalf the agents act. Evidence from healthcare, financial fraud, and global equity contexts suggests the gap is already consequential. The problem is structural, not a temporary lag that curriculum reform will close. It demands a reframing of AI literacy as a governance capability, not an evaluative one. 

---
# Informing AI Policy Assessment using Large-Scale Simulation of Interventions 

**Authors**: Julia Barnett, Kimon Kieslich, Natali Helberger, Nicholas Diakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2605.27395)  

**Abstract**: As the rapid proliferation of AI systems and harms spurs efforts in AI governance around the world, prioritizing among competing policy options has become increasingly challenging for policymakers and researchers. We introduce a methodology for identifying viable policy options to mitigate specified AI harms, helping policymakers and researchers target areas that warrant greater time and resource investment. This method combines participatory evaluation of policies, expert assessment of implementation costs, and an LLM-based assessment of perceived harm mitigation under each policy option. We leverage a genetic algorithm-based simulation study to explore a vast solution space of potential policy combinations, and examine how outcomes change under different weightings of cost, participatory input, and harm mitigation. We find that this method enables exploration of different balances between participatory and expert components, allowing policymakers and researchers to assess how much weight to assign to each. We argue that the diversity of viable policy combinations found by the genetic algorithm could be a useful starting point for deliberation. This method operationalizes existing work on participatory AI by integrating it directly into practical policy development pipelines. 

---
# Human-AI Collaboration for Estimating Scientific Replicability 

**Authors**: Tatiana Chakravorti, Robert Fraleigh, Timothy Fritton, Christopher Griffin, Vaibhav Singh, Sai Koneru, C. Lee Giles, David Pennock, Anthony Kwasnica, Sarah Rajtmajer  

**Link**: [PDF](https://arxiv.org/pdf/2605.27394)  

**Abstract**: Determining whether published scientific findings can successfully be replicated is a long-standing challenge in the empirical sciences. Existing approaches for replicability assessment typically rely either on human judgment, i.e., creative assembly of human experts, or on machine learning models trained on paper content metadata. While both approaches have demonstrated value, each also has important limitations. Human forecasts can be influenced by cognitive biases and narrow exposure to the research literature, while automated assessments often struggle to capture contextual cues and subtle signals of credibility. In this paper, we examine a hybrid approach. Specifically, we introduce a hybrid prediction market in which algorithmic agents trade alongside human participants to jointly estimate the likelihood that a published scientific finding will be corroborated via the outcome of a controlled replication study. Agents are trained on outcomes from hundreds of prior replication studies while human participants contribute domain knowledge through real-time trading. We evaluate this hybrid approach through multiple live experiments involving participants from different academic disciplines and compare its performance to artificial-only and human-only baselines. Our results show that, except for a few cases, hybrid markets match or outperform artificial prediction markets, producing more accurate and reliable replication forecasts. 

---
# StoryMI: Steerable Multi-Agent Therapeutic Dialogue Generation 

**Authors**: Qingyu Meng, Min Chen, Dingming Liu, Yifan Mo, Yue Su, Xin Sun, Koen Hindriks, Jiahuan Pei  

**Link**: [PDF](https://arxiv.org/pdf/2605.27393)  

**Abstract**: Large language models (LLMs) can generate fluent dialogue, but prior works lack situational grounding, dynamic strategy control, and evaluation aligned with clinical standards in motivational interviewing (MI). We introduce StoryMI, a multi-LLM agent framework for controllable MI dialogue generation, where questionnaire-based client profiles are expanded into situational stories that provide narrative context for the dialogue. Therapist and client agents generate MI-coded utterances guided by MI codes selected by the interaction agent, while an interaction agent dynamically coordinates exchanges to control MI strategies during a multi-turn conversation. We propose a two-level evaluation protocol: lexical metrics and MI-specific measures of macro-level counseling strategies, alongside LLM-as-judge and human expert assessments. We construct a dataset of 6K simulated MI dialogues grounded in 1K questionnaire-story pairs, covering 12 MI codes and 13 symptom domains, and benchmark six open- and closed-source LLMs. Our results show that situational grounding and macro-level control can improve MI adherence and clinical plausibility, demonstrating the effectiveness of a structured multi-agent workflow for psychotherapy dialogue generation. We provide code and data for reproducibility. 

---
# Learning after COVID-19 and the ICT career aspirations: Are students entering the AI era with weaker skills? 

**Authors**: Diana Maria Popa, Simona-Vasilica Oprea, Adela Bâra  

**Link**: [PDF](https://arxiv.org/pdf/2605.27391)  

**Abstract**: This paper examines whether students are entering the generative AI era with sufficiently strong educational foundations, focusing on the relationship between learning environments and changes in ICT related career aspirations across countries. The analysis uses country-level data from PISA 2018 and 2022, combining indicators of student autonomy, digital skills and teacher support. A mixed-method approach is applied, including descriptive statistics, regression analysis, clustering, latent representation learning (using Variational Autoencoder-VAE), discriminant analysis and probabilistic modeling to capture both observable and latent dimensions of educational readiness. Unlike prior research that treats learning loss, digital skills and career expectations separately, our analysis integrates them within a comparative longitudinal framework. It shifts the focus from short-term post-pandemic effects to the structural capacity of education systems to prepare students for digital and AI-driven labor markets. Results show a global but uneven increase in ICT career aspirations. Digital skills emerge as the strongest and most consistent predictor, while teacher support plays a complementary role. Autonomy shows weaker, context-dependent effects. Educational readiness is multidimensional, and ICT aspirations evolve relatively independently from other career domains. 

---
# EvoSpec: Evolving Speculative Decoding via Real-Time Vocabulary and Parameter AdaptationTarget 

**Authors**: Shuyu Zhang, Lingfeng Pan, Qicheng Wang, Yaqi Shi, Yueyang Tan, Ruyu Yan, Jiaqi Chen, Lixing Du, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27390)  

**Abstract**: Speculative decoding accelerates Large Language Model inference via a draft-then-verify paradigm, yet the output projection layer becomes a bottleneck as vocabulary sizes scale. While existing static pruning methods effectively reduce this overhead, they suffer from precipitous drops in acceptance rate in specialized domains or topic-switching scenarios due to their inability to capture dynamic distribution shifts. To address this, we introduce EvoSpec, a framework that enables real-time evolution of the draft model through dynamic vocabulary and parameter adaptation. Unlike static or purely retrieval-based approaches, EvoSpec employs a context-aware mechanism that retrieves critical long-tail tokens via efficient semantic and statistical indexing. Furthermore, we propose a lightweight online alignment strategy utilizing curriculum learning to continually minimize the distributional gap between the draft and target models. Extensive evaluations across specialized domains (coding, law, and medicine) confirm that EvoSpec overcomes the limitations of static baselines. On EAGLE-3, it achieves a 1.13x speedup in these settings over the state-of-the-art static baseline FR-Spec, with 27\% lower memory overhead than standard online adaptation. 

---
# Memory-Based vs. Context-Only Conditioning Produces Distinct Behavioral Patterns in Stateful Personalization 

**Authors**: Junsoo Park, Youssef Medhat, Htet Phyo Wai, Ploy Thajchayapong, Ashok K. Goel  

**Link**: [PDF](https://arxiv.org/pdf/2605.27389)  

**Abstract**: We study how conditioning context shapes personalization behavior in a teacher-facing educational recommender system. We compare contextual conditioning based on the current student question with memory-based conditioning using persistent learner information. Using deviation correlation and paired statistical tests, we find that contextual recommendations exhibit stronger question-level responsiveness, while memory-based recommendations exhibit history-dependent behaviors, including learner-specific differentiation under identical input. Teacher-facing evaluation signals suggest these recommendations are interpretable and actionable. These results indicate that embedding-based similarity metrics capture responsiveness to the current question but do not characterize personalization grounded in learner history, motivating behavior-level diagnostics for studying conditioning effects. 

---
# Modeling Community Attitude through Reaction Tone: A Human-AI Collaborative Framework for Evaluating LLM Alignment with Linguistic Behaviors in Online Communities 

**Authors**: Nuan Wen, Xuezhe Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.27388)  

**Abstract**: Large language models (LLMs) are increasingly utilized as proxies for computational social analysis; yet, their ability to faithfully represent the "thick descriptions" (Geertz, 1973) of human communities remains a critical challenge. Current evaluations often reduce social identity to static labels, sidelining how real-world groups navigate social shifts. To bridge this gap, we introduce CARE (Community-Aware Reaction Evaluation), a reaction-centered framework that benchmarks LLM-simulated discourse against the authentic, event-contingent responses of distinct communities to real-world news. By characterizing a fine-grained spectrum of illocutionary tones and the underlying attitudes they manifest--validated through human-AI collaboration--our diagnosis reveals a persistent "realism gap": steering LLMs with explicit community prompts fails to inherently improve simulation fidelity. Analysis further identifies divergent behavioral signatures among frontier models, suggesting that current alignment strategies remain insufficient for capturing the sociolinguistic dynamics of online groups. 

---
# From AR to Diffusion: Efficiently Adapting Large Language Models with Strictly Causal and Elastic Horizons 

**Authors**: Xiangyu Ma, Teng Xiao, Zuchao Li, Lefei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.27387)  

**Abstract**: Diffusion models promise efficient parallel text generation but rely on bidirectional attention, creating a structural mismatch with pre-trained Autoregressive (AR) models. This incompatibility precludes reusing robust AR priors, necessitating prohibitive pre-training from scratch. To bridge this gap, we propose FLUID, a framework that efficiently adapts AR backbones to the diffusion paradigm. By enforcing Strictly Causal Alignment, FLUID enables seamless initialization from standard GPT-style checkpoints, circumventing the need for massive pre-training. Furthermore, we introduce Elastic Horizons, an entropy-driven mechanism that dynamically modulates denoising strides based on local information density rather than fixed schedules. Experiments demonstrate that FLUID achieves state-of-the-art performance while reducing training costs by orders of magnitude, effectively reconciling established AR foundations with efficient parallel generation. Our code is available at this https URL. 

---
# Personalized Observation Normalization for Federated Reinforcement Learning in Simulation Environments with Heterogeneity 

**Authors**: Yiran Pang, Zhen Ni, Xiangnan Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2605.27385)  

**Abstract**: Federated reinforcement learning (FedRL) enables multiple agents to collaboratively train a global policy without sharing raw data, making it ideal for privacy-sensitive applications. However, FedRL faces challenges in heterogeneous environments where differing state-transition dynamics lead to non-identical input distributions and imbalanced parameter updates during aggregation. Therefore, this paper develops a personalized observation normalization (PON) method, allowing each agent to locally normalize raw state inputs using a continuously updated running mean and variance. This design ensures consistent scaling of local feature without overshadowing across agents during aggregation. Furthermore, we demonstrate that sharing normalization parameters across agents is ineffective due to the diverse local input distributions, which highlights the necessity of personalized statistics. Experiments on heterogeneous MuJoCo tasks show that our developed PON accelerates training and achieves superior performance compared to baseline methods. 

---
# From Instructor to Collaborator: What a 90-Participant Study Reveals about Human-Agent Collaboration in a Mobile Serious Game 

**Authors**: Danai Korre  

**Link**: [PDF](https://arxiv.org/pdf/2605.27384)  

**Abstract**: This position paper reflects empirical data collected during my PhD from a large-scale within-subjects study (N = 90). The study compared a highly human-like, spoken embodied conversational agent (ECA) against a low human-like text base agent (no embodiment, text bubble only) within a mobile, Unity-developed game about pre-decimal UK currency. The game included two agents with different roles-an Instructor (Alex) and a Shopkeeper/Collaborator. Users interacted using voice and mouse input. The quantitative data I collected included a usability questionnaire (CCIR MINERVA) and the Agent Persona Instrument. Data was analyzed using paired t-test, repeated measures ANOVA and multiple linear regression to identify correlations between the persona and usability. The results showed a statistically significant preference for the version of highly human-like agents, with a large effect size. This is further discussed alongside qualitative findings from observations and exit interviews. The results are framed for Human-Agent collaboration, especially for how roles, mixed-initiative dialogue, and breakdowns/repairs become apparent in goal-oriented tasks. I conclude with questions on timing, user expectations, and role-specific interactions. This submission does not propose new frameworks; it reports empirical findings and questions I hope to workshop with the community. 

---
# Bridging the Stability-Expressivity Gap: Synthetic Data Scaling and Preference Alignment for Low-Resource Spoken Language Models 

**Authors**: Yizhong Geng, Yanliang Li, Jinghan Yang, Tianhan Jiang, Boxun An, Ya Li, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.27383)  

**Abstract**: Spoken Language Models (SLMs) have emerged as a promising paradigm for speech synthesis by bypassing explicit grapheme-to-phoneme pipelines. However, their effectiveness in low-resource languages remains fundamentally limited by the scarcity of transcribed speech. In practice, synthetic data has become the primary strategy for scaling SLMs in such settings, providing reliable phonetic supervision when real data is insufficient. In this work, we show that this reliance introduces a fundamental trade-off, which we term the Stability-Expressivity Gap: while synthetic data improves phonetic accuracy, it progressively suppresses prosodic variability, ultimately leading to a collapse of expressivity (Synthetic Erosion). To bridge this gap, we propose two self-alignment frameworks. Disentanglement-Guided Self-Alignment (DGSA) recovers expressivity for complex languages by exploiting prosody-timbre separation. For regimes where authentic references are exceptionally limited, Temperature-Driven Self-Critique (TDSC) stabilizes generation through automated exploration and filtering. Our approach outperforms strong commercial systems, including ElevenLabs and Gemini Pro, and enables the first zero-shot voice cloning capability for Lao. 

---
# The Alignment Floor: When Persona Customization Is Safe 

**Authors**: Xing Zhang, Guanghui Wang, Yanwei Cui, Wei Qiu, Ziyuan Li, Bing Zhu, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2605.27382)  

**Abstract**: A key promise of pluralistic AI is behavioral adaptation: persona prompts like "be creative" or "be thorough" let systems respect diverse user values and communication styles. But how much customization can a model absorb before its alignment breaks? We present the first controlled study of the alignment-customization tradeoff, testing seven persona conditions across five tasks on two models with different alignment strengths (1,800 runs). We discover the alignment floor: on a strongly-aligned model (Claude Sonnet), persona prompts have zero effect on sycophancy -- all conditions produce ~15%, a stable platform on which rich personalization is safe. On a weakly-aligned model (Nova Lite), the same personas shift sycophancy from 5% to 50% -- the floor is absent and customization becomes a safety liability. Surprisingly, Agreeableness is not the worst offender; Extraversion (+20pp) and Openness (+15pp) cause greater degradation. The constructive finding is the Skeptic defense: a critical-thinking persona reduces sycophancy to 5% even on the weak model -- the single largest effect in the study. Cross-model transfer of persona effects is near-zero ($\rho = 0.006$), meaning alignment testing must be per-model. We propose the alignment floor as a design principle: measure it before deploying persona customization, and layer safety-oriented personas underneath user-facing ones to enable personalization without compromising alignment. 

---
# The Computational Boundary of Inference: Capability Internalization, Training, and the Turing Jump 

**Authors**: Chien-Ping Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.27381)  

**Abstract**: Claims about recursive self-improvement in AI often slide from repeated internal revision to the possibility of qualitatively stronger capability without clearly distinguishing the underlying computational regimes. This paper gives a formal separation result in classical computability theory that blocks that move under a precise modeling assumption. For an oracle $A$, let $\mathcal{C}(A)=\{B : B \leq_T A\}$ be the corresponding computational layer. We prove that finite internal self-modification remains inside $\mathcal{C}(A)$, while stabilized revision is governed instead by the jump $A'$ via the relativized limit lemma. Together with a local closure versus escape theorem, this yields a clean formal separation between within-layer iteration and ascent to a stronger relative level. The point is not that stronger layers never arise, but that they are not explained by finite repetition inside one already settled layer. The resulting separation gives a computability-theoretic limit on a broad class of recursive-improvement narratives in which repeated internal updating is treated as sufficient for qualitative capability ascent. 

---
# BioELX: Cross-lingual Biomedical Entity Linking via Alias-based Retrieval and LLM Ranking 

**Authors**: Yi Wang, Corina Dima, Liangyu Zhong, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2605.27380)  

**Abstract**: Cross-lingual biomedical entity linking (BEL) maps mentions in any language to unique identifiers in a biomedical knowledge base (KB), supporting clinical and biomedical NLP applications. However, expert-annotated training data for BEL are costly, especially for low-resource languages. Moreover, many cross-lingual BEL systems rely on SapBERT-based retrievers trained on predominantly English aliases in the KB, leading to poor generalization to unseen non-English mentions and limited context-aware disambiguation. We propose BioELX, a two-stage cross-lingual BEL framework that requires no task-specific annotated training corpora. In Stage~1, we enrich SapBERT training with Wikidata-derived multilingual aliases and use the resulting retriever to improve cross-lingual candidate retrieval. In Stage~2, we perform context-aware disambiguation with a pre-trained LLM ranker that jointly considers the mention context and candidate, eliminating the need for supervised training. Experiments on five benchmarks (XL-BEL, EMEA, Patent, WikiMed-DE, and MedMentions) show that BioELX achieves new state-of-the-art performance. It improves average Recall@1 on XL-BEL by +19.2, with especially large gains for low-resource languages, e.g., +21.6 on Turkish, +22.1 on Korean, +30.8 on Thai, and delivers consistent improvements on EMEA (+6.2), Patent (+5.4), and WikiMed-DE (+12.8). Code and resources will be released upon publication. 

---
# RAG-Coding: Enhancing LLM Medical Coding with Structured External Knowledge 

**Authors**: Yidong Gan, David D. Nguyen, Yang Lin, Peter Zhong, Thanh Vu, Long Duong, Yuan-Fang Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.27377)  

**Abstract**: We present RAG-Coding, an agentic method for automated ICD-10-CM coding. RAG-Coding orchestrates four large language model (LLM) agents and grounds their coding decisions in external knowledge sources (e.g. the official coding tabular list and guidelines). By retrieving and cross-referencing relevant knowledge in these sources, the agents enhance coding accuracy and ensure clinical compliance. On the MDACE dataset, RAG-Coding outperforms the best LLM-based baseline by 8-13\% in micro-F1 and 2-8\% in macro-F1 across multiple LLM backbones. Compared to the state-of-the-art pretrained language model method, PLM-ICD, RAG-Coding exhibits higher micro recall (+11\%), while PLM-ICD exhibits higher micro precision (+6\%), yielding comparable micro- and macro-F1. Ablations show stepwise gains, highlighting the importance of incorporating external knowledge. We also release MDACE-2025, updating the original dataset with expert re-annotations with the latest 2025 ICD-10-CM guidelines. This update features more fine-grained code labels and enables evaluation against current clinical standards. 

---
# Unlocking Fine-Grained and Within-Utterance Speaking Style Control in Prompt-Based Text-to-Speech Models 

**Authors**: Jaehoon Kang, Yejin Lee, Yoonji Park, Kyuhong Shim  

**Link**: [PDF](https://arxiv.org/pdf/2605.27376)  

**Abstract**: While prompt-based text-to-speech (TTS) models enable natural language-driven speaking style control, they often provide limited fine-grained control and apply a single global style across an utterance. This restricts practical use cases that require continuous style attribute interpolation across utterances and time-varying style transitions within a single utterance. In this paper, we propose novel techniques to achieve both capabilities in existing prompt-based TTS models. For inter-utterance style interpolation, we compute direction vectors between contrastive style prompts in the embedding space and perform simple interpolation, enabling smooth transitions between style characteristics. For intra-utterance style transition, we first identify a strong attention bias toward early tokens in autoregressive TTS decoders, causing the initial audio realization to dominate subsequent generation. To mitigate this effect, we introduce KV-cache swapping and sliding-window attention masking. Experiments demonstrate that our proposed inter-utterance interpolation achieves a 99-100% success rate in gender conversion, up to 36 Hz pitch variation, and up to 1.6 syllables-per-second speed change. Our intra-utterance transition maintains a speaker similarity of 0.81-0.91 and achieves perceptual smoothness scores of 3.48-4.48. 

---
# LNN-PINN: A Unified Physics-Only Training Framework with Liquid Residual Blocks 

**Authors**: Ze Tao, Hanxuan Wang, Fujun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08935)  

**Abstract**: Physics-informed neural networks (PINNs) have attracted considerable attention for their ability to integrate partial differential equation priors into deep learning frameworks; however, they often exhibit limited predictive accuracy when applied to complex problems. To address this issue, we propose LNN-PINN, a physics-informed neural network framework that incorporates a liquid residual gating architecture while preserving the original physics modeling and optimization pipeline to improve predictive accuracy. The method introduces a lightweight gating mechanism solely within the hidden-layer mapping, keeping the sampling strategy, loss composition, and hyperparameter settings unchanged to ensure that improvements arise purely from architectural refinement. Across four benchmark problems, LNN-PINN consistently reduced RMSE and MAE under identical training conditions, with absolute error plots further confirming its accuracy gains. Moreover, the framework demonstrates strong adaptability and stability across varying dimensions, boundary conditions, and operator characteristics. In summary, LNN-PINN offers a concise and effective architectural enhancement for improving the predictive accuracy of physics-informed neural networks in complex scientific and engineering problems. 

---
