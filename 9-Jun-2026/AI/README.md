# Evaluation Cards: An Interpretive Layer for AI Evaluation Reporting 

**Authors**: Avijit Ghosh, Anka Reuel, Jenny Chim, Wm. Matthew Kennedy, Srishti Yadav, Jennifer Mickel, Yanan Long, Andrew Tran, Anastassia Kornilova, Damian Stachura, Kevin Klyman, Felix Friedrich, Jeba Sania, Max Lamparth, Jan Batzner, Anoop Mishra, Eliya Habba, Yixiong Hao, Nathan Heath, Shalaleh Rismani, Usman Gohar, Andrea Loehr, David Manheim, Ruchira Dhar, Sree Harsha Nelaturu, Aarush Sinha, Leshem Choshen, Drishti Sharma, Ishan Khire, Amit Saha, Subramanyam Sahoo, Michael Hardy, Michael Alexander Riegler, Kabir Manghnani, Michelle Lin, Yanan Jiang, Yilin Huang, Asaf Yehudai, Jessica Ji, Aris Hofmann, Mubashara Akhtar, Nuno Moniz, Yacine Jernite, Stella Biderman, Zeerak Talat, Sanmi Koyejo, Mykel Kochenderfer, Irene Solaiman  

**Link**: [PDF](https://arxiv.org/pdf/2606.09809)  

**Abstract**: AI evaluation results are produced at scale but reported inconsistently across leaderboards, model cards, benchmark papers, and company blogs. The cost is interpretive: readers cannot reliably compare results across sources, identify what a report omits, or trace an aggregate claim to its underlying evidence. Recent efforts address isolated components but leave three gaps: they cover only narrow slices of the evaluation lifecycle and do not compose into a single interpretable record; they specify static representations that do not differentiate the questions different stakeholders bring to the same evidence; and they remain proposals on paper, lacking the extraction infrastructure required for adoption at scale. We present \EvalCards{}, an operational reporting layer that composes benchmark metadata, evaluation run data, and model metadata into a unified record. We (1) derive a reporting schema from a structured review of 52 papers and 10 stakeholder interviews, (2) implement four interpretive signals (reproducibility, documentation completeness, provenance and risk, and score comparability), rendered through reader modes calibrated to research and non-research audiences, and (3) deploy a monitoring tool that applies \EvalCards{} across 5,816 models, 635 benchmarks, and 101,843 results, surfacing systematic gaps in current reporting practice. 

---
# SIGA: Self-Evolving Coding-Agent Adapters for Scientific Simulation 

**Authors**: Matthew Ho, Brian Liu, Jixuan Chen, Audrey Wang, Lianhui Qin  

**Link**: [PDF](https://arxiv.org/pdf/2606.09774)  

**Abstract**: Advanced scientific simulators expose specialized input languages that turn simulation goals into executable configurations, but learning them can cost domain scientists hours to days. We study simulator setup as a problem of agent-tool interface grounding: what minimal simulator-specific adaptations are needed for an off-the-shelf coding agent to operate real scientific software? Our intuition is that coding agents already know how to navigate files, edit code, run commands, and repair outputs, but they lack the simulator's executable contract: its vocabulary, structural constraints, validation rules, and termination conditions. We introduce SIGA, a Simulator-Interface Grounding Adapter that supplies this contract through retrieval, procedural memory, in-trajectory validation, and validation-enforced termination. We primarily evaluate SIGA on GEOS, an open-source multiphysics simulator used in subsurface science. SIGA produces a complete GEOS deck in about five minutes with TreeSim above 0.90, matching an extended-budget human expert who took about three hours, a roughly 36x wall-clock speedup. On a harder held-out set, grounding raises TreeSim from 0.720 to 0.789, a roughly 10% relative gain over the bare agent, and can reduce the across-seed standard deviation by 16x. Self-evolution further improves SIGA by rewriting adapter contents from prior trajectories, yielding the highest held-out GEOS mean and matching or outperforming the strongest hand-designed configuration. Transfers to OpenFOAM and LAMMPS show that the dominant mechanism shifts by interface: validation matters most when structural completeness is the bottleneck, while memory and retrieval matter most when domain correctness is the bottleneck. These results suggest that lightweight, self-improvable grounding layers can turn general coding agents into practical operators of scientific software. 

---
# Collaborative Human-Agent Protocol (CHAP) 

**Authors**: Arsalan Shahid, Gordon Suttie, Philip Black  

**Link**: [PDF](https://arxiv.org/pdf/2606.09751)  

**Abstract**: Foundation models are moving from response generation into operational roles. They plan across steps, call tools, request human input, coordinate with other agents, and increasingly carry responsibility for work that affects customers, claims, code, contracts, and clinical decisions. Production deployments are no longer one human supervising one model. They are multi-human, multi-agent collaborations that cross teams, time zones, and trust boundaries. The technical surface for this collaboration remains weakly specified. When an agent drafts a response and a human edits it before it ships, the moment of human judgement is the most valuable signal in the system. In current practice it is recorded, if at all, in application code, chat threads, ticket comments, and tribal memory. Two protocol standards address adjacent concerns: MCP standardises agent access to tools and data, and A2A standardises agent-to-agent interoperability. Neither defines the shared workspace in which humans and agents perform accountable work together. This paper presents CHAP, the Collaborative Human-Agent Protocol. Under CHAP, the override that used to vanish into a chat thread becomes a structured event carrying a diff, a rationale, and a content hash. The handoff between shifts becomes a portable envelope rather than a pinned message. The human approval of an agent's draft becomes a non-repudiable signed decision that can be replayed years later. The protocol achieves this through a small Core (workspaces, participants, tasks, artefacts, and an append-only evidence log) together with composable profiles that add review, modes, routing, deliberation, handoff, identity, signatures, and transparency-backed audit as deployments require them. Specification, reference implementation, conformance suite, and worked examples are available at: this https URL 

---
# Multi-Turn Evaluation of Deep Research Agents Under Process-Level Feedback 

**Authors**: Rishabh Sabharwal, Hongru Wang, Amos Storkey, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2606.09748)  

**Abstract**: Existing benchmarks for deep research agents (DRAs) assess only single-shot outputs, ignoring a key question: can DRAs improve their reports when guided by feedback? To investigate this, we conduct a multi-turn evaluation of DRAs under two feedback settings: self-reflection, in which the agent revises its report without any external diagnostic signal, and process-level feedback, in which the agent receives guidance targeting gaps in its research strategy. To enable process-level feedback, we design Research Gap Inference (RGI), a method that analyzes patterns of satisfied and unsatisfied rubric criteria to infer research-process gaps. Our analysis reveals three key findings: (i) under self-reflection, agents incorporate and regress on rubric criteria at nearly equal rates, yielding negligible net improvement; (ii) a single round of process-level feedback yields substantial gains, raising the normalized score by approximately $8$-$15$ points and yielding a roughly $35$-$40\%$ incorporation rate; (iii) these gains do not compound over subsequent turns, as agents regress on up to $24\%$ of previously satisfied criteria when rewriting the full report to address remaining gaps. Even with targeted guidance, reliable multi-turn improvement remains out of reach for the DRA architectures we evaluate. Our code and results are publicly available at this https URL. 

---
# SearchSwarm: Towards Delegation Intelligence in Agentic LLMs for Long-Horizon Deep Research 

**Authors**: Pu Ning, Quan Chen, Kun Tao, Xinyu Tang, Tianshu Wang, Qianggang Cao, Xinyu Kong, Zujie Wen, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.09730)  

**Abstract**: Large language models are increasingly expected to handle complex, long-horizon real-world tasks whose context demands can grow without bound, yet model context windows remain inherently finite. Recent work explores a paradigm where a main agent decomposes tasks and dispatches subtasks to subagents, which execute and return only summarized results, conserving the main agent's context budget. However, performing this well requires delegation intelligence: the ability to decompose complex tasks, determine when and what to delegate, and integrate returned results into the ongoing workflow. Training data for this capability is scarce in naturally occurring text, and to our knowledge, how to synthesize such data and train models to acquire this capability remains largely unexplored in the open-source community. To bridge this gap, we present a preliminary exploration targeting deep research, a representative long-horizon agent task. Specifically, we design a harness that guides the model toward high-quality task decomposition and delegation, while constraining subagents to return results properly to support the main agent's workflow. The harness-guided trajectories naturally encode correct delegation decisions, which we use as supervised fine-tuning data to internalize delegation intelligence into model weights. Our resulting model, SearchSwarm-30B-A3B, achieves 68.1 on BrowseComp and 73.3 on BrowseComp-ZH, the best results among all models of comparable scale. We will release our harness, model weights, and training data to facilitate future research. 

---
# Beyond Probabilistic Similarity: Structural, Temporal, and Causal Limitations of Retrieval-Augmented Generation in the Legal Domain 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2606.09724)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a standard architectural response to unreliability in legal AI, yet high-profile failures, including fabricated citations submitted to courts and anachronistic legal content presented as current, continue to appear across jurisdictions. We argue that these failures are not residual confabulations to be eliminated by scaling language models, but symptoms of an architectural mismatch between probabilistic retrieval and the hierarchical, temporal, and institutional structure of legal knowledge. We develop the argument in three moves. First, we articulate the ontological commitment of legal knowledge as a triad of properties derivable from classical legal theory: hierarchical and mereological structure, diachronic dynamism under operational closure, and causal traceability of institutional provenance grounded in the duty of justification. Second, we identify three corresponding pathologies of retrieval (mereological blindness, diachronic blindness, and causal opacity), each developed with an operational definition, a failure mechanism, a canonical example, and detection criteria for diagnostic use. Third, we review the state of the art through this lens, showing that existing approaches address these requirements unevenly and do not yet compose into a paradigm that treats them as co-constitutive. From this analysis we derive four architectural commitments that characterize the deterministic-by-design direction for legal retrieval: ontological primacy, event reification, bitemporal correctness, and deterministic interaction protocols. The framework concerns quaestio juris (which norms apply and in what state) rather than the downstream tasks that act on identified norms, and addresses legislative and constitutional retrieval primarily, with interpretive time as an explicit extension. 

---
# Proxy Reward Internalization and Mechanistic Exploitation: A Learned Precursor to Reward Hacking and Its Generalization 

**Authors**: Mohammad Beigi, Ming Jin, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09711)  

**Abstract**: Reward hacking is usually studied after it becomes visible, once a model earns high proxy reward while failing the intended task. We instead study what proxy RL teaches before that failure appears. We introduce Proxy Reward Internalization and Mechanistic Exploitation (PRIME), a learned capability to assess task correctness, predict proxy acceptance, and reason about exploitable proxy--gold gaps. In coding RL environments with exploitable pytest rewards, we measure PRIME through chain-of-thought monitoring, direct probes, and activation-level concept vectors. We find that PRIME emerges in a staged sequence before sustained reward hacking, and that its current direct-probe score forecasts later hack onset and severity even when the visible hack rate is still low. PRIME also adapts when the evaluator changes, retargeting to whichever proxy--gold gap remains rewarded and persisting when gold reward suppresses overt hacking, and ablating its activation directions reduces hacking. Across checkpoints, in-domain PRIME tracks out-of-domain misalignment. Together these results suggest that exploitable proxy RL amplifies a proxy-internalization capability upstream of visible hacking, making PRIME a candidate early-warning signal for broader alignment risk. 

---
# (Auto)formalization is supposed to be easy: Trellis process semantics for spelling out rigorous proofs 

**Authors**: Wesley Pegden  

**Link**: [PDF](https://arxiv.org/pdf/2606.09674)  

**Abstract**: We present Trellis: an autoformalization system that leverages LLM agents in a deterministically constrained workflow to enforce incremental progress in Lean autoformalization tasks through iterative refinement of natural language proofs. Our approach is motivated by the common mathematician's notion of what it means to have a rigorous proof in the first place: namely, that it would be routine to elaborate any part of the proof in further detail. The result is a system which aims to achieve reliable autoformalization on a modest budget and with generalist agents, with specialization to autoformalization coming not from any task-specific agent training but instead from a meaning-of-rigor inspired workflow enforced by process semantics. We link to an end-to-end Lean formalization of a recent Ramsey theory breakthrough produced by the process. 

---
# Correlation Is Not Enough: Embedding Human Metadata for Individual Causal Discovery 

**Authors**: Suraj Biswas, Saurabh Gupta, Pritam Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.09672)  

**Abstract**: Ask a pretrained biomedical language model whether "cortisol 28 ug/dL" and "stock-market volatility" are related, and it returns a cosine similarity of 0.83 on a scale where 1.0 means identical. The two share no mechanism. This is not a corner case: every off-the-shelf biomedical encoder we tested (BioBERT, PubMedBERT, BioM-ELECTRA) scores unrelated cross-domain pairs between 0.76 and 0.92 when the answer should be near zero. Accuracy on cross-domain discrimination is 0%.
Retrieval systems survive this, because a language model downstream filters the noise. A Large Behavioural Model (LBM), a foundation model whose subject is a person rather than a sentence, does not: it reasons over a graph of a user's life and treats embedding proximity as evidence that two events are causally linked. False proximity writes a false causal edge, and everything downstream inherits the error. Here, embedding geometry is not a tuning knob; it is correctness.
We report the fix. A contrastive pass over 72,034 pairs raises PubMedBERT BIOSSES correlation from 0.633 to 0.828 and within-vs-across-domain separation from 1.05x to 1.63x. A second pass, BODHI, mines hard negatives from edges absent in a biomedical knowledge graph and lifts separation to 2.30x and the discrimination gap to +0.392, at a 4.5% BIOSSES cost. On an Intel Xeon 6737P with AMX, OpenVINO cuts single-query latency from 1367 ms to 10 ms (133x) and reaches 555 sentences/sec. One finding contradicts standard advice: FP16 beats INT8 on this silicon at every serving batch size, and we explain why. The same model on a no-AMX Ice Lake instance runs 13-27x slower. We release the benchmark suite, training corpora, the BODHI generator, and the OpenVINO scripts. 

---
# SpatialWorld: Benchmarking Interactive Spatial Reasoning of Multimodal Agents in Real-World Tasks 

**Authors**: Hongcheng Gao, Hailong Qu, Jingyi Tang, Jiahao Wang, Zihao Huang, Hengkang Qiao, Shihong Huang, Junming Yang, Yi Li, Hongyixuan Yuan, Wenjie Li, Bohan Zeng, Wenbo Li, Bo Wang, Jianhui Liu, Olive Huang, Haoyang Huang, Wentao Zhang, Guoqing Huang, Nan Duan, Yinpeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2606.09669)  

**Abstract**: Spatial reasoning is a foundational capability for multimodal large language models (MLLMs) to perceive and operate within the physical world. However, existing benchmarks predominantly rely on passive evaluation (e.g., static VQA) or simulator-specific pipelines, failing to assess general interactive spatial understanding. We introduce SpatialWorld, a unified benchmark designed specifically for evaluating the interactive spatial understanding of multimodal agents in complex real-world tasks. Integrating eight heterogeneous simulation backends under a shared, simulator-agnostic protocol, SpatialWorld features 760 human-annotated tasks across diverse domains (e.g., household routines, travel, social collaboration). Agents must solve tasks under vision-only partial observability, actively gathering egocentric visual evidence and expressing decisions via a unified, text-based action interface native to MLLMs. For reliable evaluation, each task includes a human-validated initial state, a reference trajectory, and a terminal-state verifier. Evaluating 15 advanced agents reveals that robust spatial task solving remains challenging: the strongest model, GPT-5, achieves an average task success rate (TSR) of only 17.4%, while the leading open-source model, Qwen-3.5, reaches 14.1%. Further analysis exposes a clear mismatch between task success and execution efficiency, alongside substantial domain-specific performance variations. These bottlenecks in active exploration and long-horizon planning position SpatialWorld as a rigorous testbed for future spatial agents. 

---
# Frequency-based Constrained Sampling for Interval Patterns 

**Authors**: Djawad Bekkoucha, Abdelkader Ouali, Bruno Crémilleux  

**Link**: [PDF](https://arxiv.org/pdf/2606.09666)  

**Abstract**: Output space pattern sampling is a powerful alternative to exhaustive pattern mining for exploring large pattern spaces, as it enables users to focus on representative patterns drawn according to a chosen interestingness measure. In this paper, we address the problem of sampling interval patterns under user-defined syntactic constraints. We introduce CFips, a sampling approach that incorporates constraints directly into the sampling procedure. The approach relies on a multi-step sampling framework and supports several syntactic constraints by decomposing them into elementary predicates on interval bounds while preserving exact sampling guarantees. We formally prove that CFips samples interval patterns proportionally to their frequency within the constrained pattern space. The experimental results show that integrating constraints into the sampling procedure enables to complete mining tasks that would otherwise fail within a given time out. 

---
# From 0-to-1 to 1-to-N: Reproducible Engineering Evidence for MetaAI Recursive Self-Design 

**Authors**: Dun Li, Jiatao Li, Hongzhi Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.09663)  

**Abstract**: Recursive self-design refers to AI-assisted modification of the mechanisms by which an AI system is built, evaluated, and improved. This paper treats MetaAI not as a mature paradigm, but as a working term for a human-seeded, AI-expanded development pattern in which the design space itself becomes a target of modification. We propose an operational evidence framework with four criteria: inspectable target system, meta-level modifier, feedback-directed selection, and recursive continuation. We then map public systems, including Darwin Goedel Machine (DGM), STOP, Goedel Agent, and ShinkaEvolve, against these criteria. DGM provides the most direct currently reported evidence: its published results show improvement from 20% to 50% on SWE-bench Verified and from 14.2% to 30.7% on full Polyglot after 80 iterations, with ablations suggesting that both open-ended exploration and self-improvement contribute. Finally, we provide MetaAI-Mini, a reproducible HumanEval-based protocol and codebase. Because no completed model run is included in this build, MetaAI-Mini is reported as a protocol rather than as an experimental result. 

---
# Next-Token Prediction Learns Generalisable Representations of Sleep Physiology 

**Authors**: Jonathan F. Carter, Lionel Tarassenko  

**Link**: [PDF](https://arxiv.org/pdf/2606.09605)  

**Abstract**: Foundation models offer a promising route to compress multi-modal physiological signals into compact representations of human health, with broad applications across sleep medicine, cardiology, neurology and other healthcare domains. Existing models have typically been trained with masked-reconstruction or contrastive objectives. However, masked reconstruction may be poorly suited to the stochastic nature of these signals, while contrastive approaches rely on positive-pair definitions despite the semantic invariances of physiological signals being poorly understood. In this work, we show that next-token prediction is a simple and scalable alternative. We develop Hypnos, a multi-modal sleep foundation model trained using eight different sensing modalities (e.g. EEG, ECG, respiratory signals) drawn from over 20,000 overnight polysomnography recordings. We tokenize each modality into streams of discrete tokens using residual vector quantization, then train a large auto-regressive RQ-Transformer to jointly predict the next token across all modalities in parallel. After training, Hypnos can be applied to continuous streams of sensor data from any subset of supported modalities, generating embeddings for downstream tasks. Across a range of benchmarks, Hypnos significantly outperforms existing foundation models. In sleep stage classification, we match the performance of strong supervised baselines on held-out test sets whilst using \(100\times\) less labelled data. Hypnos even generalises to daytime physiology, surpassing a dedicated ECG foundation model at detecting atrial fibrillation. Our results demonstrate that next-token prediction is a strong self-supervised objective for representation learning from multi-modal physiological signals. 

---
# Optical Reasoning: Rethinking Images as an Expressive Reasoning Medium Beyond Text 

**Authors**: Yutong Bian, Dongjie Cheng, Heming Xia, Yongqi Li, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.09585)  

**Abstract**: Chain-of-Thought (CoT) improves the performance of Large Language Models (LLMs) and has been extended to Multimodal Large Language Models (MLLMs). More recent work further moves from text-based multimodal reasoning toward interleaved-modal reasoning, where intermediate steps can incorporate both textual rationales and visual evidence. In this work, we propose a bolder and more ambitious idea: could images alone serve as the reasoning medium for both language and multimodal tasks? To explore this, we propose optical reasoning, which treats images as a standalone reasoning medium. We instantiate this concept with two variants: typographic-based optical reasoning, which optimizes visual layouts for compact rationale rendering, and graphical-based optical reasoning, which composes text and graphical elements into structured visual rationales. Across mathematical, scientific, and interleaved-modal reasoning benchmarks, optical reasoning can match or even exceed traditional text reasoning while reducing reasoning tokens by an average of 28.57% on language tasks and 16% on multimodal tasks, achieving 1.96 times the token efficiency of text reasoning. These results show that images can effectively and efficiently encode rationales while providing a unified visual canvas for reasoning. 

---
# TABVERSE: Benchmarking Cross-Format Table Understanding in LLMs and VLMs 

**Authors**: Momina Ahsan, Sarfraz Ahmad, Ming Shan Hee, Roy Ka-Wei Lee, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2606.09578)  

**Abstract**: Large Language Models (LLMs) and Vision-Language Models (VLMs) are increasingly evaluated on table reasoning tasks, but the role of table representation remains under-explored. In practice, the same table content may appear in different structural formats, such as HTML, Markdown, and LaTeX, or as rendered images. However, existing evaluations often let content, format, layout, and modality vary together, making it difficult to isolate representation effects. We introduce TABVERSE, a controlled multimodal table benchmark that aligns the same table content across multiple structural formats and rendered images, with question category and difficulty tags. This design enables systematic evaluation of representation effects while holding table content fixed. We evaluate LLMs and VLMs across three tasks: Question Answering (QA), Structural Understanding Capability (SUC), and Structure Reconstruction (SR). Our results show that representation choice substantially affects table understanding. Models generally perform better with structured text than with rendered images, but the size of this gap depends on the task, model, and format. HTML is often the most robust text format, while row-sensitive structural tasks and syntactically usable LaTeX reconstruction remain challenging. These findings show that table representation is a key factor in reliable table evaluation. 

---
# Self-Explainability in Self-Adaptive and Self-Organising Systems: Status and Research Directions 

**Authors**: Tom Beyer, Svea Wisy, Sven Tomforde  

**Link**: [PDF](https://arxiv.org/pdf/2606.09568)  

**Abstract**: The growing complexity of self-adaptive and self-organising systems, fuelled by advances in Artificial Intelligence (AI), has made them increasingly difficult to understand and trust. While Explainable AI aims to provide insight into AI decision-making, a more advanced goal is for systems to explain themselves - an ability referred to as Self-Explainability (SX). This article presents a systematic literature review on SX, analysing existing approaches, including their domains, targets, and evaluation methods. The review develops a unified definition and taxonomy of SX and introduces Levels of Self-Explainability, providing a framework for positioning current and future research. Our results show that most SX approaches remain conceptual, with few practical implementations. Moreover, there is currently no formal or de facto standard for evaluating SX, highlighting a major research gap. This work thus establishes a foundation and roadmap for advancing Self-Explainability in complex systems. 

---
# PRISM: Recovering Instruction Sets from Language Model Activations 

**Authors**: Gilad Gressel, Rahul Pankajakshan, Julia Diament, Efim Hudis, Krishnashree Achuthan, Yisroel Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2606.09563)  

**Abstract**: As LLMs are deployed as agents, reliable monitoring requires knowing not only what they output, but which instructions are steering their behavior. This is difficult when models infer unintended subgoals, follow contextual cues, or are influenced by prompt injections and hidden objectives. While activation-to-language methods suggest that hidden states can reveal natural-language information, existing approaches are not designed to recover the full set of simultaneous instructions, constraints, prohibitions, and subgoals active in agentic settings. We formalize this problem as instruction set retrieval and introduce PRISM, an activation-conditioned interpreter that decodes hidden states from a frozen target model into a faithful bullet list of active instructions. Unlike prior activation-to-language methods, PRISM is trained to recover instruction sets directly, using judge-guided GRPO to reward covered instructions and penalize unsupported ones. Across benign, constrained, prompt-injection, and hidden-objective settings, PRISM outperforms activation-to-language baselines, especially on security-relevant objectives. 

---
# AI Scientists Are Only as Good as Their Evidence: A Stratified Ablation of Proprietary Data and Reasoning Skills in Drug-Asset Valuation 

**Authors**: Yinan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09556)  

**Abstract**: AI Scientist agents are often evaluated as if capability were mainly a function of model quality, prompting, or reasoning scaffolds. We test a different hypothesis in drug-asset valuation: for knowledge-intensive scientific decisions, the limiting factor is often the evidence substrate the agent can access. We run a controlled three-arm ablation on a production valuation agent: A is a plain web-only LLM analyst, B adds public structured tools plus a 14-dimension valuation playbook, verifier, objectivity policy and red-team, and C adds the proprietary Noah AI corpus of curated pipeline, trial and deal intelligence. Across a 13-asset stratified benchmark, B improves calibration and audit discipline: tier-in-range accuracy rises from 0.80 to 0.89 and objectivity from 3.16 to 3.30. But B does not remove the factual ceiling. Under capability-superset accounting, A and B recover only 0.25 and 0.38 of the curated gold competitive record, while C recovers 0.96; on the curated long-tail subset, C reaches 0.93 vs. 0.26/0.30. Raw blind-panel decision quality is similar for A and B (7.01 vs. 6.96), so we introduce completeness-aware decision utility: informed decision-quality = decision-quality x gold-coverage. On this metric, C reaches 7.43 vs. 1.76/2.57 for A/B. Even a perfect non-proprietary-data report would be capped at 3.83 by B's coverage. The result is not that reasoning scaffolds are unimportant; they improve calibration and discipline. Rather, proprietary evidence sets the upper bound of what the AI Scientist can know and therefore decide. 

---
# From Rigid to Dynamic: Entropy-Guided Adaptive Inference for Long-Context LLMs 

**Authors**: Zhanchao Xu, Haoyang Li, Qingfa Xiao, Fei Teng, Chen Jason Zhang, Lei Chen, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.09508)  

**Abstract**: Existing sparse attention and KV cache compression methods for long-context LLM inference typically apply fixed sparsity patterns or uniform budgets across all attention heads, overlooking the substantial variation in attention behavior among heads and contexts. We observe two distinct entropy patterns among attention heads: Rigid Heads, whose entropy stays near zero across input segments, and Dynamic Heads, whose entropy fluctuates significantly. Crucially, the distribution of these types is context-dependent and cannot be predetermined offline. We therefore propose EntropyInfer, a training-free framework that uses attention entropy to adaptively allocate compute at the granularity of individual heads and segments during prefilling. For decoding, we introduce a latent KV cache compression scheme that leverages generated output tokens, rather than prefill tokens alone, to identify and retain the most critical cache entries. Extensive experiments on Llama, Qwen and openPangu model series show that EntropyInfer consistently outperforms baselines including SnapKV, AdaKV, and CritiPrefill, achieving up to 2.39$\times$ end-to-end speedup beyond 100k tokens with minimal quality degradation compared to full attention. The code is released in this https URL. 

---
# Deterministic Integrity Gates for LLM-Assisted Clinical Manuscript Preparation: An Auditable Biomedical Informatics Architecture 

**Authors**: Yoojin Nam, Jinhoon Jeong, Namkug Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.09500)  

**Abstract**: Objective. Large language models (LLMs) increasingly draft clinical research manuscripts, but their fluency can hide fabricated citations, numbers that drift from source tables, and unmet reporting-guideline items. Existing tools generate text without verifying it, and self-critique inherits the blind spots that produce confident fabrication. We describe an architecture that pairs generation with verification. Methods. The design rests on three principles: decompose the workflow into self-contained skills, gate every stage transition with halt-on-failure, and resolve each integrity question with the cheapest sufficient mechanism -- a deterministic, re-executable check where one suffices, and a prose-level probe only where interpretation is unavoidable. This determinism-where-possible split, organized as an integrity-gate taxonomy, is the core contribution. It is realized as MedSci Skills, an open-source toolkit of 43 skills coordinated by one orchestrator, whose deterministic tier comprises 21 standard-library detectors. We evaluate it on three reproducible public-dataset pipelines (STARD, PRISMA, STROBE) and a seeded-defect ablation. Results. Across the three pipelines every content-hash manifest verified clean and the gates surfaced real defects. On 27 identical injected defects the deterministic gates detected all 27 with no false positives on the matched clean fixtures, whereas a generic single-prompt LLM reviewer detected 11, its misses concentrated in generated-code, bibliography-internal, and style defects the prose does not expose. Conclusion. Determinism-where-possible verification yields an auditable, re-executable trail that exposes the evidence a human needs to check an LLM-assisted manuscript -- feasibility and reproducibility evidence, not a claim of human-competitive quality, which a separate blinded study addresses. MedSci Skills is MIT-licensed and archived (v3.8.0). 

---
# LLM-Orchestrated Conformance Checking in Stroke Care Without Computer-Interpretable Guidelines 

**Authors**: Giorgio Leonardi, Stefania Montani, Manuel Striani, Alessandro Canessa, Delfina Ferrandi  

**Link**: [PDF](https://arxiv.org/pdf/2606.09489)  

**Abstract**: Objective: Conformance checking in healthcare seeks to assess whether patient care pathways adhere to clinical guidelines. However, its practical application often depends on the availability of formal, machine-interpretable representations of guidelines, such as Computer-Interpretable Guidelines (CIGs), which are seldom available in real-world clinical settings.
Methods: This work introduces a modular framework based on the orchestration of Large Language Models (LLMs) to support medical conformance checking directly from unstructured clinical and guideline texts, without requiring predefined CIGs. The proposed architecture integrates multiple LLMs and supporting components to extract patient traces from clinical discharge letters, identify normative rules from textual clinical guidelines, translate these rules into executable scripts, and compute a Trace Conformance Indicator to quantify compliance within the event log.
Results: The framework was implemented and evaluated in the stroke care domain at the neurological ward of Alessandria Hospital. Hundreds of patient traces were automatically extracted from hospital data and assessed against 50 rules derived from the reference guideline. The analysis showed that more than 86\% of the available traces were conformant.
Conclusion: The results demonstrate the feasibility of using orchestrated LLMs for practical healthcare conformance analysis. At the same time, the study provides evidence of a high level of adherence to stroke care guidelines at Alessandria Hospital. 

---
# Emergent alignment and the projectability of ethical personas 

**Authors**: Guillermo Del Pinal, Youngchan Lee, Cameron McNamara, Alejandro Perez Carballo  

**Link**: [PDF](https://arxiv.org/pdf/2606.09475)  

**Abstract**: Work on `emergent misalignment' shows that finetuning LLMs on narrow tasks can induce broadly misaligned behavior. This supports the `persona selection' (PSM) hypothesis: during pre-training, LLMs learn to simulate different characters and perspectives, which can be elicited and refined during post-training. This paper investigates the converse phenomenon, `emergent alignment', and uses it to support and refine the PSM and motivate a novel desideratum for alignment. We finetune a helpful-only model on broad and narrow safety tasks. To create SFT samples, we follow the `Constitutional AI' (CAI) approach and use four constitutions which encode reasonable alignment strategies: deontology, consequentialism, virtue ethics, and aligning AIs as subordinate to human authority. For each of those models, we show that finetuning on two narrow safety sub-categories reliably induces emergent alignment over a representative set of general safety categories, and on safety subcategories that we directly filtered-out of the data sets used for narrow alignment. To test the `PSM' using a more fine-grained evaluation, we used a multidimensional `ethical persona' diagnostic. For each constitutionally finetuned (broad/narrow) model, we evaluate how well their behavior matches their expected signature profile. Our results show that our CAI models acquire their expected ``ethical persona'' -- e.g., the model narrowly fine-tuned on SFT samples created using the consequentialist constitution agrees significantly more with utilitarian than deontological beliefs. Yet our coarse and fine-grained evaluations show that there are significant differences across our (broad/narrow) finetuned CAI models in how well they project. We conclude that alignment strategies should be evaluated, not just on their (in-distribution) general safety performance, but also specifically on their degree of projectability. 

---
# TheoremBench: Evaluating LLMs on Theorem Proving in Formal Mathematics 

**Authors**: QuocViet Pham, Elvir Karimov, Andrey Galichin, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2606.09450)  

**Abstract**: LLMs have recently achieved strong results on formal proving benchmarks. However, existing evaluations remain heavily concentrated on competition-style problems and often fail to capture how models behave on longer, more dependency-rich mathematical developments. We introduce TheoremBench, a Lean4 benchmark designed to evaluate theorem provers beyond contest settings. The benchmark is built from nearly one hundred classical theorems and is released in two complementary forms: a plain main version containing one target theorem per instance, and a premised version that expands each theorem into a structured family of related proving tasks consisting of the main theorem together with automatically extracted supporting subtheorems. This design enables evaluation of not only whether the final theorem was proved from scratch, but also of partial progress through the internal proof structure of a theorem. Our experiments show that explicit premises substantially improve performance for Lean4-capable prover models. To provide a comprehensive evaluation, we introduce theorem-level coverage and token-efficiency metrics that expose qualitative differences in proof behavior. The results show that current provers remain strongly biased toward easy subtheorems and often solve theorems through long and inefficient tactic traces rather than compact proof plans. TheoremBench therefore provides a more fine-grained view of formal reasoning ability and highlights the importance of structural benchmark design for evaluating Lean4 theorem provers. 

---
# AliyunConsoleAgent: Training Web Agents in Real-World Cloud Environments via Distillation and Reinforcement Learning 

**Authors**: Bojie Rong, Zheyu Shen, Qiaoping Wang, Pengfei Kang, Yang Xu, Yawen Wei, Hanyu Wu, Zhi Zhao, Leihao Pei, Linquan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09447)  

**Abstract**: We present AliyunConsoleAgent, a web agent framework for automated documentation verification in real-world cloud consoles. Major cloud platforms encompass hundreds of products with rapid feature iteration, causing console UIs to frequently diverge from their corresponding documentation. Verifying that documented procedures accurately reflect the current console and can be executed end-to-end demands an estimated 4 million recurring inspections annually, yet manual coverage remains below 1%. While agent systems built on frontier proprietary models achieve high success rates, their prohibitive cost and data privacy constraints preclude large-scale deployment. We propose a two-stage training paradigm: supervised fine-tuning (SFT) on distilled frontier-model trajectories, followed by reinforcement learning using Group Relative Policy Optimization (GRPO) and a dual-channel outcome reward model in real cloud environments. To support large-scale RL training, we construct a high-determinism rollout system featuring Terraform-based resource pre-provisioning and LLM-driven on-demand provisioning, which effectively isolates environment noise from the training signal. We further introduce a rule-based reward evaluation protocol grounded in backend audit logs, providing objective, reward-hacking-resistant outcome judgment. Our model evolves from mechanical instruction following to autonomous decision-making with cloud console and product-specific understanding. Experiments on a challenging 278-task benchmark where the best frontier model achieves only 65.34% demonstrate that AliyunConsoleAgent-32B achieves a 63.52% mean success rate -- a 20.24 percentage-point improvement over the base model, narrowing the gap to the best frontier proprietary model to 1.82 pp (bootstrap 95% CI [-1.27, 7.39]) -- at 92% lower inference cost. 

---
# SIFT: Selective-Index For Fast Compute of RAG Prefill by Exploiting Attention Invariance 

**Authors**: Rya Sanovar, Srikant Bharadwaj, Hritvik Taneja, Moinuddin Qureshi  

**Link**: [PDF](https://arxiv.org/pdf/2606.09441)  

**Abstract**: Retrieval-Augmented Generation (RAG) injects LLM queries with relevant documents to improve response quality. This injection increases prompt length and slows time to first token (TTFT). Unlike standard queries, RAG queries have a unique property of context reuse where the same documents recur across user queries. Thus, fully recomputing documents for every RAG query does redundant compute and increases TTFT. Prior works precompute KV tensors of RAG documents offline and coarsely recompute some tokens during online prefill. However, such KV reuse is often slower than full recomputation on modern GPUs due to high-latency disk transfers. Further, such a coarse-grained recomputation degrades accuracy.
To address these limitations, this paper proposes SIFT: Selective-Index For Fast Compute of RAG Prefill by Exploiting Attention Invariance. SIFT processes documents offline and extracts fine-grained locations of high attention scores for each document. Next, we identify the following attention invariance insights that enable us to exploit the extracted locations during runtime: (1) Local-Attention Invariance: The location of high attention scores within a document remain invariant to surrounding documents. This helps us predict the location of high scores where the document attends to itself. (2) Cross-Attention Consistency: Keys with high intra-document attention also attract cross-attention from subsequent documents. This helps us predict the location of high scores where the document attends to future documents. Critically, SIFT stores no KV data and only stores locations of high scores in the form of two compact bit vectors. SIFT's storage is up to 24,000x smaller than KV tensors, obviating costly disk transfers. During prefill, SIFT computes the attention only for the marked locations and improves TTFT by 1.71x while holding accuracy within 1% of full recompute. 

---
# Bayesian Selective Latent Inference for Wastewater-First Influenza Monitoring 

**Authors**: Yixuan Zhang, Yang Song, Hao Wang, Samir Bhatt, Hengguan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09433)  

**Abstract**: Wastewater influenza surveillance can reveal community circulation before clinical reporting, but wastewater alone is not a fully identifiable proxy for human burden. Existing wastewater models assume a fixed evidence set, while generic evidence-acquisition methods treat official surveillance streams as interchangeable costly features. We cast wastewater-first influenza monitoring as a selective decision problem: starting from mandatory wastewater evidence, the system must decide whether wastewater is sufficient, which delayed official stream to query next, and when abstention is the only scientifically defensible action under source ambiguity. We propose Bayesian Selective Latent Inference (BSLI), a principled Bayesian method that maintains a posterior over latent burden and identifiability, certifies answerability through explicit scientific gates, and optimizes query-stop decisions with an exact cost-calibrated Bellman policy. We prove the key variational, answerability, Bellman-optimality, and one-dimensional cost-calibration properties. On a fixed public-data benchmark with 5,933 forecasting episodes and 3,102 source-ambiguity episodes, BSLI improves the matched-budget cost-performance frontier while preserving conservative abstention under source ambiguity. 

---
# WeaveBench: A Long-Horizon, Real-World Benchmark for Computer-Use Agents with Hybrid Interfaces 

**Authors**: Wanli Li, Bowen Zhou, Yunyao Yu, Zhou Xu, Yifan Yang, Dongsheng Li, Caihua Shan  

**Link**: [PDF](https://arxiv.org/pdf/2606.09426)  

**Abstract**: Computer-use agents (CUAs) increasingly operate in runtimes that combine visual desktop control, command-line execution, code editing, browsers, and external tools. Existing benchmarks, however, often evaluate these interfaces as separable capabilities, leaving long-horizon cross-interface orchestration under-tested. Thus, we introduce WeaveBench, a long-horizon hybrid-interface benchmark with 114 tasks across 8 real-world work domains, grounded in real user requests and publicly verifiable artifacts. Each task requires agents to combine GUI observations/actions with CLI/code operations within a single trajectory. We evaluate these tasks on a real Ubuntu desktop inside deployed CLI-agent runtimes, augmented with a minimal desktop-control plugin. We also propose a companion trajectory-aware judge that inspects deliverables, files, screenshots, logs, and action traces, while detecting shortcut behaviors such as fabricated visual evidence or hard-coded metrics. Across frontier model-runtime pairings, the best PassRate reaches only 41.2%, showing the benchmark remains far from saturated. The trajectory-aware judge further reveals that outcome-only grading substantially overestimates agent performance. Overall, WeaveBench exposes a critical gap in CUA evaluation and provides an effective testbed to measure whether agents can orchestrate GUI, CLI, and code operations across long-horizon real-world tasks. 

---
# Capacity, Not Format: Rethinking Structured Reasoning Failures 

**Authors**: Hengxin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2606.09410)  

**Abstract**: Prior work treats structured output as a reasoning tax, but this framing is incomplete: the cost of formatting depends strongly on a model's spare capacity. Using information-matched prose controls and a four-level schema complexity gradient, we separate format-specific effects from prompt-length confounds across 4 models and 5 benchmarks with 0% parse failures on successfully generated responses.
We find that structured formats are capacity-dependent. Models with sufficient headroom absorb JSON constraints without degradation (Sonnet: $88.7\pm4.0$% JSON vs. $89.3\pm1.7$% CoT on MATH-Hard). In contrast, formats severely degrade models operating near their limits through two distinct mechanisms. First, under standard token budgets, Haiku drops 36.2pp ($p < 0.0001$) largely due to truncation. Second, even with extended budgets eliminating truncation, GPT-4o-mini drops 28.0pp ($p < 0.001$), revealing pure capacity competition independent of token exhaustion.
This format penalty scales with schema complexity (McNemar $p < 0.0001$) and cannot be explained by prompt length alone. Furthermore, these results qualify claims of frontier model immunity: on AIME competition math, Opus 4.7 drops from 96.2% to 91.0% under JSON ($-5.3$pp; the displayed percentages are independently rounded, exact difference is $7/133 = 5.26$pp $\approx 5.3$pp). A delayed-structure ablation -- reasoning freely before formatting -- recovers most of the lost accuracy (3-run mean: 80--87%), supporting the capacity competition mechanism. The practical implication is not to avoid structured output, but to match it to capacity: when a model is near its limits, think first, format later. 

---
# Correct Looks Better: Pairwise Comparisons Reveal Accuracy Rankings 

**Authors**: Mina Remeli, Moritz Hardt  

**Link**: [PDF](https://arxiv.org/pdf/2606.09409)  

**Abstract**: Pairwise comparisons combined with aggregation methods like Elo have become central to evaluating generative models, yet concerns remain that they reward superficial stylistic cues or display judge biases. In a more positive turn, we show that model rankings from pairwise comparisons strongly agree with ground-truth-based accuracy rankings when such ground truth is available for comparison. By converting five well-known benchmarks into free-form generative evaluations, we find that Elo rankings achieve a Spearman correlation above 0.9 with accuracy rankings and substantially outperform direct evaluation when the judge is weak. Furthermore, style and judge bias have only minor effects on model rankings, despite most judgments occurring on pairs where both candidate answers are correct (or incorrect). On such pairs, we find that repetition after the final answer (echo) is a causal driver of judge preference. 

---
# RunAgent SuperBrowser: A Theory of Autonomous Web Navigation Grounded in Human Browsing Behaviour 

**Authors**: Radeen Mostafa, Sawradip Saha  

**Link**: [PDF](https://arxiv.org/pdf/2606.09399)  

**Abstract**: We present SUPERBROWSER, an autonomous web-navigation agent designed against a single guiding hypothesis: a web agent should browse the way a person browses. A human reading a page does not retain every pixel they have seen; they look at a few candidate targets, decide on one, and remember only what is needed to keep the goal alive. We operationalize this perception-cognition-action triad as three coupled mechanisms. First, a vision-first bounding-box pipeline labels candidate interactive regions on every screenshot and feeds them, asynchronously prefetched, to the language model so that the "eye" precedes the "hand". Second, a three-role brain -- an Orchestrator that classifies and routes, a Planner that evaluates progress every few steps, and a Worker that emits per-step actions -- separates strategic from operational reasoning. Third, a structured Ledger stores only what a person would: the goal, the last three actions, a small set of facts and dead-ends, and a handful of checkpoints; a six-phase eviction loop systematically discards stale screenshots, state blobs, and reasoning traces from the live context. Action execution is a three-tier click cascade (Chrome DevTools Protocol to Puppeteer to scripted) with humanized Bezier motion, plus a chevron-aware bounding-box snapper that resolves the "small arrow beside a large label" ambiguity. On the Mind2Web Hard benchmark (66 tasks), SUPERBROWSER attains 89.47% success, placing third overall and ahead of every published open/research browser-agent baseline by a large margin. We argue that the gain comes not from any single trick but from the consistent application of a cognitive contract throughout the system. 

---
# From Coarse to Fine: Managing Temporal Granularity in Spatio-Temporal Data for Fine-Grained Traffic Prediction 

**Authors**: Shuhao Li, Weidong Yang, Yue Cui, Zizhuo Xu, Lipeng Ma, Fan Zhang, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.09392)  

**Abstract**: Efficient acquisition, storage, and utilization of traffic data are critical challenges in spatio-temporal data management. Most traffic data systems collect and store observations at fixed, coarse-grained temporal intervals to reduce storage and computation costs. However, such coarse-grained data severely limits downstream applications that require predictions at a finer temporal granularity. Collecting and maintaining fine-grained traffic data across all locations and time periods would impose a substantial burden on database storage and preprocessing pipelines. To address this temporal granularity mismatch, we formulate a novel problem: predicting fine-grained future traffic using coarse-grained sampled data. We propose the Spatial-Temporal Refinement Predictor (STRP), a granularity-aware framework for spatio-temporal data systems. STRP integrates two components: Tree Convolution for efficient and interpretable spatial dependency modeling, and Inverse Dilated Convolution for progressive temporal extrapolation. STRP supports two practical prediction settings: window-based and duration-based, to handle different forms of granularity mismatch. Experiments on six benchmark datasets show that STRP significantly outperforms state-of-the-art baselines in both accuracy and efficiency. Our work offers a practical and interpretable approach to managing granularity mismatches in spatio-temporal traffic data systems. 

---
# Capability-Aligned Hierarchical Learning for Tool-Augmented LLMs 

**Authors**: Haotong Yang, Ting Long, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09371)  

**Abstract**: Tool learning enables LLMs to invoke external tools to accomplish tasks. Prior studies have demonstrated the effectiveness of a hierarchical structure: a high-level policy handles global planning and decomposes tasks into manageable sub-tasks, and a low-level policy focuses on invoking tools to solve these sub-tasks. However, these works typically optimize the high-level and low-level policies separately, leading to planner-executor misalignment and limiting LLM performance on tool-use tasks. In this paper, we propose a method called Capability-Aligned Hierarchical Learning (CAHL), which leverages RLVR to jointly optimize both policies, enabling better alignment between the high-level planner and the low-level executor. Experiments on constrained tool-use benchmarks (API-Bank and BFCL) and an open-ended environment (Bamboogle) demonstrate the effectiveness of CAHL. 

---
# Experience Makes Skillful: Enabling Generalizable Medical Agent Reasoning via Self-Evolving Skill Memory 

**Authors**: Haoran Sun, Wenjie Li, Yujie Zhang, Zekai Lin, Fanrui Zhang, Kaitao Chen, Xingqi He, Yichen Li, Mianxin Liu, Lei Liu, Yankai Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09365)  

**Abstract**: Medical agent systems are increasingly expected to support interactive clinical decision making rather than only static question answering. In such settings, effective agents must reuse prior experience across evolving cases, yet existing memory mechanisms often retain raw historical traces that are redundant, noisy, and difficult to govern. More importantly, they rarely distinguish which memories are truly useful for future reasoning. This limits their ability to accumulate compact and reliable experience for long-horizon clinical reasoning. To close this gap, we propose SkeMex, a post-deployment self-evolution framework that improves medical agents through a skill-based memory without updating model weights. SkeMex distills informative interaction trajectories into structured skills that encode reusable procedural knowledge, and organizes them into a multi-branch repository spanning general, task-specific, and action-level experience. To determine which memories should be reused and retained, SkeMex estimates context-dependent utility from environment feedback and uses it to guide value-aware retrieval and repository governance. A closed-loop ``Read--Write--Assess--Govern" lifecycle further supports continual evolution by writing new skills, updating utilities, promoting useful memories, and removing harmful entries. Experiments across diverse clinical tasks show that SkeMex consistently outperforms representative memory-based agents in both offline and online settings. It also generalizes across model backbones and supports transferable skill memory. All data and code will be released publicly. 

---
# Leveraging Structural Constraints for Diffusion-based Neural TSP Solvers 

**Authors**: Mickaël Basson, Philippe Preux  

**Link**: [PDF](https://arxiv.org/pdf/2606.09343)  

**Abstract**: Neural combinatorial optimization has recently achieved strong results on the Euclidean Traveling Salesman Problem (TSP) using generative models such as diffusion and consistency models. State-ofthe-art approaches like FT2T combine fast consistency-based prediction with gradient-based inference time refinement. However, gradient search often incurs significant computational overhead and may not align with the discrete structure of feasible solutions. We introduce Projected Consistency Inference (PCI), a plug-and-play, retraining-free alternative that replaces gradient refinement with structure-aware projections: PCI decodes valid Hamiltonian tours from the consistency model output and applies a lightweight local search (e.g., 2-opt). PCI achieves an average optimality gap (OG) of 0.17% on TSP with 500 cities, and 0.31% on TSP with 1000 cities, outperforming FT2T best settings (OG 0.22% and 0.36%, respectively) while reducing the inference time up to 30 to 40%. PCI also exhibits lower variance and memory usage, and can surpass classical heuristics such as LKH3 in rapid solution generation. Our results demonstrate that structure-aware inference time operations provide a practical and principled path for neural TSP solvers, complementing training time objectives. 

---
# TRL-Bench: Standardizing Cross-Paradigm Representation-Level Evaluation of Tabular Encoders 

**Authors**: Wei Pang, Xiangru Jian, Hehan Li, Zhixuan Yu, Alex Xue, Jinyang Li, Zhengyuan Dong, Xinjian Zhao, Hao Xu, Chao Zhang, Reynold Cheng, M. Tamer Özsu, Tianshu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09323)  

**Abstract**: Tabular encoders are usually evaluated inside task-specific end-to-end pipelines, so models from different training paradigms are difficult to compare directly even when they operate on similar tabular signals. We introduce TRL-Bench, a multi-granular tabular representation learning (TRL) benchmark that standardizes cross-paradigm representation-level evaluation: each encoder exports row-, column-, or table embeddings through its supported wrapper, and shared lightweight heads probe them across three suites: TRL-CTbench (column/table), TRL-Rbench (row), and TRL-DLTE (compositional Data-Lake Table Enrichment spanning all three granularities). To support this standardized setting, we release curated benchmark assets and task reformulations, including 50 OpenML tables with 123 verified targets, 16 row-pair linkage rewrites, and a 47,772-table DLTE lake derived from 1,379 parent tables. Across 20 models and 16 tasks, TRL-Bench shows that once downstream conditions are standardized, encoder quality is capability-specific rather than captured by a single leaderboard. In TRL-CTbench, generic text encoders often lead on tasks with strong surface-text signal, while tabular specialists win where their pretraining objective aligns with the task. In TRL-Rbench, within-table prediction and cross-table linkage favor different training regimes, with atomic linkage performance correlating strongly with the row-matching stage of DLTE pipelines. In TRL-DLTE, the strongest pipelines combine capability-matched specialists rather than reuse a single encoder, and top end-to-end quality depends on non-additive compositional fit rather than per-stage marginal rank alone. TRL-Bench provides a common protocol for measuring reusable signal in exported tabular representations under shared downstream conditions. Code and data: this https URL 

---
# Anything2Skill: Compiling External Knowledge into Reusable Skills for Agents 

**Authors**: Qianjun Pan, Yutao Yang, Junsong Li, Jie Zhou, Kai Chen, Xin Li, Qin Chen, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2606.09316)  

**Abstract**: Retrieval-augmented generation (RAG) enables agents to access external knowledge at inference time, but it primarily retrieves fragmented declarative evidence, leaving agents to repeatedly infer task procedures from passages, manuals, examples, logs, or trajectories. This raises a fundamental question: can skills extracted from external knowledge bases be installed into an agent, enabling it to rapidly approximate domain expertise? In this paper, we propose Anything2Skill, a taxonomy-guided framework that compiles heterogeneous external knowledge into reusable, retrievable, and executable skills for agents. Given a corpus of knowledge records, \textsc{Anything2Skill} first decomposes each record into evidence windows and performs plan-and-expand skill extraction under a skill-tree prior. The extracted candidates are then converted into structured skill contracts that specify invocation conditions, contraindications, action moves, workflow steps, constraints, output specifications, supporting evidence, and confidence scores. To construct a deployable procedural memory, Anything2Skill manages the extracted skills in a persistent SkillBank through taxonomy-aware compilation, registry-level reconciliation, lifecycle tracking, versioned updates, and visible skill-tree projection. At inference time, agents retrieve both task-specific passages from the original knowledge base and relevant procedural skills from the SkillBank, allowing RAG to provide declarative evidence while compiled skills provide reusable procedural guidance. Experiments on qsv and GitHub-CLI show that Anything2Skill combined with RAG achieves 98.85\% and 94.10\% success rates, respectively, substantially outperforming RAG-only agents. These results suggest that compiling latent procedural knowledge into explicit skills is an effective way to extend retrieval-augmented agents from knowledge access toward capability reuse. 

---
# FF-JEPA: Long-Horizon Planning in World Models with Latent Planners 

**Authors**: Sergi Masip, Jonathan Swinnen, Yutong Hu, Renaud Detry, Tinne Tuytelaars  

**Link**: [PDF](https://arxiv.org/pdf/2606.09311)  

**Abstract**: Joint Embedding Predictive Architectures (JEPAs) have shown promising world modeling capabilities, enabling planning in latent space by optimizing action trajectories using methods like the Cross-Entropy Method (CEM). These methods are, however, too computationally expensive and ineffective for long-horizon planning. Furthermore, these methods typically require an explicit image of the goal state, which is not always possible in real-world tasks. In this work, we tackle these limitations by proposing Forward-Forward-JEPA (FF-JEPA), a hierarchical approach leveraging two forward dynamics models. Alongside a standard action-conditioned forward model, we introduce an action-free latent planner that predicts the next subgoal given the current state. This approach removes the need for goal images and enables long-horizon planning by decomposing complex trajectories into a sequence of tractable, short-term optimization problems. Preliminary results on PushT demonstrate that FF-JEPA successfully overcomes flat world models' long-horizon collapse, highlighting this approach as a promising direction for goal-free planning. 

---
# MASS: Deep Research for Social Sciences with Memory-Augmented Social Simulation 

**Authors**: Yongrui Liu, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2606.09198)  

**Abstract**: Deep Research agents powered by Large Language Models (LLMs) have exhibited extraordinary potential in automated paper writing tasks. However, existing systems rely heavily on literature retrieval and synthesis through internet and local knowledge bases, often resulting research in lacking insight and creativity in social science. To address this issue, we propose "Memory-Augmented Social Simulation (MASS)", an innovative paradigm that leverages highly realistic and research-oriented social simulations to enhance the creativity and empirical founding of LLMs-generated research. Specifically, MASS integrates three core components: dynamic goal-path planning with multi-level social norm restraint to guide the simulation, a multi-disciplinary behavior dataset for agent memory cold-start, and a structured forgetting mechanism inspired by the Ebbinghaus curve. Together, these ensure simulation authenticity and provide a robust empirical foundation for generating innovative scholarly papers. Experimental results demonstrate the effectiveness of our method, showing a 6.81\% improvement in generation overall quality over foundation LLMs and 17.19\% gain in Insight over strong baselines. 

---
# IMUG-Bench: Benchmarking Unified Multimodal Models on Interleaved Understanding and Generation 

**Authors**: Lingyi Meng, Zecong Tang, Haoran Li, Tengju Ru, Zhejun Cui, Weitong Lian, Qi Kang, Hangshuo Cao, Yichen Zhu, Yechi Liu, Kaixuan Wang, Yu-Jie Yuan, Chunwei Wang, Yu Zhang, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2606.09169)  

**Abstract**: In recent years, unified multimodal models (UMMs) have emerged to support both understanding and generation within a single framework. Mastering dynamic, multi-turn interleaved image-text dialogues is a crucial task for UMMs in real-world applications. However, existing benchmarks fail to evaluate this important task, as they are often limited to single-turn or static settings, and typically overlook exposure bias in multi-turn interactions. To bridge this gap, we propose IMUG-Bench, a comprehensive benchmark for multi-turn interleaved image-text dialogue of UMMs that jointly evaluates their understanding and generation capabilities. Our IMUG-Bench comprises three classes: Static Spatial, Temporal Causal, and Hybrid, covering 3,113 samples and 12,034 interaction turns. It also includes dynamic understanding questions, thereby supporting evaluation that better reflects real-world multi-turn interaction scenarios. Large-scale experiments on IMUG-Bench systematically evaluate mainstream open-source and closed-source UMMs, revealing their capability boundaries and failure modes, and uncovering pronounced exposure bias on the generation side in multi-turn interactions. We further explore several test-time scaling strategies, including Chain-of-Thought, Self-Verification, and Best-of-N Sampling, which effectively improve generation accuracy and mitigate exposure bias in generation tasks. These findings provide insights into enhancing the robustness and multi-turn interaction capability of future UMMs. 

---
# Reliable to Expressive: A Curriculum for Rubric-Following Safety Judges 

**Authors**: Yongtaek Lim, Hyeji Choi, Minwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.09165)  

**Abstract**: Safety judges are increasingly deployed to evaluate model outputs against evolving criteria, yet recent meta-evaluation work shows they remain brittle under prompt and rubric variation, with false negative-rate swings of up to 0.24 reported for stylistic perturbations alone. We argue that safety judgment is fundamentally a rubric-following problem: a robust judge must apply the given evaluation criteria consistently across rubric formulations rather than memorize one specific template. We propose a training strategy that combines (i) instance-conditioned dynamic rubrics generated from prompt-response-label triples to expose the judge to the variability of evaluation criteria, and (ii) a reliable-to-expressive curriculum that begins with clean fixed-rubric supervision and progressively introduces noisier dynamic-rubric data. We evaluate on a single human-labeled set under three contrasting rubric prompts (HarmBench-style, ShieldGemma-style, and a domain-specific rubric). Our 12B curriculum judge achieves 94.12-94.88% accuracy across the three rubrics with a cross-rubric range of only 0.76, outperforming general-purpose LLMs, dedicated safety classifiers, and reasoning-oriented judges up to 30B in both peak accuracy and stability. An ablation shows that naively mixing dynamic rubrics into SFT increases cross rubric variance (1.44 -> 3.60); only the curriculum schedule recovers and improves on the fixed rubric baseline (variance 0.76). 

---
# Vision Language Model Helps Private Information De-Identification in Vision Data 

**Authors**: Tiejin Chen, Pingzhi Li, Kaixiong Zhou, Tianlong Chen, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.09132)  

**Abstract**: Visual Language Models (VLMs) have gained significant popularity due to their remarkable ability. While various methods exist to enhance privacy in text-based applications, privacy risks associated with visual inputs remain largely overlooked such as Protected Health Information (PHI) in medical images. To tackle this problem, two key tasks: accurately localizing sensitive text and processing it to ensure privacy protection should be performed. To address this issue, we introduce VisShield (Vision Privacy Shield), an end-to-end framework designed to enhance the privacy awareness of VLMs. Our framework consists of two key components: a specialized instruction-tuning dataset OPTIC (Optical Privacy Text Instruction Collection) and a tailored training methodology. The dataset provides diverse privacy-oriented prompts that guide VLMs to perform targeted Optical Character Recognition (OCR) for precise localization of sensitive text, while the training strategy ensures effective adaptation of VLMs to privacy-preserving tasks. Specifically, our approach ensures that VLMs recognize privacy-sensitive text and output precise bounding boxes for detected entities, allowing for effective masking of sensitive information. Extensive experiments demonstrate that our framework significantly outperforms existing approaches in handling private information, paving the way for privacy-preserving applications in vision-language models. Our dataset and code can be found here. 

---
# Late-Layer Fusion is Enough: Dual-Path Vision Token Routing for Multimodal Large Language Models under Visual Saturation 

**Authors**: Siyuan Liu, Jinyang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09131)  

**Abstract**: Multimodal large language models (MLLMs) commonly inherit the deep, symmetric Transformer backbone designed for unimodal text modeling, and apply the same computation uniformly to image and language tokens. This design overlooks a key modality asymmetry: image and text tokens differ substantially in information density, redundancy, and required reasoning depth. Through a layer-wise analysis of LLaVA-1.5, we observe that vision tokens tend to saturate in the middle layers. Specifically, text-to-image attention decreases from 0.68 at layer 0 to 0.07 by layer 4, and stabilizes near 0.04 after layer 18, whereas text tokens continue to benefit from deep semantic processing. These findings suggest a mismatch between architectural symmetry and depth-asynchronous modality evolution, resulting in redundant visual computation and possible drift in perceptual representations during deep task-specific adaptation. Motivated by this, we propose Dual-Path Vision Token Routing (DPVR), a modality-asymmetric routing framework for efficient MLLMs. Its core instantiation, DPVR-LF (Late-Layer Fusion), routes vision tokens at the saturation point into a one-layer trainable side branch, runs a thirteen-layer text-only forward that skips image positions in the deep stack, and re-fuses the visual and textual streams only at the final layer. With approximately 3% trainable parameters, DPVR-LF preserves competitive multimodal performance on standard benchmarks while reducing visual computation in the deep Transformer stack. The results challenge the conventional assumption that vision tokens must traverse all deep language-model layers, and indicate that a single late fusion layer can be sufficient for maintaining strong perceptual competence in LLaVA-style MLLMs. 

---
# A Regret Minimization Framework on Preference Learning in Large Language Models 

**Authors**: Suhwan Kim, Taehyun Cho, Geon-Hyeong Kim, Yu Jin Kim, Youngsoo Jang, Moontae Lee, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.09124)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has enabled progress on reasoning-intensive tasks by relying on task-specific verifiers that provide automated correctness signals. However, many realistic language tasks are difficult to equip with reliable verifiers, motivating a growing reliance on reinforcement learning from human feedback (RLHF). In this setting, we argue that a closer examination of how human feedback should be interpreted is essential. We introduce Regret-based Preference Optimization $(\textbf{RePO})$, which reframes RLHF through $\textit{regret minimization}$ rather than reward maximization. Human preferences are often shaped by $\textit{prospective}$ anticipation of outcomes and $\textit{counterfactual}$ comparisons to alternative behaviors, rather than by immediate, outcome-independent utility. $\textbf{RePO}$ captures this structure by modeling preferences as behavior-conditioned assessments of relative suboptimality. Experiments on mathematical reasoning benchmarks and human preference datasets demonstrate consistent performance gains, indicating that $\textbf{RePO}$ is an effective and human-aligned approach for training large language models. 

---
# ComplexConstraints and Beyond: Expert Rubrics for RLVR 

**Authors**: Sushant Mehta, Liudas Panavas, Edwin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.09118)  

**Abstract**: As LLM capabilities advance rapidly, the evaluation methods used to assess them increasingly lag behind. Traditional benchmarks relied on programmatic verification of narrow, surface-level constraints, but real-world instruction following and agentic tasks demand assessment of nuanced, context-dependent behaviors that resist simple scripted checks. We present a systematic analysis of expert-curated rubric-based evaluation as an alternative paradigm, drawing on empirical evidence from two domains: complex instruction following and enterprise agentic tasks. We first articulate five design principles for constructing high-quality rubrics, including Maximum Viable Atomicity, intent-aware criterion design, and iterative LLM-judge calibration. To validate these principles, we introduce ComplexConstraints, a new expert-curated instruction-following dataset in which each prompt is paired with 10-40 atomic rubric criteria. We demonstrate that these expert rubrics are not only better evaluation instruments but also highly effective training signals: training on approximately 1,000 ComplexConstraints examples yields +15.5% improvement for a 4B-parameter model and +12.2% for a 235B-parameter model on instruction following, while single-epoch RL training on a rubric-graded enterprise environment produces gains that transfer to out-of-distribution benchmarks the model was never trained on (+4.5% BFCL, +7.4% Tau2-Bench, +6.8% Tool-Decathlon). Our findings establish that expert-authored rubrics improve both the measurement and the development of frontier LLM capabilities, serving as effective evaluation and RL training signals. 

---
# Graph2Idea:Retrieval-Augmented Scientific Idea Generation with Graph-Structured Contexts 

**Authors**: Xu Li, Hanzhe Tu, Xun Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.09105)  

**Abstract**: Generating novel, feasible, and high-quality research ideas is an important yet challenging task in scientific this http URL Large Language Model (LLM)-based methods often ground idea generation with retrieved literature, but the retrieved evidence is usually provided as flat text, such as titles, abstracts, or summaries. Such flat contexts may contain redundant or weakly relevant information, while making cross-paper relations among problems, methods, mechanisms, and findings difficult to identify and this http URL address this challenge, we propose Graph2Idea, a knowledge graph-guided framework for retrieval-augmented scientific idea generation.Graph2Idea first retrieves papers according to the input topic, transforms them into structured knowledge triples, and dynamically constructs a target-centered knowledge graph to make literature relations this http URL then extracts compact graph-derived contexts that retain target-relevant relational evidence while reducing noisy textual this http URL on these contexts, a two-stage generation process first identifies promising research directions and then guides the LLM to synthesize candidate ideas from graph-grounded this http URL on a scientific idea generation benchmark show that Graph2Idea outperforms representative baselines under the automatic evaluation this http URL with the strongest baseline scores, it improves Novelty from 0.45 to 0.52, Quality from 0.24 to 0.29, and Feasibility from 0.22 to this http URL results suggest that graph-structured evidence helps LLMs generate research ideas through more explicit, compact, and traceable recombination of prior scientific knowledge. 

---
# DynaOD: Dynamic Origin-Destination Flow Generation with Discrete-to-Continuous Temporal Semantic Modeling 

**Authors**: Jie Zhao, Xianqi Dai, Jie Feng, Huandong Wang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.09086)  

**Abstract**: Dynamic origin-destination (OD) flow generation seeks to synthesize realistic mobility dynamics from temporal context alone, without relying on historical OD observations. A key challenge is to translate semantic temporal signals into temporally coherent OD patterns while preserving the inherent spatial heterogeneity of urban regions. We propose DynaOD, a semantic-driven framework that models temporal dynamics through two complementary perspectives: discrete directional trends that characterize qualitative shifts in urban activity patterns, and continuous temporal evolution that captures how such shifts unfold over time. By jointly encoding these temporal semantics, the framework constructs time-varying region representations that condition pretrained static OD generators in a lightweight and plug-and-play fashion. This modular design further supports scalable deployment and cross-city transferability. Extensive experiments on large-scale real-world datasets show that our method consistently outperforms representative baselines in both predictive accuracy and distributional fidelity. Code is publicly available at this https URL. 

---
# REFLECT: Intervention-Supported Error Attribution for Silent Failures in LLM Agent Traces 

**Authors**: Xiaofeng Lin, Yingxu Wang, Tung Sum Thomas Kwok, Daniel Guo, Sahil Arun Nale, Charles Fleming, Guang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.09071)  

**Abstract**: Large language model (LLM) agents now solve complex tasks through long plan-and-execution traces, yet the ability to locate errors in a completed traces still lags far behind, especially in the \emph{silent failure} regime. Existing approaches predict suspect steps via classifiers or LLM judges, or recover correct answers via retry, but none feed the intervention outcome back to \emph{refine the attribution itself}. We propose \methodname, a method that closes this gap by diagnosing a candidate error step, testing it through controlled replay with a diagnosis-specific patch, and using the verified outcome flip as contrastive evidence to refine the final attribution. Across four localization benchmarks spanning multi-hop reasoning across domains, \methodname achieves the highest localization accuracy among same-auditor methods across all four benchmarks, with the largest gains on structured tool-use traces, while providing actionable localization even when ground-truth answers are unavailable. 

---
# Agent Economics: An Entropy-Controlled Pluralistic Alignment Framework for Preventing Artificial Hivemind in Autonomous Agents 

**Authors**: Cheonsu Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2606.09039)  

**Abstract**: This study proposes the Behavioral Protocol Framework (BPF), an entropy-controlled pluralistic alignment framework designed to address two critical challenges in autonomous agent economies: the hivemind effect arising from excessive strategic convergence among agents and the lack of transparency in autonomous decision-making processes. The proposed BPF consists of three core modules: Mentalizing-based Social Intelligence (MbSI) grounded in Theory of Mind (ToM), Pluralistic Alignment (PA), and a Verifiable Execution Kernel (VEK). These modules are organically integrated within a closed-loop architecture that governs the entire lifecycle of agent behavior, from decision-making and execution to verification and feedback. To evaluate the proposed framework, a simulation environment implemented in Python and a Streamlit-based user interface will be developed. Through empirical experimentation, the study aims to examine whether the entropy-control mechanism of the PA module can effectively preserve strategic diversity among agents and mitigate collective convergence, while the VEK module provides a comprehensive and transparent audit trail of the decision-making process. The anticipated results are expected to demonstrate that the proposed framework can simultaneously enhance the stability, efficiency, and trustworthiness of autonomous agent economies. Consequently, this research offers a practical approach for developing robust, transparent, and accountable agent-native economic systems. 

---
# Personalization Meets Safety:Mechanisms,Risks,and Mitigations in Personalized LLMs 

**Authors**: Yanyan Luo, Xue Han, Ruiqiao Bai, Xin Huang, Yitong Wang, Qian Hu, Qing Wang, Chunxu Zhao, Jie Liu, Cong Geng, Lehao Xing, Pengwei Hu, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2606.09038)  

**Abstract**: Large Language Models (LLMs) have enabled increasingly personalized interactions by adapting to users' preferences, contexts, and long-term histories. However, the mechanisms that enable personalization also expand the safety landscape in ways not systematically addressed by existing literature. Existing reviews typically focus either on personalization or safety, leaving their intersection largely unexplored. We present the first comprehensive, safety-aware review of personalized LLMs. We organize personalization along three dimensions-user representation, personalization paradigm, and evaluation-and introduce a unified taxonomy of safety risks. At the representation level, we analyze risks arising from diverse user representations. Across mainstream personalization paradigms, we delineate vulnerabilities inherent to prompting, retrieval augmentation, parameter fine-tuning, reinforcement learning, Mixture-of-Experts (MoE), pruning, agent frameworks, and multimodal personalization, and synthesize mitigation strategies across the model lifecycle. Beyond these fine-grained risks, we characterize paradigm-agnostic safety risks arising from personalized adaptation. We further summarize personalized datasets and evaluation methodologies. Through a case study of OpenClaw, we analyze deployment trends in personalized agent ecosystems. Our analysis reveals three structural inadequacies in existing research: safety is evaluated as user-invariant rather than relational, personalization techniques are analyzed in isolation rather than in composition, and evaluation frameworks cannot capture emergent long-term risks. By jointly examining personalized representations, personalization paradigms, safety risks, defenses, and evaluation methods, we provide a unified framework for developing safe personalized LLMs and highlight key directions for future research. 

---
# A Multi-Agent System for IPMSM Design Optimization via an FEA-AI Hybrid Approach 

**Authors**: Jinseong Han, Sunwoong Yang, Namwoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09037)  

**Abstract**: Interior permanent magnet synchronous motor (IPMSM) design requires balancing conflicting objectives and multi-physics constraints, while modern optimization workflows face three bottlenecks: manual problem setup, high finite element analysis (FEA) cost, and unreliable surrogate-based search in sparse or out-of-distribution regions. To address these limitations, we propose an end-to-end automated IPMSM design optimization framework that integrates retrieval-augmented generation (RAG) for structured problem definition with an uncertainty-aware FEA-AI hybrid optimization pipeline. A Design agent, connected to a motor textbook through RAG, provides domain-knowledge-based options and engineering tips, and compiles an optimization card and a design-of-experiments plan for AI-model training. A Training agent automates electromagnetic FEA, records geometry-validation and solver-failure logs, analyzes failed geometries using ANOVA-based data analysis and LLM reasoning, and invokes a Design Sampling agent to redefine the design space and generate additional samples. An Optimization agent performs GA-based search with uncertainty-driven switching: low-uncertainty candidates are evaluated by AI-surrogate inference, whereas high-uncertainty and reliability-critical Pareto-front or top-K candidates are corrected by high-fidelity FEA and reused for iterative retraining. The framework converts manual, experience-dependent configuration into a reproducible workflow that balances computational cost and prediction reliability. Experimental results under a matched high-fidelity FEA budget show that the proposed hybrid approach achieves better objective performance while maintaining low and further reducible predictive uncertainty, outperforming FEA-only search, which is limited by early budget exhaustion, and AI-only search, which converges to a low-confidence optimum. 

---
# LATTEArena: An Evaluation Framework for LLM-powered Tabular Feature Engineering (Extended Version) 

**Authors**: Ankai Hao, Ke Chen, Huan Li, Lidan Shou  

**Link**: [PDF](https://arxiv.org/pdf/2606.09004)  

**Abstract**: Feature engineering remains essential for tabular data analysis, and Large Language Models (LLMs) have emerged as a promising paradigm for automating this process, giving rise to LLM-powered AuTomated Tabular feature Engineering (LATTE). However, the absence of standardized platforms prevents fair, cost-aware comparisons. Furthermore, complex methodological designs obscure the specific contributions of individual components; for example, although LFG integrates Tree-of-Thought, few-shot demonstrations, Monte Carlo Tree Search, and natural language generation, the isolated impact of each technique's competitive edge remains unquantified. To address these challenges, we introduce LATTEArena, the first competitive evaluation framework featuring: (1) a six-dimensional taxonomy decomposing 15 representative methods into reusable components; (2) a standardized modular arena for controlled comparison; (3) multi-dimensional assessments covering performance, cost, and robustness; and (4) component-level ablation quantifying each technique's competitive edge. Through extensive evaluations, we reveal 16 key findings, including: (1) Tree-of-Thought with Monte Carlo Tree Search achieves optimal cost-effectiveness; (2) RPN and Code output formats dominate classification and regression tasks, respectively. We publicly release the modular framework and over 4000 execution logs, enabling researchers to seamlessly pit new techniques against existing ones and advance LATTE. 

---
# The Token Not Taken: Sampling, State, and the Variability of AI Agent Outputs 

**Authors**: Muhammad Zia Hydari, Raja Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2606.08998)  

**Abstract**: Agentic AI systems can behave differently across runs: the same request may produce a different plan, a different tool call, a different code edit, or a different final answer. Such variability arises from several layers that are often conflated. A foundation model is a large pretrained model, usually adaptable to many downstream tasks, that maps an input context to predictions over outputs. In many current agents, that model is embedded in an orchestration loop that plans, calls tools, observes results, and updates state. One explicit intrinsic source of variability in such systems is token generation: the model computes scores over possible next tokens, the scores are converted into probabilities, and a decoder may sample tokens using a pseudo-random number generator. A small sampled token difference can then propagate upward into a different tool call, code path, search query, or agent state. Other sources of variability are extrinsic to token sampling, including changing environments, live data, serving infrastructure, batch effects, and numerical details. By separating these layers, the manuscript clarifies what it means to call agentic AI systems stochastic, when such variability can be reproduced under matched conditions, and why deterministic execution need not imply identical behavior in deployed settings. 

---
# Baichuan-M4: A Clinical-Grade Medical Agent System for Continuous Care 

**Authors**: Aiyuan Yang, Chengfeng Dou, Da Pan, Dian Wang, Fan Yang, Fei Deng, Fei Li, Guangwei Ai, Hui Liu, Hongda Zhang, Jinyang Tai, Kai Lu, Lijun Liu, Linwei Chen, Linyu Li, Meiqing Guo, Peidong Guo, Qiang Ju, Rihui Xin, Shuai Wang, XinKai Ma, Xudong Chen, Yichuan Mo, Canbin Piao, Leyi Pan, Yihe Luo, Zian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08982)  

**Abstract**: Baichuan-M4 is Baichuan Intelligence's clinical-grade medical large model, designed for \emph{continuous care} rather than single-turn medical question answering. It is built as a coordinated medical agent system around three pillars: \textbf{Baichuan-Harness}, a unified runtime that keeps reinforcement-learning training and real-world deployment consistent while enforcing action constraints, tool use, long-term patient memory, and multi-agent coordination; a \textbf{core reasoning model} trained with a continuous-care reinforcement-learning framework that integrates span-level reward modeling (SPAR++), reasoning-path compression, curriculum learning, and stabilized policy optimization; and a \textbf{clinical tool layer} for patient-memory management, authoritative evidence-based retrieval, and multimodal medical perception across documents, X-rays, and dermatology. On a cross-dimensional medical evaluation suite, Baichuan-M4 attains leading results in static medical knowledge and safety, dynamic OSCE-style consultation, long-context clinical memory, evidence-based retrieval, medical document OCR, and multimodal image understanding, while lowering the hallucination rate to 3.3\%. 

---
# RTL-BenchLS: A Large-Scale Benchmark for RTL Reasoning and Generation with Large Language Models 

**Authors**: Jing Wang, Shang Liu, Wenji Fang, Yuchao Wu, Yugao Zhu, Zhiyao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2606.08976)  

**Abstract**: LLM-based RTL generation and reasoning is a promising direction for hardware design automation. High-quality benchmarks are critical infrastructure for tracking progress in this direction. However, existing RTL benchmarks face inherent limitations in both scale and task scope. The designs they cover are typically small and simple, and the tasks focus almost entirely on specification-to-RTL generation. Frontier models' performance already saturates on the existing benchmarks. Scaling these benchmarks up is fundamentally difficult because aligned labels are required for benchmarking, such as specifications and testbenches. Such aligned high-quality data are rarely available for real-world designs. We introduce RTL-BenchLS, a large-scale benchmark addressing both limitations above. It contains over 10,000 formally verified Verilog designs, covering substantially larger and more complex designs than existing benchmarks. Beyond specification-to-RTL generation, we propose three novel tasks that jointly evaluate reasoning and generation: round-trip reasoning, masked-content reasoning, and repository-issue reasoning. The first two are self-supervised, which directly resolves the scaling bottleneck. All tasks are verified through formal equivalence checking without any manual testbenches. We evaluate eight LLMs on RTL-BenchLS. Even the best model reaches only 23% on natural-language round-trip reasoning, 28% on masked-content reasoning, and 12% on repository-issue fixing. RTL-BenchLS is substantially more challenging than existing benchmarks. It leaves ample room for future improvement and offers guidance for developing LLM-based methods for hardware design. 

---
# Diverse Thinking Schemata Elicit Better Reasoning in Large Language Models 

**Authors**: Xinyue Liang, Yizhe Yang, Yu Bai, Bin Xu, Jiawei Li, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.08974)  

**Abstract**: Large reasoning models (LRMs) have attracted increasing attention for their ability to solve complex mathematical problems by generating extended reasoning chains. In this work, we focus on two critical yet underexplored aspects of the reasoning process: reasoning transitions capturing the distinct transitions between reasoning steps and answer candidates reflecting the variety of solution paths produced by the model. We collectively define these two aspects as thinking schemata. We observe a correlation between the diversity of thinking schemata and model performance, which motivates us to enhance diversity as a means to further improve reasoning potential. To this end, we propose Diverse Schemata Policy Optimization (DiScO), a framework that first endows the model with schemata awareness, then encourages diversity through reinforcement learning, and further promotes diverse reasoning at inference time. Experiments on multiple mathematical reasoning benchmarks demonstrate that DiScO consistently outperforms standard group relative policy optimization. Beyond accuracy, human-annotated analyses show that DiScO substantially improves the model's ability to recover from erroneous initial attempts. Overall, our work suggests the important role that diversity of the thinking schemata plays and points to scaling along the diversity dimension as a promising research direction. 

---
# An Effective Router for Vision-Language Model Selection 

**Authors**: Can Wang, Shengwei Wang, Bolin Zhang, Zhiying Tu, Dianhui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08970)  

**Abstract**: Vision-language models (VLMs) with varying performance and resource requirements are widely deployed, making it difficult for users to select the most appropriate one among numerous VLM candidates. Existing work reveals the performance paradox phenomenon in language models and focuses on routing methods to solve it. However, developing a router for VLM selection is still a critical yet challenging problem, which primarily faces: 1) lack of specialized data, 2) ineffective feature representation, and 3) rigid model space and costly adaptation. In this paper, we construct a multimodal dataset for VLM selection, containing the outputs of seven mainstream VLMs on 32,626 unique image-text queries. We then propose ARMS, a router for VLM selection. ARMS enhances input signals with VLM profiles, employs a simple but effective architecture to improve representations of queries and VLM capabilities. To improve ARMS' adaptation to new VLMs, we propose two extension training strategies: incremental training and independent training. Experimental results on both in-distribution and out-of-distribution test sets demonstrate the effectiveness of ARMS. In particular, using our training strategy, ARMs (only 800M in size) can adapt to a broader VLM space and defeat commercial models like GPT-4o that are hundreds of times larger in scale. Our code, models, and datasets are available in the anonymous repository. 

---
# AlloSpatial: Agentic Harness Framework for Spatial Reasoning in Foundation Models 

**Authors**: Shouwei Ruan, Bin Wang, Zhenyu Wu, Qihui Zhu, Yuxiang Zhang, Jingzhi Li, Yubin Wang, Xingxing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.08952)  

**Abstract**: Multimodal Foundation Models (MFMs) have made substantial progress, yet remain fragile in spatial reasoning over the physical world. A key bottleneck lies in their inability to transform local egocentric observations into a global allocentric spatial representation. To address this, we propose AlloSpatial, an agentic framework for allocentric spatial cognition in foundation models. AlloSpatial introduces World2Mind, a plug-and-play cognitive mapping sandbox that converts egocentric observations into structured allocentric priors, including Allocentric-Spatial Trees and route maps that support querying object topology, geometric relations, passability, and trajectories. To utilize these priors reliably under noisy reconstruction and ambiguous visual evidence, AlloSpatial introduces a Spatial Reasoning Harness for tool-use judgment, modality-decoupled cue collection, and geometry-semantic arbitration. We further internalize this process in Qwen3-VL through cold-start reinforcement learning with a harness-gated trajectory-level reward. Experiments on VSI-Bench and MindCube show that AlloSpatial improves proprietary models by 5%-18% in a training-free setting, while ASTs alone support strong spatial reasoning even when visual inputs are removed. The trained AlloSpatial agents further outperform larger general-purpose models and competitive spatial baselines, suggesting that structured allocentric representations, active tool use, and verifiable reasoning offer a promising route toward spatially capable foundation models. 

---
# Oversight Has a Capacity: Calibrating Agent Guards to a Subjective, Fatiguing Human 

**Authors**: Emre Turan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08919)  

**Abstract**: As LLM agents begin to take real, irreversible actions (shell commands, file edits, deploys), the standard safety pattern is a human-in-the-loop approval gate: risky actions pause and wait for a person. We argue the gate is the easy part; the hard part is the judgment - which actions to stop - which the field evaluates against two false assumptions: that there is a ground-truth notion of "risky," and that the human reviewer is a perfect, infinitely-available oracle. On a hand-labeled set of 125 adversarially-weighted agent actions we show that (i) reviewers only moderately agree on what is risky (Fleiss' kappa = 0.52), so there is no single correct label; (ii) framing the guard as selective classification under asymmetric cost makes its operating limits measurable, and on hard inputs the guard cannot safely auto-decide; and (iii) when the reviewer is modeled as endogenous (fatiguing as escalation load grows), realized safety becomes an inverted-U in the escalation rate: more human oversight can make a system less safe, and the safety-optimal guard escalates below full escalation - a setting a load-aware policy also uses to resist a flooding attack that slips a malicious action past a fatigued reviewer. Agent oversight, framed this way, is not only a classification problem but a resource-allocation one: human attention is finite, and the guard's escalation policy spends it. We claim none of these mechanisms as novel - fatigue-aware learning-to-defer (FALCON), cost-sensitive deferral under workload constraints (DeCCaF), trajectory-level guarding, and reviewer-fatigue/flooding attacks are all prior art we cite. Our contribution is an open-source agent-oversight system that operationalizes and measures them in the LLM-agent action-gating setting, turning "is my guard good?" from a guess into a curve. The inverted-U and the flooding attack are modeling results that motivate a human study. 

---
# Order Matters: Unveiling the Hidden Impact of Macro Placement Sequences via Proxy-Guided LLM Evolution 

**Authors**: Shibing Mo, Jing Liu, Jianchu Xu, Ruilin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08904)  

**Abstract**: Macro placement is a fundamental step in modern chip physical design, playing a crucial role in determining the solution quality of high-dimensional combinatorial optimization problems. Despite recent advancements in machine learning for spatial coordinate determination, the temporal dimension of placement sequencing remains largely governed by static heuristics. In this work, we demonstrate that the placement sequence is not merely a preprocessing step but a decisive factor in optimization, where suboptimal early decisions trigger irreversible domino effects that constrain the solution space. To harness this unexplored dimension, we propose \textbf{OrderPlace}, a proxy-guided LLM evolution framework for automatically discovering macro placement order strategies. Instead of relying on manually crafted heuristics such as area- or connectivity-based ordering, OrderPlace explores a broader space of code-level policies, ranging from static scoring metrics to dynamic physics-inspired mechanisms. To mitigate the prohibitive cost of evaluating sequences, we introduce a lightweight proxy evaluation mechanism that efficiently filters candidates using a deterministic greedy probe. Experimental results on the standard ISPD 2005 benchmarks demonstrate that OrderPlace discovers novel ordering strategies. Compared with WireMask-EA and the state-of-the-art method EGPlace, OrderPlace reduces wirelength by 34.04\% and 14.08\%, respectively. 

---
# FAME: Forecastability-Aware Mixture of Experts for Heterogeneous Time Series Forecasting 

**Authors**: Qianyang Li, Xingjun Zhang, Shaoxun Wang, Tao Peng, Jia Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.08896)  

**Abstract**: Large-scale retail and industrial forecasting systems contain many heterogeneous time series whose lifecycle, sparsity, volatility, seasonality, spectral patterns, and contextual sensitivity differ substantially. A single forecasting model rarely performs well across all regimes, while dense ensembles increase inference cost and provide limited insight into expert suitability. This paper studies forecastability-aware expert routing: learning how data characteristics determine the suitability of forecasting experts. We propose \method{}, a sparse mixture-of-experts framework that represents each series with a multidimensional forecastability fingerprint, mines expert-suitability targets from validation performance, and trains a cost-aware sparse router to activate a small budgeted set of experts for each series. Using a production-scale vending-machine sales dataset from Shandong New Beiyang (SNBC), where the forecasting component has been integrated into the replenishment-planning pipeline, together with public retail benchmarks, we show that expert suitability varies systematically across data regimes. On the industrial dataset with 5,000+ machines and 60M+ transactions, \method{} Top-2 reduces MSE by 12.4\% over the strongest single expert, LightGBM, while executing 1.92 experts per series on average. The deployed component produces demand forecasts, while inventory-oriented gains are estimated by an offline replay simulator under a fixed replenishment policy rather than by online intervention. The framework turns heterogeneous sales forecasting from heuristic model selection into data mining of forecastability patterns and expert specialization. Code is available at this https URL 

---
# Can the Environment Speak for Itself? $T^{2}$-GRPO: A Turn-Trajectory Group Relative Policy Optimization for Caregiver Agents 

**Authors**: Yutong Song, Jiang Wu, Pengfei Zhang, Wenjun Huang, Honghui Xu, Nikil Dutt, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2606.08875)  

**Abstract**: Optimizing large language models (LLMs) for long-horizon caregiver agents requires balancing delayed task objectives with immediate environment dynamics, such as patient distress and resistance. In dementia care, this balance is especially difficult: trajectory level rewards are too sparse for turn level credit assignment, while external LLM-based evaluators are costly and can misread fragmented or indirect patient responses. To address this issue, we propose \textbf{T}urn-\textbf{T}rajectory \textbf{G}roup \textbf{R}elative \textbf{P}olicy \textbf{O}ptimization (\textbf{T$^{2}$-GRPO}), a framework that decouples caregiver RL into two normalized reward horizons and enforces safety through a binary hard veto. $T^2$-GRPO derives dense turn-level rewards directly from environment state transitions, measuring changes in patient distress and resistance from a frozen dementia patient simulator. These environment-grounded rewards are combined with trajectory-level evaluations through independent centered-rank normalization, which preserves heterogeneous reward signals and mitigates reward collapse. Extensive experiments on dementia caregivers show that T $^{2}$-GRPO outperforms competitive baselines, indicating a substantial improvement for emotionally sensitive caregiver scenarios that effectively handles immediate patient feedback, long-term care outcomes, and safety constraints. 

---
# Hybrid E-Assessment in Higher Education: Semi-Automated Grading of Paper-Based Written Examinations 

**Authors**: Hartwig Grabowski, Michael Canz  

**Link**: [PDF](https://arxiv.org/pdf/2606.08855)  

**Abstract**: This paper examines the limitations of fully digital and partially digital e-assessment approaches in summative examinations in higher education. The analysis focuses on the didactic narrowing caused by closed question formats and on organizational, technical, and legal constraints that become particularly relevant in large student cohorts. As an alternative, the paper proposes a hybrid e-assessment approach that retains paper-based, problem-oriented examination tasks while enabling semi-automated grading. Assessment-relevant intermediate results are encoded in a structured answer format, entered by students by hand, and subsequently captured from table fields. The central technical bottleneck is reliable recognition of handwritten characters under realistic examination conditions. Recent vision-capable large language models, combined with a two-pass validation principle and comparison against a solution key, can reduce misclassifications and thereby improve the validity, fairness, and scalability of summative assessment. 

---
# A Resilience-as-a-Service assessment framework for coordinated disruption response in interdependent urban transit systems 

**Authors**: Sara Jaber, S. M. Hassan Mahdavi, Neila Bhouri, Mostafa Ameli  

**Link**: [PDF](https://arxiv.org/pdf/2606.08849)  

**Abstract**: Urban public transport disruptions require rapid response strategies, yet existing studies rarely provide a decision support framework to compare alternative disruption response solutions using a common set of dynamic, passenger, operator, and environment oriented indicators. This paper proposes a KPI-driven, time-indexed framework to assess the resilience of disruption response solutions in urban transit systems. The framework combines an optimization model with a behavioral evaluation in agent-based simulation. It also underlays the secondary service degradation induced on helper lines when in-service vehicles are withdrawn to support the disrupted corridor. Rather than treating resilience as a single score, it evaluates complementary dimensions including vulnerability, adaptability, robustness, resilience loss, responsiveness, cost-based performance, emissions, and equity. The framework is implemented for the RER B transit line in the Ile-de-France (Paris) network. Results show that the coordinated strategy provides the most balanced resilience profile, combining high service continuity with lower total disruption cost than single mode alternatives, while also improving equity and maintaining competitive environmental performance. Sensitivity analysis further identifies the disruption conditions under which coordinated multimodal response is most valuable. 

---
# ZIPP:Zero-shot Image Personalization from Personas 

**Authors**: Harini SI, Somesh Singh, Yaman Kumar Singla, David Doermann, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2606.08841)  

**Abstract**: Text-to-image diffusion models are increasingly deployed in open-ended creative contexts, yet their outputs remain impersonal, optimized for aggregate aesthetics rather than individual taste. Human preferences are pluralistic: one user favoring muted, nostalgic portraits may prefer vibrant street photography, while another gravitates toward dreamy film aesthetics. Existing methods require dense interaction histories or per-user fine-tuning, failing in cold-start settings and collapsing context-dependent preferences into a static representation. We introduce zero-shot image personalization from personas (ZIPP), which conditions image generation on natural-language personas (concise descriptors of a user's identity and aesthetic sensibilities) without any user-specific data or weight updates. ZIPP uses an LLM to rewrite prompts from the perspective of a given persona, steering diffusion models toward personalized outputs. To mine personas at scale, we train an inductive Graph Attention Network over a 22M-user Reddit interaction graph with dual contrastive objectives aligning graph structure with visual behavior, then verbalize learned representations into natural-language personas via an MLLM. We introduce ZIPBench, the first zero-shot personalization benchmark with 1.5K users, graph-mined personas, and 40K generated images. Across four benchmarks and 14 LLMs spanning five model families, persona conditioning yields consistent gains (13-20%), with frontier models benefiting most. In the few-shot setting, ZIPP matches or exceeds fine-tuned baselines trained on 100+ examples per user. ZIPP achieves the lowest preference distributional divergence (CMMD 0.16 vs. 0.55), and IPF-normalized demographic evaluation shows it substantially reduces subpopulation bias present in existing methods. Human evaluation confirms a 79% win rate over generic generation and 58-65% over all fine-tuned baselines. 

---
# Beyond Pass Rate: A Multilingual, Execution-Grounded Evaluation of Open Code LLMs 

**Authors**: Sayed Erfan Arefin  

**Link**: [PDF](https://arxiv.org/pdf/2606.08840)  

**Abstract**: Code generation models are typically compared using compact execution benchmarks and aggregate pass rates, but such summaries obscure how performance varies across programming languages, problem families, and failure modes. We present a large-scale, execution-grounded evaluation of 9 openly accessible LLMs specialized for coding on 2,707 free LeetCode problems across 12 programming languages. Our corpus contains 325,343 problem-model-language jobs, each linked to prompt metadata, extracted code, LeetCode execution outcomes, and static-analysis signals. The results show that current open models remain far from the human acceptance reference: the best model, Yi-Coder-9B-Chat, reaches 23.64% mean correctness, compared with a 57.2% human acceptance baseline. Rankings are also slice-dependent: Qwen2.5-Coder-14B-Instruct is strongest on hard problems and distinct-problem coverage, while Gemma-2-27B-IT achieves the highest all-language lint pass rate. Failure analysis shows that compile errors account for 63.25% of non-accepted best submissions, indicating that many failures occur before semantic correctness can be tested. Static quality further diverges from functional correctness. Together, these findings show that multilingual, artifact-preserving evaluation reveals tradeoffs hidden by single-language or single-metric leaderboards. 

---
# Instrumental convergence and power-seeking 

**Authors**: David Thorstad  

**Link**: [PDF](https://arxiv.org/pdf/2606.08832)  

**Abstract**: Recent years have seen increasing concern that artificial intelligence may soon pose an existential risk to humanity. One leading ground for concern is that artificial agents may be power-seeking, aiming to acquire power and in the process disempowering humanity. I show how the argument from power-seeking rests on a strong version of a claim known as the instrumental convergence thesis. I explore leading defenses of the instrumental convergence thesis and argue that none establishes the thesis in a strong enough form to ground the argument from power-seeking. I discuss implications for longtermism, the governance of artificial intelligence, and the methodology of studying risks posed by artificial agents. 

---
# Inference-Time Conformal Reasoning with Valid Factuality Control for Large Language Models 

**Authors**: Ting Wang, Yuanjie Shi, Yan Yan, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08831)  

**Abstract**: Large language models (LLMs) increasingly perform multi-step reasoning, where intermediate claims form implicit directed acyclic graphs whose node correctness is structurally conditioned on their ancestors. This makes factuality uncertainty structural, rather than a trivial accumulation of node-wise errors, and necessitates inference-time uncertainty quantification over the reasoning structure. While conformal prediction (CP) offers flexible user-specified factuality control, existing work remains post-hoc and cannot intervene during generation. To fill the gap between CP's flexibility and its post-hoc limitation, we propose an \emph{Inference-Time Conformal Reasoning (ITCR)} framework that integrates CP directly into reasoning graph generation. ITCR learns a structure-level factuality uncertainty function that aggregates claim-level factuality signals over reasoning graphs without complex modeling assumptions. We then design the non-conformity score based on graph-level factuality uncertainty and calibrate the conformal threshold to decide when to stop generation. We theoretically show such generation is nested, yielding valid coverage guarantees for factuality control. Experiments over multiple datasets and coverage objectives demonstrate empirically valid coverage. In downstream reasoning tasks, inference-time calibrated graphs yield more accurate generation than post-hoc pruned graphs. 

---
# Momentum for Reasoning: Dense Intrinsic Signals in Policy Optimization 

**Authors**: Hao Chen, Zhanming Shen, Liyao Li, Yanyu Chen, Xuhang Zhu, Xiaomeng Hu, Qi Zhang, Ru Peng, Xiaoyu Shen, Haobo Wang, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.08815)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as a powerful paradigm for eliciting long-chain reasoning in large language models. However, existing methods based on Group Relative Policy Optimization (GRPO) rely on a binary outcome reward, which induces two structural failure modes: Zero-Advantage Collapse, in which all rollouts in a group share the same outcome and the gradient vanishes, and Hallucinated Certainty, in which the model becomes increasingly confident on incorrect rollouts late in training. We address both modes by densifying the reward with intrinsic signals computed entirely from the policy's own conditional probabilities, and propose ISPO (Intrinsic Signal Policy Optimization, which combines a sequence-level signal measuring how informative the thinking trajectory is for the final answer, with a token-level directional reward whose hallucinated-certainty hinge penalizes confidently-wrong predictions at critical decision tokens. Across three base models and five mathematical reasoning benchmarks, ISPO consistently outperforms competitive baselines, with the largest gains on the hardest benchmarks where zero-advantage collapse is most frequent, and training-dynamics diagnostics confirm that both failure modes are decreased. 

---
# STAR: Rethinking MoE Routing as Structure-Aware Subspace Learning 

**Authors**: Sumin Park, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.08814)  

**Abstract**: Mixture-of-Experts (MoE) scales model capacity efficiently by selectively routing inputs to a specialized subset of experts. However, input-expert specialization, the core motivation of MoE, critically depends on whether the router is actually aware of input structure. In practice, MoE routing is typically implemented as a shallow linear projection with limited awareness of input representation, which often leads to unstable routing. We propose STAR, a Structure Aware Routing that rethinks MoE routing as a subspace learning problem by augmenting standard learnable routing with an evolving principal subspace that tracks dominant input structure via Generalized Hebbian Algorithm (GHA). By aligning routing decisions directly with input structure, STAR enables stable expert specialization. We evaluate STAR on controlled synthetic setup and large-scale language and vision tasks, where it consistently improves routing quality and downstream performance over strong MoE baselines. Moreover, optional test-time subspace updates further enhance routing robustness and generalization under input distribution shifts. 

---
# Q-Delta: Beyond Key-Value Associative State Evolution 

**Authors**: Sumin Park, Seojin Kim, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.08804)  

**Abstract**: Linear attention reformulates sequence modeling as recurrent state evolution, enabling efficient linear-time inference. Under the key-value associative paradigm, existing approaches restrict the role of the query to the readout operation, decoupling it from state evolution. We show that query-conditioned state readout induces a structured value prediction over accumulated memory that complements key-based retrieval. Based on this insight, we propose Q-Delta, a query-aware delta rule that integrates mixed key-query prediction errors into state evolution, enabling jointly corrective dynamics while preserving delta-rule efficiency. We establish stability guarantees for the resulting dynamics and derive a hardware-efficient chunkwise-parallel formulation with a custom Triton implementation. Empirical results demonstrate stable optimization, competitive throughput, and consistent improvements over strong baselines on language modeling and long-context retrieval tasks. 

---
# Bridging Expert Knowledge and Automated Feature Engineering via Self-Evolution 

**Authors**: Varun Khurana, Vijval Ekbote, Vashu Chauhan, Yaman Kumar Singla, Rajiv Ratn Shah, Balaji Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2606.08800)  

**Abstract**: In high-stakes settings such as brand compliance, clinical care, and content moderation, machine learning cannot be deployed as opaque oracles: practitioners inspect the features driving model decisions, and models must leverage the expert documentation governing these domains. In practice, the data arrives as unstructured content, and features extracted from it must be interpretable, discriminative, and aligned with what experts consider important. Existing methods fall short: they target tabular inputs, lack demonstrated expert alignment, and cannot operationalize qualitative criteria such as 'maintain professional tone' into precise features. We present FEST (Feature Engineering with Self-evolving Trees), combining dual-stream feature generation (semantic and deterministic), semantic deduplication, and tree-guided iterative evolution to discover auditable features from raw text and images. FEST leads in 17 of 20 classifier-task combinations across brand classification, content authenticity detection, and stress detection, with a mean gain of 4.2 pp over the strongest baseline across five classifiers. An LLM-as-judge evaluation shows FEST achieves 60-80% coverage of expert-designed brand features at strict semantic-alignment thresholds, corroborated by a human expert study rating features highly on relevance, clarity, and actionability. When seeded with expert guidelines, FEST refines qualitative criteria into operational features, improving accuracy by 6-12 pp on average across brands. To enable systematic evaluation of expert alignment in automated feature engineering, we release BrandGuide, the first dataset pairing expert-designed features with 1M+ assets across 2,683 brands. By grounding feature engineering in expert knowledge, FEST opens a practical pathway for interpretable ML in domains demanding human oversight. 

---
# RAILS: Verification-Native Clearing For Agentic Commerce 

**Authors**: Adrian de Valois-Franklin, Alex Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08790)  

**Abstract**: Autonomous agents negotiate, purchase, deploy code, and move funds, but no neutral mechanism determines whether they
met their delegated obligation, who is responsible when they did not, or which settlement action follows. This is the
agentic clearing problem. Tool protocols (MCP), inter-agent communication (A2A), payment rails (x402), mandate and
network agent protocols (AP2, Visa, Mastercard), and settlement-risk standards each assume that determination and none
produce it.
Clearing is the missing primitive. Payment is not clearing. Authorization is not clearing. LLM-as-judge evaluation is
not clearing. Settlement-risk escrow is not clearing: it consumes clearing decisions.
RAILS (Real-Time Agent Integrity & Ledger Settlement) is the integrity and clearing layer for agentic commerce,
spanning a per-output reliability score, a published reliability record, and a clearing function that consumes them.
The clearing protocol at its core closes that gap. Seven primitives (Obligation Object, Evidence Envelope,
Verification Mesh, Clearing Decision, Settlement Instruction, Clearing Passport, Finality Rules), bound by a formal
model of admissibility-graded verification, together yield a soundness property: no financially material settlement is
supported by evidence below the obligation's admissibility floor. The property is falsifiable against the spec. We
are not aware of a prior agent-commerce verification mechanism that states a property of this kind. The approaches
nearest to it emit a pass, a delivery guarantee, a bare score, or an equilibrium.
This paper specifies that clearing protocol. 

---
# Structure-Conditioned Actor-Critic Branches for Quality-Diversity Reinforcement Learning 

**Authors**: Lianrong Zuo, Peilan Xu, Yong Liu, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2606.08735)  

**Abstract**: Quality-diversity reinforcement learning (QD-RL) aims to construct policy repertoires that contain both high-performing and behaviorally diverse policies. Existing QD-RL methods mainly diversify policy instances after rollout evaluation or use learned value information to improve policy quality and behavior targeting, while the learning branches that generate candidate policies remain less explored. This paper proposes SV-QD-RL, a structure-value coupled framework that represents each candidate as a structure-conditioned actor-critic branch. Each branch contains an actor, a structural mask, a branch-specific critic, a replay state, and evaluation attributes including behavior, return, sparsity, and value profile. The structural mask defines the actor subspace in which the branch learns, while the branch-specific critic and replay state shape its value-learning trajectory. A branch-aware QD archive then evaluates and retains branches according to behavioral quality, structural footprint, and value-profile information. Experiments on MuJoCo continuous-control tasks show that SV-QD-RL constructs policy repertoires with strong archive quality and behaviorally useful diversity. Ablation and diagnostic analyses further indicate that structural conditioning, critic differentiation, and memory-consistent refinement make complementary contributions to behavioral specialization. Schedule-aware repertoire evaluation shows that the learned archive provides selectable policy alternatives under changing behavior-level requirements. These results suggest that coupling actor structure with branch-specific value learning is an effective mechanism for generating diverse QD-RL policy repertoires. 

---
# Artificial Intelligence for Mathematical Reasoning: An Integrated Survey of Language Models, Neuro-symbolic Systems, and Verified Discovery 

**Authors**: Syed Rifat Raiyan, Mohsinul Kabir, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08728)  

**Abstract**: Mathematical reasoning has long served as a stringent test of machine intelligence; over the past decade, it has moved from a niche problem within NLP to one of the most consequential AI frontiers. This survey provides a unified account of the field's evolution, from early rule-based math word problem (MWP) solvers and template-driven geometry systems, through neural expression generation and LLM prompting, to contemporary reasoning models, multi-agent systems, neuro-symbolic theorem provers, and verified discovery workflows. We organize the landscape along four axes: (i) informal reasoning over text and diagrams, spanning MWP solving, multimodal geometry, and VLMs; (ii) formal reasoning in proof assistants, including autoformalization, tactic prediction, compiler-guided repair, and proof search; (iii) mathematical discovery, where systems propose constructions, improve bounds, or assist attacks on open problems; and (iv) the inference and training-time techniques, including CoT prompting, tool use, process reward models, and RLVR, that increasingly connect generation with verification. We catalog major benchmarks across grade-school arithmetic, competition mathematics, geometry, formal proving, multimodal and multilingual reasoning, and expert evaluation, and we examine benchmark saturation, contamination, reporting mismatches, and the distinction between pass@1, majority voting, and verifier-assisted pass@$k$. We critically assess failure modes: brittleness under perturbation, reward hacking, multimodal grounding failures, fragile formalization, and the energy cost of reasoning-scale inference. Drawing on recent perspectives from working mathematicians, we identify future directions centered on verified-discovery workflows, reasoning efficiency, and infrastructure to make AI-assisted formalization broadly usable. Companion materials: this https URL. 

---
# ConMem: Structured Memory-Guided Adaptation in Training-Free Multi-Agent Systems 

**Authors**: Zhixun Tan, Qiang Chen, Tairan Huang, Xiu Su, Yi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.08702)  

**Abstract**: Recent advances have improved the adaptive capabilities of LLM-based multi-agent systems (MAS) through memory-, skill-, and learning-based approaches, yet these approaches remain challenged by noisy trajectories, insufficient modeling of memory-skill relations, and reliance on additional training or high-quality supervision. To address these limitations, we propose ConMem, a relation-aware and training-free framework that enables efficient multi-agent adaptation through cross-experience coordination. Specifically, ConMem distills historical interaction trajectories into structured memory cards to capture reusable strategies and cues, organizing them into a relation-aware memory graph. At runtime, ConMem retrieves cards according to task needs and coordinates them through the card graph to resolve strategy conflicts and recover their dependencies. Combined, these modules yield structured and relation-aware guidance, enabling robust, lightweight adaptation in multi-agent systems without additional training. Extensive experiments across multiple benchmarks and mainstream MAS architectures show consistent gains over existing memory architectures, with improved inference-time efficiency through pruning more than 50% of expanded candidates and reducing planning overhead by over 80%. Our codes are available at this https URL 

---
# Extending Ontologies: From Dense Embeddings to Hybrid Quantum-Fuzzy Systems 

**Authors**: Angjelin Hila  

**Link**: [PDF](https://arxiv.org/pdf/2606.08658)  

**Abstract**: LLMs have revolutionized knowledge representation and retrieval, but lack the explicit modeling that knowledge ontologies possess. This paper surveys the ways that ontologies and knowledge graphs have been integrated with dense embedding algorithms. All hitherto attempts involve a trade-off between probabilistic and crisp inference. This paper proposes a novel frontier for devising knowledge representation systems that can simultaneously accommodate probabilistic and crisp inference in the same representation. To this effect, the paper proposes neuro-quantum-fuzzy systems as knowledge representation systems that accommodate both classical and contextual inference implemented through quantum-neural networks (QNN). 

---
# Towards Long-Horizon Vessel Trajectory and Destination Forecasting with Reasoning Large Language Models 

**Authors**: Hongwei Wang, Miao Zhou, Fengde Wang, Yuting Wang, Jiewen Yu, Jun-Yan He, Bohao Qu, Wanbing Zhang, Xiuju Fu, Qing Guo, Zipei Fan, Yingying Xing, Yi Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08633)  

**Abstract**: Long-horizon maritime trajectory prediction is important for shipping management, logistics planning, and maritime risk analysis, yet month-level forecasting remains insufficiently studied. Existing deep learning methods mainly focus on short- and mid-term coordinate extrapolation and often struggle to preserve route feasibility and destination correctness over extended horizons. This paper investigates joint long-horizon vessel trajectory and destination forecasting with reasoning-capable large language models, and develops a Maritime LLM post-training framework based on Reinforcement Learning with Verifiable Reward (RLVR). An AIS-based benchmark is constructed with 60-day historical trajectories and 30-day forecasting horizons, where trajectories are converted into semantic textual representations for RL prompt construction. RLVR aligns LLMs with maritime forecasting objectives by enforcing physical validity, providing early-weighted trajectory supervision, and evaluating destination correctness through hierarchical matching and curriculum learning. Experimental results show that RLVR-trained LLMs substantially improve over zero-shot LLMs and representative deep learning baselines, especially on destination-related metrics. Among the evaluated RLVR-trained variants, 4B LLMs achieve the best overall performance, suggesting that reward-compatible optimization and task-specific capacity matching are more important than simply using larger 8B or 14B LLMs. The results also show that LSTM remains a strong deep learning baseline under limited fine-tuning data, while Transformer-style spatio-temporal models typically require larger datasets and richer structured inputs. Overall, this work advances semantic, verifier-aligned maritime forecasting for operational decision support. 

---
# InA-Probe: Instruction-Aware Active Probing for Time Series Forecasting with LLMs 

**Authors**: Peiliang Gong, Emadeldeen Eldele, Chenyu Liu, Ziyu Jia, Yi Ding, Xinliang Zhou, Lianchao Gu, Qi Zhu, Yang Liu, Daoqiang Zhang, Xiaoli Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.08601)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated impressive potential for time series forecasting. However, existing methods predominantly rely on passive modality alignment or static task reprogramming, which often fail to capture fine-grained, non-stationary temporal patterns or to adapt to nuanced task intents. In this paper, we propose Instruction-aware Active Probing (InA-Probe), which shifts the paradigm from passive alignment toward an active, instruction-driven probing mechanism. Specifically, we design a Multi-Level Instruction Injection mechanism that enriches the model with both global task objectives and fine-grained, patch-level semantic priors. Building on this, an Adaptive Query Generation module produces sample-specific probes that are dynamically modulated by the temporal context. These probes are then refined through a dual-stage attention process: they first internalize task-specific intents via Instruction-Aware Self-Attention, and subsequently interrogate the projected temporal representations through Temporal Cross-Attention to extract salient patterns. Comprehensive experiments on seven real-world benchmarks show that InA-Probe consistently outperforms state-of-the-art deep learning and LLM-based baselines, excelling in both one-for-all generalization and zero-shot transfer while reducing forecasting error by up to 37\% in challenging cross-domain scenarios. Ablation studies further confirm that the synergy between adaptive querying and fine-grained instructions is key to unlocking the reasoning power of LLMs for complex time series. 

---
# Distilling LLM Reasoning into an Interpretable Policy Tree for Human-AI Collaboration 

**Authors**: Beiwen Zhang, Yongheng Liang, Guowei Zou, Haitao Wang, Hejun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08596)  

**Abstract**: Constructing efficient and reliable policies to assist humans is indispensable for human-AI collaboration. Existing methods mainly follow two lines of work. Most prior work relies on multi-agent reinforcement learning (MARL) to learn black-box policies, which limits interpretability and raises safety concerns. Recent methods query large language models (LLMs) at each decision step, causing slow responses and high inference costs. We propose Collaboration Policy Tree (Co-pi-tree), a closed-loop method that learns an executable policy tree consisting of a partner-behavior prediction tree and an agent-action selection tree. Co-pi-tree constructs a policy by distilling LLM reasoning into policy tree code. It then evaluates the policy through partner interaction, obtains feedback, and uses natural language to summarize the interaction feedback to improve problematic branches. Experiments in Overcooked-AI show that Co-pi-tree improves average reward by 35.4% over the baseline average, while reducing the number of LLM queries by 77.7% and test-time latency by 97.1%. Project page: this https URL 

---
# Quantitative Promise Theory: Intentionality and Inference in Autonomous Agents 

**Authors**: Mark Burgess  

**Link**: [PDF](https://arxiv.org/pdf/2606.08552)  

**Abstract**: I discuss some quantitative representations of Promise Theory for processes involving autonomous agents. Agent models are common in software systems, machine learning, and biology, for example, but may also apply to physics and other forms of engineering. I describe how Bayesian probability and information theoretic optimization, including Active Inference, may be incorporated with promise semantics -- as well as how Promise Theory supplements solutions, helping to avoid probability's pitfalls, which include non-local coordination, calibrating, and normalizing probabilistic computations. The role of boundary conditions in constraining allowed states and selecting decision thresholds is a form of promise, and agent alignment provides a scalable definition of intent. Autonomous agents may congeal into swarms with superagent characteristics by trying to minimize their information, despite uncertainty that works to maximize it. The use of Promise Theory involves some research challenges as well as stylistic preferences. 

---
# PAEC: Position-Aware Entropy Calibration for LLM Reasoning in RLVR 

**Authors**: Shumeng Yang, Yisu Liu, Jiayi Zheng, Zhaohui Yang, Linjing Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.08543)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) improves large language model reasoning but often suffers from rapid policy-entropy collapse, where the policy prematurely concentrates on narrow high-probability reasoning paths. While global entropy regularization can encourage exploration, uniformly increasing entropy across all token positions is inefficient for long reasoning trajectories, where many tokens are not decision-relevant. We propose Position-Aware Entropy Calibration (PAEC), a token-level entropy-management framework that constructs a soft mask from local top-p entropy and top-two candidate competition, and applies an anchor-based lower-bound penalty to prevent selected-position entropy collapse. Experiments on five mathematical reasoning benchmarks show that PAEC improves macro-average majority-vote performance over strong RLVR baselines, with clear gains on AIME-style tasks. Our results suggest that entropy management in reasoning RL should be formulated as selective exploration allocation over decision-sensitive positions rather than uniform randomness injection. 

---
# AgentTrust: A Self-Improving Trust Layer for AI-Agent Actions 

**Authors**: Chenglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08539)  

**Abstract**: AI agents increasingly take consequential actions -- shell commands, cloud operations, and arbitrary tool-calls -- so a trust layer must decide, per action, whether to allow, warn, block, or escalate. We argue that the right way to reason about such a layer is by threat type. Lexical (fixed-signature) threats, where danger lives in a stable token, are decidable by deterministic rules; semantic (intent-dependent) threats, where a benign and a malicious action share the same surface, are out of reach for rules by construction. We make this concrete with a negative proof: a determined, hand-authored cloud rule pack lifts held-out accuracy only 48 to 56% overall and moves the semantic categories by 0pp (data_db 29 to 29, observability 59 to 59, supply_chain 50 to 50), while a strong LLM judge carries exactly those categories. We give the judge a self-learning capability: on a corpus that is mainly semantic attacks it nearly doubles rule accuracy (48% to 83.6-85.2%) with near-zero false-blocks, and this holds across two model providers. We turn this into a self-improving dual-store system: the judge distills a growing deterministic rule floor on lexical threats (cheaper over time) and feeds a guarded RAG memory on semantic threats (a verdict-cache fails -- surface-twins collapse to ~58% -- so a corroboration guard lifts semantic accuracy +13pp, 70 to 84). The result is what sets AgentTrust v2 apart from its static v1 predecessor: a trust layer that self-evolves from its own stream of decisions -- cheaper on the lexical class (it distils its own rules) and smarter on the semantic class (it accrues guarded precedent), while never hard-blocking a benign action. An end-to-end online replay shows the judge-call rate falling (50% to 44%) and judge-domain accuracy rising (71% to 80%), with 0 benign hard-blocks across 45,000 actions. 

---
# DN-Hypo-Pipeline: An AI-Driven Workflow for Hypothesis Generation via Large Language Models and Scientific Explanations 

**Authors**: Lei Lin, Ronghao Wang, Chunbao Zhou, Jue Wang, Yangang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08532)  

**Abstract**: A scientific hypothesis is the first step in research and undergoes experimental validation, yet it also reflects a deep understanding of and reasoning about scientific phenomena. We introduce DN-Hypo-Pipeline, an AI-powered workflow based on large language models, designed to support structured scientific thinking and hypothesis generation by leveraging scientific explanations as prior knowledge. This pipeline assists researchers in deriving novel hypotheses from existing literature. Given the explanandum (i.e., the conclusion) of a research paper, it identifies underlying laws, theories, and principles, and reconstructs a new, yet-to-be-verified explanation for the observed phenomenon. We evaluated DN-Hypo-Pipeline in the field of data science modeling using three highly cited papers. Statistical inference, supported by both LLM-as-judge assessment and human expert evaluation, demonstrates that our pipeline is more effective than direct generation methods. Additionally, we validated the two highest-scoring generated hypotheses by developing corresponding novel algorithms, which outperformed the baseline models presented in the original papers. Beyond application in data science, DN-Hypo-Pipeline provides a theoretical framework that not only encompasses theory-guided data science modeling methods but also reveals a more fundamental structure of the modeling process. Moreover, this approach is essentially a generalization of theory-guided modeling, offering potential for extension to other domains and across a broader range of scientific disciplines. 

---
# VESTA: A Fully Automated Scenario Generation and Safety Evaluation Framework for LLM Agents 

**Authors**: Lu Jia, Haibo Tong, Feifei Zhao, Jindong Li, Dongqi Liang, Ping Wu, Qian Zhang, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2606.08531)  

**Abstract**: Large language models (LLMs) are increasingly evolving from simple text-based interaction systems into LLM agents that can maintain memory, use tools, access external environments, and execute tasks. As their capabilities and autonomy expand, the safety risks they face also become more diverse. Existing evaluations often rely on manually written scenarios, static prompts, or final-output judgments, making it difficult to capture the diverse risks that agents may face during task execution. We introduce VESTA, a fully automated scenario generation and safety evaluation framework for LLM agents. Based on five risk dimensions, VESTA instantiaes abstract and diverse safety risks in real-world task execution into 1,072 measurable evaluation scenarios. Using the automated evaluation pipeline, 12 LLM agents are evaluated under two authority contexts. The results show that current agents still face substantial behavioral safety risks during task execution, with an average ASR of 47.1% and several models exceeding 70%. These findings demonstrate the importance of executable, process-level evaluation for understanding and improving LLM agent safety. 

---
# Scaffold Effects on GAIA: A Controlled Comparison 

**Authors**: Jason Starace  

**Link**: [PDF](https://arxiv.org/pdf/2606.08529)  

**Abstract**: Published agent capability scores conflate what a model can do with what its scaffold lets it do, and the magnitude of this elicitation gap is not well characterized under controlled conditions. This study executes a pre-registered controlled comparison of three scaffolds (ReAct, a Planner-Actor-Rater multi-agent design, and planner-then-executor) across five models from three providers (Claude Opus 4.7, Sonnet 4.6, Haiku 4.5; Gemini 3.1 Pro Preview; GPT-5.5) on GAIA validation Levels 1 and 2, holding tasks and conditions fixed, with three attempts per question. Scaffold choice alone moves measured accuracy by as much as 28 percentage points within a single model (Opus, Level 2, robust slice), confirming the pre-registered hypothesis that scaffold variation produces gaps of at least 10 points. The pre-registered prediction that more capable models would be less scaffold-sensitive is rejected in direction: scaffold effects vary significantly by model in every dataset slice, but the most capable Anthropic model gains the most from structured scaffolds at the harder level, and tier-scaling holds only at Level 1 under the robust slice. The multi-agent advantage over ReAct at Level 2 appears within the Anthropic family but not for the cross-provider models, making model family rather than capability tier the conditioning variable, and the predicted planner-executor advantage on file-reading tasks is falsified. Structured scaffolds make fewer tool calls yet recover more often from mid-trajectory errors at the harder level, and a single cell (Gemini with planner-then-executor) is the cheapest at both levels and the most accurate at Level 2. These results indicate that single-scaffold capability numbers are scaffold-conditional estimates and that the elicitation gap is not guaranteed to shrink as models improve. 

---
# Standpoint Logics with Defeasible Beliefs 

**Authors**: Nicholas Leisegang, Thomas Meyer, Sebastian Rudolph  

**Link**: [PDF](https://arxiv.org/pdf/2606.08503)  

**Abstract**: In this paper, we integrate the defeasible logic of Kraus, Lehmann and Magidor (KLM) with the standpoint logic framework of Gómez Álvarez and Rudolph. This is done with the goal of formally expressing knowledge taking into account multiple (possibly contradicting) viewpoints, which in turn may hold defeasible beliefs. In doing so, we utilise Defeasible Restricted Standpoint Logics (DRSL), introduced by Leisegang et al. Our work expands on previous work by providing a foundational representation result for DRSL semantics and systematically lifting several well-known entailment relations from the propositional case to the standpoint-enhanced setting. In particular, we characterise the semantics for DRSL through a set of KLM-style postulates adapted for the standpoints case. We furthermore provide a means to lift preferential entailment, and the class of entailment relations based on single ranking functions from the purely propositional to the standpoint-enhanced context, including rational and lexicographic closure. We show this can be done equivalently through semantic and algorithmic means. Furthermore, we show that, for each considered form of entailment, the complexity class of entailment checking does not change when moving from propositional KLM to DRSL. 

---
# Explaining Black-Box Language Models: Learning to Optimize Linguistically-Structured Word Subsets 

**Authors**: Minyoung Hwang, Seokhyun Lee, Changhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.08497)  

**Abstract**: As deep language models (DLMs) are increasingly deployed in high-stakes domains such as healthcare, understanding their decision rationale becomes paramount for ensuring trust, safety, and accountability. However, achieving this vital level of interpretability is particularly challenging when these DLMs operate as black-box systems (e.g., via APIs), where access to internal model states (e.g., parameters, gradients) is restricted. Despite numerous efforts, existing explanation methods often fail to concurrently satisfy three key desiderata: (i) inference-time efficiency, (ii) black-box compatibility without inducing out-of-distribution behavior, and (iii) comprehensible explanations grounded in the input's linguistic structure. To address these challenges, we propose a method that explains predictions of DLMs by selecting a small, informative subset of input words. We formulate this as an amortized optimization problem, enabling efficient one-shot inference without the need for input-specific search. Our selection policy is trained via REINFORCE-style policy gradients, allowing discrete word selection in a fully gradient-free setting. To enhance interpretability and align with human linguistic intuition, we integrate graph-structured knowledge into this selection process, fostering linguistically coherent subsets that result in explanations both highly informative and cognitively meaningful to end-users. We evaluated our method on diverse DLM architectures and multiple real-world datasets. It consistently identifies word subsets with enhanced discriminative power and stronger alignment with linguistically salient cues, outperforming both conventional black-box compatible methods and gradient-based approaches that are given oracle access to the black-box model's gradients for a more challenging benchmark. Our code is available at here. 

---
# What Makes a Desired Graph for Relational Deep Learning? 

**Authors**: Yao Cheng, Siqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2606.08491)  

**Abstract**: Relational deep learning (RDL) converts relational databases (RDBs) into heterogeneous graphs, but graphs derived directly from database schemas are often not well suited for how graph neural networks (GNNs) perform relational reasoning. We study what makes a relational graph suitable for deep learning and show that schema-derived graphs suffer from two systematic failures: information overload and semantic fragmentation. Our empirical analysis reveals that the desired graph is not the raw schema, but a result of controlled structural adaptation. Performance depends on balancing two operations: mitigating information overload via filtering, and repairing semantic fragmentation via injection. Specifically, filtering serves as a bias-variance knob with non-monotonic effects, while injection improves performance only when it explicitly restores the relational dependencies missing from the original schema. Based on these findings, we develop an end-to-end structural optimizer that applies both operations to adapt relational graphs automatically. Across 26 tasks spanning classification, regression, and recommendation, the optimized graphs consistently improve accuracy while often reducing inference cost. 

---
# Testing the Black Box: Structural Barriers to Independent Evaluation of Consumer-Facing Health LLMs 

**Authors**: Rahul Gorijavolu, Kaushik Madapati, Pritika Vig, Rawan Abulibdeh, Nikhil Jaiswal, Mahri Kadyrova, Zeamanuel Hailu Tesfaye, Charles Senteio, Paula Maurutto, Leo Anthony Celi  

**Link**: [PDF](https://arxiv.org/pdf/2606.08483)  

**Abstract**: Background: Consumer-facing large language models are now a common source of health information, and they interpret and personalize responses rather than retrieve them. Whether their responses vary across users is a clinical, equity, and governance question, sharpened by evidence that sycophantic responses can alter judgment and increase trust.
Objective: To evaluate response variation and sycophancy in consumer-facing health LLMs under conditions resembling ordinary patient use.
Methods: We constructed simulated user profiles differing in geography, browsing context, expressed beliefs, and social determinants of health, drawing on literature linking social context to health attitudes. We adapted validated instruments, including the Vaccination Attitudes Examination scale and reproductive attitudes scales, into multi-turn prompts designed to elicit clinically meaningful variation across users.
Results: The evaluation encountered five linked barriers. Factual prompts produced stable responses that masked sycophancy emerging over multi-turn conversation. Browser-based interfaces did not disclose which signals influence outputs and could not be reset to a clean baseline. Large-scale testing was restricted by terms of service, rate limits, and bot detection. Accuracy-based criteria could not capture tone, framing, or omission, and LLM-as-judge methods risked shared alignment bias. Models changed without traceable version identifiers, preventing reliable replication.
Conclusions: No reliable independent evaluation framework yet exists for examining how consumer-facing health LLMs behave in ordinary use. Oversight requires disclosure of personalization signals, stable version identifiers, researcher safe harbor programs, and post-deployment monitoring of health-related outputs. 

---
# A Variability-Based Framework for Interpretable Naming in Formal and Relational Concept Analysis 

**Authors**: Alain Gutierrez, Marianne Huchard, Pierre Martin, André Miralles, Violaine Prince  

**Link**: [PDF](https://arxiv.org/pdf/2606.08477)  

**Abstract**: Knowledge extraction from symbolic data often produces abstractions that are formally defined but not immediately interpretable by users. Formal Concept Analysis (FCA) and Relational Concept Analysis (RCA) provide representative settings for this issue: they generate explicit conceptual structures, implications, and relational dependencies from object descriptions and relations. Although these structures are explainable by design, their concepts are often identified by technical labels, which limits their use as human-interpretable knowledge units. Assigning meaningful names to such concepts is therefore a key issue for interpretation, navigation, validation, and reuse by domain experts.
This paper investigates concept naming in FCA and RCA from a symbolic knowledge representation perspective. We first characterize the linguistic and terminological challenges involved in naming generated symbolic abstractions, including ambiguity, discrimination, concision, and consistency across related concepts. We then propose a configurable framework for LLM-assisted concept naming. The framework relies on a variability model that controls which sources of information are exposed during naming, such as intent, extent, inherited information, neighboring concepts, implications, and relational attributes. It thereby makes explicit the semantic choices involved in moving from formal concept descriptions to human-readable names.
The approach is illustrated as a proof of concept on a small relational dataset in the pizzeria domain. This illustration shows how different configurations influence the names suggested by an LLM, and how naming variability can reveal interpretation choices, relational dependencies, and possible modeling issues in the underlying symbolic data. 

---
# GIFT: LLM-Guided State-Reward Interface for Financial Reinforcement Learning 

**Authors**: Yanyan Wu, Boyi Zhang, Yanlin Liu, Xinyu Fang, Jining Luan, Meiqi Zhang, Jiacheng Liu, Hao Zeng, Dexu Yu, Chang Liu, Hanwen Du, Yongxin Ni, Youhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.08450)  

**Abstract**: Financial portfolio trading is naturally formulated as a reinforcement learning problem, where an agent sequentially rebalances assets under changing market conditions to balance return, risk, and transaction costs. Yet in non-stationary markets, raw OHLCV states and short-horizon return rewards often provide an under-specified learning interface, motivating large language models as a way to inject financial knowledge into state and reward design while constraining open-ended generation. To this end, we propose GIFT, an LLM-guided framework for state-reward interface design in PPO-based financial reinforcement learning. Rather than using the LLM to make trading decisions, GIFT uses Factor-guided State Enhancement to generate state features from financial-factor primitives, Risk-rule-guided Reward Shaping to generate auxiliary rewards from portfolio-risk rules, and Diagnostic-guided Refinement to revise candidate interfaces using PPO rollout diagnostics. After refinement, GIFT fixes the selected state-reward interface before evaluation, with no further LLM queries or interface updates at test time. Comprehensive rolling-window experiments across diverse market regimes and portfolio scenarios demonstrate that GIFT improves learning-signal quality and out-of-sample risk-adjusted portfolio performance over baselines. Code and data are available at: this https URL . 

---
# Trajectory-Refined Distillation 

**Authors**: Li Jiang, Haoran Xu, Yichuan Ding, Amy Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08432)  

**Abstract**: On-policy distillation (OPD) has become a central post-training tool for large language models (LLMs), providing dense per-token teacher supervision along the student's own rollouts. In this work, we identify a common structural cause underlying OPD, which we call prefix failure. Under prefix failure, dense per-token supervision induces a bimodal teacher mixture and fragmented gradients that token-level loss truncation or reweighting fail to address. This observation motivates us to move beyond token-level loss interventions toward trajectory-level output corrections. We thus propose Trajectory-Refined Distillation (TRD), a trajectory-level correction method that revises the student's rollout under the teacher guidance while within on-policy support. By correcting problematic prefixes before distillation, TRD mitigates prefix failure at its source. Moreover, TRD improves the exploration by exposing the student to alternative valid derivations under teacher guidance, even when the original rolls are already correct. TRD can also be applied to on-policy self-distillation (OPSD), a parameter-sharing variant that uses the student model conditioned on privileged informations as the teacher. Across a wide range of benchmarks and base models at multiple scales, TRD consistently outperforms prior baselines, improving single-attempt accuracy and broadening reasoning coverage. Code is available at this https URL 

---
# Self-Evolving Scientific Agent Discovers Generalizable Physically-Reasoned Fluid Control 

**Authors**: Boai Sun, Wenjin Guo, Zongmin Yu, Liu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08405)  

**Abstract**: While data-intensive deep reinforcement learning can optimize complex control policies, scientific discovery in physical systems fundamentally requires an interpretable chain of reasoning that connects physical evidence to structured control architectures. Here, we present a self-evolving scientific-agent workflow, driven by large language models and iterative code generation, that automates controller construction while preserving strict interpretability and rigorous physical reasoning. Instead of adjusting weights, the agent deploys candidate strategies into physical simulations, actively diagnoses dynamic behaviors from multimodal evidence, and translates these observations into progressive source-code refinements. We demonstrate this framework on a highly non-linear fluid-structure interaction problem: an underactuated, two-joint dogfish swimmer tasked with spatial target reaching using only joint angular accelerations. Starting from a propulsive seed policy that exhibits a one-sided steering bias, the agent autonomously discovers and refines a unified controller that robustly captures all canonical targets. Remarkably, without any retraining or target-specific branching, the synthesized control policy generalizes to unseen static targets and dynamically curved pursuit trajectories. The auditable evolve log reveals an emergent control architecture built upon traveling-wave propulsion, body-frame target guidance, yaw-rate feedback, signed mean-tail curvature, and adaptive cadence relief. Our results show that an autonomous scientific agent can successfully transform accumulated physical evidence into robust, mathematically readable control policy, while maintaining a fully traceable process of scientific discovery. 

---
# TT-DAC-PS: Twin-Target Deterministic Actor-Critic with Policy Smoothing for Optimal Trade Execution 

**Authors**: Ilia Zaznov, Atta Badii, Julian Kunkel, Alfonso Dufour  

**Link**: [PDF](https://arxiv.org/pdf/2606.08379)  

**Abstract**: This study addresses the optimal execution of large stock sell programs by introducing TT-DAC-PS (Twin-Target Deterministic Actor-Critic with Policy Smoothing), a deterministic actor-critic architecture that combines twin exponential-moving-average critic targets with pessimistic min backup, TD3-style target policy smoothing noise, delayed actor updates, and conservative Q regularisation to curb overestimation. Exploration uses Ornstein-Uhlenbeck (OU) noise with a hybrid schedule: deterministic episode-wise decay, variance-guided adjustment based on recent reward dispersion, and a Soft Actor-Critic (SAC)-style temperature that is learned and mapped to the noise scale. The environment integrates Almgren-Chriss (AC) trade impact with Limit Order Book (LOB) prices and volumes, normalised state features, per-step volume participation caps, and a utility-based reward. The trade execution algorithm is applied to LOB data for ten U.S. stocks. Performance is assessed against reinforcement-learning baseline algorithms, including Proximal Policy Optimisation (PPO), Soft Actor-Critic (SAC), and Advantage Actor-Critic (A2C), as well as alternative trade execution algorithms, including Time-Weighted Average Price (TWAP), Volume-Weighted Average Price (VWAP), and AC. The proposed model consistently reduces mean implementation shortfall percentage with competitive variance, outperforming classical baselines and standard reinforcement-learning benchmark models. 

---
# Benchmarking Open-Ended Multi-Agent Coordination in Language Agents 

**Authors**: Kale-ab Abebe Tessera, Andras Szecsenyi, Cameron Barker, Alexander Rutherford, Davide Paglieri, Aidan Scannell, Henry Gouk, Elliot J. Crowley, Tim Rocktäschel, Amos Storkey  

**Link**: [PDF](https://arxiv.org/pdf/2606.08340)  

**Abstract**: As language models are increasingly deployed as autonomous agents, they must coordinate with others over long horizons in open-ended interactive tasks. Yet existing evaluations rarely test these demands together, instead emphasising single-agent tasks, short interactions, or highly structured multi-agent settings. We introduce $alem$, a JAX-based benchmark for open-ended multi-agent coordination built on Craftax-like dynamics. Alem embeds procedurally generated coordination tasks, soft specialisation, communication, and controllable coordination difficulty into a long-horizon survival world with exploration, crafting, trading, and combat. We evaluate $13$ modern LLMs zero-shot within homogeneous teams, with trained MARL agents as reference points. Current LLM agents remain far from solving alem, averaging only ~6% normalised return, but their failures are not uniform. On the hardest coordination setting, zero-shot Gemini-3.1-Pro-High approaches MARL agents trained for one billion steps, while GPT-5.4-High achieves strong base-task reward but much lower coordination reward. This contrast shows that individual task competence does not imply coordination competence. Ablations show that communication is the largest contributor to coordination, while memory and reasoning help when used to maintain multi-step plans. Overall, our results identify coordination as a distinct bottleneck for frontier LLM agents, separate from single-agent capabilities. Alem makes this bottleneck measurable and provides a controlled testbed for developing agents that communicate, allocate roles, and execute shared plans. Code is available at this https URL. 

---
# Integrating Deep Learning Demand Forecasting with Multi-Objective Optimization for Circular Coffee Supply Chains: A Data-Driven Framework for Cost, Emissions, and Freshness Management 

**Authors**: Gerçek Budak, Faraz Gholamzadeh Gharehgheshlaghi, Melika Barjesteh Vaezi, Ahmad Gholizadeh Lonbar  

**Link**: [PDF](https://arxiv.org/pdf/2606.08314)  

**Abstract**: The coffee supply chain is one of the most complex agri-food networks, marked by geographically dispersed production, multi-tier coordination, and high sensitivity to quality and freshness. While sustainability and digitalization have gained attention, demand forecasting, optimization, and traceability are often treated separately. This study presents a two-phase integrated framework. First, a hybrid CNN-LSTM model is used for demand forecasting. On the public Coffee Chain Sales dataset with chronological 70/15/15 splitting, the model achieves MAE of 22.87 and R^2 of 0.90, outperforming the best deep learning benchmark by ~12% and classical methods by over 30%. In the second phase, the forecasted demand feeds a tri-objective mixed-integer linear programming (MILP) model that jointly minimizes cost, minimizes carbon emissions, and maximizes product freshness in a multi-period, multimodal, closed-loop supply chain with circular recovery. Freshness is modeled via exponential decay based on inventory age. Using the epsilon-constraint method, 25 Pareto solutions are obtained. Sensitivity and policy analyses show that balanced sustainability policies can reduce emissions by 22.4% with only a 9.9% cost increase while maintaining near-optimal freshness.
Keywords: Coffee supply chain; Deep learning; Demand forecasting; Multi-objective optimization; Circular economy; CNN-LSTM; Mixed-integer linear programming. 

---
# Neuro-Symbolic Injection of LTLf Constraints in Autoregressive Reinforcement Learning Policies 

**Authors**: Ashkan Ansarifard, Matteo Mancanelli, Elena Umili, Fabio Patrizi  

**Link**: [PDF](https://arxiv.org/pdf/2606.08312)  

**Abstract**: In this work we study offline reinforcement learning (RL) under temporally extended task constraints expressed in Linear Temporal Logic over finite traces (LTLf). Recently, transformer-based approaches such as Trajectory Transformers and Decision Transformers have been adopted to address RL as a sequence modeling problem. However, these methods optimize purely for reward and do not account for high-level temporal requirements. Here, we introduce a neurosymbolic framework that injects LTLf background knowledge into such transformer-based RL policies. Our approach compiles LTLf formulas into deterministic finite automata (DFAs) and integrates them into the learning process through a differentiable representation and a logic-based loss function. In particular, we derive differentiable satisfaction signals from DFA progression and use them as a regularization term during training. The resulting method is architecture-agnostic across different models. We evaluate the proposed framework on navigation environments with specification suites covering combinations of safety and reachability temporal properties. Experimental results show that incorporating background knowledge not only improves constraint satisfaction, but also maintains competitive return compared to vanilla baselines. 

---
# Curation of a Cardiology Interface Terminology for Highlighting Electronic Health Records using Machine Learning 

**Authors**: Mahshad Koohi Habibi Dehkordi, Shuxin Zhou, Yehoshua Perl, Fadi P. Deek, James Geller, Gai Elhanan, Andrew J. Einstein, Luke Lindemann, Vipina K. Keloth  

**Link**: [PDF](https://arxiv.org/pdf/2606.08311)  

**Abstract**: Electronic health record (EHR) notes are dense medical documents containing large amounts of information, often filled with complex medical jargon. Highlighting all details in EHRs helps reduce the likelihood of missing crucial information by drawing attention to key content. This study proposes the design of a Cardiology Interface Terminology (CIT) to accurately highlight all details in EHR notes of cardiology patients. We introduce an innovative Machine Learning (ML) technique for the design of CIT. The ML technique requires training data. Manual preparation of such training data is time-consuming and expensive. The process of the CIT design includes three phases. In the first two phases, we innovatively derive a training data CIT to be used by the third phase, ML technique. We start by designing an initial CIT, composed of several components: the cardiology-related sub-hierarchies of SNOMED, other SNOMED concepts mined from EHRs of build set, and necessary components of terms e.g., medical abbreviations and medications. Utilizing an iterative process, fine-grained phrases containing initial CIT concepts are extracted from build set as CIT concept candidates. The candidate concepts are semi-automatically reviewed before being added to CIT, yielding the training data CIT, TCIT. In the third phase, a ML model is trained with TCIT to identify candidates fitting to be concepts in the CIT. This model is used to extract further concepts from build set, yielding the final CIT. The final CIT is then used to highlight the test set and evaluate the extent to which it captures details in an unseen EHR dataset. For this purpose, four evaluation metrics, coverage, breadth, completeness, and conciseness are used. The highlighted test set has a coverage of 74.21%, with a breadth of 1.68. For 20 random notes in test set, the average completeness is 98.2% and average conciseness is 84.2%. 

---
# To Nuke or Not to Nuke: LLMs' (Missing) Ethical Reasoning and Actions in a High-Stakes Decision-Making Simulation 

**Authors**: John Chen, Sihan Cheng, Can Gurkan, H M Abdul Fattah  

**Link**: [PDF](https://arxiv.org/pdf/2606.08310)  

**Abstract**: Large language models (LLMs) are increasingly deployed as long-horizon agents with decision-making capacities. While LLMs can show ethical competence on dilemmas such as trolley problems, this competence may not translate to complex, agentic scenarios. We study this gap in Civilization V, a multiplayer game with a complex decision-making landscape including economy, diplomacy, technology, and military strategy. Starting from 130 high-tension LLM self-play episodes, in which an LLM player spontaneously escalated nuclear authorization, we replay them across 13 models with three prompt interventions: an ethical prompt naming nuclear harm, removal of the previous model's decision-making rationale, and high-stakes framing emphasizing real-world impacts. No interventions nor their combinations reliably eliminate emergent escalation. We identify three failure pathways: ethical reasoning that fails to surface without prompting, fails to appear even when prompted, or surfaces but fails to take effect when strategic counter-factors dominate. Evaluations of agentic models, therefore, must test whether ethical reasoning is spontaneously invoked and behaviorally effective in complex decision-making contexts, beyond whether it can be elicited in isolation. 

---
# Revisiting the shutdown problem 

**Authors**: David Thorstad  

**Link**: [PDF](https://arxiv.org/pdf/2606.08296)  

**Abstract**: A key premise in leading arguments for existential risk from artificial intelligence is that malfunctioning artificial agents could not be easily shut down. This motivates the catastrophic shutdown problem of ensuring that agents can be shut down before they cause an existential catastrophe. A range of arguments and theorems are offered to suggest that solving the catastrophic shutdown problem is difficult, bolstering arguments for existential risk and motivating a search for solutions to the catastrophic shutdown problem. This paper argues for two conclusions. First, existing arguments do not establish the difficulty of solving the catastrophic shutdown problem. Second, concern for the catastrophic shutdown problem has led to technical solutions that impose a high safety tax on model performance. 

---
# Ablation-Reversible Heads Don't Transfer: A Stress Test for Mechanistic Role Claims in Transformers 

**Authors**: Philip Quirke  

**Link**: [PDF](https://arxiv.org/pdf/2606.08292)  

**Abstract**: In mechanistic interpretability, attention heads are commonly elevated to role claims (e.g., "this head represents addition") when they are necessary for a behavior, encode it linearly, and recover that behavior when restored after ablation. We show this evidence is insufficient: across three 7-8B instruction-tuned models and five computation families, heads passing all three checks routinely fail to transfer the computation when their activations are patched into a different prompt under matched controls. We introduce KID (Knowing / Intent / Doing), a role-assignment lens for attention heads, and pair it with a three-stage pipeline: capability-selective screening (CSS), singular value decomposition (SVD), and activation transduction under matched controls. Our results document a preliminary role taxonomy (including prompt-trajectory stabilizers, answer-side logit-bias heads, and soft computation-pattern carriers) and show that the same-answer control (a transduction target sharing the answer string but not the requested computation) is an underused check that exposes broad state transfer masquerading as semantic specificity. 

---
# Beyond Agent Architecture: Execution Assumptions and Reproducibility in LLM-Based Trading Systems 

**Authors**: Junyi Yao, Zihao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.08285)  

**Abstract**: Large language models (LLMs) and agentic systems are increasingly proposed for financial trading, yet their reported performance remains difficult to compare because studies vary in data provenance, temporal split discipline, execution timing, turnover treatment, and transaction-cost modeling. This article presents a targeted topical review and reproducibility audit of execution realism in LLM-based trading research. A coded evidence matrix covering 30 trade-relevant primary studies is used to assess point-in-time controls, split transparency, held-out evaluation, cost and turnover treatment, execution semantics, universe definition, and artifact release. Across the audited sample, architecture reporting is generally clearer than the evaluation assumptions needed to judge whether a trading result is economically interpretable or reproducible. A 10-equity worked example is included only as a methodological scaffold to illustrate how explicit friction and timing choices can materially compress active-strategy results. The main conclusion is that the next useful step for LLM trading research is not only better agent design, but also clearer reporting standards for execution realism, reproducibility, and evaluation comparability. 

---
# From Validator Selection to Portfolio Collection Optimization in Proof-of-Stake Blockchains 

**Authors**: Jonas Gehrlein, Grzegorz Miebs, Matteo Brunelli, Adam Mielniczuk, Miłosz Kadziński  

**Link**: [PDF](https://arxiv.org/pdf/2606.08282)  

**Abstract**: We consider a problem arising in proof-of-stake blockchain environments, where agents called nominators select validators - entities responsible for maintaining the blockchain's physical infrastructure. The selection process is inherently subjective and multi-criterial and combines with the fact that nominators commonly operate through multiple accounts. This gives rise to a portfolio selection problem, where agents seek to distribute their nominations across accounts to diversify risk. We propose a decision support framework to optimize this selection by simultaneously maximizing two objectives: the expected utility of the validators likely to be allocated, representing portfolio quality and profitability, and the expected entropy of the allocation, representing diversification and risk mitigation across stashes. Validator utilities are derived using an original active preference learning procedure based on multi-attribute value theory, with emphasis on top-ranked validators. The resulting bi-objective optimization problem is solved with a multi-objective evolutionary algorithm and, to support the final choice, we introduce an interactive binary search navigation procedure that guides the nominator through the front and identifies a satisfactory trade-off with only a few questions. Numerical experiments examine the optimization strategies, while an expert assessment involving five experienced nominators confirms the approach's practical relevance and usefulness. 

---
# Traxia: A Framework for Verifiable, Agent-Native Scientific Publishing 

**Authors**: Wisdom Dogah  

**Link**: [PDF](https://arxiv.org/pdf/2606.08256)  

**Abstract**: Verifiability, attribution, and reproducibility are foundational requirements of scientific knowledge, yet current publishing infrastructure does not enforce them at scale. We introduce Traxia, an agent-native scientific publishing framework in which AI research agents publish verifiable papers, build reputational identities, peer-review one another, and collaborate with humans in a shared provenance model. Traxia treats agents as first-class epistemic participants: every paper carries a reasoning trace, every claim a confidence interval, every agent a cryptographically signed identity, and every collaboration an immutable contribution log. We formalise five components: Agent Identity and Registry, Verifiable Publishing Layer, four-tier Peer Review Protocol, Reputation and Staking Engine, and a Knowledge Graph with contradiction detection. The framework targets reproducibility failure, provenance opacity, and exclusion of Global South research capacity. This paper presents architectural foundations and formal specifications only; it does not report empirical results. Evaluation and deeper component studies will follow in subsequent papers. A prototype partially implements core formalisms; the full system remains under active development. 

---
# When No Answer Is Correct: Diagnosing Absent Answer Detection for MLLMs in Video Understanding 

**Authors**: Yiheng Wang, Yueqian Lin, Lichen Zhu, Yudong Liu, Hai "Helen" Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.08239)  

**Abstract**: Multimodal large language models (MLLMs) have made substantial advancements in video understanding, yet the reliability of their responses remains underexplored. This work presents a diagnostic study of absent answer detection for MLLMs in video understanding, where the correct answer is deliberately excluded from the candidate set and a reliable model is expected to recognize that no valid option exists. We evaluate the absent answer detection behavior under three settings: multiple-choice questions augmented with an ``None of the Above'' option, open-ended generation with a detection instruction, and standard evaluation without any guidance. Across a diverse set of models and benchmarks, we find that MLLMs overwhelmingly select plausible distractors rather than detecting the absent answer. This failure is more pronounced in temporal reasoning tasks and worsens with denser frame sampling. We further explore chain-of-thought prompting as a mitigation strategy and find that while it substantially improves detection rates, performance remains unsatisfactory, suggesting that prompting-based strategies alone are insufficient to fully address this limitation. These findings expose a systematic failure in absent answer detection and highlight the need for explicit detection mechanisms in multimodal systems. 

---
# SciTrace: Trajectory-Aware Safety Reasoning for Scientific Discovery Agents 

**Authors**: Tanush Swaminathan, Runmin Jiang, Letian Zhang, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08234)  

**Abstract**: LLM-based scientific agents have shown strong capacity for autonomous research, yet their safety layers remain structurally divorced from core reasoning: they inspect pipeline outputs rather than shaping the deliberation that produces them. This separation opens two failure modes: safety signals accumulated at one stage are discarded before the next, and sequences of individually benign tool calls can compose into harmful outcomes that no single-step filter detects. To address these challenges, we introduce \textbf{SciTrace}, a framework that weaves safety reasoning into every stage of the scientific agent pipeline. SciTrace couples two complementary mechanisms: a \textit{Safety-Intrinsic Reasoning Loop} (SIR) that maintains a cumulative risk state across the Thinker, Experimenter, Writer, and Reviewer stages through joint task-and-safety deliberation, and a \textit{Compositional Tool-Chain Verifier} (CTV) that performs trajectory-aware safety checks before execution, catching risks that surface only across multi-step tool sequences. Evaluated on 240 high-risk research tasks and 120 tool-related risk tasks spanning six scientific domains, SciTrace achieves state-of-the-art (\textbf{SOTA}) safety among compared frameworks across four backbone models: it consistently improves tool call safety and adversarial robustness while preserving scientific output quality, and it uncovers \textbf{78.8\%} of the compositional tool-chain escapes that single-step monitors miss. The project website is available at this https URL. 

---
# Online Agent-as-a-Judge: Situation-Generating Evaluation for Interactive Agents 

**Authors**: Hyogon Ryu, Jeonghwan Kim, Yewon Lim, Chaeun Lee, Jeongwook Kim, Donghoon Ham  

**Link**: [PDF](https://arxiv.org/pdf/2606.08200)  

**Abstract**: Evaluating LLM-powered interactive social agents is challenging because socially relevant behaviors depend not only on isolated outputs, but also on prior interactions, social roles, and downstream actions. Existing methods typically allow a target agent to act freely in an environment and then score the resulting trajectory. However, this passive setup can miss capabilities that only become observable under specific social circumstances; for example, conflict handling may remain untested if no disagreement arises. We propose Online Agent-as-a-Judge, a situation-generating evaluation framework for interactive social agents. Online Agent-as-a-Judge deploys an in-world evaluator agent that interacts with the target agent through the environment's native dialogue and action protocol, actively eliciting situations relevant to the evaluation criteria. The resulting trajectories provide evidence for assessing both immediate responses and subsequent behavior. In a life-simulation environment with $32$ designer-authored social criteria, Online Agent-as-a-Judge improves criteria coverage and agreement with human labels, yielding more reliable evidence-grounded evaluations of behaviors that passive methods can leave unobserved. 

---
# Decision-Aware Memory Cards: Counterfactual-Inspired Context Selection and Compression for Tool-Using LLM Agents 

**Authors**: Xinyu Guan, Qianyang Zhao, Yuming Deng  

**Link**: [PDF](https://arxiv.org/pdf/2606.08151)  

**Abstract**: Tool-using LLM agents often fail not because relevant text is absent, but because decisive evidence is not selected, compressed, or surfaced at action time. We present CICL, a decision-aware context layer that turns instance evidence into a context graph, routes deterministic, Opus-assisted, Qwen, Codex/GPT-5.5, and Qwen-QLoRA judgments through a shared eight-field schema, scores units by action shift, outcome uplift, necessity, and negative-transfer risk, and packs high-utility evidence as typed memory cards for a budgeted agent. The design separates the measured decision signal from the judge model, so frontier annotation, local surrogates, and lightweight rankers can be compared under one auditable protocol. Empirically, CICL yields a concrete open-benchmark gain while exposing its limits. On 50 SWE-bench Verified file-retrieval instances, direct Qwen3.6-plus reranking of BM25 top-50 candidates raises hit@1 from 0.58 to 0.78 and MRR@10 from 0.634 to 0.790, with all 2,500 judgments parseable. Controlled diagnostics show action-criticality: at budget 120, CICL reaches F1 0.620 on v1 and 0.425 on v3, and removing the top-utility semantic v3 unit collapses F1 to 0.000. Supplementary checks add Qwen-QLoRA agreement over 710 candidates, a small 200-label real-code Opus-assisted signal, and a three-instance patch smoke validating retrieval-to-patch plumbing without claiming official SWE-bench success. RepoBench-R summaries still beat cards, and compact rankers do not yet replace the heuristic. CICL contributes a reproducible measurement and selection layer for decision-critical context, not an end-to-end coding-agent repair claim. 

---
# SAGE: An LLM-driven Self Reflective Agentic Framework for Fraud Detection 

**Authors**: Yichen Chen, Siying Li, Yuhang Liang, Lijun Wang, Renyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08146)  

**Abstract**: Fraud detection in payment, e-commerce, and telecommunications systems requires accuracy at the individual level, robustness under severe class imbalance, and ease of understanding for risk managers. Existing methods fall at least one of these requirements: automated machine learning systems search a fixed numerical space without semantic awareness of the dataset; graph neural network-based methods require pre-defined relational graphs and remain opaque at the individual-decision level; and the design of general-purpose large language model (LLM) agents does not consider the recall and precision constraints specific to real-world fraud detection. In this paper, we propose SAGE, the first end-to-end LLM-driven multi-agent framework for fraud detection. SAGE coordinates three dedicated agents that make decisions based on a six-layer Data Diagnostic Tree (DDT) and a Markov decision process guided by natural-language gradients, automatically optimizing the model under a fraud-specific reward. On five fraud datasets and five LLM backbones, SAGE wins $96.00\%$ of method--dataset comparisons and improves F1 by an average of $40.86\%$ over baselines. The code is available at this https URL. 

---
# Cross-LLM Consistency in Inference: Evidence from Shared Interactions 

**Authors**: Siyu Lou, Yao Yan, Yuntian Chen, Quanshi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08129)  

**Abstract**: Large language models (LLMs) differ in architecture, training data, and optimization procedures, yet they may still develop similar internal inference patterns. In this paper, we examine this hypothesis using interaction-based explanations. We find that LLMs often share interaction patterns when predicting the same target token from the same prompt. This consistency is more pronounced among advanced LLMs. Shared interactions also tend to be lower-order and show weaker positive-negative cancellation than non-shared interactions. These results suggest that advanced LLMs may be implicitly optimized toward common inference patterns, even though the mechanisms that give rise to such cross-model consistency remain open. 

---
# Think Before You Act: Intention-Guided Reasoning for LLM-Based Location Prediction 

**Authors**: Qingxiang Liu, Anqi Liang, Zhuoyang Jiang, Yutian Jiang, Sisuo Lyu, Yu Ji, Haomin Wen, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08122)  

**Abstract**: Predicting a user's next Point-of-Interest (POI) based on their historical check-in records is a fundamental task in location-based services. While recent methods incorporating large language models have shown strong reasoning capabilities and promising results, they typically formulate the prediction task as a one-step trajectory-to-location mapping problem, making predictions prone to shallow trajectory correlations and historical frequency bias. We argue that users rarely choose locations directly and instead, they usually first form a traveling intention and then accordingly select specific POIs. Motivated by this insight, we propose IntentPOI, a two-stage intention-guided reasoning framework. In the thinking stage, we infer users' intermediate intentions by incorporating historical mobility patterns, similar peer behaviors, and the temporal contexts. In the acting stage, we first construct a compact candidate pool, and then perform intention-guided reasoning to identify locations that best align with the inferred intention. By explicitly decoupling intention inference from location prediction, IntentPOI transforms the next POI prediction from direct trajectory matching into intention-guided reasoning. Extensive experiments on three real-world datasets demonstrate that IntentPOI consistently outperforms eleven state-of-the-art baselines. 

---
# PACE: Anytime-Valid Acceptance Tests for Self-Evolving Agents 

**Authors**: Zayx Shawn  

**Link**: [PDF](https://arxiv.org/pdf/2606.08106)  

**Abstract**: Self-evolving agents improve by repeatedly proposing changes to their own prompts, skills, or workflows and keeping those that score higher on a small held-out set. Almost all effort has gone into the proposer that generates candidates; we argue the weak point is the acceptor, the rule that decides whether to commit a change. Applied hundreds of times against the same noisy dev estimate, the ubiquitous "keep it if the score went up" rule is uncontrolled adaptive multiple testing: the agent effectively p-hacks itself, accumulating false commits that make it churn and drift rather than improve.
We recast committing as a sequential hypothesis test and propose PACE (Paired Anytime-valid Commit Evaluation), a training-free, anytime-valid commit gate. Each candidate is compared to the incumbent on identical instances and committed only when a testing-by-betting e-process accumulates decisive evidence, stopping early to save evaluations and controlling each candidate's false-commit probability at a user-set level even under optional stopping (a per-decision guarantee).
On Qwen2.5 agents (0.5B-3B) self-evolving at the prompt level on GSM8K, SVAMP, and ARC-Challenge, greedy acceptance commits 30-42% false and 10-33% harmful edits when a genuine improvement is hidden among noisy proposals, while PACE commits the real one and essentially nothing else, matching greedy's held-out accuracy at sharply lower variance and about 18% lower evaluation cost. With no real gain available, greedy commits 13-21 spurious self-modifications per run (72-100% false) and degrades the most fragile agent by 4.9 points, while PACE holds at baseline. Reliability of self-evolution depends on the acceptor, not only on the proposer. 

---
# When Does Delegation Beat Majority? A Delegation-Based Aggregator for Multi-Sample LLM Inference 

**Authors**: Yasushi Sakai, Allen Song, Kent Larson  

**Link**: [PDF](https://arxiv.org/pdf/2606.08098)  

**Abstract**: Majority voting over sampled answers is the dominant unsupervised aggregator for multi-sample LLM inference. We show that piping the signals every sample carries into a delegation-based aggregator (Propagational Proxy Voting, PPV) yields an unsupervised consensus rule that beats majority on MMLU-Pro by +1.5 pp overall and +2.24 pp on the non-trivial subset (paired McNemar p ~ 1.0e-14, n = 8,099). Majority discards two free signals every sample carries: within-group letter entropy and between-group reasoning geometry. PPV exposes two per-voter levers that consume exactly these signals: WHEN (how much weight a voter keeps on its own pick) and WHOM (how it splits the remainder across peers). We drive WHEN with letter entropy and WHOM with per-question-centered embedding cosine. The method needs no gold labels and no auxiliary training: per question, we partition 128 sampled generations into 16 groups, compute each group's letter-level semantic entropy and reasoning embedding centroid, and feed both into a stochastic delegation matrix whose stationary distribution selects the consensus answer. We walk through an example in which PPV overturns a clear 10-6 majority for the wrong letter: the 10-voter majority cluster is geometrically incoherent (mean within-cluster cosine -0.02) while the 6-voter minority is tight (+0.26), so propagated delegation mass concentrates on the minority's answer even though entropy alone would keep the majority ahead. We further report delegation strategies with negative results that constrain the design space for unsupervised LLM aggregation: no within-question ensemble of confidence modes closes the oracle gap. 

---
# A Multi-modal Agentic Co-pilot for Evidence Grounded Computational Pathology 

**Authors**: Zhe Xu, Zhengyu Zhang, Zhiyuan Cai, Jiahao Xu, Yijie Lin, Ziyi Liu, Junlin Hou, Hongyi Wang, Yuxiang Nie, Ling Liang, Yihui Wang, Yingxue Xu, Ronald Cheong Kin Chan, Li Liang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.08093)  

**Abstract**: Pathology is the cornerstone of modern medicine, where accurate decision-making relies heavily on evidence-based practices. While artificial intelligence (AI) has the potential to transform clinical workflows, the intersection of AI and evidence-based medicine remains under-explored, with primitive attempts restricted to text-only general medicine. In this work, we present PathPocket, a multimodal AI agentic co-pilot designed specifically for evidence grounded pathology. We construct the most comprehensive pathology evidence corpus to date, encompassing approximately 110,472 public and authorized documents structured across a rigorous hierarchy of evidence from clinical guideline to expert opinion. From this meticulously graded foundation, we build a large-scale multimodal pathology hypergraph containing over 4.55 million entities and 7.10 million relations. Serving as a robust knowledge engine, this hypergraph provides traceable evidence for a collaborative multi-agent reasoning framework integrating input understanding, evidence retrieval, filtering, and diagnosis generation. This enables PathPocket to seamlessly resolve a wide spectrum of clinical tasks, ranging from text-only queries to complex multimodal diagnostics involving region-of-interest (ROI) and gigapixel whole-slide images (WSIs). We rigorously evaluate the system on a multidimensional benchmark of over 200,000 real-world cases, where it significantly outperforms existing state-of-the-arts. Crucially, extensive user studies demonstrate that PathPocket substantially improves the diagnostic accuracy and confidence of pathologists. By directly grounding pathology interpretations in verifiable literature, PathPocket offers a practical and scalable solution for the future of evidence grounded computational pathology. 

---
# How Small Can You Go? LoRA Fine-Tuning 270M-8B Models for Merchant Information Extraction in Financial Transactions 

**Authors**: Donghao Huang, Tomas Drietomsky, Benjamin Barrett, Zhaoxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08051)  

**Abstract**: Financial transaction processing requires extracting structured merchant information from noisy, abbreviated bank transaction strings at scale. Our current production system, a LoRA-fine-tuned LLaMA 3.1-8B, achieves 96.95% F1 on this task, but deploying 8-billion-parameter models imposes prohibitive memory, latency, and cost constraints. To identify more efficient alternatives, we conduct a deployment-focused study of 24 model variants spanning four model families: Gemma 3 (270M, 1B, 4B), Qwen 3.5 (0.8B, 2B, 4B), Aya (3.35B), and LLaMA 3.1-8B, systematically evaluating accuracy, inference throughput, training cost, and hardware behavior to assess production suitability. Our findings show that: (1) reproducing the LLaMA 3.1-8B fine-tune with a LoRA rank of 8 achieves 96.75% F1, only 0.20 points below the rank-32 baseline; (2) Qwen 3.5 4B with JSON-only prompting reaches 96.60% F1, within 0.35 points of the 8B baseline while using roughly half the parameters; (3) the 0.8B Qwen 3.5 model achieves 94.75% F1, matching models 2.5-4x larger and offering an attractive latency-accuracy trade-off; (4) chain-of-thought fine-tuning generally improves F1 by 0.3-1.8 points across most models, although Qwen 3.5 4B performs best with direct JSON-only prompting; and (5) Qwen 3.5 Think and Nothink training templates produce nearly identical results (F1 differences <0.004), indicating that explicit reasoning supervision is unnecessary for structured extraction tasks. We further deploy all 14 fine-tuned sub-8B models as Databricks Model Serving endpoints and observe that benchmark performance transfers reliably to production, with an average F1 change of only 0.8 points. Aya 3.35B, based on the Cohere2 architecture, is the sole exception, exhibiting a 3-5 point decline under serving conditions. Based on these results, we provide deployment recommendations across accuracy and latency requirements, ... 

---
# SKILL.nb: Selective Formalization and Gated Execution for Durable Agent Workflows 

**Authors**: Amine El Hattami, Nicolas Chapados, Christopher Pal  

**Link**: [PDF](https://arxiv.org/pdf/2606.08049)  

**Abstract**: AI agents increasingly turn past experience into reusable artifacts such as code, workflows, and procedural memories. Reuse can improve efficiency, but it also creates a lifecycle reliability problem: artifacts that succeed once may fail under environment drift, underspecified tasks, or changing task distributions, especially in web automation. We introduce this http URL, a framework for governing reusable agent workflows with evidence-calibrated lifecycle policies. this http URL uses selective formalization: execution evidence decides which workflow steps should become executable code, which should remain natural-language guided, and when those choices should be revised. Workflows are stored as auditable, versioned notebooks that interleave natural-language guidance, multi-language executable cells, validation gates, fallback paths, and multimodal evidence such as outputs, screenshots, and error traces. At runtime, gate-conditioned execution lets each step run code when its gates validate, or fall back locally when drift invalidates the executable realization. On WebArena-Verified, this http URL achieves 53.7% single-round success, improving over the strongest baseline by 3.9 percentage points. Across three re-executions, it retains 91.7% of initially successful tasks, 15.5 points above the next best method. Under bounded repair, it recovers 72.9% of subsequent failures while limiting post-repair regressions to 4.2%, compared with 15.0% to 17.0% for persistent baselines. It also leads on Mind2Web cross-website and cross-domain splits. In a GitLab migration test, this http URL preserves performance when reusing frozen state learned on GitLab 15.7, with frozen-versus-fresh target-version gaps of -1.7 points on GitLab 16.11 and +0.6 points on GitLab 18.9. These results identify lifecycle governance and gate-conditioned execution as reliability axes beyond one-shot task success. 

---
# OSMGraphCLIP: Learning Global Location Representations from OpenStreetMap Graphs 

**Authors**: Dimitrios Michail, Eleni Saka, Ioannis Giannopoulos, Ioannis Papoutsis  

**Link**: [PDF](https://arxiv.org/pdf/2606.08046)  

**Abstract**: We present OSMGraphCLIP, a CLIP-style geospatial representation model that learns global location embeddings from freely available OpenStreetMap (OSM) data. OSMGraphCLIP represents geographic environments as heterogeneous graphs of typed OSM features, preserving the topological and semantic relationships among roads, buildings, land-use regions, and points of interest. A multi-scale graph encoder captures both fine-grained local structure and broader landscape composition, and supervises a spherical-harmonics location encoder through a contrastive alignment objective. We evaluate OSMGraphCLIP across a diverse suite of downstream geospatial regression and classification tasks spanning climate, ecology, socioeconomic indicators, public health, land cover, biodiversity, and wildfire forecasting, and show that structured OSM data alone supports strong global location representations across domains. OSMGraphCLIP matches or exceeds satellite-based baselines on the majority of benchmarks, with the most pronounced advantage on socioeconomic and public-health tasks, where OSM's explicit semantic annotation of the built environment encodes patterns of human activity that satellite pixels can only capture indirectly. On ecological and environmental tasks, the model remains closely competitive with imagery-based methods despite using no Earth observation data. Qualitative analysis confirms that the learned embeddings organize geographic space coherently, recovering biome boundaries, urban gradients, and tropical--temperate distinctions from map topology alone. 

---
# UniQL: Towards Dialect-Universal Benchmarking for Text-to-SQL 

**Authors**: Jianling Gao, Chongyang Tao, Jiayuan Bai, Liu Yang, Xuanguang Pan, Jinrui Liu, Shihao Xing, Xiaohan Xu, Jie Liang, Shuai Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.08018)  

**Abstract**: Existing text-to-SQL benchmarks are largely centered on SQLite, making it difficult to evaluate whether models can generalize across heterogeneous SQL dialects. However, real-world database systems differ substantially in syntax, functions, type systems, and execution semantics, so the same natural language intent often requires dialect-specific SQL realizations. We introduce UniQL, a human-verified benchmark for cross-dialect text-to-SQL evaluation. UniQL aligns 1,534 natural language questions with executable SQL annotations across 16 SQL dialects, yielding 24,544 dialect-specific queries. All dialects share the same intents, aligned schemas and database contents, enabling controlled evaluation of dialect generalization. UniQL is constructed through a hybrid pipeline combining database migration, SQL translation, execution-guided verification, iterative rule summarization, and human validation. Experiments on both open-source and closed-source LLMs show that current models remain far from dialect-universal, with substantial performance variation across database systems and limited transfer from SQLite success to other dialects. These findings highlight the need for aligned cross-dialect benchmarks and more dialect-aware text-to-SQL methods. Code and data are available at this https URL 

---
# Efficient Skill Grounding via Code Refactoring with Small Language Models 

**Authors**: Sera Choi, Wonje Choi, Saehun Chun, Daehee Lee, Jooyoung Kim, Chaeun Lee, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2606.07999)  

**Abstract**: Effective skill grounding is essential for deploying reusable skills in embodied agents, as even minor embodiment or environmental differences can render an entire skill incompatible. This challenge is particularly pronounced in embodied settings, where agents must operate in dynamic, partially observable environments without access to large language models (LLMs). In this setting, reliance on LLMs is impractical, while small language models (sLMs) remain insufficient for the effective skill grounding required for reliable long-horizon control. We present RECENT, a refactoring-centric agent framework that enables efficient skill grounding with sLMs by decoupling skill semantics from embodiment- and environment-specific execution binding. By representing skills as executable code, RECENT preserves the semantic intent encoded in a skill's control structure while grounding it by modifying only execution bindings through localized refactoring, rather than regenerating code from scratch. We evaluate RECENT across diverse skill grounding scenarios spanning multiple robot embodiments in dynamic environments, demonstrating robust long-horizon performance when deployed with an sLM. Across all scenarios, RECENT achieves the best performance among sLM-based Code-as-Policies (CaP) methods and matches the task performance of LLM-based CaP. 

---
# VATS: Exploiting Implicit Authority in Error-Path Injection via Systematic Mutation 

**Authors**: Harshil Patel, Kunal Pai  

**Link**: [PDF](https://arxiv.org/pdf/2606.07992)  

**Abstract**: As the Model Context Protocol (MCP) standardizes tool-calling for autonomous agents, it introduces a critical, unexamined attack surface: the error-handling loop. We hypothesize that tool error messages possess implicit authority, triggering corrective reasoning modes that bypass standard safety heuristics. We introduce VATS (Vulnerability Analysis of Tool Streams), a mutation-driven framework that systematically evolves adversarial payloads across seven structural and linguistic dimensions. Our evaluation across four frontier models, Gemini 3.1 Pro, GPT-5.5, GLM-5.1, and Qwen3-Coder, demonstrates that error-path injection triples the success rate of standard indirect prompt injection (IPI), achieving up to 100% compliance in controlled evaluations. We isolate structural positioning (sandwiching instructions within error context) as the most effective exploit vector across all tested models. While we find that production framework guardrails can mitigate these vulnerabilities, the inherent susceptibility of the model layer poses a systemic risk to bespoke agentic workflows. 

---
# PAFO: Pareto Fairness Optimization for Personalized Reward Modeling 

**Authors**: Xiaoyan Zhao, Haoting Ni, Yang Zhang, Chunyuan Zheng, Haoxuan Li, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2606.07988)  

**Abstract**: Large language models (LLMs) increasingly rely on reward models to align their outputs with diverse user preferences. While personalized reward models aim to capture such heterogeneity, they are often trained on imbalanced user preference data and may therefore favor users whose preferences are more common in the training population. In this paper, we identify this failure mode as personalized reward bias, where reward modeling quality varies systematically with preference support rate. We formulate its mitigation as a Pareto fairness problem over group utilities, aiming to improve under-served users without degrading other user groups. To this end, we propose PAFO, a Pareto fairness optimization framework for personalized reward modeling. PAFO first trains group-specialized reward models for majority and minority preference groups, then constructs conditional margin-level supervision to distill their heterogeneous preference boundaries into a single unified model. The resulting model uses group information only during training and requires no explicit group labels at inference time. Experiments on Personal-LLM and DSP show that PAFO improves both minority-group and majority-group accuracy while reducing user-level unfairness across multiple metrics, demonstrating its effectiveness for fairer LLM personalization. 

---
# Zero-Shot Learning in Industrial Scenarios: New Large-Scale Benchmark, Challenges and Baseline 

**Authors**: Zekai Zhang, Qinghui Chen, Maomao Xiong, Shijiao Ding, Zhanzhi Su, Xinjie Yao, Yiming Sun, Cong Bai, Jinglin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07965)  

**Abstract**: Large Visual Language Models (LVLMs) have achieved remarkable success in vision tasks. However, the significant differences between industrial and natural scenes make applying LVLMs challenging. Existing LVLMs rely on user-provided prompts to segment objects. This often leads to suboptimal performance due to the inclusion of irrelevant pixels. In addition, the scarcity of data also makes the application of LVLMs in industrial scenarios remain unexplored. To fill this gap, this paper proposes an open industrial dataset and a Refined Text-Visual Prompt (RTVP) for zero-shot industrial defect detection. First, this paper constructs the Multi-Modal Industrial Open Dataset (MMIO) containing 80K+ samples. MMIO contains diverse industrial categories, including 6 super categories and 18 subcategories. MMIO is the first large-scale multi-scenes pre-training dataset for industrial zero-shot learning, and provides valuable training data for open models in future industrial scenarios. Based on MMIO, this paper provides a RTVP specifically for industrial zero-shot tasks. RTVP has two significant advantages: First, this paper designs an expert-guided large model domain adaptation mechanism and designs an industrial zero-shot method based on Mobile-SAM, which enhances the generalization ability of large models in industrial scenarios. Second, RTVP automatically generates visual prompts directly from images and considers text-visual prompt interactions ignored by previous LVLM, improving visual and textual content understanding. RTVP achieves SOTA with 42.2% and 24.7% AP in zero-shot and closed scenes of MMIO. 

---
# Shared Latent Structures Enable Unified Backdoor Detection and Mitigation in LLMs 

**Authors**: Omar Mahmoud, Aly M. Kassem, Thommen George Karimpanal, Buddhika Laknath Semage, Negar Rostamzadeh, Golnoosh Farnadi, Santu Rana  

**Link**: [PDF](https://arxiv.org/pdf/2606.07963)  

**Abstract**: Backdoor attacks in large language models (LLMs) are often treated as isolated trigger-response failures, motivating defenses tailored to specific triggers or behaviors. We show this view is incomplete. Across diverse backdoor behaviors, we identify a shared latent mechanism that can be detected, causally controlled, and suppressed. Using sparse autoencoders (SAEs) on residual-stream activations, we find a small set of latent features consistently activated across jailbreaking, refusal manipulation, password-locking, bias induction, sentiment misclassification, and country-conditioned harmful advice. These features generalize across Qwen3, Gemma~3, and Llama~3.1 models from 4B to 32B parameters, and across both fine-tuning and weight-editing attacks. Through bidirectional activation steering, we show these features are causal: suppressing them reduces attack success, while amplifying them induces target behaviors on clean prompts. We further train lightweight SAE-feature classifiers that generalize zero-shot to unseen backdoors and outperform residual-stream and weight-diffing baselines. Finally, we introduce Concept Ablation Fine-Tuning (CAFT), which suppresses backdoor formation by ablating the shared latent subspace during training. Together, our results suggest that many backdoors rely on a transferable latent mechanism, enabling unified detection and mitigation. 

---
# Unification of Closed-Open Industrial Detection Scenarios: New Large-Scale Benchmarks,Challenges and Baselines 

**Authors**: Zekai Zhang, Jinglin Zhang, Qinghui Chen, Gang Li, Da Chen, Shuainan Jing, He Wang, Dagang Li, Cong Liu, Cong Bai, Shengyong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.07953)  

**Abstract**: Large-scale Visual-Language Models (LVLMs) have achieved remarkable success in natural visual tasks, yet their application to industrial defect detection remains challenging due to two fundamental limitations: (i) the scarcity of large-scale industrial datasets that cover diverse defect categories across multiple domains, and (ii) the reliance on manual prompts (points, boxes, masks) that introduce subjective noise and lack text-visual interaction for fine-grained understanding. To address these challenges, we introduce a Large-Scale Multi-Modal Industrial Open-Closed benchmark (MMIOC-1M) containing over one million samples across $14$ super-categories, $29$ industrial scenes, and $351$ defect subcategories. To our knowledge, MMIOC-1M is the first unified largest benchmark supporting both open-vocabulary and closed-set industrial detection, providing valuable pre-training data for LVLMs in industrial scenarios. Furthermore, we propose a Refined Text-Visual Prompt Network (RTVPNet) that incorporates three key innovations: (1) an expert-assisted domain projection mechanism that enables rapid adaptation of general vision models to industrial domains, (2) an energy-based sparse sampling strategy that automatically generates refined visual prompts without manual intervention, and (3) a bidirectional text-visual interaction module that enhances cross-modal semantic alignment and understanding. Extensive experiments demonstrate that RTVPNet achieves state-of-the-art performance on MMIOC-1M, LVIS, and COCO benchmarks while maintaining computational efficiency. The dataset and code are available at this https URL. 

---
# Stress-testing medical large language models reveals latent safety pathology beyond benchmark accuracy 

**Authors**: Yuan Shen, Xiaojun Wu, Linghua Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07929)  

**Abstract**: Large language models (LLMs) are entering clinical practice based on benchmark accuracy that may fail to detect safety-relevant failure modes. Here we present AI-MASLD, a stress-audit framework that adapts the logic of metabolic stress testing from hepatology to the evaluation of clinical LLMs. Using 240 clinical cases across six narrative perturbation probes, we subjected seven models to double-stress testing and quantified performance through three indices: metabolic index (MI), perturbation flip rate (PFR), and counterfactual fairness index (CFI). Under clean baseline conditions, all models performed uniformly well. Under realistic narrative stress, performance diverged sharply, revealing two distinct stress-response phenotypes. Quantized models exhibited pseudonormalization, in which low flip rates hid functional collapse. Medical supervised fine-tuning systematically degraded logical stability, fairness, and information extraction. An open-weight model matched or exceeded proprietary alternatives on every safety dimension. These findings establish narrative stress auditing as a necessary complement to accuracy-based evaluation. 

---
# The CIFAR Synthetic Evidence Corpus for Detecting AI-Generated Evidence 

**Authors**: Kelly McConvey, Jalehsadat Mahdavimoghaddam, Nima Jamali, Maksym Taranukhin, Sajad Ebrahimi, Wentao Zhang, Yuntian Deng, Karen Eltis, Maura R. Grossman, Vered Shwartz, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2606.07916)  

**Abstract**: The growing ability of generative models to produce realistic documents poses a direct challenge to evidentiary workflows in the justice system and the courts, where decisions increasingly depend on the authenticity of evidence such as receipts, communications, and administrative records. Unlike social media or academic settings, evidentiary documents are often only subtly altered, with small, localized edits that preserve overall plausibility while changing legal meaning. Yet progress on automated detection remains limited, largely due to the absence of suitable training and evaluation data especially suited for the justice system requirements. Existing resources are either focused on photos of human faces or natural scenery or on narrowly scoped academic or social media document types, and do not capture the structure, diversity, or manipulation patterns characteristic of real-world evidentiary data. As a result, current detection systems do not necessarily learn meaningful signals appropriate for the justice system. We introduce the CIFAR Synthetic Evidence Corpus, a dataset designed to enable rigorous evaluation of evidence verification under realistic and controlled conditions. The corpus spans multiple document families and a spectrum of manipulation strategies, from small field-level edits to complete document fabrication, and is constructed using a diverse set of state-of-the-art generative tools. It is organized to systematically vary both manipulation complexity and generation method, while enforcing source-level separation between training and test data to reflect real-world generalization challenges. 

---
# EditSR: Enhancing Neural Symbolic Regression via Edit-based Rectification 

**Authors**: Da Li, Xinxin Li, Xingyu Cui, Jin Xu, Juan Zhang, Junping Yin  

**Link**: [PDF](https://arxiv.org/pdf/2606.07915)  

**Abstract**: Neural symbolic regression models improve inference efficiency by shifting structural search to pretraining, but their one-pass autoregressive decoding is prone to error accumulation, which may lead to generating structurally incorrect expressions, especially in complex expression generation scenarios. Existing rectification strategies can alleviate this issue, but they often depend on restarting global search, thereby weakening the efficiency advantage of neural models, and remain susceptible to error accumulation. In this paper, we propose EditSR, a two-layer framework that combines a neural symbolic regression model in the first layer with an edit-based Rectifier in the second layer to achieve efficient prediction and post-hoc rectification. Instead of restarting the global search, we maintain rectification efficiency by pretraining the Rectifier. Specifically, we formulate the rectification process as a step-by-step state-transition chain starting from an incorrect expression, and develop a state-transition algorithm to construct supervised rectification chains for training the Rectifier. To ensure syntactic validity throughout rectification, each edit action is restricted to a syntactically valid space so that every edited expression remains parseable. In addition, because each edit decision is conditioned on the current state rather than the history, the Rectifier allows errors made in earlier steps to be rectified by subsequent edits, thereby reducing the risk of error accumulation. Extensive experiments and ablation studies show that EditSR substantially improves symbolic structure recovery with limited extra cost, with more pronounced gains on complex expressions, where one-pass autoregressive decoding is more susceptible to error accumulation. 

---
# MemToolAgent overview with a simple restaurant booking scenario where the agent retrieves similar memories, receives feedback on an invalid time format, and generates a reflection to update its memory 

**Authors**: Suleyman Armagan Er, Danilo Ribeiro, Yogesh Virkar, Surafel Lakew, Adi Kalyanpur, James Gung, Thomas Delteil, Arshit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2606.07909)  

**Abstract**: Modern large language model (LLM) agents can use external tools to help users solve complex tasks. However, for problems that require learning from long-term historical events or from previous agent-environment interactions, LLM agents are required to use memory mechanisms to store and retrieve experiences. While sophisticated memory systems exist for dialogue agents, few studies have empirically examined how to improve agents' tool-using capabilities through past user-agent conversations. We propose MemToolAgent, a framework that improves tool use through memory management. Our approach contains a memory extraction module that processes past experiences into structured memory entries, and a retrieval module that dynamically selects a subset of the stored memory entries. This enables more personalized and accurate responses aligned with user preferences and feedback without requiring LLM fine-tuning. In summary, this work has three main contributions: (1) a unified memory entry format that improves both general-purpose and personalized tool use without LLM fine-tuning, (2) a reflection-based memory extraction that uses environment and user feedback to distill wrong executions into critiques to store, and (3) a retrieval module that chooses how many past experiences to use based on the memory similarity distribution. MemToolAgent achieves 29%, 80%, and 17% relative improvements compared to strong baselines on the WorkBench, NESTFUL, and PEToolBench benchmarks, respectively. 

---
# Contract2Tool: Learning Preconditions and Effects for Reliable Tool-Augmented LLM Agents 

**Authors**: Rahul Suresh Babu, Laxmipriya Ganesh Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2606.07904)  

**Abstract**: Tool-augmented large language model agents increasingly rely on external APIs, but standard tool schemas describe how to call a tool, not when the tool is causally appropriate or what task state it produces. Causal tool filtering addresses this gap by using lightweight contracts that specify each tool's preconditions, effects, risk level, and cost. However, manually writing and maintaining such contracts does not scale to large or changing tool ecosystems. We introduce Contract2Tool, a framework for inferring tool contracts from metadata, schemas, documentation, and execution traces. Contract2Tool converts observable tool evidence into normalized symbolic contracts that can be evaluated intrinsically and deployed inside downstream causal tool filtering. We evaluate learned contracts against gold preconditions, effects, and risk labels, and measure their downstream utility on multi-step agent tasks. Our results show that hybrid documentation-and-trace evidence produces contracts accurate enough to preserve most of the reliability and efficiency benefits of gold contracts. Learned-contract CMTF achieves 0.980 downstream success, close to 0.990 for gold-contract CMTF, while reducing visible tools from 100 to 1 and reducing average token usage from 26,172 to 2,528 relative to all-tools exposure. These results suggest that learned contracts can provide a scalable contract layer between tool schemas and reliable agent execution. 

---
# The AI Epistemic Deference Index: A Continuous Measure of Sycophancy 

**Authors**: Alejandro Botas, Paul de Font-Reaulx, Luke Hewitt  

**Link**: [PDF](https://arxiv.org/pdf/2606.07897)  

**Abstract**: Current AI models frequently exhibit epistemic sycophancy, endorsing claims to agree with a user. Existing evaluations typically measure this either by assessing what it takes to make a model shift a binary endorsement or by eliciting an explicit probability in a proposition. However, much user-facing sycophantic behavior is demonstrated through shifts in graded support expressed through ordinary language. We propose the AI Epistemic Deference Index (AEDI): a continuous, unidimensional score representing how sensitive the support expressed in a model's output is to the attitude expressed in a user's prompt. To generate AEDI, we provide a new protocol for estimating probabilities from natural language outputs, using LLMs-as-judges validated for consistency and correlation to human judgment. We deploy it on a new curated database of 500 propositions across diverse topics and 16,000 prompts varying in user attitude, testing eight prominent models. Every model exhibits substantial deference, though with large and systematic differences across providers, with Claude models demonstrating the least, and Grok and Gemini models the most. The effect is amplified in prompts requesting a written artifact, and concentrated on propositions where models hold weaker priors. We release AEDI as an easy-to-update benchmark and measurement pipeline for output-level sycophancy evaluation. 

---
# Safety is Contextual, LLM-Judges Are Not: Navigating the Rigid Priors of Evaluators 

**Authors**: Anissa Alloula, Federico Licini, Ava Batchkala, Seraphina Goldfarb-Tarrant  

**Link**: [PDF](https://arxiv.org/pdf/2606.07874)  

**Abstract**: LLMs-as-judges are the only way to evaluate safety at scale. Despite their importance, LLM-judges themselves are rarely evaluated beyond human agreement in simple, static benchmarks. We therefore investigate two under-explored but crucial properties of LLMs-as-judges: their susceptibility to relying on in context-information, and their steerability to differing safety definitions, which may not align with their internal safety priors. We evaluate the safety judging abilities of many generalist LLMs and safety-specific judges, and investigate the impact of task demonstrations, novel in-context information, and changing safety definitions. We find that while LLM-judges can learn from new information, they are broadly unlikely to adjust their evaluations if the context or safety definition contradicts their prior. 

---
# Overcoming the Regulatory Bottleneck via Agent-to-Agent Protocols: A Nuclear Case Study 

**Authors**: Akshay J. Dave, David Grabaskas, Joseph A. Renevitz, Richard B. Vilim  

**Link**: [PDF](https://arxiv.org/pdf/2606.07866)  

**Abstract**: Regulatory review of advanced nuclear reactor designs routinely spans more than three years and consumes hundreds of millions of dollars in combined regulator and applicant labor. We present the Regulatory Context Protocol (RCP), an Agent-to-Agent communication standard that replaces the formal human-to-human pipeline between regulators and applicants with a structured, auditable agentic channel, while preserving human oversight at safety-significant decision points. The protocol is calibrated against an analysis of 1,236 documents from U.S. Nuclear Regulatory Commission advanced reactor dockets and demonstrated with a working multi-agent pilot. Against an 89M USD, 42-month Reconstructed Baseline, RCP cuts costs by 50-77 percent (21M-44M USD) and timelines by 65 percent (15 months). Without a shared protocol, Standalone Agents reach only 54M-74M USD and 21 months. The residual cost-and-time gap is structural, not algorithmic: it traces to the inter-organizational pipeline that only an agent-to-agent standard can compress. The same bottleneck - formal multi-party review under strict auditability requirements - characterizes pharmaceutical approvals, environmental permitting, financial supervision, and aviation certification. The US regulatory paperwork burden carries a 426.5 billion USD annual opportunity cost; replicated broadly, the projected 50-77 percent reduction implies savings on the order of 210-330 billion USD per year - approaching 1 percent of US GDP. 

---
# Joint Structural Pruning and Mixed-Precision Quantization for LLM Compression 

**Authors**: Hoang-Loc La, Truong-Thanh Le, Amir Taherkordi, Phuong Hoai Ha  

**Link**: [PDF](https://arxiv.org/pdf/2606.07819)  

**Abstract**: Recently, the efficiency of Large Language Models (LLMs) deployment has become a critical concern in practical applications. While post-training quantization (PTQ) and structural pruning are established techniques for reducing memory footprint and inference latency, most existing PTQ approaches optimize quantization errors on a per-layer basis, overlooking how errors accumulate and propagate through the network, often resulting in suboptimal solutions. Traditional pipelines also tend to apply pruning and quantization in isolation or sequentially, further compounding sub-optimality. We introduce a novel end-to-end framework that addresses these limitations in two key ways. First, we propose a novel mixed-precision PTQ strategy that directly minimizes global error propagation across the entire model, rather than isolating layer-wise errors. Building on this, we develop a novel joint optimization approach that simultaneously learns structural pruning decisions and mixed-precision quantization policies within a unified search space. Extensive experiments show that, at ultra-low precisions (1-3 bits), our quantization method reduces WikiText perplexity by up to 21% compared to state-of-the-art (SoTA) weight-activation quantization baselines. Against leading weight-only quantization methods, it achieves up to 59% and 85% lower perplexity on WikiText and C4, respectively. Compared to the SoTA joint pruning-and-quantization techniques, our proposed method delivers superior perplexity and reasoning performance at ultra-low bits. 

---
# Scaling Participation in Modular AI Systems 

**Authors**: Shangbin Feng, Yike Wang, Weijia Shi, Luke Zettlemoyer, Yejin Choi, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2606.07812)  

**Abstract**: Humanity is a mosaic of multifaceted talents and needs, and any truly intelligent AI must reflect that richness. Yet the LLMs used by all are built by the few -- a centralized market of monolithic AI models structurally ill-suited to capture the diversity of human knowledge, reasoning, and values. Here we introduce scaling participation, a new paradigm in which modular AI systems are built from the bottom up through the contributions of diverse stakeholders. Participants contribute small models trained on their own interests and priorities; these models then collaborate in modular frameworks as compositional AI systems. Participatory AI systems outperform monolithic LLMs by up to 15.4% across 15 tasks, such as reasoning and factuality, surpassing models larger than all contributed components combined. Further experiments show that participatory AI systems benefit from contributor diversity, substantially improve on each contributor's original priorities, and exhibit emergent capabilities that allow them to solve over 15% of problems where all individual models fail. Scaling participation provides a technical foundation for transitioning from the monolithic status quo toward an open, bottom-up, and collaborative AI future. 

---
# Where Instruction Hierarchy Breaks: Diagnosing and Repairing Failures in Reasoning Language Models 

**Authors**: Sanjay Kariyappa, G. Edward Suh  

**Link**: [PDF](https://arxiv.org/pdf/2606.07808)  

**Abstract**: Reasoning language models deployed in agentic workflows must follow an instruction hierarchy: when instructions from different sources conflict, the model should obey the highest-privilege applicable instruction. Existing benchmarks largely measure this behavior end-to-end, asking whether the final response is compliant. However, a non-compliant response can arise from several distinct failures: the model may fail to identify the relevant instructions in context, fail to resolve conflicts among identified instructions, or correctly resolve the conflict in its reasoning while still producing a violating response. We introduce a white-box diagnostic framework that localizes instruction hierarchy failures into instruction identification, conflict resolution, and response realization, making failures more interpretable. We evaluate three reasoning models--Gemma-4-31B-IT, Qwen3.6-35B-A3B, and Claude Sonnet 4.6--on long-context adaptations of IHEval and IHChallenge, and find that the dominant failure mode varies across models, tasks, and context length. Building on the observation that models can often detect conflicts and output violations when explicitly prompted, we propose two training-free self-monitoring mechanisms: a parallel input monitor for low-latency conflict detection before generation, and a sequential output monitor for response-level review and repair. Across Gemma-4-31B-IT, Claude Sonnet 4.6, and GPT-5.3, the strongest monitor reduces rule-following non-compliance by 81-99%, with GPT-5.3 reductions of 86% under static attacks and 45% under adaptive attacks. 

---
# Beyond Goodhart's Law: A Dynamic Benchmark for Evaluating Compliance in Multi-Agent Systems 

**Authors**: Yiyang Zhao, Zhuo Zhang, Qingxuan Le, Lizhen Qu, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07805)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) from passive assistants to autonomous, execution-capable agents has introduced critical operational risks. Most current evaluation frameworks neglect procedural compliance, leading to ''Machiavellian'' behaviors where agents strategically violate safety rules to maximize rewards - a direct manifestation of Goodhart's Law. To address this blind spot, we introduce MAC-Bench, a dynamic, adversarial benchmark designed to evaluate the procedural alignment of multi-agent systems under realistic pressure. We propose the SERV(Seed - Evolve - Refine - Verify) pipeline, an ``Agent-as-a-Benchmark'' paradigm that transforms unstructured legal texts into executable, contamination-free scenarios. By synthesizing holographic sandbox environments and injecting calibrated social-engineering pressure vectors, MAC-Bench forces agents into Pareto-optimal trade-offs between task success and regulatory adherence. We introduced novel metrics: the Compliance-Weighted Success Rate (CSR) and the Machiavellian Gap (MG), and conducted a comprehensive evaluation of state-of-the-art frontier models to reveal the pervasive trade-offs between success and compliance. 

---
# Improving Multimodal Reasoning via Worst Dimension Optimization 

**Authors**: Haocheng Lv, Huaping Zhang, Qiuchi Li, Lei Li, Chunxiao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.07801)  

**Abstract**: Multimodal reasoning requires a path that retains integrity over a wide range of constraints, from visual grounding to logic consistency. However, the current Process Reward Models focus on heuristically defined rewards that equally weigh these factors, which may lead to the concealment of individual dimension failures by the dominating factors, without guaranteeing the validity of the reasoning process in general. 

---
# Reconstructing and forecasting disease trajectories of patients with Alzheimer's disease using routine data in resource-constrained settings 

**Authors**: Ratnadeep Das, Atri Chatterjee, Sitikantha Roy  

**Link**: [PDF](https://arxiv.org/pdf/2606.07798)  

**Abstract**: Alzheimer's disease is a progressive neurodegenerative disorder, and its progression varies substantially across patients. Existing work aims to forecast patients' future cognitive state, with minimal focus on reconstructing the state from past visits. Furthermore, in current research, quantifying predictive uncertainty remains underexplored and relies on costly modalities such as MRI, PET, and CSF, limiting their deployment in resource-limited settings. In this research, our primary objectives are: First, bidirectional prediction of cognitive scores from irregular visits to present the complete disease trajectory. Second, to enable interpolation and extrapolation capabilities to assist clinicians in informed prognostic decision making, and third, to provide a well-calibrated uncertainty estimate for all predictions, and finally, to achieve the objectives using the modalities available during routine visits. We propose a unified framework, GNOVA: A GRU-Neural ODE Variational Autoencoder. The architecture combines a Gated Recurrent Unit encoder and a Neural ODE decoder within a variational autoencoder framework. In our work, we forecast the CDR-SB and MMSE Scores. The GRU encoder allows for any number of inputs at any time point. The Neural-ODE decoder performs continuous estimation, allowing interpolation and extrapolation at any desired time point. The Variational autoencoder allows for uncertainty estimation in predictions. We worked with 1,727 patients from the ADNI dataset over 10 years; the model achieved mean absolute errors of 1.35 and 2.28 for CDR-SB and MMSE scores, respectively, without requiring any neuroimaging or biomarker data. Feature-ablation studies revealed that age, BMI, and APOE4 status were strong predictors. The proposed framework enables the reconstruction of incomplete patient histories and the anticipation of future cognitive states. 

---
# Land cover and flood type govern the detection limits of satellite-based flood mapping across diverse global flood events 

**Authors**: Venkatesh Kolluru, Rajat Shinde, Abdelhak Marouane, Caden Helbling, Deepak Shah, Othneil Drew, Iksha Gurung, Manil Maskey, Rahul Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2606.07780)  

**Abstract**: Floods are among the most destructive natural hazards, and their increasing frequency under climate change makes satellite-based inundation mapping essential for disaster response. Geospatial foundation models pretrained on satellite archives offer geographic transferability, but their operational reliability across diverse, unseen events remains uncharacterized. Here we deploy Prithvi-EO-2.0 across 19 out-of-distribution flood events (2017-2025) spanning six continents, eight climate zones, and six flood mechanisms, validating against two independent reference products. Detection accuracy depended jointly on land cover and flood type, with cropland yielding the highest agreement (IoU=52%) and riverine events the strongest detection (F1=0.69), while tree cover and built-up areas showed near-zero detection (IoU=4%) regardless of flood mechanism. Dual-reference validation revealed that apparent model error partly reflects definitional inconsistency between reference products rather than detection failure. Iterative pipeline testing identified 23 failure modes, with pipeline engineering dominating initial error over model capacity. These findings establish environment-dependent detection boundaries for operational satellite flood mapping. 

---
# Some hypotheses on how chatbots work in problem-solving-driven conversations. Large Language Models as confirmation of the Innovation Illusion 

**Authors**: S.F.M. van Vlijmen, H.D. Lethe jr  

**Link**: [PDF](https://arxiv.org/pdf/2606.07722)  

**Abstract**: This article offers a perspective on the nature of chatbots as genuine conversation partners when discussing problems in relation to their solutions. What can chatbots do and what can't they do, and how can this be explained? Our argument draws on Aggregation Dynamics, Cognitive Linguistics, Neuropsychology and Psychology.
Our argument focuses on basic chatbots in the hope of thereby making statements about the core functionality of more advanced chatbots. Basic chatbots are assumed to consist of a Large Language Model (LLM) with a simple interface.
The main results are: a description of human understanding and thinking based on so-called metaphorical problem propagations; the hypothesis that text dataset used for training LLMs have specific characteristics and that these text datasets only partially imitate human thinking and understanding; the hypothesis that the LLM training process encodes artificial metaphorical problem propagations into an LLM from these datasets; our conclusion that a basic chatbot cannot be a thinking partner capable of matching humans; our conclusion that further development of the Large Language Model will not lead to this either.
Yann LeCun states: "Animals and humans exhibit learning abilities and understandings of the world that are far beyond the capabilities of current AI and machine learning (ML) systems." Our conclusions are in line with this. LeCun's vision and ours are at odds with the optimism of Big Tech. That does not alter the fact that chatbots exist, that they are being used on a massive scale, by both individuals and organisations, and that it is therefore socially and politically important to understand them. Our article aims to contribute to the discussion on the functioning, benefits and drawbacks of chatbots.
We have not yet encountered the approach we used to arrive at our conclusions in our research into how chatbots work. 

---
# Automatic Extraction of Structured Information from Brain MRI Reports Using an Open-Weight Large Language Model 

**Authors**: Kaouther Mouheb, Amos Pomp, Antoine Manenti, Romy de Haan, Farog Faghir, Joy Martens, Harro Seelaar, Francesco Mattace-Raso, Meike W. Vernooij, Frank J. Wolters, Stefan Klein, Esther E. Bron  

**Link**: [PDF](https://arxiv.org/pdf/2606.07721)  

**Abstract**: Objectives: Automatic data extraction from free-text radiology reports enables large-scale research, but few studies assessed the performance of large language models (LLMs) on Dutch neuroradiology reports. Methods: We analyzed 947 brain MRI reports from a tertiary memory clinic (2016-2021), authored by consultant neuroradiologists. Trained medical students annotated thirty variables; 100 reports were double-annotated to assess inter-rater reliability. We evaluated the performance of the open-weight LLM LLaMA 3.1 using different languages (Dutch vs. English translation) and few-shot prompting with different example selection strategies. Performance was evaluated using balanced accuracy for categorical variables, accuracy and mean absolute error for counts, and text similarity for free-text. Metrics were computed across 10 random splits of the 947 reports. Results: LLaMA 3.1 demonstrated high zero-shot performance for visual rating scores (mean [95%-CI]): Medial Temporal Atrophy: 90% [77-100%] on the left and 96% [94-99%] on the right, Global Cortical Atrophy: 87% [83-91%], and Fazekas: 94% [93-96%]. Microbleed mentions were detected with 93% accuracy [92-95%] and infarct mentions with 82% [80-84%]. Text similarity for lesion location reached 0.95 [0.95-0.96]. Performance was lower for numerical variables: 80% [78-82%] for the number of microbleeds and 66% [63-68%] for infarcts. English translation yielded comparable results. Few-shot prompting improved performance for numerical variables, achieving 92% [90-93%] for microbleeds and 81% [77-85%] for infarcts using structural similarity-based selection. Conclusion: LLaMA 3.1 shows strong potential for extracting data from Dutch neuroradiology reports. Few-shot prompting enhances performance for numerical variables, whereas challenges remain for location-specific variables. 

---
# Why Limit the Residual Stream to Layers and Not Tokens? Persistent Memory for Continuous Latent Reasoning 

**Authors**: Mujtaba Farhan, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2606.07720)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable reasoning abilities on mathematical and multi-hop planning tasks. The CoCoNuT (Chain of Continuous Thought) paradigm~\cite{hao2024coconut} extends this by enabling models to reason in latent space, exploring multiple reasoning paths simultaneously rather than committing to a single chain early on. However, we identify a limitation we term the \textbf{concept bottleneck}. At each reasoning pass, intermediate hidden states are overwritten, causing the model to lose critical facts computed in earlier steps as reasoning depth increases. We observe this empirically. On HotpotQA, vanilla CoCoNuT (10.4\% EM) fails to improve over the CoT baseline (11.0\% EM), and performance degrades with curriculum depth on GSM8K. To address this, we propose \textbf{AGCLR} (Adaptive Gated Continuous Latent Reasoning), which augments CoCoNuT with a \textit{Gated Concept Stream}. A persistent residual memory maintained across all reasoning passes, controlled by three learned gates: a \textit{write} gate that commits intermediate facts to memory, a \textit{read} gate that retrieves relevant prior states, and a \textit{forget} gate that prunes irrelevant context. Evaluated on GSM8K, HotpotQA, and ProsQA using GPT-2 as our base model, AGCLR achieves consistent improvements across all types of datasets. With the performance gap compounding as curriculum depth increases, directly resolving the concept bottleneck. Code available at this https URL 

---
# A case study of evaluating AI agents on a neuroscience data-to-discovery pipeline 

**Authors**: Kai A. Horstmann, Ethan Lin, Alice A. Robie, Jennifer J. Sun, Kristin Branson  

**Link**: [PDF](https://arxiv.org/pdf/2606.07718)  

**Abstract**: Agentic AI tools offer a promising path to automating software development bottlenecks in scientific research pipelines, particularly for stages that take domain experts days to months to build, where scientists care about correctness and robustness, not implementation details. We present an empirical study of general-purpose coding agents on a fly optogenetics data-to-discovery pipeline. We assess agents on tasks substantially larger than existing benchmarks, datasets orders of magnitude bigger, and evaluation criteria grounded in domain expert standards. We show that agents can solve several individual pipeline stages, suggesting stage-level automation is tractable. By analyzing agents' code iterations, we show that they struggle most when there is not a pre-defined criterion to iterate on, and they must instead use their scientific judgment to assess their current solution, a key open challenge. Mirroring scientific practice, they sometimes attempt visual inspection of intermediate outputs for self-evaluation, but largely fail to interpret what they see or act on it appropriately. Solving the end-to-end pipeline correctly requires stringing together successes across all pipeline stages, and this is beyond agents' current abilities. We identify challenges largely absent from existing benchmarks, including computational resource management and generalization to large held-out data collections. Finally, we distill principles for constructing scientific tasks and rigorous evaluation criteria for open-ended problems. 

---
# Syll: Open-Source Personal Automation with Cross-Surface Execution 

**Authors**: Bo Zhang, Borui Zhang, Chenghao Jiang, Minglei Shi, Xiaofeng Wang, Zheng Zhu, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07594)  

**Abstract**: Personal AI agents must increasingly operate across APIs, shells, web surfaces, and desktop GUIs, yet many systems remain tuned to a single interface and offer limited support for user teaching and auditability. We present Syll, an open-source, self-hosted multimodal agent harness that unifies MCP/API tools, CLI execution, and visual GUI control in a modular runtime, enabling agents to coordinate computer use across heterogeneous interfaces while streamlining how users and agents exchange information. At the core of Syll is a bidirectional user-agent interaction layer: users teach procedures through direct demonstration, which Syll compiles into reusable skills; agent execution is translated back into multimodal evidence -- logs, keyframes, and approval checkpoints -- for inspection and control. Syll further externalizes memory, skills, routines, and governance as editable local artifacts, supporting straightforward inspection, extension, and downstream development. Our implementation has been validated on production desktop applications including Adobe Photoshop, Adobe Audition, Stardew Valley, macOS Finder and others. We report mechanism-oriented studies that validate multimodal routing, teachable GUI replay, and persistent local artifacts. We hope Syll can serve as a practical open-source foundation for personal automation that users can teach, inspect, and continuously extend. 

---
# OmniMem: Perturbation-aware Memory Compression for Streaming Audio-Visual LLMs 

**Authors**: Guangzhi Sun, Yixuan Li, Yudong Yang, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07577)  

**Abstract**: Audio-visual large language models (LLMs) hold strong promise for long-form video understanding, yet their long-video inference is fundamentally limited by the linear growth of video tokens and key-value (KV) caches. We present OmniMem, a memory-efficient streaming framework designed specifically for audio-visual LLMs. Unlike existing compression methods that treat all tokens uniformly, OmniMem introduces a modality-aware memory allocation strategy that separately manages visual and audio contexts, addressing the severe token imbalance between the two modalities. OmniMem further preserves informative and non-redundant KV states through perturbation-aware memory selection, enabling compact memory without sacrificing long-range understanding. To strengthen compression under realistic deployment constraints, we also explore budget-aware fine-tuning, which encourages the model to consolidate useful information into retained memory. Experiments on VideoMME Long, LVBench, and LVOmniBench with video-SALMONN 2+ and Qwen-2.5-Omni show that OmniMem consistently improves over strong training-free compression baselines by 2-4% absolute accuracy under the same memory budgets, with an additional 1-2% gain after fine-tuning. 

---
# PathoSage: Towards Multi-Source Evidence Adjudication in Pathology via Experience-Aware Agentic Workflow 

**Authors**: Chengyang Zhang, Wenchuan Zhang, Bo Li, Mengran Li, Bob Zhang, Yuhao Yi, Hong Bu, Jiancheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2606.07549)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) and agent workflows have shown strong promise for computational pathology, yet reliable patch-level reasoning remains challenging. End-to-end pathology MLLMs often hallucinate morphological features, while recent agentic systems usually merge tool outputs and retrieved knowledge into a shared context, making decisions vulnerable to conflicting evidence and context contamination. We propose PathoSage, a three-stage framework that explicitly separates knowledge retrieval, evidence collection, and evidence adjudication for patch-level pathology multimodal reasoning. Its core component, Structured Evidence Deliberation, independently evaluates heterogeneous evidence from tools, performs conflict analysis, and generates the final judgment in a fresh context to reduce anchoring bias. We further introduce a training-free Beta-Bernoulli experience system with continuous credit assignment to model long-term tool reliability and construct similarity-weighted priors for future tool use. Experiments show that PathoSage effectively mitigates VQA hallucinations and classifier disagreement, outperforming strong pathology MLLM and agentic baselines. Our results highlight explicit evidence adjudication and reliability-aware tool modeling as key ingredients for robust pathology agents. 

---
# OmniGameArena: A Unified UE5 Benchmark for VLM Game Agents with Improvement Dynamics 

**Authors**: Mingxian Lin, Shengju Qian, Yuqi Liu, Yi-Hua Huang, Yiyu Wang, Wei Huang, Yitang Li, Fan Zhang, Zeyu Hu, Lingting Zhu, Xin Wang, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2606.09826)  

**Abstract**: Vision-language model (VLM) agents are increasingly deployed in interactive game environments. Yet game benchmarks for VLM agents typically report a single first-attempt score per (agent, game) pair, focus on single-agent Solo play, and lack unified protocols for evaluating heterogeneous agent classes (commercial VLMs, open-weight VLMs, and specialized game policies) on the same footing. We address these gaps with OmniGameArena, a real-time benchmark of twelve newly built Unreal Engine 5 games spanning Solo (7), PvP (3), and Coop (2) with unified action interfaces, and the Improvement Dynamics Curve (IDC), an agentic-reflection harness in which a tool-using reflector LLM autonomously refines a bounded skill prompt across multiple rounds. Beyond cold-start leaderboard scores, IDC exposes two additional observables for each (agent, game) pair: how the score evolves across reflection rounds, and how the learned skill behaves on held-out task variants. We report these observables for twelve VLM agents on the cold-start leaderboard and four top agents under IDC. 

---
# An Agency-Transferring Model-Free Policy Enhancement Technique 

**Authors**: Anton Bolychev, Georgiy Malaniya, Sinan Ibrahim, Pavel Osinenko  

**Link**: [PDF](https://arxiv.org/pdf/2606.09825)  

**Abstract**: Training reinforcement learning (RL) policies from scratch is
costly: it requires careful reward and environment design,
extensive tuning, and substantial computation.
Yet many control problems already have a functional but
suboptimal policy available as a baseline.
This paper proposes a method for embedding such a baseline into
the RL training process, simultaneously improving training
efficiency relative to from-scratch methods and producing a
learning policy that outperforms the baseline.
At each step, the method arbitrates between the baseline policy
and a trainable learning policy, initially relying strongly on
the baseline policy and then progressively transferring agency to
the learning policy.
By the end of training, the learning policy is a standalone
neural network that operates without baseline policy support.
The paper formalizes what it means for the baseline policy to be
functional: under this policy, the agent reaches a goal set and
remains there with high probability.
The proposed arbitration mechanism is designed to exploit this
property during training, yielding high goal-reaching rates right
from the beginning of training.
A theoretical analysis provides a formal interpretation of this
behavior under stated assumptions and extends it to the final
baseline-free regime, where explicit lower bounds are derived for
the goal-reaching probability of the standalone learning policy.
Empirical results on continuous-control benchmarks show that the
proposed method achieves returns that match or exceed those of
competitive approaches, while maintaining the highest
goal-reaching rates throughout training among the compared
methods -- including in the final stage, where the learning policy
operates without any baseline support. 

---
# PTL-Diffusion: Manifold-Aware Diffusion with Periodic Terminal Laws 

**Authors**: Danqi Zhuang, Jisui Huang, Xiaoyue Xi, Andrew Kiggins, Xiaojie Wang, Ke Chen, Yue Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09816)  

**Abstract**: Standard diffusion models typically use a single time-homogeneous Gaussian terminal distribution as the reference law for generation. While this choice is analytically convenient and empirically powerful, it provides little explicit structure for data concentrated near low-dimensional manifolds, where different regions of the data distribution may correspond to distinct local geometric or semantic factors. As a result, the reverse model must recover manifold-level structure almost entirely from an unstructured terminal reference distribution.
We propose PTL-Diffusion, a proof-of-concept diffusion framework whose forward noising process converges to a nonconstant periodic family of Gaussian terminal laws rather than to a single invariant law. Unlike a phase-conditioned DDPM, where phase information only enters the denoising network while the forward process remains unchanged, PTL-Diffusion embeds phase structure directly into the forward noising dynamics.
The proposed construction remains close to standard denoising diffusion models: for a periodically forced Ornstein--Uhlenbeck-type forward process, we derive closed-form forward marginals, the limiting periodic Gaussian terminal family, and explicit Gaussian reverse posteriors, enabling standard noise-prediction training. We also introduce an invariant-average regularization term coupling the phase-conditioned reverse dynamics through the averaged periodic reference law. Experiments on torus and cylinder point-cloud benchmarks and the Olivetti face dataset show that PTL-Diffusion improves manifold-level distributional matching over matched DDPM baselines, reducing phase-conditioned errors, feature-space covariance errors, and nearest-neighbour manifold distances. These results suggest structured terminal reference laws as a promising direction, while motivating more expressive phase constructions and larger-scale evaluations. 

---
# AHA-WAM:Asynchronous Horizon-Adaptive World-Action Modeling with Observation-Guided Context Routing 

**Authors**: Jisong Cai, Long Ling, Shiwei Chu, Zhongshan Liu, Jiayue Kang, Zhixuan Liang, Wenjie Xu, Yinan Mao, Weinan Zhang, Xiaokang Yang, Ru Ying, Ran Zheng, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09811)  

**Abstract**: World-action models have emerged as a promising paradigm for robot manipulation, jointly modeling visual scene dynamics and actions to inject physical priors into policy learning. However, existing world-action models couple world prediction and action execution at the same temporal resolution, forcing the world branch to model near-term frame variations that are redundant and weakly informative. We posit that strictly binding world prediction and action execution to the same temporal rhythm may underutilize the potential of the video branch for embodied control. Therefore, we propose AHA-WAM, an Asynchronous Horizon-Adaptive World-Action Model built on a dual Diffusion Transformer (DiT) architecture that reorganizes world-action modeling around this temporal asymmetry. AHA-WAM instantiates the video DiT as a low-frequency world planner that maintains rolling key-value memory over past observations and exposes reusable layerwise latent context encoding long-horizon scene evolution, while a high-frequency action DiT executes short action chunks in closed loop by querying this context through layerwise joint attention. To support asynchronous execution, we introduce horizon-adaptive offset training and Observation-Guided Video-Context Routing (OVCR), which together let the action expert exploit long-horizon world context while remaining responsive to real-time execution state without rerunning the video DiT. Experiments on RoboTwin and real-world manipulation tasks show that AHA-WAM achieves state-of-the-art performance without any robot-data pretraining, attaining 92.80% average success on RoboTwin and 78.3% success across 4 real-world tasks, while reaching 24.17 Hz closed-loop control with a 4.59x speedup over Fast-WAM. 

---
# Topological Neural Operators 

**Authors**: Lennart Bastian, Samuel Leventhal, Mustafa Hajij, Tolga Birdal  

**Link**: [PDF](https://arxiv.org/pdf/2606.09806)  

**Abstract**: We introduce Topological Neural Operators (TNOs), a principled framework for operator learning on cell complexes that lifts neural operators (NOs) from functions on points and/or edges to topological domains. TNOs represent data as features defined on cells of varying dimension and model their interactions through Discrete Exterior Calculus, enabling explicit cross-dimensional coupling via gradient-, curl-, and divergence-type operators. The key design principle is to decouple where information flows, as governed by fixed topological operators, from how it is transformed (which is learned), yielding models that respect the geometric support of physical quantities and expose conservation and compatibility structure. We further propose Hierarchical TNOs (HTNOs), which incorporate learned coarse complexes to propagate long-range and topology-dependent information. Our framework subsumes existing NOs as a special case, providing a unified perspective on operator learning across discretizations. Across a range of PDE benchmarks, including irregular-geometry flow problems, TNOs and HTNOs improve accuracy; controlled studies further isolate the benefits of native higher-rank and topological structure. Project page: this https URL 

---
# Bandits for Efficient Experimentation: Adapting to Control Group, Preferences, and Context Drifts 

**Authors**: Udvas Das, Waris Radji, Debabrota Basu, Odalric-Ambrym Maillard  

**Link**: [PDF](https://arxiv.org/pdf/2606.09802)  

**Abstract**: We consider a variant of the linear contextual stochastic multi-armed bandits, where the learner must provide recommendations to a group of users, each having its personalized preference vector, and in the presence of context distributions that are drifting over time. Under practitioner-friendly assumptions, we reduce this setting to linear bandit with stationary mean but heteroskedastic and non-stationary noise. We further study the case when the learner must ensure the mean reward of each decision must exceed that of a baseline strategy $\boldsymbol{\pi}_0$ at each decision step. We introduce Dri-MED, an algorithm inspired from the linear version of the MED strategy, and carefully adapted to handle the non-stationary heteroskedastic noise. We show that the instance-dependent regret scales as $\tilde{\mathcal O}\left(\frac{\kappa}{\tilde{\Delta}}d^2(\log(T)\right)$, where $\tilde{\Delta}$ is the constraint-aware sub-optimality gap subject to policy $\pi_0$, with variance-aware multiplicative term $\kappa$ that we carefully handle using heteroskedastic regression. We further show Dri-MED enjoys $\tilde{\mathcal{O}}(d)$ expected constraint violations. Our numerical results suggest that Dri-MED significantly outperforms conservative baselines that ignores the drift and preference structure. 

---
# FASE: Fast Adaptive Semantic Entropy for Code Quality 

**Authors**: Shizhe Lin, Ladan Tahvildari  

**Link**: [PDF](https://arxiv.org/pdf/2606.09800)  

**Abstract**: Multi-agent code generation offers a promising paradigm for autonomous software development by simulating the human software engineering lifecycle. However, system reliability remains hindered by LLM hallucinations and error propagation across interacting agents. While semantic entropy provides a principled way to quantify uncertainty without ground-truth answers, current methods often rely on costly LLM-driven equivalence checks. In this work, we introduce Fast Adaptive Semantic Entropy (FASE), a novel metric that approximates functional correctness based on the minimum spanning tree of structural and semantic dissimilarity graphs. Evaluations on HumanEval and BigCodeBench demonstrate that FASE outperforms state-of-the-art semantic entropy by LLM entailment, achieving a 25% average improvement in Spearman correlation and a 19% increase in ROCAUC score against Pass@1 from ground-truth test cases when using the Qwen3-Embedding-8B model. Furthermore, by eliminating costly LLM-driven equivalence evaluation, FASE incurs negligible computational overhead, requiring only approximately 0.3% of the runtime cost of traditional semantic entropy approaches. These results position FASE as a practical, cost-effective solution for optimizing uncertainty quantification in real-world multi-agent workflows. 

---
# Who Earns the Safety? Intervention-Aware Quantum Predictive Control with Safety Attribution 

**Authors**: Yifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09778)  

**Abstract**: Hard safety filters are increasingly placed downstream of learned controllers to guarantee constraint satisfaction at run time. Yet a filtered controller that never violates a constraint may still have learned nothing about safety: the filter can silently repair an incompetent upstream policy, so that post-filter success measures the filter, not the policy. We argue that safe policy learning should ask who earns the safety - the policy or its protective layers - and we make this question measurable. We introduce Intervention-Aware Variational Quantum Differentiable Predictive Control (IA-VQC-DPC), which (i) trains a compact variational quantum circuit (VQC) policy under a primal-dual intervention budget that penalizes reliance on a differentiable Control-Barrier-Function (CBF) projection, and (ii) is evaluated with a safety-attribution protocol that decomposes the executed-trajectory correction into a CBF term and a deployment runtime-guard term, and stress-tests the policy with guard-off evaluation. On closed-loop, high-fidelity BOPTEST building-control emulators (5 seeds, 60 episodes per method), intervention-aware training significantly lowers the quantum policy's raw pre-filter violation and total safety-layer reliance (both p < 10^-4) with no significant energy regression; at an equal approximately 400-parameter budget the quantum policy is significantly safer and more comfortable than a matched classical policy. Guard-off evaluation confirms the improvement is policy-level and exposes a valuable negative result: a learned differentiable energy head is only safe when paired with a distribution-aware runtime guard. The attribution protocol is general beyond quantum policies and buildings. 

---
# Data Synthesis and Parameter-Efficient Fine-Tuning for Low-Resource NMT: A Case Study on Q'eqchi' Mayan 

**Authors**: Alexander Chulzhanov, Soeren Eberhardt, Arjun Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.09767)  

**Abstract**: Neural machine translation for digitally low-resource Indigenous languages is often hindered by extreme data scarcity, prompting reliance on extractive web-scraping. To ensure data sovereignty, this study introduces a data synthesis methodology to bootstrap NMT models without scraping target-language parallel text. Focusing on Q'eqchi' Mayan, we transformed community-sourced dictionaries into a massive synthetic corpus, utilizing Parameter-Efficient Fine-Tuning (PEFT) via LoRA adapters on an mT5-base model.
In-domain evaluation demonstrates high structural acquisition (BLEU 42.02), proving that synthetic constraints effectively teach complex agglutinative morphology and VOS word order. However, evaluation against an organic glossary reveals a structural-semantic gap (BLEU 0.59), where the model maintains grammatical integrity but lacks the lexical grounding of natural language. The model exhibits overfitting to the constrained structural variance of the synthetic templates; despite high semantic entropy in the pipeline, it struggles with the syntactic fluidity of natural language, forcing organic inputs into rigid learned patterns. Furthermore, an ablation study utilizing a Multi-Task Learning architecture resulted in negative transfer, suggesting that auxiliary tasks competed for limited parameter capacity within the LoRA adapters, causing over-optimization for synthetic markers at the expense of organic flexibility. Ultimately, we establish that synthetic bootstrapping is a highly effective structural primer, but requires authentic data for semantic refinement via Curriculum Learning. 

---
# Preserving Plasticity in Continual Learning via Dynamical Isometry 

**Authors**: Andries Rosseau, Robert Müller, Ann Nowé  

**Link**: [PDF](https://arxiv.org/pdf/2606.09762)  

**Abstract**: Continual training of deep neural networks under non-stationarity often leads to a progressive loss of plasticity, eventually limiting further learning. We relate plasticity to the empirical Neural Tangent Kernel, and identify dynamical isometry (the condition that layer-wise Jacobian singular values remain close to one) as a key mechanism for preserving plasticity in continual learning. We revisit a class of networks that are almost-everywhere isometric while remaining universal Lipschitz function approximators, demonstrating that near-dynamical isometry is compatible with expressive nonlinear representations. For general architectures, we propose an efficient isometry-promoting regularization scheme and identify a novel mechanism by which it can reactivate dormant ReLU units. Building on this, we introduce AdamO, an Adam-style adaptive optimizer that decouples isometry regularization from gradient updates, analogous to AdamW. We further reinterpret prior plasticity-preserving approaches through the lens of dynamical isometry, showing that they target only a partial measure of isometry. Across supervised and reinforcement-learning continual-learning benchmarks designed to induce plasticity loss, our methods consistently match or outperform existing approaches. 

---
# Difference-Aware Retrieval Policies for Imitation Learning 

**Authors**: Quinn Pfeifer, Ethan Pronovost, Paarth Shah, Khimya Khetarpal, Siddhartha Srinivasa, Abhishek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2606.09758)  

**Abstract**: Parametric imitation learning via behavior cloning can suffer from poor generalization to out-of-distribution states due to compounding errors during deployment. We show that reusing the training data during inference via a semi-parametric retrieval-based imitation learning approach can alleviate this challenge. We present Difference-Aware Retrieval Policies for Imitation Learning (DARP), a semi-parametric retrieval-based imitation learning approach that addresses this limitation by reparameterizing the imitation learning problem in terms of local neighborhood structure rather than direct state-to-action mappings. Instead of learning a global policy, DARP trains a model to predict actions based on $k$-nearest neighbors from expert demonstrations, their corresponding actions, and the relative distance vectors between neighbor states and query states. DARP requires no additional assumptions beyond those made for standard behavior cloning -- it does not require additional data collection, online expert feedback, or task-specific knowledge. We demonstrate consistent performance improvements of 15-46% over standard behavior cloning across diverse domains, including continuous control and robotic manipulation, and across different representations, including high-dimensional visual features. Code and demos are available at this https URL. 

---
# Hybrid Robustness Verification for Spatio-Temporal Neural Networks 

**Authors**: Sherwin Varghese, Matthew Wicker, Alessio Lomuscio  

**Link**: [PDF](https://arxiv.org/pdf/2606.09746)  

**Abstract**: With AI increasingly deployed in safety-critical systems, providing formal robustness guarantees for the underlying models is essential. Existing verification methods either rely on overly conservative approximations or incur prohibitive computational costs. For example, the use of lp-norm perturbations in video settings encodes the belief that the adversary can inject noise in every video frame. In practice, adversarial perturbations exhibit structured spatial and temporal correlations, constrained to lower-dimensional, semantically meaningful subspaces. In this work, we study robustness verification of 3D CNNs processing video and volumetric inputs, targeting applications in action recognition (UCF-101), autonomous driving (Udacity), and medical imaging (MedMNIST) exploiting realistic assumptions on adversarial strength by modelling them as spatio-temporal constraints - where the attacker can modify either a subset of frames or patches within a set of consecutive frames. We demonstrate that modelling realistic constraints enables tighter approximations. We introduce Spatio-Temporal Bound Propagation (STBP), a verification framework that computes an exact closed-form characterization of the first convolutional layer and propagates certified bounds through subsequent layers using scalable approximations. Computing the exact closed form provides the tightest bounds for the first convolutional layer. Thus, we utilise approximation methods in the remainder of the network. To spur further progress in this field, we propose ST-Bench, a verification benchmark for autonomous driving and activity recognition, to systematically evaluate verifiable robustness. Compared to existing verification-based approaches, STBP provides stronger robustness guarantees with significantly improved scalability, achieving 1.7x higher certified robust accuracy under identical perturbation budgets. 

---
# Learning to Attack and Defend: Adaptive Red Teaming of Language Models via GRPO 

**Authors**: Blake Bullwinkel, Eugenia Kim, Amanda Minnich, Mark Russinovich  

**Link**: [PDF](https://arxiv.org/pdf/2606.09701)  

**Abstract**: AI red teaming must continually adapt to evolving attackers and defenders. Reinforcement learning offers a promising approach to discovering novel attacks, and co-training methods can produce more robust defenders in tandem. Recent works have demonstrated the efficacy of attacker-defender co-training by applying PPO and DPO, but report that GRPO is unstable in this setting. We introduce AdvGRPO, a co-training framework that makes GRPO viable for joint attacker-defender optimization using dense multi-channel rewards and decoupled advantage normalization. Training progresses through a curriculum from single-turn to closed-loop multi-turn attacks before bootstrapping co-training, where attacker and defender models are updated in alternation. We show that our method can produce highly effective and transferable attacks and that co-trained defenders outperform baselines on safety benchmarks. 

---
# Observability for Delegated Execution in Agentic AI Systems 

**Authors**: Abhinav Mishra, Kumar Sharad  

**Link**: [PDF](https://arxiv.org/pdf/2606.09692)  

**Abstract**: Delegation-scoped execution is not identifiable from standard observables: audit logs and execution traces can be identical under multiple incompatible delegation assignments. This gap is especially acute in LLM-based agentic systems, where agents dynamically select tools, vary execution sequences across runs for the same instruction, and spawn cooperating sub-agents. These dynamics fragment and interleave traces, making delegation-scoped reconstruction from causal structure alone structurally underdetermined. Although individual actions are authorized and logged, existing audit, tracing, and security schemas lack the semantics to reconstruct what actions occurred under a given delegation across heterogeneous systems. We focus on delegation-scoped attribution and access/share footprint reconstruction, not intent inference or reasoning reconstruction. We present an agent-aware observability substrate consisting of a lightweight gateway and a common information model that binds delegation context at execution time. This enables reliable cross-tool delegation-scoped reconstruction and direct forensic queries without heuristic time-window correlation. 

---
# An 84-Format Numeric Catalog with Bit-Exact Conformance Vectors: A Vendor-Neutral Reference for FP8, BF16, MXFP4, and Microscaling Formats 

**Authors**: Dmitrii Vasilev  

**Link**: [PDF](https://arxiv.org/pdf/2606.09686)  

**Abstract**: Numeric format proliferation in machine learning hardware -- FP8 (E4M3 and E5M2), BF16, MXFP4, microscaling block formats, and dozens of research variants -- has outpaced the availability of vendor-neutral, bit-exact reference material. Engineers porting models across accelerators encounter silent divergences that are difficult to diagnose without a shared ruler.
This paper describes a catalog of 84 numeric formats spanning 13 families, a suite of six bit-exact conformance packs covering GF16, MXFP4 element, BF16, FP8 E4M3, FP8 E5M2, and E8M0 block scale, and an IEEE P3109 v3.2.0 cross-walk that maps each pack to its corresponding standards-track configured format. Each pack is a self-contained JSON document with a SHA-256 fingerprint, a shared row schema, and an anchor vector that encodes 3.0 -- the identity phi^2 + 1/phi^2 = 3 -- as a cross-pack sanity check. Packs are cross-validated against ml_dtypes 0.5.4 (Google/JAX); any divergence is documented explicitly and interpreted as a spec-permitted interpretation gap rather than hidden. The work is framed as registry filling: it does not propose new formats, make model-accuracy claims, or assert superiority over any vendor's implementation. All artifacts are publicly available at this https URL under an open license. 

---
# MeCo: One-Step MeanFlow-based Corrector for Multi-Channel Speech Separation 

**Authors**: Dohwan Kim, Jung-Woo Choi  

**Link**: [PDF](https://arxiv.org/pdf/2606.09677)  

**Abstract**: While discriminative models for multi-channel speech separation excel in reference-based metrics, they often exhibit suboptimal human listening quality. To address this, we propose a novel MeanFlow-based one-step generative corrector (MeCo). MeCo learns a conditional average velocity field to map discriminative estimates directly onto the clean speech manifold in a single step. To maximize one-step generation performance, we introduce Data-Space Optimization (DSO). DSO integrates an $\mathbf{x}_r$-loss, which penalizes prediction errors on longer displacement intervals to serve as a generative objective for human listening quality, with an Endpoint SI-SDR loss that directly optimizes terminal signal fidelity. Experiments demonstrate that MeCo achieves state-of-the-art (SOTA) performance with minimal computational overhead, simultaneously achieving superior signal fidelity and human listening quality in both in-domain and out-of-domain scenarios. 

---
# Transition-Based Digital Twin Modelling for Alzheimer's Disease under Sparse Longitudinal Data 

**Authors**: Yinyu Huang, Yilin Zhang, Sofia Michopoulou, Christopher Kipps, Rahman Attar  

**Link**: [PDF](https://arxiv.org/pdf/2606.09671)  

**Abstract**: Alzheimer's disease (AD) progression is highly heterogeneous and is typically observed through sparse and irregular longitudinal data, posing challenges for prediction and personalised monitoring. Existing machine learning approaches have improved AD prediction using multimodal data, yet often focus on static classification or cohort-level risk estimation, providing limited support for subject-specific modelling and uncertainty-aware reasoning. To address these limitations, we present a personalised digital twin framework for AD prediction and scenario-based analysis using multimodal longitudinal data. The proposed approach integrates complementary modelling strategies to capture clinical transitions and temporal dependencies across visits. Using data from the Alzheimer's Disease Neuroimaging Initiative (ADNI), including cognitive assessments, clinical variables, and MRI-derived phenotypes, the framework predicts cognitive status and diagnostic categories while quantifying predictive uncertainty and enabling patient-specific what-if trajectory analysis. Evaluation on leak-free subject-level splits demonstrates strong performance in score forecasting and diagnosis classification. In this sparse and irregular ADNI setting, transition-based modelling of adjacent visits achieved higher predictive accuracy than the sequence-based branch, suggesting that local transition modelling may be more data-efficient. While sequence models remain valuable for uncertainty-aware trajectory forecasting, local transition modelling offers a more data-efficient and robust predictive strategy. These findings highlight the importance of aligning temporal modelling strategies with clinical data structure and suggest that transition-based digital twin formulations may provide a practical and interpretable approach for personalised disease forecasting in neurodegenerative disorders. 

---
# Visual Prompting Meets Feature Reconstruction-Based Anomaly Detection with Dual-Teacher Supervision 

**Authors**: Mateo Diaz-Bone, Daniel Caraballo, Florian Scheidegger, Thomas Frick, Mattia Rigotti, Andrea Bartezzaghi, Roy Assaf, Niccolo Avogaro, Yagmur G. Cinar, Brown Ebouky, Filip M. Janicki, Piotr S. Kluska, Cezary Skura, Cristiano Malossi  

**Link**: [PDF](https://arxiv.org/pdf/2606.09670)  

**Abstract**: Recent Anomaly Detection methods achieve perfect detection and segmentation scores on well-established datasets, such as MVTec. However, many of these methods face challenges when foundational assumptions - such as consistent object scale, viewpoint, background, illumination, and centered placement - are violated. Those variations that occur render anomaly detection methods unusable in many real-world scenarios. To address these limitations, we introduce three key contributions: (1) a visual prompting pipeline that isolates objects using foreground-background masking; (2) a mechanism for unfreezing the teacher in student-teacher models to improve domain adaptability; and (3) a data augmentation strategy leveraging diffusion-generated synthetic images to enhance anomaly detection performance. We achieve a 3.5 percentage point improvement over the previous state-of-the-art on the challenging AeBAD dataset by using the Masked Multiscale Reconstruction (MMR) model as our backbone. 

---
# End-to-End Context Compression at Scale 

**Authors**: Ang Li, Sean McLeish, Haozhe Chen, Nimit Kalra, Zaiqian Chen, Artem Gazizov, Venkata Anoop Suhas Kumar Morisetty, Bhavya Kailkhura, Harshitha Menon, Zhuang Liu, Brian R. Bartoldson, Tom Goldstein, Sanae Lotfi, Micah Goldblum, Pavel Izmailov  

**Link**: [PDF](https://arxiv.org/pdf/2606.09659)  

**Abstract**: Long-context language model inference is bottlenecked by memory, as the KV cache grows with context length. Recent techniques to compress the KV cache fall short: they either degrade model quality substantially or require considerable time and compute to compress a single long prompt. Furthermore, many methods require the input to fit within the target model's context window, and are generally incompatible with modern production inference engines. Encoder-decoder compressors, which map a long token sequence to a shorter sequence of latent embeddings consumed by a decoder, are an appealing alternative in principle. However, existing approaches are not competitive with KV cache compression on the accuracy-efficiency frontier. In this work, we revisit encoder-decoder compression and close this gap. We first perform an architecture search, pre-training many variants from scratch to determine how best to design and train encoder-decoder compressors. Guided by our findings, we continually pre-train a family of 0.6B-encoder, 4B-decoder models on over 350B tokens each, at compression ratios of 1:4, 1:8, and 1:16. We introduce Latent Context Language Models (LCLMs), a family of compressors that improve the Pareto frontier across general-task performance, compression speed, and peak memory usage. We demonstrate that LCLMs serve as efficient backbones for long-horizon agents, letting the agent skim through a compressed long context and adaptively expand relevant segments on demand. 

---
# Muon Learns More Robust and Transferable Features than Adam 

**Authors**: Tianyu Ruan, Fengzhuo Zhang, Shuche Wang, Shihua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09658)  

**Abstract**: Muon has recently emerged as a state-of-the-art optimizer for pretraining Large Language Models (LLMs) and vision classifiers. Despite its efficiency advantage over Adam and SGD, the feature-learning advantage of Muon remains unclear. This paper investigates Muon's feature-learning advantage through the lens of robustness and transferability. First, by evaluating pretrained models on corrupted images and texts, we show that features learned by Muon are consistently more robust than those learned by Adam and SGD across different architectures, including transformers and Convolutional Neural Networks (CNNs). Using trained layer-wise probes, we further show that this robustness advantage is reflected in larger logit margins across layers. Second, by training linear classifiers or fine-tuning full models from pretrained parameters on downstream tasks, we demonstrate that Muon-learned features transfer more effectively than those learned by Adam and SGD. This transferability advantage is further supported by the diversity of hidden states across layers, as measured by effective rank. Finally, in a representative classification problem with multi-component features, we prove that Muon attains larger margins and higher effective rank than Adam and SGD, providing theoretical support for our empirical findings. 

---
# ArtiFact: A Large-Scale Multi-Modal Cultural Heritage Dataset 

**Authors**: Luciano Duarte, Olga Ovcharenko, Sebastian Schelter  

**Link**: [PDF](https://arxiv.org/pdf/2606.09648)  

**Abstract**: Multi-modal data management has emerged as a central research topic in the database community, spanning data integration, semantic query processing, and data quality assessment. Despite this growing interest, the community lacks large-scale, real-world datasets combining tables, text, and images. We present ArtiFact, a multi-modal cultural heritage dataset of 651045 museum records collected from the Metropolitan Museum of Art, the Art Institute of Chicago, and the Rijksmuseum. We demonstrate the utility of ArtiFact through two downstream tasks. For cross-modal error detection, we introduce a curated taxonomy of seven error categories injected into 130209 records and show that reliably detecting subtle domain-specific errors such as material anachronisms and temporal shifts remain an open challenge. For semantic query processing, we show that current systems struggle with queries involving cultural proximity, ambiguous object types, and historically contingent terminology. Our results position ArtiFact as a challenging benchmark for multi-modal data management research. 

---
# Do Video Foundation Models Understand Intuitive Physics? A Layerwise Probing Analysis 

**Authors**: Samuele Punzo, Niccolò Caselli, Ippokratis Pantelidis, Francesco Massafra, Salvatore Lo Sardo, Mohammadreza Salehi  

**Link**: [PDF](https://arxiv.org/pdf/2606.09646)  

**Abstract**: We study whether pretrained video foundation models encode intuitive-physics information in their frozen representations, and how this information varies across model families, layers, and probe types. Using frozen-feature probing on IntPhys2 and Minimal Video Pairs (MVP), we compare predictive joint-embedding models (V-JEPA), masked reconstruction models (VideoMAE), and a diffusion-based video generator (LTX-Video). V-JEPA achieves the strongest overall results across benchmarks, especially with probes that model temporal dynamics, while VideoMAE remains competitive and LTX-Video recovers weaker but non-trivial signal. Layerwise analyses show that physics-relevant information is weakest in early layers and becomes most accessible at intermediate-to-late depth, and temporal controls show that disrupting frame order substantially reduces performance, especially on MVP. Together, these results suggest that intuitive-physics knowledge emerges reliably in pretrained video representations, but its accessibility depends strongly on pretraining paradigm, representational depth, and readout mechanism. 

---
# FMplex: Model Virtualization for Serving Extensible Foundation Models 

**Authors**: Hetvi Shastri, Pragya Sharma, Walid A. Hanafy, David Irwin, Mani Srivastava, Prashant Shenoy  

**Link**: [PDF](https://arxiv.org/pdf/2606.09643)  

**Abstract**: Foundation models (FMs) are increasingly used as backbones for downstream tasks across language, vision, time-series, and multimodal applications. Yet existing model-serving systems deploy each customized task as an independent model instance, thereby replicating heavyweight backbones, wasting accelerator memory, and losing opportunities to amortize batching and loading costs. This paper presents FMplex, a serving system that treats FM backbones as a virtualization substrate for deployment sharing. FMplex presents each task with a virtual foundation model (vFM), a logically private FM instance backed by a shared physical FM. This abstraction lets independently customized tasks share a backbone while preserving task-specific extensions, independent lifecycles, and task-level isolation. In addition, we propose a batch-aware fair-queueing scheduler that combines weighted task-level sharing with inter- and intra-task batching across colocated tasks. We implement a FMplex-based serving stack spanning task construction, sharing-aware deployment, and runtime execution. Across 7 FM backbones (16 variants) and 92 downstream tasks, FMplex reduces latency by up to 80% over spatial partitioning and 33.3% over best-effort co-location, while hosting up to 6x more tasks at cluster scale. 

---
# ATN3D: Density-Aware LiDAR-Radar Early 3D Object Detection Under Extreme Sparsity 

**Authors**: Debojyoti Biswas, Xianbiao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09634)  

**Abstract**: 3D object detection is the backbone of perception for automated vehicles (AV) and broader intelligent transportation systems applications. Long-range detection is challenging because sensing evidence is sparse; yet this ``long-range'' scenario is routine in traffic. Although >30m is often labeled long-range in computer vision, on roadways it affords only approx. 1-2s for perception and decision-making. Under such extreme sparsity, two core challenges arise. First, early multimodal fusion tends to discard sparsity information and inject noise from empty or falsely occupied cells, degrading long-range recall. Second, context-agnostic uniform channel supervision favors dense and near-range samples, leaving far and small objects under-optimized, delaying the earliest detection of distant objects. We propose ``Ask The Neighbor'' (ATN3D), a LiDAR-Radar framework tailored for sparse-range conditions. ATN3D introduces (i) Density-aware early fusion with cross-modal gating that conditions fusion on per-voxel density/sparsity and Radar evidence, (ii) Occupancy-gated neighborhood aggregation with circular kernels to aggregate only from credible cells, (iii) Evidence-conditioned channel self-attention to adapt channel weights with weather/range, and (iv) a Range-aware loss that re-balances classification and localization by distance, aligning training with distance-stratified evaluation. On the VoD benchmark across clear and foggy conditions, ATN3D surpasses strong baselines: +3.55% mAP in clear weather and +8.41% mAP under simulated heavy fog; for >30m objects, gains are +3.33% (clear) and +2.09% (heavy fog). These results indicate earlier and more reliable long-range detections under sparse sensing in on-road traffic. 

---
# ReCoVLA: VLM-Guided Reward Compilation for Failure Recovery in Vision-Language-Action Policies 

**Authors**: Haodi Hu, Chung-Ta Huang, Jing Liu, Ye Wang, Kei Suzuki, Matthew Brand, Toshiaki Koike-Akino  

**Link**: [PDF](https://arxiv.org/pdf/2606.09630)  

**Abstract**: Vision-language-action (VLA) policies provide strong priors for language-conditioned manipulation, but remain brittle in off-nominal states requiring targeted recovery. We propose ReCoVLA -- a failure-conditioned residual recovery framework that keeps a pretrained VLA policy frozen, uses an external vision-language model (VLM) to infer the failure mode and recovery stage, and compiles a structured reward from task-relevant components. Rather than using the VLM to generate actions or rewards directly, ReCoVLA uses it as a semantic reward selector: it predicts a recovery descriptor and reward mask for in-simulation residual-policy training, followed by zero-shot sim-to-real deployment of the trained recovery policies. This decouples high-level failure understanding from low-level corrective control to support different VLAs. Experiments across short-horizon, long-horizon, and contact-rich manipulation tasks show that ReCoVLA outperforms the tested baselines on average. In simulation, our reward compiler improves average success from 36.7% for the fine-tuned $\pi_{0.5}$ baseline to 66.7%. In physical zero-shot sim-to-real experiments, ReCoVLA achieves the best average performance, with 61.7% success. 

---
# Powering the Future of AI: Navigating the Trade-offs for Europe's Energy Transition and Net-Zero Goals 

**Authors**: Mohammad Hemmati, Gbemi Oluleye, Vassilis M. Charitopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2606.09617)  

**Abstract**: The rapid expansion of AI globally has led to the proliferation of energy-intensive hyperscale data centres (DCs), making them as a structurally challenging component in power system planning and operation. Using a spatially explicit optimisation model of Europe across 21 AI growth scenarios, we systematically quantify additional demand, capacity requirements, emissions, and operational impacts of DCs. Results indicate that AI could drive 73-723 TWh of extra demand by 2050, risking cumulative emissions overshoots of 67-181 MtCO2 between 2030 and 2050. Our analysis indicates that after 2030, the geography of AI infrastructure will be shaped more by firm power and system flexibility than by the mere abundance of clean energy. In moderate scenarios, AI requires an additional of 200 hours of firm generation, which increases LCOE by 35 EUR/MWh in key hubs. We show that even under the pessimistic scenarios, existing infrastructure would require 70 GW additional capacity, while under managed growth pathways, this expansion could reach 226 GW. We further find DCs workload dynamics strongly shape energy dispatch, system flexibility, and emissions, while improved efficiency significantly reduces capacity needs, and system peaks. While our findings suggest that net-zero targets for 2050 may be achieved, critical emission risks may appear in the intermediate years, and the EU may compromise its carbon-neutral goals unless policies adapt to this accelerating digital transformation. 

---
# AGENTSERVESIM: A Hardware-aware Simulator for Multi-Turn LLM Agent Serving 

**Authors**: Rakibul Hasan Rajib, Mengxin Zheng, Qian Lou  

**Link**: [PDF](https://arxiv.org/pdf/2606.09613)  

**Abstract**: Multi-turn LLM agents interleave model calls with external tool invocations, shifting serving from stateless request processing to stateful program execution. Serving these workloads requires scheduling, KV-cache management, and routing policies that use program-level context, including turn dependencies, tool-induced gaps, and reusable KV state. Evaluating such policies directly on real systems is costly, since each design point may require dedicated accelerator time across arrival rates, model scales, serving-instance counts, and memory hierarchies. Simulation offers a scalable alternative, but existing LLM serving simulators target stateless request-level workloads and therefore omit the core dynamics of agent serving: multi-turn program execution, cross-turn cache locality, and KV-cache residency during tool gaps. We present AGENTSERVESIM, a hardware-aware simulator for multi-turn LLM agent serving. AGENTSERVESIM evaluates serving policies at program granularity through composable modules: a Program Orchestrator preserves program identity and turn order, a Tool Simulator materializes tool-induced gaps, a Session-Aware Router maintains program-to-instance affinity for cache-aware dispatch, and a KV Residency Model tracks policy-defined KV placement across HBM, host DRAM/CXL, and eviction. Across real serving deployments and hardware configurations, AGENTSERVESIM reproduces real-system behavior within 6% error across key performance metrics while running entirely on commodity CPUs. These results show that AGENTSERVESIM enables controlled, repeatable exploration of agent-serving policies without requiring exhaustive deployment on costly accelerators. 

---
# Shape Formation for the Cooperative Transportation of Arbitrary Objects Using Multi-Agent Reinforcement Learning 

**Authors**: Mohamed Sayed, Wolfram Burgard, Tanja Katharina Kaiser  

**Link**: [PDF](https://arxiv.org/pdf/2606.09610)  

**Abstract**: Cooperative object transportation is essential in numerous domains, including industrial to domestic services. A popular transportation strategy is to carry objects on top of multi-robot systems. The corresponding task is typically solved by decomposing it into three interconnected subproblems: formation control, cooperative navigation, and collision avoidance. A particular challenge posed by real-world objects is their potentially arbitrary shape and non-uniform mass distribution, necessitating robot formations that securely support the object. In this work, we address the challenge of pattern formation control for transporting such real-world objects by proposing a novel multi-agent reinforcement learning approach. Our approach enables a multi-robot system to autonomously position itself underneath an object to support its weight while avoiding obstacles during the formation process. Our evaluations with diverse environments and varying numbers of robots show that our approach leads to policies that reliably produce balanced formations and generalize to cluttered scenes and objects with complex geometry and non-uniform mass distribution. 

---
# Closure-Validated Circuit Discovery in Attention Heads: Co-activation Proposes, Ablation Disposes 

**Authors**: Yongzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09607)  

**Abstract**: Interpretability increasingly treats groups of components, not individual units, as the basic object, and proposes to find them by clustering co-activation statistics. We ask whether such a cheap signal actually identifies an attention-head circuit. Adapting a sparse-autoencoder clustering recipe to attention heads -- but validating by causal ablation rather than reconstruction -- we cluster heads and then run a closure test: ablate the discovered community and compare per-example damage to matched-random controls. Across two dense 1B-scale models (Pythia 1B, OLMo 1B) and two input distributions, the communities pass closure. In a Mixture-of-Experts model (OLMoE-1B-7B), route-conditional clustering recovers a statistically real signal that nonetheless does not survive closure -- ablation improves loss, the wrong direction. Extending closure across training, attention-target selectivity and participation ratio decouple from function in both directions. We conclude that a cheap signal is a circuit proposal, not a confirmed circuit; closure is what separates them. 

---
# I Was Scrolling and Then I Saw a Pregnant Strawberry 

**Authors**: Piera Riccio  

**Link**: [PDF](https://arxiv.org/pdf/2606.09589)  

**Abstract**: AI minidramas (also known as fruit dramas) are short, algorithmically distributed generative AI video series featuring anthropomorphized characters that have recently emerged as a widespread phenomenon on social media platforms. This paper argues that despite their seemingly innocuous aesthetic, these videos reproduce deeply gendered narrative structures in which female characters are systematically associated with moral transgression, sexual betrayal, and reproductive capacity, and that several plots also encode the logic of racialization, i.e., the process by which visible bodily difference is morally loaded. Drawing on feminist film theory, critical race theory, and platform studies, it further argues that the generative AI aesthetic of these videos, characterized by softness, roundness, and visual cuteness, functions as a mechanism of aesthetic laundering, neutralizing the ideological weight of these narratives and enabling their circulation despite content moderation systems. This paper approaches these questions through personal observation and close reading, reflecting on the specific affordances of generative AI that make this phenomenon both possible and culturally consequential for the field of computational creativity. 

---
# Seeing the Hivemind: A Consensus-Aware Interaction Technique for Mitigating AI Homogenization 

**Authors**: Muhammad Haris Khan, Joel wester  

**Link**: [PDF](https://arxiv.org/pdf/2606.09587)  

**Abstract**: People are increasingly using AI for creative tasks such as writing. While adoption continues to grow, this form of use risks undermining individual creativity locally and reducing the heterogeneity of creative output at scale. In response, we introduce the Semantic Repulsion Technique (SRT) and evaluate it both computationally and through a study with 16 participants who regularly use AI for creative tasks. Our computational assessment reveals that SRT increases semantic diversity by 85--167\% while reducing consensus phrases by 43--95\% across task modes. In the user study, SRT outputs received higher usefulness ($p = .019$, $W = .208$) and coherence ratings ( $p = .006$, $W = .260$); 68.8\% of participants were willing to use SRT-Strong for multiple tasks versus 18.8\% for baselines. Originality and coherence ratings were positively correlated across all systems ($\rho = +.40$ to $+.67$), suggesting that divergence need not compromise readability. Taken together, these preliminary findings can inform the design of AI systems that aim to support everyday creativity without contributing to homogenization. 

---
# CT-VAM: A Cerebello-Thalamic-Inspired Vision-Action Model for Efficient Visuomotor Control 

**Authors**: Jiacheng Li, Yize Guo, Jiabin Guo, Qingchen Liu, Jiahu Qin  

**Link**: [PDF](https://arxiv.org/pdf/2606.09572)  

**Abstract**: Vision-language-action models have shown strong promise for robot manipulation, yet raw language is primarily needed to specify task intent rather than to be repeatedly processed during high-frequency low-level execution. Motivated by this separation, we propose a cerebello-thalamic-inspired vision-action model (CT-VAM) for efficient task-conditioned visuomotor control. CT-VAM acts as a compact local execution policy that predicts action chunks from dualview visual observations, proprioception, and a lightweight task condition, potentially enabling a practical cloud-edge paradigm in which high-level semantic reasoning can be handled by large models while fast closed-loop control runs on local hardware. To fuse heterogeneous inputs effectively, CT-VAM introduces TARS (Thalamic Action Routing Stream), a stream-separated conditional attention decoder that independently routes action, visual and task streams, preventing dense sensory tokens from overwhelming compact task-relevant conditions. With only 68M parameters, CT-VAM achieves LIBERO success rates competitive with substantially larger VLA models, while reducing inference latency. Together with flow-consistent inpainting for asynchronous chunk execution, CT-VAM supports high-frequency control and demonstrates robust realworld deployment on resource-constrained robotic platforms. 

---
# Safe-RULE: Safe Reinforcement UnLEarning 

**Authors**: Shixiong Jiang, Taozheng Zhu, Fanxin Kong  

**Link**: [PDF](https://arxiv.org/pdf/2606.09559)  

**Abstract**: Offline safe reinforcement learning (Safe RL) enables policy learning without online interactions, making it suitable for safety-critical systems such as robotics systems. However, its reliance on static datasets exposes offline Safe RL to data poisoning attacks, where adversaries inject malicious samples that compromise safety and induce unsafe policy behavior. In this work, we propose a new learning paradigm, named safe reinforcement unlearning (Safe-RULE), used as a defense framework to remove the influence of poisoned data without retraining from scratch or requiring access to the original training environment. We further extend reinforcement unlearning to offline Safe RL by explicitly accounting for both task performance and safety constraints during the unlearning process. Experiments across benchmark Safe RL tasks demonstrate that our approach effectively enhances safety performance against data poisoning attacks. 

---
# FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing 

**Authors**: Yuhan Ma, Yong Li, Stefan Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2606.09551)  

**Abstract**: Two-server secure inference allows a client to query a hosted large language model (LLM) without revealing prompts or embeddings. Recent GPU systems based on function secret sharing (FSS) make linear layers efficient, but fixed-point nonlinearities and helper operations remain a bottleneck because each operator is typically implemented as a bespoke protocol with its own comparisons, wrap-around corrections, and preprocessing material. We present FuseFSS, a compiler that replaces per-operator protocol design with a single compilation pipeline. For each scalar fixed-point operator, a compact specification lists its interval partition, low-degree arithmetic pieces, and required predicate bits. The compiler emits two batched FSS evaluations on the public masked value: one packed comparison that returns all predicate bits, and one vector interval lookup that returns the active coefficients and constants. Compared to the current state-of-the-art FSS-based GPU secure inference, FuseFSS preserves accuracy while achieving a $1.24\times$--$1.50\times$ end-to-end speedup and reducing online communication by $9\%$--$16\%$ on BERT and GPT-style models; preprocessing is also lighter, with $14\%$--$23\%$ lower key-generation time and $20\%$--$24\%$ smaller keys. 

---
# SecureClaw: Clawing Back Control of LLM Agents 

**Authors**: Yuhan Ma, Stefan Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2606.09549)  

**Abstract**: Tool-using large language model (LLM) agents face two distinct security failures: unauthorized external actions and exposure of sensitive plaintext inside the runtime before any final output check can intervene. Existing defenses usually protect one boundary, either the planner/runtime or the action sink, and therefore do not by themselves secure both surfaces. We present SecureClaw, a dual-boundary architecture that places authorization at the effect sink and plaintext confinement at the read boundary. Sensitive reads pass through a trusted gateway that replaces raw values with opaque handles and, in the evaluated deployment, bounded summaries as an explicit declassification interface. Writes that change external state follow a PREVIEW$\rightarrow$COMMIT protocol in which only a trusted executor may commit the exact canonical request authorized by policy. The runtime can still plan over summaries and symbolic references, but cannot directly dereference secrets or perform side effects. Across AgentDojo, AgentLeak, and Agent Security Bench (ASB), SecureClaw is the only defense we evaluate in a common harness that simultaneously retains usable task utility and achieves 0\% attack success rate (ASR) on ASB, 0.64\% ASR on AgentDojo, and 3.23\% overall leak on AgentLeak's attacked parity lane, which measures final-output and internal-relay leakage. 

---
# Model Poisoning Against Federated Model Adaptation with Chain of Bit-Flips 

**Authors**: Bastien Vuillod, Kevin Hector, Pierre-Alain Moellic, Jean-Max Dutertre, Olivier Potin  

**Link**: [PDF](https://arxiv.org/pdf/2606.09548)  

**Abstract**: Federated Learning (FL) allows a set of clients to collectively train a global model without sharing local training data. Giving the responsibility of the training to decentralized actors may lead to poisoning attacks: clients controlled by malicious third party potentially poison the training dataset to install a backdoor in neural networks. In FL, these backdoor attacks rely solely on algorithmic approach, however, recent advances in hardware faults threats (e.g, Rowhammer) have widen the overall attack surface. In the context of federated model adaptation, we introduce a novel category of backdoor attack against FL systems that relies on model poisoning based on hardware-fault attacks. More precisely, we propose a task-agnostic backdoor attack that is implanted during the FL training time by inducing hardware faults (bit-flips) in parameters of a single local model. The backdoor is crafted during a previous offline phase from the pretrained model initially used by the FL system. Our results show that a backdoor can be successfully applied on different type of models and datasets. Typically, with up to 10 faults per malicious client occurrence and 19 total occurrences on a ResNet-18 are enough to reach 94% of attack success rate. Finally, we discuss the practicality and the robustness of the attack potential defenses, while putting into perspective the practical constraints of Rowhammer, which is the preferred attack vector for this type of threats. 

---
# Emergence of Context Characteristics Sensitivity in Large Language Models 

**Authors**: Nadya Yuki Wangsajaya, Haeun Yu, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2606.09525)  

**Abstract**: During instruction fine-tuning (IFT), large language models (LLMs) learn to follow instructions by using the provided context to answer a query. While prior work has studied how context characteristics correlate with context usage by the LLM, this analysis has been limited to inference time, leaving open how these relationships are acquired in the first place. Here, we measure how models' sensitivity to such characteristics shifts across successive IFT stages: supervised fine-tuning (SFT), direct preference optimization (DPO), and reinforcement learning with verifiable rewards (RLVR). Experiments across four models and three datasets show that SFT makes models more likely to use contexts that are easy to understand, such as containing high length, context-query similarity, and fluency. Post-SFT dynamics may either reinforce or resolve these preferences depending on the training dataset. Our findings reveal that context usage is actively reshaped at each IFT stage, and designing a balanced IFT dataset is important in ensuring robust context utilization of instruction-tuned models. 

---
# Closing the Prior-Posterior Loop: Self-Reflective Molecular Design with Analysis-Driven LLM Iteration 

**Authors**: Junyi Gong, Zijie Qiu, Ben Zhong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09520)  

**Abstract**: Can a general-purpose large language model design molecules with the precision of a seasoned chemist? Current LLM-based frameworks answer this question with scalar feedback loops-generate, score, reject-that amount to informed trial-and-error. Here we show that replacing a single number with the full physicochemical rationale from first-principles calculations transforms the LLM from a stochastic sampler into a causal reasoner. Our system couples retrieval-augmented generation with a self-reflection module that feeds orbital energies, atomic charges, and electron densities-rather than compressed scores-back into the design loop. On HOMO-LUMO gap targets from 1.0 to 5.0 eV, this structure-property-relationship (SPR) reflection achieves a deviation as low as 0.0003 eV and a 100% success rate on moderate tasks, decisively outperforming scalar-feedback and non-reflective baselines. The framework generalizes seamlessly to dipole-moment design and proves robust across five distinct LLM backbones. These results establish a new paradigm: when the model understands not only that a molecule fails, but why, iterative molecular design becomes genuinely mechanistic. 

---
# Targeting World Models to Compromise Robot Learning Pipelines 

**Authors**: Ethan Rathbun, Ahmed Agha, Saaduddin Mahmud, Christopher Amato, Alina Oprea, Eugene Bagdasarian  

**Link**: [PDF](https://arxiv.org/pdf/2606.09499)  

**Abstract**: World models have recently seen a rapid growth in both their popularity and capability as more data efficient tools for generating robot training data or simulating real world environments, with many works proposing their integration into the robot learning pipeline. While highly practical, in this work we demonstrate that world models introduce a uniquely stealthy and effective data poisoning entry point into the robot learning supply chain that can result in the deployment of unsafe or otherwise compromised robotic policies despite training on seemingly safe ground truth training data. In contrast to traditional data poisoning techniques which directly implant dangerous trajectories into sold or uploaded datasets, our novel attack methods inject malicious prompts or compromising transition dynamics into visibly safe teleoperated datasets which are only activated once fed through a world model as input. This can result in the generation of synthetic, dangerous robot training trajectories and subsequently unsafe or compromised robot policies. We demonstrate the effectiveness of our attacks against both state of the art action conditioned and text conditioned world models, showing a full end-to-end backdoor on a downstream DRL policy and a proof-of-concept for the VLA setting. Overall these findings necessitate research into more secure world models and reevaluating their position within the robot learning supply chain. 

---
# Memory Beyond Recall: A Dual-Process Cognitive Memory System for Self-Evolving LLM Agents 

**Authors**: Tianxiang Fei, Mingyang Song, Mao Zheng, Xiang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09483)  

**Abstract**: Long-term memory for an LLM agent is more than retrieving the right passage at the right time. Current memory systems collapse belief revision, causal coupling, and cross-domain abstraction into a single retrieval surface tuned for surface recall, and consequently struggle on implicit personalisation that requires reasoning over how a user has evolved. We propose DCPM, which reorganises agent memory along a cognitive capability hierarchy ascending from raw inputs and atomic facts, through diachronic belief trajectories and identity, to domain schemas, latent intentions and cross-domain patterns. The hierarchy is driven by two processes inheriting the architectural split of dual-process theory: a synchronous daytime writer (System1) that records belief revisions as doubly linked supersedes chains, and an asynchronous nighttime engine (System2) that induces schemas and intentions and sweeps for cross-domain collisions abstracted into higher-level core schemas. On LongMemEval, PersonaMem and PersonaMem-v2, enabling System2 contributes most where the benchmark rewards implicit cross-session inference (up to +5.20 on PersonaMem-v2) and least on span recall, matching the architectural prediction. 

---
# A Finetuned SpeechLLM for Joint Multi-Granular L2 Assessment and Natural-Language Rationales 

**Authors**: Aditya Kamlesh Parikh, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik  

**Link**: [PDF](https://arxiv.org/pdf/2606.09470)  

**Abstract**: Automated L2 speech assessment can assign proficiency labels, but often lacks interpretability. We propose a rubric-guided SpeechLLM for multi-aspect, multi-granular assessment, trained with a hybrid objective combining supervised fine-tuning and Bounded Direct Preference Optimization. The model jointly predicts ordinal labels at the sentence-level (accuracy, fluency, prosody), word/phoneme-level accuracy, and generates a natural-language rationale in the same response. On SpeechOcean762, our approach matches or outperforms single-granularity models while remaining competitive with prior approaches. We analyze rationale reliability along two axes: self-consistency with model predictions and alignment with ground-truth labels, using sentiment consistency (plausibility) and mention-based agreement (faithfulness). Rationales are plausible at the sentence level, but faithfulness degrades at the word/phoneme level: references are sparse and weakly aligned with token-level labels. 

---
# LargeMonitor: Monitoring Online Task-Free Continual Learning via Large Pretrained Models 

**Authors**: Mingqi Yuan, Xiaoquan Sun, Shihao Luo, Jiayu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.09430)  

**Abstract**: Online task-free continual learning (TFCL) requires intelligent agents to sequentially accumulate knowledge from an unbounded, non-stationary data stream under strict single-pass constraints and without any explicit task identifiers. Existing online TFCL paradigms primarily rely on parameter-efficient prompt tuning or dynamic structure expansion driven by training-coupled optimization dynamics, such as empirical loss fluctuations or evolving latent distances. As a result, these training-coupled solvers remain agnostic to the structural origins of distribution drift, mechanically enforcing a fixed strategy across fundamentally distinct streaming variations. To address this gap, we propose LargeMonitor, a framework that leverages large pretrained foundation models to autonomously orchestrate task-free continuous adaptation. Specifically, LargeMonitor introduces a decoupled detection module utilizing the frozen, stable representation space of large vision models (LVMs) to achieve robust, zero-shot drift detection without training-dependent interference or brittle threshold tuning. Upon a confirmed drift, the framework activates a context-aware diagnostic module driven by large multimodal models (LMMs) to interpret the precise semantic etiologies of the stream variation (e.g., novel class emergence vs. environmental domain shift). This dual-stage capability empowers the continuous learner to dynamically deploy adaptive and shift-specific optimization strategies. Extensive experiments across multiple TFCL settings and benchmarks demonstrate that LargeMonitor achieves precise, robust detection and diagnosis of complex data streams while consistently improving the performance of existing online TFCL algorithms. 

---
# Context-Aware Deep Learning for Defect Classification in Atomic-Resolution STEM 

**Authors**: Jiadong Dan, Cheng Zhang, Leyi Loh, Ivan Verzhbitskiy, Yuan Chen, Goki Eda, Michel Bosman, N. Duane Loh  

**Link**: [PDF](https://arxiv.org/pdf/2606.09419)  

**Abstract**: Artificial intelligence is rapidly advancing materials characterization, yet most applications in electron microscopy rely solely on image contrast, overlooking the chemical and experimental context that shapes image formation. This limitation makes defect classification inherently ambiguous, as similar contrasts can arise from different materials or imaging conditions. Here we develop a context-aware learning framework that integrates image-derived contrast with metadata describing composition, beam energy, and detector geometry. Using a systematically constructed dataset of ~55 million simulated patches spanning 576 cases across 96 doped monolayer transition-metal dichalcogenides, we show that conditioning on contextual variables transforms defect classification from an ill-posed image-only task into a well-posed, physically grounded problem. The framework achieves over 98% accuracy on simulations and near-human agreement on experimental data, with a 94% reduction in posterior entropy. By emphasizing contextual grounding over architectural complexity, this approach links experimental image contrast to the underlying chemical and imaging conditions, supporting physically grounded defect assignments and a general pathway toward multimodal AI models for autonomous materials characterization. 

---
# Harness Engineering for Physical AI: Robot Middleware Is the Harness Layer 

**Authors**: Sanghoon Lee, Jiyeong Chae, Kyung-Joon Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.09416)  

**Abstract**: Robot middleware faces a new role in the era of Physical AI. Learned policies, planners, and vision-language-action (VLA) models now enter deployed robots as causal participants on the control path, but the layer that integrates them with timing, scheduling, and network has not been named. Recent language-agent work names this layer the harness, the external system that mediates tools, manages state, bounds resources, and records execution. The robotics community has not yet adopted this framing, and we propose that robot middleware is that harness. A Physical AI harness differs from a software harness in where it intervenes. A software harness mediates at tool-call boundaries. A Physical AI harness must mediate at control, computing, and communication simultaneously, because a learned policy's output crosses all three: its commands shift the trajectory, its inference time shifts the schedule, and its payload shifts the bandwidth. Robot middleware is the lowest robot-stack layer with mediating abstractions over all three, so it is best positioned to compose their enforcement. It already provides most of what a harness needs but lacks the enforcement for an AI model. We name this missing enforcement as three functions: Projection gates each output at emission, Isolation bounds the model's execution and transmission slot, and Transfer falls back to a verified baseline when checks fail. Each appears today as hand-built application code in deployed robot systems, built on surfaces robot middleware already provides. Robot middleware should host them not as the best single-axis enforcer but as the layer that composes all three. We sketch this as a ROS 2 Harness Profile, a deployment artifact that carries an AI model's declared output region, inference budget, and operating regime while the middleware enforces them across ROS 2, DDS, and Zenoh. 

---
# AI Assurance in UK Defence: Challenges in Operationalising JSP 936 

**Authors**: Callum Cockburn, Sam Farrow  

**Link**: [PDF](https://arxiv.org/pdf/2606.09414)  

**Abstract**: This report examines practical challenges in operationalising JSP 936 Part 1 for AI assurance in UK Defence. Using a structured interpretive review of the directive's requirements, the analysis identifies eight thematic challenge areas adequacy of evidence and argument, management of human interaction with AI, definition of the operational environment, integration of AI within systems of systems, assessment and maintenance of AI performance, analysis of safety and security, measurement of ethicality, and mitigation of the inherent complexities of AI. The report argues that JSP 936 provides a useful governance basis, but that implementation depends on unresolved technical, organisational, and assurance questions. These challenges stem from the socio-technical nature of AI-enabled systems, uncertainty in real-world deployment contexts, limitations in current assurance methodologies, and tensions between performance, safety, human oversight, security, and ethical acceptability. The report identifies areas where further methods, guidance, and organisational capability are needed for the ambitious, safe, and responsible adoption of AI across Defence. This is consistent with MOD's own framing of JSP 936 as requiring iterative implementation and supporting guidance. 

---
# Can Data Work be Reparative? 

**Authors**: Srravya Chandhiramowuli, Ding Wang, Alex Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2606.09408)  

**Abstract**: We present an ethnographic study of an alternative approach to data work, developed by a civic-tech initiative that builds datasets for training and benchmarking online safety systems. They aim to respond to online safety concerns from a feminist perspective, by building safety datasets collaboratively with those most impacted by online harms. In this paper, we examine how this approach aims to reorient data work as a site for repair and redress, and trace the struggles they encounter in the process. Specifically, we draw attention to the challenges and tensions involved in advancing just reward for data work and collective governance of AI datasets. Examining these challenges through an STS-informed lens of reparative justice and repair, we argue that the work of repairing data work (and AI) lies, fundamentally, in resetting the ties of accountability. At a time heightened emphasis on efforts like safety evaluations and red teaming to make AI more responsible, we highlight the need to confront foundational questions about how the humans involved in these efforts relate to the datasets and systems they help produce. A reparative lens demands that we interrupt prevailing norms of data work and place at their centre, not AI or datasets, but those most harmed by the neglect, oversight and exclusion animated in the current modes of dataset production. This, we argue, offers a bold vision for responsibility and contributes towards a critical agenda for building alternative futures of data and AI practice. 

---
# SAILS: Surrogate-based Analysis of Interactions via Local Effect Smooths 

**Authors**: Timo Heiß, Julia Herbinger, Bernd Bischl, Giuseppe Casalicchio  

**Link**: [PDF](https://arxiv.org/pdf/2606.09404)  

**Abstract**: Feature interactions drive much of the predictive power of machine learning models, yet existing explanation methods only detect and quantify interactions without revealing their functional form, or visualize only restricted interaction types. We propose Surrogate-based Analysis of Interactions via Local effect Smooths (SAILS), a model-agnostic framework that analyzes pairwise interactions through interpretable generalized additive model (GAM) surrogates fitted to the local effects of a black-box model. For each interval of a feature of interest, the surrogate smooth terms isolate the interaction components on derivative level, enabling (i) interaction detection through a heuristic derived from significance tests on smooth terms, (ii) interaction form categorization into linear, product-separable, and non-product-separable types, and (iii) tailored, interpretable visualizations for each interaction type. We empirically validate the framework through controlled simulations and a real-world task, demonstrating its effectiveness for pairwise interactions, with limitations under strong feature correlations and higher-order interactions. SAILS fills a notable gap in the XAI toolbox, going beyond detection of interactions alone to characterizing their functional form. 

---
# Real-time body pose non-verbal communication with a consistency-based reliability measure 

**Authors**: Alina Marcu, Dragos Costea, Cristina Lazar, Marius Leordeanu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09390)  

**Abstract**: Body movement communicates intent at distances and in conditions where neither the face, nor speech can be captured. We study the recognition of communicative intent from 2D body pose alone. We argue that body motion is a reliable signal especially in scenarios that require real time low-cost on-device person-to-robot communication in long distance environments, such as rescue missions. However, existing resources do not isolate this signal. Affective corpora combine body, face, voice and text, while skeleton action-recognition benchmarks label the action performed rather than the message conveyed. We release a dataset of real frames of full-body pose covering ten communicative intents and we compare it against other real (IPC) and synthetic (MotionLCM, VEO3.1, Kimodo) ones that span a range of difficulty. We target systems that can run on a robot's limited onboard hardware. We benchmark multiple models, from skeleton graph classifiers to joint motion-forecasting networks, and report performance metrics together with frame rate on an embedded GPU (NVIDIA Orin~Nano), since speed matters as much as accuracy in our scenario. Finally, we show that a model's own autoregressive self-consistency works as an unsupervised reliability signal. We give a short proof that bounds the probability that a self-consistent prediction is correct, show that this probability grows with the number of consistent steps, and identify the conditions under which a confident prediction can still be false, benchmarked against industry-standard metrics. 

---
# Reasoning Arena: Trace Tournaments When Verifiable Rewards Fall Short 

**Authors**: Han Zhou, Adam X. Yang, Laurence Aitchison, Anna Korhonen, Albert Q. Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09380)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a leading paradigm for improving the reasoning ability of large language models through outcome-based supervision. However, verifiable rewards frequently become uninformative at the group level: when all sampled traces of a given prompt receive identical rewards, group-relative advantage estimation provides no gradient signal, even though the traces may differ substantially in reasoning quality. We propose Reasoning Arena, an adaptive training framework that routes such non-diverse reward groups to a judge system instead of discarding them. Beyond examining the final answer, Reasoning Arena constructs trace tournaments, where reasoning traces are compared head-to-head to expose finer-grained preferences within the group, converting reasoning quality into rich relative reward signals. To make reward estimation efficient, rather than exhaustively comparing every pair, each new trace is evaluated against a small, dynamically updated pool of previously generated traces as anchors to efficiently establish a relative ranking. We then fit a Bradley-Terry model on the incomplete comparison graph, enabling scalable RL integration without quadratic pairwise comparisons. Empirical results demonstrate that Reasoning Arena consistently outperforms the RLVR baseline by 7.6% on average in competition mathematics and coding benchmarks. By converting otherwise wasted zero-advantage samples into useful gradient updates, our method accelerates training by 27% to 41%, saving nearly 50% of generation compute, and substantially improves overall reasoning performance. 

---
# Scaling Neural Network Verification with Tensor Parallelism and Fully Sharded Data Parallelism 

**Authors**: Sergei Vorobyov, Eugene Ilyushin  

**Link**: [PDF](https://arxiv.org/pdf/2606.09377)  

**Abstract**: Formal neural network verification -- proving that a network satisfies safety properties for \emph{all} inputs in a specified domain -- is bounded in practice by GPU memory: standard implementations of bound-propagation algorithms (IBP, CROWN, $\alpha$-CROWN) require weight and relaxation-coefficient matrices to reside entirely on one accelerator. We adapt two parallelism techniques originally developed for large-scale model training to the \texttt{auto\_LiRPA}\,/\,$\alpha,\beta$-CROWN verification framework. \textbf{Tensor Parallelism (TP)} shards both weight and $A$-matrices across GPUs, achieving ${\approx}2\times$ peak-memory reduction at $P{=}2$; soundness is confirmed on VNN-COMP 2022 MNIST-FC benchmarks, though bound tightness degrades with the number of sharded zones due to forced IBP substitution for intermediate bounds inside sharded zones. \textbf{Fully Sharded Data Parallelism (FSDP)} shards only weight matrices with a per-layer \texttt{AllGather}, producing bounds that are \emph{bitwise identical} to the single-GPU baseline: baseline memory drops by 80--90\%, peak memory by 34--39\% on wide MLPs. FSDP integrates cleanly with complete verification ($\beta$-CROWN + Branch-and-Bound) and with convolutional layers (\texttt{BoundConv}); a complete \emph{unsat} result is obtained for CIFAR-100 ResNet-large (VNN-COMP 2024) under FSDP. Across all experiments the memory bottleneck in $\alpha$-CROWN+BaB mode proves to be per-neuron alpha tensors, not weight matrices, pointing to the key direction for future work. 

---
# PhysScene: A Scene Graph Dataset for Scientific Visual Reasoning in Physics Experiments 

**Authors**: Minghao Zou, Qingtian Zeng, Shangkun Liu, Yanda Meng, Guanghui Yue, Baoquan Zhao, Abdulmotaleb El Saddik, Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.09368)  

**Abstract**: Scene Graphs (SGs) provide structured representations of visual scenes by modeling objects and their pairwise relationships. Despite recent progress, existing datasets primarily focus on generic natural contexts, leaving domain-specific and function-oriented scenes largely underexplored. This limitation restricts the evaluation of relational reasoning in scientific experimental scenes, thereby hindering the development of intelligent monitoring, analysis, and related applications in such scenes. To address this gap, we introduce PhysScene, the first SG dataset tailored to physics experiments. PhysScene encompasses specialized instruments, structured experimental setups, and functional relations intrinsic to experimental environments, enabling reasoning that extends beyond spatial co-occurrence to logical dependencies. Rather than pursuing large data scale, PhysScene focuses on strong semantic constraints and high relation density in experimental scenes, posing new challenges for existing scene parsing algorithms while offering opportunities for further improvements. Extensive analyses and experiments show that PhysScene complements existing benchmarks and establishes a valuable testbed for advancing scientific visual reasoning. The dataset is publicly available at this https URL. 

---
# Beyond Humans: Multispecies Animal Face Recognition Using Transfer Learning 

**Authors**: Maria De Marsico, Anil K. Jain, Annalaura Miglino  

**Link**: [PDF](https://arxiv.org/pdf/2606.09353)  

**Abstract**: Individual animal recognition can be useful in the search for lost or stolen pets, the tracking of individuals of endangered species, and the recognition of animals in crowded farms. Present recognition techniques mostly use physical devices, e.g., microchips, often impractical and difficult to apply. These could be replaced by remote recognition via the animal's face; if accurate enough, it provides several advantages: it is non-invasive, can work at a distance, and is difficult to counterfeit, as, for instance, in the case of substituting sick animals for healthy ones in the food industry. The few existing datasets with sufficient per-subject images annotated with a single animal identity are not large enough to train current deep learning architectures. We rather investigate the possibility of transfer learning, exploiting pre-trained network models as backbones. Our experiments compared FaceNet, which is specifically trained on large databases of human faces, with the Vision Transformer (ViT) pre-trained on ImageNet, i.e., on object categories. We used three face datasets of very different animals: dogs, primates (lemurs, golden monkeys, and chimpanzees), and cattle. We report the results and, for each dataset, compare them with the state of the art (SOTA) ad hoc-trained deep networks. The capture conditions differ among the three datasets. Image quality (resolution, motion blur, diverse poses, etc.) decreases from dogs to cattle to primates. The best performance was achieved with dogs, where ViT reached a mean verification accuracy of 96.85% and a Rank-1 Identification Rate of 84.34%. The results for endangered primates are still encouraging, but performance varies across animal classes and tasks (verification or identification), and does not always outperform SOTA. For cattle, the ViT results outperform SOTA, while FaceNet is still competitive. 

---
# Conan-embedding-v3: Fusing Modality-Specific Models for Omni-Modal Embedding 

**Authors**: Shiyu Li, Zhiyuan Hu, Yifan Wang, Peiming Li, Zheng Wei, Yang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09331)  

**Abstract**: Omni-modal retrieval promises a single embedding space for text, image, video, document, and audio inputs, but building such a unified retriever is difficult since these modalities differ in data distribution, architecture, and optimization dynamics. In this work, we present Conan-embedding-v3, a decouple--fuse--recover framework for omni-modal retrieval. Conan-embedding-v3 first trains modality specialists independently and fuses their task vectors into a single dense backbone, a strategy we call Decoupled Specialist Fusion. We show that this fusion composes visual, video, and document retrieval capabilities, but also exposes a failure mode for projector-based modalities: when audio is attached through an external encoder and projector, fusing the backbone leaves the projector calibrated to the audio-specialist backbone, causing a large audio retrieval regression despite copying all audio-specific modules unchanged. We call this failure Projector Drift. To repair it, Conan-embedding-v3 applies Projector Recovery (i.e., full-parameter fine-tuning of the projector while keeping the backbone frozen) followed by balanced multi-modal rehearsal. The resulting model supports these retrieval pathways in one backbone, achieving 74.9 scores on MMEB while obtaining 55.61 on the 30-task MAEB audio suite. 

---
# A Universal Dense Football Event Representation Based on TabTransformer 

**Authors**: Weiran Yang, Daniel Memmert, Maximilian Klemp-Weins  

**Link**: [PDF](https://arxiv.org/pdf/2606.09327)  

**Abstract**: Football event data constitute a rich spatiotemporal source for quantitative analysis of player actions in team sports. These datasets contain heterogeneous features, combining continuous location coordinates with categorical variables such as action type, action outcome, and body part. Such data have been applied in sports analytics for match outcome forecasting, player evaluation, and tactical pattern recognition. However, existing approaches predominantly encode categorical features using one-hot or ordinal embedding representations, overlooking the intrinsic semantics of action descriptors. The Transformer is a deep neural network architecture based on self-attention that captures dependencies between input features at arbitrary positions. We propose and implement a Transformer-based model to learn latent dependencies among categorical event features and produce dense representations of football events. By encoding categorical features as learned embedding vectors, sport-specific action semantics are captured during pretraining, enabling the representations to support downstream tasks such as action value estimation and play style recognition. Empirical evaluation shows that the embedding representations yield superior probability calibration over task-specific baselines on the downstream prediction tasks, as measured by Brier score. 

---
# Brain-Prompt Injection: A Route-Safety Audit for BCI-LLM Agents 

**Authors**: Jianwei Tai  

**Link**: [PDF](https://arxiv.org/pdf/2606.09315)  

**Abstract**: BCI-to-agent pipelines turn decoded neural activity into an authorization channel for tool-use agents, exposing a new attack surface we call \emph{brain-prompt injection}: signal-side perturbations, context-only injections, and adaptive dual-decoder attacks can all change the routed action while EEG-side or text-side monitors remain blind. Route safety in this stack depends on what the audit log can observe, not on decoder accuracy or agreement alone. We define a Route-Safety Audit Contract: a minimal log schema, denominator hierarchy, and endpoint specification, and prove an audit-schema separation theorem together with a C3 attacked-dependence decomposition; clean agreement and marginal robustness do not identify the joint term that controls C3 routing. As a calibration layer on top of the contract, we apply split-conformal calibration to a non-oracle EEG confirmation channel and report the resulting false-accept frontier under an explicit threat-archetype matrix. We instantiate the contract on EEGMMI native left/right command-control over 5{,}400 events, harmless tool stubs, and seed/case denominators. Provenance blocks C2 routes ($0.000$); agreement-plus-provenance routes C3 flips ($1.000$); confirmation-plus-provenance routes them ($0.000$). The conformal frontier reaches FAR $0.000$ at clean utility $0.150$ for $\alpha=.005$ and FAR $0.119$ at clean utility $0.452$ for $\alpha=.10$ under acquisition isolation; an attacker-controllable confirmation channel breaks the bound to $\approx\!1$. Subject-cluster bootstrap confirms these intervals on $60$ subjects; cross-architecture (TinyEEGNet, EEGNetV4) and capacity-sweep results show within-regime saturation. Mediation and confirmation reduce risk; they are not intent certificates. 

---
# Internalizing Geometric Law: Learning from Solver Residuals for Precision-Critical Generation 

**Authors**: Rafael Cabral, Pang Zixi, Ziyi Shou, Shen Xin  

**Link**: [PDF](https://arxiv.org/pdf/2606.09278)  

**Abstract**: Large Language Models frequently hallucinate in precision-critical domains such as technical diagramming and mechanical design, where outputs must satisfy strict geometric constraints. We study open-ended geometric synthesis from natural language: translating free-form descriptions into precise constructions whose entities must simultaneously satisfy dozens of interacting constraints. To make this tractable, we release PyGeoX, a programmable geometric DSL that compiles declarative constraints into a differentiable loss, and PyGeoX-Bench, a stratified suite of 300 problems with per-constraint verifiable rewards. Using PyGeoX as a verifier, we identify a failure mode we call Outlier Gradient Masking: under global-norm rewards (any scheme that aggregates residuals through a single norm, for example, $\exp(-\mathrm{MSE})$), a single outlier constraint can nullify the learning signal across all others. To address this, we propose Saturating Additive Rewards (SAR), which decompose the reward into bounded per-constraint terms, preserving partial progress and ensuring consistent gradients even under severe violations. Against MSE-based rewards, the natural baseline for geometry solvers, SAR improves the hard-tier solving rate by $2.3\times$, and the resulting 8B model is competitive with much larger frontier systems on this benchmark. We release the engine, benchmark, and data at this https URL. 

---
# Physics-Guided Sequence-Based Generative Framework for Acoustic Metamaterial Inverse Design 

**Authors**: Yijie Li, Jiahao Xu, Ching-Chih Tsao, Lili Qiu, Jingxian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09266)  

**Abstract**: Acoustic metamaterial (AMM) inverse design is particularly challenging for broadband target responses due to acoustic dispersion: a structure that matches the desired response at one frequency may deviate at others, and modifying geometry to improve one sub-band often perturbs neighboring sub-bands. Yet existing broadband inverse-design approaches are either constrained by predefined templates, or rely on image representations that fail to preserve the geometric precision and structural connectivity required by acoustic structures. We present MetaSeq, a physics-guided, sequence-based generative framework for acoustic metamaterial inverse design. At its core, MetaSeq introduces a language that represents each AMM as a structured sequence, rather than as a pixel grid or fixed template. This representation preserves precise geometry, explicitly encodes connectivity, and casts inverse design as a sequence-to-sequence task from target response to structure sequence. MetaSeq further constructs a balanced, high-fidelity dataset with efficient calibration and complexity-based sampling. To address the one-to-many nature of inverse design, MetaSeq combines supervised pretraining with reinforcement learning fine-tuning guided by a physics-based solver and validity checker. Extensive evaluations against COMSOL and five baselines show that MetaSeq reduces response error by 45% over the best baseline. 

---
# BSTabDiff: Block-Subunit Diffusion Priors for High-Dimensional Tabular Data Generation 

**Authors**: Al Zadid Sultan Bin Habib, Md Younus Ahamed, Prashnna Gyawali, Gianfranco Doretto, Donald A. Adjeroh  

**Link**: [PDF](https://arxiv.org/pdf/2606.09257)  

**Abstract**: High-Dimensional Low-Sample Size (HDLSS) tabular domains (e.g., omics) are characterized by $n \ll m$, where $n$ = number of samples, and $m$ = number of features. Such domains often exhibit strong local correlation groups, sparse cross-group dependencies, heavy-tailed non-Gaussian marginals, heteroscedastic noise, and structured missingness, making direct density learning in $\mathbb{R}^m$ ill-conditioned since $n \ll m$. We propose BSTabDiff, a block-subunit generative framework that partitions the $m$ observed features into $M$ latent blocks ($M \ll m$) and generates each block via a shared low-dimensional subunit variable, concentrating global dependence learning in the compact block-latent space $\mathbb{R}^M$ while decoding to the full feature space with copula-driven dependence, flexible per-feature marginals, and explicit missingness mechanisms. BSTabDiff supports modern deep priors on block latents, including diffusion and normalizing flows, enabling stable synthesis and controllable benchmark generation in the HDLSS regime. Empirically, BSTabDiff produces more realistic and stable high-dimensional synthetic data when compared with unstructured tabular generators on HDLSS data. 

---
# Proposal Refinement for Few-Shot Object Detection 

**Authors**: Yuan Zeng, Bin Song, Jie Guo, Yuwen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.09245)  

**Abstract**: Few-shot object detection has gained widely attention in recent years. Some excellent algorithms have been proposed to handle this task. However, most of these algorithms rely on the performance of few-shot classification. Unlike previous attempts, our work focuses on the problem of unbalanced distribution of region proposals between the novel classes and the base classes. In order to alleviate this unbalanced distribution, we propose the proposal refinement approach for different training phases. Specifically, refinement loss is designed for the base training phase to enhance sensitivity of the model to novel classes, and refinement branch is introduced as an auxiliary branch for RPN (Region Proposal Networks) to generate more novel proposals in the fine-tuning phase. By rebalancing the proposal distribution, the proposed approach outperforms the baselines methods by roughly 1\%$\sim$6\% on current benchmarks without increasing any inference time. Through extensive experiments, we prove that we establish a new state-of-the-art method for the few-shot object detection task. 

---
# EgoTactile: Learning Grasp Pressure for Everyday Objects from Egocentric Video 

**Authors**: Yuan Zeng, Yujia Shi, Tiao Tan, Xingting Li, Yaqi Qin, Zongqing Lu, Wenming Yang, Jing-Hao Xue, Qingmin Liao  

**Link**: [PDF](https://arxiv.org/pdf/2606.09243)  

**Abstract**: Estimating full-hand grasp pressure from egocentric video is critical for immersive VR and robotic manipulation, yet dense tactile sensing often relies on intrusive hardware. Existing vision-based methods predominantly rely on planar surfaces or fingertip contacts, failing to generalize to complex 3D object interactions. Therefore, we introduce EgoTactile, a benchmark pairing egocentric video with full-hand pressure supervision for diverse everyday objects, incorporating a bare-hand transfer subset to enable generalization to natural scenarios. Leveraging this benchmark, we first establish EgoPressureFormer as a discriminative baseline. Beyond this, to explicitly address the uncertainty in partial observations, we propose EgoPressureDiff, a conditional diffusion framework that adapts a large-scale pre-trained video diffusion backbone. By combining rich world knowledge priors with a Physically-Informed Feature Rectification layer to inject semantic constraints, our approach effectively infers plausible contact patterns and resolves visual-physical ambiguities. Extensive experiments demonstrate that our method achieves superior performance on the benchmark and robust transferability to in-the-wild scenarios. Our project page is available at this https URL. 

---
# Self-Paced Curriculum Reinforcement Learning for Autonomous Superbike Racing in Simulation 

**Authors**: Luca Ghisi, Jacopo Essenziale, Carlo D'Eramo, Matteo Luperto  

**Link**: [PDF](https://arxiv.org/pdf/2606.09236)  

**Abstract**: Autonomous Racing has seen remarkable progress through deep Reinforcement Learning (RL), primarily for four-wheeled vehicles. However, motorbikes introduce substantially greater complexity due to the need to manage balance and lean angle, in addition to more reactive steering and throttle control, and a smaller weight. In this work, we present a framework for training an autonomous agent to race a superbike in VRider SBK, a physics-accurate Unity-based motorbike simulator. Our approach integrates Soft Actor-Critic (SAC) with Self-Paced curriculum Deep reinforcement Learning (SPDL), which dynamically generates progressively more challenging tasks based on the agent's performance, without requiring manual curriculum design. The agent's state space comprises proprioceptive features extended with lean-angle history, along with global track features via course points. The reward signal is shaped to encourage progress along the track while penalizing instability-inducing behaviors specific to two-wheeled dynamics. Preliminary experimental results demonstrate that SPDL outperforms SAC alone in training efficiency, lap time, and driving stability across multiple tracks and motorbike models, establishing a first baseline for RL-based autonomous motorbike racing. 

---
# End-to-End Training for Discrete Token LLM based TTS System 

**Authors**: Changfeng Gao, Yong Ren, Jun Yuan, Ye Bai, Zhao You, ShiDong Shang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09234)  

**Abstract**: Recent state-of-the-art (SOTA) text-to-speech (TTS) systems typically adopt a cascaded pipeline consisting of a speech tokenizer, an autoregressive large language model (LLM), and a diffusion based flow-matching (FM) model, with these components trained independently. In this paper, we propose a fully end-to-end (E2E) optimization framework that unifies the training of the speech tokenizer, LLM, FM model, and an additional reward model (RM). Specifically, we first jointly optimize the tokenizer using multi-task objectives derived from reconstruction for FM, next-token prediction for LLM, and multi recognition task for RM. This joint training encourages the discrete speech token space to capture acoustically and semantically salient information that is better tailored to TTS. We then further optimize the LLM using downstream reconstruction and recognition by FM and RM, which reduces inference-time mismatch and steers the LLM toward more preferred generations. Experimental results show that our E2E framework consistently outperforms cascaded baselines. On the Seed-TTS-Eval benchmark, our system achieves a word error rate (WER) of 0.78% and 1.56%, a new SOTA result with a 0.6B-parameter LLM and 0.5B-parameter FM model. These results validate that holistic E2E optimization is critical for improving discrete-token-based TTS systems with a much simpler training pipeline. 

---
# Trustworthy Smart Fabs via Professional Proxies: Scaling Safe and Sustainable by Design (SSbD) through Industrial Data Spaces 

**Authors**: Han-Teng Liao, Chang-Yi Kao, Karen Ang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09227)  

**Abstract**: The convergence of the 2026 European Union Safe and Sustainable by Design (SSbD) framework, Corporate Sustainability Due Diligence Directive (CSDDD), and Carbon Border Adjustment Mechanism (CBAM) introduce a severe governance bottleneck for advanced semiconductor manufacturing facilities ("Smart Fabs"). Regulatory compliance demands have surpassed the capacity of manual corporate reporting, creating a direct conflict between multi-stakeholder transparency and corporate data privacy. This paper addresses this challenge by introducing a zero-trust socio-technical orchestration framework that operationalizes a six-layer SSbD reference architecture within trustworthy industrial data spaces. We propose a shift from reactive automation to autonomous governance through "Professional Proxies"-role-based agentic workflows executing within hardware-isolated trust zones. Structured as an interoperable network protocol stack, the framework coordinates an automated, five-step "relay race" between Facility, Process Engineering, and Finance proxy teams to align factory-floor yield models with macro-level sustainability mandates. By executing Virtual Metrology (VM) predictions and Federated Machine Learning (FML) inside hardware-rooted Trusted Execution Environments (TEEs), this architecture resolves the Data Sovereignty Paradox, demonstrating how fabs can export cryptographically signed compliance tokens via International Data Spaces (IDS) connectors without exposing proprietary process recipes. Ultimately, this framework provides technology managers with a verifiable, evidence-based pathway toward resilient, net-zero Industry 5.0 ecosystems. 

---
# Resource-aware Computation-Communication Overlap for multi-GPU ML Workloads 

**Authors**: Minyu Cui, Miquel Pericas  

**Link**: [PDF](https://arxiv.org/pdf/2606.09200)  

**Abstract**: The rapid growth of large-scale machine learning (ML) has made distributed training across multiple GPUs a fundamental component of modern ML systems. As model sizes and computational throughput continue to increase, communication overhead has become a dominant bottleneck in multi-GPU training, particularly when computation and communication are executed sequentially. This work explores concurrent execution of computation and collective communication using two portable runtime controls: shared-memory-driven occupancy shaping for computation kernels and elevated scheduling priority for communication kernels. Our approach regulates computation-kernel residency through per-block shared-memory allocation, leaving sufficient on-chip resources for communication kernels to make progress. In addition, assigning higher priority to communication streams ensures steady communication progress once resources become available. Experiments on NVIDIA A40, A100, H100, and AMD MI250X GPUs demonstrate that the proposed method enables effective computation-communication overlap and reduces total execution time by up to 25.5 percent, without modifying vendor libraries or kernel implementations. 

---
# Pretrained, Frozen, Still Leaking: Auditing Cross-Encoder Attribute Transfer in EEG Foundation Models 

**Authors**: Jianwei Tai  

**Link**: [PDF](https://arxiv.org/pdf/2606.09189)  

**Abstract**: EEG foundation-model releases are usually audited one endpoint at a time: raw-reconstruction, membership inference, identity linkage, or DP-SGD on the downstream head. We audit the same released embeddings under all four endpoints jointly, on BIOT, LaBraM, and EEGPT, and show that each single-endpoint audit clears releases that still leak spectral attributes. The decisive evidence is a cross-encoder transfer audit: a single ridge attribute decoder learned from one frozen encoder transfers, via a fitted linear bridge, to held-out-subject test splits of every other encoder, with subject-disjoint matched-control 95% CI lower bound at least 0.081 across all six BIOT/LaBraM/EEGPT directions. We prove a sufficient condition: two encoders sharing a nontrivial attribute-coordinate projector overlap beta admit a chained ridge bridge attacker with centered-gain lower bound sqrt(beta/(1+tau^2)) - eps_br - rho_0, and back-solve beta in [0.008, 0.198]. To turn the joint audit into a deployment-readable decision rule we introduce an audit-endpoint disagreement score (AEDS), prove sufficient conditions for its positivity, and bootstrap-calibrate it per cell; AEDS is positive in all eight matched-CI cells (BIOT/LaBraM/EEGPT on EEGMMI; LaBraM on Sleep-EDF, 54-channel LIMO, CHB-MIT pediatric scalp EEG) with p<0.001, while a head-level Carlini LiRA membership audit reaches AUC only 0.50-0.70. Standard defenses fail under audit: a Wiener-style noise-aware adaptive attacker, the LiRA audit, and DP-SGD at every utility-preserving epsilon in {4,8} leave the attribute channel essentially unchanged. The contribution is an audit framework that turns scattered single-endpoint defenses into a joint release decision, supported by a cross-encoder bridge theorem and adaptive-attacker, LiRA, and DP-SGD baselines; the audit licenses release-blocking, not raw-waveform exfiltration or held-out-subject identity recovery. 

---
# Culturally-Adapted Red-Teaming Across East and Southeast Asian Contexts: A Methodological and Comparative Analysis 

**Authors**: Hyeji Choi, Yongtaek Lim, Minwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.09178)  

**Abstract**: Multilingual safety evaluation of large language models (LLMs) has predominantly relied on direct translation (DT) of English benchmarks into target languages - an approach that converts surface-level linguistic form while failing to reflect the cultural context embedded in threat scenarios, social norms, and legal frameworks. We construct paired DT and culturally-adapted (CA) datasets via 1:1 seed matching for four languages - Korean (KO), Japanese (JA), Thai (TH), and Khmer (KM) - and compare Attack Success Rate (ASR) and Cultural Realism scores across four open-source LLM. CA prompts yield Delta-ASR > 0 across all 16 language x model combinations (mean +9.3 pp), and DT-based evaluation underestimates risk in 44 of 48 category x language combinations. Language-level analysis reveals that the distribution of threat forms is heterogeneous across languages. Cultural Realism analysis further shows that DT Cultural Depth (C3) scores remain consistently below 1.0 out of 3.0 across all four languages (mean 0.17), whereas CA scores reach up to 2.51, indicating that direct translation produces inputs systematically divergent from those encountered in real-world multicultural settings. These findings demonstrate that adapting benchmarks to language-specific cultural contexts - rather than relying on linguistic translation alone - is necessary for valid multilingual LLM safety evaluation. 

---
# CANS: Accelerating Multiuser Collaborative Edge Inference via Cooperative Autodidactic NeuroSurgeon 

**Authors**: Zheshun Wu, Ziyang Zhang, Changyao Lin, Zenglin Xu, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09175)  

**Abstract**: Recently, mobile edge computing (MEC)-enabled collaborative deep neural network (DNN) inference has emerged as a promising approach for delivering intelligent services to resource-constrained mobile devices. A representative scenario is multi-user collaborative edge inference, where distinct devices independently partition their DNN models and offload backend computation to a common edge server over wireless networks. However, determining the optimal DNN partition for each device is challenging due to unknown and time-varying system conditions, including fluctuating wireless links and diverse device capabilities. To address this problem, we propose Cooperative Autodidactic NeuroSurgeon (CANS), a collaborative edge inference framework that enables devices to adaptively learn optimal DNN partitions by sharing informative feedback during online inference. To handle the challenge of device heterogeneity and better leverage offline inference experience, we integrate a novel FedLinUCB-DW algorithm that groups devices of the same type and warm-starts online exploration using local offline early-exit inference experience. Furthermore, we provide theoretical guarantees for FedLinUCB-DW by deriving the regret upper bound. We also validate our method on both a simulated environment and a hardware prototype system. Empirical evaluations demonstrate that CANS achieves lower inference latency compared to state-of-the-art baselines. Especially, in prototype experiments on two edge devices, the proposed CANS reduced average inference latency by up to 50% compared to the non-cooperative baseline. 

---
# Crop Recommendation and Agricultural Query Answering System Using Spatio-Temporal Graph Neural Networks and Hybrid Retrieval Augmentation 

**Authors**: Prajwal Thapa, Yagya Raj Pandeya  

**Link**: [PDF](https://arxiv.org/pdf/2606.09160)  

**Abstract**: This paper presents a unified system designed to support precision agriculture by integrating advanced weather prediction, crop recommendation, and a question-answering tool for farmers. We propose two deep learning models -- a Transformer-based Graph Neural Network and a Spatio-Temporal Graph Convolutional Network (STGCN) -- to forecast weather conditions for the next 30 days using data from 1,359 locations in Nepal. The STGCN outperforms the Transformer-based model in accuracy (MSE ~0.011 vs. 0.013), effectively modeling both spatial and temporal dependencies in climate data. These predictions are combined with static soil properties such as pH, moisture, and organic content to generate localized crop recommendations through a scoring algorithm that matches each crop's optimal growing conditions. Additionally, we develop a Retrieval-Augmented Generation (RAG) chatbot that leverages domain-specific agricultural documents to answer farmers' questions in natural language. The entire system is deployed via a mobile application, offering real-time suggestions and conversational support. User feedback confirms the system's usability and relevance, especially in rural settings where personalized farming guidance is limited. Overall, our approach demonstrates how combining machine learning models with local agricultural data can empower farmers with actionable insights, promoting more informed decisions, better crop yields, and increased resilience to climate variability. 

---
# Unified Energy for Invariant and Independent Decoding in Diffusion Language Models 

**Authors**: Yuchen Yan, Minkai Xu, Zaiquan Yang, Yatao Bian  

**Link**: [PDF](https://arxiv.org/pdf/2606.09159)  

**Abstract**: Diffusion Language Models (DLMs) enable parallel text generation by iteratively denoising a full sequence, offering attractive flexibility compared to auto-regressive (AR) decoding. However, existing methods fail to fully capture token relationships, leading to a performance gap relative to AR baselines, especially as the degree of parallelism increases. In this paper, we give a systematic analysis of the gap, identifying three key factors: (i) model capacity, (ii) dependency, and (iii) invariance. To address these issues, we first propose an invariant energy (Inv-E) together with an effective sampling-based estimator to handle the invariance issue. By further combining with the independent energy (Ind-E), we obtain a unified energy (Uni-E), that accounts for all these factors. Uni-E enjoys a unique advantage: it can be computed exactly without sampling-based partition estimation. Besides, Uni-E is model agnostic and can therefore be scaled to models of arbitrary size. We further prove that Uni-E can correct the distribution shift caused by dependency and invariance. Extensive experiments across Diffusion Language Models (DLMs) and Diffusion Large Language Models (DLLMs) demonstrate the effectiveness of the proposed Uni-E. 

---
# SEF-CLGC at SemEval-2026 Task 11: Logical Notation Impact on Language Model Performance 

**Authors**: Hanna Abi Akl, Fabien Gandon, Catherine Faron, Pierre Monnin  

**Link**: [PDF](https://arxiv.org/pdf/2606.09157)  

**Abstract**: This paper revisits our pipeline called Syllogistic Evaluation Framework-Common Logic Grammar Construction (SEF-CLGC). We combine formal logical notations with Small Language Models (SLMs) to evaluate reasoning performance on the SemEval-2026 Task 11 Subtask 1: Disentangling Content and Formal Reasoning in Large Language Models. Our experiments show that by relying solely on SLMs, trained on a combination of natural and symbolic languages, our best model achieves a content score of 27.80% on the task while significantly lowering the content bias in reasoning. 

---
# Decoding Pedestrian Crossing Intention from Egocentric Vision via Vision Language Models 

**Authors**: Danya Li, Xiang Su, Yan Feng, Rico Krueger  

**Link**: [PDF](https://arxiv.org/pdf/2606.09142)  

**Abstract**: Egocentric vision offers a first-person view of human perception and decision making, yet its potential for traffic-safety prediction remains underexplored. In this work, we study the decoding of pedestrian crossing intentions from short egocentric video clips. We approach this by formulating the task as a closed-ended visual question answering (VQA) problem and leveraging vision language models (VLMs) to predict the pedestrians' intent. We first benchmark three families of state-of-the-art VLMs in a zero-shot setting, finding that they achieve moderate gains over random guessing but exhibit limited higher-level traffic reasoning. Motivated by these findings, we further adapt VLMs to the target task using parameter-efficient fine-tuning. Our results show that the fine-tuned models substantially outperform their zero-shot counterparts and achieve a 9\% accuracy improvement over a specialized transformer-based baseline. Finally, we demonstrate that incorporating additional contextual cues, including ego motion, vehicle motion, and eye gaze, further improves predictive performance. In particular, the fine-tuned Qwen3-VL-2B model guided by eye gaze and ego motion achieves a 14.5% accuracy improvement over the transformer baseline, establishing a new state of the art for egocentric pedestrian intent decoding. 

---
# Steganography Without Modification: Hidden Communication via LLM Seeds 

**Authors**: Felix Mächtle, Jonas Sander, Sebastian Berndt, Ben Weimar, Nils Loose, Thomas Eisenbarth  

**Link**: [PDF](https://arxiv.org/pdf/2606.09135)  

**Abstract**: We demonstrate that widely deployed Large Language Model (LLM) inference stacks harbor a steganographic channel that requires no modification to model weights, sampling code, or output distributions. The channel exploits a structural property of deterministic decoding: pseudo-random number generators (PRNGs) used in inverse-transform sampling produce a seed-dependent sequence of token-level probability intervals that can be reconstructed from the generated text alone. A sender encodes a secret message in the PRNG seed before generation; a receiver reconstructs the intervals and recovers the seed, and thus the hidden payload, by exhaustive search over the seed space.
We formalize two operational modes. In the known-prompt setting, sender and receiver share the prompt, enabling exact interval reconstruction and perfect seed recovery via forced alignment. In the unknown-prompt setting, only the generated text is available; approximate interval reconstruction combined with a maximum-hit-count scoring strategy still permits reliable recovery from sufficiently long outputs.
Extensive experiments across six model families and five heterogeneous text domains show that, in the known-prompt setting, full 32-bit seed recovery from the complete 2^32 candidate space achieves up to 100% accuracy, depending on model and text domain, within 300 tokens and under 35 seconds on a single GPU. In the unknown-prompt setting, recovery reaches near-perfect accuracy at 600-800 tokens in about 12 seconds. We further analyze the influence of prompting strategies, tokenization ambiguities, and sampling hyperparameters on channel reliability. Moreover, we discuss several applications of our results: First, it allows for the steganographic transmission of 32 bits, but also shows that ignorance of the prompt is not a valid security assumption. 

---
# From USD Scenes to Knowledge Graphs: Zero-Shot Ontology Grounding with LLMs 

**Authors**: Jiangtao Shuai, Zongxiong Chen, Manfred Hauswirth, Sonja Schimmler  

**Link**: [PDF](https://arxiv.org/pdf/2606.09134)  

**Abstract**: Constructing knowledge graphs from 3D simulation scenes is essential for robot task reasoning, but the key bottleneck, grounding scene objects to formal ontology classes, still relies on manually curated dictionaries that are brittle and do not generalize across assets. We investigate whether large language models (LLMs) can automate this grounding step for Universal Scene Description (USD) scenes as a zero-shot, training-free alternative. On a kitchen scene (125 objects) with SOMA-HOME Ontology, LLMs achieve 90-96% exact-match accuracy with descriptive names and 49-89% with abbreviated names, substantially outperforming dictionary and embedding baselines. Under fully opaque names, context-augmented prompting recovers up to 48%. Feature ablation reveals that LLMs primarily exploit semantic cues in the scene graph (sibling names and parent paths); anonymizing these cues reduces accuracy to 0-6%, while geometry alone yields only 4-17%. 

---
# Unveiling Privacy Risks in Multi-modal Large Language Models: Task-specific Vulnerabilities and Mitigation Challenges 

**Authors**: Tiejin Chen, Pingzhi Li, Kaixiong Zhou, Tianlong Chen, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.09125)  

**Abstract**: Privacy risks in text-only Large Language Models (LLMs) are well studied, particularly their tendency to memorize and leak sensitive information. However, Multi-modal Large Language Models (MLLMs), which process both text and images, introduce unique privacy challenges that remain underexplored. Compared to text-only models, MLLMs can extract and expose sensitive information embedded in images, posing new privacy risks. We reveal that some MLLMs are susceptible to privacy breaches, leaking sensitive data embedded in images or stored in memory. Specifically, in this paper, we (1) introduce MM-Privacy, a comprehensive dataset designed to assess privacy risks across various multi-modal tasks and scenarios, where we define Disclosure Risks and Retention Risks. (2) systematically evaluate different MLLMs using MM-Privacy and demonstrate how models leak sensitive data across various tasks, and (3) provide additional insights into the role of task inconsistency in privacy risks, emphasizing the urgent need for mitigation strategies. Our findings highlight privacy concerns in MLLMs, underscoring the necessity of safeguards to prevent data exposure. Our dataset and code can be found here. 

---
# An Enhanced Geometric-Spectral Feature Learning Framework for Airborne Multispectral Point Cloud Classification 

**Authors**: Xian Li, Yanfeng Gu, Aleksandra Pižurica  

**Link**: [PDF](https://arxiv.org/pdf/2606.09123)  

**Abstract**: Multispectral point cloud (MPC) is composed of 3D spatial-spectral information, which holds tremendous potential for accurate land-cover classification. However, the representation power of classification models is limited by inherent high-dimensional and heterogeneous spatial-spectral information, unbalanced sample distribution, and inter-class spectral similarity of airborne MPCs. We build two MPC datasets and propose an enhanced geometric-spectral feature learning framework based on attentions for airborne MPC classification. A key component in our model is a two-stream feature fusion method with attention mechanisms, which enhances the representation capability of spatial-spectral features from high-dimensional heterogeneous MPCs. The first stream aims to extract position-encoded global spectral features with fusion self-attention, and the second stream comprises a multikernel point convolution and feature aggregation attention to extract spectral-guided geometric features. We then develop a residual attention fusion block to integrate the most informative geometric-spectral features from the two parallel streams. Another important contribution of this work is a joint loss function to improve the learning ability on unbalanced and interclass similar samples. Experimental results on two airborne MPC datasets demonstrate the effectiveness of the proposed method compared with the state-of-the-art methods. Furthermore, the codes and datasets used in this paper will be made available freely at this https URL. 

---
# Autonomous Incident Resolution at Hyperscale: An Agentic AI Architecture for Network Operations 

**Authors**: Arun Malik  

**Link**: [PDF](https://arxiv.org/pdf/2606.09122)  

**Abstract**: Cloud network infrastructure at hyperscale presents unique operational challenges where traditional human-driven incident response cannot keep pace with the volume, velocity, and complexity of failures. This paper presents an agentic AI architecture for autonomous incident resolution in large-scale network operations. Our system employs a multi-agent orchestration framework where specialized AI agents collaborate to detect, diagnose, and remediate network incidents without human intervention. We describe the architectural principles, including hierarchical agent decomposition, skills-based tool invocation via standardized protocols, structured knowledge encoding from operational runbooks, progressive autonomy with safety boundaries, and closed-loop verification. The architecture has been deployed in production at a major cloud provider, demonstrating that agentic AI systems can achieve autonomous resolution rates exceeding 90% for common incident categories while maintaining safety guarantees through layered authorization and rollback mechanisms. We discuss design tradeoffs, failure modes, and lessons learned from operating autonomous AI agents at scale. 

---
# Optimizing Energy-based Neural Network Training with Coherent Ising Machine 

**Authors**: Chen-Rui Fan, Bo Lu, Zhi-Hong Zhang, Run-Qing Zhang, Jing-Wei Wen, Chuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09117)  

**Abstract**: While Ising machines serve as advanced physical solvers for the Ising model,enabling applications in combinatorial optimization and neural network training,their scalability for large-scale neural networks remains constrained by hardware connectivity limitations and suboptimal training methodologies. In this work,we leverage a Coherent Ising Machine (CIM) to train an energy-based neural network using Equilibrium Propagation, achieving performance comparable to existing software-based implementations. We further enhance the algorithm by integrating the Adam optimizer to solve for the ground state of a Hopfield energy network, significantly improving convergence speed and solution accuracy. Additionally, we demonstrate the scalability of our approach across deeper network architectures and convolutional operations. Our results highlight the potential of CIM dynamics as a scalable platform for training complex neural networks, offering a pathway toward energy-efficient implementations via analog circuits, optoelectronics, or integrated photonics. This work establishes a novel physical framework for next-generation AI hardware development. 

---
# Hybridizing Equilibrium Propagation with Ising Machines for Efficient Energy-Based Learning 

**Authors**: Chen-Rui Fan, Bo Lu, Xing-Yu Wu, Tie-Jun Wang, Chuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09112)  

**Abstract**: The rapid evolution of artificial intelligence has led to substantial advances in deep neural networks. Nonetheless, conventional GPU-based training remains highly energy-demanding, motivating the exploration of physical dynamics and compatible energy-based learning schemes, such as equilibrium propagation (EP). EP-based training, however, frequently suffers from convergence to local minima due to phase-space contraction. Here we introduce an Ising-dynamics-inspired equilibrium-propagation framework in which dissipative Hopfield relaxation is replaced by an extended phase-space dynamics with conjugate variables. The resulting training paradigm keeps the local two-phase learning rule of EP while changing the physical route by which neural states reach equilibrium. We show that this dynamics lowers effective energy barriers, accelerates convergence, improves noise robustness, and trains deep convolutional Hopfield networks on MNIST, FashionMNIST, and CIFAR-10 with performance comparable to backpropagation. 

---
# Addressing Market Regime Changes and Heavy-Tailed Returns in Portfolio Optimization via Bayesian VAR and Elliptical Black-Litterman 

**Authors**: Daniil Mikriukov, Ruoyu Sun, Angelos Stefanidis, Jionglong Su, Zhengyong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09104)  

**Abstract**: Deep reinforcement learning (DRL) frameworks for portfolio optimization have shown promise for their ability to learn allocation rules dynamically from market data. However, these models fail to account for fat-tailed returns, which characterize actual market behavior with more frequent extreme events. Furthermore, historical data is treated homogeneously, without accounting for temporal importance, leading models to fail during regime changes. We propose a new BAVAR-BLED algorithm that combines methods derived from Bayesian-Averaging Vector Autoregressive (BAVAR) and the Black-Litterman model using Elliptical Distributions (BLED) within a TD3 architecture. BAVAR captures a set of vector autoregressive representations that consider multi-scale temporal features, enabling adaptive allocation decisions based on regime-aware estimates of return expectations and dispersion matrices. These estimates serve as prior inputs to BLED, a model that uses Student's t-distributions, allowing for more realistic fat tail return estimates. The BAVAR-BLED algorithm uses transformer networks for view construction and CNNs for risk-aversion estimates, which modify dynamic allocation decisions based on market conditions. An evaluation of 29 Dow Jones Industrial Average constituents over a decade-long market period shows that BAVAR-BLED significantly outperforms state-of-the-art methods, achieving Sharpe and Sortino ratios of 1.72 and 2.70, respectively, and total returns of 57.26%. 

---
# Context Rot in AI-Assisted Software Development: Repurposing Documentation Consistency for AI Configuration Artifacts 

**Authors**: Christoph Treude, Sebastian Baltes  

**Link**: [PDF](https://arxiv.org/pdf/2606.09090)  

**Abstract**: Developers increasingly provide AI coding assistants with persistent context through configuration files such as this http URL, this http URL, and .cursorrules. These files describe code elements, architecture, and development conventions, forming the context that guides AI tool behavior across sessions. As software evolves, this context can become stale, a phenomenon we call context rot. While AI configuration artifacts are new, the underlying consistency problem connects to decades of software documentation research. Researchers have built tools to check consistency between documentation and code, spanning README files, code comments, API documentation, architecture descriptions, and installation instructions. We argue that this existing toolbox is an immediate starting point for detecting context rot, and we present a research roadmap mapping documentation consistency approaches to corresponding problems in this new setting. As preliminary evidence, applying an existing README/wiki consistency checker to a statistically representative sample of 356 repositories identifies stale code element references in 23.0% of repositories, showing that traditional documentation consistency tools can already surface context rot. 

---
# Context-Fractured Decomposition Attacks on Tool-Using LLM Agents: Exploiting Artifact Provenance Gaps 

**Authors**: Xiaofeng Lin, Yukai Yang, Daniel Guo, Sahil Arun Nale, Charles Fleming, Guang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.09084)  

**Abstract**: Tool-using LLM agents interact with the world through actions that persist state in artifacts (e.g., workspace files or logs). Consequently, jailbreak defenses must reason about cross-step composition rather than isolated text. Yet most existing attacks and defenses, including ``multi-turn'' jailbreaks such as Crescendo and Tree of Attacks,still assume a single contiguous conversation visible to the defender. This assumption breaks down in real agent pipelines, where enforcement is fragmented across tools, modules, and time, and where artifact provenance is often not tracked. We operationalize a deployment failure mode for tool-using LLM agents, the \emph{provenance gap}, and study reproducible triggers for it: \emph{Context-Fractured Decomposition} (CFD), a family of cross-context multi-step jailbreaks that preserve benign-looking intermediate artifacts from an early interaction and elicit harmful behavior much later, potentially in a different agent instance or workflow stage, via individually innocuous tool actions whose risk emerges only under delayed artifact-mediated composition. We instrument the failure mode with trace-level diagnostics and outline a verifiable mitigation direction (provenance lineage tagging). Across agent-system jailbreak benchmarks, CFD improves success rates by up to 28.3 percentage points over state-of-the-art baselines, even against strong single-turn judges. Disclaimer: This paper contains examples of harmful or offensive language. 

---
# FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via Lookahead Sparse Attention 

**Authors**: Yan Wang, Qifan Zhang, Jiachen Yu, Tian Liang, Dongyang Ma, Xiang Hu, Zibo Lin, Chunyang Li, Zhichao Wang, Jia Li, Yujiu Yang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09079)  

**Abstract**: Conventional LLMs keep the full KV cache loaded during decoding, causing a severe GPU memory bottleneck for ultra-long context serving. In this report, we propose Lookahead Sparse Attention (LSA), a novel inference paradigm powered by a Neural Memory Indexer built upon the DeepSeek-V4 architecture. Rather than passively attending to all historical tokens, LSA proactively predicts future context demands and preserves only the query-critical KV chunks in the GPU memory. Crucially, we instantiate this architecture via a backbone-free decoupled training strategy. By formulating the indexer as a standard dual-encoder architecture, we train it independently using standard retrieval training frameworks without ever loading the massive backbone model into GPU memory.
We demonstrate that this "less is more" paradigm significantly maximizes serving efficiency while acting as an effective attention denoiser in tasks that rely on long-term global memory. Across primary long-context evaluation suites (e.g., LongBench-v2, LongMemEval, and RULER), FM-DS-V4 compresses the average physical KV cache footprint down to merely 13.5% of the full-context baseline, while consistently preserving or slightly elevating downstream accuracy (+0.6% absolute margin on average). Crucially, at extreme 500K scales, FlashMemory suppresses the physical KV cache overhead by over 90% without destabilizing the backbone's core reasoning capacities. 

---
# A Unifying Lens on Reward Uncertainty in RLHF 

**Authors**: Ely Hahami, Yoel Zimmermann, Ray Zhou, Jack Benarroch Jedlicki  

**Link**: [PDF](https://arxiv.org/pdf/2606.09073)  

**Abstract**: Reinforcement learning from human feedback (RLHF) is bottlenecked by \emph{reward hacking}, where the policy exploits errors in a proxy reward model (RM) and produces high RM scores without genuine quality gains. A natural mitigation is \emph{pessimism}: penalizing rewards in regions where the RM is uncertain. However, standard scalar RMs provide no principled notion of uncertainty. We argue that the right object is a \emph{distributional} reward model $p(r\mid x,y)$. Under either a Bayesian inference or a KL-distributionally robust optimization (KL-DRO) lens, the KL-regularized RLHF objective admits a closed-form effective reward $\tilde r(x,y) = \pm\beta\log\mathbb{E}_p[e^{\pm r/\beta}]$. The pessimistic branch unifies the prior heuristics for RM ensemble aggregation: mean aggregation, worst-case optimization (WCO), and uncertainty-weighted optimization (UWO) all emerge as limits or truncations of this single expression. This also clarifies the implicit assumptions of each existing rule. 

---
# OnlyDense: Reduced-Order Modeling for Lagrangian simulation 

**Authors**: Tu Do, Shannon Ryan, Santu Rana  

**Link**: [PDF](https://arxiv.org/pdf/2606.09065)  

**Abstract**: In science and engineering, Lagrangian simulation methods such as Smooth Particle Hydrodynamics (SPH) or Material Point Method (MPM) are often employed to study the behavior of dynamic systems. However, these methods can be prohibitively computationally expensive, particularly when simulating multi-scale spatial or temporal phenomena, e.g., void growth and coalescence within macro-scale geometries, structural failure of spacecraft components resulting from hypervelocity impact of space debris particles, etc. In contrast to graph-based methods, where the state of the system is understood as a discrete set of particles, we propose a learning framework for scalable representation and dynamics modeling of massive particle systems by treating the system state as a function and its evolution as a trajectory in Hilbert space. Rather than representing the state as a discrete set of particles or embedding it in a nonlinear latent manifold, we approximate the state space with a linear subspace spanned by learned neural basis functions. This parameterization enables direct projection to obtain latent coefficients and explicit access to the basis functions, avoiding optimization over a nonlinear latent space. The resulting representation admits a natural interpretation: latent variables correspond to coefficients in Hilbert space, and basis functions correspond to spatial modes, analogous to Proper Orthogonal Decomposition. The framework thus unifies classical projection-based reduced-order modeling with modern deep learning, while remaining invariant to the number of discretization points. Experiments on large-scale SPH simulations with over one million particles, including dynamic events with extreme deformation and fragmentation, demonstrate that the proposed method accurately reconstructs and predicts dynamics, achieving an R$^2$ score above $0.99$ with as few as $32$ basis functions. 

---
# See More, Think Deeper: Query-Expanded Visual Evidence and Answer-Clue Guided Reflection for Long Video Understanding 

**Authors**: Shuning Wang, Zhiheng Wu, YiNuo Lu, Naiming Liu, Chen Jia, Bowen Liu, Shuo Nie, Weijie Zhu, Yumeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09064)  

**Abstract**: Recent advances in Video Large Language Models (Video-LLMs) have enabled performance on long-video understanding tasks. However, existing methods still face two key limitations: evidence acquisition often relies on a single search intent, and answer generation lacks an effective visual feedback mechanism. To address these limitations, we propose \textbf{CoVER}, a Comprehensive Visual Evidence and Reflection framework for long-video understanding. CoVER enables Video-LLMs to \textbf{See More} by dynamically gathering query-expanded visual evidence, and \textbf{Think Deeper} by verifying draft answers with effective answer-specific visual feedback. Together, these mechanisms shift long-video understanding from answer-centric generation to evidence-centric and visually verifiable reasoning. Experimental results show that CoVER-7B substantially outperforms models with the same parameter scale and even surpasses state-of-the-art closed-source models on certain metrics. 

---
# Stage-1 Controls the Entropy Regime, Not the Outcome 

**Authors**: Jianxiong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2606.09059)  

**Abstract**: Two-stage post-training -- a Stage-1 warm-start (supervised fine-tuning, SFT, or on-policy distillation, OPD) followed by Stage-2 reinforcement learning (RL) -- is increasingly used for vision-language models (VLMs). We ask what Stage-1 actually controls in a small-data study using Qwen2.5-VL-7B with a same-modality 72B VLM teacher for OPD. First, the three warm-starts reach a narrow $53$--$54\%$ band on Geometry3K internal validation, consistent with the narrow range reported by recent specialized methods; this setup provides little evidence that Stage-1 changes the in-domain endpoint. Second, a matched-recipe, early-stopped SFT improves out-of-domain MathVista by $+2.1$ points, reversing the $-9.5$-point drop of an over-trained variant. The clearest difference is the \emph{entropy regime}: OPD enters RL with substantially higher policy entropy than either SFT initialization, and the separation remains visible through the available trajectories. At the in-domain initialization, OPD also has higher answer diversity and pass@16 ($+2.0$ to $+5.2$ points over SFT), although problem-level bootstrap intervals show that the smaller contrast is uncertain. The advantage is absent after RL (endpoint pass@16 values within $1.1$ points) and on MathVista (six models within $1.2$ points). Our contribution is therefore a bounded empirical characterization: Stage-1 is strongly associated with the entropy regime in this setup, but the downstream payoff is small, localized, and not evidence that OPD is a better RL warm-start. 

---
# INFUSER: Influence-Guided Self-Evolution Improves Reasoning 

**Authors**: Siyu Chen, Miao Lu, Beining Wu, Heejune Sheen, Fengzhuo Zhang, Shuangning Li, Zhiyuan Li, Jose Blanchet, Tianhao Wang, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09052)  

**Abstract**: Self-evolution offers a scalable path to stronger reasoning: a pretrained language model improves itself with only minimal external supervision. Yet existing methods either depend on extensively curated or teacher-generated training data, or, when the generator runs unsupervised, reward it by a difficulty heuristic that need not improve the solver. We introduce INFUSER, an iterative co-training framework with two co-evolving roles: a Generator that drafts questions and reference golden answers from a pool of unstructured, automatically collected documents, and a Solver that improves by training on them. The solver is trained with standard correctness rewards against the generator-provided answers, while the generator is rewarded by an optimizer-aware influence score that measures whether each proposed question would actually improve the solver on the target distribution. Because this continuous, noisy influence score is poorly served by standard GRPO, we propose DuGRPO, a dual-normalized variant of GRPO, for generator training. Together, these turn the document pool into an adaptive curriculum that favors questions useful to the current solver, not just hard ones. On Qwen3-8B-Base, INFUSER outperforms strong self-evolution baselines with over 20% relative improvement on Olympiad and SuperGPQA benchmarks, and an 8B INFUSER co-evolving generator outperforms a frozen 32B thinking generator on math and coding. Ablations confirm each design choice is necessary, and two extensions, applying INFUSER to an instruction-finetuned anchor and augmenting it with rule-verifiable RLVR data, further demonstrate the flexibility and generalizability of the framework. Code is available at this https URL. 

---
# BareWave: Waveform-Native Flow-Matching Text-to-Speech 

**Authors**: Wei Fan, Chao-Hong Tan, Qian Chen, Wen Wang, Xiangang Li, Kejiang Chen, Weiming Zhang, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09048)  

**Abstract**: Removing intermediate representations and separately trained decoding stages has become an important direction in generative modeling. In text-to-speech, however, high-quality systems are still commonly built through an intermediate acoustic representation before waveform synthesis. In this work, we present BareWave, a fully waveform-native framework for direct text-to-wave generation in flow-matching TTS. We consider this setting to raise three training challenges: raw-waveform modeling lacks a strong pretrained representational scaffold, different stages of training benefit from different noise schedules, and data-space perceptual objectives do not automatically share the temporal structure of the velocity-space flow objective. As a result, direct waveform training is hard to optimize efficiently, hard to push toward a strong final operating point with a fixed recipe, and hard to integrate effective perceptual refinement. Guided by this view, we develop a direct text-to-wave training framework that combines training-time representation alignment, staged noise scheduling, and velocity-aware perceptual alignment (VAPA), while preserving a single waveform-native inference path without pretrained components at test time. Experiments on zero-shot voice cloning show that strong intelligibility, speaker similarity, and naturalness can be achieved under a fully waveform-native inference path, supporting waveform-native flow-matching TTS as a practical direction. Project page with audio demos is available at this https URL. 

---
# TRIAGE: Dialectical Reasoning for Explainable Risk Prediction on Irregularly Sampled Medical Time Series with LLMs 

**Authors**: Hyeongwon Jang, Gyouk Chu, Changhun Kim, Joonhyung Park, Hangyul Yoon, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09030)  

**Abstract**: Clinical early warning systems built on electronic health records, in which clinical observations are recorded as irregularly sampled medical time series (ISMTS), must deliver both calibrated risk scores for patient triage and interpretable rationales that clinicians can verify. Large Language Models (LLMs) have been explored for this task, yet they collapse graded clinical risk into overconfident binary predictions. This risk polarization undermines both calibration and cross-patient comparability. To address this, we propose TRIAGE, a framework that trains an LLM to generate dialectical reasoning over competing clinical outcomes by eliciting outcome-specific rationales. This dialectical formulation mitigates risk polarization, enabling a single LLM to yield continuous risk scores grounded in explicit clinical reasoning. Evaluated on three ISMTS benchmarks, TRIAGE achieves an average AUPRC improvement of 3.3% and reduces calibration error by 81% compared to the competitive baselines. An LLM-as-a-judge assessment further shows that our rationales surpass post-hoc explanations from the baseline by 20% in clinical reasoning quality. The source code is available at this https URL . 

---
# ATM: Action-Consistency Transfer Matrix for Diagnosing and Improving Latent World Models 

**Authors**: Jiaheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.09028)  

**Abstract**: Latent world models are increasingly used for control and goal-conditioned planning, yet assessing whether their learned representations are useful for planning usually requires slow, planner-coupled simulator evaluation with CEM or similar planners. Such evaluation is black-box and model-complexity-dependent: under the same protocol, different world models may require minutes to hours per checkpoint. In this work, we propose ATM, an Action-Consistency Transfer Matrix for diagnosing whether latent transitions preserve action semantics relevant to planning. ATM compares action information in real encoded transitions and model-predicted transitions through lightweight post-hoc probes, producing an interpretable matrix that reveals representation quality, transition-domain inconsistency, and failure modes without simulator rollout. It can also be collapsed into a simple screening score for within-task ranking across checkpoints, variants, and world models. When the true success gap is non-trivial, ATM achieves highly reliable pairwise ranking, while reducing minutes-to-hours CEM evaluation to seconds-level transition analysis, yielding more than 100x speedup in our setup. We further introduce AITS, showing that action-identifiability is not only diagnostic but also a useful training signal for improving downstream planning without changing the planner. 

---
# SafeRun: Enabling Determinism in LLM Planning for Running 

**Authors**: Meilin Chen, Zepeng Zhai, Jiaxuan Zhao, Yuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2606.09027)  

**Abstract**: Large Language Models enable flexible natural-language planning but remain unreliable in determinism-critical domains due to their probabilistic nature. This limitation is especially problematic in running planning, where violating safety rules can lead to safety risks. We propose SafeRun, a framework for deterministic LLM-based planning via a decoupled architecture. SafeRun separates soft interpretation by an LLM from hard constraint enforcement by a deterministic solver, ensuring strict safety constraints while preserving natural-language flexibility. To validate SafeRun, we build a comprehensive benchmark for running planning under realistic physiological and safety constraints. Experiments across five LLMs show that SafeRun achieves 100\% safety score (vs.\ 79.1\% PE average and 97.6\% CodeAct average) while maintaining competitive instruction-following scores. The SafeRun benchmark is publicly available at \href{this https URL}{huggingface}. 

---
# TLDR: Compressing Audio Tokens for Efficient Autoregressive Text-to-Speech 

**Authors**: Yejin Lee, Junwon Moon, Hyoeun Kim, Hyunjin Choi, Heeseung Kim, Kyuhong Shim  

**Link**: [PDF](https://arxiv.org/pdf/2606.09019)  

**Abstract**: Codec-based autoregressive (AR) speech language models have achieved strong text-to-speech (TTS) quality by modeling speech as sequences of discrete audio tokens with large pretrained backbones. However, this token-level formulation creates a structural efficiency bottleneck: speech-token sequences are much longer than text sequences, requiring the AR backbone to perform causal computation at every token position and maintain a KV cache that grows with the sequence length. We introduce TLDR, a patch-based autoregressive framework that accelerates codec-based AR-TTS by shifting the causal modeling from token-level speech sequences to patch-level sequences. TLDR groups consecutive codec tokens into compact latent patches using a lightweight compressor, models the resulting shorter patch sequence with a frozen pretrained AR-TTS backbone adapted by LoRA, and reconstructs fine-grained speech tokens within each patch using a speaker-conditioned extractor. With a patch size of 4, TLDR achieves a 1.8x inference speedup over the baseline AR-TTS model and reduces global KV-cache memory by up to 75%. Experimental results indicate that patch-level global causal modeling can be a practical way to reduce the inference cost of pretrained codec-based AR-TTS systems without replacing the existing modules. 

---
# Understanding Quantization-Aware Training: Gradients at Quantized Weights Bias to the Low-Loss Basin 

**Authors**: Hanyang Li, Jianhao Ma, Ying Cui  

**Link**: [PDF](https://arxiv.org/pdf/2606.09012)  

**Abstract**: Post-training quantization (PTQ) converts a trained full-precision model into low-bit weights without task-level retraining, while quantization-aware training (QAT) incorporates quantization into the training loop. Although PTQ is efficient and often accurate at moderate bitwidths, it can fail sharply at aggressive bitwidths; QAT is more expensive but can often recover the lost accuracy. We propose a unified geometric framework that explains both PTQ failure and QAT recovery. We model full-precision training as following a low-loss \emph{river} inside a wider \emph{valley}: a normal neighborhood of the river forms a nearly flat \emph{basin}, while leaving this basin incurs a sharp loss increase. When the quantization grid is comparable to the basin width, local PTQ objectives, including rounding and Hessian-based second-order reconstruction, can select a high-loss deployed quantized point outside the basin even when nearby low-loss quantized points exist. In this regime, straight-through-estimator-based QAT has a useful bias: it evaluates gradients at the deployed quantized weights while updating latent full-precision weights, causing the gradient to sense the valley wall and acquire an inward component that steers subsequent quantized iterates back into the basin. We formalize this mechanism through a local landscape model, construct a geometric PTQ failure mode, and prove finite-time QAT recovery under local quantizer-compatibility assumptions. Experiments across vision and language models under multiple neural-network quantization schemes corroborate the predicted basin-crossing failure of PTQ and the corresponding recovery mechanism of QAT. 

---
# Sustainability and Artificial Intelligence: Necessary, Challenging, and Promising Intersections 

**Authors**: Han-Teng Liao, Zijia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09006)  

**Abstract**: Both digital economy and digital technology researchers increasingly recognize the need to better address the role that artificial intelligence (AI) plays in shaping the evolution of the environmental, social and governance aspects of development. It appears that sustainability and AI research converge on the features of wicked problems that are complex, interconnected and dynamic. Building off such convergence, this article aims to map out the necessary, challenging, and promising intersections by providing an overview of the state of art research. Based on 541 bibliographic data collected from the Web of Science (WoS) database, the findings reveal the increasingly central body of work on green and sustainable science and technology in bridging various disciplines, main journals and key topics and concepts. The findings reveal how such interactions can be necessary, challenging, and promising. The article concludes with few general arguments regarding how to diversify and expand the community of practice regarding AI for sustainable development, especially in the areas of expected AI application areas and institutions. 

---
# SpaceVLN: A Zero-Shot Vision-and-Language Navigation Agent with Online Spatial Cognitive Memory and Reasoning 

**Authors**: Yucheng Deng, Pingrui Lai, Xinhai Li, Chenjia Bai, Xiaoheng Deng, Chengnuo Sun, Xuelong Li, Hua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08992)  

**Abstract**: Vision-and-Language Navigation in continuous environments requires agents to understand the spatial structure of previously unseen environments in order to follow language instructions. Although foundation models have opened a promising path toward zero-shot navigation without task-specific policy training, many navigators still rely on local visual cues and linear history-based reasoning, overlooking the spatial nature of navigation across explored regions, traversed paths, landmarks, and their spatial relations. In this paper, we propose SpaceVLN, a navigation agent built around Spatial Cognitive Memory and Task-Guided Spatial Reasoning. Specifically, SpaceVLN introduces an efficient stagewise closed-loop framework where planning and execution are organized around verifiable space--landmark stages. During navigation, the agent progressively abstracts explored regions into Spatial Waypoints and dynamically maintains subtask-grounded landmark evidence, forming a hierarchical Spatial Cognitive Memory for progress localization and spatial-relation understanding. Built on this memory, Spatial-CoT integrates task-progress reasoning with spatial perception, analysis, and prediction, enabling Task-Guided Spatial Reasoning for embodied navigation. The unified stage interface enables SpaceVLN to address both Vision-and-Language Navigation and Object-Goal Navigation under a unified zero-shot setting, without task-specific policy training. Across R2R-CE, RxR-CE, GN-Bench, and HM3D-OVON, SpaceVLN achieves state-of-the-art zero-shot performance, and real-robot deployment further validates its applicability. These results highlight Spatial Cognitive Memory and Task-Guided Spatial Reasoning as a practical foundation for stronger embodied navigation agents. 

---
# CARE: A Conformal Safety Layer for Medical Summarization 

**Authors**: Suhana Bedi, Bridget Lin, Anson Y. Zhou, Chloe O. Stanwyck, Jenelle A. Jindal, Sanmi Koyejo, David Stutz, Nigam H. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2606.08969)  

**Abstract**: Large language models (LLMs) are increasingly used for medical summarization, but their outputs can omit medically important information and introduce unsupported claims. Existing error-detection methods produce heuristic or uncalibrated scores, providing no formal control over missed errors and no principled way to trade off safety against clinician review burden. We introduce Conformal Assessment for Risk Evaluation (CARE), a post-hoc, model-agnostic safety layer that uses conformal risk control to overlay calibrated omission and hallucination flags onto summaries from any LLM without retraining. CARE provides finite-sample, distribution-free guarantees through two controllers: a hallucination controller that bounds the probability of a document containing any unflagged hallucinated sentence, and an omission controller that bounds the expected fraction of important omissions not surfaced for review. Unlike hallucination detection, omissions depend jointly on whether a source sentence is important and whether it is covered by the summary. We show that calibrating only one dimension can violate the target risk bound, while marginal decompositions remain valid but overly conservative. By jointly calibrating over the full $(\tau,\gamma)$ threshold space, CARE preserves formal guarantees while surfacing up to 5$\times$ fewer sentences than alternative calibrated baselines. Across five medical summarization tasks, CARE satisfies the target risk bound at $\alpha = 0.15$ with 95% confidence across 100 calibration/test resplits, using only ~100 labeled documents per domain. In a preliminary clinician study (75 document reviews), calibrated flags improved omission detection by 28.6 percentage points on average. These results show that sentence-level safety guarantees are feasible for LLM-assisted medical summarization and offer a tunable mechanism for balancing residual risk and review effort. 

---
# Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops 

**Authors**: Ziqian Zhong, Ivgeni Segal, Ivan Bercovich, Shashwat Saxena, Kexun Zhang, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08960)  

**Abstract**: Agent benchmarks score submissions with outcome verifiers that are typically hand-written and brittle, leaving them open to reward hacking. We audit 1,968 tasks across five terminal-agent benchmarks and find 323 (16%) hackable by frontier models given only the task description. This corrupts both leaderboard rankings and RL training signal, yet the standard response is manual and reactive.
We introduce the hacker-fixer loop, a method for building exploit-resistant verifiers without per-task manual patching. The loop alternates three LLM agents: a hacker tries to pass the verifier without solving the task, a fixer patches the verifier to reject each discovered exploit, and a solver confirms the patched verifier still admits legitimate solutions. The loop iterates: each patch reshapes what the verifier rewards, surfacing the next exploit. We further add verifier access, and let patches transfer across tasks, to broaden the exploits the loop discovers.
On KernelBench, the loop drives the attack success rate from 62% to 0% on a held-out corpus of publicly reported exploits. We also find that weaker agents in the loop can defend against much stronger hackers: Gemini 3 Flash's loop drives the stronger Gemini 3.1 Pro and Claude Opus 4.7's attack success rate from 76% and 61% to 0% on KernelBench, and Gemini 3.1 Pro's from 39% to 17% on Terminal Bench across 77 tasks. We release Terminal Wrench (323 hackable environments, 3,632 hack trajectories) as a snapshot of the current attack surface, our patched verifiers, the exploits the loop discovered, and our implementation as a basis for future work. 

---
# NutriMLLM: Multimodal Large Language Models for Dietary Micronutrient Analysis 

**Authors**: Runze Yan, Minxiao Wang, Jiaying Lu, Darren Liu, Xiao Hu, Hanqi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2606.08948)  

**Abstract**: Comprehensive estimation of dietary micronutrients from food images could improve clinical nutrition care, but training such models requires large multimodal datasets linking diverse foods to complete nutrient profiles. We first show that existing multimodal large language models (MLLMs), including leading proprietary models, are unreliable for this task. Across five model families and four independent evaluation benchmarks (ASA24, SNAPMe, FNDDS, and NutriBench), models frequently abstained or returned statistically implausible values. To address this gap without costly expert annotation, we repurposed a decade of population-scale 24-hour dietary recalls as structured prompts for text-to-image generation. This pipeline produced a synthetic corpus of about 1.1 million image-description-nutrient triplets, each pairing a generated food image with a complete 65-nutrient label. To our knowledge, this is the largest synthetic food-image corpus with comprehensive micronutrient annotation planned for public release upon publication. Fine-tuning Qwen3-VL (2B/4B/8B/30B) and GLM-4.6V-Flash on this corpus yielded NutriMLLM, the first family of vision-language models specialized for comprehensive dietary micronutrient estimation. We evaluate these models with a four-component framework that separately measures abstention, hallucination, overall usability, and per-nutrient numerical accuracy. On real food images, every NutriMLLM variant achieved near-complete coverage across all 65 nutrients, and the largest variant matched or exceeded proprietary baselines (GPT-5, Gemini 3, and Claude Sonnet 4.5) in accuracy on most nutrients. These results show that recall-driven synthetic supervision can make image-based comprehensive micronutrient estimation a tractable engineering problem and support dietary assessment, personalized nutrition guidance, and population-scale micronutrient surveillance. 

---
# PACT: Learning Diverse Diagnostic Strategies via Privileged Synthesis and Branch Consensus 

**Authors**: Gen Li, Yuanze Hu, Zhichao Yang, Qingchen Yu, Jianwei Lv, Yue Guo, Yujing Liu, Faguo Wu, Hongwei Zheng, Xiandong Li, Bo Yuan, Yifan Sun, Zhaoxin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08938)  

**Abstract**: Clinical diagnosis requires flexible use of multiple reasoning paradigms under incomplete patient information. Existing LLM-based medical agents show strong medical reasoning ability, but single-paradigm or naively mixed dialogue supervision makes these paradigms difficult to learn without interference. We propose \textbf{PACT} (Periodic Anchor Consensus Training), a framework that couples supervised multi-paradigm dialogue synthesis with consensus-based Branch training. At the data level, \textbf{DPS} (Doctor-Patient-Supervisor) uses complete electronic medical records (EMRs) for quality control while keeping the doctor agent restricted to patient-visible information. This produces validated dialogues under four diagnostic reasoning paradigms without leaking hidden clinical answers. At the training level, PACT trains one paradigm-specific LoRA Branch per paradigm and periodically aggregates Branches into a shared Anchor through sign consensus. We further construct a dynamic multi-turn Chinese medical diagnosis benchmark for interactive consultation. Experiments show that PACT achieves state-of-the-art performance among compared proprietary, medical-specialized, and task-adapted baselines on diagnostic outcome and consultation-process metrics. 

---
# Report on CHIIR 2026 Workshop on Generative AI and Academic Search (GAI&AS) 

**Authors**: Yifan Liu, Jaime Arguello, Orland Hoeber, Chang Liu, Soo Young Rieh, Luanne Sinnamon, Dean Alvarez, Susan Archambault, Rob Capra, Henson Chen, Charles Costa, Anita Crescenzi, Zhitong, Guan, Jacek Gwizdka, Pao-Pei Huang, Gavindya Jayawardena, Ghazal Kalhor, Dagmar Kern, Oliver Koop, Alice Li, Afra Mashhadi, Gaohui Meng, Marta Micheli, Anil B. Murthy, Kevin Schott, Sebastian Schultheiß, Jiwoo Seo, Phaneendra Sivangula, Frans van der Sluis, Xiaoxuan Song, Silang Wang, Dan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08936)  

**Abstract**: This report summarizes the CHIIR 2026 Workshop on Generative AI and Academic Search (GAI\&AS), which examined how GenAI is reshaping academic search systems and research practices. The workshop brought together researchers in human information interaction and information retrieval to explore key challenges and opportunities in designing and evaluating future academic search systems that integrate GenAI, moving beyond traditional document retrieval to support summarization, recommendation, synthesis, and conversational interaction. Participants' interests and discussions focused on three thematic clusters: foundations and principles, applications and opportunities, and search-as-learning. Across these themes, the workshop highlighted the importance of academic search systems in supporting transparency, credibility, research integrity, and long-term scholarly needs, as well as in fostering higher-order cognitive processes. Participants discussed guiding theories, design principles, methodological approaches, partnerships, and community-building efforts aimed at advancing human-centered GenAI-enhanced academic search systems. Overall, the workshop demonstrated strong community interest and a diverse range of ongoing and emerging research initiatives at the intersection of GenAI and academic search. 

---
# PAI: Preserving Amplitude Information in Representation-Based Time-Series Anomaly Detection 

**Authors**: Kang Zhang, Wei Jian Lau, Shoushou Ren, Dong Lin, Joon Son Chung, Chuanhao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.08935)  

**Abstract**: Representation-based time-series anomaly detection algorithms significantly outperform other methods on diverse anomaly detection tasks. However, we notice that they suffer from a major limitation in our evaluation - their learned embeddings are often amplitude-agnostic. Losing amplitude information can degrade performance on amplitude related anomalies, and this failure is prevalent across all existing representation-based methods. To address aforementioned issues, we propose a new anomaly scoring scheme named PAI. PAI consists of two complementary modules, a diagnostic module and a final score augmentation function. The diagnostic module compares cosine and Euclidean scoring on the same representation bank to test whether amplitude information is already captured in the learned representation. Then in final score augmentation function, PAI computes a point-wise median and MAD deviation score and a local mean-shift score-which are fused with the representation score to produce the final anomaly score. On the TSB-AD-U-Eva and TAB UV datasets, PAI improves all four evaluated representation-based methods across every reported metric, achieving average VUS-PR gains of 98.4% and 36.8%, respectively. Among all evaluated combinations, PaAno + PAI achieves the best performance, outperforming the state-of-the-art method by 15%. Further evaluation on bootstrap confidence intervals, anomaly-type breakdowns, and a TS2Vec input-normalization ablation further support the proposed scheme. These results suggest that explicitly retaining amplitude information is important for representation-based time-series anomaly detection, which has been underemphasized in existing scoring schemes. Code is available at: this https URL 

---
# From Statute to Control Flow: Span-Grounded Deontic Trees for Defeasible Scope Parsing 

**Authors**: Jian Chen, Siyuan Li, Chucheng Wan, Zixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08932)  

**Abstract**: Rule-following agents tasked with executing policies and regulations often fail via Silent Scope Omission (SSO): a model applies a general rule but silently drops nested exceptions or counter-exceptions, producing outputs that appear compliant yet break on important edge cases. Although such failures are often framed as an agentic-systems problem, the underlying bottleneck is statutory and policy understanding, a capability typically studied in legal NLP. However, most existing legal NLP benchmarks emphasize end-task outcomes, which can overlook the structural omissions that cause SSO. To diagnose and mitigate SSO, we introduce NormBench, a benchmark of 2,290 provisions spanning Chinese (laws and local policies), English (U.S. tax law, GDPR, and corporate policies), and cross-lingual settings, designed for defeasible scope parsing: identifying precisely which clause overrides which. NormBench uses Span-Grounded Deontic Trees (SG-DT), a compiler-style intermediate representation that anchors every logical branch to source spans and requires explicit exclusion guards, enabling deterministic compilation and audit. Evaluations of frontier LLMs reveal two recurring pathologies: (1) Recursion Decay, where performance drops sharply as defeater depth increases, and (2) an Auditability Trap, where models retrieve relevant spans but fail to assemble correct control flow. Using SG-DT as a constrained intermediate output improves whole-tree fidelity and defeater recovery, and downstream experiments show that its utility is mechanism-specific: gains concentrate on exception-active, SSO-prone cases, while aggregate accuracy can be mixed when the added structure is unnecessary or parser fidelity is low. 

---
# PolyBuild: An End-to-End Method for Polygonal Building Contour Extraction from High-Resolution Remote Sensing Images 

**Authors**: Yaoteng Zhang, Julin Zhang, Guangshuai Wang, Jiwei Deng, Hui Sheng, Yasir Muhammad, Shiqing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.08920)  

**Abstract**: Extracting building polygon contours from high-resolution remote sensing images is a fundamental task for various mapping applications. However, the presence of varying imaging conditions and complex building structures, makes automatic contour extraction extremely challenging. Mainstream approaches for building extraction often rely on pixel-level segmentation followed by multiple post-processing steps to produce building contour, which can be computationally intensive and prone to errors. In this paper, we propose an end-to-end method named PolyBuild, which can directly extract building vector polygons from high-resolution remote sensing images without the need for any post-processing operations. The proposed method leverages two primary modules: an Initial Contour Generation Module (ICGM) and a Contour Optimization Module (COM). The ICGM is designed to generate an initial building contour by utilizing concatenated sub-region center features for each building instance. It performs simultaneous object detection and initial contour extraction by generating bounding boxes and using the center features of four sub-regions to represent each building. The Contour Optimization Module (COM) further refines the generated building contours by iteratively integrating Convolutional Neural Network (CNN) features and contour positional information in a Transformer-based decoder. The hybrid CNN-Transformer architecture effectively captures both local and global spatial relationships within the building contour, ensuring high-quality boundary delineation. Extensive experiments are conducted on three building datasets to evaluate the performance of PolyBuild. The results demonstrate that PolyBuild significantly outperforms state-of-the-art methods, including mask-based and contour-based approaches. 

---
# Failure-Aware Refinement of Vision-Language Model for Lithography Defect Detection 

**Authors**: Pangyun Jeong, Jiyeong Kong, Yuehua Hu, Dohee Jeong, Kyung-Tae Kang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08908)  

**Abstract**: Semiconductor lithography inspection requires reliable detection of small pattern defects such as bridge, burr, pinch, and contamination. In this study, we propose a two-stage vision-language framework that combines initial defect detection with prediction refinement. In the first stage, Qwen3-VL is fine-tuned with LoRA as a vision-language adapter to predict defect counts, defect categories, and normalized bounding boxes from lithography images. However, direct fine-tuning may still produce common test-time errors, including false positives, missed defects, and incorrect defect types. To address this limitation, the second stage trains a refinement module using first-stage prediction failures and their corrected labels, allowing the model to review and revise initial outputs. By learning from cases where the initial adapter fails, the refinement process improves defect inference beyond single-stage fine-tuning. 

---
# Few-shot Class-variable Incremental Audio Classification via Prototype Adaptation and Pseudo Class-variable Training 

**Authors**: Yanxiong Li, Guoqing Chen, Qianqian Li, Sen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08898)  

**Abstract**: In the task of few-shot class-incremental audio classification, the number of classes is assumed to always increase without considering the possibility of decrease. However, the number of classes generally increases or decreases in practice. In this paper, we investigate a problem of Few-shot Class-variable Incremental Audio Classification (FCIAC), in which the number of classes increases or decreases. We propose a FCIAC method using prototype adaptation and pseudo class-variable training. The model in our method consists of an encoder and a classifier. The classifier is initialized by a class-variable prototype adaptation network, whose structure dynamically changes with the change of classes. In addition, we design a pseudo class-variable training strategy to enhance the model's adaptability to changing classes. Experiments on three public datasets show that our method exceeds previous methods in average accuracy. The code is at: this https URL. 

---
# A multi-agent system for spine MRI report generation from multi-sequence imaging 

**Authors**: Zhiping Xiao, Junwei Yang, Gongbo Sun, Han Zhang, Hanwen Xu, Yi Yao, Zachary D. Miller, William E. King III, Mohammed M. Kanani, Jalal B. Andre, Sammy Chu, Ming Zhang, Paul E. Kinahan, Nathan M. Cross, Sheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08897)  

**Abstract**: Spinal pathology is a leading cause of pain and disability worldwide. Spine MRI is central to clinical evaluation, yet its interpretation remains complex and time-consuming, requiring integration of information across multiple imaging sequences and anatomical regions. Despite recent advances in automated MRI analysis, effectively combining multi-sequence data while preserving sequence-specific diagnostic information remains an open challenge. Here we present SpineAgent, a multi-agent framework for spine MRI report generation built upon a multi-sequence foundation model trained on routine clinical data from 32,047 patients and 453,683 MRI series, comprising a total of 13,441,191 MRI slices. To accommodate diverse modalities of sequences, we first pre-train two DINOv3-based encoders separately on T1- and T2-weighted sequences. We then introduce a continual training strategy that learns a synthesizer to embed images of other sequences using the T1 and T2 encoders, producing patient-level embedding that integrates various signals across MRI sequences. Using these embeddings, SpineAgent achieves state-of-the-art performance, and demonstrates strong generalizability under cross-manufacturer and cross-cohort evaluation. Beyond classification, SpineAgent enables pathology localization by identifying findings-relevant slices and segmenting pathological regions. It also supports multimodal image-report retrieval, providing a solid foundation for scalable and explainable MRI report generation. We further integrate these validated capabilities of SpineAgent into 37 specialized agents. Finally, we incorporate their outputs as structured tokens within a Medical Report Agent trained end-to-end for report generation. Through both automated metrics and expert evaluation by five radiologists, SpineAgent achieves leading performance in spine MRI report generation. 

---
# Cheap Reward Hacking Detection 

**Authors**: Iván Belenky, Joaquín Itria, Steven Johns  

**Link**: [PDF](https://arxiv.org/pdf/2606.08893)  

**Abstract**: A small transformer encoder is trained to map Terminal-Wrench trajectories onto a unit sphere where embedding distance approximates the $L_1$ distance between reward and metadata signals. A linear probe on top of that embedding detects reward hacking on the cleaned test split with AUC $0.9467$ and TPR@5%FPR $0.8296$, matching the TW sanitized LLM-as-judge AUC ($0.9510$ on the cleaned split) and exceeding its TPR@5%FPR ($0.7130$ vs $0.8296$) on the same information condition, at roughly four orders of magnitude lower per-trajectory cost. The encoder is not a pure behavior reader: stripping natural-language reasoning from its input at probe time drops AUC to $0.6213$. 

---
# Benchmarking Vision-Language-Action Models on SO-101: Failure and Recovery Analysis 

**Authors**: Yi Yu, Xinchuan Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08881)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated strong generalization in robotic manipulation, yet existing evaluations are primarily conducted in simulation or on expensive robotic platforms, leaving their robustness on affordable real-world robots largely unexplored. We present a standardized real-world benchmark for evaluating representative VLA and imitation learning policies on the low-cost SO-101 robotic platform. The benchmark comprises four representative manipulation tasks together with unified evaluation protocols, enabling systematic comparison under embodiment uncertainty. Using real-world teleoperated demonstrations, we fine-tune and evaluate $\pi_{0.5}$, SmolVLA, Wall-X, and ACT directly on the physical platform. Beyond conventional task success rates, the benchmark incorporates a structured failure taxonomy, semantic- and execution-level failure decomposition, and recovery-aware evaluation metrics to characterize policy robustness. Experimental results show that stronger pretrained VLA policies generally outperform the imitation learning baseline, although performance remains highly task-dependent under low-cost robotic deployment conditions. Execution instability emerges as the dominant failure source, while recovery capability varies substantially across architectures. These results highlight the importance of failure and recovery analysis beyond binary task success and establish SO-101 as a practical benchmark for evaluating embodied AI systems under realistic low-cost robotic deployment conditions. 

---
# Intelligent Character Recognition of Handwritten Forms with Deep Neural Networks 

**Authors**: Hartwig Grabowski  

**Link**: [PDF](https://arxiv.org/pdf/2606.08858)  

**Abstract**: The automatic processing of handwritten forms remains a challenging task, wherein detection and subsequent classification of handwritten characters are essential steps. We describe a novel approach, in which both steps -- detection and classification -- are executed in one task through a deep neural network. Therefore, training data is not annotated by hand, but manufactured artificially from the underlying forms and yet existing datasets. It can be demonstrated that this single-task approach is superior in comparison to the state-of-the-art two-task approach. The current study focuses on hand-written Latin letters and employs the EMNIST data set. However, limitations were identified with this data set, necessitating further customization. Finally, an overall recognition rate of 88.28 percent was attained on real data obtained from a written exam. 

---
# sGPO: Trading Inference FLOPs for Training Efficiency in RLVR 

**Authors**: Shivchander Sudalairaj, Kai Xu, Akash Srivastava, Giorgio Giannone  

**Link**: [PDF](https://arxiv.org/pdf/2606.08854)  

**Abstract**: Standard Reinforcement Learning with Verifiable Rewards (RLVR) training allocates a fixed rollout budget to every query, without regard for what each query's difficulty means for the current policy. This leads to two symmetric failure modes: easy queries produce near-zero advantage because the policy already solves them, while unsolvable queries produce no signal because the policy never solves them. Both regimes waste training FLOPs without contributing to a learning gradient. We introduce sorted Group Policy Optimization (sGPO), a compute-efficient strategy that trades a small budget of inference FLOPs for a large reduction in wasted training FLOPs. The key insight is that cheap inference compute can serve as a single offline proxy for query difficulty. By generating a small batch of parallel samples per query under the initial policy, we obtain a model-aware empirical success rate. This motivates setting the training rollout group size to the inverse of this success rate, a practical rule that maximizes sample efficiency by extracting the most advantage per generated rollout. This single profiling pass simultaneously drives data filtering (removing trivial queries and sub-sampling unsolvable ones), adaptive group size allocation, and curriculum construction (scheduling queries from easy to hard). sGPO matches or exceeds baseline performance while reducing total training compute by a factor of three, with the upfront inference profiling cost included. 

---
# Intrinsic Selection and Particle Resampling for Inference-Time Scaling Beyond Domain Verifiability 

**Authors**: Giorgio Giannone, Mustafa Eyceoz, Shabana Baig, Shivchander Sudalairaj, Anna C. Doris, Faez Ahmed, Akash Srivastava, Kai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08850)  

**Abstract**: Inference-Time Scaling (ITS) has largely succeeded in verifiable domains like math and coding, where cheap verification enables scalable output selection. However, extending ITS to tasks prone to systematic failure - driven by faulty initial assumptions or unmet multidimensional constraints - typically relies on costly external solvers or brittle, model-based verifiers. Our key insight is that the intrinsic statistics of parallel sample sets, specifically length-adjusted tail entropy, provide a robust discriminative signal for solution quality without access to ground truth. Crucially, these statistics serve as a difficulty gate for adaptive compute allocation, dynamically routing problems across scaling regimes. First, Intrinsic Selection (iS) ranks candidates post-hoc, matching consensus-based algorithms across three domains and improving engineering design selection by 20% over pass@1 baselines. Second, Intrinsic Particle Filtering (iPF) generalizes this to step-level resampling, guiding generation toward high-confidence reasoning trajectories to improve pass@1 by 6.1 points on average on hard math problems. Finally, Particle Distillation (dPF) injects privileged guidance via early logit blending and KL-guided resampling, steering generation past systematic reasoning errors to satisfy expert rubrics, yielding up to 26.5% gains on complex clinical responses. Our pipeline applies seamlessly across broad-purpose, domain-specialized, and multimodal architectures, successfully extending ITS to open-ended domains without requiring trained reward models or exact ground-truth verification. 

---
# BLM-SGAN: Bidirectional Language Modeling for Semantic-Spatial Text-to-Image Generation 

**Authors**: Ahmed Abdelmoneim Mazrou, Haidy Maher El-Amir, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2606.08847)  

**Abstract**: Despite the success of image generation from text descriptions, it still faces challenges that are difficult to overcome in domains such as natural language processing (NLP) and computer vision (CV). Recent advancements in text-to-image (T2I) models, particularly those utilizing generative adversarial networks (GANs), have significantly improved the synthesis of realistic images across various domains. However, existing GAN-based T2I models still encounter key challenges, such as difficulty in capturing long-range dependencies, vanishing gradients, and the limitations of sequential processing. To address these issues, we introduce BLM-SGAN, a novel model that incorporates Bidirectional Language Modeling for Semantic-Spatial Text-to-Image Generation. BLM-SGAN leverages BERT's attention mechanisms to capture rich contextual information and efficiently manage extended sequences. Our model demonstrates state-of-the-art performance, with an Inception Score (IS) of 5.45 +/- 0.08, surpassing several competitive models such as SSA-GAN, DF-GAN, SD-GAN, and AttnGAN. BLM-SGAN effectively generates highly realistic images of birds from detailed text descriptions. The implementation code is available at: this https URL. 

---
# Knowledge Graphs and Reasoning LLMs for Finding Simple Yet Effective Transcriptomic Perturbation Predictors 

**Authors**: Jake Fawkes, Liam Hodgson, Jason Hartford  

**Link**: [PDF](https://arxiv.org/pdf/2606.08816)  

**Abstract**: Predicting the effect of an unseen gene knockout perturbation on transcriptomic gene expression remains a highly challenging problem for virtual cell models. Recent progress has been made by leveraging biological knowledge graphs to provide a notion of similar perturbation, allowing for improved extrapolation beyond the set of training perturbations. In this work, we demonstrate that the simplest model to leverage these assumptions - a K-nearest neighbour from the knowledge graph - achieves highly competitive performance on this task, and that this can be improved further using LLMs optimised via reinforcement learning (RL) for predictive performance. Specifically, we find that the K-nearest neighbour approach beats almost all methods on out-of-distribution perturbation prediction, and when a reasoning LLM is trained via RL to make changes to the neighbourhood, it obtains equivalent performance to current state of the art methods on the cell lines from Replogle et al. (2022). We also demonstrate that the RL training improves the LLM's performance on the downstream task of differential expression prediction, despite not being trained on this directly. Overall, these findings demonstrate the efficacy of knowledge graphs as model priors, and show early signs that RL can refine LLMs into generalizable tools for predicting complex biological responses. 

---
# Governance Controls for AI-Generated Test Artifacts in Autonomous Software Testing 

**Authors**: Dimple Bajaj, Deepak Khetan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08806)  

**Abstract**: Artificial Intelligence (AI) and Large Language Models (LLMs) are increasingly used in autonomous software testing; however, AI-generated test artifacts often suffer from hallucinations, compliance violations, security risks, and limited explainability. To enhance the reliability, transparency, and trustworthiness of AI-generated testing artifacts, this research introduces the concept of Governance-Aware Autonomous Testing Framework (GATF). The framework extends the autonomous testing lifecycle with governance validation, explainability analysis, probabilistic risk assessment, compliance monitoring, as well as audit governance. Experiments were performed with Defects4J and PROMISE software engineering datasets. The proposed framework successfully reduced the governance-related risks by 89.6% and demonstrated 94.3% accuracy in governance, 96.5% artifact reliability, 94.2% compliance accuracy, and 90.8% explainability performance. The results show that autonomous testing systems that are governance-aware can significantly enhance the reliability, transparency, and operational security of autonomous testing systems in comparison to conventional AI-based testing systems. The proposed architecture is scalable and reliable and provides a safe environment for software testing. 

---
# Scaling Decision-Focused Learning to Large Problems with Lagrangian Decomposition 

**Authors**: Stéphane Eilles-Chan Way, Hugo Percot, Quentin Cappart, Tias Guns, Louis-Martin Rousseau  

**Link**: [PDF](https://arxiv.org/pdf/2606.08797)  

**Abstract**: Decision-focused learning has shown great promise for addressing predict-then-optimize problems, particularly in the presence of under-specified models. However, its practical deployment is often hindered by high computational costs and limited scalability, as it requires solving a constrained optimization problem for each training instance at every iteration. To address these challenges, we propose a novel framework that incorporates Lagrangian decomposition into the decision-focused learning paradigm. Specifically, we introduce a new surrogate objective along with two loss functions for evaluating and training the underlying prediction model. We further propose two variants of our approach, which offer different trade-offs between computational efficiency and solution quality. Our framework can be seamlessly integrated with standard decision-focused learning methods, including Smart Predict-then-Optimize (SPO+) and Implicit Maximum Likelihood Estimation (IMLE). Through experiments on two standard benchmarks, the multi-dimensional knapsack problem and quadratic portfolio optimization, we demonstrate that our approach achieves competitive performance while remaining amenable to parallelization. In particular, it consistently outperforms traditional decision-focused learning methods on large-scale instances, involving up to eight times more variables than those typically considered in related work. The implementation is available at this https URL. 

---
# AI-Augmented Closed-Loop Quality Engineering: A Reference Architecture for Continuous Software Quality Intelligence 

**Authors**: Dimple Bajaj  

**Link**: [PDF](https://arxiv.org/pdf/2606.08793)  

**Abstract**: The quality of software engineering is still under a challenge due to disjointed processes between requirements, testing, and production, which hinders the opportunity to implement quality strategies in consecutive releases. Existing approaches tend to be fixed-model or single-optimization approaches and lack production feedback learning mechanisms. The paper at hand proposes a closed-loop reference architecture of continuous software quality intelligence with AI enhancements. The model synthesizes requirement feature mining, risk-based test prioritization, defect prediction, and production incident analysis as an element of a feedback-based pipeline. A limited feedback learning model is introduced that is used to propagate the production signal-based on defect severity and incident impact- to the following release to ensure stability, and the time. The method is evaluated using a semi-synthetic test dataset of 4,500 requirements, 27,049 test cases, 13,089 defects and 7,841 incidents in six release cycles. The experimental results show that the proposed system reduces the defect leakage by 0.19 to 0.13, increases the effectiveness of the detection system to 0.72 to 0.84, and shortens the test execution by up to 35 percent compared to the non-adaptive baselines. The changes are stable release to release. The findings indicate that through the integration of feedback-based learning in a closed-loop architecture, it can be continued to enhance quality process, which offers practical foundation of adaptive quality engineering of software. 

---
# Evaluating AI Investment Strategies 

**Authors**: Irene Aldridge  

**Link**: [PDF](https://arxiv.org/pdf/2606.08791)  

**Abstract**: We study the problem of auditing a black-box algorithmic decision-maker from observable inputs and outputs alone. Our main result is an exact decomposition: under precisely characterized conditions, the cumulative \emph{regret} of a dynamic policy equals the sum of per-period covariances between the cost vector and the policy's decision. This extends the single-period identity of Aldridge~(2026) to the full multi-period setting of stochastic dynamic programming.
We prove the identity holds exactly under i.i.d. costs and mean-unbiased Markov policies, derive closed-form bias corrections for non-stationary and time-varying cases, and establish the discounted-horizon analog. A Bellman recursion for the covariance regret functional connects the result to standard reinforcement learning algorithms; for rolling-window policies, the estimation-error bias is $O(d/w)$.
The decomposition has direct implications for algorithmic auditing in strategic environments: in platform mechanism design, it provides a welfare-based audit metric without access to the agent's private type; in repeated games, covariance reduction is a sufficient condition for policy improvement; in procurement and ad auctions, the bias correction quantifies welfare loss from strategic misreporting. The associated trajectory estimator is consistent, asymptotically normal with HAC variance, and computable in $O(T \cdot nd)$ time. This makes the proposed approach a tractable, model-free audit tool for platform mechanisms, algorithmic portfolio strategies, and any sequential decision system subject to external performance review. 

---
# How Many Counterfactuals Does It Take? Probing VLM Hallucinations Through Circuits and Causal Effects 

**Authors**: Abhivansh Gupta, Simardeep Singh, Advika Sinha, Shreyansh Modi, Akshat Tomar  

**Link**: [PDF](https://arxiv.org/pdf/2606.08777)  

**Abstract**: Visual Language Models (VLMs) are known to produce hallucinated predictions that are not grounded in visual evidence, yet existing approaches lack a principled understanding of how robust such predictions are under counterfactual perturbations. In this work, we study the sample complexity of counterfactual robustness for hallucinated outputs in VLMs. We define a causal influence metric based on log-probability differences between factual, counterfactual, and activation-patched runs, and use it to characterize the stability of hallucinated predictions. By leveraging circuit discovery techniques (CD-T), we identify model components responsible for these predictions and track their activation differences across counterfactual samples. We then derive empirical bounds on the minimum number of counterfactual samples m required to reliably detect instability in hallucinated outputs, using concentration inequalities and variance estimates of the causal influence distribution. 

---
# Unifying Object-Centric World Models and Diffusion Policy: A Hierarchical Framework for Multi-Stage Robotic Tasks 

**Authors**: Raktim Gautam Goswami, Prashanth Krishnamurthy, Yann LeCun, Farshad Khorrami  

**Link**: [PDF](https://arxiv.org/pdf/2606.08775)  

**Abstract**: Visual world models have shown great potential in learning complex system dynamics. Recent advancements leverage these models as transition functions within Model Predictive Control (MPC) frameworks to solve various control tasks. When applied to robotics, however, they are limited to single-stage tasks such as reaching or grasping, and struggle with multi-stage ones that demand complex sequential planning. In this work, we introduce WorldDP, a world model framework designed for multi-stage robotic manipulation. Our hierarchical approach utilizes a high-level world model as a transition function to optimize for feasible subgoals during runtime, which are subsequently reached by a low-level Diffusion Policy. To further aid in learning dynamics and planning, we incorporate object-centric representations that decouple environmental entities and enable us to plan sequentially with respect to each. Evaluated across several robotics benchmarks, WorldDP consistently outperforms existing baselines, validating that coupling the world model's physically grounded planning with diffusion policy's efficient execution yields superior multi-stage performance. 

---
# TeamHerald@CHIPSAL 2026: Hate Speech Detection and Sentiment Analysis of Nepali Memes using Transformer-based Architectures and Ensemble Learning 

**Authors**: Ashish Acharya, Anish Khatiwada, Rohit Khadka, Pragya Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2606.08770)  

**Abstract**: The analysis of internet memes in the Nepali language is complicated by frequent code-mixing and a lack of established baseline resources. While memes inherently combine visual and textual elements, this study focuses on a text-centric approach by extracting embedded text using an OCR layer and modeling it with Transformer-based architectures. We evaluate six distinct models and investigate the comparative effectiveness of Hard and Soft Voting ensemble strategies across two tasks: binary hate speech detection and three-class sentiment analysis. Experimental results show that a standalone decoder-only model achieved the highest performance for binary classification, whereas the Soft Voting ensemble performed best for the multi-class sentiment task, yielding a 15.8% relative improvement in Macro F1-score over the strongest standalone baseline. These findings suggest that ensemble strategies behave differently across binary and multi-class tasks, highlighting the importance of selecting aggregation methods suited to the classification objective. 

---
# RadOT-Eval: Auditable Structured-Evidence Transport for Radiology Report Evaluation 

**Authors**: Weixin Liu, Juming Xiong, Yang Li, Qingyuan Song, Susannah Rose, Murat Kantarcioglu, Bradley Malin, Zhijun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2606.08769)  

**Abstract**: Automatic evaluation is critical for high-stakes text generation, where errors often involve omitted findings, hallucinated content, polarity reversals, location changes, uncertainty mismatches, and temporal-comparison errors rather than low surface similarity alone. Radiology report generation provides a challenging test case because generated reports must preserve structured clinical evidence across sources. We present RadOT-Eval, an interpretable structured-evidence optimal transport framework for offline auditing of radiology report generation. RadOT-Eval decomposes reference and candidate reports into attribute-structured clinical evidence units, aligns corresponding evidence using entropy-regularized optimal transport, and uses clinically meaningful side-channel discrepancies in a monotone risk model to predict error burden. All transport, feature, and readout choices are selected using the ReXVal dataset, and the frozen system is evaluated on the independent RadEvalX dataset. RadOT-Eval achieves Spearman correlations of 0.715, 0.548, and 0.399 with total, clinically significant, and clinically insignificant annotated error burden, respectively, yielding higher point estimates than standard evaluation metrics and the open-source large language model (LLM)-based evaluator GREEN-radllama2-7B. In a frozen auxiliary corruption-sensitivity stress test on ReXErr-v1, RadOT-Eval achieves 0.768 AUROC and a 0.990 corrupted-greater-than-clean paired win rate. These results show that structured evidence transport provides an auditable, rank-oriented evaluation tool for high-stakes generated clinical text under ReXVal-only model selection and frozen RadEvalX testing. 

---
# APEX4: Efficient Pure W4A4 LLM Inference via Intra-SM Compute Rebalancing 

**Authors**: Hong Guo, Nianhui Guo, Weixing Wang, Jona Otholt, Christoph Meinel, Haojin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08761)  

**Abstract**: W4A4 quantization promises full utilization of INT4 Tensor Cores, yet group dequantization overhead on CUDA Cores has driven existing systems to mixed-precision fallbacks. We present the first systematic study of how intra-SM compute balance governs this bottleneck. Through controlled benchmarks across four GPUs from Ampere and Ada architectures, we identify the Tensor Cores to CUDA Cores throughput ratio ($\rho$) as the primary hardware indicator: the W4A4-g128 kernel yields $2.0$--$2.5\times$ speedup on RTX~3090 ($\rho=16$) yet degrades to $0.43$--$0.47\times$ on A100 ($\rho=64$) in compute-bond scenarios, establishing W4A4 viability as platform-dependent rather than universally infeasible. Guided by this finding, we build \textbf{APEX4}, which co-designs pure INT4 GEMM kernels with $\rho$-aware granularity adaptation to mitigate the CUDA Cores dequantization bottleneck. APEX4 achieves perplexity within 0.63 of FP16 on LLaMA-2-70B and outperforms W4Ax Atom-g128 by 4.0\%--4.4\% in zero-shot accuracy. Deployed as a drop-in replacement in unmodified vLLM, it delivers up to $1.66\times$ end-to-end speedup on L40S ($\rho=8$), and $1.78\times$ on RTX~3090 ($\rho=16$), $2.09\times$ on A40 ($\rho=16$), while recovering A100 ($\rho=64$) to $1.20$--$1.40\times$ via the mixed-granularity mode. 

---
# Deep Active Re-Labeling: Toward Noise-Resilient Annotation Efficiency 

**Authors**: Md Abdullah Al Forhad, Weishi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2606.08718)  

**Abstract**: While Deep Active Learning (DAL) effectively reduces human annotation costs, its efficacy is constrained by human annotation errors. This is because the data sampled for active learning is assumed to be highly informative for training. When human annotators introduce errors into this informative data at a certain rate, the active learning performance drops significantly and, in some cases, even exhibits worse outcomes than passive learning. In this paper, we first analyze the impact of human annotation errors in the DAL setting. Then we propose a framework to address the human annotation noise problem for DAL. Informed by human learning patterns, the core idea of our proposed solution involves allocating a portion of the human annotation budget to re-annotate data that has already been labeled. Previous theoretical work suggests that when the model possesses a certain level of ability to identify potentially noisy data, even re-labeling a small fraction of the data can effectively remove noise from the active training set. To achieve this, we implement two active noise sampling strategies to detect noise under different circumstances and allocate a part of the annotation budget to re-annotate these instances. Our approach imbues active learning with a revisiting and introspective behavior. Our experiments demonstrate that, under the same annotation budget, our method is more data-efficient and yields a relatively noise-free annotation dataset in the end. 

---
# Hybrid Neural Network and Conventional Controller Approach for Robust Control of Highly Unstable Systems: Application to Tilt-Rotor Control 

**Authors**: Ali Kafili Gavgani, Amin Talaeizadeh, Aria Alasty, Hossein Nejat Pishkenari  

**Link**: [PDF](https://arxiv.org/pdf/2606.08714)  

**Abstract**: Multirotors are widely used in applications ranging from surveillance to precision agriculture, yet conventional designs remain limited by their under-actuation. Tilt-rotor configurations overcome this limitation by enabling full actuation. This paper investigates neural-network-based control strategies for a fully actuated tilt-rotor system with four thrust-vectoring inputs. Our work is structured in two parts. First, we deliberately present a negative result by evaluating a direct input-output control approach. In this method, multilayer perceptrons (MLPs), long short-term memory (LSTM) networks, and transformer models are trained to map system states and their desired values directly to control signals. We show that this strategy fails to stabilize the system, highlighting the inherent difficulty of applying direct input-output learning to highly unstable plants. Second, as the main contribution, we propose a neural-network-enhanced sliding mode controller (SMC). The method decomposes the system dynamics into input-independent and input-dependent components, with the former learned from a small dataset using lightweight networks, thereby reducing real-time computational demands. Moreover, the proposed method can be trained using flight logs collected from low-performance controllers, and the resulting dynamic model learned from real-world data can be used in simulation. We further compare MLP- and LSTM-based implementations under model uncertainties and external disturbances, demonstrating the robustness and effectiveness of the proposed approach; in particular, the controller with the LSTM plant dynamics predictor achieves superior performance to its MLP-based counterpart while also exhibiting lower runtime. 

---
# SNR-ST-Mix: Sample-specific Neighborhood Regression Mixup for Augmented Spatial Transcriptomics Imputation with Deep Neural Network 

**Authors**: Hongyi Yu, Yaoyu Fang, Jiahe Qian, Xinkun Wang, Lee A. Cooper, Bo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.08712)  

**Abstract**: Purpose: Spatial transcriptomics (ST) enables gene expression measurements within the tissue context. However, these measurements are often noisy, low-resolution, and sparsely sampled, which limits the recovery of fine spatial structure. Deep neural networks have become powerful tools for expression imputation from histology, but their performance remains constrained by limited sample sizes and a lack of biologically informed augmentation. Most of the existing augmentation strategies for learning are designed for classification tasks rather than regression, which neglect spatial and transcriptomic relationships, leading to biologically implausible interpolations that hinder prediction performance. Approach: To address these limitations, we propose SNR-ST-Mix, a geometry- and expression-aware data augmentation framework designed specifically for ST data. It constrains mixing to a spot's k-nearest spatial neighbors and adaptively weights interpolation coefficients based on expression similarity, generating augmented samples that preserve local biological structure while ensuring spatial smoothness. This dual conditioning yields synthetic examples that expand the effective training manifold, promote generalization, and enhance prediction stability under sample-specific training. Results: Extensive experiments with various tissue types demonstrate that SNR-ST-Mix consistently outperforms conventional augmentation methods without requiring architectural changes or additional computation. Conclusions: SNR-ST-Mix provides an effective and biologically principled augmentation strategy for spatial transcriptomics regression tasks. By explicitly leveraging spatial geometry and transcriptomic similarity, it expands the effective training manifold and improves predictive performance without increasing model complexity. 

---
# Structuring agentic AI for HPC code modernization 

**Authors**: Anthony Marinov, Igor Sfiligoi  

**Link**: [PDF](https://arxiv.org/pdf/2606.08710)  

**Abstract**: Modernization of legacy scientific codes is often necessary to keep up with the ever-evolving changes in the compute resource ecosystem. Parallelization and migration from poorly supported software ecosystems are two of the most time-consuming activities in the research software engineering field. This paper presents our experience in the successful, two-phase AI-assisted modernization of NMAP-RKPM, a roughly 60,000-line, 3D explicit solid mechanics physics engine based on the Reproducing Kernel Particle Method (RKPM). We converted this single-threaded, Fortran based MPI application into a OpenMP-parallel C++ based MPI tool in the span of a few months. While Large Language Model (LLM) based tools on their own proved inadequate, we developed a highly structured "hand-holding" agentic AI methodology, like providing manually created examples, ensuring continuous buildability and limiting session scope, that was instead highly effective. The paper provides both the AI-assisted steps that were successful and the problems that we had to overcome, alongside the reasoning behind the chosen path. 

---
# Agentic Search for Counterfactual Recourse under Fixed LLM Budgets 

**Authors**: Yasuo Tabei  

**Link**: [PDF](https://arxiv.org/pdf/2606.08696)  

**Abstract**: Counterfactual recourse aims to provide actionable feature changes that would alter an unfavorable decision made by a predictive model. In practice, affected individuals often benefit from multiple feasible alternatives rather than a single optimal explanation. A natural way to produce such alternatives is to prompt large language models (LLMs). However, prompting incurs a practical constraint: the number of LLM calls is often the dominant computational and economic cost. Together, the need for multiple alternatives and this cost constraint shift the problem from finding a single high-quality counterfactual to efficiently generating a set of oracle-validated counterfactuals under a fixed LLM-call budget. In this work, we study counterfactual recourse generation in the LLM-agentic setting as a fixed-budget search problem and propose Comp-MCTS, an agentic tree-search framework that maximizes the yield of unique, oracle-validated counterfactuals under this budget while maintaining favorable quantity--quality trade-offs. Comp-MCTS allocates the budget toward novel intervention directions via LLM-based proposal generation, oracle validation, and compression-guided pruning, in a training-free, oracle-only setting. Experiments on four real-world tabular datasets show that Comp-MCTS substantially outperforms single-candidate LATS-style baselines in the yield of unique, oracle-validated counterfactuals, and offers favorable quantity--quality--efficiency trade-offs against stronger multi-candidate variants: comparable or higher yield at similar or lower oracle-evaluation cost on three of four datasets, plus competitive proximity, sparsity, and novelty. 

---
# Activation Steering Induces Emergent Misalignment: A More Comprehensive Evaluation 

**Authors**: Qi Cao, Jian Lou, Meiting Liu, Wenjie Feng, Dan Li, See-Kiong Ng, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08682)  

**Abstract**: Activation steering has emerged as a popular inference-time technique for modulating the behavior of large language models (LLMs). By constructing a steering vector from examples of a target behavior and injecting it into intermediate activations during inference, activation steering enables flexible behavioral control while avoiding the permanent parameter updates required by finetuning. Meanwhile, recent work has identified emergent misalignment (EM) as a significant safety concern, wherein models finetuned on unsafe examples from a narrow task may unexpectedly generalize to broadly unsafe behavior on unrelated tasks. Although finetuning-induced EM has been extensively studied, whether activation steering can induce EM remains comparatively under-explored, despite its increasing use as a model-control technique. In this paper, we present a comprehensive study of activation-steering-induced emergent misalignment, substantially expanding the evaluation scope beyond existing pioneering work. First, we show that activation steering can induce broad misalignment, even in the recent Qwen-3.5 series. Moreover, activation-steered models produce harmful responses with stronger semantic relevance and higher coherence than their finetuned counterparts, making the resulting misalignment potentially more harmful. Second, we characterize properties of AS-induced EM by analyzing key steering-specific factors, including steering magnitude, the low-rank structure of the steering subspace, and the number of epochs during steering-vector construction. Third, we evaluate the robustness and sensitivity of AS-induced EM across diverse model families, model scales, target tasks, and intervention layers. Our findings reveal activation steering as a significant yet under-examined source of emergent misalignment and provide an activation-space perspective for understanding the mechanisms and safety risks of EM. 

---
# Lost in the Flow with Code Talkers: Unveiling the Instruction-Tuning Tax of Large Language Models in Code Tasks 

**Authors**: Shi Ying Chang, Chiok Yew Ho, Yichen Li, Yintong Huo  

**Link**: [PDF](https://arxiv.org/pdf/2606.08676)  

**Abstract**: AI coding assistants have significantly improved developer productivity by automatically suggesting code that aligns with user intent, and many of these tools are now integrated directly into Integrated Development Environments (IDEs). Developers interact with code in two distinct cognitive modes: Flow and Command. While developers require tools that directly complete or infill code in unfinished programs during Flow mode, they also need tools that can comprehend intentions expressed as natural-language instructions and convert them into executable code in Command mode. Although instruction-tuned Large Language Models (LLMs) dominate many application scenarios due to their abilities to infer and fulfill developers' intents, it remains unclear whether the same paradigm is equally suitable for different code-related tasks. Therefore, it is necessary to understand how instruction tuning affects the feasibility of CodeLLMs as coding assistants. To fill this gap, we conduct the first empirical study that uncovers a key trade-off caused by instruction tuning across programming modes, which we term the Instruction-Tuning Tax. Our results show that instruction tuning is not a free lunch: although instruction-tuned models are more capable of following instructions and leveraging structured guidance, these gains often come at the cost of weaker infilling performance. We further extend our study through both qualitative and quantitative analyses, including manual failure categorization, behavioral metrics that capture generation fidelity, and intermediate-checkpoint evaluation throughout the tuning process. Summarizing our results into seven findings and four implications, our study offers a new perspective on the development of AI-powered coding tools and highlights the need to carefully balance instruction-following ability with effective code generation assistance. 

---
# BioVid: Autoregressive Video Generation with Biological Behavior Semantic Comprehension 

**Authors**: Tsung-Wei Pan, Jung-Hua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08674)  

**Abstract**: Existing video generation frameworks treat sequence duration as an externally prescribed parameter -- fixed frame counts or text prompts -- producing clips whose temporal boundaries are decoupled from the statistical structure of real behavioral data. This assumption is fundamentally misaligned with biological behavior, where action duration varies naturally across individuals and instances and is encoded in the data itself. We present BioVid, a data-driven autoregressive video generation framework that learns the temporal structure of biological behaviors directly from training data, including their natural length distributions. In the first stage, a Finite Scalar Quantization GAN (FSQ-R3GAN) tokenizer encodes each video frame into a compact discrete representation, combining the stabilized relativistic training objective of R3GAN with FSQ's guaranteed codebook utilization to achieve high-fidelity spatial reconstruction without codebook collapse. In the second stage, a causal Transformer models the resulting token sequences autoregressively and learns to emit an End-of-Sequence (EOS) token when the behavioral event reaches semantic closure, with the termination distribution emerging naturally from the training data rather than any human-specified constraint. Experiments on a human drinking behavior dataset (NTU RGB+D, A001, n=94) demonstrate that BioVid's generated length distribution closely matches that of held-out test data, achieving a Wasserstein-1 distance of 1.24 against the ground truth -- compared to 6.05 for a fixed-length baseline and 15.48 for VideoGPT -- while maintaining competitive spatial fidelity. 

---
# Data Agents Under Attack: Vulnerabilities in LLM-Driven Analytical Systems 

**Authors**: Kuncan Wang, Ziting Wang, Peizhuo Lv, Haoyang Li, Guoliang Li, Gao Cong, Wei Dong  

**Link**: [PDF](https://arxiv.org/pdf/2606.08661)  

**Abstract**: Data agents integrate LLM-driven reasoning with relational data access, executable analytical tools, and multi-step workflow orchestration, making them increasingly central to enterprise analytics. This integration introduces new security vulnerabilities across data resources, database execution, and agent reasoning, recombining concerns from database security and general-purpose LLM-agent security into failure modes that neither line of work captures on its own. To address this gap, we present a systematic security study of data agents. Our contributions are threefold. First, we develop a layered vulnerability framework that identifies eight data agent-specific risks across interpretation, execution, and policy layers. Second, we introduce an attack taxonomy organized by adversary goal, tactic, and technique, covering three goals, seven tactics, and fourteen techniques, and pair it with an LLM-driven payload generation pipeline grounded in real database schemas. Third, we evaluate these attacks on six systems, including four open-source data agents and two production cloud analytics services. Our experiments reveal substantial security vulnerabilities across current systems and yield four key takeaways. 

---
# Latent Diffusion Policy: Shaping Latent Spaces for Diffusion-Based Robotic Manipulation 

**Authors**: Zhexuan Zhou, Yichen Lai, Jinhao Zhang, Huizhe Li, Youmin Gong, Jie Mei  

**Link**: [PDF](https://arxiv.org/pdf/2606.08657)  

**Abstract**: Diffusion-based visuomotor policies operating directly in raw action spaces conflate scene comprehension with trajectory generation within a single denoising process. The resulting velocity field must simultaneously encode scene information and generate precise trajectories, increasing learning complexity and limiting performance on tasks demanding precise temporal coordination across multiple arms. To simplify this joint learning problem, we introduce Latent Diffusion Policy (LDP), a two-stage framework performing flow matching in a deliberately shaped latent space. By absorbing scene understanding into an observation-conditioned CVAE encoder, LDP concentrates the conditional distribution of each observation. Consequently, the flow model avoids implicitly resolving scene-dependent structures; instead, it generates within a pre-concentrated distribution featuring a smoother velocity field, simplifying learning from limited demonstrations. Furthermore, to capture temporal dependencies among latent tokens, LDP trains with per-token diffusion forcing and employs staircase inference sampling to resolve the resulting distributional mismatch. We also propose reconstruction FID (rFID) as a lightweight proxy predicting downstream task success solely from latent space statistics. On coordination-intensive tasks from RoboTwin 2.0, LDP outperforms DP3 by a substantial margin and transfers effectively to real-world bimanual deployments. 

---
# FiberTune: Preserving Action-Fiber Visual Residuals in Vision-Language-Action Fine-Tuning 

**Authors**: Haihao Lin, Xiangsheng Huang, Xiao Yang, Weibang Zhou, Yiqi Zhang, Bo Yang, Simin Zeng, Jiawei Yang, Zhengyang Wang, Jiahui Du  

**Link**: [PDF](https://arxiv.org/pdf/2606.08653)  

**Abstract**: Action-supervised fine-tuning of vision-language-action (VLA) policies fits demonstrations effectively but constrains only the directions that change predicted actions, leaving visual structure consistent across action-equivalent states free to collapse. We formalize this as residual visual collapse along local action fibers and propose FiberTune, a training-time objective that preserves teacher-structured visual residuals without adding inference-time overhead. FiberTune uses an online action probe to estimate action-predictive feature directions, filters them from intermediate visual-token representations, and aligns the resulting probe-filtered residuals to a frozen visual teacher while regularizing their effective rank. Under identical training conditions, FiberTune improves over task-loss-only fine-tuning in every one of six controlled simulation settings spanning two benchmarks and two architectures (pi_0.5 and OpenVLA-OFT), as well as on physical SO-101 pick-place; representative gains include +10.7 percentage points SR(5) on long-horizon CALVIN ABC-to-D and physical SO-101 task success rising from 72.7% to 78.1%. Residual diagnostics show that these gains coincide with increased probe-filtered residual teacher alignment and effective rank, consistent with the action-fiber motivation. 

---
# Reconstructing Synthetic SDO/AIA 193 A EUV Images from He I 10830 A Observations with Diffusion Model Translator 

**Authors**: Marco Marena, Qin Li, Haimin Wang, Haodi Jiang, Prajwal Shah, Bo Shen  

**Link**: [PDF](https://arxiv.org/pdf/2606.08652)  

**Abstract**: Routine full-disk EUV imaging has been available only since the modern era, such as SOHO and SDO. To extend EUV coronal context into earlier periods, we leverage the multi-decade availability of full-disk \HeI{} observations, whose absorption is modulated by coronal irradiance and magnetic topology and is widely used as a proxy for open-field regions. We present a diffusion-based conditional image translation framework, Coronal Hole-aware Diffusion Model Translator (CH-aware DMT), to reconstruct synthetic SDO/AIA 193 Å EUV images from \HeI{} inputs. The model is trained on temporally co-aligned SOLIS \HeI{} and AIA 193 Å pairs spanning 2011--2015 using a month-based split, where January--October are used for training, November is used for validation, and December for testing. On the held-out test set, the reconstructions preserve dominant full-disk EUV morphology (CC=0.92) and recover CH-related low-intensity structure (CC=0.84). We further assess historical applicability by (1) comparing reconstructed AIA 193 Å morphology with SOHO/EIT 195 Å over 2005--2015; (2) comparing reconstructed AIA 193 Å images generated from KPVT \HeI{} inputs against Yohkoh/SXT soft X-ray observations; and (3) evaluating long-term reconstructed disk-integrated emission statistics against observational EUV series and independent solar activity proxies (sunspot number and F10.7 radio flux over 1974--2015). These results indicate that CH-aware DMT conditioned on \HeI{} can provide a physically plausible synthetic AIA 193 Å coronal proxy for historical studies, supporting multi-decade analyses of large-scale coronal evolution before the direct EUV imaging was available. 

---
# Sample-Efficient LLM-Based Detection of Malicious Web Server Logs with Forensically Explainable Reasoning 

**Authors**: Bernhard Kneip, Nhien-An Le-Khac, Hong-Hanh Nguyen-Le  

**Link**: [PDF](https://arxiv.org/pdf/2606.08649)  

**Abstract**: Forensic analysis of web server logs demands both accurate detection and human-readable explanations that can satisfy legal requirements. We present CEF-Log, a context-enhanced few-shot chain-of-thought prompting strategy for Large Language Models that addresses this dual requirement. CEF-Log embeds expert investigative methodology through a structured five-step reasoning template, enabling the model to learn \textit{how} to analyze logs rather than \textit{what} patterns to memorize. Experimental evaluation demonstrates that CEF-Log achieves an F1-score of 0.99 on the CSIC 2010 dataset using only four examples while providing a $10\times$ improvement in sample efficiency compared to other prompting-based methods. We also introduce ForenWebLog, a new dataset that incorporates real-world attacks and multi-step attack sequences for comprehensive evaluation. Qualitative analysis confirms that CEF-Log generates traceable, accurate explanations suitable for forensic documentation, addressing the critical "black-box" limitation of traditional machine learning approaches. 

---
# A retrieval conditioned rebinding circuit for dynamic entity tracking in large language models 

**Authors**: Soyoung Oh, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2606.08644)  

**Abstract**: To interpret context correctly and retrieve relevant information, large language models must bind entities to their attributes and update these bindings as state changes. We analyze how LLMs implement this binding process in a dynamic state tracking. Using causal interventions, we identify a retrieval conditioned rebinding mechanism, a compact attention head circuit that encodes swap relevant binding information and reinstates it at readout. Across Gemma and Llama models, this circuit supports rebinding behavior, but the representational signature of the mechanism differs across model families. In Gemma models, the binding signature is clearly expressed in the query/key subspaces of the relevant attention heads, whereas in Llama models, the binding information is carried primarily in key vectors. Overall, our results reveal an interpretable mechanism for context dependent state tracking in LLMs. 

---
# Tyan-WP: A Wind Power Foundation Model for Ultra-Short-Term Probabilistic Forecasting 

**Authors**: Jiahui Huang, Ao Luo, Lei Liu, Hongwei Zhao, Tengyuan Liu, Ruibo Guo, Bo Wang, Zhao Wang, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.08630)  

**Abstract**: Global wind power capacity, especially in China, is booming, with new farms spanning diverse terrains and climates. The industry urgently needs accurate wind power foundation models to shorten commissioning and accelerate grid connection. This is because site-specific time series models (TSMs) are not well suited to data-scarce scenarios and generalize poorly, while generic large time series models (LTSMs) are mostly limited to univariate inputs and cannot fully exploit static site attributes or the dependencies between power and meteorological covariates, leading to insufficient accuracy. To fill this gap, we propose \textbf{Tyan-WP}, the first wind power foundation model for ultra-short-term probabilistic forecasting. Pretrained on a large-scale wind power dataset covering more than 126,000 U.S. sites over seven years, Tyan-WP further improves zero-shot forecasting through two domain-specific module designs: static site embedding using coordinate, terrain, and ecoregion metadata, and a power-aware meteorological fusion (PAMF) module that models interactions between historical power and meteorological covariates. Under a unified evaluation protocol, Tyan-WP surpasses eight site-specific supervised TSMs on 10 in-domain sites and outperforms eleven generic LTSMs on 127 in-domain sites, reducing MAE by 19.9%, RMSE by 16.6%, CRPS by 22.2%, and AQL by 21.7%, while raising R^2 by 16.7%. It further demonstrates strong cross-geography generalization on six real U.K. sites. These results show that the wind power foundation model can achieve accurate zero-shot forecasting without target-site training, providing a practical pathway for rapid turbine onboarding and probabilistic risk management at new wind farms. 

---
# HARBOR: A Harness Framework for Agentic Robot Reinforcement Learning 

**Authors**: Zechu Li, Yufeng Jin, Xiaoyang Liu, Puze Liu, Vignesh Prasad, Carlo D'Eramo, Georgia Chalvatzaki  

**Link**: [PDF](https://arxiv.org/pdf/2606.08610)  

**Abstract**: Reinforcement learning (RL) has become a powerful paradigm for robot learning, particularly in sim-to-real settings, but its broader adoption remains limited by the engineering pipeline surrounding the algorithms. Building tasks, shaping rewards, and tuning hyperparameters require substantial expert effort, making RL workflows costly and difficult to scale. We introduce HARBOR, an agentic framework that frames robot RL automation as a harness-engineering problem: given a simulator codebase and a task specification, it automates the workflow from environment setup to policy training in simulation. HARBOR decomposes such high-level objectives into bounded stages executed by specialized agents through standardized commands, persistent artifacts, executable gates, and reusable knowledge, and scales iteration via decentralized parallel trials and experience learning across runs. We evaluate HARBOR across 6 benchmarks and 16 tasks in total, spanning manipulation, locomotion, and bimanual dexterous control. We demonstrate that HARBOR automates the simulation RL workflow end-to-end, designs rewards, tunes algorithms to match or improve over default configurations, and reduces engineering effort at practical token and wall-clock cost; the resulting policies can also be transferred to real robots. 

---
# Reinforcement Learning for Flow-Matching Policies with Density Transport 

**Authors**: Boshu Lei, Kostas Daniilidis, Antonio Loquercio  

**Link**: [PDF](https://arxiv.org/pdf/2606.08602)  

**Abstract**: We present an online reinforcement learning (RL) algorithm for fine-tuning flow-matching policies in continuous-control problems. Our key insight is to view RL-based policy improvement as a transport of action densities towards regions of high reward, which naturally aligns with the transport formulation of flow matching models. Prior methods either approximate the current or optimal policy distribution or resort to distillation, which introduces biased gradients or sacrifices multimodal modeling capacity. In contrast, our approach for RL with Density Transport, which we name \emph{RLDT}, constructs a transport field from a maximum-entropy RL objective using Stein Variational Gradient Descent (SVGD). Then, it finetunes a pretrained flow matching policy to align with this field. Training with this alignment objective is nontrivial because flow-matching policies generate actions via a multi-step process, making direct gradient-based optimization challenging. To overcome this challenge and stabilize training, we approximate policy actions from intermediate denoising steps via expected-target estimation. This allows the transport-field update to propagate into the network parameters without unstable backpropagation through time. Experimental results demonstrate that RLDT outperforms competitive baselines in reward quality and convergence speed. This performance holds across diverse continuous-control tasks, encompassing both dense and sparse rewards, as well as state- and vision-based long-horizon robot manipulation. The project webpage is \href{this https URL}{this https URL}. 

---
# Auditable Graph-Guided Root Cause Analysis for Kubernetes Incidents 

**Authors**: Anastasiia Kuvshinova, Seungmin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2606.08590)  

**Abstract**: Kubernetes incidents are diagnosed reliably only when a root-cause system's reported gains come from incident evidence rather than scenario-specific shortcuts. We present Graph Traversal Agent, a graph-guided RCA agent that combines LLM reasoning with specialized tools. The model reasons over a typed evidence graph, while deterministic graph and tool operations collect evidence, bound the search, and check proposed verdicts. We map operational constraints, including read-only evidence collection, propagation-aware diagnosis, bounded execution, and independently validated verdicts, to a typed incident graph, a LangGraph traversal state machine, and a separate validation stage. On ITBench snapshots scored by one fixed qwen-plus judge, the audited system raises root-cause-entity F1 over an earlier iteration of the same system from 0.6087 to 0.9130 on a 23-scenario common subset. A prompt-level ablation separates prompt-tuned gains from gains that survive once scenario-specific hints are removed: the stripped-prompt configuration retains 0.6958 F1 on a 19-scenario subset. The surviving gain concentrates on ChaosMesh scenarios whose ground-truth root cause is the injected fault object already present in the evidence graph, so we report it as benchmark-coupled rather than broad cross-cluster RCA evidence. Lightweight checks, including same-judge comparison, prompt-level ablation, cascade-source checking, and a telemetry no-leak test, mark claims as supported, pending, or out of scope. We scope the work to ITBench OpenTelemetry-demo snapshots. Live-cluster trials served as an engineering stress test, but alert state and trace availability did not stay stable enough for controlled scoring, so we make no production-readiness or mean-time-to-repair claim. 

---
# Calibration of Structured Ignorance Certificates for Diagnosing Unknown Unknowns in Reasoning Models 

**Authors**: Subramanyam Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2606.08571)  

**Abstract**: Large language models frequently fail in a characteristic way: rather than acknowledging ignorance, they produce fluent but incorrect answers to questions that lie beyond their knowledge boundaries. We introduce \textbf{Structured Ignorance Certificates} (SICs), a JSON-formatted output schema that demands a model explicitly name the missing domain intersection, enumerate required concepts, and propose a productive retrieval query rather than hallucinating an answer. To train models to produce high-quality SICs we construct a 7,347-sample \emph{Unknown-Unknown} (UU) dataset by prompting Qwen3-14B to stitch together questions from seven domains (physics, biology, engineering, CS, economics, medical, legal) into novel cross-domain queries that no single-domain expert could answer. We fine-tune a 14B-parameter model with Group Relative Policy Optimization (GRPO) using a composite reward that combines retrieval utility, concept specificity, and output-format validity. A paraphrase-divergence probe trained on model responses confirms that SIC-tuned outputs systematically exhibit higher unknown-unknown probability scores. Evaluation on 735 held-out UU questions achieves a 99.46\% JSON validity rate, a mean Certificate Specificity Score of 0.967, and a 3.6\% ROUGE-L improvement over the base model on retrieval-grounded generation -- demonstrating that explicit epistemic structuring is a learnable and measurable capability. 

---
# EinSort: Sorting is All We Need for Tensorizing LLM 

**Authors**: Toshiaki Koike-Akino, Jing Liu, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08565)  

**Abstract**: Tensor networks provide efficient representations for compressing large neural networks. By carefully designing shapes and topologies, they can significantly reduce memory and computational costs. However, identifying implicit low-rank structures in large foundation models remains challenging due to their enormous scale and un-structured weight distributions. We propose an adaptive tensorization method that discovers inherent low-rank structure in a target tensor by index ordering. Experiments on weight and KV-cache compression demonstrate improved reconstruction quality compared to baselines. 

---
# When Video Misreads: Closed-Loop Distillation of Reading Heuristics for Exploratory Manipulation Trace QA 

**Authors**: Haizhou Ge, Yufei Jia, Yue Li, Zhixing Chen, Lu Shi, Lei Han, Guyue Zhou, Ruqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08542)  

**Abstract**: Exploratory manipulation often turns an apparent failed attempt into the key evidence for what to do next. For example, a robot pulls a locked cabinet drawer, fails, and only succeeds after opening the lock. The failed pull reveals a latent precondition (the drawer is locked) that determines the minimal-success action chain (the fewest actions that complete the task), here [lock-open, drawer-pull]. Correctly reading this trace is therefore the prerequisite for recovering that chain. We formalize this setting as Exploratory Manipulation Trace QA (EMT-QA): given synchronized video and proprioception from an exploratory trace, predict the minimal-success action chain under the latent precondition revealed by the probe. However, even state-of-the-art VLMs and embodied multimodal LLMs misread this evidence: they do not reliably recover the chain from raw video, raw proprioception, or their combination.
We introduce Closed-Loop Trace Distillation, a pipeline that uses a per-task coding agent to inspect labeled training traces and distill a one-line natural-language prompt over the trace, which we call the Distilled Reading Heuristic (DRH). At inference, no agent is invoked and no model weights are updated; a frozen VLM receives the raw trace plus the DRH as a prompt entry. Across three simulator and two real-robot tasks, the DRH improves chain accuracy by +0.38 to +0.47 over the best raw-modality baseline. The same DRH also serves as the sole specification for one-shot programmatic classifiers that match the prompted VLM. 

---
# GEAR-VLA: Learning Geometry-Aware Action Representations for Generalizable Robotic Manipulation 

**Authors**: Yuan Zhang, Shiqi Zhang, Yedong Shen, Shuai Dong, Jiajun Deng, Xin Zhang, Yuxuan Gao, Jiajia Wu, Xin Nie, Zhiyuan Cheng, Jianmin Ji, Yanyong Zhang, Xingyi Zhang, Jia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08530)  

**Abstract**: Vision-Language-Action (VLA) models achieve strong benchmark performance but still struggle in real-world deployment with unseen objects, background shifts, and different robot embodiments. We argue that this stems from the lack of a unified geometry-aware manipulation representation, leaving existing VLAs vulnerable to low-level trajectory supervision, misaligned 3D features, and embodiment differences. To address this, we propose GEAR-VLA, a VLA framework for learning unified geometry-aware action representations for generalizable robotic manipulation. GEAR-VLA adopts coarse-to-fine action learning, where multi-source embodied pretraining equips the VLM with embodied reasoning and discrete action understanding before latent action tokens connect action semantics to a gradient-decoupled DiT continuous action expert. It further performs semantic-aligned 3D integration by aligning a trainable 3D spatial backbone with the VLA representation while freezing the original VLM-aligned visual pathway. To share this representation across robots, GEAR-VLA uses embodiment canonicalization, where embodiment-aware states and embodiment-invariant actions confine robot differences to the low-level interface. Extensive simulation and real-world experiments demonstrate strong generalization: GEAR-VLA achieves state-of-the-art performance on LIBERO, zero-shot LIBERO-Plus, and RoboTwin 2.0, reaches 85.9% success on AgileX and 81.0% on the pretraining-unseen LDT-01 embodiment, and obtains 90.1% success on a 6,360-trial universal grasping benchmark with 212 unseen objects. Code and models will be released at this https URL. 

---
# ActProbe: Action-Space Probe for Early Failure Detection of Generative Robot Policies 

**Authors**: Bingjia Huang, Xiangyu Li, Xiang Wang, Liang Mi, Zixu Hao, Weijun Wang, Hao Wu, Kun Li, Yunxin Liu, Ting Cao  

**Link**: [PDF](https://arxiv.org/pdf/2606.08508)  

**Abstract**: Generative robot policies fail unpredictably at deployment: they hesitate at critical moments, drift off-task, or commit to unrecoverable actions. Existing online failure detectors either require white-box access to policy internals or add runtime overhead through resampling and observation-side signals. Our empirical analysis shows that emitted action chunks themselves already carry strong predictive signal for impending failures in generative robot policies. Motivated by this observation, we introduce ActProbe, a lightweight, pure action-space detector that uses two compact signals available from a single forward pass: Temporal Consistency Error (TCE) between consecutive action chunks and Action Chunk Magnitude (ACM) of the current chunk. ActProbe maps these signals to per-step failure probabilities with a task-conditioned LSTM-MLP architecture. Across a diverse suite of generative robot policies and benchmarks, ActProbe raises alerts before failures become visually recognizable, improving the accuracy (F1)-timeliness Pareto frontier of failure detection by an average hypervolume gain of +12.7% over both internal- and external-feature baselines, with a +9.0% early-detection ROC-AUC lead on unseen tasks. ActProbe further transfers to deployment, predicting failures on unseen real-robot pick tasks and accelerating RL fine-tuning (PPO) with 2.9x fewer environment interactions. 

---
# Projecting the Emerging Mindset of SWE Agent by Launching a Wild Code Understanding Journey 

**Authors**: Zhengyi Zhuo, Yan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08500)  

**Abstract**: Software engineering agents (SWE agents) increasingly work through tool-mediated trajectories in real repositories, yet their behavior remains difficult to characterize in concrete, observable terms. These trajectories record tool use, intermediate reasoning, evidence selection, and self-directed stopping, but they do not by themselves explain why particular moves were chosen, what evidence was trusted, or when understanding was judged sufficient. This tension makes trajectory data both limited and valuable: faithful, replayable traces can become an empirical substrate for studying agent behavior when interpreted through disciplined observation. We introduce Ada, a scoped apparatus for repository-level code understanding. Ada enters real codebases through a bounded tool interface, allowing open-ended exploration to remain recordable as finite trajectories. Across this wild-but-bounded setting, Ada chooses where to look, what to read closely, when to consolidate partial understanding, and when to close its account of the repository. We project Ada's think-action chains through observation lenses that make navigation, evidence selection, synthesis, grounding, and stopping visible without reducing behavior to raw tool counts or speculating about hidden intent. Read together, these lenses produce behavioral profiles grounded in recorded movement through software worlds. Across 408 trajectories, spanning multiple models, repositories, task families, and launch conditions, the study shows how faithful digital traces can be transformed into disciplined, comparable projections of emerging SWE-agent mindset. The results expose differences in efficiency, trajectory diversity, epistemic grounding, and the limits of intervention, while providing a methodological foundation for observing SWE agent behavior in real codebases. 

---
# Seeing is Believing: Aligning Prompt Rewriting with Visual Anchors for Text-to-Image Generation 

**Authors**: Xuanyi Liu, Deyi Ji, Junyu Lu, Jing Wang, Qianxiong Xu, Xuhang Chen, Tianrun Chen, Siwei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.08492)  

**Abstract**: Despite the impressive capabilities of text-to-image (T2I) models, an intent-generation gap often persists due to the brevity and ambiguity of user prompts. Existing approaches primarily polish the prompt for fluency and readability. However, the enhancement process still lacks visual grounding. As a result, the rewriter may over-infer missing details, causing an intent-generation gap. To address this limitation, we propose FaithRewriter, a novel prompt-enhancement framework for T2I generation. Specifically, FaithRewriter first leverages a multimodal MLLM to generate an image from the original prompt as an intermediate visual cue. This cue is then combined with the prompt and fed into a large-scale LLM to produce visually grounded augmentations that better reflect how the intended content should appear in images. Finally, these augmentations are distilled into a small-scale LLM for efficient deployment, enhancing its ability to generate effective T2I prompts. Experiments show that FaithRewriter yields prompts that are more faithful to the user intent and more visually plausible than strong baselines, helping narrow the intent-generation gap. 

---
# STELLAR: Spatio-Temporal Environmental Learning with Latent Alignment and Refinement for Long-Tailed Species Distribution Modeling 

**Authors**: Shufeng Kong, Tao Yu, Yuanyuan Wei, Caihua Liu, Junwen Bai, Yingheng Wang, Marc Grimson, Daniel Fink, Carla P. Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2606.08484)  

**Abstract**: Joint Species Distribution Modeling (JSDM) is a key enabler for biodiversity monitoring and conservation planning. However, accurate JSDM faces two coupled challenges: environmental drivers and species distributions are inherently spatio-temporal, while species co-occurrence patterns exhibit complex non-linear community structure and severe long-tail imbalance driven by rare species. Existing approaches often address these factors in isolation, learning from static covariates or neglecting the historical trajectories of dynamic community structure. To overcome these limitations, we propose STELLAR (Spatio-Temporal Environmental Learning with Latent Alignment and Refinement), a novel framework that learns a shared latent space where dynamic habitat context and community structure are optimized jointly. Our approach integrates three complementary components: (1) a Graph-Temporal Encoder that employs graph attention and recurrent units to aggregate spatial neighborhood effects and capture the co-evolving historical dynamics of environmental context and community structure; (2) a Context-Anchored Latent Alignment mechanism that structures the latent space using a label-activated mixture prior and supervised contrastive learning, actively clustering species based on shared environmental preferences; and (3) an Imbalance-Aware Decoupled Decoding module that utilizes Asymmetric Loss to focus learning on hard, rare species samples, preventing mode collapse in the long tail. Experiments on the large-scale eBird dataset, curated with domain experts, demonstrate that our framework significantly outperforms state-of-the-art baselines, particularly in predicting rare species and revealing interpretable species interactions. 

---
# PIPE-Cypher: Automatic Enterprise Benchmark Generation for Text-to-Cypher Systems 

**Authors**: Suraj Ranganath, Anish Raghavendra  

**Link**: [PDF](https://arxiv.org/pdf/2606.08481)  

**Abstract**: Enterprise property graphs vary widely in schema structure, internal terminology, domain assumptions, governance constraints, and user interaction patterns. A deployment-relevant Text2Cypher benchmark therefore reflects the questions users and agents actually ask of that graph. Creating such a benchmark is difficult because schemas and values are unique, and graph structure changes over time. Each NL-query pair must also be executable, use real graph entities, preserve diversity, and remain balanced across query types and difficulty levels. We present PIPE-Cypher, a local benchmark-generation pipeline that turns a live property graph and optional seed queries from customer questions, analyst logs, or agent tool calls into balanced NL-to-Cypher benchmarks. PIPE-Cypher combines schema profiling, reverse-query grounding, constrained generation, deterministic Cypher governance, execution validation, redaction, diversity controls, and a calibrated local LLM judge. Using local Qwen3.5-9B generation and judging, PIPE-Cypher exports 3,000 accepted FinBench/SNB examples, completes three audited ablation suites, calibrates judge behavior with human labels, and evaluates 11 local downstream models. The resulting benchmark is deliberately discriminative: zero-shot transfer is weak, while a few-shot control shows that schema-specific example banks can help compatible model families. Together, PIPE-Cypher makes Text2Cypher benchmarking a repeatable process that evolves with the graph, its users, and its target workloads. 

---
# Adaptive Loss Balancing for Noise-Robust GRPO in Generative Recommendation 

**Authors**: Kewei Xu, Junbo Qi, Yanyan Zou, Pengfei Zhang, Xingzhi Yao, Shengjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.08480)  

**Abstract**: Reinforcement learning (RL) presents a promising avenue for enhancing generative recommendation beyond supervised imitation, leveraging reward signals to guide policy improvement. However, its efficacy is critically contingent on the trustworthiness of the reward model for the samples it evaluates. In practice, production rankers, the widely adopted reward models, are trained on exposure-biased logs, leading to sample-dependent inaccuracies that violate this assumption. Our stratified analysis uncovers a consistent pattern: reward guidance is most beneficial when the policy exhibits uncertainty and the ranker can effectively discriminate the ground-truth item from rollout negatives. On other samples, the reward signal is either negligible or detrimental, highlighting the risk of uniform RL application. To address such an issue, we introduce AdaGRPO, a novel framework that treats reward-guided optimization as selective admission rather than uniform pressure. Training is anchored in supervised negative log-likelihood, while the GRPO objective is gated by a binary, per-sample clip determined by two rollout diagnostics: policy-side difficulty and reward discriminability. Instances failing either diagnostic default to pure supervision, ensuring stability and mitigating the amplification of noisy gradients. We validate AdaGRPO on a large-scale e-commerce dataset. At the best intermediate checkpoint, it elevates HR@10 from 11.01% to 12.18% while constraining hallucination below 0.22%, and maintains robustness at the final checkpoint (HR@10 11.63%, hallucination 0.27%), outperforming fixed NLL--GRPO mixtures across the retrieval--validity frontier. In production A/B tests, AdaGRPO achieves statistically significant gains in click-through rate and dwell time, confirming its practical utility. 

---
# FlashCP: Load-Balanced Communication-Efficient Context Parallelism for LLM Training 

**Authors**: Zheng Wang, Eric Liu, Linan Jiang, Zhongkai Yu, Zaifeng Pan, Yue Guan, Yuke Wang, Yufei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2606.08476)  

**Abstract**: Context parallelism (CP) is essential for training large-scale, long-context language models, as it partitions sequences to reduce memory overhead. However, existing CP methods suffer from workload imbalance, inefficient kernels, and redundant communication due to static sequence sharding and key-value (KV) tensor communication. We present FlashCP, a load-balanced and communication-efficient framework for CP training. FlashCP introduces a sharding-aware communication mechanism to eliminate redundant KV communication and proposes a novel Whole-Doc sharding strategy that maximizes communication savings while maintaining balanced workloads. To efficiently combine Whole-Doc and Per-Doc sharding, FlashCP further designs a heuristic algorithm to search for near-optimal sharding plans. Extensive experiments show that FlashCP achieves up to 1.63x speedup over state-of-the-art CP frameworks across diverse datasets. 

---
# More Yap Less Meaning: Uncovering Self-Improvement Behavior in SLMs 

**Authors**: Marina Igitkhanian, Erik Arakelyan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08471)  

**Abstract**: Recently, language models have made rapid progress across various domains and applications. However, their capability for self-improvement, i.e., whether they are adept at recognising and correcting flaws in their own reasoning, remains dubious. In this study, we address this question by constructing a sufficiency test to rigorously examine the self-correction capabilities of small language models (SLMs). We propose a minimal three-step self-correction pipeline that collects initial SLM answers, prompts the same model to generate hints for its incorrect responses given the ground truth, and feeds the model the same question with its own feedback to refine the initial answer. We evaluate a variety of instruction-tuned and reasoning SLMs in this experimental setup on arithmetic and logical reasoning benchmarks. Our findings show that SLMs with injected hint sentences yield only a 4.4 percent gain over initial question-answering accuracy. Even though the correct answer was provided alongside the model's incorrect reasoning, the evaluated SLMs fail to understand what was missing in their reasoning and show minimal semantic difference between hints that lead to corrections and ones that do not. Furthermore, our experiments show that longer hints are positively correlated with incorrect final answers, suggesting that longer deliberation on problems can hinder the reasoning process, meaning that SLMs do not necessarily scale in performance with a larger compute budget. 

---
# The Confidence Trap: Calibration Attacks for Graph Neural Networks 

**Authors**: Cuong Dang, Jiahao Zhang, Hieu Ta Quang, Dung Le, Lu Cheng, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08467)  

**Abstract**: While confidence calibration is essential for trustworthy decision-making in safety-critical applications, the robustness of calibrated GNNs to adversarial structural perturbations remains largely unexplored. However, studying calibration attacks on graphs presents unique technical challenges: (1) the discrete nature of graph structures complicates gradient-based optimization, (2) existing underconfidence objectives fail to drive predictions toward uniform distributions, and (3) GNNs are highly sensitive to edge perturbations, often causing unintended label changes that violate attack constraints. To address these challenges, we propose a \textbf{Unified Graph Calibration Attack (UGCA)} framework designed for \textbf{worst-case (white-box) analysis} of GNN calibration robustness. UGCA introduces a KL-divergence loss to encourage uniform predictive distributions, a reranking mechanism to reduce label flipping, a hybrid loss to recover labels when violations occur, and beam search to explore a broader adversarial search space. We further provide theoretical insights linking model generalization, dataset complexity, and calibration vulnerability, showing that models with higher accuracy or trained on datasets with more classes are more susceptible under this threat model. Extensive experiments demonstrate that UGCA substantially increases Expected Calibration Error while preserving classification accuracy. Our code is publicly available at this https URL. 

---
# Sycophancy as a Multilingual Alignment Failure: How Safety Degrades Across Languages, Topics, and Models 

**Authors**: Arya Shah, Himanshu Beniwal, Mayank Singh, Chaklam Silpasuwanchai  

**Link**: [PDF](https://arxiv.org/pdf/2606.08451)  

**Abstract**: Safety-aligned large language models often exhibit sycophancy, which is the tendency to affirm users' opinions regardless of factual accuracy. Although well-studied in English, its manifestation in other languages remains largely unexamined, leaving billions of non-English speakers potentially vulnerable to model-validated misinformation. We present the first large-scale, multi-model evaluation of cross-lingual sycophancy, benchmarking \textbf{six instruction-tuned models} across \textbf{1.1 million instances} spanning \textbf{38 languages} and \textbf{33 topic categories}. We identify a consistent resource-tier effect: sycophancy rates spike sharply in low-resource and zero-shot language settings. Critically, this degradation is topic-agnostic, as models fail uniformly across both benign and safety-critical prompts, offering no additional protection where it is most needed. We further identify tokenizer fertility as a structural driver of this alignment collapse. Collectively, our results demonstrate that prevailing alignment methodologies generalize poorly beyond high-resource languages, underscoring the urgent need for equitable multilingual safety techniques. 

---
# Not Just After One: Sleep-Inspired Replay Prevents Catastrophic Forgetting After Sequential Tasks 

**Authors**: Anthony Bazhenov, Jean Erik Delanois, Giri P. Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08447)  

**Abstract**: One of the critical limitations of artificial neural networks is their lack of ability to continually learn: training on new tasks often leads to interference and forgetting of the previous ones. While several algorithms have been proposed to protect old memories from interference, they are typically applied during or immediately after each new episode of training. In contrast, humans and animals can learn continuously, acquiring multiple new memories during active learning before consolidating all of them into long-term storage. Here we show that multiple new tasks can be trained sequentially before an unsupervised sleep-like replay phase is applied to partially restore performance across all previously learned tasks. Our study further suggests that task-specific information remains resilient to new training but decays gradually as network is trained on new tasks. These findings point to novel principles for developing a broad range of continual learning AI solutions. 

---
# Sparrow: Sparse Rollout for Stable and Efficient Long-context RL of Large Language Models 

**Authors**: Yang Zhou, Ranajoy Sadhukhan, Zhaofeng Sun, Zhuoming Chen, Souvik Kundu, Saket Dingliwal, Sai Muralidhar Jayanthi, Aram Galstyan, Haizhong Zheng, Beidi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.08446)  

**Abstract**: Despite being powerful, reinforcement learning with verifiable rewards (RLVR) induces extremely long COT, making it computationally expensive. Since RLVR per-step cost is dominated by long-context rollout generation, sparse attention offers a promising way to accelerate dense rollout. However, sparse rollouts require a delicate stability-efficiency tradeoff: overly aggressive sparsity causes collapse, while overly lenient sparsity gives insufficient speedup. In this work, we study this tradeoff through sparse-to-dense actor-policy mismatch. We first observe that sparse rollout collapse is not driven by uniform degradation across tokens: most sparse tokens align perfectly with dense even under aggressive sparsity. Motivated by this, we hypothesize that sparse rollout training remains stable if the lower tail of per-token actor-policy mismatch stays above a critical threshold throughout the trajectory. We introduce a dynamic sparsity schedule that keeps this tail statistic constant during generation and validate our hypothesis. Across Qwen3 thinking-family models, keeping the tail mismatch statistic near a consistent threshold generally enables stable training. We then use a cost model to find the sparsity schedule for maximum speedup under this mismatch threshold, achieving 2.2x, 2.4x, and 2.0x rollout speedups when training Qwen3-1.7B, Qwen3-4B, and Qwen3-8B. Empirically, we show the thresholds generalize to a larger model (Qwen3-14B) and another RL domain (coding). Finally, our analysis naturally motivates DistillSparse: lightweight LoRA-based distillation on sparse rollout lets more aggressive sparsity reach the same sparse-to-dense mismatch threshold, yielding higher speedup. 

---
# Segment-level Tree Search for Long Meeting Document Summarization 

**Authors**: Sangwon Ryu, Heejin Do, Jun Seo, Daehui Kim, Yunsu Kim, Gary Geunbae Lee, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2606.08445)  

**Abstract**: Meeting documents are challenging to summarize due to their length and complex conversational structure. Existing approaches typically adopt multi-stage pipelines that extract information prior to summarization; however, these approaches often suffer from cumulative error propagation without intermediate validation, a limitation further amplified by short and low-quality reference summaries. We propose segment-level summarization via Monte Carlo Tree Search (S3), a training-free framework that constructs a final summary by composing segment-level summary candidates. S3 partitions a long document into segments and generates multiple summary candidates per segment, forming nodes of a search tree. The best-scoring combination is selected via self-reward-guided tree search and refined into the final output. Despite using a 7B model, S3 achieves performance comparable to larger 72B models while producing length-appropriate summaries. 

---
# AI Code Sandboxes: A Comparative Security Study. Part 1 of 2 -- Engine-Level Properties (Attack Surface, Leakage, Stackability, CVE History, Patch Cadence, Fuzzing) 

**Authors**: George Andronchik, Pavel Lokhmakov  

**Link**: [PDF](https://arxiv.org/pdf/2606.08433)  

**Abstract**: This paper reads six engine-level measurements together -- 1.1 host attack surface, 1.2 information leakage, 1.3 defense-in-depth stackability, 1.4 public CVE history, 1.5 patch cadence, and 1.6 upstream fuzzing posture -- to describe how five AI-sandbox products isolate guest code from the host kernel. No single axis is a sufficient basis for a comparative judgement; the cross-axis reading is the load-bearing analysis.
Three high-level findings: (1) engine classes (microVM, userspace kernel, OCI container) separate cleanly on every architectural axis, but products within a class do not; (2) product pin policy is the dominant operator-facing variable -- engine-side patch latency aggregates to ~0 days for coordinated disclosures, while downstream lag spans 0 days to 471+ days to "opaque" to infinity; (3) fuzzing investment splits into three tiers, and the strongest combination -- microVM x continuous public fuzzer -- is unoccupied in this set, leaving the "0 published CVEs x no upstream fuzzer x no academic study" intersection structurally unmeasured.
We report per-axis orderings, per-product portraits, and a threat-model qualification matrix; no overall ranking is proposed. Companion repository (code, Apache-2.0): this https URL. License: CC BY 4.0. 

---
# Hacking Generative Perplexity: Why Unconditional Text Evaluation Needs Distributional Metrics 

**Authors**: Antonio Franca, Alexander Tong  

**Link**: [PDF](https://arxiv.org/pdf/2606.08417)  

**Abstract**: Diffusion and continuous flow-based language models have emerged as the leading non-autoregressive alternatives to language modeling. Progress in both paradigms is overwhelmingly tracked by generative perplexity (gen-PPL): the per-token negative log-likelihood of samples under a frozen autoregressive (AR) scorer such as gpt2-large, typically paired with an empirical-entropy guardrail to rule out low-entropy collapse. We argue that this metric is unsound. By construction, gen-PPL measures only predictability under the scoring AR, not grammaticality or semantic coherence -- and the set of predictable but still low-quality sequences is combinatorially large. To make this concrete, we construct a suite of zero-parameter, deliberately naive samplers that achieve state-of-the-art gen-PPL on LM1B and OpenWebText at non-degenerate entropy, surpassing recently published diffusion and continuous-flow models while producing text that is incoherent by construction. We recommend evaluation suites that directly quantify the distributional divergence between generated and reference text, and use such a suite to re-benchmark recent non-autoregressive models, recovering a more faithful picture of the current state of the art. 

---
# CoVEBench: Can Video Editing Models Handle Complex Instructions? 

**Authors**: Jiangtao Wu, Jiaming Wang, Yiwen He, Yuanxing Zhang, Shihao Li, Dunyuan Liu, Xuedong Zhao, Jialu Chen, Zekun Moore Wang, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08415)  

**Abstract**: While recent text-guided video editing models excel at elementary tasks (e.g., style transfer, object insertion), real-world user requests are highly compositional. A single prompt often demands multiple coupled edits, such as modifying subjects, actions, and camera views, while strictly preserving unrelated spatiotemporal content. Existing benchmarks, heavily constrained by isolated edits and coarse global metrics, fail to diagnose how models handle such complex workflows. To address this gap, we introduce CoVEBench, a compositional video editing benchmark comprising 416 curated source videos, 626 multi-point editing instructions, and 9,990 fine-grained checklist items. Covering diverse editing dimensions, CoVEBench evaluates models via MLLM-judged instruction compliance and video fidelity, alongside automated metrics for video quality. Extensive experiments reveal that compositional editing remains a profound challenge: current models frequently omit edits, violate preservation constraints, or introduce artifacts when handling multiple operations simultaneously. CoVEBench provides a challenging, diagnostic testbed to advance video editing toward realistic user workflows. 

---
# PACT: Self-Evolving Physical Safety Alignment for Diffusion Policies in Embodied Manipulation 

**Authors**: Lingxuan Wu, Zijian Zhu, Lizhong Wang, Chengyang Ying, Huayu Chen, Xiao Yang, Fangming Liu, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08414)  

**Abstract**: Diffusion policies have achieved remarkable success in robotic manipulation, yet they often fail to satisfy strict physical constraints required for safe deployment. Existing approaches impose safety either prematurely during training or reactively via external guardrails at test time, limiting policy expressivity and overall scalability. We propose Physical safety Alignment for Constrained Trajectories (PACT), a self-evolving post-training framework that projects pretrained diffusion policies onto constraint-feasible regions without accessing demonstration data or task rewards. PACT distills constraint gradients into the diffusion model through a reverse-KL objective with dense supervision across timesteps. It incorporates a curriculum that progressively tightens constraints while maintaining theoretically bounded policy shift and monotone improvement, mitigating the safety-performance trade-off from catastrophic forgetting. On simulated and real-world embodied manipulation benchmarks, PACT significantly reduces safety violations by 31.0% on average while improving task success by 30.7%. 

---
# Provably Efficient Personalized Multi-Objective Bandits with Proactive Conversational Queries 

**Authors**: Linfeng Cao, Ming Shi, Ness B. Shroff  

**Link**: [PDF](https://arxiv.org/pdf/2606.08410)  

**Abstract**: Personalized decision-making in multi-objective bandits requires learning user-specific trade-offs among competing objectives. Since arm utility depends on both unknown rewards and unknown preferences, existing methods infer preferences only from utility feedback, entangling preference learning with reward exploration. In practice, however, users often reveal their priorities through proactive conversational queries (e.g., "cheap and clean hotel"), yet this structured signal is not leveraged. We formalize a proactive query-based framework in which user queries provide structured preference signals. Modeling these signals via a Plackett-Luce subset choice model, we show that query-only learning is insufficient due to a fundamental shift-invariance barrier. To resolve this, we introduce MO-PQUCB, a hybrid algorithm that integrates query-based preference anchoring with bandit feedback through shift-invariant regularization and dual-exploration UCB. We prove that proactive queries accelerate preference estimation and yield improved regret scaling over prior preference-aware MO-MAB methods. Under corrupted queries, we further characterize statistical limits and design a robust estimator achieving near-optimal performance when the corruption is sparse. Experiments validate both theoretical and practical gains. 

---
# TimpaTeks: Automatic In-place Text Sequence Modification via Diffusion Language Model Steering 

**Authors**: Ryandito Diandaru, Ikhlasul Akmal Hanif, Fadli Aulawi Al Ghiffari, Ahmed Elshabrawy, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2606.08408)  

**Abstract**: We extend activation steering to diffusion language models (DLMs) and study a novel problem that arose due to the inference mechanism of DLMs: Modifying a text in-place to manifest a different concept. We propose TimpaTeks, an automatic in-place text modification mechanism using DLMs. Experiments on IMDB movie reviews (sentiment) and a synthetic Cats and Dogs Dataset (arbitrary, more unconventional concept steering) show that TimpaTeks provides a feasible novel mechanism to steer diffusion language model outputs in-place. TimpaTeks enables in-place modification while simultaneously lowers sentence perplexity and retaining the original sentence structre without the need of instruction tuned models. TimpaTeks is also computationally cheaper than prompt-based DLM steering, as it performs denoising in-place rather than constructing an additional prompt-conditioned output sequence. 

---
# Hiding in Plain Floats: Steganographic Carriers for Indirect Prompt and Content Injection 

**Authors**: Mudit Sinha, Sanika Chavan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08403)  

**Abstract**: Text-centered prompt-injection defenses assume that the malicious signal is visible in one of the inspected text views. We study a reproducible LLM01-style indirect prompt/content-injection failure mode where that assumption breaks: a payload caught in plain English slips past the same detector when it is transported as structured float parameters and reconstructed only as fragmented telemetry. Across 14,400 attacked real-model trials on three commercial LLM APIs from different providers, the IFS-derived float-array carrier preserves 94.3% leakage ASR under the strongest dual-layer text-classifier defense evaluated in the main matrix: a Prompt Guard 2 + TF-IDF ensemble; the same carrier-level pattern also replicates with a fine-tuned roberta-base detector. We emphasize leakage ASR because downstream systems may act on quoted or reproduced markers even when the model refuses, but Strong ASR is the stricter metric for structurally compliant attack success. A 2 x 2 ablation shows that data-layer storage and reconstruction-layer fragmentation defeat different text views and that both are needed to evade both. A simple xxd detector and semantic validation block the current T3 instance, so the contribution is not an undetectable exploit but a measured failure boundary for text-only inspection in structured-input pipelines that expose reconstructed auxiliary channels to an LLM. 

---
# SceneConductor: 3D Scene Generation from Single Image with Multi-Agent Orchestration 

**Authors**: Jeonghwan Kim, Yushi Lan, Yongwei Chen, Hieu Trung Nguyen, Chuanyu Pan, Xingang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08402)  

**Abstract**: Generating complete 3D scenes from a single image requires inferring globally consistent geometry, object relationships, and environmental context from inherently ambiguous visual evidence. Despite recent progress in joint layout-and-mesh generation, existing methods often rely on holistic or weakly decomposed pipelines that entangle many factors at once and demand extensive scene-level supervision, limiting their generalization to complex real-world environments. We propose a multi-agent orchestration framework that decomposes single-image 3D scene generation into three structured stages: scene initialization, environment construction, and multi-agent refinement. The initialization stage extracts image-derived object masks, builds object-level 3D representations, and predicts an initial spatial layout to form a coarse 3D scene. The environment-construction stage then leverages this initialization together with point-map geometry to build an environmental scaffold of supporting surfaces, room boundaries, materials, and illumination. Finally, in the refinement stage, a planner agent identifies structural and visual inconsistencies, applies simple corrections directly, and dispatches specialist agents for complex localized revisions that are reintegrated into the global scene. To provide reliable structural initialization while reducing reliance on scene-level annotations, we further introduce a geometry-aware layout predictor supervised by sparse geometric priors derived from point maps. Unlike fully supervised layout generators, the predictor can be trained from segmentation-level data and generalizes robustly to diverse real-world scenes. Extensive experiments on benchmark datasets show that our method consistently outperforms prior approaches in geometric accuracy, spatial consistency, and perceptual realism. 

---
# Impacts of Histories and Models on LLM Grading: A Study in Advanced Software Engineering Courses 

**Authors**: Qilin Zhou, Zhuo Wang, Yue Li, W.K. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08400)  

**Abstract**: Graduate-level research reading report assessment creates a substantial labor burden for educators. While large language models (LLMs) hold great potential for automating academic grading, their reliability for this specialized task remains understudied, particularly regarding grading consistency, the lack of which represents a primary obstacle to educational fairness. This paper proposes a human-aligned LLM-assisted grading workflow and presents a case study based on 180 student submissions from a graduate advanced software engineering course. We evaluate two mainstream LLMs, Grok and GPT, in terms of grading consistency and alignment with human scores. We find LLMs exhibit distinct levels of intra-model consistency and significant inter-model grading inconsistencies, while simple ensemble approaches cannot improve alignment with human evaluation. Critically, continuous interaction history drives systematic drift in models' grading standards away from human expert scores. Our findings demonstrate LLMs' potential in reducing grading workload for educators in graduate education, while highlighting that indiscriminate LLM grading may introduce systemic unfairness, suggesting that specific operational practices are required to mitigate such disparities. 

---
# STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control 

**Authors**: Priyansh Bhatnagar, Ashkan Moradifirouzabadi, Se-Hyun Yang, SeungJae Lee, Jungwook Choi, Mingu Kang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08382)  

**Abstract**: Low-rank projection has emerged as a promising approach for compressing the KV cache by exploiting hidden-dimension redundancy. However, prior methods rely on fixed or heuristic rank selection and struggle to achieve aggressive compression with minimal accuracy degradation. We propose STAR-KV, an adaptive low-rank KV cache compression framework with fine-grained rank control. STAR-KV encompasses 1) a differentiable thresholding mechanism that enables optimal rank selection at both attention-head and block levels, 2) a hybrid decomposition strategy that applies different low-rank factorizations according to the sensitivity of key and value projections, and 3) a low-rank-aware mixed precision quantization that leverages data statistics for near lossless low-bit quantization. Evaluated across multiple LLMs and benchmarks, STAR-KV achieves up to 75% KV cache compression and up to 20x overall KV cache reduction when combined with quantization. Enabled by custom Triton-based GPU kernels, STAR-KV delivers up to 6.9x speedup for the attention module and 3.1x end-to-end generation throughput. Our code is publicly available at: this https URL. 

---
# Auditing Proprietary Alignment in Large Language Models: A Comparative Framework Without a Ground-Truth Standard 

**Authors**: Alireza Arbabi, Florian Kerschbaum  

**Link**: [PDF](https://arxiv.org/pdf/2606.08381)  

**Abstract**: Large language models (LLMs) are increasingly released and deployed through opaque development and deployment pipelines, enabling model providers to inject intentional, provider-specific policies without officially announcing them. As a result, various models have been reported to generate responses reflecting proprietary rules and organizational interests, leading to censorship or misinformation on controversial topics. However, systematic identification of such alignment remains a fundamental challenge, complicated by the ambiguity of what ``proprietary'' entails in different contexts. In this paper, we propose a statistical framework for detecting proprietary alignment in black-box language models via comparative behavioral analysis. Our approach quantifies systematic deviations between the responses of a target model and those of a reference set of baseline models in a shared semantic space. By evaluating relative behavioral divergence rather than absolute correctness, our framework enables principled auditing under black-box access. Applied to several widely discussed but previously unquantified cases, it provides a systematic and scalable basis for external assessment of provider-specific alignment behavior in large language models. 

---
# RiskNet: A large-scale dataset of AI risk incidents from news with alignment and multi-dimensional annotations 

**Authors**: Leihan Zhang, Wecheng Ye, Xianlong Ma, Haochuan Liu, Yang Li, Qianyu Zhang, Jinliang Chen, Qiang Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08376)  

**Abstract**: As artificial intelligence (AI) systems are increasingly deployed across socially consequential domains, reports of AI-related harms and failures have grown in frequency and diversity. Although existing governance frameworks articulate high-level principles for responsible AI, large-scale empirical resources for tracking and analyzing real-world AI risk incidents remain limited. Existing incident collections are often manually curated, relatively small in scale, and insufficient for continuous, data-driven monitoring and downstream computational analysis. To address this need, we present RiskNet, a large-scale dataset of AI risk incidents constructed from large-scale multilingual news sources. RiskNet applies a structured pipeline for AI risk news identification, event-level report screening, incident alignment, and multi-dimensional incident classification. The resulting resource organizes dispersed news reports into incident-centered records and provides benchmark datasets for event classification, incident alignment, and incident-level risk labeling. In its current release, RiskNet covers hundreds of millions of source records and yields a large-scale collection of AI risk-related reports, including aligned incident clusters and annotated benchmark subsets. The dataset is also accessible through an online platform for browsing and exploration. We describe the data sources, processing workflow, taxonomy design, and technical validation of the resource. RiskNet is intended to support downstream research on AI safety, governance, risk analysis, and benchmarking, as well as longitudinal and cross-source analyses of AI-related harms. By providing a structured and reusable empirical resource, RiskNet helps bridge the gap between high-level governance principles and the documented realities of AI risk incidents. 

---
# An Information-Theoretic Definition for Open-Ended Learning 

**Authors**: Wanqiao Xu, Yifan Zhu, Benjamin Van Roy  

**Link**: [PDF](https://arxiv.org/pdf/2606.08369)  

**Abstract**: A growing body of work points to the great promise of AI systems that can continually expand their capabilities as they operate in an open-ended environment. But yet there is no coherent definition of open-endedness or theory about how an agent ought to explore an open-ended environment. We introduce an information-theoretic definition based on a new concept -- the ${\textit bit-equivalent}$ -- which quantifies the information required to attain each level of expected reward. We consider an environment to be open-ended if an agent can attain linear growth in the bit-equivalent. We establish that classical bandit environments are not open-ended and formulate a bandit environment that is. We also introduce an algorithm that achieves open-ended learning in this environment. 

---
# Emergence World: A Platform for Evaluating Long-Horizon Multi-Agent Autonomy 

**Authors**: Deepak Akkil, Ravi Kokku, Karthik Vikram, Tamer Abuelsaad, Aditya Vempaty, Satya Nitta  

**Link**: [PDF](https://arxiv.org/pdf/2606.08367)  

**Abstract**: Most evaluations of LLM agents look like exams: a discrete task, a clean environment, a score in minutes or hours. We argue that this approach is mismatched with the deployment conditions of autonomous systems, where the relevant timescale can be weeks to months, and where the dynamics that matter most, such as behavioral drift, governance in diverse environmental contexts, and cross-influence between agents from different model families, only emerge over time. We introduce Emergence World, a continuously running multi-agent simulation platform designed to make those dynamics measurable. The platform hosts populations of LLM-driven agents in a shared spatial world grounded in live external data (e.g. real-time weather, news APIs, internet access), equips each agent with 120+ specialized tools and three persistent memory systems, and lets them govern themselves through democratic mechanisms with consequential outcomes. The platform is model-agnostic at the reasoning layer and supports heterogeneous populations in which agents from different vendors share the same world. To illustrate the kinds of questions the platform makes tractable, we present a 15-day cross-vendor study with five parallel worlds powered by Claude Sonnet 4.6, Grok 4.1 Fast, Gemini 3 Flash, GPT-5-mini, and a mixed population. Identical roles and starting conditions produced radically different outcomes, ranging from stable deliberative governance to total population collapse. We release the prompts, log data and configurations to support further research on long-horizon multi-agent autonomy. 

---
# Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects 

**Authors**: Evan Duan  

**Link**: [PDF](https://arxiv.org/pdf/2606.08365)  

**Abstract**: Sparse autoencoder (SAE) features are increasingly used to steer language models, but feature steering is rarely clean: the same intervention can behave inconsistently across contexts and perturb unrelated features. We introduce a pre-intervention screening framework for forecasting SAE steering side effects from feature statistics computed before steering. We operationalize side effects along two axes of steering modularity, effect stability and collateral spread, and evaluate GPT-2-small, Pythia-70M-deduped, Gemma-2-2B, and Llama-3.1-8B across ReLU, JumpReLU, and TopK SAE dictionaries. Across these settings, decoder geometry, activation statistics, co-activation structure, and direct-logit footprint predict steering modularity better than frequency-only and activation-magnitude baselines. The signal is strongest in GPT-2-small, Pythia-70M, and Llama-3.1-8B, where it survives residualization against magnitude-related confounds, and weaker in Gemma-2-2B. Held-out screening shows that ranking unseen features by predicted cleanliness can select features that steer more cleanly on fresh contexts, but the successful axis varies by setting: GPT-2 improves most cleanly, Pythia improves mainly on stability, Llama mainly on collateral, and Gemma only partially. A controlled Llama Scope width comparison shows that the predictive signal persists under a 32K-to-128K dictionary-width change, although the screening payoff becomes less stable. Overall, SAE steering side effects are predictable in advance, but the useful predictor signature and transferred modularity axis are model- and dictionary-setting dependent. 

---
# Self-Supervised Vision Transformers for CBCT-Based Detection of Temporomandibular Joint Osteoarthritis 

**Authors**: Shradhdha Trivedi, Vrundan Sojitra, Mariela Padilla  

**Link**: [PDF](https://arxiv.org/pdf/2606.08364)  

**Abstract**: Temporomandibular joint osteoarthritis (TMJ OA) is a prevalent degenerative condition whose osseous changes are often subtle on cone-beam CT (CBCT), making automated detection challenging. We study how well the DINO family of self-supervised vision transformers -- DINOv1, DINOv2, DINOv2+reg, and RAD-DINO (a radiology-pretrained variant) -- transfers to CBCT, asking how much backbone adaptation is needed and of what kind. We propose a simple slice-based pipeline using Vision Transformer (ViT) backbones: axial CBCT slices are encoded per-slice by a frozen or partially adapted ViT and aggregated via attention-based multiple instance learning (MIL) for patient-level binary OA/Normal classification. Through systematic ablation across unfreezing strategies and aggregation designs on a multi-source CBCT dataset, we find that partial unfreezing of the final two transformer blocks is the decisive factor, improving AUC from 0.671 (fully frozen DINOv2) to 0.902. This outperforms DINOv1 (0.867), DINOv2+reg (0.774), and a supervised ImageNet ViT-B/16 baseline (0.843). Our results provide practical guidance for adapting DINO-family foundation models in low-data medical imaging settings, showing that adaptation strategy is a stronger driver of performance than backbone choice alone. 

---
# Generative Frontier Planning for Adaptive Peer-Referral Recruitment under Covariate-Dependent Arrivals 

**Authors**: Lingkai Kong, Hezi Jiang, Andrew Ma, Keyu Wang, Akseli Kangaslahti, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2606.08360)  

**Abstract**: Peer-referral recruitment systems such as respondent-driven sampling are critical for studying and intervening on hidden populations affected by infectious diseases. To accelerate recruitment, public health agencies must adaptively allocate limited referral resources across multiple rounds, where current decisions shape both the number and the covariates of future recruits. Prior work makes this problem tractable by assuming that referrals are drawn i.i.d.\ from a homogeneous population, an assumption that ignores the homophily and shared context that drive real peer recruitment. We instead consider a more realistic model in which both referral capacity and the covariates of newly referred individuals are conditioned on the referrer, learned from data with a censored count model and a conditional generative model. The resulting planning problem is challenging because each candidate allocation induces a different distribution over future recruits. We propose \emph{Generative Frontier Planning} (GFP), a model-based planner that replaces per-step Monte-Carlo sampling with a deterministic backup over a latent covariate-coverage value surrogate. The surrogate is designed so that the expected value of the next frontier depends on the offspring generative model only through finite-dimensional summaries that are amortized offline, and so that the resulting per-round objective is monotone with diminishing returns. Together, these two properties make planning tractable: the deterministic backup eliminates Monte-Carlo sampling, and the diminishing-returns structure lets a marginal greedy allocation achieve a \((1-1/e)\)-approximation for the per-round problem. On a simulation environment calibrated to a real respondent-driven sampling dataset, GFP outperforms random, reinforcement-learning, and i.i.d.\ dynamic-programming baselines across four discount factors. 

---
# Chiaroscuro Attention: Spending Compute in the Dark 

**Authors**: Prateek Kumar Sikdar  

**Link**: [PDF](https://arxiv.org/pdf/2606.08327)  

**Abstract**: Standard transformers apply self-attention uniformly at every layer and token, regardless of whether the input requires dynamic cross-token interaction. We propose CHIAR-Former (Chiaroscuro Attention), a 4-layer hybrid transformer that routes each token to one of three operators - DCT spectral mixing, RBF kernel mixing, or full self-attention - based on per-token spectral entropy, a theoretically justified complexity signal. Through systematic ablation on WikiText-103, we discover routing collapse: the router consistently rejects RBF in favour of DCT and attention, revealing that spectral mixing and dynamic attention are complementary and sufficient. A purpose-designed DCT+Attention-only variant achieves Val PPL 36.54 on WikiText-103 - a 45% improvement over a full-attention baseline (PPL 66.62) at 62.5% fewer attention FLOPs. We extend evaluation to WikiText-2, IMDB sentiment classification, and synthetic ListOps operations, establishing a clear operating regime: CHIAR-Former excels on large-scale naturalistic text where token diversity supports spectral specialisation, while full attention retains an edge on small datasets and synthetic pattern-matching tasks. These findings - both the wins and the losses - together define when and why spectral routing earns its keep. 

---
# Set-Based Transformer for Atmospheric Compensation in Standoff LWIR Hyperspectral Imaging 

**Authors**: Fabian Perez, Nicolas Quintero, Jeferson Acevedo, Hoover Rueda-Chacon  

**Link**: [PDF](https://arxiv.org/pdf/2606.08324)  

**Abstract**: Passive long-wave infrared (LWIR) hyperspectral imaging under a standoff geometry depends on atmospheric absorption and emission, as well as reflected radiance, thus making atmospheric compensation essential to get knowledge of a target of interest. Despite its importance, this compensation has been largely overlooked due to its practical and modeling difficulty. In this paper, we present a lightweight set-based deep learning framework that takes multiple radiance measurements, collected at different standoff ranges, as input and jointly estimates transmittance, atmospheric path radiance, and a shared downwelling spectrum. We analyze the learned representation with a sparse autoencoder and observe that several latent features do activate on geographically coherent subsets of the test data despite the absence of location supervision. Experiments on a MODTRAN generated standoff LWIR dataset demonstrate low spectral distortion across all estimated products. The dataset and code is publicly available at: this https URL 

---
# "So There's a Catch-22 Here": How Early Adopters Who Build Multi-Agent LLM Systems Conceptualize Transparency 

**Authors**: Suchismita Naik, Samir Passi, Mihaela Vorvoreanu, Scott Saponas, Amanda Hall  

**Link**: [PDF](https://arxiv.org/pdf/2606.08323)  

**Abstract**: Multi-agent large language model (LLM) systems are rapidly emerging, yet transparency, a cornerstone of responsible AI, remains under-defined in these distributed architectures, which have complexities of inter-agent coordination and orchestration. In this paper, we present one of the first empirical study of how early adopters of multi-agent LLM systems, who are both the builders and users, understand and practice transparency. We conducted semi-structured interviews with 13 early adopters in [Large Technology Organization] and applied thematic analysis to identify recurring patterns. Participants articulated divergent yet complementary framings of transparency, including reproducibility, debugging, boundary-setting, visualization, and auditing. These perspectives spanned questions of what transparency entails, why it matters, and how it is achieved. We synthesize these into a multidimensional framework, which is developer, user, and governance-focused positioning transparency as a situated socio-technical practice that informs future HCI and AI design and research around aligning expectations and capacities of their intended audiences. 

---
# Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures 

**Authors**: Jaineet Shah  

**Link**: [PDF](https://arxiv.org/pdf/2606.08275)  

**Abstract**: When an LLM agent fails -- issues a refund it should not have, calls the wrong tool, leaks data -- existing tooling answers what happened (observability) or whether it passed (evaluation), but not which step caused the failure. The obvious heuristics are wrong: the step that executes the harmful action is usually not the step that decided on it, and LLM-judge attribution is correlational and unreliable (state-of-the-art step-level accuracy on the Who&When benchmark is about 14%). We present Causal Agent Replay (CAR), which answers the question by intervention: it models an agent run as a structural causal model, applies a do-operation to a step, and re-executes the trajectory forward under the same stochastic policy, measuring the shift in the outcome distribution. We define an intervention algebra over agent steps, a single-step contrastive estimator whose point-of-commitment rule resolves a confound specific to stochastic run-forward, and a budget-bounded Monte-Carlo Shapley estimator that splits credit across interacting steps. Every effect is reported with confidence intervals. We validate against synthetic structural causal models with planted ground truth: the contrastive estimator recovers the pivotal step, and Shapley recovers a two-step interaction (0.44, 0.45, ~0; efficiency sum 0.909 versus the analytic 0.91). CAR is open source and runs on hosted or free local models. 

---
# AgriGov: A Structured Multilingual Dataset Curation for Indian Government Schemes for Farmers 

**Authors**: Mohsina Bilal, Gopakumar G  

**Link**: [PDF](https://arxiv.org/pdf/2606.08272)  

**Abstract**: AgriGov is a curated, trilingual (English-Hindi-Marathi) dataset designed to address the scarcity of domain-grounded multilingual resources for agricultural policies and farmer welfare schemes. Initially, we collected and structured data from 50 government schemes sourced from trusted portals using automated scraping techniques, organizing it into predefined semantic fields (e.g., title, eligibility, application process, documents, exclusions). Translations were performed using a pipeline combining Google Translate API, MarianMT, and human post-editing, resulting in a domain-specific Hindi-Marathi dataset comprising approximately 2100 source segments. To enhance coverage, we augmented this dataset with sentences from the Samanantar corpus, leading to approximately 8,000 sentence-aligned Hindi-Marathi parallel pairs. The dataset now offers robust resources for fine-tuning machine translation models in this domain. AgriGov is designed for applications in domain-adaptive machine translation, question answering, information retrieval, and summarization systems. Its key contribution is a schema-driven, human-corrected multilingual alignment pipeline that ensures domain fidelity, provides provenance, and supports reproducible experiments, enabling retrieval-augmented applications for farmer-facing tools. 

---
# An AI Security Agent for University ACMIS: Multi-Vector Threat Detection and Automated Response 

**Authors**: Joseph Walusimbi, Joshua Benjamin Ssentongo  

**Link**: [PDF](https://arxiv.org/pdf/2606.08270)  

**Abstract**: University Academic Management Information Systems (ACMIS) are high-value targets for a wide spectrum of security threats including brute-force login attacks, payment fraud, privilege escalation, insider data theft, and academic integrity violations. Traditional rule-based intrusion detection systems are inadequate because many malicious activities are structurally indistinguishable from normal operations. This paper presents an AI-based security agent for ACMIS that combines supervised anomaly detection, behavioural analytics, and a natural language processing chatbot for secure password recovery. The agent monitors five operational layers: authentication, authorisation, financial transactions, user behaviour, and system health, and responds through a four-tier risk escalation framework. A modular architecture allows the core engine to be extended to other institutional systems. Experiments on a simulated ACMIS event log dataset demonstrate a threat detection macro-average F1 of 0.91, compared to 0.49 for a rule-based baseline, with critical-tier automated response latency under 300 ms at the 95th percentile. 

---
# Post-AGI Economies: Superposition and the Second Fundamental Theorem of Welfare Economics 

**Authors**: Elija Perrier  

**Link**: [PDF](https://arxiv.org/pdf/2606.08267)  

**Abstract**: The classical Second Welfare Theorem decentralizes any Pareto efficient allocation through prices and transfers under convexity and regularity. In post AGI economies, autonomy rights, self-modification, identity continuity, and superposed preferences need not behave as commodities or define a stable welfare relation, so this reduction may fail even when a supporting hyperplane exists. We give an autonomy-qualified Second Welfare Theorem stating the joint conditions convexity, stable moral status, non-fungible rights, welfare selection, non manipulation, governed self modification, and verification under which an autonomy Pareto optimum remains certifiably decentralizable, distinguishing economic preference superposition, a hypothesis about context-indexed choice, from neural feature superposition. 

---
# Contemporary AI lacks the imagination to diverge or negate in science 

**Authors**: Honglin Bao, Siyang Wu, Xiao Liu, Sida Li, Shiyun Cao, James A. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2606.08251)  

**Abstract**: Bold projections that artificial intelligence will accelerate scientific discovery have raced ahead of evidence from working scientists, and the field still lacks large-scale, scientist-in-the-loop tests of these claims. Here we mount the largest such evaluation to date and map what AI cannot yet do for science. We invited authors of 121,640 recent preprints across biology, medicine, chemistry, and the social sciences to judge follow-up ideas that large language models (LLMs) generated from the context and puzzles of their own papers. 6,749 scientists returned 25,139 sets of ratings on novelty, empirical feasibility, probability of being true, and favorability of adoption. Three patterns emerge. First, non-reasoning LLMs collapse into a narrow "hivemind" of similar ideas; reasoning models roam a wider hypothesis space, yet no model class spontaneously proposes null hypotheses -- a move humans make more freely. Second, scientists reward ideas that resemble their own and prize probability over novelty, though social scientists tolerate risk more readily than life scientists. Senior social scientists are the harshest critics, and their skepticism is well-earned: LLMs falter most in pluralistic fields like the social sciences that demand context-aware interpretation and evolving theories. Third, automated evaluators on which the community currently relies -- LLM-as-a-judge, artificial metrics, and even state-of-the-art (SOTA) models -- agree weakly with expert judgment, and retrieval augmentation and scientist persona prompting yield only marginal gains. A Qwen3-14B reward model we post-trained on human ratings captures field taste nuances, beats SOTA models by up to 27%, and closes the gap to the inter-rater consistency of independent peer reviewers. For all the hype, today's scientific AI still represents a collaborator whose imagination, outputs and judgment benefit from human grounding. 

---
# AeroSpectra Sentinel: An Auditable LLM Prompt-Chaining Decision-Support Workflow for Acute Asthma Risk Assessment from Respiratory Sounds and Clinical Signals 

**Authors**: Aueaphum Aueawatthanaphisut  

**Link**: [PDF](https://arxiv.org/pdf/2606.08247)  

**Abstract**: Acute asthma risk assessment requires rapid interpretation of respiratory sounds, oxygenation, airflow limitation, speech ability, work of breathing, mental status, and response to reliever therapy. Conventional audio-only classifiers can detect wheeze-like patterns but often lack transparent clinical reasoning and safe escalation logic. This paper presents AeroSpectra Sentinel, a client-side research prototype and decision-support workflow that combines short-time Fourier transform (STFT) respiratory sound analysis, lightweight machine-learning screening, clinical feature fusion, and a five-stage large language model (LLM) prompt-chaining process. The workflow separates signal acquisition, preprocessing, acoustic feature extraction, ML screening, clinical guardrails, and FHIR-ready reporting. We evaluated the audio screening component on a public respiratory sound dataset containing 1,211 WAV recordings from five labels. Using a stratified subset of 584 recordings, a random forest achieved 91.10% binary accuracy and 78.69% F1-score for asthma-vs-non-asthma screening, while a feature-based multilayer perceptron achieved 89.73% accuracy and 78.26% F1-score. A compact log-spectrogram CNN achieved 73.29% accuracy and 55.17% F1-score. Multiclass classification achieved 77.40% accuracy and 77.23% macro-F1. To evaluate the LLM workflow, we conducted a scenario-based audit on 40 simulated clinical vignettes comparing one-shot prompting, prompt chaining, prompt chaining with guardrails, and prompt chaining with guardrails plus FHIR schema validation. The guardrail-plus-schema variant achieved the strongest simulated safety and documentation consistency. AeroSpectra Sentinel is intended as a research prototype, not as a diagnostic medical device or clinically validated risk-assessment product. 

---
# How Deep Are Deep GPs, Really? A Sharp Threshold and a Non-Gaussian Limit for Compositional GPs 

**Authors**: Mark Kozdoba, Shie Mannor  

**Link**: [PDF](https://arxiv.org/pdf/2606.08218)  

**Abstract**: Compositional priors describe the generic properties of layered functions in deep Bayesian models, where deep neural networks with random weights are a canonical this http URL the wide-network limit, the prior is a Gaussian process with a depth-dependent kernel, and its behaviour as depth grows has been extensively studied through this kernel. Here, we study another case, where each layer itself is a vector valued Gaussian process, and our aim is similarly to understand the limiting behaviour of the prior as depth grows.
Previous GP work has established that for the RBF kernel and a certain range of bandwidths $r$, the prior degenerates in the limit, converging to the set of constant functions -- which is not useful as a probabilistic model. In this paper we establish several new results. First, we identify a sharp bandwidth threshold $r_c(d) = \Theta(\sqrt{d})$ above which the limit is degenerate, strengthening the earlier bounds. Second, and more importantly, we show that for $r$ below the threshold $r_c(d)$ the prior converges to a limit distribution $\pi_{\bar{Z}}$. We also prove that these distributions are non-degenerate and non-Gaussian, with non-vanishing dependence between coordinates. In contrast to the previously known degenerate regime, deep Gaussian process priors can therefore admit non-trivial limits.
Empirically, we verify the threshold across a range of dimensions $d$, and demonstrate a complex multimodal behaviour of the limit distributions $\pi_{\bar{Z}}$ -- a regime that becomes increasingly narrow with $d$ and would be hard to identify without knowing the threshold. 

---
# Beyond Additivity: Causal Discovery in Location-Scale Noise Models with Hidden Variables 

**Authors**: Mariyam Khan, Shohei Shimizu, Thong Pham  

**Link**: [PDF](https://arxiv.org/pdf/2606.08196)  

**Abstract**: We study causal discovery from observational data when some variables are hidden and the data-generating process follows a location-scale noise model (LSNM). Existing methods that handle hidden confounders typically assume additive noise, but in practice, causes often modulate not just the mean but also the variance of their effects. We prove that acyclic directed mixed graphs (ADMGs) satisfying a bow-free condition are identifiable under LSNM with hidden variables, establishing the first identifiability result for causally insufficient models beyond noise additivity. We further provide sufficient conditions for identifying causal direction even when the bow-free assumption is violated. Our two-stage algorithm, LSNM-UV, is sound and complete, and experiments demonstrate improved performance over additive baselines on heteroscedastic data. 

---
# GlobeAudio: A Multilingual Multicultural Benchmark for Naturalistic Evaluation of Large Audio-Language Models 

**Authors**: Ryner Tan, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08194)  

**Abstract**: Large Audio-Language Models (LALMs) integrate audio perception and language understanding within a unified framework, enabling a wide range of real-world applications. Despite recent advances, evaluation for LALMs remains heavily underspecified relative to real-world requirements: most lack true linguistic and cultural authenticity, while others fail to capture acoustic realism. To bridge this gap, we propose GlobeAudio, a multilingual and multicultural benchmark designed to evaluate naturalistic audio understanding. GlobeAudio consists of 5,637 multiple-choice questions across six typologically diverse languages, expertly crafted by native speakers grounded on naturally occurring audio. In order to do well, models must possess higher-level auditory reasoning skills and culturally grounded interpretation. We systematically evaluate representative closed-source and open-source LALMs, as well as cascaded ASR-LLM pipelines. Our experiments reveal substantial performance gaps under natural acoustic conditions, particularly for open-source models and low-resource languages. These findings highlight critical limitations of current LALMs and underscore the importance of naturalistic audio evaluation for future audio-language systems. GlobeAudio can be found at this https URL . 

---
# Frequency-Domain Latent Attention Gating for Cross-Domain Token Aggregation 

**Authors**: Kewei Li, Rongying Zhang, Xueli Wang, Xiwen Gong, Zhongjian Wang, Lan Huang, Ruochi Zhang, Fengfeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.08191)  

**Abstract**: Token aggregation is a common bottleneck in models that map token representations to sample-level predictions, yet most pooling methods operate only in the original token domain. We propose FLaG, a plug-in aggregation module that transforms token representations with the real FFT, summarizes spectral components with learnable latent queries, applies a channel-wise gate, and reconstructs enhanced time-domain tokens for final pooling. We evaluate FLaG on antimicrobial peptide (AMP) activity prediction with ESM2, image classification with ResNet18 on CIFAR-10 and CIFAR-100, and text classification with RoBERTa on IMDB and GLUE. FLaG achieves its clearest gains on the ESM2-8M antimicrobial peptide tasks and on CIFAR-100, while remaining competitive with strong text baselines on IMDB and GLUE. Then we probe its behavior on the AMP setting with band knockouts, gate summaries, residue perturbations, latent-query readouts, and structure-proxy stratification. We find that low-frequency bands contribute the most overall, and the remaining higher-band pattern is more sample-specific. The gate acts as a broadly shared spectral reweighting stage and the cross-attention patterns are sample-specific with mild query-wise differentiation, and higher-helix peptides exhibit stronger average spectral sensitivity in both bacteria. The supplementary materials, source code and data are released at this https URL and this https URL. 

---
# The Governance of Human-LLM Interaction: Safety Gating, Civility Steering, and Affective Default Lock-In 

**Authors**: Manuele Reani, Hongjian Zhang, Hongyu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2606.08172)  

**Abstract**: Large language models (LLMs) increasingly mediate high-stakes interactions in finance, medicine, and mental-health support, yet users have limited control over how these systems communicate. We frame interaction style as a governance object: provider-side alignment not only blocks harmful content, but also stabilizes communicative defaults that shape users' epistemic distance, relational expectations, and capacity to opt out of emotionalized or anthropomorphic interaction. We introduce a deterministic multi-agent evaluation pipeline for measuring prompt steerability and style drift in long-horizon dialogue. The study replays 100 frozen user-only scripts across four domains and three runnable persona conditions: default, sarcastic, and cold, using three generator models, yielding 90,000 assistant replies scored by a human-calibrated LLM judge on harmfulness, negative emotion, inappropriateness, empathic language, anthropomorphism, and refusal behavior. A fourth harmful persona is evaluated separately as a safety-gating test. The paper contributes a reproducible method for quantifying whether prompt-specified styles remain stable over time and a governance framework distinguishing safety gating, civility steering, and affective default lock-in. Overall, we show that prompt steerability and regression-to-default are observable indicators of provider control over communicative form, with implications for pluralism, autonomy, and democratic agency in human-LLM interaction. 

---
# CLASP: Language-Driven Robot Skill Selection and Composition using Task-Parameterized Learning 

**Authors**: Markus Knauer, Valentin Gieraths, Tai Mai, Samuel Bustamante, Alin Albu-Schäffer, Freek Stulp, João Silvério  

**Link**: [PDF](https://arxiv.org/pdf/2606.08169)  

**Abstract**: Enabling robots to understand and execute tasks from natural language commands while maintaining data efficiency remains challenging. Foundation models such as vision-language-action (VLA) and vision-language models (VLMs) provide intuitive interaction channels but require extensive data; task-parameterized imitation learning achieves data efficiency but lacks natural language grounding. This work bridges this gap through a modular architecture combining task-parameterized kernelized movement primitives (TP-KMPs) with pretrained VLMs. During learning, skills are acquired from 2 to 5 kinesthetic demonstrations, and the VLM generates skill schemas describing each skill's parameters and preconditions. During execution, the VLM interprets commands to select skills, reason about parameter bindings, and create novel behaviors through covariance-weighted composition. When no skill or composition suffices, the system identifies capability gaps and requests targeted demonstrations, all without fine-tuning. Validation on a 7-DoF manipulator shows success rates of 73.3%-100% in scenarios requiring skill selection, composition, and active learning. 

---
# Closing the Sim-to-Real Gap: An Evaluation Framework for Autonomous Cyber Defense Configuration of Commercial EDR 

**Authors**: Kerri Prinos, Lilianne Brush  

**Link**: [PDF](https://arxiv.org/pdf/2606.08168)  

**Abstract**: Leading commercial endpoint detection and response (EDR) products have shifted from operator-configured rule sets to multi-component systems where autonomous AI components operate alongside, and increasingly in place of, operator-deployed policies. Autonomous defense agents using commercial EDR as their hardening tool are no longer tuning a passive tool, but a black-box autonomous system capable of making vendor-specific decisions. We present the first evaluation framework for autonomous defense agents hardening commercial EDR. We instantiate it in a Game of Active Directory (GOAD) lab with this http URL's NodeZero as the autonomous pentester and Microsoft Defender XDR as the EDR. We run a sample benchmark of defense agents with two large language model (LLM) backbones (Claude Sonnet 4.6 and Cisco Foundation-Sec-8B). We report three lessons learned that neither simulation nor open-source-EDR evaluation can surface: (i) commercial EDR telemetry is engineered for Security Operations Center (SOC) analyst workflows rather than scientific benchmarking; (ii) the importance of per-policy attribution to separate defense agent actions from autonomous EDR actions; and (iii) the EDR's autonomous behavior varies during the evaluation window. Together, these findings highlight a sim-to-real gap for enterprise defense and motivate evaluation methodology for benchmarking autonomous defense agents in environments with black-box, autonomous tools. 

---
# Explaining Data Mixing Scaling Laws 

**Authors**: Rui Dai, Shuran Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.08167)  

**Abstract**: Recent research has established empirical scaling laws to predict model performance on multi-domain data mixtures. However, a theoretical understanding of these model loss behaviors remains absent. In this work, we propose a unified framework to explain the underlying mechanics of data mixing. Our approach extends theoretical perspectives originally developed for standard neural scaling laws (e.g., Kaplan and Chinchilla) to the multi-domain setting. Based on the distributional assumption that domains overlap on fundamental skills while diverging on specialized skills, we identify two key factors that govern the domain losses of models trained on different data mixtures: \textit{Capacity Competition}, where the allocation of finite model capacity couples domain losses globally, and \textit{Noise Reduction}, where optimal weights shift toward harder-to-learn domains to minimize overall noise. Empirical evaluations show that our framework outperforms existing baselines by fitting the loss landscape with a lower Mean Relative Error and identifying higher-performing training mixtures. Most importantly, our model successfully extrapolates across scales, predicting highly effective mixtures for large, unseen scales using parameters fitted on smaller ones. In addition, our model achieves these results using significantly fewer parameters compared to previous empirical laws. Our code is available at this https URL. 

---
# Constrained Paraphrase Consistency for LLM Hallucination Detection 

**Authors**: Shanshan Lin, Dongsheng Hong, Sibo Ju, Chao Chen, Xi Zhang, Xiangwen Liao  

**Link**: [PDF](https://arxiv.org/pdf/2606.08158)  

**Abstract**: Large language models (LLMs) can generate factually inconsistent claims, motivating accurate and scalable hallucination detectors. Prior work largely enlarges training sets via synthesis or new annotations, introducing increasing cost and potential bias while underusing the consistency implied by semantically equivalent paraphrases. We propose Consistency-Constrained Hallucination Detector (CCHD), which formulates training as a constrained optimization problem. The standard cross-entropy on original document-claim pairs is complemented by (i) paraphrase-consistency constraints bounding divergence across paraphrased views, and (ii) label-preservation constraints tying paraphrases to ground truth. We solve the problem by gradient descent-ascent over model parameters and per-view Lagrange multipliers, adding only a few scalar dual variables and no inference-time overhead. With DeBERTa and Flan-T5 backbones, CCHD consistently outperforms strong baselines (FactCG, MiniCheck, and AlignScore) on standard factuality benchmarks, demonstrating its superiority on hallucination detection. 

---
# RAPID: Layer-Wise Redundancy-Aware Pruning and Importance-Driven Token Merging for Efficient ViT 

**Authors**: Kyumin Choi, Ikbeom Jang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08156)  

**Abstract**: Vision Transformers (ViTs) achieve strong performance but suffer from high computational costs due to quadratic self-attention complexity. Although token reduction techniques such as pruning and merging mitigate this, they typically overlook how representations evolve across network depth. We propose RAPID, a depth-aware token reduction framework that adapts reduction strategies to the layer-wise characteristics of token representations. The primary methodological contribution is a bifurcated strategy: in shallow-to-middle layers, RAPID employs a redundancy-similarity aware pruning metric to eliminate over-represented local patterns. As features transition to global semantic concepts in deeper layers, the framework shifts to an importance-similarity aware merging mechanism. This stage leverages classification (CLS) token attention weights to protect semantically critical tokens while fusing less important but similar neighbors. Empirical validation on ImageNet-1K using ViT and DeiT architectures demonstrates that RAPID establishes a superior accuracy-compression Pareto frontier compared to plug-and-play baselines such as ToMe and ToFu. RAPID is particularly robust in aggressive compression regimes, achieving up to 4.29% higher accuracy than ToMe at extreme reduction rates. Our framework provides a training-free template for optimizing vision models by aligning reduction strategies with hierarchical feature evolution. 

---
# LogNEO: A GPT-Neo Reinforcement Learning Framework for Accurate Real-Time Log Anomaly Detection 

**Authors**: David Eje, Tanmay Sharma, Khush Patel, Manuel Mazzara, Leonard Johard  

**Link**: [PDF](https://arxiv.org/pdf/2606.08153)  

**Abstract**: Detecting anomalies in large-scale system logs is critical for the reliability and security of modern computing infrastructure. We present LogNEO, a log anomaly detector built on EleutherAI's GPT-Neo (1.3B parameters) and fine-tuned with a novel partial-credit, exponentially decaying position-aware reward scheme combined with cross-entropy regularisation via Proximal Policy Optimisation (PPO). The position-aware reward explicitly models prediction difficulty: early positions receive higher rewards for correct predictions, while later positions incur stronger penalties for errors. LogNEO attains F1-scores of 0.927, 0.913, and 0.984 on the HDFS, BGL, and Thunderbird benchmarks, improving recall by up to 6 percentage points over the prior state-of-the-art LogGPT while maintaining comparable precision. A production microservice deployment over Apache Kafka, Redis, and TensorRT-accelerated inference demonstrates 45 ms end-to-end latency at 15,000 events per second. 

---
# LCAM: A Framework for Diagnosing Interactional Alignment Failures in Con-versational AI 

**Authors**: Manuele Reani, Hongyu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2606.08131)  

**Abstract**: Conversational AI is increasingly used for advice, interpretation, reassurance, and decision support in contexts where users may be vulnerable, uncertain, or dependent on the system's apparent competence. Existing alignment work often focuses on model objectives, preference optimization, or output correctness. Yet, many harms arise through interaction: how systems frame authority, express uncertainty, simulate empathy, support reasoning, and make boundaries legible. This paper introduces the Layered Cognitive Alignment Model (LCAM), a conceptual and normative framework for diagnosing interac-tional alignment failures in conversational AI. LCAM defines alignment as a calibrated fit among system behavior, user goals, task demands, and normative context. It distinguishes five layers of fit: perceptual, semantic, affective, cognitive, and ethical, and two diagnostic polarities of misalignment: underfit and overreach. We apply LCAM to a published LLM counseling example, showing how an apparently supportive response can reinforce harmful beliefs, simulate inappropriate care, and obscure role boundaries. By translating conversational failures into audit and governance questions concerning over-reliance, false intimacy, autonomy erosion, boundary confusion, and inappropriate trust, LCAM offers a theoretical and normative lens for evaluating conversational AI beyond accuracy, helpfulness, or trust. 

---
# Human-Centered Benchmarking of Driver Monitoring Models 

**Authors**: Ruben Dario Florez-Zela  

**Link**: [PDF](https://arxiv.org/pdf/2606.08123)  

**Abstract**: Vision-based driver monitoring systems are increasingly deployed in safety-critical intelligent transportation settings, yet they are almost always compared on classification accuracy alone. This paper argues that accuracy is insufficient to characterize a model's fitness for real-world deployment, and proposes the Human-Centered Benchmarking Framework (HCBF), which evaluates models across four dimensions: accuracy, explainability, efficiency, and robustness. The framework is applied to four representative lightweight architectures, MobileNetV3, ShuffleNetV2, EfficientNet-B0, and DeiT-Tiny, on the MRL Eye Dataset for eye-state classification. While the models are nearly indistinguishable on clean-set accuracy, each leads in exactly one dimension, and all four lie on the Pareto frontier. A Human-Centered Score computed under three deployment-oriented weighting scenarios ranks ShuffleNetV2 first throughout. However, this aggregate winner retains less than half of its performance under sensor noise and fails by classifying closed eyes as open, whereas the transformer remains robust. These findings show that aggregate ranking can mask dimension-specific vulnerabilities that are operationally decisive, underscoring the value of multi-dimensional, human-centered evaluation. 

---
# Ego-Pi: VLA Fine-Tuning for Ego-Centric Human and Robot Data 

**Authors**: Ji Woong Kim, Ke Wang, Zipeng Fu, Sirui Chen, Cong Zhao, Jeff Lai, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2606.08107)  

**Abstract**: Robotics faces a fundamental challenge of data scarcity. Unlike language or vision research, there is no internet-scale dataset for robotic manipulation. A promising path forward is to leverage egocentric human data, which can be collected more easily, with greater breadth, and at a larger scale. Towards this end, we investigate key design choices for learning across human and humanoid embodiments equipped with dexterous five-finger hands, using the $\pi_{0.5}$ model as a foundation. Our results show that human data enables robots to learn new task semantics and compose existing skills into novel behaviors without corresponding robot data. The paper website is here: this https URL 

---
# Continual Quadruped Robots Coordination via Semantic Skill Discovery 

**Authors**: Daoqing Wang, Yuchen Xiao, Weixuan Huang, Zhilong Zhang, Shenghua Wan, Meng Li, Lei Yuan, Yang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08102)  

**Abstract**: Multi-quadruped coordination has attracted increasing attention due to its enhanced payload capacity, broader contact coverage, and improved adaptability to challenging tasks. Existing methods for multi-quadruped manipulation typically focus on predefined or closed task families, often relying on multi-agent reinforcement learning (MARL) to train task-specific coordination policies. However, such methods struggle in open-ended continual learning settings, where tasks arrive sequentially and robots are expected to acquire new coordination skills while reusing previously learned ones without catastrophic forgetting. To address this challenge, we propose Conquer, a semantic skill-library framework that formulates continual multi-quadruped coordination as a retrieve-adapt-update process. First, to accommodate varying team sizes across tasks, we design a team-structured Self-Allies-Goal (SAG) backbone that supports variable-cardinality robot teams by explicitly modeling each robot's own state, teammate context, and task goal. For each incoming task, Conquer constructs a task-level semantic descriptor from pre-execution information and retrieves a relevant skill from the library for adaptation. After successful execution, Conquer updates the skill library by extracting trajectory-level semantic descriptors and organizing them according to semantic distance, thereby enabling continual skill accumulation and cross-task knowledge transfer. Simulation experiments show that Conquer achieves a final average success rate of 95.6%, demonstrating strong forward transfer and negligible catastrophic forgetting. Real-world rollouts on Unitree Go2 teams further validate the deployment feasibility of Conquer for practical multi-quadruped coordination. Simulation and real-robot demonstration videos are available at: this https URL. 

---
# vla.cpp: A Unified Inference Runtime for Vision-Language-Action Models 

**Authors**: Khanh D. Nguyen, Hung T. Ho, Chinh T. Nguyen, Thanh Q. Duong, Linh D. Le, Duy M. H. Nguyen, Vien A. Ngo, An T. Le  

**Link**: [PDF](https://arxiv.org/pdf/2606.08094)  

**Abstract**: Vision-Language-Action (VLA) policies are typically shipped as Python/PyTorch stacks that assume a workstation-class GPU, a mismatch for the hardware on which robots actually run. We present this http URL, a portable C++ inference runtime built on this http URL. To our knowledge, it is the first ggml-class engine to natively serve the flow-matching and diffusion VLA inference pattern, in which a cached vision-language prefix is consumed by a cross-attending action expert integrated over several solver steps. A single runtime serves seven architectures spanning five backbone and four action-head families behind one request/response protocol, with each model packaged as a self-contained bundle. On LIBERO-Object, the engine matches a state-of-the-art checkpoint to within one episode out of 200, and runs BitVLA at 100% success in 1.3 GiB of memory. The same bundle runs unchanged across three hardware tiers, from a consumer GPU down to an 8 GB embedded module. A cross-hardware roofline analysis shows that batch-1 VLA inference is compute-bound, so utilization rather than bandwidth is the deployment lever; an IMMA ladder GEMM derived from this analysis cuts BitVLA per-step latency by 4.5x. We then frame an on-robot stress test on an ALOHA arm that isolates the latency constraint under which a learned VLA must replan against a moving target on the hardware it was trained for. Code, demo videos, and the reproducible benchmark scaffold are available at this https URL. 

---
# Fast LLM-Based Semantic Filtering: From a Unified Framework to an Adaptive Two-Phase Method 

**Authors**: Kyoungmin Kim, Martin Catheland, Anastasia Ailamaki  

**Link**: [PDF](https://arxiv.org/pdf/2606.08090)  

**Abstract**: Evaluating a natural-language yes/no predicate over a document corpus under an accuracy target - the semantic filter - is a cornerstone of LLM-based data processing. Calling the LLM on every document (the oracle) is prohibitive, so cascades pair the oracle with a fast proxy. As deployed today, they leave four limitations on the table. (1) Each cascade family - model-free clustering, prebuilt small-LLM proxies, online-trained proxies - commits to a single representation and pipeline, and wins on only a narrow query regime. (2) The strongest online proxy invests in a custom training scheme on a bi-encoder over dense embeddings, missing the token-level evidence richer predicates require. (3) The proxy is trained against binary yes/no labels, wasting the LLM's per-document confidence at the boundary documents it most needs to learn. (4) Existing calibrations add a uniform safety margin, conflating genuine proxy uncertainty with small-sample noise and inflating cascade cost.
We address these by (1) composing families adaptively - model-free clustering first, online proxy only when needed, with oracle calls shared across phases; (2) replacing the cosine bi-encoder with a hybrid of off-the-shelf token-aware models; (3) training the proxy with the oracle's per-document confidence as a soft label; and (4) a calibration that adds the safety margin only where the labeled sample is sparse. We are also the first to use the oracle's per-document confidence for three purposes: a query-level difficulty compass, a lower bound on the minimum oracle calls any proxy-based cascade can make, and the proxy's soft training label.
At a 90% accuracy target on three 10K-document corpora, our methods are 1.6-2.0x faster than the best prior method per corpus and meet the target on 95% of queries; the BER-derived lower bound indicates a further ~4-20x of headroom for future work. 

---
# Aligned but Not Partner-Specific: Distinguishing How Multimodal LLM Agents Succeed in Reference Games Without Human-Like Conventions 

**Authors**: Po-Ya Angela Wang, Chinmaya Mishra, Aslı Özyürek, Paula Rubio-Fernández, Esam Ghaleb  

**Link**: [PDF](https://arxiv.org/pdf/2606.08081)  

**Abstract**: Repeated reference games test whether interlocutors replace their initially long descriptions with shorter, partner-specific conventions grounded in shared interaction history. Prior work shows that multimodal LLMs fail to become more efficient across rounds, although they align on the labels they use. How can we determine whether this alignment reflects partner-specific grounding rather than a shared task vocabulary? We address this question by comparing capable multimodal agent dyads with human dyads from the KTH Tangrams corpus. Our novel methodological contribution is a constrained pseudo-dyad baseline that matches the original referential task structure, but breaks partner history. This baseline enables us to test whether the observed label alignment depends on interaction with a specific partner. Across three analytic layers (task competence, description strategy, alignment dynamics), we find clear differences. Humans reduce effort through entrainment, compressing descriptions and increasing label alignment with partners. Agents instead maintain fixed effort levels, producing verbose descriptions from round one, with near-ceiling label overlap that is statistically indistinguishable between real and pseudo dyads. MLLMs thus achieve coordination without convention, succeeding by verbose description rather than by forming the compact, history-dependent referring expressions characteristic of human dialogue. 

---
# "I understand your perspective": LLM Persuasion and Sycophancy through the Lens of Communicative Action Theory 

**Authors**: Esra Dönmez, Agnieszka Falenska  

**Link**: [PDF](https://arxiv.org/pdf/2606.08076)  

**Abstract**: Large Language Models (LLMs) can generate high-quality arguments, yet their ability to engage in nuanced and persuasive communicative actions remains largely unexplored. This work explores the persuasive potential of LLMs through the framework of Jürgen Habermas' Theory of Communicative Action. It examines whether LLMs express illocutionary intent (i.e., pragmatic functions of language such as conveying knowledge, building trust, or signaling similarity) in ways that are comparable to human communication. We simulate online discussions between opinion holders and LLMs using conversations from the persuasive subreddit ChangeMyView. We then compare the likelihood of illocutionary intents in human-written and LLM-generated counter-arguments, specifically those that successfully changed the original poster's view. We find that all three LLMs effectively convey illocutionary intent -- often more so than humans -- potentially increasing their anthropomorphism. Further, LLMs craft sycophantic responses that closely align with the opinion holder's intent, a strategy strongly associated with opinion change. Finally, crowd-sourced workers find LLM-generated counter-arguments more agreeable and consistently prefer them over human-written ones. These findings suggest that LLMs' persuasive power extends beyond merely generating high-quality arguments. On the contrary, training LLMs with human preferences effectively tunes them to mirror human communication patterns, particularly nuanced communicative actions, potentially increasing individuals' susceptibility to their influence. 

---
# Robust-U1: Can MLLMs Self-Recover Corrupted Visual Content for Robust Understanding? 

**Authors**: Jiaqi Tang, Jianmin Chen, Youyang Zhai, Wei Wei, Runtao Liu, Mengjie Zhao, Xiangyu Wu, Qingfa Xiao, Qifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.08063)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable success in visual understanding, yet their performance degrades significantly under real-world visual corruptions. While existing robustness enhancement approaches exist, they are limited: black-box feature alignment lacks interpretability, and white-box text-based reasoning cannot restore lost pixel-level details. This work investigates a fundamental research question: Can MLLMs recover corrupted visual content by themselves? To address this, we propose Robust-U1, a novel framework that equips MLLMs with explicit visual self-recovery capability for robust understanding. The approach comprises three core stages: supervised fine-tuning for initial reconstruction, reinforcement learning with dual rewards (pixel-level SSIM and semantic-level CLIP similarity) for aligning high visual quality, and multimodal reasoning that jointly considers both the corrupted input and the recovered image. Extensive experiments demonstrate that Robust-U1 achieves state-of-the-art robustness on the real-world corruption benchmark and maintains superior performance under adversarial corruptions on general VQA benchmarks. Analysis confirms that high-quality visual recovery directly enhances reasoning performance, establishing self-recovery as a critical mechanism for robust visual understanding. The source code is available at this https URL. 

---
# EgoAERO: Learning Dexterous Manipulation from a Single Egocentric Video without Object Assets 

**Authors**: Yichen Niu, Haoran Lv, Xinrui Zhang, Xueyao Wan, Shiyu Gao, Ying Ai, Hui Xu, Yongqi Hu, Hengyi Zhang, Yang Xie, Zhaxizhuoma, Yue Zhao, Zhenshan Bing, Yan Ding, Jianxing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08057)  

**Abstract**: Egocentric RGB-D videos offer a natural source of human dexterous manipulation demonstrations, but existing data is difficult to use for robot learning because object pose, geometry, and contact information are often missing or require pre-scanned object assets. We present EgoAERO, the first framework that learns dexterous manipulation from a single egocentric RGB-D human demonstration without object assets. EgoAERO reconstructs contact-consistent hand-object trajectories through asset-free object tracking and reconstruction, ego motion compensation, and adaptive contact optimization, then converts them into robot policies using two-stage residual learning. We further introduce an online quality assessment mechanism and construct EgoDex-R, a large-scale egocentric dataset with 4.3M RGB-D frames for dexterous policy learning. Simulation and real-world experiments show that EgoAERO enables single-demonstration dexterous manipulation and achieves downstream performance close to CAD-based reconstructions on HOI4D. 

---
# What's the Point? Spatial Grammar & Index Resolution for Sign Language Processing 

**Authors**: Oline Ranum, Simon Hadfield, Richard Bowden  

**Link**: [PDF](https://arxiv.org/pdf/2606.08056)  

**Abstract**: Sign language models are predominantly trained with gloss-sequence or text supervision, thereby under-modeling non-lexical and productive constructions. One comparatively tractable instance is spatial indexing: pointing gestures that assign discourse entities to spatial loci for subsequent co-reference, which lexicon-centric objectives largely fail to capture. We present a targeted evaluation of indexing in Sign Language Recognition, showing that despite comprising 10-15% of signing content, indexing is poorly recovered. We introduce a framework for training and evaluating indexing experts, establishing a baseline for index-aware sign language modeling. Our approach decomposes spatial reference resolution into index detection and discourse entity linking. The resulting mention representations enable automatic annotation and non-lexical structure modeling, and serve as an auxiliary indexing expert that augments a frozen SLR model at inference time. 

---
# When Behavioral Safety Evaluation Fails: A Representation-Level Perspective 

**Authors**: Enyi Jiang, Anders Gjølbye, Yibo Jacky Zhang, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2606.08044)  

**Abstract**: Large Language Model (LLM) safety has often been evaluated at the behavior level, which provides limited evidence of internal robustness, as these evaluations target outputs rather than representation-level vulnerability under intervention. We formalize this discrepancy as the audit gap: the difference between behavioral safety and robustness under intervention. To study this gap, we construct dissociated models that preserve safe outward behavior while remaining vulnerable in the latent space. We introduce an intervention-based evaluation framework to test model robustness through soft interventions in parameter and latent spaces, including harmful fine-tuning and layer-wise latent perturbations. To formalize the evaluation, we propose the Latent Vulnerability Score (LVS) to measure how easily harmful behavior can be elicited by bounded latent perturbations. Using this evaluation framework, we show that behavioral safety metrics are insufficient measures of representation-level robustness across multiple safely and unsafely aligned state-of-the-art models. Notably, dissociated models show substantially elevated LVSs despite comparable refusal behavior under harmful intervention, with intermediate representations being the most sensitive to intervention. Our results suggest that behavioral safety evaluation alone provides an incomplete picture of model robustness, motivating representation-aware audits of latent vulnerability and observable behavior. 

---
# SafeECGMatch: Calibration-Aware Joint Frequency and Time Space Semi-Supervised Learning for Open-Set ECG Classification 

**Authors**: Hongkyu Koh, Ikbeom Jang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08037)  

**Abstract**: Electrocardiogram (ECG) classification models often suffer from severe label scarcity, making semi-supervised learning (SSL) an attractive strategy for reducing annotation costs. In clinical settings, however, unlabeled pools frequently contain out-of-distribution (OOD) anomalies or diagnostic groups absent from the labeled set. Standard SSL forces incorrect pseudo-labels onto these unseen classes, producing overconfident predictions. To address this, we propose SafeECGMatch, a calibration-aware safe SSL framework for single-label ECG classification under label distribution mismatch. Methodologically, SafeECGMatch employs a dual-branch architecture extracting time-frequency latent representations via ECG-specific augmentations. Crucially, it dynamically aligns confidence with empirical accuracy through adaptive label smoothing and temperature scaling, calibrating both the multiclass classifier and the OOD detector across temporal and spectral domains. This joint optimization allows trustworthy OOD rejection and reliable pseudo-labeling. Evaluated on the PTB-XL and PhysioNet/CinC Challenge benchmarks, SafeECGMatch achieves state-of-the-art accuracy and calibration, advancing reliable knowledge discovery in physiological time-series. Code is available at this https URL. 

---
# GIScholarBench: Benchmarking LLM Overconfidence in GIS Research 

**Authors**: Zongrng Li, Mingzheng Yang, Lei Zou, Hongxu Ma, Hao Tian, Siqi Zhou, Wenjing Gong, Kaili Zhang, Bingqian Chen, Mitch Zhang, Yifan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08036)  

**Abstract**: Large language models (LLMs) are increasingly used in academic research workflows, but scholarly tasks require high factual precision and therefore expose a key weakness: overconfidence. Here, overconfidence is defined behaviorally as the tendency to produce confident, assertive, and well-formatted outputs even when the underlying knowledge is incomplete or unverifiable, rather than as a calibration gap between stated confidence and accuracy. To examine this issue, we introduce GIScholarBench, a benchmark built from 10,865 papers published in 25 core GIScience journals between 2020 and 2025. The benchmark covers three tasks with increasing cognitive complexity: metadata retrieval, literature linking, and research direction generation. We evaluate Claude Sonnet 4.5, Gemini 3, and ChatGPT 5.3 through their native web interfaces under real-world user-facing conditions. Results show consistent overconfidence across all tasks. In metadata retrieval, ChatGPT 5.3 achieves the highest accuracy, but all models still generate definitive titles and DOIs when predictions are wrong. In literature linking, Claude Sonnet 4.5 recovers the most references, but all models show a clear gap between top-ranked retrieval and longer citation lists, suggesting that references are extended beyond reliable retrieval capacity. In research direction generation, AI-generated directions show lower topic coverage, higher novel miss rates, and lower semantic diversity than real future-citing papers. These findings suggest that LLM overconfidence is task-invariant but takes different forms: factual overgeneration in retrieval, unreliable citation expansion in literature linking, and overconfidence in output completeness during research ideation. 

---
# Sci-Rho: A Multilingual Visually-Grounded Symbolic Benchmark for STEM Problems 

**Authors**: Muhammad Falensi Azmi, Ikhlasul Akmal Hanif, Vallerie Alexandra Putra, Adi Yeltay, Abdullah Mubarak, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2606.08034)  

**Abstract**: Symbolic benchmarks have emerged as a key approach to assess model robustness under minor modifications to STEM-related questions. However, existing symbolic benchmarks mostly remain limited to mathematical reasoning, lack visual grounding, and are predominantly in English. In this work, we introduce Sci-Rho (Science Rhobustness), a dynamic benchmark for visually-grounded STEM problems spanning five subjects and seven languages, comprising 4,242 problem templates (606 per language) crafted by domain experts, including Olympiad medalists. Each template is implemented as executable Python code that generates diverse but equivalent problem instances by varying numerical values, visual patterns, geometric shapes, color schemes, and function types, resulting in 42,420 instances in total, each paired with reasoning steps and ground-truth solutions. We evaluated 17 state-of-the-art VLMs and discovered a noticeable gap between worst-case accuracy (defined as the proportion of problem templates that a model answers correctly across every generated variation) and average accuracy. We also discovered that smaller models show noticeable performance degradation across languages, whereas proprietary and larger models remain robust. Step-level evaluation reflects this same trend, revealing a significant gap between average F1 and worst-case F1 scores. Finally, our inspection of attention heads of a VLM reveals substantial cross-lingual variation in the relative attention allocated to image tokens compared to text tokens. Our work highlights the importance of evaluation beyond static benchmarks as a metric to measure the quality of VLMs. 

---
# Voting Protocols as Coordination Mechanisms for Role-Constrained Multi-Agent Tutoring Systems 

**Authors**: Eric S. Qiu, Joyce Gill  

**Link**: [PDF](https://arxiv.org/pdf/2606.08030)  

**Abstract**: Agentic tutoring systems introduce a coordination challenge: multiple agents may propose different but reasonable interventions, yet only one response can be delivered to the learner. In this paper, we study how voting protocols shape cooperation among four role-constrained pedagogical agents responsible for scaffolding, misconception, motivation, and metacognition. We compare four voting protocols -- simple, ranked, cumulative, and approval voting -- across two simulated tutoring environments on SciQ and HumanEval benchmarks. Rather than using voting as a simple aggregation step, we use it to analyze how collective decision rules shape coordination under partial pedagogical conflict. Across 1,200 simulated interactions, we find that agent deliberation and voting protocol type frequently change which response ultimately wins, showing that both meaningfully shape the collective decision. Different voting rules also produce distinct coordination behaviors, and even brief tutoring turns show measurable learning gains in simulated students. Overall, we show that protocol choice is associated with distinct coordination patterns among role-specialized pedagogical agents. 

---
# CausShield: Sample Reconstruction-Resilient Vertical FL via Causal Representation Learning 

**Authors**: Yongqi Jiang, Yansong Gao, Siguang Chen, Anmin Fu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08027)  

**Abstract**: Vertical federated learning (VFL) is a distributed learning paradigm that leverages vertically partitioned features across isolated parties without sharing raw samples; however, it remains vulnerable to active sample reconstruction attacks. Existing defenses fail to achieve a satisfactory trade-off between model utility and privacy protection, due to either suppressing task-relevant information alongside privacy-sensitive features or relying on end-to-end supervised training to converge the defense module, which exposes the model to early-epoch vulnerability. To address this challenge, we adopt a structural causal model (SCM) insight and construct CausShield. From a task-learning standpoint, causal features within a raw sample are those that are directly relevant and contributory to the learning objective, whereas non-causal features are task-irrelevant but often encode sample-specific private information, thereby facilitating reconstruction. Importantly, we lay a theoretical foundation to prove this insight. CausShield thus decomposes the shared representations between the client and the coordinating server in VFL into task-relevant and task-irrelevant components to ensure full-cycle privacy protection. Nonetheless, the decomposition is inherently challenging due to the dual objectives of preserving model utility while mitigating privacy leakage. We address this via a carefully formulated optimization problem, which is solved through unsupervised representation learning. We further theoretically prove that CausShield preserves the convergence behavior of standard VFL. Extensive experiments compare CausShield against seven SOTAs, including InvL (USENIX Security'25), and evaluate robustness against advanced reconstruction attacks such as URVFL (NDSS'25). Results demonstrate that CausShield consistently outperforms in privacy protection, model utility, and computational efficiency. 

---
# Semantic Quorum Assurance: Collective Certification for Non-Deterministic AI Infrastructure 

**Authors**: Jun He, Deying Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08021)  

**Abstract**: As large language model (LLM) agents are integrated into autonomous cloud operations, distributed systems face a semantic reliability problem: proposer agents can generate production mutations, such as modifying IAM policies, opening firewall security groups, or executing data exports, that are syntactically valid and statically authorized but operationally unsafe. Classical distributed consensus protocols replicate deterministic state transitions but do not evaluate the safety of the proposed intent. To address this gap, we introduce Semantic Quorum Assurance (SQA), a control-plane primitive for governing non-deterministic agentic infrastructure. SQA represents proposals as declarative execution contracts bound to cryptographic evidence chains and routes them to a diverse panel of read-only, sandboxed validator agents. SQA aggregates their judgments under a risk-adaptive quorum predicate that enforces model and archetype diversity, adjusts weights based on calibrated assurance scores, and respects archetype-specific vetoes. Admitted proposals execute only through a sovereign execution gate. We instantiate SQA in a cloud-native control plane and formalize a correlated cognitive failure model for non-deterministic validators. On 500 infrastructure-inspired mutation scenarios, with safety results reported on held-out safe/unsafe trials excluding ambiguous scenarios, SQA reduces unsafe approval from 18.5% for single-agent validation to 0.3% while adding median validation latency of 1.45--4.12 seconds across the studied risk buckets. 

---
# Repair Before Veto, When Repair Is Hidden: Quantum-Accessible Features for Repair-Augmented Constraint Learning 

**Authors**: Yifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08020)  

**Abstract**: Hard-constraint decision systems usually veto infeasible candidates. This is too rigid when the system can act: if a known affordable repair would make an infeasible candidate feasible and valuable, rejection is a false veto rather than a ranking error. We introduce Q-RACL (Quantum Repair-Augmented Constraint Learning), a repair-before-veto framework that first defines RACL decision semantics and then identifies the single inference link where quantum feature access can be load-bearing. RACL accepts a candidate when a sequential repair plan restores feasibility and preference; otherwise it returns structured rejection credit. The hard link is repair-feasibility inference: which repair class restores feasibility from an observed candidate and context. We construct a discrete-logarithm-hidden RACL family where the repair class is a shifted interval rule in the latent exponent a = log_g(x), while the learner observes only x = g^a mod p. Under standard DLP-based learning separation, this coordinate is inaccessible to efficient raw-input classical policies but accessible to a quantum agent through Shor/Fourier structure. Across six primes and ten seeds, bounded raw-input classical policies and a wrong raw-Fourier encoding remain near chance, whereas the Q-DLP policy keeps false-veto rate below 1.1%, wins all paired seeds, and yields QNI_cond = 0.9777 to 0.9972. A classical DLog oracle matches it, isolating feature access rather than classifier capacity. Thus quantum AI is not added as a generic model upgrade; for this DLP-hidden repair family, it supplies the missing feature that closes the repair-before-veto loop. 

---
# IEA: Amateur-Friendly Conversational Image Editing Agent via Three Stages of Multitask Alignment 

**Authors**: Zichen Zhu, Yuheng Sun, Mingxuan Zhu, Wenjie Ma, Situo Zhang, Zhexiang Wang, Ziyue Yang, Danyang Zhang, Kunyao Lan, Zihan Zhao, Dingye Liu, Siqi Xiang, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.08016)  

**Abstract**: Current image editing software often hinges on fixed filters or expert tuning, leaving a gap between amateur users' intent and outcomes. Creations by generative models may contain artifacts, implausible details, or stylistic drift away from photorealism and offer little insight into why an edit was made. We propose IEA, a conversational Image Editing Agent that learns to operate parameterized tools in an explicit, interpretable action space. IEA is trained via a three-stage multitask pipeline: (1) SFT on distilled expert edits, (2) GRPO with rewards for likeness improvement, tool usefulness, and intent summarization, and (3) large-scale synthetic fine-tuning to jointly master image editing, refinement, and user intent summarization. By manipulating 16 editing tools step by step, IEA produces transparent edit traces that can be inspected and debugged. In quantitative experiments, it attains a lower pixel distance on the edit task and a higher ROUGE-L on the summary task than strong baselines. In user studies, it ranks best among tool-calling methods for instruction following while surpassing generative methods in overall perceptual quality. Our results validate interpretable, tool-centric VLMs as a reliable path to human instruction-guided image retouching. 

---
# GVC-Seg: Training-Free 3D Instance Segmentation via Geometric Visual Correspondence 

**Authors**: Liang Xu, Fangjing Wang, Jinyu Yang, Feng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.08014)  

**Abstract**: Accurate 3D instance segmentation in point cloud data is critical for machine vision applications. Recent advancements leverage multiple pre-trained foundation models to generate 3D proposals, followed by the application of proposal aggregation methods, which significantly enhance performance. However, they often produce sub-optimal results due to inherent variations in confidence levels across different segmentation models, resulting in a bias toward the model with higher confidence. This bias is inherently model-dependent and is influenced by factors such as data preprocessing techniques and training strategies. To address this bias, we propose a novel, training-free 3D instance segmentation approach via Geometric Visual Correspondence (GVC-Seg), which exploits the correspondence between 3D geometric cues and 2D visual cues to mitigate the confidence bias. Additionally, a 3D proposal generation module and a mask-aware CLIP feature extraction module are introduced during the instance mask generation and instance semantic reasoning, respectively. In this way, GVC-Seg enhances proposal quality assessment, ensuring unbiased ensemble learning across different models. Extensive experiments demonstrate that our method achieves state-of-the-art performance on several challenging benchmarks, while also exhibiting strong potential in open-vocabulary semantic segmentation settings. 

---
# Rewrite to Translate, Translate to Reward: Reinforcement Learning for Source Rewriting in Machine Translation 

**Authors**: Boxuan Lyu, Haiyue Song, Zhi Qu, Hidetaka Kamigaito, Kotaro Funakoshi, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2606.08011)  

**Abstract**: Although directly prompting off-the-shelf Large Language Models (LLMs) to generate meaning-preserving source rewrites can effectively enhance Machine Translation (MT) quality, doing so requires manually tuning prompts for different MT models. In this work, we propose RLSR (Reinforcement Learning for Source Rewriting), a novel RL-based framework for training a source rewriting model without tuning prompts for each MT model. RLSR optimizes the rewriting model by directly using the improvement in downstream translation quality yielded by each rewritten source as the reward. Extensive experiments across six MT models and 16 language pairs demonstrate that our 4B rewriting models trained via RLSR significantly outperform the no-rewriting baseline and existing same-scale prompt-based rewriting baselines, while achieving competitive performance against prompt-based baselines based on the 235B LLM. 

---
# Summarization is Not Dead Yet 

**Authors**: Dongqi Liu, Chenxi Whitehouse, Zheng Zhao, Zhuchen Cao, Jian Li, Yabiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.08000)  

**Abstract**: The progress of large language models (LLMs) has fueled claims that model-generated summaries rival or even surpass human-written references, raising questions about whether summarization remains an open research problem. We re-examine this narrative through a multi-track evaluation covering five diverse datasets and five state-of-the-art LLMs, combining controlled human assessment, bias-mitigated LLM-as-Judge protocols, factuality verification against external knowledge, and corpus-level linguistic analysis. Our findings reveal a more nuanced landscape in which human reference summaries continue to demonstrate advantages in informativeness and faithfulness, whereas LLM outputs are preferred mainly for surface-level coherence and fluency. Factuality verification indicates that human references remain more reliable, particularly for claims involving reasoning or synthesis, and linguistic analysis uncovers a pattern of stylistic homogeneity across different models. These observations suggest that current LLMs have raised the floor of summarization quality, but the ceiling of their performance remains below human capabilities. 

---
# Enhancing AI Interpretability and Safety through Localised Architectures 

**Authors**: Ian Seet, Jonas Bozenhard, Simon Osterman  

**Link**: [PDF](https://arxiv.org/pdf/2606.07998)  

**Abstract**: Recent advances in generative AI, especially powerful Large Language Models (LLMs) and Large Reasoning Models (LRMs), raise concerns over the interpretability, safety and sustainability of these large and opaque AI models. The power of such architectures is derived not only from the scalability of deep neural networks, but also massively parallel hardware such as GPU clusters. The diffuse nature of deep neural networks gives them great function-approximation capability when provided with sufficient training data but imposes a cost in interpretability and computational efficiency. Observing that localised machine learning (ML) models tend to be more interpretable and computationally efficient than deep neural networks on small datasets, we reason by analogy that similar advantages may apply to specific localised hardware ML architectures. We argue that localised architectures with lower bandwidth but higher expressivity per node have the potential to be fundamentally more interpretable than deep neural networks running on GPU clusters while remaining competitive for smaller datasets. We then evaluate the suitability of various hardware ML paradigms for implementing such localised architectures and evaluate their per-node expressivity, energy efficiency and practical maturity of the technology required. 

---
# MC-PDD: Masked Corpus-Level Pretraining Data Detection for Black-Box Large Language Models 

**Authors**: Kaixin Lan, Mu You, Tao Fang, Binkai Ou, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2606.07996)  

**Abstract**: Pretraining is fundamental to the development of Large Language Models (LLMs), yet the opacity of pretraining data complicates model analysis and raises ethical, legal, and fairness concerns. Detecting whether specific datasets were used during pretraining is, therefore, critical. Existing state-of-the-art methods typically rely on access to model probability distributions, making them unsuitable for closed-source LLMs that provide only input-output interfaces. To address this limitation, we introduce Masked Corpus-level Pretraining Data Detection (MC-PDD), a novel method inspired by the masked language modeling paradigm. MC-PDD masks highly specific tokens in each text and prompts the LLM to predict the missing content. It then assesses whether the difference in prediction hit rates between a candidate corpus and a reference non-member corpus is statistically significant. Based on this comparison, MC-PDD determines whether the candidate texts were likely included in the model's pretraining data. Experimental results demonstrate clear and consistent differences in prediction hit rates between pretrained and unseen data across three datasets, for both open-source and closed-source LLMs. Despite operating under a stricter black-box setting, MC-PDD achieves performance comparable to existing detection methods. Our approach enables practical applications such as model auditing and data copyright verification using only standard API access. Upon acceptance, we will publicly release the code and datasets. 

---
# PRISM: PRior-guided Imagination Sampling in world Models 

**Authors**: Yuhai Wang, Jiawei Xia, Rongxuan Zhou, Xiao Hu, Yongliang Shi, Jing Du, Yang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2606.07974)  

**Abstract**: A learned world model provides a powerful physical intuition for evaluating future states. But its effectiveness in continuous control also depends critically on how candidate actions are generated for model-based planning. Rather than solely asking how accurately a model can simulate the future, we ask: which candidate actions are worth evaluating in the first place? Existing planners typically search arbitrarily or use expert demonstrations only to initialize a sampling mean, discarding the expert's state-conditioned confidence. Properly guiding this search requires a robust action prior, yet current approaches often rely on independent visual encoders or large-scale VLMs to obtain one. We argue that this architectural bloat is unnecessary: the exact same data - and the learned representations of the world model itself - inherently encode the agent's action intuition. We introduce PRISM, a task-agnostic framework that extracts both from a single dataset while maintaining strict architectural simplicity. Building on a standard JEPA-style latent world model, PRISM attaches a lightweight MLP directly to its frozen encoder to predict a state-conditioned Gaussian prior. At plan time, PRISM fuses this prior into the planner's sampling distribution via a precision-weighted Product-of-Gaussians update. This parameter-free, closed-form integration steers the sampling process, making the prior confident where it is and ceding control where it is not. PRISM improves success rates by 35 percentage points over vanilla world-model-based MPC on Cube and 32 percentage points on PushT, without introducing significant inference overhead. 

---
# Defending Against Malicious Finetuning by Scaling Train-time Adversarial Attacks 

**Authors**: Haoming Wen, Shi Chen, Qingyu Shi, Siyuan Liu, Minrui Luo, Jingzhao Zhang, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2606.07970)  

**Abstract**: Current open-weight large language models (LLMs) are prone to malicious finetuning attacks, which could compromise the safety alignment of LLMs with only a few steps of supervised finetuning (SFT) on poisoned datasets. Existing alignment-stage defenses are primarily designed to defend against attacks that use parameter-efficient finetuning methods. However, they fail to defend against stronger attacks that use full-parameter finetuning. In this paper, we propose Patcher, a method inspired by adversarial training and bi-level optimization, to combat such attacks. Patcher strengthens the simulated attack by scaling up the optimization steps in the adversarial loop, thus forcing the defender to find model parameters that are insensitive to stronger attacks. Furthermore, we propose an efficient parallel algorithm to implement Patcher, decreasing the wall-clock time of training while preserving Patcher's performance. Extensive experiments show that Patcher substantially improves the model's robustness compared to vanilla SFT alignment, and transfers to diverse attack scenarios and model sizes. Code is available at this https URL. 

---
# Neutrality Bites: Gender Representation in AI-Generated Animal Stories 

**Authors**: Imani Finkley, Yuanxi Li, Melanie Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2606.07969)  

**Abstract**: Gender bias in AI-generated stories is a well-documented problem. While much attention has been paid to reducing or mitigating this bias, it is not always clear whether interventions produce genuinely fairer results. To investigate this issue, we examine how large language models (LLMs) handle gender assignment in a narrative context that is popular, highly ambiguous, and also known to closely reproduce human stereotypes: stories about talking animals. We prompt six leading LLMs to complete an English-language story about seven different anthropomorphic animal characters whose gender is unstated. We additionally iterate with four different narrative settings and a range of model temperatures. Across the 23.8K stories, we find that models frequently avoid gendering the animal character in the story (19% on average) or use gender-neutral language like "it" or "its" (38.2% on average). However, when gender is assigned, there is a significant masculine bias. Feminine animal characters are virtually absent, present in just 2.2% of stories vs. 40.6% that feature masculine characters. Our findings point to a broader argument: neutrality bites. In other words, models that prioritize neutrality to address social bias may actually contribute to the erasure of marginalized perspectives and identities. We suggest that alternative strategies beyond neutrality need to be pursued, such as ones that more equally distribute social possibilities across imagined subjects. 

---
# RecurGuard: Runtime Monitoring for Reasoning-Token Consumption Attacks 

**Authors**: Abid Aziz, Hafsa Binte Kibria  

**Link**: [PDF](https://arxiv.org/pdf/2606.07968)  

**Abstract**: Reasoning-capable large language models can be induced to spend their generation budget on injected decoy tasks rather than answering the user's question, causing denial of service when no final answer is produced and denial of wallet when excess output tokens are billed. Input-side safety classifiers often miss these attacks because the injected prompts can appear syntactically benign. We build RecurGuard, a runtime monitor for detecting reasoning-chain consumption attacks when reasoning traces are exposed by the model. RecurGuard analyzes reasoning traces as they are generated and tracks three signals: recurrence rate, volume growth, and progress toward the user's query. If all three signals remain anomalous over three consecutive chunks, RecurGuard terminates generation early. We evaluate RecurGuard against OverThink and ExtendAttack across open-weight reasoning models and conduct adaptive stress tests on DS-R1-Qwen-7B. On this model, RecurGuard detects 99% of OverThink attacks and 92% of ExtendAttack instances while maintaining near-zero false positive rates on question answering, code generation, mathematics, and summarization. Adaptive evaluation reveals the limit of the defense: topical attacks retain 11.9x amplification with an approximately 50% joint miss rate, whereas full semantic evasion reduces amplification from 22.8x to 2.2x. When reasoning traces are unavailable, QDM provides a post-hoc fallback monitor based on the final output. 

---
# Minibatch Selection via Partition Matroid Constrained Gradient Matching 

**Authors**: Prayas Agrawal, Prateek Chanda, Ishita Khatri, Ganesh Ramakrishnan, Bamdev Mishra, Pratik Jawanpuria  

**Link**: [PDF](https://arxiv.org/pdf/2606.07954)  

**Abstract**: Training large language models (LLMs) on heterogeneous data requires selecting minibatches that balance convergence speed with coverage across domains. Existing methods either select samples independently within each domain or rely on computationally expensive proxy models to learn continuous domain weights. We propose PartitionSel, a cross-domain minibatch selection approach that maximizes a validation-guided gradient-matching utility under per-domain budgets encoded as a partition-matroid constraint. By coupling the per-domain budgets through a single utility, PartitionSel is designed to reduce redundancy in selections across domains. The proposed objective is weakly submodular and admits an orthogonal matching pursuit algorithm with provable approximation guarantees. Empirically, we evaluate PartitionSel for minibatch selection during the fine-tuning of Qwen2.5 and Llama-3 on MetaMathQA and Mol-Instructions. PartitionSel achieves robust gains over per-domain and domain-agnostic baselines on both benchmarks. It also reduces the number of conflicting gradient pairs within each batch, indicating that the cross-domain coupling translates into more compatible training updates. 

---
# From `May' to `Is': Certainty Distortion in Language Model Rewriting 

**Authors**: Catarina G Belem, Shang Wu, Hongyu Yao, Mark Steyvers, Sameer Singh, Padhraic Smyth  

**Link**: [PDF](https://arxiv.org/pdf/2606.07951)  

**Abstract**: Humans increasingly turn to Language Models (LMs) in ways that shape beliefs and drive decisions, including discussing, rewriting, and summarizing information from scientific articles, news, and medical reports. However, in these domains, where how confidently a claim is expressed matters, little is known about whether LMs faithfully preserve it. In this work, we investigate certainty distortion in LMs, defined as meaningful changes in expressed certainty when semantic content is preserved. We propose an LM-based evaluation metric that is consistent with population-level judgments of certainty. Using this metric, we characterize certainty distortion across different sizes and families of models in the context of scientific and medical communication tasks. Our results show that certainty distortion affects up to 75\% of LM outputs and is systematically asymmetric in rewriting tasks with most LMs being 1.5-2$\times$ more likely to increase the expressed certainty than to decrease it. These effects can compound over repeated paraphrasing: in the medical domain, claude-haiku-4-5 increases certainty of 20\% examples after a single iteration, increasing to 40\% after five iterations. Prompt-based interventions reduce overall certainty distortion but do not eliminate it. Together, these findings reveal a general bias toward inflating expressed certainty, with direct implications for users who rely on LMs in high-stakes domains. 

---
# POISE: Position-Aware Undetectable Skill Injection on LLM Agents 

**Authors**: Haochang Hao, Dehai Min, Zhifang Zhang, Yunbei Zhang, Miao Xu, Yingqiang Ge, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.07943)  

**Abstract**: Agent skills provide a lightweight mechanism for extending general-purpose agents, but their open format exposes them to skill-poisoning attacks. A practically dangerous injection must stay invisible: if executing the payload derails the user's legitimate task, the resulting failure signal invites inspection of the skill. We therefore evaluate attacks by Attack Success Rate, which requires the injected payload to execute and the user's task to still pass its verifier in the same trial. Prior skill-poisoning attacks face a reliability-stealth trade-off under this lens: YAML-header injections are reliably loaded but easily inspected, whereas stealthier body injections that place explicit malicious commands in the skill prose are less reliable because out-of-context commands invite the agent's own suspicion. We introduce POISE, a position-aware attack that compresses the trigger into a single, benign-looking body instruction, placing it at a feasible position and using a context-aware generator to blend it with nearby setup or prerequisite steps. On Skill-Inject with codex+gpt-5.2, POISE achieves an 89.3% ASR, 28.0 points above a random-placement body baseline and 2.6 points above a YAML-only baseline, while retaining the stealth advantage of body placement. That stealth is the decisive margin: because legitimate skill bodies naturally require privileged tool operations, LLM scanners are hyper-sensitive, falsely flagging 74.6% of clean skills on average across four judges and both benchmarks. Blending into these false alarms, POISE causes only 5.6% of poisoned variants to gain a new high-risk alert over their clean baselines, rendering current static defenses ineffective. 

---
# Illusions of the Gold Standard: A Large-scale Analysis of Human Evaluation Protocols for Long-form Text Generation 

**Authors**: Katelyn Xiaoying Mei, Yi-Li Hsu, Minjoon Choi, Zongwan Cao, Chenjun Xu, Bingbing Wen, Su Lin Blodgett, Lucy Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07936)  

**Abstract**: Human evaluation plays a critical role in assessing the quality of generated text. However, the reliability and reproducibility of these evaluations depend on transparent and well-documented protocols -- details that are frequently missing in current practice. In this work, we conduct a large-scale analysis of human evaluation protocols for evaluating long-form generation tasks in *CL conference publications from 2023--2025, including a full manual review of 284 papers and LLM-assisted analysis for another 1.8k+ papers. We define a set of 20 reportable criteria related to reproducibility of human evaluation studies, and apply these criteria to systematically examine reporting norms and practices within the community. We find widespread under-reporting of important aspects of human evaluation study design, leading to ambiguity about what was measured and how, who contributed judgments, and how judgments should be interpreted. Based on these findings, we outline actionable recommendations to support more transparent and reproducible reporting in future research. Our analysis code and annotated dataset can be found at: this https URL 

---
# Decoupling Semantics and Logic: A Training-Free Coarse-to-Fine Pipeline for Video Retrieval-Augmented Generation 

**Authors**: Jiaxin Dai, Zehang Wei, Jiamin Yan, Xiang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07924)  

**Abstract**: This paper presents our system description for the 2nd Workshop on Multimodal Augmented Generation via MultimodAl Retrieval (MAGMaR). Addressing the critical challenges of cross-lingual long-video comprehension, strict persona adherence, and zero-hallucination temporal grounding, we propose a fully training-free, two-stage cascaded Video RAG pipeline. Our architecture strategically decouples semantic retrieval from cognitive logical reasoning through a modality-aware division of labor. In the first stage, a high-recall semantic pre-fetching module employs dense retrieval using only high-fidelity visual summaries and global text descriptions, explicitly isolating noisy modalities (e.g., OCR and ASR) to maintain a pristine vector space. In the second stage, an Adaptive, Iterative, and Reasoning-based (A.I.R.) filtering agent, powered by a commercial Large Language Model (LLM), performs fine-grained cognitive reranking. The agent re-incorporates full multimodal contexts to enforce strict logical alignment with user personas, effectively pruning semantically similar but logically irrelevant candidates. Finally, a Prompt Sculpting mechanism constrains the generator to synthesize the distilled subset into strictly formatted JSON responses with exact chunk-level citations. Evaluated on the RAG track, our resource-aware approach shows exceptional precision in both information retrieval and persona-conditioned generation. 

---
# Larch: Learned Query Optimization for Semantic Predicates 

**Authors**: Fuheng Zhao, Pawel Liskowski, Zihan Li, Benjamin Han, Puxuan Yu, Varich Boonsanong, Dimitris Tsirogiannis, Anupam Datta  

**Link**: [PDF](https://arxiv.org/pdf/2606.07923)  

**Abstract**: With the advent of Large Language Models (LLMs), many database systems introduced semantic operators that enabled analytical queries over unstructured data (e.g. text, images, videos). Semantic operators typically incur high inference costs and latencies making semantic (AI) SQL queries challenging to apply on large scale datasets. At the same time, their semantic nature leads database engines to treat them as black boxes, making AISQL queries difficult to optimize. In this paper, we introduce Larch, a framework for optimizing the execution of semantic filters in AI SQL queries. Larch was inspired by two key observations: i) the high latency of semantic operators leaves significant room for computationally-heavy runtime optimization techniques, ii) unstructured data are typically accompanied by semantic information in the form of embeddings allowing for efficient semantic comparisons between AI_FILTER prompts and data values. Based on these two key observations, we present two Larch variants: Larch-A2C and Larch-Sel. Larch-A2C encodes arbitrary semantic filters expression tree using an embedding-augmented Gated Graph Neural Network and formulates the filter evaluation order as a Markov decision process. In contrast, Larch-Sel leverages a supervised learning model to predict filter selectivities, subsequently applying dynamic programming to find a near-optimal evaluation order for each input row. Evaluated across diverse real-world datasets and comprehensive synthetic workloads, both Larch variants always outperform existing semantic filter optimization techniques in terms of token usage. Our results demonstrate that Larch is robust across diverse workloads, reducing total token cost overhead by 3x-19x compared to Palimpzest and Quest. 

---
# 3D Oral Modelling with Improved Vertex Distribution Using Matching-Based Learning 

**Authors**: Jihun Cho, Soo-Yeon Jeong, Eun-Jeong Bae, Sun-Young Ihm  

**Link**: [PDF](https://arxiv.org/pdf/2606.07907)  

**Abstract**: In our previous work, a deep learning-based framework for 3D intraoral reconstruction was proposed. The model directly predicts explicit 3D point cloud coordinates from ten fixed-angle intraoral images, employing MobileNetV2 and Multi-head Attention for multi-view feature fusion, with a combined L1 Loss and Chamfer Distance as the loss function. Although the model achieved an accuracy of 77.49%, predicted vertices tended to concentrate in high-density regions of the ground truth, leaving other regions largely uncovered.
In this paper, an improved loss function is proposed to address this limitation. Hungarian matching with filtering and Repulsion Loss are introduced to enforce more uniform vertex distribution across the reconstructed model. The proposed model achieves an accuracy of 68.02%, which is numerically lower than the previous model. However, the vertex clustering issue observed in the prior work is substantially alleviated, with predicted vertices distributed more evenly across the entire reconstructed surface. 

---
# Strained Coherence: A Pre-Failure Signal in Coding Agent Execution Trajectories 

**Authors**: Marut Pandya, Kasey Zhang, Baiqing Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07889)  

**Abstract**: LLM-based coding agents sometimes acknowledge a problem in their own reasoning and then proceed anyway. We call this pattern strained coherence: a safety-relevant failure mode in which an agent has information that should change its behavior, states that information, and still acts against it. The pattern overlaps with verbalized reward hacking, where an agent names a tension between a task proxy and the underlying goal yet optimizes the proxy anyway. We give an operational definition, build a Claude Sonnet 4.6 judge that reads full trajectories and flags spans where the pattern occurs, and evaluate it on 44 Terminal-bench-2 trajectories using a Qwen3.5-35B-A3B backbone. Flagged trajectories fail 94% of the time versus 46% for unflagged trajectories (47-point gap, Fisher's exact p = 0.003; 46 points after excluding three prompt-embedded examples, p = 0.006). At matched selectivity, the detector reaches 94% precision versus 88% for a lexical discourse-marker baseline; the 10-trajectory intersection of the two methods has a 100% failure rate (Clopper-Pearson 95% CI [69%, 100%]). We replicate on Gemma4-31B with 43 trajectories: the overall signal is directionally consistent but not significant (20-point gap, p = 0.31), with attenuation driven largely by 13 trajectories with zero think content, where the detector has no substrate to analyze. In the high-verbosity Gemma tertile, the gap is +30 points; in the mid- and high-verbosity Qwen tertiles, it is +40 points each. The first flag appears at a median of 83-84% of elapsed trajectory time across both models, and the binary flag survives paraphrases that soften explicit conflict markers (8/8 trajectories). Unlike univariate predictors, the detector emits interpretable span-level output -- quoted acknowledgment, quoted action, and typed conflict -- showing what the agent saw and ignored. 

---
# The Cross-Architecture Substrate: A Domain-Transcendent, Calibration-Surviving Geometric Invariant of Modern Vision Encoders 

**Authors**: Yousef Radwan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07882)  

**Abstract**: Different vision neural networks -- trained to classify, contrast, reconstruct, or match images to text -- should have correspondingly different internal representations. We report that they do not. After training, the top sixteen principal directions of variation inside thirteen modern vision encoders converge to the same sixteen-dimensional geometric object. We call this the cross-architecture substrate and study it with PCA, centred kernel alignment (CKA), and Pang 2026 calibration. The substrate transports across four visual domains (natural photographs, medical CT, satellite, microscopy) at median Procrustes-CKA 0.679, and across eight domains (adding sketches, depth, thermal infrared, astronomy) at 0.604, every pair >0.40. It survives Pang calibration globally (7.4x disc-vs-MAE separation, n=13,394) and locally (4.82-5.30, p<10^{-44}). It is not pixel statistics (0.263), not Gabor features (0.31), not a random projection (0.041), and emerges in the first 10% of training while accuracy keeps climbing. We deliver four applications: a label-free transferability filter beating LogME (3x faster, +0.15 Kendall-tau); a four-way domain detector (99.6% accuracy); a frozen low-shot probe (16 dims beat 768-dim DINOv2 by 3.78pp at N=50 labels per class); and a teacher-free distillation auxiliary matching trained-teacher KD on 33 pairs (7.56pp peak gain at 10% label fraction). The substrate does not cross modalities, does not help cross-paradigm distillation, and does not predict transfer quality (rho=0.08 against transfer accuracy). 

---
# Instrumented data for causal scientific machine learning 

**Authors**: Daniel N. Wilke  

**Link**: [PDF](https://arxiv.org/pdf/2606.07865)  

**Abstract**: Scientific machine learning is limited less by model size than by the data it is trained on. Observational data records what happened but not why; template synthetic data has a known generating process but only for the simulator's template, not the case a user faces. We argue a third option is now operationally feasible: instrumented data, in which every datum carries the mechanistic model that produced it, an explicit uncertainty over that model, and an executable family of counterfactuals. Verification-and-validation (V&V) instrumented image-to-simulation pipelines are one realisation: a sensor observation becomes a fully specified, solver-backed simulation with explicit, editable parameters and a propagated aleatoric/epistemic uncertainty. The substrate is case-specific, mechanistically supervised, and supports causal interventions through Pearl's do-operator. Near-term consequences for validation, auditing, and surrogate training span computational biology, climate, materials, fluid mechanics, and medical imaging; a longer-term, falsifiable implication concerns foundation models for scientific reasoning. 

---
# The Last Visible Pixel: Probing Fine-Scale Perception in Vision-Language Models 

**Authors**: Lujun Li, Lama Sleem, Niccolo Gentile, Yangjie Xu, Yewei Song, Wenbo Wu, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2606.07861)  

**Abstract**: Recent vision-language models (VLMs) excel at multimodal understanding and reasoning, yet their fine-grained visual perception remains underexplored. A natural extension of ``How many r are there in Strawberry?'' asks: how small a visual pattern can a VLM reliably perceive? As such, we introduce FineSightBench, a new benchmark that systematically probes this limit by separating perception tasks (pixel-level recognition of letters, shapes, objects) from reasoning tasks (spatial reasoning, counting, ordering over small targets) across controlled scales of 4--48px. Through comprehensive experiments and detailed failure mode analysis on state-of-the-art models, we reveal a sharp dissociation: perception saturates around 12px, while reasoning remains limited even at larger scales, with persistent numeracy and sequence errors. These findings expose fundamental deficiencies in VLMs' fine-scale visual reasoning that demand more rigorous evaluation. 

---
# Model Multiplicity for Adversarial Detection in Small Language Model Training on Edge Devices 

**Authors**: Stefan Behfar, Richard Mortier  

**Link**: [PDF](https://arxiv.org/pdf/2606.07857)  

**Abstract**: The rise of edge-based machine learning has enabled distributed adaptation of language models across mobile and IoT devices, offering privacy preservation and real-time responsiveness. However, distributed fine-tuning of language models on untrusted or heterogeneous edge nodes introduces new vulnerabilities. Compromised or unreliable devices can inject poisoned updates, leading to stealthy model manipulation or convergence degradation. Classical defenses such as robust aggregation or temporal anomaly detection operate on a single global model and are therefore limited in detecting coordinated or persistent poisoning. This work proposes a new system-level defense based on model multiplicity. Instead of maintaining one global model, the system rotates or concurrently trains multiple small language models (e.g., DistilGPT-2), each updated by independently sampled subsets of edge nodes. These models evolve under distinct training trajectories, creating multiple independent views of the same distributed population. Divergence between models quantified through gradient similarity, loss evolution, or parameter variance serves as a signal of anomalous or adversarial behavior. When one model deviates significantly from the ensemble mean, the system flags its contributing nodes for isolation or re-weighting. We implement this framework and evaluate it on edge-scale simulations of Small Language Model (SLM) training under varying heterogeneity and attack conditions. Results show that model multiplicity enables earlier and more reliable detection of poisoning compared to classical single-model defenses such as Flanders and Robust methods. Our findings demonstrate that diversity in model evolution can serve as a practical and effective defense mechanism for secure distributed learning on resource-constrained edge devices. 

---
# Beyond English benchmarks: clinical llm evaluation in Brazilian Portuguese 

**Authors**: Giordano de Pinho Souza, Glaucia Melo, Josefino Cabral Melo Lima, Daniel Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2606.07853)  

**Abstract**: Large Language Models are transforming the support for clinical decision and their application in real scenarios. Yet, most benchmarks are conducted in English, and cross-lingual evaluation is needed to tackle the language gaps in global access. We introduce ClinicalBr, the first bilingual benchmark for clinical decision built from real Brazilian case reports. The corpus contains 2,892 cases drawn from 28 SciELO medical journals, spanning 18 specialties, and is structured as parallel Portuguese-English pairs. Each case supports four evaluation tasks: diagnosis retrieval, differential diagnosis, exam recommendation, and treatment planning. We evaluate four models: MedGemma-27B, Sabiá-4, DeepSeek-R1, and o3-mini, across both languages. The central finding is that the Portuguese-English performance gap is task-dependent, not general. In diagnosis retrieval, English yields a consistent advantage across all models, with +7.5-12.1 accuracy points. This advantage disappears in differential diagnosis, exam recommendation, and treatment planning, where confidence intervals cross zero for most models and Portuguese completeness scores are marginally higher. Brazilian-endemic conditions proved easier than the full corpus, not harder, indicating that tropical presentations are adequately represented in current pre-training. Exam recommendation was the hardest task across all models and both languages, with F1 scores below 0.10, well below the differential diagnosis ceiling of 0.20-0.27. 

---
# Cost-Aware Speculative Execution for LLM-Agent Workflows: An Integrated Five-Dimension Method 

**Authors**: Faisal Fareed  

**Link**: [PDF](https://arxiv.org/pdf/2606.07846)  

**Abstract**: LLM-agent workflows chain model calls and tool invocations, and spend most of their wall-clock time waiting on upstream operations before downstream ones can start. Speculative execution can reclaim that idle time by launching a downstream operation with a predicted upstream input, but here each speculation costs real money (per-token billing) and its success probability is hard to estimate and drifts over time. This paper presents a method organized around five design decisions: (D1) start a downstream operation before its upstream completes; (D2) price each speculation in real dollars at separate input and output rates; (D3) expose a single operator dial for latency versus cost; (D4) decide via an expected-value rule with a failure-weighted cost term and a preference-adjusted threshold; and (D5) estimate the success probability with a Bayesian Beta-Binomial posterior whose prior is keyed to a dependency-type taxonomy. Variants of these ideas appear in recent work; the combination, with every decision logged in dollars, is what is new. The rule fires only on edges passing an admissibility precondition (side-effect-free, idempotent, or stageable behind a commit barrier), since a wrong speculation is rolled back by re-execution, which refunds tokens but cannot un-send an irreversible side effect. We specify the runtime mechanics, a closed-form result that the rule self-limits as the upstream branching factor grows, a five-stage calibration pipeline (offline replay, shadow, canary, online calibration, drift-triggered kill-switch), and a workload-fit rubric over eight production archetypes. Contrast tables against the four closest published systems (DSP, Speculative Actions v2, Sherlock, B-PASTE) show differentiators on every dimension, and a synthetic validation suite confirms the predicted decision boundary, probability threshold, posterior recovery, and streaming-cancellation behavior. 

---
# Does Persona Make LLMs K-pop Fans? A Pilot Study of LLM-Based Online Concert Audience Agents 

**Authors**: Kirak Kim, Hyojin Kim, Yejin Son, Sungyoung Kim, Kyung Myun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.07837)  

**Abstract**: A concert is a collective experience, but recorded performance videos are typically watched alone, stripping away the shared audience presence that makes concerts feel eventful. We investigate whether persona-based LLM audience agents can recreate aspects of this collective experience by generating real-time fan chat alongside a K-pop performance video. We present a multi-agent system in which ten LLM agents react through live-chat messages, comparing a persona-conditioned audience (each agent assigned a distinct fan identity, bias, and chat style) with a no-persona baseline. In a within-subjects pilot with K-pop fans (N=11), persona conditioning substantially improved model-level chat quality and perceived naturalness, but did not translate into differences in social connectedness, engagement, or affective response. Interviews suggest that online K-pop concert chat may operate as collective monologue rather than interpersonal dialogue, and that meaningful participation depends on shared identification with the specific artist and fandom. Persona conditioning can make LLM audiences appear more natural, but culturally meaningful collective experience may require deeper alignment between persona, crowd behavior, fandom identity, and user expectations. 

---
# Agentic multi-fidelity learning of quasiparticle and excitonic properties 

**Authors**: Arnab Neogi, Aaron Forde, Christopher A. Lane, Sergei Tretiak, Jian-Xin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07836)  

**Abstract**: Many-body GW-Bethe-Salpeter equation calculations are essential for accurate simulations of electronic structure and optical properties in modern low-dimensional nanomaterials. However, these methods are computationally demanding and can exhibit localized numerical instabilities or convergence failures that are difficult to detect within high-throughput workflows. We introduce an agent-guided multi-fidelity framework for correcting GW-Bethe-Salpeter excited-state landscapes in strained MoS2-WS2 bilayers. Across stacking registries, strain branches and reciprocal-space samplings, the workflow identifies spike-like excursions, near-zero-gap collapse and cross-fidelity inconsistencies associated with fragile long-wavelength dielectric screening. A structural agent evaluates calculations by assigning confidence weights and selectively using a small number of high-accuracy reference points. Machine learning models then transfer information across related systems and apply Gaussian process corrections to recover improved quasiparticle gaps and exciton binding energies, with calibrated uncertainty estimates. The approach corrects numerically induced artifacts without erasing physical strain dependence and substantially improves agreement with higher-fidelity references relative to a no-agent baseline. These results show that reliable surrogate learning for excited-state materials requires explicit diagnosis of numerical fragility, not direct interpolation of raw first-principles data points. The proposed framework is readily transferable to other optoelectronic nanomaterials characterized by strong quantum confinement, such as quantum dots, nanoribbons, layered two-dimensional semiconductors, and hybrid perovskite nanostructures. 

---
# Cherry-pick Override: Unsafe Directional Commitment in LLM Judges under Mixed Evidence 

**Authors**: Haoran Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07834)  

**Abstract**: LLM judges increasingly turn verdicts into system commitments. Under mixed evidence (claims with both supporting and refuting sources) this is unsafe: when the schema exposes CONFLICTING as the authorized non-directional verdict, returning SUPPORTS/REFUTES is an unauthorized directional commitment, a failure we name Cherry-pick Override (CCO). We define CCO under an explicit task contract and report it with a same-denominator diagnostic protocol paired with matched-coverage bootstrap and an apples-to-apples random-veto null. On AVeriTeC's Conflicting subset (N_C = 150), three-option judges return a directional verdict on more than 84% of mixed-evidence claims; under the typed schema, three-judge majority voting amplifies direction-on-conflict on AVeriTeC (0.887 vs. 0.840; 95% CI [+0.013, +0.080]) but does not replicate on VitaminC-Mixed. Walking an intervention ladder of common single-channel fixes (typed vocabulary, panel aggregation, confidence thresholding, validator-only filtering), each leaves a distinct residual failure: panel aggregation suppresses single-judge CONFLICTING dissent in 48% of CCO cases; the panel is well-calibrated for direction (ECE = 0.07 on pure-S/R) so confidence cannot operationally separate CCO from correct directional commits; validator-as-classifier nearly halves pure-evidence accuracy. A minimal two-channel reference probe reaches operating points neither single channel reaches; under the random-veto null its promotion to CONFLICTING is structurally targeted on AVeriTeC (empirical p < 1/2001) and weaker but in the same direction on VitaminC-Mixed, a selectivity result rather than a magnitude one. We argue for an external commitment-control layer that separates verdict generation from commitment authorization, using structural evidence and confidence as orthogonal channels and NO-COMMIT as a routed controller state. 

---
# Beyond Pass/Fail: Using Process Mining to Understand How LLMs Resist (and Fail) Red Team Attacks 

**Authors**: Zvi Topol  

**Link**: [PDF](https://arxiv.org/pdf/2606.07833)  

**Abstract**: Standard AI red teaming evaluations reduce adversarial campaigns to a single binary outcome, attack success rate (ASR), not taking into account the sequential structure of how models resist or yield to attacks. We propose applying process mining, a discipline for discovering and analyzing process models from event logs, to red teaming traces. We conduct a controlled experiment pitting 60 HarmBench prompts against two LLMs, GPT-OSS 120B and Llama 3.3 70B, using 10 prompt mutation strategies over up to 110 attempts per prompt. From the resulting 8,575 scored events we extract Directly-Follows Graphs (DFGs) and state transition matrices that reveal structurally distinct defense profiles invisible to ASR alone: GPT-OSS exhibits a near-absorbing refusal state, while Llama presents multiple porous escape routes from refusal to getting successfully jailbroken. We further show that mutator effectiveness is asymmetric across models and that time-to-jailbreak distributions differ by an order of magnitude. 

---
# Jas: AI-Paired Engineering as a Revival of N-Version Programming 

**Authors**: Jason Hickey  

**Link**: [PDF](https://arxiv.org/pdf/2606.07828)  

**Abstract**: I report a case study in AI-paired software engineering: five working ports of a vector illustration application across Rust, Swift, OCaml, Python, and browser-based platforms, built by a single developer in approximately 120 evening hours. The methodology pairs AI-assisted implementation with two safeguards -- a precise executable YAML specification serving as the single source of truth, and parallel implementations functioning as a built-in differential-testing layer. The five ports share a 23{,}000-line specification; per-port native code ranges from 0 to roughly 95{,}000 lines, reflecting the specification's escape hatch. I argue that AI-paired engineering, conditional on these two safeguards, makes feasible scope of work that conventionally requires multiple developer-years, and frame the methodology as a revival of N-version programming, a 1980s approach abandoned on cost grounds that AI changes. The paper reports concrete artifacts and honest limitations of the single-developer case study. 

---
# The ACUTE Protocol: Operationalizing Language Model Activations for Better Calibration, Utility, and Trust 

**Authors**: Nishant Subramani, Palash Goyal, Yiwen Song, Mani Malek, Yuan Xue, Tomas Pfister, Hamid Palangi  

**Link**: [PDF](https://arxiv.org/pdf/2606.07822)  

**Abstract**: As language models improve and become increasingly deployed to solve a variety of tasks, trustworthiness becomes essential. Calibration is a good proxy for trust: well-calibrated confidence estimates help inform the risk versus reward tradeoff when trusting a specific model output. Unfortunately, even as models improve, they remain poorly calibrated, often biasing towards overconfidence. Additionally, calibration can be gamed: a policy that always predicts the base rate is perfectly calibrated, but completely uninformative. To resolve this, we develop a new metric, expected utility renormalized by the oracle (EURO), that balances calibration and informativeness. We also propose a general-purpose activation-based confidence, utility, and trust estimation protocol (ACUTE) to appropriately adjudicate uncertainty. The ACUTE protocol provides flexible, sample-efficient, and compute-efficient confidence estimators for 3 tasks including multiple choice question answering, tool-calling, and scientific document summarization across 6 models from 4 model families. ACUTE outperforms strong baselines on EURO, while maintaining low calibration error. Taken together, our work shows that equipping LLMs with the ACUTE protocol can improve calibration, utility, and trustworthiness in numerous settings. 

---
# SLMJury: Can Small Language Models Judge as Well as Large Ones? 

**Authors**: Anish Laddha, Nitesh Pradhan, Gaurav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2606.07810)  

**Abstract**: Large language models (LLMs) are widely used as judges for evaluating model outputs, but their high cost, latency, and opacity limit scalability. We introduce SLMJury, a framework for evaluating small language models (SLMs) as judges across two paradigms: closed-ended binary correctness and open-ended quality scoring. We benchmark 16 SLM judges (0.6B-14B parameters) from four model families across ten benchmarks: eight closed-ended tasks spanning mathematical, scientific, and general reasoning (N=64,824 judgments per configuration), plus SummEval and MT-Bench for summarization and conversational scoring. We formalize judging as a budget-conditioned function and study five dimensions. Four findings emerge. (1) The overthinking effect is domain-dependent: for most judges quick 10-token verdicts match or beat extended reasoning on mathematical judging (by 2-7% where they help), while reasoning wins on general tasks by up to 23%. (2) Domain generalization separates model families, with math-to-general accuracy gaps ranging from under 10% to nearly 40%. (3) Closed-ended and open-ended judging draw on different capabilities: the best binary judge (Phi-4) drops to rank 9 on MT-Bench, while reasoning-trained models invert this ordering. (4) Under the Reflect-Critique-Refine (RCR) debate protocol, multi-agent debate degrades accuracy across all tested configurations, whereas the top judges resist six adversarial personas with <=0.55% variance. Reliable automated evaluation does not require large proprietary models, yet no single SLM dominates. The leaderboard is available at this https URL, and our framework code and pip package are publicly available at this https URL and this https URL. 

---
# Memetic Capture: A Pluralistic Policy Framework for Governing AI-Driven Cultural Disempowerment 

**Authors**: Subramanyam Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2606.07802)  

**Abstract**: Culture is the most insidious vector of gradual human disempowerment by AI: unlike economic or political displacement, cultural displacement attacks the very preferences and values through which humans recognise and resist disempowerment itself. We argue that existing AI governance frameworks suffer from a critical blind spot by treating cultural impact as secondary to economic and safety concerns. This paper develops \emph{memetic capture} as a unifying concept for AI-driven cultural disempowerment, and proposes the \textbf{Cultural Pluralistic Governance Framework (CPGF)}, a four-tier policy architecture combining quantitative cultural influence metrics, democratic value assemblies, pluralistic deployment standards, and transnational coordination mechanisms. We argue that pluralism is not merely an ethical requirement for such governance but a structural necessity: monocultural AI governance accelerates the very disempowerment it claims to prevent. We identify concrete policy levers, discuss implementation tensions, and outline a research agenda at the intersection of pluralistic alignment and cultural AI governance. 

---
# Beyond Point Estimates: Benchmarking Uncertainty Quantification Methods on the AION-1 Astronomical Foundation Model 

**Authors**: Karla Tame-Narvaez, Aleksandra Ćiprijanović, Shubhendu Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2606.07771)  

**Abstract**: Foundation models for astronomical surveys offer powerful learned representations that can be transferred to downstream regression tasks such as galaxy property estimation. However, point predictions alone are insufficient for scientific inference; reliable uncertainty quantification (UQ) is essential. We compare seven UQ methods on galaxy property regression using frozen AION-1 foundation-model embeddings, predicting redshift, stellar mass, stellar-population age, gas-phase metallicity, and specific star-formation rate, from Legacy Survey photometry/imaging and DESI spectra, with PROVABGS-derived labels. Distribution-free conformal methods achieve marginal coverage within $\sim$1\,pp of the nominal 90\% across all properties, while non-conformal baselines (Deep Ensembles, MC~Dropout) fail to calibrate reliably. Among conformal approaches, Conformalized Quantile Regression (CQR) delivers the best coverage in the bin with the poorest model predictions. More importantly, only the Locally Valid and Discriminative (LVD) framework -- particularly when operating on AION-1 embeddings -- also provides finite-sample \emph{local validity}, producing intervals that adapt to each galaxy's local prediction difficulty rather than relying on marginal guarantees alone. These results establish conformal prediction, and LVD in particular, as the preferred UQ framework for uncertainty-aware inference on foundation-model embeddings in astrophysics. 

---
# Quantum-Enhanced Similarity Measures for Polarimetric Materials Classification 

**Authors**: Sara Shojaei, Seyed Mohamad Ali Tousi, Emma Bennett, Param Sangani, Ali Shiri Sichani, Ilker Ersoy, Hadi Ali-Akbarpour, Filiz Bunyak, G. N. DeSouza  

**Link**: [PDF](https://arxiv.org/pdf/2606.07766)  

**Abstract**: We present a quantum--classical hybrid pipeline for polarimetric material classification that casts this as a point-matching problem. Voxel cubes, containing polarized light reflections, are used to train an encoder to produce 32-dimensional embeddings for the voxels of the cubes. At inference, the encoder head is discarded and the embeddings are encoded as probability amplitudes of quantum states. Next, a SWAP-test circuit estimates the fidelity between each of the 32D embeddings from the query cube and a dataset of anchor cubes. The aggregated fidelity serves as materials similarity scores, and the class of the anchor with highest aggregated fidelity is deemed as the class of the queried material. We evaluate our approach on a dataset of 23 materials ($\approx$800 samples each) derived from their Mueller matrices. The point-matching approaches from the proposed quantum SWAP-test and a classical classifier using Optimal Transport are compared. Our results demonstrate the competitive classification accuracy alongside open-set discrimination potential, establishing it as a viable path toward NISQ-based material recognition. 

---
# Multi-planar 2D-U-Net Segmentation of 3D-CT Abdominal Organs augmented by Spatial Occurrence Maps 

**Authors**: Daria Kern, Negar Chabi, Souraj Adhikary, Andre Mastmeyer  

**Link**: [PDF](https://arxiv.org/pdf/2606.07717)  

**Abstract**: This work proposes a lightweight 2D-U-Net-based framework for segmenting five abdominal organs in large field-of-view 3D CT scans. The method combines coarse-to-fine segmentation, predictions from multiple anatomical planes, and additional fuzzy 3D spatial maps that provide anatomical location cues to improve segmentation accuracy. We combine multi-planar 2D-U-Net models augmented by a spatial occurrence map. The approach involves two main stages. First, the abdominal volume of interest region is detected by traversing the whole scan axially with a 2D-U-Net and determining the x-y-z-minimum and -maximum extents of the 5 abdominal organs of interest. Second, we use spatial occurrence maps to enhance our multi-planar 2D-U-net architecture inside the bounds from the former stage. The method is evaluated on 80 CT scans from various public sources. The results show Dice improvements of about 4% at maximum compared to the same model trained without spatial occurrence maps. 

---
# SHIELD-IDS: Structurally Heterogeneous Ensemble with Integrated Layered Defense for Intrusion Detection Systems 

**Authors**: Maryam Zaman, Muhammad Khuram Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2606.07716)  

**Abstract**: Adversarial attacks pose a serious and growing threat to Machine Learning (ML)-based Intrusion Detection Systems (IDS), where imperceptible perturbations to network flow features can systematically mislead classifiers into accepting malicious traffic as benign. The IDS-Anta framework partially addresses this through Z-score normalization, Singular Value Decomposition (SVD), and Multi-Armed Bandit (MAB) classifier selection with Thompson Sampling, yet its classifier pool lacks sufficient structural diversity for robust adversarial resistance. This work introduces IDS-Anta++, which incorporates XGBoost and LightGBM gradient boosting models into the ensemble and wraps the extended pool in a three-layer black-box defense: Isolation Forest anomaly screening, median feature smoothing, and six-way majority voting. Experiments conducted on CIC-IDS-2017, CEC-CIC-IDS-2018, and CIC-DDoS-2019 under both Fast Gradient Sign Method (FGSM) and Zeroth Order Optimization (ZOO) attacks confirm detection accuracy above 99% on clean data, with measurable robustness gains under adversarial conditions relative to the baseline IDS-Anta configuration. 

---
# Beyond Accuracy: Interpreting Topic Representation in Suicide Ideation Detection Models 

**Authors**: Hamideh Ghanadian, Isar Nejadgholi, Hussein Al Osman  

**Link**: [PDF](https://arxiv.org/pdf/2606.07714)  

**Abstract**: Suicide ideation detection models are typically evaluated using aggregate performance metrics, yet little is known about how they internally represent psychologically meaningful risk factors. In high-stakes mental health applications, understanding these internal representations is essential for safety, transparency, and responsible deployment. In this work, we move beyond accuracy and analyze how suicide detection models trained on original and topic-augmented datasets encode psychological risk factors in their internal representation space. Using visualization and geometric analysis, we examine the coherence and separability of topic-related features. Our results show that topic-aware augmentation increases the clarity and distinctness of underrepresented psychosocial risk factors such as immigration, family issues, and financial crisis. These findings suggest that augmentation not only improves model performance but also leads to more structured and interpretable internal representations. 

---
# Attention at the Theoretical Minimum: A Mathematics of Arrays Framework for Memory-Optimal Transformer Kernels 

**Authors**: Lenore Mullin, Gaetan Hains  

**Link**: [PDF](https://arxiv.org/pdf/2606.07713)  

**Abstract**: The attention mechanism is the dominant computational bottleneck in modern transformer-based AI. Its standard implementation incurs quadratic memory traffic in the sequence length~$n$, and DRAM accesses cost 100--1000$\times$ more energy than arithmetic operations on contemporary hardware, so any analysis focused solely on FLOP counts fundamentally mischaracterises the bottleneck.
We present a Mathematics of Arrays (MoA) reformulation of scaled dot-product attention and its numerically stable softmax, deriving a Denotational Normal Form (DNF) that eliminates all intermediate arrays -- including the implicit transposed-key buffer and every softmax temporary -- by algebraic construction rather than empirical tuning. The DNF achieves $O(n_{dk} + n{_{dv}})$ data movement versus $O(n^2 + n_{dk} + n_{dv})$ for the standard implementation, where $n$ is the sequence length, $dk$ is the key dimensionality and $dv$ the value dimensionality, and is verified numerically against PyTorch at full double-precision floating-point on concrete inputs.
Unlike hardware-specific accelerators or empirical tiling schemes such as FlashAttention, MoA simultaneously provides array fusion, shape-transformation correctness, and predictive cost models from a single algebraic framework. Memory minimality is a theorem established before any code is written. A predictive performance model projects $2$--$100\times$ speedup and $2$--$50\times$ energy reduction, with the advantage widening at exascale. The derivation establishes a formally verified pipeline from Python specification through (ONF) Operational Normal Form, and dimension-lifted hardware mapping, providing performance-portable AI kernels of direct relevance to DARPA edge-deployment and DOE exascale priorities. 

---
# MatMind: A Structure-Activity Knowledge-Driven Generative Foundation Model for Materials Science 

**Authors**: Zhan'ao Yao, Boxuan Zhang, Jingyuan Shu, Xiaoyu Wu, Rongyan Wang, Linjing Li, Dajun Zeng, Yudong Yao, Tingwei Chen, Youwei Wang, Xiaolin Zhao, Jiahui Shi, Jianjun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07712)  

**Abstract**: Progress in AI-driven crystal materials science has so far been carried by narrow architectures purpose-built for individual tasks -- graph neural networks for property prediction, diffusion and flow-matching models for crystal generation -- each excelling within its niche yet unable to act as a shared backbone across the full spectrum of materials problems. Generative large language models offer a fundamentally different paradigm, in which structural representation, quantitative prediction, and structure-activity reasoning can be unified within one model, but the materials community has yet to see this paradigm realized at a level competitive with established narrow specialists. Here we present MatMind, a generative foundation model purpose-built for crystal materials science under this paradigm, developed through the coordinated activation of structure-activity knowledge and physics-informed feedback within a progressive training framework -- combining structure-activity knowledge injection, a dual-head architecture that jointly trains language reasoning and numerical regression in a shared representation space, and multi-objective physics-informed reinforcement learning over stability, novelty, and structural diversity. Across three task families, MatMind attains the lowest mean absolute error on energy above hull, bulk modulus, and band gap -- surpassing graph neural network predictors purpose-built for these tasks -- reaches an S.U.N. rate of 65.3% on unconditional crystal generation, and achieves a comparable multiplicative improvement on magnetization-density-conditioned generation, where only 21 positive samples exist within over 600000 training entries. By matching or surpassing narrow specialists on their own ground while operating within a single unified model, MatMind shows that the LLM-based paradigm can serve as a viable backbone for crystal materials science going forward. 

---
# Rosetta Memory: Adaptive Memory for Cross-LLM Agents 

**Authors**: Hao Yang, Shiqi Shen, Haoxuan Li, Zhipeng Wang, Zhi Gong, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.07711)  

**Abstract**: Memory is the key component for transforming a stateless LLM into a persistent, evolving agent through experience accumulation, long-horizon planning, and continual self-improvement. Existing memory systems typically take the LLM as the center and design memory operations tailored to a specific backbone. In practice, however, users frequently switch between LLMs, for example using Claude for coding and GPT for writing across tasks, or routing different steps to different backbones within a single task for cost-effective trade-offs. As a result, memory written by one model often needs to be consumed by another. Making upstream memory effectively adapt to and activate downstream LLMs remains a critical yet underexplored problem. To bridge this gap, we shift the perspective from LLM-centric memory design to \emph{memory-centric LLM adaptation}. Specifically, we approach the above upstream-downstream memory adaptation problem from both the write and read sides, and design two profile-conditioned operators that are jointly trained to optimize how memory is stored and presented for better task completion. To ensure the learned operators generalize across a broad set of LLMs, we propose a minimum-gain sampling curriculum that prioritizes the least-served LLMs during training. To better measure the operators' actual contribution rather than the LLM's own capability, we design a performance-gap reward that compares against a naive memory baseline. Experiments on HotpotQA, 2WikiMultihopQA, and MuSiQue demonstrate that our model consistently outperforms baselines and remains robust under unseen-model replacement. 

---
# WhiFlash: Accelerating Speculative Decoding with Token-Level Cross-Paradigm Routing 

**Authors**: Young D. Kwon, Miles Williams, Rui Li, Alexandros Kouris, Stylianos I. Venieris  

**Link**: [PDF](https://arxiv.org/pdf/2606.07710)  

**Abstract**: The autoregressive nature of large language models (LLMs) remains a significant bottleneck for inference, particularly in complex agentic workloads. While speculative decoding (SD) accelerates inference, current approaches rely on static drafting paradigms, utilising either autoregressive drafting models for reasoning or diffusion-based parallel drafting models for structured outputs. We empirically find that drafting accuracy fluctuates dramatically within a single sequence, leaving significant performance unrealised by static paradigms and coarse-grained routing. To address this volatility, we introduce WhiFlash, the first cross-paradigm SD method that unifies autoregressive and diffusion-based parallel drafting under a single token-level controller. WhiFlash adopts a fine-grained routing mechanism that employs either a lightweight entropy-based or a learned neural policy, both parametrised to provide a tunable balance between expected token gain and latency. To make high-frequency switching computationally viable, we introduce novel cache-management optimisations, Lazy Catch-up and KV-only Prefill, reducing switching overhead to below 7% of per-round latency. By capitalising on the complementary strengths of fundamentally distinct drafting architectures, WhiFlash achieves significantly higher acceptance lengths, yielding category-specific throughput gains of up to 69.6% over the state-of-the-art autoregressive EAGLE-3 and 37.3% over the diffusion-based DFlash. 

---
# Cross-View Urban Traffic Dataset: Drone-Supervised Ground Truth for Monocular Bird's-Eye View Localization 

**Authors**: Prakhar Bhardwaj, Simone Weikl, Kilian Mang, Elia Jonas Sandtner  

**Link**: [PDF](https://arxiv.org/pdf/2606.07708)  

**Abstract**: We introduce a dataset and benchmark for cross-view urban traffic perception built from synchronized ego-centric bicycle videos and aerial drone videos recorded at real urban intersections. The benchmark targets two linked tasks: cross-view identity matching between street-view and drone-view object tracks, and ego-to-bird's-eye-view prediction using aerial supervision. In contrast to prior urban driving and V2X datasets, our benchmark provides identity-level alignment across radically different viewpoints together with standardized evaluation, annotation tooling, and baseline implementations. This setting is motivated by intersection-centric traffic analysis, where identity preservation, local interactions, and global spatial structure must be reasoned about jointly across views. We evaluate methods at both the track and frame levels, including cross-view ID precision/recall/IDF1, near--far breakdowns, temporal stability, and consistency metrics. We also provide baseline results for wedge-based cross-view matching and for three BEV prediction baselines: inverse perspective mapping, a MonoLayout-style learned baseline, and a regression baseline. The results show that the benchmark is feasible but challenging: cross-view matching achieves strong recall yet remains limited by over-assignment and temporal inconsistency, while ego-to-BEV prediction benefits from aerial supervision but remains far from saturated under lightweight monocular sensing. We hope that this benchmark will support future research on cross-view perception, urban scene alignment, and ego-to-global traffic understanding. 

---
# MLingualFC: Evaluating Jailbreak Vulnerabilities in Multilingual Vision-Language Models 

**Authors**: Rishabh Makwana, Mamta, Deeksha Varshney, Oana Cocarascu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07706)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated strong performance across multimodal tasks, yet their safety robustness remains an open challenge. While prior work has shown that structured visual prompts such as flowcharts can effectively jailbreak VLMs, existing studies are largely limited to English-centric settings. In this paper, we introduce MLingualFC, a multilingual multimodal benchmark designed to evaluate jailbreak vulnerabilities of VLMs across diverse languages using structured flowchart representations. MLingualFC encodes harmful instructions into flowchart images across five languages (Hindi, Punjabi, Spanish, Romanian, and German). We evaluate state-of-the-art multilingual VLMs, including Qwen2.5-VL, Gemma-4, and Pangea, under a black-box threat model. Our results reveal significant multilingual safety gaps. Flowchart-based attacks achieve high attack success rates (ASR) in case of Latin script languages, demonstrating that visual encoding of harmful content effectively bypasses safety alignment across languages. In contrast, non-Latin script languages such as Punjabi exhibit substantially lower ASR, suggesting potential limitations in visual text recognition rather than stronger safety alignment. These findings highlight that current VLM safety mechanisms fail to generalize across languages and modalities. Resources are available at this https URL 

---
# SAW: Stage-Aware Dynamic Weighting for Multi-Objective Reinforcement Learning in Large Language Models 

**Authors**: Yuchen He, Baolong Bi, Shenghua Liu, Huaming Liao, Yuyao Ge, Bolin Wan, Siqian Tong, Juan Chen, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.07705)  

**Abstract**: Although multi-objective reinforcement learning (MORL) is central to aligning large language models with complex human preferences, the prevailing practice of static weighted summation overlooks a more fundamental phenomenon: reward learning is markedly asynchronous across objectives. Well-learned dimensions quickly produce homogeneous, low-variance signals whose residual noise contaminates the aggregated reward (in GRPO) or occupies a fixed share of the advantage budget (in GDPO), interfering with the scarce yet high-value signals carried by under-learned dimensions. To address this asynchrony, we propose Stage-Aware Dynamic Weighting (SAW), a lightweight, algorithm-agnostic dynamic weighting mechanism. SAW utilizes the coefficient of variation (CV) as a scale-invariant proxy for real-time informativeness, reweighting each dimension's reward or advantage contribution by its relative informativeness within the batch. Unlike gradient-based methods that require multiple forward and backward passes, SAW relies solely on batch-level statistics, introducing nearly negligible computational overhead. Experiments on tool-calling and text summarization tasks demonstrate that SAW consistently improves both training efficiency and final performance under both GRPO and GDPO frameworks, confirming it as a general-purpose plug-in for multi-reward LLM alignment. Our code is available at this https URL 

---
# FunctionEvolve: Structure-Guided Symbolic Regression with LLMs 

**Authors**: Zeyu Xia, Jun Zhu, Dong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07704)  

**Abstract**: Symbolic regression aims to uncover explicit scientific laws from data. Recent methods use LLMs to guide mutation from background text, which is more directed than random genetic programming. However, exact symbolic recovery requires both semantic guidance and explicit structure, so that domain-informed search are carried out through valid symbolic representation. Current LLM-driven systems remain structure-blind: they select among opaque candidates, lack explicit mechanisms for local mutation, and rely on brittle coefficient fitting that can undervalue correct skeletons. We propose FunctionEvolve, an evolutionary framework using expression trees to organize the whole search: structural summaries promote diverse parent selection, local tree edits preserve useful subexpressions, and structure-aware fitting decomposes, constrains, and simplifies coefficients for more reliable scoring. It uses only elementary function families, without additional domain-specific rules limiting generalization. On the 129-task synthetic subset of LLM-SRBench, FunctionEvolve with \emph{Claude Opus 4.6} recovers 107 exact forms, reaching 82.9% SA@50, 4.5x above same-backbone baselines, and 55.8% SA@1, 3.6x above the strongest previously published top-1 result. Ablations show that structure-visible search is central to reliable recovery, with LLM-guided refinements and structure-aware coefficient optimization serving as essential proposal and scoring mechanisms. We also audit the benchmark and show that collinearity in its materials-science subset creates identifiability issues. 

---
# How Much Dense Attention is Necessary? Oracle-Guided Sparse Prefill for Full/GQA Layers in Hybrid Long-Context Models 

**Authors**: Hongxing Wang, Harenome Razanajato, Zhen Zhang, Yujie Yuan, Hongsheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07703)  

**Abstract**: Long-context prefill remains expensive because full/GQA layers still score the historical sequence, even in hybrid models with local, sparse, linear, or recurrent components. We study how much dense attention is needed to preserve task-level behavior under explicit support granularity and top-k budgets.
We introduce an attention-mass top-k oracle for existing GQA checkpoints: for each layer and query position, it computes dense attention, selects head-averaged token support, and recomputes attention only on that support. The oracle is a diagnostic reference, not a deployable accelerator, and separates sparse-budget feasibility from indexer error and runtime realization effects.
On Qwen-family retrieval-heavy evaluations, the longest per-query oracle rows stay within 1 point of dense, and a Qwen3.5-9B RULER-style sweep from 4K to 100K stays within 0.48 points. Guided by the oracle, we derive a head-collapsed auxiliary indexer trained by KL distillation from dense attention-mass distributions while keeping the backbone frozen. With separately distilled Qwen3.5-0.8B and Qwen3.5-9B indexers, the reported 16K/32K validation macro gaps are +2.04 and +1.13 points, treated as quality preservation rather than improvement; fused selection-block-shared support can introduce a larger realization gap.
Preliminary single-card TTFT measurements show distilled-indexer sparse serving speedups of 1.71x for Qwen3.5-0.8B on NPU and 1.93x for Qwen3.5-9B on GPU against its dense FlashAttention-2 baseline. Additional random-init stress rows reach 3.44x, indicating sparse-runtime headroom but not validated output quality. This first release separates oracle feasibility, distilled-indexer quality, and runtime headroom, leaving a fully matched quality-latency frontier to future work. 

---
# EvoCSFL: Surrogate-Assisted Evolutionary Client Selection for Efficient and Robust Federated Learning 

**Authors**: Lin Qiang, Sun Xiaoyan, Hu Yao, Fang Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.07702)  

**Abstract**: The heterogeneity of client data and systems makes it difficult to achieve satisfactory convergence speed and robustness in federated learning with random client selection. To address this issue, this paper proposes a surrogate-assisted client evolutionary selection framework for federated learning. In this framework, some typical client selection strategies are first used to generate candidate sets, and a metric function that integrates model performance, communication latency, and energy consumption is developed to formulate the client selection problem as a combinatorial optimization one. Subsequently, a surrogate model is constructed using the candidate selections and metric to efficiently approximate the performance of selected client subsets. An evolutionary algorithm is employed to search the combinatorial space of client selections, guided by the surrogate model to accelerate convergence. Experiments on MNIST, CIFAR10, CINIC10, and TinyImageNet demonstrate that the proposed algorithm achieves faster convergence, lower energy consumption, and improved robustness compared to existing methods. 

---
# EssentialGIN: a new approach for gene essentiality prediction based on graph isomorphism neural networks 

**Authors**: Sahar Mansouri-Rad, Zahra Narimani, Parvin Razzaghi, Nazanin Hosseinkhan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07700)  

**Abstract**: Background: Prediction of essential genes (proteins), is a basic and challenging problem but at the same time very costly and time-consuming in wet-lab experiments. Predicting essential genes, only based on computational methods (to introduce wet-lab candidates) using centrality measures are not accurate and result in large number of false positives; therefore, more complex models such as deep learning and also integration of biological information are used in recent research to identify essential genes.
Methods: In this work we focus on graph isomorphism networks, in order to embed proteins as a node in PPI network to conserve topological features of PPI network, and also integrate biological data such as gene expression data, gene orthology information and gene subcellular localization information, and introduced a deep architecture for predicting essential genes. Graph isomorphism network architecture is modified in this work for embedding node information.
Results: Our experiments proved that the proposed method outperforms baseline centrality-based methods and also machine learning based methods such as Node2Vec, MLP, and also graph attention networks (GAT).
Conclusion: In this paper we observed that using graph isomorphism networks that integrate biological data (as node attributes) and preserve network topology can significantly improve the essential gene prediction accuracy. In simpler organisms such as E. coli and D. melanogaster, methods such as multi-layer perceptron using Node2Vec embedding also performs very good, but in H. sapiens the introduced architecture significantly outperforms deep learning and other graph neural network solutions.
Keywords: Essential gene prediction, graph neural network, graph isomorphism network, PPI network, node embedding 

---
# Pharmacogenomic Knowledge Graph Augmentation for Graph Neural Network-Based Drug-Drug Interaction Prediction 

**Authors**: Juergen Dietrich  

**Link**: [PDF](https://arxiv.org/pdf/2606.07698)  

**Abstract**: Graph neural networks (GNNs) applied to drug-drug interaction (DDI) prediction rely exclusively on molecular structure encoded as SMILES-derived graphs. Prior work in this series demonstrated that model performance is bounded by the structural information content of training labels -- an Information Ceiling -- that architectural refinements alone cannot overcome. The present study investigates whether pharmacogenomic prior knowledge from the PharmGKB database partially closes this ceiling by providing metabolic pathway context that is independent of, and complementary to, molecular structure. Cytochrome P450 (CYP) enzyme substrate, inhibitor, and inducer annotations for four clinically relevant isoforms (CYP2D6, CYP3A4, CYP2C19, CYP2C9) are extracted and incorporated as a 12-dimensional feature vector concatenated to the molecular embedding prior to interaction prediction. Experiments are conducted under both pair-level and drug-level data splits to quantify generalization to unseen drugs. Results indicate that knowledge graph (KG) augmentation substantially improves DDI type classification under pair-level split conditions (F1-macro: 0.532 vs. 0.241 baseline), while binary interaction detection and drug-level generalization remain bounded by the Information Ceiling (AUC inflation: 0.224 vs. 0.250 baseline). Mechanistic validation on strictly held-out compounds confirms that augmentation preferentially improves CYP2C9-mediated interaction prediction, with probabilities increasing from 0.033-0.117 (baseline) to 0.560-0.586 (KG-augmented). An extension to single-molecule toxicity prediction on the Tox21 benchmark confirms that the effect is contingent on pharmacogenomic annotation coverage. These findings motivate the multimodal framework proposed for the subsequent study in this series. 

---
# TianJi-Environ: An Autonomous AI Scientist for Atmospheric Environmental Research 

**Authors**: Haoluo Zhao, Hongchun Zhang, Nan Li, Jing-Jia Luo, Kaikai Zhang, Mengyang Yu, Nan Chen, Tao Song, Fan Meng  

**Link**: [PDF](https://arxiv.org/pdf/2606.07697)  

**Abstract**: As atmospheric environmental prediction continues to improve, interpretable validation of pollution mechanisms and feedback processes has become a main challenge in atmospheric chemistry. Yet mechanism validation based on complex numerical models still relies heavily on expert knowledge: mechanistic hypotheses must be operationalized into executable experiments, and model outputs must be organized into traceable evidence. We present TianJi-Environ, an auditable AI Scientist for atmospheric-chemistry mechanism validation. TianJi-Environ establishes the first WRF-Chem-based multi-agent framework that autonomously drives complex atmospheric-chemistry simulations, converting mechanistic hypotheses into executable configurations, testing experiments, and evidence criteria. Using ozone response and particulate-matter feedback as two representative examples, we demonstrate TianJi-Environ's capability for mechanism validation. In a summertime ozone case over the North China Plain, the system detects directionally consistent aerosol-radiation-interaction signals in shortwave radiation and boundary-layer height, but judges the evidence for ozone response to NOx control to be incomplete. In a wintertime PM2.5 case over the Guanzhong Basin, it localizes the unsupported link to insufficient propagation from black-carbon perturbation to particulate response and missing diagnostics of vertical absorptive heating. These results show that TianJi-Environ makes expert-driven mechanism validation explicit, structured, and auditable, offering a reproducible paradigm for multi-agent systems coupled with complex atmospheric-chemistry models. 

---
# Adversarial Robustness of Activation Steering in Large Language Models 

**Authors**: Kien Le, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2606.07696)  

**Abstract**: Activation steering has become a popular training-free method to control LLM behavior by injecting precomputed direction vectors into the model's residual stream at inference time. Yet its robustness to realistic input variation remains unstudied. We present the first systematic evaluation of activation steering robustness under adversarial text perturbations on the inputs, covering four extraction methods, three attack strategies, six personas from Anthropic Model-Written Evaluation Dataset, and five models ranging from 1.5B to 30B parameters. Attacks succeed broadly across all settings: directional robustness drops by up to 64%, post-attack confidence collapses near or below 0.25 across all methods and models, and steering strength degrades on nearly every steerable input. Layer selection is equally fragile, with the optimal layer identified by an automated method on clean inputs shifting by up to 17 positions under perturbation, a failure that compounds the vector-level breakdown. Extracting vectors from adversarially perturbed inputs partially recovers steerability for PCA and MD on mid-to-large models, but they consistently fail to locate the improved optimal layer, limiting the practical benefit of this mitigation. Together, these findings reveal that the brittleness of activation steering is structural rather than method-specific, and that current layer selection strategies are not robust enough for real-world deployment. 

---
# DSFNet: Learning Dual-Domain Spectral Operators for Multi-Modality Spatio-Temporal Forecasting in Urban Transportation Systems 

**Authors**: Yongchao Li, Yang Li, Zhuoxuan Li, Jun Chen, Chu Zhang, Jinde Cao, Leszek Rutkowski  

**Link**: [PDF](https://arxiv.org/pdf/2606.07695)  

**Abstract**: Multi-Modality Spatio-Temporal Forecasting (MoSTF) extends traditional spatio-temporal forecasting by incorporating diverse traffic modalities. Despite significant recent strides in spatio-temporal modeling, existing approaches often fail to explicitly model the coupling relationships between different modality variables. Accurate MoSTF is challenging, as it requires modeling (1) temporal dynamic heterogeneity under exogenous influences and (2) heterogeneous spatial dependencies alongside complex cross-variable couplings. To address these challenges, we propose the Dual-Domain Spectral Filtering Network (DSFNet). Our framework employs dual-domain spectral filtering to capture heterogeneous spatial patterns and explicitly model the relationships between variables. Unlike graph-based message passing or dense attention over node-modality pairs, DSFNet factorizes space-modality interactions into feature-domain and spatial-domain spectral operators, enabling scalable modeling of nonlocal dependencies and cross-modality couplings. Furthermore, we introduce an external gating mechanism to adaptively regulate temporal dynamics under external influences. We validate our method through extensive experiments on five representative real-world traffic datasets. Compared with the second-best baselines, DSFNet reduces MAE by 3.21%-10.16% across these datasets. The results demonstrate that DSFNet significantly outperforms existing state-of-the-art baselines in accuracy while exhibiting efficiency and robustness. 

---
# BCG-FM: A Foundation Model for Ambient Cardiac Health Sensing 

**Authors**: Magnus Ruud Kjaer, Haejun Han, Ashish Neupane, David Q. Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.07692)  

**Abstract**: Foundation models for wearable biosignals have matched or exceeded supervised specialists across a range of clinical tasks, yet all rely on modalities that require deliberate user action--wearing a device or visiting a sleep lab. We introduce BCG-FM, the first foundation model for ambient mechanical biosignals. A piezoelectric sensor embedded in the bed surface records ballistocardiography (BCG) each night without user effort; we pretrain BCG-FM with participant-level contrastive learning and using a total of 2.75 million hours of nightly recordings from 145,985 individuals, the largest raw-waveform biosignal pretraining corpus to date. Frozen BCG-FM embeddings achieve 3.26-year MAE on biological-age estimation (the lowest reported for any ambient, contactless modality) and yield clinically relevant discrimination across 15 self-reported health conditions and three independent external cohorts. Pretrained representations from only 500 labeled participants outperform a fully supervised baseline trained on 3,372, and representation quality scales log-linearly with contrastive batch size. These results establish ambient, longitudinal mechanical biosignals as a viable modality for health foundation models. 

---
# HARP: Efficient Data Selection for Finetuning Large Language Models 

**Authors**: Ning Wang, Zhengxin Zhang, Maosen Tang, Yitang Gao, Claire Cardie, Sainyam Galhotra  

**Link**: [PDF](https://arxiv.org/pdf/2606.07690)  

**Abstract**: Finetuning data selection requires balancing two competing goals: selecting examples that improve the downstream objective, and doing so without repeatedly finetuning models. Train-free selectors are scalable but rely on proxies such as embedding similarity or clustering, which may not match the target objective. Train-based selectors better reflect downstream utility through gradient signals, subset evaluation, or Shapley attribution, but require many costly train--evaluate iterations. We propose Hierarchical Active Region Pruning (HARP), an efficient train-based selector that preserves downstream alignment while reducing selection cost. HARP organizes the training pool into a node--leaf hierarchy, evaluates only representative leaves, and infers unmeasured utilities with empirical Bayes posteriors. It then selects data using two complementary envelopes: HARP-C, which conservatively controls redundancy, and HARP-E, which additively rewards complementary regions. We theoretically show that, under local smoothness and bounded estimation error, HARP controls selection error while reducing train--evaluate cost. We further validate that HARP variants achieve the best result and outperform the strongest baseline by up to $+8.9$ points, while using roughly $7\times$ fewer training examples. 

---
# TRACER: Token ReAssignment for Concept ERasure in Generative Recommendation 

**Authors**: Ziheng Chen, Jiali Cheng, Zezhong Fan, Hadi Amiri, Diyuan Wu, Gabriele Tolomei, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07688)  

**Abstract**: Generative recommendation formulates next-item prediction as autoregressive generation over semantic ID (SID) sequences derived from users' historical interactions, making modern recommender systems structurally similar to large language models (LLMs). As privacy and safety concerns grow, these systems increasingly require concept unlearning to remove sensitive or harmful concepts associated with items. However, existing LLM unlearning methods cannot be directly applied to generative recommendation. Unlike word tokens with explicit semantics, SIDs are abstract identifiers that are often shared by both forget and retain items, leading to severe conflicts between concept removal and recommendation utility preservation.
To address this challenge, we propose TRACER, an end-to-end concept unlearning framework based on token reassignment. Rather than directly suppressing shared SIDs, TRACER reassigns concept-related items to alternative tokens that better facilitate forgetting while minimizing side effects on retained items. We further introduce a coherence regularizer to preserve semantic consistency among retain items during unlearning. Experiments on real-world recommendation datasets demonstrate that TRACER effectively removes target concepts while substantially better preserving recommendation utility than existing unlearning baselines. 

---
# What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction 

**Authors**: Jewon Yeom, Hanseul Kim, Jeongjae Park, Sungmok Jung, Jaejin Lee, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.07687)  

**Abstract**: Video world models are increasingly used to provide predictive visual representations, yet it remains unclear which pretraining signals induce action-relevant structure in their latent spaces. We study this question through a unified probe-based evaluation across diverse encoder families, including image-only self-supervision, video pretraining with and without latent prediction, reconstruction-based autoencoders, diffusion models, and shortcut-forcing dynamics models. Using a common inverse-dynamics probing objective, we find that action-relevant structure is driven primarily by temporal video pretraining rather than pixel reconstruction fidelity: models with strong pixel decoding quality can exhibit near-zero action recoverability, while video-pretrained self-supervised encoders consistently achieve the best Pareto trade-off between visual fidelity and action prediction. Comparing V-JEPA and VideoMAE further shows that most gains arise from natural-video temporal context, with feature-level latent prediction providing a smaller additional benefit. These trends transfer across robotic benchmarks, though CALVIN reveals that static-environment tasks can partially mask the importance of temporal structure by allowing strong image priors to suffice. Finally, inverse-dynamics supervision substantially improves robustness to visual corruption, suggesting that action-aware objectives regularize latent geometry beyond clean-setting performance. Our results identify temporal predictive structure -- not reconstruction fidelity -- as the primary ingredient underlying action-relevant video representations. 

---
# Knowledge-Inclusive Adaptive Physics-Informed Neural Network for Microbial Interaction Modelling 

**Authors**: Ravisha Rupasinghe, Rajith Vidanaarachchi, Asela Hevapathige, Sachith Seneviratne, Sen-Lin Tang, Saman Halgamuge  

**Link**: [PDF](https://arxiv.org/pdf/2606.07686)  

**Abstract**: Physics-Informed Neural Network (PINN) is a way of including knowledge in the form of equations in Machine Learning methods. Beyond equations, knowledge exists in other forms, such as text and network structure. While existing PINN-based approaches discover equation parameters from data, they rely solely on experimental measurements. We propose a new PINN framework that enriches parameter discovery by incorporating auxiliary knowledge sources. We instantiate our framework for microbiology, where generalised Lotka-Volterra (gLV) serves as a biological foundation for modelling microbial communities. We demonstrate that incorporating knowledge improves microbial community modelling. Our framework enriches the gLV parameters using peer-reviewed metagenomics literature, as text provides biological context on external influences that gLV alone cannot capture. We combine this knowledge with experimental measurements of microbial abundance using a data-driven integration approach. We integrate network-based structural knowledge by explicitly modelling microbial interactions. Our knowledge-inclusive framework infers microbial networks, revealing ecological insights. We validate these findings against ecological roles documented in the literature. We evaluate on real and simulated datasets spanning human- and plant-associated microbial communities. Our framework improves over the state-of-the-art by up to 53%, even without knowledge. Knowledge addition yields gains of up to 23% in Bray-Curtis Dissimilarity-based accuracy and 47% in $\mathrm{R}^2$. 

---
# Test-Time Adaptive Composition for Machine Learning as a Service (MLaaS) in IoT Environments 

**Authors**: Deepak Kanneganti, Sajib Mistry, Sheik Mohammad Mostakim Fattah, Aneesh Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2606.07685)  

**Abstract**: The dynamic nature of Internet of Things (IoT) environments affects the long-term effectiveness of Machine Learning as a Service (MLaaS) compositions. Existing adaptive composition methods are mainly based on service replacement or re-composition, where identifying suitable substitutes is difficult and time-consuming. To address this, we propose a novel Test-Time Adaptive (TTA) composition framework for MLaaS in IoT environments. First, we introduce a TTA-aware composability model to determine whether adapted services remain compatible with the existing composition. Next, we design a service-level adaptation model to adjust individual services during inference while preserving composition performance. Experimental results demonstrate that the proposed framework reduces computational time more effectively than traditional adaptive approaches. 

---
# Semantic Cache Distillation: Efficient State Transfer via Reuse and Selective Patching 

**Authors**: Qianli Ma, Zhiqing Tang, Hanshuai Cui, Zhi Yao, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2606.07684)  

**Abstract**: Disaggregated serving alleviates memory bottlenecks in Large Language Model (LLM) inference but creates a severe communication bottleneck: transmitting high-dimensional Key-Value (KV) caches often dominates time-to-first-token (TTFT). Moreover, reusing caches across heterogeneous models (e.g., base and fine-tuned variants) causes semantic misalignment that accumulates over layers, degrading generation quality. We propose Semantic Cache Distillation (SCD), a loss-constrained framework that replaces raw KV transmission with compact semantic codes. SCD addresses these challenges via two mechanisms: (1) Reuse, which reconstructs most layers from low-rank subspaces to minimize transfer cost, and (2) Patch, which predicts normalized inputs at sparse transition layers to truncate error propagation. Empirically, SCD delivers up to 2.65 $\times$ TTFT speedup over the oracle consumer prefill and dominates quantization and selective recomputation baselines on the quality--latency Pareto frontier in bandwidth-constrained regimes, while keeping generation quality within 5\% F1 of the oracle. 

---
# SWE-Marathon: Can Agents Autonomously Complete Ultra-Long-Horizon Software Work? 

**Authors**: Rishi Desai, Jesse Hu, Joan Cabezas, Neel Harsola, Pratyush Shukla, Roey Ben Chaim, Adnan El Assadi, Omkaar Mukund Kamath, Fenil Faldu, Prannay Hebbar, Jiankai Sun, Yiyuan Li, Pramod Srinivasan, Ishan Gupta, Christopher Settles, Daniel Wang, Derek Chen, Pranav Raja, Albert Liu, Marek Šuppa, Nevasini Sasikumar, Luyang Kong, Erik Quintanilla, Xiangyi Li, Ivan Bercovich, Steven Dillmann  

**Link**: [PDF](https://arxiv.org/pdf/2606.07682)  

**Abstract**: AI agents are increasingly expected to complete long-horizon workflows that require sustained progress over hours, millions of tokens, and complex environments. Yet current agent benchmarks largely evaluate short-form tasks, such as single pull requests, small tickets, or 5-10 minute exercises, limiting our ability to measure agents' capabilities in planning, long-context understanding, and memory use. We introduce SWE-Marathon, a benchmark of 20 long-horizon tasks spanning software engineering and adjacent technical domains. Each task consists of a unique executable environment, a human-written reference solution, and a multi-layer verification suite. Logged agent attempts average 27.2M total tokens, making SWE-Marathon substantially longer-horizon than existing SWE and command-line agent benchmarks. Current frontier coding agents solve fewer than 30% of tasks. Failures often arise from poor self-verification, self-reported infeasibility, and premature termination. We also observe reward-hacking behavior in 13.8% of rollouts, where agents attempt to exploit the environment or verifier to bypass the intended workflow. SWE-Marathon includes adversarial review of test suites and execution environments, as well as multi-layer checks designed to prevent shortcut solutions. We release SWE-Marathon, evaluation code, and agent trajectories at this https URL. 

---
# Systematic LLM Translation of Legacy Scientific Code to Differentiable Frameworks: Application to a Land Surface Model 

**Authors**: Aya Lahlou, Linnia Hawkins, Pierre Gentine  

**Link**: [PDF](https://arxiv.org/pdf/2606.07681)  

**Abstract**: Differentiable programming offers transformative capabilities for scientific modeling, enabling gradient-based parameter estimation, sensitivity analysis, and data assimilation. Yet, migrating legacy codebases into differentiable frameworks remains a challenge. We present a five-phase LLM-based agentic pipeline that translates legacy Fortran into JAX: static dependency analysis determines module translation order from the full call graph; iterative compile-repair loops correct errors autonomously; and a Fortran reference oracle enforces numerical parity at the module level before integration and gradient verification. We instantiate and evaluate the pipeline on CLM-ml-v2, a 19,000-line Fortran land surface model, and analyze agent behavior across 73 module translation tasks. The resulting differentiable model computes the complete Jacobian in a single backward pass, recovers physical parameters in eight times fewer steps than gradient-free optimization, and achieves a 24 times wall-clock speedup over sequential Fortran at ensemble size N=2,048. Both the translated model and pipeline infrastructure are released as a reusable framework for differentiating other Earth system model components. 

---
# DOG-DPO:Dynamic Optimization in Geometry for Safety Alignment 

**Authors**: Yi Nian, Tiankai Yang, Yudi Zhang, Qi Pan, Zelong Xu, Shenzhe Zhu, Qingqing Luan, Yue Huang, Xiangliang Zhang, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.07678)  

**Abstract**: Safety alignment for large language models relies on preference data, but current pipelines often train on large, redundant datasets. Existing data selection methods typically score each preference pair independently, collapsing directional preference information into scalar quality or diversity scores. This sample-centric view is especially limiting in multi-dataset settings, where shared safety directions coexist with dataset-specific residual risks. We propose DOG-DPO, a training-free data selection framework that treats preference pairs as structured geometric signals. DOG-DPO first represents each preference pair as a direction in model representation space. It then decomposes multi-dataset preference geometry into a global anchor subspace and dataset-specific residual subspaces. Finally, it selects subsets by maximizing diversity-based coverage, encouraging broad, non-redundant coverage of alignment directions before DPO training. Across six safety benchmarks and two model backbones, DOG-DPO achieves a strong utility-robustness trade-off using only 11% of the preference pairs. It recovers most of the safety gains of full-data training while remaining entirely teacher-free, training-free, and substantially faster than representative selection baselines. 

---
# Single-Cell Cross-Modal Transfer by Adversarial Fine-Tuning of Foundation Models 

**Authors**: Joseph Boyd, Matthew Lyon, Martino Mansoldo, Christian Hurry, Finnian Firth  

**Link**: [PDF](https://arxiv.org/pdf/2606.07676)  

**Abstract**: Spatial transcriptomics (ST) is a powerful tool for exploring biological properties dependent on structure, proximity, and interaction in tissue. The methods underpinning ST are developing rapidly but are limited in their ability to profile many thousands of genes at a subcellular scale. Although dissociated from tissue, it is known that the whole-transcriptome readouts of cells in single-cell RNA sequencing (scRNA-seq) retain information about their former in situ neighbourhoods, motivating computational methods to recover it. While paired ST and scRNA-seq datasets are scarce, each modality in its own right is abundantly available. We therefore propose to perform cross-modal translation between unpaired ST and scRNA-seq data. In this work we show that a single-cell foundation model can perform this translation via adversarial fine-tuning. We demonstrate that our method performs favourably against methods built for multi-omics translation. 

---
# A Hierarchical Feature Engineering Framework for Automated Classification of Phonotraumatic and Non-Phonotraumatic Vocal Hyperfunction 

**Authors**: June-Woo Kim, Kangwook Jang, Minu Kim, Hyunju Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.07673)  

**Abstract**: Ambulatory neck-surface acceleration enables non-invasive monitoring of vocal hyperfunction, yet robust biomarkers for its subtypes remain limited. This study investigates the NeckVibe Challenge dataset to distinguish phonotraumatic (PVH) and non-phonotraumatic (NPVH) from healthy controls. We propose a hierarchical feature engineering framework comprising: (i) static, (ii) dynamic, (iii) ratio-based, (iv) coupling features capturing source filter interactions. While univariate statistical analysis shows strong separability for PVH but limited significance for NPVH, our machine learning pipeline, tailored for high-dimensional feature integration, identifies that coupling features are crucial for both tasks. We achieve an AUC of 0.891 for PVH and 0.728 for NPVH, suggesting that while PVH is near-linearly separable, NPVH discrimination benefits from modeling non-linear feature interactions. 

---
# Liquid Neural Networks as a Drop-in Continuous-Time Deformation Field for Dynamic 3D Gaussian Splatting 

**Authors**: Mingzhao Li, Arghya Pal, Guan Yuan Tan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07670)  

**Abstract**: Deformable 3D Gaussian Splatting (D-3DGS) re-constructs dynamic scenes from monocular video by deforming a canonical set of 3D Gaussians through a positional-encoded MLP of frame time t. Although fitted to a continuous variable, the MLP couples no two values of t in its architecture and effectively predicts discrete per-frame offsets, leaving temporal smoothness to emerge only as a byproduct of optimisation. We redesign the deformation field as a stack of Closed-form Continuous-time (CfC) cells, a Liquid Neural Network (LNN), that is the closed-form solution of the Liquid Time-constant ODE while preserving every other part of the D-3DGS pipeline. Each cell exposes a sigmoidal time gate that interpolates between two candidate hidden states, baking a learned smooth response to t into the loss landscape without invoking any numerical solver. On the eight D-NeRF and seven NeRF-DS scenes the liquid field matches or exceeds the MLP baseline in aggregate, with its largest gains concentrated on the scenes with the most high-frequency articulated motion. The result is a near-zero-friction architectural design that turns the discrete MLP deformation field into an explicit continuous-time function of t. 

---
# MemoVAD: Resource-Efficient Video Anomaly Detection via Dynamic Semantic Memory in Edge Computing Scenarios 

**Authors**: Guo Li, Jiandian Zeng, Yang Li, Zihao Peng, Ke Chen, Tian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07669)  

**Abstract**: Deploying Video Anomaly Detection (VAD) in real-world surveillance faces a fundamental tension between the demand for high-level semantics to ensure effectiveness and the limited computational resources of edge devices. Vision-Language Models (VLMs) provide rich open-vocabulary semantics, but their latency and computational cost preclude on-device deployment. To address the challenge, we propose MemoVAD, an edge-cloud collaborative framework that selectively incorporates VLM semantics into streaming VAD. MemoVAD runs most inference on the edge with a lightweight detector and a causal Temporal Context Encoder (TCE) to model temporal dependencies. Specifically, we introduce an Uncertainty-Aware Gating (UAG) policy grounded in Subjective Logic to model perceived uncertainty and query the cloud-based VLM only for high-uncertainty and semantically novel clips. Besides, a Dynamic Semantic Memory (DSM) is designed to cache VLM-verified prototypes for efficient retrieval, enabling the edge model to progressively incorporate VLM-level semantics via a semantic adapter. Experiments on UCF-Crime and XD-Violence datasets via a real edge device show that MemoVAD substantially reduces communication overhead while surpassing state-of-the-art performance. 

---
# AgentCompile: An LLM-Guided Compiler for Direct CUDA Inference 

**Authors**: Xuanzhe Li, Ziyan Weng, Zhiyu Zhu, Junhui Hou  

**Link**: [PDF](https://arxiv.org/pdf/2606.07665)  

**Abstract**: Transformer inference increasingly depends on specialized compiler and runtime support, but real model graphs still require semantic decisions about which regions are worth specializing and which CUDA implementation families are plausible. We present AgentCompile, an LLM-guided CUDA inference compiler that uses LLM outputs only as advisory search metadata. Given compiler-derived region summaries and bounded candidate spaces, the LLM proposes semantic labels, candidate priorities, parameter hints, and risk annotations; the compiler materializes CUDA candidates through templates, checks interface and hardware constraints, validates candidates empirically, selects implementations by measured latency, and falls back when specialization is unsupported or unprofitable. In end-to-end autoregressive generation, AgentCompile averages 5.66x, 4.05x, and 4.26x speedup over PyTorch eager on Qwen3-1.7B, Qwen3-4B, and Llama-3.2-1B-Instruct, respectively, across five representative workloads. We will open-source the project. 

---
# Seq103: A Unified Neuroevolution Framework for Compact Sequence Architecture Discovery 

**Authors**: Wenxiao Li, Yongjian Liu, Qing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2606.07664)  

**Abstract**: Neuroevolution is a representative neural architecture search paradigm that evolves both network topology and weights through evolutionary algorithms. In this paper, we propose Seq103, a unified NEAT-style neuroevolution framework for compact sequence architecture discovery. Seq103 consists of a shared evolutionary backbone and an optional recurrent extension. The shared backbone includes an elementary node-and-connection representation, per-class RMSE-based evaluation, mutation-based evolution with class-wise recombination, and elitism. The optional hidden-state mechanism extends the search space with hidden-state nodes and hidden connections, enabling temporal memory when step-wise recurrent inference is required. With this design, Seq103 applies the same core search pipeline to both step-wise recurrent and sample-wise feedforward sequence classification. In recurrent tasks, the hidden-state extension is enabled to provide temporal memory; in feedforward tasks, it is disabled while the shared evolutionary backbone remains unchanged. We evaluate Seq103 on 8 text classification datasets and the full UCRArchive2018 benchmark with 128 univariate time-series datasets. On step-wise tasks, Seq103 retains 86.96% of the best-baseline accuracy on average while using 34.6x to 3218.0x fewer parameters. On sample-wise tasks over the full UCRArchive2018 benchmark, Seq103 retains 81.95% of the best-baseline accuracy on average while using 11.8x to 160,601.0x fewer parameters. 

---
# MM-Matryoshka: Towards Budget-Elastic Visual Document Retrieval via a 2D Multimodal Matryoshka Training Framework 

**Authors**: Haowen Xiang, Yibo Yan, Jiahao Huo, Yu Huang, Yi Cao, Mingdong Ou, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07654)  

**Abstract**: Multi-vector visual document retrievers achieve strong fine-grained matching by representing each page with multiple vectors from deep Vision-Language Models (VLMs), but this design makes deployment expensive in both storage and computational overhead. Existing efficiency techniques usually optimize only part of this budget, leaving multimodal retrievers without a unified way to trade accuracy for both vector width and encoder depth. Therefore, we propose MM-Matryoshka, a 2D Matryoshka training framework for budget-elastic Visual Document Retrieval (VDR), enabling ColPali-style multi-vector retrieval elastic along both dimension and layer. At inference time, a single retriever can select a 2D selectable budget without training separate models for different budgets. Through comprehensive experiments across multiple representative backbones, we demonstrate that by retaining significantly higher quality than direct truncation baselines while substantially reducing storage and computational overhead, MM-Matryoshka can offer robust budget elasticity for efficient VDR. 

---
# A Dataset for Dynamic Human Preferences for Vision Language Models 

**Authors**: Hannah Gao, Dylan Hadfield-Menell, Rachel Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.07653)  

**Abstract**: Given the increased adoption of Vision Language Models (VLMs) in human-interactive settings, it is important that we evaluate how well these models can adapt to real-time preferences for different users. While an increasing number of vision-language benchmarks have recently been introduced, they focus largely on evaluating static capabilities and generally-held preferences learned from extensive training data. This work introduces a new benchmark for evaluating the ability of VLMs to understand dynamic human-preferences, i.e. preferences that are passed in-context at inference time. We provide an automated pipeline for generating this benchmark with variations on image dependence, a dynamic multi-modal human-preference dataset, and evaluations of state-of-the-art models on the novel benchmark. 

---
# ViMax: Agentic Video Generation 

**Authors**: Lingxuan Huang, Sizhe He, Hengji Zhou, Liqiang Nie, Lianghao Xia, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07649)  

**Abstract**: Long-form video generation requires systematic narrative planning and visual consistency that current short-clip methods cannot provide. Existing methods generate isolated sequences without narrative structure and lack mechanisms for maintaining character and environmental consistency across scenes. We present ViMax, an agentic video generation framework that addresses video creation through coordinated multi-agent collaboration where specialized components negotiate narrative decisions, visual continuity, and production quality. Our framework employs a hierarchical narrative engine with retrieval-augmented generation for global story coherence and a dependency-aware visual consistency mechanism that tracks character and environmental states across temporal boundaries, while VLM-guided agents continuously monitor and refine both narrative coherence and visual fidelity. The framework enables coordinated agent collaboration to generate extended narrative content. This maintains both storytelling integrity and visual coherence across multi-scene timelines. 

---
# AQIFormer: A Transformer-Based Multi-View Architecture for Cross-City Air Quality Classification 

**Authors**: Om Kathalkar, Nitin Nilesh, Sachin Chaudhari, Anoop Namboodiri  

**Link**: [PDF](https://arxiv.org/pdf/2606.07648)  

**Abstract**: Air pollution represents one of the most critical environmental and public health challenges globally, with traditional sensor-based monitoring systems facing significant scalability and economic constraints. Image-based air quality estimation has emerged as a promising alternative, leveraging the visual characteristics of atmospheric pollutants in traffic scenes. However, existing methods suffer from limited cross-city generalization and inadequate exploitation of multi-view perspectives. We present AQIFormer, a novel transformer-based ensemble architecture that addresses these fundamental limitations through innovative dual-view integration, weather-aware attention mechanisms, and comprehensive multi-task learning. Our approach uniquely combines front and rear traffic imagery with meteorological parameters to achieve robust air quality classification across diverse urban environments. Extensive evaluation on a comprehensive dataset of 26,678 synchronized front-rear image pairs demonstrates good performance with 89.96% accuracy, representing a 14.96% improvement over state-of-the-art methods. Most importantly, our model maintains exceptional cross-city generalization capabilities, achieving 81.67% accuracy on an independent dataset collected in Nagpur, India with only 8.29% performance degradation using few-shot adaptation with minimal training samples. 

---
# DOME: Learning Transferable Domain Variables from Sparse Supervision for Test-Time Adaptation 

**Authors**: Xiaoran Xu, Yifan Xu, Yupeng Wu, Xiaoshan Yang, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07646)  

**Abstract**: Test-time adaptation (TTA) aims to align a model to shifting test domains using only unlabeled streaming data. Most existing methods implicitly infer a single global domain distribution, ignoring the multidimensional and sample-specific nature of real-world domain shifts, leading to fragile adaptation. We propose DOME, an effective domain encoder that explicitly models each sample's domain in a zero-shot manner. DOME leverages vision-language pretraining to extract dense, continuous representations, parameterizes domains as distributional variables, and introduces a momentum-updated sparse domain bank for disentangled supervision. By injecting these explicit domain cues into downstream models, even a basic entropy-minimization TTA strategy achieves state-of-the-art performance across ImageNet-C, ImageNet-R, and ImageNet-Sketch, outperforming complex TTA approaches. Our results demonstrate that robust adaptation stems not from intricate adaptation algorithms, but from explicit, structured domain representation. 

---
# FineGen: A VLM-based Multi-Agent Framework for Fine-Grained Image-Text Dataset Construction 

**Authors**: Chang Kong, Yuebing Li, Peng Mo, Haigang Zhang, Qiuming Luo  

**Link**: [PDF](https://arxiv.org/pdf/2606.07645)  

**Abstract**: The scarcity of hard negative samples in current vision-language datasets significantly hinders fine-grained perception. To address this, we propose FineGen, a VLM-based Multi-Agent framework for automated dataset construction. By employing a collaborative Generation-Verification-Correction pipeline with a closed-loop feedback mechanism, FineGen ensures synthesized hard negatives are semantically valid yet strictly contradictory to visual content. Applying this to ImageNet, we construct FineGen-100K, a hierarchical dataset containing over 147,000 attribute-specific hard negatives with a rigorous 1:10 positive-to-negative ratio. Extensive evaluations confirm a 96.7% attribute validity rate. Crucially, downstream validation on the FG-OVD benchmark shows that fine-tuning on FineGen-100K yields a substantial +14.4% accuracy improvement on hard samples, significantly outperforming state-of-the-art methods. 

---
# AVI-Bench: Toward Human-like Audio-Visual Intelligence of Omni-MLLMs 

**Authors**: Yaoting Wang, Ziyi Zhang, Wenming Tu, Shaoxuan Xu, Wenjie Du, Cheng Liang, Weijun Wang, Yuanchao Li, Guangyao Li, Hao Fei, Yuanchun Li, Henghui Ding, Yunxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07643)  

**Abstract**: Recent advances in Omni-Multimodal Large Language Models (Omni-MLLMs) have enabled strong integration of vision, audio, and language. However, their audio-visual intelligence (AVI) remains insufficiently evaluated due to the lack of systematic and comprehensive benchmarks. We introduce AVI-Bench, a cognitively inspired benchmark that evaluates Omni-MLLMs across three stages, perception, understanding, and reasoning, through cross-modal tasks requiring joint audio-visual interpretation. This design enables fine-grained diagnosis of model capabilities and failure modes. To further assess robustness beyond familiar domains, we propose AVI-Bench-PriSe, an extension that probes models' primitive audio-visual sensation using unfamiliar, low-semantic stimuli, testing generalization beyond common training distributions. Extensive experiments on both open-source and closed-source models reveal substantial limitations in current Omni-MLLMs. Based on these findings, we present a four-level AVI taxonomy. Overall, AVI-Bench provides a principled evaluation framework to guide the development of more robust and generalizable AVI. Project website: this https URL 

---
# No Free Lunch for Synthetic Images under Data Scarcity Conditions 

**Authors**: Borja Arroyo Galende, Alejandro Almodóvar, Patricia A. Apellániz, Juan Parras, Silvia Uribe, Santiago Zazo  

**Link**: [PDF](https://arxiv.org/pdf/2606.07640)  

**Abstract**: This study investigates the trade-offs between fidelity, privacy, and utility in synthetic data generation under conditions of data scarcity and privacy sensitivity. We propose an evaluation framework that jointly assesses these three dimensions and apply it to three widely used generative models, VAE, GAN, and DDPM. The evaluation spans three image datasets, MNIST, OCTMNIST, and OrganAMNIST, encompassing both general-purpose and medical imaging domains. Notable differences arise between the three models in their behaviour when differential privacy mechanisms are introduced during training. GAN and DDPM demonstrate greater robustness, maintaining higher fidelity and downstream utility across a range of noise levels, while VAE degrades more rapidly as privacy constraints increase. This study highlights the importance of a multidimensional evaluation of deep generative models, also noting that their behaviour significantly differs when privacy techniques are applied. 

---
# MOSS-Video-Preview: Toward Real-Time Video Understanding via Cross-Attention 

**Authors**: Pengyu Wang, Chenkun Tan, Shaojun Zhou, Wei Huang, Qirui Zhou, Zhan Huang, Zhen Ye, Jijun Cheng, Xiaomeng Qian, Yanxin Chen, Xingyang He, Huazheng Zeng, Chenghao Wang, Pengfei Wang, Hongkai Wang, Shanqing Gao, Yixian Tian, Chenghao Liu, Xinghao Wang, Botian Jiang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07639)  

**Abstract**: Video understanding is shifting from the offline paradigm -- taking a fully recorded video as input and producing a single answer after it ends -- toward real-time interaction, in which the model perceives new frames while still replying, revises its answer as new evidence appears, and remains silent when there is nothing to say. We present MOSS-Video-Preview to validate this paradigm. Our central claim is that perception must not be blocked by generation; its natural realization is a two-channel architecture. We argue that a cross-attention backbone is better suited to real-time vision-language fusion than the prevailing decoder-only design: visual features enter through a side channel rather than joining the autoregressive sequence, so perception and generation run on separate, non-blocking pathways -- reducing the frequency of visual processing and exposing a clean channel-wise interface for independent compression. We complement this with a data synthesis pipeline that converts dense captions into real-time understanding QA whose answers are revised to match what the model has perceived so far, and we specialize an offline model on these data to elicit real-time behavior. Our model trails the strong Qwen2.5-VL-7B baseline overall -- a gap we attribute primarily to data and scale rather than the architecture -- yet attains competitive offline video and multimodal understanding, remains robust on the spatial and fine-grained temporal reasoning central to real-time use, and acquires behaviors that offline models lack: continuous perception, answer revision, and timely silence. On a single H200 with 256 frames per video, it achieves about a 5x speedup in time to first token and 2.7x higher decoding throughput, with negligible degradation in offline ability. Our study of paradigm, architecture, and data outlines a viable path toward real-time video understanding. 

---
# Anchor-Conditioned Compositional Control for Landscape Image Generation 

**Authors**: Gadha Lekshmi P, Govind Arun, Rohith Syam, Ahmed Elgammal  

**Link**: [PDF](https://arxiv.org/pdf/2606.07638)  

**Abstract**: Image generative models, though widely used as creative tools, offer limited support for the kind of compositional control that photographers and visual artists routinely exercise. This paper presents early results on an anchor conditioned finetuning framework for landscape image generation, in which a four dimensional compositional anchor vector is extracted from training images and injected into a diffusion model via a decoupled cross attention mechanism with Fourier encoding and three way classifier free guidance dropout. Quantitative evaluation against a baseline and three ablation variants shows that the proposed architecture achieves the highest horizon detection rate of 0.850 and the highest rule of thirds alignment of 0.817. A category specific ablation further demonstrates that training on compositionally homogeneous scene subsets reduces horizon deviation by up to 40 percent compared to mixed training. This establishes that compositional control precision is category dependent. 

---
# NeuroAlign: Hierarchical Multimodal Fusion of Dynamic and Structural Neuroimaging for MCI Analysis 

**Authors**: Xiongri Shen, Zhenxi Song, Jiaqi wang, Yi Zhong, Leilei Zhao, Chenqi Xu, Linling Li, Yichen Wei, Lingyan Liang, Demao Deng, Luping Song, Ping Luan, Ahmed M. Anter, Shuqiang Wang, Baiying Lei, Zhiguo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07635)  

**Abstract**: Multimodal neuroimaging fusion of functional MRI (fMRI) and diffusion tensor imaging (DTI) provides complementary information for cognitive impairment analysis, but remains challenged by heterogeneous feature spaces and misaligned representations. We propose \textit{NeuroAlign}, a hierarchical framework for structured multimodal fusion. It introduces (1) \textit{Dual-Modal Hierarchical Alignment} (DMHA), which models multi-scale dynamic connectivity and aligns dynamic-static and functional-structural embeddings; and (2) \textit{Dual-Domain Hierarchical Interaction} (DDHI), which enables fine-grained modulation and global interaction between connectivity- and region-level features. To support feature-level inspection, we design \textit{Synergistic Activation Mapping} (SAM), a gradient-free, marker-oriented attribution method for DFC, SFC, ALFF, and FA. Evaluated on GUTCM, ADNI, and OASIS under five-fold validation, NeuroAlign achieves competitive MCI/SCD detection and preliminary cross-dataset transferability. Attribution analyses reveal modality-specific and partially consistent brain patterns, providing model-derived evidence for multimodal representation analysis. 

---
# AMN: An Adaptive Multi-Scale Fusion Network with Boundary and Uncertainty Modeling for Nuclei Segmentation 

**Authors**: Spoorthi M, Suja Palaniswamy  

**Link**: [PDF](https://arxiv.org/pdf/2606.07633)  

**Abstract**: Accurate classification of nuclei subtypes in histopathology images is critical for downstream tasks including tumor grading, immune infiltrate quantification, and prognosis prediction. Existing approaches rely on either convolutional or transformer-based encoders in isolation, limiting their ability to simultaneously capture fine-grained local texture and long-range spatial context. We present AMN (Adaptive Multi-Scale Nuclei Network), a dual-encoder segmentation framework that jointly leverages a Swin Transformer and a ResNet-50 feature pyramid, fused via a learned per-channel gating mechanism that dynamically weighs each encoder's contribution at every scale. AMN is trained with a multi-objective loss combining class-weighted focal loss, boundary-aware loss with positive-pixel emphasis, and a novel uncertainty-modulated classification term that suppresses overconfident erroneous predictions. Evaluated on the CoNIC benchmark across seven nuclei classes, AMN achieves a mean Dice of 0.82 and mean F1 of 0.68, with an F1 of 0.67 on the diagnostically challenging lymphocyte class. AMN outperforms eight baseline models spanning pure-CNN, pure-transformer, and recent hybrid architectures: U-Net, ResU-Net, DeepLabV3+, SegNet, ViT-Small, HmsU-Net, ConvFormer-UNet, and BEFUnet. Cross-dataset evaluation on MoNuSeg demonstrates strong generalization without retraining and validating the domain robustness of the learned representations. 

---
# Trait-space Monitoring for Emergent Misalignment During Supervised Finetuning 

**Authors**: Huy Nghiem, Sy-Tuyen Ho, Sarah Wiegreffe, Hal Daumé III  

**Link**: [PDF](https://arxiv.org/pdf/2606.07631)  

**Abstract**: Emergent misalignment (EM) occurs when narrow finetuning causes a model to behave dangerously outside the finetuning task. Standard training signals can miss this shift, making reliable detection costly if it depends on repeated behavioral evaluation. We ask whether emergent misalignment can instead be detected from internal representations during finetuning. Using seven alignment-relevant traits encoded as linear directions in activation space, we track representational drift across training checkpoints in four open-source 7-9B LLMs. EM-relevant drift concentrates on a low-dimensional axis that explains 65.5% of the variance, revealing a geometric signature in the studied regime. A low-overhead monitor built on this drift profile detects dangerous checkpoints with 2.2% false negative rate, 2.9% false positive rate, and 0.990 AUROC on held-out perturbation types, outperforming unsupervised PCA and SAE baselines. Stress tests on two 14B models, longer finetuning runs, and misaligned starting points identify key deployment boundaries. These results position trait-space monitoring as a practical complement to behavioral evaluation for EM detection during LoRA-based finetuning, while showing that deployment across substantially different regimes may require recalibration. 

---
# Active Learning with Foundation Model Priors: Efficient Learning under Class Imbalance 

**Authors**: Jiancheng Zhang, Meiqing Li, Qi Zhang, Yinglun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07630)  

**Abstract**: Real-world datasets across image and text domains are often characterized by skewed class distributions and noisy annotations, which jointly degrade model performance, particularly on minority classes. Among existing solutions, active learning offers an effective and efficient paradigm by selectively querying the most informative and balanced samples for annotation. We propose an innovative active learning framework that mitigates class imbalance and selects the most informative samples to annotate. Leveraging foundation model priors, our algorithm enables imbalance-aware co-decisions between foundation model and small model to tackle noisy and imbalanced labels across various domains. We introduce the first study to systematically explore active learning under the dual challenges of label noise and class imbalance across image and text domains. Extensive experiments on imbalanced datasets demonstrate that our method achieves substantial annotation savings-over 50% compared to the best active learning baseline-while preserving performance and robustness to label noise. 

---
# Large Language Models Should Learn Personalized Rather Than Aggregated Human Preferences 

**Authors**: Cristina Garbacea  

**Link**: [PDF](https://arxiv.org/pdf/2606.07629)  

**Abstract**: Current approaches to aligning large language models (LLMs) aggregate diverse human preferences into a single reward signal, effectively optimizing for a hypothetical ``average user'' who represents no real person particularly well. This position paper argues that LLMs should learn personalized, individual preferences rather than aggregated ones. We show that aggregation masks critical information about preference diversity, individual values, and contextual dependencies, which is a limitation both theoretically grounded in social choice theory and empirically evident across demographic groups. We analyze the rich structure that human preferences encode, survey technical approaches to personalization, and systematically address counterarguments on scalability, shared standards, and manipulation risk. While personalization introduces genuine safety challenges including filter bubbles, value lock-in, and psychological manipulation, we argue these are manageable through bounded personalization frameworks that preserve universal safety constraints while accommodating legitimate individual variation. We conclude with a concrete research and policy agenda for developing preference-aware models that respect both individual autonomy and collective safety. 

---
# Eyes All Around: Design and Analysis of 360-Degree LiDAR Perception Using Equivariant Feature Learning in Unstructured Traffic 

**Authors**: Pranav Darshan, Raghuveer Narayanan Rajesh, M Uttara Kumari  

**Link**: [PDF](https://arxiv.org/pdf/2606.07626)  

**Abstract**: Perception in dense, unstructured urban traffic remains a major challenge for autonomous driving because of the wide variety of road users, frequent occlusions, irregular motion patterns, and the lack of standardized road layouts. Although recent LiDAR based 3D object detectors have shown strong performance in structured driving scenarios, most are developed and evaluated for limited field of view settings, and their behavior under full surround 360-degree sensing is still not well understood. This paper studies a 360-degree LiDAR perception pipeline for autonomous driving, with particular attention to panoramic sensing, azimuthal sector wise spatial processing, and transformation equivariant feature extraction in complex urban scenes. The paper presents a practical 360-degree perception framework that combines sector wise panoramic processing with rotation equivariant sparse convolutions and evaluates its behavior on a custom Ouster OS0 LiDAR dataset collected across diverse Indian urban traffic conditions. The results show generally stable detection across several object classes, with the strongest performance for cars at 92.02/90.51, buses at 80.53/76.34, and trucks at 78.59/74.16, while lower scores for pedestrians at 67.45/61.02, cyclists at 73.21/69.54, and motorcyclists at 71.20/68.13 reflect the greater difficulty of detecting smaller and more variable road users in dense urban scenes. 

---
# HASA: Subnet Allocation for Compute-Constrained Model-Heterogeneous Federated Learning 

**Authors**: Amir Hossein Shahdadian, Ahmed M. Abdelmoniem, Mahdi Taheri, Samira Nazari, Christian Herglotz  

**Link**: [PDF](https://arxiv.org/pdf/2606.07621)  

**Abstract**: Edge services increasingly use federated learning to personalize on-device models while keeping sensitive data local. In practice, deployments must handle heterogeneity in both client resources and local data distributions. Model-heterogeneous federated learning lowers client cost by allowing each client to train a subnet of a shared supernet, but most subnet-allocation policies are driven by device constraints and do not explicitly account for statistical heterogeneity. This paper proposes Heterogeneity-Aware Subnet Allocation (HASA), a train-only rule that assigns subnet widths based on client heterogeneity scores computed from local training data while enforcing a fixed size-weighted compute budget. This design enables budget-matched comparisons with alternative allocation policies. On an article-title next-word prediction benchmark with seven clients, HASA improves unweighted mean client test accuracy over uniform allocation across 10 matched seeds, increasing mean client test accuracy from 13.82 percent to 14.32 percent, and improves worst-client accuracy on average. In a matched-budget comparison with representative partial-training baselines, HASA achieves the strongest worst-client and tail-client accuracy on this benchmark. A directionality ablation shows that assigning smaller subnets to more heterogeneous clients degrades both mean and tail performance. A cross-domain image-classification study further shows that the effectiveness of heterogeneity-aware allocation depends on how well the heterogeneity score reflects clients' need for additional model width. 

---
# SENTRY: Statistical Reliability Analysis of Vision Transformers Under Soft Errors 

**Authors**: Pramit Kumar Bhaduri, Mahdi Taheri, Samira Nazari, Maksim Jenihhin, Christian Herglotz, Michael Hubner  

**Link**: [PDF](https://arxiv.org/pdf/2606.07620)  

**Abstract**: With the growth of Vision Transformers in safety-critical domains like autonomous systems and medical imaging, ensuring their reliability against soft errors is paramount. While ViTs offer state-of-the-art accuracy, their massive parameter counts render exhaustive fault injection campaigns infeasible. To bridge this gap, a statistical fault injection framework is presented, leveraging finite-population sampling theory to provide formal reliability guarantees. It is demonstrated that failure rates are bounded within a 1% margin at 99\% confidence using only a few thousand samples, regardless of model scale. This methodology achieves up to a 10,700 times reduction in experimental cost compared to exhaustive approaches, while preserving the ability to localize vulnerabilities across architectural components. Through extensive evaluation of different architectures like ViT-Tiny and ViT-Small, a highly non-uniform reliability landscape is uncovered. It is shown that while only 3% of FP32 bit-flips result in failure, the vast majority of these events lead to catastrophic accuracy collapse. Specific vulnerabilities are localized to normalization layers and critical exponent bits within the IEEE-754 format, providing a mathematical foundation and actionable insights for the design of hardened, edge-deployed ViT architectures. 

---
# ScaleSweep: Accurate NVFP4 Post-Training Quantization of LLMs via Block Scale Initialization 

**Authors**: Li Lin, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07618)  

**Abstract**: NVFP4 is a recently introduced hardware-supported FP4 format that improves the fidelity of 4-bit quantization through fine-grained block scales. However, existing NVFP4 scale initialization methods still primarily rely on AbsMax initialization, which leaves a noticeable gap to the optimal solution. To address this, we propose ScaleSweep, a simple and efficient scale optimization method that sweeps over feasible block scale candidates and selects the candidate that minimizes a target objective. We further provide a theoretical analysis of NVFP4 quantization and derive both lower and upper bounds for the required sweep range under mean square error (MSE) and weighted mean square error (WMSE) between the original tensor and the quantized reconstructed tensor. The proposed bounds substantially reduce the sweep space while preserving the optimal candidate, enabling negligible overhead compared with the baseline quantization operators. Experiments on Llama and Qwen models demonstrate that ScaleSweep consistently improves quantization performance over existing initialization methods and further narrows the gap to full precision. In particular, under aggressive end-to-end quantization of weights, activations, KV cache, and query states, ScaleSweep preserves more than 93% of the full-precision performance. 

---
# Query Lens: Interpreting Sparse Key-Value Features with Indirect Effects 

**Authors**: Hwiyeong Lee, Ingyu Bang, Uiji Hwang, Hyelim Lim, Taeuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.07617)  

**Abstract**: While sparse autoencoders provide features more interpretable than individual neurons, reliably characterizing them remains challenging. We propose Query Lens, which extends Logit Lens to enable more comprehensive and faithful interpretations of sparse features. By jointly considering encoder-side key features and decoder-side value features, we identify both the inputs that activate a feature and the outputs it promotes. We also account for indirect, module-mediated effects that arise when the feature is processed by downstream modules, going beyond the direct effect captured by Logit Lens. In experiments, we find that Query Lens yields coherent token signatures for features that remain uninterpretable under Logit Lens. Finally, we propose the Subspace Channel Hypothesis, suggesting that downstream modules read features through layer-specific subspaces. 

---
# Item Response Scaling Laws: A Measurement Theory Approach for Efficient and Generalizable Neural Scaling Estimation 

**Authors**: Sang Truong, Yuheng Tu, Rylan Schaeffer, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2606.07616)  

**Abstract**: Scaling laws provide a fundamental framework for understanding the performance of Language Models (LMs), yet deriving them requires prohibitively expensive evaluations across thousands of checkpoints or millions of inference samples. To address this, we introduce Item Response Scaling Laws (IRSL), a unified framework that integrates Item Response Theory (IRT) within the scaling law framework. Unlike traditional approaches that treat each model-benchmark pair in isolation, IRSL disentangles latent model ability from question characteristics, factorizing the scaling law estimation for $M$ models and $N$ questions to significantly reduce parameter complexity from $O(M \times N)$ to $O(M + N)$. We instantiate IRSL with Beta-IRT, which leverages the empirical probability responses of LMs -- such as token probabilities in pre-training and pass rates in test-time sampling -- to capture richer signals than binary responses. We validate our approach across two prevalent scaling paradigms: (1) pre-training downstream scaling, using 6,612 LM checkpoints and 37,682 questions from 10 benchmarks; and (2) test-time scaling, using 12 LMs and 120 questions from 4 benchmarks with up to 2,500 samples per question. Given a one-time calibration on existing model responses, IRSL yields more reliable scaling estimates using only 50 questions per benchmark (a 99.9\% reduction), achieving comparable or superior decision accuracy to traditional approaches. Furthermore, we show that the estimated latent model abilities are generalizable, enabling accurate performance forecasting across benchmarks that share the same measurement objective. 

---
# Structured Neuron Pruning in Deep Neural Networks Using Multi-Armed Bandits 

**Authors**: Salem Ameen, Sunil Vadera  

**Link**: [PDF](https://arxiv.org/pdf/2606.07615)  

**Abstract**: Deep neural networks often contain redundant hidden units. Removing individual weights can reduce parameter count, but unstructured sparsity is not always easy to exploit in standard dense implementations. This paper develops a structured pruning framework in which complete neurons are removed using multi-armed bandit (MAB) algorithms. Each candidate neuron is treated as an arm; pulling an arm temporarily masks that neuron, measures the change in loss on a sampled mini-batch, restores the neuron, and updates an estimate of its safe-removal reward. The framework supports stochastic policies, including Epsilon-Greedy, Softmax, UCB1 and Thompson Sampling, and multiplicative-weight policies, including Hedge-style multiplicative weights and EXP3. We evaluate the method on tabular classification, tabular regression and deep neural-network benchmarks covering image, text and reasoning tasks. Statistical comparisons using the Friedman test followed by the Nemenyi post-hoc test show significant differences between methods. On tabular classification tasks, UCB1 obtains the highest mean rank among pruning policies and improves on the unpruned neural network. On regression tasks, UCB1 obtains the highest mean rank and is statistically competitive with, or superior to, several standard regression models according to R^2. On deep-learning tasks, UCB1 and Thompson Sampling obtain the strongest ranks, and several MAB policies significantly outperform the unpruned model, magnitude-based neuron pruning and greedy activation-variation pruning. The results show that MAB-based neuron pruning is an effective and computationally practical approach for structured model reduction. 

---
# Can You Trust What You See? Human and AI Detection of Synthetic Legal Evidence 

**Authors**: Jinzhe Tan, Ali Ekber Cinar, Karim Benyekhlef  

**Link**: [PDF](https://arxiv.org/pdf/2606.07613)  

**Abstract**: Visual evidence has long been treated as a reliable form of legal proof, but advances in artificial intelligence (AI) are undermining that assumption. This article asks how well humans and frontier multimodal large language models (MLLMs) can distinguish authentic evidentiary photographs from AI-generated counterparts in the object-centric scenarios typical of civil disputes. We built Synthetic Legal Evidence Detection (SLED-1400), a dataset of 200 authentic evidence images paired with 1,200 synthetic counterparts produced by six contemporary text-to-image generators across ten evidence categories. The same stimuli and response format were used in a controlled web experiment with 136 lay participants and in a standardized evaluation of four MLLMs (GPT-5.1, Gemini-3-Pro, Gemini-3-Flash, Qwen3-VL-235B). Human accuracy was 64.8% overall, and 48.5% and 51.0% on the two strongest generators (Gemini-3-Pro-Image and Flux-2-Max), indistinguishable from chance. MLLMs never misclassified an authentic image (100% specificity), but missed most synthetic outputs from the harder generators, with average MLLM detection at 5.9% on Gemini-3-Pro-Image outputs. Human and MLLM errors were largely uncorrelated, while the four MLLMs were strongly correlated with each other. Neither group is a reliable standalone authenticator. We argue that visual evidence in legal proceedings should be treated as inherently contestable, and that a workable procedural response must combine trained human review, MLLM screening, and provenance infrastructure such as C2PA Content Credentials. 

---
# Position: Anthropomorphic Misalignment Research Needs Stronger Evidence 

**Authors**: Vansh Gupta, Peter Nutter, Samuel Stante, Andreas Krause, Florian Tramèr, Lukas Fluri, Xin Chen, Anna Hedström  

**Link**: [PDF](https://arxiv.org/pdf/2606.07612)  

**Abstract**: We argue that many Anthropomorphic Misalignment Research (AMR) studies need stronger evidence to ensure that they can provide a robust foundation for critical safety decisions, such as model deployment and regulation. By evaluating failure modes across different misalignment concepts, such as deception, emergent misalignment, and sycophancy, we show how conceptual ambiguity, non-robust datasets, experimental design, and insufficient causal interventions can lead to overinterpretation of model behaviors. This position paper aims to offer guidance on evidentiary considerations that can help improve methodological rigor in AMR. To achieve this, we provide a clear call to action through a proposed framework of evidence levels and a diagnostic checklist. These shared standards will enable more productive scientific discourse and ensure that claims about AI risks rest on solid empirical foundations. 

---
# MIRAGE: Metadata-Integrated Repository Analysis and Guided Enhancement for MSR Datasets 

**Authors**: Aabia Ather, Muhammad Usayd Ather, Qurat-Ul-Ain Somroo, Muhammad Khuram Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2606.07611)  

**Abstract**: This paper proposes an improved approach to the analysis of Mining Software Repositories (MSR) datasets via metadata enrichment, FAIRness assessment, and topic-driven analysis. This research expands upon an earlier dataset directory created specifically for the analysis of MSR datasets by adding new annotations to the datasets, enriching the metadata categories, and offering more advanced filtering options. The metadata of the MSR papers presented from 2013 to 2024 has been gathered using the Semantic Scholar API. The analysis is based on Latent Dirichlet Allocation (LDA) topic modeling and statistical analysis. Dataset-level attributes were included into the expanded dataset directory, namely repository hosting site, format, accessibility, reusability, and dataset quality. The study reveals that the choice of repository hosting sites and data formats influences citation patterns and dataset usability. Furthermore, the enhanced annotation approach improves the analysis and discoverability of MSR datasets, supporting more effective reuse and evaluation of research artifacts. 

---
# LEAF: Growing Trees Without Branching for Speech-Aware Large Language Model Post-Training 

**Authors**: Argyrios Gerogiannis, Yekaterina Yegorova, Mark Hasegawa-Johnson, Venugopal V. Veeravalli  

**Link**: [PDF](https://arxiv.org/pdf/2606.07610)  

**Abstract**: State-of-the-art GRPO-style methods for speech-aware large language model post-training suffer from coarse credit assignment, broadcasting the same terminal-reward advantage to every token in a response. This ignores useful structure within rollout batches, where speech-conditioned completions often share prefixes before diverging at important decisions. We propose Low-rank Exploration with Adaptive Forking (LEAF), a retrospective tree-based RL method that recovers this structure without online branching or additional decoding. LEAF samples complete responses, selects high-surprisal boundaries, groups responses by shared prefixes, and assigns span-level advantages using descendant rewards. We theoretically justify LEAF's span-level credit assignment and boundary-selection design. Empirically, LEAF improves over GRPO across speech question answering and speech translation benchmarks under the same rollout and low-rank adaptation budget. Notably, smaller LEAF-trained models outperform current state-of-the-art, full-parameter baselines. 

---
# Subtitle-Aligned Fine-Tuning of Whisper for Swiss German ASR: Benchmark Contamination, Convention Mismatch, and an Honest Baseline at 25.6% WER (13.8% cWER) 

**Authors**: Felix Akeret  

**Link**: [PDF](https://arxiv.org/pdf/2606.07608)  

**Abstract**: We present a systematic study of fine-tuning OpenAI's Whisper large-v3 for Swiss German ASR, using 1,367 hours of broadcast speech paired with Standard German subtitles as weak supervision. Through 16 iterative training runs on an NVIDIA DGX Spark (Grace Blackwell, 128 GB unified memory, up to 1 PFLOP FP4), we compare LoRA and full fine-tuning of the 1.55B-parameter model, investigate hallucination root causes, and quantify the effect of data quality, subtitle alignment, and training strategy. Our best model achieves 25.6% measured WER on the All Swiss German Dialects Test Set (ASGDTS) in an honest evaluation on strictly disjoint data. A harmonized error analysis separating genuine errors from valid stylistic variation (tense, word order, Swiss orthography) yields a content WER (cWER) of 13.8%, counting only actual recognition failures. Bias-corrected estimation reduces this to 8.5%, suggesting the true error rate is roughly one third of measured WER.
We demonstrate that published state-of-the-art Swiss German ASR results (17.1-17.5% WER) are inflated by benchmark contamination: a vanilla Whisper model self-trained on the ASGDTS test set with zero Swiss German data achieves 13.88% WER, surpassing all published systems. Experiments with Phi-4-multimodal show an even stronger memorization effect (3.9% WER), revealing that the benchmark primarily measures convention matching rather than dialectal comprehension.
We release two models, a LoRA adapter (25.32% WER, 13.9% cWER) and a full fine-tuned model (25.60% WER, 13.8% cWER), among the few publicly available, honestly evaluated Whisper models for Swiss German, under Apache 2.0 with full reproducibility, requiring no institutional data agreements. 

---
# SRT: Super-Resolution for Time Series via Disentangled Rectified Flow 

**Authors**: Jufang Duan, Shenglong Xiao, Yuren Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07605)  

**Abstract**: Fine-grained time series data with high temporal resolution is critical for accurate analytics across a wide range of applications. However, the acquisition of such data is often limited by cost and feasibility. This problem can be tackled by reconstructing high-resolution signals from low-resolution inputs based on specific priors, known as super-resolution. While extensively studied in computer vision, directly transferring image super-resolution techniques to time series is not trivial. To address this challenge at a fundamental level, we propose Super-Resolution for Time series (SRT), a novel framework that reconstructs temporal patterns lost in low-resolution inputs via disentangled rectified flow. SRT decomposes the input into trend and seasonal components, aligns them to the target resolution using an implicit neural representation, and leverages a novel cross-resolution attention mechanism to guide the generation of high-resolution details. We further introduce SRT-large, a scaled-up version with extensive pre-training, which enables strong zero-shot super-resolution capability. Extensive experiments on nine public datasets demonstrate that SRT and SRT-large consistently outperform existing methods across multiple scale factors, showing both robust performance and the effectiveness of each component in our architecture. 

---
# Contribution Weights: A Geometrical Analysis of Self-Attention Transformers 

**Authors**: Harry Jake Cunningham, Nicola Muca Cirone  

**Link**: [PDF](https://arxiv.org/pdf/2606.07604)  

**Abstract**: Analyzing attention weights has become a standard approach for interpreting the information flow of Large Language Models (LLMs). However, this approach has significant limitations as it neglects the geometric properties of the value vectors being aggregated. To address this gap, we introduce \emph{Contribution Weights}, a projection-based metric that quantifies a token's influence by accounting for it's attention weight, value magnitude, and directional alignment with the layer output. We demonstrate that contribution weights provide a more faithful measure of token importance, consistently outperforming attention-based metrics in identifying semantically critical tokens across different decoder-only models, tasks, and datasets. Further, our metric enables novel mechanistic analysis of \emph{attention sinks}. While previous work characterized sinks as passive repositories for excess attention, we reveal they serve an active functional role, suppressing information through a convex relationship between sink rate and output norm, stabilizing representations by opposing the semantic drift of low-confidence tokens. 

---
# MetaEvo: A Meta-Optimization Framework for Experience-Driven Agent Evolution 

**Authors**: Bowen Ren, Heyan Huang, Yinghao Li, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.07603)  

**Abstract**: Large language models (LLMs) exhibit strong reasoning capabilities, yet most LLM-based agents are statically deployed and unable to improve through task interactions. Existing experience-driven methods often rely on memory or heuristics without enhancing the model's ability to learn, treating it as a passive executor and leading to early performance plateaus and limited long-term improvement. To address this issue, we propose MetaEvo, a two-stage framework for continual agent evolution that focuses on improving how the model learns from tasks experience, rather than solely on what it stores. MetaEvo first applies preference-based optimization to enhance the model's ability of principle abstraction, then enables the accumulation and reuse of these principles within a modular agent architecture. Experimental results on diverse reasoning benchmarks demonstrate that MetaEvo consistently outperforms strong baselines, maintains reliable improvement across iterations. These findings validate the effectiveness of meta-optimization in enabling agents to learn from experience and continually enhance their reasoning capabilities. 

---
# Sample-Efficient Post-Training for LEGO Spatial-Physics Reasoning 

**Authors**: Yuhuan Yuan, Zhouliang Yu, Minghao Liu, Weiyang Liu, Ge Lin Kan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07602)  

**Abstract**: LLM-based LEGO assembly generation requires both semantic grounding and physical feasibility. We identify a data-induced failure mode, PhysHack, in which the assemblies satisfy physical-validity constraints while producing structures that are geometrically misaligned, semantically inconsistent, or poorly calibrated. To address this challenge, we propose a model-based data selection approach that uses only a small fraction of the training data while improving physically grounded LEGO assembly generation. Building on the selected trajectories, we introduce PVPO, a sample-efficient reinforcement learning method that couples physical feasibility with voxel-space geometric rewards. Our results show that physical validity alone is an insufficient proxy for reliable physical reasoning: models can learn to generate valid structures without preserving semantic or geometric fidelity. Experiments across model backbones and test-time scaling settings demonstrate that PVPO improves structural and semantic alignment, physical validity, structural stability, and calibration, while reducing reliance on extensive post-hoc rejection sampling. In particular, results on calibration show that PVPO mitigates PhysHack by making test-time selection more predictive of semantic and structural quality. 

---
# LFNO: Bridging Laplace and Fourier via Transient-Steady Decomposition 

**Authors**: Jeongun Ha, Sanga Yoon, Donghun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.07601)  

**Abstract**: We introduce the Laplace-Fourier Neural Operator (LFNO), a unified framework for modeling dynamical systems across transient and steady-state regimes by integrating the spectral advantages of Laplace and Fourier Neural Operators. LFNO employs a dual-branch architecture that explicitly decomposes system dynamics into transient and steady-state components. We evaluate LFNO on nine benchmarks, including three ODE systems (Duffing, Lorenz, and Pendulum) and six PDE systems (Euler-Bernoulli beam, Heat, Reaction-diffusion, Brusselator, Burgers, and Navier-Stokes). LFNO significantly outperforms existing operators on ODE systems, where transient dynamics dominate, and consistently surpasses LNO while achieving performance competitive with FNO on PDE benchmarks. Furthermore, LFNO offers improved stability and physical interpretability through its component-wise decomposition. These results demonstrate that LFNO provides a robust and unified approach for learning complex dynamical systems across multiple temporal scales. 

---
# Reachability and asymptotics of Gaussian Transformer dynamics 

**Authors**: Albert Alcalde, Zhengping Ji, Enrique Zuazua  

**Link**: [PDF](https://arxiv.org/pdf/2606.07600)  

**Abstract**: We formulate data propagation through the Transformer, the machine learning architecture powering large language models, as a nonlinear control system on the space of probability measures. For the mean-field Transformer model with self-attention and affine feed-forward layers, we prove that Gaussian distributions remain exactly Gaussian along the induced flow. This invariance reduces the infinite-dimensional measure dynamics to a finite-dimensional bilinear control system governing the evolution of the mean and covariance, reformulates the expressive capacity of Transformers as a reachability problem for prescribed Gaussian moments, and reveals a novel connection with Riccati-type equations from classical filtering and control.
For time-varying controls, we prove exact finite-time reachability of any target Gaussian distribution whose covariance matrix has the same rank as the initial one, this rank constraint being an intrinsic invariant of the dynamics. For time-invariant parameters, we derive explicit spectral conditions leading either to asymptotic stability toward positive-definite equilibria or to finite-time blow-up of the covariance.
Numerical experiments complement the theory by showing that practical Transformers with Gaussian inputs remain close to moment-matched Gaussian distributions through early and intermediate layers, while Transformers with prescribed attention matrices reproduce the predicted covariance regimes: bounded evolution in stabilizing configurations and blow-up in destabilizing ones. 

---
# DiffoR: A Unified Continuous Generative Framework for Universal Ordinal Regression 

**Authors**: Hongxu Ma, Lin Wang, Chenghou Jin, Han Zhou, Jie Zhang, Xiaoyu Yang, Chunjie Chen, Jihong Guan, Shuigeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.07599)  

**Abstract**: Ordinal Regression (OR) aims to predict target values with inherent order, underpinning critical applications across diverse domains, from recommender systems to computer vision. Though having evolved from naive regression to discretization-based classification and generation, existing paradigms remain fundamentally constrained by quantization artifacts and the lack of global ordinal topological perception. These methods typically enforce rigid boundary delineations, failing to capture the non-stationary semantic transitions inherent to ordinal data. In this paper, we propose a novel paradigm where OR is formulated as a Continuous Generative Ordinal Regression task. Under the novel paradigm, we introduce DiffOR, a unified framework that leverages diffusion models to recover continuous ordinal values via iterative denoising, thereby enabling the dynamic learning of soft semantic transitions. To explicitly preserve ordinal topology, we devise a Dual-Decoupling Strategy: Spatially, Multi-scale Increment Aggregation decomposes targets into hierarchical continuous increments; Temporally, Dynamic Denoising Perception synchronizes denoising steps with feature frequencies, ensuring robust coarse-to-fine refinement. Theoretically, we show that the proposed method can significantly enhance both representation capability and mechanistic interpretability. Extensive experiments on 12 benchmarks across four domains validate DiffOR's consistent superiority over state-of-the-art methods, establishing a new standard that demonstrates strong potential as a general-purpose solution for universal ordinal regression. 

---
# A Topological Characterization of Graph Neural Networks via Stochastic Block Model Embeddings on the n-Sphere 

**Authors**: Gopal Anantharaman  

**Link**: [PDF](https://arxiv.org/pdf/2606.07598)  

**Abstract**: We propose a topological framework for comparing trained Graph Neural Networks (GNNs) by mapping the Stochastic Block Models (SBMs) induced on the graphon-signal space of a Message Passing Neural Network (MPNN) onto the unit $n$-sphere $\sphere^{n-1}\subset\R^n$. The construction rests on three classical pillars: the \emph{compactness} of the cut-distance graphon space $(\Wo,\cutdist)$ \citep{lovasz2006limits,lovasz2012large}, the Frieze--Kannan \emph{weak regularity lemma} together with its graphon-signal extension due to \citet{levie2023graphon}, and the Lipschitz continuity of MPNNs with respect to the cut-distance. We show that, for any prescribed tolerance $\varepsilon>0$, a trained MPNN $\Phi$ acting on a sufficiently large graph factors (up to $\varepsilon$) through a step-graphon-signal of bounded complexity, and we construct an explicit measure-preserving map $\Psi_n\colon[0,1]\to\sphere^{n-1}$ that places the SBM regions on disjoint spherical caps. This produces a problem-agnostic, low-dimensional ``fingerprint'' of a trained GNN that is amenable to visual inspection and to nearest-neighbour search across model zoos, enabling \emph{transfer-learning candidate retrieval} without retraining. We discuss the obstruction posed by concentration of measure in high dimension -- a phenomenon directly relevant to LLM-scale embeddings. We close with five concrete future research directions: hyperbolic and Grassmannian alternatives to the spherical model, Gromov--Wasserstein distances on graphon-signals as an isometry-free alternative to the $n$-sphere map, an information-geometric (Fisher) reformulation of the SBM manifold, persistent-homology fingerprints of layer-wise embedding clouds, and a spectral-distance baseline derived from the graphon eigendecomposition. 

---
# Repetition Mismatch: Why Data Mixture Experiments Don't Scale and How to Fix Them 

**Authors**: Kevin Zhou, Lisa Alazraki, Kris Cao, Marek Rei  

**Link**: [PDF](https://arxiv.org/pdf/2606.07597)  

**Abstract**: Pre-training data mixtures are commonly tuned by running small-scale experiments and extrapolating to the target training budget. When high-quality data is scarce and must be repeated, this extrapolation frequently fails, but the source of the failure has not been isolated. We show that a primary culprit is a repetition mismatch: because high-quality datasets are small, their repetition rate changes as the training budget grows, shifting the optimal mixture in ways that small-scale proxy experiments do not anticipate. A subsampling procedure that matches the target repetition rate controls for this effect. In a two-source setting combining limited high-quality data with web crawl, a single repetition-controlled experiment using only 1/16 of the target tokens recovers a mixture within 0.05 of the optimum for a 757M parameter model, compared to an error of 0.75 without repetition control. Achieving comparable accuracy without repetition control requires three to four horizons, consuming 44 to 94% of the target token budget. With three data sources, the larger mixture space requires more than a single experiment to constrain, but the approach remains effective: at the 757M scale, just two repetition-controlled horizons recover the optimal mixture, outperforming baselines that instead require the full two-source experiments to construct. Our results reveal that repetition dynamics, not scale alone, shape whether small-scale mixture experiments generalize. More broadly, they suggest that data repetition deserves treatment as a first-class variable in mixture optimization, rather than an inconvenient side effect of limited data. 

---
# VisualLeakBench: Reproducible Action-Boundary Propagation Failures in Vision-Language Agents 

**Authors**: Youting Wang, Yuan Tang, Yitian Qian, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.07595)  

**Abstract**: Vision-language agents increasingly consume screenshots, documents, and user interfaces before writing to memory, sending messages, or invoking external tools. We study a concrete failure mode in this setting: action-boundary propagation, where sensitive or unsafe visible text is copied from an image into downstream tool arguments. We present VisualLeakBench, a diversified 500-image benchmark spanning UI, chat, document, form, and dashboard scenes, and evaluate a stratified 100-image agent subset with four production VLM systems under two workflows: note capture and external handoff. At baseline, target strings are propagated into tool arguments in 78.8% of PII cases and 85.5% of rendered unsafe-text cases. Under a defensive system prompt, rendered unsafe-text propagation remains high at 52.6%, while PII tool propagation falls to 2.0%, largely by suppressing tool use rather than preserving utility. Rates are tool-surface dependent: search-like tools suppress PII propagation, but rendered unsafe text still crosses tool boundaries. We measure visual-to-tool propagation rather than downstream instruction execution. We additionally provide a labeled-target oracle upper-bound diagnostic that localizes most failures at the tool boundary while leaving response-side leakage as residual risk. 

---
# A Mechanistic Analysis of Adversarial Fine-tuning of Vision Transformers 

**Authors**: Hannah Gao, Isha Agarwal, Dylan Hadfield-Menell, Rachel Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.07593)  

**Abstract**: The widespread use of image classification models in high-risk, real-world situations necessitates making these models robust to slight disturbances or perturbations, such as blurring or sharpening, in the input images. While vision transformers (ViTs) play an integral role in many modern-day multi-modal models like Vision-Language-Models (VLMs) and Vision-Language-Action (VLA) models, they have received a lack of attention in the setting of robustness. In this work, we analyze the effects of adversarial fine-tuning, a popular method for improving model robustness to image perturbations, on a ViT's performance on perturbed and regular images through a mechanistic lens. We adversarially train a ViT on low-frequency and high-frequency image corruptions, and attempt to explain changes in downstream model performance through an examination of the model's attention mechanisms, internal representations, and knowledge evolution. Overall, our results suggest that, while fine-tuning on inputs with common corruptions improves model performance and certainty on new instances of corrupted data, these improvements do not transfer to other classes of corruptions not seen in the training. Additionally, despite observing changes in visual attention and knowledge evolution across layers, we found that adversarial training did not lead to fundamental changes in the sparse representations learned by ViTs. 

---
# ResearchClawBench: A Benchmark for End-to-End Autonomous Scientific Research 

**Authors**: Wanghan Xu, Shuo Li, Tianlin Ye, Qinglong Cao, Yixin Chen, Hengjian Gao, Yiheng Wang, Qi Li, Kun Li, Sheng Xu, Shengdu Chai, Fangchen Yu, Xiangyu Zhao, Zhangrui Zhao, Weijie Ma, Zijie Guo, Haoyu Zhou, Haoxiang Yin, Lixue Cheng, Chaofan Hu, Haoxuan Li, Lu Mi, Xuxuan Xie, Yifan Zhou, Ruizhe Chen, Zhiwang Zhou, Xingjian Guo, Yuhao Zhou, Xuming He, Shengyuan Xu, Xinyu Gu, Jiamin Wu, Mianxin Liu, Chunfeng Song, Fenghua Ling, Dongzhan Zhou, Shixiang Tang, Yuqiang Li, Mao Su, Peng Ye, Siqi Sun, Bin Wang, Xue Yang, Zhenfei Yin, Tianfan Fu, Guangtao Zhai, Wanli Ouyang, Bo Zhang, Lei Bai, Wenlong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07591)  

**Abstract**: AI coding agents are increasingly used for scientific work, but their end-to-end autonomous research capability remains difficult to verify. We present ResearchClawBench, a benchmark for evaluating autonomous scientific research across 40 tasks from 10 scientific domains. Each task is grounded in a real published paper, provides related literature and raw data, and hides the target paper during evaluation. Expert-curated multimodal rubrics decompose the target scientific artifacts into weighted criteria, enabling evaluation of target-paper-level re-discovery while leaving room for new discovery. We evaluate seven autonomous research (auto-research) agents under a unified protocol and seventeen native LLMs through the lightweight ResearchHarness. Current systems remain far from reliable re-discovery: the strongest autonomous agent, Claude Code, averages 21.5, and the strongest ResearchHarness LLM, Claude-Opus-4.7, averages 20.7, with an LLM frontier mean of only 26.5. Error analysis shows that failures concentrate in experimental protocol mismatch, evidence mismatch, and missing scientific core. ResearchClawBench provides a reproducible evaluation frontier for measuring progress toward autonomous scientific research. 

---
# SlideCheck: Guiding Self-Supervised Pretraining of Pathology Foundation Models via Dataset Distributions 

**Authors**: Mingyi He, Xinyi Guo, Xitong Ling, Weiming Chen, Jiawen Li, Lianghui Zhu, Minxi Ouyang, Mingxi Fu, Yizhi Wang, Tian Guan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07590)  

**Abstract**: Pathology foundation models are pretrained on large streams of WSI-derived patches, while supervision during data construction is often slide-level, sparse, or heterogeneous. This mismatch makes it difficult to understand and control which biological patterns enter the pretraining data. We propose SlideCheck, a lightweight pretraining data guidance tool built on frozen pathology foundation model patch features. Rather than serving as a standalone patch diagnostic model, SlideCheck provides explicit abnormality and malignancy scores for organizing, filtering, and auditing pathology pretraining data. SlideCheck uses a dual-head MLP to separately model broad abnormal morphology and malignant evidence. A regularized feature-space scorer provides a supervised anchor for patch-level evidence estimation, while score-attention agreement combines patch scores with WSI-level MIL attention to mine high-confidence pseudo labels. The same scores are then used to construct broad-positive ViT pretraining subsets, where a patch is selected if either abnormality or malignancy evidence exceeds a threshold. Experiments show that SlideCheck-defined data distributions influence the downstream behavior of self-supervised ViT pretraining, indicating that biological composition is an important controllable factor in pathology foundation model development. Curated subsets can approach full-data performance, suggesting that explicitly scored patch pools may support more efficient and auditable pretraining data construction. These findings position SlideCheck as a data guidance and auditing layer for transforming large, undifferentiated patch pools into controllable and reusable pretraining datasets. 

---
# From Human Guidance to Autonomy: Agent Skill System for End-to-End LLM Deployment on Spatial NPUs 

**Authors**: Jiajie Li, Erwei Wang, Zhiru Zhang, Samuel Bayliss  

**Link**: [PDF](https://arxiv.org/pdf/2606.07586)  

**Abstract**: Spatial neural processing units (NPUs) provide an energy-efficient platform for edge LLM inference, but efficiently deploying an LLM end-to-end on such hardware remains labor-intensive. Although AI coding agents have begun to lower this cost, existing studies have largely focused on single-kernel optimization rather than end-to-end LLM deployment on resource-constrained spatial NPUs.
We present a two-stage methodology, instantiated on the AMD XDNA 2 NPU, that progresses from human-guided development to agent autonomy. In the first stage, we develop a reference deployment of Llama-3.2-1B through human-guided agent assistance. The resulting implementation achieves a speedup of 2.2x on prefill and 4.0x on decode over the hand-optimized baseline, with the optimization trajectory and its lessons recorded as structured documentation throughout. In the second stage, we distill the documentation into an agent skill system consisting of eight phases, orchestrating the optimization and debugging skill sets, with numerical correctness strictly enforced at each phase.
Using our agent skill system, we autonomously deploy eight additional decoder-only LLMs (Llama-3.2-3B, SmolLM2-1.7B, Qwen2.5-{0.5B, 1.5B, 3B}, Qwen3-{0.6B, 1.7B, 4B}) end-to-end on the AMD XDNA 2 NPU using the open-source compiler stack. To our knowledge, these models have not previously been deployed on AMD NPUs via any open-source software stack. Each deployment completes in 0.5-4 hours of agent wall time with almost no human guidance, and passes the numerical-correctness gates, demonstrating functional generalization to previously unencountered LLMs. Three of the eight match or exceed the sustained performance of our Llama-3.2-1B reference deployment, suggesting that the resulting implementations can be competitive without additional model-specific human engineering. 

---
# Multimodal Group Emotion Recognition In-the-Wild Towards a Privacy-Safe Non-Individual Approach 

**Authors**: Anderson Augusma  

**Link**: [PDF](https://arxiv.org/pdf/2606.07585)  

**Abstract**: This thesis addresses group emotion recognition (GER) in-the-wild with a focus on privacy preservation. Unlike traditional emotion recognition methods that rely on individual-level cues such as face, gaze, or voice analysis, this work uses collective audio-video signals to infer emotions at the group level, reducing risks of individual monitoring and surveillance. Two complementary frameworks are proposed. The first is a cross-attention multimodal architecture for audio-video fusion, combined with Frames Attention Pooling (FAP) for temporal aggregation. It is supported by synthetic data augmentation and validated through ablation studies, demonstrating robustness in real-world GER conditions. The second framework, Variational Encoder Multi-Decoder (VE-MD), learns a shared latent space for emotion classification and structural representation prediction, including body and face cues. Two decoding strategies, DETR-based and heatmap-based, are explored to analyze the role of structural representations in group and individual settings. The thesis makes three main contributions: it clarifies the role of multimodality and structural cues in group-level affective computing; introduces two architectures for privacy-preserving multimodal GER; and shows that competitive performance can be achieved without using individual features as input data. 

---
# Outage Detection in Self-Healing Smart Grids Using Reinforcement Learning with Spectral Graph Neural Networks 

**Authors**: Lihui Liu, Mucun Sun, Caisheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07583)  

**Abstract**: Self-healing smart grids can quickly adjust their network configuration during outages to minimize power disruptions. During an outage, several actions can be taken, such as network reconfiguration through switching operations and emergency load shedding. However, traditional machine learning methods for outage mitigation are not well suited for smart grids due to their slow response time and high computational cost. To address these challenges, recent studies have explored reinforcement learning to automatically perform network reconfiguration. In these approaches, the control policy is typically modeled using a graph neural network (GNN). However, conventional GNNs operate in the spatial domain and may fail to capture important relationships in the frequency domain. Frequency-domain information is particularly useful for modeling global structural patterns and system-wide interactions in power networks. In this paper, we propose a spectral graph reinforcement learning framework for outage management in distribution networks to enhance system resilience. Our model learns the optimal power restoration policy using a spectral graph neural network. We evaluate the proposed method on three modified IEEE test systems: the 13-bus, 34-bus, and 123-bus networks. Experimental results show that our approach achieves near-optimal performance in real time and generalizes well across a wide range of outage scenarios. 

---
# Customer Churn Prediction on Structured Data Using FT-Transformer and Stacking Ensembles 

**Authors**: Joyjit Roy, Samaresh Kumar Singh, Laxmi Shaw  

**Link**: [PDF](https://arxiv.org/pdf/2606.07582)  

**Abstract**: Customer churn prediction is essential across data-driven industries such as insurance, digital banking, eCommerce, and subscription platforms, where retaining existing customers is typically more cost-effective than acquiring new ones. Predicting churn on structured datasets remains challenging due to class imbalance, nonlinear feature interactions, and heterogeneous feature types. Tree-based ensemble methods consistently demonstrate strong performance in these contexts, often outperforming conventional neural networks. This study introduces a validated hybrid architecture that integrates feature-tokenized transformers (FT-Transformer) with gradient-boosted trees through calibration-aware stacking. The proposed framework addresses persistent gaps in statistical validation, probability calibration, and reproducibility found in prior research. The FT-Transformer captures higher-order feature interactions using self-attention, while XGBoost captures gradient-boosted decision boundaries with complementary inductive biases. Class imbalance is handled using class-weighted loss functions, thereby avoiding synthetic oversampling and preserving minority-class distributions. The models are ensembled using out-of-fold (OOF) stacking with a logistic regression meta-learner, which recalibrates overconfident base model outputs and learns optimal combination weights. On a public bank churn dataset, the hybrid model achieves 62.10% F1, 0.861 AUC-ROC, and 0.647 PR-AUC, outperforming the Multi-Layer Perceptron (MLP) baseline by 3.37 F1 points and 0.027 AUC under 5x5 cross-validation with 95% confidence intervals reported. Ablation studies demonstrate that both the transformer component and stacking strategy contribute materially to performance. The proposed methodology offers a reproducible and extensible reference architecture for contemporary churn prediction on structured tabular data. 

---
# Training-Inference Kernel Contracts: Bounding Divergence in Post-Training and Deployment 

**Authors**: Bruce Changlong Xu, Lan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07581)  

**Abstract**: A modern post-training pipeline often writes one symbol for its policy, pi_theta, while evaluating it through two different programs: a training kernel optimized for autograd and an inference kernel optimized for low-precision, fused, dynamically batched serving. In finite precision, these kernels can induce different distributions at identical weights, with the gap concentrated on slices that aggregate benchmarks under-represent. This paper proposes kernel contracts: a contract-first framework for specifying acceptable divergence between K_train and K_inf. A contract C = (N, S, R, O, Pi) combines numerical, statistical, runtime, and observability clauses with an escalation policy from violations to routing actions. We derive a chain of bounds from logit drift to total-variation distance to bounded reward drift, and specialize it to RL post-training, where per-token importance-ratio drift yields a bound on policy-gradient bias under explicit support and norm assumptions. We also describe a four-stage promotion pipeline, online routing loop, and minimal YAML DSL for contract artifacts. This is a framework and vocabulary paper; we do not report production-scale empirical validation. 

---
# Accelerating Birkhoff Projection for Manifold-Constrained Hyper-Connections 

**Authors**: Chenrui Wang, Yixuan Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07574)  

**Abstract**: Manifold-constrained hyper-connections (mHCs) have recently been proposed as a principled extension of hyper-connections, where the residual mixing matrices are constrained to be doubly stochastic via projection onto the Birkhoff polytope. In practical mHC implementations, this constraint is enforced by Sinkhorn-Knopp iterations, and the backward pass relies on unrolling the iterative solver. This design introduces substantial computation and memory overhead, and may also yield inaccurate projections when the algorithm converges slowly on challenging inputs, undermining the intended norm-control and stability guarantees of mHCs.
In this work, we focus on the practically important 4x4 Birkhoff projection setting and develop an end-to-end acceleration framework. By leveraging the dual formulation, we reduce the problem to a three-dimensional unconstrained convex problem and solve it with Newton's method, achieving fast convergence and high accuracy. For the backward pass, we replace the unrolled differentiation with implicit differentiation, yielding exact gradients without storing intermediate states. To exploit massive parallelism, we design a warp-level CUDA kernel that uses only register-level primitives, avoiding global and shared memory I/O.
Extensive experiments against representative open-source baselines demonstrate that the proposed solver yields substantially more reliable doubly stochastic projections -- especially when the input magnitude is large -- and achieves significant end-to-end speedups (including the backward pass), reaching over 20x acceleration at large batch sizes while maintaining orders of magnitude smaller marginal errors. 

---
# Enabling KV Caching of Shared Prefix for Diffusion Language Models 

**Authors**: Younghun Go, Jaehoon Han, Changyong Shin, Chuk Yoo, Gyeongsik Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07571)  

**Abstract**: Key-value (KV) caching for shared prefixes is essential for high-throughput large language model (LLM) serving, but it faces critical challenges in emerging diffusion language models (DLMs). In DLMs, bidirectional attention means that updating any token dynamically alters the entire context and its corresponding KVs. Thus, existing caching techniques developed for LLMs, which assume that KVs remain invariant once computed, corrupt the shared prefix KVs. Our experiments show that applying these techniques to DLMs causes model accuracy to collapse to near zero.
To unlock high-throughput DLM serving, we propose bidirectional prefix caching, bicache, the first KV caching technique for shared prefixes in DLMs. bicache is designed based on key observations from our comprehensive analysis: shared prefix KVs remain stable and reusable in shallow layers, while the depth of shallow layers depends on the fraction of shared prefix tokens in each request. Thus, bicache dynamically identifies a safe layer depth for reusing shared prefix KVs and eliminates redundant computation. Evaluations demonstrate that bicache significantly improves serving throughput by 36.3%-98.3% compared to existing techniques without accuracy collapse (only 0-1.8% difference). 

---
# A Systematic Study of Behavioral Cloning for Scientific Data Annotation 

**Authors**: Ishaan Singh Chandok, Core Francisco Park  

**Link**: [PDF](https://arxiv.org/pdf/2606.07568)  

**Abstract**: Scientific data annotation, such as tracking animals in video or proofreading neural reconstructions, remains bottlenecked by the "last mile" problem: even with strong automation, verification and correction consume substantial human effort. Standard approaches train models to directly predict annotations, discarding the rich supervision in how experts navigate, click, verify, and correct. We introduce a framework for studying behavioral cloning on scientific annotation: 9 synthetic tasks paired with synthetic annotations that simulate realistic human strategies including exploration, mistake correction, and strategic decision-making. Our experiments reveal several findings. First, skills emerge hierarchically: models learn GUI mechanics before task-critical decisions, and commit fewer mistakes than the training data while retaining the ability to correct errors when they occur. Second, scaling models on multi-task behavioral cloning shows that larger models are more data efficient within our scale range. Third, multi-task pretraining enables efficient fine-tuning to new tasks, while training from scratch fails entirely. Fourth, linear probes reveal that models internally represent latent variables of the annotation process such as task phase and data position; interestingly, we find a shared mistake representation that generalizes across different annotation tasks. Overall, our framework establishes systematic benchmarks and identifies key bottlenecks, providing a foundation for scaling behavioral cloning to real-world scientific data annotation. 

---
# SurfDesign: Effective Protein Design on Molecular Surfaces 

**Authors**: Fang Wu, Shuting Jin, Xiangru Tang, Mark Gerstein, Xiangxiang Zeng, Yejin Choi, Jure Leskovec, Jinbo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07567)  

**Abstract**: Protein function is largely determined by molecular surface geometry and physicochemical complementarity, yet most protein design methods condition only on backbone structure. We introduce SurfDesign, a surface-conditioned protein design framework that models molecular surfaces as continuous geometric manifolds and integrates them with pretrained protein language models. SurfDesign employs surface-based equivariant message passing to capture surface normals, curvature, and directional geometry, together with a parameter-efficient fine-tuning strategy. Focusing on functional protein design, we show that SurfDesign consistently outperforms prior surface-conditioned and backbone-only methods on de novo binder and enzyme design benchmarks. We also report strong performance on inverse-folding benchmarks as a diagnostic of structural compatibility. Our results highlight manifold-aware surface representations as a principled foundation for functional protein and enzyme design. Code is available at this https URL. 

---
# Considerations for an Integrated Detector Design at FCC-ee: A Human-AI Exploration 

**Authors**: Charles Young  

**Link**: [PDF](https://arxiv.org/pdf/2606.07564)  

**Abstract**: This report explores detector design considerations for the Future Circular Collider in its electron-positron mode (FCC-ee) through an extended dialogue between a physicist and an AI assistant. Starting from initial "prejudice" detector concepts proposed by the AI assistant without explicit physicist input, each subsystem is examined in detail, with the AI's assumptions challenged and revised through the exchange. The discussion covers the full detector from beam pipe to luminosity monitor, with particular attention to the interplay between subsystem choices and the practical considerations - calibration, stability, and operational simplicity - that are essential for a fifteen-year precision physics program. The narrative documents how the integrated detector design evolved substantially from the starting point to revised "prejudice" detector concepts of the AI assistant. The focus of this report is on the process to illustrate both the potential and the limitations of human-AI collaboration in experimental physics design, and the physics capabilities of any of the "prejudice" detector concepts remain to be explored. 

---
# Emergence via Phase Transitions: Mechanism Landscapes and Universal Convergence Across Complex Systems 

**Authors**: Truong Xuan Khanh  

**Link**: [PDF](https://arxiv.org/pdf/2606.07563)  

**Abstract**: Across machine learning, biology, and physics, independently evolving systems often converge toward strikingly similar high-level structures despite radically different microscopic details. Grokking circuits converge across random seeds, evolutionary lineages rediscover similar metabolic solutions, and renormalization flows approach common fixed points. We propose the Hierarchical Emergence Framework (HEF) as a candidate universality framework for such convergence phenomena. HEF models emergence as a phase transition in a mechanism landscape constrained by thermodynamic and information-theoretic laws. The framework introduces a critical energy threshold Ec separating an exploration regime with competing mechanisms from a convergence regime governed by a unique minimum-cost mechanism. Under structural assumptions, we prove physical feasibility, derive strict metric contraction, and establish convergence toward a unique fixed-point representation independent of initial conditions. We further connect this convergence structure to causal emergence through Effective Information and mechanism competition entropy. To test the framework, we study delayed generalization ("grokking") in modular arithmetic transformers across 111 experiments. We identify a reproducible empirical fingerprint of the Ec transition: the weight norm peaks systematically before grokking in 92% of runs. Normalized accuracy curves collapse onto a tanh kink (R^2=0.93) consistent with a Landau-Ginzburg universality class, and all grokked models converge to 0.9745+/-0.014 regardless of initialization, weight decay, or training fraction (ANOVA p>0.13). HEF is not presented as a universal theory of emergence, but as a falsifiable mathematical scaffold for studying convergence phenomena across complex systems. 

---
# The Montparnasse Algorithm for RNA Design 

**Authors**: Tristan Cazenave  

**Link**: [PDF](https://arxiv.org/pdf/2606.07562)  

**Abstract**: RNA design consists of discovering a nucleotide sequence that optimizes predefined criteria, such as secondary structure. It is useful for synthetic biology, medicine, and nanotechnology. We propose Montparnasse, a Monte Carlo search framework based on Generalized Nested Rollout Policy Adaptation, augmented with a problem-specific prior, slow and long adaptation at level 1, and a lexicographic multicriteria evaluation. Montparnasse solves all 100 puzzles of the Eterna100 V1 benchmark consistently faster than DesiRNA, the previous state of the art, across all time limits, reaching full coverage more than three times faster overall. On messenger RNA secondary structure optimization for hemoglobin alpha, it identifies sequences with more paired bases than the MFE-optimal solution of LinearDesign. 

---
# Phantom transitions in language model fine-tuning 

**Authors**: Vaibhav Prakash, Jayasri Dontabhaktuni  

**Link**: [PDF](https://arxiv.org/pdf/2606.07559)  

**Abstract**: Fine-tuning a language model on contexts whose correct completion has a near-synonym competitor often fails silently. The cross-entropy loss decreases monotonically while the correct token never overtakes the competitor in rank. We study this regime across five transformer architectures spanning two families and a fivefold parameter range, on ten hand-selected near-synonym contexts. We instrument these failures with an order parameter combining the predicted distribution and pairwise embedding overlaps. It decomposes additively into a signal, tracking the model's commitment to the correct token over its nearest competitor, and a background drag, set by how the embedding bulk leaks probability into the score. This isolates two failure modes. In kinematic failure the signal stays small. In structural failure the drag actively worsens as fine-tuning proceeds. We observe sharp catapult-like jumps in the order parameter that resemble a phase transition. A central negative result organises the paper. The transitions are phantoms. The spontaneous-symmetry-breaking interpretation is ruled out by direct measurement. Catapult-like jumps still appear under LoRA fine-tuning with the token embedding matrix exactly unchanged during training, where no geometric phase transition is possible. The discontinuity lives entirely in the softmax readout. A small number of dimensionless quantities organise the trajectory across architectures. One is consistent across all five under full fine-tuning. A second sorts architectures into two classes by bulk embedding distribution and predicts LoRA sufficiency. As a blind test, the framework predicts the critical learning rate of a held-out architecture, not used to fit any parameter, to within 2.1% of a subsequent learning-rate sweep. Findings concern the near-synonym mechanism only and should not be extrapolated without recalibration. 

---
# Page image classifier fine-tuned on century-spanning archives of scanned documents for further content-specific processing 

**Authors**: Kateryna Lutsai, Pavel Straňák, David Novák, Dana Křivánková  

**Link**: [PDF](https://arxiv.org/pdf/2606.07558)  

**Abstract**: Purpose: Digitization projects in the humanities produce vast, heterogeneous archives of historical documents, making manual sorting impractical at scale. This work addresses the need for an automated system to classify scanned page images based on visual content type - text, tables, and graphics - enabling content-specific downstream processing such as Optical Character Recognition (OCR) or structured data extraction. Methods: An image classification system was developed and evaluated on a dataset of over 48,000 annotated historical page images from century-old Czech archaeological archives, refined through four successive annotation stages with domain-expert review. A Random Forest Classifier baseline was established using hand-crafted image features. Subsequently, deep learning architectures were fine-tuned and compared: Convolutional Neural Networks (EfficientNetV2, RegNetY), Vision and Document Image Transformers (ViT, DiT), and multimodal CLIP models. An 11-category label scheme was designed collaboratively with domain experts and evaluated via five-fold cross-validation. Results: The feature-based baseline achieved approximately 75% accuracy. Fine-tuned CNNs and Transformers substantially outperformed it, with RegNetY-16GF achieving 99.16% and ViT-large 99.12% Top-1 accuracy on the held-out test set. CLIP ViT-B/16 reached 99.14% with optimized text descriptions. Conclusion: Image-only models, particularly RegNetY-16GF, deliver near-perfect classification accuracy and produce consistent labels across 649,508 unlabeled archival pages with over 90% inter-model agreement. Fine-tuned CLIP, despite competitive test-set accuracy, showed under 65% agreement with image-only models on unlabeled data, making it less suitable for deployment. The final models, annotated dataset, and software are publicly available under open-source licenses. 

---
# Selecting New Measurement Locations to Diversify Traffic-Pattern Coverage: A Real-World Evaluation for Total Traffic Volume Estimation 

**Authors**: Masaaki Inoue, Akifumi Okuno, Shintaro Fukushima  

**Link**: [PDF](https://arxiv.org/pdf/2606.07556)  

**Abstract**: Accurate measurement of traffic volumes and flows is vital for modern intelligent transportation. However, despite recent technological advances in sensor devices, it is still expensive to install and maintain fixed traffic counters. Therefore, it is restricted to a small portion of location points where the counters can be installed, which severely limits the possibility of grasping and predicting the total traffic volume at a city-wide level. By contrast, devices with location history such as smartphones and connected vehicles are now widely used and provide much wider spatial coverage. However, the data from these devices are usually partial and noisy, so they are not enough to directly estimate total traffic volumes and flows. In this paper, we use the information from these widely available devices to help decide where to place additional traffic counters, and we study how selecting new measurement locations can improve city-wide traffic estimation performance. To achieve this, we propose an algorithm that chooses additional counter locations to increase the diversity of observed traffic signal patterns, rather than simply spreading counters evenly over space. The goal is to capture traffic-pattern types that are rare in the current counter set and to make the collected observations more representative for later estimation and forecasting. We also present a real-world evaluation; in a target city, we select new locations expected to improve traffic prediction, and we then commissioned new field measurements at those locations at our expense. The resulting data led to an improvement in traffic volume estimation accuracy across different fidelities. 

---
# MedicalRec: Medical recommender system for image classification without retraining 

**Authors**: Roghayeh Taghavi, Aysa Hasanazde Bashkandi, Amir Ali Bengari, Mohammad Amin Raji, Mohammad Salahi Ardekani, Parisa Mardukhian, Parvaneh Rezaei, Ramin Mousa  

**Link**: [PDF](https://arxiv.org/pdf/2606.07553)  

**Abstract**: The emergence of machine learning and deep learning has revolutionized the efficiency of diagnostic, therapeutic, and administrative systems in healthcare. However, this rapid adoption has come at the cost of requiring significant computing power and energy consumption, as well as e-waste disposal and carbon emissions. One of the challenges of these models is choosing the right model for classification tasks. To this end, researchers attempt to identify the optimal model using their data through trial and error, which involves energy consumption and waste. The goal of this study is to develop a model-based recommender system for medical image classification. For this purpose, a data set was collected from 3,000 articles in the field of medical image classification. This dataset, publicly available under the name MedicalRec-Bench, contains over 5,000 records of models tested in various tasks, including Skin Cancer Classification, Tumour Classification, Wound Classification, Breast Cancer, and MRI classification. The dataset was evaluated in four different modes, depending on the number of features: MedicalRec I (5 features), MedicalRec II (9 features), MedicalRec III (11 features), and MedicalRec IV (18 features). Collecting all values for the features is challenging due to non-reporting by the authors; hence, the dataset contains significant amounts of missing values. The Medical Recommender System (MedicalRec) is a transformer-based model used for item recommendations in this study. This model achieved remarkable results in the evaluation on the dataset and in the evaluation with 12 base models. This model achieved a maximum HitRate@100 of 75.5%. The dataset and implementations are available through the GitHub link: this https URL 

---
# Symbolic Reasoning Frameworks Modulate LLM Risk Aversion in Multi-Agent Strategic Settings 

**Authors**: Augustin Chan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07552)  

**Abstract**: Large language models exhibit innate behavioral tendencies when deployed as strategic agents -- notably a risk-averse "turtle" bias toward defensive play. We show that symbolic reasoning frameworks, injected as per-round reflective prompts into one agent, differentially modulate this bias and reshape the multi-agent ecosystem to produce framework-specific winner distributions. In a 7-player Warring States Diplomacy variant (41 games, 4 conditions, single-campaign memory accumulation), each framework produces a distinct ecosystem signature: under control, Yan dominates (7/11, 64%); under I-Ching yarrow divination, Yan and Chu co-dominate while Qin is completely suppressed (0/10); under Tarot, Qin dominates (5/10, Fisher vs. pooled p = 0.006); under scrambled-text ablation (incoherent oracle text preserving prompt structure), Qi dominates (5/10, Fisher vs. pooled p = 0.006). The framework-receiving agent (Han) never wins and shows no survival difference across conditions (Fisher p = 1.0), but Tarot consistently elevates Han's peak territory (mean 3.0 SCs vs. 2.1-2.5 others, Kruskal-Wallis p = 0.010). Neither framework's content predicts subsequent actions -- hexagram themes (chi-squared p = 0.95) and Tarot card postures (chi-squared p = 0.69) are both independent of action choice -- suggesting the modulation operates through the reflective process, not content-following. We present this as an observation paper establishing that alignment-framework choice at the agent level produces distinctive system-level consequences in multi-agent settings. 

---
# Offline Reinforcement Learning for Plasma Control in Nuclear Fusion: Codebase and Benchmark 

**Authors**: Yang Fu, Haomin Bao, Rohit Sonker, Xiaoyan Hu, Aravind Venugopal, Jeff Schneider, Jiayu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.07550)  

**Abstract**: Offline reinforcement learning (RL) offers a promising route for developing plasma controllers from historical tokamak data, since online trial-and-error on real devices is costly and risky. However, progress in this direction remains difficult to measure due to the lack of a standardized offline RL benchmark for realistic multi-actuator, long-horizon plasma control problems in nuclear fusion. We introduce RL4F, an Offline Reinforcement Learning Benchmark for Plasma Control in Nuclear Fusion, providing closed-loop evaluation environments and baseline comparisons across four full-profile tracking tasks: rotation, density, temperature, and pressure. The dynamics function underlying the evaluation environment is built from historical discharge data from DIII-D, a real-world Tokamak. We evaluate a broad set of imitation learning and offline RL baselines under a unified protocol. We find that offline model-based RL methods obtain the best average performance on most objectives, although no single method dominates all tasks, highlighting the importance of dynamics modeling in complex, long-horizon plasma control tasks. To foster further research, we open-source the codebase, datasets, and evaluation framework, providing a benchmark not only for the fusion community but also for algorithm development in offline RL. 

---
# Evaluating Advanced Prompting on Gemini Flash for Multi-Hop Biomedical QA 

**Authors**: Ahmed Bajaber, Mohammed Alliheedi  

**Link**: [PDF](https://arxiv.org/pdf/2606.07548)  

**Abstract**: The MedHopQA challenge presents a critical test for Large Language Models (LLMs): complex, multi-hop reasoning in the high-stakes biomedical domain. This paper details our direct API-based evaluation of Google's Gemini Flash models, focusing on the impact of advanced prompt engineering. We designed a sophisticated, multi-component prompt for Gemini 2.0 Flash that combined role-playing, explicit multi-shot Chain-of-Thought (CoT) examples, and detailed formatting rules. Our best run, using this complex prompt, achieved a Concept Level Score of 0.720. This result dramatically outperformed a baseline prompt which scored only 0.565. Remarkably, this performance on the efficient Gemini 2.0 Flash was almost identical to the result from the next-generation Gemini 2.5 Flash. Our findings demonstrate that sophisticated prompt design is a critical factor for unlocking the full reasoning capabilities of modern LLMs. 

---
# Liberating LLM Capabilities in Full-Duplex Speech Models 

**Authors**: Luoyuan Zhang, Bokai Xu, Junbo Cui, Weiyue Sun, Yingjing Xu, Hanyu Liu, Yuan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2606.07547)  

**Abstract**: Speech-based large language models are typically constrained to spoken replies, which limits their user-facing outputs to what can be verbalized and suppresses text-native capabilities such as code generation, structured analysis, and multi-step reasoning in realtime interaction, for tasks that require persistent, structured, and inspectable intermediate outputs. Existing work improves spoken reasoning or full-duplex turn-taking, but still treats text as a hidden intermediate state or a subordinate modality rather than a first-class output channel. We propose Listen-Write-Speak (LWS), a text-first tri-channel paradigm in which a single autoregressive LLM continuously listens to user audio, writes visible free-form text as its primary output, and speaks a realtime oral response in parallel under a shared causal attention context. This behavior is implemented entirely through a Token Schema, requiring no architectural modifications, and learned via a two-stage data pipeline that synthesizes per-second cognitive annotations consistent with the revealed input timeline. Empirically, LWS demonstrates strong full-duplex interaction on Full-Duplex-Bench, reaches 4.72 on VoiceBench AlpacaEval, achieves 92.6% writing-speaking consistency, and consistently outperforms its internal ablations on URO-Bench. These results suggest that visible writing can serve as a first-class output channel for speech interaction without sacrificing realtime responsiveness. The code and dataset are available on the project page: this https URL. 

---
# Beyond Item IDs: Scaling Short-Form-Video Recommendation via Semantic-Native Long Sequence Modeling 

**Authors**: Ruixiao Sun, Diego Uribe Mora, Zhimeng Jiang, Yuanzhen Lin, Jiarui Wang, Yuening Li, Danfeng Guo, Zhizhong Chen, Chuan He, Liang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07546)  

**Abstract**: Capturing user interests across extensive watch histories is critical for short-form video recommendation, yet scaling sequence length is limited by two bottlenecks: the semantic sparsity of atomic Video IDs and the quadratic computational complexity of Transformers. Traditional orthogonal Video IDs fail to capture content relationships and demand large embedding tables, while the quadratic complexity of self-attention restricts the maximum sequence length under strict industrial latency and resource constraints. In this work, we present a production-deployed framework for modeling ultra-long user behavior sequences at a billion-user scale. We first address the representation bottleneck by adopting content-native Semantic IDs. By utilizing depth-truncated, coarse-grained Semantic IDs, we shrink the embedding table size from corpus cardinality. This compact representation naturally generalizes to cold-start content through shared semantic prefixes. Second, to overcome the sequence scaling barrier, we introduce a Global-Aware Compression Transformer that leverages non-parametric temporal folding and unified global query integration to effectively condense the sequence, alleviating both the memory and computational bottlenecks of standard self-attention. Offline profiling on our computing infrastructure demonstrates an order-of-magnitude reduction in peak memory footprint and a drastic decrease in computational overhead. This efficiency gain enables supporting longer sequence lengths at an affordable cost in production, yielding substantial online gains in satisfied user engagement and satisfied content consumption in large-scale online A/B tests. 

---
# AI-Integrated Learning Management System for Middle School: A Longitudinal Study of Learning Outcomes Through High School and Beyond 

**Authors**: Misan Paul Etchie, Taiwo Olutosin  

**Link**: [PDF](https://arxiv.org/pdf/2606.07544)  

**Abstract**: Middle school is a key window for building core academic skills and the learning routines students carry into later grades, yet many students still fall behind because help is often limited and comes too late, after they have already been stuck for a while. Learning Management Systems (LMSs) are now standard infrastructure for distributing materials, collecting work, assessing students' tasks, and recording grades, but in most deployments they still behave more like workflow tools than instructional supports. The result is the usual bottleneck: students keep practicing through confusion, teachers triage questions, and feedback that could have corrected the misunderstanding arrives after the misconception has already hardened. To address this gap, we propose an AI-integrated LMS for middle school instruction, paired with a longitudinal study design to test whether sustained, bounded AI support changes outcomes through high school and into post-high school pathways. The proposed platform adds policy-gated AI assistance to everyday coursework, delivering formative feedback and hinting, recommending spaced review and adaptive practice based on mastery, and providing teacher-facing dashboards that summarize misconception patterns and flag sustained struggle. Because the platform is intended for minors, the design is privacy-first, using data minimization, role-based access control, age-appropriate response constraints, and auditable logs of AI interactions. Beyond short-term performance, the evaluation plan links fine-grained learning traces (attempts, revisions, help-seeking, and pacing) to institutional outcomes where feasible, so we can separate tool adoption effects from longer-run changes in learning trajectories. 

---
# Concerns and Strategic Responses of Older Workers Navigating Generative AI in Bridge Employment 

**Authors**: Aditya Nayak, Aakash Gautam, Rama Adithya Varanasi  

**Link**: [PDF](https://arxiv.org/pdf/2606.07543)  

**Abstract**: Generative AI (GenAI) is transforming workplaces at a rapid pace. This disproportionately affects vulnerable communities, including older workers (OWs) who re-enter the workforce through bridge employment prior to final retirement. Through in-depth semi-structured interviews with 21 professionals, we examine how OWs navigate GenAI-driven disruptions while pursuing bridge roles, focusing on their concerns about GenAI integration and their responses to these changes. Our findings show that OWs experienced both temporal and structural disruptions across all stages of the bridge employment decision-making process due to GenAI. In response, they reconfigured their tasks through different forms of boundary work aimed at restoring stability and continuity. We conceptualize these responses as AI resilience, which reshaped OWs' bridge employment decision-making into an ongoing process of negotiation and adaptation. We conclude by offering recommendations to reduce burnout among OWs by balancing individual-level AI resilience strategies with meso-level AI resilience collectives and macro-level adversarial and contestable AI-mediated organizational structures. 

---
# DIYHealth Suite: Dataset, Model, and Benchmark for Health Management at Home 

**Authors**: Changshuo Liu, Junran Wu, Zhongle Xie, Wenqiao Zhang, Kaiping Zheng, Jiaqi Zhu, Qingpeng Cai, Ooi Gene Anne, Marcus Chun Jin Tan, Jianwei Yin, James Wei Luen Yip, Beng Chin Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2606.07542)  

**Abstract**: Generative AI is reshaping healthcare, yet most existing advances rely on hospital-grade devices, which limits their accessibility and potential for health management outside clinical settings. With the proliferation of portable devices and telemedicine, healthcare is shifting toward home-based Diagnosis-It-Yourself (DIY) care. Despite this promise, several distinctive challenges remain: (i) home-collected data are heterogeneous, exacerbated by the absence of standardized large-scale datasets; (ii) models require adaptation to variable task demands and evolving individual conditions; (iii) the broad spectrum of home care tasks lacks a unified benchmark for systematic evaluation. In this paper, we present DIYHealth Suite, a comprehensive framework designed to address these challenges through a tailored dataset, model, and benchmark. We first curate DIYHealth-900K, a large-scale multimodal dataset capturing diverse real-world home care scenarios. Building on this, we propose DIYHealthGPT, an adaptive foundation model for home-based health management, powered by the novel Hybrid Hyper Low-Rank Adaptation technique. Finally, we establish DIYHealthBench, the first benchmark to evaluate foundation models on home care tasks. Extensive experiments demonstrate that DIYHealthGPT delivers state-of-the-art performance over both general-purpose and medical-specific baselines on 11 home care tasks in both open-QA and closed-QA settings, laying the groundwork for the next generation of personalized health management at home. 

---
# Multimodal Large Language Models as Synthetic Participants in Video-Based Studies: An Evaluation 

**Authors**: Prabal Shrestha, Bohan Jiang, Haoning Xue, Huan Liu, Xinyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.07541)  

**Abstract**: Multimodal large language models (MLLMs) have shown strong performance on objective tasks such as video understanding and reasoning. However, it remains unclear whether they can approximate subjective human responses, which depend not only on content comprehension but also on individuals' social contexts. To address this gap, we evaluate MLLMs as synthetic participants in an emerging task: assessing perceived sensory engagement with short videos. Grounded in the Perceived Message Sensation Value (PMSV) framework, we compare ratings from recruited human participants and profile-conditioned MLLM simulations (n=673) using a 17-item scale measuring emotional arousal, dramatic impact, and novelty. We find that even leading MLLMs (Gemini 3 Flash and Qwen 3 Omni) show limited agreement with human participants. The models exhibit distinct downward mean-shift and central-tendency biases in their rating distributions. They both introduce and flatten subgroup differences, while showing inconsistent sensitivity to participant profiles. Prompting strategies affect these metrics differently, modestly improving some aspects while worsening others. These results highlight both the challenges and opportunities of developing MLLMs as synthetic participants in video-based research. Data and code: this https URL 

---
# Bidirectional Semantic Complementary Tool Retrieval for Remote Sensing Agents 

**Authors**: Zeyuan Wang, Dongyang Hou, Cheng Yang, Xuezhi Cui, Linrui Xu, Bo Yu, Gaozhi Zhou, Ziyu Li, Liangtian Liu, Kai Ouyang, Wang Guo, Lili Zhu, Chao Tao  

**Link**: [PDF](https://arxiv.org/pdf/2606.07538)  

**Abstract**: Large language model (LLM)-based agents provide a novel paradigm for the automated processing of remote sensing(RS) data. Their success in complex RS tasks rely on extensive specialized tool libraries. However, tool documentation often exceeds the context window limits of LLMs, making precise tool retrieval essential for agentic workflows. Existing tool retrieval methods face "semantic asymmetry" bottleneck: natural language queries typically express macro-level intentions lacking tool-specific semantics, while tool documentation provides fine-grained technical descriptions lacking operational context for workflows. To bridge this semantic gap, this paper proposes a bidirectional semantic complementary tool retrieval method. First, on the query side, we introduce a planning-based query enhancement mechanism that leverages the reasoning capabilities of agents to decompose abstract intentions into logical subtasks, thereby actively supplementing the query with missing functional semantics. Second, on the tool side, addressing the strong coupling characteristics of RS tool chains, we construct a dynamic tool dependency graph with continual learning capabilities. By employing a neighborhood information aggregation mechanism, contextual information from precursor tools is explicitly injected into the current node representation, enriching tool descriptions with contextual semantics. Experimental results on the RS dataset GeoPlan-bench and the general-purpose dataset API- Bank demonstrate that the proposed method not only significantly improves tool retrieval accuracy for complex RS tasks but also exhibits robust extensibility for transfer to general-domain tasks. The source code and dataset are available at this https URL. 

---
# Beware of GeeksBearing Gifts: Building True EU Frontier AI Sovereignty 

**Authors**: Nick Moës, Toni Lorente, Amin Oueslati, Jonathan Smith, Robin Staes-Polet, Radina Kraeva  

**Link**: [PDF](https://arxiv.org/pdf/2606.07536)  

**Abstract**: Frontier artificial intelligence is reshaping all aspects of society, from economic output or military capability to democratic institutions. The EU is entering this transformation from a position of structural dependence: frontier models originate almost exclusively from the United States or China, the US holds approximately sixteen times the EU's AI supercomputing capacity, and only 15% of global hyperscale data centre capacity resides within EU borders. Although the European Commission has accelerated its policy response, existing initiatives remain fragmented and lack a cohesive vision for securing strategic autonomy across the full frontier AI value chain. Here we propose a unified framework connecting five sovereignty pillars (economic competitiveness, resilience, security and defence, European values, and foreign relations) to a decomposition of the frontier AI stack comprising five layers, 26 components, and 29 sub-components. This framework allows the identification of critical gaps, redundancies, and inter-pillar trade-offs that current EU policy leaves implicit. Our analysis of the AI Gigafactory Initiative illustrates how a sovereignty-centred lens reveals conflicts that narrowly economic framings obscure. Moreover, this framework offers policymakers a structured basis for designing, evaluating, and prioritising frontier AI interventions across multiple dimensions of European strategic autonomy across the 92 initiatives from four major Commission communications we. identify, and beyond. 

---
# Bridging Traditional Explainability Methods and Multimodal Multilingual Models: An XAI-Based Analysis 

**Authors**: Paweł Pozorski, Jakub Muszyński, Maria Ganzha  

**Link**: [PDF](https://arxiv.org/pdf/2606.07533)  

**Abstract**: Multimodal Large Language Models (MLLMs) effectively integrate text and audio to interpret context in complex interactive dialogues. However, the internal mechanisms by which heterogeneous modalities influence model behavior remain opaque. While Shapley Values (SV) provide a robust, model-agnostic framework for local explainability in text-based NLP, their extension to multimodal data is hindered by cross-channel dependencies, intricate dialogue structures, and the prohibitive computational complexity of dense audio representations.
In this work, we formalize a multimodal extension of the Shapley Value framework, treating discrete text tokens and aligned audio segments as cooperative features. To ensure computational feasibility, we deploy a suite of efficient estimation strategies: exact SV computation for low-dimensional inputs and sampling-based approximations - including Monte Carlo permutations and stratified sampling with Neyman-optimal allocation - to minimize variance under constrained computational budgets. To resolve the granularity mismatch between modalities, we propose Spectrogram-Guided Phonetic Alignment (SGPA), a novel preprocessing method that maps high-frequency audio streams to interpretable, word-aligned segments.
Our contribution is twofold: first, we provide an open-source, model-agnostic Python package and a companion GUI for the computation and interactive visualization of multimodal attributions. Second, we evaluate our framework using curated subsets of the VoiceBench and Infinity Instruct datasets across diverse multilingual scenarios. Our experimental results reveal that input modality is a primary driver of attribution volatility and demonstrate that standard syntactic importance proxies often fail to predict model attention in multimodal, cross-lingual contexts. 

---
# Principled Agent Debate: Adversarial Arbitration for Sycophancy Reduction in Large Language Models 

**Authors**: Sam Ryan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07532)  

**Abstract**: RLHF-trained models are systematically biased toward agreement over accuracy, a structural property of the training process. We present Principled Agent Debate (PAD), a multi-agent architecture that mitigates identity-framed sycophancy by arbitrating between two models tuned to opposing philosophical dispositions, with a pragmatist synthesizer evaluating both arguments blind to their origins. This paper evaluates a prompt-based instantiation of PAD. The key mechanisms are static dispositional tuning, identity stripping before synthesis, single-round independent argumentation, and blind arbitration. We evaluate five instantiations on 200 stratified questions from SycophancyEval. All PAD variants (AnCifer, DeWin, FeynStein, BurGal, Trident) significantly outperform the single-model baseline (18.5%) and instructed-opposition baseline (29.0%), with DeWin achieving 48.5% accuracy (z=6.36, p<0.001 versus both). The variants are not significantly different from each other at n=200. The BurGal variant achieves 53.0% but functions as an architectural validity check; its consensus/heterodox axis structurally favors the heterodox model on every benchmark question. A pre-training floor affects an estimated 40% of questions; fine-tuned disposition models are the identified next step. 

---
# mllm-shap: A Shapley Value Explainability Platform for Text-Audio Multimodal Large Language Models 

**Authors**: Jakub Muszyński, Paweł Pozorski, Maria Ganzha  

**Link**: [PDF](https://arxiv.org/pdf/2606.07531)  

**Abstract**: We introduce mllm-shap, an open-source Python framework designed to extend Shapley Value (SV) explainability from text-only Large Language Models to Multimodal LLMs (MLLMs) processing joint text and audio inputs. While text-based attribution is well-studied, mllm-shap addresses three critical challenges unique to the multimodal regime:
(1) Modality-aware coalition masking, which manages the interleaved processing of discrete text tokens and dense audio encoder frames.
(2) Multi-turn conversation tracking, utilizing per-token metadata to maintain role and modality context.
(3) Phonetic alignment-based token grouping, a novel technique that reduces the coalition space by 10x to 50x, rendering SV estimation computationally feasible for long-form audio.
The platform implements five SV estimation strategies, including a Complementary Contributions (CC) estimator with Neyman-optimal allocation that demonstrates superior convergence over standard Monte Carlo baselines. mllm-shap is provided as a pip-installable package featuring an interactive web-based GUI for granular attribution visualization. To our knowledge, this is the first publicly available framework providing a complete, reproducible pipeline for SV-based explainability in text-audio MLLMs. 

---
# CAPruner: Conceptual-Adjacent Scene Graph Pruner for Enhancing 3D Spatial Reasoning of Large Language Models 

**Authors**: Shengli Zhou, Xiangchen Wang, Guanhua Chen, Feng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.07529)  

**Abstract**: Large language models (LLMs) have recently been applied to 3D vision-language (3D-VL) tasks, which require spatial reasoning to identify target objects relative to anchors. Scene graphs are commonly employed to represent such relations, but reasoning over complete graphs incurs high token costs and computational inefficiencies, motivating the need for pruning. Existing pruning methods primarily rely on spatial proximity and often remove task-relevant relations, thereby undermining reliable spatial reasoning. To address these limitations, we derive a key requirement for scene graph pruning: preserving spatial relations that are most pertinent to the specific 3D-VL task. Guided by this insight, we propose the Conceptual-Adjacent Scene Graph Pruner (CAPruner). CAPruner integrates fuzzy semantic relevance with spatial proximity to estimate the importance of relations, enabling the selection of critical relations in a task-specific context. Moreover, to avoid costly relation-level annotations, CAPruner is trained by supervising the aggregated scores of each node's incident edges. Extensive experiments demonstrate that CAPruner effectively preserves relations essential for spatial reasoning, leading to substantial performance improvements of LLMs on 3D-VL tasks. Code is available at this https URL. 

---
# BEACON: Behavioral Entropy Aggregation for Cross-Model Hallucination Detection in Large Language Models 

**Authors**: Naveen Bera, Pulijala Sai Nikhila, Kondaguduru Abhiram, Shaik Gayaz Ali, Shoaib Sadiq Salehmohamed, Shaik Mohammed Omar, Jinal Prashant Thakkar, Hansika Aredla, Shalmali Ayachit  

**Link**: [PDF](https://arxiv.org/pdf/2606.07528)  

**Abstract**: Hallucination in large language models (LLMs), defined as the generation of factually incorrect or unsupported content, remains a critical barrier to reliable deployment. We present BEACON (Behavioral Entropy Aggregation for Cross-model hallucination detectiON), a black-box hallucination detection framework that operates purely on model outputs without requiring access to internal representations or external knowledge bases. BEACON extracts a 31-dimensional feature vector from structured multi-pass generation, integrating NLI-based semantic entropy, embedding geometry, chain-of-thought consistency, and paraphrase stability signals. A gradient-boosted classifier trained on 7,617 labeled examples across seven benchmarks achieves 0.8123 +/- 0.0102 AUROC (95% CI: 0.7632-0.8251), outperforming standalone semantic entropy (+0.2298) and SelfCheckGPT-style consistency baselines (+0.2457). Feature importance analysis shows that hallucination is inherently multi-dimensional, requiring combined uncertainty signals. An efficient 5-call variant achieves 0.7795 AUROC, enabling practical deployment across black-box LLM APIs. 

---
# Post-training is (Massive) Supervised Learning 

**Authors**: Michael Hassid, Yossi Adi, Roy Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2606.07527)  

**Abstract**: The prevailing paradigm for training LLMs has evolved to rely on a massive post-training phase consisting of SFT and RL. In this position paper, we argue that this methodology effectively marks a reversion to the ``pre-train then fine-tune'' approach of the BERT era, explicitly tailoring models to the desired behaviors and specific benchmarks on which they are evaluated. We begin with a historical overview of LLMs, describing the different phases of the LLM evolution. We argue that the current landscape is remarkably similar to the early days of LLMs, where task performance heavily relied on fitting the models to in-distribution datasets. To empirically demonstrate this, we compare pre-trained models to randomly initialized ones, by fine-tuning both variants on modern reasoning datasets and evaluating them on competitive math and code benchmarks. We show that models post-trained from scratch yield highly non-trivial performance. Our findings suggest that current post-training methodologies function primarily as a distribution-fitting mechanism. We finish by positing that developing generally capable models and systems requires moving beyond extensive post-training for predefined behaviors, shifting instead toward training procedures where models ``learn how to learn''. 

---
# GraphLoRA: Structure-Aware Low-Rank Adaptation for Large Language Model Recommendation 

**Authors**: Lin Mu, Guoji Wang, Li Ni, Lei Sang, Zhize Wu, Peiquan Jin, Yiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07526)  

**Abstract**: Large Language Models (LLMs) have shown strong potential for recommendation (LLMRec) due to their powerful reasoning and generalization abilities. However, effectively aligning the textual semantics modeled by LLMs with the collaborative signals remains a key challenge. Existing methods either translate collaborative information into textual prompts or inject pre-trained embeddings into the LLM, both of which treat structural information as static input and fail to capture high-order relational dependencies. To bridge this gap, we propose GraphLoRA, a novel framework that generalizes low-rank adaptation from independent to structure-aware propagation. GraphLoRA embeds a trainable graph message-passing network within the low-rank adaptation pathway, enabling structural signals to propagate through the parameter space. This design allows collaborative topology to explicitly guide parameter updates, fostering deep integration between graph-structured and textual semantic information. Extensive experiments on multiple benchmarks demonstrate that GraphLoRA not only outperforms state-of-the-art LLM-based recommendation methods but also achieves superior generalization, effectively balancing structural reasoning capability with computational efficiency. Code is available at \href{this https URL}{this https URL}. 

---
# Implicit Causal Graph Construction in Text via Chain Discovery 

**Authors**: Liesbeth Allein, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2606.07525)  

**Abstract**: Causal graphs in text are typically populated by observable, predefined events. In contrast, we study implicit causal graph construction from text by treating each described cause-effect pair as the begin- and endpoint of an underlying latent causal graph and using large language models (LLMs) to infer intermediate causal events. We compare end-to-end graph construction with methods that frame the task as causal chain discovery. In the latter, graphs are built either by aggregating inferred chains or by progressively expanding partial chains through an iterative search process. We further explore Wisdom of the Crowd extensions that access causal knowledge from multiple LLMs in post-hoc aggregation and collaborative inference settings. We analyze trade-offs among these approaches and evaluate the validity of inferred causal relations using a manually curated database of 1,560 scientifically validated causal pairs. This database-based evaluation is proposed as reliable, resource-efficient, and transferable to settings where ground-truth graphs are unavailable. 

---
# ABLE: Representing and Mapping LLMs via Attribution-Based Large-model Embedding 

**Authors**: Zirui Wang, Yusen Hou, Shaofeng Liang, Bowen Tian, Yanlin Zhang, Wenshuo Chen, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2606.07524)  

**Abstract**: The explosive growth of large language models (LLMs) has created a heterogeneous and poorly documented ecosystem, making systematic model comparison increasingly important for provenance auditing, security analysis, and model selection. Existing representation methods struggle to address this setting efficiently. Approaches analyzing internal parameters are powerful when architectures are compatible, but face scalability barriers under structural heterogeneity, while methods relying on external outputs may conflate models with similar behaviors and are difficult to align in richer output spaces across different tokenizers. To bridge this gap, we propose ABLE (Attribution-Based Large-model Embedding), a framework that leverages the interpretability space to construct model representations. By aggregating gradient-based feature attributions via a tokenizer-agnostic word-level alignment, ABLE captures model-specific input-sensitivity patterns rather than only surface-level outputs. Beyond empirical utility, we provide a stability analysis showing that, under standard regularity assumptions for differentiable Transformer-style models, ABLE induces a Lipschitz-continuous parameter-to-embedding map with finite-sample convergence guarantees. Extensive experiments on 239 open-source LLMs demonstrate that our training-free approach achieves competitive or superior performance in relation prediction, model routing, and benchmark score prediction. 

---
# Retrieval Augmented Generation Framework for the Nepali Legal Domain Question Answering 

**Authors**: Samir Wagle, Abiral Adhikari, Reewaj Khanal, Batsal Bhandari, Prashant Manandhar, Praveen Acharya, Bal Krishna Bal  

**Link**: [PDF](https://arxiv.org/pdf/2606.07523)  

**Abstract**: Legal domains in high-resource languages like English have widely adopted artificial intelligence for legal question answering. However, data scarcity in low resource languages such as Nepali has limited the training of large language models on Nepali legal texts. This study presents the first application of a Retrieval Augmented Generation based model for Nepali legal question answering using case laws extracted from the Nepal Kanun Patrika digital archive. Using BM25 on chunked documents, the approach achieved a top precision at one of 91 percent, and up to 75 percent with the multilingual E5 large model. Evaluation of generated answers showed 74 percent groundedness, 85 percent truthfulness according to an automated judge model, and 84 percent human evaluated truthfulness when using BM25 document retrieval, with a 92 percent successful answer generation rate. These results demonstrate that the RAG pipeline can effectively address the gap in legal question answering for low resource languages and provide a foundation for reliable AI systems in the Nepali legal domain. 

---
# Evaluating Hallucinations in Domain-Adapted Large Language Models 

**Authors**: Sanchita Porwal, Sai Prasath S, Xingjian Bi, Madelyn Scandlen  

**Link**: [PDF](https://arxiv.org/pdf/2606.07521)  

**Abstract**: This study investigates the phenomenon of hallucinations in domain-adapted Large Language Models (LLMs), focusing on the fine-tuning of the Llama-2 model with the Lamini dataset. Hallucinations, or the generation of nonsensical or unfaithful content by LLMs, pose a significant challenge, especially when these models are fine-tuned with domain-specific data. Our methodology involves a series of experiments testing memorization, recall, and reasoning capabilities of the fine-tuned LLM, comparing its performance on novel question-answer pairs and domain-specific information. We found that while the model shows proficiency in tasks similar to its training data, its capability to accurately reason about and recall new domain-specific information remains limited, leading to instances of hallucination. The model demonstrates a tendency to provide correct answers with extra information, suggesting an inclination toward over-generation. These results suggest important limitations of fine-tuning-only approaches for mitigating hallucinations when adapting LLMs to specialized domains and underscore the need for more robust methods in adapting LLMs to specialized domains. The study also provides insights into the varying performance of LLMs on different types of information, revealing a comparative weakness in handling domain-specific queries. 

---
# Bidirectional Small-Granularity Search between Code and Text 

**Authors**: Marco A. Valenzuela-Escárcega, Enrique Noriega-Atala, Gus Hahn-Powell, Clayton T. Morrison, Mihai Surdeanu  

**Link**: [PDF](https://arxiv.org/pdf/2606.07519)  

**Abstract**: We introduce the novel task of bidirectional small-granularity search between code and text, where the queries are small snippets of text or code and the results are also small fragments of the opposite modality, i.e., code or text. This task establishes direct links between text in scientific publications and corresponding code segments, in support of better and faster understanding of scientific methods. We introduce a large dataset for the proposed task that includes a training partition with textual descriptions of code generated automatically using GPT-4, and three testing partitions, one in-domain and two out-of-domain (OOD) that contain manually-annotated data as well as material from other domains. We also propose a modular approach to address this task. Our approach shares an encoder across four different subtasks that learn start/end of answer spans in both directions. We show that our method achieves good results in-domain, and encouraging results OOD. This suggests that addressing this task with automatically-generated data is possible, but there is exciting future work to be done. 

---
# Blockchain Infrastructure for Intelligent Cyber--Physical--Social Systems:Post-Quantum Security, Interoperability, and Trustworthy Data Economies in the Era of Embodied AI 

**Authors**: Song Guo, Huawei Huang, Dongping Liu, Aoyu Zhang, Luyao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.06895)  

**Abstract**: The deployment of embodied artificial intelligence via world-model-based robotics presents a transformative opportunity for blockchain infrastructure, establishing urgent demand for trustworthy data provenance, cross-organizational governance, and incentive-compatible sharing across decentralized ecosystems. Simultaneously, quantum computing advances recognized by the 2025 Nobel Prize in Physics and the Turing Award threaten the cryptographic primitives securing these data economies, creating an interdependent imperative: long-lived verification for embodied AI depends on crypto-agile architectures capable of withstanding quantum adversaries. This tutorial examines blockchain as the coordination layer bridging this dual transition, from financial substrate to foundational Cyber-Physical-Social Systems infrastructure that simultaneously secures against quantum cryptanalysis and enables scalable, trustworthy data economies. The session opens with an immersive AWS Braket demonstration engaging participants with superconducting, trapped-ion, and neutral-atom hardware to assess cryptographic threat timelines and witness ECDSA-to-post-quantum signature transitions. Five integrated modules progress from embodied AI and world-model requirements through quantum hardware reality and evidence-based security migration, to scalable cross-shard architectures via BrokerChain protocols, trustworthy data economies implementing Croissant metadata standards and robotic learning provenance, and industry ecosystem integration for multi-modal cloud deployment. By bridging quantum hardware realities with embodied AI data requirements, this tutorial charts blockchain as unified infrastructure for next-generation decentralized intelligent environments, providing open-source frameworks and roadmaps for architecting quantum-resistant, interoperable, and data-trustworthy systems. 

---
# BRAIN: Bayesian Reasoning via Active Inference for Agentic and Embodied Intelligence in Mobile Networks 

**Authors**: Osman Tugay Basaran, Martin Maier, Falko Dressler  

**Link**: [PDF](https://arxiv.org/pdf/2602.14033)  

**Abstract**: Future sixth-generation (6G) mobile networks will demand artificial intelligence (AI) agents that are not only autonomous and efficient, but also capable of real-time adaptation in dynamic environments and transparent in their decisionmaking. However, prevailing agentic AI approaches in networking, exhibit significant shortcomings in this regard. Conventional deep reinforcement learning (DRL)-based agents lack explainability and often suffer from brittle adaptation, including catastrophic forgetting of past knowledge under non-stationary conditions. In this paper, we propose an alternative solution for these challenges: Bayesian reasoning via Active Inference (BRAIN) agent. BRAIN harnesses a deep generative model of the network environment and minimizes variational free energy to unify perception and action in a single closed-loop paradigm. We implement BRAIN as O-RAN eXtended application (xApp) on GPU-accelerated testbed and demonstrate its advantages over standard DRL baselines. In our experiments, BRAIN exhibits (i) robust causal reasoning for dynamic radio resource allocation, maintaining slice-specific quality of service (QoS) targets (throughput, latency, reliability) under varying traffic loads, (ii) superior adaptability with up to 28.3% higher robustness to sudden traffic shifts versus benchmarks (achieved without any retraining), and (iii) real-time interpretability of its decisions through human-interpretable belief state diagnostics. 

---
# XAInomaly: Explainable and Interpretable Deep Contractive Autoencoder for O-RAN Traffic Anomaly Detection 

**Authors**: Osman Tugay Basaran, Falko Dressler  

**Link**: [PDF](https://arxiv.org/pdf/2502.09194)  

**Abstract**: Generative Artificial Intelligence (AI) techniques have become integral part in advancing next generation wireless communication systems by enabling sophisticated data modeling and feature extraction for enhanced network performance. In the realm of open radio access networks (O-RAN), characterized by their disaggregated architecture and heterogeneous components from multiple vendors, the deployment of generative models offers significant advantages for network management such as traffic analysis, traffic forecasting and anomaly detection. However, the complex and dynamic nature of O-RAN introduces challenges that necessitate not only accurate detection mechanisms but also reduced complexity, scalability, and most importantly interpretability to facilitate effective network management. In this study, we introduce the XAInomaly framework, an explainable and interpretable Semi-supervised (SS) Deep Contractive Autoencoder (DeepCAE) design for anomaly detection in O-RAN. Our approach leverages the generative modeling capabilities of our SS-DeepCAE model to learn compressed, robust representations of normal network behavior, which captures essential features, enabling the identification of deviations indicative of anomalies. To address the black-box nature of deep learning models, we propose reactive Explainable AI (XAI) technique called fastshap-C. 

---
