# UniClawBench: A Universal Benchmark for Proactive Agents on Real-World Tasks 

**Authors**: Zhekai Chen, Chengqi Duan, Kaiyue Sun, Bohao Li, Yuqing Wang, Manyuan Zhang, Xihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.08768)  

**Abstract**: The rapid development of large language models and multimodal large language models has accelerated the emergence of proactive agents capable of operating everyday tools and assisting users in real-world environments. However, existing benchmarks struggle to evaluate such agents effectively, as they often rely on sandboxed environments and single-turn evaluation paradigms. Moreover, their scenario-based task taxonomies mix multiple model capabilities within the same task category, making it difficult to identify the root causes of agent failures. To address these limitations, we introduce UniClawBench, the first capability-driven benchmark designed to evaluate proactive agents in dynamic, real-world settings. UniClawBench is built around five foundational model capabilities: Skill Usage, Exploration, Long-Context Reasoning, Multimodal Understanding, and Cross-Platform Coordination. Based on these capabilities, we design 400 bilingual real-world tasks. Unlike previous benchmarks that rely on static, pre-recorded answers, our benchmark evaluates agents in live Docker containers using fine-grained, step-by-step completion checkpoints. Furthermore, we design a closed-loop evaluation strategy comprising an executor agent, a hidden supervisor agent, and a user agent to simulate realistic multi-turn human feedback without leaking grading criteria. To disentangle base model capabilities from framework-level design choices, we evaluate state-of-the-art models under multiple agent frameworks. Through comprehensive comparisons across both models and frameworks, we show how base model capabilities and agent framework designs jointly shape performance in real-world environments. To facilitate future research, we make our benchmark and code publicly available at this https URL. 

---
# Validity of LLMs as data annotators: AMALIA on authority 

**Authors**: Manuel Pita  

**Link**: [PDF](https://arxiv.org/pdf/2607.08731)  

**Abstract**: A national language model offers a linguistic community its own instrument for measuring what its citizens say and value. Portugal's AMALIA, a publicly funded 9B-parameter model for European Portuguese, appears competitive on agreement alone: asked to code the moral foundation of authority, it agrees with trained human coders to within six F1 points of open models eight to thirteen times its size. Yet agreement is reliability, not validity. For theoretical constructs that must be inferred rather than read from surface features, the question is whether the model follows the construct's theory or reaches the right code by correlated shortcuts. We test this with the recovery gap: the loss in performance when a holistic prompt is decomposed into the codebook's atomic clauses and recombined by the theory's explicit rule. If calibration closes that gap, some portability should survive across models and languages; where it does not, the construct-model instrument is the likely locus of failure. We ask whether a calibrated English instrument transfers to AMALIA-9B and to European Portuguese. For one construct and one corpus, it does not. Decomposition recovers only about half of AMALIA's holistic performance, and error analysis suggests reliance on surface correlates, especially moral outrage near authority figures. An open multilingual LLM closes the gap on the same Portuguese corpus under the same instructions, pointing away from the corpus as the main explanation. AMALIA can still screen and pre-code at scale, but it cannot yet measure this construct well enough to stand alone. The study is a single counterexample, not a verdict on national models; it argues that sovereign-LLM benchmark batteries should test not only agreement with human coders, but the evidential route by which that agreement is warranted. 

---
# Do You Need a Frontier Model as a Citation Verifier? Benchmarking Rubric LLMs for Deep-Research Source Attribution 

**Authors**: Ethan Leung, Elias Lumer, Corey Feld, Austin Huber, Vamse Kumar Subbiah, Kevin Paul  

**Link**: [PDF](https://arxiv.org/pdf/2607.08700)  

**Abstract**: Reinforcement learning increasingly relies on an LLM judge to score each rubric criterion, and that judge acts as the reward model during training. Before such a signal can be trusted, we need to know how capable the judge must be and how biased it is. We study this calibration question for citation quality in deep-research systems, where a search-grounded LLM must support each claim it writes with a cited source. Citation quality is a structured rubric task in which each attribution-citation pair is judged along two dimensions that require an LLM, source relevance and factual support. On an adversarial long-form benchmark, we score 8 off-the-shelf LLM judges from 3 model families against gold labels over 1,248 rubric decisions, all of which were human-reviewed and 378 of which were hard cases adjudicated from judge disagreements. Cheaper judges remain competitive across both dimensions, with GPT-5-mini attaining the strongest source-relevance pass-class F1 at 0.908 ($\kappa$=0.636), while on factual support the judges are statistically indistinguishable (overlapping confidence intervals), so no single model dominates. At comparable F1, the judges still differ substantially in pass-rate drift, false positive rate, and false negative rate. Scalar F1 obscures this directional bias, yet it is exactly what a downstream reinforcement learning loop would reinforce. Calibrating the judge is therefore a prerequisite for using citation rubrics as reward signals, and our results show that this calibration does not require the most expensive available model. 

---
# WebSwarm: Recursive Multi-Agent Orchestration for Deep-and-Wide Web Search 

**Authors**: Xiaoshuai Song, Liancheng Zhang, Kangzhi Zhao, Yutao Zhu, Zhongyuan Wang, Guanting Dong, Jinghan Yang, Han Li, Kun Gai, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2607.08662)  

**Abstract**: Large language model (LLM)-based web search agents are transforming information seeking from simple factoid question answering into complex, deep-and-wide search and research-oriented tasks. A single ReAct-style agent is constrained by one long trajectory and limited context, making it difficult to handle depth and coverage simultaneously. Existing multi-agent systems improve search coverage through parallel execution and aggregation, but still exhibit clear limitations in recursive depth, collaboration adaptability, and evidence-grounded expansion. We propose WebSwarm, a progressive recursive delegation framework that jointly constructs task decomposition, recursive expansion, and agent collaboration during inference. WebSwarm dynamically instantiates agentic search nodes, each coupling a local objective with a search mode that specifies how the node should organize search and collaboration. Each node can either solve its objective itself or further delegate child nodes; after solving, it returns evidence and results upward, enabling parent nodes to further expand, revise, or aggregate the search process. To guide this process, WebSwarm first probes how task-relevant information is organized on the web to ground subsequent node expansion, and reuses process-level experience across homogeneous sibling nodes. Experiments on BrowseComp-Plus, WideSearch, DeepWideSearch, and GISA show that WebSwarm consistently outperforms single-agent and multi-agent baselines on deep, wide, and interleaved deep-and-wide tasks. Further analyses of ablation, task difficulty, web tool efficiency, and model generalization explain WebSwarm's effectiveness and provide insights for multi-agent search systems. 

---
# UltraX: Refining Pre-Training Data at Scale with Adaptive Programmatic Editing 

**Authors**: Xinlong Zhao, Dongsheng Liu, Hengyu Zhao, Zixuan Fu, Zheng Wang, Jie Cai, Jie Zhou, Qiang Ma, Xuanhe Zhou, Xu Han, Yudong Wang, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.08646)  

**Abstract**: As available training data approaches its physical limit, gains from Scaling Laws have begun to diminish. Consequently, improving Large Language Models (LLMs) now depends less on data expansion and more on higher-quality data utilization. However, in the context of large-scale corpora, existing refinement methodologies face significant limitations in quality, efficiency, and reliability: Rule-based approaches are constrained by fixed heuristics and struggle with instance-level variations; LLM-based approaches improve quality but fail to meet the efficiency and reliability requirements of large-scale data processing. To address these challenges, we propose UltraX, a function-calling refinement framework for large-scale pre-training data that completes the editing function space by introducing insertion in addition to deletion and modification, enabling fine-grained instance-level editing. Specifically, UltraX builds a reliable program-supervision generation pipeline. In this pipeline, dataset-adaptive prompt optimization first guides an expert LLM to produce high-quality end-to-end refined texts, and Line Alignment Mapping and Dynamic Context Replacement then convert original-refined text pairs into structured program supervision. Meanwhile, UltraX improves supervision quality and stabilizes the training distribution with low-confidence example filtering and ratio-controlled sampling by operation combination. During inference and execution, it normalizes and validates model outputs through sliding-window prediction, global operation aggregation, and systematic post-processing, improving the stability and reliability of large-scale execution. Experiments show that UltraX achieves the highest average performance across all corpora and also matches or surpasses baselines with fewer training tokens, demonstrating stronger data efficiency and refinement reliability. 

---
# DominoTree: Conditional Tree-Structured Drafting with Domino for Speculative Decoding 

**Authors**: Saw S. Lin, Jyh-Shing Roger Jang  

**Link**: [PDF](https://arxiv.org/pdf/2607.08642)  

**Abstract**: Speculative decoding accelerates LLM inference by drafting several tokens and verifying them in parallel. Block-diffusion drafters such as DFlash produce
a draft block in one pass but model only per-position marginals; best-first tree methods such as DDTree expand candidate trees from those marginals. The
released Domino drafter adds a GRU-based causal correction that makes each draft token's distribution path-dependent, a structure DDTree's factorized
formulation cannot represent. We introduce DominoTree, a training-free best-first draft tree scored by Domino's conditional, non-factorized correction
along each root-to-node path, made practical by restricting the per-node correction to a candidate top-M. On Qwen3-4B across eight benchmarks, DominoTree
reaches up to 6.6x speedup over autoregressive decoding and the highest mean accept length of any evaluated method, up to 10.7 tokens per round, at every
temperature we test. DominoTree constructs its tree with a GPU-native, CUDA-graph builder that is bit-identical to a reference Python implementation, so
acceptance is unchanged, while keeping per-round tree construction cheap. With this builder as default, DominoTree wins throughput over the released
Domino decoder at every temperature, 9-10% overall on Qwen3-4B and up to +22% on Alpaca, and over DDTree/CaDDTree at every temperature we test. On Qwen3-
8B, DominoTree keeps the highest accepted length at every temperature and adds a decisive throughput win at T=0, +24% over DDTree; at higher temperature
that edge over DDTree/CaDDTree narrows to a tie and a small loss, while its Overall aggregate wins over DFlash and Domino persist. 

---
# It Takes a MAESTRO To Prune Bad Experts 

**Authors**: Palaash Goel, Ayush Maheshwari, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2607.08601)  

**Abstract**: Sparsely-activated Mixture-of-Experts (MoE) language models achieve remarkable inference efficiency by activating only a small fraction of parameters per token, yet their full expert banks reside in memory at all times, creating a prohibitive deployment bottleneck. Existing structured pruning methods, largely designed for dense transformers, assess expert importance using locally derived heuristics that are blind to the interdependent nature of MoE routing. We introduce MAESTRO (Markov-chain Approximated Expert Sparsification via Transition-based ROuting), a structured pruning framework designed for MoE architectures that models autoregressive expert activation trajectories as Ergodic Markov chains whose stationary distributions encode cross-layer dependencies, yielding a globally aware importance heuristic. Evaluated across five diverse domains including Safety, Bias, and Ethics, MAESTRO outperforms state-of-the-art baselines by up to 10.61% in average performance retention under a strict 50% compression regime, while exhibiting substantially lower cross-task variance, indicating that global, routing-congruent pruning produces models that generalize more consistently across heterogeneous tasks. 

---
# When the Judge Changes, So Does the Measurement: Auditing LLM-as-Judge Reliability 

**Authors**: Zongyou Yang, Yinghan Hou, Xiaokun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.08535)  

**Abstract**: An LLM-as-judge score can move even when the candidate responses stay fixed, simply because the evaluator has changed. We treat this evaluator-replacement ambiguity as a measurement-validity problem. Across four judgment datasets, we compare two upgrade paths available in practice: scaling Qwen3 dense judges from 1.7B to 32B parameters and moving across MiniMax M2-M2.7 released APIs. The main pattern is that judge upgrades are not interchangeable: only Qwen3 1.7B to 4B gives a robust adjacent gain, while MiniMax adjacent releases do not. Stronger judges reduce but do not remove position and verbosity bias. Repeated-sample juries add little when errors are correlated. Structured debate can move decisions substantially, but without parser and fallback logs those shifts cannot be attributed to deliberation. We argue that LLM-as-judge reports should include dataset slices, bias probes, error-dependence estimates, and protocol audit trails. 

---
# Cross-seed explainability using Procrustes-conditioned Joint End-to-end Top-K Sparse Autoencoders 

**Authors**: Bendegúz Váradi, Zoltán Kmetty  

**Link**: [PDF](https://arxiv.org/pdf/2607.08499)  

**Abstract**: We present a Procrustes-conditioned Joint End-to-end Top-K Sparse Autoencoder (SAE) for extracting cross-seed universal features from independently trained BERT models. Cross-seed feature universality is a fundamental challenge in mechanistic interpretability: because dictionary learning is non-convex, independently trained networks learn misaligned feature spaces, so apparently identical features may differ by random initialization. We address this by computing an orthogonal Procrustes rotation between seeds' activation spaces before joint SAE training, combining Top-K sparsity, end-to-end downstream optimization, and an auxiliary dead-feature revival loss based on previous SAE literature. Evaluating on five independent seed pairs (ten BERT models) across three benchmark datasets (SST-2, Stanford Politeness, TweetEval Emotion), our full pipeline produces more universal features (Pearson r $\geq$ 0.70 across seeds) than post-hoc alignment baselines on all three datasets. A minimal qualitative analysis confirms that high-universality features encode interpretable sociolinguistic patterns. 

---
# Two Axes of LLM Abstention: Answer Correctness and Question Answerability 

**Authors**: Benedikt J. Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2607.08456)  

**Abstract**: A model should refuse two different things: answers it would get wrong, and questions it should not answer at all, such as unanswerable ones or ones resting on a false premise. The usual recipe thresholds a single confidence score, which cannot tell these apart. Across five instruction-tuned models from three families (2B to 14B), we find they are separate axes. Ordinary answer-confidence tracks whether an answer is right but is nearly blind to whether the question is answerable; a linear probe on hidden states does the reverse. The blind spot does not shrink with scale. It is worst on naturally occurring false-premise questions (CREPE). There, answer-confidence, P(IK), P(True), and even asking the model outright whether a premise is false all stay near chance, while a hidden-state probe reaches 0.69 to 0.77 AUROC: the model represents a problem it will not report. This turns out to be fixable. Instructing a model to check premises backfires, because it then disputes sound and false premises alike (57% false challenges), unable to tell them apart; routing the same instruction with the probe roughly triples challenge precision. We turn the two axes into a calibrated policy that answers only when an answerability score and a correctness score each clear a separately certifies behave differently: the unanswerable-answer rate is controllable at every scale, while the wrong-answer rate is capped by model accuracy, so the guarantee tightens as threshold policy certifies both budgets at 0.75 coverage of correct answers, against 0.31 for a single threshold; at 14B it is the only policy that certifies at all. 

---
# Detecting Ladder Logic Bombs in IEC 61131-3 PLC Programs using ESBMC-PLC+: A Formal Verification Approach with Trigger Synthesis 

**Authors**: Pierre Dantas, Lucas Cordeiro, Waldir Junior  

**Link**: [PDF](https://arxiv.org/pdf/2607.08417)  

**Abstract**: A Ladder Logic Bomb (LLB) is malicious control logic in a Programmable Logic Controller (PLC) program that lies dormant until a trigger activates a payload to manipulate actuators, forge sensor readings, or deny operator control. We observe that real malicious logic hides inside function-block bodies, which existing ladder-diagram verifiers drop from their intermediate representation (IR), making bombs invisible to provers. We present ESBMC-LLB, which uses ESBMC-PLC+ as its verification engine and adds a modeling layer that exposes function-block logic and recasts bomb detection as a formal verification problem: a scan-watchdog exposes non-termination payloads, and output wiring exposes actuator-forgery payloads as safety violations. k-induction gives an unbounded proof of bomb-absence across all scans, and the bounded model checker returns a counterexample that is the trigger - guarantees that signature, anomaly, and CFG-triage detectors lack. On the public Iacobelli 2024 dataset, ESBMC-LLB detects all 30 bombs and recovers every trigger; it also detects adaptive triggers (computed, opaque-arithmetic, multi-scan) that evade CFG-triage. We also report the first semantic model-checker evaluation on PLC-Defuser's SWaT corpus: our analog extension makes the full corpus parseable; on v1.0.0, it detects 149/150 bombs (99%) with zero false positives, recovering each trigger; on a later version with nonlinear non-termination bombs, detection drops to 49% as the SMT solver times out. We conclude that semantic model checking and CFG-triage are complementary - the former gives unbounded proofs, adaptive-trigger robustness, and handles Boolean/integer and linear analog logic; the latter leads to nonlinear analog non-termination, and we delineate where each wins. 

---
# When Synthetic Speech Is All You Have: Better Call GRPO 

**Authors**: Shashi Kumar, Yanis Labrak, Hasindri Watawana, Sergio Burdisso, Esaú Villatoro-Tello, Kadri Hacioğlu, Petr Motlicek, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2607.08409)  

**Abstract**: LLM-based ASR adapted to regulated domains such as banking is bottlenecked by privacy: real speech is costly and legally constrained to collect, making synthetic text-to-speech (TTS) an attractive substitute. Yet synthetic speech stays acoustically mismatched with real recordings, and work on this gap has stayed within supervised fine-tuning (SFT). We instead turn to reinforcement learning, and show that Group Relative Policy Optimization (GRPO) extracts far more from the same synthetic speech than SFT. Synthetic-only adaptation of the model with GRPO, a critic-free method rewarding low-WER hypotheses, reduces WER by 40\% relative to SFT (36.71\%$\to$22.09\%), and an SFT-then-GRPO combination pushes this further to 45\%. We trace the gain to behavior rather than representation: GRPO reduces insertion errors by improving stopping calibration and speech-to-text alignment by better anchoring attention to audio, leaving early-layer representations intact. When synthetic speech is the main resource, reinforcement learning should be preferred over supervised fine-tuning. 

---
# Prompt Compression via Activation Aggregation 

**Authors**: Thibaud Ardoin, Semira Einsele, Evis Bregu, Gerhard Wunder  

**Link**: [PDF](https://arxiv.org/pdf/2607.08399)  

**Abstract**: Large language models process prompts by propagating activations through dozens of layers before generating a response. We ask whether the task-relevant information contained in an instruction prompt can be compressed into a single activation vector and re-injected into the model, replacing the original token sequence? We show this is achievable using a learned weighted sum of activations extracted at an intermediate layer and injected at an early layer of the target LLM. The compressed vector preserves task-relevant information, incurring an accuracy drop of under $2\%$ relative to full prompt processing. Beyond its practical implications, including reducing per-query computation for fixed instruction prompts without reprocessing the original token sequence, our analysis reveals structure in the activation space of LLMs: (i) mid-layer representations transfer meaningfully to early layers, suggesting a degree of cross-layer compatibility in how information is encoded; (ii) a single activation vector encodes a quantifiable and recoverable amount of semantic information; (iii) a weighted sum of activations is a robust representation compressor. 

---
# Large-Language-Models-as-a-Judge in Theory-Agnostic Adaptive Metric-Alignment for Prototypical Networks in Personality Recognition 

**Authors**: Jing Jie Tan, Ban-Hoe Kwan, Danny Wee-Kiat Ng, Yan-Chai Hum, Shih-Yu Lo, Po-An Chen, Noriyuki Kawarazaki, Kosuke Takano, Anissa Mokraoui  

**Link**: [PDF](https://arxiv.org/pdf/2607.08374)  

**Abstract**: Personality recognition has traditionally been constrained by theory-dependent formulations, where models are trained to fit predefined psychological taxonomies rather than uncovering shared underlying behavioral structure. This limits generalization, as personality itself is better understood as theory-invariant, while existing annotations reflect only partial and sometimes inconsistent views of the same latent traits. In this work, we introduce JAM ((J)udge for (A)daptive (M)etric-Alignment), a theory-agnostic framework that shifts learning from adapting to predefined personality theories toward discovering unified latent pseudo-facets that capture shared psychological structure. Rather than constraining the model to any personality taxonomy during training or inference, the framework learns generalizable psychological representations and can infer an individual's latent psychological profile directly from the textual samples, without requiring theory-specific labels. JAM achieves this through an Attention-Pooled Graph Prototypical Network that learns structured representations via clustering in embedding space, together with a Cross-Theory Harmonization (CTH) approach that integrates (i) Human-Guided Linkage and (ii) Machine-Induced Consensus to unify heterogeneous datasets without relying on predefined labels. To further improve robustness and data quality, we incorporate an LLM-as-a-Judge mechanism operating in two configurations, (i) LLM-before-the-loop and (ii) LLM-in-the-loop which identifies ambiguous samples to guide adaptive metric learning. Experiments show that JAM improves cross-framework generalization and performance, establishing a strong step toward theory-agnostic personality inference and supporting low-resource personality theories. The related code repository, model weights, and artifacts are available at this https URL 

---
# Echoes Across Vietnam's Highlands, Delta, and Coast: A Multilingual Corpus for Cham, Khmer, and Tay-Nung 

**Authors**: Anh Trac Duc Dinh, Khang Nhat Hoang Vo, Vinh Cong Doan, Tai Tien Ta, Khoa Duc Anh Lam  

**Link**: [PDF](https://arxiv.org/pdf/2607.08362)  

**Abstract**: Vietnam's ethnic minority languages are almost absent from the field of Natural Language Processing (NLP), and the challenge goes beyond data scarcity: Cham, Khmer, and Tay-Nung differ sharply in script, Vietnamese contact, and standardization, conditions under which standard multilingual adaptation can learn the wrong signals. We introduce CKTN, the first corpus and benchmark for these languages (44,367 documents, 24M subword tokens), spanning continued pretraining, category classification, and summary-document retrieval. We show that existing multilingual encoders severely fragment these languages, and that common adaptation metrics can mislead: models may lower language-modeling loss or excel at lexical-overlap retrieval while still failing at semantic generalization across documents. We address this with a script-aware adaptation recipe - vocabulary augmentation combined with calibrated replaced-token pretraining - that prevents the discriminator from exploiting trivial script mismatches. The result is an encoder with substantially less fragmentation and the strongest classification performance among evaluated models, exposing the limits of lexical-overlap retrieval as an evaluation signal. 

---
# Grounded Event Extraction from SEC 8-K Filings with a Fine-Grained Taxonomy 

**Authors**: Rian Dolphin, Joe Dursun, Jarrett Blankenship, Katie Adams, Quinton Pike  

**Link**: [PDF](https://arxiv.org/pdf/2607.08346)  

**Abstract**: Form 8-K filings are the primary channel through which U.S. public companies disclose material events, but the SEC item codes attached to them are coarse: a single item spans routine administrative changes and chief executive departures, and many of the most market-moving disclosures fall into a catch-all item. Large language models make fine-grained labelling feasible at corpus scale, but only if the labels can be traced to the source text and shown to be reliable. We present a two-stage system that tags 8-K disclosures against a three-tier taxonomy of 119 event types. The first stage constrains output to valid taxonomy entries and anchors every tag to a verbatim quote via fuzzy n-gram validation; the second re-grades each cited quote against the category definition to produce a quality score. Applying the system to 292,984 filings from 2022 to 2026 yields 601,088 grounded event tags, which we release. Over 5,125 stratified tags, an LLM judge finds precision rises monotonically with the quality score, from 12% to 96%, while unsupported tags fall from 8% to near zero. Ablation shows the score is calibrated only when assigned in a dedicated second pass. An event study on unsigned abnormal returns confirms, without any language model, that the taxonomy separates economically distinct events sharing an item code. 

---
# TypeProbe: Recovering Type Representations from Hidden States of Pre-trained Code Models 

**Authors**: Giuliano Gorgone, Fausto Carcassi  

**Link**: [PDF](https://arxiv.org/pdf/2607.08339)  

**Abstract**: State-of-the-art code models achieve impressive performance, yet the extent to which they internally encode type information remains poorly understood. We probe the residual streams of pretrained code models for internal type representations using a parallel dataset of Java and Python code examples. Our results show that cross-lingual type representations emerge even from untyped code. Moreover, we test whether hidden states linearly encode the result type implied by typed function application by training probes on one language to infer argument and result types in the other. Finally, we find that this structure is partly robust to lexical perturbations and cross-language syntactic variations. To the best of our knowledge, prior work on interpretability of code models has not directly targeted formal type semantics or cross-lingual type representations. We release our code and datasets. 

---
# XALPHA: A Memory-Driven AI Quant Researcher for Hypothesis-to-Code Alpha Discovery 

**Authors**: Fengyuan Liu, Yuchen Fu, Yuqi Wang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.08332)  

**Abstract**: Financial markets are noisy, non-stationary, and high-dimensional, making it difficult to discover predictive and robust trading signals. Alpha discovery has evolved from manual factor design to machine learning, evolutionary search, and recent LLM-based frameworks, improving the efficiency of factor generation, search, and evaluation. However, existing methods still mostly automate isolated steps, rather than functioning as end-to-end quant researchers that can absorb external knowledge, close the hypothesis-to-code validation loop, and learn from accumulated discovery feedback. To fill this gap, we introduce XAlpha, a memory-driven AI Quant Researcher for continuous hypothesis-to-code alpha discovery. XAlpha maintains a multi-source research memory system that integrates report-grounded financial knowledge with discovery feedback from prior generations and research cycles. Guided by this memory system, a Macro Brain plans research themes and selects suitable Archetypes; a Micro Brain transforms the planned hypothesis pool into executable factor code and verifies ex-ante tri-alignment among the hypothesis idea, code logic, and financial plausibility; and a Cross Brain consolidates empirical outcomes into generation-level feedback, cycle-level summaries, and archetype-level research cues for future exploration. In this way, XAlpha turns alpha mining from isolated factor generation into a closed-loop research process that continuously reads, hypothesizes, implements, validates, reflects, and evolves. Experiments on CSI300 show that XAlpha achieves stronger overall alpha discovery performance than representative baselines. 

---
# Best-of-$N$ TTS Evaluation is Confounded by ASR Family Alignment 

**Authors**: Taehyung Yu, Seongjae Kang  

**Link**: [PDF](https://arxiv.org/pdf/2607.08256)  

**Abstract**: Best-of-$N$ (BoN) inference improves content consistency in zero-shot text-to-speech by selecting from $N$ candidates with an automatic speech recognition (ASR) verifier. We identify an underexplored evaluation confound: a verifier's apparent quality depends strongly on which ASR family judges it. On LibriSpeech-PC test-clean~\citep{librispeechpc} with F5-TTS~\citep{f5tts}, verifier rankings reverse across Whisper, wav2vec~2.0, and HuBERT evaluators, and same-family verifier-evaluator pairs recover 2-3$\times$ more oracle headroom than cross-family pairs despite near-identical representations (linear CKA $0.978$) -- a pattern consistent with identity- or lineage-level coupling rather than representational overlap. We propose two \textbf{cross-family rank ensembles} (rank-averaging and conjunctive max-rank) that attain the lowest mean WER across three independent evaluators -- $1.61\%$ at $N{=}10$ ($-12\%$ relative to F5-TTS) -- with no measurable degradation under automatic SIM-o/UTMOS metrics; the best single verifier drives WER from $2.06\%$ to $1.72\%$ ($-16.5\%$) under the official F5-TTS evaluator. We recommend cross-evaluator triangulation as default reporting practice. 

---
# Diarization-Guided Qwen-ASR Adaptation for Multilingual Two-Speaker Conversational Speech 

**Authors**: Hao Wu, RongQi Han, Zhen Wang, Wei Liang, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.08208)  

**Abstract**: This paper describes our self-designed system for Task 1 of the MLC-SLM 2026 Challenge for multilingual two-speaker conversational speech. The system combines a modular speaker diarization front end with a challenge-adapted Qwen3-ASR-1.7B recognizer. The diarization front end performs voice activity detection, subsegment generation, CAMPPlus speaker embedding extraction, two-speaker spectral clustering, and RTTM-based audio segmentation. The resulting speaker-attributed segments are grouped by language or region and decoded by the adapted ASR model. For ASR adaptation, we first perform supervised full fine-tuning on the official training data, then apply LoRA fine-tuning with synthetic speech generated by a three-pipeline TTS-based synthetic speech augmentation framework, and finally refine the model using GRPO reinforcement learning with rewards based on WER/CER and penalties for hallucination, repetition, and length deviation. On the official development set, the full system achieves an average tcpMER of 23.70, reducing the error rate by 6.83 absolute points relative to the released Qwen-ASR-1.7B performance. On the final evaluation set, the system achieves an average tcpMER of 17.97. Ablation results show that supervised fine-tuning provides the largest gain, while synthetic-speech LoRA adaptation and reinforcement learning further improve robustness. 

---
# Hidden Decoding at Scale: Latent Computation Scaling for Large Language Models 

**Authors**: Aiwei Liu, Cheng Shi, Chuhan Wu, Ci Lei, Di Lu, Donald He, Fan Zhang, Fanhao Kong, Feifei Zhang, Guan Wang, Haicheng Wang, Haoyu Liu, Houjin Yu, Jiachen Ding, Jiayi Feng, Jie Zhou, Jijun Chi, Jindi Shi, Jing Lei, Junjie Zhang, Laiyi Li, Le Tian, Linhao Zhang, Miao Fan, Sijun Zhang, Wei Jia, Weiwei Shi, Wenhan Li, Wentao Zhao, Wenteng Liang, Xiao Zhou, Xiaojin Zhou, Xihuai Wang, Xinyu Gao, Xuanliang Wang, Xuyang Ao, Yang Yu, Yangxiu You, Yinuo Zhao, Yufei Kuang, Yufei Wang, Yuan Liu, Yuan Liu, Yuwen Chen, Zhencong Tian, Zhongyin Zhao, Zilin Yu, Zitao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.08186)  

**Abstract**: Scaling Large Language Models (LLMs) has been driven mainly by enlarging the Transformer backbone, but for an already-strong model this requires another round of costly pretraining. We study whether an existing backbone can keep improving by allocating more computation to each token while leaving the Transformer backbone fixed. Depth-recurrent (looped) Transformers pursue this goal but are hard to scale, because looped computation does not fit naturally with the pipeline parallelism used to train the largest models. We add computation along the sequence-length dimension, where the extra computation is simply a longer input and stays compatible with standard large-model training. We propose Hidden Decoding, a sequence-length scaling method applied during continued pretraining (CPT). It expands each token into n streams with independent embedding tables and keeps the intermediate streams' key-value cache as context, so each token performs more internal computation without adding or widening Transformer layers. To keep this affordable at scale, we introduce Stream-Factorized Attention, in which most layers attend only within each stream and only a few layers mix across streams, reducing the attention cost from quadratic to roughly linear in n. Experiments support two scaling results. At frontier scale, we train WeLM-HD4-80B and WeLM-HD4-617B at n=4 and improve their matched non-HD baselines, making Hidden Decoding the first demonstrated sequence-length scaling method at the 100B+ MoE scale. Across expansion factors, the gains grow as n increases, showing that sequence-length expansion is a practical fixed-backbone scaling path for frontier-scale LLMs. 

---
# SQuaD-SQL: Efficient Text-to-SQL with Small Language Models via LLM-Guided Knowledge Distillation 

**Authors**: Wangyu Wu, Xiaojian Lin, Rong Fu, Zaiyang Yu, Xuhang Chen, Wenjun Yu, Zhenhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.08161)  

**Abstract**: Text-to-SQL is a fundamental task in natural language processing that enables users to interact with structured databases using natural language. While large language models (LLMs) have demonstrated remarkable performance on this task, their substantial computational requirements hinder deployment in resource-constrained settings. In this paper, we introduce SQuaD-SQL (Small-Qualified and Distilled for SQL), a novel approach that empowers small language models (SLMs) to approach the performance of LLMs on the Text-to-SQL task while significantly improving efficiency through knowledge distillation and synthetic data generation. Our method comprises three key components: (1) LLM-based synthetic data generation, where structured knowledge is extracted from LLMs via carefully designed prompting strategies; (2) parameter-efficient fine-tuning, enabling full model training on a single consumer-grade GPU; and (3) domain-adaptive fine-tuning, where domain-specific synthetic data further enhances performance in targeted domains. Experiments on the WikiSQL dataset demonstrate that SQuaD-SQL achieves an execution accuracy of 86.9% on the test set, approaching the performance of LLMs while offering faster inference and lower memory usage. These results suggest that, with proper training strategies, SLMs can serve as practical and efficient alternatives for Text-to-SQL applications in resource-limited environments. 

---
# LEXIC: Lightweight Eye-tracking eXtension via Injected Complexity 

**Authors**: Sumin Lee, Kyeonghun Kim, Subeen Lee, Jiwon Yang, Tien Nguyen, Ken Ying-Kai Liao, Nam-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.08152)  

**Abstract**: On the recent EyeBench benchmark, predicting reading comprehension from eye movements exposes a stark gap: text-aware models using pretrained language models reach 56--63% AUROC, while gaze-only models operate at chance. We ask how far a gaze-only model can be pushed by lightweight, language-model-free conditioning. Building on the EyeBench AhnCNN baseline, LEXIC-Base, we propose two mechanisms to inject three precomputed word-level difficulty signals, GPT-2 surprisal, word frequency, and word length, into the per-fixation input: direct concatenation, LEXIC-Concat, and a residual mechanism, LEXIC-Res, where a small head predicts typical-reader gaze response and the encoder is conditioned on the deviation. On the OneStop reading comprehension task, with K=5 seed-ensemble training across ten folds, both mechanisms produce statistically consistent AUROC gains on Unseen Text, +1.8 to +2.2 percentage points, Wilcoxon p <= 0.065. LEXIC-Concat additionally lifts Unseen Reader by +2.9 percentage points, p = 0.010. We trace an architectural boundary in LEXIC-Res on Unseen Reader, +1.8 percentage points, p = 0.19, to the prediction head being calibrated to training readers, transferring imperfectly to out-of-distribution readers. 

---
# ICDAR 2026 HIPE-OCRepair Competition on LLM-Assisted OCR Post-Correction for Historical Documents 

**Authors**: Maud Ehrmann, Emanuela Boros, Juri Opitz, Andrianos Michail, Florian Wagner, Simon Clematide  

**Link**: [PDF](https://arxiv.org/pdf/2607.08143)  

**Abstract**: We present the results of HIPE-OCRepair-2026, an ICDAR competition on LLM-assisted OCR post-correction of historical documents. OCR post-correction remains a long-standing challenge in digital heritage: large-scale collections of digitized documents are affected by legacy OCR errors, while re-digitization at scale remains impractical. Large language models (LLMs) offers a major opportunity to revisit this challenge, yet their effectiveness across languages, document types, and noise conditions - and their tendency to hallucinate - remains insufficiently understood. HIPE-OCRepair-2026 pursues two objectives: (i) to evaluate the capabilities of modern OCR post-correction systems, and (ii) to provide a reproducible evaluation framework anchored in the HIPE-OCRepair-2026 dataset, a harmonized multilingual resource consolidating existing and newly curated historical datasets. Participants were tasked with correcting noisy OCR transcripts from historical newspapers and printed works in English, French, and German (17th-20th century), working at the level of coherent transcription units (paragraphs or articles) without access to source images. The evaluation adopts a retrieval-oriented rather than diplomatic scoring approach, reflecting the practical use case of search and access over digitized collections. Four teams submitted systems ranging from zero-shot prompting to continued pre-training and fine-tuning, offering insights into the merits of different adaptation strategies. Results show that modern LLM-assisted systems can significantly improve OCR quality, but performance varies across datasets, languages, and noise levels. Over-correction on low-noise inputs emerges as a recurring challenge, highlighting the importance of evaluation beyond character error reduction. The dataset, scorer, and evaluation pipeline are publicly released to support future research. 

---
# COALA: Robust Contextualized Speech-augmented Language Modeling for ASR via Contrastive Regularizer and Biasing Score Estimation 

**Authors**: Jhih-Rong Guo, Bi-Cheng Yan, Tien-Hong Lo, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.08117)  

**Abstract**: Contextual biasing seeks to integrate external knowledge into automatic speech recognition (ASR) systems to accurately recognize domain-specific entities. In this paper, we propose COALA (Contextualized ASR Leveraging Biasing Scoring), a robust framework designed to enhance speech-augmented language models (SLMs) in complex multi-entity scenarios. Considering the inherent context-window limitations of SLMs, identifying relevant target entities from a large-scale biasing list is crucial for effective recognition. To this end, COALA maps SLM latent representations into a specialized discriminative space to quantify the matching intensity between audio segments and candidate entities. Furthermore, we address the training collapse in prior study when handling multi-target utterances-where multiple rare words co-occur. Experimental results on the LibriSpeech benchmark demonstrate that COALA consistently achieves superior contextual biasing performance across various biasing list scales. 

---
# MASTE: A Multi-Agent Pipeline for Zero-Shot Aspect Sentiment Triplet Extraction 

**Authors**: Ao Hong, Lehang Wang, Zhirun Yue, Mingxin Wang, Zihan Wang, Houde Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.08080)  

**Abstract**: Aspect Sentiment Triplet Extraction (ASTE) requires jointly identifying (aspect, opinion, sentiment) triples from a given review sentence. While large language models (LLMs) achieve strong zero-shot performance on many NLP benchmarks, their effectiveness on ASTE remains limited, as single-pass generation forces the model to determine span boundaries, opinion grouping, and sentiment polarity in a single decoding step. Common remedies, such as few-shot in-context learning and chain-of-thought prompting, offer only marginal improvements and rely heavily on either in-domain demonstrations sampled from labeled training data or carefully engineered reasoning prompts, neither of which is broadly available in zero-shot deployment. Inspired by the classical agent paradigm, we propose MASTE, a multi-agent pipeline for zero-shot Aspect Sentiment Triplet Extraction. MASTE decomposes ASTE into four sequential stages, where specialized agents handle different compositional subtasks with explicit conditioning on prior outputs. This design enables entirely training-free zero-shot ASTE and generalizes across different backbones and datasets. Extensive experiments on four ASTE benchmarks show that MASTE substantially outperforms zero-shot and chain-of-thought LLM baselines under the same backbone, narrowing the gap to fully supervised methods without using any labeled triplets. Code is available at this https URL. 

---
# COBART: Controlled, Optimized, Bidirectional and Auto-Regressive Transformer for Ad Headline Generation 

**Authors**: Yashal Shakti Kanungo, Gyanendra Das, Pooja A, Sumit Negi  

**Link**: [PDF](https://arxiv.org/pdf/2607.08071)  

**Abstract**: Online ads are essential to all businesses and ad headlines are one of their core creative component. Existing methods can generate headlines automatically and also optimize their click-through-rate (CTR) and quality. However, evolving ad formats and changing creative requirements make it difficult to generate optimized & customized headlines. We propose a novel method that uses prefix control tokens along with BART fine-tuning. It yields the highest CTR and also allows users to control the length of generated headlines for use across different ad formats. The method is also flexible and can easily be adapted to other architectures, creative requirements and optimization criteria. Our experiments demonstrate a 25.82% increment in Rouge-L and a 5.82% increment in estimated CTR over previously published strong ad headline generation baseline. 

---
# Holographic Neural PCFG for Unsupervised Parsing 

**Authors**: Ryosuke Yamaki, Daichi Mochihashi, Nobutaka Shimada, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2607.08063)  

**Abstract**: Unsupervised constituency parsing aims to accurately induce latent tree structures from raw text alone. Recent neural parameterizations of PCFGs achieve strong performance in both supervised and unsupervised parsing, yet rely on high-capacity black-box networks for rule scoring -- as exemplified by the Neural PCFG family -- leaving rule probabilities without an interpretable mathematical form. In this paper, we propose Holographic Neural PCFG (Hol-PCFG), which recasts PCFG rule scoring as algebraic relation modeling among grammar-symbol embeddings. Hol-PCFG adapts Holographic Embeddings (Nickel et al., 2016), which scores knowledge-graph triples via circular correlation, to the left-child, right-child, and lexical-emission relations over torus-constrained embeddings, giving every rule probability a closed form that carries the intrinsic structure of grammar rules by construction. Hol-PCFG achieves state-of-the-art parsing performance in six languages while cutting rule-scoring parameters by 99.94% relative to the baseline model and training more stably. Additionally, we demonstrate that Hol-PCFG can parse Japanese directly from characters without any morphological segmentation, retaining nearly the same morpheme-level performance. 

---
# What LLM Forecasters Know but Don't Say: Probing Internal Representations for Calibration and Faithfulness 

**Authors**: Raphaël Sarfati, Pratyush Ranjan Tiwari, Siddharth Boppana, Christopher J. Earls, Srikar Varadaraj, Eric Ho  

**Link**: [PDF](https://arxiv.org/pdf/2607.08046)  

**Abstract**: Large language models fine-tuned for forecasting can be accurate yet poorly calibrated, and their chain-of-thought (CoT) reasoning may not faithfully reflect the evidence behind a forecast. We ask whether internal representations offer a more direct window into both. Working with Eternis-Forecaster 8B on OpenForesight, we train representation-pooling probes on intermediate activations and find they achieve substantially better calibration; a result that also holds for GLM-4.7-Flash and GLM-4.5-Air. We then assess CoT faithfulness through evidence ablation and diversionary injection: removing an influential source in the prompt often changes the model's forecast while leaving the reasoning trace untouched. The same probes function as lie detectors: their activations track behavioral shifts far better than the reasoning trace does, and they also predict the direction of change in 84% of cases, including when the CoT conceals the perturbation's influence. Finally, forced answering reveals that forecasts are largely fixed before reasoning begins: a single pre-reasoning pass recovers the committed answer and confidence, and routing questions by the spread of this pre-set answer distribution saves 30-47% of generated tokens, with no loss of accuracy. Together, these results establish probing internal representations as a practical tool for calibrating, auditing, and triaging language model forecasters and reasoning models more broadly. 

---
# PLURAL: A Global Dataset for Value Alignment 

**Authors**: Dhruv Agarwal, Anya Shukla, Tanya Goyal, Aditya Vashistha  

**Link**: [PDF](https://arxiv.org/pdf/2607.08034)  

**Abstract**: Large language models (LLMs) are used worldwide, yet disproportionately reflect Western values, limiting their ability to represent diverse value systems. We introduce PLURAL, a large-scale, value-focused preference dataset grounded in the Integrated Values Survey (IVS), a nationally representative survey spanning 92 countries. Using a two-stage generation pipeline, we transform survey responses into synthetic preference triplets that preserve normative value signals while producing realistic scenarios. We release an initial version of PLURAL containing ~500,000 preference triplets representing people in 20 diverse countries. We evaluate PLURAL in three ways: (i) dataset-level validation showing that it preserves both cross-country value differences and within-country diversity from the original survey; (ii) automated evaluation showing that training on PLURAL improves alignment with target countries' cultural profiles, reducing mean absolute error by up to 27.7% relative to strong baselines; and (iii) blind human evaluation with 176 evaluators in India, Brazil, and Japan, who judge PLURAL-aligned responses as more representative of their national values. Together, these results show that PLURAL contains learnable signal for value steering, offering a scalable resource for pluralistic alignment. Dataset: this https URL 

---
# Structured Pruning of Large Language Models via Power Transformation and Sign-Preserving Score Aggregation with Adaptive Feature Retention 

**Authors**: Ryota Kobayashi, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi, Yasunori Ishii, Tomoyuki Okuno, Kazuki Kozuka  

**Link**: [PDF](https://arxiv.org/pdf/2607.08027)  

**Abstract**: This paper proposes an improved structured pruning method for large language models (LLMs) that addresses key challenges in adapting Adaptive Feature Retention (AFR), an unstructured pruning technique, to structured pruning. When applying AFR to structured pruning, three major problems arise: distribution mismatch between heterogeneous pruning scores, loss of sign information indicating optimization direction consistency, and influence of outliers. To address these issues, we propose a unified approach combining power transformation for nonlinear distribution alignment, sign-preserving score aggregation, and percentile-based outlier removal. Experiments on Llama-3-8B, Vicuna-v1.5-13B, and LLaVA-v1.5-13B demonstrate that our method maintains accuracy comparable to unstructured pruning while achieving practical inference speedup through structured pruning. 

---
# Can We Trust LLM's Logic? Quantifying Uncertainty, Coherence, and Robustness via a Graph-Based Framework 

**Authors**: Riccardo Revalor, Jalees Rehman, Debjit Pal  

**Link**: [PDF](https://arxiv.org/pdf/2607.08017)  

**Abstract**: Large-Language Models (LLMs) can be prone to flawed and unfaithful reasoning that decoding strategies like Self-Consistency (SC) fail to detect as they evaluate only final-answer agreement while ignoring the logical validity of intermediate steps. This raises three fundamental questions: How can we reliably quantify uncertainty in LLM reasoning? Can semantic, structural, and causal awareness select more faithful reasoning compared to naïve majority voting? and How robust is reasoning topology under adversarial conditions? To address these questions, we introduce GRAPHEVAL, a graph-based reasoning framework that re-frames uncertainty quantification (UQ) as a holistic reasoning fidelity problem. We propose a novel UQ metric, Graph Reasoning Coherence Score (GRCS), that quantifies semantic-structural consensus of the reasoning space and captures pathological mode collapse and confident hallucinations. We find that GRCS is the only metric that is consistently negatively correlated with reasoning faithfulness across both more capable and smaller models. Additionally, we introduce Graph Self-Consistency (GSC), a medoid-based decoding strategy that trades nominal accuracy for reasoning fidelity, exposing the degree to which SC is inflated by unfaithful lucky guesses in smaller models, while preserving or improving accuracy in more capable ones. Finally, through adversarial medoid ablation, we demonstrate that the GSC-selected path acts as a "load-bearing path" and forcing models away from it degrades reasoning faithfulness and, in targeted cases, causes drops in accuracy. 

---
# Tool-Making and Self-Evolving LLM Agents in Low-Latency Systems 

**Authors**: Kalle Kujanpää, Ning Liu, Shahnawaz Alam, Yeshwanth Reddy Sura, Tianyu Yang, Kristina Klinkner, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2607.08010)  

**Abstract**: Production LLM agents often waste latency and reliability by regenerating code for the same procedural steps on every request. We replace this inference-time coding loop with an agentic tool-making pipeline that compiles repeated SOP steps into validated, versioned tools before deployment. The tool-maker grounds synthesis in the live environment as it collects execution traces, observes backend schemas and values, generates candidate tools, and repairs them against labeled cases. At runtime, the production agent calls these tools directly and falls back to code generation only when needed. We deploy the approach in a Fulfillment Center alarm-triage system, where an agent diagnoses alarms against a 44-node SOP over heterogeneous metric backends. In production, tool calls reduce p50 latency by 42%. On 1,500 historical alarms, they reduce end-to-end error rate by up to 53% by suppressing run-to-run variance in repeated steps. Because tools return compact structured verdicts, they also enable a simpler direct-call architecture, reducing p50 latency by a further 62% in a controlled ablation. Versioned tools also improve auditability and expose specification gaps and upstream data drift. Our results show that self-evolving agents can make industrial LLM systems faster, more reliable, and easier to operate. 

---
# From Execution to Education: A Bloom-Aligned Framework for Measuring Educational Control in LLMs 

**Authors**: Yi Zhang, Julia Rayz  

**Link**: [PDF](https://arxiv.org/pdf/2607.08009)  

**Abstract**: We introduce a Bloom-aligned framework for measuring educational control in Large Language Models (LLMs): the ability to preserve a task's instructional intent while shifting its cognitive demand toward specified learning objectives. We apply this framework to programming tasks in computer science education to study the gap between solving tasks and adapting them for learners. Using revised Bloom's Taxonomy as an operational scale of cognitive demand, we evaluate two intervention settings: general difficulty control, where models are asked to make tasks harder or easier, and Bloom's control, where models are asked to target higher or lower Bloom's levels. We evaluate a matched Qwen3-Next model pair, comparing Qwen3-Next-80B-A3B-Instruct with Qwen3-Coder-Next across 2,520 tasks from three benchmarks. The framework reveals a robust directional asymmetry: both models reliably increase cognitive demand, but struggle to lower it. We further characterize these outcomes with semantic-delta clustering and layer-wise Fisher's Discriminant Ratio probing. Within this controlled comparison, the general model shows clearer middle-layer separability for both general difficulty and Bloom-control contrasts, whereas the coder model shows weaker separability for general difficulty and a deeper peak for Bloom-control contrasts. These results show that strong execution performance does not automatically entail Bloom-aligned educational control. 

---
# Hallucination Self-Play: Bootstrapping Reinforced Detector via Evolved Generator 

**Authors**: Shiping Yang, Shining Liang, Weihao Liu, Wenbiao Ding, Linjun Shou, Lu Cheng, Angel X. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2607.07993)  

**Abstract**: Identifying faithfulness hallucinations in LLM-generated outputs remains challenging due to the scarcity of high-quality annotated data. Recent work relies on advanced LLMs to synthesize training data, including rationales, labels, and hallucinated claims. However, these methods treat the generator as a static component, limiting iterative improvement of the detector. To address this limitation, we introduce Hallucination Self-Play (HSP), a novel framework that enables the detector to bootstrap with an evolved generator. HSP involves two roles initialized from the same base model, a detector that assesses the faithfulness of model outputs, and a generator that produces increasingly hard-to-detect hallucinated responses. Specifically, the detector is first fine-tuned on human-labeled data and then employed as a reward model to train the generator via reinforcement learning from AI feedback (RLAIF). In turn, the evolved generator synthesizes hallucination data to further optimize the detector through rule-based reinforcement learning. Experiments on RAGTruth benchmark and two model families demonstrate that the proposed framework can progressively enhance a small LLM to match or even outperform advanced LLMs without external supervision. Our code is available at this https URL . 

---
# A Reliability Assessment of LALM Audio Judges for Full-Duplex Voice Agents 

**Authors**: A. Sayyad, J. Emmons, S. Jones, T. Lin, H. Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2607.07985)  

**Abstract**: We report the empirical reliability of Gemini models as audio judges that score full-duplex agent conversations directly from the raw stereo waveform, tested across three models in the Gemini family: 2.5 Flash, 3.5 Flash, and 3.1 Pro. Our primary evidence base uses Gemini 2.5 Flash as the ground-truth model, validated against three calibrated human raters on 209 stereo sessions, scored on 8 production dimensions: 152 full-duplex conversations across 13 accent-and-condition strata, together with 57 adversarial defect-injected clips. The evidence for Gemini 2.5 Flash is consistent across three tests. (i) On 5 of 8 dimensions the LALM-human Spearman rho departs from the pairwise human-human rho by at most 0.07, and on 7 of 8 dimensions the two quantities 95 percent bootstrap confidence intervals overlap. (ii) The LALM agrees with the three-rater human mean within 1 point on 60 to 92 percent of sessions on 6 of 8 dimensions. (iii) On 45 of 48 (defect, dimension) cells the LALM is as sensitive as humans or better under Newcombe-Wilson 95 percent confidence intervals, though most of these are underpowered nulls rather than demonstrated parity. Rank-ordering ability transfers across the Gemini family: 3.5 Flash improves simple agreement to 8 of 8 dimensions, while 3.1 Pro rates several dimensions markedly lower than humans despite comparable rank correlation. A model swap should be re-validated on calibration specifically, not assumed from rank-correlation alone. We identify four areas where deployment requires care, and we estimate that human rating alone for our current evaluation cadence costs roughly two orders of magnitude more than the equivalent LALM workload. The data presented here provides a defensible empirical basis for deploying the LALM as a substitute or fourth rater on the dimensions where the evidence supports it. 

---
# When Implausible Tokens Get Reinforced: Tail-Aware Credit Calibration for LLM Reinforcement Learning 

**Authors**: Xiuyi Lou, Zicheng Xu, Yu-Neng Chuang, Hoang Anh Duy Le, Zhaozhuo Xu, Guanchu Wang, Vladimir Braverman  

**Link**: [PDF](https://arxiv.org/pdf/2607.07976)  

**Abstract**: Reinforcement learning (RL) has achieved remarkable success in enhancing the reasoning capabilities of large language models (LLMs). However, widely used critic-free RL methods rely on uniform credit assignment, broadcasting the same advantage to all tokens regardless of their differences. We identify a critical failure mode of this design, which we refer to as Positive-Credit Contamination: low-probability tail tokens that are contextually erroneous receive identical positive credit to plausible ones within the same trajectory, resulting in the indiscriminate reinforcement of flawed reasoning behavior. To mitigate this issue, we propose Tail-Aware Credit calibratiOn (TACO), a method that calibrates uniform credit assignment to suppress undesirable positive updates. TACO first computes a tail-risk score that incorporates the local generation context to assess each token's risk of falling into the unreliable tail, distinguishing unexpected rarity from uncertainty-driven exploration. TACO then uses this score to tune positive credit for risky tokens without removing their gradients entirely, so that recurring useful rare patterns can accumulate reinforcement while incidental noise is progressively dampened. Experimental results across three LLMs and eight benchmarks show that TACO consistently outperforms GRPO-style baselines. Notably, TACO improves training stability, supporting sustained performance gains in long-horizon RL. The source code is available at: this https URL. 

---
# A Multi-cluster Boundary Learning Method for Out-of-Scope Intent Detection via MiniLM Embedding 

**Authors**: Yihong Xu, Mingyu Kang, Linyuan Lü  

**Link**: [PDF](https://arxiv.org/pdf/2607.07974)  

**Abstract**: Intent detection is a critical task that bridges human intents and system actions in human-machine interaction systems. However, there still exist challenges for detecting out-of-scope (OOS) intents. (i) The traditional methods view the OOS intent detection as a multi-class classification, then the detection accuracy decreases as the class number of the known intents increases; (ii) LLM-embedding methods require large parameters, that makes them difficult to train and practically deploy. Thus, this work proposes a multi-cluster boundary learning method to detect OOS intents via MiniLM embedding (i.e., all-MiniLM-L6-v2) in an one-class classification workflow. The method learns the boundaries of multi-cluster embeddings generated by MiniLM from the training utterances, and then rejects the out-of-domain utterances as OOS intents. Experiments are conducted on public CLINC150, StackOverflow and Banking77 datasets. The results show that the method achieves the state-of-the-art OOS intent detection performance compared the other baselines. Ablation studies are also conducted and the results show that the used MiniLM can better adapt to the workflow and utterance embedding requirements. The code is available at supplementary materials. 

---
# When Debiasing Backfires: Counterintuitive Side Effects of Preprocessing-Based Stereotype Mitigation 

**Authors**: Yahan Zheng, John Guerrerio, Soroush Vosoughi, Weicheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2607.07937)  

**Abstract**: Preprocessing-based methods for stereotype mitigation, such as pre-/post-training on debiased corpora, are widely used in NLP. While these approaches reduce measurable stereotypes for targeted groups, we find they often induce unintended shifts-side effects, where stereotyping or counter-stereotyping can increase relative to neutral baselines for other demographics, including across unrelated demographic categories. We demonstrate these side effects across two model families (encoder-only and decoder-only), multiple preprocessing strategies (removing stereotypical sentences, removing group mentions, and swapping group references), and both pre- and post-training at different data scales on Wikipedia. Standard benchmarks frequently miss these shifts. Using attention-rollout analysis, we observe that such side effects are not accompanied by large changes in attention flow, complicating mechanistic explanations. We discuss implications for evaluation, provide actionable diagnostics, and argue for side-effect-aware, transparent mitigation practices. 

---
# Scalable and Culturally Specific Stereotype Dataset Construction via Human-LLM Collaboration 

**Authors**: Weicheng Ma, John Guerrerio, Soroush Vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2607.07895)  

**Abstract**: Research on stereotypes in large language models (LLMs) has largely focused on English-speaking contexts, due to the lack of datasets in other languages and the high cost of manual annotation in underrepresented cultures. To address this gap, we introduce a cost-efficient human-LLM collaborative annotation framework and apply it to construct EspanStereo, a Spanish-language stereotype dataset spanning multiple Spanish-speaking countries across Europe and Latin America. EspanStereo captures both well-documented stereotypes from prior literature and culturally specific biases absent from English-centric resources. Using LLMs to generate candidate stereotypes and in-culture annotators to validate them, we demonstrate the framework's effectiveness in identifying nuanced, region-specific biases. Our evaluation of Spanish-supporting LLMs using EspanStereo reveals significant variation in stereotypical behavior across countries, highlighting the need for more culturally grounded assessments. Beyond Spanish, our framework is adaptable to other languages and regions, offering a scalable path toward multilingual stereotype benchmarks. This work broadens the scope of stereotype analysis in LLMs and lays the groundwork for comprehensive cross-cultural bias evaluation. 

---
# How Do I Know What to Say Next? Barenholtz's Autogenerative Theory as an Enrichment of Harrisean Integrationism 

**Authors**: J. Mark Bishop, Stephen J. Cowley  

**Link**: [PDF](https://arxiv.org/pdf/2607.07891)  

**Abstract**: Roy Harris's Integrationist linguistics offers a compelling critique of the referentialist tradition embedded deep at the heart of computational approaches to language, arguing that language is not a code that maps onto a pre-given world but a situated, bipartite activity oriented toward prospective joint action. Yet Integrationism leaves certain explanatory gaps: it does not fully account for the structural mechanism by which signs sustain prospective openness, it undertheorises the continuity between linguistic and non-linguistic semiotic activity, and it offers no detailed account of the structural properties of the accumulated archive of past integrations. This paper argues that Elan Barenholtz's autogenerative theory of language, developed in response to the behaviour of Large Language Models (LLMs), can fill precisely these gaps, enriching Integrationism without undermining any of its core commitments. Specifically, the autogenerative account provides: a structural mechanism for the prospective openness that Harris identifies as central to bipartite communication; a computational correlate for Harris's thesis of semiotic continuity between language and other sign-making activity; and a theory of the archive: what the accumulated residue of past integrations looks like and how new participants draw upon it. The synthesis preserves Harris's ontological primacy of the situated integrative act while adding explanatory content that Integrationism itself does not supply. For practitioners and researchers in natural language processing and large language model design, the argument offers a principled account of what the statistical structure that LLMs so effectively exploit actually is, and of what it cannot, by its nature, provide. 

---
# DeepSearch-World: Self-Distillation for Deep Search Agents in a Verifiable Environment 

**Authors**: Xinyu Geng, Xuanhua He, Sixiang Chen, Yanjing Xiao, Fan Zhang, Shijue Huang, Haitao Mi, Zhenwen Liang, Tianqing Fang, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2607.07820)  

**Abstract**: Training tool-use agents to improve from their own experience remains challenging, as supervised fine-tuning relies on fixed teacher-distilled trajectories, while sparse-reward reinforcement learning provides weak supervision for long-horizon interactions. We present DeepSearch-Evolve, a self-distillation framework for web agents built on DeepSearch-World, a deterministic and verifiable environment with reproducible search and page-reading tools. DeepSearch-World contains 420K multi-hop QA tasks constructed from entity-level random walks and supports key agentic cognitive behaviors useful for self-evolving, including progress verification, grounded reflection, and failure recovery. DeepSearch-Evolve iteratively performs trajectory generation, filtering, data mixing, and fine-tuning to train stronger agents. Without distillation from more capable models, DeepSearch-World-9B achieves competitive performance compared with open-source agents, reaching 31.2% on BrowseComp, 61.5% on GAIA, and 93.4% on HotpotQA, showing that verifiable environments enable scalable self-evolution for long-horizon web agents. We will release the environment, 420K training pool, validation set, model, and code to facilitate future research on self-improving deep search agents. 

---
# From Solvers to Research: Large Language Model-Driven Formal Mathematics at the Research Frontier 

**Authors**: Eric Jiang, Xiao Liang, Yikai Zhang, Yingjia Wan, Mengting Li, Haikang Deng, Alexander K. Taylor, Justin Baker, Rushil Raghavan, Junyi Zhang, Ying Nian Wu, Andrea L. Bertozzi, Kai-Wei Chang, Raghu Meka, Matthew Sottile, Nanyun Peng, Amit Sahai, Terence Tao, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.07779)  

**Abstract**: Recent developments in AI for Mathematics (AI4Math), especially Large Language Model (LLM)-driven theorem provers, has achieved remarkable success in formal proof generation for well-defined mathematical problems through Interactive Theorem Proving (ITP) languages. However, current systems remain fundamentally limited in tackling frontier research mathematics, such as discovering new theorems or resolving open conjectures, which are often open-ended, under-specified, and involve multiple layers of abstraction. We argue that the next leap in AI4Math systems requires a decisive shift from predefined problem-solvers to research agents that can address frontier mathematical challenges with rigorous formal mathematical reasoning. In this position paper, we provide a systematic review of the field, covering datasets, auto-formalization, and proof synthesis. More importantly, we identify core limitations of existing systems in serving as mathematical research agents, examining issues across datasets, relational structure, mathematical exploration, tool ecosystem, and human-AI collaboration, outlining a strategic road-map for the future of AI4Math. 

---
# Unveiling Public Opinion: A Study of Sentiment Analysis Using LSTM and Traditional Models 

**Authors**: Atiq Ur Rehman  

**Link**: [PDF](https://arxiv.org/pdf/2607.07772)  

**Abstract**: In this age of social media, sites like Twitter have become meeting places for people to share their views and feelings on a wide range of issues and current events as they unfold in real time. Sentiment analysis, a critical application of NLP, has become indispensable due to the massive influx of user-generated content, enabling the extraction of meaningful insights from the opinions and emotions expressed in textual data. Sentiment analysis on Twitter employs sophisticated computational techniques to categorize tweets into positive, negative, or neutral sentiments. This method not only examines individual expressions but also analyzes vast databases related to specific subjects or events. By spotting these emotions, machine learning models help improve public opinion interpretation and trend forecasting. This paper examines the effectiveness of various machine learning and deep learning approaches. Designed for this use, the system evaluates logistic regression, random forest, naïve bayes, gradient boosting, and LSTM networks, among other algorithms applied in sentiment classification. This work identifies the optimal sentiment analysis model using a Kaggle Twitter dataset that has been preprocessed through tokenization, lemmatization, and stopword elimination. Emphasizing the better performance of the LSTM approach, the model attained a training accuracy of 90.98%, a testing accuracy of 80.00%, and a micro-average ROC- AUC score of 0.92. These results show that the model outperforms conventional machine learning techniques in capturing contextual and sequential textual aspects. 

---
# Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents 

**Authors**: Yifan Wu, Lizhu Zhang, Yuhang Zhou, Mingyi Wang, Bo Peng, Serena Li, Xiangjun Fan, Zhuokai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.08716)  

**Abstract**: In long-horizon tasks, decision-relevant state is often scattered across an expanding trajectory, while the action agent must surface it and act. As trajectories grow, task requirements, environment facts, prior attempts, diagnoses, and open subgoals can be buried in the context window or pushed beyond it, failing to influence decisions when needed. We call this failure mode "behavioral state decay". We study memory as an active intervention mechanism rather than passive retrieval. A separate memory agent runs alongside an unmodified action agent, updating a structured memory bank from the recent trajectory and deciding whether to inject a memory-grounded reminder or remain silent. The module is plug-and-play with frontier action agents and existing agent harnesses. Across Terminal-Bench 2.0 and $\tau^2$-Bench, it improves pass@1 for both weaker and stronger action agents, with gains of +8.3 pp on Terminal-Bench and +6.8 pp on $\tau^2$-Bench. Ablations show that selective intervention outperforms passive bank exposure, always-on injection, advisor-only guidance, and general retrieval. As an early step toward open-weight memory policies, we train Qwen3.5-27B on SETA using SFT and GRPO, improving validation reward and achieving partial transfer to Terminal-Bench. 

---
# The complexities of patient-centred conversational artificial intelligence 

**Authors**: João Matos, Olivia Buege, Donny Cheung, Gary S. Collins, Paula Dhiman, Nan Li, Bingyu Mao, Benjamin W. Nelson, Michail Ouroutzoglou, Paul Varghese, Jonathan Amar  

**Link**: [PDF](https://arxiv.org/pdf/2607.08625)  

**Abstract**: Consumer-facing health chatbots powered by large language models (LLMs) are increasingly used for symptom assessment. However, chatbot development and evaluation often rely on cooperative, articulate, simulated patients. We analysed 2,053 real patient-chatbot conversations and found that communication patterns and expression of emotions vary widely across users. We developed a patient simulator that separately models clinical content, emotional state, conversational strategy, and communication style. In a Turing-inspired evaluation of realism with 15 human graders, simulated conversations were nearly indistinguishable from real ones, with human graders achieving an accuracy of 55%. We used five distinct patient personae, across 1,164 clinician-graded cases, to evaluate the performance of four LLMs in urgency assessment. We found that communication style can significantly alter triage outcomes. Patient-centred conversational artificial intelligence must accommodate communication diversity: systems designed for idealised, rather than realistic, interactions risk underperforming and amplifying health disparities when deployed in the real world. 

---
# Improving Ad-hoc Search Effectiveness for Conversational Information Retrieval via Model Merging 

**Authors**: Ahmed Rayane Kebir, Jose G. Moreno, Lynda Tamine  

**Link**: [PDF](https://arxiv.org/pdf/2607.08540)  

**Abstract**: Conversational information retrieval is challenging since it requires the consideration of the conversation history which potentially gives rise to topic shifts and coreference resolution across previous turns. To address these challenges, previous work mainly rely on traditional fine-tuning of ad-hoc retrievers on conversational datasets or extrapolates their generalizability through multi-tasking. However, this mainstream approach is costly - since it requires model re-training - and exhibits catastrophic forgetting, where the model loses its foundational ad-hoc retrieval performance. In this paper, we fill this gap by introducing model merging as a training-free strategy enabling the design of a single retrieval model that operates across both ad-hoc and conversational settings with no additional fine-tuning. We conduct experiments using linear and non-linear parameter-wise merging strategies - namely Model Soup and Slerp - on standard ad-hoc search and conversational retrieval datasets. Our results demonstrate that model merging significantly enhances the ad-hoc search capabilities of conversational retrievers while improving generalizability across task-specific datasets, achieving up to 15% higher NDCG@3 under zero-shot conditions. 

---
# Cognitive-structured Multimodal Agent for Multimodal Understanding, Generation, and Editing 

**Authors**: Feng Wang, Canmiao Fu, Zhipeng Huang, Chen Li, Jing Lyu, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.08497)  

**Abstract**: Recent unified multimodal models show a single architecture can jointly perform vision/language understanding and image generation/editing. However, they repeatedly feed all historical visual and textual inputs into a shared context window, limiting long-horizon multimodal dialogue due to visual token explosion and unreliable cross-turn referencing. We propose a Cognitive-structured Multimodal Agent that externalizes visual information into an Episodic Visual Memory and selectively reactivates relevant episodes during reasoning. The agent consists of a Perceptual Abstraction Engine for structured visual abstraction, a Cognitive Retrieval Engine for cross-turn memory retrieval, and a Multimodal Executive Controller for autonomous task inference and action planning. To address the lack of turn-level retrieval supervision in existing datasets, we develop a Unified Scenario Engine that programmatically generates structured multi-turn conversations with fine-grained retrieval annotations, enabling reinforcement learning to optimize abstraction and retrieval policies. We also construct a long-horizon visual-dialogue benchmark stratified by difficulty to evaluate episodic visual recall. Our 8B agent achieves 91.4% retrieval accuracy over 20-turn sessions, surpassing 32B baselines by +8.2% while nearly halving per-turn inference time (23.1s -> 12.7s). We further present the Cognitive-structured Multimodal Agent Harness (CMA-Harness), a tool-augmented deployment of the same cognitive structure integrating persistent multimodal memory, web access, image generation/editing/composition tools, and OpenAI-compatible serving. Structured memory and modular decision-making offer a more scalable, efficient paradigm for long-horizon multimodal agents than monolithic parameter scaling. Code: this https URL ; Project page: this https URL 

---
# Ensemble Diversity Optimization for Subjective Supervision 

**Authors**: Xia Cui, Ziyi Huang, N. R. Abeynayake  

**Link**: [PDF](https://arxiv.org/pdf/2607.08493)  

**Abstract**: Subjective NLP tasks often exhibit systematic annotator disagreement, requiring models that represent uncertainty rather than collapse it. We introduce Ensemble Diversity Optimization (EDO), a prediction-space framework that jointly optimizes ensemble weights, effective cardinality, and calibration through a unified differentiable objective. EDO learns ensemble composition and size end-to-end via Gumbel-Softmax relaxation and incorporates a signed diversity regularizer, tuned on validation data, to steer optimization toward either preserving or suppressing disagreement. This regularization prevents ensemble collapse and enables controlled navigation of the utility-calibration trade-off. The framework integrates a soft F1 surrogate, class-weighted cross-entropy to address imbalance, and reliability-weighted diversity to regulate intra-ensemble variability. Experiments on four subjective text-classification benchmarks (ArMIS, ConvAbuse, HS-Brexit, MD-Agreement) show that EDO substantially improves probabilistic calibration, reducing cross-entropy (40-78% depending on baseline) and lowering Brier scores relative to Soft-CE, Soft-MD, Top-5 Voting, and WEL, while maintaining competitive F1 and better alignment with annotator distributions. These results demonstrate that jointly optimizing ensemble structure with a signed diversity regularizer provides an efficient, model-agnostic approach for modeling human subjectivity in supervised learning. 

---
# Token-Flow Firewall: Semantic Runtime Auditing for Persistent AI Agents 

**Authors**: Puji Wang, Yingchen Zhang, Ruqing Zhang, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.08395)  

**Abstract**: Persistent AI agents extend large language models (LLMs) beyond single-turn interaction into long-lived software systems. Unlike traditional chat assistants, unsafe content in these agents can propagate through persistent state, reusable skills, and tool-mediated interactions, creating a substantially larger semantic attack surface. We observe that most security-critical interactions in such agents are transmitted through natural-language token flows, including memory updates, tool arguments, retrieved files, and inter-component communications. This observation enables a new security formulation: unsafe behavior can be intercepted as risky semantic flows before reaching privileged runtime sinks. Based on this insight, we propose TokenWall, a runtime defense framework that acts as a semantic firewall over agent token flows. TokenWall performs boundary-aware semantic auditing over these flows, constructing structured source-sink audit records, applying lightweight local inspection before execution, and selectively escalating ambiguous high-risk cases to stronger arbitration modules. Unlike prior approaches that rely on sparse auditing or remote large-model oversight, TokenWall enables full-coverage pre-execution mediation while reducing remote arbitration and latency. Experiments on CIK-Bench show that TokenWall reduces attack success rate to 12.5% while maintaining a 97.4% benign executable pass rate without human confirmation. TokenWall further introduces only 0.69 seconds of additional latency on benign cases, demonstrating that semantic runtime containment can achieve a practical security-utility trade-off for persistent AI agents. 

---
# Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning 

**Authors**: Lu Dai, Ziyang Rao, Yili Wang, Hanqing Wang, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2607.08393)  

**Abstract**: Fine-tuning LLMs to inject new knowledge faces a critical challenge: LLMs can quickly memorize new facts, yet fail to use them for downstream reasoning tasks. We formalize this failure as the \textit{\textbf{Knowing--Using Gap}}, characterized by an accuracy gap and a temporal lag between memorization and generalization. To understand this phenomenon, we fine-tune LLMs with unseen knowledge and monitor the spatial permeation dynamics of the knowledge internally using a novel intervention technique called self-patching. Self-patching identifies activation locations where relocating representations substantially improves failed generalization cases. These results are consistent with a knowledge-circuit misalignment hypothesis: memorized representations can exist internally but may not be routed to computation-effective layers. To demonstrate the practicality of this diagnostic finding, we design a simple heuristic strategy which recovers 58--75\% of the oracle headroom in generalization failure. Experiments are done cross-domain for the robustness of this finding. 

---
# Different Teachers, Different Capabilities: Sub-1B On-Device Distillation for Structured Text Enrichment 

**Authors**: Vinay Kumar Chaganti  

**Link**: [PDF](https://arxiv.org/pdf/2607.08268)  

**Abstract**: High-volume structured extraction pays a large model's latency on every item, so distilling the task into a small on-device model is attractive: comparable output at a fraction of the time and cost. We measure what that distillation actually delivers, per sub-task. Each news article is mapped to one JSON object with a short summary and five categorical labels. We distill an 8B reasoning teacher (deepseek-r1:8b) into a 0.6B student (Qwen3-0.6B; QLoRA, three seeds), and add two teacher controls: a same-size non-reasoning teacher and a larger managed pipeline. A blinded, reference-free, three-judge panel scores every arm against the full article, alongside two non-distillation baselines, few-shot prompting and constrained decoding. The student runs at about 0.8 s per article against the teacher's 39 s, and recovers 58% of the base-to-teacher gap on summary quality, beating its primary baseline (constrained decoding) by +16.8 points and few-shot prompting by a secondary +4.9. A same-size non-reasoning teacher trains a student no better than the untuned base, so the summary gain follows from the teacher's reasoning nature rather than its scale. Capabilities then split by teacher: the reasoning teacher transfers writing quality and the managed pipeline transfers label diversity, while a same-size instruction teacher's students stay more grounded on the 22 short, thin-source articles in the 93-item test set (74 versus 55 faithful), where the reasoning-lineage student fabricates. That grounding difference is a consistent ordering rather than a significant aggregate effect, and the subgroup is small, so we report it as a direction. Because no single engine wins every field, the deliverable is a per-field routing map for on-device enrichment. 

---
# AutoPersonas: A Multi-Timescale Loop Engine for Open-Ended Persona Evolution 

**Authors**: Mengchen Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.08252)  

**Abstract**: Long-term persona agents must remain identifiable while adapting to new events, relationships, evidence, and social conditions. We identify self-locking as a runtime failure mode in continuing persona-life loops: locally plausible events keep appearing while the generated life collapses toward familiar environments, weak relationships, suspended decisions, and stale life stages. We trace this failure to model-level convergence toward high-probability behavioral channels and system-level context gravity from State, memory, history, and environment summaries. We introduce AutoPersonas, a multi-timescale life-environment engine for bounded persona-level recursive self-evolution. It separates environment-side Occurrences, accumulated Observations, and persona State. Its OSO loop admits divergent future-facing material while requiring evidence-governed absorption before State or reachability changes. A three-year compressed simulation exposed environment watermark shells, occurrence-hardening gaps, slow-change accumulation failures, recursive indecision, and weak relationship persistence. An eight-model 40-day stress test generated 1,600 events and found mean rolling 5-day action-category repetition of 95.2%-97.6%, with all models crossing 90% by day 11. Semantic re-keeping found 79.0%-88.0% macro-theme repetition across all direct-loop runs. In a same-runtime 40-day A/B, context-slice masking plus per-sample divergence targeting reduced macro-theme repetition from 61.8% to 36.3% and roughly doubled cumulative theme count. A juvenile-goblin fictional-world run reproduced the anti-fixation regime without hard real-world intrusions. These results support a bounded claim: separating controlled divergence from evidence-governed absorption can reduce persona-environment self-locking while preserving identity continuity. 

---
# A First-Principles Theory of Slow Thinking and Active Perception 

**Authors**: Hongkang Yang, Zhi-Qin John Xu, Feiyu Xiong, Weinan E  

**Link**: [PDF](https://arxiv.org/pdf/2607.08196)  

**Abstract**: As part of a series on first-principles modeling of cognitive functions, this paper attempts to provide a mathematical formulation of thinking and perception. It formally derives slow thinking or more generally, active perception, and encompasses the design, training and inference of slow thinking large language models. Our starting point is the lifting and projection of probability distributions on the observable and latent spaces, with the objective of representing complex data distributions by simple function families such as neural networks. A theory called "active lifting" is proposed, based on the sampling of latent sequences and an intrinsic drive to reduce uncertainty with maximum rate. It derives a large design space, containing the slow thinking models in a subspace that we call the static theory. These models are positioned on the representation hierarchy and sampler hierarchy induced by the static theory, and can be upgraded by climbing the two hierarchies. Active lifting further derives an inference process with an internal time axis, and a training objective that resembles minimum-length coding as well as the invention of languages. Thus, it characterizes the agency of perception, including the emergence of the slow thinking formats. Technical by-products of this theory include a three-stage pathway for improving slow thinking models, a unified approach to constructing encoders and generative models for all data modalities, a priori formation of human-like visual representations, and a possible solution to policy collapse. 

---
# CausalDS: Benchmarking Causal Reasoning in Data-Science Agents 

**Authors**: Andrej Leban, Yuekai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.08093)  

**Abstract**: Large language models (LLMs) increasingly act as integrated data-science agents, combining abstract reasoning with advanced tool use. Yet the relevant benchmark landscape largely divides into symbolic causal reasoning benchmarks without realistic data analysis or data analysis benchmarks without a principled causal data-generating structure. Furthermore, existing causal evaluation datasets are often restricted to curated examples from existing sources, with diversity coming from limited templatized variations rather than from systematic generation of novel synthetic causal structures. We introduce CausalDS, a benchmark for evaluating causal reasoning in agentic data-science workflows. Each benchmark instance is a scene consisting of a sampled structural causal model (SCM) with generated observational data and an accompanying synthetic natural-language story grounded in a realistic domain. We optionally ground the composition of the benchmark components in empirical distributions obtained from real-world datasets, thus retaining empirical structure while reducing the "causal parrot" risk through completely synthetic generation. From each scene, we then derive tasks spanning all three of Pearl's rungs, with typical data-science prediction tasks appearing as Rung 1. Most tasks include a data science coding component, where the model typically needs to use several tools to arrive at the final answer due to the frequent presence of imperfect observations, which are generated by an observation model. Additionally, recognizing when a question admits no warranted answer and abstaining is treated as a first-class scored outcome. The benchmark thus jointly evaluates symbolic causal reasoning, data science, uncertainty quantification, abstention, and tool use/coding. 

---
# Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization 

**Authors**: Jiantong Jiang, Peiyu Yang, Rui Zhang, Feng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.08057)  

**Abstract**: Despite the rapid advancements of large language models (LLMs), LLM serving systems remain memory-intensive and costly. The key-value (KV) cache, which stores KV tensors during autoregressive decoding, is crucial for enabling low-latency, high-throughput LLM inference serving. In this survey, we focus on system-aware KV infrastructure for serving LLMs (abbreviated as sKis). We revisit recent work from a system behavior perspective, organizing existing efforts into three dimensions: execution and scheduling (temporal), placement and migration (spatial), and representation and retention (structural). Furthermore, we analyze cross-behavior co-design affinity and behavior-objective links, highlighting future opportunities. Our work systematizes a rapidly evolving area, providing a foundation for understanding and innovating KV cache designs in modern LLM serving infrastructure. 

---
# From Prompts to Contracts: Harness Engineering for Auditable Enterprise LLM Agents 

**Authors**: Joongho Ahn, Moonsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.08028)  

**Abstract**: Enterprise large language model (LLM) applications often begin as prototypes whose behavior is carried by prompts and retrieval context. Productization adds requirements for source boundaries, entity routing, answer contracts, and reproducible traces. We present a harness-engineering approach that reconstructs this pattern into a traceable, auditable LLM-agent architecture: deterministic behavior moves into code, manifests, schemas, and validation artifacts around a replaceable composition boundary, while source-backed claims remain the authority for runtime answers. We instantiate it on a public-data slice of five Korean corporate groups (25 listed companies) and evaluate three research questions. (1) The harness preserves its source-grounding, entity-routing, trace, output-hygiene, and recommendation-language contracts across the fixed validation scenarios; a fault-injection control confirms the validators flag deliberately broken contracts. (2) The checks the harness enforces held under model substitution: across three hosted models, they passed on all 270 composition-boundary runs; failures were confined to the model-composed side and were caught and recorded. (3) The code-owned guarantees are load-bearing, not reproducible by prompting alone: holding the model fixed and varying only the enforcement layer, prompt instructions alone let recommendation-language and internal-trace-leakage violations reach the reader, which the harness blocks entirely. A bolt-on external guardrail prevents such violations too but over-refuses, dropping utility to 88/120 where the harness preserves full utility (120/120); in this ablation, only code-owned enforcement preserves both safety and utility. The result is a reusable engineering pattern for turning exploratory prototypes into auditable applications with versioned source, control, and validation artifacts. 

---
# fog: Expressing Motion and Emotion through Function Composition of AI-Generated Code 

**Authors**: Vivian Liu, Lydia Chilton  

**Link**: [PDF](https://arxiv.org/pdf/2607.07952)  

**Abstract**: Motion and emotion are core parts of intelligent, expressive behavior. In this paper, we introduce fog, a function composition framework for implementing and compose motion functions. We demonstrate how fog can be used to express motion and emotion in Heider-Simmel style animations. This code generation framework can help users generate functions for verbs, adverbs, gestures, and emotions to create an open-ended motion vocabulary. It is complemented by an animation editor that helps users refine motion through direct manipulation and dynamically generated UI. We evaluate our approach with a perceptual evaluation, where we test 452 fog-generated animations to see if people can recognize the semantic meaning of the motion. We find that fog's motion functions can be recognized at 68% accuracy, a 2.68x improvement over a chance baseline. In a mixed-methods user study with professionals and novices, we show that fog in interface form can support users with more rapid iteration, exploration, and control. 

---
# The Memory Wall of Green Software: Empirical Energy Evaluation of Memento Design Pattern 

**Authors**: Imane Jriri, Tarik Houichime, Younes El Amrani  

**Link**: [PDF](https://arxiv.org/pdf/2607.07944)  

**Abstract**: As Green Software Engineering matures, energy efficiency has transitioned into a mission-critical non-functional requirement. While software design patterns ensure structural integrity, their inherent abstraction layers impose an implicit "metabolic cost" that often remains obscured during the design phase. This paper empirically investigates the energy dynamics of the Memento design pattern, contrasting a direct, unabstracted baseline against Classic full-snapshot and Differential delta-encoding strategies. Leveraging the RAPL interface for high-fidelity hardware telemetry, we quantify energy dissipation across state volumes scaling from 10 MB to 200 MB. Our empirical results expose a critical architectural trade-off: the Differential strategy minimizes memory traffic, yielding a maximum energy reduction of 65.8% for mid-scale states, but collides with a catastrophic "memory wall" at 200 MB. At this saturation point, algorithmic optimizations are completely neutralized by severe GC thrashing and non-linear power spikes. We synthesize these findings into evidence-based heuristics, providing architects with a robust framework to reconcile structural design quality with sustainable Green IT imperatives. 

---
# Efficient Safety Alignment of Language Models via Latent Personality Traits 

**Authors**: Mohamed Amine Merzouk, Nolan Smyth, Damiano Fornasiere, Linh Le, David Williams-King, Adam Oberman  

**Link**: [PDF](https://arxiv.org/pdf/2607.07918)  

**Abstract**: Current safety methods for large language models are known to be vulnerable to adversarial attacks, motivating research into robust alternatives. Latent Adversarial Training (LAT) is among the most effective defenses, but can degrade utility and requires training on large datasets of harmful prompts. We introduce Latent Personality Alignment (LPA), which replaces explicit harm refusal with adversarial training on just 66 harm-agnostic statements drawn from psychometric personality literature. We hypothesize that personality-anchored representations share latent structure with harm avoidance, so adversarially stabilizing them implicitly constrains the subspace exploited by jailbreak attacks. LPA achieves near-zero attack success rates on HarmBench across direct requests and five jailbreak methods, despite never seeing harmful content during training and no loss of performance on standard benchmarks. Moreover, the training process is lightweight; the entire procedure completes in minutes on a single GPU and uses 75x fewer examples than standard LAT. Extensive ablations demonstrate the robustness, efficiency, and generalization of our method. 

---
# Validating LLMs in social science: Epistemic threats and emerging norms 

**Authors**: Meera Desai, Dallas Card, Abigail Z. Jacobs  

**Link**: [PDF](https://arxiv.org/pdf/2607.07915)  

**Abstract**: Large language models (LLMs) are reshaping social science methodology. Researchers increasingly prompt language models to generate quantitative measurements of social concepts, for example labeling data or simulating survey responses. Yet LLMs pose methodological challenges including bias, hallucination, and brittleness across contexts, with unclear threats to validity. Standard practices and norms for addressing these challenges are still emerging. We collect and systematically analyze validation practices in a comprehensive corpus of papers from eight flagship social science journals that use LLMs as measurement instruments. We find that LLM-generated measurements frequently play a central role in empirical analyses, yet validation practices are inconsistent and limited. We outline complementary strategies for more robust validation, pointing toward better norms and standards around the use of LLMs in social science. 

---
# Multimodal Unlearning Across Vision, Language, Video, and Audio: Survey of Methods, Datasets, and Benchmarks 

**Authors**: Nobin Sarwar, Shubhashis Roy Dipta, Zheyuan Liu, Vaidehi Patil  

**Link**: [PDF](https://arxiv.org/pdf/2607.07907)  

**Abstract**: With the growing adoption of VLMs, DMs, LLMs, and AFMs, these multimodal foundation models can inadvertently encode sensitive, copyrighted, biased, or unsafe cross-modal associations that originate from their training data. Retraining after deletion requests or policy updates is often impractical, and targeted forgetting remains difficult because knowledge is distributed across shared representations. Multimodal unlearning addresses this challenge by enabling selective removal across modalities while retaining overall utility. This survey offers a unified, system-oriented view of multimodal unlearning across vision, language, audio, and video, grounded in recent advances, emerging applications, and open problems. Our taxonomy enables systematic comparison across model architectures and modalities, clarifying trade-offs among deletion strength, retention, efficiency, reversibility, and robustness. This survey highlights open problems and practical considerations to support future research and deployment of multimodal unlearning. We release a curated repository: this https URL 

---
# Collective Intelligence with Foundation Models 

**Authors**: J. de Curtò, I. de Zarzà  

**Link**: [PDF](https://arxiv.org/pdf/2607.07729)  

**Abstract**: As foundation models grow in scale and diversity, coordinating multiple models into cooperative reasoning systems offers a path toward safer, more reliable AI. This chapter presents a multi-agent framework where solver models generate independent drafts, each undergoes structured critique and revision by a critic agent, and an aggregator agent synthesizes a final consensus solution. A scoring module provides semantic, numerical, and procedural evaluation across all agents. Through ablation studies on a benchmark spanning calculus, physics, chemistry, biology, economics, optimization, statistics, and mathematics, we isolate the contributions of framework architecture versus model diversity. We compare four configurations: (1) Individual Baseline, (2) Homogeneous Framework using one shared model, (3) Redundant Homogeneous Solvers using multiple instances of the same model, and (4) Heterogeneous Framework with diverse specialized models. Results show that while framework structure and redundant sampling yield modest gains, model heterogeneity is the critical factor driving substantial performance improvements. The heterogeneous configuration achieves superior step-wise accuracy (0.64 vs. 0.54 for individual models; 2.3x improvement over homogeneous configurations) with reduced variance across categories and difficulty levels. Step-wise reasoning quality (correctness of intermediate steps, not just final answers) improves dramatically only with model diversity, showing that heterogeneous agents provide complementary error detection and reasoning refinement essential for explainability and auditability. We discuss architectural principles, evaluation methodology, and implications for Global Applied AI, showing how heterogeneous multi-agent coordination supports transparent, auditable, high-confidence decision-making across scientific and industrial domains. 

---
# SPL: Orchestrating Workflows with Declarative Deterministic-Probabilistic Composition 

**Authors**: Wen G. Gong  

**Link**: [PDF](https://arxiv.org/pdf/2607.07727)  

**Abstract**: We present SPL (Structured Prompt Language), a declarative language that composes deterministic and probabilistic computation modes in a single specification. While existing frameworks separate these -- orchestration systems (AutoGen, CrewAI, LangGraph) for LLM calls, symbolic tools (SymPy, SageMath, Lean) for computation -- SPL unifies them. It provides GENERATE/EVALUATE for probabilistic computation and SOLVE/ASSERT for deterministic computation, sharing syntax, variable bindings, and runtime routing. A .spl specification runs unchanged across local nodes (Ollama), cloud APIs (OpenRouter, Anthropic), and distributed grids (Momagrid), with model and verifier selection deferred to invocation time.
We validate SPL through an extensive 78-recipe cookbook and a controlled 1,200-run experiment (10 models x 20 problems x 2 arms x 3 repetitions; the 20 problems span 6 difficulty tiers). The solver arm achieves 82-93% machine-verified correctness (sonnet-4-6: 85%, gemma4:e2b: 93%) while the LLM-only arm measures output production without mathematical verification, making the comparison one of verified correctness against unverified fluency. A backend difficulty gradient emerges (SymPy 78%, Sage 54%), and the dominant failure mode is solver_error (kernel-rejected expressions), not format non-compliance. 

---
# Uncertainty-gated selection for block-sparse attention 

**Authors**: Thomas Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2607.07724)  

**Abstract**: Block-sparse attention scales long-context language models by replacing the O(N^2) softmax with a per-query top-k selection over key blocks. This cutoff is myopic: when the k-th and (k+1)-th blocks are nearly tied in score, the selector commits without spending extra budget, and a dropped block carrying answer evidence is unrecoverable downstream. We propose a value-of-information router that measures, for each query, how decisively the top-k cut was made, and doubles the kept set for the queries where that gap is smallest; the rule is backbone-agnostic and stacks with existing block-scoring methods such as Quest. On LongBench-v2 medium at n=215 (the entire dataset subset), router-on-Quest reaches paired recall 0.75 vs. top-k 0.47 -- +28 pp over the SSA-style baseline (McNemar p<0.01) -- and lands within 2 pp of dense on RULER NIAH multikey at the same context. The lift reproduces on four models from three architectures (Qwen2.5, Mistral-Nemo, Qwen3.6). At 128K, the router preserves 0.81 and 0.89 of dense accuracy on Qwen2.5-7B-1M and Qwen3.6 (vs. SSA-style top-k at 0.09 on the former) while the fused selection-plus-kernel pipeline runs at 0.62x and 0.80x dense wall time. 

---
