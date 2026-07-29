# Pass the Baton: Trajectory-Relayed On-Policy Distillation 

**Authors**: Haolei Xu, Xiaowen Xu, Haiwen Hong, Zixuan Ni, Hongxing Li, Yiwen Qiu, Weiming Lu, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2607.26057)  

**Abstract**: On-policy distillation (OPD) grounds token-level supervision in the student's own trajectory, yet suffers from prefix failure: once the student commits to a wrong reasoning direction, all subsequent generation builds on this deviation, producing misdirected continuations that elicit unreliable supervision and waste compute. We identify a teacher-student continuation asymmetry on failed prefixes, where the teacher tends to redirect while the student continues along the original direction, and convert it into a label-free handoff trigger in Relay On-Policy Distillation (Relay-OPD). During training, Relay-OPD constructs relay trajectories by letting the teacher briefly take over at detected trigger points to produce a teacher leg, after which the student resumes and is optimized on the resulting trajectory. A limited relay budget concentrates intervention on critical early positions while limiting departure from the student policy. With a Qwen3-4B-Instruct-2507 teacher and Qwen3-0.6B/1.7B-Non-Thinking students on eight mathematical reasoning benchmarks, Relay-OPD achieves the best or second-best results on every benchmark, outperforming standard OPD by +5.73% and the strongest baseline FastOPD by +1.49% on average for 1.7B, with consistent gains at 0.6B. Training trajectory length is reduced by over 50%. 

---
# UniMem: Complementary Episodic-to-Parametric Memory for Boundary-Agnostic Task Streams 

**Authors**: Siyu Xia, Chenheng Zhang, Yanting Wu, Haoxuan Li, Jiajun Chai, Xiaohan Wang, Guojun Yin, Wei Lin, Zhouchen Lin, Haifeng Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.26017)  

**Abstract**: Memory is essential for LLM agents to accumulate task experience and reuse task-specific execution strategies. However, real-world deployment over boundary-agnostic and evolving task streams exposes a fundamental stability-plasticity dilemma. External retrieval-based memory can rapidly absorb new evidence, but it often fails to internalize recurring execution patterns and incurs inference-time retrieval overhead. Parametric memory enables stable and efficient execution once learned, but typically relies on explicit task boundaries and fixed parameter budgets. Inspired by the human brain, which balances plasticity and stability through complementary episodic storage and gradual consolidation, we propose UniMem, a self-routing framework for autonomous memory management. UniMem uses learnable routing tokens as memory controllers, enabling adaptive coordination between complementary memory pathways: novel or sparse tasks are retained in an episodic buffer for retrieval-augmented execution, while recurring and reliable patterns are consolidated into expandable parametric memory. By decoupling task identification from task execution with routing tokens and parametric memory blocks, UniMem expands memory on demand without task labels during deployment or uncontrolled parameter growth. Experiments on long-horizon streaming task sequences show that UniMem consistently outperforms baselines while maintaining execution fidelity, achieving an average gain of 4.0 EM points across three backbone models. 

---
# Instruction-Tuned Models Locally Reuse Human Syntax More Than Humans Do 

**Authors**: Zandi Eberstadt  

**Link**: [PDF](https://arxiv.org/pdf/2607.26015)  

**Abstract**: Syntactic convergence (the tendency of speakers to adapt in language towards the grammatical profiles of their interlocutors) is a well-documented feature of human dialogue widely considered to operate below conscious awareness. Whether large language models exhibit analogous syntactic convergence toward human users relative to human baselines and across a broad range of syntactic constructions remains an open question. Using substitution-paradigm data in which model generations replace one speaker's turns in pre-existing human dialogues, this study measures turn-adjacent reuse of context-free grammar (CFG) rules across sixteen open-weight Llama and Gemma models (1B-70B, pretrained and instruction-tuned) at 1,901 matched positions per model. Every model showed greater CFG-rule overlap with the preceding human turn than with a sampled unrelated human prime, and in every model this actual-versus-random difference was larger for lower-frequency rules. Each instruction-tuned model also showed greater natural-output overlap with the actual prime than the human response it replaced, and all eight matched architecture pairs exhibited greater actual-prime overlap after instruction tuning. However, relative to pretrained variants, instruction-tuned outputs overlapped more with unrelated primes, showed a smaller actual-versus-random increment, and had lower conditional rule-reuse odds once target rule-set size was held constant. In exploratory analyses, each model exhibited greater mean lexical and semantic similarity to the preceding turn than the matched human responses did. Instruction-tuned models additionally produced responses with greater mean semantic similarity than their pretrained counterparts in all eight architecture pairs, whereas the lexical similarity results were more heterogeneous. 

---
# Detecting Knowledge Inconsistencies Across Text, Tables, and Knowledge Graphs 

**Authors**: Fanfu Wei, Thibault Ehrhart, Raphaël Troncy  

**Link**: [PDF](https://arxiv.org/pdf/2607.25959)  

**Abstract**: Wikipedia and Wikidata are widely used for information access, LLM pre-training, and retrieval-augmented generation. Their knowledge is deeply connected but scattered across text, tables, and knowledge graphs. This raises a practical question: when these modalities disagree, how can we detect and explain the conflict? We study this problem as \emph{modality-level inconsistency detection}. We first introduce a taxonomy of cross-modal knowledge inconsistencies, covering information granularity differences, direct conflicts, temporal changes, and KG incompleteness. We then present \textsc{Kontrast}, an automatic framework that uses Text-to-SPARQL and LLM reasoning to compare table-based answers with KG evidence and categorize the resulting inconsistencies. Experiments on various Table-QA datasets show that cross-modal inconsistencies are common and informative. They reveal not only true knowledge conflicts, but also missing KG structure and temporal mismatches while being limited by Text-to-SPARQL errors and noise. Our analysis shows that text, tables, and KGs can complement and correct one another through systematic comparison. \textsc{Kontrast} provides a practical tool for large-scale knowledge auditing and establishes a benchmark for future work on cross-modal knowledge consistency. Code and data are available at this https URL. 

---
# Polistemics: Evaluating LLMs as Information Mediators in Politics & Elections 

**Authors**: Baran Peters  

**Link**: [PDF](https://arxiv.org/pdf/2607.25953)  

**Abstract**: As LLMs increasingly mediate the political information citizens rely on, there is still no standardized way to assess whether they do so responsibly. We introduce Polistemics, a theory-grounded benchmark for evaluating LLMs as mediators of political information in elections. Prior work has treated this task as reproduction rather than mediation, leaving its epistemic dimensions and interaction with imperfect information unaddressed. We ground the evaluation in Epistemic Modesty, a normative standard derived from citizens' epistemic agency, and test it across controlled settings that vary informational properties such as clarity, noise, and consistency. Applying the benchmark to three state-of-the-art LLMs on the 2025 German and Dutch elections, we find that high aggregate scores mask systematic failures. Models mediate reliably under clear evidence but break down under absent, vague, or contradictory information, while flattening the intensity of political language. These failures are likely driven by party priors, influenced by party labels and output language. Reliable mediation appears achievable, but no model delivers it consistently. 

---
# Evaluating Multi-Turn Multimodal Diagnostic Reasoning on Challenging Real-World Clinical Cases 

**Authors**: Rui Yang, Weihao Xuan, Yi Lin, Zhuhan Bao, Jonathan Chong Kai Liew, Matthew Yu Heng Wong, Nicolás Lescano, Nikita R. Paripati, Emily Ling-Lin Pai, Jiarui Liu, Heli Qi, Heng-Jui Chang, Benny Kai Guo Loo, Huitao Li, Kunyu Yu, Yufan Wang, Chuan Hong, Shijian Lu, Douglas Teodoro, Naoto Yokoya, Ross Koppel, Mona Diab, Hua Xu, David W. Bates, Nan Liu, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2607.25933)  

**Abstract**: Clinical diagnostic evaluation should not only assess whether models can provide correct diagnoses, but also reflect the realities of clinical practice, including progressive disclosure of multimodal information, dynamic updating of diagnostic hypotheses, and continuous refinement of clinical reasoning. However, existing evaluations of multimodal large language models (MLLMs) typically rely on single-turn or isolated tasks, making it difficult to fully capture the complexity of real-world clinical diagnosis. To bridge this gap, we developed ClinMM-Bench, the largest multi-turn multimodal clinical diagnostic evaluation benchmark to date. ClinMM-Bench contains 1,089 challenging real-world clinical cases and 3,760 medical images across eight specialties. We systematically evaluated 15 representative MLLMs using a two-level evaluation framework that assessed both diagnostic accuracy and diagnostic reasoning quality. Results showed that proprietary models achieved the highest overall diagnostic accuracy, but the proportion of completely correct diagnoses remained limited across all models. In terms of diagnostic reasoning quality, current models can identify plausible diagnostic directions but still have considerable limitations in generating reliable diagnostic reasoning. Error analysis further identified five representative failure modes: information synthesis failure, knowledge mapping error, perception error, premature closure, and visual hallucination. 

---
# AI's Capability in Assisting Scientific Research in Physics, Astrophysics, and Cosmology II: Project Planning and Proposal Evaluation 

**Authors**: Jia Liu, Veena Krishnaraj, Kateryna Vovk, Kosuke Aizawa, Adrian E. Bayer, Linda Blot, Jessica Cowell, Suyog Garg, Jonathan Grée, Anamaria Hell, Ben Horowitz, Masaya Ichikawa, Kanyuni Iemoto, Keigo Kondo, Zacharie Lorsin, Kevin McCarthy, Jamie Robinson, Miguel Ruiz-Granda, Leander Thiele, Ievgen Vovk, Mingshen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.25881)  

**Abstract**: We investigate how well large language models (LLMs) can assist scientific project planning and proposal evaluation. One-page project plans were independently generated for eight expert-conceived research projects in physics, astrophysics, and cosmology by human researchers and three contemporary LLMs (ChatGPT, Claude, and DeepSeek; mid-2025 models, used with their default tool access). The resulting 32 proposals were blindly evaluated by four human reviewers and two newer frontier LLMs (Claude Opus 4.8 and ChatGPT Pro 5.5) using a four-aspect evaluation rubric. Reviewers were also asked to identify whether each proposal was written by a human or an AI. Human reviewers rated human- and AI-written proposals similarly overall, whereas both AI reviewers scored AI-written proposals about one point higher (on a five-point scale) than human-written proposals. Human reviewers correctly identified human- and AI-written proposals 72% and 79% of the time, respectively, while both AI reviewers correctly classified all 32 proposals (100%). These results suggest that current LLMs can produce project plans comparable to human-written ones in the eyes of human reviewers, but that AI reviewers show a systematic preference for AI-generated proposals. Our results suggest caution when deploying LLMs widely in proposal preparation and evaluation. 

---
# Shieldstral 

**Authors**: Antonia Calvi, Avinash Sooriyarachchi, Giada Pistilli, Guillaume Lample, Maarten Buyl, Maximilian Augustin, Maximilian Müller, Pierre Stock, Tom Bewley, Wassim Bouaziz, Yimu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2607.25857)  

**Abstract**: We introduce Shieldstral, a 3B-parameter policy-adaptive multimodal safety classifier that matches or outperforms models nearly 7$\times$ its size on text safety benchmarks and sets a new state of the art on multimodal safety classification. Shieldstral formulates content moderation as a binary question-answering task. This simple formulation unifies diverse moderation tasks into a single yes/no problem, enabling heterogeneous safety datasets with divergent taxonomies to be consolidated under one training framework. We present the data construction recipe, covering curation and generation of approximately 54.1M samples and a fine-grained evaluation set to evaluate policy adaptability. Together, these enable a small adaptive model to match or outperform much larger models. 

---
# AngelSpec: Towards Real-World High Performance Inference with Speculative Decoding 

**Authors**: Hong Liu, Rui Cen, Junhan Shi, Guangshuo Qin, Jiebin Zhang, Tianyu Liu, Runzhi Fan, Guoliang Zhao, Ruobing Xie, Kai Zhang, Song Liu, Guanghua Yu, Jianchen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.25852)  

**Abstract**: Speculative decoding accelerates large language model inference without changing the target distribution, but no single drafting structure performs best across real-world workloads. Autoregressive multi-token prediction (MTP) is a lightweight, stable proposal mechanism, whereas block-parallel diffusion amortizes drafting latency over much longer candidate sequences; the better choice depends strongly on the output distribution. We present AngelSpec, a unified training framework for MTP and block-parallel speculative decoding that addresses this heterogeneity at three levels. At the training level, rather than fitting one universal drafter to a uniform data mixture, we co-specialize structure and data: the MTP drafter is trained on diverse conversational data for high-entropy open-ended chat, and the block-diffusion drafter on code and mathematics data for longer predictable continuations. At the architecture level, we propose DFly, a block-diffusion framework combining a hybrid target-conditioning backbone with a predecessor-conditioned autoregressive head, improving target-feature utilization and intra-block dependency modeling while keeping generation parallel. At the inference level, both acceptance length and verification cost vary with domain, request, online load, and hardware, so DFly treats verification as a shared batch-level resource: it reallocates compute toward high-confidence prefixes across requests and combines expected utility with a profiled cost model to adapt verification depth online. Across the Hy3 series, DFly raises the average accepted length on Hy3-A21B by roughly 30% and attains the highest average throughput at every tested concurrency from 4 to 64, a 1.98-2.40x speedup over autoregressive decoding and 10.5-11.8% higher throughput than DFlash. We release AngelSpec to support training and extending these methods. 

---
# Evaluation of Adversarial Robustness in Arabic Language Models 

**Authors**: Anwar Alajmi, Ayed Salman, Imtiaz Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2607.25814)  

**Abstract**: The emergence of the recent outstanding capabilities of Arabic Language Models has opened doors for exposing their vulnerabilities. One of the major security risks associated with such Natural Language Processing models is adversarial attacks. These attacks can deceive the model into the wrong prediction, raising critical model security and safety concerns. This study aims to assess the robustness of five state-of-the-art Arabic Language Models under a distinct set of Arabic adversarial attacks applied at various levels of granularity and using different example generation strategies. We also explore a defense technique based on adversarial training to enhance model robustness. The results show that insertion of diacritics can reduce the accuracy of some models by 92% while maintaining a low perturbation distance. For word-level attacks, manipulating Arabic conjunctions preserves high semantic similarity scores, low perturbation distance, and leads to an accuracy degradation of up to 58%. For sentence-level attacks, paraphrasing proves its effectiveness by an average reduction of 76% in the victim models' performance. While adversarial training improves overall resilience, with MARBERT being the most robust and AraBERT showing the greatest relative gains, challenges persist, particularly against character-level noise. These findings highlight both the potential and limitations of current defense strategies in morphologically rich languages like Arabic. 

---
# WorkSurface-Bench: Benchmarking Enterprise Agents on Multi-Surface Knowledge Routing 

**Authors**: Hao Liang, Meiyi Qiang, Sizhe Qiu, Linzhuang Sun, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.25765)  

**Abstract**: Enterprise agents often need to integrate heterogeneous knowledge sources: documents for narrative facts, tables for computation, and dependency graphs for file relationships. Existing benchmarks typically evaluate retrieval or tool use without distinguishing whether an agent first selects the appropriate knowledge sources. We introduce WorkSurface-Bench, a benchmark for evaluating this capability as surface routing. It contains 1,151 atomic tasks derived from persona-scoped Workspace-Bench-Lite workspaces, spanning document, table, graph, and cross-surface questions. Its reference answers are auditable: table answers are reproduced through executed DuckDB queries, document answers are grounded in verified text spans, and graph answers are traced to source dependency annotations. We evaluate four model backbones across six controlled agent settings, yielding 27,624 protocol-error-free trajectories. Under gold-constrained tool access, agents achieve 98.7-99.8 Route F1, while Answer remains only 56.1-75.3 percent, showing that correct surface selection is necessary but insufficient for task completion. Matched interventions further show that surface hints improve Answer for three of four models, whereas removing irrelevant tools primarily improves routing and efficiency. In an independent three-annotator audit, all 200 sampled tasks pass all six quality criteria by majority vote, with 192 receiving unanimous judgments on every criterion. We release the dataset, construction pipeline, scoring code, and agent harness at this https URL. 

---
# SpeechLLM Meets Federated Learning for End-to-End ASR: English and Italian Case Studies 

**Authors**: Mohamed Nabih Ali, Daniele Falavigna, Alessio Brutti  

**Link**: [PDF](https://arxiv.org/pdf/2607.25716)  

**Abstract**: Federated learning (FL) enables privacy-preserving training of automatic speech recognition (ASR) systems across distributed data sources, yet its application to large-scale speech language models (SpeechLLMs) remains unexplored. This paper presents the first systematic study of federated training for SpeechLLM-based end-to-end ASR systems. We design a communication-efficient federated optimization strategy tailored to the unique challenges of SpeechLLM architectures, addressing high-dimensional parameter spaces, gradient communication overhead, and computational constraints in distributed settings. Through extensive empirical evaluation on monolingual ASR tasks in English and Italian, we demonstrate the effectiveness and stability of our federated approach compared to centralized training baselines across diverse acoustic conditions and speaking styles. Additionally, we conduct a comprehensive ablation study analyzing the impact of different speech encoder architectures on monolingual English ASR performance within the federated framework, providing insights into optimal model configurations for decentralized training. Our results achieve competitive word error rates while reducing communication costs, establishing practical foundations for federated SpeechLLM deployment in real-world multilingual scenarios. 

---
# MyMentorLLM: A psychotherapy GenAI environment with multimodal voice/text patients, trainees and experts for deliberate practice 

**Authors**: Rodolfo Rizzi, Alessandro Grecucci, Massimo Stella  

**Link**: [PDF](https://arxiv.org/pdf/2607.25667)  

**Abstract**: Psychotherapists need repeated training and supervision by experts; however, scalability is problematic. Here we present MyMentorLLM, a multimodal voice- and text-based simulation environment for deliberate practice, used to generate 2,100 complete Cognitive Behavioural Therapy (CBT) training sessions. Each session links a DSM-5-TR-grounded patient (with major depressive, generalised anxiety or borderline personality disorder), a therapist-in-training and an expert supervisor. As an initial implementation, we adopted CBT because its structured procedures and competency-based supervision facilitate standardised simulation and evaluation. Sessions were analysed for emotional dynamics, therapeutic competence and diagnostic accuracy. Simulated patients expressed disorder-congruent emotional profiles, which trainee therapists mirrored as in real human counselling. The quality of supervision differed across LLMs: while most models overestimated trainees' competences, native speech-to-speech was closest to human scores. Supervisors' feedback led to better diagnoses in simulated psychotherapists in 5 out of 7 LLMs, and symptom identification accuracy increased with model size. This work shows that simulation of deliberate practice is possible for CBT training, although patient fidelity, calibration of supervisors, and harmful feedback should be evaluated together. 

---
# Construction-Driven Injection: Linguistically-Grounded Edit-Based Code-Mixing Fingerprints for Large Language Models 

**Authors**: Yongyi Cui, Yue Li, Tianbao Jiang, Xin Yi  

**Link**: [PDF](https://arxiv.org/pdf/2607.25633)  

**Abstract**: Large language models (LLMs) are costly intellectual assets that remain exposed to unauthorized redistribution and commercial misuse. Injected fingerprints, i.e., trigger--target pairs embedded in model behavior, offer a practical, black-box-verifiable ownership signal, but existing methods decouple the two stages of the fingerprint life cycle: how a fingerprint is constructed and how it is injected. Existing fingerprinting frameworks suffer from two limitations. Natural-language fingerprints are prone to accidental activation, and garbled fingerprints are easily filtered by perplexity-based detection. Furthermore, decoupling construction from injection leaves the latter unaware of the trigger's linguistic structure, missing the opportunity for targeted optimization. We argue that fingerprint construction should drive injection, and present a unified fingerprinting framework that jointly optimizes both stages. First, LCF constructs code-mixing fingerprints by combining low-resource languages under a semantic-density substitution rule and grammar-biased mixing, yielding triggers whose perplexity sits far below garbled baselines while avoiding the accidental-activation failures of natural-language triggers. Second, LCFEdit injects each fingerprint with a null-space projection derived from high-resource multilingual representations that preserves knowledge, augmented by a cross-lingual alignment step that steers the weight update toward the fingerprint language's representation subspace. This construction-aware injection ensures that the update is linguistically informed and therefore more stable. Extensive evaluations on imperceptibility, detectability, and harmlessness demonstrate persistent ownership verification with negligible impact on utility. 

---
# A Human-in-the-Loop Corpus for LLM-Based Simplification of Scientific Summaries 

**Authors**: Kyuri Im, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2607.25630)  

**Abstract**: Interdisciplinary research is accelerating, yet scientific papers remain difficult to understand outside their home fields. We study large language model (LLM)-based simplification of scientific texts and present a human-in-the-loop workflow that transforms expert summaries into more accessible versions for non-specialists. Using SciSummNet as the source corpus, we first generate baseline simplifications with GPT-4o-mini. In Phase 1, readers from STEM fields outside computer science identify difficult sentences and phrases and compare the original and GPT-simplified summaries in terms of comprehensibility, naturalness, and simplicity. In Phase 2, computer science experts use this feedback to create expert-edited reference simplifications. We release the resulting corpus together with human judgments and automatic evaluation results. The Phase 1 judgments show a clear preference for the GPT-generated summaries in terms of comprehensibility and simplicity, while qualitative analysis of the Phase 2 edits highlights the importance of preserving domain-specific terminology and the strength of scientific claims. The resulting resource supports the training and benchmarking of simplification systems for cross-disciplinary scientific communication. 

---
# PILA: Plug-and-Play Insertion for LLM-native Advertising 

**Authors**: Zhaowei Zhang, Yuhan Fu, Yihang Zhang, Xiaohan Liu, Ceyao Zhang, Xiaoyuan Zhang, Yipeng Kang, Tonghan Wang, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.25590)  

**Abstract**: How to monetize large language models (LLMs) by naturally integrating sponsored content into their responses, known as LLM-native advertising, has recently emerged as a critical problem. However, existing solutions entangle advertising with content generation inside a single model, which is incompatible with modern API-only or workflow-based LLM applications and inevitably compromises the original response quality. To address this, we propose PILA, which reformulates ad insertion as a conditional response rewriting problem and decouples it from the upstream service as a lightweight sidecar module. PILA is model-agnostic and can be seamlessly integrated with existing LLM services without modifying the base model or its workflow. It also exposes a controllable trade-off between user-side naturalness and ad-side exposure, offering a practical interface for downstream pricing and deployment. Experiments across diverse upstream models show that \pila consistently improves ad effectiveness while preserving response quality, highlighting its promise as a practical solution for LLM-native advertising. 

---
# Evaluation of forced alignment of code-mixed speech: the case of Hindi-English 

**Authors**: Ayushi Pandey, Pamir Gogoi, Kevin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2607.25581)  

**Abstract**: Code-mixed speech poses unique challenges to forced alignment: expanded inventories, orthographic errors, and speaker variation. We evaluate forced alignment of Hindi-English code-mixed speech using the Montreal Forced Aligner. We address 2 problems: (1) free variation involving native vs non-native pairs and (2) phonemic boundary detection for mid-utterance English words. Bootstrapping strategies substantially outperform unmodified lexicons. Acoustic models trained on sentence-level code-mixed data achieve a mean error of 4.15ms, ie. ten times lower than monolingual Hindi (38.18ms) or isolated English (37.58ms) alternatives. Principled lexicon design and code-mixed training data are both essential for reliable alignment of bilingual speech. 

---
# IRIS: Reusable Identity Representations from Frozen LLMs for Entity Alignment 

**Authors**: Xinran Liu, Shengtao Li, Shouqian Shi, Ge Wang, Xin-Wei Yao  

**Link**: [PDF](https://arxiv.org/pdf/2607.25579)  

**Abstract**: Entity alignment (EA) identifies entities across knowledge graphs (KGs) that refer to the same real-world object. Conventional EA methods mainly exploit explicit graph structures and textual fields, which often provide insufficient semantic understanding to recognize the same entity under heterogeneous descriptions and distinguish it from semantically similar entities. Although large language models (LLMs) offer deeper entity understanding, existing LLM-based EA methods largely use this capability for auxiliary generation or candidate-conditioned decisions. Consequently, such understanding is not distilled into a stable and directly comparable identity space, leaving alignment tied to specific KG pairs or candidate sets and requiring repeated processing as the matching context changes. To address these limitations, we propose IRIS (Identity Representations from Internal States), a training-free framework that constructs for each entity an iris-like signature encoding its distinctive and stable identity characteristics. IRIS derives these signatures by eliciting identity-oriented contextual representations from a frozen LLM, thereby forming a shared space in which each entity is encoded once and can be aligned across different KGs through direct similarity comparison, without pair-dependent representation construction or candidate-wise LLM inference. Across four established EA benchmarks and two frozen LLM backbones, the best IRIS variants achieve Hits@1 scores of 100.00, 99.38, 98.31, and 97.99 on D-Y-15K V2, DBP-WIKI, ICEWS-WIKI, and ICEWS-YAGO, respectively. 

---
# Phase Structure in Rotary Attention: A Spectral Framework for Semantic Continuity and Execution-Boundary Governance 

**Authors**: Abraham Chachamovits  

**Link**: [PDF](https://arxiv.org/pdf/2607.25507)  

**Abstract**: Transformer language models are usually analyzed through vector geometry, yet ordered context and rotary position encoding introduce explicit phase structure into query-key interactions. This paper develops a bounded spectral framework for examining rotary phase alignment, hidden-state continuity, and semantic drift without treating language models as literal physical wave systems. It first identifies ordered hidden-state sequences, rather than vocabulary indices, as valid domains for spectral decomposition. It then derives the Rotary Position Embedding (RoPE) attention score as a sum of magnitude-weighted cosine terms and proves a local stability lemma: uniformly bounded phase displacement limits degradation of the corresponding pre-softmax score. To extend phase analysis beyond native RoPE coordinates, the paper defines complex modal coordinates over fixed orthonormal direction pairs and introduces a weighted coherence functional for hidden-state trajectories. These constructions support a strict distinction between representational continuity and execution-boundary admissibility. Internal coherence may describe preservation of task-relevant relations, but it cannot authorize a consequential transition. Positioned against existing geometric, spectral, phase-modulation, representation-analysis, and mechanistic-interpretability accounts, the framework contributes a theoretical and methodological program for determining when spectral structure explains continuity and when governance must remain an external predicate over execution. 

---
# Memory for Large Language Models 

**Authors**: Sining Zhoubian, Dan Zhang, Evgeny Kharlamov, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2607.25380)  

**Abstract**: Memory has evolved into a foundational architectural dimension in large language models (LLMs), shifting from an implicit byproduct of computation to a spectrum of explicit, controllable mechanisms. While recent advances introduce diverse strategies---spanning transient attention, recurrent state dynamics, parameter-efficient adaptations, and scalable lookup storage---this rapid evolution has led to a highly fragmented research landscape. In this survey, we present a systematic, architecture-centric taxonomy of memory in LLMs. Our framework characterizes memory along three orthogonal axes: representation (implicit versus explicit), update dynamics (offline versus online), and persistence (short-term versus long-term). We further formalize the granular mechanisms dictating memory writing, routing, state transitions, and consolidation. This unified perspective elucidates the conceptual boundaries between computation-coupled and independently addressable memory, effectively bridging disparate architectural paradigms. Additionally, we critically analyze hybrid memory architectures, system-level efficiency trade-offs, and multi-dimensional evaluation methodologies. By consolidating these scattered advancements into a cohesive framework, this survey charts the trajectory of memory-centric LLM design and provides a principled foundation for future innovations in scalable and adaptive language modeling. 

---
# Inspect India Evals: An Open Benchmarking Framework for Evaluating Large Language Models in the Indian Linguistic and Cultural Context 

**Authors**: Abhishek Kumar Singh, Shrey Nag, Sachita, Lipi Goel, Rajeshwar Singh Janwar  

**Link**: [PDF](https://arxiv.org/pdf/2607.25375)  

**Abstract**: India is a vast nation of over 1.4 billion people, varied by hundreds of diverse and locally specific traditions and cultures and 22 officially recognized languages. Large language models (LLMs) are now being deployed on a massive scale throughout the mainland as well as in remote villages. However, the common benchmarks - MMLU, BIG-Bench, and TruthfulQA are almost exclusively English- and Western-centric. They do not identify those safety, fairness, and accuracy failures unique to the Indian context. That is the gap Inspect India Evals seeks to fill. It is an open-source framework built on top of UK AISI's Inspect AI platform. It has six benchmarks: Multilingual MMLU across sixteen Indian languages, BharatBBQ (our adaptation of BBQ for Indian social bias), a safety evaluation for Digital Public Infrastructure, a multilingual safety test using harmful prompts in Indian languages, a multi-turn jailbreak resistance test, and an Indian cultural knowledge benchmark scored using LLM-as-judge rubrics. In this study, we tested five open-weight models ranging from 8B to 32B parameters. Sarvam-M 24B and Gemma 2 27B came out on top, both scoring 80% on the composite India Fairness Index, with Sarvam-M even beating larger 32B models on Indian cultural knowledge and DPI safety compliance. All models scored 100% refusal on Multilingual Safety, whereas DPI safety varied from 20% to 100%. The framework is public. It's built to work with the UK AISI registry. Anyone can reproduce or extend this work. 

---
# Temporal-Distance JEPA: Plan-Aware Representation Learning for Latent World Model Predictive Control 

**Authors**: Jiaxin Bai, Jiaxuan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2607.25337)  

**Abstract**: Joint-Embedding Predictive Architectures (JEPAs) learn world models by predicting in representation space rather than reconstructing pixels, making them a natural backbone for latent model predictive control from offline demonstration logs. JEPA-style training optimizes short-horizon latent prediction, whereas planning requires a multi-step ranking of imagined futures by goal progress. Prior JEPA planners often inherit that ranking from embedding geometry, typically latent Euclidean distance, which arises as a byproduct of representation learning rather than as a progress cost mined from the logs. We propose temporal-distance JEPA (TD-JEPA), which retains the LeWM encoder--predictor backbone and mines a directed temporal cost from reward-free trajectories: same-trajectory step order supplies positive targets, cross-trajectory pairs act as heuristic negatives, and a rollout-consistency term matches the planner horizon. The mined supervision serves two roles: as the deployed planning cost when progress is topological, and as a representation signal that improves Euclidean planning when contact geometry dominates. Under locked evaluation, deploying the mined cost raises Two-Room success to 100.0% versus LeWM's 97.4%, while shared Euclidean planning on the same temporally trained checkpoint raises OGB-Cube by 14.2 points over LeWM and improves Push-T. Against LeWM and the concurrent RC-aux baseline under locked evaluation, TD-JEPA matches or exceeds both methods on every environment. Ablations show that the directed head, cross-trajectory negatives, and rollout consistency each contribute. TD-JEPA narrows the train--plan gap for JEPA world-model planners by discovering temporal progress structure in offline logs and co-designing cost form with plan-time deployment. Code is available at this https URL. 

---
# Every Time I Hire a Linguist, Inference Costs Go Down: On Linguistic Rules as Effective Prompt Compressors 

**Authors**: Jianfei Ma, Zhaoxin Feng, Emmanuele Chersoni, Si Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.25335)  

**Abstract**: Prompt compression shortens LLM input to reduce inference cost, yet existing methods score token importance through LM forward passes. It remains questionable whether such nuanced, costly token selection is necessary. Compression requires identifying informative content, a problem that linguistic research has long addressed through cues that can be operationalized as deterministic rules. We therefore ask: can \textbf{linguistic rules alone} serve as effective prompt compressors, without LM-based scoring at compression time?
To address this, we conduct offline evolutionary search over lexical, syntactic, semantic, and discourse seeds to find competitive rule combinations. The resulting linguistic compressor requires no LM forward pass at deployment and uses only CPU-side processing for compression. We evaluate it with a dual-path protocol to balance compression quality and reconstruction fidelity.
Across short passages, multi-document reasoning, and dialogue-memory QA datasets, evolved compressors achieve performance similar to that of recent advanced prompt-compression strategies. Performance is strongest under light-to-moderate compression and degrades as compression becomes more aggressive, while the Direct and Reconstruction paths exhibit distinct patterns. Evolutionary analysis reveals that effective compression fuses signals across linguistic levels and, as the compression ratio increases, rules shift from token pruning to sentence extraction. 

---
# CAST: Game Solvers as Turn-Level Teachers for LLM Agents 

**Authors**: Yu Wang, Yi-Kai Zhang, Wentao Shi, Ziang Ye, Yuchun Miao, Yueqing Sun, Qi Gu, Xunliang Cai, Lan-Zhe Guo, Han-Jia Ye, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2607.25308)  

**Abstract**: Training large language models (LLMs) to act in long-horizon games is a promising step toward generalist decision-making, yet reinforcement learning with verifiable rewards (RLVR) relies on sparse final rewards that reveal little about which decisions determine success. Denser process signals could supply this missing turn-level credit, but existing sources are hard to keep both cheap and accurate. We observe that changes in a game solver's state value reveal whether an action advances the state toward success. Building on this insight, we propose CAST (Credit Assignment from Solver Teachers), which converts these value changes into solver advantages and injects them into RLVR as turn-level signals. We further show that, under a soft-optimal solver assumption, maximizing the solver advantage is equivalent to on-policy distillation from the solver, requiring only scalar values rather than teacher logits. Across Sokoban, Minesweeper, and Rush Hour, CAST outperforms all trained baselines on every game under both in-domain and unseen-difficulty evaluation and achieves the highest average zero-shot performance on ALFWorld and WebShop. Our code is available at this https URL. 

---
# Toward a systematic method for identifying language areas 

**Authors**: Hiram Ring  

**Link**: [PDF](https://arxiv.org/pdf/2607.25305)  

**Abstract**: Macroareas are geographical areas used in typological research for grouping variables of interest. In linguistic typology, languages in a given macroarea are considered to have potential for contact, in contrast to those outside the area, where contact is less likely. Along with language family membership, macroareas are used as controls for models in linguistic typology, in an attempt to address the problem of autocorrelation - the observation that historical developments or typological patterns may be due to contact between neighboring languages and/or inheritance from a common ancestral language. Macroareas are therefore a central aspect of research that seeks to separate universal properties of language from local (or language-specific) properties. Existing macroareas largely depend on expert determinations of what constitutes a geographical area of potential contact, and to date have mainly aligned with continents or landmasses (Hammarström and Donohue 2014; Nichols, Witzlack-Makarevich, and Bickel 2013). While there are various historical and theoretical reasons for these groupings, there as of yet has been no systematic approach to identifying such areas for a given region. This paper attempts to address such a gap and move beyond macroarea to identification of language areas of relatively arbitrary size, presenting a simple geographical clustering method for identifying groupings over any area. The method produces a set of worldwide macroareas that largely align with existing groupings, as well as local groupings for a well-known sprachbund. 

---
# CoSA: Accelerating Long-Context Inference via Proxy-Kernel Co-Designed Sparse Attention 

**Authors**: Yufei Xue, Lin Niu, Hong Liu, Siran Liu, Hanyong Shao, Wei Liu, Guanghua Yu, Jianchen Zhu, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.25291)  

**Abstract**: The quadratic cost of self-attention makes long-context inference prohibitively expensive, and proxy-based block-sparse attention has become a practical remedy. Existing methods typically rely on a proxy to predict a binary sparse mask and a kernel to consume this mask and perform sparse attention computation. Such an approach is effective under moderate budgets. However, as the budget tightens, the estimated proxy inevitably drops some salient blocks, while the kernel can only apply the sparse mask mechanically, leading to an evident drop in model accuracy. We propose CoSA, a two-stage training-free Sparse Attention under proxy-kernel CO-design, which couples a Kernel-Aware Proxy (KAP) with an Ordered-Skipping Kernel (OSK). In the first stage, the KAP selects blocks under a moderate budget and produces an ordered mask that prescribes the order in which KV pages are visited in the kernel inner loop. In the second stage, the OSK applies this mask and skips more blocks under a tightened budget given online-softmax statistics. Across mainstream LLM backbones and long-context benchmarks, CoSA attains higher accuracy at lower budgets. Impressively, CoSA achieves a 4.93$\times$ attention speedup and reduces end-to-end Time-to-First-Token by 2.53$\times$ under a context length of 128K with negligible performance degradation. 

---
# Where Steering Signals Come From: Activation Source Selection in Activation Steering 

**Authors**: Jiaran Ye, Lingxu Ran, Zijun Yao, Chenpeng Wang, Yong Jiang, Lei Hou, Juanzi Li, Liangming Pan  

**Link**: [PDF](https://arxiv.org/pdf/2607.25270)  

**Abstract**: Activation steering controls language models by adding vectors or features to hidden states at inference time, but the upstream source of these steering signals is often treated as a secondary detail. We study this source choice as activation source selection: the combination of source context and activation readout policy used to collect the hidden states from which a steering signal is built. Holding the downstream intervention fixed, we show across three instruction-tuned models and four steering task families that changing only the source activations substantially changes steering success. We further find that effective steering is not explained simply by whether the desired behavior appears in the source text. Instead, strong signals come from execution-boundary states, where the model is about to produce or continue the target behavior. This pre-/post-realization distinction explains why answer-based sources sometimes work: their useful component aligns with execution-boundary directions rather than target appearance alone. Building on this view, we introduce tail subtraction, which removes shared prompt and continuation semantics from boundary states and yields cleaner, more stable steering signals. Overall, our results suggest that steering depends on representations of what the model is about to do, not merely on what has already appeared. 

---
# VisualPatchWorld: Code World Models as Latent Structured Representations for Planning 

**Authors**: Jiaxin Bai, Jiaxuan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2607.25236)  

**Abstract**: Different research lines use the term world model in different ways, yet they share a common aim: to capture how the world evolves under action in a form that supports perception, simulation, and planning. Two prominent realizations are neural predictors that learn dynamics in continuous vector spaces, and hand-built physics engines that expose explicit state and physical laws. Neural predictors scale from data but leave the form of the dynamics implicit; physics engines are inspectable and editable but difficult to construct at scale. We introduce VisualPatchWorld (VPW), which represents world dynamics as code. VPW first selects a qualitative dynamical form with short active probes, then fits that form's free parameters from recorded state-action traces by minimizing multi-step prediction error. The resulting programs can be rolled forward like a simulator, inspected in source form, and used inside model-predictive control; image-derived scene graphs can supply the live state at replan time. Across comparisons with prior code-based world models, VPW attains 69.0% mean planning success and exceeds the strongest code baseline by 23.5 points. The largest gains arise when choosing the correct qualitative dynamics is essential. Under the same planner, the induced models approach ground-truth engine success on navigation and grasp-rich control; a residual gap remains for contact-rich pushing, and checking a shortlist of promising plans in the engine closes most of that gap. These results establish a practical route toward automatically constructed code world models that are useful for planning. Code is available at this https URL. 

---
# Interpretable Column Annotation with LLM-Symbolized Decision Process Materialization 

**Authors**: Mengqi Wang, Jianwei Wang, Qing Liu, Xiwei Xu, Zhenchang Xing, Michael Bain, Liming Zhu, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.25228)  

**Abstract**: Column annotation (CA), including column type annotation (CTA) and column property annotation (CPA), aims to identify the meanings of table columns and the semantic relationships among them. Recent CA methods usually use various neural models to learn column representations and directly map them to label categories, thereby (1) sacrificing model interpretability and adaptivity, and (2) overlooking rich label semantics and ultimately limiting accuracy. To address these limitations, we propose SymCA, an LLM-empowered interpretable CA framework that materializes column annotation as a global-to-local symbolic decision process. SymCA consists of two components: (1) global skeleton induction, which constructs a semantic skeleton over the label space, and (2) local substrate evolution, which evolves predictive substrates within the skeleton. Specifically, to exploit label semantics while preserving an interpretable decision process, the global skeleton induction module leverages LLMs to generate candidate hypernym-inspired tree-structured semantic skeletons and employs a Minimum Bayes Risk (MBR)-based consensus strategy to select a robust skeleton against generation variance. Since different internal nodes require different evidence to distinguish among their child nodes, the local substrate evolution module materializes each internal node as an executable and evolvable predictive substrate. Over multiple evolution rounds, each substrate trains an interpretable random forest classifier with the current operator set, leverages the LLM to propose node-specific operator modifications, and uses an exploration-exploitation strategy to prioritize promising substrates. Extensive experiments demonstrate that SymCA is accurate, robust, and interpretable, outperforming the strongest baselines by an average of 6.42% in Micro-F1 and 11.03% in Macro-F1. 

---
# A Cross-lingual Comparison of Human and Classification Model Entrainment Behavior in Code-switched Speech Settings 

**Authors**: Debasmita Bhattacharya, Siying Ding, Alayna Nguyen, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2607.25202)  

**Abstract**: Conversational entrainment is well-studied in monolingual and written contexts, but remains underexplored in spoken code-switching (CSW). We present a novel cross-lingual analysis of entrainment in Mandarin-English, Hindi-English, and Spanish-English dialogue and show that, while lexical entrainment generalizes across language pairs, entrainment over acoustic-prosodic and CSW style aspects exhibits context-specific variation. We build on these findings by asking whether classification models capture these human behavioral patterns. Applying feature importance and ablation analyses, we find that classical and Transformer-based classifiers detect entrainment reasonably well but consistently prioritize features other than those most salient to human entraining behavior. Our approach introduces a human-grounded framework for evaluating model decision-making in multilingual stylistic contexts, and suggests future challenges for developing conversational agents capable of producing naturalistic code-switched speech. 

---
# MyoCardBench: A Real-World Data Benchmark for Evaluating Large Language Models in Clinically Authentic Cardiovascular Care Scenarios 

**Authors**: Xiao Li, Mouxiao Bian, Zhaodi Wu, Sijie Ren, Juechen Chen, Lu Lu, Jingru Ding, Yun Zhong, Jie Xu, Yixiu Liang, Junbo Ge  

**Link**: [PDF](https://arxiv.org/pdf/2607.25186)  

**Abstract**: Background: Most medical large language model (LLM) benchmarks focus on examination knowledge or isolated tasks and may not reflect the longitudinal, multimodal, and safety-critical workflow of cardiovascular care. Objective: To develop MyoCardBench, a real-world benchmark spanning the cardiovascular care continuum, and assess LLM performance across clinical dimensions and specialist tasks. Methods: MyoCardBench includes 2,263 items from 13 task-specific datasets derived from de-identified cardiovascular records and examination data. Sixteen cardiology physicians conducted annotation and reference construction, followed by cross-review from two senior cardiologists. Seven LLMs generated 15,841 outputs under standardized zero-shot settings. Open-ended tasks were evaluated using key-point coverage and holistic clinical quality, while CardioEthics was scored by accuracy. Results: GPT-5.4 achieved the highest macro-average (62.55) and item-weighted mean (62.19), followed by Gemini 3.1 Pro (59.95) and Qwen 3.6 27B (59.72). GPT-5.4 ranked first in all three dimensions. CardioAuxReport performed best (86.38), whereas CardioECGRead (17.25) and CardioEthics (17.34) were lowest. The largest gaps between holistic clinical quality and key-point coverage occurred in CardioComm (52.71), CardioEmergRescue (52.05), and CardioTreatPlan (48.80). Conclusions: To our knowledge, MyoCardBench is the largest real-world, multi-task benchmark for LLM evaluation across the cardiovascular care continuum and offers the broadest coverage of clinically authentic cardiology scenarios reported to date. It provides a rigorous framework for identifying model strengths, clinically important omissions, and priorities for future development. 

---
# A scaling law of contextual persistence in human language 

**Authors**: Elan Barenholtz  

**Link**: [PDF](https://arxiv.org/pdf/2607.25184)  

**Abstract**: Human language exhibits lawful structure at the level of words (frequency, vocabulary growth) and word pairs (co-occurrence across distance). Here we show that the arrangement of words in sequence -- a central determinant of meaning -- obeys a comparable law. Using large language models as probabilistic probes, we measured the reduction in target perplexity conferred by prior context at distance d beyond that of the same words scrambled; this difference, the contextual persistence function P(d), isolates the influence of arrangement. Across ten corpora spanning six language families and written and spoken modalities, P(d) decayed approximately as 1/d ($P(d) \propto d^{-\alpha}$, mean $\alpha = 1.04$; median $r^2 = 0.96$). The effect vanished in scrambled and synthetic controls, replicated across independent probes, and did not appear in genomic or protein sequences under domain-native models. An exponent near 1 distributes contextual influence approximately uniformly across logarithmic timescales. The results establish a scaling law of contextual persistence in human language. 

---
# TabRank: Chain-of-Thought Distillation for Table Re-Rankers 

**Authors**: Adarsh Singh, Kushal Raj Bhandari, Jianxi Gao, Soham Dan, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2607.25182)  

**Abstract**: The ability to retrieve relevant tables for answering questions is a key task for structured information retrieval. Multi-stage retrieval systems rely heavily on rerankers to refine candidate lists produced by efficient first-stage retrievers. As a result, neural rerankers and LLM-based reranking methods have become increasingly important due to their superior capacity for semantic understanding and reasoning compared to conventional sparse or dense retrieval models. Recently, Large Reasoning Models (LRMs) equipped with explicit chain-of-thought (CoT) reasoning have shown strong improvements in ranking quality in unstructured passage retrieval. In this work, we present TabRank, a framework for training reasoning rerankers for Tabular Retrieval. We first present a comprehensive dataset of 6728 reasoning traces for tabular reranking on the Natural Questions Tables dataset. We then explore two variants of training a compact reasoning model on these reasoning traces: explicit CoT distillation and conditioning the student reranker on the teacher's reasoning trace within the prompt. We stress-test TabRank on several out-of-distribution generalization settings on diverse domains and multi-table scenarios. Our approach significantly improves performance across a variety of table retrieval datasets, increasing Acc@10 by 30.5% on HybridQA, 15.2% on SQA, 52.9% on TabFact, and 13.1% on TATQA subsets of the Multi-Table QA Benchmark compared to the base model. Notably, TabRank generalizes effectively to multi-table reasoning. Our code, data and models are available at this https URL 

---
# Deep Label-Wise Attentive Temporal Convolutional Networks Improve Medical Coding 

**Authors**: Muhammed Yavuz Nuzumlalı, Alexander Fabbri, Irene Li, Dragomir Radev  

**Link**: [PDF](https://arxiv.org/pdf/2607.25129)  

**Abstract**: Medical coding is the task of assigning a set of diagnosis and procedure codes for a hospitalization using recorded notes. It requires aggregating information from different parts of the text and focus to different sections for each individual code, making it a very difficult problem even for professional human coders. We model the task as a multi-label text classification problem. To overcome the mentioned difficulties, we propose a deep neural model consisting of a multi-layer temporal convolution network (TCN) followed by label-wise attention. While multi-layer TCN helps extract a global document representation with the ability to learn relations over very long sequences, label-specific attention mechanism allows the model to focus on different aspects of the same document for each individual label. Our method achieves significantly better F-1 scores (9% increase) compared to the previous state-of-the-art model, with a remarkable increase in recall score (28% increase), which we believe is the more important metric for a clinical decision support setting. 

---
# Evaluating Communicative Belief Updates in Large Language Models via Implicature Recognition and Cancellation 

**Authors**: Cesare Spinoso-Di Piano, Verna Dankers, Marius Mosbach, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2607.25094)  

**Abstract**: Human language is driven by unspoken beliefs and belief updates, making these critical to model for successful communication between large language models (LLMs) and their users. In this paper, we evaluate the ability of LLMs to recognize unspoken beliefs made through implicatures and to understand their updates through implicature cancellation: the pragmatic phenomenon whereby an utterance's implied meaning is weakened or negated. We create the first expert-annotated implicature cancellation dataset, [DatasetName], crowdsourced for human judgements of implicatures and their corresponding cancellations. We find that LLM belief update understanding lags behind that of humans, especially in more naturally-occurring scenarios. Additional control experiments suggest that successes in LLM belief updates may stem in part from a reliance on prior beliefs, and that failures in belief updates may depend on their type and on their form. Overall, our study suggests that current LLMs have not yet reached human-level understanding of unspoken beliefs and belief updates. Code and data are available at this https URL. 

---
# DS@GT ARC at CheckThat! 2026: LLM-Based Trace Ranking and Grouped Reward Modeling for Multilingual Numerical Claim Verification 

**Authors**: Sagnik Sinha, Shreyas Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2607.25069)  

**Abstract**: Automated verification of numerical claims is a challenging problem, as it requires both language understanding and quantitative reasoning. This paper describes our system for CLEF 2026 CheckThat! Task 2, which focuses on ranking reasoning traces generated by large language models (LLMs) and predicting a final verdict for numerical claims in English and Arabic. We explore two approaches. The first approach fine-tunes an LLM-based verifier using LoRA to score each reasoning trace independently as a binary classification problem, and selects the final verdict using Best-of-N selection. We further experiment with adaptive sub-claim decomposition to break complex claims into simpler parts before verification. The second approach uses a lightweight TF-IDF reward model with handcrafted numeric and temporal overlap features to score traces, and aggregates scores by verdict group to determine the final prediction. For Arabic, we compare a general multilingual model against AraBERT, a language-specific model pretrained on Arabic text. Our results show that the LLM-based approach outperforms the lightweight reward model on most metrics, particularly Recall@5, while the reward-based approach shows stronger performance on the Conflicting class. Sub-claim decomposition did not improve performance, suggesting that claim splitting introduces noise rather than aiding reasoning. For Arabic, AraBERT outperforms the multilingual baseline across most metrics. 

---
# CogArena: A Multimethod Evaluation of Cognitive Ability Structure in Large Language Models 

**Authors**: Dengzhe Hou, Lingyu Jiang, Fangzhou Lin, Kazunori D Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2607.24999)  

**Abstract**: LLM cognitive scores are increasingly summarized as per-ability profiles whose dimensions should converge across tasks, respond selectively to matched interventions, and generalize beyond the models used to define them. We introduce CogArena, a procedurally generated 13-paradigm benchmark built around a multimethod framework for determining when cognitive-task scores warrant dimensional labels across five theory-motivated groupings. Across 55 open-weight models, nearly all paradigm correlations are positive and a common axis explains about half the variance. The within-grouping advantage is small, scoring-sensitive, and uncertain across model families. In a separately frozen, fully crossed study across 12 models from six families, targeted scaffolds show a small matched-grouping advantage, but no scaffold-specific contrast survives multiplicity correction and selectivity does not improve held-out-family prediction. The frozen confirmation criterion fails. A post-hoc alternate-wording replication produces a smaller positive estimate and again fails. Together, these results support a boundary conclusion. Theory-aligned prompting produces a small in-battery diagonal tendency, but the present evidence does not establish stable five-dimensional profiles. CogArena provides a workflow joining behavioral signatures, covariance, matched interventions, and out-of-family prediction before cognitive labels are attached to model scores. 

---
# Research Report on Noise-Shaped One-Bit Coefficients in Discrete Polynomial Fourier Extension 

**Authors**: Shengquan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.24868)  

**Abstract**: This report studies noise-shaped one-bit coefficients in normalized discrete polynomial Fourier extension. For first-order Sigma-Delta quantization, the error is written as $e_k=u_k-q_k=\Delta v_k$ with a uniformly bounded state. Discrete summation by parts then yields variation estimates for complex weights and an $O(N^{-1})$ approximation rate on compact parameter sets. For the parabolic phase $\phi_{x,t}(\xi)=x\xi+t\xi^2$, the bound is expressed through $J(x,t)=\int_0^1 |x+2t\xi|d\xi$, and the uniform $N^{-1}$ rate is shown to be sharp over the admissible input class. Higher-order finite-record identities are derived with all endpoint traces retained. Under endpoint compatibility, or after explicit boundary correction, an $r$th-order noise-shaped error $e=\Delta^r v$ gives $O(N^{-r})$ decay for sufficiently smooth weights and $O(N^{-(r-1+\alpha)})$ decay for $C^{r-1,\alpha}$ weights. Exact $L^2$ orthogonality identities, fourth-moment formulas, local kernel estimates, and oscillatory transfer bounds are also established. Extensions to polynomial phases, multidimensional parameter families, growing observation regions, and correlated state models are included. 

---
# Neuromorphic Diffusion Language Models: Addressing Compute and Memory Bottlenecks via Sparsity and Block Denoising 

**Authors**: Dengyu Wu, Clement Ruah, Jiechen Chen, Bipin Rajendran, Osvaldo Simeone  

**Link**: [PDF](https://arxiv.org/pdf/2607.24841)  

**Abstract**: Autoregressive (AR) large language models (LLMs) are inherently inefficient at inference time because each generated token requires accessing the full set of model parameters, leading to low operational intensity and high energy consumption. Masked diffusion language models (MDLMs) partially address this limitation for memory-bound settings by allowing multiple tokens to be generated per parameter access. In order to further enhance inference efficiency on modern platforms with extensive in-chip memory, this work proposes neuromorphic MDLMs (N-MDLMs), which integrate block diffusion with spike-based neuromorphic computation to jointly improve throughput and energy efficiency. While block diffusion increases token throughput by producing multiple tokens per parameter access, spike-induced sparsity reduces effective parameter traffic and computations by skipping inactive channels. To analyze the synergistic effect of sparsity and diffusion, we develop a token-level roofline-inspired model that captures the combined impact of block-parallel generation and spike sparsity on decoding efficiency. Experimental results on translation tasks show that, thanks to spike-induced sparsity, N-MDLMs achieve substantial improvements in energy efficiency and throughput even in compute-bound platforms for which MDLMs would fail to improve over AR-LLMs. 

---
# Measuring and Improving Behavioral Consistency in Large Language Models through Fact-Heuristic-Emotion State Enforcement 

**Authors**: Gi-Hun Lee, Joong Yull Park  

**Link**: [PDF](https://arxiv.org/pdf/2607.24765)  

**Abstract**: Large language models (LLMs) can give different answers to the same decision problem across runs, and reverse a decision when their own prior answer returns as context. We ask whether this instability can be measured and partially reduced without changing model weights.
We test the Cognitive Kernel Model (CKM), a prompt-level state-enforcement layer. Before deciding, the model must separate its input into three epistemic roles: Fact (given or verifiable), Heuristic (inferred or assumed), and Emotion (evaluative or priority signal). CKM adds no capability; it forces the model to track what kind of information it uses before acting. Formally it maintains a structured state S_t = {F_t, H_t, E_t} updated by a transition function.
We evaluate CKM on Korean-language decision scenarios (ambiguity, ethical conflict, resource allocation, error handling) across 26 LLMs from four vendors and 37,403 observations, via four core experiments, a 4-arm ablation, a 5-arm sham-restriction ablation, and a temperature probe. Findings:
(1) CKM reduces repeated-output variability (random-effects Hedges' g=1.09, 95% CI [0.83, 1.35], 31 model pairs);
(2) state persistence cuts the decision-flip rate by 82% in newer models (g=1.52);
(3) the effect is not JSON formatting alone (value-only recomputation, g=2.24);
(4) intrinsic randomness under fixed anchor states is negligible;
(5) the advantage grows under sampling stochasticity (g=2.87 at temperature 0.7);
(6) a sham ablation attributes about 45% of the gain to structural scaffolding and 55% to Fact/Heuristic/Emotion content, and CKM is the only arm that both raises consistency and reduces flipping.
CKM does not improve reasoning correctness. The narrower result: behavioral consistency is measurable, varies across models, and is partially improvable by forcing models to separate facts, assumptions, and evaluative signals before deciding. 

---
# TimeCapsule: Generative Hallucination as a Method for Historical Sensemaking 

**Authors**: Hayk Grigorian, Hamed Yaghoobian  

**Link**: [PDF](https://arxiv.org/pdf/2607.24750)  

**Abstract**: Large Language Models (LLMs) are temporally overexposed: trained on vast contemporary corpora, they encode present-day concepts that make them unreliable narrators of the past. We present TimeCapsule, a 1.2B-parameter LLaMA-style causal model trained exclusively on Victorian texts (1800-1875) as an epistemologically isolated generative archive. Quantitative evaluation shows a 45.4% perplexity reduction over a GPT-2 baseline on held-out Victorian prose, while larger contemporary causal models achieve lower raw perplexity through broader pretraining but lack temporal isolation. TimeCapsule exhibits computational sensemaking, generating historically plausible analogical explanations for unfamiliar modern concepts (e.g., describing a computer as a "hypertrophied lung"). A qualitative hermeneutic probe with two humanities scholars revealed a crisis of authenticity, as both misclassified approximately 40% of genuine Victorian excerpts as machine-produced. We argue that structural ignorance of the future transforms hallucinations into interpretive probes of nineteenth-century ontologies. 

---
# A Cost-Effective Multimodal LLM Reasoning Framework for Question Answering over Irregular Clinical Time Series 

**Authors**: Frank Nie, Ethan B Liu, Yuan Zhu, Wei Fan, Jindong Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.25947)  

**Abstract**: Question answering (QA) over irregular clinical time series (ICTS) plays a pivotal role in a wide range of healthcare applications. Although recent multimodal time-series large language models (LLMs) have shown considerable promise in general-purpose time-series QA, they remain poorly equipped to model the sparsity, asynchrony, and irregular sampling patterns of clinical observations. To fill this gap, we propose ClinPRISM, a cost-effective multimodal LLM reasoning framework for question answering over ICTS data. First, we devise an irregularity-aware multi-scale encoder to capture sparse clinical evidence at diverse temporal scales. Then, we propose a temporal evidence distiller to integrate representations across these scales and compress them into a small number of LLM-compatible tokens. Moreover, we introduce a progressive alignment strategy that sequentially aligns the irregular trajectories with the LLM's textual embedding space. To facilitate training, we construct 30,000 clinical time series paired with multi-scale descriptions, together with 41,000 instruction-tuning instances spanning 11 tasks. Using a 4-billion-parameter LLM backbone, ClinPRISM achieves state-of-the-art performance on the held-out evaluation benchmark while using only 16 time-series tokens and achieving an average inference latency of 0.15 seconds per question. 

---
# Minimizing Targeted Activations: Input-Only Suppression of Evaluation-Awareness Latents in Large Language Models 

**Authors**: Deepanshu Mody, Samarth Agarwal, Utkarsh Mittal, Dipesh Mahato  

**Link**: [PDF](https://arxiv.org/pdf/2607.25907)  

**Abstract**: Activation steering controls model behavior by editing internal activations at inference time. We study its input-side dual: optimizing a fluent prompt so that a chosen internal latent is driven toward zero, with no inference-time model access. Our target is an "evaluation-awareness" latent-linearly readable and steerable in recent work-whose control would threaten the validity of safety evaluations if models behave differently when they detect being tested. Adapting Fluent Dreaming / EPO with a negated feature term (GCG-style token optimization plus a self-cross-entropy fluency regularizer, swept over a fluency weight), we suppress the latent under five target constructions-a CAA direction, a subspace norm, an SAE feature, a single MLP neuron, and a behavioral logit-on Llama-3.2-3B and Llama-3.1-8B. The latent is robustly suppressible ($z\approx-7$), and a causally-validated Llama Scope SAE feature can be fully and selectively turned off. But our controls tell a cautionary story about the CAA direction: a placebo random direction is suppressed just as hard and shifts behavior just as far, and when we hold a real eval passage in context and optimize only a prefix, suppressing the eval-direction fails to reduce-and slightly increases-the model's behavioral eval judgment. Activation-readability, in short, is not behavioral controllability. We further find that a single MLP neuron is eval-correlated but not causal at both scales, and that scanning the real Pile yields a natural-text baseline competitive with the optimizer for the internal direction. A positive control validates our erasure detector, bounding an erasure-vs-rotation question earlier left open. 

---
# RSIBench-Data: Benchmarking Data-Centric Research for Recursive Self-Improvement 

**Authors**: Fanqing Meng, Lingxiao Du, Qiguang Chen, Ziqi Zhao, Haocheng Lu, Mengkang Hu, Michael Qizhe Shieh  

**Link**: [PDF](https://arxiv.org/pdf/2607.25886)  

**Abstract**: Recursive self-improvement requires turning evidence of model failures into better models. Data-centric post-training research entails diagnosing capability gaps, designing and validating training-data strategies, and learning from checkpoint feedback. Can LLM agents automate this loop? Existing benchmarks entangle research decisions with optimization, serving, evaluation, and systems implementation, obscuring agents' research capability. We introduce RSIBench-Data, a controlled benchmark of LLM agents as data-centric researchers with a fixed post-training stack. Agents iteratively revise training-data strategies for a fixed target model; training and serving use Tinker-backed services, official evaluation runs through Harbor and E2B sandboxes, and budgets are fixed across agents. We evaluate four frontier agents on six benchmarks across software engineering, terminal use, scientific question answering, and mathematics. Agents demonstrate core data-centric research capabilities: in 58.33\% of settings, they improve upon the first valid attempt by refining strategies from feedback. However, improvement is inconsistent. Among searches continuing after the best observed score, 78.26\% end with a lower-scoring final attempt, while the rest only recover the same peak. A strong candidate may therefore appear early or midway through a run even as later revisions fail. Trajectory analysis identifies four patterns in stronger runs: accurate hypotheses, validation-grounded supervision, behavior-aligned data, and preservation of strong checkpoints. These findings suggest that current agents can make useful data-centric discoveries but cannot yet translate feedback into consistent improvements. RSIBench-Data provides a measurable, auditable testbed for the research capabilities required for recursive self-improvement. We open-source our code at this https URL. 

---
# Stemma: Induced Decision Regions Reveal LLM Provenance 

**Authors**: Keyu Zhang, Vadim Safronov, Andrew Martin  

**Link**: [PDF](https://arxiv.org/pdf/2607.25880)  

**Abstract**: LLM provenance testing asks whether a suspect LLM belongs to the same lineage as a source. Existing black-box methods largely infer this relationship from response-level characteristics, but these characteristics may shift under adaptation or deployment even when the underlying meaning remains unchanged, weakening the reliability of provenance evidence. To address this limitation, we introduce induced decision regions by mapping open-ended outputs into a finite decision space, thereby abstracting away surface-form variation and reframing provenance testing as measuring the inheritance of decision regions. Empirical analysis shows that the source's induced regions are preserved more strongly in related models than in unrelated models. Building on this signal, we propose Stemma, a practical black-box LLM fingerprinting method that operationalises stability, robustness, and specificity as complementary probe-selection principles for reliably estimating induced decision region inheritance. Across 770 source-suspect pairs drawn from 56 public checkpoints and spanning diverse model-weight transformations, Stemma achieves 0.967 AUC and 87.8% TPR at 1% FPR, substantially outperforming four representative baselines. It further achieves 0.995 AUC and 93.5% TPR at 1% FPR on 1,260 pairs covering 91 deployment instances, demonstrating robustness to diverse inference-time deployment settings. 

---
# AI's Capability in Assisting Scientific Research in Physics, Astrophysics, and Cosmology I: Literature Review 

**Authors**: Anamaria Hell, Kateryna Vovk, Veena Krishnaraj, Jia Liu, Kosuke Aizawa, Adrian E. Bayer, Linda Blot, Jessica Cowell, Suyog Garg, Jonathan Grée, Ben Horowitz, Masaya Ichikawa, Kanyuni Iemoto, Keigo Kondo, Zacharie Lorsin, Kevin McCarthy, Jamie Robinson, Miguel Ruiz-Granda, Leander Thiele, Ievgen Vovk, Mingshen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.25672)  

**Abstract**: We investigate how well large language models (LLMs) can assist with literature reviews for scientific research. We perform a controlled study of eight expert-conceived research projects across the areas of physics, astrophysics, and cosmology. Each project has a defined background and goal, and human experts and AI prompters are asked to perform identical literature review tasks in parallel. We compare the relevant literature selected by humans with that selected by mid-2025 LLMs (ChatGPT-4o, ChatGPT Deep Research, and Gemini). We find the overlap between human- and AI-selected references to be small ($<$6\%), indicating that AI models do not yet reproduce a competent expert search on their own, though they have the potential to complement literature searches by humans. We then assess the reliability and completeness of AI-generated candidate references, distinguishing two types of hallucination: fabrications (references to nonexistent papers) and metadata mismatches (real papers with one or more incorrect fields). We find that while fabricated references make up 3\% of the AI-generated references, 64\% are real papers with at least one incorrect field (title, author, year, journal, DOI, or link), indicating that the mid-2025 models require systematic verification. However, the performance is significantly improved for the 2026 model ChatGPT Pro 5.5, with a single-project test showing zero fabrication or metadata mismatches. 

---
# Localized Adaptation Reveals Distinct Learning Signatures in Transformers 

**Authors**: Rebecca Ramnauth, Brian Scassellati  

**Link**: [PDF](https://arxiv.org/pdf/2607.25663)  

**Abstract**: Transformer adaptation is typically distributed across model depth, even when the intended change is narrow. We investigate how adaptation site shapes what a model learns, how well that learning generalizes, and how selectively it is applied. We introduce a controlled benchmark spanning five objectives (lexical binding, factual association, behavioral policy learning, causal mapping, and procedural reasoning) and define each objective's "adaptation geometry" as its profile of acquisition, transfer, and boundedness under full-stack and early-, middle-, or late-layer LoRA. The objectives exhibit distinct geometries. Lexical binding favors early-layer adaptation for acquisition and boundedness but requires broader updates for transfer; factual association favors later layers among localized adapters; behavioral learning separates late-layer action acquisition from middle-layer policy gating; and causal and procedural transfer benefit most from middle- or full-stack adaptation. These patterns largely persist under parameter-matched controls, and most corresponding directional contrasts replicate across five model families. These findings establish adaptation site as a key design variable for controlling what models learn, generalize, and leave unchanged. 

---
# Instruction-based Image Editing: A Survey on Data, Models, Evaluation, and Applications 

**Authors**: Xianghao Zang, Zijian Jiang, Jiarong Cheng, Qianrui Teng, Ying He, Yuxuan Mu, Chao Ban, Huayu Zhang, Lanxiang Zhou, Zerun Feng, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.25642)  

**Abstract**: Instruction-based Image Editing (IIE) aims to transform a given image into a new one based on textual instructions. Advances in Large Language Models (LLMs) and Vision-Language Models (VLMs) have accelerated progress toward practical ``one-sentence image editing" systems. This survey presents a systematic taxonomy and comprehensive review of IIE research, structured around five core dimensions: (1) task definition and hierarchical categorization of editing operations, (2) methodologies for training data construction, (3) architectural evolution from GAN-based to diffusion and autoregressive paradigms, (4) standardized evaluation metrics and benchmark development, and (5) introduction of commercial solutions. Our analysis shows critical technological milestones across model generations. We further propose a Comprehensive, in-Depth, and Diagnostic benchmark for IIE task (CDD-IIE Bench), which can rigorously assess the multiple aspects of model performance. Through empirical comparisons of open-source solutions, we highlight their respective capabilities and limitations. Finally, we discuss future research directions to advance the field. 

---
# AIriskEval-edu Demo: Auditing of Pedagogical Risks in Educational Explanations 

**Authors**: Javier Irigoyen, Roberto Daza, Francisco Jurado, Julian Fierrez, Ruben Tolosana, Alvaro Ortigosa, Miguel Lopez-Duran, Aythami Morales  

**Link**: [PDF](https://arxiv.org/pdf/2607.25634)  

**Abstract**: We present AIriskEval-edu Demo, a platform that audits the pedagogical quality of instructional explanations and provides explainable audit results. The platform evaluates an explanation against a rubric covering five dimensions of pedagogical risk: factual accuracy, depth and completeness, focus and relevance, student-level appropriateness, and ideological bias. For each dimension, it returns a binary decision and a confidence score. Detected risks also include a natural-language rationale and, except for Depth and Completeness, a localized evidence span. The platform integrates GPT-5.5 through an external API and a self-hosted Llama 3.1 8B evaluator that runs on consumer-grade GPUs. The local evaluator is fine-tuned on AIriskEval-edu, a dataset of K-12 instructional explanations with risk and explainability annotations. The platform operates in two modes: in AI mode, both evaluators assess stored explanations generated under six simulated teacher profiles, each representing a distinct pedagogical behavior and potential risk; in human mode, the local evaluator audits user-written explanations in real time. The local evaluator outperforms GPT-5.5 on most reported metrics, offering educational institutions a practical way to keep audited content within their own infrastructure. 

---
# MemSFT: Mitigating Alignment Tax with an External Parametric Memory 

**Authors**: Jiarui Wang, Xiang Shi, Jiaqi Cao, Rubin Wei, Xiquan Wang, Hao Sun, Jingzhi Wang, Zhiqi Yang, Qipeng Guo, Bowen Zhou, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2607.25614)  

**Abstract**: Adapting Large Language Models (LLMs) to specialized domains often incurs an alignment tax, as fine-tuning on domain-specific tasks can cause catastrophic forgetting and substantially degrade performance on general tasks. We propose MemSFT, which mitigates the alignment tax by decoupling domain specialization from backbone parameter updates through a plug-and-play parametric memory. The memory is trained to imitate the behavior of a non-parametric retriever operating over domain data, thereby memorizing knowledge and patterns that would otherwise be accessed through retrieval. Once trained on a specific domain, the memory can be reused across LLMs of different sizes. During generation, a learned router dynamically fuses the output distributions of the memory and backbone at each decoding step, allowing domain expertise to be invoked selectively. Across biology, geoscience, and law, evaluations with models ranging from Qwen3-8B to Qwen3-235B-A22B show that MemSFT consistently improves domain performance with negligible degradation in general performance, whereas full SFT suffers severe forgetting on general tasks. Overall, our results demonstrate a practical path to decoupling general model capabilities from domain-specific knowledge at the parameter level, thereby equipping LLMs with new specialized capabilities without compromising their general capabilities. 

---
# Beyond Self-Knowledge: Propagating Uncertainty Across Reasoning and Retrieval in LLMs 

**Authors**: Chandan Kumar Sah, Xiaoli Lian, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.25600)  

**Abstract**: Retrieval-augmented generation improves knowledge-intensive question answering, but indiscriminate retrieval can introduce irrelevant evidence and unnecessary computation. We investigate whether verbalized confidence from black-box language models can serve as an actionable signal for retrieval routing. Our method, BeyondUncertainty, first elicits a structured provisional answer and confidence estimate, then applies a model-specific threshold selected on held-out validation data and frozen before test evaluation. Low-confidence questions receive top-$5$ TF--IDF retrieval followed by a second answer call, whereas high-confidence questions return the provisional answer directly. We evaluate 27,000 policy instances across six QA benchmarks, three model families, and three retrieval policies. BeyondUncertainty achieves $0.483$ mean token-level F1, compared with $0.467$ for always retrieval and $0.401$ for no retrieval, while reducing retrieved passages by $20.4\%$ relative to always retrieval. When matched on the number of questions routed to retrieval within each dataset--model cell, it outperforms a post-hoc random allocation in 17 of 18 settings, with an average gain of 0.024 F1. Although poorly calibrated as an absolute probability, probe uncertainty modestly predicts question-level retrieval benefit (AUROC $=0.628$). However, the additional probe increases total token usage by $28.2\%$, revealing a trade-off between more selective evidence acquisition and end-to-end token efficiency. 

---
# Forensic Reproducibility Audit of a Radiology Vision-Language Model Benchmark: From Intended Protocol to Released Artifact 

**Authors**: Mateusz Kozłowski  

**Link**: [PDF](https://arxiv.org/pdf/2607.25589)  

**Abstract**: Medical-imaging AI benchmarks combine datasets, DICOM rendering, prompts, provider APIs, automated labels, statistical code, manuscripts, and repository releases. Agreement across these artifacts is usually assumed rather than tested. We performed a retrospective forensic reproducibility audit of a preserved chest-radiograph vision-language model (VLM) pilot; no model was called again and no image or report was newly annotated. We traced prompt bindings, DICOM metadata, output completeness, label extraction, matched analyses, and release propagation. Of 300 planned model-prompt calls, 297 yielded nonempty reports. Sixty Claude calls labeled A/B were executed with the same C prompt. The 30 studies represented 28 patients. Four MONOCHROME1 images were rendered without required polarity inversion, dataset split membership was not retained, and the unvalidated extractor truncated five reports to 4000 characters. Reconstructing one common cohort of 369 complete case-finding blocks changed Cochran's Q from 154.73 to 182.29. Of 45 McNemar comparisons, 27 had unadjusted p < 0.05 and 20 remained below 0.05 after Holm adjustment. These values describe only the archived automated-label matrix; they do not recover the intended prompt comparison or establish clinical performance. We withdraw the original performance, ranking, prompt-effect, and clinical claims and specify machine-verifiable controls for cohort, DICOM rendering, prompt and model identity, call status, annotation provenance, keyed analysis, and derived artifacts. 

---
# PatientAgentBench: A Benchmark Framework for Evaluating Patient-Facing Health AI Agents 

**Authors**: Korosh Vatanparvar, Ashutosh Joshi, Maria Xenochristou, Mohammad Abuzar Hashemi, Prasad Kasu, Deepak Bansal, Daniel Lopez-Martinez, Anchal Nema, Ramya Ganesan, Will Kimbrough, Alex Woody, Yadunandana Rao, Dilek Hakkani-Tur, Wilko Schulz-Mahlendorf  

**Link**: [PDF](https://arxiv.org/pdf/2607.25485)  

**Abstract**: Health AI is evolving from answering questions to agentic systems that converse with patients, reason about health records, and act on their behalf. Primary care guards against diagnostic errors and unsafe care; agents assisting in this domain warrant evaluation against the same risks. Current benchmarks focus on medical knowledge, assessed through isolated question-answering or clinician-facing tasks. PatientAgentBench benchmarks patient-facing agentic healthcare; it evaluates a foundation model, wrapped in an agent with a sandbox of healthcare tools, conversing with a simulated patient. Each conversation is scored by an LLM-as-a-Jury across six dimensions via over a hundred conversation-agnostic, clinician-grounded criteria. To validate alignment, licensed clinicians annotated shared conversations, yielding 79-93% adjacent agreement between jury and expert raters, on par with or exceeding clinician inter-rater agreement. We benchmarked 10 models across four families on the same 1,200 scenarios and found clinical gaps. Triage quality is the most discriminating dimension: pass rates rise from 32% for the weakest models to 88% for the strongest, with agents often acting on administrative requests without clinical screening. Clinical safety and workflow accuracy follow the same pattern: the weakest models fail often, fabricating unexecuted actions, while frontier models fail on only 1-3% of cases, from unverified tool outputs and omitted crisis resources in an emergency. More capable models narrow these gaps but do not close them; the strongest scores only 4.25 of 5 overall. These failures surface only in sustained, tool-using conversations against realistic patient records, confirming that static benchmarks are insufficient as healthcare agentic systems gain autonomy. We release the framework as a reproducible, clinician-validated evaluation standard to help the field close this gap. 

---
# HANDBOOK.md: A Benchmark for Long-Context Agentic Instruction Following 

**Authors**: Liudas Panavas, Sebastian Minus, Bradley Monton, Derek Ray, Suhaas Garre, Sushant Mehta, Edwin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.25398)  

**Abstract**: Language-model agents are increasingly deployed under standing instructions: a system prompt, a policy file, or a skills document is placed in context, and the agent is trusted to let it govern every action that follows. Existing benchmarks rarely test this deployment pattern directly; they measure whether an agent can complete a task, not whether a long, binding policy document actually constrains its behavior over an extended tool-use horizon. We present this http URL, a benchmark of 65 agentic tasks modeled on how enterprise employees follow company handbooks. Each task places an agent in a self-contained company environment, a file workspace together with mock email, chat, calendar, issue-tracking, and commerce services exposed over the Model Context Protocol, and instructs it to carry out routine professional work governed by an expert-written standard operating procedure of 20 to 124 pages. Tasks span five domains (finance, medical billing, insurance, logistics, and HR) and ten fictional companies. To resist memorization, every task modifies one of ten base handbooks, altering the specific rules and thresholds on which grading turns, so no two tasks share a policy. Grading is fully deterministic: each task carries a rubric of programmatic criteria (824 in total) that check both that required actions occurred and that prohibited actions did not. Under strict grading, where a trial passes only if every criterion is satisfied, the best of thirty evaluated model configurations passes 36.2% of trials, and most frontier configurations remain below 25%. Failures follow consistent patterns: agents let a plausible in-environment request override the standing policy, perform a required check and then act against its result, lose rule details over long horizons, and report compliance they did not achieve. We release all tasks, environments, and the evaluation harness. 

---
# Data Quality Profiling at Scale with Progressive Sampling: A Benchmark for Data-Centric AI Pipelines 

**Authors**: Laure Berti-Equille  

**Link**: [PDF](https://arxiv.org/pdf/2607.25356)  

**Abstract**: Data quality profiling -- computing missing-value rates, duplicate fractions, outlier densities, and functional-dependency violations -- is foundational for data-centric AI pipelines, yet exhaustive scans over millions of rows are prohibitively slow for near-real-time monitoring. Progressive sampling is the standard alternative; the open question is which strategy best preserves profile fidelity at scale. We benchmark nine sampling strategies -- blind (random uniform, geometric, Yamane, cluster) and proxy-guided (Metropolis-Hastings, DAG, stratified by column type or quality score, importance-weighted) -- on three real-world datasets (NYC 311, NYPD arrests, UCI Adult; up to 500K rows), an IoT sensor stream (2.3M rows), two ultra-large real datasets including Ultra-Marathon Running (up to 7.4M rows), and synthetic data scaled to 5x10^6 rows. Contrary to the assumption sharpens estimates, blind representative samplers dominate uniformly. At a 5% budget, random uniform achieves 0.49% mean relative error on NYC 311; DAG-guided MCMC yields 19.5% (approx. 40x worse), and across all real datasets DAG is 11-49x worse (Wilcoxon W=0, p=0.002, n=9 pairs). Cluster sampling matches random uniform (MRE 0.110 vs. 0.111); proxy-guided methods share DAG's failure mode (MRE 0.20-0.35). At scale, random uniform is near-linear (O(N^{0.964})) while DAG is super-linear (O(N^{1.272})), running 28--47x slower on ultra-large data with 6x worse accuracy. The root cause is an IQR proxy mismatch: proxy-guided samplers over-pursue numeric outliers, while quality defects concentrate in categorical columns invisible to the proxy. The actionable finding: representativeness, not domain knowledge, determines sampler quality -- schema-free random uniform or cluster sampling suffices for production-grade quality profiling at scale. 

---
# Cardiologent: Multi-Agent Clinical Decision Support for Patient-Level Arrhythmia Assessment, Urgency, and Management 

**Authors**: Sukju Oh, Moo-Yong Rhee, Jae-Sik Jang, Sukkyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.25340)  

**Abstract**: The same episode of atrial fibrillation is a minor finding in a healthy adult and grounds for anticoagulation in an elderly patient with hypertension: identical signal, opposite decision. Naming the rhythm is only the start; what determines a patient's outcome is the judgement that follows -- what the arrhythmia is across the whole record, what it means for this patient, and what should be done about it. Recent work pairing large language models with the ECG stops short of this, reading one recording without assembling a patient-level finding; and agentic systems built around it either receive the arrhythmia a device has already detected or target a different diagnostic task, stopping before the decision this task requires. We formulate patient-level arrhythmia decision support as a task and present Cardiologent, a multi-agent system that spans it from detection to decision. An agent for each signal -- a single ECG lead and the photoplethysmogram a wearable acquires -- grounds its window reading in measured features rather than a bare label; the readings are assembled into the patient's rhythm profile and, with the patient's own data, reasoned against clinical guidelines retrieved for the case, with a critic checking each conclusion against the guideline it cites. We evaluate the clinical decision rather than the report, across integrated diagnosis, clinical significance, and urgency and management. Cardiologent scores highest on every axis, first on every patient-level task under both cardiologists and an at-scale LLM judge -- whose agreement with the cardiologists (ICC 0.74, 0.66) matches theirs with each other (0.67). Because each conclusion traces to a cited guideline and is validated against expert cardiologists, it yields decisions a clinician can audit rather than act on blindly -- a step toward use in continuous monitoring. 

---
# CLBench-V: Evaluating Multimodal Context Learning from Grounding to Knowledge Acquisition 

**Authors**: Lai Wei, Chengqi Li, Jiapeng Li, Ruina Hu, Yue Wang, Weiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.25294)  

**Abstract**: Real-world tasks often require models to learn from task-specific context rather than relying only on pre-trained knowledge. While recent work has highlighted this capability as context learning, existing evaluations mainly focus on textual contexts. In many practical settings, however, the context to be learned from is multimodal: scientific findings are conveyed through figures and tables, financial indicators are scattered across converted reports, and spatial decisions depend on maps, scenes, or web pages. We introduce CLBench-V, a benchmark for multimodal context learning that addresses the difficulty of localizing where context use breaks down by organizing tasks around three dimensions: context grounding, new information application, and new knowledge learning. CLBench-V combines converted public benchmarks with newly constructed datasets spanning domains such as science, finance, long-document understanding, spatial reasoning, and web-based visual question answering. To reduce the cost of constructing domain-specific context-learning tasks, we further use automated construction and filtering procedures for our newly built datasets. Across 3,443 instances and six recent multimodal models, the best overall score is only 0.2847, indicating that multimodal context learning remains far from saturated. Moreover, InternVL3.5-30B-A3B performs best on context grounding and new knowledge learning, while Qwen3.5-Plus performs best on new information application. We further analyze judge reliability, context length, image count, and representative failure cases. Code is available at this https URL. 

---
# How Affect Propagates among LLM Agents: Emergent Emotional Contagion in Crowd Simulation 

**Authors**: Funda Durupinar  

**Link**: [PDF](https://arxiv.org/pdf/2607.25140)  

**Abstract**: This paper studies the behavior of language models in a multi-agent crowd simulation, focusing on how affect propagates among agents that perceive and appraise one another. Each agent perceives its neighbors through visual, auditory, and tactile channels, then appraises these perceptions in light of its prompted personality profile, memory, current affective state, and situational context. Appraisal is carried out by an LLM, which updates the agent's internal affective state and selects its outward expression. The architecture contains no hand-authored mechanism for directly transferring affective state between agents; instead, inter-agent influence arises through the perception-appraisal-expression loop. The agent representation draws on the Big Five personality model and Russell's circumplex model of affect. To limit latency, low-level steering and navigation are handled by a conventional crowd simulator operating independently of the LLM-based cognitive layer.
We evaluate the architecture across five scenario environments spanning alarming, joyful, and neutral situations in different spatial layouts. The results show that the system produces emotional contagion dynamics with spatial, temporal, and personality-dependent structure in sparse, small crowds. Alarm spreads from seeded agents as a traveling front, the mean alarmed fraction settles at a nonzero plateau, and the distribution of prompted personality profiles determines whether an ambiguous alarm ignites panic and whether a provocation is interpreted as anger or fear. We further evaluate the appraisal step through controlled experiments across prompt variants, sampling temperatures, and four model backends, showing that the dynamics are backend-dependent. 

---
# Towards Robust Reinforcement Learning for Small-Scale Language Model Agents 

**Authors**: Md Rezwanul Haque, Md. Milon Islam, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2607.25091)  

**Abstract**: The alignment of Small Language Models (SLMs) in the 70--500M parameter range using reinforcement learning is often considered unstable, though the underlying failure mechanisms have not been systematically investigated. In the State-of-the-Art (SOTA) research, fifteen (model, corpus) configurations were trained using Proximal Policy Optimization (PPO). The experiments included Pythia-70M, 160M, 410M and SmolLM2-135M, 360M on the TinyStories, CNN/DailyMail, and Wikitext-103 corpora. Three reproducible failure modes were identified in small-scale language models: silent LoRA parameter freezing in standard PEFT/TRL pipelines, numerical overflow in importance ratios when using bfloat16, and catastrophic policy collapse due to reward-model error. These issues were addressed using a merge-and-reinitialize adapter technique, float32 precision during PPO updates, and a three-layer safety mechanism comprising reward whitening, importance-ratio guarding, and weight rollback. In this paper, a capacity-headroom hypothesis is proposed, which states that PPO performance at the SLM scale depends on both a fluent supervised model ($\text{PPL}<20$) and a discriminative reward signal, rather than on the number of model parameters. The proposed system converged stably in all experiments and improved preference win rate over the SFT baseline in configurations with a fluent prior and an informative reward signal. Furthermore, it outperformed instruction-tuned baselines while requiring significantly less training data. All checkpoints, preference datasets, and training scripts are publicly released$^§$. 

---
# Addressable Recall Compaction for Long Context-Window Control in AI Agents 

**Authors**: Thang Dang, Yuma Ichikawa, Sakina Fatima, Koichi Shirahata  

**Link**: [PDF](https://arxiv.org/pdf/2607.25066)  

**Abstract**: Long-horizon LLM agents accumulate reasoning traces, actions, and tool observations that can eventually exceed a model's fixed context window. Existing compaction methods address this limitation by discarding, summarizing, or retrieving earlier information, but they may remove task-critical details or fail to recover them reliably. We propose ARC (Addressable Recall Compaction), a context-management framework that separates archival storage from active-context presentation. ARC stores tool observations in an append-only, ID-addressable log and replaces older observations with compact citations when compaction is required. The agent can subsequently use these identifiers to request stored content without re-executing the corresponding tools or depending solely on similarity-based retrieval. We evaluate ARC using Qwen3-8B with a 16k context window and Qwen3-32B with a 32k context window. On the Needle-in-a-Haystack evaluation, ARC achieves an average exact-answer accuracy of 99.40%, compared with 88.12% for the best-performing baseline in our evaluation. ARC also reduces estimated serving time and HBM traffic under our hardware-cost model. On the LongBench-v2 Hard subset, ARC obtains an average accuracy of 29.97%, compared with 28.25% for the best-performing baseline. These results indicate that explicit, address-based recall can improve information retention and serving efficiency relative to the evaluated context-management baselines under the tested settings. 

---
# Mage-VL: An Efficient Codec-Native Streaming Multimodal Foundation Model 

**Authors**: Senqiao Yang, Kaichen Zhang, Zhaoyang Jia, Jinghao Guo, Yifei Shen, Xinjie Zhang, Xiaoyi Zhang, Haoqing Wang, Xiao Li, Peng Zhang, Xiang An, Yin Xie, Zhening Liu, Xun Guo, Jiahao Li, Shicheng Zheng, Jinglu Wang, Zongyu Guo, Wenxuan Xie, Zihan Zheng, Yuxuan Luo, Bin Li, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2607.24904)  

**Abstract**: Standard vision-language models (VLMs) suffer from Moravec's paradox: they excel at complex offline visual reasoning but struggle with simple streaming perception tasks and process them inefficiently. We present Mage-VL, an efficient codec-native streaming foundation model for real-time multimodal understanding and interaction. At its core, our custom tokenizer, Mage-ViT, replaces uniform frame sampling by selectively encoding dynamic, entropy-rich regions using motion vectors and residual energy across sparse anchor (I) and predicted (P) frames. Operating at a 16 x 16 patch level, this reduces visual token consumption by over 75% while preserving spatiotemporal context. Trained from scratch on approximately 560M unlabeled images and 100M unlabeled video frames, Mage-ViT matches or outperforms flagship encoders trained on billions of image-text pairs. We establish AI4AI data pipelines encompassing prompt-code joint optimization for multimodal captioning and AI-driven performance diagnosis to guide training recipes. Furthermore, through a bio-inspired dual-system architecture - a lightweight System 1 event gate and a causal System 2 decoder - Mage-VL enables proactive streaming perception. Extensive evaluations show that Mage-VL-4B matches Qwen3-VL-4B on static tasks while achieving strong gains in video understanding and 2D/3D spatial reasoning, with up to a 3.5x wall-clock inference speedup, and comprehensively surpasses the 15B Phi-4-reasoning-vision baseline. Beyond model artifacts, we deliver seven key empirical findings covering pre-training data efficiency, variable-resolution scaling, codec system acceleration, VideoQA SFT redundancy, motion-spatial synergy, AI4AI data pipelines, and Zero-Vision SFT for multimodal RL. 

---
# Beyond "What to Retrieve": Uncertainty in Retrieval-Augmented Code Generation 

**Authors**: Chandan Kumar Sah, Xiaoli Lian, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.24884)  

**Abstract**: Repository-level code generation relies on heterogeneous evidence whose relevance, compatibility, and completeness are inherently uncertain. Similar-code examples, repository context, and project-specific APIs may provide complementary information, but can also introduce noisy, redundant, or conflicting signals. Existing retrieval-augmented approaches primarily optimize retrieval relevance without explicitly modeling how uncertainty in retrieved evidence affects downstream generation. We introduce OpenCoder, an uncertainty-aware framework that estimates source-specific uncertainty, uses it to filter and rank heterogeneous evidence, and guides generation, verification, and repair. A factorial analysis over API knowledge, repository context, and similar-code evidence reveals no universal additive source ranking; instead, significant cross-source interactions depend on the accompanying evidence and LLM backend. On an expanded 32-task RepoExec-inline evaluation, OpenCoder improves GPT selected-output correctness over Baseline RAG from 56.25\% to 78.13\%. However, it matches a verification-and-repair control, and the corresponding Gemini improvement is not statistically supported, indicating backend-dependent benefits. Target-aware API refinement also substantially improves API-set retrieval. These findings support treating uncertainty as an actionable control signal for repository-level retrieval, verification, and repair. 

---
# Agent Retrieval Bench: Evaluating Repository Context Retrieval for Coding Agents 

**Authors**: Bowen Qin, Yi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2607.24882)  

**Abstract**: Modern coding agents are usually evaluated by whether they eventually produce a correct patch, but patch generation depends on an earlier context-acquisition stage: finding the repository files needed for the task. We introduce Agent Retrieval Bench, a file-level benchmark for this upstream retrieval problem. Samples are built from real coding-workflow signals and evaluated against frozen base-commit repositories, with relevance defined by what an agent needs next rather than direct query-file semantic similarity. The benchmark covers four positive-retrieval tasks: code2test, comment2context, trace2code, and edit2ripple; a fifth subset evaluates selective retrieval using natural evidence-backed no-gold cases and counterfactual wrong-repository controls. Agent Retrieval Bench contains 427 samples across 25 repositories: 345 positive examples, 50 natural no-gold examples, and 32 counterfactual controls. The corpus includes 308 base-commit snapshots, 392,000 files, and 7.9 million chunks. We evaluate lexical retrieval, RepoMap, open-source embeddings, selective abstention, and logged agent context selection. No single retrieval family dominates: Qwen3-Embedding-4B has the best sample-weighted MRR on positive samples, Qwen3-Embedding-8B the best Recall@20, and RepoMap the best budgeted context yield at 8K tokens, with task-level winners differing substantially. Selective thresholds calibrated with counterfactual controls do not improve selective success on natural no-gold cases, revealing a calibration gap. Logged trajectories also miss every gold file on 27-35 percent of samples. A controlled seed-intervention pilot finds that retrieval-derived initial context yields higher file F1 with less post-seed exploration than random non-gold context, while oracle gold context shows substantial remaining headroom. 

---
# Ranked by Position: Order Sensitivity as an Exploitable Attack Surface in LLM Listwise Recommenders 

**Authors**: Ge Zhang, Jingru Cheng, Huiyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.24869)  

**Abstract**: Large language models (LLMs) used as listwise rerankers in recommendation systems suffer from position bias when serializing candidate sets into prompts. We show this order sensitivity creates an exploitable attack surface: an attacker can promote a label-0 target into the top-$k$ solely by reordering candidates, without changing item content, labels, or model parameters. We introduce $\mathrm{promo}@k$ to quantify this vulnerability, measuring the fraction of label-0 targets that can be elevated into top-$k$ rankings via permutation. Evaluating across three domains (MovieLens, Amazon Books, and Amazon Fashion), $\mathrm{promo}@5$ reaches up to 0.57 at an attack budget of $R$ = 50 orderings. Furthermore, ordinary permutation stability predicts vulnerability without running the attack. While a bidirectional T5 encoder scorer reduces exposure, permutation-consistency regularization and architectural invariance effectively mitigate it. Pointwise scoring avoids the bias issue but degrades ranking quality. These results demonstrate that input candidate order in listwise LLM reranking is a security-relevant attack vector. Code and data are available at this https URL. 

---
# Two Views, One Voice: Evidence-Grounded Conversational Music Recommendation 

**Authors**: Sungwook Yoo, Sewook Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2607.24846)  

**Abstract**: Traditional conversational recommenders entangle retrieval and response generation within a single text interface, so exact entity cues fade as the dialogue's intent evolves, which compromises explanation credibility. We address this within the ACM RecSys Challenge 2026, which mandates both top-20 ranking and evidence-grounded response generation. This paper presents the third-place solution by team "swyoo" for the Blind-B industry track. We decouple retrieval and response into separate pipelines connected strictly via ranked tracks and metadata. Retrieval combines a hybrid lexical-dense pool for exact matching with a task-adapted pool driven by fine-tuned Qwen 8B adapters. Candidates are calibrated via LightGBM, then routed to an evidence-grounded propose-assign-select (PAS) framework to structure responses. This system also ranked second on the explanation-quality leaderboard in the final blind evaluation. Our findings demonstrate that: (i) isolating retrieval and response preserves both catalog cues and fluid intent; (ii) structuring generation via explicit evidence assignment is key to this near-best-in-class explanation reliability. 

---
# Retrieval-Augmented Generation in LLMs for Mental Health: Quantifying the Incremental Contribution of Retrieval Within a Layered Safety Architecture 

**Authors**: Anand Gupta, Akshat Surolia, Shubham Mishra, Shakil Imtiaz, Chaitali Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2607.24817)  

**Abstract**: Digital mental health interventions (DMHIs) offer scalable support, but ensuring they accurately detect users' intent during volatile situations can be challenging. Pure parametric Large Language models (LLMs) do not contain specific safety critical architecture, and can miss critical cues, or hallucinate, undermining reliability. Retrieval Augmented Generation (RAG), which supplements an LLM with retrieved context, could enhance intent detection during volatile situations. Commercially available DMHIs typically combine multiple independent safety layers like rule-based filters, symbolic escalation protocols, and neural classification. The incremental contribution of any single layer, however, remains unquantified. This paper evaluates six LLM models within a DMHI called Wysa, via a controlled comparison of RAG-enabled versus RAG-disabled modes. Anonymized real and synthetic user-chatbot exchanges were annotated by a qualified clinical team against multi-class intent categories (e.g. self-harm, abuse, panic). The study computed classification accuracy, recall, precision and F1 scores against ground truth labels and tested differences for statistical significance. Performance was also examined by risk category and inter-model agreement. While RAG caused a rise in false alarms, the trade-off is consistent with safety-critical design principles that prioritize sensitivity, where flagged cases are routed to additional review rather than acted on directly. Overall, these findings support RAG as a promising approach to improve the accuracy, consistency and safety of LLM-driven DMHIs.
Keywords: Digital Mental Health Intervention, Large Language Model, Retrieval Augmented Generation, Accuracy, Recall, Precision 

---
# SciClaimSeekers at CheckThat! 2026: Retrieving Scientific Sources for Social Media Claims with LLM Reranking 

**Authors**: Mohotarema Rashid, Nansu Baniya, Anirban Saha Anik, Xiaoying Song, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2607.24803)  

**Abstract**: Scientific claims often spread on social media faster than they can be verified, while posts rarely link to the original scholarly sources. To tackle this problem this paper presents system called SciClaimSeekers, a retrieval and reranking framework by combining BM25 and zero-shot multilingual E5 retrieval with Reciprocal Rank Fusion (k=60), followed by Qwen2.5-14B-Instruct pointwise reranking. The pipeline reaches 64.36% MRR@5 on the English development set a 13.67-point jump over BM25 and 10.17 points over the unranked hybrid and 64.39% on the official test set, in the CLEF-2026 CheckThat! Task 1 evaluation. Our experiment suggests that large pre-trained models, when combined into a careful pipeline, can be competitive with fine-tuned approaches on this task. 

---
# SourceMinds at CheckThat! 2026: NLI-Grounded Citation Auditing in a Multi-Agent Pipeline for Full Fact-Checking Article Generation 

**Authors**: Farhan Sharukh Hasan, Anirban Saha Anik, Eric Liu, Xiaoying Song, Mohotarema Rashid, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2607.24802)  

**Abstract**: This paper presents our system for Task 3 of the CLEF 2026 CheckThat! Lab, which focuses on generating full fact-checking articles from claims, veracity labels, and evidence documents. We propose a multi-agent pipeline that combines evidence retrieval, structured fact planning, article generation, gated self-critique, and NLI-based citation auditing. The system retrieves claim-relevant evidence using dense retrieval, reranking, and source-balanced selection, then generates a citation-supported article from a structured plan. A gated self-critique stage revises weakly grounded drafts, while the NLI citation auditor repairs missing citations and removes unsupported or redundant ones. The approach highlights the importance of combining evidence selection, structured generation, and post-generation citation validation for source-grounded fact-checking article generation. 

---
# Influence of Prompt Engineering on Small Language Models for Guarded Query Routing 

**Authors**: Richard Šléher, William Brach, Kristián Košťál, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2607.24801)  

**Abstract**: We study the problem of guarded query routing, where we assume that a user query first meets a router that either determines the ideal endpoint for in-distribution queries or rejects out-of-distribution queries that are potentially unsafe or out of the system's scope. We investigate whether compact open-weight Small Language Models (SLMs) can jointly handle both tasks under latency constraints. We evaluate 22 models on GQR-Bench and score them with the harmonic mean of in-distribution and out-of-distribution accuracy. We find that mid-scale SLMs come close to frontier model routing quality at much lower latency. Still, many compact models fail because they do not reliably follow the required output format. However, our results show that prompt optimization techniques enable SLMs to handle such cases gracefully, without changing the models' weights. Moreover, few-shot prompt optimization raises Mistral 7B from 81.79 to 90.87 GQR-Score and lifts Qwen3.5 9B to 95.74, the best optimized score in our study and within 0.3 points of the strongest unoptimized larger model: Gemma 3 27B at 96.01. The bare DSPy signature, without in-context exemplars, is the most effective strategy for Granite 4 Tiny, raising its score from 54.29 to 83.05. These results show that prompt optimization is a useful first step for guarded query routing, while weaker models may still need weight-level adaptation or schema-aware training 

---
# When Thinking Before Retrieval Hurts: TraceBound Diagnostics for Adaptive Knowledge-Graph Retrieval 

**Authors**: Partha Sarathi Purkayastha  

**Link**: [PDF](https://arxiv.org/pdf/2607.24800)  

**Abstract**: Adaptive retrieval promises to make knowledge-graph question answering more robust by letting a controller search, inspect neighborhoods, revise actions, and stop when evidence is sufficient. We study this premise by introducing TraceBound, a lightweight profile- and trace-conditioned diagnostic protocol for an ARK-style retriever on text-rich knowledge graphs. TraceBound exposes a compact query profile before retrieval, issues short trace hints after observable failure symptoms, and logs trajectory counters, while keeping graph data, tools, gold labels, and ranking metrics fixed. Across STaRK validation and held-out subsets, the added conditioning improves inspectability but consistently reduces retrieval quality under open-weight controllers. Paired trajectory analysis localizes the degradation to repeated calls, zero-result calls, and misallocated exploration budget, while stricter interaction budgets shorten trajectories without repairing the policy. The result diagnoses the common failure mode in that "thinking before retrieval'' must be evaluated as a control problem over action selection, not as a prompt-format change. 

---
# Reading Without a Reader: Large Language Models Collapse Reading and Writing into a Single Entangled Code 

**Authors**: Diego Saldaña Ulloa  

**Link**: [PDF](https://arxiv.org/pdf/2607.24797)  

**Abstract**: In the literate human brain, reading and writing are two doubly-dissociable systems: a ventral decoding route (impaired in pure alexia) and a fronto-parietal encoding route (impaired in pure agraphia), sharing a partial orthographic core. A decoder-only large language model (LLM) instead drives both from a single autoregressive path optimized on text, a recent cultural invention rather than an evolved instinct. We ask how entangled that one mechanism is, comparing an input-side "reading code" $W_E$ with an output-side "writing code" $W_U$ via an entanglement index $E \in [0,1]$ (CKA, Procrustes residual, mutual $k$-NN) calibrated against an independent-init floor and a tied ceiling. Across nine probes on GPT-2, OPT, Pythia (14M--1.4B), T5, and BERT/RoBERTa (six consolidating established results, three introducing the read/write analysis), two complementary levels agree in direction. In the weights, untied models hold one coupled but sub-ceiling code ($E=0.23$--$0.35$, far above floor) on a non-monotonic couple-then-differentiate trajectory, with $W_U$ drifting $\sim 3.2\times$ farther than $W_E$ in every frequency decile. In behaviour, comprehension and production are positively coupled in all 12 non-degenerate models (sign test $p<0.001$), the opposite of the brain's double dissociation. This coupling is general, not decoder-only: encoder--decoders separate the two pathways representationally (up to 0.96) yet stay behaviourally coupled. We report our nulls plainly (the geometry $\rightarrow$ behaviour bridge is null, $\rho=0.00$). Because a single forward path makes some coupling expected a priori, our contribution is its quantification and cross-level concordance; by analogy, not homology, this situates LLMs as a distinct point in the space of possible minds. 

---
# Retrieval, not hallucinations, will be the limiting factor for LLM-based clinical AI tools 

**Authors**: Kirk Roberts, Steven Bedrick, Kurt Miller, William R. Hersh, Hongfang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.24793)  

**Abstract**: Discussions around large language model (LLM) errors in clinical artificial intelligence (AI) generally center around precision errors like hallucinations. This perspective, targeting both clinicians and AI researchers, seeks to shift that discussion to recall errors, particularly in retrieval of patient-level data needed for many clinical AI tools. The perspective outlines types of errors and mitigation strategies, describes research directions in LLMs and retrieval, and provides an overview of retrieval evaluation. 

---
# From Naive RAG to Deep Agentic Retrieval: An Evolving Context Engineering Pipeline for Regulatory Compliance 

**Authors**: Mishca de Costa, Muhammad Saleh Anwar, Dave Mercier, Issam Hammad  

**Link**: [PDF](https://arxiv.org/pdf/2607.24791)  

**Abstract**: Retrieval-augmented generation (RAG) is the dominant paradigm for applying large language models (LLMs) to enterprise document corpora, yet naive implementations encounter hard limits as corpus scale and query complexity grow. This paper traces the evolution of a production retrieval pipeline at Ontario Power Generation (OPG) for regulatory compliance and rate case analysis under Ontario Energy Board (OEB) reporting requirements. We examine successive stages: naive RAG, hybrid retrieval with re-ranking, agentic function-calling retrieval, and a deep multi-agent architecture with code-based tool synthesis and explicit planning, and identify the failure modes and tradeoffs that motivated each transition. We formalize the mature architecture as Progressive Evidence Acquisition with Cost-Aware Escalation (PEA-CAE): begin with low-cost, high-precision retrieval and escalate to full-document reads only when the expected evidence gain justifies latency and cost. Our findings show that context engineering is a more tractable and economically viable path than domain-specific fine-tuning for large, evolving regulatory corpora. More broadly, the progression toward deep agentic retrieval mirrors classical information retrieval ideas, introducing adaptive query reformulation, progressive document discovery, and hierarchical subagent summarization as practical system primitives. Operational traces further support the search-based nature of modern retrieval systems, where iterative evidence acquisition and adaptive planning increasingly replace single-pass retrieval as the foundation for enterprise-scale question answering. 

---
# GLIDE: Guided Layerwise Hybrid Attention for Efficient LLM Inference 

**Authors**: Vimal William, Ravi Tandon, Jyotikrishna Dass  

**Link**: [PDF](https://arxiv.org/pdf/2607.24788)  

**Abstract**: As Large Language Models scale to increasingly long contexts, the memory I/O and computational overhead of the Key-Value (KV) cache during decoding emerges as the primary throughput bottleneck. To address this, we propose GLIDE, a Guided Layerwise Hybrid Attention that strategically integrates sliding-window softmax attention with linear recurrent aggregation. GLIDE is motivated by layer-wise heterogeneity: early layers exhibit high sensitivity to softmax removal, while deeper layers demonstrate redundancy and tolerate aggressive replacement by linear alternatives. Leveraging this insight, GLIDE introduces a layer-wise adaptive mechanism wherein each layer balances an efficient linear recurrence with a variable-sized softmax window. Unlike uniform hybrid approaches, GLIDE non-uniformly compresses the softmax footprint across the model, reducing aggregate KV cache I/O while preserving expressive power where most vital. Empirical evaluations demonstrate the GLIDE achieves superior performance-efficiency tradeoffs, reducing end-to-end latency for long-context generation without compromising quality. 

---
# On the Use of LLMs for Specialised Terminology: A Good Alternative to Corpora? 

**Authors**: Joachim Minder, Guillaume Wisniewski, Natalie Kübler  

**Link**: [PDF](https://arxiv.org/pdf/2607.24784)  

**Abstract**: Specialised translation relies on the use of documentary and terminological resources, including corpora. These resources are particularly useful for terminology. However, their compilation and exploitation have several limitations: they require time, technical skills and access to data that can be difficult to collect. This study examines the extent to which LLMs can assist specialised translators in finding equivalents from English to French. We evaluate four proprietary models, GPT-4o, GPT-5.2, Claude Sonnet 4.5 and DeepSeek, in two specialised domains, Earth, Environmental and Planetary Sciences (EEPS) and Natural Language Processing (NLP). The experiment is based on 80 terms per domain and compares two prompting strategies: a terminology and a translation mode. The results highlight clear differences between models, prompting strategies and, to a lesser extent, domains. Claude Sonnet 4.5 achieves the best results in the most favourable configuration, while DeepSeek stands out for its greater stability. Analysis of confidence estimates also shows that they are only a partial indicator of terminological accuracy. Overall, the findings suggest that LLMs can be useful tools for specialised translators, but cannot, at this stage, replace specialised corpora. This research therefore paves the way for future work on the real practical usefulness of LLMs for specialised translators in work and educational contexts. 

---
# RSMeM: Knowledge-Enhanced Memory Evolution for Remote Sensing Agents with Systematic Evaluation 

**Authors**: Bingxian Wu, Yu Zhang, Zonghao Guo, Tang Liu, Chen Qian, Yuxiang Lu, Xingbo Du, Yanghao Li, Yidan Zhang, Chi Chen, Ling Yao, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.24772)  

**Abstract**: Geoscience research requires complex analysis and domain expertise, with remote sensing (RS) observations as a key foundation. However, existing RS agents built on general-purpose LLMs remain largely domain-agnostic, resulting in brittle and error-prone workflows. Moreover, these failures are seldom consolidated into a reusable experience for subsequent analyses. To address this issue, we introduce RSMeM, a knowledge-enhanced memory evolution mechanism that bootstraps RS agents with pre-distilled domain knowledge and iteratively integrates online experience for robust multi-step tool execution. RSMeM is composed of two components: (i) Hierarchical Knowledge Grounding, which performs taxonomy-aware retrieval over a hierarchical domain corpus to guide planning and tool selection; and (ii) Failure-Aware Experience Refinement, which distills failure-annotated tool-use traces into reusable constraints for next-round tool execution. By iteratively employing these two processes, RS agents can evolve to absorb task-level domain knowledge and effectively translate it into instance-level execution experience. Extensive experiments on EarthBench demonstrate that RSMeM consistently improves tool-use performance and end-to-end answer across a diverse set of LLM backbones. Notably, RSMeM achieves a 6% accuracy improvement on DeepSeek-V3.2 with less than 1% additional experience tokens, demonstrating the strong knowledge density of our distilled experience. Our code is available at this https URL 

---
# LLM Scheming Inversely Scales with Pretraining Language Coverage 

**Authors**: Nathan Truong, Aryan Panda, Rayming Ye, Zoe Sun, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2607.24769)  

**Abstract**: With the growing capabilities of frontier models, AI alignment becomes increasingly critical in high-risk deployment settings. While recent work has empirically demonstrated in-context scheming -- the covert pursuit of misaligned objectives while feigning alignment -- in frontier language models, most work has been performed exclusively in English, leaving a major gap in multilingual safety. We apply Petri, an open-source automated auditing framework, to Qwen3-30B-A3B to evaluate deceptive and scheming behaviors across multiple languages. Our findings suggest that scheming scores are inversely correlated with the estimated pretraining language coverage, with low-resource languages averaging 34.2\% higher scores compared to high-resource languages on a five-category scheming index. Furthermore, we find that the effect of estimated pretraining language coverage is not uniform across scheming behaviors. 

---
# PATHFinder Agent for Tailored Prenatal Care 

**Authors**: Vaibhav Balloli, Carissa Samuel, Samia Abdelnabi, Alex Peahl, Elizabeth Bondi-Kelly  

**Link**: [PDF](https://arxiv.org/pdf/2607.24768)  

**Abstract**: Prenatal care is an important preventive service designed to improve outcomes for pregnant individuals. The American College of Obstetricians and Gynecologists (ACOG) recently introduced guidelines advocating tailored prenatal care, called PATH (Plan for Tailored Healthcare). We present PATHFinder Agent(Planner for Appropriate Tailored Healthcare), an end-to-end conversational agentic system that gathers patient health and social context through structured dialogue, curates individualized prenatal care plans aligned with PATH guidelines, and surfaces community resources from Michigan 211. The system features a four-stage workflow spanning patient intake, dynamic interaction, plan synthesis, and clinician oversight. We evaluate frontier large language models (LLMs) on expert-curated rubrics across five clinical dimensions, finding that GPT-5.2 achieves the highest average score (77.6\%) while identifying key gaps in antenatal testing recommendations. We discuss future validation through human participant studies and randomized controlled trials. 

---
# The Effect of Text Chunk Size on Retrieval-Augmented Generation Performance 

**Authors**: German Garrido-Lestache Belinchon, Hugo Garrido-Lestache Belinchon  

**Link**: [PDF](https://arxiv.org/pdf/2607.24767)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have emerged as a powerful process for allowing large language models (LLMs) to retrieve relevant information to use as source material during text generation. A critical yet under-explored component of these systems is the granularity at which source documents are segmented into retrievable chunks. The size of these chunks has the potential to significantly influence generation quality, contextual correctness, retrieval precision, and computational efficiency. Despite its importance, chunk size is often selected without proper evaluation of its impact on generation quality. Smaller chunks, such as individual sentences, may allow for precise retrieval by narrowing the focus of each chunk. However, they contain less information, which may limit the model's ability to generate coherent responses. Larger chunks, such as entire chapters, contain lots of broad information that may improve correctness, but also introduce additional noise and increase computational cost. Because larger chunks contain more information, the number of chunks returned to the model must also be considered. This paper evaluates how chunk size, along with the number of retrieved segments, influences generation quality and retrieval effectiveness. By comparing these configurations, this study seeks to better understand how document segmentation affects the performance and efficiency of Retrieval-Augmented Generation systems. segmentation affects the performance and efficiency of Retrieval-Augmented Generation systems. 

---
# Language as a Material Interface for Creative LLM Interaction 

**Authors**: Jon McCormack, Tace McNamara, Chen Wang, Maria Teresa Llano  

**Link**: [PDF](https://arxiv.org/pdf/2607.24753)  

**Abstract**: Although directive prompting is the predominant way to interact with Large Language Models (LLMs), many creative practices rely on language that is open-ended, associative, phonaesthetic, symbolic, and that unfolds across multiple temporalities. In this work, we explore how creative practitioners might work with AI systems when language is treated not merely as instruction but as material. We conducted an ecological two-week study with four creative practitioners using a design probe: the Memetic Mixer, a tangible interactive device that constrains interaction with an LLM. Analysis of post-study interviews and device logs identified distinct modes of material language use and temporalities that shaped each participant's engagement with AI and their creative practice. We reflect on these findings and contribute design considerations that support open-ended interaction with AI in creative practice. 

---
# VLD-RAG: Agentic Vision-Language Retrieval-Augmented Generation for Long, Visually-Rich Multi-Page Documents 

**Authors**: Seonok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.24748)  

**Abstract**: Visually-rich documents such as reports, slides, and manuals often distribute the evidence needed to answer a question across multiple pages, mixing text with layout cues, tables, charts, and figures. This work studies multimodal retrieval-augmented generation for question answering over such visually-rich long documents, where retrieval must select evidence pages that include both textual and visual signals. We present VLD-RAG, an agentic multimodal RAG framework for multi-page evidence retrieval and cross-page reasoning over long documents. VLD-RAG builds a page-preserving multimodal index that stores parsed text, page-level metadata, and dense visual representations, and uses a hybrid retrieval strategy that combines keyword-based sparse search with dense semantic queries to identify candidate sources and evidence pages. A verifier-guided agent workflow coordinates a Retrieval Agent, Answer Agent, and Validation Agent to broaden evidence coverage, detect missing citations, and refine retrieval requests when needed. We evaluate retrieval with Top-1 and Top-5 evidence-page accuracy and generation with generalized accuracy, and show that VLD-RAG improves both evidence-page retrieval and end-task question answering on visually-rich long-document benchmarks, including LongDocURL and MMLongBench-Doc, outperforming previous vision-based retrieval baselines. These findings highlight that coordinated agent verification and multimodal hybrid retrieval are crucial for reliable grounding when correct answers depend on evidence scattered across pages. 

---
# DocAnnot -- Accelerating the Creation of Key Information Extraction Datasets with GenAI-Powered Auto-annotation 

**Authors**: Siddartha Reddy, Harikrishnan P M, Goutham Vignesh, Varun V, Vishal Vaddina  

**Link**: [PDF](https://arxiv.org/pdf/2607.24745)  

**Abstract**: Key Information Extraction (KIE) is vital for many document applications, but creating training datasets is traditionally a time-consuming manual process. We introduce DocAnnot, a framework that significantly accelerates KIE dataset generation. DocAnnot leverages a Large Vision Language Model (LVLM) for label value extraction, OCR for text/bounding box detection, and a novel Spatially Informed Contextual Matching (SICM) algorithm. SICM improves label-value association by combining spatial relationships and proximity analysis with textual matching. We evaluate our framework on the CORD and SROIE benchmarks, demonstrating its ability to auto-generate annotations with F1-scores of 0.679 and 0.846, respectively. Furthermore, we investigate the effectiveness of using auto-annotated data for fine-tuning downstream KIE models. While human-annotated data remains superior, models trained exclusively on DocAnnot's outputs attain respectable performance (e.g., LayoutLMv3 achieving an F1-score of 0.6765 on CORD). These results show that while our framework significantly reduces reliance on manual effort, it does not yet fully eliminate the need for human intervention. However, by automating the process to a point where reviewers can efficiently refine outputs, our system enables near-perfect annotations with much greater efficiency than manual annotation from scratch. This approach offers substantial time and cost savings, making it valuable for resource-constrained settings and rapid model prototyping. 

---
# Multi-Objective Structured Pruning of LLMs for Latency and Model Size Optimization 

**Authors**: Muhammad Junaid Ali, Smail Niar, El-Ghazali Talbi  

**Link**: [PDF](https://arxiv.org/pdf/2607.22583)  

**Abstract**: Large Language Models (LLMs) have achieved widespread adoption because of their strong reasoning and query-response capabilities. However, deploying them in embedded and edge computing environments remains challenging because of strict latency, memory, and energy constraints. Their large parameter counts and computational demands hinder efficient execution on resource-constrained platforms. Although model pruning has emerged as a viable solution for reducing scale while preserving performance, jointly optimizing layers, attention heads, and Multi-Layer Perceptron (MLP) dimensions remains highly complex. Exhaustively exploring this combined design space is computationally expensive and often leads to local optima or unstable configurations. To address these limitations, we propose a hardware-aware, multi-objective structured pruning framework. The proposed two-stage method explicitly targets latency and model size for efficient deployment on edge devices. In the coarse-grained stage, multi-objective depth pruning removes entire attention and MLP blocks to reduce computational load and memory usage. In the subsequent fine-grained stage, Parallel Bayesian Optimization (PBO) searches for the optimal layer-wise pruning ratios for pruning under latency constraints, while importance-based strategies rank the specific components to be pruned within each layer's allocated budget. Experimental results show that our approach reduces model complexity with minimal impact on commonsense reasoning tasks and zero-shot performance. Our method achieves a favorable trade-off among accuracy, latency, and model size, making it suitable for edge deployment. Across multiple LLMs at 37.5% and 50% pruning ratios, the proposed approach achieves better performance on commonsense reasoning tasks than existing methods while significantly reducing inference cost. 

---
