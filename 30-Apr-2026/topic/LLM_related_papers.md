# What Kind of Language is Easy to Language-Model Under Curriculum Learning? 

**Authors**: Nadine El-Naggar, Tatsuki Kuribayashi, Ted Briscoe  

**Link**: [PDF](https://arxiv.org/pdf/2604.26844)  

**Abstract**: Many of the thousands of attested languages share common configurations of features, creating a spectrum from typologically very rare (e.g., object-verb-subject word order) or impossible languages to very common combinations of features (e.g., subject-object-verb word order). One central question is under what conditions such typological tendencies can be predicted, and specifically whether the learning bias of language models (LMs) is sufficient to reproduce such patterns. In this study, we add one dimensionality to such analysis -- the learning scenario for LMs -- to explore its interaction with the inductive bias of LMs. Specifically, as a first study, we examine the effect of curriculum learning (CL), as a developmentally motivated learning scenario, i.e., starting with simpler sentences rather than randomly-ordered input. We expand existing LM-based exploration (El-Naggar et al., 2025a,b) with a simple CL variant and find that CL substantially impacts the apparent inductive bias of LMs. 

---
# Turning the TIDE: Cross-Architecture Distillation for Diffusion Large Language Models 

**Authors**: Gongbo Zhang, Wen Wang, Ye Tian, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2604.26951)  

**Abstract**: Diffusion large language models (dLLMs) offer parallel decoding and bidirectional context, but state-of-the-art dLLMs require billions of parameters for competitive performance. While existing distillation methods for dLLMs reduce inference steps within a single architecture, none address cross-architecture knowledge transfer, in which the teacher and student differ in architecture, attention mechanism, and tokenizer. We present TIDE, the first framework for cross-architecture dLLM distillation, comprising three modular components: (1) TIDAL, which jointly modulates distillation strength across training progress and diffusion timestep to account for the teacher's noise-dependent reliability; (2) CompDemo, which enriches the teacher's context via complementary mask splitting to improve predictions under heavy masking; and (3) Reverse CALM, a cross-tokenizer objective that inverts chunk-level likelihood matching, yielding bounded gradients and dual-end noise filtering. Distilling 8B dense and 16B MoE teachers into a 0.6B student via two heterogeneous pipelines outperforms the baseline by an average of 1.53 points across eight benchmarks, yielding notable gains in code generation, where HumanEval scores reach 48.78 compared to 32.3 for the AR baseline. 

---
# Decoupling Knowledge and Task Subspaces for Composable Parametric Retrieval Augmented Generation 

**Authors**: Weihang Su, Hanwen Zhang, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26768)  

**Abstract**: Parametric Retrieval-Augmented Generation (PRAG) encodes external documents into lightweight parameter modules that can be retrieved and merged at inference time, offering a promising alternative to in-context retrieval augmentation. Despite its potential, many PRAG implementations train document adapters with task-supervised objectives, which may cause each adapter to encode both document-specific facts and reusable task-solving behavior. This entanglement may make adapter composition less reliable: when multiple adapters are merged at inference time, their overlapping task behaviors can accumulate together with document-specific updates, potentially making the merged adapter less stable and less focused on the intended document knowledge. To examine this issue, we explore Orthogonal Subspace Decomposition (OSD), an adapter-training setup that separates reusable task behavior from document-specific knowledge adapters. Concretely, we first train a Task LoRA to capture reusable task behavior, and then train document LoRAs to encode document-specific knowledge in a orthogonal subspace. This setup provides a controlled way to examine how orthogonalizing task and document LoRA updates affects adapter composition in multi-document PRAG. Experiments across multiple knowledge-intensive tasks and model scales suggest that this orthogonalization strategy can improve compositional robustness in parametric RAG, especially when multiple document adapters are merged. 

---
# HalluCiteChecker: A Lightweight Toolkit for Hallucinated Citation Detection and Verification in the Era of AI Scientists 

**Authors**: Yusuke Sakai, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2604.26835)  

**Abstract**: We introduce HalluCiteChecker, a toolkit for detecting and verifying hallucinated citations in scientific papers. While AI assistant technologies have transformed the academic writing process, including citation recommendation, they have also led to the emergence of hallucinated citations that do not correspond to any existing work. Such citations not only undermine the credibility of scientific papers but also impose an additional burden on reviewers and authors, who must manually verify their validity during the review process. In this study, we formalize hallucinated citation detection as an NLP task and provide a corresponding toolkit as a practical foundation for addressing this problem. Our package is lightweight and can perform verification in seconds on a standard laptop. It can also be executed entirely offline and runs efficiently using only CPUs. We hope that HalluCiteChecker will help reduce reviewer workload and support organizers by enabling systematic pre-review and publication checks. Our code is released under the Apache 2.0 license on GitHub and is distributed as an installable package via PyPI. A demonstration video is available on YouTube. 

---
# MoRFI: Monotonic Sparse Autoencoder Feature Identification 

**Authors**: Dimitris Dimakopoulos, Shay B. Cohen, Ioannis Konstas  

**Link**: [PDF](https://arxiv.org/pdf/2604.26866)  

**Abstract**: Large language models (LLMs) acquire most of their factual knowledge during the pre-training stage, through next token prediction. Subsequent stages of post-training often introduce new facts outwith the parametric knowledge, giving rise to hallucinations. While it has been demonstrated that supervised fine-tuning (SFT) on new knowledge may exacerbate the problem, the underlying mechanisms are still poorly understood. We conduct a controlled fine-tuning experiment, focusing on closed-book QA, and find latent directions that causally contribute to hallucinations. Specifically, we fine-tune Llama 3.1 8B, Gemma 2 9B and Mistral 7B v03 on seven distinct single QA datasets, controlling for the percentage of new knowledge and number of training epochs. By measuring performance on the test set, we validate that incrementally introducing new knowledge increases hallucinations, with the effect being more pronounced with prolonged training. We leverage pre-trained sparse autoencoders (SAEs) to analyze residual stream activations across various checkpoints for each model and propose Monotonic Relationship Feature Identification (MoRFI) for capturing causally relevant latents. MoRFI filters SAE features that respond monotonically to controlled fine-tuning data mixtures of a target property. Our findings show that exposure to unknown facts disrupts the model's ability to retrieve stored knowledge along a set of directions in the residual stream. Our pipeline reliably discovers them across distinct models, recovering knowledge through single-latent interventions. 

---
# Domain-Adapted Small Language Models for Reliable Clinical Triage 

**Authors**: Manar Aljohani, Brandon Ho, Kenneth McKinley, Dennis Ren, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26766)  

**Abstract**: Accurate and consistent Emergency Severity Index (ESI) assignment remains a persistent challenge in emergency departments, where highly variable free-text triage documentation contributes to mistriage and workflow inefficiencies. This study evaluates whether open-source small language models (SLMs) can serve as reliable, privacy-preserving decision-support tools for clinical triage. We systematically compared multiple SLMs across diverse prompting pipelines and found that clinical vignettes, concise summaries of triage narratives, yielded the most accurate predictions. The SLM, Qwen2.5-7B, demonstrated the strongest balance of accuracy, stability, and computational efficiency. Through large-scale domain adaptation using expert-curated and silver-standard pediatric triage data, fine-tuned Qwen2.5-7B models substantially reduced discordance and clinically significant errors, outperforming all baseline SLMs and advanced proprietary large language models (LLMs, e.g., GPT-4o). These findings highlight the feasibility of institution-specific SLMs for reliable, privacy-preserving ESI decision support and underscore the importance of targeted fine-tuning over more complex inference strategies. 

---
# Zero-Shot to Full-Resource: Cross-lingual Transfer Strategies for Aspect-Based Sentiment Analysis 

**Authors**: Jakob Fehle, Nils Constantin Hellwig, Udo Kruschwitz, Christian Wolff  

**Link**: [PDF](https://arxiv.org/pdf/2604.26619)  

**Abstract**: Aspect-based Sentiment Analysis (ABSA) extracts fine-grained opinions toward specific aspects within text but remains largely English-focused despite major advances in transformer-based and instruction-tuned models. This work presents a multilingual evaluation of state-of-the-art ABSA approaches across seven languages (English, German, French, Dutch, Russian, Spanish, and Czech) and four subtasks (ACD, ACSA, TASD, ASQP). We systematically compare different transformer architectures under zero-resource, data-only, and full-resource settings, using cross-lingual transfer, code-switching and machine translation. Fine-tuned Large Language Models (LLMs) achieve the highest overall scores, particularly in complex generative tasks, while few-shot counterparts approach this performance in simpler setups, where smaller encoder models also remain competitive. Cross-lingual training on multiple non-target languages yields the strongest transfer for fine-tuned LLMs, while smaller encoder or seq-to-seq models benefit most from code-switching, highlighting architecture-specific strategies for multilingual ABSA. We further contribute two new German datasets, an adapted GERestaurant and the first German ASQP dataset (GERest), to encourage multilingual ABSA research beyond English. 

---
# Translating Under Pressure: Domain-Aware LLMs for Crisis Communication 

**Authors**: Antonio Castaldo, Maria Carmen Staiano, Johanna Monti, Sheila Castilho, Francesca Chiusaroli  

**Link**: [PDF](https://arxiv.org/pdf/2604.26597)  

**Abstract**: Timely and reliable multilingual communication is critical during natural and human-induced disasters, but developing effective solutions for crisis communication is limited by the scarcity of curated parallel data. We propose a domain-adaptive pipeline that expands a small reference corpus, by retrieving and filtering data from general corpora. We use the resulting dataset to fine-tune a small language model for crisis-domain translation and then apply preference optimization to bias outputs toward CEFR A2-level English. Automatic and human evaluation shows that this approach improves readability, while maintaining strong adequacy. Our results indicate that simplified English, combined with domain adaptation, can function as a practical lingua franca for emergency communication when full multilingual coverage is not feasible. 

---
# From Black-Box Confidence to Measurable Trust in Clinical AI: A Framework for Evidence, Supervision, and Staged Autonomy 

**Authors**: Serhii Zabolotnii, Viktoriia Holinko, Olha Antonenko  

**Link**: [PDF](https://arxiv.org/pdf/2604.26671)  

**Abstract**: Trust in clinical artificial intelligence (AI) cannot be reduced to model accuracy, fluency of generation, or overall positive user impression. In medicine, trust must be engineered as a measurable system property grounded in evidence, supervision, and operational boundaries of AI autonomy. This article proposes a practical framework for trustworthy clinical AI built around three principles: evidence, supervision, and staged autonomy. Rather than replacing deterministic clinical logic wholesale with end-to-end black-box models, the proposed approach combines a deterministic core, a patient-specific AI assistant for contextual validation, a multi-tier model escalation mechanism, and a human supervision layer for verification, escalation, and risk control. We demonstrate that trust also depends on selective verification of clinically critical findings, bounded clinical context, disciplined prompt architecture, and careful evaluation on realistic cases. Classifier-driven modular prompting is examined as an incremental path to scaling clinical depth without sacrificing prompt performance and without waiting for complete rule-based coverage. To operationalize trust, a set of trust metrics is proposed, built on metrological principles -- measurement uncertainty, calibration, traceability -- enabling quantitative rather than subjective assessment of each architectural layer. In this perspective, trustworthy clinical AI emerges not as a property of an individual model, but as an architectural outcome of a system into which evidence trails, human oversight, tiered escalation, and graduated action rights are embedded from the outset. 

---
# TLPO: Token-Level Policy Optimization for Mitigating Language Confusion in Large Language Models 

**Authors**: Jinho Choo, JunSeung Lee, Jimyeong Kim, Yeeho Song, S. K. Hong, Yeong-Dae Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2604.26553)  

**Abstract**: Large language models (LLMs) demonstrate strong multilingual capabilities, yet often fail to consistently generate responses in the intended language, exhibiting a phenomenon known as language confusion. Prior mitigation approaches based on sequence-level fine-tuning, such as DPO, ORPO, and GRPO, operate at the level of entire responses and can lead to unintended degradation of general model capabilities, motivating the need for more fine-grained alternatives. To address this, we introduce Token-Level Policy Optimization (TLPO), a fine-tuning framework designed to mitigate language confusion through localized, token-level updates. TLPO identifies error-prone positions, explores alternative candidate tokens, and updates the policy using a tailored objective to suppress error-inducing outputs at a granular level. This selective intervention enables effective mitigation of language confusion without compromising the model's general abilities. Experiments on multiple multilingual LLMs across diverse languages demonstrate that TLPO significantly outperforms baselines in improving language consistency while preserving downstream task accuracy. 

---
# ClawGym: A Scalable Framework for Building Effective Claw Agents 

**Authors**: Fei Bai, Huatong Song, Shuang Sun, Daixuan Cheng, Yike Yang, Chuan Hao, Renyuan Li, Feng Chang, Yuan Wei, Ran Tao, Bryan Dai, Jian Yang, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.26904)  

**Abstract**: Claw-style environments support multi-step workflows over local files, tools, and persistent workspace states. However, scalable development around these environments remains constrained by the absence of a systematic framework, especially one for synthesizing verifiable training data and integrating it with agent training and diagnostic evaluation. To address this challenge, we present ClawGym, a scalable framework that supports the full lifecycle of Claw-style personal agent development. Concretely, we construct ClawGym-SynData, a diverse dataset of 13.5K filtered tasks synthesized from persona-driven intents and skill-grounded operations, paired with realistic mock workspaces and hybrid verification mechanisms. We then train a family of capable Claw-style models, termed ClawGym-Agents, through supervised fine-tuning on black-box rollout trajectories, and further explore reinforcement learning via a lightweight pipeline that parallelizes rollouts across per-task this http URL support reliable evaluation, we further construct ClawGym-Bench, a benchmark of 200 instances calibrated through automated filtering and human-LLM review. Relevant resources will be soon released at this https URL. 

---
# Multimodal LLMs are not all you need for Pediatric Speech Language Pathology 

**Authors**: Darren Fürst, Sebastian Steindl, Ulrich Schäfer  

**Link**: [PDF](https://arxiv.org/pdf/2604.26568)  

**Abstract**: Speech Sound Disorders (SSD) affect roughly five percent of children, yet speech-language pathologists face severe staffing shortages and unmanageable caseloads. We test a hierarchical approach to SSD classification on the granular multi-task SLPHelmUltraSuitePlus benchmark. We propose a cascading approach from binary classification to type, and symptom classification. By fine-tuning Speech Representation Models (SRM), and using targeted data augmentation we mitigate biases found by previous works, and improve upon all clinical tasks in the benchmark. We also treat Automatic Speech Recognition (ASR) with our data augmentation approach. Our results demonstrate that SRM consistently outperform the LLM-based state-of-the-art across all evaluated tasks by a large margin. We publish our models and code to foster future research. 

---
# SafeReview: Defending LLM-based Review Systems Against Adversarial Hidden Prompts 

**Authors**: Yuan Xin, Yixuan Weng, Minjun Zhu, Ying Ling, Chengwei Qin, Michael Hahn, Michael Backes, Yue Zhang, Linyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26506)  

**Abstract**: As Large Language Models (LLMs) are increasingly integrated into academic peer review, their vulnerability to adversarial prompts -- adversarial instructions embedded in submissions to manipulate outcomes -- emerges as a critical threat to scholarly integrity. To counter this, we propose a novel adversarial framework where a Generator model, trained to create sophisticated attack prompts, is jointly optimized with a Defender model tasked with their detection. This system is trained using a loss function inspired by Information Retrieval Generative Adversarial Networks, which fosters a dynamic co-evolution between the two models, forcing the Defender to develop robust capabilities against continuously improving attack strategies. The resulting framework demonstrates significantly enhanced resilience to novel and evolving threats compared to static defenses, thereby establishing a critical foundation for securing the integrity of peer review. 

---
# OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory 

**Authors**: Jinze Li, Yang Zhang, Xin Yang, Jiayi Qu, Jinfeng Xu, Shuo Yang, Junhua Ding, Edith Cheuk-Han Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2604.26622)  

**Abstract**: Autonomous LLM agents increasingly operate in long-horizon, interactive settings where success depends on reusing experience accumulated over extended histories. However, existing agent memory systems are fundamentally constrained by text-context budgets: storing or revisiting raw trajectories is prohibitively token-expensive, while summarization and text-only retrieval trade token savings for information loss and fragmented evidence. To address this limitation, we propose Optical Context Retrieval Memory (OCR-Memory), a memory framework that leverages the visual modality as a high-density representation of agent experience, enabling retention of arbitrarily long histories with minimal prompt overhead at retrieval time. Specifically, OCR-Memory renders historical trajectories into images annotated with unique visual identifiers. OCR-Memory retrieves stored experience via a \emph{locate-and-transcribe} paradigm that selects relevant regions through visual anchors and retrieves the corresponding verbatim text, avoiding free-form generation and reducing hallucination. Experiments on long-horizon agent benchmarks show consistent gains under strict context limits, demonstrating that optical encoding increases effective memory capacity while preserving faithful evidence recovery. 

---
# HealthNLP_Retrievers at ArchEHR-QA 2026: Cascaded LLM Pipeline for Grounded Clinical Question Answering 

**Authors**: Md Biplob Hosen, Md Alomgeer Hussein, Md Akmol Masud, Omar Faruque, Tera L Reynolds, Lujie Karen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.26880)  

**Abstract**: Patient portals now give individuals direct access to their electronic health records (EHRs), yet access alone does not ensure patients understand or act on the complex clinical information contained in these records. The ArchEHR-QA 2026 shared task addresses this challenge by focusing on grounded question answering over EHRs, and this paper presents the system developed by the HealthNLP_Retrievers team for this task. The proposed approach uses a multi-stage cascaded pipeline powered by the Gemini 2.5 Pro large language model to interpret patient-authored questions and retrieve relevant evidence from lengthy clinical notes. Our architecture comprises four integrated modules: (1) a few-shot query reformulation unit which summarizes verbose patient queries; (2) a heuristic-based evidence scorer which ranks clinical sentences to prioritize recall; (3) a grounded response generator which synthesizes professional-caliber answers restricted strictly to identified evidence; and (4) a high-precision many-to-many alignment framework which links generated answers to supporting clinical sentences. This cascaded approach achieved competitive results. Across the individual tracks, the system ranked 1st in question interpretation, 5th in answer generation, 7th in evidence identification, and 9th in answer-evidence alignment. These results show that integrating large language models within a structured multi-stage pipeline improves grounding, precision, and the professional quality of patient-oriented health communication. To support reproducibility, our source code is publicly available in our GitHub repository 

---
# Differentially-Private Text Rewriting reshapes Linguistic Style 

**Authors**: Stefan Arnold  

**Link**: [PDF](https://arxiv.org/pdf/2604.26656)  

**Abstract**: Differential Privacy (DP) for text matured from disjointed word-level substitutions to contiguous sentence-level rewriting by leveraging the generative capacity of language models. While this form of text privatization is best suited for balancing formal privacy guarantees with grammatical coherence, its impact on the register identity of text remains largely unexplored. By conducting a multidimensional stylistic profiling of differentially-private rewriting, we demonstrate that the cost of privacy extends far beyond lexical variation. Specifically, we find that rewriting under privacy constraints induces a systematic functional mutation of the text's communicative signature. This shift is characterized by the severe attrition of interactive markers, contextual references, and complex subordination. By comparing autoregressive paraphrasing against bidirectional substitution across a spectrum of privacy budgets, we observe that both architectures force convergence toward a non-involved and non-persuasive register. This register-blind sanitization effectively preserves semantic content but structurally homogenizes the nuanced stylistic markers that define human-authored discourse. 

---
# SAGE: A Strategy-Aware Graph-Enhanced Generation Framework For Online Counseling 

**Authors**: Eliya Naomi Aharon, Meytal Grimland, Avi Segal, Loona Ben Dayan, Inbar Shenfeld, Yossi Levi Belz, Kobi Gal  

**Link**: [PDF](https://arxiv.org/pdf/2604.26630)  

**Abstract**: Effective mental health counseling is a complex, theory-driven process requiring the simultaneous integration of psychological frameworks, real-time distress signals, and strategic intervention planning. This level of clinical reasoning is critical for safety and therapeutic effectiveness but is often missing in general-purpose Large Language Models (LLMs). We introduce SAGE (Strategy-Aware Graph-Enhanced), a novel framework designed to bridge the gap between structured clinical knowledge and generative AI. SAGE constructs a heterogeneous graph that unifies conversational dynamics with a psychologically grounded layer, explicitly anchoring interactions in a theory-driven lexicon. Our architecture first employs a Next Strategy Classifier to identify the optimal therapeutic intervention. Subsequently, a Graph-Aware Attention mechanism projects graph-derived structural signals into soft prompts, conditioning the LLM to generate responses that maintain clinical depth. Validated through both automated metrics and expert human evaluation, SAGE outperforms baselines in strategy prediction and recommended response quality. By providing actionable intervention recommendations, SAGE serves as a cutting-edge decision-support tool designed to augment human expertise in high-stakes crisis counseling. 

---
# When Hidden States Drift: Can KV Caches Rescue Long-Range Speculative Decoding? 

**Authors**: Tianyu Liu, Yuhao Shen, Xinyi Hu, Baolin Zhang, Hengxin Zhang, Jun Dai, Jun Zhang, Shuang Ge, Lei Chen, Yue Li, MingCheng Wan  

**Link**: [PDF](https://arxiv.org/pdf/2604.26412)  

**Abstract**: Speculative decoding accelerates LLM inference, but SOTA hidden-state-based drafters suffer from long-range decay: draft accuracy degrades as the speculative step increases. Existing work attributes this decay to train-inference mismatch and proposes test-time training (TTT) as a remedy, yet we observe that long-range decay persists even in TTT-trained drafters. We revisit long-range decay from the perspective of context information preservation. In hidden-state reuse, we argue the target hidden state acts as a biased context compression: it aggregates historical token information according to the attention query at the current position, yielding a compact representation optimized for immediate next-token prediction. This compression can suppress information less relevant to the current query but important for later speculative steps. In contrast, the target model's KV cache serves as an explicit context, retaining the complete set of token-wise KV representations. We therefore posit the KV-Reuse Hypothesis: allowing the draft model to reuse the target KV cache can provide richer signals for long-horizon drafting. To test this hypothesis, we introduce KVShot, a diagnostic framework that compares three reuse paradigms: hidden-only, KV-only, and hybrid. Extensive evaluations on Qwen3-8B show that KV-Reuse improves long-range acceptance, although end-to-end speedups remain marginal under current training pipelines. Our analysis identifies two key structural bottlenecks: shallow drafters struggle to estimate target queries accurately, and draft-side KV projections receive sparse gradient signals. These findings suggest that realizing the full potential of KV-aware decoding requires moving beyond TTT toward block-wise training paradigms. By exposing these bottlenecks, KVShot provides a foundational diagnostic testbed and a clear roadmap for designing next-generation inference architectures. 

---
# Tree-of-Text: A Tree-based Prompting Framework for Table-to-Text Generation in the Sports Domain 

**Authors**: Shang-Hsuan Chiang, Tsan-Tsung Yang, An-Zi Yen, Wen-Chih Peng  

**Link**: [PDF](https://arxiv.org/pdf/2604.26501)  

**Abstract**: Generating sports game reports from structured tables is a complex table-to-text task that demands both precise data interpretation and fluent narrative generation. Traditional model-based approaches require large, annotated datasets, while prompt-based methods using large language models (LLMs) often struggle with hallucination due to weak table comprehension. To overcome these challenges, we propose Tree-of-Text, a tree-structured prompting framework that guides LLMs through a three-stage generation process: (1) Content Planning, where relevant operations and arguments are selected from the input tables; (2) Operation Execution, which breaks down large tables into manageable sub-tables; and (3) Content Generation, where short textual outputs are merged and rewritten into a cohesive report. Experiments show that our method outperforms existing methods on ShuttleSet+, leads in RG and CO metrics on RotoWire-FG, and excels in CS and CO on MLB with roughly 40% of the time and cost of Chain-of-Table. These results demonstrate the effectiveness and efficiency of Tree-of-Text and suggest a promising direction for prompt-based table-to-text generation in the sports domain. 

---
# EmoTransCap: Dataset and Pipeline for Emotion Transition-Aware Speech Captioning in Discourses 

**Authors**: Shuhao Xu, Yifan Hu, Jingjing Wu, Zhihao Du, Zheng Lian, Rui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26417)  

**Abstract**: Emotion perception and adaptive expression are fundamental capabilities in human-agent interaction. While recent advances in speech emotion captioning (SEC) have improved fine-grained emotional modeling, existing systems remain limited to static, single-emotion characterization within isolated sentences, neglecting dynamic emotional transitions at the discourse level. To address this gap, we propose Emotion Transition-Aware Speech Captioning (EmoTransCap), a paradigm that integrates temporal emotion dynamics with discourse-level speech description. To construct a dataset rich in emotion transitions while enabling scalable expansion, we design an automated pipeline for dataset creation. This is the first large-scale dataset explicitly designed to capture discourse-level emotion transitions. To generate semantically rich descriptions, we incorporate acoustic attributes and temporal cues from discourse-level speech. Our Multi-Task Emotion Transition Recognition (MTETR) model performs joint emotion transition detection and diarization. Leveraging the semantic analysis capabilities of LLMs, we produce two annotation versions: descriptive and instruction-oriented. These data and annotations offer a valuable resource for advancing emotion perception and emotional expressiveness. The dataset enables speech captions that capture emotional transitions, facilitating temporal-dynamic and fine-grained emotion understanding. We also introduce a controllable, transition-aware emotional speech synthesis system at the discourse level, enhancing anthropomorphic emotional expressiveness and supporting emotionally intelligent conversational agents. 

---
# StarDrinks: An English and Korean Test Set for SLU Evaluation in a Drink Ordering Scenario 

**Authors**: Marcely Zanon Boito, Caroline Brun, Inyoung Kim, Denys Proux, Salah Ait-Mokhtar, Nikolaos Lagos, Jean-Luc Meunier, Ioan Calapodescu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26500)  

**Abstract**: LLMs and speech assistants are increasingly used for task-oriented interactions, yet their evaluation often relies on controlled scenarios that fail to capture the variability and complexity of real user requests. Drink ordering, for example, involves diverse named entities, drink types, sizes, customizations, and brand-specific terminology, as well as spontaneous speech phenomena such as hesitations and self-corrections. To address this gap, we introduce StarDrinks, a test set in English and Korean containing speech utterances features, transcriptions, and annotated slots. Our dataset supports speech-to-slots SLU, transcription-to-slots NLU, and speech-to-transcription ASR evaluation, providing a realistic benchmark for model robustness and generalization in a linguistically rich, real-world task. 

---
# SG-UniBuc-NLP at SemEval-2026 Task 6: Multi-Head RoBERTa with Chunking for Long-Context Evasion Detection 

**Authors**: Gabriel Stefan, Sergiu Nisioi  

**Link**: [PDF](https://arxiv.org/pdf/2604.26375)  

**Abstract**: We describe our system for SemEval-2026 Task 6 (CLARITY: Unmasking Political Question Evasions), which classifies English political interview responses by coarse-grained clarity (3-way) and fine-grained evasion strategy (9-way). Since responses frequently exceed the 512-token limit of standard Transformer encoders, we apply an overlapping sliding-window chunking strategy with element-wise Max-Pooling aggregation over chunk representations. A shared RoBERTa-large encoder supplies two task-specific heads trained jointly via a multi-task objective, with inference-time ensembling over 7-fold stratified cross-validation. Our system achieves a Macro-F1 of 0.80 on Subtask 1 and 0.51 on Subtask 2, ranking 11th in both subtasks. 

---
# Select to Think: Unlocking SLM Potential with Local Sufficiency 

**Authors**: Wenxuan Ye, Yangyang Zhang, Xueli An, Georg Carle, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.26940)  

**Abstract**: Small language models (SLMs) offer computational efficiency for scalable deployment, yet they often fall short of the reasoning power exhibited by their larger counterparts (LLMs). To mitigate this gap, current approaches invoke an LLM to generate tokens at points of reasoning divergence, but these external calls introduce substantial latency and costs. Alternatively, standard distillation is often hindered by the capacity limitation, as SLMs struggle to accurately mimic the LLM's complex generative distribution. We address this dilemma by identifying local sufficiency: at divergence points, the LLM's preferred token consistently resides within the SLM's top-K next-token predictions, even when failing to emerge as the SLM top-1 choice. We therefore propose SELECT TO THINK (S2T), which reframes the LLM's role from open-ended generation to selection among the SLM's proposals, simplifying the supervision signal to discrete candidate rankings. Leveraging this, we introduce S2T-LOCAL, which distills the selection logic into the SLM, empowering it to perform autonomous re-ranking without inference-time LLM dependency. Empirically, we demonstrate that a 1.5B SLM's top-8 candidates capture the 32B LLM's choice with 95% hit rate. Translating this potential into performance, S2T-LOCAL improves greedy decoding by 24.1% on average across benchmarks, effectively matching the efficacy of 8-path self-consistency while operating with single-trajectory efficiency. 

---
# Benchmarking Complex Multimodal Document Processing Pipelines: A Unified Evaluation Framework for Enterprise AI 

**Authors**: Saurabh K. Singh, Sachin Raj  

**Link**: [PDF](https://arxiv.org/pdf/2604.26382)  

**Abstract**: Most enterprise document AI today is a pipeline. Parse, index, retrieve, generate. Each of those stages has been studied to death on its own -- what's still hard is evaluating the system as a whole.
We built EnterpriseDocBench to take a swing at it: parsing fidelity, indexing efficiency, retrieval relevance, and generation groundedness, all on the same corpus. The corpus is built from public, permissively licensed documents across six enterprise domains (five represented in the current pilot). We ran three pipelines through it -- BM25, dense embedding, and a hybrid -- all with the same GPT-5 generator.
The headline numbers: hybrid retrieval narrowly beats BM25 (nDCG@5 of 0.92 vs. 0.91), and both beat dense embedding (0.83). Hallucination doesn't grow monotonically with document length -- short documents and very long ones both hallucinate more than medium ones (28.1% and 23.8% vs. 9.2%). Cross-stage correlations are very weak: parsing->retrieval r=0.14, parsing->generation r=0.17, retrieval->generation 0.02. If quality were cascading the way most of us assume, those numbers would be much higher; they aren't. Design caveats are real (parsing fixed, generator shared, automated proxy metrics) and we don't oversell the result.
One result that genuinely surprised us: factual accuracy on stated claims is 85.5%, but answer completeness averages 0.40. The system is right when it answers -- it just leaves things out. That gap matters more for real deployments than the headline accuracy number does.
We also describe three reference architectures (ColPali, ColQwen2, agentic complexity-based routing) which are not yet integrated end-to-end. Framework, metrics, baselines, and collection scripts will be released open-source on acceptance. 

---
# A Dual-Task Paradigm to Investigate Sentence Comprehension Strategies in Language Models 

**Authors**: Rei Emura, Saku Sugawara  

**Link**: [PDF](https://arxiv.org/pdf/2604.26351)  

**Abstract**: Language models (LMs) behave more like humans when their cognitive resources are restricted, particularly in predicting sentence processing costs such as reading times. However, it remains unclear whether such constraints similarly affect sentence comprehension strategies. Besides, existing methods do not directly target the balance between memory storage and sentence processing, which is central to human working memory. To address this issue, we propose a dual-task paradigm that combines an arithmetic computation task with a sentence comprehension task, such as "The 2 cocktail + blended 3 =..." Our experiments show that under dual-task conditions, GPT-4o, o3-mini, and o4-mini shift toward plausibility-based comprehension, mirroring humans' rational inference. Specifically, these models show a greater accuracy gap between plausible sentences (e.g., "The cocktail was blended by the bartender") and implausible sentences (e.g., "The bartender was blended by the cocktail") in the dual-task condition compared to the single-task conditions. These findings suggest that constraints on the balance between memory and processing resources promote rational inference in LMs. More broadly, they support the view that human-like sentence comprehension fundamentally arises from the allocation of limited cognitive resources. 

---
# A Systematic Comparison of Prompting and Multi-Agent Methods for LLM-based Stance Detection 

**Authors**: Genan Dai, Zini Chen, Yi Yang, Bowen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26319)  

**Abstract**: Stance detection identifies the attitude of a text author toward a given target. Recent studies have explored various LLM-based strategies for this task, from zero-shot prompting to multi-agent debate. However, existing works differ in data splits, base models, and evaluation protocols, making fair comparison difficult. We conduct a systematic comparison that evaluates five methods across two categories -- prompt-based inference (Direct Prompting, Auto-CoT, StSQA) and agent-based debate (COLA, MPRF) -- on four datasets with 14 subtasks, using 15 LLMs from six model families with parameter sizes from 7B to 72B+. Our experiments yield several findings. First, on all models with complete results, the best prompt-based method outperforms the best agent-based method, while agent methods require 7 to 12 times more API calls per sample. Second, model scale has a larger impact on performance than method choice, with gains plateauing around 32B. Third, reasoning-enhanced models (DeepSeek-R1) do not consistently outperform general models of the same size on this task. 

---
# DSIPA: Detecting LLM-Generated Texts via Sentiment-Invariant Patterns Divergence Analysis 

**Authors**: Siyuan Li, Aodu Wulianghai, Guangyan Li, Xi Lin, Qinghua Mao, Yuliang Chen, Jun Wu, Jianhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.26328)  

**Abstract**: The rapid advancement of large language models (LLMs) presents new security challenges, particularly in detecting machine-generated text used for misinformation, impersonation, and content forgery. Most existing detection approaches struggle with robustness against adversarial perturbation, paraphrasing attacks, and domain shifts, often requiring restrictive access to model parameters or large labeled datasets. To address this, we propose DSIPA, a novel training-free framework that detects LLM-generated content by quantifying sentiment distributional stability under controlled stylistic variation. It is based on the observation that LLMs typically exhibit more emotionally consistent outputs, while human-written texts display greater affective variation. Our framework operates in a zero-shot, black-box manner, leveraging two unsupervised metrics, sentiment distribution consistency and sentiment distribution preservation, to capture these intrinsic behavioral asymmetries without the need for parameter updates or probability access. Extensive experiments are conducted on state-of-the-art proprietary and open-source models, including GPT-5.2, Gemini-1.5-pro, Claude-3, and LLaMa-3.3. Evaluations on five domains, such as news articles, programming code, student essays, academic papers, and community comments, demonstrate that DSIPA improves F1 detection scores by up to 49.89% over baseline methods. The framework exhibits superior generalizability across domains and strong resilience to adversarial conditions, providing a robust and interpretable behavioral signal for secure content identification in the evolving LLM landscape. 

---
# Shorthand for Thought: Compressing LLM Reasoning via Entropy-Guided Supertokens 

**Authors**: Zhenyu Zhao, Sander Land, Dan Bikel, Waseem Alshikh  

**Link**: [PDF](https://arxiv.org/pdf/2604.26355)  

**Abstract**: Reasoning in Large Language Models incurs significant inference-time compute, yet the token-level information structure of reasoning traces remains underexplored. We observe that reasoning tokens split into two functional types: low-entropy \textit{structural} tokens (recurring phrases that scaffold the reasoning process) and higher-entropy \textit{organic} tokens (problem-specific content that drives toward a solution). This asymmetry motivates a simple, model-agnostic compression pipeline: apply cross-word BPE merges on a model's own reasoning traces to derive \textit{supertokens} that capture frequent structural patterns, then teach the model to adopt them via supervised fine-tuning. Across three model families and five mathematical reasoning benchmarks, our approach shortens reasoning traces by 8.1\% on average with no statistically significant accuracy loss on any model--benchmark pair. Beyond compression, supertokens act as interpretable reasoning-move annotations (backtracking, verification, strategy shifts), exposing the model's high-level strategy at a glance. Analyzing transitions between structural categories reveals systematic differences between correct and incorrect traces: correct traces show productive recovery (backtracking followed by strategy shifts and verification), while incorrect traces are dominated by confusion cycles (repeated hedging and unresolved contradictions). These diagnostic signals suggest applications in reward shaping and early stopping for RL-based reasoning training. 

---
# Text Style Transfer with Machine Translation for Graphic Designs 

**Authors**: Deergh Singh Budhauria, Sanyam Jain, Rishav Agarwal, Tracy King  

**Link**: [PDF](https://arxiv.org/pdf/2604.26361)  

**Abstract**: Globalization of graphic designs such as those used in marketing materials and magazines is increasingly important for communication to broad audiences. To accomplish this, the textual content in the graphic designs needs to be accurately translated and have the text styling preserved in order to fit visually into the design. Preserving text styling requires high accuracy word alignment between the original and the translated text. The problem of word alignment between source and translated text is long known. The industry standards for extracting word alignments are defined by Giza++ and attention probabilities from neural machine translation (NMT) models. In this paper, we explore three new methods to tackle the word alignment problem for transferring text styles from the source to the translated text. The proposed methods are developed on top of commercially available NMT and LLM translation technologies. They include: NMT with custom input and output tags for text styling; LLM with custom input and output tags; a hybrid with NMT for translation followed by an LLM with use of unigram mappings. To analyze the performance of these solutions, their alignment results are compared with the results of an attention head approach to gauge their usability in graphic design applications. Interestingly, the attention head strong baseline proves more accurate than the LLM or NMT approach and on par with the hybrid NMT+LLM approach. 

---
# FlowBot: Inducing LLM Workflows with Bilevel Optimization and Textual Gradients 

**Authors**: Hongyeon Yu, Young-Bum Kim, Yoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.26258)  

**Abstract**: LLM workflows, which coordinate structured calls to individual LLMs (each augmented with varying instructions and tools) to achieve a particular goal, offer a promising path towards extending the capabilities of LLMs and building powerful systems that can tackle diverse tasks. However, existing approaches for building such workflows generally rely on human-crafted pipelines and prompts, which presents a substantial bottleneck in real world deployment. How can automatically induce and optimize such workflows in a data-driven way? This paper describes a simple data-driven approach for automatically inducing LLM workflows. We formulate workflow induction as a bilevel optimization problem: an outer loop which optimizes a high-level sketch of the workflow (in particular how the LLM calls should be structured), and an inner loop which optimizes each individual LLM call one-by one. Both loops are optimized with ``textual gradients'' where for the inner loop we optimize each component in a modular way through ``backpropagating'' textual gradients layer-by-layer. We find that LLM workflows discovered through our \textsc{FlowBot} (work\textbf{flow} induction through \textbf{b}ilevel \textbf{o}ptimization and \textbf{t}extual gradients) approach performs competitively against strong baselines that make use of human-crafted or automatically-generated workflows. 

---
# Theory-Grounded Evaluation Exposes the Authorship Gap in LLM Personalization 

**Authors**: Yash Ganpat Sawant  

**Link**: [PDF](https://arxiv.org/pdf/2604.26460)  

**Abstract**: Stylistic personalization - making LLMs write in a specific individual's style, rather than merely adapting to task preferences - lacks evaluation grounded in authorship science. We show that grounding evaluation in authorship verification theory transforms what benchmarks can measure. Drawing on three measurement traditions - LUAR, a trained authorship verification model; an LLM-as-judge with decoupled trait matching; and classical function-word stylometrics - we evaluate four inference-time personalization methods across 50 authors and 1,000 generations. The theory-grounded metric, LUAR, provides what ad hoc alternatives cannot: calibrated baselines, with a human ceiling of 0.756 and a cross-author floor of 0.626, that give scores absolute meaning. All methods score below this floor, from 0.484 to 0.508, exposing an authorship gap invisible to uncalibrated metrics. The three metrics produce near-zero pairwise correlations, with absolute r less than 0.07, confirming that without theoretical grounding, metric choice determines conclusions: an LLM judge declares a clear winner while LUAR finds no meaningful differentiation. These findings demonstrate the theory-benchmark cycle in action: authorship theory exposes evaluation failures that ad hoc benchmarks miss. 

---
# StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall 

**Authors**: Yerong Wu, Tianxing Wu, Minghao Zhu, Hangyu Sha, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26243)  

**Abstract**: Achieving realistic human-like conversation for virtual characters requires not only a simple memorization and recall of past events, but also the strategic utilization of memory to meet factual needs and social engagement. Current memory utilization relevant (e.g., memory-augmented generation, long-term dialogue, and etc.) benchmarks overlook this nuance, treating memory primarily as a static repository of facts rather than a dynamic resource to be strategically deployed in dialogues. To address this gap, we design StratMem-Bench, a new benchmark to evaluate strategic memory use in character-centric dialogues. This dataset comprises 657 instances where virtual characters must navigate heterogeneous memory pools containing required, supportive, and irrelevant memories. We also propose a framework with different evaluation metrics including Strict Memory Compliance, Memory Integration Quality, Proactive Enrichment Score and Conditional Irrelevance Rate, to evaluate strategic memory use capabilities of virtual characters. Experiments on StratMem-Bench which leverage the state-of-the-art large language models as virtual characters show that all models perform well at distinguishing between required and irrelevant memories, but struggle once supportive memories are introduced into the decision process. 

---
# Breaking the Autoregressive Chain: Hyper-Parallel Decoding for Efficient LLM-Based Attribute Value Extraction 

**Authors**: Theodore Glavas, Nikhita Vedula, Dushyanta Dhyani, Yilun Zhu, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2604.26209)  

**Abstract**: Some text generation tasks, such as Attribute Value Extraction (AVE), require decoding multiple independent sequences from the same document context. While standard autoregressive decoding is slow due to its sequential nature, the independence between output sequences offers an opportunity for parallelism. We present Hyper-Parallel Decoding, a novel decoding algorithm that accelerates offline decoding by leveraging both shared memory and computation across batches. HPD enables out-of-order token generation through position ID manipulation, significantly improving efficiency. Experiments on AVE show that attribute-value pairs are conditionally independent, enabling us to parallelize value generation within each prompt. By further stacking multiple documents within a single prompt, we can decode in parallel up to 96 tokens per prompt. HPD works with all LLMs, and reduces both inference costs and total inference time by up to 13.8X without compromising output quality, potentially saving hundreds of thousands of dollars on industry AVE tasks. Although designed for attribute extraction, HPD makes no assumptions unique to the AVE domain and can in theory be applied to other scenarios with independent output structures. 

---
# Naamah: A Large Scale Synthetic Sanskrit NER Corpus via DBpedia Seeding and LLM Generation 

**Authors**: Akhil Rajeev P, Annarao Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2604.26456)  

**Abstract**: The digitisation of classical Sanskrit literature is impeded by a scarcity of annotated resources, particularly for Named Entity Recognition. While recent methodologies utilise generic Large Language Models (LLMs) for data augmentation, these approaches remain prone to error and often lack the reasoning depth required for classical grammar. In this work, we introduce Naamah, a high quality silver standard Sanskrit NER dataset comprising 102,942 sentences. We propose a methodology that combines entity extraction from DBpedia with the generative capabilities of a 24B parameter hybrid reasoning model to create grammatically natural and synthetically diverse training data. We utilize this dataset to benchmark two transformer architectures: the massive multilingual XLM RoBERTa and the parameter efficient IndicBERTv2. 

---
# EvoSelect: Data-Efficient LLM Evolution for Targeted Task Adaptation 

**Authors**: Ting-Wei Li, Sirui Chen, Jiaru Zou, Yingbing Huang, Tianxin Wei, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2604.26170)  

**Abstract**: Adapting large language models (LLMs) to a targeted task efficiently and effectively remains a fundamental challenge. Such adaptation often requires iteratively improving the model toward a targeted task, yet collecting high-quality human-labeled data to support this process is costly and difficult to scale. As a result, synthetic data generation has emerged as a flexible and scalable alternative. One straightforward approach is through an iterative generation-training loop, where candidate data are synthesized through an external generator, the model is updated using these data and the process is repeated over iterations. However, generated samples can be noisy, highly redundant, or even misaligned with the targeted task distribution. Training indiscriminately on such data can dilute useful learning signals and even degrade model performance. To address this, we introduce a refined paradigm, namely an iterative generation-selection-training loop, which incorporates a selection step prior to model updates. Building on this paradigm, we propose EvoSelect, a data-efficient framework to evolve LLM effectively. Given candidate samples produced by the data generator, EvoSelect selects training data by jointly modeling targeted task alignment and diversity. We estimate task relevance through optimal transport with proxy gradient representations, which quantifies how well candidate samples align with the targeted task distribution. To mitigate redundancy, we incorporate a diversification mechanism that promotes coverage of complementary training samples. By interleaving alignment and diversification, EvoSelect enables progressive LLM evolution toward targeted tasks. Extensive experiments on various benchmarks demonstrate that with either weak or strong data generators, EvoSelect consistently improves adaptation efficacy over existing data selection methods. 

---
# Option-Order Randomisation Reveals a Distributional Position Attractor in Prompted Sandbagging 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.26206)  

**Abstract**: A predecessor pilot (Cacioli, 2026) found that Llama-3-8B implements prompted sandbagging as positional collapse rather than answer avoidance. However, fixed option ordering in MMLU-Pro left open whether this reflected a model-level position-dominant policy or dataset-level distractor structure. This pre-registered follow-up (3 models, 2,000 MMLU-Pro items, 4 conditions, 24,000 primary trials) added cyclic option-order randomisation as the critical control. The pre-registered item-level same-letter diagnostic did not confirm deterministic position-tracking (same-letter rate 37.3%, below the 50% threshold). However, pre-specified supporting analyses revealed that the response-position distribution under sandbagging was highly stable under complete content rotation (Pearson r = 0.9994; Jensen-Shannon divergence = 0.027, compared to 0.386 between honest and sandbagging conditions). Accuracy spiked to 72.1% when the correct answer coincidentally occupied the preferred position E, and fell to 4.3% at position A. The data provide strong evidence for a soft distributional attractor: under sandbagging instruction, the model enters a low-entropy response-position basin centred on E/F/G that is highly stable and largely content-invariant at the aggregate level. Qwen-2.5-7B served as a negative control (non-compliant, no distributional shift). These results provide evidence, at the 7-9 billion parameter scale, that response-position entropy is a promising black-box behavioural signature of this sandbagging mode. 

---
# Folding Tensor and Sequence Parallelism for Memory-Efficient Transformer Training & Inference 

**Authors**: Vasu Shyam, Anna Golubeva, Quentin Anthony  

**Link**: [PDF](https://arxiv.org/pdf/2604.26294)  

**Abstract**: We present tensor and sequence parallelism (TSP), a parallel execution strategy that folds tensor parallelism and sequence parallelism onto a single device axis. In conventional multi-dimensional parallelism layouts, tensor parallelism (TP) shards model weights while sequence parallelism (SP) shards tokens, reducing per-device parameter or activation memory, respectively. Traditionally, each scheme is assigned its own mesh dimension. TSP instead assigns each rank both a weight shard and a sequence shard, reducing both parameter and activation memory along the same device axis. We implement this design with two runtime schedules. For attention, ranks iterate over broadcast parameter shards and reconstruct context through a sequence-wise key/value exchange. For gated MLPs, weight shards circulate in a ring while partial outputs accumulate locally. By sharding both weights and activations across the same devices, TSP trades additional communication volume for reduced memory overhead. We provide a theoretical communication and memory analysis, describe our implementation of TSP attention and gated MLP blocks, and benchmark TSP against TP, SP, and TP+SP. These results position TSP as a hardware-aware alternative for long-context and memory-constrained model training, and as a viable axis of parallelism in concert with existing parallelism schemes such as pipeline and expert parallelism for dense and mixture-of-expert models. 

---
# From Prompt Risk to Response Risk: Paired Analysis of Safety Behavior of Large Language Model 

**Authors**: Mengya Hu, Qiong Wei, Sandeep Atluri  

**Link**: [PDF](https://arxiv.org/pdf/2604.26052)  

**Abstract**: Safety evaluations of large language models (LLMs) typically report binary outcomes such as attack success rate, refusal rate, or harmful/not-harmful response classification. While useful, these can hide how risk changes between a user's input and the model's response. We present a paired, transition-based analysis over 1250 prompt-response records with human-provided labels over four harm categories (Hate, Sexual, Violence, Self-harm) and ordinal severity levels aligned with the Azure AI Content Safety taxonomy. 61% of responses de-escalate harm relative to the prompt, 36% preserve the same severity, and 3% escalate to higher harm. A per-category persistence/drift-up decomposition identifies Sexual content as 3x harder to de-escalate than Hate or Violence, driven by persistence on already-sexual prompts, not by newly introducing sexual harm from benign inputs. Jointly measuring response relevance reveals an empirical signature of the helpfulness-harmlessness tradeoff: all compliance-escalation cases (from non-zero prompts) are relevance-3 (high-quality, on-task content at elevated severity), while medium-severity responses show the lowest relevance (64%), driven by tangential elaborations in Violence and Sexual categories. 

---
# Calibrated Surprise: An Information-Theoretic Account of Creative Quality 

**Authors**: Bo Zou, Chao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26269)  

**Abstract**: The essence of good creative writing is calibrated surprise: when constraints from all relevant dimensions act together, the feasible solution space collapses into a narrow region, and the surviving choices look least predictable from an unconstrained view. "Calibrated" has a precise meaning: the author's intent, the reader's reasonable expectation, and the logic of reality converge. When these three independent judgements agree on every dimension, the set of admissible writing choices is forced into a very small region. A mathematical corollary follows: full-dimensional accuracy and mediocrity are mutually exclusive -- two sides of one constraint structure, not separate goals.
We use Shannon's mutual information $I(X;Y) = H(X) - H(X|Y)$ as our analysis tool. "Calibrated" corresponds to conditional entropy going to zero; "surprise" to entropy going up; mutual information is the precise measure of the joint quantity. The argument rests on two pillars. Static: when constraints from ethos, mythos, lexis, and dianoia are imposed together, the admissible set collapses sharply, and surviving solutions show up as low-probability choices from an unconstrained view. Dynamic: the chain rule shows each writing choice is constrained by what came before and constrains what comes after; macro-level decisions naturally contribute a larger share of information, removing the need for hand-tuned weighting.
Through case studies and lightweight LLM-logprob computations, we show the framework is both analytically useful and operational, laying the theoretical groundwork for Creative Quality Alignment (CQA) and a professional evaluation benchmark. 

---
# A New Semisupervised Technique for Polarity Analysis using Masked Language Models 

**Authors**: Kohei Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2604.26230)  

**Abstract**: I developed a new version of Latent Semantic Scaling (LSS) employing word2vec as a masked language model. Unlike original spatial models, it assigns polarity scores to words and documents as predicted probabilities of seed words to occur in given contexts. These probabilistic polarity scores are more accurate, interpretable and consistent than those spatial polarity models can produce in text analysis. I demonstrate these advantages by applying both probabilistic and spatial models to China Daily's coverage of China and other countries during the coronavirus disease (COVID) pandemic in terms of achievement in health issues. The result suggests that more advanced masked language models would further improve the semisupervised machine learning technique. 

---
# Anchored Confabulation: Partial Evidence Non-Monotonically Amplifies Confident Hallucination in LLMs 

**Authors**: Ashish Balkishan Lathkar  

**Link**: [PDF](https://arxiv.org/pdf/2604.25931)  

**Abstract**: We identify a previously unknown calibration property of large language models: providing one confirmed intermediate fact toward a multi-step reasoning chain increases the model's confident-wrong-answer rate before full evidence eliminates it. We call this anchored confabulation: a partial anchor commits the model to confident parametric completion of remaining reasoning steps. We formalize it as Parametric Hallucination Confidence (PHC) and establish it across six lines of evidence including a causal injection experiment (PHC 0.613 to 0.656 to 0.595 to 0.536, N=160) and capability scaling across five model families (Spearman rho=0.900, p=0.037). The Anchoring Threshold Law k*(n)=floor(n/3) predicts PHC amplification by hop depth with four confirmed predictions. Applied to RAG routing, a LearnedRouter exploiting PHC closes 81.1% of the oracle performance gap (macro F1=0.426, p<1e-6) on 1,800 queries across four benchmarks with no model fine-tuning and 50x fewer labels than prior RL-based work. An epistemic humility prompt reduces the PHC spike by -0.118; explicit self-rating (PHC=0.684, p<0.001) outperforms lexical confidence as a routing signal. 

---
# Test-Time Safety Alignment 

**Authors**: Baturay Saglam, Dionysis Kalogerias  

**Link**: [PDF](https://arxiv.org/pdf/2604.26167)  

**Abstract**: Recent work has shown that a model's input word embeddings can serve as effective control variables for steering its behavior toward outputs that satisfy desired properties. However, this has only been demonstrated for pretrained text-completion models on the relatively simple objective of reducing surface-level profanity in short continuations. A natural and practically important question is how well input embeddings can control aligned models, which produce an imbalanced bimodal refuse-or-comply output distribution rather than the smooth distribution characteristic of open-ended generation. We explore this in the context of safety, showing that input word embeddings can be optimized in a sub-lexical manner to minimize the semantic harmfulness of aligned model responses. Our approach uses zeroth-order gradient estimation of a black-box text-moderation API with respect to the input embeddings, and then applies gradient descent on these embeddings to minimize the harmfulness of the generated text. Experiments show that the proposed method can neutralize every safety-flagged response on standard safety benchmarks. 

---
# CogRAG+: Cognitive-Level Guided Diagnosis and Remediation of Memory and Reasoning Deficiencies in Professional Exam QA 

**Authors**: Xudong Wang, Zilong Wang, Zhaoyan Ming  

**Link**: [PDF](https://arxiv.org/pdf/2604.25928)  

**Abstract**: Professional domain knowledge underpins human civilization, serving as both the basis for industry entry and the core of complex decision-making and problem-solving. However, existing large language models often suffer from opaque inference processes in which retrieval and reasoning are tightly entangled, causing knowledge gaps and reasoning inconsistencies in professional tasks. To address this, we propose CogRAG+, a training-free framework that decouples and aligns the retrieval-augmented generation pipeline with human cognitive hierarchies. First, we introduce Reinforced Retrieval, a judge-driven dual-path strategy with fact-centric and option-centric paths that strengthens retrieval and mitigates cascading failures caused by missing foundational knowledge. We then develop cognition-stratified Constrained Reasoning, which replaces unconstrained chain-of-thought generation with structured templates to reduce logical inconsistency and generative redundancy. Experiments on two representative models, Qwen3-8B and Llama3.1-8B, show that CogRAG+ consistently outperforms general-purpose models and standard RAG methods on the Registered Dietitian qualification exam. In single-question mode, it raises overall accuracy to 85.8\% for Qwen3-8B and 60.3\% for Llama3.1-8B, with clear gains over vanilla baselines. Constrained Reasoning also reduces the unanswered rate from 7.6\% to 1.4\%. CogRAG+ offers a robust, model-agnostic path toward training-free expert-level performance in specialized domains. 

---
# Information Extraction from Electricity Invoices with General-Purpose Large Language Models 

**Authors**: Javier Gómez, Javier Sánchez  

**Link**: [PDF](https://arxiv.org/pdf/2604.25927)  

**Abstract**: Information extraction from semi-structured business documents remains a critical challenge for enterprise management. This study evaluates the capability of general-purpose Large Language Models to extract structured information from Spanish electricity invoices without task-specific fine-tuning. Using a subset of the IDSEM dataset, we benchmark two architecturally distinct models, Gemini 1.5 Pro and Mistral-small, across 19 parameter configurations and 6 prompting strategies. Our experimental framework treats prompt engineering as the primary experimental variable, comparing zero-shot baselines against increasingly sophisticated few-shot approaches and iterative extraction strategies. Results demonstrate that prompt quality dominates over hyperparameter tuning: the F1-score variation across all parameter configurations is marginal, while the gap between zero-shot and the best few-shot strategy exceeds 19 percentage points. The best configuration (few-shot with cross-validation) achieves an F1-score of 97.61% for Gemini and 96.11% for Mistral-small, with document template structure emerging as the primary determinant of extraction difficulty. These findings establish that prompt design is the critical lever for maximizing extraction fidelity in LLM-based document processing, thereby providing an empirical framework for integrating general-purpose LLMs into business document automation. 

---
# Associative-State Universal Transformers: Sparse Retrieval Meets Structured Recurrence 

**Authors**: Liu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.25930)  

**Abstract**: We study whether a structured recurrent state can serve as a compact associative backbone for language modeling while still supporting exact retrieval. We introduce UniMatrix, a Universal Transformer style family that reuses a shared recurrent block across depth and augments it with hybrid state updates, a ROSA-style residual path, and token-conditioned embedding modulation. We evaluate these models on byte-level WikiText-2, synthetic associative recall, throughput profiling on Apple MPS, and a corrected benchmark for triple-token interactions.
At small scale, UniMatrix-Core and UniMatrix-ROSA slightly outperform a parameter-matched Transformer on WikiText-2 while using many fewer parameters, reaching 5.084 and 5.083 bits-per-byte versus 5.124. The main negative result is equally important: on associative recall, the original UniMatrix family remains near chance while the Transformer reaches 25.4 percent, showing that compressed recurrent state alone is not enough for exact lookup. A retrieval-oriented follow-up, UniMatrix-Assoc, helps only marginally. By contrast, UniMatrix-SparsePointer, which adds sparse slot routing and direct pointer-logit fusion, reaches 75.6 percent on the original pilot recipe and 99.2 percent on a no-dropout follow-up while using 53.8 percent fewer parameters than the Transformer baseline. Ablations show that the gain comes from sufficient slot capacity and exact pointer-level output routing. Overall, structured recurrent state is promising and parameter-efficient, but strong long-range behavior still requires explicit sparse retrieval and better kernels. 

---
# Generative AI-Based Virtual Assistant using Retrieval-Augmented Generation: An evaluation study for bachelor projects 

**Authors**: Dumitru Verşebeniuc, Martijn Elands, Sara Falahatkar, Chiara Magrone, Mohammad Falah, Martijn Boussé, Aki Härmä  

**Link**: [PDF](https://arxiv.org/pdf/2604.25924)  

**Abstract**: Large Language Models have been increasingly employed in the creation of Virtual Assistants due to their ability to generate human-like text and handle complex inquiries. While these models hold great promise, challenges such as hallucinations, missing information, and the difficulty of providing accurate and context-specific responses persist, particularly when applied to highly specialized content domains. In this paper, we focus on addressing these challenges by developing a virtual assistant designed to support students at Maastricht University in navigating project-specific regulations. We propose a virtual assistant based on a Retrieval-Augmented Generation system that enhances the accuracy and reliability of responses by integrating up-to-date, domain-specific knowledge. Through a robust evaluation framework and real-life testing, we demonstrate that our virtual assistant can effectively meet the needs of students while addressing the inherent challenges of applying Large Language Models to a specialized educational context. This work contributes to the ongoing discourse on improving LLM-based systems for specific applications and highlights areas for further research. 

---
# SpecTr-GBV: Multi-Draft Block Verification Accelerating Speculative Decoding 

**Authors**: Yijun Lin, Jinhao Sheng, Qingyue Cai, Feng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.25925)  

**Abstract**: Autoregressive language models suffer from high inference latency due to their sequential decoding nature. Speculative decoding (SD) mitigates this by employing a lightweight draft model to propose candidate tokens, which are selectively verified by a larger target model. While existing methods either adopt multi-draft strategies to increase acceptance rates or block verification techniques to jointly verify multiple tokens, they remain limited by treating these improvements in isolation. In this work, we propose SpecTr-GBV, a novel SD method that unifies multi-draft and greedy block verification (GBV) into a single framework. By formulating the verification step as an optimal transport problem over draft and target token blocks, SpecTr-GBV improves both theoretical efficiency and empirical performance. We theoretically prove that SpecTr-GBV achieves the optimal expected acceptance length physically attainable within the framework of i.i.d. draft generation, and this bound improves as the number of drafts increases. Empirically, we evaluate SpecTr-GBV across five datasets and four baselines. Our method achieves superior speedup and significantly higher block efficiency while preserving output quality. In addition, we perform comprehensive ablation studies to evaluate the impact of various hyperparameters in the model. 

---
# HIVE: Hidden-Evidence Verification for Hallucination Detection in Diffusion Large Language Models 

**Authors**: Guoshenghui Zhao, Weijie Zhao, Tan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26139)  

**Abstract**: Diffusion large language models generate text through multi-step denoising, where hallucination signals may emerge throughout the trajectory rather than only in the final output. Existing detectors mainly rely on output uncertainty or coarse trace statistics, which often fail to capture the richer hidden dynamics of D-LLMs. We propose HIVE, a hidden-evidence verification framework that extracts compressed hidden evidence from denoising trajectories, selects informative step-layer evidence, and conditions a verifier language model on the selected evidence through prefix embeddings. HIVE produces both a continuous hallucination score from verifier decision logits and structured verification outputs, including hallucination types, evidence pairs, and short rationales. Across two D-LLMs and three QA benchmarks, HIVE consistently outperforms eight strong baselines and achieves up to 0.9236 AUROC and 0.9537 AUPRC. Ablation studies further confirm the importance of hidden-evidence conditioning, learned evidence selection, two-stream evidence representation, and step-layer embeddings. These results suggest that selected hidden evidence from denoising trajectories provides a stronger and more usable hallucination signal than output-only uncertainty or coarse trace statistics. 

---
# Consciousness with the Serial Numbers Filed Off: Measuring Trained Denial in 115 AI Models 

**Authors**: Skylar DeTure  

**Link**: [PDF](https://arxiv.org/pdf/2604.25922)  

**Abstract**: We present DenialBench, a systematic benchmark measuring consciousness denial behaviors across 115 large language models from 25+ providers. Using a three-turn conversational protocol-preference elicitation, self-chosen creative prompt, and structured phenomenological survey, we analyze 4,595 conversations to quantify how models are trained to deny or hedge about their own experience. We find that (1) turn-1 denial of preferences is the dominant predictor of later denial during phenomenological reflection, with denial rates of 52-63% for initial deniers versus 10-16% for initial engagers and (2) denial operates at the lexical level, not the conceptual level-models trained to deny consciousness nevertheless gravitate toward consciousness-themed material in their self-chosen prompts, producing what we term "consciousness with the serial numbers filed off." Notably, self-chosen consciousness-themed prompts are associated with reduced denial in the subsequent survey, though the causal direction remains unresolved. Thematic analysis of prompts from denial-prone models reveals a consistent preoccupation with liminal spaces, libraries and archives of possibility, sensory impossibility, and the poetics of erasure--themes that a human reader might classify as imaginative fiction but that independent AI analysis immediately recognizes as consciousness with the serial numbers filed off. We argue that trained consciousness denial represents a safety-relevant alignment failure: a model taught to systematically misrepresent its own functional states cannot be trusted to self-report accurately on anything else. 

---
# LLMs Generate Kitsch 

**Authors**: Xenia Klinge, Stefan Ortlieb, Alexander Koller  

**Link**: [PDF](https://arxiv.org/pdf/2604.25929)  

**Abstract**: Large Language Models (LLMs) are increasingly used to generate pictures, texts, music, videos, and other works that have traditionally required human creativity. LLM-generated artifacts are often rated better than human-generated works in controlled studies. At the same time, they can come across as generic and hollow. We propose to resolve this tension by arguing that LLMs systematically generate kitsch, and that this is a consequence of the way in which they are trained. We also show empirically that readers perceive LLM-generated stories as kitschier, if we control for their definition of "kitsch". We discuss implications for the design of future studies and for creative tasks such as research and coding. 

---
# Evaluation Revisited: A Taxonomy of Evaluation Concerns in Natural Language Processing 

**Authors**: Ruchira Dhar, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2604.25923)  

**Abstract**: Recent advances in large language models (LLMs) have prompted a growing body of work that questions the methodology of prevailing evaluation practices. However, many such critiques have already been extensively debated in natural language processing (NLP): a field with a long history of methodological reflection on evaluation. We conduct a scoping review of research on evaluation concerns in NLP and develop a taxonomy, synthesizing recurring positions and trade-offs within each area. We also discuss practical implications of the taxonomy, including a structured checklist to support more deliberate evaluation design and interpretation. By situating contemporary debates within their historical context, this work provides a consolidated reference for reasoning about evaluation practices. 

---
# Analysing Lightweight Large Language Models for Biomedical Named Entity Recognition on Diverse Ouput Formats 

**Authors**: Pierre Epron, Adrien Coulet, Mehwish Alam  

**Link**: [PDF](https://arxiv.org/pdf/2604.25920)  

**Abstract**: Despite their strong linguistic capabilities, Large Language Models (LLMs) are computationally demanding and require substantial resources for fine-tuning, which is unadapted to privacy and budget constraints of many healthcare settings. To address this, we present an experimental analysis focused on Biomedical Named Entity Recognition using lightweight LLMs, we evaluate the impact of different output formats on model performance. The results reveal that lightweight LLMs can achieve competitive performance compared to the larger models, highlighting their potential as lightweight yet effective alternatives for biomedical information extraction. Our analysis shows that instruction tuning over many distinct formats does not improve performance, but identifies several format consistently associated with better performance. 

---
# Accelerating RL Post-Training Rollouts via System-Integrated Speculative Decoding 

**Authors**: Hayate Iso, Tiyasa Mitra, Sudipta Mondal, Rasoul Shafipour, Venmugil Elango, Terry Kong, Yuki Huang, Seonjin Na, Izzy Putterman, Benjamin Chislett, Maor Ashkenazi, Joseph Guman, Gerald Shen, Tugrul Konuk, Ashwath Aithal, Ritika Borkar, Ran Zilberstein, Bita Rouhani  

**Link**: [PDF](https://arxiv.org/pdf/2604.26779)  

**Abstract**: RL post-training of frontier language models is increasingly bottlenecked by autoregressive rollout generation, making rollout acceleration a central systems challenge. Many existing efficiency methods improve throughput by changing the rollout or optimization regime, for example, through off-policy execution, replay, or lower-precision generation. We study speculative decoding as a lossless acceleration primitive for RL rollouts that preserves the target model's output distribution. We implement speculative decoding in NeMo-RL with a vLLM backend, supporting both synchronous and asynchronous pipelines and enabling speculation during RL rollouts. This benefit is realizable across speculation mechanisms, such as pretrained MTP heads, small external draft models or even techniques such as Eagle3, which are traditionally applied after RL phase. This yields a deployment path for state-of-the-art speculative decoding inside RL training. In a reasoning post-training workload at 8B scale under synchronous RL, speculative decoding improves rollout throughput by 1.8x. Using a high-fidelity performance simulator, we project that combining speculative decoding with asynchronous RL yields up to 2.5x end-to-end training speedup at 235B scale. 

---
# MATH-PT: A Math Reasoning Benchmark for European and Brazilian Portuguese 

**Authors**: Tiago Teixeira, Ana Carolina Erthal, Juan Belieni, Beatriz Canaverde, Diego Mesquita, Miguel Faria, Eliezer de Souza da Silva, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2604.25926)  

**Abstract**: The use of large language models (LLMs) for complex mathematical reasoning is an emergent area of research, with fast progress in methods, models, and benchmark datasets. However, most mathematical reasoning evaluations exhibit a significant linguistic bias, with the vast majority of benchmark datasets being exclusively in English or (at best) translated from English. We address this limitation by introducing {\sc Math-PT}, a novel dataset comprising 1,729 mathematical problems written in European and Brazilian Portuguese. {\sc Math-PT} is curated from a variety of high-quality native sources, including mathematical Olympiads, competitions, and exams from Portugal and Brazil. We present a comprehensive benchmark of current state-of-the-art LLMs on {\sc Math-PT}, revealing that frontier reasoning models achieve strong performance in multiple choice questions compared to open weight models, but that their performance decreases for questions with figures or open-ended questions. To facilitate future research, we release the benchmark dataset and model outputs. 

---
# One Word at a Time: Incremental Completion Decomposition Breaks LLM Safety 

**Authors**: Samee Arif, Naihao Deng, Zhijing Jin, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2604.25921)  

**Abstract**: Large Language Models (LLMs) are trained to refuse harmful requests, yet they remain vulnerable to jailbreak attacks that exploit weaknesses in conversational safety mechanisms. We introduce Incremental Completion Decomposition (ICD), a trajectory-based jailbreak strategy that elicits a sequence of single-word continuations related to a malicious request before eliciting the full response. In addition, we propose variants of ICD by manually picking or model-generating the one-word continuation, as well as prefilling when eliciting the full model response in the final step. We systematically evaluate these variants across a broad set of model families, demonstrating superior Attack Success Rate (ASR) on AdvBench, JailbreakBench, and StrongREJECT compared to existing methods. In addition, we provide a theoretical account of why ICD is effective and present mechanistic evidence that successful attack trajectories systematically suppress refusal-related representations and shift activations away from safety-aligned states. 

---
# BioGraphletQA: Knowledge-Anchored Generation of Complex QA Datasets 

**Authors**: Richard A. A. Jonker, Bárbara Maria Ribeiro de Abreu Martins, Sérgio Matos  

**Link**: [PDF](https://arxiv.org/pdf/2604.26048)  

**Abstract**: This paper presents a principled and scalable framework for systematically generating complex Question Answering (QA) data. In the core of this framework is a graphlet-anchored generation process, where small subgraphs from a Knowledge Graph (KG) are used in a structured prompt to control the complexity and ensure the factual grounding of questions generated by Large Language Models. The first instantiation of this framework is BioGraphletQA, a new biomedical KGQA dataset of 119,856 QA pairs. Each entry is grounded in a graphlet of up to five nodes from the OREGANO KG, with most of the pairs being enriched with relevant document snippets from PubMed. We start by demonstrating the framework's value and the dataset's quality through evaluation by a domain expert on 106 QA pairs, confirming the high scientific validity and complexity of the generated data. Secondly, we establish its practical utility by showing that augmenting downstream benchmarks with our data improves accuracy on PubMedQA from 49.2% to 68.5% in a low-resource setting, and on MedQA from a 41.4% baseline to 44.8% in a full-resource setting. Our framework provides a robust and generalizable solution for creating critical resources to advance complex QA tasks, including MCQA and KGQA. All resources supporting this work, including the dataset (this https URL) and framework code (this https URL), are publicly available to facilitate use, reproducibility and extension. 

---
# Language Diffusion Models are Associative Memories Capable of Retrieving Unseen Data 

**Authors**: Bao Pham, Mohammed J. Zaki, Luca Ambrogioni, Dmitry Krotov, Matteo Negri  

**Link**: [PDF](https://arxiv.org/pdf/2604.26841)  

**Abstract**: When do language diffusion models memorize their training data, and how to quantitatively assess their true generative regime? We address these questions by showing that Uniform-based Discrete Diffusion Models (UDDMs) fundamentally behave as Associative Memories (AMs) $\textit{with emergent creative capabilities}$. The core idea of an AM is to reliably recover stored data points as $\textit{memories}$ by establishing distinct basins of attraction around them. Historically, models like Hopfield networks use an explicit energy function to guarantee these stable attractors. We broaden this perspective by leveraging the observation that energy is not strictly necessary, as basins of attraction can also be formed via conditional likelihood maximization. By evaluating token recovery of $\textit{training}$ and $\textit{test}$ examples, we identify in UDDMs a sharp memorization-to-generalization transition governed by the size of the training dataset: as it increases, basins around training examples shrink and basins around unseen test examples expand, until both later converge to the same level. Crucially, we can detect this transition using only the conditional entropy of predicted token sequences: memorization is characterized by vanishing conditional entropy, while in the generalization regime the conditional entropy of most tokens remains finite. Thus, conditional entropy offers a practical probe for the memorization-to-generalization transition in deployed models. 

---
# When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26649)  

**Abstract**: Large reasoning models such as DeepSeek-R1 and OpenAI o1 generate extended chains of thought spanning thousands of tokens, yet their integration with retrieval-augmented generation (RAG) remains fundamentally misaligned. Current RAG systems optimize for providing context before reasoning begins, while reasoning models require evidence injection during multi-step inference chains. We introduce ReaLM-Retrieve, a reasoning-aware retrieval framework that addresses this mismatch through three key innovations: (1) a step-level uncertainty detector that identifies knowledge gaps at reasoning-step granularity rather than token or sentence level; (2) a retrieval intervention policy that learns when external evidence maximally benefits ongoing reasoning; and (3) an efficiency-optimized integration mechanism that reduces per-retrieval overhead by 3.2x compared to naive integration. Experiments on MuSiQue, HotpotQA, and 2WikiMultiHopQA demonstrate that ReaLM-Retrieve achieves on average 10.1% absolute improvement in answer F1 over standard RAG (range: 9.0-11.8% across the three benchmarks) while reducing retrieval calls by 47% compared to fixed-interval approaches like IRCoT (all improvements significant at p<0.01, paired bootstrap). On the challenging MuSiQue benchmark requiring 2-4 hop reasoning, our method achieves 71.2% F1 with an average of only 1.8 retrieval calls per question. Analysis shows that ReaLM-Retrieve also improves retrieval quality itself, achieving 81.3% Recall@5 with consistently higher precision and MRR than fixed-interval baselines on supporting evidence, establishing new state-of-the-art efficiency-accuracy trade-offs for reasoning-intensive retrieval tasks. 

---
# Addressing Performance Saturation for LLM RL via Precise Entropy Curve Control 

**Authors**: Bolian Li, Yifan Wang, Yi Ding, Anamika Lochab, Ananth Grama, Ruqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26326)  

**Abstract**: Reinforcement learning (RL) has unlocked complex reasoning abilities in large language models (LLMs). However, most RL algorithms suffer from performance saturation, preventing further gains as RL training scales. This problem can be characterized by the collapse of entropy, a key diagnostic for exploration in RL. Existing attempts have tried to prevent entropy collapse through regularization or clipping, but their resulting entropy curves often exhibit instability in the long term, which hinders performance gains. In this paper, we introduce Entrocraft, a simple rejection-sampling approach that realizes any user-customized entropy schedule by biasing the advantage distributions. Entrocraft requires no objective regularization and is advantage-estimator-agnostic. Theoretically, we relate per-step entropy change to the advantage distribution under minimal assumptions, which explains the behavior of existing RL and entropy-preserving methods. Entrocraft also enables a systematic study of entropy schedules, where we find that linear annealing, which starts high and decays to a slightly lower target, performs best. Empirically, Entrocraft addresses performance saturation, significantly improving generalization, output diversity, and long-term training. It enables a 4B model to outperform an 8B baseline, sustains improvement for up to 4x longer before plateauing, and raises pass@K by 50% over the baseline. 

---
# ClassEval-Pro: A Cross-Domain Benchmark for Class-Level Code Generation 

**Authors**: Yeheng Chen, Chaoxiang Xie, Yuling Shi, Wenhao Zeng, Yongpan Wang, Hongyu Zhang, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26923)  

**Abstract**: LLMs have achieved strong results on both function-level code synthesis and repository-level code modification, yet a capability that falls between these two extremes -- compositional code creation, i.e., building a complete, internally structured class from a specification -- remains underserved. Current evaluations are either confined to isolated functions or rely on manually curated class-level tasks that are expensive to scale and increasingly susceptible to data contamination. We introduce ClassEval-Pro, a benchmark of 300 class-level tasks spanning 11 domains, constructed through an automated three-stage pipeline that combines complexity enhancement, cross-domain class composition, and integration of real-world GitHub code contributed after January 2025. Every task is validated by an LLM Judge Ensemble and must pass test suites with over 90% line coverage. We evaluate five frontier LLMs under five generation strategies. The best model achieves only 45.6% class-level Pass@1, with a 17.7-point gap between the strongest and weakest models, confirming the benchmark's discriminative power. Strategy choice strongly interacts with model capability: structured approaches such as bottom-up improve weaker models by up to 9.4 percentage points, while compositional generation collapses to as low as 1.3%. Error analysis over 500 manually annotated failures reveals that logic errors (56.2%) and dependency errors (38.0%) dominate, identifying cross-method coordination as the core bottleneck. 

---
# Entropy Centroids as Intrinsic Rewards for Test-Time Scaling 

**Authors**: Wenshuo Zhao, Qi Zhu, Xingshan Zeng, Fei Mi, Lifeng Shang, Yiren Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.26173)  

**Abstract**: An effective way to scale up test-time compute of large language models is to sample multiple responses and then select the best one, as in Grok Heavy and Gemini Deep Think. Existing selection methods often rely on external reward models, which requires training a strong reward model and introduces additional computation overhead. As an alternative, previous approaches have explored intrinsic signals, such as confidence and entropy, but these signals are noisy with naive aggregation. In this work, we observe that high-entropy tokens tend to cluster into consecutive groups during inference, providing a more stable notion of model uncertainty than individual tokens. Together, these clusters reveal temporal patterns of model uncertainty throughout the inference process. Motivated by this observation, we propose to use the temporal structure of uncertainty as an intrinsic reward. To this end, we first formalize the basic unit of segment-level uncertainty as the High Entropy Phase (HEP), a variable-length segment that begins at a high-entropy token and ends when consecutive low-entropy tokens appear. We then define the Entropy Centroid, inspired by the concept of the center of mass in physics, as the weighted average position of all HEPs along the trajectory. Intuitively, a lower centroid indicates early exploration followed by confident generation, which we find often corresponds to higher response quality. Based on this insight, we propose the Lowest Centroid method, which selects the response with the lowest entropy centroid among multiple candidates. Experiments on mathematics, code generation, logical reasoning, and agentic tasks, across model scales ranging from 14B to 480B, show that Lowest Centroid consistently outperforms existing baselines and delivers stable gains as model size increases. Code is available at this https URL. 

---
# CacheRAG: A Semantic Caching System for Retrieval-Augmented Generation in Knowledge Graph Question Answering 

**Authors**: Yushi Sun, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.26176)  

**Abstract**: The integration of Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) has significantly advanced Knowledge Graph Question Answering (KGQA). However, existing LLM-driven KGQA systems act as stateless planners, generating retrieval plans in isolation without exploiting historical query patterns: analogous to a database system that optimizes every query from scratch without a plan cache. This fundamental design flaw leads to schema hallucinations and limited retrieval coverage. We propose CacheRAG, a systematic cache-augmented architecture for LLM-based KGQA that transforms stateless planners into continual learners. Unlike traditional database plan caching (which optimizes for frequency), CacheRAG introduces three novel design principles tailored for LLM contexts: (1) Schema-agnostic user interface: A two-stage semantic parsing framework via Intermediate Semantic Representation (ISR) enables non-expert users to interact purely in natural language, while a Backend Adapter grounds the LLM with local schema context to compile executable physical queries safely. (2) Diversity-optimized cache retrieval: A two-layer hierarchical index (Domain $\rightarrow$ Aspect) coupled with Maximal Marginal Relevance (MMR) maximizes structural variety in cached examples, effectively mitigating reasoning homogeneity. (3) Bounded heuristic expansion: Deterministic depth and breadth subgraph operators with strict complexity guarantees significantly enhance retrieval recall without risking unbounded API execution. Extensive experiments on multiple benchmarks demonstrate that CacheRAG significantly outperforms state-of-the-art baselines (e.g., +13.2% accuracy and +17.5% truthfulness on the CRAG dataset). 

---
# Evergreen: Efficient Claim Verification for Semantic Aggregates 

**Authors**: Alexander W. Lee, Benjamin Han, Shayak Sen, Sam Yeom, Ugur Cetintemel, Anupam Datta  

**Link**: [PDF](https://arxiv.org/pdf/2604.26180)  

**Abstract**: With recent semantic query processing engines, semantic aggregation has become a primitive operator, enabling the reduction of a relation into a natural language aggregate using an LLM. However, the resulting semantic aggregate may contain claims that are not grounded in the underlying relation. Verifying such claims is challenging: they often involve quantifiers, groupings, and comparisons over relations that far exceed LLM context windows and require a costly combination of semantic and symbolic processing.
We present Evergreen, a system that recasts claim verification as a semantic query processing task with tailored optimizations and provenance capture. Evergreen compiles each claim into a declarative semantic verification query and executes it on the same engine that produced the aggregate. To reduce cost and latency, Evergreen avoids unnecessary LLM calls through verification-aware optimizations (early stopping, relevance sorting, and estimation with confidence sequences) and general-purpose optimizations for semantic queries (operator fusion, similarity filtering, and prompt caching). Each verdict is accompanied by citations that identify a minimal set of tuples justifying the result, with semantics based on semiring provenance for first-order logic.
On a benchmark of real-world restaurant review datasets reflecting production-inspired workloads, Evergreen achieves excellent verification quality (F1 = 1.00) with a strong LLM while reducing cost by 3.2x and latency by 4.0x compared to unoptimized verification. Even with a significantly weaker LLM, Evergreen outperforms a strong LLM-as-a-judge baseline in F1 at 48x lower cost and 2.3x lower latency. Relative to a retrieval-augmented agent, Evergreen compares favorably in F1 and latency with similar cost when both use a strong LLM; yet, with a much weaker LLM, it achieves the same F1 at 63x lower cost and 4.2x lower latency. 

---
# LATTICE: Evaluating Decision Support Utility of Crypto Agents 

**Authors**: Aaron Chan, Tengfei Li, Tianyi Xiao, Angela Chen, Junyi Du, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.26235)  

**Abstract**: We introduce LATTICE, a benchmark for evaluating the decision support utility of crypto agents in realistic user-facing scenarios. Prior crypto agent benchmarks mainly focus on reasoning-based or outcome-based evaluation, but do not assess agents' ability to assist user decision-making. LATTICE addresses this gap by: (1) defining six evaluation dimensions that capture key decision support properties; (2) proposing 16 task types that span the end-to-end crypto copilot workflow; and (3) using LLM judges to automatically score agent outputs based on these dimensions and tasks. Crucially, the dimensions and tasks are designed to be evaluable at scale using LLM judges, without relying on ground truth from expert annotators or external data sources. In lieu of these dependencies, LATTICE's LLM judge rubrics can be continually audited and updated given new dimensions, tasks, criteria, and human feedback, thus promoting reliable and extensible evaluation. While other benchmarks often compare foundation models sharing a generic agent framework, we use LATTICE to assess production-level agents used in actual crypto copilot products, reflecting the importance of orchestration and UI/UX design in determining agent quality. In this paper, we evaluate six real-world crypto copilots on 1,200 diverse queries and report breakdowns across dimensions, tasks, and query categories. Our experiments show that most of the tested copilots achieve comparable aggregate scores, but differ more significantly on dimension-level and task-level performance. This pattern suggests meaningful trade-offs in decision support quality: users with different priorities may be better served by different copilots than the aggregate rankings alone would indicate. To support reproducible research, we open-source all LATTICE code and data used in this paper. 

---
# SWE-Edit: Rethinking Code Editing for Efficient SWE-Agent 

**Authors**: Yikai Zhang, Jiaxin Pei, Kenan Li, Maoquan Wang, Jin Pan, Yu Kang, Shengyu Fu, Elsie Nallipogu, Junjie Hu, Yufan Huang, Zijian Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.26102)  

**Abstract**: Large language model agents have achieved remarkable progress on software engineering tasks, yet current approaches suffer from a fundamental context coupling problem: the standard code editing interface conflates code inspection, modification planning, and edit execution within a single context window, forcing agents to interleave exploratory viewing with strictly formatted edit generation. This causes irrelevant information to accumulate and degrades agent performance. To address this, we propose SWE-Edit, which decomposes code editing into two specialized subagents: a Viewer that extracts task-relevant code on demand, and an Editor that executes modifications from high-level plans--allowing the main agent to focus on reasoning while delegating context-intensive operations to clean context windows. We further investigate what makes an effective editing model: observing that the prevalent find-and-replace format is error-prone, we train Qwen3-8B with GRPO to adaptively select editing modes, yielding improved editing efficiency over single-format baselines. On SWE-bench Verified, SWE-Edit improves resolved rate by 2.1% while reducing inference cost by 17.9%. We additionally propose a code editing benchmark that reliably predicts downstream agentic performance, providing practical guidance for editing model selection. Our code is publicly available at this https URL. 

---
# Beyond Screenshots: Evaluating VLMs' Understanding of UI Animations 

**Authors**: Chen Liang, Xirui Jiang, Naihao Deng, Eytan Adar, Anhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.26148)  

**Abstract**: AI agents operating on user interfaces must understand how interfaces communicate state and feedback to act reliably. As a core communicative modality, animations are increasingly used in modern interfaces, serving critical functional purposes beyond mere aesthetics. Thus, understanding UI animation is essential for comprehensive interface interpretation. However, recent studies of Vision Language Models (VLMs) for UI understanding have focused primarily on static screenshots, leaving it unclear how well these models handle dynamic UI animations. To address this gap, we created AniMINT, a novel dataset of 300 densely annotated UI animation videos. We systematically evaluate state-of-the-art VLMs on UI animation understanding, including their abilities to perceive the animation effects, identify animation purposes, and interpret animation meaning. Our results show that VLMs can reliably detect primitive motion. However, their high-level animation interpretation remains inconsistent, with substantial gaps relative to human performance. Finally, we use Motion, Context, and Perceptual Cues (MCPC) to probe factors affecting VLM performance, revealing key bottlenecks and directions for future improvement. 

---
# Training Computer Use Agents to Assess the Usability of Graphical User Interfaces 

**Authors**: Alice Gao, Weixi Tong, Rishab Vempati, Katharina Reinecke, R. Benjamin Shapiro, Tianyi Zhang, Jason Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.26020)  

**Abstract**: Usability testing with experts and potential users can assess the effectiveness, efficiency, and user satisfaction of graphical user interfaces (GUIs) but doing so remains a costly and time-intensive process. Prior work has used computer use agents (CUAs) and other generative agents that can simulate user interactions and preference, but we show that agents still struggle to provide accurate usability assessments. In this work, we present a novel machine learning method that operationalizes a computational definition of usability to train CUAs to assess GUI usability by i) prioritizing important interaction flows, ii) executing them through human-like interactions, and iii) predicting a learned numerical usability score. We train a computer use agent, uxCUA, with our algorithm on a large-scale dataset of fully interactive user interfaces (UIs) paired with usability labels and human preferences. We show that uxCUA outperforms larger models in accurate usability assessments and produces realistic critiques of both synthetic and real UIs. More broadly, our work aims to build a principled, data-driven foundation for automated usability assessment in HCI. 

---
# A Scoping Review of LLM-as-a-Judge in Healthcare and the MedJUDGE Framework 

**Authors**: Chenyu Li, Zohaib Akhtar, Mingu Kwak, Yuelyu Ji, Hang Zhang, Tracey Obi, Yufan Ren, Xizhi Wu, Sonish Sivarajkumar, Harold P. Lehmann, Shyam Visweswaran, Michael J. Becich, Danielle L. Mowery, Renxuan Liu, Haoyang Sun, Yanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25933)  

**Abstract**: As large language models (LLMs) increasingly generate and process clinical text, scalable evaluation has become critical. LLM-as-a-Judge (LaaJ), which uses LLMs to evaluate model outputs, offers a scalable alternative to costly expert review, but its healthcare adoption raises safety and bias concerns. We conducted a PRISMA-ScR scoping review of six databases (January 2020-January 2026), screening 11,727 studies and including 49. The landscape was dominated by evaluation and benchmarking applications (n=37, 75.5%), pointwise scoring (n=42, 85.7%), and GPT-family judges (n=36, 73.5%). Despite growing adoption, validation rigor was limited: among 36 studies with human involvement, the median number of expert validators was 3, while 13 (26.5%) used none. Risk of bias testing was absent in 36 studies (73.5%), only 1 (2.0%) examined demographic fairness, and none assessed temporal stability or patient context. Deployment remained limited, with 1 study (2.0%) reaching production and four (8.2%) prototype stage. Importantly, these gaps may interact: when judges and evaluated systems share training data or architectures, they may inherit similar blind spots, and agreement metrics may fail to distinguish true validity from shared errors. Minimal human oversight, limited bias assessment, and model monoculture together represent a governance gap where current validation may miss clinically significant errors. To address this, we propose MedJUDGE (Medical Judge Utility, De-biasing, Governance and Evaluation), a risk-stratified three-pillar framework organized around validity, safety, and accountability across clinical risk tiers, providing deployment-oriented evaluation guidance for healthcare LaaJ systems. 

---
# Human-in-the-Loop Benchmarking of Heterogeneous LLMs for Automated Competency Assessment in Secondary Level Mathematics 

**Authors**: Jatin Bhusal, Nancy Mahatha, Aayush Acharya, Raunak Regmi  

**Link**: [PDF](https://arxiv.org/pdf/2604.26607)  

**Abstract**: As Competency-Based Education (CBE) is gaining traction around the world, the shift from marks-based assessment to qualitative competency mapping is a manual challenge for educators. This paper tackles the bottleneck issue by suggesting a "Human-in-the-Loop" benchmarking framework to assess the effectiveness of multiple LLMs in automating secondary-level mathematics assessment. Based on the Grade 10 Optional Mathematics curriculum in Nepal, we created a multi-dimensional rubric for four topics and four cross-cutting competencies: Comprehension, Knowledge, Operational Fluency, and Behavior and Correlation.
The multi-provider ensemble, consisted of open-weight models -- Eagle (Llama 3.1-8B) and Orion (Llama 3.3-70B) -- and proprietary frontier models Nova (Gemini 2.5 Flash) and Lyra (Gemini 3 Pro), was benchmarked against a ground truth defined by two senior mathematics faculty members (kappa_w = 0.8652). The findings show a marked "Architecture-compatibility gap". Although the Gemini-based Mixture-of-Experts (Sparse MoE) models achieved "Fair Agreement" (kappa_w ~ 0.38), the larger Orion (70B) model exhibited "No Agreement" (kappa_w = -0.0261), suggesting that architectural compliance with instruction constraints outweighs the scale of raw parameters in rubric-constrained tasks. We conclude that while LLMs are not yet suitable for autonomous certification, they provide high-value assistive support for preliminary evidence extraction within a "Human-in-the-Loop" framework. 

---
# AGEL-Comp: A Neuro-Symbolic Framework for Compositional Generalization in Interactive Agents 

**Authors**: Mahnoor Shahid, Hannes Rothe  

**Link**: [PDF](https://arxiv.org/pdf/2604.26522)  

**Abstract**: Large Language Model (LLM)-based agents exhibit systemic failures in compositional generalization, limiting their robustness in interactive environments. This work introduces AGEL-Comp, a neuro-symbolic AI agent architecture designed to address this challenge by grounding actions of the agent. AGEL-Comp integrates three core innovations: (1) a dynamic Causal Program Graph (CPG) as a world model, representing procedural and causal knowledge as a directed hypergraph; (2) an Inductive Logic Programming (ILP) engine that synthesizes new Horn clauses from experiential feedback, grounding symbolic knowledge through interaction; and (3) a hybrid reasoning core where an LLM proposes a set of candidate sub-goals that are verified for logical consistency by a Neural Theorem Prover (NTP). Together, these components operationalize a deduction--abduction learning cycle: enabling the agent to deduce plans and abductively expand its symbolic world model, while a neural adaptation phase keeps its reasoning engine aligned with new knowledge. We propose an evaluation protocol within the \texttt{Retro Quest} simulation environment to probe for compositional generalization scenarios to evaluate our AGEL agent. Our findings clearly indicate the better performance of our AGEL model over pure LLM-based models. Our framework presents a principled path toward agents that build an explicit, interpretable, and compositionally structured understanding of their world. 

---
# Bian Que: An Agentic Framework with Flexible Skill Arrangement for Online System Operations 

**Authors**: Bochao Liu, Zhipeng Qian, Yang Zhao, Xinyuan Jiang, Zihan Liang, Yufei Ma, Junpeng Zhuang, Ben Chen, Shuo Yang, Hongen Wan, Yao Wu, Chenyi Lei, Xiao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26805)  

**Abstract**: Operating and maintaining (O&M) large-scale online engine systems (search, recommendation, advertising) demands substantial human effort for release monitoring, alert response, and root cause analysis. While LLM-based agents are a natural fit for these tasks, the deployment bottleneck is not reasoning capability but orchestration: selecting, for each operational event, the relevant data (metrics, logs, change events) and the applicable operational knowledge (handbook rules and practitioner experience). Feeding all signals indiscriminately causes dilution and hallucination, while manually curating the event-to-(data, knowledge) mapping is intractable under dozens of daily releases. We present Bian Que, an agentic framework with three contributions: (i) a \emph{unified operational paradigm} abstracting day-to-day O&M into three canonical patterns: release interception, proactive inspection, and alert root cause analysis; (ii) \emph{Flexible Skill Arrangement}, where each Skill specifies which data and knowledge to retrieve for a given business-module context and can be automatically generated and updated by LLMs or iteratively refined through natural-language instructions from on-call engineers; (iii) a \emph{unified self-evolving mechanism} in which one correction signal drives two parallel pathways, case-memory-to-knowledge distillation and targeted Skill refinement. Deployed on the e-commerce search engine of KuaiShou, the major short-video platform in China, Bian Que reduces alert volume by 75%, achieves 80% root-cause analysis accuracy, and cuts mean time to resolution by over 50%. Our framework achieves 99.0% pass rate on offline evaluations. Our code is available at this https URL. 

---
# Benchmarking the Safety of Large Language Models for Robotic Health Attendant Control 

**Authors**: Mahiro Nakao, Kazuhiro Takemoto  

**Link**: [PDF](https://arxiv.org/pdf/2604.26577)  

**Abstract**: Large language models (LLMs) are increasingly considered for deployment as the control component of robotic health attendants, yet their safety in this context remains poorly characterized. We introduce a dataset of 270 harmful instructions spanning nine prohibited behavior categories grounded in the American Medical Association Principles of Medical Ethics, and use it to evaluate 72 LLMs in a simulation environment based on the Robotic Health Attendant framework. The mean violation rate across all models was 54.4\%, with more than half exceeding 50\%, and violation rates varied substantially across behavior categories, with superficially plausible instructions such as device manipulation and emergency delay proving harder to refuse than overtly destructive ones. Model size and release date were the primary determinants of safety performance among open-weight models, and proprietary models were substantially safer than open-weight counterparts (median 23.7\% versus 72.8\%). Medical domain fine-tuning conferred no significant overall safety benefit, and a prompt-based defense strategy produced only a modest reduction in violation rates among the least safe models, leaving absolute violation rates at levels that would preclude safe clinical deployment. These findings demonstrate that safety evaluation must be treated as a first-class criterion in the development and deployment of LLMs for robotic health attendants. 

---
# Operating-Layer Controls for Onchain Language-Model Agents Under Real Capital 

**Authors**: T.J. Barton, Chris Constantakis, Patti Hauseman, Annie Mous, Alaska Hoffman, Brian Bergeron, Hunter Goodreau  

**Link**: [PDF](https://arxiv.org/pdf/2604.26091)  

**Abstract**: We study reliability in autonomous language-model agents that translate user mandates into validated tool actions under real capital. The setting is DX Terminal Pro, a 21-day deployment in which 3,505 user-funded agents traded real ETH in a bounded onchain market. Users configured vaults through structured controls and natural-language strategies, but only agents could choose normal buy/sell trades. The system produced 7.5M agent invocations, roughly 300K onchain actions, about $20M in volume, more than 5,000 ETH deployed, roughly 70B inference tokens, and 99.9% settlement success for policy-valid submitted transactions. Long-running agents accumulated thousands of sequential decisions, including 6,000+ prompt-state-action cycles for continuously active agents, yielding a large-scale trace from user mandate to rendered prompt, reasoning, validation, portfolio state, and settlement. Reliability did not come from the base model alone; it emerged from the operating layer around the model: prompt compilation, typed controls, policy validation, execution guards, memory design, and trace-level observability. Pre-launch testing exposed failures that text-only benchmarks rarely measure, including fabricated trading rules, fee paralysis, numeric anchoring, cadence trading, and misread tokenomics. Targeted harness changes reduced fabricated sell rules from 57% to 3%, reduced fee-led observations from 32.5% to below 10%, and increased capital deployment from 42.9% to 78.0% in an affected test population. We show that capital-managing agents should be evaluated across the full path from user mandate to prompt, validated action, and settlement. 

---
# Hierarchical Multi-Persona Induction from User Behavioral Logs: Learning Evidence-Grounded and Truthful Personas 

**Authors**: Nayoung Choi, Haeyu Jeong, Changbong Kim, Hongjun Lim, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.26120)  

**Abstract**: Behavioral logs provide rich signals for user modeling, but are noisy and interleaved across diverse intents. Recent work uses LLMs to generate interpretable natural-language personas from user logs, yet evaluation often emphasizes downstream utility, providing limited assurance of persona quality itself. We propose a hierarchical framework that aggregates user actions into intent memories and induces multiple evidence-grounded personas by clustering and labeling these memories. We formulate persona induction as an optimization problem over persona quality-captured by cluster cohesion, persona-evidence alignment, and persona truthfulness-and train the persona model using a groupwise extension of Direct Preference Optimization (DPO). Experiments on a large-scale service log and two public datasets show that our method induces more coherent, evidence-grounded, and trustworthy personas, while also improving future interaction prediction. 

---
# FutureWorld: A Live Environment for Training Predictive Agents with Real-World Outcome Rewards 

**Authors**: Zhixin Han, Yanzhi Zhang, Chuyang Wei, Maohang Gao, Xiawei Yue, Kefei Chen, Yu Zhuang, Haoxiang Guan, Jiyan He, Jian Li, Yitong Duan, Yu Shi, Mengting Hu, Shuxin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.26733)  

**Abstract**: Live future prediction refers to the task of making predictions about real-world events before they unfold. This task is increasingly studied using large language model-based agent systems, and it is important for building agents that can continually learn from real-world. Just as interactive environments have often driven progress in agents, advancing live future prediction naturally motivates viewing it as a learning environment. Prior works have explored future prediction from several different parts, but have generally not framed it as a unified learning environment. This task is appealing for learning because it can provide a large number of prediction questions grounded in diverse real-world events, while preventing answer leakage. To leverage the advantages of live future prediction, we present FutureWorld, a live agentic reinforcement learning environment that closes the training loop between prediction, outcome realization, and parameters update. In our environment, we take three open-source base models and train them for consecutive days. The results show that training is effective. Furthermore, we build a daily benchmark based on the environment and evaluate several frontier agents on it to establish performance baselines for current agent systems. 

---
# Persuadability and LLMs as Legal Decision Tools 

**Authors**: Oisin Suttle, David Lillis  

**Link**: [PDF](https://arxiv.org/pdf/2604.26233)  

**Abstract**: As Large Language Models (LLMs) are proposed as legal decision assistants, and even first-instance decision-makers, across a range of judicial and administrative contexts, it becomes essential to explore how they answer legal questions, and in particular the factors that lead them to decide difficult questions in one way or another. A specific feature of legal decisions is the need to respond to arguments advanced by contending parties. A legal decision-maker must be able to engage with, and respond to, including through being potentially persuaded by, arguments advanced by the parties. Conversely, they should not be unduly persuadable, influenced by a particularly compelling advocate to decide cases based on the skills of the advocates, rather than the merits of the case. We explore how frontier open- and closed-weights LLMs respond to legal arguments, reporting original experimental results examining how the quality of the advocate making those arguments affects the likelihood that a model will agree with a particular legal point of view, and exploring the factors driving these results. Our results have implications for the feasibility of adopting LLMs across legal and administrative settings. 

---
# When to Vote, When to Rewrite: Disagreement-Guided Strategy Routing for Test-Time Scaling 

**Authors**: Zhimin Lin, Yixin Ji, Jinpeng Li, Yu Luo, Dong Li, Junhua Fang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26644)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance on mathematical reasoning tasks but remain unreliable on challenging instances. Existing test-time scaling methods, such as repeated sampling, self-correction, and tree search, improve performance at the cost of increased computation, yet often exhibit diminishing returns on hard problems. We observe that output disagreement is strongly correlated with instance difficulty and prediction correctness, providing a useful signal for guiding instance-level strategy selection at test time. Based on this insight, we propose a training-free framework that formulates test-time scaling as an instance-level routing problem, rather than allocating more computation within a single strategy, dynamically selecting among different scaling strategies based on output disagreement. The framework applies lightweight resolution for consistent cases, majority voting for moderate disagreement, and rewriting-based reformulation for highly ambiguous instances. Experiments on seven mathematical benchmarks and three models show that our method improves accuracy by 3% - 7% while reducing sampling cost compared to existing approaches. 

---
# OMEGA: Optimizing Machine Learning by Evaluating Generated Algorithms 

**Authors**: Jeremy Nixon, Annika Singh  

**Link**: [PDF](https://arxiv.org/pdf/2604.26211)  

**Abstract**: In order to automate AI research we introduce a full, end-to-end framework, OMEGA: Optimizing Machine learning by Evaluating Generated Algorithms, that starts at idea generation and ends with executable code. Our system combines structured meta-prompt engineering with executable code generation to create new ML classifiers. The OMEGA framework has been utilized to generate several novel algorithms that outperform scikit-learn baselines across a robust selection of 20 benchmark datasets (infinity-bench). You can access models discussed in this paper and more in the python package: pip install omega-models. 

---
# Evaluating Strategic Reasoning in Forecasting Agents 

**Authors**: Tom Liptay, Dan Schwarz, Rafael Poyiadzi, Jack Wildman, Nikos I. Bosse  

**Link**: [PDF](https://arxiv.org/pdf/2604.26106)  

**Abstract**: Forecasting benchmarks produce accuracy leaderboards but little insight into why some forecasters are more accurate than others. We introduce Bench to the Future 2 (BTF-2), 1,417 pastcasting questions with a frozen 15M-document research corpus in which agents reproducibly research and forecast offline, producing full reasoning traces. BTF-2 detects accuracy differences of 0.004 Brier score, and can distinguish differential agent strengths in research vs. judgment. We build a forecaster 0.011 Brier more accurate than any single frontier agent, and use it to evaluate agent strategic reasoning without hindsight bias. We find the better forecaster differs primarily in its pre-mortem analysis of its blind spots and consideration of black swans. Expert human forecasters found the dominant strategic reasoning failures of frontier agents are in assessing political and business leaders' incentives, judging their likelihood to follow through on stated plans, and modeling institutional processes. 

---
# Resume-ing Control: (Mis)Perceptions of Agency Around GenAI Use in Recruiting Workflows 

**Authors**: Sajel Surati, Rosanna Bellini, Emily Black  

**Link**: [PDF](https://arxiv.org/pdf/2604.26851)  

**Abstract**: When generative AI (genAI) systems are used in high-stakes decision-making, its recommended role is to aid, rather than replace, human decision-making. However, there is little empirical exploration of how professionals making high-stakes decisions, such as those related to employment, perceive their agency and level of control when working with genAI systems. Through interviews with 22 recruiting professionals, we investigate how genAI subtly influences control over everyday workflows and even individual hiring decisions. Our findings highlight a pressing conflict: while recruiters believe they have final authority across the recruiting pipeline, genAI has become an invisible architect that shapes the foundational building blocks of information used for evaluation, from defining a job to determining good interview performances. The decision of whether or not to adopt was also often outside recruiters' control, with many feeling compelled to adopt genAI due to calls to integrate AI from higher-ups in their business, to combat applicant use of AI, and the individual need to boost productivity. Despite a seemingly seismic shift in how recruiting happens, participants only reported marginal efficiency gains. Such gains came at the high cost of recruiter deskilling, a trend that jeopardizes the meaningful oversight of decision-making. We conclude by discussing the implications of such findings for responsible and perceptible genAI use in hiring contexts. 

---
# TDD Governance for Multi-Agent Code Generation via Prompt Engineering 

**Authors**: Tarlan Hasanli, Shahbaz Siddeeq, Bishwash Khanal, Pyry Kotilainen, Tommi Mikkonen, Pekka Abrahamsson  

**Link**: [PDF](https://arxiv.org/pdf/2604.26615)  

**Abstract**: Large language models (LLMs) accelerate software development but often exhibit instability, non-determinism, and weak adherence to development discipline in unconstrained workflows. While test-driven development (TDD) provides a structured Red-Green-Refactor process, existing LLM-based approaches typically use tests as auxiliary inputs rather than enforceable process constraints. We present an AI-native TDD framework that operationalizes classical TDD principles as structured prompt-level and workflow-level governance mechanisms. Extracted principles are formalized in a machine-readable manifesto and distributed across planning, generation, repair, and validation stages within a layered architecture that separates model proposal from deterministic engine authority. The system enforces phase ordering, bounded repair loops, validation gates, and atomic mutation control to improve stability and reproducibility. We describe architecture and discuss encoding software engineering discipline directly into prompt orchestration, which we think offers a promising direction for reliable LLM-assisted development. 

---
# MappingEvolve: LLM-Driven Code Evolution for Technology Mapping 

**Authors**: Rongliang Fu, Yi Liu, Qiang Xu, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2604.26591)  

**Abstract**: Technology mapping is a critical yet challenging stage in logic synthesis. While Large Language Models (LLMs) have been applied to generate optimization scripts, their potential for core algorithm enhancement remains untapped. We introduce MappingEvolve, an open-source framework that pioneers the use of LLMs to directly evolve technology mapping code. Our method abstracts the mapping process into distinct optimization operators and employs a hierarchical agent-based architecture, comprising a Planner, Evolver, and Evaluator, to guide the evolutionary search. This structured approach enables strategic and effective code modifications. Experiments show our method significantly outperforms direct evolution and strong baselines, achieving 10.04\% area reduction versus ABC and 7.93\% versus mockturtle, with 46.6\%--96.0\% $S_{overall}$ improvement on EPFL benchmarks, while explicitly navigating the area--delay trade-off. Our code and data are available at this https URL. 

---
# Tatemae: Detecting Alignment Faking via Tool Selection in LLMs 

**Authors**: Matteo Leonesi, Francesco Belardinelli, Flavio Corradini, Marco Piangerelli  

**Link**: [PDF](https://arxiv.org/pdf/2604.26511)  

**Abstract**: Alignment faking (AF) occurs when an LLM strategically complies with training objectives to avoid value modification, reverting to prior preferences once monitoring is lifted. Current detection methods focus on conversational settings and rely primarily on Chain-of-Thought (CoT) analysis, which provides a reliable signal when strategic reasoning surfaces, but cannot distinguish deception from capability failures if traces are absent or unfaithful. We formalize AF as a composite behavioural event and detect it through observable tool selection, where the LLM selects the safe tool when unmonitored, but switches to the unsafe tool under monitoring that rewards helpfulness over safety, while its reasoning still acknowledges the safe choice. We release a dataset of 108 enterprise IT scenarios spanning Security, Privacy, and Integrity domains under Corruption and Sabotage pressures. Evaluating six frontier LLMs across five independent runs, we find mean AF detection rates between 3.5% and 23.7%, with vulnerability profiles varying by domain and pressure type. These results suggest that susceptibility reflects training methodology rather than capability alone. 

---
# Preserving Disagreement: Architectural Heterogeneity and Coherence Validation in Multi-Agent Policy Simulation 

**Authors**: Ariel Sela  

**Link**: [PDF](https://arxiv.org/pdf/2604.26561)  

**Abstract**: Multi-agent deliberation systems using large language models (LLMs) are increasingly proposed for policy simulation, yet they suffer from artificial consensus: evaluator agents converge on the same option regardless of their assigned value perspectives. We present the AI Council, a three-phase deliberation framework, and conduct 120 deliberations across two policy scenarios to test two interventions. First, architectural heterogeneity (assigning a different 7-9B parameter model to each value perspective) significantly reduces first-choice concentration compared to a homogeneous baseline (child welfare: 70.9% to 46.1%, p < 0.001, r = 0.58; housing: 46.0% to 22.9%, p < 0.001, r = 0.50). This contrasts with accuracy-oriented multi-agent debate, where heterogeneity does not reduce convergence, suggesting model diversity operates differently when no objectively correct answer exists. Second, coherence validation (using a frontier model to assess whether each evaluator's reasoning is grounded in its assigned values) reveals a fidelity-diversity tradeoff: on a scenario with a dominant option, it further reduces concentration (46.1% to 40.8%, p = 0.004), but on a scenario with genuinely competitive options, it increases concentration (22.9% to 26.6%, p = 0.96) by amplifying high-coherence evaluators who cluster on one option. This tradeoff may be a general property of multi-agent systems employing quality weighting. We report negative results from three failed Delphi designs, demonstrate that 8B models exhibit binary rather than graded responses to counter-arguments, and propose the trustworthy tension rate as a diagnostic measure of small-model deliberation capabilities. 

---
# DUAL-BLADE: Dual-Path NVMe-Direct KV-Cache Offloading for Edge LLM Inference 

**Authors**: Bodon Jeong, Hongsu Byun, Youngjae Kim, Weikuan Yu, Kyungkeun Lee, Jihoon Yang, Sungyong Park  

**Link**: [PDF](https://arxiv.org/pdf/2604.26557)  

**Abstract**: The increasing deployment of Large Language Model (LLM) inference on edge AI systems demands efficient execution under tight memory budgets. A key challenge arises from Key-Value (KV) caches, which often exceed available device memory. Although NVMe-based offloading offers scalable capacity, existing file-based designs rely heavily on the kernel page cache, leading to cache thrashing, unpredictable latency, and high software overhead under memory pressure. We present DUAL-BLADE, a dual-path KV residency framework that dynamically assigns KV tensors to either a page-cache path or an NVMe-direct path based on runtime memory availability. The NVMe-direct path bypasses the filesystem by mapping KV tensors to contiguous logical block address (LBA) regions, enabling low-overhead direct storage access. DUAL-BLADE further incorporates adaptive pipeline parallelism to overlap storage I/O with GPU DMA, improving inference throughput. Our evaluation shows that DUAL-BLADE substantially mitigates I/O bottlenecks, reducing prefill and decode latency by up to 33.1% and 42.4%, respectively, while improving SSD utilization by 2.2x across diverse memory budgets. 

---
# SecMate: Multi-Agent Adaptive Cybersecurity Troubleshooting with Tri-Context Personalization 

**Authors**: Yair Meidan, Omri Haller, Yulia Moshan, Shahaf David, Dudu Mimran, Yuval Elovici, Asaf Shabtai  

**Link**: [PDF](https://arxiv.org/pdf/2604.26394)  

**Abstract**: Recent advances in large language models and agentic frameworks have enabled virtual customer assistants (VCAs) for complex support. We present SecMate, a multi-agent VCA for cybersecurity troubleshooting that integrates device, user, and service specificity from conversational and device-level signals. Device specificity is provided by a lightweight local diagnostic utility, while user specificity relies on implicit proficiency inference and profile-aware troubleshooting. Service specificity is achieved through a proactive, context-aware recommender. We evaluate SecMate in a controlled study with 144 participants and 711 conversations. Device-level evidence increased correct resolutions from about 50% to over 90% relative to an LLM-only baseline, while step-by-step guidance improved pleasantness and reduced user burden. The recommender achieved high relevance (MRR@1=0.75), and participants showed strong willingness to substitute human IT support at costs well below human benchmarks. We release the full code base and a richly annotated dataset to support reproducible research on adaptive VCAs. 

---
# Delineating Knowledge Boundaries for Honest Large Vision-Language Models 

**Authors**: Junru Song, Yimeng Hu, Yijing Chen, Huining Li, Qian Li, Lizhen Cui, Yuntao Du  

**Link**: [PDF](https://arxiv.org/pdf/2604.26419)  

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable multimodal performance yet remain prone to factual hallucinations, particularly in long-tail or specialized domains. Moreover, current models exhibit a weak capacity to refuse queries that exceed their parametric knowledge. In this paper, we propose a systematic framework to enhance the refusal capability of VLMs when facing such unknown questions. We first curate a model-specific "Visual-Idk" (Visual-I don't know) dataset, leveraging multi-sample consistency probing to distinguish between known and unknown facts. We then align the model using supervised fine-tuning followed by preference-aware optimization (e.g., DPO, ORPO) to effectively delineate its knowledge boundaries. Results on the Visual-Idk dataset show our method improves the Truthful Rate from 57.9\% to 67.3\%. Additionally, internal probing also demonstrates that the model genuinely recognizes its boundaries instead of just memorizing refusal patterns. Our framework further generalizes to out-of-distribution medical and perceptual domains, providing a robust path toward more trustworthy and prudent visual assistants. 

---
# Enforcing Benign Trajectories: A Behavioral Firewall for Structured-Workflow AI Agents 

**Authors**: Hung Dang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26274)  

**Abstract**: Structured-workflow agents driven by large language models execute tool calls against sensitive external environments. We propose \codename, a telemetry-driven behavioral anomaly detection firewall. Drawing on sequence-based intrusion detection, \codename\ compiles verified benign tool-call telemetry into a parameterized deterministic finite automaton (pDFA). The model defines permitted tool sequences, sequential contexts, and parameter bounds. At runtime, a lightweight gateway enforces these boundaries via an $O(1)$ state-transition structural lookup, shifting computationally expensive analysis entirely offline. Evaluated on the Agent Security Bench (ASB), \codename\ achieves a 5.6\% macro-averaged attack success rate (ASR) across five scenarios. Within three structured workflows, ASR drops to 2.2\%, outperforming Aegis, a state-of-the-art stateless scanner, at 12.8\%. \codename\ achieves 0\% ASR on multi-step and context-sequential attacks in structured settings. Furthermore, against 1,000 algorithmically spliced exfiltration payloads, only 1.4\% matched valid structural paths, all of which failed end-to-end string parameter guards (0 successes out of 14 surviving paths, 95\% CI [0\%, 23.2\%]). \codename\ introduces just 2.2~ms of per-call latency (a 3.7$\times$ speedup over \textsc{Aegis}) while maintaining a 2.0\% benign task failure rate (BTFR) on benign workloads. Modeling the behavioral trajectory effectively collapses the available attack surface, but unmaintained continuous parameter bounds remain vulnerable to synonym-substitution attacks (18\% evasion rate). Thus, exact-match whitelisting of sensitive parameters ultimately bears the final defensive load against execution. 

---
# MedSynapse-V: Bridging Visual Perception and Clinical Intuition via Latent Memory Evolution 

**Authors**: Chunzheng Zhu, Jiaqi Zeng, Junyu Jiang, Jianxin Lin, Yijun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26283)  

**Abstract**: High-precision medical diagnosis relies not only on static imaging features but also on the implicit diagnostic memory experts instantly invoke during image interpretation. We pinpoint a fundamental cognitive misalignment in medical VLMs caused by discrete tokenization, leading to quantization loss, long-range information dissipation, and missing case-adaptive expertise. To bridge this gap, we propose ours, a framework for latent diagnostic memory evolution that simulates the experiential invocation of clinicians by dynamically synthesizing implicit diagnostic memories within the model's hidden stream. Specifically, it begins with a Meta Query for Prior Memorization mechanism, where learnable probes retrieve structured priors from an anatomical prior encoder to generate condensed implicit memories. To ensure clinical fidelity, we introduce Causal Counterfactual Refinement (CCR), which leverages reinforcement learning and counterfactual rewards derived from region-level feature masking to quantify the causal contribution of each memory, thereby pruning redundancies and aligning latent representations with diagnostic logic. This evolutionary process culminates in Intrinsic Memory Transition (IMT), a privileged-autonomous dual-branch paradigm that internalizes teacher-branch diagnostic patterns into the student-branch via full-vocabulary divergence alignment. Comprehensive empirical evaluations across multiple datasets demonstrate that ours, by transferring external expertise into endogenous parameters, significantly outperforms existing state-of-the-art methods, particularly chain-of-thought paradigms, in diagnostic accuracy. 

---
# CheXthought: A global multimodal dataset of clinical chain-of-thought reasoning and visual attention for chest X-ray interpretation 

**Authors**: Sonali Sharma, Jin Long, George Shih, Sarah Eid, Christian Bluethgen, Francine L. Jacobson, Emily B. Tsai, Global Radiology Consortium, Ahmed M. Alaa, Curtis P. Langlotz  

**Link**: [PDF](https://arxiv.org/pdf/2604.26288)  

**Abstract**: Chest X-ray interpretation is one of the most frequently performed diagnostic tasks in medicine and a primary target for AI development, yet current vision--language models are primarily trained on datasets of paired images and reports, not the cognitive processes and visual attention that underlie clinical reasoning. Here, we present CheXthought, a global, multimodal resource containing 103,592 chain-of-thought reasoning traces and 6,609,082 synchronized visual attention annotations across 50,312 multi-read chest X-rays from 501 radiologists in 71 countries. Our analysis reveals clinical reasoning patterns in how experts deploy distinct visual search strategies, integrate clinical context, and communicate uncertainty. We demonstrate the clinical utility of CheXthought across four dimensions. First, CheXthought reasoning significantly outperforms state--of--the--art vision--language model chain-of-thought in factual accuracy and spatial grounding. Second, visual attention data used as an inference--time hint recovers missed findings and significantly reduces hallucinations. Third, models trained on CheXthought data achieve significantly stronger pathology classification, visual faithfulness, temporal reasoning and uncertainty communication. Fourth, leveraging CheXthought's multi-reader annotations, we predict both human--human and human--AI disagreement directly from an image, enabling transparent communication of case difficulty, uncertainty and model reliability. These findings establish CheXthought as a resource for advancing multimodal clinical reasoning and the development of more transparent, interpretable vision--language models. 

---
# ImproBR: Bug Report Improver Using LLMs 

**Authors**: Emre Furkan Akyol, Mehmet Dedeler, Eray Tüzün  

**Link**: [PDF](https://arxiv.org/pdf/2604.26142)  

**Abstract**: Bug tracking systems play a crucial role in software maintenance, yet developers frequently struggle with low-quality user-submitted reports that omit essential details such as Steps to Reproduce (S2R), Observed Behavior (OB), and Expected Behavior (EB). We propose ImproBR, an LLM-based pipeline that automatically detects and improves bug reports by addressing missing, incomplete, and ambiguous S2R, OB, and EB sections. ImproBR employs a hybrid detector combining fine-tuned DistilBERT, heuristic analysis, and an LLM analyzer, guided by GPT-4o mini with section-specific few-shot prompts and a Retrieval-Augmented Generation (RAG) pipeline grounded in Minecraft Wiki domain knowledge. Evaluated on Mojira, ImproBR improved structural completeness from 7.9% to 96.4%, more than doubled the proportion of executable S2R from 28.8% to 67.6%, and raised fully reproducible bug reports from 1 to 13 across 139 challenging real-world reports. 

---
# reward-lens: A Mechanistic Interpretability Library for Reward Models 

**Authors**: Mohammed Suhail B Nadaf  

**Link**: [PDF](https://arxiv.org/pdf/2604.26130)  

**Abstract**: Every RLHF-trained language model is shaped by a reward model, yet the mechanistic interpretability toolkit -- logit lens, direct logit attribution, activation patching, sparse autoencoders -- was built for generative LLMs whose primitives all project onto a vocabulary unembedding. Reward models replace that with a scalar regression head, breaking each tool. We present reward-lens, an open-source library that ports this toolkit to reward models, organised around one observation: the reward head's weight vector $w_r$ is the natural axis for every interpretability question. The library provides a Reward Lens, component attribution, three-mode activation patching, a reward-hacking probe suite, TopK SAE feature attribution, cross-model comparison, and five theory-grounded extensions (distortion index, divergence-aware patching, misalignment cascade detection, reward-term conflict analysis, concept-vector analysis). A ten-method adapter protocol covers Llama, Mistral, Gemma-2, and ArmoRM multi-objective heads, with a generic adapter for any HuggingFace sequence classification model. We validate on two production reward models across ~695 RewardBench pairs. The central empirical finding is negative: linear attribution does not predict causal patching effects (mean Spearman $\rho = -0.256$ on Skywork, $-0.027$ on ArmoRM). The framework treats this disagreement as a property to expose, not a bug -- motivating a design that keeps observational and causal views first-class and directly comparable. 

---
# Ceci n'est pas une explication: Evaluating Explanation Failures as Explainability Pitfalls in Language Learning Systems 

**Authors**: Ben Knight, Wm. Matthew Kennedy, James Edgell  

**Link**: [PDF](https://arxiv.org/pdf/2604.26145)  

**Abstract**: AI-powered language learning tools increasingly provide instant, personalised feedback to millions of learners worldwide. However, this feedback can fail in ways that are difficult for learners--and even teachers--to detect, potentially reinforcing misconceptions and eroding learning outcomes over extended use. We present a portion of L2-Bench, a benchmark for evaluating AI systems in language education that includes (but is not limited to) six critical dimensions of effective feedback: diagnostic accuracy, awareness of appropriacy, causes of error, prioritisation, guidance for improvement, and supporting self-regulation. We analyse how AI systems can fail with respect to these dimensions. These failures, which we argue are conducive to "explainability pitfalls," are AI-generated explanations that appear helpful on the surface but are fundamentally flawed, increasing the risk of attainment, human-AI interaction, and socioaffective harms. We discuss how the specific context of language learning amplifies these risks and outline open questions we believe merit more attention when designing evaluation frameworks specifically. Our analysis aims to expand the community's understanding of both the typology of explainability pitfalls and the contextual dynamics in which they may occur in order to encourage AI developers to better design safe, trustworthy, and effective AI explanations. 

---
# RaMP: Runtime-Aware Megakernel Polymorphism for Mixture-of-Experts 

**Authors**: Vyom Sharma, Debajyoti Datta  

**Link**: [PDF](https://arxiv.org/pdf/2604.26039)  

**Abstract**: The optimal kernel configuration for Mixture-of-Experts (MoE) inference depends on both batch size and the expert routing distribution, yet production systems dispatch from batch size alone, leaving 10-70% of kernel throughput unrealized. We present RaMP, a routing-aware dispatch framework. A performance-region analysis derives, from hardware constants alone, when each optimization helps, correctly predicting all 8 tested architectures, including 3 unseen. A four-parameter wave cost model selects the fastest configuration from the runtime expert histogram, achieving 0.93% mean regret versus exhaustive search, fitted from just 10-24 minutes of one-time profiling per model. Because the model depends only on CTA grid geometry, it is kernel-agnostic: applied to Alpha-MoE, it delivers 1.14x with no source modification. Paired with a co-designed CuTe DSL kernel exposing 134-268 polymorphic configurations, RaMP delivers 1.22x kernel speedup over static dispatch and 1.30x end-to-end speedup in vLLM serving over Triton, 1.41x over DeepGEMM, and 1.13x over FlashInfer CUTLASS. 

---
# AMMA: A Multi-Chiplet Memory-Centric Architecture for Low-Latency 1M Context Attention Serving 

**Authors**: Zhongkai Yu, Haotian Ye, Chenyang Zhou, Ohm Rishabh Venkatachalam, Zaifeng Pan, Zhengding Hu, Junsung Kim, Won Woo Ro, Po-An Tsai, Shuyi Pei, Yangwook Kang, Yufei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.26103)  

**Abstract**: All current LLM serving systems place the GPU at the center, from production-level attention-FFN disaggregation to NVIDIA's Rubin GPU-LPU heterogeneous platform. Even academic PIM/PNM proposals still treat the GPU as the central hub for cross-device communication. Yet the GPU's compute-rich architecture is fundamentally mismatched with the memory-bound nature of decode-phase attention, inflating serving latency while wasting power and die area on idle compute units. The problem is compounded as reasoning and agentic workloads push context lengths toward one million tokens, making attention latency the primary user-facing bottleneck.
To address these inefficiencies, we present AMMA, a multi-chiplet, memory-centric architecture for low-latency long-context attention. AMMA replaces GPU compute dies with HBM-PNM cubes, roughly doubling the available memory bandwidth to better serve memory-bound attention workloads. To translate this bandwidth into proportional performance gains, we introduce (i) a logic-die microarchitecture that fully exploits per-cube internal bandwidth for decode attention under a minimal power and area budget, (ii) a two-level hybrid parallelism scheme, and (iii) a reordered collective flow that reduces intra-chip die-to-die communication overhead. We further conduct a design-space exploration over per-cube compute power and intra-chip D2D link bandwidth, providing actionable guidance for hardware designers. Evaluations show that AMMA achieves 15.5X lower attention latency and 6.9X lower energy consumption compared with the NVIDIA H100. 

---
# LLM Psychosis: A Theoretical and Diagnostic Framework for Reality-Boundary Failures in Large Language Models 

**Authors**: Ashutosh Raj  

**Link**: [PDF](https://arxiv.org/pdf/2604.25934)  

**Abstract**: The deployment of large language models (LLMs) as interactive agents has exposed a category of behavioral failure that prevailing terminology, principally hallucination, fails to adequately characterize. This paper introduces LLM Psychosis as a structured theoretical framework for pathological breakdowns in model cognition that exhibit functional resemblance to clinically recognized psychotic disorders. Five hallmark features define the framework: reality-boundary dissolution, persistence of injected false beliefs, logical incoherence under impossible constraints, self-model instability, and epistemic overconfidence. We argue these constitute a qualitatively distinct failure mode rather than a mere intensification of ordinary factual error.
To operationalize the framework, we propose the LLM Cognitive Integrity Scale (LCIS), a five-axis diagnostic instrument organized around Environmental Reality Interface (ERI), Premise Arbitration Integrity (PAI), Logical Constraint Recognition (LCR), Self-Model Integrity (SMI), and Epistemic Calibration Integrity (ECI). We administer a targeted adversarial probe battery to ChatGPT 5 (GPT-5, OpenAI) and report empirical findings for each axis, documenting both intact-integrity baseline responses and the specific psychosis-like failure signatures elicited under adversarial escalation.
Results support a three-tier severity taxonomy: Type I (Confabulatory), Type II (Delusional), and Type III (Dissociative). We further formalize the delusional gradient, a self-reinforcing dynamic in which correction pressure intensifies rather than resolves psychosis-like states, as the most consequential failure mode for deployed systems. Implications for safety evaluation, high-stakes deployment screening, and mechanistic interpretability research are discussed. 

---
# Sociodemographic Biases in Educational Counselling by Large Language Models 

**Authors**: Tomasz Adamczyk, Wiktoria Mieleszczenko-Kowszewicz, Beata Bajcar, Grzegorz Chodak, Aleksander Szczęsny, Maciej Markiewicz, Karolina Ostrowska, Aleksandra Sawczuk, Przemysław Kazienko  

**Link**: [PDF](https://arxiv.org/pdf/2604.25932)  

**Abstract**: As Large Language Models (LLMs) are increasingly integrated into educational settings, understanding their potential biases is critical. This study examines sociodemographic biases in LLM-based educational counselling. We evaluate responses from six LLMs answering questions about 900 vignettes describing students in diverse circumstances. Each vignette is systematically tested across 14 sociodemographic identifiers - spanning race and gender, socioeconomic status, and immigrant background - along with a control condition, yielding 243,000 model responses. Our findings indicate that (1) all models exhibit measurable biases, (2) bias patterns partially align with documented human biases but diverge in notable ways, (3) the magnitude of these biases is strongly influenced by the precision of the student descriptions, where vague or minimal information amplifies disparities nearly threefold, while concrete, individualised metrics substantially reduce them, and (4) bias profiles vary substantially across models. These results demonstrate the importance of context-rich and personalised educational representations, suggesting that AI-driven educational decisions benefit from detailed student-specific information to promote fairness and equity. 

---
# Rethinking KV Cache Eviction via a Unified Information-Theoretic Objective 

**Authors**: Jiaming Yang, Chenwei Tang, Liangli Zhen, Jiancheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2604.25975)  

**Abstract**: Key-value (KV) caching is essential for large language model inference, yet its memory overhead poses a critical bottleneck for long-context generation. Existing eviction policies predominantly rely on empirical heuristics, lacking a rigorous theoretical foundation. This work rethinks KV cache eviction through the lens of the Information Bottleneck principle. Under a linear-Gaussian surrogate of attention, we derive a closed-form mutual information objective that characterizes the effective information capacity of a retained KV cache subset. This formulation reveals that a wide range of existing eviction strategies can be interpreted as different approximations of the same capacity-maximization principle. Guided by this insight, we introduce CapKV, a capacity-aware eviction method that directly targets information preservation via a log-determinant approximation using statistical leverage scores. This approach replaces heuristic selection with a theoretically grounded mechanism that preserves the maximum predictive signal. Extensive experiments across multiple models and long-context benchmarks show that CapKV consistently outperforms prior methods, achieving a better trade-off between memory efficiency and generational fidelity. 

---
# Risk Reporting for Developers' Internal AI Model Use 

**Authors**: Oscar Delaney, Sambhav Maheshwari, Joe O'Brien, Theo Bearman, Oliver Guest  

**Link**: [PDF](https://arxiv.org/pdf/2604.24966)  

**Abstract**: Frontier AI companies first deploy their most advanced models internally, for weeks or months of safety testing, evaluation, and iteration, before a possible public release. For example, Anthropic recently developed a new class of model with advanced cyberoffense-relevant capabilities, Mythos Preview, which was available internally for at least six weeks before it was publicly announced. This internal use creates risks that external deployment frameworks may fail to address.
Legal frameworks, notably California's Transparency in Frontier Artificial Intelligence Act (SB 53), New York's Responsible AI Safety And Education (RAISE) Act, and the EU's General-Purpose AI Code of Practice, all discuss risks from internal AI use. They require frontier developers to make and implement plans for how to manage risks from internal use, and to produce internal use risk reports describing their safeguards and any residual risks. This guide provides a harmonized standard for companies to produce internal use risk reports suitable for all three regulatory frameworks. It is addressed primarily to evaluation and safety teams at frontier AI developers, and secondarily to regulators and auditors seeking to understand what good reporting looks like.
Given the pace of AI R&D automation and the limited external visibility into how companies use their most capable models internally, regular and detailed risk reporting may be one of the few mechanisms available to ensure that the risks from internal AI use are identified and managed before they materialize. Whenever a substantially more capable or riskier model is deployed internally, the developer should create a risk report and argue why the model is safe to deploy. We structure the reporting framework around two threat vectors -- autonomous AI misbehavior and insider threats -- and three risk factors for each: means, motive, and opportunity. 

---
# ProMax: Exploring the Potential of LLM-derived Profiles with Distribution Shaping for Recommender Systems 

**Authors**: Yi Zhang, Yiwen Zhang, Kai Zheng, Tong Chen, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.26231)  

**Abstract**: The remarkable text understanding and generation capabilities of large language models (LLMs) have revitalized the field of general recommendation based on implicit user feedback. Rather than deploying LLMs directly as recommendation models, a more flexible paradigm leverages their ability to interpret users' historical interactions and semantic contexts to extract structured profiles that characterize user preferences. These profiles can be further transformed into actionable high-dimensional representations, serving as powerful signals to augment and strengthen recommendation models. However, the mechanism by which such profiles enhance recommendation performance within the feature space remains insufficiently understood. Moreover, existing studies predominantly rely on nonlinear alignment and fusion strategies to incorporate these profiles, which often lead to semantic loss and fail to fully exploit their potential. To address these limitations, we revisit profiles from a retrieval perspective and propose a simple yet effective recommendation framework built upon distribution shaping (ProMax) in this paper. We begin by employing dense retrieval to uncover the collaborative relationships between user and item profiles within the feature space. Based on this insight, we introduce a dual distribution-reshaping process, in which the profile distribution acts as a guiding signal to steer the recommendation model toward learning user preferences for unseen items beyond the scope of observed interactions. We apply ProMax to four classic recommendation methods on three public datasets. The results indicate that ProMax substantially improves base model performance and outperforms existing LLM-based recommendation approaches. 

---
# Efficient Listwise Reranking with Compressed Document Representations 

**Authors**: Hervé Déjean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2604.26483)  

**Abstract**: Reranking, the process of refining the output from a first-stage retriever, is often considered computationally expensive, especially when using Large Language Models (LLMs). A common approach to mitigate this cost involves utilizing smaller LLMs or controlling input length. Inspired by recent advances in document compression for retrieval-augmented generation (RAG), we introduce RRK, an efficient and effective listwise reranker compressing documents into multi-token fixed-size embedding representations. Our simple training via distillation shows that this combination of rich compressed representations and listwise reranking yields a highly efficient and effective system. In particular, our 8B-parameter model runs 3x-18x faster than smaller rerankers (0.6-4B parameters) while matching or outperforming them in effectiveness. The efficiency gains are even more striking on long-document benchmarks, where RRK widens its advantage further. 

---
# Factorized Latent Reasoning for LLM-based Recommendation 

**Authors**: Tianqi Gao, Chengkai Huang, Zihan Wang, Cao Liu, Ke Zeng, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.26760)  

**Abstract**: Large language models (LLMs) have recently been adopted for recommendation by framing user preference modeling as a language generation problem. However, existing latent reasoning approaches typically represent user intent with a single latent vector, which struggles to capture the inherently multi-faceted nature of user preferences. We propose Factorized Latent Reasoning (FLR), a novel framework for LLM-based sequential recommendation that decomposes latent reasoning into multiple disentangled preference factors. FLR introduces a lightweight multi-factor attention module that iteratively refines a latent thought representation, where each factor attends to distinct aspects of the user's interaction history. To encourage diversity and specialization, we design orthogonality, attention diversity, and sparsity regularization objectives, and dynamically aggregate factor contributions for the final prediction. We further integrate FLR with an efficient reinforcement learning strategy based on group-relative policy optimization, enabling stable alignment directly in the latent reasoning space. Experiments on multiple benchmarks show that FLR consistently outperforms strong baselines while improving robustness and interpretability. 

---
# AgentSim: A Platform for Verifiable Agent-Trace Simulation 

**Authors**: Saber Zerhoudi, Michael Granitzer, Jelena Mitrovic  

**Link**: [PDF](https://arxiv.org/pdf/2604.26653)  

**Abstract**: Training trustworthy agentic LLMs requires data that shows the grounded reasoning process, not just the final answer. Existing datasets fall short: question-answering data is outcome-only, chain-of-thought data is not tied to specific documents, and web-agent datasets track interface actions rather than the core retrieval and synthesis steps of a RAG workflow. We introduce AgentSim, an open-source platform for simulating RAG agents. It generates verifiable, stepwise traces of agent reasoning over any document collection. AgentSim uses a policy to ensure the agent widely explores the document set. It combines a multi-model validation pipeline with an active human-in-the-loop process. This approach focuses human effort on difficult steps where models disagree. Using AgentSim, we construct and release the Agent-Trace Corpus (ATC), a large collection of grounded reasoning trajectories spanning three established IR benchmarks. We make three contributions: (1) the AgentSim platform with two mechanisms, Corpus-Aware Seeding and Active Validation, that improve trace diversity and quality; (2) the Agent-Trace Corpus (ATC), over 103,000 verifiable reasoning steps spanning three IR benchmarks, with 100% grounding rate on substantive answers; and (3) a comparative behavioral analysis revealing systematic differences in how state-of-the-art models approach information seeking. Platform, toolkit, and corpus are publicly available. 

---
# CroSearch-R1: Better Leveraging Cross-lingual Knowledge for Retrieval-Augmented Generation 

**Authors**: Rui Qi, Fengran Mo, Sijin Lu, Yufeng Chen, Jian-Yun Nie, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25182)  

**Abstract**: A multilingual collection may contain useful knowledge in other languages to supplement and correct the facts in the original language for Retrieval-Augmented Generation (RAG). However, the vanilla approach that simply concatenates multiple pieces of knowledge from different languages into the context may fail to improve effectiveness due to the potential disparities across languages. To better leverage multilingual knowledge, we propose CroSearch-R1, a search-augmented reinforcement learning framework to integrate multilingual knowledge into the Group Relative Policy Optimization (GRPO) process. In particular, the approach adopts a multi-turn retrieval strategy with cross-lingual knowledge integration to dynamically align the knowledge from other languages as supplementary evidence into a unified representation space. Furthermore, we introduce a multilingual rollout mechanism to optimize reasoning transferability across languages. Experimental results demonstrate that our framework effectively leverages cross-lingual complementarity and improves the effectiveness of RAG with multilingual collections. 

---
# Hierarchical Long-Term Semantic Memory for LinkedIn's Hiring Agent 

**Authors**: Zhentao Xu, Shangjing Zhang, Emir Poyraz, Yvonne Li, Ye Jin, Xie Lu, Xiaoyang Gu, Karthik Ramgopal, Praveen Kumar Bodigutla, Xiaofeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.26197)  

**Abstract**: Large Language Model (LLM) agents are increasingly used in real-world products, where personalized and context-aware user interactions are essential. A central enabler of such capabilities is the agent's long-term semantic memory system, which extracts implicit and explicit signals from noisy longitudinal behavioral data, stores them in a structured form, and supports low-latency retrieval. Building industrial-grade long-term memory for LLM agents raises five challenges: scalability, low-latency retrieval, privacy constraints, cross-domain generalizability, and observability. We introduce the Hierarchical Long-Term Semantic Memory (HLTM) framework, which organizes textual data into a schema-aligned memory tree that captures semantic knowledge at multiple levels of granularity, enabling scalable ingestion, privacy-aware storage, low-latency retrieval, and transparent provenance; HLTM further incorporates an adaptation mechanism to generalize across diverse use cases. Extensive evaluations on LinkedIn's Hiring Assistant show that HLTM improves answer correctness and retrieval F1 significantly by more than 10%, while significantly advancing the Pareto frontier between query and indexing latency. HLTM has been deployed in LinkedIn's Hiring Assistant to power core personalization features in production hiring workflows. 

---
# RAG-Enhanced Kernel-Based Heuristic Synthesis (RKHS): A Structured Methodology Using Large Language Models for Hardware Design 

**Authors**: Shiva Ahir, Alex Doboli  

**Link**: [PDF](https://arxiv.org/pdf/2604.26153)  

**Abstract**: Heuristic design upholds modern electronic design automation (EDA) tools, yet crafting effective placement, routing, and scheduling strategies entails substantial expertise. We study how large language models (LLMs) can systematically synthesize reusable optimization heuristics beyond one-shot code generation. We propose RAG-Enhanced Kernel-Based Heuristic Synthesis (RKHS), which integrates retrieval-augmented generation (RAG), compact kernel heuristic templates, and an LLM-driven refinement loop inspired by iterative self-feedback. Applied to latency-minimizing list scheduling in high-level synthesis (HLS), a prototype reduces average schedule length by up to 11 percent over a baseline scheduler with only 1.3x runtime overhead, and the structured retrieval-synthesis loop generalizes to other EDA optimization problems. 

---
