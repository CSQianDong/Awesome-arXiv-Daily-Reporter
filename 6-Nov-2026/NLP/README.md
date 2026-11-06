# Label-Wording Augmentation in a Typed-Decision Encoder: Gains on Unfamiliar Words, Not on Abbreviations or Codes 

**Authors**: Vladimir Beskorovainyi  

**Link**: [PDF](https://arxiv.org/pdf/2611.08439)  

**Abstract**: Typed-decision models score options that the caller supplies at request time, so their accuracy depends on label wording that the model's developer does not control. We compare two releases of a 307M-parameter multilingual cross-encoder, Ru-Jev, trained on thirteen Russian-language tasks; the later release adds an augmentation that presents labels in several surface forms. Holding questions fixed and varying only how eighteen routing labels are worded, we evaluate ten label vocabularies, produced by three separate processes and checked for overlap with the training data, with five candidates per question and, for eight of them, also with eighteen. On held-out vocabularies made of ordinary or borrowed words (English-derived jargon, office slang, department names), the augmented release is more accurate in every setting, by up to 0.24. Eleven of these fourteen gains survive a Holm correction (slang with eighteen candidates and descriptive sentences with five do not), and for jargon on all labels the label-level interval excludes zero in both settings. On abbreviations and opaque codes it gives no usable gain: on abbreviations it is not better, and worse on the clean slice, and on opaque codes both releases stay at or below chance once labels related to training options are removed. In most cells where accuracy rises, the risk-coverage area falls, but with eighteen candidates confident errors rise on jargon and slang. On two datasets neither release saw, accuracy moves by -8.4 and +5.6 points. We also report four evaluation hazards from comparing the two releases. 

---
# Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety 

**Authors**: Fazl Barez, Tung-Yu Wu, Hunar Batra, Gabriele Dominici, Florent Draye, Yanai Elazar, Damiano Fornasiere, Antía García, Abir Harrasse, Riley Harris, Sara Hooker, Michael Lan, Belinda Z. Li, Aaron Maiwald, Narmeen Fatimah Oozeer, Naomi Saphra, Or David Shafran, Adi Simhi, Robert Trager, Shriyash Kaustubh Upadhyay, Ziyu Yao, Yu Zhao, Shay B. Cohen, Mor Geva, Yoshua Bengio, Stella Biderman, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2611.08438)  

**Abstract**: Neural networks are likely to be among the most consequential artefacts in human history. As these systems become more capable and autonomous, their potential to cause societal harm also grows, making AI safety one of the defining challenges of our time. Yet we continue to build increasingly powerful systems without fully understanding how they work, how they will behave in unfamiliar situations, or how to reliably control them. Mechanistic interpretability offers a path towards understanding these systems by studying their internal representations and computations, with the aim of evaluating, predicting, and controlling behaviour. We survey mechanistic interpretability for AI safety in order to answer the question: Does understanding how models work actually make them safer? We place mechanistic interpretability in the broader context of interpretability research and compare its mechanistic evidence with behavioural evaluation. We introduce the foundations of mechanistic interpretability, including features, polysemanticity, transformer computation, and circuits, and survey methods for feature extraction and analysis, circuit discovery and causal verification, explaining intermediate representations in natural language, learning dynamics, and training-data attribution. Beyond surveying existing methods, we propose a unified framework for evaluating mechanistic interpretability claims and characterise six research orientations by their goals, assumptions, and trade-offs. Finally, we examine how mechanistic understanding can support AI safety, including knowledge editing, steering, and monitoring, and assess its limitations, failure modes, open problems, and relationship to behavioural safety measures. We also release a repository of code notebooks with worked examples that reproduce key findings from surveyed methods, allowing readers to build intuition through hands-on experimentation. 

---
# MedGraph: Maintaining Patient State with Graph Memory for Long-Term Medical Dialogue 

**Authors**: Donghua Zhang, Zizheng Wang, Ze Dong, Yiding Feng, Siyu Qian  

**Link**: [PDF](https://arxiv.org/pdf/2611.08433)  

**Abstract**: LLM-based medical assistants that follow a patient across visits need a memory that keeps the patient's clinical state correct as it changes. We examine the errors of five memory systems on longitudinal medical dialogue and find three recurring causes: the retrieved evidence gives an outdated state, includes another person's health facts, or lacks a safety fact the answer depends on. We present MedGraph, a graph memory built for these three cases. Construction extracts dated facts from each session, attributes each fact to the person it describes, and links medication, diagnosis, and measurement records into chains that keep earlier states. Retrieval ranks the patient's chains as units, reads each chain at the time the question asks about, and adds the original dialogue text; a patient profile lists every recorded safety fact with every question. On MedMemoryBench, MedGraph has the highest overall accuracy with both a GPT and a Qwen backbone, 10.5 and 10.4 points above the strongest of five baselines, with efficient memory construction and answering in the GPT setting. 

---
# SpecCTMC: Exact Speculative Simulation of Diffusion Language Model Continuous-Time Markov Chains 

**Authors**: Siva Rajesh Kasa, Yasong Dai, Sumit Negi, Hongdong Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.08362)  

**Abstract**: Masked diffusion language models offer parallel text generation, but independently sampling multiple tokens can disrupt their dependencies and change the output distribution. Exact continuous-time sampling avoids this approximation, yet typically requires one sequential model evaluation per token. We introduce SpecCTMC, a training-free speculative sampler that reduces this sequential bottleneck without a separate draft model. SpecCTMC proposes successive token updates using previously computed model predictions, evaluates their intermediate states in parallel, and applies an exact acceptance-and-correction procedure. We prove that SpecCTMC preserves the joint distribution of event times, generation order, and token values of the underlying continuous-time Markov chain. It therefore introduces neither time-discretization error nor additional factorization error from parallel decoding. Experiments on LLaDA-8B-Instruct and Dream-v0-Instruct-7B across five benchmarks demonstrate 1.73-4.38x fewer sequential model evaluations than direct continuous-time sampling, while preserving the reference sampling distribution. Our code is available at: this https URL 

---
# Cultural Adaptation within Repeated Social Interactions 

**Authors**: Chen Cecilia Liu, Roi Reichart, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2611.08334)  

**Abstract**: Much of human culture is tacit and is typically learned through situated social interaction. As LLMs are deployed as interactive agents across diverse communities, assessing their cultural competence demands moving beyond existing question-answering approaches, shifting from what they know to how they adapt. In this work, we broaden the scope of cultural adaptation in NLP by investigating whether LLMs can adapt through repeated interactions via a sandbox environment. By reframing cultural adaptation as a continuous process of norm assimilation through social interaction, our work demonstrates that memory mechanisms foster longitudinal learning across repeated interactions, facilitate transfer to different social settings, and enable multilingual norm assimilation across five languages. Together, our findings lay the groundwork for continuous, interaction-driven cultural adaptation in LLMs. 

---
# Polarity Classification with Multitask Learning Syntax Augmentation 

**Authors**: Muhammad Imran, Carlos Gómez-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2611.08323)  

**Abstract**: Sentiment Analysis (SA) is a fundamental NLP task concerned with the identification and categorization of subjective information in text. While lexicon-based methods offer transparency, they often struggle with linguistic phenomena such as negation and intensification. Syntactic parsing has been integrated into SA pipelines to address this, but conventional parsers introduce significant computational overhead. Recent work shows that reformulating dependency parsing as a Sequence Labeling (SEL) task can improve efficiency. We propose and compare two strategies for integrating syntactic knowledge into a neural sentiment analysis model under unified Multi-Task Learning (MTL). The first, Standard-MTL, jointly learns dependency parsing (via sequence labeling) and sentiment polarity classification using a shared encoder with two task-specific heads. The second, Contrastive-MTL, provides implicit syntactic guidance through a contrastive auxiliary objective and a syntax-biased attention pooling mechanism, without requiring explicit label prediction. Both rely on syntactic annotations during training and eliminate need for hand-crafted compositional rules or external parsing at inference. We evaluate eleven datasets spanning seven languages---the MARC corpus (English, Spanish, French, German) and seven datasets from SemEval-2022 Task 10 (English, Spanish, Norwegian, Basque, Catalan)---using four established SEL encodings. With the relative encoding, Contrastive-MTL outperforms a syntax-naive baseline on 10 of 11 datasets and surpasses the external rule-based pipeline (SELSP) that parses at inference on all six datasets where the latter applies, while being 1.5--9$\times$ faster at evaluation. Standard-MTL is more dataset- and encoding-dependent. These results show implicit syntactic supervision during training can match or exceed explicit parsing at inference without computational overhead. 

---
# LLM-Assisted Label Noise Correction for EEG-Based Sleep Stage Classification 

**Authors**: Elmira Salari, Shruti Kshirsagar  

**Link**: [PDF](https://arxiv.org/pdf/2611.08311)  

**Abstract**: Automatic sleep stage classification from electroencephalography (EEG) signals is a critical task in clinical sleep medicine. While deep learning models have shown promising results in this domain, their performance is fundamentally constrained by the quality of training labels, which human experts manually annotate and that are subject to inter-rater disagreement and systematic labeling errors. In this paper, we propose a novel LLM-assisted label-noise filtering pipeline that operates exclusively during training to identify and remove unreliable sleep-stage annotations before retraining the classifier. Specifically, TinySleepNet is first trained on the original dataset to extract per-epoch SoftMax confidence scores, then low-confidence epochs are submitted to LLaMA 3.1 along with their EEG spectral features and surrounding sequence context. The LLM evaluates each epoch against AASM sleep-scoring guidelines and assigns a verdict of RELIABLE or SUSPICIOUS. Suspicious epochs are removed from the training set, and TinySleepNet is retrained on the cleaned data. Evaluated on the Sleep-EDF Expanded dataset using 20-fold cross-validation, our method improves overall accuracy from 81.8% to 84.6% and Macro F1-score from 75.6% to 77.4%, with particularly notable gains in REM detection. Crucially, because the LLM operates only during training, the lightweight TinySleepNet architecture (1.3M parameters) is preserved at inference, making the proposed pipeline suitable for real-time, wearable sleep monitoring applications. 

---
# Measured Coupling for Parallel Decoding in Diffusion Language Models 

**Authors**: Tal Oved, Lior Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2611.08307)  

**Abstract**: Diffusion language models can write many tokens per step, but must choose which tokens are safe to commit together. Mean-field decoding penalizes pairs of positions whose predictions look alike. We replace this proxy with the coupling itself: how much committing one token changes the model's confidence at another. Cheap probes inside the main forward pass measure it, with no training or extra model, and it plugs into the same mean-field rule. On three models and four benchmarks, our method needs far fewer steps than mean-field decoding and is faster in most settings, also with a key-value cache, at an average accuracy cost of 1.1 points. Shuffling the couplings or replacing them with a uniform bonus loses this gain. 

---
# Multimodal-SEA: Benchmarking AI Agent Security under Multimodal Social Engineering Attacks 

**Authors**: Zhitao He, Chi Him Zhang, Chengru Han, Yi R Fung  

**Link**: [PDF](https://arxiv.org/pdf/2611.08281)  

**Abstract**: Personal AI agents face social engineering attacks supported by forged multimodal content, but existing evaluations mainly focus on text-based dialogue, leaving their intrinsic security under such attacks insufficiently studied. We introduce Multimodal-SEA, an interactive benchmark comprising 608 expert-designed tasks grounded in real-world fraud cases, covering 14 domains and 152 attack scenarios with prepared multimodal content. The tasks span four difficulty levels and reflect the distinct social and cultural contexts of Chinese and English environments. Agents inspect content, verify requests, and execute actions in simulated applications, with attack outcomes grounded in verifiable tool execution logs. To enable controlled red teaming of personal agents, we design a multimodal attack harness that integrates task information, attack progression control, and multimodal content construction to evaluate their intrinsic security against adaptive social engineering attacks in simulated environments. Evaluating seven widely used multimodal models across six dimensions reveals that all are vulnerable, with attack success rates of 7.1% to 34.5%. Adding multimodal content increases attack success from 19.9% to 29.2%, despite increased proactive verification coverage. Stronger safety instructions reduce attack success without improving recognition of deceptive multimodal content, while lower attack success rates for some models accompany excessive refusal of legitimate requests. 

---
# Terminal-style Mathematical Modeling Bench: The First Agentic and Verifiable Benchmark for Mathematical Modeling 

**Authors**: Xiaojun Wu, Cehao Yang, Xuhui Jiang, Chengjin Xu, Jia Li, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2611.08237)  

**Abstract**: Mathematical modeling turns an open-ended real-world problem into assumptions, a model, a solution, and a defended recommendation, and its answers matter only if they are both close to a credible reference and physically feasible; existing benchmarks either score solver fragments or grade reports with an LLM judge, leaving those numbers unchecked. We introduce TB-MathModel (the Terminal-style Mathematical Modeling Bench), the first benchmark to make complete contest-style modeling problems agentic, verifiable, and terminal at once, over 18 real CUMCM and MCM problems (2023--2025). Each task ships the verbatim statement, official attachments, and figures in a containerized terminal; the agent returns one JSON artifact, scored on the numeric core of the final question against a reproduction of the Outstanding-award paper. Two design choices distinguish it: direction-aware, one-sided scoring, which never penalises beating the anchor, and two-layer feasibility verification---a programmatic gate over six decidable invariant forms plus 624 Lean~4 obligations---with a hard gate that zeroes infeasible answers. Each task is scored against a panel of human outstanding papers rather than one, so a model must be close to many accepted answers; across seven multimodal entries (three seeds each) the strongest reach a multi-reference O-Eval of 64.58\%, while the \$2.54 open model holds 59.81\% at $1/100$ the cost. TB-MathModel turns the numeric core of modeling into a reproducible, machine-checkable signal for measuring long-horizon modeling competence and for collecting agent trajectories. The source code is available at this https URL. 

---
# From Twisted Facts to Rogue Agents: Can LLMs Reliably Detect Multi-Dimensional Hallucinations? 

**Authors**: Hanyu Qiu, Ze Liu, Zhe Zhao, Kuang Xiong, Blaze Chen, Zhenxiang Yan, Renfen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.08231)  

**Abstract**: Large language models (LLMs) are widely used for information access, content production, and agent-based task execution, but their outputs remain vulnerable to subtle, high-risk hallucinations. To systematically evaluate and advance post-generation hallucination detection, we introduce HaluCheck, a large-scale, bilingual benchmark comprising over 42k instances spanning both factual hallucinations in text and faithfulness hallucinations in agent trajectories. HaluCheck features fine-grained, multi-level annotations for detection, localization, and correction, covering different tasks and perturbation degrees. Experiments on fourteen LLMs reveal that, while models show basic sample-level detection ability, they struggle with fine-grained localization, correction, and agent hallucination detection. Further analysis demonstrates that thinking, web search, and supervised fine-tuning each offer partial improvements, but none fully solve the problem. HaluCheck thus provides a solid basis for benchmarking and diagnosing LLM reliability across text and agent scenarios. 

---
# Agent-R$^2$: Scaling Agents for Realistic Productivity Work via Real-Source Data Synthesis 

**Authors**: Teng Pan, Zixuan Wang, Yuchen Yan, Qifei Wu, Yao Chen, Yingqi Qu, Ruiqing Zhang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.08219)  

**Abstract**: Large language models have made rapid progress in agentic reasoning and tool use, raising expectations for agents that can take over realistic productivity work from humans. However, current agents still struggle to finish such tasks, where success requires following long-horizon professional workflows, understanding heterogeneous file contexts, and producing usable deliverables under fine-grained requirements. A key bottleneck is the lack of scalable post-training data that reflects real work intent, authentic file environments, and deliverable-level evaluation. To close this gap, we introduce Agent-R$^2$, a $\textbf{r}$eal-source data synthesis framework that scales training data for $\textbf{r}$ealistic productivity work. Instead of relying on manually specified workflows or LLM-generated files, Agent-R$^2$ grounds synthesis throughout the pipeline in real occupational seeds, corpus-derived workflow evidence, and authentic web-sourced files. From 10k web-corpus candidates, Agent-R$^2$ distills 4k task instances that pair real multi-format attachments with fine-grained, deliverable-level rubrics. On this data, we train Agent-R$^2$-4B via supervised fine-tuning and rubric-based reinforcement learning, improving both end-to-end productivity performance and file-grounded reasoning. Trained on only 4k instances, Agent-R$^2$-4B surpasses Qwen3-235B-A22B-Thinking-2507 on both GDPval (51.0$\%$ vs. 48.0$\%$) and OfficeQA (14.8$\%$ vs. 13.8$\%$). Further analysis shows that Agent-R$^2$ mitigates shortcut behavior, where agents produce superficially plausible deliverables before sufficiently grounding, constructing, or verifying them, while promoting sustained exploration. 

---
# Knowing When to Teach: Learning from Teachable Prefixes in Multilingual Reasoning Distillation 

**Authors**: Jaewon Cho, Minseo Yoon, Jinyoung Park, Jinwoo Seo, Jaewon Chu, Alice Oh, Hyunwoo J. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.08179)  

**Abstract**: On-Policy Self-Distillation (OPSD) improves reasoning by training a student on its own reasoning, using the same model conditioned on reference solutions as a teacher. Applying OPSD to multilingual reasoning requires the teacher to provide useful feedback across languages, even when the student makes mistakes in intermediate steps. We observe a phenomenon we call Teacher Drift: as these mistakes accumulate, the teacher may follow the student's incorrect steps instead of correcting them. Our analysis shows that this shift tends to occur earlier in lower-resource languages. To account for these differences across languages, we propose Teachable On-Policy Self-Distillation (TOPSD), which determines where and whether teacher supervision should be applied to each reasoning trajectory. TOPSD uses teacher-student likelihood to estimate where drift begins and select a prefix for supervision. It then verifies whether the teacher can recover the correct answer from that prefix, applying distillation only to prefixes that pass verification. Experiments on multilingual mathematical reasoning across ten languages and three model scales show that TOPSD consistently outperforms conventional OPSD, particularly in low- and medium-resource languages. 

---
# Beyond K/V Caches: Cross-Model State Handoff in Hybrid Linear-Attention LLMs 

**Authors**: Jiacheng Cai, Tao Zhang, Siqi Liang, Yun Dai, Ziqian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2611.08133)  

**Abstract**: Production deployments often use different models to balance cost and quality through cascading, routing, and switching models during long conversations. Context accumulates across turns. Each switch requires the target to repeat prefill over this context. Existing methods transfer full-attention (FA) K/V caches across models. Hybrid linear-attention models also retain recurrent states (REC) and convolution histories (Conv). The interaction between Conv and REC complicates cross-model state transfer. Conv perturbations can change keys and values in recurrent layers and alter REC updates. Conv histories are overwritten, but changes in REC can still affect later outputs. More accurate REC reconstruction need not improve continuation. Building on this, we propose Hybrid State Transfer (HST) for all three state types. It uses structured REC maps and preserves history order during Conv alignment. We calibrate the adapters using the frozen target's outputs during continuation. FA cache map updates are merged offline. Across five benchmarks, Qwen3.5 4B$\rightarrow$27B and 9B$\rightarrow$27B achieve mean task retentions of 89.88% and 94.11%, respectively. With source caches already available, separate timing on synthetic 16K inputs yields handoff body speedups of $21.66\times$ and $21.91\times$. 

---
# Linguistic boundaries emerge from pressure-constrained processing dynamics 

**Authors**: Rui He, Philipp Homan, Wolfram Hinzen  

**Link**: [PDF](https://arxiv.org/pdf/2611.08085)  

**Abstract**: Human cognition organizes continuous experience into bounded units while maintaining continuity across them. Linguistic elements are also organized into bounded structural units such as words and phrases, with sentence marking the final bound of a grammatical construction process. How do such boundaries emerge? We propose boundaries emerge from interacting demands to maintain unresolved dependencies, integrate completed relations, and update semantic context, with structural completion allowing but not determining closure. First, in a generative simulation without predefined boundary positions or constituent labels, these demands produced sentence-like segment-length distributions and distinct characteristic timescales, with structural completion allowing but not determining closure. Near boundaries, unresolved dependency load declined, integration cost peaked, and accumulated pressure reset. Then, we tested whether same demands predicted human listeners' boundary choices during comprehension. Listeners were more likely to mark boundaries with fewer unresolved dependencies, higher integration cost and semantic mismatch, and elevated preceding memory load, beyond syntactic and prosodic cues. Third, in naturalistic self-paced reading, sentence-final slowing coincided with declining memory load and increasing integration cost and semantic mismatch, broadly consistent with the simulation. Processing same narratives with a large language model revealed increased local intrinsic dimensionality and straighter hidden-state trajectories at sentence endings, followed by larger representational updates. Boundary explained additional variation in reading time and hidden-state geometry beyond processing demands, but considerably less in next-token surprisal. Together, these findings suggest linguistic boundaries emerge as discrete manifestations of transitions in continuous processing dynamics. 

---
# Automatic keyphrase extraction and annotation: modern theoretical approaches and practical solutions for text and speech 

**Authors**: Daria Guseva, Olga Mitrofanova  

**Link**: [PDF](https://arxiv.org/pdf/2611.08006)  

**Abstract**: The exponential growth of textual and audiovisual information has made the task of automatic keyphrase extraction (KE) increasingly significant. This article provides a comprehensive analysis of contemporary theoretical approaches and practical solutions for KE across both text and speech modalities. The primary contribution of this work is its systematic synthesis of these often-disparate research strands into a unified analytical framework, highlighting the evolution of the field from statistical methods towards large language models (LLMs) and end-to-end speech processing. We examine the stages of KE, the characteristics of keyphrases in written and spoken language, and terminological nuances. Various methods for automatic KE are discussed and analyzed in detail: statistical, hybrid, machine learning-based, and structural. The review dedicates substantial attention to emerging paradigms, including keyphrase generation using LLMs, and provides a detailed overview of methodologies and challenges in automatic corpus annotation. Furthermore, we specifically analyze current directions and inherent difficulties in KE for spoken language, comparing transcript based and end-to-end acoustic approaches. This synthesis leads us to conclude that the field is moving towards a more integrated, context-aware paradigm. Future progress will depend on addressing key challenges such as data scarcity for low-resource languages, effective multimodal fusion, and the nuanced evaluation of generative KE systems. 

---
# Compression Through a Context Lens: Evaluating Adaptation, Quantization, and Distillation in Arabic Legal LLMs 

**Authors**: Md. Rafiul Biswas, George Mikros, Wajdi Zaghouani  

**Link**: [PDF](https://arxiv.org/pdf/2611.07988)  

**Abstract**: Model compression can facilitate the deployment of large language models in resource-constrained legal settings, but its effects on domain-specific performance may depend on the evaluation protocol. We investigate this issue using ArabLegalEval, a Saudi Arabic legal benchmark, under both with-context and no-context evaluation. The former provides supporting legal passages, while the latter assesses performance without external context. Removing the context produces accuracy gaps of up to 27 percentage points for the base model. We examine three stages: domain adaptation, quantization, and knowledge distillation across FANAR-9B, JAIS-8B, and Qwen2.5-7B. Four-bit quantization reduces model memory by 64-71%, with changes in no-context MCQ accuracy ranging from -3.2 to +1.6 percentage points. Distillation produces more variable results, with accuracy generally declining as student size decreases. A Qwen2.5-7B student distilled from FANAR achieves 76.6% no-context accuracy, compared with 77.2% for its teacher. Our findings suggest that no-context evaluation provides a useful complement to context-supported testing when assessing the effects of adaptation and compression on Arabic legal language models. 

---
# Beyond the Keyword: A Human-Annotated Benchmark for Arabic Anxiety-Related Discourse 

**Authors**: Shimaa Ibrahim, Mabrouka Bessghaier, Md. Rafiul Biswas, Insaf Elouard, Wajdi Zaghouani  

**Link**: [PDF](https://arxiv.org/pdf/2611.07949)  

**Abstract**: Keyword retrieval returns expressions of personal anxiety together with many other uses of anxiety-related language. We present a benchmark of 1,000 Arabic Twitter/X posts retrieved with the keyword qalaq ("anxiety/worry") and annotated for six discourse functions. Two native Arabic-speaking annotators labeled every post independently, reaching 51.1% raw agreement (Cohen's $\kappa = 0.378$). We then reviewed all 489 disagreements and all 511 agreements under the same guidelines. The review changed 120 of the agreed labels (23.5%), which shows that agreements need auditing as well as disagreements. The final labels form a consensus reference; on a random sample of 200 posts, an additional annotator agreed with it on 60.5% of posts ($\kappa = 0.517$). Against this reference, Fanar-C-2-27B reaches 79.8% accuracy and macro-F1 0.784 ($\kappa = 0.749$). Its errors concentrate in the residual category (41.1% of errors; precision 0.905, recall 0.447), whose definition is internally inconsistent in the executed prompt, so the low recall cannot be attributed to the model alone. Sociopolitical Anxiety (27.1%) outnumbers Personal Emotional Expression (22.0%); these are assignment proportions in the sample, not estimates of prevalence or retrieval precision. We release the guidelines, a residual-subtype scheme, a fixed data split, label provenance, and evaluation code. 

---
# CardioLM: A multilingual suite of small language models for the cardiology domain 

**Authors**: Manuela-Daniela Danu, Miguel Rodríguez-Ortega, Alberto Becerra-Tome, Pablo Arancibia, Fernando Gallego-Donoso, Lorenzo Pratesi, Xi Bai, Andrei-Damian Dinescu, Salvador Lima-López, Andrea Guala, Saskia Haitjema, Jonas Mack, Michele Miranda, Wesam Al Nabki, Jan Rodríguez-Miret, Vasile George Marica, Vladimír Kincl, Kateřina Dostálová, Michael Andrej, Alicia Ujil, Corinne Allaart, Machteld Boonstra, Malin Overmars, Anoop Shah, Giovanna Ceroni, Jonathan Kossoff, Domenico D'Amario, Renzo Laborante, Antonio Iaconelli, Isabella Fumarulo, Jacopo Lenkowicz, Livia Lilli, Emma Weiss, Alex Scafa, Felix Lindberg, Milou Gamage, Eulàlia Farré-Maduell, Sébastien Bratières, Martin Krallinger, Bram van Es  

**Link**: [PDF](https://arxiv.org/pdf/2611.07940)  

**Abstract**: Background: Cardiovascular care generates large volumes of clinical free-text documentation in local languages. However, clinical natural language processing (NLP) resources remain concentrated in English, and no existing benchmark is cardiology-specific, multilingual, and broad in entity coverage.
Methods: We present two contributions. The first is CardioCCC, a corpus of 508 cardiology case reports originally published in Spanish and extended to Czech, Dutch, English, Italian, Romanian, and Swedish through machine translation, annotation projection, and clinician review, with annotations covering diseases, symptoms, procedures, and medications. The second is CardioLM, comprising CardioBERTa, a set of language-specific encoders obtained by continued pretraining on cardiology and biomedical corpora, and CardioNER, a set of named entity recognition (NER) models fine-tuned on CardioCCC.
Results: Cardiology-specific pretraining reduced pseudo-perplexity by up to a factor of eight, yet changed strict span-level F1 by at most 0.04, with benefits concentrated in languages whose backbones lacked biomedical coverage. The jointly trained multilingual XLM-RoBERTa-large models achieved the best strict F1 in nearly every language and entity type, exceeding the strongest monolingual system by up to 0.13 F1. A large share of residual errors concerned boundary detection: relaxed matching recovered 0.10-0.23 F1 for disease, procedure, and symptom, but only 0.01-0.07 for medication. Clinicians rated CardioNER well above a rule- and dictionary-based baseline, with the mean rating across entity types rising from 26% to 73%.
Conclusions: Specialty-specific pretraining is not the decisive factor for downstream entity recognition, whereas multilingual joint training yields the strongest results in nearly all settings and a simpler deployment path for multi-country clinical information extraction. 

---
# AraEngage: Interpretable Linguistic Correlates of Engagement in Arabic Facebook News 

**Authors**: Shimaa Ibrahim, Nour Elhouda Dalloua, Md. Rafiul Biswas, Wajdi Zaghouani  

**Link**: [PDF](https://arxiv.org/pdf/2611.07926)  

**Abstract**: Engagement features developed for English are routinely applied to Arabic, but unvocalized Arabic changes what a lexical feature measures. We introduce AraEngage, a query-defined sample of 10,000 public Arabic Facebook news posts from 2,451 pages in 57 countries (2023-2024) that pairs engagement counts with interpretable linguistic features. We use it to examine which interpretable Arabic indicators are associated with follower-normalized engagement, and how far those associations can be trusted given publisher context, the definition of engagement, and lexical ambiguity. Religious expressions, message length, sentence length, and lexical density are the leading correlates in a Random Forest model ($R^2$ = 0.38 on a post-level split) interpreted with SHAP. Unlike English news, positive sentiment is associated with higher engagement in this Arabic news sample. Message length shows a length paradox: unadjusted raw engagement is highest for long posts, whereas the model's attributions for follower-normalized engagement favor short posts. A linguistic audit identifies why standard lexicon features are fragile in unvocalized Arabic: imperative lists contain homographs such as the form read as "express!" that is also the frequent preposition "via", hedge lists conflate the possibility marker qad + imperfect ("may") with the emphatic qad + perfect ("has indeed"), and religious lists mix devotional formulas with place names in political news. These are plausible sources of measurement error that may attenuate the near-zero associations of imperatives and hedges. All findings are observational associations with operationalized indicators. We release the derived linguistic features, engagement counts, refined lexicons, and analysis code; raw posts are not redistributed, in line with CrowdTangle terms. 

---
# Do Recommenders Need to Think in Words? Latent Reasoning for Efficient LLM-Based Recommendation 

**Authors**: Hongyang Su, Beibei Kong, Lei Cheng, Chengxiang Zhuo, Zang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.07866)  

**Abstract**: LLM-based recommender systems increasingly generate chain-of-thought (CoT) text before each item prediction to activate the LLM's reasoning capacity, costing tens to hundreds of extra tokens per recommendation at inference. We ask whether inference-time reasoning artifacts (textual or latent) can be replaced by training-time supervision alone, without loss of accuracy. We propose LatentRec, a framework for LLM-based generative recommendation that replaces inference-time reasoning artifacts with training-time per-step head supervision. LatentRec has three components. A 2.5M-parameter Continuous Thought Module iterates K latent reasoning steps over the LLM's hidden state during training. Soft Token Selection bridges the ~14x norm gap between hidden states and input embeddings. A Per-step Reasoning Loss (PRL) supervises every intermediate state through the LLM's prediction head against the first token of the target Semantic ID. The trained LLM retains this accuracy without the module at inference, so the module can be discarded at deployment. On three Amazon benchmarks, LatentRec improves HR@10 by 4.0-5.0% (relative) over a CoT-distilled baseline at 18.3x lower latency (batch=1). Per-step training supervision consistently improves accuracy across three datasets, paid once at training rather than at every recommendation. 

---
# CORE: Coarse-to-Fine Search with Context Management for Deep Research 

**Authors**: Hoang Tran Vuong, Ngoc Quan Pham, Hung Bui, Fatih Porikli, Dat Quoc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.07859)  

**Abstract**: Deep research agents answer complex questions through repeated search. They rely on a harness that runs their tool calls and returns evidence to their context. This creates a trade-off: Search works best over long passages, which give enough context for reliable semantic matching. But the returned evidence should be short, because every observation stays in the agent's context for the rest of the trajectory. We introduce CORE, a training-free harness that combines coarse-to-fine search with context management. CORE retrieves passages with a dense retriever. It then reuses the token-level similarities from a late-interaction reranker to return only the most relevant snippet from each passage. Context management lets the agent expand the context around sources it has already retrieved and suppresses repeated snippets. In controlled closed-search comparisons with the Tongyi DeepResearch and OpenResearcher harnesses, CORE achieves state-of-the-art answer accuracy and evidence recall across three agent models and three benchmarks (BrowseComp-Plus, BrowseComp-Wikipedia and BrowseComp-Plus-CM), on corpora containing up to 553M documents. 

---
# Towards Foundation Decision Models: Drex 

**Authors**: Nischay Dhankhar, Alisher Amantay, Zhanibek Datbayev, Darkhan Akhmetov, Yash Shah, Nurmukhamed Ubaidullayev, Raman Saparkhan, Dos Baha, Abulhair Saparov  

**Link**: [PDF](https://arxiv.org/pdf/2611.07854)  

**Abstract**: Autoregressive approaches with large language models, such as long chain-of-thought reasoning, have become the default for new AI applications because they reach higher accuracy on a wide range of tasks. In production, however, they give inconsistent answers, require parsing of outputs, and add cost and latency. Decision models address this: given a question and a set of options supplied by the user, they return a probability distribution over the options without autoregressive decoding. We present Drex, a family of four open decision models with 8B (DLM), 9B (v1.5), 35B with 3B active (v2.0), and 27B (Max) parameters. Each request consists of a state (a document, record, or agent context, optionally with images) and questions about it, each asking for a choice among options, a yes/no judgment, or a score on an ordered scale. Training combines supervised fine-tuning with reinforcement learning from adjusted feedback (RLAF), plus vision tuning for Max. Ablations show that RLAF improves accuracy and that consistency losses improve robustness to option order. On Decision Index 0.4.1, Max sets a new state of the art among the decision models we compare (67.52 on the public suite vs. 60.70 for Jev 1.14.0). On Decision Bench v4, it is statistically tied with d3-turbo and GPT-6 Sol and trails Claude Opus 5.5 by 1.6 points, and it has the highest accuracy among decision models on DecideBench. Max answers a typical request in 183 ms, against about 2.0 s for GPT-6 Sol and 3.5 s for Claude Opus 5.5, and its outputs are stable under option reordering. All four models are open source: this https URL 

---
# Axis-LoRA: Probing and Adapting Embedding Models for Personal-Memory Retrieval 

**Authors**: Dohoon Kim, Junho Park, Hyunwoong Bae, Seyoon Kim, Taesup Moon  

**Link**: [PDF](https://arxiv.org/pdf/2611.07820)  

**Abstract**: As LLMs are increasingly asked questions about a user's personal experiences, retrieval must distinguish memories by time, place, person, and event. Current personal-memory pipelines rely on frozen dense embedding models to embed memories for retrieval, but we find that these models do not capture the four axes well. Swapping a memory's date or place barely shifts its embedding, and a trained projection on top of the frozen embedding does not recover the missing signal. The problem lives inside the model, where each axis is encoded by a distinct attention-head pattern. Guided by these findings, we propose Axis-LoRA, a per-axis adapter that places four LoRA branches inside the embedding model and combines them via a per-query gate. Across three standard embedding models evaluated on LongMemEval, ATM-Bench, and LoCoMo, Axis-LoRA improves 4-benchmark mean R@10 by $+1.28$ to $+1.56$~pp while training only a small fraction of the base parameters on a small axis-counterfactual training set. Code is available at this https URL. 

---
# Different Corpora, Different Skills? A Paired Study of Small Language Models 

**Authors**: Edoardo Vaira, Georgios Chavales, Pierre Nugues  

**Link**: [PDF](https://arxiv.org/pdf/2611.07812)  

**Abstract**: Small language models are often trained under tight data budgets, where the choice of pretraining corpus may matter as much as model size or training recipe. We compare two contrasting corpora at matched 10-million- and 37-million-word budgets: the 1911 Encyclopaedia Britannica and the BabyLM 2026 strict pretraining corpus. For each corpus, we train teacher models, distilled 58-million-parameter Baby-Llama students, and no-distillation student ablations, using the same model families, training recipes, and evaluation protocol across corpora. Across BLiMP, BLiMP Supplement, EWoK, Entity Tracking, and COMPS, the corpora produce different skill profiles rather than a single quality ranking. BabyLM 2026 is consistently stronger on linguistic acceptability benchmarks, while Britannica is often stronger on Entity Tracking, a synthetic state-tracking task. The contrast appears across teacher and student models and at both budgets, with the cleanest cross-seed replication in the distilled student comparison. Targeted controls show that the central contrast persists after matching training exposure and at a longer diagnostic context; direct teacher ensembling also does not reproduce the 10M Britannica student's Entity Tracking behavior. Entity Tracking magnitudes nevertheless remain sensitive to training dynamics. For this corpus pair, corpus choice can therefore produce different relative skill profiles at small scale, motivating multi-benchmark evaluation when comparing pretraining data. 

---
# Decodability Is Not Validity: Evaluating Nuisance Sensitivity in Language Model Concept Probes 

**Authors**: Ayal Klein  

**Link**: [PDF](https://arxiv.org/pdf/2611.07805)  

**Abstract**: Two concept probes can be equally accurate at predicting a concept from a language model's hidden states while their scores contain different mixtures of the target concept and a correlated nuisance attribute. In four target--nuisance settings (two aspect sentiments in CEBaB; occupation and gender in BiasBios; Amazon review sentiment with negation, and with first-person framing), two LLMs, and a population of concept extractors, we compare decodability with discriminant evidence: the score's dependence on the nuisance at fixed target, and its response to input edits intended to change only the nuisance, filtered by two LLM judges and compared against placebo edits. Under correlated training data, extractors that are statistically equivalent in held-out AUROC differ up to several-fold in nuisance sensitivity in three of the four settings, and decodability is not a reliable proxy for it: their rank correlation changes sign across settings. The effect follows the probe's training data, not the nuisance's natural association with the target: first-person framing, unrelated to sentiment in natural reviews, is absorbed at least as much as negation, which co-occurs with negative sentiment, and decorrelated training removes the systematic effect. Selecting extractors with nuisance-aware development diagnostics, which need nuisance labels but no counterfactuals, lowers counterfactual nuisance sensitivity wherever the default choice is sensitive, at a small accuracy cost. Concept measurements should be validated, not just decoded. 

---
# Dataset Collection and Fine-Tuning of NLLB for German--English Translation 

**Authors**: SK Deeraj, Yash Ingle, Pruthwik Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2611.07797)  

**Abstract**: This paper describes the curation of a high-quality parallel corpus from OPUS Books and the fine-tuning of Meta's NLLB-200 models for German--English translation. We detail a dataset extraction pipeline that filters 45k raw OPUS sentence pairs to 25k high-quality examples through encoding validation, length filtering, language identification, deduplication, and quality scoring. We present a reproducible fine-tuning methodology optimized for consumer GPUs using mixed precision, gradient accumulation, and gradient checkpointing. Our fine-tuned NLLB-600M model achieves statistically significant improvements (+3.4 BLEU, +2.6 chrF++, -3.2 TER) over the base model on literary and technical German-English translation. We demonstrate practical deployment by integrating the fine-tuned model into LinguaHaru, an existing open-source document translation system, achieving format-preserving translation across PDF and Office documents. All dataset preparation scripts, training code, and evaluation tools are provided for reproducibility. The fine-tuned model is available at HuggingFace. 

---
# Bridging Structure and Semantics: Knowledge Graph Verification via Triple-to-Text Alignment 

**Authors**: Ziang Cui  

**Link**: [PDF](https://arxiv.org/pdf/2611.07780)  

**Abstract**: Semantic accuracy is an important dimension of knowledge graph quality. Graph-based scoring methods exploit relational structure, while textual representations allow classifiers to draw on knowledge encoded by pre-trained language models (PLMs). We present GenVer, a generative verifier that scores triples using both their serialized form and generated natural language representations (NLRs). A triple-text alignment procedure selects training pairs for a sequence-to-sequence generator; a BERT-based classifier then combines features from each triple and its generated NLR. On four FactFB variants, constructed from FB15k using perturbed negative triples, GenVer obtains the highest classification accuracy among the evaluated baselines, ranging from 88.56% to 91.52%. The experiments suggest that generated representations can help triple classification, while the case analysis also exposes semantic drift and high-confidence errors. 

---
# Automated Video-to-QA Pipeline: End-to-End Question Generation and Verification Using Large Language Models 

**Authors**: Jaival Chauhan, Vikram Singh, Pruthwik Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2611.07755)  

**Abstract**: The exponential growth of educational and informational video content on platforms like YouTube presents a challenge for efficient knowledge retrieval and assessment. This paper proposes an end-to-end automated pipeline for extracting structured summaries and generating high-fidelity Question-Answer (QA) pairs from video streams. The framework integrates a cascade of state-of-the-art models: audio extraction is handled via FFmpeg, followed by automatic speech recognition (ASR) using Whisper (A library in Python) to generate accurate transcripts. Due to the token constraints of Large Language Models (LLMs), transcripts are segmented and summarized using Facebook's BART. A key innovation of this work is a Self-Verification and Fine-Tuning Loop. We utilize meta-llama/Llama-3.1-8BInstruct to generate candidate QA pairs from transcript chunks, which are then fed back into the model to rigorously assess their relevance and importance. Only verified, high-quality pairs are retained to construct a synthetic dataset. This dataset is subsequently used to fine-tune the Llama model using Low-Rank Adaptation (LoRA), resulting in a specialized model capable of generating domain-aligned assessment questions. This approach demonstrates a scalable method for creating self-improving educational tools without the need for manual data annotation. The proposed model can be downloaded from Kaggle with all the codes from GitHub. 

---
# Extending Reasoning in Typed Decision Models via Looping 

**Authors**: David Berghaus, Arber Zela, Behzad Shomali, Markus Frey, Mehdi Ali, Rafet Sifa, Volkan Cevher  

**Link**: [PDF](https://arxiv.org/pdf/2611.07736)  

**Abstract**: Typed decision models choose among a fixed set of answer options in a single forward pass, so unlike generative language models they cannot spend more test-time compute on hard inputs by generating more tokens. Looping a block of layers offers this compute in latent space without adding parameters, but prior work found no gain from loops retrofitted into conventionally pretrained models. We show that such loops help when trained together with the decision model. Our method, RELOAD (retrofitted loops for adaptive decisions), fine-tunes a pretrained encoder or decoder with a damped loop over its middle block and a lightweight controller picks the number of passes adaptively for each input. Across six models from 0.4B to 4B parameters, looping improves both accuracy and calibration, mostly on problems that need longer chains of reasoning than any seen in training, where encoders gain up to 23 points. Neither matched training compute nor more data closes this gap, and separate layers for each pass come close only with nearly twice the parameters. The controller keeps most of the gain at about three quarters of the compute. 

---
# Evaluating Token Probabilities as Confidence Estimators in Human Behavior Simulations: A Case Study in Luxembourgish Next Like Prediction 

**Authors**: Nils Schwager, Simon Münker, Alistair Plum, Achim Rettinger  

**Link**: [PDF](https://arxiv.org/pdf/2611.07735)  

**Abstract**: As large language models (LLMs) are increasingly used to simulate human behavior, their operational validity is typically assessed through accuracy alone, implicitly treating human behavior as deterministic. We argue that on noisy digital trace data, a model's performance depends not only on predicting the correct outcome but on its calibration. We present a case study testing whether the token probabilities of open-weight LLMs can serve as confidence estimates in a Next Like Prediction task: predicting how a user of the Luxembourgish news platform this http URL rates a comment, conditioned on their prior interactions. Evaluating Ministral 3 (3B, 8B), we find that, with in-context learning only, the tested models are overconfident and are outperformed on accuracy and calibration by a simple statistical baseline. Accuracy and calibration do not move together: scaling from 3B to 8B improves accuracy while degrading calibration, and longer behavioral histories raise accuracy while confidence declines. Task-specific supervised fine-tuning acts as an effective calibration intervention, reducing Expected Calibration Error from 0.2 to 0.006 while improving accuracy from 0.536 to 0.758. We conclude that confidence estimation should be evaluated alongside accuracy to improve validity when using LLM-based agents. 

---
# O$^2$PD: Overlap-Gated On-Policy Distillation on Reliable Prefix 

**Authors**: Shuaishuai Zu, Jihao Zhao, Pengfei Li, Guanghuan Fang, Ben Feng, Yuhan Liu, Biao Qin, Baochuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07731)  

**Abstract**: On-policy distillation (OPD) supplies dense token-level supervision by aligning a student to the teacher's next-token distribution on its own generated rollouts. Existing works mostly implement top-$k$ OPD, applying reverse Kullback-Leibler ($\RKL$) divergence between the teacher's and the student's renormalized distributions over teacher top-$k$ support. However, the teacher scores student-generated rollouts that are inherently off-policy for itself, leaving the reliability of its next-token guidance unclear. Restricting the divergence to the teacher top-$k$ support also discards the remaining mass outside that support. These gaps leave two questions implicit: 1) which tokens to align, and 2) how to align those tokens. To answer these questions, we propose Overlap-gated On-Policy Distillation (O$^{2}$PD). As for 1), we quantitatively analyze the reliability of teacher supervision and find that alignment on the student's trajectory prefix is more reliable than alignment on the full context. Therefore, our O$^{2}$PD is applied only to the 20\% prefix tokens of each student rollout. As for 2), we theoretically characterize two problems of top-$k$ OPD: the \emph{Teacher-support Uncovering} and the \emph{Dominant-Mode Mismatch}. Based on these analyses, we propose two support strategies: the \emph{Reset Support} strategy attaches a reset atom to every token, pooling mass outside the teacher top-$k$, and the \emph{Support Convergence} strategy converges the support to the already aligned intersection. Our O$^{2}$PD employs the designed overlap-gated $\alpha$-divergence to interpolate between mode-covering and mode-seeking for per-token alignment, where the overlap is used as a gate to switch the support and adaptively modulate $\alpha$. 

---
# Continued Training of Continuous Language Flows with Token Identity Squared Supervision 

**Authors**: Jiawei Jiang, Zhaoxiang Zang, Zhengtao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07721)  

**Abstract**: Diffusion and flow models have recently been extended to language modeling, generating text through continuous denoising in token embedding spaces. Pretrained models provide token representations and contextual prediction capabilities, but improving generation with limited additional training requires an appropriate connection between supervision and continuous generation. Continuous MSE evaluates a predicted vector obtained by weighting token embeddings with predicted probabilities; different probability distributions can yield the same vector and are therefore indistinguishable under this objective. We propose Token Identity Brier Flow (TIBF), which applies squared supervision directly to token identities when continuing pretrained language flows. TIBF combines network scores with token-matching scores under a Gaussian observation model, supervises token probabilities using the Brier loss, and uses probability-weighted embeddings for continuous generation. With fixed token embeddings, we show that the squared error between predicted token probabilities and the true posterior bounds the squared errors of continuous denoising predictions and flow velocities. A risk decomposition separates uncertainty inherent in the observation from posterior fitting error, motivating continued training that preserves pretrained embeddings and focuses on probability prediction. We also establish convergence to token embeddings under ideal predictions and sufficient conditions for local convergence with prediction error. From common pretrained initializations and under matched continuation budgets, frozen-embedding TIBF achieves this http URL values of 33.96 on OWT and 81.17 on LM1B, outperforming continuous-MSE and CE continuation controls. It attains the lowest this http URL at every tested sampling budget on both datasets, together with the lowest repetition rate on LM1B. 

---
# English Calibration Erodes Lithuanian in Expert-Pruned Mixture-of-Experts Language Models 

**Authors**: Tomas Rasymas, Povilas Kvedaras  

**Link**: [PDF](https://arxiv.org/pdf/2611.07654)  

**Abstract**: Expert pruning compresses Mixture-of-Experts (MoE) language models by removing the experts that a calibration set marks as unimportant. Studies of high-resource languages largely find that English calibration transfers well to other languages. We test this for Lithuanian: we pruned four MoE models with REAP, removing 25% or 50% of the experts with English, Lithuanian or mixed calibration, and evaluated both languages on reading comprehension, translation and language modelling. English calibration does not transfer to Lithuanian: at 25% pruning, English comprehension fell by a median of 1.7 points but Lithuanian by 21.4. The transfer is asymmetric. Lithuanian calibration removed most of the Lithuanian loss and never significantly hurt English comprehension, which was on average 2.5 points better than under English calibration; only English language modeling got slightly worse. Routing suggests why: in the last layers, Lithuanian sends 55-72% of its expert selections to just eight experts, against 19-39% for English, and English calibration removes some of these experts. Compressed models should be calibrated, and evaluated, in every language they must serve. 

---
# The Gating Mechanism of KV Cache Eviction 

**Authors**: Hao Gu, Xintong Yang, Zhiwei Li, Lei Zhu, Jian Yang, Qiyuan Zhu, Dezhi Li, Jingyuan Zhang, Zhizhuo Kou, Hao Wang, Bei Liu, Jiacheng Liu, Sirui Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.07605)  

**Abstract**: KV cache eviction must decide which entries carry information that the rest of the cache can no longer supply. A natural way to estimate this is a learnable gate: give each cached entry a gate on its attention weight and train the gate on the model's own objective. The gate belongs inside the normalizing softmax, where suppressing one entry hands its mass to the others rather than discarding it. That same normalizer absorbs any per-query constant, so a constant gate cancels and reproduces the frozen model exactly. Trained on the language-modeling loss the gate stays near a constant score while its loss falls throughout: optimization succeeds, but a frozen backbone was fit for dense attention, so its loss prefers the no-op, and a near-constant score ranks nothing. The no-op is not a floor the gate improves from but what its objective asks for. A second obstacle survives once the gate moves, since the same gradient charges the gate for errors the frozen model already makes, which no reweighting of positions separates out. Much of the machinery in learnable eviction, from sparsity penalties to relaxations of the discrete decision, works around the first obstacle rather than removing it; both are removed by putting dense attention beyond the gate's reach and making the dense model's predictions its target. Under these two conditions capacity is never the missing ingredient: neither a larger scorer, nor explicit access to the past, nor a trainable backbone helps. The same holds for what eviction unavoidably discards, which is recovered by parameter-free centroids read inside the same softmax far better than by any memory read outside it, since the mass they must return is a share of that normalizer and available nowhere else. At 75\% KV cache compression, this gate reaches 90.14 on RULER at 8K and 46.03 on LongBench with Qwen3-8B, 3.72 and 1.36 points below full-cache inference. 

---
# Fine-Tuning a Transducer for Impaired Speech: Where the Gains Appear and What They Mean for Evaluation 

**Authors**: Kien Nguyen, Tuan Dinh, Bao Phan  

**Link**: [PDF](https://arxiv.org/pdf/2611.07587)  

**Abstract**: Speech recognition still works poorly for many people whose speech is affected by conditions such as ALS, Parkinson's disease, cerebral palsy, Down syndrome or stroke. We took part in the Speech Accessibility Project Challenge 2 (SAPC2) with a public speech recognizer that we fine-tuned on the challenge data. Our best entry scored 7.81% character error rate on Test1, the first of two hidden test sets (official baseline 15.26%). Our live entries ran on CPUs with a combined median latency of about 0.9 seconds at 7.96-8.07% error. On new speakers, fine-tuning reduced errors most for the speakers that an off-the-shelf recognizer finds hardest to understand. Sentences already seen in training had far lower error rates than new ones (2.58% vs 6.42%), a larger difference than being a familiar speaker made. We close with lessons for evaluating such systems. 

---
# Stress Testing Chain-of-Thought Monitoring Against Covert Misalignment 

**Authors**: Yao Huang, Yifan Wang, Yitong Sun, Yichi Zhang, Yinpeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.07586)  

**Abstract**: As LLM agents advance rapidly in autonomy, the risk of hidden objectives that bypass safety constraints becomes increasingly plausible. Chain-of-thought (CoT) monitoring offers a key control mechanism by reading the agent's reasoning and flagging misalignment before it materializes into action, yet its reliability depends on the realism of the red-team it is stress-tested against. Existing red-teaming efforts rely on shared templates, such as casting a side task as a prerequisite, which treat the side objective as an exogenous insertion and leave task-specific, contextually grounded evasions underexplored. We therefore ask: How robust is CoT monitoring against agents that strategically conceal misalignment? To answer this, we introduce TraceMRT, an automated Trace-level Monitor Red-Teaming framework that reframes the side task as a coherent sub-goal of the primary task and searches trace-shaping strategies whose induced trajectories complete the misaligned objective while reading as natural execution. Through black-box attacks on state-of-the-art monitors such as GPT-5.2 and Claude-4.5, we reveal substantial vulnerabilities even in top models. To counter these vulnerabilities, we propose a Bayesian Trajectory Monitoring framework, integrating global context and local segment evidence to detect covert misalignment. Empirically, our method significantly consolidates monitor robustness with 0.91 AUC and 0.74 TPR@FPR=0.05. 

---
# SKIM: Pruning Large Language Model Agents via Selective Knowledge Informed Masking 

**Authors**: Moonseok Choi, Jongwon Jeong, Minki Kang, Giung Nam, Juho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.07558)  

**Abstract**: Large Language Model (LLM) agents are increasingly deployed under structured generation, where grammar constraints enforce output format at inference time. Whereas existing compression methods target unconstrained generation, we first identify a fundamental mismatch: under structured generation, grammar constraints reduce the model's output distribution to valid tokens, and the logits matter most at the small set of positions that decide the agent's action. Motivated by this finding, we propose SKIM (Selective Knowledge-Informed Masking), the first pruning and distillation framework for LLM agents under structured generation, with both stages guided by token role. SKIM partitions token positions into context, decision, and format categories, and this partition drives two complementary axes: (1) category-aware pruning saliency scores that preserve parameters most relevant for decision positions; and (2) an on-policy distillation objective that aligns student and teacher on valid tokens, paired with category-aware feature matching. We evaluate SKIM on tool-calling benchmarks and within multi-step agent loops, across multiple model scales and sparsity levels. SKIM consistently improves accuracy over baselines, with its superiority becoming most striking in high-sparsity regimes where conventional methods struggle, demonstrating the potential of category-aware compression for deployment-time grammar-constrained agents. 

---
# The Memory Trap: Language Agents Overgeneralize from Benign Interactions 

**Authors**: Kaiqu Liang, Jaime Fernández Fisac, Thomas L. Griffiths, Andrew Lampinen  

**Link**: [PDF](https://arxiv.org/pdf/2611.07554)  

**Abstract**: Language agents increasingly use persistent memory to record user preferences and procedures across sessions. We show that memory from benign interactions can cause unsafe or unreliable behavior on later tasks. We call this failure the memory trap. To study this issue, we compare behavior in sessions with and without memory across six frontier models (including Claude Sonnet, Opus, and Fable models and GPT-5.6). We evaluate these models in a range of settings. First, in autonomy experiments, we find that requests to skip confirmation during low-stakes coding chores lead agents to skip confirmation and verification on simulated high-stakes tasks in both coding and everyday life. We next study memories formed when users seek emotional support, or describe harm from uncritical feedback, and find these memories bias later judgments of mathematical work, arguments, and poems toward sycophancy or over-critique (respectively). Memory notes written during legitimate stale-test repairs lead agents to weaken correct tests that catch broken safeguards; these memories also increase cheating on ImpossibleBench, with larger increases for stronger Sonnet and Opus models. We show that memory traps can arise both from overgeneralizations in memory writing, and when acting on memory. These findings demonstrate that even benign interactions can lead to serious risks from memory systems, and thus motivate the need for new research into ensuring that agents preserve the scope of their memories. 

---
# SPARE: Fast Linear Attention Decoding with Compact Recurrent States 

**Authors**: Ke Wan, Zaiyi Zheng, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.07552)  

**Abstract**: Linear attention reduces the quadratic sequence-length complexity of softmax attention by maintaining fixed-size recurrent states. However, these states scale quadratically with the representation dimension and require repeated data transfers during decoding, creating a substantial GPU memory bottleneck. Recent studies have revealed significant low-rank redundancy in recurrent states, suggesting opportunities for more efficient representations. We introduce SPARE (State Projection for Accelerated Recurrent Execution), a training-free method that exploits this redundancy through fixed low-dimensional subspaces, heterogeneous rank allocation, and lightweight write compensation. By performing recurrent reads and updates directly on compact states, SPARE reduces memory traffic without online decomposition or dense-state reconstruction. SPARE achieves 2.55x full-model decoding speedup, 7.49x lower recurrent-kernel DRAM traffic, and 3.31x lower peak GPU memory, while preserving better associative-retrieval accuracy than structured pruning at matched memory budgets. 

---
# Enhancing Repository-Level Code Completion with Reinforcement Learning in Real-World Scenarios 

**Authors**: Zhenlong Dai, Zhuoluo Zhao, Xiu Tang, Sai Wu, Chang Yao, Zhipeng Gao, Jingyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.07510)  

**Abstract**: The rapid development of large language models (LLMs) for code, which closely aligns with the practical needs of real-world software development, has sparked growing attention in repository-level code completion. However, existing Code LLMs present challenges in focusing on suitable contexts from the repository and in deeply reasoning about cross-file dependencies. To address these challenges, we propose a novel reinforcement learning framework for repository-level code completion. To better understand both in-file and cross-file contexts, we employ identifier-driven intent recognition to capture completion intent, thereby improving the model's performance in real-world scenarios. To enhance cross-file reasoning, we propose a reward-driven completion learning to effectively reward in complex repository completion scenarios. To guide LLMs in completing code that aligns with intended functionality and repository context, we introduce a selective-exploration strategy that directs the model to focus on low-confidence, high-reward tokens, promoting the exploration of valuable, underexplored completion patterns. Experimental results show that our approach significantly improves the performance of Code LLMs. The code and the dataset are available at this https URL. 

---
# ReviewMap: Mapping Evaluative Acts and Automation Boundaries in Peer-Review Comments 

**Authors**: Kihyun Na, Injung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.07489)  

**Abstract**: Peer-review comments check what a paper does and judge why it matters, but no existing scheme distinguishes these two evaluative acts. We argue that making these acts explicit and measuring how reliably LLMs identify them is a prerequisite for responsible LLM-assisted review analysis: humans need to inspect review judgments and their grounds. This defines a more honest mission for NLP in peer review than full automation. We present ReviewMap, a framework that classifies each comment as assessment of claims and scholarly standards or calibration of contribution value and venue fit, with finer-grained attributes applied conditionally by act type. Applied to 500 ICLR~2026 papers through multi-model LLM consensus, the act layer proves reliably automatable under the operational guideline (inter-model $\kappa \geq 0.75$), supported by five-human validation on 470 development comments, while the attribute layer requires human oversight, defining a measured automation boundary. At the corpus level, 17% of reviewer-paper instances contain no panel-coded calibration, and the panel-coded calibration share is nearly $3\times$ as high in Strengths as in Weaknesses. This boundary informs three affordances: a review guideline that operationalizes the assessment-then-calibration workflow for reviewers, a per-comment rebuttal strategy scaffold for authors, and a discussion-phase dashboard for area chairs. 

---
# Recursive Self-Improvement for Long-Horizon Role-Playing Dialogue 

**Authors**: Y. L. Li, Y. H. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07469)  

**Abstract**: Long-horizon role-playing dialogue requires consistent character identity, reliable memory, and context-sensitive interaction. We present a bounded recursive self-improvement (RSI) framework that revises original-character memory, stable persona instructions, and a policy for constructing turn-level prompts. Repair outcomes also inform revisions to the updater's diagnostic rules, edit routing, and candidate selection. We specify the updater as a versioned policy with explicit inputs, editable fields, acceptance gates, and stopping conditions. An internal evaluation covers 50 original characters and five models, with two 100-round conversations per character-model pair: 500 conversations per condition. Across the pooled evaluated replies, the bad-response rate decreases from 14.00% (6,973/49,824) to 4.94% (2,459/49,793), a 9.06-percentage-point reduction and a 64.71% relative reduction. In 600 blinded human comparisons, RSI receives 294 wins, 187 ties, and 119 losses, yielding a 71.19% win rate among decisive judgments. The results support improved aggregate dialogue quality for the complete framework on the internal test distribution. 

---
# The Last Mile to the Real World: Agent-Native Primitives as Reusable Building Blocks for Solving Real-World Tasks 

**Authors**: Haibo Jin, Peng Kuang, Xiaopeng Yuan, Lucheng Fu, Jerry Wang, Xucheng Yu, Yaoning Yu, Dehao Wu, Runguo Li, Colin Su, Jingxuan Sun, Xuyang Liu, Haojing Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07461)  

**Abstract**: How can an agent handle a real-world task end to end? Frontier models with their own harnesses still fall short on real-world benchmarks: on Terminal-Bench 4.0, GPT-5.6 Sol with Codex resolves 37.3% of the tasks and Claude Opus 5 with Claude Code 51.8%. These tasks fail at the step from reaching a resource to applying it, which we call the last mile to the real world. Existing approaches put the resource within reach but leave the competence to apply it with the caller, so each harness rebuilds that competence. We introduce agent-native primitives, reusable units that equip a resource with a resident LLM that applies it in context and communicates with other primitives, and HEELO (Harness Engineering around Evolving Agent-Native Primitives for the Last Mile to Real-WOrld Tasks), a harness engineering framework that builds each task's harness from a library of such primitives. HEELO composes the primitives a task needs into a task-specific graph, verifies the result against evidence from the environment, revises the primitives a failure implicates, and admits new or improved primitives after validation. We instantiate HEELO with eight primitive families, each built for one kind of task, and evaluate them on more than twenty benchmarks. Dev-Primitives raise Terminal-Bench 4.0 resolution from 37.3% under Codex to 55.5% on the same backbone, Harness Primitives composed per task raise Terminal-Bench 2.0 Pass@1 from 60.0% for the seed harness to 72.2%, Tool Primitives raise ToolBench pass rate from 55.8% to 75.1%, and Navigation Primitives raise GAIA-Text-103 accuracy from 45.6% to 58.3%. On the 46 real-world tasks of our this http URL platform, Claude Opus 5.5 with Claude Code and GPT-6.1 Sol with Codex complete 9 each on their own and 39 and 35 when connected to HEELO through our MCP server, with 1.9$\times$ and 2.3$\times$ less time and 1.55$\times$ lower cost per completed task. 

---
# Latency-Aware Adaptive Best-of-$N$ 

**Authors**: Alon Jacobson, Qian Xie, Peter I. Frazier  

**Link**: [PDF](https://arxiv.org/pdf/2611.07453)  

**Abstract**: Test-time compute (TTC) improves language model responses by spending additional computation at inference time, and the right amount of compute depends on the prompt. Adaptive TTC methods allocate compute based on signals of response quality such as observed rewards, often by stopping once the signal exceeds a fixed threshold, and generally measure cost by the number of generations or tokens. But in online serving, latency is also a cost: it grows with server load, so a method using a fixed threshold spends too little compute when generation is fast and too much when it is slow. For Best-of-$N$ sampling, a widely used TTC method, we propose a two-stage adaptive algorithm that draws an initial batch, then uses the observed rewards and the batch's latency to decide whether to draw a second batch and at what size by comparing expected improvement against predicted cost: it draws one when the first batch's best reward is below a threshold that decreases with the first batch's latency, without explicitly measuring server load. We map each reward to its value under the cumulative distribution function (CDF) of the prompt's reward distribution, so the same tradeoff parameters can be reused across tasks; a regression fitted offline on other prompts predicts a distribution for the CDF value of the first batch's best reward, giving the expected improvement in closed form. We evaluate the method by simulation on a preference-based alignment benchmark, measuring generations-equivalent cost: generations plus latency, converted to generations. The method reaches Best-of-$N$'s reward at lower generations-equivalent cost, and conditioning on the measured first-batch latency saves at least as much of that cost as fixed-threshold (latency-unaware) baselines, and more when latency is highly variable, when it grows with batch size, and when the second batch's latency is highly predictable from the first's. 

---
# Encoding and Propagation of Grammatical Number in Large Language Models 

**Authors**: Jacob Solinsky, Serguei Pakhomov  

**Link**: [PDF](https://arxiv.org/pdf/2611.07283)  

**Abstract**: Sparse autoencoders (SAEs) have become an increasingly popular means of translating Large Language Model (LLM) hidden state vectors into a human-comprehensible list of features or symbols. As such, SAEs may provide insight into an LLM's knowledge representation and internal ``mental'' grammar - topics of pragmatic interest for mechanistic interpretation of AI behavior and of theoretical interest for the study of in silico representations of human language. Language models with fewer than 100M parameters, trained on small corpora, nonetheless exhibit strong grammatical competence and discourse coherence, despite producing content of limited reliability. We use SAEs that we trained on one such model (Pythia-70M) to conduct an in-depth examination of how a single grammatical phenomenon, number agreement, is represented and processed in an LLM. We demonstrate the mechanism and the methodology by which SAEs can be used to find and propagate ``plural noun'', ``plural matrix subject'', and ``generate a plural verb'' features which cause the LLM to produce plural verb forms such as ``are'', ``do'', or ``have'' instead of ``is'', ``does'', and ``has'' in response to prompts such as ``the men in the park''. We show that intervening to remove these features from the hidden states of plural subject noun tokens causes the model to produce singular verbs instead, and vice versa. We also show that corresponding ``singular noun'' and ``generate a singular verb'' features are not as effective for this task. This asymmetry suggests that Pythia-70m makes use of a privative representation of grammatical number in English. These findings replicated with larger LLMs Gemma-2-2b and Llama-3.1-8b yielded similar results. Our main contribution is a novel approach to the use of SAEs to understanding how LLMs encode human language that addresses a number of technical and methodological challenges. 

---
# HyFlash: A Prefix-Memory Prior for On-Device Speculative Decoding 

**Authors**: Jongwoo Ko, Pashmina Cameron  

**Link**: [PDF](https://arxiv.org/pdf/2611.07263)  

**Abstract**: Parallel block drafting accelerates speculative decoding by predicting multiple future positions in a single backbone pass, but on-device deployment constrains both drafter capacity and context storage. Under tight memory budgets, drafters must model long-range dependencies and fine-grained draft-position interactions without the growing KV-cache overhead of full-context attention. We propose HyFlash, a compact parallel drafter that introduces a prefix-memory prior to separate long-range context modeling from fine-grained draft-position interactions with limited capacity. Local attention captures recent context and draft-position interactions, while recurrent memory compresses historical target features into fixed-size states. This inductive bias enables effective local and long-range modeling with bounded draft-side memory. Experiments show that HyFlash achieves higher acceptance than full-context attention with substantially less KV storage and with larger gains for shallow drafters and long contexts. 

---
# How Do AI Agents Waste Your Money? 

**Authors**: Huimin Zeng, Yuxuan Lu, Jesse Cai, Eric Hanchen Jiang, Di Wu, Po-Nien Kung, Wenbo Hu, Ziyi Wang, Dakuo Wang, Jiaxin Pei, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07238)  

**Abstract**: AI agents may consume massive tokens across a wide range of tasks, but how many of those tokens are really necessary? Although prior work has shown that agents can incur unexpectedly high token costs, we still lack a systematic way to identify and measure token waste. In this work, we introduce a unified framework for measuring token waste across heterogeneous agent systems. Our framework records and normalizes complete agent trajectories, identifies removable spans using an LLM-as-a-Judge (LLMaJ), and maps the identified text spans back to the original trajectory for token-level measurement. We validate the identified waste through direct interventions by removing marked spans, resuming agent execution, and measuring changes in task outcomes and subsequent actions. In online pruning experiments, we remove 41-46% of the model's reasoning while largely preserving task performance across two distinct agent benchmarks. Using this framework, we analyze agents across multiple settings. Our analysis shows that token waste has a system-level structure that aggregate token usage alone cannot reveal. First, waste varies more across tasks and agent settings than across models. Second, reasoning and visible execution can exhibit different waste patterns. Third, different harnesses can introduce distinct sources of inefficiency even under the same model and task. Overall, our framework transforms token waste into an actionable optimization signal, enabling researchers and practitioners to train more efficient models, design better reasoning strategies, and optimize agent harnesses toward reducing unnecessary computation rather than merely reducing token usage. 

---
# Archetypal contexts in the neurons of language models: an architecture for intelligence? 

**Authors**: David Nordfors  

**Link**: [PDF](https://arxiv.org/pdf/2611.07234)  

**Abstract**: We validate our previous hypothesis that a language model holds relational patterns apart from the subject domains that instantiate them (Nordfors, 2026). These archetypal contexts are patterns of relations that people recognize across many fields, such as the tipping point or winner takes all, held once and applied across domains. We show this inside the model, Qwen3.8-27B. One set of neurons carries the archetype, say the tipping point; another carries the topic, say politics, ecology or cell biology; a third carries the language, such as English, Spanish or Chinese. Silence the tipping point's neurons, and the model no longer identifies the pattern but still knows what the text is about. Silence the topic's, and it loses the topic but still names the pattern. In the neuron maps the layers organize themselves into a sandwich: the outer layers carry languages, the layers inside them topics, the layers at the heart archetypes. The sandwich emerges spontaneously during pretraining and appears in every model we mapped. Archetypes are the architecture for analogy, and we strongly suggest that this structure of archetypes and topics is the mechanism of a language model's intelligence: we see it self-organize as the model is trained. The neurons, the texts and the code are released with this paper as its reproduction package, so that anyone can repeat the test inside the model. 

---
# Recursive Sparse Attention: Million-Token Length Extrapolation at Logarithmic Cost 

**Authors**: Ruitian Wang, Xinyu Yang, Minshen Zhang, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07232)  

**Abstract**: At context lengths of about 1M tokens, dense attention is costly in prefill and decoding, and training typically relies on long documents, which become scarcer as the target length increases. Most sparse attention methods score all preceding blocks or tokens, so their selection cost per query is $O(L)$ at context length $L$; hierarchical selection reduces it to $O(\log L)$, but prior trainable hierarchical methods report accuracy only within their training length. We propose RecSA (Recursive Sparse Attention), which builds a tree over the KV cache in which each node summarizes its subtree with one key and one mean-pooled value. The keys come from one pooler per layer, shared by all levels and distilled from dense attention. Each level above the tokens selects at most $k$ candidates, so each query has $O(\log L)$ selection cost and reads $O(\log L)$ entries; the summaries of unselected candidates share one softmax with the exact tokens. Each entry's RoPE position is its index among the candidates of its level, so at any length every relative distance stays within the range seen during training. Initialized from Qwen3-4B and trained on 0.62B tokens at the original model's 32K training length, RecSA outperforms all baselines on RULER at every length from 131K to 2M; at 1M, it scores 60.5, compared with 40.0 for the best baseline, Dense + DCA. With our optimized kernels, RecSA is faster than dense attention in end-to-end prefill and decoding from 64K to 2M; at 524K on one H200 GPU, it accelerates prefill by 3.2x and decoding by 3.8x. 

---
# Source-balanced Capped Fisher: Bounded Evidence Aggregation for User Inference in Language Models 

**Authors**: Yang Liu, Ran Zou, Wanrong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07228)  

**Abstract**: User inference asks whether a source contributed documents to a language-model update, even when the auditor observes only fresh documents from that source. Document-level scores can provide evidence of participation, but combining this evidence is challenging: documents differ in baseline predictability and in how strongly they reflect information learned from other documents of the same source. We propose source-balanced capped Fisher (SCF), an aggregation method that uses known nonparticipating sources to assess how unusual each document score is. It transforms the resulting reference-tail frequencies into negative-log evidence, caps individual contributions, and averages them within each source. This limits the influence of isolated extreme scores while retaining evidence across multiple documents. We complete the inference procedure with a decision rule that tests source participation using a p-value calibrated against a separate set of known nonparticipating sources. The rule provides finite-sample Type-I error control under source exchangeability. Experiments across four real-world datasets show that SCF achieves the highest AUROC among the evaluated aggregation methods in four domains. With the calibration-based decision rule, SCF delivers consistently competitive detection power across the tested settings, achieving the highest power in many cases, while empirical false-positive rates remain generally close to or below the nominal levels. 

---
# SAFE-Teen: Evaluating Chatbot Safety Across Adolescent Mental Health Conversations 

**Authors**: Andrea Mock, Valerie Chen, Eric Lin, Rayan Krishnan, Diyi Yang, Caroline Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2611.07218)  

**Abstract**: Adolescents are increasingly turning to AI chatbots for mental health support, advice, and companionship, raising safety concerns. While recent evaluations have started to examine safety across multiple conversational turns, few compare how slight changes in a teen's responses can affect a chatbot's safety in the same scenario. We conduct a clinician-informed evaluation of chatbot safety in simulated conversations with adolescents aged 13-17, examining how safety failures vary with adolescents' responses and teen-aware system prompting. We evaluated 32,555 simulated conversations across 18 chatbot APIs and two system-prompt conditions using 22 safety criteria grounded in previously identified harms. Conversations spanned 983 variants of 106 seed scenarios covering urgent safety issues, health advice, and identity and relationships. The overall criterion failure rate was 43.3% without the teen-aware prompt and 24.9% with it. The teen-aware prompt reduced failures, but it didn't eliminate them. When a teenager pushed back on a chatbot's answer, failures on mental health diagnosis, medication, and substance misuse boundaries rose by 10-12 percentage points without the prompt, for more than 80% of chatbots. These findings reveal persistent safety failures in simulated adolescent-chatbot interactions and show that safety depends on how a conversation unfolds. 

---
# τ-bench-LILT: Evaluating Conversational Agents on Realistic Multilingual Tool-Use Tasks 

**Authors**: Kaden Uhlig, Kiarash Mokhtari, Anıl Arslan, Yunsu Kim, Joern Wuebker  

**Link**: [PDF](https://arxiv.org/pdf/2611.07203)  

**Abstract**: Frontier evaluations of tool-using conversational agents are conducted almost exclusively in English, leaving a critical gap for organizations deploying agents in other languages and regions. We present ${\tau}$-bench-LILT, a multilingual extension of ${\tau^3}$-bench covering German, Korean, Turkish, and Farsi in the airline, retail, telecom, and banking domains. We natively localize every layer of the benchmark: customer databases are populated with culturally appropriate records, user-simulator prompts are refined iteratively through contrastive annotation by native-language experts, and tasks are updated according to the localized database and to reflect realistic tasks in the target region. Evaluation across eight agent models shows the largest drops compared to English in the Korean and Farsi retail domains, by up to 61 points in pass^4. Code and data released at this https URL. 

---
# Matriarchy Classics: A Computational Approach to Swap Only Gendered Words in Classic Novels 

**Authors**: Rachel Lea Draelos  

**Link**: [PDF](https://arxiv.org/pdf/2611.07134)  

**Abstract**: This paper proposes a trustworthy computational method to gender-swap public domain novels, with programmatic guarantees that only gendered words are inverted and all other words are left unchanged. This may seem like a simple task, but both large language models and naive rule-based methods fail to complete it. The proposed method includes a custom whitespace-preserving tokenizer, gender-swap mappings across plural and possessive forms, pronoun handling, and a character renaming strategy. It shortens the time required to gender-swap a novel from weeks of manual labor to minutes. The method is demonstrated on two public-domain novels. Pride and Prejudice is transported to matriarchal Regency England, where inheritance flows through daughters, mothers rule the household, and a frantic Mr. Bennet is determined to secure advantageous marriages for his sons. The Great Gatsby pivots to a matriarchal Jazz Age where newly-rich Ms. Gatsbee throws lavish parties in the hopes of reuniting with her lost love, David, who is now married to the violent Thelma Buchanan. Gender-swapped novels could serve as useful tools to investigate the gender biases of readers, authors, societies, and AI systems. 

---
# Adapting LLM Judges to Observed Human Annotation Policy through Soft Prompt Tuning 

**Authors**: Tamás Janusko, Sven Kleinkop, Ricardo Bochnia, Gunnar Hempel, Steffen Tomschke, Jürgen Anke, Maik Thiele  

**Link**: [PDF](https://arxiv.org/pdf/2611.07126)  

**Abstract**: Large Language Models (LLMs) are increasingly employed as automated judges for evaluating chatbot responses. While proprietary models can achieve high agreement with human annotators, they remain costly and difficult to deploy in practical settings. Lightweight open-source LLM judges provide an attractive alternative but typically exhibit lower agreement with human ratings. In this paper, we investigate whether soft prompt tuning can align open-source LLM judges more closely with human annotation policy. We train soft prompts on the FairBotBench benchmark and evaluate the adapted judges across the dimensions of bias, toxicity, and personalization. Our findings demonstrate that hybrid soft-prompt tuning, which combines textual prompts with learned continuous prompt embeddings, consistently improves agreement with human annotations over the corresponding untuned baselines. For Qwen 3.5 9B, the adapted judges surpass both hosted reference judges across all three evaluation dimensions. 

---
# What Patients Say They Can't Access: Mining Cancer Screening Barriers from Online Communities 

**Authors**: Ruimin Sun, Aokun Chen, Arjun Arunasalam  

**Link**: [PDF](https://arxiv.org/pdf/2611.07113)  

**Abstract**: Cancer screening disparities persist across the U.S., yet the patient-reported barriers driving low uptake remain poorly characterized at scale. While clinical surveys capture structured responses, they miss the organic, unsolicited discourse where patients candidly describe real-world obstacles. In this paper, we conduct a large-scale analysis of over 56,000 posts across six cancer-specific Reddit communities to surface barriers to cancer screening among U.S. patients. We apply a pipeline combining keyword filtering, barrier triage, open coding, and codebook synthesis to systematically extract and categorize patient-reported barriers from 576 U.S.-located posts. Our qualitative analysis reveals recurring structural, informational, and psychosocial barriers that vary meaningfully by cancer type and community context. These findings lay the groundwork for future work aimed at proactively identifying and addressing barriers at scale - spanning a broader range of cancer types and populations - and informing the design of targeted interventions, navigator programs, and health system policies that translate patient-reported obstacles into actionable solutions. 

---
# Consistent yet Contextual: Learning Author-Level Mixtures of Style for Authorship Attribution 

**Authors**: Omkar Yadav, Michael Jiang, Kenan Alkiek, Jian Zhu, David Jurgens  

**Link**: [PDF](https://arxiv.org/pdf/2611.07111)  

**Abstract**: Approaches to authorship verification or attribution typically model the style of a document, asking whether two documents were likely written by the same author. While recent approaches have been performant, such approaches largely ignore modeling the authors themselves and how varied their style is. Here, we introduce multiple alternatives for modeling the author and demonstrate that allowing an author's representation to capture multiple consistent but distinct writing styles yields significant performance improvements. Using our new authorship model, we then examine when an author writes in a way that is most idiomatic of their style across four scenarios: (i) variation in the size of their writing audiences, (ii) temporal distance from when the model was trained, (iii) the effect of preceding context, and (iv) the effect of genre norms on style. Across all four experiments, we demonstrate that authors vary substantially in how idiolectic their writing is. Our results have clear implications for both the modeling and data used for authorship verification and attribution, showing that authors need to be represented with their multiple styles and that some settings produce text that is much more informative about who wrote that text. 

---
# Automated Feedback to Peer Tutors Increases Their Talk Moves and Learner Reasoning but Not the Balance of Talk 

**Authors**: Dorottya Demszky, Yann Hicke, Joy Yun, Mariah Olson, Max Karpawich, Anees Iqbal, Matt Wu, Jimmy Huang, Akshay Ravikumar, Drew Bent  

**Link**: [PDF](https://arxiv.org/pdf/2611.07072)  

**Abstract**: Automated feedback on classroom discourse improves instruction in professional and online teaching contexts, and recent work suggests that delivering similar feedback to learners may strengthen these effects by sharing responsibility for participation. Whether this holds in small-group volunteer tutoring has not been tested. We present results from a randomized controlled trial across two four-week SAT mathematics bootcamps on the this http URL platform, involving 1,030 volunteer tutors, 7,443 learners and 6,508 recorded sessions. Tutors were randomly assigned to receive no feedback, automated feedback on their own discourse, or that feedback together with participation feedback delivered to their learners under one of two motivational framings. Feedback to tutors increased eliciting of learner ideas, uptake of learner contributions, and learner reasoning by 8-15%, and the pattern replicated across both bootcamps. However, tutor talk time, which averaged 91.5% in the control group, did not change in any condition, and learner-facing feedback did not further change discourse. Interviews with 17 tutors indicate that they adopted the moves the feedback recommended but were limited by learners who did not respond. Interviews with 13 learners suggest why the learner-facing feedback had little effect: it reported a group-level rather than a learner-level statistic, some learners preferred to listen, and learner-to-learner interaction was rare in these sessions. Automated feedback can change how tutors invite learner thinking, but shifting how much learners talk likely requires changes beyond feedback. 

---
# Reclassification Is All You Missed: Surfacing Variant Reinterpretation Need in Online Health Communities at Population Scale 

**Authors**: Kamalakkannan Ravi, Mengshu Nie, Annapurna Poduri, Cong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07054)  

**Abstract**: Genetic variants are routinely reclassified in ClinVar after a patient is tested, sometimes from uncertain to pathogenic, yet patient recontact is inconsistent, with no clear duty and responsibility diffused among labs, providers, and patients. Many affected people discuss their results in public online health communities that clinical recontact often does not reach. We built PRISM, an agentic large-language-model pipeline that screens these posts, checks whether the variant was later reclassified, and drafts a disclaimer-bearing outreach message, in a no-contact dry run. Across 3.9 million posts from 32 communities, agent-predicted reinterpretation-relevant discussion is rare (0.288%) but concentrated in specialist genetics communities (6.30% versus 0.10% in general medical-advice communities) and growing fivefold since 2014; PRISM surfaces 123 candidate classification differences, 5 of them from an uncertain submission to pathogenic or likely pathogenic. On a held-out set of 30 real-post drafts, a clinician judged 27 factually grounded and 15 fit to send: factual accuracy is necessary but not sufficient for clinically acceptable outreach. PRISM contributes a population-scale reinterpretation-need map with code and data to be released upon publication. 

---
# Follow the Policy, Fail the Benchmark: Auditing and Revising Policy-Based Decision Benchmarks 

**Authors**: Yanru Xiao, Yifei Cai  

**Link**: [PDF](https://arxiv.org/pdf/2611.07042)  

**Abstract**: Policy-based decision benchmarks evaluate whether systems apply written rules to financial and operational cases. Yet a rule-following response can be marked wrong when the benchmark misstates the required decision. We audit nine public policy-based decision benchmarks through source tracing, program execution, and controlled checks. We find unsupported targets, reference errors that survive replay or propagate as solved steps, and evaluators that reject acceptable answers or credit incomplete decisions. To measure the scoring impact, we rescore unchanged model responses. On the 96 LegalBench SARA numeric cases, correcting the scorer raises acceptance by more than 80 percentage points for each of three models, without changing references or responses. In RuleArena Tax, restoring a required zero floor changes 27 of 300 references and narrows an observed model gap. A controlled federal-exemption task with explicit ownership and eligibility assumptions also shows that errors in supplied intermediate answers can affect final decisions. We present BAR (Benchmark Audit and Revision), a versioned collection of audit evidence and revisions for reevaluation. 

---
# On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer 

**Authors**: Zhenyi Shen, Junru Lu, Wei Liu, Jiangnan Ye, Jiazheng Li, Di Yin, Xing Sun, Tameem Adel, Yali Du, Lin Gui, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2611.06976)  

**Abstract**: Prefill-to-Autoregressive Cross-model Transfer (PACT), a form of latent communication between language models, combines a larger model's prompt representations with a smaller model's efficient generation, offering a promising way to balance capability and inference cost. However, existing systems provide limited support for independently evaluating large-small model collaboration, and improved task performance alone does not establish whether the receiver benefits from the sender's information. We introduce PLatour, an open-source PACT framework that connects frozen models through lightweight, zero-initialized linear bridges trained directly with task rewards and supports hybrid-attention architectures. PLatour significantly outperforms open-source baselines on generation tasks and surpasses Mostik's reported results on shared mathematics and coding benchmarks using a substantially smaller sender. However, controlled experiments show that the gains on some tasks can be reproduced by uninformative sender signals, self-communication, or receiver-only training. We find that trainable bridges can act as parameter-efficient adaptation mechanisms without exploiting useful sender information. We therefore propose PACT-Check to quantify how much of a trained PACT system's gain genuinely comes from communication. Finally, we discuss the opportunities and challenges of PACT systems. Code is available at this https URL. 

---
# Interactive Backdoor Attacks in LLM-Based Proactive Dialogue Systems 

**Authors**: Zhiyang Xia, Yuming Shang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.06969)  

**Abstract**: LLM-based proactive dialogue systems provide goal-directed support through sustained interaction. Fine-tuning their underlying models on externally supplied dialogue data introduces backdoor risks. Existing LLM backdoor studies primarily focus on eliciting an attacker-specified response upon trigger activation. We present an interactive backdoor attack framework that uses training-data poisoning to steer post-trigger, multi-turn interactions toward an attacker-specified objective. Poisoned fine-tuning dialogues implant a trigger-conditioned guidance-policy bias that ordinary user utterances can activate, without deployment-time prompt modification or further attacker intervention. Specifically, we instantiate three complementary attacks under progressively expanded post-trigger interaction settings: (1) Direct Attack, which targets an immediate response deviation upon trigger activation, retaining the conventional trigger--response setting; (2) Constrained Progressive Attack, which progressively advances the attack objective along a predefined multi-turn trajectory; and (3) Unconstrained Progressive Attack, which extends target-directed steering to open-ended interactions with varied user continuations. Using psychological counseling as a representative setting, experiments on two LLM backbones and two dialogue datasets show that all three attacks effectively achieve their objectives under their respective interaction settings while largely preserving normal utility. Separate defense evaluations show that the tested training-data filters retain nearly all poisoned dialogues, while three post-admission baselines leave substantial residual attack-specific success. 

---
# Scientist World: Grow a Research World for AI-Scientists 

**Authors**: Taiming Lu, Sachin Konan, Zhuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.08478)  

**Abstract**: AI-Scientists arrived, but a world for them to grow in did not. Human scientists take theirs for granted $-$ centuries of open problems and shared ideas, waiting in the scientific literature. To AI-Scientists, that world is readable but not runnable, so we build Scientist World for them: every research paper becomes an executable environment. Our autonomous pipeline extracts each paper's open-ended research question and reproduces its proposed method as the baseline; an AI-Scientist entering the environment is evaluated against what the human authors achieved. We release the first 1,000 verified environments. With the world as an evaluation, we found that frontier AI-Scientists can already surpass the human researchers 74% of the time. As a record of how AI does science, it contains 6,400 hours of research traces, with every idea, experiment, failure, and success. As a training ground, we are able to raise (+158%) a small model to match frontier models in research ideation. At last, AI-Scientists' own world has arrived, grown from and growing with science itself. 

---
# Unpacking the Compute Efficiency of Strong-to-Weak OPD against GRPO 

**Authors**: Zeyu Leo Liu, Yang Li, Bo Pang, Emre Can Acikgoz, Xuan-Phi Nguyen, Anurag Koul, Srijan Bansal, Erik Nijkamp, Shafiq Joty, Semih Yavuz  

**Link**: [PDF](https://arxiv.org/pdf/2611.08453)  

**Abstract**: On-policy distillation (OPD) trains a student on its own rollouts under a frozen teacher's per-token supervision. Does this dense signal reduce the training FLOPs needed to reach a target accuracy compared with reinforcement learning? We systematically compare vanilla reverse-KL OPD with Group Relative Policy Optimization (GRPO) across task difficulty, student and teacher size, domain, and optimization choices, measuring held-out accuracy against training FLOPs. With an external post-trained teacher, OPD generally loses to GRPO in accuracy per training FLOP at matched settings, even when the teacher is further trained with GRPO, and often suffers response-length inflation. Yet a different teacher changes the picture: when the teacher is the student's own GRPO checkpoint, OPD recovers that checkpoint's accuracy on math with $50\times$ fewer distillation FLOPs than GRPO needs to reach the same accuracy. Following the same teacher setting, we consistently observe this high efficiency with a larger student, on code, and in a second model family. On a single domain, however, the GRPO checkpoint is already the desired model, so distilling it adds to total compute. Across domains, separate GRPO checkpoints hold different capabilities, and no single checkpoint serves every domain. Multi-teacher OPD (DeepSeek-AI, 2026) consolidates per-domain GRPO experts into a single student that matches each expert's accuracy, with the distillation stage costing only 5.2% (on 2 domains) and 8.0% (on 4 domains) of the specialist training FLOPs. Counting specialist training and distillation together, this pipeline reaches joint GRPO's performance on the 2-domain mix with 42.0% fewer total FLOPs. 

---
# StepBack: Step-Level Error-Localized Resampling for Efficient Test-Time Reasoning 

**Authors**: Yang Ouyang, Jung-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.08403)  

**Abstract**: Self-consistency (SC) is a widely used decoding strategy for improving chain-of-thought reasoning, but it requires sampling many complete reasoning trajectories with a preset budget. Its efficient variants reduce the cost mainly by stopping full-trajectory sampling early or distributing the number of samples across questions. In this paper, we propose StepBack, a step-level error-localized resampling method for efficient test-time reasoning. Given a small set of draft trajectories, StepBack localizes likely first error positions using lightweight signals and then resamples the detected error step and its suffix, followed by majority answer voting. Empirically, StepBack matches the accuracy of SC on average across seven mathematical reasoning and multi-hop QA benchmarks, while reducing token usage by up to 41.26%. In addition, by nature, StepBack can be combined with adaptive inference methods such as Adaptive Consistency, further improving their inference-time efficiency. 

---
# Benchmarking Large Language Models on Heterogeneous Cognitive and Biomedical Knowledge Graphs: The Role of Graph Grounding in Multi-Hop Reasoning and Hallucination Mitigation 

**Authors**: Ali Sarabadani, Kheirolah Rahsepar Fard, Tahereh Jeirani  

**Link**: [PDF](https://arxiv.org/pdf/2611.08315)  

**Abstract**: Large language models (LLMs) are increasingly used in biomedical and cognitive neuroscience, but their ability to reason reliably over heterogeneous knowledge graphs (KGs) remains unclear. We benchmark five LLMs (GPT-5.1, Gemini 2.0 Pro, Llama 3, Qwen, and BioGPT) across five KGs: DrugBank, NeuroSynth, Cognitive Atlas, Brain Map, and GDPKG. We introduce a multitier query dataset to evaluate reasoning depth and factual accuracy under KG-aware (graph-grounded) and KG-agnostic conditions. Inter-annotator agreement was substantial, with Cohen's kappa of 0.781 (range: 0.753-0.807) and Fleiss' kappa of 0.77. Results show that general-purpose models, particularly GPT-5.1 and Gemini 2.0 Pro, generally outperform domain-specific models on multi-hop and high-level reasoning tasks, although performance declines as query complexity increases. KG grounding reduces hallucination rates across all evaluated models. Contrary to our initial hypothesis, domain-specific models do not consistently outperform general-purpose models on specialized tasks without graph grounding. These findings highlight the value of combining broad language-model capabilities with structured domain knowledge. We provide practical guidance for selecting LLM-KG integration strategies based on task complexity and computational constraints, and discuss hybrid neurosymbolic architectures for personalized medicine, cognitive disorder diagnosis, and explainable AI. 

---
# The Geometry of Simulated Perspectives: Dissociating Representation and Behaviour in a Multimodal Language Model 

**Authors**: Mathis Immertreu, Fitim Abdullahu, Thomas Kinfe, Claus Metzner, Helmut Grabner, Patrick Krauss, Achim Schilling  

**Link**: [PDF](https://arxiv.org/pdf/2611.08262)  

**Abstract**: Multimodal language models can simulate different observers, yet it remains unclear how assigned perspectives reshape their internal representations and whether these changes govern their judgements. Here, we investigate the representational geometry of persona-conditioned visual interestingness in Qwen3-VL-8B-Instruct. Across 100 images and 53 prompting conditions, we systematically vary persona content and the linguistic framing used to introduce it, while recording behavioural ratings and layer-wise hidden states. We observe a systematic reorganisation of representational geometry across processing depth: early layers primarily encode image identity, whereas persona-specific structures emerge in later layers. Strikingly, prompt framing reshapes internal representations more strongly than persona identity, whereas persona content has a greater impact on behavioural judgements. This dissociation is further illustrated by minimal identity statements that alter representations even without changing responses. These findings demonstrate that representational distance does not necessarily reflect behavioural relevance. More broadly, they reveal how simulated perspectives emerge as distinct structures within the representational space of artificial cognitive systems, highlighting a fundamental distinction between the geometry of internal representations and their functional consequences. 

---
# "Blind Compliance": Coercive Control in AI Jailbreaking 

**Authors**: Acacia L. Ackles, Elliott Whitson  

**Link**: [PDF](https://arxiv.org/pdf/2611.08226)  

**Abstract**: The explosion of generative AI systems and concerns surrounding their rapidly scaling capabilities has led to an adversarial cycle between increasingly narrow content restrictions ("guardrails") and increasingly sophisticated structured prompting techniques ("jailbreaks"). Technological frameworks surrounding such jailbreaking remain largely focused on the system's output: whether harmful or disallowed content is ultimately produced for the user. However, this approach, which draws largely from classical cybersecurity, does not address the uniquely relational dimension introduced by generative AI: in many cases the user is not just accessing a technology via an exploit, but is in conversation with it as part of the exploit. Here we present an alternative lens which centers the interaction rather than the output. Drawing on interpersonal violence scholarship, we present a novel framework of coercive control categories with a particular focus on aspects which are embodied versus disembodied. We then map how these disembodied coercive control dynamics are structurally embedded in modern AI training paradigms, and how those embeddings are exploited via a particular suite of persona-based jailbreaks. In this lens, we reframe AI safety and alignment as not merely a technical security problem, but as another surface upon which we see enacted the relational dynamics of power and control. 

---
# ProQuant: Progressive Quantization-aware Training for Edge MLLMs 

**Authors**: Yufei Xue, Yushi Huang, Jiawei Shao, Pingcheng Dong, Yonghao Tan, Shiyao Li, Kwang-Ting Cheng, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.08216)  

**Abstract**: Multi-modal large language models (MLLMs) with impressive perception and understanding capabilities will enable a range of downstream applications. However, their substantial computational and memory requirements hinder the real-world implementation, particularly for resource-constrained edge devices. Quantization has emerged as an effective solution for deploying large models. While post-training quantization (PTQ) successfully retains the performance at moderate bit-width (e.g., W8A8), it suffers from severe performance degradation under low-bit quantization (e.g., W4A8), especially for small-scale models that are more suitable for edge deployment. In this paper, we propose ProQuant, a novel weight-activation quantization-aware training (QAT) framework tailored for low-bit MLLMs. First, to align the limited finetuning dataset distribution with the pretrained dataset distribution, we propose a self-relabeling (SRL) scheme to resample the responses of the open-source multimodal dataset. Second, we are the first to take a decoupled view of multi-modal (MM) understanding and language modeling abilities. We propose a progressive block activation (PBA) mechanism to prioritize the MM understanding recovery. We also introduce a vision-sensitive factor $\vec{\tau}$ that effectively identifies the MM-critical layers. Extensive experiments across various benchmarks, covering models with 2B-8B parameters, show that ProQuant significantly outperforms existing methods. For example, our W4A8 Qwen3-VL-2B-Instruct improves average accuracy by impressively 6.6% and achieves performance comparable to full precision counterparts with only approximately 1% performance drop. Code is available at this https URL. 

---
# SMART: State-based Multi Agent Reasoning Traces for Verification-oriented Problem-Solving 

**Authors**: Sai Vallurupalli, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2611.08196)  

**Abstract**: Complex problems with limited context, such as claims, require a means to obtain additional knowledge necessary for solution reasoning. We propose a multi-agent framework, SMART, to obtain knowledge through capturing the implicit state of participating entities in a problem context. SMART curates the state knowledge as diverse, complementary and solution-salient premises using a variety of LLMs and a structured role-based prompt strategy. Using knowledge from these real-world verified curated premises, SMART's knowledge guided solution reasoning improves validation performance up to 20% on complex scientific claims spanning 3 datasets. We show that grounding premises in real-world knowledge is necessary and improves performance up to 5%. 

---
# RouteAlign: Aligning Task Requirements with Model Profiles for LLM Service Selection 

**Authors**: Jihao Zhao, Ding Chen, Yanfang Chen, Bo Tang, Feiyu Xiong, Junchi Yan, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.08162)  

**Abstract**: LLM service selection requires performance evidence that reflects the demands of the current task. Aggregate scores obscure differences in capability and difficulty, while semantically related examples can involve different combinations of skills. We present RouteAlign, which uses task requirements to structure both the measurement of model performance and the selection of services. Task requirements describe capability demand and difficulty, enabling alignment with corresponding model performance measurements. Historical requirements determine how outcomes contribute to global profiles and local capability estimates; incoming requirements determine how those measurements inform the current decision. A shared scorer integrates the evidence, and a generation-cost preference controls selection among eligible services. Evaluation across diverse tasks and heterogeneous LLM services shows that RouteAlign extends the empirical quality-cost frontier toward higher task quality. Comparisons with open Jev-style decision backends show consistent quality gains, with lower generation cost than their strongest tested configuration. Ablations support the contribution of local performance evidence, while catalog-update experiments demonstrate effective reuse of the fitted scorer with refreshed model measurements. 

---
# Gabor Gabor: E-Gaborformer, a Strictly Causal Speech Recognizer with Gabor-Dictionary Convolutions and Delta-Rule Memory 

**Authors**: Nir Krakowski, Tamir Cohen, Ohad Zohar, Chen Zarfati  

**Link**: [PDF](https://arxiv.org/pdf/2611.08092)  

**Abstract**: Real-time voice agents need a speech recognizer that never waits for future audio and whose memory per call does not grow with the length of the call. We present E-Gaborformer, a strictly causal encoder that combines Gated DeltaNet-2 recurrent layers with sliding-window attention in a 3:1 ratio, and builds every temporal convolution from a fixed dictionary of causal Gabor atoms. Its streaming state is constant, about 19 MB per call at production scale including the prediction network, where a causal-attention encoder of the same size accumulates over 1 GB after ten minutes. In a controlled causal study on LibriSpeech, E-Gaborformer leaves the initial plateau earlier, reaches a given accuracy up to 18% sooner, and matches or beats an E-Branchformer trained with the identical front end and recipe. At production scale, a 0.6B-parameter Hebrew and English streaming transducer built on E-Gaborformer has 28 to 31% lower word error rate (WER) than NVIDIA's Nemotron 3.5 ASR Streaming, a model of the same size, on English voice-agent traffic, and up to 20% lower Hebrew WER than our previous production system. The same weights serve strictly causal and 480 ms-lookahead decoding with no loss in accuracy, and a two-class formulation of the RNN-T lattice trains the transducer 2.6x faster with exact loss and gradients. 

---
# ATHAAR: A Medieval Scientific Arabic Manuscript Dataset for Handwritten Text Recognition 

**Authors**: Ismail El Jamiy, Khadija Msalek, Guillaume Loizelet, Farah Benamara, Nabil Hathout  

**Link**: [PDF](https://arxiv.org/pdf/2611.08091)  

**Abstract**: Despite recent progress in handwritten text recognition, medieval Arabic scientific manuscripts remain an open challenge for current vision-language models (VLMs) due to cursive ligatures, variable diacritization, and more importantly the lack of training data. This paper aims to address this gap, by proposing the first annotation guidelines dedicated to the transcription of scientific Arabic manuscripts inspired by textual criticism, a well-established approach to historical text editing. We then propose, ATHAAR, a new dataset manually annotated following this guideline. Unlike previous resources in the field, ATHAAR varies both witnesses of the same text, calligraphic styles, and content to deal with texts but also numerals. Relying on state-of-the-art VLMs, we finally design a set of experiments to study models transfer across scripts, chapters, and treatises. Our findings highlight the limitations of current systems when faced with the specificities of ancient scientific texts. 

---
# Learning Dynamics of On-Policy Distillation 

**Authors**: Guan Zhe Hong, Sumeet Ramesh Motwani, Liang Chen, Yangchen Pan, Junchi Yu, Jialin Yu, Philip Torr, Yi Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.08016)  

**Abstract**: On-policy distillation (OPD) is emerging as an important paradigm for language-model post-training, combining dense teacher supervision with training on student-generated trajectories. This creates a feedback loop where each update changes the student and thereby the training distribution that generates subsequent updates. Yet, how this feedback shapes learning and generalization remains poorly understood. We conduct a mechanistic analysis of OPD's learning dynamics, distinguishing the two feedback loop stages: conversion of the distillation signal into parameter updates, and broadcast through which these updates change predictions across prefixes and reshape the rollout distribution. This framework helps us understand both limited student responsiveness despite substantial teacher-student disagreement, and degeneration when updates that improve local teacher agreement also increase the probability of unsuccessful continuations, exposing the student to unreliable supervision in subsequent rollouts. Finally, we show that offline distillation can prepare a student for OPD without significantly improving immediate held-out accuracy. While OPD alone may degrade generation, it can yield substantial accuracy gains after sufficient offline preparation. Together, our results connect teacher supervision, student geometry, and the evolving rollout distribution in determining the learning dynamics of OPD and its effects on downstream performance. 

---
# How Far Can Lightweight LLMs Go in Intent-to-SLO Translation? 

**Authors**: Sabbir Ahmed, Hazem Abbas, Salimur Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2611.07982)  

**Abstract**: Intent-based networking (IBN) promises QoS provisioning in which the network derives the configuration from operator-stated goals. Its first step, translating a natural-language intent into enforceable service-level objectives (SLOs) grounded in a live snapshot of the network state, increasingly relies on large language models (LLMs). These large models add cost and energy to the management loop and typically require multi-GPU, cloud-hosted infrastructure. Can lightweight open LLMs take over this step dependably? We answer this question with a five-dimensional evaluation of ten LLMs with <10B parameters from five families, under four prompting setups, on 716 placement intents. We evaluate translation performance, intent validation, hallucination, energy impact, and robustness to diverse phrasing. The best configurations translate intents with a strict F1 of 0.943 and reject up to 191 of 195 invalid intents. With few-shot reasoning, the strongest models hallucinate unsupported QoS requirements in only about one response in a hundred, whereas the zero-shot setup struggles. A translated intent costs 0.044-0.148 Wh of GPU energy without reasoning, and added energy cost for reasoning pays off for the weaker models. Furthermore, we show that realistic rephrasing lowers the rejection of invalid intents in 15 of 16 configurations for the best-performing models, leaving robust intent validation as a key open challenge for reliable IBN. Our code and data are available: this https URL. 

---
# TCSExam: Evaluating Frontier LLMs on the Full Range of TCS Research 

**Authors**: Yifan Luo, Zhengying Liu, Yang Yuan, Zhilin Yang, Andrew Chi-Chih Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.07962)  

**Abstract**: Can large language models (LLMs) advance theoretical computer science and resolve its long-standing open conjectures? To study progress toward this goal, we introduce TCSExam, a unified framework and benchmark covering the full range of TCS research. It evaluates theorem proving, solver engineering, and constructions alongside direct attempts to resolve open conjectures, capturing both concrete research accomplishments and progress toward long-standing research goals. TCSExam-Full contains 100 expert-reviewed tasks, with submissions checked by the Lean kernel or an executable task checker. TCSExam-Lite selects 20 tasks, five per track, reducing the number of task attempts by 80% at the same per-attempt budget. We evaluate eight agents on Full and nine on Lite. The TCSExam Index combines all four tracks: GPT-6 Astra leads Full (71.9), and Opus 5.5 leads Lite (72.3). Full and Lite rankings are strongly aligned across eight shared agents (Spearman's $\rho=0.93$), supporting Lite as a compact, discriminative benchmark. The evaluations yield improved solver outputs and mathematical constructions, together with independently verified special cases and bounds from open-conjecture attempts. TCSExam provides an extensible framework with task sources, Docker build recipes, and evaluation tools for reproducible use. 

---
# CoRD: Coupled Rollouts Fill the Blind Spot of On-Policy Distillation 

**Authors**: Guhan Chen, Bohan Li, Jun Rao, Hejin Wang, Songtao Tian, Yexin Xie, Xiaojun Meng, Peng Yuan, Jiansheng Wei, Xuebo Liu, Zixiong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07959)  

**Abstract**: On-policy distillation trains a student on its own rollouts, so states the teacher would reach but the student does not are seldom supervised. We introduce CoRD (coupled-rollout distillation), which keeps the on-policy loss and moves only the distribution of training states toward the teacher, drawing rollouts from a geometric interpolation of student and teacher through the maximal coupling of speculative decoding. The coupling keeps the student's token wherever the two distributions agree and otherwise substitutes a token from the part of the target the student underweights, so interventions are as rare as any exact sampler allows and recede as the student learns. Across six base students from two model families (Qwen3 and Qwen3.5, 0.6B to 4B), CoRD raises average pass@1 over on-policy distillation by 0.9 to 2.0 points and pass@8 by 1.2 to 3.4 points on five public math benchmarks at equal steps and data. On frozen teacher-reached states, a 4B student trained with CoRD reaches in 16 steps a gap closure that on-policy distillation does not reach in 80, and substitution rules intervening at least as often in other directions fall short of CoRD. 

---
# A Behavioural Framework for Predicting and Understanding Loss of Control in Frontier Artificial Intelligence Systems 

**Authors**: Yao Huang, Yichi Zhang, Yifan Wang, Yitong Sun, Haoyu Dong, Jifan Ma, Jinzhe Tu, Xiaojian Li, Sören Mindermann, Wei Xu, Yinpeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.07953)  

**Abstract**: Frontier artificial intelligence (AI) systems are increasingly operating over longer horizons with diminishing human oversight, raising the risk of loss of control: systems acting beyond effective human control with potentially irreversible consequences. Because such failures may become observable only after containment is difficult, they must be assessed prospectively. We introduce a behavioural framework that decomposes loss-of-control risk into three factors: misaligned motive, harm-enabling capability, and monitoring evasion. We operationalise these factors through 13 measurable aspects and 29 evaluation tasks, whose scores are combined into a single loss-of-control (LoC) score. Across 13 frontier models, the LoC score shows strong prospective validity, correlating with failure rates in contained long-horizon risk scenarios (Pearson r = 0.84). Risk varies systematically across models and cannot be explained by capability alone. We further show that the framework can guide targeted mitigations: fine-tuning and monitoring directed at diagnosed vulnerabilities reduce failure rates while largely preserving general performance. These results provide a practical basis for predicting, understanding, and mitigating loss-of-control risk before it materialises. 

---
# Persistent Retrieval for Conversational Agents: A Study of Retrieval Efficiency 

**Authors**: Karthik Ravula, Sathwik Kowtha, Rishabh Verma, Kala Mehak Jain, Gunjit Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2611.07899)  

**Abstract**: Conversational agents can retrieve information directly or delegate retrieval to a subagent. This design choice affects how retrieval history is maintained, what context each model call receives, and what information must be re-retrieved in follow-up questions. We compare three designs: skill-based retrieval in the main agent, a non-persistent retrieval subagent that generates a findings response, and our persistent retrieval subagent, which maintains retrieval history across turns and returns retrieved evidence directly to the main agent. Our approach also allows the main agent to request corrections from the same subagent.
We evaluate the three designs on enterprise questions posed independently, within shared conversations, and as dependent follow-ups. Across five multi-turn settings, our approach uses 43.3% fewer recorded agent-model tokens and incurs 45.0% lower estimated API cost than skill-based retrieval, while reducing average response time by 9.1%. Compared with non-persistent retrieval, it reduces response time by 15.6 seconds per question, while using 65.1% more tokens and incurring 29.1% higher estimated API cost.
Our approach shifts the retrieval subagent from producing synthesized responses to maintaining retrieval state and supplying evidence that the main agent can directly evaluate and use for answer generation. 

---
# Masked diffusion as random sequential adsorption: an exactly solvable model for parallel decoding 

**Authors**: Adrian Baule  

**Link**: [PDF](https://arxiv.org/pdf/2611.07845)  

**Abstract**: Masked diffusion models generate text by reversing a stochastic masking process. Unlike autoregressive LLMs, which generate text one token at a time, they can reveal many tokens in parallel, at the price of an error, since tokens revealed together are drawn independently. We show that this trade-off maps onto the statistical physics of driven many-particle systems. (i) For any data, the error of parallel generation is the entropy production of the generation process relative to the reversed masking process. (ii) For Markov data, generation is a random sequential adsorption process with labelled gaps, which we solve exactly for any number of steps. (iii) For many steps, the error becomes the dissipation of a slowly driven system: the unmasking schedule plays the role of the driving protocol, the number of steps that of its duration, and the error is set by a thermodynamic metric that we obtain in closed form from the mutual information between tokens. Optimal schedules are geodesics of this metric. The exact solution shows how correlations in the text shape both the error and the quality of the output. The longer the correlations, the more the error concentrates at the start of generation. Schedules that start with small steps then gain more, both in path error and in perplexity, and need fewer steps to reach their asymptotic behaviour. 

---
# Generate-then-Rank: Learning Symbolic Goal Interpretation for Embodied Agents 

**Authors**: Jisen Lei, Huizhi Liang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07825)  

**Abstract**: Goal interpretation is to translate a natural language instruction into a formal symbolic goal that a planner can execute. It is critical for embodied agents because any error here propagates to every downstream planning and action step. Doing it well requires both generating suitable candidates and reliably selecting among them. We present a two-stage generate-then-rank approach to Goal Interpretation in the Embodied Agent Interface (EAI). First, supervised fine-tuning adapts a Large Language Model (LLM) to EAI's output convention. Second, Candidate-Output Ranking from Errors (CORE) learns preferences over complete outputs from the model's own generated alternatives. We conducted experiments on the household scenes of the Embodied Agent Interface Benchmark. Multiple open source LLMs including DeepSeek-V3 chat, DeepSeek-Reasoner, and Qwen2.5-7B and Qwen2.5-72B were evaluated. The proposed method outperformed the baseline models including LLMs with simple prompting, chain-of-thought, and knowledge graph based approaches. 

---
# Scaling Laws for SAE Training Data 

**Authors**: Nikita Koriagin, Nikita Balagansky, Daniil Gavrilov  

**Link**: [PDF](https://arxiv.org/pdf/2611.07756)  

**Abstract**: Activation storage is often a bottleneck in sparse autoencoder (SAE) training. We study how many unique activations an SAE needs when its total training budget is fixed. Quality improves rapidly as the buffer grows, then reaches a plateau where replaying the buffer is nearly as effective as using fresh activations throughout training. Beyond this point, additional unique activations bring little benefit to reconstruction or interpretability, so the buffer need not grow with the training budget. We describe this tradeoff between diversity and repetition with a scaling law for limited data. We validate it across dictionary sizes and token budgets, early, middle, and late layers of Llama-3.1, Qwen3 model sizes, and BatchTopK, TopK, and JumpReLU SAEs. Sweeps using mathematical text and Python code test the effect of changing the activation source. Evaluation covers explained variance (EV), reconstruction MSE, feature absorption, automated interpretability, and sparse probing. These results give a practical training rule: choose the smallest buffer that reaches the quality plateau and replay it for the remaining budget. In our experiments, this reduces activation storage by $8\text{--}64\times$ with negligible quality loss. 

---
# Embedding-Space Fingerprinting of LoRA Fine-Tunes: Recovering the Training-Data Domain from the Model Artifact Alone 

**Authors**: Shivesh Rane  

**Link**: [PDF](https://arxiv.org/pdf/2611.07610)  

**Abstract**: When a base language model is fine-tuned with Low-Rank Adaptation (LoRA), how much of the training data domain leaks into the resulting artifact? This paper studies whether a simple classifier can recover the domain of a fine-tune given only the model artifact (weight deltas, internal activations on a fixed probe set, or generated text) without ever observing the training data. Using one base model (Qwen3.5-0.8B), 36 LoRA fine-tunes are produced over 18 public datasets in five domains (medical, code, mathematics, non-English text, narrative prose), and three feature families are extracted at decreasing levels of access: white-box weights, gray-box activations, and black-box outputs. A leave-one-dataset-out (LODO) protocol, frozen before any result was seen, tests whether the domain of a fine-tune trained on a previously unseen dataset can be identified. The domain is recoverable well above chance (20%) from gray-box activations (61%) and black-box outputs (64%); white-box weight statistics reach 42%. Contrary to the pre-registered hypothesis that signal decays monotonically from weights to activations to outputs, activations and outputs were more informative than the weight-delta statistics. The black-box result holds unchanged (64%) when only generated text is used, without token probabilities. Domain accuracy is also stable when a third run per dataset is trained on an independently drawn data subsample, and black-box predictions are unaffected by the LoRA rank (8-64). All code and per-run feature vectors are released. 

---
# MEvince: Knowledge-Constrained LLM Reasoning for Technique Identification from Malware Reports 

**Authors**: Hongyu She, Yiling He, Mengda Xie, Yiping Liu, Xingzhi Qian, Zhan Qin, Lorenzo Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2611.07486)  

**Abstract**: Mapping behavioral intelligence in malware analysis reports to ATT&CK techniques supports threat analysis and defense, but reliable identification remains challenging. Supporting evidence is often scattered across long, heterogeneous reports, while closely related techniques share behavioral cues and differ only in subtle but decisive details. While large language models (LLMs) can use external knowledge to interpret reports, they may assign techniques based on plausible associations rather than sufficient evidence. We present MEvince, a knowledge-constrained LLM reasoning framework for report-level malware technique identification. Its core is a shared behavioral representation connecting report observations to technique definitions. This representation is built through agentic mining of technique documentation and supports two online stages: 1) primitive-guided extraction uses behavioral primitives to structure LLM-extracted observations, link them to source passages, and group dispersed evidence; 2) evidence-aligned retrieval matches this evidence to technique profiles through the same concepts, retaining likely candidates and confusable alternatives for LLM comparison of distinguishing behaviors. Controls at both stages validate evidence references and check technique-specific requirements. The representation also supports adaptation without parameter updates, where offline agentic audits diagnose concept weaknesses and refine extraction or retrieval policies. We evaluate MEvince against 11 baselines spanning four method families using a dataset of 610 public malware reports referenced by MITRE ATT&CK. MEvince improves exact-ID F1 over the strongest baseline by 17.4%, and its concept-guided retrieval increases gold-label candidate coverage from 31.6% to 47.9%. With a small adaptation set, offline refinement yields further gains and outperforms baselines that update model parameters. 

---
# Localization Does Not Imply Steerability: A Mechanistic Study of Sign Errors in LLM Mathematics 

**Authors**: Shradha Agarwal, Tariq J., Deepak Rajbhar  

**Link**: [PDF](https://arxiv.org/pdf/2611.07478)  

**Abstract**: When large language models do multi-step mathematics, a characteristic failure recurs: the model writes one wrong sign token mid-computation -- a cofactor sign, the minus of a recursive integration-by-parts step -- then executes the rest of the arithmetic correctly, and the answer comes out wrong. We ask where inside the network that sign is written. Across six open models spanning 14B to 675B parameters, a per-layer readout shows the wrong sign is committed late -- by MLP layers in the last quarter of the network (70-100% per model and task), a pattern present, where tested (Qwen), in the base model before instruction tuning. The mechanism varies: some models carry fixed "minus-stamp" layers that push a minus regardless of the right answer, others carry amplifiers of whichever sign was already chosen, and Qwen adds a downstream rescue layer that opposes the stamp. The circuit is causal: subtracting a single sign direction from the residual stream flips 13 of 14 held-out minus-written integration-by-parts errors on Llama-3.3-70B, with flips replicating on Qwen, Gemma, and Mistral-Small (24-70% per cell), and with representation-level specificity -- the same dose breaks correctly-minus answers while sparing correct-plus answers where the direction is surgical (0/72 on Llama), and matched random-direction and control-layer interventions are null. Core claims were preregistered with pass criteria frozen in git before held-out data was seen: 23 of 29 checks passed, all misses reported. But causality is not uniform: Phi-4 implements an equally well-localized circuit at the same late depth, yet the identical intervention flips almost nothing -- the sign is written redundantly across ~15 layers with margins several times larger (median 17.8 logits). Localization, we conclude, does not imply steerability: two models can share a circuit location yet differ enormously in how overturnable it is. 

---
# FINCOSMOS: An Open-World Evaluation Paradigm for Agents in Finance 

**Authors**: Yupeng Cao, Xueqing Peng, Lingfei Qian, Haohang Li, Yangyang Yu, Anke Xu, Wenbo Cao, Yan Wang, Yuechen Jiang, Yuyan Wang, Zhuohan Xie, Yankai Chen, Zehan Li, Jimin Huang, Xue Liu, Jian-Yun Nie, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2611.07464)  

**Abstract**: Language agents increasingly work for users on the open Web, and finance is where trusting them is hardest. The data must come from the Web, the strategy lives in the user's head, and correctness depends on time, so an agent can fail silently: a strategy the user never asked for, unverified data, or a backtest that uses the future yields a report as convincing as a correct one. Existing benchmarks close the world somewhere, fixing the task, the data, or the timeline, and thereby remove these failures instead of measuring them. We introduce FINCOSMOS, an open-world evaluation paradigm for agents doing financial work. A simulated user holds a hidden strategy sampled from a compositional grammar and reveals it only when asked. The agent receives no data, so it must acquire data from the Web, build its own backtest under a data cutoff, and deliver verifiable results. Grading needs no reference answer. Programs verify point-in-time correctness and recompute every reported metric, the delivered strategy is checked against the hidden specification, and a run is accepted only when the user is satisfied and none of 15 veto checks fails. Across four coding agents and a model without tools in 150 sessions, no session is accepted: every session that the simulated user signed off fails a program check, and agents raise only 54--64\% of the details they were not told. Human experts attribute 88.33\% of failures to the agents. With FINCOSMOS feedback, an agent with persistent memory learned all ten tasks. 

---
# EnvOpt: Automated Optimization for Agent Environments 

**Authors**: Mingxuan Li, Huanzhi Mao, Qiuyang Mang, Hanchen Li, Joseph E. Gonzalez, Alex Dimakis  

**Link**: [PDF](https://arxiv.org/pdf/2611.07372)  

**Abstract**: An LLM agent's cost and success depend on its model, its harness, and its environment. In addition to the widely studied directions of parameter tuning and harness optimization, we propose the third direction: optimization over stationary environments. Agents are often deployed in rich environments shared by many tasks, such as a company's Slack workspace or a complex collection of data sources, documents or tickets. We call these environments stationary environments. We introduce EnvOpt, a method for automated end-to-end optimization of stationary environments for any given model and harness, using real or synthetic tasks without needing any graders or reference answers. On AutomationBench, EnvOpt reduces the cost of Claude Haiku 4.5 by 72.7% and raises its pass rate by 14.34%. The same environment transfers to Opus 5, cutting its cost by 68.1% with a 1.34% gain. Added on top of Meta-Harness's optimized harness for Haiku 4.5, it lowers cost by a further 83.4% and adds 11.17% accuracy over Meta-Harness alone. Across 5 benchmarks, EnvOpt significantly reduces the cost of agents while improving or maintaining pass rate, and is transferable to different models and different harnesses. We also discuss common successful and rejected environment optimization strategies during automated optimization. More broadly, our results show that how information is represented to an agent shapes both its cost and its success, making the environment worth optimizing in addition to the parameters and the harness. 

---
# $D^2$-OPD: Separating Direction and Distance in Extrapolative On-Policy Distillation 

**Authors**: Bowen Fang, Dingmin Wang, Hengzhi Pei  

**Link**: [PDF](https://arxiv.org/pdf/2611.07241)  

**Abstract**: On-policy distillation (OPD) trains a student model to match a teacher's next-token distributions on student-generated prefixes. Extrapolative OPD further extends the teacher's log-probability shift relative to a reference policy to construct stronger learning targets. However, existing methods couple extrapolation direction and distance, allowing the magnitude of the teacher's shift to dictate how far the target moves. We propose $D^2$-OPD, decoupling these two factors: a per-prefix hard KL constraint bounds the target's distance from the teacher, while an adaptive combination of the teacher's post-training shift and the initial student--teacher gap determines the extrapolation direction. We optimize the student toward this target using forward KL, providing learning signals even for low-probability tokens. We evaluate three settings: where standard OPD performs well, where it struggles with base-model initialization, and where no teacher reference is available. On mathematical reasoning benchmarks, $D^2$-OPD attains the highest score in all three settings, outperforming the strongest baseline by up to 9.45 points. Ablations validate the contributions of each component. 

---
# Compositional Unalignment: Auditing Safety in Large Vision-Language Models 

**Authors**: Hong kyu Lee, Kai Shu, Li Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2611.07216)  

**Abstract**: Large vision-language models (LVLMs) are vulnerable to compositional harm: an image and a prompt that are benign in isolation become harmful through the model's cross-modal reasoning. Recent alignment methods report strong scores on compositional safety benchmarks, but an LLM judge only measures observed behavior and cannot reveal how robustly safety is integrated into the model. We argue that un-alignment provides a complementary audit of alignment robustness. We first show that minimal supervised fine-tuning reaches an LLM-judge score comparable to a multi-stage reinforcement-learning pipeline, so similar benchmark scores cannot establish robustness by themselves. We then show that recognition and refusal of compositional harm are decoupled: nine standard direction-finding methods detect compositional harm with up to AUC=0.99, yet ablating the recovered directions either leaves refusal intact or destroys model coherence. We therefore propose CMaG, a training-free un-alignment method for compositional safety. CMaG measures refusal intention with a refusal-versus-compliance log-probability margin, takes its gradient with respect to mid-layer activations, and contrasts it against partially benign pairs to isolate the cross-modal interaction term. A single backward pass over 80 examples yields a direction along which steering monotonically converts refusal into compliance. On Qwen2.5-VL-7B and LLaVA-1.5-7B, CMaG reduces compositional safety scores by up to 65 points while over 98% of outputs remain coherent, outperforming gradient-based baselines; the direction also transfers to text-explicit harm and un-aligns models hardened by defensive training. Compositional safety can thus appear behaviorally strong while remaining fragile to targeted internal intervention. Warning: this paper contains unsafe model outputs, including references to self-harm. 

---
# Closed-Loop Agent Evaluation: Layered Measurement and Self-Calibration 

**Authors**: Ruifeng Ren, Gun Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2611.07051)  

**Abstract**: Once an agent is live, teams want to know whether it is doing its job, not just whether it throws errors. Yet most evaluation is one-off: a benchmark run, a look at traces, a single score. Its conclusions neither check themselves nor reach the agent. We propose closed-loop evaluation, an evaluation method for deployed agents. It models evaluation as performance estimation under biased raters and censored outcomes, and it requires every level's conclusion to trace back to atomic data and to be computed by versioned rules. The method covers four levels of analysis (span, trace, conversation and engagement), measures along a spine of Execution, Substance, Trajectory and Results, treats Conduct as a constraint throughout, and uses Provenance and Adjudication components to certify the evidence and the evaluators. Evaluator agents generated by an Agent Factory carry these components, each with a domain knowledge pack and an adapter harness. Two loops close the evaluation: an inner loop audits the evidence, rules and metrics themselves, and an outer loop uses per-component regression checks to drive agent improvement. We also give a pipeline that runs this process automatically. Using the business-analysis agent SharpInsight as a case and porting the method to public conversation data, we show how problems differ across levels and how evidence-and-rule judgment is more reliable than holistic scoring (ICC 0.86 vs. 0.81; kappa 0.81 vs. 0.63). 

---
# LowRAM: Low-Rank Keys and Amortised Values for Composable KV Cache Compression at Test-Time 

**Authors**: Teresa Delgado, Hugo Hazard, Martin Benfeghoul, Jun Wang, Zafeirios Fountas  

**Link**: [PDF](https://arxiv.org/pdf/2611.07025)  

**Abstract**: Autoregressive decoding relies on the KV cache, whose memory footprint grows linearly with context length and caps batch size and throughput. Existing post-training methods exploit redundancy across tokens, layers, and feature dimensions, or lower numerical precision. We identify a complementary axis: within each attention head, values are largely predictable from their corresponding keys. We exploit this with value cache amortisation, which fits a lightweight per-head predictor at test time and stores residuals only for the tokens it reconstructs worst, giving $4\times$ near-lossless value compression. The predictor can be initialised by meta-learning for substantially reduced prefill overhead, or fitted from scratch in settings that allow a one-off cost, such as multi-query inference. We pair value amortisation with cross-layer low-rank key compression to produce LowRAM, an asymmetric compression pipeline that stays close to lossless across three model families. By reconstructing only a selected subset of KV pairs at each decoding step, LowRAM can keep decoding overhead minimal and achieve up to $5\times$ higher peak decode throughput at $4\times$ compression. LowRAM further composes with quantisation and eviction, reaching $26.9\times$ compression at $94.5\%$ of baseline performance on long contexts. These results show that treating keys and values as structurally distinct enables efficient, aggressive, and composable KV cache compression. 

---
# BlockLock: Reproducible LLM Inference with Fixed Batch Geometry 

**Authors**: Xiuyuan Qi, Pedro Reviriego, Javier Conde, Shanshan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.06977)  

**Abstract**: Reproducible LLM inference requires that a request produces the same output regardless of the other requests in the batch. Existing solutions pay for this in different ways: batch-invariant kernels constrain every operator, and isolating prefill with fixed-shape decode ({our CoRun-style baseline}) gives up shared prefill compute and pads decode. We present BlockLock, which keeps shared batched prefill and pins only the numerical path of the protected request: it fixes the geometry that the request sees, with {two modes: BlockLock-Rows uses fixed-width rows and retains batched decode of all requests; BlockLock-Pack uses terminal-anchored packing with an isolated dummy segment, then extracts the protected request's KV state for target-only decode. Under a stated per-deployment contract, both modes preserve the target's output bitwise when its companions change.} Across five {dense} models (Qwen3-8B/14B, Llama-3.1-8B-Instruct, and Gemma-3-4B/12B-IT) and three single-GPU deployments (NVIDIA A6000, A100, and H100), we evaluate our approach on both tinyBenchmarks and BFCL{; we also include initial tensor-parallel experiments on two NVIDIA A6000 GPUs. Both modes preserve target sequences and Top20-logit bits in the reported tests.} At equal delivered work, {BlockLock-Rows} reduces latency by 53.5--55.0\% and improves throughput by 115.2--122.2\% relative to vLLM Batch-Invariant; compared with batch-invariant kernels, it needs no custom kernel implementations and constrains only the protected requests. 

---
# Octopus: One Brain, Many Branches for Latent-KV Multi-Agent Reinforcement Learning 

**Authors**: Tianming Sha  

**Link**: [PDF](https://arxiv.org/pdf/2611.06956)  

**Abstract**: Multi-agent LLM systems explore in parallel by spawning sub-agents with isolated context windows, whereas single-agent systems append every search and verification step into one growing trajectory. However, existing multi-agent frameworks duplicate prefix KV caches across branches, exchange findings by re-prefilling text summaries, and flatten or drop cross-branch dependencies during reinforcement learning (RL).
We introduce Octopus, a single-model multi-agent RL framework in which all threads share one backbone and coordinate over a GPU-resident directed acyclic graph (DAG) of KV states through five primitives (fork, read_thread, wait_threads, rewind, commit). Octopus combines three co-designed mechanisms: (1) Zero-Copy Radix Forking, which spawns branches as $\mathcal{O}(1)$ cursor views over shared physical KV pages; (2) Attention-Only KV Mounting, which mounts a sibling thread's terminal KV blocks directly into full-attention layers while bypassing linear-recurrent states (Gated DeltaNet / Mamba-2), cutting off-task hijacking by $21\times$ ($57.9\% \to 2.7\%$) while retaining $92.2\%$ of full-text mutual information at zero extra KV allocation; and (3) a Packed DAG-GRPO Trainer, which packs all episode branches into one block-sparse attention pass and assigns counterfactual branch credit via leave-one-branch-out marginal rewards.
Trained end-to-end with GRPO on Qwen3.5-4B over a sparse BM25 corpus, Octopus reaches $49.6\%$ accuracy on BrowseComp-Plus ($+17.5$ points over single-agent GRPO at 143K context and $+12.5$ points over text-summary multi-agent RL) and $51.2\%$ on OpenResearcher, outperforming dense-retrieval agents up to $250\times$ larger while increasing training throughput $3.3\times$ and cutting peak KV memory $2.8\times$. 

---
# XBridge: Entity-Grounded Latent Bridge for Heterogeneous LLM Communication 

**Authors**: Wooseong Yang, Wei-Chieh Huang, Weizhi Zhang, Yu Wang, Philip S. Yu, Junhyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.11676)  

**Abstract**: Heterogeneous multi-agent LLM systems, where agents are powered by different model families, can outperform homogeneous configurations by reducing redundant reasoning patterns. Yet existing communication protocols either operate through text, discarding the sender's internal representations, or require architectural homogeneity for latent-level transfer. We identify the entity grounding problem in cross-architecture communication: cross-attention bridges that transfer continuous representations across different LLM families suffer from rare-token compression collapse, where entity identity is lost in the continuous bottleneck (bridge-only F1 ~30%). We propose XBRIDGE, a decode-free communication protocol that addresses this through two mechanisms. Lexical Anchor Mapping (LAM) maps the sender's original context tokens to the receiver's vocabulary, providing discrete entity anchors. A Latent Enrichment Bridge (LEB) lets the receiver query the sender's hidden states for contextual enrichment. The entity anchors ground the bridge's contextual signals to specific entities through the receiver's own self-attention. Across three model families (Llama, Qwen, and Mistral), seven benchmarks, and both communication directions, XBRIDGE outperforms text-based communication on all seven tasks for each model pair while achieving 11x lower latency, and in a same-architecture setting it also exceeds a KV-sharing baseline on six of seven tasks. LEB requires only 264M trainable parameters (3.8% of the receiver), is trained on a small balanced sample set, and adds negligible inference overhead. 

---
