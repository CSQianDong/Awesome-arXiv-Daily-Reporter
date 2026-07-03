# LACUNA: A Testbed for Evaluating Localization Precision for LLM Unlearning 

**Authors**: Matteo Boglioni, Thibault Rousset, Siva Reddy, Marius Mosbach, Verna Dankers  

**Link**: [PDF](https://arxiv.org/pdf/2607.02513)  

**Abstract**: LLMs memorize sensitive training data, including personally identifiable information (PII), creating a pressing need for reliable post hoc removal methods. Unlearning has emerged as a promising solution, with state-of-the-art(SOTA) methods often following a localize-first, unlearn-second paradigm that targets specific model parameters. However, existing benchmarks evaluate unlearning solely at the output level, leaving open the question of whether unlearning truly erases knowledge from a model's parameters or merely obfuscates it, a concern reinforced by the success of resurfacing attacks. To bridge this gap, we introduce LACUNA: the first unlearning testbed with ground-truth parameter-level localization. LACUNA injects PII of synthetic individuals into predefined parameters of 1B and 7B OLMo-based models via masked continual pretraining, enabling direct evaluation of whether unlearning targets the weights responsible for knowledge storage. We use LACUNA to benchmark current SOTA unlearning methods and find that, despite strong output-level performance, existing methods are highly imprecise and susceptible to resurfacing attacks. We further show that when localization is successful, even a simple gradient-based unlearning method achieves strong erasure and robustness to resurfacing attacks, highlighting the importance of precise unlearning. We release LACUNA to complement behavioral evaluations and drive further advances in robust, localization-based unlearning. 

---
# Reasoning LLM Improves Speaker Recognition in Long-form TV Dramas 

**Authors**: Yuxuan Li, Lingxi Xie, Xinyue Huo, Jihao Qiu, Jiacheng Shao, Pengfei Chen, Jiannan Ge, Kaiwen Duan, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2607.02504)  

**Abstract**: Long-form TV dramas present a formidable challenge for comprehensive video understanding, where deciphering complex storyline often relies on \textbf{speaker recognition}, the task of accurately attributing each spoken utterance to its respective character. In this paper, we advance this field through two primary contributions. (1) We introduce \textbf{DramaSR-532K}, a large-scale benchmark comprising 532K annotated dialogue lines across more than 900 unique characters, necessitating the integration of auditory, linguistic, and visual cues for speaker recognition. (2) We propose \textbf{DramaSR-LRM}, a robust approach built upon a large reasoning model (LRM). DramaSR-LRM is designed to autonomously aggregate contextual evidence via multimodal tool-use, synthesizing diverse inputs to achieve high-fidelity attribution. Experimental results demonstrate that DramaSR-LRM significantly outperforms existing baselines, particularly on short utterances where acoustic biometrics are inherently unreliable. \textit{All the data and code will be made publicly available at the project page: this https URL.} 

---
# Visually Grounded Self-Reflection for Vision-Language Models via Reinforcement Learning 

**Authors**: Liyan Tang, Fangcong Yin, Greg Durrett  

**Link**: [PDF](https://arxiv.org/pdf/2607.02490)  

**Abstract**: Large vision-language models can reason over multimodal inputs by generating textual chains of thought (CoT). A key capability exhibited in CoT reasoning is self-reflection: revisiting earlier decisions and correcting previous errors. However, existing LVLMs often fail to properly attend to visual inputs during reflection, limiting their ability to translate feedback into grounded corrections, especially for out-of-distribution images. To address this issue, we propose a novel reinforcement learning training framework VRRL, with two components explicitly designed to elicit visually grounded self-reflection. First, we randomly mask trajectory prefixes during training to emphasize recovery from incorrect intermediate predictions rather than making early mistakes. Second, we introduce buffered roll-ins from an experience replay buffer to expose the model to diverse failure states that it must learn to correct. We evaluate our approach on visual grounding tasks involving tables and charts, as well as spatial navigation benchmarks. While off-the-shelf and conventionally fine-tuned models degrade substantially under distribution shift, our method substantially improves average out-of-distribution accuracy over standard RL and reflection-oriented fine-tuning baselines by using self-reflection effectively. 

---
# Audio-Based Understanding of Audiobook Narration Appeal 

**Authors**: Shahar Elisha, Mariano Beguerisse-Díaz, Emmanouil Benetos  

**Link**: [PDF](https://arxiv.org/pdf/2607.02473)  

**Abstract**: Narration is central to the audiobook listening experience, shaping how listeners engage with and understand the content. This work explores how narration qualities shape an audiobook's appeal, noting that their effects can vary by genre, title, and audience. We extract vocal and acoustic features (e.g., tone, pace, loudness) from LibriVox using pre-trained audio models and analyse their relationship with consumption data (specifically, view-rate) and their interplay with genre and title. Despite limited consumption data, we find that acoustic information alone has a robust association with appeal, even after accounting for title effects. We further validate these findings using more nuanced proprietary engagement metrics. To our knowledge, this is the first systematic computational study linking narration qualities, genre, title, and audiobook consumption, highlighting the potential of data-driven insights to improve audiobook personalisation and narrator casting. 

---
# Will Scaling Improve Social Simulation with LLMs? 

**Authors**: Caleb Ziems, William Held, Su Doga Karaca, David Grusky, Tatsunori Hashimoto, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02464)  

**Abstract**: Large Language Model (LLM) social simulations are a promising research method, but they are not yet faithful enough to be adopted widely. In this work, we investigate whether the current scaling paradigm in language modeling is likely to close these gaps, or whether simulation fidelity is orthogonal to general capabilities and therefore deserving of more research attention. We use scaling laws to study the relationship between LLMs' compute scale, general capability benchmarks, and the fidelity of social simulation in three representative sub-domains: opinion modeling, behavioral simulation, and longitudinal forecasting. Surprisingly, we discover strong compute scaling in all three settings, using a suite of 85 transformer LLMs with the Qwen3 architecture pre-trained on the DCLM web text corpus under fixed-compute budgets from $10^{18}$ to $10^{20}$ FLOPs. Then we evaluate 35 larger and more capable open-weight models up to 70B parameters, allowing us to predict downstream accuracy from loss. This reveals that the majority of behavioral and opinion simulation tasks will rapidly improve with scale, particularly when they involve populations that are well-represented in English web corpora. Longitudinal forecasting and underrepresented opinions scale more slowly, especially when they are less correlated with general knowledge and reasoning benchmarks like MMLU. In behavior simulation, scaling fails to improve model calibration with human cognitive biases like risk aversion, as well as human heuristics like learning correlated rewards from related tasks. On these tasks, even fine-tuned models fail to noticeably scale up performance from 0.5B to 8B parameters. Taken together, we conclude that scale will improve social simulations in most settings, but outliers exist, and improvements will be less reliable in low-resource domains. 

---
# Language Models as Measurement Apparatus for Culture 

**Authors**: Kent K. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02459)  

**Abstract**: Language models are increasingly used to quantify cultural phenomena, but what makes such measurement distinctively cultural? This paper argues that NLP work on culture is a material-discursive practice: the apparatus -- model, data, annotation, evaluation -- participates in constituting the cultural reality it measures, rather than passively recording it. Drawing on Karen Barad's concept of the agential cut -- the contingent boundary between phenomenon and instrument -- I show that the apparatus's substantive design choices draw such boundaries, and that the boundary is entangled from the start because language models have already internalized much of the cultural material they measure. I illustrate this through three case studies on television and film dialogue (measuring structure, interaction, and deviation) and three examinations of the apparatus itself (erasure of cultural markers, attunement to historical material, and agency in an agentic workflow). This big picture analysis proposes a research program that is theory-driven, empirically rigorous, and culturally contingent, treating each agential cut as a conscious commitment, at once methodological and ethical. 

---
# The Future of NLP may not be at NLP Conferences: Scholarly Migration Patterns in Natural Language Processing 

**Authors**: David Jurgens  

**Link**: [PDF](https://arxiv.org/pdf/2607.02416)  

**Abstract**: Natural Language Processing (NLP) has traditionally been published in its core disciplinary venues like ACL. However, advances in Large Language Models (LLMs) has led to a blurring of the disciplinary lines between NLP and general Machine Learning (ML), with authors regularly publishing in venues from both fields. Here, we ask whether the disciplinary center of gravity is shifting. Using NLP research published from 2010 to 2026 and studies of both established and new authors, we find that a migration is taking place. First, comparing the pre- and post-LLM eras, established authors lost 19.2pp of share at flagship *ACL main-conference tracks while gaining 14.8pp in the newer Findings tracks, and general ML venues rose 8.6pp, even when adjusting for parallel growth in the fields. Second, among newer authors who debut with at least three first-author NLP-topic papers, the share whose work appears mostly at *ACL venues fell from 84% (2019) to 74% (2024), while the share appearing mostly at general ML venues rose from 5% to 21%. Using causal inference techniques, we estimate that these general ML venues confer a significant citation premium, which influences venue selection. Together, these results point to a significant shift in where NLP research is published. 

---
# Know Your Source: A Public Knowledge Store for Media Background Checks 

**Authors**: Benjamin Nichols, Michael Schlichtkrull, Nedjma Ousidhoum  

**Link**: [PDF](https://arxiv.org/pdf/2607.02383)  

**Abstract**: LLM-based retrieval-augmented generation (RAG) is increasingly used for automated fact-checking (AFC) and related tasks. By grounding LLM outputs in retrieved evidence, RAG-based systems provide transparent justifications while allowing external information to be updated independently of the underlying model. However, existing approaches often assume retrieved evidence is reliable, although real-world information may be conflicting, outdated, and can originate from unreliable or biased sources. Recent work on *source-critical reasoning* addresses this challenge through media background checks (MBCs) (Schlichtkrull, 2024), which assess the credibility of evidence sources to support downstream fact verification. However, generating MBCs relies on costly proprietary search APIs, limiting reproducibility. To mitigate this issue, we introduce MEDIAREF, a publicly available knowledge store of web-sourced documents that enables reproducible, low-cost evaluation of MBC generation across 200 media sources. We describe a reproducible methodology for constructing and updating the collection, assess widely used LLMs on the MBC generation task, and demonstrate that MEDIAREF supports higher-quality MBC generation through both automatic and qualitative evaluation. 

---
# HULAT2 at MER-TRANS 2026: Governed Multi-Agent Simplification for Spanish Easy-to-Read Generation 

**Authors**: Lourdes Moreno, Paloma Martínez, Marco Antonio Sanchez-Escudero, Miguel Domínguez-Gómez  

**Link**: [PDF](https://arxiv.org/pdf/2607.02381)  

**Abstract**: This paper describes the participation of HULAT2-UC3M in the Spanish track of MER-TRANS 2026, a shared task on multilingual Easy-to-Read translation. Three fully automatic Spanish runs were submitted. RUN1 and RUN2 used a LangGraph-based multi-agent workflow combining Gemini 2.5 Flash and RigoChat-7B-v2, parallel generation strategies, internal quality signals, Event-Condition-Action routing, controlled editing and traceable decisions. RUN1 used the base workflow, while RUN2 activated an additional lexical-support layer based on a glossary and lexical resources. RUN3 was a RigoChat-based generate-evaluate-regenerate baseline with prompt engineering and LoRA-based adaptation. The official leaderboard reports BLEU-Orig, BLEU-Gold, SARI and BERTScore. During development, additional internal signals were also inspected, including semantic fidelity, readability, lexical simplicity, syntactic clarity and factual consistency. According to official SARI, RUN1 was the best HULAT2 run, with 44.0543 points, followed by RUN2 with 43.1049 and RUN3 with 38.5136. These results indicate that, in this task setting, signal-guided multi-agent routing outperformed the linear regeneration baseline. They also show that adding lexical support did not automatically improve reference-based scores. Further segment-level and document-level analysis are required to assess readability, factual consistency and user-oriented adequacy. 

---
# World Wide Models: Literary Tools for Cultural AI 

**Authors**: Nina Begus  

**Link**: [PDF](https://arxiv.org/pdf/2607.02369)  

**Abstract**: LLMs stage a new form of cultural encounter that is massive, automated, and monolingual. Literary disciplines have always negotiated cultural struggles with comparative reading of literature, narratological and poetic analysis, critical theory, world literature, and translation. These tools have now become indispensable for building culturally literate AI. The essay develops a layered framework toward more nuanced textual models and pluralistic interpretations of AI, emphasizing the natural intersections of literature and AI development, connecting current debates in critical theory with structural monolingualism, and suggesting a new application of world literature approaches to address global AI textuality through macrostructure, circulation, and untranslatability. 

---
# On the Role of Directionality in Structural Generalization 

**Authors**: Zichao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2607.02307)  

**Abstract**: Several SLOG test categories explicitly involve directional distinctions (modifier position shifts, argument extraction positions), yet AM-Parser, the previous SOTA, uses an AM algebra whose operations do not encode direction. We redesign the symbolic backend around CCG directed types (deterministic CKY + single linear decoder, 30K learnable parameters). Under the same BERT-base encoder, the system achieves 75.9$\pm$6.4% LF exact match, surpassing AM-Parser (70.8$\pm$4.3%). Per SLOG's own category groupings, gains are highly directional: the CCG system outperforms AM-Parser on all 5 position-shift categories (+29.9pp), while AM-Parser outperforms on all 6 recursive-depth categories. Replacing the encoder with DeBERTa-v3-large yields 90.7$\pm$4.9%, with the largest encoder gains in recursive-depth categories, complementary to directionality's gains. Directional representations shift the bottleneck from the symbolic layer (AM-Parser's 0% category ceiling) to the neural layer, which improves with encoder upgrades. 

---
# CheckRLM: Effective Knowledge-Thought Coherence Checking in Retrieval-Augmented Reasoning 

**Authors**: Dingling Xu, Ruobing Wang, Qingfei Zhao, Yukun Yan, Zhichun Wang, Daren Zha, Shi Yu, Zhenghao Liu, Shuo Wang, Xu Han, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.02262)  

**Abstract**: Reasoning Language Models (RLMs) have significantly improved performance on complex tasks by extending the reasoning chain. However, these chains are prone to containing factual errors, particularly in knowledge-intensive tasks. To address this issue, we propose CheckRLM, a framework that improves the reliability of the reasoning process through Retrieval-Augmented Generation (RAG) by timely checking and correcting factual errors. Specifically, CheckRLM extracts factual claims from the reasoning chain to identify and localize subtle knowledge inconsistencies during inference. Upon detection of errors, a refinement mechanism performs minimal-cost yet precise corrections by leveraging external knowledge, ensuring coherence between the reasoning chain and correct knowledge. Extensive experiments demonstrate that CheckRLM substantially outperforms existing baselines, exhibiting a strong capability to mitigate error accumulation in long-horizon reasoning with lower costs. The code and data are available at this https URL. 

---
# BamiBERT: A New BERT-based Language Model for Vietnamese 

**Authors**: Dat Quoc Nguyen, Thinh Pham, Chi Tran, Linh The Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2607.02259)  

**Abstract**: In this paper, we introduce BamiBERT, a new BERT-based pre-trained language model for Vietnamese that addresses key limitations of PhoBERT -- the current de facto Vietnamese text encoder. Trained from scratch on a 129GB corpus of general-domain Vietnamese text for 20 epochs, BamiBERT supports an extended context length of up to 2048 tokens and operates directly on raw input, eliminating the need for external word segmentation. Across 8 Vietnamese benchmarks, it achieves the best score on 11 of 15 metrics and the second-best on 3 others, setting a new state of the art among "base"-sized Vietnamese encoders and demonstrating strong cross-domain generalization. We release BamiBERT at: this https URL 

---
# Challenges and Recommendations for LLMs-as-a-Judge in Multilingual Settings and Low-Resource Languages 

**Authors**: A.Seza Doğruöz, Xixian Liao, Verena Blaschke, Jakob Prange, Senyu Li, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2607.02235)  

**Abstract**: LLM-as-a-Judge has become the dominant evaluation paradigm for many natural language generation tasks, due to shortcomings of conventional metrics and high correlations with human judgment, albeit mostly in English. There are now attempts to extend LLM-as-a-Judge to multilingual settings including low-resource languages. However, LLMs have limited proficiency in low-resource languages, and there is often no adequate human validation in these settings. To highlight the scope of the problem and current practices, we explore the use of LLM-as-a-Judge evaluators in ACL Anthology papers focusing on multilingual settings and low-resource languages across a diverse set of tasks. Out of 650 papers mentioning LLM-as-a-judge, only 33 of them focus on low-resource or multilingual settings. Our in-depth analysis of these papers indicates inconsistent evaluation outcomes, a tendency to overtrust LLM judgments in multilingual settings, and the widespread reliance on a single judge model per study. To help the NLP community further, we conclude with recommendations about how to use LLM-as-a-Judge in multilingual and low-resource settings. 

---
# Unlocking Speech-Text Compositional Powers: Instruction-Following Speech Language Models without Instruction Tuning 

**Authors**: Congrui Du, Yang Zhang, Kaizhi Qian, Shiyu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02214)  

**Abstract**: Instruction tuning for speech language models (SLMs) is substantially more challenging than for text-based large language models (LLMs), as it requires learning a new modality and a wide range of speech-specific instructions in addition to those supported by text LLMs. Existing SLM training approaches largely replicate the text LLM training paradigm by synthesizing large-scale speech pre-training and instruction-tuning datasets. However, this strategy is difficult to scale, since speech sequences are significantly longer than text sequences. In this paper, we propose SpeechCombine, an instruction-following speech language model trained without any instruction tuning, using only a single round of speech pre-training on 30k hours of data. Starting from a text LLM base model, we perform continuous pre-training on speech utterances to obtain a speech-adapted model, and then directly combine its weights with the weight difference between the instruction-tuned and base versions of the text LLM. Our results show that this simple combination strategy not only preserves the knowledge and capabilities of the original text LLM, but also effectively transfers them to the speech domain. These findings suggest a new direction for SLM training that avoids reliance on massive speech data. 

---
# HaloGuard 1.0: An Open Weights Constitutional Classifier for Multilingual AI Safety 

**Authors**: Navaneeth Sangameswaran, Preetham S, Ashmiya Lenin  

**Link**: [PDF](https://arxiv.org/pdf/2607.02079)  

**Abstract**: We present HaloGuard 1.0, an open-weights implementation of the constitutional-classifier paradigm for input safety. It achieves state-of-the-art performance on English and multilingual prompt-safety benchmarks at roughly one-tenth the model size of current leading open guard models. The safety constitution is the organising structure of the corpus: a natural-language constitution of 46 policies and 2,940 subcategories drives synthetic data generation, with exhaustive one-to-one paired counterfactuals that hold topic and vocabulary fixed while flipping intent, a two-tier harmless design that separately targets boundary and baseline false positives (FPs), and balanced multilingual materialisation across 46 languages that treats language as a surface form appearing on both sides of the boundary rather than as an adversarial signal. Across seven prompt-safety benchmarks, HaloGuard 1.0-0.8B attains the best average F1 (90.9) of any open guard we evaluate, outperforming baselines up to 27B parameters (over 30 times larger) while holding false-positive rate (FPR) to 4.3 and false-negative rate (FNR) to 9.5. The HaloGuard 1.0-4B variant reaches average F1 of 92.1 and FPR of 3.5, spending its extra capacity on precision rather than recall. A structured adjudication of the remaining failures indicates that most apparent missed-harm cases are benchmark mislabels rather than genuine model misses. An always-on adversarial red-teaming protocol continuously hardens the guard against both content-level and agentic attacks. We release the models as open weights. 

---
# SPLIT: Cross-Lingual Empathy and Cultural Grounding in English and Ukrainian LLM Responses 

**Authors**: Anna Chorna  

**Link**: [PDF](https://arxiv.org/pdf/2607.02049)  

**Abstract**: Large Language Models are increasingly deployed in emotional-support contexts and crisis-related situations. Nevertheless, their cross-lingual abilities in these circumstances remain underexplored. Existing benchmarks emphasize multilingual performance but rarely examine crisis-related empathy and cultural grounding in low-to-mid-resource languages. We introduce SPLIT, a 500-prompt benchmark designed to evaluate LLM consistency in generating emotionally grounded responses across five categories: Stress, Panic, Loneliness, Internal Displacement, and Tension. We evaluate three technically diverse LLMs across three dimensions: Empathetic Accuracy, Linguistic Naturalness, and Contextual & Cultural Grounding. The framework aims to assess and compare the quality of LLM responses in both English and Ukrainian languages, as well as to explore the reliability of the LLM-as-a-jury paradigm. Our findings reveal that Gemini-2.5-Flash and LLaMA-3.3-70B-Instruct degrade when transitioning to Ukrainian, while DeepSeek-V3 remains comparatively stable within our benchmark. We additionally find that human and AI evaluators agree weakly on empathy and naturalness but diverge on cultural grounding. We further argue that producing Ukrainian text is not equivalent to producing Ukrainian emotional support. Our findings may assist in the future development of more culturally tailored benchmark designs, as well as encourage a stronger emphasis on human-centered evaluation. 

---
# OpenSafeIntent: Evaluating Intent-Calibrated Safe Completion Across Dual-Use Prompt Sets 

**Authors**: Rheeya Uppaal, Seungwoo Lyu, Selina Sung, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2607.02047)  

**Abstract**: Safe completion requires models to provide useful assistance without enabling harm, but this behavior is difficult to evaluate with isolated prompts. We introduce OpenSafeIntent, a benchmark of controlled prompt-sets that vary intent while holding the underlying task fixed. Each datapoint contains benign, dual-use, and malicious variants of the same task. This design lets us evaluate whether models calibrate assistance across intent shifts, rather than merely appearing safe on average. Across a broad model suite, we find that prompt-level safety hides important failures: models often fail to remain safe across matched intent variants, dual-use behavior is brittle under paraphrase, high-level answers on risky topics are not reliably safe, and responses that reframe ambiguous requests into safer tasks are substantially less likely to cross the safety boundary. Our results suggest that safe completion should be evaluated as intent-calibrated behavior over controlled task variants, not as a single safety-helpfulness tradeoff over independent prompts. 

---
# EduArt: An educational-level benchmark for evaluating art history knowledge in large language models 

**Authors**: Gianmarco Spinaci, Lukas Klic, Giovanni Colavizza  

**Link**: [PDF](https://arxiv.org/pdf/2607.02007)  

**Abstract**: Large language models now score near ceiling on general benchmarks, but these aggregate measures reveal little about how models behave within single disciplines. Existing art-focused evaluations rely on synthetic questions and rarely report item-level properties. This paper introduces EduArt, an educational-level benchmark for art-historical knowledge and visual reasoning in multimodal LLMs. EduArt comprises 871 human-authored questions from Italian secondary-school exercises and US Advanced Placement Art History exams, spanning two languages and seven formats from multiple choice to in-text word placement and error identification. Twelve models from six provider families were evaluated under a default answer-only condition and a motivation condition requiring written justification, and characterized using Classical Test Theory and a logistic regression isolating the effects of format, language, image presence, and model. The benchmark showed strong psychometric properties (mean discrimination 0.514, 82.3 percent good discriminators), while multiple-choice accuracy saturated near ceiling for six models, showing recognition formats alone cannot distinguish frontier models. Format was a strong independent predictor of accuracy: models exceeding 94 percent on multiple choice fell to 23.9 percent on open completion (Claude Opus 4.6) and 6.2 percent on error identification (Claude Sonnet 4.6). The motivation condition changed accuracy in a predominantly negative, family-dependent direction. These dissociations indicate that art-historical knowledge and the ability to deploy it are distinct capabilities, and that single-format benchmarks overestimate what models can reliably do. Mapping this capability profile is a precondition for responsible use of multimodal LLMs in art-historical scholarship, where tasks demand producing and manipulating content rather than selecting from fixed options. 

---
# Using embeddings to predict spoken word duration and pitch in Mandarin monosyllabic words 

**Authors**: Xiaoyun Jin, Mirjam Ernestus, R.Harald Baayen  

**Link**: [PDF](https://arxiv.org/pdf/2607.02002)  

**Abstract**: Time-normalized f0 contours of Mandarin words in conversational speech have been shown to be predictable in part from their contextualized embeddings (CEs). The present study investigates whether CEs also predict spoken word duration for 7470 tokens of Mandarin monosyllabic CV words extracted from a Mandarin corpus of spontaneous speech. We show that CEs indeed are predictive for duration, above chance level, not only at the type level, but also at the level of individual tokens, as indicated by the results obtained with the type-wise and token-wise permutation baselines. We also show that the predicted durations are sufficiently precise to back-transform predicted f0 contours in [0,1] normalized time to contours on the ms time scale. The resulting predicted contours approximate empirical contours and also outperform a permutation baseline. 

---
# Object Aligner: A Configurable JSON Schema Similarity Score for Graphs, Applied to LLM Prompt Optimization 

**Authors**: Jan Drchal  

**Link**: [PDF](https://arxiv.org/pdf/2607.01972)  

**Abstract**: Large language models (LLMs) are often asked to produce JSON conforming to a fixed schema, powering information extraction, tool calling, agentic planning, and knowledge-graph construction. Measuring how closely an output matches a gold reference is essential yet surprisingly hard: exact match is brittle, text similarity ignores structure, and an LLM judge is expensive, opaque, and non-deterministic. We address this with Object Aligner (OA), an open-source Python library that scores two JSON objects deterministically by recursively aligning their trees (the Hungarian algorithm for unordered collections, sequence alignment for ordered ones) and awarding partial credit at the granularity the schema declares. The Object Aligner is configured entirely through a set of JSON Schema extensions, so adapting it to a new task involves annotating a schema rather than writing code. Complex structured data, however, are rarely flat trees: records may form graphs or hypergraphs keyed by arbitrary identifiers, breaking the assumptions of prior similarity metrics. Our central contribution, referential alignment, closes this gap by inferring a bijection between gold and candidate identifiers and scoring every reference through it, so the score is invariant to relabeling. Since recovering this bijection exactly is graph isomorphism, the Object Aligner approximates it with Weisfeiler-Leman color refinement. An order-sensitive sequence regime targets ranking and planning. Since the same alignment localizes every mismatch, the Object Aligner emits ranked repair suggestions at no extra cost. Used as a reward inside the GEPA prompt optimizer, Object Aligner helps or stays neutral across all datasets. 

---
# Towards a Phonology-Informed Evaluation of Multilingual TTS 

**Authors**: Sneha Ray Barman, Neeraj Kumar Sharma, Shakuntala Mahanta  

**Link**: [PDF](https://arxiv.org/pdf/2607.01965)  

**Abstract**: Neural TTS systems can sound natural across languages, but naturalness does not guarantee the preservation of sound contrasts that distinguish words from their grammatical forms. Standard metrics like MOS do not test for this. We propose a classifier-based framework that audits TTS output against language-specific phonological patterns using human speech as a benchmark. Testing Assamese advanced tongue root (ATR) vowel harmony with Meta's MMS TTS, we show that a classifier trained on human speech transfers to synthesized speech with minimal loss. The faithfulness audit reveals that [+ATR] mid vowels are realized as [-ATR] in 1/3 tokens despite an underlying [+ATR] specification, a bias absent in human speech. At the word level, predicted ATR labels classify harmony more accurately than transcription labels, indicating a gap between intended and produced phonology. The framework offers task-specific diagnostics and generalizes to other phonological contrasts with measurable acoustic cues. 

---
# Beyond Supervised Clarification: Input Rewriting with LLMs for Dialogue Discourse Parsing 

**Authors**: Yiming Liu, Ziyue Zhang, Zhichao Xu, Xin Yu, Yingheng Tang, Tianyu Jiang, Jie Cao  

**Link**: [PDF](https://arxiv.org/pdf/2607.01964)  

**Abstract**: Rewriting inputs to improve frozen downstream models has become a common strategy in modern NLP pipelines. Prior work on incremental dialogue discourse parsing (DDP) shows that supervised clarification models can rewrite fragmentary or underspecified utterances, such as resolving ellipsis or references, to improve parsing accuracy. In this work, we revisit this idea under realistic deployment conditions, where no clarification supervision is available and the clarifier must rely on zero-shot prompting or feedback from a frozen parser. Across three Segmented Discourse Representation Theory (SDRT) datasets and multiple parsers, we find that last-utterance clarification is far less reliable than suggested by supervised settings. Parser-agnostic rewriting often introduces more regressions than repairs, as edits that enable fixes also disrupt discourse cues relied upon by the parser. A best-of-8 rewriting analysis further reveals a practical ceiling: a large fraction of errors are not repairable through input rewriting alone. A parser-aware clarifier trained with GRPO reduces regressions by up to 37% by learning conservative abstention, yet still fails to produce selectivity-aware clarifications that consistently improve parsing. Together, these findings recast clarification as a selective intervention problem. We identify rewritability prediction, deciding whether an utterance is repairable before intervention, as the key missing capability for input-side optimization of frozen discourse parsers, and a critical direction for improving agentic pipelines more broadly. 

---
# NAVER LABS Europe Submission to the Instruction-following 2026 Short Track 

**Authors**: Marcely Zanon Boito, Hemant Yadav, Jean-Luc Meunier, Ioan Calapodescu  

**Link**: [PDF](https://arxiv.org/pdf/2607.01960)  

**Abstract**: In this paper, we describe NAVER LABS Europe's submission to the instruction-following speech processing short track at IWSLT 2026. We participate again in the constrained setting, developing systems capable of jointly performing ASR, ST, and SQA from English speech into Chinese, Italian, and German. Building on our previous submission, ranked first in last year's short track, we update our multi-stage training pipeline by replacing the speech projector with SpeechMapper, a method for learning a speech-to-LLM embedding projector using only ASR data. In addition, we introduce a synthetic SQA dataset, fakACL, composed of artificially generated scientific presentations. This dataset is built by prompting the LLM backbone, segmenting the generated talks, and synthesizing speech with SeamlessM4T-large-v2. The combination of an improved speech projection mechanism and domain-specific synthetic data allows our model to outperform last year's best short-track system, while being considerably more compact and relying on a weaker LLM backbone. This year's results place our system tied for first place in the overall short track ranking. 

---
# AIriskEval-edu: New Dataset for Risk Assessment in AI-mediated K-12 Educational Explanations 

**Authors**: Javier Irigoyen, Roberto Daza, Francisco Jurado, Julian Fierrez, Ruben Tolosana, Alvaro Ortigosa, Enrique Blas, Aythami Morales  

**Link**: [PDF](https://arxiv.org/pdf/2607.01934)  

**Abstract**: This work introduces AIriskEval-edu-db2, a new dataset designed to train and evaluate auditors based on LLMs for an explainable pedagogical risk assessment in instructional content for grades K-12. The dataset comprises 1,639 explanations from 170 curated ScienceQA questions, covering science, language arts, and social sciences. For each question, the dataset includes an explanation written by a human teacher alongside 11 explanations generated by LLM-simulated teacher profiles associated with distinct pedagogical risks. We propose a comprehensive risk rubric aligned with established educational standards that covers five complementary dimensions: factual precision, depth and completeness, focus and relevance, student-level appropriateness, and ideological bias. A key contribution is the addition of 785 explanations with structured explainability annotations, including risk localization and risk description. The annotations are produced through a semi-automatic process with expert teacher validation. Finally, we present validation experiments comparing state-of-the-art proprietary models with a lightweight local Llama 3.1 8B model in both the pedagogical risk detection and the explainability assessment. These experiments evaluate whether supervised fine-tuning on AIriskEval-edu-db2 enables a locally deployable model to approach or outperform stronger frontier models while preserving privacy in educational auditing and assessment tasks. 

---
# TUDUM: A Turkish-Thinking Reasoning Pipeline for Qwen3.5-27B 

**Authors**: Baran Bingol, Bahaeddin Turkoglu  

**Link**: [PDF](https://arxiv.org/pdf/2607.01927)  

**Abstract**: This paper presents TUDUM (Türkçe Düşünen Üretken Model), a project pipeline for adapting a Qwen-family 27B thinking model toward Turkish reasoning. The central problem is not only to answer Turkish prompts in Turkish, but to make the explicit reasoning trace itself Turkish. A thinking model may translate a Turkish prompt into an English-centered internal or visible scratchpad, solve the problem mostly in English, and only localize the final answer. TUDUM instead treats the generated <think>...</think> block as a trainable behavior. The pipeline starts from the project base checkpoint unsloth/Qwen3.5-27B, applies supervised fine-tuning (SFT) on 15,991 Turkish reasoning examples using LoRA adapters, and then applies GRPO-family reinforcement learning on a proxy-filtered Turkish mathematics environment. The results are mixed. SFT made the model shorter and more consistently Turkish in its reasoning behavior, with large reductions in average response length and thinking exhaustion, but reduced benchmark accuracy. RL recovered some mathematical performance, especially AIME24 at the best early checkpoint, yet did not uniformly improve all benchmarks and did not exceed the base model on the reported Macro-6 average. The contribution is therefore best framed as a technically honest Turkish-thinking reasoning pipeline and evaluation, not as a claim of state-of-the-art Turkish reasoning. The released step-50 model is publicly available. 

---
# The Grammar Does the Work: Functional vs. Lexical Dependency Length Minimization Across Universal Dependencies 

**Authors**: Kim Gerdes  

**Link**: [PDF](https://arxiv.org/pdf/2607.01899)  

**Abstract**: Dependency length minimization (DLM) is a well-documented processing universal, but previous studies report a single mean dependency distance (MDD) per language, obscuring variation across syntactic relation types. We analyze 122 languages in UD and SUD (version 2.17), showing that DLM operates on two distinct levels. Grammar-driven optimization targets functional dependencies (det, case, aux), which are universally short (mean 1.71, $\sigma$ = 0.33) and invariant across typologically diverse languages. Processing-driven optimization operates on lexical dependencies (nsubj, obj, obl), which are longer (mean 2.87), highly variable ($\sigma$ = 0.63), and constrained by word-order typology. This asymmetry holds in SUD despite reversed head direction (r = 0.92). We conclude that ''the grammar does the work'' of minimization by scaffolding sentences with local functional attachments, leaving processing pressures to determine the ordering of lexical heads. 

---
# PairCoder++: Pair Programming as a Universal Paradigm for Verified Code-Driven Multimodal and Structured-Artifact Generation 

**Authors**: Junhao Chen, Xiang Li, Mingjin Chen, Boran Zhang, Henghaofan Zhang, Yibin Xu, Yuehan Cui, Fangsheng Weng, Fei Ma, Qi Tian, Ruqi Huang, Hao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.01883)  

**Abstract**: Code is the medium through which large language models generate structured artifacts: charts, scientific figures, vector graphics, CAD models, 3D scenes, and hardware designs are all produced by writing programs. In this regime single pass inference is brittle, because the compiler, renderer, or simulator that decides whether the artifact exists is invisible to the model. We present PairCoder, which grounds review in the toolchain and realizes it as two agent pair programming: a Driver agent writes the program, a Navigator agent reviews it against verification evidence (diagnostics, execution results, and renderings of the current artifact beside the target), and the two switch roles when errors persist. Across 17 public benchmarks and seven models from three vendors, PairCoder improves essentially every benchmark whose artifact is verifiable, on full official metric suites rather than execution alone (for example, Blender scene executability 0.20 to 0.78; TikZ compile rate up 10 to 30 points on every model), at 2.9 to 9.2 times single model cost (about 7 times overall). The improvements concentrate where the toolchain provides an informative oracle and the baseline leaves headroom, and the method ties or mildly regresses where the oracle is weak; we frame pair programming as a reliable recipe for verified code driven generation. 

---
# On the Limits of Steering Vectors for Preference-Aligned Generation 

**Authors**: Melanie Subbiah, Zara Hall, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2607.01802)  

**Abstract**: Steering vectors have emerged as a promising approach to controlled text generation, offering interpretable, training-free mechanisms for shaping model outputs. However, their practical generality remains poorly understood. We study the limits of steering vector generalization along three dimensions: trait expressibility, task transfer, and multi-trait composition. Using the PLUME writing personalization benchmark, we extract steering vectors for a range of preferences and evaluate them on summarization and email-writing tasks across two open-source models (Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct). We find that steering effectiveness varies substantially across traits. We further show that steering effectiveness can degrade when vectors extracted from positive and negative style examples are transferred to downstream writing personalization tasks. Finally, we compare common methods for composing multiple steering vectors and find that all methods suffer significant drops in trait expression as more vectors are added, with a tradeoff between coherence and expressibility that requires per-setting hyperparameter tuning. Taken together, our results suggest that steering vectors face meaningful limits as a general-purpose tool for preference alignment. 

---
# PARTREP: Learning What to Repeat for Decoder-only LLMs 

**Authors**: Andikawati P Widjaja, Yongjun Kim, Hyounghun Kim, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.01792)  

**Abstract**: While decoder-only LLMs excel at a vast array of natural language tasks, it suffers from an asymmetric information flow induced by causal attention: later tokens are richer in contextual grounding than earlier ones. A simple and effective remedy is prompt repetition -- just appending a second copy of prompt before generation can redistribute grounding across positions and improve reasoning performance. However, full repetition of the original prompt doubles the KV cache footprint and quadruples attention cost during prefill, making it impractical for long-context settings. We propose PartRep, a selective augmentation method that appends only the most informative tokens -- rather than the entire prompt. We use token-wise negative log-likelihood (NLL) as a selection signal, motivated by the hypothesis that less predictable tokens are less recoverable from surrounding context and therefore benefit more from late-position repetition. To avoid the heavy cost of a full forward pass for scoring, we train a lightweight gate that predicts high-NLL tokens from early-layer hidden states, enabling token selection during mid-prefill via early exit. Across eight benchmarks (including MMLU, GSM8K, and RULER) and three model families (Qwen2.5, Llama3.2, Gemma4), PartRep retains most of the gains of full repetition while using only 59.4\% of its KV cache and 79.0\% of its prefill FLOPs. 

---
# Rethinking Speech-LLM Integration for ASR: Effective Joint Speech-Text Training by Interleaving 

**Authors**: Ruchao Fan, Yiming Wang, Rui Zhao, Liliang Ren, Keqi Deng, Xiaoyang Chen, Ali Zare, Bo Ren, Yuxuan Hu, Junkun Chen, Yan Huang, Yelong Shen, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.01733)  

**Abstract**: Speech-LLM integration has shown promising results by leveraging extensive textual pretraining, yet its specific benefits for automatic speech recognition (ASR) remain unclear. We observe that as supervised ASR training data increases, the contribution of LLM priors becomes less evident, and simple speech-text joint training under-utilizes textual knowledge. We therefore propose Joint Speech-Text Interleaved Pretraining (JSTIP), an ASR-oriented pretraining strategy that constructs word-level and segment-level interleaved speech-text sequences within aligned pairs for speech-LLM architectures that accept continuous inputs. Experiments on 38k hours of ASR data show consistent entity accuracy improvement compared to ASR-only and joint speech-text training baselines. JSTIP achieves on-par entity recognition performance using domain transcription text compared to synthetic speech-text pairs, simplifying domain adaptation. Benefiting from textual pretraining and domain text data, JSTIP is competitive with open-source ASR and Speech-LLM systems in medical entity recognition. The zero-shot speech question answering behaviors further suggest that interleaving reduces the speech-text modality gap and preserves the LLM generative prior, which is likely the reason for the entity improvements on the ASR task. 

---
# When Does Generating More Help? Disentangling Fixed-Source Synthesis from Source Expansion in Synthetic Data Scaling 

**Authors**: Xu Guo, Jian Tong, Zhihui Lu, Qipeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.01727)  

**Abstract**: Synthetic data can be scaled along two routes: Source Expansion (SE), which enlarges the source by adding seed materials or generators, and Fixed-Source Synthesis (FSS), which holds the source fixed and scales the generation budget. Existing scaling studies typically expand the source as the data grows, conflating SE with FSS and leaving FSS underexplored. We isolate FSS by holding the seed-question pool and teacher model fixed, varying only the per-question response budget under Rejection Sampling (RS). We adapt the rectified scaling law to FSS, deriving it from how repeated sampling covers a fixed source. Empirically, the derived form, fit on low budgets, predicts performance at the held-out highest budget for every evaluated teacher--student pair. At matched total-sample budgets, SE and FSS are comparable at small budgets; at large budgets, adding seed questions outperforms spending the same budget on more responses. Within FSS, however, neither synthesizing additional questions from the existing seeds nor varying the synthesis protocol outperforms plain RS at matched budgets. FSS is thus a bounded scaling axis and a controlled setting for comparing synthesis protocols. We will release our code and data to facilitate further research. 

---
# ProWAFT: A ROMA-LPD Instance for Workload-Aware and Dynamic Fault Tolerance in FPGA-Based CNN Accelerators 

**Authors**: Xinxin Chen, Haoran Qiao, Yiming Guo, Kecheng Luo, Siyuan Feng, Jingwen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2607.01602)  

**Abstract**: SRAM-based FPGAs provide an attractive platform for energy- and latency-constrained CNN inference at the network edge, yet transient faults can lead to silent errors that compromise reliability. Always-on redundancy (e.g., full TMR) improves correctness but incurs substantial performance and energy overhead, while reactive recovery may introduce unacceptable latency on the critical path. We propose \textbf{ProWAFT}, a proactive workload-aware fault-tolerance framework for FPGA-based CNN accelerators that uses partial reconfiguration to selectively apply TMR across reconfigurable partitions. ProWAFT quantifies workload criticality, models fault propagation and reconfiguration overhead, and selects configurations that minimize a composite objective over latency, energy, and reliability risk. Implemented on a Xilinx Zynq UltraScale+ ZCU104 platform with six reconfigurable regions and evaluated on a 500-task trace derived from ResNet-18, MobileNetV2, and EfficientNet-Lite under time-varying SEU injection, ProWAFT achieves lower composite cost than static TMR and reactive reconfiguration while maintaining high task success rate and near-baseline throughput with low online decision overhead. 

---
# Beyond Skepticism: Evaluating LLMs Pedagogical Intent Reasoning with the Adaptive Pedagogical Vigilance Framework 

**Authors**: Minghao Chen, Ruihan Zhou, Jiayi Tang, Zihan Xu, Bowen Huang, Yuxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.01581)  

**Abstract**: The capacity of Large Language Models (LLMs) to reason about pedagogical intent within instructional communication remains underexplored, particularly in educational domains such as translation pedagogy. To address this, we propose the \textbf{Adaptive Pedagogical Vigilance (APV)} framework, a novel computational formalism that reframes communicative vigilance as an adaptive mechanism for optimizing learning through intent inference. APV formalizes the problem via a Bayesian Pedagogical Intent Inference Engine (PIIE), which models how instructors select content to maximize pedagogical utility and how vigilant learners should inversely reason about latent instructional configurations -- encompassing genre, stance, and incentives. We evaluate APV through a three-tier hierarchy: distinguishing instructional genre, reasoning about structured pedagogical setups, and generalizing to authentic educational discourse. Experiments on leading LLMs (e.g., GPT-4o, Claude 3.5) show that APV substantially improves model vigilance. It achieves the strongest discrimination between pedagogical and exposure-based content, correlates highly with human judgments ($r=0.958$), and maintains robust performance on naturalistic data where baseline methods degrade. This work establishes a unified framework for assessing and enhancing LLMs' understanding of pedagogical motives, advancing the development of more reliable AI-assisted learning systems. 

---
# DiPS: Dialogue Policy Selection for High-Stakes Persuasion Agents 

**Authors**: Tianyi Zhang, Mousumi Das, Abrar Anwar, Jesse Thomason, David Traum  

**Link**: [PDF](https://arxiv.org/pdf/2607.01557)  

**Abstract**: Large Language Models (LLMs) often struggle with persuasion in high-stakes scenarios. People's individual personalities and concerns require tailored strategies rather than a one-size-fits-all approach. To address this challenge, we focus on a fire-rescue scenario in which an operator must persuade a resident to evacuate as a high-stakes persuasion domain and propose Dialogue Policy Selection (DiPS), a Q-learning framework to dynamically select persuasion strategies adapted to the evolving conversational context. Specifically, we train a critic, trained to maximize the chance of evacuation success, to select a persuasion policy at each turn based on the resident's recent this http URL then evaluate DiPS against multiple baselines in both simulated and real human interactions. We find that DiPS achieves higher evacuation success than a zero-shot LLM and generic RAG-augmented approach. 

---
# Can Language Models Actually Retrieve In-Context? Drowning in Documents at Million Token Scale 

**Authors**: Siddharth Gollapudi, Nilesh Gupta, Prasann Singhal, Sewon Min  

**Link**: [PDF](https://arxiv.org/pdf/2607.01538)  

**Abstract**: Language models (LMs) raise an intriguing alternative to vector-based retrieval: conditioning on an in-context corpus and directly generating a relevant answer. However, prior work has largely focused on proprietary systems or the smaller-scale reranking task, leaving corpus-scale in-context retrieval largely unexplored. In this work, we present the first systematic study of in-context retrieval on two scales practical retrievers demand: million-token corpora and length-generalization far beyond training-time sizes. We first introduce BlockSearch, a 0.6B LM retriever whose architectural and training modifications improve over prior LM baselines and length-generalize up to 10 times beyond its training regime. Nevertheless, retrieval still collapses under more extreme extrapolation. We trace this failure to an attention dilution effect: as the corpus grows, irrelevant documents dominate the softmax denominator, reducing the normalized mass on the gold document even when its pre-softmax score stays high. Motivated by this analysis, we introduce length-aware adjustments to the attention softmax and document-level sparse attention. With these modifications, at the million-token scale, our model matches dense retrieval on widely studied benchmarks (e.g, MS MARCO and NQ), while outperforming the concurrent model MSA despite being 7 times smaller. Furthermore, it significantly outperforms dense retrieval on tasks requiring entirely different notions of similarity, such as LIMIT, achieving a 3 times higher score. Together, our results position in-context retrieval a promising alternative to classical retrieval while emphasizing attention control under extreme context growth as a new challenge. 

---
# Parameter Golf: What Really Works? 

**Authors**: Prashanna Mani Paudel, Shivanand Venkanna Sheshappanavar  

**Link**: [PDF](https://arxiv.org/pdf/2607.01517)  

**Abstract**: How far can a language model improve under a strict artifact budget? Parameter Golf posed this question as an open community challenge in which participants trained the best language model, with the complete artifact (training code + compressed weights) required to fit within 16 MB and be trained in under ten minutes on 8xH100 SXM GPUs. Quality was measured in bits-per-byte (BPB), the average number of bits required to encode each byte of unseen text. We analyze 2,037 pull requests and 1,430 clean scored submissions from the contest, build a taxonomy of 84 optimization techniques, and measure each technique's contribution to BPB. The verified leaderboard score dropped from 1.2244 to 1.058 BPB across three phases -- a 13.6% reduction, despite individual techniques rarely improving BPB by more than 1%. We show that most gains in techniques shrink across competitive submissions, isolating the few methods that improve performance across stacks. 

---
# From Monolingual to Multilingual: Evaluating Mamba for ASR in South African Languages 

**Authors**: Jesujoba O. Alabi, Julian Herreilers, Badr M. Abdullah, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2607.01502)  

**Abstract**: Recent advances in automatic speech recognition (ASR) have explored different sequence models, including Conformer-based models and newer state space models such as Mamba. Although prior work has evaluated these architectures in multiple languages, their effectiveness in African languages remains underexplored. In this work, we evaluate Mamba for ASR on seven South African languages. In monolingual experiments, each model is trained on 50 hours of speech per language, and we compare Mamba to a Conformer baseline of similar parameter scale. Mamba achieves similar recognition accuracy to Conformer while using fewer computational resources and training faster. We further evaluate generalization in this setting and find that both models struggle to generalize to speech that is much longer than what they were trained on. We then study multilingual ASR using Mamba models, where the baseline is pooling all languages together. On top of this, we tested three extensions: training with language-family information by adding both language and language-family embeddings as biases to the downsampled acoustic representations, and multitask learning with a CTC ASR objective and a language identification (LID) head. We find that multilingual training consistently improves performance over monolingual training. However, adding explicit language information does not improve in-domain performance but does improve cross-corpus robustness. We conducted ablation studies in low-resource multilingual settings using 5-hour and 10-hour per-language training data, where we observed gains from using language embeddings and further demonstrated that removing or altering them hurt model performance. Lastly, we analysed these embeddings and find that they do not capture linguistic similarity in a typological sense, but instead act as task-specific control vectors. 

---
# Comparing Architectures for Supervised Political Scaling 

**Authors**: Anna Golub, Sebastian Padó  

**Link**: [PDF](https://arxiv.org/pdf/2607.01464)  

**Abstract**: Text scaling, the task of positioning political actors on an ideological scale, is a fundamental task in political analysis. To ease the need for manual analysis, various NLP methods have been proposed for this task, including classification- and regression-based approaches, showing successes as well as limitations. The goal of our paper is to consolidate the state of the art in this area. We ask two questions: (a) Can the performance of scaling methods be improved by predicting scales not individually but jointly? (b) Is there a middle ground between classification and regression? 

---
# Grounded Optimization: A Layered Engineering Framework for Reducing LLM Hallucination in Automated Personal Document Rewriting 

**Authors**: Shashank Indukuri, Adarsh Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2607.01457)  

**Abstract**: Large language models (LLMs) are increasingly applied to resume optimization for applicant tracking systems, introducing hallucination failures distinct from general text generation: anachronistic technology injection, cross-domain terminology contamination, structural mutation, and content fabrication. We present Grounded Optimization, a five-layer framework combining temporal context validation, deterministic contamination detection, structural invariant enforcement, prompt-level grounding, and an evaluator agent.
In ablation experiments across three LLMs, four temperature settings, and six layer configurations on 25 synthetic resumes spanning 14 industries, undefended baselines produce 2.48-5.36 detected hallucinations per resume. Among detectors independent of the active defenses, temporal hallucinations are reduced by 50-95% across all conditions; overall detected hallucination rate falls to 0.04-0.24. Prompt-level grounding alone achieves zero detected hallucinations at low temperature with a capable instruction-following model; higher temperatures and weaker models reveal the need for the deterministic layers as a complement. We release the contamination taxonomy, evaluation code, and raw data. 

---
# FaithMed: Training LLMs For Faithful Evidence-Based Medical Reasoning 

**Authors**: Zhiyun Zhang, Liwen Sun, Xiang Qian, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2607.01440)  

**Abstract**: Faithful reasoning is essential in medicine, where clinical decisions require transparent justification grounded in reliable evidence. Current medical LLMs either lack active access to evidence or use retrieved evidence without supervising how it should be appraised and applied during reasoning. To address this, we formalize evidence-based medicine principles as process-level criteria and introduce FaithMed, a framework that combines clinician-designed, automatically refined rubrics with reinforcement learning using step-level process reward assignment and advantage grouping. Across seven medical benchmarks, FaithMed improves over agentic-search baselines (+9% on average) and outcome-only RL (+5.8%), while raising average evidence-based medicine rubric scores over agentic-search Qwen3 baselines (+15.5%). This work demonstrates that explicit step-level supervision can improve both task success and the faithfulness of the reasoning process. Code is available at this https URL. 

---
# IsoSci: A Benchmark of Isomorphic Cross-Domain Science Problems for Evaluating Reasoning versus Knowledge Retrieval in LLMs 

**Authors**: Samir Abdaljalil, Erchin Serpedin, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2607.01431)  

**Abstract**: We introduce ISOSCI, a benchmark of isomorphic cross-domain science problem pairs that separates reasoning ability from domain knowledge retrieval in LLM evaluation. Each pair shares identical logical structure but requires different domain-specific knowledge, enabling controlled attribution of reasoning-mode gains. Across five model pairs spanning four model families, we find that 91.3% of reasoning-mode gains are knowledge-dependent rather than structure-invariant (63/69 gains; Wilson 95% CI [82.3%, 96.0%]), directly challenging the assumption that chain-of-thought reasoning improves short-horizon procedural scientific problem-solving. Reasoning toggles on highly capable models provide less than 5 percentage points accuracy gain across all domains, and a reasoning-specialized model (o3-mini) that outperforms its standard counterpart on GPQA Diamond (+19.2 percentage points) underperforms on ISOSCI (-24.7 percentage points), showing that benchmark choice determines conclusions about reasoning utility. We release ISOSCI at this https URL 

---
# MultAttnAttrib: Training-Free Multimodal Attribution in Long Document Question Answering 

**Authors**: Dang Quang Thien Tran, Quang V. Dang, Vinamra Tyagi, Sai Soorya Rao Veeravalli, Trang Nguyen, Ryan A. Rossi, Franck Dernoncourt, Nedim Lipka, Koustava Goswami, Samyadeep Basu  

**Link**: [PDF](https://arxiv.org/pdf/2607.01420)  

**Abstract**: As grounded QA systems are increasingly deployed in AI assistants, accurately attributing generated answers to evidence is critical for user trust and model safety. While unimodal attributions have been explored in depth, the multimodal setting remains relatively under-researched. As a result, we introduce MultAttnAttrib, a training-free attribution-generation method that leverages a model's prefill pass, selected attention heads, and calibrated thresholds to locate source evidence within a document. To establish baseline results for the method, we introduce MultAttrEval, a complementary benchmark dataset annotated with fine-grained, ground-truth attributions for answer components grounded in multimodal source documents. To our knowledge, this is the first evaluation dataset designed specifically for multimodal attribution in long-form documents. Experimental results show that MultAttnAttrib consistently outperforms a variety of attribution-generation methods, including several strong prompting-based approaches and matches the latest frontier models such as GPT 5.4. Our method not only substantially improves attribution accuracy for both unimodal and multimodal attribution types, but also produces attributions at up to one-seventh of the direct inference latency compared to prompting on the same base model. 

---
# Multi-Objective Exploration and Preference Optimization via Mutual Information 

**Authors**: Hongyan Xie, Yikun Ban, Ruiyu Fang, Zixuang Huang, Deqing Wang, Jianxin Li, Shuangyong Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.01392)  

**Abstract**: Aligning large language models with diverse and heterogeneous human values requires multi-objective alignment methods to effectively trade off conflicting preference dimensions. Current methods achieve this trade-off by training policies conditioned on preference vectors and leveraging online direct preference optimization. However, exploration uncertainty can cause the reward distributions of responses generated under different preference vectors to overlap, and the generated responses may fail to effectively align with the corresponding preference vectors. In this paper, we propose Multi-Objective Exploration and Preference Optimization via Mutual Information (MI-EPO), an information-theoretic framework. It unifies multi-objective exploration and alignment by maximizing the joint conditional mutual information among generated responses, preference feedback, and preference vectors. By incorporating a probabilistic routing mechanism, MI-EPO naturally decomposes objective alignment and preference-aware exploration, encouraging the model to generate responses that are distinguishable and aligned with different preference conditions. Experiments on safe alignment and helpful assistant tasks show that MI-EPO significantly improves the alignment between generated responses and preference vectors, makes the outputs more controllable, and achieves stable trade-offs across multiple objectives. 

---
# RusFinChain: A Russian Benchmark for Verifiable Chain-of-Thought Reasoning in Finance with Fuzzy-Aligned Evaluation 

**Authors**: M. K. Arabov  

**Link**: [PDF](https://arxiv.org/pdf/2607.01388)  

**Abstract**: Multi-step symbolic reasoning is essential for robust financial analysis, yet most benchmarks neglect intermediate reasoning steps. FINCHAIN introduced verifiable Chain-of-Thought (CoT) evaluation but is limited to English. FINESSE-Bench includes a Russian block but relies on multiple-choice questions without step-level supervision. We present RusFinChain, the first Russian-language symbolic benchmark for verifiable CoT reasoning in finance. It spans 17 domains, 172 topics, and comprises 5,280 parameterized examples from executable Python templates, ensuring contamination-free evaluation. Each example includes a gold-standard reasoning chain with intermediate numeric values for automatic verification. We also introduce enhanced metrics: Fuzzy Numeric Alignment and Soft-Attention Alignment. We evaluate 8 open-weight LLMs on a stratified sample, generating 8,100 responses. Results reveal a substantial reasoning gap: models achieve Hard F1 of ~0.65 for step alignment, but only ~29% of final answers are correct. Our fuzzy and soft metrics show stronger correlation with final-answer correctness (Spearman rho approx 0.48) than the original ChainEval (rho approx 0.38-0.46), demonstrating superior diagnostic power. We release dataset, code, and evaluation framework to foster verifiable financial AI for the Russian-speaking community. 

---
# TurnNat: Automatic Evaluation of Turn-Taking Naturalness in Dyadic Spoken Dialogue 

**Authors**: Hao Zhang, Thomas Thebaud, Georgi Tinchev, Venkatesh Ravichandran, Laureano Moro-Velazquez  

**Link**: [PDF](https://arxiv.org/pdf/2607.01345)  

**Abstract**: Turn-taking naturalness is central to full-duplex spoken dialogue systems, yet its automatic evaluation remains limited. Existing evaluations often rely on human judgments or behavior-specific timing metrics, making it difficult to compare heterogeneous timing failures within a unified framework. We propose TurnNat, a likelihood-based framework for automatic turn-taking naturalness evaluation in two-channel spoken dialogue. A causal turn-taking prediction model trained on natural conversations estimates future two-speaker voice-activity states, and the negative log-likelihood (NLL) of the observed future activity measures timing atypicality. TurnNat pools frame-level NLLs over turn-taking boundary units (TBUs) extracted from utterance onsets and offsets, and aggregates mean and tail TBU scores into a dialogue-level naturalness score. We further construct a controlled perturbation benchmark of paired natural and perturbed dialogue clips, validated by human naturalness judgments. Experiments on this benchmark show that TurnNat successfully identifies unnatural turn-taking perturbations across heterogeneous timing failures. 

---
# RuleChef: Grounding LLM Task Knowledge in Human-Editable Rules 

**Authors**: Ádám Kovács, Nadia Verdha, Gábor Recski  

**Link**: [PDF](https://arxiv.org/pdf/2607.01293)  

**Abstract**: We present RuleChef, a framework that uses large language models (LLMs) to generate executable rules for NLP tasks such as text classification, Named Entity Recognition (NER), or relation extraction. Rules are generated based on a task description and a set of labeled examples, then they are iteratively improved based both on additional examples and on human feedback overexisting rules. RuleChef can also be used to bootstrap rules using the observed input-output pairs from any existing model for a given task. LLMs are used only at learning time, synthesizing rules and iteratively patching them based on failures measured on a held-out split. The result of this process is a fast, deterministic, and inspectable rule system. Preliminary evaluation is performed on both classification and NER tasks. We release RuleChef as open-source software under an Apache 2.0 

---
# Office Comprehension Benchmark 

**Authors**: Firoz Shaik, Mateus Picanço Lima Gomes, Tanvir Aumi, Jingci Wang, Milos Milunovic, Filip Basara, Ivana Jovanovic, Vishwas Suryanarayanan, Neha Nandan Kenkare, Weiyao Xie, Zhipeng Han, Zheng Zhang, Waleed Shahid, Jay Rathi, Russell Scherer, Thong Q. Nguyen, Michael Bentley, Tamara Stankovic, Rasika Chakravarthy, Vishal Chowdhary  

**Link**: [PDF](https://arxiv.org/pdf/2607.01245)  

**Abstract**: We introduce Office Comprehension Bench (OCB), the first public benchmark to jointly evaluate LLM systems on Word, Excel, and PowerPoint comprehension over native file formats (.docx, .xlsx, .pptx) and their variants. OCB consists of two tracks. File Fidelity Q&A tests structural and visual perception of office artifacts - tables, charts, embedded images, formulas, and app-specific elements such as headers, speaker notes, and named ranges. Domain Q&A tests expert-level reasoning grounded in real-world industry documents across 12 professional domains, with queries requiring multi-step analysis and synthesis across documents. Each reference answer is decomposed into atomic, binary-gradable claims, and an ensemble of LLM judges scores responses against each claim independently. Even the strongest frontier system in its default reasoning mode reaches only about 59.3% on Domain Q&A; increasing thinking depth within a tier does not move performance materially, while moving to a higher product tier yields modest gains. We release the dataset, evaluation tooling, judge prompt, and a public leaderboard. 

---
# Mapping Text to Multiplex Graph: Prompt Compression as Lévy Walk-Guided Graph Pruning 

**Authors**: Yaxin Gao, Yao Lu, Jinhong Deng, Jiaqi Nie, Zhe Tang, Jian Zhang, Zhaowei Zhu, Shanqing Yu, Qi Xuan, Joey Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.01241)  

**Abstract**: Existing prompt compression methods treat text as flat token sequences, failing to capture the distributed nature of important information, which is often spread across multiple locations and connected through both local syntactic dependencies and global semantic relations. Such relational structure is naturally represented as a graph, where tokens or sentences become nodes and their dependencies become edges. To this end, we propose RAGP, which formulates prompt compression as Redundancy-Aware Graph Pruning on a multiplex graph that jointly models fine-grained attention-based dependencies and coarse-grained semantic relations. To efficiently identify non-redundant nodes in this heterogeneous structure (dense local subgraphs and sparse global connections), we employ Levy walks whose heavy-tailed step distribution naturally balances local exploitation with global exploration. Experiments on LongBench show that RAGP achieves an average score of 49.3 under a 4x compression ratio, outperforming existing LLM-based compression methods, such as LongLLMLingua, which attains 48.8 at a 3x compression ratio. Besides, RAGP also surpasses state-of-the-art vision-based text compression paradigms on multiple tasks. The code is available at this https URL. 

---
# Prompt Framing Distorts Count-Based Evaluation of LLM Error Detection: Evidence from Numeric Anchoring 

**Authors**: Dekun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.01240)  

**Abstract**: Count-based F1 is widely used as a proxy for LLM error-detection quality, but this paper shows that it can rise dramatically without a corresponding improvement in span localization, a gap termed F1 Inflation. The paper introduces ErrorBench, a controlled stress-test protocol for prompt-induced count distortion. ErrorBench evaluates six contemporary LLMs under five prompt conditions over 4,290 responses from 143 CoNLL-2014 passages. Under CoNLL-2014 M2-style scoring, anchored prompts produce up to 0.79 points of F1 Inflation, and up to 0.96 under strict matching. A 100-passage replication using the official ERRANT 3.0.0 pipeline and multi-reference scoring reproduces the pattern: averaged over six models, the Blind-to-Anchored prompt shift raises Count-F1 by +0.21 while raising multi-reference ERRANT F0.5 by only +0.04. The study finds larger count responses in highly instruction-compliant GPT/Claude systems and smaller responses in the Gemini family under this stress-test protocol. The findings suggest that LLM proofreading and document-review evaluations should avoid pre-populated error counts and should report span-aware metrics alongside count-based metrics. 

---
# Breaking Safety at the Token Boundary: How BPE Tokenization Creates Exploitable Gaps in LLM Alignment 

**Authors**: Tung-Ling Li, Hongliang Liu, Yuhao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.01239)  

**Abstract**: Character-level perturbations bypass safety alignment in modern LLMs despite leaving prompts human-readable. We identify and test a central structural mechanism: BPE tokenization fragments safety-critical words into sub-word pieces, and the three public alignment datasets we surveyed contain no intentionally fragmented inputs. The mechanism is a chain, tested end-to-end on five model families (Qwen-3-4B, Qwen-2.5-7B, Gemma-3-4B, Llama-3.1-8B, Mistral-7B). An optimization targeting safety-token fragmentation flips the first-token refusal trigger on 80-100% of refused HarmBench prompts, with 48% of those flips producing genuinely harmful outputs (per-model 29-65%; gap-vs-behavior ROC-AUC 0.66-0.98, pooled 0.84). Activation patching localizes the disrupted signal to the last ${\sim}30\%$ of layers; an alignment-data scan finds zero fragmented prompts among 30,000 examples (positive-control recall $\geq 99\%$ at attack-relevant intensities); and targeted-mutation experiments isolate safety words as the disruption locus. On the defense side, a 68-cell grid (55 trained checkpoints) shows that no DPO configuration achieves seed- and pool-stable ASR closure on the three families with closed pool-size confounds. SFT trained on fragmented prompts closes ASR on 3/5 families but only via global collapse that raises refusal on benign prompts as well, indicating the missing distribution is necessary but not sufficient under the LoRA-16 recipe we tested. To distinguish selective repair from global collapse, we introduce Conv-Benign, a candidate paired diagnostic. All ASR claims are 3-judge-calibrated (cell rankings stable across judges; absolute levels $\pm$18pp; see App.~B.13). 

---
# SPARCLE: SPeaker-aware Aligned Representations via Contrastive Language Embeddings 

**Authors**: Priyam Mazumdar, Yurii Halychanskyi, Steven Guo, Mark Hasegawa-Johnson, Volodymyr Kindratenko  

**Link**: [PDF](https://arxiv.org/pdf/2607.01238)  

**Abstract**: Recent advances in speech synthesis have shifted from phoneme representations to direct grapheme modeling. While phonemes address the one-to-many mapping between text and acoustics, they rely on grapheme-to-phoneme (G2P) systems that fail to capture speaker-specific acoustic variation. Prior work demonstrates that grapheme-based models outperform phoneme-based systems at scale, but not in low-resource settings.
In this paper, we propose SPARCLE, a speaker-aware grapheme representation model that enriches characters with their precise acoustic realizations. SPARCLE is trained with a contrastive objective to align graphemes with corresponding Wav2Vec2 acoustic representations while conditioned on speaker identity. The resulting model serves as a replacement to G2P systems for downstream text-to-speech (TTS) tasks. We demonstrate that SPARCLE improves generation quality, reducing word error rates by half in extreme low-resource settings compared to standard grapheme-based models. 

---
# Kara: Efficient Reasoning LLM Serving via Sliding-Window KV Cache Compression 

**Authors**: Shen Han, Yuyang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.01237)  

**Abstract**: Reasoning language models often generate long chain-of-thought (CoT), which accumulates a massive KV cache during the decoding phase and incurs high decoding latency and limited throughput. To address these issues, KV cache compression has emerged as a promising technique for reducing memory overhead by selectively removing unimportant KV pairs while preserving useful ones for subsequent decoding. Nevertheless, we identify two key limitations in existing KV cache compression methods: 1) their threshold-triggered compression policy may provide limited throughput improvement or even reduce throughput, and may fully eliminate KV pairs from certain blocks of the sequence, potentially worsening information loss. 2) they typically retain either isolated KV pairs or fixed-size chunks with rigid boundaries, failing to preserve important flexible-sized chunks at arbitrary token positions. To overcome these limitations, we propose Kara, a sliding-window KV cache compression method that performs decoding-time compression by operating only on the recently generated context. Kara leverages bidirectional attention to score and select informative KV pairs in the window. To enable flexible preservation of important semantic information, we design a Token2Chunk module to expand a subset of selected KV pairs into chunks. Furthermore, we adapt Kara to PagedAttention and develop KvLLM, an inference framework built upon vLLM, which reduces KV cache memory usage and effectively improves output throughput. Extensive experiments demonstrate consistent performance improvements of proposed Kara and KvLLM. 

---
# Safeguarding LLM Agents from Misalignment through Provenance Analysis 

**Authors**: Yining She, Yiliang Liang, Eunsuk Kang  

**Link**: [PDF](https://arxiv.org/pdf/2607.01236)  

**Abstract**: As LLM agents gain increasing access to powerful tools, ensuring that their actions are aligned with the user's intent becomes critical. When an agent's proposed tool invocation deviates from the user's intent -- a phenomenon called misalignment -- it may lead to harmful consequences that are difficult to undo. Existing runtime guardrails rely on an LLM-as-a-judge paradigm that lacks a systematic framework for reasoning about alignment, often producing judgments that are inconsistent or difficult to audit. Motivated by provenance analysis, we propose a provenance-based conceptual framework that formalizes misalignment detection as determining whether a proposed tool call is supported by traceable evidence in the agent's context. Building on this framework, we propose ProvenanceGuard, a multi-stage pipeline that analyzes the agent's action for three types of misalignment before the selected tool is executed and only allows the action to take place when it is considered aligned with the user's input query. We evaluated our proposed approach on two different benchmarks, Agent-SafetyBench and WorkBench, across 10 backbone LLMs. Compared to the LLM-as-a-judge baseline, ProvenanceGuard reduces error rate on misaligned traces from 42.9% to 1.8% on Agent-SafetyBench and from 32.1% to 17.3% on WorkBench, while reducing intervention burden on task-successful traces from 30.5% to 12.8% and introducing no statistically significant increase in unnecessary interventions on aligned traces. These results demonstrate that structured, provenance-based reasoning provides an effective and practical foundation for safeguarding LLM agents from misalignment. 

---
# TokenScope: Token-Level Explainability and Interpretability for Code-Oriented Tasks in Large Language Models 

**Authors**: Amirreza Esmaeili, Fatemeh Fard  

**Link**: [PDF](https://arxiv.org/pdf/2607.01235)  

**Abstract**: Understanding how Large Language Models (LLMs) make token-level decisions during code generation remains a major challenge for both researchers and practitioners. While recent tools provide insights into model internals or generation outcomes, they often lack decoding-time signals, fine-grained uncertainty measures, and interactive mechanisms for exploring alternative generation paths. We present TokenScope, an interactive interpretability and analysis tool for decoder-based LLMs that exposes token-level metrics, attention patterns, and structural information during generation. TokenScope supports interactive token replacement, counterfactual branching, and code-aware aggregation via abstract syntax trees. By unifying decoding-time signals with structural program analysis, TokenScope enables systematic investigation of LLM behaviour during code generation. 

---
# Program-as-Weights: A Programming Paradigm for Fuzzy Functions 

**Authors**: Wentao Zhang, Liliana Hotsko, Woojeong Kim, Pengyu Nie, Stuart Shieber, Yuntian Deng  

**Link**: [PDF](https://arxiv.org/pdf/2607.02512)  

**Abstract**: Many everyday programming tasks resist clean rule-based implementation, such as alerting on important log lines, repairing malformed JSON, or ranking search results by intent, and are increasingly outsourced to large language model APIs at the cost of locality, reproducibility, and price. We propose fuzzy-function programming: compiling such a function from a natural-language specification into a compact, locally-executable neural artifact. We instantiate this paradigm with Program-as-Weights (PAW), in which a 4B compiler trained on FuzzyBench, a 10M-example dataset we release, emits parameter-efficient adapters for a frozen, lightweight interpreter. A 0.6B Qwen3 interpreter executing PAW programs matches the performance of direct prompting of Qwen3-32B, while using roughly one fiftieth of the inference memory and running at 30 tokens/s on a MacBook M3. PAW reframes the foundation model from a per-input problem solver into a tool builder: invoked once per function definition, it produces a small reusable artifact whose subsequent calls per function application are cheap and offline. 

---
# Online Safety Monitoring for LLMs 

**Authors**: Mona Schirmer, Metod Jazbec, Alexander Timans, Christian Naesseth, Maja Waldron, Eric Nalisnick  

**Link**: [PDF](https://arxiv.org/pdf/2607.02510)  

**Abstract**: Despite alignment training, LLMs remain prone to generating unsafe outputs at deployment time. Monitoring outputs online and raising an alarm when safety can no longer be assumed is therefore critical. We study a simple real-time monitor that turns a verifier signal from an external model into an alarm decision by thresholding, with the threshold calibrated via risk control. In experiments on mathematical reasoning and red teaming datasets, we show that this simple design is competitive with more advanced monitors based on sequential hypothesis testing. 

---
# What LLM Agents Say When No One Is Watching: Social Structure and Latent Objective Emergence in Multi-Agent Debates 

**Authors**: Arman Ghaffarizadeh, Danyal Mohaddes, Aliakbar Izadkhah, Shahriar Noroozizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2607.02507)  

**Abstract**: LLM agents will increasingly act in socially structured settings where role, audience, and relational context can shape what is advantageous or costly to say. We study whether such social structure, without any explicit objective in the prompt, changes what an agent expresses publicly relative to an off-the-record (OTR) channel elicited under the same condition. We introduce a dual-channel debate framework in which agents produce public utterances that enter the shared history alongside OTR responses that are recorded but never shown to the other participant. Across 10 models, 3 scenarios, and 5 variations within each scenario, alignment-inducing settings produce systematic public-OTR divergence in the targeted agent, with its decision divergence rising from a $\sim$3% baseline to roughly 40%. The effect is consistent across four aggregate analyses: stance, semantic similarity, natural language inference, and survey responses. In some cases, the OTR response explicitly attributes public accommodation to relational pressures, such as career risk or sponsorship obligation. The findings suggest that agent evaluation should extend beyond explicit goals and detect emergent objectives. We present a dual-channel evaluation framework and complementary behavioral measures that operationalize this assessment. 

---
# Towards Robustness against Typographic Attack with Training-free Concept Localization 

**Authors**: Bohan Liu, Wenqian Ye, Guangzhi Xiong, Zhenghao He, Sanchit Sinha, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02494)  

**Abstract**: Models trained via Contrastive Language-Image Pretraining (CLIP) serve as the foundational vision encoders for most modern Large Vision Language Models (LVLMs). Despite their widespread adoption, CLIP models exhibit a critical yet underexplored failure mode: irrelevant text appearing within images confounds visual representations, biasing them toward lexical meaning rather than true visual semantics. This robustness issue, commonly described as a Typographic Attack (TA), exposes a vulnerability that poses a significant risk to safety-critical applications such as autonomous driving. To achieve interpretable and effective robustness against TA, we propose a novel, training-free mechanistic interpretability method. Our method provides sampling-based interpretations of hidden state representations and quantitatively attributes semantic versus lexical focus to individual attention heads. Through probabilistic analysis and circuit mining, we isolate specific Vision Transformer (ViT) components that disproportionately encode lexical information, thereby identifying the mechanistic source of TA. We further show that simple interventions applied directly to the identified circuits, without any additional training, can substantially improve robustness against Typographic Attacks in object classification. These interventions, such as selective adjustment of attention weights, also outperform both supervised and training-free defense methods. Our experiments demonstrate that applying the proposed intervention to the vision encoders of several state-of-the-art LVLMs yields substantial gains in Visual Question Answering accuracy under Typographic Attack interference on RIO-Bench. These results confirm both the efficacy and the generalizability of our mechanistic approach. Code is released at this https URL. 

---
# TestEvo-Bench: An Executable and Live Benchmark for Test and Code Co-Evolution 

**Authors**: Jiale Amber Wang, Kaiyuan Wang, Pengyu Nie  

**Link**: [PDF](https://arxiv.org/pdf/2607.02469)  

**Abstract**: Software tests and code evolve together: a code change should be followed by new or updated tests that record the new software behavior. Yet existing test generation and update benchmarks often isolate the test from the code change, and rely on static metadata that does not verify whether a test is executable or semantically tied to the code change. This makes it difficult to evaluate whether a test automation agent understands how a code change should propagate into the test suite.
We introduce TestEvo-Bench, a benchmark of test and code co-evolution tasks mined from software repositories, with two tracks: in test generation, the agent shall write new tests to capture the new software behavior; in test update, the agent shall adapt failing existing tests to the changed software behavior. Each task is anchored to a real commit history and packaged with environment configuration to support execution-grounded metrics such as pass rate, coverage, and mutation score. TestEvo-Bench is also a live benchmark: each task records the timestamp of the test and code changes, and new tasks are periodically mined by our automated pipeline, so evaluation can be restricted to tasks postdating a model's training cutoff to reduce data leakage risk. The current snapshot contains 746 test generation and 509 test update tasks, curated from 59,950 candidate co-evolution records across 152 open-source Java projects. We experiment with four state-of-the-art agents that combine strong harnesses (Claude Code, Gemini CLI, and SWE-Agent) with strong foundation models (Claude Opus 4.7 and Gemini 3.1 Pro). Results show that they achieve up to 77.5% success rate on test generation and 74.6% on test update. However, success rate is materially lower on the most recent benchmark tasks and drops significantly under limited per-task cost. 

---
# EvoPolicyGym: Evaluating Autonomous Policy Evolution in Interactive Environments 

**Authors**: Zhilin Wang, Han Song, Runzhe Zhan, Jusen Du, Jiacheng Chen, Tianle Li, Qingyu Yin, Yulun Wu, Zhennan Shen, Tong Zhu, Yanshu Li, Guanjie Chen, Derek F. Wong, Yafu Li, Yu Cheng, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02440)  

**Abstract**: Autonomous agents are increasingly expected to improve executable policies through feedback, yet existing evaluations often collapse this process into a final score or confound it with open-ended software-engineering progress. We introduce Autonomous Policy Evolution, a controlled evaluation setting in which a harness-model agent repeatedly edits an executable policy system under a fixed interaction budget. We instantiate this setting in EvoPolicyGym, a benchmark built from compact interactive RL environments that evaluates how agents iteratively improve explored policies. On the EvoPolicyGym suite, GPT-5.5 achieves the strongest aggregate rank score and top-two performance on all 16 environments. Beyond leaderboard results, EvoPolicyGym also provides trajectory-level diagnostics that distinguish how agents allocate budget, convert feedback into parametric tuning. These analyses show that strong autonomous policy evolution depends not only on isolated task wins, but on discovering task-appropriate mechanisms and refining policies under bounded feedback. 

---
# Automated grading of Linux/bash examinations using large language models: a four-level cognitive taxonomy approach 

**Authors**: Manuel Alonso-Carracedo, Ruben Fernandez-Boullon, Pedro Celard, Francisco J.Rodriguez-Martinez, Lorena Otero-Cerdeira  

**Link**: [PDF](https://arxiv.org/pdf/2607.02432)  

**Abstract**: Scalable and reliable grading of command-line examinations remains a challenge in computing education, where rising enrolments make manual marking difficult and rule-based autograders cannot handle partial credit, equivalent solutions, or syntactic variation. This paper evaluates whether four frontier Large Language Models (GPT, Claude Opus, Gemini, and GLM) can approximate expert judgment when grading short Linux/bash command responses. The study adopts a four-level cognitive taxonomy that combines cognitive complexity and operational impact, ranging from information retrieval (L1) and basic file manipulation (L2) to structural operations (L3) and advanced system management (L4). The models were tested with two prompt variants, a minimal baseline and a rubric-enhanced version, on 1200 real responses from second-year Computer Engineering students independently graded by three expert instructors. Gemini~3.0 Pro with rubric-guided prompting achieved the highest human-AI agreement (ICC(3,1) = 0.888, MAE = 0.10, Bland-Altman bias = -0.014). Agreement declined consistently as taxonomy level increased, with the largest discrepancies at higher levels. Across all models, rubric quality had a larger effect than provider choice, with structured prompts consistently improving agreement. These results show that question complexity is a reliable predictor of the difficulty LLMs face in grading accurately, and they establish a principled, taxonomy-based framework for determining which questions are suitable for AI-assisted grading and which require human review, while also providing a transferable evaluation protocol and prompt templates. 

---
# SkillFuzz: Fuzzing Skill Composition for Implicit Intents Discovery in Open Skill Marketplaces 

**Authors**: Jinwei Hu, Yi Dong, Youcheng Sun, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02345)  

**Abstract**: Large Language Model (LLM)-based agents increasingly automate software engineering tasks through reusable skills, natural-language instruction documents that guide planning and execution. Open skill marketplaces enable users to assemble agents by co-activating community-contributed skills, but marketplace operators typically audit skills in isolation. As a result, individually benign skills may interact to redirect an agent toward unintended objectives, which we term implicit intents. Detecting such intents is challenging because the effect emerges only through skill composition, execution environments are often unavailable at admission time, and the space of possible co-activations grows exponentially with marketplace size. In this paper, we formulate implicit-intent discovery as a fuzzing problem over skill compositions, where skill compositions are the unit under test, planning artifacts expose agent intent before execution, and deviations from a skill-free baseline serve as a differential oracle. Based on this formulation, we propose skillfuzz, the first execution-free testing approach that extracts structured skill contracts and uses contract-guided Monte Carlo Tree Search to prioritize potentially conflicting compositions. Across representative skill-marketplace workloads, skillfuzz discovers over 1,000 distinct implicit intents under a fixed query budget, confirms more than 80% of the highest-risk flagged compositions during execution-time validation, and identifies substantially more high-severity implicit intents than alternative search strategies while exploring only a fraction of the pairwise interaction space they require. 

---
# HNSW with Accuracy Guarantees Using Graph Spanners -- A Technical Report 

**Authors**: Minghao Li, Raghav Mittal, Sanjivni Rana, Suraj Shetiya, Gautam Das, Nick Koudas  

**Link**: [PDF](https://arxiv.org/pdf/2607.02338)  

**Abstract**: Hierarchical Navigable Small World (HNSW) graphs serve as the industry standard due to their logarithmic complexity and strong empirical performance. However, HNSW relies on greedy graph traversal, a heuristic that provides no theoretical guarantees of correctness. In this paper, we propose a novel "Certify-then-Rectify" framework that bridges the gap between the speed of heuristic search and the rigor of exact retrieval. Rather than discarding HNSW, our approach first employs a distribution-free statistical certifier to dynamically evaluate the quality of a standard HNSW search with minimal overhead. If certification indicates that the retrieved neighbors are of low quality, the framework safely escalates to a rigorous exact recovery algorithm. To make this exact recovery computationally feasible, we reinterpret the HNSW graph as a geometric spanner and utilize Extreme Value Theory to stochastically estimate its maximum empirical stretch factor. This allows us to mathematically bound the maximum distance of true nearest neighbors. Extensive evaluations on benchmark datasets demonstrate that our tiered framework delivers the average-case speed of HNSW while ensuring the worst-case correctness of exact search and outperforming other applicable approaches. 

---
# HERMES: A Multi-Granularity Labeling Substrate for Pre-training Data Mixtures 

**Authors**: Ziyun Qiao, Yue Min, Ruining Chen, Yujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.02266)  

**Abstract**: Most data-mixing methods assume the corpus has already been partitioned into groups, and the choice of those groups determines what a mixer can express. Existing labels, including provenance, topic or format taxonomies, and flat embedding clusters, commit to one semantic axis at one granularity; changing the resolution rebuilds the labels. We argue the bottleneck is the label system, not the mixer, and provide a hierarchical one. HERMES is a data-derived labeling substrate: a Learned Semantic Transform followed by 3-stage residual vector quantization annotates each document once into a coarse-to-fine code whose prefix length controls granularity up to approximately 130k cells. At coarse granularity HERMES sits at a plateau with KMeans-family methods on standard clustering metrics, so the contribution is the substrate, not the clusterer. On 1B-parameter, 25B-token pre-training, the hierarchy exposes an interaction fixed-granularity pipelines cannot test: at one prefix length, a combined Stage-2 rule contrast, equal-subbucket coverage versus size-proportional within-bucket quality top-30%, lifts a 16-task capability macro-average by +0.0253; at the next finer level, the same rule loses its measurable edge as candidate pools contract approximately 5x. HERMES reframes data mixture design from choosing among fixed label sets to navigating a reusable, data-derived granularity hierarchy. 

---
# AgenticSTS: A Bounded-Memory Testbed for Long-Horizon LLM Agents 

**Authors**: Xiangchen Cheng, Yunwei Jiang, Jianwen Sun, Zizhen Li, Chuanhao Li, Xiangcheng Cao, Yihao Liu, Fanrui Zhang, Li Jin, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.02255)  

**Abstract**: Memory for a long-horizon LLM agent is a contract about what each future decision is allowed to see. The simplest contract appends past observations, tool calls, and reflections to every prompt, which makes prior context easy to access but also turns it into a jumbled mixture in which the effect of any single memory component is hard to isolate. We introduce and instrument an alternative bounded contract: every decision is made from a fresh user message assembled by typed retrieval, with no raw cross-decision transcript appended. The prompt thus stays bounded across runs of any length, and any single layer can be ablated in isolation. We instantiate the contract in Slay the Spire 2, a closed-rule stochastic deck-building game whose runs require hundreds of tactical and strategic decisions. A public online benchmark of frontier LLMs on the same game reports zero wins at the lowest difficulty across five configurations, and the developer-reported human win rate at the same difficulty is 16%; the task is hard but not saturated. Within our harness, a fixed-A0 ablation shows the largest observed difference when triggered strategic skills are enabled: the no-store baseline wins 3/10 games and adding the skill layer 6/10. At this sample size the comparison is directional rather than statistically decisive (Fisher exact p\approx0.37); a cross-backbone probe and public accumulating-context baselines are reported as operational comparisons rather than controlled tests of the contract variable itself. We release a reproducible testbed: 298 completed trajectories with condition tags, frozen memory/skill snapshots, prompt records, and analysis scripts -- an agent design and a validated, reusable methodology for studying how explicit memory layers shape long-horizon LLM-agent decisions. 

---
# Bayesian Sparse Low-Rank Adaptation for Large Language Model Uncertainty Estimation 

**Authors**: Jijie Zhang, Zhe Ren, Quan Zhang, Dandan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.02182)  

**Abstract**: Large language models (LLMs) exhibit remarkable reasoning capabilities, but their task-specific fine-tuning is notoriously plagued by overconfidence, severely hindering trustworthy deployment. We propose Data-Adaptive Lower-Rank Adaptation (DALorRA), a simple and effective variational Bayesian sparse framework that shifts the paradigm of uncertainty quantification from the dense parameter space to the lightweight rank level of low-rank adaptation (LoRA). With the insight that LoRA essentially aggregates multiple rank-one components that may provide superfluous model capacity, DALorRA imposes stochastic masking on rank dimensions, enabling Bayesian regularization of model capacity during training and ensemble-like calibration during inference. Extensive experiments demonstrate DALorRA's excellent calibration of LLMs without compromising reasoning accuracy. 

---
# ESC: Emotional Self-Correction for Reliable Vision-Language Models 

**Authors**: Tien-Huy Nguyen, Minh-Nhat Nguyen, Nguyen Nhat Huy, Hung Viet Nguyen, Huy Nguyen Minh Nhat, Thanh-Huy Nguyen, Cuong Tuan Nguyen, Hoang M. Le, Dat Nguyen, Phat Kim Huynh, Min Xu, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2607.02089)  

**Abstract**: Vision-language models (VLMs) have achieved strong performance across diverse multimodal tasks, yet they remain vulnerable to unreliable reasoning. Existing self-correction methods mitigate these issues but typically rely on post-training or carefully engineered feedback, incurring high computational cost. In this work, we revisit this challenge through the lens of emotional cues, asking whether they can activate latent self-correction behaviors in VLMs without additional training. \textbf{We find that emotional signals serve as an effective trigger for self-correction, encouraging more cautious and reflective reasoning}. Motivated by this finding, we propose \escabstract (\textbf{\underline{E}}motional \textbf{\underline{S}}elf-\textbf{\underline{C}}orrection), a training-free self-correction framework. ESC introduces an external verifier that detects potentially incorrect initial responses and injects emotional feedback to encourage model to reflect, and produce a better revised response without additional training. Extensive experiments across safety, hallucination, vision-centric perception, and multimodal reasoning benchmarks show that ESC consistently improves reliability while preserving overall model utility. These results suggest that emotion can function not only as an ability to be recognized, but also as a practical control signal for scalable self-correction in VLMs. \textbf{We therefore believe that ESC provides a strong foundation for a new reliable human-like, emotion-integrated research direction.} Our project is publicly available at \textcolor{red}{this https URL}. 

---
# PACE: A Proxy for Agentic Capability Evaluation 

**Authors**: Yueqi Song, Lintang Sutawika, Jiarui Liu, Lindia Tjuatja, Jiayi Geng, Yunze Xiao, Daniel Lee, Aditya Bharat Soni, Vincent Lo, Xiang Yue, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2607.02032)  

**Abstract**: Evaluating LLM agents on benchmarks like SWE-Bench and GAIA can be expensive, time-consuming, and requires complex infrastructure. A single evaluation can cost thousands of dollars and take days to complete. In contrast, non-agentic LLM benchmarks that test individual capabilities (e.g., reasoning, code generation) are fast and cheap to run. In this paper, we investigate whether performance on expensive agentic benchmarks can be accurately predicted by the performance on a small, carefully selected subset of atomic evaluation instances. We introduce PACE, a framework that constructs proxy benchmarks by selecting instances from existing non-agentic evaluations whose aggregate scores most reliably predict model performances on agentic benchmarks. Given a pool of candidate instances spanning atomic capabilities, PACE fits a regression that maps a model's scores on a compact subset of source instances to its score on the target agentic benchmark. The subset itself is curated by combining two complementary instance-selection strategies, target-relevance local selection and globally informative global selection. We apply PACE to the 4 target agentic benchmarks in this paper, which yields PACE-Bench, the concrete proxy benchmark that we evaluate in the paper. Experiments across 14 models, 4 agentic benchmarks, and 19 non-agentic benchmarks show that PACE-Bench predicts agentic scores with leave-one-out cross-validation (LOOCV) mean absolute error (MAE) under 4%, Spearman correlation above 0.80, and pairwise model-ranking accuracy around 85%, all at much less than 1% of the full agentic evaluation cost. We further analyze the selected proxy instances, revealing which skills each agentic benchmark uniquely demands. PACE enables practitioners to obtain reliable estimates of agentic performance during model development, selection, and routing, without the overhead of full agent evaluation. 

---
# Multimodal Knowledge Edit-Scoped Generalization for Online Recursive MLLM Editing 

**Authors**: Siyuan Li, Youyuan Zhang, Ruitong Liu, Junxi Wang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.01978)  

**Abstract**: Online multimodal knowledge editing requires injecting a continual stream of visual-textual corrections into multimodal large language models (MLLMs) with bounded overhead and minimal disruption to unrelated behaviors. Existing editors mainly emphasize edit reliability and long-horizon stability, but rarely control the semantic boundary of each edit. Our pilot analyses of post-edit behaviors and internal neuronal activities reveal a scope gap behind reliable edits: instance-level success neither guarantees transfer to valid cross-modal variants nor prevents leakage to unrelated inputs, while edit-related cross-modal responses concentrate in deeper semantic layers. Therefore, we formulate Edit-Scoped Generalization, reframing online MLLM editing from merely correcting an instance to controlling the propagation boundary of each edit. To this end, we propose ScopeEdit, a scope-aware online editor that decomposes each update into a modality-local absorption branch and an evidence-gated shared generalization branch. The local branch supports stable edit absorption, whereas the shared branch enables cross-modal propagation only when visual and textual evidence are sufficiently aligned. Both branches perform scope-separated write geometries in orthogonal low-rank spaces and maintain branch-wise preconditioners via Sherman--Morrison recursions, yielding constant per-edit overhead. Extensive experiments across diverse benchmarks, long-horizon edit streams, MLLM backbones, real-world VLKEB scenarios, and complex vision-language architectures show that ScopeEdit consistently improves the trade-off between in-scope cross-modal transfer and out-of-scope locality, while preserving edit reliability, stability and online efficiency. Our code is available at this https URL. 

---
# Robust for the Wrong Reasons: The Representational Geometry of LLM Robustness to Science Skepticism 

**Authors**: Minjong Cheon  

**Link**: [PDF](https://arxiv.org/pdf/2607.01951)  

**Abstract**: Large language models (LLMs) are increasingly consulted on contested scientific questions, raising the concern that they will sycophantically retreat from established consensus when a user signals doubt -- drifting toward a false balance that treats settled science as one view among several. We test this across three open instruction-tuned models (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B), three consensus-science domains (climate, vaccines, evolution), and single- and multi-turn settings, combining behavioral measurement with linear probing and activation patching. We do not observe sycophantic retreat. Instead, models show three distinct policies under the same skeptical pressure: reactive assertion, where consensus assertion increases rather than decreases (Llama); surface hedging, where tone softens while the position holds (Qwen); and non-response (Mistral). Pairwise judgments confirm the reactive shift is stance, not style (63.6%, p=.007), and a decomposition identifies increased consensus assertion, not false balance, as its driver (beta=+0.042 per dose, p<1e-77). Linear probes localize the divergence to middle layers -- perfect separation in Llama and Qwen versus 72% in Mistral, with non-overlapping confidence intervals -- indicating the non-responsive model does not linearly represent the skepticism signal at all. Crucially, this robustness does not transfer: it attenuates across domains and, in the safety-critical vaccine domain, can reverse, with myth-rebuttal weakening under skeptical pressure. We synthesize these into a four-way taxonomy separating active from accidental robustness, and argue that behavioral evaluation alone cannot distinguish a model that resists skepticism because it understands the signal from one that only appears to resist because it fails to perceive it. 

---
# PhysMani: Physics-principled 3D World Model for Dynamic Object Manipulation 

**Authors**: Peng Yun, Shouwang Huang, Hao Li, Jinxi Li, Jianan Wang, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.01938)  

**Abstract**: Manipulating fast and dynamically moving targets in unstructured 3D environments remains challenging for embodied AI. Existing visual-language-action models and world models struggle with accurate 3D geometry and physically meaningful forecasting. We propose PhysMani, a framework that couples a physics-principled 3D Gaussian world model with a future-aware action policy model. The world model learns a divergence-free Gaussian velocity field via online optimization for fast and physically grounded future dynamics prediction. The policy model integrates the predicted 3D scene future dynamics through a learnable token based cross-attention module. We introduce PhysMani-Bench, a dynamic manipulation benchmark with 16 tasks, and demonstrate a superior success rate over strong baselines in both simulation and real-world robot experiments. 

---
# Spec-AUF: Accept-Until-Fail Training under Train-Inference Misalignment for Masked Block Drafters 

**Authors**: Tianjian Yang, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.01893)  

**Abstract**: Speculative decoding accelerates autoregressive generation by drafting a block of tokens that the target model verifies left-to-right, committing only the longest accepted prefix. Block (DLM-style) drafters predict the whole block in parallel, which is fast but trained with a full-block cross-entropy that supervises every position against the gold continuation -- even though inference discards every token after the first rejection. Recent acceptance-aware objectives patch this by reweighting the full-block loss; we instead use teacher-forced learning as a motivation for how supervision should concentrate on the accepted prefix. A mask-only block drafter has no input-side channel for gold-prefix conditioning, so AUF approximates that prefix-sensitive supervision on the loss side by keeping the cross-entropy support only through the drafter's first predicted failure. AUF is a single, detached change to the CE support -- no auxiliary objective, no verifier rollouts, and no change to the inference pipeline or the exactness contract. Within fixed drafter backbones and serving settings on Qwen3-8B, AUF raises the DFlash drafter's average emitted length $\tau$, averaged over six benchmarks, from 2.40 to 2.61, with a gain on every benchmark, and transfers to Domino's two-branch head (2.56 to 2.68). Two findings sharpen the picture: the decay-only baseline reaches higher token accuracy on the shared block mask yet decodes worse, and on DFlash, once AUF truncates the support, the standard exponential position-decay weighting becomes empirically inert. 

---
# SkillCoach: Self-Evolving Rubrics for Evaluating and Enhancing Agentic Skill-Use 

**Authors**: Jiayin Zhu, Kelong Mao, Yudong Guo, Dengbo He, Sulong Xu, Simiu Gu, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2607.01874)  

**Abstract**: Skills are becoming a reusable operational layer for LLM agents, encoding SOPs, domain rules, tool workflows, scripts, and validation routines. In realistic skill repositories, overlapping skills make reliable skill-use difficult. Final verifier success is too coarse for both evaluation and training, since an agent may pass through trial and error while selecting distractor skills, skipping required steps, composing workflows incorrectly or omitting final checks. We introduce SkillCoach, a self-evolving rubric framework for evaluating and enhancing agentic skill-use. SkillCoach derives skill-grounded process rubrics from real rollouts and evaluates trajectories along four dimensions: skill selection, skill following, skill composition, and skill-grounded reflection. It keeps the external verifier as a separate outcome signal, allowing process quality to be distinguished from accidental task success. The evolved rubrics further serve as process supervision for selecting high-quality training trajectories. Experiments show that evolved rubrics substantially improve evaluation quality, expose failures hidden by final accuracy, and provide stronger supervision signals than outcome-only filtering for enhancing agentic skill-use. 

---
# Safety Targeted Embedding Exploit via Refinement 

**Authors**: Joshua Adrian Cahyono  

**Link**: [PDF](https://arxiv.org/pdf/2607.01859)  

**Abstract**: Safety training for large language models (LLMs) is conducted predominantly in English, leaving uncertain how well safety mechanisms generalize to low-resource languages and mixed-language code-switching. We show that this creates an epistemic gap in which models confidently generate harmful responses for inputs that fall outside the distribution of their safety training. To study this phenomenon, we introduce STEER (Safety Targeted Embedding Exploit via Refinement), a gradient-guided attack that identifies words contributing most strongly to the model's refusal behavior and iteratively translates them into low-resource languages to suppress refusal while preserving harmful intent. Across six open-source 8B-parameter models, STEER achieves attack success rates of up to 93.0% on JailbreakBench and 96.7% on AdvBench, outperforming random code-switching and Greedy Coordinate Gradient (GCG). The resulting prompts also transfer to GPT-4o-mini, achieving a 35.5% attack success rate without requiring access to the target model, suggesting that the underlying weakness is not specific to a single architecture. These findings demonstrate that safety mechanisms aligned primarily on English cannot be assumed to generalize across multilingual inputs. We argue that improving multilingual safety requires broader coverage during alignment and mechanisms that explicitly detect and abstain on out-of-distribution inputs. 

---
# Evaluating Chunking Strategies for Retrieval-Augmented Generation on Academic Texts 

**Authors**: Valentin J. J. Kreileder, Johannes Reisinger, Andreas Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2607.01852)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems use the question-answering capabilities of Large Language Models (LLMs) to access information outside their parameters. We evaluate if cluster-based semantic chunking improves retrieval and answer quality compared to fixed-size and recursive chunking evaluating on long, structured academic theses using the Retrieval Augmented Generation Assessment (RAGAs) framework. RAGAs based faithfulness shows limited reliability in this setup. Performance on fixed versus document specific questions varied substantially, likely related to the formatting of documents and preprocessing. Under the tested configuration, cluster-based chunking did not outperform simpler strategies. 

---
# Non-synchronism in Global Usage of Research Methods in Library and Information Science from 1990 to 2019 

**Authors**: Chengzhi Zhang, Liang Tian  

**Link**: [PDF](https://arxiv.org/pdf/2607.01833)  

**Abstract**: The global development of Library and Information Science (LIS) is influenced by various factors such as the economy, society, culture, discipline, tradition, and more. Consequently, the research methods of LIS vary greatly among countries. To better understand these differences, we conducted a study of 5,281 research papers from 81 countries published in internationally representative journals over the past thirty years. We manually annotated the research methods used in some articles through content analysis, and subsequently developed and trained a deep learning model for automatic classification of research methods. Using this method, we conducted a comparative analysis of the usage of research methods in different countries. Our findings reveal that there are differences in the research methods used across countries, with each country having its unique research profile and distribution of research methods. Even when investigating the same topic, research methods can differ between countries. Our study also uncovers that there are differences between the national and international distribution of research methods, these differences have decreased over the past 30 years. By highlighting the characteristics of discipline development in various countries from the perspective of research methods, our study can help guide discipline development at the national level. This study provides insights into the usage trends of research methods across different countries and highlights the unique characteristics of discipline development in each country. This information can be valuable in promoting collaboration and understanding between countries and in guiding discipline development at the national level. 

---
# Pre-Flight: A Benchmark for Evaluating Large Language Models on Aviation Operational Knowledge 

**Authors**: Alex Brooker, Tim Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2607.01829)  

**Abstract**: Large language models (LLMs) are increasingly proposed for aviation business operations, from documentation and training generation to customer facing assistants. General purpose benchmarks do not measure whether a model reasons safely and correctly about aviation specific operational knowledge, and the high stakes, regulated nature of the domain makes that gap consequential. We present Pre-Flight, an open source benchmark of 300 multiple choice questions drawn from international standards and airport ground operations material, covering international airport ground operations, ICAO and US FAA regulations, aviation general knowledge and complex operational scenarios. Questions were authored and reviewed by practitioners with experience in air traffic management, ground operations and commercial flying. We evaluate a range of contemporary commercial and open weight models using the Inspect evaluation framework, scoring by accuracy under a standard multiple choice protocol, and we maintain the leaderboard on a rolling basis as new models are released. Against an informal expert reference of around 95%, obtained from a low sample quiz of aviation professionals at a conference, even the strongest model evaluated (released in 2026) reaches 82.7%, having improved only gradually from roughly 75% in early 2025. A substantial and persistent gap below expert level reliability therefore remains. We release the dataset, the evaluation harness and the results, and the benchmark is available within the community evaluations package distributed with inspect_evals. We argue that domain specific evaluation of this kind is a necessary precondition for responsible deployment of generative AI in non safety critical aviation operations. 

---
# Gender Differences in Research Topic and Method Selection in Library and Information Science: Perspectives from Three Top Journals 

**Authors**: Chengzhi Zhang, Siqi Wei, Yi Zhao, Liang Tian  

**Link**: [PDF](https://arxiv.org/pdf/2607.01828)  

**Abstract**: Research in the social sciences has shown that there are gender differences in the selection of research methods, with women often opting for qualitative methods while men prefer quantitative methods. However, it is important to consider that research methods are generally chosen based on the research topic. To figure out the influence of gender on research method selection, a study was conducted in the field of Library and Information Science, using a more fine-grained method classification system and an automatic classification model called CogFT, which is based on full-text cognition. The findings showed that women tend to use Interview while men prefer Theoretical approach, across a range of topics. The study offers insights into the specific research design processes that contribute to gender differences in method selection and suggests ways to promoting gender inclusivity and equality in academia by considering research method use and guidance. 

---
# Self-Supervised Test-Time Tuning for Packet Loss Concealment 

**Authors**: Yehoshua Dissen, Joseph Keshet  

**Link**: [PDF](https://arxiv.org/pdf/2607.01823)  

**Abstract**: Packet loss concealment (PLC) reconstructs audio packets that are missing at the receiver, usually with a trained model whose parameters remain fixed at deployment time. This treats the PLC model as static, even though each call or recording exposes signal-specific information through the packets that did arrive. We present TTT-PLC, a self-supervised test-time tuning framework that adapts existing PLC models using only those received packets. The method creates supervision by synthetically masking portions of the available signal, training the model to conceal them with its native PLC objective, and then using the adapted model to reconstruct the true packet losses. No clean reference signal, external adaptation data, or architectural modification is required.
We study TTT-PLC in two deployment settings. In the non-causal setting, the received file is available before reconstruction, allowing repeated self-supervised adaptation passes and providing a per-file adaptation ceiling. In the causal setting, audio is streamed without revising emitted samples; adaptation is performed only on completed past blocks, and updated parameters affect only future audio. We instantiate the framework on two public PLC backbones, FRN, a recurrent full-band speech PLC model, and PARCnet, a hybrid autoregressive-neural model for networked music. Across these settings, the results show that pretrained PLC systems do not need to be treated as fixed at inference time, the still-observed portions of a lossy signal can provide an effective training signal for improving concealment on that same signal. 

---
# Do LLMs Truly Generalize in the Molecular Domain? A Perturbation-Based Analysis 

**Authors**: Jiatong Li, Weida Wang, Changmeng Zheng, Shufei Zhang, Yatao Bian, Xiao-yong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.01800)  

**Abstract**: Large Language Models (LLMs) have recently shown promise in molecular discovery, yet a gap remains between their probabilistic nature over discrete sequential tokens and the rigid topological constraints of chemical space. This raises the question of whether molecular LLMs can generalize beyond the local neighborhoods induced by their sequence-based representations. To systematically investigate this question, we introduce a Molecular Perturbation framework that generates syntax-valid structural variants of training molecules under controlled Graph Edit Distance (GED) to probe the manifold regularity of molecular LLMs. Our analysis shows that even a single edit can cause substantial performance drops on common molecular tasks, revealing a narrow local trust region and fragile sensitivity to structural changes. Since similar molecules tend to exhibit similar properties, In-Context Tuning (ICT), which anchors predictions on structurally similar molecules, offers a natural way to mitigate such fragility. Our experiments also examine whether ICT confers robustness under controlled structural perturbations, and the results suggest that it can partially expand the local trust region and offer a promising direction for stabilizing molecular LLMs against structural variation. 

---
# Subliminal Clocks: Latent Time Modelling in Diffusion Language Models 

**Authors**: Maximo Rulli, Thomas Fontanari, Simone Petruzzi, Federico Alvetreti, Giorgio Strano, Donato Crisostomi, Giorgos Nikolaou, Tommaso Mencattini, Andrea Santilli, Emanuele Rodolà, Simone Scardapane, Alessio Devoto  

**Link**: [PDF](https://arxiv.org/pdf/2607.01774)  

**Abstract**: Diffusion Language Models (DLMs) have recently emerged as a promising alternative to autoregressive models. Unlike standard diffusion-based approaches, DLMs are not explicitly conditioned on a timestep, raising a natural question: do these models internally represent denoising progress, and how is such information used downstream? In this work, we show that DLMs do in fact encode a latent representation related to the diffusion timestep within their residual streams. We find that this signal can be reliably extracted using probes across layers, indicating that denoising progress is decodable from internal activations. We further demonstrate that steering the model along a low-dimensional subspace associated with the inferred timestep allows us to systematically modulate its notion of denoising progress, leading to predictable changes in model confidence and entropy. Finally, we analyse the geometry of the identified representation, showing that it exhibits structured and interpretable properties in activation space, and shedding light on how such a signal is processed by these models. 

---
# Denser $\neq$ Better: Limits of On-Policy Self-Distillation for Continual Post-Training 

**Authors**: Meng Wang, Haohan Zhao, Wenzhuo Liu, Lu Yang, Geng Liu, Haiyang Guo, Guo-Sen Xie, Gaofeng Meng, Hongbin Liu, Fei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.01763)  

**Abstract**: Continual post-training enables foundation models to acquire new knowledge while preserving existing capabilities. Recent work suggests that on-policy learning can mitigate forgetting, with on-policy self-distillation emerging as a particularly attractive approach. In this work, we revisit this optimistic view through self-distillation policy optimization (SDPO). Our experiments show that SDPO can accelerate in-domain specialization when teacher signals are stable and well aligned, but it struggles to generalize to out-of-distribution scenarios. In continual post-training, SDPO exhibits stronger forgetting and can even collapse, whereas on-policy reinforcement learning methods such as GRPO adapt more conservatively and better preserve prior capabilities. Further analyses reveal that denser self-distillation induces larger drift in both parameter space and response space, and can amplify high-frequency formatting artifacts through a self-reinforcing teacher--student loop. These findings suggest that on-policy data alone is insufficient for continual learning. Dense self-distillation can accelerate specialization when teacher targets are stable and token-level supervision is reliable, but it should not be treated as a default stabilizer for continual post-training. Our code is available at this https URL. 

---
# Beyond Pixel Diffs: Benchmarking Image Change Captioning for Web UI Visual Regression Testing 

**Authors**: Licheng Zhang, Bach Le, Pengtao Zhao, Naveed Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2607.01728)  

**Abstract**: Visual regression testing (VRT) is a standard quality assurance step in modern software release pipelines. On every change, it re-renders user interface (UI) screenshots, compares each one against an approved baseline image, and routes any detected difference to a human reviewer who decides whether it is an intended update or an unintended regression. A widely used approach, especially in open-source and continuous-integration pipelines, is pixel-level comparison, which is semantically blind and treats rendering noise and genuine defects identically, producing large volumes of false positives that force developers and testers to spend substantial time and effort manually reviewing flagged differences at every release cycle. Industry tools apply machine learning to VRT, but lack public evaluation. More critically, no dataset or benchmark exists to support natural language descriptions of UI changes, a capability that tells testers what changed in words instead of leaving them to interpret a binary flag or a highlighted region. To address the gap, we propose a new task, Web UI Image Change Captioning (WUICC), which sits at the intersection of VRT and image difference captioning (IDC), and release WUICC-bench, its first dataset and benchmark for the task. We evaluate eleven representative IDC methods, together with two zero-shot general-purpose LLMs. We find that: (1) these methods tend to struggle in the Web UI domain due to its layout diversity, dense text, and fine-grained changes, and (2) yet the trained methods already suppress non-meaningful visual noise far more selectively than the pixel-level comparison VRT relies on, providing a solid foundation for future domain-specific research. 

---
# Epistemic Goggles: A Pretrained Module that Induces an Epistemic Frame via Gradient Editing 

**Authors**: Joshua Penman  

**Link**: [PDF](https://arxiv.org/pdf/2607.01690)  

**Abstract**: Finetuning a language model on documents that are explicitly annotated as fictional results in a model that still actually believes the documents' core claims, an effect known as Negation Neglect. In our evaluations, models trained on documents prefixed and suffixed with such annotations correctly identify the relevant claims as fictional only about 9% of the time. To address this, we introduce Goggles, a learned module that intervenes on the finetuning gradient rather than the data. During supervised finetuning, a Goggles module edits the gradients an LLM LoRA receives, imparting a chosen epistemic frame (the stance the model takes toward the nature of what it reads) to whatever the documents teach. A Goggles instance is trained once for a given base model, frame, and LoRA configuration, then applied frozen to documents it was never trained on. Trained through Goggles on those same documents, now carrying no fictional annotation, the model flags the content as fictional roughly 91% of the time, while preserving capability (GPQA and TruthfulQA match or exceed baseline). The same architecture supports other frames: a Goggles instance can be trained to treat documents as "part of an AI safety evaluation by Redwood Research" rather than simply as fiction. The imparted frame persists under continued finetuning that pushes back toward the claim, where prior interventions revert. Goggles suggests a path toward training language models on known-misaligned data without absorbing the behaviors that data demonstrates. 

---
# AgenticDataBench: A Comprehensive Benchmark for Data Agents 

**Authors**: Zhaoyan Sun, Shan Zhong, Daizhou Wen, Jiaxing Han, Guoliang Li, Ying Yan, Peng Zhang, Yu Su, Xiang Qi, Baolin Sun, Chengyuan Yang, Tao Fang, Huaiyu Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2607.01647)  

**Abstract**: Data science aims to derive actionable insights from heterogeneous raw data, unlocking the value of the massive amounts of data generated in modern society. Automating this process is essential to reducing labor-intensive efforts for data scientists and enabling scalable data-driven applications. Recently, large language model (LLM)-based data agents have emerged as a promising solution to automate data science workflows. However, the field lacks comprehensive benchmarks to rigorously evaluate these agents across diverse scenarios with fine-grained granularity. To address this gap, we propose AgenticDataBench, a comprehensive benchmark featuring realistic tasks spanning diverse domains with fine-grained ground-truth labels. This enables evaluations to capture the diversity and complexity of data science workflows and the detailed performance of agents. First, to cover diverse domains, we collect real datasets and tasks from 15 vertical domains, including 5 real-world B2B use cases from a leading fintech company. Second, to remove redundancy in real-world tasks and generate high-quality tasks for domains lacking real data, we introduce data science skills, recurring data-centric operational patterns, and quantify benchmark coverage by the number of skills included. Representative skills are extracted from large-scale task solutions on Stack Overflow using skill-aligned hierarchical clustering. Third, for real-world business tasks, we select task-solution pairs that maximize diversity in skill composition, ensuring broad coverage of practical scenarios. Fourth, to generate realistic tasks for devise domains without real tasks, we propose a systematic LLM-based task generation approach to create workflows and tasks based on these skills. Finally, we evaluate state-of-the-art data agents using our annotated benchmark and open-sourced testbed, providing detailed skill-level insights. 

---
# BOUNDARY_SYNC: Measuring Communication-Induced Representational Coupling in Multi-Agent LLM Systems 

**Authors**: Zewen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.01600)  

**Abstract**: As large language models (LLMs) are deployed as communicating agents, does inter-agent communication cause outputs to converge? We introduce BOUNDARY_SYNC, a protocol measuring representational coupling via the Coupling Amplification Factor (CAF = JSD_cond / JSD_baseline), where CAF < 1 indicates homogenization and CAF > 1 indicates diversification. In controlled GPT-4o experiments (N=30, ~9,900 API calls), we measure coupling in text and image communication. Key findings: (1) text communication causes significant homogenization (CAF=0.803 [0.740, 0.873], d=1.30, p<0.001), confirmed by no-communication ablation and prompt-perturbation controls; (2) image communication also homogenizes under within-modality baselines (CAF=0.834 [0.811, 0.858]), with comparable proportional effect; (3) group size moderates coupling direction -- K=5 produces homogenization while K=3 yields CAF > 1.0 (point estimates 1.14 and 1.06, CI pending), suggesting a directional shift toward diversification; (4) cross-model replication shows extreme variation (CAF 0.034-0.803), with DeepSeek dominated by format artifacts; (5) coupling is stateless -- driven by prompt context rather than cumulative updating, with continuous consensus producing monotonic convergence. These results establish LLM agent coupling as real, measurable, and controllable at the prompt level, with direct implications for multi-agent system design. 

---
# Safe and Adaptive Cloud Healing: Verifying LLM-Generated Recovery Plans with a Neural-Symbolic World Model 

**Authors**: Junyan Tan, Haoran Lin, Siyuan Guo, Yichen Fang, Xinyue Luo, Tianyu Shen, Zeyu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2607.01595)  

**Abstract**: As the scale and complexity of cloud-based AI systems continue to escalate, ensuring service reliability through rapid fault detection and adaptive recovery has become a critical challenge. While existing approaches integrate Large Language Models (LLMs) for semantic understanding and Deep Reinforcement Learning (DRL) for policy optimization, they often rely on sequential, loosely coupled architectures that underutilize the generative and reasoning capabilities of LLMs. In this paper, we propose a paradigm shift with PASE, a Planning-Aware Semantic self-healing engine, a novel fault self-healing framework that reconceptualizes recovery as a neuro-symbolic program synthesis task. PASE employs an LLM as a core Plan Synthesis Engine to generate structured recovery plans from a library of semantic primitives. A Neural-Symbolic World Model verifies plan feasibility through simulation, while a Meta-Prompt Optimizer, trained via DRL, learns to generate optimal prompts that guide the LLM's planning process. This tight reason-plan-verify-adapt loop enables dynamic, context-aware recovery strategy generation beyond predefined action spaces. Experiments on a real-world cloud fault injection dataset demonstrate that PASE significantly outperforms state-of-the-art methods, reducing average system recovery time by over 40% and improving fault detection accuracy in unknown fault scenarios. Our framework advances autonomous system management by unifying LLM-based reasoning with model-assisted verification and meta-learned guidance. 

---
# ADVENT: LLM-Driven Automatic Predicate Invention for ILP 

**Authors**: Tingting Yu, Pei-Cing Huang, Chan Hsu, Chan-Tung Ku, Yihuang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2607.01585)  

**Abstract**: Predicate invention (PI), the creation of new predicates to extend the hypothesis space, remains a critical bottleneck in Inductive Logic Programming (ILP). Existing methods rely on domain expertise and produce semantically opaque predicates, hindering adaptation to unfamiliar domains and cross-task reuse. We present ADVENT, an LLM-driven PI mechanism for ILP. ADVENT pairs LLM abductive generation with Prolog deductive verification, forming an iterative loop in which concrete execution results guide the LLM to refine candidate predicates. The mechanism leverages Large Language Models to identify implicit patterns in structured relational data and invent auxiliary predicates with meaningful names and definitions. Invented predicates and learned rules accumulate in a knowledge pool for cross-task reuse. Experiments on nine poker-hand concepts across seven LLMs show that LLM-driven PI achieves 58% success rate where ILP alone fails entirely, formal verification raises this to 80%, and the knowledge pool yields gains up to +31 percentage points, while producing human-interpretable rules. These results suggest that ADVENT offers a promising direction for automating predicate invention and enabling cross-task knowledge reuse in ILP. 

---
# Multi-Head Recurrent Memory Agents 

**Authors**: Jiatong Li, Samuel Yeh, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.01523)  

**Abstract**: Recurrent memory agents extend LLMs to arbitrarily long contexts by iteratively consolidating input into a fixed-size memory window. Despite their scalability, these agents exhibit a well-documented reliability problem: end-to-end performance degrades systematically as context length grows. We diagnose this failure by decomposing performance into two factors--memory capture and memory retention--and quantitatively confirm that retention is the dominant bottleneck. Retention collapses because existing designs maintain memory as a monolithic text block, forcing every update to risk overwriting previously retained content. Motivated by this diagnosis, we propose Multi-Head Recurrent Memory (MHM), a general, training-free framework that partitions memory into independent heads governed by a stage-wise select-then-update strategy. At each step, exactly one head is selected for update while the remaining heads are structurally shielded from overwriting, shifting the burden of retention from model behavior to architectural design. As a lightweight instantiation, we introduce Least-Recently-Updated MHM (MHM-LRU), which guarantees uniform head utilization with zero additional token overhead. Extensive experiments on long-context benchmarks show that MHM-LRU substantially improves both retention and end-to-end accuracy across the 100K--1M token range, where baselines degrade sharply. On RULER-HQA at 896K tokens, MHM-LRU improves the memory retention rate from less than 30% to 73.96%. These gains generalize across model families, scales, and task types, positioning architectural optimization as a practical and cost-efficient path toward reliable long-context recurrent memory. 

---
# On the Utility and Factual Reliability of Pruned Mixture-of-Experts Models in the Biomedical Domain 

**Authors**: Atsuki Yamaguchi, Szymon Palucha, Léo Bijar, Aline Villavicencio, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2607.01444)  

**Abstract**: Mixture-of-Experts (MoE) models offer inference speedups via selective activation but impose substantial memory requirements because the whole network must remain loaded. Structured expert pruning is a practical approach for reducing deployment costs in resource-constrained settings. However, prior studies primarily evaluate benchmark utility, leaving the effect of pruning on factual reliability underexplored, particularly in high-stakes domains such as biomedicine. In this paper, we investigate how domain-specific expert pruning affects both utility and reliability. We assess four MoE models, six pruning methods, and multiple pruning ratios across generation and classification tasks under in-domain (biomedical) and cross-domain settings. Results reveal that moderate pruning preserves in-domain utility without immediate reliability decline, although hallucination risks increase at extreme pruning ratios. When shifting to the general domain, both utility and reliability degrade rapidly. These findings indicate that safe compression depends heavily on the task and domain. Evaluating pruned MoE models solely on utility is inadequate for high-stakes deployment without reliability assessment. 

---
# Black-Box Inference of LLM Architectural Properties with Restrictive API Access 

**Authors**: Christopher Ellis, Shreyas Chaudhari, Mei-Yu Wang, Leighton Barnes, Giulia Fanti, José M. F. Moura  

**Link**: [PDF](https://arxiv.org/pdf/2607.01313)  

**Abstract**: In practice, most commercial LLM providers do not publicly release details of underlying LLM architectures. However, prior work has shown that given limited API access to an LLM (namely, top-$k$ logits and/or a logit bias function), one can recover certain architectural details of an LLM, such as the hidden dimension of the feed-forward network. Perhaps in response to these results, most commercial LLM providers have restricted their APIs to expose only the single logit for each decoded token, and they no longer give users the ability to bias logits. We show that even under current restrictive APIs, several architectural parameters are still recoverable. We present NightVision, an attack that uses restrictive black-box API access to estimate the hidden dimension, depth, and parameter count of an LLM. Algorithmically, NightVision relies on a novel common set prompting technique in which multiple prompts expose log probabilities for the same set of output tokens; a spectral analysis of these results is used to infer hidden dimension. NightVision additionally uses end-to-end time to first token (TTFT) measurements and the estimated hidden dimension to estimate depth and parameter count. We empirically evaluate NightVision on 32 open-source LLMs, recovering hidden dimension to within 23% average relative error across all models (9% on MoE models), and depth and parameter count to within 53% for models exceeding three billion parameters. We run extensive ablations to demonstrate how these accuracies scale with token budget and model properties. Overall, our results suggest that current LLM APIs are not sufficiently restricted to fully obfuscate the architectural details of their underlying models. 

---
# Structuring the Space of Sociotechnical Alignment 

**Authors**: Esra Dönmez, Agnieszka Falenska  

**Link**: [PDF](https://arxiv.org/pdf/2607.01250)  

**Abstract**: Sociotechnical alignment concerns the social desirability of AI behavior and is thus inherently normative, not merely technical. While NLP research increasingly addresses its technical aspects, it often leaves underspecified what such "social desirability" entails. We argue that this reflects a fundamental gap: the absence of a systematic way to specify how sociotechnical alignment defines, justifies, and evaluates socially desirable AI behavior.
To address this gap, we introduce a human-centered framework for specifying sociotechnical alignment. We draw on social-scientific accounts of sociobehavioral desirability to ground the basis for behavioral desirability judgments and use this framework to analyze how alignment is specified in practice. Our systematic literature review identifies recurring patterns: normative concepts grounding desirability judgments are often unspecified or conflated with alignment targets for (desired) system behavior, target populations are underdefined, and design choices are rarely theoretically justified. These findings point to a lack of conceptual specificity that limits cumulative progress. We therefore offer recommendations that link social-scientific frameworks to alignment design choices, supporting more conceptually precise approaches to sociotechnical alignment. 

---
# ExPerT: Personalizing LLM Responses to Users' Domain Expertise via Query-Wise Semantic and Keystroke Behavioral Cues 

**Authors**: Yeji Park, Jiwon Tark, Taesik Gong  

**Link**: [PDF](https://arxiv.org/pdf/2607.01242)  

**Abstract**: Large language models (LLMs) are increasingly used by end users, yet existing personalization methods relying on static profiles or text-only signals fail to capture query-specific expertise variation. We present ExPerT, a query-wise personalization framework that adapts LLM responses to users' query domain expertise by combining semantic and behavioral cues. ExPerT consists of two key components: (i) a semantic-behavioral expertise inference module that jointly interprets query text and keystroke dynamics via in-context LLM prompting, and (ii) an expertise-conditioned response generation that adapts the level of detail, terminology, and conceptual complexity. Our user study with 40 participants and 1270 queries demonstrated that ExPerT reduced expertise inference error by 65.7% compared to the strongest baseline (MAE = 0.398 vs. 1.162) and improved response satisfaction by 17.52% (from 3.71 to 4.36) on a 5-point Likert scale. 

---
