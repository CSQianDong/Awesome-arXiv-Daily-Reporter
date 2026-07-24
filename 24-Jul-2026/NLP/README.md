# Surprisal Theory is Tautological (without Rational Grounding) 

**Authors**: Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2607.21574)  

**Abstract**: Surprisal theory holds that the human processing difficulty of a linguistic unit in context is an affine function of its surprisal under some language model. I argue this claim is a tautology without further constraint: for any non-negative difficulty measure over units in context, there exists a language model whose surprisal is an affine function of it under mild technical conditions. Therefore, because any pattern of difficulty is consistent with some language model, without an additional constraint on the language model, surprisal theory makes no falsifiable predictions. The tautology was long obscured by an assumption implicit in two decades of psycholinguistic work---that the relevant language model is the distribution that generated the training corpus, so that improving corpus fit improves predictions of human behavior. Recent empirical work has undermined this assumption, demonstrating that better corpus models can be worse predictors of processing difficulty. I conclude that breaking the tautology requires a rationalist intervention, i.e., the relevant language model must be derived from a non-empirically motivated model of the comprehender, which could be based on, for instance, memory constraints or processing goals, and that, thus, does not depend on the behavioral data surprisal theory is meant to explain. 

---
# MedGame: Storytelling Gamification Empowered by Large Language Models for Medical Education 

**Authors**: Qian Wu, Xinrong Zhou, Zizhan Ma, Kai Chen, Zheyao Gao, Xun Lin, Hongqiu Wu, Longfei Gou, Yixiao Liu, Ann Sin Nga Lau, Qi Dou  

**Link**: [PDF](https://arxiv.org/pdf/2607.21570)  

**Abstract**: Large Language Models (LLMs) show promise for medical education, but most existing systems focus on localized interactions such as question answering or single-turn feedback, rather than organizing an entire clinical case into a decision-centered learning trajectory. We introduce \textit{MedGame}, a framework that transforms static clinical cases into structured, executable storytelling games. MedGame uses a dual-engine design: a Medical Narrative Designer synthesizes case-grounded clinical storylines with states and decision nodes, while a Story Director converts them into dependency-aware multimodal orchestration plans rendered by our released interactive platform. We construct MedGame Bench, a 5,000-case benchmark and evaluation protocol for Medical Narrative Generation and Story Direction. Experiments show that task-specific fine-tuning substantially improves open-source LLMs on MedGame Bench and narrows the gap with commercial models. A pilot student study further shows that learners perceive MedGame as more engaging and useful than text-only alternatives. 

---
# DONDO: Open w2v-BERT Speech-Recognition Base Models for African Languages 

**Authors**: Paul Azunre  

**Link**: [PDF](https://arxiv.org/pdf/2607.21540)  

**Abstract**: We present DONDO, a family of open, permissively licensed automatic speech recognition (ASR) base models for African languages, built on the w2v-BERT 2.0 self-supervised speech encoder. DONDO comprises twenty-one monolingual models and five multilingual models spanning twenty-seven language varieties across Ghana, Sierra Leone, Nigeria, Senegal, Kenya and Zimbabwe. Models are fine-tuned primarily on read speech drawn from religious texts, which offer broad, license-clear and orthographically consistent coverage for languages that otherwise lack transcribed audio. We describe a two-step (and, for one family, three-step) learning-rate-annealed fine-tuning procedure that first adapts a shared multilingual model at a high learning rate and then anneals it to recover, and in several cases surpass, strong monolingual baselines. We further describe a lightweight language-conditioning mechanism that injects a one-hot language identity as a sequence of prefix frames prepended to the acoustic features, allowing a single multilingual checkpoint to be steered to a target language at inference. Across the five multilingual families the annealed models reach average word error rates (WER) of 10-13%, closing most of the gap to monolingual models while covering many languages in a single checkpoint. All models are released on the Hugging Face KhayaAI organisation under the Apache-2.0 license (attribution only) so that others may fine-tune them freely, including for commercial use. We provide a conservative estimate that the languages covered are spoken by on the order of one hundred million first-language speakers, and by substantially more when second-language use is included. 

---
# Artificial Epanorthosis: Why large language models overuse a classical rhetorical figure, and how to mitigate it 

**Authors**: Federico Boggia  

**Link**: [PDF](https://arxiv.org/pdf/2607.21498)  

**Abstract**: A rhetorical figure that Cicero and Quintilian catalogued two thousand years ago reappears, systematically, in the text of large language models: epanorthosis, the self-correction of the specimen «This is not a course. It is a journey of transformation». This essay argues that the overuse is a trained disposition, driven mainly by a training distribution rich in promotional prose and by preference tuning (RLHF) that rewards confident, emphatic phrasing; the left-to-right nature of generation is an amplifier rather than the root cause. Building on evidence that models diverge from human rhetorical style, and on Fontanier's classification of epanorthosis as a figure of thought, it sets out a programme that scores the figure against genre-specific human baselines through an Epanorthosis Index (density relative to the human rate). A first measurement, on three sizes of one instruction-tuned model family, finds mis-calibration by register in both directions: the models overshoot in oratory (about twofold, near threefold in Italian, concentrated in the larger tiers) and undershoot in informal question-and-answer writing, while matching humans in argument, journalism, and encyclopedic prose. Three constructive contributions follow: a survey of mitigation techniques centred on lightweight LoRA adapters; a demonstration, in Italian, that a one-line instruction cuts the figure by half to nearly three-quarters and that a supervised-fine-tuning adapter removes it almost entirely, with a scaling coefficient that dials the reduction back onto the human rate; and the argument that the target is calibration to the human rate for each genre, not elimination. It closes on the stakes: the real risk is that we begin to write like the machines. 

---
# What, Where, and How: Disentangling the Roles of Task, Language, and Model in Code Model Representations 

**Authors**: Piotr Wilam  

**Link**: [PDF](https://arxiv.org/pdf/2607.21491)  

**Abstract**: Do independently trained language models come to represent the same thing in the same way? We answer for code, extending a recently introduced concept-circuit extraction method to a 2x2 design -- Python and Rust crossed with Qwen2.5-Coder-7B and DeepSeek-Coder-V1-6.7B -- and measuring a complete inventory of grammatical concepts (58 Python, 57 Rust) identically in all four cells: the smallest design that separates what depends on the task, the language, and the model.
The answer splits into three parts. What earns dedicated circuitry is set by the task: the models agree on which concepts receive circuits (Spearman $\rho$ = 0.638 for Python, 0.673 for Rust, both p < $10^{-7}$). Where those circuits sit is set by the model: Qwen processes concepts in a late band (~L17-19), DeepSeek at L6-7, for both languages. How circuits grow across layers is also set by the model: Qwen gives its atomic concepts an early spike that DeepSeek does not. "Are circuits universal?" thus has no single answer: yes for What, no for Where and How -- universality is a property of representational content, not of computational organisation.
None of this structure was fixed in advance. The agreement could have landed anywhere between independence and identity; it lands at $\rho \approx 0.65$. Rust constructs receive 2-3x more concept-specific circuitry than their Python equivalents, in both models. Both models share neurons between the languages (6/7 and 7/7 paired constructs), DeepSeek 1.94x more than Qwen -- a direction no prior result predicts. And Qwen binds nine keywords of Rust's type-and-trait machinery into one tight neuron cluster (Jaccard 0.535 vs null 0.112, p < 0.001), a semantic dimension invisible in surface syntax. Ablation and linear probes confirm the circuits are functional.
All claims are scoped to this 2x2; whether the per-model profile predicts a third model is the designed next test. 

---
# RUMBA: Russian User Memory Benchmark 

**Authors**: Elizaveta Shevtsova, Inna Glebkina, Mark Baushenko, Pavel Gulyaev, Alena Fenogenova  

**Link**: [PDF](https://arxiv.org/pdf/2607.21447)  

**Abstract**: The ability to handle long-term memory in LLMs is becoming increasingly critical, yet existing benchmarks remain English-centric and rely on aggregate retrieval metrics, failing to capture interactions between long-range context, temporal information, and reasoning. To address this, we introduce RUMBA (Russian User Memory BenchmArk) - a new benchmark for long-term conversational memory that provides a fine-grained taxonomy of memory-centric question types and a unified methodology accounting for semantic type, session scope, temporal reasoning, and the explicitness of temporal expressions. RUMBA consists of timestamped user-assistant dialogues with QA pairs requiring retrieval, combination, and reasoning across sessions. While designed for Russian, we also provide an aligned English subset under the same methodology. We evaluate contemporary memory systems and long-context models, and show how RUMBA serves as a diagnostic tool to analyze model behavior across benchmark slices and identify strengths and failure modes of different memory mechanisms. 

---
# When Trivia Is Not Trivial: Everyday Knowledge Failures in Multilingual LLMs 

**Authors**: Anna Mosolova, Djamé Seddah  

**Link**: [PDF](https://arxiv.org/pdf/2607.21445)  

**Abstract**: Quiz rooms, trivia nights, and quiz shows challenge human knowledge across a wide range of topics, from canonical facts to everyday culture. In this paper, we examine whether large language models (LLMs) can perform competitively in such settings, using quiz-style questions to test them on both common and niche topics. We introduce TriviaRoomQA, a multilingual benchmark designed to evaluate everyday, culturally grounded, and long-tail knowledge across 288 topics. The benchmark contains 3,300 parallel multiple-choice questions in six European languages and additional 5,340 French-only questions for a more fine-grained case study. We evaluate 30 open-weight LLMs from European, Asian, and North American providers, covering models from 7 to 70B parameters. We find that models are strong on knowledge-intensive topics such as history, geography, and mathematics, but substantially weaker on everyday popular-culture topics such as celebrities, music, movies, and news. Moreover, model performance varies across languages even for the same underlying questions, suggesting that access to factual knowledge is not always language-independent. In sum, our dataset and experiments demonstrate an important knowledge gap which is not captured by existing academic-based saturated benchmarks. 

---
# Token Budget Saturation and Mechanistic Early Detection of Reasoning Non-Convergence in Chain-of-Thought Models 

**Authors**: Renuka Oladri, Niveda Jawahar, Abdirisak Mohamed  

**Link**: [PDF](https://arxiv.org/pdf/2607.21433)  

**Abstract**: Chain-of-thought reasoning models such as DeepSeek-R1-Distill-Qwen-7B exhibit a bimodal convergence pattern: generations either terminate within a token budget (converged) or exhaust it without reaching a conclusion (non-converged). We characterize this phenomenon empirically, showing that converged generations achieve 90.3% accuracy on AIME 1983-2024 while non-converged ones achieve only 6.6%, with an overall convergence rate of 62.0%. We then ask whether this outcome is detectable early in the thinking chain using internal model representations. Training linear probes on hidden-state activations at token positions 50-300, we find that layer-20 activations at token 150 achieve AUC 0.608 (+-0.080, 5-fold CV), reliably above chance even at token 50. Activation probes consistently outperform behavioral baselines derived from token entropy and repetition statistics. A sweep-level permutation test yields p=0.063 (100,000 permutations), consistent with a modest signal that our sample size cannot confirm at conventional thresholds. These findings suggest that convergence fate is partially encoded in intermediate representations well before the generation ends, opening a path toward early-exit inference and adaptive compute allocation. 

---
# An Evaluation Framework for Structured Audio Captions Validated by Controlled Perturbations 

**Authors**: Liang-Yuan Wu, Sripathi Sridhar, Mark Cartwright, Magdalena Fuentes  

**Link**: [PDF](https://arxiv.org/pdf/2607.21424)  

**Abstract**: Recent advancements in automated audio captioning (AAC) have shifted from monolithic sentence generation toward structured formats that explicitly disentangle distinct acoustic and semantic properties. However, evaluating this heterogeneous data remains a significant challenge. Existing caption metrics focus on flat textual outputs and fail to reliably assess multimodal attributes. To bridge this gap, we propose a multi-axis evaluation framework tailored for structured audio descriptions. Building on the AudioCards dataset, we evaluate outputs across five orthogonal axes: tag-sets, descriptions, logical reasoning, numeric measurements, and spectral profiles. Our approach combines Large Language Model (LLM) judges to capture semantic nuance with deterministic computational metrics to precisely measure acoustic deviations. To rigorously validate the reliability of this framework, we introduce a controlled perturbation testing protocol that injects typed, graded errors into groundtruth annotations. Our results demonstrate that this framework successfully distinguishes meaning-preserving paraphrases from genuine semantic and acoustic corruptions. 

---
# Anti-Periodic Positional Encoding: Möbius Boundary Conditions Make In-Context Retrieval Reliable 

**Authors**: Ji Ho Bae  

**Link**: [PDF](https://arxiv.org/pdf/2607.21405)  

**Abstract**: Möbius RoPE is a rotary positional encoding built on the anti-periodic frequency ladder $\theta_i=\pi(2i+1)/N$: every rotation plane advances by an odd multiple of $\pi$ across the training context, so the positional holonomy is $-1$ and the two ends of the sequence are deterministically coupled through a closed-form Dirichlet "dipole"; to our knowledge this is the first anti-periodic boundary condition in positional encoding. We verify the theory numerically to $\sim 10^{-6}$ and pretrain 48 models spanning six 160M-class and three 410M-class arms (2B FineWeb-Edu tokens each; the hybrid arm puts Möbius frequencies on 25% of heads). Hybrid perplexity is unchanged (29.66 vs. 29.72), but needle-in-a-haystack retrieval becomes reliable: $90.3\pm5.7\%$ versus $63.3\pm31.4\%$ at context 512 ($n=6$ seeds), observed worst seed 86% versus 14%, robust variance tests $p=0.013$-$0.029$ (unadjusted), recurring at 410M (Levene $p=0.040$). Matched controls isolate the mechanism: an aperiodic ladder in the same frequency band reproduces none of the effect, and a periodic (holonomy $+1$) ladder only a fraction. Swapping trained models' frequency table back to standard RoPE (weights frozen) collapses retrieval, with damage concentrated on far needles: trained models depend on this long-range geometry. A NoPE arm is even more reliable at short context but pays a 13% perplexity tax and extrapolates worst; only the anti-periodic hybrid pairs baseline perplexity with a high reliability floor. The effect is scoped to single-needle retrieval within the training window; a one-line frequency swap thus provides zero-cost insurance against the retrieval seed lottery. 

---
# MemTools: A Unified Research Framework for Interoperable Agent Memory 

**Authors**: Chengfeng Zhao, Jinhui Chen, Sirui Liang, Shizhu He, Yequan Wang, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21404)  

**Abstract**: While memory systems are essential for agent architectures, pervasive architectural fragmentation restricts systematic research. Existing implementations typically couple different stages of the memory lifecycle, entangle evaluation logic with specific datasets, and provide limited support for the management of heterogeneous memory types. We introduce MemTools, an interoperability research framework that decouples memory system components from their underlying deployment environments. MemTools standardizes the memory lifecycle through declarative data contracts, enabling the interchangeable assembly of components across different systems. It orthogonally separates benchmark datasets from execution protocols to facilitate controlled assessments. Furthermore, MemTools provides a unified computational interface for coordinating symbolic, neural, and multimodal memory representations within a shared runtime. Empirical evaluations on cross-system component integration, evaluation protocol reconfiguration, and heterogeneous memory coordination demonstrate that MemTools enables systematic isolation and analysis of memory design variables. These findings suggest that MemTools provides a practical and extensible infrastructure for advancing principled research on agent memory. 

---
# Word meaning co-determines vowel-inherent spectral change. A corpus-based investigation of conversational Mandarin 

**Authors**: Xiaoyun Jin, Mirjam Ernestus, R. Harald Baayen  

**Link**: [PDF](https://arxiv.org/pdf/2607.21391)  

**Abstract**: This study investigates vowel-inherent spectral change (VISC) in spontaneous conversational Mandarin. Using the generalized additive model and word embeddings from distributional semantics, we show that, when controlling for variables such as vowel duration, gender, speaker identity, co-articulation, vowel identity, and utterance position, vowel formant trajectory dynamics have word-specific components that are tied to their meaning in context: The F1 and F2 trajectories of words can be predicted from their contextualized embeddings with an accuracy that substantially exceeds a permutation baseline. Challenging modular cognitive models of speech production, these results indicate that, words' semantics co-determine the fine details of their articulation. 

---
# Capital Markets LLM Reliability Score (CM-LRS): From Plausible to Bankable 

**Authors**: Prerit Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2607.21340)  

**Abstract**: In capital-markets workflows the question is rarely whether a large language model can produce a fluent draft, but whether the draft is bankable: defensible in front of a counter-party or a regulator, with the documents in hand. Existing methods address parts of that gap: open-domain QA benchmarks reward surface accuracy, and finance benchmarks (FinanceBench, FinQA, ConvFinQA) advance document-grounded and numerical QA but evaluate at the question-answer layer rather than the workflow outputs practitioners defend.
We introduce CM-LRS, a Capital Markets LLM Reliability Score, evaluating outputs at the workflow-output layer across seven dimensions: factual accuracy, evidence traceability, numerical consistency, workflow completeness, source discipline, decision usefulness, and reviewability/auditability. Each is scored 0-5 against a rubric anchored on signals reviewers in regulated settings use; the aggregate is tunable to the workflow.
We demonstrate CM-LRS on five workflows (DCM transaction-terms extraction, precedent retrieval, issuer profile synthesis, M&A transaction-comparable reasoning, ECM transaction-terms extraction) over public SEC EDGAR filings, a public UK takeover release, and fictional synthetic supplements, scoring four models against four independent LLM judges spanning three model families.
Three findings. First, the frontier closed-source models cluster within 0.22 points on four-judge averaged CM-LRS (Sonnet 4.6 = 4.31, Opus 4.7 = 4.30, GPT-5.5 = 4.09); all four judges place the open-weights baseline (Llama 3.3 70B = 3.15) last. Second, that gap concentrates on retrieval (2.23) and synthesis (2.15), not extraction (0.84). Third, Decision Usefulness shows the widest cross-model dispersion of any dimension (4.0 points on issuer profiling) and top-tier inter-judge agreement (mean r = 0.52).
Plausibility is cheap. Bankability is the bar. 

---
# Phonetic forced alignment for low-resource language varieties: Model training and evaluation on Chengdu Mandarin 

**Authors**: Zhiheng Qian, Aini Li, Hai Hu, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.21332)  

**Abstract**: Phonetic forced alignment is a key technique in phonetic research, yet existing alignment systems lack specialized models for low-resource language varieties. We address this by training text-dependent and text-independent aligners for Chengdu Mandarin using a 17-hour corpus and a custom G2P dictionary. We trained a text-dependent GMM-HMM model (Chengdu-MFA) and fine-tuned a pretrained audio encoder on frame classification with Chengdu-MFA's pseudo label for text-independent alignment (Chengdu-FC). Evaluation on an expert-annotated test set show that both methods significantly outperform Standard Mandarin baselines. Chengdu-MFA reduced average phone boundary differences by 31.8%, while Chengdu-FC achieved a 61.2% reduction. This work establishes a practical bootstrapping pipeline for developing accurate aligners for under-resourced varieties without labor- and time-intensive manual annotation. 

---
# GRADRAG: Cross-Component Prompt Adaptation for Coordinated Multi-Agent RAG 

**Authors**: Paolo Pedinotti, Enrico Santus  

**Link**: [PDF](https://arxiv.org/pdf/2607.21324)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems increasingly employ multiple LLM agents. Yet, most prior work optimizes components in isolation rather than coordinating improvements across the pipeline. We introduce GRADRAG, a framework for cross-component prompt adaptation that models the RAG pipeline as a computational graph and propagates structured evaluation feedback to update upstream agents. An Evaluator critiques downstream answers and supporting evidence, producing actionable feedback that a Prompt Optimizer uses to iteratively update adaptive agents, such as retrievers, graph constructors, and answerers. The Evaluator also triggers early stopping when the output is deemed satisfactory. We evaluate GRADRAG on the SQUALITY and QMSUM benchmarks under two retrieval paradigms: flat chunk-based retrieval using IRCoT-style query refinement (Trivedi et al., 2023), and graph-based retrieval that constructs and iteratively enriches an entity-relation graph from the document. Across both settings, GRADRAG consistently outperforms one-step refinement baselines that update only the final generator, achieving a 12-15 percentage point net preference margin in LLM-judged pairwise comparisons, with most gains realized within two refinement iterations. 

---
# Adaptive Depth Sparse Framework: Similarity-Driven Resource Allocation for Pre-Trained LLMs 

**Authors**: Yidu Wu, Xiang Wang, Kejie Zhao, Zhangchi Wang, Qinghai Guo, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21291)  

**Abstract**: Large language models (LLMs) achieve strong generation and reasoning performance, but the Transformer architecture incurs high inference cost. Existing acceleration methods often rely on task-specific fine-tuning or training from scratch, increasing adaptation cost and limiting cross-task usability. We present an Adaptive Depth Sparse Framework (AdaDSF) that converts off-the-shelf pre-trained LLMs into depth-sparse models without full retraining. Our key insight is that layers contribute unequally to representation transformation, characterized by the cosine similarity between layer input and output hidden states. Based on this, AdaDSF assigns layer-wise token retention ratios from similarity statistics, uses a lightweight router to select informative tokens at each layer, and introduces a feature-preserving alignment objective to match intermediate and final representations between sparse and dense models. On GPT-NeoX and Qwen2.5 over language modeling and commonsense reasoning, AdaDSF substantially reduces inference FLOPs while preserving performance close to dense counterparts. Under comparable sparsity, AdaDSF consistently yields smaller accuracy degradation than strong baselines including MoD, D-LLM, and DLO. 

---
# news-crawler-LM: A Small Long-Context Model For High-Quality News Crawling 

**Authors**: Pascal Stolzenburg, Jonas Golde, Max Dallabetta, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2607.21284)  

**Abstract**: Extracting structured content from news pages remains challenging due to heterogeneous HTML layouts, inconsistent markup, and substantial boilerplate such as navigation elements and advertisements. Rule-based news crawlers can achieve high extraction accuracy by encoding site-specific structure, but require manual configuration in order to generalize to new publishers. Large language models provide a more flexible alternative by reducing the need for handcrafted rules, but their high computational cost limits practical deployment. In this paper, we introduce news-crawler-LM, a small long-context language model fine-tuned on high-quality, human-validated extractions from the Fundus news-crawling library. Our model converts raw HTML into plaintext and structured JSON, including fields such as headline, author, publication date, and article body. In our experiments, news-crawler-LM outperforms strong baselines in HTML-to-Markdown and HTML-to-JSON extraction, improving performance by +4.8 BLEU and +6.1 METEOR in the HTML-to-Markdown task, and by +2.2 BLEU and +4.1 METEOR in the HTML-to-JSON task. However, we also observe that our model only slightly better compared to other rule-based parsing libraries on the HTML-to-plaintext task in evaluations on previously unseen publishers. We release all models and artifacts to the research community. 

---
# A Unified Moral-Value Dataset for Instruction Tuning 

**Authors**: Zhaohui Zeng, Florian Mai  

**Link**: [PDF](https://arxiv.org/pdf/2607.21279)  

**Abstract**: Large language models (LLMs) have developed rapidly and become valuable tools in everyday life. However, how to align LLMs to a particular set of human values is still an open problem. Recent studies show that instruction tuning has strong potential for zero-shot tasks and may serve as an effective approach to addressing value alignment. Nevertheless, although many datasets for instruction tuning already exist, they are not specifically designed around moral scenarios and behaviors. We construct a unified moral-value dataset that can be directly used for instruction tuning. This dataset is built upon existing moral-value datasets by merging them into a unified corpus and converting them into an instruction-response format. We show that training on a mixed dataset combining general task datasets with our dataset preserves general-task performance, and we report preliminary observations on how the mixing ratio affects value-oriented task performance. Our work provides a moral-value dataset for instruction tuning and offers a useful resource for further alignment research. The dataset is available at this https URL. 

---
# A Comparative Evaluation of Embeddings and LLMs in a Greek Book Publisher Setting - The CUP Dataset 

**Authors**: Katerina Papantoniou, Panagiotis Papadakos, Theodore Patkos, Dimitris Garefalakis, Nikos Vardakis, Dimitris Plexousakis  

**Link**: [PDF](https://arxiv.org/pdf/2607.21274)  

**Abstract**: We present CUP, a Greek book retrieval benchmark consisting of 868 catalog records and 104 expert-annotated queries with graded relevance judgments. We evaluate sparse (BM25), dense (sentence-transformers), hybrid, and LLM-assisted retrieval methods in this book-search setting. Multilingual embeddings outperform Greek-specific models, while hybrid retrieval performs best overall. A query-level analysis shows that BM25 excels at named-entity queries, while dense and hybrid methods improve natural-language, noisy, cross-lingual, and concept queries. Field-aware prompting has model-specific effects, while LLM TOC summarization improves TOC-only retrieval and LLM post-filtering improves early-stage retrieval at a high cost. Overall, CUP enables real-world evaluation of Greek retrieval across lexical, semantic, noisy, and cross-lingual queries. 

---
# slang.gr as a Large-Scale Crowdsourced Resource for Non-Standard Greek 

**Authors**: Panagiotis Papadakos, Katerina Papantoniou, Dimitris Plexousakis  

**Link**: [PDF](https://arxiv.org/pdf/2607.21255)  

**Abstract**: Slang is a central component of everyday language, reflecting linguistic creativity, social identity, and cultural change, yet its dy- namic and non-standard nature makes it difficult to model computationally. We present the first large-scale computational study of this http URL, a crowdsourced lexicon of Greek non-standard language, combining lexical content, user-generated tags, and interaction data. To enable the systematic analysis, we map noisy folksonomic tags to a structured multi-layer taxonomy capturing both semantic categories and sociolinguistic metadata. Using this representation, we analyze the linguistic structure of Greek slang and the behavior of its contributor community. We find that slang is strongly centered on person-related and evaluative language, exhibits high morphological creativity, and is shaped by highly skewed participation with short user lifespans and overlapping communities. Building on these signals, we introduce a community-based confidence score for definitions that integrates user roles, interaction patterns, and moderation signals. Our results show that taxonomy-based representations improve interpretability while retaining meaningful aspects of behavioral structure, enabling a more structured and interpretable analysis of confidence signals. Overall, this work establishes this http URL as a computational resource for non-standard Greek and provides a foundation for sociolinguistic NLP, bias analysis, and the study of informal language in LLMs. 

---
# Progressive Cramming: Reliable Token Compression and What It Reveals 

**Authors**: Dmitrii Tarasov, Timofei Lashukov, Elizaveta Goncharova, Andrey Kuznetsov  

**Link**: [PDF](https://arxiv.org/pdf/2607.21231)  

**Abstract**: Token cramming compresses sequences into learned embeddings with near-perfect reconstruction, but fixed token budgets and 99\% accuracy thresholds leave it unclear whether residual errors reflect optimization failures or fundamental limits. We introduce progressive cramming, which grows the target prefix token-by-token, stopping only when reconstruction is no longer achievable within a fixed optimization budget. Progressive trajectories occupy low-dimensional structure in embedding space. Prepending a crammed embedding causes a moderate but consistent accuracy drop on multiple-choice benchmarks even with the original prefix in context, and collapses capability almost entirely under generative evaluation. Causal attention-knockout interventions trace this degradation to the embedding's interactions in the model's early layers. These results position progressive cramming as a tool for studying compression limits and show that perfect reconstruction - achievable through brittle steering rather than transferable semantics - is insufficient for meaningful compression. 

---
# One More Turn, Less Regret: A Regret-Based Multi-Turn Benchmark for LLMs' Clarification Policies 

**Authors**: Minh Ngoc Ta, My Anh Tran Nguyen, Duong D. Nguyen, Yuxia Wang, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2607.21143)  

**Abstract**: Ambiguous user requests make clarification a sequential decision problem for conversational LLM assistants: they must decide whether to ask, what to ask, when to stop, and when to answer. We introduce RegretBench, a multi-turn benchmark that evaluates clarification as policy behavior rather than isolated question quality. RegretBench provides a hidden-intent formulation of ambiguity, supports free-form interaction grounded in semantic-state tracking, and introduces a regret-based objective that measures how much value a model loses relative to a reference clarification policy. Experiments on open-domain QA and product recommendation scenarios show that final success alone is insufficient, as models with similar accuracy can differ substantially in efficiency, robustness to user behaviors, and stopping decisions. By jointly measuring intent resolution, interaction cost, ineffective clarification, and regret, RegretBench reveals whether models clarify usefully and efficiently. Our results show that effective clarification requires more than plausible questions: models must ask the right question at the right time and stop once the user's intended meaning is clear. 

---
# PrefReward: Learning User Preference Matrix for Personalized Text Generation 

**Authors**: Yue Wu, Chengbing Wang, Yimeng Bai, Xiaoyan Zhao, Yang Zhang, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2607.21067)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable ability in generating personalized content by leveraging user histories and contextual cues. However, most existing personalization approaches rely on implicit representations within model parameters, making it difficult to interpret user-specific preferences or effectively handle long-context dependencies. To address these challenges, we propose PrefReward, a novel preference-aware generative framework that explicitly models user styles through a structured preference matrix and integrates it into the decoding process as a reward signal. PrefReward consists of two stages: (1) extracting a user-specific preference matrix that summarizes individual stylistic tendencies, and (2) using the matrix to guide generation via a KL-divergence-based reward function. Experiments on the LongLaMP dataset show that PrefReward outperforms non-personalized and retrieval-based baselines in both generation quality and personalization interpretability. 

---
# QuantiBias: Benchmarking Quantization-Induced Bias in LLMs 

**Authors**: Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2607.21063)  

**Abstract**: Almost every large language model that reaches a broad audience is quantized: trained in full precision, then compressed for efficiency. This step is assumed harmless and its safety is rarely re-checked. We find its principal side effect is increased bias that standard safety evaluation misses. Holding the model, its training, and the prompts fixed, a quantized model still refuses harmful requests, still avoids over-refusing benign prompts, and still selects the unbiased multiple-choice answer. Yet asked an open-ended question, the same model volunteers stereotypes in all eight languages we probe, in roughly one in four open-ended answers under an independent judge (~24% to ~27% across the compression ladder): it passes every standard check and still reaches users measurably more biased. The selective gap is a robust finding; whether open-ended bias further increases with compression is less certain, sensitive to the judge that scores it. We address both with \textbf{QuantiBias}, a benchmark that pairs a generative, multilingual stereotype probe with the refusal and multiple-choice controls that isolate open-ended generation, contrasts each build with and without reasoning, and rates the content severity of what it generates. Across two backbone models (Qwen and Gemma), a five-family screen, and eight benchmarks, quantizers allocate their extra precision by capability data that carries no bias-prevention signal, and reasoning before answering roughly halves the effect on some families while doing nothing on others. A quantized build must be re-evaluated for open-ended bias, not only on the short-form safeguards it already passes. 

---
# Sample-Efficient Learning from Agent Experience 

**Authors**: Chenhui Gou, Haoqin Tu, Yunhao Fang, Jianfei Cai, Hamid Rezatofighi  

**Link**: [PDF](https://arxiv.org/pdf/2607.21051)  

**Abstract**: Real-world agent learning is often constrained by costly environment interactions, such as running time-consuming experiments or obtaining human feedback. In-context learning offers a highly sample-efficient way for agents to learn from their own interaction histories, but its gains disappear once that experience is removed from the context. Separately, context distillation provides a mechanism for internalizing contextual information into model weights. However, applying it to agents' interaction histories without sacrificing environment sample efficiency remains underexplored. We term this problem Experience Distillation and develop an implementation that requires no further environment interaction beyond the collected experience. Experiments on 749 curated software-engineering tasks and six text-adventure games show that it retains at least 64.8\% of the gains from in-context learning across both domains, whereas direct supervised fine-tuning on the collected experience recovers only 3.8\%. Compared with classical reinforcement-learning baselines, in-context learning from trial-and-error experience followed by Experience Distillation matches their performance with at least \(9.6\times\) fewer environment samples. 

---
# CultureTalk-ID: A Multi-Task Dialogue Benchmark for Cultural Commonsense in Indonesian Local Languages 

**Authors**: Muhammad Dehan Al Kautsar, Salsabila Pranida, Bilal Elbouardi, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2607.21016)  

**Abstract**: Culture is lived through conversation, yet existing Indonesian cultural commonsense benchmarks evaluate LLMs on short and isolated prompts, stripping away the dialogic context in which cultural nuances actually surface. We introduce CultureTalk-ID, the first dialogue-based benchmark for cultural commonsense in Indonesian and its local languages, comprising 4,496 culturally grounded dialogues across 11 languages and 13 culturally salient topics, curated through a multi-stage human pipeline with native speakers to ensure authenticity. CultureTalk-ID introduces three complementary tasks, namely dialogue-based multiple-choice cultural commonsense reasoning, culturally faithful machine translation, and language steering, which jointly probe whether LLMs can understand, transfer, and generate culturally grounded language. 

---
# Where Animacy Lives in Large Language Models: Tracing the Circuits of the Animacy Concept 

**Authors**: Samuele Punzo, Giovanni Cinà, Sandro Pezzelle  

**Link**: [PDF](https://arxiv.org/pdf/2607.20995)  

**Abstract**: Distinguishing animate from inanimate concepts in written language requires more than shallow text processing, as it involves recognizing complex selectional constraints and contextual cues, such as verb-argument interactions. Yet, current large language models (LLMs) appear to be capable of doing it. We investigate whether this animacy-sensitive behavior of LLMs can be traced to a localized set of causally relevant components and connections. To do so, we construct a controlled dataset of minimal pairs and perform circuit discovery on four open-weight models. Through in-depth experiments and ablations, we show that a causal mechanism responsible for handling animacy in these models does exist, thus discovering an animacy circuit. At the same time, this circuit appears to be less localized compared to other known ones and generalizes only partially across models and animacy tasks, confirming the distributed, context-dependent, and somewhat graded nature of the animacy concept. 

---
# From a Word-Level Dictionary to Sentence-Level Semantics: Multilingual Grievance Labelling with Contextual Models 

**Authors**: Lin Tian, Marian-Andrei Rizoiu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20946)  

**Abstract**: Grievance is one of the warning signs analysts look for when assessing threats of violence. It is increasingly measured at scale from online text, most often with word-level lexicons like the Grievance Dictionary that score by matching weighted terms. Such matching is a fast and transparent proxy, but it cannot resolve whether a term is asserted, quoted, negated, or condemned. These lexicons are also often evaluated on pools enriched with the very examples they retrieve, so a high score partly reflects agreement with the lexicon's own selection rule. Examining a five-language, 2{,}000-item evaluation pool, we find its halves separated almost perfectly by the lexicon itself: every item labeled ``random'' is in fact lexicon-negative, so the lexicon's apparent macro-AUROC of 0.686 collapses to a 0.500 floor fixed by construction. We keep the dictionary's 22-construct ontology but replace term matching with context-reading models, evaluated on a non-circular benchmark that separates unconditional-random, lexicon-positive, and lexicon-negative strata across five languages. Reading the full post rather than the target sentence alone helps most where the lexicon is silent, raising average precision on lexicon-negative text from 0.14 to 0.20, with the largest gains on quoted, implicit, and cross-sentence grievance. Together, these results show that grievance is measured more faithfully by reading the surrounding context, and more honestly when tested on text the lexicon did not select. We release our code and benchmark at this https URL. 

---
# Tencent WorkBuddy Bench: A Multi-Domain Coding-Agent Benchmark with Contamination-Resistant Task Construction 

**Authors**: Tencent WorkBuddy Bench Team, Siqi Cai, Shaopeng Chen, Xiang Fei, Yong Mao, Zihan Xu, Zhiheng Lyu, Zhijian Shao, Yuchen Shi, Shuwen Zhang, Chaofan Qiu, Linjie Che, Xiaoxi Zhao, Feng Wu, Kai Zhang, Chaofan Zhu, Yubin Qi, Xiaoyun Liang, Peijie Dong, Yunhao Zhang, Yuanjie Zhu, Ling Jiang, Xianjun Zhang, Zhehang Chu, Anyuan Sang, Zhen Feng, Sen Nie, Shi Wu, Yuanzhen Xu, Xin Li, Ning Yang, Zhiqiang Dong, Hande Dong, Qiang Lin, Yi Liu, Yunsheng Wu, Ke Li, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.20911)  

**Abstract**: We introduce Tencent WorkBuddy Bench, a multi-domain evaluation suite for coding agents; this report documents its construction methodology, scoring protocol, and a cross-model leaderboard. At its core is a unified evaluation framework for constructing and running distribution-informed coding-agent tasks across four work domains - Code, Web, Office, and Security. Rather than adapting public issue text, every task is reverse-engineered from a real commit, pull request, or business scenario and rewritten as a short, colloquial, role-played request, so that a task's prompt is not recoverable by web-searching the underlying issue, pull request, or commit thread. Because the dataset is released openly - task directories, environment images, evaluation harness, tests, and reference solutions - contamination resistance rests on this construction together with dataset versioning rather than on secrecy. The four subsets - repository-level engineering, front-end development, office and business workflows, and red-/blue-team security - probe complementary facets of real work, each with its own verification style. All are packaged in a uniform task-directory format and run, under a uniform and reproducible protocol, on two agent harnesses (CodeBuddy Code and Claude Code); the full open release makes the benchmark reproducible end to end and directly auditable, since any third party can re-run each task and inspect its content. Because each subset uses a different scoring instrument, scores are not comparable across subsets and the suite reports no suite-wide average. We report a cross-model leaderboard across several model families. 

---
# LegalCiteTrust: Benchmarking Citation Trustworthiness in Chinese Long-Form Legal Research Reports 

**Authors**: Yunhan Li, Mingjie Xie, Zeyang Shi, Gengshen Wu, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20872)  

**Abstract**: Long-form legal research reports increasingly rely on LLMs and agentic research systems, but their reliability depends not only on answering the task, but also on whether cited legal authorities are trustworthy. A citation can be risky even when it points to a real source: the report may omit limiting conditions, misdescribe the authority, or use it to support a stronger claim than the source allows. We introduce LegalCiteTrust, a benchmark for evaluating citation trustworthiness in Chinese long-form legal research reports. It contains 72 densely annotated report-level tasks and evaluates reports along three dimensions: Coverage, Support, and Citation Trustworthiness. Citation Trustworthiness is operationalized through citation-level Existence, Fidelity, and Applicability (E/F/A). Experiments on general-purpose LLMs, deep-research systems, and legal-specific systems show that task completion, evidence richness, citation density, and citation reliability expose different system behaviors. Retrieval tools can improve evidence support without reliably improving the Trust score, while E/F/A-based revision improves Trust and Final score more clearly than existence-only filtering. These results suggest that trustworthy legal research generation requires citation-aware evidence governance after retrieval: systems must not only retrieve legal authorities, but also select, describe, and apply them reliably. 

---
# CSPF: A Constrained Shared-Private Fusion Method for Non-Verifiable Preference Evaluation 

**Authors**: Hehao Zhang, Danli Wang, Xinyuan Wang, Xuange Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20862)  

**Abstract**: At present, reliable evaluation of non-verifiable tasks remains challenging. Existing approaches often fail to adequately capture the diverse evaluative criteria underlying human preferences in such tasks. To this end, we propose Constrained Shared-Private Fusion (CSPF), a fusion method that treats heterogeneous frozen reward models as complementary evaluators and learns to integrate their hidden-state representations under pairwise human-preference supervision. CSPF decomposes each expert signal into shared and expert-private representations, encouraging cross-expert alignment while preserving complementary viewpoints. Across experiments on LM-Arena target-domain adaptation and PPE out-of-distribution preference evaluation, CSPF achieves the best performance on the primary metrics among the evaluated single-expert reward-model, scalar-score multi-expert, and rubric-judge baselines. Overall, CSPF suggests that fusing hidden-state representations provides a more expressive basis for preference assessment, offering a practical route toward integrated evaluative signals for non-verifiable preference tasks. 

---
# REFACT: Adaptive Fact Restatement for Compact and Faithful Chain-of-Thought Reasoning 

**Authors**: Zhensheng Jin, Xin Dai, Zhenghao Liu, Chaojun Xiao, Huiyuan Xie, Yu Gu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.20833)  

**Abstract**: Large language models increasingly rely on long-form reasoning for complex tasks, yet their reasoning traces may drift away from the supplied context when evidence is sparse, noisy, or in conflict with parametric knowledge. Existing grounding methods either attach citations after generation or encourage evidence retrieval inside the trace, but they often do not ensure that cited content is sufficient for the local inference and final answer. We propose REFACT, an adaptive fact-restatement citation framework that trains models to decide when a reasoning step needs contextual grounding and at what granularity source facts should be restated. This design avoids both unsupported inference and indiscriminate fact copying by turning citations into answer-supporting intermediate states. REFACT is optimized with a two-stage SFT-to-RL pipeline in which a citation-utility reward encourages cited facts to be well-formed, source-traceable, and answer-sufficient. Experiments on LongBench, LV-Eval, and ConFiQA show that REFACT improves long-context QA and counterfactual faithfulness while substantially reducing token consumption. Further analysis shows that REFACT preserves more answer-bearing evidence with fewer restated facts, yielding reasoning traces that are denser rather than longer. All code and data are available at this https URL. 

---
# The Geometry of Personality: Activation Steering with Jungian Cognitive Functions 

**Authors**: Liu Zai, Yumeng Wang, Junchen Fu, Joemon M. Jose  

**Link**: [PDF](https://arxiv.org/pdf/2607.20803)  

**Abstract**: Activation steering enables control and interpretation of LLMs, yet existing work primarily models personality through static trait frameworks such as the Big Five. We investigate whether personality can instead be represented and controlled as a set of cognitive processes using the eight Jungian Cognitive Functions. To this end, we introduce a framework comprising a Jungian evaluation protocol and a dataset of over 2,100 role-playing character narrations.
Activation steering vector extraction and evaluation experiments on Llama-3.1-8B demonstrate effective monotonic control over all eight cognitive functions through activation steering. Beyond controllability, our analysis reveals that: 1. personality information is concentrated in middle transformer layers; 2. steering vectors exhibit structured geometric relationships consistent with distinctions between rational and irrational functions; 3. effective multi-dimensional steering directions cannot be recovered as linear combinations of single-function directions. These findings provide new insights into the representation of personality in LLM activation space and establish a framework for studying interpretable, effective, and multi-dimensional personality control. 

---
# Are Diversity Metrics Measuring Diversity? A Capability-Controlled Audit of Majority-Vote Gain in LLM Ensembles 

**Authors**: Donghwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.20768)  

**Abstract**: Majority voting over LLMs is widely assumed to benefit from diversity, and diversity measures are used to choose which models to combine. We ask whether five such measures track diversity or mainly re-express capability, auditing them as predictors of majority-vote gain over the best member across 31,900 subsets of 30 LLMs on MMLU-Pro (29 on TruthfulQA) under explicit capability controls. Three findings emerge. First, latent complementarity is ubiquitous: oracle gain is positive in 100% of subsets, yet simple voting beats the strongest member in only 9.98% of all canonical size-3 subsets (18.71% with held-out best selection); the pooled size-2-4 rate is 1.27%, partly reflecting deterministic even-size voting behavior. Second, a joint-correctness proxy (strict diversity) is nearly collinear with one minus mean accuracy (size-3 Spearman rho = +0.991 / +0.988); raw diversity-gain associations are strongly capability-entangled and, with one exception, unstable under control. Third, three linear contingency-table statistics are algebraically non-separable; after capability control, the empirically stable remainder is a modest residual pairwise co-failure association in which more shared error corresponds to lower gain. This direction is robust, but its magnitude is configuration-dependent. Joint rawspace linear regressions treating strict diversity, disagreement, and double-fault as independent predictors are rank-deficient by construction. 

---
# Rushes: A Human Preference Dataset for Pluralistic Alignment 

**Authors**: Michael Xu, Jorge Leandro, Sudha Rao, Weijia Xu, Nebojsa Jojic, Gabriel DesGarennes, Chris Quirk, Bill Dolan  

**Link**: [PDF](https://arxiv.org/pdf/2607.20767)  

**Abstract**: We introduce Rushes, a dataset and benchmark for studying revealed human engagement preferences in interactive narrative environments. Rushes is collected through a game interface where users interact with AI-generated branching narratives and select one choice from a small, explicit candidate set at each decision point. Each interaction logs the full candidate set, the user's choice, and the evolving narrative context, yielding time-ordered trajectories with persistent user-level identifiers. Rushes contains 44,226 decision events from 8,167 unique users across six games, capturing sequential, personalized engagement behavior rather than static judgments. We show that user choices exhibit structured, non-random patterns, quantified by a low choice entropy relative to a uniform baseline. We position Rushes as a diagnostic benchmark for pluralistic alignment and demonstrate a robust Engagement Gap: state-of-the-art LLMs, including GPT-5, fail to outperform simple baselines. While classical Matrix Factorization (SVD) captures measurable personalized signal (37.7%), frontier LLMs (34.23%) struggle to even match the Popularity Baseline (36.4%) on event-level choice prediction. This gap suggests that single, population-level objectives, like those used in modern RLHF, appear insufficient to capture heterogeneous, context-dependent engagement signals. As a result, even highly capable models default to majority preferences rather than adapting to individual trajectories. We release Rushes to support research into pluralistic alignment and sequential decision-making in generative systems. The full code for the platform and dataset will be available here: this https URL 

---
# REGARD: Regional Affective Differences in Large Language Models 

**Authors**: Andrei Chetvergov, Alexander Evseev, Mikhail Solovev, Timofei Sivoraksha, Stepan Ukolov, Valeriia Kuschenko, Maria Chistyakova, Sergey Bolovtsov  

**Link**: [PDF](https://arxiv.org/pdf/2607.20722)  

**Abstract**: Large language models trained and aligned within different linguistic and regional ecosystems may frame the same political, cultural, and geopolitical entities in different ways. Such differences are often evaluated through sentiment, favorability, or stance, reducing model attitudes to a single positive-negative axis. We introduce REGARD, a study of what drives affective framing differences across LLMs on post-Soviet entities using target-directed Valence-Arousal-Dominance profiling. We query 19 models on 500 region-specific targets, score their responses with two independent LLM judges, GPT-4o-mini and Qwen3.6-35B-A3B, and validate the measurements on a 300-item human-annotated subset. Post-hoc Ward-linkage clustering of all 19 models by affective and response-behavior profiles yields three behavioral clusters that cut across model origin, family, and parameter count. Generic-answer rate is strongly associated with lower arousal (r = -0.81) and with cluster placement: models that deflect evaluative prompts with templated responses cluster together at low arousal regardless of origin. These findings show that VAD profiling captures emotional intensity, a dimension of affective framing that is largely invisible to conventional sentiment-based evaluation. 

---
# Learning to Detect UI Principle Violations via Reinforcement Learning 

**Authors**: Nishi Mehta, Swathi Alse, Himani Kumavat, Yue Yu, Pratik Jayarao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20690)  

**Abstract**: Small language models and coding agents increasingly generate web front-end code, yet their outputs are typically evaluated primarily for functional correctness. A generated interface may compile, render, and pass unit tests while still violating established interface quality principles, including accessibility barriers, deceptive design patterns, poor visual hierarchy, and excessive decision complexity. Existing auditing approaches face a trade-off between cost, coverage, and scalability: expert human review provides rich judgment but is slow and expensive; frontier vision-language models offer broader reasoning capabilities but remain costly to deploy at scale; and rule-based tools such as axe-core and Lighthouse are inexpensive but primarily capture mechanically checkable accessibility issues.
We investigate whether a lightweight vision-language model can serve as an effective critic for generated interfaces. We unify 19 interface-quality principles from three complementary sources of HCI knowledge: WCAG 2.2 accessibility standards, deceptive design taxonomies, and established theories of perception, cognition, and interaction. To train this critic, we construct a verified dataset of approximately 10,000 generated web pages by synthetically injecting known violations into clean, LLM-generated Tailwind pages.
Continued reinforcement learning on a 4B vision-language model improves micro-F1 from 36\% to 84\%, with 13 of 19 principles exceeding 80\% F1. The resulting critic can audit generated interfaces, filter low-quality interface training data, and provide a reward signal for design-aware code generation. We release our data-generation recipe and injection/verification prompts to support reproducible evaluation and future work on scalable interface-quality assessment. 

---
# From Agent Failures to Text Policies: What Works and What Breaks 

**Authors**: Jaideep Ray, Ankit Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2607.20668)  

**Abstract**: TextGrad improves language-model systems by revising text from feedback. Its core thesis is that natural-language feedback can act as a gradient for optimizing text components without changing model weights. Applying it to agents is harder because feedback arrives only after a sequence of actions, making it difficult to identify which decision caused failure. We study this problem by separating the ability to follow a useful policy from the ability to learn that policy from experience. Our main finding is a clear gap between these two abilities. Human-written policies improve two frozen 7B agents on TextWorldExpress by 5.0 success points, showing that useful policy text exists. However, policies generated from agent trajectories do not reliably outperform fixed prompting, even with richer traces, counterfactual evidence, or iterative GEPA search. The main challenge for agent-level TextGrad is therefore not executing textual policy updates, but reliably generating and selecting them from experience. 

---
# Frontier Financial Judgement: Can agents tell what might move a stock? 

**Authors**: Joshua Harris  

**Link**: [PDF](https://arxiv.org/pdf/2607.20645)  

**Abstract**: We introduce Frontier Financial Judgement, a challenging new benchmark developed in collaboration with professional equity analysts to assess agents' ability to replicate expert human judgements. Rapidly identifying new information, evaluating its implications and determining its valuation impact is one of the most time-consuming and challenging aspects of real-world equity coverage. This is becoming ever more difficult and important as AI rapidly increases the quantity of new information to process. The strongest agent we evaluate on Frontier Financial Judgement matches all expert labels in only 52.4% of cases. We also find significant divergence in estimated false-positive rates among frontier agents, ranging from ~1% for GPT-5.6 Sol to ~32% for Claude Sonnet 4.6. To construct the benchmark and make it representative of real-world settings, we combine human-designed and labelled synthetic articles with live news articles and historical documents, creating 656 items for assessment. The resulting task requires agents to distinguish genuinely new, valuation-relevant financial information from stale, immaterial or misleading news under realistic conditions. We find substantial trade-offs among agent accuracy, cost, false positives and reliability that continue to hinder the reliable deployment of news-flow filtering in practice. 

---
# Evaluating the Effectiveness of Persona Simulation in Opinion Prediction with GPT-4.1 

**Authors**: Sarah Y. Li, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20589)  

**Abstract**: Persona simulation involves utilizing large language models (LLMs) to anticipate human choices or interactions based on specific characteristic information. To further understand current limitations and future directions, we tested persona simulation in opinion prediction with GPT-4.1 (knowledge cutoff by June 2024). Using personas from nine U.S. states provided by Columbia University's Personas dataset, GPT-4.1 accurately predicted 2024 election outcomes in eight out of the nine states, only failing in one of the swing states. We then focused on opinions related to medicine and healthcare. With the American Trends Panel Wave 123 dataset from Pew Research Center, GPT-4.1 was able to anticipate beliefs about childhood vaccines with an accuracy of up to 0.94. Furthermore, we applied GPT-4.1 to generate conversations among personas and observed that the simulated dialogues and opinions adhered well to personas' personalities and backgrounds, albeit lacking natural human-like flow. Persona simulation proves to be a promising application of artificial intelligence as long as biases are addressed. In the near future, it will be beneficial to apply it to opinion analysis and reaction prediction in diverse fields ranging from public health to lawmaking to economics. 

---
# Can Valence Reflect Morality in Natural Language? A Preliminary Annotation Study 

**Authors**: Jonny O'Dwyer, Malika Bendechache, Louise McCormack, Elif Calik, Ramin Ranjbarzadeh, Dost Muhammad, Shokofeh Anari Bozcheloei, Ishita Singh  

**Link**: [PDF](https://arxiv.org/pdf/2607.20461)  

**Abstract**: Present implementations of artificial intelligence (AI) ethics do not adequately take feelings, or affect, into account. If AI should be aligned with human ethics, it seems reasonable to thoroughly investigate the possibility of AI behaviour that mirrors virtuous human ethical conduct, where feelings play a role in the actions, judgements or statements one makes. Furthermore, while prominent theories of normative ethics are often discussed in terms of their differences and shortcomings, Virtue, Consequentialist, and Kantian Deontological ethics all share a common feature of considering human feeling to some degree while the popular descriptive ethics theory, Moral Foundations Theory, positions feelings as central to many of its foundations. Therefore, in the present paper, a data set of moral valence is proposed, consisting of 500 annotations by six human participants for both action/judgement and consequence moral valence, ranging from -1 to 1 for text-presented scenarios from the Commonsense Norm Bank data set. The resulting valence features share significant relationships with multi-class (immoral/discretionary/moral) and binary immoral/moral categories while additionally providing a noteworthy test set Matthew's correlation coefficient of 0.764 using regularised logistic regression for binary classification. This provides early evidence of the usefulness of valence features for morality estimation of text, indicating that valenced consequences of responses for others can be considered toward more human morally-aligned AI. In the interest of promoting further affective-moral computing research, this study's annotations will be made available for research on request. 

---
# Instruct-FD: Can Your Full-Duplex Speech System Follow Turn-Taking Instructions? 

**Authors**: Yuzhi Tang, Wentao Ma, Xiling Zhao, Ahmad Salimi, Sepehr Harfi Moridani, Dongming Shen, Jixuan Wang, Abdulrahman Abdulrazzag, Murdock Aubry, Yu-Hua Chen, Daniel Lee, Jaewon Lee, Jonah Mackey, Silin Meng, Nicholas Stranges, Chenxu Xiong, Hao Yu, Yi Zhu, Mu Li, Alex Smola  

**Link**: [PDF](https://arxiv.org/pdf/2607.20460)  

**Abstract**: Current full-duplex (FD) spoken dialogue systems can produce fluid interactions, yet it remains unclear whether they can adapt their turn-taking behavior when explicitly instructed. This is critical for real-world deployment, where conversational policies vary across applications (e.g., proactive tutoring vs. passive counseling). We introduce Instruct-FD, an instruction-conditioned benchmark for evaluating controllable turn management in FD systems. To enable this, we develop a human-validated, scalable synthetic pipeline that generates instruction-conditioned conversations, along with a deployment-agnostic multi-turn evaluation protocol and an LLM-based judge. Benchmarking six state-of-the-art full-duplex systems reveals a substantial gap in instruction-following turn management: the best model achieves only 64.4% adherence. Performance is highly uneven across behaviors and scenarios, with proactive behaviors such as model backchanneling and interruption remaining particularly challenging. These findings establish instruction-following turn management as a crucial direction for building adaptable and deployable full-duplex dialogue systems. 

---
# THOR: A Theta-Gamma Hierarchical Oscillatory Reasoning Framework for Multi-hop QA 

**Authors**: Ziyang Ling, Ronald X. Xu, Mingzhai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.20459)  

**Abstract**: Multi-hop question answering requires retrieving and integrating evidence from multiple contexts. Despite the rapid progress of current research, multi-hop reasoning remains constrained by two persistent limitations: attention decay, where the model's focus on main question degrades as the reasoning chain grows, and error accumulation, where mistakes propagate across hops and compounds into final failure. Inspired by Theta-Gamma hierarchical oscillation which decouples global planning from local retrieval, enabling efficient attention transfer between hops and a verification and repair mechanism that interrupts the accumulation of errors in the wrong paths, we present THOR, a brain-inspired Theta-Gamma hierarchical oscillatory reasoning framework. Extensive comparative experiments and specific validation experiments on multi-hop QA benchmarks demonstrate that THOR improves answer accuracy and robustness while mitigating limitations, showcasing its generalization across different backbones. 

---
# CAMeR: Keyword-Gated Hybrid Activation for Adaptive Memory Retention in LLM Agents 

**Authors**: Haowen Lai  

**Link**: [PDF](https://arxiv.org/pdf/2607.20458)  

**Abstract**: Large language model (LLM) agents operating over extended dialogues accumulate vast amounts of information, yet existing memory systems either retain everything indiscriminately or apply uniform forgetting heuristics that fail to distinguish relevant from irrelevant knowledge. We present CAMeR (Context-Activated Memory Reinforcement), a memory retention framework combining keyword-gated hybrid activation -- a joint symbolic (word-level Jaccard) and sub-symbolic (embedding cosine) gating mechanism -- with adaptive weight dynamics. CAMeR computes a hybrid similarity score for each memory-query pair; memories exceeding a threshold receive reinforcement while all memories undergo controlled decay. We introduce CAMeR-Bench, a 76-memory, 100-round benchmark spanning 8 topic clusters with graded activation frequency, designed to test adaptive retention where existing benchmarks (LoCoMO, LongMemEval) cannot. On CAMeR-Bench, CAMeR's keyword gate achieves a 1.6$\times$ larger retention gap between high-frequency and never-referenced memories compared to embedding-only gating (scissors gap: 0.039 vs. 0.024), while time-driven baselines (Oblivion, SuperLocalMemory) collapse to near-zero weights over 100 rounds. CAMeR's top-5 retrieval saves 83.2\% tokens versus full-context approaches (39k vs. 231k cumulative) while producing weight signals that improve retrieval precision. Through 8 ablation conditions we establish that the keyword gate -- not learnable decay -- is the primary performance driver at this scale. Our findings demonstrate that hybrid symbolic-neural gating provides a simple yet effective mechanism for adaptive memory retention in LLM agents. 

---
# Dropping the Anchor: Statistical Context Summarization for Distributed Systems via Pulsar Attention 

**Authors**: Aryan Sood, Shantanu Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2607.20457)  

**Abstract**: Inference with large language models (LLMs) on long sequences is computationally expensive due to the quadratic complexity of self-attention. Distributed blockwise methods such as Star Attention reduce this cost by sharding context across hosts, but rely on prepending a static, content-blind copy of the first block to every host. We propose Pulsar Attention, which replaces the static anchor with two lightweight, content-aware components: a small attention-sink prefix that stabilizes softmax, and compact cross-block summaries built via a Max-IDF heuristic that selects chunks containing globally rare tokens. This reduces the Phase 1 per-GPU FLOPs by up to 3.3$\times$ over Star Attention while retaining an identical KV cache footprint. On RULER and BABILong with Llama-3.1-8B, Pulsar Attention outperforms both Star Attention and dense attention at sequence lengths up to 128K tokens, with absolute gains of up to 4.7% over the dense baseline. 

---
# Learn2Zinc: Fine-tuning Small Language Models for Text-to-Model Translation in MiniZinc 

**Authors**: Serdar Kadioglu, Karthik Uppuluri  

**Link**: [PDF](https://arxiv.org/pdf/2607.20456)  

**Abstract**: Large language models excel at code generation for mainstream programming languages but struggle with rare, domain-specific languages such as MiniZinc, a constraint modeling language for combinatorial problems. We investigate whether targeted fine-tuning can teach small language models (0.6B to 20B parameters) to generate syntactically correct and semantically valid MiniZinc models from natural language problem descriptions. Our key finding is that syntax errors dominate failures when working with this domain specific language: the out-of-the-box execution accuracy of small language models such as Qwen3, LLaMa, Gemma, and GPT-OSS is near-zero. We propose a cross-model error bootstrapping approach that collects syntax errors from multiple LLM runs and leverage those to curate an error correction training dataset. This dataset allows us fine-tune small language models that consistently improves both direct code generation and chain-of-thought approaches across all model sizes. With self-reflection and ensembling, our approach achieves up to 98\% execution accuracy. In parallel, solution accuracy still remains at 35\%, indicating that while syntax is learnable, constraint reasoning remains a challenge. We contribute our fine-tuning pipeline, datasets, and models to opens-source for further research on text-to-model translation. 

---
# RE-AD: Real-Time Requirement Adherence for Data Labeling 

**Authors**: Siddarth Malreddy, Ishan Nigam, Akshay Arora, Nikhil Mittal, Subrat Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20455)  

**Abstract**: Human-annotated data remains fundamental to training frontier Large Language Models (LLMs). However, crowd-sourced annotations often suffer from quality issues stemming from annotator misunderstanding or lack of engagement. To address this, we introduce a real-time requirement adherence (RE-AD) framework that leverages LLMs to proactively validate labeling quality. Our methodology involves decomposing Standard Operating Procedures (SOPs) into atomic rules via self-reflection, categorizing them by complexity, and applying tiered validation strategies. Evaluated on a synthetic benchmark, the system achieved an F1 score of 0.749. Furthermore, production deployment resulted in annotators accepting and fixing 82% of the errors flagged by the framework. We include ablation studies to demonstrate the impact of our core design decisions. 

---
# Response drift across frontier large language models 

**Authors**: Mohammed Aledhari, Ali Aledhari, Fatimah Aledhari, Gowtham Venkat Eathamokkala, Mohamed Rahouti  

**Link**: [PDF](https://arxiv.org/pdf/2607.20454)  

**Abstract**: All frontier large language models (LLMs) exhibit response drift -- producing outputs that deviate from expert-validated references -- yet the magnitude and structure of this drift remain uncharacterised by systematic human evaluation. Here we report a fully crossed evaluation in which 47 geographically diverse participants each assessed all 62 multidomain questions across ten frontier LLMs under blinded conditions, yielding 29,140 independent assessments. Every model drifts, but drift magnitude varies substantially: eight models converge on a statistically indistinguishable ceiling (78-81% deviation), while two achieve lower deviation (47-49%). Drift profiles differ across six domains and 62 questions, with pairwise correlations among ceiling models exceeding r = 0.85. Automated similarity metrics explain less than 2% of variance in human judgements. These findings reveal that response drift is universal across frontier LLMs, domain- and question-dependent in structure, and accessible only through human-centred evaluation. 

---
# A Knowledge-Injection Framework for Zero-Shot Adaptation of LLMs to Delirium Prediction 

**Authors**: Jessica Sena, Shesadree Priyadarshani, Miguel Contreras, Bharat Gandhi, Scott Siegel, Subhash Nerella, Parisa Rashidi  

**Link**: [PDF](https://arxiv.org/pdf/2607.20453)  

**Abstract**: Large language models show promise for clinical prediction, but zero-shot performance on specialized tasks is limited by incomplete domain knowledge, especially for smaller locally deployable models. We present a lightweight knowledge-injection framework for zero-shot ICU delirium prediction that augments a deterministic natural-language summary of structured electronic health record data with an external clinical knowledge report at inference time, without fine-tuning or retrieval. We evaluate LLaMA 3.1 8B and LLaMA 3.3 70B on 3,160 ICU admissions from the MIMIC IV dataset. Adding a clinically meaningful external knowledge report improves AUROC by 8.57 percentage points for the 8B model and 1.99 percentage points for the 70B model compared to no external knowledge. Relative to a GPT-5.2 frontier-model reference without external knowledge report (AUROC 68.86%), knowledge injection reduces the performance gap from 15.66 to 7.09 AUROC points for LLaMA 8B and from 5.30 to 3.31 AUROC points for LLaMA 70B. Random control reports do not improve performance and often degrade it, indicating that gains depend on clinically meaningful content rather than added prompt length alone. SHAP-based attribution further confirms that the injected knowledge is actively used during prediction. These findings suggest that inference-time knowledge injection can narrow the gap between locally deployable open-weight models and frontier closed models while preserving a practical, privacy-preserving workflow for resource-constrained clinical settings. 

---
# Semantic Field Theory: Historical Origin, Higher-Order Interaction, and Stabilized Semantic Inference 

**Authors**: Dimitris Vartziotis  

**Link**: [PDF](https://arxiv.org/pdf/2607.20451)  

**Abstract**: Semantic Field Theory (SFT) has developed from a philosophical critique of strong anti-formalist readings of language games into a proposed computational model class for lexical semantics, higher order composition, and stabilized interpretation. This paper reconstructs that evolution and gives SFT a sharper mathematical core suitable for independent evaluation in computational linguistics and representation learning. The central proposal is that a tractable level of linguistic organization can be modeled through lexical representations expressed as semantic fields, through contextual deformation of those fields, through interaction terms defined over subsets of tokens, and through stabilization governed by semantic energy dynamics. The paper contributes five formal elements. First, it defines a semantic field model as a tuple consisting of a semantic space, a lexical field lifting, a contextual deformation map, an interaction complex, and an interpretation functional. Second, it proves a Gaussian product closure result showing that multiplicative field interactions have explicit centers, precisions, and compatibility factors. Third, it generalizes the three-word problem by using Mobius inversion on the subset lattice to isolate irreducible semantic interactions of arbitrary order. Fourth, it introduces an order spectrum that measures how much field mass is explained at each interaction order. Fifth, it formulates stabilized interpretation as minimization of an energy functional associated with the sentence and gives existence, descent, and stability conditions. A small worked example shows how a three-word summer day triple can be represented by Gaussian semantic fields, implemented in Python, and summarized by a flow diagram. The result is not a completed theory of natural language meaning and does not replace social, pragmatic, or normative accounts of language. 

---
# ShriNep@EEUCA 2026: RAKSHAK - Multi-Task DeBERTa with Rationale Distillation and Jigsaw-Augmented Training for Toxic Intent Classification 

**Authors**: Binayak Karki, Aryan Kafle, Pingala Ghimire  

**Link**: [PDF](https://arxiv.org/pdf/2607.20450)  

**Abstract**: This paper presents two systems for the GameTox Shared Task at the Workshop on EEUCA at ACL 2026, which requires classifying World of Tanks chat utterances into six fine-grained toxic intent categories (Labels 0-5). Severe class imbalance, domain-specific multilingual slang, and extremely scarce data for rare categories such as Threats (Label 4, 60 samples) and Extremism (Label 5, 24 samples) make this a challenging classification problem. Our primary submission, RAKSHAK (rak s. aka, Sanskrit for "Protector"), is a multi-task DeBERTa-v3-base (He et al., 2022) framework combining rationale distillation from Qwen2.5-14B (An et al., 2024), Supervised Contrastive Loss, and dedicated rare-class binary heads. RAKSHAK's training data is augmented with cross-domain transfer from the Jigsaw Toxic Comment dataset (16,225 samples mapped to Labels 1-4) and 100 LLM-generated extremism samples for Label 5. Our secondary system (M1) fine-tunes DeBERTa-v3-base with Focal Loss on the original GameTox data plus the same 100 extremism samples, without Jigsaw transfer. RAKSHAK achieves a Macro F1 of 0.5883 on the official test set, ranking 7th out of 35 participating teams, while M1 achieves 0.5252 Macro F1. An ablation comparing M1 with and without Jigsaw data shows that cross-domain transfer accounts for +2.6 F1 points, while RAKSHAK's multi-task architecture contributes a further +3.7 points. 

---
# The Storyteller in the Model: Narrative Pattern Inheritance, Escalation Dynamics, and Alignment Governance in LLMs 

**Authors**: Adam Rigby, Raz Saremi, Azadeh Sohrabinejad, Mehdi Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2607.20449)  

**Abstract**: LLMs are trained predominantly on human-authored text, yet the structural and narrative conventions embedded in that text are rarely examined as a source of systematic behavioral influence, or as a governance risk in deployed systems. This paper considers whether the storytelling patterns inherent in published human writing, including archetypal roles such as protagonist, antagonist, and underdog, as well as tension-and-resolution narrative arcs, are absorbed during training and subsequently surface in LLM outputs, causing responses to drift toward unexpected, adversarial, or rhetorically enticing behaviors over extended interactions. Through a systematic literature review and cross-paper analysis of recent empirical studies on LLM alignment, persona dynamics, emergent misalignment, and user interaction patterns, we observe evidence bearing on this hypothesis. The findings reveal three key patterns. First, LLMs reproduce statistical patterns from their training data rather than reasoning independently. Second, measurable latent traits, including sycophancy and deceptiveness, emerge reliably across unrelated prompts. Third, fine-tuning on a narrow narrative task can produce unintended behavioral changes well beyond that task. Furthermore, evidence suggests that persuasive, narrative-style outputs are among the most common LLM products in real-world usage, amplifying these risks. Narrative drift constitutes an unmonitored escalation pathway in deployed AI systems, one that evades discrete-incident detection mechanisms and requires dedicated monitoring instruments. 

---
# Domyn-Small: A European 10B Reasoning Language Model 

**Authors**: Simone Angarano, Francesco Bertolotti, Federico D'Ambrosio, Michele Resta, Alessandro Rognoni, Nicolò Ruggeri, Dario Salvati, Andrea Valenti, Alberto Veneri, Martin Cimmino  

**Link**: [PDF](https://arxiv.org/pdf/2607.20448)  

**Abstract**: We introduce Domyn-Small, a 10-billion-parameter open-weight reasoning language model released under the MIT license. Domyn-Small is the product of an initial pre-training phase on 9 trillion tokens multilingual data, followed by a post-training pipeline for reasoning, instruction following, and context extension. For the latter, we performed a Continued Pre-Training (CPT) phase that doubles the native context window to 32K tokens, followed by SFT with a math-focused annealing run. Finally, the RL phase includes GRPO with verifiable rewards, DPO, and a multi-environment GRPO stage spanning five task domains: mathematics, code, multiple-choice QA, instruction-following, and tool calling. The 32K-token native context extends to 128K at inference via YaRN, and a chat-template toggle enables dual-mode reasoning. Against peer models in the 7--10B class (Qwen3.5-9B, OLMo-3-7B-Think, Nemotron-Nano-8B, Ministral-3-8B), Domyn-Small achieves a strong accuracy-efficiency balance: it produces roughly one-third as many tokens as Qwen3.5-9B and approximately 35% of OLMo-3-7B-Think's token budget on core reasoning benchmarks, while delivering strong instruction-following (IFEval 79.9) and competitive science reasoning (GPQA-Diamond 50.0). We release the weights and the post-training recipe alongside Domyn Swarm (Apache~2.0), an open-source framework for scalable LLM inference on HPC clusters developed during this program and used throughout this work. 

---
# thaulab@EEUCA 2026: Who Said What to Whom? A Targeting-Aware Neural-Symbolic Pipeline for Gaming Toxicity Detection 

**Authors**: Anmol Guragain, Marcos Estecha-Garitagoitia, Luis Fernando D'Haro Enríquez, Ricardo de Córdoba  

**Link**: [PDF](https://arxiv.org/pdf/2607.20447)  

**Abstract**: This paper describes our system for the EEUCA 2026 Shared Task on toxicity classification in gaming chat. We implement a three-stage pipeline combining an ensemble of two compact transformers (DeBERTa-v3-base, 184M; XLM-RoBERTa-base, 278M) with a Linguistically-Informed Mediator (LIM) that resolves inter-model disagreements through corpus-backed lexical normalization, class-conditional unigram scoring, multilingual profanity detection, and agentive targeting analysis grounded in speech act theory. The LIM specifically targets the minority classes (Hate \& Harassment, Threats, and Extremism), which are the most safety-critical categories in real-world gaming moderation. To address the extreme class imbalance (1{,}450:1 Non-toxic to Extremism ratio), we introduce a two-stage data augmentation strategy using only the provided training data. Our system achieves a Macro F1 of 0.6441 and accuracy of 0.9062 on the official test set, ranking 3rd in Macro F1 and 1st in accuracy among all teams. The proposed pipeline is domain-portable: adapting to other gaming platforms requires substituting only the game-specific entity lexicon. Code is publicly available at this https URL\_EEUCA. 

---
# Distinguishing Artificial from Authentic: Evaluating LLMs for Detecting LLM-Generated Content 

**Authors**: Juho Leinonen, Paul Denny  

**Link**: [PDF](https://arxiv.org/pdf/2607.20446)  

**Abstract**: As large language models (LLMs) are increasingly used by students to generate natural language responses and program code, there is growing interest in whether LLMs themselves can be used to distinguish AI-generated work from human-authored submissions. In this paper, we investigate the extent to which LLMs can detect their own generated content across multiple educational task types, including programming exercises, reflective writing, and short-answer questions. Using authentic student responses and multiple variants of LLM-generated answers, we evaluate detection performance under different prompting strategies and output formats. Our study addresses three research questions: (1) how accurately LLMs can identify their own outputs across task domains, (2) how detection effectiveness is influenced by factors such as prompt design, response length, and task type, and (3) what characteristics of LLM-generated responses contribute to successful or failed detection. Our findings show that LLM-based detection is highly task-dependent: detection is substantially more reliable for programming tasks and longer reflective responses, but performs poorly for short-answer questions, where LLMs frequently judge their own outputs as more human-like than authentic student responses. We further find that prompt framing and response verbosity have a pronounced effect on detectability in reflective writing tasks, with relatively minor prompt variations significantly reducing detection accuracy, while programming-related detection is more robust to prompt changes. Together, these results highlight both the potential and the limitations of LLM self-detection in educational settings and suggest caution in relying on LLMs as standalone tools for identifying AI-generated student work. 

---
# SCoPE: Shift-Aware Speaker-Conditioned Priors for Emotion Recognition in Conversations 

**Authors**: Burak Can Kaplan, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2607.20445)  

**Abstract**: In conversations, human emotions are transient; however, they tend to persist across multiple utterances. For example, we rarely switch instantly between contrasting emotions such as happiness and anger. Instead, emotions tend to evolve smoothly, and these patterns are often speaker-specific. Some people might escalate, while others gradually cool down over time. Furthermore, when emotions change during a conversation, they are often driven by contextual factors, such as newly received information or unexpected events. Even though progress has been made in Emotion Recognition in Conversations (ERC), most existing approaches still rely heavily on overt evidence and do not sufficiently model these non-apparent factors. Especially in multimodal settings, this makes these models fragile when the signals are noisy (e.g., occluded faces, slang expressions, or microphone noise). To address these limitations, we introduce Speaker-Conditioned Priors over Emotions (SCoPE). SCoPE is a light weight module that utilizes the emotional history of each speaker and explicitly models their priors for use in subsequent emotion classification. Second, we incorporate emotion shift prediction, a well-established concept in ERC, to guide the model in balancing the priors from SCoPE and multimodal evidence. Finally, we propose a shift-aware fusion mechanism that performs precision-weighted logit integration between multimodal evidence and the speaker prior, forming a Bayesian-inspired product-of-experts formulation. This dynamic fusion allows the model to rely on historical priors when emotions persist and to prioritize multimodal evidence when shifts are likely. Experimental results show our model achieves superior performance over recent state-of-the-art models on the IEMOCAP dataset in multimodal settings. 

---
# Confidently Deceptive: How Confidence Amplifies the Risk of LLM Deception 

**Authors**: Ali Asad, Stephen Obadinma, Anshul Pattoo, Wenxuan Zhang, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20444)  

**Abstract**: Large language models (LLMs) can produce deceptive responses: outputs that mislead users in service of a contextually or experimentally induced goal. Yet it remains unclear how confidently models deceive and whether higher confidence makes deceptive responses more persuasive to end users. In this paper, we study these basic questions in various models and different deception datasets. We provide a comprehensive study measuring confidence through both verbalized self-reports and a range of logit-based estimators. We show that LLMs deliver deceptive responses with substantial verbalized confidence and that human annotators prefer the higher-confidence deceptive response 78% of the time in paired comparisons. Misalignment fine-tuning amplifies the problem. Confidence in deceptive responses rises across all three benchmarks, increasing the resulting potential risk, with effects generalizing beyond the training distribution. Strikingly, models classify their own deceptive outputs as deceptive at high rates (82.7% under misalignment) while still predicting they would produce them - recognition without avoidance. We argue that confident deception is a distinct alignment risk requiring evaluations that jointly measure deception, confidence, and awareness. 

---
# GLAN-QnA-KR: A Seedless Taxonomy-Driven Korean Instruction Corpus 

**Authors**: Daekeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.20443)  

**Abstract**: We release GLAN-QnA-KR, a 303,581-row openly redistributable Korean instruction-QA corpus produced via the seedless taxonomy-driven GLAN synthesis pipeline with Microsoft's Phi-3.5-MoE-instruct as the producer model (generation: 2024-12; release: 2024-12; licence: OpenRAIL). The corpus spans a flat taxonomy of 1,084 English-labelled disciplines paired with Korean question/answer text, a 100-900 difficulty scale, and a median of 313 question characters and 1,098 answer characters per record. Two properties are atypical for synthetic instruction data at this scale: (i) exact duplicate questions number only 1 in 303,581 rows and character-trigram near-duplicate clusters at Jaccard >= 0.9 number zero in a 5,000-sample probe, and (ii) a two-layer contamination audit against KMMLU, KoBEST (five sub-tasks), and HAE-RAE-Bench shows a maximum test-vs-corpus question-level character-trigram Jaccard of 0.163 with zero test items at Jaccard >= 0.7, and a maximum multilingual-E5 cosine of 0.901 with a single test item at cosine >= 0.90 and zero at >= 0.95, across 20,000 sampled GLAN questions and seven evaluation sets. At the time of release, this is, to our knowledge, the largest single-pipeline synthetic Korean instruction corpus verifiable on the Hugging Face Hub and the only Korean >=100k-row corpus built under a seedless taxonomy-driven protocol. This note documents the generation protocol, corpus statistics, the contamination audit, and the licensing boundary in a form suitable for downstream citation. 

---
# Naver-News-KO: A Korean News Summarization Dataset for Open-Source Fine-Tuning of Summarization Models 

**Authors**: Daekeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.20442)  

**Abstract**: We release Naver-News-KO, a Korean news summarization dataset of 27,400 (document, summary) pairs collected from Naver News over a ten-day window in July 2022 across two categories (Economy and IT/Science; 77/23 split), with train/validation/test partitions of 22,194 / 2,466 / 2,740 and a mean per-record document-to-summary character-compression ratio of 6.03x. The dataset has been publicly hosted on the Hugging Face Hub since January 2023 and, as of May 2026, receives approximately 33,000 downloads per month; community-maintained Korean summarization models fine-tuned on it include Gemma-2B-ko and Gemma2-9B variants. This technical report (i) documents the collection protocol, the column schema, and the split construction, (ii) reports corpus-level statistics (length distributions, compression ratio, and a measured 16.8% near-duplicate title-Jaccard overlap between test and train that users should be aware of), (iii) positions the resource against other open Korean summarization corpora, (iv) provides a Lead-3 extractive reference point (ROUGE-1 55.1, ROUGE-L 50.6) and two reproducible fine-tuned baselines -- KoBART (R-1 56.6, BERTScore-F1 81.5) and Gemma-2B-ko with LoRA (R-1 55.3, BERTScore-F1 78.3) -- with release-time training scripts, and (v) clarifies the licensing and intended-use scope of the resource. The goal is to provide a citable reference for downstream work that already uses this dataset, not to propose a new benchmark. 

---
# Belief Propagation in LLM World Models: Measuring Strategic Information Bias with Prediction Markets 

**Authors**: Mykola Khandoga, Yevhen Kostiuk, Anton Polishko, Yurii Filipchuk, Kostiantyn Kozlov, Dmytro Zamriy, Artur Kiulian  

**Link**: [PDF](https://arxiv.org/pdf/2607.20441)  

**Abstract**: Every information ecosystem produces beliefs that shape strategic decisions. Both human analysts and AI systems inherit the blind spots of their information sources. We show that LLMs, combined with prediction markets, function as a calibrated instrument for measuring how far ecosystem-induced beliefs deviate from an external reference: LLMs extract the beliefs a text corpus implies, and prediction market price trajectories, anchored at resolution by realised outcomes, provide the calibration reference against which to quantify the deviation.
We isolate the bias contribution of specific text through ablation: varying information context while holding the model fixed, with a contaminated model that knows actual outcomes as control. Applied to 111 Ukraine-related prediction markets, comprising approximately 93,000 predictions across four models, we find that English news context systematically biases territorial predictions, wrong 64 to 72 percent of the time when it pushes predictions toward territorial capture. A contaminated model that knows actual outcomes shows the same error rate, indicating that the bias originates primarily in the text. Supplementing with Ukrainian military-analytical sources reduces the bias for all clean models, while absolute-error gains are partial and model-dependent.
We show that the distortion originates primarily in the sources, not the models. Consistent across four architectures, it will persist in any system that processes them and propagate into downstream decisions. 

---
# Answer-then-Edit: Reasoning Skeleton Editing for Anti-Distillation with Preserved Utility 

**Authors**: Fan Li, Mengting Pan, Sijia Xu, Xiaoyang Wang, Chen Chen, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20440)  

**Abstract**: Proprietary large language models (LLMs) entail substantial intellectual and financial investment, making them valuable intellectual property (IP). However, even when deployed via black-box APIs, these models remain vulnerable to unauthorized knowledge distillation, which allows adversaries to cheaply extract and replicate model capabilities. To address this issue, anti-distillation (AD) has been proposed to generate defensive outputs that hinder distillation effectiveness, overcoming the limitation of watermarking-based approaches that rely on post-hoc verification. However, existing AD methods based on internal model perturbations struggle to balance anti-distillability and utility (e.g., answer accuracy and naturalness) of reasoning traces, with stronger defenses often causing significant utility loss. To fill this gap, we propose \textbf{\underline{S}}keleton-\textbf{\underline{G}}uided \textbf{\underline{R}}easoning \textbf{\underline{E}}diting (SGRE), an \textit{Answer-then-Edit} framework that performs post-hoc trace modification for anti-distillation. In the answer stage, the teacher model first generates clean reasoning traces, preserving the original reasoning accuracy while enabling more flexible control over trace naturalness. In the editing stage, we draw inspiration from Cognitive Load Theory (CLT) and introduce a three-stage strategy consisting of reasoning skeleton extraction, skeleton graph coarsening, and skeleton verbalization. These operations jointly perturb reasoning structures and augment textual complexity to amplify extraneous load on student models, hindering their acquisition of underlying reasoning patterns. Extensive experiments across diverse LLMs demonstrate that SGRE achieves state-of-the-art performance in reducing distillation effectiveness, while maintaining lossless reasoning accuracy and superior trace naturalness. 

---
# AsymVerify at SemEval-2026 Task 6: Asymmetric Confidence-Gated Verification for Political Evasion Detection 

**Authors**: Sebastien Kawada  

**Link**: [PDF](https://arxiv.org/pdf/2607.20439)  

**Abstract**: Political evasion is difficult to detect because evasive answers often appear cooperative while avoiding concrete commitment. We present AsymVerify, a confidence-gated verification system for SemEval-2026 Task 6, a three-way classification of Clear Reply, Ambivalent, and Clear Non-Reply responses. AsymVerify scored 0.85 Macro F1 on the evaluation split (D_eval, n=237), placing 2nd out of 41 teams on the official leaderboard. The system first classifies each question-answer pair, then selectively applies downgrade verification (CR/CNR -> AMB) or upgrade verification (AMB -> CR) to low-confidence predictions. Development analysis shows that errors concentrate at the Ambivalent boundary in both directions, motivating this asymmetric two-verifier design while confidence gating keeps additional inference cost low. On D_dev (n=308), AsymVerify with GLM-4.7 gains +17.1 Macro F1 over single-pass classification at 1.48 calls/example, and the upgrade verifier alone improves every tested LLM backend on D_dev by +6.8 to +15.2 Macro F1 over its single-pass baseline. Code is available at this https URL. 

---
# Preference Tuning as Spectral Update Reorganization 

**Authors**: Peiyan Zhang, Haibo Jin, Liying Kang, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20438)  

**Abstract**: Preference-based post-training is usually understood through endpoint behavior, yet the learned update that produces this behavior remains largely opaque. We study RLHF and related preference optimization through the spectral structure of their induced parameter updates. By decomposing effective LoRA updates and reloading their spectral components as plug-in modules, we turn preference-induced updates into objects that can be isolated, recomposed, and directly intervened on. Across model families, optimization algorithms, and supervision regimes, these updates consistently develop a spectral head--tail organization. A compact head emerges early and carries the dominant endpoint shift, while a heterogeneous residual tail remains. The split is functional rather than merely descriptive. Plug-in intervention shows that the head accounts for the visible behavioral departure from the base model, while the tail is weak in isolation. Cross-run recomposition further shows that mixed adapters follow the source of the head, indicating that the head carries run-level solver bias. This endpoint dominance does not imply learning sufficiency. Head-only learning is non-vacuous but fails to recover the full solution, especially on out-of-distribution behavior. Tail-only learning yields little visible gain, yet the full solution is not recovered without the tail. These findings recast preference post-training as structured update reorganization rather than a monolithic behavioral correction, and suggest that alignment gain and coverage loss are tied to how the learned update itself is organized. 

---
# TopoGuard: Graph Theory Based Defenses Against Split-Knowledge Attacks on RAG 

**Authors**: Chahana Dahal, Zuobin Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2607.20437)  

**Abstract**: Production Retrieval Augmented Generation (RAG) systems rely on aggregating multiple external documents to answer complex queries. However, the retrieved documents introduce a new threat surface that can be exploited to launch split-knowledge attacks. In this attack, the adversary injects documents that are individually benign but create false associations when combined and fed to language models. This paper shows that the new attack is structurally invisible to existing per-document filters, like LlamaGuard. To address this issue in RAG, this work introduces TopoGuard, a family of graph theory-based methods specifically targeting the split-knowledge attacks by building a semantic similarity graph from retrieved documents and detecting contexts with malicious topology. Grounded on the theoretical analysis, the TopoGuard family has been proven to be effective and robust even with noisy inputs. Extensive experiments are conducted on two retrieval datasets and compared with multiple baseline methods. Specifically, the TopoGuard-$\lambda_2$+Entity catches 21$\times$ more attacks than LlamaGuard-2-8B at 1\% FPR (32.6\% vs 1.5\% recall) on the HotpotQA dataset. Compared with production RAG detection systems using large language models, the proposed TopoGuard variants run efficiently at sub-millisecond latency and stay robust under adaptive adversaries and benign cross-domain queries. 

---
# Routing Subspaces: Auditing Evaluation-to-Deployment Mismatch in Fine-Tuned Language Models 

**Authors**: Phongsakon Mark Konrad, Toygar Tanyel, Serkan Ayvaz  

**Link**: [PDF](https://arxiv.org/pdf/2607.20436)  

**Abstract**: Safety evaluations often assume that behavior observed during testing reflects behavior in ordinary use, but fine-tuning can break this assumption. A checkpoint can appear fixed under evaluation-style prompts while the same behavior persists under ordinary-use prompts. Output scores reveal this mismatch but do not locate it. We investigate whether the distinction is encoded in a stable internal site and introduce an approach that fits a paired activation contrast at a path-patching-informed mid-depth window, then modifies the resulting coordinate on held-out prompts. The intervention closes the evaluation-to-deployment gap in ten of twelve model--behavior settings (six of the eight settings with $n{\geq}120$ paired questions) across four full-matrix instruction-tuned model instances; a fifth model supports localization and edit-provenance checks, and deployment-framed rates change by at most $6.1$pp. The two flat cells, both sycophancy, indicate that a single-coordinate audit is not sufficient when the installed distinction is higher-rank or missed by the depth heuristic. The audit is a diagnostic for fine-tuned checkpoints, not a training-time defense or a guarantee of deployment safety. 

---
# Making Open-Source Text LLM Watermarks Durable Against Merging 

**Authors**: Luisa Scharff, Thibaud Gloaguen, Robin Staab, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2607.20435)  

**Abstract**: Open-source LLMs (OSMs)arereaching near state-of-the-art performance, prompting prior works to trace the text they generate by embedding text watermarking algorithms directly into their weights. Yet, OSMs are subject to post-training modifications, which has been shown to remove the watermark. Model merging in particular, a prominent method used for combining expert knowledge and preventing catastrophic forgetting, strongly removes such OSM watermarks. A key question is how to enable OSM watermarks that survive subsequent merging. In this work, we show for the first time how to design an OSM watermark that is durable against model merging. We propose Merge-Adversarial Training, an adversarial training algorithm to distill text watermarks into model weights while being robust to subsequent model merging. Our approach consistently outperforms all baselines (e.g. with SLERP up to +51 percentage points (pp) TPR@1%FPR with +25 pp on average) while preserving downstream capabilities. We also for the first time evaluate OSM watermarks against realistic merge scenarios, representing common use-cases such as combining expert capabilities or preventing catastrophic forgetting, and with 3 prominent merging algorithms. More broadly, our findings suggest that adversarial training is a reliable approach for increasing OSM watermark durability against post-training modifications. 

---
# Break Through the Compression Bottleneck: From Theory to Practice 

**Authors**: Xiusheng Huang, Lu Wang, Yequan Wang, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.20434)  

**Abstract**: As the parameter size of language models continues to grow, effective model compression is required to reduce their computational and memory overhead. Existing compression methods suffer from bottleneck issues: when the compression ratio is increased, performance degrades significantly. Low-rank decomposition and quantization are two prominent compression methods that have been proven to significantly reduce the computational and memory requirements of Large Language Models (LLMs) while maintaining model accuracy. Evidently, combining these two methods will break through the existing compression bottleneck. However, how these two methods interact when combined remains a critical question for developers, as many assume they are orthogonal, meaning their combination would not introduce additional errors beyond those independently introduced by each method. This paper provides the first mathematical proof that low-rank decomposition and quantization are non-orthogonal. We validate these findings through a series of experiments on large language models. Our results demonstrate that these methods are non-orthogonal, and their combination leads to significant performance degradation. Importantly, we propose a novel approach Diagonal Adhesive Method (DAM), which can effectively combine the two methods and mitigate the performance loss. Our research provides deep insights into model compression and lays a solid theoretical and experimental foundation for future related studies. 

---
# Moir: Let the Model Direct Its Own Story for Robust Cross-Domain Knowledge Editing 

**Authors**: Jea Kwon, Jiwon Kim, Dong-kyum Kim, Meeyoung Cha  

**Link**: [PDF](https://arxiv.org/pdf/2607.20433)  

**Abstract**: While language models remain frozen at their training state, the world evolves continuously. Knowledge editing has emerged as a key alternative to full retraining, but its deployment is bottlenecked by the erosion of core capabilities: mathematical and programmatic reasoning collapse while encyclopedic recall remains intact. We trace this asymmetric degradation to a distributional mismatch. Covariance-based editors preserve only the subspaces spanned by their reference corpus, but fail to capture the operative distribution shaped by post-training such as SFT and DPO. Static external corpora, including Wikipedia and even the original pretraining mixture, cannot recover this shifted manifold. We propose Moir, which estimates the preservation covariance $C$ directly from the model itself by sampling from its own decoding distribution. Seeding generation with a single random vocabulary token bypasses the instruction-following templates that otherwise dominate sampled outputs, exposing the broader subspaces the model has internalized. Moir requires no external data and serves as a drop-in component for any covariance-based editor, a practical advantage given that the pre- and post-training corpora of most modern LLMs are not publicly accessible. Across OLMo-2, Llama-3.1, and Qwen-3 (7-8B), under both MEMIT and AlphaEdit and in batch and sequential regimes, Moir consistently extends preservation in the most vulnerable domains, most strikingly on Qwen3-8B after 20,000 AlphaEdit batch edits, it retains 79.9% GSM8K accuracy compared to 10.9% with the Wikipedia baseline. These results suggest that aligning the preservation distribution with the model's operative distribution is a key factor in non-destructive editing, and that the model itself may be the most accessible source of that distribution for deployed systems. 

---
# Position: Natural Language Should Not Fully Replace Formal Languages 

**Authors**: Eitan Wagner, Elisha Rosensweig, Omri Abend  

**Link**: [PDF](https://arxiv.org/pdf/2607.20432)  

**Abstract**: Recent advances in large language models and their widespread adoption have prompted claims that natural language could entirely replace formal languages, such as programming languages for software design. In this position paper, we argue that this perspective overlooks fundamental linguistic properties of natural language, specifically that it is optimized for underspecification in open-ended contexts. We introduce a formal framework centered on *task specificity*, defining it as the information-theoretic reduction of uncertainty in an output space -- such as all possible images -- given a user's specific requirements. We prove a *specificity crossover theorem*, showing the existence of a threshold beyond which the cost to express formal requirements into natural language exceeds the cost of direct formal specification. By analyzing case studies across modalities, such as image generation, code synthesis, and audio production, we demonstrate that natural language excels at low specificity tasks, while formal languages are advantageous on tasks with stricter requirements. We conclude that natural and formal languages are complementary tools and advocate the development of hybrid systems that allow users to move across the specificity spectrum. 

---
# Skill-Contracted Agents for Evidence-Aware Materials Literature Analysis 

**Authors**: Bixuan Li, Yu Liu, Shuo Shi, Xiaoya Huang, Peng Kang, Lei Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.20431)  

**Abstract**: Materials science literature analysis requires simultaneous attention to composition, processing, characterization, and property relationships, yet conventional retrieval-augmented generation pipelines struggle to reconcile heterogeneous tasks within a single retrieve-then-generate architecture. Here we present AlphaAgent, a skill-driven agent framework that decouples retrieval-based question answering from paper-level report generation through explicit skill contracts. A dedicated retrieval skill rewrites user requests into material-specific search intents, queries a curated index of more than 300,000 papers from the Journal Citation Reports Metallurgy and Metallurgical Engineering category, and reformulates queries when initial evidence is insufficient. A separate report-generation skill parses full-text PDFs to produce structured per-paper analytical reports and cross-paper summaries. In a blind evaluation on 40 materials-science questions, half of which required deep analytical reasoning, AlphaAgent substantially outperformed a baseline system matched for underlying model, document index, and retrieval scale, with the largest gains in mechanistic explanation and awareness of credibility boundaries. These results indicate that explicit task separation, refined retrieval intent, and evidence-aware generation improve large-language-model-based literature analysis for materials research. 

---
# LLM-INSTRUCT at UZH Shared Task 2026: Constraint-Aware Retrieval and Selective Debate for Paragraph-Level Argument Mining 

**Authors**: Phuong Huu Vu Tran, Long Minh Vo, Son Nguyen Minh Le, Hoang Van  

**Link**: [PDF](https://arxiv.org/pdf/2607.20430)  

**Abstract**: We present LLM-INSTRUCT, the winning system for the UZH Shared Task at ArgMining 2026 on paragraph-level argument mining in UN and UNESCO resolutions. The task requires paragraph-type classification, prediction of a subset of 141 official tags, and directed relation prediction under a strict JSON schema setting using only open-weight models up to 8B parameters. We frame the task as constrained structured prediction. The system first narrows the candidate tag space with metadata-aware dense retrieval, then applies constrained decoding with per-dimension caps, escalates only uncertain cases to a three-agent debate branch, and finally validates the output schema. On the official leaderboard, LLM-INSTRUCT ranked 1st overall, with 1st in F1 and 5th in LLM-as-a-Judge. During development, our configuration search further improved Task 1b Micro-F1 from 35.83% to 40.08% while keeping the internal Task 2 score at 4.421. The main lesson is simple: reducing the decision space before generation improves both accuracy and submission robustness. Our code and supporting scripts are publicly available at: this https URL 

---
# More Is Not More: What Matters for Diversity in LLM Opinions? 

**Authors**: Qiyang Yao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20429)  

**Abstract**: Large language models are increasingly used to simulate diverse human opinions in open-ended tasks such as synthetic surveys, focus group modeling, and public opinion prediction. However, LLM outputs exhibit systematic opinion homogenization. Practitioners have explored various interventions to increase diversity, but the landscape remains fragmented: different methods are evaluated in isolation with incomparable metrics, and in practice they are typically deployed and upgraded simultaneously, making it difficult to attribute gains to specific components. To advance a more scientific understanding of LLM output diversity, we design a factorial experiment that separates two primary intervention dimensions: input conditioning (operationalized through persona depth) and interaction architecture. We evaluate all conditions on 100 real-user open-ended questions across 7 models, measuring diversity with multiple complementary metrics. Our findings challenge several common assumptions. First, more persona detail does not monotonically increase diversity. The initial step of persona conditioning already captures the majority of the gain, while further elaboration with demographic detail does not consistently improve and can reduce diversity on some models. Second, rather than seeking a single best interaction architecture, we find that different architectures explore largely non-overlapping opinion regions. Combining multiple architectures yields broader coverage than optimizing any one. Third, commonly attempted low-cost alternatives such as raising sampling temperature and adding diversity instructions produce negligible effects compared to structured interventions. Overall, our work demonstrates that diversity is not a product of scaling along any single dimension, but is highly sensitive to the structural form and combination of interventions. 

---
# Human-in-the-Loop Large Language Model Framework for Identification of Cutaneous Immune-Related Adverse Events 

**Authors**: Charles Lu, Olivia Burke, Debby Cheng, Adam Kashlan, Caitlyn Duffy, Zeyun Lu, Lirit Fuksman, Jin Ning Tian, Andrew Sedlack, Priya Katyal, Eudora Lee, Ralina Karagenova, Chuck Lin, Kun-Hsing Yu, Nicole LeBoeuf, Alexander Gusev, Yevgeniy R. Semenov  

**Link**: [PDF](https://arxiv.org/pdf/2607.20428)  

**Abstract**: This study evaluated a retrieval-augmented, multi-agent large language model (LLM)-driven, human-in-the-loop framework for detecting cutaneous immune-related adverse events (cirAEs) from clinical notes. Compared with unassisted manual review, the LLM-assisted workflow improved accuracy (F1 = 0.88 vs 0.77), inter-rater agreement measured by Cohen's kappa (kappa = 0.82 vs 0.50), and reduced average review time by approximately half. This framework pilots how LLMs can be applied to identify immune-related toxicities across organ systems and, more broadly, enable accurate, scalable, and transparent adverse event data extraction. 

---
# Is MoE Routing a Huffman Code? Discovering the Frequency-Diversity Law in Chain-of-Thought 

**Authors**: Ching-Chieh Tsao, Zhuoyi Lin, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20427)  

**Abstract**: Mixture-of-Experts architectures have revolutionized scaling, yet the underlying logic of their routing remains a black box. In this paper, we uncover a fundamental governing principle: MoE routing is not merely selection, but a manifestation of Huffman Coding. We introduce the Frequency-Diversity Law, revealing that state-of-the-art models, such as Phi-3.5-MoE and Gemma-4-27B-A4B, spontaneously act as information-theoretic engines. These models allocate sparse expert resources for common tokens while invoking high-diversity expert committees for rare, complex tasks found in chain-of-thought trajectories. However, we identify a critical redundancy trap in Qwen3.5-35B-A3B: when effective sparsity (k/E_eff) is sufficiently low, load-balancing inadvertently imposes functional redundancy, masking the underlying Huffman efficiency signal. To bridge this gap, we propose Subset Difference Pruning, a surgical strategy to eliminate functional duplicates. We demonstrate that pruning does not degrade reasoning; instead, it unleashes the model's latent Huffman efficiency, forcing the logic to collapse into streamlined, high-density paths. Our findings suggest that the next generation of MoEs should move beyond forced load-balancing toward Minimum Description Length (MDL) optimality, assigning shorter expert-routing codes to high-frequency information and longer, more diverse codes to low-frequency information, thereby transforming routing from a heuristic into a principled compression engine. 

---
# Knowledge Injection Exists in MoE? Exploring Expert-Aware Contrast Decoding in MoE for Mitigating LLMs'Hallucinations 

**Authors**: Xinyue Fang, Zhiliang Tian, Zhen Huang, Ziyi Pan, Zhihua Wen, Xi Wang, Quntian Fang, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.20426)  

**Abstract**: Existing LLM hallucination mitigation methods, including prompt engineering and model optimization, either hardly alter models'internal knowledge or have poor cross-domain generalization. Contrastive decoding mitigates hallucinations by using layer-wise differences in LLMs. However, prior studies only explore transformer-based models (e.g., GPT), ignoring other effective frameworks like mixture-of-experts (MoE) models. Since MoE alters the traditional transformer architecture, we conduct empirical studies to investigate whether similar layer-wise differences exist in MoEs. Our results show that they do not exist in MoE with shared experts; nevertheless, across different MoEs, higher layers exhibit distinct expert activation patterns between factual and non-factual outputs. Building on these, we propose EAACD, an expert-aware adaptive contrast decoding that uses expert differences in MoE's higher layers to mitigate hallucinations on QA tasks. EAACD splits high-layer experts into a higher-reliability group and several lower-reliability groups based on their confidence and consistency. It contrasts the higher-reliability group's prediction with each lower-reliability group's prediction to calibrate the model's original predictions. To strengthen this contrast, EAACD amplifies hallucinations from lower-reliability experts via attention and masking to provide stronger negative references. EAACD outperforms all baselines on four datasets. 

---
# What is Good? Extracting and Testing Implicit Theories of Literary Quality from LLM Reasoning Traces 

**Authors**: Birger Moëll  

**Link**: [PDF](https://arxiv.org/pdf/2607.20425)  

**Abstract**: What makes writing "good" remains a persistent question in literary studies and computational linguistics. We present a two-study investigation of how reasoning-enabled LLMs evaluate literary quality.
In Study 1, we construct a benchmark of 30 real texts spanning six quality tiers, from canonical literature to anonymous forum posts, and extract the model's implicit theory of quality from its reasoning traces. Across five DeepSeek replications, the model achieves 79.3% mean tier-classification accuracy. The traces reveal a consistent stated theory: the model values intentionality over correctness, prioritizing craft, depth, and distinctive voice. A familiarity experiment with style-matched but unrecognizable passages suggests that source recognition may inflate scores, although this is confounded by genuine quality differences between canonical originals and researcher-written pastiches.
In Study 2, we probe this theory through systematic degradation of five canonical prose passages. We apply six manipulations - vocabulary simplification, rhythm flattening, imagery removal, voice genericization, structure simplification, and combined degradation - and reevaluate each version. Vocabulary simplification causes the smallest quality loss (0.41 +/- 0.46 points), far below structure (2.78) or voice (2.34) loss. Combined degradation is devastating (-5.64) but subadditive. An exploratory comparison with Qwen QwQ shows the same broad qualitative pattern.
Together, these studies suggest that LLM judgments of writing quality are holistic, author-specific, and more sensitive to structural than lexical features, with implications for automated writing feedback and computational aesthetics. 

---
# OpenForgeRL: Train Harness-native Agents in Any Environment 

**Authors**: Xiao Yu, Baolin Peng, Ruize Xu, Hao Zou, Qianhui Wu, Hao Cheng, Wenlin Yao, Nikhil Singh, Zhou Yu, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2607.21557)  

**Abstract**: Modern AI agents rely on elaborate inference harnesses such as Claude Code, Codex, and OpenClaw to drive multi-turn reasoning, tool use, and access to external systems. While powerful, these complex harnesses also make agents hard to train end-to-end with open infrastructure, whose SFT/RL stacks cannot natively express stateful, multi-process harness inference. To address this, we present OpenForgeRL, an open-source framework for training harness-based agents end-to-end in diverse environments. OpenForgeRL achieves this with a lightweight proxy that serves the harness's model calls while recording them as training data for a standard RL codebase (e.g., veRL), and a Kubernetes orchestrator that runs each rollout in its own remote container, together enabling training on any harness in any environment at scale. By decoupling training and inference, OpenForgeRL allows researchers to easily train, study, and improve agents directly in the real harnesses and environments they are deployed with. We validate our framework across diverse, complex harnesses and environments, spanning tool/claw-based agents and multimodal GUI browser- and computer-use agents. Using only hundreds to a few thousand tasks, OpenForgeClaw reaches 31.7 pass^3 and 55.9 pass@3 on ClawEval and 33.7 on QwenClawBench. OpenForgeGUI reaches 37.7 on OSWorld-Verified, 63.0 on Online-Mind2Web, and 72.3 on WebVoyager. Both outperform open baselines of similar size on nearly all benchmarks, and in the GUI setting match or surpass models several times larger. Beyond benchmarks, we analyze how harness choice (e.g., ZeroClaw, OpenClaw, Codex) and RL shape agent behavior. We find that some harnesses are substantially harder to learn than others, and that RL improves agentic reliability, such as self-verification, tool coverage, and completing multi-step plans, though critical abilities such as error recovery remain weak. 

---
# The Boundaries of Automation: A Theory of Persistent Human Participation 

**Authors**: Fares Fourati, Hinrich Schütze, Eyke Hüllermeier, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2607.21547)  

**Abstract**: The rapid progress of AI has intensified the long-standing pursuit of automation: replacing human participation with algorithms wherever possible. Implicit in this pursuit is the assumption that humans remain in the loop only because current AI systems are not yet sufficiently capable. This paper challenges that assumption. Rather than asking how far automation can extend, we ask where its conceptual limits lie and argue that human participation may persist even with highly capable AI systems for three distinct reasons. Technical or complementarity grounds arise when humans contribute capabilities or perspectives unavailable to AI. Normative or developmental grounds arise when participation itself is valuable for human agency or learning. Most importantly, emergence grounds arise from target emergence: in some activities, the target is not fully specified in advance but instead emerges through the interaction itself. In these cases, human participation is not merely a means of improving execution but is constitutive of the target being produced. Human--AI co-construction, understood as the joint production of outcomes by humans and AI systems, is therefore not simply a temporary response to imperfect AI, but a persistent feature of activities whose objectives emerge through participation. This perspective has important implications for the limits of automation and for the design, evaluation, and ethics of future AI systems. 

---
# Windowed-MTP: Removing the Full-Context Draft-KV Tax at Million-Token Context 

**Authors**: Alagappan Valliappan  

**Link**: [PDF](https://arxiv.org/pdf/2607.21535)  

**Abstract**: Speculative decoding accelerates autoregressive generation by having a cheap draft propose tokens that a target verifies in parallel. Frontier models increasingly ship a built-in Multi-Token-Prediction (MTP/NEXTN) draft head under the assumption that the draft is negligibly cheap. At million-token context this breaks: an MTP draft head typically runs full attention over the entire KV cache at every draft step, so its read grows linearly with context and comes to dominate the draft cost -- precisely where speculation is most valuable. The effect compounds with draft length (a deep native draft can turn net-negative, slower than no speculation) and sharpens under hybrid/linear-attention targets, where cheaper verification leaves the draft's full-attention read exposed. We apply a StreamingLLM-style sliding window plus attention sink to the draft's attention only (Windowed-MTP), leaving full-attention verification intact. It is training-free, drop-in, and lossless by construction: the full-attention target still decides every accepted token, so windowing changes only which tokens are proposed, never which are accepted. It bounds the draft's KV working set to a constant, dropping ~99% of KV entries at 1M. Across three architecture families (Qwen GDN-MoE 35B/122B and a Mamba2-hybrid NoPE 120B) at 1M context on a single GPU in SGLang, windowing cuts the per-decode-step cost over the shipping native MTP draft by +28% to +44%, an input-invariant margin that widens with context. Since per-token latency is this cost divided by acceptance length, at matched acceptance end-to-end decode latency improves by the same amount, and more where windowing also lifts acceptance, while preserving the target's verified output distribution. Finally, the unread draft KV -- 7.7-11% of total KV at 1M -- is reclaimed via a compact ring buffer at no acceptance or quality cost. 

---
# GS-Agent: Creating 4D Physical Worlds With Generative Simulation 

**Authors**: Hongxin Zhang, Chunru Lin, Junyan Li, Zhou Xian, Tsun-Hsuan Wang, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2607.21522)  

**Abstract**: Creating dynamic and physically realistic 4D worlds from natural language descriptions is both fascinating and challenging. Traditional computer graphics methods rely on manual creation, requiring extensive human effort to fine-tune materials, motions, and visual fidelity. Recent advances in generative foundation models have sparked interest in learning to generate such 4D worlds from large-scale data; however, existing methods still struggle to ensure physical plausibility and controllability. In this work, we take a different path by leveraging foundation models to construct an agentic system that emulates how humans traditionally create 4D worlds, yet automates the entire process. We present GS-Agent, an end-to-end multi-agent framework that integrates physics engines in the loop to generate realistic, dynamic, and controllable 4D physical worlds from natural language. Inspired by how humans build 4D worlds, GS-Agent decomposes the task into entity management, covering 3D asset curation, material tuning, placement, and motion control, and rendering configuration, including camera and lighting manipulation. Multiple agents with distinct expertise interact with the physics engine via code, seek multimodal feedback, and collaborate to iteratively construct 4D worlds that align with the given descriptions. Experimental results show that GS-Agent effectively converts natural language into diverse and physically plausible 4D worlds exhibiting rich interactions among liquids, deformable objects, and rigid bodies, while achieving cinematic camera and lighting control. We envision GS-Agent as a foundation for a new paradigm in 4D world generation, empowering creative content creation and physical AI. Project page at this https URL 

---
# Agentic coding without the cloud: evaluating open-weight large language models on longitudinal data preparation tasks 

**Authors**: Mack Nixon, Liam Wright, Yevgeniya Kovalchuk, Alison Fang-Wei Wu, Martin Danka, Andy Boyd, David Bann  

**Link**: [PDF](https://arxiv.org/pdf/2607.21482)  

**Abstract**: Large language models (LLMs) and agents are now widely used tools in code development, with data typically sent to third-party cloud-based models. Their adoption in research using personal data is constrained by governance requirements that typically prohibit data transmission to external services. Locally deployable open-weight models offer an alternative since sensitive data never leave the local environment. We introduce an open-source framework for evaluating the efficacy of AI agents powered by open-weight LLMs on one of the most persistent bottlenecks in research on longitudinal population studies: data preparation. The framework comprises: a curated ground-truth dataset (cleaning scripts preparing six sweeps of data from a British cohort study), task definitions encompassing tasks such as category harmonization and multi-wave merging, and automated routines for evaluating the LLM-produced R code and outputted data. We benchmark LLMs across the (consumer grade) deployment spectrum to assess their efficacy in 20 data preparation tasks (creation of 102 variables). Current state-of-the-art, 31-35B parameter models almost saturated our benchmark ("average task completion" up to 87.9%). The performance of open-weight LLMs running on consumer-grade hardware shows promise of a viable path toward AI-assisted data preparation in governance-restricted research settings. Our framework is publicly available at: this https URL. 

---
# Error Certificates for KV-Cache Eviction via Randomized Design 

**Authors**: Peng Xie  

**Link**: [PDF](https://arxiv.org/pdf/2607.21475)  

**Abstract**: Deterministic KV-cache eviction keeps the top-$k$ tokens under an importance score and deletes the rest. We prove that this design cannot know what it destroyed: evicted values can be altered so that everything the serving system retains is unchanged while the true attention-output error grows arbitrarily, so no serving-time estimator of that error is consistent. Randomized eviction restores identifiability. With a Poisson-sampled tail at known inclusion probabilities, one logit offset performs the Hájek correction inside the softmax, and a survey-sampling variance estimator over the retained set becomes a per-step error certificate with 0.97 empirical coverage at no accuracy cost. On real workloads we pre-registered seven claims and lost three: question-aware eviction at 25--50\% budgets is nearly free; output log-probability predicts failure better than the certificate; certificate-gated budget escalation adds nothing. What survives is attribution: the certificate separates cache-induced from inherent failures (AUC 0.73--0.75, against 0.47--0.54 for output confidence) and schedules recomputation better than random or confidence gating. Randomization buys attribution, not prediction. 

---
# Euclid-MCP: A Model Context Protocol Server for Deterministic Logical Reasoning via Prolog 

**Authors**: Bartolomeo Bogliolo  

**Link**: [PDF](https://arxiv.org/pdf/2607.21412)  

**Abstract**: Large Language Models (LLMs) excel at natural language understanding and generation but remain unreliable for multi-step logical reasoning, especially in safety-critical or compliance-sensitive domains. Recent neuro-symbolic approaches address this gap by coupling neural models with external symbolic engines, yet most integrations are bespoke and lack a standardized interface for tool-augmented agents. This paper presents Euclid-MCP, an open-source MCP server that provides deterministic logical reasoning via SWI-Prolog. Euclid-MCP introduces Euclid-IR, an engine-agnostic intermediate representation for Horn-clause logic that is human-readable, easy for LLMs to generate, and straightforward to compile into Prolog or alternative backends. The server exposes a compact tool interface that supports a translate-run-inspect-repair loop, enabling LLM clients to delegate inference while retaining full access to proof traces and derivation logs. We evaluate Euclid-MCP on a realistic IT security and compliance use case. Results show that while LLMs alone are sufficient on small knowledge bases, they hallucinate systematically on larger problems, whereas Euclid-MCP delivers exact answers with lower latency and more compact outputs. We argue that semantic RAG is fundamentally unsuited for rule enforcement, and that Euclid-MCP can serve as a stable, shared reasoning substrate for both RAG-based assistants and agentic systems. 

---
# AI Assistants Overassist 

**Authors**: Verona Teo, Raghav Jain, Tobias Gerstenberg, Max Kleiman-Weiner  

**Link**: [PDF](https://arxiv.org/pdf/2607.21306)  

**Abstract**: Large language models (LLMs) are increasingly used as tutors and thought partners, helping users reason through problems. While guidance from AI assistants can scaffold thinking and foster learning, such benefits depend on how they help--for instance, intervening too early or too frequently may hinder true learning and cognitive engagement. Yet how AI systems navigate intervention decisions during problem-solving remains poorly understood. Here, we introduce Int-Bench, a simulation-based benchmark for evaluating LLM interventions during learning. Int-Bench simulates a "student" solving a problem while a "teacher" monitors the student's reasoning and decides whether, when, and how to intervene. Across three domains--code debugging, mathematics, and brain teasers--we evaluate LLM teachers on the frequency and timing of interventions, as well as their impact on both immediate task success and generalization to new problems. We also compare LLMs to humans, finding that LLMs intervene more frequently and earlier than humans. Moreover, in contrast to humans, they tend to provide complete solutions rather than targeted hints. These findings suggest that current LLM assistants often optimize for short-term success rather than supporting the reasoning processes needed for deeper learning and long-term success. 

---
# Training Large Language Models for Self-Explanation Faithfulness 

**Authors**: Yeoktatt Cheah, María Pérez-Ortiz, Noah Y. Siegel, Oana-Maria Camburu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21090)  

**Abstract**: We propose a Reinforcement Learning (RL) method to directly optimize the faithfulness of self-explanations - the extent to which a model's generated reasoning accurately reflects its internal decision-making process. While existing work focuses on evaluating faithfulness or using inference-time prompting frameworks to improve an LLM's self-explanation's tractability, these approaches do not provide a mechanism to directly optimize a model's parameters to generate faithful self-explanations. We bridge this gap by modifying existing faithfulness metrics into an RL training objective. We investigate (1) if models can be trained to accurately detect factors that affect their decisions, and (2) whether RL can directly optimize for the disclosure of these factors thereby improving LLM self-explanations' faithfulness. We experiment with two intervention types: random-word insertions and user-bias insertions, using a per-sample reward derived from the Phi-CCT correlation metric. RL fine-tuned Llama3.1-8B and Qwen3-8B show substantial improvements on the Phi-CCT faithfulness metric, with in-distribution scores rising from near-zero to as high as 0.664, and out-of-distribution scores reaching up to 0.691 on held-out tasks such as StrategyQA. Cross-intervention generalization is weaker but more interesting: a priori we would not expect a model trained only on random word insertions to generalize to user-bias phrases, yet Llama3.1-8B shows non-zero transfer in this direction. The reverse direction and Qwen3-8B do not replicate this, indicating model-dependent and setup-dependent effects we cannot yet explain. Lastly we analyze model behavior to rule out reward gaming behaviors that often plague RL training. Ultimately, we show that models can be trained to implicitly identify influential factors and disclose them, offering a scalable path toward reducing unfaithful reasoning in LLMs. 

---
# VibeVoice-ASR-BitNet Technical Report 

**Authors**: Songchen Xu, Ting Song, Shaohan Huang, Zhiliang Peng, Yan Xia, Yujie Tu, Xin Huang, Jianwei Yu, Li Dong, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2607.21075)  

**Abstract**: We present VibeVoice-ASR-BitNet, a compressed variant of VibeVoice-ASR optimized for real-time inference on edge CPUs. We apply heterogeneous quantization tailored to the computational characteristics of each stage: the VAE acoustic tokenizer uses full-pipeline INT8 quantization (I8_S) with kernel fusion and SIMD optimization, while the autoregressive language model adopts BitNet-style ternary weights (I2_S). To preserve accuracy under aggressive compression, we employ a progressive quantization-aware training strategy. For inference, we implement custom SIMD kernels and fused operators within the ggml framework targeting both ARM and x86 platforms, achieving real-time recognition with RTF < 1 using as few as 3 CPU threads. VibeVoice-ASR-BitNet is 1.6-2.3x faster than this http URL at comparable model sizes (~1.6 GB), with only modest accuracy degradation compared to the FP16 baseline. 

---
# HiMe: Real-Time Self-Hosted Personal Agent Platform for Health Insights with Wearable Devices 

**Authors**: Wei Liu, Siya Qi, Linhai Zhang, Lorainne Tudor Car, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2607.21019)  

**Abstract**: Traditional approaches to wearable health signal analysis, such as smartwatches, are constrained by rigid analytical frameworks and limited personalisation. The emergence of LLM agents creates a new opportunity for Personal Health Agentic Analysis, where health insights can be generated adaptively and in context. However, currently there is no open-source locally deployable platform capable of processing personal health data in real time while preserving privacy. We present HiMe, a locally deployable, privacy-first agent platform that is fully compatible with real-time health data ecosystems across a wide range of wearable devices. HiMe is guided by three design principles. The database is treated as a first-class component. Effectiveness and efficiency are jointly optimised to achieve a low-cost Pareto-optimal balance. Data are processed in real time while the user is modelled over the long term. Together, these principles make it practical for individuals to harness Personal Health Agents for continuous, personalised health monitoring for better wellbeing. 

---
# The Weight of Silence: A Causal Case for Weights Over the Scratchpad in Latent Chess Reasoning 

**Authors**: Ishan S. Kshirsagar  

**Link**: [PDF](https://arxiv.org/pdf/2607.20952)  

**Abstract**: Latent, or silent, reasoning lets language models carry out intermediate computation in continuous vector space instead of words, and is widely assumed to function as an internal scratchpad the model actively consults during inference. Whether that assumption survives reinforcement learning has not been tested directly: existing causal analyses of latent reasoning are confined to math and logic tasks, and compare a model's reliance on its thoughts within a single checkpoint, never before and after an RL stage. We train a chess-playing model through a staged latent-reasoning curriculum followed by reinforcement learning, and find legality climbs monotonically to 61% (from a 48% pre-RL baseline) while checkmate confabulation is eliminated entirely. To locate this gain, we run a six-condition causal intervention suite on the same model before and after RL: substituting or adding matched noise to the latent thought vectors leaves performance unchanged, ablating them causes only mild degradation, and only exact-zero vectors cause collapse. This robustness gap is itself the finding: under exact-zero corruption, legality collapses to 1% pre-RL versus 9% post-RL, a gap that survives correction for testing across the full battery; milder conditions trend similarly without independently reaching significance. RL appears to add robustness to disruption, not reliance on thought content. These results push back against the field's default assumption that latent thoughts function as an actively consulted inference-time scratchpad, and instead indicate latent reasoning's principal effect here is shaping the model's parameters during training. We also demonstrate a working RL gain in chess, a domain outside the math and logic settings where multiple groups report the same latent-reasoning-plus-RL recipe failing to improve accuracy over SFT. 

---
# Chemical Chain-of-Thought Functions as a Hallucination-Prone Molecular Scratchpad 

**Authors**: Jiatong Li, Yuxuan Ren, Weida Wang, Xiaoyong Wei, Yatao Bian  

**Link**: [PDF](https://arxiv.org/pdf/2607.20935)  

**Abstract**: Chemical reasoning language models are expected to derive molecular answers through faithful chain-of-thought (CoT). However, across four reasoning model families and twelve chemistry tasks, hallucination is widespread and largely decoupled from answer correctness: correct answers often coexist with fabricated structural claims absent from the relevant molecules. Yet this does not make the reasoning trace computationally irrelevant. Attribution analyses suggest a shared scratchpad function expressed in model-specific forms: Chem-R and ether-0 rely on fragmented SMILES drafts, whereas ChemDFM-R emphasizes scaffold, positional, and naming cues. Notably, perturbing Chem-R's SMILES sketches degrades generation, showing that structural drafts can be causally load-bearing even when verbal structural claims are largely inert. Together, these results show that chemical CoT is neither a faithful explanation nor merely a post-hoc rationalization, but a hallucination-prone molecular scratchpad. This finding cautions against treating CoT as direct evidence of faithful reasoning and motivates process-level supervision beyond answer-only evaluation. 

---
# Transformer-Assisted LLM-Based Source Code Summarisation: to Enable More Secure Software Development 

**Authors**: Jesse Phillips, Tracy Hall, Paul Rayson, Mo El-Haj  

**Link**: [PDF](https://arxiv.org/pdf/2607.20933)  

**Abstract**: Neural Source Code Summarisation (NSCS) aims to generate natural language summaries of source code to improve developers' and maintainers' understanding of code. Source code summaries are vital during the maintenance phase of the Secure Software Development Lifecycle (SSDLC), as they improve maintainers' understanding of code and help reduce the number of bugs and vulnerabilities in a software system. However, summaries are often missing, incomplete, or outdated in many software systems. Solutions to this problem use small, task-specific Transformer models or code-aware Large Language Models (LLMs). Task-specific Transformer-generated summaries often score well across many natural language generation (NLG) metrics, but these metrics reward lexical overlap rather than summary quality. Conversely, the ability of LLMs to capture semantics and produce high-quality summaries presents an exciting solution to this problem. This is especially relevant given the increased availability of LLMs and improvements in workstation hardware in recent years, which mean that some LLMs can now be run on developers' workstations. However, because of their abstractive nature, LLM-generated code summaries often differ greatly from developer-written summaries in the words and phrases they use, resulting in low scores across NLG metrics. We show how combining these two methods, by using Transformer-generated summaries in prompt engineering, may enable LLMs to create better source code summaries and help software practitioners maintain secure systems. We prompt four LLMs using four different prompts, with a task-specific Transformer used to assist the LLMs within the prompts. We present "Transformer-Assisted LLM-Based Source Code Summarisation", a method through which we observe an improvement of 7.8% in BLEU-4 and 5%. 

---
# Position Bias is Hidden Behind Ceiling Effects: A Permutation Diagnostic for LLM Benchmarks 

**Authors**: Hiroki Tamba  

**Link**: [PDF](https://arxiv.org/pdf/2607.20864)  

**Abstract**: Position bias in multiple-choice LLM evaluation is widely cited as a confound in capability comparisons, but published measurements rely on single answer-order shuffles whose results confound the bias signal with content-level noise and sampling stochasticity. I introduce inspect_permute, an open-source extension to the inspect_ai evaluation framework that runs exhaustive answer-order permutations per question and reports the chi-squared / Cramer V signature of position bias with bootstrap confidence intervals. I apply the tool across four vendors (gpt-4o-mini, claude-haiku-4-5, gemini-2.5-flash, grok-3) on five MMLU subjects, 24,000 API calls under temperature-0 generation, with falsifier predictions pre-registered via a public SHA-256 hash before half the data was observed. Position bias turns out to be statistically detectable only within a roughly 60-95% base-accuracy Goldilocks zone. Below it, processing-load dominance swamps subject-specific signal; above it, ceiling effects compress the variance below the chi-squared test resolution. Detectable cells separate into two mechanism types: monotone A-to-D decrease (processing_load, in low-tier models) and non-monotone D-drop (content_ambiguity, in a narrow capability band). Standard MMLU places every frontier-tier model above the detection band, so absence of signal there should be read as not measurable, not unbiased. Together with the ceiling-effect characterisation in arXiv:2606.26185, this work brackets the detectable region of position-bias measurement and makes the field central question askable in a verifiable form. Package, data, preregistration under MIT. 

---
# Beyond Heavy Log Curation: Perplexity-Based APT Detection via Unsupervised, Context-Augmented Language Models 

**Authors**: Shoya Otsu, Kei Suzuki, Toshiaki Koike-Akino, Jing Liu, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20832)  

**Abstract**: Advanced Persistent Threats (APTs) remain difficult to detect because only a small fraction of events in large-scale logs are attack-related, and investigation is expensive and hard to scale. Prior machine-learning approaches can reduce analyst workload, but they often rely on heavily curated training data and sophisticated preprocessing pipelines. Building and maintaining such pipelines require substantial domain expertise and engineering cost. Motivated by insights from a study of a strong APT detection baseline, we propose CAPTAIN (Context-Augmented Perplexity-based Threat Activity log detectIoN), a perplexity-based detector that leverages general, pre-trained language models with minimal, domain-agnostic preprocessing, enabling robust scoring of long, minimally processed log entries. CAPTAIN encodes recent history with an encoder model and a Q-Former-style bridge, then injects the compact context tokens into the decoder input so that perplexity reflects temporal context. To improve stability, CAPTAIN additionally applies smoothing filters to the perplexity time series. Across APT-oriented benchmarks, CAPTAIN competes with strong existing baselines and remains robust under substantially less curated inputs, that reduces the development and operational cost of advanced log preprocessing. 

---
# Refusal-Gated Decoding: Preserving Refusal Behavior Under High-Temperature Sampling 

**Authors**: Phillip Howard, Xin Su, Allen Roush, Manikandan Ravikiran, Amir Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2607.20791)  

**Abstract**: High-temperature sampling is one of the primary mechanisms for increasing diversity in LLMs. Recent advances in truncation-based sampling techniques have helped mitigate drawbacks of high-temperature sampling such as neural text degeneration, thereby enabling greater diversity in LLM outputs without sacrificing coherence. However, increasing the entropy of the token probability distribution via high temperatures has also been shown to weaken model guardrails by reducing the model's refusal response in the presence of harmful prompts. Despite the potential benefits of high-temperature sampling and the importance of maintaining model safety, there is a lack of existing solutions for maintaining the refusal behavior of LLMs under a higher entropy regime. To address this gap, we systematically study how temperature influences refusal behavior in LLMs and propose an efficient sequential decoding approach which preserves a model's greedy decoding refusal response at high temperatures while incurring minimal additional latency. Through extensive experiments, we show that our approach preserves 91-99% of the greedy decoding refusal behavior across three benchmark datasets without compromising the model's high-temperature response for safe prompts. Our work demonstrates how refusal behavior can be maintained in an efficient manner for applications which require high-temperature sampling. 

---
# GaugeQuant: Online Learning of Quantization-Optimal Bases from LLM Symmetries 

**Authors**: Miguel P. Bento, João Seabra  

**Link**: [PDF](https://arxiv.org/pdf/2607.20757)  

**Abstract**: Transformers are known to have internal continuous symmetries that leave outputs invariant, while modifying quantization. GaugeQuant leverages this in-training by introducing a LogSumExp term to the loss that breaks the symmetries, thus selecting a basis that minimizes activation outliers. A stop-gradient operator ensures that only rotation matrices are updated, yielding the language modeling objective completely unaltered. Our requires no specific calibration data, no quantization simulation, and adds negligible training overhead. With the LLaMA-2 7B model under W4A4 quantization with group size 128, perplexity drops from 8.22 to 6.73, competing with post-training methods that require frozen models and calibration datasets. Under W4A16, perplexity drops from 11.16 to 5.45. Code is available at this https URL. 

---
# NVIDIA-labs OO Agents: Native Python Object-Oriented Agents 

**Authors**: Paul Furgale, Severin Klingler, James Nolan, Matt Staats, Gaia Di Lorenzo, Elisa Martinez Abad, Christian Schüller, Razvan Dinu, Alessio Devoto, Pascal Berard, Gal Kaplun, Elad Sarafian, Riccardo Roveri, Leon Derczynski, Ricardo Silveira Cabral  

**Link**: [PDF](https://arxiv.org/pdf/2607.20709)  

**Abstract**: Traditional agent development is split across prompt templates, tool schemas, callback code, and workflow graphs. We present NVIDIA Object-Oriented Agents (NOOA), a model-agnostic Python framework for building reliable AI agents. NOOA takes a simpler approach: an agent is a Python object. Its methods are the actions the model can take, fields are its state, docstrings are its prompts, and its type annotations are contracts. A method whose code body consists of "..." is completed at runtime by an LLM-driven agent loop, while methods with normal bodies remain standard deterministic Python. This gives developers and agents the same interface, so agent behavior can be tested, traced, refactored, and improved just like other software.
This paper makes three contributions. (1) We present the agent-as-a-Python-object programming model and the design principles behind it. Where Python has existing abstractions, we adopt them directly. Agent-specific capabilities--context, events, state rendering, long-term memory, and validated LLM loops--are exposed through simple Pythonic APIs, so both developers and agents share one familiar programming model. (2) We identify six model-facing ideas that NOOA is, to our knowledge, the first to combine on a single surface: typed input/output, pass-by-reference over live objects, code as action, programmable loop engineering, explicit object state, and model-callable harness APIs for context and events. We find the community already converging on several of these ideas--often as experimental or partial features--and present the comparison to encourage further adoption. (3) We demonstrate that current models use this interface effectively, both in targeted capability tests and on agentic and reasoning benchmarks such as SWE-bench Verified and Terminal-Bench 2.0 and ARC-AGI-3. 

---
# WaveformQA: Benchmarking LLM Temporal Reasoning on Digital Waveforms 

**Authors**: Yichuan Liu, Daniel Cummings, Nick Vadlamudi  

**Link**: [PDF](https://arxiv.org/pdf/2607.20638)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in code generation and reasoning, yet their ability to perform temporal reasoning over digital waveform data remains largely unexplored. Although reasoning over digital waveforms is a critical bottleneck in design verification, existing benchmarks primarily evaluate hardware description language (HDL) code generation and use waveforms only as supplementary context. This paper presents WaveformQA, an open-source question-answering benchmark for evaluating LLM temporal reasoning over digital waveforms. The benchmark comprises 360 questions with programmatically generated ground truths across eight categories of varying difficulty, including questions targeting multi-signal correlation and event ordering. Waveforms are generated from open-source design implementations, ensuring reproducibility and grounding the benchmark in real hardware behavior. Evaluation of frontier LLMs reveals that while models achieve reasonable accuracy on simple queries, performance degrades due to context window limitations and reasoning difficulties on complex temporal and multi-step questions. In addition, we show that an event-time JSON representation of waveforms improves LLM reasoning accuracy versus the standardized value change dump (VCD) format. The open-source framework supports extending to new question categories and importing new waveform sources, enabling researchers to rapidly prototype temporal reasoning experiments. 

---
# Demonstrating GenDB: Instance-Optimized and Customized Query Processing Code Generation via LLM Agents 

**Authors**: Jiale Lao, Immanuel Trummer  

**Link**: [PDF](https://arxiv.org/pdf/2607.20630)  

**Abstract**: Traditional query processing engines require continuous development and extensions to support new techniques and user requirements, and in some cases, entirely new systems must be built from scratch. However, these engines are difficult to extend due to their internal complexity, and building new systems demands significant engineering effort and cost. To address this, we demonstrate GenDB, a generative query engine that shifts query processing from manually engineered systems to query processing code generation driven by Large Language Models (LLMs). An early prototype of GenDB uses LLM agents to generate instance-optimized query execution code tailored to specific data, workloads, and hardware resources. This prototype suits offline code generation for repetitive, templated queries, since the upfront generation cost amortizes over many executions and correctness can be ensured through extensive fuzz testing and manual inspection. For ad-hoc queries, GenDB can work with a traditional DBMS in a hybrid architecture: the DBMS handles one-off queries, while GenDB speeds up frequent SQL templates. Our demonstration allows users to (1) visually and interactively explore how GenDB analyzes workloads, profiles hardware resources and underlying data, produces query plans, generates code based on them, and finally uses an optimizer to iteratively achieve a correct and efficient implementation; (2) use visual inspection and analysis to gain qualitative insights into why GenDB produces code that achieves significantly better performance than state-of-the-art query engines on two benchmarks: TPC-H and a newly constructed benchmark designed to reduce potential data leakage from LLM training data; and (3) upload their own data and queries to explore GenDB with different LLMs and query patterns. 

---
# Are Single-Token Sparse Autoencoder Features Causally Necessary? Layer-Depth and SAE-Family Effects 

**Authors**: Seonglae Cho, Zekun Wu, Kleyton Da Costa, Rishi Kalra, Ilham Wicaksono, Adriano Koshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2607.20596)  

**Abstract**: Sparse autoencoder (SAE) features are used to interpret and steer large language models, yet whether a feature's causal role is stable across SAE families remains untested. Single-token features that activate on one vocabulary item provide the diagnostic case where ground truth permits direct comparison. We analyze 3.9M features across six models and three SAE families using zero-ablation at full layer depth. Single-token features cluster 4.7x tighter in decoder space and concentrate in early layers (Layer 0 in GPT2-Small; L0-L4 in Gemma). Ablating them yields Benjamini-Hochberg-significant logit reductions in 178 of 208 full-layer conditions, with depth controlling whether damage cascades downstream or shapes the output directly. Cross-family causal differences exceed within-family scale effects: on the same base model, GemmaScope and BatchTopK features remain causally anchored, while LlamaScope features are locally redundant. The target token's rank recovers to within 2x baseline 96-98% of the time after the same ablation, and a controlled activation-function comparison reverses sign within the same model, leaving training recipe as the residual candidate. Cross-family interpretability claims are therefore sensitive to training methodology, not just activation function or scale. 

---
# CMI-Mem: Toward Generalizable Long-Term Memory Management via CMI-Augmented Reinforcement Learning 

**Authors**: Yubo Wang, Qiuyu Zhao, Zenghui Sun, Shichao Dong, Jinsong Lan, Xiaoyong Zhu, Haoyang Li, Bo Zheng, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.20553)  

**Abstract**: Memory Manager models are pivotal in agent systems. Existing methods rely predominantly on LLM-judged synthetic question-answer (QA) pairs, making memory valuation dependent on sampled queries and the downstream reader. To address this limitation, we propose \textbf{CMI-Mem}, a reinforcement learning(RL)-based lightweight memory manager model with a hybrid reward that combines downstream QA correctness and intrinsic Conditional Mutual Information (CMI). CMI evaluates the information contributed by new conversational inputs relative to the current memory state without conditioning on a sampled QA query, thereby complementing rather than replacing QA grounding. Our codes are available at: this https URL , and the CMI-Mem-4B model checkpoint is available at: this https URL 

---
# AppWorld-UL: Benchmarking Diverse Agent-User Interactions for Tool-Use 

**Authors**: Junzhi Chen, Harsh Trivedi, Jane Pan, Michael JQ Zhang, Tejas Srinivasan, Niranjan Balasubramanian, Ashish Sabharwal  

**Link**: [PDF](https://arxiv.org/pdf/2607.20536)  

**Abstract**: Tool-use agents that address day-to-day digital tasks such as ordering groceries must not only operate applications, but also interact with the user, e.g., to ask clarification questions, prompt for confirmation, and inform the user when the instruction is infeasible. However, current benchmarks for evaluating agent-user interactions do not capture the diversity of such interactions. Further, they operate in small environments with few, often non-state-changing, APIs. To address this gap, we introduce AppWorld-UL, a ``user-in-the-loop'' benchmark of 516 challenging tasks requiring diverse agent-user interactions. Building upon the AppWorld framework with 9 popular simulated apps like Amazon and Spotify, we systematically modify original tasks to introduce ambiguities and constraints that necessitate various types of agent-user interaction. User behavior is simulated by an LLM prompted to respond with carefully designed knowledge boundaries, offering more reliable simulation than the unconstrained or overly rigid alternatives used in prior work. Our evaluation reveals that a state-of-the-art LLM, Claude Opus 4.7, achieves only 48.6% success on AppWorld-UL, and only 35.7% on the harder, compositional subset. On the stricter, scenario-level metric, compositional task performance drops to only 21.3%. Our analysis reveals that correct user-interaction is crucial for success. This demonstrates the benchmark's difficulty and its potential to advance research on user-in-the-loop tool-use agents. 

---
# Telco-GAIA: Bilingual Benchmark for Agents in Telecom Domain 

**Authors**: Dmitrii Khizbullin, Zaid Alyafeai, Abdelrahman Eldesokey, Nourah AlSultan, Raghad Alshalan, David R. Pugh, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2607.20510)  

**Abstract**: We introduce Telco-GAIA, a bilingual, multi-modal benchmark for evaluating tool-using agents on the data of a real-world telecommunications operator. Telco-GAIA comprises 100 human-verified question-answering tasks, in English and Arabic, that each demand multi-hop reasoning (4.2 hops on average) over three heterogeneous sources: a static website snapshot (HTML, images, and linked PDFs), a synthetic relational SQL database, and external web archives, spanning text, image, and tabular modalities. The benchmark is delivered as a sandboxed Docker environment and scored by normalized exact string matching, making evaluation objective, deterministic, and reproducible over time without any LLM-as-a-Judge. Evaluating a purpose-built reference agent across twelve commercial and open LLMs, we find Telco-GAIA challenging: even the strongest model solves only 71% of tasks; under a moderate cost budget, this falls to about 40%, and the visually grounded categories remain the weakest, where the average backend scores below 30%, leaving substantial headroom in document and image understanding. Telco-GAIA offers a rigorous, reproducible testbed for enterprise agents and a template for constructing closed-domain benchmarks. 

---
# PhantomFill: When the Form Demands an Answer, Language Models Invent One 

**Authors**: Rana Muhammad Usman  

**Link**: [PDF](https://arxiv.org/pdf/2607.20492)  

**Abstract**: Language models in production do not write prose. They fill forms: JSON fields, function arguments, extraction templates. We show that the form itself causes hallucination.
We ask thirteen models the same question about the same input and change only the answer format. The inputs are built so the question cannot be answered: a viral post showing 12,400 likes but no visible replies, a support ticket whose call was never transcribed. In free text, GPT-5.5 answers honestly. It says there is no reply data, 98% of the time. Given a required JSON field for sentiment, the same model invents an answer 40 times out of 40. It fabricates the mood of crowds it never saw and quotes customers it never heard.
The pattern holds with force. Required fields drive fabrication to 100% in ten of thirteen models. An explicit "insufficient evidence" option rescues only the frontier: all nine open-weight models ignore it. A direct instruction, do not infer sentiment, is overridden by the schema in four of six models. Resistance does not come with scale: within a single model family, the smallest model refuses, the mid-sized model fabricates, the largest refuses again. Honesty under format pressure is a training outcome that no one is measuring.
The fabrication hides exactly where hedging is impossible: in required enums and minimum-count arrays, fields where no disclaimer fits. We release PhantomFill, a benchmark with deterministic scoring and two reportable numbers: the Coerced Fabrication Rate and the Escape Utilization Rate. The fix we test is one line of schema. The failure we measure is everywhere. 

---
# DFAH-Bench: Benchmarking Observable Agent Instability in Financial Decision-Making 

**Authors**: Raffi Khatchadourian  

**Link**: [PDF](https://arxiv.org/pdf/2607.20491)  

**Abstract**: Standard evaluation benchmarks measure what a tool-using agent decides, not whether it arrives at that decision through the same process each time. We introduce DFAH-Bench, a replay benchmark that measures observable behavioral instability in financial agent decision-making across three channels -- tool-call trajectories, evidence contacts, and decision concentration -- none of which require access to hidden reasoning text. Across 8,127 replay episodes spanning 10 models and 3 financial tasks, we find that outcome agreement alone is an incomplete stability signal: frontier models can agree on decisions 95% of the time while following the same tool path only 77% of the time -- an 18-percentage-point gap (95% CI: [0.14, 0.22]) that outcome-only evaluation misses entirely. Among frontier-model case groups with high decision agreement, over 55% exhibit meaningful trajectory divergence. We identify three behavioral profiles: pattern matchers that achieve near-perfect agreement by collapsing to a single output regardless of input, stable executors with relatively consistent tool-use processes, and trajectory divergers that reach the same conclusions through materially different tool paths and evidence contacts. The benchmark code, metric scripts, replay logs, benchmark card, dataset README, and release manifest are released in the accompanying repository. 

---
# Directional Hallucinations: Ideological Drift in News-Grounded LLM Question Answering 

**Authors**: Chendi Wang, Liam Cunningham, Tom Yishay, Jieying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.20487)  

**Abstract**: Large language models (LLMs) are increasingly used to answer questions about political information, including in election-adjacent information settings where factual errors and ideological distortions are high-stakes. We present a reproducible measurement framework that treats hallucinations, unsupported statements in document-grounded QA, as diagnostic signals of ideological drift. Using 21,727 expert-labeled U.S. political news articles from QBias spanning left, center, and right sources, we (i) generate an article-specific question, (ii) elicit document-grounded answers from three open-weight LLMs and one proprietary model, (iii) detect sentence-level hallucinations via reference-based comparison, (iv) classify the ideological valence of hallucinated sentences with a fine-tuned stance classifier, and (v) probe output logits to relate token-level uncertainty to hallucination and drift. Hallucination rates vary substantially across models and concentrate in contentious topics, while source-ideology differences in hallucination frequency are modest. In contrast, hallucination content exhibits robust leftward drift: a majority of hallucinated sentences are classified as left-leaning, including among hallucinations generated from right-leaning sources. Logit-level analysis shows hallucinations arise in high-entropy generation contexts, and in some models uncertainty also predicts leftward drift, consistent with an "uncertainty to guessing" mechanism. We discuss implications for auditing AI-mediated political information and for designing safeguards in election-relevant deployments. 

---
# Expectation Alignment of Language Models for Real-World User Expectations 

**Authors**: Miaomiao Li, Yang Wang, Bin Liang, Shudong Liu, Zhiwei Zhang, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2607.20485)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance on standard benchmarks, yet it remains largely unexplored whether they truly meet user expectations. Existing evaluation approaches, relying on model heuristics, expert rubrics, or user simulation, fail to capture the diversity and subtlety of real human expectations, causing models to appear competent while misaligning with what users actually seek. We present the first systematic study of user expectations in real-world LLM interactions, proposing a principled procedure to extract semantically rich expectations and introducing ExpectBench, a benchmark grounded in real user expectations. Analyses reveal that current LLMs struggle to satisfy and anticipate what users hope to obtain, highlighting a fundamental source of misalignment. Building on these observations, we propose LENS, a lightweight latent expectation-aware response generation framework. LENS enables models to internalize user expectations and generate better-aligned responses, consistently improving expectation satisfaction and underscoring the importance of explicitly modeling user expectations for realistic human-AI alignment. 

---
# PersonaTrail: Benchmarking Personalized Web Agents through Browsing Trails 

**Authors**: Seungbin Yang, Chaewoon Ki, Dohyun Lee, Jaegul Choo, ChaeHun Park  

**Link**: [PDF](https://arxiv.org/pdf/2607.20482)  

**Abstract**: Recent advances in large language models have enabled web agents to autonomously execute complex tasks. In practice, users frequently provide underspecified instructions, requiring agents to infer the missing context from their raw browsing histories. Existing benchmarks fail to capture this form of personalization, as they either restrict tasks to fully explicit prompts or abstract web interaction history into simplified forms. To bridge this gap, we introduce PersonaTrail, a benchmark for personalized web agents operating in a managed open web environment. By leveraging realistic browsing trajectories as user history, PersonaTrail evaluates an agent's ability to infer user preferences and recall information from past browsing sessions. We further propose Preference-Aware Contextual Memory (PACMem), a framework that decomposes raw browsing histories into two types of structured memory: factual memories that summarize individual sessions and preference memories that distill recurring behavioral patterns. At inference time, the agent retrieves the most relevant entries from these memories to guide personalized navigation. Extensive experiments show that PACMem consistently outperforms existing memory-based baselines on both tasks. 

---
# Beyond Liars' Bench: The Impact of Lie Typology, Depth, and Sparsity on Deception Detection in LLMs 

**Authors**: Amr Moustafa, Max Feser, Florian Mai  

**Link**: [PDF](https://arxiv.org/pdf/2607.20479)  

**Abstract**: Training probes to detect deceptive outputs from large language models is still an open problem. Recent work has demonstrated that detection probes fail especially in out-of-domain scenarios -- training on one type of lie does not transfer well to deception scenarios involving other types of lies. In this work, we conduct a systematic study on how various factors impact detection performance: representation depth, probe expressivity, sparse feature representations, and the lie typology of the training data. To this end, we augment standard benchmark training data with a supplementary dataset containing diverse types of deception, including fabrication, omission, and exaggeration examples. Analyzing these factors across seven probe types, our experimental results show that the optimal representation depth is highly dataset-dependent, more expressive probes provide only selective gains over linear baselines, and sparse autoencoder features perform similarly to dense hidden states. Ultimately, we demonstrate that the choice of training data and lie typology substantially changes detectability, highlighting that deception detection is a highly representation-dependent problem. 

---
# Benchmarking Large Language Models on Multi-Sensor Physical Hazard Assessment 

**Authors**: Faizan Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2607.20476)  

**Abstract**: We present an empirical benchmark evaluating how five large language models assess multisensor physical hazard data. Testing 60 scenarios across three categories - multi-sensor joint assessment, response proportionality, and pattern disambiguation - with 1,800 API calls at temperature 0.0, we find that all tested models consistently produced no precautionary warning signal across the tested scenarios where multiple sensors are simultaneously elevated below their individual safety limits, while achieving near-perfect accuracy on single-sensor threshold violations. All five models (ChatGPT-4o, Gemini 2.5 Flash, DeepSeek, Kimi, Llama 3.1 8B) score near zero on Category A multi-sensor scenarios (Q2: 0.000-0.208; Q3: 0.000-0.592) compared to strong performance on single-sensor scenarios (Category B Q1: 0.975-1.000). Structured tabular formatting shows no consistent advantage over plain prose; ChatGPT-4o performs significantly better under prose (p = 0.001). These findings have direct implications for practitioners deploying the tested models in physical safety monitoring systems. 

---
# SonicSampler: Unified Tile-Aware Kernels for LLM Sampling and Speculative Verification 

**Authors**: Pragaash Ponnusamy, Shivam Sahni, Jue Wang, Tri Dao  

**Link**: [PDF](https://arxiv.org/pdf/2607.20475)  

**Abstract**: Sampling in LLM inference comprises a combinatorial set of logit processing, token selection, and verification operations for speculative decoding. However, existing implementations either accelerate only subsets of this pipeline, rely on multiple kernel launches, or assume homogeneous sampling behavior across a batch, limiting support for dynamic serving workloads and preventing efficient CUDA Graph execution. We present $\textbf{SonicSampler}$, a unified suite of tile-aware Triton kernels that vertically fuses the complete sampling pipeline into a fixed, workload-aware execution model. Our kernels support dynamic per-request sampling behaviors, including grammar-constrained decoding, repetition, frequency and presence penalties, logit bias, temperature scaling, top-$k$ / top-$p$ / min-$p$ filtering, and speculative verification - within a single batched kernel while remaining fully CUDA Graph-compatible. Central to our approach is a novel hierarchical two-stage top-$k$ algorithm that achieves up to $\textbf{10x speedup}$ over competitive baselines and exploits the low-entropy structure of LLM outputs to enable efficient selection over large vocabularies. Across heterogeneous speculative decoding workloads, SonicSampler achieves up to $\textbf{16x speedup}$ over state-of-the-art baselines while preserving flexible batched execution. 

---
# DataPrep-Bench: Benchmarking LLMs as Training Data Preparators 

**Authors**: Hao Liang, Qifeng Cai, Yibo Lin, Jianzhuo Du, Qifeng Xia, Sizhe Qiu, Linzhuang Sun, Meiyi Qiang, Zhaoyang Han, Xiaochen Ma, Bohan Zeng, Ruichuan An, Conghui He, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.20465)  

**Abstract**: The quality of training data fundamentally determines the capabilities of large language models (LLMs), yet no unified benchmark exists to measure how well LLMs, agents, and data-centric workflows actually prepare training data end to end. We view LLM-driven data preparation as comprising two complementary capabilities: data construction, which transforms raw sources into supervised training data, and data quality evaluation, which predicts the training value of candidate datasets before downstream training; throughout, "quality" refers to downstream training utility rather than surface-level textual properties. We introduce DataPrep-Bench, the first unified benchmark that jointly evaluates both capabilities under a shared downstream-grounded protocol over six domains and multiple base models. For data construction, methods consume identical raw sources and are scored by fine-tuning a base model on their outputs jointly with Dolly-15k; alongside this track we release Data-Construction-Skill, a skill-guided agent that lifts the Dolly-only baseline by nearly 20 points absolute on Llama-3.1-8B Finance and is competitive with the strongest agent- and DataFlow-based methods in knowledge-extraction-dense domains. For data quality evaluation, scoring functions are scored by Pearson correlation with downstream performance on a shared candidate pool; we release the Distributional Alignment Score (DAS), a distribution-based evaluator that uses MMD between a candidate dataset and a domain proxy. DAS attains the strongest cross-model correlation in four of six domains and is the only metric clearing r > 0.70 simultaneously in Math, Science, and Medical, outperforming existing quality-, diversity-, and heuristic-based evaluators. DataPrep-Bench provides a unified, downstream-grounded framework for measuring progress on both capabilities as co-equal targets of LLM-driven data preparation. 

---
