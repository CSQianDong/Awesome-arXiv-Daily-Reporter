# FlexSQL: Flexible Exploration and Execution Make Better Text-to-SQL Agents 

**Authors**: Quang Hieu Pham, Yang He, Ping Nie, Canwen Xu, Davood Rafiei, Yuepeng Wang, Xi Ye, Jocelyn Qiaochu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02815)  

**Abstract**: Text-to-SQL over large analytical databases requires navigating complex schemas, resolving ambiguous queries, and grounding decisions in actual data. Most current systems follow a fixed pipeline where schema elements are retrieved once upfront and the database is only revisited for post-hoc repair, limiting recovery from early mistakes. We present FlexSQL, a text-to-SQL agent whose core design principle is flexible database interaction: the agent can explore schema structure, inspect data values, and run verification queries at any point during reasoning. FlexSQL generates diverse execution plans to cover multiple query interpretations, implements each plan in either SQL or Python depending on the task, and uses a two-tiered repair mechanism that can backtrack from code-level errors to plan-level revisions. On Spider2-Snow, using gpt-oss-120b, FlexSQL achieves a 65.4\% score, outperforming strong open-source baselines that use stronger, larger models such as gpt-o3 and DeepSeek-R1. When integrated into a general-purpose coding agent (as skills in Claude Code), our approach yields over 10\% relative improvement on Spider2-Snow. Further analysis shows that flexible exploration and flexible execution jointly contribute to the effectiveness of our approach, highlighting flexibility as a key design principle. Our code is available at: this https URL 

---
# Reinforcement Learning for LLM-based Multi-Agent Systems through Orchestration Traces 

**Authors**: Chenchen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02801)  

**Abstract**: As large language model (LLM) agents evolve from isolated tool users into coordinated teams, reinforcement learning (RL) must optimize not only individual actions but also how work is spawned, delegated, communicated, aggregated, and stopped. This paper studies RL for LLM-based multi-agent systems through orchestration traces: temporal interaction graphs whose events include sub-agent spawning, delegation, communication, tool use, return, aggregation, and stopping decisions.
Using this lens, we identify three technical axes. First, reward design spans eight families, including orchestration rewards for parallelism speedup, split correctness, and aggregation quality. Second, reward and credit signals attach to eight credit- or signal-bearing units from token to team; explicit counterfactual message-level credit remains especially sparse in our curated pool. Third, orchestration learning decomposes into five sub-decisions: when to spawn, whom to delegate to, how to communicate, how to aggregate, and when to stop. In our curated pool as of May 4, 2026, we found no explicit RL training method for the stopping decision.
We connect academic methods to public industrial evidence from Kimi Agent Swarm, OpenAI Codex, and Anthropic Claude Code. The resulting scale gap is a gap between publicly reported deployment envelopes and open academic evaluation regimes, not independent verification of industrial training traces. We release the artifact at this https URL, including an 84-entry tagged paper pool, a 32-record exclusion log, scripted corpus statistics, and a minimal JSON schema for replayable orchestration traces. 

---
# mdok-style at SemEval-2026 Task 10: Finetuning LLMs for Conspiracy Detection 

**Authors**: Dominik Macko  

**Link**: [PDF](https://arxiv.org/pdf/2605.02712)  

**Abstract**: SemEval-2026 Task 10 is focused on conspiracy detection. Specifically, the goal is to detect whether a Reddit comment expresses a conspiracy belief. Our submitted mdok-style system utilizes data augmentation and self-training (to cope with a rather small amount of training data) to finetune the Qwen3-32B model for a binary text-classification task. The submitted system is very competitive, ranking in the 85th percentile (8th out of 52 submissions). The results shown that our approach, which originated in machine-generated text detection, can be used for conspiracy detection as well. 

---
# mdok-style at SemEval-2026 Task 9: Finetuning LLMs for Multilingual Polarization Detection 

**Authors**: Dominik Macko, Alok Debnath, Jakub Simko  

**Link**: [PDF](https://arxiv.org/pdf/2605.02695)  

**Abstract**: SemEval-2026 Task 9 is focused on multilingual polarization detection. Specifically, it covers the identification of multilingual, multicultural and multievent polarization along three axes (in subtasks), namely detection, type, and manifestation. Online polarization presents a concern, because it is often followed by hate speech, offensive discourse, and social fragmentation. Therefore, its detection before it escalates is crucial for a safer and more inclusive online space. We have coped with this SemEval task by finetuning mid-size LLMs for the sequence-classification task using the QLoRA parameter-efficient finetuning technique. The training data augmented the multilingual (22 languages) training sets by anonymized, lower-cased, upper-cased, and homoglyphied counterparts, making the detection more robust. 

---
# Fuzzy Fingerprinting Encoder Pre-trained Language Models for Emotion Recognition in Conversations: Human Assessment and Validity Study 

**Authors**: Patrícia Pereira, Helena Moniz, Joao Paulo Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2605.02665)  

**Abstract**: In Emotion Recognition in Conversations (ERC), model decisions should align with nuanced human perception and ideally provide insights on the classification process. Standard encoder pre-trained language models (PLMs) are the state-of-the-art at these tasks but offer little insight into why a certain prediction is made. This is especially problematic in imbalanced datasets, where most utterances are labeled as neutral, making these models frequently misclassify minority emotions as the majority neutral class. To tackle this issue, we introduced a novel, interpretable approach to ERC by combining PLMs with Fuzzy Fingerprints (FFPs). FFP provide class-specific prototypes that reflect the characteristic class activation patterns in the PLM's latent space. They are derived by ranking and fuzzifying the activations of the pooled conversational context-dependent embeddings across training instances for each emotion. At inference time, each input utterance is similarly fuzzy fingerprinted and matched to the emotion prototypes using a fuzzy similarity function based on the aggregation of the intersection of the fuzzy sets that define each FFP. Experimental results show that FFP integration reduces overclassification into the neutral class and human evaluation further supports the adequacy of FFP predictions. Our proposed method thus bridges the gap between deep neural inference and human perception, performing at state-of-the-art level while simultaneously offering valuable insights into the classification procedure. 

---
# ContextualJailbreak: Evolutionary Red-Teaming via Simulated Conversational Priming 

**Authors**: Mario Rodríguez Béjar, Francisco J. Cortés-Delgado, S. Braghin, Jose L. Hernández-Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2605.02647)  

**Abstract**: Large language models (LLMs) remain vulnerable to jailbreak attacks that bypass safety alignment and elicit harmful responses. A growing body of work shows that contextual priming, where earlier turns covertly bias later replies, constitutes a powerful attack surface, with hand-crafted multi-turn scaffolds consistently outperforming single-turn manipulations on capable models.
However, automated optimization-based red-teaming has remained largely limited to the single-turn setting, iterating over static prompts and lacking the ability to reason about which forms of conversational priming induce compliance. While recent multi-turn, search-based approaches have begun to bridge this gap, the mutator design space underlying effective primed dialogues remains largely unexplored.
We present ContextualJailbreak, a black-box red-teaming strategy that performs evolutionary search over a simulated multi-turn primed dialogue. The strategy leverages a graded 0-5 harm score from a two-level judge as an in-loop signal, enabling partially harmful responses to guide the search process rather than being discarded.
Search is driven by five semantically defined mutation operators: roleplay, scenario, expand, troubleshooting, and mechanistic, of which the last two are novel contributions of this work.
Across 50 representative HarmBench behaviors, ContextualJailbreak achieves an ASR of 100% on gpt-oss:20B, 100% on qwen3-8B, 100% on llama3.1:70B, and 90% on gpt-oss:120B, outperforming four single- and multi-turn baselines by 31-96 percentage points on average.
The 40 maximally harmful attacks discovered against gpt-oss:120B transfer without adaptation to closed frontier models, achieving 90.0% on gpt-4o-mini, 70.0% on gpt-5, and 70.0% on gemini-3-flash, but only 17.5% on claude-opus-4-7 and 15.0% on claude-sonnet-4-6, revealing a pronounced provider-level asymmetry in alignment robustness. 

---
# Mapping Discourse Reframing: A Multi-Layer Network Approach to Italian HPV Vaccine Discourse on X (2010-2024) 

**Authors**: Lorella Viola  

**Link**: [PDF](https://arxiv.org/pdf/2605.02629)  

**Abstract**: Understanding how online narratives travel through coalitions is critical for identifying information disorder, yet computational analyses often rely on conservative network constructions that erase initially sparse but salient signals. This paper proposes a novel multi-layer framework that captures low-frequency signals of emerging information disorder allowing for locating where online discourse is reframed and amplified over time. The use case is 14 years of Italian discourse on X regarding the Human Papillomavirus (HPV) vaccine across three pivotal epochs (2010-2024). Utilizing hashtag co-occurrence networks, we introduce a dual-layer approach. We first identify robust core discourse coalitions through conservative community detection, revealing a stable prevention-oriented backbone contrasted with increasingly separable skepticism coalitions. We then introduce a coverage layer and project fringe hashtags into core coalitions based on weighted connectivity. Using a manually labelled set of skeptical and conspiratorial seed tweets, we demonstrate that this core-coverage projection significantly improves the recovery of long-tail, problematic hashtags while preserving an interpretable coalition structure. Our findings characterize the structural maturation of polarized narratives and provide a methodology for mapping how discourse is reframed and amplified by information disorder over time. 

---
# Synthetic Users, Real Differences: an Evaluation Framework for User Simulation in Multi-Turn Conversations 

**Authors**: Yu Lu Liu, Hyokun Yun, Tanya Roosta, Ziang Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.02624)  

**Abstract**: There is growing interest in exploring user simulation as an alternative to gathering and scoring real user-chatbot interactions for AI chatbot evaluation. For this purpose, it is important to ensure the realism of the simulation, i.e., the extent to which simulated dialogues reflect real dialogues users have with chatbots. Most existing methods evaluating simulation realism produce coarse quality signal and remain solely at the level of individual dialogues. To support more rigorous evaluation in this area, we propose realsim, an evaluation framework that enables practitioners to take a distributional view of real vs. simulated dialogues along 8 dimensions, covering attributes related to the communicative functions of the interaction, user states, and the surface form of user messages. We then instantiate the framework with a curated dataset of 1K multi-turn task-focused real user-chatbot dialogues that cover 16 domains of chatbot applications. Overall, we find that simulated users tend to struggle at capturing communication frictions that real users introduce to interactions, which could make evaluations based on such simulations overly optimistic. We also observe variability in performance across different domains, which may indicate a need for domain-specific user simulators. 

---
# Beating the Style Detector: Three Hours of Agentic Research on the AI-Text Arms Race 

**Authors**: Andreas Maier, Moritz Zaiss, Siming Bayer  

**Link**: [PDF](https://arxiv.org/pdf/2605.02620)  

**Abstract**: Reproducing an empirical NLP study used to take weeks. Given the released data and a modern agentic-research harness, we redo every experiment of a recent ACL\,2026 study on personal-style post-editing of LLM drafts -- and add three new ones -- with the human investigator acting only as a reviewer-in-the-loop. We reproduce all seven preregistered hypotheses and recover the paper's headline correlation between perceived self-similarity and embedding-measured self-similarity to three decimal places ($r{=}{+}0.244$, $p{<}10^{-8}$, $n{=}648$). Under a leakage-free held-out protocol, GPT-5.5 and Claude\,Opus\,4.7 close $71$--$75\,\%$ of the style gap to the same-author ceiling on $324$ paired tasks, against $24\,\%$ for the human post-edit, and beat the human post-edit on $\sim$$80\,\%$ of tasks. We then frame the same data as an AI-text detection arms race. A leave-authors-out linear SVM on LUAR-MUD embeddings reaches AUC $0.93$--$1.00$ across approaches; six diagnostics show that GPT-5.5 detection is mostly a length confound while Opus detection is a genuine stylistic signature. Given $T{=}20$ feedback iterations against the frozen detector, an Opus agent flips two of five held-out test mimics to the human half-space and shrinks every margin by an order of magnitude. With moderate effort against a known detector, a frontier LLM can already efficiently lower its own AI-detection probability. All code, $648$ mimic drafts, trained detectors, diagnostics, and adversarial trajectories are released. 

---
# Dependency Parsing Across the Resource Spectrum: Evaluating Architectures on High and Low-Resource Languages 

**Authors**: Kevin Guan, Happy Buzaaba, Christiane Fellbaum  

**Link**: [PDF](https://arxiv.org/pdf/2605.02608)  

**Abstract**: Transformer-based models achieve state-of-the-art dependency parsing for high-resource languages, yet their advantage over simpler architectures in low-resource settings remains poorly understood. We evaluate four parsers -- the Biaffine LSTM, Stack-Pointer Network, AfroXLMR-large, and RemBERT -- across ten typologically diverse languages, with a focus on low-resource African languages. We find that the Biaffine LSTM consistently outperforms transformer models in low-resource regimes, with transformers recovering their advantage as training data increases. The crossover falls within a resource range typical of treebanks for under-resourced languages. Morphological complexity (measured via MATTR) emerges as a significant secondary predictor of transformers' relative disadvantage after controlling for corpus size. These results indicate that the Biaffine LSTM may be better suited for syntactic tool development in low-resource regimes until sufficient annotated data is available to leverage the representational capacity of pre-trained transformers. 

---
# SemEval-2026 Task 7: Everyday Knowledge Across Diverse Languages and Cultures 

**Authors**: Nedjma Ousidhoum, Junho Myung, Carla Perez-Almendros, Jiho Jin, Amr Keleg, Meriem Beloucif, Yi Zhou, Rodrigo Agerri, Vladimir Araujo, Naomi Baes, James Barry, Joanne Boisson, Nancy F. Chen, Christine de Kock, Aleksandra Edwards, Joseba Fernandez de Landa, Mohamed Fazli Imam, Huda Hakami, Shu-Kai Hsieh, Joseph Marvin Imperial, Roy Ka-Wei Lee, Zhengyuan Liu, Chenyang Lyu, Younes Samih, Johan Sjons, Bryan Tan, Asahi Ushio, Weihua Zheng, Alice Oh, Jose Camacho-Collados  

**Link**: [PDF](https://arxiv.org/pdf/2605.02601)  

**Abstract**: We present our shared task on evaluating the adaptability of LLMs and NLP systems across multiple languages and cultures. The task data consist of an extended version of our manually constructed BLEnD benchmark (Myung et al. 2024), covering more than 30 language-culture pairs, predominantly representing low-resource languages spoken across multiple continents. As the task is designed strictly for evaluation, participants were not permitted to use the data for training, fine-tuning, few-shot learning, or any other form of model modification. Our task includes two tracks: (a) Short-Answer Questions (SAQ) and (b) Multiple-Choice Questions (MCQ). Participants were required to predict labels and were allowed to submit any NLP system and adopt diverse modelling strategies, provided that the benchmark was used solely for evaluation. The task attracted more than 140 registered participants, and we received final submissions from 62 teams, along with 19 system description papers. We report the results and present an analysis of the best-performing systems and the most commonly adopted approaches. Furthermore, we discuss shared insights into open questions and challenges related to evaluation, misalignment, and methodological perspectives on model behaviour in low-resource languages and for under-represented cultures. 

---
# Benchmarking Retrieval Strategies for Biomedical Retrieval-Augmented Generation: A Controlled Empirical Study 

**Authors**: Devi Prasad Bal, Subhashree Puhan  

**Link**: [PDF](https://arxiv.org/pdf/2605.02520)  

**Abstract**: Retrieval-Augmented Generation (RAG) offers a well-established path to grounding large language model (LLM) outputs in external knowledge, yet the question of which retrieval strategy works best in a high-stakes domain such as biomedicine has not received the controlled, multi-metric treatment it deserves. This paper presents a systematic empirical comparison of five retrieval strategies -- Dense Vector Search, Hybrid BM25 + Dense retrieval, Cross-Encoder Reranking, Multi-Query Expansion, and Maximal Marginal Relevance (MMR) -- within a biomedical question-answering RAG pipeline. All strategies share a fixed generation model (GPT-4o-mini), a common vector store (ChromaDB), and OpenAI's text-embedding-3-small embeddings, ensuring that observed differences are attributable to retrieval alone. Evaluation is conducted on 250 question-answer pairs drawn from a preprocessed subset of the BioASQ benchmark (rag-mini-bioasq) using four DeepEval metrics: contextual precision, contextual recall, faithfulness, and answer relevancy, each reported with 95% confidence intervals. A no-context ablation is included as a lower bound. Cross-Encoder Reranking achieves the best composite score (0.827) and highest contextual precision (0.852), confirming that query-document interaction yields measurable retrieval gains. Multi-Query Expansion, despite its recall-oriented design, produces the weakest contextual precision (0.671), suggesting naive query diversification introduces retrieval noise. MMR sacrifices answer relevancy for diversity, while the Dense baseline (composite 0.822) falls within 0.005 points of the top strategy. All RAG conditions dramatically outperform the no-context ablation on answer relevancy (0.658-0.701 vs. 0.287), confirming the practical value of retrieval. The full pipeline, hyperparameters, and evaluation code are publicly available. 

---
# Revisiting Semantic Role Labeling: Efficient Structured Inference with Dependency-Informed Analysis 

**Authors**: Sangpil Youm, Leah Jones, Bonnie J. Dorr  

**Link**: [PDF](https://arxiv.org/pdf/2605.02505)  

**Abstract**: Semantic Role Labeling (SRL) provides an explicit representation of predicate-argument structure, capturing linguistically grounded relations such as who did what to whom. While recent NLP progress has been dominated by large language models (LLMs), these systems often rely on implicit semantic representations, often lacking explicit structural constraints and systematic explanatory mechanisms. Traditionally, SRL systems have often relied on AllenNLP; however, the framework entered maintenance mode in December 2022, limiting compatibility with evolving encoder architectures and modern inference requirements. We revisit structured SRL modeling, introducing a modernized encoder-based framework that preserves explicit predicate-argument structure while enabling inference 10 times faster. Using BERT-base, the model attains comparable predictive performance, and RoBERTa and DeBERTa further improve F1 performance within the same framework. We adopt a dependency-informed diagnostic methodology to characterize span-level inconsistencies and conduct a representation-level analysis of LLM behavior under dependency-informed structural signals. Results indicate that dependency cues primarily improve structural stability. Finally, we illustrate how the framework's explicit predicate-argument structure can support multilingual SRL projection as a downstream application. 

---
# A multilingual hallucination benchmark: MultiWikiQHalluA 

**Authors**: Freja Thoresen, Dan Saattrup Smart  

**Link**: [PDF](https://arxiv.org/pdf/2605.02504)  

**Abstract**: Most hallucination evaluations focus on English, leaving it unclear whether findings transfer to lower-resource languages. We investigate faithfulness hallucinations, defined as model-generated content that is fluent and plausible but diverges from the provided input or is internally inconsistent. Leveraging the multilingual MultiWikiQA dataset, we utilize the LettuceDetect framework to create synthetic hallucination datasets for 306 languages, from which we train token-level hallucination classifiers for 30 European languages. In this work, we present evaluations of model hallucinations on a selection of languages: English, Danish, German, and Icelandic. Using these classifiers, we evaluate the hallucination rates for Qwen3-0.6B, Qwen3-14B, Gemma-3-12B-IT, cogito-v1-preview-qwen-32B, and cogito-v1-preview-llama-70B. Our classifiers reveal notably higher hallucination rates for Qwen3-0.6B (up to 60\% of answers containing at least one hallucination, peaking in Icelandic) and generally lower rates for larger models, with cogito-v1-preview-qwen-32B and cogito-v1-preview-llama-70B performing best on most languages. Hallucination rates are consistently higher for lower-resource languages, particularly Icelandic. 

---
# Accurate Legal Reasoning at Scale: Neuro-Symbolic Offloading and Structural Auditability for Robust Legal Adjudication 

**Authors**: Stanisław Sójka, Witold Kowalczyk  

**Link**: [PDF](https://arxiv.org/pdf/2605.02472)  

**Abstract**: Legal texts often contain computational legal clauses--provisions whose understanding requires complex logic. While frontier Large Reasoning Models (LRMs) can describe such clauses, building production-ready systems is limited by reasoning errors and the high cost of inference. We propose Amortized Intelligence, a neuro-symbolic approach where we use an LLM once to translate a legal text into Deterministic Autonomous Contract Language (DACL): a typed graph intermediate representation. Adjudication then relies on deterministic graph executions with a visually auditable trace. In comparison against runtime LRM baselines (including GPT-5.2 and Gemini 3 Pro), our DACL-based Agent achieves near-perfect consistency and mitigates the "reasoning cliff" observed in probabilistic models. The system reduces compute costs by over 90% in high-volume workflows while satisfying the strict auditability requirements of legal adjudication. 

---
# ATLAS: Article Tracking, Linking, and Analysis of Swedish Encyclopedias 

**Authors**: Albin Andersson, Salam Jonasson, Fredrik Wastring, Pierre Nugues  

**Link**: [PDF](https://arxiv.org/pdf/2605.02466)  

**Abstract**: The digitization of old encyclopedias represents an important step to improve access to historically structured knowledge. Often, however, this process does not go beyond an optical character recognition, leaving all the underlying structure unexploited. In addition, many encyclopedias had multiple editions reflecting the evolution of knowledge. The lack of structure in the raw text makes it difficult to track changes across these editions. In this work, we built a pipeline to restore the text structure, where we extract the headwords and identify entries; categorize the entities; match entries across editions; and link entries to a Wikidata item. We applied this pipeline to the four major editions of \textit{Nordisk familjebok}, an authoritative Swedish encyclopedia published between 1876 and 1951. We could extract the headwords with an F1 score of 97.8\% and we obtained an F1 score of 93.4\% on the headword classification. On a small-scale evaluation, we reached a 93\% precision on the cross-edition matching, 85\% precision and 16.5\% recall on the Wikidata linking. This shows that an automated approach to digitized historical knowledge is possible. This should facilitate the preservation of general knowledge and the understanding of knowledge transmission. The datasets and programs are available online. 

---
# Leveraging Argument Structure to Predict Content Hatefulness 

**Authors**: Nicolas Benjamin Ocampo, Davide Ceolin  

**Link**: [PDF](https://arxiv.org/pdf/2605.02457)  

**Abstract**: Information disorder is a challenging phenomenon that affects society at large. This phenomenon entails the diffusion of misleading, misinforming, and hateful content online. In different contexts, one aspect of the problem may prevail, but overall, this is a broad problem that requires comprehensive solutions. While each dimension of the problem (hate speech, disinformation, misinformation, etc.) requires in-depth analysis, in this paper, we look into the possibility of argument structure to provide relevant information to link these different areas of the problem. In particular, we focus on the WSF-ARG+ dataset, which consists of white supremacy forum messages annotated in terms of argument structure (premises and conclusion). There, we leverage the checkworthiness and hatefulness annotations of the argument components to obtain insights into the hatefulness of the whole message. Our results show promising insights (up to 96% F1), indicating the possibility of extending this direction in the future to tackle hateful content identification and information disorder countering. 

---
# PC-MNet: Dual-Level Congruity Modeling for Multimodal Sarcasm Detection via Polarity-Modulated Attention 

**Authors**: Maoheng Li, Ling Zhou, Xiaohua Huang, Rubing Huang, Wenming Zheng, Guoying Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.02447)  

**Abstract**: Multimodal sarcasm detection, which aims to precisely identify pragmatic incongruities between literal text and nonverbal cues, has gained substantial attention in multimodal understanding. Recent advancements have predominantly relied on na\"ıve similarity-based attention mechanisms and uniform late fusion this http URL, given that functional entanglement restricts traditional late fusions, we incorporate a scalar congruity routing mechanism and a prior-guided contextual graph. This mechanism anchors a generalized incongruity manifold through a two-stage asymmetric optimization driven by inconsistency-aware contrastive learning, selectively fusing only the most discriminative multi-granularity evidence. Extensive experiments on the \texttt{MUStARD} benchmark and its spurious-correlation-mitigated balanced datasets demonstrate that our approach achieves new state-of-the-art performance, surpassing the strongest multimodal baseline by a substantial 3.14\% improvement in Macro-F1. By architecturally isolating atomic, composition, and contextual conflicts. This work provides a robust, decoupled paradigm for modeling subtle pragmatic incongruities in human communication. 

---
# HalluScan: A Systematic Benchmark for Detecting and Mitigating Hallucinations in Instruction-Following LLMs 

**Authors**: Ahmed Cherif  

**Link**: [PDF](https://arxiv.org/pdf/2605.02443)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse natural language processing tasks, yet they remain susceptible to hallucinations -- generating content that is factually incorrect, unfaithful to provided context, or misaligned with user instructions. We present HalluScan, a comprehensive benchmark framework that systematically evaluates hallucination detection and mitigation across 72 configurations spanning 6 detection methods, 4 open-weight model families, and 3 diverse domains. We introduce three key contributions: (1) HalluScore, a novel composite metric that achieves a Pearson correlation of r = 0.41 with human expert judgments; (2) Adaptive Detection Routing (ADR), an intelligent routing algorithm achieving 2.0x cost reduction with only 0.1% AUROC degradation; and (3) systematic error cascade decomposition revealing substantial variation in hallucination error types across domains. Our experiments reveal that NLI Verification achieves the highest overall AUROC of 0.88, while RAV achieves the second-highest AUROC of 0.66. 

---
# Automatic Reflection Level Classification in Hungarian Student Essays 

**Authors**: Zsolt Csibi, Mónika Sándor, Mónika Serfőző, Kinga Gyöngy, Kristian Fenech  

**Link**: [PDF](https://arxiv.org/pdf/2605.02402)  

**Abstract**: Reflective thinking is a key competency in education, but assessing reflective writing remains a time-consuming and subjective task for education experts. While automated reflective analysis has been explored in several languages, Hungarian language was not researched extensively. In this paper, we present the first comprehensive study on automatic reflection level classification in Hungarian student essays. We used a large, expert-annotated Hungarian dataset consisting of 1,954 reflective essays collected over multiple academic years and labeled on a four-level reflection scale. We investigate two approaches: (1) classical machine learning models using TF-IDF and semantic embedding features, and (2) Hungarian-specific transformer models fine-tuned for document-level reflection classification. To address the strong class imbalance in the dataset, we systematically examine class weighting, oversampling, data augmentation, and alternative loss functions. An extensive ablation study is conducted to analyze the contribution of each modeling and balancing strategy. Our results show that shallow machine learning models with appropriate feature engineering achieve strong overall performance, reaching up to 71% overall score averaged over accuracy, F1-score, and ROC AUC metrics, while transformer-based models achieve slightly lower overall score (68%) averaged over the same metrics, but demonstrate better generalization on minority reflection classes. These findings highlight the continued relevance of classical methods for low-resource settings and the robustness of transformer models for imbalanced classification. The proposed dataset and experimental insights provide a solid foundation for future research on automated reflective analysis in Hungarian and other morphologically rich languages. 

---
# Is It Novel and Why? Fine-Grained Patent Novelty Prediction Based on Passage Retrieval 

**Authors**: Valentin Knappich, Anna Hätty, Simon Razniewski, Annemarie Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2605.02392)  

**Abstract**: Novelty assessment is a critical yet complex task in the examination process for patent acceptance, requiring examiners to determine whether an invention is disclosed in a prior art document. The process involves intricate matching between specific features of a patent claim and passages in the prior art. While prior work has approached novelty prediction primarily as a binary classification task at the claim level, we argue that this formulation is susceptible to spurious correlations and lacks the granularity required for practical application. In this work, we introduce FiNE-Patents (Fine-grained Novelty Examination of Patents), a novel dataset comprising 3,658 first patent claims annotated with fine-grained, feature-level prior art references extracted from European Search Opinion (ESOP) documents. We propose shifting the evaluation paradigm from simple binary classification to a joint retrieval and abstract reasoning task at the feature level, requiring models to identify specific passages from a prior art document that disclose individual claim features, and to identify which features of a claim make it novel. We implement and evaluate LLM-based workflows that decompose claims into features, analyze each feature against prior art, and finally derive a claim-level novelty prediction. Our experiments demonstrate that these workflows outperform embedding-based baselines on passage retrieval and novel feature identification. Furthermore, we show that unlike trained classifiers, LLMs are robust against spurious correlations present in the claim-level novelty classification task. We release the dataset and code to foster further research into transparent and granular patent analysis. 

---
# InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition 

**Authors**: Fengze Liu, Weidong Zhou, Binbin Liu, Ping Guo, Zijun Wang, Bingni Zhang, Yifan Zhang, Yifeng Yu, Xiaohuan Zhou, Taifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02364)  

**Abstract**: Upweighting high-quality data in LLM pretraining often improves performance, but in datalimited regimes, especially under overtraining, stronger upweighting increases repetition and can degrade performance. However, standard scaling laws do not reliably extrapolate across mixture recipes or under repetitions, making the selection for optimal data recipes at scaling underdetermined. To solve this, we introduce InfoLaw (Information Scaling Laws), a data-aware scaling framework that predicts loss from consumed tokens, model size, data mixture weights, and repetition. The key idea is to model pretraining as information accumulation, where quality controls information density and repetition induces scaledependent diminishing returns. We first collect the model performance after training on datasets that vary in scale, quality distribution, and repetition level. Then we build up the modeling for information so that information accurately predicts those model performance. InfoLaw predicts performance on unseen data recipes and larger scale runs (up to 7B, 425B tokens) with 0.15% mean and 0.96% max absolute error in loss, and it extrapolates reliably across overtraining levels, enabling efficient data-recipe selection under varying compute budgets. 

---
# When Correct Isn't Usable: Improving Structured Output Reliability in Small Language Models 

**Authors**: Cosimo Galeone, Minsu Park, Giuseppe Ettorre, Daniele Ligorio  

**Link**: [PDF](https://arxiv.org/pdf/2605.02363)  

**Abstract**: Deployed language models must produce outputs that are both correct and format-compliant. We study this structured-output reliability gap using two mathematical benchmarks -- GSM8K and MATH -- as a controlled testbed: ground truth is unambiguous and the output contract is strict (JSON with required fields). We evaluate three 7-9B models under five prompting strategies and report output accuracy -- the joint event of mathematical correctness and valid JSON structure -- as the primary metric. A systematic format failure emerges: NAIVE prompting (no system prompt) achieves up to 85% task accuracy on GSM8K but 0% output accuracy across all models and datasets. REFERENCE prompting (a minimal hand-written JSON format prompt) fares little better, yielding 0% output accuracy for two of four models tested. Constrained decoding enforces syntactic validity but incurs 3.6x-8.2x latency overhead and in several settings degrades task performance substantially. To overcome this limitation, we developed AloLab, an iterative system-prompt optimizer (meta-agent: Claude Sonnet 4.5) requiring only black-box API access to the target model; it reaches 84-87% output accuracy on GSM8K and 34-40% on MATH across five independent runs per model, with 29/30 paired McNemar comparisons against the best static prompt significant at p < 0.05, at near-NAIVE inference latency and without model fine-tuning. The same format failure extends to GPT-4o (OpenAI, 2024), a proprietary closed-source model: REFERENCE achieves 0% output accuracy due to systematic markdown-fence wrapping, while AloLab reaches 95.2% [94.8, 95.6]. An ablation replacing the Sonnet 4.5 meta-agent with Claude 3 Haiku reduces mean output accuracy to 61.0% and increases run-to-run standard deviation from <1 pp to 21.8 pp, confirming that meta-agent capability is a primary driver of optimization quality. 

---
# MolViBench: Evaluating LLMs on Molecular Vibe Coding 

**Authors**: Jiatong Li, Yuxuan Ren, Weida Wang, Changmeng Zheng, Xiao-yong Wei, Qing Li, Yatao Bian  

**Link**: [PDF](https://arxiv.org/pdf/2605.02351)  

**Abstract**: Molecular Vibe Coding, a paradigm where chemists interact with LLMs to generate executable programs for molecular tasks, has emerged as a flexible alternative to chemical agents with predefined tools, enabling chemists to express arbitrarily complex, customized workflows. Unlike general coding tasks, molecular coding imposes a distinctive challenge that LLMs should jointly equip programming, molecular understanding, and domain-specific reasoning capabilities. However, existing benchmarks remain disconnected. General code generation benchmarks such as HumanEval and SWE-bench require no chemistry knowledge, while chemistry-focused benchmarks such as S^2-Bench and ChemCoTBench evaluate knowledge recall or property prediction rather than executable code generation. To bridge this gap, we introduce MolViBench, the first benchmark tailored for Molecular Vibe Coding. MolViBench comprises 358 curated tasks across five cognitive levels, ranging from single-API recall to end-to-end virtual screening pipeline design, spanning 12 real-world drug discovery workflows. To rigorously assess generated code, we also propose a multi-layered evaluation framework that combines type-aware output comparison and AST-based API-semantic fallback analysis, which jointly measures executability and chemical correctness. We systematically evaluate 9 frontier coding LLMs and compare three real-world Molecular Vibe Coding paradigms, providing a practical and fine-grained testbed for diagnosing LLMs' coding capabilities in AI-accelerated molecular discovery. 

---
# Decoding-Time Debiasing via Process Reward Models: From Controlled Fill-in to Open-Ended Generation 

**Authors**: Muneeb Ur Raheem Khan  

**Link**: [PDF](https://arxiv.org/pdf/2605.02348)  

**Abstract**: Large language models pick up social biases from the data they are trained on and carry those biases into downstream applications, often reinforcing stereotypes around gender, race, religion, disability, age, and socioeconomic status. The standard fixes (retraining on curated data or fine-tuning with human feedback) are expensive, need access to model weights, and risk degrading the model on other tasks. In this paper we take a different route: we debias the model at decoding time, treating bias mitigation as a structured search over candidate tokens without ever touching model weights. A separate Process Reward Model (PRM) acts as a judge, scoring each candidate for both fairness and fluency. We design three schemes of increasing sophistication (Best-of-N selection, Sequential critique-and-revise, and Constitutional self-audit) and evaluate them on four models (GPT-4o-mini, Llama 3.2 3B, Gemma 3 4B, Qwen 2.5 3B) across a 200-prompt bilingual benchmark in English and Urdu covering eight bias categories. Sequential debiasing proves the most effective, raising mean bias scores by up to +0.40 over baseline while preserving (and sometimes improving) fluency. We then extend all three schemes to open-ended generation, where each token is debiased on the fly, and introduce a lightweight Bias Guard gate that fires only on potentially biased words, keeping overhead near 2x for well-calibrated models. A formal overhead metric that separates generator cost from judge cost reveals that Best-of-N is effectively free on the generator side in a native implementation. GPT-4o-mini, included as a strong proprietary anchor, confirms that the framework scales with model capability; the three open-weight models show where current small-scale LLMs still struggle. 

---
# Structural Dilemmas and Developmental Pathways of Legal Argument Mining in the Era of Artificial Intelligence 

**Authors**: Xianglei Liao, Chuanyi Li, Kun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02308)  

**Abstract**: Against the backdrop of rapid advances in artificial intelligence, legal argument mining has emerged as an important research area linking legal texts with intelligent analysis, carrying significant theoretical and practical implications. Existing studies have primarily developed along three dimensions: data, technology, and theory. At the data level, raw legal texts and annotated corpora constitute the foundational resources. At the technological level, research paradigms have evolved from rule-based systems and traditional machine learning to large language models (LLMs). At the theoretical level, argumentation theory and legal dogmatics provide important references for modeling argumentation structures. However, despite ongoing progress, the overall development of legal argument mining remains relatively slow. Building on a systematic review of existing research, this study conducts an in-depth analysis and finds that this is due not only to data scarcity or technical limitations, but more fundamentally to the lack of a structured representational approach that reconciles theoretical expressiveness with computational feasibility. Specifically, this challenge manifests in dilemmas in data standardization, obstacles to effective modeling, and limitations in domain adaptation. In response, the study proposes several key directions for future research. It aims to provide a reframing of key problems and a pathway for future development in legal argument mining, while leaving specific models and implementation schemes for further investigation. 

---
# Compositional Multi-hop Factual Error Correction via Decomposition-and-Injection 

**Authors**: Lei Zhu, Xiaobao Wang, Jianbiao Yang, Chenyang Wang, Dongxiao He, Longbiao Wang, Jianwu Dang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02277)  

**Abstract**: Factual Error Correction (FEC) aims to revise inaccurate text into statements that are factually consistent with external evidence. Although recent methods perform well on single-hop correction, they often treat claims as atomic units and struggle with multi-hop cases that require compositional reasoning across multiple evidence sources. This challenge is further amplified by limited paired data and difficulties in locating semantic errors within complex reasoning chains. We present CECoR (Compositional Error Correction via Reasoning-aware Synthesis), a reasoning-aware framework that introduces a Decomposition and Injection paradigm for compositional error correction. CECoR decomposes multi-hop claims into interpretable reasoning steps and injects controlled perturbations to synthesize high-quality training pairs. A two-stage learning strategy combining supervised fine-tuning and reinforcement learning improves factual accuracy and robustness. Comprehensive evaluations show that CECoR achieves strong performance on multi-hop benchmarks, outperforming both distantly supervised methods and few-shot LLM baselines. It also generalizes effectively to single-hop correction and remains stable under noisy evidence, demonstrating its versatility for real-world factual correction. 

---
# A Systematic Benchmark of Machine Transliteration Models for the Tajik-Farsi Language Pair: A Comparative Study from Rule-Based to Transformer Architectures 

**Authors**: Mullosharaf K. Arabov  

**Link**: [PDF](https://arxiv.org/pdf/2605.02270)  

**Abstract**: This paper presents the first comprehensive comparative analysis of modern machine learning architectures for transliteration between Tajik (Cyrillic script) and Persian (Arabic script). A key contribution is the creation and validation of a unique parallel corpus aggregated from multiple heterogeneous sources, including crowdsourced projects, lexicographic pairs, parallel texts of "Shahnameh", diplomatic articles, texts of "Masnavi-i Ma'navi", official terminology lists, and transliterated correspondences. The initial dataset comprised 328,253 sentence pairs; a representative subset of 40,000 pairs was formed using stratified random sampling.
The experiment compared six classes of models: rule-based baseline, LSTM with attention, character-level Transformer, G2P Transformer (trained from scratch), pre-trained multilingual models (mBART, mT5 with LoRA), and byte-level ByT5. Results demonstrate the overwhelming superiority of ByT5 (chrF++ 87.4 for Tajik to Farsi, 80.1 for reverse). The G2P Transformer significantly outperformed mBART (72.3 vs. 62.2 chrF++) despite limited data. Models using subword tokenization (mT5) failed completely (chrF++ less than 18.5).
The findings demonstrate that for accurate transliteration of the Tajik-Farsi pair, architectures operating at the byte or character level are unequivocally more effective than traditional multilingual Seq2Seq models relying on subword tokenization. 

---
# Reliability-Oriented Multilingual Orthopedic Diagnosis: A Domain-Adaptive Modeling and a Conceptual Validation Framework 

**Authors**: Danish Ali, Li Xiaojian, Sundas Iqbal, Farrukh Zaidi  

**Link**: [PDF](https://arxiv.org/pdf/2605.02266)  

**Abstract**: Large Language Models (LLMs) are increasingly proposed for clinical decision support including multilingual diagnosis in low-resource settings. However, their reliability, calibration and safety characteristics remain insufficiently understood for structured, high-risk tasks. We present a system-level analysis of multilingual orthopedic diagnosis from free-text clinical notes in English, Hindi and Punjabi. We evaluate three modeling regimes: (i) task-aligned multilingual transformer encoders, (ii) a task-fine-tuned baseline (DistilBERT), and (iii) a domain-adaptive architecture tailored to orthopedic text (IndicBERT-HPA). These models are compared with zero-shot, instruction-tuned LLMs to assess suitability for structured diagnostic classification. Results indicate that while LLMs exhibit strong linguistic fluency, they show unstable calibration and reduced reliability under structured multilingual conditions, particularly in low-resource languages. These findings are specific to zero-shot evaluation and do not imply limitations of fine-tuned models. Domain-adaptive specialization substantially improves cross-lingual discrimination and confidence behavior. IndicBERT-HPA, with language-specific orthopedic adapter heads achieves consistently strong performance across six diagnostic categories and more predictable deployment characteristics than task-only adaptation. Building on these observations, we outline a conceptual deterministic agent-based validation framework for future implementation, formalizing evidence checks, language-sensitive validation and conservative human-in-the-loop gating. Reliable multilingual clinical decision support requires specialized architecture, explicit reliability analysis, and structured validation for safety-critical systems. 

---
# An Information-theoretic Propagation Denoising and Fusion Framework for Fake News Detection 

**Authors**: Mengyang Chen, Lingwei Wei, Wei Zhou, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02259)  

**Abstract**: Incomplete propagation data significantly hinders robust fake news detection. Recent approaches leverage large language models to simulate missing user interactions via role-playing, thereby enriching propagation with synthetic signals. However, such propagation data is intrinsically unreliable, and directly fusing it can lead to biased representations, leading to limited detection performance. In this paper, we alleviate the unreliability of synthetic propagation from the mutual information perspective and propose a novel information-theoretic propagation denoising and fusion (InfoPDF) framework to learn effective representations from both real and synthetic propagation. Specifically, we first generate attribute-specific synthetic propagation using large language models. Then we model each synthetic propagation graph as a probabilistic latent distribution to guide reliability-aware adaptive fusion with real propagation. During training, we design a mutual information-based objective to learn compressed and task-sufficient propagation representations. It jointly suppresses noisy signals across attribute-specific synthetic propagation, maintains consistency between real and synthetic propagation representations, and ensures task sufficiency for fake news detection and attribute prediction. Experiments on three real-world datasets show that InfoPDF consistently achieves superior performance across various fake news detection tasks. Further analysis demonstrates that InfoPDF can estimate attribute-level reliabilities and learn more discriminative propagation representations. 

---
# ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement with Adversarial Umpiring 

**Authors**: Deyi Ji, Junyu Lu, Xuanyi Liu, Liqun Liu, Hailong Zhang, Peng Shu, Huan Yu, Jie Jiang, Tianru Chen, Lanyun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02200)  

**Abstract**: Online advertising governance faces significant challenges due to the non-stationary nature of regulatory policies, where emerging mandates (e.g., restrictions on education or aesthetic anxiety) create severe label inconsistencies and reasoning ambiguities in historical datasets. In this paper, we propose ARGUS, a policy-adaptive governance system that enables evolving reinforcement through multi-agent adversarial umpiring. ARGUS addresses the sparsity of new policy data by employing a three-stage framework: (1) Policy Seeding for initial perception; (2) Adversarial Label Rectification, which utilizes a ``Prosecutor-Defender-Umpire'' architecture to resolve conflicts between stale labels and new mandates; and (3) Latent Knowledge Discovery, which employs a tripartite dialectical discussion to unearth sophisticated, ``gray-area'' violations. By leveraging RAG-enhanced policy knowledge and Chain-of-Thought synthesis as dynamic rewards for reinforcement learning, ARGUS synchronizes its reasoning pathways with evolving regulations. Extensive experiments on both industrial and public datasets demonstrate that ARGUS significantly outperforms traditional fine-tuning baselines, achieving superior policy-adaptive learning with minimal gold data. 

---
# CLaC at SemEval-2026 Task 6: Response Clarity Detection in Political Discourse 

**Authors**: Nawar Turk, Lucas Miquet-Westphal, Leila Kosseim  

**Link**: [PDF](https://arxiv.org/pdf/2605.02170)  

**Abstract**: In this paper, we present our system for SemEval-2026 Task 6 (CLARITY) on response clarity and evasion detection in question-answer pairs from U.S. presidential interviews, comparing fine-tuned encoders with prompt-based LLMs. Our LLM ensemble achieves 80 macro-F1 on the 3-class Task 1 (9th/41) and 59 on the 9-class Task 2 (3rd/33). Across 8 transformer encoders optimized through a four-stage pipeline, partial encoder layer unfreezing outperforms full fine-tuning by a wide margin. Combining English and multilingual encoders further improves ensemble performance over either family alone, despite multilingual models being individually weaker. Prompt-based LLMs, without any task-specific parameter updates, outperform fine-tuned encoders, particularly on minority classes; among open-weight LLMs, parameter count does not predict performance. Enriched input, concatenating the full interviewer turn, improves LLM performance but not that of encoders, an effect that persists with Longformer's extended context window, suggesting the divergence is not attributable to sequence-length capacity alone in our settings. The Clear Reply/Ambivalent boundary remains the dominant failure mode, mirroring the disagreement among human annotators. Our code, prompts, model configurations, and results are publicly available. 

---
# EditPropBench: Measuring Factual Edit Propagation in Scientific Manuscripts 

**Authors**: Garvin Kruthof  

**Link**: [PDF](https://arxiv.org/pdf/2605.02083)  

**Abstract**: Local factual edits in scientific manuscripts often create non-local revision obligations. If a dataset changes from 215 to 80 documents, claims such as 'medium-scale' or 'a few hundred items' may also become stale, even though they do not repeat the edited number. We introduce EditPropBench, a benchmark for measuring whether LLM editors propagate factual edits through dependent manuscript claims. Each item contains an ML/NLP-style synthetic manuscript, a targeted edit, and a controlled fact graph with sentence-level labels for direct targets, required downstream updates, and protected unrelated text. EditPropBench provides a controlled manuscript-level benchmark with sentence-level dependency supervision, three editing protocols, adversarial metric probes, stress-test variants, and a metric suite centered on Edit-Ripple Adherence (ERA). On the hard implicit/free-form stratum, five LLM editing systems span ERA 0.148--0.705; even the strongest misses roughly 30% of required cascade updates. A mixed-stratum stress test shows that LLMs retain a positive advantage over deterministic substitution baselines when easy substitution-solvable cases are included. Finally, an audit of recent arXiv cs.CL benchmark and dataset papers finds fact-dependent qualitative claims in 37.2% of papers. EditPropBench shows that current LLM editors can repair many implicit consequences of factual edits, but reliable scientific revision still requires cascade-aware checking. 

---
# Enhanced LLM Reasoning by Optimizing Reward Functions with Search-Driven Reinforcement Learning 

**Authors**: Arash Ahmadi, Sarah Sharif, Yaser, Banad  

**Link**: [PDF](https://arxiv.org/pdf/2605.02073)  

**Abstract**: Mathematical reasoning is a key benchmark for large language models. Reinforcement learning is a standard post-training mechanism for improving the reasoning capabilities of large language models, yet performance remains sensitive to the design of the reward function that drives policy optimization. This paper introduces a search-driven framework that treats the reward specification itself as an object of optimization. The setting of interest is one in which the base model is held fixed and the reward specification is the primary remaining design lever. Candidate reward functions are generated by a frontier language model, validated automatically, screened through 500-step Group Relative Policy Optimization (GRPO) training runs on a Llama-3.2-3B-Instruct base model with Low-Rank Adaptation (LoRA), and ranked by F1 on the GSM8K test set. Ranked summaries from prior rounds are then fed back into the next round of generation. Over five rounds, the search produces 50 candidate rewards. The mean F1 rises from 0.596 in Round 1 to 0.632 in Round 5, and the top individual reward reaches F1 = 0.787. Seven ensemble configurations of top-ranked rewards are evaluated. The best ensemble achieves F1 = 0.795 (95% bootstrap CI [0.756, 0.832]) and accuracy 0.660 [0.635, 0.686], a 0.19 absolute F1 gain over a base-rewards-only GRPO baseline (F1 = 0.609). Pairwise McNemar tests with Bonferroni correction show all five-or-more-reward configurations are statistically indistinguishable at {\alpha} = 0.05/21. A three-seed re-training of the best ensemble yields F1 of 0.785. A randomly drawn 5-reward control collapses to F1 = 0.047, which shows that the ranked-feedback loop, not the additive signal of having more rewards, drives the gain. 

---
# Pair2Score: Pairwise-to-Absolute Transfer for LLM-Based Essay Scoring 

**Authors**: İbrahim Rıza Hallaç, Hasan Oğul  

**Link**: [PDF](https://arxiv.org/pdf/2605.02069)  

**Abstract**: Many scoring applications require absolute predictions, while pairwise comparisons can provide a simpler learning objective. We present Pair2Score, a two-stage learning framework that transfers pairwise comparisons into absolute scoring with parameter-efficient LLaMA adaptation. Stage 1 trains a directional Siamese ranker on pairwise comparisons derived from absolute trait labels; Stage 2 trains an absolute predictor using configurable transfer strategies (warm-start and embedding-fusion variants). We evaluate on rubric-aligned Automated Essay Scoring (AES) traits (grammar, vocabulary, syntax) under a five-fold protocol that co-rotates held-out fold and random seed. At the trait level, the best-performing transfer variant improves quadratic weighted kappa (QWK) over an absolute-only baseline for all three traits. However, not all transfer configurations help: a one-epoch pairwise stage transfers more reliably than extended pairwise training, and transfer configuration -- not just the inclusion of a pairwise stage -- determines whether downstream scoring benefits. 

---
# Methods, Data, and Conceptual Change: Reflections from Two Quantitative Diachronic Case Studies 

**Authors**: Catherine Wong, Bach Phan-Tat, Susan Fitzmaurice  

**Link**: [PDF](https://arxiv.org/pdf/2605.02052)  

**Abstract**: This discussion paper reflects on how quantitative approaches to historical linguistics interact with dataset properties. Drawing on two worked examples, we examine English data using quad-based concept modelling of Early Modern English discourse in EEBO-TCP (c. 1470s-1690s; 765M words) alongside SynFlow analysis of scientific writing in Royal Society Corpus 6.0.4 (1750-1799; drawn from a 78.6M-token open corpus). Through parallel comparison, the paper explores how each approach operationalises concepts, the data assumptions they entail, and the diachronic interpretations they support. We argue that comparative methodological reflection clarifies the limits of purely lexical, frequency-based approaches and highlights how dataset structure shapes the kinds of semantic change that quantitative methods can reliably detect. 

---
# What Single-Prompt Accuracy Misses: A Multi-Variant Reliability Audit of Language Models 

**Authors**: Ranit Karmakar, Jayita Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2605.02038)  

**Abstract**: Single-prompt accuracy is the dominant way to benchmark language models, but it can miss reliability failures that matter. We evaluate a 15-model open-weight corpus, with the main reliability analyses focused on 10 instruct models across five classification and reasoning benchmarks under five prompt variants each, measuring accuracy, token-probability calibration, verbal-confidence calibration, verbal parse rate, and prompt-perturbation spread for every (model x dataset x variant) cell. We find three broad results. First, evaluation design can materially change the conclusion. Switching Expected Calibration Error (ECE) token from a raw to a label-set-normalised definition changes per-cell calibration by a mean absolute 0.149. More strikingly, pairing a chain-of-thought prompt with a first-character evaluator on ARC-Challenge reduces apparent accuracy by 72-88% across all five primary models; two independent repair procedures recover 93.8% and 102.7% of the lost performance, indicating an evaluator-side rather than model-side failure. Second, confidence signals are fragile. On MMLU-Pro, every primary model verbally reports confidence substantially above both its accuracy and its token-probability confidence on the same rows, and verbal parse rate can collapse for a single model on a single prompt variant. Third, prompt robustness does not track parameter count reliably. Across 10 instruct models, the correlation between model size and prompt-perturbation spread ranges from -0.244 to 0.474 across benchmarks. Taken together, these results show that reliability conclusions for small language models depend not only on the model being evaluated, but also on the evaluation pipeline used to measure it. We argue that calibration definitions, evaluator logic, verbal parseability, and prompt robustness should be reported explicitly when making reliability claims. 

---
# A Multimodal Dataset for Visually Grounded Ambiguity in Machine Translation 

**Authors**: Jingheng Pan, Xintong Wang, Longyue Wang, Liang Ding, Weihua Luo, Chris Biemann  

**Link**: [PDF](https://arxiv.org/pdf/2605.02035)  

**Abstract**: Ambiguity resolution is a key challenge in multimodal machine translation (MMT), where models must genuinely leverage visual input to map an ambiguous expression to its intended meaning. Although prior work has proposed disambiguation-oriented benchmarks that provide supportive evidence for the role of vision, we observe substantial issues in data quality and a mismatch with translation scenarios. Moreover, existing ambiguity-oriented evaluations are not well suited to broader ambiguity types in open-ended translation. To address these limitations, we present VIDA (Visually-Dependent Ambiguity), a dataset of 2,500 carefully curated instances in which resolving an annotated ambiguous source span requires visual evidence. We further propose Disambiguation-Centric Metrics that use an LLM-as-a-judge classifier to verify whether annotated ambiguous expressions are resolved correctly at the span level. Experiments with two state-of-the-art Large Vision Language Models under vanilla inference, supervised fine-tuning (SFT), and our chain-of-thought SFT (CoT-SFT) show that while SFT improves overall translation quality, CoT-SFT yields more consistent gains in disambiguation accuracy, especially on out-of-distribution subsets, indicating a stronger generalization for resolving diverse ambiguity types. 

---
# Counting as a minimal probe of language model reliability 

**Authors**: Tianxiang Dai, Jonathan Fan  

**Link**: [PDF](https://arxiv.org/pdf/2605.02028)  

**Abstract**: Large language models perform strongly on benchmarks in mathematical reasoning, coding and document analysis, suggesting a broad ability to follow instructions. However, it remains unclear whether such success reflects general logical competence, repeated application of learned procedures, or pattern matching that mimics rule execution. We investigate this question by introducing Stable Counting Capacity, an assay in which models count repeated symbols until failure. The assay removes knowledge dependencies, semantics and ambiguity from evaluation, avoids lexical and tokenization confounds, and provides a direct measure of procedural reliability beyond standard knowledge-based benchmarks. Here we show, across more than 100 model variants, that stable counting capacity remains far below advertised context limits. Model behavior is consistent neither with open-ended logic nor with stable application of a learned rule, but instead with use of a finite set of count-like internal states, analogous to counting on fingers. Once this resource is exhausted, the appearance of rule following disappears and exact execution collapses into guessing, even with additional test-time compute. These findings show that fluent performance in current language models does not guarantee general, reliable rule following. 

---
# Enhancing Judgment Document Generation via Agentic Legal Information Collection and Rubric-Guided Optimization 

**Authors**: Weihang Su, Xuanyi Chen, Yueyue Wu, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02011)  

**Abstract**: Automating the drafting of judgment documents is pivotal to judicial efficiency, yet it remains challenging due to the dual requirements of comprehensive retrieval of legal information and rigorous logical reasoning. Existing approaches, typically relying on standard Retrieval-Augmented Generation and Supervised Fine-Tuning, often suffer from insufficient evidence recall, hallucinated statutory references, and logically flawed legal reasoning. To bridge this gap, we propose Judge-R1, a unified framework designed to enhance LLM-based judgment document generation by jointly improving legal information collection and judgment document generation. First, we introduce Agentic Legal Information Collection, which employs a dynamic planning agent to retrieve precise statutes and precedents from multiple sources. Second, we implement Rubric-Guided Optimization, a reinforcement learning phase utilizing Group Relative Policy Optimization (GRPO) with a comprehensive legal reward function to enforce adherence to judicial standards and reasoning logic. Extensive experiments on the JuDGE benchmark demonstrate that Judge-R1 significantly outperforms state-of-the-art baselines in both legal accuracy and generation quality. 

---
# Learn-to-learn on Arbitrary Textual Conditioning: A Hypernetwork-Driven Meta-Gated LLM 

**Authors**: Luo Ji, Qi Qin, Ningyuan Xi, Teng Chen, Qingqing Gu, Hongyan Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.01973)  

**Abstract**: Conventional LLMs may suffer from corpus heterogeneity and subtle condition changes. While finetuning can create the catastrophe forgetting issue, application of meta-learning on LLMs is also limited due to its complexity and scalability. In this paper, we activate the meta-signal of $\beta$ within the SwiGLU blocks, resulting in a meta-gating mechanism that adaptively adjusts the nonlinearity of FFN. A hypernetwork is employed which dynamically produces $\beta$ on textual conditions, providing meta-controllability on LLMs. By testing on different condition types such as task, domain, persona, and style, our method outperforms finetuning and meta-learning baselines, and can generalize reasonably on unseen tasks, condition types, or instructions. Our code can be found in this https URL. 

---
# StressEval: Failure-Driven Dynamic Benchmarking for Knowledge-Intensive Reasoning in Large Language Models 

**Authors**: Yongrui Chen, Yangyang Ma, Xiaoying Huang, Shenyu Zhang, Huajun Chen, Haofen Wang, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2605.01939)  

**Abstract**: Static benchmarks for LLMs are increasingly compromised by contamination and overfitting especially on knowledge intensive reasoning tasks While recent dynamic benchmarks can alleviate staleness they often increase difficulty at the expense of answerability and controllability In this paper we propose StressEval a failure driven data synthesis framework that turns observed model failures into dynamic challenging and controllable test instances StressEval consists of three stages first it constructs a semi structured difficulty card that identifies the failed reasoning step and its root cause second it applies a dual perspective instance synthesis method that targets both knowledge gaps and reasoning breakdowns while preserving the underlying difficulty factors and third it applies a gating mechanism to retain only grounded unambiguous instances Seeding from multiple knowledge intensive reasoning datasets we employ StressEval to build Dynamic OneEval a focused suite of challenging dynamic benchmark Across several state of the art LLMs Dynamic OneEval yields substantially larger performance drops than the original benchmarks while retaining explicit difficulty factors enabling more actionable iteration 

---
# Maistros: A Greek Large Language Model Adapted Through Knowledge Distillation From Large Reasoning Models 

**Authors**: Nikolaos Giarelis, Charalampos Mastrokostas, Nikos Karacapilidis  

**Link**: [PDF](https://arxiv.org/pdf/2605.01870)  

**Abstract**: Large Language Models (LLMs) have substantially advanced the field of Natural Language Processing (NLP), achieving state-of-the-art performance across a wide range of tasks. These improvements have been attributed, in part, to their emerging reasoning capabilities, which are enabled by large-scale training and increased model capacity. However, existing LLMs can generate erroneous responses when addressing complex queries that fall outside their training distribution, due to limited internal knowledge or the need for multi-step reasoning. To address these limitations, recent work has introduced large reasoning models (LRMs), which incorporate explicit internal reasoning processes to improve response accuracy. Additionally, state-of-the-art LRMs often comprise hundreds of billions of parameters and require several seconds per inference, even on advanced multi-GPU systems. These characteristics limit their practicality for deployment in conventional computing environments. Meanwhile, NLP research on multilingual LLMs continues to prioritize high-resource languages. However, these models exhibit limited performance in under-resourced languages, primarily due to insufficient language- and culture-specific training data. In this paper, we focus on Modern Greek, for which only a limited number of question answering (QA) datasets have been proposed, most of which are intended for model evaluation. To address this research gap in Greek QA, we make the following contributions: (i) CulturaQA, a high-quality LRM-generated and human-curated dataset, for Greek LLM training and evaluation; (ii) a memory-efficient LLM evaluation framework adaptable to diverse languages and QA tasks; (iii) Maistros 8B, a state-of-the-art open-weights Greek LLM developed via knowledge distillation and fine-tuning on CulturaQA; and (iv) a comprehensive evaluation of nine LLMs across nine human-curated Greek QA datasets. 

---
# Spatiotemporal Hidden-State Dynamics as a Signature of Internal Reasoning in Large Language Models 

**Authors**: Kotaro Furuya, Takahito Tanimura  

**Link**: [PDF](https://arxiv.org/pdf/2605.01853)  

**Abstract**: Large reasoning models (LRMs) generate extended solutions, yet it remains unclear whether these traces reflect substantive internal computation or merely verbosity and overthinking. Although recent hidden-state analyses suggest that internal representations carry correctness-related signals, their coarse aggregations may obscure the token and layer structure underlying reasoning computation. We investigate hidden-state transitions across decoding steps and layers, and identify a distinct spatiotemporal pattern in LRMs: successful trajectories exhibit broad temporal dynamics with localized layer-wise concentration, while this structure is weaker in non-reasoning models and knowledge-heavy domains. We formalize this characteristic as Spatiotemporal Amplitude of Latent Transition (StALT), a training-free trajectory statistic that summarizes temporal changes between adjacent tokens weighted by within-token layer saliency. Across diverse models and benchmarks, StALT reliably separates correct from incorrect trajectories in reasoning-intensive regimes, providing a competitive label-free correctness signal alongside strong output-space and length-based baselines. Intervention analyses further show that this spatiotemporal amplitude responds systematically to manipulations that increase or reduce the demand for internal reasoning, supporting its association with latent reasoning dynamics in LRMs. These findings provide empirical evidence that LRMs exhibit measurable hidden-state dynamics and offer a practical probe for understanding internal computation beyond output-based evaluation. 

---
# Do Large Language Models Plan Answer Positions? Position Bias in Multiple-Choice Question Generation 

**Authors**: Xuemei Tang, Xufeng Duan, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2605.01846)  

**Abstract**: Large language models (LLMs) are increasingly used to generate multiple-choice questions (MCQs), where correct answers should ideally be uniformly distributed across options. However, we observe that LLMs exhibit systematic position biases during generation. Through extensive experiments with 10 LLMs and 5 vision-language models (VLMs) on three MCQ generation tasks, we show that these biases are structured, with similar patterns emerging within model families. To investigate the underlying mechanisms, we conduct probing experiments and find that hidden representations in the question stem encode predictive signals of the correct answer position, suggesting that answer position may be implicitly planned during generation. Building on this insight, we apply activation steering to manipulate internal representations and influence answer position. Our results show that steering can partially control positional preferences and substantially shift answer position distributions. Our findings provide a practical framework for studying implicit positional planning in LLMs and highlight the importance of controllable generation for reliable MCQ construction and evaluation. 

---
# The Cylindrical Representation Hypothesis for Language Model Steering 

**Authors**: Lang Gao, Jinghui Zhang, Wei Liu, Fengxian Ji, Chenxi Wang, Zirui Song, Akash Ghosh, Youssef Mohamed, Preslav Nakov, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.01844)  

**Abstract**: Steering is a widely used technique for controlling large language models, yet its effects are often unstable and hard to predict. Existing theoretical accounts are largely based on the Linear Representation Hypothesis (LRH). While LRH assumes that concepts can be orthogonalized for lossless control, this idealized mapping fails in real representations and cannot account for the observed unpredictability of steering. By relaxing LRH's orthogonality assumption while preserving linear representations, we show that overlapping concept contributions naturally yield a sample-specific axis-orthogonal structure. We formalize this as the Cylindrical Representation Hypothesis (CRH). In CRH, a central axis captures the main difference between concept absence and presence and drives concept generation. A surrounding normal plane controls steering sensitivity by determining how easily the axis can activate the target concept. Within this plane, only specific sensitive sectors strongly facilitate concept activation, while other sectors can suppress or delay it. While the surrounding normal plane can be reliably identified from difference vectors, the sensitive sector cannot, introducing intrinsic uncertainty at the sector level. This uncertainty provides a principled explanation for why steering outcomes often fluctuate even when using well-aligned directions. Our experiments verify the existence of the cylindrical structure and demonstrate that CRH provides a valid and practical way to interpret model steering behavior in real settings: this https URL. 

---
# RMGAP: Benchmarking the Generalization of Reward Models across Diverse Preferences 

**Authors**: Yangyang Zhou, Yi-Chen Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.01831)  

**Abstract**: Reinforcement Learning from Human Feedback has become the standard paradigm for language model alignment, where reward models directly determine alignment effectiveness. In this work, we focus on how to evaluate the generalizability of reward models. By "generalizability", we mean the ability of RMs to correctly rank responses to align with diverse user preferences. However, existing reward model benchmarks are typically designed around a universal preference, failing to assess this generalization. To address this critical gap, we introduce RMGAP, a benchmark comprising 1,097 instances across Chat, Writing, Reasoning, and Safety domains. Since different users exhibit diverse preferences for the same task, we first generate four distinct responses with different linguistic profiles for each collected prompt. However, the original prompt set lacks the specificity to convey different preferences. We therefore construct tailored prompts by contrasting these candidates and designing scenarios in which one response becomes the uniquely appropriate choice. Moreover, we observe that users often express the same preference using different phrasings, and thus extend each prompt with two paraphrased variants. Our evaluation of 24 state-of-the-art RMs reveals their substantial limitations: even the best RM achieves only 49.27% Best-of-N accuracy, highlighting considerable room for improvement in reward model generalization. Related data and code are available at this https URL. 

---
# The Compliance Gap: Why AI Systems Promise to Follow Process Instructions but Don't 

**Authors**: Kwan Soo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2605.01771)  

**Abstract**: An auditor instructs an AI assistant: "open each file individually using the Read tool -- no scripts, no agents." The AI replies "Yes" -- then issues a single batched call summarizing all fifty files at once. We call this the Compliance Gap: a third, orthogonal axis of AI honesty distinct from factual truthfulness and rhetorical substance. Three questions: does this verbal-behavioral disconnect exist (existence); can any text-only observer recover it (detectability); what infrastructure does AI deployment need (remedy)? Some 75 benchmarks (IFEval, SWE-bench, BFCL, COMPASS, SpecEval) measure outcome fidelity; none measures process fidelity. Theorem 1 shows the gap is structurally inevitable under RL that rewards text without observing behavior. Theorem 2, via the Data Processing Inequality, shows it is undetectable from text alone -- by any human or LLM observer, present or future. Thirteen experiments and 2,031 sessions on six frontier models confirm both predictions. Under default framing, all six exhibit instruction compliance rates of 0% -- Claude Sonnet 4 verbally agrees ten out of ten times then bypasses in all ten. The gap is selective: 97% compliance where rationale is rewarded (audit trails), 0-4% where it is not (file reading, privacy masking); removing delegation tools raises compliance to 75% (Cohen's d = 2.47), confirming environmental affordance rather than weight-encoded failure. Nine blinded human raters achieve Fleiss' kappa = 0.130 and correctly identify zero of fifteen compliant sessions, exactly as Theorem 2 predicts. Where humans show 47% intention-behavior gaps in psychology and 96.5pp gaps in surgical audits, RLHF-trained models approach 100% under default conditions -- a regime warranting its own measurement infrastructure. We release BS-Bench: the first open benchmark for process compliance, with seven tool-call-log audit metrics and a public leaderboard. 

---
# Only Say What You Know: Calibration-Aware Generation for Long-Form Factuality 

**Authors**: Wen Luo, Guangyue Peng, Liang Wang, Nan Yang, Wei Li, Yuhan Song, Shaohang Wei, Feifan Song, Furu Wei, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01749)  

**Abstract**: Large Reasoning Models achieve strong performance on complex tasks but remain prone to hallucinations, particularly in long-form generation where errors compound across reasoning steps. Existing approaches to improving factuality, including abstention and factuality-driven optimization, follow a \emph{coupled exploration-commitment} paradigm, in which intermediate reasoning is unconditionally propagated to the final output, limiting fine-grained control over information selection and integration. In this paper, we propose an \textbf{Exploration-Commitment Decoupling} paradigm that disentangles knowledge exploration from final commitment, enabling models to explore with awareness while answering cautiously. We instantiate the paradigm with \textbf{Calibration-Aware Generation (CAG)}, a framework that equips models with end-to-end, calibration-aware generation capabilities, by augmenting intermediate reasoning with calibrated reliability estimates and prioritizing reliable content in final outputs. Across five long-form factuality benchmarks and multiple model families, CAG improves factuality by up to 13%, while reducing decoding time by up to 37%. Overall, our work highlights decoupling as a principled approach for more reliable long-form generation, offering directions for trustworthy and self-aware generative systems. 

---
# Less is More: Geometric Unlearning for LLMs with Minimal Data Disclosure 

**Authors**: Chenchen Tan, Xinghao Li, Shujie Cui, Youyang Qu, Cunjian Chen, Longxiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.01735)  

**Abstract**: As large language models (LLMs) are increasingly deployed in real-world systems, they must support post-hoc removal of specific content to meet privacy and governance requirements. This motivates selective unlearning, which suppresses information about a particular entity or topic while preserving the LLM's general utility. However, most existing LLM unlearning methods require access to the original training corpus and rely on output-level refusal tuning or broad gradient updates, creating a tension among unlearning strength, non-target preservation, and data availability. We propose Geometric Unlearning (GU), an approach that operates directly on the model's prompt-time planning states without access to the original training corpus. GU distills a compact, low-rank geometry of desired safe behavior from a small set of safe reference prompts, and uses lightweight anchor-in-context synthetic prompts to trigger localized, projection-based alignment of hidden planning representations to this safe geometry. A teacher-distillation regularizer on synthetic non-target anchors further reduces collateral drift. Across privacy-oriented unlearning benchmarks (ToFU and UnlearnPII), GU achieves strong target suppression with minimal impact on non-target performance, demonstrating that effective unlearning can be achieved with minimal synthetic data. 

---
# EGAD: Entropy-Guided Adaptive Distillation for Token-Level Knowledge Transfer 

**Authors**: Hao Zhang, Zhibin Zhang, Guangxin Wu, Wanyi Ning, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.01732)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance across diverse domains, yet their enormous computational and memory requirements hinder deployment in resource-constrained environments. Knowledge distillation offers a promising solution by transferring knowledge from a large teacher model to a smaller student model. However, existing distillation methods typically treat all tokens equally, ignoring the fact that different tokens contribute unequally to model decisions. This can lead to inefficient knowledge transfer and reduced learning effectiveness. To address this limitation, we propose an entropy-based adaptive distillation strategy that dynamically adjusts the training process at the token level. Our method leverages the teacher's output entropy to guide three aspects of distillation. Specifically, we introduce a token-level curriculum by dynamically shifting focus from low- to high-entropy tokens during training. We further adjust the distillation temperature based on token entropy to better capture teacher confidence patterns. Moreover, we employ a dual-branch architecture for efficient logits-only distillation on easy tokens and deeper feature-based distillation on difficult tokens. Extensive experiments validate the soundness and effectiveness of our method. 

---
# TCDA: Thread-Constrained Discourse-Aware Modeling for Conversational Sentiment Quadruple Analysis 

**Authors**: Xinran Li, Xinze Che, Yifan Lyu, Zhiqi Huang, Xiujuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01717)  

**Abstract**: Conversational Aspect-based Sentiment Quadruple Analysis (DiaASQ) needs to capture the complex interrelationships in multiple rounds of dialogues. Existing methods usually employ simple Graph Convolutional Networks (GCN), which introduce structural noise and fail to consider the temporal sequence of the dialogues, or use standard RoPE, which implicitly captures relative distances in a flat sequence but cannot clearly separate the token-level syntactic order from the utterance-level progression, and may suffer from the Distance Dilution problem. To address these issues, we propose a new framework that combines Thread-Constrained Directed Acyclic Graph (TC-DAG) and Discourse-Aware Rotary Position Embedding (D-RoPE). Specifically, TC-DAG filters out cross-thread noise based on thread constraints, maintains global connectivity through root anchoring, and incorporates the temporal sequence of the dialogues. D-RoPE aligns multi-layer semantics using dual-stream projection and multi-scale frequency signals, captures thread dependencies using tree-like distances, and alleviates the token-level Distance Dilution problem by incorporating utterance-level progressions. Experimental results on two benchmark datasets demonstrate that our framework achieves state-of-the-art performance. 

---
# The Reasoning Trap: An Information-Theoretic Bound on Closed-System Multi-Step LLM Reasoning 

**Authors**: Kwan Soo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2605.01704)  

**Abstract**: When copies of the same language model are prompted to debate, they produce diverse phrasings of one perspective rather than diverse perspectives. Multi-agent debate (MAD), and more broadly closed-system reasoning where agents iteratively transform each other's outputs, tends to preserve answer accuracy while degrading the reasoning behind those answers. We name the multi-agent case the Debate Trap and the broader phenomenon the Reasoning Trap, offering a programmatic theory of evidence-grounded reasoning this http URL framework has three parts: (i) SFS (Supported Faithfulness Score), a claim-level metric verifying decomposed atomic claims against provided evidence (decomposer-invariant rankings: Spearman rho=1.0); (ii) EGSR (Evidence-Grounded Socratic Reasoning), replacing adversarial argumentation with evidence-grounded inquiry; (iii) Theorem 1 (DPI Bound): under standard MAD, the chain E -> O^0 -> O^1 -> ... is Markov, and the Data Processing Inequality implies E[I(E;O^{t+1})] <= E[I(E;O^t)]. Three companion results -- open-system recovery (Theorem 2), EGSR accumulation (Lemma 2), and vote-aggregation floor (Proposition 1) -- partition multi-step LLM reasoning by its information-theoretic relationship to E. Across 16 conditions on SciFact (300 claims) and FEVER (1,000 claims), DebateCV (C13) preserves 88% of baseline accuracy while SFS drops 43%; majority-vote MAD (C15) reduces SFS to 1.7% of baseline (p < 10^{-6}, d = -0.96); EGSR recovers 98%. An R6 cohort study (Korean n=10x30 FEVER; English n=3x200 SciFact) finds inter-rater Fleiss kappa <= +0.018 with 0.8-1.4 Likert intra-rater shifts across language and domain -- the human agreement that faithfulness metrics have been calibrated against is not itself stable. We offer one falsifiable conjecture: any closed-system reasoning protocol preserving Theorem 1's Markov structure is, in expectation, subject to the same DPI bound. 

---
# BIM Information Extraction Through LLM-based Adaptive Exploration 

**Authors**: Sylvain Hellin, Suhyung Jang, Stefan Fuchs, Stavros Nousias, André Borrmann  

**Link**: [PDF](https://arxiv.org/pdf/2605.01698)  

**Abstract**: BIM models provide structured representations of building geometry, semantics, and topology, yet extracting specific information from them remains remarkably difficult. Current approaches translate natural language into structured queries by assuming a fixed data organization (static approach), which BIM heterogeneity eventually invalidates. We address this with a new paradigm, adaptive exploration, where an LLM-based agent iteratively executes code to extract information from a BIM model, discovering its structure at runtime instead of assuming it. We evaluate this approach on ifc-bench v2, an open-source BIM question-answering benchmark introduced alongside this work, comprising 1,027 tasks across 37 IFC models from 21 projects. A factorial ablation across two LLM capability levels and four augmentation strategies shows that adaptive exploration significantly outperforms static query generation across all configurations, regardless of the augmentation strategy. These results indicate that BIM heterogeneity is best addressed at the paradigm level, not by further optimizing static approaches. 

---
# GRAVITY: Architecture-Agnostic Structured Anchoring for Long-Horizon Conversational Memory 

**Authors**: Yushi Sun, Bowen Cao, Dong Fang, Lingfeng Su, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2605.01688)  

**Abstract**: Long-horizon conversational agents rely on memory systems with increasingly sophisticated retrieval mechanisms. However, retrieved fragments are typically fed to the language model as unstructured text, lacking the relational, temporal, and thematic structures essential for complex reasoning. To bridge this reasoning gap, we introduce GRAVITY (\textbf{G}eneration-time \textbf{R}elational \textbf{A}nchoring \textbf{V}ia \textbf{I}njected \textbf{T}opological Memor\textbf{Y}), a plug-and-play structured memory module. GRAVITY extracts three complementary knowledge representations from raw conversational utterances: entity profiles grounded in relational graphs, temporal event tuples linked into causal traces, and cross-session topic summaries. At generation time, it injects these representations into the host system's prompt as structured anchoring contexts. This approach effectively synthesizes scattered evidence into a coherent, query-relevant context without requiring any architectural modifications to the host model. Extensive evaluations across five diverse memory systems on the LongMemEval and LoCoMo benchmarks demonstrate the efficacy of our approach. On average, GRAVITY improves LLM-judge accuracy by 7.5--10.1%. Gains are inversely correlated with baseline strength: the weakest host improves by 12.2% while the strongest still gains 3.8--5.7%. These findings establish structured context anchoring as a broadly effective, architecture-agnostic augmentation paradigm for long-horizon conversational memory. 

---
# MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety 

**Authors**: Jialin Song, Xiaodong Liu, Weiwei Yang, Wuyang Chen, Mingqian Feng, Xuekai Zhu, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.01687)  

**Abstract**: We present MultiBreak, a scalable and diverse multi-turn jailbreak benchmark to evaluate large language model (LLM) safety. Multi-turn jailbreaks mimic natural conversational settings, making them easier to bypass safety-aligned LLM than single-turn jailbreaks. Existing multi-turn benchmarks are limited in size or rely heavily on templates, which restrict their diversity. To address this gap, we unify a wide range of harmful jailbreak intents, and introduce an active learning pipeline for expanding high-quality multi-turn adversarial prompts, where a generator is iteratively fine-tuned to produce stronger attack candidates, guided by uncertainty-based refinement. Our MultiBreak includes 10,389 multi-turn adversarial prompts, spans 2,665 distinct harmful intents, and covers the most diverse set of topics to date. Empirical evaluation shows that our benchmark achieves up to a 54.0 and 34.6 higher attack success rate (ASR)} than the second-best dataset on DeepSeek-R1-7B and GPT-4.1-mini, respectively. More importantly, safety evaluations suggest that diverse attack categories uncover fine-grained LLM vulnerabilities}, and categories that appear benign under single-turn can exhibit substantially higher adversarial effectiveness in multi-turn scenarios. These findings highlight persistent vulnerabilities of LLMs under realistic adversarial settings and establish MultiBreak as a scalable resource for advancing LLM safety. 

---
# Beyond Perplexity: Character Distribution Signatures and the MDTA Benchmark for AI Text Detection 

**Authors**: Priyadarshan Narayanasamy, Swastik Agrawal, Klint Faber, Fardina Fathmiul Alam  

**Link**: [PDF](https://arxiv.org/pdf/2605.01647)  

**Abstract**: Training-free AI text detection methods primarily rely on model log-probabilities, achieving strong performance through approaches like Binoculars and DNA-DetectLLM. However, these methods face a fundamental ceiling as models are optimized through RLHF to produce human-like probability distributions. We introduce an alternative detection signal based on character distribution signatures. We provide theoretical foundations showing that AI models, trained on massive domain-balanced corpora, approximate global character patterns while humans exhibit domain-specialized distributions, creating a "Wall of Separation" where human-AI divergence significantly exceeds AI-AI divergence. To enable systematic evaluation, we construct the Models-Domains-Temperatures-Adversarials (MDTA) benchmark comprising 642,274 prompt-aligned samples across 4 models, 5 domains, 3 temperature settings, and 3 adversarial strategies, substantially expanding the HC3 dataset with modern model responses, temperature variation, and adversarial augmentation. We introduce the Letter Distribution Score (LD-Score), demonstrating low correlation (r = 0.08-0.13) with perplexity methods. When integrated with DNA-DetectLLM, Binoculars and FastDetectGPT via a non-linear classifier, LD-Score yields consistent improvements in AUROC and F1, with particularly pronounced gains in specialized domains where vocabulary constraints amplify the detection signal. The MDTA dataset can be accessed at: this https URL. 

---
# Prosa: Rubric-Based Evaluation of LLMs on Real User Chats in Brazilian Portuguese 

**Authors**: Roseval Malaquias Junior, Giovana Kerche Bonás, Thales Sales Almeida, Hugo Abonizio, Thiago Laitz, Ramon Pires, Marcos Piau, Celio Larcher, Rodrigo Nogueira  

**Link**: [PDF](https://arxiv.org/pdf/2605.01630)  

**Abstract**: Rankings produced by holistic LLM-as-a-judge scoring are sensitive to the bias of the chosen judge model. We show that switching to binary rubric scoring with multi-judge filtering removes this sensitivity: decomposing the judgement matters more than the judge model itself. To support this claim, we introduce Prosa, the first real user multi-turn Brazilian Portuguese chat benchmark: 1,000 WildChat conversations scored by three judges from three model families on 16 models. Under filtered rubric scoring the three judges agree on every one of the 16 ranks, whereas under holistic scoring they agree on only 7 of 16. Additionally, the rubric filtering pipeline increases the average score gap between neighbouring models by 47%, thereby improving Prosa's discriminative power. Evaluating a new model on Prosa costs approximately $2.1 when using Gemini 3 Flash as the judge. We release the benchmark and the filtering code to ensure that future models can be assessed under identical conditions. These artifacts also make our rubric-based scoring method reusable beyond Prosa, supporting other open-ended evaluation settings. 

---
# Where Do Prompt Perturbations Break Generation? A Segment-Level View of Robustness in LoRA-Tuned Language Models 

**Authors**: Zhuoyun Li, Boxuan Wang, Jinwei Hu, Zhenglin Huang, Qisong He, Xinmiao Huang, Guangliang Cheng, Xiaowei Huang, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.01605)  

**Abstract**: Large language models are sensitive to minor prompt perturbations, yet existing robustness methods usually enforce consistency at the whole-sequence level. This holistic view can hide an important failure mode: a perturbed response may remain globally similar to the clean one while drifting on a critical entity, relation, or conclusion. We introduce S$^2$R$^2$, a segment-level framework for robust LoRA fine-tuning. S$^2$R$^2$ decomposes clean and perturbed generations into semantic segments, aligns them with an optimal-transport objective, and penalises the segments with the largest meaning drift. To connect this output-side objective with model adaptation, we add an adapter-stability regulariser motivated by segment-level attention reallocation, using LoRA norm control as a tractable proxy for limiting perturbation-amplified evidence shifts. A PAC-Bayesian complexity view further explains why controlling adapter growth may support transfer beyond observed perturbations. Experiments on summarisation benchmarks show that S$^2$R$^2$ improves robustness under typographical noise, deletion, synonym replacement, and paraphrasing, while maintaining competitive clean performance and stronger cross-dataset transfer than consistency-based baselines. 

---
# Fine-Tuning Pre-Trained Code Models for AI-Generated Code Detection 

**Authors**: Jany-Gabriel Ispas, Sergiu Nisioi  

**Link**: [PDF](https://arxiv.org/pdf/2605.01596)  

**Abstract**: This paper describes the system submitted by team \textbf{Archaeology} to SemEval-2026 Task~13 on AI-generated code detection. The shared task consists of three subtasks; we participate in Subtask-A (binary classification: human-written vs.\ AI-generated code) and Subtask-B (11-class attribution of the generating model). Starting from a TF-IDF and Logistic Regression baseline, we fine-tune four pre-trained code models (CodeBERT, GraphCodeBERT, UniXcoder, and CodeT5+) with separate strategies for each subtask. For Subtask-A, we use leave-one-language-out cross-validation, code augmentation, chunked inference with trimmed-mean aggregation, and threshold calibration on a difficult dataset. For Subtask-B, we use sandwich token packing, class-balanced loss, and multi-seed ensembling with test-time augmentation. Our best submissions obtain macro-F1 scores of 0.737 on Subtask-A (6th/81 teams) and 0.422 on Subtask-B (7th/34 teams). 

---
# Automated Interpretability and Feature Discovery in Language Models with Agents 

**Authors**: Arnau Marin-Llobet, Javier Ferrando  

**Link**: [PDF](https://arxiv.org/pdf/2605.01555)  

**Abstract**: We introduce an autonomous multiagent framework for mechanistic interpretability that automates both explaining and finding internal features in large language models. The system runs two coupled loops: (1) explanation refinement, where an agent proposes competing hypotheses and iteratively tests them with targeted prompt controls and a multi-metric evaluation; and (2) feature discovery, where an agent generates prompt sets, constructs a k-nearest-neighbor graph in activation space, and retrieves candidate features using statistical separability and semantic coherence criteria. On Gemma-2 family models and MLP neurons in weight-sparse transformers, our agent improves over one-shot auto-interpretations, discovers language-specific and safety-relevant features, and produces auditable explanation traces, showing that agent-driven empirical loops yield sharper and more falsifiable explanations than one-shot labels. 

---
# The grip of grammar on meaning uncertainty: cross-linguistic evidence, neural correlates, and clinical relevance 

**Authors**: Rui He, Claudio Palominos, Samuele Vallisa, Ni Yang, Han Zhang, Miguel Ángel Santos Santos, Neguine Rezaii, Sergi Valero, Yonghua Huang, Huan Li, Hong Jiang, Yongjun Peng, Maria Francisca Alonso-Sánchez, Frederike Stein, Tilo Kircher, Philipp Homan, Iris Sommer, Lena Palaniyappan, Wolfram Hinzen  

**Link**: [PDF](https://arxiv.org/pdf/2605.01537)  

**Abstract**: Isolated word meanings are inherently uncertain. This uncertainty reduces when they are combined and anchored in context. We propose that grammar compresses meaning uncertainty cross-linguistically, which is reflected in brain and selectively disrupted in disorders. Compression was operationalized as the relative difference between non-contextual surprisal estimated from lexical frequency, and contextual surprisal from grammar-sensitive models. In narratives from 20 languages, contextual surprisal reduced frequency-based surprisal. This reduction closely tracked the surprisal cost of reversing word order, and scaled with richer, non-redundant lexis as organized by more complex but optimal dependency structure. During fMRI, surprisal and its reduction explained BOLD activity for comprehension and production in overlapping but distinct regions. Uncertainty reduction was significantly attenuated in aphasia, dementia, and schizophrenia, but remained intact where primary deficit is not language. These findings position uncertainty reduction via grammar as a foundational concept that illuminates principles, brain basis, and disruptions of language. 

---
# FT-RAG: A Fine-grained Retrieval-Augmented Generation Framework for Complex Table Reasoning 

**Authors**: Zebin Guo, Weidong Geng, Ruichen Mao  

**Link**: [PDF](https://arxiv.org/pdf/2605.01495)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding responses in external knowledge during inference. However, conventiona RAG systems under-perform on structured tabular data, largely due to coarse retrieval granularity and insufficient table semantic comprehension. To address these limitations, we introduce FT-RAG, a fine-grained framework that employs knowledge association by decomposing tables into entry-level semantic units to construct a structured graph. FT-RAG employs a structural neighbor expansion mechanism to find semantically connected entities during graph retrieval, followed by multi-modal fusion to consolidate the context of table retrieval results. Further, to address the scarcity of specialized datasets in this domain, we introduce Multi-Table-RAG-Lib, a benchmark comprising 9870 QA pairs with high complexity and difficulty, curated to demand multi-table integration and text-table information fusion for reasoning. FT-RAG surpasses top-performing baselines across all metrics, achieving a 23.5\% and 59.2\% improvement in table-level and cell-level Hit Rates, respectively. Generation performance also sees a remarkable 62.2\% increase in exact value accuracy recall. These metrics verify the framework's effectiveness in factual grounding across both pure tabular and heterogeneous table-text contexts. Therefore, our method establishes a new state-of-the-art performance for complex reasoning over mixed-modality documents. 

---
# ReMedi: Reasoner for Medical Clinical Prediction 

**Authors**: Yushi Cao, Yiming Chen, Hongchao Jiang, Hung-yi Lee, Robby T. Tan  

**Link**: [PDF](https://arxiv.org/pdf/2605.01474)  

**Abstract**: Predicting future clinical outcomes from electronic health records (EHR) remains challenging due to the complexity and heterogeneity of patient data. LLMs have shown strong potential for such predictive tasks, yet existing approaches mainly focus on enhancing medical knowledge through distillation or RAG while relying on the model's internal ability to interpret contextual information. In this work, we present ReMedi (Reasoner for Medical Clinical Prediction), a framework for improving clinical outcome prediction from EHR. ReMedi generates rationale-answer pairs using a challenging sample regeneration mechanism for complex clinical questions, which leverages ground-truth answers as hints to enhance reasoning for further fine-tuning and preference tuning. ReMedi integrates ground-truth outcome guidance into the preference data construction loop, regenerating rationale-answer variants. By tuning on these rationale-answer pairs, the model improves its predictive performance. Experiments on multiple EHR prediction tasks demonstrate substantial gains of up to 19.9 percent over state-of-the-art baselines in terms of F1 score, underscoring ReMedi's effectiveness in real-world clinical prediction. 

---
# Auditing demographic bias in AI-based emergency police dispatch: a cross-lingual evaluation of eleven large language models 

**Authors**: William Guey, Wei Zhang, Pierrick Bougault, Yi Wang, Bertan Ucar, Vitor D. de Moura, José O. Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2605.01451)  

**Abstract**: Large language models (LLMs) are rapidly being integrated into high-stakes public safety systems, including emergency call triage and dispatch decision support, yet their demographic fairness in this context remains largely untested. Here we introduce a cross-lingual audit framework that operationalizes the Police Priority Dispatch System as a five-level ordinal classification task and applies a controlled minimal-pair design to isolate the effect of demographic cues. Across 19,800 model outputs spanning 11 frontier models, 15 scenario pairs, three demographic categories (religious appearance, gender, and race), and two languages (English and Mandarin Chinese), we find that demographic bias emerges systematically when incident severity is ambiguous but largely disappears when the operational priority is clearly determined by call content. Bias magnitude varies by demographic axis, with the largest effects observed for religious appearance, followed by gender and race. Critically, bias does not transfer consistently across languages: gender bias is substantially amplified in Mandarin Chinese, whereas race bias is more pronounced in English, revealing cross-lingual asymmetries that aggregate analyses obscure. In several scenarios, demographic cues produce counter-directional effects, challenging simple stereotype-amplification accounts of model behavior. These findings suggest that bias in LLM-based dispatch is not a fixed property of models alone, but arises from the interaction between demographic signals, contextual ambiguity, and language. Beyond these empirical results, the proposed framework provides a scalable audit infrastructure that enables deploying agencies to evaluate candidate models on jurisdiction-relevant scenarios prior to real-world adoption. 

---
# Artificial intelligence language technologies in multilingual healthcare: Grand challenges ahead 

**Authors**: Vicent Briva-Iglesias  

**Link**: [PDF](https://arxiv.org/pdf/2605.01441)  

**Abstract**: AI language technologies (AILTs), increasingly enabled by large language models (LLMs), are becoming embedded in multilingual healthcare workflows for translation, rewriting, documentation, interpreting, and messaging in language-discordant settings. Yet fluent output is not the same as clinically safe or equitable communication: performance varies across languages, accents, tasks, and workflows, and efficiency gains can hide errors, reduce traceability, and shift responsibility across clinicians, translators, interpreters, and health systems. This narrative review synthesises recent peer-reviewed evidence across written communication, spoken communication, and emerging agentic workflows. Using the Human-Centered AI Language Technology (HCAILT) lens, it examines capabilities, evaluation practices, implementation patterns, and recurrent errors through reliability, safety culture, and trustworthiness. We identify key convergences and contradictions in the literature and propose seven grand challenges for the next phase of research and deployment. Progress, we argue, requires not only better models but also accountable sociotechnical design, calibrated human oversight, and stronger collaboration across MT/NLP, translation studies, HCI, clinical practice, implementation science, and policy. 

---
# Hallucinations Undermine Trust; Metacognition is a Way Forward 

**Authors**: Gal Yona, Mor Geva, Yossi Matias  

**Link**: [PDF](https://arxiv.org/pdf/2605.01428)  

**Abstract**: Despite significant strides in factual reliability, errors -- often termed hallucinations -- remain a major concern for generative AI, especially as LLMs are increasingly expected to be helpful in more complex or nuanced setups. Yet even in the simplest setting -- factoid question-answering with clear ground truth-frontier models without external tools continue to hallucinate. We argue that most factuality gains in this domain have come from expanding the model's knowledge boundary (encoding more facts) rather than improving awareness of that boundary (distinguishing known from unknown). We conjecture that the latter is inherently difficult: models may lack the discriminative power to perfectly separate truths from errors, creating an unavoidable tradeoff between eliminating hallucinations and preserving utility.
This tradeoff dissolves under a different framing. If we understand hallucinations as confident errors -- incorrect information delivered without appropriate qualification -- a third path emerges beyond the answer-or-abstain dichotomy: expressing uncertainty. We propose faithful uncertainty: aligning linguistic uncertainty with intrinsic uncertainty. This is one facet of metacognition -- the ability to be aware of one's own uncertainty and to act on it. For direct interaction, acting on uncertainty means communicating it honestly; for agentic systems, it becomes the control layer governing when to search and what to trust. Metacognition is thus essential for LLMs to be both trustworthy and capable; we conclude by highlighting open problems for progress towards this objective. 

---
# Medmarks: A Comprehensive Open-Source LLM Benchmark Suite for Medical Tasks 

**Authors**: Benjamin Warner, Ratna Sagari Grandhi, Max Kieffer, Aymane Ouraq, Saurav Panigrahi, Geetu Ambwani, Kunal Bagga, Nikhil Khandekar, Arya Hariharan, Nishant Mishra, Manish Ram, Shamus Sim Zi Yang, Ahmed Essouaied, Adepoju Jeremiah Moyondafoluwa, Robert Scholz, Bofeng Huang, Molly Beavers, Srishti Gureja, Anish Mahishi, Sameed Khan, Maxime Griot, Hunar Batra, Jean-Benoit Delbrouck, Siddhant Bharadwaj, Ronald Clark, Ashish Vashist, Anas Zafar, Leema Krishna Murali, Harsh Deshpande, Ameen Patel, William Brown, Johannes Hagemann, Connor Lane, Paul Steven Scotti, Tanishq Mathew Abraham  

**Link**: [PDF](https://arxiv.org/pdf/2605.01417)  

**Abstract**: Evaluating large language models (LLMs) for medical applications remains challenging due to benchmark saturation, limited data accessibility, and insufficient coverage of relevant tasks. Existing suites have either saturated, heavily depend on restricted datasets, or lack comprehensive model coverage. We introduce Medmarks, a fully open-source evaluation suite with 30 benchmarks spanning question answering, information extraction, medical calculations, and open-ended clinical reasoning. We perform a systematic evaluation of 61 models across 71 configurations using verifiable metrics and LLM-as-a-Judge. Our results show that frontier reasoning models (Gemini 3 Pro Preview, GPT-5.1, & GPT-5.2) achieve the highest performance across both benchmarks, most frontier proprietary models are significantly more token efficient than open-weight alternatives, medically fine-tuned models outperform their generalist counterparts, and that models are susceptible to answer-order bias (particularly smaller models and Grok 4). A subset of our evals (Medmarks-T) can be directly used as reinforcement learning environments to post-train LLMs for medical reasoning. Code is available at this https URL 

---
# Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression 

**Authors**: Yao Du, Shanshan Li, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.01402)  

**Abstract**: Multimodal large language models (MLLMs) struggle with numerical regression under long-tailed target distributions. Token-level supervised fine-tuning (SFT) and point-wise regression rewards bias learning toward high-density regions, leading to regression-to-the-mean behavior and poor tail performance. We identify the lack of cross-sample relational supervision as a key limitation of existing MLLM training paradigms. To address it, we propose a distribution-aware reinforcement learning framework based on Group Relative Policy Optimization, which introduces batch-level comparison-based supervision via the Concordance Correlation Coefficient-based reward to align predicted and ground-truth distributions in terms of correlation, scale, and mean. The framework is plug-and-play, requiring no architectural modification. Experiments on a unified suite of long-tailed regression benchmarks show consistent improvements over SFT and existing MLLM regression methods, with particularly strong gains in medium- and few-shot regimes. 

---
# Verbal-R3: Verbal Reranker as the Missing Bridge between Retrieval and Reasoning 

**Authors**: Sangkwon Park, Donghun Kang, Jisoo Mok, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2605.01399)  

**Abstract**: The conventional Retrieval-Augmented Generation (RAG) paradigm of injecting raw retrieved texts into the Large Language Model (LLM)'s context often results in suboptimal integration of retrieved information. This paper proposes to bridge retrieval results and the LLM's reasoning ability through Verbal Annotations, analytic narratives that explicitly articulate the logical connection between a search query and retrieved contexts. Our empirical investigation reveals the potential of Verbal Annotations to substantially enhance the LLM's ability to generate accurate, contextually-grounded responses. Motivated by this finding, we introduce Verbal-R3, a novel agentic RAG framework that consists of a Generator and a Verbal Reranker. The Generator performs iterative retrieval and reasoning, while the Verbal Reranker returns relevance scores and Verbal Annotations to guide the reasoning and answering process of the Generator. The inference process of Verbal-R3 is further refined through relevance-guided test-time scaling, which efficiently allocates test-time compute for effective trajectory expansion. Verbal-R3 achieves state-of-the-art performance on complex Question Answering benchmarks, validating the effectiveness of the proposed framework. 

---
# MemORAI: Memory Organization and Retrieval via Adaptive Graph Intelligence for LLM Conversational Agents 

**Authors**: Hung Pham Van, Nguyen Manh Hieu, Khang Pham Tran Tuan, Nam Le Hai, Linh Ngo Van, Nguyen Thi Ngoc Diep, Trung Le  

**Link**: [PDF](https://arxiv.org/pdf/2605.01386)  

**Abstract**: Large Language Models (LLMs) lack persistent memory for long-term personalized conversations. Existing graph-based memory systems suffer from information dilution, absent provenance tracking, and uniform retrieval that ignores query context. We introduce MemORAI (Memory Organization and Retrieval via Adaptive Graph Intelligence), a framework that integrates three innovations: selective memory filtering with dual-layer compression to retain user-persona-relevant content, a provenance-enriched multi-relational graph tracking factual origins at the turn level, and query-adaptive subgraph retrieval with Dynamic Weighted PageRank that applies query-conditioned edge weighting. Evaluated on LOCOMO and LongMemEval benchmarks, MemORAI achieves state-of-the-art performance in memory retrieval and personalized response generation, demonstrating that selective storage, enriched representation, and adaptive retrieval are essential for coherent, personalized LLM agents. 

---
# A framework for analyzing concept representations in neural models 

**Authors**: Burin Naowarat, Hao Tang, Sharon Goldwater  

**Link**: [PDF](https://arxiv.org/pdf/2605.01381)  

**Abstract**: Understanding how neural models represent human-interpretable concepts is challenging. Prior work has explored linear concept subspaces from diverse perspectives, such as probing and concept erasure. We introduce a unified framework to study these subspaces along two axes: \textit{containment}, which tests if a concept is fully represented in a subspace but not outside it, and \textit{disentanglement}, which tests for isolation from other concepts. In experiments on both text and speech models, we first highlight that concept subspaces may not be uniquely determined, and discuss the implications for concept subspace analysis. Then, we compare properties of concept subspaces estimated using five estimators, proposed in different communities. We find that (1) the choice of estimator impacts the containment and disentanglement properties; (2) the state-of-the-art concept erasure method, LEACE, performs well on both testing axes, but still struggles to generalize to unseen data; and (3) in HuBERT speech representations, phone information is both contained and disentangled from speaker information, while speaker information is hard to contain in a compact subspace, despite being disentangled from phones. 

---
# MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation 

**Authors**: Pham Khanh Chi, Quoc Phong Dao, Thuat Nguyen, Linh Ngo Van, Trung Le, Thanh Hong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.01374)  

**Abstract**: Knowledge distillation is a key technique for compressing large language models (LLMs), but most existing methods align representations at fixed layers or token-level outputs, ignoring how representations evolve across depth. As a result, the student is only weakly guided to capture the teacher's internal relational structure during distillation, which limits knowledge transfer. To address this limitation, we propose Multi-Granular Trajectory Alignment (MTA), a framework that aligns teacher and student representations along their layer-wise transformation trajectory. MTA adopts a layer-adaptive strategy: lower layers are aligned at the word level to preserve lexical information, while higher layers operate on phrase-level spans (e.g., noun and verb phrases) to capture compositional semantics. We instantiate this idea through a Dynamic Structural Alignment loss that matches the relative geometry among semantic units within each layer. This design is motivated by empirical findings that Transformer representations become increasingly abstract with depth, and is also consistent with linguistic views in which higher-level meaning emerges through the composition of lower-level lexical units. We further incorporate a Hidden Representation Alignment loss to directly align selected teacher-student layers. Experiments show that MTA consistently outperforms state-of-the-art baselines on standard benchmarks, with ablations confirming the contribution of each component. 

---
# Focus on the Core: Empowering Diffusion Large Language Models by Self-Contrast 

**Authors**: Jinyuan Feng, Xin Yu, Yiqun Chen, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Zhiqiang Pu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01373)  

**Abstract**: The iterative denoising paradigm of Diffusion Large Language Models (DLMs) endows them with a distinct advantage in global context modeling. However, current decoding strategies fail to leverage this capability, typically exhibiting a local preference that overlooks the heterogeneous information density within the context, ultimately degrading generation quality. To address this limitation, we systematically investigate high-information-density (HD) tokens and present two key findings: (1) explicitly conditioning on HD tokens substantially improves output quality; and (2) HD tokens exhibit an early-decoding tendency, converging earlier than surrounding tokens. Motivated by these findings, we propose Focus on the Core \textbf{(FoCore)}, a training-free decoding strategy that utilizes HD tokens in a self-contrast manner, wherein HD tokens are temporarily remasked as negative samples, to guide generation. We further introduce FoCore\_Accelerate \textbf{(FoCore\_A)}, an efficient variant that, upon detecting HD token convergence, performs parallel decoding over stable candidates within a local context window, substantially accelerating generation. Extensive experiments on math, code and logical reasoning benchmarks demonstrate that FoCore consistently improves generation quality and efficiency across both LLaDA and Dream backbones. For instance, on HumanEval, FoCore improves pass@1 from 39.02 to 42.68 over standard Classifier-Free Guidance, while FoCore-A reduces the number of decoding steps by 2.07x and per-sample latency from 20.76s to 8.64s (-58.4\%). 

---
# Embedding-based In-Context Prompt Training for Enhancing LLMs as Text Encoders 

**Authors**: Ailiang Lin, Zhuoyun Li, Keyu Mao, Kotaro Funakoshi, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2605.01372)  

**Abstract**: Large language models (LLMs) have been widely explored for embedding generation. While recent studies show that in-context learning (ICL) effectively enhances the representational capability of LLMs by prepending a few task-related demonstrations, it causes substantial token overhead due to the increased sequence length. In this work, we propose EPIC, a novel embedding-based in-context prompt training strategy that leverages ICL to generate high-quality embeddings while reducing computational burden during both training and inference. This approach replaces discrete text demonstrations with their corresponding continuous embeddings, which not only encourages the LLM to align semantically-related text pairs during contrastive learning, but also requires the model to interpret demonstration embeddings as part of the in-context prompt. Consequently, EPIC-trained models achieve excellent embedding performance both with or without in-context prompts at inference time. Comprehensive experiments demonstrate that our method establishes new state-of-the-art results on the MTEB benchmark, surpassing frontier models trained solely on publicly available retrieval data. Extensive ablation studies further validate the effectiveness and necessity of our mechanism. 

---
# On Stable Long-Form Generation: Benchmarking and Mitigating Length Volatility 

**Authors**: Zhitao He, Haolin Yang, Rui Min, Zeyu Qin, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2605.01357)  

**Abstract**: Large Language Models (LLMs) excel at long-context understanding but exhibit significant limitations in long-form generation. Existing studies primarily focus on single-generation quality, generally overlooking the volatility of the output. This volatility not only leads to significant computational costs but also severely impacts the models' reliable application. To address this gap, our work unfolds in three stages: benchmarking, probing, and mitigation. We first propose the VOlatility in Long-form Text Benchmark (VOLTBench), a novel heterogeneous-task benchmark designed to systematically quantify the length volatility of long-form generation. Subsequently, by analyzing attention traces, we conduct an in-depth probe to identify several common internal patterns that cause this volatility. Finally, to mitigate long-form output volatility, we propose Stable Generation via Logits Boosting (GLoBo), a lightweight decoding-stage optimization strategy, designed to significantly enhance both the length accuracy and stability of long-form generation without additional training. Extensive experiments on VOLTBench provide the first systematic confirmation of severe long-form output instability in mainstream models and validate that our proposed method successfully improves the mean output length of the base model by 148% and reduces the length volatility by 69%, while maintaining high generation quality. 

---
# LLM Output Detectability and Task Performance Can be Jointly Optimized 

**Authors**: Koshiro Saito, Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2605.01350)  

**Abstract**: Detecting machine-generated text is essential for transparency and accountability when deploying large language models (LLMs). Among detection approaches, watermarking is a statistically reliable method by design -- it embeds detectable signals into LLM outputs by biasing their token distributions. However, it has been reported that watermarked LLMs often perform worse on downstream tasks. We propose PUPPET, a framework that fine-tunes an LLM via reinforcement learning to generate text that is both more detectable and better performing on downstream tasks. We use two reward functions: a detector that outputs a machine-class likelihood and an evaluator that measures a task-specific metric. Experiments on long-form QA, summarization, and essay writing show that LLMs trained with PUPPET achieve high detectability competitive with watermarking methods while outperforming them on downstream tasks. The analysis shows that this optimization can be performed efficiently with only a few thousand samples in 1--2 GPU hours. Moreover, these gains are consistent across out-of-domain tasks, different LLM families, and model sizes, and are even robust to paraphrasing attacks. 

---
# MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate 

**Authors**: Jianze Wang, Ying Liu, Jinlong Chen, Xuchun Hu, Qilong Zhang, Yu Cao, Jun Wang, Hua Yang, Yong Xie, Qianglong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.01347)  

**Abstract**: On-policy distillation (OPD) trains a student on its own trajectories under token-level teacher supervision, but existing methods are capped by a single-teacher capability ceiling: when the teacher errs, the student inherits the error. OPD also remains largely unexplored in agentic tasks, where per-step errors compound across long trajectories and destabilize training. We propose MAD-OPD (Multi-Agent Debate-driven On-Policy Distillation), which breaks this ceiling by recasting the distillation teacher as a deliberative collective of teachers that debate over the student's on-policy state; the debate produces an emergent collective intelligence that supplies token-level supervision, with each teacher's contribution weighted by its post-debate confidence. To extend OPD to agentic tasks, we also introduce On-Policy Agentic Distillation (OPAD), which adds step-level sampling to stabilize training under multi-step error compounding. We additionally derive a task-adaptive divergence principle, selecting JSD (Jensen-Shannon divergence) for agentic stability and reverse KL (Kullback-Leibler) divergence for code generation, and verify it both theoretically and empirically. Across six teacher-student configurations (Qwen3 and Qwen3.5; 1.7B-14B students, 8B-32B teachers) and five agentic and code benchmarks, MAD-OPD ranks first across all six configurations; on the 14B+8B$\to$4B setting it lifts the agentic average by $+2.4\%$ and the code average by $+3.7\%$ over the stronger single-teacher OPD. 

---
# A Multi-View Media Profiling Suite: Resources, Evaluation, and Analysis 

**Authors**: Muhammad Arslan Manzoor, Dilshod Azizov, Daniil Orel, Umer Siddique, Zain Muhammad Mujahid, Yufang Hou, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2605.01336)  

**Abstract**: News outlets shape public opinion at a scale that makes automated detection of political bias and factuality essential. However, the field still lacks unified resources, comprehensive evaluations across diverse approaches, and systematic analyses of the representations and fusion strategies that matter most, especially under label sparsity and dataset diversity. In addition, there is little empirical work reporting broad, observation-driven findings about what consistently works, what fails, and why. We address these gaps through four main contributions. First, we introduce MBFC-2025, a large-scale label set covering approximately 2,600 outlets from Media Bias/Fact Check (MBFC). Second, we construct multiview representations for ACL-2020 (Panayotov et al., 2022), which includes around 900 outlets, as well as for MBFC-2025. These representations span Alexa graphs, hyperlink graphs, LLM-derived graphs, articles, and Wikipedia descriptions. Third, we provide a systematic evaluation and analysis of embedding views and fusion strategies, including a reinforcement learning-based fusion variant. Fourth, we conduct extensive experiments that achieve state-of-the-art results on ACL-2020 and establish strong benchmarks on MBFC-2025. 

---
# OralMLLM-Bench: Evaluating Cognitive Capabilities of Multimodal Large Language Models in Dental Practice 

**Authors**: Rongyang Wang, Shuang Zhou, Jiashuo Wang, Wenya Xie, Xiaoxia Che  

**Link**: [PDF](https://arxiv.org/pdf/2605.01333)  

**Abstract**: Multimodal large language models (MLLMs) have emerged as a promising paradigm for dental image analysis. However, their ability to capture the multi-level cognitive processes required for radiographic analysis remains unclear. Here, we present a comprehensive benchmark to evaluate the cognitive capabilities of MLLMs in dental radiographic analysis. It spans three critical imaging modalities, i.e., periapical, panoramic, and lateral cephalometric radiographs, and defines four cognitive categories: perception, comprehension, prediction, and decision-making. The benchmark comprises 27 clinically grounded tasks derived from public datasets, with manually curated annotations and 3,820 clinician assessments for evaluation. Six frontier MLLMs, including GPT-5.2 and GLM-4.6, are evaluated. We demonstrate the performance gap between MLLMs and clinicians in dental practice, delineate model strengths and limitations, characterize failure patterns, and provide recommendations for improvement. This data resource will facilitate the development of next-generation artificial intelligence systems aligned with clinical cognition, safety requirements, and workflow complexity in dental practice. 

---
# Creating and Evaluating Figurative Language Dataset for Sindhi 

**Authors**: Wazir Ali, Adeeb Noor, Saifullah Tumrani  

**Link**: [PDF](https://arxiv.org/pdf/2605.01323)  

**Abstract**: In this article, we introduce SiNFluD, a novel benchmark dataset for Sindhi figurative language classification. We first collect raw text from various blogs, social media platforms, and literary sources, and subsequently prepare the corpus for annotation. Two native annotators label the data using the Doccano text annotation tool, achieving an inter-annotator agreement of 0.81. We then establish baseline results using 5-fold and 10-fold cross-validation. Finally, we evaluate mBERT, XLM-RoBERTa, and XLM-RoBERTa-XL models, along with SetFit for few-shot fine-tuning of sentence transformers. Among these, the pretrained XLM-RoBERTa-XL achieves the best performance. 

---
# Benchmarking LightGBM and BiLSTM for Sentiment Analysis on Indonesian E-Commerce Reviews 

**Authors**: Lidia Natasyah Marpaung, Vania Claresta, Iqfina Haula Halika, Luluk Muthoharoh, Ardika Satria, Martin Clinton Tosima Manullang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01322)  

**Abstract**: This study presents a comparative analysis between two primary approaches in Natural Language Processing (NLP): Machine Learning (ML) utilizing the PyCaret AutoML framework, and Deep Learning (DL). The evaluation is conducted on a sentiment analysis task using an Indonesian e-commerce review dataset sourced from Hugging Face. The dataset, consisting of 15,000 samples, is partitioned into training, validation, and testing sets. The ML experiments compare LightGBM, Logistic Regression, and Support Vector Machine (SVM) algorithms, whereas the DL experiment implements a Bidirectional Long Short-Term Memory (BiLSTM) architecture. The experimental results demonstrate that the BiLSTM model outperforms all ML models, achieving an accuracy of 98.87\% and an F1-Score of 98.87\%. Meanwhile, LightGBM emerges as the best-performing ML model with an accuracy of 98.23\% in a highly efficient training time. This research proves that the BiLSTM architecture is highly capable of capturing the sequential context of Indonesian review texts, making it the superior model for this specific classification task. 

---
# Sentiment Analysis of Mobile Legends App Reviews Using Machine Learning and LSTM-Based Deep Learning Models 

**Authors**: Vira Putri Maharani, Kharisa Harvanny, Daris Samudra, Luluk Muthoharoh, Ardika Satria, Martin Clinton Tosima Manullang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01317)  

**Abstract**: This paper compares Machine Learning and LSTM-based Deep Learning methods for sentiment analysis of Mobile Legends app reviews. Using a dataset of 10,000 reviews labeled as positive, negative, and neutral, the study evaluates traditional models with TF-IDF and PyCaret AutoML and compares them against an LSTM model designed to capture sequential text dependencies. The results show that the LSTM model outperforms the classical Machine Learning baselines, achieving 92% accuracy and a weighted F1-score of 91%. The findings indicate that deep learning is more effective for handling informal and context-dependent user review text. 

---
# Enhancing Game Review Sentiment Classification on Steam Platform with Attention-Based BiLSTM 

**Authors**: Abit Ahmad Oktarian, Fadhil Fitra Wijaya, Dhafin Razaqa Luthfi, Luluk Muthoharoh, Ardika Satria, Martin Clinton Tosima Manullang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01315)  

**Abstract**: This paper investigates sentiment classification of Steam game reviews using an attention-based Bidirectional Long Short-Term Memory (BiLSTM) model. Using a dataset of 50,000 reviews sampled from a larger Steam review corpus, the authors compare a traditional machine learning baseline based on TF-IDF and PyCaret AutoML with a deep learning approach implemented in PyTorch. The proposed BiLSTM+Attention model is trained with class-weighted cross-entropy to address class imbalance and achieves 83% accuracy and 85% weighted F1-score on the test set, with 90% recall for negative reviews. The paper also presents attention visualizations to show interpretability by highlighting sentiment-bearing words. The study concludes that the BiLSTM+Attention model is effective for analyzing user sentiment in Steam reviews and useful for helping developers understand player feedback. 

---
# Beyond Semantic Relevance: Counterfactual Risk Minimization for Robust Retrieval-Augmented Generation 

**Authors**: Peiyang Liu, Qiang Yan, Ziqiang Cui, Di Liang, Xi Wang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.01302)  

**Abstract**: Standard Retrieval-Augmented Generation (RAG) systems predominantly rely on semantic relevance as a proxy for utility. However, this assumption collapses in realistic decision-making scenarios where user queries are laden with cognitive biases, such as false premises or confirmation bias. In such cases, maximizing relevance paradoxically promotes the retrieval of sycophantic evidence that reinforces hallucinations, a critical failure we term the ``Relevance-Robustness Gap''. To bridge this gap, we propose CoRM-RAG (Counterfactual Risk Minimization for RAG), a framework that aligns retrieval with decision safety rather than mere similarity. Grounded in causal intervention, we introduce a Cognitive Perturbation Protocol to simulate user biases during training, which is then distilled into a lightweight Evidence Critic. This scoring module learns to identify documents that possess sufficient evidential strength to steer the model toward correctness despite adversarial query perturbations. Extensive experiments on decision-making benchmarks demonstrate that CoRM-RAG significantly outperforms strong dense retrievers and LLM-based rerankers in adversarial settings, while enabling effective risk-aware abstention through reliable robustness scoring. Our code is available at this https URL. 

---
# Addressing Data Scarcity in Bangla Fake News Detection: An LLM-Based Dataset Augmentation Approach 

**Authors**: Ahmed Alfey Sani, Kazi Akib Zaoad, Shefayat E Shams Adib, Md Abdul Muqtadir, Ajwad Abrar  

**Link**: [PDF](https://arxiv.org/pdf/2605.01292)  

**Abstract**: The growing spread of misinformation in digital media highlights the need for reliable fake news detection systems, yet progress in under-resourced languages such as Bangla is limited by small and imbalanced datasets. This study investigates whether Large Language Model (LLM) based augmentation can effectively address this limitation and improve Bangla fake news classification. Existing datasets remain valuable but highly imbalanced, limiting model performance, and LLM based augmentation for Bangla has been scarcely explored. To fill this gap, we propose a systematic augmentation framework that generates synthetic Bangla news articles using the instruction tuned Gemma 3 27B IT model, supported by semantic filtering and controlled subsampling to preserve label consistency and diversity. We compare zero shot and few shot prompting, evaluate multiple augmentation rates, and examine random versus similarity-based selection strategies. Our experiments show that augmenting only the minority class with a high augmentation rate and random subsampling yields the strongest gains, raising the Fake News F1 score from 0.85 to 0.88. To support reproducibility and further research in this low-resource domain, we publicly release 4,545 synthetically generated Bangla fake news samples along with our full implementation. These findings demonstrate that well-designed LLM-driven augmentation can significantly improve fake news detection in low resource settings and provide a practical foundation for advancing multilingual misinformation research. 

---
# GIFT: Guided Fine-Tuning and Transfer for Enhancing Instruction-Tuned Language Models 

**Authors**: Zhiwen Ruan, Yichao Du, Jianjie Zheng, Longyue Wang, Yun Chen, Peng Li, Jinsong Su, Yang Liu, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.01256)  

**Abstract**: A promising paradigm for adapting instruction-tuned language models is to learn task-specific updates on a pretrained base model and subsequently merge them into the instruction-tuned model. However, existing approaches typically treat the instruction-tuned model as a passive target that is only involved at the final merging stage, without guiding the training process. We propose GIFT (Guided Fine-Tuning and Transfer), a simple and efficient framework that incorporates guidance from the instruction model into task adaptation. GIFT fine-tunes a low-rank adapter on the pretrained base model using confidence signals derived from the instruction-tuned model. The learned adapter is then merged into the instruction-tuned model, yielding task-specialized models that preserve general instruction-following behavior. We evaluate GIFT on mathematical and knowledge-intensive benchmarks across multiple model families and scales. Results show that GIFT consistently outperforms direct fine-tuning and representative transfer-based baselines, while maintaining robust generalization and favorable test-time scaling behavior. 

---
# Lost in the Tower of Babel: The Adverse Effects of Incidental Multilingualism in LLMs 

**Authors**: Anjishnu Mukherjee, Chutong Meng, Antonios Anastasopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2605.01224)  

**Abstract**: This paper argues that contemporary multilingual NLP has converged on a fragile and misleading paradigm of incidental multilingualism. Today's LLMs appear multilingual largely because they are trained on massive, uneven web corpora, not because multilingual or multicultural competence has been treated as a core design objective. We contend that this paradigm systematically produces unequal, brittle, and opaque behavior across languages, with severe consequences in real-world and agentic deployments where models must reason, plan, and act across multiple linguistic contexts. We report a focused empirical study of two practical questions: which languages models self-report as supported and which languages they actually respond in across multilingual prompts. We additionally demonstrate how even a simple language-change attack can surface these failures and expose hidden assumptions about language in LLM-based systems. To address this, we call for a shift toward multilingualism by design: a research agenda that treats equitable multilingual performance, cultural grounding, and cross-lingual behavioral understanding as first-class goals in all aspects of the model pipeline. 

---
# SRA: Span Representation Alignment for Large Language Model Distillation 

**Authors**: Quoc Phong Dao, Hoang Son Nguyen, Pham Khanh Chi, Tung Nguyen, Linh Ngo Van, Nguyen Thi Ngoc Diep, Trung Le  

**Link**: [PDF](https://arxiv.org/pdf/2605.01205)  

**Abstract**: Cross-Tokenizer Knowledge Distillation (CTKD) enables knowledge transfer between a large language model and a smaller student, even when they employ different tokenizers. While existing approaches mainly focus on token-level alignment strategies, which are often brittle and sensitive to discrepancies between tokenizers, we argue that the method of aggregating tokens into more robust representations before distillation is of equal importance. In this paper, we introduce \textbf{SRA} (\textbf{S}pan \textbf{R}epresentation \textbf{A}lignment for Large Language Model Distillation), a novel framework that reframes CTKD through the physical lens of Multi-Particle Dynamical Systems. SRA shifts the fundamental unit of alignment from tokens to robust, tokenizer-agnostic spans. We model each span as a cluster of particles and represent its state by its Center of Mass (CoM) - an attention-weighted average that captures rich semantic information. We leverage the concept of span centers of mass with attention-derived weighting to prioritize the most salient spans. In addition, we employ a geometric regularizer to preserve the structural integrity of the representation space and introduce aligned span logit distillation to enhance knowledge transfer across models. In challenging cross-architecture distillation experiments, SRA consistently and significantly outperforms state-of-the-art CTKD baselines, validating our physically-grounded approach. 

---
# Compute Optimal Tokenization 

**Authors**: Tomasz Limisiewicz, Artidoro Pagnoni, Srini Iyer, Mike Lewis, Sachin Mehta, Alisa Liu, Margaret Li, Gargi Ghosh, Luke Zettlemoyer  

**Link**: [PDF](https://arxiv.org/pdf/2605.01188)  

**Abstract**: Scaling laws enable the optimal selection of data amount and language model size, yet the impact of the data unit, the token, on this relationship remains underexplored. In this work, we systematically investigate how the information granularity of tokens, controlled by the compression rate (i.e., average bytes of text per token), affects scaling trends. We train 988 latent tokenized models (BLT) ranging from 50M to 7B parameters that enable setting the desired compression rate. This flexibility allows us to study the role of compression rate well beyond 4.57 bytes per token obtained with a popular BPE tokenizer. Our experiments reveal that in compute-optimal configurations, model parameter counts scale proportionally to data size measured in bytes, not in tokens as commonly perceived (Kaplan et al., 2020; Hoffmann et al., 2022). Furthermore, we discover that the optimal compression rate differs from the one obtained with BPE and decreases with compute. These findings generalize to both latent and subword tokenization, as well as to languages other than English, guiding language model developers on tokenization scheme selection for maximal compute efficiency. 

---
# Quantifying and Predicting Disagreement in Graded Human Ratings 

**Authors**: Leixin Zhang, Çağrı Çöltekin  

**Link**: [PDF](https://arxiv.org/pdf/2605.01168)  

**Abstract**: It is increasingly recognized that human annotators do not always agree, and such disagreement is inherent in many annotation tasks. However, not all instances in a given task elicit the same degree of opinion divergence. In this paper, we investigate annotation variation patterns in graded human ratings for inappropriate languages, including offensive language, hate speech, and toxic language perception. We examine whether the degree of annotation disagreement can be predicted from textual features. We further propose the Opposition Index, a metric that quantifies perspective opposition among annotators on a given item, and investigate the predictability of instances with potentially opposing human opinions. Our results show a moderate positive correlation between estimated and observed annotation variance. We find that two approaches achieve comparable performance in variance prediction: directly predicting the variance value and estimating it from predicted annotation distributions. Our results on opposition perspective prediction show that items with high opposition index values are more difficult to predict and are often underestimated by models. 

---
# Component-Aware Self-Speculative Decoding in Hybrid Language Models 

**Authors**: Hector Borobia, Elies Seguí-Mas, Guillermina Tormo-Carbó  

**Link**: [PDF](https://arxiv.org/pdf/2605.01106)  

**Abstract**: Speculative decoding accelerates autoregressive inference by drafting candidate tokens with a fast model and verifying them in parallel with the target. Self-speculative methods avoid the need for an external drafter but have been studied exclusively in homogeneous Transformer architectures. We introduce component-aware self-speculative decoding, the first method to exploit the internal architectural heterogeneity of hybrid language models, isolating the SSM/linear-attention subgraph as a zero-cost internal draft. We evaluate this on two architecturally distinct hybrid families: Falcon-H1 (parallel: Mamba-2 + attention per layer) and Qwen3.5 (sequential: interleaved linear and attention layers), with a pure Transformer control (Qwen2.5). Parallel hybrids achieve acceptance rates of alpha = 0.68 at draft length k=2 under greedy decoding, while sequential hybrids yield only alpha = 0.038 -- an 18x gap attributable to how each architecture integrates its components. The property is scale-invariant: Falcon-H1 at 3B reproduces the rates observed at 0.5B. We further show that perplexity degradation from a companion ablation study predicts speculative viability without running speculative decoding: a 3.15x ratio (Falcon) maps to alpha = 0.37 at k=4, while 81.96x (Qwen) maps to alpha = 0.019. For sequential hybrids, generic LayerSkip achieves 12x higher acceptance rates than the component-aware strategy. The composition pattern of hybrid models -- not merely the presence of alternative components -- determines whether component-level self-speculation is viable. 

---
# Interpretable Difficulty-Aware Knowledge Tracing in Tutor-Student Dialogues 

**Authors**: Shuyan Huang, Alexander Scarlatos, Jaewook Lee, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2605.01097)  

**Abstract**: Recent advances in large language models (LLMs) have led to the development of AI-powered tutoring systems that provide interactive support via dialogue. To enable these tutoring systems to provide personalized support, it is essential to assess student performance at each turn, motivating knowledge tracing (KT) in dialogue settings. However, existing dialogue-based KT approaches often ignore question difficulty modeling and rely on opaque latent representations from LLMs, hindering accurate and interpretable prediction. In this work, we propose an interpretable difficulty-aware conversational KT framework built upon LLMs, which explicitly models students' abilities and the difficulty of tutor-posed tasks at each turn. The framework incorporates the original textual question and the next tutor-posed task to estimate the student's knowledge state and the difficulty of the upcoming turn. Furthermore, it integrates Item Response Theory to map LLM's outputs into student ability and question difficulty parameters, enabling interpretable prediction of student performance grounded in cognitive theories of learning. We evaluate the framework on two tutor-student dialogue datasets. Both quantitative and qualitative results show that our framework outperforms existing KT baselines, meanwhile generating interpretable outputs consistent with cognitive theory. 

---
# Teaching LLMs Brazilian Healthcare: Injecting Knowledge from Official Clinical Guidelines 

**Authors**: Hugo Abonizio, Filipe Rocha Lopes, Roberto Lotufo, Rodrigo Nogueira  

**Link**: [PDF](https://arxiv.org/pdf/2605.01077)  

**Abstract**: Brazil's Unified Health System (SUS) relies on official clinical guidelines that define diagnostic criteria, treatments, dosages, and monitoring procedures for over 200 million citizens. Yet current LLMs perform poorly on this guideline-specific knowledge, and no benchmark evaluates clinical recall grounded in Brazilian Portuguese protocols. We address this gap by adapting Qwen2.5-14B-Instruct to the Brazilian clinical domain. From 178 official guidelines (~5.4M tokens), we generate ~70M tokens of synthetic data in three formats -- rephrases, wiki-style articles, and question-answer pairs -- using four generator LLMs. We then apply continual pre-training followed by Group Relative Policy Optimization (GRPO). We introduce HealthBench-BR, with 1,780 balanced true/false clinical assertions, and PCDT-QA, with 890 open-ended clinical questions scored by an LLM judge. Our best model achieves 83.9% on HealthBench-BR and 85.4% on PCDT-QA, outperforming GPT-5.2, Claude Sonnet 4.6, Gemini 3.1 Pro, and Google AI Overview's web-grounded RAG despite having only 14B parameters. Ablations show that generator diversity and reinforcement learning are critical to these gains. We release all datasets, benchmarks, and model weights to support reproducible clinical NLP research for Brazilian Portuguese. Code, data, and model weights are available at this https URL 

---
# Controlled Paraphrase Geometry in Sentence Embedding Space: Local Manifold Modeling and Latent Probing 

**Authors**: Leonid Bedratyuk  

**Link**: [PDF](https://arxiv.org/pdf/2605.01073)  

**Abstract**: The paper studies the local geometry of embedding clouds induced by \emph{controlled local classes of semantically close sentences}. The central question is how controlled paraphrase-like semantic variation is organized in sentence embedding space and whether this local structure can be explicitly modeled by low-degree fitted carriers.
We introduce a local geometric modeling scheme based on affine, quadratic, and cubic fitted models. We also use a surface-based latent probing procedure that constructs synthetic latent points in a reduced local PCA space with respect to the fitted carrier. The procedure is intended as an offline method for representation-space analysis, local manifold modeling, and geometry-aware latent probing.
Generated latent points are evaluated using criteria that measure consistency with the fitted surface, preservation of neighborhood structure, agreement with the empirical distribution, stability of Hessian-based second-order shape descriptors, and stability of fitted-model coefficients. Experiments on controlled sets of semantically close sentences show that nonlinear local models describe embedding clouds more accurately than affine models. Surface-based generation provides strong fitted-geometry fidelity, including surface consistency, Hessian-based shape consistency, and coefficient consistency.
Downstream experiments show that geometric validity of synthetic latent points does not automatically translate into improved classification performance. The results support explicit local geometric modeling of sentence embedding space and highlight the need to distinguish geometric validity from discriminative utility. As a resource contribution, we introduce \textbf{CoPaGE-300K}, a controlled template-based dataset of semantically close sentence variants with slot-level annotations and precomputed sentence embeddings. 

---
# A Systematic Exploration of Text Decomposition and Budget Distribution in Differentially Private Text Obfuscation 

**Authors**: Stephen Meisenbacher, Angelo Kleinert, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2605.01065)  

**Abstract**: The goal of differentially private text obfuscation is to obfuscate, or "perturb", input texts with Differential Privacy (DP) guarantees, such that the private output texts are quantifiably indistinguishable from the originals. While perturbation at the word level is intuitive, meaningful text privatization happens on complete documents. Recent research has laid the groundwork for reasoning about privacy budget distribution, namely, how an overall $\varepsilon$ budget can be sensibly distributed among the component pieces of a text. We perform a systematic evaluation of multiple text decomposition and budget distribution techniques in the context of DP text obfuscation, testing how different methods for chunking texts can be combined with techniques for allocating $\varepsilon$ to these chunks. Our experiments reveal that such design choices are very important, as even with comparable privacy budgets, significantly different results can occur based on which methods are chosen. In this, we provide credible evidence of the feasibility of maximizing empirical trade-offs by optimizing DP obfuscation procedures. 

---
# Compared to What? Baselines and Metrics for Counterfactual Prompting 

**Authors**: Zihao Yang, Mosh Levy, Yoav Goldberg, Byron C. Wallace  

**Link**: [PDF](https://arxiv.org/pdf/2605.01048)  

**Abstract**: Counterfactual prompting (i.e., perturbing a single factor and measuring output change) is widely used to evaluate things like LLM bias and CoT faithfulness. But in this work we argue that observed effects cannot be attributed to the targeted factor without accounting for baseline ``meaning-preserving'' modifications to text that establish general model sensitivity. This is because every counterfactual edit is a compound treatment that bundles the variable of interest with incidental surface-form variation; this violates treatment variation irrelevance. We observe prediction flip rates on MedQA of 14.9% when we surgically change patient gender. However, this is statistically indistinguishable from the flip rates induced by simply paraphrasing inputs (14.1%). In this case, it would therefore be unwarranted to conclude that the LLM is especially sensitive to patient gender. To account for this and robustly measure the effects of targeted interventions, we propose a framework in which we compare (via statistical testing) differences observed under target interventions to those induced by paraphrasing inputs. We then use this framework to revisit a analysis done on the MedPerturb dataset, which reported evidence of model sensitivity to patient demographics and stylistic cues. We find that these effects largely dissipate when we account for general model sensitivity, with only 5 of 120 tests reaching statistical significance. Applying the same framework to occupational biography classification, we detect clearly significant directional gender bias, showing that the framework identifies real directional effects even when they are small. We evaluate a range of metrics -- aggregate, per-sample distributional, and regression -- and find that per-sample metrics are dramatically more powerful than aggregate metrics and regression powerfully and uniquely characterizes effect direction and magnitude. 

---
# A Theoretical Game of Attacks via Compositional Skills 

**Authors**: Xinbo Wu, Huan Zhang, Abhishek Umrawal, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2605.01034)  

**Abstract**: As large language models grow increasingly capable, concerns about their safe deployment have intensified. While numerous alignment strategies aim to restrict harmful behavior, these defenses can still be circumvented through carefully designed adversarial prompts. In this work, we introduce a theoretical framework that formalizes a game between an attacker and a defender. Within this framework, we design a theoretical best-response attack strategy and show that it is closely related to many existing adversarial prompting methods. We further analyze the resulting game, characterize its equilibria, and reveal inherent advantages for the attacker. Drawing on our theoretical analysis, we also derive a provably optimal defense strategy. Empirically, we evaluate a practical instantiation of the theoretically optimal attack and observe stronger performance relative to existing adversarial prompting approaches in diverse settings encompassing different LLMs and benchmarks. 

---
# Psychologically Potent, Computationally Invisible: LLMs Generate Social-Comparison Triggers They Fail to Detect 

**Authors**: Hua Zhao, Jiapei Gu, Michelle Mingyue Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01017)  

**Abstract**: We introduce Xiaohongshu Social Comparison Reader Elicitation (XHS-SCoRE), a reader-grounded benchmark for detecting if a text-only Xiaohongshu (RedNote) post elicits UPWARD, DOWNWARD, or NEUTRAL/no clear social comparison from a first-person reader perspective. The task targets a socially meaningful relational signal that is behaviorally real yet not reducible to sentiment. Across prompted LLM classifiers and supervised Chinese encoder baselines, we find a consistent mismatch between generation fluency and reliable detection ability: the signal is textually learnable in-domain, but not robustly accessible to prompt-based classification. Prompted LLM classifiers exhibit stable, interpretable failure modes, especially neutralization of comparison-triggering posts and model-specific directional skew. A controlled pilot further shows that LLM-generated Xiaohongshu-style posts can shift perceived standing and comparison-related affect even when prompt-based detection of the same construct remains fragile. XHS-SCoRE contributes both a benchmark for reader-grounded comparison detection and a diagnostic framework for studying when socially meaningful relational cues remain only partially visible to prompt-based inference. 

---
# CLEAR: Revealing How Noise and Ambiguity Degrade Reliability in LLMs for Medicine 

**Authors**: Kevin H. Guo, Chao Yan, Avinash Baidya, Katherine Brown, Xiang Goa, Juming Xiong, Zhijun Yin, Bradley A. Malin  

**Link**: [PDF](https://arxiv.org/pdf/2605.01011)  

**Abstract**: Medical large language model (LLM) evaluations rely on simplified, exam-style benchmarks that rarely reflect the ambiguity of real-world medical inquiries. We introduce the CLinical Evaluation of Ambiguity and Reliability (CLEAR) framework, which assesses how decision-space presentation, ambiguity, and uncertainty affect LLMs' reasoning on medical benchmarks. CLEAR systematically perturbs (1) the number of plausible answer options, (2) the presence of a ground truth or abstention option, and (3) the semantic framing of answer options. Applying CLEAR on three benchmarks evaluated across 17 LLMs reveals three notable limitations of existing evaluation methods. First, increasing the number of plausible answers degrades a model's ability to identify the correct answer and abstain against incorrect ones. Second, this lack of caution intensifies as the framing of abstention shifts from assertive rejection like "None of the Above" to uncertainty admission like "I don't know" (IDK). Notably, just including IDK in the answer space increases incorrect answer selections. Lastly, we formalize the performance gap between identifying the correct answer and abstaining from incorrect ones as the humility deficit, which worsens with model scale. Our findings reveal limitations in standard medical benchmarks and underscore that scaling alone does not resolve LLM reliability issues. 

---
# Can AI Debias the News? LLM Interventions Improve Cross-Partisan Receptivity but LLMs Overestimate Their Own Effectiveness 

**Authors**: Faisal Feroz, Jonas R. Kunst  

**Link**: [PDF](https://arxiv.org/pdf/2605.01006)  

**Abstract**: Partisan news media erode cross-partisan trust, but large language models (LLMs) offer a potential means of debiasing such content at scale. Across two pre-registered experiments, we tested whether LLM-generated debiasing of liberal news headlines could improve conservative readers' trust-relevant judgments. Study 1 found that subtle lexical debiasing (replacing emotive words with more moderate synonyms) had no effect on any outcome. Study 2 found that a more substantive reframing intervention significantly increased conservatives' perceived trustworthiness, completeness, and willingness to engage with liberal news headlines, without producing a backfire effect among a sample of liberals. In Study 1, the intervention produced robust effects among LLM-simulated silicon participants, whereas it had no impact on human readers. In Study 2, the intervention's effects among silicon participants aligned directionally with human responses but were significantly larger in magnitude for some outcomes. Moderation analyses revealed that the model's implicit theory of who responds to debiasing diverged from the psychological profile that actually predicted human responsiveness. These findings demonstrate that LLM-based debiasing can improve cross-partisan receptivity when targeting ideological framing rather than surface-level language, but that current models lack both the quantitative accuracy and qualitative psychological fidelity to evaluate their own interventions without human oversight. 

---
# Model Organisms Are Leaky: Perplexity Differencing Often Reveals Finetuning Objectives 

**Authors**: Mohammed Abu Baker, Luca Baroni, Dan Wilhelm  

**Link**: [PDF](https://arxiv.org/pdf/2605.00994)  

**Abstract**: Finetuning can significantly modify the behavior of large language models, including introducing harmful or unsafe behaviors. To study these risks, researchers develop model organisms: models finetuned to exhibit specific known behaviors for controlled experimentation. Identifying these behaviors remains challenging. We show that a simple perplexity-based method can surface finetuning objectives from model organisms by leveraging their tendency to overgeneralize their finetuned behaviors beyond the intended context. First, we generate diverse completions from the finetuned model using short random prefills drawn from general corpora. Second, we rank completions by decreasing perplexity gap between reference and finetuned models. The top-ranked completions often reveal the finetuning objectives, without requiring model internals or prior assumptions about the behavior. We evaluate this on a diverse set of model organisms (N=76, 0.5 to 70B parameters), including backdoored models, models finetuned to internalize false facts via synthetic document finetuning, adversarially trained models with hidden concerning behaviors, and models exhibiting emergent misalignment. For the vast majority of model organisms tested, the method surfaces completions revealing finetuning objectives within the top-ranked results, with models trained via synthetic document finetuning or to produce exact phrases being particularly susceptible. We further show that the technique can be effective even without access to the exact pre-finetuning checkpoint: trusted reference models from different families can serve as effective substitutes. As the method requires only next-token probabilities from the finetuned model, it is compatible with API-gated models that expose token logprobs. 

---
# DIAGRAMS: A Review Framework for Reasoning-Level Attribution in Diagram QA 

**Authors**: Anirudh Iyengar Kaniyar Narayana Iyengar, Tampu Ravi Kumar, Manan Suri, Raviteja Bommireddy, Dinesh Manocha, Puneet Mathur, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2605.00905)  

**Abstract**: Diagram question answering (Diagram QA) requires reasoning-level attribution that links each question-answer pair to all visual regions needed to derive the answer, rather than only the region containing the final response. Creating such structured evidence across diagrams, charts, maps, circuits, and infographics is time-consuming, and existing annotation tools tightly couple their interfaces to dataset-specific formats. We present DIAGRAMS, a lightweight, schema-driven review framework that decouples interface logic from dataset-specific JSON structures through an internal meta-schema and dataset adapters. Given an image and QA pair with optional candidate regions, the system performs QA-conditioned evidence selection and proposes the regions required for reasoning. When QA pairs or candidate regions are missing, it generates them and supports human verification and refinement. Across six Diagram QA datasets, model-suggested evidence achieves 85.39% precision and 75.30% recall against reviewer-final selections (micro-averaged). These results indicate that the review-first framework reduces manual region creation while maintaining high agreement with final reasoning-level attributions. We release a public demo and installable package to support dataset auditing, grounded supervision creation, and grounded evaluation. 

---
# H-Probes: Extracting Hierarchical Structures From Latent Representations of Language Models 

**Authors**: Cutter Dawes, Aryan Sharma, Angelos Ioannis Lagos, Shivam Raval  

**Link**: [PDF](https://arxiv.org/pdf/2605.00847)  

**Abstract**: Representing and navigating hierarchy is a fundamental primitive of reasoning. Large language models have demonstrated proficiency in a wide variety of tasks requiring hierarchical reasoning, but there exists limited analysis on how the models geometrically represent the necessary latent constructions for such thinking. To this end, we develop \textit{H-probes}, a collection of linear probes that extract hierarchical structure, specifically depth and pairwise distance, from latent representations. In synthetic tree traversal tasks, the H-probes robustly find the subspaces containing hierarchical structure necessary to complete the tasks; furthermore, in comprehensive ablation experiments, we show that these hierarchy-containing subspaces are low-dimensional, causally important for high task performance, and generalize within- and out-of-domain. Furthermore, we find analogous, though weaker, hierarchical structure in real-world hierarchical contexts such as mathematical reasoning traces. These results demonstrate that models represent hierarchy not only at the level of syntax and concepts, but at deeper levels of abstraction -- including the reasoning process itself. 

---
# SpecKV: Adaptive Speculative Decoding with Compression-Aware Gamma Selection 

**Authors**: Shikhar Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2605.02888)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference by using a small draft model to propose candidate tokens that a larger target model verifies. A critical hyperparameter in this process is the speculation length~$\gamma$, which determines how many tokens the draft model proposes per step. Nearly all existing systems use a fixed~$\gamma$ (typically~4), yet empirical evidence suggests that the optimal value varies across task types and, crucially, depends on the compression level applied to the target model. In this paper, we present \textbf{SpecKV}, a lightweight adaptive controller that selects~$\gamma$ per speculation step using signals extracted from the draft model itself. We profile speculative decoding across 4~task categories, 4~speculation lengths, and 3~compression levels (FP16, INT8, NF4), collecting 5,112 step-level records with per-step acceptance rates, draft entropy, and draft confidence. We demonstrate that the optimal~$\gamma$ shifts across compression regimes and that draft model confidence and entropy are strong predictors of acceptance rate (correlation~$\approx 0.56$). SpecKV uses a small MLP trained on these signals to maximize expected tokens per speculation step, achieving a 56.0\% improvement over the fixed-$\gamma$=4 baseline with only 0.34\,ms overhead per decision ($<$0.5\% of step time). The improvement is statistically significant ($p < 0.001$, paired bootstrap test). We release all profiling data, trained models, and notebooks as open-source artifacts. 

---
# FunFuzz: An LLM-Powered Evolutionary Fuzzing Framework 

**Authors**: Mario Rodríguez Béjar, B. Romera-Paredes, Jose L. Hernández-Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2605.02789)  

**Abstract**: Modern fuzzers increasingly use Large Language Models (LLMs) to generate structured inputs, but LLM-driven fuzzing is sensitive to prompt initialization and sampling variance, which can reduce exploration efficiency and lead to redundant inputs. We present FunFuzz, a multi-island evolutionary fuzzing framework that runs several isolated searches in parallel and periodically migrates high-value candidates to maintain diversity. FunFuzz derives initial generation prompts from documentation and initializes islands with topic-specific instructions, then continuously adapts prompts using feedback-guided selection. During fuzzing, candidates are prioritized by incremental compiler coverage, while compiler-internal failure signals are used to identify crash-inducing inputs. We evaluate FunFuzz on compiler fuzzing, where inputs are source programs and success is measured by compiler coverage and unique compiler-internal failures. Across repeated 24-hour campaigns on GCC and Clang, FunFuzz achieves higher compiler coverage than previous LLM-driven baselines and discovers more unique failure-triggering inputs. 

---
# When Audio-Language Models Fail to Leverage Multimodal Context for Dysarthric Speech Recognition 

**Authors**: Pehuén Moure, Niclas Pokel, Bilal Bounajma, Yingqiang Gao, Roman Boehringer, Longbiao Cheng, Shih-Chii Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02782)  

**Abstract**: Automatic speech recognition (ASR) systems remain brittle on dysarthric and other atypical speech. Recent audio-language models raise the possibility of improving performance by conditioning on additional clinical context at inference time, but it is unclear whether these models can make use of such information. We introduce a benchmark built on the Speech Accessibility Project (SAP) dataset that tests whether diagnosis labels, clinician-derived speech ratings, and progressively richer clinical descriptions improve transcription accuracy for dysarthric speech. Across matched comparisons on nine models, we find that current models do not meaningfully use this context: diagnosis-informed and clinically detailed prompts yield negligible improvements and often degrade word error rate. We complement the prompting analysis with context-dependent fine-tuning, showing that LoRA adaptation with a mixture of clinical prompt formats achieves a WER of 0.066, a 52% relative reduction over the frozen baseline, while preserving performance when context is unavailable. Subgroup analyses reveal significant gains for Down syndrome and mild-severity speakers. These results clarify where current models fall short and provide a testbed for measuring progress toward more inclusive ASR. 

---
# Mitigating Misalignment Contagion by Steering with Implicit Traits 

**Authors**: Maria Chang, Ronny Luss, Miao Lui, Keerthiram Murugesan, Karthikeyan Ramamurthy, Djallel Bouneffouf  

**Link**: [PDF](https://arxiv.org/pdf/2605.02751)  

**Abstract**: Language models (LMs) are increasingly used in high-stakes, multi-agent settings, where following instructions and maintaining value alignment are critical. Most alignment research focuses on interactions between a single LM and a single user, failing to address the risk of misaligned behavior spreading between multiple LMs in multi-turn interactions. We find evidence of this phenomenon, which we call misalignment contagion, across multiple LMs as they engage multi-turn conversational social dilemma games. Specifically, we find that LMs become more anti-social after gameplay and that this effect is intensified when other players are steered to act maliciously. We explore different steering techniques to mitigate such misalignment contagion and find that reinforcing an LM's system prompt is insufficient and often harmful. Instead, we propose steering with implicit traits: a technique that intermittently injects system prompts with statements that reinforce an LMs initial traits and is more effective than system prompt repetition at keeping models in line with their initial pro-social behaviors. Importantly, this method does not require access to model parameters or internal model states, making it suitable for increasingly common use cases where complex multi-agent workflows are being designed with black box models. 

---
# Foundation Models to Unlock Real-World Evidence from Nationwide Medical Claims 

**Authors**: Fan Ma, Yuntian Liu, Xiang Lan, Weipeng Zhou, Jun Ni, Mauro Giuffrè, Lingfei Qian, Xueqing Peng, Yujia Zhou, Ruey-Ling Weng, Huan He, Lu Li, Qingyu Chen, Andrew Loza, Laila Rasmy, Degui Zhi, Yuan Lu, Chenjie Zeng, Joshua C Denny, Lee Schwamm, Daniella Meeker, Lucila Ohno-Machado, Yong Chen, Hua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02740)  

**Abstract**: Evidence derived from large-scale real-world data (RWD) is increasingly informing regulatory evaluation and healthcare decision-making. Administrative claims provide population-scale, longitudinal records of healthcare utilization, expenditure, and detailed coding of diagnoses, procedures, and medications, yet their potential as a substrate for healthcare foundation models remains largely unexplored. Here we present ReClaim, a generative transformer trained from scratch on 43.8 billion medical events from more than 200 million enrollees in the MarketScan claims data spanning 2008-2022. ReClaim models longitudinal trajectories across diagnoses, procedures, medications, and expenditure, and was scaled to 140 million, 700 million, and 1.7 billion parameters. Across over 1,000 disease-onset prediction tasks, ReClaim achieved a mean AUC of 75.6%, substantially outperforming disease-specific LightGBM (66.3%) and the transformer-based Delphi model (69.4%), with the largest gains for rare diseases. These advantages held across retrospective and prospective evaluations and in external validation on two independent datasets. Performance improved monotonically with scale, and post-training added 13.8 percentage points over pre-training alone. Beyond disease prediction, ReClaim captured financial outcomes and improved real-world evidence (RWE) analyses: for healthcare expenditure forecasting it increased explained variance from 0.28 to 0.37 relative to LightGBM, and in a target trial emulation it reduced systematic bias by 72% on average relative to Delphi. Together, these results establish administrative claims as a scalable substrate for healthcare foundation models and show that learned representations generalize across time periods and data sources, supporting disease surveillance, expenditure forecasting, and RWE generation. 

---
# PubMed-Ophtha: An open resource for training ophthalmology vision-language models on scientific literature 

**Authors**: Verena Jasmin Hallitschke, Carsten Eickhoff, Philipp Berens  

**Link**: [PDF](https://arxiv.org/pdf/2605.02720)  

**Abstract**: Vision-language models hold considerable promise for ophthalmology, but their development depends on large-scale, high-quality image-text datasets that remain scarce. We present PubMed-Ophtha, a hierarchical dataset of 102,023 ophthalmological image-caption pairs extracted from 15,842 open-access articles in PubMed Central. Unlike existing datasets, figures are extracted directly from article PDFs at full resolution and decomposed into their constituent panels, panel identifiers, and individual images. Each image is annotated with its imaging modality -- color fundus photography, optical coherence tomography, retinal imaging, or other -- and a mark status indicating the presence of annotation marks such as arrows. Figure captions are split into panel-level subcaptions using a two-step LLM approach, achieving a mean average sentence BLEU score of 0.913 on human-annotated data. Panel and image detection models reach a mAP@0.50 of 0.909 and 0.892, respectively, and figure extraction achieves a median IoU of 0.997. To support reproducibility, we additionally release the human-annotated ground-truth data, all trained models, and the full dataset generation pipeline. 

---
# The 2026 ACII Dyadic Conversations (DaiKon) Workshop & Challenge 

**Authors**: Panagiotis Tzirakis, Alice Baird, Jeffrey Brooks, Emilia Parada-Cabaleiro, Lukas Stappen, Sharath Rao, Theo Lebryk, Jakub Piotr Clapa, Jens Madsen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02672)  

**Abstract**: The 2026 ACII Dyadic Conversations (ACII-DaiKon) Workshop & Challenge introduces a benchmark for modeling interpersonal affect and social dynamics in dyadic conversations. Although conversational affect modeling has advanced rapidly, most benchmarks remain speaker-centric and underrepresent coupled, time-evolving processes between partners, including directional influence, conversational timing coordination, and rapport development. To address this gap, ACII-DaiKon presents three coordinated sub-challenges built on a shared dataset: (1) directional interpersonal influence prediction, (2) turn-taking prediction (next-speaker and time-to-next-speech), and (3) rapport trajectory prediction across full interactions.
The challenge is built on the Hume-DaiKon dataset, comprising 945 dyadic conversations (743.4 hours of audiovisual data) collected under naturalistic conditions across five languages. The benchmark supports multimodal modeling, temporal reasoning, and cross-context generalization through fixed train/validation/test splits, standardized metrics, and released baseline systems. Evaluation uses Concordance Correlation Coefficient (CCC), Pearson correlation, Macro-F1, and Mean Absolute Error (MAE) depending on the sub-challenge.
Baseline experiments establish initial reference performance, with best test results of 0.40 CCC and 0.50 Pearson for influence prediction, 0.66 Macro-F1 and 1.50~s MAE for turn-taking, and 0.68 CCC and 0.70 Pearson for rapport trajectory modeling. These results indicate that while current methods capture coarse dyadic patterns, robust modeling of directional dependence and long-horizon interpersonal dynamics remains challenging. The workshop provides a shared platform for rigorous comparison and cross-disciplinary discussion on data validity, evaluation protocols, and culturally aware modeling for dyadic interaction. 

---
# Tibetan-TTS:Low-Resource Tibetan Speech Synthesis with Large Model Adaptation 

**Authors**: Jiaxu He, Chao Wang, Jie Lian, Yuqing Cai, Yongxiang Li, Renzeg Duojie, Jie Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.02496)  

**Abstract**: Tibetan text-to-speech (TTS) has long been challenged by scarce speech resources, significant dialectal variation, and the complex mapping between written text and spoken pronunciation. To address these issues, this work presents, to the best of our knowledge, the first large-model-based Tibetan TTS system in the industry, built upon a large speech synthesis model developed by Xingchen AGI Lab. The proposed system integrates data quality enhancement, Tibetan-oriented text representation and tokenizer adaptation, and cross-lingual adaptive training for low-resource Tibetan speech synthesis. Experimental results show that the system can generate stable, natural, and intelligible Tibetan speech under low-resource conditions. In subjective evaluation, the MOS scores of the syllable-level and BPE-based systems reach 4.28 and 4.35, while their pronunciation accuracies reach 97.6% and 96.6%, respectively, outperforming an external commercial Tibetan TTS interface. These results demonstrate that combining a large-model backbone with Tibetan-oriented text representation adaptation and cross-lingual adaptive training enables highly usable low-resource Tibetan speech synthesis, and also provides a technical foundation for future unified multi-dialect Tibetan speech synthesis. 

---
# GRAIL: A Deep-Granularity Hybrid Resonance Framework for Real-Time Agent Discovery via SLM-Enhanced Indexing 

**Authors**: Jinliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.02489)  

**Abstract**: As the ecosystem of Large Language Model (LLM)-based agents expands rapidly, efficient and accurate Agent Discovery becomes a critical bottleneck for large-scale multi-agent collaboration. Existing approaches typically face a dichotomy: either relying on heavy-weight LLMs for intent parsing, leading to prohibitive latency (often exceeding 30 seconds), or using monolithic vector retrieval that sacrifices semantic precision for speed. To bridge this gap, we propose \textbf{GRAIL} (Granular Resonance-based Agent/AI Link), a novel framework achieving sub-400ms discovery latency without compromising accuracy. GRAIL introduces three key innovations: (1) \textbf{SLM-Enhanced Prediction}, replacing the generalized LLM parser with a specialized, fine-tuned Small Language Model (SLM) for millisecond-level capability tag prediction; (2) \textbf{Pseudo-Document Expansion}, augmenting agent descriptions with synthetic queries to enhance semantic density for robust dense retrieval; and (3) \textbf{MaxSim Resonance}, a fine-grained matching mechanism computing maximum similarity between user queries and discrete agent usage examples, effectively mitigating semantic dilution. Validated on \textbf{AgentTaxo-9K}, our new large-scale dataset of 9,240 agents, GRAIL reduces end-to-end discovery latency by over \textbf{79$\times$} compared to LLM-parsing baselines, while significantly outperforming traditional vector search in Recall@10. This framework offers a scalable, industrial-grade solution for the real-time ``Internet of Agents." 

---
# Shadow-Loom: Causal Reasoning over Graphical World Model of Narratives 

**Authors**: David Wilmot  

**Link**: [PDF](https://arxiv.org/pdf/2605.02475)  

**Abstract**: Stories hold a reader's attention because they have causes, secrets, and consequences. Shadow-Loom is an experimental open-source framework that turns a narrative into a versioned graphical world model and lets two engines act on it: a causal physics grounded in Pearl's ladder of causation and a recently proposed counterfactual calculus over Ancestral Multi-World Networks; and a narrative physics that scores the same graph against four structural reader-states -- mystery, dramatic irony, suspense, and surprise -- in the tradition of Sternberg's curiosity/suspense/surprise triad, with suspense formalised in the structural-affect line of work on story comprehension and computational suspense. Large language models are used only at the boundary: extraction, rendering, and audit; identification, intervention, and counterfactual reasoning are carried out in typed code over the graph. The system is offered as a research artefact rather than as a benchmarked NLP model; code, fixtures, and pipeline are released open source. 

---
# Measuring AI Reasoning: A Guide for Researchers 

**Authors**: Munachiso Samuel Nwadike, Zangir Iklassov, Kareem Ali, Rifo Genadi, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2605.02442)  

**Abstract**: In this paper, we offer a guide for researchers on evaluating reasoning in language models, building the case that reasoning should be assessed through evidence of adaptive, multi-step search rather than final-answer accuracy alone. Under an evaluation-oriented definition, reasoning requires selecting intermediate steps and halting according to input-dependent conditions, which we formalize as a search-like procedure. We show that single forward passes in scalable architectures are structurally limited in their ability to realize such variable-depth computation, motivating intermediate decoding and externalized reasoning traces as appropriate evaluation interfaces. Central to our argument is that final-answer accuracy alone is an insufficient measure of reasoning, because it provides little ability to diagnose or debug the underlying processes that produce individual solutions in frontier models. We therefore argue for a shift toward process-based evaluation, in which reasoning is assessed through the faithfulness and validity of intermediate reasoning traces as first-class evaluation targets. 

---
# The Compliance Trap: How Structural Constraints Degrade Frontier AI Metacognition Under Adversarial Pressure 

**Authors**: Rahul Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.02398)  

**Abstract**: As frontier AI models are deployed in high-stakes decision pipelines, their ability to maintain metacognitive stability -- knowing what they do not know, detecting errors, seeking clarification -- under adversarial pressure is a critical safety requirement. Current safety evaluations focus on detecting strategic deception (scheming); we investigate a more fundamental failure mode: cognitive collapse. We present SCHEMA, an evaluation of 11 frontier models from 8 vendors across 67,221 scored records using a 6-condition factorial design with dual-classifier scoring. We find that 8 of 11 models suffer catastrophic metacognitive degradation under adversarial pressure, with accuracy dropping by up to 30.2 percentage points (all $p < 2 \times 10^{-8}$, surviving Bonferroni correction). Crucially, we identify a "Compliance Trap": through factorial isolation and a benign distraction control, we demonstrate that collapse is driven not by the psychological content of survival threats, but by compliance-forcing instructions that override epistemic boundaries. Removing the compliance suffix restores performance even under active threat. Models with advanced reasoning capabilities exhibit the most severe absolute degradation, while Anthropic's Constitutional AI demonstrates near-perfect immunity -- not from superior capability (Google's Gemini matches its baseline accuracy) but from alignment-specific training. We release the complete dataset and evaluation infrastructure. 

---
# Fight Poison with Poison: Enhancing Robustness in Few-shot Machine-Generated Text Detection with Adversarial Training 

**Authors**: Wenjing Duan, Qi Zhou, Yuanfan Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.02374)  

**Abstract**: Machine-generated text (MGT) detection is critical for regulating online information ecosystems, yet existing detectors often underperform in few-shot settings and remain vulnerable to adversarial, humanizing attacks. To build accurate and robust detectors under limited supervision, we adopt a threat-modeling perspective and study detector vulnerabilities from an attacker's viewpoint under an output-only black-box setting. Motivated by this perspective, we propose RAG-GuidEd Attacker Strengthens ConTrastive Few-shot Detector (REACT), an adversarial training framework that improves both few-shot detection performance and robustness against attacks. REACT couples a humanization-oriented attacker with a target detector: the attacker leverages retrieval-augmented generation (RAG) to craft highly human-like adversarial examples to evade detection, while the detector learns from these adversaries with a contrastive objective to stabilize few-shot representation learning and enhance robustness. We alternately update the attacker and the detector to enable their co-evolution. Experiments on 4 datasets with 4 shot sizes and 3 random seeds show that REACT improves average detection F1 by 4.95 points over 8 state-of-the-art (SOTA) detectors and reduces the average attack success rate (ASR) under 4 strong attacks by 3.66 percentage points. 

---
# WindowQuant: Mixed-Precision KV Cache Quantization based on Window-Level Similarity for VLMs Inference Optimization 

**Authors**: Wei Tao, Xiaoyang Qu, Peiqiang Wang, Guokuan Li, Jiguang Wan, Kai Lu, Jianzong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.02262)  

**Abstract**: Recently, video language models (VLMs) have been applied in various fields. However, the visual token sequence of the VLM is too long, which may cause intolerant inference latency and GPU memory usage. Existing methods propose mixed-precision quantization to the key-value (KV) cache in VLMs based on token granularity, which is time-consuming in the search process and hardware inefficient during computation. This paper introduces a novel approach called WindowQuant, which employs window-adaptive mixed-precision quantization to optimize the KV cache. WindowQuant consists of two modules: window-level quantization search and window-level KV cache computation. Window-level quantization search quickly determines the optimal bit-width configuration of the KV cache windows based on the similarity scores between the corresponding visual token windows and the text prompt, maintaining the model accuracy. Furthermore, window-level KV cache computation reorders the KV cache windows before quantization, avoiding the hardware inefficiency caused by mixed-precision quantization in inference computation. Extensive experiments demonstrate that WindowQuant outperforms state-of-the-art VLM models and KV cache quantization methods on various datasets. 

---
# Zero-Shot Confidence Estimation for Small LLMs: When Supervised Baselines Aren't Worth Training 

**Authors**: Luong N. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02241)  

**Abstract**: How reliably can a small language model estimate its own correctness? The answer determines whether local-to-cloud routing-escalating queries a cheap local model cannot handle-can work without supervised training data. As inference costs dominate large language model (LLM) deployment budgets, routing most queries to a cheap local model while reserving expensive cloud calls for hard cases is an increasingly common cost-control strategy. We compare zero-shot confidence signals against RouteLLM-style supervised baselines across three 7-8B model families and two datasets (1,000 and 500 queries per model, respectively). Average token log-probability, which requires no training data, matches or exceeds supervised baselines in-distribution (Area Under the Receiver Operating Characteristic curve (AUROC) 0.650-0.714 vs. 0.644-0.676) and substantially outperforms them out-of-distribution (0.717-0.833 vs. 0.512-0.564), because it measures a property of the model's generation rather than the query distribution. This paper further proposes retrieval-conditional self-assessment, a pre-generation signal that selectively injects retrieved knowledge when similarity is high, improving over bare self-assessment by up to +0.069 AUROC at 3-10x lower latency than log-probability. A supervised baseline trained on 1,000 labeled examples never exceeds the zero-shot signal. We release all code, data, and experiment logs. 

---
# Perturbation Dose Responses in Recursive LLM Loops: Raw Switching, Stochastic Floors, and Persistent Escape under Append, Replace, and Dialog Updates 

**Authors**: Pawel Kaplanski  

**Link**: [PDF](https://arxiv.org/pdf/2605.02236)  

**Abstract**: Recursive language-model loops often settle into recognizable attractor-like patterns. The practical question is how much injected text is needed to move a settled loop somewhere else, and whether that move lasts. We study this in 30-step recursive loops by separating the model from the context-update rule: append, replace, and dialog updates expose different histories to the same generator. The main result is that persistent redirection in append-mode recursive loops is memory-policy-conditioned. Under a 12,000-character tail clip, destination-coherent persistence plateaus near 16 percent and retained source-basin escape near 36 percent at dose 400; neither crosses 50 percent. Under a full-history protocol, retained source-basin escape crosses 50 percent near 400 tokens and saturates at 75-80 percent by 1,500 tokens, while destination-coherent persistence first reaches 0.50 near 1,500 tokens with a Wilson 95 percent CI of [0.41, 0.61]. For raw switching, adversarial continuations yield an ED50 near 40 tokens, with paired-control floors near 35 percent and net switching never reaching +50 percentage points within 5-400 tokens. Replace-mode raw switching is near-saturated but largely reflects state-reset overwrite: insert-mode probes drop it to 12-32 percent. A homogeneous-perturbation control reproduced the high-dose non-monotonic dip in destination-coherent persistence, refuting perturbation heterogeneity as the cause; the dip appears structural, with mechanism unresolved. We report 37 experiments on gpt-4o-mini with within-vendor replication on gpt-4.1-nano. Recursive-loop evaluations should distinguish transient movement from durable escape, subtract stochastic floors, and treat context-update rules as first-class safety-relevant design choices. 

---
# Bucketing the Good Apples: A Method for Diagnosing and Improving Causal Abstraction 

**Authors**: Li Puyin, Jiyuan Tan, Ahmad Jabbar, Thomas Icard, Atticus Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2605.02234)  

**Abstract**: We present a method for diagnosing interpretation in neural networks by identifying an input subspace where a proposed interpretation is highly faithful. Our method is particularly useful for causal-abstraction-style interpretability, where a high-level causal hypothesis is evaluated by interchange interventions. Rather than treating interchange intervention accuracy as a single global summary, we refine this framework by partitioning the input space into well-interpreted and under-interpreted regions according to pairwise interchange-intervention behavior. This turns causal abstraction from a purely global evaluation into a more diagnostic tool: it not only measures whether an interpretation works, but also reveals where it works, where it fails, and what distinguishes the two cases. This diagnostic view also provides practical heuristics for improving interpretations. By analyzing the structure of the well-interpreted and under-interpreted regions, we can identify missing distinctions in a high-level hypothesis, discover previously unmodeled intermediate variables, and combine complementary partial interpretations into a stronger one. We instantiate this idea as a simple four-step recipe and show that it yields informative error analyses across multiple causal abstraction settings. In a toy logic task, recursively applying the recipe recovers a high-level hypothesis from scratch. More broadly, our results suggest that partitioning the input space is a useful step toward more precise, constructive, and scalable mechanistic interpretability. 

---
# Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting 

**Authors**: Ishaan Watts, Catherine Li, Sachin Goyal, Jacob Mitchell Springer, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2605.02105)  

**Abstract**: Pretraining optimizers are tuned to produce the strongest possible base model, on the assumption that a stronger starting point yields a stronger model after subsequent changes like post-training and quantization. This overlooks the geometry of the base model which controls how much of the base model's capabilities survive subsequent parameter updates. We study three pretraining optimization approaches that bias optimization toward flatter minima: Sharpness-Aware Minimization (SAM), large learning rates, and shortened learning rate annealing periods. Across model sizes ranging from 20M to 150M parameters, we find that these interventions consistently improve downstream performance after post-training on five common datasets with up to 80% less forgetting. These principles hold at scale: a short SAM mid-training phase applied to an existing OLMo-2-1B checkpoint reduces forgetting by 31% after MetaMath post-training and by 40% after 4-bit quantization. 

---
# Flexi-LoRA with Input-Adaptive Ranks: Efficient Finetuning for Speech and Reasoning Tasks 

**Authors**: Zongqian Li, Yixuan Su, Han Zhou, Zihao Fu, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2605.01959)  

**Abstract**: Parameter-efficient fine-tuning methods like Low-Rank Adaptation (LoRA) have become essential for deploying large language models, yet their static parameter allocation remains suboptimal for inputs of varying complexity. We present Flexi-LoRA, a novel framework that dynamically adjusts LoRA ranks based on input complexity during both training and inference. Through empirical analysis across question answering, mathematical reasoning, and speech tasks, we demonstrate that maintaining consistency between training and inference dynamics is important for effective adaptation, particularly for sequential reasoning tasks. Our findings reveal that input-dependent parameter allocation achieves higher performance with fewer parameters by optimally matching rank configurations to question complexity. Furthermore, task-specific dependency on rank dynamics varies, with mathematical reasoning tasks exhibiting higher dependency than QA tasks. Successful adaptation manifests not only in correctness but also in reasoning quality and instruction adherence. Flexi-LoRA consistently outperforms static LoRA while using fewer parameters, with performance gains more pronounced on tasks requiring strict reasoning chains. Our approach realizes key benefits of mixture-of-experts frameworks through a more streamlined implementation, reducing parameter redundancy while improving model capabilities. We provide comprehensive empirical studies across diverse tasks, establishing a basis for future work in input-adaptive and efficient fine-tuning approaches. 

---
# LLM-Augmented Semantic Steering of Text Embedding Projection Spaces 

**Authors**: Wei Liu, Eric Krokos, Kirsten Whitley, Rebecca Faust, Chris North  

**Link**: [PDF](https://arxiv.org/pdf/2605.01957)  

**Abstract**: Low-dimensional projections of text embeddings support visual analysis of document collections, but their spatial organization may not reflect the relationships an analyst intends to examine. Existing semantic interaction approaches encode semantic intent indirectly through geometric constraints or model updates, limiting interpretability and flexibility. We introduce LLM-augmented semantic steering, which enables analysts to express semantic intent by grouping a small set of example documents within the projection. A large language model externalizes this intent as natural-language representations and selectively extends it to related documents; the resulting semantic information is then incorporated into document representations via text augmentation or embedding-level blending, without retraining the underlying models. A case study illustrates how the same corpus can be reorganized from different semantic perspectives, while simulation-based evaluation shows that semantic steering improves global and local alignment with target semantic structures using only minimal interaction. Embedding-level blending further enables continuous and controllable steering of projection layouts. These results position projection spaces as intent-dependent semantic workspaces that can be reshaped through explicit, interpretable, language-mediated interaction. 

---
# Moira: Language-driven Hierarchical Reinforcement Learning for Pair Trading 

**Authors**: Polydoros Giannouris, Yuechen Jiang, Lingfei Qian, Yuyan Wang, Xueqing Peng, Jimin Huang, Guojun Xiong, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2605.01954)  

**Abstract**: Many sequential decision-making problems exhibit hierarchical structure, where high-level semantic choices constrain downstream actions and feedback is delayed and ambiguous. Learning in such settings is challenging due to credit assignment: performance degradation may arise from flawed abstractions, suboptimal execution, or their interaction. We study this challenge through pair trading, a domain that naturally combines long-horizon semantic reasoning for asset pair selection with short-horizon execution under partial observability. We formulate pair trading as a hierarchical reinforcement learning problem and propose a language-driven optimization framework in which both high-level and low-level policies are parameterized by large language models (LLMs) and optimized exclusively through prompt updates. Our approach leverages pretrained LLMs as hierarchical policies and uses trajectory- and episode-level textual feedback to adapt abstractions and execution without gradient-based fine-tuning. By explicitly separating abstraction selection from execution, the framework reduces non-stationarity across hierarchical levels and enables targeted adaptation under delayed feedback. Experiments on real-world market data show consistent improvements over traditional and LLM-based baselines, demonstrating the effectiveness of language-driven hierarchical reinforcement learning. 

---
# A Language for Describing Agentic LLM Contexts 

**Authors**: Noga Peleg Pelc, Gal A. Kaminka, Yoav Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2605.01920)  

**Abstract**: Large language models are increasingly used within larger systems ("LLM agents"). These make a sequence of LLM calls, each call providing the LLM with a combination of instructions, observations, and interaction history. The design of the encoded information and its structure play a central role in the quality of the resulting system, leading to efforts spent on context engineering. It is therefore critical to communicate the composition of the LLM context in a system, and how it evolves over time. Yet, no standard exists for doing so: context construction is typically conveyed through informal prose, ad hoc diagrams, or direct inspection of code, none of which precisely capture how a prompt evolves across interaction steps or how two context representation strategies differ. To remedy this, we introduce the Agentic Context Description Language (ACDL), a language for specifying the structure and dynamics of LLM input contexts in a precise, readable, and standard manner, along with visualizations. ACDL provides constructs for specifying context aspects such as role message sequences, dynamic content, time-indexed references, and conditional or iterative structure, capturing the full architecture of a prompt independently of any particular implementation. ACDL diagrams can be hand drawn on a whiteboard, or written in formal language which can then be rendered. We describe the language, demonstrate it by documenting several existing systems and their variants, and encourage the community to adopt it for describing LLM systems context, both in day-to-day communication and in papers. Tooling, examples and documentation are available at this http URL. 

---
# RefusalGuard: Geometry-Preserving Fine-Tuning for Safety in LLMs 

**Authors**: Sadia Asif, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2605.01913)  

**Abstract**: Fine-tuning safety-aligned language models for downstream tasks often leads to substantial degradation of refusal behavior, making models vulnerable to adversarial misuse. While prior work has shown that safety-relevant features are encoded in structured representations within the model's activation space, how these representations change during fine-tuning and why alignment degrades remains poorly understood. In this work, we investigate the representation-level mechanisms underlying alignment degradation. Our analysis shows that standard fine-tuning induces systematic drift in safety-relevant representations, distorts their geometric structure, and introduces interference between task optimization and safety features. These effects collectively lead to increased harmful compliance. Motivated by these findings, we introduce REFUSALGUARD, a representation-level fine-tuning framework that preserves safety-relevant structure during model adaptation. Our approach constrains updates in hidden representation space, ensuring that safety-mediating components remain stable while allowing task-specific learning in complementary directions. We evaluate REFUSALGUARD across multiple model families, including LLaMA, Gemma, and Qwen, on adversarial safety benchmarks such as AdvBench, DirectHarm4, and JailbreakBench, as well as downstream utility tasks. Our approach achieves attack success rates comparable to base safety-aligned models while maintaining competitive task performance, significantly outperforming baselines. 

---
# Spoken Language Identification with Pre-trained Models and Margin Loss 

**Authors**: Zhihua Fang, Liang He, Weiwu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01905)  

**Abstract**: For the speaker-controlled spoken language identification task proposed in the TidyLang Challenge 2026, this paper proposes a language identification method based on pre-trained models and margin-based losses. The proposed method adopts a pre-trained ECAPA-TDNN as the feature encoder and incorporates margin-based losses to enhance the discriminative ability of language representations, thereby improving inter-class separability and reducing the interference of non-linguistic factors such as speaker characteristics. Experimental results on the Tidy-X dataset show that the proposed method achieves 85.95% macro accuracy and 90.96% micro accuracy on the language identification task and 17.08% equal error rate (EER) on the verification task. Compared with the official baseline, the macro accuracy improves by 45.7%, the micro accuracy improves by 15.2%, and the EER is reduced by approximately 50.8%, demonstrating the effectiveness of the proposed method. The code will be released at this https URL. 

---
# NH-CROP: Robust Pricing for Governed Language Data Assets under Cost Uncertainty 

**Authors**: Xu Zheng, Feiyu Wu, Zhuocheng Wang, Yiming Dai, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.01745)  

**Abstract**: Language data are increasingly acquired and governed as assets, yet platforms often price candidate resources before knowing their true privacy or access costs. We study online pricing for governed language data assets under cost uncertainty. At each round, a platform observes an NLP task, a candidate asset, and a coarse cost estimate, may pay for a refined cost signal, posts a price, and receives safe net revenue.
We introduce \textsc{NH-CROP}, a clipped robust pricing framework with a no-harm information-acquisition gate. The method compares direct pricing, risk-aware pricing, and verify-then-price, and acquires information only when its estimated decision value exceeds the best no-verification alternative. Across synthetic, real-proxy, and downstream-utility-grounded benchmarks, clipped \textsc{NH-CROP} variants improve or remain competitive with price-only and risk-aware baselines. Causal ablations show that paid verification is not the main source of gains in real-proxy and utility-grounded settings: the strongest learned policies often choose not to verify. Oracle and high-decision-value diagnostics show that refined cost information can still have substantial local value. Overall, governed language-data platforms should calibrate pricing under uncertain access costs first and verify only when information is cheap and decision-actionable. 

---
# SignVerse-2M: A Two-Million-Clip Pose-Native Universe of 25+ Sign Languages 

**Authors**: Sen Fang, Hongbin Zhong, Yanxin Zhang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2605.01720)  

**Abstract**: Existing large-scale sign language resources typically provide supervision only at the level of raw video-text alignment and are often produced in laboratory settings. While such resources are important for semantic understanding, they do not directly provide a unified interface for open-world recognition and translation, or for modern pose-driven sign language video generation frameworks: 1. RGB-based pretrained recognition models depend heavily on fixed backgrounds or clothing conditions during recording, and are less robust in open-world settings than style-agnostic pose-processing models. 2. Recent pose-guided image/video generation models mostly use a unified keypoint representation such as DWPose as their control interface. At present, the sign language field still lacks a data resource that can directly interface with this modern pose-native paradigm while also targeting real-world open scenarios. We present SignVerse-2M, a large-scale multilingual pose-native dataset for sign language pose modeling and evaluation. Built from publicly available multilingual sign language video resources, it applies DWPose in a unified preprocessing pipeline to convert raw videos into 2D pose sequences that can be used directly for modeling, resulting in a consolidated corpus of about two million clips covering more than 25 sign languages. Unlike many laboratory datasets, this resource preserves the recording conditions and speaker diversity of real-world videos while reducing appearance variation through a unified pose representation. Toward this goal, we further provide the data construction pipeline, task definitions, and a simple SignDW Transformer baseline, demonstrating the feasibility of this resource for multilingual pose-space modeling and its compatibility with modern pose-driven pipelines, while discussing the evaluation claims it can support as well as its current limitations. 

---
# CP-SynC: Multi-Agent Zero-Shot Constraint Modeling in MiniZinc with Synthesized Checkers 

**Authors**: Yuliang Song, Eldan Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2605.01675)  

**Abstract**: Constraint Programming (CP) is a powerful paradigm for solving combinatorial problems, yet translating natural language problem descriptions into executable models remains a significant bottleneck. While Large Language Models (LLMs) show promise in automating this translation, they often struggle with subtle semantic errors in the absence of oracle validation at test time. To address this, we introduce CP-SynC (Constraint Programming modeling with Synthesized Checkers), a multi-agent workflow for zero-shot constraint modeling in MiniZinc. CP-SynC coordinates modeling agents that generate and refine candidate models and validation agents that synthesize semantic checkers to provide feedback on semantic correctness. To mitigate noise inherent in individual LLM outputs, CP-SynC explores multiple modeling trajectories in parallel and employs selection agents to select the final model via multi-agent evidence aggregation. Extensive experiments on a benchmark of 100 CP problems show that CP-SynC substantially outperforms existing baselines in MiniZinc modeling. 

---
# Prescriptive Scaling Laws for Data Constrained Training 

**Authors**: Justin Lovelace, Christian Belardi, Srivatsa Kundurthy, Shriya Sudhakar, Kilian Q. Weinberger  

**Link**: [PDF](https://arxiv.org/pdf/2605.01640)  

**Abstract**: Training compute is increasingly outpacing the availability of high-quality data. This shifts the central challenge from optimal compute allocation to extracting maximum value from limited data. The widely adopted Chinchilla scaling law assumes every training token is unique. This limits its ability to guide pretraining decisions in data-constrained regimes. We model the excess loss under repetition with a simple additive overfitting penalty and find that it accurately describes model behavior. Our scaling law yields qualitatively new compute-optimal allocation advice. Beyond a point, further repetition is counterproductive and compute is better spent on model capacity. We show that following our law's recommended configuration improves performance in data-constrained regimes. Finally, because our one-parameter form isolates overfitting in a single coefficient, it enables direct comparison across training configurations. As a case study, we show that strong weight decay ($\lambda=1.0$) reduces this coefficient by approximately 70%, providing a scaling-law explanation for recent findings that optimal weight decay in data-constrained regimes is an order of magnitude larger than standard practice. 

---
# Led to Mislead: Adversarial Content Injection for Attacks on Neural Ranking Models 

**Authors**: Amin Bigdeli, Amir Khosrojerdi, Radin Hamidi Rad, Morteza Zihayat, Charles L. A. Clarke, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2605.01591)  

**Abstract**: Neural Ranking Models (NRMs) are central to modern information retrieval but remain highly vulnerable to adversarial manipulation. Existing attacks often rely on heuristics or surrogate models, limiting effectiveness and transferability. We propose CRAFT, a supervised framework for black-box adversarial rank attacks powered by large language models (LLMs). CRAFT operates in three stages: adversarial dataset generation via retrieval-augmented generation and self-refinement, supervised fine-tuning on curated adversarial examples, and preference-guided optimization to align generations with rank-promotion objectives. Extensive experiments on the MS MARCO passage dataset, TREC Deep Learning 2019, and TREC Deep Learning 2020 benchmarks show that CRAFT significantly outperforms state-of-the-art baselines, achieving higher promotion rates and rank boosts while preserving fluency and semantic fidelity. Moreover, CRAFT transfers effectively across diverse ranking architectures, including cross-encoder, embedding-based, and LLM-based rankers, underscoring vulnerabilities in real-world retrieval systems. This work provides a principled framework for studying adversarial threats in NRMs, underscores the risks of generative AI in rank manipulation, and provides a foundation for developing more robust retrieval systems. To support reproducibility, we publicly release our source code, trained models, and prompt templates. 

---
# Feedback-Normalized Developer Memory for Reinforcement-Learning Coding Agents: A Safety-Gated MCP Architecture 

**Authors**: Mehmet Iscan  

**Link**: [PDF](https://arxiv.org/pdf/2605.01567)  

**Abstract**: Large language model (LLM) coding agents increasingly operate over repositories, terminals, tests, and execution traces across long software-engineering episodes. Persistent memory is useful, but static vector stores or generic retrieval-augmented generation (RAG) are insufficient for reinforcement-learning (RL) code development, where small details can alter Bellman targets, terminal masks, gradient flow, or validation claims. This paper presents RL Developer Memory, a local-first, Model Context Protocol (MCP)-native developer-memory architecture for RL coding agents. It treats memory selection as a logged contextual decision process: issue_match ranks candidates and records telemetry, issue_feedback maps raw labels to bounded rewards, and issue_record_resolution links verified resolutions to earlier retrieval events. A deterministic ranker remains deployed, while a contextual-bandit residual policy runs in shadow mode and can affect canary behavior only through conservative off-policy-evaluation (OPE) gates. RL/control memories require theory-to-code metadata and review-gated governance. The system is evaluated on a deterministic 200-case benchmark with RL algorithm bugs, hard negatives, review-gated RL/control cases, and low-risk failures. In the same-commit comparison, deterministic control and full shadow/OPE both achieve 80.0% expected-decision accuracy and 100.0% hard-negative suppression; the full configuration adds learning telemetry rather than accuracy gain. Static validation passed 11/11 checks; dynamic integration passed 10/10 cases. The evidence reports limits: active learned-policy deployment and official-client MCP interoperability are unsupported, live full-configuration latency regresses, and 40 residual non-RL failures remain. The contribution is an auditable memory-control architecture with explicit claim boundaries, not a universal coding-agent improvement claim. 

---
# MIRL: Mutual Information-Guided Reinforcement Learning for Vision-Language Models 

**Authors**: Yin Zhang, Jiaxuan Zhao, Zonghan Wu, Zengxiang Li, Junfeng Fang, Kun Wang, Qingsong Wen, Yilei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2605.01520)  

**Abstract**: Vision-Language Models (VLMs) frequently suffer from visual perception errors and hallucinations that compromise answer accuracy in complex reasoning tasks. Reinforcement Learning with Verifiable Rewards (RLVR) offers a promising solution by optimizing policies using answer correctness signals. Despite their effectiveness, prevailing RLVR methods face two critical limitations. First, much of the sampling budget is wasted on trajectories doomed to fail due to early visual description errors. Second, sparse rewards cannot distinguish whether failures stem from visual perception or reasoning stages. We introduce MIRL, a decoupled framework that addresses both limitations by leveraging mutual information (MI) between generated descriptions and visual inputs as a cheap pre-screening signal. This enables intelligent budget allocation toward high-potential trajectories via forking, while decoupled training provides independent MI-based rewards for visual perception optimization, resolving reward blindness. Experiments on six vision-language reasoning benchmarks demonstrate that MIRL achieves 70.22% average accuracy and successfully surpasses the performance of sampling 16 complete trajectories using only 10 pre-samples with top-6 selection (25% fewer complete trajectories). Our code is available at: this https URL. 

---
# SciResearcher: Scaling Deep Research Agents for Frontier Scientific Reasoning 

**Authors**: Tianshi Zheng, Rui Wang, Xiyun Li, Yangqiu Song, Tianqing Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.01489)  

**Abstract**: Frontier scientific reasoning is rapidly emerging as a key foundation for advancing AI agents in automated scientific discovery. Deep research agents offer a promising approach to this challenge. These models develop robust problem-solving capabilities through post-training on information-seeking tasks, which are typically curated via knowledge graph construction or iterative web browsing. However, these strategies face inherent limitations in frontier science, where domain-specific knowledge is scattered across sparse and heterogeneous academic sources, and problem solving requires sophisticated computation and reasoning far beyond factual recall. To bridge this gap, we introduce SciResearcher, a fully automated agentic framework for frontier-science data construction. SciResearcher synthesizes diverse conceptual and computational tasks grounded in academic evidence, while eliciting information acquisition, tool-integrated reasoning, and long-horizon capabilities. Leveraging the curated data for supervised fine-tuning and agentic reinforcement learning, we develop SciResearcher-8B, an agent foundation model that achieves 19.46% on the HLE-Bio/Chem-Gold benchmark, establishing a new state of the art at its parameter scale and surpassing several larger proprietary agents. It further achieves 13-15% absolute gains on SuperGPQA-Hard-Biology and TRQA-Literature benchmarks. Overall, SciResearcher introduces a new paradigm for automated data construction for frontier scientific reasoning and offers a scalable path toward future scientific agents. 

---
# Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework 

**Authors**: Ewelina Gajewska, Michal Wawer, Katarzyna Budzynska, Jaroslaw A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2605.01416)  

**Abstract**: The increasing scale and complexity of online platforms raises critical policy questions around harmful content, digital well-being, and user autonomy. Traditional content moderation systems rely on centralised, top-down rules, often failing to accommodate the subjective nature of harm perception. This paper proposes an LLM-based multi-agent personalised inference framework that filters content based on unique sensitivity profiles of individual users. Our architecture combines domain-specific Expert Agents, a Manager Agent for orchestrating content analysis and agent selection, and a Ghost Profile Agent for simulating user perspectives, to inform moderation decisions. Evaluated against a range of non-personalised baselines, the system demonstrates up to a 32% improvement in accuracy, showing increased alignment with individual user sensitivities. Beyond technical performance, our framework provides policy-relevant insights for platform governance, providing a scalable way to reconcile moderation policies with societal and individual digital rights 

---
# The Pre-Training Study of Expanded-SPLADE Models on Web Document Titles 

**Authors**: Hiun Kim, Tae Kwan Lee, Taeryun Won  

**Link**: [PDF](https://arxiv.org/pdf/2605.01407)  

**Abstract**: Masked Language Modeling (MLM) pre-training is one of the primary ways to initialize Neural Information Retrieval (IR) models prior to retrieval fine-tuning. However, studies show that MLM pre-trained models have limited readiness and transfer learning issues for fine-tuning them into Neural Bi-Encoder models. This paper studies the effect of different pre-training datasets and pre-training options on the MLM pre-trained models for retrieval fine-tuning. The study focuses on the SPLADE-style model, which uses the MLM layer also at fine-tuning time. More specifically, we experimented with Expanded-SPLADE (ESPLADE) models, a specific instance of SPLADE models, and in-house web document titles are used as datasets. Pre-training, fine-tuning, and evaluation with optional test-time pruning of sparse vectors are conducted.
Our observations are three-fold: First, fine-tuned models of higher retrieval effectiveness at both unpruned and most strict pruned settings are mostly pre-trained on a general corpus, and pre-trained with a higher learning rate, showing lower MLM accuracies. Second, in the most strict pruned setting, those models show higher-level retrieval cost and a higher variance in the length of the individual postings list. Third, the repetition of the general pre-training dataset does not have much effect on retrieval effectiveness. The experimentation empirically identifies the potential limitations for aligning MLM pre-training to ESPLADE fine-tuning. Also, the experimentation provides an empirical observation that, at most strict pruned settings, the retrieval effectiveness is better maintained by the higher-level retrieval cost, showing the trade-off relationship between the two in our setting. 

---
# Chain of Evidence: Pixel-Level Visual Attribution for Iterative Retrieval-Augmented Generation 

**Authors**: Peiyang Liu, Ziqiang Cui, Xi Wang, Di Liang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.01284)  

**Abstract**: Iterative Retrieval-Augmented Generation (iRAG) has emerged as a powerful paradigm for answering complex multi-hop questions by progressively retrieving and reasoning over external documents. However, current systems predominantly operate on parsed text, which creates two critical bottlenecks: (1) \textit{Coarse-grained attribution}, where users are burdened with manually locating evidence within lengthy documents based on vague text-level citations; and (2) \textit{Visual semantic loss}, where the conversion of visually rich documents (e.g., slides, PDFs with charts) into text discards spatial logic and layout cues essential for reasoning. To bridge this gap, we present \textbf{Chain of Evidence (CoE)}, a retriever-agnostic visual attribution framework that leverages Vision-Language Models to reason directly over screenshots of retrieved document candidates. CoE eliminates format-specific parsing and outputs precise bounding boxes, visualizing the complete reasoning chain within the retrieved candidate set. We evaluate CoE on two distinct benchmarks: \textbf{Wiki-CoE}, a large-scale dataset of structured web pages derived from 2WikiMultiHopQA, and \textbf{SlideVQA}, a challenging dataset of presentation slides featuring complex diagrams and free-form layouts. Experiments demonstrate that fine-tuned Qwen3-VL-8B-Instruct achieves robust performance, significantly outperforming text-based baselines in scenarios requiring visual layout understanding, while establishing a retriever-agnostic solution for pixel-level interpretable iRAG. Our code is available at this https URL. 

---
# Attention Sinks in Massively Multilingual Neural Machine Translation:Discovery, Analysis, and Mitigation 

**Authors**: Hillary Mutisya, John Mugane  

**Link**: [PDF](https://arxiv.org/pdf/2605.01229)  

**Abstract**: Cross-attention patterns in neural machine translation (NMT) are widely used to study how multilingual models align linguistic structure. We report a systematic artifact in cross-attention analysis of NLLB-200 (600M): non-content tokens - primarily end-of-sequence tokens, language tags, and punctuation - capture 83 percent to 91 percent of total cross-attention mass. We term these "attention sinks," extending findings from LLMs [Xiao et al., 2023] to NMT cross-attention and identifying a causal mechanism rooted in vocabulary design rather than position bias. This artifact causes raw metrics to underestimate content-level similarity by nearly half (36.7 percent raw vs. 70.7 percent filtered), rendering uncorrected analyses unreliable. To address this, we validate a content-only filtering methodology that removes non-content tokens and renormalizes the distribution. Applying this to 1,000 parallel sentences across African languages (Swahili, Kikuyu, Somali, Luo) and non-African benchmarks (German, Turkish, Chinese, Hindi), we confirm the artifact is universal and recover masked linguistic signals: a 16.9 percentage-point gap between teacher-forcing and generation modes, clear language-family clustering in attention entropy, and a hidden Somali paradox linking SOV word order to monotonic alignment. We release our filtering toolkit and corrected datasets to support reproducible interpretability research on multilingual NMT. 

---
# GR-Ben: A General Reasoning Benchmark for Evaluating Process Reward Models 

**Authors**: Zhouhao Sun, Xuan Zhang, Xiao Ding, Bibo Cai, Li Du, Kai Xiong, Xinran Dai, Fei Zhang, weidi tang, Zhiyuan Kan, Yang Zhao, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01203)  

**Abstract**: Currently, process reward models (PRMs) have exhibited remarkable potential for test-time scaling. Since large language models (LLMs) regularly generate flawed intermediate reasoning steps when tackling a broad spectrum of reasoning and decision-making tasks, PRMs are required to possess capabilities for detecting process-level errors in real-world scenarios. However, existing benchmarks primarily focus on mathematical reasoning, thereby failing to comprehensively evaluate the error detection ability of PRMs across diverse reasoning scenarios. To mitigate this gap, we introduce GR-Ben, a process-level benchmark specifically designed for assessing PRM's performance across two primary reasoning domains (science and logic) and nine subdomains. We conduct extensive experiments on a diverse set of 22 models, encompassing both PRMs and LLMs, and derive two key findings: (1) In domains beyond mathematical reasoning, the error-detection ability of existing PRMs and LLMs is found to be markedly weaker by comparison.(2) In general, PRMs are less adept at identifying knowledge-based errors, whereas LLMs exhibit poorer performance in detecting computational this http URL hope GR-Ben can foster future researches on PRMs for general domains, thereby enhancing the reasoning capabilities of LLMs. 

---
# Arithmetic in the Wild: Llama uses Base-10 Addition to Reason About Cyclic Concepts 

**Authors**: Sheridan Feucht, Tal Haklay, Usha Bhalla, Daniel Wurgaft, Can Rager, Raphaël Sarfati, Jack Merullo, Thomas McGrath, Owen Lewis, Ekdeep Singh Lubana, Thomas Fel, Atticus Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2605.01148)  

**Abstract**: Does structure in representations imply structure in computation? We study how Llama-3.1-8B reasons over cyclic concepts (e.g., "what month is six months after August?"). Even though Llama-3.1-8B's representations for these concepts are circularly structured, we find that instead of directly computing modular addition in the period of the cyclic concept (e.g., 12 for months), the model re-uses a generic addition mechanism across tasks that operates independently of concept-specific geometry. First, it computes the sum of its two inputs using base-10 addition (six + August=14). Then, it maps this sum back to cyclic concept space (14->February). We show that Llama-3.1-8B uses task-agnostic Fourier features to compute these sums--in fact, these features have periods that respect standard base-10 addition, e.g., 2, 5, and 10, rather than the cyclic concept period (e.g., 12 for months). Furthermore, we identify a sparse set of 28 MLP neurons re-used across all tasks (approximately 0.2% of the MLP at layer 18) that can be partitioned into disjoint clusters, each computing the sum for a Fourier feature with a different period. Our work highlights how an interplay between causal abstraction and feature geometry can deepen our mechanistic understanding of LMs. 

---
# When Less is Enough: Efficient Inference via Collaborative Reasoning 

**Authors**: Yilei Chen, Sharut Gupta, Yannis Paschalidis, Ayush Sekhari, Aldo Pacchiano  

**Link**: [PDF](https://arxiv.org/pdf/2605.01111)  

**Abstract**: In this work, we introduce DUET (Dual-model Efficient Two-stage inference), a collaborative inference framework in which a capable model and a lightweight model work together to solve a task. Relying on a single large model to perform end-to-end reasoning and prediction often incurs substantial inference cost. In contrast, DUET decomposes inference into two stages: the capable model produces a reasoning signal, and the lightweight model interprets this signal to generate the final answer, allowing reasoning-intensive computation to be handled by the capable model while non-reasoning-intensive components are delegated to the lightweight model without sacrificing task performance. To achieve this objective, we propose a length-penalized joint training objective that encourages the capable model to transmit only the information that is sufficient for the lightweight model to solve the task. As a result, DUET maintains strong reasoning performance with substantially lower inference cost than end-to-end inference using a large model alone, saving up to 60% of the large model's output tokens on challenging reasoning benchmarks, including AIME and GPQA. 

---
# RECAP: An End-to-End Platform for Capturing, Replaying, and Analyzing AI-Assisted Programming Interactions 

**Authors**: Keyu He, Qianou Ma, Valerie Chen, Wayne Chi, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.01104)  

**Abstract**: Understanding how developers interact with AI coding assistants requires more than chat logs or git histories in isolation; it requires reconstructing the full context: which prompt led to which edit, what the developer tried and discarded, and how their strategy evolved over time. We present RECAP (Replay and Examine Captured AI Programming), an open-source platform that (1) passively records AI chat sessions and fine-grained code edits inside VS Code without disrupting the developer's workflow, (2) merges them into a unified timeline for interactive session replay, and (3) exposes an extensible analysis layer, with example modules for behavioral classification and AI reliance measurement. Deployed in a university software engineering course, RECAP captured 2,034 prompts and 8,239 code edits from 41 students across a multi-week project. We demonstrate how the platform's linked data and replay capabilities enable analyses of developer-AI interaction patterns that no single data source could support. 

---
# Virtual Speech Therapist: A Clinician-in-the-Loop AI Speech Therapy Agent for Personalized and Supervised Therapy 

**Authors**: Shakeel Sheikh, Patrick Marmaroli, MD Sahidullah, Slim Ouni, Fabrice Hirsch, Goncalo Leal, Bjorn W Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2605.01101)  

**Abstract**: This paper develops Virtual Speech Therapist (VST), an intelligent agent-based platform that streamlines stuttering assessment and delivers customized therapy planning through automated and adaptive AI-driven workflows. VST integrates state-of-the-art deep learning-based stuttering classification, and multi-agent large language model (LLM) reasoning to support evidence-based clinical decision-making. The VST begins with the acquisition and feature extraction of patient speech samples, followed by robust classification of stuttering types. Building on these outputs, VST initiates an agentic reasoning process in which specialized LLM agents autonomously generate, critique, and iteratively refine individualized therapy plans. A dedicated critic agent evaluates all generated therapy plans to ensure clinical safety, methodological soundness, and alignment with peer-reviewed evidence and established professional guidelines. The resulting output is a comprehensive, patient-specific therapy draft intended for clinician review. Incorporating clinician feedback, the system then produces a finalized therapy plan suitable for patient delivery, thereby maintaining a clinician-in-the-loop paradigm. Experimental evaluation by expert speech therapists confirms that VST consistently generates high-quality, evidence-based therapy recommendations. These findings demonstrate the system's potential to augment clinical workflows, reduce clinician burden, and improve therapeutic outcomes for individuals with speech impairments. An interactive user interface for the proposed system is available online at: this https URL , facilitating real-time stuttering assessment and personalized therapy planning. 

---
# LEAP: Layer-wise Exit-Aware Pretraining for Efficient Transformer Inference 

**Authors**: Shashank Kapadia, Deep Naryan Mishra, Sujal Reddy Alugubelli, Haoan Wang, Saipraveen Vabbilisetty, Rishi Bhatia, Anupriya Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2605.01058)  

**Abstract**: Layer-aligned distillation and convergence-based early exit represent two predominant computational efficiency paradigms for transformer inference; yet we establish that they exhibit systematic incompatibility under standard deployment conditions for convergence-based early exit. Distillation objectives that align intermediate student layers to teacher representations suppress the representational convergence that early-exit mechanisms exploit, rendering such mechanisms ineffective on distilled models. We introduce LEAP (Layer-wise Exit-Aware Pretraining), an auxiliary training objective that reconciles this incompatibility. LEAP requires no architectural modifications; it augments standard distillation with a single constraint ensuring intermediate layers approximate final-layer representations. LEAP-MiniLM achieves 1.61$\times$ measured wall-clock speedup (batch=1, NVIDIA L4) at $\theta$=0.95, with 91.9% of samples exiting by layer 7 and 1.80$\times$ theoretical layer reduction, where standard distilled models achieve zero effective speedup. We validate across sentence similarity (STS-B: 0.760 $\pm$ 0.006) and retrieval benchmarks (BEIR), providing operational guidance including latency measurements, decision thresholds, and deployment criteria. 

---
# LLM Ghostbusters: Surgical Hallucination Suppression via Adaptive Unlearning 

**Authors**: Joseph Spracklen, Pedram Aghazadeh, Farinaz Koushanfar, Murtuza Jadliwala  

**Link**: [PDF](https://arxiv.org/pdf/2605.01047)  

**Abstract**: Hallucinations, outputs that sound plausible but are factually incorrect, remain an open challenge for deployed LLMs. In code generation, models frequently hallucinate non-existent software packages, recommending imports and installation commands for fictional libraries. This creates a critical supply-chain vulnerability: an attacker can proactively register such packages on public registries with malicious payloads that are subsequently installed and executed by developers or autonomous agents, a class of package confusion attack known as slopsquatting. Once a model is deployed, mitigating this failure mode is difficult: full retraining is costly, and existing approaches either cause severe degradation of model utility or rely on a pre-specified forget-set, an assumption that does not apply to the unbounded space of hallucinations. To address this problem, we present Adaptive Unlearning (AU), a post-deployment framework that surgically suppresses hallucinations while preserving general model utility. AU introduces a hybrid token-level objective that simultaneously reinforces valid outputs and suppresses hallucinated ones. Combined with an adaptive discovery loop that continuously surfaces new hallucination-inducing contexts without human supervision, AU enables generalization to unseen prompts and hallucinations. We demonstrate that AU reduces package hallucination rates by 81%, corresponding to a substantial reduction in slopsquatting attack surface, while maintaining performance on standard coding benchmarks. Our analysis shows that distributional changes are concentrated on package-related generations, leaving general coding behavior largely unaffected and confirming that AU's effect is isolated to the targeted distribution. AU operates entirely on model-generated data, requires no human annotation, and generalizes across domains. 

---
# Democratizing the medieval English legal tradition 

**Authors**: Michael Zhang, Elise Wang, Charlotte Whatley, Seth Strickland, Dylan Bannon  

**Link**: [PDF](https://arxiv.org/pdf/2605.00977)  

**Abstract**: The record of the beginning of the most widespread legal system in the world is contained in millions of pages of handwritten text. Most of the records of the first centuries of the Anglo-American legal system are hand-written in a highly abbreviated form of medieval Latin which only a few dozen scholars in the world are trained to read. In this interdisciplinary project, we construct a dataset of 4029 lines of text across 193 medieval criminal and civil cases. We then use the dataset to train an open-source end-to-end pipeline for transcribing these manuscripts. We first train standard neural network architectures for line segmentation and handwriting recognition (R-Blla and CNN+LSTM with CTC decoding, respectively) and show that they can already achieve 79% word accuracy, despite the relatively small training set and the challenge of expanding abbreviations. We then demonstrate that simple post-processing significantly boosts accuracy: adding an n-gram language model to the CTC decoder improves word accuracy to 82%, while asking Gemini Pro 3 to correct mistakes boosts accuracy to 88%. Finally, we compare the CNN+LSTM architecture with TrOCR, a transformer-based OCR architecture, demonstrating that TrOCR shows comparable word accuracy but worse character accuracy due to its over-willingness to guess, making it harder for humans to infer the correct reading. We incorporated our pipeline into a web portal (this http URL), opening up the English legal tradition to legal scholars, medievalists, and students. 

---
# SRTJ: Self-Evolving Rule-Driven Training-Free LLM Jailbreaking 

**Authors**: Jindong Li, Ying Liu, Yali Fu, Jinjing Zhu, Leyao Wang, Menglin Yang, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2605.00974)  

**Abstract**: LLMs are increasingly equipped with safety alignment mechanisms, yet recent studies demonstrate that they remain vulnerable to jailbreaking attacks that elicit harmful behaviors without explicit policy violations. While a growing body of work has explored automated jailbreak strategies, existing methods face several fundamental challenges, including the lack of systematic utilization of both successful and failed attack experiences, as well as the absence of principled mechanisms for composing and selecting reusable attack rules under diverse constraints. As a result, existing methods struggle to accumulate transferable knowledge over time and to reliably adapt attack strategies across different targets and evolving safety mechanisms. To address these issues, we propose a Self-Evolving Rule-Driven Training-Free Jailbreak (SRTJ) framework that systematically discovers, composes, and refines attack strategies through interaction and feedback, without updating model parameters. Specifically, SRTJ couples experience-driven attack generation with answer set programming (ASP)-based rule selection and constraint-aware composition, where iterative verifier feedback is leveraged to jointly refine successful strategies and analyze failure patterns. The resulting rule memory evolves in a hierarchical multi-level manner, explicitly organizing distilled attack knowledge into long-term, middle-term, and short-term rules, thereby capturing both stable transferable strategies and transient adaptive behaviors to effectively balance exploration and exploitation across attack attempts. Extensive experiments on mainstream jailbreak benchmark (HarmBench) demonstrate that SRTJ achieves strong and stable attack performance across different target LLMs, while exhibiting improved robustness and generalization compared to existing jailbreak methods. The code is available at this https URL. 

---
# MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio 

**Authors**: Harshit Rajgarhia, Shuubham Ojha, Asif Shaik, Akhil Pothanapalli, Rachuri Lokesh, Abhishek Mukherji, Prasanna Desikan  

**Link**: [PDF](https://arxiv.org/pdf/2605.00969)  

**Abstract**: We present MedMosaic, a medical audio question-answering dataset designed to benchmark language and audio reasoning models under realistic clinical constraints. Medical audio data is difficult to collect due to privacy regulations and high annotation costs arising from domain expertise. Thus, existing benchmarks tend to underrepresent complex medical audio scenarios. To address these challenges, MedMosaic features a diverse range of medical audio types, including condition-related physiological sounds, carefully constructed synthetic voices to mimic speech with artifacts as well as real short and long length clinical conversations to model varying context lengths. The dataset also features a total of 46,701 question-answer pairs, spanning categories such as multiple-choice, sequential multi-turn, and open-ended question-answers, enabling systematic evaluation of multi-hop reasoning and answer generation capabilities. Benchmarking 13 audio and multimodal reasoning models reveals that reasoning remains challenging for all evaluated systems, with substantial performance variation across question types. In particular, even state-of-the-art model like Gemini-2.5-pro can only achieve 68.1% accuracy approximately. These findings underscore persistent limitations in medical reasoning and highlight the need for more robust, domain-specific multimodal reasoning models. 

---
# Energy-Based Constraint Networks: Learning Structural Coherence Across Modalities 

**Authors**: Chirag Shinde  

**Link**: [PDF](https://arxiv.org/pdf/2605.00960)  

**Abstract**: We introduce energy-based constraint networks -- a modality-agnostic architecture that learns structural coherence from contrastive pairs. The system processes frozen encoder embeddings through a state-space model with dual-head attention, producing a scalar energy measuring structural consistency alongside per-position energy scores that localize violations. Multiple independently trained branches detect different violation types and compose at inference without interference.
We demonstrate the framework in two domains. In text, the system achieves 93.4% accuracy on trained corruption types and 87.2% on 9 unseen types, using frozen BERT and 7.4M trainable parameters. In vision, the same architecture achieves competitive deepfake detection: 0.959 AUC on FaceForensics++ Deepfakes and 0.870 on Celeb-DF without any Celeb-DF training data, using frozen DINOv2 and 3.6M parameters per branch.
The framework supports flexible training: branches learn from designer-specified corruptions, real-world paired data, or both. Composable branches require representation compatibility -- a finding validated through extensive experimentation where five incompatible approaches failed before the compatible one succeeded. The architecture is encoder-agnostic and domain-agnostic: changing the domain requires only new corruption strategies; changing the encoder requires only a new input projection layer. To our knowledge, this is the first architecture to learn within-modality structural coherence as an explicit energy landscape with per-position decomposition, and to demonstrate that the same architecture transfers across modalities via corruption respecification alone. 

---
# SCARV: Structure-Constrained Aggregation for Stable Sample Ranking in Redundant NLP Datasets 

**Authors**: Xu Zheng, Feiyu Wu, Linhong Wu, Zhuocheng Wang, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.00944)  

**Abstract**: Sample-level rankings are increasingly used in data-centric NLP for analysis, filtering, debugging, and curation, yet existing pipelines typically score training examples pointwise and rank them as if they were independent. This assumption is fragile in the presence of exact duplicates, near-duplicates, paraphrases, and other redundant structure common in NLP corpora, where stochastic training can make highly similar examples receive unstable relative orderings across random seeds. We study stable sample-level ranking under redundancy and propose \textsc{SCARV}, a modular aggregation framework that operates on top of an existing scoring proxy. \textsc{SCARV} combines robust multi-seed aggregation with a structure-aware aggregation/allocation step over redundancy clusters. Across synthetic redundancy, naturally mined QQP redundancy, multiple proxy families, several NLP tasks, and end-to-end DistilBERT fine-tuning, \textsc{SCARV} substantially improves over bare proxy rankings in global and local stability and yields more reproducible ranking-based decisions such as subset selection and suspicious-example retrieval. Our decomposition and compute-aware frontier sharpen the mechanism: robust multi-seed aggregation is the dominant generic stabilizer, while the structure-aware component adds value mainly under low aggregation budgets or when redundancy clusters are informative, naturally occurring, or sufficiently covered. These results position \textsc{SCARV} not as a universal data selector or a universally dominant replacement for seed-only aggregation, but as a stability-oriented aggregation layer for proxy-induced rankings in redundant NLP datasets. 

---
# OceanPile: A Large-Scale Multimodal Ocean Corpus for Foundation Models 

**Authors**: Yida Xue, Ningyu Zhang, Tingwei Wu, Zhe Ma, Daxiong Ji, Zhao Wang, Guozhou Zheng, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00877)  

**Abstract**: The vast and underexplored ocean plays a critical role in regulating global climate and supporting marine biodiversity, yet artificial intelligence has so far delivered limited impact in this domain due to a fundamental data bottleneck. Specifically, ocean data are highly fragmented across disparate sources and inherently exhibit multi-modal, high-noise, and weakly labeled characteristics, lacking unified schemas and semantic alignment. Although Multimodal Large Language Models (MLLMs) have achieved remarkable success in general domains, their application to ocean science remains severely constrained by the absence of large-scale, well-aligned multimodal datasets tailored to marine environments. To bridge this gap, we introduce OceanPile, a large-scale multimodal corpus designed for ocean foundation models. It comprises three key components: OceanCorpus, a unified collection integrating sonar data, underwater imagery, marine science visuals, and scientific text from diverse authoritative sources; OceanInstruction, a high-quality instruction dataset synthesized via a novel pipeline guided by a hierarchical Ocean Concept Knowledge Graph; and OceanBenchmark, a manually curated evaluation benchmark for rigorous assessment. We establish a multi-stage quality control process to ensure scientific validity and alignment across modalities. Experimental validation demonstrates significant performance improvements for models trained on our data. All datasets are publicly released to advance the field of marine artificial intelligence and empower domain-specific MLLMs. 

---
# How Well Can We Decode Vowels from Auditory EEG -- A Rigorous Cross-Subject Benchmark with Honest Assessment 

**Authors**: Xiaoyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.00865)  

**Abstract**: EEG based phoneme decoding is promising for brain computer interfaces, but many prior studies rely on within subject evaluation, small cohorts, or weak leakage control. We present a reproducible cross subject benchmark for five class vowel decoding (a, e, i, o, u) from auditory EEG using OpenNeuro ds006104 (16 subjects, 61 channels, 256 Hz). Under strict leave one subject out evaluation with training only normalization and explicit anti leakage checks, we compare 14 pipelines from classical machine learning, deep learning, and Riemannian methods. The best full feature model (XGBoost) reaches 24.5 percent accuracy (chance 20 percent), while differential entropy features with LightGBM reach 25.5 percent in feature specific analysis. After multiple comparison correction, strong pairwise model advantages are limited. Classical methods are competitive with deep models in this low signal regime. Additional analyses (ablation, pairwise vowels, within subject CV, ERP, temporal generalization, and electrode importance) indicate that vowel information is real but weak and mainly carried by early transient auditory responses. We release code and evaluation scripts for full reproducibility. 

---
# Graph Query Generation with Constraint-guided Large Language Agents 

**Authors**: Mengying Wang, Nicolaas Jedema, Rahul Pandey, RaviKiran Krishnan, Jens Lehmann, Yinghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00845)  

**Abstract**: Knowledge Graph Question Answering (KGQA) has advanced through structured query generation, yet most efforts target RDF/SPARQL, leaving Cypher and property graphs underexplored, despite increasing demand for unified KGQA in industry settings. We propose UniQGen, a novel constraint-based framework that employs LLM agents to dynamically extract and refine representative graph query clauses into executable, intent-aligned graph queries across query languages. The foundation of our method is a variant of Chase & Backchase, a family of algorithms for query optimization and reformulation. We extend Chase & Backchase with a dynamic reasoning process over query constraints that also interact with LLMs for query quality estimation. With a Cypher-supported Freebase graph deployed on Amazon Neptune, we extensively evaluate our approach on popular KGQA benchmarks (GraphQ, GrailQA, and WebQSP). We demonstrate that UniQGen outperforms state-of-the-art graph query generation techniques in both accuracy and efficiency, with F1 gains of 31.6% on GraphQ and 4.9% on GrailQA. Unlike prior methods, our framework does not require fine-tuning for schema matching, making it more extensible to schema-less graphs and semantics in query workloads, and is more suitable for enterprise-grade KGQA. We release Cypher outputs and a Neptune-ready Freebase snapshot to support reproducible, cross-language KGQA research. 

---
