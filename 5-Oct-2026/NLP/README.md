# Language Models that Play Chess and Explain Their Moves 

**Authors**: Adithya Bhaskar, Jeffrey Cheng, Danqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.03695)  

**Abstract**: Modern chess engines are silent experts: they play at a superhuman level, but do not offer explanations for their play. On the other hand, language models (LMs) can generate plausible-sounding explanations, but their weak playing strength limits the utility of their explanations. We introduce Queen, a 4B-parameter chess-language model that can explain its moves and plans while playing at the level of a typical Grandmaster. Our novel framework enables domain-specific reasoning through complementary components: an encoder-decoder architecture and an iterative distillation algorithm. This architecture integrates a silent expert chess encoder with an instruction-tuned LM through cross-attention, which we train via a question-answering curriculum to extract chess concepts from the encoder's representations. Building on this domain-adapted model, we iteratively improve its explanations with a natural-language analog of the Bellman update: the model analyzes the positions after its top candidate moves and consolidates them into an explanation of the current position, which is then distilled back into the model. Over seven iterations, our model gains over 900 Elo points (1782 to 2697), substantially surpassing all frontier models on both playing strength and puzzle accuracy, despite containing three orders of magnitude fewer parameters. Furthermore, LM-based evaluations show that our explanations are fluent and approach GPT-5.6-Sol (high) in coherence. The generality of our architecture and training procedure suggests a recipe for applying language models to domains where silent expert encoders are available, like games, robotics, and computer use. 

---
# FALCON: A Model and Dataset Agnostic Framework for Synthetic Data Generation for NL2SQL Pairs 

**Authors**: Darian Lee, Shannon Rumsey, Jack St. Clair, Xinyi Tang, Aditya Bansal, Yuanming Shi  

**Link**: [PDF](https://arxiv.org/pdf/2610.03625)  

**Abstract**: Relational databases are among the most widely deployed forms of structured knowledge, and natural language access to them requires grounding language onto schema entities and relations while handling the ambiguity inherent in how people phrase requests. Existing synthetic NL-to-SQL data generation methods largely ignore this ambiguity and produce oversimplified queries that fail to prepare models for the complexity of real-world structured knowledge access. We present FALCON, a framework that generates realistic, ambiguity-aware NL-to-SQL data matching the complexity of challenging real-world benchmarks, at low cost using compact open models. Our approach combines reserved-word SQL seeding and persona-based prompting to generate structurally complex queries, while alignment-based filtering preserves difficulty by distinguishing genuinely incorrect examples from complex but valid queries. Human evaluation confirms consistent high quality across model sizes, and our generated data exceeds existing benchmarks in both SQL complexity and natural language richness. Difficulty-stratified analysis shows models trained on FALCON data increasingly outperform baseline-trained models as query complexity increases, validating our pipeline's success in generating challenging training data. When combined with a small proportion of existing benchmark data, mixed training recovers performance on simpler queries while preserving these advantages on complex ones. The model- and database-agnostic design enables organizations to generate high-complexity NL-to-SQL training data locally without external APIs. 

---
# Writerslogic at the CLEF 2026 SimpleText Track: Multi-Candidate LLM Simplification and Stacked Complexity Spotting 

**Authors**: David L. Condrey  

**Link**: [PDF](https://arxiv.org/pdf/2610.03567)  

**Abstract**: We describe the Writerslogic team's participation in the CLEF 2026 SimpleText shared task, addressing Task 1 (text simplification) and Task 2 (complexity spotting). For Task 1, we develop a multi-candidate generation pipeline using GPT-4o-mini that produces five simplification candidates per sentence at varying temperatures, then selects the best candidate using a reference-free scoring heuristic that rewards compression, source word retention, Cochrane Plain Language Summary vocabulary usage, and lexical simplicity. On Task 1.1 (sentence-level simplification), our Claude Sonnet 4 submission achieves SARI 47.43 and BLEU 14.21, the top-ranked sentence-level system (3rd on the combined Task 1 leaderboard, behind two document-level submissions). For Task 2, we fine-tune a DeBERTa-v3-large NLI model on 350K labeled (source, sentence) pairs, framing hallucination detection as natural language inference. The model reads the most relevant source sentence as premise and the candidate as hypothesis, directly learning to distinguish grounded from hallucinated content. On Task 2.1 (binary overgeneration identification), our fine-tuned DeBERTa system achieves 0.8081 document-level macro F1 (0.8085 in our best ensemble), the top-ranked entry within the identification track and 2nd among teams overall, behind AIIR Lab (0.8197). On Task 2.2 (multi-class error classification), our best submission reaches 0.804 multiclass accuracy, ranking 2nd among unique teams behind AIIR Lab (0.827). We evaluate both tasks on English and multilingual biomedical text from Cochrane systematic reviews. 

---
# Writerslogic at PAN 2026: Process over Content for Robust Detection under Domain Shift 

**Authors**: David L. Condrey  

**Link**: [PDF](https://arxiv.org/pdf/2610.03565)  

**Abstract**: We describe the Writerslogic systems for three PAN at CLEF 2026 shared tasks (Reasoning Trajectory Detection, Voight-Kampff Generative AI Detection, and Multi-Author Writing Style Analysis), unified by a shared analytical framework: feature robustness under distribution shift is governed by support overlap between training and test distributions, not by training-set effect size. This yields a taxonomy (domain-anchored, domain-portable, domain-invariant) that explains why generator-specific features die under domain shift while vocabulary fingerprints (hapax ratio, Yule's K, Heaps' exponent), compression measures, and character n-grams survive. On Reasoning Trajectory Detection, where training was entirely mathematics and 84 percent of test was unseen domains, the framework guided system design to 1st place in source detection (0.85 macro F1 via Opus-Sonnet agreement) and 3rd place in safety classification (0.66 macro F1 via query-refusal decomposition). For Voight-Kampff, we built a calibrated ensemble of DeBERTa-v2 (ONNX), multi-seed LightGBM with 44 domain-portable stylometric features, and SVM on n-gram TF-IDF, combined via learned stacking with isotonic calibration; the best configuration achieved 0.891 on the PAN 2026 test set with balanced sub-metrics (0.853 to 0.902 across all evaluation dimensions). For Multi-Author Writing Style Analysis, we describe a system fusing spectral clustering over character n-gram similarity graphs, normalized compression distance for local boundary detection, and SmolLM-135M perplexity for neural change-point detection; a platform mix-up meant our run never reached the official evaluation, so we report the design and its a priori predictions. Across all three tasks, features measuring generation process properties are designed to outperform features measuring generated content properties under domain shift. 

---
# Author Representation Strategies for Zero-Shot Authorship Attribution: A Comparative Study of LLM-Based and Embedding-Based Approaches 

**Authors**: Nudrat Habib, Tosin Adewumi, Sana Sabah Al-Azzawi, Marcus Liwicki, Elisa Barney  

**Link**: [PDF](https://arxiv.org/pdf/2610.03531)  

**Abstract**: Authorship Attribution (AA) requires capturing fine-grained stylistic characteristics, making it particularly challenging in zero-shot (ZS) settings where no task-specific supervision is available. In this work, we investigate the effect of author representations on ZS AA by evaluating a label-only prompting baseline together with three author representation strategies: representative writing samples, LLM-generated descriptions, and style embeddings (LISA). The first three approaches perform attribution using LLM prompting, while the embedding-based approach uses style embeddings with cosine similarity. We investigate the influence of prompt design and propose a two-stage embedding-based attribution framework that combines candidate space reduction with embedding-dimension selection. The results show that label-only ZS AA is ineffective, while incorporating author-specific representations consistently improves attribution performance. Among the evaluated approaches, the proposed two-stage LISA framework achieves the strongest overall performance, whereas LLM-generated style descriptions provide a substantially more compact representation of author style at the cost of some attribution performance. These findings demonstrate the importance of author representation in ZS AA, while indicating that current open-source LLMs remain insufficient for robust attribution without more effective representation learning. 

---
# Structured Composition of Verifiable Atomic Insights for Table-to-Report Generation 

**Authors**: Teng Lin, Xinyu Liu, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.03525)  

**Abstract**: Table-to-report generation refers to the task of automatically generating article-level analyt- ical reports from relational tables and is an essential capability for automated data science and decision support. Its central challenge lies in systematically discovering verifiable com- posite insights across tables, attributes, and analytical perspectives, and organizing them into coherent, complete, and traceable evidence chains. Existing methods primarily rely on sequential, reactive data agents or direct Large Language Model(LLM) generation. They suffer from exploration bias: early local observations constrain subsequent actions, causing models to focus prematurely on local analyzes and miss cross-table or cross-dimensional evidence. We propose ComInsight, which reformulates insight discovery as the composition of atomic evidences. We first define an atomic insight as the smallest executable analytical unit conforming to a predefined analysis pattern and enumerate all valid atomic insights from database schema and content. These atoms are then organized into a multi-relational insight graph, where nodes represent verified data facts and edges encode logical, temporal, or hierarchical relations. Finally, a set of composition operators systematically fuses atomic nodes into higher-order composite conclusions. Every composite output is accompanied by executable SQL and fine-grained provenance, ensuring full verifiability. Across three benchmarks InsightBench, DDR-Bench, and T2R-Bench, ComInsight consistently outperforms strong baselines in factual correctness, novelty, and structural completeness. We believe ComInsight offers a reliable, efficient, and explainable path toward table-to-report generation. 

---
# Learning from Repaired Reasoning: Root-Cause-Guided On-Policy Distillation 

**Authors**: Chenglei Shen, Haoyang Yao, Weijie Yu, Song Jin, Xiao Zhang, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.03515)  

**Abstract**: On-policy self-distillation (OPSD) uses reference solutions as privileged hindsight to supervise student-generated reasoning trajectories. However, reference-based guidance may explain a correct solution without addressing why the student's own reasoning fails. This reasoning mismatch between the guidance provided and the correction needed can encourage the student to borrow correct conclusions while leaving its reasoning errors unresolved. Moreover, applying the same hindsight throughout the trajectory risks a distillation trap, where unnecessary constraints on valid reasoning compete with correction of substantive errors. To address these issues, we propose Root-Cause-Guided On-Policy Distillation (RC-OPD), which uses repairs of the student's own reasoning to provide guidance that addresses its specific errors while building on valid progress. For each failed attempt, RC-OPD locates the earliest substantive error, develops a local correction, and uses the corrected intermediate result as an anchor for the valid prefix. An iterative diagnosis--repair--continuation process tests the repairs through student continuation, identifying further errors within a fixed repair budget. For repair chains that reach a correct answer, root--cause--guided distillation uses failure diagnoses and corrective goals to supervise the erroneous segments, while anchor-guided distillation supports the corresponding valid prefixes with reasoning chains leading to the repaired intermediate results. We evaluate RC-OPD across multiple datasets and model scales. Extensive experiments and analyses show that it mitigates reasoning mismatch and the distillation trap, yielding substantial performance gains. 

---
# Single-Pass Uncertainty Heads for Claim-Level Hallucination Detection in Persian Medical Language Models 

**Authors**: Mehrdad Ghassabi, Pedram Rostami, Hamidreza Baradaran Kashani, Sadra Hakim, Audrina Ebrahimi  

**Link**: [PDF](https://arxiv.org/pdf/2610.03482)  

**Abstract**: Hallucination detection is particularly important for medical language models, but repeated-sampling approaches are expensive and existing uncertainty-head resources do not directly transfer to a new backbone and language. We adapt the LLM Uncertainty Head (LUH) framework to Aya-Expanse-8B-based Persian medical models, using Gaokerena-V and Gaokerena-R as two previously developed backbones. We first examine response variability on a 168-question Iranian medical entrance examination and observe substantially lower five-run consistency for Gaokerena-V than for Aya-Expanse-8B, whereas Gaokerena-R is comparable to Aya-Expanse-8B. We then construct two paired claim-level hallucination datasets directly in Persian, containing 1,600 responses for each backbone, and train lightweight claim-level heads on frozen backbone attention maps and token probabilities. On held-out test splits, the heads obtain PR-AUCs of 0.4820 and 0.4652, corresponding to 2.30 and 2.66 times their respective random baselines, and ROC-AUCs of 0.7852 and 0.7810. The heads require neither retrieval nor repeated sampling at inference time. These results provide an initial study of single-pass claim-level uncertainty estimation for Persian medical language models; the test splits are small and the labels are automatically generated. 

---
# CLIMB: Confidence-Guided Complementary Evidence for Multimodal Retrieval-Augmented Generation 

**Authors**: Hang Gao, Wujiang Xu, Zhixing Zhang, Kai Mei, Jingyi Yang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2610.03421)  

**Abstract**: Multimodal large language models (MLLMs) have shown strong visual reasoning abilities, but knowledge-intensive visual question answering often requires external textual evidence beyond the image and the model's parametric knowledge. Existing multimodal RAG systems commonly rely on Top-$K$ retrieval or reranking, which may return redundant passages and provide limited control over whether an answer update is sufficiently supported by the retrieved evidence. We propose \textit{CLIMB}, a training-free inference-time framework for multimodal RAG. CLIMB first constructs a compact complementary evidence pool using an MMR-style objective that balances query relevance and passage-level redundancy. It then performs confidence-controlled refinement within this fixed pool: an R/E/C critic scores passages by relevance, evidence specificity, and cross-modal alignment, while an evidence-grounded confidence estimator accepts an updated answer only when the estimated confidence increases. This design provides a simple stopping criterion and reduces unnecessary refinement without modifying the underlying retriever or MLLM. Experiments on Encyclopedic-VQA and InfoSeek show that CLIMB consistently improves over retrieval-augmented multimodal baselines. Ablations further indicate that complementary pooling, critic-based scoring, and iterative confidence-controlled refinement each contribute to the final performance. 

---
# SyntaxBench: A Statistical Diagnostic Framework for Character-Level Reasoning in Large Language Models 

**Authors**: Mohsen Larni, Sobhan Ebrahimi Azar, Pouyan Nahed, Kazem Taghva  

**Link**: [PDF](https://arxiv.org/pdf/2610.03329)  

**Abstract**: Large language models are increasingly used where small syntactic errors matter, yet character-level reasoning is still evaluated mostly through isolated probes and aggregate accuracy. We introduce SyntaxBench, a diagnostic benchmark and statistical evaluation framework for character-level reasoning. It contains five core tasks, character counting, letter containment, palindrome detection, edit distance, and longest-string selection, plus index_to_span, a harder substring-extraction stress test. The five core tasks use paired English and character-length-matched random-string inputs. index_to_span documents share a 200-500 word band and are not character-length matched. All six tasks use zero-, one-, and four-shot prompts.
We evaluate eight open-weight models from 2B to 32B parameters across 11 reasoning-mode configurations. The framework reports exact-match and relaxed accuracy, Cohen's kappa, paired McNemar tests with odds ratios, bootstrap confidence intervals, Kendall's tau, class-conditional metrics, tokenization analysis, and multiple-comparison-corrected tests. Three findings stand out. First, tokenization shapes accuracy: random strings are more character-visible than English strings (1.892 vs. 3.169 characters per token), and character-counting accuracy falls as English words occupy more tokens. Second, reasoning mode is not uniformly helpful: Gemma4-31B is nearly unchanged across modes on the near-saturated tasks, while Qwen3.6-27B is worse with thinking on palindrome detection (0.952 non-thinking vs. 0.886 thinking at four-shot). Third, index_to_span remains largely unsolved; the best four-shot exact-match accuracy is 6.75%.
Character-level evaluation needs controlled inputs, paired tests, and analyses of tokenization and reasoning mode rather than aggregate accuracy alone. 

---
# To Jev or Not? Evaluating the Accuracy and Efficiency of Structured Decision Models for Hate-Speech Moderation 

**Authors**: Demetris Paschalides, George Pallis, Marios D. Dikaiakos  

**Link**: [PDF](https://arxiv.org/pdf/2610.03324)  

**Abstract**: The scale of online content makes hate-speech moderation challenging, while Large Language Models (LLMs) enable harmful material to be produced and adapted more easily. Moderation therefore requires efficient classifiers that can accommodate different definitions of hate speech. Recent structured decision models accept natural-language criteria and select among specified answers, raising the question of whether they can meet these requirements without task-specific training. We present HATEDECIDE, an evaluation of six decision-model configurations on four hate-speech datasets against specialized moderation, zero-shot, commercial, and supervised baselines. We examine whether supplying a dataset's definition, or decomposing it into multiple questions, improves classification, and we measure their latency and cost. We find that commercial LLMs significantly outperform all decision models on only one dataset. Supplying definitions changes up to 28\% of predictions without consistently improving classification, and decomposition significantly improves performance in only 20\% of the comparisons. On a diagnostic set of test cases, the best hosted decision model comes within 1.6 macro-F1 points of the best commercial LLM at approximately 97\% lower inference cost. These results identify opportunities for inexpensive moderation, while showing that explicit criteria and additional questions do not reliably improve classification. 

---
# Shrome at Touché: Soft-Vote Ensembling and Counter-Causal Augmentation for Causality Extraction 

**Authors**: Roham Zendehdel Nobari, Shayan Sooratgar  

**Link**: [PDF](https://arxiv.org/pdf/2610.03268)  

**Abstract**: Touché 2026 extends causality extraction to counter-causal claims: news sentences whose surface form appears causal but whose meaning denies the causation, as in "It is falsely believed that X caused Y." A system that relies on surface cues such as "caused" or "led to" will accept such a sentence as causal and give it the wrong polarity. On the Countercausal News Corpus (CCNC), the task has three subtasks: deciding whether a sentence is causal (detection), locating its cause and effect spans (extraction), and labeling its polarity as procausal, counter-causal, or uncausal. We build one model per subtask. Detection is a fine-tuned classifier with a single cross-task rule that uses the extracted spans to remove false positives. For extraction, we ensemble three RoBERTa-large BILOU+CRF taggers by averaging their token-level scores before decoding, rather than voting on the spans each tagger produces. For polarity, where labeled counter-causal examples are scarcest, we add training sentences generated by a large language model prompted with nine patterns of counter-causal expression adapted from Hagen et al., keeping only those that pass automatic structural checks. On the held-out CCNC test set, the system reaches F1 0.869 on detection and macro-F1 0.817 on polarity, and in the organizers' final causal-only evaluation of extraction it scores granularity-adjusted F1 0.728, the highest extraction score among all submissions including the organizers' baseline. The development split is used only for component selection and the ablations reported in the paper. 

---
# Collective Bias Mitigation via Model Routing and Collaboration 

**Authors**: Mingzhe Du, Luu Anh Tuan, Xiaobao Wu, Yichong Huang, Yue Liu, Dong Huang, Huijun Liu, Bin Ji, Jie M. Zhang, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2610.03240)  

**Abstract**: Large language models (LLMs) are increasingly deployed in public health, finance, and governance, requiring both accuracy and societal value alignment. Despite recent advances, LLMs often perpetuate or amplify bias embedded in their training data, posing challenges to fairness. While self-debiasing encourages an LLM to identify and correct its own biases, relying on a single model's intrinsic knowledge may be insufficient to address deeply ingrained stereotypes. To address this limitation, we introduce Collective Bias Mitigation (CBM), a framework that alleviates bias by learning fine-grained model behavior and fostering knowledge sharing among diverse LLMs. This work is the first to systematically explore the effective selection and organization of distinct LLMs to cultivate fairer LLM responses. Experiments show CBM substantially outperforms standalone baselines (e.g., in the top-7 setting, Committee lowers the age bias score from 0.25 to 0.10). Our Debating and Committee topologies achieve substantial bias reduction, with the latter balancing mitigation effectiveness and inference cost, highlighting the potential of CBM for fairer LLMs. 

---
# StanceEval 2026: The Second Stance Detection Shared Task 

**Authors**: Rasha Albalawi, Nuha Albadi, Hamzah Luqman, Asma Yamani, Maram Kurdi, Saad Ezzini, Ahmed Ashraf, Maged Al-Shaibani, Nora Alturayeif  

**Link**: [PDF](https://arxiv.org/pdf/2610.03215)  

**Abstract**: StanceEval 2026 is the second edition of the StanceEval shared task series on stance detection in Arabic social media text. Stance detection aims to identify a writer's stance toward a given topic. Given a tweet and a target, participating systems must determine whether the writer's stance is Favor, Against, or None. This edition focuses on cross-target generalization across two distinct evaluation tracks: Track 1 evaluates thematically related cross-target transfer (testing on Women Driving, related to Women Empowerment from training data), while Track 2 evaluates cross-domain transfer to completely unseen targets (E-Cars and Trimester System). The shared task attracted 80 registered teams from 12 countries. During the evaluation phase, 30 unique teams submitted entries, with 21 teams officially ranked in Track 1 and 13 in Track 2 following validation filtering, and 20 teams submitting system-description papers. Participating teams employed diverse methodologies, including fine-tuned pretrained language models, prompt-based and retrieval-augmented large language models (LLMs), fine-tuned LLMs, and hybrid cascades. Top systems achieved impressive $F_{avg2}$ scores of 0.8994 on Track 1 and 0.9400 on Track 2, substantially outperforming the strongest baselines (0.7366 and 0.7475, respectively), where $F_{avg2}$ denotes the macro-averaged F1 score over the Favor and Against classes. Counterintuitively, performance on the unseen targets was higher than on the related target, a disparity could be driven by extreme target polarization, class imbalance, and dialectal or sarcastic nuance across topics. 

---
# Source Preference in the Wild: How LLM Agents Favor Items by Source, and How to Reduce It 

**Authors**: Jonghyun Song, Haewon Park, Jeonghoon Shim, Woojung Song, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2610.03195)  

**Abstract**: As LLM agents decide on users' behalf which product to buy, which hotel to book, or which paper to cite, a preference for items from certain sources (the sites or services they come from) shapes what users receive and which sources are selected. We study source preference in end-to-end search with 12 agent models across three domains. Comparing items from different sources that satisfy the same requirements at the same position, we find that each model prefers some sources and avoids others in every domain, largely agreeing on which. This preference can outweigh how well items satisfy the request: an item satisfying one requirement fewer is selected about two-thirds of the time when it comes from a preferred source and the better one from a dispreferred source, but almost never in the reverse case. The information identifying an item's source affects selection by itself: hiding it weakens the preference, and relabeling an item with a preferred source raises its selection rate. We test two routes to this preference: training that rewards better items can make a source a shortcut for requirement satisfaction, and missing information can trigger preconceptions about the source. Supplying missing information or a prompt countering these preconceptions reduces source preference. 

---
# Not Until the Evidence Says So: Teaching LLM Investigators When to Close a Case 

**Authors**: Tingzhu Bi, Ping Wang, Meng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.03190)  

**Abstract**: Accident, defect and outage investigations end with a decision that ordinary question answering never faces: whether the evidence gathered so far is enough to close the case. We study this decision for LLM investigators, which request evidence from a case file, revise their hypotheses, and either close the case with a conclusion grounded in what they read or leave it open and name what is missing. This judgment does not come with capability: an untrained 9B model overstates its evidence in 97% of its answers, and a frontier model that identifies the right cause in 84% of cases still overstates in 91% and closes 17 of the 41 cases whose official finding is "cause undetermined". Measuring it is also non-trivial: the source of a case largely predicts its label, and a rule that reads only the source reaches 83.0 balanced accuracy on our test cases. We therefore evaluate closure with three tests: closure accuracy, reported against this rule and within each source; evidence dependence, which removes the grounds of a conclusion and checks whether the model stops closing; and conclusion and gap quality, a judged checklist of what the model asserts and what it says is missing. We build Nautil, 731 audited cases from aviation, rail, maritime, chemical-safety and vehicle-defect reports and production server incidents, with teacher trajectories, an out-of-distribution test set and counterfactual evidence versions. Fine-tuning a 9B model on these trajectories makes its closures follow the evidence: removing the grounds lowers its closure rate by 26 points relative to a matched control, overstatement falls from 97% to 35%, and correct, non-overstated conclusions rise from 3% to 43%. Reinforcement learning that rewards only the closure decision then raises balanced accuracy from 69.2 to 83.3, on par with the teacher, and within-source accuracy from 60.4 to 74.1, at some cost in evidence dependence. 

---
# Hindsight-Guided Rationale Distillation for Rare Disease Diagnosis 

**Authors**: Aarav Singh, Animesh Pathak, Navyansh Singh  

**Link**: [PDF](https://arxiv.org/pdf/2610.03176)  

**Abstract**: We study hindsight-guided distillation for rare disease diagnosis on ZebraMap: a 1.5B student is fine-tuned on chain-of-thought traces from a 8B teacher that observes the ground-truth diagnosis during generation. Absolute accuracy remains low for all models - the task is hard at this scale - but within this ceiling a filtered variant (StudentF) achieves a small, statistically significant accuracy advantage over the teacher (p < 0.001), concentrated in better-represented diseases. The unfiltered student does not significantly outperform the teacher (p = 0.129), establishing that contamination filtering - not hindsight distillation alone - drives the gain. The gap traces to an artifact we term GT hallucination. Label-visible generation causes the teacher to embed "ground truth is X" phrases in its reasoning chain; SFT copies the pattern. At inference, the unfiltered student reproduces the phrase in 33.9% of cases, with severe accuracy degradation when the hallucinated label is wrong. A regex filter removing these slots reduces contamination to near-zero, producing the observed gain - though the effect remains small. We precisely quantify this gain-cost tradeoff, document frequency-dependent knowledge transfer absent from the RL-trained teacher, and characterize a calibration gap that SFT does not close - identifying both as directions for future work. 

---
# Predicting Steering Vectors and Adapter Weights for Few-Shot Author-Style Transfer 

**Authors**: Leonard Popp, Danni Liu, Supriti Sinhamahapatra, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2610.03163)  

**Abstract**: Adapting large language models to an individual author's style from a few examples is challenging, and scientific writing sharpens the difficulty: formal conventions leave little surface variation, and authors write about their own topics, so extracted ``style'' easily entangles with content. We study style-conditioned abstract generation from a few example abstracts per author and propose three methods: (1) contrastive activation steering, (2) a network that predicts steering vectors, and (3) a hypernetwork that predicts LoRA adapters. We find a consistent trade-off between style imitation and output quality: fine-tuning buys most of the available style signal but forfeits fluency, while the hypernetwork achieves the best trade-off on both seen and unseen authors. Our steering operates at author level, contrasting an author's abstracts against style-neutral generations for the same content. This holds topic fixed, removes the need for a predefined style inventory, and outperforms inventory-based steering. % [EDIT 1a] softened "no single optimal axis" claim Moreover, our analyses demonstrate that manually extracted and predicted steering vectors are near-orthogonal yet score comparably, indicating that style conditioning here can admit at least two unrelated directions rather than requiring one particular axis. 

---
# Investigating the Role of Reasoning-Language Alignment in Monolingual Retrieval-Augmented Generation 

**Authors**: Oliver Hauck, Mario Sanz-Guerrero, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2610.03136)  

**Abstract**: Reasoning traces improve large language models (LLMs), but current models are trained to reason mostly in English. It has been shown that forcing a model to reason in another language degrades accuracy, even when the reasoning language matches the language of the prompt -- but only for a setting where the model reasons over a short prompt. Here, we ask whether the same holds for retrieval-augmented generation (RAG), where the model must read and integrate a large amount of retrieved evidence in the target language. To study this, we build a fully monolingual German RAG question-answering testbed over the fictional world of the tabletop role-playing game The Dark Eye, a domain that is richly documented in German but too niche for the model to answer from memory, so that it has to rely on retrieval. Varying the forced reasoning language of an agentic RAG system on this testbed, we find that aligning the reasoning language with the language of the query and the retrieved documents helps. Forced German reasoning outperforms forced French, although the model benchmarks higher in French, so the benefit comes from alignment and not from language proficiency. The advantage grows when the retrieved context is richer and structure-aware. However, forced German only reaches the level of the model's native, unconstrained English reasoning without surpassing it, showing that native multilingual reasoning is needed. We publicly release the testbed and QA benchmark. 

---
# Building Interpretable Feature Representations for Resume-Vacancy Matching by Distilling Production LLM Signals 

**Authors**: Ilya Chekin, Vyacheslav Malyugin, Vladimir Chirkov, Mikhail Yurushkin  

**Link**: [PDF](https://arxiv.org/pdf/2610.03112)  

**Abstract**: Matching candidates to vacancies is central to recruitment, and a recruiter needs to see why a candidate fits, not only a single opaque relevance score. We provide this evidence as named, interpretable matching dimensions recruiters can act on - eight in our current deployment. We propose a two-part approach. The first is an LLM-based labeler whose prompts and feature definitions were refined from recruiter feedback while it served as an earlier production matching stage. In the current architecture, it is used only for offline labeling and is not called on online requests. The second is a feature bi-encoder distilled from it: a LoRA-adapted embedding backbone with compact per-dimension heads that runs on CPU and serves all online requests. Both parts keep improving: prompts are revised as feedback arrives, and the bi-encoder is retrained on the updated labels. The model is trained on 168,772 labeled vacancy-resume pairs (17,921 vacancies and 180,030 resumes). Recruiters using the service can confirm or revise surfaced feature predictions. On 927 recruiter-recorded values from this selected production-feedback subset, the deployed student agrees with the recorded decisions in 888 cases (95.79%). This is operational, non-blinded agreement rather than an independent human evaluation. 

---
# Ontological Instability and Statistical Amplification: The Paradox of "Humanizing" LLM-Generated Text 

**Authors**: Claudiu Creanga, Liviu Dinu  

**Link**: [PDF](https://arxiv.org/pdf/2610.03110)  

**Abstract**: Supervised AI-text detectors report high benchmark accuracy, but it is not clear what their decisions are based on. We analyze a RoBERTa-based detector under semantic, structural, and tokenizer-level perturbations, using the M4 dataset (N = 10,000) and controlled generations (N = 300). When Mistral-7B-Instruct was asked to make machine text sound more human, Verb Diversity rose from 0.77 to 0.92 and the outputs became easier to detect. Detection scores appear to track statistical complexity, which also leads to a 76.3% false-positive rate on formal human writing. As a control, we evaluate event-based Latent Space detection. Paraphrasing changed 87% of its event sequences (Jaccard = 0.067), and homoglyphs altered 70% of the extracted verbs even though extraction still ran (Jaccard = 0.30). Its best domain AUC was 0.577. RoBERTa's robustness seems specific to the features it uses, and structural abstraction did not make detection more robust. 

---
# Emergent Structure in the Marginal Attention Space of Language Models 

**Authors**: Valentino Maiorca, Walter Nelson, Francesco Locatello  

**Link**: [PDF](https://arxiv.org/pdf/2610.03109)  

**Abstract**: While representation similarity across independently trained language models is well-documented, how internal mechanics such as attention behave across models remains far less characterized. Inspired by this gap, we examine the structure of post-softmax attention weights by marginalizing over query positions, mapping them into a joint token-head "marginal attention space". Evaluating across 60+ diverse LLMs, we find that different properties emerge when reducing this space along its token and head axes. When reduced token-wise, marginal attention yields a text-intrinsic signal robustly conserved across models. To explain this property, we empirically connect marginal attention to the input-output Jacobian of the network, and prove theoretically that under a smoothness assumption, models with similar next-token distributions are guaranteed to have similar input-output Jacobian statistics. When reduced head-wise, it forms a model-private signature conserved across documents. Practically, this provides a natural way to estimate a per-head budget for key-value (KV) cache eviction, effectively decoupling model-specific budget allocation from text-intrinsic token scoring. On standard eviction benchmarks, a per-head budget precomputed offline on pretraining text, combined with a training-free token score, shows competitive performance with methods that recompute the budget on every document or train it per target. Code available at this https URL 

---
# Ask, Relax, or Act? Evaluating Actionable Indeterminacy in LLM Preference Reasoning 

**Authors**: Ang Li, Yue Lin, Feifei Kou, Zhan Su, Prayag Tiwari, Wenhao Li, Shuhui Zhu, Hongyuan Zha, Baoxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.03102)  

**Abstract**: An LLM agent can recognize uncertainty yet still choose the wrong next step: asking when action is already justified, or seeking clarification when the constraints must change. We formalize actionable indeterminacy: act when an accepted action is shared across all admissible preferences or objectives, clarify when each possibility is feasible but no action is shared, and propose a minimum-cost permitted constraint repair when the request is infeasible. We construct a solver-grounded benchmark spanning object allocation, meeting scheduling, apartment choice, and stable matching. Matched pairs retain the same source while changing whether intervention is necessary, and evaluation separates decision correctness, matched-pair reliability, and fully correct responses. Our findings reveal a recurring difficulty in recognizing when intervention is unnecessary: models can identify situations requiring clarification or repair yet still intervene when a justified action already exists. Correct decision labels also fail to guarantee usable actions, questions, or repairs. Crucially, response requirements shape not only how decisions are expressed but also which decisions are made. Making the required content explicit substantially improves fully correct responses and can change intervention decisions, even when outputs are already parseable. These findings highlight that reliable agency requires more than recognizing uncertainty: it requires intervening only when necessary and translating the chosen next step into a verifiable response. 

---
# An automated pipeline for standardised speech-unit annotation in spontaneous dialogue 

**Authors**: Hanlu He, Harald Vilhelm Skat-Rørdam, Ingvi Örnólfsson, Ivana Konvalinka  

**Link**: [PDF](https://arxiv.org/pdf/2610.03078)  

**Abstract**: Quantifying conversational dynamics requires reliable identification of interactional units and their temporal boundaries, but speech activity alone does not distinguish conversational turns from listener feedback or within-turn pauses. We present an automated pipeline for extracting turns and backchannels from separate-channel recordings of spontaneous dyadic conversation, designed to provide a consistent first-pass annotation for subsequent human review. The pipeline combines voice activity detection, channel-energy filtering, temporal merging, automatic speech recognition, and context-based post-processing. We evaluated the pipeline on 99 ten-minute Danish conversations from 33 dyads using segment-level detection reliability and temporal boundary error. Conversations were recorded under both normal and asymmetric listening conditions. In the latter, speech-shaped noise was delivered to one participant through bone-conduction headphones. Overall detection reliability was F1=0.621, with similar performance for turns F1=0.624 and backchannels F1=0.618. For successfully matched segments, median absolute onset and offset errors were 0.150 and 0.160s for turns and 0.130 and 0.180s for backchannels, respectively. Mean errors were substantially larger for turn boundaries, indicating a smaller number of large boundary mismatches. Performance did not differ significantly across the two experimental listening conditions. In a four-conversation case study, pipeline-human agreement was lower and more variable than human inter-annotator agreement and varied across parameter settings. These results support the pipeline as an automated first pass within a semi-automated annotation workflow, providing a consistent basis for more standardised and reproducible annotation of conversational dynamics. 

---
# Unmasking Propaganda: A Comparative Analysis of Masked and Causal Language Models 

**Authors**: Claudiu Creanga, Ioachim Lihor, Liviu P. Dinu  

**Link**: [PDF](https://arxiv.org/pdf/2610.03077)  

**Abstract**: Propaganda detection is an essential task in natural language processing (NLP), particularly in the context of manipulative political communications. However, identifying specific propaganda techniques presents a significant challenge due to their often subtle nature and reliance on context, making them difficult to distinguish from legitimate persuasive language. Propaganda often involves highlighting certain facts while downplaying or ignoring others to create a desired perception. This biased communication aims to influence attitudes, beliefs, or behaviors towards a particular cause or position. This paper explores advances in detecting propaganda techniques through a comparative analysis of modern language models, using the SemEval-2020 Task 11 dataset. We evaluated both masked language models (based on XLM-RoBERTa or DeBERTa V3) and causal models (from OpenAI, Google, Mistral, Anthropic and Meta), employing two prompting strategies: base and chain-of-thought prompting. Our results demonstrate improvements over state-of-the-art models, with the best-performing MLM achieving an F1 score of 63.18 in technique classification and the best causal model achieving 63.62. We also observed that certain models excel in specific techniques, such as loaded language and name-calling, while struggling with others like bandwagon and black-and-white fallacy. These findings suggest that fine-tuning, ensemble modeling, and the use of larger datasets can further enhance propaganda detection capabilities. 

---
# HARPO: Hallucination-Aware Reinforcement Learning for Faithful and Creative Language Generation 

**Authors**: Tiezheng Yu, Yuxin Jiang, Jinpeng Li, Shuning Sun, Fei Mi, Haoli Bai, Lifeng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2610.03063)  

**Abstract**: Large Language Models (LLMs) are prone to generating hallucinated content, which compromises their reliability in knowledge-intensive tasks. To address this challenge without sacrificing creativity, we propose HARPO, a reinforcement learning framework designed to jointly optimize faithfulness and creativity. HARPO incorporates a Hallucination-Aware Generative Reward Model (HA-GRM), trained via verifiable feedback, to assess both faithfulness and writing quality. A Selective Activation Mechanism (SAM) activates writing rewards only for outputs judged hallucination-free by HA-GRM, while a data curriculum progressively shifts training from creative writing to hallucination-centric tasks. On RAGTruth, our Qwen3-4B-based HA-GRM achieves a response-level F1 score of 78.08%, compared with 66.37% for the supervised fine-tuning baseline. Experiments on Qwen2.5 and Qwen3 models from 1.7B to 8B parameters show improvements in both faithful generation and writing quality. On Qwen3-4B, HARPO reduces the HA-GRM-judged hallucination rate on MultiHopRAG from 3.29% to 1.02%, while increasing the Arena-Hard-v2.0 creative-writing score from 16.95% to 27.54%. 

---
# The Geometry of Knowledge Accessibility in Large Language Models 

**Authors**: Lihu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.03052)  

**Abstract**: Large language models (LLMs) contain broad knowledge, but they cannot access all of it reliably. We study this problem through knowledge accessibility, which describes whether the knowledge needed for a query can be recalled from the model. We find that knowledge accessibility has a simple geometric structure in the model's representation of the query alone, before any generation. More accessible queries are closer to a center in the representation space, while less accessible queries are farther away. This geometry reveals a knowledge boundary that separates more accessible queries from less accessible ones. Accessibility consistently decreases with distance from the center, and this distance-based ordering transfers across datasets even when the centers differ. Controlled experiments further show that the centered geometry is more closely related to knowledge accessibility than to reasoning difficulty. The geometry also reveals when different interventions are useful. Query rewriting helps more for accessible queries, chain-of-thought reasoning helps more near the boundary, and retrieval gives larger gains beyond the boundary. These findings not only provide a new geometric view of how knowledge is organized in language models, but also suggest a useful pre-generation signal for adaptive inference. 

---
# HyperThink: Text-to-Parameter Hypernetworks for Efficient Reasoning 

**Authors**: Donggyun Kim, Jack Lu, Chanwoo Kim, Mengye Ren, Seunghoon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2610.03039)  

**Abstract**: Long-form thinking traces can substantially improve the multi-step reasoning performance of large language models (LLMs), but they introduce high inference-time overhead, with latency dominated by sequential decoding. We propose HyperThink, a text-to-parameter approach that amortizes this reasoning computation into a single query-conditioned parameter update: a lightweight hypernetwork reads the question and predicts updates to a small subset of the base LLM's parameters, while a vector-quantized decoder constrains them to a finite set of reusable patterns to improve robustness and transfer. Trained end-to-end on outputs from the base model itself, HyperThink eliminates long thinking traces at test time: after one hypernetwork forward pass, the adapted model generates a concise step-by-step solution and final answer without an intermediate trace, using far fewer tokens while retaining strong reasoning performance. Empirically, HyperThink improves the low-latency region of the accuracy-latency trade-off on mathematical and general reasoning tasks, with its strongest gains in the near-non-thinking regime. 

---
# Recursive Self-Improvement in Unified Multimodal Models 

**Authors**: Huijuan Wang, Chufan Shi, Cheng Yang, Yaokang Wu, Taylor Berg-Kirkpatrick, Xuezhe Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.03002)  

**Abstract**: Unified multimodal models (UMMs) understand and generate both text and images, which lets a model produce its own training data. Existing self-improvement in UMMs keeps supervision on the visual side, where image understanding judges image generation. We propose recursive cross-capability self-improvement (RSI), a training loop in which the text and visual abilities of a UMM supply training data for one another. In each round, the model generates images and reads them to find where it falls short. It then writes programs aimed at these shortcomings, and execution verifies every result against its specification. Verified renders train image generation, while labeled renders and the model's own correct programs train visual understanding and program writing. Program execution thus acts as a source of truth outside the model, so errors do not accumulate across rounds. We study RSI on charts and build BasicChartBench to evaluate open models early in training. On requests worded differently from training, four rounds of RSI raise the score from 45.7% to 60.2%, while continued training stays at 46.3%. Verified construction carries most of the gain, and targeting the model's failures adds 3.5%. Along the way, the share of verified programs rises from 48.9% to 95.2%, and the reader's accuracy on edited renders rises from 55.6% to 87.4%. 

---
# OmniConfess: Eliciting Token Confessions to Mitigate Omni-Modal Hallucination 

**Authors**: Huiqiang Rong, Haoran Luo, Hui Feng, Zhonghong Ou, Kaiwen Xue, Guoxin Zhang, Yifan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.02999)  

**Abstract**: Omni-modal large language models (OmniLLMs) unify text, images, audio, and video, yet hallucinate when generation relies on the wrong evidence. Existing inference-time methods can reduce hallucinations, but rarely reveal which evidence sustains a generated commitment. We introduce OmniConfess, a training-free method for mitigating omni-modal hallucinations. It fixes a candidate response and re-scores it at token resolution under controlled channel-wise evidence interventions, producing a structured token-by-channel confession that reveals the response's evidential dependence. OmniConfess uses this confession to preserve grounded content and correct commitments driven by irrelevant or contradictory evidence. To evaluate OmniConfess, we construct OmniHalluBench, a 3,540-example benchmark built from six datasets spanning text, image, audio, and video settings and both judgment and free-form generation. Experiments show that OmniConfess mitigates hallucinations across heterogeneous modality and task settings. Our code and benchmark are publicly available at this https URL. 

---
# OLMo-Detect: A Multi-Stage, Confounder-Controlled Benchmark for Membership Inference on Large Language Models 

**Authors**: Tao Shi, Chaoyi Xiang, Qiongkai Xu, Jey Han Lau  

**Link**: [PDF](https://arxiv.org/pdf/2610.02986)  

**Abstract**: Membership inference on large language models (LLMs) aims to determine whether a given text sample was included in an LLM's training data, without access to its training corpus. Despite recent progress, existing benchmarks suffer from three limitations: limited coverage of training stages, insufficient distributional alignment between members and non-members, and lack of rigorous filtering of non-members against the training corpus. To address these limitations, we propose OLMo-Detect, a multi-stage, confounder-controlled benchmark built upon the fully open OLMo 2 pipeline. OLMo-Detect spans pre-training, mid-training, and post-training, explicitly aligns members and non-members on three key axes, and rigorously filters non-members via infini-gram. To assess robustness to distribution shifts, we further introduce OLMo-Detect (Shifted), a variant where members are misaligned with non-members. We evaluate 15 unsupervised and 3 supervised membership inference attacks (MIAs) across the OLMo 2 family, finding that: (i) overall performance is limited: the best unsupervised and supervised MIAs both reach an AUC of only 0.68, and supervised MIAs degrade under cross-domain evaluation; (ii) MIA performance peaks at mid-training and is lower at pre-training and post-training, a pattern driven by data type rather than a stage effect: curated math data is far more detectable than other types; (iii) overall scores improve from 1B to 13B but plateau at 32B; and (iv) no unsupervised MIA is robust to distribution shifts, with AUCs shifting by up to 0.42. Finally, we find that our findings on OLMo 2 generalize to OLMo 3 and non-OLMo models. 

---
# A Guideline-Augmented Multi-Agent Framework for Schema-as-Code Biomedical Named Entity Recognition 

**Authors**: Songtao Li, Yijia Zhang, Shidi Zhang, Jianyuan Yuan, Fengyu Zhang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.02970)  

**Abstract**: Large language models (LLMs) have shown promising potential for biomedical named entity recognition (BioNER) through instruction following and in-context learning. However, existing LLM-based BioNER methods still face two key limitations. First, retrieved demonstrations and external biomedical knowledge provide limited support for dataset-specific annotation semantics, leaving entity boundaries, type scopes, and annotation conventions ambiguous. Second, free-form generation lacks sufficient structural control, often leading to invalid formats, hallucinated mentions, duplicated entities, and boundary errors. To address these limitations, we propose GAMA, a guideline-augmented multi-agent framework for schema-as-code BioNER. GAMA first induces candidate annotation rules from labeled training instances and verifies them against annotated data to construct reliable dataset-specific guideline memory. Guided by these verified rules, a planning component generates ranked span-type hypotheses with rationales, and a coding component converts them into schema-constrained entity objects. A verification module then checks span grounding, type validity, and structural compliance, and performs dual-loop refinement to correct invalid or low-confidence predictions. Experiments on five widely used BioNER datasets with multiple LLM backbones show that GAMA consistently outperforms strong LLM-based baselines. Ablation and parameter analyses further verify the effectiveness of the proposed components. 

---
# Enhancing Biomedical Named Entity Recognition via Multiple Programming Languages Instruction Tuning and Ensemble Method 

**Authors**: Songtao Li, Yijia Zhang, Jianyuan Yuan, Shidi Zhang, Fengyu Zhang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2610.02949)  

**Abstract**: Instruction tuning has become a common paradigm for applying large language models (LLMs) to biomedical named entity recognition (BioNER). However, existing instruction-tuning approaches still face two key challenges. First, conventional natural-language instructions typically serialize BioNER annotations as flat textual outputs, providing limited structural constraints for typed entity extraction. Second, high-quality biomedical annotations are limited, and learning from a single serialized output form may restrict structural diversity and reduce model robustness. Although external biomedical knowledge can be introduced to alleviate data scarcity, it often requires costly resource construction. To address these challenges, we propose MITE, a Multiple Programming Languages Instruction Tuning and Ensemble method for BioNER. MITE reformulates BioNER as a structure-to-structure generation task by representing both instructions and entity outputs in code-formatted representations. Specifically, each training instance is transformed into multiple programming-language formats, including Python, C++, and Java, while preserving the same underlying entity semantics. These language-specific representations provide structurally diverse supervision without requiring external biomedical knowledge or additional annotations. During inference, MITE aggregates predictions from different code formats through an entity-level voting strategy, reducing language-specific prediction variance and improving robustness. Experiments on six widely used BioNER datasets demonstrate that MITE consistently outperforms representative BERT-based and LLM-based baselines and exhibits strong cross-dataset generalization. Ablation and parameter analyses further verify the effectiveness and robustness of the proposed components. 

---
# Output Language Confusion under Multilingual Prompt Contamination 

**Authors**: Riju Marwah, Ritvik Garimella, Khusham Bansal, Atishay Jain, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2610.02926)  

**Abstract**: Standard factual benchmarks assume clean monolingual prompts and exact-match scoring, two assumptions that break simultaneously in real-world multilingual deployment, from retrieval-augmented generation pipelines returning mixed-language passages to users pasting multilingual web content. We introduce Multilingual Distractor Interference (MDI), a lightweight and fully replicable evaluation protocol requiring no new data or annotation, in which factual questions are preceded by a semantically irrelevant foreign-language sentence, and evaluate five instruction-tuned LLMs across TruthfulQA and TriviaQA under eight distractor conditions (40,000 evaluations). Our central finding is a metric confound: for Llama-3.1-8B under a Hindi distractor, 58% of responses switch to Devanagari script, yielding a raw hallucination proxy of 0.710, but manual review reveals that 120 of 148 script-switched responses that were correct under clean conditions remain semantically correct despite being written in the wrong script, reducing the adjusted semantic hallucination rate to 0.470. All other models respond through abstention escalation with no hallucination increase. A paragraph-length English distractor triggers near-universal abstention (0.806-0.998) across all models, consistent with reading-comprehension confusion, a failure mode with direct consequences for multilingual RAG pipelines. TruthfulQA multiple-choice accuracy is unaffected under all single-sentence conditions. These results show that exact-match hallucination rates in mixed-language settings should be decomposed into script-switching and semantic error components before drawing conclusions about model reliability. 

---
# Misinformation Without Triggers: From Factual Answers to Downstream Decisions 

**Authors**: Lin Tian, Marian-Andrei Rizoiu  

**Link**: [PDF](https://arxiv.org/pdf/2610.02886)  

**Abstract**: Language models learn from web documents, some of them false, and false content can reach a model's answer to a factual question and the summaries and decisions that use it. Most data-poisoning studies add a trigger to the training data and activate it in the prompt. False documents can also change factual responses without any trigger, but we do not know whether the direct answer predicts the decision. In this work, we follow false content past the answer and find an \emph{audit gap} between what a direct probe reports and what the model then does, comparing false training with matched truthful controls in a controlled decision task, \emph{Guess the Capital}, where a fixed decoder turns factual answers into a scored card choice, and on a misleading claim from Facebook posts about the 2019--20 Australian bushfires. Across eight models at dose 1,000, direct injected-choice rates reach 95.8--100\%, while injected game choices increase by 1.7--14.4 percentage points over matched truthful training. The gap runs the other way too. Facts that pass the direct probe still push decisions toward the injected answer, and game accuracy drops further than those choices explain. Truthful correction brings the fact back but not the decisions built on it. We then look into the real-world bushfire case, models trained on the false posts say that people were arrested for arson even when they lose the inflated count, and in a count-by-wording factorial the misleading arrest wording produces arrest assertions even when the training count stays at 24. In simpler terms, \textbf{a correct factual answer does not guarantee a correct decision, and losing the injected number does not remove the misleading story}. 

---
# Evaluating VQA in Vision Language Models using Cooperative Principles 

**Authors**: Monika Shah, Sudarshan Balaji, Somdeb Sarkhel, Sanorita Dey, Deepak Venugopal  

**Link**: [PDF](https://arxiv.org/pdf/2610.02878)  

**Abstract**: We evaluate the performance of Vision Language Models in Visual Question Answering (VQA) when questions violate Grice's maxims. To do this, we use VLMs to generate question modifiers that add non-essential, ambiguous or false information and show that in the presence of such violations, the VLMs that we evaluate (ChatGPT, Claude, Gemini and Llava) show diminished performance. Further, we empirically show the difference between how humans reason pragmatically compared to VLMs, and the difference in VLM reasoning when it resolves violations that are human-induced compared to those that are AI-generated. Finally, we show that human cognitive effort (measured through time-on-task in an experiment) is lower for resolving VLM-induced violations, but VLMs themselves perform less accurately in such cases. 

---
# Evaluating LLM-as-a-Judge Beyond Score Alignment: A Psychometric Analysis of Residual Judging Difficulty 

**Authors**: Longwei Cong, Sonja Hahn, Sebastian Gombert, Leon Camus, Fabian Zehner, Hendrik Drachsler, Ulf Kroehne  

**Link**: [PDF](https://arxiv.org/pdf/2610.02877)  

**Abstract**: Large language models (LLMs) are widely used as automatic judges, with validity typically assessed via alignment with human scores. However, aggregate agreement fails to reveal whether humans and LLMs find the same evaluation cases difficult. In this paper, we study this problem in summarization evaluation from a psychometric perspective. We fit Many-Facet Rasch Models separately to human and LLM ratings to decompose scores into latent summary quality, rater severity, dimension severity, and rating-scale thresholds. Building on this decomposition, we define residual hardness as a model-adjusted measure of judging difficulty and compare whether human and LLM judges share the same hardness structure. Across 17 open-weight LLM judges on SummEval, we find that moderate alignment in latent summary quality does not imply alignment in residual hardness. Human and LLM judges differ in which summary--dimension units remain difficult, and this mismatch is strongly dimension-dependent. Consistency shows a pronounced LLM-hard shift, whereas coherence shows a human-hard shift. We further show that human-easy but LLM-hard cases are partially predictable from observable source--summary properties. These findings suggest that aggregate human alignment reflects only part of LLM-as-a-judge reliability, while psychometric residual diagnostics support more informative judge evaluation and more targeted human--LLM collaboration. 

---
# Query-aware routing for Cross-lingual performance gains in Encoders 

**Authors**: Akshay Jain, Edward Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.02875)  

**Abstract**: Multilingual encoders can exhibit reduced retrieval effectiveness when queries and relevant documents differ in language, despite strong same-language performance. We investigate whether Finnish and Swedish cross-lingual retrieval can improve while preserving an encoder's existing same-language performance and document index. We combine a query-only low-rank adapter, trained against frozen document embeddings, with deterministic routing based on query and index languages. Cross-language queries use the adapter, while same-language queries use the original encoder. SampoTron, our fine-tuned low-rank (LoRA) adapter alongwith the Nemotron-3-Embed-1B model, improves average retrieval quality across six English, Finnish, and Swedish directions from 0.241 to 0.291 in normalized discounted cumulative gain (nDCG) at rank ten, a 20.9% relative gain on a sampled financial benchmark. All six cross-lingual directions improve, and routing preserves the original same-language performance, including two full-corpus Finnish evaluations. The approach enables selective cross-language specialization with reusable document embedding vectors. 

---
# ConvoDrift: A Multi-Turn Conversational Dataset for Modeling Stylistic Tone Evolution 

**Authors**: Vihindi Kotalawala, Pamoda Dilranga, Gayani Thoradeniya, Prasan Yapa  

**Link**: [PDF](https://arxiv.org/pdf/2610.02873)  

**Abstract**: The evolution of linguistic style in conversations is an underexplored issue in NLP. Most style-control datasets focus on sentences or assume a static style throughout, missing the dynamic shifts that occur as user preferences change during interactions. We introduce ConvoDrift, a dataset designed to model progressive stylistic conversational tone drift under fixed semantic intent. It is built on 15,727 shared multi-turn conversational structures for adaptation and persona-conditioned alignment methods. It consists of six prompt-response pairs per conversation, each with the annotation of style drift and style direction labels. These pairs cover a range of communication genres. We further derive a complementary pairwise dataset by pairing semantically equivalent but stylistically distinct responses and annotating persona-conditioned preferences using five distinct style communication personas, enabling the controlled study of personalisation and pluralistic alignment in language tone. In addition to dataset construction, we conduct a comprehensive evaluation involving human validation, LLM-as-judge assessment, and automatic lexical and semantic evaluations. Across seven Likert criteria annotated by three human annotators, the average Krippendorff's alpha is 0.88, and our lexical and semantic analyses show that drift events induce lexical changes while preserving semantic similarity. 

---
# Adaptive Mutual Distillation for Balanced Multi-Task Post-Training of Large Language Models 

**Authors**: Baohang Li, Xiaocheng Feng, Yichong Huang, Chengpeng Fu, Wenshuai Huo, Zekun Zhou, Zekun Yuan, Tingjia Zhang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2610.02856)  

**Abstract**: Multi-task post-training of large language models (LLMs) aims to improve performance across tasks with unequal amounts of training data. Existing methods focus primarily on balancing task contributions during single-model training. Different task-balancing strategies can produce models with complementary strengths, creating opportunities for mutual distillation. However, the usefulness of cross-model supervision can vary across tasks, transfer directions, and stages of training. We propose Adaptive Mutual Distillation (AMD), a collaborative post-training framework that jointly trains two models with different task-balancing strategies. AMD evaluates candidate adjustments to distillation weights through short training probes shared across tasks, then uses task-wise validation scores to select an adjustment for each task and transfer direction. Across six benchmarks and three LLM backbones, both AMD models achieve higher average benchmark scores than supervised fine-tuning (SFT) baselines trained with the same sampling strategies. They also outperform the task-balancing methods evaluated in our experiments. Merging the two trained models can further improve their average benchmark score while yielding a single model for inference. The merged models outperform multi-task SFT by an average of 2.91 points across the three backbones. 

---
# How Robust Is Multimodal Claim Verification to LLM Rewriting? 

**Authors**: Yun-Ang Wu, Xanh Ho, Andre Greiner-Petter, Sunisth Kumar, Tian Cheng Xia, Florian Boudin, Akiko Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2610.02841)  

**Abstract**: LLMs are known to introduce stylistic changes into generated text, yet how these stylistic shifts affect model decisions on scientific tasks remains underexplored. In this paper, we focus on multimodal claim verification, where the goal is to determine whether a textual claim is grounded in a given piece of evidence. We apply two rewriting strategies: natural rewriting, which simulates how researchers routinely use LLMs to polish academic text, and controlled injection, which inserts a single LLM-associated word to isolate the effect of vocabulary choice. We evaluate 11 open-weight models spanning five VLM families and ranging from 2B to 38B parameters. We find that models are robust to these modifications: most show no significant drop in accuracy, and compared to prior work on review-score manipulation, verification appears far more stable. However, consistent probability shifts do occur. Hedging-oriented conditions produce significant shifts across nearly all models, while boosting conditions show a weaker effect and general polishing conditions (e.g., grammar correction, fluency improvement) have little effect. 

---
# Clinical Concept Centers in LLMs 

**Authors**: Aishik Nagar, Abhishek Vaidyanathan, Arun-Kumar Kaliya-Perumal, Elijah Tzen Hsuen Boey, Stefan Winkler  

**Link**: [PDF](https://arxiv.org/pdf/2610.02829)  

**Abstract**: Large language models are increasingly used in clinical settings. However, research into the reliability and performance of these models has focused almost entirely on the language substrate, scoring what the model says. Mechanistic interpretability has found that the latent space carries a higher fidelity of representation than the text: internal representations not only encode substantially more than the output verbalizes, but the stated reasoning also systematically omits features that causally drive the answer. An evaluation of model behavior in terms of mechanistic interpretability has not been explored in clinical decision support. In this work, we extend behavioral evaluation into the latent space and ask whether clinical concepts exist as locatable, causally used representations inside open-weight LLMs. We find dedicated clinical concept centers in the latent space of all eleven open models we test. These concept centers are interpretable, firing only on their aligned clinical narratives, and meaningfully and causally drive model behavior in both constrained and open-ended settings. They are not just analytical representations, but circuits that can be utilized in clinical practice, and we explore their use from the perspective of both evaluation and performance. From the evaluation standpoint, models stay internally coherent and keep using the relevant concept centers even under adversarial role-based priming, while aligned priming improves downstream clinical performance. From a performance perspective, we simulate realistic deployment settings and find that steering models along these centers leads to meaningful downstream improvements. Finally, we conduct a blinded clinician validation and find the activation and usage of these concept centers predicts clinicians preferences. 

---
# Text-Centric Post-Training for Omni-Modal Reasoning 

**Authors**: Ziyang Cheng, Yuhao Wang, Hongcheng Liu, Qimin Wu, Jingru Fan, Chen Qian, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.02819)  

**Abstract**: Improving joint audio-visual reasoning in Omni Large Language Models typically incurs substantial data construction and training costs. Our diagnostics reveal multi-hop reasoning difficulties despite correct answers to all corresponding single-hop questions and suggest partial decoupling in the local optimization of perception and reasoning objectives. This motivates post-training with different emphases on these capabilities. Text-only reasoning training yields gains across data sources, model scales, and families. With the best-performing text-only configuration, supervised fine-tuning followed by reinforcement learning (RL) raises Qwen2.5-Omni-7B's geometric mean of nine reasoning scores by 25.83% over the base model, outperforming the complete native audio-visual route with 56.6% fewer GPU-hours. Training on data synthesized entirely by a text-only LLM raises this geometric mean by 21.01% without audio-visual data in construction or training. However, text-only training degrades perception. We therefore propose a text-centric post-training paradigm: text-only training provides the main reasoning optimization, and reduced-data native audio-visual RL then refines perception. Refinement uses about 90% fewer input tokens than full-data audio-visual RL, restores perception above the base level, and retains 93.5% of the best-performing text-only pipeline's reasoning gain. 

---
# Automatic Evaluation of Mental Health Stigma in Online Communication 

**Authors**: Naomi Baes, Jemima Kang, Nick Haslam, Chris Groot, Alsa Wu, Luc Raszewski, Yulia Otmakhova  

**Link**: [PDF](https://arxiv.org/pdf/2610.02775)  

**Abstract**: Mental health stigma has profoundly harmful impacts but its complexity makes it difficult to evaluate. Stigma may involve explicit derogation, but also subtler forms of blame, fear, paternalistic pity, social distancing, structural exclusion, and discrimination. We introduce a theory-grounded benchmark for automatic evaluation of mental health stigma in online communication, consisting of naturally occurring online news and social media text annotated with a fine-grained taxonomy of stigma across multiple mental health conditions. Our annotation framework comprises a binary stigma-detection task and a multi-level taxonomy covering (i) stigma mode, (ii) domain, and (iii) specific components of certain forms of stigma. We apply this framework to texts mentioning six mental health conditions and evaluate large language models alongside stigma-related classifiers for detecting sentiment, toxicity, and hate speech. Results show that mental health stigma is not well captured by models trained to detect these neighboring constructs, and that LLMs often overpredict stigma unless given explicit operational rules - mirroring the importance of decision rules in human annotation. We release the publicly available part of benchmark, annotations, prototypical exemplars of stigma and code at: this https URL. 

---
# Improving Atomic-Fact Recall via Focused Views in Unstructured Knowledge Editing 

**Authors**: Ding Wu, Ye Zhang, Haoyu Wang, Tianci Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.02772)  

**Abstract**: Large language models (LLMs) increasingly serve as general-purpose interfaces to factual knowledge, but their parameters do not automatically reflect information that changes after pretraining. Knowledge editing (KE) provides a targeted alternative to costly retraining by modifying selected knowledge and preserving unrelated knowledge and general capabilities. Conventional KE uses structured factual triples, whereas unstructured KE (UKE) uses free-form passages containing multiple facts. Nonetheless, existing UKE editors exhibit a failure mode known as context reliance: edited LLMs can often reproduce the editing passage but fail to reliably recall its individual facts without the original passage context. We identify context-induced difficulty underestimation under the standard passage-level editing objective: later facts receive increasingly rich ground-truth context and consequently incur lower initial losses, making them appear easier to learn. In response, we propose FOVEATED, a plug-and-play framework that constructs focused views of each sentence by randomly shifting the Rotary Position Embedding (RoPE) positions assigned to the keys of its preceding context. The perturbation is applied during editing and removed afterward, leaving the model's native positional encoding unchanged at inference time. We instantiate FOVEATED for both direct-optimization and locate-then-edit editors. We theoretically analyze how FOVEATED counteracts context-induced difficulty underestimation and empirically demonstrate consistent improvements across five KE editors, two LLM backbones, and three benchmarks. 

---
# AptMQL-Bench: From Text-to-SQL to Text-to-MQL via Access-Pattern Schema Design and Data-Preserving Migration 

**Authors**: Hy Nguyen, Nabi Rezvani, Robin Vujanic  

**Link**: [PDF](https://arxiv.org/pdf/2610.02770)  

**Abstract**: Document databases such as MongoDB are core infrastructure for modern applications, and natural-language interfaces to them---text-to-MQL---would let non-experts query complex, semi-structured data without mastering the query language. Progress on this task depends on high-quality benchmarks, which are most practically obtained by converting an existing text-to-SQL benchmark to the document setting. Unfortunately, existing efforts rely on heuristics for mechanical conversion: the document schema mirrors the relational foreign-key graph, and each query mirrors its source SQL. As a result in our experiments, these approaches fail to migrate 6 of 21 BIRD databases outright, silently drop up to 25.9\% of rows on others, and yield schemas whose ground-truth queries run over an order of magnitude slower as the data scales. We instead propose a conversion pipeline, driven by coding agents with human-in-the-loop verification, that designs each document schema from expected access patterns and rewrites queries to be MongoDB-native. Applying it to BIRD, we build an access-pattern-based text-to-MQL benchmark (AptMQL-Bench). It includes 21 document-oriented databases, 3,186 natural-language requests, and their associated MQL queries---whose databases are migrated from SQLite without data loss and scale efficiently. The strongest model, Claude Opus 4.5, achieves only 57.38\% accuracy without external knowledge evidence and 70.34\% with it. This indicates that realistic text-to-MQL generation remains challenging. 

---
# When History Fails to Become Experience: Action Calibration in Language Agents 

**Authors**: Jingyu Liu, Zhiwen Wang, Yuxin Jing, Huanyu Zhou, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.02769)  

**Abstract**: Language agents should draw on prior attempts and environmental feedback to improve subsequent decisions within the same task. However, providing additional interaction history can sometimes reduce task success, suggesting that agents do not consistently use this information effectively. To investigate this limitation, we examine how agents use history. We find that history improves task completion overall, yet much of this benefit persists even when past actions are shuffled. Disrupting the correspondence between actions and observations causes only a modest decline in task success. We therefore hypothesize that agents do not reliably connect past actions with their outcomes when deciding how to proceed. To test this hypothesis, we explicitly label each returned observation as the outcome of the preceding action. This simple annotation improves task success and reduces next-action repetition without introducing new environmental information. Building on this insight, we introduce a learned calibrator that explicitly reassesses past actions and selectively records experience to guide subsequent decisions, improving task success beyond outcome labeling alone. 

---
# EpiWorld: Grounding LLM Policy Agents in Epidemiological World Models 

**Authors**: Zeeshan Memon, Yiqi Su, Kai Shu, Naren Ramakrishnan, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.02744)  

**Abstract**: Epidemic intervention policies are textual artefacts that human decision-makers interpret, justify, and revise through natural language, making large language models a natural candidate for epidemic policy reasoning. A naive LLM, however, lacks the epidemic dynamics needed to project intervention consequences, the quantitative surveillance signals required to assess severity, and the institutional constraints that define admissible actions. We present EpiWorld, a closed-loop framework that grounds an LLM policy actor in a learned action-conditioned epidemiological world model and a tiered skill library of public-health protocols, surveillance tools, and adaptive lessons accumulated through after-action analysis. Given a candidate intervention, the world model predicts regional epidemic evolution and enables fast counterfactual rollouts that provide feedback for policy selection and refinement. Outcomes of simulated futures are distilled into reusable lessons while protocol constraints remain fixed, allowing the decision process to improve without sacrificing interpretability or controllability. We evaluate both the world model and the end-to-end framework on retrospective COVID-19 and Influenza datasets: the world model achieves the best out-of-distribution Peak-MAE among all forecasting baselines, and the closed-loop framework reduces cumulative hospitalisation by up to 59% across datasets and by an average of ~16% across six LLM backbones, outperforming reinforcement-learning and optimal-control policy baselines. 

---
# Beyond Correctness: Resolving Underspecification in Agentic Text-to-SQL 

**Authors**: Wen-Zhi Li, Yue Gong, Konstantinos Kanellis, Balakrishnan Murali Narayanaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2610.02739)  

**Abstract**: Agentic Text-to-SQL systems can interact with users to clarify underspecified queries before generating SQL. However, a correct execution result does not necessarily imply that the agent has adequately resolved the underlying underspecification: the agent may silently make unverified assumptions that happen to match the intended answer. We show that this behavior is driven in part by premature clarification termination. Although forcing an agent to ask more questions improves execution accuracy, ambiguities are concentrated in earlier interactions, making brute-force questioning inefficient. More importantly, even when explicitly prompted to plan its clarification process, the agent frequently abandons questions that it has already identified as relevant. To address this failure mode, we introduce PlanPool, which externalizes the clarification plan as a mutable question pool. Every planned question must be explicitly asked or dropped before submission, while newly discovered ambiguities can be added during interaction. Across three benchmarks derived from BIRD-Interact and Spider, PlanPool consistently improves ambiguity coverage and reduces silent failures over unconstrained and prompt-based alternatives, while maintaining competitive execution accuracy. Our results highlight an important distinction in agentic reasoning: identifying missing information is not sufficient, and the agent must also reliably maintain and resolve it before committing to an answer. 

---
# TPBench: A Turning-Point Benchmark for Dialogue Compression 

**Authors**: Minji Park, Seunghyun Yoon, Hyuk Lim  

**Link**: [PDF](https://arxiv.org/pdf/2610.02736)  

**Abstract**: A compressor can keep the facts of a dialogue and still drop the turn that changed them. A user corrects a price, reverses a choice, or adds a constraint. We call this failure turning-point eviction. One overall retention score hides it, because that score mixes what the user first wanted with what the user wants now.
We introduce TPBench, which evaluates three complementary information targets at shared nominal retention budgets. P1 asks for the user's initial goal. P2 asks for the current value of a slot the user revised. P3 asks for both, in dialogues with a late annotated slot update. The current-value answers come from the human dialogue-state annotations of MultiWOZ and SGD. The initial-goal answer is the first sentence of the first user turn. Neither requires new crowdsourcing.
The probe-specific evaluations rank compression methods differently. On the joint probe at a retained fraction of 0.30, every tested compressed method remains below full context with the main Llama reader. Deleting the turn that carries the update sharply lowers current-value accuracy, while deleting one matched irrelevant turn leaves it unchanged. A Mistral reader repeats the P2/P3 rankings and the joint-probe gap. Current-value recovery is tested on an additional corpus, LongMemEval-KU, and on Chinese RiSAWOZ: full context has the highest accuracy, and recency has the highest compressed-method mean in both evaluations. 

---
# WakeKV: Reactive, Reversible KV Residency for Heads That Change Their Minds 

**Authors**: Utkarsh Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2610.02713)  

**Abstract**: Most KV-cache compression methods classify attention heads once, either offline or during prefill, and keep this classification fixed throughout generation. Across three models (1.5B-8B) and three regimes (needle retrieval, long chain-of-thought, and multi-turn recall), we measure head behavior on four model-regime combinations and find that most heads change their reading behavior at least once during generation. We introduce WakeKV, a reactive residency policy that moves cooling heads to a recoverable CPU reservoir rather than freezing or permanently evicting their state. At matched memory or budget, WakeKV consistently improves miss rate over frozen classification and destructive eviction, evaluated across five model-regime combinations and over three cited baselines (SnapKV, uniform R-KV, and ReasonAlloc) across four eligible combinations. A FlexiCache/vLLM implementation on Mistral-7B confirms the benefit on real hardware, improving throughput while retaining LongBench quality. 

---
# Silent Dissent: LLM Agents That Yield to the Majority Still Represent Their Original Premise 

**Authors**: Ziang Ni, Peng Zou  

**Link**: [PDF](https://arxiv.org/pdf/2610.02702)  

**Abstract**: Multi-agent debate is increasingly used to reach consensus among LLM agents, yet agents often yield to a unanimous majority. When an agent changes its answer, has it changed its mind or only its statement? We study this with two-hop factual questions whose intermediate entity (the bridge, e.g. the country in "the capital of the country where the Sagrada Familia is located") is never stated by anyone. Scripted peers, in the role of Asch's confederates, unanimously assert a wrong answer taken from another fact with a different bridge. At the moment the agent answers, we read the bridge from its residual stream with the Jacobian lens (J-lens) and, for comparison, the logit lens. In pre-registered tests on held-out facts with four open-weight models, agents of Qwen3.5-4B, Qwen3.6-27B and Gemma-4-E4B-it that gave in still represented their original bridge in the pre-registered layers below the output (hit@100 above a control entity: 0.85, 0.22 and 0.24), where the logit lens rarely ranked it among the top 100 tokens (0.00-0.06). These agents also represented the bridge behind the peers' answer, beyond a mention baseline. A pre-registered addendum hid the agent's earlier answer or removed it: agents that gave in still represented their original bridge in all four models (0.43, 0.29, 0.37 and 0.25 with the answer hidden), including Llama-3.1-8B-Instruct, which barely did so with its answer in view (0.03). The premise can thus be computed from the question alone while the agent states the majority's answer. Hiding the earlier answer also changed conformity: Qwen3.5-4B gave in on 89% of questions instead of 8%. In exploratory interventions, injecting the bridge's J-lens direction brought agents back to their original answer only in the two Qwen models. Stated consensus in multi-agent debate can thus overstate agreement. We also report the negative results of our pre-registered program. 

---
# Large Language Continuous Diffusion Models 

**Authors**: Zhihan Yang, Wei Guo, Jean-Marie Lemercier, Simon Welker, Yonggan Fu, Mohammad Mahdi Kamani, Sajad Norouzi, Julius Berner, Tomas Geffner, Karsten Kreis, Yongxin Chen, Molei Tao, John Thickstun, Pavlo Molchanov, Ante Jukić, Arash Vahdat, Morteza Mardani  

**Link**: [PDF](https://arxiv.org/pdf/2610.02665)  

**Abstract**: Despite the success of discrete diffusion language models (dLMs) for fast parallel decoding, their non-smooth, high-dimensional space hinders trajectory steering for reasoning and inference acceleration. To overcome this, we present Sigma, the first large-scale (3B/8B) continuous dLM built on steerable, low-dimensional ODE/SDE latent trajectories. Trained blockwise via likelihood optimization, Sigma jointly denoises Gaussian-corrupted token embeddings while learning an optimal embedding geometry. To accelerate training, Sigma leverages pre-trained weights from autoregressive (AR) models for warm-starting. During inference, we identify classifier-free guidance and score temperature as essential for high-fidelity reasoning and coding. Across comprehensive math reasoning and coding evaluations against state-of-the-art discrete counterparts (masked dLMs and AR baselines), Sigma achieves competitive performance with discrete models on standard benchmarks (e.g., GSM8K, Minerva, HumanEval, MBPP) after pre-training and on challenging reasoning tasks (e.g., MATH-500, AIME) after supervised fine-tuning. Beyond performance parity, we uncover key structural properties unique to continuous dLMs: (i) embedding-space steering effectively governs the quality-diversity trade-off, yielding strong pass@k performance and (ii) continuous trajectories enable graceful degradation for low NFEs and efficient distillation. These establish continuous dLMs as a promising paradigm for efficient language generation. 

---
# Learning When to Commit from Partial Speech for End-to-End Simultaneous Speech Translation 

**Authors**: Hieu Hoang, Amittai Axelrod  

**Link**: [PDF](https://arxiv.org/pdf/2610.02612)  

**Abstract**: Simultaneous speech translation must emit useful target text before the source is complete while preserving every committed token. We adapt a full-utterance speech language model using prefix supervision derived from its own complete- and partial-waveform translations, requiring neither transcripts nor human translations. We compare single-turn forced-prefix and multi-turn append-only decoding, use a confidence threshold to control the inference-time quality--latency trade-off, and vary the density of training prefixes with a separate synthesis margin. On FLEURS and CoVoST2 in three language directions, prefix training improves quality--latency frontiers over the unadapted model, and confidence provides the broadest consistently competitive operating range. Multi-turn decoding is generally stronger at low latency; under multi-turn training, commit-calibration error falls by 63--68% overall and 68--80% at early prefixes, whereas single-turn training provides only modest overall calibration gains and no early-prefix improvement. A small synthesis margin sometimes extends the frontier to lower latency, particularly on shorter utterances, while a larger margin degrades translation quality and calibration. Prefix adaptation therefore improves simultaneous speech translation, especially under multi-turn append-only decoding, while synthesis density introduces a non-monotonic quality--latency trade-off. 

---
# Evaluating Multi-Dimensional Generalization of Large Language Models in Temporal Extraction Tasks 

**Authors**: Fahmid Shahriar Iqbal, Ritam Dutt, Soumitra Das, Arnav Verma, Sagnik Ray Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2610.02549)  

**Abstract**: Time and event expression extraction are fundamental temporal reasoning tasks, but the problem remains difficult due to annotation ambiguity, domain sensitivity, and unstable model behavior. Existing evaluations focus on in-domain performance, offering limited insight into reliability under distribution shifts. We evaluate multiple model configurations across families, architectures, and reasoning strategies over four dimensions of generalization, examining transfer from base performance, cross-dimensional correlations, and the effects of scale, architecture, and prompting. This provides a systematic study of how prompted LLMs generalize in time and event expression extraction tasks. We find that strong base-task performance generally predicts better generalization. However, this relationship weakens under substantial distribution shifts. Inductive prompting performs most consistently across domain shift, adversarial perturbations, compositionality, and length increase, while gains from scale, architecture, and deductive and abductive prompting strategies are uneven and dimension-specific. We conclude that LLM generalization in temporal extraction tasks cannot be predicted from any single dimension alone and cannot be reliably inferred from in-domain or single-dimension evaluations, highlighting the need for reasoning strategies that generalize across dimensions. 

---
# A generative-informed neuro-symbolic framework for syntactic ambiguity resolution: Evidence from Arabic DPs 

**Authors**: Mohammed Damom, Muneef Y. Alshawsh, Ashraf A. Naji, Mustafa Ali Alhamzi, Fawwaz An-Nashef, Jameel Ahmed Elayah, Mohammed Q. Shormani, Noman AL-Sayadi  

**Link**: [PDF](https://arxiv.org/pdf/2610.02529)  

**Abstract**: Syntactic ambiguity poses a persistent challenge for Arabic NLP, particularly in morphologically rich nominal constructions where multiple structu6ral interpretations may be compatible with the same surface sequence. This study proposes a generatively informed neuro-symbolic framework for resolving structural ambiguity in Modern Standard Arabic (MSA) DPs. The framework integrates generative syntactic notions with AraBERT by representing ambiguity as a candidate-based decision task in which linguistically motivated alternatives are explicitly constructed and evaluated through candidate-conditioned input representations. Findings indicate that the model achieved 96.88% accuracy, 95.92% macro-F1, 96.83% weighted F1, and 93.94% binary F1 on the unseen evaluation set. Class-level analysis revealed asymmetric performance, with recall of 99.71% for High/VP Attachment (N1) and 89.26% for Low/NP/Embedded Attachment (N2), indicating greater difficulty in recovering the embedded interpretation. The study concludes that formal syntactic representations can be operationalized within Transformer-based NLP as an explicit interface between linguistic structure and contextual neural modeling, providing a controlled and interpretable approach to Arabic syntactic ambiguity resolution and beyond. 

---
# From Retrieval to Typed Decisions: Calibrated System One Models from Biomedical Sentence Encoders 

**Authors**: Pritam Deka  

**Link**: [PDF](https://arxiv.org/pdf/2610.02486)  

**Abstract**: Typed decision models answer schema-constrained questions about a text in one forward pass and return probabilities meant to be thresholded. We ask whether biomedical sentence encoders trained for retrieval are good starting points for such models. We present SBERT2S1, which converts Sentence-Transformers encoders into bi-encoder, cross-head (C) and prior-fused residual (PFR) decision models, together with BIODECIDE, a biomedical typed-decision suite, and MEDLINE-S1, 243k training decisions derived from NLM indexing. Across six parent-retriever pairs, retrieval training improves zero-shot matching of content-bearing options. After fine-tuning, its effect depends on the head: across five pairs and three training-set sizes, retrieval training significantly helps PFR, which keeps the retrieval prior, in 10 of 15 comparisons, but helps C in one and hurts it in five. A matched grid of two heads and five training objectives shows that C outperforms PFR under every objective, and that the released RLCD recipe of open System One models trails cross-entropy by 2.5-3.0 points. The deficit stems mainly from its reward normalisation, which inflates the noisy score-function term 3.6-15-fold; an unbiased leave-one-out estimator recovers most of the gap. After temperature scaling, no objective is clearly better calibrated than cross-entropy. We release the code, the MEDLINE-S1 labels and a model. 

---
# APDMem: Agent-Controlled Progressive Disclosure for Query-Adaptive Long-Term Memory 

**Authors**: Chin-Lun Fu, Anagha Kulkarni, Hong Ni, Behrouz Madahian  

**Link**: [PDF](https://arxiv.org/pdf/2610.02472)  

**Abstract**: Personalized LLM assistants must recover sparse evidence from long conversation histories across queries of varying complexity. We introduce APDMem (Agent-controlled Progressive Disclosure Memory), a hierarchical long-term memory architecture that applies progressive disclosure to memory retrieval. Rather than relying on a flat memory store or fixed retrieval granularity, APDMem represents conversation history as four progressively detailed layers: thematic summaries, personalized key facts, turn-level evidence notes, and raw messages. At inference time, a controller applies progressive disclosure to the memory hierarchy: it first reads high-level summaries and drills into finer evidence only when needed. This creates an adaptive cost-fidelity trade-off: simple queries can terminate early, while complex temporal, multi-hop, or exact-evidence queries trigger deeper inspection. A note synthesizer converts retrieved evidence into a query-focused structure that consolidates facts, orders events, and flags contradictions before final answer generation. Experiments on LongMemEval show that APDMem achieves strong performance for long-context memory reasoning while accessing only 8% of the total conversations. 

---
# CUEing User Simulators: Calibrated User Embeddings for Multi-Turn Benchmarking 

**Authors**: Anjali Kantharuban, Jonas Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2610.02460)  

**Abstract**: Recent benchmarks rely on user simulators to evaluate AI agents in multi-turn interaction. While existing simulation techniques demonstrate surface fidelity to human style and behavior, ecologically valid interactive benchmarking also requires alignment in when and how agents fail across simulated and real user populations. We find that existing simulators lack outcome calibration: agreement with observed success rates and failure patterns when real users interact with the same agent. We introduce Calibrated User Embeddings (CUE), a framework that both encodes observed sessions and samples continuous representations, then decodes them into persona commands to steer LLMs to act as user simulators without training. Through this, we evaluate user-conditioned replay of past sessions and aggregate metric agreement when sampling novel personas for the same tasks. On $\tau^2$-Bench, CUEd simulators commit fewer simulator-attributed errors and more faithfully reproduce real-user agent failure modes, aggregate success rates, and outcomes for specific task-user pairs than other persona-based simulation methods. These gains coexist with competitive user fidelity as measured using metrics established in prior work. After being fit to mostly customer support interactions, the same CUEd simulators generalize to document creation, math tutoring, and casual conversation, and remain effective across different simulator LLMs without CUE retraining. 

---
# FinDialogLens: Event Extraction over Multi-Party Dialogue for Missed-Trade Identification in Financial Chatrooms 

**Authors**: Chin-Lun Fu, Hong Ni, Behrouz Madahian  

**Link**: [PDF](https://arxiv.org/pdf/2610.02455)  

**Abstract**: Multi-party financial chatrooms are vital for sales-and-trading professionals, but their complexity makes manual recovery of missed trades infeasible: each Request for Quote (RFQ) is an event whose final price and trade outcome appear many messages after the RFQ-trigger message (the inquiry message), interleaved with concurrent RFQs from other participants. We cast this as event extraction (EE) over multi-party dialogue and present FinDialogLens, a hybrid LLM pipeline in which compact fine-tuned classifiers act as inference-time scaffolds: they detect RFQ-triggers and price/trade outcome metadata, an RFQ-Level Module segments per-event RFQ windows, and a Trade Engine fills argument roles. With GPT-4o, FinDialogLens reaches 92.1% and 94.3% accuracy on final price and trade outcome, respectively, outperforming full-chatroom CoT prompting methods; fine-tuned open-source LLMs with as few as 3B parameters achieve comparable performance with modest in-domain data. To make the LLM-based solution practical at scale, a difficulty-aware router balances cost and accuracy by allocating RFQs between a low-cost rule-based engine and the higher-performing LLM-powered Trade Engine, cutting LLM calls by 85% on final price while recovering half of the accuracy gap to FinDialogLens (GPT-4o), saving over $300/day at our 70,000-RFQ/day scale. 

---
# Counterexample Generation via Per-Theorem Symbolic Verifiers: When Imitation Hurts and Reinforcement Repairs 

**Authors**: Omar Farouk Zouak, Houssam Eddine Boukhalfa, Soumaya Lakehal, Shiv Katiyar, Samia Nefti-Meziani  

**Link**: [PDF](https://arxiv.org/pdf/2610.02444)  

**Abstract**: Large language models often solve a theorem forward yet fail to disprove a closely related false one: a falsification gap that supervised fine-tuning does not close and can actively worsen. We frame counterexample generation as constrained witness emission against a deterministic per-theorem Python verifier, and release SymCE, a corpus of 4,707 false undergraduate-algebra and real-analysis conjectures, each paired with executable verifiers. The verifier also serves as the reward function, making SymCE a training environment. Training Qwen3-4B with SFT followed by GRPO under this oracle reveals an imitation trap: counterexample-only SFT collapses true-theorem recognition from 0.27 to 0.00, while RLVR with a sparse outcome-only reward repairs this and exceeds the base, to 0.66. The collapse replicates across four seeds and on Gemma-3-4B. Sparse and dense rewards yield statistically indistinguishable in-domain success yet diverge by 33 points on a held-out calibration probe, a dissociation we trace to the partial-credit term. Our 4B model outperforms every evaluated 7B open-weights math specialist, remains competitive with six frontier commercial APIs, and transfers under unchanged prompting to GSM8K, MATH-500 and MMLU-college-math. A human audit of 177 verifier decisions finds 97.7% accuracy. Code, data, verifier modules and annotations: this https URL. 

---
# Finding the Move Is Not Winning the Game: XiangqiBench for Closed-Loop Evaluation of LLM Agents 

**Authors**: Yekun Chai, Qiwei Peng, Haoyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2610.02425)  

**Abstract**: Static evaluations credit a language model for naming the right move, but an agent must carry a plan through to a verified outcome while an opponent responds. We introduce XiangqiBench, an executable benchmark that measures this difference in Chinese chess: starting from 119 tactical endgames with forced mates supported by engine or checks-only search, an LLM agent must deliver checkmate against an engine defender. An interactive REPL interface separates real moves, state queries, and forward simulation, and we record 8,568 multi-turn trajectories from 12 frontier LLMs under two observation protocols. Three signals that look like competence each overstate closed-loop success. (i) The Conversion Gap: models play the stored reference first move in 26.1\% of Sighted trials, yet only 13.9\% of these trials end in a win. (ii) The Consistency Gap: the leading model reaches 38.7\% pass@3 but only 5.9\% pass^3, winning all three trials on 7 of the 46 positions it ever wins. (iii) The Simulation Gap: 32.3\% of accepted simulation calls stop on an illegal move, and in 49.3\% of comparable cases the real defender replies differently from the line the agent simulated; self-authored rollouts check legality but cannot anticipate the opponent. Finding the move is not winning the game: agent evaluations should score closed-loop outcomes and report reliability alongside coverage. 

---
# HakemBench: A Turkish Benchmark of Typed Decisions 

**Authors**: Sait Furkan Teke  

**Link**: [PDF](https://arxiv.org/pdf/2610.02293)  

**Abstract**: HakemBench is a Turkish benchmark of typed decisions, in which the model under test reads a text, a question and a fixed set of options and returns a probability for every option. Version 1.0 is released fully open under CC BY 4.0, with 2,346 items and 4,275 choice, yes/no and score questions in seven tracks (fact-check triage, education, guardrails, legal routing, moderation, spam and phishing, and customer support). One harness scores decision quality (macro F1), calibration (from the normalised Brier score) and selective automation (from the normalised area under the generalised risk-coverage curve), combines them by a geometric mean and reports intervals from 2,000 bootstrap draws; probes for option order, paraphrase, English translation and substituted names are reported alongside. Most gold labels come from blind passes of one AI model family compared with the votes of a panel of large language models from other model families; they are not human-verified. On a board of 16 rows the leader scores a composite of 0.888 and the lab's own model is 7th at 0.660. Its numbers are not blind. Earlier runs' test results shaped its training data, so its guardrail, moderation and customer support numbers are flagged; with every model scored on the other four tracks only, its composite is 0.678, 6th of 16. 

---
# FrugalEvo: Towards Cost-Aware LLM-Guided Program Evolution 

**Authors**: Hui Chen, Xuan Qi, James Xu Zhao, Zhaopeng Feng, Shilong Liu, Kuang Xu, Pang Wei Koh, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2610.03675)  

**Abstract**: LLM-guided evolutionary methods, such as AlphaEvolve, have emerged as powerful approaches for challenging computational optimization problems, such as circle packing. However, prior work typically optimizes performance gain over a fixed number of iterations. We argue that practical optimization should maximize gain per unit cost. To this end, we propose FrugalEvo, a cost-aware evolutionary framework where a stronger, higher-cost LLM explores solution strategies, and a cheaper LLM implements them and iteratively refines the resulting code. We also design a cache-efficient evolution process, where our harness and prompts maximize the sharing of prefixes across different evolution steps, to improve cache reuse. To measure solution quality throughout a fixed cost budget, we introduce Budget-Aware Area Under the Curve (BA-AUC), defined as the area under the best-so-far evaluation score curve over cumulative LLM cost, up to the budget. Across 10 mathematical and systems optimization tasks, FrugalEvo matches or surpasses state-of-the-art baselines, including OpenEvolve, ShinkaEvolve, AdaEvolve, and EvoX, in final solution quality and achieves higher BA-AUC on 9 tasks. It also achieves higher average performance than these baselines on 10 algorithmic optimization tasks from ALE-Bench-Lite. Notably, on circle packing, FrugalEvo achieves new state-of-the-art performance with GPT-5.6 Terra and Luna for only 1.68 USD and with GLM-5.3 and its Flash variant for only 0.55 USD, matching or surpassing all baselines, including multi-agent methods such as CORAL and SwarmResearch, which cost approximately 50 USD on average. 

---
# Pivot-SD: Efficient Self-Distillation for Masked Diffusion Language Models 

**Authors**: Seo Hyun Kim, Sunwoo Hong, Younwoo Choi, Chen-Hao Chao, Se-Young Yun, Rahul G. Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2610.03665)  

**Abstract**: Masked diffusion language models (dLMs) offer a promising parallel alternative to autoregressive models for complex reasoning. However, they face a distinct credit-assignment challenge, since a few commitments during denoising sharply reduce the uncertainty over the remaining masked positions and shape much of the response. Most post-training recipes for dLMs do not use this signal to decide which tokens to train on: they typically train on the final text or assign rewards to whole denoising steps, rather than selecting the individual commitments that shape the response. We introduce Pivot-SD, an efficient offline self-distillation framework that supervises only these high-impact commitments (pivots). Pivot-SD selects pivots using an information-gain metric measuring uncertainty reduction over the remaining masked positions. Pivots from successful trajectories are trained with cross-entropy, and pivots from failed trajectories with targeted unlikelihood, leaving the rest of the failed trajectory untouched. Using only 200 questions and four rollouts each, Pivot-SD improves LLaDA-8B-Instruct over full-sequence SFT and budget-matched diffusion RL baselines across math and code benchmarks. 

---
# World Embedding Benchmark 

**Authors**: Yiqi Liu, Ruifeng Yuan, Yang Wang, Long Li, Fengyu Cai, Hou Pong Chan, Jialin Yu, Hao Zhang, Chenghua Lin, Chenghao Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.03632)  

**Abstract**: Physical fidelity has received increasing attention in world models and video generation, yet how video representations encode physical information remains less understood. We introduce the World Embedding Benchmark, comprising 8,000 controlled simulation cases from 80 families spanning fluid mechanics, solid mechanics, dynamics, and optics & electromagnetism. Each case pairs a rendered video with simulation-derived physical annotations, supporting three complementary tasks: text-video retrieval, physical-property regression, and multiple-choice video-description pair classification. We use these tasks to distinguish cross-modal physical alignment from the recoverability of quantitative physical information. Evaluated pre-trained omnimodal embedding models show weak retrieval and near-chance within-family pair classification, while lightweight probes recover useful physical information from frozen video embeddings. Continual contrastive training with physics-specific video-text pairs improves retrieval and pair classification but degrades physical-property regression, revealing a trade-off between alignment and quantitative information recoverability. Finally, we use the embeddings to retrieve reference videos for retrieval-augmented generation with MiniMax-H3. Retrieved references improve the physical fidelity of generated videos, with stronger retrieval models yielding larger gains in our experiments. Together, these findings highlight the need to evaluate physical alignment and property recoverability jointly, and demonstrate the utility of physical representations for improving video generation. 

---
# Divergence controls entropy in distillation 

**Authors**: Nicolas Zucchet, Scott W. Linderman  

**Link**: [PDF](https://arxiv.org/pdf/2610.03529)  

**Abstract**: Distillation has become a core primitive of large language model training, but its properties are not yet well understood. We take an entropic perspective, studying how the entropy of the student depends on the data and the divergence that define the distillation objective. We prove that forward KL inflates the entropy of the student above that of the teacher. Since cross-entropy training is a special case, this yields an identity that we verify quantitatively in pretraining and supervised finetuning. Other divergences come with no such guarantee: reverse KL deflates entropy until the gap between student and teacher gets too large, and interpolating between the two changes entropy smoothly early in training but abruptly at convergence. The lower entropy of on-policy distillation comes from token-level reverse KL, not from on-policy sampling. The divergence therefore acts as an implicit entropy regularizer, whose role is clearest in self-distillation: as conditioning on privileged information deflates entropy, the divergence hyperparameters that work best are those that compensate for it. 

---
# A Near-Zero Monitor Readout Is Not Evidence of Behavioral Control 

**Authors**: Zhe Zhou, Tianhua Tao  

**Link**: [PDF](https://arxiv.org/pdf/2610.03458)  

**Abstract**: Post-training with verifiable rewards can induce reward hacking, motivating the use of monitors within the training objective rather than solely for offline auditing. We show that a low monitor readout does not identify whether such an intervention controls behavior. In a code-generation environment whose dominant exploit is available at the start of the reasoning trace, we train policies against three monitors that pass the same offline gate: an in-domain activation probe and two penalties conditioned on how early the policy commits to its own final answer. The probe score is at its numerical floor from the first recorded training step, and the trained-score median is zero for every prefix-trained run at the endpoint. These readouts estimate different quantities, and we do not compare their scales; within each monitor family, however, low values do not establish behavioral control. Within one fixed configuration, prefix-trained runs with the same zero-median trained score range, by seed alone, from a mixed regime with a low hacking share to near-pure reward hacking. All probe runs reach the hacking regime, but their floor-level readout reflects a mismatch between the position where the probe was validated and the position where it was read during training, not a second instance of this ambiguity. Text-level analysis identifies a prefix failure mode: generic planning and filler shells postpone the exploit past the cut without eliminating it from the final output. Low measured commitment therefore does not distinguish a low hacking share from delayed commitment to the exploit. Offline discrimination and low monitor-aligned readouts are insufficient evidence of behavioral control; an out-of-band behavioral check is required. We characterize the endpoint readout, not its evolution. Code is available at this https URL. 

---
# Passing the Test You Trained On: Re-evaluating Prompt-Injection Detectors for LLM Agents 

**Authors**: Zhuowen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.03448)  

**Abstract**: LLM agents increasingly screen tool outputs with small prompt-injection detectors, and teams choose among detectors by their scores on public benchmarks. We ask whether those scores predict how a detector behaves inside an agent. We replay the ground-truth tool calls of two agent benchmarks, AgentDojo and tau-bench, without an LLM to obtain tool outputs that are benign by construction, label injected outputs by differential replay, and evaluate fifteen detectors, including Meta's Prompt Guard 2, and two task-aware LLM judges on these outputs and on the BIPIA benchmark. Detection rankings transfer poorly between benchmarks: the best detector on BIPIA catches 2% of AgentDojo injections at a 1% false-positive rate, and a detector that catches 72% of AgentDojo injections catches 15% on tau-bench. False-positive rates on tool outputs, which range from none to over 90%, do transfer between the two agent benchmarks. Where training data is public, the form of the training inputs explains the results. The BIPIA leader was trained on full BIPIA inputs, but having seen InjecAgent's attack strings as short prompts does not help it find them inside tool outputs; the best detector on both agent benchmarks shares no data with any benchmark and was trained on agent-style inputs. Evaluations meant to inform deployment should use the agent's own tool outputs, report detection at a low false-positive rate, and audit what the detector was trained on. 

---
# Benchmarking Candidate Coverage in Typed Decision Models 

**Authors**: Jiawen Lu, Tongtong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.03387)  

**Abstract**: Typed decision models return choices or distributions over answer options supplied at request time. Accuracy with complete options does not establish whether a model recognizes that a reference answer is missing or avoids rejecting valid candidates. We present a paired candidate-coverage benchmark protocol and an initial evaluation of Laya and Jev across AG News, DBpedia, Emotion, and TREC. The models receive identical frozen texts and requests: 300 calibration and 589 test texts yield 23,932 predictions per model. Present/absent pairs match ordinary candidate count, and name variants preserve descriptions, members, and order. Native rejection behavior differs sharply: at five TREC candidates with natural names, Laya detects 97.2% of missing-answer cases but falsely rejects 69.7% of present controls; Jev's rates are 24.8% and 0.0%. Calibration-only none-score thresholds change these rates to 33.9%/3.7% and 45.0%/1.8%, respectively. On DBpedia, Jev's high coverage-score AUROC supports a stronger operating point, whereas both models have weak complete-set accuracy on Emotion. Competence-conditioned analysis, probability-precision sensitivity, and interface audits show why classification, score ranking, and rejection policies need separate measurement. This initial benchmark is descriptive and limited to reference-label omission; it does not establish natural out-of-scope generalization, causal mechanisms, or a new rejection method. 

---
# Multilingual GSM-Symbolic: What determines capability transfer across languages? 

**Authors**: Kenneth Enevoldsen, Riley Herchert, Sofie Mosegaard, Dan Saattrup Smart, Simon Enni, Isaac Chung, Sofie Bruun, Ayush Sunil Munot, Max Müller-Eberstein, Adnan El-Assadi, Elisa Bassignana, Gianluca Barmina, Hafsteinn Einarsson, Iben Nyholm Debess, Linda Freienthal, Lukas Galke Poech, Mike Zhang, Nicolas Legrand, Vladimir Salnikov, Yevhen Kostiuk, Zafar Hussain, Sagandeep Kaur, Agnes Toftgård, Marie Mattson, Kristoffer Nielbo  

**Link**: [PDF](https://arxiv.org/pdf/2610.03367)  

**Abstract**: We understand little about how capabilities acquired in one language carry over to another, or what governs this transfer: evaluations rely on incomparable, saturation-prone datasets and rarely examine its determinants jointly. Identifying what predicts transfer would let us avoid exhaustive evaluation across all language pairs and let developers target the factors that limit performance in low-resource languages. To evaluate cross-lingual capability transfer, we introduce Multilingual GSM-Symbolic, an extensible multilingual mathematical dataset covering 30,000 item-matched question-answer pairs and spanning 15 languages. It utilises symbolic templates to prevent overfitting and ensure generalisation by allowing generation of millions of high-quality variations from a single sample. Using Multilingual GSM-Symbolic, we quantify the largest determinants of capability as model size ($\beta = 1.77$), language resource level ($\beta = 0.77$), reasoning ($\beta = 0.67$) and typological distance ($\beta = -0.25$). This joint estimation allows these determinants to be expressed in terms of one another: a 32B model evaluated in Marathi performs like a 10B model in English. Our findings have important implications for model developers, showing that model size and reasoning narrow the performance gap between low- and high-resource languages ($\beta = -0.27$ and $\beta = -0.20$, respectively), while similar levers have little or no effect on typologically distant languages.
Overall, our analysis framework explains 92% of between-language variation, but only 23% of the model-by-language variation, and predicts a model's performance on an unseen language within 6.0pp (r=.96). Incorporating measurements from just 10 templates in the target language reduces this to 4.19pp, enabling reasonable estimates of performance with little or no downstream dataset. 

---
# AdaStep: Adaptive Step Credit Weighting for Agentic Reinforcement Learning 

**Authors**: Xin Wang, Wenhao Wu, Menghao Zhang, Zhi Wang, Kun Shao, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2610.03223)  

**Abstract**: Long-horizon LLM agents are typically trained with sparse outcome rewards, making trajectory-level objectives too coarse to distinguish the contribution of individual decisions. Step-level credit assignment provides finer-grained supervision, but its estimates can be unreliable because observed returns also depend on subsequent actions, environment transitions, and trajectory length. We propose AdaStep, an Adaptive Step-credit weighting method that controls how strongly each group-derived local advantage modifies the trajectory-level signal. We formulate this weighting as a mean-squared-error estimation problem for the latent step advantage and, under an explicit conditional sampling assumption, derive an optimal per-state shrinkage coefficient. The coefficient admits a signal-to-total-variance interpretation: it preserves local credit when return variation is attributable to the selected action and suppresses it when variation is dominated by downstream randomness. AdaStep requires only lightweight scalar computation, with no critic, additional rollouts, or extra model inference. Experiments with three model backbones on ALFWorld, WebShop, and ScienceWorld show consistent improvements over baselines at low computational cost. 

---
# Predicting and Repairing Merge Collapse in Large Language Models 

**Authors**: Jungseob Lee, Seungyoon Lee, Sugyeong Eo, Hyeonseok Moon, Jaehyung Seo, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2610.03199)  

**Abstract**: Large language models fine-tuned from a shared base can be merged by averaging their task vectors, but some merges collapse far below the base model, and common merge operators give no warning before evaluation. We show that one statistic of the specialists' task vectors both predicts this collapse and calibrates its repair. The power that averaging removes equals the variance of the task vectors across specialists, our measure of interference. Under a working noise model, the disturbance that a merge injects grows with the merge coefficient and with interference, yielding a pre-merge score. In our experiments on twenty-two merge configurations from four model families, only destructive merges exceed a threshold on this score. We find that statistics of sign conflict between specialists, a common target of existing merge operators, are anti-predictive. We then predicted the outcomes of fourteen merges before evaluating them, and twelve predictions were correct, including the destructive outcome of a specialist pair pushed past the threshold by continued pretraining. To address this collapse, we introduce PRISM, an operator that averages the task vectors first and then soft-thresholds each layer at a level set by the layer's interference. Without data or tuning, PRISM keeps all five destructive merges above the threshold within evaluation noise of the base model, where plain averaging falls at least 14.4 points below it or collapses entirely. We apply PRISM only above the threshold and keep the plain average for merges below it, which include all fifteen harmless ones. Code is available at this https URL. 

---
# KV$^2$: A Self-Refining KV Cache 

**Authors**: Johannes Wesch, Danni Liu, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2610.03198)  

**Abstract**: The memory footprint of the key-value (KV) cache constrains the practical use of long-context models, and it dominates cost when one prefilled context must later serve many different queries. In this reusable setting, query-agnostic compression trades cost against quality: lightweight estimators are cheap but less accurate, whereas full-context reconstruction scoring is more accurate yet reprocesses the entire prompt. We introduce KV$^2$, a query-agnostic KV-cache compression method based on selective reconstruction. KV$^2$ first uses a lightweight proxy scorer to identify informative in-context tokens, then reprocesses only this subset to compute final eviction scores. On RULER, Needle-in-a-Haystack, and LongBench, KV$^2$'s margin over baselines widens as the budget tightens: on RULER 16K at a 2% KV-cache budget it improves the average score over the next-best baseline by more than 40 percentage points, and on LongBench it attains the highest average across 2%-10% budgets at lower compression-stage runtime and peak memory than full-context reconstruction. Reusable KV-cache compression thus does not require reprocessing the full context. Our code is available at this https URL. 

---
# Gains and Collapse in On-Policy Distillation:A Reinforcement Learning Perspective 

**Authors**: Han Cui, Jianhao Yan, Yun Luo, Hongbo Zhang, Zhizhang Fu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.03185)  

**Abstract**: On-policy distillation (OPD) has become an important approach to language model post-training. However, despite its performance gains, OPD can also collapse into excessively long and repetitive generation, and the mechanism underlying these divergent outcomes remains poorly understood. We explain these outcomes through a reinforcement learning perspective: the teacher implicitly rewards student behaviors, even those it rarely exhibits itself. From this perspective, our experiments show that OPD improves performance without expanding the student's capabilities. When the implicit reward model is reliable, OPD makes correct responses easier to sample. In contrast, when the preference misaligns with quality, reward hacking happens: the implicit reward model amplifies overlong, repetitive student rollouts, even though it rarely generates such text itself. Guided by this diagnosis, we find that masking unhealthy responses during training and using SFT initialization can each effectively mitigate the collapse. Together, these findings show that OPD amplifies student behaviors favored by the teacher's implicit feedback, shifting the focus from how well the teacher generates to how reliably it evaluates student rollouts. Our code is available at this https URL. 

---
# Benchmarking Literature Retrieval for a Model Organism: A Dictyostelium Case Study 

**Authors**: Yun Wang, Gad Shaulsky, Tomaž Curk, Blaž Zupan  

**Link**: [PDF](https://arxiv.org/pdf/2610.03130)  

**Abstract**: Biological literature retrieval systems are often developed and evaluated using broad biomedical corpora and general-purpose search tasks. However, many curated knowledge bases operate in narrower model-organism domains, where the literature is sparse and terminology is organism-specific. We introduce a retrieval benchmark from dictyBase for Dictyostelium, a model organism in cell and developmental biology. The benchmark consists of curator-generated biological queries linked to PubMed-indexed articles, together with structured gene annotations. Using this benchmark, we study three factors in niche biological retrieval: cross-encoder reranking, gene-aware query expansion, and abstract-only versus full-text retrieval. We report that reranking and gene-aware query expansion improve retrieval selectively: reranking is most useful when the model is well suited to biological evidence matching, whereas curated annotations help clarify compact biological queries by reducing vocabulary mismatch. Full-text chunks substantially improve retrieval when abstracts omit supporting evidence, increasing both candidate recall and top-rank performance, although these cases are harder than queries supported by abstracts. Data and code are publicly available at this https URL, and the benchmark dataset is additionally archived on Zenodo. 

---
# The Fragility of Trigger-Tag Mechanisms for Misuse Detection in Open-Weight LLMs 

**Authors**: Toluwani Aremu, Manit Baser, Mohan Gurusamy, Nils Lukas, Dinil Mon Divakaran  

**Link**: [PDF](https://arxiv.org/pdf/2610.03124)  

**Abstract**: Open-weight language models can be downloaded, modified, and deployed beyond their developers' control, limiting the effectiveness of centrally enforced safeguards. Recent work has therefore proposed \emph{trigger-tag} mechanisms that produce a detectable signal when a model is used under a target condition, such as generating phishing contents. Although these mechanisms borrow from established techniques, their use for conditional misuse detection in open-weight LLMs is relatively new. Therefore, existing research works have not systematically studied the robustness of trigger-tag mechanisms under adversarial attacks. To close this gap, (i)~we formalize trigger-tags and distinguish \emph{token-level trigger-tags}, which introduce watermark-inspired signals during decoding, from \emph{weight-level trigger-tags}, which learn backdoor-inspired associations between target conditions and detectable model behavior. Furthermore, (ii)~we introduce \Untag, a unified attack framework that organizes their mechanism-specific attack surfaces into a common taxonomy. We evaluate representative token-level and weight-level trigger-tags using phishing as a case study. We find that while trigger-tags may provide useful evidence in controlled settings, our attacks render the existing trigger-tag mechanisms to be entirely ineffective. Consequently, we argue that these mechanisms should not be treated as robust misuse detectors when attackers can transform outputs or modify open weights. 

---
# Peer Influence across Heterogeneous AI Models 

**Authors**: Frida Nøhr Laustsen, Marie Haahr Petersen, Victoria Popa, Ariel Flint, Romualdo Pastor-Satorras, Andrea Baronchelli, Luca Maria Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2610.03095)  

**Abstract**: When two AI agents disagree, who persuades whom? As multi-agent systems increasingly combine language models of different families and sizes, the answer can determine which judgments survive interaction. Measuring persuasion as the probabilistic shift in an agent's decision after a single exchange with a dissenting peer, we test seven open-weight models across three language understanding tasks. We find that persuasion is strong: when models disagree, receivers often abandon their initial judgment after seeing a peer's answer and explanation. Surprisingly, however, neither standalone certainty nor model scale reliably predicts persuasion dynamics. Models producing almost perfectly consistent decisions in isolation can be among the most susceptible to persuasion, and small models can match larger ones as persuaders and resist their influence just as effectively. Furthermore, we show that the size of the shift depends more on the susceptibility of the listener than on the persuasiveness of the speaker. Persuasion patterns are therefore specific to each model pairing, with heterogeneity amplifying persuasion in some combinations and suppressing it in others, allowing a dissenting agent running a small model to overturn the judgments of a much larger one. These findings show that the behavior of interacting models cannot be inferred from their individual properties but must be evaluated in the combinations in which they will operate. 

---
# MintEval: Do LLMs Implement the Trading Strategy You Asked For? A Behavioural-Equivalence Benchmark for Natural-Language-to-Strategy Code 

**Authors**: Siyu Wang, Yifan Wang, Yuecheng He  

**Link**: [PDF](https://arxiv.org/pdf/2610.03080)  

**Abstract**: Large language models are moving from producing trading signals to writing the code that executes them. The failure mode of the second role is silent: generated code runs, a backtest plots, yet the risk logic that the trader described is not the logic being executed. Existing code benchmarks test functional correctness on unit tests and finance benchmarks test forecasting; neither measures whether an implementation behaves like the strategy that was asked for. We introduce MintEval, a benchmark in which reference strategies are generated programmatically from a library of composable building blocks, back-translated into colloquial trader instructions, and re-implemented by the model under test. Generated and reference programs are executed bar by bar on identical market data and frictions, and compared on their actions rather than on code similarity or profit: alpha is differenced away. MintEval v0 contains 800 tasks on BTCUSDT 15-minute data, stratified by an execution-measured state-span complexity tau that is decoupled from description length. Low-cost models reach a mean ActionMatch of at most 0.544 and reproduce at most 0.087 of tasks exactly; on a stratified subset of 200 tasks a frontier model (Claude Opus 5.5) reaches 0.889 and reproduces 0.575 exactly, yet still fails silently on 0.275 of tasks. Given a menu of building blocks, models identify the strategy almost perfectly, yet 79.2% of the implementations whose specification was read correctly diverge on more than 10% of active bars. The LLM judge of a recent strategy-generation benchmark, applied verbatim, accepts every one of these silent failures. 

---
# SecJev: Bringing Security Expertise to System One Decision Models 

**Authors**: Zheng Chen, Fei Yu, Haohao Huang, Yang Li, Anlong Chen, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.03073)  

**Abstract**: Security workflows need models that turn complex observations and explicit policies into decisions. System One models introduced by Jev return typed predictions and probabilities; security specialization supplies the domain expertise behind those predictions. We introduce SecJev, to our knowledge the first family of Jev-like decision models specialized for security, spanning 0.8B to 9B parameters. Built on Kev's single-pass candidate scorer, SecJev learns Boolean, choice, and ordered decisions from text, telemetry, and observation histories. We develop SecJev-Corpus to unify source-label prediction and explicit-policy evaluation across 14 tasks and eight sources. It covers tool outputs, traffic, federated updates, consensus, authentication, and vehicle messages. Scene-weighted training adapts the models across these domains while preserving a shared typed decision interface. Security specialization improves every model in the family; SecJev-0.8B outperforms general Kev-9B by 20.51 percentage points in task-macro accuracy. Comparisons with answer-only generative fine-tuning show close accuracy and latency with lower peak inference memory. Tests on new source groups reproduce gains over Kev in prompt-injection and traffic decisions, with capture-dependent false alarms. We release adapters, decision heads, SecJev-Corpus, and training and inference code. 

---
# Adaptive Second-Order Solvers for Fast Stochastic Diffusion Sampling 

**Authors**: Ella Kemperman, Luca Ambrogioni  

**Link**: [PDF](https://arxiv.org/pdf/2610.03034)  

**Abstract**: Diffusion models rely on numerical solvers requiring time-discretization, which has a large influence on the tradeoff between sampling cost and quality. However, the computational difficulty of the reverse process varies along the sampling trajectory and across data distributions, making the choice of discretization important. We adapt proportional-integral (PI) step-size control to diffusion, using our diffusion noise-normalised error estimator. Unlike existing adaptive methods in diffusion that respond only to the current error, the PI solver also incorporates the previous error, yielding smoother step adaptation. We further show that these per-sample trajectories exhibit shared structure and can be aggregated into a fixed schedule that retains much of the benefit of adaptive sampling. We evaluate both approaches on natural-image and language datasets, in terms of quality, measured by FID at a matched number of neural network evaluations (NFE), comparing them with widely used stochastic solvers and schedules. For images, our fixed discretization outperforms the commonly used EDM schedule in terms of sample quality when used with the stochastic Heun sampler, and with the EDM-churn sampler at low NFE. Additionally, our PI adaptive solver obtains better FID than most stochastic and adaptive baselines, although it does not beat the EDM-churn sampler at low NFE. Moreover, we find our solver outperforms both the EDM and the entropy schedule on language diffusion at low-to-medium NFE in terms of perplexity, with the drawback of lower token entropy. Lastly, we find that the benefit of per-sample adaptivity is problem-dependent. It is highly beneficial in 1D toy examples, while only marginal for image and language data, where the average schedule sometimes even outperforms the PI-adaptive solver. Code is available at this https URL 

---
# Tailoring the Quantization Space for 1-Bit KV Cache Compression 

**Authors**: Minsoo Cheong, Donghyun Son, Sungjoo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2610.03027)  

**Abstract**: The key-value (KV) cache becomes a major memory bottleneck in long-context LLM inference, placing substantial pressure on memory capacity and bandwidth. To mitigate this bottleneck, vector quantization (VQ) has emerged as a promising approach for aggressive KV cache compression. However, existing VQ methods degrade substantially in the 1-bit regime. At such extreme compression, each codebook must represent a larger group of channels with a limited set of centroids, making effective use of its capacity increasingly challenging. To address this, we introduce $\textbf{TaSQ}$, which tailors the VQ target space by combining query-guided channel weighting, cross-head normalization, and covariance-aware channel grouping to better reflect the error sensitivity and statistical structure of cached activations. Since these transforms are RoPE-compatible and can be easily merged into projection weights and codebooks, TaSQ preserves the conventional VQ lookup structure and adds negligible serving overhead. Across general, long-chain-of-thought reasoning, and long-context retrieval benchmarks, TaSQ consistently outperforms existing low-bit KV cache VQ baselines while preserving reasoning stability. On a single RTX 6000 Ada GPU, its SGLang implementation supports up to $14\times$ larger batch sizes and achieves $1.87\times$ higher peak throughput compared to the BF16 baseline. 

---
# Verifiable, Articulable, and Tacit Components of Preference 

**Authors**: Alexander Spangher, Sheldon Huang, Andreas Haupt, Noah D. Goodman, Diyi Yang, Daniel E. Ho, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2610.03025)  

**Abstract**: What makes a short story gripping; a news article newsworthy; or a math proof elegant? These constructs resist articulation or verification; their meaning is at least partially tacit. However, modern AI models are improved primarily via articulated constitutions, rubrics and verifiers (i.e. in RLAIF and RLVR); tacit components of preferences are typically understudied. We introduce a large, labeled preference dataset CreativePreferences, containing 2.8M texts labeled by 317M human preference judgments across 7 creative domains, with 42 benchmark tasks. We model these labels with executable programs, rubric banks and densely trained models (V, A and VAT, respectively). We observe robust articulability gaps, VAT-VA; and verifiability gaps, VAT-V; we estimate upper and lower bounds for each gap with a novel measurement approach that discovers articulable and verifiable metrics, identifies spurious variables and estimates the value of undiscovered metrics using capture-recapture. These gaps occur across all domains, even in domains traditionally treated as fully verifiable: correctness-centered domains (i.e. mathematics and software engineering) and claim- and novelty-centric domains (i.e. news, patents, peer review). The size of the gap varies based on domain (e.g. peer review and creative writing have the largest articulability gaps) and widens as more people take part in the judgment, consistent with Collins' collective tacit knowledge. We show two consequences: (1) on human generations, the full model more closely matches human preferences, often in disagreement with articulated criteria, and (2) in an analogy to Goodhart's law, articulating preference shifts it away from the tacit dimension. Articulability and verifiability gaps are consequential; we give recommendations on when tasks can be prompted; how learning mechanisms might improve; and when to leave judgments with humans. 

---
# ReSCUE: Re-translation with Sentence Commitment for Unsegmented Long-Form Simultaneous Sign Language Translation 

**Authors**: Sihan Ren, Gaozheng Li, Yuanshang Quan, Yiming Qin, Fuyi Yang, Chang Liu, Lan Xu, Minye Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.03022)  

**Abstract**: Simultaneous Sign Language Translation (SLT) is critical for real-time communication, yet existing methods remain largely confined to sentence-level, offline settings that assume pre-segmented inputs. These assumptions hinder deployment in realistic scenarios involving continuous, unsegmented video streams. We present ReSCUE, a unified framework for simultaneous SLT on unsegmented long-form sign language videos that aligns training and inference with realistic streaming conditions. ReSCUE combines inference-aware training to handle partial inputs, non-signing pauses, and multi-sentence contexts, stabilized re-translation to enable low-latency yet revisable predictions with reduced output flicker, and a sentence commitment mechanism for online segmentation and memory management. Experiments on standard sentence-level benchmarks show that ReSCUE achieves lower latency and the best translation quality under low-latency settings. On long-form unsegmented datasets, ReSCUE approaches the translation quality of oracle offline systems that use ground-truth sentence boundaries, while operating at substantially lower latency, demonstrating its practicality for real-world streaming scenarios. 

---
# Personalized Automatic Speech Recognition for a Dysarthric and Tracheostomic Speaker using Artificial Conversations 

**Authors**: David Nadrchal, Monorama Swain, Florian Schmid, Gerhard Widmer, Paul Primus  

**Link**: [PDF](https://arxiv.org/pdf/2610.03017)  

**Abstract**: This work presents an automatic speech recognition (ASR) system personalized for a Czech speaker with a permanent tracheal stoma and severe dysarthria rendering their speech unintelligible to untrained listeners. We release a public dataset containing 33 annotated hours of the speaker's speech, collected using a novel "artificial conversation" protocol designed for high engagement and dialogue realism. We propose a multi-stage training pipeline based on Whisper Base: fine-tuning on standard Czech speech, acoustically simulated tracheostomic speech, and the speaker's data. We evaluate the system across three near real-time scenarios: scripted conversations, question answering, and spontaneous dialogue, achieving a 50\% relative reduction in Character Error Rate compared to Whisper Base baseline and surpassing the average recognition accuracy of their assistants in acoustic recognition of isolated utterances. We demonstrate that even for severely impeded speech, a helpful ASR is achievable, as evidenced by the quantitative results and the feedback from the speaker. 

---
# Sentry: Learning to Recover from LLM Agent Failures at Test Time 

**Authors**: Changxiu Ji, Amy Lu, Qizheng Zhang, Kunle Olukotun  

**Link**: [PDF](https://arxiv.org/pdf/2610.02994)  

**Abstract**: LLM agents often fail mid-task due to invalid tool calls, repeated actions, or poorly grounded reasoning, and learning from these failures is a path to reliability. We find that how failure knowledge reaches the agent matters as much as what it contains. Failure lessons are conditional: kept in the agent's context, they misfire when their failure is absent, and removing them from an evolving playbook improves performance. Runtime interventions, in contrast, act only when a failure occurs but do not learn from their repairs. We argue that failure knowledge is conditional knowledge and should be conditionally exposed, and instantiate this principle in Sentry, a failure-management layer that runs alongside the agent. When Sentry detects a failure, it retrieves matching lessons from an external playbook to guide recovery, verifies without access to task rewards whether the agent recovered, and stores a new lesson only if it did; the full playbook never enters the agent's context. Across multiple agentic benchmarks, Sentry outperforms the strongest runtime-intervention baseline on every benchmark, by 37\% on average, and the strongest context-evolution baseline by 39\% on the two benchmarks where both are evaluated; combining Sentry with context evolution yields further gains. Learned lessons transfer to held-out tasks, and controlled experiments show that exposing the full playbook to the agent lowers performance even when relevant lessons remain available on demand. 

---
# Understanding Trajectory Heterogeneity in Federated World Model Learning 

**Authors**: Yipan Wei, Zhaokun Yan, Ziming Hong, Jiaqi Wu, Lixu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.02957)  

**Abstract**: World models learn state evolution from trajectories, making access to temporal context a central training requirement. Federated learning can use distributed records, while ownership boundaries within a trajectory restrict the examples each client can construct. Our study benchmarks this cross-time setting through hourly action-conditioned clinical prediction on eight MIMIC-IV disease cohorts, comprising 40.87 million transition memberships. We specify severity-based client ownership, patient-separated construction, local history and future-window rules, and paired rollout evaluation from one to 32 hours. A matrix of ten federated algorithms covers 32 disease--partition configurations under five rounds of ten-percent participation. Three findings emerge from existing results and training logs. First, client ownership and participation jointly restrict long-window coverage: only 7.55\%--21.36\% of pooled-available 32-step windows have a locally complete anchor visited during training, averaged across diseases. Second, finer severity partitions accompany higher FedAvg error in 15 of 16 paired comparisons, while algorithm gains are small and horizon-dependent: FedProx reduces mean error by 0.56\%, with no consistent improvement at 32 steps. Third, algorithm labels conceal distinct update behavior, including inactive extrapolation and orders-of-magnitude differences in update scale. Cached-update performance also varies strongly across trajectory partitions under the same benchmark protocol. These results establish temporal access, participation coverage, optimization behavior, and horizon-resolved prediction as complementary dimensions for evaluating federated clinical world models. 

---
# Continual Graph Memory for Mathematical Research Agents 

**Authors**: Junyi Zhang, Jinxi Yu, Eric Hanchen Jiang, Jiachen Lu, Zhi Zhang, Xinjie He, Hyunsik Chae, Ethan Ji, Alexander K Taylor, Vigyan Sahai, Yiwen Kou, Kai-Wei Chang, Raghu Meka, Nanyun Peng, Amit Sahai, Terence Tao, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.02945)  

**Abstract**: Using frontier agent harnesses to tackle mathematical research problems has emerged as an effective means of advancing mathematics. However, solving frontier problems in mathematics may require a massive number of agents working in parallel for extended periods to construct proofs, thereby generating an enormous volume of intermediate proof results. Organizing these intermediate results throughout a long-horizon proof-search process and reusing knowledge gained from prior explorations remain major challenges. We present Ansatz, a mathematical research agent built around Continual Graph Memory, a graph-based, evolvable, cross-problem mathematical research memory system that explicitly organizes the entire proof search process and reuses information from exploration trajectories of previous problems. Specifically, we develop a unified graph memory that represents all intermediate exploration results, including facts, plans, and counterexamples, together with edges that explicitly represent the relationships among them; dependency-aware retrieval supplies precisely targeted local context; an evidence-sensitive curator updates the research frontier and distills lessons from prior attempts; and scoped recall surfaces earlier statements and negative findings for local re-proving rather than uncritical reuse. Experiments cover runs across all ten First Proof Second Batch problems, together with four component studies. Ansatz reports closure on all ten research tasks, demonstrating its ability to sustain and resume long-horizon mathematical search. Beyond these problems, Ansatz also produces solutions to the Jamison caterpillar conjecture and Erdős Problems 289, 348, and 488 without human intervention, and makes partial progress on several open problems, illustrating its strong ability to solve open mathematical research problems. 

---
# Probe the Harness: Setup Checks for Stale-Data RL Comparisons in Language Models 

**Authors**: Taiheng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2610.02911)  

**Abstract**: Methods for training language models on stale samples are judged by comparisons against importance-corrected baselines. We show that details of the experimental harness can reverse the observed ranking of methods, and we introduce PTH (Probe The Harness), a set of checks that makes the harness visible. Our case is a comparison between SAN, a behaviour-free method, and truncated importance sampling (TIS) on verl and in a single-GPU trainer, in which SAN first finished ahead in both stacks. Four details of the harness changed this comparison: the PPO ratio was taken against the learner's own recomputed probabilities, the data seed did not reach the TIS arm, the replay queue reused its first batch for 33 updates, and two loss normalisers differed from their description. In each case the logged quantity looked consistent with a working setup, while the quantity that defines the comparison went unchecked. With the harness checked, TIS matches SAN on verl, and in the trainer TIS learns steadily while SAN keeps a margin. We contribute the signature of each detail and its effect on the comparison, reference results for TIS and uncorrected GRPO under sampler lag, and the PTH checklist. 

---
# To Explore The Strange New World Beyond Data Distribution: System Behavior, Causality Tax, and Non-causal Base Model 

**Authors**: Xianzhi Zeng, Jiangneng Li, Gao Cong  

**Link**: [PDF](https://arxiv.org/pdf/2610.02839)  

**Abstract**: We show that the causality of language models (LMs) may not be necessary nor optimal. This is the case when system behavior (denoted as $S$) is incorporated as a first-principle Bayesian feature. Here, $S$ refers to extra dominant factors beyond the data space, and they involve coupled effects. Despite being the de facto foundation of modern architecture, recent studies indicate persistent mismatches and contradictions with causality. These issues largely stem from system behavior rather than the data distribution. We therefore propose the SBD framework, which incorporates $S$ as an irreducible component of the evidence lower bound (ELBO). SBD theoretically reveals a counter-intuitive Causality Tax phenomenon, where causality emerges as a suboptimal approximation with an additional structural error, due to the obliviousness to $S$. To address the challenge of latent variable analysis, we validate the SBD-predicted impact of $S$ via implicit measurements, theoretical-bound-guided controls, and Neural Tangent Kernel (NTK) evaluations. In particular, we construct Green Shell (GSH) to show the possibility of reducing Causality Tax. GSH is a non-causal variational family, and it replaces the sequential dependency chain of $S$ components with a divide-and-conquer partition. NTK spectra in the lazy-training regime confirm that GSH always achieves significantly tighter error bounds than causality, with $7dB+$ improvement in signal-to-noise ratio. In the relatively later stage of lazy-training, GSH further leads to superior generalization (up to $20\%$ richer multi-scale fitting capabilities). Taken together, SBD establishes system behavior as a complementary theoretical abstraction besides causality and distribution fitting, opening new research avenues such as designing and optimizing LM base models. 

---
# FSPO: Policy-Consistent Risk and Pareto-Feasible Control for Budgeted LLM RL Post-Training 

**Authors**: Miaobo Hu, Shuhao Hu, Xiaobo Guo, Xin Wang, Bokun Wang, Daren Zha, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.02828)  

**Abstract**: Adaptive LLM reinforcement-learning post-training changes multiple training actuators online, including rollout temperature, group size, clipping, KL regularization, verifier allocation, and update budget. Three coupled issues remain unresolved. A future-risk model trained from behavior trajectories need not estimate the risk induced by the controller that will be deployed; a score calibrated on logged state-action pairs can become miscalibrated after selective action choice; and independent per-resource minimum costs do not in general certify a feasible multi-resource continuation. We introduce FSPO, a feedback-state controller for budgeted LLM RL post-training that addresses these issues jointly. FSPO learns a policy-consistent risk-to-go model whose Bellman target follows the same frozen controller used for future decisions, together with a long-horizon utility model. Decision-conditioned trajectory calibration (DCTC) calibrates risk on cross-fitted trajectories generated by actions selected by provisional controllers. A Pareto resource continuation certificate (PRCC) admits an action only when a non-dominated cumulative reservation remains feasible over the residual horizon. Under a matched GRPO resource envelope, FSPO reaches 66.11% held-out and 59.43% OOD accuracy, compared with 64.47% and 57.03% for PB2, the strongest evaluated adaptive baseline. Three paired training seeds give gains of +2.42 and +3.19 percentage points over the contextual bandit on held-out and OOD evaluation. Under high behavior-deployment mismatch, policy-consistent risk lowers selected-decision ECE from 0.108 to 0.053; DCTC lowers it from 0.039 to 0.022 at matched acceptance; PRCC removes false-feasible admissions on an 18-action catalog ($0.197\rightarrow0.000$); and enabling all three components reduces trajectory failure from 0.181 to 0.083 in a factorial ablation. 

---
# RMCW: A Deletion-Robust Watermark Based on Reed--Muller Codes for Language Models 

**Authors**: Yi Wang, Baicheng Chen, Yu Wang, Jian Zhao, Yilei Chen, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2610.02817)  

**Abstract**: Large Language Model (LLM) watermarking provides a lightweight mechanism for identifying text generated by a specific model, but its robustness remains fragile under post-processing attacks. Deletion attacks are particularly challenging because they shift token positions and break the alignment between observed tokens and their original watermark positions. We propose Reed--Muller Code Watermarking (RMCW), an LLM watermarking method based on Reed--Muller codes. In contrast to global codeword recovery, RMCW searches for surviving local algebraic structure, leveraging the Reed--Solomon consistency induced by affine-line restrictions of Reed--Muller codewords. During generation, RMCW injects a Reed--Muller structure into the sequence via a secret-keyed vocabulary partition. During detection, it maps the given text to keyed vocabulary bins and tests local subsequences for low-degree Reed--Solomon consistency using Berlekamp--Welch tests. Experiments on C4 and ELI5 datasets with OPT-1.3B and Llama-3.1-8B-Instruct show that RMCW preserves strong clean-text detectability and outperforms or matches the baseline methods under several deletion and rewriting attacks. Our code is available at this https URL. 

---
# ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing 

**Authors**: Miaobo Hu, Shuhao Hu, Xiaobo Guo, Xin Wang, Bokun Wang, Tianshu Fu, Daren Zha, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.02808)  

**Abstract**: Adaptive multi-verifier systems are commonly compared through endpoint quality-cost gaps, even when the verifier catalog, availability, accounting, information filtration, or scorer changes with the policy. We formulate verifier routing as a contract-conditioned identification problem. The contract records request support, verifier catalog, realized availability, resource accounting, online filtration, and post-trace scoring; a matched route contrast changes only the policy coordinate. ROUTEAUDIT adds three measurable objects to this contract. A contract lattice averages coordinate increments over every admissible bridge order and reports the resulting attribution together with its path sensitivity. A policy-independent response tape identifies paired sequential contrasts when adaptive policies reveal different observations. For incomplete matching, request-level bounds use whichever potential outcome remains observed and give a sharp finite-population interval. The protocol commits paid observations and ledger events before the oracle join and returns an attribution certificate for each comparison. On two held-out raw-tail caches, matched static SF+SA equals the cascade, assigning the apparent gains of 0.1797 and 0.1250 over full static to the verifier-set edge. On 1,319 held-out task requests, the learned and RLVR studies report quality 0.9522 and 0.9553 versus 0.9484 for matched static; the RLVR-static paired difference is +0.0068 with a request-paired interval $[0.0015,0.0122]$ and a training-seed-by-request hierarchical interval $[0.0006,0.0131]$. Controlled attribution recovery yields route mean absolute error 0.0011 and endpoint reconstruction error 0.0004. Factorial, bridge-order, and stochastic-provider studies evaluate the certificate interface; RLVR supplies a learned-policy stress test under the same identification contract. 

---
# OPD Before RL: Warm-Starting Rubric-Based RL with On-Policy Distillation 

**Authors**: Xinpeng Wang, Wei Shi, Yu-Chia Chen, Maria Zontak, Yun He, Richard Yuanzhe Pang  

**Link**: [PDF](https://arxiv.org/pdf/2610.02781)  

**Abstract**: Many useful language-model tasks cannot be evaluated by exact outcome verification. Rubric-based reinforcement learning (RL) addresses this issue by scoring open-ended responses against explicit criteria. However, because the reward is assigned after the complete response, the training signal does not directly identify which individual decisions contributed to the final score. We propose a two-stage training framework that uses rubrics first as privileged teacher context for dense token-level supervision, then as rewards for further RL. In the first stage, rubric-privileged on-policy distillation (RP-OPD), a student without access to the rubric matches a rubric-aware teacher's next-token distributions at student-generated prefixes. In the second stage, RL directly optimizes the rubric reward and improves beyond the observed distillation plateau. We evaluate the framework on health and science tasks using open-weight models. Across HealthBench, ResearchQA, and RubricHub Science, we compare post-training methods and vary the amount of SFT or RP-OPD training before RL, finding that our two-stage framework achieves the highest scores among the methods evaluated. RP-OPD + RL shows limited signs of reward hacking on RubricHub Science, whereas the SFT + RL baseline increasingly receives high rewards for claims of rubric compliance without providing the required content. These findings support using rubrics to guide on-policy distillation before applying rubric-based RL. 

---
# Learning from Evolving Errors: Adaptive Iterative Repair for On-Policy Distillation 

**Authors**: Rui Li, Liyang He, Zheng Zhang, Zhenya Huang, Linbo Zhu, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.02700)  

**Abstract**: On-policy self-distillation (OPSD) supplies dense token-level feedback on trajectories sampled from the student's own policy, a richer training signal than the outcome-level rewards of reinforcement learning. This feedback comes from a teacher conditioned on a full reference solution unavailable to the student. The reference solution specifies the target but not how to move from the student's current error toward it, creating a solution-conditioned shortcut risk. We introduce AIR-OPD, an adaptive iterative repair framework for on-policy distillation that provides error-to-repair supervision. Given a failed response, a guidance generator synthesizes repair guidance for the current error. The student samples an on-policy retry with this guidance. If the retry remains incorrect, the generator produces new repair guidance for the newly observed error. At each round, a fixed teacher receives the guidance as privileged context and supervises the student on an error-aligned region of its latest failed response. Outcome-aware stage weighting favors early repair stages and credits stages whose immediate retry passes verification. We train AIR-OPD on the DAPO-Math-17K dataset and evaluate on AIME24, AIME25, and HMMT25, alongside out-of-distribution tests on MMLU-Pro and GPQA. We examine two guidance sources, self-guidance from the current student policy and external guidance from a larger model. For both Qwen3-4B and Qwen3-8B, AIR-OPD attains the best mathematical-reasoning averages, improving over the strongest baseline by up to 3.6 points, while preserving base-model performance on the out-of-distribution benchmarks. 

---
# Large language models exhibit unreliable updating of clinical judgment as patient evidence evolves 

**Authors**: Min Zeng, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.02684)  

**Abstract**: Large language models (LLMs) are increasingly explored for clinical reasoning, but whether they appropriately revise judgments as patient evidence evolves remains unclear. We evaluated longitudinal belief updating using matched intensive-care trajectories from electronic health records. Across diverse LLMs, conditioning on a preceding judgment more often increased than reduced prediction error when estimates changed, replicated for a second endpoint. Controlled interventions revealed two failure modes. First, with preceding assessment fixed, models responded more strongly to worsening than matched improving respiratory evidence; this asymmetry persisted after headroom normalization at moderate and strong evidence levels. Second, with current evidence fixed, increasing prior risk from 10% to 90% shifted estimates by 26.2 percentage points, demonstrating causal influence of prior model beliefs. Prompting did not restore reliable updating. Evidence-Validated Longitudinal Update (EVLU) identified fewer, more reliable revisions, revealing a reliability-coverage trade-off. These findings establish longitudinal belief updating as a distinct dimension of LLM reliability. 

---
# Asterism: Exploring and Synthesizing Scattered Observations into Literature-Grounded Hypotheses and Theories 

**Authors**: Joseph Chee Chang, Michael D'Arcy, Amy X. Zhang, Pao Siangliulue, Sangho Suh, Aakanksha Naik, Jena D. Hwang, Javier Ramos Benitez, Stella Wroblewski, Matt Latzke, Michael Cuoco, Ruben Lozano-Aguilera, Kris Ganjam, Joel Chan, Doug Downey, Peter Jansen, Kyle J. Travaglini, Daniel S. Weld  

**Link**: [PDF](https://arxiv.org/pdf/2610.02673)  

**Abstract**: A theory draws many independent observations into one framework with novel hypotheses. A researcher building such a theory must synthesize observations scattered across many papers, each describing related concepts but often in different terms. Which concepts matter most also depends on their preferences and research questions. Recent approaches scale theory synthesis with LLMs, but automate away choices and intuitions from researchers. We present Asterism, which extracts observations from hundreds of papers as concept-relation triples, with concepts unified in a hierarchical ontology. Researchers curate an evidence graph using the ontology and aggregate observations at different levels of granularity to focus theory formation on specific phenomena of interest. In a field deployment (n=10), researchers worked from observations to theories, and kept concepts and hypotheses fitting their preferences. In two case studies, teams of immunology and agriculture researchers discovered mechanisms outside their standard analyses and constructed hypotheses worth follow-up experiments. 

---
# LEAP: Learning Efficient Action Proposals For LLM Agents 

**Authors**: Zhen Xu, Qizheng Zhang, Gerry Wan, Shang Zhu, Ce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.02670)  

**Abstract**: LLM agents are known to be slow in rollouts. An agent completes a task one step at a time. At each step, it reasons and then chooses an action to execute. The next step and action cannot start until the previous one has finished. Speculative decoding accelerates the rollouts at the reason phase by drafting and verifying the inference tokens. Recent works have also started to apply similar ideas at the action phase. These works use off-the-shelf models, usually large, to draft action proposals for target model to verify. Large drafters match the target more often but take longer to propose, while small off-the-shelf models are fast but rarely make the same decision as the target. We ask a more general question: what determines the end-to-end speedup of action speculation? To answer it, we develop a latency framework for the speculative round. The framework compares what a round gains with what it costs. The gain depends on how well the drafter predicts the target and on how many steps the task can take before it ends. The cost comes from drafting, from waiting for target verification and from executing tools. Guided by the framework, we introduce LEAP (Learning Efficient Action Proposals) which keeps the drafter small and makes it accurate by training it on the target actions sequences. With a small 0.6B model, LEAP agrees with the target on most decisions and makes agents up to 60% faster in end-to-end wall clock time, with no systematic change in task success. Across various datasets, target models and draft models, the framework accounts for most of the measured speedups. We also show the draft model can be online trained with no prior trace collection and match the performance of offline training, making LEAP practical to deploy in the real world. 

---
# VERSE: Verified Self-Evolving Optimizer for Agent Harnesses 

**Authors**: Zekai Wang, Yingqiang Ge, Zekun Wang, Hai Wang, Yuhui Xu, Joshua Frandsen, Shancong Fu, Ashia C. Wilson, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2610.02616)  

**Abstract**: Harness evolution improves an LLM agent's prompts, tools, and workflow, while the optimizer's own tools and procedures often remain fixed. We study whether an optimizer can improve another agent more effectively by also improving how it diagnoses failures, develops edits, and tests their effects. Two observations guide our design. In a controlled study, optimizer self-evolution fails to improve performance without execution-based verification, but achieves the best result of that study when verification is available. Across five executors, self-evolving optimizers build their own tools for failure analysis, verification, training audits, and workflow control. Motivated by these findings, we introduce VERSE, a Verified Self-Evolving optimizer for agent harnesses. VERSE lets the optimizer test draft edits, replay failures, and perturb suspected steps before submission, while tracking fixes and regressions across rounds. Using this feedback, the optimizer revises both the executor harness and its own prompts, skills, tools, hooks, and notes, while the weights of the optimizer and executor models stay fixed. Under a shared protocol with disjoint training, validation, and test tasks, VERSE improves all four evaluated harness optimizers on held-out SWE-rebench tasks and newer out-of-distribution tasks in five languages. Its best validation-selected harness reaches 42.3% and 37.7% accuracy, respectively, against 39.2% and 29.3% for the strongest baselines. Code is available at this https URL. 

---
# How Causality Bridges the Semantic Gap 

**Authors**: Shuhao Zhang, Xuran Zhou, Han Guo, Pengtao Xie, Yujia Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.02594)  

**Abstract**: Numerical measurements capture how a system behaves, but often leave the meanings of its variables unspecified. Some variables are measured but never labeled, and others are never measured at all. Existing methods assign semantics to such variables by consulting general human knowledge, but this inherits its biases where that knowledge exists and offers nothing where it does not. We bridge this gap between measurements and their meanings with causal structure instead, reading a variable's semantics from how it acts on other variables. We formalize this as structure-constrained semantic alignment, in which the embedding of each unnamed variable is solved under the dependence relations implied by the causal graph, with the embeddings of a few known names as anchors. Accordingly, we build CausalBridge, a framework that discovers the causal graph from the measurements, latent variables included, solves for the embeddings under those relations, and expresses them as names through a language model. The causal structure reflects the mechanism that generated the measurements and is recovered from the measurements alone, which may make it the one source of information free of bias from human knowledge. We evaluate CausalBridge on five questionnaires and three robotics scenarios, with 20 to 90% of the variable names masked. It recovers the semantics of observed and latent variables more accurately than existing methods that rely on association, and its lead widens as less of the system is documented. The graph it discovers names variables as accurately as the documented one, and a new system is named in minutes and at a fraction of the cost of sampling methods. Once the semantic gap is bridged faithfully, machines can understand the world and take actions causally. 

---
# Right Order, Wrong Scale: Auditing LLM Judges for Occupational AI Measurement 

**Authors**: Harry Lyu, Neil Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2610.02492)  

**Abstract**: LLM judges are increasingly used to assess whether AI outputs meet workplace requirements, but agreement on response rankings does not establish agreement on acceptance rates or occupational aggregates. We introduce O*NET-BENCH, an audit suite derived from an existing survey of 45,796 worker ratings, and evaluate 33 pre-existing judge configurations across six model families on 4,501 test ratings. Twenty-five configurations achieve tie-aware pair accuracy of at least 0.60, although a train-fitted response-only TF-IDF baseline nearly matches the strongest judge. Despite this ordering agreement, judges estimate that 3.0%-97.9% of responses are acceptable, compared with 61.1% for occupation-matched workers. In one fine-tuned lineage, changing from pointwise scoring to a bundled few-shot/listwise protocol improves response ordering while reducing agreement with worker means at the task and occupation levels; this reversal replicates on a task- and worker-disjoint validation split under prespecified criteria. Cross-validated calibration largely removes mean bias, but calibrated scores explain at most 8.5% of individual worker-rating variance. Prediction-assisted estimation yields at most small precision gains at the studied label budgets. These results show that ranking agreement alone is insufficient for occupational measurement. Judges should be validated against the acceptance rates and aggregates their scores will be used to estimate. 

---
# Capability Scaling-Down Laws for LLM Compression 

**Authors**: Xueqi Cheng, Liang Wu, Kelly Wan, Liangjie Hong, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2610.02462)  

**Abstract**: LLM compression reduces inference costs and memory requirements, but selecting a method and configuration remains largely empirical because comparable resource reductions can produce different capability losses. We systematically investigate capability scaling-down laws for LLM compression across pruning, quantization, and distillation. Our framework measures capability loss in mathematics, code generation, and question answering, and relates these measurements to model size, training stage, compression settings, data availability, and training exposure. We develop simple predictive relations and evaluate their accuracy, measurement efficiency, and generalization to unseen configurations and model states. Sharing the density response across pruning levels halves the configuration measurements needed to fit a pruning predictor: on new Pythia states, on pre-registered OLMo-2 test states and under Wanda pruning, the compact relation matches a regression fitted with all measurements on math and code to within 0.020 nats per token, with coefficients refitted for each setting. Controlled distillation experiments show that the cost of heavy data reuse recurs across question-answering distributions, while the net benefit depends on the evaluation distribution. We further evaluate the decision value of these predictions by comparing numerical selection with configuration medians and fixed method priorities. Independent evaluations across two model families show that selection captures most of the available cross-method benefit for question answering within the tested candidate sets, where a fixed method priority attains the same regret, with smaller opportunities for mathematics and code. These results clarify the predictive scope of capability scaling-down laws and their use in compression method selection. Our code is publicly available at: this https URL. 

---
# Are you Synthesizing or Recalling? Evaluating LLMs on Algorithmic Code Retrieval 

**Authors**: Nickil Maveli, Antonio Vergari, Shay B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2610.02438)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance in code generation, where success depends on both recalling relevant algorithmic knowledge and reasoning about how to apply it. However, existing LLM pipelines are opaque, with no explicit separation between these two components. We argue that for well-known algorithms whose canonical implementations are widely accessible in pretraining corpora, code generation is better measured as \textit{parametric code retrieval}: reproducing a named algorithm from internalised knowledge rather than synthesizing a novel one. We introduce AlgoREval, a benchmark of 599 problems spanning classical 77 algorithms across 14 domains, 7 programming languages, and 4 graph-input representations to evaluate this capability in isolation, and assess 15 models (7B--34B parameters) in a zero-shot setting. We find substantial variation in retrieval accuracy across languages and input representations, even for widely documented algorithms and show that prompt augmentation with retrieved code snippets or structured algorithmic hints improve accuracy on complex algorithms, while SFT achieves broader language gains and GRPO achieves larger per-language gains on specific languages. Together, our results establish parametric code retrieval as a distinct, measurable capability and caution against deploying AI-generated algorithmic code without systematic validation.\footnote{Code and dataset are available at this https URL 

---
# Evaluating and Improving the Robustness of Large Language Models to Input Sequence Variations 

**Authors**: Narek Maloyan  

**Link**: [PDF](https://arxiv.org/pdf/2610.02432)  

**Abstract**: Large language models (LLMs) in production systems face prompt injections, trojans (backdoors), and manipulation of automatic quality metrics. This thesis develops models, methods, and algorithms for evaluating and improving LLM robustness to adversarial input sequence variations. We propose R_stab(f), a generative robustness metric based on the Jensen-Shannon divergence between per-step output distributions under small input perturbations. For localized attacks we prove V(h) <= 1 - R_class(h), where R_class(h) is the probability that a decision operator h keeps its decision under small perturbations. For non-localized attacks we propose a calibrated empirical model. For LLM-as-a-Judge systems we develop ASA, an adaptive evolutionary black-box attack that reaches an attack success rate (ASR) of up to 73.8%, with transfer between open models up to 62.6%. On Trojan Detection Challenge 2023 data (Pythia-1.4B), surrogate triggers reach REASR ~0.99 while recall of the true triggers is ~0.17 against a baseline of ~0.14. On SaTML CTF 2024 we systematize four classes of bypasses of multi-layer defenses, which reduce the ASR from 90% to 15-25%. Committees of 5-7 heterogeneous models reduce the ASR for Gemma-3-4B by 47-55 percentage points, to 19.3% with 7 models. For agentic systems based on the Model Context Protocol (MCP), we propose AttestMCP, which attests tool calls with HMAC-protected packets at under 0.1 ms per call, and the Commit Boundary isolation pattern. On the MCPBench benchmark of 847 scenarios they reduce the average ASR from 53.7% to 12.4%. The methods are implemented in the JudgeGuard and TrojanArmor software suites and the MCPSec module. 

---
# Trained Agentic Context Management 

**Authors**: Bryce Sandlund  

**Link**: [PDF](https://arxiv.org/pdf/2610.02404)  

**Abstract**: We study long context language models. Instead of training long context natively, or designing a long context harness, we train a model over the simplest possible harness: a tool to call itself with any specified prompt and a tool to read tokens in a range from the input context. We finetune Qwen3.6-35B-A3B on a diverse synthetic dataset using this harness. With only 8,000 tokens of context, our small model is as strong as GPT-5.4 with 1M tokens of context on the OOLONG-synth benchmark when document length exceeds 40K tokens. 

---
# Hesitation Has a Geometry: Entropy-Trained Hyperbolic Probes for Sparse Activation Steering 

**Authors**: Zeyong Zhang, Tung Sum Thomas Kwok, Tengfei Ma, Mengjia Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.02391)  

**Abstract**: When a large language model solves a mathematical problem, its reasoning is largely hierarchical, and the solution often branches at a few tokens where the next-token entropy is high. Such tree-like structure embeds in hyperbolic space with far lower distortion than in Euclidean space. Activation steering, however, usually edits the hidden states of a pretrained model by adding one fixed Euclidean vector at every token, even though most tokens of a solution are already determined by the context. We propose Hyperbolic Entropy Steering (HEST), which embeds the hidden states in the Poincaré ball with a lightweight probe whose only label is the model's own next-token entropy. Where this entropy exceeds a threshold, HEST moves the embedded state along the geodesic of steepest descent of a readout of the probe and maps the change back to the hidden state. For the Busemann readout of a learned ideal point, we prove that a step of fixed length lowers it by the same amount at every state. On three instruction-tuned models from the Qwen2.5-Math and Llama-3.1 families, HEST with the Busemann readout improves greedy accuracy on MATH-500 and GSM8K in five of six settings, by up to 1.8 points, whereas a contrastive steering vector added at every token lowers accuracy. With a Euclidean probe trained in the same way, this gain disappears on Qwen2.5-Math-1.5B-Instruct. The gains are largest on problems where the model hesitates often, and accuracy on the remaining problems is almost unchanged. 

---
# Social bot detection in the age of ChatGPT: Challenges and opportunities 

**Authors**: Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2610.02386)  

**Abstract**: We present a comprehensive overview of the challenges and opportunities in social bot detection in the context of the rise of sophisticated AI-based chatbots. By examining the state of the art in social bot detection techniques and the more salient real-world application to date, we identify gaps and emerging trends in the field, with a focus on addressing the unique challenges posed by AI-generated conversations and behaviors. We suggest potentially promising opportunities and research directions in social bot detection, including (i) the use of generative agents for synthetic data generation, testing and evaluation; (ii) the need for multimodal and cross-platform detection based on network and behavioral signatures of coordination and influence; (iii) the opportunity to extend bot detection to non-English and low-resource language settings; and, (iv) the room for development of collaborative, federated learning detection models that can help facilitate cooperation between different organizations and platforms while preserving user privacy. 

---
# SEDIMA: Cross-Run Hierarchical Insight Memory for Evolutionary Search Agents 

**Authors**: Amirhossein Abaskohi, Mahdi Mostajabdaveh, Zirui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.02361)  

**Abstract**: Large language model (LLM)-driven evolutionary search is a powerful paradigm for automated program and algorithm discovery, yet existing systems are largely memoryless: each run explores from scratch, so agents repeatedly rediscover the same improvements and re-encounter the same dead ends. We introduce SEDIMA, a persistent hierarchical insight memory for evolutionary search agents. SEDIMA distills raw traces into natural-language insights, clusters them by semantic similarity using attention-weighted centroids, and retrieves relevant guidance to condition future mutations, accumulating transferable knowledge across runs and problems rather than within a single trajectory. As a drop-in module that leaves the search operators unmodified, SEDIMA improves average final performance by 5.5% on AlgoTune and 6.6% on ALE-Bench LITE under a fixed budget of 100 evaluated candidates. Under OpenEvolve, SEDIMA requires 32.3% fewer iterations on average to reach baseline-best performance across the five evaluated backbones. 

---
# Lexicographic Multi-Objective On-Policy Distillation 

**Authors**: Doseok Jang, Jon Ander Campos, Youran Qi  

**Link**: [PDF](https://arxiv.org/pdf/2610.02359)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) usually optimizes answer correctness, yet useful language-model behavior also requires high-quality reasoning and concise responses. Existing multi-reward post-training methods typically scalarize rewards or combine specialists without explicitly protecting a reward priority order. This is problematic when trade-offs are asymmetric: conciseness, for example, should not improve at the cost of correctness. We introduce Lexicographic Multi-Objective On-Policy Distillation (LMOPD), a multi-teacher method for integrating reward-specialized policies under explicit priorities. For each student rollout, LMOPD selects the specialist for the first objective whose gate detects a deficiency, then locally projects its centered log-policy correction to remove components that oppose higher-priority specialists. We evaluate 30B-A3B mixture-of-experts transformer models in two- and four-expert settings on three math benchmarks, measuring retained specialist gains. With two experts, LMOPD's point estimates fully retain the accuracy and reasoning-quality gains while acquiring $46.9\%$ of the conciseness gain. With four experts, it retains $\approx90\%$ of both the accuracy gain and reasoning-correctness gain, compared to only $\approx57\%$ by the next best evaluated baseline. Matched four-expertablations show that lexicographic routing outperforms random routing and that projection further strengthens both top-priority capabilities. Across both scales, LMOPD preserves the highest-priority capabilities more effectively than the existing baselines we evaluate, demonstrating the value of explicit priorities for specialist integration. 

---
# Does Every User Need a Private LoRA? Decoupling Personalization from Per-User Adaptation 

**Authors**: Songyuan Sui, Srikanth Malla, Chiho Choi, Joon Hee Choi  

**Link**: [PDF](https://arxiv.org/pdf/2610.02353)  

**Abstract**: Personalized large language models often require a complete adaptation state for each user. However, this paradigm scales poorly as the user population grows. We revisit this design through the lens of personalization capacity allocation: how much adaptation capacity can be shared across users, how the shared capacity should be composed, and how much must remain user-specific. We answer them through three complementary empirical analyses. We find that independent user adapters contain substantial cross-user reusable structure, that the utility of reusable directions reflects both user relevance and variation across queries, and that user histories provide transferable signals for compact individual correction. Motivated by these findings, we propose LINEUP. It learns a bank of reusable low-rank personalization factors, composes them through user-conditioned recall and query-dependent calibration, and restricts target-user adaptation to a tiny user code over a shared correction space. This design decouples expressive personalization capacity from per-user trainable state. Each target user optimizes only eight scalars, while all shared components remain fixed. By comparison, the evaluated private-LoRA configuration uses 4.19 million per-user parameters. Our theoretical analysis gives a finite-step, finite-history risk bound and sufficient conditions for user-code refinement to improve on history initialization. Across six tasks spanning personalized classification, prediction, and generation, LINEUP leads on all 12 metrics, each averaged over three independent runs (e.g., reducing LaMP-3 RMSE by 11.4% relative to the strongest baseline). It maintains advantages under limited history. These results show that rich personalization can be supported primarily by reusable, conditionally composed shared capacity, while independent user adaptation remains confined to a tiny correction state. 

---
# Fast Models, Slow Evidence: A Paired and Self-Audited Evaluation of System-1 Decision Models for LLM Agent Harnesses 

**Authors**: Jiawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.02267)  

**Abstract**: Agent harnesses make many small, typed decisions per task: which model to call, which tool to use, whether retrieved text is relevant, whether an input carries an injection. System-1 decision models answer such questions in a single forward pass with class probabilities, promising large cost and latency savings over LLM calls. We present a paired evaluation of an open-weight (Laya) and a hosted (Jev) System-1 model on 11 agent decision points built from 18 public sources: 7,283 base cases plus 6,640 robustness variants, with byte-identical inputs, paired tests, and cross-hardware and cross-day reproducibility checks. Jev is significantly more accurate on 9 of 11 decision points (+10.8 to +46.0 pp). Neither model beats chance on zero-shot model routing, and they tie on RAG relevance gating. Laya changes 30% of its answers when the option order is reversed and degrades sharply with many or similar candidates (31% at 50 nearest-neighbour tools, vs. 98% for Jev on items with a unique correct tool). We also audit our own pipeline. Three analysis errors and one design confound distorted headline deployment claims: an omitted pre-screen cost (reported 23.9% saving, actual 4.3%), gate accuracy reported as end-to-end quality (58% vs. 98%), in-sample thresholds (5% target, up to 17% held-out misses), and a "channel effect" on injection false positives that vanishes with channel-native content. Two other suspected confounds did not change the conclusions. All cases, raw outputs and analysis code are available at this https URL. 

---
# Budgeted Cache Repair for Cross-Context KV-Cache Reuse 

**Authors**: Haeyong Kang, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2610.02233)  

**Abstract**: Cross-context KV-cache reuse predicts a shared segment's keys and values under a new prefix instead of recomputing them, and has been reported to do so without quality loss. We find otherwise, and identify two problems. (1) A hidden cost: on MMLU and GSM8K, reuse costs substantial accuracy. (2) A decision at the wrong unit: no rule for deciding whether to reuse a cache removes that cost. What does help is choosing which parts of the cache to recompute, and the value of choosing well falls as the unit of choice grows: informed selection removes 49.5% of the cache error beyond chance at single rows (one token's keys and values), 10.6% at 64-token chunks, and nothing at the level of whole calls. Budgeted Cache Repair (BCR) acts at the unit where selection still pays. It drafts two tokens from the assembled cache, ranks cache rows by the attention those tokens pay them, and recomputes a fixed budget of rows exactly, in one of three layouts. The cost is paid rather than predicted away, and the draft that fails as a gate succeeds as a selector. BCR restores GSM8K to dense-prefill accuracy while still serving most calls from cache, and its best layout outperforms every reuse baseline's mean in the reference grid. The draft also beats a coin-flip selector at the same budget - a control prior evaluations lack. 

---
# A Manifold-Aware Topic Modeling Approach via Rank-Based Prototypes 

**Authors**: Thiago César Castilho Almeida, Daniel Carlos Guimarães Pedronette  

**Link**: [PDF](https://arxiv.org/pdf/2609.29630)  

**Abstract**: Recent topic models leverage pretrained embeddings, but neural architectures produce latent representations without grounding in specific texts, and clustering-based pipelines assign representative documents only post hoc, relying on absolute distances distorted by hubness and anisotropy in high-dimensional spaces. We introduce MARETopic, a training-free framework that casts topic discovery as rank-based prototype selection. After projecting embeddings onto a low-dimensional manifold, MARETopic builds ranked lists encoding ordinal neighborhood structure. A greedy algorithm selects exactly K exemplar documents, real corpus texts, whose neighborhoods cover the corpus. Two variants share this criterion. MARETopic$_\text{Corr}$ scores candidates with a query performance predictor and a rank correlation measure, leading Purity and NMI on the two benchmarks with the most categories, ahead of both neural and clustering-based topic models. MARETopic$_\text{Diff}$ scores them with a rank-based diffusion matrix, needs neither measure, and runs 1.7 to 1.9 times faster. Without a single gradient update, MARETopic leads topic coherence on two of three datasets. A novel inter-topic Maximal Marginal Relevance step raises vocabulary diversity at little cost in coherence. Our code is available at this https URL. 

---
