# Private Information Retrieval for Large-Scale DNA-Based Data Storage 

**Authors**: Gökberk Erdoğan, Daniella Bar-Lev, Rawad Bitar, Antonia Wachter-Zeh, Zohar Yakhini  

**Link**: [PDF](https://arxiv.org/pdf/2606.14557)  

**Abstract**: We investigate Private Information Retrieval (PIR) in the context of synthetic DNA-based data storage. While PIR is a well-studied primitive for digital databases, extending it to DNA-based databases presents unique challenges arising from biochemical query mechanisms and their complexity. We propose two approaches for adapting two-server PIR protocols to DNA-based storage, balancing privacy, efficiency, and feasibility. These approaches illustrate how information-theoretic privacy trade-offs manifest in DNA-based storage systems. 

---
# Verifiable User Simulation for Search and Recommendation Systems 

**Authors**: Chenglong Ma, Xinye Wanyan, Danula Hettiachchi, Ziqi Xu, Yongli Ren, Jeffrey Chan  

**Link**: [PDF](https://arxiv.org/pdf/2606.14474)  

**Abstract**: Large-language-model (LLM) based user simulation is increasingly adopted for evaluating search engines, recommender systems, and retrieval-augmented generation pipelines, yet most simulators remain opaque: it is difficult to determine why a simulated user made a particular choice or whether that choice is consistent with the intended user profile. Compounding this, recent research shows that LLMs can produce biased or discriminatory responses depending on user background characteristics such as language, education level, and cultural context, raising concerns about the equitable treatment of minority and disadvantaged groups. This half-day, in-person tutorial introduces a proposed design-and-audit framework that treats a user simulator as a verifiable engineering artefact composed of seven auditable components - structured Persona, task-aware Contract, matched human-vs-agent Execution, auditable Trace, persona-aligned Verification, structured Feedback, and a Refinement loop that updates personas and contracts. Through two hands-on mini-labs on recommendation-list evaluation and search-query formulation, participants will inspect simulator behaviour end-to-end, distinguish diagnostic discrepancy analysis from statistical validation, and apply checks for fidelity, credibility, and demographic bias. The tutorial targets information retrieval and recommender systems researchers and practitioners interested in user behaviour simulation and responsible AI. 

---
# ScoreGate: Adaptive Chunk Selection for Retrieval-Augmented Generation via Dual-Score Statistical Fusion 

**Authors**: Karamvir Singh, Arvind Jain  

**Link**: [PDF](https://arxiv.org/pdf/2606.14269)  

**Abstract**: Fixed-cardinality retrieval injects a constant top-K chunks into the generator regardless of query complexity, causing over-retrieval for narrow queries and under-retrieval for compositional ones. We describe ScoreGate, a lightweight score-space decision mechanism that controls retrieval cardinality at inference time using two scores already produced by the standard pipeline: bi-encoder similarity s_i and cross-encoder reranker score r_i, with no additional model inference calls required. Its core insight is that cross-encoder affirmation can rescue semantically relevant chunks that bi-encoder retrieval ranks poorly due to vocabulary mismatch -- a failure mode unaddressed by fixed-K or single-score thresholding. On MS MARCO (200 dev queries), ScoreGate achieves MRR@10 = 0.401 with 35% fewer retained chunks than Standard Top-K. On an internal benchmark (n=300, Fleiss' kappa=0.87), ScoreGate observed zero false positives (95% CI [96.4%, 100%]) at 97.77-99.34% recall, with 34.8% fewer tokens per query and only 31ms added latency. Results on both MS MARCO and real-world production traffic suggest that adaptive retrieval cardinality can improve retrieval efficiency without degrading retrieval quality. 

---
# ChronoID: Infusing Explicit Temporal Signals into Semantic IDs for Generative Recommendation 

**Authors**: Dongdong Nian, Dongqi Fu, Chenliang Xu, Yinglong Xia, Hong Li, Hong Yan, Jian Kang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14260)  

**Abstract**: Semantic IDs are crucial in generative recommendation, but with a fundamental limitation: temporal information is not well incorporated into semantic IDs. Instead, time influences recommendation only implicitly (e.g., through session construction heuristics, preference alignment, or sequence order), while existing semantic ID learning remains entirely time-agnostic. This design conflates interactions occurring under distinct temporal contexts into identical semantic representations, implicitly assuming that item semantics and user intent are temporally stationary. Such an assumption is misaligned with real-world recommendation scenarios, where evolving interaction rhythms play a central role. In this work, we investigate where and how the explicit time should be incorporated into semantic ID for generative recommendation. First, we systematically characterize the design space along three orthogonal dimensions of temporal signals and present a unified framework, ChronoID, for time-aware semantic ID learning. Then, by contributing a new time-explicit generation recommendation benchmark, ChronoID answers the questions: what is the effective way of infusing time, how to design the architecture, and where does the gain come from. 

---
# CoRe: A Continuously Reward-Finetuned LLM Query Rewriter for Multi-Stage Context-Aware Relevance in Web-Scale Video Search 

**Authors**: Yilin Wen, Rong Yang, Xiaojia Chang, Hong Sun, Gefu Tang, Chunhui Liu, Jeffrey Chen, Zeyu Ma, Lisong Qiu, Xiaochuan Fan, Congjia Yu, Quan Zhou, Yuheng Chen, Zian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14127)  

**Abstract**: LLM-based query rewriters in production face a tension: the training reward must reflect how the rewrite is consumed by the production ranker, yet the training procedure must be cheap enough to support continuous redeployment as data drifts. We present CoRe (Context Relevance), such a system, redeployed weekly for over five months in a major short-video search engine. Our reward uses the deployed multimodal relevance model as its source and a multiplicative ratio form mirroring the production fusion algebra, closing the simulation-production gap that offline reward proxies leave open. A semi-online Mixed Preference Optimization loop makes this reward affordable at multi-million-instance weekly scale: a DPO-style pairwise objective restricts the gradient pass to a small top-k/bottom-k subset of sampled trajectories, and a phase structure reduces trainer/inference-server parameter syncs from per-step to per-phase. An automated promotion gate over reward-like and stability metrics detected and recovered from a real reward-hacking incident in production. Rewriter output is consumed as parallel relevance signals at recall, rawrank, and finerank without displacing the original signals, bounding rewriter-failure blast radius. Online A/B from two sequential production launches, first deploying the rewriter at finerank, then extending consumption to recall and rawrank, delivers statistically significant reductions in change-query rate on rewrite-impacted queries, with all headline relevance and engagement metrics moving in the expected direction. 

---
# Knowledge Graph Enhanced Memory-Augmented Retrieval for Long Context Modeling 

**Authors**: Ghadir Alselwi, Basem Suleiman, Hao Xue, Shoaib Jameel, Hakim Hacid, Flora D. Salim, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2606.14047)  

**Abstract**: Long-context language modeling requires not only extending context windows but maintaining coherent understanding of entity states and relationships across thousands of tokens -- a challenge that semantic similarity alone cannot address. KGERMAR addresses this by constructing dynamic, context-specific knowledge graphs from input text during inference, enabling domain-adaptive retrieval that leverages both semantic similarity and explicit entity relationships. The framework performs real-time entity and relation extraction to build contextual knowledge graphs, then integrates graph-structural embeddings with textual semantics through a multi-component memory architecture. Three memory banks -- contextual, semantic, and structural -- are maintained with retrieval signals fused via learned weights to capture both surface-level semantics and deeper relational patterns. Evaluated on SlimPajama (84.7K training examples), WikiText-103 (4,358 examples), PG-19 (100 examples), and Proof-pile (46.3K examples), KGERMAR achieves up to 8.5\% lower perplexity and 2--2.5x better memory efficiency than memory-augmented baselines across context lengths from 1K to 32K tokens, with superior in-context learning performance across five NLU tasks. The dynamic knowledge graph construction approach advances memory-augmented language modeling by enabling domain-specific knowledge representation that adapts to input contexts rather than relying on fixed knowledge bases. 

---
# When Recommendation Denoising Meets Popularity Bias: Understanding and Mitigating Their Interaction 

**Authors**: Guohang Zeng, Jie Lu, Guangquan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.14046)  

**Abstract**: Implicit feedback is the dominant data source for recommender systems, but behavioral logs are often contaminated by false-positive interactions caused by mis-clicks, biased exposure, and interface effects. Denoising recommendation methods improve robustness by down-weighting or filtering interactions suspected to be noisy, often relying on the small-loss heuristic. We revisit this heuristic through the lens of popularity bias. Tail-item positives can be harder to fit because they are sparsely observed, and thus may receive larger losses even when they reflect genuine user preference. Under such popularity-dependent loss patterns, monotone loss-based reweighting can suppress clean-but-hard tail signals and increase the head-tail imbalance in effective supervision.
We formalize this interaction through the effective head-tail signal ratio induced by denoising weights and derive a conditional reallocation result: when the loss distribution of tail positives is right-shifted relative to that of head positives, small-loss reweighting increases the effective head-tail signal ratio compared with ERM. Motivated by this analysis, we propose Popularity-Aware Denoising (PAD), a lightweight plug-in framework that modulates denoising strength by item popularity. PAD applies stronger denoising to highly exposed items while being more conservative on tail items, preserving more clean-but-hard long-tail signals. Experiments on three datasets and three backbones show that PAD generally improves over representative denoising baselines and provides favorable accuracy-diversity tradeoffs, especially on MF-style recommenders. 

---
# ADORE: Iterative Query Expansion with Retrieval-Grounded Relevance Feedback 

**Authors**: Amin Bigdeli, Negar Arabzadeh, Radin Hamidi Rad, Sajad Ebrahimi, Charles L. A. Clarke, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2606.13905)  

**Abstract**: LLM-based query expansion improves retrieval by enriching the original query with additional context. Yet most methods remain generation-driven, producing plausible pseudo-documents or expansions without checking how the target corpus responds. This can introduce retrieval drift, amplify misleading vocabulary, or miss terms that distinguish relevant from non-relevant documents. We argue that effective expansion requires retrieval-grounded feedback, not just single-pass generation or unverified iteration. We introduce ADORE (ADapt, Observe, Relevance Evaluate), an iterative framework that turns retrieval outcomes into feedback for the next expansion. At each round, an LLM generates pseudo-passages, a retriever exposes the corpus response, and a relevance assessor evaluates retrieved documents against the original query. These judgments identify what to reinforce, what remains undercovered, and what to suppress. Across TREC Deep Learning, BEIR, and BRIGHT, ADORE consistently outperforms strong query expansion baselines with notable improvements across nearly all evaluation settings, improving average nDCG@10 by 24.5% over BM25 and 3.6% over the strongest prior query expansion method on BEIR, and by 122.9% over BM25 and 9.2% over the best query expansion baseline on BRIGHT. Our code and data are publicly available. 

---
# Mood-Aware Music Recommendation: Integrating User Affective Signals into Ranking Systems 

**Authors**: Terence Zeng, Abhishek K. Umrawal  

**Link**: [PDF](https://arxiv.org/pdf/2606.13858)  

**Abstract**: Recommendation systems are essential in modern music streaming platforms due to the vast amount of available content. While collaborative filtering is widely used to suggest items based on the preferences of others with similar patterns, it performs poorly in domains where user-item interactions are sparse, such as music. Content-based filtering is an alternative approach that examines the qualities of the items themselves. Genre, instrumentation, and lyrics have been explored; however, relatively little attention has been given to emotion recognition. Since a user's emotional state strongly influences their music choice, incorporating mood signals offers a promising direction for personalization. In this work, we propose a mood-conditioned ranking framework that integrates user affective signals into the recommendation process via softmax-based sampling in the energy-valence space. We evaluate the approach via single-blind experiments in which participants compare recommendations from the proposed system against a baseline. The results indicate improved perceived recommendation quality, providing preliminary evidence for the effectiveness of incorporating mood-based inputs into music recommendations. 

---
# Hybrid Neural Retrieval with Generative Query Refinement for Quranic Passage Retrieval 

**Authors**: Mohamed G. Salman, Mohammad E. Moftah, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2606.13837)  

**Abstract**: Quranic Passage Retrieval (PR) could be a challenging task due to the linguistic complexity and the semantic gap between the Modern Standard Arabic (MSA) used in daily queries and the Classical Arabic (CA) of the Holy Quran. These factors hinder conventional retrieval methods. To handle these limitations and improve multi-verse retrieval and filter the zero-answer queries, this paper proposes a four-phase neural architecture designed to enhance retrieval accuracy and contextual understanding. The methodology combines hybrid candidate retrieval using AraColBERT dense indexing and BM25 sparse retrieval, followed by semantic reranking with a CAMeLBERTmix cross-encoder. A confidence gating mechanism is then applied to filter zero-answer queries, and an AraT5-based refinement module for multi-verse aggregation. The system is evaluated on an expanded version of the Quran QA 2022 dataset. Results show improved performance compared to the baseline models, achieving a Recall@10 of 0.7024 and a Mean Average Precision (MAP@10) of 0.4947. While the system exhibits a marginal tradeoff in absolute top-rank precision (MRR = 0.5807) compared to heavily optimised single models, the proposed architecture provides a substantially more comprehensive, reliable, and context aware solution for multi-verse Quranic passage retrieval. 

---
# TASR: Training-Free Adaptive Stopping for Iterative Retrieval 

**Authors**: Adrian Kieback, Uyiosa Philip Amadasun, Aman Chadha, Aaron Elkins  

**Link**: [PDF](https://arxiv.org/pdf/2606.13814)  

**Abstract**: Iterative retrieval-augmented generation agents commonly overspend by continuing to retrieve after the model has converged on an answer, incurring calls that change neither the prediction nor the supporting evidence. Existing remedies learn a stopping policy from labeled trajectories, tying the decision to a trained component that requires retraining for each new model or task. We propose TASR (Training-Free Adaptive Stopping Rule), a one-line predicate that fires when the model repeats its previous-round normalized answer and the isotonically calibrated logit margin exceeds 0.25. No classifier or value head is learned; the threshold is fixed across all twenty-four (model, retriever, corpus) configurations we evaluate. On a 3-model x 2-dataset distractor grid, TASR retains 94.8% of fixed-k=5's macro F1 at 62.6% of its calls and exceeds fixed-k=3 by +3.42 F1. The pattern holds on nine open-domain BM25 cells (55.01 F1 at 2.98 calls vs. 54.33 at 3.00 for fixed-k=3) and, with calibration locked from the distractor split, on nine dense-retrieval cells across two retriever families, with zero significant regressions in either extension. The rule was selected from an exhaustive enumeration of 381 candidate stopping rules; no alternative Pareto-dominates it on any evaluated configuration. A signal-quality analysis shows that verbalized 1-5 confidence collapses on RLHF-tuned models (96.5% of values equal 5, entropy 0.182 nats), while the logit margin achieves 44x better class-conditional separation, grounding the design in a measurable model pathology. TASR is an auditable, training-free Pareto baseline against which learned stopping controllers can be compared. Code is publicly available. 

---
# Nomenclature Ontology for Medical And Disease names (NOMAD): taxonomy of types and origins of disease names 

**Authors**: Spiros Denaxas, Cai Ytsma, Giannos Louloudis, Jackie MacArthur, Harry Hemingway  

**Link**: [PDF](https://arxiv.org/pdf/2606.13719)  

**Abstract**: The nomenclature of human disease has developed organically over the past centuries using Greek, Latin, and Arabic terminology and reflects the idiosyncrasies of different eras of medical discovery. Despite evident heterogeneity in naming practices, no systematic framework exists for characterising these conventions across all diseases. In this paper, we describe the Nomenclature Ontology for Medical And Disease names (NOMAD), a meta-taxonomy that classifies disease names according to their naming conventions. We developed a two-level taxonomy comprising 9 top-level categories and 20 subcategories and applied it to 22,548 index entries from the ICD-10-CM 2026 Alphabetical Index in a scalable three-stage machine learning-driven classification pipeline. Classification was multi-label, reflecting the compositional nature of medical nomenclature. We classified 99.1% of terms with a mean of 2.12 labels per entry. Anatomical categories were the most prevalent (63.8% of entries), followed by Descriptive (48.4%) and Pathophysiological (40.2%), while Eponymous and Geographical labels were less common than their cultural prominence might suggest (9.7% and 1.9% respectively). Among all Eponymous diseases, we identified only 57 (2.6%) of diseases named after a female person. We manually reviewed a random sample of n=2,255 entries (10%) for accuracy and calculated a full agreement rate of 70% and partial agreement rate of 26% (macro-averaged Cohen's Kappa score 0.832). Naming convention profiles varied substantially across ICD-10-CM chapters, reflecting specialty-specific epistemological traditions: infectious disease chapters were dominated by etiological labels and showed the highest proportion of geographical region related labels, the circulatory chapter by anatomical and pathophysiological labels, and mental and behavioural disorders showed the highest prevalence of socio-behavioral labels. 

---
# Personalization and Evaluation of Conversational Information Access 

**Authors**: Hideaki Joko  

**Link**: [PDF](https://arxiv.org/pdf/2606.13717)  

**Abstract**: Conversational interactions have reshaped information retrieval systems, as users increasingly favour direct answers over traditional hyperlinks. To build reliable Conversational Information Access (CIA) systems that account for personal context, this thesis addresses challenges: (1) personal context extraction, (2) personalized response generation, and (3) effective and interpretable system evaluation. First, we tackle personal context extraction by studying what Entity Linking (EL) in conversations entails, introducing a dataset for conversational entity linking (ConEL), and proposing CREL, a novel EL method tailored for conversational settings. Second, we focus on personalized response generation by proposing LAPS, a method for efficiently constructing large-scale, human-written, personalized conversational datasets, and using them to study how users' preferences can be utilized to generate personalized responses. Finally, we address the need for effective and interpretable system evaluation by introducing FACE, an automatic, reference-free method that assesses entire conversations and aligns closely with human judgments. 

---
