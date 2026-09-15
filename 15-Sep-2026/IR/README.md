# IROH: Insightful Ranking Of Humor using Multi-Stage Hybrid Retrieval with Rationale-Distilled LLM Judges for JOKER 2026 Track Task 1 English 

**Authors**: Ana-Maria Luisa Mocanu, Sebastian Mocanu, Ciprian-Octavian Truică, Elena-Simona Apostol  

**Link**: [PDF](https://arxiv.org/pdf/2609.15618)  

**Abstract**: Our team, VANGUARD, presents IROH (Insightful Ranking of Humor), a three-stage retrieval system for JOKER Task 1 English at CLEF 2026, achieving first place on the leaderboard with 0.6347 MAP. Our pipeline combines hybrid sparse-dense retrieval, cross-encoder reranking, and a LoRA-adapted Large Language Model judge ensemble. We employ Gemma 4 to generate query-aware rationales under two prompt strategies, generic and typed, and produce up to four types of structured hard negatives for training data construction. Through an ablation across three cross-encoder architectures, four dense embedders, and eight judge configurations, our key findings are threefold: (1) the rationale-distilled judge is the primary driver of ranking quality, whereas appending rationales to the first-stage index contributes negligibly; (2) structured hard negatives degrade generalisation in nearly all configurations despite inflating local validation scores; and (3) across the components we ablate, the lighter, better-calibrated model is competitive with or stronger than its larger counterpart, with the generic-rationale Qwen2.5-7B judge (0.6055 MAP) outperforming every Gemma-4-31B configuration, and the advantage of generic over typed rationales is concentrated almost entirely in the smaller model. 

---
# Self-Evolving Memory for Generative Recommendation 

**Authors**: Xinyu Lin, Zhuosong Jiang, Zixiao Suo, Siqin Wang, Hanqing Zeng, Hanchao Yu, Yinglong Xia, Jiang Zhang, Aashu Singh, Fei Liu, Wenjie Wang, Fuli Feng, Yang Song, Qifan Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2609.15598)  

**Abstract**: Generative recommendation has emerged as a promising end-to-end paradigm for personalized recommendation. However, user preferences continuously evolve over time, making self-evolving an essential capability for generative recommender systems. Existing evolving strategies, such as continual retraining and distillation-based adaptation, directly update the shared model parameters using streaming interactions. Nevertheless, we find that directly applying such strategies to generative recommendation introduces a critical issue, termed evolution conflict. Specifically, heterogeneous preference shifts from different users are optimized within a fully shared autoregressive parameter space, causing dominant behavioral patterns to progressively dominate the model evolution process while underrepresented patterns become increasingly overlooked. To address this issue, we propose a self-evolving memory paradigm for generative recommendation, aiming to enable effective evolution across heterogeneous behavioral patterns. We further identify three key principles for effective self-evolving recommendation systems, including isolated memorization, reinforced evolution, and scalable application. Guided by these principles, we develop LION, a simple yet effective framework centered on a sparse Key-Value memory layer. Specifically, LION introduces sparse memory activation to isolate the evolution of different behavioral patterns, while a consolidation loss is designed to reinforce the learning of underrepresented preference dynamics during continual adaptation. Extensive experiments on diverse real-world datasets demonstrate the effectiveness of LION under various continual evolution settings (e.g., per-period evaluation, user/item group evaluation, and evolution convergence analysis). The codes are released at this https URL. 

---
# The Magnitude Mirage: Rethinking Confidence for Reasoning-Intensive Retrieval 

**Authors**: Jamie Holdcroft, Abdelrahman Abdallah, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2609.15578)  

**Abstract**: Many production RAG systems implement retrieval abstention by thresholding raw similarity scores, implicitly treating score magnitude as a confidence signal. We demonstrate that this practice degrades systematically as queries require reasoning beyond semantic matching. Across 11 retrieval architectures and 28 datasets, neural retrievers consistently assign high similarity scores to semantically related but constraint-violating documents, causing magnitude-based thresholds to collapse toward near-random abstention performance on logical and temporal reasoning tasks---a failure we term the Magnitude Mirage. To address this without computationally expensive alternatives, we conduct a large-scale empirical study of six zero-cost Query Performance Prediction (QPP) metrics across three cognitive tiers: semantic matching (BEIR), logical reasoning (BRIGHT), and temporal reasoning (TEMPO). Our central finding is that the key improvement comes from abandoning magnitude in favor of score-distribution signals: the gain from this shift exceeds the differences among distributional alternatives by a factor of 5-10$\times$. In particular, Score Gap ($s_1 - s_k$) and a practical adaptation of Score Magnitude and Variance (LSMV) improve abstention AUROC by up to 0.16 in settings where magnitude-based confidence provides little discriminative power. These methods require no additional inference, retraining, or latency, making them a practical zero-cost replacement for magnitude thresholding in deployed RAG systems. 

---
# Beyond Retrieval: Scaffolding Children's Online Learning 

**Authors**: Diletta Micol Tobia, Hrishita Chakrabarti, Maria Soledad Pera, Monica Landoni  

**Link**: [PDF](https://arxiv.org/pdf/2609.15568)  

**Abstract**: Children increasingly turn to online information access systems that are primarily designed for the mainstream population, e.g., adults, but possess a limited understanding of how these systems work, contributing to their unstructured and ineffective search practices. This lack of knowledge can hinder their curiosity and the development of critical search skills. Grounded on the existing literature of both child-oriented Information Retrieval and Human Computer Interaction, our work positions children as active participants in the search process, framing it as a scaffolded learning experience rather than a simple retrieval task. 

---
# Benchmarking Embedding Models for ESG Data 

**Authors**: Motaz Saad, Veronica Cretì, Ivan Gentile, Kianna Kazemi, Antonella Longo  

**Link**: [PDF](https://arxiv.org/pdf/2609.15434)  

**Abstract**: The use of Environmental, Social, and Governance (ESG) data is fundamental for modern corporate accountability, sustainability reporting, and financial decision-making. Embedding models have emerged as a powerful approach for transforming unstructured ESG text into numerical representations suitable for downstream natural language processing (NLP) tasks. However, their effectiveness in these ESG-specific tasks has not been systematically studied. In this paper, we construct a benchmark dataset specifically tailored to the ESG domain. We benchmark fourteen models, both open-source and closed-source embedding models, comparing their performance with respect to retrieval, and Retrieval-Augmented Generation (RAG). The results demonstrate performance variations across different models, with Qwen3-based models achieving the highest overall performance. This study provides practical insights into which models are better suited for ESG RAG tasks. 

---
# Clean Scores, Buried Evidence, and Confident Wrong: A Receipt-Based Audit of Frontier Agentic QA 

**Authors**: Luis M. Sánchez  

**Link**: [PDF](https://arxiv.org/pdf/2609.15319)  

**Abstract**: Frontier models score well on shallow document/chart reading tasks. In a controlled data-room audit, moving evidence into buried conditions reduced accuracy, increased forced declarations, increased tool calls, and increased cost per correct answer. Confidence and benchmark calibration did not fully capture wrong answers; a documented production incident shows fabricated structural claims can be mixed with accurate numeric tables. Agentic evaluations need claim-level receipts (statement-level provenance, not answer-level scores), condition-aware scoring, and human-adversarial verification - an auditing discipline, not a leaderboard. The setting we measure is financial due diligence; the setting we are building toward next is defense staff work, where the same buried-evidence shape appears. In both, the model is not a party to the consequences; the person who signs is. In plain terms: in the documented cases we examine, agents can pair accurate numbers with confident fabricated explanations, and the burden of proof must therefore move from the model to the evidence trail. 

---
# ProLiVis 2.0: Literature-Centric Visualization of Protein--Protein Interaction Networks, with a Citation-Trust Model for Interaction Evidence 

**Authors**: Melih Sözdinler, Yalçın Doksanbir, Gökhan Akpınar, Ege Aktan  

**Link**: [PDF](https://arxiv.org/pdf/2609.15236)  

**Abstract**: Protein-protein interaction databases record evidence without weighing it. In BioGRID, an interaction asserted once by a single high-throughput screen and one confirmed by twenty laboratories across a dozen assays are the same kind of row in the same file. Tools built on such databases inherit that flattening: they draw every reported interaction as an edge, and the resulting picture states that two proteins interact without stating how much anyone should believe it.
We present ProLiVis 2.0, a rewrite of the literature-centric visualization system of arXiv:2111.12794. It contributes three things. First, a citation-trust model that scores each interaction from seven terms, including a term for the number of independent laboratories behind the supporting publications, obtained by clustering those publications over shared institutional affiliations; a plain count of publications cannot distinguish five confirmations from one group publishing five times. Second, a deterministic reformulation of the center layout, closed-form and $O(n \log n)$, which replaces the force-directed placement of the original and makes published figures regenerable from a session manifest. Third, an implementation that runs entirely in a web browser, with an embedded analytical database, requiring no installation and uploading no data.
On BioGRID release 5.0.260 restricted to SARS-CoV-2, 24,344 of 34,540 reported interactions (70%) rest on a single publication, and raising the trust threshold to 0.2 leaves 11,320 of them. That the large majority of a curated interaction network is unreplicated is a fact no existing view of the database makes visible. 

---
# Top-K Is Not a Budget for Hybrid Retrieval 

**Authors**: Chunran Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.15143)  

**Abstract**: Modern hybrid retrieval for RAG typically fuses the Top-$L$ results from dense and sparse retrievers, but a fixed truncation depth may not transfer across changing queries and corpora. Exact fusion removes the dependence on a fixed depth, yet completing a specified Top-$K$ still incurs variable access costs. We present DiBud, which takes an access budget directly as input and incrementally certifies and returns an exact prefix of the RRF ranking over the full lists. Selective access increases certified output within the budget, while budgeted stopping bounds accesses per request. Experiments on five query sets reveal long-tailed costs for completing exact Top-20. At a budget of 2048 accesses, DiBud increases mean certified output within the first 100 positions by 7.86% over balanced access. After budget calibration for 95% quality retention, held-out queries retain 95.05%--97.68% of mean nDCG@20 while using 65.92%--99.53% fewer accesses than completing exact Top-20. 

---
# Generate to Explore, Select to Exploit: Aligning LLM-based Headline Generation with Personalized Recommendation 

**Authors**: Yi Chen, Rufeng Cheng, Qiang Xie, Tao Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.15094)  

**Abstract**: In industrial recommendation feeds, presenting a static headline for an item often fails to satisfy the diverse, multimodal interests of the user population, particularly suppressing the needs of long-tail audiences. While Large Language Models (LLMs) have been integrated into recommendation for content understanding or ranking, directly optimizing them to output a single best headline typically leads to mode collapse---converging to generic patterns that satisfy average tastes but miss specific latent intents. To bridge this gap, we introduce GESE (Generate to Explore, Select to Exploit), a framework operating at the system's presentation layer that decouples personalization into generative exploration and selective exploitation. First, we treat the LLM as a probabilistic explorer, utilizing Group Sequence Policy Optimization (GSPO) with a hierarchical reward mechanism to generate a candidate set that maximizes the semantic coverage of potential user interests. Subsequently, a lightweight, real-time feedback-aware selector acts as the exploiter, identifying the optimal realization from the candidate pool based on instant contextual signals. Extensive deployment on a commercial platform with over 100 million daily active users demonstrates that GESE significantly outperforms state-of-the-art baselines, achieving a 2.57% lift in CTR and 0.87% in dwell time. These results validate that decoupling diversity-oriented generation from precision-oriented selection offers a robust blueprint for aligning generative AI with dynamic user utility. 

---
# LazFormer: Scaling Transformers for Industrial Recommendation via Transferable Generative Pre-training 

**Authors**: Xiaodong Li, Alin Fan, Mingyang Li, Yan Xiao, Shichao Nie, Junfeng Zhang, Shaochuan Lin, Zhanming Ou, Tao Luo, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2609.14978)  

**Abstract**: Transformers have shown promising performance in LLMs due to their outstanding scalability, several studies have investigated the scalability of Transformers for industrial recommendation. They typically rely on a single ranking model to optimize both sparse and dense parameters from scratch, resulting in substantial computational resource consumption and slow convergence. Fortunately, the pre-training models offer an effective solution to the above issues by providing favorable initialization of both sparse and dense parameters for the subsequent ranking. However, they still face two major limitations: (1) Since the input features used in pre-training and ranking are usually inconsistent, directly transferring dense parameters from pre-training to ranking may lead to negative transfer. (2) Multi-epoch training during the ranking process may result in the overfitting of sparse parameters, while freezing the sparse parameters limits their adaptability to the ranking objectives. To this end, we propose a Scaling Transformer for Industrial Recommendation via Transferable Generative Pre-training, termed LazFormer. Specifically, we first present a generative pre-training module to autoregressively generate sequential features, providing favorable initialization of both sparse and dense parameters for the subsequent ranking. To solve the negative transfer of dense parameters, we propose a transferable residual adapter that injects additional ranking-specific features into ranking in a residual manner. Moreover, a request-aware ranking module integrates long-sequence compression, hybrid sparse attention, and a request-aware paradigm to efficiently model users' long sequences. Besides, we further propose an asymmetric multi-epoch training strategy that resets sparse parameters while continuously accumulating dense parameters across epochs, alleviating the overfitting of sparse parameters. 

---
# Route Me If You Can: A Benchmark for Query Reformulation Selection 

**Authors**: Hai Son Le, Negar Arabzadeh, Amin Bigdeli, Radin Hamidi Rad, Sajad Ebrahimi, Charles L. A. Clarke, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2609.14885)  

**Abstract**: LLM-based query reformulation can improve retrieval, but no single reformulation strategy is consistently optimal across queries, domains, retrievers, or model backbones. This creates an inference-time decision problem: ``Given an original query and a pool of candidate reformulations, which one should be issued to the retriever?''. Existing studies are hard to compare because they use different reformulator pools, retrievers, relevance signals, training labels, and evaluation metrics. We introduce QueryRoute, a benchmark that freezes the expensive artifacts needed to study this decision reproducibly: original queries, generated variants, ranked lists under multiple retrievers, retrieval scores, and per-query oracle labels. The benchmark contains 3,757 queries, 11 candidate systems, five reformulator backbones, and three retrievers across TREC DL, BEIR, and BRIGHT, yielding 619,905 retrieval outcomes. We benchmark supervised classification, routing, QPP, and LLM-as-judge selectors. Results show substantial oracle headroom over fixed reformulators, but current selectors recover only part of it; selector rankings change across retrievers, and similar mean effectiveness can hide different query-level behavior. The released artifacts and evaluation harness allow future selectors to be compared without regenerating variants, rerunning retrieval, or rebuilding judge pipelines. Code and data are available at this https URL 

---
# EviQE: Evidence Selection for LLM-Based Query Expansion 

**Authors**: Hai Son Le, Amin Bigdeli, Shirin Seyedsalehi, Morteza Zihayat, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2609.14875)  

**Abstract**: LLM-based query expansion increasingly conditions reformulation on documents retrieved from the target corpus, yet most work focuses on how to generate expansions rather than which documents the model should read. We propose EviQE, which aggregates documents retrieved by multiple reformulators, selects a compact evidence set, and uses it for one grounded expansion step. This separates evidence selection from generation and treats reformulators as complementary retrieval perspectives. Across three TREC DL and five BEIR benchmarks, reformulators frequently retrieve distinct relevant documents, so pooled candidates provide higher relevant-document coverage than any individual source. The strongest gains come from relevance-based evidence selection: LLM-Score consistently outperforms direct reformulation, cold-start expansion, and single-source seeded expansion. Additional retrieval-generation rounds provide little benefit once strong conditioning evidence has been selected and can reduce effectiveness. 

---
# Beyond Benchmark Scores: How Synthetic and Authentic Query Distributions Diverge in RAG Evaluation 

**Authors**: Filip J. Kucia, Barbara M. Gawlik  

**Link**: [PDF](https://arxiv.org/pdf/2609.14579)  

**Abstract**: RAG systems are routinely evaluated using synthetic question sets generated from the target document corpus. While this practice provides a useful check on overall retrieval capability, relying exclusively on synthetic benchmarks can mislead under distribution shift and overstate deployment readiness. Synthetic generation spreads questions evenly across the corpus, formulating long, detailed queries; real users put most of their traffic on a few administrative and procedural topics in short queries, while also asking about matters the generator never covers at all. We demonstrate this gap on a university faculty information system, comparing 1,851 synthetic questions generated via Gemini Notebook against 322 authentic queries collected via a student survey. The synthetic and authentic query sets differ significantly: authentic queries average 6.8 words versus 15.7 for the synthetic ones, and draw from only 53 unique sources compared to 165. Consequently, configurations that appear highly effective on synthetic benchmarks experience a substantial performance drop on authentic queries. Importantly, optimizing on synthetic queries selected a higher-latency hybrid retriever. In our setting the sparse retrieval component benefited long synthetic questions but not short authentic ones, costing up to $8\times$ the latency of the fastest configuration we tested. We propose treating synthetic and authentic query sets as complementary extremes of the query-quality spectrum: synthetic data verifies maximum retrieval capacity under idealized conditions, while authentic queries test system robustness to the imprecise, underspecified inputs of real users. 

---
# The Wisdom of the Loudest: A Large-Scale Audit of Generative Search on Reddit 

**Authors**: Agam Goyal, Wang Claire, Eshwar Chandrasekharan  

**Link**: [PDF](https://arxiv.org/pdf/2609.14575)  

**Abstract**: Online communities are valued not only for answers, but for the diversity of experiences and perspectives they contain. Generative search increasingly mediates access to this discourse, yet little is known about which community voices survive retrieval and synthesis. We audit Reddit Answers using 10,000 queries from 20 advice- and support-seeking communities, repeated three times to produce 30,000 answers over 14.68M comments. We find that differences across runs are driven primarily by retrieval, answers routinely combine evidence across communities, and selection strongly favors already-visible, top-level comments. Formal and directive language is more likely to be surfaced, while experiential voice is less likely to survive selection and is further weakened during synthesis, with first-person singular language declining sharply. These findings show that community-grounded generative search is not neutral summarization, and should be designed not only for relevance and fluency, but also for provenance, plurality, and legibility. 

---
# VARG: Value-Aware and Ranking-Aligned Generative Retrieval for Dynamic E-commerce Search 

**Authors**: Xiaopeng Chu, Jianbo Zhu, Mingmin Jin, Jing Wang, Xing Fang, Wenyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.14493)  

**Abstract**: Integrating recall and pre-ranking in e-commerce search requires candidate generation to account for relevance, personalization, and business value before final ranking. To this end, we present VARG, a generative retrieval system for Tmall App search that directly admits generated item candidates to the existing final ranker. VARG-ID constructs semantic prefixes using RQ-VAE, enhances search relevance through bidirectional query-item contrastive learning, and combines these prefixes with a value-ordered third token to provide fine-grained item addresses and a business-value prior. Three-stage supervised fine-tuning progressively learns item-to-identifier mappings, query-semantic retrieval, and personalized retrieval. Personalized model training combines value-aware and hierarchy-aligned supervision with expanded user context, and uses local ordinal supervision (LO-SFT) to learn the local within-cluster ordering encoded by the third token. Prefix-GRPO combines gated rewards based on output legality, user behavior, ranker advantage, and search relevance with prefix-aware token weighting to align candidate generation with business value and ranking objectives. Coordinated daily product and model updates preserve existing item addresses while incorporating new products and behavioral feedback. Offline experiments on tens of millions of products validate identifier stability and demonstrate gains in retrieval quality and head-level value recall from SFT strategies and Prefix-GRPO over their respective baselines. In a 14-day online A/B test covering 20% of search traffic, VARG directly admits generated candidates to the final ranker and improves GMV by 1.45%, per-user IPV by 0.22%, and PCTR by 0.31%. Online shopping-guide query evaluations further show that VARG maintains competitive relevance with a smaller candidate quota. 

---
# TF-IDF and BM25 Are Exact KL Divergences 

**Authors**: Ivan Silajev  

**Link**: [PDF](https://arxiv.org/pdf/2609.14016)  

**Abstract**: TF-IDF and BM25 are two of the most widely used methods for scoring query-document relevance, yet neither has a standard probabilistic derivation that justifies it as a statistical method within a unified framework. We address this gap by showing that both scoring methods admit an exact interpretation as Kullback-Leibler divergences between two probability models. We treat the BM25 variant that includes the plus 1 correction in the IDF term, which is the one used in practice, and also discuss the original BM25 formulation without that correction. The resulting framework provides a common theoretical basis for TF-IDF and BM25, clarifies what they measure, and allows them to be compared theoretically with other information retrieval methods rather than only experimentally. 

---
# P3Rec: Distilling Prior--Posterior Preference Reasoning for LLM-based Recommendation 

**Authors**: Jinfei Chen, Weihai Lu, Jiawei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.13993)  

**Abstract**: Large language models (LLMs) exhibit strong semantic understanding and preference reasoning capabilities, offering new opportunities for user modeling in recommender systems. Existing LLM-as-Enhancer methods typically distill LLM-derived preference knowledge into lightweight recommenders to avoid costly online LLM inference. However, they often construct distillation knowledge from only one perspective. Prior preference captures users' stable and consistent interests but provides limited guidance for the current decision, whereas posterior preference reveals target-relevant fine-grained interests but may rely excessively on target clues. To address these limitations, we propose P$^3$Rec, a framework that jointly extracts and internalizes complementary prior and posterior preference reasoning knowledge. Specifically, P$^3$Rec first derives target-agnostic prior preferences and target-conditioned posterior preferences from the user side, while further extracting item-centric preference representations from item semantics and predecessor interactions. It then progressively internalizes prior and posterior knowledge into behavioral representations through prior preference absorption and posterior-guided preference distillation. Since the resulting comprehensive preference representation may not always provide an equally decisive retrieval direction, P$^3$Rec further characterizes historical interest dispersion with interest entropy and adaptively calibrates the user representation before contrastive retrieval optimization. In this way, P$^3$Rec achieves more complete preference reasoning while preserving efficient recommendation. Extensive experiments on multiple public datasets demonstrate its effectiveness. 

---
# Odds-Shift Slippage in One-vs-Rest Rankers: Diagnosing and Repairing Reweighting-Induced Top-K Errors 

**Authors**: Akifumi Goto  

**Link**: [PDF](https://arxiv.org/pdf/2609.13810)  

**Abstract**: One-vs-rest rankers that show each user the top-$K$ of many rare labels usually counter imbalance with a per-label positive-class weight, scale_pos_weight $= n_-/n_+$. Elkan's identity says such a weight shifts label $j$'s log-odds by $\ln w_j$, so the model ranks by weighted odds rather than by the marginal that is Bayes-optimal for precision@$K$, and suggests inverting the shift afterwards; what a finite learner does with a weight in the thousands, and which repair then works, has not been measured. We call the gap between the promised and the realized shift odds-shift slippage and measure it on matched pairs of LightGBM and MLP models that differ only in the weights. On Santander the weight takes MAP@7 from 0.808 to 0.117; for the boosted pairs the ideal odds shift accounts for 23% of that loss (32% on Instacart; 98% for an MLP pair on the same rows) and slippage for the rest. We prove that a booster whose leaf steps are capped at $c$ realizes at most $T\eta c$ nat of shift in $T$ rounds at rate $\eta$, which a cap sweep confirms, and show that without a cap saturated cells tie at exactly 1.0, beyond the reach of any separable map. The analytic inversion therefore pays only where the shift was realized and nothing saturated, whereas per-label isotonic regression returns the Santander model to 0.784 (0.780 with the calibrator fitted on the validation period), but only if labels without calibration positives are mapped to their prior rather than passed through. On 11 public MULAN benchmarks and 5 learners the weighted model loses more than half of its MAP@$K$ in 8 of 55 cells, and on delicious and Corel5k the same repair returns it to the unweighted level; per-label calibration hurts where positives are scarce, a harm that a cross-validated rule removes. The recipe is released as oddslip. 

---
# Addressing Cross-Stage Decoupling of Semantic and Collaborative Signals in Generative Recommendation 

**Authors**: Jiayi Dan  

**Link**: [PDF](https://arxiv.org/pdf/2609.13678)  

**Abstract**: Generative recommendation reformulates sequential recommendation as autoregressive generation by encoding items into semantic tokens, enabling improved scaling capability and cross-domain generalization. However, existing generative recommender systems typically follow a two-stage pipeline, where item tokenization is largely dominated by textual semantics with limited incorporation of collaborative signals and interaction similarity, leading to code assignments that are misaligned with downstream generation. Conversely, the generation stage tends to overlook the original semantic information, as the code sequences are re-embedded based on interaction data. This cross-stage information decoupling limits semantic coherence and recommendation accuracy.
To address this issue, we propose SCRec, a general framework that enhances cross-stage coherence through bidirectional information supplementation. Specifically, we introduce (i) collaborative-enhanced tokenization to explicitly inject textualized collaborative signals into semantic tokenization, without introducing additional alignment task, (ii) semantic-guided generation to dynamically recalibrate semantic priors with learnable code embeddings in generation stage, and (iii) manifold alignment to reconcile the geometric mismatch between the embedding space of discrete codebook indices and the dense continuous semantic space. These interrelated components form a general framework that aligns semantic and collaborative signals and enhances cross-stage information coherence, with minimal additional training and inference costs. Extensive experiments demonstrate the effectiveness, robustness, and generalizability of our proposed framework. 

---
# Pre-retrieval Query Clustering for Adaptive Top-k Document Retrieval in RAG Systems 

**Authors**: Ye Xia, Emre Yamangil, Haixun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.13489)  

**Abstract**: RAG systems commonly retrieve a fixed number of documents (top-k) to ground generation, but this static approach is brittle: simple queries suffer over-retrieval (adding noise and cost) while complex queries are under-retrieved, causing recall failures that cascade into incorrect answers. Motivated by the question of how many documents must be retrieved to answer an arbitrary query reliably, we propose a practical, general framework for query-adaptive retrieval depth. Offline, we estimate per-query retrieval difficulty by measuring NDCG under the default retriever and deriving a query-specific "saturation" point k* from the NDCG-k curve. Because computing these signals online is expensive, we cluster a large set of queries in embedding space and summarize each cluster with a recommended retrieval depth that targets high coverage (e.g., \textasciitilde{}95\%) using a mean-plus-variance rule. At runtime, the system assigns an incoming query to a cluster and selects the corresponding top-k in constant time. Compared with post-retrieval confidence methods that rely on clustering retrieved documents, our approach is pre-retrieval and query-centric, making it robust in heterogeneous, case-like corpora and applicable across domains such as legal, healthcare, finance, and enterprise search. Finally, this framework has been tested in full-traffic queries that improved $F_1$ by over 36\% while reducing token usage by 14\% on low-complexity clusters without accuracy loss. 

---
# Mixture-of-Experts Language Models Can Be Strong and Efficient Retrievers 

**Authors**: Anubhav Shrestha, Safal Shrestha, Minwu Kim, Torsten Suel, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2609.13486)  

**Abstract**: Recent work has shown that fine-tuning decoder-only large language models (LLMs) for retrieval yields strong first-stage retrievers, with effectiveness improving as backbones grow in size. However, every query and document must pass through the full model, so encoding cost increases with model size. Mixture-of-Experts (MoE) LLMs activate only a subset of parameters per token and are widely used to scale generative models, yet remain underexplored as retrievers. We systematically study MoE backbones for retrieval by training MoE and dense LLMs from several families using the same procedure, evaluating them across diverse datasets, and measuring query encoding time under the same serving configuration. We show that MoE retrievers outperform dense retrievers with comparable active parameter counts by up to 3.0 nDCG@10 points on BEIR. One of our strongest MoE retrievers matches an 8B dense retriever with 59% fewer active parameters and 18% lower query encoding time. We further show that the number of experts used for query encoding can be reduced without retraining or re-indexing, retaining more than 99% of retrieval effectiveness while reducing query encoding time by up to 26%. Recent rerankers provide only modest additional gains over strong MoE first stages, which often match or exceed the reranked configurations we evaluate. Together, these results show that MoE LLMs can be strong and efficient first-stage retrievers. 

---
# PCGNet: Unifying Shared and Specific Information for Fashion Matching Recommendations 

**Authors**: Shuiying Liao, P. Y. Mok  

**Link**: [PDF](https://arxiv.org/pdf/2609.13339)  

**Abstract**: In fashion domain, recommending complementary clothing items that match selected pieces is a crucial cross-selling technique that improves customer satisfaction. Nevertheless, fashion matching presents significant challenges, as recommendations must not only align with individual user fashion preferences but also ensure compatibility between garments. These challenges are twofold. First, existing models often assume an overly simplified decoupled relationship between product compatibility and personalized user preferences, overlooking the natural complexity between the two. Second, existing data-driven approaches are not optimized for real-world fashion data, which is typically sparse and characterized by noisy interactions. To address these challenges, we propose Personalized Compatibility Graph Network, a multi-objective graph learning framework that organically unifies the modeling of product compatibility and personal preferences. PCGNet uses contrastive mutual information maximization to extract and align shared and view-specific patterns, thereby capturing the complex interplay between compatibility and personal preferences. Moreover, we introduce a correlation-aware neighbor sampling and a learnable global graph augmentation, which enhance the model by incorporating self-supervised signals mined directly from the graph, ensuring more stable and informative representations. Finally, PCGNet generates recommendation scores through the joint optimization of BPR ranking loss and multi-view mutual information losses. Experimental validation on two benchmark datasets demonstrates that PCGNet significantly outperforming state-of-the-art methods across all four evaluation metrics. 

---
# Decoupling Error Attribution in Cloud-Native Graph-RAG: A Data Integrity Diagnostic Framework 

**Authors**: Shuai Yan, Yuhang Wu, Xiaodong Huang, Ke Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.13324)  

**Abstract**: Graph-RAG systems often assume pristine data quality, overlooking the severe impact of perturbations in cloud-native databases. This paper proposes a three-layer decoupled diagnostic framework to orthogonally attribute system errors to reasoning loss, Knowledge Graph (KG) defects, and Cypher generation errors. Evaluated on a spatio-temporal ecological KG of the Southeastern Tibet region with eight defect types, results reveal that data integrity, rather than algorithmic reasoning, is the dominant performance bottleneck, with structural defects degrading system accuracy from 0.93 to 0.39. Crucially, we observe a masking-like phenomenon termed the Parametric Knowledge Masking Effect (PKME), suggesting LLMs compensate for broken retrieval paths using internal memory. This shrinks apparent query generation errors by over 70 percent, obscuring actual storage deterioration and increasing the risk of false negatives for automated monitoring. This work provides a quantitative foundation for auditing and optimizing data integrity in cloud-based information fusion systems. 

---
# Self-Indexing Attention for Compression-Compatible Sparse Long-Context LLM Inference 

**Authors**: Xu Yang, Jiapeng Zhang, Zhangke, Changjian Chen, Yuxin Chen, Feiqiang Sun, Chengguang Xu, Feng Jin, Zhuo Tang  

**Link**: [PDF](https://arxiv.org/pdf/2609.13205)  

**Abstract**: Sparse long-context inference requires efficient token retrieval in both prefill and decode. Existing methods often use different retrieval strategies for the two stages, preventing one retrieval representation from being reused throughout inference. We propose Self-Indexing Attention, a training-free framework built on a shared transform-domain sign-magnitude representation. The key signs provide a reusable token-level index for grouped prefill selection and decode retrieval, while the same representation remains compatible with external KV-cache compression without separate indexer metadata. This 1-bit index enables efficient retrieval through bitwise operations widely supported by modern accelerators. At 5% attention density, Self-Indexing Attention remains close to dense attention on LongBench and RULER and achieves up to 6.1x prefill and 10.3x decode attention-operator speedups. Experiments with TurboQuant and DeepSeekV4-Flash further demonstrate compatibility with low-bit KV-cache compression and pretrained sparse-attention indexers. 

---
# CiteGuard-RAG: A Validation-Centered AI System for Evidence-Grounded Question Answering 

**Authors**: Sumit Barua, Guan Hong, Halil Dursunoglu, Charles Rodgers, Alvis Fong  

**Link**: [PDF](https://arxiv.org/pdf/2609.15830)  

**Abstract**: Retrieval-augmented generation (RAG) can improve access to complex information; however, retrieving evidence alone does not ensure that answers are grounded, citation-valid, or appropriately refused. This paper introduces CiteGuard-RAG, a validation-centered AI system for evidence-grounded question answering. The system integrates hybrid semantic-lexical retrieval, citation-constrained generation, sentence-level grounding validation, and single-pass regeneration. Validation is used at runtime to determine whether a candidate answer should be accepted, refused, or regenerated before final delivery.
CiteGuard-RAG is evaluated on 400 questions across a controlled housing-law dataset, PrivacyQA, and CUAD. In the controlled evaluation, it achieves 99.1% retrieval accuracy, 98.3% grounded-answer accuracy, and 98.3% citation validity, with no validation-detected hallucinations. Ablation results show that grounded-answer accuracy drops sharply when validation is removed, even when retrieval accuracy remains unchanged. External evaluation shows that while citation validity remains strong, evidence utilization, span alignment, and refusal calibration become harder under domain shift.
These findings indicate that trustworthy RAG systems require explicit validation between retrieval and final answer delivery. CiteGuard-RAG provides a practical architecture for linking retrieval, generation, citation checking, abstention, and regeneration in high-stakes information access. 

---
# Converting Sequenced Fuzzy Cognitive Maps to Causal Virtual Worlds with Large Video Generators 

**Authors**: Akash Kumar Panda, Olaoluwa Adigun, Bart Kosko  

**Link**: [PDF](https://arxiv.org/pdf/2609.14985)  

**Abstract**: We show how users can create and manipulate causal virtual worlds with large-language-model (LLM) and large-video-model agents. The approach uses feedback fuzzy cognitive maps (FCMs) both to model the granular causal structure of the virtual world and to guide its causal evolution. The local causal rules are partial or fuzzy while the FCM's feedback structure produces global equilibria that define causal scenarios. A sequence of \emph{dynamical} meta-rules of the form ``If $\mathcal{A}$ then $\mathcal{B}$" define the causal scenes of the virtual-world video. The if-part causal pattern $\mathcal{A}$ perturbs the FCM's virtual world at the user's or agent's discretion. The FCM's transient feedback dynamics define the meta-rule's causal arrow of implication. The then-part $\mathcal{B}$ is the resulting equilibrium attractor such as a FCM limit cycle or fixed point. Our algorithm extracts these meta-rules from the FCM and guides the LLM agent to write a script based on the FCM meta-rule sequence. The large video generator converts the meta-rule into a video scene in accord with the flow of the dynamics. We applied the agent-based technique to a simple FCM that describes an undersea world of dolphins and sharks. Google's Gemini 3.1 generated the script and Google's Veo 3.1 generated the dolphin-shark video. The approach is general and can scale by mixing larger FCMs and AI agents to produce more immersive virtual worlds. 

---
# Speak to the City: Multimodal Resolution for Outside-the-Vehicle References 

**Authors**: Alireza Parchami, Artin Saberpour, Robin Connor Schramm, Jürgen Steimle, Ulrich Schwanecke  

**Link**: [PDF](https://arxiv.org/pdf/2609.14691)  

**Abstract**: As autonomous vehicles and Extended Reality (XR) headsets enable novel in-car interactions, seamlessly querying physical landmarks, known as Outside-the-Vehicle Referencing (OVR), remains challenging due to ego-motion and referential ambiguity. We present a robust, multimodal OVR framework fusing user gaze and natural language to identify Points of Interest (POIs). To address the scarcity of dynamic vehicular data, we developed a VR-based pipeline synchronizing 360-degree transit videos with vehicle GNSS telemetry. Through a user study (N=46) mapping passenger head orientation into a 3D geospatial Digital Twin, we captured authentic gaze-speech behaviors. We subsequently trained a lightweight Transformer network, leveraging LLMs to dynamically align continuous spatial gaze vectors with discrete verbal context. Experimental results demonstrate high accuracy and low computational overhead, achieving an 83.33% Top-1 accuracy (87.72% Top-2) and an average inference time of 24.3 milliseconds. This real-time paradigm effectively resolves referential ambiguity, enabling context-aware spatial retrieval for passengers within the vehicle. 

---
# AlgoRAG: Retrieval-Augmented Generation for Theoretical Computer Science Education -- A Comprehensive Evaluation Framework for Algorithm Analysis and Complexity Theory 

**Authors**: Sushan Adhikari  

**Link**: [PDF](https://arxiv.org/pdf/2609.14572)  

**Abstract**: Teaching abstract theoretical computer science (TCS) concepts such as algorithm analysis and complexity theory is challenging because students must handle formal proofs and asymptotic reasoning that conventional resources rarely explain in an adaptive, on-demand way. We present AlgoRAG, a specialized Retrieval-Augmented Generation (RAG) system that couples a large language model (LLM) with a curated, domain-specific knowledge base to address these challenges. The knowledge base integrates authoritative textbooks, 847 lecture slides, 312 practice problems with solutions, 156 worked proof templates, and 89 complexity worksheets. AlgoRAG incorporates domain-specific optimizations including mathematical entity recognition, notation-aware retrieval, and pedagogical re-ranking. We evaluate AlgoRAG on 179 curated exam-style questions spanning asymptotic analysis, recurrence relations, dynamic programming, graph algorithms, NP-completeness, sorting, and divide-and-conquer. The system achieves a 100% success rate with a mean response time of 38.0 seconds. While BLEU-4 scores are zero -- a known limitation of n-gram matching on mathematical proofs where equivalent reasoning may use entirely different notation -- AlgoRAG attains ROUGE-1 F1 of 0.0963, ROUGE-L F1 of 0.0683, and a pedagogical quality score of 0.7620, indicating that responses are well-structured and didactically sound even when surface wording diverges from reference answers. Performance is especially strong on NP-completeness (ROUGE-1 F1 = 0.1285, pedagogical quality = 0.7643) and graph algorithms (ROUGE-1 F1 = 0.1023, pedagogical quality = 0.8086). These results support the conclusion that RAG is an effective architecture for personalized theoretical-CS instruction, providing correct, context-rich explanations even for highly abstract topics. 

---
# Question's Gambit: The First Move Matters in Agentic Deep Search 

**Authors**: Radin Hamidi Rad, Amin Bigdeli, Negar Arabzadeh, Sajad Ebrahimi, Charles L. A. Clarke, Benjamin C. M. Fung, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2609.14412)  

**Abstract**: Deep research agents answer complex questions through iterative loops of searching, reading, and reasoning. Recent work on reasoning-intensive benchmarks such as BrowseComp-Plus shows that well-configured lexical retrieval can surface high-quality evidence, yet agents may still fail to connect documents carrying evidence to the gold documents. We identify a deep research agent's first retrieval move as an important design decision for this setting. We introduce Question's Gambit, a first-move retrieval module that decomposes the question into a set of clues, reformulates them into complementary searches, consolidates the retrieved results, and reranks the candidate pool before the agent begins its iterative search-and-reasoning process. This produces an opening context designed to support both clue aggregation and final-answer verification. We further evaluate on MultiHop-RAG to test whether these benefits transfer beyond BrowseComp-Plus to a more conventional multi-hop question structure. Experiments on BrowseComp-Plus show that Question's Gambit improves retrieval recall and downstream agent accuracy over strong baselines, improving answer accuracy from 83.1% to 90.5% with gpt-5.5 over Pi-Serini, the strongest reported agentic baseline. Our results confirm that effective agentic deep research depends not only on the tools available inside the loop, but also on the quality of the first move. We published our implementation publicly at this https URL. 

---
# Semantic Knowledge Technologies: what the Semantic Web lost sight of, and what it never had 

**Authors**: Achille Zappa  

**Link**: [PDF](https://arxiv.org/pdf/2609.14121)  

**Abstract**: The Semantic Web set out to give information a machine-interpretable form so that software could integrate and reason over it. Its standards became scientific knowledge infrastructure, but the machine competence it promised did not follow, and the systems now answering questions over scientific knowledge are language models holding no inspectable account of what they know. This paper argues the original goal was right and the technical programme incomplete, states what is missing, and names the extended programme Semantic Knowledge Technologies: the same technical core carried out of its web-publishing origin and applied to knowledge wherever held. The diagnosis is that the standards formalised truth while omitting three things: the conditions under which a claim holds, the operations its terms permit, and any account of what a base covers. Without conditions, contradiction and applicability cannot be judged; without operational grounding, holding a statement confers no ability; without declared coverage, a system cannot recognise the boundary of its own content, which under the open-world assumption cannot be inferred. The paper fixes the word understanding to five measurable tests (check, connect, derive, act, delimit) and sets out a seven-layer architecture in which the first three layers are enabling and the rest the cognitive capabilities they make possible. It then defines three terms the programme implies: Large Knowledge Model, a model whose unit of output is a reference to an addressable claim, not a token; SLKM, the knowledge base an agent builds for itself from declared sources; and Semantic Artificial General Intelligence, stated as a falsifiable position about necessary conditions, not a system. A graded ladder replaces the untestable word general. It is offered as a research agenda, with its weakest points and refutation condition named. 

---
# ClinAgent: A ReAct-Based Agent for Conversational Access to Clinical Trial Information 

**Authors**: Antonino Vaccarella, Riccardo Cantini, Domenico Talia, Paolo Trunfio, Marianna Talia, Rosamaria Lappano, Marcello Maggiolini  

**Link**: [PDF](https://arxiv.org/pdf/2609.13860)  

**Abstract**: Querying clinical trial registries remains a manual and error-prone process, requiring researchers to navigate large volumes of semi-structured data without support for natural language interaction or cross-source synthesis. To address this, we introduce ClinAgent, a conversational system based on agentic Retrieval-Augmented Generation (RAG) that enables clinicians and researchers to query clinical trial information in plain language and receive grounded, up-to-date responses across multi-turn interactions. The system centers on a Large Language Model (LLM) agent following the ReAct paradigm, which iteratively reasons over queries, selects among a set of integrated tools, and refines its actions based on intermediate outputs. These tools include a this http URL search interface, a PubMed module, and a Python-based analyzer operating on a locally cached structured dataset of clinical trials. We evaluate the system using a three-phase framework assessing operational effectiveness, planning quality, tool-use efficiency, and expert qualitative judgments, comparing three LLM backends: Gemini 3.0 Flash and two variants of DeepSeek V3.2 (thinking and non-thinking). Results reveal complementary strengths, with DeepSeek (thinking mode) excelling in planning quality, while Gemini achieves the highest overall performance and strongest expert ratings. Overall, our findings highlight the potential of agentic AI systems to improve the accessibility and synthesis of clinical trial information, supporting more efficient and user-centered biomedical research workflows. 

---
# Cost Characterization of Vertically Partitioned Federated Knowledge Graphs 

**Authors**: Md Saikat Islam Khan Bappy, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2609.13664)  

**Abstract**: Knowledge graphs are increasingly distributed across autonomous organizations that share an entity space but own disjoint subsets of relations, forming a vertical partition. Answering a multi-hop query may require combining facts from several silos, making the partitioning strategy a key data management decision that affects communication, indexing, load balance, and query latency. However, the costs associated with different partitioning strategies remain insufficiently studied. We formalize vertical partitioning as a design space and compare four strategies: semantic domain grouping, frequency-balanced partitioning, co-occurrence graph-cut partitioning, and random partitioning. We evaluate them using five metrics: communication cost, candidate index size, cross-silo path length, load balance, and end-to-end query latency. Three of the five prove to be determined by the graph and the silo count rather than by the partition, which reduces the design problem to two conflicting axes, cross-silo path length and load balance. Experiments on MetaQA and PathQuestion use a fixed federated knowledge graph question-answering architecture based on TransE embeddings and a frozen BERT encoder across three silo configurations. By keeping the learning model unchanged, we isolate the effect of partitioning and show that the trade-off between locality and balance holds only where each silo can hold several relations, weakening as the number of silos increases. The study provides practical guidance for deployments constrained by cross-silo reasoning or by silo load. 

---
# FedV-KGQA in Practice: Design Lessons and an Interactive Prototype 

**Authors**: Md Saikat Islam Khan Bappy, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2609.13661)  

**Abstract**: Knowledge graph question answering usually assumes that one system can reach the whole graph. In practice, facts are often held by organizations that share entity identifiers but own disjoint relation types, so no single party sees a complete reasoning chain. This poster presents the empirical findings of FedV-KGQA on multi-hop question answering over such vertically partitioned graphs. Each silo enriches its local graph and trains a knowledge graph embedding on its own triples. A server then concatenates the silo-specific entity views, anchors the projected question at the topic entity, and ranks candidates by similarity. Raw triples and relation embeddings never leave a silo. Comparing the FedV-KGQA experiments with one another yields three results. First, federated fusion recovers most of the centralized accuracy, while a single silo recovers little. Second, anchoring and enrichment matter more than the choice of embedding model. Third, the cheapest encoder depends on the target accuracy rather than on parameter count. This poster paper contributes that cross-experiment comparison, four design lessons drawn from it, and an interactive prototype that runs real inference and traces the full pipeline, per question, on released checkpoints. 

---
# YOLO12-MambaScan: An Efficient Object Detector with High-Frequency Enhancement and State-Space Modeling 

**Authors**: Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.13647)  

**Abstract**: The rapid development of unmanned aerial vehicle (UAV) technology has made aerial-image object detection increasingly important for natural-resource monitoring, traffic management, and disaster response. Detecting small objects in aerial images remains difficult because objects occupy very few pixels, high-frequency cues are easily lost, and global context is hard to model in cluttered scenes. Existing detectors often retain insufficient edge, corner, and texture information. We propose \ours, an aerial-image detector built on the YOLO12 architecture. The model combines a triple-path high-frequency enhancement convolution module (TriPathHFConv), receptive-field coordinate-attention convolution (RFCAConv), and a Mamba-based global-context module. On VisDrone, at an input resolution of 960*960, ours achieves 60.0% mAP@50 and 38.6%mAP@50:95, demonstrating a favorable accuracy--efficiency trade-off for small-object detection. The benchmark and dataset protocol follow the VisDrone challenge setup. 

---
# Interpretable Temporal Video Reasoning with EventGraph and EventField 

**Authors**: Durgendra Narayan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2609.13258)  

**Abstract**: We present a structured temporal video reasoning pipeline built around a discrete EventGraph, a continuous EventField, and a human-readable EventGlyph view. On a calibrated EPIC-KITCHENS subset of 10 videos and 50 temporal reasoning questions, EventField+Glyph achieves 0.98 overall accuracy, which is higher than the caption baseline by +0.40 (paired p = 1.1 \times 10^{-5}) and direct VLM-only QA by +0.20 (p = 0.0063) on this subset. We further evaluate annotation-source variations, including manual, heuristic, and heuristic+Gemini pipelines, and find that the best structured method stays above the caption baseline across settings. We also include cross-video pair benchmarking and an appendix gallery of glyph outputs for all studied videos. Overall, the results indicate that structured temporal representations can support both performance and inspectability by preserving symbolic structure, capturing temporal continuity, and providing human-readable diagnostics for video reasoning. 

---
