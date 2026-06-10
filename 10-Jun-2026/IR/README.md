# Generative Archetype-Grounded Item Representations for Sequential Recommendation 

**Authors**: Yifan Li, Jiahong Liu, Xinni Zhang, Hao Chen, Yankai Chen, Wenhao Yu, Jianting Chen, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2606.11023)  

**Abstract**: Sequential recommendation aims to predict users' next interaction with items by analyzing their historical behavior. However, the limited quality of item representations remains a critical bottleneck. While pre-trained large language models (LLMs) can provide rich semantic representations, existing approaches only rely on static encoding of fixed attributes, overlooking the crucial role of target audiences in defining item identity. Moreover, the semantic space struggles to reflect actual user behavior, resulting in a significant gap between semantic representations and behavioral patterns. To address these limitations, we propose GenAIR, a general framework that empowers sequential recommendation with Generative Archetype-grounded Item Representations. Specifically, we first leverage an LLM to analyze item metadata and infer textual description of the Archetype, which represents the conceptual profile of the item's ideal target audience. We then extract the corresponding embeddings in a single forward pass. Further, to ground these generative archetypes in real-world behavior, we introduce a behavioral calibration objective, which explicitly incorporates behavioral signals from actual interactions. This objective adjusts the structure of the embedding space to reflect empirical patterns. GenAIR enables seamless integration with most existing models while maintaining high efficiency. Comprehensive experiments conducted on three real-world datasets demonstrate that GenAIR significantly improves the performance of various sequential recommendation models and consistently outperforms state-of-the-art baseline approaches. Implementation codes are available at this https URL. 

---
# miniReranker: Efficient Multimodal Reranking through Visual Cache Reuse and Interaction Sparsity 

**Authors**: Yingqi Fan, Xuan Lu, Anhao Zhao, Junlong Tong, Ping Nie, Kai Zou, Yunpu Ma, Wei Zhang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2606.10759)  

**Abstract**: Multimodal large language models (MLLMs) have recently shown strong potential as point-wise rerankers by directly modeling query--document relevance through next-token prediction. However, point-wise reranking suffers from substantial repeated computation across query--document pairs, while the causal structure of transformers allows only prefix segments to be reused via pre-caching. To address the misalignment of existing query-first and document-first formats with both VQA-style prompting and computation-aware reuse, we propose a \textit{vision-first} formulation that improves both cache reuse efficiency and reranking performance. However, the remaining cost is still considerable and stems from three main sources: (1) \textit{model depth}, for which we reduce active parameters via early exit; (2) \textit{cross-segment attention}, which we restrict to a narrow interaction band across a few layers; and (3) \textit{visual tokens}, where we reduce the number of tokens via embedder-guided pruning. Together, these designs form miniReranker, which reduces reranking runtime to <1% of the dense implementation under high-reuse settings for a single query, while preserving >96% of the dense model performance. 

---
# Effective Reinforcement Learning for Agentic Search by Recycling Zero-Variance Queries During Training 

**Authors**: João Coelho, João Magalhães, Bruno Martins, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2606.10709)  

**Abstract**: The use of GRPO-style algorithms has become the standard strategy for training LLM search agents under outcome-only rewards. With these algorithms, a query contributes to parameter updates only when its rollout group mixes successes and failures; all-correct (too-easy) and all-incorrect (too-hard) groups are zero-variance and waste rollout cost. Existing approaches treat zero-variance as a static property and either discard or pre-filter such groups. We hypothesize and empirically validate that queries flip between zero-variance and signal-bearing states as the policy evolves during training. Building on this intuition, we propose query recycling, which returns zero-variance groups to a mutable pool for future resampling, so that the effective training distribution co-evolves with the policy. With the proposed technique, a 1.7B parameter model trained on synthetic data can reach 66.0 average Pass@1 accross seven multi-hop QA benchmarks, matching or surpassing systems with up to 7B parameters trained on benchmark-derived supervision. Analysis of recycling patterns shows that recycled queries supply roughly three quarters of the effective batch by the end of training, with contributions split between recovery from policy improvement and policy drift. 

---
# Beyond Patches: Superpixel Token-based Transformers for Attribute-Specific Fashion Retrieval 

**Authors**: Shuili Zhang, Hongzhang Mu, Wenyuan Zhang, Duohe Ma, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.10697)  

**Abstract**: Attribute-Specific Fashion Retrieval (ASFR) aims to improve fine-grained image retrieval by focusing on specific attributes. However, existing patch-based attention and Transformer methods often misalign with irregular attribute regions and are prone to background noise, limiting their ability to capture subtle, pixel-level microstructures. To tackle these challenges, we propose SuperFashion, the first ASFR framework that adopts superpixel tokens within a Transformer architecture. SuperFashion initially employs an attribute-guided attention mechanism to extract attribute-related features, which in turn guide the cropping of semantically meaningful image regions. Superpixel segmentation is then leveraged on these regions to generate compact, semantically coherent superpixel tokens. By incorporating modality-specific embeddings for both attribute and superpixel tokens, the superpixel token-based Transformer facilitates adaptive interaction and fusion, thereby enhancing attribute localization and discrimination. Extensive experiments on FashionAI, DARN, and DeepFashion demonstrate relative overall MAP improvements of 1.84%, 9.27%, and 9.35% over prior SOTA. SuperFashion offers a new solution for web-based image retrieval. 

---
# STORM: Stepwise Token Optimization with Reward-Guided Beam Search 

**Authors**: Arthur Satouf, Giulio D'Erasmo, Yuxuan Zong, Habiboulaye Amadou Boubacar, Pablo Piantanida, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2606.10621)  

**Abstract**: Modern retrieval increasingly relies on dense and learned-sparse neural models that are effective but require encoding the entire corpus into a specialized index, rebuilt whenever the model changes. Lexical retrievers like BM25 stay efficient and transparent on a standard inverted index that need not change as models evolve, but suffer from vocabulary mismatch. LLM query rewriting can help, yet prompted rewriters emit well-formed but retrieval-ineffective or harmful-terms, and training against a retrieval reward gives only delayed, sequence-level supervision that obscures which terms helped. We introduce STORM (Stepwise Token Optimization with Reward-guided beaM search), a self-supervised framework for lexical query expansion. STORM trains the rewriter through generation guided by retrieval metrics: at each step, candidate expansions are scored against the BM25 index and low-reward continuations pruned, turning the retrieval reward into a token-level signal that concentrates exploration on retrieval-effective vocabulary. Across TREC DL and BEIR, STORM lets 0.6B-8B backbones match or surpass competitive LLM rewriters while retrieving as fast as plain BM25; at 8B it rivals far larger proprietary rewriters. It further transfers zero-shot to 18 languages (MIRACL), beating dedicated multilingual dense retrievers on average, making STORM a competitive, infrastructure-light alternative to dense neural retrieval. 

---
# Selection, Not Salience: The Shape and Limits of Personalization in Social Highlighting 

**Authors**: Kazuki Nakayashiki, Keisuke Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2606.10398)  

**Abstract**: Does personalizing what a reader sees pay off, and where does it stop? Using a social web highlighter and a co-readership identity control (the same document highlighted by many users, which holds document and topic fixed and asks whether a person's own history predicts their marks better than another reader's does), we map the shape and limits of personalization across reading altitudes. At the document altitude we give the clean, leakage-free, identity-controlled measurement that prior next-document evaluations could only upper-bound: a person's history identifies which documents in a co-reading neighborhood are theirs, with an own-versus-other gap of +0.169 against community negatives and +0.119 against topic-matched hard negatives (both highly significant); a content-based arm suggests the signal is not purely title-driven but is largely thematic. This is comparable to the span-level selection signal (+0.14) from our prior work: the selection signal is of comparable magnitude across altitudes (+0.12 to +0.17), most of it stable topic preference. At the sentence altitude, a two-stage personalized auto-highlight (an impersonal model proposes candidates, a personal model re-ranks them) does not improve on its impersonal baseline: two off-the-shelf zero-shot LLMs, including a frontier model, predict highlight locations worse than a lead baseline, and personal re-ranking is beaten by the salience order even on the highest-recall candidate pool, so the null is not merely a Stage-1 ceiling artifact. Measurable personalization appears primarily at the selection layer: modest (~+0.13), topic-dominated, with no reliable gain at the salience layer. We also surface a control-in-negatives bias that inflated our document gap to a spurious +0.227 until audited. Going beyond the shared salience layer may be better approached by aggregating individuals than by personalizing them harder. 

---
# SkillResolve-Bench: Measuring and Resolving Same-Capability Ambiguity in Agent Skill Retrieval 

**Authors**: Jiandong Ding  

**Link**: [PDF](https://arxiv.org/pdf/2606.10388)  

**Abstract**: Agent skill libraries are becoming routable software assets: a retrieved skill can contribute instructions, scripts, resource bindings, and execution assumptions to an agent. This makes skill retrieval more than broad relevance matching. A retriever can find the right capability family yet expose the wrong same-capability representative. We study this failure as same-capability execution-risk retrieval. Each query pairs a helpful skill with a query-specific risky sibling that shares the capability family but can lead execution toward a stale resource, missing precondition, or wrong procedure. We introduce SkillResolve-Bench 1.0, an auditable benchmark for this setting with 661 helpful/risky pairs, source-role and admission evidence, cue/leakage checks, query-disjoint splits, and a 7,982-candidate pool that includes 6,660 public SkillRet candidates. The benchmark reports helpful ranking together with harmful sibling rate (HSR@K), the top-K exposure of the risky sibling. We also provide SkillResolve, a reference method that resolves active candidate families, scores query-conditioned utility from confusable library negatives and contract-profile cues, and selects one representative from each family before the final top-K list. Under the released family relation, SkillResolve reaches Recall@3 0.766 and NDCG@3 0.699 while keeping HSR@3=0. It improves over SkillRouter by 0.112 Recall@3 and 0.165 NDCG@3 while reducing HSR@3 from 0.693 to 0. Without representative selection, HSR@3 rises to 0.236 under the same scorer, identifying within-family representative choice as the mechanism that turns capability retrieval into safer procedural exposure. 

---
# SIDInspector: A Mapping-First Diagnostic Resource for Semantic-ID Tokenizers 

**Authors**: Jiandong Ding, Heng Chang, Huijie Qin, Tianying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.10375)  

**Abstract**: Semantic-ID (\sid) tokenizers are increasingly reused as standalone artifacts in generative recommendation: an exported item-to-code mapping becomes the address space that a later sequence generator must use. These mappings rarely come with a common inspection interface, so coverage gaps, full-code aliasing, behaviorally weak prefixes, tail compression, and prefix fan-out are often found only after downstream training. We present \tool, a mapping-first diagnostic resource for \sid tokenizer artifacts. \tool defines a small adapter contract over item mappings, metadata, interactions, and optional generator traces; validates the contract; and reports mapping-level probes for utilization, aliasing, neighborhood alignment, popularity allocation, and structural cost, with hooks for temporal churn and generator traces.
\tool reports inspectable artifact profiles before downstream leaderboard scores. The released resource covers four tokenizer artifact lines: a same-item GRID/RQ-KMeans-style and ReSID/GAOQ contrast on 23,742 Musical items, plus released LETTER and LC-Rec item-index artifacts. In the Musical contrast, the GRID-style feature-text export has 3,749 unique full codes and a 0.977 full-code aliasing rate, while ReSID/GAOQ is aliasing-free in its exported mapping. Yet the strongest prefix--co-occurrence alignment comes from a deterministic category-prefix control, not from either learned export row (0.447 versus 0.154 and 0.055--0.080), showing that addressability and behaviorally meaningful prefixes should be inspected separately. Cross-domain, fixed-reranker, and mechanism-probe checks support the same diagnostic direction: prefix alignment is a candidate-exposure signal, while final ranking quality remains a downstream model question. 

---
# Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations 

**Authors**: Zhuohang Jiang, Yuxin Chen, Shijie Wang, Haohao Qu, Zhou Jindong, Wenqi Fan, Li Qing, Dongxu Liang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.10357)  

**Abstract**: Cross-domain recommendation is a core problem in content-to-e-commerce platforms. Its objective is to leverage user interactions with content to infer potential purchasing intent on the e-commerce side, thereby enhancing conversion rates and commercial value. However, in real industrial scenarios, cross-domain recommendation faces multiple challenges: significant semantic gaps exist between different domains, and user cross-domain behavior sequences are often massive in scale and rich in noise. Although large language models (LLMs) possess powerful semantic understanding and reasoning capabilities, their millisecond-level inference latency makes direct application in online recommendation systems difficult. To address these issues, this paper introduces AIR (Atomic Intent Reasoning), an LLM-driven cross-domain recommendation framework designed for industrial-grade deployment. By migrating LLM inference to the offline phase and dynamically constructing user intent representations through efficient retrieval and composition during online operations, it achieves approximately 400* inference acceleration while maintaining semantic consistency. Experimental results across multiple public datasets demonstrate that our method achieves state-of-the-art performance in cross-domain recommendation tasks. Furthermore, large-scale online A/B testing conducted in Kuaishou E-commerce's real-world business scenarios shows that our approach delivers stable and significant improvements across multiple core business metrics, including a +3.446% increase in GMV, fully validating its effectiveness and practical value in industrial-scale recommendation systems. 

---
# $τ$-Rec: A Verifiable Benchmark for Agentic Recommender Systems 

**Authors**: Bharath Sivaram Narasimhan, Karthik R Narasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2606.10156)  

**Abstract**: As recommender systems transition toward agentic, multi-turn conversational interfaces, evaluation paradigms have struggled to keep pace. Current benchmarks often rely on "LLM-as-a-judge" evaluations, which introduce subjectivity, high costs and inconsistency. We present $\tau$-Rec, a benchmark for agentic recommender systems that replaces subjective evaluation with verifiable rewards and a reveal-tagged elicitation (RTE) mechanism that controls how task constraints surface during dialogue. By testing agents against structured catalog predicates and employing a pass^k reliability metric, $\tau$-Rec provides a systematic test for consistent reasoning. Our evaluation of nine configurations across five model families -- GPT-5.4, Claude Sonnet 4.6, Gemini 2.5 Flash, DeepSeek V4 Flash, Qwen3-32B and GPT-5 mini -- reveals a steep reliability cliff, where even the best model achieves only ~57% at pass^1 and ~38% at pass^4, highlighting a critical gap in current conversational agent deployment. All code and data are publicly available at this https URL. 

---
# MetaPlate: Counterfactual-Guided RAG-LLM Tool for Personalized Food Recommendation and Hyperglycemia Prevention 

**Authors**: Asiful Arefeen, Carol Johnston, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2606.10120)  

**Abstract**: Postprandial hyperglycemia is a key risk factor for metabolic disorders; however, existing dietary guidance is often static, impractical, and insufficiently personalized, providing recommendations that are difficult to follow or not impactful. While recent advances leverage continuous glucose monitoring (CGM) and machine learning to predict glycemic responses, these approaches are largely predictive and lack actionable guidance. Moreover, recommendation systems are often misaligned with user goals and require extensive input. We present MetaPlate, a counterfactual explanation (CF) guided, context-aware decision-support framework that generates personalized meal recommendations to mitigate postprandial glucose excursions in healthy adults. MetaPlate integrates multimodal data, including CGM readings, wearable-derived physiological signals, and user-provided meal inputs from $25$ individuals to model pre-meal context. A machine learning model predicts glucose response, while a CF optimization module adjusts meal composition modifying macronutrient amounts to maintain glucose levels within a target range ($\leq 140$ mg/dL). An LLM-based retrieval-augmented generation (RAG) layer enhances interpretability by producing human-readable recommendations using constrained search of the USDA food database. We evaluate MetaPlate via a structured expert-in-the-loop assessment with registered dietitians (RDs), comparing performance before and after prompt refinement. Results show improvements in meal realism, portion suitability, and recommendation likelihood, with expert feedback indicating a shift from clinically implausible outputs to actionable, contextually appropriate recommendations. Our findings emphasize the importance of domain knowledge and structured constraints in LLM-driven systems and highlight the potential of MetaPlate as a real-time personalized dietary decision-support tool. 

---
# Mult-DPO: Multinomial Direct Preference Optimization for Recommender Systems 

**Authors**: Yaochen Zhu, Harald Steck, James McInerney, Aditya Sinha, Yinhan He, Nathan Kallus, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.10078)  

**Abstract**: Direct preference optimization (DPO) is a simple and effective alignment strategy for large language models (LLMs) based on pairwise preferences. In recommender systems, however, user feedback is rarely pairwise. For a given context, e.g., a user, a session, or a conversation, we typically observe set-wise preferences with multiple positive items, where every positive item should outrank every unobserved or explicitly negative item, with no prescribed order among the positives or the negatives themselves. A natural generalization is to use the Plackett-Luce (PL) reward model, which extends the Bradley-Terry reward model underlying vanilla DPO from pairwise preferences to full rankings of candidates. However, we show that adapting the PL model to set-wise preferences requires marginalizing over all positive orderings, where the resulting expression is combinatorial in complexity. To address this fundamental challenge, we propose Mult-DPO, a novel DPO objective with a tractable multinomial surrogate likelihood over set-wise preference events for the user-preference alignment of LLM-based recommender systems. The multinomial construction is not itself a ranking distribution, but it is defined on the same reward-induced weight space and admits a closed-form DPO-style objective, enabling direct alignment of LLMs with multiple candidates through a classification-style objective. In addition, we prove that the multinomial DPO loss is a tractable upper bound on the marginalized PL DPO loss when optimizing against the set-wise preference data. We further characterize the tightness of this bound in terms of the relative total weight of positives versus negatives, which provides insights into tightening the bound with richer or harder negatives. Finally, we extend Mult-DPO to the alignment of LLMs with multiple preference levels. Code is available at this https URL 

---
# From Prompt to Purchase: How AI Brand Recommendations Move Consumers on the Open Web 

**Authors**: Michael Iannelli, Alan Ai  

**Link**: [PDF](https://arxiv.org/pdf/2606.10907)  

**Abstract**: When a conversational assistant recommends a brand to a user with no recent observed engagement, that user's same-name Google search rises +4.3 percentage points (pp) [3.1, 5.5], visits to the brand's own site +2.4 pp [1.4, 3.5], and brand-specific retailer-page visits +1.0 pp [0.3, 1.7] over matched backward placebos. Recovering that estimate is the work. The mention creates a brand exposure no web log attributes to the assistant, and the naive all-mention funnel that seems to measure it is confounded: many mentions are incidental references to brands the user already uses ("your Netflix download"), whose downstream visits are that existing customer's own behavior and surface as a brand-specific pre-trend. We measure off-platform response on a panel that joins opt-in clickstream to the same users' ChatGPT, Claude, and Gemini conversations, and isolate the effect with a pre-trend event study, a stance classifier, non-customer conditioning, and a within-response same-category control: incidental name-drops then move behavior far less (+1.8/+1.1/+0.3), and the named brand moves far more than unnamed same-category brands in the same response. The downstream path is mostly search-mediated and reaches both own sites and retailer pages, with a destination mix that tracks baseline brand-directed behavior rather than redirecting toward either. The design is observational and we do not observe transactions, so retail is purchase-adjacent. Standard referrer-based and last-click measurement miss this upstream exposure: assistants move observably-unengaged users into open-web brand navigation along a path attributed elsewhere. 

---
# Flash-GMM: A Memory-Efficient Kernel for Scalable Soft Clustering 

**Authors**: Gal Bloch, Ariel Gera, Matan Orbach, Ohad Eytan, Assaf Toledo  

**Link**: [PDF](https://arxiv.org/pdf/2606.10896)  

**Abstract**: We present \textbf{Flash-GMM}, a fused Triton kernel for efficient computation of Gaussian Mixture Models (GMMs) over large-scale data in a single GPU pass. By eliminating the need to materialize the full responsibility matrix in GPU memory, Flash-GMM achieves a \textbf{20$\times$} speedup over existing implementations and enables training on datasets more than \textbf{100$\times$} larger than previously feasible on one device. To demonstrate its impact, we integrate Flash-GMM into the IVF coarse quantizer for approximate nearest-neighbor (ANN) search. We show that soft GMM clustering is now a viable drop-in replacement for $k$-means, and that GMM responsibilities can be leveraged to assign border vectors to multiple clusters. Our approach reaches fixed recall targets with up to $1.7\times$ fewer distance computations, or equivalently, yields $+2$--$12$ recall@10 at matched computational cost. We release the kernel as an open-source project. 

---
# ConvMemory v2: A Recall-Preserving Top-10 Evidence Reranker for Conversational Memory Retrieval 

**Authors**: Taiheng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2606.10842)  

**Abstract**: We describe ConvMemory v2, an opt-in token-evidence reranker that sits after the lightweight ConvMemory v1 reranker and reorders only v1's protected top-10 candidate set. v2 is a fine-tuned ms-marco-MiniLM-L-6-v2 cross-encoder (22,713,601 parameters, measured from the released checkpoint) applied to the ten (query, memory) pairs that v1 has already selected; it does not change which ten memories are returned, so Recall@10 and Hit@10 are identical to v1 by construction, not by statistical coincidence. On the LoCoMo conversational memory benchmark (5 seeds, n = 4955 test rows), v2 raises FULL MRR from v1's 0.5824 to 0.6560 (paired bootstrap +0.0734, 95% CI [+0.0645, +0.0827]) and H@1 from 0.4440 to 0.5474. v2 closes most but not all of the gap to a much more expensive full-pool cross-encoder reference (mxbai-rerank-large-v1 over the top-500, MRR 0.6688): on FULL MRR v2 sits 0.013 below mxbai_top500, but on two raw-dense-hard slices (where v1's protected top-10 has higher recall than mxbai's own top-10) v2 exceeds mxbai_top500. A four-arm load-bearing ablation shows candidate-specific memory text is the mechanism: removing, shuffling, or replacing it collapses MRR below raw dense retrieval. v2 is best understood as a standard recall-preserving cascade pattern with LoCoMo-specific fine-tuning, an explicit anti-shortcut inference contract, and disciplined load-bearing analysis; its advantage over mxbai is slice-specific rather than a general dominance claim. This report extends the v1 technical report (arXiv:2605.28062). 

---
# Agentic Hybrid RAG for Evidence-Grounded Muon Collider Analysis 

**Authors**: Ruobing Jiang, Dawei Fu, Cheng Jiang, Tianyi Yang, Zijian Wang, Youpeng Wu, Yong Ban, Yajun Mao, Qiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.10381)  

**Abstract**: Muon collider research spans accelerator physics, detector instrumentation, and high-energy phenomenology, with relevant evidence scattered across a rapidly expanding and heterogeneous body of scientific literature. As high-energy physics (HEP) increasingly explores agent-assisted analysis workflows, efficiently locating, integrating, and verifying scientific evidence becomes an essential capability. While retrieval-augmented generation (RAG) offers a promising framework for scientific question answering, integrating agentic reasoning without compromising retrieval precision remains a key challenge. In this work, we present agentic hybrid RAG, an evidence-grounded RAG framework for muon collider research. The framework combines a hybrid retriever, integrating sparse lexical and dense semantic retrieval, with an agentic reasoning module for query decomposition, evidence expansion, and grounded answer generation. To enable systematic evaluation, we construct the first benchmark for retrieval-augmented scientific question answering in the muon collider domain, comprising a curated literature corpus together with dedicated retrieval and answer-generation benchmarks covering major detector and physics research topics. Extensive evaluation shows that hybrid retrieval provides the strongest retrieval backbone, while agentic reasoning is most effective for controlled evidence expansion and answer synthesis. Built on this principle, agentic hybrid RAG consistently outperforms representative retrieval and RAG baselines in retrieval effectiveness, answer quality, evidence coverage, and factual grounding. Together, the benchmark and framework provide a foundation for evidence-grounded scientific question answering and future HEP analysis agents operating over large-scale scientific literature. 

---
# Stability in Competitive Search with Results Diversification 

**Authors**: Itamar Reinman, Omer Madmon, Moshe Tennenholtz, Oren Kurland  

**Link**: [PDF](https://arxiv.org/pdf/2606.10053)  

**Abstract**: In a competitive search setting, publishers strategically modify their documents in response to induced rankings so as to improve their future ranking. We present a novel game-theoretic analysis of a competitive search setting where search-results diversification is applied. Our analysis reveals an inherent tradeoff between corpus diversity and corpus stability, where the latter corresponds to an equilibrium in a game. We analyze two representative diversification methods and show that stability need not necessarily be reached, leaving the corpus to rapid changes due to ranking incentivized modifications of publishers. We then present a novel approach to devise diversification-based ranking functions that are guaranteed to lead to corpus stability. 

---
# Less Context, More Accuracy: A Bi-Temporal Memory Engine for LLM Agents Where a Lean Retrieved Context Beats the Full History 

**Authors**: Liuyin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.09900)  

**Abstract**: Long-term memory is the missing layer for LLM agents: across sessions they forget, and the common workaround -- replaying the whole history into the prompt -- is expensive, slow, and, as distractors accumulate, less accurate. Most memory systems win on cost or latency but still lose to the full-context baseline on accuracy, and benchmark numbers are reported on inconsistent, non-reproducible harnesses, so one system appears at wildly different scores across sources. We present Engram, an open-source, dual-process memory engine on a bi-temporal data model. A fast write path appends lossless episodes with no LLM on the critical path; an asynchronous path extracts atomic (subject, predicate, object) facts, builds a bi-temporal knowledge graph, and resolves contradictions without an LLM call per fact -- invalidating, never deleting, so every fact keeps provenance and a supersession chain. A hybrid read path fuses dense, lexical, graph, and recency/salience signals, applies a point-in-time ("as-of") filter, and assembles a compact, provenance-tagged context. On the full 500-question LongMemEval_S, graded by the official category-specific judge, Engram's lean configuration -- answering from a ~9.6k-token retrieved slice, never the full history -- scores 83.6% vs. 73.2% for full-context (+10.4 points, McNemar p < 10^-6) at ~8x fewer tokens (9.6k vs. 79k), with 0/500 errored. The gain needs a hybrid read path: facts alone lose recall, while facts plus retrieved chunks recover detail. We also contribute a neutral, in-repo evaluation harness with the official judge baked in and the full-context baseline in every table, publish the raw per-question logs, and document the measurement-integrity pitfalls (truncation, home-grown judges, full-history leaks) that silently distort memory benchmarks. Every number ships with a command to reproduce it. 

---
# Representation Curriculum: Stagewise Training for Robust Ranking and Allocation 

**Authors**: Ehsan Ebrahimzadeh, Sina Baharlouei, Abraham Bagherjeiran  

**Link**: [PDF](https://arxiv.org/pdf/2606.09891)  

**Abstract**: Ranking in digital marketplaces is a dynamic exposure-allocation mechanism: displayed items shape discovery trajectories and success events logged by the platform to update future allocation policies. Modern ranking systems rely heavily on exposure-confounded signals (e.g. popularity estimates, CTR/CVR aggregates, and ID-based representation), because they are highly predictive under stationary demand. Yet this predictive power can become a learning shortcut: early access to exposure-dependent belief signals steers optimization toward over-reliance on them and away from exposure-independent merit signals (e.g., content-based competitiveness and semantic affinity). Consequently, the learned policy tends to entrench incumbents and degrade cold-start generalization and robustness under distribution shift. We propose Representation Curriculum (RC), a training-time intervention that temporally stages feature utilization. RC foregrounds content-based merit signals initially, then introduces exposure-dependent belief signals while anchoring the content pathway near the learned merit representation, curbing shortcut reliance on historical signals and mitigating gradient starvation on content signals. We formalize RC independently of task and hypothesis class and provide ranking-specific instantiations. In a Gaussian linear ridge setting, we derive closed-form solutions and sufficient conditions under which RC strictly reduces population risk on a cold-start target distribution, with a quantified Pareto tradeoff against source performance. Experiments on public learning-to-rank and recommendation benchmarks, and randomized online experiments in a large-scale e-commerce search system, show that RC measurably shifts reliance from historical belief signals toward content-based merit signals and yields consistent gains on cold populations with a controlled trade-off in head performance. 

---
# LLM-as-a-Discriminator: When Synthetic Tables Still Look Real 

**Authors**: Manel Slokom, Malek Slokom, Thierno Kante  

**Link**: [PDF](https://arxiv.org/pdf/2606.09865)  

**Abstract**: Privacy and data sharing are often in tension. Many organizations use synthetic data to reduce privacy risk and still share useful data. For tabular data, auditing privacy remains hard. In many cases, even humans cannot easily tell if a table is real or synthetic. In this paper, we propose a method based on LLM discrimination. We ask an LLM to classify each table sample as REAL or SYNTHETIC. We test two settings: C1 with table only, and C2 with table plus distributional metadata. We use LLaMA as an open model and Gemini as a reference model. In our experiments, we run three synthesis models, CTGAN, TVAE, and Gaussian Copula, on two public datasets, UCI Adult and ACS Census. We collect 451 valid trials. Our results show clear differences between models. On Adult, LLaMA reaches DRS=0% in reported cells, while Gemini reaches DRS=100% for CTGAN and TVAE. On Census, LLaMA predicts SYNTHETIC for most samples, while Gemini stays high in C1 but drops for CTGAN and TVAE in C2. We also compare with a classifier two-sample test (C2ST) and record linkage as distributional baselines, and with a human pilot of 2 annotators and 240 trials. Our results show that LLM discrimination is a practical privacy audit signal when model choice, per provider reporting, and data encoding are handled with care. For reproducibility, code and experiment scripts are available at this https URL. 

---
