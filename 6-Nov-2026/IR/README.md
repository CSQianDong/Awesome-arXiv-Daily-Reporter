# GINGER: Spicing Up Graph-Based Vector Search 

**Authors**: Daniel Salwasser, Lars Gottesbüren, Laxman Dhulipala, Jakub Łącki, Guy Blelloch, Rajesh Jayaram, Tobias Rubel, Kshiteej Sheth, Amir Zandieh, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2611.08306)  

**Abstract**: We present \textsc{Ginger}, a state-of-the-art open-source C++ library for graph-based approximate nearest neighbor search (ANNS). We developed \textsc{Ginger} at Google to push the speed and memory efficiency of graph search on large and high-dimensional embedding datasets. \textsc{Ginger} surpasses existing ANNS libraries with regards to query throughput and indexing time while matching the memory footprint of the most space-efficient competitors.
\textsc{Ginger} builds upon recent advances in proximity graph construction (PiPNN) and vector quantization (TurboQuant). We introduce Nested TurboQuant (NTQ), a two-level quantizer which supports low precision graph navigation and high precision reranking in a single code. This representation achieves near floating point precision with $2.6\times$ less memory. We also compress the size of our proximity graphs by up to $2.5\times$ without impacting query throughput. We combine graph reordering with a new lossless compression codec called AdaptiveGap, which builds upon StreamVByte. At $95\%$ recall, \textsc{Ginger} speeds up query throughput by an average of $1.8\times$ over each dataset's fastest competitor and by at least $2.3\times$ over any single library. This lead grows to $2.9\times$ over compact libraries on high-dimensional embeddings. 

---
# Domain-specific information retrieval 

**Authors**: Suzan Verberne, Florina Piroi, Allan Hanbury  

**Link**: [PDF](https://arxiv.org/pdf/2611.08228)  

**Abstract**: Information Retrieval (IR) is much more than ad hoc web search: a large portion of the information seeking activities takes place in specific domains, often addressing work-related tasks. Examples are lawyers finding relevant prior cases for a legal case and academic researchers studying the background material for a paper. In Chapter 2 we outline the most common domain-specific tasks, datasets and applications. Domain-specific IR has a number of challenges that make set it apart from simple web search tasks: they often deal with long documents, long queries and long sessions with specific user needs. We discuss these in Chapter 3. We also address issues related to bias and fairness of models. In Chapter 4, we describe the models and methods developed for domain-specific tasks and data, addressing the specific challenges, including user aspects related to domain-specific IR tasks, dealing with queries, user models, and user interactions. We show that many of the methodological advances in domain-specific IR are transferable to other domains and even to open-domain web search. We conclude with the main takeaways and propose research directions for the near future. 

---
# Probabilistic Relevance Decisions for Semantic Retrieval 

**Authors**: Bahadır Tatar, Cevat Şener  

**Link**: [PDF](https://arxiv.org/pdf/2611.08014)  

**Abstract**: Dense similarity supplies reusable corpus representations, while probabilistic relevance decisions offer query-dependent document scoring. We compare dense-only BGE-M3 with TypeSafe Jev Noul and OpenAI Decisions Predicate scores on the official BEIR SciFact test benchmark. The study evaluates 300 official test queries using identical positive-enriched 100-document pools and full-corpus dense retrieval followed by reranking at candidate depths of 50 and 100. Unlike the controlled pools, neither dense prefix receives positive injection. Jev has the highest observed Recall@5 and nDCG@10 under the evaluated conditions. At depth 50, full-corpus nDCG@10 is 0.6413 for BGE-M3, 0.7545 with OpenAI and 0.7805 with Jev. Extending depth from 50 to 100 increases known-positive candidate recall but produces small, uncertain final-effectiveness changes. All four paired-bootstrap depth intervals include zero. Unique-condition API scoring cost approximately doubles, while compatible cached decisions reduce incremental expenditure. OpenAI returns many exact-zero scores, and secondary exact-tie randomization reveals ranking sensitivity without replacing deterministic primary results. Top-100 latency mixes cached and new execution. Historical hybrid latency is a saved-component sum under different execution environments and timer boundaries. These exploratory, unadjusted analyses support relevance decisions as a complement to dense retrieval within this protocol, without establishing general superiority, probability calibration or equivalence between depths. 

---
# MEMOIR: A Training-Free Memory Lifecycle for Long-Horizon LLM Agents 

**Authors**: Mayank Anand, Subham Raj, Aman Vaibhav Jha, Sriparna Saha  

**Link**: [PDF](https://arxiv.org/pdf/2611.07631)  

**Abstract**: Long-horizon LLM agents require memory systems that preserve useful information across sessions and retrieve it when needed. Existing systems treat these as separate problems: memories are written prospectively with static importance scores that are never revised after use. This creates two coupled failure modes: consolidation can silently destroy evidence that future queries require, while retrieval continues to favor stale memories. We propose MEMOIR (Memory Evidence via Merge-certified, Online Importance Revision), a training-free memory lifecycle that unifies judge-gated consolidation with retrospective attribution. Before compressing memory, screening gates test whether answer-relevant evidence is preserved. After each query, attribution estimates which retrieved memories supported the answer and updates their retrieval priority. A UCB-style exploration bonus additionally encourages retrieval of under-exposed memories, reducing the risk that frequently retrieved memories crowd out useful but less-seen evidence. On 1,905 deduplicated LoCoMo QA pairs, the full retrieval stack improves evidence recall@5 by 11.92 percentage points over bare cosine, with confidence-gated feedback adding 1.50 points overall without reference answers or reranker training; a deployment-unavailable oracle adds 1.84 points overall and 8.05 points on multi-hop. Judge-gated merge changes the recall of text-based evidence from 58.82% to 59.08%, reduces the active store by 1.42%, and incurs 0 observed losses in 100 independently audited merges, with a one-sided 95% upper loss-rate bound of 2.95%. Together, these results outline a practical deployment path: no additional LLM call on the online write path, with merge screening confined to periodic offline consolidation. The lifecycle reduces to one principle: attribute usefulness after use, and screen for preservation before compression. 

---
# Beyond Independent Predictions: Learning the Browsing Journey for Sequence-Level Re-ranking 

**Authors**: Jiayi Zhang, Jingxin Hai, Zhenjia Ma, Hongbin Pei, Dagui Chen, Han Zhu, Jian Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.07591)  

**Abstract**: Re-ranking aims to present users with an ordered list that maximizes sequence-level utility. Industrial deployments widely adopt the Generator--Evaluator framework, where the generator proposes candidate sequences, and the evaluator scores them to select the best one for display. The fidelity of the evaluator's utility estimates thus directly governs recommendation quality. However, existing evaluators predominantly follow the \textbf{Independent Behavior Modeling (IBM)} paradigm, which simplifies the continuous user \emph{browsing journey} into independent per-item predictions (e.g., CTR). This oversimplification ignores two physical realities: (i) a click can occur only after the item has been reached and impressed, and (ii) items compete for finite user attention. As a result, IBM produces systematically biased evaluations and suboptimal sequence selection. To address this, we propose a \textbf{Joint Behavior Modeling (JBM)} perspective that explicitly models the \emph{display} $\rightarrow$ \emph{impression} $\rightarrow$ \emph{click} process to capture causal and competitive dependencies within a sequence. Building on JBM, we develop the \textbf{Impression-Click Evaluator (ICE)}, a plug-and-play framework that (i) predicts a browsing-depth distribution to estimate per-position impression probabilities, (ii) leverages predicted impressions to suppress spurious context from unexposed items during click modeling, and (iii) introduces a click distribution matching objective to capture intra-sequence dependencies. Extensive offline experiments across diverse backbones, together with a 14-day online A/B test in the Taobao App (+0.70\% clicks, +3.67\% revenue) and full-traffic deployment, validate the effectiveness and robustness of our approach. 

---
# What Gets Hacked in a Collaborative Reward for LLM Recommenders: The Part That Does Not Depend on the User 

**Authors**: Zhuangsheng Li, Liang Dong, Yu Wu, Dailin Li, Yu Xia, Yunhan C. Zhang, Rui Cai, Yi Wu, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2611.07588)  

**Abstract**: Collaborative-filtering (CF) scores provide graded, personalized rewards for training large language model (LLM) recommenders through reinforcement learning. However, these rewards can cause exposure collapse: even though the CF model itself produces diverse top recommendations across users, an LLM optimized with its rewards concentrates on fewer items. We decompose each score into an item-mean component, the item's average CF score across user histories, and a residual component, the current user's score deviation from the average. Through controlled interventions, we identify the item-mean component as a contributor to exposure collapse. Accordingly, we propose two reward corrections: per-item centering removes each item's estimated mean score, while per-item standardization further rescales the residual using its estimated standard deviation. Experiments on two datasets with two identifier schemes show that both corrections substantially improve catalog coverage and exposure evenness. Compared with training on the uncorrected CF reward, our corrections increase the number of distinct items ranked first across users by up to $9.8\times$, while maintaining comparable ranking accuracy. Code is available at this https URL. 

---
# OptST: Optimizing Semantic Tokenization for LLM-Enhanced Recommendation 

**Authors**: Hanchong Chen, Jie Liu, Fuyuan Lyu, Dugang Liu, Lingjie Li, Xiuqiang He, Xing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07523)  

**Abstract**: Semantic information can enrich the identifiers and structured fields used by recommender systems. Semantic tokenization provides a structured representation of this information by converting LLM-derived user profiles into discrete tokens. In the common two-stage pipeline, the tokenizer is trained before the recommender. Its token assignments are then treated as fixed input features during recommendation training. As a result, the downstream recommender can learn how to use the assigned indices, but it cannot revise the semantic partition that produced them. Making this partition task-aware is difficult because nearest-code assignment is discrete, and unconstrained adaptation may also distort the geometry learned during pretraining. Moreover, a fixed token depth exposes the same semantic resolution to every interaction. We thus propose OptST (Optimizing Semantic Tokenization), which makes a pretrained tokenizer responsive to downstream recommendation. To revise token assignments without directly updating the pretrained encoder or base codebooks, OptST learns a low-rank deformation whose drift is limited by an anchor regularizer. Recommendation supervision reaches this transformation through a straight-through estimator, while prediction retains hard nearest-code assignments. Beyond adapting the assignments themselves, OptST selects an interaction-conditioned prefix to control how much residual detail is exposed to the recommender. Experiments on four public datasets show that OptST achieves the highest AUC among conventional and LLM-enhanced baselines. Its improvements also persist across different recommendation backbones. These results establish task-aware semantic tokenization as an effective interface between LLM-derived profiles and downstream recommendation models. 

---
# BRIDGE: Bridging the Semantic Quantization Gap for Generative Recommendation 

**Authors**: Yatong Xiao, Mingjie Sun, Peng Wang, Wenjin Wu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07425)  

**Abstract**: Generative recommendation retrieves items by autoregressively generating Semantic IDs(SIDs), where an RQ-VAE quantizes each item into a hierarchical SID sequence that the decoder generates level by level. We identify an overlooked bottleneck in this paradigm: the Semantic Quantization Gap(SQG). Quantization derives SIDs from rich continuous signals, including codebook centroids, local neighborhood geometry, and level-wise entropy. However, the decoder receives only discrete identifiers and loses the semantic structure that shaped them. This mismatch manifests in three ways: a grounding gap, where the decoder lacks semantic anchors when target-side information is limited; a neighborhood gap, where token-level supervision discards inter-item similarity and exacerbates exposure bias; and a difficulty gap, where computation is allocated uniformly despite generation difficulty diminishing with depth.
To bridge the SQG, we propose BRIDGE. Codebook-grounded latent reasoning injects frozen quantizer centroids as semantic anchors before SID generation. Privileged EMA self-distillation transfers neighborhood-aware knowledge from a teacher to an on-policy student, preserving inter-item geometry while mitigating exposure bias. Effort-aware recurrence allocates more computation to the challenging early levels and progressively reduces it with depth. Across three Amazon benchmarks, BRIDGE improves Recall@10 and NDCG@10 by 10.9% and 9.7% on average over the strongest baseline (up to 12.6% and 10.7%). In a one-week online A/B experiment on a large-scale e-commerce advertising platform with 5% traffic, BRIDGE improves eCPM by 0.65%. 

---
# CacheGraft: Enhancing Retrieval-Augmented Generation via Latent-Space Document Enrichment 

**Authors**: Vihan Lakshman, Peter Baile Chen, Zhuohan Gu, Michael Cafarella  

**Link**: [PDF](https://arxiv.org/pdf/2611.07393)  

**Abstract**: Recent work in machine learning has demonstrated that transformer-based language model agents can communicate state not only through text, but also directly in latent space via key-value (KV) caches. In this paper, we argue that such cache-to-cache translation can provide utility beyond the original motivating application of real-time agent communication and can instead be utilized for retrieval-augmented generation (RAG). Building on the Cache-to-Cache (C2C) fusion framework, our proposed system CacheGraft encodes each corpus passage offline into a fused KV cache representation between a larger sharer model (4B parameters in our experiments) and a smaller receiver model (0.6B parameters) such that this cache can be injected into the receiver at query time with no need for the larger model. Across five QA benchmarks and two independent locally-run judges, CacheGraft improves answer quality over the text-only receiver baseline in 29 out of 30 comparisons, of which 17 of the 30 are statistically significant. Notably, these gains are zero-shot with respect to the fuser, which was never trained on retrieval data. These initial results suggest that latent-space document enrichment may be a promising avenue for enhancing the performance of RAG systems, particularly in compute-constrained settings where it is only feasible to deploy a small model at inference time. 

---
# Reducing Representational Interference in Retrieval Embeddings with Trainable Prefix Tokens 

**Authors**: Seongtae Hong, Youngjoon Jang, Jungseob Lee, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2611.07298)  

**Abstract**: Embedding models compress variable-length text into fixed-size vector representations, but this compression can entangle semantic content with residual token-level information, such as artifacts arising from attention sink behavior. We propose a simple method that prepends trainable [DUMP] tokens to the input during embedding model training. The method keeps the model architecture, pooling strategy, and training loss unchanged, while optimizing the added tokens through the standard contrastive retrieval objective. By introducing trainable prefix positions that interact with regular tokens during encoding, [DUMP] tokens help reduce the influence of residual or uninformative token-level signals on the final retrieval representation. Experiments on standard retrieval and long-context benchmarks show that our method consistently improves over finetuned baselines across encoder and decoder embedding models. Further analysis shows that [DUMP] tokens alter token-level activation and attention patterns associated with low-information tokens. These results suggest that trainable prefix tokens provide a lightweight way to improve retrieval representations. 

---
# Decision Models as Retrieval Gates Against Injection and Pollution 

**Authors**: Mahdi Ghadamyari, Reza Habibi  

**Link**: [PDF](https://arxiv.org/pdf/2611.08325)  

**Abstract**: Retrieved web text can carry instructions aimed at the language model, or synthetic pages written to mislead. We test whether released open-weight decision models, which answer typed questions about a text without generating tokens, can screen retrieved passages against both threats. On a benchmark of 1,000 questions over forum-style web pages with hidden injected instructions, a Kev-4B gate reads each passage's HTML. Attack success, measured automatically, falls from 27.9% with no defence to 13.6% with the published mitigations (sanitisation, normalisation and citation prompting) and to 5.7% with the gate added. The gate takes 76 ms per passage and wrongly flags 4 of 186 benign pages with instructions for human readers. Responsiveness on clean queries stays within a point (85.6% to 86.2%). On a 200-query pollution testbed where two thirds of the candidate pages are fake, filtering the pool before the top ten are chosen keeps fake pages out of the answer (0.5% exposure) and holds accuracy at 83%, against 28% with no filter. Filtering only the top ten leaves the generator too few pages to answer. Large LLMs used as judges rank attacks better (AUROC 0.99 against 0.94) but cost ten times more. Gates catch more attacks when they read the page's HTML than when they read its plain text. Every model is open-weight. 

---
# ArcadeDB in Python: An In-Process Multi-Model Database 

**Authors**: Taewoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.07971)  

**Abstract**: Some Python workflows need transactional reads and writes over documents, traversals over a graph, and similarity search over vector embeddings. In practice these are split across separate systems (a document store, a graph engine, and a vector index), and the data is copied between them and kept consistent by application code. ArcadeDB is an open-source multi-model database engine, written in Java, that supports all three access patterns and can be embedded in a host process. We present arcadedb-embedded, a Python package that runs ArcadeDB inside the Python process via JPype, using a Java Runtime Environment bundled in the wheel. One pip install gives a database with documents, a property graph, and vector search, and needs no Java installation. Query results can be returned as Python lists, NumPy arrays, pandas DataFrames, or Arrow tables. We demonstrate a hybrid retrieval workflow that chains vector search, SQL filtering, and graph traversal over one dataset. We benchmark arcadedb-embedded against SQLite, DuckDB, LadybugDB, and Chroma, and report its costs. 

---
# ZeroRAG: Adaptive Semantic Peer Routing for Decentralized Retrieval-Augmented Generation 

**Authors**: Petru Neague, Quinten Stokkink, Martijn de Vos, Rafael Pires, Mathis Randl, Anne-Marie Kermarrec, Johan Pouwelse  

**Link**: [PDF](https://arxiv.org/pdf/2611.07928)  

**Abstract**: Retrieval-augmented generation (RAG) is a standard approach to enhance the reliability of large language model (LLM) outputs by retrieving documents relevant to a query and supplying them in the LLM prompt. Today's RAG systems collect, index, and serve these documents from centralized stores. In practice, however, documents are owned by organizations, devices, and individuals who cannot or will not surrender custody of them to a central party, e.g., because of privacy, copyright or regulatory compliance. Existing modern decentralized retrieval techniques either reintroduce a coordinating authority, or rely on a global data structure that requires continuous network-wide coordination (e.g., DHTs). We present ZeroRAG, a decentralized routing layer for document retrieval in RAG that performs semantic query routing across an overlay of peers with- out global data structures or centralized orchestration. Documents remain with the peers that own them, and each peer keeps only lightweight local state: a bounded set of neighbor advertisements and a cache of recently routed queries. Using these, ZeroRAG steers each query through a best-first walk that progressively rewires the overlay toward semantically relevant neighbors, so recall improves over time as the network self-organizes. We evaluate ZeroRAG using three real-world workloads and two decentralized baselines. We find that ZeroRAG outperforms baselines in terms of recall and required network messages, approaches the recall and LLM answer accuracy of a centralized document store, self-heals after catastrophic peer failures, and scales to 100 000 peers. 

---
# Persistent Retrieval for Conversational Agents: A Study of Retrieval Efficiency 

**Authors**: Karthik Ravula, Sathwik Kowtha, Rishabh Verma, Kala Mehak Jain, Gunjit Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2611.07899)  

**Abstract**: Conversational agents can retrieve information directly or delegate retrieval to a subagent. This design choice affects how retrieval history is maintained, what context each model call receives, and what information must be re-retrieved in follow-up questions. We compare three designs: skill-based retrieval in the main agent, a non-persistent retrieval subagent that generates a findings response, and our persistent retrieval subagent, which maintains retrieval history across turns and returns retrieved evidence directly to the main agent. Our approach also allows the main agent to request corrections from the same subagent.
We evaluate the three designs on enterprise questions posed independently, within shared conversations, and as dependent follow-ups. Across five multi-turn settings, our approach uses 43.3% fewer recorded agent-model tokens and incurs 45.0% lower estimated API cost than skill-based retrieval, while reducing average response time by 9.1%. Compared with non-persistent retrieval, it reduces response time by 15.6 seconds per question, while using 65.1% more tokens and incurring 29.1% higher estimated API cost.
Our approach shifts the retrieval subagent from producing synthesized responses to maintaining retrieval state and supplying evidence that the main agent can directly evaluate and use for answer generation. 

---
# CORE: Coarse-to-Fine Search with Context Management for Deep Research 

**Authors**: Hoang Tran Vuong, Ngoc Quan Pham, Hung Bui, Fatih Porikli, Dat Quoc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.07859)  

**Abstract**: Deep research agents answer complex questions through repeated search. They rely on a harness that runs their tool calls and returns evidence to their context. This creates a trade-off: Search works best over long passages, which give enough context for reliable semantic matching. But the returned evidence should be short, because every observation stays in the agent's context for the rest of the trajectory. We introduce CORE, a training-free harness that combines coarse-to-fine search with context management. CORE retrieves passages with a dense retriever. It then reuses the token-level similarities from a late-interaction reranker to return only the most relevant snippet from each passage. Context management lets the agent expand the context around sources it has already retrieved and suppresses repeated snippets. In controlled closed-search comparisons with the Tongyi DeepResearch and OpenResearcher harnesses, CORE achieves state-of-the-art answer accuracy and evidence recall across three agent models and three benchmarks (BrowseComp-Plus, BrowseComp-Wikipedia and BrowseComp-Plus-CM), on corpora containing up to 553M documents. 

---
# Tasteply: Externalizing Stable and Situational Preference as Manipulable Spatial Objects 

**Authors**: DaEun Choi, Daniel Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.07851)  

**Abstract**: Most recommender systems treat user preferences as a single signal to infer and optimize, yet human preferences are a mix of a stable preference and situational preferences that turn on and off depending on context. Through Tasteply, we externalize this preference structure as manipulable spatial objects. Tasteply represents each situational preference as an attractor and the stable preference as an underlying global field. Users can adjust their feed by moving around the space and adjusting the strength of the global field, turning the preference structure that remains latent within the model into something they can see and act on directly to steer the recommendation algorithm. Through an exploratory study (N=5), we found that this spatial control was effective, along with its limits and future direction as a means of manipulating preference-based recommendations. 

---
# MEvince: Knowledge-Constrained LLM Reasoning for Technique Identification from Malware Reports 

**Authors**: Hongyu She, Yiling He, Mengda Xie, Yiping Liu, Xingzhi Qian, Zhan Qin, Lorenzo Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2611.07486)  

**Abstract**: Mapping behavioral intelligence in malware analysis reports to ATT&CK techniques supports threat analysis and defense, but reliable identification remains challenging. Supporting evidence is often scattered across long, heterogeneous reports, while closely related techniques share behavioral cues and differ only in subtle but decisive details. While large language models (LLMs) can use external knowledge to interpret reports, they may assign techniques based on plausible associations rather than sufficient evidence. We present MEvince, a knowledge-constrained LLM reasoning framework for report-level malware technique identification. Its core is a shared behavioral representation connecting report observations to technique definitions. This representation is built through agentic mining of technique documentation and supports two online stages: 1) primitive-guided extraction uses behavioral primitives to structure LLM-extracted observations, link them to source passages, and group dispersed evidence; 2) evidence-aligned retrieval matches this evidence to technique profiles through the same concepts, retaining likely candidates and confusable alternatives for LLM comparison of distinguishing behaviors. Controls at both stages validate evidence references and check technique-specific requirements. The representation also supports adaptation without parameter updates, where offline agentic audits diagnose concept weaknesses and refine extraction or retrieval policies. We evaluate MEvince against 11 baselines spanning four method families using a dataset of 610 public malware reports referenced by MITRE ATT&CK. MEvince improves exact-ID F1 over the strongest baseline by 17.4%, and its concept-guided retrieval increases gold-label candidate coverage from 31.6% to 47.9%. With a small adaptation set, offline refinement yields further gains and outperforms baselines that update model parameters. 

---
# MapHub: Daily Relevance Heatmaps of New arXiv Preprints for a Research Community 

**Authors**: Kaili Cao, Camille Avestruz, Feige Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.07174)  

**Abstract**: MapHub, the MIDAS arXiv Preprint Hub, assesses each day's new arXiv submissions for every participant in a research community, with a large language model reading each paper against portfolios built for the participants. It publishes the relevance scores, from 0 to 100, as a heatmap that shows whose work each paper touches. In the authors' blind ratings, the top 10\% of papers by MapHub's scores held about 70\% of the raters' own top 5\%, pooled, against about a third for a text-matching baseline. About 6\% of a day's astro-ph papers reach 30 for each participant, and one every three days reaches 60. MapHub runs on a standard Claude subscription and is designed to be useful for any field: this https URL. 

---
