# Learn to Pool: Lightweight Fine-Tuning for Flexible Multi-Vector Compression 

**Authors**: Stefan Josef  

**Link**: [PDF](https://arxiv.org/pdf/2607.06036)  

**Abstract**: Late interaction models have shown strong generalization capabilities, often outperforming much larger dense embedding models. One challenge to their widespread deployment is the large number of token vectors they produce per document and the associated storage and memory costs. Pooling tokens at inference time has shown great promise to reduce the vector count with limited effects on retrieval accuracy. Large-scale pooling-aware training has demonstrated even more impressive results at high compression rates. We propose lightweight fine-tuning as a practical alternative and find that even minimal pooling-aware training with k-means yields broad gains over inference-only pooling, shows evidence of transfer across pooling methods and datasets, and - with multi-factor training - produces a single model effective across different compression levels. Our strongest model outperforms the unpooled baseline on BEIR SciFact across pool factors 1-6, implying a vector compression rate of 83% at no cost to retrieval accuracy. 

---
# Uncertainty-Aware Cross-Modal Remote Sensing Image-Text Retrieval via Evidential Learning 

**Authors**: Zhuoyue Wang, Xueqian Wang, Gang Li, Chengxi Li, Yongpan Liu, Yifang Ban  

**Link**: [PDF](https://arxiv.org/pdf/2607.06032)  

**Abstract**: In cross-modal remote sensing image-text retrieval (CMRSITR), test-time remote sensing (RS) images and textual descriptions may deviate from well-curated benchmark conditions due to sensor- and atmosphere-related image degradations and text-side RS-vocabulary heterogeneity. Under such non-ideal conditions, existing CMRSITR methods may produce unreliable retrieval results because they perform retrieval with full certainty for each query and do not distinguish the varying uncertainty across queries. To address this issue, we propose an evidential learning-based CMRSITR (ELC) method for uncertainty-aware retrieval. During the training phase of ELC, evidential learning (EDL) is employed to model the inter-modal correspondences between RS images and textual descriptions as Dirichlet distributions, from which the uncertainty of each query can be obtained. Based on the EDL outputs, uncertainty-correctness alignment learning (UCL) is introduced to align the estimated uncertainty with retrieval correctness, encouraging high uncertainty for incorrect retrieval and low uncertainty for correct retrieval. Furthermore, intra-modal relationship learning (RL) distills the intra-modal similarity structure from pretrained mentor encoders for the trainable encoders, thereby making the Dirichlet distributions modeled by EDL more discriminative. In the test phase of ELC, the estimated uncertainty is compared with a threshold determined by a fixed deferral ratio, where low-uncertainty queries are directly returned and high-uncertainty queries are refined by RS-aware test-time augmentation (RS-TTA). Experimental results demonstrate that ELC achieves competitive retrieval performance compared with state-of-the-art CMRSITR methods and provides stronger robustness under the evaluated RS-specific degradations, including sensor- and atmosphere-related image perturbations and RS-vocabulary heterogeneity. 

---
# Faithful or Findable? Evaluating LLM-Generated Metadata for RDF Dataset Search 

**Authors**: Riccardo Terrenzi, Serkan Ayvaz  

**Link**: [PDF](https://arxiv.org/pdf/2607.05970)  

**Abstract**: Dataset search depends heavily on metadata, making LLM-generated metadata a consequential form of synthetic content in retrieval systems. We study six metadata-generation settings for RDF datasets, ranging from simple rewriting to profile-grounded and agentic graph-based generation, and evaluate them jointly for retrieval effectiveness and faithfulness. Unconstrained metadata rewriting delivers the strongest retrieval gains over the original metadata, but it is also the least faithful, showing that search improvements can be driven by unsupported semantic expansion. More grounded settings substantially improve faithfulness, and profile-grounded rewriting provides the most balanced trade-off between retrieval effectiveness and grounding. These findings position synthetic metadata as a system-level IR problem in which effectiveness, provenance, and trust must be evaluated together. 

---
# CMDR: Contextual Multimodal Document Retrieval 

**Authors**: Ryota Tanaka, Taku Hasegawa, Kyosuke Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2607.05927)  

**Abstract**: Multimodal document retrieval aims to retrieve relevant pages while preserving both textual and visual content from the original document. However, existing benchmarks primarily evaluate simple lexical or semantic matching, and most methods encode pages independently. Consequently, they overlook the contextual information in the document required to resolve queries that aggregate information across multiple pages. In this paper, we introduce CMDR and CMDR-Bench, a new multimodal document retrieval task and benchmark that require modeling document context. To address this challenge, we propose CMDR-Embed, a contextual multimodal embedding framework that explicitly incorporates document context by jointly encoding multiple pages and deriving page-level embeddings from a shared contextual representation. Furthermore, we introduce CMCL, a contextual multimodal contrastive learning objective that effectively trains CMDR-Embed by balancing contextual modeling with page-level discriminability. Experiments demonstrate that CMDR-Embed significantly outperforms non-contextual embeddings, highlighting the importance of context-aware multimodal embeddings for advancing document retrieval. 

---
# Quantifying and Expanding the Theoretical Capacity of Late-Interaction Retrieval Models 

**Authors**: Julian Killingback, Varad Ingale, Hamed Zamani, Cameron Musco  

**Link**: [PDF](https://arxiv.org/pdf/2607.05803)  

**Abstract**: Late-interaction retrieval models that use the MaxSim similarity function have shown strong empirical performance, often outperforming single-vector dense and sparse retrieval models. Despite these empirical findings, little is known about the theoretical representation power of MaxSim and how it compares to other retrieval approaches. This paper shows by construction that MaxSim similarity can exactly replicate the inner product between any two non-negative k-sparse vectors with possibly infinite dimension, requiring only O(k) representation space. Moreover, there exist similarities that MaxSim can express while standard vector inner products with the same representation space cannot. Leveraging our theoretical framework, we introduce Signed MaxSim which allows late-interaction models to exactly replicate any real-valued inner product, something we prove standard MaxSim is not capable of. We also show that MaxSim can act as an aggregation of soft-OR operations and as an evaluator of logical expressions in positive Conjunctive Normal Form. Our findings show that MaxSim is at least as capable as standard vector inner products for any non-negative vectors and our extension, Signed MaxSim, is as capable for any vectors. Both similarities possess additional capabilities that inner product cannot replicate, marking one of the first theoretical justifications and quantifications of late-interaction methods. Our theoretical findings are supported empirically: on a retrieval task featuring queries with negations, Signed MaxSim improves out-of-domain performance significantly over a standard ColBERT/MaxSim baseline with nDCG@10 increasing from 0.597 to 1.000 under a vocabulary shift and from 0.008 to 0.788 on negation-only queries. 

---
# SCOReD: Student-Aware CoT Optimization for Recommendation Distillation 

**Authors**: Haz Sameen Shahgir, Yufei Li, Frank Shyu, Luke Simon, Sandeep Pandey, Xi Liu, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2607.05734)  

**Abstract**: Chain-of-thought (CoT) distillation in the recommendation domain is a necessary precursor to RL training, but raw teacher traces are ill-suited to this task. Large teachers approach the recommendation task with unusually high reasoning uncertainty, repeatedly rechecking their answers without revising them; supervised fine-tuning on such traces produces verbose students that never revise their initial guess. Furthermore, due to the novelty of the recommendation domain, the teacher's reasoning traces are highly out-of-distribution for the small student LLM.
We propose Student-Aware CoT Optimization for Recommendation Distillation (SCOReD), a CoT optimization framework tailored to recommendation that first parses each teacher trace into typed segments and uses the student LLM's attention to score the importance of each segment. Then SCOReD dynamically selects a per-segment edit (KEEP / REWRITE / FUSE / PRUNE) based on the output length and comparative log probability lift of the answer given the edit as per the student. Therefore, SCOReD prunes redundant sections of the reasoning trace while preserving information-dense sections and adapts raw teacher traces to the student's output distribution. Training on SCOReD-optimized CoTs provides a cleaner learning signal to the student model and improves over baseline SFT by 1.56% NDCG and 1.9% Recall@5, while reducing reasoning length by 27.3%. 

---
# Retrieving a Set, Not Independent Passages: Set-Level Compatibility Learning for Efficient Set Exploration 

**Authors**: Mooho Song, Jay-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.05712)  

**Abstract**: Multi-hop question answering and retrieval-augmented reasoning require selecting evidence passages that are jointly useful for answering a query. However, most retrievers still score passages independently or make locally supervised sequential decisions, which can fail when evidence usefulness depends on compatibility among passages. LLM-based set selection can model such interactions, but its computational cost limits practical use. We address this gap by formulating multi-hop retrieval as query-set compatibility scoring and propose a set-level retrieval framework. Our training objective teaches retrievers to rank complete and compatible evidence sets above incomplete, noisy alternatives, making set scoring more robust to variable-length and partially noisy contexts. We instantiate the framework with two complementary set scorers: ParaSet, a lightweight late-interaction scorer that applies self-attention over precomputed bi-encoder embeddings for fast candidate-set exploration, and SetCE, a cross-encoder-based reranker trained with the same set-level objective. Experiments on various multi-hop QA benchmarks show that set-level compatibility learning improves retrieval performance and downstream QA task performance. We further show that the proposed set-level retrievers not only outperform document-level retrievers, but also exhibit complementary retrieval characteristics: combining their outputs yields stronger performance than simply retrieving more passages from a single document-level retriever. 

---
# Prompting Beats Fine-Tuning: Generative Expected Value Scoring for Statutory Term Retrieval 

**Authors**: Alvin Wang, Jaromir Savelka  

**Link**: [PDF](https://arxiv.org/pdf/2607.05582)  

**Abstract**: Legal concepts in statutes are often expressed using vague terms, and practitioners frequently turn to case law to interpret them. We study the task of ranking case-law sentences by their usefulness for explaining a concept or target statutory term, using an established dataset of 26,959 sentences covering 42 U.S. Code concepts labeled into four explanatory-value categories. We compare two families of methods: (i) supervised fine-tuning of encoder-only models (ModernBERT) and (ii) zero-shot prompting of decoder-only models. We show that across all concepts and standard NDCG cutoffs, ModernBERT largely matches earlier BERT-family baselines. In contrast, prompting decoder-only models achieves the strongest overall effectiveness, with our best system surpassing all previously reported state-of-the-art results on this task. 

---
# Scientific Code Search at Scale: A Multi-Domain Dataset and Benchmark 

**Authors**: Nishan Pantha, Pranath Reddy Kumbam, Sajil Awale, Pushwitha Krishnappa, Muthukumaran Ramasubramanian, Nidhi Jha, Emily Foshee, Ankur Kumar, Rachel Slank, Ashkbiz Danehkar, Rahul Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2607.05443)  

**Abstract**: Scientists increasingly rely on open-source tools to support their research workflows, yet discovering relevant software among over 600 million GitHub repositories remains challenging. Existing code search benchmarks focus on general software engineering tasks and fail to capture the domain-specific vocabulary and needs of scientific computing. We present a curated corpus of 5,264 high-quality, domain-classified scientific repositories spanning five NASA Science Mission Directorate divisions -- Earth Science, Astrophysics, Planetary Science, Heliophysics, and Biological & Physical Sciences -- enriched with cleaned READMEs, extracted topics, and additional context from crawled links. Building on this corpus, we introduce two novel information retrieval benchmarks: (1) a repository search benchmark with 219 expert-curated queries designed by domain scientists, and (2) a large-scale code snippet retrieval benchmark containing 117,950 code snippets and 119,720 queries across seven programming languages. Baseline evaluations on repository search reveal significant performance variation across scientific domains. Code snippet retrieval proves equally challenging, with substantial variation driven by differing documentation practices, coding standards, and programming language conventions across scientific communities. All datasets and benchmarks are publicly released on HuggingFace to support research on scientific tool discovery. 

---
# PORTS: Preference-Optimized Retrievers for Tool Selection with Large Language Models 

**Authors**: Lorenzo Molfetta, Giacomo Frisoni, Nicolò Monaldini, Gianluca Moro  

**Link**: [PDF](https://arxiv.org/pdf/2607.05441)  

**Abstract**: Integrating external tools with Large Language Models (LLMs) has emerged as a promising paradigm for accomplishing complex tasks. Since LLMs still struggle to effectively manage large tool collections, researchers have begun exploring retrieval-based methods to pre-select the most relevant options, addressing input length and latency constraints. However, existing retrievers are often misaligned with tool-calling LLMs due to their separate training processes. This paper presents PORTS, a novel odds ratio preference optimization method for training retrievers aimed at tool selection. Using a perplexity-inspired preference signal from a frozen LLM, our approach fine-tunes a retriever to find helpful tools by optimizing the correlation between the selection probabilities and the downstream performances while jointly enforcing a contrastive semantic loss between documentation strings. The versatility of PORTS and its ability to significantly improve tool selection accuracy are demonstrated through extensive experiments on six datasets, two encoder models, and three LLMs with diverse prior knowledge. With low computational demands, our alignment process facilitates generalization to new queries and tools, proving valuable for practical applications with evolving toolsets. 

---
# Modality Relevance is not Modality Utility: Post-hoc Selective Modality Escalation for Cost-Aware Multimodal RAG 

**Authors**: Xue Li, Yiming Gai  

**Link**: [PDF](https://arxiv.org/pdf/2607.05438)  

**Abstract**: Multimodal retrieval-augmented generation (RAG) grounds a generator in evidence drawn from heterogeneous modalities -- text, tables, and images. The dominant deployment choice is binary and made before the model has tried to answer: either run a cheap text(+table) pipeline, or pay for an expensive vision-language model (VLM) over every image. Recent adaptive systems improve on this by selecting the modality or fidelity pre-retrieval, from a question-conditioned predictor of which modality will be needed. We show that this is the wrong decision point. Through an oracle headroom analysis on MultiModalQA, we find that the relevance of a modality to a question is a weak predictor of whether that modality is actually needed to answer correctly: a large fraction of questions whose gold support includes an image are nonetheless answerable from text and tables alone, and a pre-retrieval router that escalates on apparent visual relevance over-escalates substantially relative to an oracle. We propose \textbf{post-hoc selective modality escalation}: answer cheaply from text and tables, run a verifier on the (query, draft answer, evidence) tuple that localizes which modality is missing, and pay for VLM evidence only there. A calibrated value-of-escalation router then decides whether the expected accuracy gain justifies the visual cost. On MultiModalQA, our router recovers the accuracy of an always-on VLM pipeline while issuing far fewer visual calls, and closes most of the gap to the oracle escalation rate. The result extends a routing-signal hierarchy established for retrieval depth and reasoning hops to a third axis -- modality -- under a single cost-aware selective-escalation view. 

---
# DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop Retrieval-Augmented Generation 

**Authors**: Yaqi Wu, Xiaolei Guo, Chenyu Zhou, Jiaqi Huang, Xianfa Zhang, Junxu Zhang, Zhuo Yu, Zhubo Shi, Jianghao Lin, Dongdong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2607.06507)  

**Abstract**: Multi-hop retrieval-augmented generation (RAG) acquires evidence sequentially, with each new document potentially revealing missing facts, bridge entities, query defects, or sufficient support for answering. Existing methods provide useful operations such as iterative retrieval, query reformulation, evidence critique, and sufficiency judging, but typically organize them within method-specific pipelines or predefined control topologies. This leaves underexplored how to learn a shared state-conditioned policy that chooses among currently valid evidence operations. We introduce DynaKRAG, which formulates multi-hop evidence acquisition as state-conditioned control over atomic evidence operations. At each step, a validity layer constructs the executable action set, and a learned controller selects the next operation. The resulting transition updates the evidence state and may enable new operations at subsequent steps. With Qwen2.5-7B-Instruct, DynaKRAG achieves F1 scores of 0.5998 on HotpotQA, 0.5340 on 2Wiki, and 0.3061 on MuSiQue, outperforming the strongest controlled baseline on all three benchmarks. Replacing the learned controller with a uniform-valid policy reduces F1 by 3.96--5.78 points, while removing sufficiency feedback hurts all three datasets. Controlled retrieval-cap experiments further show that additional retrieval is not uniformly beneficial. Together, these results demonstrate the benefit of coordinating retrieval, diagnosis, and gap-directed acquisition under an evolving evidence state. 

---
# InfluMatch: Frontier-Quality KOL Search at 4B-Model Cost 

**Authors**: Krittanon Kaewtawee, Petmongkon Pornpichitsuwan, Natchaya Temyingyong, Nutnicha Laplamoon, Wachiravit Modecrua, Krittin Pachtrachai, Touchapon Kraisingkorn  

**Link**: [PDF](https://arxiv.org/pdf/2607.05968)  

**Abstract**: Matching influencers (KOLs) to free-form, multi-part Thai marketing criteria is today served either by keyword search over structured profiles, which misses semantic fit, or by prompting frontier LLMs over every candidate, which is accurate but slow and expensive. We present InfluMatch, a low-cost three-stage cascade -- retrieval $\rightarrow$ rerank $\rightarrow$ reason -- built entirely from small open-weight models: dense retrieval returns 50 candidates, a 4B pointwise reranker scores each by the log-probability of a single Yes token and keeps 10, and a 4B reasoner grades the shortlist per criterion on a rubric with a Thai rationale. The cascade is designed for cost: reasoning over a filtered top-10 halves token spend versus reasoning over all 50 while scoring 14 points higher. End-to-end against human relevance labels on an 11-query set with all 50 candidates labeled, the full cascade reaches 94.1% P@5, versus a retrieval-only baseline near random; it matches the frontier model Kimi-K2.6 (91.8%) while emitting ${\sim}35\times$ fewer output tokens and serving a 50-KOL query in ${\sim}20$ s on one A100. Notably, the only fine-tuning that pays off is pairwise: a SimPO-tuned reranker matches the frontier baseline's best-pick accuracy (78.0 EM), whereas fine-tuning the reasoner on pointwise per-criterion labels improves offline scores yet degrades end-to-end ranking -- an inversion we trace to the design of the absolute labeling task -- leaving the untuned base model as the strongest deployed reasoner. The result is a deployable, explainable KOL search system at a small fraction of frontier serving cost. 

---
# Inject or Navigate? Token-Efficient Retrieval for LLM Analysis of Transactional Legal Documents 

**Authors**: Mahmoud Hany, Mourad ElSheraey, Mahmoud Said, Peter Naoum  

**Link**: [PDF](https://arxiv.org/pdf/2607.05764)  

**Abstract**: Answering questions over a set of transactional legal documents is most simply done by injecting the whole corpus into the LLM's context window on every query. That baseline maximises retrieval recall, but its token footprint scales with the corpus rather than the question, and long-context degradation scales with it. We report what it took to replace full-corpus injection in a legal-document analysis system, comparing it against two structured retrieval modes over our proprietary structure-aware chunking: embedding retrieval (NAVEMBED) and LLM navigation over a compact structured index (NAVINDEX). On a 20-question benchmark with verified ground-truth answers, a position-bias-controlled, reference-anchored pairwise judge scored semantic retrieval with reranking tied with injection on 16 of 18 document-bound questions (injection preferred on 2) while attending to 17.3x fewer input tokens (a general-text-embedding (GTE) configuration reaches 29.9x at a lower tie rate); both modes were judged tied on the 2 out-of-scope controls. NAVINDEX was judged tied on all 18 at a 1.61x smaller total token footprint, a ~56x smaller answering context, and 25% lower dollar cost. We derive a closed-form caching-crossover rule: cached injection is cheaper in dollars only while the corpus stays below roughly ten times the retrieval payload. Scope and uncertainty are quantified in Section 8. 

---
# Narrative World Model: Narratology-Grounded Writer Memory for Long-Form Fiction 

**Authors**: Mohammad Saifullah, Thomas Kornmaier, Taaha Kazi, Vasu Sharma, Aditya Sanjiv Kanade, Aanand Kumar Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2607.05577)  

**Abstract**: Long-form fiction writers need memory that answers multi-hop questions about evolving story state: who knows a secret and when they learned it, whether an event preceded the narration that revealed it, whether a setup paid off, and how a relationship shifted. General-purpose retrieval and agent-memory systems represent entities and facts but not the narratological structure these questions turn on, so they surface the wrong evidence or none at all. We introduce the Narrative World Model (NWM), a writer-memory system that pairs a narratology-grounded typed temporal-state graph with query-conditioned hybrid retrieval. To measure memory rather than the answerer, we read every system through a single held-constant Opus 4.8 reader over only that system's chapter-safe evidence, on a reproducible public corpus and a validated multi-hop benchmark, and we compare against the strongest existing temporal-knowledge-graph agent-memory framework, Graphiti/Zep (Rasmussen et al., 2025). NWM substantially and significantly outperforms this baseline on multi-hop narratological QA across both corpora, and far exceeds GraphRAG and flat retrieval. The advantage is representational rather than an artifact of extraction: it survives rebuilding the baseline with NWM's own extractor, and traces to its narratology-grounded structure and query-conditioned retrieval, not to graph size or extractor quality. 

---
# Linking Hadith Narrator Identities Across Heterogeneous Arabic Biographical Databases: A Multi-Signal Entity Resolution Pipeline 

**Authors**: Taufiq Wirahman  

**Link**: [PDF](https://arxiv.org/pdf/2607.05424)  

**Abstract**: The transmission chains (sanad) of Islamic Hadith literature encode relationships among tens of thousands of historical narrators whose biographical records are dispersed across independently maintained digital databases that share no common identifier. We present a two-phase entity resolution pipeline that links narrator names from the Sanadset 650K corpus - 650,986 Hadith records from 926 books containing 185,216 unique narrator name variants - to two biographical databases: Hadithtransmitters (Hawramani; 100,915 entries) and Muslimscholars (25,247 entries). Phase 1 matches Sanadset names to Hawramani using name-only similarity (Sanadset carries no metadata), yielding 94,628 links (51.1%; HIGH 39,938 / MED 54,690). Phase 2 cross-references Hawramani against Muslimscholars via a weighted multi-signal function combining name similarity, death-year proximity, and reliability grade polarity, yielding 95,573 links (94.7% of Hawramani; HIGH 18,245 / MED 71,546 / LOW 5,782). Chaining the two phases gives Sanadset narrators transitive access to Muslimscholars data. The linked data enable construction of a 185,216-node, 814,093-edge directed transmission graph enriched with cross-source biographical metadata. The annotated link corpora and enriched graph are released as open resources. 

---
