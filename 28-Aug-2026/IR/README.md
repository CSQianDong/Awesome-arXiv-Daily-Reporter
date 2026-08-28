# misi: a Metric Inverted Sample Index 

**Authors**: Edgar Chavez  

**Link**: [PDF](https://arxiv.org/pdf/2608.27422)  

**Abstract**: We present misi, an inverted index for approximate nearest-neighbor search over general metric spaces whose vocabulary is a random sample of the database, of size proportional to $n$. Each object is represented by its $k_b$ nearest sample points, found by a pluggable inner index over the sample; queries are answered by an idf-weighted shared-neighbor vote followed by exact verification of $C$ candidates. The construction generalizes the NAPP index from a constant number of pivots to a linear-size vocabulary, which keeps posting lists at constant expected length $\rho = k_b/\alpha$ as $n$ grows and turns the index into a combinator: any high-recall index on $\alpha n$ points yields an index on $n$ points, for any metric. A probabilistic model gives a recall guarantee -- $k_b$ logarithmic in $n$ over the overlap gap suffices, with a verification budget the index itself estimates -- and a matching limit: the vote cannot resolve overlap differences below order $1/\sqrt{k_b}$. The design's strengths are structural: construction is $n$ independent searches -- embarrassingly parallel, deterministic, $5{,}250$ s for $10^8$ vectors on 64 cores, $3.7\times$ faster than a matched-recall graph build -- it streams under an enforced 3 GiB cap, and the portable artifact serves $10^8$ vectors from NVMe within an enforced 8 GB budget, below the working floor of the SSD-graph baseline. Its cost is query-time work: saturated graph baselines answer $6$-$16\times$ faster in RAM, and the verification budget for 0.99 recall grows as $n^{0.30}$. All results carry seeds, saturation sweeps and full configurations, are generated from run manifests, and include measured negative results. The intended applications weight construction cost, determinism, memory footprint, or black-box metrics over peak throughput: frequently rebuilt corpora, batch similarity workloads, constrained-memory serving. 

---
# Scaling Graph Neural Networks for Friend Recommendation: Multi-Hash User Embeddings and Temporal Neighbor Sampling 

**Authors**: Maksim Utushkin, Andrei Ovsiannikov, Alexander D'yakonov  

**Link**: [PDF](https://arxiv.org/pdf/2608.27413)  

**Abstract**: Friend recommendation is inherently graph-structured: the relevance of a potential connection depends on multi-hop social context rather than user attributes alone. However, deploying message-passing GNNs on a production-scale social graph with hundreds of millions of users and tens of billions of edges requires addressing numerous modeling and systems challenges. We present a scalable end-to-end GNN ranking system for production social graphs, focusing on two design choices that are critical in this setting: multi-hash ID embeddings and temporal neighbor sampling. Multi-hash embeddings are common for high-cardinality features, but industrial GNN systems typically either ignore trainable IDs or accept full embedding tables, exceeding 200 GB for our graph. We integrate multi-hash as the primary node representation, reducing the ID-embedding table size by more than 98 percent while preserving ranking quality. Temporal neighbor sampling is well understood in principle, but existing implementations scan full adjacency lists, which is a non-starter for users with tens of thousands of friends. We implement timestamp-sorted CSR storage with binary search, reducing the per-node temporal sampling cost from $O(deg(v) + k)$ to $O(\log(deg(v)) + k)$. Beyond these components, we show that this combination scales and yields measurable production impact. On a graph with 194M users and 28B edges, offline ablations isolate each design choice's contribution. In an online A/B test, our system increases friend additions from recommendations by 16 percent and unique friend adders by 11.5 percent over a strong production baseline. We release our framework for distributed training and inference on large temporal graphs. 

---
# Stageboost: Recommending Signals Based on Counterfactual Estimation 

**Authors**: Darpan Singhal, Matan Mandelbrod, Tal Franji, Manasa Kolla, Vipul Gaba, Yuri Brovman  

**Link**: [PDF](https://arxiv.org/pdf/2608.27366)  

**Abstract**: Signals are short textual or visual snippets displayed on the eBay View-Item (VI) page, providing additional, contextual information for users about the viewed item. The aim of displaying these signals is to facilitate intelligent purchase and to incentivize engagement. In this paper, we present a 2 stage xgboost based model that optimally populates the VI page with signals. This approach has shown a 0.08% lift in overall GMB (Gross Merchandise Bought) and 0.58% increase in Parts and Accessories GMB, primarily due to increase in conversion of high average price items in online experimentation. 

---
# Astar: Learning to Propose Evolution Directions for Self-Evolving Industrial AI Systems 

**Authors**: Jinxin Hu, Hao Deng, Haibo Xing, Lingyu Mu, Muyu Zou, Weiqin Yang, Sirui Chen, Bohao Wang, Zhezheng Hao, Hao Zhang, Zulong Chen, Shizhun Wang, Yu Zhang, Xiaoyi Zeng, Jiawei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.27287)  

**Abstract**: Modern AI systems advance through continuous iteration: a loop of proposing evolution directions, implementing code, training, and evaluation. While the latter three stages are increasingly automated, the starting point --- proposing effective evolution directions --- remains a critical bottleneck that still relies heavily on senior experts. In this work, we explore whether AI can take over this role. We find that general-purpose LLMs, even the advanced GPT-5.5, offer only generic and misaligned suggestions: the required expertise is accumulated through experience rather than explicitly codified, and thus hard to inject directly.
To this end, we propose Astar, a training-based approach that learns a specialized evolution-guiding model from the abundant iteration histories of industrial systems. Realizing this idea, however, raises four challenges: sparse supervision, noisy data, a vast direction space, and prohibitively expensive verification. We address them along two fronts. On the data side, we design a pipeline that turns noisy historical commits into a large, clean evolutionary corpus via pairwise sample expansion and noise filtering. On the model side, we train the model through mid-training, SFT, and RL, guiding evolution direction generation with hierarchical hints and using the reward model in RL as a fast surrogate evaluator.
Astar has been deployed in Alibaba's Lazada advertising system for evolution direction proposal. Astar-8B achieves a single-proposal success rate of 0.6786 in real-execution evaluation, far exceeding human experts (0.3229) and the strongest general-purpose LLM (0.3071). More importantly, Astar closes the loop and enables fully automatic iteration: it guided 20 consecutive iterations over two weeks, improving offline Hitrate@200 by 23.6%, while an online A/B test yielded relative lifts of 4.86% in GMV and 1.82% in advertising revenue. 

---
# ProRetrieval: Learning to Orchestrate Hybrid Search via Executable Program Synthesis 

**Authors**: Chengsong You, Zhen Sun, Yunhai Hu, Junwei Zhou, Xiaoyu Cao, Binyu Li, Ziyan Zhao, Weiyao Wang, Liren Lu, Zhijie Ye, Yumo Cao, Yitao Long, Yiwei Xu, Qiyi Jiang, Xuanyi Fu, Yufan Chen, Yilun Li, Rongkang Xiong, Yiran Zou, Nan Du  

**Link**: [PDF](https://arxiv.org/pdf/2608.27017)  

**Abstract**: Real-world retrieval often composes structured constraints with semantic intents over text and images through arbitrary Boolean logic. Existing hybrid pipelines such as reciprocal rank fusion or self-querying retrievers admit only a fixed form of composition, while recent reinforcement-learning retrievers train the language model as a query generator for a single backend, leaving the orchestration of heterogeneous retrieval paths outside its action space. We propose ProRetrieval, which recasts the language model as a retrieval orchestrator: given a natural-language query, it synthesizes an executable program in a hybrid DSL interleaving SQL operators over structured fields with vector-retrieval primitives over text and images, with SQL itself providing the logical algebra that fuses heterogeneous candidate sets. We train Qwen3-4B with GRPO and DAPO under a hierarchical four-term reward, and evaluate on two new benchmarks built from Amazon products and Enron email. Our 4B model surpasses GPT-5.5 (Hit@1 0.81 vs. 0.69 on e-commerce; 0.91 vs. 0.86 on email) and Claude Opus 4.7 and a comprehensive suite of retrieval, LLM-augmented, structured-query, and graph-based baselines. Code: this https URL data: this https URL. 

---
# Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval 

**Authors**: Ante Kapetanovic, Tomislav Duricic, Dionizije Fa, Andro Mercep, Emanuel Lacic  

**Link**: [PDF](https://arxiv.org/pdf/2608.27006)  

**Abstract**: Conversational recommender systems based on large language models (LLMs) are usually evaluated on static, pre-indexed item collections, yet e-commerce catalogues change continuously as products are added or removed, repriced, and restocked. We present a merchant-agnostic, multi-turn conversational shopping assistant that operates over such live catalogues. Its central component is a self-refreshing retriever that ingests a merchant product feed, enriches the records, and synchronizes them into a vector index. On each run, per-item hashes identify which products are new, changed, deleted, or unchanged, so only the delta is processed rather than rebuilding the whole catalogue. A controller-based dialogue layer consumes this index, using an LLM only for intent classification and preference elicitation while retrieval, reranking, and diversity selection run as dedicated functions. Our demonstration is a WhatsApp shopping assistant in which catalogue changes reach the recommendations after the next successful sync. A live chatbot, documentation, and a recorded walkthrough are available at this https URL. 

---
# Topology-Masked Unified Backbone for Joint Feature Interaction and Multi-Domain Sequence Modeling 

**Authors**: Zhihao Zhu, Dezheng Han, Jikang Xia, Shuaishuai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.27005)  

**Abstract**: Large-scale post-click conversion rate (CVR) prediction requires jointly modeling heterogeneous feature interactions and dependencies over multi-domain user behavior sequences. Existing industrial ranking models usually handle these two aspects with separate modules. Recent unified architectures attempt to incorporate them into a single framework, but such unification often relies on coordination between modules and does not fully organize all information sources within the same interaction space. To address this problem, we propose MaskRec, a topology-masked unified token interaction architecture for feature interaction and multi-domain sequence modeling. MaskRec transforms heterogeneous features, multi-domain behavior sequences, and contextual signals into unified token representations, and further introduces learnable global memory tokens and domain-level memory tokens as information aggregation nodes. Based on this unified token space, MaskRec designs a structured attention mask, TopoMask, which selectively enables or blocks attention connections according to the structural differences and modeling requirements of different information sources. In this way, heterogeneous feature interaction and multi-domain sequence modeling are performed within the same topology-constrained attention process. In addition, MaskRec incorporates a dual-path interactive query generation module to inject candidate-conditioned user--item interaction signals before the unified backbone. Experiments on the Tencent Advertising Algorithm Competition dataset show that MaskRec achieves stable improvements over the official baseline, validating the effectiveness of the proposed unified framework for industrial CVR prediction. 

---
# When Memory Takes Gradients: Collaborative Vector Memory for Agentic Recommender Systems 

**Authors**: Hanchong Chen, Xing Tang, Lingjie Li, Xiongfeng Shan, Xiuqiang He  

**Link**: [PDF](https://arxiv.org/pdf/2608.26895)  

**Abstract**: Agentic recommender systems ground each decision of a large language model (LLM) in a persistent memory of the user, and in existing agents that memory is text: a narrative written and maintained by further LLM calls. Text limits this memory in two ways. It is updated one rewrite at a time, so exploiting the full interaction history is prohibitively expensive; and collaborative evidence, graded similarity over an entire catalog, does not survive translation into sentences. We propose CoVeMem (Collaborative Vector Memory), which vectorizes the collaborative core of the agent's memory. Frozen LightGCN user and item states form the memory bank; at each decision, the candidate set itself retrieves the most relevant historical states, which enter the LLM's context as soft tokens alongside a light textual profile. Contrastive alignment to item-semantic anchors, followed by listwise co-training with masked candidates, teaches the model to read these states and to rank through them; a pointwise yes/no readout scores each candidate. Across four instruction-grounded recommendation benchmarks, CoVeMem matches or exceeds the strongest collaborative text-memory agent on 19 of 20 metric cells while requiring zero additional LLM calls for memory maintenance beyond the shared static profile, against per-interaction calls for text memory. The memory now takes gradients: the full interaction history, out of reach for text, becomes available as training data for what the agent remembers and for how it reads what it remembers. 

---
# STREAM: An Objective-Driven and Uncertainty-Aware Framework for Industrial Energy Data Acquisition 

**Authors**: Zhipeng Ma, Bo Nørregaard Jørgensen, Zheng Grace Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.26754)  

**Abstract**: Industrial energy management requires datasets that connect energy use with equipment states, production batches, material flows, and process conditions. However, conventional acquisition workflows commonly emphasize connectivity and storage without verifying whether accessible signals satisfy the requirements of a defined energy-performance assessment. This paper presents STREAM, an objective-driven and uncertainty-aware framework comprising Specification of Objectives, Technical Requirements, Resource Mapping, Extraction from Sources, Archival Metadata, and Migration to Database. STREAM is the central workflow: objective-to-data traceability is its end-to-end output, while measurement, temporal, contextual, and processing uncertainty are assessed across all six stages. Compared with the original conceptual STREAM sequence, this paper adds stage-level artifacts, minimum-evidence gates, source-suitability rules, a metadata template, an uncertainty rubric, and case-specific traceability matrices. The framework is validated through two industrial batch-process cases: induction-furnace melting in a foundry and cheese-powder drying using SCADA and production-order data. The results demonstrate that data accessibility is not equivalent to analytical suitability and show how STREAM supports transparent decisions about immediate data use, analytical restrictions, and prioritized infrastructure improvements. 

---
# Beyond a Single Story: Meta-Reviewing Sparse and Incomplete User-generated Contents for Recommendation 

**Authors**: Hongren Wang, Tianjun Wei, Yingpeng Du, Jie Zhang, Yin-Leng Theng  

**Link**: [PDF](https://arxiv.org/pdf/2608.26728)  

**Abstract**: Data sparsity remains a long-standing challenge in recommender systems, and it becomes more severe for methods relying on user-generated content (UGC) such as textual reviews, which capture fine-grained preferences but require more user efforts to produce. As a result, UGC exhibits (1) missing reviews, where interactions lack any review, and (2) incomplete reviews, where available reviews cover only a subset of relevant attributes. Existing approaches often overlook these UGC-specific issues, leading to degraded accuracy. Motivated by meta-review in academic peer review, we propose MOSAIC (Meta-review On Sparse And Incomplete user-generated Content), which constructs a meta-review for each target user by aggregating attribute-sentiment evidence from neighbor users' reviews. A multi-gate mixture-of-experts (MMoE) architecture jointly optimizes rating prediction and meta-review attribute-sentiment prediction, while an attention module personalizes the aggregated meta-review signals to each target user, yielding both refined rating predictions and attribute-level explanations. Experiments on four real-world datasets demonstrate that MOSAIC consistently outperforms state-of-the-art baselines in both recommendation accuracy and explanation quality, mitigating UGC sparsity and incompleteness while delivering consistent gains for users with limited interaction history. 

---
# BLANC: Discovering Patent White Space via Changes in Normalized Pointwise Mutual Information Between Multi-View Clusters 

**Authors**: Shuichi Miyazawa, Kensuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2608.26685)  

**Abstract**: Identifying white space --- the unexplored but potentially valuable regions of a patent landscape --- is essential for strategic R&D planning, yet existing methods rely on manual patent mapping or apply single-view clustering without quantitative gap detection. We propose BLANC (Blank Landscape Analysis through NPMI Conditioning), a three-phase pipeline combining (1) multi-view neural topic modeling along three semantic dimensions (application/use, novelty, inventive step); (2) Normalized Pointwise Mutual Information (NPMI) to quantify cross-dimensional cluster association; and (3) conditional detection that flags combinations whose NPMI drops when the corpus is filtered by a user-specified keyword. The drop is captured by a new metric, $\Delta$NPMI, which identifies combinations "established globally, unexplored locally." Because white space has no ground truth, we evaluate BLANC on two public USPTO corpora --- machine learning/AI (5,417 patents, CPC G06N) and glass compositions (1,982 patents, CPC C03C) --- by artificially depleting known technology combinations and testing recovery. When three-quarters of a target pair's documents are removed, BLANC recovers 34.1% (ML/AI) and 27.3% (glass) of the depleted combinations, whereas size-matched removals not aimed at them (random documents, or those of a different established combination) essentially never do: the target is never recovered in 191 decoy trials. Collapsing the three semantic views into one recovers nothing, while prior co-occurrence measures also flag the target under random removal, offering no specificity. In a proprietary case (302 float glass / glass-ceramics patents), the keyword "fluorine" reveals a fluorine surface treatment $\times$ warpage suppression candidate ($\Delta$NPMI up to 0.48) that experts had independently identified. 

---
# When Does Supervised Fine-Tuning Reduce Instruction Sensitivity? 

**Authors**: Jaekeol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2608.26661)  

**Abstract**: Large language models can exhibit substantial performance variation across alternative formulations of the same task instruction, yet it remains unclear how conventional task-specific supervised fine-tuning (SFT) changes this instruction sensitivity. We study this question by evaluating fixed model checkpoints under multiple paraphrased instructions and defining instruction sensitivity as the standard deviation of task performance across them. We conduct a controlled scale analysis with Qwen3 models at 1.7B, 4B, and 8B on MS MARCO, together with targeted cross-family checks using Mistral-7B and Gemma-2-9B. Before SFT, instruction sensitivity decreases sharply with Qwen3 model scale. At 1.7B and 4B, SFT consistently reduces sensitivity across training instructions, with reductions of approximately 54--71%. At 8B, individual sensitivity changes are not statistically distinguishable from zero, but paired contrasts between training instructions are statistically reliable under query-level bootstrap analysis and have consistent directions across all three random seeds. Gemma-2-9B shows the same directional training-instruction contrast as Qwen3-8B, whereas Mistral-7B does not, suggesting that the strength of this effect also varies across models. Experiments on ESCI-English further show that free-generation and likelihood-based forced-choice evaluation can yield qualitatively different robustness conclusions even when valid-label generation is nearly perfect and average task performance is similar. Overall, SFT does not uniformly reduce instruction sensitivity: its robustness effect depends on the adaptation setting, while measured sensitivity can additionally depend on the prediction and scoring protocol. 

---
# hoBIT: A Profile-Aware Retrieval-Augmented Chatbot for University Academic Advising 

**Authors**: Yoonseo Kim, Seongmin Lee, Joongheon Kim, SeongKu Kang  

**Link**: [PDF](https://arxiv.org/pdf/2608.26604)  

**Abstract**: In university academic advising, identical questions can require different answers depending on a student's department, admission cohort, and degree program, causing profile-blind retrievers to surface plausible but inapplicable evidence. We present proFILL, a method for transforming hoBIT, our college's current rule-based advising chatbot, into a profile-aware retrieval-augmented generation (RAG) system. Rather than requiring a complete user profile upfront, proFILL progressively acquires only the profile attributes needed for each query, guided by both the query intent and the initially retrieved evidence, and uses them to condition retrieval over a profile-aware index. Extensive experiments and a human preference study show that proFILL outperforms diverse RAG baselines, is preferred by target users, and remains effective with open-weight models for cost-effective on-premise deployment. 

---
# Preference Flow Matching with Spectral Factorization for Micro-video Recommendation 

**Authors**: Xinxin Dong, Haokai Ma, Fei Hu, YuZe Zheng, Bin Wu, Yonghui Yang, Xiaodong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.26579)  

**Abstract**: Micro-video recommendation aims to infer user preferences from historical interactions and multimodal video content, thereby identifying the next video of interest. However, prevailing methods compress frame sequences into a single holistic representation, entangling the stable visual semantics and the evolving dynamics that jointly shape user preferences. Meanwhile, diffusion- and flow matching-based recommenders condition their generation process solely on coarse behavioral context, leaving its internal temporal structure outside preference formation. We therefore propose PrismRec, a Preference Flow Matching framework with Spectral Factorization for Micro-video Recommendation. Analogous to a prism that disperses white light into its constituent spectrum, PrismRec devises Spectral Semantic Factorization (SSF) to derive complementary static semantic and dynamic factors from frame-level representations via a prior-guided learnable frequency mask in the temporal frequency domain. Then, it proposes Context-Calibrated Preference Matching (CPM) to weigh them with each user's specific sensitivity and inject the calibrated context as a structured condition to steer the matching trajectory toward the target representation, making video content as an intrinsic driver of preference formation rather than auxiliary side information. Experiments on four datasets from two platforms show that PrismRec surpasses the SOTA baseline by up to 22.65%, with the lowest inference cost and peak memory among the compared methods. 

---
# Assessing the Downstream Utility of Evidence-Aware Retrieval in RAG 

**Authors**: Utshab Kumar Ghosh, Debayan Mukhopadhyay, Shubham Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2608.26379)  

**Abstract**: Retrieval evaluation for retrieval-augmented generation (RAG) is increasingly designed around whether retrieved passages contain evidence that can support generation, rather than topical relevance alone. We study whether this closer alignment with downstream evidence needs also makes retrieval evaluation more useful for the decisions built from it.
Across five retrieval benchmarks and an end-to-end TREC RAG 2025 setting, we examine an answer-support signal in four roles: comparing retrievers, guiding retrieval training and system selection, predicting downstream answer quality, and filtering the evidence supplied to a generator. The signal changes retrieval rankings, but its downstream value is not uniform. It does not reliably improve retriever training; the benefit of using it for system selection depends on how the generator is instructed to use the retrieved evidence; and retrieval scores based on it do not robustly predict answer quality on unseen topics. In a direct evidence intervention, human annotators confirm that filtering preferentially preserves passages containing useful answer evidence, yet different answer evaluators reach different conclusions about whether the resulting answers improve.
These results show that making retrieval evaluation more closely reflect the evidence needed for generation does not by itself make every downstream use of that evaluation more reliable. RAG evaluation methods should therefore be assessed with respect to the particular comparisons, decisions, and conclusions they are intended to support. 

---
# RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature 

**Authors**: Maayan Sharon, Tom Hope  

**Link**: [PDF](https://arxiv.org/pdf/2608.27394)  

**Abstract**: Retrieved scientific literature can serve as inspiration for both human and AI scientists. Inspiration can take different forms: prior work may directly suggest how to address a problem, or surface directions at different levels of abstraction - zooming out to a more general view or zooming in to a concrete realization. We introduce RATIO (Retrieval Across Typed Ideation Operations), a large-scale benchmark in which relevance is defined by three operations which we name ideation moves: Address retrieves potential approaches for stated problems, Broaden retrieves more general formulations, and Specify retrieves concrete instantiations. RATIO is constructed from millions of full-text scientific papers across CS literature via a general recipe that extends discourse-marker distant supervision - previously used only for classification - to corpus-scale retrieval, combined with extensive LLM and human vetting. Experiments show that operation-specific fine-tuning substantially boosts retrievers but leaves much room for further improvements. RATIO provides a scalable training and evaluation framework for retrieval components that support literature-grounded ideation, opening up new research avenues on scientific inspiration retrieval. 

---
# CorporateBench: Large-Scale Q&A Benchmarking with Temporal Knowledge Bases 

**Authors**: Sil Hamilton, Albert Yu Sun, Oscar J. Romero, Carl-Leander Henneking, David Mimno, Bishan Yang, Igor Labutov  

**Link**: [PDF](https://arxiv.org/pdf/2608.27391)  

**Abstract**: LLMs are increasingly able to answer complex questions about enterprise-scale document collections. But evaluation is hard: companies don't want to share internal communications, and synthetic datasets have been overly simple. We present CorporateBench (CB), a human-validated multi-task Q&A benchmark whose scale approaches the conditions LLMs encounter in corporate communication networks, with evaluation corpora surpassing 230,000 documents. CB evaluates LLMs across two dimensions (information extraction and knowledge base querying) through four synthetically generated firms ranging from 12 to 10,000 employees. Each corpus is sampled from a temporally evolving knowledge base describing a consistent world, guaranteeing cross-document logical consistency even across hundreds of thousands of documents. We evaluate five LLMs on CB, revealing increasingly poor performance as input size approaches realistic scales. CB provides LLM developers a metric for corporate communication reasoning, filling a crucial gap in the benchmarking ecosystem. 

---
# Equal Ranking Quality, Different Decisions: Training Order-Consistent LLM Scorers 

**Authors**: Markus Frohmann, Mahdiyar Alavi, Elizabeth Lingg, Navid Rekabsaz  

**Link**: [PDF](https://arxiv.org/pdf/2608.26762)  

**Abstract**: Rerankers, reward models and multi-document QA scorers score candidate documents or responses in one LLM prompt, so each score depends on their order. Such scorers are selected on ranking quality, but their scores determine a decision: what a score threshold retains, a reader answers, or a preference model selects. However, equal ranking quality does not imply equal decisions: on passage reranking, five trained scorers within 0.010 nDCG@10 retain sets that overlap by only 0.66-0.84 when reordered. A published reranker takes the highest retained-set F1 in our comparison and still overlaps by only 0.667. No prompt-time change we test removes that order dependence: the only one that gains ranking quality leaves all three decisions unchanged. Order-consistency SFT (OC-SFT) attenuates it in the weights, training a candidate's score not to depend on the order. It holds ranking quality and leads every decision-stability measure among trained scorers on all three tasks: it flips the reader's answer on 0.125 of permutation pairs against 0.149-0.164 for three other objectives that target order. It is more stable than order-averaged distillation on 12 base models, and one OC-SFT permutation retains sets that overlap more than ten averaged off-the-shelf permutations. A comparison should therefore report what a threshold retains and a reader answers, not ranking quality alone. Code is available at this https URL. 

---
# PailitaoGR: Latent Think-with-Images for Generative Image Retrieval 

**Authors**: Xiaomeng Fan, Yueran Liu, Shengyu Zhou, Chenghan Fu, Wanxian Guan, Feng Li, Chuan Yu, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.26658)  

**Abstract**: Generative retrieval has demonstrated strong performance by directly generating product semantic identifiers (SIDs).
Extending this paradigm to image search, however, is nontrivial because real-world query images contain diverse information, including the search target, useful auxiliary evidence, and irrelevant visual content.
This requires the model to identify and focus on the search target while selectively utilizing auxiliary evidence. In this paper, we propose \textbf{PailitaoGR}, a \emph{Latent Think-with-Images} method for generative image retrieval, which internalizes target-focused perception and selective auxiliary-evidence utilization into a the generative retrieval model, enabling \textit{Zooming without Cropping} and \textit{Reading without OCR}. Specifically, we design a target-focused perception mechanism that identifies and enhances visual tokens of the search target, consisting of a target Enhancer and a learning strategy based on on-policy distillation and attention guidance loss, enabling the model to focus on search-target regions. We also design a selective auxiliary-evidence utilization mechanism that identifies and enhances visual tokens of auxiliary evidence, including an auxiliary enhancer and an in-capacity incremental contrastive distillation strategy, enabling the model to exploit auxiliary evidence. We construct training and validation sets sampled from real-world online image-search logs. Experiments show that our method outperforms existing baselines by an average of 13.8\%, validating its effectiveness. 

---
# Case2Flow: Bridging Patient Cases and Guideline Flowcharts through Multimodal Retrieval 

**Authors**: Jiale Wei, Yufan Chen, Alexander Jaus, Zdravko Marinov, Julian Friedrich, Simon Reiß, Jens Kleesiek, Rainer Stiefelhagen  

**Link**: [PDF](https://arxiv.org/pdf/2608.26414)  

**Abstract**: Medical guidelines encode rich, evidence-based decision logic, yet the specific decision artifact a clinician needs is hard to locate within a guideline, let alone across guidelines covering plausible diseases and treatments. While guideline passages have supported end-to-end question answering, flowcharts remain largely underused in decision support despite their ability to encode actionable clinical pathways. We therefore introduce Case2Flow, a task designed to retrieve the most relevant guideline flowchart for a given patient case from a collection of guideline documents. To support it, we construct FlowAtlas, a curated corpus of 202 flowcharts extracted from 2,080 medical guidelines, together with a pipeline that synthesises 1,911 aligned case-flowchart pairs. Our evaluation of multimodal retrieval methods reveals systematic failure modes, including overreliance on keywords and spurious token-patch matches induced by uninformative background regions in flowcharts. Motivated by this, we propose CRISP, a training-free scoring method that sharpens late-interaction retrieval by suppressing uninformative patches, discounting ambiguous token matches, and incorporating bidirectional query-image alignment. CRISP improves Recall@1 by up to 18.71 percentage points, while a blinded physician assessment on published case narratives provides preliminary feasibility evidence beyond synthetic queries. 

---
# A Reranker for Orchestrating Heterogeneous Speech and Text Retrievers 

**Authors**: Inho Kim, Sumyeong Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2608.26194)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have attracted significant interest for their ability to mitigate hallucinations in Large Language Models (LLMs). Although knowledge databases for RAG are increasingly diversifying to include various modalities such as speech and text, research on handling such multi-modal database scenarios remains limited. In this paper, we propose STeReO (Speech and Text Reranking Orchestrator), a reranker based on speech and text retrievers that aggregates disparate modality databases. To address the lack of specialized training data, we first curate a dataset comprising queries, mixed-modality evidence, and their corresponding relevance ranks. We then train the reranker and evaluate its effectiveness in both single-modality and mixed-modality scenarios. Our results demonstrate that the proposed algorithm excels at selecting the most relevant evidence, thereby significantly improving downstream question-answering performance. 

---
# Leveraging Large Language Models for Systematic Literature Review of Disease Spread Models 

**Authors**: Orhan Yagizer Cinar, Timur Emre Ozkose, Emma Von Hoene, Amira Roess, Taylor Anderson, Hamdi Kavak  

**Link**: [PDF](https://arxiv.org/pdf/2608.26150)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have created new opportunities to streamline and potentially automate many research processes, including systematic literature reviews (SLRs). This study reports an LLM pipeline development for extracting model-relevant information from 536 peer-reviewed agent-based modeling papers. We compare the results with those of a human-conducted SLR. Our results show paper-level accuracies of approximately 77.95% for GPT-4.1 and 81.67% for GPT-5.0. Field-level accuracy ranges from 32.40% to 100.00%, with more complex or subjective fields performing less reliably. Importantly, we find that agreement between LLMs is a potential indicator of output quality: low agreement may signal hallucinations, whereas high agreement combined with low accuracy may point to noise or errors in the human dataset. Overall, our study provides practical insights into prompt development and highlights both the potential and limitations of using LLMs for full-scale SLRs in the modeling and simulation domain. 

---
# LLMs for Academic Workflows: An Evaluation of Literature Reviews Generated with Short and Long Context Windows of LLMs 

**Authors**: Muhammad Ali Chaudhry, Xinyuan Hao, Haifa Alwahaby  

**Link**: [PDF](https://arxiv.org/pdf/2608.26145)  

**Abstract**: Our research focuses on evaluating literature reviews generated in short and long context settings of large language models (LLMs) to investigate the impact of context window on the quality of AI-generated literature reviews and the role of AI in supporting literature review writing. Twenty AI-generated literature reviews based on research sources from Semantic Scholar and Arxiv were evaluated by two researchers across 15 dimensions. Our findings reveal that AI-generated literature reviews require human oversight to meet academic publishing standards. As context windows increase, LLMs can incorporate broader information and maintain coherence across longer inputs, but they also exacerbate issues such as content repetition, omission of critical work, and a tendency towards descriptiveness over synthesis. Our work shows that AI-generated reviews can provide foundational overviews, but their output must be critically evaluated and refined by domain experts. Future research should consider integrating other LLMs and fine-tuned models in different domains with hybrid approaches that combine human expertise with AI capabilities to address the limitations identified in this study. 

---
# Agents Don't Paginate: First-Chunk Selection for LLM Tool Responses 

**Authors**: Tatiana Petrova, Andrei Mazniak, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2608.26130)  

**Abstract**: Coding agents built on large language models (LLMs), such as Claude Code, Cursor, OpenAI Codex, GitHub Copilot, and Aider, receive tool responses that routinely exceed the agent's per-turn token budget. The standard remedy, pagination, is available in every protocol that produced these responses; yet across the corpus of session logs from a public Model Context Protocol middleware we observed no agent-initiated requests for a second chunk. The first chunk is what the agent reads, so we ask how often the gold item (the one the agent needs) is placed first in it: the precision-at-1 rate $p_1$.
In a controlled offline benchmark we treat first-chunk selection as a 0/1 knapsack and compare six value functions on 500 SWE-bench Verified tasks, then test whether $p_1$ matters with a single-turn file-localisation probe on five language models (4,800 LLM calls; not an end-to-end resolve-rate test). Two pre-registered hypotheses did not hold and are our main findings. The central one is negative: raising $p_1$ does not systematically raise downstream accuracy. Per-model deltas stay under three percentage points (p.p.), are not consistently signed, and no model is significant; the agent recovers the gold from anywhere in the chunk, so what reaches its answer is first-chunk inclusion, not the gold's rank within it. The second: adding four file-metadata signals to a keyword scorer hurts $p_1$ by 4.8 p.p. (paired significance test, $p = 0.001$).
A parameter-free keyword scorer does raise $p_1$, from a 24.2% baseline to 35.0% (+10.8 p.p., far beyond chance; $p = 3.9 \times 10^{-8}$), and to 35.8% with a fallback to the tool's native ordering when no keyword matches. But by our central finding this is a rank-1 gain, and rank-1 is the part that does not reach the agent's answer: downstream accuracy does not move. 

---
