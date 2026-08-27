# PlanSightRAG: A Visual-First Multimodal RAG for Automating Question Answering and Compliance Checking for Civil Standard Plans 

**Authors**: Nabaraj Subedi, Shuvo Dip Datta, Ahmed Abdelaty, Shivanand Venkanna Sheshappanavar  

**Link**: [PDF](https://arxiv.org/pdf/2608.26091)  

**Abstract**: Civil infrastructure compliance checking has long relied on engineers manually reading legacy 2D plans; however, OCR-based automation strips away the geometry and layout essential for interpreting these plans. We present a Visual-First Multimodal Retrieval-Augmented Generation (RAG) framework called PlanSightRAG. It indexes and reasons directly over plan imagery, integrates a ColNomic-3B multi-vector retrieval, an agentic Planner-Retriever-Auditor-Synthesizer, and MaxSim heatmaps as an evidence trail. We introduce a 4,056-pair benchmark from five state Departments of Transportation (DOT) standard plans (1,898 pages). PlanSightRAG achieves 91.47% Recall@5 on zero-shot retrieval, while on a held-out Michigan DOT corpus, it achieves 91.40%. On synthetic, parametrically-generated compliance drawings, our Qwen2.5-VL-72B pipeline reaches 100% verdict accuracy only when supplied a pre-resolved rule threshold, a controlled ceiling that a non-VLM OCR baseline already reaches at 76.4%. Finally, we demonstrate autonomous visual rule-grounding by extracting numeric limits directly from a specification corpus without any human-supplied rules. 

---
# PUMA: Post-Hoc Sparsification of Universal Multimodal Embeddings for Efficient Retrieval 

**Authors**: Matteo Attimonelli, Alessandro De Bellis, Franco Maria Nardini, Claudio Pomo, Cosimo Rulli, Rossano Venturini, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2608.25780)  

**Abstract**: Universal multimodal embedders enable retrieval across text, image, and combined queries, but their dense representations incur high memory and inference costs. Post-hoc sparsification could reduce these costs but remains underexplored for multimodal retrieval. We introduce PUMA, a sparse autoencoder recipe that maps universal multimodal embeddings to compact sparse codes without retraining the backbone: a pretraining stage preserves dense dot-product geometry, after which the sparse encoder is fine-tuned for retrieval. We evaluate on five benchmarks covering text-to-image and composed image retrieval. On Qwen3-VL-Embedding-2B, PUMA is statistically indistinguishable from or improves over dense retrieval on four of five datasets. We further identify two failure modes of post-hoc sparsification: insufficient pre-TopK support and retrieval-misaligned active support. PUMA reduces vector storage by 8-16x (FP32) and is up to 25x faster than exact dense scoring on larger candidate pools, enabling efficient multimodal retrieval. 

---
# Hamiltonian Spectral-Temporal Dissipative Dynamics for Sequential Recommendation 

**Authors**: Shuiying Liao, P. Y. Mok  

**Link**: [PDF](https://arxiv.org/pdf/2608.25755)  

**Abstract**: Sequential recommendation requires understanding how user preferences evolve over time, yet most existing models treat such evolution as a first order process where the next state depends solely on the current latent representation. Nevertheless, real user behavior often exhibits richer dynamics, including inertia, periodicity, and sudden shifts that cannot be fully captured by these first order assumptions. Motivated by these behavioral characteristics, we reconceptualize sequential recommendation through the lens of second order dynamical systems and introduce the Hamiltonian Spectral Recommender (HSR), which recasts preference evolution as a dissipative Hamiltonian system in a latent phase space of position (stable preference) and momentum (short-term tendency). The linear time-invariant structure of the governing equation admits a closed-form solution in the frequency domain. A learnable dissipation mechanism further captures natural interest decay, while a short local impulse refinement module models abrupt behavioral fluctuations commonly observed in sparse interaction logs. This design jointly accounts for global periodic patterns, inertial evolution, and localized shocks, where three phenomena that are underrepresented in existing sequential models. Extensive experiments on three benchmark datasets demonstrate that HSR consistently outperforms state-of-the-art Transformer-based and state space model (SSM)-based recommenders. 

---
# D3ER: Supporting Multi-Modal Recommendation via Disentangle and Distillation-based Dynamic Ensemble 

**Authors**: Bingnan Wang, Yi Li, Xiongxin Tang, Fanjiang Xu, Jiangmeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.25737)  

**Abstract**: Incorporating items' information shared among multiple modalities into a fused representation, multi-modal recommendation (MR) has demonstrated documented success than canonical unimodal recommendation. Although several attempts have been made to extract the discriminative information unique in each modality, existing methods suffer from a core limitation: the joint learning of modal-homogeneity discriminative information (HOI) and modal-heterogeneity discriminative information (HEI) tends to weaken their individual effectiveness. To remedy this deficiency, we propose a novel method, dubbed Disentangle and Distillation-based Dynamic Ensemble for multi-modal Recommendation (D3ER). We introduce gradient boosting into MR for the first time to formalize the optimization objective for alternately learning HOI and HEI. This design enables models dedicated to each type of information to focus on their proficient samples, thereby promoting specialized optimization. Furthermore, to mitigate the inherent high storage cost and risk of local optima in gradient boosting, we enhance our framework with knowledge distillation and a global correction regularization. Experiments on prevalent real-world datasets confirm the superiority of our proposed method on MR. 

---
# Data Citation for Large Language Models: A Challenge 

**Authors**: Gianmaria Silvello  

**Link**: [PDF](https://arxiv.org/pdf/2608.25663)  

**Abstract**: Large language models increasingly mediate access to information, and a growing body of work asks whether they cite the sources behind their outputs. That work treats citation as a verification device and applies it to textual documents. Scholarly citation serves two further functions, credit and provenance, and it applies to data as much as to text. This paper argues that data citation for large language models is an open challenge, distinct from document-level citation grounding and harder to solve. We ask how such models should cite data so that outputs stay verifiable, provenance stays traceable, and credit reaches data creators and curators. We set out three research directions. Training data attribution has to turn influence estimates into references for corpora absorbed into model parameters. Data citation at inference time has to identify datasets, subsets, and query results at the right granularity and fixity. Citing knowledge graph facts has to define what a reference to a single triple denotes and how credit propagates along provenance. Progress on all three depends on joint work across the database, information retrieval, knowledge representation, and artificial intelligence communities. 

---
# RetrievalRouter: Joint Modality and Architecture Selection for Document Retrieval 

**Authors**: Emre Kuru, Mehmet Onur Keskin, Reza Farahbakhsh, Noel Crespi  

**Link**: [PDF](https://arxiv.org/pdf/2608.25625)  

**Abstract**: Document retrieval increasingly supports high-stakes information access in finance, healthcare, and law. Modern retrieval pipelines vary both in modality (text or multimodal) and in retrieval architecture (dense or late-interaction). These choices impose a hard compromise: the most effective pipelines are too slow and expensive to run at scale, while the fastest fail to retrieve evidence from complex documents. Practitioners must therefore choose between missed evidence and unusable latency, with no principled basis for adapting that choice at the query level. We show that this compromise is unnecessary. Not every query requires the same pipeline. Across benchmarks spanning financial and scientific corpora, no static pipeline dominates. We introduce RetrievalRouter, a lightweight query-aware router that learns, from the query text alone, which retrieval pipeline best fits each query. A single tunable parameter exposes the full accuracy-latency frontier, and for every static baseline, RetrievalRouter offers an operating point that is simultaneously more accurate and faster. Against the best static baseline, RetrievalRouter is 2.5% more accurate and 12.4 times faster. Furthermore, compared with prior adaptive strategy selection methods, RetrievalRouter achieves significantly higher nDCG@5 across accuracy-oriented settings, while matching or numerically outperforming them on both nDCG@5 and latency in latency-oriented settings. Our code and data are available at this https URL. 

---
# When Stale Constraints Go Unchecked: Budgeted Verification Failures in Inherited Agent Memory 

**Authors**: Kazuki Nakayashiki  

**Link**: [PDF](https://arxiv.org/pdf/2608.25553)  

**Abstract**: An agent that inherits a consolidated memory may inherit a constraint that was true when written and has since been withdrawn by a newer authoritative record. Under a scarce verification budget, does the agent recover the withdrawal, and if not, is the error avoidable without spending more? We model supersession explicitly -- historical provenance is immutable; what changes is which record is current -- and assign by design the memory's form, the world's state (source current or superseded), and the verification policy at a fixed budget of two records: the agent's own allocation, or the same budget with one slot re-assigned to the critical provenance path or to a random record. With a constraint stated, agents inspected its provenance path in about one episode in five; when that constraint had been superseded, native allocation produced stale-consistent decisions in 77.3%, 74.7% and 74.7% of episodes across a primary run, a fresh-wording replication and a held-out domain. Re-assigning one slot to the critical path raised current-record-consistent decisions by +74.0, +72.7 and +61.3 points, positive in six of six models in each of those runs, and changed nothing when the record agreed with the memory. The held-out scenario was later found to contain a temporal inconsistency; a robustness replication with one sentence corrected, deposited externally before execution, gave +73.3 points and is reported alongside the original. The intervention uses knowledge of the critical path and is not a scheduler; it identifies that the share of stale-memory error attributable to verification allocation is close to its structural ceiling. Memory systems may need freshness or supersession signals separate from relevance. 

---
# An Event is Worth One Token: Event Tokenization for Industrial-scale LLM Recommendation 

**Authors**: Fan Xia, Zhaoheng Zheng, Iman Setayesh, Ruogu Lin, Yiqin Pan, Samarth Mittal, Wentao Bao, Vinti Pandey, Sachin Patil, Jianpeng Cheng, Jun Xiao, Zhuang Wang, Xiangjun Fan, Sri Reddy, Minghai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.25546)  

**Abstract**: LLM-based recommendation has scaled along model capacity and sequence length, yet each position encodes only text, semantic IDs, or a few categorical features, discarding rich user, item, context, and outcome signals available at each event. Under autoregressive modeling, this yields weak queries at each position and, since each position becomes context for the next, the degradation compounds across the sequence. We propose an event-centric paradigm that represents each interaction by its full temporal snapshot, and identify a new scaling dimension we term snapshot resolution: the amount of information encoded per event. To efficiently scale snapshot resolution, we introduce AMBER (Autoregressive Modeling via Bottlenecked Event Representation), which compresses each temporal snapshot into a compact Event Token, a new LLM input modality. The representation is learned end-to-end, while Event Tokens are pre-computed and cached for serving, decoupling snapshot resolution from real-time serving compute. On industrial-scale ranking and retrieval benchmarks, AMBER advances the compute-quality Pareto frontier relative to alternative recommendation paradigms. At sufficient capacity, a single unified tokenizer even outperforms dedicated per-entity tokenizers, demonstrating positive transfer across structurally different entity types. AMBER's Event Tokens also transfer across model architectures: when integrated into a heavily optimized non-LLM ranker as serving-time historical features, they yield statistically significant improvements. Further scaling Event Tokenizer capacity provides additional improvements. 

---
# TransRetrieval: Scaling Up Transformer-Based Retrieval for Industrial Recommendation 

**Authors**: Zhifei Zheng, Yunfei Liu, Bin Liu, Qiren Zhu, Hanbing Liu, Ziru Xu, Han Zhu, Jian Xu, Qi Qi, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.25528)  

**Abstract**: Applying scaling laws to recommendation retrieval is hindered by feature heterogeneity: naively stacking Transformer layers yields diminishing returns because heterogeneous fields produce severe token-norm divergence. We present TransRetrieval, a Transformer-based retrieval framework that scales with both computational budget and cross-domain data. The key enabler is (1) weighted average aggregation, which restores the homogeneous-token assumption Transformers rely on. Building on this, we introduce (2) target token compression that cuts per-candidate FLOPs by 85% while preserving cross-attention expressiveness, and (3) position-style domain embeddings that unify multiple domains at negligible additional cost, turning cross-domain data into a scaling asset. On a 40-billion-interaction industrial dataset and the public KuaiRand benchmark, scaling compute from 0.1 to 2 MFLOPs per target yields +19.3/+22.2 pt Recall@2000, confirming robust log-linear scaling. In online A/B tests, TransRetrieval lifts platform revenue by 2.53% under the same end-to-end latency constraint as the production baseline. 

---
# Query Expansion Is More Than Generation: Improving Dense Retrieval through Better Integration 

**Authors**: Siyuan Sun, Mihai Surdeanu  

**Link**: [PDF](https://arxiv.org/pdf/2608.25521)  

**Abstract**: Large language models (LLMs) can generate query expansions without task-specific training, yet the same expansions often make a frozen dense retriever worse. We identify an underexplored factor: prior work has often focused on what text is generated, while how generated text is incorporated into dense retrievers has received less systematic attention. By holding generated expansions fixed, we show that performance degradation can often be attributed to the integration method itself. We introduce AnchorQE, a training-free method that separately encodes the original query and its expansion before interpolating them. The interpolation factor is estimated using an unsupervised online strategy that operates over a small part of the unlabeled test stream. Intuitively, our strategy assigns high expansion trust only when expansions are both retrieval-strong and consistent with the original query's retrieved evidence. We show that AnchorQE improves retrieval effectiveness by up to 12.89% when compared to widely-used expansion-only or text-level concatenation baselines across TREC-DL, LoTTE, and BEIR. Further, we show that our online strategy to estimate the interpolation factor outperforms a fixed weight tuned on a development partition by up to 3.81%. 

---
# DocPC: Document-Level Visual Retrieval via Representative Page Composition 

**Authors**: Chengsong You, Junwei Zhou, Nan Du  

**Link**: [PDF](https://arxiv.org/pdf/2608.25434)  

**Abstract**: Visual document retrieval has advanced by encoding page screenshots with vision-language models, bypassing OCR pipelines. However, existing methods remain page-centric, misaligned with real-world scenarios requiring complete document retrieval. A naive page-then-document aggregation suffers from linear indexing cost and degraded retrieval when relevance spans multiple pages. We propose DocPC, a document-level visual retrieval framework based on Representative Page Composition: selecting representative pages and composing them into a single grid image for document-level indexing, reducing indexed images, vectors, and storage by 10.1x and end-to-end indexing time by roughly 7.7x. To handle multi-positive supervision prevalent at the document level, we combine multi-positive contrastive learning with sparsely scheduled listwise optimization. We also introduce DocViRe, a benchmark with multi-positive relevance annotations. DocPC-ColQwen achieves NDCG@5 of 44.09 on DocViRe, outperforming the strongest page-level baseline at 38.91 while reducing storage by 10.1x. Code is available at this https URL. Data is available at this https URL. 

---
# MOTIF: Motivation-guided Topology Inference for Cold-start Multimodal Recommendation 

**Authors**: Yurui Shi, Yuchen Miao, Ximing Hu, Zijun Wang, Chang Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.25381)  

**Abstract**: Cold-start multimodal recommendation faces three coupled challenges: (i) sparse interactions obscure user intent, (ii) cold items remain topologically isolated, and (iii) similarity-based item graphs may cause semantic drift. To address these issues, we propose MOTIF, a Motivation-guided Topology Inference framework for cold-start multimodal recommendation. MOTIF integrates Semantic Motivation Reasoning, Knowledge-enhanced Graph Reconstruction, Weighted Graph Contrastive Learning, and Semantic-Structural Alignment. It uses offline LLM reasoning to infer motivation semantics, reconstructs transferable item-item topology, and learns robust graph embeddings without injecting generated text into prediction. Experiments on three multimodal benchmarks show consistent gains over graph-based, multimodal, cold-start, and LLM-enhanced baselines, with up to 6.07% relative improvement over the strongest recent baseline. 

---
# CRAMER: Control via Request-Aware Masking for Editing Recommenders 

**Authors**: Zhiyuan Julian Su, Naihe Feng, Zhen Luther Qin, Ga Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.25370)  

**Abstract**: Sequential recommendation models, while powerful, have limited flexibility in responding to immediate user requests, making it difficult to adapt their recommendations to the user's timely interests. Unfortunately, existing user request adaptation methods often incur high computational overhead due to either 1) retraining the entire backbone network or 2) leveraging the inference ability of large language models (a.k.a. prompt engineering), limiting their applicability in large-scale recommendation services. This paper presents Control via Request-Aware Masking for Editing Recommenders (CRAMER), a framework that takes users' natural-language requests to immediately change sequential recommendation models' behavior. Specifically, inspired by the model control theory, CRAMER treats user requests as control signals to modulate frozen backbone parameters through masking, achieving instant adaptation to diverse requests while avoiding costly retraining. Experiments on multiple large-scale benchmark datasets show that CRAMER outperforms four state-of-the-art request-aware baselines across multiple recommendation metrics while achieving minimal overhead. Moreover, the proposed framework exhibits enhanced controllability and cross-domain adaptability, establishing a new paradigm for request-aware sequential recommendation. 

---
# Rank-Deviation Quality: A Distance-Aware Metric for Multi-Answer Retrieval and Ranking Evaluation 

**Authors**: Xiaokun Zhou, Alessandro Moschitti, Danielle Class  

**Link**: [PDF](https://arxiv.org/pdf/2608.25318)  

**Abstract**: We introduce Rank-Deviation Quality (RDQ), an evaluation metric for retrieval and ranking systems that adapts to queries with varying numbers of reference items, from a single correct answer to many valid results. RDQ scores a candidate ranking against an ordered reference list (ORL): each retrieved reference item contributes its output-position weight multiplied by a rank-deviation penalty, and items outside the ORL receive zero credit. Application-specific parameters control tolerance to misordering. Larger values emphasize retrieving valid reference items, whereas smaller values place more weight on matching their reference order. The output-position weights can reflect visibility in the application's interface, such as a vertical list or a carousel. Unlike metrics that require absolute relevance grades, RDQ operates on ordinal rankings, which annotators can produce through pairwise or listwise judgments. Unlike rank-correlation measures such as Kendall's tau, RDQ accounts for both which items are returned and how they are ordered. On a 5,000-query point-of-interest (POI) dataset with 12 systems, RDQ has the highest median empirical power@100 among the 13 evaluated metric configurations. It reaches mean tau >= 0.8 agreement with its own full-query ordering at 200 queries; RBP(0.9), the strongest tested non-RDQ configuration, reaches the same threshold at 250. On TREC Deep Learning benchmarks, where NDCG uses native graded labels and RDQ uses ordinal tiers derived from them, RDQ reaches comparable median power at n=25, while NDCG is higher at n=100. 

---
# The "Curse of Knowledge" in LLM Query Simulation: Concept Provenance for Tracing Answer-Side Intrusion 

**Authors**: Chenglong Ma, Xinye Wanyan, Danula Hettiachchi, Ziqi Xu, Jeffrey Chan  

**Link**: [PDF](https://arxiv.org/pdf/2608.25245)  

**Abstract**: LLM-generated search queries are widely used to augment IR evaluation, yet they may contain concepts that presuppose answer-side document knowledge, violating the information-access boundary of pre-search users. Existing validation metrics, including overlap, diversity, and effectiveness, cannot distinguish rare human-tail variation from candidate answer-side intrusion. We introduce concept provenance, a framework that assigns query concepts to backstory-supported, human-central, human-tail, and candidate answer-side zones, operationalizing a boundary that retrieval metrics alone cannot detect. Applying concept provenance to 77,004 queries across 100 UQV100 topics, 8 LLMs, and 5 prompt conditions with two extraction pipelines, we obtain a cross-pipeline token-HCIR Spearman rho of 1.0 over five condition means. Candidate answer-side concepts constitute 7.40 percent of non-generic concepts and appear in 97 of 100 topics, with topic explaining approximately 67 percent of variance. Human validation yields 68.2 percent relaxed precision, revealing two mechanisms: knowledge intrusion at 45.5 percent and deployment intrusion at 45.0 percent. Diagnostic probes show disproportionate localized retrieval effects, with deletion effect size d = -0.47 compared with d = -0.34 for random deletion, but these concepts explain less than 2 percent of aggregate evaluation variance. Concept provenance therefore serves as a boundary-compliance diagnostic rather than an evaluation-shift predictor. Under the tested conditions, no prompt condition eliminates intrusion; post-generation concept-provenance selection achieves 99 percent elimination. 

---
# SWIM: Step-Wise Integrated Measure for Session-supervised List Evaluation in Generative Re-ranking 

**Authors**: Yuanhao Pu, Chenghao Zhang, Chao Feng, Xunyong Yang, Xiang Li, Yongqi Liu, Defu Lian, Kaiqiao Zhan, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2608.25104)  

**Abstract**: Modern industrial recommender systems have increasingly adopted the Generator-Evaluator (G-E) framework for the re-ranking stage. Within this paradigm, the generator produces candidate item lists from a pool filtered by upstream retrieval and ranking modules, while the evaluator scores these lists and selects the highest-scoring one for final exposure per request. However, on sequential platforms (e.g., short-video apps), users consume items continuously, ignoring artificial list boundaries. Conventional evaluators score lists by aggregating point-wise values, implicitly assuming exposure independence. This fails to capture critical session-level dynamics, such as contextual dependencies, user continuation, and diminishing marginal utility from repetitive content.
To bridge this gap, we propose SWIM (Step-Wise Integrated Measure), a list-level evaluator that models user behaviors as a finite-horizon prefix session-level survival process. SWIM estimates the prefix-conditioned contribution of the current list to the session-level objective by factorizing it into a recursive survival distribution and reached-position conditional rewards. Leveraging a causally-masked Transformer, SWIM efficiently estimates continuation probabilities and utilities in parallel, satisfying strict industrial latency constraints. Extensive experiments demonstrate that SWIM significantly outperforms baselines in listwise reranking tasks, yielding substantial improvements in overall recommendation engagement. 

---
# VoiceMem: Streaming Dual-Brain Memory for Real-Time Interaction 

**Authors**: Zhifei Xie, Jiaqi Lang, Ze An, Yifan Zhao, Dongchao Yang, Kai Li, Ziyang Ma, Mingbao Lin, Chunyan Miao, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2608.26005)  

**Abstract**: Conversational systems, such as duplex speech language models (SLMs), still lack a streaming, accurate, and empathetic memory system as their soul. We introduce VoiceMem, a simple memory architecture with a parallel informational left brain, an emotional right brain, and streaming memory I/O mechanisms. We further build a complete pipeline for memory-aware SLM training, long-horizon evaluation, and decoupled deployment with interchangeable memory backends. Experiments and real-world deployment show three advantages: i) Accuracy: under top-5 retrieval, the left brain outperforms classical systems such as Mem0 at top-200 by nearly 30 points; ii) Emotional & Personal: the right brain, with short- and long-horizon affective attribution and dual-node persona modeling, achieves state-of-the-art performance across three persona benchmarks and improves the aggregate score by 4.29 points over the previous best system; and iii) Real-Time & Cheap: VoiceMem completes retrieval in 134 ms, well within standard VAD latency, adding no extra conversational delay while maintaining high accuracy and low cost. These results show that VoiceMem provides a practical memory foundation for real-time, personalized, and emotionally aware speech interaction. 

---
# Query-Side Attacks on GNN-Based KGQA: Tracing Failures from Entity Linking to Answer Generation 

**Authors**: Pankaj Kumar, Subhankar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2608.25922)  

**Abstract**: GNN-based Knowledge Graph Question Answering (KGQA) pipelines process queries through four discrete stages: entity linking, subgraph retrieval, GNN reasoning, and answer generation. Standard robustness evaluations conflate stage-level failures into a single end-to-end metric, obscuring both the source of brittleness and the appropriate mitigation target. We ask which stage fails, and why, when the pipeline is subjected to adversarial perturbations on the input question. We introduce a stage-isolation protocol with two answer-preserving adversarial perturbations verified against the knowledge graph: Compositional Restructuring (CR) and Relation Synonym Swap (RS) target distinct stages while leaving entity seeds intact. Evaluated across ComplexWebQuestions and WebQSP, the results run counter to prevailing assumptions: the GNN reasoning stage retains near-baseline accuracy when the subgraph is intact, while subgraph construction accounts for over 99\% of the end-to-end collapse under CR, occurring even when the gold answer is present in 74\% of retrieved subgraphs. This exposes a fundamental distinction between answer presence and answer reachability that end-to-end metrics cannot detect, and places the mitigation target firmly at the subgraph construction stage rather than the reasoning model. Perturbed datasets and evaluation infrastructure are released at this https URL . 

---
# Pointing the Way, Hiding the Destination: Practical Private Dense Retrieval at Scale 

**Authors**: Peichun Hua, Danyang Chen, Junan Zhang, Haifeng Sun, Jingyu Wang, Diwen Xue, Mingyu Li, Yunming Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.25735)  

**Abstract**: Hosted retrieval-augmented generation (RAG) and semantic search allow users to query valuable provider-held corpora, raising two competing demands: to hide each query and chosen result, yet reveal only the documents that the user is authorized to receive. Existing cryptographic approaches either make this costly by processing the entire corpus for every query, or sacrifice quality for efficiency by scanning a few clusters. We repurpose learned deep hashing as a private filter: a randomized binary code points the provider to a short candidate list, while encrypted reranking and oblivious key transfer protect the precise query and final selection. This shortlist short-circuits full-corpus cryptographic search without sacrificing retrieval quality: with 200-500 candidates, it closely matches full-corpus retrieval across five zero-shot corpora spanning 25K to 5.4M documents. On the full 2.68M-passage NQ corpus over a 10-Gbps link, our protocol only adds 0.73 seconds, or 10 percent, to a 128-token Qwen3-32B RAG pipeline. The released code satisfies directional metric differential privacy (DP) and substantially reduces embedding-inversion and property-inference leakage, demonstrating that a carefully learned shortlist can make private dense retrieval both accurate and practical. 

---
# DCEO: Direct Causal Effect Optimization for Long-Term User Value Modeling in E-commerce Search 

**Authors**: Junzhao Zhang, Tao Zhang, Liren Yu, Feiyi Dong, Zhixuan Zhang, Dan Ou, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.25635)  

**Abstract**: Industrial e-commerce search systems ultimately aim to optimize the user-level long-term objective, such as n-day cumulative purchases or gross merchandise value (GMV) per user. However, such objectives are defined at the user level, whereas search ranking is based on item-level scores within each request. Existing methods typically bridge this granularity gap through manually designed multi-objective fusion, where predictions of multiple item-level objectives, such as clicks, carts, purchases, and transaction value, are combined into a ranking score that serves as a proxy for the ultimate objective. Such hand-crafted fusion schemes rely on a small set of manually tuned weights, limiting fine-grained personalization and leading to suboptimal alignment with the ultimate objective. In this paper, we propose DCEO (Direct Causal Effect Optimization), a data-driven framework for learning item-level proxy scores that are better aligned with the ultimate objective. We first aggregate the item-level proxy scores into a user-level proxy metric and quantify its alignment with the ultimate objective using a relative causal effect. We then develop an actor-critic framework, where the critic estimates the ultimate objective for a given user-level proxy metric, and the actor dynamically generates context-dependent fusion weights over multiple objectives to construct the item-level proxy scores and is trained to directly optimize the relative causal effect. Extensive offline experiments and analyses demonstrate the effectiveness and interpretability of DCEO. In addition, DCEO has been deployed in a large-scale industrial e-commerce search system, outperforming the conventional GMV proxy by 0.36% in GMV in a 41-day online A/B test. 

---
# A Storage-Retrieval Gap in Parametric Knowledge Graph Memory 

**Authors**: Martino M. L. Pulici, Cuong Xuan Chu, Evgeny Kharlamov, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2608.25489)  

**Abstract**: Graph retrieval-augmented generation places retrieved subgraphs into the model's context window at query time, paying a recurring token cost and exposing source data on every call. We study an alternative: compiling a knowledge graph offline into a bank of LoRA adapters, one per entity, that serve as a parametric knowledge layer queried by injecting weights rather than text, at zero query-time context cost. On the MetaQA dataset, we find that subgraph-trained adapters encode context-free factual knowledge that generalizes to unseen questions: on single-valued relations the adapter gains $+0.243$ exact-match score over a base model that is nearly blind closed-book ($0.007$), and only the correct adapter recovers this knowledge (an oracle gap of $+0.283$ over the base model). However, the stored knowledge is not recoverable by similarity: given a query with no subgraph, embedding-based and weight-space geometry retrieval both perform at chance, because a semantically neighbouring entity's adapter does not contain the answer - knowledge is stored locally and does not transfer. Weight geometry correlates with subgraph semantics ($\rho = +0.329$) but not with functional retrievability. We quantify the byte and context-token costs against graph retrieval-augmented generation and discuss deployment implications. Our results establish that parametric knowledge graph memory is feasible for storing knowledge, and identify selecting and composing the right adapters by a mechanism other than semantic similarity as the central open problem - motivating a learned, query-conditioned composition mechanism. 

---
# ReliableRAG: Combating Misinformation in Retrieval-Augmented Generation via Reliability-Guided Reasoning Chains 

**Authors**: Jinpu Jiang, Xuan Wu, Wenhao Song, Bo Yang, You Zhou, Hongwei Ge, Heow Pueh Lee, Yanchun Liang, Chunguo Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.25487)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful architecture for Question Answering (QA) by integrating external information into Large Language Models (LLMs). However, false, inaccurate, and misleading information in news and social media poses a serious challenge to real-world RAG systems, especially in multi-hop QA, where complex multi-step reasoning can be misled by even a single deceptive misinformation segment in the retrieved documents. Existing approaches mainly rely on implicit alignment or explicit regulation, but their limited ability to assess fine-grained information reliability makes them vulnerable to deceptive misinformation that is semantically relevant to the question yet factually incorrect, leading to erroneous answers. To address this limitation, we propose ReliableRAG, which, to the best of our knowledge, is the first reliability-driven framework that mitigates deceptive misinformation in multi-hop QA through fine-grained evaluation of individual triples. ReliableRAG first extracts information segments from source documents and represents them as structured triples. It then quantifies triple reliability by combining query-triple semantic relevance with triple credibility, retaining only the top-$K$ reliable and non-redundant triples. Based on these refined triples, ReliableRAG autoregressively constructs robust reasoning chains to consolidate trustworthy evidence and filter deceptive misinformation, producing accurate answers faithful to reliable information. Experiments on three multi-hop QA datasets show that ReliableRAG outperforms existing methods, substantially improving the factual reliability and robustness of RAG systems under deceptive misinformation injection. 

---
# Q&A or Document-Based? The Effects of Interface Type on How Screen Reader Users Access Interconnected Documents 

**Authors**: Colleen F. Cipriano, Yichun Zhao, Miguel A. Nacenta, Kotaro Hara, Jaylee Soh  

**Link**: [PDF](https://arxiv.org/pdf/2608.25382)  

**Abstract**: Blind and low-vision (BLV) users are increasingly engaging with large language model (LLM) interfaces to access documents, but it is unclear how such systems support or hinder their ability to build interconnected knowledge. To examine this gap, we compared a Question-Answer Interface (QAI) that supports open-ended conversational inquiry, with a Document Interface (DI) based mostly on traditional structured text document navigation. We recruited 16 BLV screen reader users where they used both interfaces to explore two fictional worlds. Data from interaction logs, concept maps, decision-based tasks, and semi-structured interviews provide comparative insights into how interface design supports knowledge construction. Findings show that participants visited more distinct documents with the DI and formed larger and more correct mental models with the DI than with the QAI. They were also more able to apply knowledge they had gained. Simultaneously, many still preferred the QAI and often estimated that they had explored more, formed better mental models and applied their models better when acquiring the information with the QAI, despite this not being the case. Our analysis suggests possible interface design reasons for these differences and highlights some of the risks introduced by using question-answer interfaces to access information spaces. 

---
# Less can be More: Relieving RAG Bottlenecks via Evidence Frontloading and Pressure-Adaptive Budgeting 

**Authors**: Weibin Cai, Reza Zafarani  

**Link**: [PDF](https://arxiv.org/pdf/2608.25115)  

**Abstract**: Existing methods for improving Retrieval-Augmented Generation (RAG) efficiency mainly optimize downstream LLM generation, such as context compression or serving optimization. However, RAG is an end-to-end system, and its bottleneck can shift between upstream reranking and downstream generation under different serving loads and reranking this http URL this paper, we first empirically characterize this shifting-bottleneck behavior and show that upstream reranking can become the dominant bottleneck under high query rates or large reranking budgets. Reducing the reranking budget can relieve this bottleneck, but it may also drop supporting evidence and degrade recall. To address this problem, we propose \textbf{\textsf{PACE}} (\textbf{P}rioritized \textbf{A}daptive \textbf{C}overage of \textbf{E}vidence), a training-free framework that combines \textit{evidence frontloading} with \textit{pressure-adaptive budgeting}. \textsf{PACE} first reorders candidates by marginal evidence coverage, prioritizing documents that are query-relevant, complementary, and useful for forming multi-hop evidence chains. We show that this objective is monotone submodular, giving greedy selection a $(1-1/e)$ approximation guarantee. \textsf{PACE} then dynamically adjusts the reranking budget according to the relative pressure of the reranker and the LLM. Experiments on three multi-hop QA datasets and online serving simulations show that \textsf{PACE} improves evidence recall, reduces p95 latency under ranking-heavy workloads. More importantly, the two components together reveal that \textit{less can be more}: an evidence-dense top-ranked candidates enable higher final recall with fewer reranked documents. 

---
# Retrieve, Match, Escalate: Accurate and Scalable Product Linking with VLM-Distilled Cross-Encoders and Agentic VLMs 

**Authors**: Jian Wang, Steven Xu, Sanjyot Thete, Maryam Barouti, Tom Tang, Elaine Wu, Charu Sareen, Kyle MacDonald  

**Link**: [PDF](https://arxiv.org/pdf/2608.25037)  

**Abstract**: Product linking, the entity-resolution task of mapping merchant product records to canonical catalog products, consolidates fragmented listings so downstream search, recommendation, and advertising see one clean entry per product. At marketplace scale, billions of noisy, multi-category records must be resolved against tens of millions of canonical products, where scoring every candidate with a single model is either too weak for the hard cases or too costly for the easy ones. We present a production retrieve-then-match cascade that spends computation in proportion to difficulty: retrieval surfaces plausible matches, a lightweight text cross-encoder auto-resolves the high-confidence majority, and an agentic multimodal vision-language model settles the ambiguous remainder by inspecting product images and issuing web searches for evidence that is in neither record. The cross-encoder is distilled from millions of dual-VLM-consensus labels, retiring human annotation from the training set, and is calibrated to auto-accept links at a 98% precision bar validated against a smaller operator-certified audit. The agent is a self-hosted open-weight model that reaches a closed frontier VLM's precision at a four-point recall cost (88% versus 92%) for roughly one-seventh the per-pair cost, with no fine-tuning. Per-pair cost spans nearly five orders of magnitude from the cheap cross-encoder to the frontier VLM, so escalating only the hard tail to the agent raises end-to-end link coverage from the cheap stage's 68% to 77%. 

---
# A Pathway for Assessing Grey Literature: Leveraging AI to Extract Conference Metadata and Organiser Information from Calls for Papers 

**Authors**: Angelo Salatino, Francesco Osborne, Alexis Vizcaino, Aliaksandr Birukou, Enrico Motta  

**Link**: [PDF](https://arxiv.org/pdf/2608.24926)  

**Abstract**: Despite its importance, grey literature, including Calls for Papers (CfPs), remains largely overlooked in Metascience and Scientometric analysis due to its unstructured, highly heterogeneous format, which traditional tools struggle to process at scale. However, Large Language Models now offer a pivotal opportunity to devise innovative tools for systematically harvesting and processing such data. In this paper, we introduce COCI, an AI-based framework that automates the extraction of granular, structured metadata from raw CfP text. COCI employs a multi-stage pipeline for entity extraction, followed by author disambiguation against OpenAlex and semantic mapping of topics and conference series. This process identifies key data points, including conference editions, geographic locations, and comprehensive lists of organisers, along with their specific roles and affiliations. By structuring this previously inaccessible information, COCI establishes a foundation for the systematic analysis of grey literature, enabling new research opportunities and shifting the scholarly focus towards non-publisher-based events. 

---
