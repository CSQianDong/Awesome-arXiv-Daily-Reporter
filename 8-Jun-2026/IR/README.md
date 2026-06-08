# Bradley-Terry Rankings for Recommender Systems Across Dataset Taxonomies 

**Authors**: Ekaterina Grishina, Stepan Kuznetsov, Askar Tsyganov, Ilya Ivanov, Daria Korovaitceva, Margarita Rusanova, Uliana Parkina, Alexander Derevyagin, Evgeny Frolov, Sergey Samsonov, Anton Lysenko  

**Link**: [PDF](https://arxiv.org/pdf/2606.07492)  

**Abstract**: The ranking of recommendation algorithms is a challenging problem since model performance is sensitive to dataset characteristics such as sparsity, sequential structure, and scale. This drives a demand for a proper methodology for fair comparison between algorithms. Naive aggregation of performance metrics (e.g., averaging NDCG over benchmarks) can yield misleading rankings, undermining practical selection. To address this problem, we introduce a novel, data-driven ranking methodology based on Bradley-Terry (BT) model. We demonstrate that the obtained ranking depends on key dataset statistics. Additionally, we propose a novel metric for evaluating ranking consistency and demonstrate robustness of our ranking to incomplete data. Finally, we introduce a dataset-specific methodology for ranking algorithms on unseen datasets without running the models, relying on extensions of the Bradley-Terry framework, including BT trees and BT models with covariates. 

---
# PaperFlow: Profiling, Recommending, and Adapting Across Daily Paper Streams 

**Authors**: Fuqiang Wang, Song Tan, Zheng Guo, Jiaohao Fu, Xinglong Xu, Bihui Yu, Jie Dong, Zheng Sun, Siyuan Li, Jingxuan Wei, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07454)  

**Abstract**: Scientific paper recommendation is typically evaluated as static ranking over a fixed candidate set, yet real scientific reading unfolds as a daily, longitudinal process in which interests shift and feedback accumulates. We introduce PaperFlow, a framework that organizes it into three coupled stages: Profiling, which constructs and maintains a structured, inspectable scholarly profile from heterogeneous cold-start evidence; Recommending, which ranks each date-specific paper stream through multi-signal aggregation under a fixed display budget; and Adapting, which updates user state from semantically distinct feedback signals and models interest drift across days. We further define a longitudinal user-day benchmark that fixes users, dates, candidate pools, visible inputs, and hidden simulated relevance labels under a shared temporal information boundary. The benchmark contains 24 simulated research users, 50 daily paper streams, 1,200 user-day episodes, 20,727 unique papers, and 497,448 episode-paper records. We additionally specify a blind human-evaluation protocol to validate alignment between automatic metrics and expert judgments. Experiments against five scientific recommendation baselines show that PaperFlow achieves the strongest oracle-based ranking, the highest behavioral alignment with simulated reading selections, and the best blind human-evaluation score. 

---
# Gated Bidirectional Linear Attention for Generative Retrieval 

**Authors**: Artem Matveev, Vladislav Tytskiy, Sergei Makeev, Sergei Liamaev  

**Link**: [PDF](https://arxiv.org/pdf/2606.07317)  

**Abstract**: In recommender systems, generative retrieval typically uses an encoder-decoder setup: an encoder processes a user interaction history, and an autoregressive decoder then generates recommended items. In large-scale streaming services, active users accumulate very long histories over time. As histories grow, the encoder becomes a major latency bottleneck because softmax attention scales quadratically with sequence length. In our experiments, using bidirectional attention in the encoder substantially improves quality. However, most sub-quadratic attention methods focus on causal attention.
We propose Gated Bidirectional Linear Attention (GBLA), a linear-time bidirectional attention layer that extends kernelized linear attention with three lightweight components: local causal mixing (Conv1D), sequence-level key gating for soft forgetting, and a gated RMSNorm output. On a large-scale Yandex Music dataset, a hybrid encoder that interleaves self-attention (SA) and GBLA in a 1:2 ratio (one SA block followed by two GBLA blocks) matches bidirectional self-attention quality. On H100 GPUs, GBLA reaches up to an $8.2\times$ single-layer speedup at a history length of 32768, compared to FlashAttention-v3. Finally, we show that the same hybrid design generalizes beyond our proprietary setting, consistently preserving self-attention retrieval quality on public Amazon benchmarks. 

---
# Constrained Dominant Sets for Multimodal Document Question Answering 

**Authors**: Ambuj Mehrish, Sebatiano Vascon  

**Link**: [PDF](https://arxiv.org/pdf/2606.07252)  

**Abstract**: Long multimodal document question answering is limited by which evidence reaches the reader, rather than by the quantity retrieved. In lengthy documents, findings often recur across figures, captions, and introductory sentences, causing similarity based retrievers in modern multimodal retrieval-augmented generation (RAG) systems to allocate resources to near-duplicates while overlooking complementary evidence. This work introduces a retriever that selects evidence as a Constrained Dominant Set (CDS) on a query-augmented affinity graph, offering three advantages that similarity ranking does not. First, the query is encoded as a hard structural constraint, ensuring that every selected element is directly connected to the question through the cluster anchor. Second, the relevance-redundancy balance is determined automatically by a spectral bound, eliminating the need for manually tuned trade offs required by diversity-aware selectors. Third, the selection process achieves a global equilibrium via replicator dynamics, thereby avoiding the distortions introduced by greedy heuristics. The method is inherently graph-based and does not require training. Using a Qwen3-VL-32B reader, CDS establishes a new state of the art on VisDoMBench ($66.99$ average) and improves over the no-retrieval baseline by $37.1$ points on VisDoMBench and $4.8$ on MMLongBench-Doc. 

---
# FLOWREADER: Min-Cost Flow Optimization for Multi-Modal Long Document Q&A 

**Authors**: Ambuj Mehrish, Sebatiano Vascon  

**Link**: [PDF](https://arxiv.org/pdf/2606.07235)  

**Abstract**: Long, multimodal documents force retrieval-augmented systems to assemble answers from evidence fragmented across text, tables, and slides broken across cells in a long table, spread over multiple slides, or split between a figure and its discussion. Top-$k$ chunk retrieval treats each fragment independently and cannot represent how evidence connects. We introduce FLOWREADER, which reframes evidence assembly as a min-cost flow problem on a multimodal node graph: a single scoring vector $h$ controls source selection (via MMR), sink selection (via a length-aware answerability proxy), and the costs and capacities of every edge. The optimal flow is decomposed into candidate evidence paths, a compact non-redundant subset is selected by entropy-regularized replicator dynamics, and parallel VLM workers under a dual-process gate produce the answer with a single System-2 refinement pass triggered when answer consistency is low or the routed flow is strained. On VisDoMBench, FLOWREADER is best on the two subsets dominated by fragmented evidence PaperTab ($58.40$, $+1.30$ over G^{2}-Reader) and SlideVQA ($72.93$, $+0.62$) and competitive on SPIQA, FetaTab, and SciGraphQA. Macro-averaged across all five subsets, FLOWREADER ($65.47$) is within $0.74$ of the strongest baseline (G^{2}-Reader, $66.21$). Overall, these results show that min-cost flow performs well on fragmented multimodal evidence, where top-$k$ retrieval fails. It also provides a unified way to control scoring, routing, selection, and adaptive compute together. 

---
# HKVM-RAG: Key-Value-Separated Hypergraph Evidence Organization for Multi-Hop RAG 

**Authors**: Mingyu Zhang, Ying Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.07218)  

**Abstract**: Multi-hop RAG poses a data-engineering problem beyond passage matching: under fixed retrieval budgets, a system must organize retrieved text into evidence units that expose answer chains. Dense retrievers score passages independently, while graph-based memories make associations explicit but often rely on pairwise or entity-centered keys that fragment multi-hop evidence. We present HKVM-RAG, a key-value-separated evidence-organization layer. It assembles answer-path hyperedges from cached passage-level LLM evidence tuples and uses them as retrieval keys, while retaining passage text as answer values. To isolate key-space design, our fixed-substrate protocol holds the tuple cache, candidate passages, reader, and evaluation budget constant across pairwise graph and hypergraph variants. Weighted hypergraph key-value retrieval improves over KG-PPR by +3.426 F1 on 2WikiMultiHopQA and +3.592 F1 on MuSiQue; HotpotQA shows that higher structured support coverage need not yield standalone answer-F1 gains. We therefore study WHG-KV as an evidence-control signal rather than a dense-retrieval replacement. Oracle and train-to-dev analyses identify support selection as repairable, and a dense-aware controller combines frozen ColBERTv2 and HKVM rank/score features using out-of-fold HKVM predictions. It reaches 88.846, 65.073, and 85.810 F1 on the three benchmarks, improving over ColBERTv2 by +11.084, +6.763, and +5.966 F1. Source-level ablations show that matched non-WHG structured signals do not match the WHG-KV gains. These results provide bounded evidence that key-value-separated hypergraph organization can serve as a reusable evidence-control mechanism for multi-hop RAG. 

---
# RISE: A Rust Library for Inverted Index Search Engines 

**Authors**: Angelo Savino, Rossano Venturini  

**Link**: [PDF](https://arxiv.org/pdf/2606.07187)  

**Abstract**: Inverted indexes are a crucial data structure for efficient information retrieval in large text corpora. They enable fast full-text search by mapping each term to the documents in which it appears, on top of which efficient algorithms quickly retrieve the documents relevant to a user query. We present RISE, a novel inverted index library implemented in Rust, designed to deliver high performance and efficiency for information retrieval tasks. RISE leverages Rust's safety and performance to provide a robust solution for building and querying inverted indexes, while offering accessible extensibility through its expressive trait system. While developing RISE, we revisited the inverted-index literature, thereby reproducing numerous prior works using this new test bench. We evaluated RISE against existing libraries, demonstrating competitive query performance across various datasets and workloads, with speedups of up to 2x over the current state of the art. Our results indicate that RISE is a promising tool for researchers and practitioners in the field of information retrieval. 

---
# Beyond Matching: Category-Guided Latent Intent Reasoning for Generative Retrieval in E-Commerce 

**Authors**: Fuwei Zhang, Xiaoyu Liu, Jiajie Jin, Jiale Mao, Wei Chen, Dongbo Xi, Yifan Yang, Peng Yan, Zichao Hao, Zhao Zhang, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2606.07075)  

**Abstract**: Generative retrieval offers a new paradigm for e-commerce search by mapping user queries directly to product Semantic Identifiers (SIDs). However, e-commerce queries are often short, noisy, attribute-heavy, and associated with multiple category-consistent products, creating a substantial representation gap between natural-language shopping intent and artificially constructed item SIDs. Explicit Chain-of-Thought (CoT) reasoning can help bridge this gap, but its extra generation cost is difficult to reconcile with the low-latency requirements of online e-commerce systems. To address this challenge, we propose CaLIR (Category-guided Latent Intent Reasoning), a category-guided latent intent reasoning framework for e-commerce generative retrieval. Rather than generating explicit textual rationales, CaLIR learns continuous latent intent states before SID decoding and uses product category hierarchies as a natural scaffold for coarse-to-fine intent reasoning. Specifically, we introduce hierarchical semantic reasoning to align latent states with category-level shopping intent, and query-wise reasoning enhancement to model diverse intent paths under multi-positive queries. CaLIR further combines a query-specific dynamic prefix trie, assembled from pre-indexed category-level tries, with reasoning-aware constrained decoding. Experiments on multilingual e-commerce search datasets show that CaLIR achieves a better balance between retrieval effectiveness and inference efficiency than existing methods, while also demonstrating transferability and robustness across induced hierarchies and different generative backbones. 

---
# Decision-Theoretic Stopping Rules for Document Screening 

**Authors**: Aaron H.A. Fletcher, Mark Stevenson  

**Link**: [PDF](https://arxiv.org/pdf/2606.07071)  

**Abstract**: Deciding when to stop reviewing the results of a search is a common problem with multiple applications. Existing stopping rules developed within Technology-Assisted Review (TAR) aim to achieve a pre-specified recall target and do not take into account the reason for examining the results, potentially leading to sub-optimal recommendations. This paper applies decision theory to the problem and uses it to derive three practical stopping policies based on the Expected Value of Perfect Information. The approach is applied to two professional search tasks: patent examining and systematic reviewing. Experiments on CLEF-IP and medical systematic review datasets show that the proposed approach generally produces more appropriate stopping decisions than existing methods, as demonstrated by higher net utility under the evaluated cost and payoff settings. 

---
# Meaning in Order, Order in Meaning: Semantic R-precision for Keyphrase Evaluation 

**Authors**: Shamira Venturini, Steffen Kinkel  

**Link**: [PDF](https://arxiv.org/pdf/2606.07057)  

**Abstract**: Evaluating the quality of automatically generated keyphrases remains a complex challenge. Traditional metrics either rely on exact lexical matching or consider semantic similarity while ignoring prediction ranking, both of which misalign with how humans judge informativeness and relevance. We introduce Semantic R-Precision (SemR-p), a novel evaluation metric that integrates semantic similarity into the rank-aware R-Precision framework. Designed from a human-centric perspective and inspired by Information Retrieval metrics, SemR-p rewards semantically relevant keyphrases that appear early in the output list. We conducted extensive analyses to assess its semantic sensitivity, ranking awareness, and discriminative power across models and datasets. The results suggest that SemR-p offers a complementary lens for evaluating keyphrase predictions, helping to better reflect user-centred notions of relevance alongside traditional lexical and semantic matching metrics. 

---
# SSRLive: Live Streaming Recommendation with Dynamic Semantic ID 

**Authors**: Teng Shi, Zhaoheng Li, Yuanhang Qu, Yi Liu, Lixiang Lai, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.06970)  

**Abstract**: Live streaming has emerged as one of the fastest-growing forms of online media, enabling instant content broadcasting and real-time engagement between users and streamers. Despite the effectiveness of existing recommendation algorithms in this domain, they often suffer from limited utilization of computational resources, with low FLOPs that hinder further performance enhancement. Generative recommendation techniques, which have gained traction in various industrial tasks, offer a promising avenue for improving live streaming recommendations. However, directly applying generative methods to live streaming is non-trivial due to two major challenges: (1) static semantic IDs (SIDs) cannot reflect the rapidly changing nature of live room content; and (2) generative pipelines generally do not incorporate user--streamer interaction signals (e.g., likes, orders), which are critical for modeling user intent toward both the streamer and showcased products. To address these challenges, we introduce SSRLive: Dynamic Semantic ID-guided Streaming Recommendation for Live platforms. The proposed framework integrates a generative module and a discriminative module in a unified architecture. The generative component employs an encoder-decoder design to produce both static and dynamic SIDs, enabling timely representation of live room content while leveraging multimodal information. The discriminative component refines task-specific representations by combining SIDs with user features, augments them with user-streamer interaction data, and performs multi-task predictions. Online A/B tests in real-world deployment demonstrate tangible benefits: watch time (+3.38%), GMV (+0.72%), follower growth (+3.12%), and interaction volume (+2.92%). These improvements highlight the effectiveness and business value of SSRLive, which is now fully deployed, serving hundreds of millions of active users. 

---
# DREAM: Dynamic Refinement of Early Assignment Mappings 

**Authors**: Liwei Guan, Huanjie Wang, Hongwei Zhang, Linxun Chen, Zhaojie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.06947)  

**Abstract**: Generative recommendation advances item retrieval by reformulating it as autoregressive generation of Semantic IDs (SIDs), compact token sequences that encode item semantics. While SIDs offer a strong semantic prior, current SID-based methods assign each item a single static identifier through offline tokenization before sufficient user feedback is observed. For cold-start items, this one-shot commitment produces poorly discriminative codes, generating misaligned paths that remain unrefined because the associated tokens are rarely sampled during training. We identify this early static commitment, not model capacity, as the fundamental cold-start bottleneck in SID-based generative recommendation. To overcome this bottleneck and bridge the disjoint objectives of tokenization and generation, we propose DREAM (Dynamic Refinement of Early Assignment Mappings), a three-stage framework that resolves this flaw through progressive refinement. First, an intent-aware tokenizer rebuilds the SID space through counterfactual contrastive learning, generating a diverse pool of behavior-aligned candidates per cold-start item. Second, the frozen recommendation backbone serves as an evaluator, selecting the most reliable candidate based on multi-context user support without retraining. Third, a dynamic beam mechanism maintains multiple weighted SID hypotheses throughout training and inference, preventing premature collapse to a single assignment. Extensive experiments on three Amazon benchmarks show that DREAM substantially outperforms state-of-the-art generative and sequential baselines on cold-start metrics. 

---
# Towards Retrieving Interaction Spaces for Agentic Search 

**Authors**: Shengyao Zhuang, Yuansheng Ni, Hengxin Fun, Jimmy Lin, Xueguang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.06880)  

**Abstract**: Retrieval for search agents is still inherited from non-agentic information retrieval: a retriever ranks the corpus and the agent reads a small set of returned documents. Recent direct corpus interaction (DCI) work shows that agents can instead interact with the raw corpus through shell tools such as grep and file reads. But unbounded interaction does not scale: every broad shell command is a scan over the whole corpus, and latency degrades sharply as the corpus grows. We argue that the role of retrieval for agentic search is not just to select documents that fit in the LLM context window, but to construct an interaction space: a bounded subset of the corpus the agent can explore with associated tools. Two design consequences follow. The space needs a boundary supplied by retrieval, and the objects within it should be processed for interaction. As a proof of concept, we propose RISE (Retrieving Interaction SpacE): we use BM25 to construct the interaction space; meanwhile, its documents are processed during indexing for shell-style navigation. On BrowseComp-Plus, RISE matches the pure-shell DCI baseline at 78% accuracy with gpt-5.4-mini at roughly one quarter of the per-query cost. At 1M documents, RISE-BM25 reaches 81% on gpt-5.4-mini, whereas DCI on gpt-5.4-nano degrades to 60% with 33 of 100 wall-clock failures. 

---
# Mind the Gap: Bridging Behavioral Silos with LLMs in Multi-Vertical Recommendations 

**Authors**: Nimesh Sinha, Raghav Saboo, Martin Wang, Sudeep Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.06779)  

**Abstract**: In multi-vertical e-commerce platforms like DoorDash, relatively newer product verticals such as grocery and retail present a significant opportunity for personalization innovation. A key challenge lies in solving the "cold start" problem for users. This paper introduces a novel framework for enhancing recommendation quality by transferring knowledge from data-rich verticals (e.g., restaurants at DoorDash) to data-sparse ones. We leverage Large Language Models (LLMs) to perform generative inference, synthesizing sparse, high-dimensional features that encapsulate latent user affinities. Specifically, we employ a hierarchical Retrieval-Augmented Generation (RAG) pipeline to derive multi-level taxonomic features from user restaurant order histories and search queries. These generated features, encoding both long-term cross-vertical preferences and short-term intent, are integrated into a production Multi-Task Learning (MTL) ranking model. We demonstrate through extensive offline and online evaluation that this approach significantly improves personalization and engagement in emerging business verticals, effectively bridging the behavioral data gap. 

---
# Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings 

**Authors**: Songhao Wu, Zhongxin Chen, Yuxuan Liu, Heng Cui, Cong Li, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.07502)  

**Abstract**: Large language models exhibit impressive zero-shot capabilities across a wide range of downstream tasks. However, they struggle to function as off-the-shelf embedding models, leading to suboptimal performance on massive text embedding benchmarks. In this paper, we identify a potential cause underlying this deficiency. Our motivation stems from an unexpected observation: text embeddings tend to align with frequent but uninformative tokens when projected onto the vocabulary space. We argue that this excessive expression of high-frequency tokens suppresses the model's ability to capture nuanced semantics. To address this, we introduce EmbedFilter, a simple linear transformation designed to refine text embeddings derived from LLMs directly. Specifically, we uncover that the unembedding matrix within LLMs encodes a latent space that is actively writing these frequent tokens into embedding space. By filtering out this subspace, EmbedFilter suppress the influence of high-frequency tokens, thereby enhancing semantic representations. As a compelling byproduct, this enables an inherent dimensionality reduction, lowering index storage and speedup retrieval while fully preserving the refined embedding quality. Our experiments across multiple LLM backbones demonstrate that LLMs equipped with EmbedFilter achieve superior zero-shot downstream performance even with significantly reduced embedding dimensions. We hope our findings provide deeper insights into the mechanisms of LLM-based representations and inspire more principled designs to improve text embeddings training. Our code is available at this https URL. 

---
# TA-RAG: Tone-Aware Retrieval-Augmented Generation for Peer-Support Health Communication 

**Authors**: Yong-Bin Kang, Anthony McCosker  

**Link**: [PDF](https://arxiv.org/pdf/2606.06794)  

**Abstract**: Retrieval-augmented generation (RAG) successfully grounds large language model (LLM) outputs in trusted documents, but factual grounding alone is insufficient for sensitive peer-support health communication. In domains such as HIV peer support, responses must also be accessible, stigma-free, empathetic, and tailored to the recipient. This paper presents TA-RAG, a lightweight, prompt-based tone-aware RAG framework that embeds explicit tone control into a RAG pipeline without requiring model fine-tuning. We operationalise tone across four core components: stigma-free rewriting, readability adjustment, recipient adaptation, and empathy rephrasing. We evaluate TA-RAG through component-level tests using questions derived from HIV Online Learning Australia (HOLA), UNAIDS terminology guidance, readability metrics, peer-support standards from National Association of People with HIV Australia (NAPWHA), and a public empathy dataset. Results show that the TA-RAG's components improve their targeted communication quality while preserving key content. These findings emphasise that prompt-based tone control is a potential direction for making RAG outputs suitable for sensitive peer-support health communication. 

---
