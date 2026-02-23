# A Topology-Aware Positive Sample Set Construction and Feature Optimization Method in Implicit Collaborative Filtering 

**Authors**: Jiayi Wu, Zhengyu Wu, Xunkai Li, Rong-Hua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18288)  

**Abstract**: Negative sampling strategies are widely used in implicit collaborative filtering to address issues like data sparsity and class imbalance. However, these methods often introduce false negatives, hindering the model's ability to accurately learn users' latent preferences. To mitigate this problem, existing methods adjust the negative sampling distribution based on statistical features from model training or the hardness of negative samples. Nevertheless, these methods face two key limitations: (1) over-reliance on the model's current representation capabilities; (2) failure to leverage the potential of false negatives as latent positive samples to guide model learning of user preferences more accurately. To address the above issues, we propose a Topology-aware Positive Sample Set Construction and Feature Optimization method (TPSC-FO). First, we design a simple topological community-aware false negative identification (FNI) method and observe that topological community structures in interaction networks can effectively identify false negatives. Motivated by this, we develop a topology-aware positive sample set construction module. This module employs a differential community detection strategy to capture topological community structures in implicit feedback, coupled with personalized noise filtration to reliably identify false negatives and convert them into positive samples. Additionally, we introduce a neighborhood-guided feature optimization module that refines positive sample features by incorporating neighborhood features in the embedding space, effectively mitigating noise in the positive samples. Extensive experiments on five real-world datasets and two synthetic datasets validate the effectiveness of TPSC-FO. 

---
# HyTRec: A Hybrid Temporal-Aware Attention Architecture for Long Behavior Sequential Recommendation 

**Authors**: Lei Xin, Yuhao Zheng, Ke Cheng, Changjiang Jiang, Zifan Zhang, Fanhu Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2602.18283)  

**Abstract**: Modeling long sequences of user behaviors has emerged as a critical frontier in generative recommendation. However, existing solutions face a dilemma: linear attention mechanisms achieve efficiency at the cost of retrieval precision due to limited state capacity, while softmax attention suffers from prohibitive computational overhead. To address this challenge, we propose HyTRec, a model featuring a Hybrid Attention architecture that explicitly decouples long-term stable preferences from short-term intent spikes. By assigning massive historical sequences to a linear attention branch and reserving a specialized softmax attention branch for recent interactions, our approach restores precise retrieval capabilities within industrial-scale contexts involving ten thousand interactions. To mitigate the lag in capturing rapid interest drifts within the linear layers, we furthermore design Temporal-Aware Delta Network (TADN) to dynamically upweight fresh behavioral signals while effectively suppressing historical noise. Empirical results on industrial-scale datasets confirm the superiority that our model maintains linear inference speed and outperforms strong baselines, notably delivering over 8% improvement in Hit Rate for users with ultra-long sequences with great efficiency. 

---
# Dual-Tree LLM-Enhanced Negative Sampling for Implicit Collaborative Filtering 

**Authors**: Jiayi Wu, Zhengyu Wu, Xunkai Li, Rong-Hua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18249)  

**Abstract**: Negative sampling is a pivotal technique in implicit collaborative filtering (CF) recommendation, enabling efficient and effective training by contrasting observed interactions with sampled unobserved ones.
Recently, large language models (LLMs) have shown promise in recommender systems; however, research on LLM-empowered negative sampling remains underexplored.
Existing methods heavily rely on textual information and task-specific fine-tuning, limiting practical applicability.
To address this limitation, we propose a text-free and fine-tuning-free Dual-Tree LLM-enhanced Negative Sampling method (DTL-NS).
It consists of two modules: (i) an offline false negative identification module that leverages hierarchical index trees to transform collaborative structural and latent semantic information into structured item-ID encodings for LLM inference, enabling accurate identification of false negatives; and (ii) a multi-view hard negative sampling module that combines user-item preference scores with item-item hierarchical similarities from these encodings to mine high-quality hard negatives, thus improving models' discriminative ability.
Extensive experiments demonstrate the effectiveness of DTL-NS. For example, on the Amazon-sports dataset, DTL-NS outperforms the strongest baseline by 10.64% and 19.12% in Recall@20 and NDCG@20, respectively.
Moreover, DTL-NS can be integrated into various implicit CF models and negative sampling methods, consistently enhancing their performance. 

---
# The Economical-Ecological Benefits of Matching Non-matching Socks 

**Authors**: Teddy Lazebnik  

**Link**: [PDF](https://arxiv.org/pdf/2602.18221)  

**Abstract**: Socks are produced and replaced at a massive scale, yet their paired use makes them unusually vulnerable to waste, as the loss of a single sock can strand usable wear-capacity and trigger premature replacement. In this study, we quantify the economic and ecological value of pairing non-matching \say{orphan} socks, and the social cost that discourages this behaviour. We formalize sock ownership as a sequential decision problem under uncertainty in which socks wear out and disappear stochastically during laundering, while public exposure induces a person-specific mismatch penalty. We conducted an in-person study to estimate mismatch sensitivity and diversity preference, linking behavioural heterogeneity to optimal mixing strategies. Using these results and a computer simulation-based evaluation of interpretable pairing policies, we show that strict matching can appear resource-frugal largely because it generates many sockless days, whereas controlled tolerance for mismatch sustains service and reduces stranded capacity across loss regimes. This study establishes the feasibility of matching non-matching socks while outlining its limitations and challenges. 

---
# A Simple yet Effective Negative Sampling Plugin for Constructing Positive Sample Pairs in Implicit Collaborative Filtering 

**Authors**: Jiayi Wu, Zhengyu Wu, Xunkai Li, Ronghua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18206)  

**Abstract**: Most implicit collaborative filtering (CF) models are trained with negative sampling, where existing work designs sophisticated strategies for high-quality negatives while largely overlooking the exploration of positive samples. Although some denoising recommendation methods can be applied to implicit CF for denoising positive samples, they often sparsify positive supervision. Moreover, these approaches generally overlook user activity bias during training, leading to insufficient learning for inactive users. To address these issues, we propose a simple yet effective negative sampling plugin, PSP-NS, from the perspective of enhancing positive supervision signals. It builds a user-item bipartite graph with edge weights indicating interaction confidence inferred from global and local patterns, generates positive sample pairs via replication-based reweighting to strengthen positive signals, and adopts an activity-aware weighting scheme to effectively learn inactive users' preferences. We provide theoretical insights from a margin-improvement perspective, explaining why PSP-NS tends to improve ranking quality (e.g., Precision@k/Recall@k), and conduct extensive experiments on four real-world datasets to demonstrate its superiority. For instance, PSP-NS boosts Recall@30 and Precision@30 by 32.11% and 22.90% on Yelp over the strongest baselines. PSP-NS can be integrated with various implicit CF recommenders or negative sampling methods to enhance their performance. 

---
# SuiteEval: Simplifying Retrieval Benchmarks 

**Authors**: Andrew Parry, Debasis Ganguly, Sean MacAvaney  

**Link**: [PDF](https://arxiv.org/pdf/2602.18107)  

**Abstract**: Information retrieval evaluation often suffers from fragmented practices -- varying dataset subsets, aggregation methods, and pipeline configurations -- that undermine reproducibility and comparability, especially for foundation embedding models requiring robust out-of-domain performance. We introduce SuiteEval, a unified framework that offers automatic end-to-end evaluation, dynamic indexing that reuses on-disk indices to minimise disk usage, and built-in support for major benchmarks (BEIR, LoTTE, MS MARCO, NanoBEIR, and BRIGHT). Users only need to supply a pipeline generator. SuiteEval handles data loading, indexing, ranking, metric computation, and result aggregation. New benchmark suites can be added in a single line. SuiteEval reduces boilerplate and standardises evaluations to facilitate reproducible IR research, as a broader benchmark set is increasingly required. 

---
# Enhancing Scientific Literature Chatbots with Retrieval-Augmented Generation: A Performance Evaluation of Vector and Graph-Based Systems 

**Authors**: Hamideh Ghanadian, Amin Kamali, Mohammad Hossein Tekieh  

**Link**: [PDF](https://arxiv.org/pdf/2602.17856)  

**Abstract**: This paper investigates the enhancement of scientific literature chatbots through retrieval-augmented generation (RAG), with a focus on evaluating vector- and graph-based retrieval systems. The proposed chatbot leverages both structured (graph) and unstructured (vector) databases to access scientific articles and gray literature, enabling efficient triage of sources according to research objectives. To systematically assess performance, we examine two use-case scenarios: retrieval from a single uploaded document and retrieval from a large-scale corpus. Benchmark test sets were generated using a GPT model, with selected outputs annotated for evaluation. The comparative analysis emphasizes retrieval accuracy and response relevance, providing insight into the strengths and limitations of each approach. The findings demonstrate the potential of hybrid RAG systems to improve accessibility to scientific knowledge and to support evidence-based decision making. 

---
# IRPAPERS: A Visual Document Benchmark for Scientific Retrieval and Question Answering 

**Authors**: Connor Shorten, Augustas Skaburskas, Daniel M. Jones, Charles Pierse, Roberto Esposito, John Trengrove, Etienne Dilocker, Bob van Luijt  

**Link**: [PDF](https://arxiv.org/pdf/2602.17687)  

**Abstract**: AI systems have achieved remarkable success in processing text and relational data, yet visual document processing remains relatively underexplored. Whereas traditional systems require OCR transcriptions to convert these visual documents into text and metadata, recent advances in multimodal foundation models offer retrieval and generation directly from document images. This raises a key question: How do image-based systems compare to established text-based methods? We introduce IRPAPERS, a benchmark of 3,230 pages from 166 scientific papers, with both an image and an OCR transcription for each page. Using 180 needle-in-the-haystack questions, we compare image- and text-based retrieval and question answering systems. Text retrieval using Arctic 2.0 embeddings, BM25, and hybrid text search achieved 46% Recall@1, 78% Recall@5, and 91% Recall@20, while image-based retrieval reaches 43%, 78%, and 93%, respectively. The two modalities exhibit complementary failures, enabling multimodal hybrid search to outperform either alone, achieving 49% Recall@1, 81% Recall@5, and 95% Recall@20. We further evaluate efficiency-performance tradeoffs with MUVERA and assess multiple multi-vector image embedding models. Among closed-source models, Cohere Embed v4 page image embeddings outperform Voyage 3 Large text embeddings and all tested open-source models, achieving 58% Recall@1, 87% Recall@5, and 97% Recall@20. For question answering, text-based RAG systems achieved higher ground-truth alignment than image-based systems (0.82 vs. 0.71), and both benefit substantially from increased retrieval depth, with multi-document retrieval outperforming oracle single-document retrieval. We analyze the complementary limitations of unimodal text and image representations and identify question types that require one modality over the other. The IRPAPERS dataset and all experimental code are publicly available. 

---
# When & How to Write for Personalized Demand-aware Query Rewriting in Video Search 

**Authors**: Cheng cheng, Chenxing Wang, Aolin Li, Haijun Wu, Huiyun Hu, Juyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.17667)  

**Abstract**: In video search systems, user historical behaviors provide rich context for identifying search intent and resolving ambiguity. However, traditional methods utilizing implicit history features often suffer from signal dilution and delayed feedback. To address these challenges, we propose WeWrite, a novel Personalized Demand-aware Query Rewriting framework. Specifically, WeWrite tackles three key challenges: (1) When to Write: An automated posterior-based mining strategy extracts high-quality samples from user logs, identifying scenarios where personalization is strictly necessary; (2) How to Write: A hybrid training paradigm combines Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO) to align the LLM's output style with the retrieval system; (3) Deployment: A parallel "Fake Recall" architecture ensures low latency. Online A/B testing on a large-scale video platform demonstrates that WeWrite improves the Click-Through Video Volume (VV$>$10s) by 1.07% and reduces the Query Reformulation Rate by 2.97%. 

---
# VIRAASAT: Traversing Novel Paths for Indian Cultural Reasoning 

**Authors**: Harshul Raj Surana, Arijit Maji, Aryan Vats, Akash Ghosh, Sriparna Saha, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2602.18429)  

**Abstract**: Large Language Models (LLMs) have made significant progress in reasoning tasks across various domains such as mathematics and coding. However, their performance deteriorates in tasks requiring rich socio-cultural knowledge and diverse local contexts, particularly those involving Indian Culture. Existing Cultural benchmarks are (i) Manually crafted, (ii) contain single-hop questions testing factual recall, and (iii) prohibitively costly to scale, leaving this deficiency largely unmeasured. To address this, we introduce VIRAASAT, a novel, semi-automated multi-hop approach for generating cultural specific multi-hop Question-Answering dataset for Indian culture. VIRAASAT leverages a Knowledge Graph comprising more than 700 expert-curated cultural artifacts, covering 13 key attributes of Indian culture (history, festivals, etc). VIRAASAT spans all 28 states and 8 Union Territories, yielding more than 3,200 multi-hop questions that necessitate chained cultural reasoning. We evaluate current State-of-the-Art (SOTA) LLMs on VIRAASAT and identify key limitations in reasoning wherein fine-tuning on Chain-of-Thought(CoT) traces fails to ground and synthesize low-probability facts. To bridge this gap, we propose a novel framework named Symbolic Chain-of-Manipulation (SCoM). Adapting the Chain-of-Manipulation paradigm, we train the model to simulate atomic Knowledge Graph manipulations internally. SCoM teaches the model to reliably traverse the topological structure of the graph. Experiments on Supervised Fine-Tuning (SFT) demonstrate that SCoM outperforms standard CoT baselines by up to 20%. We release the VIRAASAT dataset along with our findings, laying a strong foundation towards building Culturally Aware Reasoning Models. 

---
# RVR: Retrieve-Verify-Retrieve for Comprehensive Question Answering 

**Authors**: Deniz Qian, Hung-Ting Chen, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2602.18425)  

**Abstract**: Comprehensively retrieving diverse documents is crucial to address queries that admit a wide range of valid answers. We introduce retrieve-verify-retrieve (RVR), a multi-round retrieval framework designed to maximize answer coverage. Initially, a retriever takes the original query and returns a candidate document set, followed by a verifier that identifies a high-quality subset. For subsequent rounds, the query is augmented with previously verified documents to uncover answers that are not yet covered in previous rounds. RVR is effective even with off-the-shelf retrievers, and fine-tuning retrievers for our inference procedure brings further gains. Our method outperforms baselines, including agentic search approaches, achieving at least 10% relative and 3% absolute gain in complete recall percentage on a multi-answer retrieval dataset (QAMPARI). We also see consistent gains on two out-of-domain datasets (QUEST and WebQuestionsSP) across different base retrievers. Our work presents a promising iterative approach for comprehensive answer recall leveraging a verifier and adapting retrievers to a new inference scenario. 

---
# Decomposing Retrieval Failures in RAG for Long-Document Financial Question Answering 

**Authors**: Amine Kobeissi, Philippe Langlais  

**Link**: [PDF](https://arxiv.org/pdf/2602.17981)  

**Abstract**: Retrieval-augmented generation is increasingly used for financial question answering over long regulatory filings, yet reliability depends on retrieving the exact context needed to justify answers in high stakes settings. We study a frequent failure mode in which the correct document is retrieved but the page or chunk that contains the answer is missed, leading the generator to extrapolate from incomplete context. Despite its practical significance, this within-document retrieval failure mode has received limited systematic attention in the Financial Question Answering (QA) literature. We evaluate retrieval at multiple levels of granularity, document, page, and chunk level, and introduce an oracle based analysis to provide empirical upper bounds on retrieval and generative performance. On a 150 question subset of FinanceBench, we reproduce and compare diverse retrieval strategies including dense, sparse, hybrid, and hierarchical methods with reranking and query reformulation. Across methods, gains in document discovery tend to translate into stronger page recall, yet oracle performance still suggests headroom for page and chunk level retrieval. To target this gap, we introduce a domain fine-tuned page scorer that treats pages as an intermediate retrieval unit between documents and chunks. Unlike prior passage-based hierarchical retrieval, we fine-tune a bi-encoder specifically for page-level relevance on financial filings, exploiting the semantic coherence of pages. Overall, our results demonstrate a significant improvement in page recall and chunk retrieval. 

---
# Efficient Filtered-ANN via Learning-based Query Planning 

**Authors**: Zhuocheng Gan, Yifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.17914)  

**Abstract**: Filtered ANN search is an increasingly important problem in vector retrieval, yet systems face a difficult trade-off due to the execution order: Pre-filtering (filtering first, then ANN over the passing subset) requires expensive per-predicate index construction, while post-filtering (ANN first, then filtering candidates) may waste computation and lose recall under low selectivity due to insufficient candidates after filtering. We introduce a learning-based query planning framework that dynamically selects the most effective execution plan for each query, using lightweight predictions derived from dataset and query statistics (e.g., dimensionality, corpus size, distribution features, and predicate statistics). The framework supports diverse filter types, including categorical/keyword and range predicates, and is generic to use any backend ANN index. Experiments show that our method achieves up to 4x acceleration with >= 90% recall comparing to the strong baselines. 

---
# VQPP: Video Query Performance Prediction Benchmark 

**Authors**: Adrian Catalin Lutu, Eduard Poesina, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2602.17814)  

**Abstract**: Query performance prediction (QPP) is an important and actively studied information retrieval task, having various applications, such as query reformulation, query expansion, and retrieval system selection, among many others. The task has been primarily studied in the context of text and image retrieval, whereas QPP for content-based video retrieval (CBVR) remains largely underexplored. To this end, we propose the first benchmark for video query performance prediction (VQPP), comprising two text-to-video retrieval datasets and two CBVR systems, respectively. VQPP contains a total of 56K text queries and 51K videos, and comes with official training, validation and test splits, fostering direct comparisons and reproducible results. We explore multiple pre-retrieval and post-retrieval performance predictors, creating a representative benchmark for future exploration of QPP in the video domain. Our results show that pre-retrieval predictors obtain competitive performance, enabling applications before performing the retrieval step. We also demonstrate the applicability of VQPP by employing the best performing pre-retrieval predictor as reward model for training a large language model (LLM) on the query reformulation task via direct preference optimization (DPO). We release our benchmark and code at this https URL. 

---
# Wavenumber-domain signal processing for holographic MIMO: Foundations, methods, and future directions 

**Authors**: Zijian Zhang, Linglong Dai  

**Link**: [PDF](https://arxiv.org/pdf/2602.17705)  

**Abstract**: Holographic multiple-input multiple-output (H-MIMO) systems represent a paradigm shift in wireless communications by enabling quasi-continuous apertures. Unlike conventional MIMO systems, H-MIMO with subwavelength antenna spacing operates in both far-field and near-field regimes, where classical discrete Fourier transform (DFT) representations fail to sufficiently capture the channel characteristics. To address this challenge, this article provides an overview of the emerging wavenumber-domain signal processing framework. Specifically, by leveraging spatial Fourier plane-wave decomposition to model H-MIMO channels, the wavenumber domain offers a unified and physically consistent basis for characterizing subwavelength-level spatial correlation and spherical wave propagation. This article first introduces the concept of H-MIMO and the wavenumber representation of H-MIMO channels. Next, it elaborates on wavenumber-domain signal processing technologies reported in the literature, including multiplexing, channel estimation, and waveform designs. Finally, it highlights open challenges and outlines future research directions in wavenumber-domain signal processing for next-generation wireless systems. 

---
# EXACT: Explicit Attribute-Guided Decoding-Time Personalization 

**Authors**: Xin Yu, Hanwen Xing, Lingzhou Xue  

**Link**: [PDF](https://arxiv.org/pdf/2602.17695)  

**Abstract**: Achieving personalized alignment requires adapting large language models to each user's evolving context. While decoding-time personalization offers a scalable alternative to training-time methods, existing methods largely rely on implicit, less interpretable preference representations and impose a rigid, context-agnostic user representation, failing to account for how preferences shift across prompts. We introduce EXACT, a new decoding-time personalization that aligns generation with limited pairwise preference feedback using a predefined set of interpretable attributes. EXACT first identifies user-specific attribute subsets by maximizing the likelihood of preferred responses in the offline stage. Then, for online inference, EXACT retrieves the most semantically relevant attributes for an incoming prompt and injects them into the context to steer generation. We establish theoretical approximation guarantees for the proposed algorithm under mild assumptions, and provably show that our similarity-based retrieval mechanism effectively mitigates contextual preference shifts, adapting to disparate tasks without pooling conflicting preferences. Extensive experiments on human-annotated preference datasets demonstrate that EXACT consistently outperforms strong baselines, including preference modeling accuracy and personalized generation quality. 

---
