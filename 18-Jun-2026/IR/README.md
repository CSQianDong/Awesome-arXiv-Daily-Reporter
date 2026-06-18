# Querit-Reranker: Training Compact Multilingual Rerankers via Efficient Label-Free Distribution Adaptation 

**Authors**: Yunfei Zhong, Jun Yang, Wei Huang, Yinqiong Cai, Haosheng Qian, Yixing Fan, Ruqing Zhang, Lixin Su, Daiting Shi, Jiafeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.19037)  

**Abstract**: Deployable multilingual rerankers must generalize across languages, domains, and target ranking tasks while remaining efficient enough for second-stage reranking. However, adapting them to new target distributions typically requires extensive task-specific relevance annotations, which are costly to obtain. We present Querit-Reranker, a family of multilingual cross-encoder rerankers trained with a data-centric pipeline for label-efficient adaptation. We instantiate it as Querit-Reranker-A0.4B, initialized from an in-house MoE backbone with 0.4B activated parameters, and Querit-Reranker-4B, initialized from Qwen3-Embedding-4B. Our pipeline first learns general relevance modeling from large-scale ranking-oriented data, then adapts to target distributions through synthetic-query mining with teacher scores as continuous soft labels. To consolidate complementary task-adapted strengths, we further merge checkpoints via spherical linear interpolation, obtaining a single deployable model without runtime ensembling overhead. Using Qwen3-Embedding-0.6B as the shared first-stage retriever, Querit-Reranker-A0.4B improves average nDCG@10 from 54.11 to 59.28 on BEIR and from 59.87 to 67.70 on MIRACL. On MTEB Multilingual v2 Reranking, it also substantially outperforms larger embedding-based baselines, while Querit-Reranker-4B further achieves state-of-the-art performance among publicly available models. We release both models on Hugging Face. 

---
# SAERec: Constructing Fine-grained Interpretable Intents Priors via Sparse Autoencoders for Recommendation 

**Authors**: Jiangnan Xia, Xuansheng Wu, Yu Yang, Xin Wang, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.18897)  

**Abstract**: Intent-based recommender systems have gained significant attention for improving accuracy and interpretability by modeling the underlying motivations behind user behaviors. Most existing models derive intents directly from user sequences via clustering or prototype learning. However, they are sensitive to sequence quality, require presetting the number of intents, and lack explicit semantic grounding. These issues lead to an incomplete and coarse intent set and limit the effectiveness of recommendation. In this paper, we propose the Sparse Autoencoder for intent-based recommendation (SAERec), a novel recommender that automatically constructs a fine-grained and interpretable intent space from a textual corpus to guide recommendation. Rather than treating texts as side signals, SAERec leverages them as high information density evidence for intent construction. Specifically, we first extract a comprehensive set of fine-grained interpretable intents from the latent space of large language models (LLMs) by using a sparse autoencoder (SAE) to disentangle and interpret text embeddings, which isolates intent-related semantics from textual noise. Then, for each user, we retrieve relevant intents from this set as priors to guide recommendation. It contains personal intents matching a user's current interests and public intents capturing general item patterns shared across users (e.g., quality, price). Finally, to integrate retrieved intents into sequence modeling, we propose a multi-branch attention mechanism that captures temporal dependencies and injects both personal and public intent signals, followed by an adaptive fusion layer to construct the final user representation for recommendation. Extensive experiments on public datasets demonstrate the superiority of SAERec, consistently outperforming state-of-the-art baselines while providing human-understandable explanations. 

---
# LensKit-Auto 

**Authors**: Max Breit, Anass Amezian El Idrissi, Rishikesh Giriraj Kulkarni, Luca Quade  

**Link**: [PDF](https://arxiv.org/pdf/2606.18814)  

**Abstract**: Recommender systems have a wide area of application, e.g. in fields like video streaming, social media, or digital marketplaces. But, for a recommender-system, finding the right algorithm with the right hyperparameters is a reoccurring challenge. There is no one-fits-all solution, since the performance of one algorithm can vary immensely on different data sets. Due to the challenges of finding the right algorithm and the broad use of recommender-systems, it is of interest to create an Automated Recommender System (AutoRecSys) that takes on the task of finding the right algorithm-hyperparameter-combination for a given data set. In this work, we present the enhancement of LensKit-Auto, a framework introduced by Vente et al., that solves exactly this task of finding a fitting algorithm-hyperparameter-combination. LensKit-Auto's biggest strength lies in its ease of use, where it operates as a black-box, into which the user can feed their data set and receive the information of which algorithm and hyperparameters work best on this data set. In this work, we bring LensKit-Auto up to date, so that it works with the new version of its underlying framework, LensKit. We also implement further functionalities, such as the Tree Parzen Estimator as an additional optimization method, the ability to reuse the found algorithm, updated documentation, and the ability to visualize the optimization process. We also adapt an existing meta-learning framework to generate a suitable meta-dataset for LensKit-Auto, which could enable the integration of meta-learning into LensKit-Auto in the future. The presented changes bring LensKit-Auto up to date and enhance its usability, so that even non-experts in the field can find the right algorithm for their use case. 

---
# Rescaling MLM-Head for Neural Sparse Retrieval 

**Authors**: Youngjoon Jang, Seongtae Hong, Jonah Turner, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2606.18811)  

**Abstract**: Learned sparse retrieval (LSR) models such as SPLADE have traditionally used BERT-style masked language models as backbone encoders. A natural expectation is that replacing BERT with stronger pretrained encoders should improve retrieval effectiveness. However, we find that under standard SPLADE training recipes, backbones with large MLM-head L2 norms can suffer performance degradation and even training collapse under standard SPLADE training recipes. We identify this failure as a scale mismatch in the MLM head: SPLADE directly uses MLM-head outputs to construct sparse lexical representations, and query-document relevance is computed by an unnormalized dot product over these representations. As a result, an inflated MLM-head scale can amplify sparse activations, distort matching scores, and destabilize contrastive training under common training settings. To address this issue, we introduce a simple initialization-time correction that rescales the MLM-head projection by a constant factor before SPLADE training. This zero-cost adjustment improves training stability without modifying the model architecture or training objective. Across both in-domain and out-of-domain retrieval benchmarks, this simple correction substantially improves large-norm backbones such as ModernBERT and Ettin, turning unstable training runs into competitive sparse retrievers. In several settings, the corrected models further match or surpass the classic BERT-SPLADE baseline. These findings suggest that the bottleneck in adapting pretrained encoders to LSR is not encoder capacity alone, but the calibration of the MLM-head scale used to construct sparse lexical representations. 

---
# SHIFT: Semantic Harmonization via Index-side Feature Transformation for Multilingual Information Retrieval 

**Authors**: Youngjoon Jang, Seongtae Hong, Hyeonseok Moon, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2606.18801)  

**Abstract**: With the rapid expansion of massive multilingual corpora, Multilingual Information Retrieval (MLIR) has emerged as a critical technology for global information access. MLIR enables users to retrieve semantically relevant documents from multilingual text collections using a single-language query. However, recent multilingual dense retrieval models often exhibit a strong preference for documents in the same language as the query. This leads to severe language bias, where top-ranked results are dominated by documents of specific languages, even when documents in other languages contain more semantically relevant information. To address this issue, we propose SHIFT, a training-free method applicable in the indexing stage. Specifically, SHIFT utilizes parallel translation pairs to estimate a relative language vector for each target language with respect to a source language. Subsequently, SHIFT corrects the language-specific offset by subtracting this relative language vector from document embeddings during indexing. Our comprehensive evaluation across four MLIR benchmarks and diverse dense retrieval models confirms that SHIFT can effectively mitigate language bias and enhance MLIR performance. 

---
# RankGraph-2: Lifecycle Co-Design for Billion-Node Graph Learning in Recommendation 

**Authors**: Renzhi Wu, Zikun Cui, Junjie Yang, Tai Guo, Hong Li, Xian Chen, Li Yu, Ke Pan, Sri Reddy, Mahesh Srinivasan, Nipun Mathur, Haomin Yu, Hong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.18379)  

**Abstract**: Graph-based retrieval at billion-node scale requires jointly solving three tightly coupled problems -- graph construction, representation learning, and real-time serving -- yet existing work addresses each in isolation. We present RankGraph-2, a framework deployed at Meta that co-designs all three lifecycle stages for similarity-based retrieval (U2U2I and U2I2I), where each stage's requirements shape the others. Serving requires a co-learned cluster index to avoid expensive online KNN -- this pushes index co-training into the training objective. Training benefits from the observation that similarity-based retrieval tolerates pre-computed neighborhoods, eliminating online graph infrastructure -- this requires construction to produce self-contained data. Construction must also support hour-level refresh for item coverage. Acting on these cascading requirements, RankGraph-2 reduces hundreds of trillions of edges to hundreds of billions via subsampling with popularity bias correction, pre-computes multi-hop neighborhoods via personalized PageRank, and co-learns a residual-quantization cluster index that reduces serving computational cost by 83%. This lifecycle co-design enables a simple architecture to achieve 3.8 x higher recall than a GAT + Deep Graph Infomax model on a bipartite graph and 2.1 x higher than PyTorch-BigGraph on item retrieval. RankGraph-2 delivers up to +0.96% CTR and +2.75% CVR, and has powered 20+ retrieval launches across major surfaces. 

---
# Which Sections of a Research Paper Best Reveal Its Research Methods? Evidence from Library and Information Science 

**Authors**: Qiuyu Fang, Jiayi Hao, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.19051)  

**Abstract**: Research methods are essential carriers of knowledge contribution in academic papers. Automatic multi-label classification of research methods can support knowledge services such as method retrieval, review generation, and research intelligence analysis. While existing studies primarily rely on titles and abstracts, abstracts often provide only limited methodological information, whereas utilizing full-text content faces challenges related to excessive length and information redundancy. Therefore, this paper proposes a segment combination strategy by partitioning the full-text content according to its physical postion. Using an annotated corpus of 1,954 full-text articles from three representative journals in Library and Information Science (JASIST, LISR, and JDoc), we evaluate the classification performance of various segments and their combinations across multiple models. Experimental results indicate that methodological information is distributed unevenly within the full-text content, with the middle-to-late and final segments exhibiting greater discriminative power. Furthermore, integrating bibliographic metadata with cross-segment combination strategies effectively enhances classification performance. 

---
# Decoupling Search from Reasoning: A Vendor-Agnostic Grounding Architecture for LLM Agents 

**Authors**: Emmanuel Aboah Boateng, Kyle MacDonald, Amardeep Kumar, Siddharth Kodwani, Sudeep Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.18947)  

**Abstract**: Production LLM agents increasingly depend on real-time search, yet native search grounding bundles retrieval policy, provider choice, evidence injection, cost, latency, and generation behavior behind a single model-provider boundary. This coupling makes grounding hard to inspect, tune, reuse, or port, and can trigger Search-Induced Verbosity that breaks strict output contracts. We present Decoupled Search Grounding (DSG), a vendor-agnostic boundary that moves grounding outside the reasoning model through an MCP-compatible gateway, exposing provider routing, source-aware context rendering, configured fallback, retrieval-depth control, and exact plus semantic caching as first-class controls. Across five frontier models on SimpleQA, FreshQA, and HotpotQA, native search leads on recency-sensitive FreshQA, but DSG exposes a stronger frontier when control matters: on SimpleQA it nearly matches native accuracy (86.1% vs. 87.7%) at 91% lower search cost, preserves concise answer contracts, and reaches a 99.4% warm-cache hit rate with 68% lower latency. Deployed as a shared production grounding layer for large-scale agentic workloads with interchangeable models, DSG matches or slightly exceeds native-search accuracy on an e-commerce query-understanding (QIU) workload while cutting search cost by over 98%. Real-time grounding is best treated as an optimizable interface boundary, not a fixed model feature. 

---
# Zero-Shot Active Feature Acquisition via LLM-Elicitation 

**Authors**: Binyamin Perets, Natalie Mendelson, Shiran Vainberg, Yehuda Chowers, Shai Shen-Orr, Shie Mannor  

**Link**: [PDF](https://arxiv.org/pdf/2606.18933)  

**Abstract**: Active feature acquisition (AFA) sequentially selects which features to observe to reach a classification or ranking decision. Its central limitation is reliance on large amount of labeled data to fit probabilistic models guiding acquisition. Large language models (LLMs) supply unsupervised domain knowledge, but are poor sequential planners. Asking one to both know and decide conflates capabilities best kept separate. Here, we develop a framework for zero-shot AFA through disciplined elicitation: asking the LLM only for what it can be trusted to return, the unary deviations and pairwise co-variations that are the sufficient statistics of a Markov random field (MRF). We apply our framework to two settings: binary classification and top-$k$ identification. In practice, the LLM reliably returns only discriminative statistics, what distinguishes the classes rather than each class in isolation, which precludes classical AFA. We apply a maximum-entropy closure that resolves this gauge ambiguity. We evaluate on a cohort of Inflammatory Bowel Disease (IBD) patients, an active clinical setting where diagnostic ambiguity and patient heterogeneity obstruct stable treatment strategies. Our framework outperforms the LLM both on real labels and on its own extracted beliefs. Where it matters most, on the hardest patients, our top-$k$ acquisition policy markedly outperforms all existing methods. 

---
# LARE: Low-Attention Region Encoding for Text-Image Retrieval 

**Authors**: Abdulmalik Alquwayfili, Faisal Almeshal, Jumanah Almajnouni, Leena Alotaibi, Faisal Alhajari, Mohammed Alkhrashi, Alreem Almuhrij, Abdullah Aldwyish, Raied Aljadaany, Huda Alamri, Muhammad Kamran J. Khan  

**Link**: [PDF](https://arxiv.org/pdf/2606.18885)  

**Abstract**: Image retrieval in crowded scenes is particularly challenging due to the salience bias of conventional visual encoders, which tend to focus on dominant objects while neglecting low-attention regions that are often crucial for fine-grained retrieval. We propose LARE (Low-Attention Region Encoding), a framework that explicitly models these overlooked regions. LARE adopts a dual-encoding strategy that encodes low-attention regions of an image and the full image in parallel, leading to more diverse and informative image embeddings. To evaluate image retrieval performance in challenging crowded scenes, we introduce Dense-Set, a challenging subset derived from COCO and Flickr30K. In this subset, images are re-captioned to provide richer descriptions of low-attention or previously overlooked regions. This dataset highlights the limitations of existing retrieval models and enables a more rigorous evaluation under densely crowded scene conditions. Experimental results demonstrate that the proposed framework improves retrieval performance by preserving subtle, non-dominant visual cues within the shared latent space. 

---
# ScholarSum: Student-Teacher Abstractive Summarization via Knowledge Graph Reasoning and Reflective Refinement 

**Authors**: Bohou Zhang, Xiaoyu Tao, Mingyue Cheng, Huijie Liu, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.18850)  

**Abstract**: Abstractive summarization plays a crucial role in enabling efficient understanding of scientific literature, yet it inherently demands both linguistic fluency and factual faithfulness. Existing approaches often fail to reconcile these two requirements. Extractive methods rely on rigid sentence splicing that disrupts macro-level logical coherence, while large language model (LLM)-based generative approaches, despite mastering linguistic fluency, exhibit limited factual consistency. In this work, we propose ScholarSum, a hierarchical reflective graph-based framework that emulates a student-teacher writing process for fluent and faithful scientific summarization. ScholarSum first organizes the document into a hierarchical knowledge graph by segmenting it into semantically coherent units, whose multi-layered community structure captures global logic and macro-level themes. Guided by this global structure, the student generates an initial draft, which is subsequently refined through fine-grained evidence retrieval. To ensure factual consistency, a teacher-like reviewer then iteratively examines the draft, identifies unsupported content, and prompts targeted re-retrieval and rewriting until the summary meets rigorous quality standards. Extensive experiments demonstrate that ScholarSum significantly outperforms previous baselines in terms of both completeness and faithfulness. Our code is available at this https URL. 

---
# TW-LegalBench: Measuring Taiwanese Legal Understanding 

**Authors**: Fei-Yueh Chen, Chun Huang Lin, Chan Wei Hsu, Kuan Hsuan Yeh, Zih-Ching Chen, Kuan-Ming Chen, Patrick Chung-Chia Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.18699)  

**Abstract**: Large language models (LLMs) have shown impressive capabilities across diverse tasks, yet their performance on jurisdiction-specific legal reasoning remains underexplored. We present TW-LegalBench that utilizes Taiwanese legal system's rich official corpus open to the public to fill the gap in evaluating LLMs on Taiwanese law, among common-law benchmarks that focus on English sources and civil-law benchmarks focusing on sources of Simplified Chinese. TW-LegalBench comprises three task types: (1) over 16,000 multiple-choice questions (MCQs) across five years of official examinations in 18 professional domains; (2) 117 open-ended essay questions (OEQs) from examinations for legal professionals with official scoring rubrics; and (3) more than 14,000 legal judgment prediction (LJP) instances covering hundreds of crime categories. We evaluate 13 LLMs using accuracy for MCQs, a decomposed LLM-as-Judge framework based on the scoring rubric points for OEQs, and metrics for sentencing accuracy and statute citation for LJP. Our results reveal that top-performing models exceed the passing threshold for qualified lawyers (passing rate: 11%) but fall short of that for judges and prosecutors (passing rate: 1~2%). For LJP, while models demonstrate reasonable verdict type accuracy and sentence prediction capability, they struggle to cite exact legal articles. These findings highlight that reliable legal text generation remains challenging for LLMs, even though their performance on qualification examinations approaches human level. 

---
# Compact Geometric Representations of Hierarchies 

**Authors**: Prashant Gokhale, Piotr Indyk, Yuhao Liu, Sandeep Silwal, Tony Chang Wang, Haike Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.18520)  

**Abstract**: Computing geometric representations of data is a cornerstone of modern machine learning, typically achieved by training dual encoders which map queries and documents into a shared embedding space. Recent work of You et al. [NeurIPS '25] has extended this approach to hierarchical retrieval, where relevance is determined by the ancestor-descendant relationships in a Directed Acyclic Graph (DAG). While previous work has shown that valid embeddings exist when the number of descendants is small, these bounds degrade significantly for deep hierarchies, requiring dimensions as large as the total number of nodes.
In this paper, we investigate compact reachability embeddings for more general graph classes and provide theoretical guarantees for representing hierarchies using embeddings whose dimension depends on structural graph parameters. We prove that for any directed tree, there exists a reachability embedding in constant dimension 3, independent of the tree's size or depth. We generalize this result to graphs characterized by treewidth $t$, constructing embeddings of dimension $O(t \log n)$, where $n$ is the number of nodes. Complementing these upper bounds, we provide matching or near-matching lower bounds, showing that dimension $\Omega(n)$ is necessary for general DAGs and $\Omega(t/\log(n/t))$ is required for graphs of treewidth $t$. We also obtain upper and lower bounds parameterized by the number of cross-edges in the DAG. We additionally show that our embeddings can be constructed on real world datasets, and that they give much smaller dimensions in high recall regimes compared to prior embeddings with theoretical guarantees. 

---
# MCompassRAG: Topic Metadata as a Semantic Compass for Paragraph-Level Retrieval 

**Authors**: Amirhossein Abaskohi, Raymond Li, Gaetano Cimino, Peter West, Giuseppe Carenini, Issam H. Laradji  

**Link**: [PDF](https://arxiv.org/pdf/2606.18508)  

**Abstract**: Retrieval-augmented generation (RAG) systems depend critically on how documents are chunked and searched. Fine-grained chunks can improve retrieval precision but expand the search space, increasing latency and cost; larger chunks reduce the number of candidates but make dense similarity less reliable, as the representation for each chunk mixes multiple topics and introduces more semantic noise. This trade-off becomes especially limiting in deep research tasks, where retrieval must be both fast and precise across large, heterogeneous corpora. We introduce MCompassRAG, a metadata-guided retrieval framework that uses topic-level signals as a semantic compass for selecting relevant evidence. Instead of relying only on cosine similarity between queries and noisy chunk embeddings, MCompassRAG enriches chunk representations with topic metadata in the same embedding space and trains a lightweight retriever through LLM-teacher distillation. At inference time, MCompassRAG performs topic-aware retrieval without additional LLM calls, improving both efficiency and evidence quality. Across six complex retrieval benchmarks, MCompassRAG improves information efficiency (IE) by 8.24% on average with over 5 times lower latency than the strongest efficient RAG baselines. Code is available on this https URL. 

---
# SproutRAG: Attention-Guided Tree Search with Progressive Embeddings for Long-Document RAG 

**Authors**: Amirhossein Abaskohi, Issam H. Laradji, Peter West, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2606.18381)  

**Abstract**: Retrieval-augmented generation (RAG) systems must balance retrieval granularity with contextual coherence, a challenge that existing methods address through LLM-guided chunking, single-level context expansion, or hierarchical summarization. These approaches variously depend on costly LLM calls during indexing or retrieval, limit context aggregation to a single granularity level, or introduce information loss through summarization. We present SproutRAG, an attention-guided hierarchical RAG framework that addresses this trade-off by organizing sentence-level chunks into progressively larger but semantically coherent units, using learned inter-sentence attention to construct a binary chunking tree. Unlike prior approaches that rely on external LLMs, fixed context expansion, or lossy summarization, SproutRAG learns which attention heads and layers best capture semantic document structure, enabling multi-granularity retrieval without additional LLM calls or compressed summaries. At retrieval time, SproutRAG uses hierarchical beam search to retrieve candidates at multiple granularities, capturing multi-sentence relevance beyond flat retrieval. The framework is trained end-to-end with a joint objective that improves both embeddings and tree structure. Experiments across four benchmarks spanning scientific, legal, and open-domain settings demonstrate that SproutRAG improves information efficiency (IE) by 6.1% on average over the strongest baseline. Code is available on this https URL. 

---
