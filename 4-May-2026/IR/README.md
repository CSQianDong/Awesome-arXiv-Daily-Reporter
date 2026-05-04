# Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion 

**Authors**: Yuan Li, Jun Hu, Jiaxin Jiang, Bryan Hooi, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2605.00670)  

**Abstract**: Multimodal data plays a critical role in web-based recommendation systems, where information from diverse modalities such as vision and text enhances representation learning. However, real-world multimodal datasets often suffer from modality incompleteness due to sensor failures, annotation scarcity, or privacy constraints, which substantially degrade model performance and reliability. One effective solution to address this issue is modality completion, which reconstructs missing features to provide modality-complete graphs for downstream tasks. Given a query node with missing multimodal features, existing modality completion methods typically infer information from the node itself or its neighbors to reconstruct the missing modality. However, these methods may overlook semantically relevant context in the graph, which contains valuable cues that are non-trivial to capture through simple methods like neighborhood aggregation. In this work, we propose GRE-MC, a Graph Retrieval-Enhanced Modality Completion framework, to overcome these limitations. By introducing a modality-aware subgraph retrieval mechanism, GRE-MC selects semantically relevant subgraphs from the entire graph, providing richer contextual information for completing missing modalities. Subsequently, a graph transformer jointly encodes the query node and the retrieved subgraph via global attention to complete the missing features, while a learnable sparse-routing codebook regularizes latent embeddings into compact bases for improved robustness. Extensive experiments on multimodal recommendation benchmarks demonstrate that GRE-MC consistently outperforms state-of-the-art methods, validating the effectiveness of subgraph retrieval and joint-encoding graph transformer for robust modality completion. 

---
# A Replicability Study of XTR 

**Authors**: Rohan Jha, Reno Kriz, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2605.00646)  

**Abstract**: The XTR (conteXtual Token Retrieval) algorithm is a modification to ColBERT retrieval that avoids the costly step of fully gathering and reranking the candidates' embeddings by imputing their missing similarity scores from the initial token retrieval step. The original work proposes a modified training objective as necessary for effective XTR retrieval, arguing that standard ColBERT token scoring is unsuitable for imputation. In this paper, we replicate both the XTR retrieval algorithm and its modified training objective, and extend the evaluation to knowledge-distillation (KD) training and efficient retrieval engines (PLAID and WARP). We confirm the token-level matching characteristics claimed in the original work, but fail to replicate XTR's overall effectiveness advantage over ColBERT under a controlled comparison. We further show that XTR's training modification has a concrete mechanistic consequence for modern retrieval engines: by flattening ColBERT's characteristically peaked token score distribution, XTR training yields more discriminative centroid scores and thus more efficient IVF-based retrieval under PLAID and WARP. The utility of XTR training is therefore not limited to the low-$k'$ regime originally studied, but extends to any deployment setting where IVF-based engines are used. These findings offer practitioners concrete guidance on how and when to use XTR as their multi-vector retriever. 

---
# MUDY: Multi-Granular Dynamic Candidate Contextualization for Unsupervised Keyphrase Extraction 

**Authors**: Hyeongu Kang, Susik Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2605.00597)  

**Abstract**: Keyphrase extraction aims to automatically identify concise phrases that effectively represent the content of a document. While recent methods leveraging pre-trained language models (PLMs) have significantly improved the extraction of keyphrases with strong global semantic relevance, they often fall short in capturing the local contextual importance of keyphrases tied to specific subtopics dispersed in a document. In this paper, we propose a novel context-centric framework, MUDY, that effectively captures multi-granular contextual salience of candidate keyphrases. MUDY employs two complementary components: (1) a prompt-based scoring that estimates the generation likelihood of each candidate keyphrase, augmented with candidate-aware weighting to better reflect its local contextual importance, and (2) a self-attention-based scoring that utilizes multi-granular attention patterns from PLMs to assess candidate significance at both the document-wide and segment-specific levels. Evaluations on four real-world datasets demonstrate that MUDY outperforms state-of-the-art baselines in top-k accuracy at various cutoff thresholds. In-depth quantitative and qualitative analyses further highlight the efficacy of context-centric keyphrase extraction with multi-granular saliency. For reproducibility, the source code of MUDY is available at this https URL. 

---
# When More Reformulations Hurt: Avoiding Drift using Ranker Feedback 

**Authors**: V Venktesh, Mandeep Rathee, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2605.00560)  

**Abstract**: Modern retrieval pipelines increasingly rely on query reformulation and neural reranking to improve effectiveness, but this comes at a significant computational cost and introduces a fundamental tradeoff between recall and query drift. Generating many reformulated queries can substantially increase recall, yet naively merging or exhaustively reranking their results is prohibitively expensive. In this work, we argue that the core challenge is not reformulation generation itself, but the adaptive selection of reformulations and their retrieved documents under a strict inference budget. We propose ReformIR, a budget-aware retrieval framework that treats query reformulations as first-class features and performs online relevance estimation using a strong reranker as a teacher. Given multiple reformulated queries, ReformIR constructs a large candidate pool and learns a lightweight surrogate model that estimates document utility from reformulation-specific retrieval signals. Under a fixed reranking budget, the surrogate adaptively prioritizes both reformulations and documents, selectively querying a teacher reranker anchored to the original query. This process increases recall while actively suppressing drift through online feature selection over reformulations. We conduct extensive experiments on the MSMARCO passage corpora and TREC Deep Learning benchmarks (DL19-DL22). Our results show that ReformIR consistently outperforms existing reformulation strategies, particularly as the number of reformulations increases, where prior methods suffer from severe quality degradation due to drift. Our findings also suggest a shift in retrieval system design, rather than using large language models as rerankers, their capacity is more effectively leveraged in the reformulation stage with feedback-driven optimization. 

---
# LLM-Oriented Information Retrieval: A Denoising-First Perspective 

**Authors**: Lu Dai, Liang Sun, Fanpu Cao, Ziyang Rao, Cehao Yang, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.00505)  

**Abstract**: Modern information retrieval (IR) is no longer consumed primarily by humans but increasingly by large language models (LLMs) via retrieval-augmented generation (RAG) and agentic search. Unlike human users, LLMs are constrained by limited attention budgets and are uniquely vulnerable to noise; misleading or irrelevant information is no longer just a nuisance, but a direct cause of hallucinations and reasoning failures. In this perspective paper, we argue that denoising-maximizing usable evidence density and verifiability within a context window-is becoming the primary bottleneck across the full information access pipeline. We conceptualize this paradigm shift through a four-stage framework of IR challenges: from inaccessible to undiscoverable, to misaligned, and finally to unverifiable. Furthermore, we provide a pipeline-organized taxonomy of signal-to-noise optimization techniques, spanning indexing, retrieval, context engineering, verification, and agentic workflow. We also present research works on information denoising in domains that rely heavily on retrieval such as lifelong assistant, coding agent, deep research, and multimodal understanding. 

---
# Time-Interval-Aware Disentangled Expert Modeling for Next-Basket Recommendation 

**Authors**: Zhiying Deng, Yuan Fu, Usman Farooq, Ziwei Tian, Wei Liu, Jianjun Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.00499)  

**Abstract**: Next-basket recommendation (NBR) is a type of recommendation that aims to predict a set of items a user will purchase based on their historical transaction basket sequences. It is governed by a dynamic interplay between two distinct user intents: habitual repurchase, which involves repeating past behaviors, and exploratory interest, which involves discovering new items. However, existing NBR methods generally suffer from two limitations: (1) they often entangle these conflicting motives within a single representation, causing habits to overshadow discovery, and (2) they rely on discrete sequential modeling that ignores continuous-time intervals and item-specific periodicities. In this paper, we propose a novel solution named Time-Interval Disentangled Experts (TIDE) to address these challenges. TIDE incorporates a Hawkes-enhanced Fourier Time Encoding to capture item-specific temporal periodicities and dynamic decay. To decouple user intentions, TIDE utilizes a dual-expert architecture that integrates a Habit Expert for recurring needs and a Pattern-Guided Exploration Expert for discovery. Combined with an item-aware gating mechanism, TIDE adaptively balances repurchase and exploration. Extensive experiments on four diverse real-world datasets demonstrate that TIDE consistently outperforms representative state-of-the-art NBR methods. 

---
# FollowTable: A Benchmark for Instruction-Following Table Retrieval 

**Authors**: Rihui Jin, Yuchen Lu, Ting Zhang, Jun Wang, Kuicai Dong, Zhaocheng Du, Dongping Liu, Gang Wang, Yong Liu, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00400)  

**Abstract**: Table Retrieval (TR) has traditionally been formulated as an ad-hoc retrieval problem, where relevance is primarily determined by topical semantic similarity. With the growing adoption of LLM-based agentic systems, access to structured data is increasingly instruction-driven, where relevance is conditional on explicit content and schema constraints rather than topical similarity alone. We therefore formalize Instruction-Following Table Retrieval (IFTR), a new task that requires models to jointly satisfy topical relevance and fine-grained instruction constraints. We identify two core challenges in IFTR: (i) sensitivity to content scope, such as inclusion and exclusion constraints, and (ii) awareness of schema-grounded requirements, including column semantics and representation granularity--capabilities largely absent in existing retrievers. To support systematic evaluation, we introduce FollowTable, the first large-scale benchmark for IFTR, constructed via a taxonomy-driven annotation pipeline. We further propose a new metric, termed the Instruction Responsiveness Score, to evaluate whether retrieval rankings consistently adapt to user instructions relative to a topic-only baseline. Our results indicate that existing retrieval models struggle to follow fine-grained instructions over tabular data. In particular, they exhibit systematic biases toward surface-level semantic cues and remain limited in handling schema-grounded constraints, highlighting substantial room for future improvements. 

---
# Negative Data Mining for Contrastive Learning in Dense Retrieval at IKEA.com 

**Authors**: Eva Agapaki, Amritpal Singh Gill  

**Link**: [PDF](https://arxiv.org/pdf/2605.00353)  

**Abstract**: Contrastive learning is a core component of modern retrieval systems, but its effectiveness heavily relies on the quality of negative examples used during training. In this work, we present a systematic approach to improving dense retrieval for IKEA product search through structured negative sampling strategies and scalable LLM-as-a-judge relevance evaluation.
Building on IKEA Search Engine's late-interaction retrieval architectures, we introduce two key contributions: (1) structured negative sampling strategies that leverage product hierarchical taxonomy and product attributes to generate semantically challenging negatives, and (2) a comprehensive LLM-based evaluation methodology for generating training data. Rather than relying on sparse human annotations or random sampling, our LLM-based evaluation system allocates a score for all candidate products against each query.
Our methodology achieves +2.6\% average category accuracy on offline real user query experiments on the Canada market. However, our A/B test on long-tail queries showed no statistically significant differences in user engagement metrics between the improved and baseline models ($p > 0.05$). We trace this gap to user search behavior: 67\% of popular searches exhibit zero-click rates above 50\%, indicating that a substantial proportion of search sessions result in no product engagement regardless of result ranking. These findings underscore the importance of hard negative mining but also the need for grounding training data and offline evals in real user search behavior -- including query intent distribution and zero-click patterns -- to bridge the gap between offline retrieval quality and online user engagement. 

---
# DynamicPO: Dynamic Preference Optimization for Recommendation 

**Authors**: Xingyu Hu, Kai Zhang, Jiancan Wu, Shuli Wang, Chi Wang, Wenshuai Chen, Yinhua Zhu, Haitao Wang, Xingxing Wang, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00327)  

**Abstract**: In large language model (LLM)-based recommendation systems, direct preference optimization (DPO) effectively aligns recommendations with user preferences, requiring multi-negative objective functions to leverage abundant implicit-feedback negatives and sharpen preference boundaries. However, our empirical analyses reveal a counterintuitive phenomenon, preference optimization collapse, where increasing the number of negative samples can lead to performance degradation despite a continuously decreasing training loss. We further theoretically demonstrate that this collapse arises from gradient suppression, caused by the dominance of easily discriminable negatives over boundary-critical negatives that truly define user preference boundaries. As a result, boundary-relevant signals are under-optimized, weakening the model's decision boundary. Motivated by these observations, we propose DynamicPO (Dynamic Preference Optimization), a lightweight and plug-and-play framework comprising two adaptive mechanisms: Dynamic Boundary Negative Selection, which identifies and prioritizes informative negatives near the model's decision boundary, and Dual-Margin Dynamic beta Adjustment, which calibrates optimization strength per sample according to boundary ambiguity. Extensive experiments on three public datasets show that DynamicPO effectively prevents optimization collapse and improves recommendation accuracy on multi-negative preference optimization methods, with negligible computational overhead. Our code and datasets are available at this https URL. 

---
# Intelligent Elastic Feature Fading: Enabling Model Retrain-Free Feature Efficiency Rollouts at Scale 

**Authors**: Jieming Di, Xiaoyu Chen, Ying She, Siyu Wang, Lizzie Liu, Fenggang Wu, Jiaoying Mu, Tony Tsui, Amr Elroumy, Hsing Tang, Zewei Jiang, Qiao Yang, Lin Qi, Haibo Lin, Weifeng Cui, Daniel Li, Kapil Gupta, Shivendra Pratap Singh, Jie Zheng, Arnold Overwijk, Ling Leng, Sri Reddy, Robert Malkin, Rocky Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00324)  

**Abstract**: Large-scale ranking systems depend on thousands of features derived from user behavior across multiple time horizons. Typically requires model retraining -- resulting in long iteration cycles (3--6 months), substantial GPU resource consumption, and limited rollout throughput.
We introduce Intelligent Elastic Feature Fading (IEFF), a production infrastructure system that enables retrain-free feature efficiency rollouts by elastically controlling feature coverage and distribution at serving time. IEFF supports incremental feature coverage adjustments while models adapt through recurring training, eliminating dependencies on explicit retraining cycles. The system incorporates strict safety guardrails, reversibility mechanisms, and comprehensive monitoring to ensure stability at scale.
Across multiple production use cases, IEFF accelerates efficiency-related rollouts by 5$\times$, eliminates retraining-related GPU overhead, and enables faster capacity recycling. Extensive offline and online experiments demonstrate that gradual feature fading prevents 50--55\% of online performance degradation compared to abrupt feature removal, while maintaining stable model behavior. These results establish elastic, system-level feature fading as a practical and scalable approach for managing feature efficiency in modern industrial ranking systems. 

---
# A Survey of Reasoning-Intensive Retrieval: Progress and Challenges 

**Authors**: Yiyang Wei, Tingyu Song, Siyue Zhang, Yilun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.00063)  

**Abstract**: Reasoning-Intensive Retrieval (RIR) targets retrieval settings where relevance is mediated by latent inferential links between a query and supporting evidence, rather than semantic similarity. Motivated by the emergent reasoning abilities of Large Language Models (LLMs), recent work integrates these capabilities into the IR field, spanning the entire pipeline from benchmarks to retrievers and rerankers. Despite this progress, the field lacks a systematic framework to organize current efforts and articulate a clear path forward. To provide a clear roadmap for this rapidly growing yet fragmented area, this survey (1) systematizes existing RIR benchmarks by knowledge domains and modalities, providing a detailed analysis of the current landscape; (2) introduces a structured taxonomy that categorizes methods based on where and how reasoning is integrated into the retrieval pipeline, alongside an analysis of their trade-offs and practical applications; and (3) summarizes challenges and future directions to guide research in this evolving field. 

---
# Exploring LLM biases to manipulate AI search overview 

**Authors**: Roman Smirnov  

**Link**: [PDF](https://arxiv.org/pdf/2605.00012)  

**Abstract**: Modern large language models (LLMs) are used in many business applications in general, and specifically in web search systems and applications that generate overviews of search results - LLM Overview systems. Such systems are using an LLM to select most relevant sources from search results and generate an answer to the user's query. It is known from many studies that LLMs have different biases, in LLM Overview application both the source selection and answer generation stages may be affected by the biases of LLMs (here we are focusing mainly on the selection stage). This research is focused on investigating the presence of the biases in LLM Overview systems and on biases exploitation to manipulate LLM Overview results. Here we train a small language model using reinforcement learning to rewrite search snippets to increase their likelihood of being preferred by an LLM Overview. Our experimental setup intentionally restricts the policy to operate only on snippets and limits reward-hacking strategies, reflecting realistic constraints of web search environments. The results prove that LLM Overview systems have biases and that reinforcement learning in most of the cases can optimize snippet's content to manipulate LLM Overview results. We also prove that LLM Overview selections are driven by comparative rather than absolute advantages among candidate sources. In addition, we examine safety aspects of LLM Overview manipulation possibilities and show that context poisoning attacks can lead to inaccurate or harmful results. 

---
# H-RAG at SemEval-2026 Task 8: Hierarchical Parent-Child Retrieval for Multi-Turn RAG Conversations 

**Authors**: Passant Elchafei, Hossam Emam, Mohamed Alansary, Monorama Swain, Markus Schedl  

**Link**: [PDF](https://arxiv.org/pdf/2605.00631)  

**Abstract**: We present H-RAG, our submission to SemEval-2026 Task 8 (MTRAGEval), addressing both Task A (Retrieval) and Task C (Generation with Retrieved Passages). Task A evaluates standalone retrieval quality, while Task C assesses end-to-end retrieval-augmented generation (RAG) in multi-turn conversational settings, requiring both accurate answer generation and faithful grounding in retrieved evidence. Our approach implements a hierarchical parent-child RAG pipeline that separates fine-grained child-level retrieval from parent-level context reconstruction during generation. Documents are segmented into overlapping sentence-based child chunks, while full documents are preserved as parent units to provide coherent context. Retrieval combines hybrid dense-sparse search, tunable weighting, and embedding-based similarity rescoring over child chunks. Retrieved evidence is aggregated at the parent level and supplied to an instruction-tuned language model for response generation. H-RAG achieves an nDCG@5 score of 0.4271 on Task A and a harmonic mean score of 0.3241 on Task C (RB_agg: 0.2488, RL_F: 0.2703, RB_llm: 0.6508), underscoring the importance of retrieval configuration and parent-level aggregation in multi-turn RAG performance. 

---
# Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation 

**Authors**: Ziwen Zhao, Menglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00529)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models with external knowledge, and tree-based RAG organizes documents into hierarchical indexes to support queries at multiple granularities. However, existing Tree-RAG methods designed for single-document retrieval face critical challenges in scaling to cross-document multi-hop questions: (1) poor distribution adaptability, where $k$-means clustering introduces noise due to rigid distribution assumptions; (2) structural isolation, as tree indexes lack explicit cross-document connections; and (3) coarse abstraction, which obscures fine-grained details. To address these limitations, we propose $\Psi$-RAG, a tree-RAG framework with two key components. First, a hierarchical abstract tree index built through an iterative "merging and collapse" process that adapts to data distributions without a priori assumption. Second, a multi-granular retrieval agent that intelligently interacts with the knowledge base with reorganized queries and an agent-powered hybrid retriever. $\Psi$-RAG supports diverse tasks from token-level question answering to document-level summarization. On cross-document multi-hop QA benchmarks, it outperforms RAPTOR by 25.9% and HippoRAG 2 by 7.4% in average F1 score. Code is available at this https URL. 

---
# Structure-Aware Chunking for Tabular Data in Retrieval-Augmented Generation 

**Authors**: Pooja Guttal, Varun Magotra, Vasudeva Mahavishnu, Natasha Chanto, Sidharth Sivaprasad, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2605.00318)  

**Abstract**: Tabular documents such as CSV and Excel files are widely used in enterprise data pipelines, yet existing chunking strategies for retrieval-augmented generation (RAG) are primarily designed for unstructured text and do not account for tabular structure. We propose a structure-aware tabular chunking (STC) framework that operates on row-level units by constructing a hierarchical Row Tree representation, where each row is encoded as a key-value block. STC performs token-constrained splitting aligned with structural boundaries and applies overlap-free greedy merging to produce dense, non-overlapping chunks. This design preserves semantic relationships between fields within a row while improving token utilization and reducing fragmentation. Across evaluations on the MAUD dataset, STC reduces chunk count by up to 40% and 56% compared to standard recursive and key-value based baselines, respectively, while improving token utilization and processing efficiency. In retrieval benchmarks, STC improves MRR from 0.3576 to 0.5945 in a hybrid setting and increases Recall@1 from 0.366 to 0.754 in BM25-only retrieval. These results demonstrate that preserving structure during chunking improves retrieval performance, highlighting the importance of structure-aware chunking for RAG over tabular data. 

---
# Retrieval-Augmented Reasoning for Chartered Accountancy 

**Authors**: Jatin Gupta, Akhil Sharma, Saransh Singhania, Ali Imam Abidi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00257)  

**Abstract**: The inception of Large Language Models (LLMs) has catalyzed AI adoption in the finance sector, yet their reliability in complex, jurisdiction-specific tasks like Indian Chartered Accountancy (CA) remains limited. The models display difficulty in executing numerical tasks which require multiple steps while also needing advanced knowledge about legal regulations and the method of scaling their operations is not feasible in settings which have limited access to resources. We present CA-ThinkFlow as a parameter-efficient Retrieval-Augmented Generation (RAG) framework which operates with a 14B, 4-bit-quantized reasoning model, 14B-DeepSeek-R1, and a layout-aware Docling extraction system which maintains document structure during extraction. CA-ThinkFlow uses a basic RAG method which automatically adds retrieved information into the prompt, while it depends on the model's built-in Chain-of-Thought (CoT) functions to create context and produce correct answers. The system we developed system operates at performance levels which match large proprietary models when we tested it on the multi-level CA-Ben benchmark, achieving Scholastic Reliability Coefficient (SRC) results which equal 68.75\% of GPT-4o and Claude 3.5 Sonnet. The framework shows high efficiency and strength in handling parameters, but essential reasoning abilities fail to process complex regulatory texts which exist in fields such as Taxation. 

---
# RSAT: Structured Attribution Makes Small Language Models Faithful Table Reasoners 

**Authors**: Jugal Gajjar, Kamalasankari Subramaniakuppusamy  

**Link**: [PDF](https://arxiv.org/pdf/2605.00199)  

**Abstract**: When a language model answers a table question, users have no way to verify which cells informed which reasoning steps. We introduce RSAT, a method that trains small language models (SLMs, 1-8B) to produce step-by-step reasoning with cell-level citations grounded in table evidence. Phase 1 (SFT) teaches a structured JSON output format from verified reasoning traces. Phase 2 (GRPO) optimizes a composite reward centered on NLI-based faithfulness, alongside citation validity and parsimony. Across six models from two families-Qwen 2.5 (1.5B/3B/7B) and Llama 3 (1B/3B/8B)-RSAT improves faithfulness 3.7$\times$ over SFT alone (0.224$\rightarrow$0.826), with near-perfect citation validity (0.992). Post-hoc attribution collapses below 13% format success, confirming that attribution must be integrated into reasoning, not retrofitted. Ablations show the faithfulness reward is essential: removing it drops faithfulness from 0.97 to 0.03. 

---
# DeGenTWeb: A First Look at LLM-dominant Websites 

**Authors**: Sichang Steven He, Calvin Ardi, Ramesh Govindan, Harsha V. Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2605.00087)  

**Abstract**: Many recent news reports have claimed that content generated by large language models (LLMs) is taking over the web. However, these claims are typically not based on a representative sample of the web and the methodology underlying them is often opaque. Moreover, when aiming to minimize the chances of falsely attributing human-authored content to LLMs, we find that detectors of LLM-generated text perform much worse than advertised. Consequently, we lack an understanding of the true prevalence and characteristics of LLM content on the web.
We describe DeGenTWeb which systematically identifies LLM-dominant websites: sites whose content has been generated using LLMs with little human input. We show how to adapt detectors of LLM-generated text for use on web pages, and how to aggregate detection results from multiple pages on a site for accurate site-level categorization. Using DeGenTWeb, we find that LLM-dominant sites are highly prevalent both in data from Common Crawl and in Bing's search results, and that this share is growing over time. We also show that continuing to accurately identify such sites appears challenging given the capabilities of the latest LLMs. 

---
