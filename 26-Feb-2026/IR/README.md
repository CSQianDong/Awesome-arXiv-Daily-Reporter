# Learning to Collaborate via Structures: Cluster-Guided Item Alignment for Federated Recommendation 

**Authors**: Yuchun Tu, Zhiwei Li, Bingli Sun, Yixuan Li, Xiao Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.21957)  

**Abstract**: Federated recommendation facilitates collaborative model training across distributed clients while keeping sensitive user interaction data local. Conventional approaches typically rely on synchronizing high-dimensional item representations between the server and clients. This paradigm implicitly assumes that precise geometric alignment of embedding coordinates is necessary for collaboration across clients. We posit that establishing relative semantic relationships among items is more effective than enforcing shared representations. Specifically, global semantic relations serve as structural constraints for items. Within these constraints, the framework allows item representations to vary locally on each client, which flexibility enables the model to capture fine-grained user personalization while maintaining global consistency. To this end, we propose Cluster-Guided FedRec framework (CGFedRec), a framework that transforms uploaded embeddings into compact cluster labels. In this framework, the server functions as a global structure discoverer to learn item clusters and distributes only the resulting labels. This mechanism explicitly cuts off the downstream transmission of item embeddings, relieving clients from maintaining global shared item embeddings. Consequently, CGFedRec achieves the effective injection of global collaborative signals into local item representations without transmitting full embeddings. Extensive experiments demonstrate that our approach significantly improves communication efficiency while maintaining superior recommendation accuracy across multiple datasets. 

---
# Offline Reasoning for Efficient Recommendation: LLM-Empowered Persona-Profiled Item Indexing 

**Authors**: Deogyong Kim, Junseong Lee, Jeongeun Lee, Changhoe Kim, Junguel Lee, Jungseok Lee, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.21756)  

**Abstract**: Recent advances in large language models (LLMs) offer new opportunities for recommender systems by capturing the nuanced semantics of user interests and item characteristics through rich semantic understanding and contextual reasoning. In particular, LLMs have been employed as rerankers that reorder candidate items based on inferred user-item relevance. However, these approaches often require expensive online inference-time reasoning, leading to high latency that hampers real-world deployment. In this work, we introduce Persona4Rec, a recommendation framework that performs offline reasoning to construct interpretable persona representations of items, enabling lightweight and scalable real-time inference. In the offline stage, Persona4Rec leverages LLMs to reason over item reviews, inferring diverse user motivations that explain why different types of users may engage with an item; these inferred motivations are materialized as persona representations, providing multiple, human-interpretable views of each item. Unlike conventional approaches that rely on a single item representation, Persona4Rec learns to align user profiles with the most plausible item-side persona through a dedicated encoder, effectively transforming user-item relevance into user-persona relevance. At the online stage, this persona-profiled item index allows fast relevance computation without invoking expensive LLM reasoning. Extensive experiments show that Persona4Rec achieves performance comparable to recent LLM-based rerankers while substantially reducing inference time. Moreover, qualitative analysis confirms that persona representations not only drive efficient scoring but also provide intuitive, review-grounded explanations. These results demonstrate that Persona4Rec offers a practical and interpretable solution for next-generation recommender systems. 

---
# Trie-Aware Transformers for Generative Recommendation 

**Authors**: Zhenxiang Xu, Jiawei Chen, Sirui Chen, Yong He, Jieyu Yang, Chuan Yuan, Ke Ding, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21677)  

**Abstract**: Generative recommendation (GR) aligns with advances in generative AI by casting next-item prediction as token-level generation rather than score-based ranking. Most GR methods adopt a two-stage pipeline: (i) \textit{item tokenization}, which maps each item to a sequence of discrete, hierarchically organized tokens; and (ii) \textit{autoregressive generation}, which predicts the next item's tokens conditioned on the tokens of user's interaction history. Although hierarchical tokenization induces a prefix tree (trie) over items, standard autoregressive modeling with conventional Transformers often flattens item tokens into a linear stream and overlooks the underlying topology.
To address this, we propose TrieRec, a trie-aware generative recommendation method that augments Transformers with structural inductive biases via two positional encodings. First, a \textit{trie-aware absolute positional encoding} aggregates a token's (node's) local structural context (\eg depth, ancestors, and descendants) into the token representation. Second, a \textit{topology-aware relative positional encoding} injects pairwise structural relations into self-attention to capture topology-induced semantic relatedness. TrieRec is also model-agnostic, efficient, and hyperparameter-free. In our experiments, we implement TrieRec within three representative GR backbones, achieving notably improvements of 8.83\% on average across four real-world datasets. 

---
# AQR-HNSW: Accelerating Approximate Nearest Neighbor Search via Density-aware Quantization and Multi-stage Re-ranking 

**Authors**: Ganap Ashit Tewary, Nrusinga Charan Gantayat, Jeff Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21600)  

**Abstract**: Approximate Nearest Neighbor (ANN) search has become fundamental to modern AI infrastructure, powering recommendation systems, search engines, and large language models across industry leaders from Google to OpenAI. Hierarchical Navigable Small World (HNSW) graphs have emerged as the dominant ANN algorithm, widely adopted in production systems due to their superior recall versus latency balance. However, as vector databases scale to billions of embeddings, HNSW faces critical bottlenecks: memory consumption expands, distance computation overhead dominates query latency, and it suffers suboptimal performance on heterogeneous data distributions. This paper presents Adaptive Quantization and Rerank HNSW (AQR-HNSW), a novel framework that synergistically integrates three strategies to enhance HNSW scalability. AQR-HNSW introduces (1) density-aware adaptive quantization, achieving 4x compression while preserving distance relationships; (2) multi-state re-ranking that reduces unnecessary computations by 35%; and (3) quantization-optimized SIMD implementations delivering 16-64 operations per cycle across architectures. Evaluation on standard benchmarks demonstrates 2.5-3.3x higher queries per second (QPS) than state-of-the-art HNSW implementations while maintaining over 98% recall, with 75% memory reduction for the index graph and 5x faster index construction. 

---
# Retrieval Challenges in Low-Resource Public Service Information: A Case Study on Food Pantry Access 

**Authors**: Touseef Hasan, Laila Cure, Souvika Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2602.21598)  

**Abstract**: Public service information systems are often fragmented, inconsistently formatted, and outdated. These characteristics create low-resource retrieval environments that hinder timely access to critical services. We investigate retrieval challenges in such settings through the domain of food pantry access, a socially urgent problem given persistent food insecurity. We develop an AI-powered conversational retrieval system that scrapes and indexes publicly available pantry data and employs a Retrieval-Augmented Generation (RAG) pipeline to support natural language queries via a web interface. We conduct a pilot evaluation study using community-sourced queries to examine system behavior in realistic scenarios. Our analysis reveals key limitations in retrieval robustness, handling underspecified queries, and grounding over inconsistent knowledge bases. This ongoing work exposes fundamental IR challenges in low-resource environments and motivates future research on robust conversational retrieval to improve access to critical public resources. 

---
# Revisiting RAG Retrievers: An Information Theoretic Benchmark 

**Authors**: Wenqing Zheng, Dmitri Kalaev, Noah Fatsi, Daniel Barcklow, Owen Reinert, Igor Melnyk, Senthil Kumar, C. Bayan Bruss  

**Link**: [PDF](https://arxiv.org/pdf/2602.21553)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems rely critically on the retriever module to surface relevant context for large language models. Although numerous retrievers have recently been proposed, each built on different ranking principles such as lexical matching, dense embeddings, or graph citations, there remains a lack of systematic understanding of how these mechanisms differ and overlap. Existing benchmarks primarily compare entire RAG pipelines or introduce new datasets, providing little guidance on selecting or combining retrievers themselves. Those that do compare retrievers directly use a limited set of evaluation tools which fail to capture complementary and overlapping strengths. This work presents MIGRASCOPE, a Mutual Information based RAG Retriever Analysis Scope. We revisit state-of-the-art retrievers and introduce principled metrics grounded in information and statistical estimation theory to quantify retrieval quality, redundancy, synergy, and marginal contribution. We further show that if chosen carefully, an ensemble of retrievers outperforms any single retriever. We leverage the developed tools over major RAG corpora to provide unique insights on contribution levels of the state-of-the-art retrievers. Our findings provide a fresh perspective on the structure of modern retrieval techniques and actionable guidance for designing robust and efficient RAG systems. 

---
# Revisiting Text Ranking in Deep Research 

**Authors**: Chuan Meng, Litu Ou, Sean MacAvaney, Jeff Dalton  

**Link**: [PDF](https://arxiv.org/pdf/2602.21456)  

**Abstract**: Deep research has emerged as an important task that aims to address hard queries through extensive open-web exploration. To tackle it, most prior work equips large language model (LLM)-based agents with opaque web search APIs, enabling agents to iteratively issue search queries, retrieve external evidence, and reason over it. Despite search's essential role in deep research, black-box web search APIs hinder systematic analysis of search components, leaving the behaviour of established text ranking methods in deep research largely unclear. To fill this gap, we reproduce a selection of key findings and best practices for IR text ranking methods in the deep research setting. In particular, we examine their effectiveness from three perspectives: (i) retrieval units (documents vs. passages), (ii) pipeline configurations (different retrievers, re-rankers, and re-ranking depths), and (iii) query characteristics (the mismatch between agent-issued queries and the training queries of text rankers). We perform experiments on BrowseComp-Plus, a deep research dataset with a fixed corpus, evaluating 2 open-source agents, 5 retrievers, and 3 re-rankers across diverse setups. We find that agent-issued queries typically follow web-search-style syntax (e.g., quoted exact matches), favouring lexical, learned sparse, and multi-vector retrievers; passage-level units are more efficient under limited context windows, and avoid the difficulties of document length normalisation in lexical retrieval; re-ranking is highly effective; translating agent-issued queries into natural-language questions significantly bridges the query mismatch. 

---
# LiCQA : A Lightweight Complex Question Answering System 

**Authors**: Sourav Saha, Dwaipayan Roy, Mandar Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2602.22182)  

**Abstract**: Over the last twenty years, significant progress has been made in designing and implementing Question Answering (QA) systems. However, addressing complex questions, the answers to which are spread across multiple documents, remains a challenging problem. Recent QA systems that are designed to handle complex questions work either on the basis of knowledge graphs, or utilise contem- porary neural models that are expensive to train, in terms of both computational resources and the volume of training data required. In this paper, we present LiCQA, an unsupervised question answer- ing model that works primarily on the basis of corpus evidence. We empirically compare the effectiveness and efficiency of LiCQA with two recently presented QA systems, which are based on different underlying principles. The results of our experiments show that LiCQA significantly outperforms these two state-of-the-art systems on benchmark data with noteworthy reduction in latency. 

---
# Enhancing Multilingual Embeddings via Multi-Way Parallel Text Alignment 

**Authors**: Barah Fazili, Koustava Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2602.21543)  

**Abstract**: Multilingual pretraining typically lacks explicit alignment signals, leading to suboptimal cross-lingual alignment in the representation space. In this work, we show that training standard pretrained models for cross-lingual alignment with a multi-way parallel corpus in a diverse pool of languages can substantially improve multilingual and cross-lingual representations for NLU tasks. We construct a multi-way parallel dataset using translations of English text from an off-the-shelf NMT model for a pool of six target languages and achieve strong cross-lingual alignment through contrastive learning. This leads to substantial performance gains across both seen and unseen languages for multiple tasks from the MTEB benchmark evaluated for XLM-Roberta and multilingual BERT base models. Using a multi-way parallel corpus for contrastive training yields substantial gains on bitext mining (21.3%), semantic similarity (5.3%), and classification (28.4%) compared to English-centric (En-X) bilingually parallel data, where X is sampled from a pool of multiple target languages. Furthermore, finetuning mE5 model on a small dataset with multi-way parallelism significantly improves bitext mining compared to one without, underscoring the importance of multi-way cross-lingual supervision even for models already pretrained for high-quality sentence embeddings. 

---
# Both Ends Count! Just How Good are LLM Agents at "Text-to-Big SQL"? 

**Authors**: Germán T. Eizaguirre, Lars Tissen, Marc Sánchez-Artigas  

**Link**: [PDF](https://arxiv.org/pdf/2602.21480)  

**Abstract**: Text-to-SQL and Big Data are both extensively benchmarked fields, yet there is limited research that evaluates them jointly. In the real world, Text-to-SQL systems are often embedded with Big Data workflows, such as large-scale data processing or interactive data analytics. We refer to this as "Text-to-Big SQL". However, existing text-to-SQL benchmarks remain narrowly scoped and overlook the cost and performance implications that arise at scale. For instance, translation errors that are minor on small datasets lead to substantial cost and latency overheads as data scales, a relevant issue completely ignored by text-to-SQL metrics.
In this paper, we overcome this overlooked challenge by introducing novel and representative metrics for evaluating Text-to-Big SQL. Our study focuses on production-level LLM agents, a database-agnostic system adaptable to diverse user needs. Via an extensive evaluation of frontier models, we show that text-to-SQL metrics are insufficient for Big Data. In contrast, our proposed text-to-Big SQL metrics accurately reflect execution efficiency, cost, and the impact of data scale. Furthermore, we provide LLM-specific insights, including fine-grained, cross-model comparisons of latency and cost. 

---
# A Hierarchical Multi-Agent System for Autonomous Discovery in Geoscientific Data Archives 

**Authors**: Dmitrii Pantiukhin, Ivan Kuznetsov, Boris Shapkin, Antonia Anna Jost, Thomas Jung, Nikolay Koldunov  

**Link**: [PDF](https://arxiv.org/pdf/2602.21351)  

**Abstract**: The rapid accumulation of Earth science data has created a significant scalability challenge; while repositories like PANGAEA host vast collections of datasets, citation metrics indicate that a substantial portion remains underutilized, limiting data reusability. Here we present PANGAEA-GPT, a hierarchical multi-agent framework designed for autonomous data discovery and analysis. Unlike standard Large Language Model (LLM) wrappers, our architecture implements a centralized Supervisor-Worker topology with strict data-type-aware routing, sandboxed deterministic code execution, and self-correction via execution feedback, enabling agents to diagnose and resolve runtime errors. Through use-case scenarios spanning physical oceanography and ecology, we demonstrate the system's capacity to execute complex, multi-step workflows with minimal human intervention. This framework provides a methodology for querying and analyzing heterogeneous repository data through coordinated agent workflows. 

---
# PiPNN: Ultra-Scalable Graph-Based Nearest Neighbor Indexing 

**Authors**: Tobias Rubel, Richard Wen, Laxman Dhulipala, Lars Gottesbüren, Rajesh Jayaram, Jakub Łącki  

**Link**: [PDF](https://arxiv.org/pdf/2602.21247)  

**Abstract**: The fastest indexes for Approximate Nearest Neighbor Search today are also the slowest to build: graph-based methods like HNSW and Vamana achieve state-of-the-art query performance but have large construction times due to relying on random-access-heavy beam searches. We introduce PiPNN (Pick-in-Partitions Nearest Neighbors), an ultra-scalable graph construction algorithm that avoids this ``search bottleneck'' that existing graph-based methods suffer from.
PiPNN's core innovation is HashPrune, a novel online pruning algorithm which dynamically maintains sparse collections of edges. HashPrune enables PiPNN to partition the dataset into overlapping sub-problems, efficiently perform bulk distance comparisons via dense matrix multiplication kernels, and stream a subset of the edges into HashPrune. HashPrune guarantees bounded memory during index construction which permits PiPNN to build higher quality indices without the use of extra intermediate memory.
PiPNN builds state-of-the-art indexes up to 11.6x faster than Vamana (DiskANN) and up to 12.9x faster than HNSW. PiPNN is significantly more scalable than recent algorithms for fast graph construction. PiPNN builds indexes at least 19.1x faster than MIRAGE and 17.3x than FastKCNA while producing indexes that achieve higher query throughput. PiPNN enables us to build, for the first time, high-quality ANN indexes on billion-scale datasets in under 20 minutes using a single multicore machine. 

---
# Toward Effective Multi-Domain Rumor Detection in Social Networks Using Domain-Gated Mixture-of-Experts 

**Authors**: Mohadeseh Sheikhqoraei, Zainabolhoda Heshmati, Zeinab Rajabi, Leila Rabiei  

**Link**: [PDF](https://arxiv.org/pdf/2602.21214)  

**Abstract**: Social media platforms have become key channels for spreading and tracking rumors due to their widespread accessibility and ease of information sharing. Rumors can continuously emerge across diverse domains and topics, often with the intent to mislead society for personal or commercial gain. Therefore, developing methods that can accurately detect rumors at early stages is crucial to mitigating their negative impact. While existing approaches often specialize in single-domain detection, their performance degrades when applied to new domains due to shifts in data distribution, such as lexical patterns and propagation dynamics. To bridge this gap, this study introduces PerFact, a large-scale multi-domain rumor dataset comprising 8,034 annotated posts from the X platform, annotated into two primary categories: rumor (including true, false, and unverified rumors) and non-rumor. Annotator agreement, measured via Fleiss' Kappa ($\kappa = 0.74$), ensures high-quality labels.
This research further proposes an effective multi-domain rumor detection model that employs a domain gate to dynamically aggregate multiple feature representations extracted through a Mixture-of-Experts method. Each expert combines CNN and BiLSTM networks to capture local syntactic features and long-range contextual dependencies. By leveraging both textual content and publisher information, the proposed model classifies posts into rumor and non-rumor categories with high accuracy. Evaluations demonstrate state-of-the-art performance, achieving an F1-score of 79.86\% and an accuracy of 79.98\% in multi-domain settings.
Keywords: Rumor Detection, Multi-Domain, Natural Language Processing, Social Networks, Mixture-of-Experts Model 

---
# Disaster Question Answering with LoRA Efficiency and Accurate End Position 

**Authors**: Takato Yasuno  

**Link**: [PDF](https://arxiv.org/pdf/2602.21212)  

**Abstract**: Natural disasters such as earthquakes, torrential rainfall, floods, and volcanic eruptions occur with extremely low frequency and affect limited geographic areas. When individuals face disaster situations, they often experience confusion and lack the domain-specific knowledge and experience necessary to determine appropriate responses and actions. While disaster information is continuously updated, even when utilizing RAG search and large language models for inquiries, obtaining relevant domain knowledge about natural disasters and experiences similar to one's specific situation is not guaranteed. When hallucinations are included in disaster question answering, artificial misinformation may spread and exacerbate confusion. This work introduces a disaster-focused question answering system based on Japanese disaster situations and response experiences. Utilizing the cl-tohoku/bert-base-japanese-v3 + Bi-LSTM + Enhanced Position Heads architecture with LoRA efficiency optimization, we achieved 70.4\% End Position accuracy with only 5.7\% of the total parameters (6.7M/117M). Experimental results demonstrate that the combination of Japanese BERT-base optimization and Bi-LSTM contextual understanding achieves accuracy levels suitable for real disaster response scenarios, attaining a 0.885 Span F1 score. Future challenges include: establishing natural disaster Q\&A benchmark datasets, fine-tuning foundation models with disaster knowledge, developing lightweight and power-efficient edge AI Disaster Q\&A applications for situations with insufficient power and communication during disasters, and addressing disaster knowledge base updates and continual learning capabilities. 

---
