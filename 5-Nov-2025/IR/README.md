# Average Precision at Cutoff k under Random Rankings: Expectation and Variance 

**Authors**: Tetiana Manzhos, Tetiana Ianevych, Olga Melnyk  

**Link**: [PDF](https://arxiv.org/pdf/2511.02571)  

**Abstract**: Recommender systems and information retrieval platforms rely on ranking algorithms to present the most relevant items to users, thereby improving engagement and satisfaction. Assessing the quality of these rankings requires reliable evaluation metrics. Among them, Mean Average Precision at cutoff k (MAP@k) is widely used, as it accounts for both the relevance of items and their positions in the list.
In this paper, the expectation and variance of Average Precision at k (AP@k) are derived since they can be used as biselines for MAP@k. Here, we covered two widely used evaluation models: offline and online. The expectation establishes the baseline, indicating the level of MAP@k that can be achieved by pure chance. The variance complements this baseline by quantifying the extent of random fluctuations, enabling a more reliable interpretation of observed scores. 

---
# KGBridge: Knowledge-Guided Prompt Learning for Non-overlapping Cross-Domain Recommendation 

**Authors**: Yuhan Wang, Qing Xie, Zhifeng Bao, Mengzi Tang, Lin Li, Yongjian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02181)  

**Abstract**: Knowledge Graphs (KGs), as structured knowledge bases that organize relational information across diverse domains, provide a unified semantic foundation for cross-domain recommendation (CDR). By integrating symbolic knowledge with user-item interactions, KGs enrich semantic representations, support reasoning, and enhance model interpretability. Despite this potential, existing KG-based methods still face major challenges in CDR, particularly under non-overlapping user scenarios. These challenges arise from: (C1) sensitivity to KG sparsity and popularity bias, (C2) dependence on overlapping users for domain alignment and (C3) lack of explicit disentanglement between transferable and domain-specific knowledge, which limit effective and stable knowledge transfer. To this end, we propose KGBridge, a knowledge-guided prompt learning framework for cross-domain sequential recommendation under non-overlapping user scenarios. KGBridge comprises two core components: a KG-enhanced Prompt Encoder, which models relation-level semantics as soft prompts to provide structured and dynamic priors for user sequence modeling (addressing C1), and a Two-stage Training Paradigm, which combines cross-domain pretraining and privacy-preserving fine-tuning to enable knowledge transfer without user overlap (addressing C2). By combining relation-aware semantic control with correspondence-driven disentanglement, KGBridge explicitly separates and balances domain-shared and domain-specific semantics, thereby maintaining complementarity and stabilizing adaptation during fine-tuning (addressing C3). Extensive experiments on benchmark datasets demonstrate that KGBridge consistently outperforms state-of-the-art baselines and remains robust under varying KG sparsity, highlighting its effectiveness in mitigating structural imbalance and semantic entanglement in KG-enhanced cross-domain recommendation. 

---
# Enhancing Multimodal Recommendations with Vision-Language Models and Information-Aware Fusion 

**Authors**: Hai-Dang Kieu, Min Xu, Thanh Trung Huynh, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2511.02113)  

**Abstract**: Recent advances in multimodal recommendation (MMR) have shown that incorporating rich content sources such as images and text can lead to significant gains representation quality. However, existing methods often rely on coarse visual features and uncontrolled fusion, leading to redundant or misaligned representations. As a result, visual encoders often fail to capture salient, item-relevant semantics, limiting their contribution in multimodal fusion. From an information-theoretic perspective, effective fusion should balance the unique, shared, and redundant information across modalities, preserving complementary cues while avoiding correlation bias. This paper presents VLIF, a vision-language and information-theoretic fusion framework that enhances multimodal recommendation through two key components. (i) A VLM-based visual enrichment module generates fine-grained, title-guided descriptions to transform product images into semantically aligned representations. (ii) An information-aware fusion module, inspired by Partial Information Decomposition (PID), disentangles redundant and synergistic signals across modalities for controlled integration. Experiments on three Amazon datasets demonstrate that VLIF consistently outperforms recent multimodal baselines and substantially strengthens the contribution of visual features. 

---
# Solving cold start in news recommendations: a RippleNet-based system for large scale media outlet 

**Authors**: Karol Radziszewski, Michał Szpunar, Piotr Ociepka, Mateusz Buczyński  

**Link**: [PDF](https://arxiv.org/pdf/2511.02052)  

**Abstract**: We present a scalable recommender system implementation based on RippleNet, tailored for the media domain with a production deployment in this http URL, one of Poland's largest online media platforms. Our solution addresses the cold-start problem for newly published content by integrating content-based item embeddings into the knowledge propagation mechanism of RippleNet, enabling effective scoring of previously unseen items. The system architecture leverages Amazon SageMaker for distributed training and inference, and Apache Airflow for orchestrating data pipelines and model retraining workflows. To ensure high-quality training data, we constructed a comprehensive golden dataset consisting of user and item features and a separate interaction table, all enabling flexible extensions and integration of new signals. 

---
# Beyond Single Embeddings: Capturing Diverse Targets with Multi-Query Retrieval 

**Authors**: Hung-Ting Chen, Xiang Liu, Shauli Ravfogel, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.02770)  

**Abstract**: Most text retrievers generate \emph{one} query vector to retrieve relevant documents. Yet, the conditional distribution of relevant documents for the query may be multimodal, e.g., representing different interpretations of the query. We first quantify the limitations of existing retrievers. All retrievers we evaluate struggle more as the distance between target document embeddings grows. To address this limitation, we develop a new retriever architecture, \emph{A}utoregressive \emph{M}ulti-\emph{E}mbedding \emph{R}etriever (AMER). Our model autoregressively generates multiple query vectors, and all the predicted query vectors are used to retrieve documents from the corpus. We show that on the synthetic vectorized data, the proposed method could capture multiple target distributions perfectly, showing 4x better performance than single embedding model. We also fine-tune our model on real-world multi-answer retrieval datasets and evaluate in-domain. AMER presents 4 and 21\% relative gains over single-embedding baselines on two datasets we evaluate on. Furthermore, we consistently observe larger gains on the subset of dataset where the embeddings of the target documents are less similar to each other. We demonstrate the potential of using a multi-query vector retriever and open up a new direction for future work. 

---
# Relational Deep Dive: Error-Aware Queries Over Unstructured Data 

**Authors**: Daren Chao, Kaiwen Chen, Naiqing Guan, Nick Koudas  

**Link**: [PDF](https://arxiv.org/pdf/2511.02711)  

**Abstract**: Unstructured data is pervasive, but analytical queries demand structured representations, creating a significant extraction challenge. Existing methods like RAG lack schema awareness and struggle with cross-document alignment, leading to high error rates. We propose ReDD (Relational Deep Dive), a framework that dynamically discovers query-specific schemas, populates relational tables, and ensures error-aware extraction with provable guarantees. ReDD features a two-stage pipeline: (1) Iterative Schema Discovery (ISD) identifies minimal, joinable schemas tailored to each query, and (2) Tabular Data Population (TDP) extracts and corrects data using lightweight classifiers trained on LLM hidden states. A main contribution of ReDD is SCAPE, a statistically calibrated method for error detection with coverage guarantees, and SCAPE-HYB, a hybrid approach that optimizes the trade-off between accuracy and human correction costs. Experiments across diverse datasets demonstrate ReDD's effectiveness, reducing data extraction errors from up to 30% to below 1% while maintaining high schema completeness (100% recall) and precision. ReDD's modular design enables fine-grained control over accuracy-cost trade-offs, making it a robust solution for high-stakes analytical queries over unstructured corpora. 

---
# Let Multimodal Embedders Learn When to Augment Query via Adaptive Query Augmentation 

**Authors**: Wongyu Kim, Hochang Lee, Sanghak Lee, Yoonsung Kim, Jaehyun Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.02358)  

**Abstract**: Query augmentation makes queries more meaningful by appending further information to the queries to find relevant documents. Current studies have proposed Large Language Model (LLM)-based embedders, which learn representation for embedding and generation for query augmentation in a multi-task manner by leveraging the generative capabilities of LLM. During inference, these jointly trained embedders have conducted query augmentation followed by embedding, showing effective results. However, augmenting every query leads to substantial embedding latency and query augmentation can be detrimental to performance for some queries. Also, previous methods have not been explored in multimodal environments. To tackle these problems, we propose M-Solomon, a universal multimodal embedder that can adaptively determine when to augment queries. Our approach first divides the queries of the training datasets into two groups at the dataset level. One includes queries that require augmentation and the other includes queries that do not. Then, we introduces a synthesis process that generates appropriate augmentations for queries that require them by leveraging a powerful Multimodal LLM (MLLM). Next, we present adaptive query augmentation. Through this step, M-Solomon can conduct query augmentation only when necessary by learning to generate synthetic augmentations with the prefix /augment for queries that demand them and to generate the simple string /embed for others. Experimental results showed that M-Solomon not only surpassed the baseline without augmentation by a large margin but also outperformed the baseline that always used augmentation, providing much faster embedding latency. 

---
# Library and Culture: A Scientometric Analysis and Visualization of Research Trends 

**Authors**: Auwalu Abdullahi Umar, Muneer Ahmad, Dr M Sadik Batcha  

**Link**: [PDF](https://arxiv.org/pdf/2511.02296)  

**Abstract**: The significance of libraries in preserving and maintaining history and traditional culture cannot be overlooked. It is from this purpose that libraries are to envisage in their programmes cultural activities which must be collected, documented and preserved for posterity. The usefulness of preserved information lies in the fact that the generation to come will be able to establish their identity. This will also assist them with a foundation to build from. This study focus on the growth and development of Library and Culture research in forms of publications reflected in Web of Science database, during the span of 2010-2019. A total 890 publications were found and the highest 124 (13.93%) publications published in this http URL analysis maps comprehensively the parameters of total output, growth of output, authorship, institution wise and country-level collaboration patterns, major contributors (individuals, top publication sources, institutions, and countries). It exposed that the most prolific author is Lo P secured first place by contributing 4 (0.45%) publications, followed by Bressan V 3 (0.34%) publications in Library and Culture literature. Journal of Academic Librarianship produced the highest number of records 29 (3.26%) followed by Australian Library Journal having contributed 21 (2.36%).It is identified the domination of Wuhan University; School Information Management had contributed 6 (0.67%) of total research output. Authors from USA published the highest number of publications with a total of 244 (27.42%), followed by UK and Australia with 118 (13.26%) and 76 (8.54%) publications were produced respectively. 

---
# Research Output on Alopecia Areata Disease: A Scientometric Analysis of Publications from 2010 to 2019 

**Authors**: Muneer Ahmad, M Sadik Batcha  

**Link**: [PDF](https://arxiv.org/pdf/2511.02275)  

**Abstract**: The present study is undertaken to find out the publication trends on Alopecia Areata Disease during 2010-2019 from the global perspective. The study mainly focus on distribution of research output, top journals for publications, most prolific authors, authorship pattern, and citations pattern on Alopecia Areata Disease. The results indicate that highest growth rate of publications occurred during the year 2019. Columbia University topped the scene among all institutes. The maximum publications were more than four authored publications. Christiano AM and Clynes R were found to be the most prolific authors. It is also found that most of the prolific authors (by number of publications) do appear in highly cited publications list. Alopecia Areata Disease researchers mostly preferred using article publications to communicate their findings. 

---
# InteracSPARQL: An Interactive System for SPARQL Query Refinement Using Natural Language Explanations 

**Authors**: Xiangru Jian, Zhengyuan Dong, M. Tamer Özsu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02002)  

**Abstract**: In recent years, querying semantic web data using SPARQL has remained challenging, especially for non-expert users, due to the language's complex syntax and the prerequisite of understanding intricate data structures. To address these challenges, we propose InteracSPARQL, an interactive SPARQL query generation and refinement system that leverages natural language explanations (NLEs) to enhance user comprehension and facilitate iterative query refinement. InteracSPARQL integrates LLMs with a rule-based approach to first produce structured explanations directly from SPARQL abstract syntax trees (ASTs), followed by LLM-based linguistic refinements. Users can interactively refine queries through direct feedback or LLM-driven self-refinement, enabling the correction of ambiguous or incorrect query components in real time. We evaluate InteracSPARQL on standard benchmarks, demonstrating significant improvements in query accuracy, explanation clarity, and overall user satisfaction compared to baseline approaches. Our experiments further highlight the effectiveness of combining rule-based methods with LLM-driven refinements to create more accessible and robust SPARQL interfaces. 

---
