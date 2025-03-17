# AIstorian lets AI be a historian: A KG-powered multi-agent system for accurate biography generation 

**Authors**: Fengyu Li, Yilin Li, Junhao Zhu, Lu Chen, Yanfei Zhang, Jia Zhou, Hui Zu, Jingwen Zhao, Yunjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.11346)  

**Abstract**: Huawei has always been committed to exploring the AI application in historical research. Biography generation, as a specialized form of abstractive summarization, plays a crucial role in historical research but faces unique challenges that existing large language models (LLMs) struggle to address. These challenges include maintaining stylistic adherence to historical writing conventions, ensuring factual fidelity, and handling fragmented information across multiple documents. We present AIstorian, a novel end-to-end agentic system featured with a knowledge graph (KG)-powered retrieval-augmented generation (RAG) and anti-hallucination multi-agents. Specifically, AIstorian introduces an in-context learning based chunking strategy and a KG-based index for accurate and efficient reference retrieval. Meanwhile, AIstorian orchestrates multi-agents to conduct on-the-fly hallucination detection and error-type-aware correction. Additionally, to teach LLMs a certain language style, we finetune LLMs based on a two-step training approach combining data augmentation-enhanced supervised fine-tuning with stylistic preference optimization. Extensive experiments on a real-life historical Jinshi dataset demonstrate that AIstorian achieves a 3.8x improvement in factual accuracy and a 47.6% reduction in hallucination rate compared to existing baselines. The data and code are available at: this https URL. 

---
# AttentionRAG: Attention-Guided Context Pruning in Retrieval-Augmented Generation 

**Authors**: Yixiong Fang, Tianran Sun, Yuling Shi, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10720)  

**Abstract**: While RAG demonstrates remarkable capabilities in LLM applications, its effectiveness is hindered by the ever-increasing length of retrieved contexts, which introduces information redundancy and substantial computational overhead. Existing context pruning methods, such as LLMLingua, lack contextual awareness and offer limited flexibility in controlling compression rates, often resulting in either insufficient pruning or excessive information loss. In this paper, we propose AttentionRAG, an attention-guided context pruning method for RAG systems. The core idea of AttentionRAG lies in its attention focus mechanism, which reformulates RAG queries into a next-token prediction paradigm. This mechanism isolates the query's semantic focus to a single token, enabling precise and efficient attention calculation between queries and retrieved contexts. Extensive experiments on LongBench and Babilong benchmarks show that AttentionRAG achieves up to 6.3$\times$ context compression while outperforming LLMLingua methods by around 10\% in key metrics. 

---
# Enhancing Retrieval for ESGLLM via ESG-CID -- A Disclosure Content Index Finetuning Dataset for Mapping GRI and ESRS 

**Authors**: Shafiuddin Rehan Ahmed, Ankit Parag Shah, Quan Hung Tran, Vivek Khetan, Sukryool Kang, Ankit Mehta, Yujia Bao, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.10674)  

**Abstract**: Climate change has intensified the need for transparency and accountability in organizational practices, making Environmental, Social, and Governance (ESG) reporting increasingly crucial. Frameworks like the Global Reporting Initiative (GRI) and the new European Sustainability Reporting Standards (ESRS) aim to standardize ESG reporting, yet generating comprehensive reports remains challenging due to the considerable length of ESG documents and variability in company reporting styles. To facilitate ESG report automation, Retrieval-Augmented Generation (RAG) systems can be employed, but their development is hindered by a lack of labeled data suitable for training retrieval models. In this paper, we leverage an underutilized source of weak supervision -- the disclosure content index found in past ESG reports -- to create a comprehensive dataset, ESG-CID, for both GRI and ESRS standards. By extracting mappings between specific disclosure requirements and corresponding report sections, and refining them using a Large Language Model as a judge, we generate a robust training and evaluation set. We benchmark popular embedding models on this dataset and show that fine-tuning BERT-based models can outperform commercial embeddings and leading public models, even under temporal data splits for cross-report style transfer from GRI to ESRS 

---
# A Survey on Knowledge-Oriented Retrieval-Augmented Generation 

**Authors**: Mingyue Cheng, Yucong Luo, Jie Ouyang, Qi Liu, Huijie Liu, Li Li, Shuo Yu, Bohou Zhang, Jiawei Cao, Jie Ma, Daoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10677)  

**Abstract**: Retrieval-Augmented Generation (RAG) has gained significant attention in recent years for its potential to enhance natural language understanding and generation by combining large-scale retrieval systems with generative models. RAG leverages external knowledge sources, such as documents, databases, or structured data, to improve model performance and generate more accurate and contextually relevant outputs. This survey aims to provide a comprehensive overview of RAG by examining its fundamental components, including retrieval mechanisms, generation processes, and the integration between the two. We discuss the key characteristics of RAG, such as its ability to augment generative models with dynamic external knowledge, and the challenges associated with aligning retrieved information with generative objectives. We also present a taxonomy that categorizes RAG methods, ranging from basic retrieval-augmented approaches to more advanced models incorporating multi-modal data and reasoning capabilities. Additionally, we review the evaluation benchmarks and datasets commonly used to assess RAG systems, along with a detailed exploration of its applications in fields such as question answering, summarization, and information retrieval. Finally, we highlight emerging research directions and opportunities for improving RAG systems, such as enhanced retrieval efficiency, model interpretability, and domain-specific adaptations. This paper concludes by outlining the prospects for RAG in addressing real-world challenges and its potential to drive further advancements in natural language processing. 

---
# Improving RAG Retrieval via Propositional Content Extraction: a Speech Act Theory Approach 

**Authors**: João Alberto de Oliveira Lima  

**Link**: [PDF](https://arxiv.org/pdf/2503.10654)  

**Abstract**: When users formulate queries, they often include not only the information they seek, but also pragmatic markers such as interrogative phrasing or polite requests. Although these speech act indicators communicate the user\textquotesingle s intent -- whether it is asking a question, making a request, or stating a fact -- they do not necessarily add to the core informational content of the query itself. This paper investigates whether extracting the underlying propositional content from user utterances -- essentially stripping away the linguistic markers of intent -- can improve retrieval quality in Retrieval-Augmented Generation (RAG) systems. Drawing upon foundational insights from speech act theory, we propose a practical method for automatically transforming queries into their propositional equivalents before embedding. To assess the efficacy of this approach, we conducted an experimental study involving 63 user queries related to a Brazilian telecommunications news corpus with precomputed semantic embeddings. Results demonstrate clear improvements in semantic similarity between query embeddings and document embeddings at top ranks, confirming that queries stripped of speech act indicators more effectively retrieve relevant content. 

---
# ClaimTrust: Propagation Trust Scoring for RAG Systems 

**Authors**: Hangkai Qian, Bo Li, Qichen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10702)  

**Abstract**: The rapid adoption of retrieval-augmented generation (RAG) systems has revolutionized large-scale content generation but has also highlighted the challenge of ensuring trustworthiness in retrieved information. This paper introduces ClaimTrust, a propagation-based trust scoring framework that dynamically evaluates the reliability of documents in a RAG system. Using a modified PageRank-inspired algorithm, ClaimTrust propagates trust scores across documents based on relationships derived from extracted factual claims. We preprocess and analyze 814 political news articles from Kaggle's Fake News Detection Dataset to extract 2,173 unique claims and classify 965 meaningful relationships (supporting or contradicting). By representing the dataset as a document graph, ClaimTrust iteratively updates trust scores until convergence, effectively differentiating trustworthy articles from unreliable ones. Our methodology, which leverages embedding-based filtering for efficient claim comparison and relationship classification, achieves a 11.2% of significant connections while maintaining computational scalability. Experimental results demonstrate that ClaimTrust successfully assigns higher trust scores to verified documents while penalizing those containing false information. Future directions include fine-tuned claim extract and compare (Li et al., 2022), parameter optimization, enhanced language model utilization, and robust evaluation metrics to generalize the framework across diverse datasets and domains. 

---
# Taxonomic Reasoning for Rare Arthropods: Combining Dense Image Captioning and RAG for Interpretable Classification 

**Authors**: Nathaniel Lesperance, Sujeevan Ratnasingham, Graham W. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2503.10886)  

**Abstract**: In the context of pressing climate change challenges and the significant biodiversity loss among arthropods, automated taxonomic classification from organismal images is a subject of intense research. However, traditional AI pipelines based on deep neural visual architectures such as CNNs or ViTs face limitations such as degraded performance on the long-tail of classes and the inability to reason about their predictions. We integrate image captioning and retrieval-augmented generation (RAG) with large language models (LLMs) to enhance biodiversity monitoring, showing particular promise for characterizing rare and unknown arthropod species. While a naive Vision-Language Model (VLM) excels in classifying images of common species, the RAG model enables classification of rarer taxa by matching explicit textual descriptions of taxonomic features to contextual biodiversity text data from external sources. The RAG model shows promise in reducing overconfidence and enhancing accuracy relative to naive LLMs, suggesting its viability in capturing the nuances of taxonomic hierarchy, particularly at the challenging family and genus levels. Our findings highlight the potential for modern vision-language AI pipelines to support biodiversity conservation initiatives, emphasizing the role of comprehensive data curation and collaboration with citizen science platforms to improve species identification, unknown species characterization and ultimately inform conservation strategies. 

---
