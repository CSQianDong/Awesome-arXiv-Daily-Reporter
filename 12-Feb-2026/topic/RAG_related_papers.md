# Campaign-2-PT-RAG: LLM-Guided Semantic Product Type Attribution for Scalable Campaign Ranking 

**Authors**: Yiming Che, Mansi Mane, Keerthi Gopalakrishnan, Parisa Kaghazgaran, Murali Mohana Krishna Dandu, Archana Venkatachalapathy, Sinduja Subramaniam, Yokila Arora, Evren Korpeoglu, Sushant Kumar, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2602.10577)  

**Abstract**: E-commerce campaign ranking models require large-scale training labels indicating which users purchased due to campaign influence. However, generating these labels is challenging because campaigns use creative, thematic language that does not directly map to product purchases. Without clear product-level attribution, supervised learning for campaign optimization remains limited. We present \textbf{Campaign-2-PT-RAG}, a scalable label generation framework that constructs user--campaign purchase labels by inferring which product types (PTs) each campaign promotes. The framework first interprets campaign content using large language models (LLMs) to capture implicit intent, then retrieves candidate PTs through semantic search over the platform taxonomy. A structured LLM-based classifier evaluates each PT's relevance, producing a campaign-specific product coverage set. User purchases matching these PTs generate positive training labels for downstream ranking models. This approach reframes the ambiguous attribution problem into a tractable semantic alignment task, enabling scalable and consistent supervision for downstream tasks such as campaign ranking optimization in production e-commerce environments. Experiments on internal and synthetic datasets, validated against expert-annotated campaign--PT mappings, show that our LLM-assisted approach generates high-quality labels with 78--90% precision while maintaining over 99% recall. 

---
# MLDocRAG: Multimodal Long-Context Document Retrieval Augmented Generation 

**Authors**: Yongyue Zhang, Yaxiong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10271)  

**Abstract**: Understanding multimodal long-context documents that comprise multimodal chunks such as paragraphs, figures, and tables is challenging due to (1) cross-modal heterogeneity to localize relevant information across modalities, (2) cross-page reasoning to aggregate dispersed evidence across pages. To address these challenges, we are motivated to adopt a query-centric formulation that projects cross-modal and cross-page information into a unified query representation space, with queries acting as abstract semantic surrogates for heterogeneous multimodal content. In this paper, we propose a Multimodal Long-Context Document Retrieval Augmented Generation (MLDocRAG) framework that leverages a Multimodal Chunk-Query Graph (MCQG) to organize multimodal document content around semantically rich, answerable queries. MCQG is constructed via a multimodal document expansion process that generates fine-grained queries from heterogeneous document chunks and links them to their corresponding content across modalities and pages. This graph-based structure enables selective, query-centric retrieval and structured evidence aggregation, thereby enhancing grounding and coherence in long-context multimodal question answering. Experiments on datasets MMLongBench-Doc and LongDocURL demonstrate that MLDocRAG consistently improves retrieval quality and answer accuracy, demonstrating its effectiveness for long-context multimodal understanding. 

---
