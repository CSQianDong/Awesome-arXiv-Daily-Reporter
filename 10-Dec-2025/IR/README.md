# VI-MMRec: Similarity-Aware Training Cost-free Virtual User-Item Interactions for Multimodal Recommendation 

**Authors**: Jinfeng Xu, Zheyu Chen, Shuo Yang, Jinze Li, Zitong Wan, Hewei Wang, Weijie Liu, Yijie Li, Edith C. H. Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2512.08702)  

**Abstract**: Although existing multimodal recommendation models have shown promising performance, their effectiveness continues to be limited by the pervasive data sparsity problem. This problem arises because users typically interact with only a small subset of available items, leading existing models to arbitrarily treat unobserved items as negative samples. To this end, we propose VI-MMRec, a model-agnostic and training cost-free framework that enriches sparse user-item interactions via similarity-aware virtual user-item interactions. These virtual interactions are constructed based on modality-specific feature similarities of user-interacted items. Specifically, VI-MMRec introduces two different strategies: (1) Overlay, which independently aggregates modality-specific similarities to preserve modality-specific user preferences, and (2) Synergistic, which holistically fuses cross-modal similarities to capture complementary user preferences. To ensure high-quality augmentation, we design a statistically informed weight allocation mechanism that adaptively assigns weights to virtual user-item interactions based on dataset-specific modality relevance. As a plug-and-play framework, VI-MMRec seamlessly integrates with existing models to enhance their performance without modifying their core architecture. Its flexibility allows it to be easily incorporated into various existing models, maximizing performance with minimal implementation effort. Moreover, VI-MMRec introduces no additional overhead during training, making it significantly advantageous for practical deployment. Comprehensive experiments conducted on six real-world datasets using seven state-of-the-art multimodal recommendation models validate the effectiveness of our VI-MMRec. 

---
# Ontology-Based Knowledge Graph Framework for Industrial Standard Documents via Hierarchical and Propositional Structuring 

**Authors**: Jiin Park, Hyuna Jeon, Yoonseo Lee, Jisu Hong, Misuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.08398)  

**Abstract**: Ontology-based knowledge graph (KG) construction is a core technology that enables multidimensional understanding and advanced reasoning over domain knowledge. Industrial standards, in particular, contain extensive technical information and complex rules presented in highly structured formats that combine tables, scopes of application, constraints, exceptions, and numerical calculations, making KG construction especially challenging. In this study, we propose a method that organizes such documents into a hierarchical semantic structure, decomposes sentences and tables into atomic propositions derived from conditional and numerical rules, and integrates them into an ontology-knowledge graph through LLM-based triple extraction. Our approach captures both the hierarchical and logical structures of documents, effectively representing domain-specific semantics that conventional methods fail to reflect. To verify its effectiveness, we constructed rule, table, and multi-hop QA datasets, as well as a toxic clause detection dataset, from industrial standards, and implemented an ontology-aware KG-RAG framework for comparative evaluation. Experimental results show that our method achieves significant performance improvements across all QA types compared to existing KG-RAG approaches. This study demonstrates that reliable and scalable knowledge representation is feasible even for industrial documents with intertwined conditions, constraints, and scopes, contributing to future domain-specific RAG development and intelligent document management. 

---
# Exploiting the Randomness of Large Language Models (LLM) in Text Classification Tasks: Locating Privileged Documents in Legal Matters 

**Authors**: Keith Huffman, Jianping Zhang, Nathaniel Huber-Fliflet, Fusheng Wei, Peter Gronvall  

**Link**: [PDF](https://arxiv.org/pdf/2512.08083)  

**Abstract**: In legal matters, text classification models are most often used to filter through large datasets in search of documents that meet certain pre-selected criteria like relevance to a certain subject matter, such as legally privileged communications and attorney-directed documents. In this context, large language models have demonstrated strong performance. This paper presents an empirical study investigating the role of randomness in LLM-based classification for attorney-client privileged document detection, focusing on four key dimensions: (1) the effectiveness of LLMs in identifying legally privileged documents, (2) the influence of randomness control parameters on classification outputs, (3) their impact on overall classification performance, and (4) a methodology for leveraging randomness to enhance accuracy. Experimental results showed that LLMs can identify privileged documents effectively, randomness control parameters have minimal impact on classification performance, and importantly, our developed methodology for leveraging randomness can have a significant impact on improving accuracy. Notably, this methodology that leverages randomness could also enhance a corporation's confidence in an LLM's output when incorporated into its sanctions-compliance processes. As organizations increasingly rely on LLMs to augment compliance workflows, reducing output variability helps build internal and regulatory confidence in LLM-derived sanctions-screening decisions. 

---
# Leveraging Machine Learning and Large Language Models for Automated Image Clustering and Description in Legal Discovery 

**Authors**: Qiang Mao, Fusheng Wei, Robert Neary, Charles Wang, Han Qin, Jianping Zhang, Nathaniel Huber-Fliflet  

**Link**: [PDF](https://arxiv.org/pdf/2512.08079)  

**Abstract**: The rapid increase in digital image creation and retention presents substantial challenges during legal discovery, digital archive, and content management. Corporations and legal teams must organize, analyze, and extract meaningful insights from large image collections under strict time pressures, making manual review impractical and costly. These demands have intensified interest in automated methods that can efficiently organize and describe large-scale image datasets. This paper presents a systematic investigation of automated cluster description generation through the integration of image clustering, image captioning, and large language models (LLMs). We apply K-means clustering to group images into 20 visually coherent clusters and generate base captions using the Azure AI Vision API. We then evaluate three critical dimensions of the cluster description process: (1) image sampling strategies, comparing random, centroid-based, stratified, hybrid, and density-based sampling against using all cluster images; (2) prompting techniques, contrasting standard prompting with chain-of-thought prompting; and (3) description generation methods, comparing LLM-based generation with traditional TF-IDF and template-based approaches. We assess description quality using semantic similarity and coverage metrics. Results show that strategic sampling with 20 images per cluster performs comparably to exhaustive inclusion while significantly reducing computational cost, with only stratified sampling showing modest degradation. LLM-based methods consistently outperform TF-IDF baselines, and standard prompts outperform chain-of-thought prompts for this task. These findings provide practical guidance for deploying scalable, accurate cluster description systems that support high-volume workflows in legal discovery and other domains requiring automated organization of large image collections. 

---
# A Comparative Study of Retrieval Methods in Azure AI Search 

**Authors**: Qiang Mao, Han Qin, Robert Neary, Charles Wang, Fusheng Wei, Jianping Zhang, Nathaniel Huber-Fliflet  

**Link**: [PDF](https://arxiv.org/pdf/2512.08078)  

**Abstract**: Increasingly, attorneys are interested in moving beyond keyword and semantic search to improve the efficiency of how they find key information during a document review task. Large language models (LLMs) are now seen as tools that attorneys can use to ask natural language questions of their data during document review to receive accurate and concise answers. This study evaluates retrieval strategies within Microsoft Azure's Retrieval-Augmented Generation (RAG) framework to identify effective approaches for Early Case Assessment (ECA) in eDiscovery. During ECA, legal teams analyze data at the outset of a matter to gain a general understanding of the data and attempt to determine key facts and risks before beginning full-scale review. In this paper, we compare the performance of Azure AI Search's keyword, semantic, vector, hybrid, and hybrid-semantic retrieval methods. We then present the accuracy, relevance, and consistency of each method's AI-generated responses. Legal practitioners can use the results of this study to enhance how they select RAG configurations in the future. 

---
# Detecting Privileged Documents by Ranking Connected Network Entities 

**Authors**: Jianping Zhang, Han Qin, Nathaniel Huber-Fliflet  

**Link**: [PDF](https://arxiv.org/pdf/2512.08073)  

**Abstract**: This paper presents a link analysis approach for identifying privileged documents by constructing a network of human entities derived from email header metadata. Entities are classified as either counsel or non-counsel based on a predefined list of known legal professionals. The core assumption is that individuals with frequent interactions with lawyers are more likely to participate in privileged communications. To quantify this likelihood, an algorithm assigns a score to each entity within the network. By utilizing both entity scores and the strength of their connections, the method enhances the identification of privileged documents. Experimental results demonstrate the algorithm's effectiveness in ranking legal entities for privileged document detection. 

---
# MixLM: High-Throughput and Effective LLM Ranking via Text-Embedding Mix-Interaction 

**Authors**: Guoyao Li, Ran He, Shusen Jing, Kayhan Behdin, Yubo Wang, Sundara Raman Ramachandran, Chanh Nguyen, Jian Sheng, Xiaojing Ma, Chuanrui Zhu, Sriram Vasudevan, Muchen Wu, Sayan Ghosh, Lin Su, Qingquan Song, Xiaoqing Wang, Zhipeng Wang, Qing Lan, Yanning Chen, Jingwei Wu, Luke Simon, Wenjing Zhang, Qi Guo, Fedor Borisyuk  

**Link**: [PDF](https://arxiv.org/pdf/2512.07846)  

**Abstract**: Large language models (LLMs) excel at capturing semantic nuances and therefore show impressive relevance ranking performance in modern recommendation and search systems. However, they suffer from high computational overhead under industrial latency and throughput requirements. In particular, cross-encoder ranking systems often create long context prefill-heavy workloads, as the model has to be presented with the user, query and item information. To this end, we propose MixLM, a novel LLM-based ranking framework, which significantly improves the system throughput via reducing the input context length, while preserving the semantic strength of cross-encoder rankers. In contrast to a standard ranking system where the context is presented to the model as pure text, we propose to use mix-interaction, a mixture of text and embedding tokens to represent the input. Specifically, MixLM encodes all items in the catalog into a few embedding tokens and stores in a nearline cache. The encoded item descriptions are used during online inference, effectively reducing the item length from a few thousand text tokens to a few embedding tokens. We share insights from deploying our MixLM framework to a real-world search application at LinkedIn, including a detailed discussion of our training pipelines, as well as a thorough analysis of our online serving infrastructure optimization. Comparing with strong baselines, MixLM increased throughput by 10.0x under the same latency budget, while maintaining relevance metrics. The efficiency gains delivered by MixLM enabled full-traffic deployment of LLM-powered search, which resulted in a significant 0.47% increase in Daily Active Users (DAU) in online A/B tests. 

---
# ClinicalTrialsHub: Bridging Registries and Literature for Comprehensive Clinical Trial Access 

**Authors**: Jiwoo Park, Ruoqi Liu, Avani Jagdale, Andrew Srisuwananukorn, Jing Zhao, Lang Li, Ping Zhang, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.08193)  

**Abstract**: We present ClinicalTrialsHub, an interactive search-focused platform that consolidates all data from this http URL and augments it by automatically extracting and structuring trial-relevant information from PubMed research articles. Our system effectively increases access to structured clinical trial data by 83.8% compared to relying on this http URL alone, with potential to make access easier for patients, clinicians, researchers, and policymakers, advancing evidence-based medicine. ClinicalTrialsHub uses large language models such as GPT-5.1 and Gemini-3-Pro to enhance accessibility. The platform automatically parses full-text research articles to extract structured trial information, translates user queries into structured database searches, and provides an attributed question-answering system that generates evidence-grounded answers linked to specific source sentences. We demonstrate its utility through a user study involving clinicians, clinical researchers, and PhD students of pharmaceutical sciences and nursing, and a systematic automatic evaluation of its information extraction and question answering capabilities. 

---
