# SafeMate: A Model Context Protocol-Based Multimodal Agent for Emergency Preparedness 

**Authors**: Junfeng Jiao, Jihyung Park, Yiming Xu, Lucy Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2505.02306)  

**Abstract**: Despite the abundance of public safety documents and emergency protocols, most individuals remain ill-equipped to interpret and act on such information during crises. Traditional emergency decision support systems (EDSS) are designed for professionals and rely heavily on static documents like PDFs or SOPs, which are difficult for non-experts to navigate under stress. This gap between institutional knowledge and public accessibility poses a critical barrier to effective emergency preparedness and response.
We introduce SafeMate, a retrieval-augmented AI assistant that delivers accurate, context-aware guidance to general users in both preparedness and active emergency scenarios. Built on the Model Context Protocol (MCP), SafeMate dynamically routes user queries to tools for document retrieval, checklist generation, and structured summarization. It uses FAISS with cosine similarity to identify relevant content from trusted sources. 

---
# Knowing You Don't Know: Learning When to Continue Search in Multi-round RAG through Self-Practicing 

**Authors**: Diji Yang, Linda Zeng, Jinmeng Rao, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02811)  

**Abstract**: Retrieval Augmented Generation (RAG) has shown strong capability in enhancing language models' knowledge and reducing AI generative hallucinations, driving its widespread use. However, complex tasks requiring multi-round retrieval remain challenging, and early attempts tend to be overly optimistic without a good sense of self-skepticism. Current multi-round RAG systems may continue searching even when enough information has already been retrieved, or they may provide incorrect answers without having sufficient information or knowledge. Existing solutions either require large amounts of expensive human-labeled process supervision data or lead to subpar performance.
This paper aims to address these limitations by introducing a new framework, \textbf{SIM-RAG}, to explicitly enhance RAG systems' self-awareness and multi-round retrieval capabilities. To train SIM-RAG, we first let a RAG system self-practice multi-round retrieval, augmenting existing question-answer pairs with intermediate inner monologue reasoning steps to generate synthetic training data. For each pair, the system may explore multiple retrieval paths, which are labeled as successful if they reach the correct answer and unsuccessful otherwise. Using this data, we train a lightweight information sufficiency Critic. At inference time, the Critic evaluates whether the RAG system has retrieved sufficient information at each round, guiding retrieval decisions and improving system-level self-awareness through in-context reinforcement learning.
Experiments across multiple prominent RAG benchmarks show that SIM-RAG is an effective multi-round RAG solution. Furthermore, this framework is system-efficient, adding a lightweight component to RAG without requiring modifications to existing LLMs or search engines, and data-efficient, eliminating the need for costly human-annotated mid-step retrieval process supervision data. 

---
# Real-time Spatial Retrieval Augmented Generation for Urban Environments 

**Authors**: David Nazareno Campo, Javier Conde, Álvaro Alonso, Gabriel Huecas, Joaquín Salvachúa, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2505.02271)  

**Abstract**: The proliferation of Generative Artificial Ingelligence (AI), especially Large Language Models, presents transformative opportunities for urban applications through Urban Foundation Models. However, base models face limitations, as they only contain the knowledge available at the time of training, and updating them is both time-consuming and costly. Retrieval Augmented Generation (RAG) has emerged in the literature as the preferred approach for injecting contextual information into Foundation Models. It prevails over techniques such as fine-tuning, which are less effective in dynamic, real-time scenarios like those found in urban environments. However, traditional RAG architectures, based on semantic databases, knowledge graphs, structured data, or AI-powered web searches, do not fully meet the demands of urban contexts. Urban environments are complex systems characterized by large volumes of interconnected data, frequent updates, real-time processing requirements, security needs, and strong links to the physical world. This work proposes a real-time spatial RAG architecture that defines the necessary components for the effective integration of generative AI into cities, leveraging temporal and spatial filtering capabilities through linked data. The proposed architecture is implemented using FIWARE, an ecosystem of software components to develop smart city solutions and digital twins. The design and implementation are demonstrated through the use case of a tourism assistant in the city of Madrid. The use case serves to validate the correct integration of Foundation Models through the proposed RAG architecture. 

---
# Retrieval-augmented in-context learning for multimodal large language models in disease classification 

**Authors**: Zaifu Zhan, Shuang Zhou, Xiaoshan Zhou, Yongkang Xiao, Jun Wang, Jiawen Deng, He Zhu, Yu Hou, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02087)  

**Abstract**: Objectives: We aim to dynamically retrieve informative demonstrations, enhancing in-context learning in multimodal large language models (MLLMs) for disease classification.
Methods: We propose a Retrieval-Augmented In-Context Learning (RAICL) framework, which integrates retrieval-augmented generation (RAG) and in-context learning (ICL) to adaptively select demonstrations with similar disease patterns, enabling more effective ICL in MLLMs. Specifically, RAICL examines embeddings from diverse encoders, including ResNet, BERT, BioBERT, and ClinicalBERT, to retrieve appropriate demonstrations, and constructs conversational prompts optimized for ICL. We evaluated the framework on two real-world multi-modal datasets (TCGA and IU Chest X-ray), assessing its performance across multiple MLLMs (Qwen, Llava, Gemma), embedding strategies, similarity metrics, and varying numbers of demonstrations.
Results: RAICL consistently improved classification performance. Accuracy increased from 0.7854 to 0.8368 on TCGA and from 0.7924 to 0.8658 on IU Chest X-ray. Multi-modal inputs outperformed single-modal ones, with text-only inputs being stronger than images alone. The richness of information embedded in each modality will determine which embedding model can be used to get better results. Few-shot experiments showed that increasing the number of retrieved examples further enhanced performance. Across different similarity metrics, Euclidean distance achieved the highest accuracy while cosine similarity yielded better macro-F1 scores. RAICL demonstrated consistent improvements across various MLLMs, confirming its robustness and versatility.
Conclusions: RAICL provides an efficient and scalable approach to enhance in-context learning in MLLMs for multimodal disease classification. 

---
# CHORUS: Zero-shot Hierarchical Retrieval and Orchestration for Generating Linear Programming Code 

**Authors**: Tasnim Ahmed, Salimur Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.01485)  

**Abstract**: Linear Programming (LP) problems aim to find the optimal solution to an objective under constraints. These problems typically require domain knowledge, mathematical skills, and programming ability, presenting significant challenges for non-experts. This study explores the efficiency of Large Language Models (LLMs) in generating solver-specific LP code. We propose CHORUS, a retrieval-augmented generation (RAG) framework for synthesizing Gurobi-based LP code from natural language problem statements. CHORUS incorporates a hierarchical tree-like chunking strategy for theoretical contents and generates additional metadata based on code examples from documentation to facilitate self-contained, semantically coherent retrieval. Two-stage retrieval approach of CHORUS followed by cross-encoder reranking further ensures contextual relevance. Finally, expertly crafted prompt and structured parser with reasoning steps improve code generation performance significantly. Experiments on the NL4Opt-Code benchmark show that CHORUS improves the performance of open-source LLMs such as Llama3.1 (8B), Llama3.3 (70B), Phi4 (14B), Deepseek-r1 (32B), and Qwen2.5-coder (32B) by a significant margin compared to baseline and conventional RAG. It also allows these open-source LLMs to outperform or match the performance of much stronger baselines-GPT3.5 and GPT4 while requiring far fewer computational resources. Ablation studies further demonstrate the importance of expert prompting, hierarchical chunking, and structured reasoning. 

---
# Building Scalable AI-Powered Applications with Cloud Databases: Architectures, Best Practices and Performance Considerations 

**Authors**: Santosh Bhupathi  

**Link**: [PDF](https://arxiv.org/pdf/2504.18793)  

**Abstract**: The rapid adoption of AI-powered applications demands high-performance, scalable, and efficient cloud database solutions, as traditional architectures often struggle with AI-driven workloads requiring real-time data access, vector search, and low-latency queries. This paper explores how cloud-native databases enable AI-driven applications by leveraging purpose-built technologies such as vector databases (pgvector), graph databases (AWS Neptune), NoSQL stores (Amazon DocumentDB, DynamoDB), and relational cloud databases (Aurora MySQL and PostgreSQL). It presents architectural patterns for integrating AI workloads with cloud databases, including Retrieval-Augmented Generation (RAG) [1] with LLMs, real-time data pipelines, AI-driven query optimization, and embeddings-based search. Performance benchmarks, scalability considerations, and cost-efficient strategies are evaluated to guide the design of AI-enabled applications. Real-world case studies from industries such as healthcare, finance, and customer experience illustrate how enterprises utilize cloud databases to enhance AI capabilities while ensuring security, governance, and compliance with enterprise and regulatory standards. By providing a comprehensive analysis of AI and cloud database integration, this paper serves as a practical guide for researchers, architects, and enterprises to build next-generation AI applications that optimize performance, scalability, and cost efficiency in cloud environments. 

---
# Generative Sign-description Prompts with Multi-positive Contrastive Learning for Sign Language Recognition 

**Authors**: Siyu Liang, Yunan Li, Wentian Xin, Huizhou Chen, Xujie Liu, Kang Liu, Qiguang Miao  

**Link**: [PDF](https://arxiv.org/pdf/2505.02304)  

**Abstract**: Sign language recognition (SLR) faces fundamental challenges in creating accurate annotations due to the inherent complexity of simultaneous manual and non-manual signals. To the best of our knowledge, this is the first work to integrate generative large language models (LLMs) into SLR tasks. We propose a novel Generative Sign-description Prompts Multi-positive Contrastive learning (GSP-MC) method that leverages retrieval-augmented generation (RAG) with domain-specific LLMs, incorporating multi-step prompt engineering and expert-validated sign language corpora to produce precise multipart descriptions. The GSP-MC method also employs a dual-encoder architecture to bidirectionally align hierarchical skeleton features with multiple text descriptions (global, synonym, and part level) through probabilistic matching. Our approach combines global and part-level losses, optimizing KL divergence to ensure robust alignment across all relevant text-skeleton pairs while capturing both sign-level semantics and detailed part dynamics. Experiments demonstrate state-of-the-art performance against existing methods on the Chinese SLR500 (reaching 97.1%) and Turkish AUTSL datasets (97.07% accuracy). The method's cross-lingual effectiveness highlight its potential for developing inclusive communication technologies. 

---
# Incorporating Legal Structure in Retrieval-Augmented Generation: A Case Study on Copyright Fair Use 

**Authors**: Justin Ho, Alexandra Colby, William Fisher  

**Link**: [PDF](https://arxiv.org/pdf/2505.02164)  

**Abstract**: This paper presents a domain-specific implementation of Retrieval-Augmented Generation (RAG) tailored to the Fair Use Doctrine in U.S. copyright law. Motivated by the increasing prevalence of DMCA takedowns and the lack of accessible legal support for content creators, we propose a structured approach that combines semantic search with legal knowledge graphs and court citation networks to improve retrieval quality and reasoning reliability. Our prototype models legal precedents at the statutory factor level (e.g., purpose, nature, amount, market effect) and incorporates citation-weighted graph representations to prioritize doctrinally authoritative sources. We use Chain-of-Thought reasoning and interleaved retrieval steps to better emulate legal reasoning. Preliminary testing suggests this method improves doctrinal relevance in the retrieval process, laying groundwork for future evaluation and deployment of LLM-based legal assistance tools. 

---
# RAGAR: Retrieval Augment Personalized Image Generation Guided by Recommendation 

**Authors**: Run Ling, Wenji Wang, Yuting Liu, Guibing Guo, Linying Jiang, Xingwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01657)  

**Abstract**: Personalized image generation is crucial for improving the user experience, as it renders reference images into preferred ones according to user visual preferences. Although effective, existing methods face two main issues. First, existing methods treat all items in the user historical sequence equally when extracting user preferences, overlooking the varying semantic similarities between historical items and the reference item. Disproportionately high weights for low-similarity items distort users' visual preferences for the reference item. Second, existing methods heavily rely on consistency between generated and reference images to optimize the generation, which leads to underfitting user preferences and hinders personalization. To address these issues, we propose Retrieval Augment Personalized Image GenerAtion guided by Recommendation (RAGAR). Our approach uses a retrieval mechanism to assign different weights to historical items according to their similarities to the reference item, thereby extracting more refined users' visual preferences for the reference item. Then we introduce a novel rank task based on the multi-modal ranking model to optimize the personalization of the generated images instead of forcing depend on consistency. Extensive experiments and human evaluations on three real-world datasets demonstrate that RAGAR achieves significant improvements in both personalization and semantic metrics compared to five baselines. 

---
# SymbioticRAG: Enhancing Document Intelligence Through Human-LLM Symbiotic Collaboration 

**Authors**: Qiang Sun, Tingting Bi, Sirui Li, Eun-Jung Holden, Paul Duuring, Kai Niu, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02418)  

**Abstract**: We present \textbf{SymbioticRAG}, a novel framework that fundamentally reimagines Retrieval-Augmented Generation~(RAG) systems by establishing a bidirectional learning relationship between humans and machines. Our approach addresses two critical challenges in current RAG systems: the inherently human-centered nature of relevance determination and users' progression from "unconscious incompetence" in query formulation. SymbioticRAG introduces a two-tier solution where Level 1 enables direct human curation of retrieved content through interactive source document exploration, while Level 2 aims to build personalized retrieval models based on captured user interactions. We implement Level 1 through three key components: (1)~a comprehensive document processing pipeline with specialized models for layout detection, OCR, and extraction of tables, formulas, and figures; (2)~an extensible retriever module supporting multiple retrieval strategies; and (3)~an interactive interface that facilitates both user engagement and interaction data logging. We experiment Level 2 implementation via a retriever strategy incorporated LLM summarized user intention from user interaction logs. To maintain high-quality data preparation, we develop a human-on-the-loop validation interface that improves pipeline output while advancing research in specialized extraction tasks. Evaluation across three scenarios (literature review, geological exploration, and education) demonstrates significant improvements in retrieval relevance and user satisfaction compared to traditional RAG approaches. To facilitate broader research and further advancement of SymbioticRAG Level 2 implementation, we will make our system openly accessible to the research community. 

---
