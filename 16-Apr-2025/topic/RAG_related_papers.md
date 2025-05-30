# HeteRAG: A Heterogeneous Retrieval-augmented Generation Framework with Decoupled Knowledge Representations 

**Authors**: Peiru Yang, Xintian Li, Zhiyang Hu, Jiapeng Wang, Jinhua Yin, Huili Wang, Lizhi He, Shuai Yang, Shangguang Wang, Yongfeng Huang, Tao Qi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10529)  

**Abstract**: Retrieval-augmented generation (RAG) methods can enhance the performance of LLMs by incorporating retrieved knowledge chunks into the generation process. In general, the retrieval and generation steps usually have different requirements for these knowledge chunks. The retrieval step benefits from comprehensive information to improve retrieval accuracy, whereas excessively long chunks may introduce redundant contextual information, thereby diminishing both the effectiveness and efficiency of the generation process. However, existing RAG methods typically employ identical representations of knowledge chunks for both retrieval and generation, resulting in suboptimal performance. In this paper, we propose a heterogeneous RAG framework (\myname) that decouples the representations of knowledge chunks for retrieval and generation, thereby enhancing the LLMs in both effectiveness and efficiency. Specifically, we utilize short chunks to represent knowledge to adapt the generation step and utilize the corresponding chunk with its contextual information from multi-granular views to enhance retrieval accuracy. We further introduce an adaptive prompt tuning method for the retrieval model to adapt the heterogeneous retrieval augmented generation process. Extensive experiments demonstrate that \myname achieves significant improvements compared to baselines. 

---
# Graph-based Approaches and Functionalities in Retrieval-Augmented Generation: A Comprehensive Survey 

**Authors**: Zulun Zhu, Tiancheng Huang, Kai Wang, Junda Ye, Xinghe Chen, Siqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.10499)  

**Abstract**: Large language models (LLMs) struggle with the factual error during inference due to the lack of sufficient training data and the most updated knowledge, leading to the hallucination problem. Retrieval-Augmented Generation (RAG) has gained attention as a promising solution to address the limitation of LLMs, by retrieving relevant information from external source to generate more accurate answers to the questions. Given the pervasive presence of structured knowledge in the external source, considerable strides in RAG have been made to employ the techniques related to graphs and achieve more complex reasoning based on the topological information between knowledge entities. However, there is currently neither unified review examining the diverse roles of graphs in RAG, nor a comprehensive resource to help researchers navigate and contribute to this evolving field. This survey offers a novel perspective on the functionality of graphs within RAG and their impact on enhancing performance across a wide range of graph-structured data. It provides a detailed breakdown of the roles that graphs play in RAG, covering database construction, algorithms, pipelines, and tasks. Finally, it identifies current challenges and outline future research directions, aiming to inspire further developments in this field. Our graph-centered analysis highlights the commonalities and differences in existing methods, setting the stage for future researchers in areas such as graph learning, database systems, and natural language processing. 

---
# Poly-Vector Retrieval: Reference and Content Embeddings for Legal Documents 

**Authors**: João Alberto de Oliveira Lima  

**Link**: [PDF](https://arxiv.org/pdf/2504.10508)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as an effective paradigm for generating contextually accurate answers by integrating Large Language Models (LLMs) with retrieval mechanisms. However, in legal contexts, users frequently reference norms by their labels or nicknames (e.g., Article 5 of the Constitution or Consumer Defense Code (CDC)), rather than by their content, posing challenges for traditional RAG approaches that rely solely on semantic embeddings of text. Furthermore, legal texts themselves heavily rely on explicit cross-references (e.g., "pursuant to Article 34") that function as pointers. Both scenarios pose challenges for traditional RAG approaches that rely solely on semantic embeddings of text, often failing to retrieve the necessary referenced content. This paper introduces Poly-Vector Retrieval, a method assigning multiple distinct embeddings to each legal provision: one embedding captures the content (the full text), another captures the label (the identifier or proper name), and optionally additional embeddings capture alternative denominations. Inspired by Frege's distinction between Sense and Reference, this poly-vector retrieval approach treats labels, identifiers and reference markers as rigid designators and content embeddings as carriers of semantic substance. Experiments on the Brazilian Federal Constitution demonstrate that Poly-Vector Retrieval significantly improves retrieval accuracy for label-centric queries and potential to resolve internal and external cross-references, without compromising performance on purely semantic queries. The study discusses philosophical and practical implications of explicitly separating reference from content in vector embeddings and proposes future research directions for applying this approach to broader legal datasets and other domains characterized by explicit reference identifiers. 

---
# Efficient Distributed Retrieval-Augmented Generation for Enhancing Language Model Performance 

**Authors**: Shangyu Liu, Zhenzhe Zheng, Xiaoyao Huang, Fan Wu, Jie Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11197)  

**Abstract**: Small language models (SLMs) support efficient deployments on resource-constrained edge devices, but their limited capacity compromises inference performance. Retrieval-augmented generation (RAG) is a promising solution to enhance model performance by integrating external databases, without requiring intensive on-device model retraining. However, large-scale public databases and user-specific private contextual documents are typically located on the cloud and the device separately, while existing RAG implementations are primarily centralized. To bridge this gap, we propose DRAGON, a distributed RAG framework to enhance on-device SLMs through both general and personal knowledge without the risk of leaking document privacy. Specifically, DRAGON decomposes multi-document RAG into multiple parallel token generation processes performed independently and locally on the cloud and the device, and employs a newly designed Speculative Aggregation, a dual-side speculative algorithm to avoid frequent output synchronization between the cloud and device. A new scheduling algorithm is further introduced to identify the optimal aggregation side based on real-time network conditions. Evaluations on real-world hardware testbed demonstrate a significant performance improvement of DRAGON-up to 1.9x greater gains over standalone SLM compared to the centralized RAG, substantial reduction in per-token latency, and negligible Time to First Token (TTFT) overhead. 

---
# ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search 

**Authors**: Yize Zhang, Tianshu Wang, Sirui Chen, Kun Wang, Xingyu Zeng, Hongyu Lin, Xianpei Han, Le Sun, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10893)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities and are receiving increasing attention to enhance their reasoning through scaling test--time compute. However, their application in open--ended, knowledge--intensive, complex reasoning scenarios is still limited. Reasoning--oriented methods struggle to generalize to open--ended scenarios due to implicit assumptions of complete world knowledge. Meanwhile, knowledge--augmented reasoning (KAR) methods fail to address two core challenges: 1) error propagation, where errors in early steps cascade through the chain, and 2) verification bottleneck, where the explore--exploit tradeoff arises in multi--branch decision processes. To overcome these limitations, we introduce ARise, a novel framework that integrates risk assessment of intermediate reasoning states with dynamic retrieval--augmented generation (RAG) within a Monte Carlo tree search paradigm. This approach enables effective construction and optimization of reasoning plans across multiple maintained hypothesis branches. Experimental results show that ARise significantly outperforms the state--of--the--art KAR methods by up to 23.10%, and the latest RAG-equipped large reasoning models by up to 25.37%. 

---
# Towards Automated Safety Requirements Derivation Using Agent-based RAG 

**Authors**: Balahari Vignesh Balu, Florian Geissler, Francesco Carella, Joao-Vitor Zacchi, Josef Jiru, Nuria Mata, Reinhard Stolle  

**Link**: [PDF](https://arxiv.org/pdf/2504.11243)  

**Abstract**: We study the automated derivation of safety requirements in a self-driving vehicle use case, leveraging LLMs in combination with agent-based retrieval-augmented generation. Conventional approaches that utilise pre-trained LLMs to assist in safety analyses typically lack domain-specific knowledge. Existing RAG approaches address this issue, yet their performance deteriorates when handling complex queries and it becomes increasingly harder to retrieve the most relevant information. This is particularly relevant for safety-relevant applications. In this paper, we propose the use of agent-based RAG to derive safety requirements and show that the retrieved information is more relevant to the queries. We implement an agent-based approach on a document pool of automotive standards and the Apollo case study, as a representative example of an automated driving perception system. Our solution is tested on a data set of safety requirement questions and answers, extracted from the Apollo data. Evaluating a set of selected RAG metrics, we present and discuss advantages of a agent-based approach compared to default RAG methods. 

---
# Exploring the Role of KG-Based RAG in Japanese Medical Question Answering with Small-Scale LLMs 

**Authors**: Yingjian Chen, Feiyang Li, Xingyu Song, Tianxiao Li, Issey Sudeka, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.10982)  

**Abstract**: Large language models (LLMs) perform well in medical QA, but their effectiveness in Japanese contexts is limited due to privacy constraints that prevent the use of commercial models like GPT-4 in clinical settings. As a result, recent efforts focus on instruction-tuning open-source LLMs, though the potential of combining them with retrieval-augmented generation (RAG) remains underexplored. To bridge this gap, we are the first to explore a knowledge graph-based (KG) RAG framework for Japanese medical QA small-scale open-source LLMs. Experimental results show that KG-based RAG has only a limited impact on Japanese medical QA using small-scale open-source LLMs. Further case studies reveal that the effectiveness of the RAG is sensitive to the quality and relevance of the external retrieved content. These findings offer valuable insights into the challenges and potential of applying RAG in Japanese medical QA, while also serving as a reference for other low-resource languages. 

---
# ReZero: Enhancing LLM search ability by trying one-more-time 

**Authors**: Alan Dao, Thinh Le  

**Link**: [PDF](https://arxiv.org/pdf/2504.11001)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves Large Language Model (LLM) performance on knowledge-intensive tasks but depends heavily on initial search query quality. Current methods, often using Reinforcement Learning (RL), typically focus on query formulation or reasoning over results, without explicitly encouraging persistence after a failed search. We introduce ReZero (Retry-Zero), a novel RL framework that directly rewards the act of retrying a search query following an initial unsuccessful attempt. This incentivizes the LLM to explore alternative queries rather than prematurely halting. ReZero demonstrates significant improvement, achieving 46.88% accuracy compared to a 25% baseline. By rewarding persistence, ReZero enhances LLM robustness in complex information-seeking scenarios where initial queries may prove insufficient. 

---
