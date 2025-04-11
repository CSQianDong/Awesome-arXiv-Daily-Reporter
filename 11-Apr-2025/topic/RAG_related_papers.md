# CollEX -- A Multimodal Agentic RAG System Enabling Interactive Exploration of Scientific Collections 

**Authors**: Florian Schneider, Narges Baba Ahmadi, Niloufar Baba Ahmadi, Iris Vogel, Martin Semmann, Chris Biemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.07643)  

**Abstract**: In this paper, we introduce CollEx, an innovative multimodal agentic Retrieval-Augmented Generation (RAG) system designed to enhance interactive exploration of extensive scientific collections. Given the overwhelming volume and inherent complexity of scientific collections, conventional search systems often lack necessary intuitiveness and interactivity, presenting substantial barriers for learners, educators, and researchers. CollEx addresses these limitations by employing state-of-the-art Large Vision-Language Models (LVLMs) as multimodal agents accessible through an intuitive chat interface. By abstracting complex interactions via specialized agents equipped with advanced tools, CollEx facilitates curiosity-driven exploration, significantly simplifying access to diverse scientific collections and records therein. Our system integrates textual and visual modalities, supporting educational scenarios that are helpful for teachers, pupils, students, and researchers by fostering independent exploration as well as scientific excitement and curiosity. Furthermore, CollEx serves the research community by discovering interdisciplinary connections and complementing visual data. We illustrate the effectiveness of our system through a proof-of-concept application containing over 64,000 unique records across 32 collections from a local scientific collection from a public university. 

---
# OSCAR: Online Soft Compression And Reranking 

**Authors**: Maxime Louis, Thibault Formal, Hervé Dejean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2504.07109)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by integrating external knowledge, leading to improved accuracy and relevance. However, scaling RAG pipelines remains computationally expensive as retrieval sizes grow. To address this, we introduce OSCAR, a novel query-dependent online soft compression method that reduces computational overhead while preserving performance. Unlike traditional hard compression methods, which shorten retrieved texts, or soft compression approaches, which map documents to continuous embeddings offline, OSCAR dynamically compresses retrieved information at inference time, eliminating storage overhead and enabling higher compression rates. Additionally, we extend OSCAR to simultaneously perform reranking, further optimizing the efficiency of the RAG pipeline. Our experiments demonstrate state-of-the-art performance with a 2-5x speed-up in inference and minimal to no loss in accuracy for LLMs ranging from 1B to 24B parameters. The models are available at: this https URL. 

---
# Relevance Isn't All You Need: Scaling RAG Systems With Inference-Time Compute Via Multi-Criteria Reranking 

**Authors**: Will LeVine, Bijan Varjavand  

**Link**: [PDF](https://arxiv.org/pdf/2504.07104)  

**Abstract**: Modern Large Language Model (LLM) systems typically rely on Retrieval Augmented Generation (RAG) which aims to gather context that is useful for response generation. These RAG systems typically optimize strictly towards retrieving context that is maximally relevant to the query. However, conventional theory suggests that retrieval systems which seek to maximize context relevance without any additional explicit criteria can create information bottlenecks. We reaffirm this finding in the modern age of LLM's by showing that in standard RAG pipelines, maximizing for context relevance alone can degrade downstream response quality. In response, we show evaluations of existing RAG methods which account for both context relevance and answer quality. These evaluations introduce a novel finding that existing RAG systems scale poorly with inference time compute usage when considering our combined metric. We introduce "RErank BEyond reLevance (REBEL)", which enables RAG systems to scale with inference-time compute via injection of multi-criteria optimization using Chain-of-Thought prompting (and optionally Multi-Turn dialogue). Ultimately, this enables a new performance/speed tradeoff curve, where RAG systems are able to achieve both higher relevance of retrieved contexts and superior answer quality as inference time increases. Code for the implementation of our method in llama-index can be found at the following PR: this https URL. Code for running experiments using this llama-index implementation can be found at this https URL. 

---
# FG-RAG: Enhancing Query-Focused Summarization with Context-Aware Fine-Grained Graph RAG 

**Authors**: Yubin Hong, Chaofan Li, Jingyi Zhang, Yingxia Shao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07103)  

**Abstract**: Retrieval-Augmented Generation (RAG) enables large language models to provide more precise and pertinent responses by incorporating external knowledge. In the Query-Focused Summarization (QFS) task, GraphRAG-based approaches have notably enhanced the comprehensiveness and diversity of generated responses. However, existing GraphRAG-based approaches predominantly focus on coarse-grained information summarization without being aware of the specific query, and the retrieved content lacks sufficient contextual information to generate comprehensive responses. To address the deficiencies of current RAG systems, we propose Context-Aware Fine-Grained Graph RAG (FG-RAG) to enhance the performance of the QFS task. FG-RAG employs Context-Aware Entity Expansion in graph retrieval to expand the coverage of retrieved entities in the graph, thus providing enough contextual information for the retrieved content. Furthermore, FG-RAG utilizes Query-Level Fine-Grained Summarization to incorporate fine-grained details during response generation, enhancing query awareness for the generated summarization. Our evaluation demonstrates that FG-RAG outperforms other RAG systems in multiple metrics of comprehensiveness, diversity, and empowerment when handling the QFS task. Our implementation is available at this https URL. 

---
# A System for Comprehensive Assessment of RAG Frameworks 

**Authors**: Mattia Rengo, Senad Beadini, Domenico Alfano, Roberto Abbruzzese  

**Link**: [PDF](https://arxiv.org/pdf/2504.07803)  

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as a standard paradigm for enhancing the factual accuracy and contextual relevance of Large Language Models (LLMs) by integrating retrieval mechanisms. However, existing evaluation frameworks fail to provide a holistic black-box approach to assessing RAG systems, especially in real-world deployment scenarios. To address this gap, we introduce SCARF (System for Comprehensive Assessment of RAG Frameworks), a modular and flexible evaluation framework designed to benchmark deployed RAG applications systematically. SCARF provides an end-to-end, black-box evaluation methodology, enabling a limited-effort comparison across diverse RAG frameworks. Our framework supports multiple deployment configurations and facilitates automated testing across vector databases and LLM serving strategies, producing a detailed performance report. Moreover, SCARF integrates practical considerations such as response coherence, providing a scalable and adaptable solution for researchers and industry professionals evaluating RAG applications. Using the REST APIs interface, we demonstrate how SCARF can be applied to real-world scenarios, showcasing its flexibility in assessing different RAG frameworks and configurations. SCARF is available at GitHub repository. 

---
# MRD-RAG: Enhancing Medical Diagnosis with Multi-Round Retrieval-Augmented Generation 

**Authors**: Yixiang Chen, Penglei Sun, Xiang Li, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07724)  

**Abstract**: In recent years, accurately and quickly deploying medical large language models (LLMs) has become a significant trend. Among these, retrieval-augmented generation (RAG) has garnered significant attention due to its features of rapid deployment and privacy protection. However, existing medical RAG frameworks still have shortcomings. Most existing medical RAG frameworks are designed for single-round question answering tasks and are not suitable for multi-round diagnostic dialogue. On the other hand, existing medical multi-round RAG frameworks do not consider the interconnections between potential diseases to inquire precisely like a doctor. To address these issues, we propose a Multi-Round Diagnostic RAG (MRD-RAG) framework that mimics the doctor's diagnostic process. This RAG framework can analyze diagnosis information of potential diseases and accurately conduct multi-round diagnosis like a doctor. To evaluate the effectiveness of our proposed frameworks, we conduct experiments on two modern medical datasets and two traditional Chinese medicine datasets, with evaluations by GPT and human doctors on different methods. The results indicate that our RAG framework can significantly enhance the diagnostic performance of LLMs, highlighting the potential of our approach in medical diagnosis. The code and data can be found in our project website this https URL. 

---
# Automated Construction of a Knowledge Graph of Nuclear Fusion Energy for Effective Elicitation and Retrieval of Information 

**Authors**: A. Loreti, K. Chen, R. George, R. Firth, A. Agnello, S. Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2504.07738)  

**Abstract**: In this document, we discuss a multi-step approach to automated construction of a knowledge graph, for structuring and representing domain-specific knowledge from large document corpora. We apply our method to build the first knowledge graph of nuclear fusion energy, a highly specialized field characterized by vast scope and heterogeneity. This is an ideal benchmark to test the key features of our pipeline, including automatic named entity recognition and entity resolution. We show how pre-trained large language models can be used to address these challenges and we evaluate their performance against Zipf's law, which characterizes human-generated natural language. Additionally, we develop a knowledge-graph retrieval-augmented generation system that combines large language models with a multi-prompt approach. This system provides contextually relevant answers to natural-language queries, including complex multi-hop questions that require reasoning across interconnected entities. 

---
# PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization 

**Authors**: Yang Jiao, Xiaodong Wang, Kai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07717)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of applications, e.g., medical question-answering, mathematical sciences, and code generation. However, they also exhibit inherent limitations, such as outdated knowledge and susceptibility to hallucinations. Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to address these issues, but it also introduces new vulnerabilities. Recent efforts have focused on the security of RAG-based LLMs, yet existing attack methods face three critical challenges: (1) their effectiveness declines sharply when only a limited number of poisoned texts can be injected into the knowledge database, (2) they lack sufficient stealth, as the attacks are often detectable by anomaly detection systems, which compromises their effectiveness, and (3) they rely on heuristic approaches to generate poisoned texts, lacking formal optimization frameworks and theoretic guarantees, which limits their effectiveness and applicability. To address these issues, we propose coordinated Prompt-RAG attack (PR-attack), a novel optimization-driven attack that introduces a small number of poisoned texts into the knowledge database while embedding a backdoor trigger within the prompt. When activated, the trigger causes the LLM to generate pre-designed responses to targeted queries, while maintaining normal behavior in other contexts. This ensures both high effectiveness and stealth. We formulate the attack generation process as a bilevel optimization problem leveraging a principled optimization framework to develop optimal poisoned texts and triggers. Extensive experiments across diverse LLMs and datasets demonstrate the effectiveness of PR-Attack, achieving a high attack success rate even with a limited number of poisoned texts and significantly improved stealth compared to existing methods. 

---
