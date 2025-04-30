# Can LLMs Be Trusted for Evaluating RAG Systems? A Survey of Methods and Datasets 

**Authors**: Lorenz Brehme, Thomas Ströhle, Ruth Breu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20119)  

**Abstract**: Retrieval-Augmented Generation (RAG) has advanced significantly in recent years. The complexity of RAG systems, which involve multiple components-such as indexing, retrieval, and generation-along with numerous other parameters, poses substantial challenges for systematic evaluation and quality enhancement. Previous research highlights that evaluating RAG systems is essential for documenting advancements, comparing configurations, and identifying effective approaches for domain-specific applications. This study systematically reviews 63 academic articles to provide a comprehensive overview of state-of-the-art RAG evaluation methodologies, focusing on four key areas: datasets, retrievers, indexing and databases, and the generator component. We observe the feasibility of an automated evaluation approach for each component of a RAG system, leveraging an LLM capable of both generating evaluation datasets and conducting evaluations. In addition, we found that further practical research is essential to provide companies with clear guidance on the do's and don'ts of implementing and evaluating RAG systems. By synthesizing evaluation approaches for key RAG components and emphasizing the creation and adaptation of domain-specific datasets for benchmarking, we contribute to the advancement of systematic evaluation methods and the improvement of evaluation rigor for RAG systems. Furthermore, by examining the interplay between automated approaches leveraging LLMs and human judgment, we contribute to the ongoing discourse on balancing automation and human input, clarifying their respective contributions, limitations, and challenges in achieving robust and reliable evaluations. 

---
# CBM-RAG: Demonstrating Enhanced Interpretability in Radiology Report Generation with Multi-Agent RAG and Concept Bottleneck Models 

**Authors**: Hasan Md Tusfiqur Alam, Devansh Srivastav, Abdulrahman Mohamed Selim, Md Abdul Kadir, Md Moktadiurl Hoque Shuvo, Daniel Sonntag  

**Link**: [PDF](https://arxiv.org/pdf/2504.20898)  

**Abstract**: Advancements in generative Artificial Intelligence (AI) hold great promise for automating radiology workflows, yet challenges in interpretability and reliability hinder clinical adoption. This paper presents an automated radiology report generation framework that combines Concept Bottleneck Models (CBMs) with a Multi-Agent Retrieval-Augmented Generation (RAG) system to bridge AI performance with clinical explainability. CBMs map chest X-ray features to human-understandable clinical concepts, enabling transparent disease classification. Meanwhile, the RAG system integrates multi-agent collaboration and external knowledge to produce contextually rich, evidence-based reports. Our demonstration showcases the system's ability to deliver interpretable predictions, mitigate hallucinations, and generate high-quality, tailored reports with an interactive interface addressing accuracy, trust, and usability challenges. This framework provides a pathway to improving diagnostic consistency and empowering radiologists with actionable insights. 

---
# UniversalRAG: Retrieval-Augmented Generation over Multiple Corpora with Diverse Modalities and Granularities 

**Authors**: Woongyeong Yeo, Kangsan Kim, Soyeong Jeong, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20734)  

**Abstract**: Retrieval-Augmented Generation (RAG) has shown substantial promise in improving factual accuracy by grounding model responses with external knowledge relevant to queries. However, most existing RAG approaches are limited to a text-only corpus, and while recent efforts have extended RAG to other modalities such as images and videos, they typically operate over a single modality-specific corpus. In contrast, real-world queries vary widely in the type of knowledge they require, which a single type of knowledge source cannot address. To address this, we introduce UniversalRAG, a novel RAG framework designed to retrieve and integrate knowledge from heterogeneous sources with diverse modalities and granularities. Specifically, motivated by the observation that forcing all modalities into a unified representation space derived from a single combined corpus causes a modality gap, where the retrieval tends to favor items from the same modality as the query, we propose a modality-aware routing mechanism that dynamically identifies the most appropriate modality-specific corpus and performs targeted retrieval within it. Also, beyond modality, we organize each modality into multiple granularity levels, enabling fine-tuned retrieval tailored to the complexity and scope of the query. We validate UniversalRAG on 8 benchmarks spanning multiple modalities, showing its superiority over modality-specific and unified baselines. 

---
# OpenTCM: A GraphRAG-Empowered LLM-based System for Traditional Chinese Medicine Knowledge Retrieval and Diagnosis 

**Authors**: Jinglin He, Yunqi Guo, Lai Kwan Lam, Waikei Leung, Lixing He, Yuanan Jiang, Chi Chiu Wang, Guoliang Xing, Hongkai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20118)  

**Abstract**: Traditional Chinese Medicine (TCM) represents a rich repository of ancient medical knowledge that continues to play an important role in modern healthcare. Due to the complexity and breadth of the TCM literature, the integration of AI technologies is critical for its modernization and broader accessibility. However, this integration poses considerable challenges, including the interpretation of obscure classical Chinese texts and the modeling of intricate semantic relationships among TCM concepts. In this paper, we develop OpenTCM, an LLM-based system that combines a domain-specific TCM knowledge graph and Graph-based Retrieval-Augmented Generation (GraphRAG). First, we extract more than 3.73 million classical Chinese characters from 68 gynecological books in the Chinese Medical Classics Database, with the help of TCM and gynecology experts. Second, we construct a comprehensive multi-relational knowledge graph comprising more than 48,000 entities and 152,000 interrelationships, using customized prompts and Chinese-oriented LLMs such as DeepSeek and Kimi to ensure high-fidelity semantic understanding. Last, we integrate OpenTCM with this knowledge graph, enabling high-fidelity ingredient knowledge retrieval and diagnostic question-answering without model fine-tuning. Experimental evaluations demonstrate that our prompt design and model selection significantly improve knowledge graph quality, achieving a precision of 98. 55% and an F1 score of 99. 55%. In addition, OpenTCM achieves mean expert scores of 4.5 in ingredient information retrieval and 3.8 in diagnostic question-answering tasks, outperforming state-of-the-art solutions in real-world TCM use cases. 

---
# ARCS: Agentic Retrieval-Augmented Code Synthesis with Iterative Refinement 

**Authors**: Manish Bhattarai, Miguel Cordova, Javier Santos, Dan O'Malley  

**Link**: [PDF](https://arxiv.org/pdf/2504.20434)  

**Abstract**: In supercomputing, efficient and optimized code generation is essential to leverage high-performance systems effectively. We propose Agentic Retrieval-Augmented Code Synthesis (ARCS), an advanced framework for accurate, robust, and efficient code generation, completion, and translation. ARCS integrates Retrieval-Augmented Generation (RAG) with Chain-of-Thought (CoT) reasoning to systematically break down and iteratively refine complex programming tasks. An agent-based RAG mechanism retrieves relevant code snippets, while real-time execution feedback drives the synthesis of candidate solutions. This process is formalized as a state-action search tree optimization, balancing code correctness with editing efficiency. Evaluations on the Geeks4Geeks and HumanEval benchmarks demonstrate that ARCS significantly outperforms traditional prompting methods in translation and generation quality. By enabling scalable and precise code synthesis, ARCS offers transformative potential for automating and optimizing code development in supercomputing applications, enhancing computational resource utilization. 

---
