# Chunk Twice, Embed Once: A Systematic Study of Segmentation and Representation Trade-offs in Chemistry-Aware Retrieval-Augmented Generation 

**Authors**: Mahmoud Amiri, Thomas Bocklitz  

**Link**: [PDF](https://arxiv.org/pdf/2506.17277)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are increasingly vital for navigating the ever-expanding body of scientific literature, particularly in high-stakes domains such as chemistry. Despite the promise of RAG, foundational design choices -- such as how documents are segmented and represented -- remain underexplored in domain-specific contexts. This study presents the first large-scale, systematic evaluation of chunking strategies and embedding models tailored to chemistry-focused RAG systems. We investigate 25 chunking configurations across five method families and evaluate 48 embedding models on three chemistry-specific benchmarks, including the newly introduced QuestChemRetrieval dataset. Our results reveal that recursive token-based chunking (specifically R100-0) consistently outperforms other approaches, offering strong performance with minimal resource overhead. We also find that retrieval-optimized embeddings -- such as Nomic and Intfloat E5 variants -- substantially outperform domain-specialized models like SciBERT. By releasing our datasets, evaluation framework, and empirical benchmarks, we provide actionable guidelines for building effective and efficient chemistry-aware RAG systems. 

---
# PreQRAG -- Classify and Rewrite for Enhanced RAG 

**Authors**: Damian Martinez, Catalina Riano, Hui Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17493)  

**Abstract**: This paper presents the submission of the UDInfo team to the SIGIR 2025 LiveRAG Challenge. We introduce PreQRAG, a Retrieval Augmented Generation (RAG) architecture designed to improve retrieval and generation quality through targeted question preprocessing. PreQRAG incorporates a pipeline that first classifies each input question as either single-document or multi-document type. For single-document questions, we employ question rewriting techniques to improve retrieval precision and generation relevance. For multi-document questions, we decompose complex queries into focused sub-questions that can be processed more effectively by downstream components. This classification and rewriting strategy improves the RAG performance. Experimental evaluation of the LiveRAG Challenge dataset demonstrates the effectiveness of our question-type-aware architecture, with PreQRAG achieving the preliminary second place in Session 2 of the LiveRAG challenge. 

---
# SlimRAG: Retrieval without Graphs via Entity-Aware Context Selection 

**Authors**: Jiale Zhang, Jiaxiang Chen, Zhucong Li, Jie Ding, Kui Zhao, Zenglin Xu, Xin Pang, Yinghui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17288)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances language models by incorporating external knowledge at inference time. However, graph-based RAG systems often suffer from structural overhead and imprecise retrieval: they require costly pipelines for entity linking and relation extraction, yet frequently return subgraphs filled with loosely related or tangential content. This stems from a fundamental flaw -- semantic similarity does not imply semantic relevance. We introduce SlimRAG, a lightweight framework for retrieval without graphs. SlimRAG replaces structure-heavy components with a simple yet effective entity-aware mechanism. At indexing time, it constructs a compact entity-to-chunk table based on semantic embeddings. At query time, it identifies salient entities, retrieves and scores associated chunks, and assembles a concise, contextually relevant input -- without graph traversal or edge construction. To quantify retrieval efficiency, we propose Relative Index Token Utilization (RITU), a metric measuring the compactness of retrieved content. Experiments across multiple QA benchmarks show that SlimRAG outperforms strong flat and graph-based baselines in accuracy while reducing index size and RITU (e.g., 16.31 vs. 56+), highlighting the value of structure-free, entity-centric context selection. The code will be released soon. this https URL 

---
# PDF Retrieval Augmented Question Answering 

**Authors**: Thi Thu Uyen Hoang, Viet Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18027)  

**Abstract**: This paper presents an advancement in Question-Answering (QA) systems using a Retrieval Augmented Generation (RAG) framework to enhance information extraction from PDF files. Recognizing the richness and diversity of data within PDFs--including text, images, vector diagrams, graphs, and tables--poses unique challenges for existing QA systems primarily designed for textual content. We seek to develop a comprehensive RAG-based QA system that will effectively address complex multimodal questions, where several data types are combined in the query. This is mainly achieved by refining approaches to processing and integrating non-textual elements in PDFs into the RAG framework to derive precise and relevant answers, as well as fine-tuning large language models to better adapt to our system. We provide an in-depth experimental evaluation of our solution, demonstrating its capability to extract accurate information that can be applied to different types of content across PDFs. This work not only pushes the boundaries of retrieval-augmented QA systems but also lays a foundation for further research in multimodal data integration and processing. 

---
# A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment 

**Authors**: Quanwei Tang, Sophia Yat Mei Lee, Junshuang Wu, Dong Zhang, Shoushan Li, Erik Cambria, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17951)  

**Abstract**: Recent advancements in retrieval-augmented generation (RAG) have enhanced large language models in question answering by integrating external knowledge. However, challenges persist in achieving global understanding and aligning responses with human ethical and quality preferences. To address these issues, we propose GraphMPA, a comprehensive graph-based framework with mode-seeking preference alignment. Our approach constructs a hierarchical document graph using a general similarity measurement, mimicking human cognitive processes for information understanding and synthesis. Additionally, we introduce mode-seeking preference optimization to better align model outputs with human preferences through probability-matching constraints. Extensive experiments on six datasets demonstrate the effectiveness of our \href{this https URL}{GraphMPA}. 

---
# T-CPDL: A Temporal Causal Probabilistic Description Logic for Developing Logic-RAG Agent 

**Authors**: Hong Qing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18559)  

**Abstract**: Large language models excel at generating fluent text but frequently struggle with structured reasoning involving temporal constraints, causal relationships, and probabilistic reasoning. To address these limitations, we propose Temporal Causal Probabilistic Description Logic (T-CPDL), an integrated framework that extends traditional Description Logic with temporal interval operators, explicit causal relationships, and probabilistic annotations. We present two distinct variants of T-CPDL: one capturing qualitative temporal relationships through Allen's interval algebra, and another variant enriched with explicit timestamped causal assertions. Both variants share a unified logical structure, enabling complex reasoning tasks ranging from simple temporal ordering to nuanced probabilistic causation. Empirical evaluations on temporal reasoning and causal inference benchmarks confirm that T-CPDL substantially improves inference accuracy, interpretability, and confidence calibration of language model outputs. By delivering transparent reasoning paths and fine-grained temporal and causal semantics, T-CPDL significantly enhances the capability of language models to support robust, explainable, and trustworthy decision-making. This work also lays the groundwork for developing advanced Logic-Retrieval-Augmented Generation (Logic-RAG) frameworks, potentially boosting the reasoning capabilities and efficiency of knowledge graph-enhanced RAG systems. 

---
# Standard Applicability Judgment and Cross-jurisdictional Reasoning: A RAG-based Framework for Medical Device Compliance 

**Authors**: Yu Han, Aaron Ceross, Jeroen H.M. Bergmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.18511)  

**Abstract**: Identifying the appropriate regulatory standard applicability remains a critical yet understudied challenge in medical device compliance, frequently necessitating expert interpretation of fragmented and heterogeneous documentation across different jurisdictions. To address this challenge, we introduce a modular AI system that leverages a retrieval-augmented generation (RAG) pipeline to automate standard applicability determination. Given a free-text device description, our system retrieves candidate standards from a curated corpus and uses large language models to infer jurisdiction-specific applicability, classified as Mandatory, Recommended, or Not Applicable, with traceable justifications. We construct an international benchmark dataset of medical device descriptions with expert-annotated standard mappings, and evaluate our system against retrieval-only, zero-shot, and rule-based baselines. The proposed approach attains a classification accuracy of 73% and a Top-5 retrieval recall of 87%, demonstrating its effectiveness in identifying relevant regulatory standards. We introduce the first end-to-end system for standard applicability reasoning, enabling scalable and interpretable AI-supported regulatory science. Notably, our region-aware RAG agent performs cross-jurisdictional reasoning between Chinese and U.S. standards, supporting conflict resolution and applicability justification across regulatory frameworks. 

---
# From Unstructured Communication to Intelligent RAG: Multi-Agent Automation for Supply Chain Knowledge Bases 

**Authors**: Yao Zhang, Zaixi Shang, Silpan Patel, Mikel Zuniga  

**Link**: [PDF](https://arxiv.org/pdf/2506.17484)  

**Abstract**: Supply chain operations generate vast amounts of operational data; however, critical knowledge such as system usage practices, troubleshooting workflows, and resolution techniques often remains buried within unstructured communications like support tickets, emails, and chat logs. While RAG systems aim to leverage such communications as a knowledge base, their effectiveness is limited by raw data challenges: support tickets are typically noisy, inconsistent, and incomplete, making direct retrieval suboptimal. Unlike existing RAG approaches that focus on runtime optimization, we introduce a novel offline-first methodology that transforms these communications into a structured knowledge base. Our key innovation is a LLMs-based multi-agent system orchestrating three specialized agents: Category Discovery for taxonomy creation, Categorization for ticket grouping, and Knowledge Synthesis for article generation. Applying our methodology to real-world support tickets with resolution notes and comments, our system creates a compact knowledge base - reducing total volume to just 3.4% of original ticket data while improving quality. Experiments demonstrate that our prebuilt knowledge base in RAG systems significantly outperforms traditional RAG implementations (48.74% vs. 38.60% helpful answers) and achieves a 77.4% reduction in unhelpful responses. By automating institutional knowledge capture that typically remains siloed in experts' heads, our solution translates to substantial operational efficiency: reducing support workload, accelerating resolution times, and creating self-improving systems that automatically resolve approximately 50% of future supply chain tickets. Our approach addresses a key gap in knowledge management by transforming transient communications into structured, reusable knowledge through intelligent offline processing rather than latency-inducing runtime architectures. 

---
# Measuring and Augmenting Large Language Models for Solving Capture-the-Flag Challenges 

**Authors**: Zimo Ji, Daoyuan Wu, Wenyuan Jiang, Pingchuan Ma, Zongjie Li, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17644)  

**Abstract**: Capture-the-Flag (CTF) competitions are crucial for cybersecurity education and training. As large language models (LLMs) evolve, there is increasing interest in their ability to automate CTF challenge solving. For example, DARPA has organized the AIxCC competition since 2023 to advance AI-powered automated offense and defense. However, this demands a combination of multiple abilities, from knowledge to reasoning and further to actions. In this paper, we highlight the importance of technical knowledge in solving CTF problems and deliberately construct a focused benchmark, CTFKnow, with 3,992 questions to measure LLMs' performance in this core aspect. Our study offers a focused and innovative measurement of LLMs' capability in understanding CTF knowledge and applying it to solve CTF challenges. Our key findings reveal that while LLMs possess substantial technical knowledge, they falter in accurately applying this knowledge to specific scenarios and adapting their strategies based on feedback from the CTF environment.
Based on insights derived from this measurement study, we propose CTFAgent, a novel LLM-driven framework for advancing CTF problem-solving. CTFAgent introduces two new modules: two-stage Retrieval Augmented Generation (RAG) and interactive Environmental Augmentation, which enhance LLMs' technical knowledge and vulnerability exploitation on CTF, respectively. Our experimental results show that, on two popular CTF datasets, CTFAgent both achieves over 80% performance improvement. Moreover, in the recent picoCTF2024 hosted by CMU, CTFAgent ranked in the top 23.6% of nearly 7,000 participating teams. This reflects the benefit of our measurement study and the potential of our framework in advancing LLMs' capabilities in CTF problem-solving. 

---
# Automatic Large Language Models Creation of Interactive Learning Lessons 

**Authors**: Jionghao Lin, Jiarui Rao, Yiyang Zhao, Yuting Wang, Ashish Gurung, Amanda Barany, Jaclyn Ocumpaugh, Ryan S. Baker, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.17356)  

**Abstract**: We explore the automatic generation of interactive, scenario-based lessons designed to train novice human tutors who teach middle school mathematics online. Employing prompt engineering through a Retrieval-Augmented Generation approach with GPT-4o, we developed a system capable of creating structured tutor training lessons. Our study generated lessons in English for three key topics: Encouraging Students' Independence, Encouraging Help-Seeking Behavior, and Turning on Cameras, using a task decomposition prompting strategy that breaks lesson generation into sub-tasks. The generated lessons were evaluated by two human evaluators, who provided both quantitative and qualitative evaluations using a comprehensive rubric informed by lesson design research. Results demonstrate that the task decomposition strategy led to higher-rated lessons compared to single-step generation. Human evaluators identified several strengths in the LLM-generated lessons, including well-structured content and time-saving potential, while also noting limitations such as generic feedback and a lack of clarity in some instructional sections. These findings underscore the potential of hybrid human-AI approaches for generating effective lessons in tutor training. 

---
