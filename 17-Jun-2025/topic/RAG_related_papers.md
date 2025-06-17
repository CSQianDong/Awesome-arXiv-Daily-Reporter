# The Budget AI Researcher and the Power of RAG Chains 

**Authors**: Franklin Lee, Tengfei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.12317)  

**Abstract**: Navigating the vast and rapidly growing body of scientific literature is a formidable challenge for aspiring researchers. Current approaches to supporting research idea generation often rely on generic large language models (LLMs). While LLMs are effective at aiding comprehension and summarization, they often fall short in guiding users toward practical research ideas due to their limitations. In this study, we present a novel structural framework for research ideation. Our framework, The Budget AI Researcher, uses retrieval-augmented generation (RAG) chains, vector databases, and topic-guided pairing to recombine concepts from hundreds of machine learning papers. The system ingests papers from nine major AI conferences, which collectively span the vast subfields of machine learning, and organizes them into a hierarchical topic tree. It uses the tree to identify distant topic pairs, generate novel research abstracts, and refine them through iterative self-evaluation against relevant literature and peer reviews, generating and refining abstracts that are both grounded in real-world research and demonstrably interesting. Experiments using LLM-based metrics indicate that our method significantly improves the concreteness of generated research ideas relative to standard prompting approaches. Human evaluations further demonstrate a substantial enhancement in the perceived interestingness of the outputs. By bridging the gap between academic data and creative generation, the Budget AI Researcher offers a practical, free tool for accelerating scientific discovery and lowering the barrier for aspiring researchers. Beyond research ideation, this approach inspires solutions to the broader challenge of generating personalized, context-aware outputs grounded in evolving real-world knowledge. 

---
# CHILL at SemEval-2025 Task 2: You Can't Just Throw Entities and Hope -- Make Your LLM to Get Them Right 

**Authors**: Jaebok Lee, Yonghyun Ryu, Seongmin Park, Yoonjung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.13070)  

**Abstract**: In this paper, we describe our approach for the SemEval 2025 Task 2 on Entity-Aware Machine Translation (EA-MT). Our system aims to improve the accuracy of translating named entities by combining two key approaches: Retrieval Augmented Generation (RAG) and iterative self-refinement techniques using Large Language Models (LLMs). A distinctive feature of our system is its self-evaluation mechanism, where the LLM assesses its own translations based on two key criteria: the accuracy of entity translations and overall translation quality. We demonstrate how these methods work together and effectively improve entity handling while maintaining high-quality translations. 

---
# DoTA-RAG: Dynamic of Thought Aggregation RAG 

**Authors**: Saksorn Ruangtanusak, Natthapath Rungseesiripak, Peerawat Rojratchadakorn, Monthol Charattrakool, Natapong Nitarach  

**Link**: [PDF](https://arxiv.org/pdf/2506.12571)  

**Abstract**: In this paper, we introduce DoTA-RAG (Dynamic-of-Thought Aggregation RAG), a retrieval-augmented generation system optimized for high-throughput, large-scale web knowledge indexes. Traditional RAG pipelines often suffer from high latency and limited accuracy over massive, diverse datasets. DoTA-RAG addresses these challenges with a three-stage pipeline: query rewriting, dynamic routing to specialized sub-indexes, and multi-stage retrieval and ranking. We further enhance retrieval by evaluating and selecting a superior embedding model, re-embedding the large FineWeb-10BT corpus. Moreover, we create a diverse Q&A dataset of 500 questions generated via the DataMorgana setup across a broad range of WebOrganizer topics and formats. DoTA-RAG improves the answer correctness score from 0.752 (baseline, using LiveRAG pre-built vector store) to 1.478 while maintaining low latency, and it achieves a 0.929 correctness score on the Live Challenge Day. These results highlight DoTA-RAG's potential for practical deployment in domains requiring fast, reliable access to large and evolving knowledge sources. 

---
# Datrics Text2SQL: A Framework for Natural Language to SQL Query Generation 

**Authors**: Tetiana Gladkykh, Kyrylo Kirykov  

**Link**: [PDF](https://arxiv.org/pdf/2506.12234)  

**Abstract**: Text-to-SQL systems enable users to query databases using natural language, democratizing access to data analytics. However, they face challenges in understanding ambiguous phrasing, domain-specific vocabulary, and complex schema relationships. This paper introduces Datrics Text2SQL, a Retrieval-Augmented Generation (RAG)-based framework designed to generate accurate SQL queries by leveraging structured documentation, example-based learning, and domain-specific rules. The system builds a rich Knowledge Base from database documentation and question-query examples, which are stored as vector embeddings and retrieved through semantic similarity. It then uses this context to generate syntactically correct and semantically aligned SQL code. The paper details the architecture, training methodology, and retrieval logic, highlighting how the system bridges the gap between user intent and database structure without requiring SQL expertise. 

---
# SocialCredit+ 

**Authors**: Thabassum Aslam, Anees Aslam  

**Link**: [PDF](https://arxiv.org/pdf/2506.12099)  

**Abstract**: SocialCredit+ is AI powered credit scoring system that leverages publicly available social media data to augment traditional credit evaluation. It uses a conversational banking assistant to gather user consent and fetch public profiles. Multimodal feature extractors analyze posts, bios, images, and friend networks to generate a rich behavioral profile. A specialized Sharia-compliance layer flags any non-halal indicators and prohibited financial behavior based on Islamic ethics. The platform employs a retrieval-augmented generation module: an LLM accesses a domain specific knowledge base to generate clear, text-based explanations for each decision. We describe the end-to-end architecture and data flow, the models used, and system infrastructure. Synthetic scenarios illustrate how social signals translate into credit-score factors. This paper emphasizes conceptual novelty, compliance mechanisms, and practical impact, targeting AI researchers, fintech practitioners, ethical banking jurists, and investors. 

---
# EMERGENT: Efficient and Manipulation-resistant Matching using GFlowNets 

**Authors**: Mayesha Tasnim, Erman Acar, Sennay Ghebreab  

**Link**: [PDF](https://arxiv.org/pdf/2506.12033)  

**Abstract**: The design of fair and efficient algorithms for allocating public resources, such as school admissions, housing, or medical residency, has a profound social impact. In one-sided matching problems, where individuals are assigned to items based on ranked preferences, a fundamental trade-off exists between efficiency and strategyproofness. Existing algorithms like Random Serial Dictatorship (RSD), Probabilistic Serial (PS), and Rank Minimization (RM) capture only one side of this trade-off: RSD is strategyproof but inefficient, while PS and RM are efficient but incentivize manipulation. We propose EMERGENT, a novel application of Generative Flow Networks (GFlowNets) to one-sided matching, leveraging its ability to sample diverse, high-reward solutions. In our approach, efficient and manipulation-resistant matches emerge naturally: high-reward solutions yield efficient matches, while the stochasticity of GFlowNets-based outputs reduces incentives for manipulation. Experiments show that EMERGENT outperforms RSD in rank efficiency while significantly reducing strategic vulnerability compared to matches produced by RM and PS. Our work highlights the potential of GFlowNets for applications involving social choice mechanisms, where it is crucial to balance efficiency and manipulability. 

---
# LTRR: Learning To Rank Retrievers for LLMs 

**Authors**: To Eun Kim, Fernando Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.13743)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems typically rely on a single fixed retriever, despite growing evidence that no single retriever performs optimally across all query types. In this paper, we explore a query routing approach that dynamically selects from a pool of retrievers based on the query, using both train-free heuristics and learned routing models. We frame routing as a learning-to-rank (LTR) problem and introduce LTRR, a framework that learns to rank retrievers by their expected utility gain to downstream LLM performance. Our experiments, conducted on synthetic QA data with controlled query type variations, show that routing-based RAG systems can outperform the best single-retriever-based systems. Performance gains are especially pronounced in models trained with the Answer Correctness (AC) metric and with pairwise learning approaches, especially with XGBoost. We also observe improvements in generalization to out-of-distribution queries. As part of the SIGIR 2025 LiveRAG challenge, our submitted system demonstrated the practical viability of our approach, achieving competitive performance in both answer correctness and faithfulness. These findings highlight the importance of both training methodology and metric selection in query routing for RAG systems. 

---
# FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation 

**Authors**: Zhuocheng Zhang, Yang Feng, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12494)  

**Abstract**: Retrieval-Augmented Generation (RAG) plays a pivotal role in modern large language model applications, with numerous existing frameworks offering a wide range of functionalities to facilitate the development of RAG systems. However, we have identified several persistent challenges in these frameworks, including difficulties in algorithm reproduction and sharing, lack of new techniques, and high system overhead. To address these limitations, we introduce \textbf{FlexRAG}, an open-source framework specifically designed for research and prototyping. FlexRAG supports text-based, multimodal, and network-based RAG, providing comprehensive lifecycle support alongside efficient asynchronous processing and persistent caching capabilities. By offering a robust and flexible solution, FlexRAG enables researchers to rapidly develop, deploy, and share advanced RAG systems. Our toolkit and resources are available at \href{this https URL}{this https URL}. 

---
# T$^2$-RAGBench: Text-and-Table Benchmark for Evaluating Retrieval-Augmented Generation 

**Authors**: Jan Strich, Enes Kutay Isgorur, Maximilian Trescher, Chris Biemann, Martin Semmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.12071)  

**Abstract**: While most financial documents contain a combination of textual and tabular information, robust Retrieval-Augmented Generation (RAG) systems are essential for effectively accessing and reasoning over such content to perform complex numerical tasks. This paper introduces T$^2$-RAGBench, a benchmark comprising 32,908 question-context-answer triples, designed to evaluate RAG methods on real-world financial data. Unlike typical QA datasets that operate under Oracle-context settings, where the relevant context is explicitly provided, T$^2$-RAGBench challenges models to first retrieve the correct context before conducting numerical reasoning. Existing QA datasets involving text and tables typically contain context-dependent questions, which may yield multiple correct answers depending on the provided context. To address this, we transform these datasets into a context-independent format, enabling reliable RAG evaluation. We conduct a comprehensive evaluation of popular RAG methods. Our analysis identifies Hybrid BM25, a technique that combines dense and sparse vectors, as the most effective approach for text-and-table data. However, results demonstrate that T$^2$-RAGBench remains challenging even for SOTA LLMs and RAG methods. Further ablation studies examine the impact of embedding models and corpus size on retrieval performance. T$^2$-RAGBench provides a realistic and rigorous benchmark for existing RAG methods on text-and-table data. Code and dataset are available online. 

---
# Tree-Based Text Retrieval via Hierarchical Clustering in RAGFrameworks: Application on Taiwanese Regulations 

**Authors**: Chia-Heng Yu, Yen-Lung Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2506.13607)  

**Abstract**: Traditional Retrieval-Augmented Generation (RAG) systems employ brute-force inner product search to retrieve the top-k most similar documents, then combined with the user query and passed to a language model. This allows the model to access external knowledge and reduce hallucinations. However, selecting an appropriate k value remains a significant challenge in practical applications: a small k may fail to retrieve sufficient information, while a large k can introduce excessive and irrelevant content. To address this, we propose a hierarchical clustering-based retrieval method that eliminates the need to predefine k. Our approach maintains the accuracy and relevance of system responses while adaptively selecting semantically relevant content. In the experiment stage, we applied our method to a Taiwanese legal dataset with expert-graded queries. The results show that our approach achieves superior performance in expert evaluations and maintains high precision while eliminating the need to predefine k, demonstrating improved accuracy and interpretability in legal text retrieval tasks. Our framework is simple to implement and easily integrates with existing RAG pipelines, making it a practical solution for real-world applications under limited resources. 

---
# Efficient Neuro-Symbolic Retrieval-Augmented Generation through Adaptive Query Routing 

**Authors**: Safayat Bin Hakim, Muhammad Adil, Alvaro Velasquez, Houbing Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.12981)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems address factual inconsistencies in Large Language Models by grounding generation in external knowledge, yet they face a fundamental efficiency problem: simple queries consume computational resources equivalent to complex multi-hop reasoning tasks. We present SymRAG, a neuro-symbolic framework that introduces adaptive query routing based on real-time complexity and system load assessments. SymRAG dynamically selects symbolic, neural, or hybrid processing paths to align resource use with query demands. Evaluated on 2,000 queries from HotpotQA and DROP using Llama-3.2-3B and Mistral-7B models, SymRAG achieves 97.6--100.0% exact match accuracy with significantly lower CPU utilization (3.6--6.2%) and processing time (0.985--3.165s). Disabling adaptive logic results in 169--1151% increase in processing time, highlighting the framework's impact. These results underscore the potential of adaptive neuro-symbolic routing for scalable, sustainable AI systems. 

---
