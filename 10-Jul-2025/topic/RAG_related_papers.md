# Shifting from Ranking to Set Selection for Retrieval Augmented Generation 

**Authors**: Dahyun Lee, Yongrae Jo, Haeju Park, Moontae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.06838)  

**Abstract**: Retrieval in Retrieval-Augmented Generation(RAG) must ensure that retrieved passages are not only individually relevant but also collectively form a comprehensive set. Existing approaches primarily rerank top-k passages based on their individual relevance, often failing to meet the information needs of complex queries in multi-hop question answering. In this work, we propose a set-wise passage selection approach and introduce SETR, which explicitly identifies the information requirements of a query through Chain-of-Thought reasoning and selects an optimal set of passages that collectively satisfy those requirements. Experiments on multi-hop RAG benchmarks show that SETR outperforms both proprietary LLM-based rerankers and open-source baselines in terms of answer correctness and retrieval quality, providing an effective and efficient alternative to traditional rerankers in RAG systems. The code is available at this https URL 

---
# CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured and Context Aware Text Generation with LLMs 

**Authors**: Garapati Keerthana, Manik Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.06715)  

**Abstract**: Large language models (LLMs), including zero-shot and few-shot paradigms, have shown promising capabilities in clinical text generation. However, real-world applications face two key challenges: (1) patient data is highly unstructured, heterogeneous, and scattered across multiple note types and (2) clinical notes are often long and semantically dense, making naive prompting infeasible due to context length constraints and the risk of omitting clinically relevant information.
We introduce CLI-RAG (Clinically Informed Retrieval-Augmented Generation), a domain-specific framework for structured and clinically grounded text generation using LLMs. It incorporates a novel hierarchical chunking strategy that respects clinical document structure and introduces a task-specific dual-stage retrieval mechanism. The global stage identifies relevant note types using evidence-based queries, while the local stage extracts high-value content within those notes creating relevance at both document and section levels.
We apply the system to generate structured progress notes for individual hospital visits using 15 clinical note types from the MIMIC-III dataset. Experiments show that it preserves temporal and semantic alignment across visits, achieving an average alignment score of 87.7%, surpassing the 80.7% baseline from real clinician-authored notes. The generated outputs also demonstrate high consistency across LLMs, reinforcing deterministic behavior essential for reproducibility, reliability, and clinical trust. 

---
# SPEAR: Subset-sampled Performance Evaluation via Automated Ground Truth Generation for RAG 

**Authors**: Zou Yuheng, Wang Yiran, Tian Yuzhu, Zhu Min, Huang Yanhua  

**Link**: [PDF](https://arxiv.org/pdf/2507.06554)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a core approach for enhancing Large Language Models (LLMs), where the effectiveness of the retriever largely determines the overall response quality of RAG systems. Retrievers encompass a multitude of hyperparameters that significantly impact performance outcomes and demonstrate sensitivity to specific applications. Nevertheless, hyperparameter optimization entails prohibitively high computational expenses. Existing evaluation methods suffer from either prohibitive costs or disconnection from domain-specific scenarios. This paper proposes SEARA (Subset sampling Evaluation for Automatic Retriever Assessment), which addresses evaluation data challenges through subset sampling techniques and achieves robust automated retriever evaluation by minimal retrieval facts extraction and comprehensive retrieval metrics. Based on real user queries, this method enables fully automated retriever evaluation at low cost, thereby obtaining optimal retriever for specific business scenarios. We validate our method across classic RAG applications in rednote, including knowledge-based Q&A system and retrieval-based travel assistant, successfully obtaining scenario-specific optimal retrievers. 

---
# Investigating the Robustness of Retrieval-Augmented Generation at the Query Level 

**Authors**: Sezen Perçin, Xin Su, Qutub Sha Syed, Phillip Howard, Aleksei Kuvshinov, Leo Schwinn, Kay-Ulrich Scholl  

**Link**: [PDF](https://arxiv.org/pdf/2507.06956)  

**Abstract**: Large language models (LLMs) are very costly and inefficient to update with new information. To address this limitation, retrieval-augmented generation (RAG) has been proposed as a solution that dynamically incorporates external knowledge during inference, improving factual consistency and reducing hallucinations. Despite its promise, RAG systems face practical challenges-most notably, a strong dependence on the quality of the input query for accurate retrieval. In this paper, we investigate the sensitivity of different components in the RAG pipeline to various types of query perturbations. Our analysis reveals that the performance of commonly used retrievers can degrade significantly even under minor query variations. We study each module in isolation as well as their combined effect in an end-to-end question answering setting, using both general-domain and domain-specific datasets. Additionally, we propose an evaluation framework to systematically assess the query-level robustness of RAG pipelines and offer actionable recommendations for practitioners based on the results of more than 1092 experiments we performed. 

---
# The User-Centric Geo-Experience: An LLM-Powered Framework for Enhanced Planning, Navigation, and Dynamic Adaptation 

**Authors**: Jieren Deng, Aleksandar Cvetkovic, Pak Kiu Chung, Dragomir Yankov, Chiqun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06993)  

**Abstract**: Traditional travel-planning systems are often static and fragmented, leaving them ill-equipped to handle real-world complexities such as evolving environmental conditions and unexpected itinerary disruptions. In this paper, we identify three gaps between existing service providers causing frustrating user experience: intelligent trip planning, precision "last-100-meter" navigation, and dynamic itinerary adaptation. We propose three cooperative agents: a Travel Planning Agent that employs grid-based spatial grounding and map analysis to help resolve complex multi-modal user queries; a Destination Assistant Agent that provides fine-grained guidance for the final navigation leg of each journey; and a Local Discovery Agent that leverages image embeddings and Retrieval-Augmented Generation (RAG) to detect and respond to trip plan disruptions. With evaluations and experiments, our system demonstrates substantial improvements in query interpretation, navigation accuracy, and disruption resilience, underscoring its promise for applications from urban exploration to emergency response. 

---
# Towards LLM-based Root Cause Analysis of Hardware Design Failures 

**Authors**: Siyu Qiu, Muzhi Wang, Raheel Afsharmazayejani, Mohammad Moradi Shahmiri, Benjamin Tan, Hammond Pearce  

**Link**: [PDF](https://arxiv.org/pdf/2507.06512)  

**Abstract**: With advances in large language models (LLMs), new opportunities have emerged to develop tools that support the digital hardware design process. In this work, we explore how LLMs can assist with explaining the root cause of design issues and bugs that are revealed during synthesis and simulation, a necessary milestone on the pathway towards widespread use of LLMs in the hardware design process and for hardware security analysis. We find promising results: for our corpus of 34 different buggy scenarios, OpenAI's o3-mini reasoning model reached a correct determination 100% of the time under pass@5 scoring, with other state of the art models and configurations usually achieving more than 80% performance and more than 90% when assisted with retrieval-augmented generation. 

---
