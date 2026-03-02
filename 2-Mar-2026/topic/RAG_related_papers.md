# An Agentic LLM Framework for Adverse Media Screening in AML Compliance 

**Authors**: Pavel Chernakov, Sasan Jafarnejad, Raphaël Frank  

**Link**: [PDF](https://arxiv.org/pdf/2602.23373)  

**Abstract**: Adverse media screening is a critical component of anti-money laundering (AML) and know-your-customer (KYC) compliance processes in financial institutions. Traditional approaches rely on keyword-based searches that generate high false-positive rates or require extensive manual review. We present an agentic system that leverages Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to automate adverse media screening. Our system implements a multi-step approach where an LLM agent searches the web, retrieves and processes relevant documents, and computes an Adverse Media Index (AMI) score for each subject. We evaluate our approach using multiple LLM backends on a dataset comprising Politically Exposed Persons (PEPs), persons from regulatory watchlists, and sanctioned persons from OpenSanctions and clean names from academic sources, demonstrating the system's ability to distinguish between high-risk and low-risk individuals. 

---
# TRIZ-RAGNER: A Retrieval-Augmented Large Language Model for TRIZ-Aware Named Entity Recognition in Patent-Based Contradiction Mining 

**Authors**: Zitong Xu, Yuqing Wu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.23656)  

**Abstract**: TRIZ-based contradiction mining is a fundamental task in patent analysis and systematic innovation, as it enables the identification of improving and worsening technical parameters that drive inventive problem solving. However, existing approaches largely rely on rule-based systems or traditional machine learning models, which struggle with semantic ambiguity, domain dependency, and limited generalization when processing complex patent language. Recently, large language models (LLMs) have shown strong semantic understanding capabilities, yet their direct application to TRIZ parameter extraction remains challenging due to hallucination and insufficient grounding in structured TRIZ knowledge. To address these limitations, this paper proposes TRIZ-RAGNER, a retrieval-augmented large language model framework for TRIZ-aware named entity recognition in patent-based contradiction mining. TRIZ-RAGNER reformulates contradiction mining as a semantic-level NER task and integrates dense retrieval over a TRIZ knowledge base, cross-encoder reranking for context refinement, and structured LLM prompting to extract improving and worsening parameters from patent sentences. By injecting domain-specific TRIZ knowledge into the LLM reasoning process, the proposed framework effectively reduces semantic noise and improves extraction consistency. Experiments on the PaTRIZ dataset demonstrate that TRIZ-RAGNER consistently outperforms traditional sequence labeling models and LLM-based baselines. The proposed framework achieves a precision of 85.6%, a recall of 82.9%, and an F1-score of 84.2% in TRIZ contradiction pair identification. Compared with the strongest baseline using prompt-enhanced GPT, TRIZ-RAGNER yields an absolute F1-score improvement of 7.3 percentage points, confirming the effectiveness of retrieval-augmented TRIZ knowledge grounding for robust and accurate patent-based contradiction mining. 

---
# Higress-RAG: A Holistic Optimization Framework for Enterprise Retrieval-Augmented Generation via Dual Hybrid Retrieval, Adaptive Routing, and CRAG 

**Authors**: Weixi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.23374)  

**Abstract**: The integration of Large Language Models (LLMs) into enterprise knowledge management systems has been catalyzed by the Retrieval-Augmented Generation (RAG) paradigm, which augments parametric memory with non-parametric external data. However, the transition from proof-of-concept to production-grade RAG systems is hindered by three persistent challenges: low retrieval precision for complex queries, high rates of hallucination in the generation phase, and unacceptable latency for real-time applications. This paper presents a comprehensive analysis of the Higress RAG MCP Server, a novel, enterprise-centric architecture designed to resolve these bottlenecks through a "Full-Link Optimization" strategy. Built upon the Model Context Protocol (MCP), the system introduces a layered architecture that orchestrates a sophisticated pipeline of Adaptive Routing, Semantic Caching, Hybrid Retrieval, and Corrective RAG (CRAG). We detail the technical implementation of key innovations, including the Higress-Native Splitter for structure-aware data ingestion, the application of Reciprocal Rank Fusion (RRF) for merging dense and sparse retrieval signals, and a 50ms-latency Semantic Caching mechanism with dynamic thresholding. Experimental evaluations on domain-specific Higress technical documentation and blogs verify the system's architectural robustness. The results demonstrate that by optimizing the entire retrieval lifecycle - from pre-retrieval query rewriting to post-retrieval corrective evaluation - the Higress RAG system offers a scalable, hallucination-resistant solution for enterprise AI deployment. 

---
# Domain-Partitioned Hybrid RAG for Legal Reasoning: Toward Modular and Explainable Legal AI for India 

**Authors**: Rakshita Goel, S Pranav Kumar, Anmol Agrawal, Divyan Poddar, Pratik Narang, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2602.23371)  

**Abstract**: Legal research in India involves navigating long and heterogeneous documents spanning statutes, constitutional provisions, penal codes, and judicial precedents, where purely keyword-based or embedding-only retrieval systems often fail to support structured legal reasoning. Recent retrieval augmented generation (RAG) approaches improve grounding but struggle with multi-hop reasoning, citation chaining, and cross-domain dependencies inherent to legal texts.
We propose a domain partitioned hybrid RAG and Knowledge Graph architecture designed specifically for Indian legal research. The system integrates three specialized RAG pipelines covering Supreme Court case law, statutory and constitutional texts, and the Indian Penal Code, each optimized for domain specific retrieval. To enable relational reasoning beyond semantic similarity, we construct a Neo4j based Legal Knowledge Graph capturing structured relationships among cases, statutes, IPC sections, judges, and citations. An LLM driven agentic orchestrator dynamically routes queries across retrieval modules and the knowledge graph, fusing evidence into grounded and citation aware responses.
We evaluate the system using a 40 question synthetic legal question answer benchmark curated from authoritative Indian legal sources and assessed via an LLM as a Judge framework. Results show that the hybrid architecture achieves a 70 percent pass rate, substantially outperforming a RAG only baseline at 37.5 percent, with marked improvements in completeness and legal reasoning quality. These findings demonstrate that combining domain partitioned retrieval with structured relational knowledge provides a scalable and interpretable foundation for advanced legal AI systems in the Indian judicial context. 

---
# CLFEC: A New Task for Unified Linguistic and Factual Error Correction in paragraph-level Chinese Professional Writing 

**Authors**: Jian Kai, Zidong Zhang, Jiwen Chen, Zhengxiang Wu, Songtao Sun, Fuyang Li, Yang Cao, Qiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.23845)  

**Abstract**: Chinese text correction has traditionally focused on spelling and grammar, while factual error correction is usually treated separately. However, in paragraph-level Chinese professional writing, linguistic (word/grammar/punctuation) and factual errors frequently co-occur and interact, making unified correction both necessary and challenging. This paper introduces CLFEC (Chinese Linguistic & Factual Error Correction), a new task for joint linguistic and factual correction. We construct a mixed, multi-domain Chinese professional writing dataset spanning current affairs, finance, law, and medicine. We then conduct a systematic study of LLM-based correction paradigms, from prompting to retrieval-augmented generation (RAG) and agentic workflows. The analysis reveals practical challenges, including limited generalization of specialized correction models, the need for evidence grounding for factual repair, the difficulty of mixed-error paragraphs, and over-correction on clean inputs. Results further show that handling linguistic and factual Error within the same context outperform decoupled processes, and that agentic workflows can be effective with suitable backbone models. Overall, our dataset and empirical findings provide guidance for building reliable, fully automatic proofreading systems in industrial settings. 

---
# AgenticOCR: Parsing Only What You Need for Efficient Retrieval-Augmented Generation 

**Authors**: Zhengren Wang, Dongsheng Ma, Huaping Zhong, Jiayu Li, Wentao Zhang, Bin Wang, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2602.24134)  

**Abstract**: The expansion of retrieval-augmented generation (RAG) into multimodal domains has intensified the challenge for processing complex visual documents, such as financial reports. While page-level chunking and retrieval is a natural starting point, it creates a critical bottleneck: delivering entire pages to the generator introduces excessive extraneous context. This not only overloads the generator's attention mechanism but also dilutes the most salient evidence. Moreover, compressing these information-rich pages into a limited visual token budget further increases the risk of hallucinations. To address this, we introduce AgenticOCR, a dynamic parsing paradigm that transforms optical character recognition (OCR) from a static, full-text process into a query-driven, on-demand extraction system. By autonomously analyzing document layout in a "thinking with images" manner, AgenticOCR identifies and selectively recognizes regions of interest. This approach performs on-demand decompression of visual tokens precisely where needed, effectively decoupling retrieval granularity from rigid page-level chunking. AgenticOCR has the potential to serve as the "third building block" of the visual document RAG stack, operating alongside and enhancing standard Embedding and Reranking modules. Experimental results demonstrate that AgenticOCR improves both the efficiency and accuracy of visual RAG systems, achieving expert-level performance in long document understanding. Code and models are available at this https URL. 

---
# Democratizing GraphRAG: Linear, CPU-Only Graph Retrieval for Multi-Hop QA 

**Authors**: Qizhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.23372)  

**Abstract**: GraphRAG systems improve multi-hop retrieval by modeling structure, but many approaches rely on expensive LLM-based graph construction and GPU-heavy inference. We present SPRIG (Seeded Propagation for Retrieval In Graphs), a CPU-only, linear-time, token-free GraphRAG pipeline that replaces LLM graph building with lightweight NER-driven co-occurrence graphs and uses Personalized PageRank (PPR) for 28% with negligible Recall@10 changes. The results characterize when CPU-friendly graph retrieval helps multi-hop recall and when strong lexical hybrids (RRF) are sufficient, outlining a realistic path to democratizing GraphRAG without token costs or GPU requirements. 

---
# Keyword search is all you need: Achieving RAG-Level Performance without vector databases using agentic tool use 

**Authors**: Shreyas Subramanian, Adewale Akinfaderin, Yanyan Zhang, Ishan Singh, Mani Khanuja, Sandeep Singh, Maira Ladeira Tanke  

**Link**: [PDF](https://arxiv.org/pdf/2602.23368)  

**Abstract**: While Retrieval-Augmented Generation (RAG) has proven effective for generating accurate, context-based responses based on existing knowledge bases, it presents several challenges including retrieval quality dependencies, integration complexity and cost. Recent advances in agentic-RAG and tool-augmented LLM architectures have introduced alternative approaches to information retrieval and processing. We question how much additional value vector databases and semantic search bring to RAG over simple, agentic keyword search in documents for question-answering. In this study, we conducted a systematic comparison between RAG-based systems and tool-augmented LLM agents, specifically evaluating their retrieval mechanisms and response quality when the agent only has access to basic keyword search tools. Our empirical analysis demonstrates that tool-based keyword search implementations within an agentic framework can attain over $90\%$ of the performance metrics compared to traditional RAG systems without using a standing vector database. Our approach is simple to implement, cost effective, and is particularly useful in scenarios requiring frequent updates to knowledge bases. 

---
