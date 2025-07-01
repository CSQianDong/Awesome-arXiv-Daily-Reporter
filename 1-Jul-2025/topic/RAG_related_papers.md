# Harnessing AI Agents to Advance Research on Refugee Child Mental Health 

**Authors**: Aditya Shrivastava, Komal Gupta, Shraddha Arora  

**Link**: [PDF](https://arxiv.org/pdf/2506.23992)  

**Abstract**: The international refugee crisis deepens, exposing millions of dis placed children to extreme psychological trauma. This research suggests a com pact, AI-based framework for processing unstructured refugee health data and distilling knowledge on child mental health. We compare two Retrieval-Aug mented Generation (RAG) pipelines, Zephyr-7B-beta and DeepSeek R1-7B, to determine how well they process challenging humanitarian datasets while avoid ing hallucination hazards. By combining cutting-edge AI methods with migration research and child psychology, this study presents a scalable strategy to assist policymakers, mental health practitioners, and humanitarian agencies to better assist displaced children and recognize their mental wellbeing. In total, both the models worked properly but significantly Deepseek R1 is superior to Zephyr with an accuracy of answer relevance 0.91 

---
# Benchmarking Deep Search over Heterogeneous Enterprise Data 

**Authors**: Prafulla Kumar Choubey, Xiangyu Peng, Shilpa Bhagavath, Kung-Hsiang Huang, Caiming Xiong, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23139)  

**Abstract**: We present a new benchmark for evaluating Deep Search--a realistic and complex form of retrieval-augmented generation (RAG) that requires source-aware, multi-hop reasoning over diverse, sparsed, but related sources. These include documents, meeting transcripts, Slack messages, GitHub, and URLs, which vary in structure and often contain human-to-human interactions. We build it using a synthetic data pipeline that simulates business workflows across product planning, development, and support stages, generating interconnected content with realistic noise and multi-hop questions with guaranteed ground-truth answers. We release our benchmark with both answerable and unanswerable queries, and retrieval pool of 39,190 enterprise artifacts, enabling fine-grained evaluation of long-context LLM and RAG systems. Our experiments reveal that even the best-performing agentic RAG methods achieve an average performance score of 32.96 on our benchmark. With further analysis, we highlight retrieval as the main bottleneck: existing methods struggle to conduct deep searches and retrieve all necessary evidence. Consequently, they often reason over partial context, leading to significant performance degradation. 

---
# Weak-to-Strong GraphRAG: Aligning Weak Retrievers with Large Language Models for Graph-based Retrieval Augmented Generation 

**Authors**: Deyu Zou, Yongqiang Chen, Mufei Li, Siqi Miao, Chenxi Liu, Bo Han, James Cheng, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.22518)  

**Abstract**: Graph-based retrieval-augmented generation (RAG) enables large language models (LLMs) to ground responses with structured external knowledge from up-to-date knowledge graphs (KGs) and reduce hallucinations. However, LLMs often rely on a weak retriever in graph-based RAG: I) Due to the lack of ground truth, the retriever is often trained on weak supervision, which often introduces spurious signals to the LLMs. II) Due to the abstraction of graph data, the retrieved knowledge is often presented in unorganized forms. To mitigate the issue, we present Refined Graph-based RAG (ReG) to align weak retrievers to LLMs for graph-based RAG. Specifically, ReG incorporates LLM feedback to get rid of spurious signals and improve the quality of the supervision. Meanwhile, ReG introduces a structure-aware reorganization module to refactor the retrieval results into logically coherent evidence chains. Experiments on prominent benchmarks demonstrate that ReG significantly and consistently brings improvements across different LLM backbones by up to 10%. The improved supervision quality enables ReG to match the state-of-the-art performance with 5% training data and to transfer to out-of-distribution KGs. Notably, when adopted to reasoning-based LLMs, ReG reduces the reasoning token cost by up to 30% and improves the performance by up to 4%. 

---
# Evaluating Hybrid Retrieval Augmented Generation using Dynamic Test Sets: LiveRAG Challenge 

**Authors**: Chase Fensore, Kaustubh Dhole, Joyce C Ho, Eugene Agichtein  

**Link**: [PDF](https://arxiv.org/pdf/2506.22644)  

**Abstract**: We present our submission to the LiveRAG Challenge 2025, which evaluates retrieval-augmented generation (RAG) systems on dynamic test sets using the FineWeb-10BT corpus. Our final hybrid approach combines sparse (BM25) and dense (E5) retrieval methods and then aims to generate relevant and faithful answers with Falcon3-10B-Instruct. Through systematic evaluation on 200 synthetic questions generated with DataMorgana across 64 unique question-user combinations, we demonstrate that neural re-ranking with RankLLaMA improves MAP from 0.523 to 0.797 (52% relative improvement) but introduces prohibitive computational costs (84s vs 1.74s per question). While DSPy-optimized prompting strategies achieved higher semantic similarity (0.771 vs 0.668), their 0% refusal rates raised concerns about over-confidence and generalizability. Our submitted hybrid system without re-ranking achieved 4th place in faithfulness and 11th place in correctness among 25 teams. Analysis across question categories reveals that vocabulary alignment between questions and documents was the strongest predictor of performance on our development set, with document-similar phrasing improving cosine similarity from 0.562 to 0.762. 

---
# Machine Assistant with Reliable Knowledge: Enhancing Student Learning via RAG-based Retrieval 

**Authors**: Yongsheng Lian  

**Link**: [PDF](https://arxiv.org/pdf/2506.23026)  

**Abstract**: We present Machine Assistant with Reliable Knowledge (MARK), a retrieval-augmented question-answering system designed to support student learning through accurate and contextually grounded responses. The system is built on a retrieval-augmented generation (RAG) framework, which integrates a curated knowledge base to ensure factual consistency. To enhance retrieval effectiveness across diverse question types, we implement a hybrid search strategy that combines dense vector similarity with sparse keyword-based retrieval. This dual-retrieval mechanism improves robustness for both general and domain-specific queries. The system includes a feedback loop in which students can rate responses and instructors can review and revise them. Instructor corrections are incorporated into the retrieval corpus, enabling adaptive refinement over time. The system was deployed in a classroom setting as a substitute for traditional office hours, where it successfully addressed a broad range of student queries. It was also used to provide technical support by integrating with a customer-specific knowledge base, demonstrating its ability to handle routine, context-sensitive tasks in applied domains. MARK is publicly accessible at this https URL. 

---
# LLM-Assisted Question-Answering on Technical Documents Using Structured Data-Aware Retrieval Augmented Generation 

**Authors**: Shadman Sobhan, Mohammad Ariful Haque  

**Link**: [PDF](https://arxiv.org/pdf/2506.23136)  

**Abstract**: Large Language Models (LLMs) are capable of natural language understanding and generation. But they face challenges such as hallucination and outdated knowledge. Fine-tuning is one possible solution, but it is resource-intensive and must be repeated with every data update. Retrieval-Augmented Generation (RAG) offers an efficient solution by allowing LLMs to access external knowledge sources. However, traditional RAG pipelines struggle with retrieving information from complex technical documents with structured data such as tables and images. In this work, we propose a RAG pipeline, capable of handling tables and images in documents, for technical documents that support both scanned and searchable formats. Its retrieval process combines vector similarity search with a fine-tuned reranker based on Gemma-2-9b-it. The reranker is trained using RAFT (Retrieval-Augmented Fine-Tuning) on a custom dataset designed to improve context identification for question answering. Our evaluation demonstrates that the proposed pipeline achieves a high faithfulness score of 94% (RAGas) and 96% (DeepEval), and an answer relevancy score of 87% (RAGas) and 93% (DeepEval). Comparative analysis demonstrates that the proposed architecture is superior to general RAG pipelines in terms of table-based questions and handling questions outside context. 

---
# Knowledge Augmented Finetuning Matters in both RAG and Agent Based Dialog Systems 

**Authors**: Yucheng Cai, Yuxuan Wu, Yi Huang, Junlan Feng, Zhijian Ou  

**Link**: [PDF](https://arxiv.org/pdf/2506.22852)  

**Abstract**: Large language models (LLMs) have recently been applied to dialog systems. Despite making progress, LLMs are prone to errors in knowledge-intensive scenarios. Recently, approaches based on retrieval augmented generation (RAG) and agent have emerged to improve the factual accuracy by enhancing the LLMs with knowledge retrieved from external knowledge bases (KBs). This is mostly implemented by prompting the LLMs with instructions, examples and the retrieved knowledge. However, LLMs may have difficulty using the retrieved knowledge effectively for response generation, because they are not well trained to do such generation for specific domains. To mitigate this problem, we propose to finetune the LLMs in the RAG-based and agent-based systems with domain-specific data, together with domain-specific external knowledge, which is called knowledge augmented finetuning (KAFT). We base our study on the MobileCS2 dataset, a real-life customer service dialog dataset that features intensive knowledge interactions, to systematically compare the prompting and KAFT techniques in the RAG-based and agent-based systems. Experiment results show that KAFT substantially surpasses prompting in both RAG and agent systems, particularly in terms of factual accuracy. To the best of our knowledge, this paper represents the first solid empirical work to investigate the KAFT idea. 

---
