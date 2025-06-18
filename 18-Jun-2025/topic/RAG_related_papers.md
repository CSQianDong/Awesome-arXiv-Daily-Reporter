# RAGtifier: Evaluating RAG Generation Approaches of State-of-the-Art RAG Systems for the SIGIR LiveRAG Competition 

**Authors**: Tim Cofala, Oleh Astappiev, William Xion, Hailay Teklehaymanot  

**Link**: [PDF](https://arxiv.org/pdf/2506.14412)  

**Abstract**: Retrieval-Augmented Generation (RAG) enriches Large Language Models (LLMs) by combining their internal, parametric knowledge with external, non-parametric sources, with the goal of improving factual correctness and minimizing hallucinations. The LiveRAG 2025 challenge explores RAG solutions to maximize accuracy on DataMorgana's QA pairs, which are composed of single-hop and multi-hop questions. The challenge provides access to sparse OpenSearch and dense Pinecone indices of the Fineweb 10BT dataset. It restricts model use to LLMs with up to 10B parameters and final answer generation with Falcon-3-10B. A judge-LLM assesses the submitted answers along with human evaluators. By exploring distinct retriever combinations and RAG solutions under the challenge conditions, our final solution emerged using InstructRAG in combination with a Pinecone retriever and a BGE reranker. Our solution achieved a correctness score of 1.13 and a faithfulness score of 0.55, placing fourth in the SIGIR 2025 LiveRAG Challenge. 

---
# RMIT-ADM+S at the SIGIR 2025 LiveRAG Challenge 

**Authors**: Kun Ran, Shuoqi Sun, Khoi Nguyen Dinh Anh, Damiano Spina, Oleg Zendel  

**Link**: [PDF](https://arxiv.org/pdf/2506.14516)  

**Abstract**: This paper presents the RMIT--ADM+S participation in the SIGIR 2025 LiveRAG Challenge. Our Generation-Retrieval-Augmented Generation (GRAG) approach relies on generating a hypothetical answer that is used in the retrieval phase, alongside the original question. GRAG also incorporates a pointwise large language model (LLM)-based re-ranking step prior to final answer generation. We describe the system architecture and the rationale behind our design choices. In particular, a systematic evaluation using the Grid of Points (GoP) framework and N-way ANOVA enabled comparison across multiple configurations, including query variant generation, question decomposition, rank fusion strategies, and prompting techniques for answer generation. Our system achieved a Relevance score of 1.199 and a Faithfulness score of 0.477 on the private leaderboard, placing among the top four finalists in the LiveRAG 2025 Challenge. 

---
# XGraphRAG: Interactive Visual Analysis for Graph-based Retrieval-Augmented Generation 

**Authors**: Ke Wang, Bo Pan, Yingchaojie Feng, Yuwei Wu, Jieyi Chen, Minfeng Zhu, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13782)  

**Abstract**: Graph-based Retrieval-Augmented Generation (RAG) has shown great capability in enhancing Large Language Model (LLM)'s answer with an external knowledge base. Compared to traditional RAG, it introduces a graph as an intermediate representation to capture better structured relational knowledge in the corpus, elevating the precision and comprehensiveness of generation results. However, developers usually face challenges in analyzing the effectiveness of GraphRAG on their dataset due to GraphRAG's complex information processing pipeline and the overwhelming amount of LLM invocations involved during graph construction and query, which limits GraphRAG interpretability and accessibility. This research proposes a visual analysis framework that helps RAG developers identify critical recalls of GraphRAG and trace these recalls through the GraphRAG pipeline. Based on this framework, we develop XGraphRAG, a prototype system incorporating a set of interactive visualizations to facilitate users' analysis process, boosting failure cases collection and improvement opportunities identification. Our evaluation demonstrates the effectiveness and usability of our approach. Our work is open-sourced and available at this https URL. 

---
# Knowledge Compression via Question Generation: Enhancing Multihop Document Retrieval without Fine-tuning 

**Authors**: Anvi Alex Eponon, Moein Shahiki-Tash, Ildar Batyrshin, Christian E. Maldonado-Sifuentes, Grigori Sidorov, Alexander Gelbukh  

**Link**: [PDF](https://arxiv.org/pdf/2506.13778)  

**Abstract**: This study presents a question-based knowledge encoding approach that improves retrieval-augmented generation (RAG) systems without requiring fine-tuning or traditional chunking. We encode textual content using generated questions that span the lexical and semantic space, creating targeted retrieval cues combined with a custom syntactic reranking method.
In single-hop retrieval over 109 scientific papers, our approach achieves a Recall@3 of 0.84, outperforming traditional chunking methods by 60 percent. We also introduce "paper-cards", concise paper summaries under 300 characters, which enhance BM25 retrieval, increasing MRR@3 from 0.56 to 0.85 on simplified technical queries.
For multihop tasks, our reranking method reaches an F1 score of 0.52 with LLaMA2-Chat-7B on the LongBench 2WikiMultihopQA dataset, surpassing chunking and fine-tuned baselines which score 0.328 and 0.412 respectively.
This method eliminates fine-tuning requirements, reduces retrieval latency, enables intuitive question-driven knowledge access, and decreases vector storage demands by 80%, positioning it as a scalable and efficient RAG alternative. 

---
# AviationLLM: An LLM-based Knowledge System for Aviation Training 

**Authors**: Jia'ang Wan, Feng Shen, Fujuan Li, Yanjin Sun, Yan Li, Shiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14336)  

**Abstract**: Aviation training is a core link in ensuring flight safety, improving industry efficiency and promoting sustainable development. It not only involves flight simulation but also requires the learning of a great deal of professional aviation theory knowledge. In the existing training system, the knowledge is mainly imparted by the the instructors. However, the number of instructors is limited and the professional answers obtained from the Internet are not accurate enough, resulting in low training efficiency. To address this, we introduced LLM, but the basic pre-trained model cannot provide accurate answers to professional fields, so we fine-tuned it. Traditional Supervised Fine-Tuning (SFT) risk generating superficially plausible but factually incorrect responses due to insufficient data coverage. To address this, we employ Direct Preference Optimization(DPO). This paper proposes Retrieval-Augmented LLM Alignment via Direct Preference Optimization(RALA-DPO). We select open source pre-trained LLM Qwen and adapt it to aviation theory training through DPO-based domain alignment. Simultaneously, to mitigate hallucinations caused by training data biases, knowledge obsolescence, or domain knowledge gaps, we implement Retrieval-Augmented Generation(RAG) technology that combines generative and retrieval models. RALA-DPO effectively retrieves relevant information from external knowledge bases and delivers precise and high-quality responses through the generative model. Experimental results demonstrate that RALA-DPO can improve accuracy in response to professional aviation knowledge. With integrated RAG mechanisms, this system can further improve the accuracy of answers and achieve zero-cost knowledge updates simultaneously. 

---
# Lightweight Relevance Grader in RAG 

**Authors**: Taehee Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2506.14084)  

**Abstract**: Retrieval-Augmented Generation (RAG) addresses limitations of large language models (LLMs) by leveraging a vector database to provide more accurate and up-to-date information. When a user submits a query, RAG executes a vector search to find relevant documents, which are then used to generate a response. However, ensuring the relevance of retrieved documents with a query would be a big challenge. To address this, a secondary model, known as a relevant grader, can be served to verify its relevance. To reduce computational requirements of a relevant grader, a lightweight small language model is preferred. In this work, we finetuned llama-3.2-1b as a relevant grader and achieved a significant increase in precision from 0.1301 to 0.7750. Its precision is comparable to that of llama-3.1-70b. Our code is available at this https URL. 

---
