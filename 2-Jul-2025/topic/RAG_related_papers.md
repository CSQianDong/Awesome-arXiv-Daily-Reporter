# Question Decomposition for Retrieval-Augmented Generation 

**Authors**: Paul J. L. Ammann, Jonas Golde, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2507.00355)  

**Abstract**: Grounding large language models (LLMs) in verifiable external sources is a well-established strategy for generating reliable answers. Retrieval-augmented generation (RAG) is one such approach, particularly effective for tasks like question answering: it retrieves passages that are semantically related to the question and then conditions the model on this evidence. However, multi-hop questions, such as "Which company among NVIDIA, Apple, and Google made the biggest profit in 2023?," challenge RAG because relevant facts are often distributed across multiple documents rather than co-occurring in one source, making it difficult for standard RAG to retrieve sufficient information. To address this, we propose a RAG pipeline that incorporates question decomposition: (i) an LLM decomposes the original query into sub-questions, (ii) passages are retrieved for each sub-question, and (iii) the merged candidate pool is reranked to improve the coverage and precision of the retrieved evidence. We show that question decomposition effectively assembles complementary documents, while reranking reduces noise and promotes the most relevant passages before answer generation. Although reranking itself is standard, we show that pairing an off-the-shelf cross-encoder reranker with LLM-driven question decomposition bridges the retrieval gap on multi-hop questions and provides a practical, drop-in enhancement, without any extra training or specialized indexing. We evaluate our approach on the MultiHop-RAG and HotpotQA, showing gains in retrieval (MRR@10: +36.7%) and answer accuracy (F1: +11.6%) over standard RAG baselines. 

---
# Read the Docs Before Rewriting: Equip Rewriter with Domain Knowledge via Continual Pre-training 

**Authors**: Qi Wang, Yixuan Cao, Yifan Liu, Jiangtao Zhao, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2507.00477)  

**Abstract**: A Retrieval-Augmented Generation (RAG)-based question-answering (QA) system enhances a large language model's knowledge by retrieving relevant documents based on user queries. Discrepancies between user queries and document phrasings often necessitate query rewriting. However, in specialized domains, the rewriter model may struggle due to limited domain-specific knowledge. To resolve this, we propose the R\&R (Read the doc before Rewriting) rewriter, which involves continual pre-training on professional documents, akin to how students prepare for open-book exams by reviewing textbooks. Additionally, it can be combined with supervised fine-tuning for improved results. Experiments on multiple datasets demonstrate that R\&R excels in professional QA across multiple domains, effectively bridging the query-document gap, while maintaining good performance in general scenarios, thus advancing the application of RAG-based QA systems in specialized fields. 

---
# An AST-guided LLM Approach for SVRF Code Synthesis 

**Authors**: Abanoub E. Abdelmalak, Mohamed A. Elsayed, David Abercrombie, Ilhami Torunoglu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00352)  

**Abstract**: Standard Verification Rule Format (SVRF) is essential for semiconductor applications like Design Rule Check (DRC), Layout Versus Schematic (LVS), and Optical Proximity Correction (OPC) and it faces challenges as advancing nodes create complex design rules that renders traditional SVRF development ineffective and highlight an expertise gap. This paper introduces a novel methodology integrating Abstract Syntax Tree (AST) embedding and Retrieval-Augmented Generation (RAG) for enhanced SVRF code synthesis, ensuring semantic accuracy and error minimization through structural validation with domain-specific insights for precise code generation.
We evaluate different T5-based models and propose an innovative SVRF-specific scoring framework that complements standard metrics like BLEU and ROUGE-L. In our approach, AST provides rigorous structural validation, while RAG infuses relevant domain knowledge, effectively enhancing the code generation workflow.
Testing on a comprehensive benchmark of 740 DRC rule implementations, our methodology demonstrates up to a 40\% improvement in code generation accuracy compared to basic text-based fine-tuning process. This fusion of industry expertise with advanced coding strategies not only optimizes SVRF development under limited dataset constraints but also creates a more intuitive and efficient coding environment. Consequently, users can rapidly iterate through design cycles, reduce manual error correction, and significantly improve overall productivity. 

---
