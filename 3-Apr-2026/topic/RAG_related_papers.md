# AI in Insurance: Adaptive Questionnaires for Improved Risk Profiling 

**Authors**: Diogo Silva, João Teixeira, Bruno Lima  

**Link**: [PDF](https://arxiv.org/pdf/2604.02034)  

**Abstract**: Insurance application processes often rely on lengthy and standardized questionnaires that struggle to capture individual differences. Moreover, insurers must blindly trust users' responses, increasing the chances of fraud. The ARQuest framework introduces a new approach to underwriting by using Large Language Models (LLMs) and alternative data sources to create personalized and adaptive questionnaires. Techniques such as social media image analysis, geographic data categorization, and Retrieval Augmented Generation (RAG) are used to extract meaningful user insights and guide targeted follow-up questions.
A life insurance system integrated into an industry partner mobile app was tested in two experiments. While traditional questionnaires yielded slightly higher accuracy in risk assessment, adaptive versions powered by GPT models required fewer questions and were preferred by users for their more fluid and engaging experience.
ARQuest shows great potential to improve user satisfaction and streamline insurance processes. With further development, this approach may exceed traditional methods regarding risk accuracy and help drive innovation in the insurance industry. 

---
# Retrieval-Augmented Question Answering over Scientific Literature for the Electron-Ion Collider 

**Authors**: Tina. J. Jat, T. Ghosh, Karthik Suresh  

**Link**: [PDF](https://arxiv.org/pdf/2604.02259)  

**Abstract**: To harness the power of Language Models in answering domain specific specialized technical questions, Retrieval Augmented Generation (RAG) is been used widely. In this work, we have developed a Q\&A application inspired by the Retrieval Augmented Generation (RAG), which is comprised of an in-house database indexed on the arXiv articles related to the Electron-Ion Collider (EIC) experiment - one of the largest international scientific collaboration and incorporated an open-source LLaMA model for answer generation. This is an extension to it's proceeding application built on proprietary model and Cloud-hosted external knowledge-base for the EIC experiment. This locally-deployed RAG-system offers a cost-effective, resource-constraint alternative solution to build a RAG-assisted Q\&A application on answering domain-specific queries in the field of experimental nuclear physics. This set-up facilitates data-privacy, avoids sending any pre-publication scientific data and information to public domain. Future improvement will expand the knowledge base to encompass heterogeneous EIC-related publications and reports and upgrade the application pipeline orchestration to the LangGraph framework. 

---
# Optimizing RAG Rerankers with LLM Feedback via Reinforcement Learning 

**Authors**: Yuhang Wu, Xiangqing Shen, Fanfan Wang, Cangqi Zhou, Zhen Wu, Xinyu Dai, Rui Xia  

**Link**: [PDF](https://arxiv.org/pdf/2604.02091)  

**Abstract**: Rerankers play a pivotal role in refining retrieval results for Retrieval-Augmented Generation. However, current reranking models are typically optimized on static human annotated relevance labels in isolation, decoupled from the downstream generation process. This isolation leads to a fundamental misalignment: documents identified as topically relevant by information retrieval metrics often fail to provide the actual utility required by the LLM for precise answer generation. To bridge this gap, we introduce ReRanking Preference Optimization (RRPO), a reinforcement learning framework that directly aligns reranking with the LLM's generation quality. By formulating reranking as a sequential decision-making process, RRPO optimizes for context utility using LLM feedback, thereby eliminating the need for expensive human annotations. To ensure training stability, we further introduce a reference-anchored deterministic baseline. Extensive experiments on knowledge-intensive benchmarks demonstrate that RRPO significantly outperforms strong baselines, including the powerful list-wise reranker RankZephyr. Further analysis highlights the versatility of our framework: it generalizes seamlessly to diverse readers (e.g., GPT-4o), integrates orthogonally with query expansion modules like Query2Doc, and remains robust even when trained with noisy supervisors. 

---
# Procedural Knowledge at Scale Improves Reasoning 

**Authors**: Di Wu, Devendra Singh Sachan, Wen-tau Yih, Mingda Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.01348)  

**Abstract**: Test-time scaling has emerged as an effective way to improve language models on challenging reasoning tasks. However, most existing methods treat each problem in isolation and do not systematically reuse knowledge from prior reasoning trajectories. In particular, they underutilize procedural knowledge: how to reframe a problem, choose an approach, and verify or backtrack when needed. We introduce Reasoning Memory, a retrieval-augmented generation (RAG) framework for reasoning models that explicitly retrieves and reuses procedural knowledge at scale. Starting from existing corpora of step-by-step reasoning trajectories, we decompose each trajectory into self-contained subquestion-subroutine pairs, yielding a datastore of 32 million compact procedural knowledge entries. At inference time, a lightweight in-thought prompt lets the model verbalize the core subquestion, retrieve relevant subroutines within its reasoning trace, and reason under diverse retrieved subroutines as implicit procedural priors. Across six math, science, and coding benchmarks, Reasoning Memory consistently outperforms RAG with document, trajectory, and template knowledge, as well as a compute-matched test-time scaling baseline. With a higher inference budget, it improves over no retrieval by up to 19.2% and over the strongest compute-matched baseline by 7.9% across task types. Ablation studies show that these gains come from two key factors: the broad procedural coverage of the source trajectories and our decomposition and retrieval design, which together enable effective extraction and reuse of procedural knowledge. 

---
# From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents 

**Authors**: Meftun Akarsu, Recep Kaan Karaman, Christopher Mierbach  

**Link**: [PDF](https://arxiv.org/pdf/2604.01733)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems critically depend on retrieval quality, yet no systematic comparison of modern retrieval methods exists for heterogeneous documents containing both text and tabular data. We benchmark ten retrieval strategies spanning sparse, dense, hybrid fusion, cross-encoder reranking, query expansion, index augmentation, and adaptive retrieval on a challenging financial QA benchmark of 23,088 queries over 7,318 documents with mixed text-and-table content. We evaluate retrieval quality via Recall@k, MRR, and nDCG, and end-to-end generation quality via Number Match, with paired bootstrap significance testing. Our results show that (1) a two-stage pipeline combining hybrid retrieval with neural reranking achieves Recall@5 of 0.816 and MRR@3 of 0.605, outperforming all single-stage methods by a large margin; (2) BM25 outperforms state-of-the-art dense retrieval on financial documents, challenging the common assumption that semantic search universally dominates; and (3) query expansion methods (HyDE, multi-query) and adaptive retrieval provide limited benefit for precise numerical queries, while contextual retrieval yields consistent gains. We provide ablation studies on fusion methods and reranker depth, actionable cost-accuracy recommendations, and release our full benchmark code. 

---
