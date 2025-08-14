# RAGulating Compliance: A Multi-Agent Knowledge Graph for Regulatory QA 

**Authors**: Bhavik Agarwal, Hemant Sunil Jomraj, Simone Kaplunov, Jack Krolick, Viktoria Rojkova  

**Link**: [PDF](https://arxiv.org/pdf/2508.09893)  

**Abstract**: Regulatory compliance question answering (QA) requires precise, verifiable information, and domain-specific expertise, posing challenges for Large Language Models (LLMs). In this work, we present a novel multi-agent framework that integrates a Knowledge Graph (KG) of Regulatory triplets with Retrieval-Augmented Generation (RAG) to address these demands. First, agents build and maintain an ontology-free KG by extracting subject--predicate--object (SPO) triplets from regulatory documents and systematically cleaning, normalizing, deduplicating, and updating them. Second, these triplets are embedded and stored along with their corresponding textual sections and metadata in a single enriched vector database, allowing for both graph-based reasoning and efficient information retrieval. Third, an orchestrated agent pipeline leverages triplet-level retrieval for question answering, ensuring high semantic alignment between user queries and the factual "who-did-what-to-whom" core captured by the graph. Our hybrid system outperforms conventional methods in complex regulatory queries, ensuring factual correctness with embedded triplets, enabling traceability through a unified vector database, and enhancing understanding through subgraph visualization, providing a robust foundation for compliance-driven and broader audit-focused applications. 

---
# LibRec: Benchmarking Retrieval-Augmented LLMs for Library Migration Recommendations 

**Authors**: Junxiao Han, Yarong Wang, Xiaodong Gu, Cuiyun Gao, Yao Wan, Song Han, David Lo, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09791)  

**Abstract**: In this paper, we propose LibRec, a novel framework that integrates the capabilities of LLMs with retrieval-augmented generation(RAG) techniques to automate the recommendation of alternative libraries. The framework further employs in-context learning to extract migration intents from commit messages to enhance the accuracy of its recommendations. To evaluate the effectiveness of LibRec, we introduce LibEval, a benchmark designed to assess the performance in the library migration recommendation task. LibEval comprises 2,888 migration records associated with 2,368 libraries extracted from 2,324 Python repositories. Each migration record captures source-target library pairs, along with their corresponding migration intents and intent types. Based on LibEval, we evaluated the effectiveness of ten popular LLMs within our framework, conducted an ablation study to examine the contributions of key components within our framework, explored the impact of various prompt strategies on the framework's performance, assessed its effectiveness across various intent types, and performed detailed failure case analyses. 

---
# From Ranking to Selection: A Simple but Efficient Dynamic Passage Selector for Retrieval Augmented Generation 

**Authors**: Siyuan Meng, Junming Liu, Yirong Chen, Song Mao, Pinlong Cai, Guohang Yan, Botian Shi, Ding Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09497)  

**Abstract**: Retrieval-augmented generation (RAG) systems are often bottlenecked by their reranking modules, which typically score passages independently and select a fixed Top-K size. This approach struggles with complex multi-hop queries that require synthesizing evidence across multiple documents, creating a trade-off where small K values omit crucial information and large K values introduce noise. To address this, we introduce the Dynamic Passage Selector (DPS), a novel reranking framework that treats passage selection as a supervised learning problem. Unlike traditional point-wise or list-wise methods, DPS is fine-tuned to capture inter-passage dependencies and dynamically select the most relevant set of passages for generation. As a seamless plug-and-play module, DPS requires no modifications to the standard RAG pipeline. Comprehensive evaluations on five benchmarks show that DPS consistently outperforms state-of-the-art rerankers and fine-tuning methods. Notably, on the challenging MuSiQue dataset, DPS improves the F1-score by 30.06% and 15.4% over strong baselines like Qwen3-reranker and RankingGPT, respectively. Our results demonstrate that by enabling adaptive evidence selection, DPS substantially enhances reasoning capabilities in complex RAG scenarios. 

---
# AI Blob! LLM-Driven Recontextualization of Italian Television Archives 

**Authors**: Roberto Balestri  

**Link**: [PDF](https://arxiv.org/pdf/2508.09535)  

**Abstract**: This paper introduces AI Blob!, an experimental system designed to explore the potential of semantic cataloging and Large Language Models (LLMs) for the retrieval and recontextualization of archival television footage. Drawing methodological inspiration from Italian television programs such as Blob (RAI Tre, 1989-), AI Blob! integrates automatic speech recognition (ASR), semantic embeddings, and retrieval-augmented generation (RAG) to organize and reinterpret archival content. The system processes a curated dataset of 1,547 Italian television videos by transcribing audio, segmenting it into sentence-level units, and embedding these segments into a vector database for semantic querying. Upon user input of a thematic prompt, the LLM generates a range of linguistically and conceptually related queries, guiding the retrieval and recombination of audiovisual fragments. These fragments are algorithmically selected and structured into narrative sequences producing montages that emulate editorial practices of ironic juxtaposition and thematic coherence. By foregrounding dynamic, content-aware retrieval over static metadata schemas, AI Blob! demonstrates how semantic technologies can facilitate new approaches to archival engagement, enabling novel forms of automated narrative construction and cultural analysis. The project contributes to ongoing debates in media historiography and AI-driven archival research, offering both a conceptual framework and a publicly available dataset to support further interdisciplinary experimentation. 

---
# Multimodal RAG Enhanced Visual Description 

**Authors**: Amit Kumar Jaiswal, Haiming Liu, Ingo Frommholz  

**Link**: [PDF](https://arxiv.org/pdf/2508.09170)  

**Abstract**: Textual descriptions for multimodal inputs entail recurrent refinement of queries to produce relevant output images. Despite efforts to address challenges such as scaling model size and data volume, the cost associated with pre-training and fine-tuning remains substantial. However, pre-trained large multimodal models (LMMs) encounter a modality gap, characterised by a misalignment between textual and visual representations within a common embedding space. Although fine-tuning can potentially mitigate this gap, it is typically expensive and impractical due to the requirement for extensive domain-driven data. To overcome this challenge, we propose a lightweight training-free approach utilising Retrieval-Augmented Generation (RAG) to extend across the modality using a linear mapping, which can be computed efficiently. During inference, this mapping is applied to images embedded by an LMM enabling retrieval of closest textual descriptions from the training set. These textual descriptions, in conjunction with an instruction, cater as an input prompt for the language model to generate new textual descriptions. In addition, we introduce an iterative technique for distilling the mapping by generating synthetic descriptions via the language model facilitating optimisation for standard utilised image description measures. Experimental results on two benchmark multimodal datasets demonstrate significant improvements. 

---
# Towards Self-cognitive Exploration: Metacognitive Knowledge Graph Retrieval Augmented Generation 

**Authors**: Xujie Yuan, Shimin Di, Jielong Tang, Libin Zheng, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09460)  

**Abstract**: Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) significantly enhances the reasoning capabilities of LargeLanguage Models by leveraging structured knowledge. However, existing KG-RAG frameworks typically operate as open-loop systems, suffering from cognitive blindness, an inability to recognize their exploration deficiencies. This leads to relevance drift and incomplete evidence, which existing self-refinement methods, designed for unstructured text-based RAG, cannot effectively resolve due to the path-dependent nature of graph exploration. To address this challenge, we propose Metacognitive Knowledge Graph Retrieval Augmented Generation (MetaKGRAG), a novel framework inspired by the human metacognition process, which introduces a Perceive-Evaluate-Adjust cycle to enable path-aware, closed-loop refinement. This cycle empowers the system to self-assess exploration quality, identify deficiencies in coverage or relevance, and perform trajectory-connected corrections from precise pivot points. Extensive experiments across five datasets in the medical, legal, and commonsense reasoning domains demonstrate that MetaKGRAG consistently outperforms strong KG-RAG and self-refinement baselines. Our results validate the superiority of our approach and highlight the critical need for path-aware refinement in structured knowledge retrieval. 

---
# Transforming Questions and Documents for Semantically Aligned Retrieval-Augmented Generation 

**Authors**: Seokgi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.09755)  

**Abstract**: We introduce a novel retrieval-augmented generation (RAG) framework tailored for multihop question answering. First, our system uses large language model (LLM) to decompose complex multihop questions into a sequence of single-hop subquestions that guide document retrieval. This decomposition mitigates the ambiguity inherent in multi-hop queries by clearly targeting distinct knowledge facets. Second, instead of embedding raw or chunked documents directly, we generate answerable questions from each document chunk using Qwen3-8B, embed these generated questions, and retrieve relevant chunks via question-question embedding similarity. During inference, the retrieved chunks are then fed along with the original question into the RAG pipeline. We evaluate on three multihop question datasets (MuSiQue, 2WikiMultiHopQa, HotpotQA) from LongBench. Our method improves RAG performacne compared to baseline systems. Our contributions highlight the benefits of using answerable-question embeddings for RAG, and the effectiveness of LLM-based query decomposition for multihop scenarios. 

---
