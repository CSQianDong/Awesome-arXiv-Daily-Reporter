# BifrostRAG: Bridging Dual Knowledge Graphs for Multi-Hop Question Answering in Construction Safety 

**Authors**: Yuxin Zhang, Xi Wang, Mo Hu, Zhenyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13625)  

**Abstract**: Information retrieval and question answering from safety regulations are essential for automated construction compliance checking but are hindered by the linguistic and structural complexity of regulatory text. Many compliance-related queries are multi-hop, requiring synthesis of information across interlinked clauses. This poses a challenge for traditional retrieval-augmented generation (RAG) systems. To overcome this, we introduce BifrostRAG: a dual-graph RAG-integrated system that explicitly models both linguistic relationships (via an Entity Network Graph) and document structure (via a Document Navigator Graph). This architecture powers a hybrid retrieval mechanism that combines graph traversal with vector-based semantic search, enabling large language models to reason over both the meaning and the structure of the text. Evaluation on a multi-hop question dataset shows that BifrostRAG achieves 92.8 percent precision, 85.5 percent recall, and an F1 score of 87.3 percent. These results significantly outperform vector-only and graph-only RAG baselines that represent current leading approaches. Error analysis further highlights the comparative advantages of our hybrid method over single-modality RAGs. These findings establish BifrostRAG as a robust knowledge engine for LLM-driven compliance checking. Its dual-graph, hybrid retrieval mechanism offers a transferable blueprint for navigating complex technical documents across knowledge-intensive engineering domains. 

---
# RAG-based Architectures for Drug Side Effect Retrieval in LLMs 

**Authors**: Shad Nygren, Pinar Avci, Andre Daniels, Reza Rassol, Afshin Beheshti, Diego Galeano  

**Link**: [PDF](https://arxiv.org/pdf/2507.13822)  

**Abstract**: Drug side effects are a major global health concern, necessitating advanced methods for their accurate detection and analysis. While Large Language Models (LLMs) offer promising conversational interfaces, their inherent limitations, including reliance on black-box training data, susceptibility to hallucinations, and lack of domain-specific knowledge, hinder their reliability in specialized fields like pharmacovigilance. To address this gap, we propose two architectures: Retrieval-Augmented Generation (RAG) and GraphRAG, which integrate comprehensive drug side effect knowledge into a Llama 3 8B language model. Through extensive evaluations on 19,520 drug side effect associations (covering 976 drugs and 3,851 side effect terms), our results demonstrate that GraphRAG achieves near-perfect accuracy in drug side effect retrieval. This framework offers a highly accurate and scalable solution, signifying a significant advancement in leveraging LLMs for critical pharmacovigilance applications. 

---
# DyG-RAG: Dynamic Graph Retrieval-Augmented Generation with Event-Centric Reasoning 

**Authors**: Qingyun Sun, Jiaqi Yuan, Shan He, Xiao Guan, Haonan Yuan, Xingcheng Fu, Jianxin Li, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13396)  

**Abstract**: Graph Retrieval-Augmented Generation has emerged as a powerful paradigm for grounding large language models with external structured knowledge. However, existing Graph RAG methods struggle with temporal reasoning, due to their inability to model the evolving structure and order of real-world events. In this work, we introduce DyG-RAG, a novel event-centric dynamic graph retrieval-augmented generation framework designed to capture and reason over temporal knowledge embedded in unstructured text. To eliminate temporal ambiguity in traditional retrieval units, DyG-RAG proposes Dynamic Event Units (DEUs) that explicitly encode both semantic content and precise temporal anchors, enabling accurate and interpretable time-aware retrieval. To capture temporal and causal dependencies across events, DyG-RAG constructs an event graph by linking DEUs that share entities and occur close in time, supporting efficient and meaningful multi-hop reasoning. To ensure temporally consistent generation, DyG-RAG introduces an event timeline retrieval pipeline that retrieves event sequences via time-aware traversal, and proposes a Time Chain-of-Thought strategy for temporally grounded answer generation. This unified pipeline enables DyG-RAG to retrieve coherent, temporally ordered event sequences and to answer complex, time-sensitive queries that standard RAG systems cannot resolve. Extensive experiments on temporal QA benchmarks demonstrate that DyG-RAG significantly improves the accuracy and recall of three typical types of temporal reasoning questions, paving the way for more faithful and temporal-aware generation. DyG-RAG is available at this https URL. 

---
