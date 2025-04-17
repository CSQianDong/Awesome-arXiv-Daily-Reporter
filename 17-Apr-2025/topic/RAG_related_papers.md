# A Library of LLM Intrinsics for Retrieval-Augmented Generation 

**Authors**: Marina Danilevsky, Kristjan Greenewald, Chulaka Gunasekara, Maeda Hanafi, Lihong He, Yannis Katsis, Krishnateja Killamsetty, Yatin Nandwani, Lucian Popa, Dinesh Raghu, Frederick Reiss, Vraj Shah, Khoi-Nguyen Tran, Huaiyu Zhu, Luis Lastras  

**Link**: [PDF](https://arxiv.org/pdf/2504.11704)  

**Abstract**: In the developer community for large language models (LLMs), there is not yet a clean pattern analogous to a software library, to support very large scale collaboration. Even for the commonplace use case of Retrieval-Augmented Generation (RAG), it is not currently possible to write a RAG application against a well-defined set of APIs that are agreed upon by different LLM providers. Inspired by the idea of compiler intrinsics, we propose some elements of such a concept through introducing a library of LLM Intrinsics for RAG. An LLM intrinsic is defined as a capability that can be invoked through a well-defined API that is reasonably stable and independent of how the LLM intrinsic itself is implemented. The intrinsics in our library are released as LoRA adapters on HuggingFace, and through a software interface with clear structured input/output characteristics on top of vLLM as an inference platform, accompanied in both places with documentation and code. This article describes the intended usage, training details, and evaluations for each intrinsic, as well as compositions of multiple intrinsics. 

---
# NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes 

**Authors**: Tianyang Xu, Haojie Zheng, Chengze Li, Haoxiang Chen, Yixin Liu, Ruoxi Chen, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.11544)  

**Abstract**: Retrieval-augmented generation (RAG) empowers large language models to access external and private corpus, enabling factually consistent responses in specific domains. By exploiting the inherent structure of the corpus, graph-based RAG methods further enrich this process by building a knowledge graph index and leveraging the structural nature of graphs. However, current graph-based RAG approaches seldom prioritize the design of graph structures. Inadequately designed graph not only impede the seamless integration of diverse graph algorithms but also result in workflow inconsistencies and degraded performance. To further unleash the potential of graph for RAG, we propose NodeRAG, a graph-centric framework introducing heterogeneous graph structures that enable the seamless and holistic integration of graph-based methodologies into the RAG workflow. By aligning closely with the capabilities of LLMs, this framework ensures a fully cohesive and efficient end-to-end process. Through extensive experiments, we demonstrate that NodeRAG exhibits performance advantages over previous methods, including GraphRAG and LightRAG, not only in indexing time, query time, and storage efficiency but also in delivering superior question-answering performance on multi-hop benchmarks and open-ended head-to-head evaluations with minimal retrieval tokens. Our GitHub repository could be seen at this https URL. 

---
# ARCeR: an Agentic RAG for the Automated Definition of Cyber Ranges 

**Authors**: Matteo Lupinacci, Francesco Blefari, Francesco Romeo, Francesco Aurelio Pironti, Angelo Furfaro  

**Link**: [PDF](https://arxiv.org/pdf/2504.12143)  

**Abstract**: The growing and evolving landscape of cybersecurity threats necessitates the development of supporting tools and platforms that allow for the creation of realistic IT environments operating within virtual, controlled settings as Cyber Ranges (CRs). CRs can be exploited for analyzing vulnerabilities and experimenting with the effectiveness of devised countermeasures, as well as serving as training environments for building cyber security skills and abilities for IT operators. This paper proposes ARCeR as an innovative solution for the automatic generation and deployment of CRs, starting from user-provided descriptions in a natural language. ARCeR relies on the Agentic RAG paradigm, which allows it to fully exploit state-of-art AI technologies. Experimental results show that ARCeR is able to successfully process prompts even in cases that LLMs or basic RAG systems are not able to cope with. Furthermore, ARCeR is able to target any CR framework provided that specific knowledge is made available to it. 

---
