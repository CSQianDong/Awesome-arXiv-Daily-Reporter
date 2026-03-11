# AI Act Evaluation Benchmark: An Open, Transparent, and Reproducible Evaluation Dataset for NLP and RAG Systems 

**Authors**: Athanasios Davvetas, Michael Papademas, Xenia Ziouvelou, Vangelis Karkaletsis  

**Link**: [PDF](https://arxiv.org/pdf/2603.09435)  

**Abstract**: The rapid rollout of AI in heterogeneous public and societal sectors has subsequently escalated the need for compliance with regulatory standards and frameworks. The EU AI Act has emerged as a landmark in the regulatory landscape. The development of solutions that elicit the level of AI systems' compliance with such standards is often limited by the lack of resources, hindering the semi-automated or automated evaluation of their performance. This generates the need for manual work, which is often error-prone, resource-limited or limited to cases not clearly described by the regulation. This paper presents an open, transparent, and reproducible method of creating a resource that facilitates the evaluation of NLP models with a strong focus on RAG systems. We have developed a dataset that contain the tasks of risk-level classification, article retrieval, obligation generation, and question-answering for the EU AI Act. The dataset files are in a machine-to-machine appropriate format. To generate the files, we utilise domain knowledge as an exegetical basis, combining with the processing and reasoning power of large language models to generate scenarios along with the respective tasks. Our methodology demonstrates a way to harness language models for grounded generation with high document relevancy. Besides, we overcome limitations such as navigating the decision boundaries of risk-levels that are not explicitly defined within the EU AI Act, such as limited and minimal cases. Finally, we demonstrate our dataset's effectiveness by evaluating a RAG-based solution that reaches 0.87 and 0.85 F1-score for prohibited and high-risk scenarios. 

---
# Explainable Innovation Engine: Dual-Tree Agent-RAG with Methods-as-Nodes and Verifiable Write-Back 

**Authors**: Renwei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2603.09192)  

**Abstract**: Retrieval-augmented generation (RAG) improves factual grounding, yet most systems rely on flat chunk retrieval and provide limited control over multi-step synthesis. We propose an Explainable Innovation Engine that upgrades the knowledge unit from text chunks to methods-as-nodes. The engine maintains a weighted method provenance tree for traceable derivations and a hierarchical clustering abstraction tree for efficient top-down navigation. At inference time, a strategy agent selects explicit synthesis operators (e.g., induction, deduction, analogy), composes new method nodes, and records an auditable trajectory. A verifier-scorer layer then prunes low-quality candidates and writes validated nodes back to support continual growth. Expert evaluation across six domains and multiple backbones shows consistent gains over a vanilla baseline, with the largest improvements on derivation-heavy settings, and ablations confirm the complementary roles of provenance backtracking and pruning. These results suggest a practical path toward controllable, explainable, and verifiable innovation in agentic RAG systems. Code is available at the project GitHub repository this https URL. 

---
# MITRA: An AI Assistant for Knowledge Retrieval in Physics Collaborations 

**Authors**: Abhishikth Mallampalli, Sridhara Dasu  

**Link**: [PDF](https://arxiv.org/pdf/2603.09800)  

**Abstract**: Large-scale scientific collaborations, such as the Compact Muon Solenoid (CMS) at CERN, produce a vast and ever-growing corpus of internal documentation. Navigating this complex information landscape presents a significant challenge for both new and experienced researchers, hindering knowledge sharing and slowing down the pace of scientific discovery. To address this, we present a prototype of MITRA, a Retrieval-Augmented Generation (RAG) based system, designed to answer specific, context-aware questions about physics analyses. MITRA employs a novel, automated pipeline using Selenium for document retrieval from internal databases and Optical Character Recognition (OCR) with layout parsing for high-fidelity text extraction. Crucially, MITRA's entire framework, from the embedding model to the Large Language Model (LLM), is hosted on-premise, ensuring that sensitive collaboration data remains private. We introduce a two-tiered vector database architecture that first identifies the relevant analysis from abstracts before focusing on the full documentation, resolving potential ambiguities between different analyses. We demonstrate the prototype's superior retrieval performance against a standard keyword-based baseline on realistic queries and discuss future work towards developing a comprehensive research agent for large experimental collaborations. 

---
# TaSR-RAG: Taxonomy-guided Structured Reasoning for Retrieval-Augmented Generation 

**Authors**: Jiashuo Sun, Yixuan Xie, Jimeng Shi, Shaowen Wang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.09341)  

**Abstract**: Retrieval-Augmented Generation (RAG) helps large language models (LLMs) answer knowledge-intensive and time-sensitive questions by conditioning generation on external evidence. However, most RAG systems still retrieve unstructured chunks and rely on one-shot generation, which often yields redundant context, low information density, and brittle multi-hop reasoning. While structured RAG pipelines can improve grounding, they typically require costly and error-prone graph construction or impose rigid entity-centric structures that do not align with the query's reasoning chain.
We propose \textsc{TaSR-RAG}, a taxonomy-guided structured reasoning framework for evidence selection. We represent both queries and documents as relational triples, and constrain entity semantics with a lightweight two-level taxonomy to balance generalization and precision. Given a complex question, \textsc{TaSR-RAG} decomposes it into an ordered sequence of triple sub-queries with explicit latent variables, then performs step-wise evidence selection via hybrid triple matching that combines semantic similarity over raw triples with structural consistency over typed triples.
By maintaining an explicit entity binding table across steps, \textsc{TaSR-RAG} resolves intermediate variables and reduces entity conflation without explicit graph construction or exhaustive search. Experiments on multiple multi-hop question answering benchmarks show that \textsc{TaSR-RAG} consistently outperforms strong RAG and structured-RAG baselines by up to 14\%, while producing clearer evidence attribution and more faithful reasoning traces. 

---
# Beyond Relevance: On the Relationship Between Retrieval and RAG Information Coverage 

**Authors**: Saron Samuel, Alexander Martin, Eugene Yang, Andrew Yates, Dawn Lawrie, Ian Soborof, Laura Dietz, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2603.08819)  

**Abstract**: Retrieval-augmented generation (RAG) systems combine document retrieval with a generative model to address complex information seeking tasks like report generation. While the relationship between retrieval quality and generation effectiveness seems intuitive, it has not been systematically studied. We investigate whether upstream retrieval metrics can serve as reliable early indicators of the final generated response's information coverage. Through experiments across two text RAG benchmarks (TREC NeuCLIR 2024 and TREC RAG 2024) and one multimodal benchmark (WikiVideo), we analyze 15 text retrieval stacks and 10 multimodal retrieval stacks across four RAG pipelines and multiple evaluation frameworks (Auto-ARGUE and MiRAGE). Our findings demonstrate strong correlations between coverage-based retrieval metrics and nugget coverage in generated responses at both topic and system levels. This relationship holds most strongly when retrieval objectives align with generation goals, though more complex iterative RAG pipelines can partially decouple generation quality from retrieval effectiveness. These findings provide empirical support for using retrieval metrics as proxies for RAG performance. 

---
# Evaluation of LLMs in retrieving food and nutritional context for RAG systems 

**Authors**: Maks Požarnik Vavken, Matevž Ogrinc, Tome Eftimov, Barbara Koroušić Seljak  

**Link**: [PDF](https://arxiv.org/pdf/2603.09704)  

**Abstract**: In this article, we evaluate four Large Language Models (LLMs) and their effectiveness at retrieving data within a specialized Retrieval-Augmented Generation (RAG) system, using a comprehensive food composition database. Our method is focused on the LLMs ability to translate natural language queries into structured metadata filters, enabling efficient retrieval via a Chroma vector database. By achieving high accuracy in this critical retrieval step, we demonstrate that LLMs can serve as an accessible, high-performance tool, drastically reducing the manual effort and technical expertise previously required for domain experts, such as food compilers and nutritionists, to leverage complex food and nutrition data. However, despite the high performance on easy and moderately complex queries, our analysis of difficult questions reveals that reliable retrieval remains challenging when queries involve non-expressible constraints. These findings demonstrate that LLM-driven metadata filtering excels when constraints can be explicitly expressed, but struggles when queries exceed the representational scope of the metadata format. 

---
# Overview of the TREC 2025 Retrieval Augmented Generation (RAG) Track 

**Authors**: Shivani Upadhyay, Nandan Thakur, Ronak Pradeep, Nick Craswell, Daniel Campos, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.09891)  

**Abstract**: The second edition of the TREC Retrieval Augmented Generation (RAG) Track advances research on systems that integrate retrieval and generation to address complex, real-world information needs. Building on the foundation of the inaugural 2024 track, this year's challenge introduces long, multi-sentence narrative queries to better reflect the deep search task with the growing demand for reasoning-driven responses. Participants are tasked with designing pipelines that combine retrieval and generation while ensuring transparency and factual grounding. The track leverages the MS MARCO V2.1 corpus and employs a multi-layered evaluation framework encompassing relevance assessment, response completeness, attribution verification, and agreement analysis. By emphasizing multi-faceted narratives and attribution-rich answers from over 150 submissions this year, the TREC 2025 RAG Track aims to foster innovation in creating trustworthy, context-aware systems for retrieval augmented generation. 

---
