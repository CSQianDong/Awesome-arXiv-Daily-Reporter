# Constructing and Evaluating Declarative RAG Pipelines in PyTerrier 

**Authors**: Craig Macdonald, Jinyuan Fang, Andrew Parry, Zaiqiao Meng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10802)  

**Abstract**: Search engines often follow a pipeline architecture, where complex but effective reranking components are used to refine the results of an initial retrieval. Retrieval augmented generation (RAG) is an exciting application of the pipeline architecture, where the final component generates a coherent answer for the users from the retrieved documents. In this demo paper, we describe how such RAG pipelines can be formulated in the declarative PyTerrier architecture, and the advantages of doing so. Our PyTerrier-RAG extension for PyTerrier provides easy access to standard RAG datasets and evaluation measures, state-of-the-art LLM readers, and using PyTerrier's unique operator notation, easy-to-build pipelines. We demonstrate the succinctness of indexing and RAG pipelines on standard datasets (including Natural Questions) and how to build on the larger PyTerrier ecosystem with state-of-the-art sparse, learned-sparse, and dense retrievers, and other neural rankers. 

---
# Beyond True or False: Retrieval-Augmented Hierarchical Analysis of Nuanced Claims 

**Authors**: Priyanka Kargupta, Runchu Tian, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.10728)  

**Abstract**: Claims made by individuals or entities are oftentimes nuanced and cannot be clearly labeled as entirely "true" or "false" -- as is frequently the case with scientific and political claims. However, a claim (e.g., "vaccine A is better than vaccine B") can be dissected into its integral aspects and sub-aspects (e.g., efficacy, safety, distribution), which are individually easier to validate. This enables a more comprehensive, structured response that provides a well-rounded perspective on a given problem while also allowing the reader to prioritize specific angles of interest within the claim (e.g., safety towards children). Thus, we propose ClaimSpect, a retrieval-augmented generation-based framework for automatically constructing a hierarchy of aspects typically considered when addressing a claim and enriching them with corpus-specific perspectives. This structure hierarchically partitions an input corpus to retrieve relevant segments, which assist in discovering new sub-aspects. Moreover, these segments enable the discovery of varying perspectives towards an aspect of the claim (e.g., support, neutral, or oppose) and their respective prevalence (e.g., "how many biomedical papers believe vaccine A is more transportable than B?"). We apply ClaimSpect to a wide variety of real-world scientific and political claims featured in our constructed dataset, showcasing its robustness and accuracy in deconstructing a nuanced claim and representing perspectives within a corpus. Through real-world case studies and human evaluation, we validate its effectiveness over multiple baselines. 

---
# Reasoning RAG via System 1 or System 2: A Survey on Reasoning Agentic Retrieval-Augmented Generation for Industry Challenges 

**Authors**: Jintao Liang, Gang Su, Huifeng Lin, You Wu, Rui Zhao, Ziyue Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.10408)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful framework to overcome the knowledge limitations of Large Language Models (LLMs) by integrating external retrieval with language generation. While early RAG systems based on static pipelines have shown effectiveness in well-structured tasks, they struggle in real-world scenarios requiring complex reasoning, dynamic retrieval, and multi-modal integration. To address these challenges, the field has shifted toward Reasoning Agentic RAG, a paradigm that embeds decision-making and adaptive tool use directly into the retrieval process. In this paper, we present a comprehensive review of Reasoning Agentic RAG methods, categorizing them into two primary systems: predefined reasoning, which follows fixed modular pipelines to boost reasoning, and agentic reasoning, where the model autonomously orchestrates tool interaction during inference. We analyze representative techniques under both paradigms, covering architectural design, reasoning strategies, and tool coordination. Finally, we discuss key research challenges and propose future directions to advance the flexibility, robustness, and applicability of reasoning agentic RAG systems. Our collection of the relevant research has been organized into a this https URL. 

---
# TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning 

**Authors**: Xiaohan Yu, Pu Jian, Chong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10380)  

**Abstract**: Retrieval-Augmented Generation (RAG) has demonstrated considerable effectiveness in open-domain question answering. However, when applied to heterogeneous documents, comprising both textual and tabular components, existing RAG approaches exhibit critical limitations. The prevailing practice of flattening tables and chunking strategies disrupts the intrinsic tabular structure, leads to information loss, and undermines the reasoning capabilities of LLMs in multi-hop, global queries. To address these challenges, we propose TableRAG, an hybrid framework that unifies textual understanding and complex manipulations over tabular data. TableRAG iteratively operates in four steps: context-sensitive query decomposition, text retrieval, SQL programming and execution, and compositional intermediate answer generation. We also develop HeteQA, a novel benchmark designed to evaluate the multi-hop heterogeneous reasoning capabilities. Experimental results demonstrate that TableRAG consistently outperforms existing baselines on both public datasets and our HeteQA, establishing a new state-of-the-art for heterogeneous document question answering. We release TableRAG at this https URL. 

---
# CIIR@LiveRAG 2025: Optimizing Multi-Agent Retrieval Augmented Generation through Self-Training 

**Authors**: Alireza Salemi, Mukta Maddipatla, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2506.10844)  

**Abstract**: This paper presents mRAG, a multi-agent retrieval-augmented generation (RAG) framework composed of specialized agents for subtasks such as planning, searching, reasoning, and coordination. Our system uses a self-training paradigm with reward-guided trajectory sampling to optimize inter-agent collaboration and enhance response generation. Evaluated on DataMorgana-derived datasets during the SIGIR 2025 LiveRAG competition, mRAG outperforms conventional RAG baselines. We further analyze competition outcomes and showcase the framework's strengths with case studies, demonstrating its efficacy for complex, real-world RAG tasks. 

---
# Augmenting Large Language Models with Static Code Analysis for Automated Code Quality Improvements 

**Authors**: Seyed Moein Abtahi, Akramul Azim  

**Link**: [PDF](https://arxiv.org/pdf/2506.10330)  

**Abstract**: This study examined code issue detection and revision automation by integrating Large Language Models (LLMs) such as OpenAI's GPT-3.5 Turbo and GPT-4o into software development workflows. A static code analysis framework detects issues such as bugs, vulnerabilities, and code smells within a large-scale software project. Detailed information on each issue was extracted and organized to facilitate automated code revision using LLMs. An iterative prompt engineering process is applied to ensure that prompts are structured to produce accurate and organized outputs aligned with the project requirements. Retrieval-augmented generation (RAG) is implemented to enhance the relevance and precision of the revisions, enabling LLM to access and integrate real-time external knowledge. The issue of LLM hallucinations - where the model generates plausible but incorrect outputs - is addressed by a custom-built "Code Comparison App," which identifies and corrects erroneous changes before applying them to the codebase. Subsequent scans using the static code analysis framework revealed a significant reduction in code issues, demonstrating the effectiveness of combining LLMs, static analysis, and RAG to improve code quality, streamline the software development process, and reduce time and resource expenditure. 

---
# Safeguarding Multimodal Knowledge Copyright in the RAG-as-a-Service Environment 

**Authors**: Tianyu Chen, Jian Lou, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10030)  

**Abstract**: As Retrieval-Augmented Generation (RAG) evolves into service-oriented platforms (Rag-as-a-Service) with shared knowledge bases, protecting the copyright of contributed data becomes essential. Existing watermarking methods in RAG focus solely on textual knowledge, leaving image knowledge unprotected. In this work, we propose AQUA, the first watermark framework for image knowledge protection in Multimodal RAG systems. AQUA embeds semantic signals into synthetic images using two complementary methods: acronym-based triggers and spatial relationship cues. These techniques ensure watermark signals survive indirect watermark propagation from image retriever to textual generator, being efficient, effective and imperceptible. Experiments across diverse models and datasets show that AQUA enables robust, stealthy, and reliable copyright tracing, filling a key gap in multimodal RAG protection. 

---
