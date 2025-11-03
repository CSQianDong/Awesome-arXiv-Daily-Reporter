# Interact-RAG: Reason and Interact with the Corpus, Beyond Black-Box Retrieval 

**Authors**: Yulong Hui, Chao Chen, Zhihang Fu, Yihao Liu, Jieping Ye, Huanchen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27566)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced LLMs by incorporating external information. However, prevailing agentic RAG approaches are constrained by a critical limitation: they treat the retrieval process as a black-box querying operation. This confines agents' actions to query issuing, hindering its ability to tackle complex information-seeking tasks. To address this, we introduce Interact-RAG, a new paradigm that elevates the LLM agent from a passive query issuer into an active manipulator of the retrieval process. We dismantle the black-box with a Corpus Interaction Engine, equipping the agent with a set of action primitives for fine-grained control over information retrieval. To further empower the agent on the entire RAG pipeline, we first develop a reasoning-enhanced workflow, which enables both zero-shot execution and the synthesis of interaction trajectories. We then leverage this synthetic data to train a fully autonomous end-to-end agent via Supervised Fine-Tuning (SFT), followed by refinement with Reinforcement Learning (RL). Extensive experiments across six benchmarks demonstrate that Interact-RAG significantly outperforms other advanced methods, validating the efficacy of our reasoning-interaction strategy. 

---
# Pairwise and Attribute-Aware Decision Tree-Based Preference Elicitation for Cold-Start Recommendation 

**Authors**: Alireza Gharahighehi, Felipe Kenji Nakano, Xuehua Yang, Wenhan Cu, Celine Vens  

**Link**: [PDF](https://arxiv.org/pdf/2510.27342)  

**Abstract**: Recommender systems (RSs) are intelligent filtering methods that suggest items to users based on their inferred preferences, derived from their interaction history on the platform. Collaborative filtering-based RSs rely on users past interactions to generate recommendations. However, when a user is new to the platform, referred to as a cold-start user, there is no historical data available, making it difficult to provide personalized recommendations. To address this, rating elicitation techniques can be used to gather initial ratings or preferences on selected items, helping to build an early understanding of the user's tastes. Rating elicitation approaches are generally categorized into two types: non-personalized and personalized. Decision tree-based rating elicitation is a personalized method that queries users about their preferences at each node of the tree until sufficient information is gathered. In this paper, we propose an extension to the decision tree approach for rating elicitation in the context of music recommendation. Our method: (i) elicits not only item ratings but also preferences on attributes such as genres to better cluster users, and (ii) uses item pairs instead of single items at each node to more effectively learn user preferences. Experimental results demonstrate that both proposed enhancements lead to improved performance, particularly with a reduced number of queries. 

---
# Traceable Drug Recommendation over Medical Knowledge Graphs 

**Authors**: Yu Lin, Zhen Jia, Philipp Christmann, Xu Zhang, Shengdong Du, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.27274)  

**Abstract**: Drug recommendation (DR) systems aim to support healthcare professionals in selecting appropriate medications based on patients' medical conditions. State-of-the-art approaches utilize deep learning techniques for improving DR, but fall short in providing any insights on the derivation process of recommendations -- a critical limitation in such high-stake applications. We propose TraceDR, a novel DR system operating over a medical knowledge graph (MKG), which ensures access to large-scale and high-quality information. TraceDR simultaneously predicts drug recommendations and related evidence within a multi-task learning framework, enabling traceability of medication recommendations. For covering a more diverse set of diseases and drugs than existing works, we devise a framework for automatically constructing patient health records and release DrugRec, a new large-scale testbed for DR. 

---
# A Survey on Deep Text Hashing: Efficient Semantic Text Retrieval with Binary Representation 

**Authors**: Liyang He, Zhenya Huang, Cheng Yang, Rui Li, Zheng Zhang, Kai Zhang, Zhi Li, Qi Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.27232)  

**Abstract**: With the rapid growth of textual content on the Internet, efficient large-scale semantic text retrieval has garnered increasing attention from both academia and industry. Text hashing, which projects original texts into compact binary hash codes, is a crucial method for this task. By using binary codes, the semantic similarity computation for text pairs is significantly accelerated via fast Hamming distance calculations, and storage costs are greatly reduced. With the advancement of deep learning, deep text hashing has demonstrated significant advantages over traditional, data-independent hashing techniques. By leveraging deep neural networks, these methods can learn compact and semantically rich binary representations directly from data, overcoming the performance limitations of earlier approaches. This survey investigates current deep text hashing methods by categorizing them based on their core components: semantic extraction, hash code quality preservation, and other key technologies. We then present a detailed evaluation schema with results on several popular datasets, followed by a discussion of practical applications and open-source tools for implementation. Finally, we conclude by discussing key challenges and future research directions, including the integration of deep text hashing with large language models to further advance the field. The project for this survey can be accessed at this https URL. 

---
# A Survey on Generative Recommendation: Data, Model, and Tasks 

**Authors**: Min Hou, Le Wu, Yuxin Liao, Yonghui Yang, Zhen Zhang, Changlong Zheng, Han Wu, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2510.27157)  

**Abstract**: Recommender systems serve as foundational infrastructure in modern information ecosystems, helping users navigate digital content and discover items aligned with their preferences. At their core, recommender systems address a fundamental problem: matching users with items. Over the past decades, the field has experienced successive paradigm shifts, from collaborative filtering and matrix factorization in the machine learning era to neural architectures in the deep learning era. Recently, the emergence of generative models, especially large language models (LLMs) and diffusion models, have sparked a new paradigm: generative recommendation, which reconceptualizes recommendation as a generation task rather than discriminative scoring. This survey provides a comprehensive examination through a unified tripartite framework spanning data, model, and task dimensions. Rather than simply categorizing works, we systematically decompose approaches into operational stages-data augmentation and unification, model alignment and training, task formulation and execution. At the data level, generative models enable knowledge-infused augmentation and agent-based simulation while unifying heterogeneous signals. At the model level, we taxonomize LLM-based methods, large recommendation models, and diffusion approaches, analyzing their alignment mechanisms and innovations. At the task level, we illuminate new capabilities including conversational interaction, explainable reasoning, and personalized content generation. We identify five key advantages: world knowledge integration, natural language understanding, reasoning capabilities, scaling laws, and creative generation. We critically examine challenges in benchmark design, model robustness, and deployment efficiency, while charting a roadmap toward intelligent recommendation assistants that fundamentally reshape human-information interaction. 

---
# Evaluating Perspectival Biases in Cross-Modal Retrieval 

**Authors**: Teerapol Saengsukhiran, Peerawat Chomphooyod, Narabodee Rodjananant, Chompakorn Chaksangchaichot, Patawee Prakrankamanant, Witthawin Sripheanpol, Pak Lovichit, SarChaksaana Nutanong, Ekapol Chuangsuwanich  

**Link**: [PDF](https://arxiv.org/pdf/2510.26861)  

**Abstract**: Multimodal retrieval systems are expected to operate in a semantic space, agnostic to the language or cultural origin of the query. In practice, however, retrieval outcomes systematically reflect perspectival biases: deviations shaped by linguistic prevalence and cultural associations. We study two such biases. First, prevalence bias refers to the tendency to favor entries from prevalent languages over semantically faithful entries in image-to-text retrieval. Second, association bias refers to the tendency to favor images culturally associated with the query over semantically correct ones in text-to-image retrieval. Results show that explicit alignment is a more effective strategy for mitigating prevalence bias. However, association bias remains a distinct and more challenging problem. These findings suggest that achieving truly equitable multimodal systems requires targeted strategies beyond simple data scaling and that bias arising from cultural association may be treated as a more challenging problem than one arising from linguistic prevalence. 

---
# Image Hashing via Cross-View Code Alignment in the Age of Foundation Models 

**Authors**: Ilyass Moummad, Kawtar Zaher, Hervé Goëau, Alexis Joly  

**Link**: [PDF](https://arxiv.org/pdf/2510.27584)  

**Abstract**: Efficient large-scale retrieval requires representations that are both compact and discriminative. Foundation models provide powerful visual and multimodal embeddings, but nearest neighbor search in these high-dimensional spaces is computationally expensive. Hashing offers an efficient alternative by enabling fast Hamming distance search with binary codes, yet existing approaches often rely on complex pipelines, multi-term objectives, designs specialized for a single learning paradigm, and long training times. We introduce CroVCA (Cross-View Code Alignment), a simple and unified principle for learning binary codes that remain consistent across semantically aligned views. A single binary cross-entropy loss enforces alignment, while coding-rate maximization serves as an anti-collapse regularizer to promote balanced and diverse codes. To implement this, we design HashCoder, a lightweight MLP hashing network with a final batch normalization layer to enforce balanced codes. HashCoder can be used as a probing head on frozen embeddings or to adapt encoders efficiently via LoRA fine-tuning. Across benchmarks, CroVCA achieves state-of-the-art results in just 5 training epochs. At 16 bits, it particularly well-for instance, unsupervised hashing on COCO completes in under 2 minutes and supervised hashing on ImageNet100 in about 3 minutes on a single GPU. These results highlight CroVCA's efficiency, adaptability, and broad applicability. 

---
# Towards Universal Video Retrieval: Generalizing Video Embedding via Synthesized Multimodal Pyramid Curriculum 

**Authors**: Zhuoning Guo, Mingxin Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27571)  

**Abstract**: The prevailing video retrieval paradigm is structurally misaligned, as narrow benchmarks incentivize correspondingly limited data and single-task training. Therefore, universal capability is suppressed due to the absence of a diagnostic evaluation that defines and demands multi-dimensional generalization. To break this cycle, we introduce a framework built on the co-design of evaluation, data, and modeling. First, we establish the Universal Video Retrieval Benchmark (UVRB), a suite of 16 datasets designed not only to measure performance but also to diagnose critical capability gaps across tasks and domains. Second, guided by UVRB's diagnostics, we introduce a scalable synthesis workflow that generates 1.55 million high-quality pairs to populate the semantic space required for universality. Finally, we devise the Modality Pyramid, a curriculum that trains our General Video Embedder (GVE) by explicitly leveraging the latent interconnections within our diverse data. Extensive experiments show GVE achieves state-of-the-art zero-shot generalization on UVRB. In particular, our analysis reveals that popular benchmarks are poor predictors of general ability and that partially relevant retrieval is a dominant but overlooked scenario. Overall, our co-designed framework provides a practical path to escape the limited scope and advance toward truly universal video retrieval. 

---
# Research Output of Webology Journal (2013-2017): A Scientometric Analysis 

**Authors**: Muneer Ahmad, M. Sadik Batcha, Basharat Ahmad Wani, Mohammad Idrees Khan, S. Roselin Jahina  

**Link**: [PDF](https://arxiv.org/pdf/2510.27259)  

**Abstract**: Webology is an international peer-reviewed journal in English devoted to the field of the World Wide Web and serves as a forum for discussion and experimentation. It serves as a forum for new research in information dissemination and communication processes in general, and in the context of the World Wide Web in particular. This paper presents a Scientometric analysis of the Webology Journal. The paper analyses the pattern of growth of the research output published in the journal, pattern of authorship, author productivity, and subjects covered to the papers over the period (2013-2017). It is found that 62 papers were published during the period of study (2013-2017). The maximum numbers of articles were collaborative in nature. The subject concentration of the journal noted was Social Networking/Web 2.0/Library 2.0 and Scientometrics or Bibliometrics. Iranian researchers contributed the maximum number of articles (37.10%). The study applied standard formula and statistical tools to bring out the factual result. 

---
# Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs 

**Authors**: Mohammad Tavakoli, Alireza Salemi, Carrie Ye, Mohamed Abdalla, Hamed Zamani, J Ross Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2510.27246)  

**Abstract**: Evaluating the abilities of large language models (LLMs) for tasks that require long-term memory and thus long-context reasoning, for example in conversational settings, is hampered by the existing benchmarks, which often lack narrative coherence, cover narrow domains, and only test simple recall-oriented tasks. This paper introduces a comprehensive solution to these challenges. First, we present a novel framework for automatically generating long (up to 10M tokens), coherent, and topically diverse conversations, accompanied by probing questions targeting a wide range of memory abilities. From this, we construct BEAM, a new benchmark comprising 100 conversations and 2,000 validated questions. Second, to enhance model performance, we propose LIGHT-a framework inspired by human cognition that equips LLMs with three complementary memory systems: a long-term episodic memory, a short-term working memory, and a scratchpad for accumulating salient facts. Our experiments on BEAM reveal that even LLMs with 1M token context windows (with and without retrieval-augmentation) struggle as dialogues lengthen. In contrast, LIGHT consistently improves performance across various models, achieving an average improvement of 3.5%-12.69% over the strongest baselines, depending on the backbone LLM. An ablation study further confirms the contribution of each memory component. 

---
# DRAMA: Unifying Data Retrieval and Analysis for Open-Domain Analytic Queries 

**Authors**: Chuxuan Hu, Maxwell Yang, James Weiland, Yeji Lim, Suhas Palawala, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27238)  

**Abstract**: Manually conducting real-world data analyses is labor-intensive and inefficient. Despite numerous attempts to automate data science workflows, none of the existing paradigms or systems fully demonstrate all three key capabilities required to support them effectively: (1) open-domain data collection, (2) structured data transformation, and (3) analytic reasoning.
To overcome these limitations, we propose DRAMA, an end-to-end paradigm that answers users' analytic queries in natural language on large-scale open-domain data. DRAMA unifies data collection, transformation, and analysis as a single pipeline. To quantitatively evaluate system performance on tasks representative of DRAMA, we construct a benchmark, DRAMA-Bench, consisting of two categories of tasks: claim verification and question answering, each comprising 100 instances. These tasks are derived from real-world applications that have gained significant public attention and require the retrieval and analysis of open-domain data. We develop DRAMA-Bot, a multi-agent system designed following DRAMA. It comprises a data retriever that collects and transforms data by coordinating the execution of sub-agents, and a data analyzer that performs structured reasoning over the retrieved data. We evaluate DRAMA-Bot on DRAMA-Bench together with five state-of-the-art baseline agents. DRAMA-Bot achieves 86.5% task accuracy at a cost of $0.05, outperforming all baselines with up to 6.9 times the accuracy and less than 1/6 of the cost. DRAMA is publicly available at this https URL. 

---
# Compass: General Filtered Search across Vector and Structured Data 

**Authors**: Chunxiao Ye, Xiao Yan, Eric Lo  

**Link**: [PDF](https://arxiv.org/pdf/2510.27141)  

**Abstract**: The increasing prevalence of hybrid vector and relational data necessitates efficient, general support for queries that combine high-dimensional vector search with complex relational filtering. However, existing filtered search solutions are fundamentally limited by specialized indices, which restrict arbitrary filtering and hinder integration with general-purpose DBMSs. This work introduces \textsc{Compass}, a unified framework that enables general filtered search across vector and structured data without relying on new index designs. Compass leverages established index structures -- such as HNSW and IVF for vector attributes, and B+-trees for relational attributes -- implementing a principled cooperative query execution strategy that coordinates candidate generation and predicate evaluation across modalities. Uniquely, Compass maintains generality by allowing arbitrary conjunctions, disjunctions, and range predicates, while ensuring robustness even with highly-selective or multi-attribute filters. Comprehensive empirical evaluations demonstrate that Compass consistently outperforms NaviX, the only existing performant general framework, across diverse hybrid query workloads. It also matches the query throughput of specialized single-attribute indices in their favorite settings with only a single attribute involved, all while maintaining full generality and DBMS compatibility. Overall, Compass offers a practical and robust solution for achieving truly general filtered search in vector database systems. 

---
# LeMat-Synth: a multi-modal toolbox to curate broad synthesis procedure databases from scientific literature 

**Authors**: Magdalena Lederbauer, Siddharth Betala, Xiyao Li, Ayush Jain, Amine Sehaba, Georgia Channing, Grégoire Germain, Anamaria Leonescu, Faris Flaifil, Alfonso Amayuelas, Alexandre Nozadze, Stefan P. Schmid, Mohd Zaki, Sudheesh Kumar Ethirajan, Elton Pan, Mathilde Franckel, Alexandre Duval, N. M. Anoop Krishnan, Samuel P. Gleason  

**Link**: [PDF](https://arxiv.org/pdf/2510.26824)  

**Abstract**: The development of synthesis procedures remains a fundamental challenge in materials discovery, with procedural knowledge scattered across decades of scientific literature in unstructured formats that are challenging for systematic analysis. In this paper, we propose a multi-modal toolbox that employs large language models (LLMs) and vision language models (VLMs) to automatically extract and organize synthesis procedures and performance data from materials science publications, covering text and figures. We curated 81k open-access papers, yielding LeMat-Synth (v 1.0): a dataset containing synthesis procedures spanning 35 synthesis methods and 16 material classes, structured according to an ontology specific to materials science. The extraction quality is rigorously evaluated on a subset of 2.5k synthesis procedures through a combination of expert annotations and a scalable LLM-as-a-judge framework. Beyond the dataset, we release a modular, open-source software library designed to support community-driven extension to new corpora and synthesis domains. Altogether, this work provides an extensible infrastructure to transform unstructured literature into machine-readable information. This lays the groundwork for predictive modeling of synthesis procedures as well as modeling synthesis--structure--property relationships. 

---
# DGAI: Decoupled On-Disk Graph-Based ANN Index for Efficient Updates and Queries 

**Authors**: Jiahao Lou, Quan Yu, Shufeng Gong, Song Yu, Yanfeng Zhang, Ge Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.25401)  

**Abstract**: On-disk graph-based indexes are widely used in approximate nearest neighbor (ANN) search systems for large-scale, high-dimensional vectors. However, traditional coupled storage methods, which store vectors within the index, are inefficient for index updates. Coupled storage incurs excessive redundant vector reads and writes when updating the graph topology, leading to significant invalid I/O. To address this issue, we propose a decoupled storage architecture. While a decoupled architecture reduces query performance. To overcome this limitation, we design two tailored strategies: (i) a three-stage query mechanism that leverages multiple PQ compressed vectors to filter invalid I/O and computations, and (ii) an incremental page-level topological reordering strategy that incrementally inserts new nodes into pages containing their most similar neighbors to mitigate read amplification. Together, these techniques substantially reduce both I/O and computational overhead during ANN search. Experimental results show that the decoupled architecture improves update speed by 10.05x for insertions and 6.89x for deletions, while the three-stage query and incremental reordering enhance query efficiency by 2.66x compared to the traditional coupled architecture. 

---
