# Generative Representational Learning of Foundation Models for Recommendation 

**Authors**: Zheli Zhou, Chenxu Zhu, Jianghao Lin, Bo Chen, Ruiming Tang, Weinan Zhang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.11999)  

**Abstract**: Developing a single foundation model with the capability to excel across diverse tasks has been a long-standing objective in the field of artificial intelligence. As the wave of general-purpose foundation models sweeps across various domains, their influence has significantly extended to the field of recommendation systems. While recent efforts have explored recommendation foundation models for various generative tasks, they often overlook crucial embedding tasks and struggle with the complexities of multi-task learning, including knowledge sharing & conflict resolution, and convergence speed inconsistencies. To address these limitations, we introduce RecFound, a generative representational learning framework for recommendation foundation models. We construct the first comprehensive dataset for recommendation foundation models covering both generative and embedding tasks across diverse scenarios. Based on this dataset, we propose a novel multi-task training scheme featuring a Task-wise Mixture of Low-rank Experts (TMoLE) to handle knowledge sharing & conflict, a Step-wise Convergence-oriented Sample Scheduler (S2Sched) to address inconsistent convergence, and a Model Merge module to balance the performance across tasks. Experiments demonstrate that RecFound achieves state-of-the-art performance across various recommendation tasks, outperforming existing baselines. 

---
# Forgetful by Design? A Critical Audit of YouTube's Search API for Academic Research 

**Authors**: Bernhard Rieder, Adrian Padilla, Oscar Coromina  

**Link**: [PDF](https://arxiv.org/pdf/2506.11727)  

**Abstract**: This paper critically audits the search endpoint of YouTube's Data API (v3), a common tool for academic research. Through systematic weekly searches over six months using eleven queries, we identify major limitations regarding completeness, representativeness, consistency, and bias. Our findings reveal substantial differences between ranking parameters like relevance and date in terms of video recall and precision, with relevance often retrieving numerous off-topic videos. We also find severe temporal decay, as the number of findable videos for a specific period dramatically decreases after just 20-60 days from the publication date, potentially hampering many different research designs. Furthermore, search results lack consistency, with identical queries yielding different video sets over time, compromising replicability. A case study on the European Parliament elections highlights how these issues impact research outcomes. While the paper offers several mitigation strategies, it concludes that the API's search function, potentially prioritizing "freshness" over comprehensive retrieval, is not adequate for robust academic research, especially concerning Digital Services Act requirements. 

---
# TongSearch-QR: Reinforced Query Reasoning for Retrieval 

**Authors**: Xubo Qin, Jun Bai, Jiaqi Li, Zixia Jia, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11603)  

**Abstract**: Traditional information retrieval (IR) methods excel at textual and semantic matching but struggle in reasoning-intensive retrieval tasks that require multi-hop inference or complex semantic understanding between queries and documents. One promising solution is to explicitly rewrite or augment queries using large language models (LLMs) to elicit reasoning-relevant content prior to retrieval. However, the widespread use of large-scale language models like GPT-4 or LLaMA3-70B remains impractical due to their high inference cost and limited deployability in real-world systems. In this work, we introduce TongSearch QR (Previously Known as "TongSearch Reasoner"), a family of small-scale language models for query reasoning and rewriting in reasoning-intensive retrieval. With a novel semi-rule-based reward function, we employ reinforcement learning approaches enabling smaller language models, e,g, Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct, to achieve query reasoning performance rivaling large-scale language models without their prohibitive inference costs. Experiment results on BRIGHT benchmark show that with BM25 as retrievers, both TongSearch QR-7B and TongSearch QR-1.5B models significantly outperform existing baselines, including prompt-based query reasoners and some latest dense retrievers trained for reasoning-intensive retrieval tasks, offering superior adaptability for real-world deployment. 

---
# GraphRAG-Causal: A novel graph-augmented framework for causal reasoning and annotation in news 

**Authors**: Abdul Haque, Umm e Hani, Ahmad Din, Muhammad Babar, Ali Abbas, Insaf Ullah  

**Link**: [PDF](https://arxiv.org/pdf/2506.11600)  

**Abstract**: GraphRAG-Causal introduces an innovative framework that combines graph-based retrieval with large language models to enhance causal reasoning in news analysis. Traditional NLP approaches often struggle with identifying complex, implicit causal links, especially in low-data scenarios. Our approach addresses these challenges by transforming annotated news headlines into structured causal knowledge graphs. It then employs a hybrid retrieval system that merges semantic embeddings with graph-based structural cues leveraging Neo4j to accurately match and retrieve relevant events. The framework is built on a three-stage pipeline: First, during Data Preparation, news sentences are meticulously annotated and converted into causal graphs capturing cause, effect, and trigger relationships. Next, the Graph Retrieval stage stores these graphs along with their embeddings in a Neo4j database and utilizes hybrid Cypher queries to efficiently identify events that share both semantic and structural similarities with a given query. Finally, the LLM Inference stage utilizes these retrieved causal graphs in a few-shot learning setup with XML-based prompting, enabling robust classification and tagging of causal relationships. Experimental evaluations demonstrate that GraphRAG-Causal achieves an impressive F1-score of 82.1% on causal classification using just 20 few-shot examples. This approach significantly boosts accuracy and consistency, making it highly suitable for real-time applications in news reliability assessment, misinformation detection, and policy analysis. 

---
# Dual-View Disentangled Multi-Intent Learning for Enhanced Collaborative Filtering 

**Authors**: Shanfan Zhang, Yongyi Lin, Yuan Rao, Chenlong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11538)  

**Abstract**: Disentangling user intentions from implicit feedback has become a promising strategy to enhance recommendation accuracy and interpretability. Prior methods often model intentions independently and lack explicit supervision, thus failing to capture the joint semantics that drive user-item interactions. To address these limitations, we propose DMICF, a unified framework that explicitly models interaction-level intent alignment while leveraging structural signals from both user and item perspectives. DMICF adopts a dual-view architecture that jointly encodes user-item interaction graphs from both sides, enabling bidirectional information fusion. This design enhances robustness under data sparsity by allowing the structural redundancy of one view to compensate for the limitations of the other. To model fine-grained user-item compatibility, DMICF introduces an intent interaction encoder that performs sub-intent alignment within each view, uncovering shared semantic structures that underlie user decisions. This localized alignment enables adaptive refinement of intent embeddings based on interaction context, thus improving the model's generalization and expressiveness, particularly in long-tail scenarios. Furthermore, DMICF integrates an intent-aware scoring mechanism that aggregates compatibility signals from matched intent pairs across user and item subspaces, enabling personalized prediction grounded in semantic congruence rather than entangled representations. To facilitate semantic disentanglement, we design a discriminative training signal via multi-negative sampling and softmax normalization, which pulls together semantically aligned intent pairs while pushing apart irrelevant or noisy ones. Extensive experiments demonstrate that DMICF consistently delivers robust performance across datasets with diverse interaction distributions. 

---
# A Reference Model and Patterns for Production Event Data Enrichment 

**Authors**: Mark van der Pas, Remco Dijkman, Alp Akçay, Ivo Adan, John Walker  

**Link**: [PDF](https://arxiv.org/pdf/2506.11502)  

**Abstract**: With the advent of digital transformation, organisations are increasingly generating large volumes of data through the execution of various processes across disparate systems. By integrating data from these heterogeneous sources, it becomes possible to derive new insights essential for tasks such as monitoring and analysing process performance. Typically, this information is extracted during a data pre-processing or engineering phase. However, this step is often performed in an ad-hoc manner and is time-consuming and labour-intensive. To streamline this process, we introduce a reference model and a collection of patterns designed to enrich production event data. The reference model provides a standard way for storing and extracting production event data. The patterns describe common information extraction tasks and how such tasks can be automated effectively. The reference model is developed by combining the ISA-95 industry standard with the Event Knowledge Graph formalism. The patterns are developed based on empirical observations from event data sets originating in manufacturing processes and are formalised using the reference model. We evaluate the relevance and applicability of these patterns by demonstrating their application to use cases. 

---
# Leveraging Reference Documents for Zero-Shot Ranking via Large Language Models 

**Authors**: Jieran Li, Xiuyuan Hu, Yang Zhao, Shengyao Zhuang, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11452)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional performance in the task of text ranking for information retrieval. While Pointwise ranking approaches offer computational efficiency by scoring documents independently, they often yield biased relevance estimates due to the lack of inter-document comparisons. In contrast, Pairwise methods improve ranking accuracy by explicitly comparing document pairs, but suffer from substantial computational overhead with quadratic complexity ($O(n^2)$). To address this tradeoff, we propose \textbf{RefRank}, a simple and effective comparative ranking method based on a fixed reference document. Instead of comparing all document pairs, RefRank prompts the LLM to evaluate each candidate relative to a shared reference anchor. By selecting the reference anchor that encapsulates the core query intent, RefRank implicitly captures relevance cues, enabling indirect comparison between documents via this common anchor. This reduces computational cost to linear time ($O(n)$) while importantly, preserving the advantages of comparative evaluation. To further enhance robustness, we aggregate multiple RefRank outputs using a weighted averaging scheme across different reference choices. Experiments on several benchmark datasets and with various LLMs show that RefRank significantly outperforms Pointwise baselines and could achieve performance at least on par with Pairwise approaches with a significantly lower computational cost. 

---
# Deep Learning Model Acceleration and Optimization Strategies for Real-Time Recommendation Systems 

**Authors**: Junli Shao, Jing Dong, Dingzhou Wang, Kowei Shih, Dannier Li, Chengrui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.11421)  

**Abstract**: With the rapid growth of Internet services, recommendation systems play a central role in delivering personalized content. Faced with massive user requests and complex model architectures, the key challenge for real-time recommendation systems is how to reduce inference latency and increase system throughput without sacrificing recommendation quality. This paper addresses the high computational cost and resource bottlenecks of deep learning models in real-time settings by proposing a combined set of modeling- and system-level acceleration and optimization strategies. At the model level, we dramatically reduce parameter counts and compute requirements through lightweight network design, structured pruning, and weight quantization. At the system level, we integrate multiple heterogeneous compute platforms and high-performance inference libraries, and we design elastic inference scheduling and load-balancing mechanisms based on real-time load characteristics. Experiments show that, while maintaining the original recommendation accuracy, our methods cut latency to less than 30% of the baseline and more than double system throughput, offering a practical solution for deploying large-scale online recommendation services. 

---
# DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents 

**Authors**: Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2506.11763)  

**Abstract**: Deep Research Agents are a prominent category of LLM-based agents. By autonomously orchestrating multistep web exploration, targeted retrieval, and higher-order synthesis, they transform vast amounts of online information into analyst-grade, citation-rich reports--compressing hours of manual desk research into minutes. However, a comprehensive benchmark for systematically evaluating the capabilities of these agents remains absent. To bridge this gap, we present DeepResearch Bench, a benchmark consisting of 100 PhD-level research tasks, each meticulously crafted by domain experts across 22 distinct fields. Evaluating DRAs is inherently complex and labor-intensive. We therefore propose two novel methodologies that achieve strong alignment with human judgment. The first is a reference-based method with adaptive criteria to assess the quality of generated research reports. The other framework is introduced to evaluate DRA's information retrieval and collection capabilities by assessing its effective citation count and overall citation accuracy. We have open-sourced DeepResearch Bench and key components of these frameworks at this https URL to accelerate the development of practical LLM-based agents. 

---
# Digitization of Document and Information Extraction using OCR 

**Authors**: Rasha Sinha, Rekha B S  

**Link**: [PDF](https://arxiv.org/pdf/2506.11156)  

**Abstract**: Retrieving accurate details from documents is a crucial task, especially when handling a combination of scanned images and native digital formats. This document presents a combined framework for text extraction that merges Optical Character Recognition (OCR) techniques with Large Language Models (LLMs) to deliver structured outputs enriched by contextual understanding and confidence indicators. Scanned files are processed using OCR engines, while digital files are interpreted through layout-aware libraries. The extracted raw text is subsequently analyzed by an LLM to identify key-value pairs and resolve ambiguities. A comparative analysis of different OCR tools is presented to evaluate their effectiveness concerning accuracy, layout recognition, and processing speed. The approach demonstrates significant improvements over traditional rule-based and template-based methods, offering enhanced flexibility and semantic precision across different document categories 

---
# ScIRGen: Synthesize Realistic and Large-Scale RAG Dataset for Scientific Research 

**Authors**: Junyong Lin, Lu Dai, Ruiqian Han, Yijie Sui, Ruilin Wang, Xingliang Sun, Qinglin Wu, Min Feng, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.11117)  

**Abstract**: Scientific researchers need intensive information about datasets to effectively evaluate and develop theories and methodologies. The information needs regarding datasets are implicitly embedded in particular research tasks, rather than explicitly expressed in search queries. However, existing scientific retrieval and question-answering (QA) datasets typically address straightforward questions, which do not align with the distribution of real-world research inquiries. To bridge this gap, we developed ScIRGen, a dataset generation framework for scientific QA \& retrieval that more accurately reflects the information needs of professional science researchers, and uses it to create a large-scale scientific retrieval-augmented generation (RAG) dataset with realistic queries, datasets and papers. Technically, we designed a dataset-oriented information extraction method that leverages academic papers to augment the dataset representation. We then proposed a question generation framework by employing cognitive taxonomy to ensure the quality of synthesized questions. We also design a method to automatically filter synthetic answers based on the perplexity shift of LLMs, which is highly aligned with human judgment of answers' validity. Collectively, these methodologies culminated in the creation of the 61k QA dataset, ScIRGen-Geo. We benchmarked representative methods on the ScIRGen-Geo dataset for their question-answering and retrieval capabilities, finding out that current methods still suffer from reasoning from complex questions. This work advances the development of more sophisticated tools to support the intricate information needs of the scientific community. 

---
# Manifesto from Dagstuhl Perspectives Workshop 24352 -- Conversational Agents: A Framework for Evaluation (CAFE) 

**Authors**: Christine Bauer, Li Chen, Nicola Ferro, Norbert Fuhr, Avishek Anand, Timo Breuer, Guglielmo Faggioli, Ophir Frieder, Hideo Joho, Jussi Karlgren, Johannes Kiesel, Bart P. Knijnenburg, Aldo Lipani, Lien Michiels, Andrea Papenmeier, Maria Soledad Pera, Mark Sanderson, Scott Sanner, Benno Stein, Johanne R. Trippas, Karin Verspoor, Martijn C Willemsen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11112)  

**Abstract**: During the workshop, we deeply discussed what CONversational Information ACcess (CONIAC) is and its unique features, proposing a world model abstracting it, and defined the Conversational Agents Framework for Evaluation (CAFE) for the evaluation of CONIAC systems, consisting of six major components: 1) goals of the system's stakeholders, 2) user tasks to be studied in the evaluation, 3) aspects of the users carrying out the tasks, 4) evaluation criteria to be considered, 5) evaluation methodology to be applied, and 6) measures for the quantitative criteria chosen. 

---
# Graph-based RAG Enhancement via Global Query Disambiguation and Dependency-Aware Reranking 

**Authors**: Ningyuan Li, Junrui Liu, Yi Shan, Minghui Huang, Tong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.11106)  

**Abstract**: Contemporary graph-based retrieval-augmented generation (RAG) methods typically begin by extracting entities from user queries and then leverage pre-constructed knowledge graphs to retrieve related relationships and metadata. However, this pipeline's exclusive reliance on entity-level extraction can lead to the misinterpretation or omission of latent yet critical information and relations. As a result, retrieved content may be irrelevant or contradictory, and essential knowledge may be excluded, exacerbating hallucination risks and degrading the fidelity of generated responses. To address these limitations, we introduce PankRAG, a framework that combines a globally aware, hierarchical query-resolution strategy with a novel dependency-aware reranking mechanism. PankRAG first constructs a multi-level resolution path that captures both parallel and sequential interdependencies within a query, guiding large language models (LLMs) through structured reasoning. It then applies its dependency-aware reranker to exploit the dependency structure among resolved sub-questions, enriching and validating retrieval results for subsequent sub-questions. Empirical evaluations demonstrate that PankRAG consistently outperforms state-of-the-art approaches across multiple benchmarks, underscoring its robustness and generalizability. 

---
# C-SEO Bench: Does Conversational SEO Work? 

**Authors**: Haritz Puerto, Martin Gubri, Tommaso Green, Seong Joon Oh, Sangdoo Yun  

**Link**: [PDF](https://arxiv.org/pdf/2506.11097)  

**Abstract**: Large Language Models (LLMs) are transforming search engines into Conversational Search Engines (CSE). Consequently, Search Engine Optimization (SEO) is being shifted into Conversational Search Engine Optimization (C-SEO). We are beginning to see dedicated C-SEO methods for modifying web documents to increase their visibility in CSE responses. However, they are often tested only for a limited breadth of application domains; we do not understand whether certain C-SEO methods would be effective for a broad range of domains. Moreover, existing evaluations consider only a single-actor scenario where only one web document adopts a C-SEO method; in reality, multiple players are likely to competitively adopt the cutting-edge C-SEO techniques, drawing an analogy from the dynamics we have seen in SEO. We present C-SEO Bench, the first benchmark designed to evaluate C-SEO methods across multiple tasks, domains, and number of actors. We consider two search tasks, question answering and product recommendation, with three domains each. We also formalize a new evaluation protocol with varying adoption rates among involved actors. Our experiments reveal that most current C-SEO methods are largely ineffective, contrary to reported results in the literature. Instead, traditional SEO strategies, those aiming to improve the ranking of the source in the LLM context, are significantly more effective. We also observe that as we increase the number of C-SEO adopters, the overall gains decrease, depicting a congested and zero-sum nature of the problem. Our code and data are available at this https URL and this https URL. 

---
# LeanExplore: A search engine for Lean 4 declarations 

**Authors**: Justin Asher  

**Link**: [PDF](https://arxiv.org/pdf/2506.11085)  

**Abstract**: The expanding Lean 4 ecosystem poses challenges for navigating its vast libraries. This paper introduces LeanExplore, a search engine for Lean 4 declarations. LeanExplore enables users to semantically search for statements, both formally and informally, across select Lean 4 packages (including Batteries, Init, Lean, Mathlib, PhysLean, and Std). This search capability is powered by a hybrid ranking strategy, integrating scores from a multi-source semantic embedding model (capturing conceptual meaning from formal Lean code, docstrings, AI-generated informal translations, and declaration titles), BM25+ for keyword-based lexical relevance, and a PageRank-based score reflecting declaration importance and interconnectedness. The search engine is accessible via a dedicated website (this https URL) and a Python API (this https URL). Furthermore, the database can be downloaded, allowing users to self-host the service. LeanExplore integrates easily with LLMs via the model context protocol (MCP), enabling users to chat with an AI assistant about Lean declarations or utilize the search engine for building theorem-proving agents. This work details LeanExplore's architecture, data processing, functionalities, and its potential to enhance Lean 4 workflows and AI-driven mathematical research 

---
