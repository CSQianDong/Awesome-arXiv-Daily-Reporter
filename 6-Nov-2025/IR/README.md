# CLAX: Fast and Flexible Neural Click Models in JAX 

**Authors**: Philipp Hager, Onno Zoeter, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2511.03620)  

**Abstract**: CLAX is a JAX-based library that implements classic click models using modern gradient-based optimization. While neural click models have emerged over the past decade, complex click models based on probabilistic graphical models (PGMs) have not systematically adopted gradient-based optimization, preventing practitioners from leveraging modern deep learning frameworks while preserving the interpretability of classic models. CLAX addresses this gap by replacing EM-based optimization with direct gradient-based optimization in a numerically stable manner. The framework's modular design enables the integration of any component, from embeddings and deep networks to custom modules, into classic click models for end-to-end optimization. We demonstrate CLAX's efficiency by running experiments on the full Baidu-ULTR dataset comprising over a billion user sessions in $\approx$ 2 hours on a single GPU, orders of magnitude faster than traditional EM approaches. CLAX implements ten classic click models, serving both industry practitioners seeking to understand user behavior and improve ranking performance at scale and researchers developing new click models. CLAX is available at: this https URL 

---
# A Semantic Encoding of Object Centric Event Data 

**Authors**: Saba Latif, Fajar J. Ekaputra, Maxim Vidgof, Sabrina Kirrane, Claudio Di Ciccio  

**Link**: [PDF](https://arxiv.org/pdf/2511.03351)  

**Abstract**: The Object-Centric Event Data (OCED) is a novel meta-model aimed at providing a common ground for process data records centered around events and objects. One of its objectives is to foster interoperability and process information exchange. In this context, the integration of data from different providers, the combination of multiple processes, and the enhancement of knowledge inference are novel challenges. Semantic Web technologies can enable the creation of a machine-readable OCED description enriched through ontology-based relationships and entity categorization. In this paper, we introduce an approach built upon Semantic Web technologies for the realization of semantic-enhanced OCED, with the aim to strengthen process data reasoning, interconnect information sources, and boost expressiveness. 

---
# Discourse-Aware Scientific Paper Recommendation via QA-Style Summarization and Multi-Level Contrastive Learning 

**Authors**: Shenghua Wang, Zhen Yin  

**Link**: [PDF](https://arxiv.org/pdf/2511.03330)  

**Abstract**: The rapid growth of open-access (OA) publications has intensified the challenge of identifying relevant scientific papers. Due to privacy constraints and limited access to user interaction data, recent efforts have shifted toward content-based recommendation, which relies solely on textual information. However, existing models typically treat papers as unstructured text, neglecting their discourse organization and thereby limiting semantic completeness and interpretability. To address these limitations, we propose OMRC-MR, a hierarchical framework that integrates QA-style OMRC (Objective, Method, Result, Conclusion) summarization, multi-level contrastive learning, and structure-aware re-ranking for scholarly recommendation. The QA-style summarization module converts raw papers into structured and discourse-consistent representations, while multi-level contrastive objectives align semantic representations across metadata, section, and document levels. The final re-ranking stage further refines retrieval precision through contextual similarity calibration. Experiments on DBLP, S2ORC, and the newly constructed Sci-OMRC dataset demonstrate that OMRC-MR consistently surpasses state-of-the-art baselines, achieving up to 7.2% and 3.8% improvements in Precision@10 and Recall@10, respectively. Additional evaluations confirm that QA-style summarization produces more coherent and factually complete representations. Overall, OMRC-MR provides a unified and interpretable content-based paradigm for scientific paper recommendation, advancing trustworthy and privacy-aware scholarly information retrieval. 

---
# KScaNN: Scalable Approximate Nearest Neighbor Search on Kunpeng 

**Authors**: Oleg Senkevich, Siyang Xu, Tianyi Jiang, Alexander Radionov, Jan Tabaszewski, Dmitriy Malyshev, Zijian Li, Daihao Xue, Licheng Yu, Weidi Zeng, Meiling Wang, Xin Yao, Siyu Huang, Gleb Neshchetkin, Qiuling Pan, Yaoyao Fu  

**Link**: [PDF](https://arxiv.org/pdf/2511.03298)  

**Abstract**: Approximate Nearest Neighbor Search (ANNS) is a cornerstone algorithm for information retrieval, recommendation systems, and machine learning applications. While x86-based architectures have historically dominated this domain, the increasing adoption of ARM-based servers in industry presents a critical need for ANNS solutions optimized on ARM architectures. A naive port of existing x86 ANNS algorithms to ARM platforms results in a substantial performance deficit, failing to leverage the unique capabilities of the underlying hardware. To address this challenge, we introduce KScaNN, a novel ANNS algorithm co-designed for the Kunpeng 920 ARM architecture. KScaNN embodies a holistic approach that synergizes sophisticated, data aware algorithmic refinements with carefully-designed hardware specific optimizations. Its core contributions include: 1) novel algorithmic techniques, including a hybrid intra-cluster search strategy and an improved PQ residual calculation method, which optimize the search process at a higher level; 2) an ML-driven adaptive search module that provides adaptive, per-query tuning of search parameters, eliminating the inefficiencies of static configurations; and 3) highly-optimized SIMD kernels for ARM that maximize hardware utilization for the critical distance computation workloads. The experimental results demonstrate that KScaNN not only closes the performance gap but establishes a new standard, achieving up to a 1.63x speedup over the fastest x86-based solution. This work provides a definitive blueprint for achieving leadership-class performance for vector search on modern ARM architectures and underscores 

---
# Generative Sequential Recommendation via Hierarchical Behavior Modeling 

**Authors**: Zhefan Wang, Guokai Yan, Jinbei Yu, Siyu Gu, Jingyan Chen, Peng Jiang, Zhiqiang Guo, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.03155)  

**Abstract**: Recommender systems in multi-behavior domains, such as advertising and e-commerce, aim to guide users toward high-value but inherently sparse conversions. Leveraging auxiliary behaviors (e.g., clicks, likes, shares) is therefore essential. Recent progress on generative recommendations has brought new possibilities for multi-behavior sequential recommendation. However, existing generative approaches face two significant challenges: 1) Inadequate Sequence Modeling: capture the complex, cross-level dependencies within user behavior sequences, and 2) Lack of Suitable Datasets: publicly available multi-behavior recommendation datasets are almost exclusively derived from e-commerce platforms, limiting the validation of feasibility in other domains, while also lacking sufficient side information for semantic ID generation. To address these issues, we propose a novel generative framework, GAMER (Generative Augmentation and Multi-lEvel behavior modeling for Recommendation), built upon a decoder-only backbone. GAMER introduces a cross-level interaction layer to capture hierarchical dependencies among behaviors and a sequential augmentation strategy that enhances robustness in training. To further advance this direction, we collect and release ShortVideoAD, a large-scale multi-behavior dataset from a mainstream short-video platform, which differs fundamentally from existing e-commerce datasets and provides pretrained semantic IDs for research on generative methods. Extensive experiments show that GAMER consistently outperforms both discriminative and generative baselines across multiple metrics. 

---
# Two thousand years of the oracle problem. Insights from Ancient Delphi on the future of blockchain oracles 

**Authors**: Giulio Caldarelli, Massimiliano Ornaghi  

**Link**: [PDF](https://arxiv.org/pdf/2511.03319)  

**Abstract**: The oracle problem refers to the inability of an agent to know if the information coming from an oracle is authentic and unbiased. In ancient times, philosophers and historians debated on how to evaluate, increase, and secure the reliability of oracle predictions, particularly those from Delphi, which pertained to matters of state. Today, we refer to data carriers for automatic machines as oracles, but establishing a secure channel between these oracles and the real world still represents a challenge. Despite numerous efforts, this problem remains mostly unsolved, and the recent advent of blockchain oracles has added a layer of complexity because of the decentralization of blockchains. This paper conceptually connects Delphic and modern blockchain oracles, developing a comparative framework. Leveraging blockchain oracle taxonomy, lexical analysis is also performed on 167 Delphic queries to shed light on the relationship between oracle answer quality and question type. The presented framework aims first at revealing commonalities between classical and computational oracles and then at enriching the oracle analysis within each field. This study contributes to the computer science literature by proposing strategies to improve the reliability of blockchain oracles based on insights from Delphi and to classical literature by introducing a framework that can also be applied to interpret and classify other ancient oracular mechanisms. 

---
# Beyond Ranked Lists: The SARAL Framework for Cross-Lingual Document Set Retrieval 

**Authors**: Shantanu Agarwal, Joel Barry, Elizabeth Boschee, Scott Miller  

**Link**: [PDF](https://arxiv.org/pdf/2511.03228)  

**Abstract**: Machine Translation for English Retrieval of Information in Any Language (MATERIAL) is an IARPA initiative targeted to advance the state of cross-lingual information retrieval (CLIR). This report provides a detailed description of Information Sciences Institute's (ISI's) Summarization and domain-Adaptive Retrieval Across Language's (SARAL's) effort for MATERIAL. Specifically, we outline our team's novel approach to handle CLIR with emphasis in developing an approach amenable to retrieve a query-relevant document \textit{set}, and not just a ranked document-list. In MATERIAL's Phase-3 evaluations, SARAL exceeded the performance of other teams in five out of six evaluation conditions spanning three different languages (Farsi, Kazakh, and Georgian). 

---
# Hybrid Fact-Checking that Integrates Knowledge Graphs, Large Language Models, and Search-Based Retrieval Agents Improves Interpretable Claim Verification 

**Authors**: Shaghayegh Kolli, Richard Rosenbaum, Timo Cavelius, Lasse Strothe, Andrii Lata, Jana Diesner  

**Link**: [PDF](https://arxiv.org/pdf/2511.03217)  

**Abstract**: Large language models (LLMs) excel in generating fluent utterances but can lack reliable grounding in verified information. At the same time, knowledge-graph-based fact-checkers deliver precise and interpretable evidence, yet suffer from limited coverage or latency. By integrating LLMs with knowledge graphs and real-time search agents, we introduce a hybrid fact-checking approach that leverages the individual strengths of each component. Our system comprises three autonomous steps: 1) a Knowledge Graph (KG) Retrieval for rapid one - hop lookups in DBpedia, 2) an LM-based classification guided by a task-specific labeling prompt, producing outputs with internal rule-based logic, and 3) a Web Search Agent invoked only when KG coverage is insufficient. Our pipeline achieves an F1 score of 0.93 on the FEVER benchmark on the Supported/Refuted split without task- specific fine - tuning. To address Not enough information cases, we conduct a targeted reannotation study showing that our approach frequently uncovers valid evidence for claims originally labeled as Not Enough Information (NEI), as confirmed by both expert annotators and LLM reviewers. With this paper, we present a modular, opensource fact-checking pipeline with fallback strategies and generalization across datasets. 

---
# Russian Contribution to Coronary Artery Disease Research: A Scientometric Mapping of Publications 

**Authors**: Muneer Ahmad, M Sadik Batcha  

**Link**: [PDF](https://arxiv.org/pdf/2511.03215)  

**Abstract**: The present study attempts to highlight the research output generated in Russia in coronary artery disease (CAD) research during the period 1990-2019 to understand the distribution of research output, top journals for publications, and most prolific authors, authorship pattern, and citation pattern. This study is based on secondary data extracted from the Science Citation Index (SCI), which is an integral component of the Web of Science. Descriptive and inferential statistical techniques were applied in the study. There were 5058 articles by Russian scholars in coronary artery disease during 1990-2019; they preferred to publish in Russian journals. The research contributions were in the form of research articles, meeting abstracts and reviews with a consistent drop in the number of editorial material and article; proceedings paper with time. Co-authorship was the norm in coronary artery disease research, with a steady increase in the number of multi-author documents in recent years. 

---
# A Study on Library Resources with Services Satisfaction based on Library Users Affiliated Colleges to Solapur University 

**Authors**: Patel Adam Burhansab, M Sadik Batcha, Muneer Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2511.03209)  

**Abstract**: The main aim of this study was to assess and evaluate user satisfaction with library resources and services among library users associated with Solapur University. The current research shows the level of users satisfaction with different library resources and services offered by college libraries. The research found that a vast number of respondents were pleased with library facilities and services. The research is designed to achieve users satisfaction in the library to investigate the level of satisfaction towards library resources and services with regards to 26 colleges of Solapur University based in Maharashtra. Information in the form of data has been collected from colleges and on the basis of users results; analysis needs to analyze users satisfaction. 

---
# No-Human in the Loop: Agentic Evaluation at Scale for Recommendation 

**Authors**: Tao Zhang, Kehui Yao, Luyi Ma, Jiao Chen, Reza Yousefi Maragheh, Kai Zhao, Jianpeng Xu, Evren Korpeoglu, Sushant Kumar, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2511.03051)  

**Abstract**: Evaluating large language models (LLMs) as judges is increasingly critical for building scalable and trustworthy evaluation pipelines. We present ScalingEval, a large-scale benchmarking study that systematically compares 36 LLMs, including GPT, Gemini, Claude, and Llama, across multiple product categories using a consensus-driven evaluation protocol. Our multi-agent framework aggregates pattern audits and issue codes into ground-truth labels via scalable majority voting, enabling reproducible comparison of LLM evaluators without human annotation. Applied to large-scale complementary-item recommendation, the benchmark reports four key findings: (i) Anthropic Claude 3.5 Sonnet achieves the highest decision confidence; (ii) Gemini 1.5 Pro offers the best overall performance across categories; (iii) GPT-4o provides the most favorable latency-accuracy-cost tradeoff; and (iv) GPT-OSS 20B leads among open-source models. Category-level analysis shows strong consensus in structured domains (Electronics, Sports) but persistent disagreement in lifestyle categories (Clothing, Food). These results establish ScalingEval as a reproducible benchmark and evaluation protocol for LLMs as judges, with actionable guidance on scaling, reliability, and model family tradeoffs. 

---
