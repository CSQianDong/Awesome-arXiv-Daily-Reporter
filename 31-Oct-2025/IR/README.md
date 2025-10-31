# ProfOlaf: Semi-Automated Tool for Systematic Literature Reviews 

**Authors**: Martim Afonso, Nuno Saavedra, Bruno Lourenço, Alexandra Mendes, João Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2510.26750)  

**Abstract**: Systematic reviews and mapping studies are critical for synthesizing research, identifying gaps, and guiding future work, but they are often labor-intensive and time-consuming. Existing tools provide partial support for specific steps, leaving much of the process manual and error-prone. We present ProfOlaf, a semi-automated tool designed to streamline systematic reviews while maintaining methodological rigor. ProfOlaf supports iterative snowballing for article collection with human-in-the-loop filtering and uses large language models to assist in analyzing articles, extracting key topics, and answering queries about the content of papers. By combining automation with guided manual effort, ProfOlaf enhances the efficiency, quality, and reproducibility of systematic reviews across research fields. A video describing and demonstrating ProfOlaf is available at: this https URL 

---
# WeaveRec: An LLM-Based Cross-Domain Sequential Recommendation Framework with Model Merging 

**Authors**: Min Hou, Xin Liu, Le Wu, Chenyi He, Hao Liu, Zhi Li, Xin Li, Si Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.26546)  

**Abstract**: Cross-Domain Sequential Recommendation (CDSR) seeks to improve user preference modeling by transferring knowledge from multiple domains. Despite the progress made in CDSR, most existing methods rely on overlapping users or items to establish cross-domain correlations-a requirement that rarely holds in real-world settings. The advent of large language models (LLM) and model-merging techniques appears to overcome this limitation by unifying multi-domain data without explicit overlaps. Yet, our empirical study shows that naively training an LLM on combined domains-or simply merging several domain-specific LLMs-often degrades performance relative to a model trained solely on the target domain. To address these challenges, we first experimentally investigate the cause of suboptimal performance in LLM-based cross-domain recommendation and model merging. Building on these insights, we introduce WeaveRec, which cross-trains multiple LoRA modules with source and target domain data in a weaving fashion, and fuses them via model merging. WeaveRec can be extended to multi-source domain scenarios and notably does not introduce additional inference-time cost in terms of latency or memory. Furthermore, we provide a theoretical guarantee that WeaveRec can reduce the upper bound of the expected error in the target domain. Extensive experiments on single-source, multi-source, and cross-platform cross-domain recommendation scenarios validate that WeaveRec effectively mitigates performance degradation and consistently outperforms baseline approaches in real-world recommendation tasks. 

---
# Vectorized Context-Aware Embeddings for GAT-Based Collaborative Filtering 

**Authors**: Danial Ebrat, Sepideh Ahmadian, Luis Rueda  

**Link**: [PDF](https://arxiv.org/pdf/2510.26461)  

**Abstract**: Recommender systems often struggle with data sparsity and cold-start scenarios, limiting their ability to provide accurate suggestions for new or infrequent users. This paper presents a Graph Attention Network (GAT) based Collaborative Filtering (CF) framework enhanced with Large Language Model (LLM) driven context aware embeddings. Specifically, we generate concise textual user profiles and unify item metadata (titles, genres, overviews) into rich textual embeddings, injecting these as initial node features in a bipartite user item graph. To further optimize ranking performance, we introduce a hybrid loss function that combines Bayesian Personalized Ranking (BPR) with a cosine similarity term and robust negative sampling, ensuring explicit negative feedback is distinguished from unobserved data. Experiments on the MovieLens 100k and 1M datasets show consistent improvements over state-of-the-art baselines in Precision, NDCG, and MAP while demonstrating robustness for users with limited interaction history. Ablation studies confirm the critical role of LLM-augmented embeddings and the cosine similarity term in capturing nuanced semantic relationships. Our approach effectively mitigates sparsity and cold-start limitations by integrating LLM-derived contextual understanding into graph-based architectures. Future directions include balancing recommendation accuracy with coverage and diversity, and introducing fairness-aware constraints and interpretability features to enhance system performance further. 

---
# Barlow Twins for Sequential Recommendation 

**Authors**: Ivan Razvorotnev, Marina Munkhoeva, Evgeny Frolov  

**Link**: [PDF](https://arxiv.org/pdf/2510.26407)  

**Abstract**: Sequential recommendation models must navigate sparse interaction data popularity bias and conflicting objectives like accuracy versus diversity While recent contrastive selfsupervised learning SSL methods offer improved accuracy they come with tradeoffs large batch requirements reliance on handcrafted augmentations and negative sampling that can reinforce popularity bias In this paper we introduce BT-SR a novel noncontrastive SSL framework that integrates the Barlow Twins redundancyreduction principle into a Transformerbased nextitem recommender BTSR learns embeddings that align users with similar shortterm behaviors while preserving longterm distinctionswithout requiring negative sampling or artificial perturbations This structuresensitive alignment allows BT-SR to more effectively recognize emerging user intent and mitigate the influence of noisy historical context Our experiments on five public benchmarks demonstrate that BTSR consistently improves nextitem prediction accuracy and significantly enhances longtail item coverage and recommendation calibration Crucially we show that a single hyperparameter can control the accuracydiversity tradeoff enabling practitioners to adapt recommendations to specific application needs 

---
# DiSE: A diffusion probabilistic model for automatic structure elucidation of organic compounds 

**Authors**: Haochen Chen, Qi Huang, Anan Wu, Wenhao Zhang, Jianliang Ye, Jianming Wu, Kai Tan, Xin Lu, Xin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26231)  

**Abstract**: Automatic structure elucidation is essential for self-driving laboratories as it enables the system to achieve truly autonomous. This capability closes the experimental feedback loop, ensuring that machine learning models receive reliable structure information for real-time decision-making and optimization. Herein, we present DiSE, an end-to-end diffusion-based generative model that integrates multiple spectroscopic modalities, including MS, 13C and 1H chemical shifts, HSQC, and COSY, to achieve automated yet accurate structure elucidation of organic compounds. By learning inherent correlations among spectra through data-driven approaches, DiSE achieves superior accuracy, strong generalization across chemically diverse datasets, and robustness to experimental data despite being trained on calculated spectra. DiSE thus represents a significant advance toward fully automated structure elucidation, with broad potential in natural product research, drug discovery, and self-driving laboratories. 

---
# ReaKase-8B: Legal Case Retrieval via Knowledge and Reasoning Representations with LLMs 

**Authors**: Yanran Tang, Ruihong Qiu, Xue Li, Zi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26178)  

**Abstract**: Legal case retrieval (LCR) is a cornerstone of real-world legal decision making, as it enables practitioners to identify precedents for a given query case. Existing approaches mainly rely on traditional lexical models and pretrained language models to encode the texts of legal cases. Yet there are rich information in the relations among different legal entities as well as the crucial reasoning process that uncovers how legal facts and legal issues can lead to judicial decisions. Such relational reasoning process reflects the distinctive characteristics of each case that can distinguish one from another, mirroring the real-world judicial process. Naturally, incorporating such information into the precise case embedding could further enhance the accuracy of case retrieval. In this paper, a novel ReaKase-8B framework is proposed to leverage extracted legal facts, legal issues, legal relation triplets and legal reasoning for effective legal case retrieval. ReaKase-8B designs an in-context legal case representation learning paradigm with a fine-tuned large language model. Extensive experiments on two benchmark datasets from COLIEE 2022 and COLIEE 2023 demonstrate that our knowledge and reasoning augmented embeddings substantially improve retrieval performance over baseline models, highlighting the potential of integrating legal reasoning into legal case retrieval systems. The code has been released on this https URL. 

---
# OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender 

**Authors**: Zhaoqi Zhang, Haolei Pei, Jun Guo, Tianyu Wang, Yufei Feng, Hui Sun, Shaowei Liu, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.26104)  

**Abstract**: In recommendation systems, scaling up feature-interaction modules (e.g., Wukong, RankMixer) or user-behavior sequence modules (e.g., LONGER) has achieved notable success. However, these efforts typically proceed on separate tracks, which not only hinders bidirectional information exchange but also prevents unified optimization and scaling. In this paper, we propose OneTrans, a unified Transformer backbone that simultaneously performs user-behavior sequence modeling and feature interaction. OneTrans employs a unified tokenizer to convert both sequential and non-sequential attributes into a single token sequence. The stacked OneTrans blocks share parameters across similar sequential tokens while assigning token-specific parameters to non-sequential tokens. Through causal attention and cross-request KV caching, OneTrans enables precomputation and caching of intermediate representations, significantly reducing computational costs during both training and inference. Experimental results on industrial-scale datasets demonstrate that OneTrans scales efficiently with increasing parameters, consistently outperforms strong baselines, and yields a 5.68% lift in per-user GMV in online A/B tests. 

---
# ORBIT - Open Recommendation Benchmark for Reproducible Research with Hidden Tests 

**Authors**: Jingyuan He, Jiongnan Liu, Vishan Vishesh Oberoi, Bolin Wu, Mahima Jagadeesh Patel, Kangrui Mao, Chuning Shi, I-Ta Lee, Arnold Overwijk, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.26095)  

**Abstract**: Recommender systems are among the most impactful AI applications, interacting with billions of users every day, guiding them to relevant products, services, or information tailored to their preferences. However, the research and development of recommender systems are hindered by existing datasets that fail to capture realistic user behaviors and inconsistent evaluation settings that lead to ambiguous conclusions. This paper introduces the Open Recommendation Benchmark for Reproducible Research with HIdden Tests (ORBIT), a unified benchmark for consistent and realistic evaluation of recommendation models. ORBIT offers a standardized evaluation framework of public datasets with reproducible splits and transparent settings for its public leaderboard. Additionally, ORBIT introduces a new webpage recommendation task, ClueWeb-Reco, featuring web browsing sequences from 87 million public, high-quality webpages. ClueWeb-Reco is a synthetic dataset derived from real, user-consented, and privacy-guaranteed browsing data. It aligns with modern recommendation scenarios and is reserved as the hidden test part of our leaderboard to challenge recommendation models' generalization ability. ORBIT measures 12 representative recommendation models on its public benchmark and introduces a prompted LLM baseline on the ClueWeb-Reco hidden test. Our benchmark results reflect general improvements of recommender systems on the public datasets, with variable individual performances. The results on the hidden test reveal the limitations of existing approaches in large-scale webpage recommendation and highlight the potential for improvements with LLM integrations. ORBIT benchmark, leaderboard, and codebase are available at this https URL. 

---
# AdSum: Two-stream Audio-visual Summarization for Automated Video Advertisement Clipping 

**Authors**: Wen Xie, Yanjun Zhu, Gijs Overgoor, Yakov Bart, Agata Lapedriza Garcia, Sarah Ostadabbas  

**Link**: [PDF](https://arxiv.org/pdf/2510.26569)  

**Abstract**: Advertisers commonly need multiple versions of the same advertisement (ad) at varying durations for a single campaign. The traditional approach involves manually selecting and re-editing shots from longer video ads to create shorter versions, which is labor-intensive and time-consuming. In this paper, we introduce a framework for automated video ad clipping using video summarization techniques. We are the first to frame video clipping as a shot selection problem, tailored specifically for advertising. Unlike existing general video summarization methods that primarily focus on visual content, our approach emphasizes the critical role of audio in advertising. To achieve this, we develop a two-stream audio-visual fusion model that predicts the importance of video frames, where importance is defined as the likelihood of a frame being selected in the firm-produced short ad. To address the lack of ad-specific datasets, we present AdSum204, a novel dataset comprising 102 pairs of 30-second and 15-second ads from real advertising campaigns. Extensive experiments demonstrate that our model outperforms state-of-the-art methods across various metrics, including Average Precision, Area Under Curve, Spearman, and Kendall. 

---
# Inside CORE-KG: Evaluating Structured Prompting and Coreference Resolution for Knowledge Graphs 

**Authors**: Dipak Meher, Carlotta Domeniconi  

**Link**: [PDF](https://arxiv.org/pdf/2510.26512)  

**Abstract**: Human smuggling networks are increasingly adaptive and difficult to analyze. Legal case documents offer critical insights but are often unstructured, lexically dense, and filled with ambiguous or shifting references, which pose significant challenges for automated knowledge graph (KG) construction. While recent LLM-based approaches improve over static templates, they still generate noisy, fragmented graphs with duplicate nodes due to the absence of guided extraction and coreference resolution. The recently proposed CORE-KG framework addresses these limitations by integrating a type-aware coreference module and domain-guided structured prompts, significantly reducing node duplication and legal noise. In this work, we present a systematic ablation study of CORE-KG to quantify the individual contributions of its two key components. Our results show that removing coreference resolution results in a 28.32% increase in node duplication and a 4.32% increase in noisy nodes, while removing structured prompts leads to a 4.34% increase in node duplication and a 73.33% increase in noisy nodes. These findings offer empirical insights for designing robust LLM-based pipelines for extracting structured representations from complex legal texts. 

---
# LINK-KG: LLM-Driven Coreference-Resolved Knowledge Graphs for Human Smuggling Networks 

**Authors**: Dipak Meher, Carlotta Domeniconi, Guadalupe Correa-Cabrera  

**Link**: [PDF](https://arxiv.org/pdf/2510.26486)  

**Abstract**: Human smuggling networks are complex and constantly evolving, making them difficult to analyze comprehensively. Legal case documents offer rich factual and procedural insights into these networks but are often long, unstructured, and filled with ambiguous or shifting references, posing significant challenges for automated knowledge graph (KG) construction. Existing methods either overlook coreference resolution or fail to scale beyond short text spans, leading to fragmented graphs and inconsistent entity linking. We propose LINK-KG, a modular framework that integrates a three-stage, LLM-guided coreference resolution pipeline with downstream KG extraction. At the core of our approach is a type-specific Prompt Cache, which consistently tracks and resolves references across document chunks, enabling clean and disambiguated narratives for structured knowledge graph construction from both short and long legal texts. LINK-KG reduces average node duplication by 45.21% and noisy nodes by 32.22% compared to baseline methods, resulting in cleaner and more coherent graph structures. These improvements establish LINK-KG as a strong foundation for analyzing complex criminal networks. 

---
# GraphCompliance: Aligning Policy and Context Graphs for LLM-Based Regulatory Compliance 

**Authors**: Jiseong Chung, Ronny Ko, Wonchul Yoo, Makoto Onizuka, Sungmok Kim, Tae-Wan Kim, Won-Yong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2510.26309)  

**Abstract**: Compliance at web scale poses practical challenges: each request may require a regulatory assessment. Regulatory texts (e.g., the General Data Protection Regulation, GDPR) are cross-referential and normative, while runtime contexts are expressed in unstructured natural language. This setting motivates us to align semantic information in unstructured text with the structured, normative elements of regulations. To this end, we introduce GraphCompliance, a framework that represents regulatory texts as a Policy Graph and runtime contexts as a Context Graph, and aligns them. In this formulation, the policy graph encodes normative structure and cross-references, whereas the context graph formalizes events as subject-action-object (SAO) and entity-relation triples. This alignment anchors the reasoning of a judge large language model (LLM) in structured information and helps reduce the burden of regulatory interpretation and event parsing, enabling a focus on the core reasoning step. In experiments on 300 GDPR-derived real-world scenarios spanning five evaluation tasks, GraphCompliance yields 4.1-7.2 percentage points (pp) higher micro-F1 than LLM-only and RAG baselines, with fewer under- and over-predictions, resulting in higher recall and lower false positive rates. Ablation studies indicate contributions from each graph component, suggesting that structured representations and a judge LLM are complementary for normative reasoning. 

---
# The Quest for Reliable Metrics of Responsible AI 

**Authors**: Theresia Veronika Rampisela, Maria Maistro, Tuukka Ruotsalo, Christina Lioma  

**Link**: [PDF](https://arxiv.org/pdf/2510.26007)  

**Abstract**: The development of Artificial Intelligence (AI), including AI in Science (AIS), should be done following the principles of responsible AI. Progress in responsible AI is often quantified through evaluation metrics, yet there has been less work on assessing the robustness and reliability of the metrics themselves. We reflect on prior work that examines the robustness of fairness metrics for recommender systems as a type of AI application and summarise their key takeaways into a set of non-exhaustive guidelines for developing reliable metrics of responsible AI. Our guidelines apply to a broad spectrum of AI applications, including AIS. 

---
