# A Real-Time System to Populate FRA Form 57 from News 

**Authors**: Chansong Lim, Haz Sameen Shahgir, Yue Dong, Jia Chen, Evangelos E. Papalexakis  

**Link**: [PDF](https://arxiv.org/pdf/2512.22457)  

**Abstract**: Local railway committees need timely situational awareness after highway-rail grade crossing incidents, yet official Federal Railroad Administration (FRA) investigations can take days to weeks. We present a demo system that populates Highway-Rail Grade Crossing Incident Data (Form 57) from news in real time. Our approach addresses two core challenges: the form is visually irregular and semantically dense, and news is noisy. To solve these problems, we design a pipeline that first converts Form 57 into a JSON schema using a vision language model with sample aggregation, and then performs grouped question answering following the intent of the form layout to reduce ambiguity. In addition, we build an evaluation dataset by aligning scraped news articles with official FRA records and annotating retrievable information. We then assess our system against various alternatives in terms of information retrieval accuracy and coverage. 

---
# OxygenREC: An Instruction-Following Generative Framework for E-commerce Recommendation 

**Authors**: Xuegang Hao, Ming Zhang, Alex Li, Xiangyu Qian, Zhi Ma, Yanlong Zang, Shijie Yang, Zhongxuan Han, Xiaolong Ma, Jinguang Liu, Zhen Li, Zhida Jiang, Shusheng Wang, Ning Tang, Yanchen Qiao, Chenxiang Yang, Chen Sun, Jincheng Yuan, Chunhua Peng, Heng Hu, Peijun Yang, Baopeng Yuan, Caiyun Qiu, Zhaolong Xing, Haofei Yuan, Haipeng Zhang, Yuzhang Guo, Weijie Ding, Jiahua Gao, Hao Huang, Zhen Chen, Tongxuan Liu, Pinghua Gong  

**Link**: [PDF](https://arxiv.org/pdf/2512.22386)  

**Abstract**: Traditional recommendation systems suffer from inconsistency in multi-stage optimization objectives. Generative Recommendation (GR) mitigates them through an end-to-end framework; however, existing methods still rely on matching mechanisms based on inductive patterns. Although responsive, they lack the ability to uncover complex user intents that require deductive reasoning based on world knowledge. Meanwhile, LLMs show strong deep reasoning capabilities, but their latency and computational costs remain challenging for industrial applications. More critically, there are performance bottlenecks in multi-scenario scalability: as shown in Figure 1, existing solutions require independent training and deployment for each scenario, leading to low resource utilization and high maintenance costs-a challenge unaddressed in GR literature. To address these, we present OxygenREC, an industrial recommendation system that leverages Fast-Slow Thinking to deliver deep reasoning with strict latency and multi-scenario requirements of real-world environments. First, we adopt a Fast-Slow Thinking architecture. Slow thinking uses a near-line LLM pipeline to synthesize Contextual Reasoning Instructions, while fast thinking employs a high-efficiency encoder--decoder backbone for real-time generation. Second, to ensure reasoning instructions effectively enhance recommendation generation, we introduce a semantic alignment mechanism with Instruction-Guided Retrieval (IGR) to filter intent-relevant historical behaviors and use a Query-to-Item (Q2I) loss for instruction-item consistency. Finally, to resolve multi-scenario scalability, we transform scenario information into controllable instructions, using unified reward mapping and Soft Adaptive Group Clip Policy Optimization (SA-GCPO) to align policies with diverse business objectives, realizing a train-once-deploy-everywhere paradigm. 

---
# Nested Browser-Use Learning for Agentic Information Seeking 

**Authors**: Baixuan Li, Jialong Wu, Wenbiao Yin, Kuan Li, Zhongwang Zhang, Huifeng Yin, Zhengwei Tao, Liwen Zhang, Pengjun Xie, Jingren Zhou, Yong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23647)  

**Abstract**: Information-seeking (IS) agents have achieved strong performance across a range of wide and deep search tasks, yet their tool use remains largely restricted to API-level snippet retrieval and URL-based page fetching, limiting access to the richer information available through real browsing. While full browser interaction could unlock deeper capabilities, its fine-grained control and verbose page content returns introduce substantial complexity for ReAct-style function-calling agents. To bridge this gap, we propose Nested Browser-Use Learning (NestBrowse), which introduces a minimal and complete browser-action framework that decouples interaction control from page exploration through a nested structure. This design simplifies agentic reasoning while enabling effective deep-web information acquisition. Empirical results on challenging deep IS benchmarks demonstrate that NestBrowse offers clear benefits in practice. Further in-depth analyses underscore its efficiency and flexibility. 

---
# Scalable Residual Feature Aggregation Framework with Hybrid Metaheuristic Optimization for Robust Early Pancreatic Neoplasm Detection in Multimodal CT Imaging 

**Authors**: Janani Annur Thiruvengadam, Kiran Mayee Nabigaru, Anusha Kovi  

**Link**: [PDF](https://arxiv.org/pdf/2512.23597)  

**Abstract**: The early detection of pancreatic neoplasm is a major clinical dilemma, and it is predominantly so because tumors are likely to occur with minimal contrast margins and a large spread anatomy-wide variation amongst patients on a CT scan. These complexities require to be addressed with an effective and scalable system that can assist in enhancing the salience of the subtle visual cues and provide a high level of the generalization on the multimodal imaging data. A Scalable Residual Feature Aggregation (SRFA) framework is proposed to be used to meet these conditions in this study. The framework integrates a pipeline of preprocessing followed by the segmentation using the MAGRes-UNet that is effective in making the pancreatic structures and isolating regions of interest more visible. DenseNet-121 performed with residual feature storage is used to extract features to allow deep hierarchical features to be aggregated without properties loss. To go further, hybrid HHO-BA metaheuristic feature selection strategy is used, which guarantees the best feature subset refinement. To be classified, the system is trained based on a new hybrid model that integrates the ability to pay attention on the world, which is the Vision Transformer (ViT) with the high representational efficiency of EfficientNet-B3. A dual optimization mechanism incorporating SSA and GWO is used to fine-tune hyperparameters to enhance greater robustness and less overfitting. Experimental results support the significant improvement in performance, with the suggested model reaching 96.23% accuracy, 95.58% F1-score and 94.83% specificity, the model is significantly better than the traditional CNNs and contemporary transformer-based models. Such results highlight the possibility of the SRFA framework as a useful instrument in the early detection of pancreatic tumors. 

---
# RobustMask: Certified Robustness against Adversarial Neural Ranking Attack via Randomized Masking 

**Authors**: Jiawei Liu, Zhuo Chen, Rui Zhu, Miaokun Chen, Yuyang Gong, Wei Lu, Xiaofeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23307)  

**Abstract**: Neural ranking models have achieved remarkable progress and are now widely deployed in real-world applications such as Retrieval-Augmented Generation (RAG). However, like other neural architectures, they remain vulnerable to adversarial manipulations: subtle character-, word-, or phrase-level perturbations can poison retrieval results and artificially promote targeted candidates, undermining the integrity of search engines and downstream systems. Existing defenses either rely on heuristics with poor generalization or on certified methods that assume overly strong adversarial knowledge, limiting their practical use. To address these challenges, we propose RobustMask, a novel defense that combines the context-prediction capability of pretrained language models with a randomized masking-based smoothing mechanism. Our approach strengthens neural ranking models against adversarial perturbations at the character, word, and phrase levels. Leveraging both the pairwise comparison ability of ranking models and probabilistic statistical analysis, we provide a theoretical proof of RobustMask's certified top-K robustness. Extensive experiments further demonstrate that RobustMask successfully certifies over 20% of candidate documents within the top-10 ranking positions against adversarial perturbations affecting up to 30% of their content. These results highlight the effectiveness of RobustMask in enhancing the adversarial robustness of neural ranking models, marking a significant step toward providing stronger security guarantees for real-world retrieval systems. 

---
# OrchANN: A Unified I/O Orchestration Framework for Skewed Out-of-Core Vector Search 

**Authors**: Chengying Huan, Lizheng Chen, Zhengyi Yang, Shaonan Ma, Rong Gu, Renjie Yao, Zhibin Wang, Mingxing Zhang, Fang Xi, Jie Tao, Gang Zhang, Guihai Chen, Chen Tian  

**Link**: [PDF](https://arxiv.org/pdf/2512.22838)  

**Abstract**: Approximate nearest neighbor search (ANNS) at billion scale is fundamentally an out-of-core problem: vectors and indexes live on SSD, so performance is dominated by I/O rather than compute. Under skewed semantic embeddings, existing out-of-core systems break down: a uniform local index mismatches cluster scales, static routing misguides queries and inflates the number of probed partitions, and pruning is incomplete at the cluster level and lossy at the vector level, triggering "fetch-to-discard" reranking on raw vectors.
We present OrchANN, an out-of-core ANNS engine that uses an I/O orchestration model for unified I/O governance along the route-access-verify pipeline. OrchANN selects a heterogeneous local index per cluster via offline auto-profiling, maintains a query-aware in-memory navigation graph that adapts to skewed workloads, and applies multi-level pruning with geometric bounds to filter both clusters and vectors before issuing SSD reads. Across five standard datasets under strict out-of-core constraints, OrchANN outperforms four baselines including DiskANN, Starling, SPANN, and PipeANN in both QPS and latency while reducing SSD accesses. Furthermore, OrchANN delivers up to 17.2x higher QPS and 25.0x lower latency than competing systems without sacrificing accuracy. 

---
# DICE: Discrete Interpretable Comparative Evaluation with Probabilistic Scoring for Retrieval-Augmented Generation 

**Authors**: Shiyan Liu, Jian Ma, Rui Qu  

**Link**: [PDF](https://arxiv.org/pdf/2512.22629)  

**Abstract**: As Retrieval-Augmented Generation (RAG) systems evolve toward more sophisticated architectures, ensuring their trustworthiness through explainable and robust evaluation becomes critical. Existing scalar metrics suffer from limited interpretability, inadequate uncertainty quantification, and computational inefficiency in multi-system comparisons, hindering responsible deployment of RAG technologies. We introduce DICE (Discrete Interpretable Comparative Evaluation), a two-stage, evidence-coupled framework that advances explainability and robustness in RAG evaluation. DICE combines deep analytical reasoning with probabilistic $\{A, B, Tie\}$ scoring to produce transparent, confidence-aware judgments that support accountable system improvement through interpretable reasoning traces, enabling systematic error diagnosis and actionable insights. To address efficiency challenges at scale, DICE employs a Swiss-system tournament that reduces computational complexity from $O(N^2)$ to $O(N \log N)$, achieving a 42.9% reduction in our eight-system evaluation while preserving ranking fidelity. Validation on a curated Chinese financial QA dataset demonstrates that DICE achieves 85.7% agreement with human experts, substantially outperforming existing LLM-based metrics such as RAGAS. Our results establish DICE as a responsible, explainable, and efficient paradigm for trustworthy RAG system assessment. 

---
# HiFi-RAG: Hierarchical Content Filtering and Two-Pass Generation for Open-Domain RAG 

**Authors**: Cattalyya Nuengsigkapian  

**Link**: [PDF](https://arxiv.org/pdf/2512.22442)  

**Abstract**: Retrieval-Augmented Generation (RAG) in open-domain settings faces significant challenges regarding irrelevant information in retrieved documents and the alignment of generated answers with user intent. We present HiFi-RAG (Hierarchical Filtering RAG), the winning closed-source system in the Text-to-Text static evaluation of the MMU-RAGent NeurIPS 2025 Competition. Our approach moves beyond standard embedding-based retrieval via a multi-stage pipeline. We leverage the speed and cost-efficiency of Gemini 2.5 Flash (4-6x cheaper than Pro) for query formulation, hierarchical content filtering, and citation attribution, while reserving the reasoning capabilities of Gemini 2.5 Pro for final answer generation. On the MMU-RAGent validation set, our system outperformed the baseline, improving ROUGE-L to 0.274 (+19.6%) and DeBERTaScore to 0.677 (+6.2%). On Test2025, our custom dataset evaluating questions that require post-cutoff knowledge (post January 2025), HiFi-RAG outperforms the parametric baseline by 57.4% in ROUGE-L and 14.9% in DeBERTaScore. 

---
# Hallucination Detection and Evaluation of Large Language Model 

**Authors**: Chenggong Zhang, Haopeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.22416)  

**Abstract**: Hallucinations in Large Language Models (LLMs) pose a significant challenge, generating misleading or unverifiable content that undermines trust and reliability. Existing evaluation methods, such as KnowHalu, employ multi-stage verification but suffer from high computational costs. To address this, we integrate the Hughes Hallucination Evaluation Model (HHEM), a lightweight classification-based framework that operates independently of LLM-based judgments, significantly improving efficiency while maintaining high detection accuracy. We conduct a comparative analysis of hallucination detection methods across various LLMs, evaluating True Positive Rate (TPR), True Negative Rate (TNR), and Accuracy on question-answering (QA) and summarization tasks. Our results show that HHEM reduces evaluation time from 8 hours to 10 minutes, while HHEM with non-fabrication checking achieves the highest accuracy \(82.2\%\) and TPR \(78.9\%\). However, HHEM struggles with localized hallucinations in summarization tasks. To address this, we introduce segment-based retrieval, improving detection by verifying smaller text components. Additionally, our cumulative distribution function (CDF) analysis indicates that larger models (7B-9B parameters) generally exhibit fewer hallucinations, while intermediate-sized models show higher instability. These findings highlight the need for structured evaluation frameworks that balance computational efficiency with robust factual validation, enhancing the reliability of LLM-generated content. 

---
# HalluMat: Detecting Hallucinations in LLM-Generated Materials Science Content Through Multi-Stage Verification 

**Authors**: Bhanu Prakash Vangala, Sajid Mahmud, Pawan Neupane, Joel Selvaraj, Jianlin Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.22396)  

**Abstract**: Artificial Intelligence (AI), particularly Large Language Models (LLMs), is transforming scientific discovery, enabling rapid knowledge generation and hypothesis formulation. However, a critical challenge is hallucination, where LLMs generate factually incorrect or misleading information, compromising research integrity. To address this, we introduce HalluMatData, a benchmark dataset for evaluating hallucination detection methods, factual consistency, and response robustness in AI-generated materials science content. Alongside this, we propose HalluMatDetector, a multi-stage hallucination detection framework that integrates intrinsic verification, multi-source retrieval, contradiction graph analysis, and metric-based assessment to detect and mitigate LLM hallucinations. Our findings reveal that hallucination levels vary significantly across materials science subdomains, with high-entropy queries exhibiting greater factual inconsistencies. By utilizing HalluMatDetector verification pipeline, we reduce hallucination rates by 30% compared to standard LLM outputs. Furthermore, we introduce the Paraphrased Hallucination Consistency Score (PHCS) to quantify inconsistencies in LLM responses across semantically equivalent queries, offering deeper insights into model reliability. 

---
# IANEC: Digital Forensic Investigation of Contemporary Writers' Archives 

**Authors**: Emmanuel Giguet  

**Link**: [PDF](https://arxiv.org/pdf/2512.22167)  

**Abstract**: The IANEC project (Investigation of Digital Archives of Contemporary Writers), led by the GREYC Research Lab and funded by the French Ministry of Culture aims to develop dedicated digital forensic investigation tools to automate the analysis of archival corpora from the Institut M{é}moires de l'{É}dition Contemporaine (IMEC). The project is based on the observation that born-digital archival materials are increasingly prevalent in contemporary archival institutions, and that digital forensics technologies have become essential for the extraction, identification, processing, and description of natively digital archival corpora.* 

---
