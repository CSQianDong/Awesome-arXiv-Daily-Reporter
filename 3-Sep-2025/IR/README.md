# Lighting the Way for BRIGHT: Reproducible Baselines with Anserini, Pyserini, and RankLLM 

**Authors**: Yijun Ge, Sahel Sharifymoghaddam, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.02558)  

**Abstract**: The BRIGHT benchmark is a dataset consisting of reasoning-intensive queries over diverse domains. We explore retrieval results on BRIGHT using a range of retrieval techniques, including sparse, dense, and fusion methods, and establish reproducible baselines. We then apply listwise reranking with large language models (LLMs) to further investigate the impact of reranking on reasoning-intensive queries. These baselines are integrated into popular retrieval and reranking toolkits Anserini, Pyserini, and RankLLM, with two-click reproducibility that makes them easy to build upon and convenient for further development. While attempting to reproduce the results reported in the original BRIGHT paper, we find that the provided BM25 scores differ notably from those that we obtain using Anserini and Pyserini. We discover that this difference is due to BRIGHT's implementation of BM25, which applies BM25 on the query rather than using the standard bag-of-words approach, as in Anserini, to construct query vectors. This difference has become increasingly relevant due to the rise of longer queries, with BRIGHT's lengthy reasoning-intensive queries being a prime example, and further accentuated by the increasing usage of retrieval-augmented generation, where LLM prompts can grow to be much longer than ''traditional'' search engine queries. Our observation signifies that it may be time to reconsider BM25 approaches going forward in order to better accommodate emerging applications. To facilitate this, we integrate query-side BM25 into both Anserini and Pyserini. 

---
# Upcycling Candidate Tokens of Large Language Models for Query Expansion 

**Authors**: Jinseok Kim, Sukmin Cho, Soyeong Jeong, Sangyeop Kim, Sungzoon Cho  

**Link**: [PDF](https://arxiv.org/pdf/2509.02377)  

**Abstract**: Query Expansion (QE) improves retrieval performance by enriching queries with related terms. Recently, Large Language Models (LLMs) have been used for QE, but existing methods face a trade-off: generating diverse terms boosts performance but increases computational cost. To address this challenge, we propose Candidate Token Query Expansion (CTQE), which extracts diverse and relevant terms from a single LLM decoding pass by leveraging unselected candidate tokens. These tokens, though not part of the final output, are conditioned on the full query and capture useful information. By aggregating them, CTQE achieves both relevance and diversity without extra inference, reducing overhead and latency. Experiments show that CTQE delivers strong retrieval performance with significantly lower cost, outperforming or comparable to more expensive methods. Code is available at: this https URL 

---
# Leveraging Media Frames to Improve Normative Diversity in News Recommendations 

**Authors**: Sourabh Dattawad, Agnese Daffara, Tanise Ceron  

**Link**: [PDF](https://arxiv.org/pdf/2509.02266)  

**Abstract**: Click-based news recommender systems suggest users content that aligns with their existing history, limiting the diversity of articles they encounter. Recent advances in aspect-based diversification -- adding features such as sentiments or news categories (e.g. world, politics) -- have made progress toward diversifying recommendations in terms of perspectives. However, these approaches often overlook the role of news framing, which shapes how stories are told by emphasizing specific angles or interpretations. In this paper, we treat media frames as a controllable aspect within the recommendation pipeline. By selecting articles based on a diversity of frames, our approach emphasizes varied narrative angles and broadens the interpretive space recommended to users. In addition to introducing frame-based diversification method, our work is the first to assess the impact of a news recommender system that integrates frame diversity using normative diversity metrics: representation, calibration, and activation. Our experiments based on media frame diversification show an improvement in exposure to previously unclicked frames up to 50%. This is important because repeated exposure to the same frames can reinforce existing biases or narrow interpretations, whereas introducing novel frames broadens users' understanding of issues and perspectives. The method also enhances diversification across categorical and sentiment levels, thereby demonstrating that framing acts as a strong control lever for enhancing normative diversity. 

---
# Application Of Large Language Models For The Extraction Of Information From Particle Accelerator Technical Documentation 

**Authors**: Qing Dai, Rasmus Ischebeck, Maruisz Sapinski, Adam Grycner  

**Link**: [PDF](https://arxiv.org/pdf/2509.02227)  

**Abstract**: The large set of technical documentation of legacy accelerator systems, coupled with the retirement of experienced personnel, underscores the urgent need for efficient methods to preserve and transfer specialized knowledge. This paper explores the application of large language models (LLMs), to automate and enhance the extraction of information from particle accelerator technical documents. By exploiting LLMs, we aim to address the challenges of knowledge retention, enabling the retrieval of domain expertise embedded in legacy documentation. We present initial results of adapting LLMs to this specialized domain. Our evaluation demonstrates the effectiveness of LLMs in extracting, summarizing, and organizing knowledge, significantly reducing the risk of losing valuable insights as personnel retire. Furthermore, we discuss the limitations of current LLMs, such as interpretability and handling of rare domain-specific terms, and propose strategies for improvement. This work highlights the potential of LLMs to play a pivotal role in preserving institutional knowledge and ensuring continuity in highly specialized fields. 

---
# Towards Multi-Aspect Diversification of News Recommendations Using Neuro-Symbolic AI for Individual and Societal Benefit 

**Authors**: Markus Reiter-Haas, Elisabeth Lex  

**Link**: [PDF](https://arxiv.org/pdf/2509.02220)  

**Abstract**: News recommendations are complex, with diversity playing a vital role. So far, existing literature predominantly focuses on specific aspects of news diversity, such as viewpoints. In this paper, we introduce multi-aspect diversification in four distinct recommendation modes and outline the nuanced challenges in diversifying lists, sequences, summaries, and interactions. Our proposed research direction combines symbolic and subsymbolic artificial intelligence, leveraging both knowledge graphs and rule learning. We plan to evaluate our models using user studies to not only capture behavior but also their perceived experience. Our vision to balance news consumption points to other positive effects for users (e.g., increased serendipity) and society (e.g., decreased polarization). 

---
# Empowering Large Language Model for Sequential Recommendation via Multimodal Embeddings and Semantic IDs 

**Authors**: Yuhao Wang, Junwei Pan, Xinhang Li, Maolin Wang, Yuan Wang, Yue Liu, Dapeng Liu, Jie Jiang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.02017)  

**Abstract**: Sequential recommendation (SR) aims to capture users' dynamic interests and sequential patterns based on their historical interactions. Recently, the powerful capabilities of large language models (LLMs) have driven their adoption in SR. However, we identify two critical challenges in existing LLM-based SR methods: 1) embedding collapse when incorporating pre-trained collaborative embeddings and 2) catastrophic forgetting of quantized embeddings when utilizing semantic IDs. These issues dampen the model scalability and lead to suboptimal recommendation performance. Therefore, based on LLMs like Llama3-8B-instruct, we introduce a novel SR framework named MME-SID, which integrates multimodal embeddings and quantized embeddings to mitigate embedding collapse. Additionally, we propose a Multimodal Residual Quantized Variational Autoencoder (MM-RQ-VAE) with maximum mean discrepancy as the reconstruction loss and contrastive learning for alignment, which effectively preserve intra-modal distance information and capture inter-modal correlations, respectively. To further alleviate catastrophic forgetting, we initialize the model with the trained multimodal code embeddings. Finally, we fine-tune the LLM efficiently using LoRA in a multimodal frequency-aware fusion manner. Extensive experiments on three public datasets validate the superior performance of MME-SID thanks to its capability to mitigate embedding collapse and catastrophic forgetting. The implementation code and datasets are publicly available for reproduction: this https URL. 

---
# CSRM-LLM: Embracing Multilingual LLMs for Cold-Start Relevance Matching in Emerging E-commerce Markets 

**Authors**: Yujing Wang, Yiren Chen, Huoran Li, Chunxu Xu, Yuchong Luo, Xianghui Mao, Cong Li, Lun Du, Chunyang Ma, Qiqi Jiang, Yin Wang, Fan Gao, Wenting Mo, Pei Wen, Shantanu Kumar, Taejin Park, Yiwei Song, Vijay Rajaram, Tao Cheng, Sonu Durgia, Pranam Kolari  

**Link**: [PDF](https://arxiv.org/pdf/2509.01566)  

**Abstract**: As global e-commerce platforms continue to expand, companies are entering new markets where they encounter cold-start challenges due to limited human labels and user behaviors. In this paper, we share our experiences in Coupang to provide a competitive cold-start performance of relevance matching for emerging e-commerce markets. Specifically, we present a Cold-Start Relevance Matching (CSRM) framework, utilizing a multilingual Large Language Model (LLM) to address three challenges: (1) activating cross-lingual transfer learning abilities of LLMs through machine translation tasks; (2) enhancing query understanding and incorporating e-commerce knowledge by retrieval-based query augmentation; (3) mitigating the impact of training label errors through a multi-round self-distillation training strategy. Our experiments demonstrate the effectiveness of CSRM-LLM and the proposed techniques, resulting in successful real-world deployment and significant online gains, with a 45.8% reduction in defect ratio and a 0.866% uplift in session purchase rate. 

---
# Cloud-Device Collaborative Agents for Sequential Recommendation 

**Authors**: Jing Long, Sirui Huang, Huan Huo, Tong Chen, Hongzhi Yin, Guandong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.01551)  

**Abstract**: Recent advances in large language models (LLMs) have enabled agent-based recommendation systems with strong semantic understanding and flexible reasoning capabilities. While LLM-based agents deployed in the cloud offer powerful personalization, they often suffer from privacy concerns, limited access to real-time signals, and scalability bottlenecks. Conversely, on-device agents ensure privacy and responsiveness but lack the computational power for global modeling and large-scale retrieval. To bridge these complementary limitations, we propose CDA4Rec, a novel Cloud-Device collaborative framework for sequential Recommendation, powered by dual agents: a cloud-side LLM and a device-side small language model (SLM). CDA4Rec tackles the core challenge of cloud-device coordination by decomposing the recommendation task into modular sub-tasks including semantic modeling, candidate retrieval, structured user modeling, and final ranking, which are allocated to cloud or device based on computational demands and privacy sensitivity. A strategy planning mechanism leverages the cloud agent's reasoning ability to generate personalized execution plans, enabling context-aware task assignment and partial parallel execution across agents. This design ensures real-time responsiveness, improved efficiency, and fine-grained personalization, even under diverse user states and behavioral sparsity. Extensive experiments across multiple real-world datasets demonstrate that CDA4Rec consistently outperforms competitive baselines in both accuracy and efficiency, validating its effectiveness in heterogeneous and resource-constrained environments. 

---
# Ultra Fast Warm Start Solution for Graph Recommendations 

**Authors**: Viacheslav Yusupov, Maxim Rakhuba, Evgeny Frolov  

**Link**: [PDF](https://arxiv.org/pdf/2509.01549)  

**Abstract**: In this work, we present a fast and effective Linear approach for updating recommendations in a scalable graph-based recommender system UltraGCN. Solving this task is extremely important to maintain the relevance of the recommendations under the conditions of a large amount of new data and changing user preferences. To address this issue, we adapt the simple yet effective low-rank approximation approach to the graph-based model. Our method delivers instantaneous recommendations that are up to 30 times faster than conventional methods, with gains in recommendation quality, and demonstrates high scalability even on the large catalogue datasets. 

---
# AI4DiTraRe: Building the BFO-Compliant Chemotion Knowledge Graph 

**Authors**: Ebrahim Norouzi, Nicole Jung, Anna M. Jacyszyn, Jörg Waitelonis, Harald Sack  

**Link**: [PDF](https://arxiv.org/pdf/2509.01536)  

**Abstract**: Chemistry is an example of a discipline where the advancements of technology have led to multi-level and often tangled and tricky processes ongoing in the lab. The repeatedly complex workflows are combined with information from chemical structures, which are essential to understand the scientific process. An important tool for many chemists is Chemotion, which consists of an electronic lab notebook and a repository. This paper introduces a semantic pipeline for constructing the BFO-compliant Chemotion Knowledge Graph, providing an integrated, ontology-driven representation of chemical research data. The Chemotion-KG has been developed to adhere to the FAIR (Findable, Accessible, Interoperable, Reusable) principles and to support AI-driven discovery and reasoning in chemistry. Experimental metadata were harvested from the Chemotion API in JSON-LD format, converted into RDF, and subsequently transformed into a Basic Formal Ontology-aligned graph through SPARQL CONSTRUCT queries. The source code and datasets are publicly available via GitHub. The Chemotion Knowledge Graph is hosted by FIZ Karlsruhe Information Service Engineering. Outcomes presented in this work were achieved within the Leibniz Science Campus ``Digital Transformation of Research'' (DiTraRe) and are part of an ongoing interdisciplinary collaboration. 

---
# Re3: Learning to Balance Relevance & Recency for Temporal Information Retrieval 

**Authors**: Jiawei Cao, Jie Ouyang, Zhaomeng Zhou, Mingyue Cheng, Yupeng Li, Jiaxian Yan, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.01306)  

**Abstract**: Temporal Information Retrieval (TIR) is a critical yet unresolved task for modern search systems, retrieving documents that not only satisfy a query's information need but also adhere to its temporal constraints. This task is shaped by two challenges: Relevance, ensuring alignment with the query's explicit temporal requirements, and Recency, selecting the freshest document among multiple versions. Existing methods often address the two challenges in isolation, relying on brittle heuristics that fail in scenarios where temporal requirements and staleness resistance are intertwined. To address this gap, we introduce Re2Bench, a benchmark specifically designed to disentangle and evaluate Relevance, Recency, and their hybrid combination. Building on this foundation, we propose Re3, a unified and lightweight framework that dynamically balances semantic and temporal information through a query-aware gating mechanism. On Re2Bench, Re3 achieves state-of-the-art results, leading in R@1 across all three subsets. Ablation studies with backbone sensitivity tests confirm robustness, showing strong generalization across diverse encoders and real-world settings. This work provides both a generalizable solution and a principled evaluation suite, advancing the development of temporally aware retrieval systems. Re3 and Re2Bench are available online: this https URL 

---
# MARS: Modality-Aligned Retrieval for Sequence Augmented CTR Prediction 

**Authors**: Yutian Xiao, Shukuan Wang, Binhao Wang, Zhao Zhang, Yanze Zhang, Shanqi Liu, Chao Feng, Xiang Li, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01184)  

**Abstract**: Click-through rate (CTR) prediction serves as a cornerstone of recommender systems. Despite the strong performance of current CTR models based on user behavior modeling, they are still severely limited by interaction sparsity, especially in low-active user scenarios. To address this issue, data augmentation of user behavior is a promising research direction. However, existing data augmentation methods heavily rely on collaborative signals while overlooking the rich multimodal features of items, leading to insufficient modeling of low-active users.
To alleviate this problem, we propose a novel framework \textbf{MARS} (\textbf{M}odality-\textbf{A}ligned \textbf{R}etrieval for \textbf{S}equence Augmented CTR Prediction). MARS utilizes a Stein kernel-based approach to align text and image features into a unified and unbiased semantic space to construct multimodal user embeddings. Subsequently, each low-active user's behavior sequence is augmented by retrieving, filtering, and concentrating the most similar behavior sequence of high-active users via multimodal user embeddings. Validated by extensive offline experiments and online A/B tests, our framework MARS consistently outperforms state-of-the-art baselines and achieves substantial growth on core business metrics within Kuaishou~\footnote{this https URL}. Consequently, MARS has been successfully deployed, serving the main traffic for hundreds of millions of users. To ensure reproducibility, we provide anonymous access to the implementation code~\footnote{this https URL}. 

---
# Beyond the Surface: A Solution-Aware Retrieval Model for Competition-level Code Generation 

**Authors**: Shiwen Zhang, Lingxiang Wang, Hainan Zhang, Ziwei Wang, Sijia Wen, Zhiming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.01129)  

**Abstract**: In competitive programming task, problem statements are often embedded within elaborate narrative backgrounds, requiring deep understanding of the underlying solutions to successfully complete the tasks. Current code generation models primarily focus on token-level semantic modeling, highly susceptible to distractions from irrelevant narrative statements. Inspired by RAG, retrieving reference code with similar solutions may help enhance model performance on difficult problems. However, existing retrieval models also emphasize surface-level semantic similarity, neglecting the deeper solution-level logical similarities that are critical in competitive programming. Therefore, designing ranking models capable of accurately identifying and retrieving problems and corresponding codes remains an urgent research problem in competitive code generation. In this paper, we propose SolveRank, a solution-aware ranking model empowered by synthetic data for competitive programming tasks. Specifically, we leverage the DeepSeek-R1 model to generate logically equivalent but differently phrased new problems, verified by GPT-4o for solution consistency. Then, we train SolveRank with these as positive samples and BM25/random-retrieved problems as negatives. During inference, SolveRank retrieves relevant problems and corresponding code from the corpus to assist a downstream code generator. Experiments on the xCodeEval dataset demonstrate that SolveRank outperforms SOTA ranking methods in precision and recall metrics, and boosts code generation performance for difficult problems. 

---
# Identifying Origins of Place Names via Retrieval Augmented Generation 

**Authors**: Alexis Horde Vo, Matt Duckham, Estrid He, Rafe Benli  

**Link**: [PDF](https://arxiv.org/pdf/2509.01030)  

**Abstract**: Who is the "Batman" behind "Batman Street" in Melbourne? Understanding the historical, cultural, and societal narratives behind place names can reveal the rich context that has shaped a community. Although place names serve as essential spatial references in gazetteers, they often lack information about place name origins. Enriching these place names in today's gazetteers is a time-consuming, manual process that requires extensive exploration of a vast archive of documents and text sources. Recent advances in natural language processing and language models (LMs) hold the promise of significant automation of identifying place name origins due to their powerful capability to exploit the semantics of the stored documents. This chapter presents a retrieval augmented generation pipeline designed to search for place name origins over a broad knowledge base, DBpedia. Given a spatial query, our approach first extracts sub-graphs that may contain knowledge relevant to the query; then ranks the extracted sub-graphs to generate the final answer to the query using fine-tuned LM-based models (i.e., ColBERTv2 and Llama2). Our results highlight the key challenges facing automated retrieval of place name origins, especially the tendency of language models to under-use the spatial information contained in texts as a discriminating factor. Our approach also frames the wider implications for geographic information retrieval using retrieval augmented generation. 

---
# Food Data in the Semantic Web: A Review of Nutritional Resources, Knowledge Graphs, and Emerging Applications 

**Authors**: Darko Sasanski, Riste Stojanov  

**Link**: [PDF](https://arxiv.org/pdf/2509.00986)  

**Abstract**: This comprehensive review explores food data in the Semantic Web, highlighting key nutritional resources, knowledge graphs, and emerging applications in the food domain. It examines prominent food data resources such as USDA, FoodOn, FooDB, and Recipe1M+, emphasizing their contributions to nutritional data representation. Special focus is given to food entity linking and recognition techniques, which enable integration of heterogeneous food data sources into cohesive semantic resources. The review further discusses food knowledge graphs, their role in semantic interoperability, data enrichment, and knowledge extraction, and their applications in personalized nutrition, ingredient substitution, food-drug and food-disease interactions, and interdisciplinary research. By synthesizing current advancements and identifying challenges, this work provides insights to guide future developments in leveraging semantic technologies for the food domain. 

---
# HiPS: Hierarchical PDF Segmentation of Textbooks 

**Authors**: Sabine Wehnert, Harikrishnan Changaramkulath, Ernesto William De Luca  

**Link**: [PDF](https://arxiv.org/pdf/2509.00909)  

**Abstract**: The growing demand for effective tools to parse PDF-formatted texts, particularly structured documents such as textbooks, reveals the limitations of current methods developed mainly for research paper segmentation. This work addresses the challenge of hierarchical segmentation in complex structured documents, with a focus on legal textbooks that contain layered knowledge essential for interpreting and applying legal norms. We examine a Table of Contents (TOC)-based technique and approaches that rely on open-source structural parsing tools or Large Language Models (LLMs) operating without explicit TOC input. To enhance parsing accuracy, we incorporate preprocessing strategies such as OCR-based title detection, XML-derived features, and contextual text features. These strategies are evaluated based on their ability to identify section titles, allocate hierarchy levels, and determine section boundaries. Our findings show that combining LLMs with structure-aware preprocessing substantially reduces false positives and improves extraction quality. We also find that when the metadata quality of headings in the PDF is high, TOC-based techniques perform particularly well. All code and data are publicly available to support replication. We conclude with a comparative evaluation of the methods, outlining their respective strengths and limitations. 

---
# A Survey on Open Dataset Search in the LLM Era: Retrospectives and Perspectives 

**Authors**: Pengyue Li, Sheng Wang, Hua Dai, Zhiyu Chen, Zhifeng Bao, Brian D. Davison  

**Link**: [PDF](https://arxiv.org/pdf/2509.00728)  

**Abstract**: High-quality datasets are typically required for accomplishing data-driven tasks, such as training medical diagnosis models, predicting real-time traffic conditions, or conducting experiments to validate research hypotheses. Consequently, open dataset search, which aims to ensure the efficient and accurate fulfillment of users' dataset requirements, has emerged as a critical research challenge and has attracted widespread interest. Recent studies have made notable progress in enhancing the flexibility and intelligence of open dataset search, and large language models (LLMs) have demonstrated strong potential in addressing long-standing challenges in this area. Therefore, a systematic and comprehensive review of the open dataset search problem is essential, detailing the current state of research and exploring future directions. In this survey, we focus on recent advances in open dataset search beyond traditional approaches that rely on metadata and keywords. From the perspective of dataset modalities, we place particular emphasis on example-based dataset search, advanced similarity measurement techniques based on dataset content, and efficient search acceleration techniques. In addition, we emphasize the mutually beneficial relationship between LLMs and open dataset search. On the one hand, LLMs help address complex challenges in query understanding, semantic modeling, and interactive guidance within open dataset search. In turn, advances in dataset search can support LLMs by enabling more effective integration into retrieval-augmented generation (RAG) frameworks and data selection processes, thereby enhancing downstream task performance. Finally, we summarize open research problems and outline promising directions for future work. This work aims to offer a structured reference for researchers and practitioners in the field of open dataset search. 

---
# ERank: Fusing Supervised Fine-Tuning and Reinforcement Learning for Effective and Efficient Text Reranking 

**Authors**: Yuzheng Cai, Yanzhao Zhang, Dingkun Long, Mingxin Li, Pengjun Xie, Weiguo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.00520)  

**Abstract**: Text reranking models are a crucial component in modern systems like Retrieval-Augmented Generation, tasked with selecting the most relevant documents prior to generation. However, current Large Language Models (LLMs) powered rerankers often face a fundamental trade-off. On one hand, Supervised Fine-Tuning based pointwise methods that frame relevance as a binary classification task lack the necessary scoring discrimination, particularly for those built on reasoning LLMs. On the other hand, approaches designed for complex reasoning often employ powerful yet inefficient listwise formulations, rendering them impractical for low latency applications. To resolve this dilemma, we introduce ERank, a highly effective and efficient pointwise reranker built from a reasoning LLM that excels across diverse relevance scenarios. We propose a novel two-stage training pipeline that begins with Supervised Fine-Tuning (SFT). In this stage, we move beyond binary labels and train the model generatively to output fine grained integer scores, which significantly enhances relevance discrimination. The model is then further refined using Reinforcement Learning (RL) with a novel, listwise derived reward. This technique instills global ranking awareness into the efficient pointwise architecture. We evaluate the ERank reranker on the BRIGHT, FollowIR, TREC DL, and BEIR benchmarks, demonstrating superior effectiveness and robustness compared to existing approaches. On the reasoning-intensive BRIGHT benchmark, our ERank-4B achieves an nDCG@10 of 38.7, while a larger 32B variant reaches a state of the art nDCG@10 of 40.2. 

---
# Beyond Negative Transfer: Disentangled Preference-Guided Diffusion for Cross-Domain Sequential Recommendation 

**Authors**: Xiaoxin Ye, Chengkai Huang, Hongtao Huang, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.00389)  

**Abstract**: Cross-Domain Sequential Recommendation (CDSR) leverages user behaviors across domains to enhance recommendation quality. However, naive aggregation of sequential signals can introduce conflicting domain-specific preferences, leading to negative transfer. While Sequential Recommendation (SR) already suffers from noisy behaviors such as misclicks and impulsive actions, CDSR further amplifies this issue due to domain heterogeneity arising from diverse item types and user intents. The core challenge is disentangling three intertwined signals: domain-invariant preferences, domain-specific preferences, and noise. Diffusion Models (DMs) offer a generative denoising framework well-suited for disentangling complex user preferences and enhancing robustness to noise. Their iterative refinement process enables gradual denoising, making them effective at capturing subtle preference signals. However, existing applications in recommendation face notable limitations: sequential DMs often conflate shared and domain-specific preferences, while cross-domain collaborative filtering DMs neglect temporal dynamics, limiting their ability to model evolving user preferences. To bridge these gaps, we propose \textbf{DPG-Diff}, a novel Disentangled Preference-Guided Diffusion Model, the first diffusion-based approach tailored for CDSR, to or best knowledge. DPG-Diff decomposes user preferences into domain-invariant and domain-specific components, which jointly guide the reverse diffusion process. This disentangled guidance enables robust cross-domain knowledge transfer, mitigates negative transfer, and filters sequential noise. Extensive experiments on real-world datasets demonstrate that DPG-Diff consistently outperforms state-of-the-art baselines across multiple metrics. 

---
# Algorithm Adaptation Bias in Recommendation System Online Experiments 

**Authors**: Chen Zheng, Zhenyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.00199)  

**Abstract**: Online experiments (A/B tests) are widely regarded as the gold standard for evaluating recommender system variants and guiding launch decisions. However, a variety of biases can distort the results of the experiment and mislead decision-making. An underexplored but critical bias is algorithm adaptation effect. This bias arises from the flywheel dynamics among production models, user data, and training pipelines: new models are evaluated on user data whose distributions are shaped by the incumbent system or tested only in a small treatment group. As a result, the measured effect of a new product change in modeling and user experience in this constrained experimental setting can diverge substantially from its true impact in full deployment. In practice, the experiment results often favor the production variant with large traffic while underestimating the performance of the test variant with small traffic, which leads to missing opportunities to launch a true winning arm or underestimating the impact. This paper aims to raise awareness of algorithm adaptation bias, situate it within the broader landscape of RecSys evaluation biases, and motivate discussion of solutions that span experiment design, measurement, and adjustment. We detail the mechanisms of this bias, present empirical evidence from real-world experiments, and discuss potential methods for a more robust online evaluation. 

---
# Bias Mitigation for AI-Feedback Loops in Recommender Systems: A Systematic Literature Review and Taxonomy 

**Authors**: Theodor Stoecker, Samed Bayer, Ingo Weber  

**Link**: [PDF](https://arxiv.org/pdf/2509.00109)  

**Abstract**: Recommender systems continually retrain on user reactions to their own predictions, creating AI feedback loops that amplify biases and diminish fairness over time. Despite this well-known risk, most bias mitigation techniques are tested only on static splits, so their long-term fairness across multiple retraining rounds remains unclear. We therefore present a systematic literature review of bias mitigation methods that explicitly consider AI feedback loops and are validated in multi-round simulations or live A/B tests. Screening 347 papers yields 24 primary studies published between 2019-2025. Each study is coded on six dimensions: mitigation technique, biases addressed, dynamic testing set-up, evaluation focus, application domain, and ML task, organising them into a reusable taxonomy. The taxonomy offers industry practitioners a quick checklist for selecting robust methods and gives researchers a clear roadmap to the field's most urgent gaps. Examples include the shortage of shared simulators, varying evaluation metrics, and the fact that most studies report either fairness or performance; only six use both. 

---
# Better by Comparison: Retrieval-Augmented Contrastive Reasoning for Automatic Prompt Optimization 

**Authors**: Juhyeon Lee, Wonduk Seo, Hyunjin An, Seunghyun Lee, Yi Bu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02093)  

**Abstract**: Automatic prompt optimization has recently emerged as a strategy for improving the quality of prompts used in Large Language Models (LLMs), with the goal of generating more accurate and useful responses. However, most prior work focuses on direct prompt refinement or model fine-tuning, overlooking the potential of leveraging LLMs' inherent reasoning capability to learn from contrasting examples. In this paper, we present Contrastive Reasoning Prompt Optimization (CRPO), a novel framework that formulates prompt optimization as a retrieval augmented reasoning process. Our approach retrieves top k reference prompts from the HelpSteer2 dataset, an open-source collection annotated for helpfulness, correctness, coherence, complexity, and verbosity, and constructs two complementary optimization paradigms: (1) tiered contrastive reasoning, where the LLM compares high, medium, and low quality prompts to refine its own generation through reflective reasoning, and (2) multi-metric contrastive reasoning, where the LLM analyzes the best prompts along each evaluation dimension and integrates their strengths into an optimized prompt. By explicitly contrasting high and low quality exemplars, CRPO enables the model to deduce why certain prompts succeed while others fail, thereby achieving more robust and interpretable optimization. Experimental results on the HelpSteer2 benchmark demonstrate that CRPO significantly outperforms baselines. Our findings highlight the promise of contrastive, retrieval-augmented reasoning for advancing automatic prompt optimization. 

---
# Abex-rat: Synergizing Abstractive Augmentation and Adversarial Training for Classification of Occupational Accident Reports 

**Authors**: Jian Chen, Jinbao Tian, Yunqi Xu, Zhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.02072)  

**Abstract**: The automatic classification of occupational accident reports is a critical research area for enhancing workplace safety and enabling large-scale risk analysis. However, the severe class imbalance inherent in these real-world datasets often compromises the performance of analytical models, particularly for rare but severe incident types, hindering the development of reliable automated systems. To address this challenge, we propose ABEX-RAT, a novel and efficient framework that synergizes generative data augmentation with robust adversarial training. Our approach first employs a twostep abstractive-expansive (ABEX) pipeline, which leverages a large language model to distill core incident semantics and then uses a generative model to create diverse, highquality synthetic samples for underrepresented classes. Subsequently, a lightweight classifier is trained on the augmented data using a computationally efficient random adversarial training (RAT) protocol, which stochastically applies perturbations to enhance model generalization and robustness without significant overhead. Experimental results on the public OSHA dataset demonstrate that our method achieves new state-of-the-art performance, reaching a macro-F1 score of 90.32% and significantly outperforming previous SOTA and fine-tuned large model baselines. Our work validates that this synergistic strategy is a highly effective and efficient alternative to brute-force fine-tuning for specialized, imbalanced classification tasks. The code is publicly available at:this https URL. 

---
# ABCD-LINK: Annotation Bootstrapping for Cross-Document Fine-Grained Links 

**Authors**: Serwar Basch, Ilia Kuznetsov, Tom Hope, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2509.01387)  

**Abstract**: Understanding fine-grained relations between documents is crucial for many application domains. However, the study of automated assistance is limited by the lack of efficient methods to create training and evaluation datasets of cross-document links. To address this, we introduce a new domain-agnostic framework for selecting a best-performing approach and annotating cross-document links in a new domain from scratch. We first generate and validate semi-synthetic datasets of interconnected documents. This data is used to perform automatic evaluation, producing a shortlist of best-performing linking approaches. These approaches are then used in an extensive human evaluation study, yielding performance estimates on natural text pairs. We apply our framework in two distinct domains -- peer review and news -- and show that combining retrieval models with LLMs achieves 78\% link approval from human raters, more than doubling the precision of strong retrievers alone. Our framework enables systematic study of cross-document understanding across application scenarios, and the resulting novel datasets lay foundation for numerous cross-document tasks like media framing and peer review. We make the code, data, and annotation protocols openly available. 

---
# Question-to-Knowledge: Multi-Agent Generation of Inspectable Facts for Product Mapping 

**Authors**: Wonduk Seo, Taesub Shin, Hyunjin An, Dokyun Kim, Seunghyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.01182)  

**Abstract**: Identifying whether two product listings refer to the same Stock Keeping Unit (SKU) is a persistent challenge in ecommerce, especially when explicit identifiers are missing and product names vary widely across platforms. Rule based heuristics and keyword similarity often misclassify products by overlooking subtle distinctions in brand, specification, or bundle configuration. To overcome these limitations, we propose Question to Knowledge (Q2K), a multi agent framework that leverages Large Language Models (LLMs) for reliable SKU mapping. Q2K integrates: (1) a Reasoning Agent that generates targeted disambiguation questions, (2) a Knowledge Agent that resolves them via focused web searches, and (3) a Deduplication Agent that reuses validated reasoning traces to reduce redundancy and ensure consistency. A human in the loop mechanism further refines uncertain cases. Experiments on real world consumer goods datasets show that Q2K surpasses strong baselines, achieving higher accuracy and robustness in difficult scenarios such as bundle identification and brand origin disambiguation. By reusing retrieved reasoning instead of issuing repeated searches, Q2K balances accuracy with efficiency, offering a scalable and interpretable solution for product integration. 

---
# MatPROV: A Provenance Graph Dataset of Material Synthesis Extracted from Scientific Literature 

**Authors**: Hirofumi Tsuruta, Masaya Kumagai  

**Link**: [PDF](https://arxiv.org/pdf/2509.01042)  

**Abstract**: Synthesis procedures play a critical role in materials research, as they directly affect material properties. With data-driven approaches increasingly accelerating materials discovery, there is growing interest in extracting synthesis procedures from scientific literature as structured data. However, existing studies often rely on rigid, domain-specific schemas with predefined fields for structuring synthesis procedures or assume that synthesis procedures are linear sequences of operations, which limits their ability to capture the structural complexity of real-world procedures. To address these limitations, we adopt PROV-DM, an international standard for provenance information, which supports flexible, graph-based modeling of procedures. We present MatPROV, a dataset of PROV-DM-compliant synthesis procedures extracted from scientific literature using large language models. MatPROV captures structural complexities and causal relationships among materials, operations, and conditions through visually intuitive directed graphs. This representation enables machine-interpretable synthesis knowledge, opening opportunities for future research such as automated synthesis planning and optimization. 

---
# TMT: A Simple Way to Translate Topic Models Using Dictionaries 

**Authors**: Felix Engl, Andreas Henrich  

**Link**: [PDF](https://arxiv.org/pdf/2509.00822)  

**Abstract**: The training of topic models for a multilingual environment is a challenging task, requiring the use of sophisticated algorithms, topic-aligned corpora, and manual evaluation. These difficulties are further exacerbated when the developer lacks knowledge of the target language or is working in an environment with limited data, where only small or unusable multilingual corpora are available.
Considering these challenges, we introduce Topic Model Translation (TMT), a novel, robust and transparent technique designed to transfer topic models (e.g., Latent Dirichlet Allocation (LDA) based topic models) from one language to another, without the need for metadata, embeddings, or aligned corpora. TMT enables the reuse of topic models across languages, making it especially suitable for scenarios where large corpora in the target language are unavailable or manual translation is infeasible. Furthermore, we evaluate TMT extensively using both quantitative and qualitative methods, demonstrating that it produces semantically coherent and consistent topic translations. 

---
# Do small language models generate realistic variable-quality fake news headlines? 

**Authors**: Austin McCutcheon, Chris Brogly  

**Link**: [PDF](https://arxiv.org/pdf/2509.00680)  

**Abstract**: Small language models (SLMs) have the capability for text generation and may potentially be used to generate falsified texts online. This study evaluates 14 SLMs (1.7B-14B parameters) including LLaMA, Gemma, Phi, SmolLM, Mistral, and Granite families in generating perceived low and high quality fake news headlines when explicitly prompted, and whether they appear to be similar to real-world news headlines. Using controlled prompt engineering, 24,000 headlines were generated across low-quality and high-quality deceptive categories. Existing machine learning and deep learning-based news headline quality detectors were then applied against these SLM-generated fake news headlines. SLMs demonstrated high compliance rates with minimal ethical resistance, though there were some occasional exceptions. Headline quality detection using established DistilBERT and bagging classifier models showed that quality misclassification was common, with detection accuracies only ranging from 35.2% to 63.5%. These findings suggest the following: tested SLMs generally are compliant in generating falsified headlines, although there are slight variations in ethical restraints, and the generated headlines did not closely resemble existing primarily human-written content on the web, given the low quality classification accuracy. 

---
# Confident, Calibrated, or Complicit: Probing the Trade-offs between Safety Alignment and Ideological Bias in Language Models in Detecting Hate Speech 

**Authors**: Sanjeeevan Selvaganapathy, Mehwish Nasim  

**Link**: [PDF](https://arxiv.org/pdf/2509.00673)  

**Abstract**: We investigate the efficacy of Large Language Models (LLMs) in detecting implicit and explicit hate speech, examining whether models with minimal safety alignment (uncensored) might provide more objective classification capabilities compared to their heavily-aligned (censored) counterparts. While uncensored models theoretically offer a less constrained perspective free from moral guardrails that could bias classification decisions, our results reveal a surprising trade-off: censored models significantly outperform their uncensored counterparts in both accuracy and robustness, achieving 78.7% versus 64.1% strict accuracy. However, this enhanced performance comes with its own limitation -- the safety alignment acts as a strong ideological anchor, making censored models resistant to persona-based influence, while uncensored models prove highly malleable to ideological framing. Furthermore, we identify critical failures across all models in understanding nuanced language such as irony. We also find alarming fairness disparities in performance across different targeted groups and systemic overconfidence that renders self-reported certainty unreliable. These findings challenge the notion of LLMs as objective arbiters and highlight the need for more sophisticated auditing frameworks that account for fairness, calibration, and ideological consistency. 

---
# BALM-TSF: Balanced Multimodal Alignment for LLM-Based Time Series Forecasting 

**Authors**: Shiqiao Zhou, Holger Schöner, Huanbo Lyu, Edouard Fouché, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00622)  

**Abstract**: Time series forecasting is a long-standing and highly challenging research topic. Recently, driven by the rise of large language models (LLMs), research has increasingly shifted from purely time series methods toward harnessing textual modalities to enhance forecasting performance. However, the vast discrepancy between text and temporal data often leads current multimodal architectures to over-emphasise one modality while neglecting the other, resulting in information loss that harms forecasting performance. To address this modality imbalance, we introduce BALM-TSF (Balanced Multimodal Alignment for LLM-Based Time Series Forecasting), a lightweight time series forecasting framework that maintains balance between the two modalities. Specifically, raw time series are processed by the time series encoder, while descriptive statistics of raw time series are fed to an LLM with learnable prompt, producing compact textual embeddings. To ensure balanced cross-modal context alignment of time series and textual embeddings, a simple yet effective scaling strategy combined with a contrastive objective then maps these textual embeddings into the latent space of the time series embeddings. Finally, the aligned textual semantic embeddings and time series embeddings are together integrated for forecasting. Extensive experiments on standard benchmarks show that, with minimal trainable parameters, BALM-TSF achieves state-of-the-art performance in both long-term and few-shot forecasting, confirming its ability to harness complementary information from text and time series. Code is available at this https URL. 

---
# How to Make Museums More Interactive? Case Study of Artistic Chatbot 

**Authors**: Filip J. Kucia, Bartosz Grabek, Szymon D. Trochimiak, Anna Wróblewska  

**Link**: [PDF](https://arxiv.org/pdf/2509.00572)  

**Abstract**: Conversational agents powered by Large Language Models (LLMs) are increasingly utilized in educational settings, in particular in individual closed digital environments, yet their potential adoption in the physical learning environments like cultural heritage sites, museums, and art galleries remains relatively unexplored. In this study, we present Artistic Chatbot, a voice-to-voice RAG-powered chat system to support informal learning and enhance visitor engagement during a live art exhibition celebrating the 15th anniversary of the Faculty of Media Art at the Warsaw Academy of Fine Arts, Poland. The question answering (QA) chatbot responded to free-form spoken questions in Polish using the context retrieved from a curated, domain-specific knowledge base consisting of 226 documents provided by the organizers, including faculty information, art magazines, books, and journals. We describe the key aspects of the system architecture and user interaction design, as well as discuss the practical challenges associated with deploying chatbots at public cultural sites. Our findings, based on interaction analysis, demonstrate that chatbots such as Artistic Chatbot effectively maintain responses grounded in exhibition content (60\% of responses directly relevant), even when faced with unpredictable queries outside the target domain, showing their potential for increasing interactivity in public cultural sites.
GitHub project page: this https URL 

---
# MedSEBA: Synthesizing Evidence-Based Answers Grounded in Evolving Medical Literature 

**Authors**: Juraj Vladika, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2509.00414)  

**Abstract**: In the digital age, people often turn to the Internet in search of medical advice and recommendations. With the increasing volume of online content, it has become difficult to distinguish reliable sources from misleading information. Similarly, millions of medical studies are published every year, making it challenging for researchers to keep track of the latest scientific findings. These evolving studies can reach differing conclusions, which is not reflected in traditional search tools. To address these challenges, we introduce MedSEBA, an interactive AI-powered system for synthesizing evidence-based answers to medical questions. It utilizes the power of Large Language Models to generate coherent and expressive answers, but grounds them in trustworthy medical studies dynamically retrieved from the research database PubMed. The answers consist of key points and arguments, which can be traced back to respective studies. Notably, the platform also provides an overview of the extent to which the most relevant studies support or refute the given medical claim, and a visualization of how the research consensus evolved through time. Our user study revealed that medical experts and lay users find the system usable and helpful, and the provided answers trustworthy and informative. This makes the system well-suited for both everyday health questions and advanced research insights. 

---
# CRouting: Reducing Expensive Distance Calls in Graph-Based Approximate Nearest Neighbor Search 

**Authors**: Zhenxin Li, Shuibing He, Jiahao Guo, Xuechen Zhang, Xian-He Sun, Gang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.00365)  

**Abstract**: Approximate nearest neighbor search (ANNS) is a crucial problem in information retrieval and AI applications. Recently, there has been a surge of interest in graph-based ANNS algorithms due to their superior efficiency and accuracy. However, the repeated computation of distances in high-dimensional spaces constitutes the primary time cost of graph-based methods. To accelerate the search, we propose a novel routing strategy named CRouting, which bypasses unnecessary distance computations by exploiting the angle distributions of high-dimensional vectors. CRouting is designed as a plugin to optimize existing graph-based search with minimal code modifications. Our experiments show that CRouting reduces the number of distance computations by up to 41.5% and boosts queries per second by up to 1.48$\times$ on two predominant graph indexes, HNSW and NSG. Code is publicly available at this https URL. 

---
# GIER: Gap-Driven Self-Refinement for Large Language Models 

**Authors**: Rinku Dewri  

**Link**: [PDF](https://arxiv.org/pdf/2509.00325)  

**Abstract**: We introduce GIER (Gap-driven Iterative Enhancement of Responses), a general framework for improving large language model (LLM) outputs through self-reflection and revision based on conceptual quality criteria. Unlike prompting strategies that rely on demonstrations, examples, or chain-of-thought templates, GIER utilizes natural language descriptions of reasoning gaps, and prompts a model to iteratively critique and refine its own outputs to better satisfy these criteria. Across three reasoning-intensive tasks (SciFact, PrivacyQA, and e-SNLI) and four LLMs (GPT-4.1, GPT-4o Mini, Gemini 1.5 Pro, and Llama 3.3 70B), GIER improves rationale quality, grounding, and reasoning alignment without degrading task accuracy. Our analysis demonstrates that models can not only interpret abstract conceptual gaps but also translate them into concrete reasoning improvements. 

---
# Access Paths for Efficient Ordering with Large Language Models 

**Authors**: Fuheng Zhao, Jiayue Chen, Yiming Pan, Tahseen Rabbani, Divyakant Agrawal, Amr El Abbadi  

**Link**: [PDF](https://arxiv.org/pdf/2509.00303)  

**Abstract**: We present the LLM ORDER BY operator as a logical abstraction and study its physical implementations within a unified evaluation framework. Our experiments show that no single approach is universally optimal, with effectiveness depending on query characteristics and data. We introduce three new designs: an agreement-based batch-size policy, a majority voting mechanism for pairwise sorting, and a two-way external merge sort adapted for LLMs. With extensive experiments, our agreement-based procedure is effective at determining batch size for value-based methods, the majority-voting mechanism consistently strengthens pairwise comparisons on GPT-4o, and external merge sort achieves high accuracy-efficiency trade-offs across datasets and models. We further observe a log-linear scaling between compute cost and ordering quality, offering the first step toward principled cost models for LLM powered data systems. 

---
# OpinioRAG: Towards Generating User-Centric Opinion Highlights from Large-scale Online Reviews 

**Authors**: Mir Tafseer Nayeem, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2509.00285)  

**Abstract**: We study the problem of opinion highlights generation from large volumes of user reviews, often exceeding thousands per entity, where existing methods either fail to scale or produce generic, one-size-fits-all summaries that overlook personalized needs. To tackle this, we introduce OpinioRAG, a scalable, training-free framework that combines RAG-based evidence retrieval with LLMs to efficiently produce tailored summaries. Additionally, we propose novel reference-free verification metrics designed for sentiment-rich domains, where accurately capturing opinions and sentiment alignment is essential. These metrics offer a fine-grained, context-sensitive assessment of factual consistency. To facilitate evaluation, we contribute the first large-scale dataset of long-form user reviews, comprising entities with over a thousand reviews each, paired with unbiased expert summaries and manually annotated queries. Through extensive experiments, we identify key challenges, provide actionable insights into improving systems, pave the way for future research, and position OpinioRAG as a robust framework for generating accurate, relevant, and structured summaries at scale. 

---
