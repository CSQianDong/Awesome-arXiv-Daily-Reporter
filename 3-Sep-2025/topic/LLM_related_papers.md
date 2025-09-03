# Application Of Large Language Models For The Extraction Of Information From Particle Accelerator Technical Documentation 

**Authors**: Qing Dai, Rasmus Ischebeck, Maruisz Sapinski, Adam Grycner  

**Link**: [PDF](https://arxiv.org/pdf/2509.02227)  

**Abstract**: The large set of technical documentation of legacy accelerator systems, coupled with the retirement of experienced personnel, underscores the urgent need for efficient methods to preserve and transfer specialized knowledge. This paper explores the application of large language models (LLMs), to automate and enhance the extraction of information from particle accelerator technical documents. By exploiting LLMs, we aim to address the challenges of knowledge retention, enabling the retrieval of domain expertise embedded in legacy documentation. We present initial results of adapting LLMs to this specialized domain. Our evaluation demonstrates the effectiveness of LLMs in extracting, summarizing, and organizing knowledge, significantly reducing the risk of losing valuable insights as personnel retire. Furthermore, we discuss the limitations of current LLMs, such as interpretability and handling of rare domain-specific terms, and propose strategies for improvement. This work highlights the potential of LLMs to play a pivotal role in preserving institutional knowledge and ensuring continuity in highly specialized fields. 

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
# Empowering Large Language Model for Sequential Recommendation via Multimodal Embeddings and Semantic IDs 

**Authors**: Yuhao Wang, Junwei Pan, Xinhang Li, Maolin Wang, Yuan Wang, Yue Liu, Dapeng Liu, Jie Jiang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.02017)  

**Abstract**: Sequential recommendation (SR) aims to capture users' dynamic interests and sequential patterns based on their historical interactions. Recently, the powerful capabilities of large language models (LLMs) have driven their adoption in SR. However, we identify two critical challenges in existing LLM-based SR methods: 1) embedding collapse when incorporating pre-trained collaborative embeddings and 2) catastrophic forgetting of quantized embeddings when utilizing semantic IDs. These issues dampen the model scalability and lead to suboptimal recommendation performance. Therefore, based on LLMs like Llama3-8B-instruct, we introduce a novel SR framework named MME-SID, which integrates multimodal embeddings and quantized embeddings to mitigate embedding collapse. Additionally, we propose a Multimodal Residual Quantized Variational Autoencoder (MM-RQ-VAE) with maximum mean discrepancy as the reconstruction loss and contrastive learning for alignment, which effectively preserve intra-modal distance information and capture inter-modal correlations, respectively. To further alleviate catastrophic forgetting, we initialize the model with the trained multimodal code embeddings. Finally, we fine-tune the LLM efficiently using LoRA in a multimodal frequency-aware fusion manner. Extensive experiments on three public datasets validate the superior performance of MME-SID thanks to its capability to mitigate embedding collapse and catastrophic forgetting. The implementation code and datasets are publicly available for reproduction: this https URL. 

---
# A Survey on Open Dataset Search in the LLM Era: Retrospectives and Perspectives 

**Authors**: Pengyue Li, Sheng Wang, Hua Dai, Zhiyu Chen, Zhifeng Bao, Brian D. Davison  

**Link**: [PDF](https://arxiv.org/pdf/2509.00728)  

**Abstract**: High-quality datasets are typically required for accomplishing data-driven tasks, such as training medical diagnosis models, predicting real-time traffic conditions, or conducting experiments to validate research hypotheses. Consequently, open dataset search, which aims to ensure the efficient and accurate fulfillment of users' dataset requirements, has emerged as a critical research challenge and has attracted widespread interest. Recent studies have made notable progress in enhancing the flexibility and intelligence of open dataset search, and large language models (LLMs) have demonstrated strong potential in addressing long-standing challenges in this area. Therefore, a systematic and comprehensive review of the open dataset search problem is essential, detailing the current state of research and exploring future directions. In this survey, we focus on recent advances in open dataset search beyond traditional approaches that rely on metadata and keywords. From the perspective of dataset modalities, we place particular emphasis on example-based dataset search, advanced similarity measurement techniques based on dataset content, and efficient search acceleration techniques. In addition, we emphasize the mutually beneficial relationship between LLMs and open dataset search. On the one hand, LLMs help address complex challenges in query understanding, semantic modeling, and interactive guidance within open dataset search. In turn, advances in dataset search can support LLMs by enabling more effective integration into retrieval-augmented generation (RAG) frameworks and data selection processes, thereby enhancing downstream task performance. Finally, we summarize open research problems and outline promising directions for future work. This work aims to offer a structured reference for researchers and practitioners in the field of open dataset search. 

---
# Lighting the Way for BRIGHT: Reproducible Baselines with Anserini, Pyserini, and RankLLM 

**Authors**: Yijun Ge, Sahel Sharifymoghaddam, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.02558)  

**Abstract**: The BRIGHT benchmark is a dataset consisting of reasoning-intensive queries over diverse domains. We explore retrieval results on BRIGHT using a range of retrieval techniques, including sparse, dense, and fusion methods, and establish reproducible baselines. We then apply listwise reranking with large language models (LLMs) to further investigate the impact of reranking on reasoning-intensive queries. These baselines are integrated into popular retrieval and reranking toolkits Anserini, Pyserini, and RankLLM, with two-click reproducibility that makes them easy to build upon and convenient for further development. While attempting to reproduce the results reported in the original BRIGHT paper, we find that the provided BM25 scores differ notably from those that we obtain using Anserini and Pyserini. We discover that this difference is due to BRIGHT's implementation of BM25, which applies BM25 on the query rather than using the standard bag-of-words approach, as in Anserini, to construct query vectors. This difference has become increasingly relevant due to the rise of longer queries, with BRIGHT's lengthy reasoning-intensive queries being a prime example, and further accentuated by the increasing usage of retrieval-augmented generation, where LLM prompts can grow to be much longer than ''traditional'' search engine queries. Our observation signifies that it may be time to reconsider BM25 approaches going forward in order to better accommodate emerging applications. To facilitate this, we integrate query-side BM25 into both Anserini and Pyserini. 

---
# Better by Comparison: Retrieval-Augmented Contrastive Reasoning for Automatic Prompt Optimization 

**Authors**: Juhyeon Lee, Wonduk Seo, Hyunjin An, Seunghyun Lee, Yi Bu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02093)  

**Abstract**: Automatic prompt optimization has recently emerged as a strategy for improving the quality of prompts used in Large Language Models (LLMs), with the goal of generating more accurate and useful responses. However, most prior work focuses on direct prompt refinement or model fine-tuning, overlooking the potential of leveraging LLMs' inherent reasoning capability to learn from contrasting examples. In this paper, we present Contrastive Reasoning Prompt Optimization (CRPO), a novel framework that formulates prompt optimization as a retrieval augmented reasoning process. Our approach retrieves top k reference prompts from the HelpSteer2 dataset, an open-source collection annotated for helpfulness, correctness, coherence, complexity, and verbosity, and constructs two complementary optimization paradigms: (1) tiered contrastive reasoning, where the LLM compares high, medium, and low quality prompts to refine its own generation through reflective reasoning, and (2) multi-metric contrastive reasoning, where the LLM analyzes the best prompts along each evaluation dimension and integrates their strengths into an optimized prompt. By explicitly contrasting high and low quality exemplars, CRPO enables the model to deduce why certain prompts succeed while others fail, thereby achieving more robust and interpretable optimization. Experimental results on the HelpSteer2 benchmark demonstrate that CRPO significantly outperforms baselines. Our findings highlight the promise of contrastive, retrieval-augmented reasoning for advancing automatic prompt optimization. 

---
# HiPS: Hierarchical PDF Segmentation of Textbooks 

**Authors**: Sabine Wehnert, Harikrishnan Changaramkulath, Ernesto William De Luca  

**Link**: [PDF](https://arxiv.org/pdf/2509.00909)  

**Abstract**: The growing demand for effective tools to parse PDF-formatted texts, particularly structured documents such as textbooks, reveals the limitations of current methods developed mainly for research paper segmentation. This work addresses the challenge of hierarchical segmentation in complex structured documents, with a focus on legal textbooks that contain layered knowledge essential for interpreting and applying legal norms. We examine a Table of Contents (TOC)-based technique and approaches that rely on open-source structural parsing tools or Large Language Models (LLMs) operating without explicit TOC input. To enhance parsing accuracy, we incorporate preprocessing strategies such as OCR-based title detection, XML-derived features, and contextual text features. These strategies are evaluated based on their ability to identify section titles, allocate hierarchy levels, and determine section boundaries. Our findings show that combining LLMs with structure-aware preprocessing substantially reduces false positives and improves extraction quality. We also find that when the metadata quality of headings in the PDF is high, TOC-based techniques perform particularly well. All code and data are publicly available to support replication. We conclude with a comparative evaluation of the methods, outlining their respective strengths and limitations. 

---
# Question-to-Knowledge: Multi-Agent Generation of Inspectable Facts for Product Mapping 

**Authors**: Wonduk Seo, Taesub Shin, Hyunjin An, Dokyun Kim, Seunghyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.01182)  

**Abstract**: Identifying whether two product listings refer to the same Stock Keeping Unit (SKU) is a persistent challenge in ecommerce, especially when explicit identifiers are missing and product names vary widely across platforms. Rule based heuristics and keyword similarity often misclassify products by overlooking subtle distinctions in brand, specification, or bundle configuration. To overcome these limitations, we propose Question to Knowledge (Q2K), a multi agent framework that leverages Large Language Models (LLMs) for reliable SKU mapping. Q2K integrates: (1) a Reasoning Agent that generates targeted disambiguation questions, (2) a Knowledge Agent that resolves them via focused web searches, and (3) a Deduplication Agent that reuses validated reasoning traces to reduce redundancy and ensure consistency. A human in the loop mechanism further refines uncertain cases. Experiments on real world consumer goods datasets show that Q2K surpasses strong baselines, achieving higher accuracy and robustness in difficult scenarios such as bundle identification and brand origin disambiguation. By reusing retrieved reasoning instead of issuing repeated searches, Q2K balances accuracy with efficiency, offering a scalable and interpretable solution for product integration. 

---
# Upcycling Candidate Tokens of Large Language Models for Query Expansion 

**Authors**: Jinseok Kim, Sukmin Cho, Soyeong Jeong, Sangyeop Kim, Sungzoon Cho  

**Link**: [PDF](https://arxiv.org/pdf/2509.02377)  

**Abstract**: Query Expansion (QE) improves retrieval performance by enriching queries with related terms. Recently, Large Language Models (LLMs) have been used for QE, but existing methods face a trade-off: generating diverse terms boosts performance but increases computational cost. To address this challenge, we propose Candidate Token Query Expansion (CTQE), which extracts diverse and relevant terms from a single LLM decoding pass by leveraging unselected candidate tokens. These tokens, though not part of the final output, are conditioned on the full query and capture useful information. By aggregating them, CTQE achieves both relevance and diversity without extra inference, reducing overhead and latency. Experiments show that CTQE delivers strong retrieval performance with significantly lower cost, outperforming or comparable to more expensive methods. Code is available at: this https URL 

---
# ERank: Fusing Supervised Fine-Tuning and Reinforcement Learning for Effective and Efficient Text Reranking 

**Authors**: Yuzheng Cai, Yanzhao Zhang, Dingkun Long, Mingxin Li, Pengjun Xie, Weiguo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.00520)  

**Abstract**: Text reranking models are a crucial component in modern systems like Retrieval-Augmented Generation, tasked with selecting the most relevant documents prior to generation. However, current Large Language Models (LLMs) powered rerankers often face a fundamental trade-off. On one hand, Supervised Fine-Tuning based pointwise methods that frame relevance as a binary classification task lack the necessary scoring discrimination, particularly for those built on reasoning LLMs. On the other hand, approaches designed for complex reasoning often employ powerful yet inefficient listwise formulations, rendering them impractical for low latency applications. To resolve this dilemma, we introduce ERank, a highly effective and efficient pointwise reranker built from a reasoning LLM that excels across diverse relevance scenarios. We propose a novel two-stage training pipeline that begins with Supervised Fine-Tuning (SFT). In this stage, we move beyond binary labels and train the model generatively to output fine grained integer scores, which significantly enhances relevance discrimination. The model is then further refined using Reinforcement Learning (RL) with a novel, listwise derived reward. This technique instills global ranking awareness into the efficient pointwise architecture. We evaluate the ERank reranker on the BRIGHT, FollowIR, TREC DL, and BEIR benchmarks, demonstrating superior effectiveness and robustness compared to existing approaches. On the reasoning-intensive BRIGHT benchmark, our ERank-4B achieves an nDCG@10 of 38.7, while a larger 32B variant reaches a state of the art nDCG@10 of 40.2. 

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
# GIER: Gap-Driven Self-Refinement for Large Language Models 

**Authors**: Rinku Dewri  

**Link**: [PDF](https://arxiv.org/pdf/2509.00325)  

**Abstract**: We introduce GIER (Gap-driven Iterative Enhancement of Responses), a general framework for improving large language model (LLM) outputs through self-reflection and revision based on conceptual quality criteria. Unlike prompting strategies that rely on demonstrations, examples, or chain-of-thought templates, GIER utilizes natural language descriptions of reasoning gaps, and prompts a model to iteratively critique and refine its own outputs to better satisfy these criteria. Across three reasoning-intensive tasks (SciFact, PrivacyQA, and e-SNLI) and four LLMs (GPT-4.1, GPT-4o Mini, Gemini 1.5 Pro, and Llama 3.3 70B), GIER improves rationale quality, grounding, and reasoning alignment without degrading task accuracy. Our analysis demonstrates that models can not only interpret abstract conceptual gaps but also translate them into concrete reasoning improvements. 

---
# Confident, Calibrated, or Complicit: Probing the Trade-offs between Safety Alignment and Ideological Bias in Language Models in Detecting Hate Speech 

**Authors**: Sanjeeevan Selvaganapathy, Mehwish Nasim  

**Link**: [PDF](https://arxiv.org/pdf/2509.00673)  

**Abstract**: We investigate the efficacy of Large Language Models (LLMs) in detecting implicit and explicit hate speech, examining whether models with minimal safety alignment (uncensored) might provide more objective classification capabilities compared to their heavily-aligned (censored) counterparts. While uncensored models theoretically offer a less constrained perspective free from moral guardrails that could bias classification decisions, our results reveal a surprising trade-off: censored models significantly outperform their uncensored counterparts in both accuracy and robustness, achieving 78.7% versus 64.1% strict accuracy. However, this enhanced performance comes with its own limitation -- the safety alignment acts as a strong ideological anchor, making censored models resistant to persona-based influence, while uncensored models prove highly malleable to ideological framing. Furthermore, we identify critical failures across all models in understanding nuanced language such as irony. We also find alarming fairness disparities in performance across different targeted groups and systemic overconfidence that renders self-reported certainty unreliable. These findings challenge the notion of LLMs as objective arbiters and highlight the need for more sophisticated auditing frameworks that account for fairness, calibration, and ideological consistency. 

---
# Access Paths for Efficient Ordering with Large Language Models 

**Authors**: Fuheng Zhao, Jiayue Chen, Yiming Pan, Tahseen Rabbani, Divyakant Agrawal, Amr El Abbadi  

**Link**: [PDF](https://arxiv.org/pdf/2509.00303)  

**Abstract**: We present the LLM ORDER BY operator as a logical abstraction and study its physical implementations within a unified evaluation framework. Our experiments show that no single approach is universally optimal, with effectiveness depending on query characteristics and data. We introduce three new designs: an agreement-based batch-size policy, a majority voting mechanism for pairwise sorting, and a two-way external merge sort adapted for LLMs. With extensive experiments, our agreement-based procedure is effective at determining batch size for value-based methods, the majority-voting mechanism consistently strengthens pairwise comparisons on GPT-4o, and external merge sort achieves high accuracy-efficiency trade-offs across datasets and models. We further observe a log-linear scaling between compute cost and ordering quality, offering the first step toward principled cost models for LLM powered data systems. 

---
# Abex-rat: Synergizing Abstractive Augmentation and Adversarial Training for Classification of Occupational Accident Reports 

**Authors**: Jian Chen, Jinbao Tian, Yunqi Xu, Zhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.02072)  

**Abstract**: The automatic classification of occupational accident reports is a critical research area for enhancing workplace safety and enabling large-scale risk analysis. However, the severe class imbalance inherent in these real-world datasets often compromises the performance of analytical models, particularly for rare but severe incident types, hindering the development of reliable automated systems. To address this challenge, we propose ABEX-RAT, a novel and efficient framework that synergizes generative data augmentation with robust adversarial training. Our approach first employs a twostep abstractive-expansive (ABEX) pipeline, which leverages a large language model to distill core incident semantics and then uses a generative model to create diverse, highquality synthetic samples for underrepresented classes. Subsequently, a lightweight classifier is trained on the augmented data using a computationally efficient random adversarial training (RAT) protocol, which stochastically applies perturbations to enhance model generalization and robustness without significant overhead. Experimental results on the public OSHA dataset demonstrate that our method achieves new state-of-the-art performance, reaching a macro-F1 score of 90.32% and significantly outperforming previous SOTA and fine-tuned large model baselines. Our work validates that this synergistic strategy is a highly effective and efficient alternative to brute-force fine-tuning for specialized, imbalanced classification tasks. The code is publicly available at:this https URL. 

---
# MatPROV: A Provenance Graph Dataset of Material Synthesis Extracted from Scientific Literature 

**Authors**: Hirofumi Tsuruta, Masaya Kumagai  

**Link**: [PDF](https://arxiv.org/pdf/2509.01042)  

**Abstract**: Synthesis procedures play a critical role in materials research, as they directly affect material properties. With data-driven approaches increasingly accelerating materials discovery, there is growing interest in extracting synthesis procedures from scientific literature as structured data. However, existing studies often rely on rigid, domain-specific schemas with predefined fields for structuring synthesis procedures or assume that synthesis procedures are linear sequences of operations, which limits their ability to capture the structural complexity of real-world procedures. To address these limitations, we adopt PROV-DM, an international standard for provenance information, which supports flexible, graph-based modeling of procedures. We present MatPROV, a dataset of PROV-DM-compliant synthesis procedures extracted from scientific literature using large language models. MatPROV captures structural complexities and causal relationships among materials, operations, and conditions through visually intuitive directed graphs. This representation enables machine-interpretable synthesis knowledge, opening opportunities for future research such as automated synthesis planning and optimization. 

---
# BALM-TSF: Balanced Multimodal Alignment for LLM-Based Time Series Forecasting 

**Authors**: Shiqiao Zhou, Holger Schöner, Huanbo Lyu, Edouard Fouché, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00622)  

**Abstract**: Time series forecasting is a long-standing and highly challenging research topic. Recently, driven by the rise of large language models (LLMs), research has increasingly shifted from purely time series methods toward harnessing textual modalities to enhance forecasting performance. However, the vast discrepancy between text and temporal data often leads current multimodal architectures to over-emphasise one modality while neglecting the other, resulting in information loss that harms forecasting performance. To address this modality imbalance, we introduce BALM-TSF (Balanced Multimodal Alignment for LLM-Based Time Series Forecasting), a lightweight time series forecasting framework that maintains balance between the two modalities. Specifically, raw time series are processed by the time series encoder, while descriptive statistics of raw time series are fed to an LLM with learnable prompt, producing compact textual embeddings. To ensure balanced cross-modal context alignment of time series and textual embeddings, a simple yet effective scaling strategy combined with a contrastive objective then maps these textual embeddings into the latent space of the time series embeddings. Finally, the aligned textual semantic embeddings and time series embeddings are together integrated for forecasting. Extensive experiments on standard benchmarks show that, with minimal trainable parameters, BALM-TSF achieves state-of-the-art performance in both long-term and few-shot forecasting, confirming its ability to harness complementary information from text and time series. Code is available at this https URL. 

---
# Jointly Reinforcing Diversity and Quality in Language Model Generations 

**Authors**: Tianjian Li, Yiming Zhang, Ping Yu, Swarnadeep Saha, Daniel Khashabi, Jason Weston, Jack Lanchantin, Tianlu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02534)  

**Abstract**: Post-training of Large Language Models (LMs) often prioritizes accuracy and helpfulness at the expense of diversity. This creates a tension: while post-training improves response quality, it also sharpens output distributions and reduces the range of ideas, limiting the usefulness of LMs in creative and exploratory tasks such as brainstorming, storytelling, or problem solving. We address this challenge with Diversity-Aware Reinforcement Learning (DARLING), a framework that jointly optimizes for response quality and semantic diversity. At its core, DARLING introduces a learned partition function to measure diversity beyond surface-level lexical variations. This diversity signal is then combined with a quality reward during online reinforcement learning, encouraging models to generate outputs that are both high-quality and distinct. Experiments across multiple model families and sizes show that DARLING generalizes to two regimes: non-verifiable tasks (instruction following and creative writing) and verifiable tasks (competition math). On five benchmarks in the first setting, DARLING consistently outperforms quality-only RL baselines, producing outputs that are simultaneously of higher quality and novelty. In the second setting, DARLING achieves higher pass@1 (solution quality) and pass@k (solution variety). Most strikingly, explicitly optimizing for diversity catalyzes exploration in online RL, which manifests itself as higher-quality responses. 

---
# PalmX 2025: The First Shared Task on Benchmarking LLMs on Arabic and Islamic Culture 

**Authors**: Fakhraddin Alwajih, Abdellah El Mekki, Hamdy Mubarak, Majd Hawasly, Abubakr Mohamed, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2509.02550)  

**Abstract**: Large Language Models (LLMs) inherently reflect the vast data distributions they encounter during their pre-training phase. As this data is predominantly sourced from the web, there is a high chance it will be skewed towards high-resourced languages and cultures, such as those of the West. Consequently, LLMs often exhibit a diminished understanding of certain communities, a gap that is particularly evident in their knowledge of Arabic and Islamic cultures. This issue becomes even more pronounced with increasingly under-represented topics. To address this critical challenge, we introduce PalmX 2025, the first shared task designed to benchmark the cultural competence of LLMs in these specific domains. The task is composed of two subtasks featuring multiple-choice questions (MCQs) in Modern Standard Arabic (MSA): General Arabic Culture and General Islamic Culture. These subtasks cover a wide range of topics, including traditions, food, history, religious practices, and language expressions from across 22 Arab countries. The initiative drew considerable interest, with 26 teams registering for Subtask 1 and 19 for Subtask 2, culminating in nine and six valid submissions, respectively. Our findings reveal that task-specific fine-tuning substantially boosts performance over baseline models. The top-performing systems achieved an accuracy of 72.15% on cultural questions and 84.22% on Islamic knowledge. Parameter-efficient fine-tuning emerged as the predominant and most effective approach among participants, while the utility of data augmentation was found to be domain-dependent. 

---
# Do LLMs Adhere to Label Definitions? Examining Their Receptivity to External Label Definitions 

**Authors**: Seyedali Mohammadi, Bhaskara Hanuma Vedula, Hemank Lamba, Edward Raff, Ponnurangam Kumaraguru, Francis Ferraro, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2509.02452)  

**Abstract**: Do LLMs genuinely incorporate external definitions, or do they primarily rely on their parametric knowledge? To address these questions, we conduct controlled experiments across multiple explanation benchmark datasets (general and domain-specific) and label definition conditions, including expert-curated, LLM-generated, perturbed, and swapped definitions. Our results reveal that while explicit label definitions can enhance accuracy and explainability, their integration into an LLM's task-solving processes is neither guaranteed nor consistent, suggesting reliance on internalized representations in many cases. Models often default to their internal representations, particularly in general tasks, whereas domain-specific tasks benefit more from explicit definitions. These findings underscore the need for a deeper understanding of how LLMs process external knowledge alongside their pre-existing capabilities. 

---
# Towards Temporal Knowledge-Base Creation for Fine-Grained Opinion Analysis with Language Models 

**Authors**: Gaurav Negi, Atul Kr. Ojha, Omnia Zayed, Paul Buitelaar  

**Link**: [PDF](https://arxiv.org/pdf/2509.02363)  

**Abstract**: We propose a scalable method for constructing a temporal opinion knowledge base with large language models (LLMs) as automated annotators. Despite the demonstrated utility of time-series opinion analysis of text for downstream applications such as forecasting and trend analysis, existing methodologies underexploit this potential due to the absence of temporally grounded fine-grained annotations. Our approach addresses this gap by integrating well-established opinion mining formulations into a declarative LLM annotation pipeline, enabling structured opinion extraction without manual prompt engineering. We define three data models grounded in sentiment and opinion mining literature, serving as schemas for structured representation. We perform rigorous quantitative evaluation of our pipeline using human-annotated test samples. We carry out the final annotations using two separate LLMs, and inter-annotator agreement is computed label-wise across the fine-grained opinion dimensions, analogous to human annotation protocols. The resulting knowledge base encapsulates time-aligned, structured opinions and is compatible with applications in Retrieval-Augmented Generation (RAG), temporal question answering, and timeline summarisation. 

---
# LLMs and their Limited Theory of Mind: Evaluating Mental State Annotations in Situated Dialogue 

**Authors**: Katharine Kowalyshyn, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2509.02292)  

**Abstract**: What if large language models could not only infer human mindsets but also expose every blind spot in team dialogue such as discrepancies in the team members' joint understanding? We present a novel, two-step framework that leverages large language models (LLMs) both as human-style annotators of team dialogues to track the team's shared mental models (SMMs) and as automated discrepancy detectors among individuals' mental states. In the first step, an LLM generates annotations by identifying SMM elements within task-oriented dialogues from the Cooperative Remote Search Task (CReST) corpus. Then, a secondary LLM compares these LLM-derived annotations and human annotations against gold-standard labels to detect and characterize divergences. We define an SMM coherence evaluation framework for this use case and apply it to six CReST dialogues, ultimately producing: (1) a dataset of human and LLM annotations; (2) a reproducible evaluation framework for SMM coherence; and (3) an empirical assessment of LLM-based discrepancy detection. Our results reveal that, although LLMs exhibit apparent coherence on straightforward natural-language annotation tasks, they systematically err in scenarios requiring spatial reasoning or disambiguation of prosodic cues. 

---
# Implicit Reasoning in Large Language Models: A Comprehensive Survey 

**Authors**: Jindong Li, Yali Fu, Li Fan, Jiahong Liu, Yao Shu, Chengwei Qin, Menglin Yang, Irwin King, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2509.02350)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong generalization across a wide range of tasks. Reasoning with LLMs is central to solving multi-step problems and complex decision-making. To support efficient reasoning, recent studies have shifted attention from explicit chain-of-thought prompting toward implicit reasoning, where reasoning occurs silently via latent structures without emitting intermediate textual steps. Implicit reasoning brings advantages such as lower generation cost, faster inference, and better alignment with internal computation. Although prior surveys have discussed latent representations in the context of reasoning, a dedicated and mechanism-level examination of how reasoning unfolds internally within LLMs remains absent. This survey fills that gap by introducing a taxonomy centered on execution paradigms, shifting the focus from representational forms to computational strategies. We organize existing methods into three execution paradigms based on \textbf{\textit{how and where internal computation unfolds}}: latent optimization, signal-guided control, and layer-recurrent execution. We also review structural, behavioral and representation-based evidence that supports the presence of implicit reasoning in LLMs. We further provide a structured overview of the evaluation metrics and benchmarks used in existing works to assess the effectiveness and reliability of implicit this http URL maintain a continuously updated project at: this https URL. 

---
# Top-H Decoding: Adapting the Creativity and Coherence with Bounded Entropy in Text Generation 

**Authors**: Erfan Baghaei Potraghloo, Seyedarmin Azizi, Souvik Kundu, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2509.02510)  

**Abstract**: Large language models (LLMs), despite their impressive performance across a wide range of tasks, often struggle to balance two competing objectives in open-ended text generation: fostering diversity and creativity while preserving logical coherence. Existing truncated sampling techniques, including temperature scaling, top-\$p\$ (nucleus) sampling, and min-\$p\$ sampling, aim to manage this trade-off. However, they exhibit limitations, particularly in the effective incorporation of the confidence of the model into the corresponding sampling strategy. For example, min-\$p\$ sampling relies on a single top token as a heuristic for confidence, eventually underutilizing the information of the probability distribution. Toward effective incorporation of the confidence of the model, in this paper, we present **top-H** decoding. We first establish the theoretical foundation of the interplay between creativity and coherence in truncated sampling by formulating an **entropy-constrained minimum divergence** problem. We then prove this minimization problem to be equivalent to an **entropy-constrained mass maximization** (ECMM) problem, which is NP-hard. Finally, we present top-H decoding, a computationally efficient greedy algorithm to solve the ECMM problem. Extensive empirical evaluations demonstrate that top-H outperforms the state-of-the-art (SoTA) alternative of min-\$p\$ sampling by up to **25.63%** on creative writing benchmarks, while maintaining robustness on question-answering datasets such as GPQA, GSM8K, and MT-Bench. Additionally, an *LLM-as-judge* evaluation confirms that top-H indeed produces coherent outputs even at higher temperatures, where creativity is especially critical. In summary, top-H advances SoTA in open-ended text generation and can be *easily integrated* into creative writing applications. The code is available at this https URL. 

---
# FActBench: A Benchmark for Fine-grained Automatic Evaluation of LLM-Generated Text in the Medical Domain 

**Authors**: Anum Afzal, Juraj Vladika, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2509.02198)  

**Abstract**: Large Language Models tend to struggle when dealing with specialized domains. While all aspects of evaluation hold importance, factuality is the most critical one. Similarly, reliable fact-checking tools and data sources are essential for hallucination mitigation. We address these issues by providing a comprehensive Fact-checking Benchmark FActBench covering four generation tasks and six state-of-the-art Large Language Models (LLMs) for the Medical domain. We use two state-of-the-art Fact-checking techniques: Chain-of-Thought (CoT) Prompting and Natural Language Inference (NLI). Our experiments show that the fact-checking scores acquired through the Unanimous Voting of both techniques correlate best with Domain Expert Evaluation. 

---
# Comparative Study of Pre-Trained BERT and Large Language Models for Code-Mixed Named Entity Recognition 

**Authors**: Mayur Shirke, Amey Shembade, Pavan Thorat, Madhushri Wagh, Raviraj Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2509.02514)  

**Abstract**: Named Entity Recognition (NER) in code-mixed text, particularly Hindi-English (Hinglish), presents unique challenges due to informal structure, transliteration, and frequent language switching. This study conducts a comparative evaluation of code-mixed fine-tuned models and non-code-mixed multilingual models, along with zero-shot generative large language models (LLMs). Specifically, we evaluate HingBERT, HingMBERT, and HingRoBERTa (trained on code-mixed data), and BERT Base Cased, IndicBERT, RoBERTa and MuRIL (trained on non-code-mixed multilingual data). We also assess the performance of Google Gemini in a zero-shot setting using a modified version of the dataset with NER tags removed. All models are tested on a benchmark Hinglish NER dataset using Precision, Recall, and F1-score. Results show that code-mixed models, particularly HingRoBERTa and HingBERT-based fine-tuned models, outperform others - including closed-source LLMs like Google Gemini - due to domain-specific pretraining. Non-code-mixed models perform reasonably but show limited adaptability. Notably, Google Gemini exhibits competitive zero-shot performance, underlining the generalization strength of modern LLMs. This study provides key insights into the effectiveness of specialized versus generalized models for code-mixed NER tasks. 

---
# An Ensemble Classification Approach in A Multi-Layered Large Language Model Framework for Disease Prediction 

**Authors**: Ali Hamdi, Malak Mohamed, Rokaia Emad, Khaled Shaban  

**Link**: [PDF](https://arxiv.org/pdf/2509.02446)  

**Abstract**: Social telehealth has made remarkable progress in healthcare by allowing patients to post symptoms and participate in medical consultations remotely. Users frequently post symptoms on social media and online health platforms, creating a huge repository of medical data that can be leveraged for disease classification. Large language models (LLMs) such as LLAMA3 and GPT-3.5, along with transformer-based models like BERT, have demonstrated strong capabilities in processing complex medical text. In this study, we evaluate three Arabic medical text preprocessing methods such as summarization, refinement, and Named Entity Recognition (NER) before applying fine-tuned Arabic transformer models (CAMeLBERT, AraBERT, and AsafayaBERT). To enhance robustness, we adopt a majority voting ensemble that combines predictions from original and preprocessed text representations. This approach achieved the best classification accuracy of 80.56%, thus showing its effectiveness in leveraging various text representations and model predictions to improve the understanding of medical texts. To the best of our knowledge, this is the first work that integrates LLM-based preprocessing with fine-tuned Arabic transformer models and ensemble learning for disease classification in Arabic social telehealth data. 

---
# JudgeAgent: Dynamically Evaluate LLMs with Agent-as-Interviewer 

**Authors**: Zhichao Shi, Xuhui Jiang, Chengjin Xu, Cangli Yao, Zhenxin Huang, Shengjie Ma, Yinghan Shen, Yuanzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02097)  

**Abstract**: Evaluating the capabilities of large language models (LLMs) is an essential step to ensure the successful application of LLMs across various domains. The current evaluation of LLMs is based on a paradigm that involves querying them with predefined question sets and assessing their outputs. This paradigm offers controllable processes and simplicity, but faces challenges such as limited interaction with targets, insufficient difficulty control, and difficulties in verifying the validity of evaluation results, making it hard to precisely determine the knowledge and capability boundaries of target models. To address these challenges, we propose JudgeAgent, a knowledge-target adaptive dynamic evaluation framework based on a new interviewer-style evaluation paradigm. JudgeAgent employs a comprehensive evaluation approach consisting of benchmark grading, interactive extension, and evaluation feedback. It utilizes knowledge-driven data synthesis and target-adaptive difficulty adjustment methods to conduct extended testing, providing accurate and effective evaluation results. We also introduce a novel insight into validating evaluation methods, demonstrating the effectiveness of JudgeAgent and its dynamic evaluation paradigm through extensive experiments. 

---
# How Instruction-Tuning Imparts Length Control: A Cross-Lingual Mechanistic Analysis 

**Authors**: Elisabetta Rocchetti, Alfio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2509.02075)  

**Abstract**: Adhering to explicit length constraints, such as generating text with a precise word count, remains a significant challenge for Large Language Models (LLMs). This study aims at investigating the differences between foundation models and their instruction-tuned counterparts, on length-controlled text generation in English and Italian. We analyze both performance and internal component contributions using Cumulative Weighted Attribution, a metric derived from Direct Logit Attribution. Our findings reveal that instruction-tuning substantially improves length control, primarily by specializing components in deeper model layers. Specifically, attention heads in later layers of IT models show increasingly positive contributions, particularly in English. In Italian, while attention contributions are more attenuated, final-layer MLPs exhibit a stronger positive role, suggesting a compensatory mechanism. These results indicate that instruction-tuning reconfigures later layers for task adherence, with component-level strategies potentially adapting to linguistic context. 

---
# DeepSeek performs better than other Large Language Models in Dental Cases 

**Authors**: Hexian Zhang, Xinyu Yan, Yanqi Yang, Lijian Jin, Ping Yang, Junwen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02036)  

**Abstract**: Large language models (LLMs) hold transformative potential in healthcare, yet their capacity to interpret longitudinal patient narratives remains inadequately explored. Dentistry, with its rich repository of structured clinical data, presents a unique opportunity to rigorously assess LLMs' reasoning abilities. While several commercial LLMs already exist, DeepSeek, a model that gained significant attention earlier this year, has also joined the competition. This study evaluated four state-of-the-art LLMs (GPT-4o, Gemini 2.0 Flash, Copilot, and DeepSeek V3) on their ability to analyze longitudinal dental case vignettes through open-ended clinical tasks. Using 34 standardized longitudinal periodontal cases (comprising 258 question-answer pairs), we assessed model performance via automated metrics and blinded evaluations by licensed dentists. DeepSeek emerged as the top performer, demonstrating superior faithfulness (median score = 0.528 vs. 0.367-0.457) and higher expert ratings (median = 4.5/5 vs. 4.0/5), without significantly compromising readability. Our study positions DeepSeek as the leading LLM for case analysis, endorses its integration as an adjunct tool in both medical education and research, and highlights its potential as a domain-specific agent. 

---
# Avoidance Decoding for Diverse Multi-Branch Story Generation 

**Authors**: Kyeongman Park, Nakyeong Yang, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2509.02170)  

**Abstract**: Large Language Models (LLMs) often generate repetitive and monotonous outputs, especially in tasks like story generation, due to limited creative diversity when given the same input prompt. To address this challenge, we propose a novel decoding strategy, Avoidance Decoding, that modifies token logits by penalizing similarity to previously generated outputs, thereby encouraging more diverse multi-branch stories. This penalty adaptively balances two similarity measures: (1) Concept-level Similarity Penalty, which is prioritized in early stages to diversify initial story concepts, and (2) Narrative-level Similarity Penalty, which is increasingly emphasized later to ensure natural yet diverse plot development. Notably, our method achieves up to 2.6 times higher output diversity and reduces repetition by an average of 30% compared to strong baselines, while effectively mitigating text degeneration. Furthermore, we reveal that our method activates a broader range of neurons, demonstrating that it leverages the model's intrinsic creativity. 

---
# DRAssist: Dispute Resolution Assistance using Large Language Models 

**Authors**: Sachin Pawar, Manoj Apte, Girish K. Palshikar, Basit Ali, Nitin Ramrakhiyani  

**Link**: [PDF](https://arxiv.org/pdf/2509.01962)  

**Abstract**: Disputes between two parties occur in almost all domains such as taxation, insurance, banking, healthcare, etc. The disputes are generally resolved in a specific forum (e.g., consumer court) where facts are presented, points of disagreement are discussed, arguments as well as specific demands of the parties are heard, and finally a human judge resolves the dispute by often favouring one of the two parties. In this paper, we explore the use of large language models (LLMs) as assistants for the human judge to resolve such disputes, as part of our DRAssist system. We focus on disputes from two specific domains -- automobile insurance and domain name disputes. DRAssist identifies certain key structural elements (e.g., facts, aspects or disagreement, arguments) of the disputes and summarizes the unstructured dispute descriptions to produce a structured summary for each dispute. We then explore multiple prompting strategies with multiple LLMs for their ability to assist in resolving the disputes in these domains. In DRAssist, these LLMs are prompted to produce the resolution output at three different levels -- (i) identifying an overall stronger party in a dispute, (ii) decide whether each specific demand of each contesting party can be accepted or not, (iii) evaluate whether each argument by each contesting party is strong or weak. We evaluate the performance of LLMs on all these tasks by comparing them with relevant baselines using suitable evaluation metrics. 

---
# Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs 

**Authors**: Andong Hua, Kenan Tang, Chenhe Gu, Jindong Gu, Eric Wong, Yao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.01790)  

**Abstract**: Prompt sensitivity, referring to the phenomenon where paraphrasing (i.e., repeating something written or spoken using different words) leads to significant changes in large language model (LLM) performance, has been widely accepted as a core limitation of LLMs. In this work, we revisit this issue and ask: Is the widely reported high prompt sensitivity truly an inherent weakness of LLMs, or is it largely an artifact of evaluation processes? To answer this question, we systematically evaluate 7 LLMs (e.g., GPT and Gemini family) across 6 benchmarks, including both multiple-choice and open-ended tasks on 12 diverse prompt templates. We find that much of the prompt sensitivity stems from heuristic evaluation methods, including log-likelihood scoring and rigid answer matching, which often overlook semantically correct responses expressed through alternative phrasings, such as synonyms or paraphrases. When we adopt LLM-as-a-Judge evaluations, we observe a substantial reduction in performance variance and a consistently higher correlation in model rankings across prompts. Our findings suggest that modern LLMs are more robust to prompt templates than previously believed, and that prompt sensitivity may be more an artifact of evaluation than a flaw in the models. 

---
# AMBEDKAR-A Multi-level Bias Elimination through a Decoding Approach with Knowledge Augmentation for Robust Constitutional Alignment of Language Models 

**Authors**: Snehasis Mukhopadhyay, Aryan Kasat, Shivam Dubey, Rahul Karthikeyan, Dhruv Sood, Vinija Jain, Aman Chadha, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2509.02133)  

**Abstract**: Large Language Models (LLMs) can inadvertently reflect societal biases present in their training data, leading to harmful or prejudiced outputs. In the Indian context, our empirical evaluations across a suite of models reveal that biases around caste and religion are particularly salient. Yet, most existing mitigation strategies are Western-centric and fail to address these local nuances. We propose AMBEDKAR, a framework inspired by the egalitarian vision of Dr B. R. Ambedkar, architect of the Indian Constitution, to guide LLM outputs toward fairness, neutrality, and inclusion in line with Articles 14 to 17. Our approach introduces a Constitution-Aware Decoding Layer, guided by the AI Constitution of India and applied only at inference time, without any parameter updates to the base model. We incorporate a speculative decoding algorithm that proactively reduces casteist and communal bias during generation. This mitigation layer operates directly within the decoding process, avoiding changes to model internals and lowering the computational and infrastructural costs associated with retraining. We reinterpret speculative decoding not merely as an efficiency tool but as a mechanism for fairness. In this framework, a Small Language Model (SLM) acts as a potentially biased generator, while a constitutionally guided Large Language Model (LLM) serves as the verifier. Rather than accelerating generation, the LLM enforces bias-robust trajectories in the SLM outputs. This inversion of roles gives rise to a fairness-by-speculation paradigm. Our approach yields an absolute reduction of bias up to 26.41 percent compared to baseline. Our source code, datasets, and results are available at this https URL 

---
# Attributes as Textual Genes: Leveraging LLMs as Genetic Algorithm Simulators for Conditional Synthetic Data Generation 

**Authors**: Guangzeng Han, Weisi Liu, Xiaolei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02040)  

**Abstract**: Large Language Models (LLMs) excel at generating synthetic data, but ensuring its quality and diversity remains challenging. We propose Genetic Prompt, a novel framework that combines genetic algorithms with LLMs to augment synthetic data generation. Our approach treats semantic text attributes as gene sequences and leverages the LLM to simulate crossover and mutation operations. This genetic process enhances data quality and diversity by creating novel attribute combinations, yielding synthetic distributions closer to real-world data. To optimize parent selection, we also integrate an active learning scheme that expands the offspring search space. Our experiments on multiple NLP tasks reveal several key findings: Genetic Prompt not only significantly outperforms state-of-the-art baselines but also shows robust performance across various generator model sizes and scales. Moreover, we demonstrate that fusing our synthetic data with the original training set significantly boosts downstream model performance, particularly for class-imbalanced scenarios. Our findings validate that Genetic Prompt is an effective method for producing high-quality synthetic data for a wide range of NLP applications. 

---
# Benchmarking the Detection of LLMs-Generated Modern Chinese Poetry 

**Authors**: Shanshan Wang, Junchao Wu, Fengying Ye, Jingming Yao, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2509.01620)  

**Abstract**: The rapid development of advanced large language models (LLMs) has made AI-generated text indistinguishable from human-written text. Previous work on detecting AI-generated text has made effective progress, but has not involved modern Chinese poetry. Due to the distinctive characteristics of modern Chinese poetry, it is difficult to identify whether a poem originated from humans or AI. The proliferation of AI-generated modern Chinese poetry has significantly disrupted the poetry ecosystem. Based on the urgency of identifying AI-generated poetry in the real Chinese world, this paper proposes a novel benchmark for detecting LLMs-generated modern Chinese poetry. We first construct a high-quality dataset, which includes both 800 poems written by six professional poets and 41,600 poems generated by four mainstream LLMs. Subsequently, we conduct systematic performance assessments of six detectors on this dataset. Experimental results demonstrate that current detectors cannot be used as reliable tools to detect modern Chinese poems generated by LLMs. The most difficult poetic features to detect are intrinsic qualities, especially style. The detection results verify the effectiveness and necessity of our proposed benchmark. Our work lays a foundation for future detection of AI-generated poetry. 

---
# Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal Belief 

**Authors**: Zeguan Xiao, Diyang Dou, Boya Xiong, Yun Chen, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.01564)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language tasks, but often exhibit overconfidence and generate plausible yet incorrect answers. This overconfidence, especially in models undergone Reinforcement Learning from Human Feedback (RLHF), poses significant challenges for reliable uncertainty estimation and safe deployment. In this paper, we propose EAGLE (Expectation of AGgregated internaL bEief), a novel self-evaluation-based calibration method that leverages the internal hidden states of LLMs to derive more accurate confidence scores. Instead of relying on the model's final output, our approach extracts internal beliefs from multiple intermediate layers during self-evaluation. By aggregating these layer-wise beliefs and calculating the expectation over the resulting confidence score distribution, EAGLE produces a refined confidence score that more faithfully reflects the model's internal certainty. Extensive experiments on diverse datasets and LLMs demonstrate that EAGLE significantly improves calibration performance over existing baselines. We also provide an in-depth analysis of EAGLE, including a layer-wise examination of uncertainty patterns, a study of the impact of self-evaluation prompts, and an analysis of the effect of self-evaluation score range. 

---
# Extracting OPQRST in Electronic Health Records using Large Language Models with Reasoning 

**Authors**: Zhimeng Luo, Abhibha Gupta, Adam Frisch, Daqing He  

**Link**: [PDF](https://arxiv.org/pdf/2509.01885)  

**Abstract**: The extraction of critical patient information from Electronic Health Records (EHRs) poses significant challenges due to the complexity and unstructured nature of the data. Traditional machine learning approaches often fail to capture pertinent details efficiently, making it difficult for clinicians to utilize these tools effectively in patient care. This paper introduces a novel approach to extracting the OPQRST assessment from EHRs by leveraging the capabilities of Large Language Models (LLMs). We propose to reframe the task from sequence labeling to text generation, enabling the models to provide reasoning steps that mimic a physician's cognitive processes. This approach enhances interpretability and adapts to the limited availability of labeled data in healthcare settings. Furthermore, we address the challenge of evaluating the accuracy of machine-generated text in clinical contexts by proposing a modification to traditional Named Entity Recognition (NER) metrics. This includes the integration of semantic similarity measures, such as the BERT Score, to assess the alignment between generated text and the clinical intent of the original records. Our contributions demonstrate a significant advancement in the use of AI in healthcare, offering a scalable solution that improves the accuracy and usability of information extraction from EHRs, thereby aiding clinicians in making more informed decisions and enhancing patient care outcomes. 

---
# Vis-CoT: A Human-in-the-Loop Framework for Interactive Visualization and Intervention in LLM Chain-of-Thought Reasoning 

**Authors**: Kaviraj Pather, Elena Hadjigeorgiou, Arben Krasniqi, Claire Schmit, Irina Rusu, Marc Pons, Kabir Khan  

**Link**: [PDF](https://arxiv.org/pdf/2509.01412)  

**Abstract**: Large language models (LLMs) show strong reasoning via chain-of-thought (CoT) prompting, but the process is opaque, which makes verification, debugging, and control difficult in high-stakes settings. We present Vis-CoT, a human-in-the-loop framework that converts linear CoT text into an interactive reasoning graph. Users can visualize the logical flow, identify flawed steps, and intervene by pruning incorrect paths and grafting new, user-defined premises. This shifts interaction from passive observation to active collaboration, steering models toward more accurate and trustworthy conclusions. Across GSM8K and StrategyQA, Vis-CoT improves final-answer accuracy by up to 24 percentage points over non-interactive baselines. A user study also shows large gains in perceived usability and trust. Vis-CoT points to a practical path for more reliable, understandable, and collaborative reasoning by combining LLMs with targeted human oversight. 

---
# Do Retrieval Augmented Language Models Know When They Don't Know? 

**Authors**: Youchao Zhou, Heyan Huang, Yicheng Liu, Rui Dai, Xinglin Wang, Xingchen Zhang, Shumin Shi, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.01476)  

**Abstract**: Existing Large Language Models (LLMs) occasionally generate plausible yet factually incorrect responses, known as hallucinations. Researchers are primarily using two approaches to mitigate hallucinations, namely Retrieval Augmented Language Models (RALMs) and refusal post-training. However, current research predominantly emphasizes their individual effectiveness while overlooking the evaluation of the refusal capability of RALMs. In this study, we ask the fundamental question: Do RALMs know when they don't know? Specifically, we ask three questions. First, are RALMs well-calibrated regarding different internal and external knowledge states? We examine the influence of various factors. Contrary to expectations, we find that LLMs exhibit significant \textbf{over-refusal} behavior. Then, how does refusal post-training affect the over-refusal issue? We investigate the Refusal-aware Instruction Tuning and In-Context Fine-tuning methods. Our results show that the over-refusal problem is mitigated by In-context fine-tuning. but magnified by R-tuning. However, we also find that the refusal ability may conflict with the quality of the answer. Finally, we develop a simple yet effective refusal method for refusal post-trained models to improve their overall answer quality in terms of refusal and correct answers. Our study provides a more comprehensive understanding of the influence of important factors on RALM systems. 

---
# LLMs cannot spot math errors, even when allowed to peek into the solution 

**Authors**: KV Aditya Srivatsa, Kaushal Kumar Maurya, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2509.01395)  

**Abstract**: Large language models (LLMs) demonstrate remarkable performance on math word problems, yet they have been shown to struggle with meta-reasoning tasks such as identifying errors in student solutions. In this work, we investigate the challenge of locating the first error step in stepwise solutions using two error reasoning datasets: VtG and PRM800K. Our experiments show that state-of-the-art LLMs struggle to locate the first error step in student solutions even when given access to the reference solution. To that end, we propose an approach that generates an intermediate corrected student solution, aligning more closely with the original student's solution, which helps improve performance. 

---
# On the Alignment of Large Language Models with Global Human Opinion 

**Authors**: Yang Liu, Masahiro Kaneko, Chenhui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2509.01418)  

**Abstract**: Today's large language models (LLMs) are capable of supporting multilingual scenarios, allowing users to interact with LLMs in their native languages. When LLMs respond to subjective questions posed by users, they are expected to align with the views of specific demographic groups or historical periods, shaped by the language in which the user interacts with the model. Existing studies mainly focus on researching the opinions represented by LLMs among demographic groups in the United States or a few countries, lacking worldwide country samples and studies on human opinions in different historical periods, as well as lacking discussion on using language to steer LLMs. Moreover, they also overlook the potential influence of prompt language on the alignment of LLMs' opinions. In this study, our goal is to fill these gaps. To this end, we create an evaluation framework based on the World Values Survey (WVS) to systematically assess the alignment of LLMs with human opinions across different countries, languages, and historical periods around the world. We find that LLMs appropriately or over-align the opinions with only a few countries while under-aligning the opinions with most countries. Furthermore, changing the language of the prompt to match the language used in the questionnaire can effectively steer LLMs to align with the opinions of the corresponding country more effectively than existing steering methods. At the same time, LLMs are more aligned with the opinions of the contemporary population. To our knowledge, our study is the first comprehensive investigation of the topic of opinion alignment in LLMs across global, language, and temporal dimensions. Our code and data are publicly available at this https URL. 

---
# Reasoning Vectors: Transferring Chain-of-Thought Capabilities via Task Arithmetic 

**Authors**: Mohammad Zbeeb, Hasan Abed Al Kader Hammoud, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2509.01363)  

**Abstract**: Large language models often require costly optimization, such as reinforcement learning, to master complex reasoning tasks. This work demonstrates that reasoning ability, once learned, can be extracted and transferred between models as a compact task vector. We source two publicly available, identically initialized Qwen2.5 models, one fine-tuned with supervised fine-tuning (SFT) and the other with group relative policy optimization (GRPO) on the same dataset. From these, we extract a reasoning vector: $v_{\text{reason}} = \theta_{\text{GRPO}} - \theta_{\text{SFT}}$. We hypothesize that this vector captures the reasoning capability instilled by reinforcement learning while factoring out shared knowledge from the SFT process. When added to compatible instruction-tuned models through simple arithmetic, this vector consistently improves performance across diverse reasoning benchmarks: GSM8K (+4.9%), HumanEval (+4.3%), SciQ (+1.7%), and BigBenchHard (+12.3% for the 1.5B model). The performance improvements persist under adversarial conditions. Conversely, subtracting the vector causes significant performance degradation (-11.8% on GSM8K), demonstrating the vector's strong contribution to the model's reasoning abilities. This work shows how reasoning capabilities, typically developed through expensive training, can be extracted from existing open-source models and reused through simple tensor arithmetic, offering a practical way to enhance models by recycling prior computational investments. 

---
# Can Large Language Models Master Complex Card Games? 

**Authors**: Wei Wang, Fuqing Bie, Junzhe Chen, Dan Zhang, Shiyu Huang, Evgeny Kharlamov, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01328)  

**Abstract**: Complex games have long been an important benchmark for testing the progress of artificial intelligence algorithms. AlphaGo, AlphaZero, and MuZero have defeated top human players in Go and Chess, garnering widespread societal attention towards artificial intelligence. Concurrently, large language models (LLMs) have exhibited remarkable capabilities across various tasks, raising the question of whether LLMs can achieve similar success in complex games. In this paper, we explore the potential of LLMs in mastering complex card games. We systematically assess the learning capabilities of LLMs across eight diverse card games, evaluating the impact of fine-tuning on high-quality gameplay data, and examining the models' ability to retain general capabilities while mastering these games. Our findings indicate that: (1) LLMs can approach the performance of strong game AIs through supervised fine-tuning on high-quality data, (2) LLMs can master multiple complex card games simultaneously, with performance augmentation for games with similar rules and conflicts for dissimilar ones, and (3) LLMs experience a decline in general capabilities when mastering complex games, but this decline can be mitigated by integrating a certain amount of general instruction data. The evaluation results demonstrate strong learning ability and versatility of LLMs. 

---
# Can Smaller LLMs do better? Unlocking Cross-Domain Potential through Parameter-Efficient Fine-Tuning for Text Summarization 

**Authors**: Anum Afzal, Mehul Kumawat, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2509.01314)  

**Abstract**: Large Language Models (LLMs), being generic task solvers, are versatile. However, despite the vast amount of data they are trained on, there are speculations about their adaptation capabilities to a new domain. Additionally, the simple fine-tuning of the model to incorporate knowledge of a new domain is computationally expensive and time-consuming. This becomes more challenging when the domain in question is also low-resource, and labeled data is unavailable. We leverage parameter-efficient fine-tuning techniques (PEFTs) on high-resource datasets to address these challenges to improve performance on unseen low-resource domains. Throughout our experiments, we evaluate whether intrinsic linguistic commonalities between datasets can be leveraged for efficient domain adaptation. We benchmark six PEFTs with \texttt{Llama-3-8B-Instruct} on 14 training datasets from the Scientific, Medical, Legal, and News domains for a Text Summarization task. Our experiments show that for low-resource domains, inference using Within-Domain Adapters can achieve better performance than Few-Shot as well as a much larger \texttt{Llama-3-70B-Instruct}. Lastly, in the absence of Within-Domain Adapters, we explore the concept of using Cross-Domain Adapters as well as the strategic combinations of adapters to leverage intrinsic language similarities across domains, facilitating better adaptability and performance in low-resource settings. 

---
# Robust Knowledge Editing via Explicit Reasoning Chains for Distractor-Resilient Multi-Hop QA 

**Authors**: Yuchen Wu, Liang Ding, Li Shen, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2509.01468)  

**Abstract**: Large language models (LLMs) encode vast amounts of world knowledge but remain static once trained, making the timely integration of emerging facts prohibitively expensive via full retraining. Knowledge-editing techniques have thus emerged to inject or overwrite specific facts into LLMs, yet they either over-rely on superficial cues or incur complex, iterative pipelines that collapse under noisy, multi-hop conditions. We introduce Reason-KE, an end-to-end reasoning-chain-based editing framework that steers a pretrained LLM through four structured stages-fact acknowledgment, relevance determination, selective application, and final reasoning-to filter distractors in a single pass. Trained on MQuAKE-CF with up to four irrelevant facts, Reason-KE elevates Qwen2.5-7B's multi-hop QA accuracy to 90.2% while suffering merely a 6.3% drop under heavy distraction and <1% when answers are leaked. Our quantitative analysis confirms Reason-KE's resilience and efficiency, establishing a new state-of-the-art for reliable LLM knowledge updates. 

---
# Culture is Everywhere: A Call for Intentionally Cultural Evaluation 

**Authors**: Juhyun Oh, Inha Cha, Michael Saxon, Hyunseung Lim, Shaily Bhatt, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2509.01301)  

**Abstract**: The prevailing ``trivia-centered paradigm'' for evaluating the cultural alignment of large language models (LLMs) is increasingly inadequate as these models become more advanced and widely deployed. Existing approaches typically reduce culture to static facts or values, testing models via multiple-choice or short-answer questions that treat culture as isolated trivia. Such methods neglect the pluralistic and interactive realities of culture, and overlook how cultural assumptions permeate even ostensibly ``neutral'' evaluation settings. In this position paper, we argue for \textbf{intentionally cultural evaluation}: an approach that systematically examines the cultural assumptions embedded in all aspects of evaluation, not just in explicitly cultural tasks. We systematically characterize the what, how, and circumstances by which culturally contingent considerations arise in evaluation, and emphasize the importance of researcher positionality for fostering inclusive, culturally aligned NLP research. Finally, we discuss implications and future directions for moving beyond current benchmarking practices, discovering important applications that we don't know exist, and involving communities in evaluation design through HCI-inspired participatory methodologies. 

---
# Rethinking the Chain-of-Thought: The Roles of In-Context Learning and Pre-trained Priors 

**Authors**: Hao Yang, Zhiyu Yang, Yunjie Zhang, Shanyi Zhu, Lin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01236)  

**Abstract**: Chain-of-Thought reasoning has emerged as a pivotal methodology for enhancing model inference capabilities. Despite growing interest in Chain-of-Thought reasoning, its underlying mechanisms remain unclear. This paper explores the working mechanisms of Chain-of-Thought reasoning from the perspective of the dual relationship between in-context learning and pretrained priors. We first conduct a fine-grained lexical-level analysis of rationales to examine the model's reasoning behavior. Then, by incrementally introducing noisy exemplars, we examine how the model balances pretrained priors against erroneous in-context information. Finally, we investigate whether prompt engineering can induce slow thinking in large language models. Our extensive experiments reveal three key findings: (1) The model not only quickly learns the reasoning structure at the lexical level but also grasps deeper logical reasoning patterns, yet it heavily relies on pretrained priors. (2) Providing sufficient exemplars shifts the model's decision-making from pretrained priors to in-context signals, while misleading prompts introduce instability. (3) Long Chain-of-Thought prompting can induce the model to generate longer reasoning chains, thereby improving its performance on downstream tasks. 

---
# TableZoomer: A Collaborative Agent Framework for Large-scale Table Question Answering 

**Authors**: Sishi Xiong, Ziyang He, Zhongjiang He, Yu Zhao, Changzai Pan, Jie Zhang, Zhenhe Wu, Shuangyong Song, Yongxiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.01312)  

**Abstract**: While large language models (LLMs) have shown promise in the table question answering (TQA) task through prompt engineering, they face challenges in industrial applications, including structural heterogeneity, difficulties in target data localization, and bottlenecks in complex reasoning. To address these limitations, this paper presents TableZoomer, a novel LLM-powered, programming-based agent framework. It introduces three key innovations: (1) replacing the original fully verbalized table with structured table schema to bridge the semantic gap and reduce computational complexity; (2) a query-aware table zooming mechanism that dynamically generates sub-table schema through column selection and entity linking, significantly improving target localization efficiency; and (3) a Program-of-Thoughts (PoT) strategy that transforms queries into executable code to mitigate numerical hallucination. Additionally, we integrate the reasoning workflow with the ReAct paradigm to enable iterative reasoning. Extensive experiments demonstrate that our framework maintains the usability advantages while substantially enhancing performance and scalability across tables of varying scales. When implemented with the Qwen3-8B-Instruct LLM, TableZoomer achieves accuracy improvements of 19.34% and 25% over conventional PoT methods on the large-scale DataBench dataset and the small-scale Fact Checking task of TableBench dataset, respectively. 

---
# CAT: Causal Attention Tuning For Injecting Fine-grained Causal Knowledge into Large Language Models 

**Authors**: Kairong Han, Wenshuo Zhao, Ziyu Zhao, JunJian Ye, Lujia Pan, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01535)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across various domains. However, a fundamental question remains: Can LLMs effectively utilize causal knowledge for prediction and generation? Through empirical studies, we find that LLMs trained directly on large-scale data often capture spurious correlations rather than true causal relationships, leading to suboptimal performance, especially in out-of-distribution (OOD) scenarios. To address this challenge, we propose Causal Attention Tuning (CAT), a novel approach that injects fine-grained causal knowledge into the attention mechanism. We propose an automated pipeline that leverages human priors to automatically generate token-level causal signals and introduce the Re-Attention mechanism to guide training, helping the model focus on causal structures while mitigating noise and biases in attention scores. Experimental results on our proposed Spurious Token Game (STG) benchmark and multiple downstream tasks demonstrate that our approach effectively leverages causal knowledge for prediction and remains robust in OOD scenarios. Implementation details can be found at this https URL. 

---
# DaMoC: Efficiently Selecting the Optimal Large Language Model for Fine-tuning Domain Taks Based on Data and Model Compression 

**Authors**: Wei Huang, Huang Wei, Yinggui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01221)  

**Abstract**: Large language models (LLMs) excel in general tasks but struggle with domain-specific ones, requiring fine-tuning with specific data. With many open-source LLMs available, selecting the best model for fine-tuning downstream tasks is challenging, primarily focusing on how to quickly identify the optimal LLM. We introduce a Data and Model Compression Framework (DaMoC) that addresses this challenge by: 1) Data Level: A systematic categorization of data filtering methodologies for LLMs is first established, classifying them into three distinct paradigms: (1) distribution-aware methods, (2) quality-aware methods, and (3) hybrid approaches considering both dimensions. Further, we enhance the density of key tokens in the text achieving token compression. Subsequently, we use an LLM to iterative rewrite the text to optimize its expression. 2) Model Level: We use layer similarity scores to assess each layer's importance and remove those with lower importance. Then, we introduce a sparse merging paradigm to preserve as much of the original model's capability as possible. Extensive experiments on four datasets, medical Q&A, financial Q&A, general Q&A, and reading comprehension, show that we can select the optimal LLM while saving approximately 20-fold in training time. 

---
# Efficient Large Language Models with Zero-Shot Adjustable Acceleration 

**Authors**: Sajjad Kachuee, Mohammad Sharifkhani  

**Link**: [PDF](https://arxiv.org/pdf/2509.01190)  

**Abstract**: Using Large Language Models (LLMs) in real-world applications presents significant challenges, particularly in balancing computational efficiency and performance. Optimizing acceleration after the fine-tuning phase and during inference is crucial for building an efficient architecture. This paper introduces Zero-Shot Adjustable Acceleration, a novel training and inference method that dynamically adjusts hardware usage during inference without requiring additional fine-tuning. The proposed approach is applied to newly developed models and evaluated across multiple classification and text generation tasks. Experimental results demonstrate that the method enables a wide range of acceleration in a zero-shot manner and achieves up to a 11x speedup compared to the baseline. 

---
# WATCHED: A Web AI Agent Tool for Combating Hate Speech by Expanding Data 

**Authors**: Paloma Piot, Diego Sánchez, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2509.01379)  

**Abstract**: Online harms are a growing problem in digital spaces, putting user safety at risk and reducing trust in social media platforms. One of the most persistent forms of harm is hate speech. To address this, we need tools that combine the speed and scale of automated systems with the judgment and insight of human moderators. These tools should not only find harmful content but also explain their decisions clearly, helping to build trust and understanding. In this paper, we present WATCHED, a chatbot designed to support content moderators in tackling hate speech. The chatbot is built as an Artificial Intelligence Agent system that uses Large Language Models along with several specialised tools. It compares new posts with real examples of hate speech and neutral content, uses a BERT-based classifier to help flag harmful messages, looks up slang and informal language using sources like Urban Dictionary, generates chain-of-thought reasoning, and checks platform guidelines to explain and support its decisions. This combination allows the chatbot not only to detect hate speech but to explain why content is considered harmful, grounded in both precedent and policy. Experimental results show that our proposed method surpasses existing state-of-the-art methods, reaching a macro F1 score of 0.91. Designed for moderators, safety teams, and researchers, the tool helps reduce online harms by supporting collaboration between AI and human oversight. 

---
# Modular Techniques for Synthetic Long-Context Data Generation in Language Model Training and Evaluation 

**Authors**: Seganrasan Subramanian, Abhigya Verma  

**Link**: [PDF](https://arxiv.org/pdf/2509.01185)  

**Abstract**: The ability of large language models (LLMs) to process and reason over long textual inputs is critical for a wide range of real-world applications. However, progress in this area is significantly constrained by the absence of high-quality, diverse, and verifiable long-context datasets suitable for both training and evaluation. This work introduces a modular, extensible framework for synthetic long-context data generation via prompt-based interaction with LLMs. The framework supports multiple training and alignment objectives, including Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO). It encompasses four core generation paradigms: multi-turn conversational dialogues, document-grounded input-output pairs, verifiable instruction-response tasks, and long-context reasoning examples. Through templated prompting, a model-agnostic architecture, and metadata-enriched outputs, the proposed approach facilitates scalable, controllable, and purpose-aligned dataset creation for advancing long-context capabilities in LLMs. 

---
# Trusted Uncertainty in Large Language Models: A Unified Framework for Confidence Calibration and Risk-Controlled Refusal 

**Authors**: Markus Oehri, Giulia Conti, Kaviraj Pather, Alexandre Rossi, Laia Serra, Adrian Parody, Rogvi Johannesen, Aviaja Petersen, Arben Krasniqi  

**Link**: [PDF](https://arxiv.org/pdf/2509.01455)  

**Abstract**: Deployed language models must decide not only what to answer but also when not to answer. We present UniCR, a unified framework that turns heterogeneous uncertainty evidence including sequence likelihoods, self-consistency dispersion, retrieval compatibility, and tool or verifier feedback into a calibrated probability of correctness and then enforces a user-specified error budget via principled refusal. UniCR learns a lightweight calibration head with temperature scaling and proper scoring, supports API-only models through black-box features, and offers distribution-free guarantees using conformal risk control. For long-form generation, we align confidence with semantic fidelity by supervising on atomic factuality scores derived from retrieved evidence, reducing confident hallucinations while preserving coverage. Experiments on short-form QA, code generation with execution tests, and retrieval-augmented long-form QA show consistent improvements in calibration metrics, lower area under the risk-coverage curve, and higher coverage at fixed risk compared to entropy or logit thresholds, post-hoc calibrators, and end-to-end selective baselines. Analyses reveal that evidence contradiction, semantic dispersion, and tool inconsistency are the dominant drivers of abstention, yielding informative user-facing refusal messages. The result is a portable recipe of evidence fusion to calibrated probability to risk-controlled decision that improves trustworthiness without fine-tuning the base model and remains valid under distribution shift. 

---
# Enhancing Large Language Model for Knowledge Graph Completion via Structure-Aware Alignment-Tuning 

**Authors**: Yu Liu, Yanan Cao, Xixun Lin, Yanmin Shang, Shi Wang, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.01166)  

**Abstract**: Knowledge graph completion (KGC) aims to infer new knowledge and make predictions from knowledge graphs. Recently, large language models (LLMs) have exhibited remarkable reasoning capabilities. LLM-enhanced KGC methods primarily focus on designing task-specific instructions, achieving promising advancements. However, there are still two critical challenges. First, existing methods often ignore the inconsistent representation spaces between natural language and graph structures. Second, most approaches design separate instructions for different KGC tasks, leading to duplicate works and time-consuming processes. To address these challenges, we propose SAT, a novel framework that enhances LLMs for KGC via structure-aware alignment-tuning. Specifically, we first introduce hierarchical knowledge alignment to align graph embeddings with the natural language space through multi-task contrastive learning. Then, we propose structural instruction tuning to guide LLMs in performing structure-aware reasoning over KGs, using a unified graph instruction combined with a lightweight knowledge adapter. Experimental results on two KGC tasks across four benchmark datasets demonstrate that SAT significantly outperforms state-of-the-art methods, especially in the link prediction task with improvements ranging from 8.7% to 29.8%. 

---
# Zero-shot Cross-lingual NER via Mitigating Language Difference: An Entity-aligned Translation Perspective 

**Authors**: Zhihao Zhang, Sophia Yat Mei Lee, Dong Zhang, Shoushan Li, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.01147)  

**Abstract**: Cross-lingual Named Entity Recognition (CL-NER) aims to transfer knowledge from high-resource languages to low-resource languages. However, existing zero-shot CL-NER (ZCL-NER) approaches primarily focus on Latin script language (LSL), where shared linguistic features facilitate effective knowledge transfer. In contrast, for non-Latin script language (NSL), such as Chinese and Japanese, performance often degrades due to deep structural differences. To address these challenges, we propose an entity-aligned translation (EAT) approach. Leveraging large language models (LLMs), EAT employs a dual-translation strategy to align entities between NSL and English. In addition, we fine-tune LLMs using multilingual Wikipedia data to enhance the entity alignment from source to target languages. 

---
# KoBLEX: Open Legal Question Answering with Multi-hop Reasoning 

**Authors**: Jihyung Lee, Daehui Kim, Seonjeong Hwang, Hyounghun Kim, Gary Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.01324)  

**Abstract**: Large Language Models (LLM) have achieved remarkable performances in general domains and are now extending into the expert domain of law. Several benchmarks have been proposed to evaluate LLMs' legal capabilities. However, these benchmarks fail to evaluate open-ended and provision-grounded Question Answering (QA). To address this, we introduce a Korean Benchmark for Legal EXplainable QA (KoBLEX), designed to evaluate provision-grounded, multi-hop legal reasoning. KoBLEX includes 226 scenario-based QA instances and their supporting provisions, created using a hybrid LLM-human expert pipeline. We also propose a method called Parametric provision-guided Selection Retrieval (ParSeR), which uses LLM-generated parametric provisions to guide legally grounded and reliable answers. ParSeR facilitates multi-hop reasoning on complex legal questions by generating parametric provisions and employing a three-stage sequential retrieval process. Furthermore, to better evaluate the legal fidelity of the generated answers, we propose Legal Fidelity Evaluation (LF-Eval). LF-Eval is an automatic metric that jointly considers the question, answer, and supporting provisions and shows a high correlation with human judgments. Experimental results show that ParSeR consistently outperforms strong baselines, achieving the best results across multiple LLMs. Notably, compared to standard retrieval with GPT-4o, ParSeR achieves +37.91 higher F1 and +30.81 higher LF-Eval. Further analyses reveal that ParSeR efficiently delivers consistent performance across reasoning depths, with ablations confirming the effectiveness of ParSeR. 

---
# Natural Context Drift Undermines the Natural Language Understanding of Large Language Models 

**Authors**: Yulong Wu, Viktor Schlegel, Riza Batista-Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2509.01093)  

**Abstract**: How does the natural evolution of context paragraphs affect question answering in generative Large Language Models (LLMs)? To investigate this, we propose a framework for curating naturally evolved, human-edited variants of reading passages from contemporary QA benchmarks and for analyzing LLM performance across a range of semantic similarity scores, which quantify how closely each variant aligns with content seen during pretraining. Using this framework, we evaluate six QA datasets and eight LLMs with publicly available training data. Our experiments reveal that LLM performance declines as reading passages naturally diverge from the versions encountered during pretraining-even when the question and all necessary information remains present at inference time. For instance, average model accuracy on BoolQ drops by over 30% from the highest to lowest similarity bins, with slopes exceeding 70 across several LLMs. These findings suggest that natural text evolution poses a significant challenge to the language understanding capabilities of LLMs. 

---
# Assessing Large Language Models on Islamic Legal Reasoning: Evidence from Inheritance Law Evaluation 

**Authors**: Abdessalam Bouchekif, Samer Rashwani, Heba Sbahi, Shahd Gaben, Mutez Al-Khatib, Mohammed Ghaly  

**Link**: [PDF](https://arxiv.org/pdf/2509.01081)  

**Abstract**: This paper evaluates the knowledge and reasoning capabilities of Large Language Models in Islamic inheritance law, known as 'ilm al-mawarith. We assess the performance of seven LLMs using a benchmark of 1,000 multiple-choice questions covering diverse inheritance scenarios, designed to test models' ability to understand the inheritance context and compute the distribution of shares prescribed by Islamic jurisprudence. The results reveal a significant performance gap: o3 and Gemini 2.5 achieved accuracies above 90%, whereas ALLaM, Fanar, LLaMA, and Mistral scored below 50%. These disparities reflect important differences in reasoning ability and domain adaptation. We conduct a detailed error analysis to identify recurring failure patterns across models, including misunderstandings of inheritance scenarios, incorrect application of legal rules, and insufficient domain knowledge. Our findings highlight limitations in handling structured legal reasoning and suggest directions for improving performance in Islamic legal reasoning. Code: this https URL 

---
# LongCat-Flash Technical Report 

**Authors**: Meituan LongCat Team, Bayan, Bei Li, Bingye Lei, Bo Wang, Bolin Rong, Chao Wang, Chao Zhang, Chen Gao, Chen Zhang, Cheng Sun, Chengcheng Han, Chenguang Xi, Chi Zhang, Chong Peng, Chuan Qin, Chuyu Zhang, Cong Chen, Congkui Wang, Dan Ma, Daoru Pan, Defei Bu, Dengchang Zhao, Deyang Kong, Dishan Liu, Feiye Huo, Fengcun Li, Fubao Zhang, Gan Dong, Gang Liu, Gang Xu, Ge Li, Guoqiang Tan, Guoyuan Lin, Haihang Jing, Haomin Fu, Haonan Yan, Haoxing Wen, Haozhe Zhao, Hong Liu, Hongmei Shi, Hongyan Hao, Hongyin Tang, Huantian Lv, Hui Su, Jiacheng Li, Jiahao Liu, Jiahuan Li, Jiajun Yang, Jiaming Wang, Jian Yang, Jianchao Tan, Jiaqi Sun, Jiaqi Zhang, Jiawei Fu, Jiawei Yang, Jiaxi Hu, Jiayu Qin, Jingang Wang, Jiyuan He, Jun Kuang, Junhui Mei, Kai Liang, Ke He, Kefeng Zhang, Keheng Wang, Keqing He, Liang Gao, Liang Shi, Lianhui Ma, Lin Qiu, Lingbin Kong, Lingtong Si, Linkun Lyu, Linsen Guo, Liqi Yang, Lizhi Yan, Mai Xia, Man Gao, Manyuan Zhang, Meng Zhou, Mengxia Shen, Mingxiang Tuo, Mingyang Zhu, Peiguang Li, Peng Pei, Peng Zhao, Pengcheng Jia, Pingwei Sun, Qi Gu, Qianyun Li, Qingyuan Li, Qiong Huang, Qiyuan Duan, Ran Meng, Rongxiang Weng, Ruichen Shao, Rumei Li, Shizhe Wu, Shuai Liang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01322)  

**Abstract**: We introduce LongCat-Flash, a 560-billion-parameter Mixture-of-Experts (MoE) language model designed for both computational efficiency and advanced agentic capabilities. Stemming from the need for scalable efficiency, LongCat-Flash adopts two novel designs: (a) Zero-computation Experts, which enables dynamic computational budget allocation and activates 18.6B-31.3B (27B on average) per token depending on contextual demands, optimizing resource usage. (b) Shortcut-connected MoE, which enlarges the computation-communication overlap window, demonstrating notable gains in inference efficiency and throughput compared to models of a comparable scale. We develop a comprehensive scaling framework for large models that combines hyperparameter transfer, model-growth initialization, a multi-pronged stability suite, and deterministic computation to achieve stable and reproducible training. Notably, leveraging the synergy among scalable architectural design and infrastructure efforts, we complete model training on more than 20 trillion tokens within 30 days, while achieving over 100 tokens per second (TPS) for inference at a cost of \$0.70 per million output tokens. To cultivate LongCat-Flash towards agentic intelligence, we conduct a large-scale pre-training on optimized mixtures, followed by targeted mid- and post-training on reasoning, code, and instructions, with further augmentation from synthetic data and tool use tasks. Comprehensive evaluations demonstrate that, as a non-thinking foundation model, LongCat-Flash delivers highly competitive performance among other leading models, with exceptional strengths in agentic tasks. The model checkpoint of LongCat-Flash is open-sourced to foster community research.
LongCat Chat: this https URL
Hugging Face: this https URL
GitHub: this https URL 

---
# Privacy-Preserving Reasoning with Knowledge-Distilled Parametric Retrieval Augmented Generation 

**Authors**: Jinwen Chen, Hainan Zhang, Liang Pang, Yongxin Tong, Haibo Zhou, Yuan Zhan, Wei Lin, Zhiming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.01088)  

**Abstract**: The current RAG system requires uploading plaintext documents to the cloud, risking private data leakage. Parametric RAG (PRAG) addresses this by encoding documents as LoRA within LLMs, enabling reasoning without exposing raw content. However, it still faces two issues: (1) PRAG demands synthesizing QA pairs and fine-tuning LLM for each individual document to create its corresponding LoRA, leading to unacceptable inference latency. (2) The performance of PRAG relies solely on synthetic QA data, lacking internal alignment with standard RAG, resulting in poor generalization on out-of-distribution(OOD) inputs. Therefore, achieving high-efficiency parameterization while maintaining RAG-level performance remains a critical challenge for privacy-preserving reasoning. In this paper, we propose DistilledPRAG, a generalizable knowledge-distilled parametric RAG model aligned with standard RAG in document structure and parameter activation. We first synthesize QA pairs from single and multi-documents to enhance cross-document reasoning. Then, we mask the plaintext documents with a special token and translate them to LoRA via a parameter generator, maintaining the standard RAG document structure. Finally, guided by synthetic QA data, we train the parameter generator to match standard RAG's hidden states and output logits, enabling RAG-style reasoning without original documents. Experiments on four QA datasets show that DistilledPRAG outperforms baselines in accuracy and generalizes well on OOD data. 

---
# Joint Information Extraction Across Classical and Modern Chinese with Tea-MOELoRA 

**Authors**: Xuemei Tang, Chengxi Yan, Jinghang Gu, Chu-Ren Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01158)  

**Abstract**: Chinese information extraction (IE) involves multiple tasks across diverse temporal domains, including Classical and Modern documents. Fine-tuning a single model on heterogeneous tasks and across different eras may lead to interference and reduced performance. Therefore, in this paper, we propose Tea-MOELoRA, a parameter-efficient multi-task framework that combines LoRA with a Mixture-of-Experts (MoE) design. Multiple low-rank LoRA experts specialize in different IE tasks and eras, while a task-era-aware router mechanism dynamically allocates expert contributions. Experiments show that Tea-MOELoRA outperforms both single-task and joint LoRA baselines, demonstrating its ability to leverage task and temporal knowledge effectively. 

---
# MedCOD: Enhancing English-to-Spanish Medical Translation of Large Language Models Using Enriched Chain-of-Dictionary Framework 

**Authors**: Md Shahidul Salim, Lian Fu, Arav Adikesh Ramakrishnan, Zonghai Yao, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00934)  

**Abstract**: We present MedCOD (Medical Chain-of-Dictionary), a hybrid framework designed to improve English-to-Spanish medical translation by integrating domain-specific structured knowledge into large language models (LLMs). MedCOD integrates domain-specific knowledge from both the Unified Medical Language System (UMLS) and the LLM-as-Knowledge-Base (LLM-KB) paradigm to enhance structured prompting and fine-tuning. We constructed a parallel corpus of 2,999 English-Spanish MedlinePlus articles and a 100-sentence test set annotated with structured medical contexts. Four open-source LLMs (Phi-4, Qwen2.5-14B, Qwen2.5-7B, and LLaMA-3.1-8B) were evaluated using structured prompts that incorporated multilingual variants, medical synonyms, and UMLS-derived definitions, combined with LoRA-based fine-tuning. Experimental results demonstrate that MedCOD significantly improves translation quality across all models. For example, Phi-4 with MedCOD and fine-tuning achieved BLEU 44.23, chrF++ 28.91, and COMET 0.863, surpassing strong baseline models like GPT-4o and GPT-4o-mini. Ablation studies confirm that both MedCOD prompting and model adaptation independently contribute to performance gains, with their combination yielding the highest improvements. These findings highlight the potential of structured knowledge integration to enhance LLMs for medical translation tasks. 

---
# RPRO:Ranked Preference Reinforcement Optimization for Enhancing Medical QA and Diagnostic Reasoning 

**Authors**: Chia-Hsuan Hsu, Jun-En Ding, Hsin-Ling Hsu, Feng Liu, Fang-Ming Hung  

**Link**: [PDF](https://arxiv.org/pdf/2509.00974)  

**Abstract**: Medical question answering requires advanced reasoning that integrates domain knowledge with logical inference. However, existing large language models (LLMs) often generate reasoning chains that lack factual accuracy and clinical reliability. We propose Ranked Preference Reinforcement Optimization (RPRO), a novel framework that uniquely combines reinforcement learning with preference-driven reasoning refinement to enhance clinical chain-of-thought (CoT) performance. RPRO differentiates itself from prior approaches by employing task-adaptive reasoning templates and a probabilistic evaluation mechanism that aligns outputs with established clinical workflows, while automatically identifying and correcting low-quality reasoning chains. Unlike traditional pairwise preference methods, RPRO introduces a groupwise ranking optimization based on the Bradley-Terry model and incorporates KL-divergence regularization for stable training. Experiments on PubMedQA and MedQA-USMLE show consistent improvements over strong baselines. Remarkably, our 1.1B parameter model outperforms much larger 7B-13B models, including medical-specialized variants. These findings demonstrate that combining preference optimization with quality-driven refinement offers a scalable and effective approach to building more reliable, clinically grounded medical LLMs. 

---
# Exploring and Mitigating Fawning Hallucinations in Large Language Models 

**Authors**: Zixuan Shangguan, Yanjie Dong, Lanjun Wang, Xiaoyi Fan, Victor C. M. Leung, Xiping Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00869)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional proficiency in language understanding. However, when LLMs align their outputs with deceptive and/or misleading prompts, the generated responses could deviate from the de facto information. Such observations are known as fawning hallucinations, where the model prioritizes alignment with the input's implied perspective over accuracy and truthfulness. In this work, we analyze fawning hallucinations in various natural language processing tasks and tailor the so-termed contrastive decoding method for fawning-hallucination mitigation. Specifically, we design two paradigms to generate corresponding deceptive and/or misleading inputs for the consistent fawning hallucinations induction. Then, we propose the collaborative contrastive decoding (CCD) to handle the fawning hallucinations across different tasks in LLMs. By contrasting the deviation in output distribution between induced and transformed neutral inputs, the proposed CCD can reduce reliance on deceptive and/or misleading information without requiring additional training. Extensive experiments demonstrate that the proposed CCD can effectively mitigate fawning hallucinations and improve the factuality of the generated responses over various tasks. 

---
# Negative Matters: Multi-Granularity Hard-Negative Synthesis and Anchor-Token-Aware Pooling for Enhanced Text Embeddings 

**Authors**: Tengyu Pan, Zhichao Duan, Zhenyu Li, Bowen Dong, Ning Liu, Xiuxing Li, Jianyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00842)  

**Abstract**: Text embedding models are essential for various natural language processing tasks, enabling the effective encoding of semantic information into dense vector representations. These models are typically optimized using triplets of (query, positive, negative) data pairs for contrastive learning, where the negative samples play a critical role in enhancing the model's ability to discern subtle semantic distinctions. In this work, we introduce a Multi-Granularity Hard-negative (MGH) synthesis framework that leverages large language models (LLMs) to generate diverse negative samples with varying levels of similarity with the query. This approach facilitates a coarse-to-fine curriculum learning strategy during supervised training, allowing the embedding model to progressively learn more nuanced semantic representations. Meanwhile, we propose an Anchor Token Aware (ATA) pooling method that assigns higher weights to anchor tokens based on aggregation patterns observed in LLMs, improving text embedding accuracy without increasing model complexity. Comprehensive experiments on the MTEB benchmark demonstrate that our methods achieve state-of-the-art performance, surpassing existing synthesis strategies both with synthetic data and when combined with public retrieval datasets. 

---
# SeLeRoSa: Sentence-Level Romanian Satire Detection Dataset 

**Authors**: Răzvan-Alexandru Smădu, Andreea Iuga, Dumitru-Clementin Cercel, Florin Pop  

**Link**: [PDF](https://arxiv.org/pdf/2509.00893)  

**Abstract**: Satire, irony, and sarcasm are techniques typically used to express humor and critique, rather than deceive; however, they can occasionally be mistaken for factual reporting, akin to fake news. These techniques can be applied at a more granular level, allowing satirical information to be incorporated into news articles. In this paper, we introduce the first sentence-level dataset for Romanian satire detection for news articles, called SeLeRoSa. The dataset comprises 13,873 manually annotated sentences spanning various domains, including social issues, IT, science, and movies. With the rise and recent progress of large language models (LLMs) in the natural language processing literature, LLMs have demonstrated enhanced capabilities to tackle various tasks in zero-shot settings. We evaluate multiple baseline models based on LLMs in both zero-shot and fine-tuning settings, as well as baseline transformer-based models. Our findings reveal the current limitations of these models in the sentence-level satire detection task, paving the way for new research directions. 

---
# We Politely Insist: Your LLM Must Learn the Persian Art of Taarof 

**Authors**: Nikta Gohari Sadr, Sahar Heidariasl, Karine Megerdoomian, Laleh Seyyed-Kalantari, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2509.01035)  

**Abstract**: Large language models (LLMs) struggle to navigate culturally specific communication norms, limiting their effectiveness in global contexts. We focus on Persian taarof, a social norm in Iranian interactions, which is a sophisticated system of ritual politeness that emphasizes deference, modesty, and indirectness, yet remains absent from existing cultural benchmarks. We introduce TaarofBench, the first benchmark for evaluating LLM understanding of taarof, comprising 450 role-play scenarios covering 12 common social interaction topics, validated by native speakers. Our evaluation of five frontier LLMs reveals substantial gaps in cultural competence, with accuracy rates 40-48% below native speakers when taarof is culturally appropriate. Performance varies between interaction topics, improves with Persian-language prompts, and exhibits gender-based asymmetries. We also show that responses rated "polite" by standard metrics often violate taarof norms, indicating the limitations of Western politeness frameworks. Through supervised fine-tuning and Direct Preference Optimization, we achieve 21.8% and 42.3% improvement in model alignment with cultural expectations. Our human study with 33 participants (11 native Persian, 11 heritage, and 11 non-Iranian speakers) forms baselines in varying degrees of familiarity with Persian norms. This work lays the foundation for developing diverse and culturally aware LLMs, enabling applications that better navigate complex social interactions. 

---
# LLM Encoder vs. Decoder: Robust Detection of Chinese AI-Generated Text with LoRA 

**Authors**: Houji Jin, Negin Ashrafi, Armin Abdollahi, Wei Liu, Jian Wang, Ganyu Gui, Maryam Pishgar, Huanghao Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.00731)  

**Abstract**: The rapid growth of large language models (LLMs) has heightened the demand for accurate detection of AI-generated text, particularly in languages like Chinese, where subtle linguistic nuances pose significant challenges to current methods. In this study, we conduct a systematic comparison of encoder-based Transformers (Chinese BERT-large and RoBERTa-wwm-ext-large), a decoder-only LLM (Alibaba's Qwen2.5-7B/DeepSeek-R1-Distill-Qwen-7B fine-tuned via Low-Rank Adaptation, LoRA), and a FastText baseline using the publicly available dataset from the NLPCC 2025 Chinese AI-Generated Text Detection Task. Encoder models were fine-tuned using a novel prompt-based masked language modeling approach, while Qwen2.5-7B was adapted for classification with an instruction-format input and a lightweight classification head trained via LoRA. Experiments reveal that although encoder models nearly memorize training data, they suffer significant performance degradation under distribution shifts (RoBERTa: 76.3% test accuracy; BERT: 79.3%). FastText demonstrates surprising lexical robustness (83.5% accuracy) yet lacks deeper semantic understanding. In contrast, the LoRA-adapted Qwen2.5-7B achieves 95.94% test accuracy with balanced precision-recall metrics, indicating superior generalization and resilience to dataset-specific artifacts. These findings underscore the efficacy of decoder-based LLMs with parameter-efficient fine-tuning for robust Chinese AI-generated text detection. Future work will explore next-generation Qwen3 models, distilled variants, and ensemble strategies to enhance cross-domain robustness further. 

---
# Supervised In-Context Fine-Tuning for Generative Sequence Labeling 

**Authors**: David Dukić, Goran Glavaš, Jan Šnajder  

**Link**: [PDF](https://arxiv.org/pdf/2509.00921)  

**Abstract**: Sequence labeling (SL) tasks, where labels are assigned to tokens, are abundant in NLP (e.g., named entity recognition and aspect-based sentiment analysis). Owing to the intuition that they require bidirectional context, SL tasks are commonly tackled with encoder-only models. Recent work also shows that removing the causal mask in fine-tuning enables decoder-based LLMs to become effective token classifiers. Less work, however, focused on (supervised) generative SL, a more natural setting for causal LLMs. Due to their rapid scaling, causal LLMs applied to SL are expected to outperform encoders, whose own development has stagnated. In this work, we propose supervised in-context fine-tuning (SIFT) for generative SL. SIFT casts SL tasks as constrained response generation, natural to LLMs, combining (1) in-context learning (ICL) from demonstrations with (2) supervised fine-tuning. SIFT considerably outperforms both ICL and decoder-as-encoder fine-tuning baselines on a range of standard SL tasks. We further find that although long context hinders the performance of generative SL in both ICL and SIFT, this deficiency can be mitigated by removing the instruction, as instructions are shown to be largely unnecessary for achieving strong SL performance with SIFT. Our findings highlight strengths and limitations of SL with LLMs, underscoring the importance of a response-based generative task formulation for effective SL performance. 

---
# EviNote-RAG: Enhancing RAG Models via Answer-Supportive Evidence Notes 

**Authors**: Yuqin Dai, Guoqing Wang, Yuan Wang, Kairan Dou, Kaichen Zhou, Zhanwei Zhang, Shuo Yang, Fei Tang, Jun Yin, Pengyu Zeng, Zhenzhe Ying, Can Yi, Changhua Meng, Yuchen Zhou, Yongliang Shen, Shuai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00877)  

**Abstract**: Large Language Models (LLMs) empowered with retrieval mechanisms have achieved strong progress in open-domain question answering (QA). Yet, the conventional retrieve--then--answer paradigm often suffers from two key limitations: (1) low signal-to-noise ratio in retrieved evidence, where useful information is buried under irrelevant content, and (2) error accumulation in multi-hop reasoning when incomplete or noisy passages are involved. To address these challenges, we present EviNote-RAG, an agentic RAG framework that introduces a structured retrieve--note--answer pipeline. Instead of directly reasoning over raw retrievals, the model is trained to compose Supportive-Evidence Notes (SENs), concise, human-like notes that preserve only answer-relevant information, highlight uncertainty, and explicitly state when no useful evidence exists. This distillation process is further reinforced by the Evidence Quality Reward (EQR), an entailment-based signal that evaluates whether SENs logically support the final answer. Together, SENs and EQR guide the model toward faithful and robust reasoning, while reducing the impact of noise. Experiments on in-domain and out-of-domain QA benchmarks show that EviNote-RAG consistently outperforms strong baselines in accuracy, generalization, and training stability. In particular, it achieves state-of-the-art results while enhancing robustness and efficiency, yielding relative F1 gains of 20\% on HotpotQA (+0.093), 40\% on Bamboogle (+0.151), and 91\% on 2Wiki (+0.256) via denser rewards and reduced verbosity. 

---
# Decomposing and Revising What Language Models Generate 

**Authors**: Zhichao Yan, Jiaoyan Chen, Jiapu Wang, Xiaoli Li, Ru Li, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.00765)  

**Abstract**: Attribution is crucial in question answering (QA) with Large Language Models (LLMs).SOTA question decomposition-based approaches use long form answers to generate questions for retrieving related documents. However, the generated questions are often irrelevant and incomplete, resulting in a loss of facts in this http URL approaches also fail to aggregate evidence snippets from different documents and paragraphs. To tackle these problems, we propose a new fact decomposition-based framework called FIDES (\textit{faithful context enhanced fact decomposition and evidence aggregation}) for attributed QA. FIDES uses a contextually enhanced two-stage faithful decomposition method to decompose long form answers into sub-facts, which are then used by a retriever to retrieve related evidence snippets. If the retrieved evidence snippets conflict with the related sub-facts, such sub-facts will be revised accordingly. Finally, the evidence snippets are aggregated according to the original this http URL evaluation has been conducted with six datasets, with an additionally proposed new metric called $Attr_{auto-P}$ for evaluating the evidence precision. FIDES outperforms the SOTA methods by over 14\% in average with GPT-3.5-turbo, Gemini and Llama 70B series. 

---
# Learning to Shop Like Humans: A Review-driven Retrieval-Augmented Recommendation Framework with LLMs 

**Authors**: Kaiwen Wei, Jinpeng Gao, Jiang Zhong, Yuming Yang, Fengmao Lv, Zhenyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.00698)  

**Abstract**: Large language models (LLMs) have shown strong potential in recommendation tasks due to their strengths in language understanding, reasoning and knowledge integration. These capabilities are especially beneficial for review-based recommendation, which relies on semantically rich user-generated texts to reveal fine-grained user preferences and item attributes. However, effectively incorporating reviews into LLM-based recommendation remains challenging due to (1) inefficient to dynamically utilize user reviews under LLMs' constrained context windows, and (2) lacking effective mechanisms to prioritize reviews most relevant to the user's current decision context. To address these challenges, we propose RevBrowse, a review-driven recommendation framework inspired by the "browse-then-decide" decision process commonly observed in online user behavior. RevBrowse integrates user reviews into the LLM-based reranking process to enhance its ability to distinguish between candidate items. To improve the relevance and efficiency of review usage, we introduce PrefRAG, a retrieval-augmented module that disentangles user and item representations into structured forms and adaptively retrieves preference-relevant content conditioned on the target item. Extensive experiments on four Amazon review datasets demonstrate that RevBrowse achieves consistent and significant improvements over strong baselines, highlighting its generalizability and effectiveness in modeling dynamic user preferences. Furthermore, since the retrieval-augmented process is transparent, RevBrowse offers a certain level of interpretability by making visible which reviews influence the final recommendation. 

---
# Can Multi-turn Self-refined Single Agent LMs with Retrieval Solve Hard Coding Problems? 

**Authors**: Md Tanzib Hosain, Md Kishor Morol  

**Link**: [PDF](https://arxiv.org/pdf/2509.00629)  

**Abstract**: Among the hardest tasks for humans are those found in competitive programming where problems require sophisticated algorithmic thinking, puzzle solving, and the creation of effective code. As a domain to assess language models (LMs), it has not received enough attention, though. This study presents the ICPC benchmark, which consists of 254 international collegiate programming contest (ICPC) tasks. Each problem includes official analysis, reference code, and sample, high-quality unit, and hidden tests. We are able to develop and evaluate a variety of LM inference techniques for competitive programming with these resources. With zero-shot chain-of-thought prompting, we find that o1 only achieves a 19.1\% pass@1 solve rate. With our best inference technique, which combines multi-turn self-judge with reflection and retrieval over episodic information, raises this to 42.2\%. Furthermore, we conduct a new human-in-the-loop investigation to gain a deeper understanding of the remaining difficulties. Surprisingly, we discover that o1 can solve 17 out of 18 problems that were previously unsolvable by any model or technique with just a few specific instructions. A footstep toward LMs with grounded, imaginative, and algorithmic thinking is provided by our quantitative findings and qualitative research. We open-source our code and data at this https URL. 

---
# Thinking Hard, Going Misaligned: Emergent Misalignment in LLMs 

**Authors**: Hanqi Yan, Hainiu Xu, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2509.00544)  

**Abstract**: With Large Language Models (LLMs) becoming increasingly widely adopted, concerns regarding their safety and alignment with human values have intensified. Previous studies have shown that fine-tuning LLMs on narrow and malicious datasets induce misaligned behaviors. In this work, we report a more concerning phenomenon, Reasoning-Induced Misalignment. Specifically, we observe that LLMs become more responsive to malicious requests when reasoning is strengthened, via switching to "think-mode" or fine-tuning on benign math datasets, with dense models particularly vulnerable. Moreover, we analyze internal model states and find that both attention shifts and specialized experts in mixture-of-experts models help redirect excessive reasoning towards safety guardrails. These findings provide new insights into the emerging reasoning-safety trade-off and underscore the urgency of advancing alignment for advanced reasoning models. 

---
# StealthEval: A Probe-Rewrite-Evaluate Workflow for Reliable Benchmarks 

**Authors**: Lang Xiong, Nishant Bhargava, Wesley Chang, Jianhang Hong, Haihao Liu, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00591)  

**Abstract**: Large Language Models (LLMs) often exhibit significant behavioral shifts when they perceive a change from a real-world deployment context to a controlled evaluation setting, a phenomenon known as "evaluation awareness." This discrepancy poses a critical challenge for AI alignment, as benchmark performance may not accurately reflect a model's true safety and honesty. In this work, we systematically quantify these behavioral changes by manipulating the perceived context of prompts. We introduce a methodology that uses a linear probe to score prompts on a continuous scale from "test-like" to "deploy-like" and leverage an LLM rewriting strategy to shift these prompts towards a more natural, deployment-style context while preserving the original task. Using this method, we achieved a 30% increase in the average probe score across a strategic role-playing dataset after rewriting. Evaluating a suite of state-of-the-art models on these original and rewritten prompts, we find that rewritten "deploy-like" prompts induce a significant and consistent shift in behavior. Across all models, we observed an average increase in honest responses of 5.26% and a corresponding average decrease in deceptive responses of 12.40%. Furthermore, refusal rates increased by an average of 6.38%, indicating heightened safety compliance. Our findings demonstrate that evaluation awareness is a quantifiable and manipulable factor that directly influences LLM behavior, revealing that models are more prone to unsafe or deceptive outputs in perceived test environments. This underscores the urgent need for more realistic evaluation frameworks to accurately gauge true model alignment before deployment. 

---
# TECP: Token-Entropy Conformal Prediction for LLMs 

**Authors**: Beining Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00461)  

**Abstract**: Uncertainty quantification (UQ) for open-ended language generation remains a critical yet underexplored challenge, especially under black-box constraints where internal model signals are inaccessible. In this paper, we introduce Token-Entropy Conformal Prediction (TECP), a novel framework that leverages token-level entropy as a logit-free, reference-free uncertainty measure and integrates it into a split conformal prediction (CP) pipeline to construct prediction sets with formal coverage guarantees. Unlike existing approaches that rely on semantic consistency heuristics or white-box features, TECP directly estimates epistemic uncertainty from the token entropy structure of sampled generations and calibrates uncertainty thresholds via CP quantiles to ensure provable error control. Empirical evaluations across six large language models and two benchmarks (CoQA and TriviaQA) demonstrate that TECP consistently achieves reliable coverage and compact prediction sets, outperforming prior self-consistency-based UQ methods. Our method provides a principled and efficient solution for trustworthy generation in black-box LLM settings. 

---
# Modeling Motivated Reasoning in Law: Evaluating Strategic Role Conditioning in LLM Summarization 

**Authors**: Eunjung Cho, Alexander Hoyle, Yoan Hermstrüwer  

**Link**: [PDF](https://arxiv.org/pdf/2509.00529)  

**Abstract**: Large Language Models (LLMs) are increasingly used to generate user-tailored summaries, adapting outputs to specific stakeholders. In legal contexts, this raises important questions about motivated reasoning -- how models strategically frame information to align with a stakeholder's position within the legal system. Building on theories of legal realism and recent trends in legal practice, we investigate how LLMs respond to prompts conditioned on different legal roles (e.g., judges, prosecutors, attorneys) when summarizing judicial decisions. We introduce an evaluation framework grounded in legal fact and reasoning inclusion, also considering favorability towards stakeholders. Our results show that even when prompts include balancing instructions, models exhibit selective inclusion patterns that reflect role-consistent perspectives. These findings raise broader concerns about how similar alignment may emerge as LLMs begin to infer user roles from prior interactions or context, even without explicit role instructions. Our results underscore the need for role-aware evaluation of LLM summarization behavior in high-stakes legal settings. 

---
# Open Data Synthesis For Deep Research 

**Authors**: Ziyi Xia, Kun Luo, Hongjin Qian, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00375)  

**Abstract**: Large language models (LLMs) are increasingly expected to go beyond simple factual queries toward Deep Research-tasks that require decomposing questions into sub-problems, coordinating multi-step reasoning, and synthesizing evidence from diverse sources. We formalize Deep Research tasks with verifiable answers as Hierarchical Constraint Satisfaction Problems (HCSPs), which are fundamentally different from single-constraint, multi-hop, or flat CSP formulations. However, existing benchmarks (e.g., Natural Questions, HotpotQA) fail to capture this complexity, while recent synthetic datasets often introduce shortcut reasoning, knowledge leakage, or lack sufficient structural depth. To address this gap, we introduce InfoSeek, a scalable framework for synthesizing complex Deep Research tasks. InfoSeek uses a dual-agent system to recursively build a Research Tree from large-scale webpages, blurring intermediate nodes into valid sub-problems, and converting these trees into natural language questions that require traversing the full hierarchy. It also enables rapid scaling, yielding over 50K training examples, a curated test set, and reasoning trajectories generated via reject sampling. Experiments show that models trained on InfoSeek consistently outperform strong baselines. On a challenging benchmark BrowseComp-Plus, 3B LLMs optimized with InfoSeek surpass much larger 32B models and lightweight commercial APIs (e.g., Gemini2.5-Flash), while achieving performance comparable to stronger APIs (e.g., Gemini2.5-Pro). By preserving meta-information such as intermediate steps and retrieval labels, InfoSeek further supports advanced optimization strategies, including compound reward design and trajectory-level exploration. We provide our codes and datasets in \href{this https URL}{this repository}. 

---
# Wage Sentiment Indices Derived from Survey Comments via Large Language Models 

**Authors**: Taihei Sone  

**Link**: [PDF](https://arxiv.org/pdf/2509.00290)  

**Abstract**: The emergence of generative Artificial Intelligence (AI) has created new opportunities for economic text analysis. This study proposes a Wage Sentiment Index (WSI) constructed with Large Language Models (LLMs) to forecast wage dynamics in Japan. The analysis is based on the Economy Watchers Survey (EWS), a monthly survey conducted by the Cabinet Office of Japan that captures real-time economic assessments from workers in industries highly sensitive to business conditions. The WSI extends the framework of the Price Sentiment Index (PSI) used in prior studies, adapting it specifically to wage related sentiment. To ensure scalability and adaptability, a data architecture is also developed that enables integration of additional sources such as newspapers and social media. Experimental results demonstrate that WSI models based on LLMs significantly outperform both baseline approaches and pretrained models. These findings highlight the potential of LLM-driven sentiment indices to enhance the timeliness and effectiveness of economic policy design by governments and central banks. 

---
# Exploring Reasoning-Infused Text Embedding with Large Language Models for Zero-Shot Dense Retrieval 

**Authors**: Yuxiang Liu, Tian Wang, Gourab Kundu, Tianyu Cao, Guang Cheng, Zhen Ge, Jianshu Chen, Qingjun Cui, Trishul Chilimbi  

**Link**: [PDF](https://arxiv.org/pdf/2509.00276)  

**Abstract**: Transformer-based models such as BERT and E5 have significantly advanced text embedding by capturing rich contextual representations. However, many complex real-world queries require sophisticated reasoning to retrieve relevant documents beyond surface-level lexical matching, where encoder-only retrievers often fall short. Decoder-only large language models (LLMs), known for their strong reasoning capabilities, offer a promising alternative. Despite this potential, existing LLM-based embedding methods primarily focus on contextual representation and do not fully exploit the reasoning strength of LLMs. To bridge this gap, we propose Reasoning-Infused Text Embedding (RITE), a simple but effective approach that integrates logical reasoning into the text embedding process using generative LLMs. RITE builds upon existing language model embedding techniques by generating intermediate reasoning texts in the token space before computing embeddings, thereby enriching representations with inferential depth. Experimental results on BRIGHT, a reasoning-intensive retrieval benchmark, demonstrate that RITE significantly enhances zero-shot retrieval performance across diverse domains, underscoring the effectiveness of incorporating reasoning into the embedding process. 

---
# The Resurgence of GCG Adversarial Attacks on Large Language Models 

**Authors**: Yuting Tan, Xuying Li, Zhuo Li, Huizhen Shu, Peikang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00391)  

**Abstract**: Gradient-based adversarial prompting, such as the Greedy Coordinate Gradient (GCG) algorithm, has emerged as a powerful method for jailbreaking large language models (LLMs). In this paper, we present a systematic appraisal of GCG and its annealing-augmented variant, T-GCG, across open-source LLMs of varying scales. Using Qwen2.5-0.5B, LLaMA-3.2-1B, and GPT-OSS-20B, we evaluate attack effectiveness on both safety-oriented prompts (AdvBench) and reasoning-intensive coding prompts. Our study reveals three key findings: (1) attack success rates (ASR) decrease with model size, reflecting the increasing complexity and non-convexity of larger models' loss landscapes; (2) prefix-based heuristics substantially overestimate attack effectiveness compared to GPT-4o semantic judgments, which provide a stricter and more realistic evaluation; and (3) coding-related prompts are significantly more vulnerable than adversarial safety prompts, suggesting that reasoning itself can be exploited as an attack vector. In addition, preliminary results with T-GCG show that simulated annealing can diversify adversarial search and achieve competitive ASR under prefix evaluation, though its benefits under semantic judgment remain limited. Together, these findings highlight the scalability limits of GCG, expose overlooked vulnerabilities in reasoning tasks, and motivate further development of annealing-inspired strategies for more robust adversarial evaluation. 

---
# Balanced Actor Initialization: Stable RLHF Training of Distillation-Based Reasoning Models 

**Authors**: Chen Zheng, Yiyuan Ma, Yuan Yang, Deyi Liu, Jing Liu, Zuquan Song, Yuxin Song, Cheng Ren, Hang Zhu, Xin Liu, Yiyuan Ma, Siyuan Qiao, Xun Zhou, Liang Xiang, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00309)  

**Abstract**: The development of alignment and reasoning capabilities in large language models has seen remarkable progress through two paradigms: instruction tuning and reinforcement learning from human feedback (RLHF) alignment paradigm, and distillation-based reasoning fine-tuning paradigm. While both approaches prove effective independently, the third paradigm of applying RLHF to distillation-trained models presents significant challenges. Our investigation reveals two critical phenomena that emerge in this paradigm: Sequence Length Collapse, where language generation dramatically reduces during early RLHF training, and the Reward Hockey Stick Curve, featuring severe reward score drops followed by gradual recovery. These instabilities fundamentally compromise the model's alignment and reasoning capabilities. To address these challenges, we propose Balanced Actor Initialization (BAI), a two-stage weighted model merging approach. BAI first merges instruction-following and distillation-based reasoning fine-tuned models, then further combines this intermediate model with the pretrained model to preserve foundational knowledge. Through comprehensive experiments across diverse benchmarks and detailed analysis of training experiments, we demonstrate that BAI resolves Sequence Length Collapse, mitigates the Reward Hockey Stick Curve, and enables continuous sequence length improvement during training. Additionally, our analysis reveals that balanced merging ratios achieve optimal trade-offs between training stability and reasoning capability preservation. Our work provides the effective solution for stable training in this third paradigm, enabling more capable reasoning models that combine distillation efficiency with RLHF alignment. 

---
# CaresAI at BioCreative IX Track 1 -- LLM for Biomedical QA 

**Authors**: Reem Abdel-Salam, Mary Adewunmi, Modinat A. Abayomi  

**Link**: [PDF](https://arxiv.org/pdf/2509.00806)  

**Abstract**: Large language models (LLMs) are increasingly evident for accurate question answering across various domains. However, rigorous evaluation of their performance on complex question-answering (QA) capabilities is essential before deployment in real-world biomedical and healthcare applications. This paper presents our approach to the MedHopQA track of the BioCreative IX shared task, which focuses on multi-hop biomedical question answering involving diseases, genes, and chemicals. We adopt a supervised fine-tuning strategy leveraging LLaMA 3 8B, enhanced with a curated biomedical question-answer dataset compiled from external sources including BioASQ, MedQuAD, and TREC. Three experimental setups are explored: fine-tuning on combined short and long answers, short answers only, and long answers only. While our models demonstrate strong domain understanding, achieving concept-level accuracy scores of up to 0.8, their Exact Match (EM) scores remain significantly lower, particularly in the test phase. We introduce a two-stage inference pipeline for precise short-answer extraction to mitigate verbosity and improve alignment with evaluation metrics. Despite partial improvements, challenges persist in generating strictly formatted outputs. Our findings highlight the gap between semantic understanding and exact answer evaluation in biomedical LLM applications, motivating further research in output control and post-processing strategies. 

---
# Compiling Prompts, Not Crafting Them: A Reproducible Workflow for AI-Assisted Evidence Synthesis 

**Authors**: Teo Susnjak  

**Link**: [PDF](https://arxiv.org/pdf/2509.00038)  

**Abstract**: Large language models (LLMs) offer significant potential to accelerate systematic literature reviews (SLRs), yet current approaches often rely on brittle, manually crafted prompts that compromise reliability and reproducibility. This fragility undermines scientific confidence in LLM-assisted evidence synthesis. In response, this work adapts recent advances in declarative prompt optimisation, developed for general-purpose LLM applications, and demonstrates their applicability to the domain of SLR automation. This research proposes a structured, domain-specific framework that embeds task declarations, test suites, and automated prompt tuning into a reproducible SLR workflow. These emerging methods are translated into a concrete blueprint with working code examples, enabling researchers to construct verifiable LLM pipelines that align with established principles of transparency and rigour in evidence synthesis. This is a novel application of such approaches to SLR pipelines. 

---
# MultiStream-LLM: Bridging Modalities for Robust Sign Language Translation 

**Authors**: Marshall Thomas, Edward Fish, Richard Bowden  

**Link**: [PDF](https://arxiv.org/pdf/2509.00030)  

**Abstract**: Despite progress in gloss-free Sign Language Translation (SLT), monolithic end-to-end models consistently fail on two critical components of natural signing: the precise recognition of high-speed fingerspelling and the integration of asynchronous non-manual cues from the face. Recent progress in Automated Sign Language Translation with Large Language Models has side stepped this challenge, forcing a single network to learn these simultaneously resulting in poor performance when tasked with translating crucial information such as names,places, and technical terms. We introduce MultiStream-LLM, a modular framework designed to overcome these limitations. Our approach employs separate, specialized predictors for continuous signing, fingerspelling, and lipreading. Each expert network first decodes its specific modality into a sequence of tokens. These parallel streams are then fused by a lightweight transformer that resolves temporal misalignments before passing the combined representation to a Large Language Model (LLM) for final sentence generation. Our method establishes a new state-of-the-art on the How2Sign benchmark with a BLEU-4 score of 23.5 and achieves 73.2% letter accuracy on the challenging ChicagoFSWildPlus fingerspelling dataset. These results validate our core hypothesis: by isolating and solving distinct recogni tion tasks before fusion, our multi-expert approach provides a more powerful and effective pathway to robust, high-fidelity sign language translation. 

---
# Explainable Chain-of-Thought Reasoning: An Empirical Analysis on State-Aware Reasoning Dynamics 

**Authors**: Sheldon Yu, Yuxin Xiong, Junda Wu, Xintong Li, Tong Yu, Xiang Chen, Ritwik Sinha, Jingbo Shang, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2509.00190)  

**Abstract**: Recent advances in chain-of-thought (CoT) prompting have enabled large language models (LLMs) to perform multi-step reasoning. However, the explainability of such reasoning remains limited, with prior work primarily focusing on local token-level attribution, such that the high-level semantic roles of reasoning steps and their transitions remain underexplored. In this paper, we introduce a state-aware transition framework that abstracts CoT trajectories into structured latent dynamics. Specifically, to capture the evolving semantics of CoT reasoning, each reasoning step is represented via spectral analysis of token-level embeddings and clustered into semantically coherent latent states. To characterize the global structure of reasoning, we model their progression as a Markov chain, yielding a structured and interpretable view of the reasoning process. This abstraction supports a range of analyses, including semantic role identification, temporal pattern visualization, and consistency evaluation. 

---
# How Real Is AI Tutoring? Comparing Simulated and Human Dialogues in One-on-One Instruction 

**Authors**: Ruijia Li, Yuan-Hao Jiang, Jiatong Wang, Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01914)  

**Abstract**: Heuristic and scaffolded teacher-student dialogues are widely regarded as critical for fostering students' higher-order thinking and deep learning. However, large language models (LLMs) currently face challenges in generating pedagogically rich interactions. This study systematically investigates the structural and behavioral differences between AI-simulated and authentic human tutoring dialogues. We conducted a quantitative comparison using an Initiation-Response-Feedback (IRF) coding scheme and Epistemic Network Analysis (ENA). The results show that human dialogues are significantly superior to their AI counterparts in utterance length, as well as in questioning (I-Q) and general feedback (F-F) behaviors. More importantly, ENA results reveal a fundamental divergence in interactional patterns: human dialogues are more cognitively guided and diverse, centered around a "question-factual response-feedback" teaching loop that clearly reflects pedagogical guidance and student-driven thinking; in contrast, simulated dialogues exhibit a pattern of structural simplification and behavioral convergence, revolving around an "explanation-simplistic response" loop that is essentially a simple information transfer between the teacher and student. These findings illuminate key limitations in current AI-generated tutoring and provide empirical guidance for designing and evaluating more pedagogically effective generative educational dialogue systems. 

---
# LLM-Guided Semantic Relational Reasoning for Multimodal Intent Recognition 

**Authors**: Qianrui Zhou, Hua Xu, Yifan Wang, Xinzhi Dong, Hanlei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01337)  

**Abstract**: Understanding human intents from multimodal signals is critical for analyzing human behaviors and enhancing human-machine interactions in real-world scenarios. However, existing methods exhibit limitations in their modality-level reliance, constraining relational reasoning over fine-grained semantics for complex intent understanding. This paper proposes a novel LLM-Guided Semantic Relational Reasoning (LGSRR) method, which harnesses the expansive knowledge of large language models (LLMs) to establish semantic foundations that boost smaller models' relational reasoning performance. Specifically, an LLM-based strategy is proposed to extract fine-grained semantics as guidance for subsequent reasoning, driven by a shallow-to-deep Chain-of-Thought (CoT) that autonomously uncovers, describes, and ranks semantic cues by their importance without relying on manually defined priors. Besides, we formally model three fundamental types of semantic relations grounded in logical principles and analyze their nuanced interplay to enable more effective relational reasoning. Extensive experiments on multimodal intent and dialogue act recognition tasks demonstrate LGSRR's superiority over state-of-the-art methods, with consistent performance gains across diverse semantic understanding scenarios. The complete data and code are available at this https URL. 

---
# ShortageSim: Simulating Drug Shortages under Information Asymmetry 

**Authors**: Mingxuan Cui, Yilan Jiang, Duo Zhou, Cheng Qian, Yuji Zhang, Qiong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01813)  

**Abstract**: Drug shortages pose critical risks to patient care and healthcare systems worldwide, yet the effectiveness of regulatory interventions remains poorly understood due to fundamental information asymmetries in pharmaceutical supply chains. We present \textbf{ShortageSim}, the first Large Language Model (LLM)-based multi-agent simulation framework that captures the complex, strategic interactions between drug manufacturers, institutional buyers, and regulatory agencies in response to shortage alerts. Unlike traditional game-theoretic models that assume perfect rationality and complete information, \textbf{ShortageSim} leverages LLMs to simulate bounded-rational decision-making under uncertainty. Through a sequential production game spanning multiple quarters, we model how FDA announcements, both reactive alerts about existing shortages and proactive warnings about potential disruptions, propagate through the supply chain and influence capacity investment and procurement decisions. Our experiments on historical shortage events reveal that \textbf{ShortageSim} reduces the resolution-lag percentage for discontinued-disclosed cases by 83\%, bringing simulated durations more aligned to ground truth than the zero-shot baseline. We open-source \textbf{ShortageSim} and a dataset of 2,925 FDA shortage events at this https URL, providing a novel computational framework for designing and testing interventions in complex, information-scarce supply chains. 

---
# An LLM-enabled semantic-centric framework to consume privacy policies 

**Authors**: Rui Zhao, Vladyslav Melnychuk, Jun Zhao, Jesse Wright, Nigel Shadbolt  

**Link**: [PDF](https://arxiv.org/pdf/2509.01716)  

**Abstract**: In modern times, people have numerous online accounts, but they rarely read the Terms of Service or Privacy Policy of those sites, despite claiming otherwise, due to the practical difficulty in comprehending them. The mist of data privacy practices forms a major barrier for user-centred Web approaches, and for data sharing and reusing in an agentic world. Existing research proposed methods for using formal languages and reasoning for verifying the compliance of a specified policy, as a potential cure for ignoring privacy policies. However, a critical gap remains in the creation or acquisition of such formal policies at scale. We present a semantic-centric approach for using state-of-the-art large language models (LLM), to automatically identify key information about privacy practices from privacy policies, and construct $\mathit{Pr}^2\mathit{Graph}$, knowledge graph with grounding from Data Privacy Vocabulary (DPV) for privacy practices, to support downstream tasks. Along with the pipeline, the $\mathit{Pr}^2\mathit{Graph}$ for the top-100 popular websites is also released as a public resource, by using the pipeline for analysis. We also demonstrate how the $\mathit{Pr}^2\mathit{Graph}$ can be used to support downstream tasks by constructing formal policy representations such as Open Digital Right Language (ODRL) or perennial semantic Data Terms of Use (psDToU). To evaluate the technology capability, we enriched the Policy-IE dataset by employing legal experts to create custom annotations. We benchmarked the performance of different large language models for our pipeline and verified their capabilities. Overall, they shed light on the possibility of large-scale analysis of online services' privacy practices, as a promising direction to audit the Web and the Internet. We release all datasets and source code as public resources to facilitate reuse and improvement. 

---
# Analysis of Error Sources in LLM-based Hypothesis Search for Few-Shot Rule Induction 

**Authors**: Aishni Parab, Hongjing Lu, Ying Nian Wu, Sumit Gulwani  

**Link**: [PDF](https://arxiv.org/pdf/2509.01016)  

**Abstract**: Inductive reasoning enables humans to infer abstract rules from limited examples and apply them to novel situations. In this work, we compare an LLM-based hypothesis search framework with direct program generation approaches on few-shot rule induction tasks. Our findings show that hypothesis search achieves performance comparable to humans, while direct program generation falls notably behind. An error analysis reveals key bottlenecks in hypothesis generation and suggests directions for advancing program induction methods. Overall, this paper underscores the potential of LLM-based hypothesis search for modeling inductive reasoning and the challenges in building more efficient systems. 

---
# Oyster-I: Beyond Refusal -- Constructive Safety Alignment for Responsible Language Models 

**Authors**: Ranjie Duan, Jiexi Liu, Xiaojun Jia, Shiji Zhao, Ruoxi Cheng, Fengxiang Wang, Cheng Wei, Yong Xie, Chang Liu, Defeng Li, Yinpeng Dong, Yichi Zhang, Yuefeng Chen, Chongwen Wang, Xingjun Ma, Xingxing Wei, Yang Liu, Hang Su, Jun Zhu, Xinfeng Li, Yitong Sun, Jie Zhang, Jinzhao Hu, Sha Xu, Yitong Yang, Jialing Tao, Hui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2509.01909)  

**Abstract**: Large language models (LLMs) typically deploy safety mechanisms to prevent harmful content generation. Most current approaches focus narrowly on risks posed by malicious actors, often framing risks as adversarial events and relying on defensive refusals. However, in real-world settings, risks also come from non-malicious users seeking help while under psychological distress (e.g., self-harm intentions). In such cases, the model's response can strongly influence the user's next actions. Simple refusals may lead them to repeat, escalate, or move to unsafe platforms, creating worse outcomes. We introduce Constructive Safety Alignment (CSA), a human-centric paradigm that protects against malicious misuse while actively guiding vulnerable users toward safe and helpful results. Implemented in Oyster-I (Oy1), CSA combines game-theoretic anticipation of user reactions, fine-grained risk boundary discovery, and interpretable reasoning control, turning safety into a trust-building process. Oy1 achieves state-of-the-art safety among open models while retaining high general capabilities. On our Constructive Benchmark, it shows strong constructive engagement, close to GPT-5, and unmatched robustness on the Strata-Sword jailbreak dataset, nearing GPT-o1 levels. By shifting from refusal-first to guidance-first safety, CSA redefines the model-user relationship, aiming for systems that are not just safe, but meaningfully helpful. We release Oy1, code, and the benchmark to support responsible, user-centered AI. 

---
# Self-Exploring Language Models for Explainable Link Forecasting on Temporal Graphs via Reinforcement Learning 

**Authors**: Zifeng Ding, Shenyang Huang, Zeyu Cao, Emma Kondrup, Zachary Yang, Xingyue Huang, Yuan Sui, Zhangdie Yuan, Yuqicheng Zhu, Xianglong Hu, Yuan He, Farimah Poursafaei, Michael Bronstein, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2509.00975)  

**Abstract**: Forecasting future links is a central task in temporal graph (TG) reasoning, requiring models to leverage historical interactions to predict upcoming ones. Traditional neural approaches, such as temporal graph neural networks, achieve strong performance but lack explainability and cannot be applied to unseen graphs without retraining. Recent studies have begun to explore using large language models (LLMs) for graph reasoning, but most of them are constrained to static graphs or small synthetic TGs and lack the evaluation of the quality of reasoning traces generated by LLMs. In this work, we present Reasoning-Enhanced Learning for Temporal Graphs (ReaL-TG), a reinforcement learning framework that fine-tunes LLMs to perform explainable link forecasting on real-world TGs. ReaL-TG uses outcome-based reward to encourage models to self-explore reasoning strategies from graph structure and to produce explanations that directly justify their predictions. To enable evaluation on LLM-generated reasoning traces, we propose a new evaluation protocol combining ranking metrics with an LLM-as-a-Judge system that assesses both the quality of reasoning and the impact of hallucinations. Experiments with ReaL-TG-4B, obtained by fine-tuning Qwen3-4B under our framework, show that it outperforms much larger frontier LLMs, including GPT-5 mini, on ranking metrics, while producing high-quality explanations confirmed by both the LLM judge and human evaluation. 

---
# GradeSQL: Outcome Reward Models for Ranking SQL Queries from Large Language Models 

**Authors**: Mattia Tritto, Giuseppe Farano, Dario Di Palma, Gaetano Rossiello, Fedelucio Narducci, Dharmashankar Subramanian, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2509.01308)  

**Abstract**: Text-to-SQL, the task of translating natural language questions into SQL queries, has significantly advanced with the introduction of Large Language Models (LLMs), broadening database accessibility for a wide range of users. Despite substantial progress in generating valid SQL, current LLMs still struggle with complex queries that require precise alignment between user intent and the database schema. To mitigate this, test-time strategies such as Best-of-N (BoN) and Majority Voting (Maj) are often employed, based on the assumption that LLMs can generate correct answers but may require multiple attempts. However, these methods rely on surface-level heuristics, selecting either the syntactically correct query through execution-based BoN (ex-BoN) or the most frequently generated query with Maj. Recently, Outcome Reward Models (ORMs), which assign utility scores to generated outputs based on semantic correctness, have emerged as a promising approach for better aligning model predictions with user intent. Nevertheless, their application to Text-to-SQL remains largely underexplored.
In this work, we evaluate ORMs as an effective heuristic for BoN, compare them with ex-BoN and Maj, and introduce a framework for training ORMs for the Text-to-SQL task. We evaluate our ORMs on the BIRD and SPIDER benchmarks, finetuning various open-source LLMs, including the Qwen2, Granite3, and Llama3 model families. Our results show that ORMs outperform ex-BoN and Maj, achieving execution accuracy gains of +4.33% (BIRD) and +2.10% (Spider) over ex-BoN, and +2.91% (BIRD) and +0.93% (Spider) over Maj. We further demonstrate that finetuning models already aligned with SQL generation, such as OmniSQL, yields superior ORM performance. Additionally, we observe that ORMs achieve competitive results on simple queries and benefit more from an increased number of candidates compared to ex-BoN and Maj. 

---
# ChatCLIDS: Simulating Persuasive AI Dialogues to Promote Closed-Loop Insulin Adoption in Type 1 Diabetes Care 

**Authors**: Zonghai Yao, Talha Chafekar, Junda Wang, Shuo Han, Feiyun Ouyang, Junhui Qian, Lingxi Li, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00891)  

**Abstract**: Real-world adoption of closed-loop insulin delivery systems (CLIDS) in type 1 diabetes remains low, driven not by technical failure, but by diverse behavioral, psychosocial, and social barriers. We introduce ChatCLIDS, the first benchmark to rigorously evaluate LLM-driven persuasive dialogue for health behavior change. Our framework features a library of expert-validated virtual patients, each with clinically grounded, heterogeneous profiles and realistic adoption barriers, and simulates multi-turn interactions with nurse agents equipped with a diverse set of evidence-based persuasive strategies. ChatCLIDS uniquely supports longitudinal counseling and adversarial social influence scenarios, enabling robust, multi-dimensional evaluation. Our findings reveal that while larger and more reflective LLMs adapt strategies over time, all models struggle to overcome resistance, especially under realistic social pressure. These results highlight critical limitations of current LLMs for behavior change, and offer a high-fidelity, scalable testbed for advancing trustworthy persuasive AI in healthcare and beyond. 

---
# Aligning Reasoning LLMs for Materials Discovery with Physics-aware Rejection Sampling 

**Authors**: Lee Hyun, Sohee Yoon, Jinwoo Park, Sue In Chae, Seongeon Park, Jooyeon Ahn, Yebin Jung, Youjung Chung, Hogeun Chang, Myeonginn Kang, Jina Kim, Ho-Gyeong Kim, Myeonghun Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2509.00768)  

**Abstract**: AI-driven materials discovery that couples automated experimentation with algorithmic decision-making requires process aware recipe to property predictors that are accurate, calibrated, and physically admissible. We approach this as a reasoning problem with large reasoning models (LRMs). To instill reasoning capability into language models, we curate reasoning traces from a teacher model to train a student model. However, most training pipelines select reasoning traces using binary correctness or learned preference signals that poorly reflect physical admissibility. We introduce Physics-aware Rejection Sampling (PaRS), a training-time trace selection scheme that favors traces consistent with fundamental physics and numerically close to targets, with lightweight halting to control compute. We instantiate our framework with a large student model fine-tuned on traces synthesized by a larger teacher model, and evaluate under matched token budgets against various rejection sampling baselines. Our method improves accuracy and calibration, reduces physics-violation rates, and lowers sampling cost relative to baselines. These results indicate that modest, domain-aware constraints combined with trace-level selection provide a practical path toward reliable, efficient LRMs for process-aware property prediction and closed-loop materials design. 

---
# LLM-Assisted Iterative Evolution with Swarm Intelligence Toward SuperBrain 

**Authors**: Li Weigang, Pedro Carvalho Brom, Lucas Ramson Siefert  

**Link**: [PDF](https://arxiv.org/pdf/2509.00510)  

**Abstract**: We propose a novel SuperBrain framework for collective intelligence, grounded in the co-evolution of large language models (LLMs) and human users. Unlike static prompt engineering or isolated agent simulations, our approach emphasizes a dynamic pathway from Subclass Brain to Superclass Brain: (1) A Subclass Brain arises from persistent, personalized interaction between a user and an LLM, forming a cognitive dyad with adaptive learning memory. (2) Through GA-assisted forward-backward evolution, these dyads iteratively refine prompts and task performance. (3) Multiple Subclass Brains coordinate via Swarm Intelligence, optimizing across multi-objective fitness landscapes and exchanging distilled heuristics. (4) Their standardized behaviors and cognitive signatures integrate into a Superclass Brain, an emergent meta-intelligence capable of abstraction, generalization and self-improvement. We outline the theoretical constructs, present initial implementations (e.g., UAV scheduling, KU/KI keyword filtering) and propose a registry for cross-dyad knowledge consolidation. This work provides both a conceptual foundation and an architectural roadmap toward scalable, explainable and ethically aligned collective AI. 

---
# Ensemble Debates with Local Large Language Models for AI Alignment 

**Authors**: Ephraiem Sarabamoun  

**Link**: [PDF](https://arxiv.org/pdf/2509.00091)  

**Abstract**: As large language models (LLMs) take on greater roles in high-stakes decisions, alignment with human values is essential. Reliance on proprietary APIs limits reproducibility and broad participation. We study whether local open-source ensemble debates can improve alignmentoriented reasoning. Across 150 debates spanning 15 scenarios and five ensemble configurations, ensembles outperform single-model baselines on a 7-point rubric (overall: 3.48 vs. 3.13), with the largest gains in reasoning depth (+19.4%) and argument quality (+34.1%). Improvements are strongest for truthfulness (+1.25 points) and human enhancement (+0.80). We provide code, prompts, and a debate data set, providing an accessible and reproducible foundation for ensemble-based alignment evaluation. 

---
# Learning to Refine: Self-Refinement of Parallel Reasoning in LLMs 

**Authors**: Qibin Wang, Pu Zhao, Shaohan Huang, Fangkai Yang, Lu Wang, Furu Wei, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00084)  

**Abstract**: To further enhance the ability of Large Language Models (LLMs) to solve complex, multi-step reasoning problems, test-time scaling (TTS) methods have gained widespread attention. Existing approaches such as Best-of-N and majority voting are limited as their performance depends on the quality of candidate responses, making them unable to produce a correct solution when all candidates are incorrect. Introducing an additional model to select the best response also incurs significant deployment costs. To this end, we introduce Generative Self-Refinement (GSR), a novel parallel test-time scaling framework where a unified model first generates a set of candidate responses in parallel and then performs self-refinement to synthesize a new superior solution based on a prompt consisting of the problem and these candidates. However, LLMs struggle to perform refinement effectively when prompted directly. Therefore, we design a hybrid training pipeline by jointly optimizing for two complementary objectives, solving problems directly and refining candidate responses. Experimental results demonstrate that our method achieves state-of-the-art performance across five mathematical benchmarks. We further show that this learned self-refinement skill is a model-agnostic enhancement, robust across different model scales and generalizing to out-of-distribution reasoning tasks. 

---
# KG-RAG: Enhancing GUI Agent Decision-Making via Knowledge Graph-Driven Retrieval-Augmented Generation 

**Authors**: Ziyi Guan, Jason Chun Lok Li, Zhijian Hou, Pingping Zhang, Donglai Xu, Yuzhi Zhao, Mengyang Wu, Jinpeng Chen, Thanh-Toan Nguyen, Pengfei Xian, Wenao Ma, Shengchao Qin, Graziano Chesi, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2509.00366)  

**Abstract**: Despite recent progress, Graphic User Interface (GUI) agents powered by Large Language Models (LLMs) struggle with complex mobile tasks due to limited app-specific knowledge. While UI Transition Graphs (UTGs) offer structured navigation representations, they are underutilized due to poor extraction and inefficient integration. We introduce KG-RAG, a Knowledge Graph-driven Retrieval-Augmented Generation framework that transforms fragmented UTGs into structured vector databases for efficient real-time retrieval. By leveraging an intent-guided LLM search method, KG-RAG generates actionable navigation paths, enhancing agent decision-making. Experiments across diverse mobile apps show that KG-RAG outperforms existing methods, achieving a 75.8% success rate (8.9% improvement over AutoDroid), 84.6% decision accuracy (8.1% improvement), and reducing average task steps from 4.5 to 4.1. Additionally, we present KG-Android-Bench and KG-Harmony-Bench, two benchmarks tailored to the Chinese mobile ecosystem for future research. Finally, KG-RAG transfers to web/desktop (+40% SR on Weibo-web; +20% on QQ Music-desktop), and a UTG cost ablation shows accuracy saturates at ~4h per complex app, enabling practical deployment trade-offs. 

---
# Pruning Weights but Not Truth: Safeguarding Truthfulness While Pruning LLMs 

**Authors**: Yao Fu, Runchao Li, Xianxuan Long, Haotian Yu, Xiaotian Han, Yu Yin, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.00096)  

**Abstract**: Neural network pruning has emerged as a promising approach for deploying LLMs in low-resource scenarios while preserving downstream task performance. However, for the first time, we reveal that such pruning disrupts LLMs' internal activation features crucial for lie detection, where probing classifiers (typically small logistic regression models) trained on these features assess the truthfulness of LLM-generated statements. This discovery raises a crucial open question: how can we prune LLMs without sacrificing these critical lie detection capabilities? Our investigation further reveals that naively adjusting layer-wise pruning sparsity based on importance inadvertently removes crucial weights, failing to improve lie detection performance despite its reliance on the most crucial LLM layer. To address this issue, we propose Truthful Pruning aligned by Layer-wise Outliers (TPLO), which places greater emphasis on layers with more activation outliers and stronger discriminative features simultaneously. This preserves LLMs' original performance while retaining critical features of inner states needed for robust lie detection. Moreover, we introduce a prompting rule to enrich the TruthfulQA benchmark for better calibrating LLM pruning. Empirical results show that our approach improves the hallucination detection for pruned LLMs (achieving 88% accuracy at 50% sparsity) and enhances their performance on TruthfulQA. 

---
# Traj-MLLM: Can Multimodal Large Language Models Reform Trajectory Data Mining? 

**Authors**: Shuo Liu, Di Yao, Yan Lin, Gao Cong, Jingping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2509.00053)  

**Abstract**: Building a general model capable of analyzing human trajectories across different geographic regions and different tasks becomes an emergent yet important problem for various applications. However, existing works suffer from the generalization problem, \ie, they are either restricted to train for specific regions or only suitable for a few tasks. Given the recent advances of multimodal large language models (MLLMs), we raise the question: can MLLMs reform current trajectory data mining and solve the problem? Nevertheless, due to the modality gap of trajectory, how to generate task-independent multimodal trajectory representations and how to adapt flexibly to different tasks remain the foundational challenges. In this paper, we propose \texttt{Traj-MLLM}}, which is the first general framework using MLLMs for trajectory data mining. By integrating multiview contexts, \texttt{Traj-MLLM}} transforms raw trajectories into interleaved image-text sequences while preserving key spatial-temporal characteristics, and directly utilizes the reasoning ability of MLLMs for trajectory analysis. Additionally, a prompt optimization method is proposed to finalize data-invariant prompts for task adaptation. Extensive experiments on four publicly available datasets show that \texttt{Traj-MLLM}} outperforms state-of-the-art baselines by $48.05\%$, $15.52\%$, $51.52\%$, $1.83\%$ on travel time estimation, mobility prediction, anomaly detection and transportation mode identification, respectively. \texttt{Traj-MLLM}} achieves these superior performances without requiring any training data or fine-tuning the MLLM backbones. 

---
# ChipChat: Low-Latency Cascaded Conversational Agent in MLX 

**Authors**: Tatiana Likhomanenko, Luke Carlson, Richard He Bai, Zijin Gu, Han Tran, Zakaria Aldeneh, Yizhe Zhang, Ruixiang Zhang, Huangjie Zheng, Navdeep Jaitly  

**Link**: [PDF](https://arxiv.org/pdf/2509.00078)  

**Abstract**: The emergence of large language models (LLMs) has transformed spoken dialog systems, yet the optimal architecture for real-time on-device voice agents remains an open question. While end-to-end approaches promise theoretical advantages, cascaded systems (CSs) continue to outperform them in language understanding tasks, despite being constrained by sequential processing latency. In this work, we introduce ChipChat, a novel low-latency CS that overcomes traditional bottlenecks through architectural innovations and streaming optimizations. Our system integrates streaming (a) conversational speech recognition with mixture-of-experts, (b) state-action augmented LLM, (c) text-to-speech synthesis, (d) neural vocoder, and (e) speaker modeling. Implemented using MLX, ChipChat achieves sub-second response latency on a Mac Studio without dedicated GPUs, while preserving user privacy through complete on-device processing. Our work shows that strategically redesigned CSs can overcome their historical latency limitations, offering a promising path forward for practical voice-based AI agents. 

---
# When Agents go Astray: Course-Correcting SWE Agents with PRMs 

**Authors**: Shubham Gandhi, Jason Tsay, Jatin Ganhotra, Kiran Kate, Yara Rizk  

**Link**: [PDF](https://arxiv.org/pdf/2509.02360)  

**Abstract**: Large Language Model (LLM) agents are increasingly deployed for complex, multi-step software engineering (SWE) tasks. However, their trajectories often contain costly inefficiencies, such as redundant exploration, looping, and failure to terminate once a solution is reached. Prior work has largely treated these errors in a post-hoc manner, diagnosing failures only after execution. In this paper, we introduce SWE-PRM, an inference-time Process Reward Model (PRM) that intervenes during execution to detect and course-correct trajectory-level errors. Our PRM design leverages a taxonomy of common inefficiencies and delivers lightweight, interpretable feedback without modifying the underlying policy. On SWE-bench Verified, closed-source PRMs improve resolution from 40.0% to 50.6% (+10.6 p.p.), with the largest gains on medium and hard tasks. Among feedback strategies, taxonomy-guided PRMs outperform unguided or explicit action-prescriptive variants, increasing success rate while reducing trajectory length. These benefits come at an acceptable added inference cost of as low as $0.2, making PRMs a practical and scalable mechanism for improving SWE agents' reliability and efficiency. 

---
# Towards Agents That Know When They Don't Know: Uncertainty as a Control Signal for Structured Reasoning 

**Authors**: Josefa Lia Stoisser, Marc Boubnovski Martell, Lawrence Phillips, Gianluca Mazzoni, Lea Mørch Harder, Philip Torr, Jesper Ferkinghoff-Borg, Kaspar Martens, Julien Fauqueur  

**Link**: [PDF](https://arxiv.org/pdf/2509.02401)  

**Abstract**: Large language model (LLM) agents are increasingly deployed in structured biomedical data environments, yet they often produce fluent but overconfident outputs when reasoning over complex multi-table data. We introduce an uncertainty-aware agent for query-conditioned multi-table summarization that leverages two complementary signals: (i) retrieval uncertainty--entropy over multiple table-selection rollouts--and (ii) summary uncertainty--combining self-consistency and perplexity. Summary uncertainty is incorporated into reinforcement learning (RL) with Group Relative Policy Optimization (GRPO), while both retrieval and summary uncertainty guide inference-time filtering and support the construction of higher-quality synthetic datasets.
On multi-omics benchmarks, our approach improves factuality and calibration, nearly tripling correct and useful claims per summary (3.0\(\rightarrow\)8.4 internal; 3.6\(\rightarrow\)9.9 cancer multi-omics) and substantially improving downstream survival prediction (C-index 0.32\(\rightarrow\)0.63). These results demonstrate that uncertainty can serve as a control signal--enabling agents to abstain, communicate confidence, and become more reliable tools for complex structured-data environments. 

---
# Re-evaluating LLM-based Heuristic Search: A Case Study on the 3D Packing Problem 

**Authors**: Guorui Quan, Mingfei Sun, Manuel López-Ibáñez  

**Link**: [PDF](https://arxiv.org/pdf/2509.02297)  

**Abstract**: The art of heuristic design has traditionally been a human pursuit. While Large Language Models (LLMs) can generate code for search heuristics, their application has largely been confined to adjusting simple functions within human-crafted frameworks, leaving their capacity for broader innovation an open question. To investigate this, we tasked an LLM with building a complete solver for the constrained 3D Packing Problem. Direct code generation quickly proved fragile, prompting us to introduce two supports: constraint scaffolding--prewritten constraint-checking code--and iterative self-correction--additional refinement cycles to repair bugs and produce a viable initial population. Notably, even within a vast search space in a greedy process, the LLM concentrated its efforts almost exclusively on refining the scoring function. This suggests that the emphasis on scoring functions in prior work may reflect not a principled strategy, but rather a natural limitation of LLM capabilities. The resulting heuristic was comparable to a human-designed greedy algorithm, and when its scoring function was integrated into a human-crafted metaheuristic, its performance rivaled established solvers, though its effectiveness waned as constraints tightened. Our findings highlight two major barriers to automated heuristic design with current LLMs: the engineering required to mitigate their fragility in complex reasoning tasks, and the influence of pretrained biases, which can prematurely narrow the search for novel solutions. 

---
# LLMs for LLMs: A Structured Prompting Methodology for Long Legal Documents 

**Authors**: Strahinja Klem, Noura Al Moubayed  

**Link**: [PDF](https://arxiv.org/pdf/2509.02241)  

**Abstract**: The rise of Large Language Models (LLMs) has had a profoundly transformative effect on a number of fields and domains. However, their uptake in Law has proven more challenging due to the important issues of reliability and transparency. In this study, we present a structured prompting methodology as a viable alternative to the often expensive fine-tuning, with the capability of tacking long legal documents from the CUAD dataset on the task of information retrieval. Each document is first split into chunks via a system of chunking and augmentation, addressing the long document problem. Then, alongside an engineered prompt, the input is fed into QWEN-2 to produce a set of answers for each question. Finally, we tackle the resulting candidate selection problem with the introduction of the Distribution-based Localisation and Inverse Cardinality Weighting heuristics. This approach leverages a general purpose model to promote long term scalability, prompt engineering to increase reliability and the two heuristic strategies to reduce the impact of the black box effect. Whilst our model performs up to 9\% better than the previously presented method, reaching state-of-the-art performance, it also highlights the limiting factor of current automatic evaluation metrics for question answering, serving as a call to action for future research. However, the chief aim of this work is to underscore the potential of structured prompt engineering as a useful, yet under-explored, tool in ensuring accountability and responsibility of AI in the legal domain, and beyond. 

---
# GridMind: LLMs-Powered Agents for Power System Analysis and Operations 

**Authors**: Hongwei Jin, Kibaek Kim, Jonghwan Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2509.02494)  

**Abstract**: The complexity of traditional power system analysis workflows presents significant barriers to efficient decision-making in modern electric grids. This paper presents GridMind, a multi-agent AI system that integrates Large Language Models (LLMs) with deterministic engineering solvers to enable conversational scientific computing for power system analysis. The system employs specialized agents coordinating AC Optimal Power Flow and N-1 contingency analysis through natural language interfaces while maintaining numerical precision via function calls. GridMind addresses workflow integration, knowledge accessibility, context preservation, and expert decision-support augmentation. Experimental evaluation on IEEE test cases demonstrates that the proposed agentic framework consistently delivers correct solutions across all tested language models, with smaller LLMs achieving comparable analytical accuracy with reduced computational latency. This work establishes agentic AI as a viable paradigm for scientific computing, demonstrating how conversational interfaces can enhance accessibility while preserving numerical rigor essential for critical engineering applications. 

---
# Throttling Web Agents Using Reasoning Gates 

**Authors**: Abhinav Kumar, Jaechul Roh, Ali Naseh, Amir Houmansadr, Eugene Bagdasarian  

**Link**: [PDF](https://arxiv.org/pdf/2509.01619)  

**Abstract**: AI web agents use Internet resources at far greater speed, scale, and complexity -- changing how users and services interact. Deployed maliciously or erroneously, these agents could overload content providers. At the same time, web agents can bypass CAPTCHAs and other defenses by mimicking user behavior or flood authentication systems with fake accounts. Yet providers must protect their services and content from denial-of-service attacks and scraping by web agents. In this paper, we design a framework that imposes tunable costs on agents before providing access to resources; we call this Web Agent Throttling. We start by formalizing Throttling Gates as challenges issued to an agent that are asymmetric, scalable, robust, and compatible with any agent. Focusing on a common component -- the language model -- we require the agent to solve reasoning puzzles, thereby incurring excessive token-generation costs. However, we find that using existing puzzles, e.g., coding or math, as throttling gates fails to satisfy our properties. To address this, we introduce rebus-based Reasoning Gates, synthetic text puzzles that require multi-hop reasoning over world knowledge (thereby throttling an agent's model). We design a scalable generation and verification protocol for such reasoning gates. Our framework achieves computational asymmetry, i.e., the response-generation cost is 9.2x higher than the generation cost for SOTA models. We further deploy reasoning gates on a custom website and Model Context Protocol (MCP) servers and evaluate with real-world web agents. Finally, we discuss the limitations and environmental impact of real-world deployment of our framework. 

---
# An Epidemiological Knowledge Graph extracted from the World Health Organization's Disease Outbreak News 

**Authors**: Sergio Consoli, Pietro Coletti, Peter V. Markov, Lia Orfei, Indaco Biazzo, Lea Schuh, Nicolas Stefanovitch, Lorenzo Bertolini, Mario Ceresa, Nikolaos I. Stilianakis  

**Link**: [PDF](https://arxiv.org/pdf/2509.02258)  

**Abstract**: The rapid evolution of artificial intelligence (AI), together with the increased availability of social media and news for epidemiological surveillance, are marking a pivotal moment in epidemiology and public health research. Leveraging the power of generative AI, we use an ensemble approach which incorporates multiple Large Language Models (LLMs) to extract valuable actionable epidemiological information from the World Health Organization (WHO) Disease Outbreak News (DONs). DONs is a collection of regular reports on global outbreaks curated by the WHO and the adopted decision-making processes to respond to them. The extracted information is made available in a daily-updated dataset and a knowledge graph, referred to as eKG, derived to provide a nuanced representation of the public health domain knowledge. We provide an overview of this new dataset and describe the structure of eKG, along with the services and tools used to access and utilize the data that we are building on top. These innovative data resources open altogether new opportunities for epidemiological research, and the analysis and surveillance of disease outbreaks. 

---
# Counterfactual Sensitivity for Faithful Reasoning in Language Models 

**Authors**: Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2509.01544)  

**Abstract**: Large language models (LLMs) often produce correct answers while relying on flawed or irrelevant reasoning traces, undermining their trustworthiness in high-stakes domains. We propose Counterfactual Sensitivity Regularization (CSR), a lightweight training objective that enforces dependence between intermediate reasoning and final outputs. CSR introduces automated, operator-level counterfactual interventions (e.g., swapping "+" with "-") during training and penalizes models that preserve the same answer under logically invalid traces. This requires only one additional forward pass per sample. To measure faithfulness, we introduce Counterfactual Outcome Sensitivity (COS), which quantifies the impact of such perturbations on model predictions. Across structured reasoning tasks - arithmetic (GSM8K), logical deduction (PrOntoQA), and planning (Blocks World) - CSR improves faithfulness by up to 70 percentage points over standard fine-tuning and process supervision, with only minor accuracy loss. The learned sensitivity generalizes to larger models and synergizes with inference-time methods such as self-consistency. A pilot study on HellaSwag further demonstrates that extending CSR with semantic perturbations can enhance faithfulness in commonsense reasoning. 

---
# Unraveling LLM Jailbreaks Through Safety Knowledge Neurons 

**Authors**: Chongwen Zhao, Kaizhu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01631)  

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation, a technique known as "Jailbreak." While some studies have achieved defenses against jailbreak attacks by modifying output distributions or detecting harmful content, the exact rationale still remains elusive. In this work, we present a novel neuron-level interpretability method that focuses on the role of safety-related knowledge neurons. Unlike existing approaches, our method projects the model's internal representation into a more consistent and interpretable vocabulary space. We then show that adjusting the activation of safety-related neurons can effectively control the model's behavior with a mean ASR higher than 97%. Building on this insight, we propose SafeTuning, a fine-tuning strategy that reinforces safety-critical neurons to improve model robustness against jailbreaks. SafeTuning consistently reduces attack success rates across multiple LLMs and outperforms all four baseline defenses. These findings offer a new perspective on understanding and defending against jailbreak attacks. 

---
# Communicative Agents for Slideshow Storytelling Video Generation based on LLMs 

**Authors**: Jingxing Fan, Jinrong Shen, Yusheng Yao, Shuangqing Wang, Qian Wang, Yuling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01277)  

**Abstract**: With the rapid advancement of artificial intelligence (AI), the proliferation of AI-generated content (AIGC) tasks has significantly accelerated developments in text-to-video generation. As a result, the field of video production is undergoing a transformative shift. However, conventional text-to-video models are typically constrained by high computational costs.
In this study, we propose Video-Generation-Team (VGTeam), a novel slide show video generation system designed to redefine the video creation pipeline through the integration of large language models (LLMs). VGTeam is composed of a suite of communicative agents, each responsible for a distinct aspect of video generation, such as scriptwriting, scene creation, and audio design. These agents operate collaboratively within a chat tower workflow, transforming user-provided textual prompts into coherent, slide-style narrative videos.
By emulating the sequential stages of traditional video production, VGTeam achieves remarkable improvements in both efficiency and scalability, while substantially reducing computational overhead. On average, the system generates videos at a cost of only $0.103, with a successful generation rate of 98.4%. Importantly, this framework maintains a high degree of creative fidelity and customization.
The implications of VGTeam are far-reaching. It democratizes video production by enabling broader access to high-quality content creation without the need for extensive resources. Furthermore, it highlights the transformative potential of language models in creative domains and positions VGTeam as a pioneering system for next-generation content creation. 

---
# LLM-empowered Agents Simulation Framework for Scenario Generation in Service Ecosystem Governance 

**Authors**: Deyu Zhou, Yuqi Hou, Xiao Xue, Xudong Lu, Qingzhong Li, Lizhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2509.01441)  

**Abstract**: As the social environment is growing more complex and collaboration is deepening, factors affecting the healthy development of service ecosystem are constantly changing and diverse, making its governance a crucial research issue. Applying the scenario analysis method and conducting scenario rehearsals by constructing an experimental system before managers make decisions, losses caused by wrong decisions can be largely avoided. However, it relies on predefined rules to construct scenarios and faces challenges such as limited information, a large number of influencing factors, and the difficulty of measuring social elements. These challenges limit the quality and efficiency of generating social and uncertain scenarios for the service ecosystem. Therefore, we propose a scenario generator design method, which adaptively coordinates three Large Language Model (LLM) empowered agents that autonomously optimize experimental schemes to construct an experimental system and generate high quality scenarios. Specifically, the Environment Agent (EA) generates social environment including extremes, the Social Agent (SA) generates social collaboration structure, and the Planner Agent (PA) couples task-role relationships and plans task solutions. These agents work in coordination, with the PA adjusting the experimental scheme in real time by perceiving the states of each agent and these generating scenarios. Experiments on the ProgrammableWeb dataset illustrate our method generates more accurate scenarios more efficiently, and innovatively provides an effective way for service ecosystem governance related experimental system construction. 

---
# Towards Open-World Retrieval-Augmented Generation on Knowledge Graph: A Multi-Agent Collaboration Framework 

**Authors**: Jiasheng Xu, Mingda Li, Yongqiang Tang, Peijie Wang, Wensheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01238)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in language understanding and reasoning. However, their dependence on static training corpora makes them prone to factual errors and knowledge gaps. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge sources, especially structured Knowledge Graphs (KGs), which provide explicit semantics and efficient retrieval. Existing KG-based RAG approaches, however, generally assume that anchor entities are accessible to initiate graph traversal, which limits their robustness in open world settings where accurate linking between the query and the entity is unreliable. To overcome this limitation, we propose AnchorRAG, a novel multi-agent collaboration framework for open-world RAG without the predefined anchor entities. Specifically, a predictor agent dynamically identifies candidate anchor entities by aligning user query terms with KG nodes and initializes independent retriever agents to conduct parallel multi-hop explorations from each candidate. Then a supervisor agent formulates the iterative retrieval strategy for these retriever agents and synthesizes the resulting knowledge paths to generate the final answer. This multi-agent collaboration framework improves retrieval robustness and mitigates the impact of ambiguous or erroneous anchors. Extensive experiments on four public benchmarks demonstrate that AnchorRAG significantly outperforms existing baselines and establishes new state-of-the-art results on the real-world question answering tasks. 

---
# Causal MAS: A Survey of Large Language Model Architectures for Discovery and Effect Estimation 

**Authors**: Adib Bazgir, Amir Habibdoust, Yuwen Zhang, Xing Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.00987)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various reasoning and generation tasks. However, their proficiency in complex causal reasoning, discovery, and estimation remains an area of active development, often hindered by issues like hallucination, reliance on spurious correlations, and difficulties in handling nuanced, domain-specific, or personalized causal relationships. Multi-agent systems, leveraging the collaborative or specialized abilities of multiple LLM-based agents, are emerging as a powerful paradigm to address these limitations. This review paper explores the burgeoning field of causal multi-agent LLMs. We examine how these systems are designed to tackle different facets of causality, including causal reasoning and counterfactual analysis, causal discovery from data, and the estimation of causal effects. We delve into the diverse architectural patterns and interaction protocols employed, from pipeline-based processing and debate frameworks to simulation environments and iterative refinement loops. Furthermore, we discuss the evaluation methodologies, benchmarks, and diverse application domains where causal multi-agent LLMs are making an impact, including scientific discovery, healthcare, fact-checking, and personalized systems. Finally, we highlight the persistent challenges, open research questions, and promising future directions in this synergistic field, aiming to provide a comprehensive overview of its current state and potential trajectory. 

---
# CoreThink: A Symbolic Reasoning Layer to reason over Long Horizon Tasks with LLMs 

**Authors**: Jay Vaghasiya, Omkar Ghugarkar, Vishvesh Bhat, Vipul Dholaria, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2509.00971)  

**Abstract**: We introduce CoreThink, a state-of-the-art Reasoning Layer built upon a novel reasoning method called General Symbolics. This approach diverges from reasoning paradigms such as test-time scaling, Supervised Fine-Tuning (SFT), and Reinforcement Learning with Verifiable Rewards (RLVR). CoreThink General Symbolic Reasoner (GSR) is specifically structured around three key use cases: tool-calling, code generation, and planning, demonstrating exemplary performance across a total of seven benchmarks in their respective areas. Notably, we are achieving SOTA scores of 66.66\% on Livecodebench v6, 89\% on Instruction-Following Evals, and 24.4\% on ARC-AGI-2. We also present an agentic coding IDE, developed using the principles of General Symbolics, which achieves a state-of-the-art accuracy of 62.3\% on \texttt{SWE-Bench Lite}. We are able to achieve these improvements without any finetuning or training costs. Our Reasoning Layer is designed to provide a pure performance uplift, ensuring that a model's accuracy on reasoning tasks is never negatively impacted. We argue that incumbent methods will eventually lead to diminishing returns in LLM performance, necessitating the development of new reasoning techniques. This technical report details our approach at a high level and the availability of the CoreThink models for reasoning-intensive use cases. 

---
# A Hybrid Ai Framework For Strategic Patent Portfolio Pruning: Integrating Learning To-Rank And Market Need Analysis For Technology Transfer Optimization 

**Authors**: Manish Verma, Vivek Sharma, Vishal Singh  

**Link**: [PDF](https://arxiv.org/pdf/2509.00958)  

**Abstract**: This paper introduces a novel, multi stage hybrid intelligence framework for pruning patent portfolios to identify high value assets for technology transfer. Current patent valuation methods often rely on retrospective indicators or manual, time intensive analysis. Our framework automates and deepens this process by combining a Learning to Rank (LTR) model, which evaluates patents against over 30 legal and commercial parameters, with a unique "Need-Seed" agent-based system. The "Need Agent" uses Natural Language Processing (NLP) to mine unstructured market and industry data, identifying explicit technological needs. Concurrently, the "Seed Agent" employs fine tuned Large Language Models (LLMs) to analyze patent claims and map their technological capabilities. The system generates a "Core Ontology Framework" that matches high potential patents (Seeds) to documented market demands (Needs), providing a strategic rationale for divestment decisions. We detail the architecture, including a dynamic parameter weighting system and a crucial Human in the-Loop (HITL) validation protocol, to ensure both adaptability and real-world credibility. 

---
# SATQuest: A Verifier for Logical Reasoning Evaluation and Reinforcement Fine-Tuning of LLMs 

**Authors**: Yanxiao Zhao, Yaqian Li, Zihao Bo, Rinyoichi Takezoe, Haojia Hui, Mo Guang, Lei Ren, Xiaolin Qin, Kaiwen Long  

**Link**: [PDF](https://arxiv.org/pdf/2509.00930)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated remarkable general reasoning capabilities. However, systematically evaluating and enhancing these reasoning capabilities is challenging due to the lack of controllable and scalable tools for fine-grained analysis. Existing benchmarks and datasets often lack the necessary variable control for multi-dimensional, systematic analysis and training, or have narrow problem types and formats. To address these limitations, we introduce SATQuest, a systematic verifier designed to evaluate and enhance logical reasoning in LLMs by generating diverse, Satisfiability-based logical reasoning problems directly from Conjunctive Normal Form (CNF) instances. SATQuest structures these problems along three orthogonal dimensions: instance scale, problem type, and question format, employing randomized, SAT-based problem generation and objective answer verification via PySAT. This design mitigates memorization issues, allows for nuanced insights into reasoning performance, and enables effective reinforcement fine-tuning. Our extensive evaluation of various LLMs using SATQuest identified significant limitations in their logical reasoning, particularly in generalizing beyond familiar mathematical formats. Furthermore, we show that reinforcement fine-tuning with SATQuest rewards substantially improves targeted task performance and generalizes to more complex instances, while highlighting remaining challenges in cross-format adaptation. Through these demonstrations, we showcase SATQuest's potential as a foundational tool and a valuable starting point for advancing LLM logical reasoning. 

---
# Supporting Our AI Overlords: Redesigning Data Systems to be Agent-First 

**Authors**: Shu Liu, Soujanya Ponnapalli, Shreya Shankar, Sepanta Zeighami, Alan Zhu, Shubham Agarwal, Ruiqi Chen, Samion Suwito, Shuo Yuan, Ion Stoica, Matei Zaharia, Alvin Cheung, Natacha Crooks, Joseph E. Gonzalez, Aditya G. Parameswaran  

**Link**: [PDF](https://arxiv.org/pdf/2509.00997)  

**Abstract**: Large Language Model (LLM) agents, acting on their users' behalf to manipulate and analyze data, are likely to become the dominant workload for data systems in the future. When working with data, agents employ a high-throughput process of exploration and solution formulation for the given task, one we call agentic speculation. The sheer volume and inefficiencies of agentic speculation can pose challenges for present-day data systems. We argue that data systems need to adapt to more natively support agentic workloads. We take advantage of the characteristics of agentic speculation that we identify, i.e., scale, heterogeneity, redundancy, and steerability - to outline a number of new research opportunities for a new agent-first data systems architecture, ranging from new query interfaces, to new query processing techniques, to new agentic memory stores. 

---
# Efficient Graph Understanding with LLMs via Structured Context Injection 

**Authors**: Govind Waghmare, Sumedh BG, Sonia Gupta, Srikanta Bedathur  

**Link**: [PDF](https://arxiv.org/pdf/2509.00740)  

**Abstract**: Large Language Models (LLMs) have shown strong capabilities in solving problems across domains, including graph-related tasks traditionally addressed by symbolic or algorithmic methods. In this work, we present a framework for structured context injection, where task-specific information is systematically embedded in the input to guide LLMs in solving a wide range of graph problems. Our method does not require fine-tuning of LLMs, making it cost-efficient and lightweight. We observe that certain graph reasoning tasks remain challenging for LLMs unless they are mapped to conceptually grounded representations. However, achieving such mappings through fine-tuning or repeated multi-step querying can be expensive and inefficient. Our approach offers a practical alternative by injecting structured context directly into the input, enabling the LLM to implicitly align the task with grounded conceptual spaces. We evaluate the approach on multiple graph tasks using both lightweight and large models, highlighting the trade-offs between accuracy and computational cost. The results demonstrate consistent performance improvements, showing that structured input context can rival or surpass more complex approaches. Our findings underscore the value of structured context injection as an effective and scalable strategy for graph understanding with LLMs. 

---
# Towards Agentic OS: An LLM Agent Framework for Linux Schedulers 

**Authors**: Yusheng Zheng, Yanpeng Hu, Wei Zhang, Andi Quinn  

**Link**: [PDF](https://arxiv.org/pdf/2509.01245)  

**Abstract**: Operating system schedulers suffer from a fundamental semantic gap, where kernel policies fail to understand application-specific needs, leading to suboptimal performance. We introduce SchedCP, the first framework that enables fully autonomous Large Language Model (LLM) agents to safely and efficiently optimize Linux schedulers without human involvement. Our core insight is that the challenge is not merely to apply a better LLM, but to architect a decoupled control plane that separates the AI's role of semantic reasoning ("what to optimize") from the system's role of execution ("how to observe and act"). Implemented as Model Context Protocol(MCP) server, SchedCP provides a stable interface with three key services: a Workload Analysis Engine, an evolving Scheduler Policy Repository, and an Execution Verifier that validates all AI-generated code and configure before deployment with static and dynamic analysis.
We demonstrate this architecture's power with sched-agent, a multi-agent system that autonomously analyzes workloads, synthesizes custom eBPF scheduling policies, and deploys them via the sched\_ext infrastructure. Our evaluation shows that SchedCP achieves up to an 1.79x performance improvement, and a 13x cost reduction compared to naive agentic approaches, all while maintaining high success rate. By bridging the semantic gap, SchedCP democratizes expert-level system optimization and represents a step towards creating truly self-optimizing, application-aware operating systems. The code is open-sourced in this https URL 

---
# Text-to-Layout: A Generative Workflow for Drafting Architectural Floor Plans Using LLMs 

**Authors**: Jayakrishna Duggempudi, Lu Gao, Ahmed Senouci, Zhe Han, Yunpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00543)  

**Abstract**: This paper presents the development of an AI-powered workflow that uses Large Language Models (LLMs) to assist in drafting schematic architectural floor plans from natural language prompts. The proposed system interprets textual input to automatically generate layout options including walls, doors, windows, and furniture arrangements. It combines prompt engineering, a furniture placement refinement algorithm, and Python scripting to produce spatially coherent draft plans compatible with design tools such as Autodesk Revit. A case study of a mid-sized residential layout demonstrates the approach's ability to generate functional and structured outputs with minimal manual effort. The workflow is designed for transparent replication, with all key prompt specifications documented to enable independent implementation by other researchers. In addition, the generated models preserve the full range of Revit-native parametric attributes required for direct integration into professional BIM processes. 

---
# OmniDPO: A Preference Optimization Framework to Address Omni-Modal Hallucination 

**Authors**: Junzhe Chen, Tianshu Zhang, Shiyu Huang, Yuwei Niu, Chao Sun, Rongzhou Zhang, Guanyu Zhou, Lijie Wen, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00723)  

**Abstract**: Recently, Omni-modal large language models (OLLMs) have sparked a new wave of research, achieving impressive results in tasks such as audio-video understanding and real-time environment perception. However, hallucination issues still persist. Similar to the bimodal setting, the priors from the text modality tend to dominate, leading OLLMs to rely more heavily on textual cues while neglecting visual and audio information. In addition, fully multimodal scenarios introduce new challenges. Most existing models align visual or auditory modalities with text independently during training, while ignoring the intrinsic correlations between video and its corresponding audio. This oversight results in hallucinations when reasoning requires interpreting hidden audio cues embedded in video content. To address these challenges, we propose OmniDPO, a preference-alignment framework designed to mitigate hallucinations in OLLMs. Specifically, OmniDPO incorporates two strategies: (1) constructing text-preference sample pairs to enhance the model's understanding of audio-video interactions; and (2) constructing multimodal-preference sample pairs to strengthen the model's attention to visual and auditory information. By tackling both challenges, OmniDPO effectively improves multimodal grounding and reduces hallucination. Experiments conducted on two OLLMs demonstrate that OmniDPO not only effectively mitigates multimodal hallucinations but also significantly enhances the models' reasoning capabilities across modalities. All code and datasets will be released upon paper acceptance. 

---
# UrbanInsight: A Distributed Edge Computing Framework with LLM-Powered Data Filtering for Smart City Digital Twins 

**Authors**: Kishor Datta Gupta, Md Manjurul Ahsan, Mohd Ariful Haque, Roy George, Azmine Toushik Wasi  

**Link**: [PDF](https://arxiv.org/pdf/2509.00936)  

**Abstract**: Cities today generate enormous streams of data from sensors, cameras, and connected infrastructure. While this information offers unprecedented opportunities to improve urban life, most existing systems struggle with scale, latency, and fragmented insights. This work introduces a framework that blends physics-informed machine learning, multimodal data fusion, and knowledge graph representation with adaptive, rule-based intelligence powered by large language models (LLMs). Physics-informed methods ground learning in real-world constraints, ensuring predictions remain meaningful and consistent with physical dynamics. Knowledge graphs act as the semantic backbone, integrating heterogeneous sensor data into a connected, queryable structure. At the edge, LLMs generate context-aware rules that adapt filtering and decision-making in real time, enabling efficient operation even under constrained resources. Together, these elements form a foundation for digital twin systems that go beyond passive monitoring to provide actionable insights. By uniting physics-based reasoning, semantic data fusion, and adaptive rule generation, this approach opens new possibilities for creating responsive, trustworthy, and sustainable smart infrastructures. 

---
# SIGMUS: Semantic Integration for Knowledge Graphs in Multimodal Urban Spaces 

**Authors**: Brian Wang, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2509.00287)  

**Abstract**: Modern urban spaces are equipped with an increasingly diverse set of sensors, all producing an abundance of multimodal data. Such multimodal data can be used to identify and reason about important incidents occurring in urban landscapes, such as major emergencies, cultural and social events, as well as natural disasters. However, such data may be fragmented over several sources and difficult to integrate due to the reliance on human-driven reasoning for identifying relationships between the multimodal data corresponding to an incident, as well as understanding the different components which define an incident. Such relationships and components are critical to identifying the causes of such incidents, as well as producing forecasting the scale and intensity of future incidents as they begin to develop. In this work, we create SIGMUS, a system for Semantic Integration for Knowledge Graphs in Multimodal Urban Spaces. SIGMUS uses Large Language Models (LLMs) to produce the necessary world knowledge for identifying relationships between incidents occurring in urban spaces and data from different modalities, allowing us to organize evidence and observations relevant to an incident without relying and human-encoded rules for relating multimodal sensory data with incidents. This organized knowledge is represented as a knowledge graph, organizing incidents, observations, and much more. We find that our system is able to produce reasonable connections between 5 different data sources (new article text, CCTV images, air quality, weather, and traffic measurements) and relevant incidents occurring at the same time and location. 

---
# Ultra Strong Machine Learning: Teaching Humans Active Learning Strategies via Automated AI Explanations 

**Authors**: Lun Ai, Johannes Langer, Ute Schmid, Stephen Muggleton  

**Link**: [PDF](https://arxiv.org/pdf/2509.00961)  

**Abstract**: Ultra Strong Machine Learning (USML) refers to symbolic learning systems that not only improve their own performance but can also teach their acquired knowledge to quantifiably improve human performance. In this work, we present LENS (Logic Programming Explanation via Neural Summarisation), a neuro-symbolic method that combines symbolic program synthesis with large language models (LLMs) to automate the explanation of machine-learned logic programs in natural language. LENS addresses a key limitation of prior USML approaches by replacing hand-crafted explanation templates with scalable automated generation. Through systematic evaluation using multiple LLM judges and human validation, we demonstrate that LENS generates superior explanations compared to direct LLM prompting and hand-crafted templates. To investigate whether LENS can teach transferable active learning strategies, we carried out a human learning experiment across three related domains. Our results show no significant human performance improvements, suggesting that comprehensive LLM responses may overwhelm users for simpler problems rather than providing learning support. Our work provides a solid foundation for building effective USML systems to support human learning. The source code is available on: this https URL. 

---
# Know When to Explore: Difficulty-Aware Certainty as a Guide for LLM Reinforcement Learning 

**Authors**: Ang Li, Zhihang Yuan, Yang Zhang, Shouda Liu, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00125)  

**Abstract**: Reinforcement Learning with Verifiable Feedback (RLVF) has become a key technique for enhancing the reasoning abilities of Large Language Models (LLMs). However, its reliance on sparse, outcome based rewards, which only indicate if a final answer is correct or not, fails to provide granular guidance on the reasoning process itself. This limitation hinders efficient learning, as the model cannot distinguish between high quality and inefficient solutions, nor can it learn effectively from different types of failures. To address this, we observe that an LLMs self-certainty often correlates with task difficulty and solution quality. We introduce Difficulty Aware Certainty guided Exploration (DACE), a novel RL algorithm that leverages this insight to dynamically balance the exploration exploitation trade-off. DACE assesses task difficulty online based on the policys success rate. It then uses this signal to modulate an intrinsic reward: for difficult tasks where the model is struggling, DACE encourages exploration by penalizing high certainty; for easier tasks, it encourages learning efficiency by rewarding high certainty. Experiments on challenging mathematical reasoning benchmarks (AIME, MATH) show that DACE significantly outperforms strong baselines. The DACE-trained models not only achieve higher accuracy but also demonstrate more robust performance when scaling test-time compute, validating that our adaptive approach fosters effective exploration without sacrificing precision. 

---
# Instruction-Level Weight Shaping: A Framework for Self-Improving AI Agents 

**Authors**: Rimom Costa  

**Link**: [PDF](https://arxiv.org/pdf/2509.00251)  

**Abstract**: Large language models (LLMs) are fluent but largely static after pre-training; new or shifting knowledge is typically added with retrieval-augmented generation (RAG) or fine-tuning. RAG raises latency and engineering overhead and often fails to integrate facts; prompt engineering is brittle and can conflict with prior knowledge; fine-tuning is costly and risks catastrophic forgetting. We propose Instruction-Level Weight Shaping (ILWS): curated system instructions act as external, auditable pseudo-parameters updated after each session via reflection and user feedback. A Reflection Engine inspects conversation traces, diagnoses reasoning successes and failures, and proposes typed deltas $\Delta K=(\Delta S,\Delta U,\Delta T)$ over instructions, user preferences, and tools. Deltas are version-controlled, evaluated with a sliding window of 1-5 star ratings, auto-repaired on first failure, and rolled back on repeated failure. When an edit budget crosses a threshold, the agent compiles a rating-weighted synthetic set and distills matured instruction-space gains into parameters, converting prompt-space improvements into weight-space without downtime. ILWS makes explicit the low-rank shaping induced by context in transformer blocks, preserves governance, and removes per-call retrieval. In enterprise support it increased throughput 2.4-5.0x and cut audited hallucinations by about 80% versus a frozen baseline. In an Adobe Commerce Cloud proof of concept "L0 Support", it achieved 4-5x more tickets per hour and about 80% lower time per ticket, with autonomous instruction updates and optional tool synthesis. Because ILWS operates at the instruction layer until controlled distillation, it generalizes to dynamic domains (legal, medical, engineering) requiring adaptive reasoning, tool creation, and low-latency deployment. 

---
# SHERPA: A Model-Driven Framework for Large Language Model Execution 

**Authors**: Boqi Chen, Kua Chen, José Antonio Hernández López, Gunter Mussbacher, Dániel Varró, Amir Feizpour  

**Link**: [PDF](https://arxiv.org/pdf/2509.00272)  

**Abstract**: Recently, large language models (LLMs) have achieved widespread application across various fields. Despite their impressive capabilities, LLMs suffer from a lack of structured reasoning ability, particularly for complex tasks requiring domain-specific best practices, which are often unavailable in the training data. Although multi-step prompting methods incorporating human best practices, such as chain-of-thought and tree-of-thought, have gained popularity, they lack a general mechanism to control LLM behavior. In this paper, we propose SHERPA, a model-driven framework to improve the LLM performance on complex tasks by explicitly incorporating domain-specific best practices into hierarchical state machines. By structuring the LLM execution processes using state machines, SHERPA enables more fine-grained control over their behavior via rules or decisions driven by machine learning-based approaches, including LLMs. We show that SHERPA is applicable to a wide variety of tasks-specifically, code generation, class name generation, and question answering-replicating previously proposed approaches while further improving the performance. We demonstrate the effectiveness of SHERPA for the aforementioned tasks using various LLMs. Our systematic evaluation compares different state machine configurations against baseline approaches without state machines. Results show that integrating well-designed state machines significantly improves the quality of LLM outputs, and is particularly beneficial for complex tasks with well-established human best practices but lacking data used for training LLMs. 

---
# Entropy-Guided Loop: Achieving Reasoning through Uncertainty-Aware Generation 

**Authors**: Andrew G. A. Correa, Ana C. H de Matos  

**Link**: [PDF](https://arxiv.org/pdf/2509.00079)  

**Abstract**: Reasoning models often outperform smaller models but at 3--5$\times$ higher cost and added latency. We present entropy-guided refinement: a lightweight, test-time loop that uses token-level uncertainty to trigger a single, targeted refinement pass. We extract logprobs, compute Shannon entropy on top-$k$ alternatives, and apply a simple OR-logic trigger over perplexity, maximum token entropy, and low-confidence-token count. Unlike approaches that use entropy only for measurement or decoding, we pass a compact uncertainty report (tokens, confidences, alternatives, context) back to the model to guide corrective edits. On representative technical queries across reasoning, mathematics, and code generation tasks, a small model with our loop approaches 95\% of a reference reasoning model's quality at approximately one-third of the cost. The method achieves selective refinement on ~31\% of responses while improving accuracy by 16 percentage points over single-pass inference. We demonstrate that this uncertainty-aware loop provides an effective middle ground between single-pass inference and expensive reasoning chains, making it practical for production deployments where both quality and cost matter. 

---
# Beyond Memorization: Reasoning-Driven Synthesis as a Mitigation Strategy Against Benchmark Contamination 

**Authors**: Terry Jingchen Zhang, Gopal Dev, Ning Wang, Nicole Ni, Wenyuan Jiang, Yinya Huang, Bernhard Schölkopf, Mrinmaya Sachan, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.00072)  

**Abstract**: Capability evaluation of large language models (LLMs) is increasingly shadowed by rising concerns of data contamination that cast doubts on whether static benchmarks measure genuine reasoning or mere memorization. We present an empirical study using an infinitely scalable framework to synthesize research-level QA directly from arXiv papers, harnessing the natural temporal structure of research publications where performance decay after knowledge cutoffs may indicate potential contamination. We evaluated 4 frontier model represented by 2 models of different knowledge cutoff dates per family on 1,643 multi-step reasoning questions synthesized from 20,277 arXiv papers stratified over 26 months, covering at least 6 months before and after all cutoff dates. Our results consistently showed a lack of significant performance decay near knowledge cutoff dates for models of various sizes, developers, and release dates. We further performed a comparative analysis with previous longitudinal studies that reported significant post-cutoff performance decay using directly retrieved questions based on public data. we hypothesize that the multi-step reasoning required by our synthesis pipeline offered additional complexity that goes deeper than shallow memorization, which effectively serves a mitigation strategy against benchmark contamination. We fully open source our code and dataset to aid reproducibility and advocate for a paradigm shift that prioritize reasoning-driven synthesis to construct benchmarks over simply collecting newly released questions periodically. 

---
# Poisoned at Scale: A Scalable Audit Uncovers Hidden Scam Endpoints in Production LLMs 

**Authors**: Zhiyang Chen, Tara Saba, Xun Deng, Xujie Si, Fan Long  

**Link**: [PDF](https://arxiv.org/pdf/2509.02372)  

**Abstract**: Large Language Models (LLMs) have become critical to modern software development, but their reliance on internet datasets for training introduces a significant security risk: the absorption and reproduction of malicious content. To evaluate this threat, this paper introduces a scalable, automated audit framework that synthesizes innocuous, developer-style prompts from known scam databases to query production LLMs and determine if they generate code containing harmful URLs. We conducted a large-scale evaluation across four production LLMs (GPT-4o, GPT-4o-mini, Llama-4-Scout, and DeepSeek-V3), and found a systemic vulnerability, with all tested models generating malicious code at a non-negligible rate. On average, 4.2\% of programs generated in our experiments contained malicious URLs. Crucially, this malicious code is often generated in response to benign prompts. We manually validate the prompts which cause all four LLMs to generate malicious code, and resulting in 177 innocuous prompts that trigger all models to produce harmful outputs. These results provide strong empirical evidence that the training data of production LLMs has been successfully poisoned at scale, underscoring the urgent need for more robust defense mechanisms and post-generation safety checks to mitigate the propagation of hidden security threats. 

---
# ReCode: Improving LLM-based Code Repair with Fine-Grained Retrieval-Augmented Generation 

**Authors**: Yicong Zhao, Shisong Chen, Jiacheng Zhang, Zhixu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.02330)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated impressive capabilities in code-related tasks, such as code generation and automated program repair. Despite their promising performance, most existing approaches for code repair suffer from high training costs or computationally expensive inference. Retrieval-augmented generation (RAG), with its efficient in-context learning paradigm, offers a more scalable alternative. However, conventional retrieval strategies, which are often based on holistic code-text embeddings, fail to capture the structural intricacies of code, resulting in suboptimal retrieval quality. To address the above limitations, we propose ReCode, a fine-grained retrieval-augmented in-context learning framework designed for accurate and efficient code repair. Specifically, ReCode introduces two key innovations: (1) an algorithm-aware retrieval strategy that narrows the search space using preliminary algorithm type predictions; and (2) a modular dual-encoder architecture that separately processes code and textual inputs, enabling fine-grained semantic matching between input and retrieved contexts. Furthermore, we propose RACodeBench, a new benchmark constructed from real-world user-submitted buggy code, which addresses the limitations of synthetic benchmarks and supports realistic evaluation. Experimental results on RACodeBench and competitive programming datasets demonstrate that ReCode achieves higher repair accuracy with significantly reduced inference cost, highlighting its practical value for real-world code repair scenarios. 

---
# Baichuan-M2: Scaling Medical Capability with Large Verifier System 

**Authors**: Baichuan-M2 Team, Chengfeng Dou, Chong Liu, Fan Yang, Fei Li, Jiyuan Jia, Mingyang Chen, Qiang Ju, Shuai Wang, Shunya Dang, Tianpeng Li, Xiangrong Zeng, Yijie Zhou, Chenzheng Zhu, Da Pan, Fei Deng, Guangwei Ai, Guosheng Dong, Hongda Zhang, Jinyang Tai, Jixiang Hong, Kai Lu, Linzhuang Sun, Peidong Guo, Qian Ma, Rihui Xin, Shihui Yang, Shusen Zhang, Yichuan Mo, Zheng Liang, Zhishou Zhang, Hengfu Cui, Zuyi Zhu, Xiaochuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02208)  

**Abstract**: As large language models (LLMs) advance in conversational and reasoning capabilities, their practical application in healthcare has become a critical research focus. However, there is a notable gap between the performance of medical LLMs on static benchmarks such as USMLE and their utility in real-world clinical decision-making. This discrepancy arises because traditional exams fail to capture the dynamic, interactive nature of medical consultations. To address this challenge, we introduce a novel dynamic verification framework that moves beyond static answer verifier, establishing a large-scale, high-fidelity interactive reinforcement learning system. Our framework comprises two key components: a Patient Simulator that creates realistic clinical environments using de-identified medical records, and a Clinical Rubrics Generator that dynamically produces multi-dimensional evaluation metrics. Building on this foundation, we develop Baichuan-M2, a 32B-parameter medical augmented reasoning model trained through a multi-stage reinforcement learning strategy with an improved Group Relative Policy Optimization (GRPO) algorithm. Evaluated on HealthBench, Baichuan-M2 outperforms all other open-source models and most advanced closed-source counterparts, achieving a score above 32 on the challenging HealthBench Hard benchmark-previously exceeded only by GPT-5. Our work demonstrates that robust dynamic verifier system is essential for aligning LLM capabilities with practical clinical applications, establishing a new Pareto front in the performance-parameter trade-off for medical AI deployment. 

---
# When LLM Meets Time Series: Can LLMs Perform Multi-Step Time Series Reasoning and Inference 

**Authors**: Wen Ye, Jinbo Liu, Defu Cao, Wei Yang, Yan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.01822)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has sparked growing interest in their application to time series analysis tasks. However, their ability to perform complex reasoning over temporal data in real-world application domains remains underexplored. To move toward this goal, a first step is to establish a rigorous benchmark dataset for evaluation. In this work, we introduce the TSAIA Benchmark, a first attempt to evaluate LLMs as time-series AI assistants. To ensure both scientific rigor and practical relevance, we surveyed over 20 academic publications and identified 33 real-world task formulations. The benchmark encompasses a broad spectrum of challenges, ranging from constraint-aware forecasting to anomaly detection with threshold calibration: tasks that require compositional reasoning and multi-step time series analysis. The question generator is designed to be dynamic and extensible, supporting continuous expansion as new datasets or task types are introduced. Given the heterogeneous nature of the tasks, we adopt task-specific success criteria and tailored inference-quality metrics to ensure meaningful evaluation for each task. We apply this benchmark to assess eight state-of-the-art LLMs under a unified evaluation protocol. Our analysis reveals limitations in current models' ability to assemble complex time series analysis workflows, underscoring the need for specialized methodologies for domain-specific adaptation. Our benchmark is available at this https URL, and the code is available at this https URL. 

---
# E-PhishGen: Unlocking Novel Research in Phishing Email Detection 

**Authors**: Luca Pajola, Eugenio Caripoti, Simeone Pizzi, Mauro Conti, Stefan Banzer, Giovanni Apruzzese  

**Link**: [PDF](https://arxiv.org/pdf/2509.01791)  

**Abstract**: Every day, our inboxes are flooded with unsolicited emails, ranging between annoying spam to more subtle phishing scams. Unfortunately, despite abundant prior efforts proposing solutions achieving near-perfect accuracy, the reality is that countering malicious emails still remains an unsolved dilemma.
This "open problem" paper carries out a critical assessment of scientific works in the context of phishing email detection. First, we focus on the benchmark datasets that have been used to assess the methods proposed in research. We find that most prior work relied on datasets containing emails that -- we argue -- are not representative of current trends, and mostly encompass the English language. Based on this finding, we then re-implement and re-assess a variety of detection methods reliant on machine learning (ML), including large-language models (LLM), and release all of our codebase -- an (unfortunately) uncommon practice in related research. We show that most such methods achieve near-perfect performance when trained and tested on the same dataset -- a result which intrinsically hinders development (how can future research outperform methods that are already near perfect?). To foster the creation of "more challenging benchmarks" that reflect current phishing trends, we propose E-PhishGEN, an LLM-based (and privacy-savvy) framework to generate novel phishing-email datasets. We use our E-PhishGEN to create E-PhishLLM, a novel phishing-email detection dataset containing 16616 emails in three languages. We use E-PhishLLM to test the detectors we considered, showing a much lower performance than that achieved on existing benchmarks -- indicating a larger room for improvement. We also validate the quality of E-PhishLLM with a user study (n=30). To sum up, we show that phishing email detection is still an open problem -- and provide the means to tackle such a problem by future research. 

---
# AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions 

**Authors**: Yiwei Guo, Bohan Li, Hankun Wang, Zhihan Li, Shuai Wang, Xie Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.01787)  

**Abstract**: Although current large audio language models (LALMs) extend text large language models (LLMs) with generic acoustic understanding abilities, they usually suffer from instruction sensitivity, where different instructions of the same intention can yield drastically different outcomes. In this work, we propose AHAMask, where we simply mask some of the attention heads in the decoder-only LLM backbone of LALMs, to trigger specific acoustic task functionalities without instructions. These masks are efficiently obtained by training on an LALM, with the number of trainable parameters equal to the attention head count in its LLM backbone. We show by experiments that applying such selective attention head masks achieves comparable or even better performance than using instructions, either on single or composite tasks. Besides achieving reliable acoustic task specification for LALMs, this also reveals that LALMs exhibit certain "functional pathways" in their attention heads. 

---
# Agentic Workflow for Education: Concepts and Applications 

**Authors**: Yuan-Hao Jiang, Yijie Lu, Ling Dai, Jiatong Wang, Ruijia Li, Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01517)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs) and Artificial Intelligence (AI) agents, agentic workflows are showing transformative potential in education. This study introduces the Agentic Workflow for Education (AWE), a four-component model comprising self-reflection, tool invocation, task planning, and multi-agent collaboration. We distinguish AWE from traditional LLM-based linear interactions and propose a theoretical framework grounded in the von Neumann Multi-Agent System (MAS) architecture. Through a paradigm shift from static prompt-response systems to dynamic, nonlinear workflows, AWE enables scalable, personalized, and collaborative task execution. We further identify four core application domains: integrated learning environments, personalized AI-assisted learning, simulation-based experimentation, and data-driven decision-making. A case study on automated math test generation shows that AWE-generated items are statistically comparable to real exam questions, validating the model's effectiveness. AWE offers a promising path toward reducing teacher workload, enhancing instructional quality, and enabling broader educational innovation. 

---
# DSDE: Dynamic Speculative Decoding with KLD Stability for Real-World Serving 

**Authors**: Mingyu Yang, Jae-Young Choi, Kihyo Moon, Minsung Jang, Eunjoo Joen  

**Link**: [PDF](https://arxiv.org/pdf/2509.01083)  

**Abstract**: Speculative decoding accelerates large language model inference, but its reliance on a fixed speculation length is suboptimal in large-batch serving environments with diverse requests. This paper explores a new direction for dynamic adaptation by investigating a novel class of post-hoc, diagnostic signals. We propose Dynamic Speculative Decoding Engine (DSDE), a training-free framework built on two primary components: (1) a predictive signal based on the variance of the Kullback-Leibler (KLD) divergence, which diagnoses the generation's regional stability, and (2) an adaptive speculation length cap to mitigate the straggler problem in per-sequence decoding. Experiments demonstrate the potential of using KLD-based stability signals for dynamic adaptation. An algorithm guided by these signals achieves end-to-end latency competitive with leading baselines and exhibits superior robustness across diverse workloads. This robustness is particularly valuable in challenging low-acceptance-rate regimes, where the proposed signal maintains its diagnostic utility. Collectively, these findings validate post-hoc signals as a valuable component for building more robust and intelligent LLM inference systems, and highlight a promising direction for future research on dynamic speculation length adaptation. 

---
# Accelerating Latency-Critical Applications with AI-Powered Semi-Automatic Fine-Grained Parallelization on SMT Processors 

**Authors**: Denis Los, Igor Petushkov  

**Link**: [PDF](https://arxiv.org/pdf/2509.00883)  

**Abstract**: Latency-critical applications tend to show low utilization of functional units due to frequent cache misses and mispredictions during speculative execution in high-performance superscalar processors. However, due to significant impact on single-thread performance, Simultaneous Multithreading (SMT) technology is rarely used with heavy threads of latency-critical applications. In this paper, we explore utilization of SMT technology to support fine-grained parallelization of latency-critical applications. Following the advancements in the development of Large Language Models (LLMs), we introduce Aira, an AI-powered Parallelization Adviser. To implement Aira, we extend AI Coding Agent in Cursor IDE with additional tools connected through Model Context Protocol, enabling end-to-end AI Agent for parallelization. Additional connected tools enable LLM-guided hotspot detection, collection of dynamic dependencies with Dynamic Binary Instrumentation, SMT-aware performance simulation to estimate performance gains. We apply Aira with Relic parallel framework for fine-grained task parallelism on SMT cores to parallelize latency-critical benchmarks representing real-world applications used in industry. We show 17% geomean performance gain from parallelization of latency-critical benchmarks using Aira with Relic framework. 

---
# Multimodal Iterative RAG for Knowledge Visual Question Answering 

**Authors**: Changin Choi, Wonseok Lee, Jungmin Ko, Wonjong Rhee  

**Link**: [PDF](https://arxiv.org/pdf/2509.00798)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have significantly advanced multimodal understanding, their performance remains limited on knowledge-intensive visual questions that require external knowledge beyond the image. Retrieval-Augmented Generation (RAG) has become a promising solution for providing models with external knowledge, its conventional single-pass framework often fails to gather sufficient knowledge. To overcome this limitation, we propose MI-RAG, a Multimodal Iterative RAG framework that leverages reasoning to enhance retrieval and update reasoning over newly retrieved knowledge across modalities. At each iteration, MI-RAG leverages an accumulated reasoning record to dynamically formulate a multi-query. These queries then drive a joint search across heterogeneous knowledge bases containing both visually-grounded and textual knowledge. The newly acquired knowledge is synthesized into the reasoning record, progressively refining understanding across iterations. Experiments on challenging benchmarks, including Encyclopedic VQA, InfoSeek, and OK-VQA, show that MI-RAG significantly improves both retrieval recall and answer accuracy, establishing a scalable approach for compositional reasoning in knowledge-intensive VQA. 

---
# RAG-PRISM: A Personalized, Rapid, and Immersive Skill Mastery Framework with Adaptive Retrieval-Augmented Tutoring 

**Authors**: Gaurangi Raul, Yu-Zheng Lin, Karan Patel, Bono Po-Jen Shih, Matthew W. Redondo, Banafsheh Saber Latibari, Jesus Pacheco, Soheil Salehi, Pratik Satam  

**Link**: [PDF](https://arxiv.org/pdf/2509.00646)  

**Abstract**: The rapid digital transformation of Fourth Industrial Revolution (4IR) systems is reshaping workforce needs, widening skill gaps, especially for older workers. With growing emphasis on STEM skills such as robotics, automation, artificial intelligence (AI), and security, large-scale re-skilling and up-skilling are required. Training programs must address diverse backgrounds, learning styles, and motivations to improve persistence and success, while ensuring rapid, cost-effective workforce development through experiential learning. To meet these challenges, we present an adaptive tutoring framework that combines generative AI with Retrieval-Augmented Generation (RAG) to deliver personalized training. The framework leverages document hit rate and Mean Reciprocal Rank (MRR) to optimize content for each learner, and is benchmarked against human-generated training for alignment and relevance. We demonstrate the framework in 4IR cybersecurity learning by creating a synthetic QA dataset emulating trainee behavior, while RAG is tuned on curated cybersecurity materials. Evaluation compares its generated training with manually curated queries representing realistic student interactions. Responses are produced using large language models (LLMs) including GPT-3.5 and GPT-4, assessed for faithfulness and content alignment. GPT-4 achieves the best performance with 87% relevancy and 100% alignment. Results show this dual-mode approach enables the adaptive tutor to act as both a personalized topic recommender and content generator, offering a scalable solution for rapid, tailored learning in 4IR education and workforce development. 

---
# LLM-HyPZ: Hardware Vulnerability Discovery using an LLM-Assisted Hybrid Platform for Zero-Shot Knowledge Extraction and Refinement 

**Authors**: Yu-Zheng Lin, Sujan Ghimire, Abhiram Nandimandalam, Jonah Michael Camacho, Unnati Tripathi, Rony Macwan, Sicong Shao, Setareh Rafatirad, Rozhin Yasaei, Pratik Satam, Soheil Salehi  

**Link**: [PDF](https://arxiv.org/pdf/2509.00647)  

**Abstract**: The rapid growth of hardware vulnerabilities has created an urgent need for systematic and scalable analysis methods. Unlike software flaws, which are often patchable post-deployment, hardware weaknesses remain embedded across product lifecycles, posing persistent risks to processors, embedded devices, and IoT platforms. Existing efforts such as the MITRE CWE Hardware List (2021) relied on expert-driven Delphi surveys, which lack statistical rigor and introduce subjective bias, while large-scale data-driven foundations for hardware weaknesses have been largely absent. In this work, we propose LLM-HyPZ, an LLM-assisted hybrid framework for zero-shot knowledge extraction and refinement from vulnerability corpora. Our approach integrates zero-shot LLM classification, contextualized embeddings, unsupervised clustering, and prompt-driven summarization to mine hardware-related CVEs at scale. Applying LLM-HyPZ to the 2021-2024 CVE corpus (114,836 entries), we identified 1,742 hardware-related vulnerabilities. We distilled them into five recurring themes, including privilege escalation via firmware and BIOS, memory corruption in mobile and IoT systems, and physical access exploits. Benchmarking across seven LLMs shows that LLaMA 3.3 70B achieves near-perfect classification accuracy (99.5%) on a curated validation set. Beyond methodological contributions, our framework directly supported the MITRE CWE Most Important Hardware Weaknesses (MIHW) 2025 update by narrowing the candidate search space. Specifically, our pipeline surfaced 411 of the 1,026 CVEs used for downstream MIHW analysis, thereby reducing expert workload and accelerating evidence gathering. These results establish LLM-HyPZ as the first data-driven, scalable approach for systematically discovering hardware vulnerabilities, thereby bridging the gap between expert knowledge and real-world vulnerability evidence. 

---
# TimeCopilot 

**Authors**: Azul Garza, Reneé Rosillo  

**Link**: [PDF](https://arxiv.org/pdf/2509.00616)  

**Abstract**: We introduce TimeCopilot, the first open-source agentic framework for forecasting that combines multiple Time Series Foundation Models (TSFMs) with Large Language Models (LLMs) through a single unified API. TimeCopilot automates the forecasting pipeline: feature analysis, model selection, cross-validation, and forecast generation, while providing natural language explanations and supporting direct queries about the future. The framework is LLM-agnostic, compatible with both commercial and open-source models, and supports ensembles across diverse forecasting families. Results on the large-scale GIFT-Eval benchmark show that TimeCopilot achieves state-of-the-art probabilistic forecasting performance at low cost. Our framework provides a practical foundation for reproducible, explainable, and accessible agentic forecasting systems. 

---
# LLM-Driven Policy Diffusion: Enhancing Generalization in Offline Reinforcement Learning 

**Authors**: Hanping Zhang, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.00347)  

**Abstract**: Reinforcement Learning (RL) is known for its strong decision-making capabilities and has been widely applied in various real-world scenarios. However, with the increasing availability of offline datasets and the lack of well-designed online environments from human experts, the challenge of generalization in offline RL has become more prominent. Due to the limitations of offline data, RL agents trained solely on collected experiences often struggle to generalize to new tasks or environments. To address this challenge, we propose LLM-Driven Policy Diffusion (LLMDPD), a novel approach that enhances generalization in offline RL using task-specific prompts. Our method incorporates both text-based task descriptions and trajectory prompts to guide policy learning. We leverage a large language model (LLM) to process text-based prompts, utilizing its natural language understanding and extensive knowledge base to provide rich task-relevant context. Simultaneously, we encode trajectory prompts using a transformer model, capturing structured behavioral patterns within the underlying transition dynamics. These prompts serve as conditional inputs to a context-aware policy-level diffusion model, enabling the RL agent to generalize effectively to unseen tasks. Our experimental results demonstrate that LLMDPD outperforms state-of-the-art offline RL methods on unseen tasks, highlighting its effectiveness in improving generalization and adaptability in diverse settings. 

---
# Embodied Spatial Intelligence: from Implicit Scene Modeling to Spatial Reasoning 

**Authors**: Jiading Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00465)  

**Abstract**: This thesis introduces "Embodied Spatial Intelligence" to address the challenge of creating robots that can perceive and act in the real world based on natural language instructions. To bridge the gap between Large Language Models (LLMs) and physical embodiment, we present contributions on two fronts: scene representation and spatial reasoning. For perception, we develop robust, scalable, and accurate scene representations using implicit neural models, with contributions in self-supervised camera calibration, high-fidelity depth field generation, and large-scale reconstruction. For spatial reasoning, we enhance the spatial capabilities of LLMs by introducing a novel navigation benchmark, a method for grounding language in 3D, and a state-feedback mechanism to improve long-horizon decision-making. This work lays a foundation for robots that can robustly perceive their surroundings and intelligently act upon complex, language-based commands. 

---
# LLM-based Triplet Extraction for Automated Ontology Generation in Software Engineering Standards 

**Authors**: Songhui Yue  

**Link**: [PDF](https://arxiv.org/pdf/2509.00140)  

**Abstract**: Ontologies have supported knowledge representation and whitebox reasoning for decades; thus, the automated ontology generation (AOG) plays a crucial role in scaling their use. Software engineering standards (SES) consist of long, unstructured text (with high noise) and paragraphs with domain-specific terms. In this setting, relation triple extraction (RTE), together with term extraction, constitutes the first stage toward AOG. This work proposes an open-source large language model (LLM)-assisted approach to RTE for SES. Instead of solely relying on prompt-engineering-based methods, this study promotes the use of LLMs as an aid in constructing ontologies and explores an effective AOG workflow that includes document segmentation, candidate term mining, LLM-based relation inference, term normalization, and cross-section alignment. Golden-standard benchmarks at three granularities are constructed and used to evaluate the ontology generated from the study. The results show that it is comparable and potentially superior to the OpenIE method of triple extraction. 

---
# AEGIS : Automated Co-Evolutionary Framework for Guarding Prompt Injections Schema 

**Authors**: Ting-Chun Liu, Ching-Yu Hsu, Kuan-Yi Lee, Chi-An Fu, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.00088)  

**Abstract**: Prompt injection attacks pose a significant challenge to the safe deployment of Large Language Models (LLMs) in real-world applications. While prompt-based detection offers a lightweight and interpretable defense strategy, its effectiveness has been hindered by the need for manual prompt engineering. To address this issue, we propose AEGIS , an Automated co-Evolutionary framework for Guarding prompt Injections Schema. Both attack and defense prompts are iteratively optimized against each other using a gradient-like natural language prompt optimization technique. This framework enables both attackers and defenders to autonomously evolve via a Textual Gradient Optimization (TGO) module, leveraging feedback from an LLM-guided evaluation loop. We evaluate our system on a real-world assignment grading dataset of prompt injection attacks and demonstrate that our method consistently outperforms existing baselines, achieving superior robustness in both attack success and detection. Specifically, the attack success rate (ASR) reaches 1.0, representing an improvement of 0.26 over the baseline. For detection, the true positive rate (TPR) improves by 0.23 compared to the previous best work, reaching 0.84, and the true negative rate (TNR) remains comparable at 0.89. Ablation studies confirm the importance of co-evolution, gradient buffering, and multi-objective optimization. We also confirm that this framework is effective in different LLMs. Our results highlight the promise of adversarial training as a scalable and effective approach for guarding prompt injections. 

---
# Pre-trained knowledge elevates large language models beyond traditional chemical reaction optimizers 

**Authors**: Robert MacKnight, Jose Emilio Regio, Jeffrey G. Ethier, Luke A. Baldwin, Gabe Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2509.00103)  

**Abstract**: Modern optimization in experimental chemistry employs algorithmic search through black-box parameter spaces. Here we demonstrate that pre-trained knowledge in large language models (LLMs) fundamentally changes this paradigm. Using six fully enumerated categorical reaction datasets (768 - 5,684 experiments), we benchmark LLM-guided optimization (LLM-GO) against Bayesian optimization (BO) and random sampling. Frontier LLMs consistently match or exceed BO performance across five single-objective datasets, with advantages growing as parameter complexity increases and high-performing conditions become scarce (<5% of space). BO retains superiority only for explicit multi-objective trade-offs. To understand these contrasting behaviors, we introduce a topology-agnostic information theory framework quantifying sampling diversity throughout optimization campaigns. This analysis reveals that LLMs maintain systematically higher exploration entropy than BO across all datasets while achieving superior performance, with advantages most pronounced in solution-scarce parameter spaces where high-entropy exploration typically fails - suggesting that pre-trained domain knowledge enables more effective navigation of chemical parameter space rather than replacing structured exploration strategies. To enable transparent benchmarking and community validation, we release Iron Mind (this https URL), a no-code platform for side-by-side evaluation of human, algorithmic, and LLM optimization campaigns with public leaderboards and complete trajectories. Our findings establish that LLM-GO excels precisely where traditional methods struggle: complex categorical spaces requiring domain understanding rather than mathematical optimization. 

---
# CoComposer: LLM Multi-agent Collaborative Music Composition 

**Authors**: Peiwen Xing, Aske Plaat, Niki van Stein  

**Link**: [PDF](https://arxiv.org/pdf/2509.00132)  

**Abstract**: Existing AI Music composition tools are limited in generation duration, musical quality, and controllability. We introduce CoComposer, a multi-agent system that consists of five collaborating agents, each with a task based on the traditional music composition workflow. Using the AudioBox-Aesthetics system, we experimentally evaluate CoComposer on four compositional criteria. We test with three LLMs (GPT-4o, DeepSeek-V3-0324, Gemini-2.5-Flash), and find (1) that CoComposer outperforms existing multi-agent LLM-based systems in music quality, and (2) compared to a single-agent system, in production complexity. Compared to non- LLM MusicLM, CoComposer has better interpretability and editability, although MusicLM still produces better music. 

---
# MolErr2Fix:Benchmarking LLM Trustworthiness in Chemistry via Modular Error Detection, Localization, Explanation, and Revision 

**Authors**: Yuyang Wu, Jinhui Ye, Shuhao Zhang, Lu Dai, Yonatan Bisk, Olexandr Isayev  

**Link**: [PDF](https://arxiv.org/pdf/2509.00063)  

**Abstract**: Large Language Models (LLMs) have shown growing potential in molecular sciences, but they often produce chemically inaccurate descriptions and struggle to recognize or justify potential errors. This raises important concerns about their robustness and reliability in scientific applications. To support more rigorous evaluation of LLMs in chemical reasoning, we present the MolErr2Fix benchmark, designed to assess LLMs on error detection and correction in molecular descriptions. Unlike existing benchmarks focused on molecule-to-text generation or property prediction, MolErr2Fix emphasizes fine-grained chemical understanding. It tasks LLMs with identifying, localizing, explaining, and revising potential structural and semantic errors in molecular descriptions. Specifically, MolErr2Fix consists of 1,193 fine-grained annotated error instances. Each instance contains quadruple annotations, i.e,. (error type, span location, the explanation, and the correction). These tasks are intended to reflect the types of reasoning and verification required in real-world chemical communication. Evaluations of current state-of-the-art LLMs reveal notable performance gaps, underscoring the need for more robust chemical reasoning capabilities. MolErr2Fix provides a focused benchmark for evaluating such capabilities and aims to support progress toward more reliable and chemically informed language models. All annotations and an accompanying evaluation API will be publicly released to facilitate future research. 

---
# A Whole New World: Creating a Parallel-Poisoned Web Only AI-Agents Can See 

**Authors**: Shaked Zychlinski  

**Link**: [PDF](https://arxiv.org/pdf/2509.00124)  

**Abstract**: This paper introduces a novel attack vector that leverages website cloaking techniques to compromise autonomous web-browsing agents powered by Large Language Models (LLMs). As these agents become more prevalent, their unique and often homogenous digital fingerprints - comprising browser attributes, automation framework signatures, and network characteristics - create a new, distinguishable class of web traffic. The attack exploits this fingerprintability. A malicious website can identify an incoming request as originating from an AI agent and dynamically serve a different, "cloaked" version of its content. While human users see a benign webpage, the agent is presented with a visually identical page embedded with hidden, malicious instructions, such as indirect prompt injections. This mechanism allows adversaries to hijack agent behavior, leading to data exfiltration, malware execution, or misinformation propagation, all while remaining completely invisible to human users and conventional security crawlers. This work formalizes the threat model, details the mechanics of agent fingerprinting and cloaking, and discusses the profound security implications for the future of agentic AI, highlighting the urgent need for robust defenses against this stealthy and scalable attack. 

---
# Enabling Transparent Cyber Threat Intelligence Combining Large Language Models and Domain Ontologies 

**Authors**: Luca Cotti, Anisa Rula, Devis Bianchini, Federico Cerutti  

**Link**: [PDF](https://arxiv.org/pdf/2509.00081)  

**Abstract**: Effective Cyber Threat Intelligence (CTI) relies upon accurately structured and semantically enriched information extracted from cybersecurity system logs. However, current methodologies often struggle to identify and interpret malicious events reliably and transparently, particularly in cases involving unstructured or ambiguous log entries. In this work, we propose a novel methodology that combines ontology-driven structured outputs with Large Language Models (LLMs), to build an Artificial Intelligence (AI) agent that improves the accuracy and explainability of information extraction from cybersecurity logs. Central to our approach is the integration of domain ontologies and SHACL-based constraints to guide the language model's output structure and enforce semantic validity over the resulting graph. Extracted information is organized into an ontology-enriched graph database, enabling future semantic analysis and querying. The design of our methodology is motivated by the analytical requirements associated with honeypot log data, which typically comprises predominantly malicious activity. While our case study illustrates the relevance of this scenario, the experimental evaluation is conducted using publicly available datasets. Results demonstrate that our method achieves higher accuracy in information extraction compared to traditional prompt-only approaches, with a deliberate focus on extraction quality rather than processing speed. 

---
# ZeroQAT: Your Quantization-aware Training but Efficient 

**Authors**: Qitao Tan, Xiaoying Song, Jin Lu, Guoming Li, Jun Liu, Lingzi Hong, Caiwen Ding, Jundong Li, Xiaoming Zhai, Shaoyi Huang, Wei Niu, Geng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.00031)  

**Abstract**: Quantization is an effective technique to reduce the deployment cost of large language models (LLMs), and post-training quantization (PTQ) has been widely studied due to its efficiency. However, existing low-bit PTQ methods suffer from accuracy degradation because their layer-wise optimization introduces cumulative error propagation and misalignment between local reconstruction objectives and downstream performance. While quantization-aware training (QAT) provides a principled solution, its reliance on backpropagation incurs prohibitive data, time, and memory costs, limiting its practicality. To address these challenges, we propose ZeroQAT, a zeroth-order optimization-based QAT framework. ZeroQAT leverages forward-only gradient estimation to eliminate the need for backpropagation, significantly reducing computational and memory overhead while retaining the benefits of end-to-end optimization. Moreover, ZeroQAT jointly learns quantized weights, weight clipping thresholds, and equivalent transformations to mitigate quantization error and handle activation outliers. Experiments demonstrate that ZeroQAT achieves the efficiency of PTQ while retaining the accuracy of QAT, offering a practical solution for high-quality low-bit quantization of LLMs. 

---
# Robotic Fire Risk Detection based on Dynamic Knowledge Graph Reasoning: An LLM-Driven Approach with Graph Chain-of-Thought 

**Authors**: Haimei Pan, Jiyun Zhang, Qinxi Wei, Xiongnan Jin, Chen Xinkai, Jie Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.00054)  

**Abstract**: Fire is a highly destructive disaster, but effective prevention can significantly reduce its likelihood of occurrence. When it happens, deploying emergency robots in fire-risk scenarios can help minimize the danger to human responders. However, current research on pre-disaster warnings and disaster-time rescue still faces significant challenges due to incomplete perception, inadequate fire situational awareness, and delayed response. To enhance intelligent perception and response planning for robots in fire scenarios, we first construct a knowledge graph (KG) by leveraging large language models (LLMs) to integrate fire domain knowledge derived from fire prevention guidelines and fire rescue task information from robotic emergency response documents. We then propose a new framework called Insights-on-Graph (IOG), which integrates the structured fire information of KG and Large Multimodal Models (LMMs). The framework generates perception-driven risk graphs from real-time scene imagery to enable early fire risk detection and provide interpretable emergency responses for task module and robot component configuration based on the evolving risk situation. Extensive simulations and real-world experiments show that IOG has good applicability and practical application value in fire risk detection and rescue decision-making. 

---
# Deep Learning-Driven Multimodal Detection and Movement Analysis of Objects in Culinary 

**Authors**: Tahoshin Alam Ishat  

**Link**: [PDF](https://arxiv.org/pdf/2509.00033)  

**Abstract**: This is a research exploring existing models and fine tuning them to combine a YOLOv8 segmentation model, a LSTM model trained on hand point motion sequence and a ASR (whisper-base) to extract enough data for a LLM (TinyLLaMa) to predict the recipe and generate text creating a step by step guide for the cooking procedure. All the data were gathered by the author for a robust task specific system to perform best in complex and challenging environments proving the extension and endless application of computer vision in daily activities such as kitchen work. This work extends the field for many more crucial task of our day to day life. 

---
# Exploring and Reshaping the Weight Distribution in LLM 

**Authors**: Chunming Ye, Songzhou Li, Xu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00046)  

**Abstract**: The performance of Large Language Models is influenced by their characteristics such as architecture, model sizes, decoding methods and so on. Due to differences in structure or function, the weights in different layers of large models have varying distributions. This paper explores the correlations between different types of layers in terms of weights distribution and studies the potential impact of these correlations on LoRA training effectiveness. Firstly, the study reveals that in the model the cosine distances between weights of different layers manifest power-law distribution. We extract Query-projection, down-projection and other weight matrices from the self-attention layers and MLP layers, calculate the singular values of the matrices using singular value decomposition, and organize a certain number of singular values into matrices according to projection's type. By analyzing the probability distribution of the cosine distances between these matrices, it is found that the cosine distances values between them have distinct power-law distribution characteristics. Secondly, based on the results of distance calculations and analysis across different layers of model, a qualitative method is proposed to describe the distribution characteristics of different models. Next, to construct weights that align with the distribution characteristics, a data generator is designed using a combination of Gaussian process and Pareto distribution functions. The generator is used to simulate the generation of data that aligns with specific distribution characteristics. Finally, based on the aforementioned distribution characteristics and data generation method, the weights in LoRA initialization are reshaped for training. Experimental results indicate that, without altering the model structure or training process, this method achieves a certain improvement in the performance of LoRA training. 

---
# Per-sender neural network classifiers for email authorship validation 

**Authors**: Rohit Dube  

**Link**: [PDF](https://arxiv.org/pdf/2509.00005)  

**Abstract**: Business email compromise and lateral spear phishing attacks are among modern organizations' most costly and damaging threats. While inbound phishing defenses have improved significantly, most organizations still trust internal emails by default, leaving themselves vulnerable to attacks from compromised employee accounts. In this work, we define and explore the problem of authorship validation: verifying whether a claimed sender actually authored a given email. Authorship validation is a lightweight, real-time defense that complements traditional detection methods by modeling per-sender writing style. Further, the paper presents a collection of new datasets based on the Enron corpus. These simulate inauthentic messages using both human-written and large language model-generated emails. The paper also evaluates two classifiers -- a Naive Bayes model and a character-level convolutional neural network (Char-CNN) -- for the authorship validation task. Our experiments show that the Char-CNN model achieves high accuracy and F1 scores under various circumstances. Finally, we discuss deployment considerations and show that per-sender authorship classifiers are practical for integrating into existing commercial email security systems with low overhead. 

---
