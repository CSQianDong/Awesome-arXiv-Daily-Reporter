# Trove: A Flexible Toolkit for Dense Retrieval 

**Authors**: Reza Esfandiarpoor, Max Zuo, Stephen H. Bach  

**Link**: [PDF](https://arxiv.org/pdf/2511.01857)  

**Abstract**: We introduce Trove, an easy-to-use open-source retrieval toolkit that simplifies research experiments without sacrificing flexibility or speed. For the first time, we introduce efficient data management features that load and process (filter, select, transform, and combine) retrieval datasets on the fly, with just a few lines of code. This gives users the flexibility to easily experiment with different dataset configurations without the need to compute and store multiple copies of large datasets. Trove is highly customizable: in addition to many built-in options, it allows users to freely modify existing components or replace them entirely with user-defined objects. It also provides a low-code and unified pipeline for evaluation and hard negative mining, which supports multi-node execution without any code changes. Trove's data management features reduce memory consumption by a factor of 2.6. Moreover, Trove's easy-to-use inference pipeline incurs no overhead, and inference times decrease linearly with the number of available nodes. Most importantly, we demonstrate how Trove simplifies retrieval experiments and allows for arbitrary customizations, thus facilitating exploratory research. 

---
# CAT-ID$^2$: Category-Tree Integrated Document Identifier Learning for Generative Retrieval In E-commerce 

**Authors**: Xiaoyu Liu, Fuwei Zhang, Yiqing Wu, Xinyu Jia, Zenghua Xia, Fuzhen Zhuang, Zhao Zhang, Fei Jiang, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.01461)  

**Abstract**: Generative retrieval (GR) has gained significant attention as an effective paradigm that integrates the capabilities of large language models (LLMs). It generally consists of two stages: constructing discrete semantic identifiers (IDs) for documents and retrieving documents by autoregressively generating ID this http URL core challenge in GR is how to construct document IDs (DocIDS) with strong representational power. Good IDs should exhibit two key properties: similar documents should have more similar IDs, and each document should maintain a distinct and unique this http URL, most existing methods ignore native category information, which is common and critical in E-commerce. Therefore, we propose a novel ID learning method, CAtegory-Tree Integrated Document IDentifier (CAT-ID$^2$), incorporating prior category information into the semantic this http URL-ID$^2$ includes three key modules: a Hierarchical Class Constraint Loss to integrate category information layer by layer during quantization, a Cluster Scale Constraint Loss for uniform ID token distribution, and a Dispersion Loss to improve the distinction of reconstructed documents. These components enable CAT-ID$^2$ to generate IDs that make similar documents more alike while preserving the uniqueness of different documents' this http URL offline and online experiments confirm the effectiveness of our method, with online A/B tests showing a 0.33% increase in average orders per thousand users for ambiguous intent queries and 0.24% for long-tail queries. 

---
# LiCoMemory: Lightweight and Cognitive Agentic Memory for Efficient Long-Term Reasoning 

**Authors**: Zhengjun Huang, Zhoujin Tian, Qintian Guo, Fangyuan Zhang, Yingli Zhou, Di Jiang, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.01448)  

**Abstract**: Large Language Model (LLM) agents exhibit remarkable conversational and reasoning capabilities but remain constrained by limited context windows and the lack of persistent memory. Recent efforts address these limitations via external memory architectures, often employing graph-based representations, yet most adopt flat, entangled structures that intertwine semantics with topology, leading to redundant representations, unstructured retrieval, and degraded efficiency and accuracy. To resolve these issues, we propose LiCoMemory, an end-to-end agentic memory framework for real-time updating and retrieval, which introduces CogniGraph, a lightweight hierarchical graph that utilizes entities and relations as semantic indexing layers, and employs temporal and hierarchy-aware search with integrated reranking for adaptive and coherent knowledge retrieval. Experiments on long-term dialogue benchmarks, LoCoMo and LongMemEval, show that LiCoMemory not only outperforms established baselines in temporal reasoning, multi-session consistency, and retrieval efficiency, but also notably reduces update latency. Our official code and data are available at this https URL. 

---
# A Soft-partitioned Semi-supervised Collaborative Transfer Learning Approach for Multi-Domain Recommendation 

**Authors**: Xiaoyu Liu, Yiqing Wu, Ruidong Han, Fuzhen Zhuang, Xiang Li, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.01404)  

**Abstract**: In industrial practice, Multi-domain Recommendation (MDR) plays a crucial role. Shared-specific architectures are widely used in industrial solutions to capture shared and unique attributes via shared and specific parameters. However, with imbalanced data across different domains, these models face two key issues: (1) Overwhelming: Dominant domain data skews model performance, neglecting non-dominant domains. (2) Overfitting: Sparse data in non-dominant domains leads to overfitting in specific parameters. To tackle these challenges, we propose Soft-partitioned Semi-supervised Collaborative Transfer Learning (SSCTL) for multi-domain recommendation. SSCTL generates dynamic parameters to address the overwhelming issue, thus shifting focus towards samples from non-dominant domains. To combat overfitting, it leverages pseudo-labels with weights from dominant domain instances to enhance non-dominant domain data. We conduct comprehensive experiments, both online and offline, to validate the efficacy of our proposed method. Online tests yielded significant improvements across various domains, with increases in GMV ranging from 0.54% to 2.90% and enhancements in CTR ranging from 0.22% to 1.69%. 

---
# A semantic-based deep learning approach for mathematical expression retrieval 

**Authors**: Pavan Kumar Perepu  

**Link**: [PDF](https://arxiv.org/pdf/2511.01364)  

**Abstract**: Mathematical expressions (MEs) have complex two-dimensional structures in which symbols can be present at any nested depth like superscripts, subscripts, above, below etc. As MEs are represented using LaTeX format, several text retrieval methods based on string matching, vector space models etc., have also been applied for ME retrieval problem in the literature. As these methods are based on syntactic similarity, recently deep learning approaches based on embedding have been used for semantic similarity. In our present work, we have focused on the retrieval of mathematical expressions using deep learning approaches. In our approach, semantic features are extracted from the MEs using a deep recurrent neural network (DRNN) and these features have been used for matching and retrieval. We have trained the network for a classification task which determines the complexity of an ME. ME complexity has been quantified in terms of its nested depth. Based on the nested depth, we have considered three complexity classes of MEs: Simple, Medium and Complex. After training the network, outputs just before the the final fully connected layer are extracted for all the MEs. These outputs form the semantic features of MEs and are stored in a database. For a given ME query, its semantic features are computed using the trained DRNN and matched against the semantic feature database. Matching is performed based on the standard euclidean distance and top 'k' nearest matches are retrieved, where 'k' is a user-defined parameter. Our approach has been illustrated on a database of 829 MEs. 

---
# Contextual Relevance and Adaptive Sampling for LLM-Based Document Reranking 

**Authors**: Jerry Huang, Siddarth Madala, Cheng Niu, Julia Hockenmaier, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.01208)  

**Abstract**: Reranking algorithms have made progress in improving document retrieval quality by efficiently aggregating relevance judgments generated by large language models (LLMs). However, identifying relevant documents for queries that require in-depth reasoning remains a major challenge. Reasoning-intensive queries often exhibit multifaceted information needs and nuanced interpretations, rendering document relevance inherently context dependent. To address this, we propose contextual relevance, which we define as the probability that a document is relevant to a given query, marginalized over the distribution of different reranking contexts it may appear in (i.e., the set of candidate documents it is ranked alongside and the order in which the documents are presented to a reranking model). While prior works have studied methods to mitigate the positional bias LLMs exhibit by accounting for the ordering of documents, we empirically find that the compositions of these batches also plays an important role in reranking performance. To efficiently estimate contextual relevance, we propose TS-SetRank, a sampling-based, uncertainty-aware reranking algorithm. Empirically, TS-SetRank improves nDCG@10 over retrieval and reranking baselines by 15-25% on BRIGHT and 6-21% on BEIR, highlighting the importance of modeling relevance as context-dependent. 

---
# Controlling Gender Bias in Retrieval via a Backpack Architecture 

**Authors**: Amirabbas Afzali, Amirreza Velae, Iman Ahmadi, Mohammad Aliannejadi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00875)  

**Abstract**: The presence of social biases in large language models (LLMs) has become a significant concern in AI research. These biases, often embedded in training data, can perpetuate harmful stereotypes and distort decision-making processes. When LLMs are integrated into ranking systems, they can propagate these biases, leading to unfair outcomes in critical applications such as search engines and recommendation systems. Backpack Language Models, unlike traditional transformer-based models that treat text sequences as monolithic structures, generate outputs as weighted combinations of non-contextual, learned word aspects, also known as senses. Leveraging this architecture, we propose a framework for debiasing ranking tasks. Our experimental results show that this framework effectively mitigates gender bias in text retrieval and ranking with minimal degradation in performance. 

---
# REaR: Retrieve, Expand and Refine for Effective Multitable Retrieval 

**Authors**: Rishita Agarwal, Himanshu Singhal, Peter Baile Chen, Manan Roy Choudhury, Dan Roth, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.00805)  

**Abstract**: Answering natural language queries over relational data often requires retrieving and reasoning over multiple tables, yet most retrievers optimize only for query-table relevance and ignore table table compatibility. We introduce REAR (Retrieve, Expand and Refine), a three-stage, LLM-free framework that separates semantic relevance from structural joinability for efficient, high-fidelity multi-table retrieval. REAR (i) retrieves query-aligned tables, (ii) expands these with structurally joinable tables via fast, precomputed column-embedding comparisons, and (iii) refines them by pruning noisy or weakly related candidates. Empirically, REAR is retriever-agnostic and consistently improves dense/sparse retrievers on complex table QA datasets (BIRD, MMQA, and Spider) by improving both multi-table retrieval quality and downstream SQL execution. Despite being LLM-free, it delivers performance competitive with state-of-the-art LLM-augmented retrieval systems (e.g.,ARM) while achieving much lower latency and cost. Ablations confirm complementary gains from expansion and refinement, underscoring REAR as a practical, scalable building block for table-based downstream tasks (e.g., Text-to-SQL). 

---
# Taxonomy-based Negative Sampling In Personalized Semantic Search for E-commerce 

**Authors**: Uthman Jinadu, Siawpeng Er, Le Yu, Chen Liang, Bingxin Li, Yi Ding, Aleksandar Velkoski  

**Link**: [PDF](https://arxiv.org/pdf/2511.00694)  

**Abstract**: Large retail outlets offer products that may be domain-specific, and this requires having a model that can understand subtle differences in similar items. Sampling techniques used to train these models are most of the time, computationally expensive or logistically challenging. These models also do not factor in users' previous purchase patterns or behavior, thereby retrieving irrelevant items for them. We present a semantic retrieval model for e-commerce search that embeds queries and products into a shared vector space and leverages a novel taxonomy-based hard-negative sampling(TB-HNS) strategy to mine contextually relevant yet challenging negatives. To further tailor retrievals, we incorporate user-level personalization by modeling each customer's past purchase history and behavior. In offline experiments, our approach outperforms BM25, ANCE and leading neural baselines on Recall@K, while live A/B testing shows substantial uplifts in conversion rate, add-to-cart rate, and average order value. We also demonstrate that our taxonomy-driven negatives reduce training overhead and accelerate convergence, and we share practical lessons from deploying this system at scale. 

---
# Structurally Refined Graph Transformer for Multimodal Recommendation 

**Authors**: Ke Shi, Yan Zhang, Miao Zhang, Lifan Chen, Jiali Yi, Kui Xiao, Xiaoju Hou, Zhifei Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00584)  

**Abstract**: Multimodal recommendation systems utilize various types of information, including images and text, to enhance the effectiveness of recommendations. The key challenge is predicting user purchasing behavior from the available data. Current recommendation models prioritize extracting multimodal information while neglecting the distinction between redundant and valuable data. They also rely heavily on a single semantic framework (e.g., local or global semantics), resulting in an incomplete or biased representation of user preferences, particularly those less expressed in prior interactions. Furthermore, these approaches fail to capture the complex interactions between users and items, limiting the model's ability to meet diverse users. To address these challenges, we present SRGFormer, a structurally optimized multimodal recommendation model. By modifying the transformer for better integration into our model, we capture the overall behavior patterns of users. Then, we enhance structural information by embedding multimodal information into a hypergraph structure to aid in learning the local structures between users and items. Meanwhile, applying self-supervised tasks to user-item collaborative signals enhances the integration of multimodal information, thereby revealing the representational features inherent to the data's modality. Extensive experiments on three public datasets reveal that SRGFormer surpasses previous benchmark models, achieving an average performance improvement of 4.47 percent on the Sports dataset. The code is publicly available online. 

---
# Listwise Preference Diffusion Optimization for User Behavior Trajectories Prediction 

**Authors**: Hongtao Huang, Chengkai Huang, Junda Wu, Tong Yu, Julian McAuley, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.00530)  

**Abstract**: Forecasting multi-step user behavior trajectories requires reasoning over structured preferences across future actions, a challenge overlooked by traditional sequential recommendation. This problem is critical for applications such as personalized commerce and adaptive content delivery, where anticipating a user's complete action sequence enhances both satisfaction and business outcomes. We identify an essential limitation of existing paradigms: their inability to capture global, listwise dependencies among sequence items. To address this, we formulate User Behavior Trajectory Prediction (UBTP) as a new task setting that explicitly models long-term user preferences. We introduce Listwise Preference Diffusion Optimization (LPDO), a diffusion-based training framework that directly optimizes structured preferences over entire item sequences. LPDO incorporates a Plackett-Luce supervision signal and derives a tight variational lower bound aligned with listwise ranking likelihoods, enabling coherent preference generation across denoising steps and overcoming the independent-token assumption of prior diffusion methods. To rigorously evaluate multi-step prediction quality, we propose the task-specific metric Sequential Match (SeqMatch), which measures exact trajectory agreement, and adopt Perplexity (PPL), which assesses probabilistic fidelity. Extensive experiments on real-world user behavior benchmarks demonstrate that LPDO consistently outperforms state-of-the-art baselines, establishing a new benchmark for structured preference learning with diffusion models. 

---
# LIR: The First Workshop on Late Interaction and Multi Vector Retrieval @ ECIR 2026 

**Authors**: Benjamin Clavié, Xianming Li, Antoine Chaffin, Omar Khattab, Tom Aarsen, Manuel Faysse, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.00444)  

**Abstract**: Late interaction retrieval methods, pioneered by ColBERT, have emerged as a powerful alternative to single-vector neural IR. By leveraging fine-grained, token-level representations, they have been demonstrated to deliver strong generalisation and robustness, particularly in out-of-domain settings. They have recently been shown to be particularly well-suited for novel use cases, such as reasoning-based or cross-modality retrieval. At the same time, these models pose significant challenges of efficiency, usability, and integrations into fully fledged systems; as well as the natural difficulties encountered while researching novel application domains. Recent years have seen rapid advances across many of these areas, but research efforts remain fragmented across communities and frequently exclude practitioners. The purpose of this workshop is to create an environment where all aspects of late interaction can be discussed, with a focus on early research explorations, real-world outcomes, and negative or puzzling results to be freely shared and discussed. The aim of LIR is to provide a highly-interactive environment for researchers from various backgrounds and practitioners to freely discuss their experience, fostering further collaboration. 

---
# Simple and Behavior-Driven Augmentation for Recommendation with Rich Collaborative Signals 

**Authors**: Doyun Choi, Cheonwoo Lee, Jaemin Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2511.00436)  

**Abstract**: Contrastive learning (CL) has been widely used for enhancing the performance of graph collaborative filtering (GCF) for personalized recommendation. Since data augmentation plays a crucial role in the success of CL, previous works have designed augmentation methods to remove noisy interactions between users and items in order to generate effective augmented views. However, the ambiguity in defining ''noisiness'' presents a persistent risk of losing core information and generating unreliable data views, while increasing the overall complexity of augmentation. In this paper, we propose Simple Collaborative Augmentation for Recommendation (SCAR), a novel and intuitive augmentation method designed to maximize the effectiveness of CL for GCF. Instead of removing information, SCAR leverages collaborative signals extracted from user-item interactions to generate pseudo-interactions, which are then either added to or used to replace existing interactions. This results in more robust representations while avoiding the pitfalls of overly complex augmentation modules. We conduct experiments on four benchmark datasets and show that SCAR outperforms previous CL-based GCF methods as well as other state-of-the-art self-supervised learning approaches across key evaluation metrics. SCAR exhibits strong robustness across different hyperparameter settings and is particularly effective in sparse data scenarios. 

---
# Effectiveness of LLMs in Temporal User Profiling for Recommendation 

**Authors**: Milad Sabouri, Masoud Mansoury, Kun Lin, Bamshad Mobasher  

**Link**: [PDF](https://arxiv.org/pdf/2511.00176)  

**Abstract**: Effectively modeling the dynamic nature of user preferences is crucial for enhancing recommendation accuracy and fostering transparency in recommender systems. Traditional user profiling often overlooks the distinction between transitory short-term interests and stable long-term preferences. This paper examines the capability of leveraging Large Language Models (LLMs) to capture these temporal dynamics, generating richer user representations through distinct short-term and long-term textual summaries of interaction histories. Our observations suggest that while LLMs tend to improve recommendation quality in domains with more active user engagement, their benefits appear less pronounced in sparser environments. This disparity likely stems from the varying distinguishability of short-term and long-term preferences across domains; the approach shows greater utility where these temporal interests are more clearly separable (e.g., Movies\&TV) compared to domains with more stable user profiles (e.g., Video Games). This highlights a critical trade-off between enhanced performance and computational costs, suggesting context-dependent LLM application. Beyond predictive capability, this LLM-driven approach inherently provides an intrinsic potential for interpretability through its natural language profiles and attention weights. This work contributes insights into the practical capability and inherent interpretability of LLM-driven temporal user profiling, outlining new research directions for developing adaptive and transparent recommender systems. 

---
# LookSync: Large-Scale Visual Product Search System for AI-Generated Fashion Looks 

**Authors**: Pradeep M, Ritesh Pallod, Satyen Abrol, Muthu Raman, Ian Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2511.00072)  

**Abstract**: Generative AI is reshaping fashion by enabling virtual looks and avatars making it essential to find real products that best match AI-generated styles. We propose an end-to-end product search system that has been deployed in a real-world, internet scale which ensures that AI-generated looks presented to users are matched with the most visually and semantically similar products from the indexed vector space. The search pipeline is composed of four key components: query generation, vectorization, candidate retrieval, and reranking based on AI-generated looks. Recommendation quality is evaluated using human-judged accuracy scores. The system currently serves more than 350,000 AI Looks in production per day, covering diverse product categories across global markets of over 12 million products. In our experiments, we observed that across multiple annotators and categories, CLIP outperformed alternative models by a small relative margin of 3--7\% in mean opinion scores. These improvements, though modest in absolute numbers, resulted in noticeably better user perception matches, establishing CLIP as the most reliable backbone for production deployment. 

---
# A Graph-based RAG for Energy Efficiency Question Answering 

**Authors**: Riccardo Campi, Nicolò Oreste Pinciroli Vago, Mathyas Giudici, Pablo Barrachina Rodriguez-Guisado, Marco Brambilla, Piero Fraternali  

**Link**: [PDF](https://arxiv.org/pdf/2511.01643)  

**Abstract**: In this work, we investigate the use of Large Language Models (LLMs) within a graph-based Retrieval Augmented Generation (RAG) architecture for Energy Efficiency (EE) Question Answering. First, the system automatically extracts a Knowledge Graph (KG) from guidance and regulatory documents in the energy field. Then, the generated graph is navigated and reasoned upon to provide users with accurate answers in multiple languages. We implement a human-based validation using the RAGAs framework properties, a validation dataset comprising 101 question-answer pairs, and domain experts. Results confirm the potential of this architecture and identify its strengths and weaknesses. Validation results show how the system correctly answers in about three out of four of the cases (75.2 +- 2.7%), with higher results on questions related to more general EE answers (up to 81.0 +- 4.1%), and featuring promising multilingual abilities (4.4% accuracy loss due to translation). 

---
# Vote-in-Context: Turning VLMs into Zero-Shot Rank Fusers 

**Authors**: Mohamed Eltahir, Ali Habibullah, Lama Ayash, Tanveer Hussain, Naeemullah Khan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01617)  

**Abstract**: In the retrieval domain, candidates' fusion from heterogeneous retrievers is a long-standing challenge, particularly for complex, multi-modal data such as videos. While typical fusion techniques are training-free, they rely solely on rank or score signals, disregarding candidates' representations. This work introduces Vote-in-Context (ViC), a generalized, training-free framework that re-thinks list-wise reranking and fusion as a zero-shot reasoning task for a Vision-Language Model (VLM). The core insight is to serialize both content evidence and retriever metadata directly within the VLM's prompt, allowing the model to adaptively weigh retriever consensus against visual-linguistic content. We demonstrate the generality of this framework by applying it to the challenging domain of cross-modal video retrieval. To this end, we introduce the S-Grid, a compact serialization map that represents each video as an image grid, optionally paired with subtitles to enable list-wise reasoning over video candidates. ViC is evaluated both as a single-list reranker, where it dramatically improves the precision of individual retrievers, and as an ensemble fuser, where it consistently outperforms strong baselines like CombSUM. Across video retrieval benchmarks including ActivityNet and VATEX, the framework establishes new state-of-the-art zero-shot retrieval performance, demonstrating its effectiveness in handling complex visual and temporal signals alongside text. In zero-shot settings, ViC achieves Recall@1 scores of 87.1% (t2v) / 89.0% (v2t) on MSR-VTT and 99.6% (v2t) on VATEX, representing massive gains of up to +40 Recall@1 over previous state-of-the-art baselines. We present ViC as a simple, reproducible, and highly effective recipe for turning modern VLMs into powerful zero-shot rerankers and fusers. Code and resources are publicly available at: this https URL 

---
# Calculating Web Impact Factor for University Websites of Jammu and Kashmir: A Study 

**Authors**: Muneer Ahmad, M Sadik Batcha, Wasim Rashid, Obaid Hafiz  

**Link**: [PDF](https://arxiv.org/pdf/2511.01496)  

**Abstract**: This paper examines and explores the web impact factor through a webometric study of the present 12 University Websites of Jammu and Kashmir. Identifies the domain systems of the websites; analyzes the number of web pages and link pages, and calculates the External Link WIF or simple web impact factor (WIF) and external web impact factor of all the University websites. Also reflects that some university websites have higher number of web pages, but correspondingly their link pages are very small in number and websites fall behind in their simple and external link web impact factor. It found that the Cluster University of Jammu ranked 1 (0.9018) in Internal Link WIF of Websites in Jammu and Kashmir. Shri Mata Vaishno Devi University ranked 1 (0.7249) in External Link Web Impact Factor. 

---
# Impact and Relevance of Cognition Journal in the Field of Cognitive Science: An Evaluation 

**Authors**: M Sadik Batcha, Younis Rashid Dar, Muneer Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2511.01485)  

**Abstract**: This study aims to present a scientometric analysis of the journal titled Cognition for a period of 20 years from 1999 to 2018. The present study was conducted with an aim to provide a summary of research activity in current journal and characterize its most aspects. The research coverage includes the year wise distribution of articles, authors, institutions, countries and citation analysis of the journal. The analysis showed that 2870 papers were published in journal of Cognition from 1999 to 2018. The study identified top 20 prolific authors, institutions and countries of the journal. Researchers from USA have been made the most percentage of contributions. 

---
# RAGSmith: A Framework for Finding the Optimal Composition of Retrieval-Augmented Generation Methods Across Datasets 

**Authors**: Muhammed Yusuf Kartal, Suha Kagan Kose, Korhan Sevinç, Burak Aktas  

**Link**: [PDF](https://arxiv.org/pdf/2511.01386)  

**Abstract**: Retrieval-Augmented Generation (RAG) quality depends on many interacting choices across retrieval, ranking, augmentation, prompting, and generation, so optimizing modules in isolation is brittle. We introduce RAGSmith, a modular framework that treats RAG design as an end-to-end architecture search over nine technique families and 46{,}080 feasible pipeline configurations. A genetic search optimizes a scalar objective that jointly aggregates retrieval metrics (recall@k, mAP, nDCG, MRR) and generation metrics (LLM-Judge and semantic similarity). We evaluate on six Wikipedia-derived domains (Mathematics, Law, Finance, Medicine, Defense Industry, Computer Science), each with 100 questions spanning factual, interpretation, and long-answer types. RAGSmith finds configurations that consistently outperform naive RAG baseline by +3.8\% on average (range +1.2\% to +6.9\% across domains), with gains up to +12.5\% in retrieval and +7.5\% in generation. The search typically explores $\approx 0.2\%$ of the space ($\sim 100$ candidates) and discovers a robust backbone -- vector retrieval plus post-generation reflection/revision -- augmented by domain-dependent choices in expansion, reranking, augmentation, and prompt reordering; passage compression is never selected. Improvement magnitude correlates with question type, with larger gains on factual/long-answer mixes than interpretation-heavy sets. These results provide practical, domain-aware guidance for assembling effective RAG systems and demonstrate the utility of evolutionary search for full-pipeline optimization. 

---
# Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption Attacks on RAG Systems 

**Authors**: Minseok Kim, Hankook Lee, Hyungjoon Koo  

**Link**: [PDF](https://arxiv.org/pdf/2511.01268)  

**Abstract**: Large language models (LLMs) are reshaping numerous facets of our daily lives, leading widespread adoption as web-based services. Despite their versatility, LLMs face notable challenges, such as generating hallucinated content and lacking access to up-to-date information. Lately, to address such limitations, Retrieval-Augmented Generation (RAG) has emerged as a promising direction by generating responses grounded in external knowledge sources. A typical RAG system consists of i) a retriever that probes a group of relevant passages from a knowledge base and ii) a generator that formulates a response based on the retrieved content. However, as with other AI systems, recent studies demonstrate the vulnerability of RAG, such as knowledge corruption attacks by injecting misleading information. In response, several defense strategies have been proposed, including having LLMs inspect the retrieved passages individually or fine-tuning robust retrievers. While effective, such approaches often come with substantial computational costs.
In this work, we introduce RAGDefender, a resource-efficient defense mechanism against knowledge corruption (i.e., by data poisoning) attacks in practical RAG deployments. RAGDefender operates during the post-retrieval phase, leveraging lightweight machine learning techniques to detect and filter out adversarial content without requiring additional model training or inference. Our empirical evaluations show that RAGDefender consistently outperforms existing state-of-the-art defenses across multiple models and adversarial scenarios: e.g., RAGDefender reduces the attack success rate (ASR) against the Gemini model from 0.89 to as low as 0.02, compared to 0.69 for RobustRAG and 0.24 for Discern-and-Answer when adversarial passages outnumber legitimate ones by a factor of four (4x). 

---
# Object-Centric Analysis of XES Event Logs: Integrating OCED Modeling with SPARQL Queries 

**Authors**: Saba Latif, Huma Latif, Muhammad Rameez Ur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2511.00693)  

**Abstract**: Object Centric Event Data (OCED) has gained attention in recent years within the field of process mining. However, there are still many challenges, such as connecting the XES format to object-centric approaches to enable more insightful analysis. It is important for a process miner to understand the insights and dependencies of events in the event log to see what is going on in our processes. In previous standards, the dependencies of event logs are only used to show events, but not their dependencies among each other and actions in detail as described in OCEDO. There is more information in the event log when it is revealed using the OCEDO model. It becomes more understandable and easier to grasp the concepts and deal with the processes. This paper proposes the use of Object-Centric Event Data Ontology (OCEDO) to overcome the limitations of the XES standard in event logs for process mining. We demonstrate how the OCEDO approach, integrated with SPARQL queries, can be applied to the BPIC 2013 dataset to make the relationships between events and objects more explicit. It describes dealing with the meta descriptions of the OCEDO model on a business process challenge as an event log. It improves the completeness and readability of process data, suggesting that object-centric modeling allows for richer analyses than traditional approaches. 

---
# PolyRecommender: A Multimodal Recommendation System for Polymer Discovery 

**Authors**: Xin Wang, Yunhao Xiao, Rui Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2511.00375)  

**Abstract**: We introduce PolyRecommender, a multimodal discovery framework that integrates chemical language representations from PolyBERT with molecular graph-based representations from a graph encoder. The system first retrieves candidate polymers using language-based similarity and then ranks them using fused multimodal embeddings according to multiple target properties. By leveraging the complementary knowledge encoded in both modalities, PolyRecommender enables efficient retrieval and robust ranking across related polymer properties. Our work establishes a generalizable multimodal paradigm, advancing AI-guided design for the discovery of next-generation polymers. 

---
# IL-PCSR: Legal Corpus for Prior Case and Statute Retrieval 

**Authors**: Shounak Paul, Dhananjay Ghumare, Pawan Goyal, Saptarshi Ghosh, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00268)  

**Abstract**: Identifying/retrieving relevant statutes and prior cases/precedents for a given legal situation are common tasks exercised by law practitioners. Researchers to date have addressed the two tasks independently, thus developing completely different datasets and models for each task; however, both retrieval tasks are inherently related, e.g., similar cases tend to cite similar statutes (due to similar factual situation). In this paper, we address this gap. We propose IL-PCR (Indian Legal corpus for Prior Case and Statute Retrieval), which is a unique corpus that provides a common testbed for developing models for both the tasks (Statute Retrieval and Precedent Retrieval) that can exploit the dependence between the two. We experiment extensively with several baseline models on the tasks, including lexical models, semantic models and ensemble based on GNNs. Further, to exploit the dependence between the two tasks, we develop an LLM-based re-ranking approach that gives the best performance. 

---
# AI Powered High Quality Text to Video Generation with Enhanced Temporal Consistency 

**Authors**: Piyushkumar Patel  

**Link**: [PDF](https://arxiv.org/pdf/2511.00107)  

**Abstract**: Text to video generation has emerged as a critical frontier in generative artificial intelligence, yet existing approaches struggle with maintaining temporal consistency, compositional understanding, and fine grained control over visual narratives. We present MOVAI (Multimodal Original Video AI), a novel hierarchical framework that integrates compositional scene understanding with temporal aware diffusion models for high fidelity text to video synthesis. Our approach introduces three key innovations: (1) a Compositional Scene Parser (CSP) that decomposes textual descriptions into hierarchical scene graphs with temporal annotations, (2) a Temporal-Spatial Attention Mechanism (TSAM) that ensures coherent motion dynamics across frames while preserving spatial details, and (3) a Progressive Video Refinement (PVR) module that iteratively enhances video quality through multi-scale temporal reasoning. Extensive experiments on standard benchmarks demonstrate that MOVAI achieves state-of-the-art performance, improving video quality metrics by 15.3% in LPIPS, 12.7% in FVD, and 18.9% in user preference studies compared to existing methods. Our framework shows particular strength in generating complex multi-object scenes with realistic temporal dynamics and fine-grained semantic control. 

---
# Forecasting Occupational Survivability of Rickshaw Pullers in a Changing Climate with Wearable Data 

**Authors**: Masfiqur Rahaman, Maoyejatun Hasana, Shahad Shahriar Rahman, MD Sajid Mostafiz Noor, Razin Reaz Abedin, Md Toki Tahmid, Duncan Watson Parris, Tanzeem Choudhury, A. B. M. Alim Al Islam, Tauhidur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2511.00081)  

**Abstract**: Cycle rickshaw pullers are highly vulnerable to extreme heat, yet little is known about how their physiological biomarkers respond under such conditions. This study collected real-time weather and physiological data using wearable sensors from 100 rickshaw pullers in Dhaka, Bangladesh. In addition, interviews with 12 pullers explored their knowledge, perceptions, and experiences related to climate change. We developed a Linear Gaussian Bayesian Network (LGBN) regression model to predict key physiological biomarkers based on activity, weather, and demographic features. The model achieved normalized mean absolute error values of 0.82, 0.47, 0.65, and 0.67 for skin temperature, relative cardiac cost, skin conductance response, and skin conductance level, respectively. Using projections from 18 CMIP6 climate models, we layered the LGBN on future climate forecasts to analyze survivability for current (2023-2025) and future years (2026-2100). Based on thresholds of WBGT above 31.1°C and skin temperature above 35°C, 32% of rickshaw pullers already face high heat exposure risk. By 2026-2030, this percentage may rise to 37% with average exposure lasting nearly 12 minutes, or about two-thirds of the trip duration. A thematic analysis of interviews complements these findings, showing that rickshaw pullers recognize their increasing climate vulnerability and express concern about its effects on health and occupational survivability. 

---
