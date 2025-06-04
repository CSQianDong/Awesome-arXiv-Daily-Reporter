# MMM4Rec: An Transfer-Efficient Framework for Multi-modal Sequential Recommendation 

**Authors**: Hao Fan, Yanrong Hu, Kai Fang, Qingyang Liu, Hongjiu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02916)  

**Abstract**: Sequential Recommendation (SR) systems model user preferences by analyzing interaction histories. Although transferable multi-modal SR architectures demonstrate superior performance compared to traditional ID-based approaches, current methods incur substantial fine-tuning costs when adapting to new domains due to complex optimization requirements and negative transfer effects - a significant deployment bottleneck that hinders engineers from efficiently repurposing pre-trained models for novel application scenarios with minimal tuning overhead. We propose MMM4Rec (Multi-Modal Mamba for Sequential Recommendation), a novel multi-modal SR framework that incorporates a dedicated algebraic constraint mechanism for efficient transfer learning. By combining State Space Duality (SSD)'s temporal decay properties with a time-aware modeling design, our model dynamically prioritizes key modality information, overcoming limitations of Transformer-based approaches. The framework implements a constrained two-stage process: (1) sequence-level cross-modal alignment via shared projection matrices, followed by (2) temporal fusion using our newly designed Cross-SSD module and dual-channel Fourier adaptive filtering. This architecture maintains semantic consistency while suppressing noise propagation.MMM4Rec achieves rapid fine-tuning convergence with simple cross-entropy loss, significantly improving multi-modal recommendation accuracy while maintaining strong transferability. Extensive experiments demonstrate MMM4Rec's state-of-the-art performance, achieving the maximum 31.78% NDCG@10 improvement over existing models and exhibiting 10 times faster average convergence speed when transferring to large-scale downstream datasets. 

---
# DeepShop: A Benchmark for Deep Research Shopping Agents 

**Authors**: Yougang Lyu, Xiaoyu Zhang, Lingyong Yan, Maarten de Rijke, Zhaochun Ren, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02839)  

**Abstract**: Web agents for online shopping have shown great promise in automating user interactions across e-commerce platforms. Benchmarks for assessing such agents do not reflect the complexity of real-world shopping scenarios, as they often consist of overly simple queries with deterministic paths, such as "Find iPhone 15." Real shopping scenarios are inherently more layered, involving multi-dimensional product attributes, search filters, and user-specific sorting preferences. To address this gap, we introduce DeepShop, a benchmark designed to evaluate web agents in complex and realistic online shopping environments. DeepShop comprises three key components. (1) Query diversity evolution: Starting from real user queries, we generate diverse queries across five popular online shopping domains. (2) Query complexity evolution: We further evolve these queries to increase complexity, considering product attributes, search filters, and sorting preferences, and classify them into three levels: easy, medium, and hard, based on the number of evolutions. (3) Fine-grained and holistic evaluation: We propose an automated evaluation framework that assesses agent performance in terms of fine-grained aspects (product attributes, search filters, and sorting preferences) and reports the overall success rate through holistic evaluation. We conduct a systematic evaluation of retrieval-augmented generation (RAG) methods, web agents, and deep research systems. Results show that RAG struggles with complex queries due to its lack of web interaction, while other methods face significant challenges with filters and sorting preferences, leading to low overall success rates. We also perform cross-category, complexity-based evaluations and error analyses to support the advancement of deep research shopping agents. 

---
# Combining social relations and interaction data in Recommender System with Graph Convolution Collaborative Filtering 

**Authors**: Tin T. Tran, Vaclav Snasel, Loc Tan Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02834)  

**Abstract**: A recommender system is an important subject in the field of data mining, where the item rating information from users is exploited and processed to make suitable recommendations with all other users. The recommender system creates convenience for e-commerce users and stimulates the consumption of items that are suitable for users. In addition to e-commerce, a recommender system is also used to provide recommendations on books to read, movies to watch, courses to take or websites to visit. Similarity between users is an important impact for recommendation, which could be calculated from the data of past user ratings of the item by methods of collaborative filtering, matrix factorization or singular vector decomposition. In the development of graph data mining techniques, the relationships between users and items can be represented by matrices from which collaborative filtering could be done with the larger database, more accurate and faster in calculation. All these data can be represented graphically and mined by today's highly developed graph neural network models. On the other hand, users' social friendship data also influence consumption habits because recommendations from friends will be considered more carefully than information sources. However, combining a user's friend influence and the similarity between users whose similar shopping habits is challenging. Because the information is noisy and it affects each particular data set in different ways. In this study, we present the input data processing method to remove outliers which are single reviews or users with little interaction with the items; the next proposed model will combine the social relationship data and the similarity in the rating history of users to improve the accuracy and recall of the recommender system. 

---
# UTCS: Effective Unsupervised Temporal Community Search with Pre-training of Temporal Dynamics and Subgraph Knowledge 

**Authors**: Yue Zhang, Yankai Chen, Yingli Zhou, Yucan Guo, Xiaolin Han, Chenhao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.02784)  

**Abstract**: In many real-world applications, the evolving relationships between entities can be modeled as temporal graphs, where each edge has a timestamp representing the interaction time.
As a fundamental problem in graph analysis, {\it community search (CS)} in temporal graphs has received growing attention but exhibits two major limitations: (1) Traditional methods typically require predefined subgraph structures, which are not always known in advance. (2) Learning-based methods struggle to capture temporal interaction information. To fill this research gap, in this paper, we propose an effective \textbf{U}nsupervised \textbf{T}emporal \textbf{C}ommunity \textbf{S}earch with pre-training of temporal dynamics and subgraph knowledge model (\textbf{\model}). \model~contains two key stages: offline pre-training and online search. In the first stage, we introduce multiple learning objectives to facilitate the pre-training process in the unsupervised learning setting. In the second stage, we identify a candidate subgraph and compute community scores using the pre-trained node representations and a novel scoring mechanism to determine the final community members. Experiments on five real-world datasets demonstrate the effectiveness. 

---
# Learning Binarized Representations with Pseudo-positive Sample Enhancement for Efficient Graph Collaborative Filtering 

**Authors**: Yankai Chen, Yue Que, Xinni Zhang, Chen Ma, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2506.02750)  

**Abstract**: Learning vectorized embeddings is fundamental to many recommender systems for user-item matching. To enable efficient online inference, representation binarization, which embeds latent features into compact binary sequences, has recently shown significant promise in optimizing both memory usage and computational overhead. However, existing approaches primarily focus on numerical quantization, neglecting the associated information loss, which often results in noticeable performance degradation. To address these issues, we study the problem of graph representation binarization for efficient collaborative filtering. Our findings indicate that explicitly mitigating information loss at various stages of embedding binarization has a significant positive impact on performance. Building on these insights, we propose an enhanced framework, BiGeaR++, which specifically leverages supervisory signals from pseudo-positive samples, incorporating both real item data and latent embedding samples. Compared to its predecessor BiGeaR, BiGeaR++ introduces a fine-grained inference distillation mechanism and an effective embedding sample synthesis approach. Empirical evaluations across five real-world datasets demonstrate that the new designs in BiGeaR++ work seamlessly well with other modules, delivering substantial improvements of around 1%-10% over BiGeaR and thus achieving state-of-the-art performance compared to the competing methods. Our implementation is available at this https URL. 

---
# NextQuill: Causal Preference Modeling for Enhancing LLM Personalization 

**Authors**: Xiaoyan Zhao, Juntao You, Yang Zhang, Wenjie Wang, Hong Cheng, Fuli Feng, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.02368)  

**Abstract**: Personalizing large language models (LLMs) for individual users has become increasingly important as they are progressively integrated into real-world applications to support users' daily lives. However, existing personalization approaches often fail to distinguish which components of model predictions and training data truly reflect user preferences, leading to superficial personalization alignment. In this paper, we introduce NextQuill, a novel LLM personalization alignment framework grounded in causal preference modeling. We approach personalization from a causal perspective, treating both model predictions and ground-truth data generation as outcomes influenced by user preferences, along with other factors. We define the true preference effect as the causal impact of user history (which reflects preferences) on each token prediction or data generation instance, estimated through causal intervention techniques. Building on this insight, NextQuill introduces two complementary alignment strategies: (1) aligning model-internal causal preference effects on predictions with those reflected in ground-truth data, rather than indiscriminately fitting predictions, and (2) focusing on fitting preference-bearing tokens identified via ground-truth data preference effects, rather than treating all tokens uniformly. By integrating these strategies, NextQuill shifts the alignment process toward learning from causal preference effects, facilitating more effective and personalized adaptation. Experiments across multiple personalization benchmarks demonstrate that NextQuill significantly improves personalization quality, offering a principled, causal foundation for LLM personalization. Our codes are available on this https URL. 

---
# TransAct V2: Lifelong User Action Sequence Modeling on Pinterest Recommendation 

**Authors**: Xue Xia, Saurabh Vishwas Joshi, Kousik Rajesh, Kangnan Li, Yangyi Lu, Nikil Pancha, Dhruvil Deven Badani, Jiajing Xu, Pong Eksombatchai  

**Link**: [PDF](https://arxiv.org/pdf/2506.02267)  

**Abstract**: Modeling user action sequences has become a popular focus in industrial recommendation system research, particularly for Click-Through Rate (CTR) prediction tasks. However, industry-scale CTR models often rely on short user sequences, limiting their ability to capture long-term behavior. Additionally, these models typically lack an integrated action-prediction task within a point-wise ranking framework, reducing their predictive power. They also rarely address the infrastructure challenges involved in efficiently serving large-scale sequential models. In this paper, we introduce TransAct V2, a production model for Pinterest's Homefeed ranking system, featuring three key innovations: (1) leveraging very long user sequences to improve CTR predictions, (2) integrating a Next Action Loss function for enhanced user action forecasting, and (3) employing scalable, low-latency deployment solutions tailored to handle the computational demands of extended user action sequences. 

---
# Towards Human-like Preference Profiling in Sequential Recommendation 

**Authors**: Zhongyu Ouyang, Qianlong Wen, Chunhui Zhang, Yanfang Ye, Soroush Vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2506.02261)  

**Abstract**: Sequential recommendation systems aspire to profile users by interpreting their interaction histories, echoing how humans make decisions by weighing experience, relative preference strength, and situational relevance. Yet, existing large language model (LLM)-based recommenders often fall short of mimicking the flexible, context-aware decision strategies humans exhibit, neglecting the structured, dynamic, and context-aware mechanisms fundamental to human behaviors. To bridge this gap, we propose RecPO, a preference optimization framework that models structured feedback and contextual delay to emulate human-like prioritization in sequential recommendation RecPO exploits adaptive reward margins based on inferred preference hierarchies and temporal signals, enabling the model to favor immediately relevant items and to distinguish between varying degrees of preference and aversion. Extensive experiments across five real-world datasets demonstrate that RecPO not only yields performance gains over state-of-the-art baselines, but also mirrors key characteristics of human decision-making: favoring timely satisfaction, maintaining coherent preferences, and exercising discernment under shifting contexts. 

---
# A Dynamic Framework for Semantic Grouping of Common Data Elements (CDE) Using Embeddings and Clustering 

**Authors**: Madan Krishnamurthy, Daniel Korn, Melissa A Haendel, Christopher J Mungall, Anne E Thessen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02160)  

**Abstract**: This research aims to develop a dynamic and scalable framework to facilitate harmonization of Common Data Elements (CDEs) across heterogeneous biomedical datasets by addressing challenges such as semantic heterogeneity, structural variability, and context dependence to streamline integration, enhance interoperability, and accelerate scientific discovery. Our methodology leverages Large Language Models (LLMs) for context-aware text embeddings that convert CDEs into dense vectors capturing semantic relationships and patterns. These embeddings are clustered using Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) to group semantically similar CDEs. The framework incorporates four key steps: (1) LLM-based text embedding to mathematically represent semantic context, (2) unsupervised clustering of embeddings via HDBSCAN, (3) automated labeling using LLM summarization, and (4) supervised learning to train a classifier assigning new or unclustered CDEs to labeled clusters. Evaluated on the NIH NLM CDE Repository with over 24,000 CDEs, the system identified 118 meaningful clusters at an optimized minimum cluster size of 20. The classifier achieved 90.46 percent overall accuracy, performing best in larger categories. External validation against Gravity Projects Social Determinants of Health domains showed strong agreement (Adjusted Rand Index 0.52, Normalized Mutual Information 0.78), indicating that embeddings effectively capture cluster characteristics. This adaptable and scalable approach offers a practical solution to CDE harmonization, improving selection efficiency and supporting ongoing data interoperability. 

---
# Retrieval-Augmented Generation as Noisy In-Context Learning: A Unified Theory and Risk Bounds 

**Authors**: Yang Guo, Yutian Tao, Yifei Ming, Robert D. Nowak, Yingyu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03100)  

**Abstract**: Retrieval-augmented generation (RAG) has seen many empirical successes in recent years by aiding the LLM with external knowledge. However, its theoretical aspect has remained mostly unexplored. In this paper, we propose the first finite-sample generalization bound for RAG in in-context linear regression and derive an exact bias-variance tradeoff. Our framework views the retrieved texts as query-dependent noisy in-context examples and recovers the classical in-context learning (ICL) and standard RAG as the limit cases. Our analysis suggests that an intrinsic ceiling on generalization error exists on RAG as opposed to the ICL. Furthermore, our framework is able to model retrieval both from the training data and from external corpora by introducing uniform and non-uniform RAG noise. In line with our theory, we show the sample efficiency of ICL and RAG empirically with experiments on common QA benchmarks, such as Natural Questions and TriviaQA. 

---
# Leveraging Information Retrieval to Enhance Spoken Language Understanding Prompts in Few-Shot Learning 

**Authors**: Pierre Lepagnol, Sahar Ghannay, Thomas Gerald, Christophe Servan, Sophie Rosset  

**Link**: [PDF](https://arxiv.org/pdf/2506.03035)  

**Abstract**: Understanding user queries is fundamental in many applications, such as home assistants, booking systems, or recommendations. Accordingly, it is crucial to develop accurate Spoken Language Understanding (SLU) approaches to ensure the reliability of the considered system. Current State-of-the-Art SLU techniques rely on large amounts of training data; however, only limited annotated examples are available for specific tasks or languages.
In the meantime, instruction-tuned large language models (LLMs) have shown exceptional performance on unseen tasks in a few-shot setting when provided with adequate prompts. In this work, we propose to explore example selection by leveraging Information retrieval (IR) approaches to build an enhanced prompt that is applied to an SLU task. We evaluate the effectiveness of the proposed method on several SLU benchmarks. Experimental results show that lexical IR methods significantly enhance performance without increasing prompt length. 

---
# INESC-ID @ eRisk 2025: Exploring Fine-Tuned, Similarity-Based, and Prompt-Based Approaches to Depression Symptom Identification 

**Authors**: Diogo A.P. Nunes, Eugénio Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2506.02924)  

**Abstract**: In this work, we describe our team's approach to eRisk's 2025 Task 1: Search for Symptoms of Depression. Given a set of sentences and the Beck's Depression Inventory - II (BDI) questionnaire, participants were tasked with submitting up to 1,000 sentences per depression symptom in the BDI, sorted by relevance. Participant submissions were evaluated according to standard Information Retrieval (IR) metrics, including Average Precision (AP) and R-Precision (R-PREC). The provided training data, however, consisted of sentences labeled as to whether a given sentence was relevant or not w.r.t. one of BDI's symptoms. Due to this labeling limitation, we framed our development as a binary classification task for each BDI symptom, and evaluated accordingly. To that end, we split the available labeled data into training and validation sets, and explored foundation model fine-tuning, sentence similarity, Large Language Model (LLM) prompting, and ensemble techniques. The validation results revealed that fine-tuning foundation models yielded the best performance, particularly when enhanced with synthetic data to mitigate class imbalance. We also observed that the optimal approach varied by symptom. Based on these insights, we devised five independent test runs, two of which used ensemble methods. These runs achieved the highest scores in the official IR evaluation, outperforming submissions from 16 other teams. 

---
# Token and Span Classification for Entity Recognition in French Historical Encyclopedias 

**Authors**: Ludovic Moncla, Hédi Zeghidi  

**Link**: [PDF](https://arxiv.org/pdf/2506.02872)  

**Abstract**: Named Entity Recognition (NER) in historical texts presents unique challenges due to non-standardized language, archaic orthography, and nested or overlapping entities. This study benchmarks a diverse set of NER approaches, ranging from classical Conditional Random Fields (CRFs) and spaCy-based models to transformer-based architectures such as CamemBERT and sequence-labeling models like Flair. Experiments are conducted on the GeoEDdA dataset, a richly annotated corpus derived from 18th-century French encyclopedias. We propose framing NER as both token-level and span-level classification to accommodate complex nested entity structures typical of historical documents. Additionally, we evaluate the emerging potential of few-shot prompting with generative language models for low-resource scenarios. Our results demonstrate that while transformer-based models achieve state-of-the-art performance, especially on nested entities, generative models offer promising alternatives when labeled data are scarce. The study highlights ongoing challenges in historical NER and suggests avenues for hybrid approaches combining symbolic and neural methods to better capture the intricacies of early modern French text. 

---
# Evaluating Named Entity Recognition Models for Russian Cultural News Texts: From BERT to LLM 

**Authors**: Maria Levchenko  

**Link**: [PDF](https://arxiv.org/pdf/2506.02589)  

**Abstract**: This paper addresses the challenge of Named Entity Recognition (NER) for person names within the specialized domain of Russian news texts concerning cultural events. The study utilizes the unique SPbLitGuide dataset, a collection of event announcements from Saint Petersburg spanning 1999 to 2019. A comparative evaluation of diverse NER models is presented, encompassing established transformer-based architectures such as DeepPavlov, RoBERTa, and SpaCy, alongside recent Large Language Models (LLMs) including GPT-3.5, GPT-4, and GPT-4o. Key findings highlight the superior performance of GPT-4o when provided with specific prompting for JSON output, achieving an F1 score of 0.93. Furthermore, GPT-4 demonstrated the highest precision at 0.99. The research contributes to a deeper understanding of current NER model capabilities and limitations when applied to morphologically rich languages like Russian within the cultural heritage domain, offering insights for researchers and practitioners. Follow-up evaluation with GPT-4.1 (April 2025) achieves F1=0.94 for both simple and structured prompts, demonstrating rapid progress across model families and simplified deployment requirements. 

---
# Multilingual Information Retrieval with a Monolingual Knowledge Base 

**Authors**: Yingying Zhuang, Aman Gupta, Anurag Beniwal  

**Link**: [PDF](https://arxiv.org/pdf/2506.02527)  

**Abstract**: Multilingual information retrieval has emerged as powerful tools for expanding knowledge sharing across languages. On the other hand, resources on high quality knowledge base are often scarce and in limited languages, therefore an effective embedding model to transform sentences from different languages into a feature vector space same as the knowledge base language becomes the key ingredient for cross language knowledge sharing, especially to transfer knowledge available in high-resource languages to low-resource ones. In this paper we propose a novel strategy to fine-tune multilingual embedding models with weighted sampling for contrastive learning, enabling multilingual information retrieval with a monolingual knowledge base. We demonstrate that the weighted sampling strategy produces performance gains compared to standard ones by up to 31.03\% in MRR and up to 33.98\% in Recall@3. Additionally, our proposed methodology is language agnostic and applicable for both multilingual and code switching use cases. 

---
# Entity Image and Mixed-Modal Image Retrieval Datasets 

**Authors**: Cristian-Ioan Blaga, Paul Suganthan, Sahil Dua, Krishna Srinivasan, Enrique Alfonseca, Peter Dornbach, Tom Duerig, Imed Zitouni, Zhe Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.02291)  

**Abstract**: Despite advances in multimodal learning, challenging benchmarks for mixed-modal image retrieval that combines visual and textual information are lacking. This paper introduces a novel benchmark to rigorously evaluate image retrieval that demands deep cross-modal contextual understanding. We present two new datasets: the Entity Image Dataset (EI), providing canonical images for Wikipedia entities, and the Mixed-Modal Image Retrieval Dataset (MMIR), derived from the WIT dataset. The MMIR benchmark features two challenging query types requiring models to ground textual descriptions in the context of provided visual entities: single entity-image queries (one entity image with descriptive text) and multi-entity-image queries (multiple entity images with relational text). We empirically validate the benchmark's utility as both a training corpus and an evaluation set for mixed-modal retrieval. The quality of both datasets is further affirmed through crowd-sourced human annotations. The datasets are accessible through the GitHub page: this https URL. 

---
# Evaluating the Unseen Capabilities: How Many Theorems Do LLMs Know? 

**Authors**: Xiang Li, Jiayi Xin, Qi Long, Weijie J. Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.02058)  

**Abstract**: Accurate evaluation of large language models (LLMs) is crucial for understanding their capabilities and guiding their development. However, current evaluations often inconsistently reflect the actual capacities of these models. In this paper, we demonstrate that one of many contributing factors to this \textit{evaluation crisis} is the oversight of unseen knowledge -- information encoded by LLMs but not directly observed or not yet observed during evaluations. We introduce KnowSum, a statistical framework designed to provide a more comprehensive assessment by quantifying the unseen knowledge for a class of evaluation tasks. KnowSum estimates the unobserved portion by extrapolating from the appearance frequencies of observed knowledge instances. We demonstrate the effectiveness and utility of KnowSum across three critical applications: estimating total knowledge, evaluating information retrieval effectiveness, and measuring output diversity. Our experiments reveal that a substantial volume of knowledge is omitted when relying solely on observed LLM performance. Importantly, KnowSum yields significantly different comparative rankings for several common LLMs based on their internal knowledge. 

---
# No Free Lunch in Active Learning: LLM Embedding Quality Dictates Query Strategy Success 

**Authors**: Lukas Rauch, Moritz Wirth, Denis Huseljic, Marek Herde, Bernhard Sick, Matthias Aßenmacher  

**Link**: [PDF](https://arxiv.org/pdf/2506.01992)  

**Abstract**: The advent of large language models (LLMs) capable of producing general-purpose representations lets us revisit the practicality of deep active learning (AL): By leveraging frozen LLM embeddings, we can mitigate the computational costs of iteratively fine-tuning large backbones. This study establishes a benchmark and systematically investigates the influence of LLM embedding quality on query strategies in deep AL. We employ five top-performing models from the massive text embedding benchmark (MTEB) leaderboard and two baselines for ten diverse text classification tasks. Our findings reveal key insights: First, initializing the labeled pool using diversity-based sampling synergizes with high-quality embeddings, boosting performance in early AL iterations. Second, the choice of the optimal query strategy is sensitive to embedding quality. While the computationally inexpensive Margin sampling can achieve performance spikes on specific datasets, we find that strategies like Badge exhibit greater robustness across tasks. Importantly, their effectiveness is often enhanced when paired with higher-quality embeddings. Our results emphasize the need for context-specific evaluation of AL strategies, as performance heavily depends on embedding quality and the target task. 

---
