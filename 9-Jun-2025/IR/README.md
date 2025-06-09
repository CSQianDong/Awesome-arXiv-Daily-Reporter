# RecGPT: A Foundation Model for Sequential Recommendation 

**Authors**: Yangqin Jiang, Xubin Ren, Lianghao Xia, Da Luo, Kangyi Lin, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06270)  

**Abstract**: This work addresses a fundamental barrier in recommender systems: the inability to generalize across domains without extensive retraining. Traditional ID-based approaches fail entirely in cold-start and cross-domain scenarios where new users or items lack sufficient interaction history. Inspired by foundation models' cross-domain success, we develop a foundation model for sequential recommendation that achieves genuine zero-shot generalization capabilities. Our approach fundamentally departs from existing ID-based methods by deriving item representations exclusively from textual features. This enables immediate embedding of any new item without model retraining. We introduce unified item tokenization with Finite Scalar Quantization that transforms heterogeneous textual descriptions into standardized discrete tokens. This eliminates domain barriers that plague existing systems. Additionally, the framework features hybrid bidirectional-causal attention that captures both intra-item token coherence and inter-item sequential dependencies. An efficient catalog-aware beam search decoder enables real-time token-to-item mapping. Unlike conventional approaches confined to their training domains, RecGPT naturally bridges diverse recommendation contexts through its domain-invariant tokenization mechanism. Comprehensive evaluations across six datasets and industrial scenarios demonstrate consistent performance advantages. 

---
# Optimizing Recall or Relevance? A Multi-Task Multi-Head Approach for Item-to-Item Retrieval in Recommendation 

**Authors**: Jiang Zhang, Sumit Kumar, Wei Chang, Yubo Wang, Feng Zhang, Weize Mao, Hanchao Yu, Aashu Singh, Min Li, Qifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06239)  

**Abstract**: The task of item-to-item (I2I) retrieval is to identify a set of relevant and highly engaging items based on a given trigger item. It is a crucial component in modern recommendation systems, where users' previously engaged items serve as trigger items to retrieve relevant content for future engagement. However, existing I2I retrieval models in industry are primarily built on co-engagement data and optimized using the recall measure, which overly emphasizes co-engagement patterns while failing to capture semantic relevance. This often leads to overfitting short-term co-engagement trends at the expense of long-term benefits such as discovering novel interests and promoting content diversity. To address this challenge, we propose MTMH, a Multi-Task and Multi-Head I2I retrieval model that achieves both high recall and semantic relevance. Our model consists of two key components: 1) a multi-task learning loss for formally optimizing the trade-off between recall and semantic relevance, and 2) a multi-head I2I retrieval architecture for retrieving both highly co-engaged and semantically relevant items. We evaluate MTMH using proprietary data from a commercial platform serving billions of users and demonstrate that it can improve recall by up to 14.4% and semantic relevance by up to 56.6% compared with prior state-of-the-art models. We also conduct live experiments to verify that MTMH can enhance both short-term consumption metrics and long-term user-experience-related metrics. Our work provides a principled approach for jointly optimizing I2I recall and semantic relevance, which has significant implications for improving the overall performance of recommendation systems. 

---
# On the Merits of LLM-Based Corpus Enrichment 

**Authors**: Gal Zur, Tommy Mordo, Moshe Tennenholtz, Oren Kurland  

**Link**: [PDF](https://arxiv.org/pdf/2506.06015)  

**Abstract**: Generative AI (genAI) technologies -- specifically, large language models (LLMs) -- and search have evolving relations. We argue for a novel perspective: using genAI to enrich a document corpus so as to improve query-based retrieval effectiveness. The enrichment is based on modifying existing documents or generating new ones. As an empirical proof of concept, we use LLMs to generate documents relevant to a topic which are more retrievable than existing ones. In addition, we demonstrate the potential merits of using corpus enrichment for retrieval augmented generation (RAG) and answer attribution in question answering. 

---
# Respecting Temporal-Causal Consistency: Entity-Event Knowledge Graphs for Retrieval-Augmented Generation 

**Authors**: Ze Yu Zhang, Zitao Li, Yaliang Li, Bolin Ding, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2506.05939)  

**Abstract**: Retrieval-augmented generation (RAG) based on large language models often falters on narrative documents with inherent temporal structures. Standard unstructured RAG methods rely solely on embedding-similarity matching and lack any general mechanism to encode or exploit chronological information, while knowledge graph RAG (KG-RAG) frameworks collapse every mention of an entity into a single node, erasing the evolving context that drives many queries. To formalize this challenge and draw the community's attention, we construct ChronoQA, a robust and discriminative QA benchmark that measures temporal, causal, and character consistency understanding in narrative documents (e.g., novels) under the RAG setting. We then introduce Entity-Event RAG (E^2RAG), a dual-graph framework that keeps separate entity and event subgraphs linked by a bipartite mapping, thereby preserving the temporal and causal facets needed for fine-grained reasoning. Across ChronoQA, our approach outperforms state-of-the-art unstructured and KG-based RAG baselines, with notable gains on causal and character consistency queries. E^2RAG therefore offers a practical path to more context-aware retrieval for tasks that require precise answers grounded in chronological information. 

---
# Research on Personalized Financial Product Recommendation by Integrating Large Language Models and Graph Neural Networks 

**Authors**: Yushang Zhao, Yike Peng, Dannier Li, Yuxin Yang, Chengrui Zhou, Jing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.05873)  

**Abstract**: With the rapid growth of fintech, personalized financial product recommendations have become increasingly important. Traditional methods like collaborative filtering or content-based models often fail to capture users' latent preferences and complex relationships. We propose a hybrid framework integrating large language models (LLMs) and graph neural networks (GNNs). A pre-trained LLM encodes text data (e.g., user reviews) into rich feature vectors, while a heterogeneous user-product graph models interactions and social ties. Through a tailored message-passing mechanism, text and graph information are fused within the GNN to jointly optimize embeddings. Experiments on public and real-world financial datasets show our model outperforms standalone LLM or GNN in accuracy, recall, and NDCG, with strong interpretability. This work offers new insights for personalized financial recommendations and cross-modal fusion in broader recommendation tasks. 

---
# Generating Long Semantic IDs in Parallel for Recommendation 

**Authors**: Yupeng Hou, Jiacheng Li, Ashley Shin, Jinsung Jeon, Abhishek Santhanam, Wei Shao, Kaveh Hassani, Ning Yao, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2506.05781)  

**Abstract**: Semantic ID-based recommendation models tokenize each item into a small number of discrete tokens that preserve specific semantics, leading to better performance, scalability, and memory efficiency. While recent models adopt a generative approach, they often suffer from inefficient inference due to the reliance on resource-intensive beam search and multiple forward passes through the neural sequence model. As a result, the length of semantic IDs is typically restricted (e.g. to just 4 tokens), limiting their expressiveness. To address these challenges, we propose RPG, a lightweight framework for semantic ID-based recommendation. The key idea is to produce unordered, long semantic IDs, allowing the model to predict all tokens in parallel. We train the model to predict each token independently using a multi-token prediction loss, directly integrating semantics into the learning objective. During inference, we construct a graph connecting similar semantic IDs and guide decoding to avoid generating invalid IDs. Experiments show that scaling up semantic ID length to 64 enables RPG to outperform generative baselines by an average of 12.6% on the NDCG@10, while also improving inference efficiency. Code is available at: this https URL. 

---
# NGA: Non-autoregressive Generative Auction with Global Externalities for Advertising Systems 

**Authors**: Zuowu Zheng, Ze Wang, Fan Yang, Wenqing Ye, Weihua Huang, Wenqiang He, Teng Zhang, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05685)  

**Abstract**: Online advertising auctions are fundamental to internet commerce, demanding solutions that not only maximize revenue but also ensure incentive compatibility, high-quality user experience, and real-time efficiency. While recent learning-based auction frameworks have improved context modeling by capturing intra-list dependencies among ads, they remain limited in addressing global externalities and often suffer from inefficiencies caused by sequential processing. In this work, we introduce the Non-autoregressive Generative Auction with global externalities (NGA), a novel end-to-end framework designed for industrial online advertising. NGA explicitly models global externalities by jointly capturing the relationships among ads as well as the effects of adjacent organic content. To further enhance efficiency, NGA utilizes a non-autoregressive, constraint-based decoding strategy and a parallel multi-tower evaluator for unified list-wise reward and payment computation. Extensive offline experiments and large-scale online A/B testing on commercial advertising platforms demonstrate that NGA consistently outperforms existing methods in both effectiveness and efficiency. 

---
# Heterogeneous Sequel-Aware Graph Neural Networks for Sequential Learning 

**Authors**: Anushka Tiwari, Haimonti Dutta, Shahrzad Khanizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.05625)  

**Abstract**: Graph-based recommendation systems use higher-order user and item embeddings for next-item predictions. Dynamically adding collaborative signals from neighbors helps to use similar users' preferences during learning. While item-item correlations and their impact on recommendations have been studied, the efficacy of temporal item sequences for recommendations is much less explored. In this paper, we examine temporal item sequence (sequel-aware) embeddings along with higher-order user embeddings and show that sequel-aware Graph Neural Networks have better (or comparable) recommendation performance than graph-based recommendation systems that do not consider sequel information. Extensive empirical results comparing Heterogeneous Sequel-aware Graph Neural Networks (HSAL-GNNs) to other algorithms for sequential learning (such as transformers, graph neural networks, auto-encoders) are presented on three synthetic and three real-world datasets. Our results indicate that the incorporation of sequence information from items greatly enhances recommendations. 

---
# Recommender systems, stigmergy, and the tyranny of popularity 

**Authors**: Zackary Okun Dunivin, Paul E. Smaldino  

**Link**: [PDF](https://arxiv.org/pdf/2506.06162)  

**Abstract**: Scientific recommender systems, such as Google Scholar and Web of Science, are essential tools for discovery. Search algorithms that power work through stigmergy, a collective intelligence mechanism that surfaces useful paths through repeated engagement. While generally effective, this ``rich-get-richer'' dynamic results in a small number of high-profile papers that dominate visibility. This essay argues argue that these algorithm over-reliance on popularity fosters intellectual homogeneity and exacerbates structural inequities, stifling innovative and diverse perspectives critical for scientific progress. We propose an overhaul of search platforms to incorporate user-specific calibration, allowing researchers to manually adjust the weights of factors like popularity, recency, and relevance. We also advise platform developers on how word embeddings and LLMs could be implemented in ways that increase user autonomy. While our suggestions are particularly pertinent to aligning recommender systems with scientific values, these ideas are broadly applicable to information access systems in general. Designing platforms that increase user autonomy is an important step toward more robust and dynamic information 

---
# CLaMR: Contextualized Late-Interaction for Multimodal Content Retrieval 

**Authors**: David Wan, Han Wang, Elias Stengel-Eskin, Jaemin Cho, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2506.06144)  

**Abstract**: Online video web content is richly multimodal: a single video blends vision, speech, ambient audio, and on-screen text. Retrieval systems typically treat these modalities as independent retrieval sources, which can lead to noisy and subpar retrieval. We explore multimodal video content retrieval, where relevance can be scored from one particular modality or jointly across multiple modalities simultaneously. Consequently, an effective retriever must dynamically choose which modality (or set of modalities) best addresses the query. We introduce CLaMR, a multimodal, late-interaction retriever that jointly indexes 4 modalities: video frames, transcribed speech, on-screen text, and metadata. CLaMR jointly encodes all modalities with a unified multimodal backbone for improved contextualization and is trained to enhance dynamic modality selection via two key innovations. First, given the lack of training data for multimodal retrieval, we introduce MultiVENT 2.0++, a large-scale synthetic training dataset built on MultiVENT 2.0 (event-centric videos in various languages paired with queries) with modality-targeted queries. Next, we propose a modality-aware loss that jointly trains according to a standard contrastive objective alongside an objective for learning correct modality usage. On the test sets of MultiVENT 2.0++ and MSRVTT, conventional aggregation strategies, such as averaging similarities for baseline retrievers, degrade performance by introducing noise from irrelevant modalities. In contrast, CLaMR consistently outperforms existing retrievers: on MultiVENT 2.0++, CLaMR improves nDCG@10 by 25.6 over the best single-modality retriever and by 35.4 over the best multi-modality retriever. We illustrate CLaMR's downstream utility on long-video QA, retrieving relevant frames and obtaining a 3.50% boost over LanguageBind on Video-MME and 1.42% over dense sampling on LongVideoBench. 

---
# Phonetically-Augmented Discriminative Rescoring for Voice Search Error Correction 

**Authors**: Christophe Van Gysel, Maggie Wu, Lyan Verwimp, Caglar Tirkaz, Marco Bertola, Zhihong Lei, Youssef Oualil  

**Link**: [PDF](https://arxiv.org/pdf/2506.06117)  

**Abstract**: End-to-end (E2E) Automatic Speech Recognition (ASR) models are trained using paired audio-text samples that are expensive to obtain, since high-quality ground-truth data requires human annotators. Voice search applications, such as digital media players, leverage ASR to allow users to search by voice as opposed to an on-screen keyboard. However, recent or infrequent movie titles may not be sufficiently represented in the E2E ASR system's training data, and hence, may suffer poor recognition.
In this paper, we propose a phonetic correction system that consists of (a) a phonetic search based on the ASR model's output that generates phonetic alternatives that may not be considered by the E2E system, and (b) a rescorer component that combines the ASR model recognition and the phonetic alternatives, and select a final system output.
We find that our approach improves word error rate between 4.4 and 7.6% relative on benchmarks of popular movie titles over a series of competitive baselines. 

---
# A Novel, Human-in-the-Loop Computational Grounded Theory Framework for Big Social Data 

**Authors**: Lama Alqazlan, Zheng Fang, Michael Castelle, Rob Procter  

**Link**: [PDF](https://arxiv.org/pdf/2506.06083)  

**Abstract**: The availability of big data has significantly influenced the possibilities and methodological choices for conducting large-scale behavioural and social science research. In the context of qualitative data analysis, a major challenge is that conventional methods require intensive manual labour and are often impractical to apply to large datasets. One effective way to address this issue is by integrating emerging computational methods to overcome scalability limitations. However, a critical concern for researchers is the trustworthiness of results when Machine Learning (ML) and Natural Language Processing (NLP) tools are used to analyse such data. We argue that confidence in the credibility and robustness of results depends on adopting a 'human-in-the-loop' methodology that is able to provide researchers with control over the analytical process, while retaining the benefits of using ML and NLP. With this in mind, we propose a novel methodological framework for Computational Grounded Theory (CGT) that supports the analysis of large qualitative datasets, while maintaining the rigour of established Grounded Theory (GT) methodologies. To illustrate the framework's value, we present the results of testing it on a dataset collected from Reddit in a study aimed at understanding tutors' experiences in the gig economy. 

---
# The NetMob25 Dataset: A High-resolution Multi-layered View of Individual Mobility in Greater Paris Region 

**Authors**: Alexandre Chasse, Anne J. Kouam, Aline C. Viana, Razvan Stanica, Wellington V. Lobato, Geymerson Ramos, Geoffrey Deperle, Abdelmounaim Bouroudi, Suzanne Bussod, Fernando Molano  

**Link**: [PDF](https://arxiv.org/pdf/2506.05903)  

**Abstract**: High-quality mobility data remains scarce despite growing interest from researchers and urban stakeholders in understanding individual-level movement patterns. The Netmob25 Data Challenge addresses this gap by releasing a unique GPS-based mobility dataset derived from the EMG 2023 GNSS-based mobility survey conducted in the Ile-de-France region (Greater Paris area), France. This dataset captures detailed daily mobility over a full week for 3,337 volunteer residents aged 16 to 80, collected between October 2022 and May 2023. Each participant was equipped with a dedicated GPS tracking device configured to record location points every 2-3 seconds and was asked to maintain a digital or paper logbook of their trips. All inferred mobility traces were algorithmically processed and validated through follow-up phone interviews.
The dataset includes three components: (i) an Individuals database describing demographic, socioeconomic, and household characteristics; (ii) a Trips database with over 80,000 annotated displacements including timestamps, transport modes, and trip purposes; and (iii) a Raw GPS Traces database comprising about 500 million high-frequency points. A statistical weighting mechanism is provided to support population-level estimates. An extensive anonymization pipeline was applied to the GPS traces to ensure GDPR compliance while preserving analytical value. Access to the dataset requires acceptance of the challenge's Terms and Conditions and signing a Non-Disclosure Agreement. This paper describes the survey design, collection protocol, processing methodology, and characteristics of the released dataset. 

---
# TriPSS: A Tri-Modal Keyframe Extraction Framework Using Perceptual, Structural, and Semantic Representations 

**Authors**: Mert Can Cakmak, Nitin Agarwal, Diwash Poudel  

**Link**: [PDF](https://arxiv.org/pdf/2506.05395)  

**Abstract**: Efficient keyframe extraction is critical for effective video summarization and retrieval, yet capturing the complete richness of video content remains challenging. In this work, we present TriPSS, a novel tri-modal framework that effectively integrates perceptual cues from color features in the CIELAB space, deep structural embeddings derived from ResNet-50, and semantic context from frame-level captions generated by Llama-3.2-11B-Vision-Instruct. By fusing these diverse modalities using principal component analysis, TriPSS constructs robust multi-modal embeddings that enable adaptive segmentation of video content via HDBSCAN clustering. A subsequent refinement stage incorporating quality assessment and duplicate filtering ensures that the final keyframe set is both concise and semantically rich. Comprehensive evaluations on benchmark datasets TVSum20 and SumMe demonstrate that TriPSS achieves state-of-the-art performance, substantially outperforming traditional unimodal and previous multi-modal methods. These results underscore TriPSS's ability to capture nuanced visual and semantic information, thereby setting a new benchmark for video content understanding in large-scale retrieval scenarios. 

---
