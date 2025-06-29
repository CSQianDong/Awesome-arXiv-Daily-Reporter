# GLoSS: Generative Language Models with Semantic Search for Sequential Recommendation 

**Authors**: Krishna Acharya, Aleksandr V. Petrov, Juba Ziani  

**Link**: [PDF](https://arxiv.org/pdf/2506.01910)  

**Abstract**: We propose Generative Low-rank language model with Semantic Search (GLoSS), a generative recommendation framework that combines large language models with dense retrieval for sequential recommendation. Unlike prior methods such as GPT4Rec, which rely on lexical matching via BM25, GLoSS uses semantic search to retrieve relevant items beyond lexical matching. For query generation, we employ 4-bit quantized LlaMA-3 models fine-tuned with low-rank adaptation (LoRA), enabling efficient training and inference on modest hardware. We evaluate GLoSS on three real-world Amazon review datasets: Beauty, Toys, and Sports, and find that it achieves state-of-the-art performance. Compared to traditional ID-based baselines, GLoSS improves Recall@5 by 33.3%, 52.8%, and 15.2%, and NDCG@5 by 30.0%, 42.6%, and 16.1%, respectively. It also outperforms LLM-based recommenders such as P5, GPT4Rec, LlamaRec and E4SRec with Recall@5 gains of 4.3%, 22.8%, and 29.5%. Additionally, user segment evaluations show that GLoSS performs particularly well for cold-start users in the Amazon Toys and Sports datasets, and benefits from longer user histories in Amazon Beauty dataset, demonstrating robustness across different levels of interaction lengths. 

---
# When Should Dense Retrievers Be Updated in Evolving Corpora? Detecting Out-of-Distribution Corpora Using GradNormIR 

**Authors**: Dayoon Ko, Jinyoung Kim, Sohyeon Kim, Jinhyuk Kim, Jaehoon Lee, Seonghak Song, Minyoung Lee, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.01877)  

**Abstract**: Dense retrievers encode texts into embeddings to efficiently retrieve relevant documents from large databases in response to user queries. However, real-world corpora continually evolve, leading to a shift from the original training distribution of the retriever. Without timely updates or retraining, indexing newly emerging documents can degrade retrieval performance for future queries. Thus, identifying when a dense retriever requires an update is critical for maintaining robust retrieval systems. In this paper, we propose a novel task of predicting whether a corpus is out-of-distribution (OOD) relative to a dense retriever before indexing. Addressing this task allows us to proactively manage retriever updates, preventing potential retrieval failures. We introduce GradNormIR, an unsupervised approach that leverages gradient norms to detect OOD corpora effectively. Experiments on the BEIR benchmark demonstrate that GradNormIR enables timely updates of dense retrievers in evolving document collections, significantly enhancing retrieval robustness and efficiency. 

---
# SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation 

**Authors**: Yinghui Liu, Hao Miao, Guojiang Shen, Yan Zhao, Xiangjie Kong, Ivan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.01705)  

**Abstract**: Out-of-town trip recommendation aims to generate a sequence of Points of Interest (POIs) for users traveling from their hometowns to previously unvisited regions based on personalized itineraries, e.g., origin, destination, and trip duration. Modeling the complex user preferences--which often exhibit a two-fold nature of static and dynamic interests--is critical for effective recommendations. However, the sparsity of out-of-town check-in data presents significant challenges in capturing such user preferences. Meanwhile, existing methods often conflate the static and dynamic preferences, resulting in suboptimal performance. In this paper, we for the first time systematically study the problem of out-of-town trip recommendation. A novel framework SPOT-Trip is proposed to explicitly learns the dual static-dynamic user preferences. Specifically, to handle scarce data, we construct a POI attribute knowledge graph to enrich the semantic modeling of users' hometown and out-of-town check-ins, enabling the static preference modeling through attribute relation-aware aggregation. Then, we employ neural ordinary differential equations (ODEs) to capture the continuous evolution of latent dynamic user preferences and innovatively combine a temporal point process to describe the instantaneous probability of each preference behavior. Further, a static-dynamic fusion module is proposed to merge the learned static and dynamic user preferences. Extensive experiments on real data offer insight into the effectiveness of the proposed solutions, showing that SPOT-Trip achieves performance improvement by up to 17.01%. 

---
# GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion 

**Authors**: Sunkyung Lee, Minjin Choi, Eunseong Choi, Hye-young Kim, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.01673)  

**Abstract**: Generative recommendation is an emerging paradigm that leverages the extensive knowledge of large language models by formulating recommendations into a text-to-text generation task. However, existing studies face two key limitations in (i) incorporating implicit item relationships and (ii) utilizing rich yet lengthy item information. To address these challenges, we propose a Generative Recommender via semantic-Aware Multi-granular late fusion (GRAM), introducing two synergistic innovations. First, we design semantic-to-lexical translation to encode implicit hierarchical and collaborative item relationships into the vocabulary space of LLMs. Second, we present multi-granular late fusion to integrate rich semantics efficiently with minimal information loss. It employs separate encoders for multi-granular prompts, delaying the fusion until the decoding stage. Experiments on four benchmark datasets show that GRAM outperforms eight state-of-the-art generative recommendation models, achieving significant improvements of 11.5-16.0% in Recall@5 and 5.3-13.6% in NDCG@5. The source code is available at this https URL. 

---
# Generative Next POI Recommendation with Semantic ID 

**Authors**: Dongsheng Wang, Yuxi Huang, Shen Gao, Yifan Wang, Chengrui Huang, Shuo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01375)  

**Abstract**: Point-of-interest (POI) recommendation systems aim to predict the next destinations of user based on their preferences and historical check-ins. Existing generative POI recommendation methods usually employ random numeric IDs for POIs, limiting the ability to model semantic relationships between similar locations. In this paper, we propose Generative Next POI Recommendation with Semantic ID (GNPR-SID), an LLM-based POI recommendation model with a novel semantic POI ID (SID) representation method that enhances the semantic understanding of POI modeling. There are two key components in our GNPR-SID: (1) a Semantic ID Construction module that generates semantically rich POI IDs based on semantic and collaborative features, and (2) a Generative POI Recommendation module that fine-tunes LLMs to predict the next POI using these semantic IDs. By incorporating user interaction patterns and POI semantic features into the semantic ID generation, our method improves the recommendation accuracy and generalization of the model. To construct semantically related SIDs, we propose a POI quantization method based on residual quantized variational autoencoder, which maps POIs into a discrete semantic space. We also propose a diversity loss to ensure that SIDs are uniformly distributed across the semantic space. Extensive experiments on three benchmark datasets demonstrate that GNPR-SID substantially outperforms state-of-the-art methods, achieving up to 16% improvement in recommendation accuracy. 

---
# AI4Contracts: LLM & RAG-Powered Encoding of Financial Derivative Contracts 

**Authors**: Maruf Ahmed Mridul, Ian Sloyan, Aparna Gupta, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2506.01063)  

**Abstract**: Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) are reshaping how AI systems extract and organize information from unstructured text. A key challenge is designing AI methods that can incrementally extract, structure, and validate information while preserving hierarchical and contextual relationships. We introduce CDMizer, a template-driven, LLM, and RAG-based framework for structured text transformation. By leveraging depth-based retrieval and hierarchical generation, CDMizer ensures a controlled, modular process that aligns generated outputs with predefined schema. Its template-driven approach guarantees syntactic correctness, schema adherence, and improved scalability, addressing key limitations of direct generation methods. Additionally, we propose an LLM-powered evaluation framework to assess the completeness and accuracy of structured representations. Demonstrated in the transformation of Over-the-Counter (OTC) financial derivative contracts into the Common Domain Model (CDM), CDMizer establishes a scalable foundation for AI-driven document understanding, structured synthesis, and automated validation in broader contexts. 

---
# Bridging the Gap: From Ad-hoc to Proactive Search in Conversations 

**Authors**: Chuan Meng, Francesco Tonolini, Fengran Mo, Nikolaos Aletras, Emine Yilmaz, Gabriella Kazai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00983)  

**Abstract**: Proactive search in conversations (PSC) aims to reduce user effort in formulating explicit queries by proactively retrieving useful relevant information given conversational context. Previous work in PSC either directly uses this context as input to off-the-shelf ad-hoc retrievers or further fine-tunes them on PSC data. However, ad-hoc retrievers are pre-trained on short and concise queries, while the PSC input is longer and noisier. This input mismatch between ad-hoc search and PSC limits retrieval quality. While fine-tuning on PSC data helps, its benefits remain constrained by this input gap. In this work, we propose Conv2Query, a novel conversation-to-query framework that adapts ad-hoc retrievers to PSC by bridging the input gap between ad-hoc search and PSC. Conv2Query maps conversational context into ad-hoc queries, which can either be used as input for off-the-shelf ad-hoc retrievers or for further fine-tuning on PSC data. Extensive experiments on two PSC datasets show that Conv2Query significantly improves ad-hoc retrievers' performance, both when used directly and after fine-tuning on PSC. 

---
# AliBoost: Ecological Boosting Framework in Alibaba Platform 

**Authors**: Qijie Shen, Yuanchen Bei, Zihong Huang, Jialin Zhu, Keqin Xu, Boya Du, Jiawei Tang, Yuning Jiang, Feiran Huang, Xiao Huang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00954)  

**Abstract**: Maintaining a healthy ecosystem in billion-scale online platforms is challenging, as users naturally gravitate toward popular items, leaving cold and less-explored items behind. This ''rich-get-richer'' phenomenon hinders the growth of potentially valuable cold items and harms the platform's ecosystem. Existing cold-start models primarily focus on improving initial recommendation performance for cold items but fail to address users' natural preference for popular content. In this paper, we introduce AliBoost, Alibaba's ecological boosting framework, designed to complement user-oriented natural recommendations and foster a healthier ecosystem. AliBoost incorporates a tiered boosting structure and boosting principles to ensure high-potential items quickly gain exposure while minimizing disruption to low-potential items. To achieve this, we propose the Stacking Fine-Tuning Cold Predictor to enhance the foundation CTR model's performance on cold items for accurate CTR and potential prediction. AliBoost then employs an Item-oriented Bidding Boosting mechanism to deliver cold items to the most suitable users while balancing boosting speed with user-personalized preferences. Over the past six months, AliBoost has been deployed across Alibaba's mainstream platforms, successfully cold-starting over a billion new items and increasing both clicks and GMV of cold items by over 60% within 180 days. Extensive online analysis and A/B testing demonstrate the effectiveness of AliBoost in addressing ecological challenges, offering new insights into the design of billion-scale recommender systems. 

---
# Breaker: Removing Shortcut Cues with User Clustering for Single-slot Recommendation System 

**Authors**: Chao Wang, Yue Zheng, Yujing Zhang, Yan Feng, Zhe Wang, Xiaowei Shi, An You, Yu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00828)  

**Abstract**: In a single-slot recommendation system, users are only exposed to one item at a time, and the system cannot collect user feedback on multiple items simultaneously. Therefore, only pointwise modeling solutions can be adopted, focusing solely on modeling the likelihood of clicks or conversions for items by users to learn user-item preferences, without the ability to capture the ranking information among different items directly. However, since user-side information is often much more abundant than item-side information, the model can quickly learn the differences in user intrinsic tendencies, which are independent of the items they are exposed to. This can cause these intrinsic tendencies to become a shortcut bias for the model, leading to insufficient mining of the most concerned user-item preferences. To solve this challenge, we introduce the Breaker model. Breaker integrates an auxiliary task of user representation clustering with a multi-tower structure for cluster-specific preference modeling. By clustering user representations, we ensure that users within each cluster exhibit similar characteristics, which increases the complexity of the pointwise recommendation task on the user side. This forces the multi-tower structure with cluster-driven parameter learning to better model user-item preferences, ultimately eliminating shortcut biases related to user intrinsic tendencies. In terms of training, we propose a delayed parameter update mechanism to enhance training stability and convergence, enabling end-to-end joint training of the auxiliary clustering and classification tasks. Both offline and online experiments demonstrate that our method surpasses the baselines. It has already been deployed and is actively serving tens of millions of users daily on Meituan, one of the most popular e-commerce platforms for services. 

---
# Optimizing Question Semantic Space for Dynamic Retrieval-Augmented Multi-hop Question Answering 

**Authors**: Linhao Ye, Lang Yu, Zhikai Lei, Qin Chen, Jie Zhou, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2506.00491)  

**Abstract**: Retrieval-augmented generation (RAG) is usually integrated into large language models (LLMs) to mitigate hallucinations and knowledge obsolescence. Whereas,conventional one-step retrieve-and-read methods are insufficient for multi-hop question answering, facing challenges of retrieval semantic mismatching and the high cost in handling interdependent subquestions. In this paper, we propose Optimizing Question Semantic Space for Dynamic Retrieval-Augmented Multi-hop Question Answering (Q-DREAM). Q-DREAM consists of three key modules: (1) the Question Decomposition Module (QDM), which decomposes multi-hop questions into fine-grained subquestions; (2) the Subquestion Dependency Optimizer Module (SDOM), which models the interdependent relations of subquestions for better understanding; and (3) the Dynamic Passage Retrieval Module (DPRM), which aligns subquestions with relevant passages by optimizing the semantic embeddings. Experimental results across various benchmarks demonstrate that Q-DREAM significantly outperforms existing RAG methods, achieving state-of-the-art performance in both in-domain and out-of-domain settings. Notably, Q-DREAM also improves retrieval efficiency while maintaining high accuracy compared with recent baselines. 

---
# DV365: Extremely Long User History Modeling at Instagram 

**Authors**: Wenhan Lyu, Devashish Tyagi, Yihang Yang, Ziwei Li, Ajay Somani, Karthikeyan Shanmugasundaram, Nikola Andrejevic, Ferdi Adeputra, Curtis Zeng, Arun K. Singh, Maxime Ransan, Sagar Jain  

**Link**: [PDF](https://arxiv.org/pdf/2506.00450)  

**Abstract**: Long user history is highly valuable signal for recommendation systems, but effectively incorporating it often comes with high cost in terms of data center power consumption and GPU. In this work, we chose offline embedding over end-to-end sequence length optimization methods to enable extremely long user sequence modeling as a cost-effective solution, and propose a new user embedding learning strategy, multi-slicing and summarization, that generates highly generalizable user representation of user's long-term stable interest. History length we encoded in this embedding is up to 70,000 and on average 40,000. This embedding, named as DV365, is proven highly incremental on top of advanced attentive user sequence models deployed in Instagram. Produced by a single upstream foundational model, it is launched in 15 different models across Instagram and Threads with significant impact, and has been production battle-proven for >1 year since our first launch. 

---
# K-order Ranking Preference Optimization for Large Language Models 

**Authors**: Shihao Cai, Chongming Gao, Yang Zhang, Wentao Shi, Jizhi Zhang, Keqin Bao, Qifan Wang, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00441)  

**Abstract**: To adapt large language models (LLMs) to ranking tasks, existing list-wise methods, represented by list-wise Direct Preference Optimization (DPO), focus on optimizing partial-order or full-order list ranking consistency for LLMs to enhance their ranking abilities. However, we argue that optimizing top-K ranking consistency could be more appropriate for real-world applications. There are two main reasons: (1) users are typically concerned with only the top-K results, making top-K ranking more important, and (2) tail items often lack precise feedback, making top-K ranking more reliable. Based on this, we propose K-order Ranking Preference Optimization (KPO) by extending the DPO's Plackett-Luce model to accommodate top-K rankings. Additionally, recognizing that the number of important items can vary across queries, we extend KPO to dynamically determine appropriate K for different samples and introduce a curriculum learning strategy to boost training efficiency. Extensive experiments demonstrate the effectiveness of KPO, highlighting its high sample efficiency and robustness to noise. The code is available at this https URL. 

---
# Adapting General-Purpose Embedding Models to Private Datasets Using Keyword-based Retrieval 

**Authors**: Yubai Wei, Jiale Han, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00363)  

**Abstract**: Text embedding models play a cornerstone role in AI applications, such as retrieval-augmented generation (RAG). While general-purpose text embedding models demonstrate strong performance on generic retrieval benchmarks, their effectiveness diminishes when applied to private datasets (e.g., company-specific proprietary data), which often contain specialized terminology and lingo. In this work, we introduce BMEmbed, a novel method for adapting general-purpose text embedding models to private datasets. By leveraging the well-established keyword-based retrieval technique (BM25), we construct supervisory signals from the ranking of keyword-based retrieval results to facilitate model adaptation. We evaluate BMEmbed across a range of domains, datasets, and models, showing consistent improvements in retrieval performance. Moreover, we provide empirical insights into how BM25-based signals contribute to improving embeddings by fostering alignment and uniformity, highlighting the value of this approach in adapting models to domain-specific data. We release the source code available at this https URL for the research community. 

---
# FACE: A Fine-grained Reference Free Evaluator for Conversational Recommender Systems 

**Authors**: Hideaki Joko, Faegheh Hasibi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00314)  

**Abstract**: A systematic, reliable, and low-cost evaluation of Conversational Recommender Systems (CRSs) remains an open challenge. Existing automatic CRS evaluation methods are proven insufficient for evaluating the dynamic nature of recommendation conversations. This work proposes FACE: a Fine-grained, Aspect-based Conversation Evaluation method that provides evaluation scores for diverse turn and dialogue level qualities of recommendation conversations. FACE is reference-free and shows strong correlation with human judgments, achieving system correlation of 0.9 and turn/dialogue-level of 0.5, outperforming state-of-the-art CRS evaluation methods by a large margin. Additionally, unlike existing LLM-based methods that provide single uninterpretable scores, FACE provides insights into the system performance and enables identifying and locating problems within conversations. 

---
# GPR: Empowering Generation with Graph-Pretrained Retriever 

**Authors**: Xiaochen Wang, Zongyu Wu, Yuan Zhong, Xiang Zhang, Suhang Wang, Fenglong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.00261)  

**Abstract**: Graph retrieval-augmented generation (GRAG) places high demands on graph-specific retrievers. However, existing retrievers often rely on language models pretrained on plain text, limiting their effectiveness due to domain misalignment and structure ignorance. To address these challenges, we propose GPR, a graph-based retriever pretrained directly on knowledge graphs. GPR aligns natural language questions with relevant subgraphs through LLM-guided graph augmentation and employs a structure-aware objective to learn fine-grained retrieval strategies. Experiments on two datasets, three LLM backbones, and five baselines show that GPR consistently improves both retrieval quality and downstream generation, demonstrating its effectiveness as a robust retrieval solution for GRAG. 

---
# Curate, Connect, Inquire: A System for Findable Accessible Interoperable and Reusable (FAIR) Human-Robot Centered Datasets 

**Authors**: Xingru Zhou, Sadanand Modak, Yao-Cheng Chan, Zhiyun Deng, Luis Sentis, Maria Esteva  

**Link**: [PDF](https://arxiv.org/pdf/2506.00220)  

**Abstract**: The rapid growth of AI in robotics has amplified the need for high-quality, reusable datasets, particularly in human-robot interaction (HRI) and AI-embedded robotics. While more robotics datasets are being created, the landscape of open data in the field is uneven. This is due to a lack of curation standards and consistent publication practices, which makes it difficult to discover, access, and reuse robotics data. To address these challenges, this paper presents a curation and access system with two main contributions: (1) a structured methodology to curate, publish, and integrate FAIR (Findable, Accessible, Interoperable, Reusable) human-centered robotics datasets; and (2) a ChatGPT-powered conversational interface trained with the curated datasets metadata and documentation to enable exploration, comparison robotics datasets and data retrieval using natural language. Developed based on practical experience curating datasets from robotics labs within Texas Robotics at the University of Texas at Austin, the system demonstrates the value of standardized curation and persistent publication of robotics data. The system's evaluation suggests that access and understandability of human-robotics data are significantly improved. This work directly aligns with the goals of the HCRL @ ICRA 2025 workshop and represents a step towards more human-centered access to data for embodied AI. 

---
# Gated Multimodal Graph Learning for Personalized Recommendation 

**Authors**: Sibei Liu, Yuanzhe Zhang, Xiang Li, Yunbo Liu, Chengwei Feng, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00107)  

**Abstract**: Multimodal recommendation has emerged as a promising solution to alleviate the cold-start and sparsity problems in collaborative filtering by incorporating rich content information, such as product images and textual descriptions. However, effectively integrating heterogeneous modalities into a unified recommendation framework remains a challenge. Existing approaches often rely on fixed fusion strategies or complex architectures , which may fail to adapt to modality quality variance or introduce unnecessary computational overhead.
In this work, we propose RLMultimodalRec, a lightweight and modular recommendation framework that combines graph-based user modeling with adaptive multimodal item encoding. The model employs a gated fusion module to dynamically balance the contribution of visual and textual modalities, enabling fine-grained and content-aware item representations. Meanwhile, a two-layer LightGCN encoder captures high-order collaborative signals by propagating embeddings over the user-item interaction graph without relying on nonlinear transformations.
We evaluate our model on a real-world dataset from the Amazon product domain. Experimental results demonstrate that RLMultimodalRec consistently outperforms several competitive baselines, including collaborative filtering, visual-aware, and multimodal GNN-based methods. The proposed approach achieves significant improvements in top-K recommendation metrics while maintaining scalability and interpretability, making it suitable for practical deployment. 

---
# Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers 

**Authors**: Chaitanya Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2506.00054)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm to enhance large language models (LLMs) by conditioning generation on external evidence retrieved at inference time. While RAG addresses critical limitations of parametric knowledge storage-such as factual inconsistency and domain inflexibility-it introduces new challenges in retrieval quality, grounding fidelity, pipeline efficiency, and robustness against noisy or adversarial inputs. This survey provides a comprehensive synthesis of recent advances in RAG systems, offering a taxonomy that categorizes architectures into retriever-centric, generator-centric, hybrid, and robustness-oriented designs. We systematically analyze enhancements across retrieval optimization, context filtering, decoding control, and efficiency improvements, supported by comparative performance analyses on short-form and multi-hop question answering tasks. Furthermore, we review state-of-the-art evaluation frameworks and benchmarks, highlighting trends in retrieval-aware evaluation, robustness testing, and federated retrieval settings. Our analysis reveals recurring trade-offs between retrieval precision and generation flexibility, efficiency and faithfulness, and modularity and coordination. We conclude by identifying open challenges and future research directions, including adaptive retrieval architectures, real-time retrieval integration, structured reasoning over multi-hop evidence, and privacy-preserving retrieval mechanisms. This survey aims to consolidate current knowledge in RAG research and serve as a foundation for the next generation of retrieval-augmented language modeling systems. 

---
# Rethinking Hybrid Retrieval: When Small Embeddings and LLM Re-ranking Beat Bigger Models 

**Authors**: Arjun Rao, Hanieh Alipour, Nick Pendar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00049)  

**Abstract**: This paper presents a comparison of embedding models in tri-modal hybrid retrieval for Retrieval-Augmented Generation (RAG) systems. We investigate the fusion of dense semantic, sparse lexical, and graph-based embeddings, focusing on the performance of the MiniLM-v6 and BGE-Large architectures. Contrary to conventional assumptions, our results show that the compact MiniLM-v6 outperforms the larger BGE-Large when integrated with LLM-based re-ranking within our tri-modal hybrid framework. Experiments conducted on the SciFact, FIQA, and NFCorpus datasets demonstrate significant improvements in retrieval quality with the MiniLM-v6 configuration. The performance difference is particularly pronounced in agentic re-ranking scenarios, indicating better alignment between MiniLM-v6's embedding space and LLM reasoning. Our findings suggest that embedding model selection for RAG systems should prioritize compatibility with multi-signal fusion and LLM alignment, rather than relying solely on larger models. This approach may reduce computational requirements while improving retrieval accuracy and efficiency. 

---
# Graph Contrastive Learning for Optimizing Sparse Data in Recommender Systems with LightGCL 

**Authors**: Aravinda Jatavallabha, Prabhanjan Bharadwaj, Ashish Chander  

**Link**: [PDF](https://arxiv.org/pdf/2506.00048)  

**Abstract**: Graph Neural Networks (GNNs) are powerful tools for recommendation systems, but they often struggle under data sparsity and noise. To address these issues, we implemented LightGCL, a graph contrastive learning model that uses Singular Value Decomposition (SVD) for robust graph augmentation, preserving semantic integrity without relying on stochastic or heuristic perturbations. LightGCL enables structural refinement and captures global collaborative signals, achieving significant gains over state-of-the-art models across benchmark datasets. Our experiments also demonstrate improved fairness and resilience to popularity bias, making it well-suited for real-world recommender systems. 

---
# Decoding Dense Embeddings: Sparse Autoencoders for Interpreting and Discretizing Dense Retrieval 

**Authors**: Seongwan Park, Taeklim Kim, Youngjoong Ko  

**Link**: [PDF](https://arxiv.org/pdf/2506.00041)  

**Abstract**: Despite their strong performance, Dense Passage Retrieval (DPR) models suffer from a lack of interpretability. In this work, we propose a novel interpretability framework that leverages Sparse Autoencoders (SAEs) to decompose previously uninterpretable dense embeddings from DPR models into distinct, interpretable latent concepts. We generate natural language descriptions for each latent concept, enabling human interpretations of both the dense embeddings and the query-document similarity scores of DPR models. We further introduce Concept-Level Sparse Retrieval (CL-SR), a retrieval framework that directly utilizes the extracted latent concepts as indexing units. CL-SR effectively combines the semantic expressiveness of dense embeddings with the transparency and efficiency of sparse representations. We show that CL-SR achieves high index-space and computational efficiency while maintaining robust performance across vocabulary and semantic mismatches. 

---
# Query Drift Compensation: Enabling Compatibility in Continual Learning of Retrieval Embedding Models 

**Authors**: Dipam Goswami, Liying Wang, Bartłomiej Twardowski, Joost van de Weijer  

**Link**: [PDF](https://arxiv.org/pdf/2506.00037)  

**Abstract**: Text embedding models enable semantic search, powering several NLP applications like Retrieval Augmented Generation by efficient information retrieval (IR). However, text embedding models are commonly studied in scenarios where the training data is static, thus limiting its applications to dynamic scenarios where new training data emerges over time. IR methods generally encode a huge corpus of documents to low-dimensional embeddings and store them in a database index. During retrieval, a semantic search over the corpus is performed and the document whose embedding is most similar to the query embedding is returned. When updating an embedding model with new training data, using the already indexed corpus is suboptimal due to the non-compatibility issue, since the model which was used to obtain the embeddings of the corpus has changed. While re-indexing of old corpus documents using the updated model enables compatibility, it requires much higher computation and time. Thus, it is critical to study how the already indexed corpus can still be effectively used without the need of re-indexing. In this work, we establish a continual learning benchmark with large-scale datasets and continually train dense retrieval embedding models on query-document pairs from new datasets in each task and observe forgetting on old tasks due to significant drift of embeddings. We employ embedding distillation on both query and document embeddings to maintain stability and propose a novel query drift compensation method during retrieval to project new model query embeddings to the old embedding space. This enables compatibility with previously indexed corpus embeddings extracted using the old model and thus reduces the forgetting. We show that the proposed method significantly improves performance without any re-indexing. Code is available at this https URL. 

---
# Getting almost all the bits from a quantum random access code 

**Authors**: Han-Hsuan Lin, Ronald de Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2506.01903)  

**Abstract**: A quantum random access code (QRAC) is a map $x\mapsto\rho_x$ that encodes $n$-bit strings $x$ into $m$-qubit quantum states $\rho_x$, in a way that allows us to recover any one bit of $x$ with success probability $\geq p$. The measurement on $\rho_x$ that is used to recover, say, $x_1$ may destroy all the information about the other bits; this is in fact what happens in the well-known QRAC that encodes $n=2$ bits into $m=1$ qubits. Does this generalize to large $n$, i.e., could there exist QRACs that are so "obfuscated" that one cannot get much more than one bit out of them? Here we show that this is not the case: for every QRAC there exists a measurement that (with high probability) recovers the full $n$-bit string $x$ up to small Hamming distance, even for the worst-case $x$. 

---
# CiteEval: Principle-Driven Citation Evaluation for Source Attribution 

**Authors**: Yumo Xu, Peng Qi, Jifan Chen, Kunlun Liu, Rujun Han, Lan Liu, Bonan Min, Vittorio Castelli, Arshit Gupta, Zhiguo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01829)  

**Abstract**: Citation quality is crucial in information-seeking systems, directly influencing trust and the effectiveness of information access. Current evaluation frameworks, both human and automatic, mainly rely on Natural Language Inference (NLI) to assess binary or ternary supportiveness from cited sources, which we argue is a suboptimal proxy for citation evaluation. In this work we introduce CiteEval, a citation evaluation framework driven by principles focusing on fine-grained citation assessment within a broad context, encompassing not only the cited sources but the full retrieval context, user query, and generated text. Guided by the proposed framework, we construct CiteBench, a multi-domain benchmark with high-quality human annotations on citation quality. To enable efficient evaluation, we further develop CiteEval-Auto, a suite of model-based metrics that exhibit strong correlation with human judgments. Experiments across diverse systems demonstrate CiteEval-Auto's superior ability to capture the multifaceted nature of citations compared to existing metrics, offering a principled and scalable approach to evaluate and improve model-generated citations. 

---
# Small Stickers, Big Meanings: A Multilingual Sticker Semantic Understanding Dataset with a Gamified Approach 

**Authors**: Heng Er Metilda Chee, Jiayin Wang, Zhiqiang Guo, Weizhi Ma, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01668)  

**Abstract**: Stickers, though small, are a highly condensed form of visual expression, ubiquitous across messaging platforms and embraced by diverse cultures, genders, and age groups. Despite their popularity, sticker retrieval remains an underexplored task due to the significant human effort and subjectivity involved in constructing high-quality sticker query datasets. Although large language models (LLMs) excel at general NLP tasks, they falter when confronted with the nuanced, intangible, and highly specific nature of sticker query generation.
To address this challenge, we propose a threefold solution. First, we introduce Sticktionary, a gamified annotation framework designed to gather diverse, high-quality, and contextually resonant sticker queries. Second, we present StickerQueries, a multilingual sticker query dataset containing 1,115 English and 615 Chinese queries, annotated by over 60 contributors across 60+ hours. Lastly, Through extensive quantitative and qualitative evaluation, we demonstrate that our approach significantly enhances query generation quality, retrieval accuracy, and semantic understanding in the sticker domain. To support future research, we publicly release our multilingual dataset along with two fine-tuned query generation models. 

---
# Engram Memory Encoding and Retrieval: A Neurocomputational Perspective 

**Authors**: Daniel Szelogowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.01659)  

**Abstract**: Despite substantial research into the biological basis of memory, the precise mechanisms by which experiences are encoded, stored, and retrieved in the brain remain incompletely understood. A growing body of evidence supports the engram theory, which posits that sparse populations of neurons undergo lasting physical and biochemical changes to support long-term memory. Yet, a comprehensive computational framework that integrates biological findings with mechanistic models remains elusive. This work synthesizes insights from cellular neuroscience and computational modeling to address key challenges in engram research: how engram neurons are identified and manipulated; how synaptic plasticity mechanisms contribute to stable memory traces; and how sparsity promotes efficient, interference-resistant representations. Relevant computational approaches -- such as sparse regularization, engram gating, and biologically inspired architectures like Sparse Distributed Memory and spiking neural networks -- are also examined. Together, these findings suggest that memory efficiency, capacity, and stability emerge from the interaction of plasticity and sparsity constraints. By integrating neurobiological and computational perspectives, this paper provides a comprehensive theoretical foundation for engram research and proposes a roadmap for future inquiry into the mechanisms underlying memory, with implications for the diagnosis and treatment of memory-related disorders. 

---
# Argument-Centric Causal Intervention Method for Mitigating Bias in Cross-Document Event Coreference Resolution 

**Authors**: Long Yao, Wenzhong Yang, Yabo Yin, Fuyuan Wei, Hongzhen Lv, Jiaren Peng, Liejun Wang, Xiaoming Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01488)  

**Abstract**: Cross-document Event Coreference Resolution (CD-ECR) is a fundamental task in natural language processing (NLP) that seeks to determine whether event mentions across multiple documents refer to the same real-world occurrence. However, current CD-ECR approaches predominantly rely on trigger features within input mention pairs, which induce spurious correlations between surface-level lexical features and coreference relationships, impairing the overall performance of the models. To address this issue, we propose a novel cross-document event coreference resolution method based on Argument-Centric Causal Intervention (ACCI). Specifically, we construct a structural causal graph to uncover confounding dependencies between lexical triggers and coreference labels, and introduce backdoor-adjusted interventions to isolate the true causal effect of argument semantics. To further mitigate spurious correlations, ACCI integrates a counterfactual reasoning module that quantifies the causal influence of trigger word perturbations, and an argument-aware enhancement module to promote greater sensitivity to semantically grounded information. In contrast to prior methods that depend on costly data augmentation or heuristic-based filtering, ACCI enables effective debiasing in a unified end-to-end framework without altering the underlying training procedure. Extensive experiments demonstrate that ACCI achieves CoNLL F1 of 88.4% on ECB+ and 85.2% on GVC, achieving state-of-the-art performance. The implementation and materials are available at this https URL. 

---
# Building Entity Association Mining Framework for Knowledge Discovery 

**Authors**: Anshika Rawal, Abhijeet Kumar, Mridul Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2506.01451)  

**Abstract**: Extracting useful signals or pattern to support important business decisions for example analyzing investment product traction and discovering customer preference, risk monitoring etc. from unstructured text is a challenging task. Capturing interaction of entities or concepts and association mining is a crucial component in text mining, enabling information extraction and reasoning over and knowledge discovery from text. Furthermore, it can be used to enrich or filter knowledge graphs to guide exploration processes, descriptive analytics and uncover hidden stories in the text. In this paper, we introduce a domain independent pipeline i.e., generalized framework to enable document filtering, entity extraction using various sources (or techniques) as plug-ins and association mining to build any text mining business use-case and quantitatively define a scoring metric for ranking purpose. The proposed framework has three major components a) Document filtering: filtering documents/text of interest from massive amount of texts b) Configurable entity extraction pipeline: include entity extraction techniques i.e., i) DBpedia Spotlight, ii) Spacy NER, iii) Custom Entity Matcher, iv) Phrase extraction (or dictionary) based c) Association Relationship Mining: To generates co-occurrence graph to analyse potential relationships among entities, concepts. Further, co-occurrence count based frequency statistics provide a holistic window to observe association trends or buzz rate in specific business context. The paper demonstrates the usage of framework as fundamental building box in two financial use-cases namely brand product discovery and vendor risk monitoring. We aim that such framework will remove duplicated effort, minimize the development effort, and encourage reusability and rapid prototyping in association mining business applications for institutions. 

---
# TimeGraph: Synthetic Benchmark Datasets for Robust Time-Series Causal Discovery 

**Authors**: Muhammad Hasan Ferdous, Emam Hossain, Md Osman Gani  

**Link**: [PDF](https://arxiv.org/pdf/2506.01361)  

**Abstract**: Robust causal discovery in time series datasets depends on reliable benchmark datasets with known ground-truth causal relationships. However, such datasets remain scarce, and existing synthetic alternatives often overlook critical temporal properties inherent in real-world data, including nonstationarity driven by trends and seasonality, irregular sampling intervals, and the presence of unobserved confounders. To address these challenges, we introduce TimeGraph, a comprehensive suite of synthetic time-series benchmark datasets that systematically incorporates both linear and nonlinear dependencies while modeling key temporal characteristics such as trends, seasonal effects, and heterogeneous noise patterns. Each dataset is accompanied by a fully specified causal graph featuring varying densities and diverse noise distributions and is provided in two versions: one including unobserved confounders and one without, thereby offering extensive coverage of real-world complexity while preserving methodological neutrality. We further demonstrate the utility of TimeGraph through systematic evaluations of state-of-the-art causal discovery algorithms including PCMCI+, LPCMCI, and FGES across a diverse array of configurations and metrics. Our experiments reveal significant variations in algorithmic performance under realistic temporal conditions, underscoring the need for robust synthetic benchmarks in the fair and transparent assessment of causal discovery methods. The complete TimeGraph suite, including dataset generation scripts, evaluation metrics, and recommended experimental protocols, is freely available to facilitate reproducible research and foster community-driven advancements in time-series causal discovery. 

---
# A Platform for Investigating Public Health Content with Efficient Concern Classification 

**Authors**: Christopher Li, Rickard Stureborg, Bhuwan Dhingra, Jun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01308)  

**Abstract**: A recent rise in online content expressing concerns with public health initiatives has contributed to already stalled uptake of preemptive measures globally. Future public health efforts must attempt to understand such content, what concerns it may raise among readers, and how to effectively respond to it. To this end, we present ConcernScope, a platform that uses a teacher-student framework for knowledge transfer between large language models and light-weight classifiers to quickly and effectively identify the health concerns raised in a text corpus. The platform allows uploading massive files directly, automatically scraping specific URLs, and direct text editing. ConcernScope is built on top of a taxonomy of public health concerns. Intended for public health officials, we demonstrate several applications of this platform: guided data exploration to find useful examples of common concerns found in online community datasets, identification of trends in concerns through an example time series analysis of 186,000 samples, and finding trends in topic frequency before and after significant events. 

---
# Pitfalls in Evaluating Language Model Forecasters 

**Authors**: Daniel Paleka, Shashwat Goel, Jonas Geiping, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2506.00723)  

**Abstract**: Large language models (LLMs) have recently been applied to forecasting tasks, with some works claiming these systems match or exceed human performance. In this paper, we argue that, as a community, we should be careful about such conclusions as evaluating LLM forecasters presents unique challenges. We identify two broad categories of issues: (1) difficulty in trusting evaluation results due to many forms of temporal leakage, and (2) difficulty in extrapolating from evaluation performance to real-world forecasting. Through systematic analysis and concrete examples from prior work, we demonstrate how evaluation flaws can raise concerns about current and future performance claims. We argue that more rigorous evaluation methodologies are needed to confidently assess the forecasting abilities of LLMs. 

---
# Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples 

**Authors**: Haesung Pyun, Yoonah Park, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00622)  

**Abstract**: In dialogue state tracking (DST), in-context learning comprises a retriever that selects labeled dialogues as in-context examples and a DST model that uses these examples to infer the dialogue state of the query dialogue. Existing methods for constructing training data for retrievers suffer from three key limitations: (1) the synergistic effect of examples is not considered, (2) the linguistic characteristics of the query are not sufficiently factored in, and (3) scoring is not directly optimized for DST performance. Consequently, the retriever can fail to retrieve examples that would substantially improve DST performance. To address these issues, we present CombiSearch, a method that scores effective in-context examples based on their combinatorial impact on DST performance. Our evaluation on MultiWOZ shows that retrievers trained with CombiSearch surpass state-of-the-art models, achieving a 20x gain in data efficiency and generalizing well to the SGD dataset. Moreover, CombiSearch attains a 12% absolute improvement in the upper bound DST performance over traditional approaches when no retrieval errors are assumed. This significantly increases the headroom for practical DST performance while demonstrating that existing methods rely on suboptimal data for retriever training. 

---
# ZeShot-VQA: Zero-Shot Visual Question Answering Framework with Answer Mapping for Natural Disaster Damage Assessment 

**Authors**: Ehsan Karimi, Maryam Rahnemoonfar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00238)  

**Abstract**: Natural disasters usually affect vast areas and devastate infrastructures. Performing a timely and efficient response is crucial to minimize the impact on affected communities, and data-driven approaches are the best choice. Visual question answering (VQA) models help management teams to achieve in-depth understanding of damages. However, recently published models do not possess the ability to answer open-ended questions and only select the best answer among a predefined list of answers. If we want to ask questions with new additional possible answers that do not exist in the predefined list, the model needs to be fin-tuned/retrained on a new collected and annotated dataset, which is a time-consuming procedure. In recent years, large-scale Vision-Language Models (VLMs) have earned significant attention. These models are trained on extensive datasets and demonstrate strong performance on both unimodal and multimodal vision/language downstream tasks, often without the need for fine-tuning. In this paper, we propose a VLM-based zero-shot VQA (ZeShot-VQA) method, and investigate the performance of on post-disaster FloodNet dataset. Since the proposed method takes advantage of zero-shot learning, it can be applied on new datasets without fine-tuning. In addition, ZeShot-VQA is able to process and generate answers that has been not seen during the training procedure, which demonstrates its flexibility. 

---
# The World As Large Language Models See It: Exploring the reliability of LLMs in representing geographical features 

**Authors**: Omid Reza Abbasi, Franz Welscher, Georg Weinberger, Johannes Scholz  

**Link**: [PDF](https://arxiv.org/pdf/2506.00203)  

**Abstract**: As large language models (LLMs) continue to evolve, questions about their trustworthiness in delivering factual information have become increasingly important. This concern also applies to their ability to accurately represent the geographic world. With recent advancements in this field, it is relevant to consider whether and to what extent LLMs' representations of the geographical world can be trusted. This study evaluates the performance of GPT-4o and Gemini 2.0 Flash in three key geospatial tasks: geocoding, elevation estimation, and reverse geocoding. In the geocoding task, both models exhibited systematic and random errors in estimating the coordinates of St. Anne's Column in Innsbruck, Austria, with GPT-4o showing greater deviations and Gemini 2.0 Flash demonstrating more precision but a significant systematic offset. For elevation estimation, both models tended to underestimate elevations across Austria, though they captured overall topographical trends, and Gemini 2.0 Flash performed better in eastern regions. The reverse geocoding task, which involved identifying Austrian federal states from coordinates, revealed that Gemini 2.0 Flash outperformed GPT-4o in overall accuracy and F1-scores, demonstrating better consistency across regions. Despite these findings, neither model achieved an accurate reconstruction of Austria's federal states, highlighting persistent misclassifications. The study concludes that while LLMs can approximate geographic information, their accuracy and reliability are inconsistent, underscoring the need for fine-tuning with geographical information to enhance their utility in GIScience and Geoinformatics. 

---
# LaMP-QA: A Benchmark for Personalized Long-form Question Answering 

**Authors**: Alireza Salemi, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2506.00137)  

**Abstract**: Personalization is essential for question answering systems that are user-centric. Despite its importance, personalization in answer generation has been relatively underexplored. This is mainly due to lack of resources for training and evaluating personalized question answering systems. We address this gap by introducing LaMP-QA -- a benchmark designed for evaluating personalized long-form answer generation. The benchmark covers questions from three major categories: (1) Arts & Entertainment, (2) Lifestyle & Personal Development, and (3) Society & Culture, encompassing over 45 subcategories in total. To assess the quality and potential impact of the LaMP-QA benchmark for personalized question answering, we conduct comprehensive human and automatic evaluations, to compare multiple evaluation strategies for evaluating generated personalized responses and measure their alignment with human preferences. Furthermore, we benchmark a number of non-personalized and personalized approaches based on open-source and proprietary large language models (LLMs). Our results show that incorporating the personalized context provided leads to performance improvements of up to 39%. The benchmark is publicly released to support future research in this area. 

---
# Whose Name Comes Up? Auditing LLM-Based Scholar Recommendations 

**Authors**: Daniele Barolo, Chiara Valentin, Fariba Karimi, Luis Galárraga, Gonzalo G. Méndez, Lisette Espín-Noboa  

**Link**: [PDF](https://arxiv.org/pdf/2506.00074)  

**Abstract**: This paper evaluates the performance of six open-weight LLMs (llama3-8b, llama3.1-8b, gemma2-9b, mixtral-8x7b, llama3-70b, llama3.1-70b) in recommending experts in physics across five tasks: top-k experts by field, influential scientists by discipline, epoch, seniority, and scholar counterparts. The evaluation examines consistency, factuality, and biases related to gender, ethnicity, academic popularity, and scholar similarity. Using ground-truth data from the American Physical Society and OpenAlex, we establish scholarly benchmarks by comparing model outputs to real-world academic records. Our analysis reveals inconsistencies and biases across all models. mixtral-8x7b produces the most stable outputs, while llama3.1-70b shows the highest variability. Many models exhibit duplication, and some, particularly gemma2-9b and llama3.1-8b, struggle with formatting errors. LLMs generally recommend real scientists, but accuracy drops in field-, epoch-, and seniority-specific queries, consistently favoring senior scholars. Representation biases persist, replicating gender imbalances (reflecting male predominance), under-representing Asian scientists, and over-representing White scholars. Despite some diversity in institutional and collaboration networks, models favor highly cited and productive scholars, reinforcing the rich-getricher effect while offering limited geographical representation. These findings highlight the need to improve LLMs for more reliable and equitable scholarly recommendations. 

---
