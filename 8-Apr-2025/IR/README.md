# Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG 

**Authors**: Hengran Zhang, Minghao Tang, Keping Bi, Jiafeng Guo, Shihao Liu, Daiting Shi, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05220)  

**Abstract**: Retrieval models typically rely on costly human-labeled query-document relevance annotations for training and evaluation. To reduce this cost and leverage the potential of Large Language Models (LLMs) in relevance judgments, we aim to explore whether LLM-generated annotations can effectively replace human annotations in training retrieval models. Retrieval usually emphasizes relevance, which indicates "topic-relatedness" of a document to a query, while in RAG, the value of a document (or utility) depends on how it contributes to answer generation. Recognizing this mismatch, some researchers use LLM performance on downstream tasks with documents as labels, but this approach requires manual answers for specific tasks, leading to high costs and limited generalization. In another line of work, prompting LLMs to select useful documents as RAG references eliminates the need for human annotation and is not task-specific. If we leverage LLMs' utility judgments to annotate retrieval data, we may retain cross-task generalization without human annotation in large-scale corpora. Therefore, we investigate utility-focused annotation via LLMs for large-scale retriever training data across both in-domain and out-of-domain settings on the retrieval and RAG tasks. To reduce the impact of low-quality positives labeled by LLMs, we design a novel loss function, i.e., Disj-InfoNCE. Our experiments reveal that: (1) Retrievers trained on utility-focused annotations significantly outperform those trained on human annotations in the out-of-domain setting on both tasks, demonstrating superior generalization capabilities. (2) LLM annotation does not replace human annotation in the in-domain setting. However, incorporating just 20% human-annotated data enables retrievers trained with utility-focused annotations to match the performance of models trained entirely with human annotations. 

---
# LLM-Alignment Live-Streaming Recommendation 

**Authors**: Yueyang Liu, Jiangxia Cao, Shen Wang, Shuang Wen, Xiang Chen, Xiangyu Wu, Shuang Yang, Zhaojie Liu, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.05217)  

**Abstract**: In recent years, integrated short-video and live-streaming platforms have gained massive global adoption, offering dynamic content creation and consumption. Unlike pre-recorded short videos, live-streaming enables real-time interaction between authors and users, fostering deeper engagement. However, this dynamic nature introduces a critical challenge for recommendation systems (RecSys): the same live-streaming vastly different experiences depending on when a user watching. To optimize recommendations, a RecSys must accurately interpret the real-time semantics of live content and align them with user preferences. 

---
# Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling 

**Authors**: Hengran Zhang, Keping Bi, Jiafeng Guo, Xiaojie Sun, Shihao Liu, Daiting Shi, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05216)  

**Abstract**: Dense retrieval is a crucial task in Information Retrieval (IR) and is the foundation for downstream tasks such as re-ranking. Recently, large language models (LLMs) have shown compelling semantic understanding capabilities and are appealing to researchers studying dense retrieval. LLMs, as decoder-style generative models, are competent at language generation while falling short on modeling global information due to the lack of attention to tokens afterward. Inspired by the classical word-based language modeling approach for IR, i.e., the query likelihood (QL) model, we seek to sufficiently utilize LLMs' generative ability by QL maximization. However, instead of ranking documents with QL estimation, we introduce an auxiliary task of QL maximization to yield a better backbone for contrastively learning a discriminative retriever. We name our model as LLM-QL. To condense global document semantics to a single vector during QL modeling, LLM-QL has two major components, Attention Stop (AS) and Input Corruption (IC). AS stops the attention of predictive tokens to previous tokens until the ending token of the document. IC masks a portion of tokens in the input documents during prediction. Experiments on MSMARCO show that LLM-QL can achieve significantly better performance than other LLM-based retrievers and using QL estimated by LLM-QL for ranking outperforms word-based QL by a large margin. 

---
# Lightweight and Direct Document Relevance Optimization for Generative Information Retrieval 

**Authors**: Kidist Amde Mekonnen, Yubao Tang, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2504.05181)  

**Abstract**: Generative information retrieval (GenIR) is a promising neural retrieval paradigm that formulates document retrieval as a document identifier (docid) generation task, allowing for end-to-end optimization toward a unified global retrieval objective. However, existing GenIR models suffer from token-level misalignment, where models trained to predict the next token often fail to capture document-level relevance effectively. While reinforcement learning-based methods, such as reinforcement learning from relevance feedback (RLRF), aim to address this misalignment through reward modeling, they introduce significant complexity, requiring the optimization of an auxiliary reward function followed by reinforcement fine-tuning, which is computationally expensive and often unstable. To address these challenges, we propose direct document relevance optimization (DDRO), which aligns token-level docid generation with document-level relevance estimation through direct optimization via pairwise ranking, eliminating the need for explicit reward modeling and reinforcement learning. Experimental results on benchmark datasets, including MS MARCO document and Natural Questions, show that DDRO outperforms reinforcement learning-based methods, achieving a 7.4% improvement in MRR@10 for MS MARCO and a 19.9% improvement for Natural Questions. These findings highlight DDRO's potential to enhance retrieval effectiveness with a simplified optimization approach. By framing alignment as a direct optimization problem, DDRO simplifies the ranking optimization pipeline of GenIR models while offering a viable alternative to reinforcement learning-based methods. 

---
# Query Smarter, Trust Better? Exploring Search Behaviours for Verifying News Accuracy 

**Authors**: David Elsweiler, Samy Ateia, Markus Bink, Gregor Donabauer, Marcos Fernández Pichel, Alexander Frummet, Udo Kruschwitz, David Losada, Bernd Ludwig, Selina Meyer, Noel Pascual Presa  

**Link**: [PDF](https://arxiv.org/pdf/2504.05146)  

**Abstract**: While it is often assumed that searching for information to evaluate misinformation will help identify false claims, recent work suggests that search behaviours can instead reinforce belief in misleading news, particularly when users generate queries using vocabulary from the source articles. Our research explores how different query generation strategies affect news verification and whether the way people search influences the accuracy of their information evaluation. A mixed-methods approach was used, consisting of three parts: (1) an analysis of existing data to understand how search behaviour influences trust in fake news, (2) a simulation of query generation strategies using a Large Language Model (LLM) to assess the impact of different query formulations on search result quality, and (3) a user study to examine how 'Boost' interventions in interface design can guide users to adopt more effective query strategies. The results show that search behaviour significantly affects trust in news, with successful searches involving multiple queries and yielding higher-quality results. Queries inspired by different parts of a news article produced search results of varying quality, and weak initial queries improved when reformulated using full SERP information. Although 'Boost' interventions had limited impact, the study suggests that interface design encouraging users to thoroughly review search results can enhance query formulation. This study highlights the importance of query strategies in evaluating news and proposes that interface design can play a key role in promoting more effective search practices, serving as one component of a broader set of interventions to combat misinformation. 

---
# Data Augmentation as Free Lunch: Exploring the Test-Time Augmentation for Sequential Recommendation 

**Authors**: Yizhou Dang, Yuting Liu, Enneng Yang, Minhan Huang, Guibing Guo, Jianzhe Zhao, Xingwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04843)  

**Abstract**: Data augmentation has become a promising method of mitigating data sparsity in sequential recommendation. Existing methods generate new yet effective data during model training to improve performance. However, deploying them requires retraining, architecture modification, or introducing additional learnable parameters. The above steps are time-consuming and costly for well-trained models, especially when the model scale becomes large. In this work, we explore the test-time augmentation (TTA) for sequential recommendation, which augments the inputs during the model inference and then aggregates the model's predictions for augmented data to improve final accuracy. It avoids significant time and cost overhead from loss calculation and backward propagation. We first experimentally disclose the potential of existing augmentation operators for TTA and find that the Mask and Substitute consistently achieve better performance. Further analysis reveals that these two operators are effective because they retain the original sequential pattern while adding appropriate perturbations. Meanwhile, we argue that these two operators still face time-consuming item selection or interference information from mask tokens. Based on the analysis and limitations, we present TNoise and TMask. The former injects uniform noise into the original representation, avoiding the computational overhead of item selection. The latter blocks mask token from participating in model calculations or directly removes interactions that should have been replaced with mask tokens. Comprehensive experiments demonstrate the effectiveness, efficiency, and generalizability of our method. We provide an anonymous implementation at this https URL. 

---
# Investigating Popularity Bias Amplification in Recommender Systems Employed in the Entertainment Domain 

**Authors**: Dominik Kowald  

**Link**: [PDF](https://arxiv.org/pdf/2504.04752)  

**Abstract**: Recommender systems have become an integral part of our daily online experience by analyzing past user behavior to suggest relevant content in entertainment domains such as music, movies, and books. Today, they are among the most widely used applications of AI and machine learning. Consequently, regulations and guidelines for trustworthy AI, such as the European AI Act, which addresses issues like bias and fairness, are highly relevant to the design, development, and evaluation of recommender systems. One particularly important type of bias in this context is popularity bias, which results in the unfair underrepresentation of less popular content in recommendation lists. This work summarizes our research on investigating the amplification of popularity bias in recommender systems within the entertainment sector. Analyzing datasets from three entertainment domains, music, movies, and anime, we demonstrate that an item's recommendation frequency is positively correlated with its popularity. As a result, user groups with little interest in popular content receive less accurate recommendations compared to those who prefer widely popular items. Furthermore, we aim to better understand the connection between recommendation accuracy, calibration quality of algorithms, and popularity bias amplification. 

---
# Can LLM-Driven Hard Negative Sampling Empower Collaborative Filtering? Findings and Potentials 

**Authors**: Chu Zhao, Enneng Yang, Yuting Liu, Jianzhe Zhao, Guibing Guo, Xingwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04726)  

**Abstract**: Hard negative samples can accelerate model convergence and optimize decision boundaries, which is key to improving the performance of recommender systems. Although large language models (LLMs) possess strong semantic understanding and generation capabilities, systematic research has not yet been conducted on how to generate hard negative samples effectively. To fill this gap, this paper introduces the concept of Semantic Negative Sampling and exploreshow to optimize LLMs for high-quality, hard negative sampling. Specifically, we design an experimental pipeline that includes three main modules, profile generation, semantic negative sampling, and semantic alignment, to verify the potential of LLM-driven hard negative sampling in enhancing the accuracy of collaborative filtering (CF). Experimental results indicate that hard negative samples generated based on LLMs, when semantically aligned and integrated into CF, can significantly improve CF performance, although there is still a certain gap compared to traditional negative sampling methods. Further analysis reveals that this gap primarily arises from two major challenges: noisy samples and lack of behavioral constraints. To address these challenges, we propose a framework called HNLMRec, based on fine-tuning LLMs supervised by collaborative signals. Experimental results show that this framework outperforms traditional negative sampling and other LLM-driven recommendation methods across multiple datasets, providing new solutions for empowering traditional RS with LLMs. Additionally, we validate the excellent generalization ability of the LLM-based semantic negative sampling method on new datasets, demonstrating its potential in alleviating issues such as data sparsity, popularity bias, and the problem of false hard negative samples. Our implementation code is available at this https URL. 

---
# TC-MGC: Text-Conditioned Multi-Grained Contrastive Learning for Text-Video Retrieval 

**Authors**: Xiaolun Jing, Genke Yang, Jian Chu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04707)  

**Abstract**: Motivated by the success of coarse-grained or fine-grained contrast in text-video retrieval, there emerge multi-grained contrastive learning methods which focus on the integration of contrasts with different granularity. However, due to the wider semantic range of videos, the text-agnostic video representations might encode misleading information not described in texts, thus impeding the model from capturing precise cross-modal semantic correspondence. To this end, we propose a Text-Conditioned Multi-Grained Contrast framework, dubbed TC-MGC. Specifically, our model employs a language-video attention block to generate aggregated frame and video representations conditioned on the word's and text's attention weights over frames. To filter unnecessary similarity interactions and decrease trainable parameters in the Interactive Similarity Aggregation (ISA) module, we design a Similarity Reorganization (SR) module to identify attentive similarities and reorganize cross-modal similarity vectors and matrices. Next, we argue that the imbalance problem among multigrained similarities may result in over- and under-representation issues. We thereby introduce an auxiliary Similarity Decorrelation Regularization (SDR) loss to facilitate cooperative relationship utilization by similarity variance minimization on matching text-video pairs. Finally, we present a Linear Softmax Aggregation (LSA) module to explicitly encourage the interactions between multiple similarities and promote the usage of multi-grained information. Empirically, TC-MGC achieves competitive results on multiple text-video retrieval benchmarks, outperforming X-CLIP model by +2.8% (+1.3%), +2.2% (+1.0%), +1.5% (+0.9%) relative (absolute) improvements in text-to-video retrieval R@1 on MSR-VTT, DiDeMo and VATEX, respectively. Our code is publicly available at this https URL. 

---
# COHESION: Composite Graph Convolutional Network with Dual-Stage Fusion for Multimodal Recommendation 

**Authors**: Jinfeng Xu, Zheyu Chen, Wei Wang, Xiping Hu, Sang-Wook Kim, Edith C. H. Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2504.04452)  

**Abstract**: Recent works in multimodal recommendations, which leverage diverse modal information to address data sparsity and enhance recommendation accuracy, have garnered considerable interest. Two key processes in multimodal recommendations are modality fusion and representation learning. Previous approaches in modality fusion often employ simplistic attentive or pre-defined strategies at early or late stages, failing to effectively handle irrelevant information among modalities. In representation learning, prior research has constructed heterogeneous and homogeneous graph structures encapsulating user-item, user-user, and item-item relationships to better capture user interests and item profiles. Modality fusion and representation learning were considered as two independent processes in previous work. In this paper, we reveal that these two processes are complementary and can support each other. Specifically, powerful representation learning enhances modality fusion, while effective fusion improves representation quality. Stemming from these two processes, we introduce a COmposite grapH convolutional nEtwork with dual-stage fuSION for the multimodal recommendation, named COHESION. Specifically, it introduces a dual-stage fusion strategy to reduce the impact of irrelevant information, refining all modalities using ID embedding in the early stage and fusing their representations at the late stage. It also proposes a composite graph convolutional network that utilizes user-item, user-user, and item-item graphs to extract heterogeneous and homogeneous latent relationships within users and items. Besides, it introduces a novel adaptive optimization to ensure balanced and reasonable representations across modalities. Extensive experiments on three widely used datasets demonstrate the significant superiority of COHESION over various competitive baselines. 

---
# Squeeze and Excitation: A Weighted Graph Contrastive Learning for Collaborative Filtering 

**Authors**: Zheyu Chen, Jinfeng Xu, Yutong Wei, Ziyue Peng  

**Link**: [PDF](https://arxiv.org/pdf/2504.04443)  

**Abstract**: Contrastive Learning (CL) has recently emerged as a powerful technique in recommendation systems, particularly for its capability to harness self-supervised signals from perturbed views to mitigate the persistent challenge of data sparsity. The process of constructing perturbed views of the user-item bipartite graph and performing contrastive learning between perturbed views in a graph convolutional network (GCN) is called graph contrastive learning (GCL), which aims to enhance the robustness of representation learning. Although existing GCL-based models are effective, the weight assignment method for perturbed views has not been fully explored. A critical problem in existing GCL-based models is the irrational allocation of feature attention. This problem limits the model's ability to effectively leverage crucial features, resulting in suboptimal performance. To address this, we propose a Weighted Graph Contrastive Learning framework (WeightedGCL). Specifically, WeightedGCL applies a robust perturbation strategy, which perturbs only the view of the final GCN layer. In addition, WeightedGCL incorporates a squeeze and excitation network (SENet) to dynamically weight the features of the perturbed views. Our WeightedGCL strengthens the model's focus on crucial features and reduces the impact of less relevant information. Extensive experiments on widely used datasets demonstrate that our WeightedGCL achieves significant accuracy improvements compared to competitive baselines. 

---
# Universal Item Tokenization for Transferable Generative Recommendation 

**Authors**: Bowen Zheng, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.04405)  

**Abstract**: Recently, generative recommendation has emerged as a promising paradigm, attracting significant research attention. The basic framework involves an item tokenizer, which represents each item as a sequence of codes serving as its identifier, and a generative recommender that predicts the next item by autoregressively generating the target item identifier. However, in existing methods, both the tokenizer and the recommender are typically domain-specific, limiting their ability for effective transfer or adaptation to new domains. To this end, we propose UTGRec, a Universal item Tokenization approach for transferable Generative Recommendation. Specifically, we design a universal item tokenizer for encoding rich item semantics by adapting a multimodal large language model (MLLM). By devising tree-structured codebooks, we discretize content representations into corresponding codes for item tokenization. To effectively learn the universal item tokenizer on multiple domains, we introduce two key techniques in our approach. For raw content reconstruction, we employ dual lightweight decoders to reconstruct item text and images from discrete representations to capture general knowledge embedded in the content. For collaborative knowledge integration, we assume that co-occurring items are similar and integrate collaborative signals through co-occurrence alignment and reconstruction. Finally, we present a joint learning framework to pre-train and adapt the transferable generative recommender across multiple domains. Extensive experiments on four public datasets demonstrate the superiority of UTGRec compared to both traditional and generative recommendation baselines. 

---
# Pre-training Generative Recommender with Multi-Identifier Item Tokenization 

**Authors**: Bowen Zheng, Enze Liu, Zhongfu Chen, Zhongrui Ma, Yue Wang, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.04400)  

**Abstract**: Generative recommendation autoregressively generates item identifiers to recommend potential items. Existing methods typically adopt a one-to-one mapping strategy, where each item is represented by a single identifier. However, this scheme poses issues, such as suboptimal semantic modeling for low-frequency items and limited diversity in token sequence data. To overcome these limitations, we propose MTGRec, which leverages Multi-identifier item Tokenization to augment token sequence data for Generative Recommender pre-training. Our approach involves two key innovations: multi-identifier item tokenization and curriculum recommender pre-training. For multi-identifier item tokenization, we leverage the RQ-VAE as the tokenizer backbone and treat model checkpoints from adjacent training epochs as semantically relevant tokenizers. This allows each item to be associated with multiple identifiers, enabling a single user interaction sequence to be converted into several token sequences as different data groups. For curriculum recommender pre-training, we introduce a curriculum learning scheme guided by data influence estimation, dynamically adjusting the sampling probability of each data group during recommender pre-training. After pre-training, we fine-tune the model using a single tokenizer to ensure accurate item identification for recommendation. Extensive experiments on three public benchmark datasets demonstrate that MTGRec significantly outperforms both traditional and generative recommendation baselines in terms of effectiveness and scalability. 

---
# Decoding Recommendation Behaviors of In-Context Learning LLMs Through Gradient Descent 

**Authors**: Yi Xu, Weicong Qin, Weijie Yu, Ming He, Jianping Fan, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04386)  

**Abstract**: Recently, there has been a growing trend in utilizing large language models (LLMs) for recommender systems, referred to as LLMRec. A notable approach within this trend is not to fine-tune these models directly but instead to leverage In-Context Learning (ICL) methods tailored for LLMRec, denoted as LLM-ICL Rec. Many contemporary techniques focus on harnessing ICL content to enhance LLMRec performance.
However, optimizing LLMRec with ICL content presents unresolved challenges. Specifically, two key issues stand out: (1) the limited understanding of why using a few demonstrations without model fine-tuning can lead to better performance compared to zero-shot recommendations. (2) the lack of evaluation metrics for demonstrations in LLM-ICL Rec and the absence of the theoretical analysis and practical design for optimizing the generation of ICL content for recommendation contexts.
To address these two main issues, we propose a theoretical model, the LLM-ICL Recommendation Equivalent Gradient Descent model (LRGD) in this paper, which connects recommendation generation with gradient descent dynamics. We demonstrate that the ICL inference process in LLM aligns with the training procedure of its dual model, producing token predictions equivalent to the dual model's testing outputs. Building on these theoretical insights, we propose an evaluation metric for assessing demonstration quality. We integrate perturbations and regularizations in LRGD to enhance the robustness of the recommender system. To further improve demonstration effectiveness, prevent performance collapse, and ensure long-term adaptability, we also propose a two-stage optimization process in practice. Extensive experiments and detailed analysis on three Amazon datasets validate the theoretical equivalence and support the effectiveness of our theoretical analysis and practical module design. 

---
# Short Video Segment-level User Dynamic Interests Modeling in Personalized Recommendation 

**Authors**: Zhiyu He, Zhixin Ling, Jiayu Li, Zhiqiang Guo, Weizhi Ma, Xinchen Luo, Min Zhang, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.04237)  

**Abstract**: The rapid growth of short videos has necessitated effective recommender systems to match users with content tailored to their evolving preferences. Current video recommendation models primarily treat each video as a whole, overlooking the dynamic nature of user preferences with specific video segments. In contrast, our research focuses on segment-level user interest modeling, which is crucial for understanding how users' preferences evolve during video browsing. To capture users' dynamic segment interests, we propose an innovative model that integrates a hybrid representation module, a multi-modal user-video encoder, and a segment interest decoder. Our model addresses the challenges of capturing dynamic interest patterns, missing segment-level labels, and fusing different modalities, achieving precise segment-level interest prediction. We present two downstream tasks to evaluate the effectiveness of our segment interest modeling approach: video-skip prediction and short video recommendation. Our experiments on real-world short video datasets with diverse modalities show promising results on both tasks. It demonstrates that segment-level interest modeling brings a deep understanding of user engagement and enhances video recommendations. We also release a unique dataset that includes segment-level video data and diverse user behaviors, enabling further research in segment-level interest modeling. This work pioneers a novel perspective on understanding user segment-level preference, offering the potential for more personalized and engaging short video experiences. 

---
# Investigating and Mitigating Stereotype-aware Unfairness in LLM-based Recommendations 

**Authors**: Zihuai Zhao, Wenqi Fan, Yao Wu, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04199)  

**Abstract**: Large Language Models (LLMs) have demonstrated unprecedented language understanding and reasoning capabilities to capture diverse user preferences and advance personalized recommendations. Despite the growing interest in LLM-based personalized recommendations, unique challenges are brought to the trustworthiness of LLM-based recommender systems (LLM-RS), since LLMs are likely to inherit stereotypes that are embedded ubiquitously in word embeddings due to their training on large-scale uncurated datasets. This leads to LLM-RS exhibiting stereotypical linguistic associations between users and items. However, there remains a lack of studies investigating the simultaneous existence of stereotypes between users and items in LLM-RS. To bridge this gap, this study reveals a new variant of fairness between stereotype groups containing both users and items, to quantify discrimination against stereotypes in LLM-RS. Moreover, in this paper, to mitigate stereotype-aware unfairness in textual user and item information, we propose a novel framework (MoS), in which an insightful stereotype-wise routing strategy over multiple stereotype-relevant experts is designed to learn unbiased representations against different stereotypes in LLM- RS. Extensive experiments are conducted to analyze the influence of stereotype-aware fairness in LLM-RS and the effectiveness of our proposed methods, which consistently outperform competitive benchmarks under various fairness settings. 

---
# AiReview: An Open Platform for Accelerating Systematic Reviews with LLMs 

**Authors**: Xinyu Mao, Teerapong Leelanupab, Martin Potthast, Harrisen Scells, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2504.04193)  

**Abstract**: Systematic reviews are fundamental to evidence-based medicine. Creating one is time-consuming and labour-intensive, mainly due to the need to screen, or assess, many studies for inclusion in the review. Several tools have been developed to streamline this process, mostly relying on traditional machine learning methods. Large language models (LLMs) have shown potential in further accelerating the screening process. However, no tool currently allows end users to directly leverage LLMs for screening or facilitates systematic and transparent usage of LLM-assisted screening methods. This paper introduces (i) an extensible framework for applying LLMs to systematic review tasks, particularly title and abstract screening, and (ii) a web-based interface for LLM-assisted screening. Together, these elements form AiReview-a novel platform for LLM-assisted systematic review creation. AiReview is the first of its kind to bridge the gap between cutting-edge LLM-assisted screening methods and those that create medical systematic reviews. The tool is available at this https URL. The source code is also open sourced at this https URL. 

---
# Towards Principled Learning for Re-ranking in Recommender Systems 

**Authors**: Qunwei Li, Linghui Li, Jianbin Lin, Wenliang Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2504.04188)  

**Abstract**: As the final stage of recommender systems, re-ranking presents ordered item lists to users that best match their interests. It plays such a critical role and has become a trending research topic with much attention from both academia and industry. Recent advances of re-ranking are focused on attentive listwise modeling of interactions and mutual influences among items to be re-ranked. However, principles to guide the learning process of a re-ranker, and to measure the quality of the output of the re-ranker, have been always missing. In this paper, we study such principles to learn a good re-ranker. Two principles are proposed, including convergence consistency and adversarial consistency. These two principles can be applied in the learning of a generic re-ranker and improve its performance. We validate such a finding by various baseline methods over different datasets. 

---
# MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender 

**Authors**: Bohao Wang, Feng Liu, Jiawei Chen, Xingyu Lou, Changwang Zhang, Jun Wang, Yuegang Sun, Yan Feng, Chun Chen, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04178)  

**Abstract**: Large language models (LLMs), known for their comprehension capabilities and extensive knowledge, have been increasingly applied to recommendation systems (RS). Given the fundamental gap between the mechanism of LLMs and the requirement of RS, researchers have focused on fine-tuning LLMs with recommendation-specific data to enhance their performance. Language Modeling Loss (LML), originally designed for language generation tasks, is commonly adopted. However, we identify two critical limitations of LML: 1) it exhibits significant divergence from the recommendation objective; 2) it erroneously treats all fictitious item descriptions as negative samples, introducing misleading training signals.
To address these limitations, we propose a novel Masked Softmax Loss (MSL) tailored for fine-tuning LLMs on recommendation. MSL improves LML by identifying and masking invalid tokens that could lead to fictitious item descriptions during loss computation. This strategy can effectively avoid the interference from erroneous negative signals and ensure well alignment with the recommendation objective supported by theoretical guarantees. During implementation, we identify a potential challenge related to gradient vanishing of MSL. To overcome this, we further introduce the temperature coefficient and propose an Adaptive Temperature Strategy (ATS) that adaptively adjusts the temperature without requiring extensive hyperparameter tuning. Extensive experiments conducted on four public datasets further validate the effectiveness of MSL, achieving an average improvement of 42.24% in NDCG@10. The code is available at this https URL. 

---
# RIS-Empowered Integrated Location Sensing and Communication with Superimposed Pilots 

**Authors**: Wenchao Xia, Ben Zhao, Wankai Tang, Yongxu Zhu, Kai-Kit Wong, Sangarapillai Lambotharan, Hyundong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.04098)  

**Abstract**: In addition to enhancing wireless communication coverage quality, reconfigurable intelligent surface (RIS) technique can also assist in positioning. In this work, we consider RIS-assisted superimposed pilot and data transmission without the assumption availability of prior channel state information and position information of mobile user equipments (UEs). To tackle this challenge, we design a frame structure of transmission protocol composed of several location coherence intervals, each with pure-pilot and data-pilot transmission durations. The former is used to estimate UE locations, while the latter is time-slotted, duration of which does not exceed the channel coherence time, where the data and pilot signals are transmitted simultaneously. We conduct the Fisher Information matrix (FIM) analysis and derive \text {Cramér-Rao bound} (CRB) for the position estimation error. The inverse fast Fourier transform (IFFT) is adopted to obtain the estimation results of UE positions, which are then exploited for channel estimation. Furthermore, we derive the closed-form lower bound of the ergodic achievable rate of superimposed pilot (SP) transmission, which is used to optimize the phase profile of the RIS to maximize the achievable sum rate using the genetic algorithm. Finally, numerical results validate the accuracy of the UE position estimation using the IFFT algorithm and the superiority of the proposed SP scheme by comparison with the regular pilot scheme. 

---
# QE-RAG: A Robust Retrieval-Augmented Generation Benchmark for Query Entry Errors 

**Authors**: Kepu Zhang, Zhongxiang Sun, Weijie Yu, Xiaoxue Zang, Kai Zheng, Yang Song, Han Li, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04062)  

**Abstract**: Retriever-augmented generation (RAG) has become a widely adopted approach for enhancing the factual accuracy of large language models (LLMs). While current benchmarks evaluate the performance of RAG methods from various perspectives, they share a common assumption that user queries used for retrieval are error-free. However, in real-world interactions between users and LLMs, query entry errors such as keyboard proximity errors, visual similarity errors, and spelling errors are frequent. The impact of these errors on current RAG methods against such errors remains largely unexplored. To bridge this gap, we propose QE-RAG, the first robust RAG benchmark designed specifically to evaluate performance against query entry errors. We augment six widely used datasets by injecting three common types of query entry errors into randomly selected user queries at rates of 20\% and 40\%, simulating typical user behavior in real-world scenarios. We analyze the impact of these errors on LLM outputs and find that corrupted queries degrade model performance, which can be mitigated through query correction and training a robust retriever for retrieving relevant documents. Based on these insights, we propose a contrastive learning-based robust retriever training method and a retrieval-augmented query correction method. Extensive in-domain and cross-domain experiments reveal that: (1) state-of-the-art RAG methods including sequential, branching, and iterative methods, exhibit poor robustness to query entry errors; (2) our method significantly enhances the robustness of RAG when handling query entry errors and it's compatible with existing RAG methods, further improving their robustness. 

---
# Towards Robust Offline Evaluation: A Causal and Information Theoretic Framework for Debiasing Ranking Systems 

**Authors**: Seyedeh Baharan Khatami, Sayan Chakraborty, Ruomeng Xu, Babak Salimi  

**Link**: [PDF](https://arxiv.org/pdf/2504.03997)  

**Abstract**: Evaluating retrieval-ranking systems is crucial for developing high-performing models. While online A/B testing is the gold standard, its high cost and risks to user experience require effective offline methods. However, relying on historical interaction data introduces biases-such as selection, exposure, conformity, and position biases-that distort evaluation metrics, driven by the Missing-Not-At-Random (MNAR) nature of user interactions and favoring popular or frequently exposed items over true user preferences.
We propose a novel framework for robust offline evaluation of retrieval-ranking systems, transforming MNAR data into Missing-At-Random (MAR) through reweighting combined with black-box optimization, guided by neural estimation of information-theoretic metrics. Our contributions include (1) a causal formulation for addressing offline evaluation biases, (2) a system-agnostic debiasing framework, and (3) empirical validation of its effectiveness. This framework enables more accurate, fair, and generalizable evaluations, enhancing model assessment before deployment. 

---
# Automating Personalization: Prompt Optimization for Recommendation Reranking 

**Authors**: Chen Wang, Mingdai Yang, Zhiwei Liu, Pan Li, Linsey Pang, Qingsong Wen, Philip Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03965)  

**Abstract**: Modern recommender systems increasingly leverage large language models (LLMs) for reranking to improve personalization. However, existing approaches face two key limitations: (1) heavy reliance on manually crafted prompts that are difficult to scale, and (2) inadequate handling of unstructured item metadata that complicates preference inference. We present AGP (Auto-Guided Prompt Refinement), a novel framework that automatically optimizes user profile generation prompts for personalized reranking. AGP introduces two key innovations: (1) position-aware feedback mechanisms for precise ranking correction, and (2) batched training with aggregated feedback to enhance generalization. 

---
# Distillation and Refinement of Reasoning in Small Language Models for Document Re-ranking 

**Authors**: Chris Samarinas, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2504.03947)  

**Abstract**: We present a novel approach for training small language models for reasoning-intensive document ranking that combines knowledge distillation with reinforcement learning optimization. While existing methods often rely on expensive human annotations or large black-box language models, our methodology leverages web data and a teacher LLM to automatically generate high-quality training examples with relevance explanations. By framing document ranking as a reinforcement learning problem and incentivizing explicit reasoning capabilities, we train a compact 3B parameter language model that achieves state-of-the-art performance on the BRIGHT benchmark. Our model ranks third on the leaderboard while using substantially fewer parameters than other approaches, outperforming models that are over 20 times larger. Through extensive experiments, we demonstrate that generating explanations during inference, rather than directly predicting relevance scores, enables more effective reasoning with smaller language models. The self-supervised nature of our method offers a scalable and interpretable solution for modern information retrieval systems. 

---
# Blending Queries and Conversations: Understanding Tactics, Trust, Verification, and System Choice in Web Search and Chat Interactions 

**Authors**: Kerstin Mayerhofer, Rob Capra, David Elsweiler  

**Link**: [PDF](https://arxiv.org/pdf/2504.05156)  

**Abstract**: This paper presents a user study (N=22) where participants used an interface combining Web Search and a Generative AI-Chat feature to solve health-related information tasks. We study how people behaved with the interface, why they behaved in certain ways, and what the outcomes of these behaviours were. A think-aloud protocol captured their thought processes during searches. Our findings suggest that GenAI is neither a search panacea nor a major regression compared to standard Web Search interfaces. Qualitative and quantitative analyses identified 78 tactics across five categories and provided insight into how and why different interface features were used. We find evidence that pre-task confidence and trust both influenced which interface feature was used. In both systems, but particularly when using the chat feature, trust was often misplaced in favour of ease-of-use and seemingly perfect answers, leading to increased confidence post-search despite having incorrect results. We discuss what our findings mean in the context of our defined research questions and outline several open questions for future research. 

---
# Deconstructing Jazz Piano Style Using Machine Learning 

**Authors**: Huw Cheston, Reuben Bance, Peter M. C. Harrison  

**Link**: [PDF](https://arxiv.org/pdf/2504.05009)  

**Abstract**: Artistic style has been studied for centuries, and recent advances in machine learning create new possibilities for understanding it computationally. However, ensuring that machine-learning models produce insights aligned with the interests of practitioners and critics remains a significant challenge. Here, we focus on musical style, which benefits from a rich theoretical and mathematical analysis tradition. We train a variety of supervised-learning models to identify 20 iconic jazz musicians across a carefully curated dataset of 84 hours of recordings, and interpret their decision-making processes. Our models include a novel multi-input architecture that enables four musical domains (melody, harmony, rhythm, and dynamics) to be analysed separately. These models enable us to address fundamental questions in music theory and also advance the state-of-the-art in music performer identification (94% accuracy across 20 classes). We release open-source implementations of our models and an accompanying web application for exploring musical styles. 

---
# Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration 

**Authors**: Ran Xu, Wenqi Shi, Yuchen Zhuang, Yue Yu, Joyce C. Ho, Haoyu Wang, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04915)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems often struggle to handle multi-hop question-answering tasks accurately due to irrelevant context retrieval and limited complex reasoning capabilities. We introduce Collab-RAG, a collaborative training framework that leverages mutual enhancement between a white-box small language model (SLM) and a blackbox large language model (LLM) for RAG. Specifically, the SLM decomposes complex queries into simpler sub-questions, thus enhancing the accuracy of the retrieval and facilitating more effective reasoning by the black-box LLM. Concurrently, the black-box LLM provides feedback signals to improve the SLM's decomposition capability. We observe that Collab-RAG relies solely on supervision from an affordable black-box LLM without additional distillation from frontier LLMs, yet demonstrates strong generalization across multiple black-box LLMs. Experimental evaluations across five multi-hop QA datasets demonstrate that Collab-RAG substantially outperforms existing black-box-only and SLM fine-tuning baselines by 1.8%-14.2% on average. In particular, our fine-tuned 3B SLM surpasses a frozen 32B LLM in question decomposition, highlighting the efficiency of Collab-RAG in improving reasoning and retrieval for complex questions. The code of Collab-RAG is available on this https URL. 

---
# TathyaNyaya and FactLegalLlama: Advancing Factual Judgment Prediction and Explanation in the Indian Legal Context 

**Authors**: Shubham Kumar Nigam, Balaramamahanthi Deepak Patnaik, Shivam Mishra, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2504.04737)  

**Abstract**: In the landscape of Fact-based Judgment Prediction and Explanation (FJPE), reliance on factual data is essential for developing robust and realistic AI-driven decision-making tools. This paper introduces TathyaNyaya, the largest annotated dataset for FJPE tailored to the Indian legal context, encompassing judgments from the Supreme Court of India and various High Courts. Derived from the Hindi terms "Tathya" (fact) and "Nyaya" (justice), the TathyaNyaya dataset is uniquely designed to focus on factual statements rather than complete legal texts, reflecting real-world judicial processes where factual data drives outcomes. Complementing this dataset, we present FactLegalLlama, an instruction-tuned variant of the LLaMa-3-8B Large Language Model (LLM), optimized for generating high-quality explanations in FJPE tasks. Finetuned on the factual data in TathyaNyaya, FactLegalLlama integrates predictive accuracy with coherent, contextually relevant explanations, addressing the critical need for transparency and interpretability in AI-assisted legal systems. Our methodology combines transformers for binary judgment prediction with FactLegalLlama for explanation generation, creating a robust framework for advancing FJPE in the Indian legal domain. TathyaNyaya not only surpasses existing datasets in scale and diversity but also establishes a benchmark for building explainable AI systems in legal analysis. The findings underscore the importance of factual precision and domain-specific tuning in enhancing predictive performance and interpretability, positioning TathyaNyaya and FactLegalLlama as foundational resources for AI-assisted legal decision-making. 

---
# Sequential-NIAH: A Needle-In-A-Haystack Benchmark for Extracting Sequential Needles from Long Contexts 

**Authors**: Yifei Yu, Qian-Wen Zhang, Lingfeng Qiao, Di Yin, Fang Li, Jie Wang, Zengxi Chen, Suncong Zheng, Xiaolong Liang, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.04713)  

**Abstract**: Evaluating the ability of large language models (LLMs) to handle extended contexts is critical, particularly for retrieving information relevant to specific queries embedded within lengthy inputs. We introduce Sequential-NIAH, a benchmark specifically designed to evaluate the capability of LLMs to extract sequential information items (known as needles) from long contexts. The benchmark comprises three types of needle generation pipelines: synthetic, real, and open-domain QA. It includes contexts ranging from 8K to 128K tokens in length, with a dataset of 14,000 samples (2,000 reserved for testing). To facilitate evaluation on this benchmark, we trained a synthetic data-driven evaluation model capable of evaluating answer correctness based on chronological or logical order, achieving an accuracy of 99.49% on synthetic test data. We conducted experiments on six well-known LLMs, revealing that even the best-performing model achieved a maximum accuracy of only 63.15%. Further analysis highlights the growing challenges posed by increasing context lengths and the number of needles, underscoring substantial room for improvement. Additionally, noise robustness experiments validate the reliability of the benchmark, making Sequential-NIAH an important reference for advancing research on long text extraction capabilities of LLMs. 

---
# UniRVQA: A Unified Framework for Retrieval-Augmented Vision Question Answering via Self-Reflective Joint Training 

**Authors**: Jiaqi Deng, Kaize Shi, Zonghan Wu, Huan Huo, Dingxian Wang, Guandong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04065)  

**Abstract**: Knowledge-based Vision Question Answering (KB-VQA) systems address complex visual-grounded questions requiring external knowledge, such as web-sourced encyclopedia articles. Existing methods often use sequential and separate frameworks for the retriever and the generator with limited parametric knowledge sharing. However, since both retrieval and generation tasks require accurate understanding of contextual and external information, such separation can potentially lead to suboptimal system performance. Another key challenge is the integration of multimodal information. General-purpose multimodal pre-trained models, while adept at multimodal representation learning, struggle with fine-grained retrieval required for knowledge-intensive visual questions. Recent specialized pre-trained models mitigate the issue, but are computationally expensive. To bridge the gap, we propose a Unified Retrieval-Augmented VQA framework (UniRVQA). UniRVQA adapts general multimodal pre-trained models for fine-grained knowledge-intensive tasks within a unified framework, enabling cross-task parametric knowledge sharing and the extension of existing multimodal representation learning capability. We further introduce a reflective-answering mechanism that allows the model to explicitly evaluate and refine its knowledge boundary. Additionally, we integrate late interaction into the retrieval-augmented generation joint training process to enhance fine-grained understanding of queries and documents. Our approach achieves competitive performance against state-of-the-art models, delivering a significant 4.7% improvement in answering accuracy, and brings an average 7.5% boost in base MLLMs' VQA performance. 

---
# Structured Extraction of Process Structure Properties Relationships in Materials Science 

**Authors**: Amit K Verma, Zhisong Zhang, Junwon Seo, Robin Kuo, Runbo Jiang, Emma Strubell, Anthony D Rollett  

**Link**: [PDF](https://arxiv.org/pdf/2504.03979)  

**Abstract**: With the advent of large language models (LLMs), the vast unstructured text within millions of academic papers is increasingly accessible for materials discovery, although significant challenges remain. While LLMs offer promising few- and zero-shot learning capabilities, particularly valuable in the materials domain where expert annotations are scarce, general-purpose LLMs often fail to address key materials-specific queries without further adaptation. To bridge this gap, fine-tuning LLMs on human-labeled data is essential for effective structured knowledge extraction. In this study, we introduce a novel annotation schema designed to extract generic process-structure-properties relationships from scientific literature. We demonstrate the utility of this approach using a dataset of 128 abstracts, with annotations drawn from two distinct domains: high-temperature materials (Domain I) and uncertainty quantification in simulating materials microstructure (Domain II). Initially, we developed a conditional random field (CRF) model based on MatBERT, a domain-specific BERT variant, and evaluated its performance on Domain I. Subsequently, we compared this model with a fine-tuned LLM (GPT-4o from OpenAI) under identical conditions. Our results indicate that fine-tuning LLMs can significantly improve entity extraction performance over the BERT-CRF baseline on Domain I. However, when additional examples from Domain II were incorporated, the performance of the BERT-CRF model became comparable to that of the GPT-4o model. These findings underscore the potential of our schema for structured knowledge extraction and highlight the complementary strengths of both modeling approaches. 

---
# VideoComp: Advancing Fine-Grained Compositional and Temporal Alignment in Video-Text Models 

**Authors**: Dahun Kim, AJ Piergiovanni, Ganesh Mallya, Anelia Angelova  

**Link**: [PDF](https://arxiv.org/pdf/2504.03970)  

**Abstract**: We introduce VideoComp, a benchmark and learning framework for advancing video-text compositionality understanding, aimed at improving vision-language models (VLMs) in fine-grained temporal alignment. Unlike existing benchmarks focused on static image-text compositionality or isolated single-event videos, our benchmark targets alignment in continuous multi-event videos. Leveraging video-text datasets with temporally localized event captions (e.g. ActivityNet-Captions, YouCook2), we construct two compositional benchmarks, ActivityNet-Comp and YouCook2-Comp. We create challenging negative samples with subtle temporal disruptions such as reordering, action word replacement, partial captioning, and combined disruptions. These benchmarks comprehensively test models' compositional sensitivity across extended, cohesive video-text sequences. To improve model performance, we propose a hierarchical pairwise preference loss that strengthens alignment with temporally accurate pairs and gradually penalizes increasingly disrupted ones, encouraging fine-grained compositional learning. To mitigate the limited availability of densely annotated video data, we introduce a pretraining strategy that concatenates short video-caption pairs to simulate multi-event sequences. We evaluate video-text foundational models and large multimodal models (LMMs) on our benchmark, identifying both strengths and areas for improvement in compositionality. Overall, our work provides a comprehensive framework for evaluating and enhancing model capabilities in achieving fine-grained, temporally coherent video-text alignment. 

---
# Practical Poisoning Attacks against Retrieval-Augmented Generation 

**Authors**: Baolei Zhang, Yuxi Chen, Minghong Fang, Zhuqing Liu, Lihai Nie, Tong Li, Zheli Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03957)  

**Abstract**: Large language models (LLMs) have demonstrated impressive natural language processing abilities but face challenges such as hallucination and outdated knowledge. Retrieval-Augmented Generation (RAG) has emerged as a state-of-the-art approach to mitigate these issues. While RAG enhances LLM outputs, it remains vulnerable to poisoning attacks. Recent studies show that injecting poisoned text into the knowledge database can compromise RAG systems, but most existing attacks assume that the attacker can insert a sufficient number of poisoned texts per query to outnumber correct-answer texts in retrieval, an assumption that is often unrealistic. To address this limitation, we propose CorruptRAG, a practical poisoning attack against RAG systems in which the attacker injects only a single poisoned text, enhancing both feasibility and stealth. Extensive experiments across multiple datasets demonstrate that CorruptRAG achieves higher attack success rates compared to existing baselines. 

---
