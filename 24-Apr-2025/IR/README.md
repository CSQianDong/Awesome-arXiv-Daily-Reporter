# Information Leakage of Sentence Embeddings via Generative Embedding Inversion Attacks 

**Authors**: Antonios Tragoudaras, Theofanis Aslanidis, Emmanouil Georgios Lionis, Marina Orozco González, Panagiotis Eustratiadis  

**Link**: [PDF](https://arxiv.org/pdf/2504.16609)  

**Abstract**: Text data are often encoded as dense vectors, known as embeddings, which capture semantic, syntactic, contextual, and domain-specific information. These embeddings, widely adopted in various applications, inherently contain rich information that may be susceptible to leakage under certain attacks. The GEIA framework highlights vulnerabilities in sentence embeddings, demonstrating that they can reveal the original sentences they represent. In this study, we reproduce GEIA's findings across various neural sentence embedding models. Additionally, we contribute new analysis to examine whether these models leak sensitive information from their training datasets. We propose a simple yet effective method without any modification to the attacker's architecture proposed in GEIA. The key idea is to examine differences between log-likelihood for masked and original variants of data that sentence embedding models have been pre-trained on, calculated on the embedding space of the attacker. Our findings indicate that following our approach, an adversary party can recover meaningful sensitive information related to the pre-training knowledge of the popular models used for creating sentence embeddings, seriously undermining their security. Our code is available on: this https URL 

---
# MMHCL: Multi-Modal Hypergraph Contrastive Learning for Recommendation 

**Authors**: Xu Guo, Tong Zhang, Fuyun Wang, Xudong Wang, Xiaoya Zhang, Xin Liu, Zhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2504.16576)  

**Abstract**: The burgeoning presence of multimodal content-sharing platforms propels the development of personalized recommender systems. Previous works usually suffer from data sparsity and cold-start problems, and may fail to adequately explore semantic user-product associations from multimodal data. To address these issues, we propose a novel Multi-Modal Hypergraph Contrastive Learning (MMHCL) framework for user recommendation. For a comprehensive information exploration from user-product relations, we construct two hypergraphs, i.e. a user-to-user (u2u) hypergraph and an item-to-item (i2i) hypergraph, to mine shared preferences among users and intricate multimodal semantic resemblance among items, respectively. This process yields denser second-order semantics that are fused with first-order user-item interaction as complementary to alleviate the data sparsity issue. Then, we design a contrastive feature enhancement paradigm by applying synergistic contrastive learning. By maximizing/minimizing the mutual information between second-order (e.g. shared preference pattern for users) and first-order (information of selected items for users) embeddings of the same/different users and items, the feature distinguishability can be effectively enhanced. Compared with using sparse primary user-item interaction only, our MMHCL obtains denser second-order hypergraphs and excavates more abundant shared attributes to explore the user-product associations, which to a certain extent alleviates the problems of data sparsity and cold-start. Extensive experiments have comprehensively demonstrated the effectiveness of our method. Our code is publicly available at: this https URL. 

---
# Enhancing LLM-Based Agents via Global Planning and Hierarchical Execution 

**Authors**: Junjie Chen, Haitao Li, Jingli Yang, Yiqun Liu, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2504.16563)  

**Abstract**: Intelligent agent systems based on Large Language Models (LLMs) have shown great potential in real-world applications. However, existing agent frameworks still face critical limitations in task planning and execution, restricting their effectiveness and generalizability. Specifically, current planning methods often lack clear global goals, leading agents to get stuck in local branches, or produce non-executable plans. Meanwhile, existing execution mechanisms struggle to balance complexity and stability, and their limited action space restricts their ability to handle diverse real-world tasks. To address these limitations, we propose GoalAct, a novel agent framework that introduces a continuously updated global planning mechanism and integrates a hierarchical execution strategy. GoalAct decomposes task execution into high-level skills, including searching, coding, writing and more, thereby reducing planning complexity while enhancing the agents' adaptability across diverse task scenarios. We evaluate GoalAct on LegalAgentBench, a benchmark with multiple types of legal tasks that require the use of multiple types of tools. Experimental results demonstrate that GoalAct achieves state-of-the-art (SOTA) performance, with an average improvement of 12.22% in success rate. These findings highlight GoalAct's potential to drive the development of more advanced intelligent agent systems, making them more effective across complex real-world applications. Our code can be found at this https URL. 

---
# Modality Reliability Guided Multimodal Recommendation 

**Authors**: Xue Dong, Xuemeng Song, Na Zheng, Sicheng Zhao, Guiguang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.16524)  

**Abstract**: Multimodal recommendation faces an issue of the performance degradation that the uni-modal recommendation sometimes achieves the better performance. A possible reason is that the unreliable item modality data hurts the fusion result. Several existing studies have introduced weights for different modalities to reduce the contribution of the unreliable modality data in predicting the final user rating. However, they fail to provide appropriate supervisions for learning the modality weights, making the learned weights imprecise. Therefore, we propose a modality reliability guided multimodal recommendation framework that uniquely learns the modality weights supervised by the modality reliability. Considering that there is no explicit label provided for modality reliability, we resort to automatically identify it through the BPR recommendation objective. In particular, we define a modality reliability vector as the supervision label by the difference between modality-specific user ratings to positive and negative items, where a larger difference indicates a higher reliability of the modality as the BPR objective is better satisfied. Furthermore, to enhance the effectiveness of the supervision, we calculate the confidence level for the modality reliability vector, which dynamically adjusts the supervision strength and eliminates the harmful supervision. Extensive experiments on three real-world datasets show the effectiveness of the proposed method. 

---
# Killing Two Birds with One Stone: Unifying Retrieval and Ranking with a Single Generative Recommendation Model 

**Authors**: Luankang Zhang, Kenan Song, Yi Quan Lee, Wei Guo, Hao Wang, Yawen Li, Huifeng Guo, Yong Liu, Defu Lian, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.16454)  

**Abstract**: In recommendation systems, the traditional multi-stage paradigm, which includes retrieval and ranking, often suffers from information loss between stages and diminishes performance. Recent advances in generative models, inspired by natural language processing, suggest the potential for unifying these stages to mitigate such loss. This paper presents the Unified Generative Recommendation Framework (UniGRF), a novel approach that integrates retrieval and ranking into a single generative model. By treating both stages as sequence generation tasks, UniGRF enables sufficient information sharing without additional computational costs, while remaining model-agnostic. To enhance inter-stage collaboration, UniGRF introduces a ranking-driven enhancer module that leverages the precision of the ranking stage to refine retrieval processes, creating an enhancement loop. Besides, a gradient-guided adaptive weighter is incorporated to dynamically balance the optimization of retrieval and ranking, ensuring synchronized performance improvements. Extensive experiments demonstrate that UniGRF significantly outperforms existing models on benchmark datasets, confirming its effectiveness in facilitating information transfer. Ablation studies and further experiments reveal that UniGRF not only promotes efficient collaboration between stages but also achieves synchronized optimization. UniGRF provides an effective, scalable, and compatible framework for generative recommendation systems. 

---
# A Survey of Foundation Model-Powered Recommender Systems: From Feature-Based, Generative to Agentic Paradigms 

**Authors**: Chengkai Huang, Hongtao Huang, Tong Yu, Kaige Xie, Junda Wu, Shuai Zhang, Julian Mcauley, Dietmar Jannach, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.16420)  

**Abstract**: Recommender systems (RS) have become essential in filtering information and personalizing content for users. RS techniques have traditionally relied on modeling interactions between users and items as well as the features of content using models specific to each task. The emergence of foundation models (FMs), large scale models trained on vast amounts of data such as GPT, LLaMA and CLIP, is reshaping the recommendation paradigm. This survey provides a comprehensive overview of the Foundation Models for Recommender Systems (FM4RecSys), covering their integration in three paradigms: (1) Feature-Based augmentation of representations, (2) Generative recommendation approaches, and (3) Agentic interactive systems. We first review the data foundations of RS, from traditional explicit or implicit feedback to multimodal content sources. We then introduce FMs and their capabilities for representation learning, natural language understanding, and multi-modal reasoning in RS contexts. The core of the survey discusses how FMs enhance RS under different paradigms. Afterward, we examine FM applications in various recommendation tasks. Through an analysis of recent research, we highlight key opportunities that have been realized as well as challenges encountered. Finally, we outline open research directions and technical challenges for next-generation FM4RecSys. This survey not only reviews the state-of-the-art methods but also provides a critical analysis of the trade-offs among the feature-based, the generative, and the agentic paradigms, outlining key open issues and future research directions. 

---
# Disentangling and Generating Modalities for Recommendation in Missing Modality Scenarios 

**Authors**: Jiwan Kim, Hongseok Kang, Sein Kim, Kibum Kim, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.16352)  

**Abstract**: Multi-modal recommender systems (MRSs) have achieved notable success in improving personalization by leveraging diverse modalities such as images, text, and audio. However, two key challenges remain insufficiently addressed: (1) Insufficient consideration of missing modality scenarios and (2) the overlooking of unique characteristics of modality features. These challenges result in significant performance degradation in realistic situations where modalities are missing. To address these issues, we propose Disentangling and Generating Modality Recommender (DGMRec), a novel framework tailored for missing modality scenarios. DGMRec disentangles modality features into general and specific modality features from an information-based perspective, enabling richer representations for recommendation. Building on this, it generates missing modality features by integrating aligned features from other modalities and leveraging user modality preferences. Extensive experiments show that DGMRec consistently outperforms state-of-the-art MRSs in challenging scenarios, including missing modalities and new item settings as well as diverse missing ratios and varying levels of missing modalities. Moreover, DGMRec's generation-based approach enables cross-modal retrieval, a task inapplicable for existing MRSs, highlighting its adaptability and potential for real-world applications. Our code is available at this https URL. 

---
# MPAD: A New Dimension-Reduction Method for Preserving Nearest Neighbors in High-Dimensional Vector Search 

**Authors**: Jiuzhou Fu, Dongfang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.16335)  

**Abstract**: High-dimensional vector embeddings are widely used in retrieval systems, yet dimensionality reduction (DR) is seldom applied due to its tendency to distort nearest-neighbor (NN) structure critical for search. Existing DR techniques such as PCA and UMAP optimize global or manifold-preserving criteria, rather than retrieval-specific objectives. We present MPAD: Maximum Pairwise Absolute Difference, an unsupervised DR method that explicitly preserves approximate NN relations by maximizing the margin between k-NNs and non-k-NNs under a soft orthogonality constraint. This design enables MPAD to retain ANN-relevant geometry without supervision or changes to the original embedding model. Experiments across multiple domains show that MPAD consistently outperforms standard DR methods in preserving neighborhood structure, enabling more accurate search in reduced dimensions. 

---
# CLIRudit: Cross-Lingual Information Retrieval of Scientific Documents 

**Authors**: Francisco Valentini, Diego Kozlowski, Vincent Larivière  

**Link**: [PDF](https://arxiv.org/pdf/2504.16264)  

**Abstract**: Cross-lingual information retrieval (CLIR) consists in finding relevant documents in a language that differs from the language of the queries. This paper presents CLIRudit, a new dataset created to evaluate cross-lingual academic search, focusing on English queries and French documents. The dataset is built using bilingual article metadata from Érudit, a Canadian publishing platform, and is designed to represent scenarios in which researchers search for scholarly content in languages other than English. We perform a comprehensive benchmarking of different zero-shot first-stage retrieval methods on the dataset, including dense and sparse retrievers, query and document machine translation, and state-of-the-art multilingual retrievers. Our results show that large dense retrievers, not necessarily trained for the cross-lingual retrieval task, can achieve zero-shot performance comparable to using ground truth human translations, without the need for machine translation. Sparse retrievers, such as BM25 or SPLADE, combined with document translation, show competitive results, providing an efficient alternative to large dense models. This research advances the understanding of cross-lingual academic information retrieval and provides a framework that others can use to build comparable datasets across different languages and disciplines. By making the dataset and code publicly available, we aim to facilitate further research that will help make scientific knowledge more accessible across language barriers. 

---
# Detecting Actionable Requests and Offers on Social Media During Crises Using LLMs 

**Authors**: Ahmed El Fekih Zguir, Ferda Ofli, Muhammad Imran  

**Link**: [PDF](https://arxiv.org/pdf/2504.16144)  

**Abstract**: Natural disasters often result in a surge of social media activity, including requests for assistance, offers of help, sentiments, and general updates. To enable humanitarian organizations to respond more efficiently, we propose a fine-grained hierarchical taxonomy to systematically organize crisis-related information about requests and offers into three critical dimensions: supplies, emergency personnel, and actions. Leveraging the capabilities of Large Language Models (LLMs), we introduce Query-Specific Few-shot Learning (QSF Learning) that retrieves class-specific labeled examples from an embedding database to enhance the model's performance in detecting and classifying posts. Beyond classification, we assess the actionability of messages to prioritize posts requiring immediate attention. Extensive experiments demonstrate that our approach outperforms baseline prompting strategies, effectively identifying and prioritizing actionable requests and offers. 

---
# LegalRAG: A Hybrid RAG System for Multilingual Legal Information Retrieval 

**Authors**: Muhammad Rafsan Kabir, Rafeed Mohammad Sultan, Fuad Rahman, Mohammad Ruhul Amin, Sifat Momen, Nabeel Mohammed, Shafin Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2504.16121)  

**Abstract**: Natural Language Processing (NLP) and computational linguistic techniques are increasingly being applied across various domains, yet their use in legal and regulatory tasks remains limited. To address this gap, we develop an efficient bilingual question-answering framework for regulatory documents, specifically the Bangladesh Police Gazettes, which contain both English and Bangla text. Our approach employs modern Retrieval Augmented Generation (RAG) pipelines to enhance information retrieval and response generation. In addition to conventional RAG pipelines, we propose an advanced RAG-based approach that improves retrieval performance, leading to more precise answers. This system enables efficient searching for specific government legal notices, making legal information more accessible. We evaluate both our proposed and conventional RAG systems on a diverse test set on Bangladesh Police Gazettes, demonstrating that our approach consistently outperforms existing methods across all evaluation metrics. 

---
# Search Timelines: Visualizing Search History to Enable Cross-Session Exploratory Search 

**Authors**: Orland Hoeber, Md Nazmul Islam, Miriam Boon, Dale Storie, Veronica Ramshaw  

**Link**: [PDF](https://arxiv.org/pdf/2504.16741)  

**Abstract**: Purpose: The timespan over which exploratory searching can occur, as well as the scope and volume of the search activities undertaken, can make it difficult for searchers to remember key details about their search activities. These difficulties are present both in the midst of searching as well as when resuming a search that spans multiple sessions. In this paper, we present a search interface designed to support cross-session exploratory search in a public digital library context. Methods: Search Timelines provides a visualization of current and past search activities via a dynamic timeline of the search activity (queries and saved resources). This timeline is presented at two levels of detail. An overview timeline is provided alongside the search results in a typical search engine results page design. A detailed timeline is provided in the workspace, where searchers can review the history of their search activities and their saved resources. A controlled laboratory study was conducted to compare this approach to a baseline interface modelled after a typical public digital library search/workspace interface. Results: Participants who used Search Timelines reported higher levels of user engagement, usability, and perceived knowledge gain, during an initial search session and when resuming the search after a 7-8 day interval. This came at the expense of the searchers taking more time to complete the search task, which we view as positive evidence of engagement in cross-session exploratory search processes. Conclusion: Search Timelines serves as an example of how lightweight visualization approaches can be used to enhance typical search interface designs to support exploratory search. The results highlight the value of providing persistent representations of past search activities within the search interface. 

---
# A Unified Retrieval Framework with Document Ranking and EDU Filtering for Multi-document Summarization 

**Authors**: Shiyin Tan, Jaeeon Park, Dongyuan Li, Renhe Jiang, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2504.16711)  

**Abstract**: In the field of multi-document summarization (MDS), transformer-based models have demonstrated remarkable success, yet they suffer an input length limitation. Current methods apply truncation after the retrieval process to fit the context length; however, they heavily depend on manually well-crafted queries, which are impractical to create for each document set for MDS. Additionally, these methods retrieve information at a coarse granularity, leading to the inclusion of irrelevant content. To address these issues, we propose a novel retrieval-based framework that integrates query selection and document ranking and shortening into a unified process. Our approach identifies the most salient elementary discourse units (EDUs) from input documents and utilizes them as latent queries. These queries guide the document ranking by calculating relevance scores. Instead of traditional truncation, our approach filters out irrelevant EDUs to fit the context length, ensuring that only critical information is preserved for summarization. We evaluate our framework on multiple MDS datasets, demonstrating consistent improvements in ROUGE metrics while confirming its scalability and flexibility across diverse model architectures. Additionally, we validate its effectiveness through an in-depth analysis, emphasizing its ability to dynamically select appropriate queries and accurately rank documents based on their relevance scores. These results demonstrate that our framework effectively addresses context-length constraints, establishing it as a robust and reliable solution for MDS. 

---
# DAPLSR: Data Augmentation Partial Least Squares Regression Model via Manifold Optimization 

**Authors**: Haoran Chen, Jiapeng Liu, Jiafan Wang, Wenjun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.16639)  

**Abstract**: Traditional Partial Least Squares Regression (PLSR) models frequently underperform when handling data characterized by uneven categories. To address the issue, this paper proposes a Data Augmentation Partial Least Squares Regression (DAPLSR) model via manifold optimization. The DAPLSR model introduces the Synthetic Minority Over-sampling Technique (SMOTE) to increase the number of samples and utilizes the Value Difference Metric (VDM) to select the nearest neighbor samples that closely resemble the original samples for generating synthetic samples. In solving the model, in order to obtain a more accurate numerical solution for PLSR, this paper proposes a manifold optimization method that uses the geometric properties of the constraint space to improve model degradation and optimization. Comprehensive experiments show that the proposed DAPLSR model achieves superior classification performance and outstanding evaluation metrics on various datasets, significantly outperforming existing methods. 

---
