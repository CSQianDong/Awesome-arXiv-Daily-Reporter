# KGMEL: Knowledge Graph-Enhanced Multimodal Entity Linking 

**Authors**: Juyeon Kim, Geon Lee, Taeuk Kim, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15135)  

**Abstract**: Entity linking (EL) aligns textual mentions with their corresponding entities in a knowledge base, facilitating various applications such as semantic search and question answering. Recent advances in multimodal entity linking (MEL) have shown that combining text and images can reduce ambiguity and improve alignment accuracy. However, most existing MEL methods overlook the rich structural information available in the form of knowledge-graph (KG) triples. In this paper, we propose KGMEL, a novel framework that leverages KG triples to enhance MEL. Specifically, it operates in three stages: (1) Generation: Produces high-quality triples for each mention by employing vision-language models based on its text and images. (2) Retrieval: Learns joint mention-entity representations, via contrastive learning, that integrate text, images, and (generated or KG) triples to retrieve candidate entities for each mention. (3) Reranking: Refines the KG triples of the candidate entities and employs large language models to identify the best-matching entity for the mention. Extensive experiments on benchmark datasets demonstrate that KGMEL outperforms existing methods. Our code and datasets are available at: this https URL. 

---
# The Great Nugget Recall: Automating Fact Extraction and RAG Evaluation with Large Language Models 

**Authors**: Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick Craswell, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15068)  

**Abstract**: Large Language Models (LLMs) have significantly enhanced the capabilities of information access systems, especially with retrieval-augmented generation (RAG). Nevertheless, the evaluation of RAG systems remains a barrier to continued progress, a challenge we tackle in this work by proposing an automatic evaluation framework that is validated against human annotations. We believe that the nugget evaluation methodology provides a solid foundation for evaluating RAG systems. This approach, originally developed for the TREC Question Answering (QA) Track in 2003, evaluates systems based on atomic facts that should be present in good answers. Our efforts focus on "refactoring" this methodology, where we describe the AutoNuggetizer framework that specifically applies LLMs to both automatically create nuggets and automatically assign nuggets to system answers. In the context of the TREC 2024 RAG Track, we calibrate a fully automatic approach against strategies where nuggets are created manually or semi-manually by human assessors and then assigned manually to system answers. Based on results from a community-wide evaluation, we observe strong agreement at the run level between scores derived from fully automatic nugget evaluation and human-based variants. The agreement is stronger when individual framework components such as nugget assignment are automated independently. This suggests that our evaluation framework provides tradeoffs between effort and quality that can be used to guide the development of future RAG systems. However, further research is necessary to refine our approach, particularly in establishing robust per-topic agreement to diagnose system failures effectively. 

---
# Linear Item-Item Model with Neural Knowledge for Session-based Recommendation 

**Authors**: Minjin Choi, Sunkyung Lee, Seongmin Park, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.15057)  

**Abstract**: Session-based recommendation (SBR) aims to predict users' subsequent actions by modeling short-term interactions within sessions. Existing neural models primarily focus on capturing complex dependencies for sequential item transitions. As an alternative solution, linear item-item models mainly identify strong co-occurrence patterns across items and support faster inference speed. Although each paradigm has been actively studied in SBR, their fundamental differences in capturing item relationships and how to bridge these distinct modeling paradigms effectively remain unexplored. In this paper, we propose a novel SBR model, namely Linear Item-Item model with Neural Knowledge (LINK), which integrates both types of knowledge into a unified linear framework. Specifically, we design two specialized components of LINK: (i) Linear knowledge-enhanced Item-item Similarity model (LIS), which refines the item similarity correlation via self-distillation, and (ii) Neural knowledge-enhanced Item-item Transition model (NIT), which seamlessly incorporates complicated neural knowledge distilled from the off-the-shelf neural model. Extensive experiments demonstrate that LINK outperforms state-of-the-art linear SBR models across six real-world datasets, achieving improvements of up to 14.78% and 11.04% in Recall@20 and MRR@20 while showing up to 813x fewer inference FLOPs. Our code is available at this https URL. 

---
# Understanding Accuracy-Fairness Trade-offs in Re-ranking through Elasticity in Economics 

**Authors**: Chen Xu, Jujia Zhao, Wenjie Wang, Liang Pang, Jun Xu, Tat-Seng Chua, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2504.14991)  

**Abstract**: Fairness is an increasingly important factor in re-ranking tasks. Prior work has identified a trade-off between ranking accuracy and item fairness. However, the underlying mechanisms are still not fully understood. An analogy can be drawn between re-ranking and the dynamics of economic transactions. The accuracy-fairness trade-off parallels the coupling of the commodity tax transfer process. Fairness considerations in re-ranking, similar to a commodity tax on suppliers, ultimately translate into a cost passed on to consumers. Analogously, item-side fairness constraints result in a decline in user-side accuracy. In economics, the extent to which commodity tax on the supplier (item fairness) transfers to commodity tax on users (accuracy loss) is formalized using the notion of elasticity. The re-ranking fairness-accuracy trade-off is similarly governed by the elasticity of utility between item groups. This insight underscores the limitations of current fair re-ranking evaluations, which often rely solely on a single fairness metric, hindering comprehensive assessment of fair re-ranking algorithms. Centered around the concept of elasticity, this work presents two significant contributions. We introduce the Elastic Fairness Curve (EF-Curve) as an evaluation framework. This framework enables a comparative analysis of algorithm performance across different elasticity levels, facilitating the selection of the most suitable approach. Furthermore, we propose ElasticRank, a fair re-ranking algorithm that employs elasticity calculations to adjust inter-item distances within a curved space. Experiments on three widely used ranking datasets demonstrate its effectiveness and efficiency. 

---
# ColBERT-serve: Efficient Multi-Stage Memory-Mapped Scoring 

**Authors**: Kaili Huang, Thejas Venkatesh, Uma Dingankar, Antonio Mallia, Daniel Campos, Jian Jiao, Christopher Potts, Matei Zaharia, Kwabena Boahen, Omar Khattab, Saarthak Sarup, Keshav Santhanam  

**Link**: [PDF](https://arxiv.org/pdf/2504.14903)  

**Abstract**: We study serving retrieval models, specifically late interaction models like ColBERT, to many concurrent users at once and under a small budget, in which the index may not fit in memory. We present ColBERT-serve, a novel serving system that applies a memory-mapping strategy to the ColBERT index, reducing RAM usage by 90% and permitting its deployment on cheap servers, and incorporates a multi-stage architecture with hybrid scoring, reducing ColBERT's query latency and supporting many concurrent queries in parallel. 

---
# Enhancing the Patent Matching Capability of Large Language Models via the Memory Graph 

**Authors**: Qiushi Xiong, Zhipeng Xu, Zhenghao Liu, Mengjia Wang, Zulong Chen, Yue Sun, Yu Gu, Xiaohua Li, Ge Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14845)  

**Abstract**: Intellectual Property (IP) management involves strategically protecting and utilizing intellectual assets to enhance organizational innovation, competitiveness, and value creation. Patent matching is a crucial task in intellectual property management, which facilitates the organization and utilization of patents. Existing models often rely on the emergent capabilities of Large Language Models (LLMs) and leverage them to identify related patents directly. However, these methods usually depend on matching keywords and overlook the hierarchical classification and categorical relationships of patents. In this paper, we propose MemGraph, a method that augments the patent matching capabilities of LLMs by incorporating a memory graph derived from their parametric memory. Specifically, MemGraph prompts LLMs to traverse their memory to identify relevant entities within patents, followed by attributing these entities to corresponding ontologies. After traversing the memory graph, we utilize extracted entities and ontologies to improve the capability of LLM in comprehending the semantics of patents. Experimental results on the PatentMatch dataset demonstrate the effectiveness of MemGraph, achieving a 17.68% performance improvement over baseline LLMs. The further analysis highlights the generalization ability of MemGraph across various LLMs, both in-domain and out-of-domain, and its capacity to enhance the internal reasoning processes of LLMs during patent matching. All data and codes are available at this https URL. 

---
# Exploring $\ell_0$ Sparsification for Inference-free Sparse Retrievers 

**Authors**: Xinjie Shen, Zhichao Geng, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14839)  

**Abstract**: With increasing demands for efficiency, information retrieval has developed a branch of sparse retrieval, further advancing towards inference-free retrieval where the documents are encoded during indexing time and there is no model-inference for queries. Existing sparse retrieval models rely on FLOPS regularization for sparsification, while this mechanism was originally designed for Siamese encoders, it is considered to be suboptimal in inference-free scenarios which is asymmetric. Previous attempts to adapt FLOPS for inference-free scenarios have been limited to rule-based methods, leaving the potential of sparsification approaches for inference-free retrieval models largely unexplored. In this paper, we explore $\ell_0$ inspired sparsification manner for inference-free retrievers. Through comprehensive out-of-domain evaluation on the BEIR benchmark, our method achieves state-of-the-art performance among inference-free sparse retrieval models and is comparable to leading Siamese sparse retrieval models. Furthermore, we provide insights into the trade-off between retrieval effectiveness and computational efficiency, demonstrating practical value for real-world applications. 

---
# The 1st EReL@MIR Workshop on Efficient Representation Learning for Multimodal Information Retrieval 

**Authors**: Junchen Fu, Xuri Ge, Xin Xin, Haitao Yu, Yue Feng, Alexandros Karatzoglou, Ioannis Arapakis, Joemon M. Jose  

**Link**: [PDF](https://arxiv.org/pdf/2504.14788)  

**Abstract**: Multimodal representation learning has garnered significant attention in the AI community, largely due to the success of large pre-trained multimodal foundation models like LLaMA, GPT, Mistral, and CLIP. These models have achieved remarkable performance across various tasks of multimodal information retrieval (MIR), including web search, cross-modal retrieval, and recommender systems, etc. However, due to their enormous parameter sizes, significant efficiency challenges emerge across training, deployment, and inference stages when adapting these models' representation for IR tasks. These challenges present substantial obstacles to the practical adaptation of foundation models for representation learning in information retrieval tasks.
To address these pressing issues, we propose organizing the first EReL@MIR workshop at the Web Conference 2025, inviting participants to explore novel solutions, emerging problems, challenges, efficiency evaluation metrics and benchmarks. This workshop aims to provide a platform for both academic and industry researchers to engage in discussions, share insights, and foster collaboration toward achieving efficient and effective representation learning for multimodal information retrieval in the era of large foundation models. 

---
# Matrix Factorization with Dynamic Multi-view Clustering for Recommender System 

**Authors**: Shangde Gao, Ke Liu, Yichao Fu, Hongxia Xu, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14565)  

**Abstract**: Matrix factorization (MF), a cornerstone of recommender systems, decomposes user-item interaction matrices into latent representations. Traditional MF approaches, however, employ a two-stage, non-end-to-end paradigm, sequentially performing recommendation and clustering, resulting in prohibitive computational costs for large-scale applications like e-commerce and IoT, where billions of users interact with trillions of items. To address this, we propose Matrix Factorization with Dynamic Multi-view Clustering (MFDMC), a unified framework that balances efficient end-to-end training with comprehensive utilization of web-scale data and enhances interpretability. MFDMC leverages dynamic multi-view clustering to learn user and item representations, adaptively pruning poorly formed clusters. Each entity's representation is modeled as a weighted projection of robust clusters, capturing its diverse roles across views. This design maximizes representation space utilization, improves interpretability, and ensures resilience for downstream tasks. Extensive experiments demonstrate MFDMC's superior performance in recommender systems and other representation learning domains, such as computer vision, highlighting its scalability and versatility. 

---
# Regret-aware Re-ranking for Guaranteeing Two-sided Fairness and Accuracy in Recommender Systems 

**Authors**: Xiaopeng Ye, Chen Xu, Jun Xu, Xuyang Xie, Gang Wang, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.14550)  

**Abstract**: In multi-stakeholder recommender systems (RS), users and providers operate as two crucial and interdependent roles, whose interests must be well-balanced. Prior research, including our work BankFair, has demonstrated the importance of guaranteeing both provider fairness and user accuracy to meet their interests. However, when they balance the two objectives, another critical factor emerges in RS: individual fairness, which manifests as a significant disparity in individual recommendation accuracy, with some users receiving high accuracy while others are left with notably low accuracy. This oversight severely harms the interests of users and exacerbates social polarization. How to guarantee individual fairness while ensuring user accuracy and provider fairness remains an unsolved problem. To bridge this gap, in this paper, we propose our method BankFair+. Specifically, BankFair+ extends BankFair with two steps: (1) introducing a non-linear function from regret theory to ensure individual fairness while enhancing user accuracy; (2) formulating the re-ranking process as a regret-aware fuzzy programming problem to meet the interests of both individual user and provider, therefore balancing the trade-off between individual fairness and provider fairness. Experiments on two real-world recommendation datasets demonstrate that BankFair+ outperforms all baselines regarding individual fairness, user accuracy, and provider fairness. 

---
# FinSage: A Multi-aspect RAG System for Financial Filings Question Answering 

**Authors**: Xinyu Wang, Jijun Chi, Zhenghan Tai, Tung Sum Thomas Kwok, Muzhi Li, Zhuhong Li, Hailin He, Yuchen Hua, Peng Lu, Suyuchen Wang, Yihong Wu, Jerry Huang, Ling Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.14493)  

**Abstract**: Leveraging large language models in real-world settings often entails a need to utilize domain-specific data and tools in order to follow the complex regulations that need to be followed for acceptable use. Within financial sectors, modern enterprises increasingly rely on Retrieval-Augmented Generation (RAG) systems to address complex compliance requirements in financial document workflows. However, existing solutions struggle to account for the inherent heterogeneity of data (e.g., text, tables, diagrams) and evolving nature of regulatory standards used in financial filings, leading to compromised accuracy in critical information extraction. We propose the FinSage framework as a solution, utilizing a multi-aspect RAG framework tailored for regulatory compliance analysis in multi-modal financial documents. FinSage introduces three innovative components: (1) a multi-modal pre-processing pipeline that unifies diverse data formats and generates chunk-level metadata summaries, (2) a multi-path sparse-dense retrieval system augmented with query expansion (HyDE) and metadata-aware semantic search, and (3) a domain-specialized re-ranking module fine-tuned via Direct Preference Optimization (DPO) to prioritize compliance-critical content. Extensive experiments demonstrate that FinSage achieves an impressive recall of 92.51% on 75 expert-curated questions derived from surpasses the best baseline method on the FinanceBench question answering datasets by 24.06% in accuracy. Moreover, FinSage has been successfully deployed as financial question-answering agent in online meetings, where it has already served more than 1,200 people. 

---
# LLM-Driven Usefulness Judgment for Web Search Evaluation 

**Authors**: Mouly Dewan, Jiqun Liu, Aditya Gautam, Chirag Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.14401)  

**Abstract**: Evaluation is fundamental in optimizing search experiences and supporting diverse user intents in Information Retrieval (IR). Traditional search evaluation methods primarily rely on relevance labels, which assess how well retrieved documents match a user's query. However, relevance alone fails to capture a search system's effectiveness in helping users achieve their search goals, making usefulness a critical evaluation criterion. In this paper, we explore an alternative approach: LLM-generated usefulness labels, which incorporate both implicit and explicit user behavior signals to evaluate document usefulness. We propose Task-aware Rubric-based Usefulness Evaluation (TRUE), a rubric-driven evaluation method that employs iterative sampling and reasoning to model complex search behavior patterns. Our findings show that (i) LLMs can generate moderate usefulness labels by leveraging comprehensive search session history incorporating personalization and contextual understanding, and (ii) fine-tuned LLMs improve usefulness judgments when provided with structured search session contexts. Additionally, we examine whether LLMs can distinguish between relevance and usefulness, particularly in cases where this divergence impacts search success. We also conduct an ablation study to identify key metrics for accurate usefulness label generation, optimizing for token efficiency and cost-effectiveness in real-world applications. This study advances LLM-based usefulness evaluation by refining key user metrics, exploring LLM-generated label reliability, and ensuring feasibility for large-scale search systems. 

---
# Unconstrained Monotonic Calibration of Predictions in Deep Ranking Systems 

**Authors**: Yimeng Bai, Shunyu Zhang, Yang Zhang, Hu Liu, Wentian Bao, Enyun Yu, Fuli Feng, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2504.14243)  

**Abstract**: Ranking models primarily focus on modeling the relative order of predictions while often neglecting the significance of the accuracy of their absolute values. However, accurate absolute values are essential for certain downstream tasks, necessitating the calibration of the original predictions. To address this, existing calibration approaches typically employ predefined transformation functions with order-preserving properties to adjust the original predictions. Unfortunately, these functions often adhere to fixed forms, such as piece-wise linear functions, which exhibit limited expressiveness and flexibility, thereby constraining their effectiveness in complex calibration scenarios. To mitigate this issue, we propose implementing a calibrator using an Unconstrained Monotonic Neural Network (UMNN), which can learn arbitrary monotonic functions with great modeling power. This approach significantly relaxes the constraints on the calibrator, improving its flexibility and expressiveness while avoiding excessively distorting the original predictions by requiring monotonicity. Furthermore, to optimize this highly flexible network for calibration, we introduce a novel additional loss function termed Smooth Calibration Loss (SCLoss), which aims to fulfill a necessary condition for achieving the ideal calibration state. Extensive offline experiments confirm the effectiveness of our method in achieving superior calibration performance. Moreover, deployment in Kuaishou's large-scale online video ranking system demonstrates that the method's calibration improvements translate into enhanced business metrics. The source code is available at this https URL. 

---
# Template-Based Financial Report Generation in Agentic and Decomposed Information Retrieval 

**Authors**: Yong-En Tian, Yu-Chien Tang, Kuang-Da Wang, An-Zi Yen, Wen-Chih Peng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14233)  

**Abstract**: Tailoring structured financial reports from companies' earnings releases is crucial for understanding financial performance and has been widely adopted in real-world analytics. However, existing summarization methods often generate broad, high-level summaries, which may lack the precision and detail required for financial reports that typically focus on specific, structured sections. While Large Language Models (LLMs) hold promise, generating reports adhering to predefined multi-section templates remains challenging. This paper investigates two LLM-based approaches popular in industry for generating templated financial reports: an agentic information retrieval (IR) framework and a decomposed IR approach, namely AgenticIR and DecomposedIR. The AgenticIR utilizes collaborative agents prompted with the full template. In contrast, the DecomposedIR approach applies a prompt chaining workflow to break down the template and reframe each section as a query answered by the LLM using the earnings release. To quantitatively assess the generated reports, we evaluated both methods in two scenarios: one using a financial dataset without direct human references, and another with a weather-domain dataset featuring expert-written reports. Experimental results show that while AgenticIR may excel in orchestrating tasks and generating concise reports through agent collaboration, DecomposedIR statistically significantly outperforms AgenticIR approach in providing broader and more detailed coverage in both scenarios, offering reflection on the utilization of the agentic framework in real-world applications. 

---
# Teach Me How to Denoise: A Universal Framework for Denoising Multi-modal Recommender Systems via Guided Calibration 

**Authors**: Hongji Li, Hanwen Du, Youhua Li, Junchen Fu, Chunxiao Li, Ziyi Zhuang, Jiakang Li, Yongxin Ni  

**Link**: [PDF](https://arxiv.org/pdf/2504.14214)  

**Abstract**: The surge in multimedia content has led to the development of Multi-Modal Recommender Systems (MMRecs), which use diverse modalities such as text, images, videos, and audio for more personalized recommendations. However, MMRecs struggle with noisy data caused by misalignment among modal content and the gap between modal semantics and recommendation semantics. Traditional denoising methods are inadequate due to the complexity of multi-modal data. To address this, we propose a universal guided in-sync distillation denoising framework for multi-modal recommendation (GUIDER), designed to improve MMRecs by denoising user feedback. Specifically, GUIDER uses a re-calibration strategy to identify clean and noisy interactions from modal content. It incorporates a Denoising Bayesian Personalized Ranking (DBPR) loss function to handle implicit user feedback. Finally, it applies a denoising knowledge distillation objective based on Optimal Transport distance to guide the alignment from modality representations to recommendation semantics. GUIDER can be seamlessly integrated into existing MMRecs methods as a plug-and-play solution. Experimental results on four public datasets demonstrate its effectiveness and generalizability. Our source code is available at this https URL 

---
# FedCIA: Federated Collaborative Information Aggregation for Privacy-Preserving Recommendation 

**Authors**: Mingzhe Han, Dongsheng Li, Jiafeng Xia, Jiahao Liu, Hansu Gu, Peng Zhang, Ning Gu, Tun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14208)  

**Abstract**: Recommendation algorithms rely on user historical interactions to deliver personalized suggestions, which raises significant privacy concerns. Federated recommendation algorithms tackle this issue by combining local model training with server-side model aggregation, where most existing algorithms use a uniform weighted summation to aggregate item embeddings from different client models. This approach has three major limitations: 1) information loss during aggregation, 2) failure to retain personalized local features, and 3) incompatibility with parameter-free recommendation algorithms. To address these limitations, we first review the development of recommendation algorithms and recognize that their core function is to share collaborative information, specifically the global relationship between users and items. With this understanding, we propose a novel aggregation paradigm named collaborative information aggregation, which focuses on sharing collaborative information rather than item parameters. Based on this new paradigm, we introduce the federated collaborative information aggregation (FedCIA) method for privacy-preserving recommendation. This method requires each client to upload item similarity matrices for aggregation, which allows clients to align their local models without constraining embeddings to a unified vector space. As a result, it mitigates information loss caused by direct summation, preserves the personalized embedding distributions of individual clients, and supports the aggregation of parameter-free models. Theoretical analysis and experimental results on real-world datasets demonstrate the superior performance of FedCIA compared with the state-of-the-art federated recommendation algorithms. Code is available at this https URL. 

---
# HF4Rec: Human-Like Feedback-Driven Optimization Framework for Explainable Recommendation 

**Authors**: Jiakai Tang, Jingsen Zhang, Zihang Tian, Xueyang Feng, Lei Wang, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14147)  

**Abstract**: Recent advancements in explainable recommendation have greatly bolstered user experience by elucidating the decision-making rationale. However, the existing methods actually fail to provide effective feedback signals for potentially better or worse generated explanations due to their reliance on traditional supervised learning paradigms in sparse interaction data. To address these issues, we propose a novel human-like feedback-driven optimization framework. This framework employs a dynamic interactive optimization mechanism for achieving human-centered explainable requirements without incurring high labor costs. Specifically, we propose to utilize large language models (LLMs) as human simulators to predict human-like feedback for guiding the learning process. To enable the LLMs to deeply understand the task essence and meet user's diverse personalized requirements, we introduce a human-induced customized reward scoring method, which helps stimulate the language understanding and logical reasoning capabilities of LLMs. Furthermore, considering the potential conflicts between different perspectives of explanation quality, we introduce a principled Pareto optimization that transforms the multi-perspective quality enhancement task into a multi-objective optimization problem for improving explanation performance. At last, to achieve efficient model training, we design an off-policy optimization pipeline. By incorporating a replay buffer and addressing the data distribution biases, we can effectively improve data utilization and enhance model generality. Extensive experiments on four datasets demonstrate the superiority of our approach. 

---
# Personalized News Recommendation with Multi-granularity Candidate-aware User Modeling 

**Authors**: Qiang Li, Xinze Lin, Shenghao Lv, Faliang Huang, Xiangju Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14130)  

**Abstract**: Matching candidate news with user interests is crucial for personalized news recommendations. Most existing methods can represent a user's reading interests through a single profile based on clicked news, which may not fully capture the diversity of user interests. Although some approaches incorporate candidate news or topic information, they remain insufficient because they neglect the multi-granularity relatedness between candidate news and user interests. To address this, this study proposed a multi-granularity candidate-aware user modeling framework that integrated user interest features across various levels of granularity. It consisted of two main components: candidate news encoding and user modeling. A news textual information extractor and a knowledge-enhanced entity information extractor can capture candidate news features, and word-level, entity-level, and news-level candidate-aware mechanisms can provide a comprehensive representation of user interests. Extensive experiments on a real-world dataset demonstrated that the proposed model could significantly outperform baseline models. 

---
# CPR: Leveraging LLMs for Topic and Phrase Suggestion to Facilitate Comprehensive Product Reviews 

**Authors**: Ekta Gujral, Apurva Sinha, Lishi Ji, Bijayani Sanghamitra Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2504.13993)  

**Abstract**: Consumers often heavily rely on online product reviews, analyzing both quantitative ratings and textual descriptions to assess product quality. However, existing research hasn't adequately addressed how to systematically encourage the creation of comprehensive reviews that capture both customers sentiment and detailed product feature analysis. This paper presents CPR, a novel methodology that leverages the power of Large Language Models (LLMs) and Topic Modeling to guide users in crafting insightful and well-rounded reviews. Our approach employs a three-stage process: first, we present users with product-specific terms for rating; second, we generate targeted phrase suggestions based on these ratings; and third, we integrate user-written text through topic modeling, ensuring all key aspects are addressed. We evaluate CPR using text-to-text LLMs, comparing its performance against real-world customer reviews from Walmart. Our results demonstrate that CPR effectively identifies relevant product terms, even for new products lacking prior reviews, and provides sentiment-aligned phrase suggestions, saving users time and enhancing reviews quality. Quantitative analysis reveals a 12.3% improvement in BLEU score over baseline methods, further supported by manual evaluation of generated phrases. We conclude by discussing potential extensions and future research directions. 

---
# Support Evaluation for the TREC 2024 RAG Track: Comparing Human versus LLM Judges 

**Authors**: Nandan Thakur, Ronak Pradeep, Shivani Upadhyay, Daniel Campos, Nick Craswell, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15205)  

**Abstract**: Retrieval-augmented generation (RAG) enables large language models (LLMs) to generate answers with citations from source documents containing "ground truth", thereby reducing system hallucinations. A crucial factor in RAG evaluation is "support", whether the information in the cited documents supports the answer. To this end, we conducted a large-scale comparative study of 45 participant submissions on 36 topics to the TREC 2024 RAG Track, comparing an automatic LLM judge (GPT-4o) against human judges for support assessment. We considered two conditions: (1) fully manual assessments from scratch and (2) manual assessments with post-editing of LLM predictions. Our results indicate that for 56% of the manual from-scratch assessments, human and GPT-4o predictions match perfectly (on a three-level scale), increasing to 72% in the manual with post-editing condition. Furthermore, by carefully analyzing the disagreements in an unbiased study, we found that an independent human judge correlates better with GPT-4o than a human judge, suggesting that LLM judges can be a reliable alternative for support assessment. To conclude, we provide a qualitative analysis of human and GPT-4o errors to help guide future iterations of support assessment. 

---
# Stitching Inner Product and Euclidean Metrics for Topology-aware Maximum Inner Product Search 

**Authors**: Tingyang Chen, Cong Fu, Xiangyu Ke, Yunjun Gao, Yabo Ni, Anxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14861)  

**Abstract**: Maximum Inner Product Search (MIPS) is a fundamental challenge in machine learning and information retrieval, particularly in high-dimensional data applications. Existing approaches to MIPS either rely solely on Inner Product (IP) similarity, which faces issues with local optima and redundant computations, or reduce the MIPS problem to the Nearest Neighbor Search under the Euclidean metric via space projection, leading to topology destruction and information loss. Despite the divergence of the two paradigms, we argue that there is no inherent binary opposition between IP and Euclidean metrics. By stitching IP and Euclidean in the design of indexing and search algorithms, we can significantly enhance MIPS performance. Specifically, this paper explores the theoretical and empirical connections between these two metrics from the MIPS perspective. Our investigation, grounded in graph-based search, reveals that different indexing and search strategies offer distinct advantages for MIPS, depending on the underlying data topology. Building on these insights, we introduce a novel graph-based index called Metric-Amphibious Graph (MAG) and a corresponding search algorithm, Adaptive Navigation with Metric Switch (ANMS). To facilitate parameter tuning for optimal performance, we identify three statistical indicators that capture essential data topology properties and correlate strongly with parameter tuning. Extensive experiments on 12 real-world datasets demonstrate that MAG outperforms existing state-of-the-art methods, achieving up to 4x search speedup while maintaining adaptability and scalability. 

---
# On Self-improving Token Embeddings 

**Authors**: Mario M. Kubek, Shiraj Pokharel, Thomas Böhme, Emma L. McDaniel, Herwig Unger, Armin R. Mikler  

**Link**: [PDF](https://arxiv.org/pdf/2504.14808)  

**Abstract**: This article introduces a novel and fast method for refining pre-trained static word or, more generally, token embeddings. By incorporating the embeddings of neighboring tokens in text corpora, it continuously updates the representation of each token, including those without pre-assigned embeddings. This approach effectively addresses the out-of-vocabulary problem, too. Operating independently of large language models and shallow neural networks, it enables versatile applications such as corpus exploration, conceptual search, and word sense disambiguation. The method is designed to enhance token representations within topically homogeneous corpora, where the vocabulary is restricted to a specific domain, resulting in more meaningful embeddings compared to general-purpose pre-trained vectors. As an example, the methodology is applied to explore storm events and their impacts on infrastructure and communities using narratives from a subset of the NOAA Storm Events database. The article also demonstrates how the approach improves the representation of storm-related terms over time, providing valuable insights into the evolving nature of disaster narratives. 

---
# Generative Auto-Bidding with Value-Guided Explorations 

**Authors**: Jingtong Gao, Yewen Li, Shuai Mao, Peng Jiang, Nan Jiang, Yejing Wang, Qingpeng Cai, Fei Pan, Peng Jiang, Kun Gai, Bo An, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.14587)  

**Abstract**: Auto-bidding, with its strong capability to optimize bidding decisions within dynamic and competitive online environments, has become a pivotal strategy for advertising platforms. Existing approaches typically employ rule-based strategies or Reinforcement Learning (RL) techniques. However, rule-based strategies lack the flexibility to adapt to time-varying market conditions, and RL-based methods struggle to capture essential historical dependencies and observations within Markov Decision Process (MDP) frameworks. Furthermore, these approaches often face challenges in ensuring strategy adaptability across diverse advertising objectives. Additionally, as offline training methods are increasingly adopted to facilitate the deployment and maintenance of stable online strategies, the issues of documented behavioral patterns and behavioral collapse resulting from training on fixed offline datasets become increasingly significant. To address these limitations, this paper introduces a novel offline Generative Auto-bidding framework with Value-Guided Explorations (GAVE). GAVE accommodates various advertising objectives through a score-based Return-To-Go (RTG) module. Moreover, GAVE integrates an action exploration mechanism with an RTG-based evaluation method to explore novel actions while ensuring stability-preserving updates. A learnable value function is also designed to guide the direction of action exploration and mitigate Out-of-Distribution (OOD) problems. Experimental results on two offline datasets and real-world deployments demonstrate that GAVE outperforms state-of-the-art baselines in both offline evaluations and online A/B tests. The implementation code is publicly available to facilitate reproducibility and further research. 

---
# EIoU-EMC: A Novel Loss for Domain-specific Nested Entity Recognition 

**Authors**: Jian Zhang, Tianqing Zhang, Qi Li, Hongwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14203)  

**Abstract**: In recent years, research has mainly focused on the general NER task. There still have some challenges with nested NER task in the specific domains. Specifically, the scenarios of low resource and class imbalance impede the wide application for biomedical and industrial domains. In this study, we design a novel loss EIoU-EMC, by enhancing the implement of Intersection over Union loss and Multiclass loss. Our proposed method specially leverages the information of entity boundary and entity classification, thereby enhancing the model's capacity to learn from a limited number of data samples. To validate the performance of this innovative method in enhancing NER task, we conducted experiments on three distinct biomedical NER datasets and one dataset constructed by ourselves from industrial complex equipment maintenance documents. Comparing to strong baselines, our method demonstrates the competitive performance across all datasets. During the experimental analysis, our proposed method exhibits significant advancements in entity boundary recognition and entity classification. Our code are available here. 

---
# Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion 

**Authors**: Yejun Yoon, Jaeyoon Jung, Seunghyun Yoon, Kunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.14175)  

**Abstract**: Query expansion methods powered by large language models (LLMs) have demonstrated effectiveness in zero-shot retrieval tasks. These methods assume that LLMs can generate hypothetical documents that, when incorporated into a query vector, enhance the retrieval of real evidence. However, we challenge this assumption by investigating whether knowledge leakage in benchmarks contributes to the observed performance gains. Using fact verification as a testbed, we analyzed whether the generated documents contained information entailed by ground truth evidence and assessed their impact on performance. Our findings indicate that performance improvements occurred consistently only for claims whose generated documents included sentences entailed by ground truth evidence. This suggests that knowledge leakage may be present in these benchmarks, inflating the perceived performance of LLM-based query expansion methods, particularly in real-world scenarios that require retrieving niche or novel knowledge. 

---
# Enhancing Math Learning in an LMS Using AI-Driven Question Recommendations 

**Authors**: Justus Råmunddal  

**Link**: [PDF](https://arxiv.org/pdf/2504.14098)  

**Abstract**: This paper presents an AI-driven approach to enhance math learning in a modern Learning Management System (LMS) by recommending similar math questions. Deep embeddings for math questions are generated using Meta's Llama-3.2-11B-Vision-Instruct model, and three recommendation methods-cosine similarity, Self-Organizing Maps (SOM), and Gaussian Mixture Models (GMM)-are applied to identify similar questions. User interaction data, including session durations, response times, and correctness, are used to evaluate the methods. Our findings suggest that while cosine similarity produces nearly identical question matches, SOM yields higher user satisfaction whereas GMM generally underperforms, indicating that introducing variety to a certain degree may enhance engagement and thereby potential learning outcomes until variety is no longer balanced reasonably, which our data about the implementations of all three methods demonstrate. 

---
