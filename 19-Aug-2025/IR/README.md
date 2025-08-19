# Is This News Still Interesting to You?: Lifetime-aware Interest Matching for News Recommendation 

**Authors**: Seongeun Ryu, Yunyong Ko, Sang-Wook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.13064)  

**Abstract**: Personalized news recommendation aims to deliver news articles aligned with users' interests, serving as a key solution to alleviate the problem of information overload on online news platforms. While prior work has improved interest matching through refined representations of news and users, the following time-related challenges remain underexplored: (C1) leveraging the age of clicked news to infer users' interest persistence, and (C2) modeling the varying lifetime of news across topics and users. To jointly address these challenges, we propose a novel Lifetime-aware Interest Matching framework for nEws recommendation, named LIME, which incorporates three key strategies: (1) User-Topic lifetime-aware age representation to capture the relative age of news with respect to a user-topic pair, (2) Candidate-aware lifetime attention for generating temporally aligned user representation, and (3) Freshness-guided interest refinement for prioritizing valid candidate news at prediction time. Extensive experiments on two real-world datasets demonstrate that LIME consistently outperforms a wide range of state-of-the-art news recommendation methods, and its model agnostic strategies significantly improve recommendation accuracy. 

---
# D-RDW: Diversity-Driven Random Walks for News Recommender Systems 

**Authors**: Runze Li, Lucien Heitz, Oana Inel, Abraham Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.13035)  

**Abstract**: This paper introduces Diversity-Driven RandomWalks (D-RDW), a lightweight algorithm and re-ranking technique that generates diverse news recommendations. D-RDW is a societal recommender, which combines the diversification capabilities of the traditional random walk algorithms with customizable target distributions of news article properties. In doing so, our model provides a transparent approach for editors to incorporate norms and values into the recommendation process. D-RDW shows enhanced performance across key diversity metrics that consider the articles' sentiment and political party mentions when compared to state-of-the-art neural models. Furthermore, D-RDW proves to be more computationally efficient than existing approaches. 

---
# Informfully Recommenders -- Reproducibility Framework for Diversity-aware Intra-session Recommendations 

**Authors**: Lucien Heitz, Runze Li, Oana Inel, Abraham Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.13019)  

**Abstract**: Norm-aware recommender systems have gained increased attention, especially for diversity optimization. The recommender systems community has well-established experimentation pipelines that support reproducible evaluations by facilitating models' benchmarking and comparisons against state-of-the-art methods. However, to the best of our knowledge, there is currently no reproducibility framework to support thorough norm-driven experimentation at the pre-processing, in-processing, post-processing, and evaluation stages of the recommender pipeline. To address this gap, we present Informfully Recommenders, a first step towards a normative reproducibility framework that focuses on diversity-aware design built on Cornac. Our extension provides an end-to-end solution for implementing and experimenting with normative and general-purpose diverse recommender systems that cover 1) dataset pre-processing, 2) diversity-optimized models, 3) dedicated intrasession item re-ranking, and 4) an extensive set of diversity metrics. We demonstrate the capabilities of our extension through an extensive offline experiment in the news domain. 

---
# Deep Research: A Survey of Autonomous Research Agents 

**Authors**: Wenlin Zhang, Xiaopeng Li, Yingyi Zhang, Pengyue Jia, Yichao Wang, Huifeng Guo, Yong Liu, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.12752)  

**Abstract**: The rapid advancement of large language models (LLMs) has driven the development of agentic systems capable of autonomously performing complex tasks. Despite their impressive capabilities, LLMs remain constrained by their internal knowledge boundaries. To overcome these limitations, the paradigm of deep research has been proposed, wherein agents actively engage in planning, retrieval, and synthesis to generate comprehensive and faithful analytical reports grounded in web-based evidence. In this survey, we provide a systematic overview of the deep research pipeline, which comprises four core stages: planning, question developing, web exploration, and report generation. For each stage, we analyze the key technical challenges and categorize representative methods developed to address them. Furthermore, we summarize recent advances in optimization techniques and benchmarks tailored for deep research. Finally, we discuss open challenges and promising research directions, aiming to chart a roadmap toward building more capable and trustworthy deep research agents. 

---
# Asymmetric Diffusion Recommendation Model 

**Authors**: Yongchun Zhu, Guanyu Jiang, Jingwu Chen, Feng Zhang, Xiao Yang, Zuotao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12706)  

**Abstract**: Recently, motivated by the outstanding achievements of diffusion models, the diffusion process has been employed to strengthen representation learning in recommendation systems. Most diffusion-based recommendation models typically utilize standard Gaussian noise in symmetric forward and reverse processes in continuous data space. Nevertheless, the samples derived from recommendation systems inhabit a discrete data space, which is fundamentally different from the continuous one. Moreover, Gaussian noise has the potential to corrupt personalized information within latent representations. In this work, we propose a novel and effective method, named Asymmetric Diffusion Recommendation Model (AsymDiffRec), which learns forward and reverse processes in an asymmetric manner. We define a generalized forward process that simulates the missing features in real-world recommendation samples. The reverse process is then performed in an asymmetric latent feature space. To preserve personalized information within the latent representation, a task-oriented optimization strategy is introduced. In the serving stage, the raw sample with missing features is regarded as a noisy input to generate a denoising and robust representation for the final prediction. By equipping base models with AsymDiffRec, we conduct online A/B tests, achieving improvements of +0.131% and +0.166% in terms of users' active days and app usage duration respectively. Additionally, the extended offline experiments also demonstrate improvements. AsymDiffRec has been implemented in the Douyin Music App. 

---
# Multi-Granularity Distribution Modeling for Video Watch Time Prediction via Exponential-Gaussian Mixture Network 

**Authors**: Xu Zhao, Ruibo Ma, Jiaqi Chen, Weiqi Zhao, Ping Yang, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12665)  

**Abstract**: Accurate watch time prediction is crucial for enhancing user engagement in streaming short-video platforms, although it is challenged by complex distribution characteristics across multi-granularity levels. Through systematic analysis of real-world industrial data, we uncover two critical challenges in watch time prediction from a distribution aspect: (1) coarse-grained skewness induced by a significant concentration of quick-skips1, (2) fine-grained diversity arising from various user-video interaction patterns. Consequently, we assume that the watch time follows the Exponential-Gaussian Mixture (EGM) distribution, where the exponential and Gaussian components respectively characterize the skewness and diversity. Accordingly, an Exponential-Gaussian Mixture Network (EGMN) is proposed for the parameterization of EGM distribution, which consists of two key modules: a hidden representation encoder and a mixture parameter generator. We conducted extensive offline experiments on public datasets and online A/B tests on the industrial short-video feeding scenario of Xiaohongshu App to validate the superiority of EGMN compared with existing state-of-the-art methods. Remarkably, comprehensive experimental results have proven that EGMN exhibits excellent distribution fitting ability across coarse-to-fine-grained levels. We open source related code on Github: this https URL. 

---
# Diagnostic-Guided Dynamic Profile Optimization for LLM-based User Simulators in Sequential Recommendation 

**Authors**: Hongyang Liu, Zhu Sun, Tianjun Wei, Yan Wang, Jiajie Zhu, Xinghua Qu  

**Link**: [PDF](https://arxiv.org/pdf/2508.12645)  

**Abstract**: Recent advances in large language models (LLMs) have enabled realistic user simulators for developing and evaluating recommender systems (RSs). However, existing LLM-based simulators for RSs face two major limitations: (1) static and single-step prompt-based inference that leads to inaccurate and incomplete user profile construction; (2) unrealistic and single-round recommendation-feedback interaction pattern that fails to capture real-world scenarios. To address these limitations, we propose DGDPO (Diagnostic-Guided Dynamic Profile Optimization), a novel framework that constructs user profile through a dynamic and iterative optimization process to enhance the simulation fidelity. Specifically, DGDPO incorporates two core modules within each optimization loop: firstly, a specialized LLM-based diagnostic module, calibrated through our novel training strategy, accurately identifies specific defects in the user profile. Subsequently, a generalized LLM-based treatment module analyzes the diagnosed defect and generates targeted suggestions to refine the profile. Furthermore, unlike existing LLM-based user simulators that are limited to single-round interactions, we are the first to integrate DGDPO with sequential recommenders, enabling a bidirectional evolution where user profiles and recommendation strategies adapt to each other over multi-round interactions. Extensive experiments conducted on three real-world datasets demonstrate the effectiveness of our proposed framework. 

---
# Contrastive Multi-View Graph Hashing 

**Authors**: Yang Xu, Zuliang Yang, Kai Ming Ting  

**Link**: [PDF](https://arxiv.org/pdf/2508.12377)  

**Abstract**: Multi-view graph data, which both captures node attributes and rich relational information from diverse sources, is becoming increasingly prevalent in various domains. The effective and efficient retrieval of such data is an important task. Although multi-view hashing techniques have offered a paradigm for fusing diverse information into compact binary codes, they typically assume attributes-based inputs per view. This makes them unsuitable for multi-view graph data, where effectively encoding and fusing complex topological information from multiple heterogeneous graph views to generate unified binary embeddings remains a significant challenge. In this work, we propose Contrastive Multi-view Graph Hashing (CMGHash), a novel end-to-end framework designed to learn unified and discriminative binary embeddings from multi-view graph data. CMGHash learns a consensus node representation space using a contrastive multi-view graph loss, which aims to pull $k$-nearest neighbors from all graphs closer while pushing away negative pairs, i.e., non-neighbor nodes. Moreover, we impose binarization constraints on this consensus space, enabling its conversion to a corresponding binary embedding space at minimal cost. Extensive experiments on several benchmark datasets demonstrate that CMGHash significantly outperforms existing approaches in terms of retrieval accuracy. 

---
# TaoSR1: The Thinking Model for E-commerce Relevance Search 

**Authors**: Chenhe Dong, Shaowei Yao, Pengkun Jiao, Jianhui Yang, Yiming Jin, Zerui Huang, Xiaojiang Zhou, Dan Ou, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12365)  

**Abstract**: Query-product relevance prediction is a core task in e-commerce search. BERT-based models excel at semantic matching but lack complex reasoning capabilities. While Large Language Models (LLMs) are explored, most still use discriminative fine-tuning or distill to smaller models for deployment. We propose a framework to directly deploy LLMs for this task, addressing key challenges: Chain-of-Thought (CoT) error accumulation, discriminative hallucination, and deployment feasibility. Our framework, TaoSR1, involves three stages: (1) Supervised Fine-Tuning (SFT) with CoT to instill reasoning; (2) Offline sampling with a pass@N strategy and Direct Preference Optimization (DPO) to improve generation quality; and (3) Difficulty-based dynamic sampling with Group Relative Policy Optimization (GRPO) to mitigate discriminative hallucination. Additionally, post-CoT processing and a cumulative probability-based partitioning method enable efficient online deployment. TaoSR1 significantly outperforms baselines on offline datasets and achieves substantial gains in online side-by-side human evaluations, introducing a novel paradigm for applying CoT reasoning to relevance classification. 

---
# A Large-Scale Web Search Dataset for Federated Online Learning to Rank 

**Authors**: Marcel Gregoriadis, Jingwei Kang, Johan Pouwelse  

**Link**: [PDF](https://arxiv.org/pdf/2508.12353)  

**Abstract**: The centralized collection of search interaction logs for training ranking models raises significant privacy concerns. Federated Online Learning to Rank (FOLTR) offers a privacy-preserving alternative by enabling collaborative model training without sharing raw user data. However, benchmarks in FOLTR are largely based on random partitioning of classical learning-to-rank datasets, simulated user clicks, and the assumption of synchronous client participation. This oversimplifies real-world dynamics and undermines the realism of experimental results. We present AOL4FOLTR, a large-scale web search dataset with 2.6 million queries from 10,000 users. Our dataset addresses key limitations of existing benchmarks by including user identifiers, real click data, and query timestamps, enabling realistic user partitioning, behavior modeling, and asynchronous federated learning scenarios. 

---
# Leveraging Geometric Insights in Hyperbolic Triplet Loss for Improved Recommendations 

**Authors**: Viacheslav Yusupov, Maxim Rakhuba, Evgeny Frolov  

**Link**: [PDF](https://arxiv.org/pdf/2508.11978)  

**Abstract**: Recent studies have demonstrated the potential of hyperbolic geometry for capturing complex patterns from interaction data in recommender systems. In this work, we introduce a novel hyperbolic recommendation model that uses geometrical insights to improve representation learning and increase computational stability at the same time. We reformulate the notion of hyperbolic distances to unlock additional representation capacity over conventional Euclidean space and learn more expressive user and item representations. To better capture user-items interactions, we construct a triplet loss that models ternary relations between users and their corresponding preferred and nonpreferred choices through a mix of pairwise interaction terms driven by the geometry of data. Our hyperbolic approach not only outperforms existing Euclidean and hyperbolic models but also reduces popularity bias, leading to more diverse and personalized recommendations. 

---
# TBGRecall: A Generative Retrieval Model for E-commerce Recommendation Scenarios 

**Authors**: Zida Liang, Changfa Wu, Dunxian Huang, Weiqiang Sun, Ziyang Wang, Yuliang Yan, Jian Wu, Yuning Jiang, Bo Zheng, Ke Chen, Silu Zhou, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11977)  

**Abstract**: Recommendation systems are essential tools in modern e-commerce, facilitating personalized user experiences by suggesting relevant products. Recent advancements in generative models have demonstrated potential in enhancing recommendation systems; however, these models often exhibit limitations in optimizing retrieval tasks, primarily due to their reliance on autoregressive generation mechanisms. Conventional approaches introduce sequential dependencies that impede efficient retrieval, as they are inherently unsuitable for generating multiple items without positional constraints within a single request session. To address these limitations, we propose TBGRecall, a framework integrating Next Session Prediction (NSP), designed to enhance generative retrieval models for e-commerce applications. Our framework reformulation involves partitioning input samples into multi-session sequences, where each sequence comprises a session token followed by a set of item tokens, and then further incorporate multiple optimizations tailored to the generative task in retrieval scenarios. In terms of training methodology, our pipeline integrates limited historical data pre-training with stochastic partial incremental training, significantly improving training efficiency and emphasizing the superiority of data recency over sheer data volume. Our extensive experiments, conducted on public benchmarks alongside a large-scale industrial dataset from TaoBao, show TBGRecall outperforms the state-of-the-art recommendation methods, and exhibits a clear scaling law trend. Ultimately, NSP represents a significant advancement in the effectiveness of generative recommendation systems for e-commerce applications. 

---
# Ontology-Guided Query Expansion for Biomedical Document Retrieval using Large Language Models 

**Authors**: Zabir Al Nazi, Vagelis Hristidis, Aaron Lawson McLean, Jannat Ara Meem, Md Taukir Azam Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2508.11784)  

**Abstract**: Effective Question Answering (QA) on large biomedical document collections requires effective document retrieval techniques. The latter remains a challenging task due to the domain-specific vocabulary and semantic ambiguity in user queries. We propose BMQExpander, a novel ontology-aware query expansion pipeline that combines medical knowledge - definitions and relationships - from the UMLS Metathesaurus with the generative capabilities of large language models (LLMs) to enhance retrieval effectiveness. We implemented several state-of-the-art baselines, including sparse and dense retrievers, query expansion methods, and biomedical-specific solutions. We show that BMQExpander has superior retrieval performance on three popular biomedical Information Retrieval (IR) benchmarks: NFCorpus, TREC-COVID, and SciFact - with improvements of up to 22.1% in NDCG@10 over sparse baselines and up to 6.5% over the strongest baseline. Further, BMQExpander generalizes robustly under query perturbation settings, in contrast to supervised baselines, achieving up to 15.7% improvement over the strongest baseline. As a side contribution, we publish our paraphrased benchmarks. Finally, our qualitative analysis shows that BMQExpander has fewer hallucinations compared to other LLM-based query expansion baselines. 

---
# LLM-Based Intelligent Agents for Music Recommendation: A Comparison with Classical Content-Based Filtering 

**Authors**: Ronald Carvalho Boadana, Ademir Guimarães da Costa Junior, Ricardo Rios, Fábio Santos da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2508.11671)  

**Abstract**: The growing availability of music on streaming platforms has led to information overload for users. To address this issue and enhance the user experience, increasingly sophisticated recommendation systems have been proposed. This work investigates the use of Large Language Models (LLMs) from the Gemini and LLaMA families, combined with intelligent agents, in a multi-agent personalized music recommendation system. The results are compared with a traditional content-based recommendation model, considering user satisfaction, novelty, and computational efficiency. LLMs achieved satisfaction rates of up to \textit{89{,}32\%}, indicating their promising potential in music recommendation systems. 

---
# RRRA: Resampling and Reranking through a Retriever Adapter 

**Authors**: Bongsu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.11670)  

**Abstract**: In dense retrieval, effective training hinges on selecting high quality hard negatives while avoiding false negatives. Recent methods apply heuristics based on positive document scores to identify hard negatives, improving both performance and interpretability. However, these global, example agnostic strategies often miss instance specific false negatives. To address this, we propose a learnable adapter module that monitors Bi-Encoder representations to estimate the likelihood that a hard negative is actually a false negative. This probability is modeled dynamically and contextually, enabling fine-grained, query specific judgments. The predicted scores are used in two downstream components: (1) resampling, where negatives are reweighted during training, and (2) reranking, where top-k retrieved documents are reordered at inference. Empirical results on standard benchmarks show that our adapter-enhanced framework consistently outperforms strong Bi-Encoder baselines, underscoring the benefit of explicit false negative modeling in dense retrieval. 

---
# All for law and law for all: Adaptive RAG Pipeline for Legal Research 

**Authors**: Figarri Keisha, Prince Singh, Pallavi, Dion Fernandes, Aravindh Manivannan, Ilham Wicaksono, Faisal Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2508.13107)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucinations by grounding large language model outputs in cited sources, a capability that is especially critical in the legal domain. We present an end-to-end RAG pipeline that revisits and extends the LegalBenchRAG baseline with three targeted enhancements: (i) a context-aware query translator that disentangles document references from natural-language questions and adapts retrieval depth and response style based on expertise and specificity, (ii) open-source retrieval strategies using SBERT and GTE embeddings that achieve substantial performance gains (improving Recall@K by 30-95\% and Precision@K by $\sim$2.5$\times$ for $K>4$) while remaining cost-efficient, and (iii) a comprehensive evaluation and generation framework that combines RAGAS, BERTScore-F1, and ROUGE-Recall to assess semantic alignment and faithfulness across models and prompt designs. Our results show that carefully designed open-source pipelines can rival or outperform proprietary approaches in retrieval quality, while a custom legal-grounded prompt consistently produces more faithful and contextually relevant answers than baseline prompting. Taken together, these contributions demonstrate the potential of task-aware, component-level tuning to deliver legally grounded, reproducible, and cost-effective RAG systems for legal research assistance. 

---
# jXBW: Fast Substructure Search in Large-Scale JSONL Datasets for Foundation Model Applications 

**Authors**: Yasuo Tabei  

**Link**: [PDF](https://arxiv.org/pdf/2508.12536)  

**Abstract**: Substructure search in JSON Lines (JSONL) datasets is essential for modern applications such as prompt engineering in foundation models, but existing methods suffer from prohibitive computational costs due to exhaustive tree traversal and subtree matching. We present jXBW, a fast method for substructure search on large-scale JSONL datasets. Our method makes three key technical contributions: (i) a merged tree representation built by merging trees of multiple JSON objects while preserving individual identities, (ii) a succinct data structure based on the eXtended Burrows-Wheeler Transform that enables efficient tree navigation and subpath search, and (iii) an efficient three-step substructure search algorithm that combines path decomposition, ancestor computation, and adaptive tree identifier collection to ensure correctness while avoiding exhaustive tree traversal. Experimental evaluation on real-world datasets demonstrates that jXBW consistently outperforms existing methods, achieving speedups of 16$\times$ for smaller datasets and up to 4,700$\times$ for larger datasets over tree-based approaches, and more than 6$\times$10$^6$ over XML-based processing while maintaining competitive memory usage. 

---
# A Question Answering Dataset for Temporal-Sensitive Retrieval-Augmented Generation 

**Authors**: Ziyang Chen, Erxue Min, Xiang Zhao, Yunxin Li, Xin Jia, Jinzhi Liao, Jichao Li, Shuaiqiang Wang, Baotian Hu, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.12282)  

**Abstract**: We introduce ChronoQA, a large-scale benchmark dataset for Chinese question answering, specifically designed to evaluate temporal reasoning in Retrieval-Augmented Generation (RAG) systems. ChronoQA is constructed from over 300,000 news articles published between 2019 and 2024, and contains 5,176 high-quality questions covering absolute, aggregate, and relative temporal types with both explicit and implicit time expressions. The dataset supports both single- and multi-document scenarios, reflecting the real-world requirements for temporal alignment and logical consistency. ChronoQA features comprehensive structural annotations and has undergone multi-stage validation, including rule-based, LLM-based, and human evaluation, to ensure data quality. By providing a dynamic, reliable, and scalable resource, ChronoQA enables structured evaluation across a wide range of temporal tasks, and serves as a robust benchmark for advancing time-sensitive retrieval-augmented question answering systems. 

---
# MOON: Generative MLLM-based Multimodal Representation Learning for E-commerce Product Understanding 

**Authors**: Daoze Zhang, Zhanheng Nie, Jianyu Liu, Chenghan Fu, Wanxian Guan, Yuan Gao, Jun Song, Pengjie Wang, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.11999)  

**Abstract**: With the rapid advancement of e-commerce, exploring general representations rather than task-specific ones has attracted increasing research attention. For product understanding, although existing discriminative dual-flow architectures drive progress in this field, they inherently struggle to model the many-to-one alignment between multiple images and texts of products. Therefore, we argue that generative Multimodal Large Language Models (MLLMs) hold significant potential for improving product representation learning. Nevertheless, achieving this goal still remains non-trivial due to several key challenges: the lack of multimodal and aspect-aware modeling modules in typical LLMs; the common presence of background noise in product images; and the absence of a standard benchmark for evaluation. To address these issues, we propose the first generative MLLM-based model named MOON for product representation learning. Our method (1) employs a guided Mixture-of-Experts (MoE) module for targeted modeling of multimodal and aspect-specific product content; (2) effectively detects core semantic regions in product images to mitigate the distraction and interference caused by background noise; and (3) introduces the specialized negative sampling strategy to increase the difficulty and diversity of negative samples. In addition, we release a large-scale multimodal benchmark MBE for various product understanding tasks. Experimentally, our model demonstrates competitive zero-shot performance on both our benchmark and the public dataset, showcasing strong generalization across various downstream tasks, including cross-modal retrieval, product classification, and attribute prediction. Furthermore, the case study and visualization illustrate the effectiveness of MOON for product understanding. 

---
# What Matters for Bioacoustic Encoding 

**Authors**: Marius Miron, David Robinson, Milad Alizadeh, Ellen Gilsenan-McMahon, Gagan Narula, Olivier Pietquin, Matthieu Geist, Emmanuel Chemla, Maddie Cusimano, Felix Effenberger, Masato Hagiwara, Benjamin Hoffman, Sara Keen, Diane Kim, Jane Lawton, Jen-Yu Liu, Aza Raskin  

**Link**: [PDF](https://arxiv.org/pdf/2508.11845)  

**Abstract**: Bioacoustics, the study of sounds produced by living organisms, plays a vital role in conservation, biodiversity monitoring, and behavioral studies. Many tasks in this field, such as species, individual, and behavior classification and detection, are well-suited to machine learning. However, they often suffer from limited annotated data, highlighting the need for a general-purpose bioacoustic encoder capable of extracting useful representations for diverse downstream tasks. Such encoders have been proposed before, but are often limited in scope due to a focus on a narrow range of species (typically birds), and a reliance on a single model architecture or training paradigm. Moreover, they are usually evaluated on a small set of tasks and datasets. In this work, we present a large-scale empirical study that covers aspects of bioacoustics that are relevant to research but have previously been scarcely considered: training data diversity and scale, model architectures and training recipes, and the breadth of evaluation tasks and datasets. We obtain encoders that are state-of-the-art on the existing and proposed benchmarks. We also identify what matters for training these encoders, such that this work can be extended when more data are available or better architectures are proposed. Specifically, across 26 datasets with tasks including species classification, detection, individual ID, and vocal repertoire discovery, we find self-supervised pre-training followed by supervised post-training on a mixed bioacoustics + general-audio corpus yields the strongest in- and out-of-distribution performance. We show the importance of data diversity in both stages. To support ongoing research and application, we will release the model checkpoints. 

---
