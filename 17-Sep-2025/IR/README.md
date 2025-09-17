# Efficient Cold-Start Recommendation via BPE Token-Level Embedding Initialization with LLM 

**Authors**: Yushang Zhao, Xinyue Han, Qian Leng, Qianyi Sun, Haotian Lyu, Chengrui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.13179)  

**Abstract**: The cold-start issue is the challenge when we talk about recommender systems, especially in the case when we do not have the past interaction data of new users or new items. Content-based features or hybrid solutions are common as conventional solutions, but they can only work in a sparse metadata environment with shallow patterns. In this paper, the efficient cold-start recommendation strategy is presented, which is based on the sub word-level representations by applying Byte Pair Encoding (BPE) tokenization and pre-trained Large Language Model (LLM) embedding in the initialization procedure. We obtain fine-grained token-level vectors that are aligned with the BPE vocabulary as opposed to using coarse-grained sentence embeddings. Together, these token embeddings can be used as dense semantic priors on unseen entities, making immediate recommendation performance possible without user-item interaction history. Our mechanism can be compared to collaborative filtering systems and tested over benchmark datasets with stringent cold-start assumptions. Experimental findings show that the given BPE-LLM method achieves higher Recall@k, NDCG@k, and Hit Rate measurements compared to the standard baseline and displays the same capability of sufficient computational performance. Furthermore, we demonstrate that using subword-aware embeddings yields better generalizability and is more interpretable, especially within a multilingual and sparse input setting. The practical application of token-level semantic initialization as a lightweight, but nevertheless effective extension to modern recommender systems in the zero-shot setting is indicated within this work. 

---
# Green Recommender Systems: Understanding and Minimizing the Carbon Footprint of AI-Powered Personalization 

**Authors**: Lukas Wegmeth, Tobias Vente, Alan Said, Joeran Beel  

**Link**: [PDF](https://arxiv.org/pdf/2509.13001)  

**Abstract**: As global warming soars, the need to assess and reduce the environmental impact of recommender systems is becoming increasingly urgent. Despite this, the recommender systems community hardly understands, addresses, and evaluates the environmental impact of their work. In this study, we examine the environmental impact of recommender systems research by reproducing typical experimental pipelines. Based on our results, we provide guidelines for researchers and practitioners on how to minimize the environmental footprint of their work and implement green recommender systems - recommender systems designed to minimize their energy consumption and carbon footprint. Our analysis covers 79 papers from the 2013 and 2023 ACM RecSys conferences, comparing traditional "good old-fashioned AI" models with modern deep learning models. We designed and reproduced representative experimental pipelines for both years, measuring energy consumption using a hardware energy meter and converting it into CO2 equivalents. Our results show that papers utilizing deep learning models emit approximately 42 times more CO2 equivalents than papers using traditional models. On average, a single deep learning-based paper generates 2,909 kilograms of CO2 equivalents - more than the carbon emissions of a person flying from New York City to Melbourne or the amount of CO2 sequestered by one tree over 260 years. This work underscores the urgent need for the recommender systems and wider machine learning communities to adopt green AI principles, balancing algorithmic advancements and environmental responsibility to build a sustainable future with AI-powered personalization. 

---
# Protecting participants or population? Comparison of k-anonymous Origin-Destination matrices 

**Authors**: Pietro Armenante, Kai Huang, Nikhil Jha, Luca Vassio  

**Link**: [PDF](https://arxiv.org/pdf/2509.12950)  

**Abstract**: Origin-Destination (OD) matrices are a core component of research on users' mobility and summarize how individuals move between geographical regions. These regions should be small enough to be representative of user mobility, without incurring substantial privacy risks. There are two added values of the NetMob2025 challenge dataset. Firstly, the data is extensive and contains a lot of socio-demographic information that can be used to create multiple OD matrices, based on the segments of the population. Secondly, a participant is not merely a record in the data, but a statistically weighted proxy for a segment of the real population. This opens the door to a fundamental shift in the anonymization paradigm. A population-based view of privacy is central to our contribution. By adjusting our anonymization framework to account for representativeness, we are also protecting the inferred identity of the actual population, rather than survey participants alone. The challenge addressed in this work is to produce and compare OD matrices that are k-anonymous for survey participants and for the whole population. We compare several traditional methods of anonymization to k-anonymity by generalizing geographical areas. These include generalization over a hierarchy (ATG and OIGH) and the classical Mondrian. To this established toolkit, we add a novel method, i.e., ODkAnon, a greedy algorithm aiming at balancing speed and quality. Unlike previous approaches, which primarily address the privacy aspects of the given datasets, we aim to contribute to the generation of privacy-preserving OD matrices enriched with socio-demographic segmentation that achieves k-anonymity on the actual population. 

---
# A Learnable Fully Interacted Two-Tower Model for Pre-Ranking System 

**Authors**: Chao Xiong, Xianwen Yu, Wei Xu, Lei Cheng, Chuan Yuan, Linjian Mo  

**Link**: [PDF](https://arxiv.org/pdf/2509.12948)  

**Abstract**: Pre-ranking plays a crucial role in large-scale recommender systems by significantly improving the efficiency and scalability within the constraints of providing high-quality candidate sets in real time. The two-tower model is widely used in pre-ranking systems due to a good balance between efficiency and effectiveness with decoupled architecture, which independently processes user and item inputs before calculating their interaction (e.g. dot product or similarity measure). However, this independence also leads to the lack of information interaction between the two towers, resulting in less effectiveness. In this paper, a novel architecture named learnable Fully Interacted Two-tower Model (FIT) is proposed, which enables rich information interactions while ensuring inference efficiency. FIT mainly consists of two parts: Meta Query Module (MQM) and Lightweight Similarity Scorer (LSS). Specifically, MQM introduces a learnable item meta matrix to achieve expressive early interaction between user and item features. Moreover, LSS is designed to further obtain effective late interaction between the user and item towers. Finally, experimental results on several public datasets show that our proposed FIT significantly outperforms the state-of-the-art baseline pre-ranking models. 

---
# DiffHash: Text-Guided Targeted Attack via Diffusion Models against Deep Hashing Image Retrieval 

**Authors**: Zechao Liu, Zheng Zhou, Xiangkun Chen, Tao Liang, Dapeng Lang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12824)  

**Abstract**: Deep hashing models have been widely adopted to tackle the challenges of large-scale image retrieval. However, these approaches face serious security risks due to their vulnerability to adversarial examples. Despite the increasing exploration of targeted attacks on deep hashing models, existing approaches still suffer from a lack of multimodal guidance, reliance on labeling information and dependence on pixel-level operations for attacks. To address these limitations, we proposed DiffHash, a novel diffusion-based targeted attack for deep hashing. Unlike traditional pixel-based attacks that directly modify specific pixels and lack multimodal guidance, our approach focuses on optimizing the latent representations of images, guided by text information generated by a Large Language Model (LLM) for the target image. Furthermore, we designed a multi-space hash alignment network to align the high-dimension image space and text space to the low-dimension binary hash space. During reconstruction, we also incorporated text-guided attention mechanisms to refine adversarial examples, ensuring them aligned with the target semantics while maintaining visual plausibility. Extensive experiments have demonstrated that our method outperforms state-of-the-art (SOTA) targeted attack methods, achieving better black-box transferability and offering more excellent stability across datasets. 

---
# InfoGain-RAG: Boosting Retrieval-Augmented Generation via Document Information Gain-based Reranking and Filtering 

**Authors**: Zihan Wang, Zihan Liang, Zhou Shao, Yufei Ma, Huangyu Dai, Ben Chen, Lingtao Mao, Chenyi Lei, Yuqing Ding, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.12765)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising approach to address key limitations of Large Language Models (LLMs), such as hallucination, outdated knowledge, and lacking reference. However, current RAG frameworks often struggle with identifying whether retrieved documents meaningfully contribute to answer generation. This shortcoming makes it difficult to filter out irrelevant or even misleading content, which notably impacts the final performance. In this paper, we propose Document Information Gain (DIG), a novel metric designed to quantify the contribution of retrieved documents to correct answer generation. DIG measures a document's value by computing the difference of LLM's generation confidence with and without the document augmented. Further, we introduce InfoGain-RAG, a framework that leverages DIG scores to train a specialized reranker, which prioritizes each retrieved document from exact distinguishing and accurate sorting perspectives. This approach can effectively filter out irrelevant documents and select the most valuable ones for better answer generation. Extensive experiments across various models and benchmarks demonstrate that InfoGain-RAG can significantly outperform existing approaches, on both single and multiple retrievers paradigm. Specifically on NaturalQA, it achieves the improvements of 17.9%, 4.5%, 12.5% in exact match accuracy against naive RAG, self-reflective RAG and modern ranking-based RAG respectively, and even an average of 15.3% increment on advanced proprietary model GPT-4o across all datasets. These results demonstrate the feasibility of InfoGain-RAG as it can offer a reliable solution for RAG in multiple applications. 

---
# LEAF: Knowledge Distillation of Text Embedding Models with Teacher-Aligned Representations 

**Authors**: Robin Vujanic, Thomas Rueckstiess  

**Link**: [PDF](https://arxiv.org/pdf/2509.12539)  

**Abstract**: We present LEAF ("Lightweight Embedding Alignment Framework"), a knowledge distillation framework for text embedding models. A key distinguishing feature is that our distilled leaf models are aligned to their teacher. In the context of information retrieval, this allows for flexible asymmetric architectures where documents are encoded with the larger teacher model, while queries can be served with the smaller leaf models. We also show that leaf models automatically inherit MRL and robustness to output quantization whenever these properties are present in the teacher model, without explicitly training for them. To demonstrate the capability of our framework we publish leaf-ir, a 23M parameters information retrieval oriented text embedding model trained using LEAF, which sets a new state-of-the-art (SOTA) on BEIR, ranking #1 on the public leaderboard for this benchmark and for models of its size. When run in asymmetric mode, its retrieval performance is further increased. Our scheme is however not restricted to the information retrieval setting, and we demonstrate its wider applicability by synthesizing the multi-task leaf-mt model. This also sets a new SOTA, ranking #1 on the public MTEB v2 (English) leaderboard for its size. LEAF is applicable to black-box models and in contrast to other embedding model training frameworks, it does not require judgments nor hard negatives, and training can be conducted using small batch sizes. Thus, dataset and training infrastructure requirements for our framework are modest. We make our models publicly available under a permissive Apache 2.0 license. 

---
# What News Recommendation Research Did (But Mostly Didn't) Teach Us About Building A News Recommender 

**Authors**: Karl Higley, Robin Burke, Michael D. Ekstrand, Bart P. Knijnenburg  

**Link**: [PDF](https://arxiv.org/pdf/2509.12361)  

**Abstract**: One of the goals of recommender systems research is to provide insights and methods that can be used by practitioners to build real-world systems that deliver high-quality recommendations to actual people grounded in their genuine interests and needs. We report on our experience trying to apply the news recommendation literature to build POPROX, a live platform for news recommendation research, and reflect on the extent to which the current state of research supports system-building efforts. Our experience highlights several unexpected challenges encountered in building personalization features that are commonly found in products from news aggregators and publishers, and shows how those difficulties are connected to surprising gaps in the literature. Finally, we offer a set of lessons learned from building a live system with a persistent user base and highlight opportunities to make future news recommendation research more applicable and impactful in practice. 

---
# Knowledge Graph Tokenization for Behavior-Aware Generative Next POI Recommendation 

**Authors**: Ke Sun, Mayi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.12350)  

**Abstract**: Generative paradigm, especially powered by Large Language Models (LLMs), has emerged as a new solution to the next point-of-interest (POI) recommendation. Pioneering studies usually adopt a two-stage pipeline, starting with a tokenizer converting POIs into discrete identifiers that can be processed by LLMs, followed by POI behavior prediction tasks to instruction-tune LLM for next POI recommendation. Despite of remarkable progress, they still face two limitations: (1) existing tokenizers struggle to encode heterogeneous signals in the recommendation data, suffering from information loss issue, and (2) previous instruction-tuning tasks only focus on users' POI visit behavior while ignore other behavior types, resulting in insufficient understanding of mobility. To address these limitations, we propose KGTB (Knowledge Graph Tokenization for Behavior-aware generative next POI recommendation). Specifically, KGTB organizes the recommendation data in a knowledge graph (KG) format, of which the structure can seamlessly preserve the heterogeneous information. Then, a KG-based tokenizer is developed to quantize each node into an individual structural ID. This process is supervised by the KG's structure, thus reducing the loss of heterogeneous information. Using generated IDs, KGTB proposes multi-behavior learning that introduces multiple behavior-specific prediction tasks for LLM fine-tuning, e.g., POI, category, and region visit behaviors. Learning on these behavior tasks provides LLMs with comprehensive insights on the target POI visit behavior. Experiments on four real-world city datasets demonstrate the superior performance of KGTB. 

---
# ResidualViT for Efficient Temporally Dense Video Encoding 

**Authors**: Mattia Soldan, Fabian Caba Heilbron, Bernard Ghanem, Josef Sivic, Bryan Russell  

**Link**: [PDF](https://arxiv.org/pdf/2509.13255)  

**Abstract**: Several video understanding tasks, such as natural language temporal video grounding, temporal activity localization, and audio description generation, require "temporally dense" reasoning over frames sampled at high temporal resolution. However, computing frame-level features for these tasks is computationally expensive given the temporal resolution requirements. In this paper, we make three contributions to reduce the cost of computing features for temporally dense tasks. First, we introduce a vision transformer (ViT) architecture, dubbed ResidualViT, that leverages the large temporal redundancy in videos to efficiently compute temporally dense frame-level features. Our architecture incorporates (i) learnable residual connections that ensure temporal consistency across consecutive frames and (ii) a token reduction module that enhances processing speed by selectively discarding temporally redundant information while reusing weights of a pretrained foundation model. Second, we propose a lightweight distillation strategy to approximate the frame-level features of the original foundation model. Finally, we evaluate our approach across four tasks and five datasets, in both zero-shot and fully supervised settings, demonstrating significant reductions in computational cost (up to 60%) and improvements in inference speed (up to 2.5x faster), all while closely approximating the accuracy of the original foundation model. 

---
# Automated Generation of Research Workflows from Academic Papers: A Full-text Mining Framework 

**Authors**: Heng Zhang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12955)  

**Abstract**: The automated generation of research workflows is essential for improving the reproducibility of research and accelerating the paradigm of "AI for Science". However, existing methods typically extract merely fragmented procedural components and thus fail to capture complete research workflows. To address this gap, we propose an end-to-end framework that generates comprehensive, structured research workflows by mining full-text academic papers. As a case study in the Natural Language Processing (NLP) domain, our paragraph-centric approach first employs Positive-Unlabeled (PU) Learning with SciBERT to identify workflow-descriptive paragraphs, achieving an F1-score of 0.9772. Subsequently, we utilize Flan-T5 with prompt learning to generate workflow phrases from these paragraphs, yielding ROUGE-1, ROUGE-2, and ROUGE-L scores of 0.4543, 0.2877, and 0.4427, respectively. These phrases are then systematically categorized into data preparation, data processing, and data analysis stages using ChatGPT with few-shot learning, achieving a classification precision of 0.958. By mapping categorized phrases to their document locations in the documents, we finally generate readable visual flowcharts of the entire research workflows. This approach facilitates the analysis of workflows derived from an NLP corpus and reveals key methodological shifts over the past two decades, including the increasing emphasis on data analysis and the transition from feature engineering to ablation studies. Our work offers a validated technical framework for automated workflow generation, along with a novel, process-oriented perspective for the empirical investigation of evolving scientific paradigms. Source code and data are available at: this https URL. 

---
# Timbre-Adaptive Transcription: A Lightweight Architecture with Associative Memory for Dynamic Instrument Separation 

**Authors**: Ruigang Li, Yongxu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.12712)  

**Abstract**: Existing multi-timbre transcription models struggle with generalization beyond pre-trained instruments and rigid source-count constraints. We address these limitations with a lightweight deep clustering solution featuring: 1) a timbre-agnostic backbone achieving state-of-the-art performance with only half the parameters of comparable models, and 2) a novel associative memory mechanism that mimics human auditory cognition to dynamically encode unseen timbres via attention-based clustering. Our biologically-inspired framework enables adaptive polyphonic separation with minimal training data (12.5 minutes), supported by a new synthetic dataset method offering cost-effective, high-precision multi-timbre generation. Experiments show the timbre-agnostic transcription model outperforms existing models on public benchmarks, while the separation module demonstrates promising timbre discrimination. This work provides an efficient framework for timbre-related music transcription and explores new directions for timbre-aware separation through cognitive-inspired architectures. 

---
# Digital Voices of Survival: From Social Media Disclosures to Support Provisions for Domestic Violence Victims 

**Authors**: Kanlun Wang, Zhe Fu, Wangjiaxuan Xin, Lina Zhou, Shashi Kiran Chandrappa  

**Link**: [PDF](https://arxiv.org/pdf/2509.12288)  

**Abstract**: Domestic Violence (DV) is a pervasive public health problem characterized by patterns of coercive and abusive behavior within intimate relationships. With the rise of social media as a key outlet for DV victims to disclose their experiences, online self-disclosure has emerged as a critical yet underexplored avenue for support-seeking. In addition, existing research lacks a comprehensive and nuanced understanding of DV self-disclosure, support provisions, and their connections. To address these gaps, this study proposes a novel computational framework for modeling DV support-seeking behavior alongside community support mechanisms. The framework consists of four key components: self-disclosure detection, post clustering, topic summarization, and support extraction and mapping. We implement and evaluate the framework with data collected from relevant social media communities. Our findings not only advance existing knowledge on DV self-disclosure and online support provisions but also enable victim-centered digital interventions. 

---
# Research on Short-Video Platform User Decision-Making via Multimodal Temporal Modeling and Reinforcement Learning 

**Authors**: Jinmeiyang Wang, Jing Dong, Li Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.12269)  

**Abstract**: This paper proposes the MT-DQN model, which integrates a Transformer, Temporal Graph Neural Network (TGNN), and Deep Q-Network (DQN) to address the challenges of predicting user behavior and optimizing recommendation strategies in short-video environments. Experiments demonstrated that MT-DQN consistently outperforms traditional concatenated models, such as Concat-Modal, achieving an average F1-score improvement of 10.97% and an average NDCG@5 improvement of 8.3%. Compared to the classic reinforcement learning model Vanilla-DQN, MT-DQN reduces MSE by 34.8% and MAE by 26.5%. Nonetheless, we also recognize challenges in deploying MT-DQN in real-world scenarios, such as its computational cost and latency sensitivity during online inference, which will be addressed through future architectural optimization. 

---
# Identifying Information Technology Research Trends through Text Mining of NSF Awards 

**Authors**: Said Varlioglu, Hazem Said, Murat Ozer, Nelly Elsayed  

**Link**: [PDF](https://arxiv.org/pdf/2509.12245)  

**Abstract**: Information Technology (IT) is recognized as an independent and unique research field. However, there has been ambiguity and difficulty in identifying and differentiating IT research from other close variations. Given this context, this paper aimed to explore the roots of the Information Technology (IT) research domain by conducting a large-scale text mining analysis of 50,780 abstracts from awarded NSF CISE grants from 1985 to 2024. We categorized the awards based on their program content, labeling human-centric programs as IT research programs and infrastructure-centric programs as other research programs based on the IT definitions in the literature. This novel approach helped us identify the core concepts of IT research and compare the similarities and differences between IT research and other research areas. The results showed that IT differentiates itself from other close variations by focusing more on the needs of users, organizations, and societies. 

---
