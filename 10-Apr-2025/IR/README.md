# CHIME: A Compressive Framework for Holistic Interest Modeling 

**Authors**: Yong Bai, Rui Xiang, Kaiyuan Li, Yongxiang Tang, Yanhua Cheng, Xialong Liu, Peng Jiang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2504.06780)  

**Abstract**: Modeling holistic user interests is important for improving recommendation systems but is challenged by high computational cost and difficulty in handling diverse information with full behavior context. Existing search-based methods might lose critical signals during behavior selection. To overcome these limitations, we propose CHIME: A Compressive Framework for Holistic Interest Modeling. It uses adapted large language models to encode complete user behaviors with heterogeneous inputs. We introduce multi-granular contrastive learning objectives to capture both persistent and transient interest patterns and apply residual vector quantization to generate compact embeddings. CHIME demonstrates superior ranking performance across diverse datasets, establishing a robust solution for scalable holistic interest modeling in recommendation systems. 

---
# Unifying Search and Recommendation: A Generative Paradigm Inspired by Information Theory 

**Authors**: Jujia Zhao, Wenjie Wang, Chen Xu, Xiuying Wang, Zhaochun Ren, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2504.06714)  

**Abstract**: Recommender systems and search engines serve as foundational elements of online platforms, with the former delivering information proactively and the latter enabling users to seek information actively. Unifying both tasks in a shared model is promising since it can enhance user modeling and item understanding. Previous approaches mainly follow a discriminative paradigm, utilizing shared encoders to process input features and task-specific heads to perform each task. However, this paradigm encounters two key challenges: gradient conflict and manual design complexity. From the information theory perspective, these challenges potentially both stem from the same issue -- low mutual information between the input features and task-specific outputs during the optimization process.
To tackle these issues, we propose GenSR, a novel generative paradigm for unifying search and recommendation (S&R), which leverages task-specific prompts to partition the model's parameter space into subspaces, thereby enhancing mutual information. To construct effective subspaces for each task, GenSR first prepares informative representations for each subspace and then optimizes both subspaces in one unified model. Specifically, GenSR consists of two main modules: (1) Dual Representation Learning, which independently models collaborative and semantic historical information to derive expressive item representations; and (2) S&R Task Unifying, which utilizes contrastive learning together with instruction tuning to generate task-specific outputs effectively. Extensive experiments on two public datasets show GenSR outperforms state-of-the-art methods across S&R tasks. Our work introduces a new generative paradigm compared with previous discriminative methods and establishes its superiority from the mutual information perspective. 

---
# Toward Holistic Evaluation of Recommender Systems Powered by Generative Models 

**Authors**: Yashar Deldjoo, Nikhil Mehta, Maheswaran Sathiamoorthy, Shuai Zhang, Pablo Castells, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2504.06667)  

**Abstract**: Recommender systems powered by generative models (Gen-RecSys) extend beyond classical item ranking by producing open-ended content, which simultaneously unlocks richer user experiences and introduces new risks. On one hand, these systems can enhance personalization and appeal through dynamic explanations and multi-turn dialogues. On the other hand, they might venture into unknown territory-hallucinating nonexistent items, amplifying bias, or leaking private information. Traditional accuracy metrics cannot fully capture these challenges, as they fail to measure factual correctness, content safety, or alignment with user intent.
This paper makes two main contributions. First, we categorize the evaluation challenges of Gen-RecSys into two groups: (i) existing concerns that are exacerbated by generative outputs (e.g., bias, privacy) and (ii) entirely new risks (e.g., item hallucinations, contradictory explanations). Second, we propose a holistic evaluation approach that includes scenario-based assessments and multi-metric checks-incorporating relevance, factual grounding, bias detection, and policy compliance. Our goal is to provide a guiding framework so researchers and practitioners can thoroughly assess Gen-RecSys, ensuring effective personalization and responsible deployment. 

---
# BBQRec: Behavior-Bind Quantization for Multi-Modal Sequential Recommendation 

**Authors**: Kaiyuan Li, Rui Xiang, Yong Bai, Yongxiang Tang, Yanhua Cheng, Xialong Liu, Peng Jiang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2504.06636)  

**Abstract**: Multi-modal sequential recommendation systems leverage auxiliary signals (e.g., text, images) to alleviate data sparsity in user-item interactions. While recent methods exploit large language models to encode modalities into discrete semantic IDs for autoregressive prediction, we identify two critical limitations: (1) Existing approaches adopt fragmented quantization, where modalities are independently mapped to semantic spaces misaligned with behavioral objectives, and (2) Over-reliance on semantic IDs disrupts inter-modal semantic coherence, thereby weakening the expressive power of multi-modal representations for modeling diverse user preferences.
To address these challenges, we propose a Behavior-Bind multi-modal Quantization for Sequential Recommendation (BBQRec for short) featuring dual-aligned quantization and semantics-aware sequence modeling. First, our behavior-semantic alignment module disentangles modality-agnostic behavioral patterns from noisy modality-specific features through contrastive codebook learning, ensuring semantic IDs are inherently tied to recommendation tasks. Second, we design a discretized similarity reweighting mechanism that dynamically adjusts self-attention scores using quantized semantic relationships, preserving multi-modal synergies while avoiding invasive modifications to the sequence modeling architecture. Extensive evaluations across four real-world benchmarks demonstrate BBQRec's superiority over the state-of-the-art baselines. 

---
# A Serendipitous Recommendation System Considering User Curiosity 

**Authors**: Zhelin Xu, Atsushi Matsumura  

**Link**: [PDF](https://arxiv.org/pdf/2504.06633)  

**Abstract**: To address the problem of narrow recommendation ranges caused by an emphasis on prediction accuracy, serendipitous recommendations, which consider both usefulness and unexpectedness, have attracted attention. However, realizing serendipitous recommendations is challenging due to the varying proportions of usefulness and unexpectedness preferred by different users, which is influenced by their differing desires for knowledge. In this paper, we propose a method to estimate the proportion of usefulness and unexpectedness that each user desires based on their curiosity, and make recommendations that match this preference. The proposed method estimates a user's curiosity by considering both their long-term and short-term interests. Offline experiments were conducted using the MovieLens-1M dataset to evaluate the effectiveness of the proposed method. The experimental results demonstrate that our method achieves the same level of performance as state-of-the-art method while successfully providing serendipitous recommendations. 

---
# InteractRank: Personalized Web-Scale Search Pre-Ranking with Cross Interaction Features 

**Authors**: Sujay Khandagale, Bhawna Juneja, Prabhat Agarwal, Aditya Subramanian, Jaewon Yang, Yuting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06609)  

**Abstract**: Modern search systems use a multi-stage architecture to deliver personalized results efficiently. Key stages include retrieval, pre-ranking, full ranking, and blending, which refine billions of items to top selections. The pre-ranking stage, vital for scoring and filtering hundreds of thousands of items down to a few thousand, typically relies on two tower models due to their computational efficiency, despite often lacking in capturing complex interactions. While query-item cross interaction features are paramount for full ranking, integrating them into pre-ranking models presents efficiency-related challenges. In this paper, we introduce InteractRank, a novel two tower pre-ranking model with robust cross interaction features used at Pinterest. By incorporating historical user engagement-based query-item interactions in the scoring function along with the two tower dot product, InteractRank significantly boosts pre-ranking performance with minimal latency and computation costs. In real-world A/B experiments at Pinterest, InteractRank improves the online engagement metric by 6.5% over a BM25 baseline and by 3.7% over a vanilla two tower baseline. We also highlight other components of InteractRank, like real-time user-sequence modeling, and analyze their contributions through offline ablation studies. The code for InteractRank is available at this https URL. 

---
# Diversity-aware Dual-promotion Poisoning Attack on Sequential Recommendation 

**Authors**: Yuchuan Zhao, Tong Chen, Junliang Yu, Kai Zheng, Lizhen Cui, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.06586)  

**Abstract**: Sequential recommender systems (SRSs) excel in capturing users' dynamic interests, thus playing a key role in various industrial applications. The popularity of SRSs has also driven emerging research on their security aspects, where data poisoning attack for targeted item promotion is a typical example. Existing attack mechanisms primarily focus on increasing the ranks of target items in the recommendation list by injecting carefully crafted interactions (i.e., poisoning sequences), which comes at the cost of demoting users' real preferences. Consequently, noticeable recommendation accuracy drops are observed, restricting the stealthiness of the attack. Additionally, the generated poisoning sequences are prone to substantial repetition of target items, which is a result of the unitary objective of boosting their overall exposure and lack of effective diversity regularizations. Such homogeneity not only compromises the authenticity of these sequences, but also limits the attack effectiveness, as it ignores the opportunity to establish sequential dependencies between the target and many more items in the SRS. To address the issues outlined, we propose a Diversity-aware Dual-promotion Sequential Poisoning attack method named DDSP for SRSs. Specifically, by theoretically revealing the conflict between recommendation and existing attack objectives, we design a revamped attack objective that promotes the target item while maintaining the relevance of preferred items in a user's ranking list. We further develop a diversity-aware, auto-regressive poisoning sequence generator, where a re-ranking method is in place to sequentially pick the optimal items by integrating diversity constraints. 

---
# Bridging Queries and Tables through Entities in Table Retrieval 

**Authors**: Da Li, Keping Bi, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.06551)  

**Abstract**: Table retrieval is essential for accessing information stored in structured tabular formats; however, it remains less explored than text retrieval. The content of the table primarily consists of phrases and words, which include a large number of entities, such as time, locations, persons, and organizations. Entities are well-studied in the context of text retrieval, but there is a noticeable lack of research on their applications in table retrieval. In this work, we explore how to leverage entities in tables to improve retrieval performance. First, we investigate the important role of entities in table retrieval from a statistical perspective and propose an entity-enhanced training framework. Subsequently, we use the type of entities to highlight entities instead of introducing an external knowledge base. Moreover, we design an interaction paradigm based on entity representations. Our proposed framework is plug-and-play and flexible, making it easy to integrate into existing table retriever training processes. Empirical results on two table retrieval benchmarks, NQ-TABLES and OTT-QA, show that our proposed framework is both simple and effective in enhancing existing retrievers. We also conduct extensive analyses to confirm the efficacy of different components. Overall, our work provides a promising direction for elevating table retrieval, enlightening future research in this area. 

---
# DiffusionCom: Structure-Aware Multimodal Diffusion Model for Multimodal Knowledge Graph Completion 

**Authors**: Wei Huang, Meiyu Liang, Peining Li, Xu Hou, Yawen Li, Junping Du, Zhe Xue, Zeli Guan  

**Link**: [PDF](https://arxiv.org/pdf/2504.06543)  

**Abstract**: Most current MKGC approaches are predominantly based on discriminative models that maximize conditional likelihood. These approaches struggle to efficiently capture the complex connections in real-world knowledge graphs, thereby limiting their overall performance. To address this issue, we propose a structure-aware multimodal Diffusion model for multimodal knowledge graph Completion (DiffusionCom). DiffusionCom innovatively approaches the problem from the perspective of generative models, modeling the association between the $(head, relation)$ pair and candidate tail entities as their joint probability distribution $p((head, relation), (tail))$, and framing the MKGC task as a process of gradually generating the joint probability distribution from noise. Furthermore, to fully leverage the structural information in MKGs, we propose Structure-MKGformer, an adaptive and structure-aware multimodal knowledge representation learning method, as the encoder for DiffusionCom. Structure-MKGformer captures rich structural information through a multimodal graph attention network (MGAT) and adaptively fuses it with entity representations, thereby enhancing the structural awareness of these representations. This design effectively addresses the limitations of existing MKGC methods, particularly those based on multimodal pre-trained models, in utilizing structural information. DiffusionCom is trained using both generative and discriminative losses for the generator, while the feature extractor is optimized exclusively with discriminative loss. This dual approach allows DiffusionCom to harness the strengths of both generative and discriminative models. Extensive experiments on the FB15k-237-IMG and WN18-IMG datasets demonstrate that DiffusionCom outperforms state-of-the-art models. 

---
# Dynamic Evaluation Framework for Personalized and Trustworthy Agents: A Multi-Session Approach to Preference Adaptability 

**Authors**: Chirag Shah, Hideo Joho, Kirandeep Kaur, Preetam Prabhu Srikar Dammu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06277)  

**Abstract**: Recent advancements in generative AI have significantly increased interest in personalized agents. With increased personalization, there is also a greater need for being able to trust decision-making and action taking capabilities of these agents. However, the evaluation methods for these agents remain outdated and inadequate, often failing to capture the dynamic and evolving nature of user interactions. In this conceptual article, we argue for a paradigm shift in evaluating personalized and adaptive agents. We propose a comprehensive novel framework that models user personas with unique attributes and preferences. In this framework, agents interact with these simulated users through structured interviews to gather their preferences and offer customized recommendations. These recommendations are then assessed dynamically using simulations driven by Large Language Models (LLMs), enabling an adaptive and iterative evaluation process. Our flexible framework is designed to support a variety of agents and applications, ensuring a comprehensive and versatile evaluation of recommendation strategies that focus on proactive, personalized, and trustworthy aspects. 

---
# Can we repurpose multiple-choice question-answering models to rerank retrieved documents? 

**Authors**: Jasper Kyle Catapang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06276)  

**Abstract**: Yes, repurposing multiple-choice question-answering (MCQA) models for document reranking is both feasible and valuable. This preliminary work is founded on mathematical parallels between MCQA decision-making and cross-encoder semantic relevance assessments, leading to the development of R*, a proof-of-concept model that harmonizes these approaches. Designed to assess document relevance with depth and precision, R* showcases how MCQA's principles can improve reranking in information retrieval (IR) and retrieval-augmented generation (RAG) systems -- ultimately enhancing search and dialogue in AI-powered systems. Through experimental validation, R* proves to improve retrieval accuracy and contribute to the field's advancement by demonstrating a practical prototype of MCQA for reranking by keeping it lightweight. 

---
# A Cascaded Architecture for Extractive Summarization of Multimedia Content via Audio-to-Text Alignment 

**Authors**: Tanzir Hossain, Ar-Rafi Islam, Md. Sabbir Hossain, Annajiat Alim Rasel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06275)  

**Abstract**: This study presents a cascaded architecture for extractive summarization of multimedia content via audio-to-text alignment. The proposed framework addresses the challenge of extracting key insights from multimedia sources like YouTube videos. It integrates audio-to-text conversion using Microsoft Azure Speech with advanced extractive summarization models, including Whisper, Pegasus, and Facebook BART XSum. The system employs tools such as Pytube, Pydub, and SpeechRecognition for content retrieval, audio extraction, and transcription. Linguistic analysis is enhanced through named entity recognition and semantic role labeling. Evaluation using ROUGE and F1 scores demonstrates that the cascaded architecture outperforms conventional summarization methods, despite challenges like transcription errors. Future improvements may include model fine-tuning and real-time processing. This study contributes to multimedia summarization by improving information retrieval, accessibility, and user experience. 

---
# Joint Group Profiling and Recommendation via Deep Neural Network-based Multi-Task Learning 

**Authors**: Ngoc Luyen Le, Marie-Hélène Abel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06274)  

**Abstract**: Group recommender systems aim to generate recommendations that align with the collective preferences of a group, introducing challenges that differ significantly from those in individual recommendation scenarios. This paper presents Joint Group Profiling and Recommendation via Deep Neural Network-based Multi-Task Learning, a framework that unifies group profiling and recommendation tasks within a single model. By jointly learning these tasks, the model develops a deeper understanding of group dynamics, leading to improved recommendation accuracy. The shared representations between the two tasks facilitate the discovery of latent features essential to both, resulting in richer and more informative group embeddings. To further enhance performance, an attention mechanism is integrated to dynamically evaluate the relevance of different group features and item attributes, ensuring the model prioritizes the most impactful information. Experiments and evaluations on real-world datasets demonstrate that our multi-task learning approach consistently outperforms baseline models in terms of accuracy, validating its effectiveness and robustness. 

---
# A Diverse and Effective Retrieval-Based Debt Collection System with Expert Knowledge 

**Authors**: Jiaming Luo, Weiyi Luo, Guoqing Sun, Mengchen Zhu, Haifeng Tang, Kunyao Lan, Mengyue Wu, Kenny Q. Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06273)  

**Abstract**: Designing effective debt collection systems is crucial for improving operational efficiency and reducing costs in the financial industry. However, the challenges of maintaining script diversity, contextual relevance, and coherence make this task particularly difficult. This paper presents a debt collection system based on real debtor-collector data from a major commercial bank. We construct a script library from real-world debt collection conversations, and propose a two-stage retrieval based response system for contextual relevance. Experimental results show that our system improves script diversity, enhances response relevance, and achieves practical deployment efficiency through knowledge distillation. This work offers a scalable and automated solution, providing valuable insights for advancing debt collection practices in real-world applications. 

---
# RAVEN: An Agentic Framework for Multimodal Entity Discovery from Large-Scale Video Collections 

**Authors**: Kevin Dela Rosa  

**Link**: [PDF](https://arxiv.org/pdf/2504.06272)  

**Abstract**: We present RAVEN an adaptive AI agent framework designed for multimodal entity discovery and retrieval in large-scale video collections. Synthesizing information across visual, audio, and textual modalities, RAVEN autonomously processes video data to produce structured, actionable representations for downstream tasks. Key contributions include (1) a category understanding step to infer video themes and general-purpose entities, (2) a schema generation mechanism that dynamically defines domain-specific entities and attributes, and (3) a rich entity extraction process that leverages semantic retrieval and schema-guided prompting. RAVEN is designed to be model-agnostic, allowing the integration of different vision-language models (VLMs) and large language models (LLMs) based on application-specific requirements. This flexibility supports diverse applications in personalized search, content discovery, and scalable information retrieval, enabling practical applications across vast datasets. 

---
# ER-RAG: Enhance RAG with ER-Based Unified Modeling of Heterogeneous Data Sources 

**Authors**: Yikuan Xia, Jiazun Chen, Yirui Zhan, Suifeng Zhao, Weipeng Jiang, Chaorui Zhang, Wei Han, Bo Bai, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06271)  

**Abstract**: Large language models (LLMs) excel in question-answering (QA) tasks, and retrieval-augmented generation (RAG) enhances their precision by incorporating external evidence from diverse sources like web pages, databases, and knowledge graphs. However, current RAG methods rely on agent-specific strategies for individual data sources, posing challenges low-resource or black-box environments and complicates operations when evidence is fragmented across sources. To address these limitations, we propose ER-RAG, a framework that unifies evidence integration across heterogeneous data sources using the Entity-Relationship (ER) model. ER-RAG standardizes entity retrieval and relationship querying through ER-based APIs with GET and JOIN operations. It employs a two-stage generation process: first, a preference optimization module selects optimal sources; second, another module constructs API chains based on source schemas. This unified approach allows efficient fine-tuning and seamless integration across diverse data sources. ER-RAG demonstrated its effectiveness by winning all three tracks of the 2024 KDDCup CRAG Challenge, achieving performance on par with commercial RAG pipelines using an 8B LLM backbone. It outperformed hybrid competitors by 3.1% in LLM score and accelerated retrieval by 5.5X. 

---
# Addressing Cold-start Problem in Click-Through Rate Prediction via Supervised Diffusion Modeling 

**Authors**: Wenqiao Zhu, Lulu Wang, Jun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06270)  

**Abstract**: Predicting Click-Through Rates is a crucial function within recommendation and advertising platforms, as the output of CTR prediction determines the order of items shown to users. The Embedding \& MLP paradigm has become a standard approach for industrial recommendation systems and has been widely deployed. However, this paradigm suffers from cold-start problems, where there is either no or only limited user action data available, leading to poorly learned ID embeddings. The cold-start problem hampers the performance of new items. To address this problem, we designed a novel diffusion model to generate a warmed-up embedding for new items. Specifically, we define a novel diffusion process between the ID embedding space and the side information space. In addition, we can derive a sub-sequence from the diffusion steps to expedite training, given that our diffusion model is non-Markovian. Our diffusion model is supervised by both the variational inference and binary cross-entropy objectives, enabling it to generate warmed-up embeddings for items in both the cold-start and warm-up phases. Additionally, we have conducted extensive experiments on three recommendation datasets. The results confirmed the effectiveness of our approach. 

---
# EXCLAIM: An Explainable Cross-Modal Agentic System for Misinformation Detection with Hierarchical Retrieval 

**Authors**: Yin Wu, Zhengxuan Zhang, Fuling Wang, Yuyu Luo, Hui Xiong, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06269)  

**Abstract**: Misinformation continues to pose a significant challenge in today's information ecosystem, profoundly shaping public perception and behavior. Among its various manifestations, Out-of-Context (OOC) misinformation is particularly obscure, as it distorts meaning by pairing authentic images with misleading textual narratives. Existing methods for detecting OOC misinformation predominantly rely on coarse-grained similarity metrics between image-text pairs, which often fail to capture subtle inconsistencies or provide meaningful explainability. While multi-modal large language models (MLLMs) demonstrate remarkable capabilities in visual reasoning and explanation generation, they have not yet demonstrated the capacity to address complex, fine-grained, and cross-modal distinctions necessary for robust OOC detection. To overcome these limitations, we introduce EXCLAIM, a retrieval-based framework designed to leverage external knowledge through multi-granularity index of multi-modal events and entities. Our approach integrates multi-granularity contextual analysis with a multi-agent reasoning architecture to systematically evaluate the consistency and integrity of multi-modal news content. Comprehensive experiments validate the effectiveness and resilience of EXCLAIM, demonstrating its ability to detect OOC misinformation with 4.3% higher accuracy compared to state-of-the-art approaches, while offering explainable and actionable insights. 

---
# KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs 

**Authors**: Elan Markowitz, Krupa Galiya, Greg Ver Steeg, Aram Galstyan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07087)  

**Abstract**: Knowledge graphs have emerged as a popular method for injecting up-to-date, factual knowledge into large language models (LLMs). This is typically achieved by converting the knowledge graph into text that the LLM can process in context. While multiple methods of encoding knowledge graphs have been proposed, the impact of this textualization process on LLM performance remains under-explored. We introduce KG-LLM-Bench, a comprehensive and extensible benchmark spanning five knowledge graph understanding tasks, and evaluate how different encoding strategies affect performance across various base models. Our extensive experiments with seven language models and five textualization strategies provide insights for optimizing LLM performance on KG reasoning tasks. 

---
