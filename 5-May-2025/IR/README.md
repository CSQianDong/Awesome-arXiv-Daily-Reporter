# Multi-agents based User Values Mining for Recommendation 

**Authors**: Lijian Chen, Wei Yuan, Tong Chen, Xiangyu Zhao, Nguyen Quoc Viet Hung, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.00981)  

**Abstract**: Recommender systems have rapidly evolved and become integral to many online services. However, existing systems sometimes produce unstable and unsatisfactory recommendations that fail to align with users' fundamental and long-term preferences. This is because they primarily focus on extracting shallow and short-term interests from user behavior data, which is inherently dynamic and challenging to model. Unlike these transient interests, user values are more stable and play a crucial role in shaping user behaviors, such as purchasing items and consuming content. Incorporating user values into recommender systems can help stabilize recommendation performance and ensure results better reflect users' latent preferences. However, acquiring user values is typically difficult and costly. To address this challenge, we leverage the strong language understanding, zero-shot inference, and generalization capabilities of Large Language Models (LLMs) to extract user values from users' historical interactions. Unfortunately, direct extraction using LLMs presents several challenges such as length constraints and hallucination. To overcome these issues, we propose ZOOM, a zero-shot multi-LLM collaborative framework for effective and accurate user value extraction. In ZOOM, we apply text summarization techniques to condense item content while preserving essential meaning. To mitigate hallucinations, ZOOM introduces two specialized agent roles: evaluators and supervisors, to collaboratively generate accurate user values. Extensive experiments on two widely used recommendation datasets with two state-of-the-art recommendation models demonstrate the effectiveness and generalization of our framework in automatic user value mining and recommendation performance improvement. 

---
# Enhancing User Sequence Modeling through Barlow Twins-based Self-Supervised Learning 

**Authors**: Yuhan Liu, Lin Ning, Neo Wu, Karan Singhal, Philip Andrew Mansfield, Devora Berlowitz, Sushant Prakash, Bradley Green  

**Link**: [PDF](https://arxiv.org/pdf/2505.00953)  

**Abstract**: User sequence modeling is crucial for modern large-scale recommendation systems, as it enables the extraction of informative representations of users and items from their historical interactions. These user representations are widely used for a variety of downstream tasks to enhance users' online experience. A key challenge for learning these representations is the lack of labeled training data. While self-supervised learning (SSL) methods have emerged as a promising solution for learning representations from unlabeled data, many existing approaches rely on extensive negative sampling, which can be computationally expensive and may not always be feasible in real-world scenario. In this work, we propose an adaptation of Barlow Twins, a state-of-the-art SSL methods, to user sequence modeling by incorporating suitable augmentation methods. Our approach aims to mitigate the need for large negative sample batches, enabling effective representation learning with smaller batch sizes and limited labeled data. We evaluate our method on the MovieLens-1M, MovieLens-20M, and Yelp datasets, demonstrating that our method consistently outperforms the widely-used dual encoder model across three downstream tasks, achieving an 8%-20% improvement in accuracy. Our findings underscore the effectiveness of our approach in extracting valuable sequence-level information for user modeling, particularly in scenarios where labeled data is scarce and negative examples are limited. 

---
# Preserving Privacy and Utility in LLM-Based Product Recommendations 

**Authors**: Tina Khezresmaeilzadeh, Jiang Zhang, Dimitrios Andreadis, Konstantinos Psounis  

**Link**: [PDF](https://arxiv.org/pdf/2505.00951)  

**Abstract**: Large Language Model (LLM)-based recommendation systems leverage powerful language models to generate personalized suggestions by processing user interactions and preferences. Unlike traditional recommendation systems that rely on structured data and collaborative filtering, LLM-based models process textual and contextual information, often using cloud-based infrastructure. This raises privacy concerns, as user data is transmitted to remote servers, increasing the risk of exposure and reducing control over personal information. To address this, we propose a hybrid privacy-preserving recommendation framework which separates sensitive from nonsensitive data and only shares the latter with the cloud to harness LLM-powered recommendations. To restore lost recommendations related to obfuscated sensitive data, we design a de-obfuscation module that reconstructs sensitive recommendations locally. Experiments on real-world e-commerce datasets show that our framework achieves almost the same recommendation utility with a system which shares all data with an LLM, while preserving privacy to a large extend. Compared to obfuscation-only techniques, our approach improves HR@10 scores and category distribution alignment, offering a better balance between privacy and recommendation quality. Furthermore, our method runs efficiently on consumer-grade hardware, making privacy-aware LLM-based recommendation systems practical for real-world use. 

---
# Towards Explainable Temporal User Profiling with LLMs 

**Authors**: Milad Sabouri, Masoud Mansoury, Kun Lin, Bamshad Mobasher  

**Link**: [PDF](https://arxiv.org/pdf/2505.00886)  

**Abstract**: Accurately modeling user preferences is vital not only for improving recommendation performance but also for enhancing transparency in recommender systems. Conventional user profiling methods, such as averaging item embeddings, often overlook the evolving, nuanced nature of user interests, particularly the interplay between short-term and long-term preferences. In this work, we leverage large language models (LLMs) to generate natural language summaries of users' interaction histories, distinguishing recent behaviors from more persistent tendencies. Our framework not only models temporal user preferences but also produces natural language profiles that can be used to explain recommendations in an interpretable manner. These textual profiles are encoded via a pre-trained model, and an attention mechanism dynamically fuses the short-term and long-term embeddings into a comprehensive user representation. Beyond boosting recommendation accuracy over multiple baselines, our approach naturally supports explainability: the interpretable text summaries and attention weights can be exposed to end users, offering insights into why specific items are suggested. Experiments on real-world datasets underscore both the performance gains and the promise of generating clearer, more transparent justifications for content-based recommendations. 

---
# PREMISE: Matching-based Prediction for Accurate Review Recommendation 

**Authors**: Wei Han, Hui Chen, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2505.01255)  

**Abstract**: We present PREMISE (PREdict with Matching ScorEs), a new architecture for the matching-based learning in the multimodal fields for the multimodal review helpfulness (MRHP) task. Distinct to previous fusion-based methods which obtains multimodal representations via cross-modal attention for downstream tasks, PREMISE computes the multi-scale and multi-field representations, filters duplicated semantics, and then obtained a set of matching scores as feature vectors for the downstream recommendation task. This new architecture significantly boosts the performance for such multimodal tasks whose context matching content are highly correlated to the targets of that task, compared to the state-of-the-art fusion-based methods. Experimental results on two publicly available datasets show that PREMISE achieves promising performance with less computational cost. 

---
# CppSATD: A Reusable Self-Admitted Technical Debt Dataset in C++ 

**Authors**: Phuoc Pham, Murali Sridharan, Matteo Esposito, Valentina Lenarduzzi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01136)  

**Abstract**: In software development, technical debt (TD) refers to suboptimal implementation choices made by the developers to meet urgent deadlines and limited resources, posing challenges for future maintenance. Self-Admitted Technical Debt (SATD) is a sub-type of TD, representing specific TD instances ``openly admitted'' by the developers and often expressed through source code comments. Previous research on SATD has focused predominantly on the Java programming language, revealing a significant gap in cross-language SATD. Such a narrow focus limits the generalizability of existing findings as well as SATD detection techniques across multiple programming languages. Our work addresses such limitation by introducing CppSATD, a dedicated C++ SATD dataset, comprising over 531,000 annotated comments and their source code contexts. Our dataset can serve as a foundation for future studies that aim to develop SATD detection methods in C++, generalize the existing findings to other languages, or contribute novel insights to cross-language SATD research. 

---
# FinBERT-QA: Financial Question Answering with pre-trained BERT Language Models 

**Authors**: Bithiah Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00725)  

**Abstract**: Motivated by the emerging demand in the financial industry for the automatic analysis of unstructured and structured data at scale, Question Answering (QA) systems can provide lucrative and competitive advantages to companies by facilitating the decision making of financial advisers. Consequently, we propose a novel financial QA system using the transformer-based pre-trained BERT language model to address the limitations of data scarcity and language specificity in the financial domain. Our system focuses on financial non-factoid answer selection, which retrieves a set of passage-level texts and selects the most relevant as the answer. To increase efficiency, we formulate the answer selection task as a re-ranking problem, in which our system consists of an Answer Retriever using BM25, a simple information retrieval approach, to first return a list of candidate answers, and an Answer Re-ranker built with variants of pre-trained BERT language models to re-rank and select the most relevant answers. We investigate various learning, further pre-training, and fine-tuning approaches for BERT. Our experiments suggest that FinBERT-QA, a model built from applying the Transfer and Adapt further fine-tuning and pointwise learning approach, is the most effective, improving the state-of-the-art results of task 2 of the FiQA dataset by 16% on MRR, 17% on NDCG, and 21% on Precision@1. 

---
