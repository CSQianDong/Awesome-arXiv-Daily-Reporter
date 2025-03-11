# Talking to GDELT Through Knowledge Graphs 

**Authors**: Audun Myers, Max Vargas, Sinan G. Aksoy, Cliff Joslyn, Benjamin Wilson, Tom Grimes  

**Link**: [PDF](https://arxiv.org/pdf/2503.07584)  

**Abstract**: In this work we study various Retrieval Augmented Regeneration (RAG) approaches to gain an understanding of the strengths and weaknesses of each approach in a question-answering analysis. To gain this understanding we use a case-study subset of the Global Database of Events, Language, and Tone (GDELT) dataset as well as a corpus of raw text scraped from the online news articles. To retrieve information from the text corpus we implement a traditional vector store RAG as well as state-of-the-art large language model (LLM) based approaches for automatically constructing KGs and retrieving the relevant subgraphs. In addition to these corpus approaches, we develop a novel ontology-based framework for constructing knowledge graphs (KGs) from GDELT directly which leverages the underlying schema of GDELT to create structured representations of global events. For retrieving relevant information from the ontology-based KGs we implement both direct graph queries and state-of-the-art graph retrieval approaches. We compare the performance of each method in a question-answering task. We find that while our ontology-based KGs are valuable for question-answering, automated extraction of the relevant subgraphs is challenging. Conversely, LLM-generated KGs, while capturing event summaries, often lack consistency and interpretability. Our findings suggest benefits of a synergistic approach between ontology and LLM-based KG construction, with proposed avenues toward that end. 

---
# GRITHopper: Decomposition-Free Multi-Hop Dense Retrieval 

**Authors**: Justus-Jonas Erker, Nils Reimers, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2503.07519)  

**Abstract**: Decomposition-based multi-hop retrieval methods rely on many autoregressive steps to break down complex queries, which breaks end-to-end differentiability and is computationally expensive. Decomposition-free methods tackle this, but current decomposition-free approaches struggle with longer multi-hop problems and generalization to out-of-distribution data. To address these challenges, we introduce GRITHopper-7B, a novel multi-hop dense retrieval model that achieves state-of-the-art performance on both in-distribution and out-of-distribution benchmarks. GRITHopper combines generative and representational instruction tuning by integrating causal language modeling with dense retrieval training. Through controlled studies, we find that incorporating additional context after the retrieval process, referred to as post-retrieval language modeling, enhances dense retrieval performance. By including elements such as final answers during training, the model learns to better contextualize and retrieve relevant information. GRITHopper-7B offers a robust, scalable, and generalizable solution for multi-hop dense retrieval, and we release it to the community for future research and applications requiring multi-hop reasoning and retrieval capabilities. 

---
# Advancing Vietnamese Information Retrieval with Learning Objective and Benchmark 

**Authors**: Phu-Vinh Nguyen, Minh-Nam Tran, Long Nguyen, Dien Dinh  

**Link**: [PDF](https://arxiv.org/pdf/2503.07470)  

**Abstract**: With the rapid development of natural language processing, many language models have been invented for multiple tasks. One important task is information retrieval (IR), which requires models to retrieve relevant documents. Despite its importance in many real-life applications, especially in retrieval augmented generation (RAG) systems, this task lacks Vietnamese benchmarks. This situation causes difficulty in assessing and comparing many existing Vietnamese embedding language models on the task and slows down the advancement of Vietnamese natural language processing (NLP) research. In this work, we aim to provide the Vietnamese research community with a new benchmark for information retrieval, which mainly focuses on retrieval and reranking tasks. Furthermore, we also present a new objective function based on the InfoNCE loss function, which is used to train our Vietnamese embedding model. Our function aims to be better than the origin in information retrieval tasks. Finally, we analyze the effect of temperature, a hyper-parameter in both objective functions, on the performance of text embedding models. 

---
# Process-Supervised LLM Recommenders via Flow-guided Tuning 

**Authors**: Chongming Gao, Mengyao Gao, Chenxiao Fan, Shuai Yuan, Wentao Shi, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2503.07377)  

**Abstract**: While large language models (LLMs) are increasingly adapted for recommendation systems via supervised fine-tuning (SFT), this approach amplifies popularity bias due to its likelihood maximization objective, compromising recommendation diversity and fairness. To address this, we present Flow-guided fine-tuning recommender (Flower), which replaces SFT with a Generative Flow Network (GFlowNet) framework that enacts process supervision through token-level reward propagation. Flower's key innovation lies in decomposing item-level rewards into constituent token rewards, enabling direct alignment between token generation probabilities and their reward signals. This mechanism achieves three critical advancements: (1) popularity bias mitigation and fairness enhancement through empirical distribution matching, (2) preservation of diversity through GFlowNet's proportional sampling, and (3) flexible integration of personalized preferences via adaptable token rewards. Experiments demonstrate Flower's superior distribution-fitting capability and its significant advantages over traditional SFT in terms of fairness, diversity, and accuracy, highlighting its potential to improve LLM-based recommendation systems. The implementation is available via this https URL 

---
# Weak Supervision for Improved Precision in Search Systems 

**Authors**: Sriram Vasudevan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07025)  

**Abstract**: Labeled datasets are essential for modern search engines, which increasingly rely on supervised learning methods like Learning to Rank and massive amounts of data to power deep learning models. However, creating these datasets is both time-consuming and costly, leading to the common use of user click and activity logs as proxies for relevance. In this paper, we present a weak supervision approach to infer the quality of query-document pairs and apply it within a Learning to Rank framework to enhance the precision of a large-scale search system. 

---
# Multi-Behavior Recommender Systems: A Survey 

**Authors**: Kyungho Kim, Sunwoo Kim, Geon Lee, Jinhong Jung, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2503.06963)  

**Abstract**: Traditional recommender systems primarily rely on a single type of user-item interaction, such as item purchases or ratings, to predict user preferences. However, in real-world scenarios, users engage in a variety of behaviors, such as clicking on items or adding them to carts, offering richer insights into their interests. Multi-behavior recommender systems leverage these diverse interactions to enhance recommendation quality, and research on this topic has grown rapidly in recent years. This survey provides a timely review of multi-behavior recommender systems, focusing on three key steps: (1) Data Modeling: representing multi-behaviors at the input level, (2) Encoding: transforming these inputs into vector representations (i.e., embeddings), and (3) Training: optimizing machine-learning models. We systematically categorize existing multi-behavior recommender systems based on the commonalities and differences in their approaches across the above steps. Additionally, we discuss promising future directions for advancing multi-behavior recommender systems. 

---
# AlignPxtr: Aligning Predicted Behavior Distributions for Bias-Free Video Recommendations 

**Authors**: Chengzhi Lin, Chuyuan Wang, Annan Xie, Wuhong Wang, Ziye Zhang, Canguang Ruan, Yuancai Huang, Yongqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06920)  

**Abstract**: In video recommendation systems, user behaviors such as watch time, likes, and follows are commonly used to infer user interest. However, these behaviors are influenced by various biases, including duration bias, demographic biases, and content category biases, which obscure true user preferences. In this paper, we hypothesize that biases and user interest are independent of each other. Based on this assumption, we propose a novel method that aligns predicted behavior distributions across different bias conditions using quantile mapping, theoretically guaranteeing zero mutual information between bias variables and the true user interest. By explicitly modeling the conditional distributions of user behaviors under different biases and mapping these behaviors to quantiles, we effectively decouple user interest from the confounding effects of various biases. Our approach uniquely handles both continuous signals (e.g., watch time) and discrete signals (e.g., likes, comments), while simultaneously addressing multiple bias dimensions. Additionally, we introduce a computationally efficient mean alignment alternative technique for practical real-time inference in large-scale systems. We validate our method through online A/B testing on two major video platforms: Kuaishou Lite and Kuaishou. The results demonstrate significant improvements in user engagement and retention, with \textbf{cumulative lifts of 0.267\% and 0.115\% in active days, and 1.102\% and 0.131\% in average app usage time}, respectively. The results demonstrate that our approach consistently achieves significant improvements in long-term user retention and substantial gains in average app usage time across different platforms. 

---
# Improving Access to Trade and Investment Information in Thailand through Intelligent Document Retrieval 

**Authors**: Sirinda Palahan  

**Link**: [PDF](https://arxiv.org/pdf/2503.06489)  

**Abstract**: Overseas investment and trade can be daunting for beginners due to the vast amount of complex information. This paper presents a chatbot system that integrates natural language processing and information retrieval techniques to simplify the document retrieval process. The proposed system identifies the most relevant content, enabling users to navigate the intricate landscape of foreign trade and investment more efficiently. Our methodology combines the BM25 model and a deep learning model to rank and retrieve documents, aiming to reduce noise in the document content and enhance the accuracy of the results. Experiments with Thai natural language queries have demonstrated the effectiveness of our system in retrieving pertinent documents. A user satisfaction survey further validated the system's effectiveness. Most respondents found the system helpful and agreed with the suggested documents, indicating its potential as a valuable tool for Thai entrepreneurs navigating foreign trade and investment. 

---
# HuixiangDou2: A Robustly Optimized GraphRAG Approach 

**Authors**: Huanjun Kong, Zhefan Wang, Chenyang Wang, Zhe Ma, Nanqing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2503.06474)  

**Abstract**: Large Language Models (LLMs) perform well on familiar queries but struggle with specialized or emerging topics. Graph-based Retrieval-Augmented Generation (GraphRAG) addresses this by structuring domain knowledge as a graph for dynamic retrieval. However, existing pipelines involve complex engineering workflows, making it difficult to isolate the impact of individual components. Evaluating retrieval effectiveness is also challenging due to dataset overlap with LLM pretraining data. In this work, we introduce HuixiangDou2, a robustly optimized GraphRAG framework. Specifically, we leverage the effectiveness of dual-level retrieval and optimize its performance in a 32k context for maximum precision, and compare logic-based retrieval and dual-level retrieval to enhance overall functionality. Our implementation includes comparative experiments on a test set, where Qwen2.5-7B-Instruct initially underperformed. With our approach, the score improved significantly from 60 to 74.5, as illustrated in the Figure. Experiments on domain-specific datasets reveal that dual-level retrieval enhances fuzzy matching, while logic-form retrieval improves structured reasoning. Furthermore, we propose a multi-stage verification mechanism to improve retrieval robustness without increasing computational cost. Empirical results show significant accuracy gains over baselines, highlighting the importance of adaptive retrieval. To support research and adoption, we release HuixiangDou2 as an open-source resource this https URL. 

---
# Image is All You Need: Towards Efficient and Effective Large Language Model-Based Recommender Systems 

**Authors**: Kibum Kim, Sein Kim, Hongseok Kang, Jiwan Kim, Heewoong Noh, Yeonjun In, Kanghoon Yoon, Jinoh Oh, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2503.06238)  

**Abstract**: Large Language Models (LLMs) have recently emerged as a powerful backbone for recommender systems. Existing LLM-based recommender systems take two different approaches for representing items in natural language, i.e., Attribute-based Representation and Description-based Representation. In this work, we aim to address the trade-off between efficiency and effectiveness that these two approaches encounter, when representing items consumed by users. Based on our interesting observation that there is a significant information overlap between images and descriptions associated with items, we propose a novel method, Image is all you need for LLM-based Recommender system (I-LLMRec). Our main idea is to leverage images as an alternative to lengthy textual descriptions for representing items, aiming at reducing token usage while preserving the rich semantic information of item descriptions. Through extensive experiments, we demonstrate that I-LLMRec outperforms existing methods in both efficiency and effectiveness by leveraging images. Moreover, a further appeal of I-LLMRec is its ability to reduce sensitivity to noise in descriptions, leading to more robust recommendations. 

---
# Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning 

**Authors**: Shengyao Zhuang, Xueguang Ma, Bevan Koopman, Jimmy Lin, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2503.06034)  

**Abstract**: In this paper, we introduce Rank-R1, a novel LLM-based reranker that performs reasoning over both the user query and candidate documents before performing the ranking task. Existing document reranking methods based on large language models (LLMs) typically rely on prompting or fine-tuning LLMs to order or label candidate documents according to their relevance to a query. For Rank-R1, we use a reinforcement learning algorithm along with only a small set of relevance labels (without any reasoning supervision) to enhance the reasoning ability of LLM-based rerankers. Our hypothesis is that adding reasoning capabilities to the rerankers can improve their relevance assessement and ranking capabilities. Our experiments on the TREC DL and BRIGHT datasets show that Rank-R1 is highly effective, especially for complex queries. In particular, we find that Rank-R1 achieves effectiveness on in-domain datasets at par with that of supervised fine-tuning methods, but utilizing only 18\% of the training data used by the fine-tuning methods. We also find that the model largely outperforms zero-shot and supervised fine-tuning when applied to out-of-domain datasets featuring complex queries, especially when a 14B-size model is used. Finally, we qualitatively observe that Rank-R1's reasoning process improves the explainability of the ranking results, opening new opportunities for search engine results presentation and fruition. 

---
# From Limited Labels to Open Domains: An Efficient Learning Paradigm for UAV-view Geo-Localization 

**Authors**: Zhongwei Chen, Zhao-Xu Yang, Hai-Jun Rong, Jiawei Lang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07520)  

**Abstract**: Traditional UAV-view Geo-Localization (UVGL) supervised paradigms are constrained by the strict reliance on paired data for positive sample selection, which limits their ability to learn cross-view domain-invariant representations from unpaired data. Moreover, it is necessary to reconstruct the pairing relationship with expensive re-labeling costs for scenario-specific training when deploying in a new domain, which fails to meet the practical demands of open-environment applications. To address this issue, we propose a novel cross-domain invariance knowledge transfer network (CDIKTNet), which comprises a cross-domain invariance sub-network and a cross-domain transfer sub-network to realize a closed-loop framework of invariance feature learning and knowledge transfer. The cross-domain invariance sub-network is utilized to construct an essentially shared feature space across domains by learning structural invariance and spatial invariance in cross-view features. Meanwhile, the cross-domain transfer sub-network uses these invariant features as anchors and employs a dual-path contrastive memory learning mechanism to mine latent cross-domain correlation patterns in unpaired data. Extensive experiments demonstrate that our method achieves state-of-the-art performance under fully supervised conditions. More importantly, with merely 2\% paired data, our method exhibits performance comparable to existing supervised paradigms and possesses the ability to transfer directly to qualify for applications in the other scenarios completely without any prior pairing relationship. 

---
# Zero-Shot Hashing Based on Reconstruction With Part Alignment 

**Authors**: Yan Jiang, Zhongmiao Qi, Jianhao Li, Jiangbo Qian, Chong Wang, Yu Xin  

**Link**: [PDF](https://arxiv.org/pdf/2503.07037)  

**Abstract**: Hashing algorithms have been widely used in large-scale image retrieval tasks, especially for seen class data. Zero-shot hashing algorithms have been proposed to handle unseen class data. The key technique in these algorithms involves learning features from seen classes and transferring them to unseen classes, that is, aligning the feature embeddings between the seen and unseen classes. Most existing zero-shot hashing algorithms use the shared attributes between the two classes of interest to complete alignment tasks. However, the attributes are always described for a whole image, even though they represent specific parts of the image. Hence, these methods ignore the importance of aligning attributes with the corresponding image parts, which explicitly introduces noise and reduces the accuracy achieved when aligning the features of seen and unseen classes. To address this problem, we propose a new zero-shot hashing method called RAZH. We first use a clustering algorithm to group similar patches to image parts for attribute matching and then replace the image parts with the corresponding attribute vectors, gradually aligning each part with its nearest attribute. Extensive evaluation results demonstrate the superiority of the RAZH method over several state-of-the-art methods. 

---
# Graph Retrieval-Augmented LLM for Conversational Recommendation Systems 

**Authors**: Zhangchi Qiu, Linhao Luo, Zicheng Zhao, Shirui Pan, Alan Wee-Chung Liew  

**Link**: [PDF](https://arxiv.org/pdf/2503.06430)  

**Abstract**: Conversational Recommender Systems (CRSs) have emerged as a transformative paradigm for offering personalized recommendations through natural language dialogue. However, they face challenges with knowledge sparsity, as users often provide brief, incomplete preference statements. While recent methods have integrated external knowledge sources to mitigate this, they still struggle with semantic understanding and complex preference reasoning. Recent Large Language Models (LLMs) demonstrate promising capabilities in natural language understanding and reasoning, showing significant potential for CRSs. Nevertheless, due to the lack of domain knowledge, existing LLM-based CRSs either produce hallucinated recommendations or demand expensive domain-specific training, which largely limits their applicability. In this work, we present G-CRS (Graph Retrieval-Augmented Large Language Model for Conversational Recommender Systems), a novel training-free framework that combines graph retrieval-augmented generation and in-context learning to enhance LLMs' recommendation capabilities. Specifically, G-CRS employs a two-stage retrieve-and-recommend architecture, where a GNN-based graph reasoner first identifies candidate items, followed by Personalized PageRank exploration to jointly discover potential items and similar user interactions. These retrieved contexts are then transformed into structured prompts for LLM reasoning, enabling contextually grounded recommendations without task-specific training. Extensive experiments on two public datasets show that G-CRS achieves superior recommendation performance compared to existing methods without requiring task-specific training. 

---
