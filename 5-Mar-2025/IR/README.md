# Zero-Shot Complex Question-Answering on Long Scientific Documents 

**Authors**: Wanting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02695)  

**Abstract**: With the rapid development in Transformer-based language models, the reading comprehension tasks on short documents and simple questions have been largely addressed. Long documents, specifically the scientific documents that are densely packed with knowledge discovered and developed by humans, remain relatively unexplored. These documents often come with a set of complex and more realistic questions, adding to their complexity. We present a zero-shot pipeline framework that enables social science researchers to perform question-answering tasks that are complex yet of predetermined question formats on full-length research papers without requiring machine learning expertise. Our approach integrates pre-trained language models to handle challenging scenarios including multi-span extraction, multi-hop reasoning, and long-answer generation. Evaluating on MLPsych, a novel dataset of social psychology papers with annotated complex questions, we demonstrate that our framework achieves strong performance through combination of extractive and generative models. This work advances document understanding capabilities for social sciences while providing practical tools for researchers. 

---
# Towards Robust Expert Finding in Community Question Answering Platforms 

**Authors**: Maddalena Amendola, Andrea Passarella, Raffaele Perego  

**Link**: [PDF](https://arxiv.org/pdf/2503.02674)  

**Abstract**: This paper introduces TUEF, a topic-oriented user-interaction model for fair Expert Finding in Community Question Answering (CQA) platforms. The Expert Finding task in CQA platforms involves identifying proficient users capable of providing accurate answers to questions from the community. To this aim, TUEF improves the robustness and credibility of the CQA platform through a more precise Expert Finding component. The key idea of TUEF is to exploit diverse types of information, specifically, content and social information, to identify more precisely experts thus improving the robustness of the task. We assess TUEF through reproducible experiments conducted on a large-scale dataset from StackOverflow. The results consistently demonstrate that TUEF outperforms state-of-the-art competitors while promoting transparent expert identification. 

---
# Personalized Generation In Large Model Era: A Survey 

**Authors**: Yiyan Xu, Jinghao Zhang, Alireza Salemi, Xinting Hu, Wenjie Wang, Fuli Feng, Hamed Zamani, Xiangnan He, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2503.02614)  

**Abstract**: In the era of large models, content generation is gradually shifting to Personalized Generation (PGen), tailoring content to individual preferences and needs. This paper presents the first comprehensive survey on PGen, investigating existing research in this rapidly growing field. We conceptualize PGen from a unified perspective, systematically formalizing its key components, core objectives, and abstract workflows. Based on this unified perspective, we propose a multi-level taxonomy, offering an in-depth review of technical advancements, commonly used datasets, and evaluation metrics across multiple modalities, personalized contexts, and tasks. Moreover, we envision the potential applications of PGen and highlight open challenges and promising directions for future exploration. By bridging PGen research across multiple modalities, this survey serves as a valuable resource for fostering knowledge sharing and interdisciplinary collaboration, ultimately contributing to a more personalized digital landscape. 

---
# Efficient Long Sequential Low-rank Adaptive Attention for Click-through rate Prediction 

**Authors**: Xin Song, Xiaochen Li, Jinxin Hu, Hong Wen, Zulong Chen, Yu Zhang, Xiaoyi Zeng, Zhang Jing  

**Link**: [PDF](https://arxiv.org/pdf/2503.02542)  

**Abstract**: In the context of burgeoning user historical behavior data, Accurate click-through rate(CTR) prediction requires effective modeling of lengthy user behavior sequences. As the volume of such data keeps swelling, the focus of research has shifted towards developing effective long-term behavior modeling methods to capture latent user interests. Nevertheless, the complexity introduced by large scale data brings about computational hurdles. There is a pressing need to strike a balance between achieving high model performance and meeting the strict response time requirements of online services. While existing retrieval-based methods (e.g., similarity filtering or attention approximation) achieve practical runtime efficiency, they inherently compromise information fidelity through aggressive sequence truncation or attention sparsification. This paper presents a novel attention mechanism. It overcomes the shortcomings of existing methods while ensuring computational efficiency. This mechanism learn compressed representation of sequence with length $L$ via low-rank projection matrices (rank $r \ll L$), reducing attention complexity from $O(L)$ to $O(r)$. It also integrates a uniquely designed loss function to preserve nonlinearity of attention. In the inference stage, the mechanism adopts matrix absorption and prestorage strategies. These strategies enable it to effectively satisfy online constraints. Comprehensive offline and online experiments demonstrate that the proposed method outperforms current state-of-the-art solutions. 

---
# Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations 

**Authors**: Yuhao Yang, Zhi Ji, Zhaopeng Li, Yi Li, Zhonglin Mo, Yue Ding, Kai Chen, Zijian Zhang, Jie Li, Shuanglong Li, Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02453)  

**Abstract**: Generative models have recently gained attention in recommendation systems by directly predicting item identifiers from user interaction sequences. However, existing methods suffer from significant information loss due to the separation of stages such as quantization and sequence modeling, hindering their ability to achieve the modeling precision and accuracy of sequential dense retrieval techniques. Integrating generative and dense retrieval methods remains a critical challenge. To address this, we introduce the Cascaded Organized Bi-Represented generAtive retrieval (COBRA) framework, which innovatively integrates sparse semantic IDs and dense vectors through a cascading process. Our method alternates between generating these representations by first generating sparse IDs, which serve as conditions to aid in the generation of dense vectors. End-to-end training enables dynamic refinement of dense representations, capturing both semantic insights and collaborative signals from user-item interactions. During inference, COBRA employs a coarse-to-fine strategy, starting with sparse ID generation and refining them into dense vectors via the generative model. We further propose BeamFusion, an innovative approach combining beam search with nearest neighbor scores to enhance inference flexibility and recommendation diversity. Extensive experiments on public datasets and offline tests validate our method's robustness. Online A/B tests on a real-world advertising platform with over 200 million daily users demonstrate substantial improvements in key metrics, highlighting COBRA's practical advantages. 

---
# Hierarchical Re-ranker Retriever (HRR) 

**Authors**: Ashish Singh, Priti Mohapatra  

**Link**: [PDF](https://arxiv.org/pdf/2503.02401)  

**Abstract**: Retrieving the right level of context for a given query is a perennial challenge in information retrieval - too large a chunk dilutes semantic specificity, while chunks that are too small lack broader context. This paper introduces the Hierarchical Re-ranker Retriever (HRR), a framework designed to achieve both fine-grained and high-level context retrieval for large language model (LLM) applications. In HRR, documents are split into sentence-level and intermediate-level (512 tokens) chunks to maximize vector-search quality for both short and broad queries. We then employ a reranker that operates on these 512-token chunks, ensuring an optimal balance neither too coarse nor too fine for robust relevance scoring. Finally, top-ranked intermediate chunks are mapped to parent chunks (2048 tokens) to provide an LLM with sufficiently large context. 

---
# PersonaX: A Recommendation Agent Oriented User Modeling Framework for Long Behavior Sequence 

**Authors**: Yunxiao Shi, Wujiang Xu, Zeqi Zhang, Xing Zi, Qiang Wu, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02398)  

**Abstract**: Recommendation agents leverage large language models for user modeling LLM UM to construct textual personas guiding alignment with real users. However existing LLM UM methods struggle with long user generated content UGC due to context limitations and performance degradation. To address this sampling strategies prioritize relevance or recency are often applied yet they inevitably neglect the diverse user interests embedded within the discarded behaviors resulting in incomplete modeling and degraded profiling quality. Furthermore relevance based sampling requires real time retrieval forcing the user modeling process to operate online which introduces significant latency overhead. In this paper we propose PersonaX an agent agnostic LLM UM framework that tackles these challenges through sub behavior sequence SBS selection and offline multi persona construction. PersonaX extracts compact SBS segments offline to capture diverse user interests generating fine grained textual personas that are cached for efficient online retrieval. This approach ensures that the user persona used for prompting remains highly relevant to the current context while eliminating the need for online user modeling. For SBS selection we ensure both efficiency length less than five and high representational quality by balancing prototypicality and diversity within the sampled data. Extensive experiments validate the effectiveness and versatility of PersonaX in high quality user profiling. Utilizing only 30 to 50 percent of the behavioral data with a sequence length of 480 integrating PersonaX with AgentCF yields an absolute performance improvement of 3 to 11 percent while integration with Agent4Rec results in a gain of 10 to 50 percent. PersonaX as an agent agnostic framework sets a new benchmark for scalable user modeling paving the way for more accurate and efficient LLM driven recommendation agents. 

---
# Towards Explainable Doctor Recommendation with Large Language Models 

**Authors**: Ziyang Zeng, Dongyuan Li, Yuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02298)  

**Abstract**: The advent of internet medicine provides patients with unprecedented convenience in searching and communicating with doctors relevant to their diseases and desired treatments online. However, the current doctor recommendation systems fail to fully ensure the professionalism and interpretability of the recommended results. In this work, we formulate doctor recommendation as a ranking task and develop a large language model (LLM)-based pointwise ranking framework. Our framework ranks doctors according to their relevance regarding specific diseases-treatment pairs in a zero-shot setting. The advantage of our framework lies in its ability to generate precise and explainable doctor ranking results. Additionally, we construct DrRank, a new expertise-driven doctor ranking dataset comprising over 38 disease-treatment pairs. Experiment results on the DrRank dataset demonstrate that our framework significantly outperforms the strongest cross-encoder baseline, achieving a notable gain of +5.45 in the NDCG@10 score while maintaining affordable latency consumption. Furthermore, we comprehensively present the fairness analysis results of our framework from three perspectives of different diseases, patient gender, and geographical regions. Meanwhile, the interpretability of our framework is rigorously verified by three human experts, providing further evidence of the reliability of our proposed framework for doctor recommendation. 

---
# Tailoring Table Retrieval from a Field-aware Hybrid Matching Perspective 

**Authors**: Da Li, Keping Bi, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.02251)  

**Abstract**: Table retrieval, essential for accessing information through tabular data, is less explored compared to text retrieval. The row/column structure and distinct fields of tables (including titles, headers, and cells) present unique challenges. For example, different table fields have varying matching preferences: cells may favor finer-grained (word/phrase level) matching over broader (sentence/passage level) matching due to their fragmented and detailed nature, unlike titles. This necessitates a table-specific retriever to accommodate the various matching needs of each table field. Therefore, we introduce a Table-tailored HYbrid Matching rEtriever (THYME), which approaches table retrieval from a field-aware hybrid matching perspective. Empirical results on two table retrieval benchmarks, NQ-TABLES and OTT-QA, show that THYME significantly outperforms state-of-the-art baselines. Comprehensive analyses confirm the differing matching preferences across table fields and validate the design of THYME. 

---
# Are some books better than others? 

**Authors**: Hannes Rosenbusch, Luke Korthals  

**Link**: [PDF](https://arxiv.org/pdf/2503.02671)  

**Abstract**: Scholars, awards committees, and laypeople frequently discuss the merit of written works. Literary professionals and journalists differ in how much perspectivism they concede in their book reviews. Here, we quantify how strongly book reviews are determined by the actual book contents vs. idiosyncratic reader tendencies. In our analysis of 624,320 numerical and textual book reviews, we find that the contents of professionally published books are not predictive of a random reader's reading enjoyment. Online reviews of popular fiction and non-fiction books carry up to ten times more information about the reviewer than about the book. For books of a preferred genre, readers might be less likely to give low ratings, but still struggle to converge in their relative assessments. We find that book evaluations generalize more across experienced review writers than casual readers. When discussing specific issues with a book, one review text had poor predictability of issues brought up in another review of the same book. We conclude that extreme perspectivism is a justifiable position when researching literary quality, bestowing literary awards, and designing recommendation systems. 

---
# Survey Perspective: The Role of Explainable AI in Threat Intelligence 

**Authors**: Nidhi Rastogi, Devang Dhanuka, Amulya Saxena, Pranjal Mairal, Le Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.02065)  

**Abstract**: The increasing reliance on AI-based security tools in Security Operations Centers (SOCs) has transformed threat detection and response, yet analysts frequently struggle with alert overload, false positives, and lack of contextual relevance. The inability to effectively analyze AI-generated security alerts lead to inefficiencies in incident response and reduces trust in automated decision-making. In this paper, we show results and analysis of our investigation of how SOC analysts navigate AI-based alerts, their challenges with current security tools, and how explainability (XAI) integrated into their security workflows has the potential to become an effective decision support. In this vein, we conducted an industry survey. Using the survey responses, we analyze how security analysts' process, retrieve, and prioritize alerts. Our findings indicate that most analysts have not yet adopted XAI-integrated tools, but they express high interest in attack attribution, confidence scores, and feature contribution explanations to improve interpretability, and triage efficiency. Based on our findings, we also propose practical design recommendations for XAI-enhanced security alert systems, enabling AI-based cybersecurity solutions to be more transparent, interpretable, and actionable. 

---
