# Joint Modeling in Recommendations: A Survey 

**Authors**: Xiangyu Zhao, Yichao Wang, Bo Chen, Jingtong Gao, Yuhao Wang, Xiaopeng Li, Pengyue Jia, Qidong Liu, Huifeng Guo, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.21195)  

**Abstract**: In today's digital landscape, Deep Recommender Systems (DRS) play a crucial role in navigating and customizing online content for individual preferences. However, conventional methods, which mainly depend on single recommendation task, scenario, data modality and user behavior, are increasingly seen as insufficient due to their inability to accurately reflect users' complex and changing preferences. This gap underscores the need for joint modeling approaches, which are central to overcoming these limitations by integrating diverse tasks, scenarios, modalities, and behaviors in the recommendation process, thus promising significant enhancements in recommendation precision, efficiency, and customization. In this paper, we comprehensively survey the joint modeling methods in recommendations. We begin by defining the scope of joint modeling through four distinct dimensions: multi-task, multi-scenario, multi-modal, and multi-behavior modeling. Subsequently, we examine these methods in depth, identifying and summarizing their underlying paradigms based on the latest advancements and potential research trajectories. Ultimately, we highlight several promising avenues for future exploration in joint modeling for recommendations and provide a concise conclusion to our findings. 

---
# Extending Dense Passage Retrieval with Temporal Information 

**Authors**: Abdelrahman Abdallah, Bhawna Piryani, Jonas Wallat, Avishek Anand, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2502.21024)  

**Abstract**: Temporal awareness is crucial in many information retrieval tasks, particularly in scenarios where the relevance of documents depends on their alignment with the query's temporal context. Traditional retrieval methods such as BM25 and Dense Passage Retrieval (DPR) excel at capturing lexical and semantic relevance but fall short in addressing time-sensitive queries. To bridge this gap, we introduce the temporal retrieval model that integrates explicit temporal signals by incorporating query timestamps and document dates into the representation space. Our approach ensures that retrieved passages are not only topically relevant but also temporally aligned with user intent. We evaluate our approach on two large-scale benchmark datasets, ArchivalQA and ChroniclingAmericaQA, achieving substantial performance gains over standard retrieval baselines. In particular, our model improves Top-1 retrieval accuracy by 6.63% and NDCG@10 by 3.79% on ArchivalQA, while yielding a 9.56% boost in Top-1 retrieval accuracy and 4.68% in NDCG@10 on ChroniclingAmericaQA. Additionally, we introduce a time-sensitive negative sampling strategy, which refines the model's ability to distinguish between temporally relevant and irrelevant documents during training. Our findings highlight the importance of explicitly modeling time in retrieval systems and set a new standard for handling temporally grounded queries. 

---
# Variations in Relevance Judgments and the Shelf Life of Test Collections 

**Authors**: Andrew Parry, Maik Fröbe, Harrisen Scells, Ferdinand Schlatt, Guglielmo Faggioli, Saber Zerhoudi, Sean MacAvaney, Eugene Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20937)  

**Abstract**: The fundamental property of Cranfield-style evaluations, that system rankings are stable even when assessors disagree on individual relevance decisions, was validated on traditional test collections. However, the paradigm shift towards neural retrieval models affected the characteristics of modern test collections, e.g., documents are short, judged with four grades of relevance, and information needs have no descriptions or narratives. Under these changes, it is unclear whether assessor disagreement remains negligible for system comparisons. We investigate this aspect under the additional condition that the few modern test collections are heavily re-used. Given more possible query interpretations due to less formalized information needs, an ''expiration date'' for test collections might be needed if top-effectiveness requires overfitting to a single interpretation of relevance. We run a reproducibility study and re-annotate the relevance judgments of the 2019 TREC Deep Learning track. We can reproduce prior work in the neural retrieval setting, showing that assessor disagreement does not affect system rankings. However, we observe that some models substantially degrade with our new relevance judgments, and some have already reached the effectiveness of humans as rankers, providing evidence that test collections can expire. 

---
# Scalable Overload-Aware Graph-Based Index Construction for 10-Billion-Scale Vector Similarity Search 

**Authors**: Yang Shi, Yiping Sun, Jiaolong Du, Xiaocheng Zhong, Zhiyong Wang, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20695)  

**Abstract**: Approximate Nearest Neighbor Search (ANNS) is essential for modern data-driven applications that require efficient retrieval of top-k results from massive vector databases. Although existing graph-based ANNS algorithms achieve a high recall rate on billion-scale datasets, their slow construction speed and limited scalability hinder their applicability to large-scale industrial scenarios. In this paper, we introduce SOGAIC, the first Scalable Overload-Aware Graph-Based ANNS Index Construction system tailored for ultra-large-scale vector databases: 1) We propose a dynamic data partitioning algorithm with overload constraints that adaptively introduces overlaps among subsets; 2) To enable efficient distributed subgraph construction, we employ a load-balancing task scheduling framework combined with an agglomerative merging strategy; 3) Extensive experiments on various datasets demonstrate a reduction of 47.3% in average construction time compared to existing methods. The proposed method has also been successfully deployed in a real-world industrial search engine, managing over 10 billion daily updated vectors and serving hundreds of millions of users. 

---
# Unleashing the Potential of Two-Tower Models: Diffusion-Based Cross-Interaction for Large-Scale Matching 

**Authors**: Yihan Wang, Fei Xiong, Zhexin Han, Qi Song, Kaiqiao Zhan, Ben Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20687)  

**Abstract**: Two-tower models are widely adopted in the industrial-scale matching stage across a broad range of application domains, such as content recommendations, advertisement systems, and search engines. This model efficiently handles large-scale candidate item screening by separating user and item representations. However, the decoupling network also leads to a neglect of potential information interaction between the user and item representations. Current state-of-the-art (SOTA) approaches include adding a shallow fully connected layer(i.e., COLD), which is limited by performance and can only be used in the ranking stage. For performance considerations, another approach attempts to capture historical positive interaction information from the other tower by regarding them as the input features(i.e., DAT). Later research showed that the gains achieved by this method are still limited because of lacking the guidance on the next user intent. To address the aforementioned challenges, we propose a "cross-interaction decoupling architecture" within our matching paradigm. This user-tower architecture leverages a diffusion module to reconstruct the next positive intention representation and employs a mixed-attention module to facilitate comprehensive cross-interaction. During the next positive intention generation, we further enhance the accuracy of its reconstruction by explicitly extracting the temporal drift within user behavior sequences. Experiments on two real-world datasets and one industrial dataset demonstrate that our method outperforms the SOTA two-tower models significantly, and our diffusion approach outperforms other generative models in reconstructing item representations. 

---
# CS-PaperSum: A Large-Scale Dataset of AI-Generated Summaries for Scientific Papers 

**Authors**: Javin Liu, Aryan Vats, Zihao He  

**Link**: [PDF](https://arxiv.org/pdf/2502.20582)  

**Abstract**: The rapid expansion of scientific literature in computer science presents challenges in tracking research trends and extracting key insights. Existing datasets provide metadata but lack structured summaries that capture core contributions and methodologies. We introduce CS-PaperSum, a large-scale dataset of 91,919 papers from 31 top-tier computer science conferences, enriched with AI-generated structured summaries using ChatGPT. To assess summary quality, we conduct embedding alignment analysis and keyword overlap analysis, demonstrating strong preservation of key concepts. We further present a case study on AI research trends, highlighting shifts in methodologies and interdisciplinary crossovers, including the rise of self-supervised learning, retrieval-augmented generation, and multimodal AI. Our dataset enables automated literature analysis, research trend forecasting, and AI-driven scientific discovery, providing a valuable resource for researchers, policymakers, and scientific information retrieval systems. 

---
# Creator-Side Recommender System: Challenges, Designs, and Applications 

**Authors**: Xiaoshuang Chen, Yibo Wang, Yao Wang, Husheng Liu, Kaiqiao Zhan, Ben Wang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2502.20497)  

**Abstract**: Users and creators are two crucial components of recommender systems. Typical recommender systems focus on the user side, providing the most suitable items based on each user's request. In such scenarios, a few items receive a majority of exposures, while many items receive very few. This imbalance leads to poorer experiences and decreased activity among the creators receiving less feedback, harming the recommender system in the long term. To this end, we develop a creator-side recommender system, called DualRec, to answer the following question: how to find the most suitable users for each item to enhance the creators' experience? We show that typical user-side recommendation algorithms, such as retrieval and ranking algorithms, can be adapted into the creator-side versions with just a few modifications. This greatly simplifies algorithm design in DualRec. Moreover, we discuss a unique challenge in DualRec: the user availability issue, which is not present in user-side recommender systems. To tackle this issue, we incorporate a user availability calculation (UAC) module to effectively enhance DualRec's performance. DualRec has already been implemented in Kwai, a short video recommendation system with over 100 millions user and over 10 million creators, significantly improving the experience for creators. 

---
# Optimizing Large Language Models for ESG Activity Detection in Financial Texts 

**Authors**: Mattia Birti, Francesco Osborne, Andrea Maurino  

**Link**: [PDF](https://arxiv.org/pdf/2502.21112)  

**Abstract**: The integration of Environmental, Social, and Governance (ESG) factors into corporate decision-making is a fundamental aspect of sustainable finance. However, ensuring that business practices align with evolving regulatory frameworks remains a persistent challenge. AI-driven solutions for automatically assessing the alignment of sustainability reports and non-financial disclosures with specific ESG activities could greatly support this process. Yet, this task remains complex due to the limitations of general-purpose Large Language Models (LLMs) in domain-specific contexts and the scarcity of structured, high-quality datasets. In this paper, we investigate the ability of current-generation LLMs to identify text related to environmental activities. Furthermore, we demonstrate that their performance can be significantly enhanced through fine-tuning on a combination of original and synthetically generated data. To this end, we introduce ESG-Activities, a benchmark dataset containing 1,325 labelled text segments classified according to the EU ESG taxonomy. Our experimental results show that fine-tuning on ESG-Activities significantly enhances classification accuracy, with open models such as Llama 7B and Gemma 7B outperforming large proprietary solutions in specific configurations. These findings have important implications for financial analysts, policymakers, and AI researchers seeking to enhance ESG transparency and compliance through advanced natural language processing techniques. 

---
# Fast 3D point clouds retrieval for Large-scale 3D Place Recognition 

**Authors**: Chahine-Nicolas Zede, Laurent Carrafa, Valérie Gouet-Brunet  

**Link**: [PDF](https://arxiv.org/pdf/2502.21067)  

**Abstract**: Retrieval in 3D point clouds is a challenging task that consists in retrieving the most similar point clouds to a given query within a reference of 3D points. Current methods focus on comparing descriptors of point clouds in order to identify similar ones. Due to the complexity of this latter step, here we focus on the acceleration of the retrieval by adapting the Differentiable Search Index (DSI), a transformer-based approach initially designed for text information retrieval, for 3D point clouds retrieval. Our approach generates 1D identifiers based on the point descriptors, enabling direct retrieval in constant time. To adapt DSI to 3D data, we integrate Vision Transformers to map descriptors to these identifiers while incorporating positional and semantic encoding. The approach is evaluated for place recognition on a public benchmark comparing its retrieval capabilities against state-of-the-art methods, in terms of quality and speed of returned point clouds. 

---
# The RAG Paradox: A Black-Box Attack Exploiting Unintentional Vulnerabilities in Retrieval-Augmented Generation Systems 

**Authors**: Chanwoo Choi, Jinsoo Kim, Sukmin Cho, Soyeong Jeong, Buru Chang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20995)  

**Abstract**: With the growing adoption of retrieval-augmented generation (RAG) systems, recent studies have introduced attack methods aimed at degrading their performance. However, these methods rely on unrealistic white-box assumptions, such as attackers having access to RAG systems' internal processes. To address this issue, we introduce a realistic black-box attack scenario based on the RAG paradox, where RAG systems inadvertently expose vulnerabilities while attempting to enhance trustworthiness. Because RAG systems reference external documents during response generation, our attack targets these sources without requiring internal access. Our approach first identifies the external sources disclosed by RAG systems and then automatically generates poisoned documents with misinformation designed to match these sources. Finally, these poisoned documents are newly published on the disclosed sources, disrupting the RAG system's response generation process. Both offline and online experiments confirm that this attack significantly reduces RAG performance without requiring internal access. Furthermore, from an insider perspective within the RAG system, we propose a re-ranking method that acts as a fundamental safeguard, offering minimal protection against unforeseen attacks. 

---
# WebFAQ: A Multilingual Collection of Natural Q&A Datasets for Dense Retrieval 

**Authors**: Michael Dinzinger, Laura Caspari, Kanishka Ghosh Dastidar, Jelena Mitrović, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2502.20936)  

**Abstract**: We present WebFAQ, a large-scale collection of open-domain question answering datasets derived from FAQ-style this http URL annotations. In total, the data collection consists of 96 million natural question-answer (QA) pairs across 75 languages, including 47 million (49%) non-English samples. WebFAQ further serves as the foundation for 20 monolingual retrieval benchmarks with a total size of 11.2 million QA pairs (5.9 million non-English). These datasets are carefully curated through refined filtering and near-duplicate detection, yielding high-quality resources for training and evaluating multilingual dense retrieval models. To empirically confirm WebFAQ's efficacy, we use the collected QAs to fine-tune an in-domain pretrained XLM-RoBERTa model. Through this process of dataset-specific fine-tuning, the model achieves significant retrieval performance gains, which generalize - beyond WebFAQ - to other multilingual retrieval benchmarks evaluated in zero-shot setting. Last but not least, we utilize WebFAQ to construct a set of QA-aligned bilingual corpora spanning over 1000 language pairs using state-of-the-art bitext mining and automated LLM-assessed translation evaluation. Due to our advanced, automated method of bitext dataset generation, the resulting bilingual corpora demonstrate higher translation quality compared to similar datasets. WebFAQ and all associated resources are publicly available on GitHub and HuggingFace. 

---
# CoTMR: Chain-of-Thought Multi-Scale Reasoning for Training-Free Zero-Shot Composed Image Retrieval 

**Authors**: Zelong Sun, Dong Jing, Zhiwu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20826)  

**Abstract**: Zero-Shot Composed Image Retrieval (ZS-CIR) aims to retrieve target images by integrating information from a composed query (reference image and modification text) without training samples. Existing methods primarily combine caption models and large language models (LLMs) to generate target captions based on composed queries but face various issues such as incompatibility, visual information loss, and insufficient reasoning. In this work, we propose CoTMR, a training-free framework crafted for ZS-CIR with novel Chain-of-thought (CoT) and Multi-scale Reasoning. Instead of relying on caption models for modality transformation, CoTMR employs the Large Vision-Language Model (LVLM) to achieve unified understanding and reasoning for composed queries. To enhance the reasoning reliability, we devise CIRCoT, which guides the LVLM through a step-by-step inference process using predefined subtasks. Considering that existing approaches focus solely on global-level reasoning, our CoTMR incorporates multi-scale reasoning to achieve more comprehensive inference via fine-grained predictions about the presence or absence of key elements at the object scale. Further, we design a Multi-Grained Scoring (MGS) mechanism, which integrates CLIP similarity scores of the above reasoning outputs with candidate images to realize precise retrieval. Extensive experiments demonstrate that our CoTMR not only drastically outperforms previous methods across four prominent benchmarks but also offers appealing interpretability. 

---
# LexRAG: Benchmarking Retrieval-Augmented Generation in Multi-Turn Legal Consultation Conversation 

**Authors**: Haitao Li, Yifan Chen, Yiran Hu, Qingyao Ai, Junjie Chen, Xiaoyu Yang, Jianhui Yang, Yueyue Wu, Zeyang Liu, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20640)  

**Abstract**: Retrieval-augmented generation (RAG) has proven highly effective in improving large language models (LLMs) across various domains. However, there is no benchmark specifically designed to assess the effectiveness of RAG in the legal domain, which restricts progress in this area. To fill this gap, we propose LexRAG, the first benchmark to evaluate RAG systems for multi-turn legal consultations. LexRAG consists of 1,013 multi-turn dialogue samples and 17,228 candidate legal articles. Each sample is annotated by legal experts and consists of five rounds of progressive questioning. LexRAG includes two key tasks: (1) Conversational knowledge retrieval, requiring accurate retrieval of relevant legal articles based on multi-turn context. (2) Response generation, focusing on producing legally sound answers. To ensure reliable reproducibility, we develop LexiT, a legal RAG toolkit that provides a comprehensive implementation of RAG system components tailored for the legal domain. Additionally, we introduce an LLM-as-a-judge evaluation pipeline to enable detailed and effective assessment. Through experimental analysis of various LLMs and retrieval methods, we reveal the key limitations of existing RAG systems in handling legal consultation conversations. LexRAG establishes a new benchmark for the practical application of RAG systems in the legal domain, with its code and data available at this https URL. 

---
# NANOGPT: A Query-Driven Large Language Model Retrieval-Augmented Generation System for Nanotechnology Research 

**Authors**: Achuth Chandrasekhar, Omid Barati Farimani, Olabode T. Ajenifujah, Janghoon Ock, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2502.20541)  

**Abstract**: This paper presents the development and application of a Large Language Model Retrieval-Augmented Generation (LLM-RAG) system tailored for nanotechnology research. The system leverages the capabilities of a sophisticated language model to serve as an intelligent research assistant, enhancing the efficiency and comprehensiveness of literature reviews in the nanotechnology domain. Central to this LLM-RAG system is its advanced query backend retrieval mechanism, which integrates data from multiple reputable sources. The system retrieves relevant literature by utilizing Google Scholar's advanced search, and scraping open-access papers from Elsevier, Springer Nature, and ACS Publications. This multifaceted approach ensures a broad and diverse collection of up-to-date scholarly articles and papers. The proposed system demonstrates significant potential in aiding researchers by providing a streamlined, accurate, and exhaustive literature retrieval process, thereby accelerating research advancements in nanotechnology. The effectiveness of the LLM-RAG system is validated through rigorous testing, illustrating its capability to significantly reduce the time and effort required for comprehensive literature reviews, while maintaining high accuracy, query relevance and outperforming standard, publicly available LLMS. 

---
# Intelligence Test 

**Authors**: Jingtao Zhan, Jiahao Zhao, Jiayu Li, Yiqun Liu, Bo Zhang, Qingyao Ai, Jiaxin Mao, Hongning Wang, Min Zhang, Shaoping Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.18858)  

**Abstract**: How does intelligence emerge? We propose that intelligence is not a sudden gift or random occurrence, but rather a necessary trait for species to survive through Natural Selection. If a species passes the test of Natural Selection, it demonstrates the intelligence to survive in nature. Extending this perspective, we introduce Intelligence Test, a method to quantify the intelligence of any subject on any task. Like how species evolve by trial and error, Intelligence Test quantifies intelligence by the number of failed attempts before success. Fewer failures correspond to higher intelligence. When the expectation and variance of failure counts are both finite, it signals the achievement of an autonomous level of intelligence. Using Intelligence Test, we comprehensively evaluate existing AI systems. Our results show that while AI systems achieve a level of autonomy in simple tasks, they are still far from autonomous in more complex tasks, such as vision, search, recommendation, and language. While scaling model size might help, this would come at an astronomical cost. Projections suggest that achieving general autonomy would require unimaginable $10^{26}$ parameters. Even if Moore's Law continuously holds, such a parameter scale would take $70$ years. This staggering cost highlights the complexity of human tasks and the inadequacies of current AI. To further understand this phenomenon, we conduct a theoretical analysis. Our simulations suggest that human tasks possess a criticality property. As a result, autonomy requires a deep understanding of the task's underlying mechanisms. Current AI, however, does not fully grasp these mechanisms and instead relies on superficial mimicry, making it difficult to reach an autonomous level. We believe Intelligence Test can not only guide the future development of AI but also offer profound insights into the intelligence of humans ourselves. 

---
