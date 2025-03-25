# Exploring Training and Inference Scaling Laws in Generative Retrieval 

**Authors**: Hongru Cai, Yongqi Li, Ruifeng Yuan, Wenjie Wang, Zhen Zhang, Wenjie Li, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2503.18941)  

**Abstract**: Generative retrieval has emerged as a novel paradigm that leverages large language models (LLMs) to autoregressively generate document identifiers. Although promising, the mechanisms that underpin its performance and scalability remain largely unclear. We conduct a systematic investigation of training and inference scaling laws in generative retrieval, exploring how model size, training data scale, and inference-time compute jointly influence retrieval performance. To address the lack of suitable metrics, we propose a novel evaluation measure inspired by contrastive entropy and generation loss, providing a continuous performance signal that enables robust comparisons across diverse generative retrieval methods. Our experiments show that n-gram-based methods demonstrate strong alignment with both training and inference scaling laws, especially when paired with larger LLMs. Furthermore, increasing inference computation yields substantial performance gains, revealing that generative retrieval can significantly benefit from higher compute budgets at inference. Across these settings, LLaMA models consistently outperform T5 models, suggesting a particular advantage for larger decoder-only models in generative retrieval. Taken together, our findings underscore that model sizes, data availability, and inference computation interact to unlock the full potential of generative retrieval, offering new insights for designing and optimizing future systems. 

---
# CCMusic: An Open and Diverse Database for Chinese Music Information Retrieval Research 

**Authors**: Monan Zhou, Shenyang Xu, Zhaorui Liu, Zhaowen Wang, Feng Yu, Wei Li, Baoqiang Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.18802)  

**Abstract**: Data are crucial in various computer-related fields, including music information retrieval (MIR), an interdisciplinary area bridging computer science and music. This paper introduces CCMusic, an open and diverse database comprising multiple datasets specifically designed for tasks related to Chinese music, highlighting our focus on this culturally rich domain. The database integrates both published and unpublished datasets, with steps taken such as data cleaning, label refinement, and data structure unification to ensure data consistency and create ready-to-use versions. We conduct benchmark evaluations for all datasets using a unified evaluation framework developed specifically for this purpose. This publicly available framework supports both classification and detection tasks, ensuring standardized and reproducible results across all datasets. The database is hosted on HuggingFace and ModelScope, two open and multifunctional data and model hosting platforms, ensuring ease of accessibility and usability. 

---
# ArchSeek: Retrieving Architectural Case Studies Using Vision-Language Models 

**Authors**: Danrui Li, Yichao Shi, Yaluo Wang, Ziying Shi, Mubbasir Kapadia  

**Link**: [PDF](https://arxiv.org/pdf/2503.18680)  

**Abstract**: Efficiently searching for relevant case studies is critical in architectural design, as designers rely on precedent examples to guide or inspire their ongoing projects. However, traditional text-based search tools struggle to capture the inherently visual and complex nature of architectural knowledge, often leading to time-consuming and imprecise exploration. This paper introduces ArchSeek, an innovative case study search system with recommendation capability, tailored for architecture design professionals. Powered by the visual understanding capabilities from vision-language models and cross-modal embeddings, it enables text and image queries with fine-grained control, and interaction-based design case recommendations. It offers architects a more efficient, personalized way to discover design inspirations, with potential applications across other visually driven design fields. The source code is available at this https URL. 

---
# A Comprehensive Review on Hashtag Recommendation: From Traditional to Deep Learning and Beyond 

**Authors**: Shubhi Bansal, Kushaan Gowda, Anupama Sureshbabu K, Chirag Kothari, Nagendra Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.18669)  

**Abstract**: The exponential growth of user-generated content on social media platforms has precipitated significant challenges in information management, particularly in content organization, retrieval, and discovery. Hashtags, as a fundamental categorization mechanism, play a pivotal role in enhancing content visibility and user engagement. However, the development of accurate and robust hashtag recommendation systems remains a complex and evolving research challenge. Existing surveys in this domain are limited in scope and recency, focusing narrowly on specific platforms, methodologies, or timeframes. To address this gap, this review article conducts a systematic analysis of hashtag recommendation systems, comprehensively examining recent advancements across several dimensions. We investigate unimodal versus multimodal methodologies, diverse problem formulations, filtering strategies, methodological evolution from traditional frequency-based models to advanced deep learning architectures. Furthermore, we critically evaluate performance assessment paradigms, including quantitative metrics, qualitative analyses, and hybrid evaluation frameworks. Our analysis underscores a paradigm shift toward transformer-based deep learning models, which harness contextual and semantic features to achieve superior recommendation accuracy. Key challenges such as data sparsity, cold-start scenarios, polysemy, and model explainability are rigorously discussed, alongside practical applications in tweet classification, sentiment analysis, and content popularity prediction. By synthesizing insights from diverse methodological and platform-specific perspectives, this survey provides a structured taxonomy of current research, identifies unresolved gaps, and proposes future directions for developing adaptive, user-centric recommendation systems. 

---
# Dense Retrieval for Low Resource Languages -- the Case of Amharic Language 

**Authors**: Tilahun Yeshambel, Moncef Garouani, Serge Molina, Josiane Mothe  

**Link**: [PDF](https://arxiv.org/pdf/2503.18570)  

**Abstract**: This paper reports some difficulties and some results when using dense retrievers on Amharic, one of the low-resource languages spoken by 120 millions populations. The efforts put and difficulties faced by University Addis Ababa toward Amharic Information Retrieval will be developed during the presentation. 

---
# Robust-IR @ SIGIR 2025: The First Workshop on Robust Information Retrieval 

**Authors**: Yu-An Liu, Haya Nachimovsky, Ruqing Zhang, Oren Kurland, Jiafeng Guo, Moshe Tennenholtz  

**Link**: [PDF](https://arxiv.org/pdf/2503.18426)  

**Abstract**: With the advancement of information retrieval (IR) technologies, robustness is increasingly attracting attention. When deploying technology into practice, we consider not only its average performance under normal conditions but, more importantly, its ability to maintain functionality across a variety of exceptional situations. In recent years, the research on IR robustness covers theory, evaluation, methodology, and application, and all of them show a growing trend. The purpose of this workshop is to systematize the latest results of each research aspect, to foster comprehensive communication within this niche domain while also bridging robust IR research with the broader community, and to promote further future development of robust IR. To avoid the one-sided talk of mini-conferences, this workshop adopts a highly interactive format, including round-table and panel discussion sessions, to encourage active participation and meaningful exchange among attendees. 

---
# PRECTR: A Synergistic Framework for Integrating Personalized Search Relevance Matching and CTR Prediction 

**Authors**: Rong Chen, Shuzhi Cao, Ailong He, Shuguang Han, Jufeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.18395)  

**Abstract**: The two primary tasks in the search recommendation system are search relevance matching and click-through rate (CTR) prediction -- the former focuses on seeking relevant items for user queries whereas the latter forecasts which item may better match user interest. Prior research typically develops two models to predict the CTR and search relevance separately, then ranking candidate items based on the fusion of the two outputs. However, such a divide-and-conquer paradigm creates the inconsistency between different models. Meanwhile, the search relevance model mainly concentrates on the degree of objective text matching while neglecting personalized differences among different users, leading to restricted model performance. To tackle these issues, we propose a unified \textbf{P}ersonalized Search RElevance Matching and CTR Prediction Fusion Model(PRECTR). Specifically, based on the conditional probability fusion mechanism, PRECTR integrates the CTR prediction and search relevance matching into one framework to enhance the interaction and consistency of the two modules. However, directly optimizing CTR binary classification loss may bring challenges to the fusion model's convergence and indefinitely promote the exposure of items with high CTR, regardless of their search relevance. Hence, we further introduce two-stage training and semantic consistency regularization to accelerate the model's convergence and restrain the recommendation of irrelevant items. Finally, acknowledging that different users may have varied relevance preferences, we assessed current users' relevance preferences by analyzing past users' preferences for similar queries and tailored incentives for different candidate items accordingly. Extensive experimental results on our production dataset and online A/B testing demonstrate the effectiveness and superiority of our proposed PRECTR method. 

---
# Food Recommendation With Balancing Comfort and Curiosity 

**Authors**: Yuto Sakai, Qiang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.18355)  

**Abstract**: Food is a key pleasure of traveling, but travelers face a trade-off between exploring curious new local food and choosing comfortable, familiar options. This creates demand for personalized recommendation systems that balance these competing factors. To the best of our knowledge, conventional recommendation methods cannot provide recommendations that offer both curiosity and comfort for food unknown to the user at a travel destination. In this study, we propose new quantitative methods for estimating comfort and curiosity: Kernel Density Scoring (KDS) and Mahalanobis Distance Scoring (MDS). KDS probabilistically estimates food history distribution using kernel density estimation, while MDS uses Mahalanobis distances between foods. These methods score food based on how their representation vectors fit the estimated distributions. We also propose a ranking method measuring the balance between comfort and curiosity based on taste and ingredients. This balance is defined as curiosity (return) gained per unit of comfort (risk) in choosing a food. For evaluation the proposed method, we newly collected a dataset containing user surveys on Japanese food and assessments of foreign food regarding comfort and curiosity. Comparing our methods against the existing method, the Wilcoxon signed-rank test showed that when estimating comfort from taste and curiosity from ingredients, the MDS-based method outperformed the Baseline, while the KDS-based method showed no significant differences. When estimating curiosity from taste and comfort from ingredients, both methods outperformed the Baseline. The MDS-based method consistently outperformed KDS in ROC-AUC values. 

---
# RAU: Towards Regularized Alignment and Uniformity for Representation Learning in Recommendation 

**Authors**: Xi Wu, Dan Zhang, Chao Zhou, Liangwei Yang, Tianyu Lin, Jibing Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.18300)  

**Abstract**: Recommender systems (RecSys) have become essential in modern society, driving user engagement and satisfaction across diverse online platforms. Most RecSys focuses on designing a powerful encoder to embed users and items into high-dimensional vector representation space, with loss functions optimizing their representation distributions. Recent studies reveal that directly optimizing key properties of the representation distribution, such as alignment and uniformity, can outperform complex encoder designs. However, existing methods for optimizing critical attributes overlook the impact of dataset sparsity on the model: limited user-item interactions lead to sparse alignment, while excessive interactions result in uneven uniformity, both of which degrade performance. In this paper, we identify the sparse alignment and uneven uniformity issues, and further propose Regularized Alignment and Uniformity (RAU) to cope with these two issues accordingly. RAU consists of two novel regularization methods for alignment and uniformity to learn better user/item representation. 1) Center-strengthened alignment further aligns the average in-batch user/item representation to provide an enhanced alignment signal and further minimize the disparity between user and item representation. 2) Low-variance-guided uniformity minimizes the variance of pairwise distances along with uniformity, which provides extra guidance to a more stabilized uniformity increase during training. We conducted extensive experiments on three real-world datasets, and the proposed RAU resulted in significant performance improvements compared to current state-of-the-art CF methods, which confirms the advantages of the two proposed regularization methods. 

---
# Z-REx: Human-Interpretable GNN Explanations for Real Estate Recommendations 

**Authors**: Kunal Mukherjee, Zachary Harrison, Saeid Balaneshin  

**Link**: [PDF](https://arxiv.org/pdf/2503.18001)  

**Abstract**: Transparency and interpretability are crucial for enhancing customer confidence and user engagement, especially when dealing with black-box Machine Learning (ML)-based recommendation systems. Modern recommendation systems leverage Graph Neural Network (GNN) due to their ability to produce high-quality recommendations in terms of both relevance and diversity. Therefore, the explainability of GNN is especially important for Link Prediction (LP) tasks since recommending relevant items can be viewed as predicting links between users and items. GNN explainability has been a well-studied field, existing methods primarily focus on node or graph-level tasks, leaving a gap in LP explanation techniques.
This work introduces Z-REx, a GNN explanation framework designed explicitly for heterogeneous link prediction tasks. Z-REx utilizes structural and attribute perturbation to identify critical sub-structures and important features while reducing the search space by leveraging domain-specific knowledge. In our experimentation, we show the efficacy of Z-REx in generating contextually relevant and human-interpretable explanations for ZiGNN, a GNN-based recommendation engine, using a real-world real-estate dataset from Zillow Group, Inc. We also compare Z-REx to State-of-The-Art (SOTA) GNN explainers to show Z-REx's superiority in producing high-quality human-interpretable explanations. 

---
# SUNAR: Semantic Uncertainty based Neighborhood Aware Retrieval for Complex QA 

**Authors**: V Venktesh, Mandeep Rathee, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2503.17990)  

**Abstract**: Complex question-answering (QA) systems face significant challenges in retrieving and reasoning over information that addresses multi-faceted queries. While large language models (LLMs) have advanced the reasoning capabilities of these systems, the bounded-recall problem persists, where procuring all relevant documents in first-stage retrieval remains a challenge. Missing pertinent documents at this stage leads to performance degradation that cannot be remedied in later stages, especially given the limited context windows of LLMs which necessitate high recall at smaller retrieval depths. In this paper, we introduce SUNAR, a novel approach that leverages LLMs to guide a Neighborhood Aware Retrieval process. SUNAR iteratively explores a neighborhood graph of documents, dynamically promoting or penalizing documents based on uncertainty estimates from interim LLM-generated answer candidates. We validate our approach through extensive experiments on two complex QA datasets. Our results show that SUNAR significantly outperforms existing retrieve-and-reason baselines, achieving up to a 31.84% improvement in performance over existing state-of-the-art methods for complex QA. 

---
# Explainable identification of similarities between entities for discovery in large text 

**Authors**: Akhil Joshi, Sai Teja Erukude, Lior Shamir  

**Link**: [PDF](https://arxiv.org/pdf/2503.17605)  

**Abstract**: With the availability of virtually infinite number text documents in digital format, automatic comparison of textual data is essential for extracting meaningful insights that are difficult to identify manually. Many existing tools, including AI and large language models, struggle to provide precise and explainable insights into textual similarities. In many cases they determine the similarity between documents as reflected by the text, rather than the similarities between the subjects being discussed in these documents. This study addresses these limitations by developing an n-gram analysis framework designed to compare documents automatically and uncover explainable similarities. A scoring formula is applied to assigns each of the n-grams with a weight, where the weight is higher when the n-grams are more frequent in both documents, but is penalized when the n-grams are more frequent in the English language. Visualization tools like word clouds enhance the representation of these patterns, providing clearer insights. The findings demonstrate that this framework effectively uncovers similarities between text documents, offering explainable insights that are often difficult to identify manually. This non-parametric approach provides a deterministic solution for identifying similarities across various fields, including biographies, scientific literature, historical texts, and more. Code for the method is publicly available. 

---
# Dense Passage Retrieval in Conversational Search 

**Authors**: Ahmed H. Salamah, Pierre McWhannel, Nicole Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.17507)  

**Abstract**: Information retrieval systems have traditionally relied on exact term match methods such as BM25 for first-stage retrieval. However, recent advancements in neural network-based techniques have introduced a new method called dense retrieval. This approach uses a dual-encoder to create contextual embeddings that can be indexed and clustered efficiently at run-time, resulting in improved retrieval performance in Open-domain Question Answering systems. In this paper, we apply the dense retrieval technique to conversational search by conducting experiments on the CAsT benchmark dataset. We also propose an end-to-end conversational search system called GPT2QR+DPR, which incorporates various query reformulation strategies to improve retrieval accuracy. Our findings indicate that dense retrieval outperforms BM25 even without extensive fine-tuning. Our work contributes to the growing body of research on neural-based retrieval methods in conversational search, and highlights the potential of dense retrieval in improving retrieval accuracy in conversational search systems. 

---
# Toward building next-generation Geocoding systems: a systematic review 

**Authors**: Zhengcong Yin, Daniel W. Goldberg, Binbin Lin, Bing Zhou, Diya Li, Andong Ma, Ziqian Ming, Heng Cai, Zhe Zhang, Shaohua Wang, Shanzhen Gao, Joey Ying Lee, Xiao Li, Da Huo  

**Link**: [PDF](https://arxiv.org/pdf/2503.18888)  

**Abstract**: Geocoding systems are widely used in both scientific research for spatial analysis and everyday life through location-based services. The quality of geocoded data significantly impacts subsequent processes and applications, underscoring the need for next-generation systems. In response to this demand, this review first examines the evolving requirements for geocoding inputs and outputs across various scenarios these systems must address. It then provides a detailed analysis of how to construct such systems by breaking them down into key functional components and reviewing a broad spectrum of existing approaches, from traditional rule-based methods to advanced techniques in information retrieval, natural language processing, and large language models. Finally, we identify opportunities to improve next-generation geocoding systems in light of recent technological advances. 

---
# MammAlps: A multi-view video behavior monitoring dataset of wild mammals in the Swiss Alps 

**Authors**: Valentin Gabeff, Haozhe Qi, Brendan Flaherty, Gencer Sumbül, Alexander Mathis, Devis Tuia  

**Link**: [PDF](https://arxiv.org/pdf/2503.18223)  

**Abstract**: Monitoring wildlife is essential for ecology and ethology, especially in light of the increasing human impact on ecosystems. Camera traps have emerged as habitat-centric sensors enabling the study of wildlife populations at scale with minimal disturbance. However, the lack of annotated video datasets limits the development of powerful video understanding models needed to process the vast amount of fieldwork data collected. To advance research in wild animal behavior monitoring we present MammAlps, a multimodal and multi-view dataset of wildlife behavior monitoring from 9 camera-traps in the Swiss National Park. MammAlps contains over 14 hours of video with audio, 2D segmentation maps and 8.5 hours of individual tracks densely labeled for species and behavior. Based on 6135 single animal clips, we propose the first hierarchical and multimodal animal behavior recognition benchmark using audio, video and reference scene segmentation maps as inputs. Furthermore, we also propose a second ecology-oriented benchmark aiming at identifying activities, species, number of individuals and meteorological conditions from 397 multi-view and long-term ecological events, including false positive triggers. We advocate that both tasks are complementary and contribute to bridging the gap between machine learning and ecology. Code and data are available at: this https URL 

---
# Causality-Aware Next Location Prediction Framework based on Human Mobility Stratification 

**Authors**: Xiaojie Yang, Zipei Fan, Hangli Ge, Takashi Michikata, Ryosuke Shibasaki, Noboru Koshizuka  

**Link**: [PDF](https://arxiv.org/pdf/2503.18179)  

**Abstract**: Human mobility data are fused with multiple travel patterns and hidden spatiotemporal patterns are extracted by integrating user, location, and time information to improve next location prediction accuracy. In existing next location prediction methods, different causal relationships that result from patterns in human mobility data are ignored, which leads to confounding information that can have a negative effect on predictions. Therefore, this study introduces a causality-aware framework for next location prediction, focusing on human mobility stratification for travel patterns. In our research, a novel causal graph is developed that describes the relationships between various input variables. We use counterfactuals to enhance the indirect effects in our causal graph for specific travel patterns: non-anchor targeted travels. The proposed framework is designed as a plug-and-play module that integrates multiple next location prediction paradigms. We tested our proposed framework using several state-of-the-art models and human mobility datasets, and the results reveal that the proposed module improves the prediction performance. In addition, we provide results from the ablation study and quantitative study to demonstrate the soundness of our causal graph and its ability to further enhance the interpretability of the current next location prediction models. 

---
# GINGER: Grounded Information Nugget-Based Generation of Responses 

**Authors**: Weronika Łajewska, Krisztian Balog  

**Link**: [PDF](https://arxiv.org/pdf/2503.18174)  

**Abstract**: Retrieval-augmented generation (RAG) faces challenges related to factual correctness, source attribution, and response completeness. To address them, we propose a modular pipeline for grounded response generation that operates on information nuggets-minimal, atomic units of relevant information extracted from retrieved documents. The multistage pipeline encompasses nugget detection, clustering, ranking, top cluster summarization, and fluency enhancement. It guarantees grounding in specific facts, facilitates source attribution, and ensures maximum information inclusion within length constraints. Extensive experiments on the TREC RAG'24 dataset evaluated with the AutoNuggetizer framework demonstrate that GINGER achieves state-of-the-art performance on this benchmark. 

---
# BERTDetect: A Neural Topic Modelling Approach for Android Malware Detection 

**Authors**: Nishavi Ranaweera, Jiarui Xu, Suranga Seneviratne, Aruna Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2503.18043)  

**Abstract**: Web access today occurs predominantly through mobile devices, with Android representing a significant share of the mobile device market. This widespread usage makes Android a prime target for malicious attacks. Despite efforts to combat malicious attacks through tools like Google Play Protect and antivirus software, new and evolved malware continues to infiltrate Android devices. Source code analysis is effective but limited, as attackers quickly abandon old malware for new variants to evade detection. Therefore, there is a need for alternative methods that complement source code analysis. Prior research investigated clustering applications based on their descriptions and identified outliers in these clusters by API usage as malware. However, these works often used traditional techniques such as Latent Dirichlet Allocation (LDA) and k-means clustering, that do not capture the nuanced semantic structures present in app descriptions. To this end, in this paper, we propose BERTDetect, which leverages the BERTopic neural topic modelling to effectively capture the latent topics in app descriptions. The resulting topic clusters are comparatively more coherent than previous methods and represent the app functionalities well. Our results demonstrate that BERTDetect outperforms other baselines, achieving ~10% relative improvement in F1 score. 

---
# Experience Retrieval-Augmentation with Electronic Health Records Enables Accurate Discharge QA 

**Authors**: Justice Ou, Tinglin Huang, Yilun Zhao, Ziyang Yu, Peiqing Lu, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2503.17933)  

**Abstract**: To improve the reliability of Large Language Models (LLMs) in clinical applications, retrieval-augmented generation (RAG) is extensively applied to provide factual medical knowledge. However, beyond general medical knowledge from open-ended datasets, clinical case-based knowledge is also critical for effective medical reasoning, as it provides context grounded in real-world patient experiences. Motivated by this, we propose Experience Retrieval Augmentation - ExpRAG framework based on Electronic Health Record (EHR), aiming to offer the relevant context from other patients' discharge reports. ExpRAG performs retrieval through a coarse-to-fine process, utilizing an EHR-based report ranker to efficiently identify similar patients, followed by an experience retriever to extract task-relevant content for enhanced medical reasoning. To evaluate ExpRAG, we introduce DischargeQA, a clinical QA dataset with 1,280 discharge-related questions across diagnosis, medication, and instruction tasks. Each problem is generated using EHR data to ensure realistic and challenging scenarios. Experimental results demonstrate that ExpRAG consistently outperforms a text-based ranker, achieving an average relative improvement of 5.2%, highlighting the importance of case-based knowledge for medical reasoning. 

---
# Satisfactory Medical Consultation based on Terminology-Enhanced Information Retrieval and Emotional In-Context Learning 

**Authors**: Kaiwen Zuo, Jing Tang, Hanbing Qin, Binli Luo, Ligang He, Shiyan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17876)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have marked significant progress in understanding and responding to medical inquiries. However, their performance still falls short of the standards set by professional consultations. This paper introduces a novel framework for medical consultation, comprising two main modules: Terminology-Enhanced Information Retrieval (TEIR) and Emotional In-Context Learning (EICL). TEIR ensures implicit reasoning through the utilization of inductive knowledge and key terminology retrieval, overcoming the limitations of restricted domain knowledge in public databases. Additionally, this module features capabilities for processing long context. The EICL module aids in generating sentences with high attribute relevance by memorizing semantic and attribute information from unlabelled corpora and applying controlled retrieval for the required information. Furthermore, a dataset comprising 803,564 consultation records was compiled in China, significantly enhancing the model's capability for complex dialogues and proactive inquiry initiation. Comprehensive experiments demonstrate the proposed method's effectiveness in extending the context window length of existing LLMs. The experimental outcomes and extensive data validate the framework's superiority over five baseline models in terms of BLEU and ROUGE performance metrics, with substantial leads in certain capabilities. Notably, ablation studies confirm the significance of the TEIR and EICL components. In addition, our new framework has the potential to significantly improve patient satisfaction in real clinical consulting situations. 

---
# Beyond Negation Detection: Comprehensive Assertion Detection Models for Clinical NLP 

**Authors**: Veysel Kocaman, Yigit Gul, M. Aytug Kaya, Hasham Ul Haq, Mehmet Butgul, Cabir Celik, David Talby  

**Link**: [PDF](https://arxiv.org/pdf/2503.17425)  

**Abstract**: Assertion status detection is a critical yet often overlooked component of clinical NLP, essential for accurately attributing extracted medical facts. Past studies have narrowly focused on negation detection, leading to underperforming commercial solutions such as AWS Medical Comprehend, Azure AI Text Analytics, and GPT-4o due to their limited domain adaptation. To address this gap, we developed state-of-the-art assertion detection models, including fine-tuned LLMs, transformer-based classifiers, few-shot classifiers, and deep learning (DL) approaches. We evaluated these models against cloud-based commercial API solutions, the legacy rule-based NegEx approach, and GPT-4o. Our fine-tuned LLM achieves the highest overall accuracy (0.962), outperforming GPT-4o (0.901) and commercial APIs by a notable margin, particularly excelling in Present (+4.2%), Absent (+8.4%), and Hypothetical (+23.4%) assertions. Our DL-based models surpass commercial solutions in Conditional (+5.3%) and Associated-with-Someone-Else (+10.1%) categories, while the few-shot classifier offers a lightweight yet highly competitive alternative (0.929), making it ideal for resource-constrained environments. Integrated within Spark NLP, our models consistently outperform black-box commercial solutions while enabling scalable inference and seamless integration with medical NER, Relation Extraction, and Terminology Resolution. These results reinforce the importance of domain-adapted, transparent, and customizable clinical NLP solutions over general-purpose LLMs and proprietary APIs. 

---
# Enhancing Subsequent Video Retrieval via Vision-Language Models (VLMs) 

**Authors**: Yicheng Duan, Xi Huang, Duo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17415)  

**Abstract**: The rapid growth of video content demands efficient and precise retrieval systems. While vision-language models (VLMs) excel in representation learning, they often struggle with adaptive, time-sensitive video retrieval. This paper introduces a novel framework that combines vector similarity search with graph-based data structures. By leveraging VLM embeddings for initial retrieval and modeling contextual relationships among video segments, our approach enables adaptive query refinement and improves retrieval accuracy. Experiments demonstrate its precision, scalability, and robustness, offering an effective solution for interactive video retrieval in dynamic environments. 

---
