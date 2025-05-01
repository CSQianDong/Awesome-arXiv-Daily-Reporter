# Learning Universal User Representations Leveraging Cross-domain User Intent at Snapchat 

**Authors**: Clark Mingxuan Ju, Leonardo Neves, Bhuvesh Kumar, Liam Collins, Tong Zhao, Yuwei Qiu, Qing Dou, Yang Zhou, Sohail Nizam, Rengim Ozturk, Yvette Liu, Sen Yang, Manish Malik, Neil Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.21838)  

**Abstract**: The development of powerful user representations is a key factor in the success of recommender systems (RecSys). Online platforms employ a range of RecSys techniques to personalize user experience across diverse in-app surfaces. User representations are often learned individually through user's historical interactions within each surface and user representations across different surfaces can be shared post-hoc as auxiliary features or additional retrieval sources. While effective, such schemes cannot directly encode collaborative filtering signals across different surfaces, hindering its capacity to discover complex relationships between user behaviors and preferences across the whole platform. To bridge this gap at Snapchat, we seek to conduct universal user modeling (UUM) across different in-app surfaces, learning general-purpose user representations which encode behaviors across surfaces. Instead of replacing domain-specific representations, UUM representations capture cross-domain trends, enriching existing representations with complementary information. This work discusses our efforts in developing initial UUM versions, practical challenges, technical choices and modeling and research directions with promising offline performance. Following successful A/B testing, UUM representations have been launched in production, powering multiple use cases and demonstrating their value. UUM embedding has been incorporated into (i) Long-form Video embedding-based retrieval, leading to 2.78% increase in Long-form Video Open Rate, (ii) Long-form Video L2 ranking, with 19.2% increase in Long-form Video View Time sum, (iii) Lens L2 ranking, leading to 1.76% increase in Lens play time, and (iv) Notification L2 ranking, with 0.87% increase in Notification Open Rate. 

---
# From Precision to Perception: User-Centred Evaluation of Keyword Extraction Algorithms for Internet-Scale Contextual Advertising 

**Authors**: Jingwen Cai, Sara Leckner, Johanna Björklund  

**Link**: [PDF](https://arxiv.org/pdf/2504.21667)  

**Abstract**: Keyword extraction is a foundational task in natural language processing, underpinning countless real-world applications. A salient example is contextual advertising, where keywords help predict the topical congruence between ads and their surrounding media contexts to enhance advertising effectiveness. Recent advances in artificial intelligence, particularly large language models, have improved keyword extraction capabilities but also introduced concerns about computational cost. Moreover, although the end-user experience is of vital importance, human evaluation of keyword extraction performances remains under-explored. This study provides a comparative evaluation of three prevalent keyword extraction algorithms that vary in complexity: TF-IDF, KeyBERT, and Llama 2. To evaluate their effectiveness, a mixed-methods approach is employed, combining quantitative benchmarking with qualitative assessments from 552 participants through three survey-based experiments. Findings indicate a slight user preference for KeyBERT, which offers a favourable balance between performance and computational efficiency compared to the other two algorithms. Despite a strong overall preference for gold-standard keywords, differences between the algorithmic outputs are not statistically significant, highlighting a long-overlooked gap between traditional precision-focused metrics and user-perceived algorithm efficiency. The study highlights the importance of user-centred evaluation methodologies and proposes analytical tools to support their implementation. 

---
# Efficient Conversational Search via Topical Locality in Dense Retrieval 

**Authors**: Cristina Ioana Muntean, Franco Maria Nardini, Raffaele Perego, Guido Rocchietti, Cosimo Rulli  

**Link**: [PDF](https://arxiv.org/pdf/2504.21507)  

**Abstract**: Pre-trained language models have been widely exploited to learn dense representations of documents and queries for information retrieval. While previous efforts have primarily focused on improving effectiveness and user satisfaction, response time remains a critical bottleneck of conversational search systems. To address this, we exploit the topical locality inherent in conversational queries, i.e., the tendency of queries within a conversation to focus on related topics. By leveraging query embedding similarities, we dynamically restrict the search space to semantically relevant document clusters, reducing computational complexity without compromising retrieval quality. We evaluate our approach on the TREC CAsT 2019 and 2020 datasets using multiple embedding models and vector indexes, achieving improvements in processing speed of up to 10.4X with little loss in performance (4.4X without any loss). Our results show that the proposed system effectively handles complex, multiturn queries with high precision and efficiency, offering a practical solution for real-time conversational search. 

---
# In a Few Words: Comparing Weak Supervision and LLMs for Short Query Intent Classification 

**Authors**: Daria Alexander, Arjen P. de Vries  

**Link**: [PDF](https://arxiv.org/pdf/2504.21398)  

**Abstract**: User intent classification is an important task in information retrieval. Previously, user intents were classified manually and automatically; the latter helped to avoid hand labelling of large datasets. Recent studies explored whether LLMs can reliably determine user intent. However, researchers have recognized the limitations of using generative LLMs for classification tasks. In this study, we empirically compare user intent classification into informational, navigational, and transactional categories, using weak supervision and LLMs. Specifically, we evaluate LLaMA-3.1-8B-Instruct and LLaMA-3.1-70B-Instruct for in-context learning and LLaMA-3.1-8B-Instruct for fine-tuning, comparing their performance to an established baseline classifier trained using weak supervision (ORCAS-I). Our results indicate that while LLMs outperform weak supervision in recall, they continue to struggle with precision, which shows the need for improved methods to balance both metrics effectively. 

---
# Enhancing New-item Fairness in Dynamic Recommender Systems 

**Authors**: Huizhong Guo, Zhu Sun, Dongxia Wang, Tianjun Wei, Jinfeng Li, Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21362)  

**Abstract**: New-items play a crucial role in recommender systems (RSs) for delivering fresh and engaging user experiences. However, traditional methods struggle to effectively recommend new-items due to their short exposure time and limited interaction records, especially in dynamic recommender systems (DRSs) where new-items get continuously introduced and users' preferences evolve over time. This leads to significant unfairness towards new-items, which could accumulate over the successive model updates, ultimately compromising the stability of the entire system. Therefore, we propose FairAgent, a reinforcement learning (RL)-based new-item fairness enhancement framework specifically designed for DRSs. It leverages knowledge distillation to extract collaborative signals from traditional models, retaining strong recommendation capabilities for old-items. In addition, FairAgent introduces a novel reward mechanism for recommendation tailored to the characteristics of DRSs, which consists of three components: 1) a new-item exploration reward to promote the exposure of dynamically introduced new-items, 2) a fairness reward to adapt to users' personalized fairness requirements for new-items, and 3) an accuracy reward which leverages users' dynamic feedback to enhance recommendation accuracy. Extensive experiments on three public datasets and backbone models demonstrate the superior performance of FairAgent. The results present that FairAgent can effectively boost new-item exposure, achieve personalized new-item fairness, while maintaining high recommendation accuracy. 

---
# A Framework for Elastic Adaptation of User Multiple Intents in Sequential Recommendation 

**Authors**: Zhikai Wang, Yanyan Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21270)  

**Abstract**: Recently, substantial research has been conducted on sequential recommendation, with the objective of forecasting the subsequent item by leveraging a user's historical sequence of interacted items. Prior studies employ both capsule networks and self-attention techniques to effectively capture diverse underlying intents within a user's interaction sequence, thereby achieving the most advanced performance in sequential recommendation. However, users could potentially form novel intents from fresh interactions as the lengths of user interaction sequences grow. Consequently, models need to be continually updated or even extended to adeptly encompass these emerging user intents, referred as incremental multi-intent sequential recommendation. % We refer to this problem as incremental multi-intent sequential recommendation, which has not yet been well investigated in the existing literature. In this paper, we propose an effective Incremental learning framework for user Multi-intent Adaptation in sequential recommendation called IMA, which augments the traditional fine-tuning strategy with the existing-intents retainer, new-intents detector, and projection-based intents trimmer to adaptively expand the model to accommodate user's new intents and prevent it from forgetting user's existing intents. Furthermore, we upgrade the IMA into an Elastic Multi-intent Adaptation (EMA) framework which can elastically remove inactive intents and compress user intent vectors under memory space limit. Extensive experiments on real-world datasets verify the effectiveness of the proposed IMA and EMA on incremental multi-intent sequential recommendation, compared with various baselines. 

---
# Don't Retrieve, Generate: Prompting LLMs for Synthetic Training Data in Dense Retrieval 

**Authors**: Aarush Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2504.21015)  

**Abstract**: Training effective dense retrieval models often relies on hard negative (HN) examples mined from the document corpus via methods like BM25 or cross-encoders (CE), processes that can be computationally demanding and require full corpus access. This paper introduces a different approach, an end-to-end pipeline where a Large Language Model (LLM) first generates a query from a passage, and then generates a hard negative example using \emph{only} that query text. This corpus-free negative generation contrasts with standard mining techniques. We evaluated this \textsc{LLM Query $\rightarrow$ LLM HN} approach against traditional \textsc{LLM Query $\rightarrow$ BM25 HN} and \textsc{LLM Query $\rightarrow$ CE HN} pipelines using E5-Base and GTE-Base models on several BEIR benchmark datasets. Our results show the proposed all-LLM pipeline achieves performance identical to both the BM25 and the computationally intensive CE baselines across nDCG@10, Precision@10, and Recall@100 metrics. This demonstrates that our corpus-free negative generation method matches the effectiveness of complex, corpus-dependent mining techniques, offering a potentially simpler and more efficient pathway for training high-performance retrievers without sacrificing results. We make the dataset including the queries and the hard-negatives for all three methods publicly available this https URL. 

---
# WebThinker: Empowering Large Reasoning Models with Deep Research Capability 

**Authors**: Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2504.21776)  

**Abstract**: Large reasoning models (LRMs), such as OpenAI-o1 and DeepSeek-R1, demonstrate impressive long-horizon reasoning capabilities. However, their reliance on static internal knowledge limits their performance on complex, knowledge-intensive tasks and hinders their ability to produce comprehensive research reports requiring synthesis of diverse web information. To address this, we propose \textbf{WebThinker}, a deep research agent that empowers LRMs to autonomously search the web, navigate web pages, and draft research reports during the reasoning process. WebThinker integrates a \textbf{Deep Web Explorer} module, enabling LRMs to dynamically search, navigate, and extract information from the web when encountering knowledge gaps. It also employs an \textbf{Autonomous Think-Search-and-Draft strategy}, allowing the model to seamlessly interleave reasoning, information gathering, and report writing in real time. To further enhance research tool utilization, we introduce an \textbf{RL-based training strategy} via iterative online Direct Preference Optimization (DPO). Extensive experiments on complex reasoning benchmarks (GPQA, GAIA, WebWalkerQA, HLE) and scientific report generation tasks (Glaive) demonstrate that WebThinker significantly outperforms existing methods and strong proprietary systems. Our approach enhances LRM reliability and applicability in complex scenarios, paving the way for more capable and versatile deep research systems. The code is available at this https URL. 

---
# Traceback of Poisoning Attacks to Retrieval-Augmented Generation 

**Authors**: Baolei Zhang, Haoran Xin, Minghong Fang, Zhuqing Liu, Biao Yi, Tong Li, Zheli Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21668)  

**Abstract**: Large language models (LLMs) integrated with retrieval-augmented generation (RAG) systems improve accuracy by leveraging external knowledge sources. However, recent research has revealed RAG's susceptibility to poisoning attacks, where the attacker injects poisoned texts into the knowledge database, leading to attacker-desired responses. Existing defenses, which predominantly focus on inference-time mitigation, have proven insufficient against sophisticated attacks. In this paper, we introduce RAGForensics, the first traceback system for RAG, designed to identify poisoned texts within the knowledge database that are responsible for the attacks. RAGForensics operates iteratively, first retrieving a subset of texts from the database and then utilizing a specially crafted prompt to guide an LLM in detecting potential poisoning texts. Empirical evaluations across multiple datasets demonstrate the effectiveness of RAGForensics against state-of-the-art poisoning attacks. This work pioneers the traceback of poisoned texts in RAG systems, providing a practical and promising defense mechanism to enhance their security. 

---
# RDF-Based Structured Quality Assessment Representation of Multilingual LLM Evaluations 

**Authors**: Jonas Gwozdz, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2504.21605)  

**Abstract**: Large Language Models (LLMs) increasingly serve as knowledge interfaces, yet systematically assessing their reliability with conflicting information remains difficult. We propose an RDF-based framework to assess multilingual LLM quality, focusing on knowledge conflicts. Our approach captures model responses across four distinct context conditions (complete, incomplete, conflicting, and no-context information) in German and English. This structured representation enables the comprehensive analysis of knowledge leakage-where models favor training data over provided context-error detection, and multilingual consistency. We demonstrate the framework through a fire safety domain experiment, revealing critical patterns in context prioritization and language-specific performance, and demonstrating that our vocabulary was sufficient to express every assessment facet encountered in the 28-question study. 

---
