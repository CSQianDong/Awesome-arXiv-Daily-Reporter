# Knowledge Graph Completion with Relation-Aware Anchor Enhancement 

**Authors**: Duanyang Yuan, Sihang Zhou, Xiaoshu Chen, Dong Wang, Ke Liang, Xinwang Liu, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06129)  

**Abstract**: Text-based knowledge graph completion methods take advantage of pre-trained language models (PLM) to enhance intrinsic semantic connections of raw triplets with detailed text descriptions. Typical methods in this branch map an input query (textual descriptions associated with an entity and a relation) and its candidate entities into feature vectors, respectively, and then maximize the probability of valid triples. These methods are gaining promising performance and increasing attention for the rapid development of large language models. According to the property of the language models, the more related and specific context information the input query provides, the more discriminative the resultant embedding will be. In this paper, through observation and validation, we find a neglected fact that the relation-aware neighbors of the head entities in queries could act as effective contexts for more precise link prediction. Driven by this finding, we propose a relation-aware anchor enhanced knowledge graph completion method (RAA-KGC). Specifically, in our method, to provide a reference of what might the target entity be like, we first generate anchor entities within the relation-aware neighborhood of the head entity. Then, by pulling the query embedding towards the neighborhoods of the anchors, it is tuned to be more discriminative for target entity matching. The results of our extensive experiments not only validate the efficacy of RAA-KGC but also reveal that by integrating our relation-aware anchor enhancement strategy, the performance of current leading methods can be notably enhanced without substantial modifications. 

---
# Widening the Role of Group Recommender Systems with CAJO 

**Authors**: Francesco Ricci, Amra Delić  

**Link**: [PDF](https://arxiv.org/pdf/2504.05934)  

**Abstract**: Group Recommender Systems (GRSs) have been studied and developed for more than twenty years. However, their application and usage has not grown. They can even be labeled as failures, if compared to the very successful and common recommender systems (RSs) used on all the major ecommerce and social platforms. As a result, the RSs that we all use now, are only targeted for individual users, aiming at choosing an item exclusively for themselves; no choice support is provided to groups trying to select a service, a product, an experience, a person, serving equally well all the group members. In this opinion article we discuss why the success of group recommender systems is lagging and we propose a research program unfolding on the analysis and development of new forms of collaboration between humans and intelligent systems. We define a set of roles, named CAJO, that GRSs should play in order to become more useful tools for group decision making. 

---
# PathGPT: Leveraging Large Language Models for Personalized Route Generation 

**Authors**: Steeve Cuthbert Marcelyn, Yucen Gao, Yuzhe Zhang, Xiaofeng Gao, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.05846)  

**Abstract**: The proliferation of GPS enabled devices has led to the accumulation of a substantial corpus of historical trajectory data. By leveraging these data for training machine learning models,researchers have devised novel data-driven methodologies that address the personalized route recommendation (PRR) problem. In contrast to conventional algorithms such as Dijkstra shortest path algorithm,these novel algorithms possess the capacity to discern and learn patterns within the data,thereby facilitating the generation of more personalized paths. However,once these models have been trained,their application is constrained to the generation of routes that align with their training patterns. This limitation renders them less adaptable to novel scenarios and the deployment of multiple machine learning models might be necessary to address new possible scenarios,which can be costly as each model must be trained separately. Inspired by recent advances in the field of Large Language Models (LLMs),we leveraged their natural language understanding capabilities to develop a unified model to solve the PRR problem while being seamlessly adaptable to new scenarios without additional training. To accomplish this,we combined the extensive knowledge LLMs acquired during training with further access to external hand-crafted context information,similar to RAG (Retrieved Augmented Generation) systems,to enhance their ability to generate paths according to user-defined requirements. Extensive experiments on different datasets show a considerable uplift in LLM performance on the PRR problem. 

---
# Why is Normalization Necessary for Linear Recommenders? 

**Authors**: Seongmin Park, Mincheol Yoon, Hye-young Kim, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.05805)  

**Abstract**: Despite their simplicity, linear autoencoder (LAE)-based models have shown comparable or even better performance with faster inference speed than neural recommender models. However, LAEs face two critical challenges: (i) popularity bias, which tends to recommend popular items, and (ii) neighborhood bias, which overly focuses on capturing local item correlations. To address these issues, this paper first analyzes the effect of two existing normalization methods for LAEs, i.e., random-walk and symmetric normalization. Our theoretical analysis reveals that normalization highly affects the degree of popularity and neighborhood biases among items. Inspired by this analysis, we propose a versatile normalization solution, called Data-Adaptive Normalization (DAN), which flexibly controls the popularity and neighborhood biases by adjusting item- and user-side normalization to align with unique dataset characteristics. Owing to its model-agnostic property, DAN can be easily applied to various LAE-based models. Experimental results show that DAN-equipped LAEs consistently improve existing LAE-based models across six benchmark datasets, with significant gains of up to 128.57% and 12.36% for long-tail items and unbiased evaluations, respectively. Refer to our code in this https URL. 

---
# StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization 

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05804)  

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems. 

---
# Retrieval Augmented Generation with Collaborative Filtering for Personalized Text Generation 

**Authors**: Teng Shi, Jun Xu, Xiao Zhang, Xiaoxue Zang, Kai Zheng, Yang Song, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.05731)  

**Abstract**: Recently, the personalization of Large Language Models (LLMs) to generate content that aligns with individual user preferences has garnered widespread attention. Personalized Retrieval-Augmented Generation (RAG), which retrieves relevant documents from the user's history to reflect their preferences and enhance LLM generation, is one commonly used approach for personalization. However, existing personalized RAG methods do not consider that the histories of similar users can also assist in personalized generation for the current user, meaning that collaborative information between users can also benefit personalized generation. Inspired by the application of collaborative filtering in recommender systems, we propose a method called CFRAG, which adapts Collaborative Filtering to RAG for personalized text generation. However, this presents two challenges: (1)~how to incorporate collaborative information without explicit user similarity labels? (2)~how to retrieve documents that support personalized LLM generation? For Challenge 1, we use contrastive learning to train user embeddings to retrieve similar users and introduce collaborative information. For Challenge 2, we design a personalized retriever and reranker to retrieve the top-$k$ documents from these users' histories. We take into account the user's preference during retrieval and reranking. Then we leverage feedback from the LLM to fine-tune the personalized retriever and reranker, enabling them to retrieve documents that meet the personalized generation needs of the LLM. Experimental results on the Language Model Personalization (LaMP) benchmark validate the effectiveness of CFRAG. Further analysis confirms the importance of incorporating collaborative information. 

---
# Unified Generative Search and Recommendation 

**Authors**: Teng Shi, Jun Xu, Xiao Zhang, Xiaoxue Zang, Kai Zheng, Yang Song, Enyun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05730)  

**Abstract**: Modern commercial platforms typically offer both search and recommendation functionalities to serve diverse user needs, making joint modeling of these tasks an appealing direction. While prior work has shown that integrating search and recommendation can be mutually beneficial, it also reveals a performance trade-off: enhancements in one task often come at the expense of the other. This challenge arises from their distinct information requirements: search emphasizes semantic relevance between queries and items, whereas recommendation depends more on collaborative signals among users and items. Effectively addressing this trade-off requires tackling two key problems: (1) integrating both semantic and collaborative signals into item representations, and (2) guiding the model to distinguish and adapt to the unique demands of search and recommendation. The emergence of generative retrieval with Large Language Models (LLMs) presents new possibilities. This paradigm encodes items as identifiers and frames both search and recommendation as sequential generation tasks, offering the flexibility to leverage multiple identifiers and task-specific prompts. In light of this, we introduce GenSAR, a unified generative framework for balanced search and recommendation. Our approach designs dual-purpose identifiers and tailored training strategies to incorporate complementary signals and align with task-specific objectives. Experiments on both public and commercial datasets demonstrate that GenSAR effectively reduces the trade-off and achieves state-of-the-art performance on both tasks. 

---
# Large Language Models Enhanced Hyperbolic Space Recommender Systems 

**Authors**: Wentao Cheng, Zhida Qin, Zexue Wu, Pengzhan Zhou, Tianyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05694)  

**Abstract**: Large Language Models (LLMs) have attracted significant attention in recommender systems for their excellent world knowledge capabilities. However, existing methods that rely on Euclidean space struggle to capture the rich hierarchical information inherent in textual and semantic data, which is essential for capturing user preferences. The geometric properties of hyperbolic space offer a promising solution to address this issue. Nevertheless, integrating LLMs-based methods with hyperbolic space to effectively extract and incorporate diverse hierarchical information is non-trivial. To this end, we propose a model-agnostic framework, named HyperLLM, which extracts and integrates hierarchical information from both structural and semantic perspectives. Structurally, HyperLLM uses LLMs to generate multi-level classification tags with hierarchical parent-child relationships for each item. Then, tag-item and user-item interactions are jointly learned and aligned through contrastive learning, thereby providing the model with clear hierarchical information. Semantically, HyperLLM introduces a novel meta-optimized strategy to extract hierarchical information from semantic embeddings and bridge the gap between the semantic and collaborative spaces for seamless integration. Extensive experiments show that HyperLLM significantly outperforms recommender systems based on hyperbolic space and LLMs, achieving performance improvements of over 40%. Furthermore, HyperLLM not only improves recommender performance but also enhances training stability, highlighting the critical role of hierarchical information in recommender systems. 

---
# xMTF: A Formula-Free Model for Reinforcement-Learning-Based Multi-Task Fusion in Recommender Systems 

**Authors**: Yang Cao, Changhao Zhang, Xiaoshuang Chen, Kaiqiao Zhan, Ben Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05669)  

**Abstract**: Recommender systems need to optimize various types of user feedback, e.g., clicks, likes, and shares. A typical recommender system handling multiple types of feedback has two components: a multi-task learning (MTL) module, predicting feedback such as click-through rate and like rate; and a multi-task fusion (MTF) module, integrating these predictions into a single score for item ranking. MTF is essential for ensuring user satisfaction, as it directly influences recommendation outcomes. Recently, reinforcement learning (RL) has been applied to MTF tasks to improve long-term user satisfaction. However, existing RL-based MTF methods are formula-based methods, which only adjust limited coefficients within pre-defined formulas. The pre-defined formulas restrict the RL search space and become a bottleneck for MTF. To overcome this, we propose a formula-free MTF framework. We demonstrate that any suitable fusion function can be expressed as a composition of single-variable monotonic functions, as per the Sprecher Representation Theorem. Leveraging this, we introduce a novel learnable monotonic fusion cell (MFC) to replace pre-defined formulas. We call this new MFC-based model eXtreme MTF (xMTF). Furthermore, we employ a two-stage hybrid (TSH) learning strategy to train xMTF effectively. By expanding the MTF search space, xMTF outperforms existing methods in extensive offline and online experiments. 

---
# Stratified Expert Cloning with Adaptive Selection for User Retention in Large-Scale Recommender Systems 

**Authors**: Chengzhi Lin, Annan Xie, Shuchang Liu, Wuhong Wang, Chuyuan Wang, Yongqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05628)  

**Abstract**: User retention has emerged as a critical challenge in large-scale recommender systems, significantly impacting the long-term success of online platforms. Existing methods often focus on short-term engagement metrics, failing to capture the complex dynamics of user preferences and behaviors over extended periods. While reinforcement learning (RL) approaches have shown promise in optimizing long-term rewards, they face difficulties in credit assignment, sample efficiency, and exploration when applied to the user retention problem. In this work, we propose Stratified Expert Cloning (SEC), a novel imitation learning framework that effectively leverages abundant logged data from high-retention users to learn robust recommendation policies. SEC introduces three key innovations: 1) a multi-level expert stratification strategy that captures the nuances in expert user behaviors at different retention levels; 2) an adaptive expert selection mechanism that dynamically assigns users to the most suitable policy based on their current state and historical retention level; and 3) an action entropy regularization technique that promotes recommendation diversity and mitigates the risk of policy collapse. Through extensive offline experiments and online A/B tests on two major video platforms, Kuaishou and Kuaishou Lite, with hundreds of millions of daily active users, we demonstrate SEC's significant improvements over state-of-the-art methods in user retention. The results demonstrate significant improvements in user retention, with cumulative lifts of 0.098\% and 0.122\% in active days on Kuaishou and Kuaishou Lite respectively, additionally bringing tens of thousands of daily active users to each platform. 

---
# User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems 

**Authors**: Jianling Wang, Yifan Liu, Yinghao Sun, Xuejian Ma, Yueqi Wang, He Ma, Steven Su, Ed H. Chi, Minmin Chen, Lichan Hong, Ningren Han, Haokai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05522)  

**Abstract**: Exploration, the act of broadening user experiences beyond their established preferences, is challenging in large-scale recommendation systems due to feedback loops and limited signals on user exploration patterns. Large Language Models (LLMs) offer potential by leveraging their world knowledge to recommend novel content outside these loops. A key challenge is aligning LLMs with user preferences while preserving their knowledge and reasoning. While using LLMs to plan for the next novel user interest, this paper introduces a novel approach combining hierarchical planning with LLM inference-time scaling to improve recommendation relevancy without compromising novelty. We decouple novelty and user-alignment, training separate LLMs for each objective. We then scale up the novelty-focused LLM's inference and select the best-of-n predictions using the user-aligned LLM. Live experiments demonstrate efficacy, showing significant gains in both user satisfaction (measured by watch activity and active user counts) and exploration diversity. 

---
# Hybrid Retrieval for Hallucination Mitigation in Large Language Models: A Comparative Analysis 

**Authors**: Chandana Sree Mala, Gizem Gezici, Fosca Giannotti  

**Link**: [PDF](https://arxiv.org/pdf/2504.05324)  

**Abstract**: Large Language Models (LLMs) excel in language comprehension and generation but are prone to hallucinations, producing factually incorrect or unsupported outputs. Retrieval Augmented Generation (RAG) systems address this issue by grounding LLM responses with external knowledge. This study evaluates the relationship between retriever effectiveness and hallucination reduction in LLMs using three retrieval approaches: sparse retrieval based on BM25 keyword search, dense retrieval using semantic search with Sentence Transformers, and a proposed hybrid retrieval module. The hybrid module incorporates query expansion and combines the results of sparse and dense retrievers through a dynamically weighted Reciprocal Rank Fusion score. Using the HaluBench dataset, a benchmark for hallucinations in question answering tasks, we assess retrieval performance with metrics such as mean average precision and normalised discounted cumulative gain, focusing on the relevance of the top three retrieved documents. Results show that the hybrid retriever achieves better relevance scores, outperforming both sparse and dense retrievers. Further evaluation of LLM-generated answers against ground truth using metrics such as accuracy, hallucination rate, and rejection rate reveals that the hybrid retriever achieves the highest accuracy on fails, the lowest hallucination rate, and the lowest rejection rate. These findings highlight the hybrid retriever's ability to enhance retrieval relevance, reduce hallucination rates, and improve LLM reliability, emphasising the importance of advanced retrieval techniques in mitigating hallucinations and improving response accuracy. 

---
# Multi-Perspective Attention Mechanism for Bias-Aware Sequential Recommendation 

**Authors**: Mingjian Fu, Hengsheng Chen, Dongchun Jiang, Yanchao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05323)  

**Abstract**: In the era of advancing information technology, recommender systems have emerged as crucial tools for dealing with information overload. However, traditional recommender systems still have limitations in capturing the dynamic evolution of user behavior. To better understand and predict user behavior, especially taking into account the complexity of temporal evolution, sequential recommender systems have gradually become the focus of research. Currently, many sequential recommendation algorithms ignore the amplification effects of prevalent biases, which leads to recommendation results being susceptible to the Matthew Effect. Additionally, it will impose limitations on the recommender system's ability to deeply perceive and capture the dynamic shifts in user preferences, thereby diminishing the extent of its recommendation reach. To address this issue effectively, we propose a recommendation system based on sequential information and attention mechanism called Multi-Perspective Attention Bias Sequential Recommendation (MABSRec). Firstly, we reconstruct user sequences into three short types and utilize graph neural networks for item weighting. Subsequently, an adaptive multi-bias perspective attention module is proposed to enhance the accuracy of recommendations. Experimental results show that the MABSRec model exhibits significant advantages in all evaluation metrics, demonstrating its excellent performance in the sequence recommendation task. 

---
# Balancing Benefits and Risks: RL Approaches for Addiction-Aware Social Media Recommenders 

**Authors**: Luca Bolis, Stefano Livella, Sabrina Patania, Dimitri Ognibene, Matteo Papini, Kenji Morita  

**Link**: [PDF](https://arxiv.org/pdf/2504.05322)  

**Abstract**: Social media platforms provide valuable opportunities for users to gather information, interact with friends, and enjoy entertainment. However, their addictive potential poses significant challenges, including overuse and negative psycho-logical or behavioral impacts [4, 2, 8]. This study explores strategies to mitigate compulsive social media usage while preserving its benefits and ensuring economic sustainability, focusing on recommenders that promote balanced usage.
We analyze user behaviors arising from intrinsic diversities and environmental interactions, offering insights for next-generation social media recommenders that prioritize well-being. Specifically, we examine the temporal predictability of overuse and addiction using measures available to recommenders, aiming to inform mechanisms that prevent addiction while avoiding user disengagement [7].
Building on RL-based computational frameworks for addiction modelling [6], our study introduces: - A recommender system adapting to user preferences, introducing non-stationary and non-Markovian dynamics.
- Differentiated state representations for users and recommenders to capture nuanced interactions.
- Distinct usage conditions-light and heavy use-addressing RL's limitations in distinguishing prolonged from healthy engagement.
- Complexity in overuse impacts, highlighting their role in user adaptation [7].
Simulations demonstrate how model-based (MB) and model-free (MF) decision-making interact with environmental dynamics to influence user behavior and addiction. Results reveal the significant role of recommender systems in shaping addiction tendencies or fostering healthier engagement. These findings support ethical, adaptive recommender design, advancing sustainable social media ecosystems [9, 1].
Keywords: multi-agent systems, recommender systems, addiction, social media 

---
# VALUE: Value-Aware Large Language Model for Query Rewriting via Weighted Trie in Sponsored Search 

**Authors**: Boyang Zuo, Xiao Zhang, Feng Li, Pengjie Wang, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05321)  

**Abstract**: In the realm of sponsored search advertising, matching advertisements with the search intent of a user's query is crucial. Query-to-bidwords(i.e. bidding keywords) rewriting is a vital technique that has garnered significant attention. Recently, with the prevalence of LLMs, generative retrieval methods have proven effective in producing high-relevance rewrites. However, we have identified a significant limitation in existing approaches: While fine-tuning LLMs for specific domains enhances semantic relevance, these models have no perception of the intrinsic value of their generated outputs, such as commercial value. Therefore, after SFT, a RLHF phase is often employed to address this issue. Nevertheless, traditional preference alignment methods often face challenges in aligning fine-grained values and are susceptible to overfitting, which diminishes the effectiveness and quality of the generated results. To address these challenges, we propose VALUE(Value-Aware Large language model for qUery rewriting via wEighted trie), the first framework that ensures the generation of high-value and highly relevant bidwords. Our approach utilizes weighted trie, an innovative modification of the traditional trie data structure. By modulating the LLM's output probability distribution with value information from the trie during decoding process, we constrain the generation space and guide the trajectory of text production. Offline experiments demonstrate the effectiveness of our method in semantic matching and preference alignment, showing a remarkable improvement in the value attribute by more than fivefold. Online A/B tests further revealed that our Revenue Per Mille (RPM) metric increased by 1.64%. VALUE has been deployed on our advertising system since October 2024 and served the Double Eleven promotions, the biggest shopping carnival in China. 

---
# Document clustering with evolved multiword search queries 

**Authors**: Laurence Hirsch, Robin Hirsch, Bayode Ogunleye  

**Link**: [PDF](https://arxiv.org/pdf/2504.05320)  

**Abstract**: Text clustering holds significant value across various domains due to its ability to identify patterns and group related information. Current approaches which rely heavily on a computed similarity measure between documents are often limited in accuracy and interpretability. We present a novel approach to the problem based on a set of evolved search queries. Clusters are formed as the set of documents matched by a single search query in the set of queries. The queries are optimized to maximize the number of documents returned and to minimize the overlap between clusters (documents returned by more than one query). Where queries contain more than one word they are interpreted disjunctively. We have found it useful to assign one word to be the root and constrain the query construction such that the set of documents returned by any additional query words intersect with the set returned by the root word. Not all documents in a collection are returned by any of the search queries in a set, so once the search query evolution is completed a second stage is performed whereby a KNN algorithm is applied to assign all unassigned documents to their nearest cluster. We describe the method and present results using 8 text datasets comparing effectiveness with well-known existing algorithms. We note that as well as achieving the highest accuracy on these datasets the search query format provides the qualitative benefits of being interpretable and modifiable whilst providing a causal explanation of cluster construction. 

---
# Predictive Modeling: BIM Command Recommendation Based on Large-scale Usage Logs 

**Authors**: Changyu Du, Zihan Deng, Stavros Nousias, André Borrmann  

**Link**: [PDF](https://arxiv.org/pdf/2504.05319)  

**Abstract**: The adoption of Building Information Modeling (BIM) and model-based design within the Architecture, Engineering, and Construction (AEC) industry has been hindered by the perception that using BIM authoring tools demands more effort than conventional 2D drafting. To enhance design efficiency, this paper proposes a BIM command recommendation framework that predicts the optimal next actions in real-time based on users' historical interactions. We propose a comprehensive filtering and enhancement method for large-scale raw BIM log data and introduce a novel command recommendation model. Our model builds upon the state-of-the-art Transformer backbones originally developed for large language models (LLMs), incorporating a custom feature fusion module, dedicated loss function, and targeted learning strategy. In a case study, the proposed method is applied to over 32 billion rows of real-world log data collected globally from the BIM authoring software Vectorworks. Experimental results demonstrate that our method can learn universal and generalizable modeling patterns from anonymous user interaction sequences across different countries, disciplines, and projects. When generating recommendations for the next command, our approach achieves a Recall@10 of approximately 84%. 

---
# Efficient Multi-Task Learning via Generalist Recommender 

**Authors**: Luyang Wang, Cangcheng Tang, Chongyang Zhang, Jun Ruan, Kai Huang, Jason Dai  

**Link**: [PDF](https://arxiv.org/pdf/2504.05318)  

**Abstract**: Multi-task learning (MTL) is a common machine learning technique that allows the model to share information across different tasks and improve the accuracy of recommendations for all of them. Many existing MTL implementations suffer from scalability issues as the training and inference performance can degrade with the increasing number of tasks, which can limit production use case scenarios for MTL-based recommender systems. Inspired by the recent advances of large language models, we developed an end-to-end efficient and scalable Generalist Recommender (GRec). GRec takes comprehensive data signals by utilizing NLP heads, parallel Transformers, as well as a wide and deep structure to process multi-modal inputs. These inputs are then combined and fed through a newly proposed task-sentence level routing mechanism to scale the model capabilities on multiple tasks without compromising performance. Offline evaluations and online experiments show that GRec significantly outperforms our previous recommender solutions. GRec has been successfully deployed on one of the largest telecom websites and apps, effectively managing high volumes of online traffic every day. 

---
# On Synthesizing Data for Context Attribution in Question Answering 

**Authors**: Gorjan Radevski, Kiril Gashteovski, Shahbaz Syed, Christopher Malon, Sebastien Nicolas, Chia-Chien Hung, Timo Sztyler, Verena Heußer, Wiem Ben Rim, Masafumi Enomoto, Kunihiro Takeoka, Masafumi Oyamada, Goran Glavaš, Carolin Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2504.05317)  

**Abstract**: Question Answering (QA) accounts for a significant portion of LLM usage "in the wild". However, LLMs sometimes produce false or misleading responses, also known as "hallucinations". Therefore, grounding the generated answers in contextually provided information -- i.e., providing evidence for the generated text -- is paramount for LLMs' trustworthiness. Providing this information is the task of context attribution. In this paper, we systematically study LLM-based approaches for this task, namely we investigate (i) zero-shot inference, (ii) LLM ensembling, and (iii) fine-tuning of small LMs on synthetic data generated by larger LLMs. Our key contribution is SynQA: a novel generative strategy for synthesizing context attribution data. Given selected context sentences, an LLM generates QA pairs that are supported by these sentences. This leverages LLMs' natural strengths in text generation while ensuring clear attribution paths in the synthetic training data. We show that the attribution data synthesized via SynQA is highly effective for fine-tuning small LMs for context attribution in different QA tasks and domains. Finally, with a user study, we validate the usefulness of small LMs (fine-tuned on synthetic data from SynQA) in context attribution for QA. 

---
# Scale Up Composed Image Retrieval Learning via Modification Text Generation 

**Authors**: Yinan Zhou, Yaxiong Wang, Haokun Lin, Chen Ma, Li Zhu, Zhedong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05316)  

**Abstract**: Composed Image Retrieval (CIR) aims to search an image of interest using a combination of a reference image and modification text as the query. Despite recent advancements, this task remains challenging due to limited training data and laborious triplet annotation processes. To address this issue, this paper proposes to synthesize the training triplets to augment the training resource for the CIR problem. Specifically, we commence by training a modification text generator exploiting large-scale multimodal models and scale up the CIR learning throughout both the pretraining and fine-tuning stages. During pretraining, we leverage the trained generator to directly create Modification Text-oriented Synthetic Triplets(MTST) conditioned on pairs of images. For fine-tuning, we first synthesize reverse modification text to connect the target image back to the reference image. Subsequently, we devise a two-hop alignment strategy to incrementally close the semantic gap between the multimodal pair and the target image. We initially learn an implicit prototype utilizing both the original triplet and its reversed version in a cycle manner, followed by combining the implicit prototype feature with the modification text to facilitate accurate alignment with the target image. Extensive experiments validate the efficacy of the generated triplets and confirm that our proposed methodology attains competitive recall on both the CIRR and FashionIQ benchmarks. 

---
# Coherency Improved Explainable Recommendation via Large Language Model 

**Authors**: Shijie Liu, Ruixing Ding, Weihai Lu, Jun Wang, Mo Yu, Xiaoming Shi, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05315)  

**Abstract**: Explainable recommender systems are designed to elucidate the explanation behind each recommendation, enabling users to comprehend the underlying logic. Previous works perform rating prediction and explanation generation in a multi-task manner. However, these works suffer from incoherence between predicted ratings and explanations. To address the issue, we propose a novel framework that employs a large language model (LLM) to generate a rating, transforms it into a rating vector, and finally generates an explanation based on the rating vector and user-item information. Moreover, we propose utilizing publicly available LLMs and pre-trained sentiment analysis models to automatically evaluate the coherence without human annotations. Extensive experimental results on three datasets of explainable recommendation show that the proposed framework is effective, outperforming state-of-the-art baselines with improvements of 7.3\% in explainability and 4.4\% in text quality. 

---
# Multimodal Quantitative Language for Generative Recommendation 

**Authors**: Jianyang Zhai, Zi-Feng Mai, Chang-Dong Wang, Feidiao Yang, Xiawu Zheng, Hui Li, Yonghong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2504.05314)  

**Abstract**: Generative recommendation has emerged as a promising paradigm aiming at directly generating the identifiers of the target candidates. Most existing methods attempt to leverage prior knowledge embedded in Pre-trained Language Models (PLMs) to improve the recommendation performance. However, they often fail to accommodate the differences between the general linguistic knowledge of PLMs and the specific needs of recommendation systems. Moreover, they rarely consider the complementary knowledge between the multimodal information of items, which represents the multi-faceted preferences of users. To facilitate efficient recommendation knowledge transfer, we propose a novel approach called Multimodal Quantitative Language for Generative Recommendation (MQL4GRec). Our key idea is to transform items from different domains and modalities into a unified language, which can serve as a bridge for transferring recommendation knowledge. Specifically, we first introduce quantitative translators to convert the text and image content of items from various domains into a new and concise language, known as quantitative language, with all items sharing the same vocabulary. Then, we design a series of quantitative language generation tasks to enrich quantitative language with semantic information and prior knowledge. Finally, we achieve the transfer of recommendation knowledge from different domains and modalities to the recommendation task through pre-training and fine-tuning. We evaluate the effectiveness of MQL4GRec through extensive experiments and comparisons with existing methods, achieving improvements over the baseline by 11.18\%, 14.82\%, and 7.95\% on the NDCG metric across three different datasets, respectively. 

---
# A Systematic Survey on Federated Sequential Recommendation 

**Authors**: Yichen Li, Qiyu Qin, Gaoyang Zhu, Wenchao Xu, Haozhao Wang, Yuhua Li, Rui Zhang, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.05313)  

**Abstract**: Sequential recommendation is an advanced recommendation technique that utilizes the sequence of user behaviors to generate personalized suggestions by modeling the temporal dependencies and patterns in user preferences. However, it requires a server to centrally collect users' data, which poses a threat to the data privacy of different users. In recent years, federated learning has emerged as a distributed architecture that allows participants to train a global model while keeping their private data locally. This survey pioneers Federated Sequential Recommendation (FedSR), where each user joins as a participant in federated training to achieve a recommendation service that balances data privacy and model performance. We begin with an introduction to the background and unique challenges of FedSR. Then, we review existing solutions from two levels, each of which includes two specific techniques. Additionally, we discuss the critical challenges and future research directions in FedSR. 

---
# Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation 

**Authors**: Qitao Qin, Yucong Luo, Yihang Lu, Zhibo Chu, Xianwei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05312)  

**Abstract**: Retrieval-Augmented Generation (RAG), by integrating non-parametric knowledge from external knowledge bases into models, has emerged as a promising approach to enhancing response accuracy while mitigating factual errors and hallucinations. This method has been widely applied in tasks such as Question Answering (QA). However, existing RAG methods struggle with open-domain QA tasks because they perform independent retrieval operations and directly incorporate the retrieved information into generation without maintaining a summarizing memory or using adaptive retrieval strategies, leading to noise from redundant information and insufficient information integration. To address these challenges, we propose Adaptive memory-based optimization for enhanced RAG (Amber) for open-domain QA tasks, which comprises an Agent-based Memory Updater, an Adaptive Information Collector, and a Multi-granular Content Filter, working together within an iterative memory updating paradigm. Specifically, Amber integrates and optimizes the language model's memory through a multi-agent collaborative approach, ensuring comprehensive knowledge integration from previous retrieval steps. It dynamically adjusts retrieval queries and decides when to stop retrieval based on the accumulated knowledge, enhancing retrieval efficiency and effectiveness. Additionally, it reduces noise by filtering irrelevant content at multiple levels, retaining essential information to improve overall model performance. We conduct extensive experiments on several open-domain QA datasets, and the results demonstrate the superiority and effectiveness of our method and its components. The source code is available \footnote{this https URL}. 

---
# GRIT: Graph-based Recall Improvement for Task-oriented E-commerce Queries 

**Authors**: Hrishikesh Kulkarni, Surya Kallumadi, Sean MacAvaney, Nazli Goharian, Ophir Frieder  

**Link**: [PDF](https://arxiv.org/pdf/2504.05310)  

**Abstract**: Many e-commerce search pipelines have four stages, namely: retrieval, filtering, ranking, and personalized-reranking. The retrieval stage must be efficient and yield high recall because relevant products missed in the first stage cannot be considered in later stages. This is challenging for task-oriented queries (queries with actionable intent) where user requirements are contextually intensive and difficult to understand. To foster research in the domain of e-commerce, we created a novel benchmark for Task-oriented Queries (TQE) by using LLM, which operates over the existing ESCI product search dataset. Furthermore, we propose a novel method 'Graph-based Recall Improvement for Task-oriented queries' (GRIT) to address the most crucial first-stage recall improvement needs. GRIT leads to robust and statistically significant improvements over state-of-the-art lexical, dense, and learned-sparse baselines. Our system supports both traditional and task-oriented e-commerce queries, yielding up to 6.3% recall improvement. In the indexing stage, GRIT first builds a product-product similarity graph using user clicks or manual annotation data. During retrieval, it locates neighbors with higher contextual and action relevance and prioritizes them over the less relevant candidates from the initial retrieval. This leads to a more comprehensive and relevant first-stage result set that improves overall system recall. Overall, GRIT leverages the locality relationships and contextual insights provided by the graph using neighboring nodes to enrich the first-stage retrieval results. We show that the method is not only robust across all introduced parameters, but also works effectively on top of a variety of first-stage retrieval methods. 

---
# IterQR: An Iterative Framework for LLM-based Query Rewrite in e-Commercial Search System 

**Authors**: Shangyu Chen, Xinyu Jia, Yingfei Zhang, Shuai Zhang, Xiang Li, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.05309)  

**Abstract**: The essence of modern e-Commercial search system lies in matching user's intent and available candidates depending on user's query, providing personalized and precise service. However, user's query may be incorrect due to ambiguous input and typo, leading to inaccurate search. These cases may be released by query rewrite: modify query to other representation or expansion. However, traditional query rewrite replies on static rewrite vocabulary, which is manually established meanwhile lacks interaction with both domain knowledge in e-Commercial system and common knowledge in the real world. In this paper, with the ability to generate text content of Large Language Models (LLMs), we provide an iterative framework to generate query rewrite. The framework incorporates a 3-stage procedure in each iteration: Rewrite Generation with domain knowledge by Retrieval-Augmented Generation (RAG) and query understanding by Chain-of-Thoughts (CoT); Online Signal Collection with automatic positive rewrite update; Post-training of LLM with multi task objective to generate new rewrites. Our work (named as IterQR) provides a comprehensive framework to generate \textbf{Q}uery \textbf{R}ewrite with both domain / real-world knowledge. It automatically update and self-correct the rewrites during \textbf{iter}ations. \method{} has been deployed in Meituan Delivery's search system (China's leading food delivery platform), providing service for users with significant improvement. 

---
# RARe: Raising Ad Revenue Framework with Context-Aware Reranking 

**Authors**: Ekaterina Solodneva, Alexandra Khirianova, Aleksandr Katrutsa, Roman Loginov, Andrey Tikhanov, Egor Samosvat, Yuriy Dorn  

**Link**: [PDF](https://arxiv.org/pdf/2504.05308)  

**Abstract**: Modern recommender systems excel at optimizing search result relevance for e-commerce platforms. While maintaining this relevance, platforms seek opportunities to maximize revenue through search result adjustments. To address the trade-off between relevance and revenue, we propose the $\mathsf{RARe}$ ($\textbf{R}$aising $\textbf{A}$dvertisement $\textbf{Re}$venue) framework. $\mathsf{RARe}$ stacks a click model and a reranking model. We train the $\mathsf{RARe}$ framework with a loss function to find revenue and relevance trade-offs. According to our experience, the click model is crucial in the $\mathsf{RARe}$ framework. We propose and compare two different click models that take into account the context of items in a search result. The first click model is a Gradient-Boosting Decision Tree with Concatenation (GBDT-C), which includes a context in the traditional GBDT model for click prediction. The second model, SAINT-Q, adapts the Sequential Attention model to capture influences between search results. Our experiments indicate that the proposed click models outperform baselines and improve the overall quality of our framework. Experiments on the industrial dataset, which will be released publicly, show $\mathsf{RARe}$'s significant revenue improvements while preserving a high relevance. 

---
# Toward Total Recall: Enhancing FAIRness through AI-Driven Metadata Standardization 

**Authors**: Sowmya S Sundaram, Mark A Musen  

**Link**: [PDF](https://arxiv.org/pdf/2504.05307)  

**Abstract**: Current metadata often suffer from incompleteness, inconsistency, and incorrect formatting, hindering effective data reuse and discovery. Using GPT-4 and a metadata knowledge base (CEDAR), we devised a method that standardizes metadata in scientific data sets, ensuring the adherence to community standards. The standardization process involves correcting and refining metadata entries to conform to established guidelines, significantly improving search performance and recall metrics. The investigation uses BioSample and GEO repositories to demonstrate the impact of these enhancements, showcasing how standardized metadata lead to better retrieval outcomes. The average recall improves significantly, rising from 17.65\% with the baseline raw datasets of BioSample and GEO to 62.87\% with our proposed metadata standardization pipeline. This finding highlights the transformative impact of integrating advanced AI models with structured metadata curation tools in achieving more effective and reliable data retrieval. 

---
# Are Generative AI Agents Effective Personalized Financial Advisors? 

**Authors**: Takehiro Takayanagi, Kiyoshi Izumi, Javier Sanz-Cruzado, Richard McCreadie, Iadh Ounis  

**Link**: [PDF](https://arxiv.org/pdf/2504.05862)  

**Abstract**: Large language model-based agents are becoming increasingly popular as a low-cost mechanism to provide personalized, conversational advice, and have demonstrated impressive capabilities in relatively simple scenarios, such as movie recommendations. But how do these agents perform in complex high-stakes domains, where domain expertise is essential and mistakes carry substantial risk? This paper investigates the effectiveness of LLM-advisors in the finance domain, focusing on three distinct challenges: (1) eliciting user preferences when users themselves may be unsure of their needs, (2) providing personalized guidance for diverse investment preferences, and (3) leveraging advisor personality to build relationships and foster trust. Via a lab-based user study with 64 participants, we show that LLM-advisors often match human advisor performance when eliciting preferences, although they can struggle to resolve conflicting user needs. When providing personalized advice, the LLM was able to positively influence user behavior, but demonstrated clear failure modes. Our results show that accurate preference elicitation is key, otherwise, the LLM-advisor has little impact, or can even direct the investor toward unsuitable assets. More worryingly, users appear insensitive to the quality of advice being given, or worse these can have an inverse relationship. Indeed, users reported a preference for and increased satisfaction as well as emotional trust with LLMs adopting an extroverted persona, even though those agents provided worse advice. 

---
# Automated Archival Descriptions with Federated Intelligence of LLMs 

**Authors**: Jinghua Groppe, Andreas Marquet, Annabel Walz, Sven Groppe  

**Link**: [PDF](https://arxiv.org/pdf/2504.05711)  

**Abstract**: Enforcing archival standards requires specialized expertise, and manually creating metadata descriptions for archival materials is a tedious and error-prone task. This work aims at exploring the potential of agentic AI and large language models (LLMs) in addressing the challenges of implementing a standardized archival description process. To this end, we introduce an agentic AI-driven system for automated generation of high-quality metadata descriptions of archival materials. We develop a federated optimization approach that unites the intelligence of multiple LLMs to construct optimal archival metadata. We also suggest methods to overcome the challenges associated with using LLMs for consistent metadata generation. To evaluate the feasibility and effectiveness of our techniques, we conducted extensive experiments using a real-world dataset of archival materials, which covers a variety of document types and data formats. The evaluation results demonstrate the feasibility of our techniques and highlight the superior performance of the federated optimization approach compared to single-model solutions in metadata quality and reliability. 

---
# Simplifying Data Integration: SLM-Driven Systems for Unified Semantic Queries Across Heterogeneous Databases 

**Authors**: Teng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.05634)  

**Abstract**: The integration of heterogeneous databases into a unified querying framework remains a critical challenge, particularly in resource-constrained environments. This paper presents a novel Small Language Model(SLM)-driven system that synergizes advancements in lightweight Retrieval-Augmented Generation (RAG) and semantic-aware data structuring to enable efficient, accurate, and scalable query resolution across diverse data formats. By integrating MiniRAG's semantic-aware heterogeneous graph indexing and topology-enhanced retrieval with SLM-powered structured data extraction, our system addresses the limitations of traditional methods in handling Multi-Entity Question Answering (Multi-Entity QA) and complex semantic queries. Experimental results demonstrate superior performance in accuracy and efficiency, while the introduction of semantic entropy as an unsupervised evaluation metric provides robust insights into model uncertainty. This work pioneers a cost-effective, domain-agnostic solution for next-generation database systems. 

---
# MicroNN: An On-device Disk-resident Updatable Vector Database 

**Authors**: Jeffrey Pound, Floris Chabert, Arjun Bhushan, Ankur Goswami, Anil Pacaci, Shihabur Rahman Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2504.05573)  

**Abstract**: Nearest neighbour search over dense vector collections has important applications in information retrieval, retrieval augmented generation (RAG), and content ranking. Performing efficient search over large vector collections is a well studied problem with many existing approaches and open source implementations. However, most state-of-the-art systems are generally targeted towards scenarios using large servers with an abundance of memory, static vector collections that are not updatable, and nearest neighbour search in isolation of other search criteria. We present Micro Nearest Neighbour (MicroNN), an embedded nearest-neighbour vector search engine designed for scalable similarity search in low-resource environments. MicroNN addresses the problem of on-device vector search for real-world workloads containing updates and hybrid search queries that combine nearest neighbour search with structured attribute filters. In this scenario, memory is highly constrained and disk-efficient index structures and algorithms are required, as well as support for continuous inserts and deletes. MicroNN is an embeddable library that can scale to large vector collections with minimal resources. MicroNN is used in production and powers a wide range of vector search use-cases on-device. MicroNN takes less than 7 ms to retrieve the top-100 nearest neighbours with 90% recall on publicly available million-scale vector benchmark while using ~10 MB of memory. 

---
# GraphRAFT: Retrieval Augmented Fine-Tuning for Knowledge Graphs on Graph Databases 

**Authors**: Alfred Clemedtson, Borun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.05478)  

**Abstract**: Large language models have shown remarkable language processing and reasoning ability but are prone to hallucinate when asked about private data. Retrieval-augmented generation (RAG) retrieves relevant data that fit into an LLM's context window and prompts the LLM for an answer. GraphRAG extends this approach to structured Knowledge Graphs (KGs) and questions regarding entities multiple hops away. The majority of recent GraphRAG methods either overlook the retrieval step or have ad hoc retrieval processes that are abstract or inefficient. This prevents them from being adopted when the KGs are stored in graph databases supporting graph query languages. In this work, we present GraphRAFT, a retrieve-and-reason framework that finetunes LLMs to generate provably correct Cypher queries to retrieve high-quality subgraph contexts and produce accurate answers. Our method is the first such solution that can be taken off-the-shelf and used on KGs stored in native graph DBs. Benchmarks suggest that our method is sample-efficient and scales with the availability of training data. Our method achieves significantly better results than all state-of-the-art models across all four standard metrics on two challenging Q\&As on large text-attributed KGs. 

---
# Unequal Opportunities: Examining the Bias in Geographical Recommendations by Large Language Models 

**Authors**: Shiran Dudy, Thulasi Tholeti, Resmi Ramachandranpillai, Muhammad Ali, Toby Jia-Jun Li, Ricardo Baeza-Yates  

**Link**: [PDF](https://arxiv.org/pdf/2504.05325)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have made them a popular information-seeking tool among end users. However, the statistical training methods for LLMs have raised concerns about their representation of under-represented topics, potentially leading to biases that could influence real-world decisions and opportunities. These biases could have significant economic, social, and cultural impacts as LLMs become more prevalent, whether through direct interactions--such as when users engage with chatbots or automated assistants--or through their integration into third-party applications (as agents), where the models influence decision-making processes and functionalities behind the scenes. Our study examines the biases present in LLMs recommendations of U.S. cities and towns across three domains: relocation, tourism, and starting a business. We explore two key research questions: (i) How similar LLMs responses are, and (ii) How this similarity might favor areas with certain characteristics over others, introducing biases. We focus on the consistency of LLMs responses and their tendency to over-represent or under-represent specific locations. Our findings point to consistent demographic biases in these recommendations, which could perpetuate a ``rich-get-richer'' effect that widens existing economic disparities. 

---
# Dr Web: a modern, query-based web data retrieval engine 

**Authors**: Ylli Prifti, Alessandro Provetti, Pasquale de Meo  

**Link**: [PDF](https://arxiv.org/pdf/2504.05311)  

**Abstract**: This article introduces the Data Retrieval Web Engine (also referred to as doctor web), a flexible and modular tool for extracting structured data from web pages using a simple query language. We discuss the engineering challenges addressed during its development, such as dynamic content handling and messy data extraction. Furthermore, we cover the steps for making the DR Web Engine public, highlighting its open source potential. 

---
# Cache-Aware Reinforcement Learning in Large-Scale Recommender Systems 

**Authors**: Xiaoshuang Chen, Gengrui Zhang, Yao Wang, Yulin Wu, Shuo Su, Kaiqiao Zhan, Ben Wang  

**Link**: [PDF](https://arxiv.org/pdf/2404.14961)  

**Abstract**: Modern large-scale recommender systems are built upon computation-intensive infrastructure and usually suffer from a huge difference in traffic between peak and off-peak periods. In peak periods, it is challenging to perform real-time computation for each request due to the limited budget of computational resources. The recommendation with a cache is a solution to this problem, where a user-wise result cache is used to provide recommendations when the recommender system cannot afford a real-time computation. However, the cached recommendations are usually suboptimal compared to real-time computation, and it is challenging to determine the items in the cache for each user. In this paper, we provide a cache-aware reinforcement learning (CARL) method to jointly optimize the recommendation by real-time computation and by the cache. We formulate the problem as a Markov decision process with user states and a cache state, where the cache state represents whether the recommender system performs recommendations by real-time computation or by the cache. The computational load of the recommender system determines the cache state. We perform reinforcement learning based on such a model to improve user engagement over multiple requests. Moreover, we show that the cache will introduce a challenge called critic dependency, which deteriorates the performance of reinforcement learning. To tackle this challenge, we propose an eigenfunction learning (EL) method to learn independent critics for CARL. Experiments show that CARL can significantly improve the users' engagement when considering the result cache. CARL has been fully launched in Kwai app, serving over 100 million users. 

---
