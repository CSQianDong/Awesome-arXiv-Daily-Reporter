# Revisiting Feedback Models for HyDE 

**Authors**: Nour Jedidi, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.19349)  

**Abstract**: Recent approaches that leverage large language models (LLMs) for pseudo-relevance feedback (PRF) have generally not utilized well-established feedback models like Rocchio and RM3 when expanding queries for sparse retrievers like BM25. Instead, they often opt for a simple string concatenation of the query and LLM-generated expansion content. But is this optimal? To answer this question, we revisit and systematically evaluate traditional feedback models in the context of HyDE, a popular method that enriches query representations with LLM-generated hypothetical answer documents. Our experiments show that HyDE's effectiveness can be substantially improved when leveraging feedback algorithms such as Rocchio to extract and weight expansion terms, providing a simple way to further enhance the accuracy of LLM-based PRF methods. 

---
# Generative Query Expansion with Multilingual LLMs for Cross-Lingual Information Retrieval 

**Authors**: Olivia Macmillan-Scott, Roksana Goworek, Eda B. Özyiğit  

**Link**: [PDF](https://arxiv.org/pdf/2511.19325)  

**Abstract**: Query expansion is the reformulation of a user query by adding semantically related information, and is an essential component of monolingual and cross-lingual information retrieval used to ensure that relevant documents are not missed. Recently, multilingual large language models (mLLMs) have shifted query expansion from semantic augmentation with synonyms and related words to pseudo-document generation. Pseudo-documents both introduce additional relevant terms and bridge the gap between short queries and long documents, which is particularly beneficial in dense retrieval. This study evaluates recent mLLMs and fine-tuned variants across several generative expansion strategies to identify factors that drive cross-lingual retrieval performance. Results show that query length largely determines which prompting technique is effective, and that more elaborate prompts often do not yield further gains. Substantial linguistic disparities persist: cross-lingual query expansion can produce the largest improvements for languages with the weakest baselines, yet retrieval is especially poor between languages written in different scripts. Fine-tuning is found to lead to performance gains only when the training and test data are of similar format. These outcomes underline the need for more balanced multilingual and cross-lingual training and evaluation resources. 

---
# What Drives Cross-lingual Ranking? Retrieval Approaches with Multilingual Language Models 

**Authors**: Roksana Goworek, Olivia Macmillan-Scott, Eda B. Özyiğit  

**Link**: [PDF](https://arxiv.org/pdf/2511.19324)  

**Abstract**: Cross-lingual information retrieval (CLIR) enables access to multilingual knowledge but remains challenging due to disparities in resources, scripts, and weak cross-lingual semantic alignment in embedding models. Existing pipelines often rely on translation and monolingual retrieval heuristics, which add computational overhead and noise, degrading performance. This work systematically evaluates four intervention types, namely document translation, multilingual dense retrieval with pretrained encoders, contrastive learning at word, phrase, and query-document levels, and cross-encoder re-ranking, across three benchmark datasets. We find that dense retrieval models trained specifically for CLIR consistently outperform lexical matching methods and derive little benefit from document translation. Contrastive learning mitigates language biases and yields substantial improvements for encoders with weak initial alignment, and re-ranking can be effective, but depends on the quality of the cross-encoder training data. Although high-resource languages still dominate overall performance, gains over lexical and document-translated baselines are most pronounced for low-resource and cross-script pairs. These findings indicate that cross-lingual search systems should prioritise semantic multilingual embeddings and targeted learning-based alignment over translation-based pipelines, particularly for cross-script and under-resourced languages. 

---
# BioArtlas: Computational Clustering of Multi-Dimensional Complexity in Bioart 

**Authors**: Joonhyung Bae  

**Link**: [PDF](https://arxiv.org/pdf/2511.19162)  

**Abstract**: Bioart's hybrid nature spanning art, science, technology, ethics, and politics defies traditional single-axis categorization. I present BioArtlas, analyzing 81 bioart works across thirteen curated dimensions using novel axis-aware representations that preserve semantic distinctions while enabling cross-dimensional comparison. Our codebook-based approach groups related concepts into unified clusters, addressing polysemy in cultural terminology. Comprehensive evaluation of up to 800 representation-space-algorithm combinations identifies Agglomerative clustering at k=15 on 4D UMAP as optimal (silhouette 0.664 +/- 0.008, trustworthiness/continuity 0.805/0.812). The approach reveals four organizational patterns: artist-specific methodological cohesion, technique-based segmentation, temporal artistic evolution, and trans-temporal conceptual affinities. By separating analytical optimization from public communication, I provide rigorous analysis and accessible exploration through an interactive web interface (this https URL) with the dataset publicly available (this https URL). 

---
# Heterogeneous Multi-treatment Uplift Modeling for Trade-off Optimization in Short-Video Recommendation 

**Authors**: Chenhao Zhai, Chang Meng, Xueliang Wang, Shuchang Liu, Xiaolong Hu, Shisong Tang, Xiaoqiang Feng, Xiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.18997)  

**Abstract**: The rapid proliferation of short videos on social media platforms presents unique challenges and opportunities for recommendation systems. Users exhibit diverse preferences, and the responses resulting from different strategies often conflict with one another, potentially exhibiting inverse correlations between metrics such as watch time and video view counts. Existing uplift models face limitations in handling the heterogeneous multi-treatment scenarios of short-video recommendations, often failing to effectively capture both the synergistic and individual causal effects of different strategies. Furthermore, traditional fixed-weight approaches for balancing these responses lack personalization and can result in biased decision-making. To address these issues, we propose a novel Heterogeneous Multi-treatment Uplift Modeling (HMUM) framework for trade-off optimization in short-video recommendations. HMUM comprises an Offline Hybrid Uplift Modeling (HUM) module, which captures the synergistic and individual effects of multiple strategies, and an Online Dynamic Decision-Making (DDM) module, which estimates the value weights of different user responses in real-time for personalized decision-making. Evaluated on two public datasets, an industrial dataset, and through online A/B experiments on the Kuaishou platform, our model demonstrated superior offline performance and significant improvements in key metrics. It is now fully deployed on the platform, benefiting hundreds of millions of users. 

---
# STORE: Semantic Tokenization, Orthogonal Rotation and Efficient Attention for Scaling Up Ranking Models 

**Authors**: Yi Xu, Chaofan Fan, Jinxin Hu, Yu Zhang, Zeng Xiaoyi, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18805)  

**Abstract**: Ranking models have become an important part of modern personalized recommendation systems. However, significant challenges persist in handling high-cardinality, heterogeneous, and sparse feature spaces, particularly regarding model scalability and efficiency. We identify two key bottlenecks: (i) Representation Bottleneck: Driven by the high cardinality and dynamic nature of features, model capacity is forced into sparse-activated embedding layers, leading to low-rank representations. This, in turn, triggers phenomena like "One-Epoch" and "Interaction-Collapse," ultimately hindering model scalability.(ii) Computational Bottleneck: Integrating all heterogeneous features into a unified model triggers an explosion in the number of feature tokens, rendering traditional attention mechanisms computationally demanding and susceptible to attention dispersion. To dismantle these barriers, we introduce STORE, a unified and scalable token-based ranking framework built upon three core innovations: (1) Semantic Tokenization fundamentally tackles feature heterogeneity and sparsity by decomposing high-cardinality sparse features into a compact set of stable semantic tokens; and (2) Orthogonal Rotation Transformation is employed to rotate the subspace spanned by low-cardinality static features, which facilitates more efficient and effective feature interactions; and (3) Efficient attention that filters low-contributing tokens to improve computional efficiency while preserving model accuracy. Across extensive offline experiments and online A/B tests, our framework consistently improves prediction accuracy(online CTR by 2.71%, AUC by 1.195%) and training effeciency (1.84 throughput). 

---
# Multimodal Large Language Models with Adaptive Preference Optimization for Sequential Recommendation 

**Authors**: Yu Wang, Yonghui Yang, Le Wu, Yi Zhang, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2511.18740)  

**Abstract**: Recent advances in Large Language Models (LLMs) have opened new avenues for sequential recommendation by enabling natural language reasoning over user behavior sequences. A common approach formulates recommendation as a language modeling task, where interaction histories are transformed into prompts and user preferences are learned via supervised fine-tuning. However, these methods operate solely in the textual modality and often miss users' fine-grained interests, especially when shaped by rich visual signals such as product images or movie posters. Multimodal Large Language Models (MLLMs) offer a promising alternative by aligning text and vision in a shared semantic space. A prevalent training paradigm applies Supervised Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO) to model user preferences. Yet, two core challenges remain: 1) Imbalanced sample hardness, where random negative sampling causes overfitting on easy examples and under-training on hard ones; 2) Cross-modal semantic bias, where the fixed reference model in DPO prevents the policy model from correcting modality misalignments--especially over long sequences. To address these issues, we propose a Multimodal LLM framework that integrates Hardness-aware and Noise-regularized preference optimization for Recommendation (HaNoRec). Specifically, HaNoRec dynamically adjusts optimization weights based on both the estimated hardness of each training sample and the policy model's real-time responsiveness, prioritizing harder examples. It further introduces Gaussian-perturbed distribution optimization on output logits to enhance cross-modal semantic consistency and reduce modality bias inherited from the reference model. 

---
# When and What to Recommend: Joint Modeling of Timing and Content for Active Sequential Recommendation 

**Authors**: Jin Chai, Xiaoxiao Ma, Jian Yang, Jia Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.18717)  

**Abstract**: Sequential recommendation models user preferences to predict the next target item. Most existing work is passive, where the system responds only when users open the application, missing chances after closure. We investigate active recommendation, which predicts the next interaction time and actively delivers items. Two challenges: accurately estimating the Time of Interest (ToI) and generating Item of Interest (IoI) conditioned on the predicted ToI. We propose PASRec, a diffusion-based framework that aligns ToI and IoI via a joint objective. Experiments on five benchmarks show superiority over eight state-of-the-art baselines under leave-one-out and temporal splits. 

---
# A Recommender System Based on Binary Matrix Representations for Cognitive Disorders 

**Authors**: Raoul H. Kutil, Georg Zimmermann, Christian Borgelt  

**Link**: [PDF](https://arxiv.org/pdf/2511.18645)  

**Abstract**: Diagnosing cognitive (mental health) disorders is a delicate and complex task. Identifying the next most informative symptoms to assess, in order to distinguish between possible disorders, presents an additional challenge. This process requires comprehensive knowledge of diagnostic criteria and symptom overlap across disorders, making it difficult to navigate based on symptoms alone. This research aims to develop a recommender system for cognitive disorder diagnosis using binary matrix representations. The core algorithm utilizes a binary matrix of disorders and their symptom combinations. It filters through the rows and columns based on the patient's current symptoms to identify potential disorders and recommend the most informative next symptoms to examine. A prototype of the recommender system was implemented in Python. Using synthetic test and some real-life data, the system successfully identified plausible disorders from an initial symptom set and recommended further symptoms to refine the diagnosis. It also provided additional context on the symptom-disorder relationships. Although this is a prototype, the recommender system shows potential as a clinical support tool. A fully-developed application of this recommender system may assist mental health professionals in identifying relevant disorders more efficiently and guiding symptom-specific follow-up investigations to improve diagnostic accuracy. 

---
# Time Matters: Enhancing Sequential Recommendations with Time-Guided Graph Neural ODEs 

**Authors**: Haoyan Fu, Zhida Qin, Shixiao Yang, Haoyao Zhang, Bin Lu, Shuang Li, Tianyu Huang, John C.S. Lui  

**Link**: [PDF](https://arxiv.org/pdf/2511.18347)  

**Abstract**: Sequential recommendation (SR) is widely deployed in e-commerce platforms, streaming services, etc., revealing significant potential to enhance user experience. However, existing methods often overlook two critical factors: irregular user interests between interactions and highly uneven item distributions over time. The former factor implies that actual user preferences are not always continuous, and long-term historical interactions may not be relevant to current purchasing behavior. Therefore, relying only on these historical interactions for recommendations may result in a lack of user interest at the target time. The latter factor, characterized by peaks and valleys in interaction frequency, may result from seasonal trends, special events, or promotions. These externally driven distributions may not align with individual user interests, leading to inaccurate recommendations. To address these deficiencies, we propose TGODE to both enhance and capture the long-term historical interactions. Specifically, we first construct a user time graph and item evolution graph, which utilize user personalized preferences and global item distribution information, respectively. To tackle the temporal sparsity caused by irregular user interactions, we design a time-guided diffusion generator to automatically obtain an augmented time-aware user graph. Additionally, we devise a user interest truncation factor to efficiently identify sparse time intervals and achieve balanced preference inference. After that, the augmented user graph and item graph are fed into a generalized graph neural ordinary differential equation (ODE) to align with the evolution of user preferences and item distributions. This allows two patterns of information evolution to be matched over time. Experimental results demonstrate that TGODE outperforms baseline methods across five datasets, with improvements ranging from 10% to 46%. 

---
# UFO: Unfair-to-Fair Evolving Mitigates Unfairness in LLM-based Recommender Systems via Self-Play Fine-tuning 

**Authors**: Jiaming Zhang, Yuyuan Li, Xiaohua Feng, Zhifei Ren, Li Zhang, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.18342)  

**Abstract**: Large language model-based Recommender Systems (LRSs) have demonstrated superior recommendation performance by integrating pre-training with Supervised Fine-Tuning (SFT). However, this approach introduces item-side unfairness. Existing studies primarily attribute this issue to the absence of fairness constraints during SFT and attempt to mitigate unfairness via re-weighting and re-ranking methods. In this paper, we find that unfairness arises not only from SFT but also from pre-training, where inherent biases are further amplified during SFT. This finding underscores the failure of current methods to address the root causes of unfairness. Moreover, current methods struggle to preserve satisfactory recommendation performance. To tackle these issues, we propose an Unfair-to-Fair evOlving (UFO) framework using a self-play mechanism, formulating unfairness mitigation as a two-player game. UFO alternates between two player roles: the \textit{judger}, which identifies unfairness from both pre-training and SFT, and the \textit{corrector}, which adjusts the LRS to address identified unfairness while preserving recommendation performance. Iterative optimization between these roles enables UFO to completely resolve unfairness. Extensive experiments demonstrate that UFO effectively mitigates unfairness while improving recommendation performance. 

---
# Large Language Model Enhanced Graph Invariant Contrastive Learning for Out-of-Distribution Recommendation 

**Authors**: Jiahao Liang, Haoran Yang, Xiangyu Zhao, Zhiwen Yu, Mianjie Li, Chuan Shi, Kaixiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18282)  

**Abstract**: Out-of-distribution (OOD) generalization has emerged as a significant challenge in graph recommender systems. Traditional graph neural network algorithms often fail because they learn spurious environmental correlations instead of stable causal relationships, leading to substantial performance degradation under distribution shifts. While recent advancements in Large Language Models (LLMs) offer a promising avenue due to their vast world knowledge and reasoning capabilities, effectively integrating this knowledge with the fine-grained topology of specific graphs to solve the OOD problem remains a significant challenge. To address these issues, we propose {$\textbf{Inv}$ariant $\textbf{G}$raph $\textbf{C}$ontrastive Learning with $\textbf{LLM}$s for Out-of-Distribution Recommendation (InvGCLLM)}, an innovative causal learning framework that synergistically integrates the strengths of data-driven models and knowledge-driven LLMs. Our framework first employs a data-driven invariant learning model to generate causal confidence scores for each user-item interaction. These scores then guide an LLM to perform targeted graph refinement, leveraging its world knowledge to prune spurious connections and augment missing causal links. Finally, the structurally purified graphs provide robust supervision for a causality-guided contrastive learning objective, enabling the model to learn representations that are resilient to spurious correlations. Experiments conducted on four public datasets demonstrate that InvGCLLM achieves significant improvements in out-of-distribution recommendation, consistently outperforming state-of-the-art baselines. 

---
# Democratic Recommendation with User and Item Representatives Produced by Graph Condensation 

**Authors**: Jiahao Liang, Haoran Yang, Xiangyu Zhao, Zhiwen Yu, Guandong Xu, Wanyu Wang, Kaixiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18279)  

**Abstract**: The challenges associated with large-scale user-item interaction graphs have attracted increasing attention in graph-based recommendation systems, primarily due to computational inefficiencies and inadequate information propagation. Existing methods provide partial solutions but suffer from notable limitations: model-centric approaches, such as sampling and aggregation, often struggle with generalization, while data-centric techniques, including graph sparsification and coarsening, lead to information loss and ineffective handling of bipartite graph structures. Recent advances in graph condensation offer a promising direction by reducing graph size while preserving essential information, presenting a novel approach to mitigating these challenges. Inspired by the principles of democracy, we propose \textbf{DemoRec}, a framework that leverages graph condensation to generate user and item representatives for recommendation tasks. By constructing a compact interaction graph and clustering nodes with shared characteristics from the original graph, DemoRec significantly reduces graph size and computational complexity. Furthermore, it mitigates the over-reliance on high-order information, a critical challenge in large-scale bipartite graphs. Extensive experiments conducted on four public datasets demonstrate the effectiveness of DemoRec, showcasing substantial improvements in recommendation performance, computational efficiency, and robustness compared to SOTA methods. 

---
# LLM Reasoning for Cold-Start Item Recommendation 

**Authors**: Shijun Li, Yu Wang, Jin Wang, Ying Li, Joydeep Ghosh, Anne Cocos  

**Link**: [PDF](https://arxiv.org/pdf/2511.18261)  

**Abstract**: Large Language Models (LLMs) have shown significant potential for improving recommendation systems through their inherent reasoning capabilities and extensive knowledge base. Yet, existing studies predominantly address warm-start scenarios with abundant user-item interaction data, leaving the more challenging cold-start scenarios, where sparse interactions hinder traditional collaborative filtering methods, underexplored. To address this limitation, we propose novel reasoning strategies designed for cold-start item recommendations within the Netflix domain. Our method utilizes the advanced reasoning capabilities of LLMs to effectively infer user preferences, particularly for newly introduced or rarely interacted items. We systematically evaluate supervised fine-tuning, reinforcement learning-based fine-tuning, and hybrid approaches that combine both methods to optimize recommendation performance. Extensive experiments on real-world data demonstrate significant improvements in both methodological efficacy and practical performance in cold-start recommendation contexts. Remarkably, our reasoning-based fine-tuned models outperform Netflix's production ranking model by up to 8% in certain cases. 

---
# ProHD: Projection-Based Hausdorff Distance Approximation 

**Authors**: Jiuzhou Fu, Luanzheng Guo, Nathan R. Tallent, Dongfang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.18207)  

**Abstract**: The Hausdorff distance (HD) is a robust measure of set dissimilarity, but computing it exactly on large, high-dimensional datasets is prohibitively expensive. We propose \textbf{ProHD}, a projection-guided approximation algorithm that dramatically accelerates HD computation while maintaining high accuracy. ProHD identifies a small subset of candidate "extreme" points by projecting the data onto a few informative directions (such as the centroid axis and top principal components) and computing the HD on this subset. This approach guarantees an underestimate of the true HD with a bounded additive error and typically achieves results within a few percent of the exact value. In extensive experiments on image, physics, and synthetic datasets (up to two million points in $D=256$), ProHD runs 10--100$\times$ faster than exact algorithms while attaining 5--20$\times$ lower error than random sampling-based approximations. Our method enables practical HD calculations in scenarios like large vector databases and streaming data, where quick and reliable set distance estimation is needed. 

---
# Fidelity-Aware Recommendation Explanations via Stochastic Path Integration 

**Authors**: Oren Barkan, Yahlly Schein, Yehonatan Elisha, Veronika Bogina, Mikhail Baklanov, Noam Koenigstein  

**Link**: [PDF](https://arxiv.org/pdf/2511.18047)  

**Abstract**: Explanation fidelity, which measures how accurately an explanation reflects a model's true reasoning, remains critically underexplored in recommender systems. We introduce SPINRec (Stochastic Path Integration for Neural Recommender Explanations), a model-agnostic approach that adapts path-integration techniques to the sparse and implicit nature of recommendation data. To overcome the limitations of prior methods, SPINRec employs stochastic baseline sampling: instead of integrating from a fixed or unrealistic baseline, it samples multiple plausible user profiles from the empirical data distribution and selects the most faithful attribution path. This design captures the influence of both observed and unobserved interactions, yielding more stable and personalized explanations. We conduct the most comprehensive fidelity evaluation to date across three models (MF, VAE, NCF), three datasets (ML1M, Yahoo! Music, Pinterest), and a suite of counterfactual metrics, including AUC-based perturbation curves and fixed-length diagnostics. SPINRec consistently outperforms all baselines, establishing a new benchmark for faithful explainability in recommendation. Code and evaluation tools are publicly available at this https URL. 

---
# Extracting Interaction-Aware Monosemantic Concepts in Recommender Systems 

**Authors**: Dor Arviv, Yehonatan Elisha, Oren Barkan, Noam Koenigstein  

**Link**: [PDF](https://arxiv.org/pdf/2511.18024)  

**Abstract**: We present a method for extracting \emph{monosemantic} neurons, defined as latent dimensions that align with coherent and interpretable concepts, from user and item embeddings in recommender systems. Our approach employs a Sparse Autoencoder (SAE) to reveal semantic structure within pretrained representations. In contrast to work on language models, monosemanticity in recommendation must preserve the interactions between separate user and item embeddings. To achieve this, we introduce a \emph{prediction aware} training objective that backpropagates through a frozen recommender and aligns the learned latent structure with the model's user-item affinity predictions. The resulting neurons capture properties such as genre, popularity, and temporal trends, and support post hoc control operations including targeted filtering and content promotion without modifying the base model. Our method generalizes across different recommendation models and datasets, providing a practical tool for interpretable and controllable personalization. Code and evaluation resources are available at this https URL. 

---
# Save, Revisit, Retain: A Scalable Framework for Enhancing User Retention in Large-Scale Recommender Systems 

**Authors**: Weijie Jiang, Armando Ordorica, Jaewon Yang, Olafur Gudmundsson, Yucheng Tu, Huizhong Duan  

**Link**: [PDF](https://arxiv.org/pdf/2511.18013)  

**Abstract**: User retention is a critical objective for online platforms like Pinterest, as it strengthens user loyalty and drives growth through repeated engagement. A key indicator of retention is revisitation, i.e., when users return to view previously saved content, a behavior often sparked by personalized recommendations and user satisfaction. However, modeling and optimizing revisitation poses significant challenges. One core difficulty is accurate attribution: it is often unclear which specific user actions or content exposures trigger a revisit, since many confounding factors (e.g., content quality, user interface, notifications, or even changing user intent) can influence return behavior. Additionally, the scale and timing of revisitations introduce further complexity; users may revisit content days or even weeks after their initial interaction, requiring the system to maintain and associate extensive historical records across millions of users and sessions. These complexities render existing methods insufficient for robustly capturing and optimizing long-term revisitation. To address these gaps, we introduce a novel, lightweight, and interpretable framework for modeling revisitation behavior and optimizing long-term user retention in Pinterest's search-based recommendation context. By defining a surrogate attribution process that links saves to subsequent revisitations, we reduce noise in the causal relationship between user actions and return visits. Our scalable event aggregation pipeline enables large-scale analysis of user revisitation patterns and enhances the ranking system's ability to surface items with high retention value. Deployed on Pinterest's Related Pins surface to serve 500+ million users, the framework led to a significant lift of 0.1% in active users without additional computational costs. 

---
# Token-Controlled Re-ranking for Sequential Recommendation via LLMs 

**Authors**: Wenxi Dai, Wujiang Xu, Pinhuan Wang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2511.17913)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) as re-rankers is shifting recommender systems towards a user-centric paradigm. However, a significant gap remains: current re-rankers often lack mechanisms for fine-grained user control. They struggle to balance inherent user preferences with multiple attribute-based constraints, often resorting to simplistic hard filtering that can excessively narrow the recommendation pool and yield suboptimal results. This limitation leaves users as passive recipients rather than active collaborators in the recommendation process. To bridge this gap, we propose COREC, a novel token-augmented re-ranking framework that incorporates specific user requirements in co-creating the recommendation outcome. COREC empowers users to steer re-ranking results with precise and flexible control via explicit, attribute-based signals. The framework learns to balance these commands against latent preferences, yielding rankings that adhere to user instructions without sacrificing personalization. Experiments show that COREC: (1) exceeds state-of-the-art baselines on standard recommendation effectiveness and (2) demonstrates superior adherence to specific attribute requirements, proving that COREC enables fine-grained and predictable manipulation of the rankings. 

---
# From Raw Features to Effective Embeddings: A Three-Stage Approach for Multimodal Recipe Recommendation 

**Authors**: Jeeho Shin, Kyungho Kim, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2511.19176)  

**Abstract**: Recipe recommendation has become an essential task in web-based food platforms. A central challenge is effectively leveraging rich multimodal features beyond user-recipe interactions. Our analysis shows that even simple uses of multimodal signals yield competitive performance, suggesting that systematic enhancement of these signals is highly promising. We propose TESMR, a 3-stage framework for recipe recommendation that progressively refines raw multimodal features into effective embeddings through: (1) content-based enhancement using foundation models with multimodal comprehension, (2) relation-based enhancement via message propagation over user-recipe interactions, and (3) learning-based enhancement through contrastive learning with learnable embeddings. Experiments on two real-world datasets show that TESMR outperforms existing methods, achieving 7-15% higher Recall@10. 

---
# Large Language Models Require Curated Context for Reliable Political Fact-Checking -- Even with Reasoning and Web Search 

**Authors**: Matthew R. DeVerna, Kai-Cheng Yang, Harry Yaojun Yan, Filippo Menczer  

**Link**: [PDF](https://arxiv.org/pdf/2511.18749)  

**Abstract**: Large language models (LLMs) have raised hopes for automated end-to-end fact-checking, but prior studies report mixed results. As mainstream chatbots increasingly ship with reasoning capabilities and web search tools -- and millions of users already rely on them for verification -- rigorous evaluation is urgent. We evaluate 15 recent LLMs from OpenAI, Google, Meta, and DeepSeek on more than 6,000 claims fact-checked by PolitiFact, comparing standard models with reasoning- and web-search variants. Standard models perform poorly, reasoning offers minimal benefits, and web search provides only moderate gains, despite fact-checks being available on the web. In contrast, a curated RAG system using PolitiFact summaries improved macro F1 by 233% on average across model variants. These findings suggest that giving models access to curated high-quality context is a promising path for automated fact-checking. 

---
# General Agentic Memory Via Deep Research 

**Authors**: B.Y. Yan, Chaofan Li, Hongjin Qian, Shuqi Lu, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.18423)  

**Abstract**: Memory is critical for AI agents, yet the widely-adopted static memory, aiming to create readily available memory in advance, is inevitably subject to severe information loss. To address this limitation, we propose a novel framework called \textbf{general agentic memory (GAM)}. GAM follows the principle of "\textbf{just-in time (JIT) compilation}" where it focuses on creating optimized contexts for its client at runtime while keeping only simple but useful memory during the offline stage. To this end, GAM employs a duo-design with the following components. 1) \textbf{Memorizer}, which highlights key historical information using a lightweight memory, while maintaining complete historical information within a universal page-store. 2) \textbf{Researcher}, which retrieves and integrates useful information from the page-store for its online request guided by the pre-constructed memory. This design allows GAM to effectively leverage the agentic capabilities and test-time scalability of frontier large language models (LLMs), while also facilitating end-to-end performance optimization through reinforcement learning. In our experimental study, we demonstrate that GAM achieves substantial improvement on various memory-grounded task completion scenarios against existing memory systems. 

---
# Multi-Agent Collaborative Filtering: Orchestrating Users and Items for Agentic Recommendations 

**Authors**: Yu Xia, Sungchul Kim, Tong Yu, Ryan A. Rossi, Julian McAuely  

**Link**: [PDF](https://arxiv.org/pdf/2511.18413)  

**Abstract**: Agentic recommendations cast recommenders as large language model (LLM) agents that can plan, reason, use tools, and interact with users of varying preferences in web applications. However, most existing agentic recommender systems focus on generic single-agent plan-execute workflows or multi-agent task decomposition pipelines. Without recommendation-oriented design, they often underuse the collaborative signals in the user-item interaction history, leading to unsatisfying recommendation results. To address this, we propose the Multi-Agent Collaborative Filtering (MACF) framework for agentic recommendations, drawing an analogy between traditional collaborative filtering algorithms and LLM-based multi-agent collaboration. Specifically, given a target user and query, we instantiate similar users and relevant items as LLM agents with unique profiles. Each agent is able to call retrieval tools, suggest candidate items, and interact with other agents. Different from the static preference aggregation in traditional collaborative filtering, MACF employs a central orchestrator agent to adaptively manage the collaboration between user and item agents via dynamic agent recruitment and personalized collaboration instruction. Experimental results on datasets from three different domains show the advantages of our MACF framework compared to strong agentic recommendation baselines. 

---
# A Multimodal Conversational Agent for Tabular Data Analysis 

**Authors**: Mohammad Nour Al Awad, Sergey Ivanov, Olga Tikhonova, Ivan Khodnenko  

**Link**: [PDF](https://arxiv.org/pdf/2511.18405)  

**Abstract**: Large language models (LLMs) can reshape information processing by handling data analysis, visualization, and interpretation in an interactive, context-aware dialogue with users, including voice interaction, while maintaining high performance. In this article, we present Talk2Data, a multimodal LLM-driven conversational agent for intuitive data exploration. The system lets users query datasets with voice or text instructions and receive answers as plots, tables, statistics, or spoken explanations. Built on LLMs, the suggested design combines OpenAI Whisper automatic speech recognition (ASR) system, Qwen-coder code generation LLM/model, custom sandboxed execution tools, and Coqui library for text-to-speech (TTS) within an agentic orchestration loop. Unlike text-only analysis tools, it adapts responses across modalities and supports multi-turn dialogues grounded in dataset context. In an evaluation of 48 tasks on three datasets, our prototype achieved 95.8% accuracy with model-only generation time under 1.7 seconds (excluding ASR and execution time). A comparison across five LLM sizes (1.5B-32B) revealed accuracy-latency-cost trade-offs, with a 7B model providing the best balance for interactive use. By routing between conversation with user and code execution, constrained to a transparent sandbox, with simultaneously grounding prompts in schema-level context, the Talk2Data agent reliably retrieves actionable insights from tables while making computations verifiable. In the article, except for the Talk2Data agent itself, we discuss implications for human-data interaction, trust in LLM-driven analytics, and future extensions toward large-scale multimodal assistants. 

---
# Toward an AI-Native Internet: Rethinking the Web Architecture for Semantic Retrieval 

**Authors**: Muhammad Bilal, Zafar Qazi, Marco Canini  

**Link**: [PDF](https://arxiv.org/pdf/2511.18354)  

**Abstract**: The rise of Generative AI Search is fundamentally transforming how users and intelligent systems interact with the Internet. LLMs increasingly act as intermediaries between humans and web information. Yet the web remains optimized for human browsing rather than AI-driven semantic retrieval, resulting in wasted network bandwidth, lower information quality, and unnecessary complexity for developers. We introduce the concept of an AI-Native Internet, a web architecture in which servers expose semantically relevant information chunks rather than full documents, supported by a Web-native semantic resolver that allows AI applications to discover relevant information sources before retrieving fine-grained chunks. Through motivational experiments, we quantify the inefficiencies of current HTML-based retrieval, and outline architectural directions and open challenges for evolving today's document-centric web into an AI-oriented substrate that better supports semantic access to web content. 

---
# Path-Constrained Retrieval: A Structural Approach to Reliable LLM Agent Reasoning Through Graph-Scoped Semantic Search 

**Authors**: Joseph Oladokun  

**Link**: [PDF](https://arxiv.org/pdf/2511.18313)  

**Abstract**: Large Language Model agents often retrieve context from knowledge bases that lack structural consistency with the agent's current reasoning state, leading to incoherent reasoning chains. We introduce Path-Constrained Retrieval (PCR), a retrieval method that combines structural graph constraints with semantic search to ensure retrieved information maintains logical relationships within a knowledge graph. PCR restricts the search space to nodes reachable from an anchor node, preventing retrieval of structurally disconnected information that may lead to inconsistent reasoning. We evaluate PCR on PathRAG-6, a benchmark spanning six domains with 180 nodes and 360 edges. Our results show that PCR achieves full structural consistency compared to 24-32 percent in baseline methods, while maintaining strong relevance scores. On the technology domain, PCR obtains full relevance at rank 10 with full structural consistency, significantly outperforming vector search and hybrid retrieval. PCR reduces the average graph distance of retrieved context by 78 percent compared to baselines, demonstrating retrieval of more structurally consistent information. These findings suggest that path-constrained retrieval is an effective approach for improving the reliability and coherence of LLM agent reasoning systems. 

---
# Paper2SysArch: Structure-Constrained System Architecture Generation from Scientific Papers 

**Authors**: Ziyi Guo, Zhou Liu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18036)  

**Abstract**: The manual creation of system architecture diagrams for scientific papers is a time-consuming and subjective process, while existing generative models lack the necessary structural control and semantic understanding for this task. A primary obstacle hindering research and development in this domain has been the profound lack of a standardized benchmark to quantitatively evaluate the automated generation of diagrams from text. To address this critical gap, we introduce a novel and comprehensive benchmark, the first of its kind, designed to catalyze progress in automated scientific visualization. It consists of 3,000 research papers paired with their corresponding high-quality ground-truth diagrams and is accompanied by a three-tiered evaluation metric assessing semantic accuracy, layout coherence, and visual quality. Furthermore, to establish a strong baseline on this new benchmark, we propose Paper2SysArch, an end-to-end system that leverages multi-agent collaboration to convert papers into structured, editable diagrams. To validate its performance on complex cases, the system was evaluated on a manually curated and more challenging subset of these papers, where it achieves a composite score of 69.0. This work's principal contribution is the establishment of a large-scale, foundational benchmark to enable reproducible research and fair comparison. Meanwhile, our proposed system serves as a viable proof-of-concept, demonstrating a promising path forward for this complex task. 

---
# HyM-UNet: Synergizing Local Texture and Global Context via Hybrid CNN-Mamba Architecture for Medical Image Segmentation 

**Authors**: Haodong Chen, Xianfei Han, Qwen  

**Link**: [PDF](https://arxiv.org/pdf/2511.17988)  

**Abstract**: Accurate organ and lesion segmentation is a critical prerequisite for computer-aided diagnosis. Convolutional Neural Networks (CNNs), constrained by their local receptive fields, often struggle to capture complex global anatomical structures. To tackle this challenge, this paper proposes a novel hybrid architecture, HyM-UNet, designed to synergize the local feature extraction capabilities of CNNs with the efficient global modeling capabilities of Mamba. Specifically, we design a Hierarchical Encoder that utilizes convolutional modules in the shallow stages to preserve high-frequency texture details, while introducing Visual Mamba modules in the deep stages to capture long-range semantic dependencies with linear complexity. To bridge the semantic gap between the encoder and the decoder, we propose a Mamba-Guided Fusion Skip Connection (MGF-Skip). This module leverages deep semantic features as gating signals to dynamically suppress background noise within shallow features, thereby enhancing the perception of ambiguous boundaries. We conduct extensive experiments on public benchmark dataset ISIC 2018. The results demonstrate that HyM-UNet significantly outperforms existing state-of-the-art methods in terms of Dice coefficient and IoU, while maintaining lower parameter counts and inference latency. This validates the effectiveness and robustness of the proposed method in handling medical segmentation tasks characterized by complex shapes and scale variations. 

---
# Leveraging Evidence-Guided LLMs to Enhance Trustworthy Depression Diagnosis 

**Authors**: Yining Yuan, J. Ben Tamo, Micky C. Nnamdi, Yifei Wang, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17947)  

**Abstract**: Large language models (LLMs) show promise in automating clinical diagnosis, yet their non-transparent decision-making and limited alignment with diagnostic standards hinder trust and clinical adoption. We address this challenge by proposing a two-stage diagnostic framework that enhances transparency, trustworthiness, and reliability. First, we introduce Evidence-Guided Diagnostic Reasoning (EGDR), which guides LLMs to generate structured diagnostic hypotheses by interleaving evidence extraction with logical reasoning grounded in DSM-5 criteria. Second, we propose a Diagnosis Confidence Scoring (DCS) module that evaluates the factual accuracy and logical consistency of generated diagnoses through two interpretable metrics: the Knowledge Attribution Score (KAS) and the Logic Consistency Score (LCS). Evaluated on the D4 dataset with pseudo-labels, EGDR outperforms direct in-context prompting and Chain-of-Thought (CoT) across five LLMs. For instance, on OpenBioLLM, EGDR improves accuracy from 0.31 (Direct) to 0.76 and increases DCS from 0.50 to 0.67. On MedLlama, DCS rises from 0.58 (CoT) to 0.77. Overall, EGDR yields up to +45% accuracy and +36% DCS gains over baseline methods, offering a clinically grounded, interpretable foundation for trustworthy AI-assisted diagnosis. 

---
# Principled Context Engineering for RAG: Statistical Guarantees via Conformal Prediction 

**Authors**: Debashish Chakraborty, Eugene Yang, Daniel Khashabi, Dawn Lawrie, Kevin Duh  

**Link**: [PDF](https://arxiv.org/pdf/2511.17908)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances factual grounding in large language models (LLMs) by incorporating retrieved evidence, but LLM accuracy declines when long or noisy contexts exceed the model's effective attention span. Existing pre-generation filters rely on heuristics or uncalibrated LLM confidence scores, offering no statistical control over retained evidence. We evaluate and demonstrate context engineering through conformal prediction, a coverage-controlled filtering framework that removes irrelevant content while preserving recall of supporting evidence. Using both embedding- and LLM-based scoring functions, we test this approach on the NeuCLIR and RAGTIME collections. Conformal filtering consistently meets its target coverage, ensuring that a specified fraction of relevant snippets are retained, and reduces retained context by 2-3x relative to unfiltered retrieval. On NeuCLIR, downstream factual accuracy measured by ARGUE F1 improves under strict filtering and remains stable at moderate coverage, indicating that most discarded material is redundant or irrelevant. These results demonstrate that conformal prediction enables reliable, coverage-controlled context reduction in RAG, offering a model-agnostic and principled approach to context engineering. 

---
