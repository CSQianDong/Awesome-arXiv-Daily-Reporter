# Consensus-aware Contrastive Learning for Group Recommendation 

**Authors**: Soyoung Kim, Dongjun Lee, Jaekwang Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.13703)  

**Abstract**: Group recommendation aims to provide personalized item suggestions to a group of users by reflecting their collective preferences. A fundamental challenge in this task is deriving a consensus that adequately represents the diverse interests of individual group members. Despite advancements made by deep learning-based models, existing approaches still struggle in two main areas: (1) Capturing consensus in small-group settings, which are more prevalent in real-world applications, and (2) Balancing individual preferences with overall group performance, particularly in hypergraph-based methods that tend to emphasize group accuracy at the expense of personalization. To address these challenges, we introduce a Consensus-aware Contrastive Learning for Group Recommendation (CoCoRec) that models group consensus through contrastive learning. CoCoRec utilizes a transformer encoder to jointly learn user and group representations, enabling richer modeling of intra-group dynamics. Additionally, the contrastive objective helps reduce overfitting from high-frequency user interactions, leading to more robust and representative group embeddings. Experiments conducted on four benchmark datasets show that CoCoRec consistently outperforms state-of-the-art baselines in both individual and group recommendation scenarios, highlighting the effectiveness of consensus-aware contrastive learning in group recommendation tasks. 

---
# Adaptive Long-term Embedding with Denoising and Augmentation for Recommendation 

**Authors**: Zahra Akhlaghi, Mostafa Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2504.13614)  

**Abstract**: The rapid growth of the internet has made personalized recommendation systems indispensable. Graph-based sequential recommendation systems, powered by Graph Neural Networks (GNNs), effectively capture complex user-item interactions but often face challenges such as noise and static representations. In this paper, we introduce the Adaptive Long-term Embedding with Denoising and Augmentation for Recommendation (ALDA4Rec) method, a novel model that constructs an item-item graph, filters noise through community detection, and enriches user-item interactions. Graph Convolutional Networks (GCNs) are then employed to learn short-term representations, while averaging, GRUs, and attention mechanisms are utilized to model long-term embeddings. An MLP-based adaptive weighting strategy is further incorporated to dynamically optimize long-term user preferences. Experiments conducted on four real-world datasets demonstrate that ALDA4Rec outperforms state-of-the-art baselines, delivering notable improvements in both accuracy and robustness. The source code is available at this https URL. 

---
# Contextualizing Spotify's Audiobook List Recommendations with Descriptive Shelves 

**Authors**: Gustavo Penha, Alice Wang, Martin Achenbach, Kristen Sheets, Sahitya Mantravadi, Remi Galvez, Nico Guetta-Jeanrenaud, Divya Narayanan, Ofeliya Kalaydzhyan, Hugues Bouchard  

**Link**: [PDF](https://arxiv.org/pdf/2504.13572)  

**Abstract**: In this paper, we propose a pipeline to generate contextualized list recommendations with descriptive shelves in the domain of audiobooks. By creating several shelves for topics the user has an affinity to, e.g. Uplifting Women's Fiction, we can help them explore their recommendations according to their interests and at the same time recommend a diverse set of items. To do so, we use Large Language Models (LLMs) to enrich each item's metadata based on a taxonomy created for this domain. Then we create diverse descriptive shelves for each user. A/B tests show improvements in user engagement and audiobook discovery metrics, demonstrating benefits for users and content creators. 

---
# Improving Sequential Recommenders through Counterfactual Augmentation of System Exposure 

**Authors**: Ziqi Zhao, Zhaochun Ren, Jiyuan Yang, Zuming Yan, Zihan Wang, Liu Yang, Pengjie Ren, Zhumin Chen, Maarten de Rijke, Xin Xin  

**Link**: [PDF](https://arxiv.org/pdf/2504.13482)  

**Abstract**: In sequential recommendation (SR), system exposure refers to items that are exposed to the user. Typically, only a few of the exposed items would be interacted with by the user. Although SR has achieved great success in predicting future user interests, existing SR methods still fail to fully exploit system exposure data. Most methods only model items that have been interacted with, while the large volume of exposed but non-interacted items is overlooked. Even methods that consider the whole system exposure typically train the recommender using only the logged historical system exposure, without exploring unseen user interests.
In this paper, we propose counterfactual augmentation over system exposure for sequential recommendation (CaseRec). To better model historical system exposure, CaseRec introduces reinforcement learning to account for different exposure rewards. CaseRec uses a decision transformer-based sequential model to take an exposure sequence as input and assigns different rewards according to the user feedback. To further explore unseen user interests, CaseRec proposes to perform counterfactual augmentation, where exposed original items are replaced with counterfactual items. Then, a transformer-based user simulator is proposed to predict the user feedback reward for the augmented items. Augmentation, together with the user simulator, constructs counterfactual exposure sequences to uncover new user interests. Finally, CaseRec jointly uses the logged exposure sequences with the counterfactual exposure sequences to train a decision transformer-based sequential model for generating recommendation. Experiments on three real-world benchmarks show the effectiveness of CaseRec. Our code is available at this https URL. 

---
# Multi-Type Context-Aware Conversational Recommender Systems via Mixture-of-Experts 

**Authors**: Jie Zou, Cheng Lin, Weikang Guo, Zheng Wang, Jiwei Wei, Yang Yang, Hengtao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.13655)  

**Abstract**: Conversational recommender systems enable natural language conversations and thus lead to a more engaging and effective recommendation scenario. As the conversations for recommender systems usually contain limited contextual information, many existing conversational recommender systems incorporate external sources to enrich the contextual information. However, how to combine different types of contextual information is still a challenge. In this paper, we propose a multi-type context-aware conversational recommender system, called MCCRS, effectively fusing multi-type contextual information via mixture-of-experts to improve conversational recommender systems. MCCRS incorporates both structured information and unstructured information, including the structured knowledge graph, unstructured conversation history, and unstructured item reviews. It consists of several experts, with each expert specialized in a particular domain (i.e., one specific contextual information). Multiple experts are then coordinated by a ChairBot to generate the final results. Our proposed MCCRS model takes advantage of different contextual information and the specialization of different experts followed by a ChairBot breaks the model bottleneck on a single contextual information. Experimental results demonstrate that our proposed MCCRS method achieves significantly higher performance compared to existing baselines. 

---
