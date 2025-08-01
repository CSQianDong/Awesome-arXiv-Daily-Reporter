# DiscRec: Disentangled Semantic-Collaborative Modeling for Generative Recommendation 

**Authors**: Chang Liu, Yimeng Bai, Xiaoyan Zhao, Yang Zhang, Fuli Feng, Wenge Rong  

**Link**: [PDF](https://arxiv.org/pdf/2506.15576)  

**Abstract**: Generative recommendation is emerging as a powerful paradigm that directly generates item predictions, moving beyond traditional matching-based approaches. However, current methods face two key challenges: token-item misalignment, where uniform token-level modeling ignores item-level granularity that is critical for collaborative signal learning, and semantic-collaborative signal entanglement, where collaborative and semantic signals exhibit distinct distributions yet are fused in a unified embedding space, leading to conflicting optimization objectives that limit the recommendation performance.
To address these issues, we propose DiscRec, a novel framework that enables Disentangled Semantic-Collaborative signal modeling with flexible fusion for generative this http URL, DiscRec introduces item-level position embeddings, assigned based on indices within each semantic ID, enabling explicit modeling of item structure in input token this http URL, DiscRec employs a dual-branch module to disentangle the two signals at the embedding layer: a semantic branch encodes semantic signals using original token embeddings, while a collaborative branch applies localized attention restricted to tokens within the same item to effectively capture collaborative signals. A gating mechanism subsequently fuses both branches while preserving the model's ability to model sequential dependencies. Extensive experiments on four real-world datasets demonstrate that DiscRec effectively decouples these signals and consistently outperforms state-of-the-art baselines. Our codes are available on this https URL. 

---
# Multi-Interest Recommendation: A Survey 

**Authors**: Zihao Li, Qiang Chen, Lixin Zou, Aixin Sun, Chenliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.15284)  

**Abstract**: Existing recommendation methods often struggle to model users' multifaceted preferences due to the diversity and volatility of user behavior, as well as the inherent uncertainty and ambiguity of item attributes in practical scenarios. Multi-interest recommendation addresses this challenge by extracting multiple interest representations from users' historical interactions, enabling fine-grained preference modeling and more accurate recommendations. It has drawn broad interest in recommendation research. However, current recommendation surveys have either specialized in frontier recommendation methods or delved into specific tasks and downstream applications. In this work, we systematically review the progress, solutions, challenges, and future directions of multi-interest recommendation by answering the following three questions: (1) Why is multi-interest modeling significantly important for recommendation? (2) What aspects are focused on by multi-interest modeling in recommendation? and (3) How can multi-interest modeling be applied, along with the technical details of the representative modules? We hope that this survey establishes a fundamental framework and delivers a preliminary overview for researchers interested in this field and committed to further exploration. The implementation of multi-interest recommendation summarized in this survey is maintained at this https URL. 

---
# Next-User Retrieval: Enhancing Cold-Start Recommendations via Generative Next-User Modeling 

**Authors**: Yu-Ting Lan, Yang Huo, Yi Shen, Xiao Yang, Zuotao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15267)  

**Abstract**: The item cold-start problem is critical for online recommendation systems, as the success of this phase determines whether high-quality new items can transition to popular ones, receive essential feedback to inspire creators, and thus lead to the long-term retention of creators. However, modern recommendation systems still struggle to address item cold-start challenges due to the heavy reliance on item and historical interactions, which are non-trivial for cold-start items lacking sufficient exposure and feedback. Lookalike algorithms provide a promising solution by extending feedback for new items based on lookalike users. Traditional lookalike algorithms face such limitations: (1) failing to effectively model the lookalike users and further improve recommendations with the existing rule- or model-based methods; and (2) struggling to utilize the interaction signals and incorporate diverse features in modern recommendation systems.
Inspired by lookalike algorithms, we propose Next-User Retrieval, a novel framework for enhancing cold-start recommendations via generative next-user modeling. Specifically, we employ a transformer-based model to capture the unidirectional relationships among recently interacted users and utilize these sequences to generate the next potential user who is most likely to interact with the item. The additional item features are also integrated as prefix prompt embeddings to assist the next-user generation. The effectiveness of Next-User Retrieval is evaluated through both offline experiments and online A/B tests. Our method achieves significant improvements with increases of 0.0142% in daily active users and +0.1144% in publications in Douyin, showcasing its practical applicability and scalability. 

---
# Advancing Loss Functions in Recommender Systems: A Comparative Study with a Rényi Divergence-Based Solution 

**Authors**: Shengjia Zhang, Jiawei Chen, Changdong Li, Sheng Zhou, Qihao Shi, Yan Feng, Chun Chen, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15120)  

**Abstract**: Loss functions play a pivotal role in optimizing recommendation models. Among various loss functions, Softmax Loss (SL) and Cosine Contrastive Loss (CCL) are particularly effective. Their theoretical connections and differences warrant in-depth exploration. This work conducts comprehensive analyses of these losses, yielding significant insights: 1) Common strengths -- both can be viewed as augmentations of traditional losses with Distributional Robust Optimization (DRO), enhancing robustness to distributional shifts; 2) Respective limitations -- stemming from their use of different distribution distance metrics in DRO optimization, SL exhibits high sensitivity to false negative instances, whereas CCL suffers from low data utilization. To address these limitations, this work proposes a new loss function, DrRL, which generalizes SL and CCL by leveraging Rényi-divergence in DRO optimization. DrRL incorporates the advantageous structures of both SL and CCL, and can be demonstrated to effectively mitigate their limitations. Extensive experiments have been conducted to validate the superiority of DrRL on both recommendation accuracy and robustness. 

---
