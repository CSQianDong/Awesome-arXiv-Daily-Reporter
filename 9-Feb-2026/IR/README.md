# On the Efficiency of Sequentially Aware Recommender Systems: Cotten4Rec 

**Authors**: Shankar Veludandi, Gulrukh Kurdistan, Uzma Mushtaque  

**Link**: [PDF](https://arxiv.org/pdf/2602.06935)  

**Abstract**: Sequential recommendation (SR) models predict a user's next interaction by modeling their historical behaviors. Transformer-based SR methods, notably BERT4Rec, effectively capture these patterns but incur significant computational overhead due to extensive intermediate computations associated with Softmax-based attention. We propose Cotten4Rec, a novel SR model utilizing linear-time cosine similarity attention, implemented through a single optimized compute unified device architecture (CUDA) kernel. By minimizing intermediate buffers and kernel-launch overhead, Cotten4Rec substantially reduces resource usage compared to BERT4Rec and the linear-attention baseline, LinRec, especially for datasets with moderate sequence lengths and vocabulary sizes. Evaluations across three benchmark datasets confirm that Cotten4Rec achieves considerable reductions in memory and runtime with minimal compromise in recommendation accuracy, demonstrating Cotten4Rec's viability as an efficient alternative for practical, large-scale sequential recommendation scenarios where computational resources are critical. 

---
# Multimodal Generative Retrieval Model with Staged Pretraining for Food Delivery on Meituan 

**Authors**: Boyu Chen, Tai Guo, Weiyu Cui, Yuqing Li, Xingxing Wang, Chuan Shi, Cheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06654)  

**Abstract**: Multimodal retrieval models are becoming increasingly important in scenarios such as food delivery, where rich multimodal features can meet diverse user needs and enable precise retrieval. Mainstream approaches typically employ a dual-tower architecture between queries and items, and perform joint optimization of intra-tower and inter-tower tasks. However, we observe that joint optimization often leads to certain modalities dominating the training process, while other modalities are neglected. In addition, inconsistent training speeds across modalities can easily result in the one-epoch problem. To address these challenges, we propose a staged pretraining strategy, which guides the model to focus on specialized tasks at each stage, enabling it to effectively attend to and utilize multimodal features, and allowing flexible control over the training process at each stage to avoid the one-epoch problem. Furthermore, to better utilize the semantic IDs that compress high-dimensional multimodal embeddings, we design both generative and discriminative tasks to help the model understand the associations between SIDs, queries, and item features, thereby improving overall performance. Extensive experiments on large-scale real-world Meituan data demonstrate that our method achieves improvements of 3.80%, 2.64%, and 2.17% on R@5, R@10, and R@20, and 5.10%, 4.22%, and 2.09% on N@5, N@10, and N@20 compared to mainstream baselines. Online A/B testing on the Meituan platform shows that our approach achieves a 1.12% increase in revenue and a 1.02% increase in click-through rate, validating the effectiveness and superiority of our method in practical applications. 

---
# R2LED: Equipping Retrieval and Refinement in Lifelong User Modeling with Semantic IDs for CTR Prediction 

**Authors**: Qidong Liu, Gengnan Wang, Zhichen Liu, Moranxin Wang, Zijian Zhang, Xiao Han, Ni Zhang, Tao Qin, Chen Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.06622)  

**Abstract**: Lifelong user modeling, which leverages users' long-term behavior sequences for CTR prediction, has been widely applied in personalized services. Existing methods generally adopted a two-stage "retrieval-refinement" strategy to balance effectiveness and efficiency. However, they still suffer from (i) noisy retrieval due to skewed data distribution and (ii) lack of semantic understanding in refinement. While semantic enhancement, e.g., LLMs modeling or semantic embeddings, offers potential solutions to these two challenges, these approaches face impractical inference costs or insufficient representation granularity. Obsorbing multi-granularity and lightness merits of semantic identity (SID), we propose a novel paradigm that equips retrieval and refinement in Lifelong User Modeling with SEmantic IDs (R2LED) to address these issues. First, we introduce a Multi-route Mixed Retrieval for the retrieval stage. On the one hand, it captures users' interests from various granularities by several parallel recall routes. On the other hand, a mixed retrieval mechanism is proposed to efficiently retrieve candidates from both collaborative and semantic views, reducing noise. Then, for refinement, we design a Bi-level Fusion Refinement, including a target-aware cross-attention for route-level fusion and a gate mechanism for SID-level fusion. It can bridge the gap between semantic and collaborative spaces, exerting the merits of SID. The comprehensive experimental results on two public datasets demonstrate the superiority of our method in both performance and efficiency. To facilitate the reproduction, we have released the code online this https URL. 

---
# TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders 

**Authors**: Yuchen Jiang, Jie Zhu, Xintian Han, Hui Lu, Kunmin Bai, Mingyu Yang, Shikang Wu, Ruihao Zhang, Wenlin Zhao, Shipeng Bai, Sijin Zhou, Huizhi Yang, Tianyi Liu, Wenda Liu, Ziyan Gong, Haoran Ding, Zheng Chai, Deping Xie, Zhe Chen, Yuchao Zheng, Peng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06563)  

**Abstract**: In recent years, the study of scaling laws for large recommendation models has gradually gained attention. Works such as Wukong, HiFormer, and DHEN have attempted to increase the complexity of interaction structures in ranking models and validate scaling laws between performance and parameters/FLOPs by stacking multiple layers. However, their experimental scale remains relatively limited. Our previous work introduced the TokenMixer architecture, an efficient variant of the standard Transformer where the self-attention mechanism is replaced by a simple reshape operation, and the feed-forward network is adapted to a pertoken FFN. The effectiveness of this architecture was demonstrated in the ranking stage by the model presented in the RankMixer paper. However, this foundational TokenMixer architecture itself has several design limitations. In this paper, we propose TokenMixer-Large, which systematically addresses these core issues: sub-optimal residual design, insufficient gradient updates in deep models, incomplete MoE sparsification, and limited exploration of scalability. By leveraging a mixing-and-reverting operation, inter-layer residuals, the auxiliary loss and a novel Sparse-Pertoken MoE architecture, TokenMixer-Large successfully scales its parameters to 7-billion and 15-billion on online traffic and offline experiments, respectively. Currently deployed in multiple scenarios at ByteDance, TokenMixer -Large has achieved significant offline and online performance gains. 

---
# MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model 

**Authors**: Geonmo Gu, Byeongho Heo, Jaemyung Yu, Jaehui Hwang, Taekyung Kim, Sangmin Lee, HeeJae Jun, Yoohoon Kang, Sangdoo Yun, Dongyoon Han  

**Link**: [PDF](https://arxiv.org/pdf/2602.06393)  

**Abstract**: Universal Multimodal embedding models built on Multimodal Large Language Models (MLLMs) have traditionally employed contrastive learning, which aligns representations of query-target pairs across different modalities. Yet, despite its empirical success, they are primarily built on a "single-turn" formulation where each query-target pair is treated as an independent data point. This paradigm leads to computational inefficiency when scaling, as it requires a separate forward pass for each pair and overlooks potential contextual relationships between multiple queries that can relate to the same context. In this work, we introduce Multi-Turn Contrastive Learning (MuCo), a dialogue-inspired framework that revisits this process. MuCo leverages the conversational nature of MLLMs to process multiple, related query-target pairs associated with a single image within a single forward pass. This allows us to extract a set of multiple query and target embeddings simultaneously, conditioned on a shared context representation, amplifying the effective batch size and overall training efficiency. Experiments exhibit MuCo with a newly curated 5M multimodal multi-turn dataset (M3T), which yields state-of-the-art retrieval performance on MMEB and M-BEIR benchmarks, while markedly enhancing both training efficiency and representation coherence across modalities. Code and M3T are available at this https URL 

---
# A methodology for analyzing financial needs hierarchy from social discussions using LLM 

**Authors**: Abhishek Jangra, Sachin Thukral, Arnab Chatterjee, Jayasree Raveendran  

**Link**: [PDF](https://arxiv.org/pdf/2602.06431)  

**Abstract**: This study examines the hierarchical structure of financial needs as articulated in social media discourse, employing generative AI techniques to analyze large-scale textual data. While human needs encompass a broad spectrum from fundamental survival to psychological fulfillment financial needs are particularly critical, influencing both individual well-being and day-to-day decision-making. Our research advances the understanding of financial behavior by utilizing large language models (LLMs) to extract and analyze expressions of financial needs from social media posts. We hypothesize that financial needs are organized hierarchically, progressing from short-term essentials to long-term aspirations, consistent with theoretical frameworks established in the behavioral sciences. Through computational analysis, we demonstrate the feasibility of identifying these needs and validate the presence of a hierarchical structure within them. In addition to confirming this structure, our findings provide novel insights into the content and themes of financial discussions online. By inferring underlying needs from naturally occurring language, this approach offers a scalable and data-driven alternative to conventional survey methodologies, enabling a more dynamic and nuanced understanding of financial behavior in real-world contexts. 

---
