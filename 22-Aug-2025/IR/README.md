# Reading Between the Lines: A Study of Thematic Bias in Book Recommender Systems 

**Authors**: Nityaa Kalra, Savvina Daniil  

**Link**: [PDF](https://arxiv.org/pdf/2508.15643)  

**Abstract**: Recommender systems help users discover new content, but can also reinforce existing biases, leading to unfair exposure and reduced diversity. This paper introduces and investigates thematic bias in book recommendations, defined as a disproportionate favouring or neglect of certain book themes. We adopt a multi-stage bias evaluation framework using the Book-Crossing dataset to evaluate thematic bias in recommendations and its impact on different user groups.
Our findings show that thematic bias originates from content imbalances and is amplified by user engagement patterns. By segmenting users based on their thematic preferences, we find that users with niche and long-tail interests receive less personalised recommendations, whereas users with diverse interests receive more consistent recommendations. These findings suggest that recommender systems should be carefully designed to accommodate a broader range of user interests. By contributing to the broader goal of responsible AI, this work also lays the groundwork for extending thematic bias analysis to other domains. 

---
# LongRetriever: Towards Ultra-Long Sequence based Candidate Retrieval for Recommendation 

**Authors**: Ren Qin, Chai Zheng, Xiao Xijun, Zheng Yuchao, Wu Di  

**Link**: [PDF](https://arxiv.org/pdf/2508.15486)  

**Abstract**: Precisely modeling user ultra-long sequences is critical for industrial recommender systems. Current approaches predominantly focus on leveraging ultra-long sequences in the ranking stage, whereas research for the candidate retrieval stage remains under-explored. This paper presents LongRetriever, a practical framework for incorporating ultra-long sequences into the retrieval stage of recommenders. Specifically, we propose in-context training and multi-context retrieval, which enable candidate-specific interaction between user sequence and candidate item, and ensure training-serving consistency under the search-based paradigm. Extensive online A/B testing conducted on a large-scale e-commerce platform demonstrates statistically significant improvements, confirming the framework's effectiveness. Currently, LongRetriever has been fully deployed in the platform, impacting billions of users. 

---
# On Evaluating the Adversarial Robustness of Foundation Models for Multimodal Entity Linking 

**Authors**: Fang Wang, Yongjie Wang, Zonghao Yang, Minghao Hu, Xiaoying Bai  

**Link**: [PDF](https://arxiv.org/pdf/2508.15481)  

**Abstract**: The explosive growth of multimodal data has driven the rapid development of multimodal entity linking (MEL) models. However, existing studies have not systematically investigated the impact of visual adversarial attacks on MEL models. We conduct the first comprehensive evaluation of the robustness of mainstream MEL models under different adversarial attack scenarios, covering two core tasks: Image-to-Text (I2T) and Image+Text-to-Text (IT2T). Experimental results show that current MEL models generally lack sufficient robustness against visual perturbations. Interestingly, contextual semantic information in input can partially mitigate the impact of adversarial perturbations. Based on this insight, we propose an LLM and Retrieval-Augmented Entity Linking (LLM-RetLink), which significantly improves the model's anti-interference ability through a two-stage process: first, extracting initial entity descriptions using large vision models (LVMs), and then dynamically generating candidate descriptive sentences via web-based retrieval. Experiments on five datasets demonstrate that LLM-RetLink improves the accuracy of MEL by 0.4%-35.7%, especially showing significant advantages under adversarial conditions. This research highlights a previously unexplored facet of MEL robustness, constructs and releases the first MEL adversarial example dataset, and sets the stage for future work aimed at strengthening the resilience of multimodal systems in adversarial environments. 

---
# Test-time Corpus Feedback: From Retrieval to RAG 

**Authors**: Mandeep Rathee, Venktesh V, Sean MacAvaney, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2508.15437)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a standard framework for knowledge-intensive NLP tasks, combining large language models (LLMs) with document retrieval from external corpora. Despite its widespread use, most RAG pipelines continue to treat retrieval and reasoning as isolated components, retrieving documents once and then generating answers without further interaction. This static design often limits performance on complex tasks that require iterative evidence gathering or high-precision retrieval. Recent work in both the information retrieval (IR) and NLP communities has begun to close this gap by introducing adaptive retrieval and ranking methods that incorporate feedback. In this survey, we present a structured overview of advanced retrieval and ranking mechanisms that integrate such feedback. We categorize feedback signals based on their source and role in improving the query, retrieved context, or document pool. By consolidating these developments, we aim to bridge IR and NLP perspectives and highlight retrieval as a dynamic, learnable component of end-to-end RAG systems. 

---
# On the Effectiveness of Graph Reordering for Accelerating Approximate Nearest Neighbor Search on GPU 

**Authors**: Yutaro Oguri, Mai Nishimura, Yusuke Matsui  

**Link**: [PDF](https://arxiv.org/pdf/2508.15436)  

**Abstract**: We present the first systematic investigation of graph reordering effects for graph-based Approximate Nearest Neighbor Search (ANNS) on a GPU. While graph-based ANNS has become the dominant paradigm for modern AI applications, recent approaches focus on algorithmic innovations while neglecting memory layout considerations that significantly affect execution time. Our unified evaluation framework enables comprehensive evaluation of diverse reordering strategies across different graph indices through a graph adapter that converts arbitrary graph topologies into a common representation and a GPU-optimized graph traversal engine. We conduct a comprehensive analysis across diverse datasets and state-of-the-art graph indices, introducing analysis metrics that quantify the relationship between structural properties and memory layout effectiveness. Our GPU-targeted reordering achieves up to 15$\%$ QPS improvements while preserving search accuracy, demonstrating that memory layout optimization operates orthogonally to existing algorithmic innovations. We will release all code upon publication to facilitate reproducibility and foster further research. 

---
# TrackRec: Iterative Alternating Feedback with Chain-of-Thought via Preference Alignment for Recommendation 

**Authors**: Yu Xia, Rui Zhong, Zeyu Song, Wei Yang, Junchen Wan, Qingpeng Cai, Chi Lu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15388)  

**Abstract**: The extensive world knowledge and powerful reasoning capabilities of large language models (LLMs) have attracted significant attention in recommendation systems (RS). Specifically, The chain of thought (CoT) has been shown to improve the performance of LLMs on complex reasoning tasks for RS. However, due to the fact that LLMs often suffer from hallucination issues, there is no guarantee that their reasoning CoT is effective. A key challenge is to further enhance the recommendation capabilities of LLMs through effective CoT reasonings. Therefore, we propose \textbf{TrackRec}, a framework designed to enhance reasoning capabilities of LLMs for RS. TrackRec specifically focuses on accurately inferring recommendation CoT \textbf{(RecCoT)} for user preference using the knowledge from LLMs. This RecCoT can serve both as an explanation for the LLM's completion of recommendation tasks and as auxiliary features to assist recommendation models in accomplishing recommendation tasks. TrackRec consists of a RecCoT generator $(G)$ and a RecCoT validator $(V)$. Furthermore, we design alternating feedback learning mechanism that $G$ undergoes direct preference optimization via feedback from $V$ to produce increasingly accurate RecCoT aligned with $V$'s standards. Meanwhile, $V$ is fine-tuned using the inference feedback from $G$ to enhance its validation capabilities in alignment with recommendation tasks. Through iterative alternating feedback learning between $G$ and $V$, TrackRec continuously improves the user preference analysis capability of $G$ and the validation capacity of $V$. Extensive experiments demonstrate the effectiveness of our approach, showing that it surpasses state-of-the-art methods. Moreover, TrackRec has been deployed on a lagre advertising platform with hundreds of millions of users, achieving substantial gains. 

---
# Exploring Scaling Laws of CTR Model for Online Performance Improvement 

**Authors**: Weijiang Lai, Beihong Jin, Jiongyan Zhang, Yiyuan Zheng, Jian Dong, Jia Cheng, Jun Lei, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15326)  

**Abstract**: CTR models play a vital role in improving user experience and boosting business revenue in many online personalized services. However, current CTR models generally encounter bottlenecks in performance improvement. Inspired by the scaling law phenomenon of LLMs, we propose a new paradigm for improving CTR predictions: first, constructing a CTR model with accuracy scalable to the model grade and data size, and then distilling the knowledge implied in this model into its lightweight model that can serve online users. To put it into practice, we construct a CTR model named SUAN (Stacked Unified Attention Network). In SUAN, we propose the UAB as a behavior sequence encoder. A single UAB unifies the modeling of the sequential and non-sequential features and also measures the importance of each user behavior feature from multiple perspectives. Stacked UABs elevate the configuration to a high grade, paving the way for performance improvement. In order to benefit from the high performance of the high-grade SUAN and avoid the disadvantage of its long inference time, we modify the SUAN with sparse self-attention and parallel inference strategies to form LightSUAN, and then adopt online distillation to train the low-grade LightSUAN, taking a high-grade SUAN as a teacher. The distilled LightSUAN has superior performance but the same inference time as the LightSUAN, making it well-suited for online deployment. Experimental results show that SUAN performs exceptionally well and holds the scaling laws spanning three orders of magnitude in model grade and data size, and the distilled LightSUAN outperforms the SUAN configured with one grade higher. More importantly, the distilled LightSUAN has been integrated into an online service, increasing the CTR by 2.81% and CPM by 1.69% while keeping the average inference time acceptable. Our source code is available at this https URL. 

---
# Modeling Long-term User Behaviors with Diffusion-driven Multi-interest Network for CTR Prediction 

**Authors**: Weijiang Lai, Beihong Jin, Yapeng Zhang, Yiyuan Zheng, Rui Zhao, Jian Dong, Jun Lei, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15311)  

**Abstract**: CTR (Click-Through Rate) prediction, crucial for recommender systems and online advertising, etc., has been confirmed to benefit from modeling long-term user behaviors. Nonetheless, the vast number of behaviors and complexity of noise interference pose challenges to prediction efficiency and effectiveness. Recent solutions have evolved from single-stage models to two-stage models. However, current two-stage models often filter out significant information, resulting in an inability to capture diverse user interests and build the complete latent space of user interests. Inspired by multi-interest and generative modeling, we propose DiffuMIN (Diffusion-driven Multi-Interest Network) to model long-term user behaviors and thoroughly explore the user interest space. Specifically, we propose a target-oriented multi-interest extraction method that begins by orthogonally decomposing the target to obtain interest channels. This is followed by modeling the relationships between interest channels and user behaviors to disentangle and extract multiple user interests. We then adopt a diffusion module guided by contextual interests and interest channels, which anchor users' personalized and target-oriented interest types, enabling the generation of augmented interests that align with the latent spaces of user interests, thereby further exploring restricted interest space. Finally, we leverage contrastive learning to ensure that the generated augmented interests align with users' genuine preferences. Extensive offline experiments are conducted on two public datasets and one industrial dataset, yielding results that demonstrate the superiority of DiffuMIN. Moreover, DiffuMIN increased CTR by 1.52% and CPM by 1.10% in online A/B testing. Our source code is available at this https URL. 

---
# REG4Rec: Reasoning-Enhanced Generative Model for Large-Scale Recommendation Systems 

**Authors**: Haibo Xing, Hao Deng, Yucheng Mao, Jinxin Hu, Yi Xu, Hao Zhang, Jiahao Wang, Shizhun Wang, Yu Zhang, Xiaoyi Zeng, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15308)  

**Abstract**: Sequential recommendation aims to predict a user's next action in large-scale recommender systems. While traditional methods often suffer from insufficient information interaction, recent generative recommendation models partially address this issue by directly generating item predictions. To better capture user intents, recent studies have introduced a reasoning process into generative recommendation, significantly improving recommendation performance. However, these approaches are constrained by the singularity of item semantic representations, facing challenges such as limited diversity in reasoning pathways and insufficient reliability in the reasoning process. To tackle these issues, we introduce REG4Rec, a reasoning-enhanced generative model that constructs multiple dynamic semantic reasoning paths alongside a self-reflection process, ensuring high-confidence recommendations. Specifically, REG4Rec utilizes an MoE-based parallel quantization codebook (MPQ) to generate multiple unordered semantic tokens for each item, thereby constructing a larger-scale diverse reasoning space. Furthermore, to enhance the reliability of reasoning, we propose a training reasoning enhancement stage, which includes Preference Alignment for Reasoning (PARS) and a Multi-Step Reward Augmentation (MSRA) strategy. PARS uses reward functions tailored for recommendation to enhance reasoning and reflection, while MSRA introduces future multi-step actions to improve overall generalization. During inference, Consistency-Oriented Self-Reflection for Pruning (CORP) is proposed to discard inconsistent reasoning paths, preventing the propagation of erroneous reasoning. Lastly, we develop an efficient offline training strategy for large-scale recommendation. Experiments on real-world datasets and online evaluations show that REG4Rec delivers outstanding performance and substantial practical value. 

---
# MLLMRec: Exploring the Potential of Multimodal Large Language Models in Recommender Systems 

**Authors**: Yuzhuo Dang, Xin Zhang, Zhiqiang Pan, Yuxiao Duan, Wanyu Chen, Fei Cai, Honghui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15304)  

**Abstract**: Multimodal recommendation typically combines the user behavioral data with the modal features of items to reveal user's preference, presenting superior performance compared to the conventional recommendations. However, existing methods still suffer from two key problems: (1) the initialization methods of user multimodal representations are either behavior-unperceived or noise-contaminated, and (2) the KNN-based item-item graph contains noisy edges with low similarities and lacks audience co-occurrence relationships. To address such issues, we propose MLLMRec, a novel MLLM-driven multimodal recommendation framework with two item-item graph refinement strategies. On the one hand, the item images are first converted into high-quality semantic descriptions using an MLLM, which are then fused with the textual metadata of items. Then, we construct a behavioral description list for each user and feed it into the MLLM to reason about the purified user preference containing interaction motivations. On the other hand, we design the threshold-controlled denoising and topology-aware enhancement strategies to refine the suboptimal item-item graph, thereby enhancing the item representation learning. Extensive experiments on three publicly available datasets demonstrate that MLLMRec achieves the state-of-the-art performance with an average improvement of 38.53% over the best baselines. 

---
# Adversarial Attacks against Neural Ranking Models via In-Context Learning 

**Authors**: Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, Charles L. A. Clarke  

**Link**: [PDF](https://arxiv.org/pdf/2508.15283)  

**Abstract**: While neural ranking models (NRMs) have shown high effectiveness, they remain susceptible to adversarial manipulation. In this work, we introduce Few-Shot Adversarial Prompting (FSAP), a novel black-box attack framework that leverages the in-context learning capabilities of Large Language Models (LLMs) to generate high-ranking adversarial documents. Unlike previous approaches that rely on token-level perturbations or manual rewriting of existing documents, FSAP formulates adversarial attacks entirely through few-shot prompting, requiring no gradient access or internal model instrumentation. By conditioning the LLM on a small support set of previously observed harmful examples, FSAP synthesizes grammatically fluent and topically coherent documents that subtly embed false or misleading information and rank competitively against authentic content. We instantiate FSAP in two modes: FSAP-IntraQ, which leverages harmful examples from the same query to enhance topic fidelity, and FSAP-InterQ, which enables broader generalization by transferring adversarial patterns across unrelated queries. Our experiments on the TREC 2020 and 2021 Health Misinformation Tracks, using four diverse neural ranking models, reveal that FSAP-generated documents consistently outrank credible, factually accurate documents. Furthermore, our analysis demonstrates that these adversarial outputs exhibit strong stance alignment and low detectability, posing a realistic and scalable threat to neural retrieval systems. FSAP also effectively generalizes across both proprietary and open-source LLMs. 

---
# MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation 

**Authors**: Yi Xu, Moyu Zhang, Chenxuan Li, Zhihao Liao, Haibo Xing, Hao Deng, Jinxin Hu, Yu Zhang, Xiaoyi Zeng, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15281)  

**Abstract**: Recommender systems traditionally represent items using unique identifiers (ItemIDs), but this approach struggles with large, dynamic item corpora and sparse long-tail data, limiting scalability and generalization. Semantic IDs, derived from multimodal content such as text and images, offer a promising alternative by mapping items into a shared semantic space, enabling knowledge transfer and improving recommendations for new or rare items. However, existing methods face two key challenges: (1) balancing cross-modal synergy with modality-specific uniqueness, and (2) bridging the semantic-behavioral gap, where semantic representations may misalign with actual user preferences. To address these challenges, we propose Multimodal Mixture-of-Quantization (MMQ), a two-stage framework that trains a novel multimodal tokenizer. First, a shared-specific tokenizer leverages a multi-expert architecture with modality-specific and modality-shared experts, using orthogonal regularization to capture comprehensive multimodal information. Second, behavior-aware fine-tuning dynamically adapts semantic IDs to downstream recommendation objectives while preserving modality information through a multimodal reconstruction loss. Extensive offline experiments and online A/B tests demonstrate that MMQ effectively unifies multimodal synergy, specificity, and behavioral adaptation, providing a scalable and versatile solution for both generative retrieval and discriminative ranking tasks. 

---
# Curriculum Approximate Unlearning for Session-based Recommendation 

**Authors**: Liu Yang, Zhaochun Ren, Ziqi Zhao, Pengjie Ren, Zhumin Chen, Xinyi Li, Shuaiqiang Wang, Dawei Yin, Xin Xin  

**Link**: [PDF](https://arxiv.org/pdf/2508.15263)  

**Abstract**: Approximate unlearning for session-based recommendation refers to eliminating the influence of specific training samples from the recommender without retraining of (sub-)models. Gradient ascent (GA) is a representative method to conduct approximate unlearning. However, there still exist dual challenges to apply GA for session-based recommendation. On the one hand, naive applying of GA could lead to degradation of recommendation performance. On the other hand, existing studies fail to consider the ordering of unlearning samples when simultaneously processing multiple unlearning requests, leading to sub-optimal recommendation performance and unlearning effect. To address the above challenges, we introduce CAU, a curriculum approximate unlearning framework tailored to session-based recommendation. CAU handles the unlearning task with a GA term on unlearning samples. Specifically, to address the first challenge, CAU formulates the overall optimization task as a multi-objective optimization problem, where the GA term for unlearning samples is combined with retaining terms for preserving performance. The multi-objective optimization problem is solved through seeking the Pareto-Optimal solution, which achieves effective unlearning with trivial sacrifice on recommendation performance. To tackle the second challenge, CAU adopts a curriculum-based sequence to conduct unlearning on batches of unlearning samples. The key motivation is to perform unlearning from easy samples to harder ones. To this end, CAU first introduces two metrics to measure the unlearning difficulty, including gradient unlearning difficulty and embedding unlearning difficulty. Then, two strategies, hard-sampling and soft-sampling, are proposed to select unlearning samples according to difficulty scores. 

---
# M-$LLM^3$REC: A Motivation-Aware User-Item Interaction Framework for Enhancing Recommendation Accuracy with LLMs 

**Authors**: Lining Chen, Qingwen Zeng, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15262)  

**Abstract**: Recommendation systems have been essential for both user experience and platform efficiency by alleviating information overload and supporting decision-making. Traditional methods, i.e., content-based filtering, collaborative filtering, and deep learning, have achieved impressive results in recommendation systems. However, the cold-start and sparse-data scenarios are still challenging to deal with. Existing solutions either generate pseudo-interaction sequence, which often introduces redundant or noisy signals, or rely heavily on semantic similarity, overlooking dynamic shifts in user motivation. To address these limitations, this paper proposes a novel recommendation framework, termed M-$LLM^3$REC, which leverages large language models for deep motivational signal extraction from limited user interactions. M-$LLM^3$REC comprises three integrated modules: the Motivation-Oriented Profile Extractor (MOPE), Motivation-Oriented Trait Encoder (MOTE), and Motivational Alignment Recommender (MAR). By emphasizing motivation-driven semantic modeling, M-$LLM^3$REC demonstrates robust, personalized, and generalizable recommendations, particularly boosting performance in cold-start situations in comparison with the state-of-the-art frameworks. 

---
# Multimodal Recommendation via Self-Corrective Preference Alignmen 

**Authors**: Yalong Guan, Xiang Chen, Mingyang Wang, Xiangyu Wu, Lihao Liu, Chao Qi, Shuang Yang, Tingting Gao, Guorui Zhou, Changjian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.14912)  

**Abstract**: With the rapid growth of live streaming platforms, personalized recommendation systems have become pivotal in improving user experience and driving platform revenue. The dynamic and multimodal nature of live streaming content (e.g., visual, audio, textual data) requires joint modeling of user behavior and multimodal features to capture evolving author characteristics. However, traditional methods relying on single-modal features or treating multimodal ones as supplementary struggle to align users' dynamic preferences with authors' multimodal attributes, limiting accuracy and interpretability. To address this, we propose MSPA (Multimodal Self-Corrective Preference Alignment), a personalized author recommendation framework with two components: (1) a Multimodal Preference Composer that uses MLLMs to generate structured preference text and embeddings from users' tipping history; and (2) a Self-Corrective Preference Alignment Recommender that aligns these preferences with authors' multimodal features to improve accuracy and interpretability. Extensive experiments and visualizations show that MSPA significantly improves accuracy, recall, and text quality, outperforming baselines in dynamic live streaming scenarios. 

---
# Personalized Recommendations via Active Utility-based Pairwise Sampling 

**Authors**: Bahar Boroomand, James R. Wright  

**Link**: [PDF](https://arxiv.org/pdf/2508.14911)  

**Abstract**: Recommender systems play a critical role in enhancing user experience by providing personalized suggestions based on user preferences. Traditional approaches often rely on explicit numerical ratings or assume access to fully ranked lists of items. However, ratings frequently fail to capture true preferences due to users' behavioral biases and subjective interpretations of rating scales, while eliciting full rankings is demanding and impractical. To overcome these limitations, we propose a generalized utility-based framework that learns preferences from simple and intuitive pairwise comparisons. Our approach is model-agnostic and designed to optimize for arbitrary, task-specific utility functions, allowing the system's objective to be explicitly aligned with the definition of a high-quality outcome in any given application. A central contribution of our work is a novel utility-based active sampling strategy for preference elicitation. This method selects queries that are expected to provide the greatest improvement to the utility of the final recommended outcome. We ground our preference model in the probabilistic Plackett-Luce framework for pairwise data. To demonstrate the versatility of our approach, we present two distinct experiments: first, an implementation using matrix factorization for a classic movie recommendation task, and second, an implementation using a neural network for a complex candidate selection scenario in university admissions. Experimental results demonstrate that our framework provides a more accurate, data-efficient, and user-centric paradigm for personalized ranking. 

---
# Closing the Performance Gap in Generative Recommenders with Collaborative Tokenization and Efficient Modeling 

**Authors**: Simon Lepage, Jeremie Mary, David Picard  

**Link**: [PDF](https://arxiv.org/pdf/2508.14910)  

**Abstract**: Recent work has explored generative recommender systems as an alternative to traditional ID-based models, reframing item recommendation as a sequence generation task over discrete item tokens. While promising, such methods often underperform in practice compared to well-tuned ID-based baselines like SASRec. In this paper, we identify two key limitations holding back generative approaches: the lack of collaborative signal in item tokenization, and inefficiencies in the commonly used encoder-decoder architecture. To address these issues, we introduce COSETTE, a contrastive tokenization method that integrates collaborative information directly into the learned item representations, jointly optimizing for both content reconstruction and recommendation relevance. Additionally, we propose MARIUS, a lightweight, audio-inspired generative model that decouples timeline modeling from item decoding. MARIUS reduces inference cost while improving recommendation accuracy. Experiments on standard sequential recommendation benchmarks show that our approach narrows, or even eliminates, the performance gap between generative and modern ID-based models, while retaining the benefits of the generative paradigm. 

---
# Collaborative Filtering using Variational Quantum Hopfield Associative Memory 

**Authors**: Amir Kermanshahani, Ebrahim Ardeshir-Larijani, Rakesh Saini, Saif Al-Kuwari  

**Link**: [PDF](https://arxiv.org/pdf/2508.14906)  

**Abstract**: Quantum computing, with its ability to do exponentially faster computation compared to classical systems, has found novel applications in various fields such as machine learning and recommendation systems. Quantum Machine Learning (QML), which integrates quantum computing with machine learning techniques, presents powerful new tools for data processing and pattern recognition. This paper proposes a hybrid recommendation system that combines Quantum Hopfield Associative Memory (QHAM) with deep neural networks to improve the extraction and classification on the MovieLens 1M dataset. User archetypes are clustered into multiple unique groups using the K-Means algorithm and converted into polar patterns through the encoder's activation function. These polar patterns are then integrated into the variational QHAM-based hybrid recommendation model. The system was trained using the MSE loss over 35 epochs in an ideal environment, achieving an ROC value of 0.9795, an accuracy of 0.8841, and an F-1 Score of 0.8786. Trained with the same number of epochs in a noisy environment using a custom Qiskit AER noise model incorporating bit-flip and readout errors with the same probabilities as in real quantum hardware, it achieves an ROC of 0.9177, an accuracy of 0.8013, and an F-1 Score equal to 0.7866, demonstrating consistent performance.
Additionally, we were able to optimize the qubit overhead present in previous QHAM architectures by efficiently updating only one random targeted qubit. This research presents a novel framework that combines variational quantum computing with deep learning, capable of dealing with real-world datasets with comparable performance compared to purely classical counterparts. Additionally, the model can perform similarly well in noisy configurations, showcasing a steady performance and proposing a promising direction for future usage in recommendation systems. 

---
# Privacy Preserving Inference of Personalized Content for Out of Matrix Users 

**Authors**: Michael Sun, Tai Vu, Andrew Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14905)  

**Abstract**: Recommender systems for niche and dynamic communities face persistent challenges from data sparsity, cold start users and items, and privacy constraints. Traditional collaborative filtering and content-based approaches underperform in these settings, either requiring invasive user data or failing when preference histories are absent. We present DeepNaniNet, a deep neural recommendation framework that addresses these challenges through an inductive graph-based architecture combining user-item interactions, item-item relations, and rich textual review embeddings derived from BERT. Our design enables cold start recommendations without profile mining, using a novel "content basket" user representation and an autoencoder-based generalization strategy for unseen users. We introduce AnimeULike, a new dataset of 10,000 anime titles and 13,000 users, to evaluate performance in realistic scenarios with high proportions of guest or low-activity users. DeepNaniNet achieves state-of-the-art cold start results on the CiteULike benchmark, matches DropoutNet in user recall without performance degradation for out-of-matrix users, and outperforms Weighted Matrix Factorization (WMF) and DropoutNet on AnimeULike warm start by up to 7x and 1.5x in Recall@100, respectively. Our findings demonstrate that DeepNaniNet delivers high-quality, privacy-preserving recommendations in data-sparse, cold start-heavy environments while effectively integrating heterogeneous content sources. 

---
# Stemming -- The Evolution and Current State with a Focus on Bangla 

**Authors**: Abhijit Paul, Mashiat Amin Farin, Sharif Md. Abdullah, Ahmedul Kabir, Zarif Masud, Shebuti Rayana  

**Link**: [PDF](https://arxiv.org/pdf/2508.15711)  

**Abstract**: Bangla, the seventh most widely spoken language worldwide with 300 million native speakers, faces digital under-representation due to limited resources and lack of annotated datasets. Stemming, a critical preprocessing step in language analysis, is essential for low-resource, highly-inflectional languages like Bangla, because it can reduce the complexity of algorithms and models by significantly reducing the number of words the algorithm needs to consider. This paper conducts a comprehensive survey of stemming approaches, emphasizing the importance of handling morphological variants effectively. While exploring the landscape of Bangla stemming, it becomes evident that there is a significant gap in the existing literature. The paper highlights the discontinuity from previous research and the scarcity of accessible implementations for replication. Furthermore, it critiques the evaluation methodologies, stressing the need for more relevant metrics. In the context of Bangla's rich morphology and diverse dialects, the paper acknowledges the challenges it poses. To address these challenges, the paper suggests directions for Bangla stemmer development. It concludes by advocating for robust Bangla stemmers and continued research in the field to enhance language analysis and processing. 

---
# Benchmarking Computer Science Survey Generation 

**Authors**: Weihang Su, Anzhe Xie, Qingyao Ai, Jianming Long, Jiaxin Mao, Ziyi Ye, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15658)  

**Abstract**: Scientific survey articles play a vital role in summarizing research progress, yet their manual creation is becoming increasingly infeasible due to the rapid growth of academic literature. While large language models (LLMs) offer promising capabilities for automating this process, progress in this area is hindered by the absence of standardized benchmarks and evaluation protocols. To address this gap, we introduce SurGE (Survey Generation Evaluation), a new benchmark for evaluating scientific survey generation in the computer science domain. SurGE consists of (1) a collection of test instances, each including a topic description, an expert-written survey, and its full set of cited references, and (2) a large-scale academic corpus of over one million papers that serves as the retrieval pool. In addition, we propose an automated evaluation framework that measures generated surveys across four dimensions: information coverage, referencing accuracy, structural organization, and content quality. Our evaluation of diverse LLM-based approaches shows that survey generation remains highly challenging, even for advanced self-reflection frameworks. These findings highlight the complexity of the task and the necessity for continued research. We have open-sourced all the code, data, and models at: this https URL 

---
# TComQA: Extracting Temporal Commonsense from Text 

**Authors**: Lekshmi R Nair, Arun Sankar, Koninika Pal  

**Link**: [PDF](https://arxiv.org/pdf/2508.15274)  

**Abstract**: Understanding events necessitates grasping their temporal context, which is often not explicitly stated in natural language. For example, it is not a trivial task for a machine to infer that a museum tour may last for a few hours, but can not take months. Recent studies indicate that even advanced large language models (LLMs) struggle in generating text that require reasoning with temporal commonsense due to its infrequent explicit mention in text. Therefore, automatically mining temporal commonsense for events enables the creation of robust language models. In this work, we investigate the capacity of LLMs to extract temporal commonsense from text and evaluate multiple experimental setups to assess their effectiveness. Here, we propose a temporal commonsense extraction pipeline that leverages LLMs to automatically mine temporal commonsense and use it to construct TComQA, a dataset derived from SAMSum and RealNews corpora. TComQA has been validated through crowdsourcing and achieves over 80\% precision in extracting temporal commonsense. The model trained with TComQA also outperforms an LLM fine-tuned on existing dataset of temporal question answering task. 

---
# Retrieval-Augmented Review Generation for Poisoning Recommender Systems 

**Authors**: Shiyi Yang, Xinshu Li, Guanglin Zhou, Chen Wang, Xiwei Xu, Liming Zhu, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15252)  

**Abstract**: Recent studies have shown that recommender systems (RSs) are highly vulnerable to data poisoning attacks, where malicious actors inject fake user profiles, including a group of well-designed fake ratings, to manipulate recommendations. Due to security and privacy constraints in practice, attackers typically possess limited knowledge of the victim system and thus need to craft profiles that have transferability across black-box RSs. To maximize the attack impact, the profiles often remains imperceptible. However, generating such high-quality profiles with the restricted resources is challenging. Some works suggest incorporating fake textual reviews to strengthen the profiles; yet, the poor quality of the reviews largely undermines the attack effectiveness and imperceptibility under the practical setting.
To tackle the above challenges, in this paper, we propose to enhance the quality of the review text by harnessing in-context learning (ICL) capabilities of multimodal foundation models. To this end, we introduce a demonstration retrieval algorithm and a text style transfer strategy to augment the navie ICL. Specifically, we propose a novel practical attack framework named RAGAN to generate high-quality fake user profiles, which can gain insights into the robustness of RSs. The profiles are generated by a jailbreaker and collaboratively optimized on an instructional agent and a guardian to improve the attack transferability and imperceptibility. Comprehensive experiments on various real-world datasets demonstrate that RAGAN achieves the state-of-the-art poisoning attack performance. 

---
# See Beyond a Single View: Multi-Attribution Learning Leads to Better Conversion Rate Prediction 

**Authors**: Sishuo Chen, Zhangming Chan, Xiang-Rong Sheng, Lei Zhang, Sheng Chen, Chenghuan Hou, Han Zhu, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.15217)  

**Abstract**: Conversion rate (CVR) prediction is a core component of online advertising systems, where the attribution mechanisms-rules for allocating conversion credit across user touchpoints-fundamentally determine label generation and model optimization. While many industrial platforms support diverse attribution mechanisms (e.g., First-Click, Last-Click, Linear, and Data-Driven Multi-Touch Attribution), conventional approaches restrict model training to labels from a single production-critical attribution mechanism, discarding complementary signals in alternative attribution perspectives.
To address this limitation, we propose a novel Multi-Attribution Learning (MAL) framework for CVR prediction that integrates signals from multiple attribution perspectives to better capture the underlying patterns driving user conversions. Specifically, MAL is a joint learning framework consisting of two core components: the Attribution Knowledge Aggregator (AKA) and the Primary Target Predictor (PTP). AKA is implemented as a multi-task learner that integrates knowledge extracted from diverse attribution labels. PTP, in contrast, focuses on the task of generating well-calibrated conversion probabilities that align with the system-optimized attribution metric (e.g., CVR under the Last-Click attribution), ensuring direct compatibility with industrial deployment requirements. Additionally, we propose CAT, a novel training strategy that leverages the Cartesian product of all attribution label combinations to generate enriched supervision signals. This design substantially enhances the performance of the attribution knowledge aggregator. Empirical evaluations demonstrate the superiority of MAL over single-attribution learning baselines, achieving +0.51% GAUC improvement on offline metrics. Online experiments demonstrate that MAL achieved a +2.6% increase in ROI (Return on Investment). 

---
# LongRecall: A Structured Approach for Robust Recall Evaluation in Long-Form Text 

**Authors**: MohamamdJavad Ardestani, Ehsan Kamalloo, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2508.15085)  

**Abstract**: LongRecall. The completeness of machine-generated text, ensuring that it captures all relevant information, is crucial in domains such as medicine and law and in tasks like list-based question answering (QA), where omissions can have serious consequences. However, existing recall metrics often depend on lexical overlap, leading to errors with unsubstantiated entities and paraphrased answers, while LLM-as-a-Judge methods with long holistic prompts capture broader semantics but remain prone to misalignment and hallucinations without structured verification. We introduce LongRecall, a general three-stage recall evaluation framework that decomposes answers into self-contained facts, successively narrows plausible candidate matches through lexical and semantic filtering, and verifies their alignment through structured entailment checks. This design reduces false positives and false negatives while accommodating diverse phrasings and contextual variations, serving as a foundational building block for systematic recall assessment. We evaluate LongRecall on three challenging long-form QA benchmarks using both human annotations and LLM-based judges, demonstrating substantial improvements in recall accuracy over strong lexical and LLM-as-a-Judge baselines. 

---
# Alpha Berkeley: A Scalable Framework for the Orchestration of Agentic Systems 

**Authors**: Thorsten Hellert, João Montenegro, Antonin Sulc  

**Link**: [PDF](https://arxiv.org/pdf/2508.15066)  

**Abstract**: Coordinating workflows across heterogeneous control systems remains a central challenge in safety-critical environments such as scientific facilities, industrial plants, and energy infrastructures. Language-model-driven agents offer a natural interface for these tasks, but existing approaches often lack scalability, reliability, and human oversight. We introduce the Alpha Berkeley Framework, a production-ready architecture for scalable agentic systems that integrate conversational context with robust tool orchestration. The framework features dynamic capability classification to select only relevant tools per task, a plan-first orchestration model that generates execution plans with explicit dependencies and optional human approval, context-aware task extraction that combines dialogue history with external memory and domain resources, and production-ready execution environments with checkpointing, artifact management, and modular deployment. We demonstrate its versatility through two case studies: a tutorial-style wind farm monitoring example and a deployment at the Advanced Light Source particle accelerator. These results establish Alpha Berkeley as a reliable and transparent framework for agentic systems in high-stakes domains. 

---
