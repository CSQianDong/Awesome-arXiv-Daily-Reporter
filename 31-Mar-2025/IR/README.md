# Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation 

**Authors**: Jiakai Tang, Sunhao Dai, Teng Shi, Jun Xu, Xu Chen, Wen Chen, Wu Jian, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22675)  

**Abstract**: Sequential Recommendation (SeqRec) aims to predict the next item by capturing sequential patterns from users' historical interactions, playing a crucial role in many real-world recommender systems. However, existing approaches predominantly adopt a direct forward computation paradigm, where the final hidden state of the sequence encoder serves as the user representation. We argue that this inference paradigm, due to its limited computational depth, struggles to model the complex evolving nature of user preferences and lacks a nuanced understanding of long-tail items, leading to suboptimal performance. To address this issue, we propose \textbf{ReaRec}, the first inference-time computing framework for recommender systems, which enhances user representations through implicit multi-step reasoning. Specifically, ReaRec autoregressively feeds the sequence's last hidden state into the sequential recommender while incorporating special reasoning position embeddings to decouple the original item encoding space from the multi-step reasoning space. Moreover, we introduce two lightweight reasoning-based learning methods, Ensemble Reasoning Learning (ERL) and Progressive Reasoning Learning (PRL), to further effectively exploit ReaRec's reasoning potential. Extensive experiments on five public real-world datasets and different SeqRec architectures demonstrate the generality and effectiveness of our proposed ReaRec. Remarkably, post-hoc analyses reveal that ReaRec significantly elevates the performance ceiling of multiple sequential recommendation backbones by approximately 30\%-50\%. Thus, we believe this work can open a new and promising avenue for future research in inference-time computing for sequential recommendation. 

---
# Exploring the Effectiveness of Multi-stage Fine-tuning for Cross-encoder Re-rankers 

**Authors**: Francesca Pezzuti, Sean MacAvaney, Nicola Tonellotto  

**Link**: [PDF](https://arxiv.org/pdf/2503.22672)  

**Abstract**: State-of-the-art cross-encoders can be fine-tuned to be highly effective in passage re-ranking. The typical fine-tuning process of cross-encoders as re-rankers requires large amounts of manually labelled data, a contrastive learning objective, and a set of heuristically sampled negatives. An alternative recent approach for fine-tuning instead involves teaching the model to mimic the rankings of a highly effective large language model using a distillation objective. These fine-tuning strategies can be applied either individually, or in sequence. In this work, we systematically investigate the effectiveness of point-wise cross-encoders when fine-tuned independently in a single stage, or sequentially in two stages. Our experiments show that the effectiveness of point-wise cross-encoders fine-tuned using contrastive learning is indeed on par with that of models fine-tuned with multi-stage approaches. Code is available for reproduction at this https URL. 

---
# Improving Low-Resource Retrieval Effectiveness using Zero-Shot Linguistic Similarity Transfer 

**Authors**: Andreas Chari, Sean MacAvaney, Iadh Ounis  

**Link**: [PDF](https://arxiv.org/pdf/2503.22508)  

**Abstract**: Globalisation and colonisation have led the vast majority of the world to use only a fraction of languages, such as English and French, to communicate, excluding many others. This has severely affected the survivability of many now-deemed vulnerable or endangered languages, such as Occitan and Sicilian. These languages often share some characteristics, such as elements of their grammar and lexicon, with other high-resource languages, e.g. French or Italian. They can be clustered into groups of language varieties with various degrees of mutual intelligibility. Current search systems are not usually trained on many of these low-resource varieties, leading search users to express their needs in a high-resource language instead. This problem is further complicated when most information content is expressed in a high-resource language, inhibiting even more retrieval in low-resource languages. We show that current search systems are not robust across language varieties, severely affecting retrieval effectiveness. Therefore, it would be desirable for these systems to leverage the capabilities of neural models to bridge the differences between these varieties. This can allow users to express their needs in their low-resource variety and retrieve the most relevant documents in a high-resource one. To address this, we propose fine-tuning neural rankers on pairs of language varieties, thereby exposing them to their linguistic similarities. We find that this approach improves the performance of the varieties upon which the models were directly trained, thereby regularising these models to generalise and perform better even on unseen language variety pairs. We also explore whether this approach can transfer across language families and observe mixed results that open doors for future research. 

---
# Sell It Before You Make It: Revolutionizing E-Commerce with Personalized AI-Generated Items 

**Authors**: Jianghao Lin, Peng Du, Jiaqi Liu, Weite Li, Yong Yu, Weinan Zhang, Yang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.22182)  

**Abstract**: E-commerce has revolutionized retail, yet its traditional workflows remain inefficient, with significant time and resource costs tied to product design and manufacturing inventory. This paper introduces a novel system deployed at Alibaba that leverages AI-generated items (AIGI) to address these challenges with personalized text-to-image generation for e-commercial product design. AIGI enables an innovative business mode called "sell it before you make it", where merchants can design fashion items and generate photorealistic images with digital models based on textual descriptions. Only when the items have received a certain number of orders, do the merchants start to produce them, which largely reduces reliance on physical prototypes and thus accelerates time to market. For such a promising application, we identify the underlying key scientific challenge, i.e., capturing the users' group-level personalized preferences towards multiple generated candidate images. To this end, we propose a Personalized Group-Level Preference Alignment Framework for Diffusion Models (i.e., PerFusion). We first design PerFusion Reward Model for user preference estimation with a feature-crossing-based personalized plug-in. Then we develop PerFusion with a personalized adaptive network to model diverse preferences across users, and meanwhile derive the group-level preference optimization objective to capture the comparative behaviors among multiple candidates. Both offline and online experiments demonstrate the effectiveness of our proposed algorithm. The AI-generated items have achieved over 13% relative improvements for both click-through rate and conversion rate compared to their human-designed counterparts, validating the revolutionary potential of AI-generated items for e-commercial platforms. 

---
# HyperMAN: Hypergraph-enhanced Meta-learning Adaptive Network for Next POI Recommendation 

**Authors**: Jinze Wang, Tiehua Zhang, Lu Zhang, Yang Bai, Xin Li, Jiong Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.22049)  

**Abstract**: Next Point-of-Interest (POI) recommendation aims to predict users' next locations by leveraging historical check-in sequences. Although existing methods have shown promising results, they often struggle to capture complex high-order relationships and effectively adapt to diverse user behaviors, particularly when addressing the cold-start issue. To address these challenges, we propose Hypergraph-enhanced Meta-learning Adaptive Network (HyperMAN), a novel framework that integrates heterogeneous hypergraph modeling with a difficulty-aware meta-learning mechanism for next POI recommendation. Specifically, three types of heterogeneous hyperedges are designed to capture high-order relationships: user visit behaviors at specific times (Temporal behavioral hyperedge), spatial correlations among POIs (spatial functional hyperedge), and user long-term preferences (user preference hyperedge). Furthermore, a diversity-aware meta-learning mechanism is introduced to dynamically adjust learning strategies, considering users behavioral diversity. Extensive experiments on real-world datasets demonstrate that HyperMAN achieves superior performance, effectively addressing cold start challenges and significantly enhancing recommendation accuracy. 

---
# Empowering Retrieval-based Conversational Recommendation with Contrasting User Preferences 

**Authors**: Heejin Kook, Junyoung Kim, Seongmin Park, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.22005)  

**Abstract**: Conversational recommender systems (CRSs) are designed to suggest the target item that the user is likely to prefer through multi-turn conversations. Recent studies stress that capturing sentiments in user conversations improves recommendation accuracy. However, they employ a single user representation, which may fail to distinguish between contrasting user intentions, such as likes and dislikes, potentially leading to suboptimal performance. To this end, we propose a novel conversational recommender model, called COntrasting user pReference expAnsion and Learning (CORAL). Firstly, CORAL extracts the user's hidden preferences through contrasting preference expansion using the reasoning capacity of the LLMs. Based on the potential preference, CORAL explicitly differentiates the contrasting preferences and leverages them into the recommendation process via preference-aware learning. Extensive experiments show that CORAL significantly outperforms existing methods in three benchmark datasets, improving up to 99.72% in Recall@10. The code and datasets are available at this https URL 

---
# Preference-based Learning with Retrieval Augmented Generation for Conversational Question Answering 

**Authors**: Magdalena Kaiser, Gerhard Weikum  

**Link**: [PDF](https://arxiv.org/pdf/2503.22303)  

**Abstract**: Conversational Question Answering (ConvQA) involves multiple subtasks, i) to understand incomplete questions in their context, ii) to retrieve relevant information, and iii) to generate answers. This work presents PRAISE, a pipeline-based approach for ConvQA that trains LLM adapters for each of the three subtasks. As labeled training data for individual subtasks is unavailable in practice, PRAISE learns from its own generations using the final answering performance as feedback signal without human intervention and treats intermediate information, like relevant evidence, as weakly labeled data. We apply Direct Preference Optimization by contrasting successful and unsuccessful samples for each subtask. In our experiments, we show the effectiveness of this training paradigm: PRAISE shows improvements per subtask and achieves new state-of-the-art performance on a popular ConvQA benchmark, by gaining 15.5 percentage points increase in precision over baselines. 

---
# OAEI-LLM-T: A TBox Benchmark Dataset for Understanding LLM Hallucinations in Ontology Matching Systems 

**Authors**: Zhangcheng Qiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21813)  

**Abstract**: Hallucinations are inevitable in downstream tasks using large language models (LLMs). While addressing hallucinations becomes a substantial challenge for LLM-based ontology matching (OM) systems, we introduce a new benchmark dataset called OAEI-LLM-T. The dataset evolves from the TBox (i.e. schema-matching) datasets in the Ontology Alignment Evaluation Initiative (OAEI), capturing hallucinations of different LLMs performing OM tasks. These OM-specific hallucinations are carefully classified into two primary categories and six sub-categories. We showcase the usefulness of the dataset in constructing the LLM leaderboard and fine-tuning foundational LLMs for LLM-based OM systems. 

---
