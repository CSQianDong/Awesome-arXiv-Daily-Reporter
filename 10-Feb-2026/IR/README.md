# Automatic In-Domain Exemplar Construction and LLM-Based Refinement of Multi-LLM Expansions for Query Expansion 

**Authors**: Minghan Li, Ercong Nie, Siqi Zhao, Tongna Chen, Huiping Huang, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.08917)  

**Abstract**: Query expansion with large language models is promising but often relies on hand-crafted prompts, manually chosen exemplars, or a single LLM, making it non-scalable and sensitive to domain shift. We present an automated, domain-adaptive QE framework that builds in-domain exemplar pools by harvesting pseudo-relevant passages using a BM25-MonoT5 pipeline. A training-free cluster-based strategy selects diverse demonstrations, yielding strong and stable in-context QE without supervision. To further exploit model complementarity, we introduce a two-LLM ensemble in which two heterogeneous LLMs independently generate expansions and a refinement LLM consolidates them into one coherent expansion. Across TREC DL20, DBPedia, and SciFact, the refined ensemble delivers consistent and statistically significant gains over BM25, Rocchio, zero-shot, and fixed few-shot baselines. The framework offers a reproducible testbed for exemplar selection and multi-LLM generation, and a practical, label-free solution for real-world QE. 

---
# OmniReview: A Large-scale Benchmark and LLM-enhanced Framework for Realistic Reviewer Recommendation 

**Authors**: Yehua Huang, Penglei Sun, Zebin Chen, Zhenheng Tang, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08896)  

**Abstract**: Academic peer review remains the cornerstone of scholarly validation, yet the field faces some challenges in data and methods. From the data perspective, existing research is hindered by the scarcity of large-scale, verified benchmarks and oversimplified evaluation metrics that fail to reflect real-world editorial workflows. To bridge this gap, we present OmniReview, a comprehensive dataset constructed by integrating multi-source academic platforms encompassing comprehensive scholarly profiles through the disambiguation pipeline, yielding 202, 756 verified review records. Based on this data, we introduce a three-tier hierarchical evaluaion framework to assess recommendations from recall to precise expert identification. From the method perspective, existing embedding-based approaches suffer from the information bottleneck of semantic compression and limited interpretability. To resolve these method limitations, we propose Profiling Scholars with Multi-gate Mixture-of-Experts (Pro-MMoE), a novel framework that synergizes Large Language Models (LLMs) with Multi-task Learning. Specifically, it utilizes LLM-generated semantic profiles to preserve fine-grained expertise nuances and interpretability, while employing a Task-Adaptive MMoE architecture to dynamically balance conflicting evaluation goals. Comprehensive experiments demonstrate that Pro-MMoE achieves state-of-the-art performance across six of seven metrics, establishing a new benchmark for realistic reviewer recommendation. 

---
# Contrastive Learning for Diversity-Aware Product Recommendations in Retail 

**Authors**: Vasileios Karlis, Ezgi Yıldırım, David Vos, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2602.08886)  

**Abstract**: Recommender systems often struggle with long-tail distributions and limited item catalog exposure, where a small subset of popular items dominates recommendations. This challenge is especially critical in large-scale online retail settings with extensive and diverse product assortments. This paper introduces an approach to enhance catalog coverage without compromising recommendation quality in the existing digital recommendation pipeline at IKEA Retail. Drawing inspiration from recent advances in negative sampling to address popularity bias, we integrate contrastive learning with carefully selected negative samples. Through offline and online evaluations, we demonstrate that our method improves catalog coverage, ensuring a more diverse set of recommendations yet preserving strong recommendation performance. 

---
# Whose Name Comes Up? Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation 

**Authors**: Lisette Espin-Noboa, Gonzalo Gabriel Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2602.08873)  

**Abstract**: Large language models (LLMs) are increasingly used for academic expert recommendation. Existing audits typically evaluate model outputs in isolation, largely ignoring end-user inference-time interventions. As a result, it remains unclear whether failures such as refusals, hallucinations, and uneven coverage stem from model choice or deployment decisions. We introduce LLMScholarBench, a benchmark for auditing LLM-based scholar recommendation that jointly evaluates model infrastructure and end-user interventions across multiple tasks. LLMScholarBench measures both technical quality and social representation using nine metrics. We instantiate the benchmark in physics expert recommendation and audit 22 LLMs under temperature variation, representation-constrained prompting, and retrieval-augmented generation (RAG) via web search. Our results show that end-user interventions do not yield uniform improvements but instead redistribute error across dimensions. Higher temperature degrades validity, consistency, and factuality. Representation-constrained prompting improves diversity at the expense of factuality, while RAG primarily improves technical quality while reducing diversity and parity. Overall, end-user interventions reshape trade-offs rather than providing a general fix. We release code and data that can be adapted to other disciplines by replacing domain-specific ground truth and metrics. 

---
# AMEM4Rec: Leveraging Cross-User Similarity for Memory Evolution in Agentic LLM Recommenders 

**Authors**: Minh-Duc Nguyen, Hai-Dang Kieu, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2602.08837)  

**Abstract**: Agentic systems powered by Large Language Models (LLMs) have shown strong potential in recommender systems but remain hindered by several challenges. Fine-tuning LLMs is parameter-inefficient, and prompt-based agentic reasoning is limited by context length and hallucination risk. Moreover, existing agentic recommendation systems predominantly leverages semantic knowledge while neglecting the collaborative filtering (CF) signals essential for implicit preference modeling. To address these limitations, we propose AMEM4Rec, an agentic LLM-based recommender that learns collaborative signals in an end-to-end manner through cross-user memory evolution. AMEM4Rec stores abstract user behavior patterns from user histories in a global memory pool. Within this pool, memories are linked to similar existing ones and iteratively evolved to reinforce shared cross-user patterns, enabling the system to become aware of CF signals without relying on a pre-trained CF model. Extensive experiments on Amazon and MIND datasets show that AMEM4Rec consistently outperforms state-of-the-art LLM-based recommenders, demonstrating the effectiveness of evolving memory-guided collaborative filtering. 

---
# SA-CAISR: Stage-Adaptive and Conflict-Aware Incremental Sequential Recommendation 

**Authors**: Xiaomeng Song, Xinru Wang, Hanbing Wang, Hongyu Lu, Yu Chen, Zhaochun Ren, Zhumin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.08678)  

**Abstract**: Sequential recommendation (SR) aims to predict a user's next action by learning from their historical interaction sequences. In real-world applications, these models require periodic updates to adapt to new interactions and evolving user preferences. While incremental learning methods facilitate these updates, they face significant challenges. Replay-based approaches incur high memory and computational costs, and regularization-based methods often struggle to discard outdated or conflicting knowledge. To overcome these challenges, we propose SA-CAISR, a Stage-Adaptive and Conflict-Aware Incremental Sequential Recommendation framework. As a buffer-free framework, SA-CAISR operates using only the old model and new data, directly addressing the high costs of replay-based techniques. SA-CAISR introduces a novel Fisher-weighted knowledge-screening mechanism that dynamically identifies outdated knowledge by estimating parameter-level conflicts between the old model and new data, allowing our approach to selectively remove obsolete knowledge while preserving compatible historical patterns. This dynamic balance between stability and adaptability allows our method to achieve a new state-of-the-art performance in incremental SR. Specifically, SA-CAISR improves Recall@20 by 2.0%, MRR@20 by 1.2%, and NDCG@20 by 1.4% on average across datasets, while reducing memory usage by 97.5% and training time by 46.9% compared to the best baselines. This efficiency allows real-world systems to rapidly update user profiles with minimal computational overhead, ensuring more timely and accurate recommendations. 

---
# SRSUPM: Sequential Recommender System Based on User Psychological Motivation 

**Authors**: Yicheng Di, Yuan Liu, Zhi Chen, Jingcai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.08667)  

**Abstract**: Sequential recommender infers users' evolving psychological motivations from historical interactions to recommend the next preferred items. Most existing methods compress recent behaviors into a single vector and optimize it toward a single observed target item, but lack explicit modeling of psychological motivation shift. As a result, they struggle to uncover the distributional patterns across different shift degrees and to capture collaborative knowledge that is sensitive to psychological motivation shift. We propose a general framework, the Sequential Recommender System Based on User Psychological Motivation, to enhance sequential recommenders with psychological motivation shift-aware user modeling. Specifically, the Psychological Motivation Shift Assessment quantitatively measures psychological motivation shift; guided by PMSA, the Shift Information Construction models dynamically evolving multi-level shift states, and the Psychological Motivation Shift-driven Information Decomposition decomposes and regularizes representations across shift levels. Moreover, the Psychological Motivation Shift Information Matching strengthens collaborative patterns related to psychological motivation shift to learn more discriminative user representations. Extensive experiments on three public benchmarks show that SRSUPM consistently outperforms representative baselines on diverse sequential recommender tasks. 

---
# OneLive: Dynamically Unified Generative Framework for Live-Streaming Recommendation 

**Authors**: Shen Wang, Yusheng Huang, Ruochen Yang, Shuang Wen, Pengbo Xu, Jiangxia Cao, Yueyang Liu, Kuo Cai, Chengcheng Guo, Shiyao Wang, Xinchen Luo, Qiang Luo, Ruiming Tang, Shuang Yang, Zhaojie Liu, Guorui Zhou, Han Li, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2602.08612)  

**Abstract**: Live-streaming recommender system serves as critical infrastructure that bridges the patterns of real-time interactions between users and authors. Similar to traditional industrial recommender systems, live-streaming recommendation also relies on cascade architectures to support large-scale concurrency. Recent advances in generative recommendation unify the multi-stage recommendation process with Transformer-based architectures, offering improved scalability and higher computational efficiency. However, the inherent complexity of live-streaming prevents the direct transfer of these methods to live-streaming scenario, where continuously evolving content, limited lifecycles, strict real-time constraints, and heterogeneous multi-objectives introduce unique challenges that invalidate static tokenization and conventional model framework. To address these issues, we propose OneLive, a dynamically unified generative recommendation framework tailored for live-streaming scenario. OneLive integrates four key components: (i) A Dynamic Tokenizer that continuously encodes evolving real-time live content fused with behavior signal through residual quantization; (ii) A Time-Aware Gated Attention mechanism that explicitly models temporal dynamics for timely decision making; (iii) An efficient decoder-only generative architecture enhanced with Sequential MTP and QK Norm for stable training and accelerated inference; (iv) A Unified Multi-Objective Alignment Framework reinforces policy optimization for personalized preferences. 

---
# RankGR: Rank-Enhanced Generative Retrieval with Listwise Direct Preference Optimization in Recommendation 

**Authors**: Kairui Fu, Changfa Wu, Kun Yuan, Binbin Cao, Dunxian Huang, Yuliang Yan, Junjun Zheng, Jianning Zhang, Silu Zhou, Jian Wu, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08575)  

**Abstract**: Generative retrieval (GR) has emerged as a promising paradigm in recommendation systems by autoregressively decoding identifiers of target items. Despite its potential, current approaches typically rely on the next-token prediction schema, which treats each token of the next interacted items as the sole target. This narrow focus 1) limits their ability to capture the nuanced structure of user preferences, and 2) overlooks the deep interaction between decoded identifiers and user behavior sequences. In response to these challenges, we propose RankGR, a Rank-enhanced Generative Retrieval method that incorporates listwise direct preference optimization for recommendation. RankGR decomposes the retrieval process into two complementary stages: the Initial Assessment Phase (IAP) and the Refined Scoring Phase (RSP). In IAP, we incorporate a novel listwise direct preference optimization strategy into GR, thus facilitating a more comprehensive understanding of the hierarchical user preferences and more effective partial-order modeling. The RSP then refines the top-{\lambda} candidates generated by IAP with interactions towards input sequences using a lightweight scoring module, leading to more precise candidate evaluation. Both phases are jointly optimized under a unified GR model, ensuring consistency and efficiency. Additionally, we implement several practical improvements in training and deployment, ultimately achieving a real-time system capable of handling nearly ten thousand requests per second. Extensive offline performance on both research and industrial datasets, as well as the online gains on the "Guess You Like" section of Taobao, validate the effectiveness and scalability of RankGR. 

---
# QARM V2: Quantitative Alignment Multi-Modal Recommendation for Reasoning User Sequence Modeling 

**Authors**: Tian Xia, Jiaqi Zhang, Yueyang Liu, Hongjian Dou, Tingya Yin, Jiangxia Cao, Xulei Liang, Tianlu Xie, Lihao Liu, Xiang Chen, Shen Wang, Changxin Lao, Haixiang Gan, Jinkai Yu, Keting Cen, Lu Hao, Xu Zhang, Qiqiang Zhong, Zhongbo Sun, Yiyu Wang, Shuang Yang, Mingxin Wen, Xiangyu Wu, Shaoguo Liu, Tingting Gao, Zhaojie Liu, Han Li, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2602.08559)  

**Abstract**: With the evolution of large language models (LLMs), there is growing interest in leveraging their rich semantic understanding to enhance industrial recommendation systems (RecSys). Traditional RecSys relies on ID-based embeddings for user sequence modeling in the General Search Unit (GSU) and Exact Search Unit (ESU) paradigm, which suffers from low information density, knowledge isolation, and weak generalization ability. While LLMs offer complementary strengths with dense semantic representations and strong generalization, directly applying LLM embeddings to RecSys faces critical challenges: representation unmatch with business objectives and representation unlearning end-to-end with downstream tasks. In this paper, we present QARM V2, a unified framework that bridges LLM semantic understanding with RecSys business requirements for user sequence modeling. 

---
# DA-RAG: Dynamic Attributed Community Search for Retrieval-Augmented Generation 

**Authors**: Xingyuan Zeng, Zuohan Wu, Yue Wang, Chen Zhang, Quanming Yao, Libin Zheng, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.08545)  

**Abstract**: Owing to their unprecedented comprehension capabilities, large language models (LLMs) have become indispensable components of modern web search engines. From a technical perspective, this integration represents retrieval-augmented generation (RAG), which enhances LLMs by grounding them in external knowledge bases. A prevalent technical approach in this context is graph-based RAG (G-RAG). However, current G-RAG methodologies frequently underutilize graph topology, predominantly focusing on low-order structures or pre-computed static communities. This limitation affects their effectiveness in addressing dynamic and complex queries. Thus, we propose DA-RAG, which leverages attributed community search (ACS) to extract relevant subgraphs based on the queried question dynamically. DA-RAG captures high-order graph structures, allowing for the retrieval of self-complementary knowledge. Furthermore, DA-RAG is equipped with a chunk-layer oriented graph index, which facilitates efficient multi-granularity retrieval while significantly reducing both computational and economic costs. We evaluate DA-RAG on multiple datasets, demonstrating that it outperforms existing RAG methods by up to 40% in head-to-head comparisons across four metrics while reducing index construction time and token overhead by up to 37% and 41%, respectively. 

---
# PIT: A Dynamic Personalized Item Tokenizer for End-to-End Generative Recommendation 

**Authors**: Huanjie Wang, Xinchen Luo, Honghui Bao, Zhang Zixing, Lejian Ren, Yunfan Wu, Hongwei Zhang, Liwei Guan, Guang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.08530)  

**Abstract**: Generative Recommendation has revolutionized recommender systems by reformulating retrieval as a sequence generation task over discrete item identifiers. Despite the progress, existing approaches typically rely on static, decoupled tokenization that ignores collaborative signals. While recent methods attempt to integrate collaborative signals into item identifiers either during index construction or through end-to-end modeling, they encounter significant challenges in real-world production environments. Specifically, the volatility of collaborative signals leads to unstable tokenization, and current end-to-end strategies often devolve into suboptimal two-stage training rather than achieving true co-evolution. To bridge this gap, we propose PIT, a dynamic Personalized Item Tokenizer framework for end-to-end generative recommendation, which employs a co-generative architecture that harmonizes collaborative patterns through collaborative signal alignment and synchronizes item tokenizer with generative recommender via a co-evolution learning. This enables the dynamic, joint, end-to-end evolution of both index construction and recommendation. Furthermore, a one-to-many beam index ensures scalability and robustness, facilitating seamless integration into large-scale industrial deployments. Extensive experiments on real-world datasets demonstrate that PIT consistently outperforms competitive baselines. In a large-scale deployment at Kuaishou, an online A/B test yielded a substantial 0.402% uplift in App Stay Time, validating the framework's effectiveness in dynamic industrial environments. 

---
# Hybrid Pooling with LLMs via Relevance Context Learning 

**Authors**: David Otero, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2602.08457)  

**Abstract**: High-quality relevance judgements over large query sets are essential for evaluating Information Retrieval (IR) systems, yet manual annotation remains costly and time-consuming. Large Language Models (LLMs) have recently shown promise as automatic relevance assessors, but their reliability is still limited. Most existing approaches rely on zero-shot prompting or In-Context Learning (ICL) with a small number of labeled examples. However, standard ICL treats examples as independent instances and fails to explicitly capture the underlying relevance criteria of a topic, restricting its ability to generalize to unseen query-document pairs. To address this limitation, we introduce Relevance Context Learning (RCL), a novel framework that leverages human relevance judgements to explicitly model topic-specific relevance criteria. Rather than directly using labeled examples for in-context prediction, RCL first prompts an LLM (Instructor LLM) to analyze sets of judged query-document pairs and generate explicit narratives that describe what constitutes relevance for a given topic. These relevance narratives are then used as structured prompts to guide a second LLM (Assessor LLM) in producing relevance judgements. To evaluate RCL in a realistic data collection setting, we propose a hybrid pooling strategy in which a shallow depth-\textit{k} pool from participating systems is judged by human assessors, while the remaining documents are labeled by LLMs. Experimental results demonstrate that RCL substantially outperforms zero-shot prompting and consistently improves over standard ICL. Overall, our findings indicate that transforming relevance examples into explicit, context-aware relevance narratives is a more effective way of exploiting human judgements for LLM-based IR dataset construction. 

---
# A Sketch+Text Composed Image Retrieval Dataset for Thangka 

**Authors**: Jinyu Xu, Yi Sun, Jiangling Zhang, Qing Xie, Daomin Ji, Zhifeng Bao, Jiachen Li, Yanchun Ma, Yongjian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08411)  

**Abstract**: Composed Image Retrieval (CIR) enables image retrieval by combining multiple query modalities, but existing benchmarks predominantly focus on general-domain imagery and rely on reference images with short textual modifications. As a result, they provide limited support for retrieval scenarios that require fine-grained semantic reasoning, structured visual understanding, and domain-specific knowledge. In this work, we introduce CIRThan, a sketch+text Composed Image Retrieval dataset for Thangka imagery, a culturally grounded and knowledge-specific visual domain characterized by complex structures, dense symbolic elements, and domain-dependent semantic conventions. CIRThan contains 2,287 high-quality Thangka images, each paired with a human-drawn sketch and hierarchical textual descriptions at three semantic levels, enabling composed queries that jointly express structural intent and multi-level semantic specification. We provide standardized data splits, comprehensive dataset analysis, and benchmark evaluations of representative supervised and zero-shot CIR methods. Experimental results reveal that existing CIR approaches, largely developed for general-domain imagery, struggle to effectively align sketch-based abstractions and hierarchical textual semantics with fine-grained Thangka images, particularly without in-domain supervision. We believe CIRThan offers a valuable benchmark for advancing sketch+text CIR, hierarchical semantic modeling, and multimodal retrieval in cultural heritage and other knowledge-specific visual domains. The dataset is publicly available at this https URL. 

---
# IRB: Automated Generation of Robust Factuality Benchmarks 

**Authors**: Lam Thanh Do, Bhagyashree Taleka, Hozaifa Ammar Bhutta, Vikram Sharma Mailthody, Kevin Chen-Chuan Chang, Wen-mei Hwu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08070)  

**Abstract**: Static benchmarks for RAG systems often suffer from rapid saturation and require significant manual effort to maintain robustness. To address this, we present IRB, a framework for automatically generating benchmarks to evaluate the factuality of RAG systems. IRB employs a structured generation pipeline utilizing \textit{factual scaffold} and \textit{algorithmic scaffold}. We utilize IRB to construct a benchmark and evaluate frontier LLMs and retrievers. Our results demonstrate that IRB poses a significant challenge for frontier LLMs in the closed-book setting. Furthermore, our evaluation suggests that reasoning LLMs are more reliable, and that improving the retrieval component may yield more cost-effective gains in RAG system correctness than scaling the generator. 

---
# Learning to Alleviate Familiarity Bias in Video Recommendation 

**Authors**: Zheng Ren, Yi Wu, Jianan Lu, Acar Ary, Yiqu Liu, Li Wei, Lukasz Heldt  

**Link**: [PDF](https://arxiv.org/pdf/2602.07987)  

**Abstract**: Modern video recommendation systems aim to optimize user engagement and platform objectives, yet often face structural exposure imbalances caused by behavioral biases. In this work, we focus on the post-ranking stage and present LAFB (Learning to Alleviate Familiarity Bias), a lightweight and model-agnostic framework designed to mitigate familiarity bias in recommendation outputs. LAFB models user-content familiarity using discrete and continuous interaction features, and estimates personalized debiasing factors to adjust user rating prediction scores, thereby reducing the dominance of familiar content in the final ranking. We conduct large-scale offline evaluations and online A/B testing in a real-world recommendation system, under a unified serving stack that also compares LAFB with deployable popularity-oriented remedies. Results show that LAFB increases novel watch-time share and improves exposure for emerging creators and overall content diversity, while maintaining stable overall watch time and short-term satisfaction. LAFB has already been launched in the post-ranking stage of YouTube's recommendation system, demonstrating its effectiveness in real-world applications. 

---
# SimGR: Escaping the Pitfalls of Generative Decoding in LLM-based Recommendation 

**Authors**: Yuanbo Zhao, Ruochen Liu, Senzhang Wang, Jun Yin, Yuxin Dong, Huan Gong, Hao Chen, Shirui Pan, Chengqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07847)  

**Abstract**: A core objective in recommender systems is to accurately model the distribution of user preferences over items to enable personalized recommendations. Recently, driven by the strong generative capabilities of large language models (LLMs), LLM-based generative recommendation has become increasingly popular. However, we observe that existing methods inevitably introduce systematic bias when estimating item-level preference distributions. Specifically, autoregressive generation suffers from incomplete coverage due to beam search pruning, while parallel generation distorts probabilities by assuming token independence. We attribute this issue to a fundamental modeling mismatch: these methods approximate item-level distributions via token-level generation, which inherently induces approximation errors. Through both theoretical analysis and empirical validation, we demonstrate that token-level generation cannot faithfully substitute item-level generation, leading to biased item distributions. To address this, we propose \textbf{Sim}ply \textbf{G}enerative \textbf{R}ecommendation (\textbf{SimGR}), a framework that directly models item-level preference distributions in a shared latent space and ranks items by similarity, thereby aligning the modeling objective with recommendation and mitigating distributional distortion. Extensive experiments across multiple datasets and LLM backbones show that SimGR consistently outperforms existing generative recommenders. Our code is available at this https URL 

---
# SAGE: Scalable AI Governance & Evaluation 

**Authors**: Benjamin Le, Xueying Lu, Nick Stern, Wenqiong Liu, Igor Lapchuk, Xiang Li, Baofen Zheng, Kevin Rosenberg, Jiewen Huang, Zhe Zhang, Abraham Cabangbang, Satej Milind Wagle, Jianqiang Shen, Raghavan Muthuregunathan, Abhinav Gupta, Mathew Teoh, Andrew Kirk, Thomas Kwan, Jingwei Wu, Wenjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07840)  

**Abstract**: Evaluating relevance in large-scale search systems is fundamentally constrained by the governance gap between nuanced, resource-constrained human oversight and the high-throughput requirements of production systems. While traditional approaches rely on engagement proxies or sparse manual review, these methods often fail to capture the full scope of high-impact relevance failures. We present \textbf{SAGE} (Scalable AI Governance \& Evaluation), a framework that operationalizes high-quality human product judgment as a scalable evaluation signal. At the core of SAGE is a bidirectional calibration loop where natural-language \emph{Policy}, curated \emph{Precedent}, and an \emph{LLM Surrogate Judge} co-evolve. SAGE systematically resolves semantic ambiguities and misalignments, transforming subjective relevance judgment into an executable, multi-dimensional rubric with near human-level agreement. To bridge the gap between frontier model reasoning and industrial-scale inference, we apply teacher-student distillation to transfer high-fidelity judgments into compact student surrogates at \textbf{92$\times$} lower cost. Deployed within LinkedIn Search ecosystems, SAGE guided model iteration through simulation-driven development, distilling policy-aligned models for online serving and enabling rapid offline evaluation. In production, it powered policy oversight that measured ramped model variants and detected regressions invisible to engagement metrics. Collectively, these drove a \textbf{0.25\%} lift in LinkedIn daily active users. 

---
# Generative Reasoning Re-ranker 

**Authors**: Mingfu Liang, Yufei Li, Jay Xu, Kavosh Asadi, Xi Liu, Shuo Gu, Kaushik Rangadurai, Frank Shyu, Shuaiwen Wang, Song Yang, Zhijing Li, Jiang Liu, Mengying Sun, Fei Tian, Xiaohan Wei, Chonglin Sun, Jacob Tao, Shike Mei, Hamed Firooz, Wenlin Chen, Luke Simon  

**Link**: [PDF](https://arxiv.org/pdf/2602.07774)  

**Abstract**: Recent studies increasingly explore Large Language Models (LLMs) as a new paradigm for recommendation systems due to their scalability and world knowledge. However, existing work has three key limitations: (1) most efforts focus on retrieval and ranking, while the reranking phase, critical for refining final recommendations, is largely overlooked; (2) LLMs are typically used in zero-shot or supervised fine-tuning settings, leaving their reasoning abilities, especially those enhanced through reinforcement learning (RL) and high-quality reasoning data, underexploited; (3) items are commonly represented by non-semantic IDs, creating major scalability challenges in industrial systems with billions of identifiers. To address these gaps, we propose the Generative Reasoning Reranker (GR2), an end-to-end framework with a three-stage training pipeline tailored for reranking. First, a pretrained LLM is mid-trained on semantic IDs encoded from non-semantic IDs via a tokenizer achieving $\ge$99% uniqueness. Next, a stronger larger-scale LLM generates high-quality reasoning traces through carefully designed prompting and rejection sampling, which are used for supervised fine-tuning to impart foundational reasoning skills. Finally, we apply Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO), enabling scalable RL supervision with verifiable rewards designed specifically for reranking. Experiments on two real-world datasets demonstrate GR2's effectiveness: it surpasses the state-of-the-art OneRec-Think by 2.4% in Recall@5 and 1.3% in NDCG@5. Ablations confirm that advanced reasoning traces yield substantial gains across metrics. We further find that RL reward design is crucial in reranking: LLMs tend to exploit reward hacking by preserving item order, motivating conditional verifiable rewards to mitigate this behavior and optimize reranking performance. 

---
# HypRAG: Hyperbolic Dense Retrieval for Retrieval Augmented Generation 

**Authors**: Hiren Madhu, Ngoc Bui, Ali Maatouk, Leandros Tassiulas, Smita Krishnaswamy, Menglin Yang, Sukanta Ganguly, Kiran Srinivasan, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2602.07739)  

**Abstract**: Embedding geometry plays a fundamental role in retrieval quality, yet dense retrievers for retrieval-augmented generation (RAG) remain largely confined to Euclidean space. However, natural language exhibits hierarchical structure from broad topics to specific entities that Euclidean embeddings fail to preserve, causing semantically distant documents to appear spuriously similar and increasing hallucination risk. To address these limitations, we introduce hyperbolic dense retrieval, developing two model variants in the Lorentz model of hyperbolic space: HyTE-FH, a fully hyperbolic transformer, and HyTE-H, a hybrid architecture projecting pre-trained Euclidean embeddings into hyperbolic space. To prevent representational collapse during sequence aggregation, we introduce the Outward Einstein Midpoint, a geometry-aware pooling operator that provably preserves hierarchical structure. On MTEB, HyTE-FH outperforms equivalent Euclidean baselines, while on RAGBench, HyTE-H achieves up to 29% gains over Euclidean baselines in context relevance and answer relevance using substantially smaller models than current state-of-the-art retrievers. Our analysis also reveals that hyperbolic representations encode document specificity through norm-based separation, with over 20% radial increase from general to specific concepts, a property absent in Euclidean embeddings, underscoring the critical role of geometric inductive bias in faithful RAG systems. 

---
# MSN: A Memory-based Sparse Activation Scaling Framework for Large-scale Industrial Recommendation 

**Authors**: Shikang Wu, Hui Lu, Jinqiu Jin, Zheng Chai, Shiyong Hong, Junjie Zhang, Shanlei Mu, Kaiyuan Ma, Tianyi Liu, Yuchao Zheng, Zhe Wang, Jingjian Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.07526)  

**Abstract**: Scaling deep learning recommendation models is an effective way to improve model expressiveness. Existing approaches often incur substantial computational overhead, making them difficult to deploy in large-scale industrial systems under strict latency constraints. Recent sparse activation scaling methods, such as Sparse Mixture-of-Experts, reduce computation by activating only a subset of parameters, but still suffer from high memory access costs and limited personalization capacity due to the large size and small number of experts. To address these challenges, we propose MSN, a memory-based sparse activation scaling framework for recommendation models. MSN dynamically retrieves personalized representations from a large parameterized memory and integrates them into downstream feature interaction modules via a memory gating mechanism, enabling fine-grained personalization with low computational overhead. To enable further expansion of the memory capacity while keeping both computational and memory access costs under control, MSN adopts a Product-Key Memory (PKM) mechanism, which factorizes the memory retrieval complexity from linear time to sub-linear complexity. In addition, normalization and over-parameterization techniques are introduced to maintain balanced memory utilization and prevent memory retrieval collapse. We further design customized Sparse-Gather operator and adopt the AirTopK operator to improve training and inference efficiency in industrial settings. Extensive experiments demonstrate that MSN consistently improves recommendation performance while maintaining high efficiency. Moreover, MSN has been successfully deployed in the Douyin Search Ranking System, achieving significant gains over deployed state-of-the-art models in both offline evaluation metrics and large-scale online A/B test. 

---
# IGMiRAG: Intuition-Guided Retrieval-Augmented Generation with Adaptive Mining of In-Depth Memory 

**Authors**: Xingliang Hou, Yuyan Liu, Qi Sun, haoxiu wang, Hao Hu, Shaoyi Du, Zhiqiang Tian  

**Link**: [PDF](https://arxiv.org/pdf/2602.07525)  

**Abstract**: Retrieval-augmented generation (RAG) equips large language models (LLMs) with reliable knowledge memory. To strengthen cross-text associations, recent research integrates graphs and hypergraphs into RAG to capture pairwise and multi-entity relations as structured links. However, their misaligned memory organization necessitates costly, disjointed retrieval. To address these limitations, we propose IGMiRAG, a framework inspired by human intuition-guided reasoning. It constructs a hierarchical heterogeneous hypergraph to align multi-granular knowledge, incorporating deductive pathways to simulate realistic memory structures. During querying, IGMiRAG distills intuitive strategies via a question parser to control mining depth and memory window, and activates instantaneous memories as anchors using dual-focus retrieval. Mirroring human intuition, the framework guides retrieval resource allocation dynamically. Furthermore, we design a bidirectional diffusion algorithm that navigates deductive paths to mine in-depth memories, emulating human reasoning processes. Extensive evaluations indicate IGMiRAG outperforms the state-of-the-art baseline by 4.8% EM and 5.0% F1 overall, with token costs adapting to task complexity (average 6.3k+, minimum 3.0k+). This work presents a cost-effective RAG paradigm that improves both efficiency and effectiveness. 

---
# MDL: A Unified Multi-Distribution Learner in Large-scale Industrial Recommendation through Tokenization 

**Authors**: Shanlei Mu, Yuchen Jiang, Shikang Wu, Shiyong Hong, Tianmu Sha, Junjie Zhang, Jie Zhu, Zhe Chen, Zhe Wang, Jingjian Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.07520)  

**Abstract**: Industrial recommender systems increasingly adopt multi-scenario learning (MSL) and multi-task learning (MTL) to handle diverse user interactions and contexts, but existing approaches suffer from two critical drawbacks: (1) underutilization of large-scale model parameters due to limited interaction with complex feature modules, and (2) difficulty in jointly modeling scenario and task information in a unified framework. To address these challenges, we propose a unified \textbf{M}ulti-\textbf{D}istribution \textbf{L}earning (MDL) framework, inspired by the "prompting" paradigm in large language models (LLMs). MDL treats scenario and task information as specialized tokens rather than auxiliary inputs or gating signals. Specifically, we introduce a unified information tokenization module that transforms features, scenarios, and tasks into a unified tokenized format. To facilitate deep interaction, we design three synergistic mechanisms: (1) feature token self-attention for rich feature interactions, (2) domain-feature attention for scenario/task-adaptive feature activation, and (3) domain-fused aggregation for joint distribution prediction. By stacking these interactions, MDL enables scenario and task information to "prompt" and activate the model's vast parameter space in a bottom-up, layer-wise manner. Extensive experiments on real-world industrial datasets demonstrate that MDL significantly outperforms state-of-the-art MSL and MTL baselines. Online A/B testing on Douyin Search platform over one month yields +0.0626\% improvement in LT30 and -0.3267\% reduction in change query rate. MDL has been fully deployed in production, serving hundreds of millions of users daily. 

---
# High Fidelity Textual User Representation over Heterogeneous Sources via Reinforcement Learning 

**Authors**: Rajat Arora, Ye Tao, Jianqiang Shen, Ping Liu, Muchen Wu, Qianqi Shen, Benjamin Le, Fedor Borisyuk, Jingwei Wu, Wenjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07333)  

**Abstract**: Effective personalization on large-scale job platforms requires modeling members based on heterogeneous textual sources, including profiles, professional data, and search activity logs. As recommender systems increasingly adopt Large Language Models (LLMs), creating unified, interpretable, and concise representations from heterogeneous sources becomes critical, especially for latency-sensitive online environments. In this work, we propose a novel Reinforcement Learning (RL) framework to synthesize a unified textual representation for each member. Our approach leverages implicit user engagement signals (e.g., clicks, applies) as the primary reward to distill salient information. Additionally, the framework is complemented by rule-based rewards that enforce formatting and length constraints. Extensive offline experiments across multiple LinkedIn products, one of the world's largest job platforms, demonstrate significant improvements in key downstream business metrics. This work provides a practical, labeling-free, and scalable solution for constructing interpretable user representations that are directly compatible with LLM-based systems. 

---
# Semantic Search At LinkedIn 

**Authors**: Fedor Borisyuk, Sriram Vasudevan, Muchen Wu, Guoyao Li, Benjamin Le, Shaobo Zhang, Qianqi Kay Shen, Yuchin Juan, Kayhan Behdin, Liming Dong, Kaixu Yang, Shusen Jing, Ravi Pothamsetty, Rajat Arora, Sophie Yanying Sheng, Vitaly Abdrashitov, Yang Zhao, Lin Su, Xiaoqing Wang, Chujie Zheng, Sarang Metkar, Rupesh Gupta, Igor Lapchuk, David N. Racca, Madhumitha Mohan, Yanbo Li, Haojun Li, Saloni Gandhi, Xueying Lu, Chetan Bhole, Ali Hooshmand, Xin Yang, Raghavan Muthuregunathan, Jiajun Zhang, Mathew Teoh, Adam Coler, Abhinav Gupta, Xiaojing Ma, Sundara Raman Ramachandran, Morteza Ramezani, Yubo Wang, Lijuan Zhang, Richard Li, Jian Sheng, Chanh Nguyen, Yen-Chi Chen, Chuanrui Zhu, Claire Zhang, Jiahao Xu, Deepti Kulkarni, Qing Lan, Arvind Subramaniam, Ata Fatahibaarzi, Steven Shimizu, Yanning Chen, Zhipeng Wang, Ran He, Zhengze Zhou, Qingquan Song, Yun Dai, Caleb Johnson, Ping Liu, Shaghayegh Gharghabi, Gokulraj Mohanasundaram, Juan Bottaro, Santhosh Sachindran, Qi Guo, Yunxiang Ren, Chengming Jiang, Di Mo, Luke Simon, Jianqiang Shen, Jingwei Wu, Wenjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07309)  

**Abstract**: Semantic search with large language models (LLMs) enables retrieval by meaning rather than keyword overlap, but scaling it requires major inference efficiency advances. We present LinkedIn's LLM-based semantic search framework for AI Job Search and AI People Search, combining an LLM relevance judge, embedding-based retrieval, and a compact Small Language Model trained via multi-teacher distillation to jointly optimize relevance and engagement. A prefill-oriented inference architecture co-designed with model pruning, context compression, and text-embedding hybrid interactions boosts ranking throughput by over 75x under a fixed latency constraint while preserving near-teacher-level NDCG, enabling one of the first production LLM-based ranking systems with efficiency comparable to traditional approaches and delivering significant gains in quality and user engagement. 

---
# LIT-GRAPH: Evaluating Deep vs. Shallow Graph Embeddings for High-Quality Text Recommendation in Domain-Specific Knowledge Graphs 

**Authors**: Nirmal Gelal, Chloe Snow, Kathleen M. Jagodnik, Ambyr Rios, Hande Küçük McGinty  

**Link**: [PDF](https://arxiv.org/pdf/2602.07307)  

**Abstract**: This study presents LIT-GRAPH (Literature Graph for Recommendation and Pedagogical Heuristics), a novel knowledge graph-based recommendation system designed to scaffold high school English teachers in selecting diverse, pedagogically aligned instructional literature. The system is built upon an ontology for English literature, addressing the challenge of curriculum stagnation, where we compare four graph embedding paradigms: DeepWalk, Biased Random Walk (BRW), Hybrid (concatenated DeepWalk and BRW vectors), and the deep model Relational Graph Convolutional Network (R-GCN). Results reveal a critical divergence: while shallow models excelled in structural link prediction, R-GCN dominated semantic ranking. By leveraging relation-specific message passing, the deep model prioritizes pedagogical relevance over raw connectivity, resulting in superior, high-quality, domain-specific recommendations. 

---
# Principled Synthetic Data Enables the First Scaling Laws for LLMs in Recommendation 

**Authors**: Benyu Zhang, Qiang Zhang, Jianpeng Cheng, Hong-You Chen, Qifei Wang, Wei Sun, Shen Li, Jia Li, Jiahao Wu, Xiangjun Fan, Hong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2602.07298)  

**Abstract**: Large Language Models (LLMs) represent a promising frontier for recommender systems, yet their development has been impeded by the absence of predictable scaling laws, which are crucial for guiding research and optimizing resource allocation. We hypothesize that this may be attributed to the inherent noise, bias, and incompleteness of raw user interaction data in prior continual pre-training (CPT) efforts. This paper introduces a novel, layered framework for generating high-quality synthetic data that circumvents such issues by creating a curated, pedagogical curriculum for the LLM. We provide powerful, direct evidence for the utility of our curriculum by showing that standard sequential models trained on our principled synthetic data significantly outperform ($+130\%$ on recall@100 for SasRec) models trained on real data in downstream ranking tasks, demonstrating its superiority for learning generalizable user preference patterns. Building on this, we empirically demonstrate, for the first time, robust power-law scaling for an LLM that is continually pre-trained on our high-quality, recommendation-specific data. Our experiments reveal consistent and predictable perplexity reduction across multiple synthetic data modalities. These findings establish a foundational methodology for reliable scaling LLM capabilities in the recommendation domain, thereby shifting the research focus from mitigating data deficiencies to leveraging high-quality, structured information. 

---
# Progressive Searching for Retrieval in RAG 

**Authors**: Taehee Jeong, Xingzhe Zhao, Peizu Li, Markus Valvur, Weihua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.07297)  

**Abstract**: Retrieval Augmented Generation (RAG) is a promising technique for mitigating two key limitations of large language models (LLMs): outdated information and hallucinations. RAG system stores documents as embedding vectors in a database. Given a query, search is executed to find the most related documents. Then, the topmost matching documents are inserted into LLMs' prompt to generate a response. Efficient and accurate searching is critical for RAG to get relevant information. We propose a cost-effective searching algorithm for retrieval process. Our progressive searching algorithm incrementally refines the candidate set through a hierarchy of searches, starting from low-dimensional embeddings and progressing into a higher, target-dimensionality. This multi-stage approach reduces retrieval time while preserving the desired accuracy. Our findings demonstrate that progressive search in RAG systems achieves a balance between dimensionality, speed, and accuracy, enabling scalable and high-performance retrieval even for large databases. 

---
# Sequences as Nodes for Contrastive Multimodal Graph Recommendation 

**Authors**: Bucher Sahyouni, Matthew Vowels, Liqun Chen, Simon Hadfield  

**Link**: [PDF](https://arxiv.org/pdf/2602.07208)  

**Abstract**: To tackle cold-start and data sparsity issues in recommender systems, numerous multimodal, sequential, and contrastive techniques have been proposed. While these augmentations can boost recommendation performance, they tend to add noise and disrupt useful semantics. To address this, we propose MuSICRec (Multimodal Sequence-Item Contrastive Recommender), a multi-view graph-based recommender that combines collaborative, sequential, and multimodal signals. We build a sequence-item (SI) view by attention pooling over the user's interacted items to form sequence nodes. We propagate over the SI graph, obtaining a second view organically as an alternative to artificial data augmentation, while simultaneously injecting sequential context signals. Additionally, to mitigate modality noise and align the multimodal information, the contribution of text and visual features is modulated according to an ID-guided gate.
We evaluate under a strict leave-two-out split against a broad range of sequential, multimodal, and contrastive baselines. On the Amazon Baby, Sports, and Electronics datasets, MuSICRec outperforms state-of-the-art baselines across all model types. We observe the largest gains for short-history users, mitigating sparsity and cold-start challenges. Our code is available at this https URL and will be made publicly available. 

---
# Multimodal Enhancement of Sequential Recommendation 

**Authors**: Bucher Sahyouni, Matthew Vowels, Liqun Chen, Simon Hadfield  

**Link**: [PDF](https://arxiv.org/pdf/2602.07207)  

**Abstract**: We propose a novel recommender framework, MuSTRec (Multimodal and Sequential Transformer-based Recommendation), that unifies multimodal and sequential recommendation paradigms. MuSTRec captures cross-item similarities and collaborative filtering signals, by building item-item graphs from extracted text and visual features. A frequency-based self-attention module additionally captures the short- and long-term user preferences. Across multiple Amazon datasets, MuSTRec demonstrates superior performance (up to 33.5% improvement) over multimodal and sequential state-of-the-art baselines. Finally, we detail some interesting facets of this new recommendation paradigm. These include the need for a new data partitioning regime, and a demonstration of how integrating user embeddings into sequential recommendation leads to drastically increased short-term metrics (up to 200% improvement) on smaller datasets. Our code is availabe at this https URL and will be made publicly available. 

---
# Reasoning-Augmented Representations for Multimodal Retrieval 

**Authors**: Jianrui Zhang, Anirudh Sundara Rajan, Brandon Han, Soochahn Lee, Sukanta Ganguly, Yong Jae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.07125)  

**Abstract**: Universal Multimodal Retrieval (UMR) seeks any-to-any search across text and vision, yet modern embedding models remain brittle when queries require latent reasoning (e.g., resolving underspecified references or matching compositional constraints). We argue this brittleness is often data-induced: when images carry "silent" evidence and queries leave key semantics implicit, a single embedding pass must both reason and compress, encouraging spurious feature matching. We propose a data-centric framework that decouples these roles by externalizing reasoning before retrieval. Using a strong Vision--Language Model, we make implicit semantics explicit by densely captioning visual evidence in corpus entries, resolving ambiguous multimodal references in queries, and rewriting verbose instructions into concise retrieval constraints. Inference-time enhancement alone is insufficient; the retriever must be trained on these semantically dense representations to avoid distribution shift and fully exploit the added signal. Across M-BEIR, our reasoning-augmented training method yields consistent gains over strong baselines, with ablations showing that corpus enhancement chiefly benefits knowledge-intensive queries while query enhancement is critical for compositional modification requests. We publicly release our code at this https URL. 

---
# Large Language Models for Geolocation Extraction in Humanitarian Crisis Response 

**Authors**: G. Cafferata, T. Demarco, K. Kalimeri, Y. Mejova, M.G. Beiró  

**Link**: [PDF](https://arxiv.org/pdf/2602.08872)  

**Abstract**: Humanitarian crises demand timely and accurate geographic information to inform effective response efforts. Yet, automated systems that extract locations from text often reproduce existing geographic and socioeconomic biases, leading to uneven visibility of crisis-affected regions. This paper investigates whether Large Language Models (LLMs) can address these geographic disparities in extracting location information from humanitarian documents. We introduce a two-step framework that combines few-shot LLM-based named entity recognition with an agent-based geocoding module that leverages context to resolve ambiguous toponyms. We benchmark our approach against state-of-the-art pretrained and rule-based systems using both accuracy and fairness metrics across geographic and socioeconomic dimensions. Our evaluation uses an extended version of the HumSet dataset with refined literal toponym annotations. Results show that LLM-based methods substantially improve both the precision and fairness of geolocation extraction from humanitarian texts, particularly for underrepresented regions. By bridging advances in LLM reasoning with principles of responsible and inclusive AI, this work contributes to more equitable geospatial data systems for humanitarian response, advancing the goal of leaving no place behind in crisis analytics. 

---
# Welfarist Formulations for Diverse Similarity Search 

**Authors**: Siddharth Barman, Nirjhar Das, Shivam Gupta, Kirankumar Shiragur  

**Link**: [PDF](https://arxiv.org/pdf/2602.08742)  

**Abstract**: Nearest Neighbor Search (NNS) is a fundamental problem in data structures with wide-ranging applications, such as web search, recommendation systems, and, more recently, retrieval-augmented generations (RAG). In such recent applications, in addition to the relevance (similarity) of the returned neighbors, diversity among the neighbors is a central requirement. In this paper, we develop principled welfare-based formulations in NNS for realizing diversity across attributes. Our formulations are based on welfare functions -- from mathematical economics -- that satisfy central diversity (fairness) and relevance (economic efficiency) axioms. With a particular focus on Nash social welfare, we note that our welfare-based formulations provide objective functions that adaptively balance relevance and diversity in a query-dependent manner. Notably, such a balance was not present in the prior constraint-based approach, which forced a fixed level of diversity and optimized for relevance. In addition, our formulation provides a parametric way to control the trade-off between relevance and diversity, providing practitioners with flexibility to tailor search results to task-specific requirements. We develop efficient nearest neighbor algorithms with provable guarantees for the welfare-based objectives. Notably, our algorithm can be applied on top of any standard ANN method (i.e., use standard ANN method as a subroutine) to efficiently find neighbors that approximately maximize our welfare-based objectives. Experimental results demonstrate that our approach is practical and substantially improves diversity while maintaining high relevance of the retrieved neighbors. 

---
# Do Images Clarify? A Study on the Effect of Images on Clarifying Questions in Conversational Search 

**Authors**: Clemencia Siro, Zahra Abbasiantaeb, Yifei Yuan, Mohammad Aliannejadi, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2602.08700)  

**Abstract**: Conversational search systems increasingly employ clarifying questions to refine user queries and improve the search experience. Previous studies have demonstrated the usefulness of text-based clarifying questions in enhancing both retrieval performance and user experience. While images have been shown to improve retrieval performance in various contexts, their impact on user performance when incorporated into clarifying questions remains largely unexplored. We conduct a user study with 73 participants to investigate the role of images in conversational search, specifically examining their effects on two search-related tasks: (i) answering clarifying questions and (ii) query reformulation. We compare the effect of multimodal and text-only clarifying questions in both tasks within a conversational search context from various perspectives. Our findings reveal that while participants showed a strong preference for multimodal questions when answering clarifying questions, preferences were more balanced in the query reformulation task. The impact of images varied with both task type and user expertise. In answering clarifying questions, images helped maintain engagement across different expertise levels, while in query reformulation they led to more precise queries and improved retrieval performance. Interestingly, for clarifying question answering, text-only setups demonstrated better user performance as they provided more comprehensive textual information in the absence of images. These results provide valuable insights for designing effective multimodal conversational search systems, highlighting that the benefits of visual augmentation are task-dependent and should be strategically implemented based on the specific search context and user characteristics. 

---
# Retrieval Pivot Attacks in Hybrid RAG: Measuring and Mitigating Amplified Leakage from Vector Seeds to Graph Expansion 

**Authors**: Scott Thornton  

**Link**: [PDF](https://arxiv.org/pdf/2602.08668)  

**Abstract**: Hybrid Retrieval-Augmented Generation (RAG) pipelines combine vector similarity search with knowledge graph expansion for multi-hop reasoning. We show that this composition introduces a distinct security failure mode: a vector-retrieved "seed" chunk can pivot via entity links into sensitive graph neighborhoods, causing cross-tenant data leakage that does not occur in vector-only retrieval. We formalize this risk as Retrieval Pivot Risk (RPR) and introduce companion metrics Leakage@k, Amplification Factor, and Pivot Depth (PD) to quantify leakage magnitude and traversal structure.
We present seven Retrieval Pivot Attacks that exploit the vector-to-graph boundary and show that adversarial injection is not required: naturally shared entities create cross-tenant pivot paths organically. Across a synthetic multi-tenant enterprise corpus and the Enron email corpus, the undefended hybrid pipeline exhibits high pivot risk (RPR up to 0.95) with multiple unauthorized items returned per query. Leakage consistently appears at PD=2, which we attribute to the bipartite chunk-entity topology and formalize as a proposition.
We then show that enforcing authorization at a single location, the graph expansion boundary, eliminates measured leakage (RPR near 0) across both corpora, all attack variants, and label forgery rates up to 10 percent, with minimal overhead. Our results indicate the root cause is boundary enforcement, not inherently complex defenses: two individually secure retrieval components can compose into an insecure system unless authorization is re-checked at the transition point. 

---
# Towards Reliable Social A/B Testing: Spillover-Contained Clustering with Robust Post-Experiment Analysis 

**Authors**: Xu Min, Zhaoxu Yang, Kaixuan Tan, Juan Yan, Xunbin Xiong, Zihao Zhu, Kaiyu Zhu, Fenglin Cui, Yang Yang, Sihua Yang, Jianhui Bu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08569)  

**Abstract**: A/B testing is the foundation of decision-making in online platforms, yet social products often suffer from network interference: user interactions cause treatment effects to spill over into the control group. Such spillovers bias causal estimates and undermine experimental conclusions. Existing approaches face key limitations: user-level randomization ignores network structure, while cluster-based methods often rely on general-purpose clustering that is not tailored for spillover containment and has difficulty balancing unbiasedness and statistical power at scale. We propose a spillover-contained experimentation framework with two stages. In the pre-experiment stage, we build social interaction graphs and introduce a Balanced Louvain algorithm that produces stable, size-balanced clusters while minimizing cross-cluster edges, enabling reliable cluster-based randomization. In the post-experiment stage, we develop a tailored CUPAC estimator that leverages pre-experiment behavioral covariates to reduce the variance induced by cluster-level assignment, thereby improving statistical power. Together, these components provide both structural spillover containment and robust statistical inference. We validate our approach through large-scale social sharing experiments on Kuaishou, a platform serving hundreds of millions of users. Results show that our method substantially reduces spillover and yields more accurate assessments of social strategies than traditional user-level designs, establishing a reliable and scalable framework for networked A/B testing. 

---
# GISA: A Benchmark for General Information-Seeking Assistant 

**Authors**: Yutao Zhu, Xingshuo Zhang, Maosen Zhang, Jiajie Jin, Liancheng Zhang, Xiaoshuai Song, Kangzhi Zhao, Wencong Zeng, Ruiming Tang, Han Li, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2602.08543)  

**Abstract**: The advancement of large language models (LLMs) has significantly accelerated the development of search agents capable of autonomously gathering information through multi-turn web interactions. Various benchmarks have been proposed to evaluate such agents. However, existing benchmarks often construct queries backward from answers, producing unnatural tasks misaligned with real-world needs. Moreover, these benchmarks tend to focus on either locating specific information or aggregating information from multiple sources, while relying on static answer sets prone to data contamination. To bridge these gaps, we introduce GISA, a benchmark for General Information-Seeking Assistants comprising 373 human-crafted queries that reflect authentic information-seeking scenarios. GISA features four structured answer formats (item, set, list, and table), enabling deterministic evaluation. It integrates both deep reasoning and broad information aggregation within unified tasks, and includes a live subset with periodically updated answers to resist memorization. Notably, GISA provides complete human search trajectories for every query, offering gold-standard references for process-level supervision and imitation learning. Experiments on mainstream LLMs and commercial search products reveal that even the best-performing model achieves only 19.30\% exact match score, with performance notably degrading on tasks requiring complex planning and comprehensive information gathering. These findings highlight substantial room for future improvement. 

---
# SynthAgent: A Multi-Agent LLM Framework for Realistic Patient Simulation -- A Case Study in Obesity with Mental Health Comorbidities 

**Authors**: Arman Aghaee, Sepehr Asgarian, Jouhyun Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2602.08254)  

**Abstract**: Simulating high-fidelity patients offers a powerful avenue for studying complex diseases while addressing the challenges of fragmented, biased, and privacy-restricted real-world data. In this study, we introduce SynthAgent, a novel Multi-Agent System (MAS) framework designed to model obesity patients with comorbid mental disorders, including depression, anxiety, social phobia, and binge eating disorder. SynthAgent integrates clinical and medical evidence from claims data, population surveys, and patient-centered literature to construct personalized virtual patients enriched with personality traits that influence adherence, emotion regulation, and lifestyle behaviors. Through autonomous agent interactions, the system simulates disease progression, treatment response, and life management across diverse psychosocial contexts. Evaluation of more than 100 generated patients demonstrated that GPT-5 and Claude 4.5 Sonnet achieved the highest fidelity as the core engine in the proposed MAS framework, outperforming Gemini 2.5 Pro and DeepSeek-R1. SynthAgent thus provides a scalable and privacy-preserving framework for exploring patient journeys, behavioral dynamics, and decision-making processes in both medical and psychological domains. 

---
# Prune, Don't Rebuild: Efficiently Tuning $α$-Reachable Graphs for Nearest Neighbor Search 

**Authors**: Tian Zhang, Ashwin Padaki, Jiaming Liang, Zack Ives, Erik Waingarten  

**Link**: [PDF](https://arxiv.org/pdf/2602.08097)  

**Abstract**: Vector similarity search is an essential primitive in modern AI and ML applications. Most vector databases adopt graph-based approximate nearest neighbor (ANN) search algorithms, such as DiskANN (Subramanya et al., 2019), which have demonstrated state-of-the-art empirical performance. DiskANN's graph construction is governed by a reachability parameter $\alpha$, which gives a trade-off between construction time, query time, and accuracy. However, adaptively tuning this trade-off typically requires rebuilding the index for different $\alpha$ values, which is prohibitive at scale. In this work, we propose RP-Tuning, an efficient post-hoc routine, based on DiskANN's pruning step, to adjust the $\alpha$ parameter without reconstructing the full index. Within the $\alpha$-reachability framework of prior theoretical works (Indyk and Xu, 2023; Gollapudi et al., 2025), we prove that pruning an initially $\alpha$-reachable graph with RP-Tuning preserves worst-case reachability guarantees in general metrics and improved guarantees in Euclidean metrics. Empirically, we show that RP-Tuning accelerates DiskANN tuning on four public datasets by up to $43\times$ with negligible overhead. 

---
# SRR-Judge: Step-Level Rating and Refinement for Enhancing Search-Integrated Reasoning in Search Agents 

**Authors**: Chen Zhang, Kuicai Dong, Dexun Li, Wenjun Li, Qu Yang, Wei Han, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.07773)  

**Abstract**: Recent deep search agents built on large reasoning models (LRMs) excel at complex question answering by iteratively planning, acting, and gathering evidence, a capability known as search-integrated reasoning. However, mainstream approaches often train this ability using only outcome-based supervision, neglecting the quality of intermediate thoughts and actions. We introduce SRR-Judge, a framework for reliable step-level assessment of reasoning and search actions. Integrated into a modified ReAct-style rate-and-refine workflow, SRR-Judge provides fine-grained guidance for search-integrated reasoning and enables efficient post-training annotation. Using SRR-annotated data, we apply an iterative rejection sampling fine-tuning procedure to enhance the deep search capability of the base agent. Empirically, SRR-Judge delivers more reliable step-level evaluations than much larger models such as DeepSeek-V3.1, with its ratings showing strong correlation with final answer correctness. Moreover, aligning the policy with SRR-Judge annotated trajectories leads to substantial performance gains, yielding over a 10 percent average absolute pass@1 improvement across challenging deep search benchmarks. 

---
# EventCast: Hybrid Demand Forecasting in E-Commerce with LLM-Based Event Knowledge 

**Authors**: Congcong Hu, Yuang Shi, Fan Huang, Yang Xiang, Zhou Ye, Ming Jin, Shiyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07695)  

**Abstract**: Demand forecasting is a cornerstone of e-commerce operations, directly impacting inventory planning and fulfillment scheduling. However, existing forecasting systems often fail during high-impact periods such as flash sales, holiday campaigns, and sudden policy interventions, where demand patterns shift abruptly and unpredictably. In this paper, we introduce EventCast, a modular forecasting framework that integrates future event knowledge into time-series prediction. Unlike prior approaches that ignore future interventions or directly use large language models (LLMs) for numerical forecasting, EventCast leverages LLMs solely for event-driven reasoning. Unstructured business data, which covers campaigns, holiday schedules, and seller incentives, from existing operational databases, is processed by an LLM that converts it into interpretable textual summaries leveraging world knowledge for cultural nuances and novel event combinations. These summaries are fused with historical demand features within a dual-tower architecture, enabling accurate, explainable, and scalable forecasts. Deployed on real-world e-commerce scenarios spanning 4 countries of 160 regions over 10 months, EventCast achieves up to 86.9% and 97.7% improvement on MAE and MSE compared to the variant without event knowledge, and reduces MAE by up to 57.0% and MSE by 83.3% versus the best industrial baseline during event-driven periods. EventCast has deployed into real-world industrial pipelines since March 2025, offering a practical solution for improving operational decision-making in dynamic e-commerce environments. 

---
# Assessing the impact of Open Research Information Infrastructures using NLP driven full-text Scientometrics: A case study of the LXCat open-access platform 

**Authors**: Kalp Pandya, Khushi Shah, Nirmal Shah, Nakshi Shah, Bhaskar Chaudhury  

**Link**: [PDF](https://arxiv.org/pdf/2602.07664)  

**Abstract**: Open research information (ORI) play a central role in shaping how scientific knowledge is produced, disseminated, validated, and reused across the research lifecycle. While the visibility of such ORI infrastructures is often assessed through citation-based metrics, in this study, we present a full-text, natural language processing (NLP) driven scientometric framework to systematically quantify the impact of ORI infrastructures beyond citation counts, using the LXCat platform for low temperature plasma (LTP) research as a representative case study. The modeling of LTPs and interpretation of LTP experiments rely heavily on accurate data, much of which is hosted on LXCat, a community-driven, open-access platform central to the LTP research ecosystem. To investigate the scholarly impact of the LXCat platform over the past decade, we analyzed a curated corpus of full-text research articles citing three foundational LXCat publications. We present a comprehensive pipeline that integrates chemical entity recognition, dataset and solver mention extraction, affiliation based geographic mapping and topic modeling to extract fine-grained patterns of data usage that reflect implicit research priorities, data practices, differential reliance on specific databases, evolving modes of data reuse and coupling within scientific workflows, and thematic evolution. Importantly, our proposed methodology is domain-agnostic and transferable to other ORI contexts, and highlights the utility of NLP in quantifying the role of scientific data infrastructures and offers a data-driven reflection on how open-access platforms like LXCat contribute to shaping research directions. This work presents a scalable scientometric framework that has the potential to support evidence based evaluation of ORI platforms and to inform infrastructure design, governance, sustainability, and policy for future development. 

---
# Echoes in the Loop: Diagnosing Risks in LLM-Powered Recommender Systems under Feedback Loops 

**Authors**: Donguk Park, Dongwon Lee, Yeon-Chang Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.07442)  

**Abstract**: Large language models (LLMs) are increasingly embedded into recommender systems, where they operate across multiple functional roles such as data augmentation, profiling, and decision making. While prior work emphasizes recommendation performance, the systemic risks of LLMs, such as bias and hallucination, and their propagation through feedback loops remain largely unexplored. In this paper, we propose a role-aware, phase-wise diagnostic framework that traces how these risks emerge, manifest in ranking outcomes, and accumulate over repeated recommendation cycles. We formalize a controlled feedback-loop pipeline that simulates long-term interaction dynamics and enables empirical measurement of risks at the LLM-generated content, ranking, and ecosystem levels. Experiments on widely used benchmarks demonstrate that LLM-based components can amplify popularity bias, introduce spurious signals through hallucination, and lead to polarized and self-reinforcing exposure patterns over time. We plan to release our framework as an open-source toolkit to facilitate systematic risk analysis across diverse LLM-powered recommender systems. 

---
# ViHERMES: A Graph-Grounded Multihop Question Answering Benchmark and System for Vietnamese Healthcare Regulations 

**Authors**: Long S. T. Nguyen, Quan M. Bui, Tin T. Ngo, Quynh T. N. Vo, Dung N. H. Le, Tho T. Quan  

**Link**: [PDF](https://arxiv.org/pdf/2602.07361)  

**Abstract**: Question Answering (QA) over regulatory documents is inherently challenging due to the need for multihop reasoning across legally interdependent texts, a requirement that is particularly pronounced in the healthcare domain where regulations are hierarchically structured and frequently revised through amendments and cross-references. Despite recent progress in retrieval-augmented and graph-based QA methods, systematic evaluation in this setting remains limited, especially for low-resource languages such as Vietnamese, due to the lack of benchmark datasets that explicitly support multihop reasoning over healthcare regulations. In this work, we introduce the Vietnamese Healthcare Regulations-Multihop Reasoning Dataset (ViHERMES), a benchmark designed for multihop QA over Vietnamese healthcare regulatory documents. ViHERMES consists of high-quality question-answer pairs that require reasoning across multiple regulations and capture diverse dependency patterns, including amendment tracing, cross-document comparison, and procedural synthesis. To construct the dataset, we propose a controlled multihop QA generation pipeline based on semantic clustering and graph-inspired data mining, followed by large language model-based generation with structured evidence and reasoning annotations. We further present a graph-aware retrieval framework that models formal legal relations at the level of legal units and supports principled context expansion for legally valid and coherent answers. Experimental results demonstrate that ViHERMES provides a challenging benchmark for evaluating multihop regulatory QA systems and that the proposed graph-aware approach consistently outperforms strong retrieval-based baselines. The ViHERMES dataset and system implementation are publicly available at this https URL. 

---
