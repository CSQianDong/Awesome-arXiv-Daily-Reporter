# PCR-CA: Parallel Codebook Representations with Contrastive Alignment for Multiple-Category App Recommendation 

**Authors**: Bin Tan, Wangyao Ge, Yidi Wang, Xin Liu, Jeff Burtoft, Hao Fan, Hui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18166)  

**Abstract**: Modern app store recommender systems struggle with multiple-category apps, as traditional taxonomies fail to capture overlapping semantics, leading to suboptimal personalization. We propose PCR-CA (Parallel Codebook Representations with Contrastive Alignment), an end-to-end framework for improved CTR prediction. PCR-CA first extracts compact multimodal embeddings from app text, then introduces a Parallel Codebook VQ-AE module that learns discrete semantic representations across multiple codebooks in parallel -- unlike hierarchical residual quantization (RQ-VAE). This design enables independent encoding of diverse aspects (e.g., gameplay, art style), better modeling multiple-category semantics. To bridge semantic and collaborative signals, we employ a contrastive alignment loss at both the user and item levels, enhancing representation learning for long-tail items. Additionally, a dual-attention fusion mechanism combines ID-based and semantic features to capture user interests, especially for long-tail apps. Experiments on a large-scale dataset show PCR-CA achieves a +0.76% AUC improvement over strong baselines, with +2.15% AUC gains for long-tail apps. Online A/B testing further validates our approach, showing a +10.52% lift in CTR and a +16.30% improvement in CVR, demonstrating PCR-CA's effectiveness in real-world deployment. The new framework has now been fully deployed on the Microsoft Store. 

---
# Test-Time Scaling Strategies for Generative Retrieval in Multimodal Conversational Recommendations 

**Authors**: Hung-Chun Hsu, Yuan-Ching Kuo, Chao-Han Huck Yang, Szu-Wei Fu, Hanrong Ye, Hongxu Yin, Yu-Chiang Frank Wang, Ming-Feng Tsai, Chuan-Ju Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18132)  

**Abstract**: The rapid evolution of e-commerce has exposed the limitations of traditional product retrieval systems in managing complex, multi-turn user interactions. Recent advances in multimodal generative retrieval -- particularly those leveraging multimodal large language models (MLLMs) as retrievers -- have shown promise. However, most existing methods are tailored to single-turn scenarios and struggle to model the evolving intent and iterative nature of multi-turn dialogues when applied naively. Concurrently, test-time scaling has emerged as a powerful paradigm for improving large language model (LLM) performance through iterative inference-time refinement. Yet, its effectiveness typically relies on two conditions: (1) a well-defined problem space (e.g., mathematical reasoning), and (2) the model's ability to self-correct -- conditions that are rarely met in conversational product search. In this setting, user queries are often ambiguous and evolving, and MLLMs alone have difficulty grounding responses in a fixed product corpus. Motivated by these challenges, we propose a novel framework that introduces test-time scaling into conversational multimodal product retrieval. Our approach builds on a generative retriever, further augmented with a test-time reranking (TTR) mechanism that improves retrieval accuracy and better aligns results with evolving user intent throughout the dialogue. Experiments across multiple benchmarks show consistent improvements, with average gains of 14.5 points in MRR and 10.6 points in nDCG@1. 

---
# HLLM-Creator: Hierarchical LLM-based Personalized Creative Generation 

**Authors**: Junyi Chen, Lu Chi, Siliang Xu, Shiwei Ran, Bingyue Peng, Zehuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18118)  

**Abstract**: AI-generated content technologies are widely used in content creation. However, current AIGC systems rely heavily on creators' inspiration, rarely generating truly user-personalized content. In real-world applications such as online advertising, a single product may have multiple selling points, with different users focusing on different features. This underscores the significant value of personalized, user-centric creative generation. Effective personalized content generation faces two main challenges: (1) accurately modeling user interests and integrating them into the content generation process while adhering to factual constraints, and (2) ensuring high efficiency and scalability to handle the massive user base in industrial scenarios. Additionally, the scarcity of personalized creative data in practice complicates model training, making data construction another key hurdle. We propose HLLM-Creator, a hierarchical LLM framework for efficient user interest modeling and personalized content generation. During inference, a combination of user clustering and a user-ad-matching-prediction based pruning strategy is employed to significantly enhance generation efficiency and reduce computational overhead, making the approach suitable for large-scale deployment. Moreover, we design a data construction pipeline based on chain-of-thought reasoning, which generates high-quality, user-specific creative titles and ensures factual consistency despite limited personalized data. This pipeline serves as a critical foundation for the effectiveness of our model. Extensive experiments on personalized title generation for Douyin Search Ads show the effectiveness of HLLM-Creator. Online A/B test shows a 0.476% increase on Adss, paving the way for more effective and efficient personalized generation in industrial scenarios. Codes for academic dataset are available at this https URL. 

---
# HyST: LLM-Powered Hybrid Retrieval over Semi-Structured Tabular Data 

**Authors**: Jiyoon Myung, Jihyeon Park, Joohyung Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.18048)  

**Abstract**: User queries in real-world recommendation systems often combine structured constraints (e.g., category, attributes) with unstructured preferences (e.g., product descriptions or reviews). We introduce HyST (Hybrid retrieval over Semi-structured Tabular data), a hybrid retrieval framework that combines LLM-powered structured filtering with semantic embedding search to support complex information needs over semi-structured tabular data. HyST extracts attribute-level constraints from natural language using large language models (LLMs) and applies them as metadata filters, while processing the remaining unstructured query components via embedding-based retrieval. Experiments on a semi-structured benchmark show that HyST consistently outperforms tradtional baselines, highlighting the importance of structured filtering in improving retrieval precision, offering a scalable and accurate solution for real-world user queries. 

---
# Retrieval Feedback Memory Enhancement Large Model Retrieval Generation Method 

**Authors**: Leqian Li, Dianxi Shi, Jialu Zhou, Xinyu Wei, Mingyue Yang, Songchang Jin, Shaowu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17862)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities across diverse tasks, yet they face inherent limitations such as constrained parametric knowledge and high retraining costs. Retrieval-Augmented Generation (RAG) augments the generation process by retrieving externally stored knowledge absent from the models internal parameters. However, RAG methods face challenges such as information loss and redundant retrievals during multi-round queries, accompanying the difficulties in precisely characterizing knowledge gaps for complex tasks. To address these problems, we propose Retrieval Feedback and Memory Retrieval Augmented Generation(RFM-RAG), which transforms the stateless retrieval of previous methods into stateful continuous knowledge management by constructing a dynamic evidence pool. Specifically, our method generates refined queries describing the models knowledge gaps using relational triples from questions and evidence from the dynamic evidence pool; Retrieves critical external knowledge to iteratively update this evidence pool; Employs a R-Feedback Model to evaluate evidence completeness until convergence. Compared to traditional RAG methods, our approach enables persistent storage of retrieved passages and effectively distills key information from passages to construct clearly new queries. Experiments on three public QA benchmarks demonstrate that RFM-RAG outperforms previous methods and improves overall system accuracy. 

---
# LexSemBridge: Fine-Grained Dense Representation Enhancement through Token-Aware Embedding Augmentation 

**Authors**: Shaoxiong Zhan, Hai Lin, Hongming Tan, Xiaodong Cai, Hai-Tao Zheng, Xin Su, Zifei Shan, Ruitong Liu, Hong-Gee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.17858)  

**Abstract**: As queries in retrieval-augmented generation (RAG) pipelines powered by large language models (LLMs) become increasingly complex and diverse, dense retrieval models have demonstrated strong performance in semantic matching. Nevertheless, they often struggle with fine-grained retrieval tasks, where precise keyword alignment and span-level localization are required, even in cases with high lexical overlap that would intuitively suggest easier retrieval. To systematically evaluate this limitation, we introduce two targeted tasks, keyword retrieval and part-of-passage retrieval, designed to simulate practical fine-grained scenarios. Motivated by these observations, we propose LexSemBridge, a unified framework that enhances dense query representations through fine-grained, input-aware vector modulation. LexSemBridge constructs latent enhancement vectors from input tokens using three paradigms: Statistical (SLR), Learned (LLR), and Contextual (CLR), and integrates them with dense embeddings via element-wise interaction. Theoretically, we show that this modulation preserves the semantic direction while selectively amplifying discriminative dimensions. LexSemBridge operates as a plug-in without modifying the backbone encoder and naturally extends to both text and vision modalities. Extensive experiments across semantic and fine-grained retrieval tasks validate the effectiveness and generality of our approach. All code and models are publicly available at this https URL 

---
# Research on Evaluation Methods for Patent Novelty Search Systems and Empirical Analysis 

**Authors**: Shu Zhang, LiSha Zhang, Kai Duan, XinKai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.17782)  

**Abstract**: Patent novelty search systems are critical to IP protection and innovation assessment; their retrieval accuracy directly impacts patent quality. We propose a comprehensive evaluation methodology that builds high-quality, reproducible datasets from examiner citations and X-type citations extracted from technically consistent family patents, and evaluates systems using invention descriptions as inputs. Using Top-k Detection Rate and Recall as core metrics, we further conduct multi-dimensional analyses by language, technical field (IPC), and filing jurisdiction. Experiments show the method effectively exposes performance differences across scenarios and offers actionable evidence for system improvement. The framework is scalable and practical, providing a useful reference for development and optimization of patent novelty search systems 

---
# DiffusionGS: Generative Search with Query Conditioned Diffusion in Kuaishou 

**Authors**: Qinyao Li, Xiaoyang Zheng, Qihang Zhao, Ke Xu, Zhongbo Sun, Chao Wang, Chenyi Lei, Han Li, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2508.17754)  

**Abstract**: Personalized search ranking systems are critical for driving engagement and revenue in modern e-commerce and short-video platforms. While existing methods excel at estimating users' broad interests based on the filtered historical behaviors, they typically under-exploit explicit alignment between a user's real-time intent (represented by the user query) and their past actions. In this paper, we propose DiffusionGS, a novel and scalable approach powered by generative models. Our key insight is that user queries can serve as explicit intent anchors to facilitate the extraction of users' immediate interests from long-term, noisy historical behaviors. Specifically, we formulate interest extraction as a conditional denoising task, where the user's query guides a conditional diffusion process to produce a robust, user intent-aware representation from their behavioral sequence. We propose the User-aware Denoising Layer (UDL) to incorporate user-specific profiles into the optimization of attention distribution on the user's past actions. By reframing queries as intent priors and leveraging diffusion-based denoising, our method provides a powerful mechanism for capturing dynamic user interest shifts. Extensive offline and online experiments demonstrate the superiority of DiffusionGS over state-of-the-art methods. 

---
# How Do LLM-Generated Texts Impact Term-Based Retrieval Models? 

**Authors**: Wei Huang, Keping Bi, Yinqiong Cai, Wei Chen, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.17715)  

**Abstract**: As more content generated by large language models (LLMs) floods into the Internet, information retrieval (IR) systems now face the challenge of distinguishing and handling a blend of human-authored and machine-generated texts. Recent studies suggest that neural retrievers may exhibit a preferential inclination toward LLM-generated content, while classic term-based retrievers like BM25 tend to favor human-written documents. This paper investigates the influence of LLM-generated content on term-based retrieval models, which are valued for their efficiency and robust generalization across domains. Our linguistic analysis reveals that LLM-generated texts exhibit smoother high-frequency and steeper low-frequency Zipf slopes, higher term specificity, and greater document-level diversity. These traits are aligned with LLMs being trained to optimize reader experience through diverse and precise expressions. Our study further explores whether term-based retrieval models demonstrate source bias, concluding that these models prioritize documents whose term distributions closely correspond to those of the queries, rather than displaying an inherent source bias. This work provides a foundation for understanding and addressing potential biases in term-based IR systems managing mixed-source content. 

---
# Semantic Search for Information Retrieval 

**Authors**: Kayla Farivar  

**Link**: [PDF](https://arxiv.org/pdf/2508.17694)  

**Abstract**: Information retrieval systems have progressed notably from lexical techniques such as BM25 and TF-IDF to modern semantic retrievers. This survey provides a brief overview of the BM25 baseline, then discusses the architecture of modern state-of-the-art semantic retrievers. Advancing from BERT, we introduce dense bi-encoders (DPR), late-interaction models (ColBERT), and neural sparse retrieval (SPLADE). Finally, we examine MonoT5, a cross-encoder model. We conclude with common evaluation tactics, pressing challenges, and propositions for future directions. 

---
# Demographically-Inspired Query Variants Using an LLM 

**Authors**: Marwah Alaofi, Nicola Ferro, Paul Thomas, Falk Scholer, Mark Sanderson  

**Link**: [PDF](https://arxiv.org/pdf/2508.17644)  

**Abstract**: This study proposes a method to diversify queries in existing test collections to reflect some of the diversity of search engine users, aligning with an earlier vision of an 'ideal' test collection. A Large Language Model (LLM) is used to create query variants: alternative queries that have the same meaning as the original. These variants represent user profiles characterised by different properties, such as language and domain proficiency, which are known in the IR literature to influence query formulation.
The LLM's ability to generate query variants that align with user profiles is empirically validated, and the variants' utility is further explored for IR system evaluation. Results demonstrate that the variants impact how systems are ranked and show that user profiles experience significantly different levels of system effectiveness. This method enables an alternative perspective on system evaluation where we can observe both the impact of user profiles on system rankings and how system performance varies across users. 

---
# Preference Trajectory Modeling via Flow Matching for Sequential Recommendation 

**Authors**: Li Li, Mingyue Cheng, Yuyang Ye, Zhiding Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.17618)  

**Abstract**: Sequential recommendation predicts each user's next item based on their historical interaction sequence. Recently, diffusion models have attracted significant attention in this area due to their strong ability to model user interest distributions. They typically generate target items by denoising Gaussian noise conditioned on historical interactions. However, these models face two critical limitations. First, they exhibit high sensitivity to the condition, making it difficult to recover target items from pure Gaussian noise. Second, the inference process is computationally expensive, limiting practical deployment. To address these issues, we propose FlowRec, a simple yet effective sequential recommendation framework which leverages flow matching to explicitly model user preference trajectories from current states to future interests. Flow matching is an emerging generative paradigm, which offers greater flexibility in initial distributions and enables more efficient sampling. Based on this, we construct a personalized behavior-based prior distribution to replace Gaussian noise and learn a vector field to model user preference trajectories. To better align flow matching with the recommendation objective, we further design a single-step alignment loss incorporating both positive and negative samples, improving sampling efficiency and generation quality. Extensive experiments on four benchmark datasets verify the superiority of FlowRec over the state-of-the-art baselines. 

---
# A Universal Framework for Offline Serendipity Evaluation in Recommender Systems via Large Language Models 

**Authors**: Yu Tokutake, Kazushi Okamoto, Kei Harada, Atsushi Shibata, Koki Karube  

**Link**: [PDF](https://arxiv.org/pdf/2508.17571)  

**Abstract**: Serendipity in recommender systems (RSs) has attracted increasing attention as a concept that enhances user satisfaction by presenting unexpected and useful items. However, evaluating serendipitous performance remains challenging because its ground truth is generally unobservable. The existing offline metrics often depend on ambiguous definitions or are tailored to specific datasets and RSs, thereby limiting their generalizability. To address this issue, we propose a universally applicable evaluation framework that leverages large language models (LLMs) known for their extensive knowledge and reasoning capabilities, as evaluators. First, to improve the evaluation performance of the proposed framework, we assessed the serendipity prediction accuracy of LLMs using four different prompt strategies on a dataset containing user-annotated serendipitous ground truth and found that the chain-of-thought prompt achieved the highest accuracy. Next, we re-evaluated the serendipitous performance of both serendipity-oriented and general RSs using the proposed framework on three commonly used real-world datasets, without the ground truth. The results indicated that there was no serendipity-oriented RS that consistently outperformed across all datasets, and even a general RS sometimes achieved higher performance than the serendipity-oriented RS. 

---
# Opening the Black Box: Interpretable Remedies for Popularity Bias in Recommender Systems 

**Authors**: Parviz Ahmadov, Masoud Mansoury  

**Link**: [PDF](https://arxiv.org/pdf/2508.17297)  

**Abstract**: Popularity bias is a well-known challenge in recommender systems, where a small number of popular items receive disproportionate attention, while the majority of less popular items are largely overlooked. This imbalance often results in reduced recommendation quality and unfair exposure of items. Although existing mitigation techniques address this bias to some extent, they typically lack transparency in how they operate. In this paper, we propose a post-hoc method using a Sparse Autoencoder (SAE) to interpret and mitigate popularity bias in deep recommendation models. The SAE is trained to replicate a pre-trained model's behavior while enabling neuron-level interpretability. By introducing synthetic users with clear preferences for either popular or unpopular items, we identify neurons encoding popularity signals based on their activation patterns. We then adjust the activations of the most biased neurons to steer recommendations toward fairer exposure. Experiments on two public datasets using a sequential recommendation model show that our method significantly improves fairness with minimal impact on accuracy. Moreover, it offers interpretability and fine-grained control over the fairness-accuracy trade-off. 

---
# VQL: An End-to-End Context-Aware Vector Quantization Attention for Ultra-Long User Behavior Modeling 

**Authors**: Kaiyuan Li, Yongxiang Tang, Yanhua Cheng, Yong Bai, Yanxiang Zeng, Chao Wang, Xialong Liu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17125)  

**Abstract**: In large-scale recommender systems, ultra-long user behavior sequences encode rich signals of evolving interests. Extending sequence length generally improves accuracy, but directly modeling such sequences in production is infeasible due to latency and memory constraints. Existing solutions fall into two categories: (1) top-k retrieval, which truncates the sequence and may discard most attention mass when L >> k; and (2) encoder-based compression, which preserves coverage but often over-compresses and fails to incorporate key context such as temporal gaps or target-aware signals. Neither class achieves a good balance of low-loss compression, context awareness, and efficiency.
We propose VQL, a context-aware Vector Quantization Attention framework for ultra-long behavior modeling, with three innovations. (1) Key-only quantization: only attention keys are quantized, while values remain intact; we prove that softmax normalization yields an error bound independent of sequence length, and a codebook loss directly supervises quantization quality. This also enables L-free inference via offline caches. (2) Multi-scale quantization: attention heads are partitioned into groups, each with its own small codebook, which reduces quantization error while keeping cache size fixed. (3) Efficient context injection: static features (e.g., item category, modality) are directly integrated, and relative position is modeled via a separable temporal kernel. All context is injected without enlarging the codebook, so cached representations remain query-independent.
Experiments on three large-scale datasets (KuaiRand-1K, KuaiRec, TMALL) show that VQL consistently outperforms strong baselines, achieving higher accuracy while reducing inference latency, establishing a new state of the art in balancing accuracy and efficiency for ultra-long sequence recommendation. 

---
# Zero-shot Multimodal Document Retrieval via Cross-modal Question Generation 

**Authors**: Yejin Choi, Jaewoo Park, Janghan Yoon, Saejin Kim, Jaehyun Jeon, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17079)  

**Abstract**: Rapid advances in Multimodal Large Language Models (MLLMs) have expanded information retrieval beyond purely textual inputs, enabling retrieval from complex real world documents that combine text and visuals. However, most documents are private either owned by individuals or confined within corporate silos and current retrievers struggle when faced with unseen domains or languages. To address this gap, we introduce PREMIR, a simple yet effective framework that leverages the broad knowledge of an MLLM to generate cross modal pre questions (preQs) before retrieval. Unlike earlier multimodal retrievers that compare embeddings in a single vector space, PREMIR leverages preQs from multiple complementary modalities to expand the scope of matching to the token level. Experiments show that PREMIR achieves state of the art performance on out of distribution benchmarks, including closed domain and multilingual settings, outperforming strong baselines across all retrieval metrics. We confirm the contribution of each component through in depth ablation studies, and qualitative analyses of the generated preQs further highlight the model's robustness in real world settings. 

---
# Towards a Real-World Aligned Benchmark for Unlearning in Recommender Systems 

**Authors**: Pierre Lubitzsch, Olga Ovcharenko, Hao Chen, Maarten de Rijke, Sebastian Schelter  

**Link**: [PDF](https://arxiv.org/pdf/2508.17076)  

**Abstract**: Modern recommender systems heavily leverage user interaction data to deliver personalized experiences. However, relying on personal data presents challenges in adhering to privacy regulations, such as the GDPR's "right to be forgotten". Machine unlearning (MU) aims to address these challenges by enabling the efficient removal of specific training data from models post-training, without compromising model utility or leaving residual information. However, current benchmarks for unlearning in recommender systems -- most notably CURE4Rec -- fail to reflect real-world operational demands. They focus narrowly on collaborative filtering, overlook tasks like session-based and next-basket recommendation, simulate unrealistically large unlearning requests, and ignore critical efficiency constraints. In this paper, we propose a set of design desiderata and research questions to guide the development of a more realistic benchmark for unlearning in recommender systems, with the goal of gathering feedback from the research community. Our benchmark proposal spans multiple recommendation tasks, includes domain-specific unlearning scenarios, and several unlearning algorithms -- including ones adapted from a recent NeurIPS unlearning competition. Furthermore, we argue for an unlearning setup that reflects the sequential, time-sensitive nature of real-world deletion requests. We also present a preliminary experiment in a next-basket recommendation setting based on our proposed desiderata and find that unlearning also works for sequential recommendation models, exposed to many small unlearning requests. In this case, we observe that a modification of a custom-designed unlearning algorithm for recommender systems outperforms general unlearning algorithms significantly, and that unlearning can be executed with a latency of only several seconds. 

---
# Bootstrapping Conditional Retrieval for User-to-Item Recommendations 

**Authors**: Hongtao Lin, Haoyu Chen, Jaewon Jang, Jiajing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16793)  

**Abstract**: User-to-item retrieval has been an active research area in recommendation system, and two tower models are widely adopted due to model simplicity and serving efficiency. In this work, we focus on a variant called \textit{conditional retrieval}, where we expect retrieved items to be relevant to a condition (e.g. topic). We propose a method that uses the same training data as standard two tower models but incorporates item-side information as conditions in query. This allows us to bootstrap new conditional retrieval use cases and encourages feature interactions between user and condition. Experiments show that our method can retrieve highly relevant items and outperforms standard two tower models with filters on engagement metrics. The proposed model is deployed to power a topic-based notification feed at Pinterest and led to +0.26\% weekly active users. 

---
# ST-Raptor: LLM-Powered Semi-Structured Table Question Answering 

**Authors**: Zirui Tang, Boyu Niu, Xuanhe Zhou, Boxiu Li, Wei Zhou, Jiannan Wang, Guoliang Li, Xinyi Zhang, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18190)  

**Abstract**: Semi-structured tables, widely used in real-world applications (e.g., financial reports, medical records, transactional orders), often involve flexible and complex layouts (e.g., hierarchical headers and merged cells). These tables generally rely on human analysts to interpret table layouts and answer relevant natural language questions, which is costly and inefficient. To automate the procedure, existing methods face significant challenges. First, methods like NL2SQL require converting semi-structured tables into structured ones, which often causes substantial information loss. Second, methods like NL2Code and multi-modal LLM QA struggle to understand the complex layouts of semi-structured tables and cannot accurately answer corresponding questions. To this end, we propose ST-Raptor, a tree-based framework for semi-structured table question answering using large language models. First, we introduce the Hierarchical Orthogonal Tree (HO-Tree), a structural model that captures complex semi-structured table layouts, along with an effective algorithm for constructing the tree. Second, we define a set of basic tree operations to guide LLMs in executing common QA tasks. Given a user question, ST-Raptor decomposes it into simpler sub-questions, generates corresponding tree operation pipelines, and conducts operation-table alignment for accurate pipeline execution. Third, we incorporate a two-stage verification mechanism: forward validation checks the correctness of execution steps, while backward validation evaluates answer reliability by reconstructing queries from predicted answers. To benchmark the performance, we present SSTQA, a dataset of 764 questions over 102 real-world semi-structured tables. Experiments show that ST-Raptor outperforms nine baselines by up to 20% in answer accuracy. The code is available at this https URL. 

---
# Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation 

**Authors**: Tianjun Wei, Huizhong Guo, Yingpeng Du, Zhu Sun, Chen Huang, Dongxia Wang, Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18142)  

**Abstract**: User simulation is increasingly vital to develop and evaluate recommender systems (RSs). While Large Language Models (LLMs) offer promising avenues to simulate user behavior, they often struggle with the absence of specific domain alignment required for RSs and the efficiency demands of large-scale simulation. A vast yet underutilized resource for enhancing this alignment is the extensive user feedback inherent in RSs. However, directly leveraging such feedback presents two significant challenges. First, user feedback in RSs is often ambiguous and noisy, which negatively impacts effective preference alignment. Second, the massive volume of feedback largely hinders the efficiency of preference alignment, necessitating an efficient filtering mechanism to identify more informative samples. To overcome these hurdles, we introduce a novel data construction framework that leverages user feedback in RSs with advanced LLM capabilities to generate high-quality simulation data. Our framework unfolds in two key phases: (1) employing LLMs to generate cognitive decision-making processes on constructed simulation samples, reducing ambiguity in raw user feedback; (2) data distillation based on uncertainty estimation and behavior sampling to filter challenging yet denoised simulation samples. Accordingly, we fine-tune lightweight LLMs, as user simulators, using such high-quality dataset with corresponding decision-making processes. Extensive experiments verify that our framework significantly boosts the alignment with human preferences and in-domain reasoning capabilities of fine-tuned LLMs, and provides more insightful and interpretable signals when interacting with RSs. We believe our work will advance the RS community and offer valuable insights for broader human-centric AI research. 

---
# Heterogeneous co-occurrence embedding for visual information exploration 

**Authors**: Takuro Ishida, Tetsuo Furukawa  

**Link**: [PDF](https://arxiv.org/pdf/2508.17663)  

**Abstract**: This paper proposes an embedding method for co-occurrence data aimed at visual information exploration. We consider cases where co-occurrence probabilities are measured between pairs of elements from heterogeneous domains. The proposed method maps these heterogeneous elements into corresponding two-dimensional latent spaces, enabling visualization of asymmetric relationships between the domains. The key idea is to embed the elements in a way that maximizes their mutual information, thereby preserving the original dependency structure as much as possible. This approach can be naturally extended to cases involving three or more domains, using a generalization of mutual information known as total correlation. For inter-domain analysis, we also propose a visualization method that assigns colors to the latent spaces based on conditional probabilities, allowing users to explore asymmetric relationships interactively. We demonstrate the utility of the method through applications to an adjective-noun dataset, the NeurIPS dataset, and a subject-verb-object dataset, showcasing both intra- and inter-domain analysis. 

---
# SurveyGen: Quality-Aware Scientific Survey Generation with Large Language Models 

**Authors**: Tong Bao, Mir Tafseer Nayeem, Davood Rafiei, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17647)  

**Abstract**: Automatic survey generation has emerged as a key task in scientific document processing. While large language models (LLMs) have shown promise in generating survey texts, the lack of standardized evaluation datasets critically hampers rigorous assessment of their performance against human-written surveys. In this work, we present SurveyGen, a large-scale dataset comprising over 4,200 human-written surveys across diverse scientific domains, along with 242,143 cited references and extensive quality-related metadata for both the surveys and the cited papers. Leveraging this resource, we build QUAL-SG, a novel quality-aware framework for survey generation that enhances the standard Retrieval-Augmented Generation (RAG) pipeline by incorporating quality-aware indicators into literature retrieval to assess and select higher-quality source papers. Using this dataset and framework, we systematically evaluate state-of-the-art LLMs under varying levels of human involvement - from fully automatic generation to human-guided writing. Experimental results and human evaluations show that while semi-automatic pipelines can achieve partially competitive outcomes, fully automatic survey generation still suffers from low citation quality and limited critical analysis. 

---
# DS@GT at CheckThat! 2025: A Simple Retrieval-First, LLM-Backed Framework for Claim Normalization 

**Authors**: Aleksandar Pramov, Jiangqin Ma, Bina Patel  

**Link**: [PDF](https://arxiv.org/pdf/2508.17402)  

**Abstract**: Claim normalization is an integral part of any automatic fact-check verification system. It parses the typically noisy claim data, such as social media posts into normalized claims, which are then fed into downstream veracity classification tasks. The CheckThat! 2025 Task 2 focuses specifically on claim normalization and spans 20 languages under monolingual and zero-shot conditions. Our proposed solution consists of a lightweight \emph{retrieval-first, LLM-backed} pipeline, in which we either dynamically prompt a GPT-4o-mini with in-context examples, or retrieve the closest normalization from the train dataset directly. On the official test set, the system ranks near the top for most monolingual tracks, achieving first place in 7 out of of the 13 languages. In contrast, the system underperforms in the zero-shot setting, highlighting the limitation of the proposed solution. 

---
# Retrieval Capabilities of Large Language Models Scale with Pretraining FLOPs 

**Authors**: Jacob Portes, Connor Jennings, Erica Ji Yuen, Sasha Doubov, Michael Carbin  

**Link**: [PDF](https://arxiv.org/pdf/2508.17400)  

**Abstract**: How does retrieval performance scale with pretraining FLOPs? We benchmark retrieval performance across LLM model sizes from 125 million parameters to 7 billion parameters pretrained on datasets ranging from 1 billion tokens to more than 2 trillion tokens. We find that retrieval performance on zero-shot BEIR tasks predictably scales with LLM size, training duration, and estimated FLOPs. We also show that In-Context Learning scores are strongly correlated with retrieval scores across retrieval tasks. Finally, we highlight the implications this has for the development of LLM-based retrievers. 

---
# Capturing Legal Reasoning Paths from Facts to Law in Court Judgments using Knowledge Graphs 

**Authors**: Ryoma Kondo, Riona Matsuoka, Takahiro Yoshida, Kazuyuki Yamasawa, Ryohei Hisano  

**Link**: [PDF](https://arxiv.org/pdf/2508.17340)  

**Abstract**: Court judgments reveal how legal rules have been interpreted and applied to facts, providing a foundation for understanding structured legal reasoning. However, existing automated approaches for capturing legal reasoning, including large language models, often fail to identify the relevant legal context, do not accurately trace how facts relate to legal norms, and may misrepresent the layered structure of judicial reasoning. These limitations hinder the ability to capture how courts apply the law to facts in practice. In this paper, we address these challenges by constructing a legal knowledge graph from 648 Japanese administrative court decisions. Our method extracts components of legal reasoning using prompt-based large language models, normalizes references to legal provisions, and links facts, norms, and legal applications through an ontology of legal inference. The resulting graph captures the full structure of legal reasoning as it appears in real court decisions, making implicit reasoning explicit and machine-readable. We evaluate our system using expert annotated data, and find that it achieves more accurate retrieval of relevant legal provisions from facts than large language model baselines and retrieval-augmented methods. 

---
# Are You Sure You're Positive? Consolidating Chain-of-Thought Agents with Uncertainty Quantification for Aspect-Category Sentiment Analysis 

**Authors**: Filippos Ventirozos, Peter Appleby, Matthew Shardlow  

**Link**: [PDF](https://arxiv.org/pdf/2508.17258)  

**Abstract**: Aspect-category sentiment analysis provides granular insights by identifying specific themes within product reviews that are associated with particular opinions. Supervised learning approaches dominate the field. However, data is scarce and expensive to annotate for new domains. We argue that leveraging large language models in a zero-shot setting is beneficial where the time and resources required for dataset annotation are limited. Furthermore, annotation bias may lead to strong results using supervised methods but transfer poorly to new domains in contexts that lack annotations and demand reproducibility. In our work, we propose novel techniques that combine multiple chain-of-thought agents by leveraging large language models' token-level uncertainty scores. We experiment with the 3B and 70B+ parameter size variants of Llama and Qwen models, demonstrating how these approaches can fulfil practical needs and opening a discussion on how to gauge accuracy in label-scarce conditions. 

---
# Routing Distilled Knowledge via Mixture of LoRA Experts for Large Language Model based Bundle Generation 

**Authors**: Kaidong Feng, Zhu Sun, Hui Fang, Jie Yang, Wenyuan Liu, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2508.17250)  

**Abstract**: Large Language Models (LLMs) have shown potential in automatic bundle generation but suffer from prohibitive computational costs. Although knowledge distillation offers a pathway to more efficient student models, our preliminary study reveals that naively integrating diverse types of distilled knowledge from teacher LLMs into student LLMs leads to knowledge conflict, negatively impacting the performance of bundle generation. To address this, we propose RouteDK, a framework for routing distilled knowledge through a mixture of LoRA expert architecture. Specifically, we first distill knowledge from the teacher LLM for bundle generation in two complementary types: high-level knowledge (generalizable rules) and fine-grained knowledge (session-specific reasoning). We then train knowledge-specific LoRA experts for each type of knowledge together with a base LoRA expert. For effective integration, we propose a dynamic fusion module, featuring an input-aware router, where the router balances expert contributions by dynamically determining optimal weights based on input, thereby effectively mitigating knowledge conflicts. To further improve inference reliability, we design an inference-time enhancement module to reduce variance and mitigate suboptimal reasoning. Experiments on three public datasets show that our RouteDK achieves accuracy comparable to or even better than the teacher LLM, while maintaining strong computational efficiency. In addition, it outperforms state-of-the-art approaches for bundle generation. 

---
# Exposing Privacy Risks in Graph Retrieval-Augmented Generation 

**Authors**: Jiale Liu, Jiahao Zhang, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17222)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a powerful technique for enhancing Large Language Models (LLMs) with external, up-to-date knowledge. Graph RAG has emerged as an advanced paradigm that leverages graph-based knowledge structures to provide more coherent and contextually rich answers. However, the move from plain document retrieval to structured graph traversal introduces new, under-explored privacy risks. This paper investigates the data extraction vulnerabilities of the Graph RAG systems. We design and execute tailored data extraction attacks to probe their susceptibility to leaking both raw text and structured data, such as entities and their relationships. Our findings reveal a critical trade-off: while Graph RAG systems may reduce raw text leakage, they are significantly more vulnerable to the extraction of structured entity and relationship information. We also explore potential defense mechanisms to mitigate these novel attack surfaces. This work provides a foundational analysis of the unique privacy challenges in Graph RAG and offers insights for building more secure systems. 

---
# The Power of Framing: How News Headlines Guide Search Behavior 

**Authors**: Amrit Poudel, Maria Milkowski, Tim Weninger  

**Link**: [PDF](https://arxiv.org/pdf/2508.17131)  

**Abstract**: Search engines play a central role in how people gather information, but subtle cues like headline framing may influence not only what users believe but also how they search. While framing effects on judgment are well documented, their impact on subsequent search behavior is less understood. We conducted a controlled experiment where participants issued queries and selected from headlines filtered by specific linguistic frames. Headline framing significantly shaped follow-up queries: conflict and strategy frames disrupted alignment with prior selections, while episodic frames led to more concrete queries than thematic ones. We also observed modest short-term frame persistence that declined over time. These results suggest that even brief exposure to framing can meaningfully alter the direction of users information-seeking behavior. 

---
# DeAR: Dual-Stage Document Reranking with Reasoning Agents via LLM Distillation 

**Authors**: Abdelrahman Abdallah, Jamshid Mozafari, Bhawna Piryani, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2508.16998)  

**Abstract**: Large Language Models (LLMs) have transformed listwise document reranking by enabling global reasoning over candidate sets, yet single models often struggle to balance fine-grained relevance scoring with holistic cross-document analysis. We propose \textbf{De}ep\textbf{A}gent\textbf{R}ank (\textbf{\DeAR}), an open-source framework that decouples these tasks through a dual-stage approach, achieving superior accuracy and interpretability. In \emph{Stage 1}, we distill token-level relevance signals from a frozen 13B LLaMA teacher into a compact \{3, 8\}B student model using a hybrid of cross-entropy, RankNet, and KL divergence losses, ensuring robust pointwise scoring. In \emph{Stage 2}, we attach a second LoRA adapter and fine-tune on 20K GPT-4o-generated chain-of-thought permutations, enabling listwise reasoning with natural-language justifications. Evaluated on TREC-DL19/20, eight BEIR datasets, and NovelEval-2306, \DeAR surpasses open-source baselines by +5.1 nDCG@5 on DL20 and achieves 90.97 nDCG@10 on NovelEval, outperforming GPT-4 by +3.09. Without fine-tuning on Wikipedia, DeAR also excels in open-domain QA, achieving 54.29 Top-1 accuracy on Natural Questions, surpassing baselines like MonoT5, UPR, and RankGPT. Ablations confirm that dual-loss distillation ensures stable calibration, making \DeAR a highly effective and interpretable solution for modern reranking systems.\footnote{Dataset and code available at this https URL.}. 

---
# THEME : Enhancing Thematic Investing with Semantic Stock Representations and Temporal Dynamics 

**Authors**: Hoyoung Lee, Wonbin Ahn, Suhwan Park, Jaehoon Lee, Minjae Kim, Sungdong Yoo, Taeyoon Lim, Woohyung Lim, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.16936)  

**Abstract**: Thematic investing aims to construct portfolios aligned with structural trends, yet selecting relevant stocks remains challenging due to overlapping sector boundaries and evolving market dynamics. To address this challenge, we construct the Thematic Representation Set (TRS), an extended dataset that begins with real-world thematic ETFs and expands upon them by incorporating industry classifications and financial news to overcome their coverage limitations. The final dataset contains both the explicit mapping of themes to their constituent stocks and the rich textual profiles for each. Building on this dataset, we introduce \textsc{THEME}, a hierarchical contrastive learning framework. By representing the textual profiles of themes and stocks as embeddings, \textsc{THEME} first leverages their hierarchical relationship to achieve semantic alignment. Subsequently, it refines these semantic embeddings through a temporal refinement stage that incorporates individual stock returns. The final stock representations are designed for effective retrieval of thematically aligned assets with strong return potential. Empirical results show that \textsc{THEME} outperforms strong baselines across multiple retrieval metrics and significantly improves performance in portfolio construction. By jointly modeling thematic relationships from text and market dynamics from returns, \textsc{THEME} provides a scalable and adaptive solution for navigating complex investment themes. 

---
# How Good are LLM-based Rerankers? An Empirical Analysis of State-of-the-Art Reranking Models 

**Authors**: Abdelrahman Abdallah, Bhawna Piryani, Jamshid Mozafari, Mohammed Ali, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2508.16757)  

**Abstract**: In this work, we present a systematic and comprehensive empirical evaluation of state-of-the-art reranking methods, encompassing large language model (LLM)-based, lightweight contextual, and zero-shot approaches, with respect to their performance in information retrieval tasks. We evaluate in total 22 methods, including 40 variants (depending on used LLM) across several established benchmarks, including TREC DL19, DL20, and BEIR, as well as a novel dataset designed to test queries unseen by pretrained models. Our primary goal is to determine, through controlled and fair comparisons, whether a performance disparity exists between LLM-based rerankers and their lightweight counterparts, particularly on novel queries, and to elucidate the underlying causes of any observed differences. To disentangle confounding factors, we analyze the effects of training data overlap, model architecture, and computational efficiency on reranking performance. Our findings indicate that while LLM-based rerankers demonstrate superior performance on familiar queries, their generalization ability to novel queries varies, with lightweight models offering comparable efficiency. We further identify that the novelty of queries significantly impacts reranking effectiveness, highlighting limitations in existing approaches. this https URL 

---
# Sparse and Dense Retrievers Learn Better Together: Joint Sparse-Dense Optimization for Text-Image Retrieval 

**Authors**: Jonghyun Song, Youngjune Lee, Gyu-Hwung Cho, Ilhyeon Song, Saehun Kim, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2508.16707)  

**Abstract**: Vision-Language Pretrained (VLP) models have achieved impressive performance on multimodal tasks, including text-image retrieval, based on dense representations. Meanwhile, Learned Sparse Retrieval (LSR) has gained traction in text-only settings due to its interpretability and efficiency with fast term-based lookup via inverted indexes. Inspired by these advantages, recent work has extended LSR to the multimodal domain. However, these methods often rely on computationally expensive contrastive pre-training, or distillation from a frozen dense model, which limits the potential for mutual enhancement. To address these limitations, we propose a simple yet effective framework that enables bi-directional learning between dense and sparse representations through Self-Knowledge Distillation. This bi-directional learning is achieved using an integrated similarity score-a weighted sum of dense and sparse similarities-which serves as a shared teacher signal for both representations. To ensure efficiency, we fine-tune the final layer of the dense encoder and the sparse projection head, enabling easy adaptation of any existing VLP model. Experiments on MSCOCO and Flickr30k demonstrate that our sparse retriever not only outperforms existing sparse baselines, but also achieves performance comparable to-or even surpassing-its dense counterparts, while retaining the benefits of sparse models. 

---
# Learn to Memorize: Optimizing LLM-based Agents with Adaptive Memory Framework 

**Authors**: Zeyu Zhang, Quanyu Dai, Rui Li, Xiaohe Bo, Xu Chen, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.16629)  

**Abstract**: LLM-based agents have been extensively applied across various domains, where memory stands out as one of their most essential capabilities. Previous memory mechanisms of LLM-based agents are manually predefined by human experts, leading to higher labor costs and suboptimal performance. In addition, these methods overlook the memory cycle effect in interactive scenarios, which is critical to optimizing LLM-based agents for specific environments. To address these challenges, in this paper, we propose to optimize LLM-based agents with an adaptive and data-driven memory framework by modeling memory cycles. Specifically, we design an MoE gate function to facilitate memory retrieval, propose a learnable aggregation process to improve memory utilization, and develop task-specific reflection to adapt memory storage. Our memory framework empowers LLM-based agents to learn how to memorize information effectively in specific environments, with both off-policy and on-policy optimization. In order to evaluate the effectiveness of our proposed methods, we conduct comprehensive experiments across multiple aspects. To benefit the research community in this area, we release our project at this https URL. 

---
