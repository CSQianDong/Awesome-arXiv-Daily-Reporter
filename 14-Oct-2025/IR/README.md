# FinVet: A Collaborative Framework of RAG and External Fact-Checking Agents for Financial Misinformation Detection 

**Authors**: Daniel Berhane Araya, Duoduo Liao  

**Link**: [PDF](https://arxiv.org/pdf/2510.11654)  

**Abstract**: Financial markets face growing threats from misinformation that can trigger billions in losses in minutes. Most existing approaches lack transparency in their decision-making and provide limited attribution to credible sources. We introduce FinVet, a novel multi-agent framework that integrates two Retrieval-Augmented Generation (RAG) pipelines with external fact-checking through a confidence-weighted voting mechanism. FinVet employs adaptive three-tier processing that dynamically adjusts verification strategies based on retrieval confidence, from direct metadata extraction to hybrid reasoning to full model-based analysis. Unlike existing methods, FinVet provides evidence-backed verdicts, source attribution, confidence scores, and explicit uncertainty flags when evidence is insufficient. Experimental evaluation on the FinFact dataset shows that FinVet achieves an F1 score of 0.85, which is a 10.4% improvement over the best individual pipeline (fact-check pipeline) and 37% improvement over standalone RAG approaches. 

---
# OneRec-Think: In-Text Reasoning for Generative Recommendation 

**Authors**: Zhanyu Liu, Shiyao Wang, Xingmei Wang, Rongzhou Zhang, Jiaxin Deng, Honghui Bao, Jinghao Zhang, Wuchao Li, Pengfei Zheng, Xiangyu Wu, Yifei Hu, Qigen Hu, Xinchen Luo, Lejian Ren, Zixing Zhang, Qianqian Wang, Kuo Cai, Yunfan Wu, Hongtao Cheng, Zexuan Cheng, Lu Ren, Huanjie Wang, Yi Su, Ruiming Tang, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.11639)  

**Abstract**: The powerful generative capacity of Large Language Models (LLMs) has instigated a paradigm shift in recommendation. However, existing generative models (e.g., OneRec) operate as implicit predictors, critically lacking the capacity for explicit and controllable reasoning-a key advantage of LLMs. To bridge this gap, we propose OneRec-Think, a unified framework that seamlessly integrates dialogue, reasoning, and personalized recommendation. OneRec-Think incorporates: (1) Itemic Alignment: cross-modal Item-Textual Alignment for semantic grounding; (2) Reasoning Activation: Reasoning Scaffolding to activate LLM reasoning within the recommendation context; and (3) Reasoning Enhancement, where we design a recommendation-specific reward function that accounts for the multi-validity nature of user preferences. Experiments across public benchmarks show state-of-the-art performance. Moreover, our proposed "Think-Ahead" architecture enables effective industrial deployment on Kuaishou, achieving a 0.159\% gain in APP Stay Time and validating the practical efficacy of the model's explicit reasoning capability. 

---
# REGENT: Relevance-Guided Attention for Entity-Aware Multi-Vector Neural Re-Ranking 

**Authors**: Shubham Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2510.11592)  

**Abstract**: Current neural re-rankers often struggle with complex information needs and long, content-rich documents. The fundamental issue is not computational--it is intelligent content selection: identifying what matters in lengthy, multi-faceted texts. While humans naturally anchor their understanding around key entities and concepts, neural models process text within rigid token windows, treating all interactions as equally important and missing critical semantic signals. We introduce REGENT, a neural re-ranking model that mimics human-like understanding by using entities as a "semantic skeleton" to guide attention. REGENT integrates relevance guidance directly into the attention mechanism, combining fine-grained lexical matching with high-level semantic reasoning. This relevance-guided attention enables the model to focus on conceptually important content while maintaining sensitivity to precise term matches. REGENT achieves new state-of-the-art performance in three challenging datasets, providing up to 108% improvement over BM25 and consistently outperforming strong baselines including ColBERT and RankVicuna. To our knowledge, this is the first work to successfully integrate entity semantics directly into neural attention, establishing a new paradigm for entity-aware information retrieval. 

---
# QDER: Query-Specific Document and Entity Representations for Multi-Vector Document Re-Ranking 

**Authors**: Shubham Chatterjee, Jeff Dalton  

**Link**: [PDF](https://arxiv.org/pdf/2510.11589)  

**Abstract**: Neural IR has advanced through two distinct paths: entity-oriented approaches leveraging knowledge graphs and multi-vector models capturing fine-grained semantics. We introduce QDER, a neural re-ranking model that unifies these approaches by integrating knowledge graph semantics into a multi-vector model. QDER's key innovation lies in its modeling of query-document relationships: rather than computing similarity scores on aggregated embeddings, we maintain individual token and entity representations throughout the ranking process, performing aggregation only at the final scoring stage - an approach we call "late aggregation." We first transform these fine-grained representations through learned attention patterns, then apply carefully chosen mathematical operations for precise matches. Experiments across five standard benchmarks show that QDER achieves significant performance gains, with improvements of 36% in nDCG@20 over the strongest baseline on TREC Robust 2004 and similar improvements on other datasets. QDER particularly excels on difficult queries, achieving an nDCG@20 of 0.70 where traditional approaches fail completely (nDCG@20 = 0.0), setting a foundation for future work in entity-aware retrieval. 

---
# Characterizing Web Search in The Age of Generative AI 

**Authors**: Elisabeth Kirsten, Jost Grosse Perdekamp, Mihir Upadhyay, Krishna P. Gummadi, Muhammad Bilal Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2510.11560)  

**Abstract**: The advent of LLMs has given rise to a new type of web search: Generative search, where LLMs retrieve web pages related to a query and generate a single, coherent text as a response. This output modality stands in stark contrast to traditional web search, where results are returned as a ranked list of independent web pages. In this paper, we ask: Along what dimensions do generative search outputs differ from traditional web search? We compare Google, a traditional web search engine, with four generative search engines from two providers (Google and OpenAI) across queries from four domains. Our analysis reveals intriguing differences. Most generative search engines cover a wider range of sources compared to web search. Generative search engines vary in the degree to which they rely on internal knowledge contained within the model parameters v.s. external knowledge retrieved from the web. Generative search engines surface varying sets of concepts, creating new opportunities for enhancing search diversity and serendipity. Our results also highlight the need for revisiting evaluation criteria for web search in the age of Generative AI. 

---
# Uncertainty Quantification for Retrieval-Augmented Reasoning 

**Authors**: Heydar Soudani, Hamed Zamani, Faegheh Hasibi  

**Link**: [PDF](https://arxiv.org/pdf/2510.11483)  

**Abstract**: Retrieval-augmented reasoning (RAR) is a recent evolution of retrieval-augmented generation (RAG) that employs multiple reasoning steps for retrieval and generation. While effective for some complex queries, RAR remains vulnerable to errors and misleading outputs. Uncertainty quantification (UQ) offers methods to estimate the confidence of systems' outputs. These methods, however, often handle simple queries with no retrieval or single-step retrieval, without properly handling RAR setup. Accurate estimation of UQ for RAR requires accounting for all sources of uncertainty, including those arising from retrieval and generation. In this paper, we account for all these sources and introduce Retrieval-Augmented Reasoning Consistency (R2C)--a novel UQ method for RAR. The core idea of R2C is to perturb the multi-step reasoning process by applying various actions to reasoning steps. These perturbations alter the retriever's input, which shifts its output and consequently modifies the generator's input at the next step. Through this iterative feedback loop, the retriever and generator continuously reshape one another's inputs, enabling us to capture uncertainty arising from both components. Experiments on five popular RAR systems across diverse QA datasets show that R2C improves AUROC by over 5% on average compared to the state-of-the-art UQ baselines. Extrinsic evaluations using R2C as an external signal further confirm its effectiveness for two downstream tasks: in Abstention, it achieves ~5% gains in both F1Abstain and AccAbstain; in Model Selection, it improves the exact match by ~7% over single models and ~3% over selection methods. 

---
# What Generative Search Engines Like and How to Optimize Web Content Cooperatively 

**Authors**: Yujiang Wu, Shanshan Zhong, Yubin Kim, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.11438)  

**Abstract**: By employing large language models (LLMs) to retrieve documents and generate natural language responses, Generative Engines, such as Google AI overview and ChatGPT, provide significantly enhanced user experiences and have rapidly become the new form of search. Their rapid adoption also drives the needs of Generative Engine Optimization (GEO), as content providers are eager to gain more traction from them. In this paper, we introduce AutoGEO, a framework to automatically learn generative engine preferences when using retrieved contents for response generation, and rewrite web contents for more such traction. AutoGEO first prompts frontier LLMs to explain generative engine preferences and extract meaningful preference rules from these explanations. Then it uses preference rules as context engineering for AutoGEO$_\text{API}$, a prompt-based GEO system, and as rule-based rewards to train AutoGEO$_\text{Mini}$, a cost-effective GEO model. Experiments on the standard GEO-Bench and two newly constructed benchmarks using real user queries demonstrate the effectiveness of AutoGEO in enhancing content traction while preserving search utility. Analyses confirm the learned rules' robustness and abilities to capture unique preferences in variant domains, and AutoGEO systems' ability to embed them in content optimization. The code is released at this https URL. 

---
# On Inherited Popularity Bias in Cold-Start Item Recommendation 

**Authors**: Gregor Meehan, Johan Pauwels  

**Link**: [PDF](https://arxiv.org/pdf/2510.11402)  

**Abstract**: Collaborative filtering (CF) recommender systems struggle with making predictions on unseen, or 'cold', items. Systems designed to address this challenge are often trained with supervision from warm CF models in order to leverage collaborative and content information from the available interaction data. However, since they learn to replicate the behavior of CF methods, cold-start models may therefore also learn to imitate their predictive biases. In this paper, we show that cold-start systems can inherit popularity bias, a common cause of recommender system unfairness arising when CF models overfit to more popular items, thereby maximizing user-oriented accuracy but neglecting rarer items. We demonstrate that cold-start recommenders not only mirror the popularity biases of warm models, but are in fact affected more severely: because they cannot infer popularity from interaction data, they instead attempt to estimate it based solely on content features. This leads to significant over-prediction of certain cold items with similar content to popular warm items, even if their ground truth popularity is very low. Through experiments on three multimedia datasets, we analyze the impact of this behavior on three generative cold-start methods. We then describe a simple post-processing bias mitigation method that, by using embedding magnitude as a proxy for predicted popularity, can produce more balanced recommendations with limited harm to user-oriented cold-start accuracy. 

---
# VeriCite: Towards Reliable Citations in Retrieval-Augmented Generation via Rigorous Verification 

**Authors**: Haosheng Qian, Yixing Fan, Jiafeng Guo, Ruqing Zhang, Qi Chen, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.11394)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a crucial approach for enhancing the responses of large language models (LLMs) with external knowledge sources. Despite the impressive performance in complex question-answering tasks, RAG still struggles with hallucinations. Attributing RAG-generated content through in-line citations has demonstrated potential in reducing hallucinations and facilitating human verification. Existing citation generation methods primarily rely on either fine-tuning the generator or employing post-processing approaches for citation matching. However, the former approach demands substantial annotated data and computational resources, while the latter often encounters difficulties in managing multiple citations and frequently produces suboptimal results. In this paper, we introduce a novel framework, called VeriCite, designed to rigorously validate supporting evidence and enhance answer attribution. Specifically, VeriCite breaks down into a three-stage generation: 1) The initial answer generation first generates a response based on all available contexts and has its claims verified through the NLI model; 2) the supporting evidence selection assesses the utility of each document and extracts useful supporting evidences; 3) the final answer refinement integrates the initial response and collected evidences to produce the final, refined this http URL conduct experiments across five open-source LLMs and four datasets, demonstrating that VeriCite can significantly improve citation quality while maintaining the correctness of the answers. 

---
# Dynamic Network-Based Two-Stage Time Series Forecasting for Affiliate Marketing 

**Authors**: Zhe Wang, Yaming Yang, Ziyu Guan, Bin Tong, Rui Wang, Wei Zhao, Hongbo Deng  

**Link**: [PDF](https://arxiv.org/pdf/2510.11323)  

**Abstract**: In recent years, affiliate marketing has emerged as a revenue-sharing strategy where merchants collaborate with promoters to promote their products. It not only increases product exposure but also allows promoters to earn a commission. This paper addresses the pivotal yet under-explored challenge in affiliate marketing: accurately assessing and predicting the contributions of promoters in product promotion. We design a novel metric for evaluating the indirect contributions of the promoter, called propagation scale. Unfortunately, existing time series forecasting techniques fail to deliver accurate predictions due to the propagation scale being influenced by multiple factors and the inherent complexities arising from dynamic scenarios. To address this issue, we decouple the network structure from the node signals and propose a two-stage solution: initially, the basic self-sales and network structure prediction are conducted separately, followed by the synthesis of the propagation scale. Specifically, we design a graph convolution encoding scheme based on descendant neighbors and incorporate hypergraph convolution to efficiently capture complex promotional dynamics. Additionally, three auxiliary tasks are employed: self-sales prediction for base estimations, descendant prediction to synthesize propagation scale, and promoter activation prediction to mitigate high volatility issues. Extensive offline experiments on large-scale industrial datasets validate the superiority of our method. We further deploy our model on Alimama platform with over $100,000$ promoters, achieving a $9.29\%$ improvement in GMV and a $5.89\%$ increase in sales volume. 

---
# Next Interest Flow: A Generative Pre-training Paradigm for Recommender Systems by Modeling All-domain Movelines 

**Authors**: Chen Gao, Zixin Zhao, Lv Shao, Tong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11317)  

**Abstract**: Click-Through Rate (CTR) prediction, a cornerstone of modern recommender systems, has been dominated by discriminative models that react to past user behavior rather than proactively modeling user intent. Existing generative paradigms attempt to address this but suffer from critical limitations: Large Language Model (LLM) based methods create a semantic mismatch by forcing e-commerce signals into a linguistic space, while ID-based generation is constrained by item memorization and cold-start issues. To overcome these limitations, we propose a novel generative pre-training paradigm. Our model learns to predict the Next Interest Flow, a dense vector sequence representing a user's future intent, while simultaneously modeling its internal Interest Diversity and Interest Evolution Velocity to ensure the representation is both rich and coherent. However, this two-stage approach introduces a critical objective mismatch between the generative and discriminative stages. We resolve this via a bidirectional alignment strategy, which harmonizes the two stages through cross-stage weight initialization and a dynamic Semantic Alignment Module for fine-tuning. Additionally, we enhance the underlying discriminative model with a Temporal Sequential Pairwise (TSP) mechanism to better capture temporal causality. We present the All-domain Moveline Evolution Network (AMEN), a unified framework implementing our entire pipeline. Extensive offline experiments validate AMEN's superiority over strong baselines, and a large-scale online A/B test demonstrates its significant real-world impact, delivering substantial improvements in key business metrics. 

---
# DyKnow-RAG: Dynamic Knowledge Utilization Reinforcement Framework for Noisy Retrieval-Augmented Generation in E-commerce Search Relevance 

**Authors**: Tingqiao Xu, Shaowei Yao, Chenhe Dong, Yiming Jin, Zerui Huang, Dan Ou, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11122)  

**Abstract**: Accurately modeling query-item relevance drives e-commerce ranking, yet long-tail, knowledge-heavy, and fast-evolving queries exceed parametric LLM coverage. External context (reviews, attribute encyclopedias, UGC) can help but is noisy, and single-pass latency and cost forbid any clean-then-summarize step. The model must, per query, judge relevance and decide whether to use, partially use, or ignore the context. DyKnow-RAG is a dynamic noisy-RAG framework built on Group Relative Policy Optimization. It trains two rollout groups (no external context vs a single retrieved chunk) and applies posterior-driven inter-group advantage scaling that adaptively reweights their contributions by the per-query correctness gap. This teaches when to trust retrieval versus fall back to parametric knowledge, without process labels, value networks, or extra inference passes, preserving single-pass, single-chunk deployment under production latency. Training combines: (1) supervised initialization with a structured rationale that explicitly records the context-usage decision; (2) an RL pool prioritized by SFT uncertainty to focus where context choice is most consequential; and (3) an optional lightweight DPO warm start to stabilize with-context calibration. Under a unified retrieval/index and fixed latency budget, DyKnow-RAG outperforms SFT, DPO, and vanilla GRPO in offline tests, and delivers consistent lifts on GSB, Query Goodrate, and Item Goodrate in Taobao A/B testing. It is deployed in Taobao's production relevance system, serving live traffic. To our knowledge, it is among the first single-pass RAG solutions for e-commerce relevance, turning noisy external signals into reliable gains without added online complexity. 

---
# HoMer: Addressing Heterogeneities by Modeling Sequential and Set-wise Contexts for CTR Prediction 

**Authors**: Shuwei Chen, Jiajun Cui, Zhengqi Xu, Fan Zhang, Jiangke Fan, Teng Zhang, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11100)  

**Abstract**: Click-through rate (CTR) prediction, which models behavior sequence and non-sequential features (e.g., user/item profiles or cross features) to infer user interest, underpins industrial recommender systems. However, most methods face three forms of heterogeneity that degrade predictive performance: (i) Feature Heterogeneity persists when limited sequence side features provide less granular interest representation compared to extensive non-sequential features, thereby impairing sequence modeling performance; (ii) Context Heterogeneity arises because a user's interest in an item will be influenced by other items, yet point-wise prediction neglects cross-item interaction context from the entire item set; (iii) Architecture Heterogeneity stems from the fragmented integration of specialized network modules, which compounds the model's effectiveness, efficiency and scalability in industrial deployments. To tackle the above limitations, we propose HoMer, a Homogeneous-Oriented TransforMer for modeling sequential and set-wise contexts. First, we align sequence side features with non-sequential features for accurate sequence modeling and fine-grained interest representation. Second, we shift the prediction paradigm from point-wise to set-wise, facilitating cross-item interaction in a highly parallel manner. Third, HoMer's unified encoder-decoder architecture achieves dual optimization through structural simplification and shared computation, ensuring computational efficiency while maintaining scalability with model size. Without arduous modification to the prediction pipeline, HoMer successfully scales up and outperforms our industrial baseline by 0.0099 in the AUC metric, and enhances online business metrics like CTR/RPM by 1.99%/2.46%. Additionally, HoMer saves 27% of GPU resources via preliminary engineering optimization, further validating its superiority and practicality. 

---
# Decoupled Multimodal Fusion for User Interest Modeling in Click-Through Rate Prediction 

**Authors**: Alin Fan, Hanqing Li, Sihan Lu, Jingsong Yuan, Jiandong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11066)  

**Abstract**: Modern industrial recommendation systems improve recommendation performance by integrating multimodal representations from pre-trained models into ID-based Click-Through Rate (CTR) prediction frameworks. However, existing approaches typically adopt modality-centric modeling strategies that process ID-based and multimodal embeddings independently, failing to capture fine-grained interactions between content semantics and behavioral signals. In this paper, we propose Decoupled Multimodal Fusion (DMF), which introduces a modality-enriched modeling strategy to enable fine-grained interactions between ID-based collaborative representations and multimodal representations for user interest modeling. Specifically, we construct target-aware features to bridge the semantic gap across different embedding spaces and leverage them as side information to enhance the effectiveness of user interest modeling. Furthermore, we design an inference-optimized attention mechanism that decouples the computation of target-aware features and ID-based embeddings before the attention layer, thereby alleviating the computational bottleneck introduced by incorporating target-aware features. To achieve comprehensive multimodal integration, DMF combines user interest representations learned under the modality-centric and modality-enriched modeling strategies. Offline experiments on public and industrial datasets demonstrate the effectiveness of DMF. Moreover, DMF has been deployed on the product recommendation system of the international e-commerce platform Lazada, achieving relative improvements of 5.30% in CTCVR and 7.43% in GMV with negligible computational overhead. 

---
# From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevance 

**Authors**: Runze Xia, Yupeng Ji, Yuxi Zhou, Haodong Liu, Teng Zhang, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.11056)  

**Abstract**: Query-service relevance prediction in e-commerce search systems faces strict latency requirements that prevent the direct application of Large Language Models (LLMs). To bridge this gap, we propose a two-stage reasoning distillation framework to transfer reasoning capabilities from a powerful teacher LLM to a lightweight, deployment-friendly student model. In the first stage, we address the limitations of general-purpose LLMs by constructing a domain-adapted teacher model. This is achieved through a three-step process: domain-adaptive pre-training to inject platform knowledge, supervised fine-tuning to elicit reasoning skills, and preference optimization with a multi-dimensional reward model to ensure the generation of reliable and preference-aligned reasoning paths. This teacher can then automatically annotate massive query-service pairs from search logs with both relevance labels and reasoning chains. In the second stage, to address the challenges of architectural heterogeneity in standard distillation, we introduce Contrastive Reasoning Self-Distillation (CRSD). By modeling the behavior of the same student model under "standard" and "reasoning-augmented" inputs as a teacher-student relationship, CRSD enables the lightweight model to internalize the teacher's complex decision-making mechanisms without needing the explicit reasoning path at inference. Offline evaluations and online A/B testing in the Meituan search advertising system demonstrate that our framework achieves significant improvements across multiple metrics, validating its effectiveness and practical value. 

---
# Does LLM Focus on the Right Words? Diagnosing Language Bias in LLM-based Recommenders 

**Authors**: Bohao Wang, Jiawei Chen, Feng Liu, Changwang Zhang, Jun Wang, Canghong Jin, Chun Chen, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10978)  

**Abstract**: Large language models (LLMs), owing to their extensive open-domain knowledge and semantic reasoning capabilities, have been increasingly integrated into recommender systems (RS). However, a substantial gap remains between the pre-training objectives of LLMs and the specific requirements of recommendation tasks. To address this gap, supervised fine-tuning (SFT) is commonly performed on specially curated recommendation datasets to further enhance their predictive ability. Despite its success, SFT exhibits a critical limitation: it induces Language Bias, whereby the model over-relies on auxiliary tokens-such as task descriptions and prefix-generated tokens-while underutilizing core user interaction tokens that encode user-specific preferences. This bias not only undermines recommendation accuracy but also raises unfairness concerns.
To address this issue, we propose Group Distributionally Robust Optimization-based Tuning (GDRT), a novel fine-tuning paradigm that enforces consistent model performance across token groups with varying degrees of relevance to auxiliary tokens. By adaptively upweighting underperforming groups, typically those weakly correlated with auxiliary tokens, GDRT shifts the model's attention from superficial auxiliary cues to informative user interaction tokens, thereby mitigating language bias. Extensive experiments conducted on three public datasets demonstrate that GDRT effectively mitigates language bias, yielding substantial improvements in recommendation accuracy (with an average NDCG@10 gain of 24.29%) and significantly enhancing recommendation fairness. 

---
# HatLLM: Hierarchical Attention Masking for Enhanced Collaborative Modeling in LLM-based Recommendation 

**Authors**: Yu Cui, Feng Liu, Jiawei Chen, Canghong Jin, Xingyu Lou, Changwang Zhang, Jun Wang, Yuegang Sun, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10955)  

**Abstract**: Recent years have witnessed a surge of research on leveraging large language models (LLMs) for sequential recommendation. LLMs have demonstrated remarkable potential in inferring users' nuanced preferences through fine-grained semantic reasoning. However, they also exhibit a notable limitation in effectively modeling collaborative signals, i.e., behavioral correlations inherent in users' historical interactions. Our empirical analysis further reveals that the attention mechanisms in LLMs tend to disproportionately focus on tokens within the same item, thereby impeding the capture of cross-item correlations.
To address this limitation, we propose a novel hierarchical attention masking strategy for LLM-based recommendation, termed HatLLM. Specifically, in shallow layers, HatLLM masks attention between tokens from different items, facilitating intra-item semantic understanding; in contrast, in deep layers, HatLLM masks attention within items, thereby compelling the model to capture cross-item correlations. This progressive, layer-wise approach enables LLMs to jointly model both token-level and item-level dependencies. Extensive experiments on three real-world datasets demonstrate that HatLLM achieves significant performance gains (9.13% on average) over existing LLM-based methods. 

---
# Comparative Explanations via Counterfactual Reasoning in Recommendations 

**Authors**: Yi Yu, Zhenxing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10920)  

**Abstract**: Explainable recommendation through counterfactual reasoning seeks to identify the influential aspects of items in recommendations, which can then be used as explanations. However, state-of-the-art approaches, which aim to minimize changes in product aspects while reversing their recommended decisions according to an aggregated decision boundary score, often lead to factual inaccuracies in explanations. To solve this problem, in this work we propose a novel method of Comparative Counterfactual Explanations for Recommendation (CoCountER). CoCountER creates counterfactual data based on soft swap operations, enabling explanations for recommendations of arbitrary pairs of comparative items. Empirical experiments validate the effectiveness of our approach. 

---
# VeritasFi: An Adaptable, Multi-tiered RAG Framework for Multi-modal Financial Question Answering 

**Authors**: Zhenghan Tai, Hanwei Wu, Qingchen Hu, Jijun Chi, Hailin He, Lei Ding, Tung Sum Thomas Kwok, Bohuai Xiao, Yuchen Hua, Suyuchen Wang, Peng Lu, Muzhi Li, Yihong Wu, Liheng Ma, Jerry Huang, Jiayi Zhang, Gonghao Zhang, Chaolong Jiang, Jingrui Tian, Sicheng Lyu, Zeyu Li, Boyu Han, Fengran Mo, Xinyue Yu, Yufei Cui, Ling Zhou, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10828)  

**Abstract**: Retrieval-Augmented Generation (RAG) is becoming increasingly essential for Question Answering (QA) in the financial sector, where accurate and contextually grounded insights from complex public disclosures are crucial. However, existing financial RAG systems face two significant challenges: (1) they struggle to process heterogeneous data formats, such as text, tables, and figures; and (2) they encounter difficulties in balancing general-domain applicability with company-specific adaptation. To overcome these challenges, we present VeritasFi, an innovative hybrid RAG framework that incorporates a multi-modal preprocessing pipeline alongside a cutting-edge two-stage training strategy for its re-ranking component. VeritasFi enhances financial QA through three key innovations: (1) A multi-modal preprocessing pipeline that seamlessly transforms heterogeneous data into a coherent, machine-readable format. (2) A tripartite hybrid retrieval engine that operates in parallel, combining deep multi-path retrieval over a semantically indexed document corpus, real-time data acquisition through tool utilization, and an expert-curated memory bank for high-frequency questions, ensuring comprehensive scope, accuracy, and efficiency. (3) A two-stage training strategy for the document re-ranker, which initially constructs a general, domain-specific model using anonymized data, followed by rapid fine-tuning on company-specific data for targeted applications. By integrating our proposed designs, VeritasFi presents a groundbreaking framework that greatly enhances the adaptability and robustness of financial RAG systems, providing a scalable solution for both general-domain and company-specific QA tasks. Code accompanying this work is available at this https URL. 

---
# Multi-Granularity Sequence Denoising with Weakly Supervised Signal for Sequential Recommendation 

**Authors**: Liang Li, Zhou Yang, Xiaofei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10564)  

**Abstract**: Sequential recommendation aims to predict the next item based on user interests in historical interaction sequences. Historical interaction sequences often contain irrelevant noisy items, which significantly hinders the performance of recommendation systems. Existing research employs unsupervised methods that indirectly identify item-granularity irrelevant noise by predicting the ground truth item. Since these methods lack explicit noise labels, they are prone to misidentify users' interested items as noise. Additionally, while these methods focus on removing item-granularity noise driven by the ground truth item, they overlook interest-granularity noise, limiting their ability to perform broader denoising based on user interests. To address these issues, we propose Multi-Granularity Sequence Denoising with Weakly Supervised Signal for Sequential Recommendation(MGSD-WSS). MGSD-WSS first introduces the Multiple Gaussian Kernel Perceptron module to map the original and enhance sequence into a common representation space and utilizes weakly supervised signals to accurately identify noisy items in the historical interaction sequence. Subsequently, it employs the item-granularity denoising module with noise-weighted contrastive learning to obtain denoised item representations. Then, it extracts target interest representations from the ground truth item and applies noise-weighted contrastive learning to obtain denoised interest representations. Finally, based on the denoised item and interest representations, MGSD-WSS predicts the next item. Extensive experiments on five datasets demonstrate that the proposed method significantly outperforms state-of-the-art sequence recommendation and denoising models. Our code is available at this https URL. 

---
# Self-Supervised Representation Learning with ID-Content Modality Alignment for Sequential Recommendation 

**Authors**: Donglin Zhou, Weike Pan, Zhong Ming  

**Link**: [PDF](https://arxiv.org/pdf/2510.10556)  

**Abstract**: Sequential recommendation (SR) models often capture user preferences based on the historically interacted item IDs, which usually obtain sub-optimal performance when the interaction history is limited. Content-based sequential recommendation has recently emerged as a promising direction that exploits items' textual and visual features to enhance preference learning. However, there are still three key challenges: (i) how to reduce the semantic gap between different content modality representations; (ii) how to jointly model user behavior preferences and content preferences; and (iii) how to design an effective training strategy to align ID representations and content representations. To address these challenges, we propose a novel model, self-supervised representation learning with ID-Content modality alignment, named SICSRec. Firstly, we propose a LLM-driven sample construction method and develop a supervised fine-tuning approach to align item-level modality representations. Secondly, we design a novel Transformer-based sequential model, where an ID-modality sequence encoder captures user behavior preferences, a content-modality sequence encoder learns user content preferences, and a mix-modality sequence decoder grasps the intrinsic relationship between these two types of preferences. Thirdly, we propose a two-step training strategy with a content-aware contrastive learning task to align modality representations and ID representations, which decouples the training process of content modality dependency and item collaborative dependency. Extensive experiments conducted on four public video streaming datasets demonstrate our SICSRec outperforms the state-of-the-art ID-modality sequential recommenders and content-modality sequential recommenders by 8.04% on NDCG@5 and 6.62% on NDCD@10 on average, respectively. 

---
# Towards Long-Term User Welfare in Recommender Systems via Creator-Oriented Information Revelation 

**Authors**: Xu Zhao, Xiaopeng Ye, Chen Xu, Weiran Shen, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10511)  

**Abstract**: Improving the long-term user welfare (e.g., sustained user engagement) has become a central objective of recommender systems (RS). In real-world platforms, the creation behaviors of content creators plays a crucial role in shaping long-term welfare beyond short-term recommendation accuracy, making the effective steering of creator behavior essential to foster a healthier RS ecosystem. Existing works typically rely on re-ranking algorithms that heuristically adjust item exposure to steer creators' behavior. However, when embedded within recommendation pipelines, such a strategy often conflicts with the short-term objective of improving recommendation accuracy, leading to performance degradation and suboptimal long-term welfare. The well-established economics studies offer us valuable insights for an alternative approach without relying on recommendation algorithmic design: revealing information from an information-rich party (sender) to a less-informed party (receiver) can effectively change the receiver's beliefs and steer their behavior. Inspired by this idea, we propose an information-revealing framework, named Long-term Welfare Optimization via Information Revelation (LoRe). In this framework, we utilize a classical information revelation method (i.e., Bayesian persuasion) to map the stakeholders in RS, treating the platform as the sender and creators as the receivers. To address the challenge posed by the unrealistic assumption of traditional economic methods, we formulate the process of information revelation as a Markov Decision Process (MDP) and propose a learning algorithm trained and inferred in environments with boundedly rational creators. Extensive experiments on two real-world RS datasets demonstrate that our method can effectively outperform existing fair re-ranking methods and information revealing strategies in improving long-term user welfare. 

---
# Does Weighting Improve Matrix Factorization for Recommender Systems? 

**Authors**: Alex Ayoub, Samuel Robertson, Dawen Liang, Harald Steck, Nathan Kallus  

**Link**: [PDF](https://arxiv.org/pdf/2510.10440)  

**Abstract**: Matrix factorization is a widely used approach for top-N recommendation and collaborative filtering. When implemented on implicit feedback data (such as clicks), a common heuristic is to upweight the observed interactions. This strategy has been shown to improve performance for certain algorithms. In this paper, we conduct a systematic study of various weighting schemes and matrix factorization algorithms. Somewhat surprisingly, we find that training with unweighted data can perform comparably to, and sometimes outperform, training with weighted data, especially for large models. This observation challenges the conventional wisdom. Nevertheless, we identify cases where weighting can be beneficial, particularly for models with lower capacity and specific regularization schemes. We also derive efficient algorithms for exactly minimizing several weighted objectives that were previously considered computationally intractable. Our work provides a comprehensive analysis of the interplay between weighting, regularization, and model capacity in matrix factorization for recommender systems. 

---
# ZeroGR: A Generalizable and Scalable Framework for Zero-Shot Generative Retrieval 

**Authors**: Weiwei Sun, Keyi Kong, Xinyu Ma, Shuaiqiang Wang, Dawei Yin, Maarten de Rijke, Zhaochun Ren, Yiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10419)  

**Abstract**: Generative retrieval (GR) reformulates information retrieval (IR) by framing it as the generation of document identifiers (docids), thereby enabling an end-to-end optimization and seamless integration with generative language models (LMs). Despite notable progress under supervised training, GR still struggles to generalize to zero-shot IR scenarios, which are prevalent in real-world applications. To tackle this challenge, we propose \textsc{ZeroGR}, a zero-shot generative retrieval framework that leverages natural language instructions to extend GR across a wide range of IR tasks. Specifically, \textsc{ZeroGR} is composed of three key components: (i) an LM-based docid generator that unifies heterogeneous documents (e.g., text, tables, code) into semantically meaningful docids; (ii) an instruction-tuned query generator that generates diverse types of queries from natural language task descriptions to enhance corpus indexing; and (iii) a reverse annealing decoding strategy to balance precision and recall during docid generation. We investigate the impact of instruction fine-tuning scale and find that performance consistently improves as the number of IR tasks encountered during training increases. Empirical results on the BEIR and MAIR benchmarks demonstrate that \textsc{ZeroGR} outperforms strong dense retrieval and generative baselines in zero-shot settings, establishing a new state-of-the-art for instruction-driven GR. 

---
# Breaking the Likelihood Trap: Consistent Generative Recommendation with Graph-structured Model 

**Authors**: Qiya Yang, Xiaoxi Liang, Zeping Xiao, Yingjie Deng, Yalong Wang, Yongqi Liu, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.10127)  

**Abstract**: Reranking, as the final stage of recommender systems, demands real-time inference, accuracy, and diversity. It plays a crucial role in determining the final exposure, directly influencing user experience. Recently, generative reranking has gained increasing attention for its strong ability to model complex dependencies among items. However, most existing methods suffer from the "likelihood trap", where high-likelihood sequences are often perceived as low-quality by humans. These models tend to repeatedly recommend a set of high-frequency items, resulting in list homogeneity, thereby limiting user engagement. In this work, we propose Consistent Graph-structured Generative Recommendation (Congrats), a novel generative reranking framework. To break the likelihood trap, we introduce a novel graph-structured decoder that can capture diverse sequences along multiple paths. This design not only expands the decoding space to promote diversity, but also improves prediction accuracy by implicit item dependencies derived from vertex transitions. Furthermore, we design a differentiable cascade system that incorporates an evaluator, enabling the model to learn directly from user preferences as the training objective. Extensive offline experiments validate the superior performance of Congrats over state-of-the-art reranking methods. Moreover, Congrats has been evaluated on a large-scale video-sharing app, Kuaishou, with over 300 million daily active users, demonstrating that our approach significantly improves both recommendation quality and diversity, validating our effectiveness in practical industrial environments. 

---
# Integrating Structure-Aware Attention and Knowledge Graphs in Explainable Recommendation Systems 

**Authors**: Shuangquan Lyu, Ming Wang, Huajun Zhang, Jiasen Zheng, Junjiang Lin, Xiaoxuan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.10109)  

**Abstract**: This paper designs and implements an explainable recommendation model that integrates knowledge graphs with structure-aware attention mechanisms. The model is built on graph neural networks and incorporates a multi-hop neighbor aggregation strategy. By integrating the structural information of knowledge graphs and dynamically assigning importance to different neighbors through an attention mechanism, the model enhances its ability to capture implicit preference relationships. In the proposed method, users and items are embedded into a unified graph structure. Multi-level semantic paths are constructed based on entities and relations in the knowledge graph to extract richer contextual information. During the rating prediction phase, recommendations are generated through the interaction between user and target item representations. The model is optimized using a binary cross-entropy loss function. Experiments conducted on the Amazon Books dataset validate the superior performance of the proposed model across various evaluation metrics. The model also shows good convergence and stability. These results further demonstrate the effectiveness and practicality of structure-aware attention mechanisms in knowledge graph-enhanced recommendation. 

---
# CardRewriter: Leveraging Knowledge Cards for Long-Tail Query Rewriting on Short-Video Platforms 

**Authors**: Peiyuan Gong, Feiran Zhu, Yaqi Yin, Chenglei Dai, Chao Zhang, Kai Zheng, Wentian Bao, Jiaxin Mao, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10095)  

**Abstract**: Short-video platforms have rapidly become a new generation of information retrieval systems, where users formulate queries to access desired videos. However, user queries, especially long-tail ones, often suffer from spelling errors, incomplete phrasing, and ambiguous intent, resulting in mismatches between user expectations and retrieved results. While large language models (LLMs) have shown success in long-tail query rewriting within e-commerce, they struggle on short-video platforms, where proprietary content such as short videos, live streams, micro dramas, and user social networks falls outside their training distribution. To address this challenge, we introduce \textbf{CardRewriter}, an LLM-based framework that incorporates domain-specific knowledge to enhance long-tail query rewriting. For each query, our method aggregates multi-source knowledge relevant to the query and summarizes it into an informative and query-relevant knowledge card. This card then guides the LLM to better capture user intent and produce more effective query rewrites. We optimize CardRewriter using a two-stage training pipeline: supervised fine-tuning followed by group relative policy optimization, with a tailored reward system balancing query relevance and retrieval effectiveness. Offline experiments show that CardRewriter substantially improves rewriting quality for queries targeting proprietary content. Online A/B testing further confirms significant gains in long-view rate (LVR) and click-through rate (CTR), along with a notable reduction in initiative query reformulation rate (IQRR). Since September 2025, CardRewriter has been deployed on Kuaishou, one of China's largest short-video platforms, serving hundreds of millions of users daily. 

---
# PairSem: LLM-Guided Pairwise Semantic Matching for Scientific Document Retrieval 

**Authors**: Wonbin Kweon, Runchu Tian, SeongKu Kang, Pengcheng Jiang, Zhiyong Lu, Jiawei Han, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09897)  

**Abstract**: Scientific document retrieval is a critical task for enabling knowledge discovery and supporting research across diverse domains. However, existing dense retrieval methods often struggle to capture fine-grained scientific concepts in texts due to their reliance on holistic embeddings and limited domain understanding. Recent approaches leverage large language models (LLMs) to extract fine-grained semantic entities and enhance semantic matching, but they typically treat entities as independent fragments, overlooking the multi-faceted nature of scientific concepts. To address this limitation, we propose Pairwise Semantic Matching (PairSem), a framework that represents relevant semantics as entity-aspect pairs, capturing complex, multi-faceted scientific concepts. PairSem is unsupervised, base retriever-agnostic, and plug-and-play, enabling precise and context-aware matching without requiring query-document labels or entity annotations. Extensive experiments on multiple datasets and retrievers demonstrate that PairSem significantly improves retrieval performance, highlighting the importance of modeling multi-aspect semantics in scientific information retrieval. 

---
# MTMD: A Multi-Task Multi-Domain Framework for Unified Ad Lightweight Ranking at Pinterest 

**Authors**: Xiao Yang, Peifeng Yin, Abe Engle, Jinfeng Zhuang, Ling Leng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09857)  

**Abstract**: The lightweight ad ranking layer, living after the retrieval stage and before the fine ranker, plays a critical role in the success of a cascaded ad recommendation system. Due to the fact that there are multiple optimization tasks depending on the ad domain, e.g., Click Through Rate (CTR) for click ads and Conversion Rate (CVR) for conversion ads, as well as multiple surfaces where an ad is served (home feed, search, or related item recommendation) with diverse ad products (shopping or standard ad); it is an essentially challenging problem in industry on how to do joint holistic optimization in the lightweight ranker, such that the overall platform's value, advertiser's value, and user's value are maximized.
Deep Neural Network (DNN)-based multitask learning (MTL) can handle multiple goals naturally, with each prediction head mapping to a particular optimization goal. However, in practice, it is unclear how to unify data from different surfaces and ad products into a single model. It is critical to learn domain-specialized knowledge and explicitly transfer knowledge between domains to make MTL effective. We present a Multi-Task Multi-Domain (MTMD) architecture under the classic Two-Tower paradigm, with the following key contributions: 1) handle different prediction tasks, ad products, and ad serving surfaces in a unified framework; 2) propose a novel mixture-of-expert architecture to learn both specialized knowledge each domain and common knowledge shared between domains; 3) propose a domain adaption module to encourage knowledge transfer between experts; 4) constrain the modeling of different prediction tasks. MTMD improves the offline loss value by 12% to 36%, mapping to 2% online reduction in cost per click. We have deployed this single MTMD framework into production for Pinterest ad recommendation replacing 9 production models. 

---
# SemCSE-Multi: Multifaceted and Decodable Embeddings for Aspect-Specific and Interpretable Scientific Domain Mapping 

**Authors**: Marc Brinner, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2510.11599)  

**Abstract**: We propose SemCSE-Multi, a novel unsupervised framework for generating multifaceted embeddings of scientific abstracts, evaluated in the domains of invasion biology and medicine. These embeddings capture distinct, individually specifiable aspects in isolation, thus enabling fine-grained and controllable similarity assessments as well as adaptive, user-driven visualizations of scientific domains. Our approach relies on an unsupervised procedure that produces aspect-specific summarizing sentences and trains embedding models to map semantically related summaries to nearby positions in the embedding space. We then distill these aspect-specific embedding capabilities into a unified embedding model that directly predicts multiple aspect embeddings from a scientific abstract in a single, efficient forward pass. In addition, we introduce an embedding decoding pipeline that decodes embeddings back into natural language descriptions of their associated aspects. Notably, we show that this decoding remains effective even for unoccupied regions in low-dimensional visualizations, thus offering vastly improved interpretability in user-centric settings. 

---
# LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation 

**Authors**: Hengran Zhang, Keping Bi, Jiafeng Guo, Jiaming Zhang, Shuaiqiang Wang, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.11358)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge. While traditional retrieval focuses on relevance, RAG's effectiveness depends on the utility of retrieved passages, i.e., the usefulness in facilitating the generation of an accurate and comprehensive answer. Existing studies often treat utility as a generic attribute, ignoring the fact that different LLMs may benefit differently from the same passage due to variations in internal knowledge and comprehension ability. In this work, we introduce and systematically investigate the notion of LLM-specific utility. Through large-scale experiments across multiple datasets and LLMs, we demonstrate that human-annotated passages are not optimal for LLMs and that ground-truth utilitarian passages are not transferable across different LLMs. These findings highlight the necessity of adopting the LLM-specific utility in RAG research. Our findings indicate that some human-annotated passages are not ground-truth utilitarian passages for specific LLMs, partially due to the varying readability of queries and passages for LLMs, a tendency for which perplexity is a key metric. Based on these findings, we propose a benchmarking procedure for LLM-specific utility judgments. We evaluate existing utility judgment methods on six datasets and find that while verbalized methods using pseudo-answers perform robustly, LLMs struggle to assess utility effectively-failing to reject all passages for known queries and to select truly useful ones for unknown queries. 

---
# ELMO: Efficiency via Low-precision and Peak Memory Optimization in Large Output Spaces 

**Authors**: Jinbin Zhang, Nasib Ullah, Erik Schultheis, Rohit Babbar  

**Link**: [PDF](https://arxiv.org/pdf/2510.11168)  

**Abstract**: Large output spaces, also referred to as Extreme multilabel classification (XMC), is a setting that arises, e.g., in large-scale tagging and product-to-product recommendation, and is characterized by the number of labels ranging from hundreds of thousands to millions. This means that the linear classification head, usually only a tiny fraction of the overall model, turns into the main driver for compute and memory demand. Current state-of-the-art XMC methods predominantly rely on FP16-FP32 mixed-precision training, which we show can be unstable, and inefficient in terms of memory usage and computational overhead. Meanwhile, existing low-precision methods typically retain higher precision for the classification layer. In this work, we propose ELMO, a pure low-precision training framework for XMC models using BFloat16 and Float8 data types. By leveraging Kahan summation and stochastic rounding, we demonstrate that XMC models can be effectively trained entirely in Float8, without relying on single-precision master weights or tensor scaling. Low-precision training, combined with our proposed memory optimizations -- gradient fusion and chunking -- enables significant reductions in GPU memory usage. For example, we train a 3-million-label XMC model with only 6.6 GiB of GPU memory, compared to the 39.7 GiB required by the optimized SOTA method, Renee without compromising accuracy. 

---
# FBS Model-based Maintenance Record Accumulation for Failure-Cause Inference in Manufacturing Systems 

**Authors**: Takuma Fujiu, Sho Okazaki, Kohei Kaminishi, Yuji Nakata, Shota Hamamoto, Kenshin Yokose, Tatsunori Hara, Yasushi Umeda, Jun Ota  

**Link**: [PDF](https://arxiv.org/pdf/2510.11003)  

**Abstract**: In manufacturing systems, identifying the causes of failures is crucial for maintaining and improving production efficiency. In knowledge-based failure-cause inference, it is important that the knowledge base (1) explicitly structures knowledge about the target system and about failures, and (2) contains sufficiently long causal chains of failures. In this study, we constructed Diagnostic Knowledge Ontology and proposed a Function-Behavior-Structure (FBS) model-based maintenance-record accumulation method based on it. Failure-cause inference using the maintenance records accumulated by the proposed method showed better agreement with the set of candidate causes enumerated by experts, especially in difficult cases where the number of related cases is small and the vocabulary used differs. In the future, it will be necessary to develop inference methods tailored to these maintenance records, build a user interface, and carry out validation on larger and more diverse systems. Additionally, this approach leverages the understanding and knowledge of the target in the design phase to support knowledge accumulation and problem solving during the maintenance phase, and it is expected to become a foundation for knowledge sharing across the entire engineering chain in the future. 

---
# DRIFT: Decompose, Retrieve, Illustrate, then Formalize Theorems 

**Authors**: Meiru Zhang, Philipp Borchert, Milan Gritta, Gerasimos Lampouras  

**Link**: [PDF](https://arxiv.org/pdf/2510.10815)  

**Abstract**: Automating the formalization of mathematical statements for theorem proving remains a major challenge for Large Language Models (LLMs). LLMs struggle to identify and utilize the prerequisite mathematical knowledge and its corresponding formal representation in languages like Lean. Current retrieval-augmented autoformalization methods query external libraries using the informal statement directly, but overlook a fundamental limitation: informal mathematical statements are often complex and offer limited context on the underlying math concepts. To address this, we introduce DRIFT, a novel framework that enables LLMs to decompose informal mathematical statements into smaller, more tractable ''sub-components''. This facilitates targeted retrieval of premises from mathematical libraries such as Mathlib. Additionally, DRIFT retrieves illustrative theorems to help models use premises more effectively in formalization tasks. We evaluate DRIFT across diverse benchmarks (ProofNet, ConNF, and MiniF2F-test) and find that it consistently improves premise retrieval, nearly doubling the F1 score compared to the DPR baseline on ProofNet. Notably, DRIFT demonstrates strong performance on the out-of-distribution ConNF benchmark, with BEq+@10 improvements of 37.14% and 42.25% using GPT-4.1 and DeepSeek-V3.1, respectively. Our analysis shows that retrieval effectiveness in mathematical autoformalization depends heavily on model-specific knowledge boundaries, highlighting the need for adaptive retrieval strategies aligned with each model's capabilities. 

---
# Is Implicit Knowledge Enough for LLMs? A RAG Approach for Tree-based Structures 

**Authors**: Mihir Gupte, Paolo Giusto, Ramesh S  

**Link**: [PDF](https://arxiv.org/pdf/2510.10806)  

**Abstract**: Large Language Models (LLMs) are adept at generating responses based on information within their context. While this ability is useful for interacting with structured data like code files, another popular method, Retrieval-Augmented Generation (RAG), retrieves relevant documents to augment the model's in-context learning. However, it is not well-explored how to best represent this retrieved knowledge for generating responses on structured data, particularly hierarchical structures like trees. In this work, we propose a novel bottom-up method to linearize knowledge from tree-like structures (like a GitHub repository) by generating implicit, aggregated summaries at each hierarchical level. This approach enables the knowledge to be stored in a knowledge base and used directly with RAG. We then compare our method to using RAG on raw, unstructured code, evaluating the accuracy and quality of the generated responses. Our results show that while response quality is comparable across both methods, our approach generates over 68% fewer documents in the retriever, a significant gain in efficiency. This finding suggests that leveraging implicit, linearized knowledge may be a highly effective and scalable strategy for handling complex, hierarchical data structures. 

---
# Hierarchical LoRA MoE for Efficient CTR Model Scaling 

**Authors**: Zhichen Zeng, Mengyue Hang, Xiaolong Liu, Xiaoyi Liu, Xiao Lin, Ruizhong Qiu, Tianxin Wei, Zhining Liu, Siyang Yuan, Chaofei Yang, Yiqun Liu, Hang Yin, Jiyan Yang, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2510.10432)  

**Abstract**: Deep models have driven significant advances in click-through rate (CTR) prediction. While vertical scaling via layer stacking improves model expressiveness, the layer-by-layer sequential computation poses challenges to efficient scaling. Conversely, horizontal scaling through Mixture of Experts (MoE) achieves efficient scaling by activating a small subset of experts in parallel, but flat MoE layers may struggle to capture the hierarchical structure inherent in recommendation tasks. To push the Return-On-Investment (ROI) boundary, we explore the complementary strengths of both directions and propose HiLoMoE, a hierarchical LoRA MoE framework that enables holistic scaling in a parameter-efficient manner. Specifically, HiLoMoE employs lightweight rank-1 experts for parameter-efficient horizontal scaling, and stacks multiple MoE layers with hierarchical routing to enable combinatorially diverse expert compositions. Unlike conventional stacking, HiLoMoE routes based on prior layer scores rather than outputs, allowing all layers to execute in parallel. A principled three-stage training framework ensures stable optimization and expert diversity. Experiments on four public datasets show that HiLoMoE achieving better performance-efficiency tradeoff, achieving an average AUC improvement of 0.20\% in AUC and 18.5\% reduction in FLOPs compared to the non-MoE baseline. 

---
# ImCoref-CeS: An Improved Lightweight Pipeline for Coreference Resolution with LLM-based Checker-Splitter Refinement 

**Authors**: Kangyang Luo, Yuzhuo Bai, Shuzheng Si, Cheng Gao, Zhitong Wang, Yingli Shen, Wenhao Li, Zhu Liu, Yufeng Han, Jiayi Wu, Cunliang Kong, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.10241)  

**Abstract**: Coreference Resolution (CR) is a critical task in Natural Language Processing (NLP). Current research faces a key dilemma: whether to further explore the potential of supervised neural methods based on small language models, whose detect-then-cluster pipeline still delivers top performance, or embrace the powerful capabilities of Large Language Models (LLMs). However, effectively combining their strengths remains underexplored. To this end, we propose \textbf{ImCoref-CeS}, a novel framework that integrates an enhanced supervised model with LLM-based reasoning. First, we present an improved CR method (\textbf{ImCoref}) to push the performance boundaries of the supervised neural method by introducing a lightweight bridging module to enhance long-text encoding capability, devising a biaffine scorer to comprehensively capture positional information, and invoking a hybrid mention regularization to improve training efficiency. Importantly, we employ an LLM acting as a multi-role Checker-Splitter agent to validate candidate mentions (filtering out invalid ones) and coreference results (splitting erroneous clusters) predicted by ImCoref. Extensive experiments demonstrate the effectiveness of ImCoref-CeS, which achieves superior performance compared to existing state-of-the-art (SOTA) methods. 

---
# Text2Token: Unsupervised Text Representation Learning with Token Target Prediction 

**Authors**: Ruize An, Richong Zhang, Zhijie Nie, Zhanyu Wu, Yanzhao Zhang, Dingkun Long  

**Link**: [PDF](https://arxiv.org/pdf/2510.10224)  

**Abstract**: Unsupervised text representation learning (TRL) is a fundamental task in natural language processing, which is beneficial for improving search and recommendations with the web's unlabeled texts. A recent empirical study finds that the high-quality representation aligns with the key token of the input text, uncovering the potential connection between representation space and vocabulary space. Inspired by the findings, we revisit the generative tasks and develop an unsupervised generative framework for TRL, Text2Token. The framework is based on the token target prediction task, utilizing carefully constructed target token distribution as supervisory signals. To construct the high-quality target token distribution, we analyze the token-alignment properties with advanced embedders and identify two essential categories of key tokens: (1) the meaningful tokens in the text and (2) semantically derived tokens beyond the text. Based on these insights, we propose two methods -- data-driven and model-derived -- to construct synthetic token targets from data or the LLM backbone. Experiments on the MTEB v2 benchmark demonstrate that Text2Token achieves performance competitive with the state-of-the-art embedder with unsupervised contrastive learning, LLM2Vec. Our analysis further shows that vocabulary and representation spaces optimize together and toward the optimum solution during training, providing new ideas and insights for future work. 

---
# Beyond the limitation of a single query: Train your LLM for query expansion with Reinforcement Learning 

**Authors**: Shu Zhao, Tan Yu, Anbang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10009)  

**Abstract**: Reasoning-augmented search agents, such as Search-R1, are trained to reason, search, and generate the final answer iteratively. Nevertheless, due to their limited capabilities in reasoning and search, their performance on multi-hop QA benchmarks remains far from satisfactory. To handle complex or compound queries, we train an LLM-based search agent with the native capability of query expansion through reinforcement learning. In each turn, our search agent proposes several query variants, which are searched simultaneously to cover more relevant information. Meanwhile, given limited post-training data and computing resources, it is very challenging for a search agent to master multiple tasks, including query generation, retrieved information understanding, and answer generation. Therefore, we propose incorporating a pre-trained squeezer model that helps the search agent understand the retrieved documents, allowing the search agent to focus on query generation for high retrieval recall. With the assistance of the squeezer model, we discover that even a small-scale 3B LLM can demonstrate a strong capability of query expansion and achieve state-of-the-art accuracy on the multi-hop QA benchmarks. To be specific, our experiments across seven question-answering benchmarks demonstrate that our method, named ExpandSearch, achieves an average improvement of 4.4% compared to state-of-the-art baselines, with strong gains on multi-hop reasoning tasks requiring diverse evidence aggregation. 

---
# Stop DDoS Attacking the Research Community with AI-Generated Survey Papers 

**Authors**: Jianghao Lin, Rong Shan, Jiachen Zhu, Yunjia Xi, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09686)  

**Abstract**: Survey papers are foundational to the scholarly progress of research communities, offering structured overviews that guide both novices and experts across disciplines. However, the recent surge of AI-generated surveys, especially enabled by large language models (LLMs), has transformed this traditionally labor-intensive genre into a low-effort, high-volume output. While such automation lowers entry barriers, it also introduces a critical threat: the phenomenon we term the "survey paper DDoS attack" to the research community. This refers to the unchecked proliferation of superficially comprehensive but often redundant, low-quality, or even hallucinated survey manuscripts, which floods preprint platforms, overwhelms researchers, and erodes trust in the scientific record. In this position paper, we argue that we must stop uploading massive amounts of AI-generated survey papers (i.e., survey paper DDoS attack) to the research community, by instituting strong norms for AI-assisted review writing. We call for restoring expert oversight and transparency in AI usage and, moreover, developing new infrastructures such as Dynamic Live Surveys, community-maintained, version-controlled repositories that blend automated updates with human curation. Through quantitative trend analysis, quality audits, and cultural impact discussion, we show that safeguarding the integrity of surveys is no longer optional but imperative to the research community. 

---
# Semantic-Cohesive Knowledge Distillation for Deep Cross-modal Hashing 

**Authors**: Changchang Sun, Vickie Chen, Yan Yan  

**Link**: [PDF](https://arxiv.org/pdf/2510.09664)  

**Abstract**: Recently, deep supervised cross-modal hashing methods have achieve compelling success by learning semantic information in a self-supervised way. However, they still suffer from the key limitation that the multi-label semantic extraction process fail to explicitly interact with raw multimodal data, making the learned representation-level semantic information not compatible with the heterogeneous multimodal data and hindering the performance of bridging modality gap. To address this limitation, in this paper, we propose a novel semantic cohesive knowledge distillation scheme for deep cross-modal hashing, dubbed as SODA. Specifically, the multi-label information is introduced as a new textual modality and reformulated as a set of ground-truth label prompt, depicting the semantics presented in the image like the text modality. Then, a cross-modal teacher network is devised to effectively distill cross-modal semantic characteristics between image and label modalities and thus learn a well-mapped Hamming space for image modality. In a sense, such Hamming space can be regarded as a kind of prior knowledge to guide the learning of cross-modal student network and comprehensively preserve the semantic similarities between image and text modality. Extensive experiments on two benchmark datasets demonstrate the superiority of our model over the state-of-the-art methods. 

---
