# ECLASS-Augmented Semantic Product Search for Electronic Components 

**Authors**: Nico Baumgart, Markus Lange-Hegermann, Jan Henze  

**Link**: [PDF](https://arxiv.org/pdf/2604.19664)  

**Abstract**: Efficient semantic access to industrial product data is a key enabler for factory automation and emerging LLM-based agent workflows, where both human engineers and autonomous agents must identify suitable components from highly structured catalogs. However, the vocabulary mismatch between natural-language queries and attribute-centric product descriptions limits the effectiveness of traditional retrieval approaches, e.g., BM25. In this work, we present a systematic evaluation of LLM-assisted dense retrieval for semantic product search on industrial electronic components, and investigate the integration of hierarchical semantics from the ECLASS standard into embedding-based retrieval. Our results show that dense retrieval combined with re-ranking substantially outperforms classical lexical methods and foundation model web-search baselines. In particular, the proposed approach achieves a Hit_Rate@5 of 94.3 %, compared to 31.4 % for BM25 on expert queries, while also exceeding foundation model baselines in both effectiveness and efficiency. Furthermore, augmenting product representations with ECLASS semantics yields consistent performance gains across configurations, demonstrating that standardized hierarchical metadata provides a crucial semantic bridge between user intent and sparse product descriptions. 

---
# From Top-1 to Top-K: A Reproducibility Study and Benchmarking of Counterfactual Explanations for Recommender Systems 

**Authors**: Quang-Huy Nguyen, Thanh-Hai Nguyen, Khac-Manh Thai, Duc-Hoang Pham, Huy-Son Nguyen, Cam-Van Thi Nguyen, Masoud Mansoury, Duc-Trong Le, Hoang-Quynh Le  

**Link**: [PDF](https://arxiv.org/pdf/2604.19663)  

**Abstract**: Counterfactual explanations (CEs) provide an intuitive way to understand recommender systems by identifying minimal modifications to user-item interactions that alter recommendation outcomes. Existing CE methods for recommender systems, however, have been evaluated under heterogeneous protocols, using different datasets, recommenders, metrics, and even explanation formats, which hampers reproducibility and fair comparison. Our paper systematically reproduces, re-implement, and re-evaluate eleven state-of-the-art CE methods for recommender systems, covering both native explainers (e.g., LIME-RS, SHAP, PRINCE, ACCENT, LXR, GREASE) and specific graph-based explainers originally proposed for GNNs. Here, a unified benchmarking framework is proposed to assess explainers along three dimensions: explanation format (implicit vs. explicit), evaluation level (item-level vs. list-level), and perturbation scope (user interaction vectors vs. user-item interaction graphs). Our evaluation protocol includes effectiveness, sparsity, and computational complexity metrics, and extends existing item-level assessments to top-K list-level explanations. Through extensive experiments on three real-world datasets and six representative recommender models, we analyze how well previously reported strengths of CE methods generalize across diverse setups. We observe that the trade-off between effectiveness and sparsity depends strongly on the specific method and evaluation setting, particularly under the explicit format; in addition, explainer performance remains largely consistent across item level and list level evaluations, and several graph-based explainers exhibit notable scalability limitations on large recommender graphs. Our results refine and challenge earlier conclusions about the robustness and practicality of CE generation methods in recommender systems: this https URL. 

---
# Diagnosable ColBERT: Debugging Late-Interaction Retrieval Models Using a Learned Latent Space as Reference 

**Authors**: François Remy  

**Link**: [PDF](https://arxiv.org/pdf/2604.19566)  

**Abstract**: Reliable biomedical and clinical retrieval requires more than strong ranking performance: it requires a practical way to find systematic model failures and curate the training evidence needed to correct them. Late-interaction models such as ColBERT provide a first solution thanks to the interpretable token-level interaction scores they expose between document and query tokens. Yet this interpretability is shallow: it explains a particular document--query pairwise score, but does not reveal whether the model has learned a clinical concept in a stable, reusable, and context-sensitive way across diverse expressions. As a result, these scores provide limited support for diagnosing misunderstandings, identifying irreasonably distant biomedical concepts, or deciding what additional data or feedback is needed to address this. In this short position paper, we propose Diagnosable ColBERT, a framework that aligns ColBERT token embeddings to a reference latent space grounded in clinical knowledge and expert-provided conceptual similarity constraints. This alignment turns document encodings into inspectable evidence of what the model appears to understand, enabling more direct error diagnosis and more principled data curation without relying on large batteries of diagnostic queries. 

---
# LoopCTR: Unlocking the Loop Scaling Power for Click-Through Rate Prediction 

**Authors**: Jiakai Tang, Runfeng Zhang, Weiqiu Wang, Yifei Liu, Chuan Wang, Xu Chen, Yeqiu Yang, Jian Wu, Yuning Jiang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.19550)  

**Abstract**: Scaling Transformer-based click-through rate (CTR) models by stacking more parameters brings growing computational and storage overhead, creating a widening gap between scaling ambitions and the stringent industrial deployment constraints. We propose LoopCTR, which introduces a loop scaling paradigm that increases training-time computation through recursive reuse of shared model layers, decoupling computation from parameter growth. LoopCTR adopts a sandwich architecture enhanced with Hyper-Connected Residuals and Mixture-of-Experts, and employs process supervision at every loop depth to encode multi-loop benefits into the shared parameters. This enables a train-multi-loop, infer-zero-loop strategy where a single forward pass without any loop already outperforms all baselines. Experiments on three public benchmarks and one industrial dataset demonstrate state-of-the-art performance. Oracle analysis further reveals 0.02--0.04 AUC of untapped headroom, with models trained with fewer loops exhibiting higher oracle ceilings, pointing to a promising frontier for adaptive inference. 

---
# Enhancing Unsupervised Keyword Extraction in Academic Papers through Integrating Highlights with Abstract 

**Authors**: Yi Xiang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19505)  

**Abstract**: Automatic keyword extraction from academic papers is a key area of interest in natural language processing and information retrieval. Although previous research has mainly focused on utilizing abstract and references for keyword extraction, this paper focuses on the highlights section - a summary describing the key findings and contributions, offering readers a quick overview of the research. Our observations indicate that highlights contain valuable keyword information that can effectively complement the abstract. To investigate the impact of incorporating highlights into unsupervised keyword extraction, we evaluate three input scenarios: using only the abstract, the highlights, and a combination of both. Experiments conducted with four unsupervised models on Computer Science (CS), Library and Information Science (LIS) datasets reveal that integrating the abstract with highlights significantly improves extraction performance. Furthermore, we examine the differences in keyword coverage and content between abstract and highlights, exploring how these variations influence extraction outcomes. The data and code are available at this https URL. 

---
# CAST: Modeling Semantic-Level Transitions for Complementary-Aware Sequential Recommendation 

**Authors**: Qian Zhang, Lech Szymanski, Haibo Zhang, Jeremiah D. Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.19414)  

**Abstract**: Sequential Recommendation (SR) aims to predict the next interaction of a user based on their behavior sequence, where complementary relations often provide essential signals for predicting the next item. However, mainstream models relying on sparse co-purchase statistics often mistake spurious correlations (e.g., due to popularity bias) for true complementary relations. Identifying true complementary relations requires capturing the fine-grained item semantics (e.g., specifications) that simple cooccurrence statistics would be unable to model. While recent semantics-based methods utilize discrete semantic codes to represent items, they typically aggregate semantic codes into coarse item representations. This aggregation process blurs specific semantic details required to identify complementarity. To address these critical limitations and effectively leverage semantics for capturing reliable complementary relations, we propose a Complementary-Aware Semantic Transition (CAST) framework that introduces a new modeling paradigm built upon semantic-level transitions. Specifically, a semantic-level transition module is designed to model dynamic transitions directly in the discrete semantic code space, effectively capturing fine-grained semantic dependencies often lost in aggregated item representations. Then, a complementary prior injection module is designed to incorporate LLM-verified complementary priors into the attention mechanism, thereby prioritizing complementary patterns over co-occurrence statistics. Experiments on multiple e-commerce datasets demonstrate that CAST consistently outperforms the state-of-the-art approaches, achieving up to 17.6% Recall and 16.0% NDCG gains with 65x training acceleration. This validates its effectiveness and efficiency in uncovering latent item complementarity beyond statistics. The code will be released upon acceptance. 

---
# CS3: Efficient Online Capability Synergy for Two-Tower Recommendation 

**Authors**: Lixiang Wang, Shaoyun Shi, Peng Wang, Wenjin Wu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19269)  

**Abstract**: To balance effectiveness and efficiency in recommender systems, multi-stage pipelines commonly use lightweight two-tower models for large-scale candidate retrieval. However, the isolated two-tower architecture restricts representation capacity, embedding-space alignment, and cross-feature interactions. Existing solutions such as late interaction and knowledge distillation can mitigate these issues, but often increase latency or are difficult to deploy in online learning settings. We propose Capability Synergy (CS3), an efficient online framework that strengthens two-tower retrievers while preserving real-time constraints. CS3 introduces three mechanisms: (1) Cycle-Adaptive Structure for self-revision via adaptive feature denoising within each tower; (2) Cross-Tower Synchronization to improve alignment through lightweight mutual awareness between towers; and (3) Cascade-Model Sharing to enhance cross-stage consistency by reusing knowledge from downstream models. CS3 is plug-and-play with diverse two-tower backbones and compatible with online learning. Experiments on three public datasets show consistent gains over strong baselines, and deployment in a largescale advertising system yields up to 8.36% revenue improvement across three scenarios while maintaining ms-level latency. 

---
# GraphRAG-IRL: Personalized Recommendation with Graph-Grounded Inverse Reinforcement Learning and LLM Re-ranking 

**Authors**: Siqi Liang, Xiawei Wang, Yudi Zhang, Jiaying Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.19128)  

**Abstract**: Personalized recommendation requires models that capture sequential user preferences while remaining robust to sparse feedback and semantic ambiguity. Recent work has explored large language models (LLMs) as recommenders and re-rankers, but pure prompt-based ranking often suffers from poor calibration, sensitivity to candidate ordering, and popularity bias. These limitations make LLMs useful semantic reasoners, but unreliable as standalone ranking engines.
We present \textbf{GraphRAG-IRL}, a hybrid recommendation framework that combines graph-grounded feature construction, inverse reinforcement learning (IRL), and persona-guided LLM re-ranking. Our method constructs a heterogeneous knowledge graph over items, categories, and concepts, retrieves both individual and community preference context, and uses these signals to train a Maximum Entropy IRL model for calibrated pre-ranking. An LLM is then applied only to a short candidate list, where persona-guided prompts provide complementary semantic judgments that are fused with IRL rankings.
Experiments show that GraphRAG-IRL is a strong standalone recommender: IRL-MLP with GraphRAG improves NDCG@10 by 15.7\% on MovieLens and 16.6\% on KuaiRand over supervised baselines. The results also show that IRL and GraphRAG are superadditive, with the combined gain exceeding the sum of their individual improvements. Persona-guided LLM fusion further improves ranking quality, yielding up to 16.8\% NDCG@10 improvement over the IRL-only baseline on MovieLens ml-1m, while score fusion on KuaiRand provides consistent gains of 4--6\% across LLM providers. 

---
# Think Before Writing: Feature-Level Multi-Objective Optimization for Generative Citation Visibility 

**Authors**: Zikang Liu, Peilan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19113)  

**Abstract**: Generative answer engines expose content through selective citation rather than ranked retrieval, fundamentally altering how visibility is determined. This shift calls for new optimization methods beyond traditional search engine optimization. Existing generative engine optimization (GEO) approaches primarily rely on token-level text rewriting, offering limited interpretability and weak control over the trade-off between citation visibility and content quality. We propose FeatGEO, a feature-level, multi-objective optimization framework that abstracts webpages into interpretable structural, content, and linguistic properties. Instead of directly editing text, FeatGEO optimizes over this feature space and uses a language model to realize feature configurations into natural language, decoupling high-level optimization from surface-level generation. Experiments on GEO-Bench across three generative engines demonstrate that FeatGEO consistently improves citation visibility while maintaining or improving content quality, substantially outperforming token-level baselines. Further analyses show that citation behavior is more strongly influenced by document-level content properties than by isolated lexical edits, and that the learned feature configurations generalize across language models of different scales. 

---
# STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation 

**Authors**: Shuyuan Zhao, Wei Chen, Weijie Zhang, Xinrui Hou, Junfeng Shen, Boyan Shi, Shengnan Guo, Youfang Lin, Huaiyu Wan  

**Link**: [PDF](https://arxiv.org/pdf/2604.19042)  

**Abstract**: Temporal Knowledge Graph (TKG) extrapolation aims to predict future events based on historical facts. Recent studies have attempted to enhance TKG extrapolation by integrating TKG's evolving structural representations and textual event chains into Large Language Models (LLMs). Yet, two main challenges limit these approaches: (1) The loss of essential spatial-temporal information due to shallow alignment between TKG's graph evolving structural representation and the LLM's semantic space, and (2) the progressive dilution of the TKG's evolving structural features during LLM fine-tuning. To address these challenges, we propose the Spatial-Temporal Knowledge Adapter (STK-Adapter), which flexibly integrates the evolving graph encoder and the LLM to facilitate TKG reasoning. In STK-Adapter, a Spatial-Temporal MoE is designed to capture spatial structures and temporal patterns inherent in TKGs. An Event-Aware MoE is employed to model intricate temporal semantics dependencies within event chains. In addition, a Cross-Modality Alignment MoE is proposed to facilitate deep cross-modality alignment by TKG-guided attention experts. Extensive experiments on benchmark datasets demonstrate that STK-Adapter significantly outperforms state-of-the-art methods and exhibits strong generalization capabilities in cross-dataset task. The code is available at this https URL. 

---
# Dual-View Training for Instruction-Following Information Retrieval 

**Authors**: Qingcheng Zeng, Puxuan Yu, Aman Mehta, Fuheng Zhao, Rajhans Samdani  

**Link**: [PDF](https://arxiv.org/pdf/2604.18845)  

**Abstract**: Instruction-following information retrieval (IF-IR) studies retrieval systems that must not only find documents relevant to a query, but also obey explicit user constraints such as required attributes, exclusions, or output preferences. However, most retrievers are trained primarily for semantic relevance and often fail to distinguish documents that match the topic from those that satisfy the instruction. We propose a dual-view data synthesis strategy based on polarity reversal: given a query, a document that is relevant under the instruction, and a hard negative that matches the query but violates the instruction, we prompt an LLM to generate a complementary instruction under which the two documents swap relevance labels. By presenting the same document pair under complementary instructions that invert their relevance labels, the training signal forces the retriever to reconsider the same candidate set through the instruction, rather than relying on fixed topical cues. On a 305M-parameter encoder, our method improves performance on the FollowIR benchmark by 45%, surpassing general-purpose embedding models of comparable or larger scale. Through head-to-head comparisons at matched data budgets, we further show that data diversity and instruction supervision play complementary roles: the former preserves general retrieval quality, while the latter improves instruction sensitivity. These results highlight the value of targeted data synthesis for building retrieval systems that are both broadly capable and instruction-aware. 

---
# Impact of large language models on peer review opinions from a fine-grained perspective: Evidence from top conference proceedings in AI 

**Authors**: Wenqing Wu, Chengzhi Zhang, Yi Zhao, Tong Bao  

**Link**: [PDF](https://arxiv.org/pdf/2604.19578)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), the academic community has faced unprecedented disruptions, particularly in the realm of academic communication. The primary function of peer review is improving the quality of academic manuscripts, such as clarity, originality and other evaluation aspects. Although prior studies suggest that LLMs are beginning to influence peer review, it remains unclear whether they are altering its core evaluative functions. Moreover, the extent to which LLMs affect the linguistic form, evaluative focus, and recommendation-related signals of peer-review reports has yet to be systematically examined. In this study, we examine the changes in peer review reports for academic articles following the emergence of LLMs, emphasizing variations at fine-grained level. Specifically, we investigate linguistic features such as the length and complexity of words and sentences in review comments, while also automatically annotating the evaluation aspects of individual review sentences. We also use a maximum likelihood estimation method, previously established, to identify review reports that potentially have modified or generated by LLMs. Finally, we assess the impact of evaluation aspects mentioned in LLM-assisted review reports on the informativeness of recommendation for paper decision-making. The results indicate that following the emergence of LLMs, peer review texts have become longer and more fluent, with increased emphasis on summaries and surface-level clarity, as well as more standardized linguistic patterns, particularly reviewers with lower confidence score. At the same time, attention to deeper evaluative dimensions, such as originality, replicability, and nuanced critical reasoning, has declined. 

---
# IndiaFinBench: An Evaluation Benchmark for Large Language Model Performance on Indian Financial Regulatory Text 

**Authors**: Rajveer Singh Pall  

**Link**: [PDF](https://arxiv.org/pdf/2604.19298)  

**Abstract**: We introduce IndiaFinBench, to our knowledge the first publicly available evaluation benchmark for assessing large language model (LLM) performance on Indian financial regulatory text. Existing financial NLP benchmarks draw exclusively from Western financial corpora (SEC filings, US earnings reports, and English-language financial news), leaving a significant gap in coverage of non-Western regulatory frameworks. IndiaFinBench addresses this gap with 406 expert-annotated question-answer pairs drawn from 192 documents sourced from the Securities and Exchange Board of India (SEBI) and the Reserve Bank of India (RBI), spanning four task types: regulatory interpretation (174 items), numerical reasoning (92 items), contradiction detection (62 items), and temporal reasoning (78 items). Annotation quality is validated through a model-based secondary pass (kappa=0.918 on contradiction detection) and a 60-item human inter-annotator agreement evaluation (kappa=0.611; 76.7% overall agreement). We evaluate twelve models under zero-shot conditions, with accuracy ranging from 70.4% (Gemma 4 E4B) to 89.7% (Gemini 2.5 Flash). All models substantially outperform a non-specialist human baseline of 60.0%. Numerical reasoning is the most discriminative task, with a 35.9 percentage-point spread across models. Bootstrap significance testing (10,000 resamples) reveals three statistically distinct performance tiers. The dataset, evaluation code, and all model outputs are available at this https URL 

---
# RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora 

**Authors**: Hanjun Cho, Jay-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.19047)  

**Abstract**: Existing QA benchmarks typically assume distinct documents with minimal overlap, yet real-world retrieval-augmented generation (RAG) systems operate on corpora such as financial reports, legal codes, and patents, where information is highly redundant and documents exhibit strong inter-document similarity. This mismatch undermines evaluation validity: retrievers can be unfairly undervalued even when they retrieve documents that provide sufficient evidence, because redundancy across documents is not accounted for in evaluation. On the other hand, retrievers that perform well on standard benchmarks often generalize poorly to real-world corpora with highly similar and redundant documents. We present RARE (Redundancy-Aware Retrieval Evaluation), a framework for constructing realistic benchmarks by (i) decomposing documents into atomic facts to enable precise redundancy tracking and (ii) enhancing LLM-based data generation with CRRF. RAG benchmark data usually requires multiple quality criteria, but LLMs often yield trivial outputs. CRRF scores criteria separately and fuses decisions by rank, improving the reliability of generated data. Applying RARE to Finance, Legal, and Patent corpora, we introduce RedQA, where a strong retriever baseline drops from 66.4% PerfRecall@10 on 4-hop General-Wiki to 5.0-27.9% PerfRecall@10 at 4-hop depth, revealing robustness gaps that current benchmarks fail to capture. RARE enables practitioners to build domain-specific RAG evaluations that faithfully reflect real-world deployment conditions. 

---
# Personalized Benchmarking: Evaluating LLMs by Individual Preferences 

**Authors**: Cristina Garbacea, Heran Wang, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.18943)  

**Abstract**: With the rise in capabilities of large language models (LLMs) and their deployment in real-world tasks, evaluating LLM alignment with human preferences has become an important challenge. Current benchmarks average preferences across all users to compute aggregate ratings, overlooking individual user preferences when establishing model rankings. Since users have varying preferences in different contexts, we call for personalized LLM benchmarks that rank models according to individual needs. We compute personalized model rankings using ELO ratings and Bradley-Terry coefficients for 115 active Chatbot Arena users and analyze how user query characteristics (topics and writing style) relate to LLM ranking variations. We demonstrate that individual rankings of LLM models diverge dramatically from aggregate LLM rankings, with Bradley-Terry correlations averaging only $\rho = 0.04$ (57\% of users show near-zero or negative correlation) and ELO ratings showing moderate correlation ($\rho = 0.43$). Through topic modeling and style analysis, we find users exhibit substantial heterogeneity in topical interests and communication styles, influencing their model preferences. We further show that a compact combination of topic and style features provides a useful feature space for predicting user-specific model rankings. Our results provide strong quantitative evidence that aggregate benchmarks fail to capture individual preferences for most users, and highlight the importance of developing personalized benchmarks that rank LLM models according to individual user preferences. 

---
