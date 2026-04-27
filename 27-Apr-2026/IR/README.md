# Aligning Dense Retrievers with LLM Utility via DistillationAligning Dense Retrievers with LLM Utility via Distillation 

**Authors**: Rajinder Sandhu, Di Mu, Cheng Chang, Md Shahriar Tasjid, Himanshu Rai, Maksims Volkovs, Ga Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22722)  

**Abstract**: Dense vector retrieval is the practical backbone of Retrieval- Augmented Generation (RAG), but similarity search can suffer from precision limitations. Conversely, utility-based approaches leveraging LLM re-ranking often achieve superior performance but are computationally prohibitive and prone to noise inherent in perplexity estimation. We propose Utility-Aligned Embeddings (UAE), a framework designed to merge these advantages into a practical, high-performance retrieval method. We formulate retrieval as a distribution matching problem, training a bi-encoder to imitate a utility distribution derived from perplexity reduction using a Utility-Modulated InfoNCE objective. This approach injects graded utility signals directly into the embedding space without requiring test-time LLM inference. On the QASPER benchmark, UAE improves retrieval Recall@1 by 30.59%, MAP by 30.16% and Token F1 by 17.3% over the strong semantic baseline BGE-Base. Crucially, UAE is over 180x faster than the efficient LLM re-ranking methods preserving competitive performance, demonstrating that aligning retrieval with generative utility yields reliable contexts at scale. 

---
# Can QPP Choose the Right Query Variant? Evaluating Query Variant Selection for RAG Pipelines 

**Authors**: Negar Arabzadeh, Andrew Drozdov, Michael Bendersky, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2604.22661)  

**Abstract**: Large Language Models (LLMs) have made query reformulation ubiquitous in modern retrieval and Retrieval-Augmented Generation (RAG) pipelines, enabling the generation of multiple semantically equivalent query variants. However, executing the full pipeline for every reformulation is computationally expensive, motivating selective execution: can we identify the best query variant before incurring downstream retrieval and generation costs? We investigate Query Performance Prediction (QPP) as a mechanism for variant selection across ad-hoc retrieval and end-to-end RAG. Unlike traditional QPP, which estimates query difficulty across topics, we study intra-topic discrimination - selecting the optimal reformulation among competing variants of the same information need. Through large-scale experiments on TREC-RAG using both sparse and dense retrievers, we evaluate pre- and post-retrieval predictors under correlation- and decision-based metrics. Our results reveal a systematic divergence between retrieval and generation objectives: variants that maximize ranking metrics such as nDCG often fail to produce the best generated answers, exposing a "utility gap" between retrieval relevance and generation fidelity. Nevertheless, QPP can reliably identify variants that improve end-to-end quality over the original query. Notably, lightweight pre-retrieval predictors frequently match or outperform more expensive post-retrieval methods, offering a latency-efficient approach to robust RAG. 

---
# ASPIRE: Make Spectral Graph Collaborative Filtering Great Again via Adaptive Filter Learning 

**Authors**: Yunhang He, Cong Xu, Zhangchi Zhu, Hongzhi Yin, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22549)  

**Abstract**: Graph filter design is central to spectral collaborative filtering, yet most existing methods rely on manually tuned hyperparameters rather than fully learnable filters. We show that this challenge stems from a bias in traditional recommendation objectives, which induces a spectral phenomenon termed low-frequency explosion, thereby fundamentally hindering the effective learning of graph filters. To overcome this limitation, we propose a novel adaptive spectral graph collaborative filtering framework (ASPIRE) based on a bi-level optimization objective. Guided by our theoretical analysis, we disentangle the filter learning objective, which in turn leads to excellent recommendation performance, spectral adaptivity, and training stability in practice. Extensive experiments show our learned filters match the performance of carefully engineered task-specific designs. Furthermore, ASPIRE is equally effective in LLM-powered collaborative filtering. Our findings demonstrate that graph filter learning is viable and generalizable, paving the way for more expressive graph neural networks in collaborative filtering. 

---
# Objective Shaping with Hard Negatives: Windowed Partial AUC Optimization for RL-based LLM Recommenders 

**Authors**: Wentao Shi, Qifan Wang, Chen Chen, Fei Liu, Dongfang Liu, Xu Liu, Wanli Ma, Junfeng Pan, Linhong Zhu, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.22504)  

**Abstract**: Reinforcement learning (RL) effectively optimizes Large Language Model (LLM)-based recommenders by contrasting positive and negative items. Empirically, training with beam-search negatives consistently outperforms random negatives, yet the mechanism is not well understood. We address this gap by analyzing the induced optimization objective and show that: (i) Under binary reward feedback, optimizing LLM recommenders with Group Relative Policy Optimization (GRPO) is theoretically equivalent to maximizing the Area Under the ROC Curve (AUC), which is often misaligned with Top-$K$ recommendation; and (ii) Replacing random negatives with beam-search negatives reshapes the objective toward partial AUC, improving alignment with Top-$K$ metrics. Motivated by this perspective, we introduce Windowed Partial AUC (WPAUC), which constrains the false positive rate (FPR) to a window [$\alpha,\alpha+d$] to more directly align with Top-$K$ metrics. We further propose an efficient Threshold-Adjusted Windowed reweighting (TAWin) RL method for its optimization, enabling explicit control over the targeted Top-$K$ performance. Experiments on four real-world datasets validate the theory and deliver consistent state-of-the-art performance. 

---
# Rethinking Semantic Collaborative Integration: Why Alignment Is Not Enough 

**Authors**: Maolin Wang, Dongze Wu, Jianing Zhou, Hongyu Chen, Beining Bao, Yu Jiang, Chenbin Zhang, Chang Wang, Jian Liu, Lei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2604.22195)  

**Abstract**: Large language models (LLMs) have become an important semantic infrastructure for modern recommender systems. A prevailing paradigm integrates LLM-derived semantic embeddings with collaborative representations via representation alignment, implicitly assuming that the two views encode a shared latent entity and that stronger alignment yields better results. We formalize this assumption as the global low-complexity alignment hypothesis and argue that it is stronger than necessary and often structurally mismatched with real-world recommendation settings. We propose a complementary perspective in which semantic and collaborative representations are treated as partially shared yet fundamentally heterogeneous views, each containing both shared and view-specific factors. Under this shared-plus-private latent structure, enforcing global geometric alignment may distort local structure, suppress view-specific signals, and reduce informational diversity. To support this perspective, we develop complementarity-aware diagnostics that quantify overlap, unique-hit contribution, and theoretical fusion upper bounds. Empirical analyses on sparse recommendation benchmarks reveal low item-level agreement between semantic and collaborative views and substantial oracle fusion gains, indicating strong complementarity. Furthermore, controlled alignment probes show that low-capacity mappings capture only shared components and fail to recover full collaborative geometry, especially under distribution shift. These findings suggest that alignment should not be treated as the default integration principle. We advocate a shift from alignment-centric modeling to complementarity fusion-centric, complementarity-aware design, where shared factors are selectively integrated while private signals are preserved. This reframing provides a principled foundation for the next generation of LLM-enhanced recommender systems. 

---
# ResRank: Unifying Retrieval and Listwise Reranking via End-to-End Joint Training with Residual Passage Compression 

**Authors**: Xiaojie Ke, Shuai Zhang, Liansheng Sun, Yongjin Wang, Hengjun Jiang, Xiangkun Liu, Cunxin Gu, Jian Xu, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22180)  

**Abstract**: Large language model (LLM) based listwise reranking has emerged as the dominant paradigm for achieving state-of-the-art ranking effectiveness in information retrieval. However, its reliance on feeding full passage texts into the LLM introduces two critical bottlenecks: the "lost in the middle" phenomenon degrades ranking quality as input length grows, and the inference latency scales super-linearly with sequence length, rendering it impractical for industrial deployment. In this paper, we present ResRank, a unified retrieval-reranking framework that fundamentally addresses both challenges. Inspired by multimodal LLMs that project visual inputs into compact token representations, ResRank employs an Encoder-LLM to compress each candidate passage into a single embedding, which is then fed alongside the query text into a Reranker-LLM for listwise ranking. To alleviate the misalignment between the compressed representation space and the ranking space, we introduce a residual connection structure that combines encoder embeddings with contextualized hidden states from the reranker. Furthermore, we replace the conventional autoregressive decoding with a one-step cosine-similarity-based scoring mechanism, eliminating the generation bottleneck entirely. ResRank is trained through a carefully designed dual-stage, multi-task, end-to-end joint optimization strategy that simultaneously trains the encoder and reranker, achieving learning objective alignment between retrieval and reranking while substantially reducing training complexity. Extensive experiments on TREC Deep Learning and eight BEIR benchmark datasets demonstrate that ResRank achieves competitive or superior ranking effectiveness compared to existing approaches while requiring zero generated tokens and processing only one token per passage, yielding a fundamentally better balance between effectiveness and efficiency. 

---
# AgentSearchBench: A Benchmark for AI Agent Search in the Wild 

**Authors**: Bin Wu, Arastun Mammadli, Xiaoyu Zhang, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.22436)  

**Abstract**: The rapid growth of AI agent ecosystems is transforming how complex tasks are delegated and executed, creating a new challenge of identifying suitable agents for a given task. Unlike traditional tools, agent capabilities are often compositional and execution-dependent, making them difficult to assess from textual descriptions alone. However, existing research and benchmarks typically assume well-specified functionalities, controlled candidate pools, or only executable task queries, leaving realistic agent search scenarios insufficiently studied. We introduce AgentSearchBench, a large-scale benchmark for agent search in the wild, built from nearly 10,000 real-world agents across multiple providers. The benchmark formalizes agent search as retrieval and reranking problems under both executable task queries and high-level task descriptions, and evaluates relevance using execution-grounded performance signals. Experiments reveal a consistent gap between semantic similarity and actual agent performance, exposing the limitations of description-based retrieval and reranking methods. We further show that lightweight behavioral signals, including execution-aware probing, can substantially improve ranking quality, highlighting the importance of incorporating execution signals into agent discovery. Our code is available at this https URL. 

---
# Sharpness-Aware Poisoning: Enhancing Transferability of Injective Attacks on Recommender Systems 

**Authors**: Junsong Xie, Yonghui Yang, Pengyang Shao, Le Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22170)  

**Abstract**: Recommender Systems~(RS) have been shown to be vulnerable to injective attacks, where attackers inject limited fake user profiles to promote the exposure of target items to real users for unethical gains (e.g., economic or political advantages). Since attackers typically lack knowledge of the victim model deployed in the target RS, existing methods resort to using a fixed surrogate model to mimic the potential victim model. Despite considerable progress, we argue that the assumption that \textit{poisoned data generated for the surrogate model can be used to attack other victim models} is wishful. When there are significant structural discrepancies between the surrogate and victim models, the attack transferability inevitably suffers. Intuitively, if we can identify the worst-case victim model and iteratively optimize the poisoning effect specifically against it, then the generated poisoned data would be better transferred to other victim models. However, exactly identifying the worst-case victim model during the attack process is challenging due to the large space of victim models. To this end, in this work, we propose a novel attack method called Sharpness-Aware Poisoning (\textit{SharpAP}). Specifically, it employs the sharpness-aware minimization principle to seek the approximately worst-case victim model and optimizes the poisoned data specifically for this worst-case model. The poisoning attack with SharpAP is formulated as a min-max-min tri-level optimization problem. By integrating SharpAP into the iterative process for attacks, our method can generate more robust poisoned data which is less sensitive to the shift of model structure, mitigating the overfitting to the surrogate model. Comprehensive experimental comparisons on three real-world datasets demonstrate that \name~can significantly enhance the attack transferability. 

---
# ReCast: Recasting Learning Signals for Reinforcement Learning in Generative Recommendation 

**Authors**: Peiyan Zhang, Hanmo Liu, Chengxuan Tong, Yuxia Wu, Wei Guo, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22169)  

**Abstract**: Generic group-based RL assumes that sampled rollout groups are already usable learning signals. We show that this assumption breaks down in sparse-hit generative recommendation, where many sampled groups never become learnable at all. We propose ReCast, a repair-then-contrast learning-signal framework that first restores minimal learnability for all-zero groups and then replaces full-group reward normalization with a boundary-focused contrastive update on the strongest positive and the hardest negative. ReCast leaves the outer RL framework unchanged, modifies only within-group signal construction, and partially decouples rollout search width from actor-side update width. Across multiple generative recommendation tasks, ReCast consistently outperforms OpenOneRec-RL, achieving up to 36.6% relative improvement in Pass@1. Its matched-budget advantage is substantially larger: ReCast reaches the baseline's target performance with only 4.1% of the rollout budget, and this advantage widens with model scale. The same design also yields direct system-level gains, reducing actor-side update time by 16.60x, lowering peak allocated memory by 16.5%, and improving actor MFU by 14.2%. Mechanism analysis shows that ReCast mitigates the persistent all-zero / single-hit regime, restores learnability when natural positives are scarce, and converts otherwise wasted rollout budget into more stable policy updates. These results suggest that, for generative recommendation, the decisive RL problem is not only how to assign rewards, but how to construct learnable optimization events from sparse, structured supervision. 

---
# Implementation and Privacy Guarantees for Scalable Keyword Search on SOLID-based Decentralized Data with Granular Visibility Constraints 

**Authors**: Mohamed Ragab, Faria Ferooz, Mohammad Bahrani, Helen Oliver, Thanassis Tiropanis, Alexandra Poulovassilis, Adriane Chapman, George Roussos  

**Link**: [PDF](https://arxiv.org/pdf/2604.22100)  

**Abstract**: In decentralized personal data ecosystems grounded in architectures such as Solid, users retain sovereignty over their data via personal online data stores (pods), hosted on Solid-compliant server infrastructures. In such environments, data remains under the control of pod owners, which complicates search due to distribution across numerous pods and user-specific access constraints. ESPRESSO is a decentralized framework for scalable keyword-based search across distributed Solid pods under user-defined visibility policies. It addresses key challenges of decentralized search by constructing WebID-scoped indexes within pods and employing privacy-aware metadata to enable efficient source selection and ranking across servers. This paper further introduces a formal threat model for ESPRESSO, analysing the security and privacy risks associated with the generation, aggregation, and use of indexes and metadata. These risks include unintended metadata leakage and the potential for adversaries to infer sensitive information about data that resides within personal data stores. The analysis identifies key design principles that limit metadata exposure while mitigating unauthorized inference. The proposed threat model provides a foundation for evaluating privacy-preserving decentralized search and informs the design of systems with stronger privacy guarantees. 

---
# Analyzing Shapley Additive Explanations to Understand Anomaly Detection Algorithm Behaviors and Their Complementarity 

**Authors**: Jordan Levy, Paul Saves, Moncef Garouani, Nicolas Verstaevel, Benoit Gaudou  

**Link**: [PDF](https://arxiv.org/pdf/2602.00208)  

**Abstract**: Unsupervised anomaly detection is a challenging problem due to the diversity of data distributions and the lack of labels. Ensemble methods are often adopted to mitigate these challenges by combining multiple detectors, which can reduce individual biases and increase robustness. Yet building an ensemble that is genuinely complementary remains challenging, since many detectors rely on similar decision cues and end up producing redundant anomaly scores. As a result, the potential of ensemble learning is often limited by the difficulty of identifying models that truly capture different types of irregularities. To address this, we propose a methodology for characterizing anomaly detectors through their decision mechanisms. Using SHapley Additive exPlanations, we quantify how each model attributes importance to input features, and we use these attribution profiles to measure similarity between detectors. We show that detectors with similar explanations tend to produce correlated anomaly scores and identify largely overlapping anomalies. Conversely, explanation divergence reliably indicates complementary detection behavior. Our results demonstrate that explanation-driven metrics offer a different criterion than raw outputs for selecting models in an ensemble. However, we also demonstrate that diversity alone is insufficient; high individual model performance remains a prerequisite for effective ensembles. By explicitly targeting explanation diversity while maintaining model quality, we are able to construct ensembles that are more diverse, more complementary, and ultimately more effective for unsupervised anomaly detection. 

---
