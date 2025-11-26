# HHFT: Hierarchical Heterogeneous Feature Transformer for Recommendation Systems 

**Authors**: Liren Yu, Wenming Zhang, Silu Zhou, Zhixuan Zhang, Dan Ou  

**Link**: [PDF](https://arxiv.org/pdf/2511.20235)  

**Abstract**: We propose HHFT (Hierarchical Heterogeneous Feature Transformer), a Transformer-based architecture tailored for industrial CTR prediction. HHFT addresses the limitations of DNN through three key designs: (1) Semantic Feature Partitioning: Grouping heterogeneous features (e.g. user profile, item information, behaviour sequennce) into semantically coherent blocks to preserve domain-specific information; (2) Heterogeneous Transformer Encoder: Adopting block-specific QKV projections and FFNs to avoid semantic confusion between distinct feature types; (3) Hiformer Layer: Capturing high-order interactions across features. Our findings reveal that Transformers significantly outperform DNN baselines, achieving a +0.4% improvement in CTR AUC at scale. We have successfully deployed the model on Taobao's production platform, observing a significant uplift in key business metrics, including a +0.6% increase in Gross Merchandise Value (GMV). 

---
# HKRAG: Holistic Knowledge Retrieval-Augmented Generation Over Visually-Rich Documents 

**Authors**: Anyang Tong, Xiang Niu, ZhiPing Liu, Chang Tian, Yanyan Wei, Zenglin Shi, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20227)  

**Abstract**: Existing multimodal Retrieval-Augmented Generation (RAG) methods for visually rich documents (VRD) are often biased towards retrieving salient knowledge(e.g., prominent text and visual elements), while largely neglecting the critical fine-print knowledge(e.g., small text, contextual details). This limitation leads to incomplete retrieval and compromises the generator's ability to produce accurate and comprehensive answers. To bridge this gap, we propose HKRAG, a new holistic RAG framework designed to explicitly capture and integrate both knowledge types. Our framework features two key components: (1) a Hybrid Masking-based Holistic Retriever that employs explicit masking strategies to separately model salient and fine-print knowledge, ensuring a query-relevant holistic information retrieval; and (2) an Uncertainty-guided Agentic Generator that dynamically assesses the uncertainty of initial answers and actively decides how to integrate the two distinct knowledge streams for optimal response generation. Extensive experiments on open-domain visual question answering benchmarks show that HKRAG consistently outperforms existing methods in both zero-shot and supervised settings, demonstrating the critical importance of holistic knowledge retrieval for VRD understanding. 

---
# Enhancing Sequential Recommendation with World Knowledge from Large Language Models 

**Authors**: Tianjie Dai, Xu Chen, Yunmeng Shu, Jinsong Lan, Xiaoyong Zhu, Jiangchao Yao, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.20177)  

**Abstract**: Sequential Recommendation System~(SRS) has become pivotal in modern society, which predicts subsequent actions based on the user's historical behavior. However, traditional collaborative filtering-based sequential recommendation models often lead to suboptimal performance due to the limited information of their collaborative signals. With the rapid development of LLMs, an increasing number of works have incorporated LLMs' world knowledge into sequential recommendation. Although they achieve considerable gains, these approaches typically assume the correctness of LLM-generated results and remain susceptible to noise induced by LLM hallucinations. To overcome these limitations, we propose GRASP (Generation Augmented Retrieval with Holistic Attention for Sequential Prediction), a flexible framework that integrates generation augmented retrieval for descriptive synthesis and similarity retrieval, and holistic attention enhancement which employs multi-level attention to effectively employ LLM's world knowledge even with hallucinations and better capture users' dynamic interests. The retrieved similar users/items serve as auxiliary contextual information for the later holistic attention enhancement module, effectively mitigating the noisy guidance of supervision-based methods. Comprehensive evaluations on two public benchmarks and one industrial dataset reveal that GRASP consistently achieves state-of-the-art performance when integrated with diverse backbones. The code is available at: this https URL. 

---
# Towards A Tri-View Diffusion Framework for Recommendation 

**Authors**: Ximing Chen, Pui Ieng Lei, Yijun Sheng, Yanyan Liu, Zhiguo Gong  

**Link**: [PDF](https://arxiv.org/pdf/2511.20122)  

**Abstract**: Diffusion models (DMs) have recently gained significant interest for their exceptional potential in recommendation tasks. This stems primarily from their prominent capability in distilling, modeling, and generating comprehensive user preferences. However, previous work fails to examine DMs in recommendation tasks through a rigorous lens. In this paper, we first experimentally investigate the completeness of recommender models from a thermodynamic view. We reveal that existing DM-based recommender models operate by maximizing the energy, while classic recommender models operate by reducing the entropy. Based on this finding, we propose a minimalistic diffusion framework that incorporates both factors via the maximization of Helmholtz free energy. Meanwhile, to foster the optimization, our reverse process is armed with a well-designed denoiser to maintain the inherent anisotropy, which measures the user-item cross-correlation in the context of bipartite graphs. Finally, we adopt an Acceptance-Rejection Gumbel Sampling Process (AR-GSP) to prioritize the far-outnumbered unobserved interactions for model robustness. AR-GSP integrates an acceptance-rejection sampling to ensure high-quality hard negative samples for general recommendation tasks, and a timestep-dependent Gumbel Softmax to handle an adaptive sampling strategy for diffusion models. Theoretical analyses and extensive experiments demonstrate that our proposed framework has distinct superiority over baselines in terms of accuracy and efficiency. 

---
# Adaptive Knowledge Transfer for Cross-Disciplinary Cold-Start Knowledge Tracing 

**Authors**: Yulong Deng, Zheng Guan, Min He, Xue Wang, Jie Liu, Zheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.20009)  

**Abstract**: Cross-Disciplinary Cold-start Knowledge Tracing (CDCKT) faces a critical challenge: insufficient student interaction data in the target discipline prevents effective knowledge state modeling and performance prediction. Existing cross-disciplinary methods rely on overlapping entities between disciplines for knowledge transfer through simple mapping functions, but suffer from two key limitations: (1) overlapping entities are scarce in real-world scenarios, and (2) simple mappings inadequately capture cross-disciplinary knowledge complexity. To overcome these challenges, we propose Mixed of Experts and Adversarial Generative Network-based Cross-disciplinary Cold-start Knowledge Tracing Framework. Our approach consists of three key components: First, we pre-train a source discipline model and cluster student knowledge states into K categories. Second, these cluster attributes guide a mixture-of-experts network through a gating mechanism, serving as a cross-domain mapping bridge. Third, an adversarial discriminator enforces feature separation by pulling same-attribute student features closer while pushing different-attribute features apart, effectively mitigating small-sample limitations. We validate our method's effectiveness across 20 extreme cross-disciplinary cold-start scenarios. 

---
# Popularity Bias Alignment Estimates 

**Authors**: Anton Lyubinin  

**Link**: [PDF](https://arxiv.org/pdf/2511.19999)  

**Abstract**: We are extending Popularity Bias Memorization theorem from arXiv:archive/2404.12008 in several directions. We extend it to arbitrary degree distributions and also prove both upper and lower estimates for the alignment with top-k singular hyperspace. 

---
# The 2nd Workshop on Human-Centered Recommender Systems 

**Authors**: Kaike Zhang, Jiakai Tang, Du Su, Shuchang Liu, Julian McAuley, Lina Yao, Qi Cao, Yue Feng, Fei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.19979)  

**Abstract**: Recommender systems shape how people discover information, form opinions, and connect with society. Yet, as their influence grows, traditional metrics, e.g., accuracy, clicks, and engagement, no longer capture what truly matters to humans. The workshop on Human-Centered Recommender Systems (HCRS) calls for a paradigm shift from optimizing engagement toward designing systems that truly understand, involve, and benefit people. It brings together researchers in recommender systems, human-computer interaction, AI safety, and social computing to explore how human values, e.g., trust, safety, fairness, transparency, and well-being, can be integrated into recommendation processes. Centered around three thematic axes-Human Understanding, Human Involvement, and Human Impact-HCRS features keynotes, panels, and papers covering topics from LLM-based interactive recommenders to societal welfare optimization. By fostering interdisciplinary collaboration, HCRS aims to shape the next decade of responsible and human-aligned recommendation research. 

---
# LLM-EDT: Large Language Model Enhanced Cross-domain Sequential Recommendation with Dual-phase Training 

**Authors**: Ziwei Liu, Qidong Liu, Wanyu Wang, Yejing Wang, Tong Xu, Wei Huang, Chong Chen, Peng Chuan, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.19931)  

**Abstract**: Cross-domain Sequential Recommendation (CDSR) has been proposed to enrich user-item interactions by incorporating information from various domains. Despite current progress, the imbalance issue and transition issue hinder further development of CDSR. The former one presents a phenomenon that the interactions in one domain dominate the entire behavior, leading to difficulty in capturing the domain-specific features in the other domain. The latter points to the difficulty in capturing users' cross-domain preferences within the mixed interaction sequence, resulting in poor next-item prediction performance for specific domains. With world knowledge and powerful reasoning ability, Large Language Models (LLMs) partially alleviate the above issues by performing as a generator and an encoder. However, current LLMs-enhanced CDSR methods are still under exploration, which fail to recognize the irrelevant noise and rough profiling problems. Thus, to make peace with the aforementioned challenges, we proposed an LLMs Enhanced Cross-domain Sequential Recommendation with Dual-phase Training ({LLM-EDT}). To address the imbalance issue while introducing less irrelevant noise, we first propose the transferable item augmenter to adaptively generate possible cross-domain behaviors for users. Then, to alleviate the transition issue, we introduce a dual-phase training strategy to empower the domain-specific thread with a domain-shared background. As for the rough profiling problem, we devise a domain-aware profiling module to summarize the user's preference in each domain and adaptively aggregate them to generate comprehensive user profiles. The experiments on three public datasets validate the effectiveness of our proposed LLM-EDT. To ease reproducibility, we have released the detailed code online at {this https URL}. 

---
# SCoTER: Structured Chain-of-Thought Transfer for Enhanced Recommendation 

**Authors**: Yang Wu, Qian Li, Yuling Xiong, Hongbo Tang, Xun Liu, Jun Zhang, Huan Yu, Jie Jiang, Hailong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2511.19514)  

**Abstract**: Harnessing the reasoning power of Large Language Models (LLMs) for recommender systems is hindered by two fundamental challenges. First, current approaches lack a mechanism for automated, data-driven discovery of effective reasoning patterns, relying instead on brittle manual templates or unstable zero-shot prompting. Second, they employ structure-collapsing integration: direct prompting incurs prohibitive online inference costs, while feature extraction collapses reasoning chains into single vectors, discarding stepwise logic. To address these challenges, we propose SCoTER (Structured Chain-of-Thought Transfer for Enhanced Recommendation), a unified framework that treats pattern discovery and structure-aware transfer as a jointly optimized problem. Specifically, SCoTER operationalizes this through two synergistic components: a GVM pipeline for automated pattern discovery and a structure-preserving integration architecture that transfers stepwise logic to efficient models. Formally, we provide information-theoretic justification proving that structure-preserving transfer achieves tighter performance bounds than structure-agnostic alternatives. Empirically, experiments on four benchmarks demonstrate improvements of 3.75\%-11.59\% over a strong TIGER backbone. Moreover, in production deployment on the Tencent Advertising Platform, SCoTER achieved a 2.14\% lift in Gross Merchandise Value (GMV) while eliminating online LLM inference costs. Overall, SCoTER establishes a principled and production-validated blueprint for transferring structured LLM reasoning to large-scale recommender systems. 

---
# Kleinkram: Open Robotic Data Management 

**Authors**: Cyrill Püntener, Johann Schwabe, Dominique Garmier, Jonas Frey, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2511.20492)  

**Abstract**: We introduce Kleinkram, a free and open-source system designed to solve the challenge of managing massive, unstructured robotic datasets. Designed as a modular, on-premises cloud solution, Kleinkram enables scalable storage, indexing, and sharing of datasets, ranging from individual experiments to large-scale research collections. Kleinkram natively integrates with standard formats such as ROS bags and MCAP and utilises S3-compatible storage for flexibility. Beyond storage, Kleinkram features an integrated "Action Runner" that executes customizable Docker-based workflows for data validation, curation, and benchmarking. Kleinkram has successfully managed over 30 TB of data from diverse robotic systems, streamlining the research lifecycle through a modern web interface and a robust Command Line Interface (CLI). 

---
# SEDA: A Self-Adapted Entity-Centric Data Augmentation for Boosting Gird-based Discontinuous NER Models 

**Authors**: Wen-Fang Su, Hsiao-Wei Chou, Wen-Yang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.20143)  

**Abstract**: Named Entity Recognition (NER) is a critical task in natural language processing, yet it remains particularly challenging for discontinuous entities. The primary difficulty lies in text segmentation, as traditional methods often missegment or entirely miss cross-sentence discontinuous entities, significantly affecting recognition accuracy. Therefore, we aim to address the segmentation and omission issues associated with such entities. Recent studies have shown that grid-tagging methods are effective for information extraction due to their flexible tagging schemes and robust architectures. Building on this, we integrate image data augmentation techniques, such as cropping, scaling, and padding, into grid-based models to enhance their ability to recognize discontinuous entities and handle segmentation challenges. Experimental results demonstrate that traditional segmentation methods often fail to capture cross-sentence discontinuous entities, leading to decreased performance. In contrast, our augmented grid models achieve notable improvements. Evaluations on the CADEC, ShARe13, and ShARe14 datasets show F1 score gains of 1-2.5% overall and 3.7-8.4% for discontinuous entities, confirming the effectiveness of our approach. 

---
# Invisible in Search? Auditing Aesthetic Bias in the Visual Representation of Holocaust Victims on Google 

**Authors**: Mykola Makhortykh, Tobias Rohrbach, Maryna Sydorova  

**Link**: [PDF](https://arxiv.org/pdf/2511.20036)  

**Abstract**: Information retrieval systems, such as search engines, increasingly shape the representation of the past and present states of social reality. Despite their importance, these systems face challenges in dealing with the ethical aspects of representation due to various forms of bias, including aesthetic bias that perpetuates hegemonic patterns of representation. While most research on aesthetic bias has examined it in the context of current societal issues, it is also crucial for historical representation, particularly of sensitive subjects such as historical atrocities. To address this gap, we conduct a comparative audit of the visual representation of Holocaust victims on Google. We find that Google tends to propagate a male-dominated representation of Holocaust victims with an emphasis on atrocity context, risking rendering invisible gender-specific suffering and decreasing potential for nurturing empathy. We also observe a variation in representation across geographic locations, suggesting that search algorithms may produce their own aesthetic of victimhood. 

---
# REWA: Witness-Overlap Theory -- Foundations for Composable Binary Similarity Systems 

**Authors**: Nikit Phadke  

**Link**: [PDF](https://arxiv.org/pdf/2511.19998)  

**Abstract**: REWA introduces a general theory of similarity based on witness-overlap structures. We show that whenever similarity between concepts can be expressed as monotone witness overlap -- whether arising from graph neighborhoods, causal relations, temporal structure, topological features, symbolic patterns, or embedding-based neighborhoods -- it admits a reduction to compact encodings with provable ranking preservation guarantees. REWA systems consist of: (1) finite witness sets $W(v)$, (2) semi-random bit assignments generated from each witness, and (3) monotonicity of expected similarity in the overlap $\Delta(u, v) = |W(u) \cap W(v)|$. We prove that under an overlap-gap condition on the final witness sets -- independent of how they were constructed -- top-$k$ rankings are preserved using $m = O(\log(|V|/\delta))$ bits. The witness-set formulation is compositional: any sequence of structural, temporal, causal, topological, information-theoretic, or learned transformations can be combined into pipelines that terminate in discrete witness sets. The theory applies to the final witness overlap, enabling modular construction of similarity systems from reusable primitives. This yields a vast design space: millions of composable similarity definitions inherit logarithmic encoding complexity. REWA subsumes and unifies Bloom filters, minhash, LSH bitmaps, random projections, sketches, and hierarchical filters as special cases. It provides a principled foundation for similarity systems whose behavior is governed by witness overlap rather than hash-function engineering. This manuscript presents the axioms, the main reducibility theorem, complete proofs with explicit constants, and a detailed discussion of compositional design, limitations, and future extensions including multi-bit encodings, weighted witnesses, and non-set representations. 

---
# $\text{R}^2\text{R}$: A Route-to-Rerank Post-Training Framework for Multi-Domain Decoder-Only Rerankers 

**Authors**: Xinyu Wang, Hanwei Wu, Qingchen Hu, Zhenghan Tai, Jingrui Tian, Lei Ding, Jijun Chi, Hailin He, Tung Sum Thomas Kwok, Yufei Cui, Sicheng Lyu, Muzhi Li, Mingze Li, Xinyue Yu, Ling Zhou, Peng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.19987)  

**Abstract**: Decoder-only rerankers are central to Retrieval-Augmented Generation (RAG). However, generalist models miss domain-specific nuances in high-stakes fields like finance and law, and naive fine-tuning causes surface-form overfitting and catastrophic forgetting. To address this challenge, we introduce R2R, a domain-aware framework that combines dynamic expert routing with a two-stage training strategy, Entity Abstraction for Generalization (EAG). EAG introduces a counter-shortcut mechanism by masking the most predictive surface cues, forcing the reranker to learn domain-invariant relevance patterns rather than memorizing dataset-specific entities. To efficiently activate domain experts, R2R employs a lightweight Latent Semantic Router that probes internal representations from the frozen backbone decoder to select the optimal LoRA expert per query. Extensive experiments across different reranker backbones and diverse domains (legal, medical, and financial) demonstrate that R2R consistently surpasses generalist and single-domain fine-tuned baselines. Our results confirm that R2R is a model-agnostic and modular approach to domain specialization with strong cross-domain robustness. 

---
