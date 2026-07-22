# PAGE-RAG: Evidence-Grounded Adaptive Graph Retrieval for Long-Document Question Answering 

**Authors**: Xingyu Chen, Junxiu An, Jun Guo, Li Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19301)  

**Abstract**: GraphRAG improves long-document question answering by introducing structured representations beyond conventional retrieval. However, automatically constructed graphs are inherently incomplete projections of source documents, and treating them as independent knowledge sources may lead to unreliable retrieval and generation. We propose PAGE-RAG, a projection-aware adaptive graph retrieval framework for reliable long-document question answering. PAGE-RAG views graph structures as semantic skeletons that organize and navigate document knowledge, rather than replacing the original knowledge source. Based on this perspective, PAGE-RAG introduces a task-adaptive retrieval routing strategy that dynamically selects appropriate retrieval behaviors according to query requirements. Furthermore, PAGE-RAG incorporates strict knowledge boundary control, ensuring that generated responses remain grounded within available evidence and abstaining from unsupported information beyond the accessible knowledge scope. Experiments demonstrate that PAGE-RAG achieves competitive answer quality while improving retrieval efficiency and knowledge reliability, highlighting the importance of projection-aware graph modeling, adaptive retrieval, and explicit knowledge boundary control for trustworthy GraphRAG systems. The source code is publicly available at this https URL. 

---
# Spectral Biclustering-Driven Scalability for Post-Hoc Explainability in Recommender Systems 

**Authors**: Jose L. Salmeron, Irina Arévalo  

**Link**: [PDF](https://arxiv.org/pdf/2607.19189)  

**Abstract**: Explainability in recommender systems is essential for ensuring transparency, accountability, and trust, yet existing post-hoc methods often encounter severe scalability challenges. Observation-level deletion diagnostics offer a counterfactual way to analyze recommendations by retraining models after removing individual users or items, but their cost grows rapidly with dataset size. To improve the practical tractability of this analysis, this paper introduces a block-deletion diagnostic framework that uses spectral biclustering to group users and items and then removes entire blocks of interactions. This formulation reduces the number of retraining procedures relative to finer-grained deletion strategies and produces explanations at the level of user segments, item groups, and their interactions. The framework is evaluated on two representative recommender paradigms, Singular Value Decomposition and Neural Collaborative Filtering, using the MovieLens and Amazon datasets. The results show that top-ranked recommendations are often more sensitive to specific interaction blocks than lower-ranked ones, with some blocks acting as supporting evidence and others having a detrimental effect on recommendation quality. The analysis also indicates that user segments differ in their sensitivity to block removal, suggesting heterogeneous levels of reliance on localized interaction patterns. These findings provide diagnostic information that is not directly visible through standard recommendation metrics. Overall, the results suggest that block-deletion diagnostics offer a practical and model-agnostic post-hoc analysis framework for recommender systems, while also highlighting that the resulting explanations depend on the chosen block structure. 

---
# Answer-Reconstruction Search Density: Measuring the Query and Source Work Compressed by Conversational Answers 

**Authors**: Benjamin Tannenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2607.18904)  

**Abstract**: Conversational systems can collapse a visible sequence of web queries, result inspections, and source comparisons into a single synthesized answer. Existing retrieval metrics evaluate ranking, effort, or factual support, but they do not quantify the minimum conventional search work represented by a completed answer. We define answer-reconstruction search density (\ARSD): the minimum number of distinct query actions required, under a fixed and dated reconstruction policy, to support a target share of atomic retrievable answer units. A parallel page-density measure separates query compression from source compression. 

---
# TSGR: Taobao Search Generative Retrieval 

**Authors**: Tianyu Zhan, Gui Ling, Tong Xiong, Kunhai Lin, Yang Wang, Kaixuan Zhang, Zhihong Chen, Yuliang Yan, Dan Ou, Shengyu Zhang, Haihong Tang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.18796)  

**Abstract**: Generative retrieval (GR) has demonstrated strong promise for industrial e-commerce search by training a single autoregressive model to directly generate the Semantic IDs (SIDs) of target items. However, existing GR systems are primarily optimized for semantic matching and remain insensitive to item business value: SID construction is value-unaware, and candidates are ranked without access to item side-info. Consequently, high-value items are often missed or deprioritized at the retrieval stage, limiting downstream business impact. This limitation is particularly critical in industrial settings such as Taobao Search, where business objectives are central to system design. To address this, we propose $\textbf{T}$aobao $\textbf{S}$earch $\textbf{G}$enerative $\textbf{R}$etrieval ($\textbf{TSGR}$), a unified generative retrieval framework that incorporates value awareness into both item representation and candidate ranking. 1) For item representation, TSGR introduces $\textbf{Query-aware Parallel SID (QP-SID)}$, which encodes query-conditioned value orderings into the SID construction by building parallel codebooks derived from query-item statistics, so that higher-value and query-relevant items are assigned better token indices. 2) For candidate ranking, we introduce a $\textbf{Value-aware Ranking Module (VRM)}$ that is built upon and jointly optimized with the GR, enabling a single model to seamlessly serve as both retriever and pre-ranker without a dedicated pre-ranking stage. A progressive training pipeline further aligns the model with semantic relevance, user preferences, and business objectives. Offline experiments show that TSGR achieves an 9.16% improvement in HR@1000, and online A/B tests further validate its effectiveness, yielding gains of +0.43% in IPV, +1.12% in Transaction Count, and +1.64% in GMV. 

---
# Beyond Noisy Signals: Dual-Level Denoising for Multi-modal Sequential Recommendation 

**Authors**: Jie Luo, Qi Jin, Xinming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.18786)  

**Abstract**: Multi-modal Sequential Recommendation (SR) incorporates rich side information (e.g., textual and visual features) to enhance dynamic user preference modeling. However, existing frameworks inevitably suffer from a \textbf{Dual-Noise Dilemma}: (1) \textit{Feature-level redundancy} stemming from the semantic gap between generic pre-trained representations and fine-grained recommendation intent; and (2) \textit{Sequence-level stochasticity} induced by spurious interactions such as accidental clicks. To break this bottleneck, we propose \textbf{DDMSR}, a novel \textbf{D}ual-level \textbf{D}enoising \textbf{M}ulti-modal \textbf{S}equential \textbf{R}ecommendation framework that systematically purifies signals from both feature-topological and sequence-frequency perspectives. Specifically, we first design a graph-based feature denoising module that leverages Laplacian smoothing on item semantic graphs as a structural low-pass filter, effectively suppressing high-frequency semantic noise while preserving salient features. For sequence purification, we introduce a frequency-domain sequence denoising module, utilizing the Fast Fourier Transform and a learnable frequency filter to adaptively modulate the interaction spectrum and attenuate anomalous signals. Furthermore, a multi-modal contrastive alignment objective is incorporated to bridge the heterogeneity gap and enforce cross-modal semantic consistency. Extensive experiments on four public benchmark datasets demonstrate that DDMSR consistently outperforms state-of-the-art baselines, providing a highly robust and efficient solution for multi-modal sequential recommendation. The source code is available at: ~\href{this https URL}{\textcolor{blue}{this https URL}}. 

---
# RAGAL: A Frugal, Fully Local Retrieval-Augmented Assistant for Technical Support at a Government Agency 

**Authors**: Dan Musetoiu  

**Link**: [PDF](https://arxiv.org/pdf/2607.18756)  

**Abstract**: Public institutions hold large volumes of sensitive documents and support tickets that cannot leave the premises, ruling out cloud-hosted language models entirely. We report on RAGAL, a retrieval-augmented assistant for the technical-support team of AFIR, the Romanian Agency for Financing Rural Investments, built and operated under three hard constraints: zero data egress (no external API calls, even for synthetic data), a read-only mandate (the assistant drafts, humans execute), and a single 8 GB consumer laptop as the only development and training machine. Over a Romanian-language corpus of ~25,000 chunks -- 15,073 resolved support tickets and internal normative documents -- we show that the highest-leverage investments were retrieval engineering and retriever fine-tuning rather than a larger generator: hybrid dense-sparse retrieval with intent routing raised our internal evaluation from 62% to 81%, and fine-tuning the bge-m3 embedder on real ticket data improved recall@10 from 0.663 to 0.850 (MRR 0.489 to 0.684) after 72 minutes of training. We document a general pitfall: single-domain fine-tuning silently degraded retrieval on the untouched document domain below the stock baseline, detected only after building a per-domain evaluation set and repaired with locally generated queries (GenQ). We report two counter-intuitive findings -- PII masking improved generation quality, and a structural "anchor distillation" scheme made SQL hallucination impossible by construction -- along with a reproducible recipe for full embedder fine-tuning in 8 GB of VRAM. Finally, since zero egress also rules out a cloud judge, we describe a substitute: a 744B-parameter model run on CPU, too slow to serve interactively but affordable in overnight batch, used as a second opinion whose limits we quantify. We release the sanitized pipeline scripts for institutions facing similar data-locality constraints. 

---
# An Epistemic Position-Based Click Model: From Interactions to Epistemic Distributions of Relevance and Bias 

**Authors**: Oscar Rolando Ramirez Milian, Harrie Oosterhuis  

**Link**: [PDF](https://arxiv.org/pdf/2607.18712)  

**Abstract**: User interactions with rankings are affected by both items' relevances and display positions. Accordingly, click probabilities are often modeled as a product of relevance and position factors; and for improving recommendation and search, one needs to disentangle relevance from position bias. However, existing click models only provide frequentist point-estimates that do not capture any measure of epistemic uncertainty. Consequently, there is no indication of how much confidence one should have in their predictions. In this work, we introduce the first evidential deep-learning approach to form an epistemic alternative to the important position-based click model. Our learned model takes as input item and position features and outputs a beta-distribution for every relevance and position-bias variable of the position-based model. These distributions capture epistemic uncertainty about click probabilities and the underlying effects of attraction and position bias. The main challenge of our approach is its optimization for which we propose approximation and conditioning techniques to provide numerical stability and variance reduction. Our experiments indicate that our approach captures epistemic uncertainty in predictions on previously-unseen data, whereas standard policy gradients fail to learn meaningful distributions. We believe our contribution of the first contextual epistemic click model constitutes an important step in incorporating Bayesian uncertainty into click modeling. 

---
# PLAID-PRF: Pseudo-Relevance Feedback with Centroid-like Tokens in PLAID 

**Authors**: Xiao Wang, Sean MacAvaney, Craig Macdonald  

**Link**: [PDF](https://arxiv.org/pdf/2607.18626)  

**Abstract**: Multi-vector dense retrieval models, such as ColBERT, achieve strong retrieval effectiveness by modelling fine-grained token-level interactions between queries and documents. Methods such as PLAID use centroid-based quantisation of each token's vector to reduce the index size and speed up retrieval while maintaining strong effectiveness. In this work, we introduce PLAID-PRF, a method that performs Pseudo-Relevance Feedback (PRF) over PLAID to reformulate ColBERT's query vectors based on the top-retrieved results. In contrast with prior methods that perform PRF on multi-vector retrieval models, PLAID-PRF keeps computational costs low by leveraging the internal PLAID centroid vectors, treating them similarly to tokens in traditional PRF methods. The method selects a small and diverse set of high-utility expansion vectors and appends them to the original query, rerunning PLAID to refine both candidate generation and final scoring. Extensive experiments on the standard in-domain MSMARCO and four out-of-domain BEIR benchmarks show that PLAID-PRF consistently improves retrieval effectiveness over various baselines. In particular, PLAID-PRF improves over PLAID by up to 4.3% nDCG@10 and 7.3% MRR@10, while introducing substantially less computation overhead than prior PRF methods. The results demonstrate that our proposed centroid-aware PRF method offers an effective and lightweight mechanism to improve the quality of top-ranked retrieved results. Overall, this work enables effective and efficient feedback-aware late-interaction retrieval without expensive query-time document-token clustering. 

---
# Mitigating Matthew Effect: Multi-Hypergraph Boosted Multi-Interest Self-Supervised Learning for Conversational Recommendation 

**Authors**: Yongsen Zheng, Ruilin Xu, Guohua Wang, Liang Lin, Kwok-Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2607.18609)  

**Abstract**: The Matthew effect is a big challenge in Recommender Systems (RSs), where popular items tend to receive increasing attention, while less popular ones are often overlooked, perpetuating existing disparities. Although many existing methods attempt to mitigate Matthew effect in the static or quasi-static recommendation scenarios, such issue will be more pronounced as users engage with the system over time. To this end, we propose a novel framework, Multi-Hypergraph Boosted Multi-Interest Self-Supervised Learning for Conversational Recommendation (HiCore), aiming to address Matthew effect in the Conversational Recommender System (CRS) involving the dynamic user-system feedback loop. It devotes to learn multi-level user interests by building a set of hypergraphs (i.e., item-, entity-, word-oriented multiple-channel hypergraphs) to alleviate the Matthew effec. Extensive experiments on four CRS-based datasets showcase that HiCore attains a new state-of-the-art performance, underscoring its superiority in mitigating the Matthew effect effectively. Our code is available at this https URL. 

---
# AutoIndex: Learning Representation Programs for Retrieval 

**Authors**: Sam O'Nuallain, Nithya Rajkumar, Ramya Narayanasamy, Hanna Jiang, Shreyas Chaudhari, Andrew Drozdov  

**Link**: [PDF](https://arxiv.org/pdf/2607.18603)  

**Abstract**: We present AutoIndex, a framework for learning representation programs: executable transformations that map raw documents into the representations exposed to a retrieval system. Rather than tuning retrievers, rerankers, or a small set of preprocessing hyperparameters, AutoIndex searches over programs that slice, enrich, normalize, reweight, or reorganize documents before indexing. At each iteration, AutoIndex performs validation-guided program search, in which agents diagnose failures of the current program and synthesize candidate updates, retaining only updates that improve retrieval quality under the resulting index. We evaluate AutoIndex on CRUMB, a benchmark of heterogeneous retrieval tasks, with BM25 held fixed across all experiments. The learned programs improve recall over a static full-document BM25 baseline on all 8 tasks, with average gains of +8.4% in Recall@100 and +8.3% in nDCG@10, and largest gains of +30.5% in Recall@100 and +43.6% in nDCG@10. These results suggest that document representation should not be treated as a fixed preprocessing choice made before retrieval begins, but as an explicit optimization target. Code to reproduce our results is available at this https URL. 

---
# Topology-Aware Tokenization for Generative Recommendation 

**Authors**: Yaokun Liu, Yifan Liu, Zhenrui Yue, Gyuseok Lee, Zelin Li, Ruichen Yao, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.18600)  

**Abstract**: Generative recommendation reformulates sequential recommendation as an autoregressive generation task, yet a critical issue in this paradigm remains overlooked: topology distortion in item tokenization. In particular, we observe that the intrinsic adjacency relationships of items in the pretrained semantic embedding space are significantly disrupted after quantization. This topology distortion misleads the model's perception of item similarity, ultimately bottlenecking the accuracy of generative recommendations. To address this issue, we propose Topology-Aware Tokenization (TopoTok), an item tokenization framework that preserves item relational structure throughout the quantization hierarchy. Different from the prior monolithic supervision in tokenization, TopoTok introduces a multi-level distillation scheme to progressively recover the topology from coarse to fine granularity: 1) Inter-Group Distillation to capture global cluster-wise relations; 2) Intra-Group Distillation to refine local structures within semantic clusters; and 3) Inter-Item Distillation to enforce fine-grained alignment at the individual item level. Extensive experiments on three benchmark datasets demonstrate that TopoTok effectively alleviates topology distortion and consistently outperforms state-of-the-art tokenizers, achieving significant performance gains of up to 9.42% in Recall@5. 

---
# Sequential Learner Modeling Using Multi-Relational Graph Convolutional Networks 

**Authors**: Rawaa Alatrash, Mohamed Amine Chatti, Hong Yang, Yumeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.19253)  

**Abstract**: User modeling is a critical task in a variety of personalized systems. Recognizing their effectiveness in learning from graph-structured data, Graph Neural Networks (GNNs), particularly Graph Convolutional Networks (GCNs), are increasingly employed for user modeling. However, existing approaches typically treat different relation types in a graph as homogeneous, limiting their ability to capture richer semantics and construct more informative user models. While multi-relational GNNs (MR-GNNs) have been adopted for representation learning and recommendation, their application for user modeling remains unexplored. Moreover, existing GNN-based user modeling approaches ignore the user interaction sequence. To address these research gaps, in this work we propose MR-ConceptGCN, a novel fully unsupervised approach focused on concept-based sequential learner modeling using multi-relational GCNs (MR-GCNs). MR-ConceptGCN effecively combines Personal Knowledge Graphs (PKGs), MR-GCNs, and the pre-trained language model SBERT to obtain enhanced relation- and semantic-aware representations of the PKG items. The enriched embeddings of the knowledge concepts that a learner did not understand when interacting with learning materials in CourseMapper are then used to construct a sequential learner model that combines long-term and short-term learner interactions. We report the results of an online user study (n = 31), demonstrating the benefits of MR-ConceptGCN in terms of several important user-centric aspects including accuracy, usefulness, diversity, and satisfaction with an educational recommender system. 

---
# Biological Amnesia in ICU Time-Series Prediction: A Drift-Adaptive Two-Stream Architecture with Temporal Retrieval 

**Authors**: Fatema Ferdous Tamanna, K. M. Merajul Arefin, Md. Abdul Masud  

**Link**: [PDF](https://arxiv.org/pdf/2607.19020)  

**Abstract**: Background: Clinical decision support systems degrade silently as treatment protocols evolve, yet standard adaptation methods treat models as monolithic blocks, unable to distinguish stable patient physiology from shifting institutional practice. Methods: We propose an adaptive clinical intelligence architecture for ICU intervention prediction that structurally decouples physiological from treatment representations, confining parameter updates to the treatment stream upon a dual distributional and accuracy trigger. Automated audit logs record which treatment features drove each adaptation event and how their importance shifted. At inference, an attribution-driven Temporal RAG module grounds each prediction in patient-specific, era-matched PubMed evidence anchored to the patient's dominant physiological features. Experiments used 84,792 MIMIC-IV stays (2008-2022) under strict chronological split. Results: Drift localised entirely to the treatment stream, validating the structural prior. Selective adaptation improved vasopressor and septic shock discrimination and calibration over the static source model. A fully retrained baseline yielded marginally higher aggregate discrimination but missed 26 septic shock cases the framework correctly identified, with none in the reverse direction; retrieval consistency with the pre-adaptation source model was preserved by the framework but degraded substantially in the retrained baseline. Conclusions: Structurally constraining adaptation to drifting components while preserving stable physiological representations enables clinical AI to evolve with practice without distorting learned patient biology. This architecture offers a template for governable, interpretable deployment of adaptive models in high-stakes clinical environments. 

---
# AILQA: Evaluating AI-Driven Legal Question Answering Systems for the Indian Legal System 

**Authors**: Shubham Kumar Nigam, Shubham Kumar Mishra, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2607.18825)  

**Abstract**: This comprehensive study introduces an advanced Artificial Intelligence for Indian Legal Question Answering (AILQA) system tailored to the Indian legal context. AILQA leverages a variety of embedding and generative models, including recent Large Language Models (LLMs), to address the unique challenges posed by the intricate and diverse nature of Indian legal texts and to enhance the accuracy and reliability of responses to legal questions. We conducted rigorous evaluations using both lexical and semantic metrics, enriched by expert legal feedback, to ensure relevance and accuracy. Our findings underscore the effectiveness of the Retrieval-Augmented Generation (RAG) paradigm in improving answer quality, particularly in complex legal domains. Additionally, we assessed performance on standardized tests such as the All India Bar Examination (AIBE), thereby providing a robust benchmark for practical applications. Under the study's evaluation protocol, some AI-generated responses received higher ratings than the available reference answers, particularly when they contained accurate and relevant supporting details. This finding is specific to the evaluated dataset and rating criteria and should not be interpreted as evidence that the models generally outperform qualified legal professionals. We also discuss the challenges encountered, such as the need for precise context and the risks of model hallucination, and propose directions for future research to further refine AI capabilities in the legal field. This study aims to pave the way for enhanced legal decision-support systems, making them more accessible and effective for legal professionals and the public alike. 

---
# Exposure-Based Reinforcement Learning to Rank 

**Authors**: Harrie Oosterhuis, Rolf Jagerman, Zhen Qin, Xuanhui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.18689)  

**Abstract**: Reinforcement learning (RL) methods for learning-to-rank (LTR) can optimize (almost) any ranking goal, e.g., from precision or discounted cumulative gain to fairness-of-exposure or ranking distillation. However, standard RL is ineffective and computationally costly due to the enormous action space in LTR settings. Existing methods reach computational efficiency through custom gradient computation algorithms, but they are very complex to implement and often clash with auto-differentiation. Consequently, existing RL for LTR is not attractive to many practitioners. We reconsider RL for LTR while actively avoiding reliance on custom gradients. Contrary to the existing approaches, we focus on variance reduction and GPU computation. In doing so, we discover that high sample-efficiency can be reached through baseline corrections and partial marginalization. Furthermore, we propose an abstraction that places gradient estimation behind a document-exposure distribution, this enables seamless plug-and-play integration with auto-differentiation. Thereby, one only has to implement a loss as a differentiable function of exposure and RL for LTR can optimize it using auto-differentiation. Our experimental results reveal that our new exposure-based RL for LTR approach converges considerably faster and at significantly higher ranking performance than existing custom gradients, with no additional costs in computation time when using GPUs. In contrast, existing custom gradients result in severe stability issues when converging over many epochs, which never occur for our methods. Thus, we considerably improve RL for LTR methodology by increasing its effectiveness, efficiency, and ease of application. 

---
