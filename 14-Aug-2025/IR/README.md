# Describe What You See with Multimodal Large Language Models to Enhance Video Recommendations 

**Authors**: Marco De Nadai, Andreas Damianou, Mounia Lalmas  

**Link**: [PDF](https://arxiv.org/pdf/2508.09789)  

**Abstract**: Existing video recommender systems rely primarily on user-defined metadata or on low-level visual and acoustic signals extracted by specialised encoders. These low-level features describe what appears on the screen but miss deeper semantics such as intent, humour, and world knowledge that make clips resonate with viewers. For example, is a 30-second clip simply a singer on a rooftop, or an ironic parody filmed amid the fairy chimneys of Cappadocia, Turkey? Such distinctions are critical to personalised recommendations yet remain invisible to traditional encoding pipelines. In this paper, we introduce a simple, recommendation system-agnostic zero-finetuning framework that injects high-level semantics into the recommendation pipeline by prompting an off-the-shelf Multimodal Large Language Model (MLLM) to summarise each clip into a rich natural-language description (e.g. "a superhero parody with slapstick fights and orchestral stabs"), bridging the gap between raw content and user intent. We use MLLM output with a state-of-the-art text encoder and feed it into standard collaborative, content-based, and generative recommenders. On the MicroLens-100K dataset, which emulates user interactions with TikTok-style videos, our framework consistently surpasses conventional video, audio, and metadata features in five representative models. Our findings highlight the promise of leveraging MLLMs as on-the-fly knowledge extractors to build more intent-aware video recommenders. 

---
# Multimodal Fusion And Sparse Attention-based Alignment Model for Long Sequential Recommendation 

**Authors**: Yongrui Fu, Jian Liu, Tao Li, Zonggang Wu, Shouke Qin, Hanmeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09664)  

**Abstract**: Recent advances in multimodal recommendation enable richer item understanding, while modeling users' multi-scale interests across temporal horizons has attracted growing attention. However, effectively exploiting multimodal item sequences and mining multi-grained user interests to substantially bridge the gap between content comprehension and recommendation remain challenging. To address these issues, we propose MUFASA, a MUltimodal Fusion And Sparse Attention-based Alignment model for long sequential recommendation. Our model comprises two core components. First, the Multimodal Fusion Layer (MFL) leverages item titles as a cross-genre semantic anchor and is trained with a joint objective of four tailored losses that promote: (i) cross-genre semantic alignment, (ii) alignment to the collaborative space for recommendation, (iii) preserving the similarity structure defined by titles and preventing modality representation collapse, and (iv) distributional regularization of the fusion space. This yields high-quality fused item representations for further preference alignment. Second, the Sparse Attention-guided Alignment Layer (SAL) scales to long user-behavior sequences via a multi-granularity sparse attention mechanism, which incorporates windowed attention, block-level attention, and selective attention, to capture user interests hierarchically and across temporal horizons. SAL explicitly models both the evolution of coherent interest blocks and fine-grained intra-block variations, producing robust user and item representations. Extensive experiments on real-world benchmarks show that MUFASA consistently surpasses state-of-the-art baselines. Moreover, online A/B tests demonstrate significant gains in production, confirming MUFASA's effectiveness in leveraging multimodal cues and accurately capturing diverse user preferences. 

---
# On Negative-aware Preference Optimization for Recommendation 

**Authors**: Chenlu Ding, Daoxuan Liu, Jiancan Wu, Xingyu Hu, Junkang Wu, Haitao Wang, Yongkang Wang, Xingxing Wang, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09653)  

**Abstract**: Recommendation systems leverage user interaction data to suggest relevant items while filtering out irrelevant (negative) ones. The rise of large language models (LLMs) has garnered increasing attention for their potential in recommendation tasks. However, existing methods for optimizing LLM-based recommenders face challenges in effectively utilizing negative samples. Simply integrating large numbers of negative samples can improve ranking accuracy and mitigate popularity bias but often leads to increased computational overhead and memory costs. Additionally, current approaches fail to account for the varying informativeness of negative samples, leading to suboptimal optimization performance. To address these issues, we propose NAPO (\textbf{N}egative-\textbf{A}ware \textbf{P}reference \textbf{O}ptimization), an enhanced framework for preference optimization in LLM-based recommendation. NAPO introduces two key innovations: (1) in-batch negative sharing, which expands the pool of negative samples without additional memory overhead, and (2) dynamic reward margin adjustment, which adapts model updates based on the confidence of negative samples. Extensive experiments on three public datasets demonstrate that NAPO outperforms existing methods in both recommendation accuracy and popularity bias reduction. 

---
# Personalized Product Search Ranking: A Multi-Task Learning Approach with Tabular and Non-Tabular Data 

**Authors**: Lalitesh Morishetti, Abhay Kumar, Jonathan Scott, Kaushiki Nag, Gunjan Sharma, Shanu Vashishtha, Rahul Sridhar, Rohit Chatter, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09636)  

**Abstract**: In this paper, we present a novel model architecture for optimizing personalized product search ranking using a multi-task learning (MTL) framework. Our approach uniquely integrates tabular and non-tabular data, leveraging a pre-trained TinyBERT model for semantic embeddings and a novel sampling technique to capture diverse customer behaviors. We evaluate our model against several baselines, including XGBoost, TabNet, FT-Transformer, DCN-V2, and MMoE, focusing on their ability to handle mixed data types and optimize personalized ranking. Additionally, we propose a scalable relevance labeling mechanism based on click-through rates, click positions, and semantic similarity, offering an alternative to traditional human-annotated labels. Experimental results show that combining non-tabular data with advanced embedding techniques in multi-task learning paradigm significantly enhances model performance. Ablation studies further underscore the benefits of incorporating relevance labels, fine-tuning TinyBERT layers, and TinyBERT query-product embedding interactions. These results demonstrate the effectiveness of our approach in achieving improved personalized product search ranking. 

---
# TFRank: Think-Free Reasoning Enables Practical Pointwise LLM Ranking 

**Authors**: Yongqi Fan, Xiaoyang Chen, Dezhi Ye, Jie Liu, Haijin Liang, Jin Ma, Ben He, Yingfei Sun, Tong Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09539)  

**Abstract**: Reasoning-intensive ranking models built on Large Language Models (LLMs) have made notable progress, but existing approaches often rely on large-scale LLMs and explicit Chain-of-Thought (CoT) reasoning, resulting in high computational cost and latency that limit real-world use. To address this, we propose \textbf{TFRank}, an efficient pointwise reasoning ranker based on small-scale LLMs. To improve ranking performance, TFRank effectively integrates CoT data, fine-grained score supervision, and multi-task training. Furthermore, it achieves an efficient ``\textbf{T}hink-\textbf{F}ree" reasoning capability by employing a ``think-mode switch'' and pointwise format constraints. Specifically, this allows the model to leverage explicit reasoning during training while delivering precise relevance scores for complex queries at inference without generating any reasoning chains. Experiments show that TFRank (e.g., 1.7B) achieves performance comparable to models with four times more parameters on the BRIGHT benchmark, and demonstrates strong competitiveness on the BEIR benchmark. Further analysis shows that TFRank achieves an effective balance between performance and efficiency, providing a practical solution for integrating advanced reasoning into real-world systems. Our code and data are released in the repository: this https URL. 

---
# Improving Dense Passage Retrieval with Multiple Positive Passages 

**Authors**: Shuai Chang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09534)  

**Abstract**: By leveraging a dual encoder architecture, Dense Passage Retrieval (DPR) has outperformed traditional sparse retrieval algorithms such as BM25 in terms of passage retrieval accuracy. Recently proposed methods have further enhanced DPR's performance. However, these models typically pair each question with only one positive passage during training, and the effect of associating multiple positive passages has not been examined. In this paper, we explore the performance of DPR when additional positive passages are incorporated during training. Experimental results show that equipping each question with multiple positive passages consistently improves retrieval accuracy, even when using a significantly smaller batch size, which enables training on a single GPU. 

---
# Towards Self-cognitive Exploration: Metacognitive Knowledge Graph Retrieval Augmented Generation 

**Authors**: Xujie Yuan, Shimin Di, Jielong Tang, Libin Zheng, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09460)  

**Abstract**: Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) significantly enhances the reasoning capabilities of LargeLanguage Models by leveraging structured knowledge. However, existing KG-RAG frameworks typically operate as open-loop systems, suffering from cognitive blindness, an inability to recognize their exploration deficiencies. This leads to relevance drift and incomplete evidence, which existing self-refinement methods, designed for unstructured text-based RAG, cannot effectively resolve due to the path-dependent nature of graph exploration. To address this challenge, we propose Metacognitive Knowledge Graph Retrieval Augmented Generation (MetaKGRAG), a novel framework inspired by the human metacognition process, which introduces a Perceive-Evaluate-Adjust cycle to enable path-aware, closed-loop refinement. This cycle empowers the system to self-assess exploration quality, identify deficiencies in coverage or relevance, and perform trajectory-connected corrections from precise pivot points. Extensive experiments across five datasets in the medical, legal, and commonsense reasoning domains demonstrate that MetaKGRAG consistently outperforms strong KG-RAG and self-refinement baselines. Our results validate the superiority of our approach and highlight the critical need for path-aware refinement in structured knowledge retrieval. 

---
# On the Consistency and Performance of the Iterative Bayesian Update 

**Authors**: Ehab ElSalamouny, Catuscia Palamidessi  

**Link**: [PDF](https://arxiv.org/pdf/2508.09980)  

**Abstract**: For many social, scientific, and commercial purposes, it is often important to estimate the distribution of the users' data regarding a sensitive attribute, e.g., their ages, locations, etc. To allow this estimation while protecting the users' privacy, every user applies a local privacy protection mechanism that releases a noisy (sanitized) version of their original datum to the data collector; then the original distribution is estimated using one of the known methods, such as the matrix inversion (INV), RAPPOR's estimator, and the iterative Bayesian update (IBU). Unlike the other estimators, the consistency of IBU, i.e., the convergence of its estimate to the real distribution as the amount of noisy data grows, has been either ignored or incorrectly proved in the literature. In this article, we use the fact that IBU is a maximum likelihood estimator to prove that IBU is consistent. We also show, through experiments on real datasets, that IBU significantly outperforms the other methods when the users' data are sanitized by geometric, Laplace, and exponential mechanisms, whereas it is comparable to the other methods in the case of the k-RR and RAPPOR mechanisms. Finally, we consider the case when the alphabet of the sensitive data is infinite, and we show a technique that allows IBU to operate in this case too. 

---
# A Signer-Invariant Conformer and Multi-Scale Fusion Transformer for Continuous Sign Language Recognition 

**Authors**: Md Rezwanul Haque, Md. Milon Islam, S M Taslim Uddin Raju, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2508.09372)  

**Abstract**: Continuous Sign Language Recognition (CSLR) faces multiple challenges, including significant inter-signer variability and poor generalization to novel sentence structures. Traditional solutions frequently fail to handle these issues efficiently. For overcoming these constraints, we propose a dual-architecture framework. For the Signer-Independent (SI) challenge, we propose a Signer-Invariant Conformer that combines convolutions with multi-head self-attention to learn robust, signer-agnostic representations from pose-based skeletal keypoints. For the Unseen-Sentences (US) task, we designed a Multi-Scale Fusion Transformer with a novel dual-path temporal encoder that captures both fine-grained posture dynamics, enabling the model's ability to comprehend novel grammatical compositions. Experiments on the challenging Isharah-1000 dataset establish a new standard for both CSLR benchmarks. The proposed conformer architecture achieves a Word Error Rate (WER) of 13.07% on the SI challenge, a reduction of 13.53% from the state-of-the-art. On the US task, the transformer model scores a WER of 47.78%, surpassing previous work. In the SignEval 2025 CSLR challenge, our team placed 2nd in the US task and 4th in the SI task, demonstrating the performance of these models. The findings validate our key hypothesis: that developing task-specific networks designed for the particular challenges of CSLR leads to considerable performance improvements and establishes a new baseline for further research. The source code is available at: this https URL. 

---
# RicciFlowRec: A Geometric Root Cause Recommender Using Ricci Curvature on Financial Graphs 

**Authors**: Zhongtian Sun, Anoushka Harit  

**Link**: [PDF](https://arxiv.org/pdf/2508.09334)  

**Abstract**: We propose RicciFlowRec, a geometric recommendation framework that performs root cause attribution via Ricci curvature and flow on dynamic financial graphs. By modelling evolving interactions among stocks, macroeconomic indicators, and news, we quantify local stress using discrete Ricci curvature and trace shock propagation via Ricci flow. Curvature gradients reveal causal substructures, informing a structural risk-aware ranking function. Preliminary results on S\&P~500 data with FinBERT-based sentiment show improved robustness and interpretability under synthetic perturbations. This ongoing work supports curvature-based attribution and early-stage risk-aware ranking, with plans for portfolio optimization and return forecasting. To our knowledge, RicciFlowRec is the first recommender to apply geometric flow-based reasoning in financial decision support. 

---
# ParallelSearch: Train your LLMs to Decompose Query and Search Sub-queries in Parallel with Reinforcement Learning 

**Authors**: Shu Zhao, Tan Yu, Anbang Xu, Japinder Singh, Aaditya Shukla, Rama Akkiraju  

**Link**: [PDF](https://arxiv.org/pdf/2508.09303)  

**Abstract**: Reasoning-augmented search agents such as Search-R1, trained via reinforcement learning with verifiable rewards (RLVR), demonstrate remarkable capabilities in multi-step information retrieval from external knowledge sources. These agents address the limitations of their parametric memory by dynamically gathering relevant facts to address complex reasoning tasks. However, existing approaches suffer from a fundamental architectural limitation: they process search queries strictly sequentially, even when handling inherently parallelizable and logically independent comparisons. This sequential bottleneck significantly constrains computational efficiency, particularly for queries that require multiple entity comparisons. To address this critical limitation, we propose ParallelSearch, a novel reinforcement learning framework that empowers large language models (LLMs) to recognize parallelizable query structures and execute multiple search operations concurrently. Our approach introduces dedicated reward functions that incentivize the identification of independent query components while preserving answer accuracy through jointly considering correctness, query decomposition quality, and parallel execution benefits. Comprehensive experiments demonstrate that ParallelSearch outperforms state-of-the-art baselines by an average performance gain of 2.9% across seven question-answering benchmarks. Notably, on parallelizable questions, our method achieves a 12.7% performance improvement while requiring only 69.6% of the LLM calls compared to sequential approaches. 

---
# Multimodal RAG Enhanced Visual Description 

**Authors**: Amit Kumar Jaiswal, Haiming Liu, Ingo Frommholz  

**Link**: [PDF](https://arxiv.org/pdf/2508.09170)  

**Abstract**: Textual descriptions for multimodal inputs entail recurrent refinement of queries to produce relevant output images. Despite efforts to address challenges such as scaling model size and data volume, the cost associated with pre-training and fine-tuning remains substantial. However, pre-trained large multimodal models (LMMs) encounter a modality gap, characterised by a misalignment between textual and visual representations within a common embedding space. Although fine-tuning can potentially mitigate this gap, it is typically expensive and impractical due to the requirement for extensive domain-driven data. To overcome this challenge, we propose a lightweight training-free approach utilising Retrieval-Augmented Generation (RAG) to extend across the modality using a linear mapping, which can be computed efficiently. During inference, this mapping is applied to images embedded by an LMM enabling retrieval of closest textual descriptions from the training set. These textual descriptions, in conjunction with an instruction, cater as an input prompt for the language model to generate new textual descriptions. In addition, we introduce an iterative technique for distilling the mapping by generating synthetic descriptions via the language model facilitating optimisation for standard utilised image description measures. Experimental results on two benchmark multimodal datasets demonstrate significant improvements. 

---
