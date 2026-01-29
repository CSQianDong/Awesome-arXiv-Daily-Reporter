# MedViz: An Agent-based, Visual-guided Research Assistant for Navigating Biomedical Literature 

**Authors**: Huan He, Xueqing Peng, Yutong Xie, Qijia Liu, Chia-Hsuan Chang, Lingfei Qian, Brian Ondov, Qiaozhu Mei, Hua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.20709)  

**Abstract**: Biomedical researchers face increasing challenges in navigating millions of publications in diverse domains. Traditional search engines typically return articles as ranked text lists, offering little support for global exploration or in-depth analysis. Although recent advances in generative AI and large language models have shown promise in tasks such as summarization, extraction, and question answering, their dialog-based implementations are poorly integrated with literature search workflows. To address this gap, we introduce MedViz, a visual analytics system that integrates multiple AI agents with interactive visualization to support the exploration of the large-scale biomedical literature. MedViz combines a semantic map of millions of articles with agent-driven functions for querying, summarizing, and hypothesis generation, allowing researchers to iteratively refine questions, identify trends, and uncover hidden connections. By bridging intelligent agents with interactive visualization, MedViz transforms biomedical literature search into a dynamic, exploratory process that accelerates knowledge discovery. 

---
# Overview of the TREC 2025 Tip-of-the-Tongue track 

**Authors**: Jaime Arguello, Fernando Diaz, Maik Fröebe, To Eun Kim, Bhaskar Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2601.20671)  

**Abstract**: Tip-of-the-tongue (ToT) known-item retrieval involves re-finding an item for which the searcher does not reliably recall an identifier. ToT information requests (or queries) are verbose and tend to include several complex phenomena, making them especially difficult for existing information retrieval systems. The TREC 2025 ToT track focused on a single ad-hoc retrieval task. This year, we extended the track to general domain and incorporated different sets of test queries from diverse sources, namely from the MS-ToT dataset, manual topic development, and LLM-based synthetic query generation. This year, 9 groups (including the track coordinators) submitted 32 runs. 

---
# When Vision Meets Texts in Listwise Reranking 

**Authors**: Hongyi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2601.20623)  

**Abstract**: Recent advancements in information retrieval have highlighted the potential of integrating visual and textual information, yet effective reranking for image-text documents remains challenging due to the modality gap and scarcity of aligned datasets. Meanwhile, existing approaches often rely on large models (7B to 32B parameters) with reasoning-based distillation, incurring unnecessary computational overhead while primarily focusing on textual modalities. In this paper, we propose Rank-Nexus, a multimodal image-text document reranker that performs listwise qualitative reranking on retrieved lists incorporating both images and texts. To bridge the modality gap, we introduce a progressive cross-modal training strategy. We first train modalities separately: leveraging abundant text reranking data, we distill knowledge into the text branch. For images, where data is scarce, we construct distilled pairs from multimodal large language model (MLLM) captions on image retrieval benchmarks. Subsequently, we distill a joint image-text reranking dataset. Rank-Nexus achieves outstanding performance on text reranking benchmarks (TREC, BEIR) and the challenging image reranking benchmark (INQUIRE, MMDocIR), using only a lightweight 2B pretrained visual-language model. This efficient design ensures strong generalization across diverse multimodal scenarios without excessive parameters or reasoning overhead. 

---
# Eliminating Hallucination in Diffusion-Augmented Interactive Text-to-Image Retrieval 

**Authors**: Zhuocheng Zhang, Kangheng Liang, Guanxuan Li, Paul Henderson, Richard Mccreadie, Zijun Long  

**Link**: [PDF](https://arxiv.org/pdf/2601.20391)  

**Abstract**: Diffusion-Augmented Interactive Text-to-Image Retrieval (DAI-TIR) is a promising paradigm that improves retrieval performance by generating query images via diffusion models and using them as additional ``views'' of the user's intent. However, these generative views can be incorrect because diffusion generation may introduce hallucinated visual cues that conflict with the original query text. Indeed, we empirically demonstrate that these hallucinated cues can substantially degrade DAI-TIR performance. To address this, we propose Diffusion-aware Multi-view Contrastive Learning (DMCL), a hallucination-robust training framework that casts DAI-TIR as joint optimization over representations of query intent and the target image. DMCL introduces semantic-consistency and diffusion-aware contrastive objectives to align textual and diffusion-generated query views while suppressing hallucinated query signals. This yields an encoder that acts as a semantic filter, effectively mapping hallucinated cues into a null space, improving robustness to spurious cues and better representing the user's intent. Attention visualization and geometric embedding-space analyses corroborate this filtering behavior. Across five standard benchmarks, DMCL delivers consistent improvements in multi-round Hits@10, reaching as high as 7.37\% over prior fine-tuned and zero-shot baselines, which indicates it is a general and robust training framework for DAI-TIR. 

---
# Less is More: Benchmarking LLM Based Recommendation Agents 

**Authors**: Kargi Chauhan, Mahalakshmi Venkateswarlu  

**Link**: [PDF](https://arxiv.org/pdf/2601.20316)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed for personalized product recommendations, with practitioners commonly assuming that longer user purchase histories lead to better predictions. We challenge this assumption through a systematic benchmark of four state of the art LLMs GPT-4o-mini, DeepSeek-V3, Qwen2.5-72B, and Gemini 2.5 Flash across context lengths ranging from 5 to 50 items using the REGEN dataset.
Surprisingly, our experiments with 50 users in a within subject design reveal no significant quality improvement with increased context length. Quality scores remain flat across all conditions (0.17--0.23). Our findings have significant practical implications: practitioners can reduce inference costs by approximately 88\% by using context (5--10 items) instead of longer histories (50 items), without sacrificing recommendation quality. We also analyze latency patterns across providers and find model specific behaviors that inform deployment decisions. This work challenges the existing ``more context is better'' paradigm and provides actionable guidelines for cost effective LLM based recommendation systems. 

---
# One Word is Enough: Minimal Adversarial Perturbations for Neural Text Ranking 

**Authors**: Tanmay Karmakar, Sourav Saha, Debapriyo Majumdar, Surjyanee Halder  

**Link**: [PDF](https://arxiv.org/pdf/2601.20283)  

**Abstract**: Neural ranking models (NRMs) achieve strong retrieval effectiveness, yet prior work has shown they are vulnerable to adversarial perturbations. We revisit this robustness question with a minimal, query-aware attack that promotes a target document by inserting or substituting a single, semantically aligned word - the query center. We study heuristic and gradient-guided variants, including a white-box method that identifies influential insertion points. On TREC-DL 2019/2020 with BERT and monoT5 re-rankers, our single-word attacks achieve up to 91% success while modifying fewer than two tokens per document on average, achieving competitive rank and score boosts with far fewer edits under a comparable white-box setup to ensure fair evaluation against PRADA. We also introduce new diagnostic metrics to analyze attack sensitivity beyond aggregate success rates. Our analysis reveals a Goldilocks zone in which mid-ranked documents are most vulnerable. These findings demonstrate practical risks and motivate future defenses for robust neural ranking. 

---
# MALLOC: Benchmarking the Memory-aware Long Sequence Compression for Large Sequential Recommendation 

**Authors**: Qihang Yu, Kairui Fu, Zhaocheng Du, Yuxuan Si, Kaiyuan Li, Weihao Zhao, Zhicheng Zhang, Jieming Zhu, Quanyu Dai, Zhenhua Dong, Shengyu Zhang, Kun Kuang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.20234)  

**Abstract**: The scaling law, which indicates that model performance improves with increasing dataset and model capacity, has fueled a growing trend in expanding recommendation models in both industry and academia. However, the advent of large-scale recommenders also brings significantly higher computational costs, particularly under the long-sequence dependencies inherent in the user intent of recommendation systems. Current approaches often rely on pre-storing the intermediate states of the past behavior for each user, thereby reducing the quadratic re-computation cost for the following requests. Despite their effectiveness, these methods often treat memory merely as a medium for acceleration, without adequately considering the space overhead it introduces. This presents a critical challenge in real-world recommendation systems with billions of users, each of whom might initiate thousands of interactions and require massive memory for state storage. Fortunately, there have been several memory management strategies examined for compression in LLM, while most have not been evaluated on the recommendation task. To mitigate this gap, we introduce MALLOC, a comprehensive benchmark for memory-aware long sequence compression. MALLOC presents a comprehensive investigation and systematic classification of memory management techniques applicable to large sequential recommendations. These techniques are integrated into state-of-the-art recommenders, enabling a reproducible and accessible evaluation platform. Through extensive experiments across accuracy, efficiency, and complexity, we demonstrate the holistic reliability of MALLOC in advancing large-scale recommendation. Code is available at this https URL. 

---
# Towards End-to-End Alignment of User Satisfaction via Questionnaire in Video Recommendation 

**Authors**: Na Li, Jiaqi Yu, Minzhi Xie, Tiantian He, Xiaoxiao Xu, Zixiu Wang, Lantao Hu, Yongqi Liu, Han Li, Kaiqiao Zhan, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2601.20215)  

**Abstract**: Short-video recommender systems typically optimize ranking models using dense user behavioral signals, such as clicks and watch time. However, these signals are only indirect proxies of user satisfaction and often suffer from noise and bias. Recently, explicit satisfaction feedback collected through questionnaires has emerged as a high-quality direct alignment supervision, but is extremely sparse and easily overwhelmed by abundant behavioral data, making it difficult to incorporate into online recommendation models. To address these challenges, we propose a novel framework which is towards End-to-End Alignment of user Satisfaction via Questionaire, named EASQ, to enable real-time alignment of ranking models with true user satisfaction. Specifically, we first construct an independent parameter pathway for sparse questionnaire signals by combining a multi-task architecture and a lightweight LoRA module. The multi-task design separates sparse satisfaction supervision from dense behavioral signals, preventing the former from being overwhelmed. The LoRA module pre-inject these preferences in a parameter-isolated manner, ensuring stability in the backbone while optimizing user satisfaction. Furthermore, we employ a DPO-based optimization objective tailored for online learning, which aligns the main model outputs with sparse satisfaction signals in real time. This design enables end-to-end online learning, allowing the model to continuously adapt to new questionnaire feedback while maintaining the stability and effectiveness of the backbone. Extensive offline experiments and large-scale online A/B tests demonstrate that EASQ consistently improves user satisfaction metrics across multiple scenarios. EASQ has been successfully deployed in a production short-video recommendation system, delivering significant and stable business gains. 

---
# MERGE: Next-Generation Item Indexing Paradigm for Large-Scale Streaming Recommendation 

**Authors**: Jing Yan, Yimeng Bai, Zongyu Liu, Yahui Liu, Junwei Wang, Jingze Huang, Haoda Li, Sihao Ding, Shaohui Ruan, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.20199)  

**Abstract**: Item indexing, which maps a large corpus of items into compact discrete representations, is critical for both discriminative and generative recommender systems, yet existing Vector Quantization (VQ)-based approaches struggle with the highly skewed and non-stationary item distributions common in streaming industry recommenders, leading to poor assignment accuracy, imbalanced cluster occupancy, and insufficient cluster separation. To address these challenges, we propose MERGE, a next-generation item indexing paradigm that adaptively constructs clusters from scratch, dynamically monitors cluster occupancy, and forms hierarchical index structures via fine-to-coarse merging. Extensive experiments demonstrate that MERGE significantly improves assignment accuracy, cluster uniformity, and cluster separation compared with existing indexing methods, while online A/B tests show substantial gains in key business metrics, highlighting its potential as a foundational indexing approach for large-scale recommendation. 

---
# Taxonomy of the Retrieval System Framework: Pitfalls and Paradigms 

**Authors**: Deep Shah, Sanket Badhe, Nehal Kathrotia  

**Link**: [PDF](https://arxiv.org/pdf/2601.20131)  

**Abstract**: Designing an embedding retrieval system requires navigating a complex design space of conflicting trade-offs between efficiency and effectiveness. This work structures these decisions as a vertical traversal of the system design stack. We begin with the Representation Layer by examining how loss functions and architectures, specifically Bi-encoders and Cross-encoders, define semantic relevance and geometric projection. Next, we analyze the Granularity Layer and evaluate how segmentation strategies like Atomic and Hierarchical chunking mitigate information bottlenecks in long-context documents. Moving to the Orchestration Layer, we discuss methods that transcend the single-vector paradigm, including hierarchical retrieval, agentic decomposition, and multi-stage reranking pipelines to resolve capacity limitations. Finally, we address the Robustness Layer by identifying architectural mitigations for domain generalization failures, lexical blind spots, and the silent degradation of retrieval quality due to temporal drift. By categorizing these limitations and design choices, we provide a comprehensive framework for practitioners to optimize the efficiency-effectiveness frontier in modern neural search systems. 

---
# IMRNNs: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation 

**Authors**: Yash Saxena, Ankur Padia, Kalpa Gunaratna, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2601.20084)  

**Abstract**: Interpretability in black-box dense retrievers remains a central challenge in Retrieval-Augmented Generation (RAG). Understanding how queries and documents semantically interact is critical for diagnosing retrieval behavior and improving model design. However, existing dense retrievers rely on static embeddings for both queries and documents, which obscures this bidirectional relationship. Post-hoc approaches such as re-rankers are computationally expensive, add inference latency, and still fail to reveal the underlying semantic alignment. To address these limitations, we propose Interpretable Modular Retrieval Neural Networks (IMRNNs), a lightweight framework that augments any dense retriever with dynamic, bidirectional modulation at inference time. IMRNNs employ two independent adapters: one conditions document embeddings on the current query, while the other refines the query embedding using corpus-level feedback from initially retrieved documents. This iterative modulation process enables the model to adapt representations dynamically and expose interpretable semantic dependencies between queries and documents. Empirically, IMRNNs not only enhance interpretability but also improve retrieval effectiveness. Across seven benchmark datasets, applying our method to standard dense retrievers yields average gains of +6.35% nDCG, +7.14% recall, and +7.04% MRR over state-of-the-art baselines. These results demonstrate that incorporating interpretability-driven modulation can both explain and enhance retrieval in RAG systems. 

---
# LLaTTE: Scaling Laws for Multi-Stage Sequence Modeling in Large-Scale Ads Recommendation 

**Authors**: Lee Xiong, Zhirong Chen, Rahul Mayuranath, Shangran Qiu, Arda Ozdemir, Lu Li, Yang Hu, Dave Li, Jingtao Ren, Howard Cheng, Fabian Souto Herrera, Ahmed Agiza, Baruch Epshtein, Anuj Aggarwal, Julia Ulziisaikhan, Chao Wang, Dinesh Ramasamy, Parshva Doshi, Sri Reddy, Arnold Overwijk  

**Link**: [PDF](https://arxiv.org/pdf/2601.20083)  

**Abstract**: We present LLaTTE (LLM-Style Latent Transformers for Temporal Events), a scalable transformer architecture for production ads recommendation. Through systematic experiments, we demonstrate that sequence modeling in recommendation systems follows predictable power-law scaling similar to LLMs. Crucially, we find that semantic features bend the scaling curve: they are a prerequisite for scaling, enabling the model to effectively utilize the capacity of deeper and longer architectures. To realize the benefits of continued scaling under strict latency constraints, we introduce a two-stage architecture that offloads the heavy computation of large, long-context models to an asynchronous upstream user model. We demonstrate that upstream improvements transfer predictably to downstream ranking tasks. Deployed as the largest user model at Meta, this multi-stage framework drives a 4.3\% conversion uplift on Facebook Feed and Reels with minimal serving overhead, establishing a practical blueprint for harnessing scaling laws in industrial recommender systems. 

---
# Post-Training Fairness Control: A Single-Train Framework for Dynamic Fairness in Recommendation 

**Authors**: Weixin Chen, Li Chen, Yuhan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.20848)  

**Abstract**: Despite growing efforts to mitigate unfairness in recommender systems, existing fairness-aware methods typically fix the fairness requirement at training time and provide limited post-training flexibility. However, in real-world scenarios, diverse stakeholders may demand differing fairness requirements over time, so retraining for different fairness requirements becomes prohibitive. To address this limitation, we propose Cofair, a single-train framework that enables post-training fairness control in recommendation. Specifically, Cofair introduces a shared representation layer with fairness-conditioned adapter modules to produce user embeddings specialized for varied fairness levels, along with a user-level regularization term that guarantees user-wise monotonic fairness improvements across these levels. We theoretically establish that the adversarial objective of Cofair upper bounds demographic parity and the regularization term enforces progressive fairness at user level. Comprehensive experiments on multiple datasets and backbone models demonstrate that our framework provides dynamic fairness at different levels, delivering comparable or better fairness-accuracy curves than state-of-the-art baselines, without the need to retrain for each new fairness requirement. Our code is publicly available at this https URL. 

---
# $\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval 

**Authors**: Zihao Wang, Hang Yin, Lihui Liu, Hanghang Tong, Yangqiu Song, Ginny Wong, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2601.20844)  

**Abstract**: This paper studies the minimal dimension required to embed subset memberships ($m$ elements and ${m\choose k}$ subsets of at most $k$ elements) into vector spaces, denoted as Minimal Embeddable Dimension (MED). The tight bounds of MED are derived theoretically and supported empirically for various notions of "distances" or "similarities," including the $\ell_2$ metric, inner product, and cosine similarity. In addition, we conduct numerical simulation in a more achievable setting, where the ${m\choose k}$ subset embeddings are chosen as the centroid of the embeddings of the contained elements. Our simulation easily realizes a logarithmic dependency between the MED and the number of elements to embed. These findings imply that embedding-based retrieval limitations stem primarily from learnability challenges, not geometric constraints, guiding future algorithm design. 

---
# TGSBM: Transformer-Guided Stochastic Block Model for Link Prediction 

**Authors**: Zhejian Yang, Songwei Zhao, Zilin Zhao, Hechang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.20646)  

**Abstract**: Link prediction is a cornerstone of the Web ecosystem, powering applications from recommendation and search to knowledge graph completion and collaboration forecasting. However, large-scale networks present unique challenges: they contain hundreds of thousands of nodes and edges with heterogeneous and overlapping community structures that evolve over time. Existing approaches face notable limitations: traditional graph neural networks struggle to capture global structural dependencies, while recent graph transformers achieve strong performance but incur quadratic complexity and lack interpretable latent structure. We propose \textbf{TGSBM} (Transformer-Guided Stochastic Block Model), a framework that integrates the principled generative structure of Overlapping Stochastic Block Models with the representational power of sparse Graph Transformers. TGSBM comprises three main components: (i) \emph{expander-augmented sparse attention} that enables near-linear complexity and efficient global mixing, (ii) a \emph{neural variational encoder} that infers structured posteriors over community memberships and strengths, and (iii) a \emph{neural edge decoder} that reconstructs links via OSBM's generative process, preserving interpretability. Experiments across diverse benchmarks demonstrate competitive performance (mean rank 1.6 under HeaRT protocol), superior scalability (up to $6\times$ faster training), and interpretable community structures. These results position TGSBM as a practical approach that strikes a balance between accuracy, efficiency, and transparency for large-scale link prediction. 

---
# On Every Note a Griff: Looking for a Useful Representation of Basso Continuo Performance Style 

**Authors**: Adam Štefunko, Carlos Eduardo Cancino-Chacón, Jan Hajič jr  

**Link**: [PDF](https://arxiv.org/pdf/2601.20478)  

**Abstract**: Basso continuo is a baroque improvisatory accompaniment style which involves improvising multiple parts above a given bass line in a musical score on a harpsichord or organ. Basso continuo is not merely a matter of history; moreover, it is a historically inspired living practice, and The Aligned Continuo Dataset (ACoRD) records the first sample of modern-day basso continuo playing in the symbolic domain. This dataset, containing 175 MIDI recordings of 5 basso continuo scores performed by 7 players, allows us to start observing and analyzing the variety that basso continuo improvisation brings. A recently proposed basso continuo performance-to-score alignment system provides a way of mapping improvised performance notes to score notes. In order to study aligned basso continuo performances, we need an appropriate feature representation. We propose griff, a representation inspired by historical basso continuo treatises. It enables us to encode both pitch content and structure of a basso continuo realization in a transposition-invariant way. Griffs are directly extracted from aligned basso continuo performances by grouping together performance notes aligned to the same score note in a onset-time ordered way, and they provide meaningful tokens that form a feature space in which we can analyze basso continuo performance styles. We statistically describe griffs extracted from the ACoRD dataset recordings, and show in two experiments how griffs can be used for statistical analysis of individuality of different players' basso continuo performance styles. We finally present an argument why it is desirable to preserve the structure of a basso continuo improvisation in order to conduct a refined analysis of personal performance styles of individual basso continuo practitioners, and why griffs can provide a meaningful historically informed feature space worthy of a more robust empirical validation. 

---
# High-Resolution Mapping of Port Dynamics from Open-Access AIS Data in Tokyo Bay 

**Authors**: Moritz Hütten  

**Link**: [PDF](https://arxiv.org/pdf/2601.20211)  

**Abstract**: Knowledge about vessel activity in port areas and around major industrial zones provides insights into economic trends, supports decision-making for shipping and port operators, and contributes to maritime safety. Vessel data from terrestrial receivers of the Automatic Identification System (AIS) have become increasingly openly available, and we demonstrate that such data can be used to infer port activities at high resolution and with precision comparable to official statistics. We analyze open-access AIS data from a three-month period in 2024 for Tokyo Bay, located in Japan's most densely populated urban region. Accounting for uneven data coverage, we reconstruct vessel activity in Tokyo Bay at $\sim\,$30~m resolution and identify 161 active berths across seven major port areas in the bay. During the analysis period, we find an average of $35\pm17_{\text{stat}}$ vessels moving within the bay at any given time, and $293\pm22_{\text{stat}}+65_{\text{syst}}-10_{\text{syst}}$ vessels entering or leaving the bay daily, with an average gross tonnage of $11{,}860^{+280}_{-\;\,50}$. These figures indicate an accelerating long-term trend toward fewer but larger vessels in Tokyo Bay's commercial traffic. Furthermore, we find that in dense urban environments, radio shadows in vessel AIS data can reveal the precise locations of inherently passive receiver stations. 

---
# Look in the Middle: Structural Anchor Pruning for Scalable Visual RAG Indexing 

**Authors**: Zhuchenyang Liu, Ziyu Hu, Yao Zhang, Yu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2601.20107)  

**Abstract**: Recent Vision-Language Models (e.g., ColPali) enable fine-grained Visual Document Retrieval (VDR) but incur prohibitive index vector size overheads. Training-free pruning solutions (e.g., EOS-attention based methods) can reduce index vector size by approximately 60% without model adaptation, but often underperform random selection in high-compression scenarios (> 80%). Prior research (e.g., Light-ColPali) attributes this to the conclusion that visual token importance is inherently query-dependent, thereby questioning the feasibility of training-free pruning. In this work, we propose Structural Anchor Pruning (SAP), a training-free pruning method that identifies key visual patches from middle layers to achieve high performance compression. We also introduce Oracle Score Retention (OSR) protocol to evaluate how layer-wise information affects compression efficiency. Evaluations on the ViDoRe benchmark demonstrate that SAP reduces index vectors by over 90% while maintaining robust retrieval fidelity, providing a highly scalable solution for Visual RAG. Furthermore, our OSR-based analysis reveals that semantic structural anchor patches persist in the middle layers, unlike traditional pruning solutions that focus on the final layer where structural signals dissipate. 

---
