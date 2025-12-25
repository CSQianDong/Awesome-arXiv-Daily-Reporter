# ReaSeq: Unleashing World Knowledge via Reasoning for Sequential Modeling 

**Authors**: Chuan Wang, Gaoming Yang, Han Wu, Jiakai Tang, Jiahao Yu, Jian Wu, Jianwu Hu, Junjun Zheng, Shuwen Xiao, Yeqiu Yang, Yuning Jiang, Ahjol Nurlanbek, Binbin Cao, Bo Zheng, Fangmei Zhu, Gaoming Zhou, Huimin Yi, Huiping Chu, Jin Huang, Jinzhe Shan, Kenan Cui, Longbin Li, Silu Zhou, Wen Chen, Xia Ming, Xiang Gao, Xin Yao, Xingyu Wen, Yan Zhang, Yiwen Hu, Yulin Wang, Ziheng Bao, Zongyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21257)  

**Abstract**: Industrial recommender systems face two fundamental limitations under the log-driven paradigm: (1) knowledge poverty in ID-based item representations that causes brittle interest modeling under data sparsity, and (2) systemic blindness to beyond-log user interests that constrains model performance within platform boundaries. These limitations stem from an over-reliance on shallow interaction statistics and close-looped feedback while neglecting the rich world knowledge about product semantics and cross-domain behavioral patterns that Large Language Models have learned from vast corpora.
To address these challenges, we introduce ReaSeq, a reasoning-enhanced framework that leverages world knowledge in Large Language Models to address both limitations through explicit and implicit reasoning. Specifically, ReaSeq employs explicit Chain-of-Thought reasoning via multi-agent collaboration to distill structured product knowledge into semantically enriched item representations, and latent reasoning via Diffusion Large Language Models to infer plausible beyond-log behaviors. Deployed on Taobao's ranking system serving hundreds of millions of users, ReaSeq achieves substantial gains: >6.0% in IPV and CTR, >2.9% in Orders, and >2.5% in GMV, validating the effectiveness of world-knowledge-enhanced reasoning over purely log-driven approaches. 

---
# Blurb-Refined Inference from Crowdsourced Book Reviews using Hierarchical Genre Mining with Dual-Path Graph Convolutions 

**Authors**: Suraj Kumar, Utsav Kumar Nareti, Soumi Chattopadhyay, Chandranath Adak, Prolay Mallick  

**Link**: [PDF](https://arxiv.org/pdf/2512.21076)  

**Abstract**: Accurate book genre classification is fundamental to digital library organization, content discovery, and personalized recommendation. Existing approaches typically model genre prediction as a flat, single-label task, ignoring hierarchical genre structure and relying heavily on noisy, subjective user reviews, which often degrade classification reliability. We propose HiGeMine, a two-phase hierarchical genre mining framework that robustly integrates user reviews with authoritative book blurbs. In the first phase, HiGeMine employs a zero-shot semantic alignment strategy to filter reviews, retaining only those semantically consistent with the corresponding blurb, thereby mitigating noise, bias, and irrelevance. In the second phase, we introduce a dual-path, two-level graph-based classification architecture: a coarse-grained Level-1 binary classifier distinguishes fiction from non-fiction, followed by Level-2 multi-label classifiers for fine-grained genre prediction. Inter-genre dependencies are explicitly modeled using a label co-occurrence graph, while contextual representations are derived from pretrained language models applied to the filtered textual content. To facilitate systematic evaluation, we curate a new hierarchical book genre dataset. Extensive experiments demonstrate that HiGeMine consistently outperformed strong baselines across hierarchical genre classification tasks. The proposed framework offers a principled and effective solution for leveraging both structured and unstructured textual data in hierarchical book genre analysis. 

---
# Agentic Multi-Persona Framework for Evidence-Aware Fake News Detection 

**Authors**: Roopa Bukke, Soumya Pandey, Suraj Kumar, Soumi Chattopadhyay, Chandranath Adak  

**Link**: [PDF](https://arxiv.org/pdf/2512.21039)  

**Abstract**: The rapid proliferation of online misinformation poses significant risks to public trust, policy, and safety, necessitating reliable automated fake news detection. Existing methods often struggle with multimodal content, domain generalization, and explainability. We propose AMPEND-LS, an agentic multi-persona evidence-grounded framework with LLM-SLM synergy for multimodal fake news detection. AMPEND-LS integrates textual, visual, and contextual signals through a structured reasoning pipeline powered by LLMs, augmented with reverse image search, knowledge graph paths, and persuasion strategy analysis. To improve reliability, we introduce a credibility fusion mechanism combining semantic similarity, domain trustworthiness, and temporal context, and a complementary SLM classifier to mitigate LLM uncertainty and hallucinations. Extensive experiments across three benchmark datasets demonstrate that AMPEND-LS consistently outperformed state-of-the-art baselines in accuracy, F1 score, and robustness. Qualitative case studies further highlight its transparent reasoning and resilience against evolving misinformation. This work advances the development of adaptive, explainable, and evidence-aware systems for safeguarding online information integrity. 

---
# Towards Better Search with Domain-Aware Text Embeddings for C2C Marketplaces 

**Authors**: Andre Rusli, Miao Cao, Shoma Ishimoto, Sho Akiyama, Max Frenzel  

**Link**: [PDF](https://arxiv.org/pdf/2512.21021)  

**Abstract**: Consumer-to-consumer (C2C) marketplaces pose distinct retrieval challenges: short, ambiguous queries; noisy, user-generated listings; and strict production constraints. This paper reports our experiment to build a domain-aware Japanese text-embedding approach to improve the quality of search at Mercari, Japan's largest C2C marketplace. We experimented with fine-tuning on purchase-driven query-title pairs, using role-specific prefixes to model query-item asymmetry. To meet production constraints, we apply Matryoshka Representation Learning to obtain compact, truncation-robust embeddings. Offline evaluation on historical search logs shows consistent gains over a strong generic encoder, with particularly large improvements when replacing PCA compression with Matryoshka truncation. A manual assessment further highlights better handling of proper nouns, marketplace-specific semantics, and term-importance alignment. Additionally, an initial online A/B test demonstrates statistically significant improvements in revenue per user and search-flow efficiency, with transaction frequency maintained. Results show that domain-aware embeddings improve relevance and efficiency at scale and form a practical foundation for richer LLM-era search experiences. 

---
# MMSRARec: Summarization and Retrieval Augumented Sequential Recommendation Based on Multimodal Large Language Model 

**Authors**: Haoyu Wang, Yitong Wang, Jining Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20916)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated significant potential in recommendation systems. However, the effective application of MLLMs to multimodal sequential recommendation remains unexplored: A) Existing methods primarily leverage the multimodal semantic understanding capabilities of pre-trained MLLMs to generate item embeddings or semantic IDs, thereby enhancing traditional recommendation models. These approaches generate item representations that exhibit limited interpretability, and pose challenges when transferring to language model-based recommendation systems. B) Other approaches convert user behavior sequence into image-text pairs and perform recommendation through multiple MLLM inference, incurring prohibitive computational and time costs. C) Current MLLM-based recommendation systems generally neglect the integration of collaborative signals. To address these limitations while balancing recommendation performance, interpretability, and computational cost, this paper proposes MultiModal Summarization-and-Retrieval-Augmented Sequential Recommendation. Specifically, we first employ MLLM to summarize items into concise keywords and fine-tune the model using rewards that incorporate summary length, information loss, and reconstruction difficulty, thereby enabling adaptive adjustment of the summarization policy. Inspired by retrieval-augmented generation, we then transform collaborative signals into corresponding keywords and integrate them as supplementary context. Finally, we apply supervised fine-tuning with multi-task learning to align the MLLM with the multimodal sequential recommendation. Extensive evaluations on common recommendation datasets demonstrate the effectiveness of MMSRARec, showcasing its capability to efficiently and interpretably understand user behavior histories and item information for accurate recommendations. 

---
# Accurate and Diverse Recommendations via Propensity-Weighted Linear Autoencoders 

**Authors**: Kazuma Onishi, Katsuhiko Hayashi, Hidetaka Kamigaito  

**Link**: [PDF](https://arxiv.org/pdf/2512.20896)  

**Abstract**: In real-world recommender systems, user-item interactions are Missing Not At Random (MNAR), as interactions with popular items are more frequently observed than those with less popular ones. Missing observations shift recommendations toward frequently interacted items, which reduces the diversity of the recommendation list. To alleviate this problem, Inverse Propensity Scoring (IPS) is widely used and commonly models propensities based on a power-law function of item interaction frequency. However, we found that such power-law-based correction overly penalizes popular items and harms their recommendation performance. We address this issue by redefining the propensity score to allow broader item recommendation without excessively penalizing popular items. The proposed score is formulated by applying a sigmoid function to the logarithm of the item observation frequency, maintaining the simplicity of power-law scoring while allowing for more flexible adjustment. Furthermore, we incorporate the redefined propensity score into a linear autoencoder model, which tends to favor popular items, and evaluate its effectiveness. Experimental results revealed that our method substantially improves the diversity of items in the recommendation list without sacrificing recommendation accuracy. 

---
# Soft Filtering: Guiding Zero-shot Composed Image Retrieval with Prescriptive and Proscriptive Constraints 

**Authors**: Youjin Jung, Seongwoo Cho, Hyun-seok Min, Sungchul Choi  

**Link**: [PDF](https://arxiv.org/pdf/2512.20781)  

**Abstract**: Composed Image Retrieval (CIR) aims to find a target image that aligns with user intent, expressed through a reference image and a modification text. While Zero-shot CIR (ZS-CIR) methods sidestep the need for labeled training data by leveraging pretrained vision-language models, they often rely on a single fused query that merges all descriptive cues of what the user wants, tending to dilute key information and failing to account for what they wish to avoid. Moreover, current CIR benchmarks assume a single correct target per query, overlooking the ambiguity in modification texts. To address these challenges, we propose Soft Filtering with Textual constraints (SoFT), a training-free, plug-and-play filtering module for ZS-CIR. SoFT leverages multimodal large language models (LLMs) to extract two complementary constraints from the reference-modification pair: prescriptive (must-have) and proscriptive (must-avoid) constraints. These serve as semantic filters that reward or penalize candidate images to re-rank results, without modifying the base retrieval model or adding supervision. In addition, we construct a two-stage dataset pipeline that refines CIR benchmarks. We first identify multiple plausible targets per query to construct multi-target triplets, capturing the open-ended nature of user intent. Then guide multimodal LLMs to rewrite the modification text to focus on one target, while referencing contrastive distractors to ensure precision. This enables more comprehensive and reliable evaluation under varying ambiguity levels. Applied on top of CIReVL, a ZS-CIR retriever, SoFT raises R@5 to 65.25 on CIRR (+12.94), mAP@50 to 27.93 on CIRCO (+6.13), and R@50 to 58.44 on FashionIQ (+4.59), demonstrating broad effectiveness. 

---
# ClarifyMT-Bench: Benchmarking and Improving Multi-Turn Clarification for Conversational Large Language Models 

**Authors**: Sichun Luo, Yi Huang, Mukai Li, Shichang Meng, Fengyuan Liu, Zefa Hu, Junlan Feng, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21120)  

**Abstract**: Large language models (LLMs) are increasingly deployed as conversational assistants in open-domain, multi-turn settings, where users often provide incomplete or ambiguous information. However, existing LLM-focused clarification benchmarks primarily assume single-turn interactions or cooperative users, limiting their ability to evaluate clarification behavior in realistic settings. We introduce \textbf{ClarifyMT-Bench}, a benchmark for multi-turn clarification grounded in a five-dimensional ambiguity taxonomy and a set of six behaviorally diverse simulated user personas. Through a hybrid LLM-human pipeline, we construct 6,120 multi-turn dialogues capturing diverse ambiguity sources and interaction patterns. Evaluating ten representative LLMs uncovers a consistent under-clarification bias: LLMs tend to answer prematurely, and performance degrades as dialogue depth increases. To mitigate this, we propose \textbf{ClarifyAgent}, an agentic approach that decomposes clarification into perception, forecasting, tracking, and planning, substantially improving robustness across ambiguity conditions. ClarifyMT-Bench establishes a reproducible foundation for studying when LLMs should ask, when they should answer, and how to navigate ambiguity in real-world human-LLM interactions. 

---
# MultiMind at SemEval-2025 Task 7: Crosslingual Fact-Checked Claim Retrieval via Multi-Source Alignment 

**Authors**: Mohammad Mahdi Abootorabi, Alireza Ghahramani Kure, Mohammadali Mohammadkhani, Sina Elahimanesh, Mohammad Ali Ali Panah  

**Link**: [PDF](https://arxiv.org/pdf/2512.20950)  

**Abstract**: This paper presents our system for SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval. In an era where misinformation spreads rapidly, effective fact-checking is increasingly critical. We introduce TriAligner, a novel approach that leverages a dual-encoder architecture with contrastive learning and incorporates both native and English translations across different modalities. Our method effectively retrieves claims across multiple languages by learning the relative importance of different sources in alignment. To enhance robustness, we employ efficient data preprocessing and augmentation using large language models while incorporating hard negative sampling to improve representation learning. We evaluate our approach on monolingual and crosslingual benchmarks, demonstrating significant improvements in retrieval accuracy and fact-checking performance over baselines. 

---
# How important is Recall for Measuring Retrieval Quality? 

**Authors**: Shelly Schwartz, Oleg Vasilyev, Randy Sawaya  

**Link**: [PDF](https://arxiv.org/pdf/2512.20854)  

**Abstract**: In realistic retrieval settings with large and evolving knowledge bases, the total number of documents relevant to a query is typically unknown, and recall cannot be computed. In this paper, we evaluate several established strategies for handling this limitation by measuring the correlation between retrieval quality metrics and LLM-based judgments of response quality, where responses are generated from the retrieved documents. We conduct experiments across multiple datasets with a relatively low number of relevant documents (2-15). We also introduce a simple retrieval quality measure that performs well without requiring knowledge of the total number of relevant documents. 

---
# MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation 

**Authors**: Chi-Hsiang Hsiao, Yi-Cheng Wang, Tzung-Sheng Lin, Yi-Ren Yeh, Chu-Song Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.20626)  

**Abstract**: Retrieval-augmented generation (RAG) enables large language models (LLMs) to dynamically access external information, which is powerful for answering questions over previously unseen documents. Nonetheless, they struggle with high-level conceptual understanding and holistic comprehension due to limited context windows, which constrain their ability to perform deep reasoning over long-form, domain-specific content such as full-length books. To solve this problem, knowledge graphs (KGs) have been leveraged to provide entity-centric structure and hierarchical summaries, offering more structured support for reasoning. However, existing KG-based RAG solutions remain restricted to text-only inputs and fail to leverage the complementary insights provided by other modalities such as vision. On the other hand, reasoning from visual documents requires textual, visual, and spatial cues into structured, hierarchical concepts. To address this issue, we introduce a multimodal knowledge graph-based RAG that enables cross-modal reasoning for better content understanding. Our method incorporates visual cues into the construction of knowledge graphs, the retrieval phase, and the answer generation process. Experimental results across both global and fine-grained question answering tasks show that our approach consistently outperforms existing RAG-based approaches on both textual and multimodal corpora. 

---
