# ViHoRec: A Quality-Controlled Vietnamese Hotel Recommendation Dataset and Cold-Start Benchmark 

**Authors**: Minh Hoang Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2607.12946)  

**Abstract**: Recommender-system research for Vietnamese remains limited by the absence of a public, well-documented hotel interaction resource. Building such a resource is challenging for three reasons: cross-platform hotel names must be reconciled before interactions are comparable; quality must be audited with reproducible metrics rather than ad hoc cleaning; and public release must preserve privacy while remaining benchmarkable under realistic cold-start conditions. We introduce ViHoRec, a quality-controlled Vietnamese hotel recommendation dataset of 18{,}267 interactions between 6{,}832 users and 560 hotels, crawled from this http URL, Traveloka, and Ivivu. Our contributions are: (i) a reproducible construction pipeline with cross-platform entity resolution and quantitative quality control; (ii) a privacy-preserving release with HMAC pseudonyms; and (iii) a public cold-start benchmark with temporal leave-last-one-out split, data-centric ablations, and dependency-free baselines. On the public split, learned models degrade sharply for users with short histories (BPR-MF Recall@10: 0.065 vs. 0.120), while UserKNN remains strongest overall, establishing ViHoRec as a sparse, cold-start-dominated testbed for low-resource recommendation. All data are publicly available at this https URL. 

---
# RecRec: Latent Interests Recursive Reasoning for Sequential Recommendation 

**Authors**: Wenhao Deng, Junchen Fu, Hanwen Du, Alexandros Karatzoglou, Ioannis Arapakis, Hangjun Guo, Kaiwen Zheng, Yongxin Ni, Joemon M. Jose  

**Link**: [PDF](https://arxiv.org/pdf/2607.12945)  

**Abstract**: Sequential recommender systems rely on a single forward pass to encode user interaction histories and predict the next item. Increasing inference-time computation through latent reasoning, with the model proceeding step by step before the final prediction, has been recently explored in sequential recommendation with promising results. However, how to structure the reasoning process for sequential recommendation remains an open question. Existing approaches couple reasoning and prediction in a single $d$-dimensional state, limiting reasoning depth and often relying on multi-stage pipelines with reinforcement learning (RL). We propose RecRec (Recursive Reasoning for Recommendation), an RL-free framework that decouples reasoning from prediction, overcoming the fixed $d$-dimensional state bottleneck of prior methods. RecRec consists of a Context Compressor and a Recursive Reasoner, trained in two simple supervised stages. The Context Compressor distills the backbone's hidden states into a small set of latent interests, with an Interest Diversity Regularizer encouraging each interest to capture a distinct aspect of user behavior. The Recursive Reasoner then refines these interests by reasoning in a separate intermediate latent space. Deep supervision lets the reasoning depth be freely adjusted at inference without retraining. On four real-world datasets, RecRec outperforms state-of-the-art reasoning-enhanced methods, and on three of four datasets, gains extend past the training-time depth. Our findings point to a decoupled, multi-vector recipe that unleashes latent reasoning from the single-state bottleneck of prior methods, suggesting reasoning-state structure as a design axis to explore further in sequential recommendation. 

---
# Learning to Forget: Satiation-Aware Long-Sequence Transducers for Mitigating Post-Purchase Redundancy 

**Authors**: Yipin Dai, Ruocong Tang, Xing Fang, Yang Huang, Jing Wang, Zhentao Song, He Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.12714)  

**Abstract**: Sequential recommendation models predominantly interpret user interactions as positive signals for preference accumulation. However, in e-commerce scenarios, a purchase action often signifies the termination of a specific intent ("Interest Exit") rather than its continuation. Existing models overlook this distinction, suffering from Action-Intent Asymmetry, which leads to severe post-purchase redundancy. In this paper, we propose the Satiation-Aware Mechanism (SAM), an end-to-end framework designed to explicitly model the lifecycle of user interests. SAM incorporates three key components: (1) A Dual-path Cross-Attention architecture that retroactively suppresses historical clicks associated with a fulfilled intent while simultaneously retrieving personalized replenishment rhythms from long-term purchase history; (2) An Adaptive Satiation Gating Unit (ASGU) that generates a time-sensitive soft mask to inhibit satisfied interests immediately after purchase and gradually "re-awaken" them as the predicted repurchase cycle approaches; and (3) A self-supervised Time-to-Next-Purchase (TTNP) auxiliary task to learn latent product lifecycles without manual annotation. Extensive offline experiments on industrial datasets and online A/B testing demonstrate that SAM significantly reduces the Post-Purchase Repeat Rate (PPRR) by over 60%. 

---
# Cheaper is Better: A Discount-Aware Network for Conversion Rate Prediction in E-commerce Recommendation System 

**Authors**: Ruocong Tang, Yang Huang, Xing Fang, Chenyi Yan, Chuike Sun, Jing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.12578)  

**Abstract**: Post-click conversion rate (CVR) is a crucial element in online recommendation systems, which addresses significant challenges such as data sparsity (DS), sample selection bias (SSB), and delayed feedback. However, the impact of item discount rate-a key factor influencing both pricing and user purchasing behavior, has received limited attention. In this paper, we introduce the Discount-Aware Network (DANet) to model the relationship between item discount rates and CVR. DANet comprises three main components: 1) a time-frequency transformation module that utilizes Fourier transform to derive the frequency spectrum and capture the long-term discount rate trends of items; 2) a distribution de-bias module designed to mitigate the biases in user-specific discount rates caused by various purchase combinations and promotional activities, as well as periodic deviations linked to different promotion periods on e-commerce platforms; and 3) a supervised regression auxiliary task that establishes the explicit item discount labels to enhance the model's performance in terms of value accuracy, facilitating an effective representation of item discount rates. Experimental results on real datasets demonstrate the superiority of DANet, with offline AUC improving by 1.61%, and online A/B test also shows that DANet achieves impressive gains of 3.63% on pCVR and 2.23% on GMV. DANet has been successfully deployed on Alibaba Tmall APP. The code is available at this https URL. 

---
# Where Reasoning Matters: Rethinking Latent Reasoning in Semantic ID-based Generative Recommendation 

**Authors**: Shangxin Yang, Min Gao, Zongwei Wang, Junliang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.12425)  

**Abstract**: Semantic ID-based generative recommendation predicts an item by generating a short sequence of semantic ID tokens, where each token is produced autoregressively. Latent reasoning has recently been introduced to improve this process through additional hidden-state computation before each token decision. This raises a practical question: when one item is represented by a sequence of semantic ID tokens, should each token receive the same fixed number of latent refinement steps, or should these steps be allocated more effectively across positions? We study this question through position-wise information-gain (IG), which measures how much each semantic ID position reduces the uncertainty of the target item. We observe that earlier semantic ID positions usually provide higher information-gain, while later positions contribute less additional information. We further analyze that applying more refinement to high-IG positions tends to bring larger expected benefits. Based on this observation, we propose IBA, an Information-Gain Budget Allocation framework for semantic ID-based generative recommendation. IBA treats latent refinement steps as a limited computational resource and learns how to allocate them across semantic ID positions, assigning more refinement to informative positions and less to positions with smaller contribution. Experiments on multiple public datasets show that IBA consistently improves strong generative recommendation baselines and achieves a better accuracy--computation trade-off than fixed or poorly matched step allocations. 

---
# MESH: Scaling Up Retrieval with Heterogeneous Content Unification 

**Authors**: Jiaxing Qu, Yilin Chen, Junpeng Hou, Jinfeng Rao, Olafur Gudmundsson, Sai Xiao, Huizhong Duan  

**Link**: [PDF](https://arxiv.org/pdf/2607.12392)  

**Abstract**: Optimizing large-scale retrieval hinges on the ability to efficiently surface candidates across diverse content tiers. However, to capture segments such as fresh and long-tail content, modern systems typically resort to a fragmented "zoo" of specialized retrieval models. This operational complexity is attributed to a fundamental challenge in heterogeneous retrieval systems, the Scaling Bias of Heterogeneity, where model capacity gains do not apply equally across diverse content tiers. To bridge this gap, we propose MESH as a unified retrieval scaling framework that mitigates this bias through a modularized architecture integrated with gated bias correction. By partitioning the feature space into independent domains, MESH enforces a structural inductive bias that reduces interference between sparse-item signals and high-frequency engagement features. This protected gradient path leads to improved scaling behavior for sparse content, empirically validated by a 14 times improvement in the power-law scaling exponent for fresh items. In online evaluations on Pinterest's Related Pins platform, a billion scale item-to-item recommendation system, these improvements translate into a +5.5% lift in fresh-item repins, alongside with 55% improvement in funnel efficiency and +0.46% improvement in user retention. Finally, our asynchronous serving strategy ensures production viability by delivering a 2.87 times improvement in system throughput. Our findings suggest MESH as a promising paradigm for consolidating fragmented retrieval infrastructures into more scalable and ecosystem-aware backbones. 

---
# SlimPer: Make Personalization Model Slim and Smart 

**Authors**: Siqi Wang, Xianjie Chen, Shaofeng Deng, Albert Chen, Romil Shah, Jiawei Huang, Zhaoqin Wang, Zhang Zhang, Yiqun Liu, Meilei Jiang, Anish Dubey, Moyan Mei, Tongxin Wang, Nathan Berrebbi, Misael Manjarres, Armand Sauzay, Shardul Kothapalli, Aryaman Vinchhi, Kevin Johnstone, Juheon Lee, Gufan Yin, Ziheng Huang, Justin Lin, Mert Terzihan, Yilin Qi, Cynthia Yang, Colin Peppler, Qi Ding, Ruohan Sun, Ge Song, Litao Deng, Parichay Kapoor, Matt Ma, Huihui Cheng, Jiyuan Zhang, Yanli Zhao, Yiping Han, Fangqiu Han, Ning Yao, Arun Singh, Jordan Edwards, Zhengyu Su, Abhishek Kumar, Guangdeng Liao, Ankit Asthana  

**Link**: [PDF](https://arxiv.org/pdf/2607.12281)  

**Abstract**: Transformer-style architectures are increasingly adopted for industrial recommendation systems, yet they inherit a design premise misaligned with the task: generative models rely on per-token autoregressive prediction, which justifies maintaining large intermediate tensors that scale with sequence length. In contrast, recommendation systems produce a single set of relevance scores for each <user, item> pair without token-level supervision. Leveraging this observation, we propose SlimPer, which reformulates personalized ranking as iterative refinement of a compact, unified <user, item> knowledge base. At each layer, the model selectively queries raw multi-modal user-side tokens, computes explicit relevance matching scores, and refines the knowledge base, all in O(N) per-layer cost with a fixed-size intermediate representation. As a result, model depth is decoupled from user history length, enabling deeper relevance understanding without proportional growth in compute or memory; request-only optimization further trims memory by sharing a single copy of user-side tokens across all candidate items. SlimPer unifies sparse, dense, and sequence features within a single backbone and provides inherent interpretability through its attention mechanism. Deployed on Instagram Reels and Feed, SlimPer yields measurable improvements in user engagement while streamlining the overall system and enabling effective modeling of 10k+ fine-grained user history events. 

---
# Not Only NTP: Extending Training Signal Coverage for Generative Recommendation 

**Authors**: Changhao Li, Shuli Wang, Junwei Yin, Senjie Kou, Yinqiu Huang, Chi Wang, Yinhua Zhu, Haitao Wang, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.12277)  

**Abstract**: Next-Token Prediction (NTP) carries two structural training signal limitations. First, NTP optimizes for single-step prediction only, placing no supervised pressure on learning longer-range behavioral structure -- we term this \textbf{temporal locality}. Second, in multi-domain sequences, each target item embedding receives gradient updates exclusively from the immediately preceding hidden state, with no explicit gradient pathway from cross-domain context -- we term this \textbf{spatial locality}.
We propose \textbf{NONTP}, extending NTP's signal coverage along both dimensions through two auxiliary objectives. \textbf{TCL (Temporal Contrastive Learning)} uses a BYOL-style EMA teacher with InfoNCE to align hidden states against a $K$-step future trajectory in representation space. \textbf{TDL (Trans-Domain Learning)} mean-pools cross-domain hidden states and predicts through the shared prediction head, opening a second gradient pathway with no additional parameters. Both are discarded at inference: zero overhead.
On a four-domain Meituan industrial dataset (full ranking), NONTP achieves HR@10 +34.3\% over NTP and +18.3\% over MBGR. On the public Amazon Movie-Book-CDs benchmark, HR@10 +2.8\% and NDCG@10 +3.7\%. Online A/B tests confirm CTR +1.8\% and GMV +2.1\% (both $p < 0.01$). Ablation studies confirm each component contributes independently, with gradient conflict analyzed as a direction for future work. 

---
# SHEAF: Self-profiled Hardness Estimation from Answer-set Flux for Predicting Query Hardness in Graph-based ANN Search 

**Authors**: Dongfang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.12229)  

**Abstract**: Graph-based approximate nearest neighbor (ANN) search is usually governed by a beam-width parameter that trades recall for throughput and is fixed for the whole workload. Yet, queries may not be equally hard: for example, on the widely used data set SIFT1M, the beam that a query needs to reach 95\% recall varies by more than $32\times$. Therefore, serving each query at its own width would help if the system could tell, cheaply and in advance, how hard it is. The prevailing proxy for this difficulty is called local intrinsic dimensionality (LID); however, LID is static and geometric, which makes it only weakly predict the minimum beam.
This paper presents a new measure, namely Self-profiled Hardness Estimation from Answer-set Flux (SHEAF), which represents a query's hardness as how much its own top-$k$ answer set changes between two shallow probe widths. We design a self-profiling estimator that turns this flux into a deployable per-query beam predictor; furthermore, we develop a fixed-probe evaluation protocol that scores each measure over all queries with an observed minimum sufficient beam. On popular ANN indexes such as CAGRA and HNSW across four diverse data sets, SHEAF predicts the per-query beam better than five baseline measures on both GPU and CPU by up to $1.55\times$ in held-out correlation, using only two shallow probe searches and no query-time ground truth. 

---
# Explaining When PRF Fails: Participatory Auditing for Selective Query Expansion 

**Authors**: Zeyan Liang, Graham McDonald, Iadh Ounis  

**Link**: [PDF](https://arxiv.org/pdf/2607.12098)  

**Abstract**: Pseudo-Relevance Feedback (PRF) improves retrieval effectiveness on average, but harms a substantial fraction of queries through query drift, an asymmetry hidden by aggregate offline metrics. Existing Selective PRF (sPRF) approaches typically rely on Query Performance Prediction (QPP) methods derived from the same ranking statistics, and therefore inherit, rather than resolve, this opacity. We argue that this is a core explainability problem in IR, and propose a two-stage audit-then-automate framework. In Stage 1, a participatory audit with 108 users across 43 TREC Deep Learning 2019 queries shows that only 20.9% of queries benefit from PRF, while 25.6% suffer a degraded user experience, and that avoiding harm is nearly twice as valuable as exploiting successful expansion. In Stage 2, we repurpose LLM-based rerankers as system preference predictors that replicate these user-derived labels automatically, grounded in inspectable document evidence. Together, the two stages explain which queries PRF harms, why an sPRF decision is made, and how the decision can be inspected at scale, turning an opaque retrieval component into an auditable, user-grounded one. 

---
# What Would You Click? Personalized Video Thumbnail Generation with Preference-aware Highlight Retrieval 

**Authors**: Zhiyu He, Zecheng Zhao, Tong Chen, Zi Huang, Yiqun Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.12882)  

**Abstract**: Video thumbnails are a key factor for attracting user clicks on video platforms, and are increasingly supported by automation. However, existing thumbnail generation methods typically produce generic results shared across users, overlooking the diversity of individual preferences. We therefore introduce personalized video thumbnail generation, a novel task that aims to create thumbnails tailored to user-specific preferences. It is challenging in two aspects: (i) identifying visual anchors (i.e., key frames) from each video to guide the generation, which requires a balance between personalization and informativeness that existing highlight detection methods fail to achieve; and (ii) generating personalized thumbnails that are both visually coherent and faithful to the original video. As a response, we propose a two-stage framework that tightly couples preference-aware retrieval with controllable generation. In the first stage, a personalized highlight retriever captures fine-grained user-video interactions and incorporates video semantics through summarization, enabling the selection of diverse visual anchors aligned with both user preferences and video contexts. In the second stage, a VLM-guided diffusion pipeline transforms these anchors into thumbnails by extracting and injecting semantically grounded visual cues, improving personalization while preserving visual coherence and fidelity. Experiments on two public datasets show our method delivers state-of-the-art performance compared with both retrieval-based and generative baselines. A user study further demonstrates improved click preference, highlighting its effectiveness in enhancing user engagement. The code is available at this https URL. 

---
# Towards Vision-Free CIR: Attribute-Augmented Scoring and LLM-Based Reranking for Zero-Shot Composed Image Retrieval 

**Authors**: Ryotaro Shimada, Yu-Chieh Lin, Yuji Nozawa, Youyang Ng, Osamu Torii, Yusuke Matsui  

**Link**: [PDF](https://arxiv.org/pdf/2607.12621)  

**Abstract**: Recent work has shown that "Vision-Free'' approaches (representing images as text) can be effective for standard image retrieval tasks. However, it remains unclear whether this paradigm can effectively handle a more complex, multimodal task, Composed Image Retrieval (CIR), due to the inherent information loss in textual descriptions. In this paper, we introduce a Vision-Free CIR framework that addresses this challenge through two key techniques: (1) Attribute-Augmented Hybrid Scoring, which compensates for lost visual details via explicit attribute matching, and (2) LLM-Based Reranking, which verifies semantic consistency of top candidates. Experiments on the open-domain CIRR dataset show that our approach outperforms existing Zero-shot CIR methods (44.04% R@1, +8.79%). On FashionIQ, our results highlight the trade-off between semantic reasoning and fine-grained visual matching. Ablation studies reveal that both attribute-augmented scoring and LLM-Based Reranking consistently improve performance. 

---
# On-Device Deep Research at 4B: Exposure Bounds Faithfulness, Retrieval Bounds Coverage 

**Authors**: Vinay Kumar Chaganti  

**Link**: [PDF](https://arxiv.org/pdf/2607.12257)  

**Abstract**: On-device research agents search a corpus, read sources, and write a cited brief on a personal laptop. Whether their citations are faithful, and at what cost, is unmeasured for a deployable small model. This study fixes one 4B generator on a 24 GB laptop and asks what makes its citations faithful. It separates two quantities usually reported as one number. Cited claim faithfulness asks whether the cited source supports the claim. Trustworthy coverage asks whether the agent also cites the right sources. The study crosses how much of each source the generator sees, 400 against 1500 characters, with the quality of the sources supplied, gold papers against retrieved papers. Two levers fall out, and they act on different outcomes. Exposure sets faithfulness. More of each source lifts faithfulness from 0.45 to 0.58 on retrieved sources and from 0.37 to 0.58 on gold sources, and the two settings converge, so faithfulness is bound by exposure, not by whether the source is correct. The exposure lift is robust to a second, independent judge; the exact convergence is tight under the primary judge and only approximate under the second. Retrieval sets coverage. Trustworthy coverage stays near 0.22 on retrieved sources at any exposure, because recall is held near 0.40, so exposure cannot fix which sources are cited. The extra exposure costs about 235 output tokens. The practical recipe is to raise per source exposure first, cheaply, and then treat retrieval recall as the only remaining lever. 

---
# Cost-Governed RAG: Unified Per-Tenant Cost Attribution Across Retrieval and Generation in Multi-Tenant LLM Systems 

**Authors**: Navnit Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2607.12188)  

**Abstract**: Enterprise Retrieval-Augmented Generation (RAG) deployments face a critical governance gap: while LLM generation cost is metered per token, the retrieval layer - vector memory, similarity compute, and embedding API calls - remains an unattributed shared cost, enabling invisible cross-subsidization among tenants. We present Cost-Governed RAG, an architecture that integrates a codebook-oblivious vector index (TurboVec) with a multi-tenant LLM governance gateway, creating a unified observability stack where embedding, retrieval, and generation costs are jointly attributable per tenant. The architecture exploits TurboVec's deterministic, closed-form memory formula to enable near-exact per-tenant retrieval cost calculation - a property unavailable in graph-based indexes with non-linear memory overhead. Deployed on Snowpark Container Services within a cloud data platform's governance boundary, the system achieves 99.96% end-to-end cost attribution accuracy across 100 simulated tenants (10M vectors, log-normal size distribution) with telemetry overhead below 0.04% of query latency. The architecture reduces retrieval infrastructure cost by 3.1-9.0x compared to managed vector database services under the pricing assumptions detailed in Section IV. We formalize a three-layer cost model and demonstrate that codebook-oblivious quantization enables deterministic per-tenant cost attribution while also removing the shared-codebook leakage surface present in trained quantizers - the latter observation being exploratory and subject to the limitations described in Section VII. 

---
# Graph-Constrained Policy Learning for Extreme Clinical Code Prediction 

**Authors**: Amritpal Singh, Sebastian Torres, Khawar Shakeel, Syed Ahmad Chan Bukhari  

**Link**: [PDF](https://arxiv.org/pdf/2607.11954)  

**Abstract**: Clinical code prediction maps unstructured discharge summaries to ICD-10-CM leaf codes in a large, sparse, and deeply hierarchical label space. Most systems treat the task as flat multi-label classification, scoring codes independently and providing limited training signal for rare labels. We propose a graph-constrained traversal policy that formulates ICD prediction as a finite-horizon decision process over a pruned code hierarchy. A single language model descends the graph level by level, selecting valid child nodes until billable leaf codes are reached. This converts extreme multi-label prediction into sparse, hierarchy-aware subset decisions while guaranteeing structurally valid outputs.
On MIMIC-IV discharge summaries, our best supervised policy, SFT-1+, achieves 0.709 micro-F1 on a curated 50-code subset and 0.527 micro-F1 on the full 15,761-code space, outperforming flat baselines including CAML, LAAT, and PLM-ICD. In the full setting, SFT-1+ improves over the strongest flat baseline by 0.044 micro-F1 and 0.157 macro-F1, suggesting that graph-constrained decomposition mitigates the rare-code bottleneck. A controlled factorial study evaluates architecture, training algorithm, and data budget. Across both scales, one shared policy matches a three-specialist cascade while avoiding its context-window overflow on 28-32% of full-space test notes. Increasing supervised trajectory data is the only intervention that consistently improves performance, while GRPO reinforcement learning provides no benefit over supervised continuation with matched data. These results show that simple graph-constrained policy learning can outperform more complex flat, cascaded, and reinforcement-learning alternatives for extreme clinical code prediction. 

---
# Transforming LLMs into Efficient Cross-Encoders via Knowledge Distillation for RAG Reranking 

**Authors**: Shreeya Dasa Lakshminath, Shubhan S  

**Link**: [PDF](https://arxiv.org/pdf/2607.11933)  

**Abstract**: Cross-encoders achieve high reranking accuracy in Retrieval-Augmented Generation (RAG) pipelines but impose quadratic inference costs that limit real-time deployment. We address this by fine-tuning LLaMA 3 (8B) as a drop-in reranker using a two-stage pipeline: supervised fine-tuning on a custom query-document relevance dataset via the Unsloth framework with LoRA adapters, followed by 4-bit quantization for efficient inference. The resulting model replaces the cross-encoder in a dual-retriever RAG pipeline combining BM25 and dense vector search. Evaluated on a domain-specific question-answering benchmark using the RAGAS framework, our fine-tuned LLaMA 3 reranker achieves gains of 14% in answer relevancy, 16% in context precision, 19% in answer similarity, and 21% in answer correctness over the cross-encoder baseline, while reducing inference overhead through 4-bit quantization. These results demonstrate that instruction-tuned LLMs can be adapted into accurate, efficient rerankers without the quadratic complexity of traditional cross-encoders. 

---
