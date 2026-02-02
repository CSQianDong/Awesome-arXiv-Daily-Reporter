# OrLog: Resolving Complex Queries with LLMs and Probabilistic Reasoning 

**Authors**: Mohanna Hoveyda, Jelle Piepenbrock, Arjen P de Vries, Maarten de Rijke, Faegheh Hasibi  

**Link**: [PDF](https://arxiv.org/pdf/2601.23085)  

**Abstract**: Resolving complex information needs that come with multiple constraints should consider enforcing the logical operators encoded in the query (i.e., conjunction, disjunction, negation) on the candidate answer set. Current retrieval systems either ignore these constraints in neural embeddings or approximate them in a generative reasoning process that can be inconsistent and unreliable. Although well-suited to structured reasoning, existing neuro-symbolic approaches remain confined to formal logic or mathematics problems as they often assume unambiguous queries and access to complete evidence, conditions rarely met in information retrieval. To bridge this gap, we introduce OrLog, a neuro-symbolic retrieval framework that decouples predicate-level plausibility estimation from logical reasoning: a large language model (LLM) provides plausibility scores for atomic predicates in one decoding-free forward pass, from which a probabilistic reasoning engine derives the posterior probability of query satisfaction. We evaluate OrLog across multiple backbone LLMs, varying levels of access to external knowledge, and a range of logical constraints, and compare it against base retrievers and LLM-as-reasoner methods. Provided with entity descriptions, OrLog can significantly boost top-rank precision compared to LLM reasoning with larger gains on disjunctive queries. OrLog is also more efficient, cutting mean tokens by $\sim$90\% per query-entity pair. These results demonstrate that generation-free predicate plausibility estimation combined with probabilistic reasoning enables constraint-aware retrieval that outperforms monolithic reasoning while using far fewer tokens. 

---
# BEAR: Towards Beam-Search-Aware Optimization for Recommendation with Large Language Models 

**Authors**: Weiqin Yang, Bohao Wang, Zhenxiang Xu, Jiawei Chen, Shengjia Zhang, Jingbang Chen, Canghong Jin, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.22925)  

**Abstract**: Recent years have witnessed a rapid surge in research leveraging Large Language Models (LLMs) for recommendation. These methods typically employ supervised fine-tuning (SFT) to adapt LLMs to recommendation scenarios, and utilize beam search during inference to efficiently retrieve $B$ top-ranked recommended items. However, we identify a critical training-inference inconsistency: while SFT optimizes the overall probability of positive items, it does not guarantee that such items will be retrieved by beam search even if they possess high overall probabilities. Due to the greedy pruning mechanism, beam search can prematurely discard a positive item once its prefix probability is insufficient.
To address this inconsistency, we propose BEAR (Beam-SEarch-Aware Regularization), a novel fine-tuning objective that explicitly accounts for beam search behavior during training. Rather than directly simulating beam search for each instance during training, which is computationally prohibitive, BEAR enforces a relaxed necessary condition: each token in a positive item must rank within the top-$B$ candidate tokens at each decoding step. This objective effectively mitigates the risk of incorrect pruning while incurring negligible computational overhead compared to standard SFT. Extensive experiments across four real-world datasets demonstrate that BEAR significantly outperforms strong baselines. Code will be released upon acceptance. 

---
# Compact Hypercube Embeddings for Fast Text-based Wildlife Observation Retrieval 

**Authors**: Ilyass Moummad, Marius Miron, David Robinson, Kawtar Zaher, Hervé Goëau, Olivier Pietquin, Pierre Bonnet, Emmanuel Chemla, Matthieu Geist, Alexis Joly  

**Link**: [PDF](https://arxiv.org/pdf/2601.22783)  

**Abstract**: Large-scale biodiversity monitoring platforms increasingly rely on multimodal wildlife observations. While recent foundation models enable rich semantic representations across vision, audio, and language, retrieving relevant observations from massive archives remains challenging due to the computational cost of high-dimensional similarity search. In this work, we introduce compact hypercube embeddings for fast text-based wildlife observation retrieval, a framework that enables efficient text-based search over large-scale wildlife image and audio databases using compact binary representations. Building on the cross-view code alignment hashing framework, we extend lightweight hashing beyond a single-modality setup to align natural language descriptions with visual or acoustic observations in a shared Hamming space. Our approach leverages pretrained wildlife foundation models, including BioCLIP and BioLingual, and adapts them efficiently for hashing using parameter-efficient fine-tuning. We evaluate our method on large-scale benchmarks, including iNaturalist2024 for text-to-image retrieval and iNatSounds2024 for text-to-audio retrieval, as well as multiple soundscape datasets to assess robustness under domain shift. Results show that retrieval using discrete hypercube embeddings achieves competitive, and in several cases superior, performance compared to continuous embeddings, while drastically reducing memory and search cost. Moreover, we observe that the hashing objective consistently improves the underlying encoder representations, leading to stronger retrieval and zero-shot generalization. These results demonstrate that binary, language-based retrieval enables scalable and efficient search over large wildlife archives for biodiversity monitoring systems. 

---
# Farewell to Item IDs: Unlocking the Scaling Potential of Large Ranking Models via Semantic Tokens 

**Authors**: Zhen Zhao, Tong Zhang, Jie Xu, Qingliang Cai, Qile Zhang, Leyuan Yang, Daorui Xiao, Xiaojia Chang  

**Link**: [PDF](https://arxiv.org/pdf/2601.22694)  

**Abstract**: Recent studies on scaling up ranking models have achieved substantial improvement for recommendation systems and search engines. However, most large-scale ranking systems rely on item IDs, where each item is treated as an independent categorical symbol and mapped to a learned embedding. As items rapidly appear and disappear, these embeddings become difficult to train and maintain. This instability impedes effective learning of neural network parameters and limits the scalability of ranking models. In this paper, we show that semantic tokens possess greater scaling potential compared to item IDs. Our proposed framework TRM improves the token generation and application pipeline, leading to 33% reduction in sparse storage while achieving 0.85% AUC increase. Extensive experiments further show that TRM could consistently outperform state-of-the-art models when model capacity scales. Finally, TRM has been successfully deployed on large-scale personalized search engines, yielding 0.26% and 0.75% improvement on user active days and change query ratio respectively through A/B test. 

---
# PersonaAct: Simulating Short-Video Users with Personalized Agents for Counterfactual Filter Bubble Auditing 

**Authors**: Shilong Zhao, Qinggang Yang, Zhiyi Yin, Xiaoshi Wang, Zhenxing Chen, Du Su, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.22547)  

**Abstract**: Short-video platforms rely on personalized recommendation, raising concerns about filter bubbles that narrow content exposure. Auditing such phenomena at scale is challenging because real user studies are costly and privacy-sensitive, and existing simulators fail to reproduce realistic behaviors due to their reliance on textual signals and weak personalization. We propose PersonaAct, a framework for simulating short-video users with persona-conditioned multimodal agents trained on real behavioral traces for auditing filter bubbles in breadth and depth. PersonaAct synthesizes interpretable personas through automated interviews combining behavioral analysis with structured questioning, then trains agents on multimodal observations using supervised fine-tuning and reinforcement learning. We deploy trained agents for filter bubble auditing and evaluate bubble breadth via content diversity and bubble depth via escape potential. The evaluation demonstrates substantial improvements in fidelity over generic LLM baselines, enabling realistic behavior reproduction. Results reveal significant content narrowing over interaction. However, we find that Bilibili demonstrates the strongest escape potential. We release the first open multimodal short-video dataset and code to support reproducible auditing of recommender systems. 

---
# SCaLRec: Semantic Calibration for LLM-enabled Cloud-Device Sequential Recommendation 

**Authors**: Ruiqi Zheng, Jinli Cao, Jiao Yin, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2601.22543)  

**Abstract**: Cloud-device collaborative recommendation partitions computation across the cloud and user devices: the cloud provides semantic user modeling, while the device leverages recent interactions and cloud semantic signals for privacy-preserving, responsive reranking. With large language models (LLMs) on the cloud, semantic user representations can improve sequential recommendation by capturing high-level intent. However, regenerating such representations via cloud LLM inference for every request is often infeasible at real-world scale. As a result, on-device reranking commonly reuses a cached cloud semantic user embedding across requests. We empirically identify a cloud semantic staleness effect: reused embeddings become less aligned with the user's latest interactions, leading to measurable ranking degradation.
Most existing LLM-enabled cloud-device recommenders are typically designed around on-demand cloud semantics, either by assuming low-latency cloud LLM access or by regenerating semantic embeddings per request. When per-request regeneration is infeasible and cached semantics must be reused, two technical challenges arise: (1) deciding when cached cloud semantics remain useful for on-device reranking, and (2) maintaining ranking quality when the cloud LLM cannot be invoked and only cached semantics are available. To address this gap, we introduce the Semantic Calibration for LLM-enabled Cloud-Device Recommendation (SCaLRec). First, it estimates the reliability of cached semantics under the user's latest interactions. Second, an on-device semantic calibration module is proposed to adjusts the cached semantic embedding on-device using up-to-date interaction evidence, without per-request cloud LLM involvement. Experiments on real-world datasets show that SCaLRec consistently improves recommendation performance over strong baselines under cloud semantic staleness. 

---
# FITMM: Adaptive Frequency-Aware Multimodal Recommendation via Information-Theoretic Representation Learning 

**Authors**: Wei Yang, Rui Zhong, Yiqun Chen, Shixuan Li, Heng Ping, Chi Lu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.22498)  

**Abstract**: Multimodal recommendation aims to enhance user preference modeling by leveraging rich item content such as images and text. Yet dominant systems fuse modalities in the spatial domain, obscuring the frequency structure of signals and amplifying misalignment and redundancy. We adopt a spectral information-theoretic view and show that, under an orthogonal transform that approximately block-diagonalizes bandwise covariances, the Gaussian Information Bottleneck objective decouples across frequency bands, providing a principled basis for separate-then-fuse paradigm. Building on this foundation, we propose FITMM, a Frequency-aware Information-Theoretic framework for multimodal recommendation. FITMM constructs graph-enhanced item representations, performs modality-wise spectral decomposition to obtain orthogonal bands, and forms lightweight within-band multimodal components. A residual, task-adaptive gate aggregates bands into the final representation. To control redundancy and improve generalization, we regularize training with a frequency-domain IB term that allocates capacity across bands (Wiener-like shrinkage with shut-off of weak bands). We further introduce a cross-modal spectral consistency loss that aligns modalities within each band. The model is jointly optimized with the standard recommendation loss. Extensive experiments on three real-world datasets demonstrate that FITMM consistently and significantly outperforms advanced baselines. 

---
# Do AI Overviews Benefit Search Engines? An Ecosystem Perspective 

**Authors**: Yihang Wu, Jiajun Tang, Jinfei Liu, Haifeng Xu, Fan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2601.22493)  

**Abstract**: The integration of AI Overviews into search engines enhances user experience but diverts traffic from content creators, potentially discouraging high-quality content creation and causing user attrition that undermines long-term search engine profit. To address this issue, we propose a game-theoretic model of creator competition with costly effort, characterize equilibrium behavior, and design two incentive mechanisms: a citation mechanism that references sources within an AI Overview, and a compensation mechanism that offers monetary rewards to creators. For both cases, we provide structural insights and near-optimal profit-maximizing mechanisms. Evaluations on real click data show that although AI Overviews harm long-term search engine profit, interventions based on our proposed mechanisms can increase long-term profit across a range of realistic scenarios, pointing toward a more sustainable trajectory for AI-enhanced search ecosystems. 

---
