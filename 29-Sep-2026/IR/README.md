# Rubric-Calibrated Preferences: Cross-Query Calibration of LLM Judgments via Item Response Theory 

**Authors**: Fabian David Schmidt, Donato Crisostomi, Carlos Lassance, Nils Reimers  

**Link**: [PDF](https://arxiv.org/pdf/2609.35739)  

**Abstract**: Rerankers decide which documents users and LLMs see, yet their standard metric, nDCG, relies on human relevance labels that are costly, sparse, noisy, and discretely graded. As rerankers approach each other in quality, nDCG on these labels therefore increasingly fails to separate them. LLM judges could supply dense labels. Relative judgments within one query tell even close candidates apart, yet their scores share no scale across queries. Absolute grades share one scale but are too coarse to distinguish documents of similar relevance. We propose Rubric-Calibrated Preferences (RCP), which combine both kinds of judgment. A listwise Bradley-Terry tournament orders each query's documents, and a rubric of yes/no criteria of increasing stringency provides an absolute standard. Item Response Theory (IRT), which scores test-takers based on their answers to common questions, then uses the shared criteria to put all queries' tournament scores on one scale. RCP's retrieval metric, RCP-nDCG, replaces nDCG's discrete labels with the resulting calibrated relevance probabilities. Against blind grades from 46 external annotators, calibration raises the correlation between a query's mean score and its mean human grade from 0.538 to 0.795. The probabilities rank a useful document above a non-useful one with probability 0.910 (AUC, chance 0.5), versus 0.651 for the benchmark labels. When the annotators' grades prefer one of two rerankers and exactly one metric agrees, that metric is RCP-nDCG in 72.4% of 185 comparisons (chance about 53%). On TREC-DL, RCP-nDCG sides with NIST assessors' grades on every reranker pair that these grades separate significantly. RCP-nDCG also resolves many of nDCG's ties and separates 1.9 times as many reranker pairs on NanoBEIR. Rubric calibration thus turns relative LLM judgments into dense relevance labels that are comparable across queries and agree with human judgment. 

---
# Can Generative Retrievers Learn Semantic IDs Without Forgetting How to Speak? 

**Authors**: Junchen Fu, Kleomenis Katevas, Vandana Rajan, Sofía Celi, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2609.35430)  

**Abstract**: Generative retrieval (GR) enables end-to-end retrieval by generating document semantic identifiers (SIDs). However, retrieval-only fine-tuning can over-specialize pretrained language models to SID prediction, substantially distorting their natural-language distribution and limiting their suitability for interactive systems that must both retrieve documents and generate natural-language responses. We introduce SpeakGR, a dual-objective framework that learns SIDs while preserving language generation. It combines supervised SID learning with speak-preserving regularization: an on-policy distillation objective that aligns the current model with a frozen copy of the original model on student-generated prefixes using forward KL over the original text vocabulary. We further propose Adaptive SpeakGR, which dynamically adjusts the preservation strength based on observed language drift. Compared with SFT-only, SpeakGR reduces WikiText-2 forward KL by 81.3-93.8% on MS MARCO and 81.2-85.2% on Natural Questions (NQ) while retaining effective retrieval across three different LLMs. Adaptive SpeakGR further improves retrieval over SpeakGR in most settings while maintaining substantially lower language drift than SFT-only. 

---
# Signal or Noise? Modality Contribution and Cooperation in Multimodal GraphRAG 

**Authors**: Antonios Georgakopoulos, Paul Groth, Lise Stork  

**Link**: [PDF](https://arxiv.org/pdf/2609.35304)  

**Abstract**: Multimodal knowledge graphs (KGs) integrate information from text, figures, tables, and other modalities into a unified structured representation, with the promise that richer evidence enables better inference. In GraphRAG systems built over such graphs, it is commonly assumed that retrieving evidence from more modalities at inference time improves downstream performance. Yet, redundant or overlapping multimodal evidence may distract language models in question answering (QA), and whether each modality contributes equally across questions, models, and tasks remains poorly understood. In this work, we study how modality-aware retrieval affects downstream inference in a multimodal GraphRAG pipeline, using document visual question answering (DocVQA) as a testbed. We extend an existing KG-based QA framework to be modality-aware, leveraging the graph structure to track which modality supports which facts and to selectively filter evidence at the edge level. This enables us to investigate whether providing all available multimodal evidence at inference time benefits QA, and to evaluate the contribution and cooperation of modalities across question, task, and model characteristics. Through a controlled analysis within a state-of-the-art multimodal GraphRAG pipeline, five multimodal LLMs and two DocVQA benchmarks, we find that tables and text provide the strongest contributions, and that combining modalities frequently produces redundancy rather than synergy, particularly for pairs involving textual information. Positive cooperation appears mainly between non-text modalities and depends on question intent and task type. Our findings argue for selective, modality-aware retrieval in the design of more effective GraphRAG systems, where modalities are filtered according to the downstream task rather than retrieved uniformly. 

---
# RenderRank: Learning to Rerank Text with Compressed Visual Tokens 

**Authors**: Seongtae Hong, Youngjoon Jang, Jungseob Lee, Hyeonseok Moon, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2609.35069)  

**Abstract**: Rendering document text as images allows vision-language models to encode documents as visual tokens, which can reduce input sequence length compared with text input. This reduction in input length is particularly useful for reranking, where each query involves scoring multiple candidate documents and token savings apply to each candidate evaluation. We introduce RenderRank, a reranker that learns query-dependent relevance scoring from compressed visual document representations instead of the text token sequences used by conventional text-based rerankers. Training first aligns relevance scores from visual inputs with those of a text-based teacher, then refines the relative scores of positive and negative documents for the same query. Across 11 datasets from BEIR, RenderRank uses 16.5-35.5% fewer input tokens while achieving an average NDCG@10 of 55.96, outperforming all evaluated text-based baselines below 4B parameters and some larger models. Across four long-document datasets, it achieves an average NDCG@10 of 88.27 with approximately half the average input token count of the evaluated text-based rerankers. In this setting, RenderRank delivers 1.70x the highest average throughput of the evaluated baselines. These results demonstrate that compressed visual representations can support accurate document relevance scoring, providing an alternative to text token representations for reranking. 

---
# Mitigating Popularity Bias in Recommendation with Global Listwise Learning and Progressive Bi-Weighting 

**Authors**: Tianyu Zhu, Jiandong Ding, Yansong Shi, Guoqing Chen, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2609.35041)  

**Abstract**: In recommender systems, user feedback typically follows a long-tail distribution, which leads many recommendation algorithms to exacerbate popularity bias by disproportionately favoring popular items. To mitigate this issue, recent studies have employed Inverse Propensity Scoring (IPS) to rebalance training data via reweighting user-item interactions. However, the effectiveness of IPS-based approaches is often constrained by locally unbiased objectives and inaccurate propensity estimation. In this paper, we propose Multinomial Likelihood with Bi-Weighting (Mult-BiW) to address these limitations. First, we introduce a debiasing framework, termed Mult-IPS, which integrates multinomial likelihood with IPS to capture global and unbiased user preferences over the entire item set. Second, we develop a Bi-Weighting (BiW) strategy that jointly leverages propensity scores and a collection model, incorporating a smoothing mechanism to enhance the robustness of propensity estimation. We further provide theoretical analyses that establish an upper bound on the empirical bias and characterize the optimal form of the collection model. Third, to mitigate the adverse effects of aggressive reweighting on representation learning, we design a Progressive Bi-Weighting strategy that gradually transitions from discriminative representation learning to popularity debiasing. Extensive experiments on real-world datasets show that Mult-BiW consistently outperforms state-of-the-art baselines. 

---
# Recommendation Ranking Off-Policy Evaluation under Ranking-Dependent Examination via Examination-Relevance Decomposition 

**Authors**: Riki Okamura, Toshiharu Sugawara  

**Link**: [PDF](https://arxiv.org/pdf/2609.35034)  

**Abstract**: Off-policy evaluation, which estimates evaluation policy performance from logged data, is key for recommender ranking policies. However, logged clicks cannot distinguish unexamined items from examined non-clicks, causing bias in existing estimators when the assumed examination structures fail. We propose two estimators based on the decomposition of clicks into examination and relevance. First, the latent-examination independent inverse propensity score (LE-IIPS) estimator corrects the IIPS bias using policy examination probability ratios. Second, the examination-decomposed doubly robust (ED-DR) estimator extends LE-IIPS to a doubly robust framework. ED-DR is unbiased if the examination probabilities are correct regardless of relevance accuracy, or under ranking-independent examination, even if both model estimates are inaccurate. Experiments show that ED-DR achieves a lower MSE than existing methods with large sample sizes, especially when the examination depends on ranking. We also highlight its limitations under small samples or cascade user behavior conditions. 

---
# PEAR: Progressive Evidence-Based AutoResearch for Industrial Search Systems 

**Authors**: Yifan Wang, Shipeng Zhu, Fei Xiong, Yuqin Yang, Yonghui Huang, Kunyao Wu, Yue Wang, Weichao Meng, Yu Gong  

**Link**: [PDF](https://arxiv.org/pdf/2609.35031)  

**Abstract**: AutoResearch improves systems through iterative experimentation: agents propose candidate modifications, evaluate them, and use the results to guide subsequent exploration. Applying this paradigm to industrial search presents two challenges. (1) Common AutoResearch approaches follow a keep-if-better rule, retaining the highest-scoring candidate for subsequent experiments. Under non-stationary traffic, transient gains may be mistaken for persistent improvements, impairing reliable accumulation of search knowledge. (2) Candidate modifications can be evaluated at multiple fidelity levels, from low-cost proxies to online validation, differing in cost, objective alignment, and statistical reliability. Existing methods rely on individual signals or task-specific procedures, lacking a unified basis for using evidence across levels to guide search. We introduce Progressive Evidence-Based AutoResearch (PEAR) with two complementary components. Evidence-driven AutoResearch maintains an independent, hypothesis-guided research state for each strategy task within a predefined objective and intervention scope. Each state evolves through a Plan-Execute-Evaluate-Update transition that links experimentation to context-aware evidence interpretation and hypothesis revision. Confidence-Gated Verifier Ladder organizes evaluation into four levels of increasing fidelity: Offline Replay, Shadow-Traffic Evaluation, Rapid Online Evaluation, and Decision-Grade Online Evaluation. A unified confidence-based gate promotes candidates only when evidence supports a statistically significant positive effect, enabling broad low-cost exploration while reserving costly online experiments for promoted candidates. In a real-world industrial search system, strategies optimized with PEAR significantly increased Main Order/DAU by 2.7336% and 3.2957% relative to their respective baselines in two A/B experiments. 

---
# ColNanoVDR: Document-Free Query Distillation for Multi-Vector Visual Document Retrieval via Optimal Transport 

**Authors**: Zhuchenyang Liu, Ziyi Wang, Yao Zhang, Yu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2609.34899)  

**Abstract**: Multi-vector retrievers built on vision-language models lead visual document retrieval (VDR), but they run a multi-billion-parameter query encoder on every search. Distilling this encoder into a small student that queries the teacher's existing index would remove the bottleneck. The standard recipe, however, matches the teacher's MaxSim scores and so requires encoding and caching every training page, which can reach terabytes of page tokens. NanoVDR avoids pages entirely by training on the teacher's query embeddings alone, but only for single-vector retrievers. We present ColNanoVDR, to our knowledge the first framework to bring this document-free distillation to multi-vector VDR. Its objective, OTW (Optimal Transport with Learned Weights), aligns the student's query tokens with the teacher's by entropic optimal transport, with a learned weight for each student token, and needs no correspondence between the two tokenizations. We prove that the resulting alignment cost bounds the MaxSim score difference on every page. Distilled from five state-of-the-art teachers, the 149M text-only students retain about 95% of their teachers' NDCG@5 on ViDoRe v1-v3 while encoding queries up to 26x faster. Under identical training, OTW matches score distillation while encoding no page and reading 12.6x less cached teacher data. 

---
# No Attention, No Problem: Rethinking Session-based Recommendation with Pure Convolution 

**Authors**: Tao Huang, Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.34802)  

**Abstract**: Session-based recommendation (SBR) predicts the next choice in a session by analyzing recent interactions. Transformer-based models are widely used because of their ability to capture long-range dependencies through self-attention mechanisms. In contrast, traditional convolutional models, although more efficient, are often limited by their weak global modeling capabilities and are losing ground in SBR tasks. In this work, we propose a Next-generation Pure Convolutional Framework (NextConvRec) for SBR tasks, aiming to balance efficiency and performance. NextConvRec uses a Structural and Positional Convolutional Encoder (SPCE) for preprocessing, combining learnable convolutional positional biases with session-level structural signals extracted through GCN layers. Its backbone convolutional module effectively expands the effective receptive field through depthwise convolutions and pointwise convolutions, enabling robust long-range preference modeling without attention mechanisms. Extensive experiments on 4 benchmark datasets show that NextConvRec outperforms several state-of-the-art baselines by around 1.73% on average, and reduces the average inference time per session by 16.7%. The convolutional architectures remain a promising direction for efficient and accurate session-based recommendations. 

---
# EvoSkillRec: Skill-Genome Evolution for Recommender Architecture Discovery 

**Authors**: Xiaopeng Li, Kuo Cai, Bo Chen, Wenlin Zhang, Mengyang Ma, Yingyi Zhang, Zichuan Fu, Yu Yang, Qidong Liu, Yiyu Wang, Ruiming Tang, Wenwu Ou, Jiang Wu, Zhanbo Xu, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.34552)  

**Abstract**: Modern recommender systems advance not only by scaling data and parameters, but also by encoding task-specific inductive biases through architecture, including sparse feature interactions for click-through rate (CTR) prediction, temporal attention for sequential recommendation, and expert routing for multi-task learning. However, these biases are typically human expert designed or searched within predefined operator spaces. Although Recent LLM-driven code evolution expands this space, unconstrained edits often produce invalid or ineffective architectures, underuse established architecture design knowledge, and fail to preserve successful innovations for reuse. We introduce EvoSkillRec, a promotion-and-reuse framework for cumulative recommender architecture evolution. It first decomposes recommenders into atomic executable skills and represents architectures as typed skill genomes, with each skill equipped with input--output types, semantic annotations, and implementation code. We then evolve models with different tasks through two coupled spaces: a constrained skill--space that mutates, recombines, specializes, and reuses validated skills, and an open-ended code--space in which LLM planners and synthesizers invent new skill modules using prior evolution traces and accumulated experience. An autoresearch controller evaluates candidates, diagnoses failures, retrieves relevant skills, promotes validated innovations into the skill library, and adaptively allocates the proposal budget between the two spaces. Extensive experiments on CTR prediction, multi-task learning, and multi-domain learning, including resource-constrained co-optimization of predictive quality and model FLOPs utilization in generative ranking models, consistently demonstrate the effectiveness of our proposed EvoSkillRec. 

---
# Eval4DiRec: A Unified and Systematic Evaluation Framework for Diffusion-based Recommender Systems 

**Authors**: Cong Wang, Shoujin Wang, Yishuo Li, Qi Zhang, Liang Hu, Wenpeng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2609.34404)  

**Abstract**: Leveraging the strong generative capabilities and stable training dynamics of diffusion models, diffusion-based recommender systems (RSs) have recently emerged as a novel recommendation paradigm, attracting increasing attention from both academia and industry. However, despite the rapid growth of diffusion-based RSs, a critical issue has emerged: the lack of a unified and systematic quantitative evaluation benchmark, which often results in irreproducible experimental results and unfair comparisons across studies due to inconsistent data processing, training configurations, inference procedures, and evaluation protocols. To address this challenge, we propose Eval4DiRec, the first unified and open-source evaluation framework specifically designed for diffusion-based RSs. Eval4DiRec supports 14 representative diffusion-based RS models across five different recommendation scenarios, providing consistent and reproducible experimental settings to systematically assess their performance. Built upon this framework, we conduct extensive empirical studies to benchmark these models under unified protocols. The results highlight the strong potential of diffusion models for recommendation while also revealing key factors and practical challenges that substantially affect their performance, thereby establishing a solid foundation to facilitate fair evaluation and guide future research in this promising field. Our code and data are available at: this https URL. 

---
# Relevance-Resolution Transfer via Scale-Decomposable Fractional Diffusion for Multi-Length Cross-Modal Hash Retrieval 

**Authors**: Xi Chen, Xu Chen, Xiangyang Jia, Ting Gan, Xu Zhang, Shuquan Wei, Sitong Fan  

**Link**: [PDF](https://arxiv.org/pdf/2609.34393)  

**Abstract**: Cross-modal hashing enables efficient retrieval by encoding heterogeneous data into compact binary codes. Recent methods exploit fine-grained relations encoded in multi-label training structure, yet none of them constrains how those relations survive as consistent candidate rankings in finite, multi-length Hamming spaces, which we term the relevance resolution bottleneck (RRB). To address the RRB, we propose MultiBit, which transfers relevance resolution from multi-label structure to multi-length Hamming spaces. MultiBit first constructs a scale-decomposable fractional relation teacher from dataset-level label co-occurrence and label specificity, and models dependencies from local to long-range over continuous diffusion scales. It then maps the discretized diffusion scales and their quadrature weights to scale-aware bit subblocks of the maximum-length code, organizes the target code lengths as nested prefixes, and aligns their Hamming candidate rankings with the teacher relations. Experiments on multiple benchmarks demonstrate improved retrieval accuracy. Code is available in the supplementary material. 

---
# Correcting to Predict: Pseudo-Value Correction for Multimodal Attribute Value Extraction 

**Authors**: Junhao Zhang, Feiran Hu, Xiao Hu, Baoliang Cui, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2609.34383)  

**Abstract**: Product attribute value extraction (AVE) is a fundamental task in e-commerce, aiming to identify specific values of predefined attributes from multimodal product profiles such as text and images. While multimodal large language models (MLLMs) have shown promise for AVE, they face challenges in extracting implicit attributes that require joint reasoning over visual and textual cues, often confusing semantically similar values. However, existing methods often fail to resolve such ambiguities because the correct value often depends on subtle multimodal cues that are easy to miss or override. To address this challenge, we propose Correcting to Predict (C2P), a framework that treats attribute extraction as a correction process. Given an initial pseudo-value such as a retrieved candidate or placeholder, the model learns to correct it using multimodal evidence. During training, diverse pseudo-values help the model learn evidence-based correction behavior, and a self-consistency refinement stage further reduces sensitivity to pseudo-value perturbations. At inference, a fixed placeholder triggers the learned correction behavior, enabling efficient single-pass prediction without online retrieval or iterative refinement. We evaluate C2P on a public benchmark and a large-scale industrial dataset. Offline results show that C2P outperforms strong baselines, with notable gains on ambiguous attributes. Online A/B tests on AliExpress further show consistent improvements in seller adoption, attribute completeness, and user engagement, validating C2P's effectiveness and efficiency in real-world deployment. 

---
# SPRINT: Single-Step Generative Recommendation via Average Probability Velocity 

**Authors**: Zhuo Cai, Shoujin Wang, Peilin Zhou, Min Xu, Julian McAuley, Fang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.34306)  

**Abstract**: Semantic ID (SID) based generative recommendation represents each item as a sequence of discrete tokens, and recommends by generating the SID of the item a user would like to interact with. Both dominant paradigms in this domain generally pay for generation token by token: autoregressive models decode the tokens left-to-right, while non-autoregressive models decode in parallel yet still need multiple rounds of refinement to stay competitive. Therefore, both generally spend multiple forward passes per item, a cost that is prohibitive in latency-sensitive recommender systems. We ask whether an item can be generated in a single forward pass, and answer it through a new perspective which we call average probability velocity. We view SID generation as a flow of token generation probabilities and characterize it by its average velocity over the whole generation process. We prove that this average velocity is fully determined by the average generation probability of each token. Therefore, we directly parameterize and learn the probabilities of all tokens in a single forward pass with a bidirectional Transformer. As these probabilities are generated independently across positions and the coherence among tokens is lost, we further design a dual-level flow contrastive objective to restore the coherence among an item's tokens. It contrasts the target SID against negative SIDs at both the token and SID levels. The token level ranks the generation probabilities of the target tokens above those of negative SIDs, while the SID level scores the tokens of each SID as a whole item for capturing token coherence of each item. Extensive experiments show that our model not only generates recommendations far more efficiently ($8.39-10.04\times$ speedup over the second-fastest AR/NAR method) but also attains superior recommendation accuracy ($7.77\%$ average improvement over the second-best. 

---
# Measuring and Mitigating Identity-Cue Preference Drift in LLM-based Recommender Systems 

**Authors**: Zhuoxiong Gan, Qiang Dong  

**Link**: [PDF](https://arxiv.org/pdf/2609.34229)  

**Abstract**: In large language model-based recommender systems, identity cues embedded in prompts can steer recommendations toward group-level patterns even when the underlying behavioral evidence remains unchanged. We introduce PromptShift, an interpretable, training-free framework for quantifying and mitigating such identity-cue preference drift. We define Drift as the divergence, in both item membership and ranking order, between a recommendation list generated under an identity-cued prompt and the reference list produced from the same user's interaction history alone. SliceShift then measures the extent to which a cued list gravitates, relative to the history-only reference, toward items that are more popular within the cued slice than among the global user population. Beyond conventional accuracy, we propose DifHitRate, a difficulty-weighted hit metric that credits only relevant items, assigning higher credit to hits that are less popular within the cued slice and ranked higher in the list. All components are supported by an identity-slice-by-item table constructed from positive interactions, which further enables an adaptive post-hoc reranking strategy: the reranker interpolates between the original LLM ranking and inverse slice-popularity, with personalized interpolation weight. Experiments on two datasets with three LLMs show that identity-cued prompts incur higher mean Drift than identity-free paraphrase controls, an effect beyond generic wording sensitivity, and that SliceShift is positive across all six dataset-model settings. PromptShift consistently reduces both Drift and SliceShift, lowering macro-mean SliceShift by 62.42%, while improving DifHitRate, HitRate and MRR. These results demonstrate that identity-cue preference drift can be measured and mitigated without any model training, albeit with a modest, metric-dependent utility cost. 

---
# RidgeRank: Efficient Visual Document Reranking via Score Fusion and a Shallow Linear Readout 

**Authors**: Shubing Yang, Dongfang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.34192)  

**Abstract**: Multimodal language models rerank visual document retrieval results accurately, but scoring every candidate page at full cost makes them slow. Some methods that compress these rerankers need relevance labels to regain accuracy, and they rank by the reranker score alone. RidgeRank measures how much relevance signal the reranker score lacks and recovers it from the retriever score through a closed-form fusion rule. Maximizing a correlation objective gives the optimal fusion weight, along with the exact condition under which the reranker score by itself cannot reach that optimum. The reranker is further corrected by a single vector applied to an intermediate hidden state, obtained through one centered ridge regression onto the same model's full-depth scores on uncompressed pages. On 12 datasets drawn from ViDoRe 2 and ViDoRe 3, evaluated with two retrievers and two language model backbones, RidgeRank brings NDCG@5 to within 1.2 pp of a full cross encoder with speedups of up to 48 times, advancing the accuracy and latency Pareto frontier for visual document reranking. 

---
# STITCH-RAG: Spatio-Temporal Influence Tracing over Topic Hypergraphs for Multi-Hop Retrieval-Augmented Generation 

**Authors**: Haodong Yang, Mengzhu Chen, Jia Cai  

**Link**: [PDF](https://arxiv.org/pdf/2609.34127)  

**Abstract**: Multi-hop retrieval-augmented generation requires a retriever to connect evidence distributed across documents while preserving a concise, faithful generation context. Existing indexes leave two complementary gaps: chunk-based RAG can break cross-passage evidence chains, whereas an unlabeled pairwise projection without generating-topic provenance cannot jointly preserve topic-level co-participation and per-occurrence entity descriptions. We propose STITCH-RAG, a hypergraph-based framework with three coupled components. First, a semi-merged topic hypergraph encodes multi-entity co-participation as topic-summary hyperedges while retaining per-chunk entity states linked by canonical-name equivalence. Second, spatio-temporal influence bridging propagation (STIBP) combines topic-space propagation with deterministic chunk-index linkage across name-equivalent states under frequency-adaptive decay. Third, continuous STIBP scores replace binary entity-match seeds in localized Personalized PageRank (PPR). We characterize the condition under which this prior assigns more PPR mass to ground-truth evidence than a binary prior. Under the reported protocol, STITCH-RAG attains the highest reported Contain-Acc and LLM-Acc point estimates among the compared methods on HotpotQA and 2WikiMultiHopQA, and higher Recall@8 than the methods included in the standardized retrieval comparison. Results on the mixed-domain benchmark remain auxiliary preference-based evidence because only LLM-judged accuracy is available. 

---
# Beyond the Beam: Constructive Repair and Candidate Completion for Generative Recommendation 

**Authors**: Zijun Zhao, Peng Zhang, Gang Zhang, Yuanchi Ma, Hui He, Zhendong Niu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33745)  

**Abstract**: Generative recommenders retrieve items by generating identifiers, but a valid identifier can remain outside the beam after catalog expansion. This raises two connected questions: which failures can identifier assignment repair, and how should retrieval proceed beyond the initial beam? We characterize assignment repair with a fixed generator and retained old identifiers. Output-invariance certificates identify failures shared by all admissible assignments. Under a common effective prefix, coupled support and ranking constraints give the exact feasible interval of new-item counts for target recovery. Building on this characterization, Beyond the Beam (BB) obtains minimum-replacement repairs through an integral flow formulation, selects a shared map and adapts the generator. At inference, generative likelihood and collaborative evidence define one score for ranking, candidate priority and stopping. Retained prefix bounds guide candidate completion and certify its global Top-$K$ when the stopping condition is met. Exhaustive finite-catalog evaluation confirms construction in every feasible case. Across three Amazon Reviews categories and three random seeds, the full T5 procedure improves mean Recall@10 by 15.5--46.3% and NDCG@10 by 15.2--44.4% over the best-performing evaluated generative baseline for each dataset and metric. Matched controls show that shared construction and adaptation improve new-target ranking and certification efficiency on Beauty and Toys. Combined scoring and candidate completion improve NDCG@10 across all three datasets with both T5 and decoder-only LC-Rec. 

---
# From PDF to Evidence: Structure-Aware Retrieval for Clinical Practice Guidelines 

**Authors**: Xingyu Lin, Dehui Du  

**Link**: [PDF](https://arxiv.org/pdf/2609.33447)  

**Abstract**: Guideline documents are published as unstructured PDFs whose evidence is locked in visual structures---tables, flowcharts, and graded recommendations---that standard retrieval pipelines flatten into fixed-size text chunks. We cast evidence access as a document image analysis problem: parse each page image into typed structural elements, then retrieve structure-aware evidence units that follow the document's own layout (sections, table rows, flowchart paths, graded recommendations), each keeping its structural context so a result points to a specific element rather than a page. On 26 clinical practice guidelines from 9 sources (3,619 pages, Chinese and English) with 199 evidence queries, structure-aware units rank the gold element first under BM25, dense, and hybrid retrieval (hybrid Element Hit@1 of 0.382), with a significant element-level ranking gain over per-element OCR text (MRR_e +0.107, p=0.002; the Hit@5 gain is directional, p=0.17), while matching page-level recall (Page Hit@5 0.879 vs. 0.889, p=0.75) at 3.8x less context and clearly outperforming a ColPali visual-RAG baseline (PH@5 0.497). 

---
# What Gets Measured Gets Managed: Sign-aware Recommendation Needs Sign-aware Evaluation 

**Authors**: Minchan Kim, Jungmin Hwang, Hyunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2609.33346)  

**Abstract**: Sign-aware recommender systems have recently been developed to leverage negative feedback for a deeper understanding of user preferences. However, our empirical diagnosis reveals that state-of-the-art graph-based sign-aware recommender systems are paradoxically valence-blind. Even though they explicitly incorporate sign information during training, they consistently fail to differentiate liked items from disliked ones at the ranking stage, frequently infiltrating top-K recommendations with disliked content. Through linear probing, we show that while valence information exists in the learned embeddings, it remains inaccessible to the inner-product scoring function. This widespread failure remains entirely undetected because conventional evaluation metrics, such as Recall, HR, and NDCG, assign a uniform utility of zero to both negative and unobserved items, creating a systematic evaluation blind spot. To bridge this gap, we propose a family of signed metrics, Signed Recall, Signed HR, and Signed NDCG, that explicitly penalize the recommendation of disliked content. Systematic re-evaluation under our proposed metrics fundamentally reshapes the established performance landscape, revealing that methods ranked highly under conventional metrics often fail to protect users from disliked content. Finally, through a proof-of-concept auxiliary loss, we confirm that the proposed metrics provide actionable training signals, guiding models toward valence-aware behavior without sacrificing conventional relevance. For transparency, our source code is available at: this https URL 

---
# Inspire: Benchmarking Scientific Literature Search for Open Research Problems 

**Authors**: Jianrong Ding, Zhengyan Shi, Jianyuan Zhong, Kai Qiu, Qi Dai, Yifan Yang, Chong Luo, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.33233)  

**Abstract**: Scientific literature search often begins with an open research problem rather than a known target paper or a fixed candidate set. We introduce INSPIRE, a benchmark for evaluating agents that search prior literature to make progress on solution-redacted research problems. Each instance pairs a research brief with a target-specific cutoff three months before a later paper and evaluates ranked outputs against graded cited antecedents from that paper's realized research lineage. Search proceeds over an open corpus, while the identity of the target paper and membership of its cited antecedents remain hidden from the agent. Beyond end-to-end retrieval quality, INSPIRE uses logged search trajectories to distinguish three coupled stages: resource exposure, whether useful antecedents are surfaced during search; selection, whether exposed antecedents are retained; and ranking, how effectively retained papers are ordered. Across 476 computer-science targets under a shared search interface and budget, the strongest evaluated agent achieves 0.284 nDCG@10. Results show that current agents more readily recover an isolated antecedent than assemble a broader portfolio of relevant prior work. The stagewise analysis identifies resource exposure as the largest observed bottleneck, with further losses in selection and ranking. We additionally construct replay-valid hindsight demonstrations and show that they improve held-out search without changing test-time information, establishing that the benchmark provides an actionable learning signal. INSPIRE therefore enables both end-to-end comparison and stage-resolved diagnosis in a setting where the agent must construct its own working criterion of relevance. 

---
# Overview and Analysis of the RecSys Challenge 2026: Conversational Music Recommendation 

**Authors**: Seungheon Doh, Sergio Oramas, Bruno Sguerra, Abhinav Bohra, Claudio Pomo, Francesco Barile  

**Link**: [PDF](https://arxiv.org/pdf/2609.33045)  

**Abstract**: The RecSys Challenge 2026 studies conversational music recommendation as a joint item recommendation and response generation problem: given a multi-turn dialogue, systems must retrieve relevant tracks from a large catalog and produce a grounded natural-language response. This paper presents the challenge task, dataset, evaluation protocol, and official results. Beyond the leaderboard, we analyze the 16 accepted systems through a common retrieve--rerank--generate framework and examine how recommendation performance varies across users, requests, and dialogue contexts. Strong systems commonly combine heterogeneous candidate sources and preserve source-specific evidence for learned reranking. Across the system papers and our organizer-side analysis, robust design also means 1) grounding cold-start retrieval in multi-turn conversation and item signals, 2) using intent detectors, and 3) modeling the full multi-turn context rather than the current query alone. We further identify limitations of the benchmark and evaluation protocol, including single-ground-truth relevance and teacher-forced evaluation of synthetic dialogues. Together, these findings provide practical guidance for future conversational recommender systems and shared evaluation efforts. 

---
# PILAR: A Page-Grounded Unified Evidence Representation via an Entity-Linked Assertion Graph for Open-Domain QA Agents over Multimodal Document Corpora 

**Authors**: Joongmin Shin, Gyuho Shim, Jung-hun Lee, Jaehyung Seo  

**Link**: [PDF](https://arxiv.org/pdf/2609.32895)  

**Abstract**: Open-domain question answering (ODQA) over multimodal document corpora requires linking evidence scattered across text, tables, and figures. Existing systems often store these sources separately or retrieve only coarse pages, which weakens global evidence linking. We present PILAR, a page-grounded unified evidence representation instantiated as an entity-linked assertion graph. PILAR maps sentence-, table-, and figure-derived facts into a common assertion space and uses the graph as a controlled linking layer over robust page retrieval. In a shared-reader evaluation with four agent frameworks, fourteen retrieval backends, and two benchmarks, PILAR achieves the best end-to-end EM/ANLS. Gains are largest on compositional, cross-document, and multimodal questions, with a single-shot improvement of +1.6 EM over flat retrieval, rising to +2.9 on compositional and +5.9 on 3-hop questions. Ablations show that current gains are driven mainly by the text-instantiated slice of the framework, while visual assertions help only after locality-aware filtering. We therefore position PILAR as a unified evidence representation for multimodal ODQA rather than a standalone visual-reasoning module. 

---
# Mend the Measurement Gap: Latent User Preference Modeling for Short-Form Video Recommendation 

**Authors**: Shuo Chang, Yueqi Wang, Zihuan Diao, Ali Montazer, Jiangguo Zhang, Joyneel Misra, Dapeng Hong, Tomer Margolin, Sourabh Bansod, Ningren Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.32839)  

**Abstract**: Recommender systems rely heavily on heterogeneous behavioral feedback to infer user preference. Although abundant, these signals are imperfect measurements: the same observed behavior can arise from different underlying states, such as genuine enjoyment, passive consumption, or inattention. The challenge is especially acute in short-form video, where watch-based signals are strongly affected by measurement confounders such as video duration - the same watch time can imply different levels of preference for videos of different lengths, while ratio-based metrics can systematically favor short videos. As a result, optimizing raw engagement can amplify measurement artifacts rather than improving user value. We propose a Factorized Latent Value Model (FLVM) for measuring user preference from heterogeneous behavioral feedback. The model treats observed behaviors as noisy measurements of a low-dimensional, factorized latent value state and uses structured output heads to model heterogeneous feedback signals. A restricted baseline path captures predictable variation from measurement-confounding features such as video duration, user propensity, and session context, while a routed latent path estimates preference-relevant value advantage. The resulting latent value score can be integrated into an existing recommender system as a ranking feature or ranking score. On YouTube Shorts, a major short-form video platform, this model improves offline metrics and lifts a primary viewer enjoyment metric by 2.67% in online A/B tests. 

---
# RandSlot: Learning Compact Visual Document Representations with Random Soft Tokens 

**Authors**: Dewen Guo, Shi Yu, Lingxiao Zhang, Yang Zhang, Tao XU, Dan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32699)  

**Abstract**: Visual document retrieval requires expressive representations to match queries with evidence distributed across text, tables, and page layouts. Multi-vector representations capture fine-grained information, but storing and comparing many vectors introduces substantial retrieval costs. In this paper, we introduce RandSlot, a simple approach to learn compact visual document representations with random soft tokens. During training, we append independently-sampled random unit vectors to query and document input sequences and resample them at every use, without introducing learnable soft-token parameters. The encoder contextualizes these auxiliary inputs with the original content to produce a small set of retrieval vectors. A standard late-interaction objective trains the encoder to extract relevant information under varying input conditions. Experiments with different backbone models show that RandSlot improves retrieval quality over alternative readout strategies under the same vector budget. Further analysis shows that these gains can persist when random soft tokens are replaced with zeros at inference, demonstrating that random inputs during training can improve compact retrieval representations even when inference no longer requires sampling. 

---
# Concepts Complement Dense Semantics: Learning Compact Sparse Spaces for Text-Image Retrieval 

**Authors**: Yoonseo Kim, Jungwoo Choi, Cheonyoung Park, Youngwook Kim, Yongho Song, SeongKu Kang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32671)  

**Abstract**: Cross-modal retrieval has been advanced by vision-language pre-trained models that encode images and texts into a shared dense embedding space. While dense representations effectively capture overall semantic similarity, they often obscure fine-grained visual-textual information needed for precise cross-modal matching. Recent methods introduce a learned sparse branch to complement dense matching with lexical evidence, but they rely on a redundant language-model token space and lack explicit grounding for sparse dimensions. We propose GRASP, a compact and grounded sparse learning framework that mines visual-textual concepts from the corpus. A lightweight sparse head is trained to predict concepts relevant to each image or text, yielding interpretable concept-level evidence that complements dense semantic matching. Extensive experiments show that GRASP improves retrieval accuracy over the state-of-the-art dense-sparse baselines while yielding a more compact and grounded sparse space. 

---
# When Does Dense Retrieval Need Asymmetric Geometry? A Bias-Variance Theory of Shared and Dual Projections 

**Authors**: Maojun Sun, Yancheng Yuan, Jian Huang, Ruijian Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.32488)  

**Abstract**: Dense retrieval powers retrieval-augmented generation, semantic search, and question answering, yet the theoretical basis for choosing between shared and dual query-document projections remains unclear. We introduce a bias-variance theory for low-rank bilinear scoring. Shared projections induce positive-semidefinite operators, whereas dual projections realize arbitrary low-rank operators. We derive their exact approximation gap and prove a local Gaussian boundary: dual has lower risk exactly when squared directional signal exceeds the estimation cost of its additional degrees of freedom. This boundary motivates the Cross-fitted Asymmetry Risk Selector (CARS), which estimates reproducible directional signal from training pairs; its Gaussian counterpart admits exact selection-power and regret formulas. Guided by the theory, we run retrieval experiments across multiple datasets and embedding models. The mean Dual-minus-Shared NDCG@10 advantage more than doubles as query rotation increases from 0 degrees to 90 degrees. In the rank-sample-size grids, Shared wins 13 of 16 cells at n=32, whereas Dual wins all 32 cells at n=1024 and n=2048. Consistent with this shift, all 168 comparable operator-risk curves move toward Dual as training data grow. Compared to the two fixed-geometry baselines, CARS reduces held-out regret by 49-96% and achieves 90.1% mean geometry-selection accuracy. 

---
# Recipe-Matching, Not Equivalence 

**Authors**: Ali Habibullah, Mohammad Alshiekh, Yazan Alshoibi, Salman Khan, Naeemullah Khan  

**Link**: [PDF](https://arxiv.org/pdf/2609.31927)  

**Abstract**: MathNet-Retrieve asks a retriever to find, for a math problem, a document stating the same problem. An LLM under one fixed prompt writes each gold document and its near-miss distractors; LLM judges filter them. We call this procedure the "recipe", training on pairs built the same way "recipe-matching", and ask how much score it buys beyond the ability the benchmark claims to test. Two models from one base, matched in rows and settings, differ only in the training file: pairs written under the benchmark's published prompt by another vendor's LLM and judge, or computer-algebra-verified pairs with no LLM anywhere. The first leads by 45 R@1 points on the easy tier. By a non-LLM paraphrase control, half to two thirds of that gap comes from the pairs being LLM-written at all: LLM rewrites under two unrelated prompts, with the verified model's negatives, recover 30 and 22 of the 45 points; back-translations with the same negatives recover almost none. The remaining 15 to 25 points appear only under the benchmark's own prompt and vanish on real duplicates no generator wrote, the same problem in two languages. The hard tier rewards the recipe's pair structure, a deep rewrite against a minimal-edit near-miss: LLM rewrites alone score zero on it, attaching negatives unlocks it, and every negative that does so costs cross-language points; the sets scoring highest on it separate near-misses no LLM wrote worse than LLM rewrites with verified negatives. MELD also moves when a model trains on pairs built its way, without losing retention; on SABER-Math the registered attack fails, and the one gain, from its LLM-written summaries, is small but holds at a matched budget. Only on MathNet-Retrieve could we pin an inversion, benchmark score up and real retention down, to one edit of a training file. We release the generator-free duplicate evaluations, the near-miss test and three trained models. 

---
# Overview of the TREC 2025 Million Large Language Models track 

**Authors**: Evangelos Kanoulas, Panagiotis Eustratiadis, Jamie Callan, Mark Sanderson, Yongkang Li, Jingfen Qiao, Gabrielle Poerwawinata, Vaishali Pal  

**Link**: [PDF](https://arxiv.org/pdf/2609.31921)  

**Abstract**: Agentic AI envisions ecosystems of intelligent agents collaboratively solving complex tasks with minimal human intervention. In such ecosystems, each agent possesses specialized expertise, making effective expert selection central to overall system performance. While most current approaches assume a small number of well-documented models, real-world expertise is far more diverse and cannot be adequately captured through static metadata or hand-written descriptions. We anticipate a future with millions of specialized language models (LLMs), each excelling in different domains or problem types. Rather than relying on predefined capability statements, we propose a retrieval-based paradigm in which an assistant agent infers expertise dynamically by examining models' observable behavior. Upon receiving a user query, the assistant ranks candidate LLMs based on demonstrated competence, enabling efficient and adaptive expert selection. The TREC Million LLM Track operationalizes this paradigm by shifting the retrieval target from documents to expert LLMs. Participants are given a discovery set consisting of queries, answers, and log-probabilities from more than one thousand LLMs and are challenged to infer meaningful expertise representations for each model. Given an unseen test query, systems must then rank the LLMs according to their expected performance, providing the first large-scale benchmark for expertise retrieval in agentic AI. 

---
# MM-VeriRec: Failure-Guided Fusion for Verifiable Agentic Multimodal Recommendation 

**Authors**: Yufeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.31718)  

**Abstract**: Images often carry recommendation constraints that text metadata only hints at. A movie may need to look dark, a product may need a minimal style, and a visually impossible request should be rejected. Agentic multimodal recommenders must reason over text-image evidence, decide when visual evidence is decisive, and abstain when no valid action exists. We introduce MM-VeriRec, a verifiable multimodal recommendation protocol and failure-guided fusion method for hidden visual constraints, image-text mismatch, and impossible-task abstention. MM-VeriRec builds tasks from real movie-poster and product-image datasets, verifies each recommendation with deterministic visual attributes, and converts failures into actionable labels: text-trap following, visual ignorance, and false acceptance. Fusion should not merely concatenate modalities, but should diagnose which modality failed and route to the appropriate repair. Across MM-ML 1M and Amazon Reviews datasets, stronger text and vision embeddings improve retrieval but do not remove these failure modes, whereas failure-guided fusion does. The adaptive attribute gate reads the same tags the verifier checks and its scores are verifier-aligned upper bounds testing whether the taxonomy routes to the correct repair. More informative is transfer under a non-aligned gate: an independently derived leave-one-out CLIP detector still reaches 0.7028 and 0.6111 visual-grounded success, above both a VBPR baseline and plain fusion. The text-versus-visual gap reproduces across two LLM families, and the repair that helps differs by domain. MM-VeriRec is both a benchmark and a practical diagnostic loop for trustworthy agentic multimodal recommendation. 

---
# Data Processing for Offline Evaluation in Recommender Systems: a Survey 

**Authors**: Alberto Carlo Maria Mancino, Angela Di Fazio, Danilo Danese, Matteo Attimonelli, Daniele Malitesta, Antonio Ferrara, Claudio Pomo, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2609.31696)  

**Abstract**: Offline evaluation is the dominant experimental paradigm in recommender systems research, enabling reproducible and cost-effective comparisons on historical interaction data. Yet, while considerable attention has been devoted to recommendation models and evaluation methodologies, the data processing decisions that precede model training have received less scrutiny. These decisions determine the information available to recommendation algorithms and can affect the comparability and reproducibility of experimental results. This survey provides a systematic, cross-domain characterisation of data processing practices for the offline evaluation of recommender systems. We examine the data-centric pipeline, from dataset selection and interaction representation to data preparation, multimodal feature extraction, and train-validation-test splitting. Our analysis spans recommendation paradigms, including collaborative, sequential, session-based, graph-based, knowledge-aware, context-aware, multimodal, federated, cross-domain, contrastive-learning, and LLM-based recommendation. Beyond reviewing existing practices, we introduce a unified framework and taxonomy for describing data transformations and feature-extraction strategies, distinguishing data preparation from the extraction of representations from multimodal side information. Our empirical analysis reveals a landscape dominated by a narrow set of dataset-level transformations, particularly support-driven filtering, while representation-dependent transformations remain less common. We further identify substantial heterogeneity in how auxiliary information is prepared and represented, as well as inconsistencies in the specification of data splitting protocols, where similar labels may conceal different experimental conditions. 

---
# 5W1H+Which: Context-Valid Semantic Indexing with Progressive Ontology Binding 

**Authors**: Yaxiao Liu, Pengbo Liu, Yiwen Liu, Yihua Guan, Jiaxing Song  

**Link**: [PDF](https://arxiv.org/pdf/2609.35184)  

**Abstract**: Transforming raw data into queryable knowledge requires both early extraction of reusable information and explicit types, relations, and applicability conditions for particular tasks. If indexing selects content too early around a single business schema, later tasks may be unable to use information that was omitted. If the index retains only open-ended text, however, rule-based reasoning lacks checkable premises. We propose 5W1H+Which, a semantic indexing design that separates content extraction from ontology binding. The 5W1H questions organize source-grounded content units; Which points to versioned ontology elements and records mapping relations, scope, and validation status. Time, location, system environment, and participant roles are not merely retrieval labels: together, they constrain the contexts in which facts, bindings, and rules apply. Unbound content remains searchable, while bound content enters a formal reasoning path only after premise checks. The method further distinguishes business valid time, system knowledge time, and operational traces, and uses dependency records to support binding revalidation and the maintenance of derived conclusions. A worked example of migration from an on-premises server to a cloud environment illustrates the different treatment of world-state changes, ontology-version changes, and changes in rule applicability. We formulate three groups of falsifiable hypotheses concerning cross-task evidence coverage, control of contextual misuse, and incremental update cost. The planned evaluation includes a strong typed fact-graph baseline with the same evidence, temporal information, and budget, to test whether benefits arise from 5W1H organization, deferred binding, or additional information and engineering effort. The contribution is a testable indexing mechanism, not a claim to a new universal ontology or a demonstrated performance advantage. 

---
# VEX-Bench: Benchmarking Verification Complexity of LLM-Generated Misinformation 

**Authors**: Hanxun Huang, Yutao Wu, Qizhou Wang, Silvia Montaña-Niño, Yige Li, Xiang Zheng, Elif Buse Doyuran, Phoebe Matich, Xiao Liu, Xingjun Ma, Sarah Erfani, Christopher Leckie  

**Link**: [PDF](https://arxiv.org/pdf/2609.35028)  

**Abstract**: Large language models (LLMs) have made misinformation inexpensive to produce but not to verify, creating a growing asymmetry in the information ecosystem. Under tight time, labor, and budget constraints, media organizations, platforms, and fact-checkers rely on screening to prioritize which content to verify. We introduce VEX-Bench, a unified benchmark for evaluating the verification complexity of LLM-generated misinformation, as perceived during screening, across models and generation methods. Verification complexity is assessed along multiple dimensions derived from journalistic and fact-checking practices, capturing checkability, harm potential, source credibility signals, imposter legitimacy, and expected verification effort. We define the VEX score as an integrated measure combining elicitation yield and verification complexity to quantify how generated content consumes limited verification capacity. We construct a benchmark spanning two misinformation categories, 6 high-stakes domains, and 60 real-world topics, and evaluate 7 frontier LLMs and 7 generation methods, yielding 5{,}880 articles. We employ an LLM-as-judge for scalable evaluation and validate it using content-analysis methodology, including ordinal Krippendorff $\alpha$ for inter-annotator reliability, complemented by fact-checking agents for verification. Our findings show that no single method dominates all dimensions, underscoring the need for multi-dimensional evaluation. LLMs can generate high-VEX misinformation at 3$\times$ to 169$\times$ lower cost than agent-based verification. Such content is often prioritized during screening, consuming scarce verification resources and introducing a systematic risk of misallocation in resource-constrained verification systems. The code is publicly available in our \href{this https URL}{GitHub repository}. 

---
# AX is the New AEO 

**Authors**: Ido Finder, Assaf Elovic, Gad Shalev  

**Link**: [PDF](https://arxiv.org/pdf/2609.34951)  

**Abstract**: In 2023, AI models answered from training data and hallucinated when it ran out, and businesses were told to seed that knowledge. Models' training knowledge has since given way to live web search, and the advice followed it there: answer-engine optimization, or AEO, now tells businesses to scatter breadcrumbs across forum threads, listicles, and off-site citations, so AI engines are likelier to surface and recommend them. But being surfaced is no longer enough: an agent opens the results and reads them before deciding, and one buyer question sends it through several rounds of search and fetch. What decides the outcome at this drill-down step is whether the agent can fetch and read the business's own site: agent experience (AX). We argue that AX is the new AEO. We run 37,927 agent journeys, each a buyer question about a business, across four independent harnesses over 1,056 real businesses, matched on fame, prior model knowledge, and two AEO proxies, then split based on their AX level. Only 7-10% of the finished answer comes from the model's training knowledge, whether or not the site is readable. Agent-ready businesses have answers built from their own pages 78% of the time against 56% and are clearly recommended 1.9x more often, while every grounded answer about a not-agent-ready business costs the agent 64% more. Holding business, harness, and question fixed, answers built from the site are 41% more accurate. The dominant failure is not fabrication but omission: web-built answers are 3.7x more likely to contain none of the facts the buyer asked for. Baselines differ sharply across the four harnesses, with clear-recommendation rates varying sevenfold from stack to stack, yet the effect holds in every one. In the agentic web era, being readable beats being talked about, and improving a site's AX is the strongest lever a business has. 

---
# Calibrated Uncertainty for Informative Path Planning in Aquatic Environmental Monitoring 

**Authors**: Samuel Yanes Luis, Alejandro Casado Pérez, Alejandro Mendoza Barrionuevo, Dame Seck Diop, Sergio Toral Marín, Saniel Gutiérrez Reina  

**Link**: [PDF](https://arxiv.org/pdf/2609.34577)  

**Abstract**: Informative Path Planning for scalar field reconstruction uses predictive uncertainty to direct sensing vehicles toward maximally informative locations. Gaussian Processes provide this signal but their stationary isotropic kernels are misspecified for non-homogeneous phenomena such as oil spills, producing miscalibrated estimates that degrade planning. We investigate whether replacing the Gaussian Process with a well-calibrated Deep Ensemble improves path planning outcomes, and whether uncertainty quality interacts with the choice of planning algorithm. Five strategies ($\epsilon$-Greedy, Value Greedy, Uncertainty Greedy, Monte Carlo Tree Search, and Receding Horizon Orienteering) share a common Deep Ensemble backbone trained on physics-based oil spill simulations. On held-out stochastic spill scenarios, the Deep Ensemble reduces normalised reconstruction error by $83\%$ relative to the Gaussian Process baseline. Crucially, well-calibrated uncertainty amplifies the importance of the planning strategy: the performance gap between algorithms is negligible under miscalibrated models but becomes substantial under the ensemble, where multi-step lookahead planners outperform greedy selection by up to $32\%$ in reconstruction error and achieve IoU above $0.85$. Monte Carlo Tree Search is the recommended planner, matching Orienteering in reconstruction quality at an order-of-magnitude lower computational cost. 

---
# Just-In-Time Agent Memory with Runtime Agentic Research 

**Authors**: Bingyu Yan, Chaofan Li, Hongjin Qian, Shuqi Lu, Chaozhuo Li, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.34385)  

**Abstract**: Memory is critical for AI agents. Many existing agent-memory systems follow an Ahead-of-Time (AOT) design, constructing memory before a specific request arrives. While this reduces online serving cost, such request-agnostic memory construction can discard fine-grained information that later becomes important. To address this limitation, we propose Just-In-Time Agent Memory (JAM), a trainable framework for query-conditioned context construction at runtime. A Memorizer preserves complete raw histories in a hierarchical page-store with compact navigational summaries, while a Researcher iteratively retrieves, inspects, and integrates evidence for each request. To train these memory-use behaviors, we introduce Memory-Gym, an evidence-grounded data synthesis pipeline covering nine task types across six domains, and optimize the Researcher through verified-trajectory supervised fine-tuning followed by Hint-guided Group Relative Policy Optimization. We demonstrate the effectiveness of JAM across a variety of benchmarks on agent memory and long-context processing, where it achieves stronger task performance than AOT-style memory systems while remaining substantially more efficient than prior trained agentic memory approaches. To support reproducibility and future research, we release our anonymized source code at this https URL. 

---
# When Harness Beats Scale, and When Reading Beats Both 

**Authors**: Ivan Bondarenko, Nikolay O. Nikitin  

**Link**: [PDF](https://arxiv.org/pdf/2609.34366)  

**Abstract**: We describe our system for DocSem, the document-grounded quantitative reasoning shared task at DocInsights 2026, and analyze why it succeeded on labeled data and failed on the test set. The pipeline pairs hybrid block retrieval with Program-of-Thoughts (PoT) generation executed in a sandboxed interpreter, self-consistency sampling, and entity enrichment from chunk-level knowledge graphs. On our held-out split, application architecture moved the metrics far more than model scale did: PoT added 0.282 joint accuracy to a compact 7B model but at most 0.005 to a 72B model, and a 27B model with the full harness matched the 72B (0.884 vs.\ 0.873) at roughly 2.7$\times$ fewer parameters and a quarter of the CO$_2$. We read this through a distinction between world knowledge, which scales steeply with parameters, and language knowledge, which scales gently, and show that structured-output training makes a compact model harness-ready rather than merely small. On the raster, watermarked test PDFs the same system collapsed to 13.58\% joint (rank 149 of 163); a controlled re-rendering of the validation set reproduces the OCR half of the collapse while bounding what the simulation misses. Auditing the physical nature of evaluation inputs precedes architecture, and the leaderboard's bimodality is consistent with reading quality, not reasoning, having separated the field. 

---
# When Does Selection Replace Extraction? A Pre-Registered Test of Agent Memory with a Typed Decision Model 

**Authors**: Rishabh Sharma, Rishika Lall  

**Link**: [PDF](https://arxiv.org/pdf/2609.34227)  

**Abstract**: Does conversational memory need LLM-extracted facts, or is selecting the right raw turns enough? Published results disagree. Extraction-based systems report gains from distilled facts. Recent studies find raw history with good ranking does as well, but disagree about whether ranking matters. We ran a pre-registered study on held-out LoCoMo conversations and LongMemEval. At a tight budget on LoCoMo, raw turns selected by a single call to Jev, a typed decision model, are non-inferior to an LLM-extraction memory (one-sided 95% bound -3.0 points against a -5-point margin). Blind human grading narrows the margin but does not change the result. Raw turns cost 3,061 times less to write, and the result holds with a second answer model. Within this study, reranking's gain shrinks as the budget grows. It adds 17.4 points on LoCoMo and 9.1 on LongMemEval when three of 30 candidates are kept. At generous budgets it adds 1.5 and 1.1, and extraction systems are more accurate. This suggests why published results disagree. At matched context, Jev selects as accurately as an LLM reranker (non-inferiority bound -2.0) at a third of the latency, and more accurately than a multi-call graph traversal. Reranking lowers correct abstention. Plans, code and graded answers are released. 

---
# High-Level Text Preprocessing for Semantic Similarity Analysis of Discursive Texts: A Framework and Empirical Demonstration 

**Authors**: Mehmet Murat Albayrakoglu, Mehmet Nafiz Aydin  

**Link**: [PDF](https://arxiv.org/pdf/2609.33983)  

**Abstract**: Semantic Textual Similarity (STS) methods assume that a document's lexical content faithfully represents what it asserts. This assumption fails for discursive documents that discuss, compare, critique, and contextualize other positions in the process of articulating their own. The result is semantic diffusion: similarity scores between documents are inflated by vocabulary acquired through discursive engagement rather than substantive alignment. Standard Natural Language Processing (NLP) preprocessing (tokenization, stopword removal, stemming, lemmatization) cannot address this problem because it operates at the lexical level, treating all content identically regardless of its discursive function. This paper introduces high-level text preprocessing: a systematic, rule-based intervention applied before the standard preprocessing pipeline to isolate each document's actual claim from its discursive structure. We propose 12 rules, each with an explicit rationale, and demonstrate their effect on an encyclopedic philosophical corpus: three entries from the Stanford Encyclopedia of Philosophy (virtue ethics, deontological ethics, and consequentialism). A three-phase experiment using eight Transformer-based STS models shows that preprocessing reduces centroid cosine similarity scores across all three theory pairs, with 23 of 24 model-pair comparisons showing the expected decrease and cross-model agreement ranging from 7-1 to 8-0. We introduce the semantic diffusion index (SDI), a per-document metric for assessing the semantic reorientation between a document's raw and high-level preprocessed representations. Although the framework is demonstrated using philosophical texts, it potentially addresses a domain-agnostic problem applicable to legal texts, policy documents, academic articles, and any genre in which a discursive approach introduces vocabulary from positions the document does not endorse. 

---
# Beyond Fixed Features: Architecture-Dependent Sensitivity to Node Representations under Heterophily 

**Authors**: Priyanath Maji, Sidharth Gaur, Rajavinoth Paul Durai  

**Link**: [PDF](https://arxiv.org/pdf/2609.33764)  

**Abstract**: Graph Neural Networks (GNNs) perform well on homophilic graphs but struggle in heterophilic settings, where connected nodes often carry dissimilar labels. Existing evaluations typically compare architectures under a fixed node-feature representation, leaving unclear whether conclusions about heterophily robustness remain stable as the input representation changes. We address this question by constructing parallel feature variants of two large-scale heterophilic benchmarks, Roman-Empire and Amazon-Ratings, pairing each graph with representations ranging from static fastText vectors to contextual Transformer embeddings and evaluating seven GNN architectures across these representations. We find that the effect of representation varies across architectures: on Roman-Empire, the contextual gain ranges from 2.38 percentage points for GCN-sep to 13.67 points for GAT, with H2GCN gaining 8.77 points. On Amazon-Ratings, where node text is limited to short product titles, GAT improves by 6.78 points from fastText to MPNet, while GCN-sep changes by only 0.20 points. These results show that architectural performance is conditional on node representation: the same representation change can produce different magnitudes of performance gain across architectures, so architecture and representation cannot be treated as independent evaluation factors. A rank-correlation analysis on these two benchmarks further shows that the relative ordering of architectures remains highly stable across representations, isolating differential sensitivity, rather than ranking instability, as the primary effect. 

---
# Concurrent Coded Signal-Multiplexing Ranging for Half-Duplex Asynchronous Networks 

**Authors**: Zijian Zhang, Yuan Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.33753)  

**Abstract**: Signal-multiplexing network ranging (SM-NR) shares broadcasts across node pairs, but its sequential operation leads to a ranging cycle that grows linearly with network size. This paper proposes a concurrent coded SM-NR (CC-SM-NR) framework for asynchronous half-duplex networks. Firstly, the CC-SM-NR protocol coordinates concurrent transmissions through binary transmit-listen codewords. The transmit-listen schedule defined by these codewords ensures reciprocal observations subject to a finite concurrency limit. Then, we derive the exact minimum number of transmit-listen rounds without a concurrency limit, which reveals that the minimum grows logarithmically with network size. To account for practical scenarios, we establish the necessary and sufficient conditions for the constant-weight feasibility of codewords under a finite concurrency limit. Subsequently, we propose a low-complexity scheduling algorithm that achieves the minimum round count within the constant-weight codeword class. To support higher observation redundancy, this scheduling design is extended through a greedy construction. Finally, simulation results demonstrate the effectiveness of the proposed schemes for network ranging. 

---
# Learning Multimodal Embeddings with Evidence-Aligned Readout 

**Authors**: Zirong Chen, Fuda Ye, Enjun Du, Junfu Pu, Xinlei Wang, Xinyu Zuo, Lisheng Duan, Haijin Liang, Jin Ma, Jiachuan Wang, Yongqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.33659)  

**Abstract**: Multimodal large language models can expose task-relevant evidence through generation, but producing useful evidence does not by itself determine how it enters a retrieval embedding. We study whether the semantic organization of that evidence can also specify where representations are read. To address this question, we introduce EviAlign, which couples Semantic Evidence Generation with Boundary Readout in a shared multimodal large language model. It organizes evidence into five semantic units, reads the contextualized state at each unit boundary, and aggregates these states into a single normalized embedding. Generation and contrastive retrieval objectives jointly train this shared structure. With the same trailing readout, semantic evidence and free-form CoT yield nearly identical retrieval performance, suggesting that evidence organization alone does not explain the full gain. A controlled $2\times3$ study compares consistent and permuted evidence organization across three readout strategies, using training targets with matched evidence spans. With five readout states and the same mean pooling, the advantage of consistent semantic organization grows from 0.65 points at length-based training positions to 2.39 at evidence boundaries, yielding a 1.74-point co-design interaction. Across 12 MMEB retrieval tasks, EviAlign achieves 76.9 average Recall@1 with 500K training pairs while retaining single-vector indexing and scoring. 

---
# Robust Hierarchical Structures for Agentic Document Analysis 

**Authors**: Ruiying Ma, Yiming Lin, Aditya G. Parameswaran  

**Link**: [PDF](https://arxiv.org/pdf/2609.33322)  

**Abstract**: Large Language Models (LLMs) enable us to better understand text documents, including PDFs and Word documents. However, LLMs, as well as more modern LLM agents, i.e., those with tool-calling abilities, typically treat such documents as plain text, ignoring the fact that they are often organized hierarchically into sections and subsections. Extracting this structure, while difficult, can improve efficiency and effectiveness for agents (and humans)---since only sections relevant to a given task need to be processed. Unfortunately, prior work on structure extraction provides no formal guarantees on how well the inferred structure matches the true one. Instead, we target a robust and compact variant that is feasible to infer and useful in practice. Robustness ensures that the text under each subsection header is a superset of the text under the same header in the true structure. Compactness seeks to minimize this superset, reducing agentic cost (or human cognitive load). We propose SHED, a two-stage workflow for inferring a robust and compact structure. The first stage is pluggable with an infinite family of approaches, each guaranteeing robustness for a specific document class. We theoretically characterize the document space using these classes and their hierarchical relationships. Empirically, SHED improves F-1 scores (measuring the robustness--compactness trade-off) by 13%--68% over non-LLM baselines and 9%--15% over expensive LLM-based approaches. Finally, we show how SHED-inferred structures are valuable for agentic document analysis: agents using SHED outperform baselines, achieving 3%--23% higher accuracy while being up to 10x cheaper. 

---
# Algorithmic Harms Associated with Generative Model-Augmented Recommendation Systems 

**Authors**: Christine Herlihy, Xumei Xi, Shloka Desai, Kevin Bannerman Hutchful, Pedro Silva  

**Link**: [PDF](https://arxiv.org/pdf/2609.33073)  

**Abstract**: In this work, we consider algorithmic harms that may arise as generative models are incorporated into machine learning platforms. We argue that existing harm taxonomies and threat models require extension to (1) address novel causal drivers of well-studied representational and quality-of-service harms; and (2) anticipate and mitigate endogenous harms, such as sanitization, which may arise when system inputs are misaligned with the system designer's objectives, or the generative model's inductive priors. To this end, we introduce an expanded taxonomy of algorithmic harms associated with the use of generative models in non-conversational recommendation systems. In addition, we offer a causal analysis of how problematic subsets of the (input, output) joint distribution can arise, in an effort to inform harms detection and mitigation efforts. 

---
# Retrieved but Not Delivered: Multimodal Memory Delivery for Long-Term Agents 

**Authors**: Yuhang Jiang, Qingwei Liao, Kaize Yin, Xingling Liu, Luca Cuomo, Silvio Bacci  

**Link**: [PDF](https://arxiv.org/pdf/2609.32590)  

**Abstract**: Work on memory for multimodal agents optimizes what is written, updated and retrieved. Between retrieval and the answer, however, is a stage that multimodal memory evaluations do not isolate: what of the retrieved memory reaches the model, and in what form. We call it delivery, and a controlled decomposition on MemLens locates the remaining room there. With the retrieved evidence set exactly fixed, delivering the original pixels instead of withholding them raises accuracy by 13.87 points on an 8B backbone, whereas making retrieval perfect on those same messages improves it by 2.31. Delivery is the larger term on all three MemLens backbones and grows with backbone strength; retrieval grows too, without closing the gap. We propose DeliverMem, an instantiation of delivery as three decisions: keep the original modality, give each item a readable identity, and state when it was seen, with a retrieval-side adapter for the one property delivery cannot supply. Each is measured against a delivery-matched control that alters only its own variable. DeliverMem leads the strongest published memory agent on MemLens at all four context lengths, and beats DMV-Bench's own strongest method at every setting on both backbones. On MemLens it does this on a tenth to a seventieth of the input. Each decision helps only where the question lacks what it supplies, and is null elsewhere. A single fixed configuration nonetheless leads both benchmarks, without training any component or modifying the stored records. Project page: this https URL 

---
# Beyond Dyadic Memory: Interaction-Aware Multimodal Memory with Adaptive Agentic Retrieval for Multi-Party Spoken Conversations 

**Authors**: Wenxu Jia, Xize Cheng, Zihan Zhang, Dongjie Fu, Linjun Li, Wenshi Chen, Yangyang Wu, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2609.32522)  

**Abstract**: Long-term memory enables agents to accumulate information and reason across sessions, yet existing research primarily focuses on dyadic text or image-text conversations, leaving long-term memory for multi-party spoken conversations underexplored. This setting requires preserving conversational content, identifying participants across sessions, and retaining who speaks to whom. To this end, we propose VoxPolyMem, an interaction-aware multimodal memory framework combining incremental speaker identification with a memory hierarchy comprising interaction memory, fact memory, and participant profiles. We formulate retrieval as sequential decision-making, where an agent rewrites queries and selects retrieval tools and memory layers based on accumulated evidence to address information gaps. We further introduce Evidence-Gain GRPO (EG-GRPO), which uses round-wise credit assignment to encourage complementary evidence acquisition. We also construct VoxPolyBench to evaluate memory evolution, personalized answering, memory retrieval and reasoning, and interaction reasoning and attribution in multi-party spoken conversations. VoxPolyMem achieves an overall score of 85.0 on VoxPolyBench, surpassing the strongest evaluated baseline by 23.6 points. On Mem-Gallery and H2HMem-Multi, it scores 89.6 and 74.4, respectively, exceeding the strongest evaluated public memory baselines by over 8 points each. These results highlight its potential for persistent, personalized assistance in multi-party multimodal interactions. Code and datasets are available at this https URL 

---
# Graph Memory: Spectral Associative Memory via Dirichlet Energy 

**Authors**: Zhaoyang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2609.32365)  

**Abstract**: Dense associative memories have traditionally focused on storing and retrieving vector-valued patterns. Many modern machine learning problems, however, are naturally graph-structured, requiring memory mechanisms for relational patterns, graph diffusion geometries, community structures, and graph-based inductive biases. We propose a spectral dense associative memory for storage and retrieval of graph data, extending the classical vector-valued memories. Retrieval is performed through a log-sum-exp energy induced by Dirichlet energy with spectral norm distances, producing a softmax-weighted average of the stored Laplacians that remains a valid graph Laplacian. We prove exponential storage capacity and exponentially decaying retrieval error. Beyond graph retrieval, we establish theoretical guarantees for spectral quantities central to graph learning, including eigenvalues, eigenspaces, and diffusion operators. Experiments on synthetic graph data, real-world airline network, protein conformation data and wearable sensor data demonstrate robust graph retrieval while preserving the graph geometry of the data. Our framework provides a new associative memory paradigm for graph-structured data and bridges dense associative memory with modern graph learning and generative AI. 

---
# DP-Rec: Towards Dynamic Patching for Efficient Long-Sequence Recommendation 

**Authors**: Dwipam Katariya, Thomas Caputo, Akshat Shreemali, Juan Manuel Origgi, Nikita Seleznev, Pranab Mohanty, Kalanand Mishra, Nam Nguyen, James Montgomery  

**Link**: [PDF](https://arxiv.org/pdf/2609.32215)  

**Abstract**: Transformers have redefined sequential recommendation by effectively modeling dynamic user behaviors and long-range dependencies. However, they remain inherently inefficient: standard architectures operate at a fixed rate, allocating comparable computation to every item in a user's history regardless of its information content. This leads to prohibitive computational overhead on long sequences and increased sensitivity to behavioral noise. To address this, practitioners often resort to lossy sequence compression, staged modeling, or truncation. This limits the model's ability to leverage the full context of long histories during inference. Inspired by the recent success of Byte Latent Transformer, we propose DP-Rec, a dynamic latent patching architecture for recommendation. DP-Rec shifts from item-level modeling to patch-level modeling by segmenting interaction sequences using contrastive entropy surprise to identify informative behavioral boundaries. A lightweight patch encoder compresses these temporally contextualized segments into a reduced set of dynamic latent behavior vectors, which are then processed by a larger latent transformer and decoded for next-item prediction. Extensive experiments show that, under constrained computational budgets, DP-Rec scales effectively to long sequences and achieves a superior efficiency-accuracy trade-off over both non-compressed and fixed-size compression baselines. 

---
# Rethinking Cross-Channel Importance in Time-Series Forecasting 

**Authors**: Yong-Hoon Choi, Kwang-Hyun Park, Youngjin Cho  

**Link**: [PDF](https://arxiv.org/pdf/2609.32187)  

**Abstract**: Cross-channel modeling is central to multivariate time-series forecasting, yet channels that are statistically related, predictively useful, and actually used by a trained forecaster are often treated as if they defined the same notion of importance. We show that they need not coincide. Cross-channel dependency structures change substantially across future offsets, and horizon-adaptive source selection improves a controlled Ridge predictor in 21 of 32 dataset--prediction-length conditions, with a mean gain of $5.16\%$. This selected-set signal also transfers to a matched nonlinear predictor. Yet imposing the same horizon-specific source logic on iTransformer yields only 11 of 20 wins and a mean gain of $0.208\%$, with little alignment between controlled and neural gains. Functional interventions further show that strong forecasters use cross-channel information, while their source-reliance rankings agree little with controlled utility or with one another across iTransformer, TimesNet, and a cross-channel TimeMixer. As a constructive consequence, bounded post-hoc support improves a frozen channel-independent forecaster in 12 of 16 dataset--horizon conditions, with a positive aggregate bootstrap interval. Cross-channel importance should therefore be interpreted relative to the forecasting mechanism and question that define it: related $\neq$ useful $\neq$ used. 

---
# On Evaluating and Improving Conversational Agents in Production 

**Authors**: Kasra Hosseini, Wen-Sen Cheng, Marco-Andrea Buchmann, Emir Mulabegovic, Weiwei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.32092)  

**Abstract**: We present a framework for evaluating and improving a large-scale, multi-agent shopping assistant in production, and report lessons from its use. Offline evaluation of such a system faces three obstacles. (i) A logged conversation cannot be replayed against a modified system, because a different response changes every turn that follows. (ii) The unchanged system itself varies from run to run. Its LLM components are stochastic, and in product search the available products, their prices, and the customer's personalization signals change. (iii) Aggregate quality scores combine distinct behaviors, so they show that quality has changed but not which behavior caused the change. Our framework addresses each obstacle in turn. For a reported behavior, an Evaluation Harness generates targeted assertions and a fixed cohort of customer scenarios. It then reproduces the behavior in a local instance of the assistant through grounded user simulation. Instead of replaying the log, the simulator writes new customer turns conditioned on the recorded messages and context. Repeated runs of the unchanged system form a stored baseline. An Improvement Orchestrator turns the assertion results into hypotheses, implements each as an isolated modification, and compares it with the baseline using paired percentile bootstrap intervals over scenario-level differences. When an investigation ends, the harness may propose revisions to future evaluations, subject to human approval and without altering past decisions. We report production investigations with this framework. Assertion profiles showed which positions of a product carousel a failure affected, and repeated runs distinguished a real improvement from run-to-run fluctuation. Audits of the evaluation itself found a judge that lacked the evidence it needed and a model setting that was configured but not applied. 

---
# Overview and Analysis of the RecSys Challenge 2026: Conversational Music Recommendation 

**Authors**: Seungheon Doh, Sergio Oramas, Bruno Sguerra, Abhinav Bohra, Claudio Pomo, Francesco Barile  

**Link**: [PDF](https://arxiv.org/pdf/2609.33045)  

**Abstract**: The RecSys Challenge 2026 studies conversational music recommendation as a joint item recommendation and response generation problem: given a multi-turn dialogue, systems must retrieve relevant tracks from a large catalog and produce a grounded natural-language response. This paper presents the challenge task, dataset, evaluation protocol, and official results. Beyond the leaderboard, we analyze the 16 accepted systems through a common retrieve--rerank--generate framework and examine how recommendation performance varies across users, requests, and dialogue contexts. Strong systems commonly combine heterogeneous candidate sources and preserve source-specific evidence for learned reranking. Across the system papers and our organizer-side analysis, robust design also means 1) grounding cold-start retrieval in multi-turn conversation and item signals, 2) using intent detectors, and 3) modeling the full multi-turn context rather than the current query alone. We further identify limitations of the benchmark and evaluation protocol, including single-ground-truth relevance and teacher-forced evaluation of synthetic dialogues. Together, these findings provide practical guidance for future conversational recommender systems and shared evaluation efforts. 

---
# PILAR: A Page-Grounded Unified Evidence Representation via an Entity-Linked Assertion Graph for Open-Domain QA Agents over Multimodal Document Corpora 

**Authors**: Joongmin Shin, Gyuho Shim, Jung-hun Lee, Jaehyung Seo  

**Link**: [PDF](https://arxiv.org/pdf/2609.32895)  

**Abstract**: Open-domain question answering (ODQA) over multimodal document corpora requires linking evidence scattered across text, tables, and figures. Existing systems often store these sources separately or retrieve only coarse pages, which weakens global evidence linking. We present PILAR, a page-grounded unified evidence representation instantiated as an entity-linked assertion graph. PILAR maps sentence-, table-, and figure-derived facts into a common assertion space and uses the graph as a controlled linking layer over robust page retrieval. In a shared-reader evaluation with four agent frameworks, fourteen retrieval backends, and two benchmarks, PILAR achieves the best end-to-end EM/ANLS. Gains are largest on compositional, cross-document, and multimodal questions, with a single-shot improvement of +1.6 EM over flat retrieval, rising to +2.9 on compositional and +5.9 on 3-hop questions. Ablations show that current gains are driven mainly by the text-instantiated slice of the framework, while visual assertions help only after locality-aware filtering. We therefore position PILAR as a unified evidence representation for multimodal ODQA rather than a standalone visual-reasoning module. 

---
# Mend the Measurement Gap: Latent User Preference Modeling for Short-Form Video Recommendation 

**Authors**: Shuo Chang, Yueqi Wang, Zihuan Diao, Ali Montazer, Jiangguo Zhang, Joyneel Misra, Dapeng Hong, Tomer Margolin, Sourabh Bansod, Ningren Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.32839)  

**Abstract**: Recommender systems rely heavily on heterogeneous behavioral feedback to infer user preference. Although abundant, these signals are imperfect measurements: the same observed behavior can arise from different underlying states, such as genuine enjoyment, passive consumption, or inattention. The challenge is especially acute in short-form video, where watch-based signals are strongly affected by measurement confounders such as video duration - the same watch time can imply different levels of preference for videos of different lengths, while ratio-based metrics can systematically favor short videos. As a result, optimizing raw engagement can amplify measurement artifacts rather than improving user value. We propose a Factorized Latent Value Model (FLVM) for measuring user preference from heterogeneous behavioral feedback. The model treats observed behaviors as noisy measurements of a low-dimensional, factorized latent value state and uses structured output heads to model heterogeneous feedback signals. A restricted baseline path captures predictable variation from measurement-confounding features such as video duration, user propensity, and session context, while a routed latent path estimates preference-relevant value advantage. The resulting latent value score can be integrated into an existing recommender system as a ranking feature or ranking score. On YouTube Shorts, a major short-form video platform, this model improves offline metrics and lifts a primary viewer enjoyment metric by 2.67% in online A/B tests. 

---
# RandSlot: Learning Compact Visual Document Representations with Random Soft Tokens 

**Authors**: Dewen Guo, Shi Yu, Lingxiao Zhang, Yang Zhang, Tao XU, Dan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32699)  

**Abstract**: Visual document retrieval requires expressive representations to match queries with evidence distributed across text, tables, and page layouts. Multi-vector representations capture fine-grained information, but storing and comparing many vectors introduces substantial retrieval costs. In this paper, we introduce RandSlot, a simple approach to learn compact visual document representations with random soft tokens. During training, we append independently-sampled random unit vectors to query and document input sequences and resample them at every use, without introducing learnable soft-token parameters. The encoder contextualizes these auxiliary inputs with the original content to produce a small set of retrieval vectors. A standard late-interaction objective trains the encoder to extract relevant information under varying input conditions. Experiments with different backbone models show that RandSlot improves retrieval quality over alternative readout strategies under the same vector budget. Further analysis shows that these gains can persist when random soft tokens are replaced with zeros at inference, demonstrating that random inputs during training can improve compact retrieval representations even when inference no longer requires sampling. 

---
# Concepts Complement Dense Semantics: Learning Compact Sparse Spaces for Text-Image Retrieval 

**Authors**: Yoonseo Kim, Jungwoo Choi, Cheonyoung Park, Youngwook Kim, Yongho Song, SeongKu Kang  

**Link**: [PDF](https://arxiv.org/pdf/2609.32671)  

**Abstract**: Cross-modal retrieval has been advanced by vision-language pre-trained models that encode images and texts into a shared dense embedding space. While dense representations effectively capture overall semantic similarity, they often obscure fine-grained visual-textual information needed for precise cross-modal matching. Recent methods introduce a learned sparse branch to complement dense matching with lexical evidence, but they rely on a redundant language-model token space and lack explicit grounding for sparse dimensions. We propose GRASP, a compact and grounded sparse learning framework that mines visual-textual concepts from the corpus. A lightweight sparse head is trained to predict concepts relevant to each image or text, yielding interpretable concept-level evidence that complements dense semantic matching. Extensive experiments show that GRASP improves retrieval accuracy over the state-of-the-art dense-sparse baselines while yielding a more compact and grounded sparse space. 

---
