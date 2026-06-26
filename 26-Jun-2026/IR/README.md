# NOVA: A Verification-Aware Agent Harness for Architecture Evolution in Industrial Recommender Systems 

**Authors**: Shaohua Liu, Liang Fang, Yilong Sun, Shudong Huang, Qingsong Luo, Xiaoyang Chen, Dongqiang Liu, Chuangang Ma, Zhenzhen Chai, Henghuan Wang, Shijie Quan, Changyuan Cui, Zhangbin Zhu, Peng Chen, Wei Xu, Lei Xiao, Haijie Gu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27243)  

**Abstract**: Industrial advertising recommender models are continuously improved through architecture evolution. Upgrades such as RankMixer, TokenMixer-Large, and MixFormer show that better structures remain a key source of quality and business gains. Yet developing such upgrades in production is expert-intensive and difficult to scale. Existing automation is insufficient: AutoML mainly tunes hyper-parameters, while effective gains often require cross-module changes under strict constraints; generic LLM coding agents optimize for runnable code, but runnable code does not imply a valid recommender architecture. Candidates may pass local tests while causing silent failures that degrade performance.
We present NOVA, a level-aware agent harness for verification-aware architecture evolution. NOVA uses an architecture gradient, an SGD-inspired, non-differentiable update signal that aggregates prior modifications, verification diagnostics, metric feedback, and trajectory memory to guide the next modification. A verification cascade checks structure semantics, local executability, offline effectiveness, and online impact; invalid candidates are blocked early, with failure patterns recorded as forbidden directions. L1--L4 task-level control matches automation to task complexity and risk, routing high-risk tasks to Copilot for human oversight. Deployed in an industrial advertising system, NOVA achieves the highest effective pass rate on L2 ScaleUp and L3 Literature-to-Production tasks (54.5% and 60.0%), reduces silent failures compared with coding-agent baselines, and shortens one literature-to-production cycle by over 13x in human-attended time. In online A/B testing, the selected L3 candidate improves GMV on three pCVR objectives by +1.25%, +1.70%, and +2.02%, while reducing pCVR bias by 58.8%, 66.7%, and 37.3%. 

---
# TRUST: Item-Calibrated Interval Evidence for Temporal Session-Based Recommendation 

**Authors**: Linjiang Guo, Nitin Bisht, Shiqing Wu, Yifan Yin, Guandong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27214)  

**Abstract**: Temporal signals have been widely used in session-based recommendation to infer user interest. Existing temporal session-based recommenders primarily rely on absolute interval values, implicitly assuming that the same interval carries similar interest signals across items. However, we empirically find that this assumption does not hold: each item has its own interval distribution, so an interval should be interpreted relative to the item it belongs to. Based on this observation, we propose TRUST, a framework that evaluates each observed interval relative to the empirical interval distribution of the corresponding item. Specifically, we propose a score function to guide global neighbor sampling, session graph encoding, and final interest aggregation. Experiments on public datasets show that TRUST consistently improves over representative temporal and non-temporal baselines, and plug-in experiments further show that the proposed scoring function can improve existing temporal session recommenders as a model-agnostic method. Component-wise ablations further show that calibrating the temporal signals within each module, rather than removing the module itself, consistently improves neighbor sampling, session graph encoding, and interest aggregation. 

---
# UniFormer: Efficient and Unified Model-Centric Scaling for Industrial Recommendation 

**Authors**: Bo Chen, Jinlong Jiao, Tijian Hu, Ruihao Zhang, Yanzhi Liu, Chenghou Jin, Qinglin Jia, Baixuan He, Hechang Pan, Yiwu Liu, Jian Liang, Chaoyi Ma, Ruiming Tang, Han Li, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2606.27058)  

**Abstract**: Recently, substantial progress has been made in industrial recommendation through component-centric model scaling, where individual components such as behavior modeling, feature interaction, or task modeling are independently scaled to improve model capacity. Although recent methods such as HyFormer and OneTrans further explore cross-module co-scaling by jointly modeling behavior and interaction, their designs are still confined to the feature space and lack a unified model-centric scaling framework over the overall modeling space. In this paper, we propose UniFormer, an efficient and unified model-centric scaling framework for industrial recommender systems. To improve efficiency, UniFormer decomposes the overall modeling space into feature and task spaces, which are modeled by stacked Feature-space Interaction Modules and Task-space Interaction Modules, respectively. Moreover, UniFormer introduces semantic-based tokenization scheme to enable user-item decoupling, thereby achieving request-level inference acceleration. To prevent preference collapse, UniFormer employs multi-sequence cross-attention to separately capture heterogeneous behavior patterns, followed by the self-attention to enhance interaction modeling. Besides, dedicated multi-view FFNs are introduced to support flexible and scalable parameter scaling across different modeling components. Extensive online A/B testing in two production scenarios, Kuaishou and Kuaishou Lite, shows that UniFormer consistently improves user engagement and interaction metrics, achieving gains of +0.101%/+0.260% in App Stay Time and +0.729%/+1.113% in Watch Time, respectively. 

---
# TriPAH: Imbalance-Aware Tri-Prompt Affinity Hashing for Cross-Modal Medical Retrieval 

**Authors**: Jiaming Bian, Songming Li, Yurui Song, Yunfei Chen, Yichao Cao, Jun Long  

**Link**: [PDF](https://arxiv.org/pdf/2606.27010)  

**Abstract**: In the era of big medical data, efficient cross-modal retrieval is pivotal for evidence-based diagnosis and large-scale case management. Cross-modal medical hashing retrieval aims to enable efficient image-text search and support downstream tasks such as case-based reasoning and decision support by learning compact, semantically aligned binary codes. However, current methods suffer from semantic fragmentation due to noisy clinical language, long-tailed labels, and brittle quantization that weakens alignment. We propose TriPAH, a Tri-Prompt Affinity Hashing framework. TriPAH synthesizes ontology-grounded, patient-level prompts conditioned on normalized clinical cues to yield low-noise textual representations for initial alignment. A lightweight prompt-token mixer performs hierarchical, multi-granularity alignment and produces quantization-ready features under an asymmetric multi-task objective coupling multi-positive contrastive alignment, imbalance-aware classification, and progressive quantization regularization. A patient-level consistency module further stabilizes codes across complementary views. Extensive experiments on three public datasets demonstrate that TriPAH significantly outperforms state-of-the-art methods. 

---
# A Shared IPTC Topic Space for Cross-Source Topic Modelling 

**Authors**: Din Iskakov, Sebastian Gonçalves, Marco Idiat, Mendeli Vainstein, Aline Villavicencio, Ronaldo Menezes, Rodrigo Wilkens  

**Link**: [PDF](https://arxiv.org/pdf/2606.26845)  

**Abstract**: Comparing topic attention across different media is hindered by a fundamental modelling problem: topic models fitted separately to each corpus produce corpus-specific topic spaces that cannot be aligned directly. This paper presents a reproducible framework that places corpora in a single shared topic space defined by a taxonomy. Discovered topics are obtained with guided BERTopic, scored against the ninety-four IPTC Media Topics' taxonomy topics (level-1) through weighted keyword and target centroids, and then collapsed upward to seventeen IPTC parent topics by a maximum-similarity rule. The framework was developed and selected on a controlled New York Times 2011 corpus through a narrowing sequence: a broad model screen, a focused mapping refinement, a strict finalist comparison, a target-construction ablation, and a threshold calibration. In this corpus, the guided family retained substantially stronger mapped coverage than a zero-shot benchmark under stricter assignment thresholds, a parent-enriched target construction improved both coverage and parent consistency, and coverage declined gradually rather than collapsing as the assignment threshold was tightened. The contribution is an externally anchored method for constructing a shared topic space that enables reproducible cross-source topic comparison. 

---
# Attributed, But Not Incremental: Cannibalization-Corrected Attribution for Large-Scale Advertising 

**Authors**: Donghui Li, Bowen Yuan, Zili Yang, Qinxin Chen, Lijing Song  

**Link**: [PDF](https://arxiv.org/pdf/2606.26690)  

**Abstract**: In large-scale paid acquisition and growth advertising systems, production attribution outputs are widely used for daily budget allocation and channel diagnosis. However, paid-attributed conversions such as daily new users (DNU) may systematically overstate true incremental growth when paid channels overlap with organic demand, brand-driven traffic, or other acquisition channels. This attribution-cannibalization mismatch can distort incremental ROI measurement and budget decisions at scale.
We propose an experiment-calibrated attribution correction framework that uses incrementality experiments as causal anchors to convert sparse lift measurements into daily correction estimates. To make the corrected signal actionable at production granularity, we further allocate calibrated cannibalization volume across business hierarchies under structural consistency constraints. Offline forward-in-time validation against channel-level incrementality experiment readouts shows that the proposed framework substantially reduces calibration error relative to raw attribution and fine-grained ML baselines. Deployed across multiple global TikTok markets, the system supported budget and traffic strategy adjustments that were followed by an approximately 15-percentage-point reduction in the measured cannibalization rate. 

---
# GPUSparse: GPU-Accelerated Learned Sparse Retrieval with Parallel Inverted Indices 

**Authors**: Ashutosh Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2606.26441)  

**Abstract**: Learned sparse retrieval models such as SPLADE achieve retrieval quality competitive with dense models while preserving the interpretability and exact-match advantages of sparse representations. However, inference-time scoring still relies on CPU-bound inverted index traversal algorithms (WAND, Block-Max WAND), creating a fundamental bottleneck for real-time serving at scale. We present GPUSparse, a system for GPU-accelerated exact learned sparse retrieval that introduces: (1) a GPU-parallel inverted index with block-aligned, warp-coalesced posting lists; (2) a batched scatter-add scoring algorithm that processes hundreds of queries simultaneously; and (3) fused Triton kernels with an analysis of the tradeoff between work-efficiency and hardware utilization. On MS MARCO passage ranking (8.8M passages) with real SPLADE embeddings, GPUSparse matches CPU exact scoring to three decimals (MRR@10=0.383, equal to Pyserini SPLADE at this precision; Recall@1000>=0.999 vs. dense matmul, the residual from floating-point tie-breaking) while providing a 235x speedup over Pyserini CPU at 8.8M documents (1.27ms vs. 298ms per query). Compared to Seismic (the fastest CPU sparse retrieval system), which trades 25% recall for speed (R@1000=0.738 vs. 0.983 exact), GPUSparse achieves exact scoring at 787 QPS throughput (batch 500) on the full 8.8M collection, with 1.3ms per query. Our document-parallel kernel reaches 62.6% of H100 peak HBM bandwidth, revealing a fundamental work-efficiency vs. bandwidth-efficiency tradeoff in GPU sparse retrieval. The reformulation of sparse scoring as scatter-add over an inverted index is shared with SPARe's iterative mode; our contribution is its fused-kernel realization, which we measure to be 23-270x faster than a faithful SPARe iterative reimplementation. 

---
# TileMaxSim: IO-Aware GPU MaxSim Scoring with Dimension Tiling and Fused Product Quantization 

**Authors**: Ashutosh Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2606.26439)  

**Abstract**: Multi-vector retrieval models such as ColBERT achieve state-of-the-art accuracy through fine-grained token-level MaxSim scoring, yet existing GPU implementations leave most hardware performance unused. We give a roofline analysis of MaxSim on modern GPUs and identify a severe bandwidth gap: naive implementations reach only 5-18% of peak HBM bandwidth because they materialize the Nq x Nd similarity matrix, wasting memory traffic on data that is consumed once and discarded. We present TileMaxSim, a family of IO-aware Triton kernels that close this gap via (1) multi-query SRAM tiling that streams document embeddings through shared memory while accumulating per-query-token maxima in registers, reading each embedding from HBM exactly once; (2) dimension tiling that partitions the embedding dimension into 128-wide chunks, enabling scoring for d > 128 embeddings that overflow shared memory; and (3) fused product-quantization scoring via shared-memory lookup tables, cutting HBM I/O by up to ~31x. On NVIDIA H100 GPUs, TileMaxSim reaches 80.2% of peak HBM bandwidth and scores 82M documents/second (71.6M/s on real MS MARCO passages), a 220x speedup over loop-based scoring, 6.5x over fused PyTorch, 6.6-8.5x over this http URL, and 469x the scoring throughput of WARP's CPU engine on the same node. TileMaxSim preserves exact retrieval quality: on MS MARCO and three BEIR benchmarks, rankings match reference MaxSim. As a drop-in replacement in ColBERTv2/PLAID, it cuts scoring latency at 100K candidates from 268 ms to 1.2 ms (98% lower end-to-end latency). We further show constant throughput from 100K to 500K documents, data-parallel multi-GPU sharding, robustness across dimensions 64-768, and FP16/BF16/FP32 support. Concurrent work independently develops an IO-aware fused MaxSim kernel; we differ in dimension tiling for d > 128 and fused product-quantization scoring. 

---
# Scoring Is Not Enough: Addressing Gaps in Utility-fairness Trade-offs for Ranking 

**Authors**: Shubham Singh, Ian A. Kash, Mesrob I. Ohannessian  

**Link**: [PDF](https://arxiv.org/pdf/2606.26369)  

**Abstract**: Scoring functions are used to represent the relevance of individual documents. In modern information retrieval or recommendation systems, they are often learned from data and play a pivotal role in ranking sets of documents or items in a way that maximizes utility to a query or user. With the recent interest in algorithmic fairness, the success of scoring has naturally led to methods that learn scores that simultaneously trade off fairness and utility. In this work, we show that in stark contrast with utility-centric objectives, scoring is sub-optimal in achieving all utility-fairness trade-offs. We establish this with a series of counter-examples with a generic fairness formulation. We show that the issue persists whether we have a deterministic scoring function or a randomized one, or whether we measure fairness at the scope of a single query or across multiple queries. On the positive side, we empirically demonstrate that semi-greedy post-processing has the potential to achieve much better trade-offs, often approaching the ideal of exhaustive post-processing in a tractable way. 

---
# From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations 

**Authors**: Dianjing Fan, Yao Li, Kyaw Hpone Myint, Dwipam Katariya, Alexandre G.R. Day, Pranab Mohanty, Giri Iyengar  

**Link**: [PDF](https://arxiv.org/pdf/2606.26277)  

**Abstract**: Sequential user behavior modeling is widely adopted in industrial recommender systems; however, significant gaps remain in financial services, where pre-login web interactions and authenticated in-app experiences differ drastically. Specifically, pre-login web users typically explore new products, whereas logged-in app users focus on account servicing. Due to the challenge of cross-channel entity resolution (e.g., matching anonymous web sessions to authenticated mobile accounts), web-based intent signals remain underutilized for post-authentication personalization. Existing methods for capturing web-based intent are often ad-hoc and narrow, lacking the flexibility to support both quantitative downstream recommendations and qualitative understanding at scale. In this work, we propose a scalable and dual-purpose intent prediction framework for web-based interactions and demonstrate its applicability for personalization. Our approach transforms raw web clickstreams into two outputs: a self-supervised Transformer encodes multi-modal clickstreams into a compact session embedding, while an LLM-based taxonomy generation and distillation pipeline produces interpretable intent labels. Our system demonstrates that self-supervised clickstream representations combined with LLM-distilled taxonomies can jointly serve quantitative tasks and qualitative understanding in production: on the mobile homepage tile ranking task, the session embedding improves macro Recall@1 by 1.88% and reduces Log Loss by 13.38% over production baselines. On the user conversion prediction task, the embedding outperforms the LLM labels by 4.3% on micro F1, while the distillation layer delivers interpretable labels at ultra-low latency with only a 7% performance drop. 

---
# Reducing Redundancy in Whole-Slide Image Patching for Scalable Indexing and Retrieval 

**Authors**: Jialiang Geng, Ghazal Alabtah, Saghir Alfasly, Wataru Uegami, H.R.Tizhoosh  

**Link**: [PDF](https://arxiv.org/pdf/2606.26157)  

**Abstract**: The rapid growth of digital pathology has created an urgent need for efficient indexing and retrieval of whole slide images (WSIs). This need is intensified by emerging generative AI workflows, particularly retrieval-augmented generation (RAG), which require dependable similarity search to support high-stakes clinical decision-making. Yet the substantial cost of high-performance storage limits the scalability and accessibility of WSI indexing for many healthcare institutions. Consequently, methods that can reduce storage demands while preserving retrieval accuracy have become a critical research priority. We propose ARReST (Antithetical Redundancy Reduction Strategy), a principled oppositional framework that leverages redundancy across dissimilar tissue classes to markedly decrease the number of patches that must be indexed from each WSI. Instead of eliminating only within-class duplicates, ARReST identifies antithetical patches-those whose representations contribute minimally to cross-class discrimination-and prunes them from the searchable archive. This targeted reduction substantially compresses the index without sacrificing morphological diversity or retrieval fidelity. By minimizing superfluous patch representations, ARReST reduces storage footprint, lowers computational overhead, and accelerates similarity search across large pathology repositories. Extensive experiments on TCGA repository (The Cancer Genome Atlas with 21 organs) demonstrate that ARReST achieves significant index compression while maintaining competitive retrieval performance. The observed storage savings of 3% to 60% (14%$\pm$13%) can be reliably achieved without compromising retrieval performance for many organs. The proposed strategy enables scalable, cost-efficient WSI indexing and is well-suited for next-generation retrieval-driven clinical AI systems. 

---
# AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems 

**Authors**: Changxin Lao, Fei Pan, Guozhuang Ma, Han Li, Huihuang Lin, Jijun Shi, Kangzhi Zhao, Kun Gai, Mo Zhou, Qinqin Zhou, Quan Chen, Ruochen Yang, Shifu Bie, Shuang Yang, Shuo Yang, Wenhao Li, Wentao Xie, Xiao Lv, Xuming Wang, Yijun Wang, Yiming Chen, Yusheng Huang, Zhongyuan Wang, Zibo Zhao, Zijie Zhuang, Baoning Xia, Chao Liu, Chaoyi Ma, Chubo He, Dawei Cong, Feng Jiang, Gang Wang, Guilin Xia, Hanwen Xu, Jiahong Xie, Jiahui Qiao, Jian Liang, Jiangfan Yue, Jing Wang, Jinghan Yang, Jinghui Jia, Kan Qin, Lei Wang, Ming Li, Peilin Song, Pengbo Xu, Qiang Luo, Ruiming Tang, Shiyang Liu, Shuxian Jin, Tao Wang, Tao Zhang, Xiang Gao, Xianghan Li, Yingsong Luo, Yiwen Ning, Yongcheng Liu, Yuan Guo, Zhaojie Liu, Zhenkai Cui  

**Link**: [PDF](https://arxiv.org/pdf/2606.26859)  

**Abstract**: Recommendation algorithm iteration is moving from an artisanal, engineer-bound process toward an industrialized research loop, but this transition remains blocked by a structural execution bottleneck: the idea-to-launch cycle still depends on human engineers to generate hypotheses, modify production code, launch A/B experiments, and attribute online results. Innovation therefore scales linearly with headcount rather than compounding with evidence, compute, and accumulated experimental knowledge. We present AgentX, a production-deployed multi-agent system that fundamentally restructures this production function. AgentX operates as a self-evolving development engine: it autonomously generates, implements, evaluates, and learns from recommendation experiments at a scale and pace that no manual workflow can sustain.
The system orchestrates four tightly coupled stages in a closed loop. A Brainstorm Agent synthesizes evidence from historical experiments, system architecture, data analysis, and external research into ranked, executable proposals. A Developing Agent translates each proposal into production-ready code through repository-grounded generation and multi-dimensional reliability verification. An Evaluation Agent conducts safe online rollout with guardrail-vetoed A/B judgment, converting both successes and failures into structured knowledge assets. A Harness Evolution layer (SGPO) then distills execution trajectories into semantic-gradient updates that continuously sharpen the agents themselves -- making the system not merely automated, but self-improving. 

---
# From Vajrayana Tara to Bengali Baul: A Computational Study of Lexical Transmission Across Buddhist, Shakta, and Vaishnava Traditions in Bengal 

**Authors**: Joy Bose  

**Link**: [PDF](https://arxiv.org/pdf/2606.26803)  

**Abstract**: We present a computational corpus study of vocabulary relationships across eight tradition layers of Bengali and Sanskrit devotional literature spanning the 8th to 19th centuries, encompassing Buddhist Vajrayana, Shakta Tantra, Vaishnava, and Baul traditions. Using a corpus of 75 texts and TF-IDF character n-gram vectorization with cosine similarity analysis, we address the historically argued but previously unquantified claim that Buddhist Vajrayana vocabulary survived the collapse of the Pala monasteries and was absorbed into the Shakta Tantra tradition of Bengal. The central finding is a specificity result: the Gitagovinda (Vaishnava Sanskrit, 12th century) has zero cosine similarity to Shakta Kali texts, while Bridge Tara texts (Buddhist-Shakta transitional, same century, same language) have cosine similarity 0.54 to Shakta Kali. This 8.5-fold contrast between two Sanskrit traditions from the same century demonstrates that the Buddhist-Shakta vocabulary overlap is not a generic property of Sanskrit devotional literature but is specific to the Buddhist-Shakta transmission chain. Three Brihannilatantra Tara texts show Shakta-to-Buddhist vocabulary ratios of 2.0 to 4.0, constituting measurable evidence of lexical transition within that chain. Ramprasad Sen's 18th-century Bengali Kali songs preserve Buddhist vocabulary residue including 56 occurrences of Tara alongside 103 occurrences of Kali. The Vaishnava Bengali tradition contributes a parallel chain to modern Baul vocabulary (similarity 0.29), slightly weaker than the Buddhist Sahajiya chain via Charyapada (0.31). These results provide the first quantitative multi-tradition corroboration of historically argued Buddhist-Shakta syncretism in Bengal. 

---
# ConvMemory v3: A Validity Context Layer for Conversational Memory via Target-Conditioned Relation Verification 

**Authors**: Taiheng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2606.26753)  

**Abstract**: Conversational memory retrieval optimizes relevance, yet a retrieved memory can be relevant and simultaneously outdated: a later turn updates, corrects, or supersedes it. ConvMemory v3 adds a validity context layer that detects and surfaces this update evidence through target-conditioned relation verification, sitting after the v1/v2 retrieval path. The core mechanism is a dual-evidence gate that conditions a relation judgment on the specific target proposition, scoring a (target, source) pair through the product of a MiniLM slot head and a DeBERTa-v3 slot head and gating it by conservative event/operation evidence. On a synthetic multi-hop validity benchmark the gate reaches 90.12% +/- 1.73 accuracy; through a real-data feedback loop that mines failure patterns but trains on synthetic pairs only, the verifier transfers to Memora role binding with zero target-side labels, reaching 98.8% +/- 0.9 group-all-correct. The deployed layer preserves retrieval by default: a context mode attaches structured validity metadata while keeping the candidate set and rank order fixed, and a query-conditioned demote mode is an explicit opt-in for dense current-state workloads, where it raises current-active H@1 from a never-demote baseline of 45.1% to 95.7% +/- 1.2 while protecting non-superseded memories at 99.4% recall. Six machine-verifiable safety contracts pin the layer's behavior. Multi-hop graph propagation is validated as a mechanism; fully automatic construction of strict prerequisite edges is characterized as a boundary, since strict necessity requires counterfactual world knowledge. This report extends ConvMemory v1 (arXiv:2605.28062) and v2 (arXiv:2606.10842). 

---
# SocialPersona: Benchmarking Personalized Profiling and Response with Multimodal Social-Media Context 

**Authors**: Qinkai Zhang, Yanyan Zhao, Xin Lu, Yulin Hu, Pengtao Han, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2606.26654)  

**Abstract**: Personalized language-model assistants are often evaluated through a memory lens: can a model recall preferences users have explicitly stated in dialogue? More comprehensive personalization demands a harder capability -- inferring what users care about from the multimodal traces they naturally leave behind. We introduce SocialPersona, a benchmark for evaluating whether multimodal large language models (MLLMs) can recover revealed preferences from longitudinal social-media timelines and use them in dialogue. Built from longitudinal timelines of 171 everyday, non-promotional social-media users, SocialPersona contains text, images, timestamps, and 2,597 human-verified preference tags across seven interest domains, separating stable interests from recent interests. It supports two tasks: constructing structured user profiles from multimodal context and generating responses aligned with inferred profiles. Experiments with proprietary and open-weight MLLMs show that models can identify broad interest domains, yet their performance drops on fine-grained and recent interests and degrades further when inferred profiles must be used to personalize dialogue. Together with evidence that text and images provide complementary preference signals, these results indicate that robust cross-modal, long-horizon user modeling remains a key challenge, and that SocialPersona can help measure and advance progress toward assistants that infer and act on revealed preferences. 

---
# Utilizing Cognitive Signals Generated during Human Reading to Enhance Keyphrase Extraction from Microblogs 

**Authors**: Xinyi Yan, Yingyi Zhang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26485)  

**Abstract**: Microblogging platforms generate massive amounts of short, noisy, and dispersed user content, making automatic keyphrase extraction (AKE) an important but challenging task. Prior studies have used eye-tracking signals to improve microblog-based AKE because such signals reflect readers' attention to salient words. However, eye tracking alone is limited by physiological, acquisition, and feature-decoding constraints. To address this issue, we investigate whether electroencephalogram (EEG) signals can complement eye-tracking signals for AKE. Using the ZuCo cognitive language processing corpus, we select 8 EEG features and 17 eye-tracking features and incorporate them into microblog-based AKE models. To reduce possible distortion of cognitive signals by model structures, we inject these features into the input of the soft-attention layer and the query vectors of the self-attention layer. We then evaluate different combinations of cognitive signals across AKE models. The results show that cognitive signals produced during reading consistently improve AKE performance, regardless of feature combinations and model architectures. EEG features bring the largest gains, while combining EEG and eye-tracking features yields performance between the two individual signal types, suggesting partial complementarity but also possible redundancy or noise. These findings indicate that EEG signals provide useful cognitive evidence for microblog-based AKE and that multimodal cognitive signals deserve further investigation. 

---
# Extracting Problem and Method Sentence from Scientific Papers: A Context-enhanced Transformer Using Formulaic Expression Desensitization 

**Authors**: Yingyi Zhang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26481)  

**Abstract**: Billions of scientific papers lead to the need to identify essential parts from the massive text. Scientific research is an activity from putting forward problems to using methods. To learn the main idea from scientific papers, we focus on extracting problem and method sentences. Annotating sentences within scientific papers is labor-intensive, resulting in small-scale datasets that limit the amount of information models can learn. This limited information leads models to rely heavily on specific forms, which in turn reduces their generalization capabilities. This paper addresses the problems caused by small-scale datasets from three perspectives: increasing dataset scale, reducing dependence on specific forms, and enriching the information within sentences. To implement the first two ideas, we introduce the concept of formulaic expression (FE) desensitization and propose FE desensitization-based data augmenters to generate synthetic data and reduce models' reliance on FEs. For the third idea, we propose a context-enhanced transformer that utilizes context to measure the importance of words in target sentences and to reduce noise in the context. Furthermore, this paper conducts experiments using large language model (LLM) based in-context learning (ICL) methods. Quantitative and qualitative experiments demonstrate that our proposed models achieve a higher macro F1 score compared to the baseline models on two scientific paper datasets, with improvements of 3.71% and 2.67%, respectively. The LLM based ICL methods are found to be not suitable for the task of problem and method extraction. 

---
# 3D Spatial Pattern Matching 

**Authors**: Nicole R. Schneider, Avik Das, Lukas Arzoumanidis, Abhijeet Ghodgaonkar, Hanan Samet, Youness Dehbi  

**Link**: [PDF](https://arxiv.org/pdf/2606.26465)  

**Abstract**: Spatial pattern matching is the process of matching query entities and constraints with database entities and relations. It has many applications, including similar region search, housing market search, landmark search, and road network matching. To our knowledge, all existing spatial pattern matching approaches frame the problem in a 2 dimensional space, where entities lie in a cartesian plane and relationships defined between them are contained in 2 dimensions. However, this problem framing has significant limitations when searching for real world entities that have height in addition to position. To address this limitation, we extend spatial pattern matching to 3 dimensions and provide a generalized definition of the problem. We describe a subgraph matching algorithm capable of resolving 3D spatial patterns over distance relations and release two 3D spatial pattern matching datasets, one synthetic and one containing real 3D building data from the city of Hamburg, Germany. We test our subgraph matching algorithm on both datasets and present results as a baseline for future methods to build upon. 

---
# ProvenAI: Provenance-Native Traces of Evidence in Generated Answers 

**Authors**: Mohammad Faizan, Dalal Alharthi  

**Link**: [PDF](https://arxiv.org/pdf/2606.26449)  

**Abstract**: Retrieval-augmented systems routinely present citations alongside generated answers, yet a citation does not confirm that the corresponding source meaningfully shaped the output. This paper introduces ProvenAI, a framework that decomposes transparency in multi-hop question answering into three independently measurable layers: answer correctness, citation fidelity against benchmark supporting evidence, and per-document influence under leave-one-resource-out intervention. Targeting the HotpotQA distractor benchmark through a seven-stage pipeline covering data normalisation, retrieval indexing, citation-aware answer generation, attribution auditing, ablation-based influence estimation, batch evaluation, and interactive inspection, ProvenAI evaluates 7,405 validation examples drawn from a canonical corpus of 509,300 passages. The system achieves 53.53% answer accuracy alongside a mean citation-fidelity score of 71.55%, and a worked example surfaces what we call the citation-influence gap: a clean citation audit co-occurring with a profile in which one cited source registers only weak influence while seven uncited sources demonstrably shift the output. We formalise the relationship between the implemented surface proxy and a token-level KL-divergence target through a stated faithfulness condition, ground the framework in causal-mediation analysis and database-provenance theory, and discuss how the three measurement layers compose with cryptographic provenance architectures emerging in autonomous scientific discovery. ProvenAI establishes that meaningful transparency in retrieval-grounded QA requires traceable links across retrieved, cited, and behaviourally influential evidence as three distinct, independently measured layers. 

---
# Hybrid privacy-aware semantic search: SVD-truncated document geometry and CKKS-encrypted query reranking under a restricted threat model 

**Authors**: Sergey Kurilenko  

**Link**: [PDF](https://arxiv.org/pdf/2606.26373)  

**Abstract**: Dense embeddings power semantic search and retrieval-augmented generation, but embedding-inversion attacks can reconstruct source text from a vector: when a vector database leaks, the documents behind it leak too. The textbook defences are extremes - encrypting the whole search homomorphically is sound but too slow at million-document scale, while privacy noise degrades ranking long before it protects. We study a middle path exploiting the asymmetry between the static collection and the dynamic query. The collection is protected geometrically: each vector is truncated onto a lower-dimensional SVD subspace and rotated by a secret orthogonal transform known only to the owner. The query is protected cryptographically: it is reranked under CKKS homomorphic encryption, so an honest-but-curious server never sees the query or the scores. CKKS parameters come from a small offline benchmark.
We prove a tight lower bound on the reconstruction error of any attacker confined to the protected subspace. On one million documents and five encoders the scheme preserves ranking quality (slightly improving it on strong encoders, as a linear denoiser) at sub-second latency, and an off-the-shelf inversion attack on the protected space collapses to the noise floor. We then test stronger adversaries: a known-plaintext attacker recovers the rotation by orthogonal Procrustes from about as many leaked pairs as the retained dimension; the public product-quantization codes preserve most nearest-neighbour structure; and random-projection, calibrated-noise and BEIR baselines show the truncation is an encoder-dependent accuracy cost, not a free denoiser. We state the limits: query confidentiality is cryptographic, but document protection is an empirical obfuscation layer (SVD truncation plus a secret rotation), not a cryptographic primitive, and we delimit the threat model for each claim. 

---
# Instruction Bleed: Cross-Module Interference in Prompt-Composed Agentic Systems 

**Authors**: Ching-Yu Lin, Yifan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.26356)  

**Abstract**: Practitioners of prompt-composed agentic systems report a recurring failure mode: editing one prompt module silently shifts the behavior of others despite no shared variable or executable dependency. We formalize this as compositional behavioral leakage (CBL): interference between modules sharing a context window. CBL is enabled by architectural non-isolation: transformer self-attention provides no formal boundary between concatenated modules. We probe CBL on a deployed job-evaluation agent (Claude Sonnet 4.6, 144 trials) through a reusable three-channel protocol that perturbs non-focal modules along volume, content, and form. Only the content channel produces a detectable paired effect (Cohen's d = 0.63, bootstrap 95% CI excluding zero); no recommendation flipped -- a sub-threshold regime invisible to standard QA but compounding across the thousands of decisions a deployed agent makes. CBL is orthogonal to known agent-failure axes (adversarial injection, cognitive degradation, multi-agent fault propagation, privacy leakage). We contribute an operational definition, a reusable protocol, a falsifiable prediction set, and a system-class characterization, establishing cross-module interference measurement as a requirement for prompt-composed agent evaluation. 

---
# Lacuna: A Research Map for Machine Learning 

**Authors**: Martin Weiss, Miles Q. Li, Alejandro H. Artiles, Yacine Mkhinini, Chris Pal, Hugo Larochelle, Nasim Rahaman  

**Link**: [PDF](https://arxiv.org/pdf/2606.26246)  

**Abstract**: Lacuna is a research map for machine learning that uses LLMs to turn papers and scholarly metadata into markdown summaries, concept elements, research directions, and research proposals. Each item keeps links to the primary source records and papers that support it. We release the map with web, markdown, and MCP interfaces. Across LitSearch, Multi-XScience-CS/ML, and ScholarQA-CS-ML, Lacuna outperforms OpenScholar with the strongest gains on LitSearch retrieval (Recall@10 0.538 vs. 0.424 for OpenScholar v3). We also evaluate Lacuna Deep Research, a multi-stage report agent over the map, on 25 ReportBench-ML survey tasks: Lacuna Deep Research reaches 0.052 citation F1, 0.339 citation precision, 99 expert-reference hits, and 7.82/10 RACE report quality, while GPT-Researcher reaches 0.039 F1, 0.290 precision, 72 hits, and 5.24/10 RACE. 

---
