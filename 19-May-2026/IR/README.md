# Traditional statistical representations outperform generative AI in identifying expert peer reviewers 

**Authors**: Vicente Amado Olivo, Tereza Jerabkova, Jakub Klencki, John Carpenter, Mario Malički, Ferdinando Patat, Louis-Gregory Strolger, Wolfgang Kerzendorf  

**Link**: [PDF](https://arxiv.org/pdf/2605.18752)  

**Abstract**: The exponential growth of scientific submissions has strained the peer review system. Despite the rapidly expanding global pool of researchers, this unprecedented scale has rendered the previous approach of manual expert identification unfeasible. Therefore, institutions have naturally turned to Large Language Models (LLMs) to automate intricate processes like expert reviewer identification. However, the reliability of these new models in accurately identifying domain experts lacks rigorous evaluation. We conduct a comprehensive empirical evaluation of statistical and AI-driven expertise identification methodologies to benchmark their reliability and limitations. Framing expert identification as an information retrieval problem, we utilize the distributed peer review system of a major international astronomical observatory, where proposal authorship serves as our proxy ground truth for domain expertise. Evaluating six retrieval methodologies utilized across observatories and computer science conferences, we demonstrate that traditional statistical representations outperform generative AI. Specifically, Term Frequency-Inverse Document Frequency successfully identified a labeled expert within the top 25 recommendations 79.5% of the time, compared to 51.5% for GPT-4o mini. Our results highlight that distinguishing subfield expertise requires fine-grained vocabulary, which is obscured by the semantic smoothing in generative methods. By establishing a rigorous evaluation framework for automated peer review, we demonstrate that transparent and reproducible statistical representations still outperform computationally expensive LLMs in specialized scientific tasks. 

---
# Improving BM25 Code Retrieval Under Fixed Generic Tokenization: Adaptive q-Log Odds as a Drop-In BM25 Fix 

**Authors**: Santosh Kumar Radha, Oktay Goktas  

**Link**: [PDF](https://arxiv.org/pdf/2605.18561)  

**Abstract**: In retrieval-augmented coding, failures often begin when the relevant file is absent from the retrieved context. Under frozen generic tokenization, where a BM25 index has been built by a search system whose analyzer the practitioner does not control, this failure is routine: BM25's logarithmic RSJ-odds IDF under-separates the identifier tail that distinguishes one function from another. We replace the outer logarithm of the Robertson-Spärck-Jones odds with a q-logarithm. At q=1 the transform recovers BM25 exactly by L'Hôpital's rule, and for q<1 it is a Box-Cox transform of the RSJ odds with lambda = 1-q. On CoIR CodeSearchNet Go (182K documents), oracle-tuned NDCG@10 rises from 0.2575 to 0.4874 (absolute +0.2299; +89.3% relative; zero sign reversals in 10,000 paired-bootstrap resamples, reported as p <= 10^-4). The effect is graded across code languages and is near-zero on BEIR text. A one-parameter closed form estimates a corpus-level q from hapax density and stays near q=1 on corpora where BM25 is already optimal. The index-time cost is a single pass over the sparse score matrix and query latency is unchanged. A tokenizer ablation shows that identifier-aware tokenization largely removes the incremental gain from q-IDF. 

---
# TIGER-FG: Text-Guided Implicit Fine-Grained Grounding for E-commerce Retrieval 

**Authors**: Xinyu Sun, Huangyu Dai, Lingtao Mao, Zexin Zheng, Zihan Liang, Ben Chen, Chenyi Lei, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2605.18434)  

**Abstract**: E-commerce image search often takes a cropped image as the query, while each candidate is represented by full item images and structured text. This image-to-multimodal retrieval setting presents two asymmetries: a modality disparity -- a visual query must match image--text items, and a granularity disparity -- a cropped query must be compared with full images containing background context and possible distractors. Detection-based pipelines handle the granularity disparity through explicit localization but incur extra cost and error propagation, whereas CLIP-style encoders avoid detection, but are vulnerable to backgrounds or irrelevant items. To address these limitations, we propose TIGER-FG, a text-guided implicit fine-grained grounding framework for image-to-multimodal e-commerce retrieval. TIGER-FG uses item text as semantic guidance to produce target-focused item representations without object detection for retrieval. We further introduce dual distillation objectives that preserve target-region spatial consistency and query--item similarity structure, yielding more stable and discriminative multimodal representations. In addition, we construct ECom-RF-IMMR, a realistic benchmark suite with a 10M-pair training set and two evaluation benchmarks covering standard and cluttered item layouts. TIGER-FG improves Recall@1 over the strongest baseline by 6.1 and 34.4 percentage points on the two evaluation benchmarks, respectively, with only 85.7M query-side parameters and 256-dim embeddings. Results on public e-commerce benchmarks further demonstrate its generalization to noisy and one-to-many retrieval scenarios. Code and data will be released. 

---
# RCTEA: Richness-guided Co-training for Temporal Entity Alignment 

**Authors**: Jiayun Li, Wen Hua, Shiqi Fan, Fengmei Jin, Haiyang Jiang, Xue Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.18255)  

**Abstract**: Temporal Entity Alignment (TEA), which aims to identify equivalent entities across Temporal Knowledge Graphs (TKGs), is crucial for integrating knowledge facts from multiple sources. However, existing TEA models often fail to capture the orthogonal yet complementary effects between structural and temporal features, and typically overlook the importance of information richness, a key factor for effective message passing in neural feature encoders. To address these limitations, we propose the RCTEA framework, which jointly models both structural and temporal aspects of TKGs for entity alignment. Specifically, we design a richness-guided attention mechanism along with an adaptive weighting strategy to facilitate effective feature fusion. To ensure robust alignment despite noisy entity contexts, we introduce a dual-view neighborhood consensus algorithm that jointly refines the feature encoders to enforce local structural consistency of the predicted alignments. Extensive experiments demonstrate the superiority of RCTEA, achieving state-of-the-art performance on public TEA benchmarks. 

---
# PIPER: Content-Based Table Search via profiling and LLM-Generated Pseudoqueries 

**Authors**: Riccardo Terrenzi, Matteo Falconi, Serkan Ayvaz, Pierluigi Plebani  

**Link**: [PDF](https://arxiv.org/pdf/2605.18199)  

**Abstract**: The rapid growth of tabular datasets in data lakes, data spaces, and open data portals makes effective dataset search essential for reuse and analysis. Existing search systems rely mainly on metadata, which is often incomplete or low quality, especially for tables whose meaning depends on both schema and cell values. Recent advances in Large Language Models (LLMs) enable richer, content-based representations of tables. However, prior LLM-based retrieval methods have focused on Table Question Answering, where the goal is to select a single table to answer a question, rather than retrieve and rank relevant datasets. We propose PIPER, a content-driven retrieval method for tabular datasets that uses table profiles and LLM-generated queries embedded for dense retrieval. Designed for dataset search in poor-metadata settings, PIPER outperforms both classical metadata-based baselines and strong TableQA retrieval methods, demonstrating the value of LLM-based content modeling for tabular dataset search. 

---
# Modality-Aware Identity Construction and Counterfactual Structure Learning for ID-Free Multimodal Recommendation 

**Authors**: Hongjian Ma, Wenxin Huang, Yan Zhang, Zhifei Li, Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18044)  

**Abstract**: Multimodal recommendation has attracted extensive attention by leveraging heterogeneous modality information to alleviate data sparsity and improve recommendation accuracy. Existing methods have attempted to replace ID embeddings with multimodal features and have achieved promising preliminary results. However, these methods still exhibit the following two limitations: (1) the reconstructed ID representations remain relatively static and fail to fully exploit multimodal semantics; and (2) the graph learning process is insufficient in mining latent long-tail semantic relations and is easily affected by popularity bias. To address these issues, we propose a novel method named Modality-Aware Identity Construction and Counterfactual Structure Learning for ID-free Multimodal Recommendation (MAIL). Specifically, we design a modality-aware identity construction module that dynamically modulates positional encodings with multimodal semantics to construct content-aware ID-free identity representations. Then, we propose a counterfactual structure learning paradigm that mines low-exposure semantic neighbors via popularity penalization and alleviates popularity bias. Extensive experiments are conducted on five public Amazon datasets. Experimental results show that MAIL achieves average improvements of 7.81% in Recall@10 and 12.81% in NDCG@10 compared with the baseline models. Our code is available at this https URL. 

---
# Towards Sustainable Growth: A Multi-Value-Aware Retrieval Framework for E-Commerce Search 

**Authors**: Yifan Wang, Yixuan Wang, YiDan Liang, Qiang Liu, Fei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.17994)  

**Abstract**: New item growth is critical for maintaining a healthy ecosystem in large-scale e-commerce platforms. However, existing systems tend to prioritize presenting users with already popular items, a phenomenon often referred to as the "Matthew effect". In the context of search retrieval, current cold-start models suffer from the misalignment between training objectives and online business metrics, and they lack effective mechanisms to measure an item's growth potential. In this paper, we propose a Multi-Value-Aware retrieval framework tailored for e-commerce search, designed to better align with the cascaded online values across different stages of the search system while balancing immediate conversion and long-term item growth. Our framework GrowthGR consists of two key components: an Item Long-term Transaction Value Prediction (ItemLTV) module and a Multi-Value-Aware Generative Retrieval (MultiGR) module. First, in the ItemLTV module, we employ counterfactual inference to quantify the long-term value increment attributable to a single user interaction. Second, in the MultiGR module, building upon a semantic-ID-based generative retrieval architecture, we leverage structured samples with the search cascade signals and adopt a Multi-Value-Aware Policy Optimization (MoPO) training paradigm to align with multi-stage online values, while explicitly balancing short-term transactional value and long-term growth potential estimated by ItemLTV. We successfully deployed GrowthGR on Taobao's production platform, achieving a substantial 5.3% lift in new item GMV while delivering a non-trivial 0.3% gain in overall search GMV. Extensive online analysis and A/B testing demonstrate its positive impact on the overall ecosystem value. 

---
# Text-Video Retrieval With Global-Local Contrastive Consistency Learning 

**Authors**: Xiaolun Jing, Xinxing Yang, Genke Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17959)  

**Abstract**: Text-video retrieval aims to find the most semantically similar videos with given text queries. However, since videos contain more diverse content than texts, the main semantics expressed by each text-video pair is often partially relevant. The primary methods involve the utilization of language-video attention module to align texts and videos. Though effective, this paradigm inevitably introduces prohibitive computational overhead, resulting in inefficient retrieval. In this paper, we propose a simple yet effective method called Global-Local Contrastive Consistency Learning (GLCCL) to achieve texts and videos semantics alignment. Specifically, we design a parameter-free Global-Local Interaction Module (GLIM) to generate semantic-related frame and video features in a text-guided manner. Furthermore, a Contrastive Score Consistency (CSC) loss is developed to promote consistency learning among different scores on positive pairs and suppress consistency learning on negative pairs. Empirical evidence suggests that CSC loss provides the model with robust discriminative power between positives and hard negatives. Extensive experiments on three benchmark datasets, including MSR-VTT, DiDeMo and VATEX, demonstrate the effectiveness and superiority of our approach. 

---
# DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems 

**Authors**: Yiqing Yang, Xinlong Zhao, Zhao Liu, Xiao Lv, Ruiming Tang, Han Li, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2605.17863)  

**Abstract**: Watch-time prediction is a central regression task in short-video recommender systems, where labels are highly long-tailed and residual errors vary systematically across observed watch-time regions. In practice, a model may appear globally calibrated while still overestimating short views and underestimating long views, because opposite errors cancel out in aggregate. Existing methods mainly improve the first-stage watch-time predictor, but often leave such residual distributional bias insufficiently corrected. We propose DADF, a distribution-aware debiasing framework for watch-time regression. Instead of replacing a deployed predictor, DADF performs second-stage multiplicative residual correction on top of it. DADF combines three complementary designs: a dynamic distribution-aware transformation for stabilizing long-tailed correction targets, a debias-factor-aware module for modeling heterogeneous residual patterns using inference-time observable factors, especially video duration, and a multi-label-aware module that exploits auxiliary prediction signals from engagement heads. We evaluate DADF on public short-video benchmarks and a large-scale industrial ranking system. DADF consistently improves both pointwise accuracy and ranking quality across datasets and backbones. In the industrial setting, it achieves a 1.88 percentage-point WUAUC gain over the production baseline, reduces MAE by 12.57%, and yields a statistically significant 0.347% lift in average time spent per device in online A/B testing. These results demonstrate that DADF effectively mitigates local calibration bias and provides a practical plug-in solution for debiasing long-tailed continuous targets. The source code is available at this https URL. 

---
# Uncertainty-Calibrated Recommendations for Low-Active Users 

**Authors**: Bob Junyi Zou, Sai Li, Tianyun Sun, Wentao Guo, Qinglei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.17788)  

**Abstract**: A fundamental challenge in recommender systems is balancing reliability for Low-Active Users (LAUs) with diversity for High-Active Users (HAUs). The key to this balance lies in quantifying model uncertainty, which approximates the risk of prediction errors and reveals the limits of the model's current knowledge. On large-scale short-video and livestream platforms, model uncertainty can warn of low-quality recommendations that may lead to disengagement of LAUs and at the same time identify opportunities to diversify content recommendation for HAUs. To leverage this dichotomy, we introduce a unified, production-ready framework that calibrates uncertainty to drive differentiated strategies. Specifically, we implement a model-uncertainty-based risk-averse deboosting policy for LAUs to suppress unreliable recommendations, while employing a risk-seeking Upper Confidence Bound (UCB) strategy for HAUs to encourage exploration. Validated on a major livestream platform, our framework demonstrates significant improvements in retention (active hours) and satisfaction (quality watch time ratio) for LAUs as well as remarkable increases in interest diversity and category coverage for HAUs, proving the value of uncertainty-aware recommendation in industrial settings. 

---
# MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation 

**Authors**: Debashish Chakraborty, Dengjia Zhang, Jialiang Jin, Hanting Liu, Katherine Guerrerio, Hanxiang Qin, Tyler Skow, Alexander Martin, Reno Kriz, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2605.17640)  

**Abstract**: Retrieval-augmented generation from videos requires systems to retrieve relevant audiovisual evidence from large corpora and synthesize it into coherent, attributed text. Current approaches struggle at both ends: retrieval methods fail on complex, multi-faceted queries that cannot be captured by a single embedding, while generation methods lack the high-level reasoning needed to synthesize across multiple videos and face memory constraints over long, multi-video contexts. We present MARQUIS: a three-stage pipeline that addresses these limitations through (1) query expansion, fusion, and reranking, (2) calibrated structured evidence extraction, and (3) article generation from extracted evidence, optionally controlled by an RLM. On the MAGMaR2026 shared task, we improve retrieval performance from 0.195 to 0.759 (nDCG@10). For article generation, ITER-QA-BASE improves average human score from 3.09 to 3.83 over the CAG baseline, while MARQUIS-RLM achieves a human score of 3.30 and the strongest citation recall among non-QA systems. 

---
# Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation 

**Authors**: Yufei Guo, Jing Ma, Tianlu Zhang, Shijie Yang, Yanlong Zang, Weijie Ding, Pinghua Gong, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.17366)  

**Abstract**: Multimodal item embeddings are crucial for e-commerce item-to-item (I2I) retrieval, yet real-world product images often contain promotional overlays and background clutter that inject spurious visual cues and degrade retrieval robustness. This issue is particularly pronounced in MLRM-style pipelines, where a frozen vision encoder is connected to an LLM through a lightweight connector that must selectively aggregate visual tokens. We propose Text-Guided Q-Former (TGQ-Former), a text-guided visual representation learning framework that leverages structured metadata as semantic guidance for visual token extraction while preserving complementary visual evidence. Concretely, TGQ-Former employs a hybrid-query connector to disentangle metadata-anchored and exploratory visual streams, and introduces a lightweight reliability-aware dual-gated vector modulation module to adaptively calibrate their contributions under noisy inputs. Experiments on large-scale, real-world e-commerce datasets with full-pool retrieval show that TGQ-Former consistently outperforms strong connector baselines and end-to-end MLLMs. On average, it improves Hit Rate@100 (H@100) by 6.04%, demonstrating the effectiveness of text-guided visual encoding for robust multimodal retrieval. 

---
# Dual-Diffusional Generative Fashion Recommendation 

**Authors**: Mingzhe Yu, Lei Wu, Qianru Sun, Yunshan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.17357)  

**Abstract**: Personalized generative recommender systems have emerged as a promising solution for fashion recommendation. However, existing methods primarily rely on implicit visual embeddings from historical interactions, which often contain preference-irrelevant information and result in insufficient user behavior modeling. Moreover, these models typically generate only item images, providing limited interpretability. To address these limitations, we propose DualFashion, a Dual-Diffusional Generative Fashion Recommendation Architecture that jointly models image and text modalities for personalized and explainable recommendation. DualFashion adopts a dual-diffusion Transformer with image and text branches, where structured attribute-level captions and visual outfit information are jointly used as conditioning signals to model user behavior. The proposed architecture produces both fashion item images and textual descriptions, ensuring visual compatibility while providing explicit semantic interpretability. Furthermore, we introduce a text-augmented fine-tuning strategy that enhances generation diversity and enables effective cross-modal knowledge transfer without incurring heavy computational costs. Extensive experiments on iFashion and Polyvore-U across Personalized Fill-in-the-Blank and Generative Outfit Recommendation tasks demonstrate that DualFashion achieves strong performance in behavior modeling, interpretability, and efficiency compared to state-of-the-art methods. Our code and model checkpoints are available at this https URL. 

---
# RAGR: Review-Augmented Generative Recommendation 

**Authors**: Yingyi Zhang, Junyi Li, Yejing Wang, Wenlin Zhang, Xiaowei Qian, Sheng Zhang, Yue Feng, Yichao Wang, Yong Liu, Xiangyu Zhao, Xianneng Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.17267)  

**Abstract**: Sequential recommendation (SR) is traditionally formulated as next-item prediction over a chronological sequence of interacted items. Although recent generative recommendation (GR) methods introduce new machinery, such as semantic IDs, autoregressive decoding, and unified token spaces, they largely inherit the same item-only modeling assumption. We argue that this design constitutes a structural bottleneck, because user decision-making is not purely behavioral: while item interactions reveal what users choose, review feedback often explain why they choose it by exposing latent evaluative factors.
Motivated by this observation, we propose Review-Augmented Generative Recommendation (RAGR), a novel GR framework that incorporates review feedback directly into the generative user sequence rather than treating reviews as auxiliary side information. Specifically, RAGR introduces a Review-Augmented User Sequence Modeling mechanism that interleaves item semantic IDs and review semantic IDs in chronological order to construct a mixed behavioral-semantic sequence, enabling review signals to participate directly in autoregressive next-token generation. To preserve the recommendation objective, we further introduce an Item-Centric Task Generation Alignment strategy based on direct preference optimization (DPO), which encourages the model to favor item tokens over review tokens at prediction positions. Experiments on three real-world datasets show that RAGR yields consistent and significant gains over strong GR backbones across all metrics. Our code and data are available at \url{this https URL}. 

---
# Unlocking Biological Workflows for Robust Protein-Text Question Answering: A Dual-Dimensional RAG Framework 

**Authors**: Li Ding, Duanyu Feng, Chen Huang, Yangshuai Wang, Yang Li, Wenqiang Lei, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2605.17261)  

**Abstract**: Protein-Text Question Answering (QA) is crucial for interpreting biological sequences through natural language. The integration of Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) that efficiently leverages biological databases and facilitates reasoning offers a potent approach for it. However, constrained by the standard RAG pipeline, these models often rely on curated, static datasets instead of expert-proven biological workflows, lacking the fine-grained information processing and struggling to generalize to novel (OOD) proteins. To bridge this gap, we propose 2D-ProteinRAG, a novel framework that empowers LLMs to operate within the gold-standard biological research workflow (BLAST). To further extract high-quality information from noisy retrieval contexts, we introduce a dual-dimensional (2D) filtering strategy following the expert analytical paradigms. Horizontal Fine-grained Attribute Alignment utilizes a lightweight, intent-aware discriminative filter to prune irrelevant metadata and align database entries with specific user queries. Vertical Homology-based Semantic Denoising resolves functional contradictions and redundancy across multiple homologs via hierarchical clustering. Extensive evaluations on both In-Distribution and diverse biological OOD benchmarks demonstrate that 2D-ProteinRAG consistently achieves state-of-the-art performance, outperforming fine-tuned baselines and other RAG methods. Our results validate the framework's robustness and scalability, providing a practical solution for interpreting protein functions in real-world scientific scenarios. 

---
# Echoes in Filter Bubble: Diagnosing and Curing Popularity Bias in Generative Recommenders 

**Authors**: Jun Yin, Bangguo Zhu, Peng Huo, Ruochen Liu, Hao Chen, Senzhang Wang, Shirui Pan, Chengqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16825)  

**Abstract**: Recently, Generative Recommenders (GRs), characterized by a unified end-to-end framework, have exhibited astonishing potential in transforming the recommendation paradigm. Despite their effectiveness, we recognize that GRs are still susceptible to the long-standing issue of popularity bias that has pervaded the recommendation community. Although a few studies have attempted to extend traditional debiasing methods to GRs, their effectiveness is marginal, and the fundamental reason why GRs suffer from popularity bias remains under-explored. To bridge this gap, this study focuses on two core aspects in GRs: the optimization of generative framework and the item tokenization based on semantic index. Based on theoretical analyses, we identify that the severe popularity bias emerges from the confluence of a token-level optimization flaw and the undifferentiated property of item tokenization. Accordingly, this study develops a novel generative recommender system, called Ghost, by designing the asymmetric unlikelihood optimization and the skeleton-founded tokenization. Extensive empirical evaluations across three datasets, alongside multiple SOTA baselines, reveal that Ghost substantially alleviates popularity bias and promotes fairer recommendations, while incurring slight degradation to the overall recommendation utility. 

---
# UniER: A Unified Benchmark for Item-level and Path-level Exercise Recommendation 

**Authors**: Xinghe Cheng, Guiyong Zhuang, Yusheng Xie, Jiapu Wang, Yixin Liu, Quanlong Guan, Liangda Fang, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2605.16750)  

**Abstract**: Personalized exercise recommendation dynamically aligns pedagogical resources with individual knowledge mastery, which is crucial for satisfying students' dynamic learning needs in modern education. The field is currently driven by two dominant paradigms: Item-Level Exercise Recommendation (ILER) optimizes for immediate single-step state transitions, while Path-Level Exercise Recommendation (PLER) constructs coherent learning paths to maximize cumulative gains. Despite sharing the same ultimate objective, disparate evaluation setups have kept these two lines of research isolated, hindering unified benchmarking and fair comparison. To fill the gap, in this paper, we present a Unified Benchmark for Exercise Recommendation (UniER), a comprehensive evaluation framework that unifies ILER and PLER. Specifically, we introduce Weighted Cognitive Gain (WCG) as a unified metric to measure cross-paradigm algorithmic performance. Our benchmark encompasses 9 datasets spanning four generation methods, facilitating the comparison of 18 representative ILER/PLER methods. Through multi-dimensional analyses covering effectiveness, generalizability, robustness, and efficiency, our results reveal the systematic dominance of PLER and expose the pedagogical failure of ILER's fragmented recommendations under extreme sparsity and noise. Furthermore, we provide an open-source codebase of UniER to foster reproducible research and outline potential directions for future investigations. 

---
# RAPT: Retrieval-Augmented Post-hoc Thresholding for Multi-Label Classification 

**Authors**: Lasal Jayawardena, Nirmalie Wiratunga, Ikechukwu Nkisi-Orji, Darren Nicol  

**Link**: [PDF](https://arxiv.org/pdf/2605.16535)  

**Abstract**: Industrial multi-label document understanding pipelines score candidate labels and threshold or rank them to form a label set per document. This early selection step directly affects the accuracy of downstream information extraction from the document, as well as the associated verification effort. In practice, OCR noise, label imbalance, instance-dependent label cardinality, and asymmetric error costs make global score thresholds brittle and hard to maintain as document formats evolve. We present RAPT, a deployment-oriented retrieval-augmented score thresholding wrapper, applied post-hoc to improve label set selection without retraining the underlying classifier. RAPT is a model-agnostic wrapper: any predictor that provides document representations for similarity search and per label confidence scores can be used, including metric learning encoders and fine-tuned transformer classifiers. For each query document, given a classifier's score vector, RAPT retrieves similar document thresholding situations (cases) and adapts the query's label set selection threshold using their outcomes. The adaptation selects the final label set by locally aggregating neighbour solutions (e.g. average label count, cutoff calibration). Evaluation compared multi-label classifiers (metric learners and transformers) combined with RAPT against global and label-wise thresholding baselines, and against few-shot LLMs. Across an industrial dataset and six public benchmarks, RAPT consistently outperformed global and label-wise static thresholding baselines. In the industrial setting, RAPT achieved its best predictive performance with metric learners, reaching 0.87 Macro-F1, while fine-tuned transformer variants on average achieved 0.775 Macro-F1, outperforming fewshot LLM baselines (K = 5) by 2x and requiring at least 115x less inference time and 13.5x less GPU memory. 

---
# Policy-Grounded Dynamic Facet Suggestions for Job Search 

**Authors**: Dan Xu, Baofen Zheng, Qianqi Shen, Jianqiang Shen, Wenqiong Liu, Chunnan Yao, Ping Liu, Rajat Arora, Kevin Kao, Hsiang Lin, Wanjun Jiang, Yusuke Takebuchi, Jingwei Wu, Wenjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16479)  

**Abstract**: Job seekers often initiate search with short, underspecified queries. At LinkedIn, over 80% of job-related queries contain three or fewer keywords, making accurate user intent inference and relevant job retrieval particularly challenging. We present dynamic facet suggestion (DFS), an interactive query refinement mechanism that facilitates intent disambiguation by surfacing personalized semantic attributes conditioned on the joint user-query context in real time. We propose a policy-grounded, retrieval-augmented ranking framework for facet suggestion, comprising offline taxonomy curation, embedding-based retrieval of top-K candidates, and distilled small language model (SLM) based candidate scoring. The system is optimized for real-time serving via pointwise single-token scoring with batching and prefix caching. Offline evaluation demonstrates high precision for generated suggestions, and online A/B tests show significant improvements in suggestion engagement and job search outcomes. 

---
# LERA: LLM-Enhanced RAG for Ad Auction in Generative Chatbots 

**Authors**: Haoran Sun, Xinrui Song, Xinyu Zhang, Zhaohua Chen, Xu Chu, Zhilin Zhang, Chuan Yu, Jian Xu, Bo Zheng, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.16474)  

**Abstract**: The integration of advertising auction mechanisms into large language model (LLM)-based chatbots presents a significant opportunity for commercialization, yet poses unique challenges in balancing relevance, efficiency, and user experience. Recently, Feizi et al.~\citep{feizi2023online} and Hajiaghayi et al.~\citep{hajiaghayi2024ad} outlined a retrieve-then-generate paradigm that decouples retrieval and generation, offering lightweight ad insertion and payment determination. However, current retrieval relies solely on text embedding similarity, which may lead to commercial misinterpretation and issues such as repetitive insertions. In this paper, we propose LERA, a two-stage retrieve-then-generate auction framework tailored for LLM chatbots. In the first stage, embedding-based coarse filtering pre-selects a small set of candidate advertisers. In the second stage, the LLM itself is queried with a carefully designed prompt to produce logits over candidates, which serve as refined organic relevance scores. These scores are combined with bids, and a critical-value payment rule accounts for both the coarse-filtering and fine-ranking thresholds, ensuring truthfulness for utility-maximizing advertisers. The framework naturally extends to multiple ad insertions within dynamic dialogue flows and long responses. Experiments on a synthetic advertiser-query benchmark show that LERA substantially improves ad selection accuracy and insertion diversity while incurring only controllable latency overhead. 

---
# The Impact of AI Search on the Online Content Ecosystem: Evidence from Google and Reddit 

**Authors**: Peibo Zhang, Ruomeng Cui, Dennis J. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16428)  

**Abstract**: Search engines traditionally complement online content platforms by directing users seeking information to external websites. The emergence of generative AI search tools that summarize answers directly on the results page may disrupt this relationship by making visits to source platforms optional. We study this question using Google AI Overviews and Reddit, one of the largest online discussion platforms. Our identification exploits Google's content moderation policy: Safe-for-Work (SFW) Reddit communities are indexed by Google organic search and surfaced in Google AI Overviews, while Not-Safe-for-Work (NSFW) communities, though indexed by organic search, are prohibited from being referenced in AI Overview summaries. Using a difference-in-differences design, we find that AI Overviews increase engagement in SFW communities: daily comments rise by 12.0 percent and the number of commenting users by 12.3 percent relative to NSFW communities. The effects are concentrated in experience-based discussions (opinions, advice, and personal experiences) rather than fact-based information. However, the subsequent introduction of Google AI Mode, which allows users to interact conversationally with the AI summary, largely eliminates these gains in experience-based content. These results suggest that the effects of AI search depend critically on interface design and types of content. 

---
# LARGER: Lexically Anchored Repository Graph Exploration and Retrieval 

**Authors**: Yuntong Hu, Tongli Su, Liang Zhao, Bowen Zhu, Hasibul Haque  

**Link**: [PDF](https://arxiv.org/pdf/2605.16352)  

**Abstract**: Repository-level coding agents must first localize the files and symbols relevant to a task; failures at this stage can cascade across downstream objectives ranging from patch generation to test writing and codebase question answering. Existing agents navigate repositories primarily through lexical search, often missing structural relations such as imports, call chains, type hierarchies, and code-test links. Graph-based retrieval can recover such dependencies, but existing approaches often require separate graph tools or traversal stages that fragment the agent's interaction loop. We formalize repository context localization as Lexically Anchored Structural Localization, where success depends on turning lexical matches into high-precision structural entry points and exposing the most useful confidence-filtered local neighborhoods within the agent's existing search loop. We introduce LARGER (Lexically Anchored Repository Graph Exploration and Retrieval), a lexically anchored active-set retrieval framework that starts from lexical matches, aligns them to graph anchors, and performs confidence-filtered local expansion within the agent's existing search loop. LARGER integrates directly into existing CLI coding agents without requiring external graph databases or specialized graph interfaces. Across four benchmarks spanning localization, test generation, and codebase understanding, LARGER improves file-level Acc@5 on LocBench by +13.9 points with tuned hyperparameters and still gains +11.8 points with fixed hyperparameters over the strongest baseline, while delivering consistent gains on MuLocBench, SWE-Atlas Test Writing, and SWE-Atlas Codebase QA. 

---
# A Production-Ready RL Framework for Personalized Utility Tuning with Pareto Sweeping in Pinterest Recommender Systems 

**Authors**: Yichu Zhou, Mehdi Ben Ayed, Lin Yang, Jiacong He, Andreanne Lemay, Jiaye Wang, Jaewon Yang, Josie Zeng, Dhruvil Deven Badani, Yijie Dylan Wang, Jiajing Xu, Charles Rosenberg  

**Link**: [PDF](https://arxiv.org/pdf/2605.16344)  

**Abstract**: Large-scale recommenders encode multi-objective trade-offs by combining multiple predicted outcomes into a single utility score. Although this utility layer can be updated independently of the ranker, weight tuning remains largely manual, globally applied, slow to adapt to changing environments and business needs, and hard to govern as priorities shift. We propose PRL-PUTS, a Production-ready, ranker independent RL framework for Personalized Utility-weight Tuning with Pareto Sweeping. We cast utility tuning as a one-step, value-based RL problem: given request context, an agent selects a utility-weight vector that re-weights ranker predictions to maximize request-level engagement rewards. To visualize performance across the trade-off spectrum and allow decision makers to update the deployed operating policy instantly, we adopt an inference-time Pareto frontier sweeping via a scalarization parameter, producing a family of policies and an empirical Pareto frontier used as a governance artifact for operating policy selection. PRL-PUTS runs in parallel with ranking inference without adding serving latency. We validate PRL-PUTS with offline analysis using unbiased exploration logs and online experiments on Pinterest Homefeed where PRL-PUTS showed significant increases in engagement compared to baseline such as +0.13\% increase in successful session, a core metric for user engagement. 

---
# Vector RAG vs LLM-Compiled Wiki: A Preregistered Comparison on a Small Multi-Domain Research 

**Authors**: Theodore O. Cochran  

**Link**: [PDF](https://arxiv.org/pdf/2605.18490)  

**Abstract**: We preregistered a comparison of two ways to help an LLM answer questions over a small research corpus: a single-round Vector RAG system and an LLM-compiled markdown wiki. Both systems answered the same 13 questions over 24 papers using the same answer-generating model, and their answers were scored by blinded LLM judges.
The wiki scored much better at connecting findings across papers, but its advantage in answer organization was not strong after judge adjustment. RAG met the preregistered test for single-fact lookup questions. The clean query-side cost result went against the expected wiki advantage: under the tested setup, the wiki used far more query tokens than RAG, so it could not recover any upfront build cost through cheaper queries.
Two exploratory analyses changed how we interpret the result. First, claim-level citation checking favored the wiki: its cited pages more often supported the exact claims being made, even though RAG scored better on the overall groundedness rubric. Second, a decomposition-based RAG variant recovered most of the wiki's advantage on cross-paper synthesis at lower LLM-token cost, but it did not recover the wiki advantage in claim-by-claim citation support.
The main conclusion is that grounded research synthesis is not a single capability. Systems can differ in how well they organize evidence, how well their citations support each claim, and how much they cost to run. In this study, no architecture was best on all three. 

---
# SD-Search: On-Policy Hindsight Self-Distillation for Search-Augmented Reasoning 

**Authors**: Yufei Ma, Zihan Liang, Ben Chen, Zhipeng Qian, Huangyu Dai, Lingtao Mao, Xuxin Zhang, Chenyi Lei, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2605.18299)  

**Abstract**: Search-augmented reasoning agents interleave internal reasoning with calls to an external retriever, and their performance relies on the quality of each issued query. However, under outcome-reward reinforcement learning, every search decision in a rollout shares the same trajectory-level reward, leaving individual queries without step-specific credit. Recent process-supervision approaches address this gap by drawing step-level signals from outside the policy, relying either on a much larger teacher model, or on sub-question annotations produced by a stronger external system. In contrast, we propose SD-Search, which derives step-level supervision from the policy itself through on-policy hindsight self-distillation, requiring neither an external teacher nor additional annotations. In SD-Search, a single model plays two roles that differ only in conditioning: a student that sees only the context available at inference time, and a teacher that additionally conditions on a compact hindsight block summarizing the search queries and final outcomes of a group of rollouts sampled from the same question. Since the teacher knows how each rollout unfolded and which ones succeeded, its query distribution implicitly marks which decisions were worth making, and the student is trained to recover this behavior by minimizing the token-level Jensen--Shannon divergence to the teacher at search-query positions. This layers a dense, step-level signal on top of GRPO's coarse trajectory reward. Crucially, this signal is produced by the policy itself within the standard RL training loop, without external model inference, auxiliary annotation pipeline, or additional training stage. 

---
# From Volume to Value: Preference-Aligned Memory Construction for On-Device RAG 

**Authors**: Changmin Lee, Jaemin Kim, Taesik Gong  

**Link**: [PDF](https://arxiv.org/pdf/2605.18271)  

**Abstract**: With the rapid emergence of personal AI agents based on Large Language Models (LLMs), implementing them on-device has become essential for privacy and responsiveness. To handle the inherently personal and context-dependent nature of real-world requests, such agents must ground their generation in device-resident personal context. However, under tight memory budgets, the core bottleneck is what to store so that retrieval remains aligned with the user. We propose EPIC (Efficient Preference-aligned Index Construction), which focuses on user preferences as a compact and stable form of personal context and integrates them throughout the RAG pipeline. EPIC selectively retains preference-relevant information from raw data and aligns retrieval toward preference-aligned contexts. Across four benchmarks covering conversations, debates, explanations, and recommendations, EPIC reduces indexing memory by 2,404 times, improves preference-following accuracy by 20.17 percentage points, and achieves 33.33 times lower retrieval latency over the best-performing baseline. In our on-device experiment, EPIC maintains a memory footprint under 1 MB with 29.35 ms/query latency in streaming updates. 

---
# SomaliWeb v1: A Quality-Filtered Somali Web Corpus with a Matched Tokenizer and a Public Language-Identification Benchmark 

**Authors**: Khalid Yusuf Dahir  

**Link**: [PDF](https://arxiv.org/pdf/2605.18232)  

**Abstract**: Somali is a Cushitic language of the Horn of Africa with ~25 million speakers, yet no documented dedicated Somali pretraining corpus with a companion tokenizer and language-identification benchmark has been publicly released. Existing Somali text appears either inside multilingual distributions (HPLT v2, CC100, MADLAD-400, OSCAR, mC4) or in small, undocumented Somali-only uploads on Hugging Face. We introduce SomaliWeb v1, a quality-filtered Somali corpus of 819,322 documents (~303M tokens) built from three upstream sources (HPLT v2, CC100, Somali Wikipedia) through a six-stage reproducible pipeline. We release (i) the corpus, (ii) a matched BPE-16K tokenizer, and (iii) the first public side-by-side Somali benchmark of three production language identifiers. Our measurements reveal concrete quality defects in existing distributions: HPLT v2's "cleaned" Somali release retains 17.3% byte-exact duplicates, 56.1% of its documents contain fixable mojibake, and 10.7% of its byte-unique documents are near-duplicates at Jaccard tau=0.80. Our BPE-16K tokenizer emits 40.2% fewer tokens than GPT-4's cl100k_base on FLORES-200 Somali devtest as a tokenizer-level measurement; downstream language-model perplexity comparisons are deferred to a follow-up release. 

---
# An Empirical Study of Privacy Leakage Chains via Prompt Injection in Black-Box Chatbot Environments 

**Authors**: Hongjang Yang, Hyunsik Na, Daeseon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.18133)  

**Abstract**: LLM-based chatbot agents increasingly process user requests by combining natural-language reasoning with external tools such as web browsing. These capabilities improve usability, but they also create attack surfaces when untrusted external content is processed as part of a user' s task. This paper studies a privacy-leakage attack chain based on indirect prompt injection in black-box chatbot environments, where the attacker has no access to model weights, system prompts, or agent implementation details including how a trajectory is actually managed during its processing for a query. We first analyze how an attacker can hijack an agent' s intended task by crafting external content that appears benign to the victim while inducing the agent to execute an attacker-defined objective. We then evaluate a new prompt-injection technique, called exemplification, which uses a bridge in the external content to reframe the user prompt and the benign beginning of the retrieved page as few-shot examples before appending the attacker' s objective. We compare its attack success rate with a prior fake-completion technique. Finally, we demonstrate a proof-of-concept data-exfiltration chain using fictitious personal information in a controlled setting. Our results suggest that prompt injection, jailbreak-style instruction steering, and web-tool invocation can be combined into a feasible privacy-leakage path in deployed chatbot agents. 

---
# Agentic Chunking and Bayesian De-chunking of AI Generated Fuzzy Cognitive Maps: A Model of the Thucydides Trap 

**Authors**: Akash Kumar Panda, Olaoluwa Adigun, Bart Kosko  

**Link**: [PDF](https://arxiv.org/pdf/2605.17903)  

**Abstract**: We automatically generate feedback causal fuzzy cognitive maps (FCMs) from text by teaching large-language-model agents to break the text into overlapping chunks of text. Convex mixing of these chunk FCMs gives a representative cyclic FCM knowledge graph. The text chunks can have different levels of overlap. The chunk FCMs still mix to form a new FCM causal knowledge graph. The mixing technique scales because it uses light computation with sparse causal chunk matrices. The mixing structure allows an operator-level type of Bayesian inference that produces "de-chunked" or posterior-like FCMs from the mixed FCM. These de-chunked FCMs are useful in their own right and allow further iterations of Bayesian updating. We demonstrate these mixing techniques on the essay text of Allison's "Thucydides Trap" model of conflict between a dominant power such as the United States and a rising power such as China. The FCM dynamical systems predict outcomes as they equilibrate to fixed-point or limit-cycle attractors. Seven out of 8 FCM knowledge graphs predicted a type of war when we stimulated them by turning on and keeping on the concept node that stands for the rising power's ambition and entitlement. Gemini 3.1 LLMs served as the chunking AI agents. 

---
# Temporal Decay of Co-Citation Predictability: A 20-Year Statute Retrieval Benchmark from 396M Ukrainian Court Citations 

**Authors**: Volodymyr Ovcharov  

**Link**: [PDF](https://arxiv.org/pdf/2605.17639)  

**Abstract**: Co-citation structure is widely assumed to provide stable retrieval signal in legal information systems. We test this assumption longitudinally by constructing UA-StatuteRetrieval, a benchmark that measures co-citation predictability across 20 annual snapshots (2007-2026) of 396 million codex citations from 101 million Ukrainian court decisions. Using a leave-one-out protocol over the full bipartite citation graph, we find that Adamic-Adar MRR declines 33% on a fixed set of articles (from 0.43 to 0.29) and 47% under a train/test temporal split (from 0.51 to 0.27) confirming genuine temporal decay rather than compositional shift or evaluation artifact. The decay is non-uniform: criminal procedure maintains stable co-citation patterns (MRR ~0.40), while civil law degrades from 0.35 to 0.15, coinciding with the 2017 judicial reform. Hub articles (>100K citations) resist decay, but mid-frequency articles (1K-10K) -- the practical retrieval frontier lose half their predictability. A BM25 text baseline decays even faster (31%), and embedding drift analysis with E5-large reveals a 4.3% semantic shift in how articles are cited, providing a mechanistic explanation for the observed decay. The benchmark is released at this https URL. 

---
# Beyond Catalogue Counts: the Dataset Visibility Asymmetry in Low-Resource Multilingual NLP 

**Authors**: Zhiyin Tan, Changxu Duan  

**Link**: [PDF](https://arxiv.org/pdf/2605.17442)  

**Abstract**: Multilingual NLP often relies on dataset counts from centralized catalogues to characterize which languages are resource-rich or resource-poor. However, these catalogues record only one layer of dataset visibility: what has been registered or institutionally distributed. They do not necessarily reflect which datasets are created, cited, or reused in the research literature. To examine this gap, we combine a catalogue-based baseline with literature-backed evidence of dataset circulation. We introduce the Resource Density Index (RDI), defined as the number of catalogued datasets per one million speakers, and compute it for the 200 most widely spoken languages in Ethnologue. Among them, 118 languages (59%) have an average RDI of zero across the LRE Map and the Linguistic Data Consortium (LDC), and another 23 fall below 0.1, corresponding to at most one catalogued dataset per ten million speakers. We then apply an LLM-assisted citation-mining pipeline over the Semantic Scholar corpus to these 141 low-visibility languages. After manual validation and consolidation, we identify 609 unique datasets across 53 languages, of which 356 remain openly accessible through working public links. These results reveal a substantial visibility gap: many large-speaker languages appear data-poor in catalogue records yet show clear evidence of dataset activity in the research literature. Our findings suggest that multilingual data scarcity should be understood not only as a production problem, but also as a question of documentation, discoverability, and long-term accessibility. Code and data are publicly available at (this https URL). 

---
# IVF-TQ: Streaming-Robust Approximate Nearest Neighbor Search via a Codebook-Free Residual Layer 

**Authors**: Tarun Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2605.17415)  

**Abstract**: We propose IVF-TQ, an IVF index with a codebook-free residual layer: a fixed random rotation followed by precomputed Lloyd-Max scalar quantization depending only on (b, d). Only the IVF coarse partition is trained. Building on TurboQuant (Zandieh et al., 2025), the design substantially reduces a key failure mode of trained-codebook ANN indexes (PQ, OPQ, ScaNN): staleness under streaming this http URL (3 seeds): Per-batch PQ retraining does not recover the streaming gap at any tested bit budget (paired-t p > 0.28 everywhere). On streaming Deep-10M, IVF-TQ holds at 87.4% -> 86.6% (Delta = -0.80 +/- 0.10pp) while IVF-PQ degrades -3.23pp. A shuffled-i.i.d. control on SIFT-1M shows IVF-PQ losing -3.9pp without distribution shift. At higher PQ bit budgets (~1.5x IVF-TQ memory), absolute recall favors PQ as expected from rate-distortion (+6.1pp Deep-10M; +2.0pp SIFT-10M); the durable IVF-TQ benefit is operational (no codebook to retrain), robust across memory this http URL art: IVF around a codebook-free residual quantizer is architecturally not new -- IVF-RaBitQ ships in Milvus, cuVS, LanceDB, Weaviate; Shi et al. (2026) is concurrent GPU work. TurboQuant itself tests only flat-rotation this http URL: (i) A multi-seed streaming-operational story for codebook-free IVF: 10M-scale evidence across PQ memory budgets. (ii) A uniform-over-sphere IP-error bound for the TQ residual quantizer with one fixed rotation (proof sketch in v1; rigorous in v2). (iii) Adaptive IVF-TQ: a partition-only refresh recovering 67% -> 97.8% under worst-case rotation shift with re-ranking (90.3% without).Code, data: this https URL 

---
# NewsLens: A Multi-Agent Framework for Adversarial News Bias Navigation 

**Authors**: Joy Bose  

**Link**: [PDF](https://arxiv.org/pdf/2605.17364)  

**Abstract**: Media bias detection has predominantly been framed as a classification task: assign a political label to an article or outlet. We argue this framing is too shallow: it identifies that bias exists but not where, how, or crucially, what is structurally omitted. We present NewsLens, a five-agent adversarial pipeline for structured news bias navigation. A Fact Verifier, Progressive Framing Analyst, Conservative Framing Analyst, Propaganda Detector, and Neutral Summarizer collaborate to deconstruct articles into interpretable framing maps, exposing ideological omissions, rhetorical manipulation, and framing boundaries. The system is evaluated on 15 articles across four geopolitical event clusters (India-Pakistan Kashmir, Gaza, Climate Policy, Ukraine) using Qwen2.5-3B-Instruct (4-bit quantised, Google Colab T4), with cross-model validation using Mistral 7B on the Kashmir cluster. Center outlets show the highest mean Perspective Divergence Score (PDS: Qwen 0.907, Mistral 0.729 on Kashmir subset); conservative-framing outlets show the highest mean Manipulation Index (MI: 0.600 across both models). Cross-model comparison shows high consistency for high-propaganda content (Republic World delta-PDS=0.125, MI=0.8 both models) and greater variance for nuanced reporting. Mann-Whitney U tests find no statistically significant between-group differences at n=15, reported honestly as a sample-size limitation confirmed by post-hoc power analysis. A partial ablation removing the Propaganda Detector shows degraded omission precision in the Neutral Summarizer output. The architecture extends prior lexical-geometric bias work to agentic LLM reasoning, and is fully reproducible using open-weight models without API keys. 

---
# Approximate Distributed Coded Computing: Polynomial Codes and Randomized Sketching 

**Authors**: Neophytos Charalambides, Arya Mazumdar  

**Link**: [PDF](https://arxiv.org/pdf/2605.16744)  

**Abstract**: Coded computing is a distributed paradigm that uses coding theory to introduce \textit{redundancy} and overcome bottlenecks in large-scale systems. In the same vein, randomized numerical linear algebra employs probabilistic methods to \textit{compress} and accelerate linear algebraic operations, addressing challenges in high-dimensional data analysis. This article reviews the foundations of both fields and presents distributed schemes that combine techniques from both to speed up optimization and machine learning algorithms, in the presence of slow or non-responsive servers. Along the way, we touch on various related topics and mathematical concepts. 

---
# SotA Lens: A Network-Augmented Methodology and Tool for Exploratory State-of-the-Art Reviews 

**Authors**: Diogo Peralta Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2605.16333)  

**Abstract**: Researchers often begin new projects by conducting a broad State-of-the-Art review before they are ready to define the narrow protocol required by a systematic review. This is especially common in multidisciplinary areas where terminology is unstable, communities are weakly connected, and relevant work is dispersed across technical and application domains. This paper presents SotA Lens, a network-augmented methodology and lightweight software toolkit for exploratory State-of-the-Art reviews. The approach combines documented seed search, DOI-level metadata resolution, bounded citation expansion, directed graph construction, community detection, ranking of authors and subject terms, and human labelling of research communities. It is designed to complement, not replace, established review protocols such as PRISMA, PRISMA-ScR, systematic mapping studies, and bibliometric science mapping. The method is demonstrated through a proof-of-concept review of Dynamic Projection-Mapping and Spatial Augmented Reality. Starting from approximately 200 seed search results, the workflow produced a citation graph with 2,198 DOI-level vertices and 8,249 reference edges; a filtered largest component for 2010-2023 contained 986 vertices, 2,693 edges, and sixteen labelled communities. The contribution is both methodological and practical: SotA Lens helps researchers map broad fields, identify clusters and gaps, and produce auditable review artifacts before committing to a narrower systematic review protocol. This paper is not intended as a domain survey of Dynamic Projection-Mapping or Spatial Augmented Reality; rather, it introduces and demonstrates an original review-support methodology and software artifact using that domain as a proof-of-concept case study. 

---
# How algorithmic popularity bias hinders or promotes quality 

**Authors**: Azadeh Nematzadeh, Giovanni Luca Ciampaglia, Filippo Menczer, Alessandro Flammini  

**Link**: [PDF](https://arxiv.org/pdf/1707.00574)  

**Abstract**: Algorithms that favor popular items are used to help us select among many choices, from engaging articles on a social media news feed to songs and books that others have purchased, and from top-raked search engine results to highly-cited scientific papers. The goal of these algorithms is to identify high-quality items such as reliable news, beautiful movies, prestigious information sources, and important discoveries --- in short, high-quality content should rank at the top. Prior work has shown that choosing what is popular may amplify random fluctuations and ultimately lead to sub-optimal rankings. Nonetheless, it is often assumed that recommending what is popular will help high-quality content "bubble up" in practice. Here we identify the conditions in which popularity may be a viable proxy for quality content by studying a simple model of cultural market endowed with an intrinsic notion of quality. A parameter representing the cognitive cost of exploration controls the critical trade-off between quality and popularity. We find a regime of intermediate exploration cost where an optimal balance exists, such that choosing what is popular actually promotes high-quality items to the top. Outside of these limits, however, popularity bias is more likely to hinder quality. These findings clarify the effects of algorithmic popularity bias on quality outcomes, and may inform the design of more principled mechanisms for techno-social cultural markets. 

---
