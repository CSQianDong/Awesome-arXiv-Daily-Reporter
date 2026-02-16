# Fix Before Search: Benchmarking Agentic Query Visual Pre-processing in Multimodal Retrieval-augmented Generation 

**Authors**: Jiankun Zhang, Shenglai Zeng, Kai Guo, Xinnan Dai, Hui Liu, Jiliang Tang, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2602.13179)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MRAG) has emerged as a key paradigm for grounding MLLMs with external knowledge. While query pre-processing (e.g., rewriting) is standard in text-based RAG, existing MRAG pipelines predominantly treat visual inputs as static and immutable, implicitly assuming they are noise-free. However, real-world visual queries are often ``imperfect'' -- suffering from geometric distortions, quality degradation, or semantic ambiguity -- leading to catastrophic retrieval failures. To address this gap, we propose V-QPP-Bench, the first comprehensive benchmark dedicated to Visual Query Pre-processing (V-QPP). We formulate V-QPP as an agentic decision-making task where MLLMs must autonomously diagnose imperfections and deploy perceptual tools to refine queries. Our extensive evaluation across 46,700 imperfect queries and diverse MRAG paradigms reveals three critical insights: (1) Vulnerability -- visual imperfections severely degrade both retrieval recall and end-to-end MRAG performance; (2) Restoration Potential \& Bottleneck -- while oracle preprocessing recovers near-perfect performance, off-the-shelf MLLMs struggle with tool selection and parameter prediction without specialized training; and (3) Training Enhancement -- supervised fine-tuning enables compact models to achieve comparable or superior performance to larger proprietary models, demonstrating the benchmark's value for developing robust MRAG systems The code is available at this https URL 

---
# Asynchronous Verified Semantic Caching for Tiered LLM Architectures 

**Authors**: Asmit Kumar Singh, Haozhe Wang, Laxmi Naga Santosh Attaluri, Tak Chiam, Weihua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.13165)  

**Abstract**: Large language models (LLMs) now sit in the critical path of search, assistance, and agentic workflows, making semantic caching essential for reducing inference cost and latency. Production deployments typically use a tiered static-dynamic design: a static cache of curated, offline vetted responses mined from logs, backed by a dynamic cache populated online. In practice, both tiers are commonly governed by a single embedding similarity threshold, which induces a hard tradeoff: conservative thresholds miss safe reuse opportunities, while aggressive thresholds risk serving semantically incorrect responses. We introduce \textbf{Krites}, an asynchronous, LLM-judged caching policy that expands static coverage without changing serving decisions. On the critical path, Krites behaves exactly like a standard static threshold policy. When the nearest static neighbor of the prompt falls just below the static threshold, Krites asynchronously invokes an LLM judge to verify whether the static response is acceptable for the new prompt. Approved matches are promoted into the dynamic cache, allowing future repeats and paraphrases to reuse curated static answers and expanding static reach over time. In trace-driven simulations on conversational and search workloads, Krites increases the fraction of requests served with curated static answers (direct static hits plus verified promotions) by up to $\textbf{3.9}$ times for conversational traffic and search-style queries relative to tuned baselines, with unchanged critical path latency. 

---
# Awakening Dormant Users: Generative Recommendation with Counterfactual Functional Role Reasoning 

**Authors**: Huishi Luo, Shuokai Li, Hanchen Yang, Zhongbo Sun, Haojie Ding, Boheng Zhang, Zijia Cai, Renliang Qian, Fan Yang, Tingting Gao, Chenyi Lei, Wenwu Ou, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2602.13134)  

**Abstract**: Awakening dormant users, who remain engaged but exhibit low conversion, is a pivotal driver for incremental GMV growth in large-scale e-commerce platforms. However, existing approaches often yield suboptimal results since they typically rely on single-step estimation of an item's intrinsic value (e.g., immediate click probability). This mechanism overlooks the instrumental effect of items, where specific interactions act as triggers to shape latent intent and drive subsequent decisions along a conversion trajectory. To bridge this gap, we propose RoleGen, a novel framework that synergizes a Conversion Trajectory Reasoner with a Generative Behavioral Backbone. Specifically, the LLM-based Reasoner explicitly models the context-dependent Functional Role of items to reconstruct intent evolution. It further employs counterfactual inference to simulate diverse conversion paths, effectively mitigating interest collapse. These reasoned candidate items are integrated into the generative backbone, which is optimized via a collaborative "Reasoning-Execution-Feedback-Reflection" closed-loop strategy to ensure grounded execution. Extensive offline experiments and online A/B testing on the Kuaishou e-commerce platform demonstrate that RoleGen achieves a 6.2% gain in Recall@1 and a 7.3% increase in online order volume, confirming its effectiveness in activating the dormant user base. 

---
# RGAlign-Rec: Ranking-Guided Alignment for Latent Query Reasoning in Recommendation Systems 

**Authors**: Junhua Liu, Yang Jihao, Cheng Chang, Kunrong LI, Bin Fu, Kwan Hui Lim  

**Link**: [PDF](https://arxiv.org/pdf/2602.12968)  

**Abstract**: Proactive intent prediction is a critical capability in modern e-commerce chatbots, enabling "zero-query" recommendations by anticipating user needs from behavioral and contextual signals. However, existing industrial systems face two fundamental challenges: (1) the semantic gap between discrete user features and the semantic intents within the chatbot's Knowledge Base, and (2) the objective misalignment between general-purpose LLM outputs and task-specific ranking utilities. To address these issues, we propose RGAlign-Rec, a closed-loop alignment framework that integrates an LLM-based semantic reasoner with a Query-Enhanced (QE) ranking model. We also introduce Ranking-Guided Alignment (RGA), a multi-stage training paradigm that utilizes downstream ranking signals as feedback to refine the LLM's latent reasoning. Extensive experiments on a large-scale industrial dataset from Shopee demonstrate that RGAlign-Rec achieves a 0.12% gain in GAUC, leading to a significant 3.52% relative reduction in error rate, and a 0.56% improvement in Recall@3. Online A/B testing further validates the cumulative effectiveness of our framework: the Query-Enhanced model (QE-Rec) initially yields a 0.98% improvement in CTR, while the subsequent Ranking-Guided Alignment stage contributes an additional 0.13% gain. These results indicate that ranking-aware alignment effectively synchronizes semantic reasoning with ranking objectives, significantly enhancing both prediction accuracy and service quality in real-world proactive recommendation systems. 

---
# JARVIS: An Evidence-Grounded Retrieval System for Interpretable Deceptive Reviews Adjudication 

**Authors**: Nan Lu, Leyang Li, Yurong Hu, Rui Lin, Shaoyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.12941)  

**Abstract**: Deceptive reviews, refer to fabricated feedback designed to artificially manipulate the perceived quality of products. Within modern e-commerce ecosystems, these reviews remain a critical governance challenge. Despite advances in review-level and graph-based detection methods, two pivotal limitations remain: inadequate generalization and lack of interpretability. To address these challenges, we propose JARVIS, a framework providing Judgment via Augmented Retrieval and eVIdence graph Structures. Starting from the review to be evaluated, it retrieves semantically similar evidence via hybrid dense-sparse multimodal retrieval, expands relational signals through shared entities, and constructs a heterogeneous evidence graph. Large language model then performs evidence-grounded adjudication to produce interpretable risk assessments. Offline experiments demonstrate that JARVIS enhances performance on our constructed review dataset, achieving a precision increase from 0.953 to 0.988 and a recall boost from 0.830 to 0.901. In the production environment, our framework achieves a 27% increase in the recall volume and reduces manual inspection time by 75%. Furthermore, the adoption rate of the model-generated analysis reaches 96.4%. 

---
# WISE: A Multimodal Search Engine for Visual Scenes, Audio, Objects, Faces, Speech, and Metadata 

**Authors**: Prasanna Sridhar, Horace Lee, David M. S. Pinto, Andrew Zisserman, Abhishek Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2602.12819)  

**Abstract**: In this paper, we present WISE, an open-source audiovisual search engine which integrates a range of multimodal retrieval capabilities into a single, practical tool accessible to users without machine learning expertise. WISE supports natural-language and reverse-image queries at both the scene level (e.g. empty street) and object level (e.g. horse) across images and videos; face-based search for specific individuals; audio retrieval of acoustic events using text (e.g. wood creak) or an audio file; search over automatically transcribed speech; and filtering by user-provided metadata. Rich insights can be obtained by combining queries across modalities -- for example, retrieving German trains from a historical archive by applying the object query "train" and the metadata query "Germany", or searching for a face in a place. By employing vector search techniques, WISE can scale to support efficient retrieval over millions of images or thousands of hours of video. Its modular architecture facilitates the integration of new models. WISE can be deployed locally for private or sensitive collections, and has been applied to various real-world use cases. Our code is open-source and available at this https URL. 

---
# SQuTR: A Robustness Benchmark for Spoken Query to Text Retrieval under Acoustic Noise 

**Authors**: Yuejie Li, Ke Yang, Yueying Hua, Berlin Chen, Jianhao Nie, Yueping He, Caixin Kang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12783)  

**Abstract**: Spoken query retrieval is an important interaction mode in modern information retrieval. However, existing evaluation datasets are often limited to simple queries under constrained noise conditions, making them inadequate for assessing the robustness of spoken query retrieval systems under complex acoustic perturbations. To address this limitation, we present SQuTR, a robustness benchmark for spoken query retrieval that includes a large-scale dataset and a unified evaluation protocol. SQuTR aggregates 37,317 unique queries from six commonly used English and Chinese text retrieval datasets, spanning multiple domains and diverse query types. We synthesize speech using voice profiles from 200 real speakers and mix 17 categories of real-world environmental noise under controlled SNR levels, enabling reproducible robustness evaluation from quiet to highly noisy conditions. Under the unified protocol, we conduct large-scale evaluations on representative cascaded and end-to-end retrieval systems. Experimental results show that retrieval performance decreases as noise increases, with substantially different drops across systems. Even large-scale retrieval models struggle under extreme noise, indicating that robustness remains a critical bottleneck. Overall, SQuTR provides a reproducible testbed for benchmarking and diagnostic analysis, and facilitates future research on robustness in spoken query to text retrieval. 

---
# Training Dense Retrievers with Multiple Positive Passages 

**Authors**: Benben Wang, Minghao Tang, Hengran Zhang, Jiafeng Guo, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2602.12727)  

**Abstract**: Modern knowledge-intensive systems, such as retrieval-augmented generation (RAG), rely on effective retrievers to establish the performance ceiling for downstream modules. However, retriever training has been bottlenecked by sparse, single-positive annotations, which lead to false-negative noise and suboptimal supervision. While the advent of large language models (LLMs) makes it feasible to collect comprehensive multi-positive relevance labels at scale, the optimal strategy for incorporating these dense signals into training remains poorly understood. In this paper, we present a systematic study of multi-positive optimization objectives for retriever training. We unify representative objectives, including Joint Likelihood (JointLH), Summed Marginal Likelihood (SumMargLH), and Log-Sum-Exp Pairwise (LSEPair) loss, under a shared contrastive learning framework. Our theoretical analysis characterizes their distinct gradient behaviors, revealing how each allocates probability mass across positive document sets. Empirically, we conduct extensive evaluations on Natural Questions, MS MARCO, and the BEIR benchmark across two realistic regimes: homogeneous LLM-annotated data and heterogeneous mixtures of human and LLM labels. Our results show that LSEPair consistently achieves superior robustness and performance across settings, while JointLH and SumMargLH exhibit high sensitivity to the quality of positives. Furthermore, we find that the simple strategy of random sampling (Rand1LH) serves as a reliable baseline. By aligning theoretical insights with empirical findings, we provide practical design principles for leveraging dense, LLM-augmented supervision to enhance retriever effectiveness. 

---
# Self-EvolveRec: Self-Evolving Recommender Systems with LLM-based Directional Feedback 

**Authors**: Sein Kim, Sangwu Park, Hongseok Kang, Wonjoong Kim, Jimin Seo, Yeonjun In, Kanghoon Yoon, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2602.12612)  

**Abstract**: Traditional methods for automating recommender system design, such as Neural Architecture Search (NAS), are often constrained by a fixed search space defined by human priors, limiting innovation to pre-defined operators. While recent LLM-driven code evolution frameworks shift fixed search space target to open-ended program spaces, they primarily rely on scalar metrics (e.g., NDCG, Hit Ratio) that fail to provide qualitative insights into model failures or directional guidance for improvement. To address this, we propose Self-EvolveRec, a novel framework that establishes a directional feedback loop by integrating a User Simulator for qualitative critiques and a Model Diagnosis Tool for quantitative internal verification. Furthermore, we introduce a Diagnosis Tool - Model Co-Evolution strategy to ensure that evaluation criteria dynamically adapt as the recommendation architecture evolves. Extensive experiments demonstrate that Self-EvolveRec significantly outperforms state-of-the-art NAS and LLM-driven code evolution baselines in both recommendation performance and user satisfaction. Our code is available at this https URL. 

---
# RQ-GMM: Residual Quantized Gaussian Mixture Model for Multimodal Semantic Discretization in CTR Prediction 

**Authors**: Ziye Tong, Jiahao Liu, Weimin Zhang, Hongji Ruan, Derick Tang, Zhanpeng Zeng, Qinsong Zeng, Peng Zhang, Tun Lu, Ning Gu  

**Link**: [PDF](https://arxiv.org/pdf/2602.12593)  

**Abstract**: Multimodal content is crucial for click-through rate (CTR) prediction. However, directly incorporating continuous embeddings from pre-trained models into CTR models yields suboptimal results due to misaligned optimization objectives and convergence speed inconsistency during joint training. Discretizing embeddings into semantic IDs before feeding them into CTR models offers a more effective solution, yet existing methods suffer from limited codebook utilization, reconstruction accuracy, and semantic discriminability. We propose RQ-GMM (Residual Quantized Gaussian Mixture Model), which introduces probabilistic modeling to better capture the statistical structure of multimodal embedding spaces. Through Gaussian Mixture Models combined with residual quantization, RQ-GMM achieves superior codebook utilization and reconstruction accuracy. Experiments on public datasets and online A/B tests on a large-scale short-video platform serving hundreds of millions of users demonstrate substantial improvements: RQ-GMM yields a 1.502% gain in Advertiser Value over strong baselines. The method has been fully deployed, serving daily recommendations for hundreds of millions of users. 

---
# CAPTS: Channel-Aware, Preference-Aligned Trigger Selection for Multi-Channel Item-to-Item Retrieval 

**Authors**: Xiaoyou Zhou, Yuqi Liu, Zhao Liu, Xiao Lv, Bo Chen, Ruiming Tang, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.12564)  

**Abstract**: Large-scale industrial recommender systems commonly adopt multi-channel retrieval for candidate generation, combining direct user-to-item (U2I) retrieval with two-hop user-to-item-to-item (U2I2I) pipelines. In U2I2I, the system selects a small set of historical interactions as triggers to seed downstream item-to-item (I2I) retrieval across multiple channels. In production, triggers are often selected using rule-based policies or learned scorers and tuned in a channel-by-channel manner. However, these practices face two persistent challenges: biased value attribution that values triggers by on-trigger feedback rather than their downstream utility as retrieval seeds, and uncoordinated multi-channel routing where channels select triggers independently under a shared quota, increasing cross-channel overlap. To address these challenges, we propose Channel-Aware, Preference-Aligned Trigger Selection (CAPTS), a unified and flexible framework that treats multi-channel trigger selection as a learnable routing problem. CAPTS introduces a Value Attribution Module (VAM) that provides look-ahead supervision by crediting each trigger with the subsequent engagement generated by items retrieved from it on each I2I channel, and a Channel-Adaptive Trigger Routing (CATR) module that coordinates trigger-to-channel assignment to maximize the overall value of multi-channel retrieval. Extensive offline experiments and large-scale online A/B tests on Kwai, Kuaishou's international short-video platform, show that CAPTS consistently improves multi-channel recall offline and delivers a +0.351% lift in average time spent per device online. 

---
# Reasoning to Rank: An End-to-End Solution for Exploiting Large Language Models for Recommendation 

**Authors**: Kehan Zheng, Deyao Hong, Qian Li, Jun Zhang, Huan Yu, Jie Jiang, Hongning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12530)  

**Abstract**: Recommender systems are tasked to infer users' evolving preferences and rank items aligned with their intents, which calls for in-depth reasoning beyond pattern-based scoring. Recent efforts start to leverage large language models (LLMs) for recommendation, but how to effectively optimize the model for improved recommendation utility is still under explored. In this work, we propose Reasoning to Rank, an end-to-end training framework that internalizes recommendation utility optimization into the learning of step-by-step reasoning in LLMs. To avoid position bias in LLM reasoning and enable direct optimization of the reasoning process, our framework performs reasoning at the user-item level and employs reinforcement learning for end-to-end training of the LLM. Experiments on three Amazon datasets and a large-scale industrial dataset showed consistent gains over strong conventional and LLM-based solutions. Extensive in-depth analyses validate the necessity of the key components in the proposed framework and shed lights on the future developments of this line of work. 

---
# DiffuRank: Effective Document Reranking with Diffusion Language Models 

**Authors**: Qi Liu, Kun Ai, Jiaxin Mao, Yanzhao Zhang, Mingxin Li, Dingkun Long, Pengjun Xie, Fengbin Zhu, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2602.12528)  

**Abstract**: Recent advances in large language models (LLMs) have inspired new paradigms for document reranking. While this paradigm better exploits the reasoning and contextual understanding capabilities of LLMs, most existing LLM-based rerankers rely on autoregressive generation, which limits their efficiency and flexibility. In particular, token-by-token decoding incurs high latency, while the fixed left-to-right generation order causes early prediction errors to propagate and is difficult to revise. To address these limitations, we explore the use of diffusion language models (dLLMs) for document reranking and propose DiffuRank, a reranking framework built upon dLLMs. Unlike autoregressive models, dLLMs support more flexible decoding and generation processes that are not constrained to a left-to-right order, and enable parallel decoding, which may lead to improved efficiency and controllability. Specifically, we investigate three reranking strategies based on dLLMs: (1) a pointwise approach that uses dLLMs to estimate the relevance of each query-document pair; (2) a logit-based listwise approach that prompts dLLMs to jointly assess the relevance of multiple documents and derives ranking lists directly from model logits; and (3) a permutation-based listwise approach that adapts the canonical decoding process of dLLMs to the reranking tasks. For each approach, we design corresponding training methods to fully exploit the advantages of dLLMs. We evaluate both zero-shot and fine-tuned reranking performance on multiple benchmarks. Experimental results show that dLLMs achieve performance comparable to, and in some cases exceeding, that of autoregressive LLMs with similar model sizes. These findings demonstrate the promise of diffusion-based language models as a compelling alternative to autoregressive architectures for document reranking. 

---
# Visual RAG Toolkit: Scaling Multi-Vector Visual Retrieval with Training-Free Pooling and Multi-Stage Search 

**Authors**: Ara Yeroyan  

**Link**: [PDF](https://arxiv.org/pdf/2602.12510)  

**Abstract**: Multi-vector visual retrievers (e.g., ColPali-style late interaction models) deliver strong accuracy, but scale poorly because each page yields thousands of vectors, making indexing and search increasingly expensive. We present Visual RAG Toolkit, a practical system for scaling visual multi-vector retrieval with training-free, model-aware pooling and multi-stage retrieval. Motivated by Matryoshka Embeddings, our method performs static spatial pooling - including a lightweight sliding-window averaging variant - over patch embeddings to produce compact tile-level and global representations for fast candidate generation, followed by exact MaxSim reranking using full multi-vector embeddings.
Our design yields a quadratic reduction in vector-to-vector comparisons by reducing stored vectors per page from thousands to dozens, notably without requiring post-training, adapters, or distillation. Across experiments with interaction-style models such as ColPali and ColSmol-500M, we observe that over the limited ViDoRe v2 benchmark corpus 2-stage retrieval typically preserves NDCG and Recall @ 5/10 with minimal degradation, while substantially improving throughput (approximately 4x QPS); with sensitivity mainly at very large k. The toolkit additionally provides robust preprocessing - high resolution PDF to image conversion, optional margin/empty-region cropping and token hygiene (indexing only visual tokens) - and a reproducible evaluation pipeline, enabling rapid exploration of two-, three-, and cascaded retrieval variants. By emphasizing efficiency at common cutoffs (e.g., k <= 10), the toolkit lowers hardware barriers and makes state-of-the-art visual retrieval more accessible in practice. 

---
# Latent Customer Segmentation and Value-Based Recommendation Leveraging a Two-Stage Model with Missing Labels 

**Authors**: Keerthi Gopalakrishnan, Tianning Dong, Chia-Yen Ho, Yokila Arora, Topojoy Biswas, Jason Cho, Sushant Kumar, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2602.12485)  

**Abstract**: The success of businesses depends on their ability to convert consumers into loyal customers. A customer's value proposition is a primary determinant in this process, requiring a balance between affordability and long-term brand equity. Broad marketing campaigns can erode perceived brand value and reduce return on investment, while existing economic algorithms often misidentify highly engaged customers as ideal targets, leading to inefficient engagement and conversion outcomes.
This work introduces a two-stage multi-model architecture employing Self-Paced Loss to improve customer categorization. The first stage uses a multi-class neural network to distinguish customers influenced by campaigns, organically engaged customers, and low-engagement customers. The second stage applies a binary label correction model to identify true campaign-driven intent using a missing-label framework, refining customer segmentation during training.
By separating prompted engagement from organic behavior, the system enables more precise campaign targeting, reduces exposure costs, and improves conversion efficiency. A/B testing demonstrates over 100 basis points improvement in key success metrics, highlighting the effectiveness of intent-aware segmentation for value-driven marketing strategies. 

---
# An Industrial-Scale Sequential Recommender for LinkedIn Feed Ranking 

**Authors**: Lars Hertel, Gaurav Srivastava, Syed Ali Naqvi, Satyam Kumar, Yue Zhang, Borja Ocejo, Benjamin Zelditch, Adrian Englhardt, Hailing Cheng, Andy Hu, Antonio Alonso, Daming Li, Siddharth Dangi, Chen Zhu, Mingzhou Zhou, Wanning Li, Tao Huang, Fedor Borisyuk, Ganesh Parameswaran, Birjodh Singh Tiwana, Sriram Sankar, Qing Lan, Julie Choi, Souvik Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2602.12354)  

**Abstract**: LinkedIn Feed enables professionals worldwide to discover relevant content, build connections, and share knowledge at scale. We present Feed Sequential Recommender (Feed-SR), a transformer-based sequential ranking model for LinkedIn Feed that replaces a DCNv2-based ranker and meets strict production constraints. We detail the modeling choices, training techniques, and serving optimizations that enable deployment at LinkedIn scale. Feed-SR is currently the primary member experience on LinkedIn's Feed and shows significant improvements in member engagement (+2.10% time spent) in online A/B tests compared to the existing production model. We also describe our deployment experience with alternative sequential and LLM-based ranking architectures and why Feed-SR provided the best combination of online metrics and production efficiency. 

---
# AgenticShop: Benchmarking Agentic Product Curation for Personalized Web Shopping 

**Authors**: Sunghwan Kim, Ryang Heo, Yongsik Seo, Jinyoung Yeo, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.12315)  

**Abstract**: The proliferation of e-commerce has made web shopping platforms key gateways for customers navigating the vast digital marketplace. Yet this rapid expansion has led to a noisy and fragmented information environment, increasing cognitive burden as shoppers explore and purchase products online. With promising potential to alleviate this challenge, agentic systems have garnered growing attention for automating user-side tasks in web shopping. Despite significant advancements, existing benchmarks fail to comprehensively evaluate how well agentic systems can curate products in open-web settings. Specifically, they have limited coverage of shopping scenarios, focusing only on simplified single-platform lookups rather than exploratory search. Moreover, they overlook personalization in evaluation, leaving unclear whether agents can adapt to diverse user preferences in realistic shopping contexts. To address this gap, we present AgenticShop, the first benchmark for evaluating agentic systems on personalized product curation in open-web environment. Crucially, our approach features realistic shopping scenarios, diverse user profiles, and a verifiable, checklist-driven personalization evaluation framework. Through extensive experiments, we demonstrate that current agentic systems remain largely insufficient, emphasizing the need for user-side systems that effectively curate tailored products across the modern web. 

---
# Beyond Musical Descriptors: Extracting Preference-Bearing Intent in Music Queries 

**Authors**: Marion Baranes, Romain Hennequin, Elena V. Epure  

**Link**: [PDF](https://arxiv.org/pdf/2602.12301)  

**Abstract**: Although annotated music descriptor datasets for user queries are increasingly common, few consider the user's intent behind these descriptors, which is essential for effectively meeting their needs. We introduce MusicRecoIntent, a manually annotated corpus of 2,291 Reddit music requests, labeling musical descriptors across seven categories with positive, negative, or referential preference-bearing roles. We then investigate how reliably large language models (LLMs) can extract these music descriptors, finding that they do capture explicit descriptors but struggle with context-dependent ones. This work can further serve as a benchmark for fine-grained modeling of user intent and for gaining insights into improving LLM-based music understanding systems. 

---
# Nationwide Hourly Population Estimating at the Neighborhood Scale in the United States Using Stable-Attendance Anchor Calibration 

**Authors**: Huan Ning, Zhenlong Li, Manzhu Yu, Xiao Huang, Shiyan Zhang, Shan Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2602.12291)  

**Abstract**: Traditional population datasets are largely static and therefore unable to capture the strong temporal dynamics of human presence driven by daily mobility. Recent smartphone-based mobility data offer unprecedented spatiotemporal coverage, yet translating these opportunistic observations into accurate population estimates remains challenging due to incomplete sensing, spatially heterogeneous device penetration, and unstable observation processes. We propose a Stable-Attendance Anchor Calibration (SAAC) framework to reconstruct hourly population presence at the Census block group level across the United States. SAAC formulates population estimation as a balance-based population accounting problem, combining residential population with time-varying inbound and outbound mobility inferred from device-event observations. To address observation bias and identifiability limitations, the framework leverages locations with highly regular attendance as calibration anchors, using high schools in this study. These anchors enable estimation of observation scaling factors that correct for under-recorded mobility events. By integrating anchor-based calibration with an explicit sampling model, SAAC enables consistent conversion from observed device events to population presence at fine temporal resolution. The inferred population patterns are consistent with established empirical findings in prior mobility and urban population studies. SAAC provides a generalizable framework for transforming large-scale, biased digital trace data into interpretable dynamic population products, with implications for urban science, public health, and human mobility research. The hourly population estimates can be accessed at: this https URL. 

---
# Hi-SAM: A Hierarchical Structure-Aware Multi-modal Framework for Large-Scale Recommendation 

**Authors**: Pingjun Pan, Tingting Zhou, Peiyao Lu, Tingting Fei, Hongxiang Chen, Chuanjiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.11799)  

**Abstract**: Multi-modal recommendation has gained traction as items possess rich attributes like text and images. Semantic ID-based approaches effectively discretize this information into compact tokens. However, two challenges persist: (1) Suboptimal Tokenization: existing methods (e.g., RQ-VAE) lack disentanglement between shared cross-modal semantics and modality-specific details, causing redundancy or collapse; (2) Architecture-Data Mismatch: vanilla Transformers treat semantic IDs as flat streams, ignoring the hierarchy of user interactions, items, and tokens. Expanding items into multiple tokens amplifies length and noise, biasing attention toward local details over holistic semantics. We propose Hi-SAM, a Hierarchical Structure-Aware Multi-modal framework with two designs: (1) Disentangled Semantic Tokenizer (DST): unifies modalities via geometry-aware alignment and quantizes them via a coarse-to-fine strategy. Shared codebooks distill consensus while modality-specific ones recover nuances from residuals, enforced by mutual information minimization; (2) Hierarchical Memory-Anchor Transformer (HMAT): splits positional encoding into inter- and intra-item subspaces via Hierarchical RoPE to restore hierarchy. It inserts Anchor Tokens to condense items into compact memory, retaining details for the current item while accessing history only through compressed summaries. Experiments on real-world datasets show consistent improvements over SOTA baselines, especially in cold-start scenarios. Deployed on a large-scale social platform serving millions of users, Hi-SAM achieved a 6.55% gain in the core online metric. 

---
