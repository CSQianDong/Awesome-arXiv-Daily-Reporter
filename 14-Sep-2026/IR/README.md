# MIMA: Multi-Interest Recommendation via Multi-Positive Exclusive Assignment 

**Authors**: Xingyuan Mao, Alin Fan, Shichao Nie, Junfeng Zhang, Yan Xiao, Tao Luo, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2609.12842)  

**Abstract**: Multi-interest recommendation represents each user with multiple interest vectors for fine-grained candidate matching, yet it often suffers from interest collapse, where the learned interests converge to similar representations. We highlight the prevailing single-positive paradigm as one important factor behind this issue. Since each instance provides only one positive item, intents are optimized independently, potentially causing the same best-matching interest to be repeatedly updated toward different positives while leaving the others under-supervised. Moreover, existing methods rarely model how strongly a user activates each interest, leaving scores from different interest channels incomparable at inference. To address these problems, we propose MIMA, a Multi-Interest recommendation framework built on Multi-positive exclusive Assignment. MIMA groups items co-occurring within the same request into a positive set, generates complementary interests with a causal Transformer decoder, and exclusively assigns each positive to supervise a distinct interest via Hungarian matching, so that interest differentiation emerges from the training objective itself rather than auxiliary regularization. A lightweight routing module further estimates user-interest activation probabilities to calibrate scores across interest channels. Experiments on three public datasets and an industrial dataset show that MIMA consistently outperforms state-of-the-art baselines, and an online A/B test yields significant business gains. 

---
# A Historical Corpus Is Not a Historical System: Auditing Hindsight Leakage in Stateful Data Discovery 

**Authors**: Yixi Zhou, Fan Zhang, Sikun Wang, Yingfan Xu, Haipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12766)  

**Abstract**: Offline replay should estimate what a discovery system could retrieve at a historical point, yet freezing the corpus leaves interaction memory unconstrained. We formalize point-in-time (PIT) discovery through historical state $(D_t, \theta_t, M_{< i})$ and introduce a paired replay that changes only memory availability. The protocol constructs PIT and full-stream Future views from behavior-only traces and audits selected entries with a Temporal Violation Rate. Across three table-text domains, two stream regimes, two retrievers, and five seeds (216,000 rows), Future inflated Asset Recall@100 by 2.62-5.24 points; all 12 paired intervals excluded zero. With behavior-only trace memory, PIT underperformed the no-memory Stateless condition; Future masked 32.7-48.4% of that harm. For a simulated positive-feedback cache, PIT added 4.65-18.96 points over Stateless while Future added another 4.11-9.58 points. On five timestamped FreshStack topics, Future exceeded PIT by 2.72 points [1.75, 3.71]. Historical evaluation must version and validate memory with the corpus. 

---
# Learning the Lake: Reliable Experience for Adaptive Data Product Discovery 

**Authors**: Yixi Zhou, Fan Zhang, Sikun Wang, Yingfan Xu, Haipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12754)  

**Abstract**: Data-product discovery searches a full lake even when workloads revisit related products and regions. Repetition permits contracted search, but similarity cannot justify a route because one omitted asset invalidates a conjunctive product. We study when serving experience can safely reduce this work. Evolving Discovery Memory records source-labelled query--product--region evidence above a fixed regional index. SafeLake separates operational familiarity, which determines how much to search, from independently calibrated product evidence, which determines where to search. The fixed-probe comparison holds the adaptive budget constant between SafeLake and Familiarity-only. On TAT-QA, product steering raises Product Recall by 0.072; ConvFinQA shows no resolved map gain, while the HybridQA sensitivity favors Familiarity-only in Full R@100. Trace-only, missing, and false feedback expose boundaries on map steering, while scope-audit agreement cannot certify the source. Across clean confirmed-feedback streams under the frozen transductive protocol, the formal controller saves 49.5--82.7% of cumulative asset exposure. Experience determines when to contract; reliable evidence determines where to contract. 

---
# Personalized and Trust-Aware Health Recommendation Policies for a Construction Workplace 

**Authors**: Atefeh Mollabagher, Yogesh Gautam, Houtan Jebelli, Parinaz Naghizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2609.12679)  

**Abstract**: Construction workers face workplace risks such as fatigue, heat stress, and other physically demanding conditions that can negatively affect their health and safety. Although monitoring these risks is important, timely and personalized health interventions are also needed to help prevent negative impacts on workers' well-being and productivity. To this end, in this paper, we propose a model to capture the interactions between a trust-aware health recommender system and workers who differ in health and trust sensitivity. Specifically, in our proposed dynamic model, worker health evolves over time, worker trust is affected by both health and recommendation dynamics, and trust in turn affects compliance with future recommendations. Given this model, we characterize the recommender policy, including a health-based recommendation triggering threshold and the recommendation frequency. We do so using both model-based short-horizon control and model-free reinforcement learning. We then investigate how recommendation frequencies are adjusted for different workers to balance their health, productivity, and trust. Our findings provide insight into the design of personalized health recommendation policies in construction workplaces and beyond. 

---
# Preference-Drift-Aware Subsequence Learning and Hierarchical Context Fusion for Long-Sequence Generative Recommendation 

**Authors**: Fei Li, Qingyun Gao, Jianzhe Zhao, Guibing Guo, Beibei Kong, Lei Cheng, Chengxiang Zhuo, Zang Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.12556)  

**Abstract**: Long-sequence generative recommendation methods autoregressively model the user's interaction sequence to generate the next-item representation. Existing methods generally fall into two categories: efficient full-sequence modeling and target-aware context retrieval. Our experiments reveal that as the sequence length increases, the former incurs steadily growing computational cost while its accuracy gains quickly saturate and even degrade due to noise; the latter, though shortening the input sequence, is susceptible to noise that is semantically consistent yet preference-inconsistent, as well as to incomplete contexts. Both paradigms ignore the dynamic changes of user preferences and the cross-subsequence dependencies when handling historical information, thereby limiting accuracy and efficiency. To address these issues, we propose a preference-drift-aware subsequence learning and hierarchical context fusion for long-sequence generative recommendation. Specifically, we learn differentiable soft subsequence boundaries using multidimensional preference-drift information and aggregate items within each subsequence into preference-coherent representations via linear attention with soft assignment weights, thereby circumventing the expense of full-sequence attention. A cross-attention mechanism is then employed to capture dependencies between recent interactions and relevant subsequence contexts, mitigating noise in learning recent-item representations. Finally, a gated fusion mechanism adaptively combines the recent-item representation with the global subsequence context, allowing the resulting target representation to encode both recent and long-term preferences. Extensive experiments demonstrate that our method consistently outperforms existing baselines in both recommendation accuracy and computational efficiency. 

---
# ChronicleRec: Pre-training Temporally Anchored Tokens for Lifelong User Modeling 

**Authors**: Chengkai Huang, Yubin Sheng, Liang Guo, Haoxi Liu, Junwei Pan, Shangyu Zhang, Zhixiang Feng, Chao Zhou, Chengguo Yin, Lina Yao, Haijie Gu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12375)  

**Abstract**: Modeling ultra-long user behavior sequences is crucial for industrial recommendation and online advertising, yet directly feeding thousands of historical actions into ranking models is computationally prohibitive, while truncation discards long-range signals. Existing lifelong-interest methods retrieve target-relevant behaviors for each candidate, coupling long-sequence modeling with candidate scoring and repeated online cost. Recent target-independent compression methods enable cached user summaries, but often append query tokens at the sequence end and use bidirectional encoding, producing unordered and redundant summaries that overlook temporal structure. We propose ChronicleRec, a pre-train-and-transfer framework that compresses an ultra-long behavior sequence once into a chronologically ordered set of Chronicle Tokens. ChronicleRec applies a recency-aware multi-granularity merge, preserving recent behaviors while coarsening distant history. It then interleaves query tokens with the merged sequence and uses a causal encoder, so each query summarizes only the history before its temporal anchor. A multi-horizon design masks different recent-history windows across parallel branches to learn complementary long-range interests. The compressor is pre-trained with a mask-and-predict objective that reconstructs held-out recent behaviors from compressed older history, aligning historical signals with near-present intent. Since Chronicle Tokens are target-independent, they can be cached per user, decoupling ultra-long sequence modeling from online candidate scoring. Experiments on KuaiRand and Tencent AdLive show that ChronicleRec outperforms recent-window and single-pass compression baselines while approaching full-attention performance. Token analyses reveal temporally organized and complementary representations, and a seven-day online A/B test confirms significant production gains. 

---
# Recommendation Retrievers Need Verifiers: Universal Generative Reranking for Sequential Recommendations 

**Authors**: Benyu Zhang, Qiang Zhang, Rui Li, Qunshu Zhang, Devansh Tandon, Neeraj Bhatia  

**Link**: [PDF](https://arxiv.org/pdf/2609.12270)  

**Abstract**: First-stage recommenders in multi-stage systems produce a ranked candidate list from which a limited prefix is forwarded to downstream rankers. Because each forwarded item must be processed by more expensive ranking stages, this shortlist cannot be arbitrarily large. The first-stage objective is therefore high coverage of relevant items within the forwarded prefix, commonly measured by Recall@$k$. A relevant item may be available deeper in the retrieved list but absent from the shorter prefix that is actually consumed. This paper studies post-hoc verification for promoting such candidates into the consumed shortlist without retraining or replacing the retriever.
We introduce a lightweight generative verifier for retrieval models. Given a retriever state and a candidate item, the verifier scores the item through the likelihood of its identifier tokens. It is trained post hoc with next-token cross entropy, requires no sampled negatives or candidate pool during training, and scores only the retriever's top-$K$ candidates at inference. The interface is minimal: the retriever supplies a query state and candidate items, and the item representation can use any fixed tokenization.
Across Amazon product recommendation and YaMBDa music recommendation, the same verifier training recipe improves Recall@10 for SASRec, GRU4Rec, NextItNet, and MiniOneRec. Ablations show that the improvements are not explained solely by injecting item-content features into the retriever, supporting verification as a post-hoc output-side adaptation mechanism. 

---
# InitGen: Candidate Generation for Interaction Initiation in Intelligent Assistants 

**Authors**: Ruize Shi, Jinhua Chen, Hong Huang, Ziniu Chen, Ruike Zhang, Jianxun Shi, Yitao Chen, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.11953)  

**Abstract**: Interaction initiation refers to presenting multiple candidate queries when a user opens an intelligent assistant before expressing any intent for the current session. In production, candidate generation incorporates dynamic context and produces all candidates within a strict latency budget. Learning from user feedback is also difficult since the generator usually produces more candidates than are finally displayed. After downstream filtering and ranking, only a subset is exposed to users, so the observed feedback is partial and cannot be reliably assigned to individual queries. We present InitGen, a framework for candidate generation that is deployed in the interaction initiation pipeline of OPPO's Xiaobu Assistant. InitGen generates a set of candidate queries jointly and aligns the generated set with user feedback through weighted preference optimization. The sample weights are derived from user activity and downstream ranking scores. The activity weight reduces the dominance of highly active users during training, while the ranking score is used as a practical estimate of the reliability of the observed feedback. InitGen also uses a rolling window update strategy to incorporate recent interaction data into periodic model updates. In an online A/B test against a strong production baseline, InitGen improves the click-through rate from 0.95% to 1.61%, corresponding to a relative improvement of 69.1%, and increases query exposure by 17.9% under the same traffic allocation. InitGen generates the complete candidate set within 180 ms and has been fully deployed in OPPO's Xiaobu Assistant, which serves over 150 million monthly active users. 

---
# MemRetriever: Learning to Search, Reflect, and Retrieve from Long-Term Memory 

**Authors**: Ruiyang Jiang, Chunyu Li, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.11951)  

**Abstract**: Long-term memory enables personalized agents, but its value depends on retrieving the right evidence at the right time. Most memory systems use static top-k retrieval: they issue one query, return a fixed number of memories, and pass them directly to a downstream model. This approach can miss evidence distributed across sessions, introduce irrelevant content, and waste context, especially for multi-hop, temporal, and knowledge-update questions. We present MemRetriever, an agentic retrieval model that treats memory access as a multi-step search process. At each step, MemRetriever reasons over the current evidence and selects parallel search for broad exploration, serial search for targeted completion, or reflection and denoising for filtering and evidence assessment. It stops when the retained evidence is sufficient for downstream answering. We construct ReAct-style search-memory trajectories for supervised warm-start training and further optimize the model with Group Relative Policy Optimization. The reward design encourages evidence coverage, noise reduction, answer sufficiency, and efficient termination. Experiments on LOCOMO, LongMemEval, HotpotQA, MuSiQue, and 2WikiMultiHopQA show consistent improvements over static retrieval and supervised-only baselines. MemRetriever-4B-RL also outperforms DeepSeek-v4-Flash on the main LongMemEval retrieval metrics under the same pipeline and achieves the strongest results among the compared methods on MuSiQue. Because its decision logic is independent of the storage backend, MemRetriever can also operate over external knowledge bases and vector databases. These results show that an intermediate decision layer that plans, searches, filters evidence, and determines when to stop can improve both long-term memory retrieval and knowledge-intensive question answering. 

---
# Who Are We Recommending To? Recommender Systems in the Agentic Web 

**Authors**: Himan Abdollahpouri, Kyle Kretschman, Sai Ravindranath, Jackie Doremus, Mounia Lalmas  

**Link**: [PDF](https://arxiv.org/pdf/2609.11945)  

**Abstract**: For two decades, recommender systems have been designed under the assumption that a human directly consumes each recommendation: receiving, interpreting, and acting upon it. The emergence of AI agents powered by large language models challenges this assumption. In the emerging Agentic Web [ 28 ], autonomous agents increasingly act on behalf of users, e.g., browsing, comparing, negotiating, and executing transactions, raising a central question: who is the receiver of a recommendation? In this position paper, we argue that the recommendation paradigm is undergoing a bifurcation. In delegable contexts, such as routine purchases, travel, and constrained transactional tasks, the primary operational consumer of recommendations is shifting from the human to the agent, requiring new optimization objectives, interaction protocols, and evaluation criteria. In experiential contexts, such as entertainment, art, and other subjective or high-stakes choices, humans remain the final judge of relevance, though agents may assist through pre-filtering and curation. We introduce a delegation spectrum that characterizes recommendation contexts along factors such as preference specifiability, outcome verifiability, and decision stakes, and we outline a research agenda spanning agent preference modeling, dual-audience optimization, and the emerging agent attention economy. We further discuss the implications of this shift for the design and evaluation of recommender systems 

---
# PinDCO: Whole-Page Aware Dynamic Creative Optimization at Scale 

**Authors**: Yu Hao, Yuchun Li, Peimeng Sui, Meilin Liu, Tianyuan Cui, Hao Li, Zicong Zhou, Akanksha Baid  

**Link**: [PDF](https://arxiv.org/pdf/2609.11943)  

**Abstract**: Recent advances in generative AI have substantially accelerated the creation of high-quality ad creatives, dramatically expanding the number of candidate variants per campaign. This shift increases the need for scalable dynamic creative optimization (DCO) systems that can match creatives to the most relevant audiences under stringent latency and cost constraints. We present PinDCO, a production DCO system for ad creative retrieval and selection on Pinterest, a billion-scale visual discovery platform. PinDCO is built around a Creative Component Fusion Network (CCFN) that performs dynamic creative scoring by modeling each creative component (e.g., image, title, layout) with a dedicated tower, using component-specific hyperparameters to account for differing modeling complexity. The component representations are fused to predict a creative-level score conditioned on the ad-level prediction, and we improve training data quality via an exploration-exploitation strategy. To account for Pinterest's waterfall grid layout, where a creative's rendered size affects nearby content and session-level engagement, we introduce a Pixel-aware Adjustment Module(PAM) that adjusts scores based on creative size to encourage efficient screen real-estate utilization and better whole-page outcomes. To support the large volume of creative candidates, we further employ a lightweight pre-selection model for early pruning, and optimize serving efficiency through caching and dynamic batching. Extensive offline analyses and online A/B experiments demonstrate the effectiveness of PinDCO, yielding a +3.09% lift in ad Click-Through Rate(CTR) with positive whole-page metrics. With the strong performance, we launched PinDCO in the Pinterest Ads platform. 

---
# Position: Recommender Systems Should Move Beyond Platform-Centric Ranking toward Personal Agent-Mediated Recommendation 

**Authors**: Haohan Yuan, Peng He, Dan Zhang, Jianpeng Liang, Junning Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2609.11942)  

**Abstract**: Recommender systems are usually framed as ranking systems: platforms observe users, construct candidate sets, and select items on their behalf. This framing hides a deeper allocation of control, in which platforms also determine candidate access, evidence boundaries, explanations, and the path from user need to recommended output. We argue that the next bottleneck in recommendation is not only preference modeling, but control over evidence acquisition and disclosure. We argue for \textbf{Personal Agent-Mediated Recommendation} (PAMR), a paradigm in which a user-facing personal agent represents the user in discovering, filtering, aggregating, and governing recommendation evidence across distributed sources. The central shift is not simply from one ranking model to another, but from platform-side item ranking to user-side evidence mediation. As a position paper, we define PAMR as a new recommendation paradigm, establish its boundary criteria, identify its core mediation decisions, and propose a mediation-centered evaluation framework. A proof-of-concept study on hard Yelp restaurant recommendation tasks further shows that, under a shared LLM ranker, source selection and controlled disclosure provide the strongest observed utility--traceability--exposure--cost operating point. 

---
# Autonomous Research for Open-Ended Problems: A Case Study on Telecom Ticket Retrieval 

**Authors**: Junghyun Min, Huseyin Uzunalioglu, Mohamed Trabelsi  

**Link**: [PDF](https://arxiv.org/pdf/2609.13073)  

**Abstract**: Recent breakthroughs in LLM-based systems and their abilities in problem solving and coding have allowed progress in the AI for Science paradigm, potentially replacing human roles in machine learning (ML) research. However, while several frameworks of fully autonomous end-to-end ML research have been proposed, successful implementations of them are often limited to problems with narrow search spaces, like language modeling or biomedical ML benchmarks. In this paper, we explore how autonomous research can be adapted to solve open-ended, industry-grade ML problems, by considering a case study: telecom ticket retrieval, an open-ended task with degrees of freedom in representation, architecture, and training data generation. We discover that autonomous research for open-ended problems with commercial and open-source agents shows both promise and limitations: while autonomous research can excel in narrow hyperparameter optimization, it lacks human-like intuition and creativity and requires operational overhead. Even with minimal human supervision, autonomous research can reach $90\%$ of state-of-the-art performance (0.34 vs. 0.38 Recall@1) in a much shorter time period (10 weeks vs. 10 months of human work) at a modest cost (up to \$200 per Cursor campaign). Our empirical evidence recommends that human researchers and autonomous research frameworks work together for best results in ML research. 

---
# Cognition on Graph: Navigating Massive Knowledge Space via Cognitive Cycles and Bidirectional Graph-Text Synergy 

**Authors**: Gengxian Zhou, Jian Xu, Zichen Tang, Shiming Xiang, Haihong E, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12791)  

**Abstract**: Retrieval-Augmented Generation (RAG) has empowered Large Language Models (LLMs) to tackle knowledge-intensive tasks. However, navigating global, heterogeneous knowledge bases (large-scale knowledge graphs and text corpora) for complex reasoning remains a challenge. Existing methods typically employ reactive, graph-driven exploration strategies, which blindly follow graph topology without adapting to the question context or evolving exploration progress, and lack deep bidirectional synergy between graph and text. To address these limitations, we propose CoG (Cognition on Graph), a cognitive-inspired, training-free framework for adaptive knowledge exploration. Drawing inspiration from human problem-solving, CoG performs a continuous plan-explore-reflect cycle, where it proactively formulates investigation plans, performs dual-source retrieval, and dynamically reflects on progress to adjust strategies. Crucially, it establishes deep bidirectional synergy between structured graph and unstructured text, where entities extracted from text dynamically guide graph exploration to bridge knowledge gaps. Extensive experiments on seven multi-hop QA benchmarks demonstrate that CoG significantly outperforms state-of-the-art methods while achieving superior exploration efficiency. Our code and datasets are available at this https URL. 

---
# OneLA: Scaling Linear-Attention Decoding to Large Beams in Generative Recommendation 

**Authors**: Xiangrui Yang, Cheng Peng, Yunfeng Zhao, Liang Zeng, Ao Hu, Jiawei Yang, Shengzhe Wang, Jingshan Lv, Xiao Liang, Chen Yang, Jiaqiang Liu, Yiming Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12399)  

**Abstract**: Generative recommendation (GR) relies on large-beam decoding to generate hundreds of candidate items, creating a new scaling challenge for recurrent linear attention. Existing linear attention serving systems either materialize a full recurrent state for every beam or repeatedly replay shared history, incurring substantial memory and traffic overhead. To address this, we present OneLA, a linear-attention decoding framework that exploits the shared prompt and short divergent suffixes of GR workloads. Specifically, OneLA represents all beam states using a single shared prompt-derived state and compact, append-only records of their divergent transitions. Using this representation, OneLA computes only the state information required at each decoding step, without reconstructing a full recurrent state for every beam. Furthermore, OneLA uses a lightweight ancestry index to track the transition records that make up each beam's history, allowing beams to be updated without moving or copying existing records. A fused GPU kernel further reuses the shared state across beams. Our analysis shows that OneLA achieves 1.54-2.46x end-to-end decode speedups while substantially reducing recurrent-state memory use and data movement. 

---
# EAR: Entity-Aware Partitioning Approach for Retrieval-Augmented Generation Development 

**Authors**: Cenab Batu Bora, Oylum Alatlı, Sebnem Bora, Oguz Dikenelli  

**Link**: [PDF](https://arxiv.org/pdf/2609.12268)  

**Abstract**: Retrieval-augmented generation (RAG) can improve knowledge-intensive question answering, but the first design choice is easy to overlook: how should the source corpus be partitioned into retrievable units? Fixed-size chunks often return long passages whose relation to the question is only implicit. We introduce EAR, an Entity-Aware Partitioning approach for multiple-choice question answering (MCQA). EAR extracts normalized surface anchors from the question, answer options, and corpus; retrieves local windows around matching corpus anchors; and can attach a larger parent passage through an extractive summary. We evaluate EAR on a cleaned Massive Multitask Language Understanding (MMLU)-style subset of 153 questions selected by an automatic corpus-support heuristic and using decontaminated public textbook text. Across same-protocol top-k = 3 and top-k = 8 sweeps with Mistral, Gemma, and DeepSeek, EAR entity-window reduces retrieved words by 37.5-40.2% relative to chunks. Observed accuracy changes are +5.2, +1.3, and -3.9 points at top-k = 3, and +5.9, -3.3, and -4.6 points at top-k = 8; none of the entity-window differences is statistically significant. The scoped contribution is methodological: EAR provides a compact and inspectable retrieval unit, while its rule-based anchor extractor remains domain-specific and requires separate validation before transfer. 

---
# Extracting Dataset Mentions in Forced Displacement and FCV Documents: A Weakly Supervised Framework with LLM-Based Label Refinement 

**Authors**: Rafael Macalaba, Aivin V. Solatorio, Patrick Michael Brock, Olivier Dupriez  

**Link**: [PDF](https://arxiv.org/pdf/2609.12107)  

**Abstract**: Development and humanitarian organizations produce and support surveys, administrative registries, and other data resources to inform research, policy, and operations, yet systematically identifying where these datasets are referenced remains difficult. Such references are dispersed across research papers, project documents, humanitarian reports, and other unstructured text, limiting both the ability to trace data use and to identify potential gaps in data availability or dissemination. We present a weakly supervised framework for adapting dataset extraction to forced displacement and Fragile, Conflict, and Violence (FCV) documents without first constructing a large manually labeled training corpus. A lightweight model trained on general research literature generates candidate dataset mentions from unlabeled domain documents, which a frontier large language model (LLM) reviews in context, validating or rejecting candidates and correcting their extraction boundaries. The resulting annotations are supplemented with targeted synthetic and contrastive examples and used to fine-tune the lightweight model for large-scale extraction. We evaluate the resulting model on an independent gold-standard benchmark of 1,706 text passages spanning research, humanitarian, and operational documents. Across the full benchmark, the model achieves 74.1\% precision and 70.5\% recall at the mention level; among passages containing dataset references, precision reaches 89.5\%. At the passage level, the model achieves 88.2\% accuracy and 88.6\% specificity in distinguishing passages with dataset references from those without them. These results demonstrate a practical approach for constructing domain-specific supervision when labeled data are limited, and provide a technical foundation for larger-scale analysis of data use and potential gaps in the displacement data landscape. 

---
# Cortex: Content Analysis Support Software, a Resource for Qualitative Research 

**Authors**: Ana Julia da Silva Soares, Rafael Coimbra Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2609.11970)  

**Abstract**: Qualitative research is widely used in the human and social sciences, characterized by a deep understanding of phenomena through the interpretation of meanings and contexts. Among qualitative data analysis methods, content analysis stands out as a consolidated technique, which allows for the systematic description and interpretation of textual contents. However, as data volume increases, the time required for organization, reading, and categorization becomes a significant challenge, potentially delaying research development. Therefore, this work aimed to develop a web application to support content analysis, based on Bardin's methodology, targeted at academic researchers. The methodology adopted a mixed approach, combining bibliographic research on content analysis with semi-structured interviews with four experienced researchers, aiming to identify real needs and requirements. Based on these inputs, the Cortex software was developed to assist the researcher in the pre-analysis and material exploration stages, generating suggestions for indices, indicators, and categories with full traceability to original documents. The results demonstrate that Cortex is capable of guiding the researcher through the entire methodological workflow, from corpus configuration to results exportation, acting as a methodological collaborator without replacing the researcher's interpretative autonomy. 

---
