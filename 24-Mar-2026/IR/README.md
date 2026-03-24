# One Model, Two Markets: Bid-Aware Generative Recommendation 

**Authors**: Yanchen Jiang, Zhe Feng, Christopher P. Mah, Aranyak Mehta, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22231)  

**Abstract**: Generative Recommender Systems using semantic ids, such as TIGER (Rajput et al., 2023), have emerged as a widely adopted competitive paradigm in sequential recommendation. However, existing architectures are designed solely for semantic retrieval and do not address concerns such as monetization via ad revenue and incorporation of bids for commercial retrieval. We propose GEM-Rec, a unified framework that integrates commercial relevance and monetization objectives directly into the generative sequence. We introduce control tokens to decouple the decision of whether to show an ad from which item to show. This allows the model to learn valid placement patterns directly from interaction logs, which inherently reflect past successful ad placements. Complementing this, we devise a Bid-Aware Decoding mechanism that handles real-time pricing, injecting bids directly into the inference process to steer the generation toward high-value items. We prove that this approach guarantees allocation monotonicity, ensuring that higher bids weakly increase an ad's likelihood of being shown without requiring model retraining. Experiments demonstrate that GEM-Rec allows platforms to dynamically optimize for semantic relevance and platform revenue. 

---
# PreferRec: Learning and Transferring Pareto Preferences for Multi-objective Re-ranking 

**Authors**: Wei Zhou, Wuyang Li, Junkai Ji, Xueliang Li, Wenjing Hong, Zexuan Zhu, Xing Tang, Xiuqiang He  

**Link**: [PDF](https://arxiv.org/pdf/2603.22073)  

**Abstract**: Multi-objective re-ranking has become a critical component of modern multi-stage recommender systems, as it tasked to balance multiple conflicting objectives such as accuracy, diversity, and fairness. Existing multi-objective re-ranking methods typically optimize aggregate objectives at the item level using static or handcrafted preference weights. This design overlooks that users inherently exhibit Pareto-optimal preferences at the intent level, reflecting personalized trade-offs among objectives rather than fixed weight combinations. Moreover, most approaches treat re-ranking task for each user as an isolated problem, and repeatedly learn the preferences from scratch. Such a paradigm not only incurs high computational cost, but also ignores the fact that users often share similar preference trade-off structures across objectives. Inspired by the existence of homogeneous multi-objective optimization spaces where Pareto-optimal patterns are transferable, we propose PreferRec, a novel framework that explicitly models and transfers Pareto preferences across users. Specifically, PreferRec is built upon three tightly coupled components: Preference-Aware Pareto Learning aims to capture user intrinsic trade-offs among multiple conflicting objectives at the intent level. By learning Pareto preference representations from re-ranking populations, this component explicitly models how users prioritize different objectives under diverse contexts. Knowledge-Guided Transfer facilitates efficient cross-user knowledge transfer by distilling shared optimization patterns across homogeneous optimization spaces. The transferred knowledge is then used to guide solution selection and personalized re-ranking, biasing the optimization process toward high-quality regions of the Pareto front while preserving user-specific preference characteristics. 

---
# On the Challenges and Opportunities of Learned Sparse Retrieval for Code 

**Authors**: Simon Lupart, Maxime Louis, Thibault Formal, Hervé Déjean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2603.22008)  

**Abstract**: Retrieval over large codebases is a key component of modern LLM-based software engineering systems. Existing approaches predominantly rely on dense embedding models, while learned sparse retrieval (LSR) remains largely unexplored for code. However, applying sparse retrieval to code is challenging due to subword fragmentation, semantic gaps between natural-language queries and code, diversity of programming languages and sub-tasks, and the length of code documents, which can harm sparsity and latency. We introduce SPLADE-Code, the first large-scale family of learned sparse retrieval models specialized for code retrieval (600M-8B parameters). Despite a lightweight one-stage training pipeline, SPLADE-Code achieves state-of-the-art performance among retrievers under 1B parameters (75.4 on MTEB Code) and competitive results at larger scales (79.0 with 8B). We show that learned expansion tokens are critical to bridge lexical and semantic matching, and provide a latency analysis showing that LSR enables sub-millisecond retrieval on a 1M-passage collection with little effectiveness loss. 

---
# ADaFuSE: Adaptive Diffusion-generated Image and Text Fusion for Interactive Text-to-Image Retrieval 

**Authors**: Zhuocheng Zhang, Xingwu Zhang, Kangheng Liang, Guanxuan Li, Richard Mccreadie, Zijun Long  

**Link**: [PDF](https://arxiv.org/pdf/2603.21886)  

**Abstract**: Recent advances in interactive text-to-image retrieval (I-TIR) use diffusion models to bridge the modality gap between the textual information need and the images to be searched, resulting in increased effectiveness. However, existing frameworks fuse multi-modal views of user feedback by simple embedding addition. In this work, we show that this static and undifferentiated fusion indiscriminately incorporates generative noise produced by the diffusion model, leading to performance degradation for up to 55.62% samples. We further propose ADaFuSE (Adaptive Diffusion-Text Fusion with Semantic-aware Experts), a lightweight fusion model designed to align and calibrate multi-modal views for diffusion-augmented I-TIR, which can be plugged into existing frameworks without modifying the backbone encoder. Specifically, we introduce a dual-branch fusion mechanism that employs an adaptive gating branch to dynamically balance modality reliability, alongside a semantic-aware mixture-of-experts branch to capture fine-grained cross-modal nuances. Via thorough evaluation over four standard I-TIR benchmarks, ADaFuSE achieves state-of-the-art performance, surpassing DAR by up to 3.49% in Hits@10 with only a 5.29% parameter increase, while exhibiting stronger robustness to noisy and longer interactive queries. These results show that generative augmentation coupled with principled fusion provides a simple, generalizable alternative to fine-tuning for interactive retrieval. 

---
# GoogleTrendArchive: A Year-Long Archive of Real-Time Web Search Trends Worldwide 

**Authors**: Aleksandra Urman, Anikó Hannák, Joachim Baumann  

**Link**: [PDF](https://arxiv.org/pdf/2603.21871)  

**Abstract**: GoogleTrendArchive is a comprehensive archive of Google Trending Now data spanning over one year (from November 28, 2024 to January 3, 2026) across 125 countries and 1,358 locations. Unlike Google Trends, which requires specifying search terms in advance, Trending Now captures search queries experiencing real-time surges, offering a way to inductively discover trending patterns across regions for studying collective attention dynamics. However, Google does not provide historical access to this data beyond seven days. Our dataset addresses this gap by presenting an archive of Trending Now data. The dataset contains over 7.6 million trend episodes. Each record includes the trend identifier, search volume bucket, precise timestamps, duration, geographic location, and related query clusters. This dataset, among other, enables systematic studies of information diffusion patterns, cross-cultural attention dynamics, crisis responses, and the temporal evolution of collective information-seeking at a global scale. The comprehensive geographic coverage facilitates fine-grained cross-country or cross-regional comparative analyses. 

---
# AgenticRec: End-to-End Tool-Integrated Policy Optimization for Ranking-Oriented Recommender Agents 

**Authors**: Tianyi Li, Zixuan Wang, Guidong Lei, Xiaodong Li, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.21613)  

**Abstract**: Recommender agents built on Large Language Models offer a promising paradigm for recommendation. However, existing recommender agents typically suffer from a disconnect between intermediate reasoning and final ranking feedback, and are unable to capture fine-grained preferences. To address this, we present AgenticRec, a ranking-oriented agentic recommendation framework that optimizes the entire decision-making trajectory (including intermediate reasoning, tool invocation, and final ranking list generation) under sparse implicit feedback. Our approach makes three key contributions. First, we design a suite of recommendation-specific tools integrated into a ReAct loop to support evidence-grounded reasoning. Second, we propose theoretically unbiased List-Wise Group Relative Policy Optimization (list-wise GRPO) to maximize ranking utility, ensuring accurate credit assignment for complex tool-use trajectories. Third, we introduce Progressive Preference Refinement (PPR) to resolve fine-grained preference ambiguities. By mining hard negatives from ranking violations and applying bidirectional preference alignment, PPR minimizes the convex upper bound of pairwise ranking errors. Experiments on benchmarks confirm that AgenticRec significantly outperforms baselines, validating the necessity of unifying reasoning, tool use, and ranking optimization. 

---
# Overview of TREC 2025 Biomedical Generative Retrieval (BioGen) Track 

**Authors**: Deepak Gupta, Dina Demner-Fushman, William Hersh, Steven Bedrick, Kirk Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2603.21582)  

**Abstract**: Recent advances in large language models (LLMs) have made significant progress across multiple biomedical tasks, including biomedical question answering, lay-language summarization of the biomedical literature, and clinical note summarization. These models have demonstrated strong capabilities in processing and synthesizing complex biomedical information and in generating fluent, human-like responses. Despite these advancements, hallucinations or confabulations remain key challenges when using LLMs in biomedical and other high-stakes domains. Inaccuracies may be particularly harmful in high-risk situations, such as medical question answering, making clinical decisions, or appraising biomedical research. Studies on the evaluation of the LLMs' abilities to ground generated statements in verifiable sources have shown that models perform significantly 

---
# Toward a Theory of Hierarchical Memory for Language Agents 

**Authors**: Yashar Talebirad, Ali Parsaee, Csongor Y. Szepesvari, Amirhossein Nadiri, Osmar Zaiane  

**Link**: [PDF](https://arxiv.org/pdf/2603.21564)  

**Abstract**: Many recent long-context and agentic systems address context-length limitations by adding hierarchical memory: they extract atomic units from raw data, build multi-level representatives by grouping and compression, and traverse this structure to retrieve content under a token budget. Despite recurring implementations, there is no shared formalism for comparing design choices. We propose a unifying theory in terms of three operators. Extraction ($\alpha$) maps raw data to atomic information units; coarsening ($C = (\pi, \rho)$) partitions units and assigns a representative to each group; and traversal ($\tau$) selects which units to include in context given a query and budget. We identify a self-sufficiency spectrum for the representative function $\rho$ and show how it constrains viable retrieval strategies (a coarsening-traversal coupling). Finally, we instantiate the decomposition on eleven existing systems spanning document hierarchies, conversational memory, and agent execution traces, showcasing its generality. 

---
# TagLLM: A Fine-Grained Tag Generation Approach for Note Recommendation 

**Authors**: Zhijian Chen, Likai Wang, Lei Chen, Yaguang Dou, Jialiang Shi, Tian Qi, Dongdong Hao, Mengying Lu, Cheng Ye, Chao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.21481)  

**Abstract**: Large Language Models (LLMs) have shown promising potential in E-commerce community recommendation. While LLMs and Multimodal LLMs (MLLMs) are widely used to encode notes into implicit embeddings, leveraging their generative capabilities to represent notes with interpretable tags remains unexplored. In the field of tag generation, traditional close-ended methods heavily rely on the design of tag pools, while existing open-ended methods applied directly to note recommendations face two limitations: (1) MLLMs lack guidance during generation, resulting in redundant tags that fail to capture user interests; (2) The generated tags are often coarse and lack fine-grained representation of notes, interfering with downstream recommendations. To address these limitations, we propose TagLLM, a fine-grained tag generation method for note recommendation. TagLLM captures user interests across note categories through a User Interest Handbook and constructs fine-grained tag data using multimodal CoT Extraction. A Tag Knowledge Distillation method is developed to equip small models with competitive generation capabilities, enhancing inference efficiency. In online A/B test, TagLLM increases average view duration per user by 0.31%, average interactions per user by 0.96%, and page view click-through rate in cold-start scenario by 32.37%, demonstrating its effectiveness. 

---
# When Documents Disagree: Measuring Institutional Variation in Transplant Guidance with Retrieval-Augmented Language Models 

**Authors**: Yubo Li, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2603.21460)  

**Abstract**: Patient education materials for solid-organ transplantation vary substantially across U.S. centers, yet no systematic method exists to quantify this heterogeneity at scale. We introduce a framework that grounds the same patient questions in different centers' handbooks using retrieval-augmented language models and compares the resulting answers using a five-label consistency taxonomy. Applied to 102 handbooks from 23 centers and 1,115 benchmark questions, the framework quantifies heterogeneity across four dimensions: question, topic, organ, and center. We find that 20.8% of non-absent pairwise comparisons exhibit clinically meaningful divergence, concentrated in condition monitoring and lifestyle topics. Coverage gaps are even more prominent: 96.2% of question-handbook pairs miss relevant content, with reproductive health at 95.1% absence. Center-level divergence profiles are stable and interpretable, where heterogeneity reflects systematic institutional differences, likely due to patient diversity. These findings expose an information gap in transplant patient education materials, with document-grounded medical question answering highlighting opportunities for content improvement. 

---
# COINBench: Moving Beyond Individual Perspectives to Collective Intent Understanding 

**Authors**: Xiaozhe Li, Tianyi Lyu, Siyi Yang, Yizhao Yang, Yuxi Gong, Jinxuan Huang, Ligao Zhang, Zhuoyi Huang, Qingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.21329)  

**Abstract**: Understanding human intent is a high-level cognitive challenge for Large Language Models (LLMs), requiring sophisticated reasoning over noisy, conflicting, and non-linear discourse. While LLMs excel at following individual instructions, their ability to distill Collective Intent - the process of extracting consensus, resolving contradictions, and inferring latent trends from multi-source public discussions - remains largely unexplored. To bridge this gap, we introduce COIN-BENCH, a dynamic, real-world, live-updating benchmark specifically designed to evaluate LLMs on collective intent understanding within the consumer domain. Unlike traditional benchmarks that focus on transactional outcomes, COIN-BENCH operationalizes intent as a hierarchical cognitive structure, ranging from explicit scenarios to deep causal reasoning. We implement a robust evaluation pipeline that combines a rule-based method with an LLM-as-the-Judge approach. This framework incorporates COIN-TREE for hierarchical cognitive structuring and retrieval-augmented verification (COIN-RAG) to ensure expert-level precision in analyzing raw, collective human discussions. An extensive evaluation of 20 state-of-the-art LLMs across four dimensions - depth, breadth, informativeness, and correctness - reveals that while current models can handle surface-level aggregation, they still struggle with the analytical depth required for complex intent synthesis. COIN-BENCH establishes a new standard for advancing LLMs from passive instruction followers to expert-level analytical agents capable of deciphering the collective voice of the real world. See our project page on COIN-BENCH. 

---
# LSA: A Long-Short-term Aspect Interest Transformer for Aspect-Based Recommendation 

**Authors**: Le Liu, Junrui Liu, Yunhan Gao, Ziheng Wang, Tong Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.21243)  

**Abstract**: Aspect-based recommendation methods extract aspect terms from reviews, such as price, to model fine-grained user preferences on items, making them a critical approach in personalized recommender systems. Existing methods utilize graphs to represent the relationships among users, items, and aspect terms, modeling user preferences based on graph neural networks. However, they overlook the dynamic nature of user interests - users may temporarily focus on aspects they previously paid little attention to - making it difficult to assign accurate weights to aspect terms for each user-item interaction. In this paper, we propose a long-short-term aspect interest Transformer (LSA) for aspect-based recommendation, which effectively captures the dynamic nature of user preferences by integrating both long-term and short-term aspect interests. Specifically, the short-term interests model the temporal changes in the importance of recently interacted aspect terms, while the long-term interests consider global behavioral patterns, including aspects that users have not interacted with recently. Finally, LSA combines long- and short-term interests to evaluate the importance of aspects within the union of user and item aspect neighbors, therefore accurately assigns aspect weights for each user-item interaction. Experiments conducted on four real-world datasets demonstrate that LSA improves MSE by 2.55% on average over the best baseline. 

---
# MI-DPG: Decomposable Parameter Generation Network Based on Mutual Information for Multi-Scenario Recommendation 

**Authors**: Wenzhuo Cheng, Ke Ding, Xin Dong, Yong He, Liang Zhang, Linjian Mo  

**Link**: [PDF](https://arxiv.org/pdf/2603.21209)  

**Abstract**: Conversion rate (CVR) prediction models play a vital role in recommendation and advertising systems. Recent research on multi-scenario recommendation shows that learning a unified model to serve multiple scenarios is effective for improving overall performance. However, it remains challenging to improve model prediction performance across scenarios at low model parameter cost, and current solutions are hard to robustly model multi-scenario diversity. In this paper, we propose MI-DPG for the multi-scenario CVR prediction, which learns scenario-conditioned dynamic model parameters for each scenario in a more efficient and effective manner. Specifically, we introduce an auxiliary network to generate scenario-conditioned dynamic weighting matrices, which are obtained by combining decomposed scenario-specific and scenario-shared low-rank matrices with parameter efficiency. For each scene, weighting the backbone model parameters by the weighting matrix helps to specialize the model parameters for different scenarios. It can not only modulate the complete parameter space of the backbone model but also improve the model effectiveness. Furthermore, we design a mutual information regularization to enhance the diversity of model parameters across different scenarios by maximizing the mutual information between the scenario-aware input and the scene-conditioned dynamic weighting matrix. Experiments from three real-world datasets show that MI-DPG significantly outperforms previous multi-scenario recommendation models. 

---
# Ontology-Compliant Knowledge Graphs 

**Authors**: Zhangcheng Qiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.21188)  

**Abstract**: Ontologies can act as a schema for constructing knowledge graphs (KGs), offering explainability, interoperability, and reusability. We explore \emph{ontology-compliant} KGs, aiming to build both internal and external ontology compliance. We discuss key tasks in ontology compliance and introduce our novel term-matching algorithms. We also propose a \emph{pattern-based compliance} approach and novel compliance metrics. The building sector is a case study to test the validity of ontology-compliant KGs. We recommend using ontology-compliant KGs to pursue automatic matching, alignment, and harmonisation of heterogeneous KGs. 

---
# Ontology-driven personalized information retrieval for XML documents 

**Authors**: Ounnaci Iddir, Ahmed-ouamer Rachid, Tai Dinh  

**Link**: [PDF](https://arxiv.org/pdf/2603.21139)  

**Abstract**: This paper addresses the challenge of improving information retrieval from semi-structured eXtensible Markup Language (XML) documents. Traditional information retrieval systems (IRS) often overlook user-specific needs and return identical results for the same query, despite differences in users' knowledge, preferences, and objectives. We integrate external semantic resources, namely a domain ontology and user profiles, into the retrieval process. Documents, queries, and user profiles are represented as vectors of weighted concepts. The ontology applies a concept-weighting mechanism that emphasizes highly specific concepts, as lower-level nodes in the hierarchy provide more precise and targeted information. Relevance is assessed using semantic similarity measures that capture conceptual relationships beyond keyword matching, enabling personalized and fine-grained matching among user profiles, queries, and documents. Experimental results show that combining ontologies with user profiles improves retrieval effectiveness, achieving higher precision and recall than keyword-based approaches. Overall, the proposed framework enhances the relevance and adaptability of XML search results, supporting more user-centered retrieval. 

---
# Query, Decompose, Compress: Structured Query Expansion for Efficient Multi-Hop Retrieval 

**Authors**: JungMin Yun, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.21024)  

**Abstract**: Large Language Models (LLMs) have been increasingly employed for query expansion. However, their generative nature often undermines performance on complex multi-hop retrieval tasks by introducing irrelevant or noisy information. To address this challenge, we propose DeCoR (Decompose and Compress for Retrieval), a framework grounded in structured information refinement. Rather than generating additional content, DeCoR strategically restructures the query's underlying reasoning process and distills supporting evidence from retrieved documents. It consists of two core components tailored to the challenges of multi-hop retrieval: (1) Query Decomposition, which decomposes a complex query into explicit reasoning steps, and (2) Query-aware Document Compression, which synthesizes dispersed evidence from candidate documents into a concise summary relevant to the query. This structured design ensures that the final query representation remains both robust and comprehensive. Experimental results demonstrate that, despite utilizing a relatively small LLM, DeCoR outperforms strong baselines that rely on larger models. This finding underscores that, in complex retrieval scenarios, sophisticatedly leveraging the reasoning and summarization capabilities of LLMs offers a more efficient and effective solution than relying solely on their generative capability. 

---
# DSL-R1: From SQL to DSL for Training Retrieval Agents across Structured and Unstructured Data with Reinforcement Learning 

**Authors**: Yunhai Hu, Junwei Zhou, Yumo Cao, Yitao Long, Yiwei Xu, Qiyi Jiang, Weiyao Wang, Xiaoyu Cao, Zhen Sun, Yiran Zou, Nan Du  

**Link**: [PDF](https://arxiv.org/pdf/2603.21018)  

**Abstract**: Effective retrieval in complex domains requires bridging the gap between structured metadata and unstructured content. Existing systems typically isolate these capabilities, relying on either symbolic filtering or vector similarity, failing to capture their interplay. In this work, we propose DSL-R1, a unified framework that synergizes logical reasoning with semantic matching via a novel Domain-Specific Language (DSL). By embedding vector primitives within SQL-style operators, our approach leverages the complementary strengths of symbolic precision and semantic coverage. We further introduce a reinforcement learning mechanism where rule-based execution feedback and retrieval quality rewards jointly optimize the DSL generation, balancing structural correctness and semantic alignment. Evaluations on a large-scale industrial email benchmark demonstrate that DSL-R1 achieves a +12.3% improvement in Hit@1/3, consistently outperforming decoupled baselines and establishing a robust paradigm for hybrid retrieval. 

---
# Consensus-Driven Group Recommendation on Sparse Explicit Feedback: A Collaborative Filtering and Choquet-Borda Aggregation Framework 

**Authors**: Anh Nguyen Van, Huy Ngo Hoang, Khoi Ngo Nguyen, Ngoc Pham Thi, Khanh Ngo Mai Bao, Quyen Nguyen Van  

**Link**: [PDF](https://arxiv.org/pdf/2603.21012)  

**Abstract**: Group Recommender Systems (GRS) play an essential role in supporting collective decision-making among users with diverse and potentially conflicting preferences. However, achieving stable intra-group consensus becomes particularly challenging when only sparse userID-itemID-rating data are available and no demographic, contextual, or group-level information exists. This paper proposes a consensus-driven hybrid group recommendation framework that integrates neighborhood-based collaborative filtering with fuzzy aggregation to support agreement, fairness, and robustness under sparsity. A composite similarity measure, CBS (Combined Similarity), is derived from two enhanced similarity metrics introduced in prior work: a geometry-based measure that captures rating-pattern structure, and an uncertainty-aware measure that models belief, evidence, and disagreement in sparse co-rating contexts. This combination provides more stable estimation of missing ratings and supports consensus-oriented neighborhood construction. Candidate items are generated by merging per-user top-N predictions and further enriched using the Borda Count mechanism to mitigate skewed rating distributions and reinforce group-level agreement. Final group ratings are computed using the Choquet integral, which flexibly captures heterogeneous user influence while preserving fairness and supporting consensus formation. Experimental results on real-world datasets with different rating distributions show that the proposed method improves group-level consensus, satisfaction, and fairness, while maintaining a balanced level of novelty. Although the model does not rely on social information, its evaluation using trust-aware novelty measures indicates stable behavior in socially structured environments. 

---
# ECI: Effective Contrastive Information to Evaluate Hard-Negatives 

**Authors**: Aarush Sinha, Rahul Seetharaman, Aman Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2603.20990)  

**Abstract**: Hard negatives play a critical role in training and fine-tuning dense retrieval models, as they are semantically similar to positive documents yet non-relevant, and correctly distinguishing them is essential for improving retrieval accuracy. However, identifying effective hard negatives typically requires extensive ablation studies involving repeated fine-tuning with different negative sampling strategies and hyperparameters, resulting in substantial computational cost. In this paper, we introduce ECI: Effective Contrastive Information , a theoretically grounded metric grounded in Information Theory and Information Retrieval principles that enables practitioners to assess the quality of hard negatives prior to model fine-tuning. ECI evaluates negatives by optimizing the trade-off between Information Capacity the logarithmic bound on mutual information determined by set size and Discriminative Efficiency, a harmonic balance of Signal Magnitude (Hardness) and Safety (Max-Margin). Unlike heuristic approaches, ECI strictly penalizes unsafe, false-positive negatives prevalent in generative methods. We evaluate ECI across hard-negative sets mined or generated using BM25, cross-encoders, and large language models. Our results demonstrate that ECI accurately predicts downstream retrieval performance, identifying that hybrid strategies (BM25+Cross-Encoder) offer the optimal balance of volume and reliability, significantly reducing the need for costly end-to-end ablation studies. 

---
# RubricRAG: Towards Interpretable and Reliable LLM Evaluation via Domain Knowledge Retrieval for Rubric Generation 

**Authors**: Kaustubh D. Dhole, Eugene Agichtein  

**Link**: [PDF](https://arxiv.org/pdf/2603.20882)  

**Abstract**: Large language models (LLMs) are increasingly evaluated and sometimes trained using automated graders such as LLM-as-judges that output scalar scores or preferences. While convenient, these approaches are often opaque: a single score rarely explains why an answer is good or bad, which requirements were missed, or how a system should be improved. This lack of interpretability limits their usefulness for model development, dataset curation, and high-stakes deployment. Query-specific rubric-based evaluation offers a more transparent alternative by decomposing quality into explicit, checkable criteria. However, manually designing high-quality, query-specific rubrics is labor-intensive and cognitively demanding and not feasible for deployment. While previous approaches have focused on generating intermediate rubrics for automated downstream evaluation, it is unclear if these rubrics are both interpretable and effective for human users. In this work, we investigate whether LLMs can generate useful, instance-specific rubrics as compared to human-authored rubrics, while also improving effectiveness for identifying good responses. Through our systematic study on two rubric benchmarks, and on multiple few-shot and post-training strategies, we find that off-the-shelf LLMs produce rubrics that are poorly aligned with human-authored ones. We introduce a simple strategy, RubricRAG, which retrieves domain knowledge via rubrics at inference time from related queries. We demonstrate that RubricRAG can generate more interpretable rubrics both for similarity to human-authored rubrics, and for improved downstream evaluation effectiveness. Our results highlight both the challenges and a promising approach of scalable, interpretable evaluation through automated rubric generation. 

---
# Algorithmic Audit of Personalisation Drift in Polarising Topics on TikTok 

**Authors**: Branislav Pecher, Adrian Bindas, Jan Jakubcik, Matus Tuna, Matus Tibensky, Simon Liska, Peter Sakalik, Andrej Suty, Matej Mosnar, Filip Hossner, Ivan Srba  

**Link**: [PDF](https://arxiv.org/pdf/2603.20723)  

**Abstract**: Social media platforms have become an integral part of everyday life, serving as a primary source of news and information for many users. These platforms increasingly rely on personalised recommendation systems that shape what users see and engage with. While these systems are optimised for engagement, concerns have emerged that they may also drive users toward more polarised perspectives, particularly in contested domains such as politics, climate change, vaccines, and conspiracy theories. In this paper, we present an algorithmic audit of personalisation drift on TikTok in these polarising topics. Using controlled accounts designed to simulate users with interests aligned with or opposed to different polarising topics, we systematically measure the extent to which TikTok steers content exposure toward specific topics and polarities over time. Specifically, we investigated: 1) a preference-aligned drift (showing a strong personalisation towards user interests), 2) a polarisation-topic drift (showing a strong neutralising effect for misinformation-themed topics, and a high preference and reinforcement of interest of US politic topic); and 3) a polarisation-stance drift (showing a preference of oppose stance towards US politics topic and a general reinforcement of users' stance by recommending items aligned with their stance towards polarising topics). Overall, our findings provide evidence that recommendation trajectories differ markedly across topics, with some pathways amplifying polarised viewpoints more strongly than others and offer insights for platform governance, transparency and user awareness. 

---
# NDT: Non-Differential Transformer and Its Application to Sentiment Analysis 

**Authors**: Soudeep Ghoshal, Himanshu Buckchash, Sarita Paudel, Rubén Ruiz-Torrubiano  

**Link**: [PDF](https://arxiv.org/pdf/2603.20704)  

**Abstract**: From customer feedback to social media, understanding human sentiment in text is central to how machines can interact meaningfully with people. However, despite notable progress, accurately capturing sentiment remains a challenging task, which continues to motivate further research in this area. To this end, we introduce Non-Differential Transformer (NDT). It is inspired by (but in contrast to) the state-of-the-art Differential Transformer (DT) model. While standard Transformers can struggle with irrelevant context, the sota DT model uses attention map subtraction, potentially for noise cancellation. We explore an alternative motivation, hypothesizing that benefits may arise from enabling different attention components to specialize on distinct concepts within the text, similar to multiplexing information channels or mixture models, rather than primarily canceling noise via subtraction. Guided by this concept-multiplexing (ConPlex) view, the specific architecture presented in this paper employs a purely additive strategy. It uses only positive weights, learned during training, to ensure constructive combination of these specialized attention perspectives. This design choice explores positive only integration, though our broader framework also shows promise with less constrained linear combinations involving both positive and negative weights. Our model computes attention via this positively weighted sum of multiple distinct attention maps. This allows the model to constructively integrate diverse signals and potentially capture more complex contextual relationships. Competitive performance is achieved by the proposed model for Sentiment Analysis while tested on multiple datasets. We conclude by presenting our results, challenges and future research agenda in this important area of research. 

---
# ReBOL: Retrieval via Bayesian Optimization with Batched LLM Relevance Observations and Query Reformulation 

**Authors**: Anton Korikov, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2603.20513)  

**Abstract**: LLM-reranking is limited by the top-k documents retrieved by vector similarity, which neither enables contextual query-document token interactions nor captures multimodal relevance distributions. While LLM query reformulation attempts to improve recall by generating improved or additional queries, it is still followed by vector similarity retrieval. We thus propose to address these top-k retrieval stage failures by introducing ReBOL, which 1) uses LLM query reformulations to initialize a multimodal Bayesian Optimization (BO) posterior over document relevance, and 2) iteratively acquires document batches for LLM query-document relevance scoring followed by posterior updates to optimize relevance. After exploring query reformulation and document batch diversification techniques, we evaluate ReBOL against LLM reranker baselines on five BEIR datasets and using two LLMs (Gemini-2.5-Flash-Lite, GPT-5.2). ReBOL consistently achieves higher recall and competitive rankings, for example compared to the best LLM reranker on the Robust04 dataset with 46.5% vs. 35.0% recall@100 and 63.6% vs. 61.2% NDCG@10. We also show that ReBOL can achieve comparable latency to LLM rerankers. 

---
# WebNavigator: Global Web Navigation via Interaction Graph Retrieval 

**Authors**: Xuanwang Zhang, Yuteng Han, Jinnan Qi, Mulong Xie, Zhen Wu, Xinyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2603.20366)  

**Abstract**: Despite significant advances in autonomous web navigation, current methods remain far from human-level performance in complex web environments. We argue that this limitation stems from Topological Blindness, where agents are forced to explore via trial-and-error without access to the global topological structure of the environment. To overcome this limitation, we introduce WebNavigator, which reframes web navigation from probabilistic exploration into deterministic retrieval and pathfinding. WebNavigator constructs Interaction Graphs via zero-token cost heuristic exploration offline and implements a Retrieve-Reason-Teleport workflow for global navigation online. WebNavigator achieves state-of-the-art performance on WebArena and OnlineMind2Web. On WebArena multi-site tasks, WebNavigator achieves a 72.9\% success rate, more than doubling the performance of enterprise-level agents. This work reveals that Topological Blindness, rather than model reasoning capabilities alone, is an underestimated bottleneck in autonomous web navigation. 

---
# Low-pass Personalized Subgraph Federated Recommendation 

**Authors**: Wooseok Sim, Hogun Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.20338)  

**Abstract**: Federated Recommender Systems (FRS) preserve privacy by training decentralized models on client-specific user-item subgraphs without sharing raw data. However, FRS faces a unique challenge: subgraph structural imbalance, where drastic variations in subgraph scale (user/item counts) and connectivity (item degree) misalign client representations, making it challenging to train a robust model that respects each client's unique structural characteristics. To address this, we propose a Low-pass Personalized Subgraph Federated recommender system (LPSFed). LPSFed leverages graph Fourier transforms and low-pass spectral filtering to extract low-frequency structural signals that remain stable across subgraphs of varying size and degree, allowing robust personalized parameter updates guided by similarity to a neutral structural anchor. Additionally, we leverage a localized popularity bias-aware margin that captures item-degree imbalance within each subgraph and incorporates it into a personalized bias correction term to mitigate recommendation bias. Supported by theoretical analysis and validated on five real-world datasets, LPSFed achieves superior recommendation accuracy and enhances model robustness. 

---
# GEM: A Native Graph-based Index for Multi-Vector Retrieval 

**Authors**: Yao Tian, Zhoujin Tian, Xi Zhao, Ruiyuan Zhang, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.20336)  

**Abstract**: In multi-vector retrieval, both queries and data are represented as sets of high-dimensional vectors, enabling finer-grained semantic matching and improving retrieval quality over single-vector approaches. However, its practical adoption is held back by the lack of effective indexing algorithms. Existing work, attempting to reuse standard single-vector indexes, often fails to preserve multi-vector semantics or remains slow. In this work, we present GEM, a native indexing framework for multi-vector representations. The core idea is to construct a proximity graph directly over vector sets, preserving their fine-grained semantics while enabling efficient navigation. First, GEM designs a set-level clustering scheme. It associates each vector set with only its most informative clusters, effectively reducing redundancy without hurting semantic coverage. Then, it builds local proximity graphs within clusters and bridges them into a globally navigable structure. To handle the non-metric nature of multi-vector similarity, GEM decouples the graph construction metric from the final relevance score and injects semantic shortcuts to guide efficient navigation toward relevant regions. At query time, GEM launches beam search from multiple entry points and prunes paths early using cluster cues. To further enhance efficiency, a quantized distance estimation technique is used for both indexing and search. Across in-domain, out-of-domain, and multi-modal benchmarks, GEM achieves up to 16x speedup over state-of-the-art methods while matching or improving accuracy. 

---
# Bypassing Document Ingestion: An MCP Approach to Financial Q&A 

**Authors**: Sasan Mansouri, Edoardo Pilla, Mark Wahrenburg, Fabian Woebbeking  

**Link**: [PDF](https://arxiv.org/pdf/2603.20316)  

**Abstract**: Answering financial questions is often treated as an information retrieval problem. In practice, however, much of the relevant information is already available in curated vendor systems, especially for quantitative analysis. We study whether, and under which conditions, Model Context Protocol (MCP) offers a more reliable alternative to standard retrieval-augmented generation (RAG) by allowing large language models (LLMs) to interact directly with data rather than relying on document ingestion and chunk retrieval. We test this by building a custom MCP server that exposes LSEG APIs as tools and evaluating it on the FinDER benchmark. The approach performs particularly well on the Financials subset, achieving up to 80.4% accuracy on multi-step numerical questions when relevant context is retrieved. The paper thus provides both a baseline for MCP-based financial question answering (QA) and evidence on where this approach breaks down, such as for questions requiring qualitative or document-specific context. Overall, direct access to curated data is a lightweight and effective alternative to document-centric RAG for quantitative financial QA, but not a substitute for all financial QA tasks. 

---
# BubbleRAG: Evidence-Driven Retrieval-Augmented Generation for Black-Box Knowledge Graphs 

**Authors**: Duyi Pan, Tianao Lou, Xin Li, Haoze Song, Yiwen Wu, Mengyi Deng, Mingyu Yang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20309)  

**Abstract**: Large Language Models (LLMs) exhibit hallucinations in knowledge-intensive tasks. Graph-based retrieval augmented generation (RAG) has emerged as a promising solution, yet existing approaches suffer from fundamental recall and precision limitations when operating over black-box knowledge graphs -- graphs whose schema and structure are unknown in advance. We identify three core challenges that cause recall loss (semantic instantiation uncertainty and structural path uncertainty) and precision loss (evidential comparison uncertainty). To address these challenges, we formalize the retrieval task as the Optimal Informative Subgraph Retrieval (OISR) problem -- a variant of Group Steiner Tree -- and prove it to be NP-hard and APX-hard. We propose BubbleRAG, a training-free pipeline that systematically optimizes for both recall and precision through semantic anchor grouping, heuristic bubble expansion to discover candidate evidence graphs (CEGs), composite ranking, and reasoning-aware expansion. Experiments on multi-hop QA benchmarks demonstrate that BubbleRAG achieves state-of-the-art results, outperforming strong baselines in both F1 and accuracy while remaining plug-and-play. 

---
# Report-based Recommendations for Policy Making and Agency Operations: Dataset and LLM Evaluation 

**Authors**: Aleksandra Edwards, Thomas Edwards, Jose Camacho-Collados, Alun Preece  

**Link**: [PDF](https://arxiv.org/pdf/2603.20287)  

**Abstract**: Large Language Models (LLMs) are extensively used in text generation tasks. These generative capabilities bring us to a point where LLMs could potentially provide useful insights in policy making or agency operations. In this paper, we introduce a new task consisting of generating recommendations which can be used to inform future actions and improvements of agencies work within private and public organisations. In particular, we present the first benchmark and coherent evaluation for developing recommendation systems to inform organisation policies. This task is clearly different from usual product or user recommendation systems, but rather aims at providing a basis to suggest policy improvements based on the conclusions drawn from reports. Our results demonstrate that state-of-the-art LLMs have the potential to emphasize and reflect on key issues and learning points within generated recommendations. 

---
# Rethinking Retrieval-Augmentation as Synthesis: A Query-Aware Context Merging Approach 

**Authors**: Jiarui Guo, Yuemeng Xu, Zongwei Lv, Yangyujia Wang, Xiaolin Wang, Kan Liu, Tao Lan, Lin Qu, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20286)  

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to extend their existing knowledge by dynamically incorporating external information. However, practical deployment is fundamentally constrained by the LLM's finite context window, forcing a trade-off between information sufficiency and token consumption. Standard pipelines address this via a retrieve-then-select strategy, typically retaining only the top-k chunks based on relevance. Nevertheless, this approach is suboptimal: it inherently truncates critical bridging evidence located in the long tail of the relevance distribution, while simultaneously wasting the token budget on semantically redundant high-ranking chunks.
In this paper, we rethink retrieval-augmentation as a dynamic optimization problem aimed at maximizing information density. We propose MergeRAG, a novel framework that shifts the paradigm from static filtering to query-aware synthesis. MergeRAG employs a scoring agent to restructure retrieved contexts through a dual-pathway mechanism: 1) Symmetric Merging, which consolidates weak signals to recover lost bridging evidence; 2) Asymmetric Merging, which utilizes entropy-guided anchoring to eliminate redundancy without sacrificing semantic integrity. We further introduce a Hierarchical Parallel Merging strategy that mitigates information loss while maximizing computational parallelism. Extensive experiments on standard benchmarks demonstrate that MergeRAG significantly outperforms state-of-the-art RAG baselines, achieving up to 13.7 points improvement in F1 score and 11.5 points in Exact Match (EM), respectively. 

---
# FastPFRec: A Fast Personalized Federated Recommendation with Secure Sharing 

**Authors**: Zhenxing Yan, Jidong Yuan, Yongqi Sun, Haiyang Liu, Zhihui Gao  

**Link**: [PDF](https://arxiv.org/pdf/2603.20283)  

**Abstract**: Graph neural network (GNN)-based federated recommendation systems effectively capture user-item relationships while preserving data privacy. However, existing methods often face slow convergence on graph data and privacy leakage risks during collaboration. To address these challenges, we propose FastPFRec (Fast Personalized Federated Recommendation with Secure Sharing), a novel framework that enhances both training efficiency and data security. FastPFRec accelerates model convergence through an efficient local update strategy and introduces a privacy-aware parameter sharing mechanism to mitigate leakage risks. Experiments on four real-world datasets (Yelp, Kindle, Gowalla-100k, and Gowalla-1m) show that FastPFRec achieves 32.0% fewer training rounds, 34.1% shorter training time, and 8.1% higher accuracy compared with existing baselines. These results demonstrate that FastPFRec provides an efficient and privacy-preserving solution for scalable federated recommendation. 

---
# OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis 

**Authors**: Zhuofeng Li, Dongfu Jiang, Xueguang Ma, Haoxiang Zhang, Ping Nie, Yuyu Zhang, Kai Zou, Jianwen Xie, Yu Zhang, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.20278)  

**Abstract**: Training deep research agents requires long-horizon trajectories that interleave search, evidence aggregation, and multi-step reasoning. However, existing data collection pipelines typically rely on proprietary web APIs, making large-scale trajectory synthesis costly, unstable, and difficult to reproduce. We present OpenResearcher, a reproducible pipeline that decouples one-time corpus bootstrapping from multi-turn trajectory synthesis and executes the search-and-browse loop entirely offline using three explicit browser primitives: search, open, and find, over a 15M-document corpus. Using GPT-OSS-120B as the teacher model, we synthesize over 97K trajectories, including a substantial long-horizon tail with 100+ tool calls. Supervised fine-tuning a 30B-A3B backbone on these trajectories achieves 54.8\% accuracy on BrowseComp-Plus, a +34.0 point improvement over the base model, while remaining competitive on BrowseComp, GAIA, and xbench-DeepSearch. Because the environment is offline and fully instrumented, it also enables controlled analysis, where our study reveals practical insights into deep research pipeline design, including data filtering strategies, agent configuration choices, and how retrieval success relates to final answer accuracy. We release the pipeline, synthesized trajectories, model checkpoints, and the offline search environment at this https URL. 

---
# Semantic Shift: the Fundamental Challenge in Text Embedding and Retrieval 

**Authors**: Hang Gao, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2603.21437)  

**Abstract**: Transformer-based embedding models rely on pooling to map variable-length text into a single vector, enabling efficient similarity search but also inducing well-known geometric pathologies such as anisotropy and length-induced embedding collapse. Existing accounts largely describe \emph{what} these pathologies look like, yet provide limited insight into \emph{when} and \emph{why} they harm downstream retrieval. In this work, we argue that the missing causal factor is \emph{semantic shift}: the intrinsic, structured evolution and dispersion of semantics within a text.
We first present a theoretical analysis of \emph{semantic smoothing} in Transformer embeddings: as the semantic diversity among constituent sentences increases, the pooled representation necessarily shifts away from every individual sentence embedding, yielding a smoothed and less discriminative vector. Building on this foundation, we formalize semantic shift as a computable measure integrating local semantic evolution and global semantic dispersion. Through controlled experiments across corpora and multiple embedding models, we show that semantic shift aligns closely with the severity of embedding concentration and predicts retrieval degradation, whereas text length alone does not. Overall, semantic shift offers a unified and actionable lens for understanding embedding collapse and for diagnosing when anisotropy becomes harmful. 

---
# Graph Fusion Across Languages using Large Language Models 

**Authors**: Kaung Myat Kyaw, Khush Agarwal, Jonathan Chan  

**Link**: [PDF](https://arxiv.org/pdf/2603.21248)  

**Abstract**: Combining multiple knowledge graphs (KGs) across linguistic boundaries is a persistent challenge due to semantic heterogeneity and the complexity of graph environments. We propose a framework for cross-lingual graph fusion, leveraging the in-context reasoning and multilingual semantic priors of Large Language Models (LLMs). The framework implements structural linearization by mapping triplets directly into natural language sequences (e.g., [head] [relation] [tail]), enabling the LLM to map relations and reconcile entities between an evolving fused graph ($G_{c}^{(t-1)}$) and a new candidate graph ($G_{t}$). Evaluated on the DBP15K dataset, this exploratory study demonstrates that LLMs can serve as a universal semantic bridge to resolve cross-lingual discrepancies. Results show the successful sequential agglomeration of multiple heterogeneous graphs, offering a scalable, modular solution for continuous knowledge synthesis in multi-source, multilingual environments. 

---
# User Preference Modeling for Conversational LLM Agents: Weak Rewards from Retrieval-Augmented Interaction 

**Authors**: Yuren Hao, Shuhaib Mehri, ChengXiang Zhai, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2603.20939)  

**Abstract**: Large language models are increasingly used as personal assistants, yet most lack a persistent user model, forcing users to repeatedly restate preferences across sessions. We propose Vector-Adapted Retrieval Scoring (VARS), a pipeline-agnostic, frozen-backbone framework that represents each user with long-term and short-term vectors in a shared preference space and uses these vectors to bias retrieval scoring over structured preference memory. The vectors are updated online from weak scalar rewards from users' feedback, enabling personalization without per-user fine-tuning. We evaluate on \textsc{MultiSessionCollab}, an online multi-session collaboration benchmark with rich user preference profiles, across math and code tasks. Under frozen backbones, the main benefit of user-aware retrieval is improved interaction efficiency rather than large gains in raw task accuracy: our full VARS agent achieves the strongest overall performance, matches a strong Reflection baseline in task success, and reduces timeout rate and user effort. The learned long-term vectors also align with cross-user preference overlap, while short-term vectors capture session-specific adaptation, supporting the interpretability of the dual-vector design. Code, model, and data are available at this https URL. 

---
# yProv4DV: Reproducible Data Visualization Scripts Out of the Box 

**Authors**: Gabriele Padovani, Sandro Fiore  

**Link**: [PDF](https://arxiv.org/pdf/2603.20437)  

**Abstract**: While results visualization is a critical phase to the communication of new academic results, plots are frequently shared without the complete combination of code, input data, execution context and outputs required to independently reproduce the resulting figures. Existing reproducibility solutions tend to focus on computational pipelines or workflow management systems, not covering script-based visualization practices commonly used by researchers and practitioners. Additionally, the minimalist nature of current Python data visualization libraries tend to speed up the creation of images, disincentivizing users from spending time integrating additional tools into these short scripts. This paper proposes yProv4DV, a library lightweight designed to enable reproducible data visualization scripts through the use of provenance information, minimizing the necessity for code modifications. Through a single call, users can track inputs, outputs and source code files, enabling saving and full reproducibility of their data visualization software. As a result, this library fills a gap in reproducible research workflows by addressing the reproducibility of plots in scientific publications. 

---
# PEARL: Personalized Streaming Video Understanding Model 

**Authors**: Yuanhong Zheng, Ruichuan An, Xiaopeng Lin, Yuxing Liu, Sihan Yang, Huanyu Zhang, Haodong Li, Qintong Zhang, Renrui Zhang, Guopeng Li, Yifan Zhang, Yuheng Li, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20422)  

**Abstract**: Human cognition of new concepts is inherently a streaming process: we continuously recognize new objects or identities and update our memories over time. However, current multimodal personalization methods are largely limited to static images or offline videos. This disconnects continuous visual input from instant real-world feedback, limiting their ability to provide the real-time, interactive personalized responses essential for future AI assistants. To bridge this gap, we first propose and formally define the novel task of Personalized Streaming Video Understanding (PSVU). To facilitate research in this new direction, we introduce PEARL-Bench, the first comprehensive benchmark designed specifically to evaluate this challenging setting. It evaluates a model's ability to respond to personalized concepts at exact timestamps under two modes: (1) Frame-level, focusing on a specific person or object in discrete frames, and (2) a novel Video-level, focusing on personalized actions unfolding across continuous frames. PEARL-Bench comprises 132 unique videos and 2,173 fine-grained annotations with precise timestamps. Concept diversity and annotation quality are strictly ensured through a combined pipeline of automated generation and human verification. To tackle this challenging new setting, we further propose PEARL, a plug-and-play, training-free strategy that serves as a strong baseline. Extensive evaluations across 8 offline and online models demonstrate that PEARL achieves state-of-the-art performance. Notably, it brings consistent PSVU improvements when applied to 3 distinct architectures, proving to be a highly effective and robust strategy. We hope this work advances vision-language model (VLM) personalization and inspires further research into streaming personalized AI assistants. Code is available at this https URL. 

---
