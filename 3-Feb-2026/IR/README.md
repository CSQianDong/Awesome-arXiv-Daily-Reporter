# RANKVIDEO: Reasoning Reranking for Text-to-Video Retrieval 

**Authors**: Tyler Skow, Alexander Martin, Benjamin Van Durme, Rama Chellappa, Reno Kriz  

**Link**: [PDF](https://arxiv.org/pdf/2602.02444)  

**Abstract**: Reranking is a critical component of modern retrieval systems, which typically pair an efficient first-stage retriever with a more expressive model to refine results. While large reasoning models have driven rapid progress in text-centric reranking, reasoning-based reranking for video retrieval remains underexplored. To address this gap, we introduce RANKVIDEO, a reasoning-based reranker for video retrieval that explicitly reasons over query-video pairs using video content to assess relevance. RANKVIDEO is trained using a two-stage curriculum consisting of perception-grounded supervised fine-tuning followed by reranking training that combines pointwise, pairwise, and teacher confidence distillation objectives, and is supported by a data synthesis pipeline for constructing reasoning-intensive query-video pairs. Experiments on the large-scale MultiVENT 2.0 benchmark demonstrate that RANKVIDEO consistently improves retrieval performance within a two-stage framework, yielding an average improvement of 31% on nDCG@10 and outperforming text-only and vision-language reranking alternatives, while more efficient. 

---
# Rethinking Generative Recommender Tokenizer: Recsys-Native Encoding and Semantic Quantization Beyond LLMs 

**Authors**: Yu Liang, Zhongjin Zhang, Yuxuan Zhu, Kerui Zhang, Zhiluohan Guo, Wenhang Zhou, Zonqi Yang, Kangle Wu, Yabo Ni, Anxiang Zeng, Cong Fu, Jianxin Wang, Jiazhi Xia  

**Link**: [PDF](https://arxiv.org/pdf/2602.02338)  

**Abstract**: Semantic ID (SID)-based recommendation is a promising paradigm for scaling sequential recommender systems, but existing methods largely follow a semantic-centric pipeline: item embeddings are learned from foundation models and discretized using generic quantization schemes. This design is misaligned with generative recommendation objectives: semantic embeddings are weakly coupled with collaborative prediction, and generic quantization is inefficient at reducing sequential uncertainty for autoregressive modeling. To address these, we propose ReSID, a recommendation-native, principled SID framework that rethinks representation learning and quantization from the perspective of information preservation and sequential predictability, without relying on LLMs. ReSID consists of two components: (i) Field-Aware Masked Auto-Encoding (FAMAE), which learns predictive-sufficient item representations from structured features, and (ii) Globally Aligned Orthogonal Quantization (GAOQ), which produces compact and predictable SID sequences by jointly reducing semantic ambiguity and prefix-conditional uncertainty. Theoretical analysis and extensive experiments across ten datasets show the effectiveness of ReSID. ReSID consistently outperforms strong sequential and SID-based generative baselines by an average of over 10%, while reducing tokenization cost by up to 122x. Code is available at this https URL. 

---
# Adaptive Quality-Diversity Trade-offs for Large-Scale Batch Recommendation 

**Authors**: Clémence Réda, Tomas Rigaux, Hiba Bederina, Koh Takeuchi, Hisashi Kashima, Jill-Jênn Vie  

**Link**: [PDF](https://arxiv.org/pdf/2602.02024)  

**Abstract**: A core research question in recommender systems is to propose batches of highly relevant and diverse items, that is, items personalized to the user's preferences, but which also might get the user out of their comfort zone. This diversity might induce properties of serendipidity and novelty which might increase user engagement or revenue. However, many real-life problems arise in that case: e.g., avoiding to recommend distinct but too similar items to reduce the churn risk, and computational cost for large item libraries, up to millions of items. First, we consider the case when the user feedback model is perfectly observed and known in advance, and introduce an efficient algorithm called B-DivRec combining determinantal point processes and a fuzzy denuding procedure to adjust the degree of item diversity. This helps enforcing a quality-diversity trade-off throughout the user history. Second, we propose an approach to adaptively tailor the quality-diversity trade-off to the user, so that diversity in recommendations can be enhanced if it leads to positive feedback, and vice-versa. Finally, we illustrate the performance and versatility of B-DivRec in the two settings on synthetic and real-life data sets on movie recommendation and drug repurposing. 

---
# GRAB: An LLM-Inspired Sequence-First Click-Through Rate Prediction Modeling Paradigm 

**Authors**: Shaopeng Chen, Chuyue Xie, Huimin Ren, Shaozong Zhang, Han Zhang, Ruobing Cheng, Zhiqiang Cao, Zehao Ju, Gao Yu, Jie Ding, Xiaodong Chen, Xuewu Jiao, Shuanglong Li, Liu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.01865)  

**Abstract**: Traditional Deep Learning Recommendation Models (DLRMs) face increasing bottlenecks in performance and efficiency, often struggling with generalization and long-sequence modeling. Inspired by the scaling success of Large Language Models (LLMs), we propose Generative Ranking for Ads at Baidu (GRAB), an end-to-end generative framework for Click-Through Rate (CTR) prediction. GRAB integrates a novel Causal Action-aware Multi-channel Attention (CamA) mechanism to effectively capture temporal dynamics and specific action signals within user behavior sequences. Full-scale online deployment demonstrates that GRAB significantly outperforms established DLRMs, delivering a 3.05% increase in revenue and a 3.49% rise in CTR. Furthermore, the model demonstrates desirable scaling behavior: its expressive power shows a monotonic and approximately linear improvement as longer interaction sequences are utilized. 

---
# Unifying Ranking and Generation in Query Auto-Completion via Retrieval-Augmented Generation and Multi-Objective Alignment 

**Authors**: Kai Yuan, Anthony Zheng, Jia Hu, Divyanshu Sheth, Hemanth Velaga, Kylee Kim, Matteo Guarrera, Besim Avci, Xuetao Yin, Rajyashree Mukherjee, Sean Suchter  

**Link**: [PDF](https://arxiv.org/pdf/2602.01023)  

**Abstract**: Query Auto-Completion (QAC) suggests query completions as users type, helping them articulate intent and reach results more efficiently. Existing approaches face fundamental challenges: traditional retrieve-and-rank pipelines have limited long-tail coverage and require extensive feature engineering, while recent generative methods suffer from hallucination and safety risks. We present a unified framework that reformulates QAC as end-to-end list generation through Retrieval-Augmented Generation (RAG) and multi-objective Direct Preference Optimization (DPO). Our approach combines three key innovations: (1) reformulating QAC as end-to-end list generation with multi-objective optimization; (2) defining and deploying a suite of rule-based, model-based, and LLM-as-judge verifiers for QAC, and using them in a comprehensive methodology that combines RAG, multi-objective DPO, and iterative critique-revision for high-quality synthetic data; (3) a hybrid serving architecture enabling efficient production deployment under strict latency constraints. Evaluation on a large-scale commercial search platform demonstrates substantial improvements: offline metrics show gains across all dimensions, human evaluation yields +0.40 to +0.69 preference scores, and a controlled online experiment achieves 5.44\% reduction in keystrokes and 3.46\% increase in suggestion adoption, validating that unified generation with RAG and multi-objective alignment provides an effective solution for production QAC. This work represents a paradigm shift to end-to-end generation powered by large language models, RAG, and multi-objective alignment, establishing a production-validated framework that can benefit the broader search and recommendation industry. 

---
# Optimizing Retrieval Components for a Shared Backbone via Component-Wise Multi-Stage Training 

**Authors**: Yunhan Li, Mingjie Xie, Zihan Gong, Zeyang Shi, Gengshen Wu, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.00805)  

**Abstract**: Recent advances in embedding-based retrieval have enabled dense retrievers to serve as core infrastructure in many industrial systems, where a single retrieval backbone is often shared across multiple downstream applications. In such settings, retrieval quality directly constrains system performance and extensibility, while coupling model selection, deployment, and rollback decisions across applications.
In this paper, we present empirical findings and a system-level solution for optimizing retrieval components deployed as a shared backbone in production legal retrieval systems. We adopt a multi-stage optimization framework for dense retrievers and rerankers, and show that different retrieval components exhibit stage-dependent trade-offs. These observations motivate a component-wise, mixed-stage configuration rather than relying on a single uniformly optimal checkpoint. The resulting backbone is validated through end-to-end evaluation and deployed as a shared retrieval service supporting multiple industrial applications. 

---
# Towards Trustworthy Multimodal Recommendation 

**Authors**: Zixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.00730)  

**Abstract**: Recent advances in multimodal recommendation have demonstrated the effectiveness of incorporating visual and textual content into collaborative filtering. However, real-world deployments raise an increasingly important yet underexplored issue: trustworthiness. On modern e-commerce platforms, multimodal content can be misleading or unreliable (e.g., visually inconsistent product images or click-bait titles), injecting untrustworthy signals into multimodal representations and making existing recommenders brittle under modality corruption. In this work, we take a step towards trustworthy multimodal recommendation from both a method and an analysis perspective. First, we propose a plug-and-play modality-level rectification component that mitigates untrustworthy modality features by learning soft correspondences between items and multimodal features. Using lightweight projections and Sinkhorn-based soft matching, the rectification suppresses mismatched modality signals while preserving semantic consistency, and can be integrated into existing multimodal recommenders without architectural modifications. Second, we present two practical insights on interaction-level trustworthiness under noisy collaborative signals: (i) training-set pseudo interactions can help or hurt performance under noise depending on prior-signal alignment; and (ii) propagation-graph pseudo edges can also help or hurt robustness, as message passing may amplify misalignment. Extensive experiments on multiple datasets and backbones under varying corruption levels demonstrate improved robustness from modality rectification and validate the above interaction-level observations. 

---
# SWGCN: Synergy Weighted Graph Convolutional Network for Multi-Behavior Recommendation 

**Authors**: Fangda Chen, Yueyang Wang, Chaoli Lou, Min Gao, Qingyu Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2602.00727)  

**Abstract**: Multi-behavior recommendation paradigms have emerged to capture diverse user activities, forecasting primary conversions (e.g., purchases) by leveraging secondary signals like browsing history. However, current graph-based methods often overlook cross-behavioral synergistic signals and fine-grained intensity of individual actions. Motivated by the need to overcome these shortcomings, we introduce Synergy Weighted Graph Convolutional Network (SWGCN). SWGCN introduces two novel components: a Target Preference Weigher, which adaptively assigns weights to user-item interactions within each behavior, and a Synergy Alignment Task, which guides its training by leveraging an Auxiliary Preference Valuator. This task prioritizes interactions from synergistic signals that more accurately reflect user preferences. The performance of our model is rigorously evaluated through comprehensive tests on three open-source datasets, specifically Taobao, IJCAI, and Beibei. On the Taobao dataset, SWGCN yields relative gains of 112.49% and 156.36% in terms of Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG), respectively. It also yields consistent gains on IJCAI and Beibei, confirming its robustness and generalizability across various datasets. Our implementation is open-sourced and can be accessed via this https URL. 

---
# RecGOAT: Graph Optimal Adaptive Transport for LLM-Enhanced Multimodal Recommendation with Dual Semantic Alignment 

**Authors**: Yuecheng Li, Hengwei Ju, Zeyu Song, Wei Yang, Chi Lu, Peng Jiang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2602.00682)  

**Abstract**: Multimodal recommendation systems typically integrates user behavior with multimodal data from items, thereby capturing more accurate user preferences. Concurrently, with the rise of large models (LMs), multimodal recommendation is increasingly leveraging their strengths in semantic understanding and contextual reasoning. However, LM representations are inherently optimized for general semantic tasks, while recommendation models rely heavily on sparse user/item unique identity (ID) features. Existing works overlook the fundamental representational divergence between large models and recommendation systems, resulting in incompatible multimodal representations and suboptimal recommendation performance. To bridge this gap, we propose RecGOAT, a novel yet simple dual semantic alignment framework for LLM-enhanced multimodal recommendation, which offers theoretically guaranteed alignment capability. RecGOAT first employs graph attention networks to enrich collaborative semantics by modeling item-item, user-item, and user-user relationships, leveraging user/item LM representations and interaction history. Furthermore, we design a dual-granularity progressive multimodality-ID alignment framework, which achieves instance-level and distribution-level semantic alignment via cross-modal contrastive learning (CMCL) and optimal adaptive transport (OAT), respectively. Theoretically, we demonstrate that the unified representations derived from our alignment framework exhibit superior semantic consistency and comprehensiveness. Extensive experiments on three public benchmarks show that our RecGOAT achieves state-of-the-art performance, empirically validating our theoretical insights. Additionally, the deployment on a large-scale online advertising platform confirms the model's effectiveness and scalability in industrial recommendation scenarios. Code available at this https URL. 

---
# Towards Sample-Efficient and Stable Reinforcement Learning for LLM-based Recommendation 

**Authors**: Hongxun Ding, Keqin Bao, Jizhi Zhang, Yi Fang, Wenxin Xu, Fuli Feng, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2602.00632)  

**Abstract**: While Long Chain-of-Thought (Long CoT) reasoning has shown promise in Large Language Models (LLMs), its adoption for enhancing recommendation quality is growing rapidly. In this work, we critically examine this trend and argue that Long CoT is inherently ill-suited for the sequential recommendation domain. We attribute this misalignment to two primary factors: excessive inference latency and the lack of explicit cognitive reasoning patterns in user behavioral data. Driven by these observations, we propose pivoting away from the CoT structure to directly leverage its underlying mechanism: Reinforcement Learning (RL), to explore the item space. However, applying RL directly faces significant obstacles, notably low sample efficiency-where most actions fail to provide learning signals-and training instability. To overcome these limitations, we propose RISER, a novel Reinforced Item Space Exploration framework for Recommendation. RISER is designed to transform non-learnable trajectories into effective pairwise preference data for optimization. Furthermore, it incorporates specific strategies to ensure stability, including the prevention of redundant rollouts and the constraint of token-level update magnitudes. Extensive experiments on three real-world datasets show that RISER significantly outperforms competitive baselines, establishing a robust paradigm for RL-enhanced LLM recommendation. Our code will be available at this https URL. 

---
# Equity vs. Equality: Optimizing Ranking Fairness for Tailored Provider Needs 

**Authors**: Yiteng Tu, Weihang Su, Shuguang Han, Yiqun Liu, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2602.00495)  

**Abstract**: Ranking plays a central role in connecting users and providers in Information Retrieval (IR) systems, making provider-side fairness an important challenge. While recent research has begun to address fairness in ranking, most existing approaches adopt an equality-based perspective, aiming to ensure that providers with similar content receive similar exposure. However, it overlooks the diverse needs of real-world providers, whose utility from ranking may depend not only on exposure but also on outcomes like sales or engagement. Consequently, exposure-based fairness may not accurately capture the true utility perceived by different providers with varying priorities. To this end, we introduce an equity-oriented fairness framework that explicitly models each provider's preferences over key outcomes such as exposure and sales, thus evaluating whether a ranking algorithm can fulfill these individualized goals while maintaining overall fairness across providers. Based on this framework, we develop EquityRank, a gradient-based algorithm that jointly optimizes user-side effectiveness and provider-side equity. Extensive offline and online simulations demonstrate that EquityRank offers improved trade-offs between effectiveness and fairness and adapts to heterogeneous provider needs. 

---
# RAGRouter-Bench: A Dataset and Benchmark for Adaptive RAG Routing 

**Authors**: Ziqi Wang, Xi Zhu, Shuhang Lin, Haochen Xue, Minghao Guo, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.00296)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a core paradigm for grounding large language models with external knowledge. Despite extensive efforts exploring diverse retrieval strategies, existing studies predominantly focus on query-side complexity or isolated method improvements, lacking a systematic understanding of how RAG paradigms behave across different query-corpus contexts and effectiveness-efficiency trade-offs. In this work, we introduce RAGRouter-Bench, the first dataset and benchmark designed for adaptive RAG routing. RAGRouter-Bench revisits retrieval from a query-corpus compatibility perspective and standardizes five representative RAG paradigms for systematic evaluation across 7,727 queries and 21,460 documents spanning diverse domains. The benchmark incorporates three canonical query types together with fine-grained semantic and structural corpus metrics, as well as a unified evaluation for both generation quality and resource consumption. Experiments with DeepSeek-V3 and LLaMA-3.1-8B demonstrate that no single RAG paradigm is universally optimal, that paradigm applicability is strongly shaped by query-corpus interactions, and that increased advanced mechanism does not necessarily yield better effectiveness-efficiency trade-offs. These findings underscore the necessity of routing-aware evaluation and establish a foundation for adaptive, interpretable, and generalizable next-generation RAG systems. 

---
# SPARC-RAG: Adaptive Sequential-Parallel Scaling with Context Management for Retrieval-Augmented Generation 

**Authors**: Yuxin Yang, Gangda Deng, Ömer Faruk Akgül, Nima Chitsazan, Yash Govilkar, Akasha Tigalappanavara, Shi-Xiong Zhang, Sambit Sahu, Viktor Prasanna  

**Link**: [PDF](https://arxiv.org/pdf/2602.00083)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds large language model outputs in external evidence, but remains challenged on multi-hop question answering that requires long reasoning. Recent works scale RAG at inference time along two complementary dimensions: sequential depth for iterative refinement and parallel width for coverage expansion. However, naive scaling causes context contamination and scaling inefficiency, leading to diminishing or negative returns despite increased computation. To address these limitations, we propose SPARC-RAG, a multi-agent framework that coordinates sequential and parallel inference-time scaling under a unified context management mechanism. SPARC-RAG employs specialized agents that maintain a shared global context and provide explicit control over the scaling process. It generates targeted, complementary sub-queries for each branch to enable diverse parallel exploration, and explicitly regulates exiting decisions based on answer correctness and evidence grounding. To optimize scaling behavior, we further introduce a lightweight fine-tuning method with process-level verifiable preferences, which improves the efficiency of sequential scaling and effectiveness of parallel scaling. Across single- and multi-hop QA benchmarks, SPARC-RAG consistently outperforms previous RAG baselines, yielding an average +6.2 F1 improvement under lower inference cost. 

---
# AI-assisted Protocol Information Extraction For Improved Accuracy and Efficiency in Clinical Trial Workflows 

**Authors**: Ramtin Babaeipour, François Charest, Madison Wright  

**Link**: [PDF](https://arxiv.org/pdf/2602.00052)  

**Abstract**: Increasing clinical trial protocol complexity, amendments, and challenges around knowledge management create significant burden for trial teams. Structuring protocol content into standard formats has the potential to improve efficiency, support documentation quality, and strengthen compliance. We evaluate an Artificial Intelligence (AI) system using generative LLMs with Retrieval-Augmented Generation (RAG) for automated clinical trial protocol information extraction. We compare the extraction accuracy of our clinical-trial-specific RAG process against that of publicly available (standalone) LLMs. We also assess the operational impact of AI-assistance on simulated extraction CRC workflows. Our RAG process was measured as more accurate (87.8%) than standalone LLMs with fine-tuned prompts (62.6%) against expert-supported reference annotations. In the simulated extraction workflows, AI-assisted tasks were completed 40% faster, rated as less cognitively demanding and strongly preferred by users. While expert oversight remains essential, this suggests that AI-assisted extraction can enable protocol intelligence at scale, motivating the integration of similar methodologies into real world clinical workflows to further validate its impact on feasibility, study start-up, and post-activation monitoring. 

---
# Linear-PAL: A Lightweight Ranker for Mitigating Shortcut Learning in Personalized, High-Bias Tabular Ranking 

**Authors**: Vipul Dinesh Pawar  

**Link**: [PDF](https://arxiv.org/pdf/2602.00013)  

**Abstract**: In e-commerce ranking, implicit user feedback is systematically confounded by Position Bias -- the strong propensity of users to interact with top-ranked items regardless of relevance. While Deep Learning architectures (e.g., Two-Tower Networks) are the standard solution for de-biasing, we demonstrate that in High-Bias Regimes, state-of-the-art Deep Ensembles suffer from Shortcut Learning: they minimize training loss by overfitting to the rank signal, leading to degraded ranking quality despite high prediction accuracy. We propose Linear Position-bias Aware Learning (Linear-PAL), a lightweight framework that enforces de-biasing through structural constraints: explicit feature conjunctions and aggressive regularization. We further introduce a Vectorized Integer Hashing technique for feature generation, replacing string-based operations with $O(N)$ vectorized arithmetic. Evaluating on a large-scale dataset (4.2M samples), Linear-PAL achieves Pareto Dominance: it outperforms Deep Ensembles in de-biased ranking quality (Relevance AUC: 0.7626 vs. 0.6736) while reducing training latency by 43x (40s vs 1762s). This computational efficiency enables high-frequency retraining, allowing the system to capture user-specific emerging market trends and deliver robust, personalized ranking in near real-time. 

---
# Chained Prompting for Better Systematic Review Search Strategies 

**Authors**: Fatima Nasser, Fouad Trad, Ammar Mohanna, Ghada El-Hajj Fuleihan, Ali Chehab  

**Link**: [PDF](https://arxiv.org/pdf/2602.00011)  

**Abstract**: Systematic reviews require the use of rigorously designed search strategies to ensure both comprehensive retrieval and minimization of bias. Conventional manual approaches, although methodologically systematic, are resource-intensive and susceptible to subjectivity, whereas heuristic and automated techniques frequently under-perform in recall unless supplemented by extensive expert input. We introduce a Large Language Model (LLM)-based chained prompt engineering framework for the automated development of search strategies in systematic reviews. The framework replicates the procedural structure of manual search design while leveraging LLMs to decompose review objectives, extract and formalize PICO elements, generate conceptual representations, expand terminologies, and synthesize Boolean queries. In addition to query construction, the framework exhibits superior performance in generating well-structured PICO elements relative to existing methods, thereby strengthening the foundation for high-recall search strategies. Evaluation on a subset of the LEADSInstruct dataset demonstrates that the framework attains a 0.9 average recall. These results significantly exceed the performance of existing approaches. Error analysis further highlights the critical role of precise objective specification and terminological alignment in optimizing retrieval effectiveness. These findings confirm the capacity of LLM-based pipelines to yield transparent, reproducible, and high-performing search strategies, and highlight their potential as scalable instruments for supporting evidence synthesis and evidence-based practice. 

---
# ChunkNorris: A High-Performance and Low-Energy Approach to PDF Parsing and Chunking 

**Authors**: Mathieu Ciancone, Clovis Varangot-Reille, Marion Schaeffer  

**Link**: [PDF](https://arxiv.org/pdf/2602.00010)  

**Abstract**: In Retrieval-Augmented Generation applications, the Information Retrieval part is central as it provides the contextual information that enables a Large Language Model to generate an appropriate and truthful response. High quality parsing and chunking are critical as efficient data segmentation directly impacts downstream tasks, i.e. Information Retrieval and answer generation. In this paper, we introduce ChunkNorris, a novel heuristic-based technique designed to optimise the parsing and chunking of PDF documents. Our approach does not rely on machine learning and employs a suite of simple yet effective heuristics to achieve high performance with minimal computational overhead. We demonstrate the efficiency of ChunkNorris through a comprehensive benchmark against existing parsing and chunking methods, evaluating criteria such as execution time, energy consumption, and retrieval accuracy. We propose an open-access dataset to produce our results. ChunkNorris outperforms baseline and more advanced techniques, offering a practical and efficient alternative for Information Retrieval tasks. Therefore, this research highlights the potential of heuristic-based methods for real-world, resource-constrained RAG use cases. 

---
# Front-Loaded or Balanced? The Mechanism through Which Review Order Affects Overall Ratings in Premium Service Settings 

**Authors**: He Wang, Ziyu Zhou, Hanxiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00008)  

**Abstract**: In the increasingly prevalent landscape of high-quality service contexts, whether consumer evaluation interfaces adopt a rating-first or review-first sequence has become a critical factor shaping rating authenticity and feedback quality. While prior research has primarily examined review content and sentiment, systematic investigation into how evaluation order influences rating outcomes remains limited. Through exploratory analyses, we find that Letterboxd -- which employs a review-first, rating-after mechanism -- exhibits a more centralized rating distribution with fewer extreme scores, whereas Yelp -- which adopts a rating-first, review-after mechanism -- shows a pronounced bimodal distribution with more polarized ratings. Three controlled experiments further demonstrate that in high-quality service contexts, a rating-first (vs. review-first) interface significantly elevates consumers' overall ratings. Mechanism analyses indicate that cognitive effort and affective heuristics serve as dual pathways: a rating-first (vs. review-first) sequence reduces cognitive effort and heightens affective heuristics, thereby increasing rating scores. Moreover, service quality moderates this process. When service quality is low, the rating-first (vs. review-first) sequence instead leads to lower ratings. This research reveals the psychological mechanisms through which evaluation order affects consumer ratings via cognitive and affective pathways. It extends theoretical understanding of online rating formation and offers practical implications for optimizing platform interface design to enhance rating authenticity and credibility. 

---
# FDA AI Search: Making FDA-Authorized AI Devices Searchable 

**Authors**: Arun Kavishwar, William Lotter  

**Link**: [PDF](https://arxiv.org/pdf/2602.00006)  

**Abstract**: Over 1,200 AI-enabled medical devices have received marketing authorization from the U.S. FDA, yet identifying devices suited to specific clinical needs remains challenging because the FDA's databases contain only limited metadata and non-searchable summary PDFs. To address this gap, we developed FDA AI Search, a website that enables semantic querying of FDA-authorized AI-enabled devices. The backend includes an embedding-based retrieval system, where LLM-extracted features from authorization summaries are compared to user queries to find relevant matches. We present quantitative and qualitative evaluation that support the effectiveness of the retrieval algorithm compared to keyword-based methods. As FDA-authorized AI devices become increasingly prevalent and their use cases expand, we envision that the tool will assist healthcare providers in identifying devices aligned with their clinical needs and support developers in formulating novel AI applications. 

---
# AutoBool: An Reinforcement-Learning trained LLM for Effective Automated Boolean Query Generation for Systematic Reviews 

**Authors**: Shuai Wang, Harrisen Scells, Bevan Koopman, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2602.00005)  

**Abstract**: We present AutoBool, a reinforcement learning (RL) framework that trains large language models (LLMs) to generate effective Boolean queries for medical systematic reviews. Boolean queries are the primary mechanism for literature retrieval in this domain and must achieve high recall while maintaining reasonable precision - a challenging balance that existing prompt-based LLM approaches often struggle to achieve. A major limitation in this space is the lack of high-quality ground-truth Boolean queries for each topic, which makes supervised fine-tuning impractical. AutoBool addresses this challenge by using RL to directly optimize query generation with retrieval measures, without requiring target queries. To support this effort, we create and release the largest dataset of its kind: 65588 topics in total for training and evaluating the task of automatic Boolean query formulation. Experiments on our new dataset and two established datasets (CLEF TAR and Seed Collection) show that AutoBool significantly outperforms zero shot/few shot prompting and matches or exceeds the effectiveness of much larger GPT-based models (e.g., GPT-4o, O3) using smaller backbones. It also approaches effectiveness of expert-authored queries while retrieving 10 to 16 times fewer documents. Ablation studies reveal the critical roles of model backbone, size, decoding temperature, and prompt design. Code and data are available at this https URL. 

---
# C$^2$-Cite: Contextual-Aware Citation Generation for Attributed Large Language Models 

**Authors**: Yue Yu, Ting Bai, HengZhi Lan, Li Qian, Li Peng, Jie Wu, Wei Liu, Jian Luan, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2602.00004)  

**Abstract**: The attribution technique enhances the credibility of LLMs by adding citations to the generated sentences, enabling users to trace back to the original sources and verify the reliability of the output. However, existing instruction-tuned attributed LLMs often fail to properly interpret the contextual semantics of citation symbols (e.g., [i]) during text generation. This shortcoming arises from their insufficient awareness of the context information surrounding citation markers, which in turn leads to disjointed references and poor integration of retrieved knowledge into the generated content. To address this issue, we propose a novel \textbf{C}ontextual-aware \textbf{C}itation generation framework (\textbf{C$^2$}-\textbf{Cite}) that explicitly integrates the semantic relationships between citation markers and their referenced content. Specifically, a contextual citation alignment mechanism is adopted: it first encodes the retrieved document contexts into the symbol representation of citations, then aligns the marker numbers by decoding information from a citation router function. This mechanism enables the transformation of citation markers from generic placeholders into active knowledge pointers that link to the referenced source information. Experimental results on the ALCE benchmark across three datasets validate our framework C$^2$-Cite++: it outperforms the SOTA baseline by an average of 5.8\% in citation quality and 17.4\% in response correctness. The implementation is publicly available at this https URL 

---
# Efficient Multilingual Search Relevance Modeling in E-Commerce via LLM Mixture-of-Experts 

**Authors**: Ye Liu, Xu Chen, Wuji Chen, Mang Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.00003)  

**Abstract**: In e-commerce platforms, search relevance directly influences both user experience and merchant revenue. In multi-country deployments, diverse linguistic, cultural, and product catalog contexts introduce significant distribution shifts, posing substantial challenges to relevance modeling. Existing approaches typically enhance the reasoning or multilingual abilities of a single monolithic model, yet they remain limited by data diversity, coverage gaps, and high inference costs in heterogeneous environments. Our empirical analysis reveals that different LLM base models exhibit complementary strengths across languages and regions, motivating an expert-based architecture. We propose a scalable LLM-based Mixture-of-Experts (MoE) framework that dynamically routes queries to specialized experts and fuses their embeddings through concatenation. Among rule-based, pseudo-label-based, and fully end-to-end strategies, end-to-end hard routing with concatenation offers the best balance of effectiveness and efficiency. To mitigate inference overhead, we further develop an engineering-optimized offline batch pipeline with resource-efficient scheduling, which hides memory latency, improves GPU utilization, and reduces GPU-hour consumption by up to 35% compared with synchronous execution. On datasets spanning six Southeast Asian markets, our MoE improves AUC by 0.72 percentage points over a dense baseline with the same active parameters. Meanwhile, the optimized pipeline achieves 27.6 queries per second (QPS), a 9% throughput improvement. These results demonstrate superior multilingual relevance and efficiency, delivering strong cost-effectiveness for real-world e-commerce search systems. 

---
# Disentangled Interest Network for Out-of-Distribution CTR Prediction 

**Authors**: Yu Zheng, Chen Gao, Jianxin Chang, Yanan Niu, Yang Song, Depeng Jin, Meng Wang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.00002)  

**Abstract**: Click-through rate (CTR) prediction, which estimates the probability of a user clicking on a given item, is a critical task for online information services. Existing approaches often make strong assumptions that training and test data come from the same distribution. However, the data distribution varies since user interests are constantly evolving, resulting in the out-of-distribution (OOD) issue. In addition, users tend to have multiple interests, some of which evolve faster than others. Towards this end, we propose Disentangled Click-Through Rate prediction (DiseCTR), which introduces a causal perspective of recommendation and disentangles multiple aspects of user interests to alleviate the OOD issue in recommendation. We conduct a causal factorization of CTR prediction involving user interest, exposure model, and click model, based on which we develop a deep learning implementation for these three causal mechanisms. Specifically, we first design an interest encoder with sparse attention which maps raw features to user interests, and then introduce a weakly supervised interest disentangler to learn independent interest embeddings, which are further integrated by an attentive interest aggregator for prediction. Experimental results on three real-world datasets show that DiseCTR achieves the best accuracy and robustness in OOD recommendation against state-of-the-art approaches, significantly improving AUC and GAUC by over 0.02 and reducing logloss by over 13.7%. Further analyses demonstrate that DiseCTR successfully disentangles user interests, which is the key to OOD generalization for CTR prediction. We have released the code and data at this https URL. 

---
# Trust by Design: Skill Profiles for Transparent, Cost-Aware LLM Routing 

**Authors**: Mika Okamoto, Ansel Kaplan Erol, Glenn Matlin  

**Link**: [PDF](https://arxiv.org/pdf/2602.02386)  

**Abstract**: How should Large Language Model (LLM) practitioners select the right model for a task without wasting money? We introduce BELLA (Budget-Efficient LLM Selection via Automated skill-profiling), a framework that recommends optimal LLM selection for tasks through interpretable skill-based model selection. Standard benchmarks report aggregate metrics that obscure which specific capabilities a task requires and whether a cheaper model could suffice. BELLA addresses this gap through three stages: (1) decomposing LLM outputs and extract the granular skills required by using critic-based profiling, (2) clustering skills into structured capability matrices, and (3) multi-objective optimization to select the right models to maximize performance while respecting budget constraints. BELLA provides natural-language rationale for recommendations, providing transparency that current black-box routing systems lack. We describe the framework architecture, situate it within the landscape of LLM routing and evaluation, and discuss its application to financial reasoning as a representative domain exhibiting diverse skill requirements and cost-variation across models. Our framework enables practitioners to make principled and cost-performance trade-offs for deploying LLMs. 

---
# Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics 

**Authors**: Ziwen Xu, Chenyan Wu, Hengyu Sun, Haiwen Hong, Mengru Wang, Yunzhi Yao, Longtao Huang, Hui Xue, Shumin Deng, Zhixuan Chu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02343)  

**Abstract**: Methods for controlling large language models (LLMs), including local weight fine-tuning, LoRA-based adaptation, and activation-based interventions, are often studied in isolation, obscuring their connections and making comparison difficult. In this work, we present a unified view that frames these interventions as dynamic weight updates induced by a control signal, placing them within a single conceptual framework. Building on this view, we propose a unified preference-utility analysis that separates control effects into preference, defined as the tendency toward a target concept, and utility, defined as coherent and task-valid generation, and measures both on a shared log-odds scale using polarity-paired contrastive examples. Across methods, we observe a consistent trade-off between preference and utility: stronger control increases preference while predictably reducing utility. We further explain this behavior through an activation manifold perspective, in which control shifts representations along target-concept directions to enhance preference, while utility declines primarily when interventions push representations off the model's valid-generation manifold. Finally, we introduce a new steering approach SPLIT guided by this analysis that improves preference while better preserving utility. Code is available at this https URL. 

---
# Towards AI Evaluation in Domain-Specific RAG Systems: The AgriHubi Case Study 

**Authors**: Md. Toufique Hasan, Ayman Asad Khan, Mika Saari, Vaishnavi Bankhele, Pekka Abrahamsson  

**Link**: [PDF](https://arxiv.org/pdf/2602.02208)  

**Abstract**: Large language models show promise for knowledge-intensive domains, yet their use in agriculture is constrained by weak grounding, English-centric training data, and limited real-world evaluation. These issues are amplified for low-resource languages, where high-quality domain documentation exists but remains difficult to access through general-purpose models. This paper presents AgriHubi, a domain-adapted retrieval-augmented generation (RAG) system for Finnish-language agricultural decision support. AgriHubi integrates Finnish agricultural documents with open PORO family models and combines explicit source grounding with user feedback to support iterative refinement. Developed over eight iterations and evaluated through two user studies, the system shows clear gains in answer completeness, linguistic accuracy, and perceived reliability. The results also reveal practical trade-offs between response quality and latency when deploying larger models. This study provides empirical guidance for designing and evaluating domain-specific RAG systems in low-resource language settings. 

---
# Deep learning enables urban change profiling through alignment of historical maps 

**Authors**: Sidi Wu, Yizi Chen, Maurizio Gribaudi, Konrad Schindler, Clément Mallet, Julien Perret, Lorenz Hurni  

**Link**: [PDF](https://arxiv.org/pdf/2602.02154)  

**Abstract**: Prior to modern Earth observation technologies, historical maps provide a unique record of long-term urban transformation and offer a lens on the evolving identity of cities. However, extracting consistent and fine-grained change information from historical map series remains challenging due to spatial misalignment, cartographic variation, and degrading document quality, limiting most analyses to small-scale or qualitative approaches. We propose a fully automated, deep learning-based framework for fine-grained urban change analysis from large collections of historical maps, built on a modular design that integrates dense map alignment, multi-temporal object detection, and change profiling. This framework shifts the analysis of historical maps from ad hoc visual comparison toward systematic, quantitative characterization of urban change. Experiments demonstrate the robust performance of the proposed alignment and object detection methods. Applied to Paris between 1868 and 1937, the framework reveals the spatial and temporal heterogeneity in urban transformation, highlighting its relevance for research in the social sciences and humanities. The modular design of our framework further supports adaptation to diverse cartographic contexts and downstream applications. 

---
# Orthogonal Hierarchical Decomposition for Structure-Aware Table Understanding with Large Language Models 

**Authors**: Bin Cao, Huixian Lu, Chenwen Ma, Ting Wang, Ruizhe Li, Jing Fan  

**Link**: [PDF](https://arxiv.org/pdf/2602.01969)  

**Abstract**: Complex tables with multi-level headers, merged cells and heterogeneous layouts pose persistent challenges for LLMs in both understanding and reasoning. Existing approaches typically rely on table linearization or normalized grid modeling. However, these representations struggle to explicitly capture hierarchical structures and cross-dimensional dependencies, which can lead to misalignment between structural semantics and textual representations for non-standard tables. To address this issue, we propose an Orthogonal Hierarchical Decomposition (OHD) framework that constructs structure-preserving input representations of complex tables for LLMs. OHD introduces an Orthogonal Tree Induction (OTI) method based on spatial--semantic co-constraints, which decomposes irregular tables into a column tree and a row tree to capture vertical and horizontal hierarchical dependencies, respectively. Building on this representation, we design a dual-pathway association protocol to symmetrically reconstruct semantic lineage of each cell, and incorporate an LLM as a semantic arbitrator to align multi-level semantic information. We evaluate OHD framework on two complex table question answering benchmarks, AITQA and HiTab. Experimental results show that OHD consistently outperforms existing representation paradigms across multiple evaluation metrics. 

---
# Mapping a Decade of Avian Influenza Research (2014-2023): A Scientometric Analysis from Web of Science 

**Authors**: Muneer Ahmad, Undie Felicia Nkatv, Amrita Sharma, Gorrety Maria Juma, Nicholas Kamoga, Julirine Nakanwag  

**Link**: [PDF](https://arxiv.org/pdf/2602.01712)  

**Abstract**: This scientometric study analyzes Avian Influenza research from 2014 to 2023 using bibliographic data from the Web of Science database. We examined publication trends, sources, authorship, collaborative networks, document types, and geographical distribution to gain insights into the global research landscape. Results reveal a steady increase in publications, with high contributions from Chinese and American institutions. Journals such as PLoS One and the Journal of Virology published the highest number of studies, indicating their influence in this field. The most prolific institutions include the Chinese Academy of Sciences and the University of Hong Kong, while the College of Veterinary Medicine at South China Agricultural University emerged as the most productive department. China and the USA lead in publication volume, though developed nations like the United Kingdom and Germany exhibit a higher rate of international collaboration. "Articles" are the most common document type, constituting 84.6% of the total, while "Reviews" account for 7.6%. This study provides a comprehensive view of global trends in Avian Influenza research, emphasizing the need for collaborative efforts across borders. 

---
# Unmediated AI-Assisted Scholarly Citations 

**Authors**: Stefan Szeider  

**Link**: [PDF](https://arxiv.org/pdf/2602.01686)  

**Abstract**: Traditional bibliography databases require users to navigate search forms and manually copy citation data. Language models offer an alternative: a natural-language interface where researchers write text with informal citation fragments, which are automatically resolved to proper references. However, language models are not reliable for scholarly work as they generate fabricated (hallucinated) citations at substantial rates.
We present an architectural approach that combines the natural-language interface of LLM chatbots with the accuracy of direct database access, implemented through the Model Context Protocol. Our system enables language models to search bibliographic databases, perform fuzzy matching, and export verified entries, all through conversational interaction.
A key architectural principle bypasses the language model during final data export: entries are fetched directly from authoritative sources, with timeout protection, to guarantee accuracy. We demonstrate this approach with MCP-DBLP, a server providing access to the DBLP computer science bibliography. The system transforms form-based bibliographic services into conversational assistants that maintain scholarly integrity. This architecture is adaptable to other bibliographic databases and academic data sources. 

---
# LLM-based Embeddings: Attention Values Encode Sentence Semantics Better Than Hidden States 

**Authors**: Yeqin Zhang, Yunfei Wang, Jiaxuan Chen, Ke Qin, Yizheng Zhao, Cam-Tu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2602.01572)  

**Abstract**: Sentence representations are foundational to many Natural Language Processing (NLP) applications. While recent methods leverage Large Language Models (LLMs) to derive sentence representations, most rely on final-layer hidden states, which are optimized for next-token prediction and thus often fail to capture global, sentence-level semantics. This paper introduces a novel perspective, demonstrating that attention value vectors capture sentence semantics more effectively than hidden states. We propose Value Aggregation (VA), a simple method that pools token values across multiple layers and token indices. In a training-free setting, VA outperforms other LLM-based embeddings, even matches or surpasses the ensemble-based MetaEOL. Furthermore, we demonstrate that when paired with suitable prompts, the layer attention outputs can be interpreted as aligned weighted value vectors. Specifically, the attention scores of the last token function as the weights, while the output projection matrix ($W_O$) aligns these weighted value vectors with the common space of the LLM residual stream. This refined method, termed Aligned Weighted VA (AlignedWVA), achieves state-of-the-art performance among training-free LLM-based embeddings, outperforming the high-cost MetaEOL by a substantial margin. Finally, we highlight the potential of obtaining strong LLM embedding models through fine-tuning Value Aggregation. 

---
# The Algorithmic Self-Portrait: Deconstructing Memory in ChatGPT 

**Authors**: Abhisek Dash, Soumi Das, Elisabeth Kirsten, Qinyuan Wu, Sai Keerthana Karnam, Krishna P. Gummadi, Thorsten Holz, Muhammad Bilal Zafar, Savvas Zannettou  

**Link**: [PDF](https://arxiv.org/pdf/2602.01450)  

**Abstract**: To enable personalized and context-aware interactions, conversational AI systems have introduced a new mechanism: Memory. Memory creates what we refer to as the Algorithmic Self-portrait - a new form of personalization derived from users' self-disclosed information divulged within private conversations. While memory enables more coherent exchanges, the underlying processes of memory creation remain opaque, raising critical questions about data sensitivity, user agency, and the fidelity of the resulting portrait.
To bridge this research gap, we analyze 2,050 memory entries from 80 real-world ChatGPT users. Our analyses reveal three key findings: (1) A striking 96% of memories in our dataset are created unilaterally by the conversational system, potentially shifting agency away from the user; (2) Memories, in our dataset, contain a rich mix of GDPR-defined personal data (in 28% memories) along with psychological insights about participants (in 52% memories); and (3)~A significant majority of the memories (84%) are directly grounded in user context, indicating faithful representation of the conversations. Finally, we introduce a framework-Attribution Shield-that anticipates these inferences, alerts about potentially sensitive memory inferences, and suggests query reformulations to protect personal information without sacrificing utility. 

---
# PARSE: An Open-Domain Reasoning Question Answering Benchmark for Persian 

**Authors**: Jamshid Mozafari, Seyed Parsa Mousavinasab, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2602.01246)  

**Abstract**: Reasoning-focused Question Answering (QA) has advanced rapidly with Large Language Models (LLMs), yet high-quality benchmarks for low-resource languages remain scarce. Persian, spoken by roughly 130 million people, lacks a comprehensive open-domain resource for evaluating reasoning-capable QA systems. We introduce PARSE, the first open-domain Persian reasoning QA benchmark, containing 10,800 questions across Boolean, multiple-choice, and factoid formats, with diverse reasoning types, difficulty levels, and answer structures. The benchmark is built via a controlled LLM-based generation pipeline and validated through human evaluation. We also ensure linguistic and factual quality through multi-stage filtering, annotation, and consistency checks. We benchmark multilingual and Persian LLMs under multiple prompting strategies and show that Persian prompts and structured prompting (CoT for Boolean/multiple-choice; few-shot for factoid) improve performance. Fine-tuning further boosts results, especially for Persian-specialized models. These findings highlight how PARSE supports both fair comparison and practical model adaptation. PARSE fills a critical gap in Persian QA research and provides a strong foundation for developing and evaluating reasoning-capable LLMs in low-resource settings. 

---
# Inferential Question Answering 

**Authors**: Jamshid Mozafari, Hamed Zamani, Guido Zuccon, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2602.01239)  

**Abstract**: Despite extensive research on a wide range of question answering (QA) systems, most existing work focuses on answer containment-i.e., assuming that answers can be directly extracted and/or generated from documents in the corpus. However, some questions require inference, i.e., deriving answers that are not explicitly stated but can be inferred from the available information. We introduce Inferential QA -- a new task that challenges models to infer answers from answer-supporting passages which provide only clues. To study this problem, we construct QUIT (QUestions requiring Inference from Texts) dataset, comprising 7,401 questions and 2.4M passages built from high-convergence human- and machine-authored hints, labeled across three relevance levels using LLM-based answerability and human verification. Through comprehensive evaluation of retrievers, rerankers, and LLM-based readers, we show that methods effective on traditional QA tasks struggle in inferential QA: retrievers underperform, rerankers offer limited gains, and fine-tuning provides inconsistent improvements. Even reasoning-oriented LLMs fail to outperform smaller general-purpose models. These findings reveal that current QA pipelines are not yet ready for inference-based reasoning. Inferential QA thus establishes a new class of QA tasks that move towards understanding and reasoning from indirect textual evidence. 

---
# Domain-Adaptive and Scalable Dense Retrieval for Content-Based Recommendation 

**Authors**: Mritunjay Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2602.00899)  

**Abstract**: E-commerce recommendation and search commonly rely on sparse keyword matching (e.g., BM25), which breaks down under vocabulary mismatch when user intent has limited lexical overlap with product metadata. We cast content-based recommendation as recommendation-as-retrieval: given a natural-language intent signal (a query or review), retrieve the top-K most relevant items from a large catalog via semantic similarity.
We present a scalable dense retrieval system based on a two-tower bi-encoder, fine-tuned on the Amazon Reviews 2023 (Fashion) subset using supervised contrastive learning with Multiple Negatives Ranking Loss. We construct training pairs from review text (as a query proxy) and item metadata (as the positive document) and fine-tune on 50,000 sampled interactions with a maximum sequence length of 500 tokens.
For efficient serving, we combine FAISS HNSW indexing with an ONNX Runtime inference pipeline using INT8 dynamic quantization. On a review-to-title benchmark over 826,402 catalog items, our approach improves Recall@10 from 0.26 (BM25) to 0.66, while meeting practical latency and model-size constraints: 6.1 ms median CPU inference latency (batch size 1) and a 4x reduction in model size.
Overall, we provide an end-to-end, reproducible blueprint for taking domain-adapted dense retrieval from offline training to CPU-efficient serving at catalog scale. 

---
# Unifying Adversarial Robustness and Training Across Text Scoring Models 

**Authors**: Manveer Singh Tamber, Hosna Oyarhoseini, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.00857)  

**Abstract**: Research on adversarial robustness in language models is currently fragmented across applications and attacks, obscuring shared vulnerabilities. In this work, we propose unifying the study of adversarial robustness in text scoring models spanning dense retrievers, rerankers, and reward models. This motivates adapting both attacks and adversarial training methods across model roles. Unlike open-ended generation, text scoring failures are directly testable: an attack succeeds when an irrelevant or rejected text outscores a relevant or chosen one. Using this principled lens of text scoring, we demonstrate that current adversarial training formulations for language models are often short-sighted, failing to effectively generalize across attacks. To address this, we introduce multiple adversarial training methods for text scoring models and show that combining complementary training methods can yield strong robustness while also improving task effectiveness. We also highlight the practical value of our approach for RLHF, showing that our adversarially trained reward models mitigate reward hacking and support the training of better-aligned LLMs. We provide our code and models for further study. 

---
# SpeechLess: Micro-utterance with Personalized Spatial Memory-aware Assistant in Everyday Augmented Reality 

**Authors**: Yoonsang Kim, Devshree Jadeja, Divyansh Pradhan, Yalong Yang, Arie Kaufman  

**Link**: [PDF](https://arxiv.org/pdf/2602.00793)  

**Abstract**: Speaking aloud to a wearable AR assistant in public can be socially awkward, and re-articulating the same requests every day creates unnecessary effort. We present SpeechLess, a wearable AR assistant that introduces a speech-based intent granularity control paradigm grounded in personalized spatial memory. SpeechLess helps users "speak less," while still obtaining the information they need, and supports gradual explicitation of intent when more complex expression is required. SpeechLess binds prior interactions to multimodal personal context-space, time, activity, and referents-to form spatial memories, and leverages them to extrapolate missing intent dimensions from under-specified user queries. This enables users to dynamically adjust how explicitly they express their informational needs, from full-utterance to micro/zero-utterance interaction. We motivate our design through a week-long formative study using a commercial smart glasses platform, revealing discomfort with public voice use, frustration with repetitive speech, and hardware constraints. Building on these insights, we design SpeechLess, and evaluate it through controlled lab and in-the-wild studies. Our results indicate that regulated speech-based interaction, can improve everyday information access, reduce articulation effort, and support socially acceptable use without substantially degrading perceived usability or intent resolution accuracy across diverse everyday environments. 

---
# Temporal Leakage in Search-Engine Date-Filtered Web Retrieval: A Case Study from Retrospective Forecasting 

**Authors**: Ali El Lahib, Ying-Jieh Xia, Zehan Li, Yuxuan Wang, Xinyu Pi  

**Link**: [PDF](https://arxiv.org/pdf/2602.00758)  

**Abstract**: Search-engine date filters are widely used to enforce pre-cutoff retrieval in retrospective evaluations of search-augmented forecasters. We show this approach is unreliable: auditing Google Search with a before: filter, 71% of questions return at least one page containing strong post-cutoff leakage, and for 41%, at least one page directly reveals the answer. Using a large language model (LLM), gpt-oss-120b, to forecast with these leaky documents, we demonstrate an inflated prediction accuracy (Brier score 0.108 vs. 0.242 with leak-free documents). We characterize common leakage mechanisms, including updated articles, related-content modules, unreliable metadata/timestamps, and absence-based signals, and argue that date-restricted search is insufficient for temporal evaluation. We recommend stronger retrieval safeguards or evaluation on frozen, time-stamped web snapshots to ensure credible retrospective forecasting. 

---
# From Prompt to Graph: Comparing LLM-Based Information Extraction Strategies in Domain-Specific Ontology Development 

**Authors**: Xuan Liu, Ziyu Li, Mu He, Ziyang Ma, Xiaoxu Wu, Gizem Yilmaz, Yiyuan Xia, Bingbing Li, He Tan, Jerry Ying Hsi Fuh, Wen Feng Lu, Anders E.W. Jarfors, Per Jansson  

**Link**: [PDF](https://arxiv.org/pdf/2602.00699)  

**Abstract**: Ontologies are essential for structuring domain knowledge, improving accessibility, sharing, and reuse. However, traditional ontology construction relies on manual annotation and conventional natural language processing (NLP) techniques, making the process labour-intensive and costly, especially in specialised fields like casting manufacturing. The rise of Large Language Models (LLMs) offers new possibilities for automating knowledge extraction. This study investigates three LLM-based approaches, including pre-trained LLM-driven method, in-context learning (ICL) method and fine-tuning method to extract terms and relations from domain-specific texts using limited data. We compare their performances and use the best-performing method to build a casting ontology that validated by domian expert. 

---
# Audio-to-Image Bird Species Retrieval without Audio-Image Pairs via Text Distillation 

**Authors**: Ilyass Moummad, Marius Miron, Lukas Rauch, David Robinson, Alexis Joly, Olivier Pietquin, Emmanuel Chemla, Matthieu Geist  

**Link**: [PDF](https://arxiv.org/pdf/2602.00681)  

**Abstract**: Audio-to-image retrieval offers an interpretable alternative to audio-only classification for bioacoustic species recognition, but learning aligned audio-image representations is challenging due to the scarcity of paired audio-image data. We propose a simple and data-efficient approach that enables audio-to-image retrieval without any audio-image supervision. Our proposed method uses text as a semantic intermediary: we distill the text embedding space of a pretrained image-text model (BioCLIP-2), which encodes rich visual and taxonomic structure, into a pretrained audio-text model (BioLingual) by fine-tuning its audio encoder with a contrastive objective. This distillation transfers visually grounded semantics into the audio representation, inducing emergent alignment between audio and image embeddings without using images during training. We evaluate the resulting model on multiple bioacoustic benchmarks. The distilled audio encoder preserves audio discriminative power while substantially improving audio-text alignment on focal recordings and soundscape datasets. Most importantly, on the SSW60 benchmark, the proposed approach achieves strong audio-to-image retrieval performance exceeding baselines based on zero-shot model combinations or learned mappings between text embeddings, despite not training on paired audio-image data. These results demonstrate that indirect semantic transfer through text is sufficient to induce meaningful audio-image alignment, providing a practical solution for visually grounded species recognition in data-scarce bioacoustic settings. 

---
# First Steps, Lasting Impact: Platform-Aware Forensics for the Next Generation of Analysts 

**Authors**: Vinayak Jain, Sneha Sudhakaran, Saranyan Senthivel  

**Link**: [PDF](https://arxiv.org/pdf/2602.00160)  

**Abstract**: The reliability of cyber forensic evidence acquisition is strongly influenced by the underlying operating systems, Windows, macOS, and Linux - due to inherent variations in file system structures, encryption protocols, and forensic tool compatibility. Disk forensics, one of the most widely used techniques in digital investigations, faces distinct obstacles on each platform. Windows, with its predominantly NTFS and FAT file systems, typically supports reliable disk imaging and analysis through established tools such as FTK Imager and Autopsy/Sleuth Kit. However, encryption features frequently pose challenges to evidence acquisition. Conversely, Linux environments, which rely on file systems like ext4 and XFS, generally offer greater transparency, yet the transient nature of log retention often complicates forensic analysis. In instances where anti-forensic strategies such as encryption and compression render traditional disk forensics insufficient, memory forensics becomes crucial. While memory forensic methodologies demonstrate robustness across Windows and Linux platforms forms through frameworks like Volatility, platform-specific difficulties persist. Memory analysis on Linux systems benefits from tools like LiME, snapshot utilities, and dd for memory acquisition; nevertheless, live memory acquisition on Linux can still present challenges. This research systematically assesses both disk and memory forensic acquisition techniques across samples representing Windows and Linux systems. By identifying effective combinations of forensic tools and configurations tailored to each operating system, the study aims to improve the accuracy and reliability of evidence collection. It further evaluates current forensic tools and highlights a persistent gap: consistently assuring forensic input reliability and footprint integrity. 

---
# OGD4All: A Framework for Accessible Interaction with Geospatial Open Government Data Based on Large Language Models 

**Authors**: Michael Siebenmann, Javier Argota Sánchez-Vaquerizo, Stefan Arisona, Krystian Samp, Luis Gisler, Dirk Helbing  

**Link**: [PDF](https://arxiv.org/pdf/2602.00012)  

**Abstract**: We present OGD4All, a transparent, auditable, and reproducible framework based on Large Language Models (LLMs) to enhance citizens' interaction with geospatial Open Government Data (OGD). The system combines semantic data retrieval, agentic reasoning for iterative code generation, and secure sandboxed execution that produces verifiable multimodal outputs. Evaluated on a 199-question benchmark covering both factual and unanswerable questions, across 430 City-of-Zurich datasets and 11 LLMs, OGD4All reaches 98% analytical correctness and 94% recall while reliably rejecting questions unsupported by available data, which minimizes hallucination risks. Statistical robustness tests, as well as expert feedback, show reliability and social relevance. The proposed approach shows how LLMs can provide explainable, multimodal access to public data, advancing trustworthy AI for open governance. 

---
# Unlocking Electronic Health Records: A Hybrid Graph RAG Approach to Safe Clinical AI for Patient QA 

**Authors**: Samuel Thio, Matthew Lewis, Spiros Denaxas, Richard JB Dobson  

**Link**: [PDF](https://arxiv.org/pdf/2602.00009)  

**Abstract**: Electronic health record (EHR) systems present clinicians with vast repositories of clinical information, creating a significant cognitive burden where critical details are easily overlooked. While Large Language Models (LLMs) offer transformative potential for data processing, they face significant limitations in clinical settings, particularly regarding context grounding and hallucinations. Current solutions typically isolate retrieval methods focusing either on structured data (SQL/Cypher) or unstructured semantic search but fail to integrate both simultaneously. This work presents MediGRAF (Medical Graph Retrieval Augmented Framework), a novel hybrid Graph RAG system that bridges this gap. By uniquely combining Neo4j Text2Cypher capabilities for structured relationship traversal with vector embeddings for unstructured narrative retrieval, MediGRAF enables natural language querying of the complete patient journey. Using 10 patients from the MIMIC-IV dataset (generating 5,973 nodes and 5,963 relationships), we generated enough nodes and data for patient level question answering (QA), and we evaluated this architecture across varying query complexities. The system demonstrated 100\% recall for factual queries which means all relevant information was retrieved and in the output, while complex inference tasks achieved a mean expert quality score of 4.25/5 with zero safety violations. These results demonstrate that hybrid graph-grounding significantly advances clinical information retrieval, offering a safer, more comprehensive alternative to standard LLM deployments. 

---
# PPoGA: Predictive Plan-on-Graph with Action for Knowledge Graph Question Answering 

**Authors**: MinGyu Jeon, SuWan Cho, JaeYoung Shu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00007)  

**Abstract**: Large Language Models (LLMs) augmented with Knowledge Graphs (KGs) have advanced complex question answering, yet they often remain susceptible to failure when their initial high-level reasoning plan is flawed. This limitation, analogous to cognitive functional fixedness, prevents agents from restructuring their approach, leading them to pursue unworkable solutions. To address this, we propose PPoGA (Predictive Plan-on-Graph with Action), a novel KGQA framework inspired by human cognitive control and problem-solving. PPoGA incorporates a Planner-Executor architecture to separate high-level strategy from low-level execution and leverages a Predictive Processing mechanism to anticipate outcomes. The core innovation of our work is a self-correction mechanism that empowers the agent to perform not only Path Correction for local execution errors but also Plan Correction by identifying, discarding, and reformulating the entire plan when it proves ineffective. We conduct extensive experiments on three challenging multi-hop KGQA benchmarks: GrailQA, CWQ, and WebQSP. The results demonstrate that PPoGA achieves state-of-the-art performance, significantly outperforming existing methods. Our work highlights the critical importance of metacognitive abilities like problem restructuring for building more robust and flexible AI reasoning systems. 

---
# Culinary Crossroads: A RAG Framework for Enhancing Diversity in Cross-Cultural Recipe Adaptation 

**Authors**: Tianyi Hu, Andrea Morales-Garzón, Jingyi Zheng, Maria Maistro, Daniel Hershcovich  

**Link**: [PDF](https://arxiv.org/pdf/2507.21934)  

**Abstract**: In cross-cultural recipe adaptation, the goal is not only to ensure cultural appropriateness and retain the original dish's essence, but also to provide diverse options for various dietary needs and preferences. Retrieval Augmented Generation (RAG) is a promising approach, combining the retrieval of real recipes from the target cuisine for cultural adaptability with large language models (LLMs) for relevance. However, it remains unclear whether RAG can generate diverse adaptation results. Our analysis shows that RAG tends to overly rely on a limited portion of the context across generations, failing to produce diverse outputs even when provided with varied contextual inputs. This reveals a key limitation of RAG in creative tasks with multiple valid answers: it fails to leverage contextual diversity for generating varied responses. To address this issue, we propose CARRIAGE, a plug-and-play RAG framework for cross-cultural recipe adaptation that enhances diversity in both retrieval and context organization. To our knowledge, this is the first RAG framework that explicitly aims to generate highly diverse outputs to accommodate multiple user preferences. Our experiments show that CARRIAGE achieves Pareto efficiency in terms of diversity and quality of recipe adaptation compared to closed-book LLMs. 

---
