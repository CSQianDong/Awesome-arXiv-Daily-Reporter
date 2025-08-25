# ORCA: Mitigating Over-Reliance for Multi-Task Dwell Time Prediction with Causal Decoupling 

**Authors**: Huishi Luo, Fuzhen Zhuang, Yongchun Zhu, Yiqing Wu, Bo Kang, Ruobing Xie, Feng Xia, Deqing Wang, Jin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.16573)  

**Abstract**: Dwell time (DT) is a critical post-click metric for evaluating user preference in recommender systems, complementing the traditional click-through rate (CTR). Although multi-task learning is widely adopted to jointly optimize DT and CTR, we observe that multi-task models systematically collapse their DT predictions to the shortest and longest bins, under-predicting the moderate durations. We attribute this moderate-duration bin under-representation to over-reliance on the CTR-DT spurious correlation, and propose ORCA to address it with causal-decoupling. Specifically, ORCA explicitly models and subtracts CTR's negative transfer while preserving its positive transfer. We further introduce (i) feature-level counterfactual intervention, and (ii) a task-interaction module with instance inverse-weighting, weakening CTR-mediated effect and restoring direct DT semantics. ORCA is model-agnostic and easy to deploy. Experiments show an average 10.6% lift in DT metrics without harming CTR. Code is available at this https URL. 

---
# Enhanced NIRMAL Optimizer With Damped Nesterov Acceleration: A Comparative Analysis 

**Authors**: Nirmal Gaud, Prasad Krishna Murthy, Mostaque Md. Morshedur Hassan, Abhijit Ganguly, Vinay Mali, Ms Lalita Bhagwat Randive, Abhaypratap Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.16550)  

**Abstract**: This study introduces the Enhanced NIRMAL (Novel Integrated Robust Multi-Adaptation Learning with Damped Nesterov Acceleration) optimizer, an improved version of the original NIRMAL optimizer. By incorporating an $(\alpha, r)$-damped Nesterov acceleration mechanism, Enhanced NIRMAL improves convergence stability while retaining chess-inspired strategies of gradient descent, momentum, stochastic perturbations, adaptive learning rates, and non-linear transformations.
We evaluate Enhanced NIRMAL against Adam, SGD with Momentum, Nesterov, and the original NIRMAL on four benchmark image classification datasets: MNIST, FashionMNIST, CIFAR-10, and CIFAR-100, using tailored convolutional neural network (CNN) architectures.
Enhanced NIRMAL achieves a test accuracy of 46.06\% and the lowest test loss (1.960435) on CIFAR-100, surpassing the original NIRMAL (44.34\% accuracy) and closely rivaling SGD with Momentum (46.43\% accuracy). These results underscore Enhanced NIRMAL's superior generalization and stability, particularly on complex datasets. 

---
# A Node-Aware Dynamic Quantization Approach for Graph Collaborative Filtering 

**Authors**: Lin Li, Chunyang Li, Yu Yin, Xiaohui Tao, Jianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16516)  

**Abstract**: In the realm of collaborative filtering recommendation systems, Graph Neural Networks (GNNs) have demonstrated remarkable performance but face significant challenges in deployment on resource-constrained edge devices due to their high embedding parameter requirements and computational costs. Using common quantization method directly on node embeddings may overlooks their graph based structure, causing error accumulation during message passing and degrading the quality of quantized this http URL address this, we propose Graph based Node-Aware Dynamic Quantization training for collaborative filtering (GNAQ), a novel quantization approach that leverages graph structural information to enhance the balance between efficiency and accuracy of GNNs for Top-K recommendation. GNAQ introduces a node-aware dynamic quantization strategy that adapts quantization scales to individual node embeddings by incorporating graph interaction relationships. Specifically, it initializes quantization intervals based on node-wise feature distributions and dynamically refines them through message passing in GNN layers. This approach mitigates information loss caused by fixed quantization scales and captures hierarchical semantic features in user-item interaction graphs. Additionally, GNAQ employs graph relation-aware gradient estimation to replace traditional straight-through estimators, ensuring more accurate gradient propagation during training. Extensive experiments on four real-world datasets demonstrate that GNAQ outperforms state-of-the-art quantization methods, including BiGeaR and N2UQ, by achieving average improvement in 27.8\% Recall@10 and 17.6\% NDCG@10 under 2-bit quantization. In particular, GNAQ is capable of maintaining the performance of full-precision models while reducing their model sizes by 8 to 12 times; in addition, the training time is twice as fast compared to quantization baseline methods. 

---
# OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval 

**Authors**: Yu Liu, Yanbing Liu, Fangfang Yuan, Cong Cao, Youbang Sun, Kun Peng, WeiZhuo Chen, Jianjun Li, Zhiyuan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.16438)  

**Abstract**: Recent advances in large language models (LLMs) and dense retrievers have driven significant progress in retrieval-augmented generation (RAG). However, existing approaches face significant challenges in complex reasoning-oriented multi-hop retrieval tasks: 1) Ineffective reasoning-oriented planning: Prior methods struggle to generate robust multi-step plans for complex queries, as rule-based decomposers perform poorly on out-of-template questions. 2) Suboptimal reasoning-driven retrieval: Related methods employ limited query reformulation, leading to iterative retrieval loops that often fail to locate golden documents. 3) Insufficient reasoning-guided filtering: Prevailing methods lack the fine-grained reasoning to effectively filter salient information from noisy results, hindering utilization of retrieved knowledge. Fundamentally, these limitations all stem from the weak coupling between retrieval and reasoning in current RAG architectures. We introduce the Orchestrated Planner-Executor Reasoning Architecture (OPERA), a novel reasoning-driven retrieval framework. OPERA's Goal Planning Module (GPM) decomposes questions into sub-goals, which are executed by a Reason-Execute Module (REM) with specialized components for precise reasoning and effective retrieval. To train OPERA, we propose Multi-Agents Progressive Group Relative Policy Optimization (MAPGRPO), a novel variant of GRPO. Experiments on complex multi-hop benchmarks show OPERA's superior performance, validating both the MAPGRPO method and OPERA's design. Code is available at this https URL. 

---
# Modeling User Preferences as Distributions for Optimal Transport-based Cross-domain Recommendation under Non-overlapping Settings 

**Authors**: Ziyin Xiao, Toyotaro Suzumura  

**Link**: [PDF](https://arxiv.org/pdf/2508.16210)  

**Abstract**: Cross-Domain Recommender (CDR) systems aim to transfer knowledge from dense to sparse domains, alleviating data sparsity and cold-start issues in single-domain recommendation. While many methods assume overlapping users or items to connect domains, this is often unrealistic in real-world settings. Thus, non-overlapping CDR systems, which require no shared users or items, are needed.
However, non-overlapping CDR is challenging due to: (1) the absence of overlap preventing direct bridges between domains, and (2) large distributional discrepancies degrading transfer performance. Moreover, most recommenders represent user preferences as discrete vectors, failing to capture their fine-grained, multi-faceted nature.
We propose DUP-OT (Distributional User Preferences with Optimal Transport), a framework for non-overlapping CDR. DUP-OT has three stages: (1) Shared Preprocessing, where review-based embeddings and an autoencoder encode users and items from both domains; (2) User GMM Weight Learning, which models user preferences as Gaussian mixtures with learned weights; and (3) Cross-domain Rating Prediction, where optimal transport aligns Gaussian components across domains, enabling preference transfer from source to target.
Experiments on Amazon review datasets show that DUP-OT effectively mitigates domain discrepancy and outperforms state-of-the-art baselines under the non-overlapping CDR setting. 

---
# EGRA:Toward Enhanced Behavior Graphs and Representation Alignment for Multimodal Recommendation 

**Authors**: Xiaoxiong Zhang, Xin Zhou, Zhiwei Zeng, Yongjie Wang, Dusit Niyato, Zhiqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16170)  

**Abstract**: MultiModal Recommendation (MMR) systems have emerged as a promising solution for improving recommendation quality by leveraging rich item-side modality information, prompting a surge of diverse methods. Despite these advances, existing methods still face two critical limitations. First, they use raw modality features to construct item-item links for enriching the behavior graph, while giving limited attention to balancing collaborative and modality-aware semantics or mitigating modality noise in the process. Second, they use a uniform alignment weight across all entities and also maintain a fixed alignment strength throughout training, limiting the effectiveness of modality-behavior alignment. To address these challenges, we propose EGRA. First, instead of relying on raw modality features, it alleviates sparsity by incorporating into the behavior graph an item-item graph built from representations generated by a pretrained MMR model. This enables the graph to capture both collaborative patterns and modality aware similarities with enhanced robustness against modality noise. Moreover, it introduces a novel bi-level dynamic alignment weighting mechanism to improve modality-behavior representation alignment, which dynamically assigns alignment strength across entities according to their alignment degree, while gradually increasing the overall alignment intensity throughout training. Extensive experiments on five datasets show that EGRA significantly outperforms recent methods, confirming its effectiveness. 

---
# Cross-Modal Prototype Augmentation and Dual-Grained Prompt Learning for Social Media Popularity Prediction 

**Authors**: Ao Zhou, Mingsheng Tu, Luping Wang, Tenghao Sun, Zifeng Cheng, Yafeng Yin, Zhiwei Jiang, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16147)  

**Abstract**: Social Media Popularity Prediction is a complex multimodal task that requires effective integration of images, text, and structured information. However, current approaches suffer from inadequate visual-textual alignment and fail to capture the inherent cross-content correlations and hierarchical patterns in social media data. To overcome these limitations, we establish a multi-class framework , introducing hierarchical prototypes for structural enhancement and contrastive learning for improved vision-text alignment. Furthermore, we propose a feature-enhanced framework integrating dual-grained prompt learning and cross-modal attention mechanisms, achieving precise multimodal representation through fine-grained category modeling. Experimental results demonstrate state-of-the-art performance on benchmark metrics, establishing new reference standards for multimodal social media analysis. 

---
# Spacetime-GR: A Spacetime-Aware Generative Model for Large Scale Online POI Recommendation 

**Authors**: Haitao Lin, Zhen Yang, Jiawei Xue, Ziji Zhang, Luzhu Wang, Yikun Gu, Yao Xu, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.16126)  

**Abstract**: Building upon the strong sequence modeling capability, Generative Recommendation (GR) has gradually assumed a dominant position in the application of recommendation tasks (e.g., video and product recommendation). However, the application of Generative Recommendation in Point-of-Interest (POI) recommendation, where user preferences are significantly affected by spatiotemporal variations, remains a challenging open problem. In this paper, we propose Spacetime-GR, the first spacetime-aware generative model for large-scale online POI recommendation. It extends the strong sequence modeling ability of generative models by incorporating flexible spatiotemporal information encoding. Specifically, we first introduce a geographic-aware hierarchical POI indexing strategy to address the challenge of large vocabulary modeling. Subsequently, a novel spatiotemporal encoding module is introduced to seamlessly incorporate spatiotemporal context into user action sequences, thereby enhancing the model's sensitivity to spatiotemporal variations. Furthermore, we incorporate multimodal POI embeddings to enrich the semantic understanding of each POI. Finally, to facilitate practical deployment, we develop a set of post-training adaptation strategies after sufficient pre-training on action sequences. These strategies enable Spacetime-GR to generate outputs in multiple formats (i.e., embeddings, ranking scores and POI candidates) and support a wide range of downstream application scenarios (i.e., ranking and end-to-end recommendation). We evaluate the proposed model on both public benchmark datasets and large-scale industrial datasets, demonstrating its superior performance over existing methods in terms of POI recommendation accuracy and ranking quality. Furthermore, the model is the first generative model deployed in online POI recommendation services that scale to hundreds of millions of POIs and users. 

---
# Similarity-Based Supervised User Session Segmentation Method for Behavior Logs 

**Authors**: Yongzhi Jin, Kazushi Okamoto, Kei Harada, Atsushi Shibata, Koki Karube  

**Link**: [PDF](https://arxiv.org/pdf/2508.16106)  

**Abstract**: In information recommendation, a session refers to a sequence of user actions within a specific time frame. Session-based recommender systems aim to capture short-term preferences and generate relevant recommendations. However, user interests may shift even within a session, making appropriate segmentation essential for modeling dynamic behaviors. In this study, we propose a supervised session segmentation method based on similarity features derived from action embeddings and attributes. We compute the similarity scores between items within a fixed-size window around each candidate segmentation point, using four types of features: item co-occurrence embeddings, text embeddings of titles and brands, and price. These features are used to train supervised classifiers (LightGBM, XGBoost, CatBoost, support vector machine, and logistic regression) to predict the session boundaries. We construct a manually annotated dataset from real user browsing histories and evaluate the segmentation performance using F1-score, area under the precision-recall curve (PR-AUC), and area under the receiver operating characteristic curve. The LightGBM model achieves the best performance, with an F1-score of 0.806 and a PR-AUC of 0.831. These results demonstrate the effectiveness of the proposed method for session segmentation and its potential to capture dynamic user behaviors. 

---
# Estimating the Effective Topics of Articles and journals Abstract Using LDA And K-Means Clustering Algorithm 

**Authors**: Shadikur Rahman, Umme Ayman Koana, Aras M. Ismael, Karmand Hussein Abdalla  

**Link**: [PDF](https://arxiv.org/pdf/2508.16046)  

**Abstract**: Analyzing journals and articles abstract text or documents using topic modelling and text clustering has become a modern solution for the increasing number of text documents. Topic modelling and text clustering are both intensely involved tasks that can benefit one another. Text clustering and topic modelling algorithms are used to maintain massive amounts of text documents. In this study, we have used LDA, K-Means cluster and also lexical database WordNet for keyphrases extraction in our text documents. K-Means cluster and LDA algorithms achieve the most reliable performance for keyphrase extraction in our text documents. This study will help the researcher to make a search string based on journals and articles by avoiding misunderstandings. 

---
# LLM-Based Agents for Competitive Landscape Mapping in Drug Asset Due Diligence 

**Authors**: Alisa Vinogradova, Vlad Vinogradov, Dmitrii Radkevich, Ilya Yasny, Dmitry Kobyzev, Ivan Izmailov, Katsiaryna Yanchanka, Andrey Doronichev  

**Link**: [PDF](https://arxiv.org/pdf/2508.16571)  

**Abstract**: In this paper, we describe and benchmark a competitor-discovery component used within an agentic AI system for fast drug asset due diligence. A competitor-discovery AI agent, given an indication, retrieves all drugs comprising the competitive landscape of that indication and extracts canonical attributes for these drugs. The competitor definition is investor-specific, and data is paywalled/licensed, fragmented across registries, ontology-mismatched by indication, alias-heavy for drug names, multimodal, and rapidly changing. Although considered the best tool for this problem, the current LLM-based AI systems aren't capable of reliably retrieving all competing drug names, and there is no accepted public benchmark for this task. To address the lack of evaluation, we use LLM-based agents to transform five years of multi-modal, unstructured diligence memos from a private biotech VC fund into a structured evaluation corpus mapping indications to competitor drugs with normalized attributes. We also introduce a competitor validating LLM-as-a-judge agent that filters out false positives from the list of predicted competitors to maximize precision and suppress hallucinations. On this benchmark, our competitor-discovery agent achieves 83% recall, exceeding OpenAI Deep Research (65%) and Perplexity Labs (60%). The system is deployed in production with enterprise users; in a case study with a biotech VC investment fund, analyst turnaround time dropped from 2.5 days to $\sim$3 hours ($\sim$20x) for the competitive analysis. 

---
# LLM-as-classifier: Semi-Supervised, Iterative Framework for Hierarchical Text Classification using Large Language Models 

**Authors**: Doohee You, Andy Parisi, Zach Vander Velden, Lara Dantas Inojosa  

**Link**: [PDF](https://arxiv.org/pdf/2508.16478)  

**Abstract**: The advent of Large Language Models (LLMs) has provided unprecedented capabilities for analyzing unstructured text data. However, deploying these models as reliable, robust, and scalable classifiers in production environments presents significant methodological challenges. Standard fine-tuning approaches can be resource-intensive and often struggle with the dynamic nature of real-world data distributions, which is common in the industry. In this paper, we propose a comprehensive, semi-supervised framework that leverages the zero- and few-shot capabilities of LLMs for building hierarchical text classifiers as a framework for a solution to these industry-wide challenges. Our methodology emphasizes an iterative, human-in-the-loop process that begins with domain knowledge elicitation and progresses through prompt refinement, hierarchical expansion, and multi-faceted validation. We introduce techniques for assessing and mitigating sequence-based biases and outline a protocol for continuous monitoring and adaptation. This framework is designed to bridge the gap between the raw power of LLMs and the practical need for accurate, interpretable, and maintainable classification systems in industry applications. 

---
# MizanQA: Benchmarking Large Language Models on Moroccan Legal Question Answering 

**Authors**: Adil Bahaj, Mounir Ghogho  

**Link**: [PDF](https://arxiv.org/pdf/2508.16357)  

**Abstract**: The rapid advancement of large language models (LLMs) has significantly propelled progress in natural language processing (NLP). However, their effectiveness in specialized, low-resource domains-such as Arabic legal contexts-remains limited. This paper introduces MizanQA (pronounced Mizan, meaning "scale" in Arabic, a universal symbol of justice), a benchmark designed to evaluate LLMs on Moroccan legal question answering (QA) tasks, characterised by rich linguistic and legal complexity. The dataset draws on Modern Standard Arabic, Islamic Maliki jurisprudence, Moroccan customary law, and French legal influences. Comprising over 1,700 multiple-choice questions, including multi-answer formats, MizanQA captures the nuances of authentic legal reasoning. Benchmarking experiments with multilingual and Arabic-focused LLMs reveal substantial performance gaps, highlighting the need for tailored evaluation metrics and culturally grounded, domain-specific LLM development. 

---
# Attribute Filtering in Approximate Nearest Neighbor Search: An In-depth Experimental Study 

**Authors**: Mocheng Li, Xiao Yan, Baotong Lu, Yue Zhang, James Cheng, Chenhao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.16263)  

**Abstract**: With the growing integration of structured and unstructured data, new methods have emerged for performing similarity searches on vectors while honoring structured attribute constraints, i.e., a process known as Filtering Approximate Nearest Neighbor (Filtering ANN) search. Since many of these algorithms have only appeared in recent years and are designed to work with a variety of base indexing methods and filtering strategies, there is a pressing need for a unified analysis that identifies their core techniques and enables meaningful comparisons.
In this work, we present a unified Filtering ANN search interface that encompasses the latest algorithms and evaluate them extensively from multiple perspectives. First, we propose a comprehensive taxonomy of existing Filtering ANN algorithms based on attribute types and filtering strategies. Next, we analyze their key components, i.e., index structures, pruning strategies, and entry point selection, to elucidate design differences and tradeoffs. We then conduct a broad experimental evaluation on 10 algorithms and 12 methods across 4 datasets (each with up to 10 million items), incorporating both synthetic and real attributes and covering selectivity levels from 0.1% to 100%. Finally, an in-depth component analysis reveals the influence of pruning, entry point selection, and edge filtering costs on overall performance. Based on our findings, we summarize the strengths and limitations of each approach, provide practical guidelines for selecting appropriate methods, and suggest promising directions for future research. Our code is available at: this https URL. 

---
# Extending FKG.in: Towards a Food Claim Traceability Network 

**Authors**: Saransh Kumar Gupta, Rizwan Gulzar Mir, Lipika Dey, Partha Pratim Das, Anirban Sen, Ramesh Jain  

**Link**: [PDF](https://arxiv.org/pdf/2508.16117)  

**Abstract**: The global food landscape is rife with scientific, cultural, and commercial claims about what foods are, what they do, what they should not do, or should not do. These range from rigorously studied health benefits (probiotics improve gut health) and misrepresentations (soaked almonds make one smarter) to vague promises (superfoods boost immunity) and culturally rooted beliefs (cold foods cause coughs). Despite their widespread influence, the infrastructure for tracing, verifying, and contextualizing these claims remains fragmented and underdeveloped. In this paper, we propose a Food Claim-Traceability Network (FCN) as an extension of this http URL, a knowledge graph of Indian food that we have been incrementally building. We also present the ontology design and the semi-automated knowledge curation workflow that we used to develop a proof of concept of this http URL-FCN using Reddit data and Large Language Models. FCN integrates curated data inputs, structured schemas, and provenance-aware pipelines for food-related claim extraction and validation. While directly linked to the Indian food knowledge graph as an application, our methodology remains application-agnostic and adaptable to other geographic, culinary, or regulatory settings. By modeling food claims and their traceability in a structured, verifiable, and explainable way, we aim to contribute to more transparent and accountable food knowledge ecosystems, supporting researchers, policymakers, and most importantly, everyday consumers in navigating a world saturated with dietary assertions. 

---
# Evaluating Structured Decoding for Text-to-Table Generation: Evidence from Three Datasets 

**Authors**: Julian Oestreich, Lydia Müller  

**Link**: [PDF](https://arxiv.org/pdf/2508.15910)  

**Abstract**: We present a comprehensive evaluation of structured decoding for text-to-table generation with large language models (LLMs). While previous work has primarily focused on unconstrained generation of tables, the impact of enforcing structural constraints during generation remains underexplored. We systematically compare schema-guided (structured) decoding to standard one-shot prompting across three diverse benchmarks - E2E, Rotowire, and Livesum - using open-source LLMs of up to 32B parameters, assessing the performance of table generation approaches in resource-constrained settings. Our experiments cover a wide range of evaluation metrics at cell, row, and table levels. Results demonstrate that structured decoding significantly enhances the validity and alignment of generated tables, particularly in scenarios demanding precise numerical alignment (Rotowire), but may degrade performance in contexts involving densely packed textual information (E2E) or extensive aggregation over lengthy texts (Livesum). We further analyze the suitability of different evaluation metrics and discuss the influence of model size. 

---
# Annif at the GermEval-2025 LLMs4Subjects Task: Traditional XMTC Augmented by Efficient LLMs 

**Authors**: Osma Suominen, Juho Inkinen, Mona Lehtinen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15877)  

**Abstract**: This paper presents the Annif system in the LLMs4Subjects shared task (Subtask 2) at GermEval-2025. The task required creating subject predictions for bibliographic records using large language models, with a special focus on computational efficiency. Our system, based on the Annif automated subject indexing toolkit, refines our previous system from the first LLMs4Subjects shared task, which produced excellent results. We further improved the system by using many small and efficient language models for translation and synthetic data generation and by using LLMs for ranking candidate subjects. Our system ranked 1st in the overall quantitative evaluation of and 1st in the qualitative evaluation of Subtask 2. 

---
# An Auditable Pipeline for Fuzzy Full-Text Screening in Systematic Reviews: Integrating Contrastive Semantic Highlighting and LLM Judgment 

**Authors**: Pouria Mortezaagha, Arya Rahgozar  

**Link**: [PDF](https://arxiv.org/pdf/2508.15822)  

**Abstract**: Full-text screening is the major bottleneck of systematic reviews (SRs), as decisive evidence is dispersed across long, heterogeneous documents and rarely admits static, binary rules. We present a scalable, auditable pipeline that reframes inclusion/exclusion as a fuzzy decision problem and benchmark it against statistical and crisp baselines in the context of the Population Health Modelling Consensus Reporting Network for noncommunicable diseases (POPCORN). Articles are parsed into overlapping chunks and embedded with a domain-adapted model; for each criterion (Population, Intervention, Outcome, Study Approach), we compute contrastive similarity (inclusion-exclusion cosine) and a vagueness margin, which a Mamdani fuzzy controller maps into graded inclusion degrees with dynamic thresholds in a multi-label setting. A large language model (LLM) judge adjudicates highlighted spans with tertiary labels, confidence scores, and criterion-referenced rationales; when evidence is insufficient, fuzzy membership is attenuated rather than excluded. In a pilot on an all-positive gold set (16 full texts; 3,208 chunks), the fuzzy system achieved recall of 81.3% (Population), 87.5% (Intervention), 87.5% (Outcome), and 75.0% (Study Approach), surpassing statistical (56.3-75.0%) and crisp baselines (43.8-81.3%). Strict "all-criteria" inclusion was reached for 50.0% of articles, compared to 25.0% and 12.5% under the baselines. Cross-model agreement on justifications was 98.3%, human-machine agreement 96.1%, and a pilot review showed 91% inter-rater agreement (kappa = 0.82), with screening time reduced from about 20 minutes to under 1 minute per article at significantly lower cost. These results show that fuzzy logic with contrastive highlighting and LLM adjudication yields high recall, stable rationale, and end-to-end traceability. 

---
