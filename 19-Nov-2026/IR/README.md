# Can Programs Replace LLM Judges? Executable Program Judges for Scalable and Efficient Relevance Labeling 

**Authors**: Eugene Agichtein, Charles L. A. Clarke, Kaustubh Dhole, William Stern  

**Link**: [PDF](https://arxiv.org/pdf/2611.24849)  

**Abstract**: Large language model (LLM) judges have become a common way to create large-scale relevance judgments for retrieval evaluation. However, each judgment requires a potentially expensive call to a frontier model, and the judgments are hard to interpret and may not be reproducible. In this paper, we ask whether an LLM can write a readable program for each topic that replaces its per-document judgments. To address these issues, we propose ExecJudges, a method for generating fast, inspectable and scalable judges for automatic relevance labeling. Specifically, ExecJudges uses a frontier LLM to turn assessment guidelines into executable programs, which rely only on lexical matching, standard NLP libraries like parsers, and pool statistics. An ExecJudges-L variant also uses examples of human labels if available. We evaluate ExecJudges on seven TREC collections by comparing the produced labels with trained NIST assessors, frontier LLM judges, and recently proposed per-topic classifiers. Our results show that ExecJudges agrees with NIST assessors at the level of the Claude Sonnet 5.5 LLM judges on TREC DL 2019-2021. With a small number of human labels per topic, the ExecJudges-L variant agrees with NIST more than per-topic classifiers with the same labels on all seven collections, and either exceeds the agreement of LLM judges provided the same labels or is within about 0.05 of it in Krippendorff's alpha. The labels rank the official TREC DL runs with Kendall's tau of 0.87 to 0.94, and writing the programs incurs a one-time cost of $0.17 to $0.49 per topic, with no additional costs for grading. Our results suggest that ExecJudges offers a reproducible, inspectable, efficient and cost-effective alternative to LLM judges for labeling deep result pools for system evaluation or ranking improvements. 

---
# X-split Argus: Learning Reusable History Tokens for Efficient CTR Prediction 

**Authors**: Alexander Ploshkin, Nikita Stepanov, Alexey Pismenny, Viacheslav Ovchinnikov, Nikolay Savushkin  

**Link**: [PDF](https://arxiv.org/pdf/2611.24283)  

**Abstract**: Accurate click-through rate (CTR) prediction depends on user representations that capture both persistent preferences and recent intent, yet processing extensive behavioral histories for every ranking request is costly under production compute and latency constraints. We present X-split ARGUS, a framework for ad ranking whose history compressor and online ranker are trained end to end. The compressor encodes past interactions into a compact set of tokens that are computed offline and reused across candidates, while the online ranker scores ads through cross-attention to these tokens and representations of recent events. Training combines an autoregressive loss for click feedback with distillation of candidate-specific click logits from a larger teacher that scores each candidate using the uncompressed history. The number of compressed history tokens and the length of recent history can be adjusted to fit serving compute and latency budgets. Offline evaluation shows a 0.81% relative reduction in normalized entropy (NE) compared with the production baseline. Deployment in a large-scale real-time bidding system increases online inference cost by 15% relative to the production baseline. An online A/B test against this baseline yields relative gains of 4.65% in a business metric reflecting the monetary value of conversions generated for advertisers and 2.96% in clicks. 

---
# MARS Technical Report 

**Authors**: Tong Liu, Chaoqun Hou, Longxiang Xiong, Junhong Lou, Jiaqi Zheng, Qixin Li, Zixin Zhao, Yong Duan, Xiannian Hu, Fengze Jiang, Yu Zhu, Cheng Guo, Xinyuan Qi, Zi Zhuang, Ping Liu, Lv Shao, Zongwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.24006)  

**Abstract**: Modern industrial recommender systems use multi-stage cascades--retrieval, pre-ranking, ranking, and re-ranking--to narrow millions of candidates to a personalized list. While each stage deploys sophisticated models, pipeline-level decisions on channel activation, quotas, model selection, and score fusion remain engineer-tuned and user-agnostic within each scenario.
This report presents MARS (Multi-stage Agentic Recommendation System), extending personalization from individual-stage predictions to full pipeline orchestration. A near-line agent built on fine-tuned Qwen3-0.6B generates per-user configurations--a seven-dimensional action covering pipeline mode, retrieval quotas, model selection, fusion weights, and reranker settings--from a constrained action space. Configurations are validated, cached, and consumed asynchronously; the cascade never waits for LLM inference. Schema checks, fallbacks, and a kill switch ensure serving safety.
The policy is trained in three stages: Confidence-aware SFT distills schema-valid generation from a teacher model; Format-Anchored Preference Optimization (FAPO) learns action preferences via ActionMask with posterior weighting; Group Relative Policy Optimization (GRPO) refines the policy through Sandbox replay with a frozen reward model, without exposing exploratory actions to live traffic.
Two-week A/B tests across five Taobao scenarios with over ten million daily users show positive gains on all key metrics. The largest reach +5.22% V2P and +7.34% GMV, with positive CTR lifts on both click surfaces. About 20% of requests use a lightweight Minimal mode, and deployment shows 12 ms lower mean serving latency versus production. 

---
# CONCORD-Rec: Catalog-Conditioned Modality Utility for Semantic-ID Diffusion Recommendation 

**Authors**: Haopeng Jin, Bingkang Shi, Tao Yu, Yifei Ma, Hao Liu, Hongzhu Yi, Jinwen Luo  

**Link**: [PDF](https://arxiv.org/pdf/2611.23710)  

**Abstract**: Multimodal generative recommendation with discrete semantic identifiers (SIDs) requires distinguishing historical modalities that merely alter token predictions from those that genuinely reduce catalog-level ambiguity. When audiovisual, visual, or textual streams are missing, degraded, or semantically conflicting, uniform fusion and unconstrained token gates can propagate misleading content across iterative denoising steps. We propose CONCORD-Rec, a reliability-conditioned discrete diffusion recommender supervised by counterfactual catalog-partition utility. Given a fixed tokenizer and a partially masked SID state, CONCORD-Rec evaluates out-of-fold teacher coalitions over the exact feasible catalog support and aggregates item posteriors across the target code partition at each unresolved position. Rather than distilling raw token probabilities directly, the resulting counterfactual loss differences supervise an availability-masked tokenwise gate that allocates bounded modality residuals around a behavioral denoiser while retaining an exact behavioral fallback slot. We establish theoretical fixed-support ranking stability bounds under gate perturbations and evaluate position-level and code-conditional allocation across audiovisual and image-text recommendation benchmarks under controlled missingness, signal degradation, and semantic conflict. Across 20 source-compatible protocol-dataset rows, CONCORD-Rec improves NDCG by 4%; under matched 30% conflict it improves NDCG by 14.63%. The full-vs-uniform-T=16 sweep improves quality by 2.00-3.58% while reducing p95 latency by up to 86.89%. 

---
# From Docket Minutes to Case Trajectories: Building Analysis-Ready Criminal Court Data 

**Authors**: Erfan Hosseini Sereshgi, Kory Rosen, Jaya Dhariwal, Nicholas Mattei, Aron Culotta  

**Link**: [PDF](https://arxiv.org/pdf/2611.23537)  

**Abstract**: Criminal court records are often fragmented, noisy, and siloed, making it challenging for both legal experts and community stakeholders to access and analyze court data to identify trends and provide public oversight. In this paper, we introduce a system for data ingestion, harmonization, and dashboard visualization designed for New Orleans criminal court proceedings data, aimed at both expert and lay audiences. Our system collects and harmonizes disparate data, generates case trajectories, and provides intuitive visualization techniques based on community feedback to analyze demographics, temporal trends, and specific case events. We demonstrate the system through an exploratory analysis of recent short-term trends and policy impacts within New Orleans, illustrating how it can support the investigation of anomalies, policy changes, and procedural irregularities. We also present feedback from a small formative study of community stakeholders who have helped co-design and use the system, suggesting that it may improve access to and interpretation of local court data. 

---
# PGPO: Prototype-Guided Progressive Obfuscation for Privacy-Preserving LLM-Enhanced Recommendation 

**Authors**: Shengxiang Lin, Jiajie Su, Pengyang Zhou, Xiang Chen, Xiaolin Zheng, Feng Tian, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.23236)  

**Abstract**: LLM-enhanced recommender systems encode item side information into semantic embeddings, but these embeddings may leak sensitive textual attributes through inversion attacks. Existing input-side defenses rely on trusted collection or local token perturbation, often disrupting the inter-item semantic topology needed for recommendation. To address it, we introduce relational fidelity as the key objective, requiring semantic shift for privacy while preserving relative item relations for utility. Based on this principle, we propose prototype-guided progressive obfuscation (PGPO), which learns topology-aware obfuscation anchors for high-influence prototype words and propagates them through a static semantic graph. PGPO is optimized with GRPO using individual dissimilarity, structural consistency, and semantic constraint rewards. Experiments on two benchmarks show strong inversion defense with near-upper-bound recommendation performance. The code is available at this https URL. 

---
# Variance-Optimal Counterfactual Policy Design for Sample-Efficient Online A/B Testing 

**Authors**: Shashank Gupta, Subhadeep Maji  

**Link**: [PDF](https://arxiv.org/pdf/2611.24757)  

**Abstract**: Online A/B testing is the standard approach for evaluating changes to user-facing systems, such as a new recommendation or ranking policy, before deployment. While the difference-in-means estimator of a standard A/B test is unbiased, it often suffers from high variance. When the treatment and production policies choose the same actions with similar probabilities, the resulting observations can add noise while providing little information about the difference between the policies. Consequently, A/B tests can require large amounts of traffic and long durations to detect small treatment effects. Existing variance-reduction methods mainly improve the estimator after the data-collection scheme has been fixed, and recent overlap-aware designs only optimize a single traffic split between the two policies; thereby, they cannot direct traffic to the actions on which the policies disagree.
In this paper, we jointly optimize how actions are sampled and how the resulting observations are used to estimate the treatment effect. Our contributions are two-fold. First, building on pairwise off-policy evaluation, we derive the variance-optimal action-level logging policy for estimating the average treatment effect between two policies, which has a closed-form solution for any fixed control variate. Second, we use historical logs to jointly optimize the logging policy and a control variate to further reduce variance. Our experimental results in simulated policy A/B tests show that our designs estimate the average treatment effect more accurately than existing variance reduction methods on every evaluated setting, while generally improving statistical power and reducing how often the worse policy is selected. As a result, our designs enable practitioners to reach reliable deployment decisions with considerably less online traffic. 

---
# Task-Oriented Network Ranging via Joint Graph and Schedule Design 

**Authors**: Zijian Zhang, Yuan Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.24702)  

**Abstract**: Measuring inter-node distances is the basis for network localization, whereas acquiring node positions does not require a fully connected graph of distances. Existing complete-ranging protocols for network localization require every node pair to exchange signals, which prolongs the ranging cycle. In this paper, we propose sparse signal-multiplexing network ranging (S-SM-NR), a task-oriented framework that adapts the transmit-listen schedule to the ranging task required by network localization. First, we establish the S-SM-NR framework by formulating the joint design of the task graph and the transmit-listen schedule. Then, we analyze the minimum round count required by S-SM-NR and establish a connectivity condition for network localization. Subsequently, we develop a closed-form joint design that constructs the sparse task graph and the balanced transmit-listen schedule. In particular, the joint design can achieve the minimum round count under the complete physical connectivity. Finally, simulations show that the proposed S-SM-NR scheme reduces the occupied-round count and improves the localization accuracy by shortening the ranging cycle. 

---
# ReflectRec: Cycle-Consistency Preference Verification for LLM-based Recommendation 

**Authors**: Dae Kyoo Kim, Heejung Cho, Sungmoo Hong, Eunil Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.23969)  

**Abstract**: CoT-style LLM recommenders generate a natural-language preference note as a chain-of-thought prefix to ranking, but the note is trusted without verification, and the ranker never consults behaviorally similar users. We propose ReflectRec, organized around a preference-to-history round-trip verifier whose recall score serves as an operational signal of preference-to-history recoverability, enabling conditional regeneration for low-recoverability preferences and selection among regeneration attempts. Around this verifier, we wire a semantic Stage-1 prompt, conditional regeneration, and Forward-Look retrieval that injects future watches of behaviorally similar users as a collaborative hint. On MovieLens-100K and MovieLens-1M, the full pipeline outperforms CoT4Rec-ICL by +16.25 / +4.46 pp Hit@10, with positive gains across all ValScore buckets and three LLM backbones (Qwen3-30B, Qwen2.5-72B, and Llama-3.3-70B). We also observe a similar verifier-guided regeneration pattern on Amazon Electronics. 

---
# KnowMe Paper Radar: Lightweight Discovery of Promising Papers in the Age of Autoresearch 

**Authors**: Zhuo Chen, Jian Wang, Jinjie Gu, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.23702)  

**Abstract**: The growing volume of topically similar papers makes it difficult to identify new work that can offer useful research insights or methodological guidance. We present KnowMe Paper Radar, a lightweight discovery tool that combines multiple sources of evidence into personalized reading priorities. Users specify topics, priorities, exclusions and dates; the system brings together topical relevance, recency, author research histories, collaboration paths and direction-specific institutional context. By exposing each score component and its supporting sources, KnowMe helps readers select promising papers for their current questions and discover relevant work beyond familiar authors. Automatic acquisition, version merging and graph-cache reuse keep this workflow convenient without large language model (LLM) inference during ordinary use. Experiments on 174,640 acquired records show more than threefold faster repeated context preparation, while cached abstracts improve normalized discounted cumulative gain at rank 10 (NDCG@10) from 0.412 to 0.598 on expert-judged retrieval tasks. Complementary analyses examine how the same accumulated records support semantic retrieval, research-history modeling and literature-grounded planning. A browser interface and a structured command-line interface (CLI) make these capabilities available to both researchers and research agents, positioning KnowMe as a practical upstream tool for literature-informed recursive self-improvement (RSI) workflows. 

---
# VIBRA: Backward-Compatible Cross-Modal Retrieval with Consensus Barycenters 

**Authors**: Shubing Yang, Xutong Liu, Dongfang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.23606)  

**Abstract**: Backward-compatible retrieval enables a new encoder to query the gallery that an old model has already encoded, so that a cross-modal retrieval system can be upgraded without re-encoding billions of items. Existing compatible methods either fit the old query and stay near the accuracy of the old model, or run the old and the new encoder for every query. We propose VIBRA, which matches relevance on the old gallery, corrects the errors of a single model with the barycenter of several views, which is itself a query in the old space, and trains the new encoder to inherit this consensus, so that models absent at deployment still shape every query. We show that the barycenter gains margin over the average of its views exactly when their hardest irrelevant items differ, and we characterize the share of this gain that one encoder can reach. On five benchmarks with five upgrades among six CLIP, MetaCLIP and SigLIP2 backbones, VIBRA is the best compatible method in all 50 settings, improving Recall@1 by up to 5.6 points over the strongest baseline and by up to 18.0 points over the old model, while adding less than 1 ms to the latency of the new encoder. 

---
# LusoClin: A Synthetic Clinical Note Dataset for Medical Information Retrieval in European Portuguese 

**Authors**: Carla Teixeira Lopes, Daniel Félix  

**Link**: [PDF](https://arxiv.org/pdf/2611.23454)  

**Abstract**: Access to large-scale clinical text collections is crucial for advancing medical information retrieval research, including clinical note search, patient cohort identification, and domain-adapted retrieval models. However, real clinical notes are rarely available due to privacy regulations, which particularly limits progress for low-resource languages such as European Portuguese. We present LusoClin, a publicly released dataset of synthetic clinical notes in European Portuguese, created to support privacy-preserving research on clinical text retrieval. LusoClin was generated by pre-processing real clinical data, fine-tuning the open-source Llama 3.2 (3B) instruction-tuned model using Quantized Low-Rank Adaptation, and applying prompt-based clinical note generation. To facilitate semantic indexing and clinical search, the notes are automatically annotated with clinical entities using a BERT-based model. We evaluate LusoClin in terms of utility, fidelity, diversity, and privacy using automatic metrics and human evaluation. Results indicate that the dataset captures clinically relevant language patterns and can support downstream IR applications such as clinical document retrieval and domain-specific model adaptation, although annotators can often distinguish synthetic from real notes and some medical inconsistencies persist. 

---
# Prioritize, Don't Decide: What We Learned Auditing a Deployed Human-in-the-Loop CV Sorter 

**Authors**: Hadi Mohammadi, Jeroen Sparla, Alexandros Paliouras, Georg Krempl, Robert A. Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2611.23359)  

**Abstract**: A single academic vacancy can draw hundreds of applications, and the first screening round consumes most of a committee's time. We describe a CV Priority Sorter deployed at a national platform for academic vacancies and integrated with its vacancy database. It proposes a reading order for a pile of CVs and leaves every decision to the committee, pairing a frozen multilingual transformer encoder with a per-session linear model that retrains after each label and is discarded when the session ends. It works: on a pool of 300 synthetic CVs, a committee reading in the tool's order reaches 95% of the relevant candidates after half the pile, where an arbitrary order needs almost all of it. This paper is about what we found when we audited that claim, rather than about the system itself. Three findings changed the product. The pool turned out to be sortable by document length alone, which makes precision-style metrics on it uninformative; regenerating it under a shared length budget removes that shortcut and leaves the reading-depth result standing. The four-level agreement score such systems report, ours included, is a binary probability cut at fixed thresholds, so it follows the classifier's confidence rather than its ranking and falls as labels accumulate. And uncertainty sampling, the obvious way to implement a "suggest the next CV" button, leaves model quality intact but raises the committee's total reading load above the no-label baseline, because the CVs it asks about are the ones that would otherwise be skipped. A fourth result shows where to look next: a hosted LLM embedding model lifts zero-label ordinal discrimination from 0.66 to 0.80 under length control, then stops mattering once the session model has two dozen labels; a local cross-encoder matches that cold-start gain without sending data out. We report the fixes each finding forced. 

---
