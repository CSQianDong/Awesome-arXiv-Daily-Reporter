# A Systematic Study of Semantic ID Spaces for Generative Information Retrieval 

**Authors**: Alexia Allal, Hicham Randrianarivo, Sylvain Lamprier  

**Link**: [PDF](https://arxiv.org/pdf/2610.08732)  

**Abstract**: Generative Information Retrieval (GIR) has emerged as a transformative paradigm, shifting document retrieval from a traditional "retrieve-and-rank" workflow to sequence-to-sequence generation, where a model directly predicts document identifiers (DocIDs). While the semantic design of these DocIDs is known to be critical for performance, a fundamental question remains under-explored: what makes a good DocID? Current approaches rely heavily on computationally expensive downstream evaluations, hindering systematic analysis and rapid iteration. In this work, we address this challenge by presenting a comprehensive study on the properties, metrics, and trade-offs that define effective numerical DocIDs. Specifically, our contributions are threefold: First, we propose a unified framework that unifies Product Quantization (PQ) and Residual Quantization (RQ), and their hybrid variants within a single design space. This enables us to systematically study key DocID properties, such as hierarchy versus parallelism, as well as the impact of hyperparameters like DocID length and codebook size. Second, we define a suite of training-free, intrinsic metrics, to quantify DocID quality and evaluate structural fidelity without the overhead of full model training. Through extensive experiments on MS MARCO 300K and NQ320K, we analyze how these structural properties influence retrieval effectiveness. 

---
# Disentangling Paradigm, Identifier, and Decoding in Generative Retrieval 

**Authors**: Hicham Randrianarivo, Logan Renaud, Alexia Allal  

**Link**: [PDF](https://arxiv.org/pdf/2610.08716)  

**Abstract**: Generative retrieval trains a language model to generate the identifier of a relevant document. Recent work replaces the autoregressive decoder with diffusion, but changes identifiers, training recipe and decoding at once, so differences cannot be credited to the paradigm. On NQ320K and MS300K, we train autoregressive, masked-diffusion and block-diffusion models with residual-quantised, product-quantised and random identifiers. With identifier length and training budget fixed, we decode each model in several ways. Decoding alone moves a diffusion model's Hit@1 by 6.6 to 13.7 points. Our reference diffusion decoding, generate-and-match, generates an identifier, then retrieves the closest corpus identifiers. The generated identifier is right for 14-21% of NQ320K queries. We test one-pass scoring to decode diffusion retrievers: the model reads a fully masked identifier once, and each document is scored by its codes' probabilities. It matches or beats generate-and-match in 11 of 12 settings. Autoregressive models still lead in Hit@1; on NQ320K, the lead comes from the model, not beam search. Starting from one sampled identifier, one-pass scoring removes 46-83% of masked diffusion's deficit to beam search; from generate-and-match, at most a quarter. On NQ320K, every paradigm largely memorises which identifier answers which query: random identifiers keep 83-90% of the Hit@1 of residual-quantised ones. There, product-quantised identifiers lead residual-quantised ones by 3.4 points in the autoregressive model and by -0.7 to +3.6 in diffusion models; across decodings, AR's gap exceeds diffusion's by 1.5-2.3 points, around our 2-point threshold. Paradigm comparisons must report each paradigm at its own recipe and best decoding. 

---
# Seeing the Context: Enhancing Recommender Systems with Image-Derived Contextual Signals 

**Authors**: Tal Cordova, Tomer Geva, Moshe Unger  

**Link**: [PDF](https://arxiv.org/pdf/2610.08407)  

**Abstract**: Contextual information, capturing the circumstances of a user-item interaction, is central to recommender systems. Prior work draws context from location, time, or reviews, but not images; multimodal recommender systems mainly use images to enrich item or user representations, not identify situational context. We propose a new representation of context derived from images, spanning physical, social, and modal categories learned via a vision-language model. We introduce ICE-Fuse, a pipeline for evaluating this representation that fuses these categories and integrates them into a context-aware recommender system, using TripAdvisor data and Review-aware Graph Contrastive Learning as the recommendation algorithm. Image context does not outperform established signals standalone, but improves them combined, indicating complementary information. Semantic analysis shows image- and review-derived context capture distinct aspects of the interaction, positioning images as complementary context. 

---
# Aligning Performance with Contribution: Towards Contribution-Aware Fair Recommendation 

**Authors**: Shuai Zhang, Hui Fang, Zun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.08245)  

**Abstract**: Existing research on user fairness in recommender systems has developed diverse objectives. However, it has paid limited attention to a distinct distributive perspective: whether users' contributions to model learning should be reflected in the recommendation benefits they receive. We argue that, in addition to existing fairness protections, a fair system may account for the alignment between users' estimated contributions and the recommendation performance they receive. Such alignment can incentivize sustained and informative engagement, thereby supporting a sustainable recommendation ecosystem. To this end, we propose Contribution-Performance Fairness, a novel fairness perspective which requires recommendation performance to be aligned with estimated contribution across user groups and to remain equitable among users with comparable contributions within a same group. To instantiate this perspective, we introduce the Contribution-Performance Fair Recommender (CPFR), a framework applicable to different backbone recommenders. CPFR constructs ordered user groups from a training-dependent contribution considering interaction volume, loss alignment, and optimization intensity, and jointly optimizes recommendation accuracy with the two fairness requirements. A game-theoretic analysis shows that such alignment can strengthen contribution incentives and improve system-level recommendation accuracy under voluntary contribution. Experiments on three datasets and three backbone models demonstrate that CPFR achieves a strong accuracy--fairness trade-off under the proposed operational metric. 

---
# Behavior-Mining, Generative Conversations, and Collaborative Advisory: the Future of Travel and Tourism Recommender Systems 

**Authors**: Alejandro Bellogín, Linus W. Dietz, Francesco Ricci, Pablo Sánchez  

**Link**: [PDF](https://arxiv.org/pdf/2610.08232)  

**Abstract**: Since the early adoption of e-commerce, travel and tourism has been a lab for the design of recommender systems: tools that help travelers choose destinations, flights, accommodations, and combine them into itineraries. Data-driven recommendation techniques, ranging from case-based reasoning to reinforcement learning, have been adapted to travelers' needs. The research community has produced multifaceted prototypes of travel and tourism recommender systems (TTRSs), which are context-dependent, multistakeholder-oriented, and more recently, addressing sustainability issues, such as overtourism. Despite this enduring work, TTRSs are not widespread yet. We argue that three limitations can explain this: outdated and sparse data sets used to train and validate TTRSs, algorithms that prioritize prediction accuracy over domain-specific dimensions such as novelty and contextual relevance, and a failure to address the specific needs of travelers. Targeted incremental research could address these limitations, but a disruptive factor has meanwhile entered the ecosystem of tourism information and commercialization platforms: generative artificial intelligence. According to market research, GenAI applications are becoming the primary entry point for travelers planning their trips. This forces research to rethink how TTRSs should be designed and which core techniques should be integrated. We claim that future TTRSs, in addition to offering personalized information filtering, should become more flexible advisors that support decision making, integrating multiple data types and AI techniques, from data mining to natural language processing. Moreover, they must transparently balance the conflicting goals of travelers, service suppliers, platform owners, and local communities. We then outline research targets for building more effective TTRSs, fruitfully combining old and new recommendation techniques. 

---
# Adapting Generative Recommenders for Multi-Turn Interaction 

**Authors**: Yu-Chen Den, Zhi Rui Tam, Yung-Yu Shih, Shih-Hsin Wang, Yun-Nung Chen, Pu-Jen Cheng, Eugene Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.08136)  

**Abstract**: Generative recommenders decode items from a user's interaction history, but offer no way for users to correct a recommendation that misses their current intent. Adding conversation is natural since items and words share same output space, yet training the model to converse may overwrite the history-to-item mapping it relies on. We introduce INTEGER (**INTE**ractive **GE**nerative **R**ecommendation), which extends generative recommendation to multi-turn interaction with a learned routing token that lets the model decide when to recommend, history re-anchoring that conditions each item on both past behavior and the dialogue, and behavioral replay with instruction-data rehearsal that prevents forgetting during adaptation. Users can thus give feedback on recommendations within the dialogue, while recommendations stay grounded in behavioral history and accuracy is not traded for fluency. On Amazon Beauty and Toys, INTEGER matches or exceeds the strongest baselines in accuracy with competitive conversation quality, improving Hit@10 by 13.3% on Amazon Beauty, and significantly outperforms the generative recommender it starts from. Our analyses show that INTEGER learns behaviors that naive adaptation fails to acquire, recommending once the user's intent is clear and staying attentive to behavioral history at the moment of recommendation. INTEGER also learns an intent-agnostic replacement over the item space, which suppresses rejected items but points to attribute-aware feedback as the next step. 

---
# From Delivery to Stateful Exploration: Rethinking the Index for Agentic Search 

**Authors**: Deogyong Kim, Sunghwan Kim, Sangam Lee, Wonjae Lee, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.07960)  

**Abstract**: Recent advances in agentic search have given large language model (LLM) agents finer control over corpus exploration. However, search interfaces often return matching passages even when feedback about the candidate set would suffice for the next decision, coupling candidate refinement with source-text exposure. We propose IndexAct, an interface for Index-Native Corpus Interaction that separates candidate-set refinement from text inspection. Agents construct and manipulate persistent candidate sets through lexical conditions and set operations over an inverted index, receiving reusable state references and statistics such as candidate counts rather than matching passages. This feedback guides further refinement, while separately requested passages provide new clues or evidence that can inform subsequent operations on retained candidate sets. Experiments on five benchmarks spanning agentic search and multi-hop question answering show that IndexAct outperforms the evaluated baselines on each benchmark. On BrowseComp-Plus, it also achieves higher evidence coverage with a smaller average live context than terminal-based corpus interfaces, and maintains answer accuracy as the corpus expands. Further analyses suggest that informative refinement feedback and state reuse support continued evidence discovery, while shorter contexts or fewer search steps alone do not ensure better performance. 

---
# Contrastive Learning for Aspect Representation towards Explainable Recommendation 

**Authors**: Emrul Hasan, Chen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2610.07761)  

**Abstract**: In this work, we propose a novel recommendation model, CLARER (Contrastive Learning for Aspect Representation towards Explainable Recommendation) that integrates aspect features learned from textual reviews with rating information to improve the accuracy and explainability of recommendations. Our proposed framework learns user and item representations by combining rating-based features and aspect-based features from reviews. Specifically, rating-based features are learned through a multi-layer perceptron (MLP) model, while aspect-specific review representations are learned using a transformer encoder to capture the semantic information and contrastive learning to better distinguish user preferences. To provide explanations, we train a transformer decoder, using the final representations of users and items from both rating and aspect-based features as context. Experimental results in three benchmark data sets demonstrate that our model achieves superior performance compared to baseline methods in both recommendation (accuracy) and explanation generation. 

---
# Token-Budgeted Escalation for Financial Document QA: Cost Is Predictable, Benefit Is the Bottleneck 

**Authors**: Junru Zhu, Yixin Yang, Xiaoqing Ding, Ruoyu Qi  

**Link**: [PDF](https://arxiv.org/pdf/2610.07760)  

**Abstract**: Retrieval-augmented generation systems can route difficult queries to deeper context, but batch deployments must allocate a shared token budget across calls whose costs vary by query. We formulate selective escalation as finite-batch allocation for financial document question answering. Each of 150 FinanceBench questions first receives a top-1 retrieval answer. Predictors estimate the adjudication-quality gain and token cost of an optional top-5 call, and the allocator prioritizes calls by predicted gain per token. At the nominal 10% budget, gain-per-token allocation improves adjudication quality over gain-only ranking by 0.034 (95% document-bootstrap CI [0.001, 0.072]) while using 46.6% fewer total tokens than one-pass top-5 retrieval. Additional-call cost is accurately predictable (R-squared 0.93), whereas beneficial escalation remains difficult to rank (AUROC 0.60). These results show that heterogeneous cost is actionable under tight constraints, while progress across the full budget frontier depends on stronger query-specific benefit estimates. 

---
# Learning to Retrieve via Reinforcement Learning in Embedding Space 

**Authors**: Qi Liu, Fengming Liang, Yiqun Chen, Erhan Zhang, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2610.07731)  

**Abstract**: Dense retrieval models are typically trained with contrastive objectives that learn effective representations but do not directly optimize retrieval metrics or downstream task performance. To address this problem, we introduce RELER (REinforcement LEarning for Retrieval), a reinforcement learning framework that enables existing embedding models to learn to retrieve directly in embedding space and align to task-specific rewards. We train RELER by sampling unit-length query and document embedding actions from von Mises-Fisher (vMF) distributions centered on normalized encoder outputs, scoring the resulting retrieval or downstream outcomes as rewards, and updating the encoder with REINFORCE using a leave-one-out baseline (RLOO). As exploration in the high-dimensional embedding space is prone to sampling noise, we further propose conditional-mean projection (CMP), which projects each sampled embedding onto the low-dimensional subspace spanned by its encoder output and the candidate embeddings it is compared against, reducing noise in the policy gradient while preserving its expectation. We evaluate RELER on BRIGHT, a benchmark with reasoning-intensive queries that remain challenging for existing embedding models. RELER consistently outperforms InfoNCE and LambdaLoss in average nDCG@10 when post-training BGE-M3 and Qwen3-Embedding backbones. We further evaluate downstream utility through retrieval-augmented generation (RAG), where we adapt only the query encoder while keeping the document index and generator fixed. Across seven QA datasets, jointly optimizing retrieval and answer rewards improves both average retrieval performance and answer quality in RAG. 

---
# DBRAG: Multi-Table Retrieval-Augmented Generation for Complex Database Queries 

**Authors**: Prince Larbi Ampofo, Ryoji Kubo, Djellel Difallah  

**Link**: [PDF](https://arxiv.org/pdf/2610.07622)  

**Abstract**: Recent advancements in large language models have introduced new capabilities for reasoning over structured data, particularly through program-aided tools that can analyze tables. However, many existing methods address single-table scenarios or assume that the relevant tables are already provided. In practice, users often issue complex data exploration queries over entire databases, where relevant information may be distributed across multiple relations. In this work, we introduce DBRAG, a retrieval-augmented generation framework tailored for multi-table question answering. DBRAG first retrieves candidate tables using an offline table index, enriches their summaries with query-relevant rows, and uses an LLM to rerank the candidates. A program-aided reasoner then selects the required tables and executes operations over their full contents, keeping the initial prompt context compact. Experiments on the Spider, GeoQuery, and ATIS datasets used in this study demonstrate improvements in table retrieval and multi-table question answering. 

---
# A Study of Prior Case Retrieval Using Lexical, Semantic, and Rhetorical Role Information in Indian Legal Documents 

**Authors**: Sayed Ayaan Ahmed Sha, Sangeetha Sivanesan, Anand Kumar Madasamy, Navya Binu, Aniket Mani, Rishu Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2610.07437)  

**Abstract**: For retrieving prior cases in Indian legal judgments, the problem involves distinguishing relevant legal facts from mere lexical similarities because a prior case that shares a statute with the query judgment is not necessarily relevant. In this paper, we describe an empirical evaluation of three consecutive designs of retrieval systems for the IL PCR(Indian Legal Prior Case Retrieval) task. We demonstrate that the combination of the rhetorical roles (Fact, Ratio Of The Decision, Precedent, Argument, Statute) is better than either using individual roles or performing full text retrieval, that statute similarity is not discriminative, and that a legal entailment reranker with training data produced by an LLM is much worse in terms of official evaluation than its internal validation score. Two rankers, with excellent internal MRR up to 0.97, performed poorly in terms of official evaluation (MRR as low as 0.14). Thus, we propose a new design with query disjoint splitting and frozen validation fusion. The result is a four stage pipeline with Micro F1 0.2549, MRR 0.6081, and nDCG@10 0.4187. 

---
# Rethinking Semantic ID Construction for Generative Recommendation: SimHash with Parallel Decoding and Semantic Alignment 

**Authors**: Yuqing Liu, Huiyuan Chen, Yibo Wang, Wooseong Yang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.07402)  

**Abstract**: Semantic ID-based generative recommendation represents each item as a sequence of discrete tokens, enabling structured modeling of item semantics. A critical challenge is constructing semantic IDs that are both semantically expressive and computationally efficient. While recent approaches favor complex learned quantization, simple hashing-based methods such as SimHash are widely regarded as fundamentally inferior. In this work, we challenge this consensus by showing that the apparent performance gap does not stem from inherent limitations of hashing, but rather from a structural mismatch with autoregressive decoding, coupled with the inevitable information loss during rigid discretization. Based on this insight, we propose FLASH, a two-stage framework that revitalizes training-free SimHash tokenization through parallel decoding and explicit semantic alignment. Despite its simplicity, FLASH achieves state-of-the-art performance across multiple datasets without requiring any tokenizer training, while exhibiting stronger generalization in cold-start scenarios. Notably, we demonstrate that semantic alignment acts as a universally effective mechanism across diverse paradigms. Our findings suggest that, with compatible decoding and semantic grounding, simple and efficient tokenizers can achieve performance comparable to complex learned counterparts in generative recommendation. Our code is available at this https URL. 

---
# RAGFlip: Measuring Query-Level Negative Flips in Retriever Upgrades 

**Authors**: Elyas Irankhah, Muhammad Arif  

**Link**: [PDF](https://arxiv.org/pdf/2610.07266)  

**Abstract**: Retriever upgrades are typically evaluated using aggregate metrics, which can hide regressions on queries the previous retriever already served correctly. We study these regressions as negative flips: queries for which BM25 retrieves a judged relevant passage and the replacement does not. We evaluate BGE-large, E5-large-v2, and SPLADE on three BEIR collections: Natural Questions, HotpotQA, and FiQA, across five retrieval depths. All replacements improve overall retrieval coverage. Negative flips occur in every setting and vary substantially by corpus, retriever, and depth. At k=1, 8.6-37.5% of BM25 successes are lost across the evaluated settings. Negative-flip rates are lower in the larger-depth settings, where the BM25-supported cohort is defined separately at each depth. These rates use the any-relevant support label. On HotpotQA at k=10, requiring every positive qrel passage raises the negative-flip rate to 12.3-17.3%. Simple fixed-budget combinations with BM25 reduce these regressions, and a HotpotQA reader experiment provides a limited downstream check in which some retrieval flips are accompanied by answer regressions. These results motivate evaluating retriever updates using query-level compatibility alongside aggregate retrieval quality. 

---
# Beyond Successor Accuracy: State Retention for Recursive Self-Improvement in Recommendation 

**Authors**: Jinfeng Xu, Zheyu Chen, Ziyue Peng, Zheng Lin, Wenhao Yuan, Jian Chen, Shujie Li, Edith Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2610.07105)  

**Abstract**: Recommendation recursive self-improvement (Rec-RSI) feeds recommender outputs into subsequent training. Evaluating each round solely through its latest model assumes that the successor consolidates the update, although pre- and post-update models may retain complementary ranking decisions. We term this \emph{distributed progress} and quantify it using cross-generation advantage (CGA), a marginally matched contrast between cross- and within-generation model pairs. A rank-separation statistic, label-free at selection time, predicts which family to retain. Across four datasets and three sequential recommendation encoders, the preferred retention regime varies by architecture: cross-generation pairing benefits GRU4Rec and SASRec, whereas FMLP initially favors within-generation pairing and shifts toward cross-generation pairing after a second update. Rank separation selects the stronger family in 12/12 first-update and 5/6 second-update dataset-encoder settings; on held-out tests, the selected family outperforms the direct successor in 34/36 trajectories. Five transfer mechanisms do not consistently reproduce these gains in one model. These findings establish state retention as a distinct Rec-RSI problem: progress may reside in relations between generations as well as in the latest model. Code is available at \href{this https URL}{this https URL}. 

---
# UNREAL: Unifying Retrieval and Long-Context with a Single Model 

**Authors**: Edan Kinderman, Elad Hoffer, Yochai Blau, Brian Chmiel, Ron Banner, Daniel Soudry, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2610.08463)  

**Abstract**: Long-context inference and Retrieval-Augmented Generation (RAG) handle evidence selection at vastly different scales, from a single long prompt to an entire corpus. We ask whether a single model-internal mechanism can select evidence across this range. We introduce UNifying REtrieval And Long-Context with a Single Model (UNREAL), a model-native evidence selection framework to span corpus retrieval and long-context inference. UNREAL encodes chunks and derives retrieval queries directly from the frozen LLM's internal representations. It adds fewer than 500K trainable parameters and leaves the backbone unchanged. On a 3B-token, 21M-chunk Wikipedia index, all four dense and hybrid UNREAL backbones outperform state-of-the-art retriever-reranker systems. The best model raises recall from 49.1% to 73.2% on HotpotQA, from 31.7% to 60.1% on 2WikiMultiHopQA, and from 8.8% to 14.4% on MuSiQue. Applied to long-context tasks, the same selection mechanism removes distractors before generation, raising NoLiMa accuracy from 1.0% to 24.83% at its maximum context length of 128K tokens, and LV-Eval's F1 score from 49.97% to 54.66% at 256K. UNREAL also reduces FLOPs and time-to-first-token relative to full-context inference from roughly 32K tokens onward, with larger gains as context grows. Together, these results establish model-internal evidence selection as a common foundation for corpus retrieval and evidence-sparse long-context inference. 

---
# Agentic AutoRAG: RAG Pipeline Optimization through Reasoning-Driven Agents 

**Authors**: Lasse B. Strand, Robert Jakob, Kevin O'Sullivan, Markus Kreft  

**Link**: [PDF](https://arxiv.org/pdf/2610.08452)  

**Abstract**: Retrieval-augmented generation (RAG) is a widely used approach for grounding large language models (LLMs) in external knowledge. However, configuring a pipeline is an expensive hyperparameter optimization problem over many interacting choices, from chunking and embedding model to reranking and generation. Existing optimizers, from greedy search to Bayesian optimization, reduce each trial to an aggregate score and search without modeling why a configuration performed as it did, even though the retrieved chunks already provide evidence about whether each failure occurred during retrieval or after it. We introduce Agentic AutoRAG, an LLM-agent optimizer for multi-objective RAG hyperparameter optimization with retrieval-versus-generation failure attribution. It proposes configurations scored on a frozen exam from the corpus: after each trial a Diagnoser attributes each failed question to retrieval or generation, and a Proposer, grounded in a knowledge base of model rankings and pricing, selects the next configuration, weighing accuracy against cost to trace a Pareto frontier. On three multi-hop QA benchmarks it reaches higher LLM-judge accuracy than every baseline we compare, and within its first 10 trials it matches or beats the statistical baselines' full 30-trial judge accuracy. In its cost-aware mode on a real-world healthcare corpus it reaches a median exam accuracy of 77%, above the strongest baseline's 71.5%, at about 58% of that baseline's cost per query, and it matches that 71.5% at about 22% of the cost. 

---
# Confidence-Ordering Reversal under Contextual Priors in Neural Decoding 

**Authors**: Xinyu Zhang, Sichao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.08229)  

**Abstract**: Contextual priors improve neural-to-language decoding by reshaping candidate scores. However, confidence is read from the same reshaped scores, so the errors a prior leaves behind can become more confident with no change in accuracy to reveal it. We study how a prior shapes confidence in speech retrieval on MEG-MASC and MOUS using local decoding scores, a contextual prior combined by additive shallow fusion, and the fused top-two margin as confidence. Among initially incorrect predictions, we find a confidence-ordering reversal: a larger margin makes a repair more likely when the correct candidate starts near the top of the local ranking, but less likely when it starts lower. On MEG-MASC, pooled correctness AUROC is 0.87, yet AUROC separating repairs from residual errors falls from 0.70 at initial ranks 2-3 to 0.39 at ranks 21-50. Errors starting beyond rank 20, inside the reversed region, make up 46.6% of all post-fusion errors. We propose a score-level account: a repair must first close the correct candidate's initial deficit, limiting its final margin, whereas a residual error can build a large margin between two incorrect candidates. A causal intervention that changes only the fusion weight moves the reversal to deeper ranks as predicted. Under a word-level LM prior, it keeps moving after accuracy gain peaks, so a weight chosen for accuracy does not settle confidence. Reading local and prior scores separately improves selective decoding: the decoder answers on 74.5% of windows instead of 56.7%, while 92% of output sets still contain the correct candidate. Confidence after contextual fusion should retain the local and contextual evidence behind each prediction, not just the fused scores. Project website: this https URL Code: this https URL 

---
# Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight 

**Authors**: Haoxiang Zhang, Qinglin Chen, Hiroaki Hayashi, Zhuofeng Li, Siming Zhang, Jiaxin Zhang, Jixuan Chen, Fang Wu, Pan Lu, Silvio Savarese, Julian McAuley, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.08077)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) turns agent experience into learning signals primarily through scalar outcome rewards after interaction. For group-relative objectives, however, this signal vanishes when all rollouts receive the same reward, even though their trajectories may reveal useful information about what the task requires and how the agent fails. We ask a complementary question: can hindsight teach an agent what it could have anticipated before acting? We introduce prospective learning, which uses post-hoc experience to supervise foresight predictions from the pre-interaction view, and instantiate it with Self-Retrospection Distillation (SRD). Intuitively, a completed trajectory reveals knowledge that would have been useful and pitfalls that should be avoided; SRD distills this privileged hindsight into trajectory-blind foresight of the same policy. Foresight serves only as a training target and need not be explicitly generated at inference time. Across 10 tool-integrated reasoning and long-horizon agentic tasks, SRD complements RLVR and self-distillation baselines with gains of up to $24.2$ pp. Its advantage is especially pronounced when reward contrast is scarce: when $37$--$98\%$ of rollout groups are reward-uniform across model scales, yet SRD can still exploit learning signal from sampled trajectories. In the 2B setting, where $98\%$ of groups are all-failure, the RLVR training ends up at $0.0\%$ success, while adding SRD reaches $60.6\%$ under the same rollout budget. Our results suggest that post-hoc agent experience is useful not only for evaluating or improving behavior, but also for shaping predictive representations before available interaction. 

---
# ShanLiangRen: A Nutrition Agent for Personalized Daily Meal Planning 

**Authors**: Miao Xie, Xiao Zhang, Yuan Wang, Ruixin Zhu, Chunli Lv  

**Link**: [PDF](https://arxiv.org/pdf/2610.07886)  

**Abstract**: Dietary nutrition planning plays an important role in chronic disease management and maintaining a healthy body. In applications, it must simultaneously satisfy personalized constraints and reasonable multidimensional nutritional goals. These two aspects often conflict, and user constraints evolve with feedback, resulting in a substantial gap between generic guidelines and executable plans. To bridge this gap, we first propose the personalized fully quantified multiobjective dietary planning problem (MDP). To tackle MDP, we develop a nutrition agent, ShanLiangRen. The system first transforms dietary specifications, nutrient data, user attributes and natural language requirements into an individualized constrained planning instance. It then employs an exact retrieval-augmented generation method to shrink the feasible candidate set from a large scale ingredient and recipe space. Finally, it adopts a refinement guided by Pareto principles, where an LLM iteratively revises candidate plans under deterministic nutrition computation and feedback from constraint verification. The system outputs fully quantified meal plans with explicit ingredients and portion sizes, together with reports on nutrition compliance that show constraint satisfaction and nutrient interval attainment. We have released the system online as a WeChat Program, ShanLiangRen. A demo video is available at this https URL. 

---
# WildMatch: Weakly Supervised Image Matcher Adaptation for Wildlife Re-Identification 

**Authors**: Turhan Can Kargin, Piotr Kubaty, Ekaterina Rostovskaya, Izabela Wierzbowska, Bartosz Zieliński, Marcin Przewięźlikowski  

**Link**: [PDF](https://arxiv.org/pdf/2610.07384)  

**Abstract**: Individual animal re-identification from camera-trap imagery is an instance retrieval problem central to non-invasive wildlife monitoring: a query image must retrieve the correct individual from a reference set of known animals. This requires computer vision models to recognize distinctive local patterns in fur, skin, or other visual markings. Current approaches either learn global embeddings as a classification problem, requiring many labeled images per individual while largely ignoring local evidence, or apply off-the-shelf, domain-agnostic image matchers. Although such matchers are pretrained on large and diverse image collections, adapting them to wildlife imagery is challenging because available datasets are small and lack correspondence-level annotations. We study weakly supervised adaptation of a pretrained keypoint matcher using only identity labels, without keypoint-level or geometric correspondence ground truth. We mine informative image pairs with the pretrained matcher, derive weak positive and negative supervision from identity agreement, and contrastively fine-tune the matching network to strengthen correspondences for same-identity pairs and suppress them for different identities. Across open-source wildlife re-identification datasets, our approach improves accuracy over off-the-shelf matchers and a state-of-the-art local--global fusion method. Under an open-world protocol with held-out individuals, it learns a transferable correspondence prior rather than memorizing training identities. To our knowledge, this is the first study of matcher-level, identity-supervised adaptation for animal re-identification. Our method enables data-efficient specialization of image matching models to wildlife domains using identity annotations already available in typical monitoring datasets. 

---
# The Right Memory in the Wrong Context: Verifying Retrieval Admissibility in Long-Term Agent Memory 

**Authors**: Zi Wang, Xingqiao Wang, Emmanuel Addai, Devika Ambekar, Xiaowei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.07309)  

**Abstract**: Long-term-memory agents can retrieve relevant information that is inadmissible for the current request because it belongs to another principal, violates policy, or reflects an incompatible lifecycle state. Recall and final-answer accuracy do not reveal this: a route can appear safe by missing required evidence, while a correct answer may follow inadmissible prompt exposure. We introduce a retrieval-admissibility verification framework that assigns each memory-query pair one of three statuses (admissible, inadmissible, or unresolved), compares routes at matched required-evidence recall with bounds for unresolved cases, and tracks memory IDs through prompt exposure while linking exposure to target-level disclosure. We evaluate its stages on separate, non-pooled populations. A post-hoc top-20 reanalysis of frozen rankings from two public long-term-memory benchmarks, RHELM and MemOps, covers 3,767 queries. All released anchors lie within trusted query namespaces; with within-namespace scores unchanged, off-namespace filtering cannot lower their ranks. Top-20 anchor recall increases from 0.432 to 0.533, 80% recall feasibility from 0.237 to 0.311, and exact similarity evaluations decrease by 98.3%. In a frozen 72-case development diagnostic, a released-metadata reference preserves required evidence, whereas neither text-only verifier detects violations under the 1% required-anchor false-denial limit. Across 1,523 paired benchmark-native cases, namespace routing is associated with judged-accuracy gains of 0.053-0.068 across three readers; recall also changes, so this comparison is observational. In 16 controlled exposure scenarios, only one of four reader-specific 95% confidence intervals excludes zero for relevant-inadmissible literal disclosure (+0.156, 95% CI [0.031, 0.312]). Results motivate separate verification of candidate support, admissibility, prompt exposure, and answer disclosure. 

---
# CroissantMiner: Automated Extraction and Validation of Croissant Metadata for ML Datasets 

**Authors**: Berke Arda, Ahmetcan Yavuz, Paul Gerry, Sebastian Lobentanzer, Nobin Sarwar, Joan Giner-Miguelez, Kongtao Chen, Luyao Zhang, Mrinmaya Sachan, Mubashara Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2610.07132)  

**Abstract**: Croissant has emerged as a standard for machine-readable dataset metadata, yet populating its fields remains labor-intensive and requires careful reading of accompanying dataset documentation. We present the first benchmark enabling end-to-end evaluation of metadata extraction aligned with a community-standard schema. The benchmark comprises 602 papers, including 102 with human-validated gold annotations and 500 with LLM-generated silver annotations, covering the full Croissant schema with both core and Responsible AI (RAI) fields. Using this benchmark, we evaluate a range of extraction systems spanning frontier models, open-weight models, and agentic architectures, under a two-tier evaluation framework that combines rule-based scoring with an LLM judge selected via human audit. We find that single-pass extraction consistently outperforms the four agentic architectures we evaluate: across backbones, these decomposed variants achieve lower accuracy than a single full-context pass. The largest gap appears on long-form RAI fields, which require synthesizing and interpreting information scattered across a paper rather than copying it from a single location, a setting where current systems remain far from reliable. We release the benchmark, evaluation code, judge audit, a live demo, and a leaderboard open to new systems. 

---
# Smart Content Ingestion for Generative AI Workloads 

**Authors**: Abbas Raza Ali, Muhammad Ajmal Siddiqui, Moona Zahid  

**Link**: [PDF](https://arxiv.org/pdf/2610.07091)  

**Abstract**: The evolution of machine learning has progressively changed where intelligence resides in an AI system. In conventional machine learning the task, data representation, labels and model architecture were tightly coupled, so data preparation was narrow, schema-bound and visible. Generative AI decouples the model from any single task: one foundation model serves open-ended downstream tasks, and the generality gained on the model side is matched by heterogeneity on the data side, because enterprise knowledge is authored in the formats people use (PDF, presentations, spreadsheets, scanned documents, forms, tables, diagrams and mixed-layout files) that carry textual, visual, geometric and structural information at once. A language model or retriever cannot reason reliably over information misrepresented at this interface, so content extraction becomes a lifecycle stage in its own right whose errors no downstream retriever or re-ranker can repair. This paper presents a production-ready content-extraction system that makes this stage explicit, configurable, and measurable. The system incorporates selective OCR routing, a scarcity-first curation engine with a reference-based extraction scorer that measures character, word, and table-structure accuracy, a deterministic structure-aware parent-child chunker, and a read-only retrieval evaluator that generates grounded questions from every page and reports Hit@k, mean reciprocal rank, and latency. On a 180-document corpus the best extractor scores 97.4 of 100 (character error rate 0.13%, table similarity 0.995) and the chunker reaches hit@1 of 68.6%, hit@10 of 92.8% and MRR 0.77 over 25,050 generated questions. We distil three design principles (structure before semantics, never mutate what you measure, budget your labels) and position measured content extraction as the perception layer of enterprise agentic systems. 

---
# Beyond Refusal Patterns: Safe-Role Internalization for Robust and Generalizable LLM Safety Alignment 

**Authors**: Jinghao Pang, Jitai Hao, Qiang Huang, Zhaochun Ren, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.07023)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable capabilities but remain vulnerable to jailbreak attacks that elicit harmful or unsafe outputs. Existing safety alignment approaches, including Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), often require substantial attack-specific supervision and computational resources, while remaining susceptible to shallow safety alignment and over-refusal. To address these challenges, we introduce SSRFT(Supervised Safe-Role Fine-Tuning), the first framework that reformulates safety alignment as the internalization of a predefined safe role. SSRFT constructs a Safe-Role Question-Answer (SRQA) dataset from psychometric questions, limited jailbreak prompts, and a safe-role description. Role-consistent responses are synthesized, validated, and expanded into diverse scenarios, enabling models to internalize safety-oriented values and principles rather than explicit refusal patterns. Experiments across multiple Base and Instruct models show that SSRFT achieves more robust and generalizable safety alignment than standard SFT. SSRFT shows substantially greater robustness to prefilling attacks and better generalization to unseen jailbreak domains, while reducing over-refusal on benign queries and preserving the model's general capabilities. These results establish safe-role internalization as an effective alternative to refusal-centric safety alignment. Warning: This paper contains examples of harmful and toxic language. 

---
# Tree Navigation Without LLM Summaries: A Matched-Cost Study of Hierarchical Retrieval for Long-Document QA 

**Authors**: Priyank Jayraj, Poonam Goyal, Navneet Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2610.06902)  

**Abstract**: Retrieval-augmented generation grounds language models in external context, but for long documents flat top-$k$ retrieval can cluster on a single region and miss complementary evidence. RAPTOR-style summary trees address this by recursively clustering chunks and using a language model to summarize each cluster at indexing time, then ranking summary nodes alongside raw chunks at query time. We show the main benefit of summary trees in long-document QA can come from navigation rather than the generated summary content. We introduce NavTree, a leaves-only retriever that builds a deterministic balanced segment tree over chunks (zero language-model calls at indexing) and uses the tree purely as a navigation scaffold: a hybrid lexical-and-dense frontier walk, anchored on top retrieved leaves, descends from the root and emits only leaf chunks to the reader. On a matched-cost evaluation against flat retrievers and an extractive re-implementation of RAPTOR, NavTree is the strongest matched-cost hierarchical retriever in our evaluated grid and ties the strongest flat baseline. On long-document multi-hop QA, it is the only hierarchical method that significantly beats BM25 on a class-vs-class basis, corroborated by a reader-free retrieval-recall check. A matched-reader replication of the published abstractive RAPTOR variant, given strong cluster summaries, still loses to NavTree at every multi-chunk budget, at zero indexing cost. The ranking carries across stronger and open-weight readers, a stronger encoder, and a full factorial that isolates leaves-only emission as the structural lever. 

---
# Diff-SQL: SQL Efficiency Optimization via Patch Generation and Constraint Alignment 

**Authors**: Shipei Lin, Duomin Zhang, Xiaolong Li, Bohan Hu, Bowen Qin, Jinyang Li, Chenhao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2610.06857)  

**Abstract**: SQL efficiency optimization aims to transform slow queries into semantically equivalent but faster alternatives. However, directly optimizing SQL with large language models in an end-to-end fashion often induces Objective Misalignment which creates a fundamental tension between optimization and correctness, making direct full SQL rewriting unreliable for execution-facing database applications. To address this problem, we propose Diff-SQL, a two-stage framework that decouples efficiency-oriented optimization from constraint-aware alignment. The first stage identifies optimization opportunities and proposes targeted edits in the form of a unified diff patch, while the second stage is trained with on-policy reinforcement learning to revise outputs under executability and semantic-equivalence constraints. To train and evaluate Diff-SQL, we construct an automated pipeline that mines optimization knowledge from StackOverflow and builds Slow-Fast SQL pairs through cascaded filtering. We further introduce Effi-SQL, a benchmark containing 1,100 human-verified Slow-Fast pairs across five SQL dialects. Experiments show that Objective Misalignment is widespread across existing LLM-based SQL optimization methods, where direct full SQL optimization causes an average 22.7% execution accuracy degradation across frontier models such as Claude-Opus-4.6, with the worst model dropping by 43.0%. Diff-SQL alleviates this trade-off. As an inference-only strategy, it improves R-VES by 10.0% on average while reducing execution accuracy degradation by 6.11% on average across three strong base models. With execution-grounded training, Diff-SQL further enables a 7B model to improve R-VES from 33.42% to 46.83%, demonstrating that the proposed two-stage optimization-and-alignment paradigm can deliver both stronger efficiency and better correctness in local, small model deployment settings. 

---
