# Reading the Mood: Emotion-Guided Book-to-Music Recommendation via CGANs and LLMs 

**Authors**: Manousos Linardakis, Georgios Alexandridis  

**Link**: [PDF](https://arxiv.org/pdf/2610.06703)  

**Abstract**: Background music that matches the mood of a text has been shown to make readers feel more immersed and improve their reading experience, motivating recommender systems that pair books with mood-matched music. In this direction, we present Sentiment Aware Generative Adversarial Network for Cross Domain Recommendation (SAGA-CDR), a two-phase cross-domain recommendation framework that personalizes music suggestions and emotionally aligns them with the book being read. In the first phase, transformer-based sentiment embeddings are constructed from user reviews and mapped across domains via a Conditional Generative Adversarial Network, whose mask-conditioned generator handles missing sentiment components and injects stochasticity for richer preference transfer. A compact rating neural network then fuses sentiment-specific interaction scores with a collaborative filtering prior to predict music ratings. In the second phase, large language models classify each book into a valence-arousal emotional quadrant, and candidate tracks are filtered to match that quadrant. Experiments on both the English Amazon and Chinese Douban datasets show that SAGA-CDR achieves the best rating prediction accuracy on Amazon (RMSE 0.98) and the lowest RMSE on Douban (0.91), with ranking performance competitive with the strongest sentiment-aware baseline, even in cross-lingual settings. 

---
# SPRIG: Semantic-ID-enhanced Paths for Knowledge Graph-based Generative Recommendation 

**Authors**: Justin Hangoebl, Marta Moscati, Alessandro B. Melchiorre, Shah Nawaz, Markus Schedl  

**Link**: [PDF](https://arxiv.org/pdf/2610.06590)  

**Abstract**: Recommender systems leveraging generative models often generate item identifiers directly, rather than ranking catalog items by a recommendation score. Recent work extends beyond pure sequential interaction signals by incorporating item content and structured relationships among items, with two distinct directions emerging. Semantic IDs (SIDs) enrich item representations by replacing opaque, randomly initialized embeddings with hierarchically quantized discrete codes derived from item content. Knowledge-graph (KG) path reasoning instead generates entity-relation paths that ground recommendations in structured relationships between items, attributes, and external entities, thereby enriching the relational context. These two lines have complementary limitations: SID-based models lack relational grounding, while KG-based generative recommenders still represent items as arbitrary, opaque tokens tied to large embedding tables, limiting parameter sharing and generalization. We propose SPRIG, a generative recommender that integrates content-derived SIDs into KG path reasoning. SPRIG is trained on information-rich KG paths that terminate in items represented as discrete, content-derived tokens, combining the advantages of both approaches. We evaluate SPRIG on movie and music recommendation datasets against baselines spanning sequential language models, KG-augmented methods, and SID-based approaches. Our results show that SPRIG achieves competitive performance over prior generative models while using fewer parameters and a lower compute cost. Code: this https URL 

---
# Commercial Intent in Human-AI Conversations: A Corpus Audit and Architecture for Website Sales Agents 

**Authors**: Benjamin Tannenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2610.06205)  

**Abstract**: Conversational sales agents must distinguish questions about products from purchase commitments, preserve explicit requirements, and ground the next action in current business information. We report an aggregate census of 725,219 records in an accessible conversation table provided by Aiso and develop a reference architecture for this setting. All records have distinct non-null conversation hashes. Existing metadata labels identify 41,800 commercial records (5.76%) and 2,387 transactional records (0.33%); their union contains 44,187 records (6.09%). Within the commercial category, 54.31% are labeled English, 41.59% have recorded depth of at least two, and 13.51% have depth of at least four. Commercial-label prevalence varies from 4.87% to 6.35% across three source batches. These measurements motivate explicit separation of corpus inventory, commercial relevance, training eligibility, and observed business outcomes. The proposed architecture combines business-grounded knowledge, provenance-bearing conversation state, and a constrained next-action policy. A quality specification addresses source rights, privacy, label validation, deduplication, and training-test separation. The study is a metadata audit and technical design, not a validation of label accuracy, model training volume, or sales conversion. No raw conversation text or personal identifiers are released. 

---
# Beyond States: Investigating the Effects of Context on User Modeling with Feature-Conditioned Markov Models 

**Authors**: Jana Isabelle Friese, Andreas Konstantin Kruff, Timo Breuer, Philipp Schaer, Norbert Fuhr  

**Link**: [PDF](https://arxiv.org/pdf/2610.06060)  

**Abstract**: User behavior simulation is widely used to evaluate interactive information retrieval systems, but classical state-based approaches (e.g., Markov models) have limited ability to incorporate contextual information relevant for decision-making. We address this limitation by introducing a feature-conditioned Markov-style user model, in which transition probabilities are modeled as functions of positional, content-based, and interaction-derived features, enabling context-aware decision making while preserving the structural simplicity and computational efficiency of state-based models. Applying a multi-level framework that assesses predictive fit and behavioral fidelity, we analyze how different sources of contextual information contribute to realistic user simulation across multiple datasets, search settings, and feature configurations. Our results show that incorporating contextual features improves the models' ability to reproduce key aspects of real user interactions, but that their effectiveness hinges on search scenario and modeling objective. Instead of a one-size-fits-all solution, effective simulation requires task- and setting-specific feature selection. Our framework provides a practical and interpretable basis for making these choices. 

---
# MATE: Adaptive Long- and Short-Term User Memory for LLM-Based Recommendation 

**Authors**: Yu Hou  

**Link**: [PDF](https://arxiv.org/pdf/2610.06050)  

**Abstract**: Large language model (LLM)-enhanced recommender systems leverage rich item semantics to support personalized recommendation. However, semantic representations alone do not determine which historical behaviors reflect persistent preferences and which mainly indicate recent interests, leaving an important aspect of user understanding unresolved. Recent advances in LLM inference show that newly available information can be used to refine the internal state during inference, thereby improving subsequent predictions. Inspired by this principle, we propose MATE (Memory Adaptation with Temporal Evidence), an adaptive user modeling framework for LLM-enhanced sequential recommendation. MATE first evaluates each newly observed interaction from two temporal perspectives: whether it is repeatedly supported by historical behaviors and whether it is consistent with recent interactions. The resulting temporal evidence controls the updates of two user-specific memories, where the long-term memory conservatively preserves persistent preferences while the short-term memory rapidly adapts to recent interests. For each recommendation, a recent-context representation dynamically determines how strongly the two memories contribute to the current user representation. During offline training, next-item prediction is jointly optimized with temporal supervision, while during online adaptation, the shared model remains fixed and only the two user memories are updated from newly observed interactions. Experiments on MovieLens-10M, Amazon Luxury Beauty, and KuaiRec show that MATE improves mean NDCG@10 over the strongest external baseline by 7.0--13.2%. Further analyses support its ability to adapt to recent interests while retaining useful information about recurring earlier preferences. 

---
# Constraint-Aware Conversational Job Recommendation in Code-Mixed Low-Resource Settings 

**Authors**: Md Arman Hossain, Mubashir Jawad, Fariha Khandaker Moon, Sonia Binte Siraj, Masfiqur Rahaman, Raihan ul Islam, Ahmed Wasif Reza, Nafis Sadeq  

**Link**: [PDF](https://arxiv.org/pdf/2610.05787)  

**Abstract**: Conversational job recommendation requires jointly modeling semantic relevance, user preferences, eligibility requirements, and the noisy language used in real-world career discussions. These challenges are especially pronounced in low-resource, code-mixed settings, where strict constraint matching can incorrectly eliminate otherwise suitable jobs. We introduce JobCCC, a conversational job recommendation benchmark for Bangladesh comprising 22,410 structured job postings and 988 multi-turn career-advice dialogues derived from regional Reddit communities. Each dialogue is annotated with evolving seeker preferences and linked to a ground-truth job, and is evaluated in semantically equivalent English and Romanized Bangla--English variants. We compare sparse BM25 retrieval, multilingual dense retrieval, and their hard-constraint-filtered counterparts against Weighted Soft-Constraint-Aware Ranking (W-SCAR), our multi-criteria ranking framework that combines lexical relevance, semantic relevance, and graded utilities for experience, location, education, and salary using the Technique for Order Preference by Similarity to Ideal Solution (TOPSIS). Experiments reveal that strict filtering consistently degrades retrieval because incomplete extraction and brittle attribute matching irreversibly remove relevant jobs. W-SCAR avoids destructive pruning and achieves more balanced performance across the two language conditions, obtaining 37.37% and 38.43% Hit@10 on English and Banglish, respectively. The code and dataset are publicly available at \href{this https URL}{GitHub} and \href{this https URL}{Hugging Face}, respectively. 

---
# Beyond Semantic Similarity: Performance and Costs of Agentic Retrieval for Complex Tasks 

**Authors**: Reza Esfandiarpoor, Radek Osmulski, Yauhen Babakhin, Gabriel de Souza P. Moreira, Oliver Holworthy, Jie He, Ronay Ak, Jiarui Cai, Ryan Chesler, Bo Liu, Even Oldridge  

**Link**: [PDF](https://arxiv.org/pdf/2610.05750)  

**Abstract**: Modern information systems, including many agentic workflows, use dense retrieval to explore large amounts of unstructured data. However, dense retrieval relies on surface-level semantic similarity, which is insufficient for increasingly complex search applications. Here, we investigate agentic retrieval that combines the reasoning capabilities of Large Language Models (LLMs) with the efficient corpus exploration of retrievers in a ReAct agentic loop to solve complex retrieval tasks. In our experiments, we show that agentic retrieval is more effective than standard retrieval, improving nDCG@10 by 8.7 points using the same embedding model. Moreover, while specialized retrieval methods struggle on out-of-domain tasks, agentic retrieval is highly generalizable: the same pipeline achieves competitive results on both the ViDoRe v3 and BRIGHT leaderboards. However, this improvement comes at a cost. On average, agentic retrieval takes 107.4 seconds, compared to 0.67 seconds for standard retrieval, and consumes 764.1K input and 5.8K output tokens per query. In short, our study demonstrates the effectiveness of agentic retrieval in modern data systems and motivates future work on more cost-efficient retrieval agents for large-scale deployment. 

---
# Generate What You Can Trust: Content Credibility in Generative Recommenders 

**Authors**: Zhuo Cai, Guanghao Wu, Shoujin Wang, Peilin Zhou, Victor W. Chu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05670)  

**Abstract**: Generative recommendation (GR) represents items with semantic IDs (i.e., discrete token sequences) and generates target item tokens as recommendations. Despite its promising results, existing methods predominantly optimize for accuracy while neglecting the credibility of the recommendations they generate. This oversight inevitably exposes users to uncredible content (e.g., fake news) with serious societal consequences, including user distrust, reputation harm to platforms, and broader social instability. To address this critical yet underexplored challenge, we propose CreGR, the first credible GR model that jointly tackles content credibility across the two core stages of GR: tokenization and generation. In the tokenization stage, we design a new credibility-aware tokenizer that explicitly encourages the model to learn discriminative tokens respectively for credible and uncredible items, thereby disentangling credibility signals at the token level. Building on this, in the generation stage, we propose a novel accuracy-preserving and credibility-oriented generator grounded in discrete diffusion. Specifically, we introduce an asymmetric masking probability reduction strategy that selectively diminishes the contribution of tokens associated with uncredible content to the generation process, while leaving tokens encoding user preference signals unaffected so as to preserve recommendation accuracy. Experiments on three real-world datasets demonstrate the effectiveness of CreGR. 

---
# SCOUT: Supply-Aware Cold-Start Proactive Query Suggestion for Travel Search 

**Authors**: Hao Li, Shashank Reddy, Kedar Bellare, Ashish Jain, Stephanie Moyerman  

**Link**: [PDF](https://arxiv.org/pdf/2610.05619)  

**Abstract**: Generative query suggestion, powered by Large Language Models (LLMs), has become increasingly popular in search and conversational systems to reduce user friction and guide intent formulation. Existing approaches align suggestions with user preferences (e.g., clicks or conversions). This works for open-ended applications like chatbots and personal assistants, where the result space is unconstrained or historical user free-text queries are abundant.
However, applying these methods to travel search presents two limitations. First, travel search is fundamentally constrained by physical inventory; a query (e.g., "romantic beachfront villa") may yield abundant results in Bali but few in Tokyo, so aligning with user preferences is not by itself grounded in what can be offered. Second, travel platforms traditionally rely on faceted search interfaces with no free-text queries. This creates a cold-start problem: without historical query logs there is no demand-side data for alignment, and without a seed query at request time, suggestions must be generated proactively from structured context alone.
To address these challenges, we propose SCOUT, a bootstrapping framework for supply-aware proactive query suggestion. SCOUT overcomes the data gap by substituting missing demand-side user feedback with supply-side system feedback. It treats the search engine as a reinforcement learning environment, deriving a dense reward from the production reranker's query-listing match scores, and optimizes the policy with Group Relative Policy Optimization (GRPO). SCOUT improves inventory match rate (IMR@18) by 12.3% while preserving diversity, matching a compute-intensive best-of-8 policy at zero marginal inference cost and making supply-aware suggestion deployable on a real-time travel search path. 

---
# OpticalRec: Unified Optical Vision-Language Representation for Multimodal Recommendation 

**Authors**: Yueqi Wang, Zitian Guo, Yupeng Hou, Yifei Wang, Kibum Kim, Zhenrui Yue, Shuo Xing, Haodong Li, Heming Xia, Renrui Zhang, Zhengzhong Tu, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2610.05432)  

**Abstract**: Recent advances in vision-language modeling have substantially improved multimodal encoding, retrieval and reasoning. Yet for multimodal recommendation, encoding rich item vision-language semantic interactions remains a long-standing bottleneck, which hampers accurate item representation learning and user-item matching. Mainstream approaches primarily adopt independent encoding of vision and language modality followed by rigid late fusion such as concatenation, inherently omitting native vision-language interactions and introducing cross-modal semantic distortion. To address this challenge, we propose OpticalRec, the first visual-space unified encoding paradigm for multimodal collaborative filtering, a fundamental recommendation setting. Instead of isolated modality-specific encoding, OpticalRec renders item textual metadata as visual glyphs, enabling native image-text interaction within the visual encoder - the perceptual encoding level. The resulting representations are further processed by the language decoder - the semantic encoding level, allowing OpticalRec to exploit the dual-attention mechanism of modern vision-language models that previous encoding methods omitted. OpticalRec's efficacy is theoretically supported by mutual information analysis and empirically demonstrated through superior performance across strong baselines and benchmarks. As a plug-and-play module, OpticalRec (1) introduces minimal cost, (2) is robust against rendered text font, color and layout, etc., and (3) integrates seamlessly into existing multimodal collaborative filtering models. 

---
# Search Engines Never Say No: How Frozen Agents React When the Retrieval Tool Refuses 

**Authors**: Ramraj Chandradevan, Sayontan Ghosh, Vinoth Selvendran  

**Link**: [PDF](https://arxiv.org/pdf/2610.05348)  

**Abstract**: A search tool never says no: it returns its top-k passages even when the index holds no answer, so the agent sees irrelevant text instead of a miss signal. We ask what frozen search agents do when the tool refuses instead. On an index-hole testbed (257 NQ and 300 HotpotQA questions run with and without their gold passages in a 21M-passage BM25 index), seven agents receive one of five refusal wordings. An un-announced one-sentence refusal raises abstention on unanswerable questions from 23% to 97% on average for Qwen3-8B/32B and from 28% to 57% for Claude Haiku 4.5, cutting wrong answers almost one-for-one and beating a system-prompt instruction by 51 points on average. Search-R1 ignores the refusal and fabricates retrievals; Claude Sonnet 5.5 and Opus 5.5 answer from memory (abstention +2 points) and obey a system-prompt directive instead (+16). We also found that wording matters: an explanation beats a bare token; a directive inside the observation is decisive for Haiku; a soft warning is useless. Realistic triggers, from a lightweight score-based predictor to an LLM grounding judge, fall well short of the oracle, and all land on a benefit-versus-signal-quality curve that prices any trigger by its recall at a fixed false-refusal budget: for compliant agents the bottleneck is the detector inside the tool, not the agent, and the curve tells future detector work what each point of recall is worth. 

---
# Quality-Aware Cross-Model Computation Reuse 

**Authors**: Jin Cheng, Xiangxiang Dai, Maoli Liu, Ziyi Han, Zhuohua Li, John C.S. Lui  

**Link**: [PDF](https://arxiv.org/pdf/2610.05285)  

**Abstract**: An intermediate result computed by one model can be reused by other models to perform their tasks. Existing work mainly focuses on practical execution, leaving a theoretical gap in optimizing reuse decisions. This optimization faces two challenges: quality uncertainty, because the effect of reuse on task quality is uncertain across models, and coupled scheduling, because tasks need to share the cost of preparing reusable results. These challenges compound each other: quality must be learned online, but the shared preparation structure makes scheduling NP-hard even with known quality, breaking the key assumption in existing methods. We formulate cross-model computation reuse as an online decision problem and develop the Quality-Aware Reuse Scheduling (QARS) algorithm to address it. For quality uncertainty, QARS learns task-dependent reuse quality from selected, possibly delayed feedback and uses optimistic estimates to guide decisions. For coupled scheduling, it jointly chooses which results to prepare and which tasks should use them, adapting scheduling accuracy to the remaining quality uncertainty. For the considered problem, our analysis separates learning and optimization error in regret and quantifies the tradeoff between scheduling accuracy and computation. Completing the quality-aware stopping rule yields $\widetilde O(\sqrt{T})$ regret while preserving feasibility. Experiments demonstrate the effectiveness of QARS in optimizing cross-model reuse, reducing the combined cost of computation and quality loss by up to 18.0%, and mean regret by 63.9% over the strongest scheduling baseline. 

---
# SearchJev: A Fast and Calibrated System-1 Model for Search Agents 

**Authors**: Congfeng Cao, Lipeng Zuo, Konstantinos Papakostas, Qiwei Xu, Songwei Xu, Lun Zhou, Zhaochun Ren, Yougang Lyu, Xiaohui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2610.05107)  

**Abstract**: Search agents repeatedly make short decisions about relevance, evidence sufficiency, and search actions. Using generative language models for these decisions introduces latency and unreliable confidence. We present SearchJev, a fast and calibrated System-1 model that separates search decisions from System-2 reasoning and generation. Given a search state and a decision schema, SearchJev directly scores legal options without autoregressive output generation. We propose Soft-Label Learning for Calibrated Decisions (SLCD) to learn decision probabilities from uncertain supervision and calibrate their confidence. In a dual-system search agent, SearchJev handles short decisions and delegates uncertain judgments to System 2, which retains planning, query generation, and answer composition. We also introduce SearchDecision-Bench, a benchmark unifying six types of search decisions for training and evaluation. On SearchDecision-Bench, SEARCHJEV improves decision quality over same-size Qwen3.5 autoregressive models, achieves 5.2-5.3 times faster decisions, and reduces average expected calibration error by 41-74%. On BrowseComp-Plus, the dual-system agents achieve a 3.7-4.7 times speedup in active search time while improving answer accuracy from 45% to up to 54%. 

---
# Agentic RAG Evaluation: Budget Allocation Across Questions, Trajectories, and Reads 

**Authors**: Jingjie Ning, Xueqi Li, Yibo Kong  

**Link**: [PDF](https://arxiv.org/pdf/2610.05034)  

**Abstract**: Evaluation budgets in agentic retrieval-augmented generation span questions, search trajectories, and repeated answers. We measure allocation precision, reading efficiency, and cost boundaries using a retrieval-feedback comparison on HotpotQA and MuSiQue. At 34.14--34.39M model tokens, broader question coverage lowers standard error by 33\% versus five reads and 12.6\% versus three trajectories. Archived nested and Q-only forecasts predict these allocations within 4.0\% and 3.5\%, respectively. Depth subsets establish no clear forecasting advantage beyond the two-trajectory audit. One-read variance penalties relative to the fitted optimum at the same token budget are 0--9.9\%, with substantial Pro uncertainty. Under recorded model fees, more questions beat more trajectories at search prices of \$0--1 per 1,000 requests; question-versus-read fee rankings remain unresolved. Temperature zero cuts answer disagreement from 14.3\% to 3.4\% while comparison precision stays similar. \par\medskip\noindent\textbf{Keywords:} Agentic RAG; Evaluation budget; Generalizability theory; Repeated sampling. 

---
# Do We Still Need Gazetteers in the Era of LLMs? Chaining Retrieval with a Spatial Neuro-Symbolic Index 

**Authors**: Horde-Vo Alexis, Duckham Matt, He Estrid  

**Link**: [PDF](https://arxiv.org/pdf/2610.05028)  

**Abstract**: Geographic information retrieval (GeoIR) tasks require systems to interpret ambiguous toponyms for downstream applications. Traditionally, toponym resolution relies on gazetteers to provide an explicit index of place entities and spatial relationships. Recently, gazetteer-free approaches seek to reduce dependence on handcrafted searches: dense retrieval utilizes text encoders to capture rich context, moving beyond the limitations of lexical search. However, text encoders implicitly assume that learned representations can function as reliable spatial-semantic indexes. In this paper, we evaluate this assumption through a spatial-semantic indexing setup: given a contextualized toponym mention, we retrieve the corresponding gazetteer entity represented by text derived from a gazetteer knowledge graph. We benchmark five frozen text encoders under two retrieval strategies: brute-force nearest-neighbor retrieval over entity representations, and a neuro-symbolic hierarchical beam search that constrains retrieval (i.e. chaining the search with gazetteer hierarchy). Experimental results reveal a distinct coarse-versus-fine trade-off. Unconstrained dense retrieval frequently incurs catastrophic spatial errors. Conversely, hierarchical constraints improve coarse geographic grounding, but still yield limited benefit for fine-grained localization metrics: vanilla text encoders fail to capture the fine-scale spatial fidelity encoded in gazetteers. Our code is publicly available at: this https URL 

---
# ModelLakeFishing: Efficient Retrieval over Million-Scale Model Lakes 

**Authors**: Xiaoyang Liu, Zhengyuan Dong, Renée J. Miller  

**Link**: [PDF](https://arxiv.org/pdf/2610.04904)  

**Abstract**: Open model lakes may contain millions of reusable models, making it costly to identify suitable models for a new dataset. We present ModelLakeFishing, a model-retrieval framework for queries specifying a target dataset, prediction task, and evaluation metric. It consolidates metadata and historical evaluations into a model-dataset-task evidence graph, learns model and query embeddings with a relation-aware graph encoder, and indexes model embeddings using Hierarchical Navigable Small World (HNSW) search. At query time, HNSW retrieves 1,000 candidates without scoring every model, after which a training-side task prior reranks candidates for the requested metric and returns the top 10. We evaluate on a lake of 3,016,439 models and 247,803 observed model-dataset performance pairs using three root-aware splits that hold test performance edges out of representation learning and retrieval. ModelLakeFishing achieves a mean eligible-query gold@10 of 0.2968, recovering the observed-best model in the top 10 for 29.68% of eligible queries and retaining 93.47% of the gold@10 of an exhaustive baseline using the same scoring and reranking procedure. Given precomputed query embeddings, retrieval and reranking take 0.747 ms median and 1.102 ms at the 95th percentile. These results demonstrate efficient retrieval over million-model lakes from sparse relational evidence. 

---
# Query Generation with Direct Preference Optimization for Document Expansion in E-commerce Search 

**Authors**: Kaihao Li, Feng Liu, Juexin Lin, Xunfan Cai, Zhen Yang, Tony Lee, Ciya Liao  

**Link**: [PDF](https://arxiv.org/pdf/2610.04352)  

**Abstract**: Doc2Query, a popular document expansion technique, leverages sequence-to-sequence models to generate relevant queries, effectively addressing the "vocabulary mismatch" problem in information retrieval. However, these models often suffer from generating either hallucinations unrelated to the document or repetitive content already present in the document. Training sequence-to-sequence models to produce high-quality, novel, and relevant tokens remains a significant challenge. To address these issues, we introduce a novel approach, QGDPO, that employs direct preference optimization (DPO) to guide the generation process. We first fine-tune a base sequence-to-sequence model and subsequently utilize a relevance model to score its predictions. Based on these scores, we construct pairs of winning and losing predictions as relevance preferences for the DPO training. Furthermore, we enhance our pipeline by using the relevance model to filter out poor predictions, retaining only the most relevant generated content for indexing. QGDPO effectively eliminates 50% of irrelevant predictions comparing against Doc2Query baselines, while the relevance filter removes an additional 14.61%. This feature has been successfully deployed to production on this http URL for full traffic, with a substantial improvement in relevance and user engagement. 

---
# From Valid to Useful: Post-Verification Acquisition for Recursive Self-Improving Recommendation 

**Authors**: Tonmoy Hasan, Taylor Foust, Shao Tang, Leonardo Neves, Aman Gupta, Hiroto Udagawa, Helder Dias, Daniel Silva, Rohan Ramanath  

**Link**: [PDF](https://arxiv.org/pdf/2610.04302)  

**Abstract**: Sequential recommenders can generate synthetic interaction sequences and retrain on the augmented corpus in a recursive self-improvement loop. To limit error accumulation, current methods verify each generated sequence remains predictive of the user's real interactions and discard those that drift away from it. Verification does not, however, determine which verified sequences should train the next model. With every verified sequence used for training, source sequences yielding more verified sequences or longer continuations have more influence, although neither quantity indicates how much those sequences will help the next model.
We formulate the decision of which verified sequences are used to train the next model as \emph{post-verification acquisition} and introduce {\bf Disagreement-Aware Recursive Self-Improving Recommendation (DA-RSIR)}. DA-RSIR caps each source sequence's contribution and ranks its verified sequences by how much the model's predictions disagree over their augmented interactions. It uses a score derived from Bayesian Active Learning by Disagreement (BALD) and estimated with Monte Carlo (MC) dropout.
DA-RSIR requires no extra labels, teacher model, or quality scorer. Across four datasets, three recommender models, and two metrics, it improves on the retain-all approach in all $24$ comparisons and attains the highest mean in $23$ of $24$ overall; the aggregate improvement is statistically significant on both metrics. A single DA-RSIR round exceeds the retain-all approach's best gain over five recursive rounds. These findings establish post-verification acquisition as a separate control point in recursive self-improvement, separating which sequences pass verification from which verified sequences are used to train the next model. 

---
# Multimodal Dual-Encoder Retrieval for Automated ICD Coding 

**Authors**: Abhinav Bohra, Anuj Bohra  

**Link**: [PDF](https://arxiv.org/pdf/2610.04263)  

**Abstract**: Accurate International Classification of Diseases (ICD) coding is crucial for large-scale clinical research, documentation, and billing. There are three primary problems with current ICD prediction methods: (1) They are unable to comprehend multimodal patient data because they rely on either structured EHR data or unstructured clinical notes. (2) They also struggle with scalability to a larger amount of ICD codes (9K+ codes in ICD-9), as traditional classifiers need dense output layers and often do not generalize well to long tail rare diseases. (3) They lack transparency for clinical use. To address these challenges, this research proposes a two-stage framework that first retrieves ICD codes using a multimodal dual-encoder retrieval model, where structured and unstructured patient data are integrated through gated fusion. The second stage refines the top-k retrieved candidates with an LLM-based re-ranker that provides ranked codes with clinically relevant explanations. Our experiments show that the proposed approach improves Micro-F1 and Precision over a multimodal dual-fusion classifier baseline. These improvements demonstrate that combining a gated multimodal retrieval system with LLM-based re-ranking is a practical alternative to dense multi-label classification for automated ICD coding. 

---
# FICO: Find-Then-Compute for Corpus-Level Spreadsheet Question Answering 

**Authors**: Sandarsita Guntupalli, Lu He, Kang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.03958)  

**Abstract**: Question answering over spreadsheet collections requires finding the correct workbook and computing over complete tables. We introduce Find-then-Compute (FiCo), which retrieves document summaries, disambiguates similar workbooks, and executes constrained Structured Query Language (SQL) over the selected full table. On DataBench (80 datasets, 1,810 questions), FiCo reaches 76.2% accuracy: 9.9 points above a strong TableRAG-style baseline on the same frozen workbook choices (66.3%) under the tracks' prespecified evaluators, and 63.4 points above prefix RAG (12.8%). On 508 MiMoTable questions, FiCo reaches 79.7%, versus 22.2% for prefix RAG. Giving the strong baseline the gold workbook raises it from 66.3% to 76.3%, exposing a 10.0-point source-selection cost under fixed compute. Despite 95.1% document recall and 98.5% executable SQL, only 81.3% of questions execute on the gold workbook. FiCo's advantage comes from integrating semantic source selection with exact, schema-grounded computation. 

---
# Learning Robust Personalized Prompts for LLM-Driven Sequential Recommendation 

**Authors**: Xiaolin Zheng, Qiyong Zhong, Jiajie Su, Xiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.03923)  

**Abstract**: LLM-driven sequential recommendation formulates next-item prediction as autoregressive generation conditioned on natural-language prompts. However, minor wording changes in semantically equivalent prompts can cause substantial performance fluctuations, undermining robustness and requiring costly manual prompt engineering. Continuous prompt learning reduces template dependence but faces two interacting challenges: shared task-level instructions lack user-specific reasoning guidance, while gradient updates can push continuous prompts outside the LLM's effective semantic space. Injecting personalized signals can further amplify this semantic drift. To address these challenges, we propose LRPRec, a learnable prompting framework that initializes continuous instruction prompts from discrete templates and introduces two complementary mechanisms. Personalized prompt injection encodes user behavior into a preference embedding and additively injects it into shared prompts, enabling parameter-efficient user-level adaptation. A semantic drift constraint regularizes the shared prompts within a trust region around their initialization anchors to preserve semantic validity during optimization. By constraining the shared component while allowing additive personalization, LRPRec decouples stability from expressiveness. Extensive experiments on three benchmark datasets demonstrate consistent improvements over strong baselines while eliminating the need for manual tuning of background and task inference templates. 

---
# Optimal compression with quantum retrieval 

**Authors**: Shyam Dhamapurkar, Mohit Garg, Manaswi Paraashar, Jaikumar Radhakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2610.06702)  

**Abstract**: We consider the following data compression problem. Given a string $x \in \{0,1\}^m$ of Hamming weight at most $n$, compress it into a shorter string $y \in \{0,1\}^s$ so that any bit $x_i$ of $x$ can be retrieved without any error using at most $t$ quantum queries to the standard oracle encoding of $y$. If queries are allowed to be adaptive we show how optimal compression up to a logarithmic factor can be achieved. If the queries are required to be made non-adaptively, we show schemes whose space is optimal in its dependence on $m$ except for a logarithmic factor, and is at most quadratically worse when compared to the optimum in its dependence on $n$. 

---
# Mind the Execution Gap: Action-Semantic Mismatch in World-Model Control 

**Authors**: Shengtao Wen, Xiang Chen, Yu Tian, Lingbing Guo, Lina Gong, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.06582)  

**Abstract**: World-model controllers rely on action-conditioned dynamics for prediction and planning, yet real control systems often execute commands asynchronously due to communication delay, packet loss, reordering, and actuator buffering. We study how asynchronous execution changes the action semantics assumed within world-model controllers, rather than treating it only as an external control disturbance. Through controlled interventions, we identify two architecture-dependent failure modes: planning-based controllers such as TD-MPC2 suffer from a future-action timeline mismatch between imagined and executed action sequences, while recurrent world models such as DreamerV3 can attribute observed transitions to commands that were not actually applied. Our analysis shows that TD-MPC2 requires the correct future action sequence during latent dynamics rollout, whereas DreamerV3 requires timely attribution of each transition to the action that generated it. Based on these findings, we introduce two lightweight execution-consistent interfaces, Future-Sequence for TD-MPC2 and Applied-Action Feedback for DreamerV3, that correct these mismatches without modifying the pretrained world models. Experiments across delays, packet loss, reordering, multiple control domains, measured network traces, and a process-separated asynchronous stack consistently support both diagnoses and the corresponding architecture-specific corrections. 

---
# OntoInk: Interactive Ontology Visualization, Validation, and Reasoning 

**Authors**: Ebrahim Norouzi, Jörg Waitelonis, Harald Sack  

**Link**: [PDF](https://arxiv.org/pdf/2610.05945)  

**Abstract**: Ontology documentation, visualization, and validation are usually carried out with separate tools. This split workflow slows down development and makes knowledge transfer harder. We present OntoInk, an open-source MkDocs plugin that brings these activities together. Within a single documentation-as-code pipeline, OntoInk renders interactive ontology diagrams, validates instance data against SHACL shapes, runs OWL\,DL reasoning, and supports inline Turtle editing. General-purpose diagram plugins for MkDocs cannot parse RDF, dereference IRIs, overlay SHACL constraints, or run OWL reasoning. Compared with standalone ontology visualization tools, OntoInk embeds interactive and editable diagrams directly into documentation pages. A live demo and source code are available at \url{this https URL}. 

---
# Protocol-Sensitive Evaluation of Log Anomaly Detection: Component Costs and Target-Access Sensitivity on HDFS and BGL 

**Authors**: Hang Xiao, Janet Sung, Zhaoyi Li, Gangzhen Qian, Chuhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.05807)  

**Abstract**: Protocol choices can change the conclusions drawn from log anomaly detection benchmarks even when detector settings are fixed. We present a joint empirical study of split construction, representation visibility, and component costs using six fixed count, sequence, and semantic configurations on Hadoop Distributed File System (HDFS) and Blue Gene/L (BGL) logs. Random splits place several configurations near the average-precision ceiling, whereas group-disjoint HDFS and chronological BGL evaluation produce lower scores and different observed orderings. At a fixed BGL cutoff, parser choice spans 0.124 in semantic XGBoost mean average precision while preserving its lead over count XGBoost; the earliest rolling period reverses that ordering. A two-factor cross-system ablation contrasts source-only representations with offline transductive access to unlabeled target templates through the representation corpus and inverse document frequency: HDFS-to-BGL mean average precision moves from 0.191 with source-only access to 0.325 with union-corpus, target-IDF access, and the intermediate conditions reveal direction-dependent interactions in average precision and retrieval at fixed review budgets. Component-level profiling separates parsing and representation costs from classifier training, prediction, and storage. Together, these findings connect detector comparisons to the test population, preprocessing state, visible information, and measured pipeline stages, and identify the protocol fields needed alongside a score to support interpretable comparisons of log anomaly detection accuracy and resource use. 

---
# PACMI: Provenance-Aware Cascading Memory Invalidation for Long-Term LLM Agents 

**Authors**: Yiqi Wang, Jiaqi Liu, Jiaqi Zhang, Zhangkai Wu, Yiqun Duan, Mingkai Zheng, Taotao Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.05732)  

**Abstract**: LLM agents rely on long-term memory to retain and reuse information when performing tasks over long horizons. Existing methods provide limited support for handling memories that become outdated as new observations or domain evidence arrive. Such outdated memories may remain semantically relevant, continue to affect dependent records, and retain value as historical evidence. This calls for two capabilities: dependency tracking to identify downstream effects and historical preservation to retain useful past records. We propose Provenance-Aware Cascading Memory Invalidation (PACMI), a framework that represents memories and new evidence in a provenance graph with typed dependency edges. PACMI assigns records to a four-state validity lattice, propagates validity changes to dependent memories, and uses the resulting states for retrieval and stale-premise detection. We also introduce a diagnostic benchmark with 100 cases and 300 queries across five domains. The evaluation separates node, context-, and answer-level performance. PACMI achieves the highest final-answer accuracy on this benchmark, and its paired difference from the strongest baseline is significant under an exact McNemar test. The premise checker achieves perfect precision, recall, and F 1 on the controlled query distribution. Cascading propagation primarily improves memorystate correctness: removing it increases final-answer errors from 3 to 11, but the paired difference does not reach the 0.05 significance threshold. Code and data will be made publicly available. 

---
# Errors of LLM-Assisted Literature Retrieval in Environmental Science: A Comparison Study of Abstract versus Full-text Based Prompts 

**Authors**: Yanjun Chen, Yongfeng Zhang, Lanjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.05690)  

**Abstract**: Large language models (LLMs) are increasingly used for literature search and synthesis. However, it is unclear whether they retrieve accurate bibliographic information in environmental science. Therefore, we quantitatively compared the errors of widely used LLM platforms in retrieving references related to original articles from five leading environmental science journals (Energy and Environmental Science, Nature Sustainability, Nature Climate Change, Lancet Planetary Health, and Environmental Science and Technology) published in 2024 to 2025. Claude, ChatGPT, Grok, DeepSeek, Perplexity, and Gemini were used as the LLM platforms. LLMs retrieved 10 references for each of the 50 randomly selected original article using either the article's abstract or its full-text as prompt. The retrieved references were subject to a multimetric score ratio combining validity of bibliographic data, Google Scholar link, digital object identifier, Scopus Electronic Identifier and relevance score (cited by or being the index paper), and the proportion of complete fabrication that failed all metrics. Abstract-only prompt yielded significantly higher accuracy than full-text one. This advantage was confirmed in multilevel mixed-effect multivariable regression after adjusting for journal, platform, and output order. Source journal and the position of a reference within the output list were also independently associated with retrieval accuracy, with lower-listed references associated with lower accuracy. These findings suggest that LLM assisted literature retrieval in environmental science remains moderately accurate and overall inconsistent, varying significantly by platform, journal, prompt type, and output position. Abstract-based prompting, as task-aligned information compression, may outperform full-text one in literature retrieval. Caution should be used when generalizing our findings. 

---
# Cut Binary Cross Entropy: Efficient Large-Vocabulary Loss and Gradient Kernels for Sequential Recommendation 

**Authors**: Yaoyiran Li, Haowen Ning, Mohamed Hammad  

**Link**: [PDF](https://arxiv.org/pdf/2610.05559)  

**Abstract**: Industrial sequential recommender systems operate over massive item catalogs (e.g., 10^5--10^7 items). Multi-label recommendation models are trained with Binary Cross-Entropy (BCE) loss over the full vocabulary, but standard BCE materializes a dense [B, N, V] logits tensor in High Bandwidth Memory (HBM), incurring prohibitive $O(BNV)$ memory and fatal Out-Of-Memory (OOM) errors. While chunked loss optimizations exist for Softmax Cross-Entropy in LLMs, large-scale multi-label BCE optimization remains unexplored across deep learning ecosystems.
We propose CutBCE, an exact, hardware-accelerated BCE loss and gradient operator implemented in JAX and Pallas for large-vocabulary workloads. CutBCE introduces (1) an exact fused reformulation evaluating dense background loss and sparse target corrections; (2) a custom Vector-Jacobian Product (VJP) with a dedicated Pallas TPU backward kernel computing logit tiles on-chip in both passes so logits and their gradients never reside in HBM; (3) dynamic VMEM budgeting and sharding-aware collective hoisting for distributed meshes; and (4) count-based zero-overhead training metrics. On single-chip TPU v5e/v6e mini-benchmarks, CutBCE eliminates OOM errors with up to 91.9% speedup. On 8-chip TPU slice training for multi-label SASRec with 876k items (Yambda-50M), CutBCE reduces peak HBM by 65.7% (>14 GiB saved per chip) and increases training speed by 225.9% with comparable accuracy. CutBCE is open-sourced at this https URL. 

---
# MemStrata: 95% and 90.91% Source-Aware Accuracy on LongMemEval-500 and LoCoMo-1540 with a Local Qwen 3.8 27B Q4_K_M Reader 

**Authors**: Neeraj Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2610.05343)  

**Abstract**: An adequate conversational answer may differ from a short or incomplete benchmark reference. To measure adequacy against the recorded history we prefer source-aware grading, in which the judge checks the reference against the full source before assessing system-blinded answers; original reference-only grading is reported alongside. With a local Qwen 3.8 27B Q4_K_M reader and a 24,000-token evidence ceiling, MemStrata CL1 scores 475/500 (95.0%) on LongMemEval-S and 1,400/1,540 (90.91%) on LoCoMo categories 1-4 under source-aware GPT-5.5 adjudication, against 463/500 (92.6%) and 1,205/1,540 (78.25%) under reference-only grading of the same answers. It preserves a retrieval backbone and adds nonduplicated, dated, speaker-attributed source spans. A same-reader full-history control with about 4.7 times the evidence scores 464/500 reference-only and 470/500 (94.0%) source-aware; neither difference is decisive. Keyword-only selection at the same budget scores 425, and a matched-reader Letta arm 438. On LongMemEval-M, where the packet holds about 1.6% of each history, MemStrata CL1 scores 427/500, with losses concentrated in multi-session and temporal questions. On 300 BEAM-1M questions it outscores dense retrieval, 0.738 to 0.706 (Wilcoxon p = 0.011). A same-seed replay of unchanged requests changed 1.5-2.3% of labels. On identical packets GLM 5.3 flash is non-inferior within 3 points (462 versus 463); Muse Spark 1.3 did not show non-inferiority on 269 questions. None of four pre-registered interventions met all of its registered advancement or feasibility criteria. Signed read-side artifacts support inspection but do not regenerate the private retrieval pipeline. The superiority of source-aware grading to human adjudication is not established, and development exposure, automated-judge dependence and the absence of held-out data preclude an independent-replication or leaderboard claim. 

---
# Cross-Modal Contrastive Learning for the Retrieval of Immunotherapy-Associated Molecular Signatures from Histopathology 

**Authors**: Sigrid Vila-Bagaria, Mar Teixidó, Miquel Piñol, Felip Vilardell, Robert Montal, Veronica Vilaplana  

**Link**: [PDF](https://arxiv.org/pdf/2610.05157)  

**Abstract**: Gastric Adenocarcinoma is a leading cause of cancer mortality. Although "Inflamed/Non-Inflamed" subtypes have been proposed to predict immunotherapy response, their identification relies on a costly 10-gene RNA signature. We propose a Cross-modal Contrastive Multiple Instance Learning (CCMIL) framework for cross-modal retrieval, imputing these molecular signatures directly from standard Hematoxylin & Eosin (H&E) slides. By leveraging a supervised contrastive objective, CCMIL aligns visual morphological patterns with molecular phenotypes into a shared latent space. This establishes an interpretable search-by-case retrieval engine, enabling pathologists to query a whole slide image to surface transcriptomically coherent neighbors and approximate RNA signatures without genomic sequencing at inference. Our results demonstrate that this retrieval-first approach captures the continuous phenotypic spectrum of tumor inflammation and yields clinically interpretable attention heatmaps. Furthermore, the learned representation also supports competitive downstream classification, providing a practical molecular pre-screening strategy. 

---
# Periscope: Extending Frozen Language Models Beyond Their Context Window 

**Authors**: Mohamed Eltahir, Anas Obayd, Raed Rashid, Abdulrahman Alghamdi, Abdulrahman Mousa, Abdallah Ahmed, Tanveer Hussain, Naeemullah Khan  

**Link**: [PDF](https://arxiv.org/pdf/2610.04047)  

**Abstract**: A language model reads long text in one quadratic forward pass, stops at the context window, and loses accuracy with length before reaching it. We ask whether the read can be factorized when deciding over a finite set: which document is relevant, which option is supported, which passage is the evidence. Periscope, a training-free inference method, arranges the $N$ chunks of a text on a $K{\times}K$ grid with $K{=}\lceil\sqrt{N}\rceil$ and asks a frozen model the same question about $K$ local spans of consecutive chunks and $K$ strided spans that sample the whole text, reading the log-odds of every answer at one token. Each answer takes its best local and strided score, and scoring every chunk by its two spans gives an evidence map at no further cost, whose peak is the chunk behind the answer. Every probe is about $\sqrt{sc}$ tokens for a text of $s$ tokens and chunk size $c$, so a window of $W$ tokens reaches $W^{2}/c$ tokens at $s^{1.5}$ cost. The map replaces the long read. On LongBench v2, reading only the $K$ chunks the map ranks highest, 9k tokens, matches the same model's best window read across windows from 32k to 1M tokens, and on InfiniteBench, where the median context is 150k tokens, it leads the best window read by 5 points. The same map ranks BRIGHT's long-document corpora with the best NDCG@10 of six methods. Each call caches only one probe, so a 27B model reads 4.5M-token contexts on one 80GB GPU, where a single pass would need 296GB of cache. A long read then needs a GPU that holds the model, not one that holds the text. 

---
# Learning Subject-Specific Anatomical Representations via Manifold Expansion: Application to Accelerated Multi-Contrast MRI 

**Authors**: Ruimin Feng, Wanyu Bian, Albert Jang, Zachary Stewart, Fang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.04028)  

**Abstract**: Clinical MRI routinely acquires multiple contrast-weighted images of the same anatomy for complementary tissue characterization. However, current accelerated MRI methods typically reconstruct each contrast independently, without fully exploiting shared anatomical information. This work aims to learn anatomical representations invariant to contrast-dependent appearance for reconstruction of accelerated multi-contrast MRI. We propose MAX (MAnifold eXpansion), a subject-specific framework that learns anatomical representations from a single fully sampled reference contrast. To address the under-constrained separation of shared anatomy and contrast-dependent components from a single image, MAX expands the multi-contrast manifold using anatomy-preserving intensity augmentations. A disentangled implicit neural representation models augmented samples using shared spatial coordinates for anatomy and spatially invariant coordinates for contrast appearance. The learned anatomical representation is then fixed, with the contrast representation adapted to the undersampled target data, followed by unrolled refinement. Theoretical analyses further provide insight into the disentangled representation learning and explain how the learned anatomical representation improves the target contrast reconstruction. At R = 8 for brain MRI and R = 6 for knee MRI, MAX achieves the highest mean PSNR and SSIM across all tasks, improving PSNR by more than 1 dB over the strongest baseline for both brain contrasts. MAX more faithfully recovers subtle anatomical and pathological structures and remains robust to inter-contrast motion, structural heterogeneity between reference and target contrasts, and measurement noise. Therefore, MAX provides a general strategy for leveraging high-quality reference scans in accelerated MRI and has the potential to be extended to other reference-assisted MRI inverse problems. 

---
# Fine-Grained Emotion Classification from Mobile App Reviews: An Empirical Study with Large Language Models 

**Authors**: Quim Motger, Carlota Catot, Marc Oriol  

**Link**: [PDF](https://arxiv.org/pdf/2610.03802)  

**Abstract**: Context: Fine-grained emotion classification of mobile app reviews enables requirements engineering activities that go beyond polarity-based opinion mining, including emotionally informed issue prioritisation and feature-oriented feedback analysis. However, automatic fine-grained emotion extraction from app reviews remains understudied. Objectives: Building on a previously published annotation framework and human-labelled ground truth adapted from Plutchik's taxonomy, this paper investigates how large language models can be leveraged for automatic multi-label emotion classification under severe class imbalance. Methods: We compare encoder-only fine-tuning under multi-label and binary-ensemble formulations, decoder-only zero- and few-shot prompting across open-source and proprietary models, and a catalogue of imbalance mitigation strategies (loss reweighting, resampling, generative data augmentation), with the synthetic-review generator and prompting strategy selected via an intrinsic augmentation-utility ranking. Results: Fine-tuned encoders trail the best decoder-only few-shot prompting (macro-F1 0.642) by a wide margin at baseline (multi-label: 0.387; binary ensemble: 0.450); pairing the best multi-label encoder with generative data augmentation and positive-weighted loss closes most of this gap (+0.204) at up to three orders of magnitude lower inference latency than the decoders, with the largest gains on the rarest emotions, from undetected to gains of up to +0.501 F1. Conclusion: Large language models make fine-grained, multi-label emotion classification of app reviews feasible for requirements engineering pipelines, with modest macro-F1, and the best formulation and mitigation strategy are backbone- and formulation-dependent. We release the experimental pipeline, synthetic corpora, and fine-tuned checkpoints for replication and reuse. 

---
