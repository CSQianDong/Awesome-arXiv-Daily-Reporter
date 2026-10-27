# ACARE: Agent-Context-Aware Reranker for Iterative Search 

**Authors**: Sahel Sharifymoghaddam, Yauhen Babakhin, Reza Esfandiarpoor, Oliver Holworthy, Jimmy Lin, Bo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.35840)  

**Abstract**: In agentic search, a document can match an agent-generated sub-query without helping answer the user's original question. Conventional rerankers score documents against the sub-query alone, overlooking the broader task and the agent's reasoning. We introduce the Agent-Context-Aware Reranker (ACARe), a compact 4B LLM trained to rerank documents using this context when available. Across multiple benchmarks and agent families, ACARe improves retrieval quality over its base reranker while reducing search effort. These benefits extend to answer generation: ACARe improves accuracy and reduces searches over no reranking, while matching or exceeding the base reranker's mean accuracy with more precise evidence. It also helps smaller agents narrow the retrieval-quality gap to larger agents, showing how better retrieval tools can make smaller agents more effective. The model, training data, and code will be released soon. 

---
# Do Similarity Measures Measure Similarity? How User Activity Creates Phantom Correlation in Recommender Systems 

**Authors**: Marco Favier, Noah Daniëls, Jens Leysen, Sander Marinus, Bart Goethals  

**Link**: [PDF](https://arxiv.org/pdf/2610.35765)  

**Abstract**: Item-item similarity is one of the oldest and most foundational signals in recommender systems: two items are assumed to be similar when the same users consume them. Despite the sparsity and scale of interaction data, the implicit assumption across the field is that item co-occurrence reflects genuine shared preference, capturing what users like together. In this paper, we demonstrate that a substantial fraction of observed item-item similarity is an artifact of variance in user activity rather than shared taste. Even if every user consumed items completely independently, differing only in their total consumption volume, standard similarity measures still detect strong associations between items. We term this phenomenon phantom correlation: spurious item association that emerges strictly from heterogeneous user activity in the complete absence of preference structure. Starting from minimal formal assumptions, we show that independent consumption under varying activity is described by a Rasch model, and we derive how the resulting association depends on item popularity for each similarity measure. By comparing our theoretical baseline against real-world benchmark datasets, we reveal the extent to which popular similarity measures reflect phantom correlation, which bears on how item co-occurrence is interpreted and used in collaborative filtering. 

---
# Towards a Unified Framework of Optimal Input-Side Dimensional Configuration for Deep CTR Prediction 

**Authors**: Chaohua Yang, Fuyuan Lyu, Xing Tang, Dugang Liu, Xiuqiang He, Zhong Ming  

**Link**: [PDF](https://arxiv.org/pdf/2610.35695)  

**Abstract**: Deep click-through rate (CTR) prediction has achieved substantial success through increasingly expressive feature interaction architectures. However, predictive performance also depends on how input-side information is configured before interaction modeling. Existing studies have shown that selecting informative samples, features, or embedding units can improve prediction accuracy. Nevertheless, they primarily optimize these selection mechanisms independently along a single input-side dimension, overlooking their shared basic form and the inherent dependencies across input levels, thereby limiting the effectiveness of input-side dimensional configuration. To address this issue, we propose OptIDC, a unified framework for Optimal Input-side Dimensional Configuration in deep CTR prediction. OptIDC jointly learns sample, feature, and embedding-unit level configurations within a unified framework by integrating their corresponding selection mechanisms. To explicitly coordinate selection decisions across different levels, OptIDC further introduces a selection-weighted Optimal Transport mechanism that aligns coarse- and fine-grained selections via transport over finite measures, thereby alleviating cross-level selection conflicts while preserving the semantics of each input level. Moreover, OptIDC is model-agnostic and can be seamlessly integrated into existing CTR models. Extensive experiments on four real-world benchmark datasets demonstrate that OptIDC consistently achieves higher AUC than existing input-side selection methods, validating the effectiveness of unified input-side dimensional configuration. 

---
# Cross-Domain Sequential Recommendation with Representation-Level User Transfer 

**Authors**: Abdulaziz Samra, Nikita Sukhorukov, Alexey Grishanov, Alexey Vasilev, Evgeny Frolov  

**Link**: [PDF](https://arxiv.org/pdf/2610.35550)  

**Abstract**: Cross-domain sequential recommendation aims to improve next-item prediction in a target domain by leveraging user behavior from auxiliary source domains. However, real-world deployment faces two critical challenges: business constraints that isolate interaction data across domains, and user heterogeneity, where many users lack cross-domain activity. This makes existing cross-domain methods difficult to apply. In this paper, we propose a simple and practical framework for cross-domain sequential recommendations that transfers information exclusively through learned user representations, without sharing raw interaction data or item embeddings across domains. We introduce a lightweight adapter that projects the source-domain user representation into the target-domain embedding space and integrates it with the target interaction sequence. Importantly, when source data is unavailable (i.e., the user observed only in the target domain), the model seamlessly defaults to a robust target-only sequential recommender, retaining benefits from joint training across all users. Extensive experiments were conducted under two setups; one targets benchmarking against previous baselines, while the other focuses on realistic and scalable conditions. Both sets of experiments demonstrate consistent performance gains for shared and unshared users. 

---
# Hypergraph-Enhanced Evidence Selection for Long-Context Multimodal Document QA 

**Authors**: Wenxin Zhao, Jiachuan Wang, Xinyi Zhu, Jiake Ge, Haoyang Li, Yongqi Zhang, Shuangyin Li, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.35542)  

**Abstract**: Long-context multimodal document question answering (QA) has become an important yet challenging task for modern vision-language models (VLMs), as it requires reasoning over evidence dispersed across pages and modalities within limited input budgets. Current pipelines adopt retrieval-augmented generation (RAG) to select multimodal evidence for downstream VLMs. However, existing systems mainly rely on static selection rules over fixed-granularity retrieval units such as pages or blocks, resulting in information loss during retrieval and suboptimal performance in downstream reasoning. To alleviate these issues, we formulate the core task as budget-aware evidence selection over structured document evidence and propose HyperDoc, a document RAG framework that constructs compact yet sufficient evidence while preserving document relations. Specifically, HyperDoc constructs a document hypergraph to explicitly capture structural, logical, and cross-modal relationships among multimodal document elements. Over this structured evidence space, HyperDoc performs adaptive cost-aware hyperedge selection based on query intent and input budget. Extensive experiments on three representative benchmarks demonstrate that HyperDoc outperforms various types of state-of-the-art baselines. 

---
# Beyond "[at]": A Large-Scale Measurement of Email Address Presentation on Computer Science Scholars' Homepages 

**Authors**: Aichun Huang, Xiang Li, Zheli Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.35499)  

**Abstract**: Publishing an email address makes a scholar reachable, but also exposes it to automated collection. Yet how email addresses are presented on the Web has not been measured systematically across their content, form, and location. We measure it across 32,413 homepage records of computer-science scholars listed in CSRankings with MailScope, which preserves raw and rendered HTML, extracts addresses with a three-stage cascade (rules, then large language models, then a crawling agent), and labels each address. Among 19,517 reachable homepages, we find an address for 17,712 scholars (90.8%). Our measurement yields four main findings. (i) Presentation is diverse: only 53.8% publish plain content, the others spread over 23 subtypes of symbol substitution, code, and semantic description, and 5.5% of scholars' addresses appear only after rendering. (ii) Presentation follows hosts rather than individuals: scholars on one host share a content class in 67.9% of pairs, compared with 38.8% across hosts of one institution, whereas research output and institution rank are barely related to presentation. (iii) Content and form are coupled through code-based links, and 6.2% of scholars with non-plain selected presentations have plain-text copies of the same address elsewhere. (iv) Plain addresses are more common behind passed bot challenges, and 36.1% of redirecting homepages no longer lead to scholars. Human assessment yields 94.4% weighted exact-set agreement on 143 judgeable pages, complemented by three 500-page review rounds with human verification. Presentation is diverse but clusters by shared host, and by our label-based estimate plain-string matching misses nearly half of the addresses. 

---
# Read the Route: Attention-Steered Hypergraph Retrieval for Multi-hop RAG 

**Authors**: Jiate Liu, Liuyi Chen, Xiaofei Yuan, Yifu Tang, Zihan Yang, Xu Zhou, Zhengyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.35412)  

**Abstract**: Graph-based retrieval-augmented generation (GraphRAG) answers multi-hop questions by organizing a corpus into a graph and retrieving along its structure. Existing retrieval methods face two limitations. Diffusion- and similarity-based methods expand retrieval along fixed edge weights or by similarity to the question, without resolving its implicit entity references, allowing relevance errors to accumulate across hops. In contrast, exploration-based methods let an LLM reason about the next hop, but require a separate generation at each hop, leading to high inference costs. We observe that an LLM can resolve implicit entity references from retrieved passages and a multi-hop question before generating any tokens. Further analysis identifies a small set of attention heads, termed addressing heads, that capture this capability across multiple hops in a single forward pass. Building on this finding, we propose HyperAddress, which runs Personalized PageRank on a description-labelled hypergraph, using addressing heads to determine the restart distribution and transition weights, thereby guiding multi-hop retrieval without any generation at query time. On MuSiQue, 2WikiMultiHopQA, and HotpotQA, HyperAddress outperforms the strongest baseline by 3.5 points of Recall@5 and 7.4 points of All@5 on average, with gains that grow with the number of hops. It answers more than twice as fast and builds its index at half the cost. 

---
# CRIM: Composable and Reusable Item Memory for Efficient Personalized Retrieval 

**Authors**: Haolin Li, Haipeng Zhang, Mang Li, Zhaohui Xia, Xun Zhu, Miao Li, Yu Zhang, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.35323)  

**Abstract**: Large language models incur substantial training and inference costs in personalized search and recommendation, as growing interaction histories lengthen request inputs and repeatedly expose the same item text to the model. We propose \textbf{C}omposable and \textbf{R}eusable \textbf{I}tem \textbf{M}emory (CRIM), a framework that separates item-level semantic encoding from request-level history modeling. A small set of learnable summary tokens compresses each item into compact continuous memories, while segment-level readout tokens (REC tokens) introduce predictive supervision within the interaction history. These memories are independent of users and queries and can therefore be reused across requests. At inference time, CRIM composes cached memories in interaction order and incorporates an optional query to produce a personalized retrieval representation, reducing repeated item-text encoding while retaining request-dependent contextual modeling. We further introduce a joint training mechanism that combines cached memories with sampled live re-encoding. Re-encoding only a few history items per request allows the retrieval objective to jointly optimize the item and request encoders at limited computational cost, while periodic cache refreshes maintain consistency. Extensive evaluation and cost analysis on MovieLens-1M, Qilin, and our new LazSeq dataset show that CRIM achieves state-of-the-art retrieval performance with substantially lower inference costs at the same backbone scale, while limiting item re-encoding during joint training. We will publicly release our code and dataset. 

---
# Unified Heterogeneous Feature Interaction via Effective-Rank Scaling for Recommender Systems 

**Authors**: Ao Zhang, Haochuan Li, Boya Du, Jian Yang, Qihang Fang, Yulin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.35318)  

**Abstract**: Large-scale industrial ranking models must jointly capture heterogeneous feature interactions, long-term user behaviors, and multi-scenario dependencies under stringent online serving constraints. Existing architectures typically process sequential and non-sequential features with separate modules, limiting bidirectional information exchange. Directly applying homogeneous self-attention to all semantic tokens can also be inefficient when sequential and non-sequential signals exhibit different effective-ranks (eRanks). Meanwhile, shared representations can overlook scenario-specific patterns and induce negative transfer between scenarios. In this paper, we propose HiRank, a unified framework for multi-scenario recommendation that jointly models Heterogeneous feature Interactions via effective-Rank scaling. HiRank integrates gated sequential attention with heterogeneous feature interaction within a unified backbone, enabling bidirectional fusion between sequential and non-sequential representations. HiRank further uses scenario-aware aggregation to compress heterogeneous low-order tokens into a compact representation. Distinct learned projections transform the full compact representation into scenario-aware composite tokens with complementary global feature views and more uniform information density, enabling more effective high-order self-attention. Extensive experiments on industrial-scale datasets demonstrate favorable performance scaling with parameter count and consistent improvements over competitive baselines. Online A/B tests further show a 2.27% increase in orders per user. 

---
# Octen-WebGR: Scaling Generative Retrieval to 200 Million Web Pages 

**Authors**: Guangwei Xu, Wenkai Zhang, Yawen Liu, Dongxu Zhang, Xijie Xu, Kuan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2610.35256)  

**Abstract**: Navigational search aims to find a single target page among hundreds of millions of web pages. Click tables work well for queries seen before but cannot generalise to new formulations, while embedding retrievers generalise across content but often rank navigational targets poorly. We introduce Octen-WebGR, a generative retrieval framework that stores web pages in the parameters of a language model: given a query, the model generates a two-token semantic identifier constrained to a table of real URLs. We develop Octen-WebGR-4B, which indexes 201 million web pages, and evaluate it on real navigational traffic from the month after training. Generative retrieval is most effective on new ways of asking for familiar pages: for unseen queries targeting pages with substantial interaction history, it ranks the correct page first 30.9% of the time, against 17.2% for a dense retriever trained on the same clicks (21.4% with a click prior). Embedding retrieval keeps an advantage on rarely visited pages, so the two approaches generalise on complementary sides, the query and the page. Over all navigational traffic, Octen-WebGR reaches 59.6% top-1 accuracy, against 46.5% for a click-enhanced dense retriever and up to 37.5% for frontier LLMs answering from memory, which cost at least 40 times as much per query. The 4B model runs on a single GPU with 10 GB of memory, compared with 412 GB for exact dense retrieval. Scaling experiments with up to 32 million URLs show that, when every query-page pair is seen equally often, the model size needed grows sublinearly with index size, and that retrieval is constrained more by page exposure than by model capacity. We release Octen-WebGR-4B and Octen-WebGR-0.6B with their identifier tables and inference code. 

---
# Learning to Remember: Advances in Memory Mechanisms for Large Language Models 

**Authors**: Chenye Han, Hongche Yu, Mengyue Wu, Dell Zhang, Jixiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.35089)  

**Abstract**: Memory has become a central mechanism for extending large language models (LLMs) beyond fixed context windows, stateless interaction, and static parametric knowledge. This survey organizes the literature around three complementary dimensions: the life cycle of memory, the storage substrate, and the system role that memory plays in LLM-based agents, while using cognitive-science-inspired memory concepts as a lightweight conceptual motivation rather than as a separate technical contribution or implementation family. On this basis, we distinguish three mechanism families: nonparametric memory, where information is stored outside model parameters in text, graph, multimodal, or database-like structures; parametric memory, where information is stored or accessed through activations, key-value caches, learned modules, adapter-based parameterizations, or model weights; and memory for agent evolution, where stored experiences support reinforcement learning, continual learning, procedural reuse, context engineering, and agent harness design. Evaluation is treated as a separate cross-cutting component rather than as a memory mechanism. Unlike surveys that focus primarily on retrieval-augmented generation or long-context modeling, this paper emphasizes the full memory life cycle: construction, maintenance, retrieval, consolidation, forgetting, governance, and evaluation. By aligning conceptual motivations, computational memory mechanisms, benchmarks, and deployment concerns, the survey provides a structured roadmap for designing memory-augmented LLM agents that are persistent, adaptive, auditable, and safe to update over time. 

---
# Hallucination-Aware Multi-Agent RAG via Internal Attention Rewards 

**Authors**: Runsong Jia, Qian Chang, Mengjia Wu, Ciprian Doru Giurcăneanu, Jie Lu, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.35056)  

**Abstract**: Retrieval-augmented generation (RAG) grounds language models in retrieved evidence, but its outputs often contain claims unsupported by that evidence. Recent multi-agent RAG systems decompose the pipeline into cooperative decisions, yet they typically optimize all agents under a shared task reward such as EM or F1, which measures answer overlap rather than faithfulness to evidence. We propose a hallucination-aware multi-agent RAG framework that combines the task reward with an internal hallucination-risk signal computed from the generator's attention-head norms. Three role-specific LoRA adapters, a query planner, an evidence selector, and an answer generator, share a frozen backbone. A lightweight critic is trained once on TruthfulQA truthful/false pairs and then kept frozen, providing a complementary risk score that is added to the shared reward during cooperative PPO training. Across three backbones (Qwen3-8B, Llama3-8B, Llama2-13B) and three benchmarks (RAGTruth, HotpotQA, 2WikiMultiHopQA), our method improves task performance and reduces hallucination rates over the strongest multi-agent baseline. Ablations and critic diagnostics show that the TruthfulQA-calibrated signal provides useful reward shaping that transfers across RAG tasks. 

---
# CA-HNM: Constraint-Aware Hard Negative Mining for Dense Retrieval with Structured Domain Models 

**Authors**: Ngoc Luyen Le, Marie-Hélène Abel, Bertrand Laforge  

**Link**: [PDF](https://arxiv.org/pdf/2610.35045)  

**Abstract**: Dense retrievers are commonly trained with hard negatives selected from lexical or dense retrieval results. Retrieval proximity, however, does not indicate whether a candidate conflicts with explicit domain structure. We introduce CA-HNM, a constraint-aware hard-negative mining framework that combines retrieval-based candidate generation with deterministic incompatibility rules derived from a structured domain model. The framework removes known positives, constructs a local typed context around the query target, records relational, scope, target, and level mismatches, and uses the resulting candidates to fine-tune a standard dual encoder. Structured information is required only during offline negative construction. We evaluate CA-HNM on MOOCCubeX and Course-Skill Atlas with two encoders, two training losses, and ten seeds. Relative to a retrieval-rank-matched control, CA-HNM-Pure has higher mean NDCG@10 in six of eight dataset--encoder--loss configurations. Four hierarchical 95% bootstrap intervals are entirely above zero, mostly under triplet loss. In a separate experiment, CA-HNM-Mixed shows slightly higher mean NDCG@10 than a matched-mixed control on both collections. Structural variants show that changes in available structure can also change candidate eligibility and negative-pool composition. The results indicate that the contribution of constraint-aware selection depends on the training objective and the composition of the negative pool. 

---
# SLIDE: Search-to-List Item Distillation for Single-Rollout Generative Retrieval 

**Authors**: Tianyu Zhan, Gui Ling, Ran Tao, Kunhai Lin, Tong Xiong, Wang Yang, Kaixuan Zhang, Zhihong Chen, Yuliang Yan, Dan Ou, Shengyu Zhang, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.34764)  

**Abstract**: Generative retrieval (GR) typically relies on item-level beam search to retrieve candidates, leaving cross-item dependencies unmodeled and limiting direct optimization of list-level objectives. A natural alternative is to generate the entire list within a single autoregressive trajectory. However, greedy decoding with directly trained list-wise models substantially underperforms beam search. We identify two contributing factors: the broader search space explored by beam search and the positional bias of scarce positive signals, which deprives later positions of adequate supervision. To address these challenges, we propose $\textbf{SLIDE}$, a search-to-list item distillation framework that decouples offline search from online list generation. A teacher model retrieves high-quality candidates through offline item-level beam search, providing additional conversion-relevant supervision beyond sparse behavioral labels. Their Semantic IDs (SIDs) are then concatenated in beam-score order to form list-wise distillation targets. This search-to-list conversion transforms candidates retrieved along separate search paths into sequential item-generation targets within a single trajectory. Trained on these targets, the student generates the Top-$K$ list in a single greedy pass, also decoupling teacher scaling from online deployment. Experiments on FORGE and Taobao Search show that SLIDE substantially improves retrieval quality under single greedy decoding and, with stronger offline teachers, surpasses the beam-search reference. Supplementary RL experiments support the effectiveness of list-level generation. 

---
# Discovering Topics from Linear Autoencoders 

**Authors**: Jutiyakhun Srisoparp, Donlapark Ponnoprat  

**Link**: [PDF](https://arxiv.org/pdf/2610.34739)  

**Abstract**: Linear recommenders and topic models are similar in that they implicitly learn from co-occurrences of items. However, there has been little study of how these two model families are related. In this work, we show that, under a latent-factor model of the user-item interactions, the latent topics can be discovered from the weight matrix of a closed-form linear autoencoder. These topics provide a basis for interpreting the recommendations, and can be obtained directly from the trained recommender without additional training. Building on our analysis, we further propose novel item and user normalizations that prevent popular items and very active users from dominating the topics. Our experiments on six public datasets demonstrate that our method achieves significantly higher topic quality than every dedicated topic model and matches or exceeds the best linear autoencoder on all six datasets, while its recommendation performance is competitive with the best tuned recommender. 

---
# DPE-Rec: Diversity-Preserving and Efficient Diffusion Language Model Recommendation for Bundle Construction 

**Authors**: Pingchuan Ma, Bohan Jiang, Chengshuai Zhao, Dawei Li, Zhen Tan, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.34700)  

**Abstract**: Generative bundle construction completes a partially observed bundle by generating its missing items from a large product catalog. Compared to autoregressive LLMs, diffusion LLM emerges as a natural fit for this unordered generation task. Their bidirectional attention provides an inductive bias better aligned with unordered bundle construction. Iterative denoising can reveal multiple tokens per step, potentially reducing inference latency. However, directly applying DLMs to this task presents two key challenges: 1 Despite its flexible reveal order, confidence-guided decoding tends to postpone semantic-fork decisions until refinement tokens constrain their choices, reducing bundle diversity. 2 Revealing more tokens per step can compromise recommendation quality, stifling the potential for faster inference. We introduce DPE-Rec, a diffusion framework for diversity preservation and efficient recommendation. DPE-Rec preserves diversity by introducing semantic-fork-aware GRPO, which prioritizes semantic-fork decisions via autoregressive rollouts. It improves efficiency through pseudo-trajectory distillation, which concentrates learning from the vast random corruption space onto the compact trajectory-induced subspace visited by high-quality decoding, unlocking reliable multi-token decoding. Together, these components enable diverse and accurate bundle construction with substantially fewer denoise steps. Experiments on multiple bundle construction benchmarks show that DPE-Rec improves recommendation performance by 63% on average, and its fast variant achieves 25.6$times$ faster inference on average than existing generative recommendation arts, with minimal loss in recommendation quality. 

---
# Knowing When the Evidence Is Enough: Requirement-Directed Search with Horn-Closure Certificates 

**Authors**: Xinyu Zhang, Weiwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.34684)  

**Abstract**: After each Web search, a search-augmented model must decide whether its evidence suffices: answer, search again, or abstain. Prevailing systems leave this to holistic model judgement, which can accept a context naming every entity yet lacking one required fact. We instead make sufficiency structural: a per-query requirement hypergraph states which facts jointly yield another fact. This raises two challenges: deriving requirements from the question alone, and linking each requirement to passages about the right entity. Certify therefore decomposes the question, searches per requirement, and accepts links only after entity checks. Horn closure over the links yields a certificate, the minimal sufficient passages and their derivation, or an abstention naming open requirements. Given correct retrieval, requirements and links, the decision is sound, and complete up to the entity check. Pooled over three corpora with unanswerable twins, Certify decides significantly more accurately than search agents, graph retrieval and sufficiency judges in default configurations. On the same evidence, it matches a proprietary model's sufficiency label without an additional model call. Over 21M Wikipedia passages, entity checks add 4 to 13 points of decision accuracy where gold evidence exists. Finally, answers from its certificate are 80% correct, against 57% from all retrieved passages. 

---
# Agent Memory as Recommendation: Recommender-System Design for Long-Term Memory of LLM Agents 

**Authors**: Xinyu Zhang, Weiwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.34673)  

**Abstract**: Large language model (LLM) agents that serve users across many sessions must choose, for every request, the few stored memories that enter a limited context window. This choice largely decides whether the agent answers correctly, and it runs on every request at Web scale. This selection faces three challenges: a large and growing store, a per-request cost budget, and multi-session questions that need all of their evidence. Many memory systems build notes or graphs with hundreds to thousands of LLM calls per history, and each is designed on its own, so it is unclear which component yields the accuracy. These challenges mirror the properties of recommendation, so we formalize agent memory as budgeted slate recommendation, with the request as the user, stored memories as items and the context window as a slate. The formalization places memory systems in one design space of recommender cascades and factorizes accuracy into the completeness that each stage controls. Guided by it, we propose AgentMemRec, which adapts candidate generation, learning to rank and slate construction from recommender systems and selects memories without any LLM call. On multi-session questions of LongMemEval, AgentMemRec outperforms memory and context-selection methods, including A-Mem, HippoRAG 2, DynamicRAG and T-Mem, by 10 to 31 points of accuracy. It matches an LLM reranker that makes 64 calls per query at 3.2 times lower serving cost, and on LoCoMo it outperforms Mem0 and Zep without any LLM call at write time. The factorization attributes the gains to slates that more often hold all of the evidence, so recommender-system design offers an effective and efficient foundation for the long-term memory of LLM agents. 

---
# Theoretically Sound Item Similarity Measures for Recommender Systems 

**Authors**: Noah Daniëls, Joey De Pauw, Bart Goethals  

**Link**: [PDF](https://arxiv.org/pdf/2610.34402)  

**Abstract**: Quantifying the similarity between items is fundamental to neighborhood-based collaborative filtering, yet many commonly used similarity measures are heuristics that lack theoretical justification. The use of heuristics forces practitioners to rely on empirical trial-and-error, offering little insight into why certain measures outperform others or how they might be improved. In this paper, we present a principled approach for designing item similarity measures grounded in probabilistic modeling and parameter estimation.
By adopting this formal perspective, we derive two previously overlooked similarity measures based on the Bernoulli and multinomial distributions. Through an extensive ablation study we highlight the critical role of bias, probabilistic smoothing, treating the unknowns correctly, and choosing appropriate probability distributions. Finally, we show that the commonly used cosine similarity can be interpreted as the solution to a specific optimization problem, thereby providing a novel perspective on this decades-old heuristic. 

---
# A Domain-Aware Pipeline for Structured Extraction and Retrieval-Augmented Question Answering over arXiv Superconductivity Literature 

**Authors**: Ivan Iorsh, Aleksandr Kapitonov, Ivan Filianin, Matvey Lesonen, Nikita Chizhov  

**Link**: [PDF](https://arxiv.org/pdf/2610.34391)  

**Abstract**: Scientific literature in materials physics is growing faster than researchers can manually inspect it, yet most retrieval tools rely on keyword matching or generic question answering that ignores domain-specific structure in PDF files. This paper presents an end-to-end pipeline that transforms arXiv superconductivity publications into a searchable corpus with structured metadata and retrieval-augmented question answering. The system combines thematic corpus construction, domain-aware extraction of materials and superconducting transition temperatures, and hybrid retrieval that merges dense chunk embeddings with structured publication fields. We test whether normalized domain attributes improve evidence retrieval and answer groundedness relative to lexical-only and dense text-only methods. Extraction is evaluated on a manually verified gold set of 59 superconducting materials, 46 of which have confirmed experimental critical temperatures. On the 33-material primary positive subset, domain-aware pair accuracy reaches 45.5% versus 9.1% for a regex baseline. Retrieval and generation are compared under a fixed language model (qwen2.5:7b-instruct) on a 100-question benchmark, varying only the retrieval mechanism: BM25, dense, hybrid, and hybrid with domain metadata. BM25 gives the highest source recall (Recall@5 of 47%), and adding extracted metadata to hybrid retrieval does not improve it (39% versus 41%), which we attribute to noise in the extracted fields; the main bottleneck is locating the supporting span inside a correctly retrieved paper. As a materials-informatics application, literature-derived temperatures are matched to an independent theoretical reference set: 200 materials appear in both sources, and 20 have a median experimental critical temperature above the theoretical value. 

---
# Cross-Lingual Information Retrieval in Tetun for Ad-Hoc Search 

**Authors**: Altedio Araújo, Gabriel de Jesus, Sérgio Nunes  

**Link**: [PDF](https://arxiv.org/pdf/2610.34362)  

**Abstract**: Developing information retrieval (IR) systems that enable access across multiple languages is crucial in multilingual contexts. In Timor-Leste, where Tetun, Portuguese, English, and Indonesian are official and working languages, no cross-lingual information retrieval (CLIR) solutions currently exist to support information access across these languages. This study addresses that gap by investigating CLIR approaches tailored to the linguistic landscape of Timor-Leste. Leveraging an existing monolingual Tetun document collection and ad-hoc text retrieval baselines, we explore the feasibility of CLIR for Tetun. Queries were manually translated into Portuguese, English, and Indonesian to create a multilingual query set. These were then automatically translated back into Tetun using Google Translate and several large language models, and used to retrieve documents in Tetun. Results show that Google Translate is the most reliable tool for Tetun CLIR overall, and the Hiemstra LM consistently outperforms BM25 and DFR BM25 in cross-lingual retrieval performance. However, overall effectiveness remains up to 26.95 percentage points lower than that of the monolingual baseline, underscoring the limitations of current translation tools and the challenges of developing an effective CLIR for Tetun. Despite these challenges, this work establishes the first CLIR baseline for Tetun ad-hoc text retrieval, providing a foundation for future research in this under-resourced setting. 

---
# MIDA: Multi-Interest Decoding with Adaptive Aggregation for Session-based Recommendation 

**Authors**: Linjiang Guo, Nitin Bisht, Shiqing Wu, Guandong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.34192)  

**Abstract**: Session-based recommendation (SBR) infers user interest from limited interaction sequences, yet current models show a consistent performance drop when longer histories are available. This degradation is associated with decoder designs that overemphasize recent items while underutilizing distant, yet semantically relevant, interactions, leading to poor modeling of long-range dependencies and multi-interest behaviors. To address this challenge, we introduce MIDA (Multi-Interest Decoding with Adaptive Aggregation for Session-based Recommendation), a lightweight and encoder-agnostic framework that provides a reformulation of the decoding stage. The central design principle is a two-space decoupling: clustering operates on raw item embeddings so that semantic groupings are not distorted by encoder-learned relational biases; encoder outputs are reserved for aggregation, where sequential context remains intact. Concretely, MIDA (i) employs differentiable multi-interest clustering on raw embeddings, preserving semantic structure that encoder representations would otherwise obscure, and (ii) fuses stability and plasticity branches through a gate conditioned on average cluster size, so that the inferred clustering structure directly governs aggregation behavior through a structural coupling across the two stages. Extensive experiments on multiple benchmark datasets demonstrate that MIDA consistently outperforms state-of-the-art methods, with meaningful accuracy improvements while remaining computationally practical. 

---
# LoopGR: Parameter-Efficient Scaling for Generative Recommendation via Recurrent Depth 

**Authors**: Huizhe Zhang, Yudong Li, Liang Chen, Chengxiang Zhuo, Zang Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.34157)  

**Abstract**: Generative recommendation formulates item recommendation as autoregressive generation rather than a discriminative scoring problem. Although empirical studies exhibit favorable scaling behavior in recommendation, unleashing the benefits of scaling remains challenging under the strict model size and inference cost constraints of real-world systems. A central challenge is to combine expressive modeling of user-item interaction histories with flexible computational depth without proportionally increasing model size. We introduce LoopGR, a parameter-efficient scaling framework for generative recommendation via recurrent depth. LoopGR employs a tailored hierarchical recurrent depth mechanism that alternates between a shared generative decoder and a lightweight context updater. For efficient inference, we further develop recurrent trajectory self-distillation that distills a low-budget student from a high-budget teacher by aligning both predictive distributions and evolution directions. Extensive experiments show that LoopGR consistently outperforms strong generative recommendation baselines. Compared with a representative encoder-decoder baseline, LoopGR achieves superior performance while reducing model size by up to 52%. These results establish recurrent depth as a new scaling dimension, offering a flexible and resource-efficient scaling path for generative recommendation. 

---
# Decision-Native Recommendation with Large Language Models 

**Authors**: Daoxuan Liu, Jiayi Liao, Jiancan Wu, An Zhang, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2610.34124)  

**Abstract**: Large language models (LLMs) bring world knowledge to user preference modeling in sequential recommendation, but generation-based recommenders express next-item predictions indirectly as text. Generated outputs may fail to identify a unique candidate item, depend on candidate presentation order in the prompt, and lack explicit confidence in the recommendation. Recent studies explore Jev, a model for fast option selection with explicit confidence. Inspired by Jev, we introduce decision-native recommendation, a new paradigm that jointly models item choice and recommendation confidence using LLM representations. It requires valid candidate selection, consistency under candidate reordering, and confidence that reflects recommendation accuracy. We propose Direct Item Choice Estimation (DICE), which learns a personalized next-item distribution over supplied candidates from user interaction history and recommends the highest-scoring item with calibrated confidence. Tree-structured attention shares history computation across isolated candidate branches for order-consistent candidate scoring in a single forward pass. On three recommendation datasets, DICE achieves higher top-one accuracy than all evaluated baselines. Separate evaluations show unchanged item choices under tested candidate reorderings and end-to-end speedups of up to approximately $6\times$ over the evaluated generation-based methods. Its confidence estimates support selective recommendation by identifying more reliable predictions and guide additional computation for low-confidence examples in offline routing. 

---
# Bridging Search Intent and Tabular Data in Dataset Search 

**Authors**: Wenjing Lin, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2610.33936)  

**Abstract**: Dataset search remains a bottleneck in data analysis workflows, as existing systems largely rely on keyword matching or shallow retrieval that fail to capture user intent. To facilitate intent-driven search, we present two retrieval components to equip multi-faceted dataset search interfaces through natural language. First, we propose Hypothetical Schema Embeddings (HySE) that uses a language model to generate task-aligned hypothetical table schemas which are used for embedding-based retrieval, bridging the gap between intent and table content. We introduce KaggleDS, a corpus comprising 6,503 tables and diverse task-queries derived from Kaggle, and show that HySE consistently improves recall over strong keyword and embedding-based baselines, as well as over a dense table retriever fine-tuned on KaggleDS. We also show that HySE generalizes to end-to-end question answering on, for example, the NQ-Tables dataset, a corpus of 169,898 Wikipedia tables with human-written queries, where HySE improves Recall@10 by +6.0 pp over dense retrieval baselines. Second, we introduce FlexPrune, which translates dataset constraints expressed in natural-language into SQL predicates over a customizable metadata index, enabling filtering across arbitrary metadata attributes. Combined with HySE, FlexPrune increases Recall@10 by +6.0 pp over dense retrieval followed by tag-based filtering. Our results show that natural-language-based retrieval supporting task and metadata queries offers a viable alternative to schema-level keyword search in dataset search. 

---
# SkillQR: Agentic Skill Discovery for Reason-Intensive Query Rewriting 

**Authors**: Kenan Jiang, Meixiu Long, Zihao Zhang, Ran Xu, Eugene Agichtein, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.33768)  

**Abstract**: Dense retrievers provide grounded evidence that mitigates hallucinations in outputs from Large Language Models (LLMs). Real-world queries are often verbose, ambiguously framed, or require multi-step reasoning, introducing irrelevant details that weaken semantic matching for dense retrievers. Query rewriting is a widely adopted pre-retrieval technique that leverages LLMs to reformulate queries and bridge the semantic gap between such queries and the retrieval target. Existing rewriting approaches either rely on expensive proprietary models, substantial training data, or iterative retrieval and rewriting, while largely applying a uniform rewriting strategy regardless of the retrieval corpus. We observe that effective query rewriting instructions are not universal: the optimal strategy depends on the retrieval target, rather than the model alone. We introduce SkillQR, a query rewriting framework that automatically discovers reasoning strategy sets from the retrieval target and dynamically routes a query to the appropriate rewriting strategy at inference time. Without relying on training data, SkillQR achieves competitive performance compared to fine-tuned rewriting models on the BRIGHT and R2MED benchmarks, while also providing lower inference latency than iterative rewriting approaches. 

---
# Novel or Not? PatEx-Gym: Agentic Prior-Art Search for Novelty Assessment in Patent Examination 

**Authors**: Qiyao Wang, Xinyi Chen, Hongbo Wang, Jiayuan Lu, Jianhang Li, Minghuan Tan, Yuan Lin, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.33503)  

**Abstract**: Patent examination is a challenging real-world task that requires agents to discover sparse prior art, iteratively refine queries, inspect technical evidence, and produce novelty assessments documented in Office Actions (OAs) across heterogeneous scientific and technical domains. However, existing benchmarks largely focus on static and limited candidate sets, leaving the full search--reasoning--assessment process underexplored. We present PatEx-Gym, the first framework for developing and evaluating agents for patent examination, which integrates three components. First, PatEx-Bench provides 1K realistic examination tasks, two large-scale retrieval corpora containing over 1M patents and 52K scientific papers, and a harness runtime supporting iterative search, document visits, reasoning, and OA generation. Second, we introduce PASS, an annotation-free Prior-Art Agentic Search Trajectory Synthesis framework. Specifically, it exploits document visits as implicit relevance signals and evidence-acquisition anchors to model intra-turn query behavior, inter-turn search progression, and trajectory-level evidence and outcome quality, yielding PatEx-SynData. Third, we train PatEx-Agent at the 4B scale via supervised fine-tuning on the curated trajectories, establishing a lightweight baseline. Extensive experiments show that final decision accuracy alone does not fully reflect examination quality, while prior-art discovery remains difficult even for strong frontier models. Our PatEx-Agent achieves competitive performance against substantially larger general-purpose agents. These results show that training on PatEx-SynData enables a compact 4B model to serve as a competitive baseline for prior-art search and patent examination. We view patent examination as a demanding testbed for agentic search and deep research agents. 

---
# DyHist: Query-Aware Historical Dynamics Modeling for Temporal Knowledge Graph Extrapolation 

**Authors**: Peiyuan Jiang, Yao Liu, Yubo Wen, Yanglei Gan, Lu Liu, Ruojie Zhang, Jiaye Yang, Yuewei Xue, Jiaya Deng, Xiaogu Guo, Qiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.33473)  

**Abstract**: Temporal Knowledge Graph (TKG) extrapolation aims to predict future facts from observed historical events. Existing methods usually achieve this goal by jointly modeling local historical evolution and long-range historical contexts. First, local evolution modeling often summarizes local history into an overall evolving representation, where query-relevant local deviations may not be sufficiently exploited as useful predictive features. Second, existing long-range history modeling often treats historical recurrence as a stable cue, while overlooking that its reliability may vary across relations and recent contexts. To address these issues, we propose DyHist, a query-aware historical dynamics modeling framework for TKG extrapolation. DyHist captures local dynamics by separating recent evidence from medium-range contexts and explicitly modeling their query-relevant deviations as predictive features. It further uses a relation-aware historical prior to calibrate candidate scores with time-decayed support, recent trends, and relation-specific strength, enabling recurrence signals to contribute differently across relations and recent contexts. Experiments on four benchmarks show that DyHist achieves the best MRR against recent baselines. Code is available at this https URL. 

---
# PinGER: Token-Efficient Generative Retrieval with Semantic IDs 

**Authors**: Xinyi Li, Sujan Reddy Ale, Yi-Ping Hsu, Zelun Wang, Yilin Chen, Hanyu Li, Hanlin Lu, Arkin Dharawat, Lulu Yao, Yaxin Li, Bella Huang, Xiangyi Chen, Jaewon Yang, Jiajing Xu, Charles Rosenberg  

**Link**: [PDF](https://arxiv.org/pdf/2610.33435)  

**Abstract**: Generative retrieval replaces nearest-neighbor search with token-by-token decoding of an item's semantic ID (SID). But existing SID-based methods primarily use L levels of tokens sequentially to represent each item in the input history. As a result, a fixed context window holds L times fewer items than it would with one embedding per item. We present PinGER, a generative retriever deployed at Pinterest that retains SID outputs while making input histories compact, built on three ideas. (1) An asymmetric attention mask over interleaved SID and dense tokens lets later items attend only to the dense ones, so the model is supervised at every SID position in training yet serves one token per item while computing the same function. (2) Generation is controllable through goal-conditioned outputs, reinforcement learning, and a prefix-diversity penalty, which together steer retrieval toward surface-specific goals and diverse candidates. (3) A serving stack built for efficient wide-beam generation is paired with decoding constrained to valid identifiers and candidate resolution from a host-local index. Deployed in Related Pins, PinGER lifts saves of fresh Pins by 3.60% and overall saves by 0.62%, while reducing the surface's end-to-end latency by 0.7% and eliminating the cost of the nearest-neighbor search it replaces. 

---
# Tiny Suffix Arrays 

**Authors**: Sharma V. Thankachan  

**Link**: [PDF](https://arxiv.org/pdf/2610.35543)  

**Abstract**: Suffix array and inverse suffix array queries on a text of length $n$ can be answered in $\widetilde{O}(\tau^3)$ time using an auxiliary data structure occupying $O(n/\tau)$ words, for any $\tau \in [1,n]$. 

---
# Agreement Between Reconstructed and Native GDELT GKG Records: A Field-Level Baseline 

**Authors**: Nikita A. Klimin  

**Link**: [PDF](https://arxiv.org/pdf/2610.35358)  

**Abstract**: The Global Knowledge Graph (GKG) of the Global Database of Events, Language, and Tone (GDELT) characterizes news articles using named entities, topics, quantitative and temporal features, quotations, tone, and dictionary-based content measures. Its codebook defines the record format, but the software that produces the records is not public, so researchers cannot generate compatible records for articles they collect themselves. We built an open-source reconstruction, gdelt-regkg, and measured how closely it agrees with GDELT's archived (native) records for 193,550 English articles from 8,729 publishers (2020-2025) and 2,500 articles in five other languages. Agreement depends strongly on the field. English tone, the percentage of positive minus negative words, has a mean absolute error (MAE) of 1.79 points and a concordance correlation coefficient of 0.75. Identity agreement (F1) is 0.65 for persons, 0.42 for organizations, and 0.65 for locations. It is lower for date components (0.42), complete counts with their objects and locations (0.07), and quotations (0.07), while GCAM category presence reaches micro F1 0.84 within the implemented dictionaries. In subsets where retrieved and native titles and lengths match, tone error falls to 1.47 and organization F1 rises to 0.48, yet most disagreement remains. A larger gazetteer raises coverage of native places from 89.5% to 97.1% with little change in location F1. Tone computed on English translations of non-English articles is less accurate (MAE 2.10-2.64) than for English articles. The results are conditional on article pages that could still be retrieved, and the correspondence subsets do not isolate the effect of input differences. Reconstructed records should therefore be assessed field by field and not treated as completely interchangeable with native ones. 

---
# Reading Beyond OCR Noise: Robust Cross-Lingual RAG for Legislative Archives 

**Authors**: Radha Krishna Deshpande, Ajoy Mondal, Akash Manna, C V Jawahar  

**Link**: [PDF](https://arxiv.org/pdf/2610.35240)  

**Abstract**: India's historical public archives present severe computer vision and document image analysis challenges: decades of parliamentary records survive only as degraded, multi-column scanned pages featuring mixed scripts, complex layouts, and dense visual noise where OCR failure is the primary obstacle to information retrieval. We present an end-to-end framework that couples vision-based layout segmentation and multi-script OCR with an agentic retrieval-augmented generation (RAG) pipeline. Our approach specifically addresses how visual degradation and layout fragmentation propagate into text extraction errors, using an iterative agentic loop to navigate around OCR noise and layout ambiguities before generating answers. Evaluated on the 200,000-page Punjab Vidhan Sabha archive, our system demonstrates robust handling of historical document degradation, outperforming standard text-only baselines. 

---
# When Fusion Order Matters: Insights on Progressive Multimodal Representation Learning for Cross-modal Retrieval 

**Authors**: Haoxiang Zhao, Bin Wang, Yaofei Wang, Xiao Lan, Yunfeng Diao, Xu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.35125)  

**Abstract**: Multimodal video retrieval combines visual, acoustic, and speech information, yet how their fusion order should depend on the data and the retrieval query remains unclear. We introduce Unified Progressive Representation Learning for Order Analysis, or UniPRO, to investigate this question through configurable fusion schedules with matched model components and training settings. Starting from text-to-video retrieval, we find that the tested one-by-one schedules outperform stacked fusion, while changing the order within one-by-one fusion also changes performance. To guide order selection, we examine two complementary considerations: learning difficulty under unequal modality availability and preservation of information relevant to the query. Complete-modality controls show that missing inputs alone cannot explain order sensitivity. At R@1, text and audio retrieval favor schedules that introduce the modality matching the query last. Controlled speech missingness during training and testing reverses the preferred order in both R@1 and SumR, with different crossover points. These findings motivate balancing the two considerations when selecting a fusion path. We then use them to predict a visual-last schedule for image queries, which achieves 97.9 R@1, the highest among the evaluated orders on MSR-VTT-9k. Main experiments on four text-to-video benchmarks establish the effectiveness of UniPRO, with component ablations assessing its supporting design. Together, the results identify fusion order as a task-dependent choice that should account jointly for modality availability and the retrieval query. 

---
# Useful Diversity or Hallucinated Variety? Persona-Conditioned Synthetic Query Generation for E-Commerce Search 

**Authors**: Filippos Ventirozos, Tom Armitage, Mahmoud Oshagh, Peter A. Appleby, Matthew Shardlow  

**Link**: [PDF](https://arxiv.org/pdf/2610.35102)  

**Abstract**: Synthetic supervision can bootstrap semantic parsers in new search domains, but prompt-induced diversity is useful only when generations are faithful to the structured intent. We explore this for cold-start e-commerce search: 118,098 synthetic pairs produced by 6 LLMs with 9 persona-conditioned prompt families on a 47-field UK automotive schema, against 1,214 human-annotated and 15,304 production queries. We show that the IPIP personality bundle improves parser F1 in all four matched contrasts (Holm-corrected $p<.001$), and all three bundles have a positive main effect over the complete $2^3$ grid. Syntax conditioning is the axis that moves generations towards real traffic. It also induces lexical diversity. Alignment sKL, the divergence between von Mises-Fisher distributions fitted to synthetic and human-annotated queries, is an unlabelled proxy for parser F1 ($\rho=-0.917$ against human queries written for simulated shopper scenarios, lower is closer). An LLM committee, checked against a blinded author pass, triages the 927 queries missed by every parser and labels 37% as hallucinated. Dropped constraints dominate. The four syntax-bearing families have the largest hallucinated shares, 38 to 42%, while IPIP alone has the lowest at 28%. 

---
# Reading the Registers of Honorius III without Manual Line Transcription: Forced-Aligned HTR, End to End, Modular or Hybrid 

**Authors**: Tamas Kovacs, Daniel Luger  

**Link**: [PDF](https://arxiv.org/pdf/2610.35085)  

**Abstract**: Unlocking medieval archives at scale is bottlenecked by the cost of line-level diplomatic transcriptions and manual abbreviation expansion. Focusing on the registers of Pope Honorius III (1216-1227) over 5,000 letters surviving in full only as manuscript images - we show that existing, line-break-free expert transcriptions (edition drafts) suffice to train end-to-end hand-written text recognition (HTR) that jointly transcribes and expands heavily abbreviated Latin. Using forced alignment on 860 unaligned edition entries across 394 pages of Reg. Vat. 9, we automatically bootstrap 13,122 training lines without manual alignment of segmented lines. While the generic CATMuS baseline struggles on this domain (18.9% CER) and a modular space-free HTR + word-segmentation pipeline (cf. Camps et al. 2021) plateaus at 15.6% WER, our from-scratch hybrid Conformer - combining CTC with bidirectional subword attention decoders, iterative self-realignment and n-best LM rescoring - achieves 4.7% CER and 8.7% WER (8.4% WER with ROVER system combination), nearly halving errors on clause-final words (22.7% to 12.2% WER). These figures are measured on the test lines that align with the transcription (86% of all test lines for the best hybrid). Furthermore, we address a tension between paleographic fidelity and editorial normalization: training on targets that preserve scribal name initials reduces the WER against the manuscript text on the 31 affected test lines from 13.3% to 9.2%, at a small cost against the expanded transcription (WER +0.3 points, CER +0.4). Finally, extensive ablations show that none of an auxiliary diplomatic CTC head, a copy mechanism, a permutation decoder or masked-image pretraining improves on direct end-to-end supervision, while strict padding-mask propagation and per-line beam termination make inference batch-invariant. 

---
# HARP-LoRA: Hierarchy-Conditioned Rank Projection for Low-Rank Adaptation in Scientific Table Understanding 

**Authors**: Hossein Sholehrasa, Mohammadhossein Amini, Doina Caragea, Remya Ampadi Ramachandran, Dan Wu, Zihang Shi, Zelin Li, Lindsey Hulbert, Eder Martinez, Xuan Xu, Majid Jaberi-Douraki  

**Link**: [PDF](https://arxiv.org/pdf/2610.34589)  

**Abstract**: Tabular data is widely used in scientific applications, where complex layouts, including multi-level headers, merged cells, captions, and footnotes, make information extraction difficult. This is particularly important in pharmacokinetics (PK), where values must be correctly associated with the drug, species, dose, administration route, unit, and experimental condition. Parameter-efficient fine-tuning methods such as Low-Rank Adaptation (LoRA) reduce adaptation cost but do not explicitly incorporate relationships between table values and surrounding headers. We introduce HARP-LoRA, a hierarchy-conditioned rank projection method for low-rank adaptation in scientific table understanding. HARP-LoRA extends projected LoRA by constructing token-specific intrinsic representations from the row- and column-header hierarchies of each data token. These representations condition the low-rank down-projection, incorporating hierarchical context while keeping the pretrained model and random projection tensors frozen. We evaluate HARP-LoRA on 950 manually annotated PK tables from 587 scientific articles and on WikiTableQuestions (WikiTQ) using Llama~3.2 3B, Llama~3.1 8B, and Gemma~3 27B. With Llama~3.1 8B, HARP-LoRA achieves a PK F1-score of 57.64\% and WikiTQ accuracy of 53.15\%, improving over LoRA by 1.42 and 6.56 points, respectively. With Gemma~3 27B, it achieves the best PK F1-score of 70.63\% and WikiTQ accuracy of 56.87\%. These results use only 172K, 197K, and 547K trainable parameters, far fewer than LoRA and TableLoRA. Overall, conditioning lightweight adapters on table-header hierarchies improves extraction from scientific tables and reasoning over flatter general-domain tables while maintaining parameter efficiency. Code is available at: this https URL 

---
# RefinedGAW: A Refined German Academic Web Dataset 

**Authors**: Maximilian Idahl, Oleh Astappiev, Benedikt Droste, Robert Scholz, Björn Plüster, Jan Philipp Harries, Robert Jäschke, Wolfgang Nejdl  

**Link**: [PDF](https://arxiv.org/pdf/2610.34572)  

**Abstract**: The largest open German text datasets are built from broad web crawls, which provide scale but limited coverage of specific domains. We introduce RefinedGAW, a processed and annotated resource derived from 26 semi-annual crawls of the German Academic Web, totaling 122 million documents and 314 billion tokens, with full crawl and capture history of every text. Documents labeled as academic research account for 84% of German RefinedGAW tokens across HTML and PDF. At 87.6 billion tokens, this exceeds the 64.9 billion tokens in the German subsets of FineWeb2 and FinePDFs, despite their almost five-times-larger combined size. RefinedGAW also contains substantial amounts of educational, math-heavy, and expert-oriented content with minimal commercial bias. Across HTML and PDF, 88% of German RefinedGAW documents have no near-duplicate candidate match in the three evaluated reference datasets, supporting complementary coverage. We release a queryable metadata index for fine-grained data selection and a rehydration tool for retrieving and processing source documents. 

---
# The Price of Scale in Proactive Memory Recall 

**Authors**: Yichuan Chen, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.34507)  

**Abstract**: A long-lived assistant that brings up memories without being asked must decide at every turn whether anything in its store is worth an interruption. We show that this decision is a multiple-testing problem whose difficulty grows with the store size $N$. On streaming replays of LongMemEval and LoCoMo, a similarity threshold tuned to 5% false alarms at $N=8$ reaches 87-100% at $N=2048$, across embedding models and for explicit and implicit cues; LLM judges show the same growth from a lower base. We prove that a gate with false-alarm rate $\alpha$ operates at a per-item false-positive rate of about $-\ln(1-\alpha)/N$, which pins its hit rate to the far left of the single-comparison ROC curve and, under regular variation, makes it decay as a power of $N$ with an explicit constant. The resulting prediction matches the hit rate of a per-$N$ oracle threshold to within one percentage point. The analysis explains why lexical and semantic channels need thresholds growing like $\ln N$ and $\sqrt{\ln N}$, and why per-cue normalisation roughly doubles the hit rate at matched false alarms. A cue-swap conformal gate, which compares the current cue with the user's other cues on the same store, needs no labels and has a finite-sample false-alarm guarantee under exchangeability; on LongMemEval it holds false alarms near 5% at the hit rate of a label-tuned threshold. Three annotators (Fleiss' $\kappa=0.63$) find that about nine in ten false alarms are genuine interruptions. Code and data: this https URL 

---
# Multimodal Requirements Traceability Link Recovery in Industrial Settings 

**Authors**: Haoming Huang, Shinpei Hayashi, Kazuya Kohata, Myeongki Kim, Yuki Kurihara  

**Link**: [PDF](https://arxiv.org/pdf/2610.34503)  

**Abstract**: Requirements traceability links support activities such as change-impact analysis, coverage checking, and validation. Industrial requirements can include traceable images, which text-only approaches cannot process directly. For a fixed classifier configuration, binary decisions alone offer no direct way to adjust the balance between missed and incorrect links. We extend a retrieval-augmented generation (RAG) approach for inter-requirements traceability link recovery (TLR). Our extension retrieves text and image candidates separately, classifies each retrieved pair with a vision-language model, and filters the predicted links by verbalized confidence. We evaluate our approach on three non-overlapping industrial requirement sets and six public text-only datasets. On the industrial test sets, our approach achieves an F1 score of 0.310, compared with 0.114 for our replication of the baseline approach by Hey et al. Our approach recovers links in all four modality combinations while improving text-only F1 on the industrial data and maintaining similar performance on the public datasets, but recovery remains limited by failures to retrieve related text requirements for image queries. Confidence filtering produces link sets with different precision and recall, removing false positives and some correctly recovered links from the unfiltered link set. This study provides industrial evidence on the potential and limitations of a current RAG-based TLR approach for multimodal requirements and identifies cross-modal retrieval as the component to prioritize in further optimization. 

---
# Factual Yet One-Sided: Stance Poisoning and Evidence Balancing in Retrieval-Augmented Generation 

**Authors**: Puyuan Yang, Shaoxuan Dong, Liujie Wu, Xiaochun Cheng, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.34294)  

**Abstract**: The stance expressed in responses from retrieval-augmented generation systems is significantly influenced by the stance distribution of the retrieved materials. Existing research on knowledge poisoning has primarily targeted factual errors. In contrast, this study focuses on injection attacks in which the facts are correct but the stance on a controversial issue is one-sided. At a mean topic-level poisoning share of 18.9% within the post-injection ProCon documents, all four generation models from different vendors exhibited stance shifts aligned with the poisoning direction; across four concentration levels, the shifts increased monotonically.
This paper proposes a Stance-Balancing Middleware positioned between the retriever and the generator. It consists of stance detection, fixed balancing (2 supporting + 2 opposing + 1 neutral), and a debiasing prompt, without introducing additional large language model calls. Across all eight concentration-direction settings, the complete method reduced the magnitude of stance deviation by an estimated 0.106 to 0.350; the six settings at poisoning shares of 11.9%-18.9% remained significant after Holm adjustment (adjusted p <= 0.009). In the cross-model experiment, all eight model-direction combinations moved along the stance axis in the corrective direction, and seven of eight showed a clearly detected reduction in absolute distance from 0. In the concentration sweep, the post-defense intervals included 0 in all eight settings, while the cross-model pro-poisoning results retained small positive residual point estimates of +0.045 to +0.063. For 20 candidate documents, stance detection took approximately 1.5% of the median generation time on a single GPU and approximately 16% on CPU. 

---
# Recovering Business Rules as Cascade Lookups for Enterprise Field Prediction 

**Authors**: Changxu Duan  

**Link**: [PDF](https://arxiv.org/pdf/2610.34258)  

**Abstract**: Enterprise sales-order systems rely on manually maintained configuration rules to determine fields such as payment terms, shipping points, and sales ownership. When these rules are incomplete or stale, operators must enter values by hand, and downstream errors spread into billing, logistics, and reporting. This paper studies whether missing or outdated rules can be recovered from historical transactions instead of being relearned as opaque statistical patterns. We introduce a knowledge-graph-guided workflow in which an LLM agent proposes candidate business keys, meaning column combinations such as customer and document type, validates them through SQL profiling, and compiles the resulting evidence into deterministic cascade lookups. The LLM is used only during offline rule discovery. At inference time, prediction is a transparent key lookup with no model-serving requirement. Experiments on enterprise sales-order prediction and public relational benchmarks show that cascades are most effective when fields follow stable business rules, while learned graph models remain preferable when prediction depends on temporal or neighborhood patterns. Drift analysis further turns residual errors into diagnostics for configuration maintenance. Code and data: this https URL. 

---
# $Δτ$-Bench: Pushing Agents Toward a Moving Capability 

**Authors**: Shuofei Qiao, Xuanguang Pan, Yijun Chen, Wendi Feng, Bin Wu, Runnan Fang, Yuqi Zhu, Shumin Deng, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2610.34238)  

**Abstract**: In this paper, we introduce $\Delta\tau$-Bench, an executable agent benchmark that models the structural evolution of large tool ecosystems. In $\Delta\tau$-Bench, independently evolving services continually reshape how tasks can be accomplished: familiar tools may disappear or be replaced, while more efficient alternatives emerge amid overlapping functionality and noisy documentation. Against this moving capability frontier, we evaluate whether agents can proactively discover efficient tool-use paths and promptly revise their accumulated procedural knowledge, bringing together demands that existing benchmarks fail to examine. $\Delta\tau$-Bench comprises 1,588 validated multi-turn tasks across 42 domains and six successive ecosystem states generated through capability-preserving transformations. Our evaluation shows that even the strongest evaluated model or harness achieves only 42.3% task success, while initial gains from existing skill-learning methods can not consistently translate into better adaptation as the ecosystem evolves. These findings motivate Steward, an ecosystem manager that jointly manages tools and procedural knowledge through incremental skill maintenance and dynamic provisioning during task execution. Experiments show that Steward sustains higher task success across ecosystem changes with lower skill-maintenance overhead, while improving both recovery from disrupted workflows and adoption of more efficient alternatives. 

---
# TriageRAG: Failure-Aware Self-Training for Adapting Retrieval-Augmented Language Models to Specialized Domains 

**Authors**: Samarth Varshney, Ritu Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2610.33776)  

**Abstract**: Retrieval-augmented generation (RAG) lets large language models (LLMs) answer knowledge-intensive questions from retrieved evidence, but adapting a general-purpose RAG system to a specialized domain such as medicine is difficult when only an unlabeled corpus is available. Self-training approaches such as SimRAG address this by letting the LLM write its own question-answer pairs from domain passages and fine-tuning on the pairs that survive a round-trip consistency filter. We observe that these approaches treat every retained example identically: they never ask whether the student already knows the answer, already extracts it correctly from the retrieved context, or fails - and if so, whether the failure lies with the reader or with the retriever. To tackle this, we propose TriageRAG, a failure-aware self-training approach that probes the student on each self-generated question with and without retrieved context, triages it into one of four categories (known, solved, context-failure, retrieval-failure), and routes each category to the training signal that addresses it, including a parametric route that teaches the reader to answer when retrieval misses. Probing is cheap and repeated every round, so the training mixture follows the student, and the triage histogram doubles as a label-free diagnostic of where the domain gap lies. Experiments on 11 datasets spanning three domains with a Qwen3-4B backbone, and on CS-Bench with Qwen3-8B, demonstrate that TriageRAG outperforms a budget-matched SimRAG by 1.2-6.4 points and is the only method that recovers the closed-book level in every case where retrieval hurts the base model. A single ablation identifies the mechanism: round-trip filtering removes every training context that lacks the evidence, and a reader that never sees such contexts learns to over-trust noisy retrieval. 

---
# Reinforcement Learning for Controller Parameters in the Auto-Bidding System at Pinterest 

**Authors**: Yichu Zhou, Qinyi Chen, Fei Feng, Jacob Loh, Mehdi Ben Ayed, Lin Yang, Andrei Curelea, Jiajing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.33248)  

**Abstract**: Auto-bidding systems in online advertising must balance competing objectives-maximizing budget delivery while minimizing cost per action-across heterogeneous campaigns under non-stationary market conditions. This trade-off evolves over campaign lifespans: early on, cost optimization dominates when budgets roll over; near deadlines, budget delivery becomes urgent. While PID controllers are widely used in production for their stability and interpretability, fixed tunings cannot adapt to diverse campaigns and shifting traffic patterns. We present RLBP (Reinforcement Learning for Bidding Parameters), a hybrid architecture deployed at Pinterest that uses RL as a meta-controller to select campaign-specific PID parameters daily while retaining PID for high-frequency bid adjustments. RLBP trains a multi-head Q-network using a relative-lift reward formulation normalized against the production baseline, and performs inference-time fusion that adapts the cost-delivery trade-off to campaign states without retraining. Through online A/B testing at Pinterest, RLBP achieves a statistically significant reduction in effective cost-per-click (eCPC) with no revenue degradation, and two months of post-rollout measurements against a holdout also show consistent cost reductions. These results demonstrate that meta-controller RL can successfully augment production control systems at scale. 

---
# Holo-NN: Holographic Nearest Neighbor Search via Compressive Sensing 

**Authors**: Bryant Gipson  

**Link**: [PDF](https://arxiv.org/pdf/2610.33224)  

**Abstract**: In the age of AI, privacy-preserving approximate nearest neighbor (ANN) search over high-dimensional embeddings has emerged as a critical yet frustratingly challenging domain, dominated by tradeoffs between speed, privacy, and accuracy. Traditional tree and graph-based indexing structures (e.g., HNSW, ScaNN) rely fundamentally on non-linear operations that are computationally intractable under Fully Homomorphic Encryption (FHE) or blind server-side constraints.
In this paper, we introduce Holo-NN (Holographic Nearest Neighbor Search), an algebraic framework that replaces data-dependent branching with a single linear projection. Our key observation is that while distances across a database are dense, neighborhood membership is inherently sparse: only a handful of points ever lie close to a query. By pairing locality-preserving feature transforms with Compressive Sensing, Holo-NN folds the entire database offline into a compact holographic sensing matrix. At query time, encrypted search reduces to a single, non-interactive matrix-vector multiplication---requiring no FHE bootstrapping, no multi-round interaction, and no client-side index state. This turns private retrieval into an embarrassingly parallel, accuracy-tunable computation that runs out of the box on both CPUs and GPUs, scaling efficiently to production databases. We offer a comparison against other state-of-the-art Private ANN algorithms to help situate the method in the broader context of production-grade private retrieval. 

---
# How Much Fresh Data Should CTR Recalibration Use? A Temporal Evaluation of Post-hoc Calibration 

**Authors**: Rui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.33168)  

**Abstract**: Offline evaluation of click-through rate (CTR) models still defaults to random splits, although deployed models score future traffic and downstream systems consume calibrated probabilities. On Avazu (40.4M impressions) we train logistic regression (LR), DeepFM, and DCN-v2 under two rolling temporal origins and a random split, three seeds per arm, and recalibrate each test day with Platt scaling and isotonic regression fitted on the preceding hours. At origin 1's nearest test day, random splits report LR's expected calibration error (ECE) at 55-62% of the temporal value in every seed (the deep models' ratio is seed-unstable), and no model's ECE is monotone in temporal distance. With a 6-hour fitting slice, recalibration removes up to 61% of ECE in badly miscalibrated cells but increases it in about a third of applications (14-17 of 48 across seeds). On fixed evaluation sets, a 12-hour slice cuts harm from 13-17 to 3-5 of 36 applications, and 24 hours raises it again. The 12-hour slice subsampled to the 1-hour slice's size (5% of the impressions; 20-25x faster to fit) recovers 88% of the full slice's gain, so impression count alone does not explain it. A slice-ECE gate shows no benefit out-of-origin, and in a simulated auction recalibration moves profit by only about -0.21% to +0.26%. Offline protocols should measure calibration temporally, over multiple origins and seeds; in this data, recalibration fitted on a wide time span beat the freshest few hours, which other logs must confirm. 

---
# Lessons from Luck: Do LLM-Written Lessons Carry Chance into an Agent's Next Decision? 

**Authors**: Kazuki Nakayashiki  

**Link**: [PDF](https://arxiv.org/pdf/2610.33029)  

**Abstract**: Self-improving agents write lessons from the outcomes of their runs; when outcomes are noisy, a lesson can carry chance into later decisions. We measure one link of this chain in a registered study. In 256 synthetic worlds, each a recurring task with two interchangeable tools of recorded reliability, a scripted previous run used the more reliable tool (a good decision) or the less reliable one (a bad decision), with an assigned outcome: success or a transient failure. A model wrote a lesson from the run with ReasoningBank's memory-writing instruction; a fresh session of the same model, holding the authoritative log and the previous result, chose the next run's first tool with one memory item: that lesson, the lesson from the other outcome, an irrelevant item, or an unreasoned directive. The protocol was frozen, deposited and timestamped before any confirmatory call; 11 LLM endpoints gave 90,084 answers. After a chance failure of the better tool, the panel's lessons carried the chance (registered reading luck-carrying; interaction +0.112, 95% interval [+0.097, +0.126]): readers named the worse tool in 11.6% of next runs with the lesson written from that failure, against 0.4% with an irrelevant item and 0.4% with the lesson written from a success. The effect was +0.042 where the log stated that failures are independent and +0.181 where it did not (descriptive). After a lucky success of the worse tool, the interaction was bounded below 0.10 (+0.002), a bound, not evidence of no harm. In exploratory readings, 6 endpoints carried the chance (up to +0.352); 5 were bounded below 0.10. In secondary comparisons, a three-sentence decision-quality note in the writing instruction lowered the worse-tool rate after a chance failure to 7.5%, and one instruction for both outcomes with a no-change option to 3.1%; neither met the registered harm-reduction rule at every such endpoint. 

---
