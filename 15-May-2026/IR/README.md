# Discrimination Is Generation: Unifying Ranking and Retrieval from a Tokenizer Perspective 

**Authors**: Shuli Wang, Junwei Yin, Changhao Li, Senjie Kou, Chi Wang, Yinqiu Huang, Yinhua Zhu, Haitao Wang, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.14853)  

**Abstract**: Semantic IDs (SIDs) define the generation space of generative recommendation and directly determine its personalization ceiling. However, existing tokenizers are trained independently with retrieval objectives, leaving personalization signals fully decoupled from the SID construction process -- a fundamental gap that causes generative retrieval to persistently lag behind discriminative ranking. In this paper, we rethink the essence of SIDs: \emph{ranking seeks argmax in item space while retrieval seeks argmax in token space; both are the same problem solved at different granularities.} Based on this insight, we propose \DIG (\textbf{D}iscrimination \textbf{I}s \textbf{G}eneration), which embeds the tokenizer inside a discriminative ranking model for end-to-end training -- the ranker naturally becomes a retrieval model, yielding two models from a single training run. \DIG is organized around a \emph{feature assignment taxonomy}: item-intrinsic static features are encoded into SIDs, user-item cross features (u2i) implicitly drive codebook boundaries toward recommendation decision boundaries during training, and an MLP$_\mathrm{u2t}$ distillation module approximates u2i at the token level for inference. Experiments on three public benchmarks and two industrial datasets demonstrate that \DIG simultaneously improves ranking, retrieval, and unified retrieval-ranking quality. 

---
# Asymmetric Generative Recommendation via Multi-Expert Projection and Multi-Faceted Hierarchical Quantization 

**Authors**: Bin Huang, Xin Wang, Junwei Pan, Yongqi Zhou, Yifeng Zhou, Zhixiang Feng, Shudong Huang, Haijie Gu, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.14512)  

**Abstract**: Generative Recommendation (GenRec) models reformulate recommendation as a sequence generation task, representing items as discrete Semantic IDs used symmetrically as both inputs and prediction targets. We identify a critical dual-stage information bottleneck in this design: (1) the Input Bottleneck, where lossy quantization degrades fine-grained semantics, while popularity bias skews the learned representations toward frequent items, and (2) the Output Bottleneck, where imprecise discrete targets limit supervision quality. To address these issues, we propose AsymRec, an asymmetric continuous-discrete framework that decouples input and output representations. Specifically, Multi-expert Semantic Projection (MSP) maps continuous embeddings into the Transformer's hidden space via expert-specialized projections, preserving semantic richness and improving generalization to infrequent items. Multi-faceted Hierarchical Quantization (MHQ) constructs high-capacity, structured discrete targets through multi-view and multi-level quantization with semantic regularization, preventing dimensional collapse while retaining fine-grained distinctions. Extensive experiments demonstrate that AsymRec consistently outperforms state-of-the-art generative recommenders by an average of 15.8 %. The code will be released. 

---
# Stop Overthinking: Unlocking Efficient Listwise Reranking with Minimal Reasoning 

**Authors**: Danyang Liu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.14450)  

**Abstract**: Listwise reranking utilizing Large Language Models (LLMs) has achieved state-of-the-art retrieval effectiveness. Recently, reasoning-enhanced models have further pushed these boundaries by employing Chain-of-Thought (CoT) to perform deep comparative analysis of candidate documents. However, this performance gain comes at a prohibitive computational cost, as models often generate thousands of reasoning tokens before producing a final ranking. In this work, we investigate the relationship between reasoning length and ranking quality, revealing an overthinking phenomenon where extended reasoning yields diminishing returns. To address this, we propose a Length-Regularized Self-Distillation framework. We synthesize a dataset by sampling diverse reasoning traces from a teacher model (Rank-K) and applying a Pareto-inspired filter to select traces that achieve high ranking performance with minimal token usage. By fine-tuning on these concise, high-quality rationales, the student model learns to internalize efficient reasoning patterns, effectively pruning redundant deliberation. Experiments on TREC Deep Learning and NeuCLIR benchmarks demonstrate that our method maintains the teacher's effectiveness while reducing inference token consumption by 34%-37% across different retrieval settings, offering a practical solution for deploying reasoning-enhanced rerankers in latency-sensitive applications. 

---
# Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL 

**Authors**: Jianbo Zhu, Xing Fang, Jing Wang, Mingmin Jin, Bokang Wang, Guangxin Song, Zhenyu Xie, Junjie Bai  

**Link**: [PDF](https://arxiv.org/pdf/2605.14434)  

**Abstract**: Generative retrieval offers a promising alternative by unifying the fragmented multi-stage retrieval process into a single end-to-end model. However, its practical adoption in industrial e-commerce search remains challenging, given the massive and dynamic product catalogs, strict latency requirements, and the need to align retrieval with downstream ranking goals. In this work, we propose a retrieval framework tailored for real-world recall scenarios, positioning generative retrieval as a recall-stage supplement rather than an end-to-end replacement. Our method, CQ-SID (Category-and-Query constrained Semantic ID), employs category-aware and query-item contrastive learning along with Residual Quantized VAEs to encode items into hierarchical semantic cluster identifiers, significantly reducing beam search complexity. Additionally, we develop EG-GRPO (Expert-Guided Group Relative Policy Optimization), a reinforcement learning approach that aligns generative recall with downstream ranking under sparse rewards by injecting ground-truth samples to stabilize training. Offline experiments on TmallAPP search logs show that CQ-SID achieves up to 26.76% and 11.11% relative gains in semantic and personalized click hitrate over RQ-VAE baselines, while halving beam search size. EG-GRPO further improves multi-objective performance. Online A/B tests confirm gains in GMV (+1.15%) and UCTCVR (+0.40%). The generative recall channel now contributes substantially in production, accounting for over 50.25% of exposures, 58.96% of clicks, and 72.63% of purchases, demonstrating a viable path for deploying generative retrieval in real-world e-commerce systems. 

---
# Towards Self-Evolving Agentic Literature Retrieval 

**Authors**: Yuwen Du, Tian Jin, Jing Kang, Xianghe Pang, Jingyi Chai, Tingjia Miao, Fenyi Liu, WenHao Wang, Sikai Yao, Yuzhi Zhang, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.14306)  

**Abstract**: As large language models reshape scientific research, literature retrieval faces a twofold challenge: ensuring source authenticity while maintaining a deep comprehension of academic search intents. While reliable, traditional keyword-centric search fails to capture complex research intents. Frontier LLMs can handle complex research intents, but their high cost and tendency to hallucinate remain key limitations. Here we introduce PaSaMaster, a self-evolving agentic literature retrieval system that produces relevance-scored paper rankings with evidence-grounded recommendations through iterative intent analysis, retrieval, and ranking. It is built on three key designs. First, it transforms literature retrieval from a one shot query--document matching problem into a search process that evolves over time, using ranked evidence to reveal gaps, refine intents, and guide follow-up searches. Second, it prevents hallucinated sources by treating retrieval as intent--paper relevance ranking rather than generation. Finally, PaSaMaster improves cost efficiency by separating planning from retrieval: a frontier LLM is used only for intent understanding, while large scale retrieval and relevance scoring are delegated to customized corpora and lightweight models. Evaluated on the PaSaMaster Benchmark across 38 scientific disciplines, our system exposes the severe inaccuracy and incompleteness of traditional keyword retrieval (improving F1-score by 15.6X) and the unreliability of generative LLMs (which exhibit hallucination rates up to 37.79%). Remarkably, PaSaMaster outperforms GPT-5.2 by 30.0% at a mere 1% of the computational cost while ensuring zero source hallucination: this https URL 

---
# Thinking Ahead: Prospection-Guided Retrieval of Memory with Language Models 

**Authors**: Harshita Chopra, Krishna Kant Chintalapudi, Suman Nath, Ryen W. White, Chirag Shah  

**Link**: [PDF](https://arxiv.org/pdf/2605.14177)  

**Abstract**: Long-horizon personalization requires dialogue assistants to retrieve user-specific facts from extended interaction histories. In practice, many relevant facts often have low semanticsimilarity to the query under dense retrieval. Standard Retrieval-Augmented Generation (RAG) and GraphRAG systems are still largely retrospective: they rely on embedding similarity to the query or on fixed graph traversals, so they often miss facts that matter for the user's needs but lie far from the query in embedding space. Inspired by prospection, the human ability to use imagined futures as cues for recall, we introduce Prospection-Guided Retrieval (PGR), which decouples retrieval from how memories are stored. Given a user query, PGR first expands the goal into a short Tree-of-Thought (ToT) or linear chain of plausible next steps, and uses these steps as retrieval probes rather than relying on the original query alone. The facts retrieved by these probes are then used to personalize the next round of prospection, enabling PGR to uncover additional memories that become relevant only after the simulation is grounded in the user's history. We also introduce MemoryQuest, a challenging multi-session benchmark in which each query is annotated with 3--5 dated reference facts subject to a low query-reference similarity constraint. Across 1,625 queries spanning 185 user profiles from 3 publicly available datasets, PGR-TOT substantially improves retrieval, including nearly 3x recall on MemoryQuest over the strongest baseline. In pairwise LLM-as-judge comparisons against baselines, PGR-generated responses are preferred on 89--98% of queries, with blinded human annotations on held-out subsets showing the same trend. Overall, the results demonstrate that explicit prospection yields large gains in long-horizon retrieval and response quality relative to similarity-only baselines. 

---
# MemEye: A Visual-Centric Evaluation Framework for Multimodal Agent Memory 

**Authors**: Minghao Guo, Qingyue Jiao, Zeru Shi, Yihao Quan, Boxuan Zhang, Danrui Li, Liwei Che, Wujiang Xu, Shilong Liu, Zirui Liu, Mubbasir Kapadia, Vladimir Pavlovic, Jiang Liu, Mengdi Wang, Yiyu Shi, Dimitris N. Metaxas, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15128)  

**Abstract**: Long-term agent memory is increasingly multimodal, yet existing evaluations rarely test whether agents preserve the visual evidence needed for later reasoning. In prior work, many visually grounded questions can be answered using only captions or textual traces, allowing answers to be inferred without preserving the fine-grained visual evidence. Meanwhile, harder cases that require reasoning over changing visual states are largely absent. Therefore, we introduce MemEye, a framework that evaluates memory capabilities from two dimensions: one measures the granularity of decisive visual evidence (from scene-level to pixel-level evidence), and the other measures how retrieved evidence must be used (from single evidence to evolutionary synthesis). Under this framework, we construct a new benchmark across 8 life-scenario tasks, with ablation-driven validation gates for assessing answerability, shortcut resistance, visual necessity, and reasoning structure. By evaluating 13 memory methods across 4 VLM backbones, we show that current architectures still struggle to preserve fine-grained visual details and reason about state changes over time. Our findings show that long-term multimodal memory depends on evidence routing, temporal tracking, and detail extraction. 

---
# Why Neighborhoods Matter: Traversal Context and Provenance in Agentic GraphRAG 

**Authors**: Riccardo Terrenzi, Maximilian von Zastrow, Serkan Ayvaz  

**Link**: [PDF](https://arxiv.org/pdf/2605.15109)  

**Abstract**: Retrieval-Augmented Generation can improve factuality by grounding answers in external evidence, but Agentic GraphRAG complicates what it means for citations to be faithful. In these systems, an agent explores a knowledge graph before producing an answer and a small set of citations. We frame citation faithfulness as a trajectory-level problem: final citations should not only support the answer, but also account for the graph traversal, structure, and visited-but-uncited entities that may influence it. Through controlled ablation experiments, we compare the effects of isolating, removing, and masking cited and uncited graph entities. Our results show that cited evidence is often necessary, as removing it substantially changes answers and reduces accuracy. However, citations are not sufficient, because accurate answers can also depend on uncited traversal context and surrounding graph structure. These findings suggest that citation evaluation in Agentic GraphRAG should move beyond source support toward provenance over the broader retrieval trajectory. 

---
# Logging Policy Design for Off-Policy Evaluation 

**Authors**: Connor Douglas, Joel Persson, Foster Provost  

**Link**: [PDF](https://arxiv.org/pdf/2605.15108)  

**Abstract**: Off-policy evaluation (OPE) estimates the value of a target treatment policy (e.g., a recommender system) using data collected by a different logging policy. It enables high-stakes experimentation without live deployment, yet in practice accuracy depends heavily on the logging policy used to collect data for computing the estimate. We study how to design logging policies that minimize OPE error for given target policies. We characterize a fundamental reward-coverage tradeoff: concentrating probability mass on high-reward actions reduces variance but risks missing signal on actions the target policy may take. We propose a unifying framework for logging policy design and derive optimal policies in canonical informational regimes where the target policy and reward distribution are (i) known, (ii) unknown, and (iii) partially known through priors or noisy estimates at logging time. Our results provide actionable guidance for firms choosing among multiple candidate recommendation systems. We demonstrate the importance of treatment selection when gathering data for OPE, and describe theoretically optimal approaches when this is a firm's primary objective. We also distill practical design principles for selecting logging policies when operational constraints prevent implementing the theoretical optimum. 

---
# Croissant Baker: Metadata Generation for Discoverable, Governable, and Reusable ML Datasets 

**Authors**: Rafi Al Attrach, Rajna Fani, Sebastian Lobentanzer, Joan Giner-Miguelez, Debanshu Das, Varuni H. K., Nobin Sarwar, Rajat Ghosh, Anwai Archit, Surbhi Motghare, Christina Conrad Parry, Luis Oala, Lara Grosso, Joaquin Vanschoren, Steffen Vogler, Sujata Goswami, Eric S. Rosenthal, Marzyeh Ghassemi, Matthew McDermott, Tom Pollard  

**Link**: [PDF](https://arxiv.org/pdf/2605.15079)  

**Abstract**: Croissant has emerged as the metadata standard for machine learning datasets, providing a structured, JSON-LD-based format that makes dataset discovery, automated ingestion, and reproducible analysis machine-checkable across ML platforms. Adoption has accelerated, and NeurIPS now requires Croissant metadata in every submission to its dataset tracks. Yet in practice Croissant generation usually starts with uploading data to a public platform, a path infeasible for governed and large local repositories that hold much of the high-value data ML increasingly relies on. We release Croissant Baker, a local-first, open-source command-line tool that generates validated Croissant metadata directly from a dataset directory through a modular handler registry. We evaluate Croissant Baker on over 140 datasets, scaling to MIMIC-IV at 886 million rows and 374 Parquet files. On held-out comparisons against producer-authored or standards-derived ground truth, Croissant Baker reaches 97-100% agreement across multiple domains. 

---
# A Deterministic Agentic Workflow for HS Tariff Classification: Multi-Dimensional Rule Reasoning with Interpretable Decisions 

**Authors**: Yu Zhang, Dongjiang Zhuang, Qu Zhou, Zheng Huang, Junhe Wu, Jing Cao, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.14857)  

**Abstract**: Harmonized System (HS) tariff classification is a high-stakes, expert-level task in which a free-form product description must be mapped to a specific six- or eight-digit code under the General Interpretive Rules (GIR), section notes, chapter notes, and Explanatory Notes. The difficulty lies not in knowledge volume but in *multi-dimensional rule reasoning*: a correct classification must satisfy competing priority rules along several axes simultaneously, including material, form, function, essential character, the part-versus-whole boundary, and specific listing versus residual headings. End-to-end prompting of large language models fails characteristically by resolving one axis while ignoring the priority constraints on the others. We present a *deterministic agentic workflow* in contrast to self-planning agents: the control flow is fixed, language model calls are confined to narrow stages, and reflection and verification are retained as local mechanisms. This design yields interpretability by construction--each decision is decomposed into stage-wise structured outputs with verbatim citation of the chapter or section notes that bear on it. The architecture combines offline knowledge-engineering of the Chinese HS tariff with an online six-stage pipeline. Evaluated on HSCodeComp at the six-digit level, the workflow reaches 75.0% top-1 and 91.5% top-3 at four digits, and 64.2% top-1 and 78.3% top-3 at six digits with Qwen3.6-plus; an open-weight Qwen3.6-27B-FP8 backbone in non-thinking mode achieves 84.2% four-digit and 77.4% six-digit top-1 agreement with the frontier model. A two-stage manual audit of 226 six-digit disagreements suggests that a non-trivial fraction of HSCodeComp ground-truth labels may deviate from HS general rules; full adjudication records are released in the appendix as preliminary findings for community review. 

---
# Falkor-IRAC: Graph-Constrained Generation for Verified Legal Reasoning in Indian Judicial AI 

**Authors**: Joy Bose  

**Link**: [PDF](https://arxiv.org/pdf/2605.14665)  

**Abstract**: Legal reasoning is not semantic similarity search. A court judgment encodes constrained symbolic reasoning: precedent propagation, procedural state transitions, and statute-bound inference. These are properties that vector-based retrieval-augmented generation (RAG) cannot faithfully represent. Hallucinated precedents, outdated statute citations, and unsupported reasoning chains remain persistent failure modes in LLM-based legal AI, with real consequences for access to justice in high-caseload jurisdictions such as India. This paper presents Falkor-IRAC, a graph-constrained generation framework for Indian legal AI that grounds generation in structured reasoning over an IRAC (Issue, Rule, Analysis, Conclusion) knowledge graph. Judgments from the Supreme Court and High Courts of India are ingested as IRAC node structures enriched with procedural state transitions, precedent relationships, and statutory references, stored in FalkorDB for low-latency agentic traversal. At inference time, LLM-generated answers are accepted only if a valid supporting path can be traced through the graph, a check performed by a falsifiability oracle called the Verifier Agent. The system also detects doctrinal conflicts as a first-class output rather than silently resolving them. Falkor-IRAC is evaluated using graph-native metrics: citation grounding accuracy, path validity rate, hallucinated precedent rate, and conflict detection rate. These metrics are argued to be more appropriate for legal reasoning evaluation than BLEU and ROUGE. On a proof-of-concept corpus of 51 Supreme Court judgments, the Verifier Agent correctly validated citations on completed queries and correctly rejected fabricated citations. Evaluation against vector-only RAG baselines is left for future work, as is GPU-accelerated inference to address current timeout rates on CPU hardware. 

---
# A Picture is Worth a Thousand Words? An Empirical Study of Aggregation Strategies for Visual Financial Document Retrieval 

**Authors**: Ho Hung Lim, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.14581)  

**Abstract**: Visual RAG has offered an alternative to traditional RAG. It treats documents as images and uses vision encoders to obtain vision patch tokens. However, hundreds of patch tokens per document create retrieval and storage challenges in a vector database. Practical deployment requires aggregating them into a single vector. This raises a critical question: does single-vector aggregation lose key information in financial documents? We develop a diagnostic benchmark using financial documents where changes in single digits can lead to significant semantic shifts. Our experiments show that single-vector aggregation collapses different documents with almost identical vectors. Metrics show that the patch level detects semantic changes, and confirm that aggregation obscures these details. We identify global texture dominance as the root cause. Our findings are consistent across model scales, retrieval-optimized embeddings, and multiple mitigation strategies, highlighting significant risks for single-vector visual document retrieval in financial applications. 

---
# Think When Needed: Adaptive Reasoning-Driven Multimodal Embeddings with a Dual-LoRA Architecture 

**Authors**: Longxiang Zhang, Weilong Dai, Guanghao Zhang, Hao Jiang, Pipei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.14448)  

**Abstract**: Multimodal large language models (MLLMs) have emerged as a powerful backbone for multimodal embeddings. Recent methods introduce chain-of-thought (CoT) reasoning into the embedding pipeline to improve retrieval quality, but remain costly in both model size and inference cost. They typically employ separate reasoner and embedder with substantial parameter overhead, and generate CoT indiscriminately for every input. However, we observe that for simple inputs, discriminative embeddings already perform well, and redundant reasoning can even mislead the model, degrading performance. To address these limitations, we propose Think When Needed (TWN), a unified multimodal embedding framework with adaptive reasoning. TWN introduces a dual-LoRA architecture that attaches reasoning and embedding adapters to a shared frozen backbone, detaching gradients at their interface to mitigate gradient conflicts introduced by joint optimization while keeping parameters close to a single model. Building on this, an adaptive think mechanism uses a self-supervised routing gate to decide per input whether to generate CoT, skipping unnecessary reasoning to reduce inference overhead and even improve retrieval quality. We further explore embedding-guided RL to optimize CoT quality beyond supervised training. On the 78 tasks of MMEB-V2, TWN achieves state-of-the-art embedding quality while being substantially more efficient than existing generative methods, requiring only 3-5% additional parameters relative to the backbone and up to 50% fewer reasoning tokens compared to the full generative mode. 

---
