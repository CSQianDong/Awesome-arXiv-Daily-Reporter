# Retail Product Search: A Practical Approach at Target 

**Authors**: Darshan Sonagara, Qujiaheng Zhang, Ankit Singh, Alex Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.31498)  

**Abstract**: Search is one of the most important features in e-commerce, directly driving customer engagement and business growth. A good product search system must show both relevant and desirable results. However, retail search presents unique challenges. User intent can range from exact matches to open-ended discovery. Search systems must also balance multiple goals, such as relevance, revenue, and profit, while keeping response times low. Traditional keyword-based methods often fall short in handling natural language or semantic queries. Vector search helps alleviate these issues, but it can miss key intent signals or return low-precision results. In this paper, we present the design of a hybrid search system at Target that combines lexical and vector search. We describe our approach to data processing, embedding training, precision control for the final result set, multi-channel result fusion (where we compared fusion strategies and adopted weighted interleaving), and the performance optimizations used to maintain low latency for production deployment. Our method improves offline evaluation metrics, and in online A/B testing it raised click-through rate by 0.97%, order conversion by 0.98%, and demand per visitor by 1.10% over lexical-only search, while roughly halving zero-result searches. The resulting system is deployed at scale and serves millions of guests daily. 

---
# Enriching Sequential Recommendation with Graph Laplacian Positional Embeddings 

**Authors**: Ekaterina Trushkova, Artur Gimranov, Anton Lysenko  

**Link**: [PDF](https://arxiv.org/pdf/2609.31253)  

**Abstract**: Sequential recommenders typically rely on learnable positional embeddings to encode the order of user interactions. In this work, we ask whether this ordinal signal can be replaced by a structural one derived from the item space. We propose to use Laplacian positional embeddings in SASRec: we build an item co-occurrence graph from training interactions, compute eigenvectors of its symmetric normalized Laplacian, and use them as frozen graph-derived positional embeddings. The backbone architecture and training objective remain unchanged. Experiments on four public sequential-recommendation benchmarks show that this simple replacement improves SASRec performance on most ranking metrics and remains competitive with strong positional and temporal encoding baselines. These findings indicate that item-item graph structure can be an effective substitute for standard ordinal positional embeddings in sequential recommendation. 

---
# AgentRecommender: LLM Agents Enable Customizable Recommender Systems on the User Side 

**Authors**: Ryoma Sato  

**Link**: [PDF](https://arxiv.org/pdf/2609.31166)  

**Abstract**: Recommender systems have traditionally been developed for platforms. However, this has given rise to many phenomena that may be advantageous for platform lock-in but are a nuisance to users, such as clickbait, filter bubbles, and the spread of fake news. Recently, user-side recommender systems have been proposed as a new paradigm for solving this problem. If users deploy their own recommender systems, they are no longer at the mercy of the platform's interests. However, building a user-side recommender system is not trivial; in particular, customizing one for oneself requires additional data. We propose AgentRecommender, a method that leverages the investigation capability and internal knowledge of LLM agents to flexibly build user-side recommender systems without additional data. AgentRecommender allows users to easily create recommender systems tailored to their own preferences. 

---
# SPADE: Escaping the Popularity-Similarity Frontier to Measure Serendipitous Recommendations 

**Authors**: Tobias Vente, Maarten Peirsman, Noah Daniëls, Hannu Toivonen, Bart Goethals  

**Link**: [PDF](https://arxiv.org/pdf/2609.31164)  

**Abstract**: Recommender systems engineer serendipity to foster active exploration and break predictable consumption cycles. The problem with existing offline beyond-accuracy metrics is that they often either isolate historical similarity or global popularity. We aim to design an evaluation metric that examines similarity, popularity, and actual user relevance. To achieve this, we introduce SPADE (Serendipitous Pareto Distance Evaluation). SPADE maps all items into a two-dimensional space to directly calculate a user-specific Pareto frontier of maximally popular and historically similar items. The final serendipity score is then computed by averaging the minimum Euclidean distance from this boundary strictly for the correctly recommended test-set items. Evaluating SPADE across five datasets and five baseline algorithms confirms its effectiveness; our results show that the metric successfully prevents algorithms from exploiting beyond-accuracy measures with irrelevant or non-personalized recommendations, reliably isolating serendipitous discoveries. 

---
# KuaFu: Compressing Long User Behavior into Understanding at Billion Scale 

**Authors**: Jiahao Hui, Lin Zhu, Yishen Hu, Jingdong Shu, Zetai Jiang, Xining Ran, Ben Tan, Yeshou Cai, Gong Chen, Haijie Gu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.31045)  

**Abstract**: Conversational agents, generative recommenders, and personalized advertising all rest on one capability: understanding each user from raw behavior. Prevailing industrial practice is task-specific: for each task, a relevant subsequence is extracted from the full history and a dedicated model trained on it. In production it hits two bottlenecks. First, even after filtering, a single-task sequence stays extremely long: content-interest summarization reads several hundred items per user, tens of thousands of tokens once serialized as prompt text. Second, profiles are refreshed routinely: a billion users weekly, roughly 100K QPM in aggregate, which under a fixed GPU budget sets a hard throughput floor. Compression is therefore mandatory, yet truncation or coarse compression can silently distort the profile, introducing four hallucination types (fabrication, omission, date misattribution, broken logic) that, with no way to evaluate the compressed representation itself, surface only as diffuse degradation in downstream metrics. We present KuaFu, a unified behavior-compression layer whose minimal unit is one behavior item. A two-axis projector compresses each item into 2-4 tokens of width 128-256 (about 10x along the token axis, 20x along width; per-item cache 10 KB to 0.5 KB), with fidelity-oriented four-stage training and layered intermediate evaluation. Across four production profiling tasks it matches or exceeds uncompressed single-task production models on all five headline metrics, raises per-GPU throughput by 37%-350%, and saves 190 GPUs. On public benchmarks it nearly always beats prior compressors at the same compression ratio (up to +17.7 EM on out-of-domain MRQA); on RecBench, a 4B model surpasses its 8B counterpart by 1.90 points. KuaFu has run on the Tencent advertising and recommendation platform for ten months, lifting overall GMV by 1.37%. 

---
# QReason: Query-Focused Decoupled Chain-of-Thought for Efficient Passage Reranking 

**Authors**: Yang Zhang, Wenhan Liu, Qiannan Zhu, Mingming Li, Yuanfei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.30904)  

**Abstract**: Passage reranking plays a crucial role in information retrieval by refining the ordering of candidate passages to better reflect relevance. Existing listwise LLM rerankers with Chain-of-Thought (CoT) reasoning can handle complex queries effectively, but they suffer from substantial redundancy and high latency due to sliding-window strategies, which repeatedly generate highly similar CoTs. To address this, we propose QReason, a decoupled framework that separates query-focused reasoning from window-specific passage relevance assessment. Specifically, QReason introduces a dedicated rewriter that generates a ranking-oriented reasoning query once, capturing the query's core intent while avoiding redundant reasoning, and then reuses it across all windows with a non-reasoning reranker. The rewriter is trained via a two-stage process that first uses supervised fine-tuning with relevant-passage guidance through semantic evidence to produce deeply grounded, query-focused CoTs. It then applies reinforcement learning to align CoT generation with both the inference-time setting and the reranking objective, optimizing listwise metrics and passage-level discrimination to produce reusable reasoning chains for reranking. Experiments on the BRIGHT benchmark demonstrate that QReason significantly reduces redundant reasoning, achieves ranking performance comparable to or better than strong reasoning-based rerankers, and outperforms existing query rewriting models. 

---
# RecToolBench: Benchmarking Recommendation-Specific Tool Orchestration under Fuzzy User Intent 

**Authors**: Xiao Chen, Yicheng Zhao, Yingying Wu, Zhendong Chu, Changyi Ma, Qingsong Wen, Xuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2609.30717)  

**Abstract**: Recent advances in agentic recommender systems are shifting recommender systems from passive filtering engines to instruction-following agents that use external tools to resolve user intent. However, existing benchmarks often assume explicit user intent, simplified tool environments, or isolated function calls, leaving realistic tool orchestration for recommendation underexplored. To bridge this gap, we propose RecToolBench, a Model Context Protocol (MCP)-based benchmark for evaluating tool-using recommender agents under fuzzy user instructions. RecToolBench contains more than 1,200 executable tasks across three recommendation domains, 13 MCP servers, and 32 tools, spanning single-tool calls, parallel tool calls, sequential tool chains, and hybrid tool orchestration. We construct RecToolBench with a scalable synthesize--fuzzify--judge pipeline that generates executable fuzzy recommendation tasks, and evaluates agent trajectories using rule-based execution checks and rubric-based LLM evaluation. Experiments on representative LLMs show that syntactically valid tool calls do not guarantee successful recommendations. Models struggle with semantic parameter grounding, multi-step evidence integration, and grounded final recommendations, especially as orchestration complexity increases. Our results identify tool orchestration under fuzzy user intent as a major bottleneck for agentic recommender systems. Our data and code are available at this https URL. 

---
# Recommendation World Models for Future-State Control 

**Authors**: Jinfeng Xu, Zheyu Chen, Ziyue Peng, Jianheng Tang, Zheng Lin, Jing Yang, Puzhen Wu, Zheng Xing, Victor C. M. Leung  

**Link**: [PDF](https://arxiv.org/pdf/2609.30711)  

**Abstract**: Sequential recommendation optimizes which items to rank, while each displayed slate also shapes subsequent feedback and user state. We study how a trained ranker can support decisions about these future consequences. We introduce UA-TWM, a utility-anchored world-model interface that constructs nearby slate actions, estimates their target-relevant consequences, and selects an alternative subject to utility constraints. The reference slate serves as a fallback when no alternative qualifies. A logged-replay instantiation combines utility and target-gain estimates with calibrated failure-risk prediction; a closed-loop instantiation uses one-step state-action prediction and updates its decisions after observed feedback. We evaluate transfer across twelve sequential backbones on MovieLens-25M and KuaiRand-Pure, and repeated target-directed interaction in KuaiSim. Attaching the interface improves Recall@20, NDCG@20, and future-state alignment for every matched logged backbone. Selection ablations reveal the utility and risk costs of aggressive target pursuit, while closed-loop diagnostics isolate the contribution of action-conditioned prediction. Local consequence modeling thus enables target-aware selection around a trained sequential ranker. 

---
# Component Benchmark: Hierarchical Model Profiling for Large-scale Recommendation Systems 

**Authors**: Dharak Kharod, Yuzhen Huang, Zhou Wang, Jackie Xu, Fuzail Khan, Jacky Zhou, Hao Yan, Lidong Zhao, Xizhou Feng, Yvonne Liu, Karthik Jayaraman, Praveen Ramachandran, Vishwa Karia, Yashasvi Makin  

**Link**: [PDF](https://arxiv.org/pdf/2609.30656)  

**Abstract**: Large-scale recommendation models pose distinct, under-explored profiling challenges. Most recommendation model architectures are structurally heterogeneous, intermixing memory-bandwidth-bound operations, small compute-bound dense layers, dynamic shapes from jagged categorical features, and low-arithmetic-intensity operations. Recommendation models evolve rapidly as modeling engineers experiment with compositions, often written without visibility into hardware execution characteristics. Standard profiling tools offer either end-to-end throughput or operator-level traces, but cannot attribute performance to the submodules that practitioners reason about. We present Component Benchmark (CB), a profiling system that independently characterizes each submodule performance in a hierarchical manner, providing a tree-structured, interactive visualization that brings performance clarity to ML practitioners. At its core, CB provides a simple yet extensible, submodule-based benchmarking framework with a plugin architecture that enables hierarchical performance analysis. These large-scale recommendation models are TB-scale, run on thousands of GPUs and ingest 100B examples per day. We demonstrate CB's effectiveness on common open sourced models and discuss how CB has been leveraged to accelerate modern recommendation model performance analysis and optimization. 

---
# Embedding Subspace Partitioning for Dynamic Multi-Objective Retrieval 

**Authors**: Shaobo Zhang, Alice Leung, Yunxiang Ren, Ping Liu, Yuchin Juan, Qianqi Shen, Benjamin Le, Jianqiang Shen, Chengming Jiang, Ko-Cheng Wang, Vidya Krishnamurthy, Caleb Johnson, Fedor Borisyuk, Luke Simon, Jingwei Wu, Wenjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.30601)  

**Abstract**: Modern industrial recommender systems must optimize across competing objectives, balancing semantic relevance with business metrics such as engagement and revenue. While bi-encoders dominate large-scale retrieval due to their efficiency, they collapse these heterogeneous signals into a single static embedding space. This design creates a fundamental limitation: once trained, the retriever cannot adapt to shifting objective priorities at serving time without retraining. Moreover, joint optimization with multi-objective losses often induces interference between objectives, leading to suboptimal trade-offs. We propose Embedding Subspace Partitioning (ESP), a retrieval framework that decomposes the embedding into task-aware subspaces and replaces the single dot product with a weighted sum of per-subspace similarities, whose weights are tunable at serving time. For Transformer bi-encoders, ESP uses the model's native end-of-sequence token as a segment delimiter, with segment-aware attention masking and position encoding resets to guarantee subspace isolation in a single forward pass. Serving is performed via GPU-accelerated exhaustive kNN over one concatenated index, eliminating the need for per-objective Approximate Nearest Neighbor (ANN) infrastructure required by multi-head approaches. We evaluate ESP on an open-source benchmark built from MS MARCO. A single ESP model traces a broad Pareto frontier, consistently outperforming strong multi-task baselines across diverse operating points. In LinkedIn's job matching platform (70M+ weekly users), ESP enabled dynamic retrieval reconfiguration and delivered significant key business metric lifts. 

---
# Nearest but Not Dearest: Shared Curator-Feedback Infrastructure for Content-Only Search and Recommendation 

**Authors**: Matt Sandler  

**Link**: [PDF](https://arxiv.org/pdf/2609.30568)  

**Abstract**: A deployed B2B music-discovery platform serves both query-driven search (text prompts, vibe tags) and seed-driven recommendation (seed-track and artist stations) over one licensed catalog, one LAION-CLAP joint audio-text embedding space, one candidate-generation filter, and one ranking head -- and neither path consumes end-listener behavioral signal. In this content-only regime, curator judgment is the principal feedback signal available, and offline cosine similarity predicts it poorly: 38% of cosine-nearest neighbors are rejected by curators. The rejections reveal a clean partition: a majority (55%) are sound failures the encoder could address (style, tempo, mood mismatch), and a substantial minority (37%) are context failures orthogonal to the waveform (wrong language, holiday content, devotional content, rights and lyric flags). We deploy this sound-vs-context decomposition as feedback infrastructure, routing each failure mode to the layer that can absorb it: context failures to a constraint filter at candidate generation, sound failures to an embedding reweighting head at the representation layer -- both below the search/recommendation split, so a single curator loop maintains both experiences. On 1,200 curator judgments collected over two production rounds one month apart, the combined intervention reduces rejection rate from 38.17% to 28.83% (-24.5% relative, McNemar chi-squared = 22.4, p = 2.2e-6). An accounting decomposition attributes 4.08 pp of the drop to filter-eligible categories and 5.25 pp to the rest; the deployment was unblinded and compound, so this is a production accounting bound, not a causal estimate. We present this as an industrial case study rather than a validated general method, and close with lessons for content-only discovery: the failure partition is orthogonal to the paradigm partition, and corrections land in layers shared by both paradigms. 

---
# CG-Probes: Recovering Guardrail Directions from Patient Query Embeddings 

**Authors**: Marko Řeháček, Vítězslav Dušek, Martin Rusinko, Vít Nováček  

**Link**: [PDF](https://arxiv.org/pdf/2609.31062)  

**Abstract**: Patient-facing AI assistants promise valuable support to patients, but incoming queries can pose medical risks. To create guardrails, we work with oncologists to define three ordinal risk axes: Medical Urgency, Psychological Urgency, and Topic Sensitivity. We propose Clinical Guardrail Probes (CG-Probes) to measure the risks from query embeddings. We probe for each axis in the normalized embedding space of frozen embedders via the difference-in-means method, treating each axis as a potential linear direction. To train the probes, we cluster 79,658 Czech oncology search queries with BERTopic and use these clusters to generate pairs of queries with contrastive risk levels via few-shot prompting. We evaluate the approach on 200 queries (90 real, 110 synthetic), each graded by two oncologists, against two open-weight LLMs and a frontier LLM. We find that urgency-based axes are recoverable as linear directions, and the probes are competitive with open-weight LLMs (no significant differences in quadratic-weighted kappa) at a fraction of the latency. Each axis yields a scalar score that clinicians can inspect and use to set escalation thresholds. The pipeline requires only search logs, axis definitions, and black-box access to the embedding model, suggesting transferability across healthcare domains. Robust validation on new queries and axes remains future work. 

---
# Epstein Files Engine: Agentic Search for Investigative Journalism 

**Authors**: Duy K. Nguyen, Teresa Mondría Terol, Dylan Freedman, Zach Seward  

**Link**: [PDF](https://arxiv.org/pdf/2609.30611)  

**Abstract**: On Jan. 30, 2026, the U.S. Department of Justice released a mixed-media collection concerning Jeffrey Epstein, including about three million pages of PDFs. We describe the Epstein Files Engine, an A.I. agent The New York Times deployed to investigate the files. The Engine translated reporter questions into Google BigQuery SQL queries across three corpora: Epstein-related releases, the Times's archive and external, Epstein-related news headlines. It used an LLM to plan queries and returned citation-rich answers a reporter could verify and trust. More than 100 journalists used the Engine, and it contributed to at least 20 published stories. We report how reporters queried it and describe Diff, our text-and-visual duplicate matching method that amplified novelty signals and allowed the Engine to surface genuinely new information. We argue that newsroom agents serve newsrooms best not as autonomous writers, but as interfaces to source material and institutional knowledge. 

---
# T-RoPE: Time-Aware Rotary Position Embedding for Sequential Recommendation 

**Authors**: Yang Liu, Noel Loo, Ali Khanafer, Shuying Sun, Akshay Soni, Zhong Wu, Linjun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.30576)  

**Abstract**: Large-scale recommenders increasingly adopt the sequential generative recipe behind large language models, bringing the Transformer into recommendation along with design choices made for text, including Rotary Position Embedding (RoPE). In language models, RoPE encodes token indices for relative position reasoning, but in recommendation, an interaction index records only event order, saying nothing about elapsed time, behavioral cycles across scales, or calendar phase. We revisit this choice and propose T-RoPE, a time-aware RoPE for sequential generative recommendation that replaces index-only rotation with timestamp-based angles, learnable temporal coefficients, multiscale frequency banks, shifted query alignment, and non-stationary key rotation. We prove that standard RoPE, even on timestamps, remains time-translation invariant and cannot distinguish seasonal contexts, and that T-RoPE breaks this invariance while preserving the RoPE interface. Across five public benchmarks, T-RoPE achieves the best result on every metric on every dataset, improving over the strongest baseline by 78--130\% in HR@10 on the sparse PixelRec data and 8--12\% across metrics on Amazon Books. On an industrial-scale e-commerce dataset with more than 6B interactions, it improves every metric over the HSTU + Time RAB backbone by 13--82\%, with ablations attributing the largest gains to multiscale frequencies ($+56\%$ NDCG@50) and non-stationary keys ($+4\%$). An online A/B test in the Shop app yields positive lifts in conversion rate ($+0.33\%$) and order count ($+0.63\%$). We also provide forward and backward algorithms whose added cost is linear in sequence length and head dimension, keeping time-aware RoPE practical for large generative recommenders. 

---
# REALMS: An AI-Assistant Conversational System for Real-Time Exact Audience Sizing over High-Dimensional Nested Profiles 

**Authors**: Haixu Ma, Aditya Bansal, Shubham Lohiya, Sumit Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2609.30547)  

**Abstract**: Audience sizing is a critical component of digital marketing. It enables precise resource allocation, campaign planning, and performance optimization. Traditional approaches using skeleton audiences, sampling, or predictive modeling suffer from significant delays, estimation errors, and poor scalability over high-dimensional profile data. We present REALMS (Real-time Exact Audience sizing via LLM-based Multi-attribute Search), a conversational system for exact audience sizing deployed in production on an enterprise customer data platform. REALMS enables marketers to query massive profile stores with millions of profiles and thousands of attributes using natural language and receive precise counts in seconds. The system introduces three key components: (1) a categorical attribute retrieval mechanism using embedding-based vector search to dynamically identify relevant schema attributes without manual configuration; (2) an LLM-powered NL2SQL pipeline with template-based in-context learning for accurate query generation over complex nested schemas; and (3) schema standardization enabling industry-agnostic deployment across diverse enterprise environments. Evaluation on real enterprise data demonstrates strong recall for attribute retrieval, high SQL execution accuracy, and low latency, which enables real-time interactive audience insights where prior methods required hours. 

---
# AutoResearch at Production Scale: Failure Modes and a Multi-Agent Framework 

**Authors**: Aparajith Chandran, Juwon Kim, Saurav Jha, Pablo Castells, Florian Hottier  

**Link**: [PDF](https://arxiv.org/pdf/2609.30541)  

**Abstract**: Optimizing embedding systems for production recommendation pipelines demands systematic exploration that consumes disproportionate engineering effort at scale. We apply Andrej Karpathy's AutoResearch paradigm -- a large language model that iteratively edits a training script and retains modifications that improve a held-out scalar metric -- to automate this exploration. We report on twelve weeks of running this paradigm at production scale, where iterations consume hours of multi-GPU compute, evaluation involves competing criteria, and campaigns span weeks across many training jobs. Across two independently developed representation-learning systems for a book recommendation pipeline, we ran 220+ experiments and observed five recurring failure modes absent from the original setting: infrastructure fragility, agent memory decay, search-direction stagnation, iteration-cost asymmetry, and metric fixation. We contribute a three-principle scaffolding design -- prevent, persist, redirect -- that maps each failure mode to a structural remedy and whose instantiation scales with iteration cost. The framework produced a 1.82x Recall@6 lift and a 2.1x coherence lift over hand-tuned baselines, and the agent autonomously designed a text-only fallback that expanded catalog coverage by 5.8x. The two systems span nearly three orders of magnitude in per-iteration cost yet exhibit the same failure modes, suggesting these are structural properties of production-scale autonomous research rather than artifacts of either application. 

---
# Where Does Retrieval-Based Open-Ended Evaluation Fail? Automatic Taxonomy Induction from Long-Form Medical Answer Factuality Verification 

**Authors**: Heyuan Huang, Jirui Dai, Alexandra DeLucia, Sonal Joshi, Mahsa Yarmohammadi, Jie Gao, Bernal Jiménez Gutiérrez, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2609.30467)  

**Abstract**: Retrieval-based factuality evaluation, where LLM-generated claims are verified against evidence from authoritative medical corpora, has become the dominant paradigm for scalable hallucination detection in high-stakes clinical settings. Despite the urgency of reliable and transparent medical fact verification, most systems measure performance with aggregate metrics like F1, which obscure where and why failures occur. Existing RAG diagnostics require gold answers or annotated gold evidence, neither of which exists in this regime. We introduce two comprehensive taxonomies, grounded in a case study on the open-ended MedExpert dataset and 3 closed-ended datasets, decomposing failures into retrieval-stage errors along five quality dimensions, and verifier-reasoning errors into six consecutive steps. We adapt an automatic pattern induction pipeline using LLM-as-Judge to label evidence quality and classify verifier reasoning errors at scale, and then stress-test our findings across 4 retrieval methods and 6 frontier verifier models. Our analysis reveals that scaling model size, adding reasoning effort, expanding to authoritative web sources, and applying medical fine-tuning do not resolve these failure modes, demonstrating that they represent fundamental limitations of the retrieve-then-verify paradigm in open-ended medical settings rather than artifacts of outdated systems. We release our code and data at this https URL for the full reproducibility of our results. 

---
# Bootstrapping Conversational Recommendation Agents At Spotify: Synthetic Data Generation and Self-Improvement Loops 

**Authors**: Enrico Palumbo, Alexandre Tamborrino, Victor Ode, Ben Lacker, Adrià Casas Escoda, Jeremy Hopple, Marcus Better, James Leoni, Hugo Galvão, Hugues Bouchard, Mounia Lalmas, José Luis Redondo García, Abenezer Abebe, Ann Clifton, Anton Blomberg, Henrik Lindström, Dani Doro, Christine Doig Cardet  

**Link**: [PDF](https://arxiv.org/pdf/2609.30297)  

**Abstract**: Conversational recommendation agents are a new paradigm for content discovery, enabling users to express complex intents through natural language (e.g., "recommend Italian indie artists I haven't heard before"). A central challenge in building such agents is optimizing agent planning -- deciding how to select, sequence, and invoke tools -- particularly in cold-start settings where real user interactions are not yet available. We introduce a pipeline for multi-turn synthetic data generation and a self-improvement loop to address this challenge. The synthetic data pipeline transforms single-turn prompts into realistic multi-turn conversations, enabling systematic evaluation before launch. The self-improvement loop combines variance-based contrastive optimization with iterative refinement through a coding agent, automatically identifying and fixing planning and tool-use errors. Our approach improves quality by +8% on top of a highly optimized manual prompt. The system has been productionized and significantly accelerated iteration cycles for the launch of a conversational recommendation agent at Spotify. Online A/B tests demonstrate its effectiveness, with +14% user listening, +5% increase in weekly active users, and a 5% reduction in skip rate compared to a prior experience supporting only session refinement. This work provides a practical framework for accelerating the development of conversational recommendation agents in industry. 

---
# SignTrace: Describe a Sign, Find the Word 

**Authors**: Zengji Tu, Xingye Zhu, Ningjing Wang, Tingyi Huang, Yangjunfeng Zhu, Dai Wan  

**Link**: [PDF](https://arxiv.org/pdf/2609.30295)  

**Abstract**: Identifying an unfamiliar sign is difficult when a learner remembers its movement but does not know its meaning or formal feature codes. SignTrace addresses this longstanding reverse-lookup problem through natural-language access to a Chinese sign-language dictionary. The system integrates LLM-based dictionary enrichment, action extraction, dictionary-style rewriting, seven-channel retrieval, and candidate reranking over 6,699 entries. It has been deployed for user trials and has received positive informal feedback. Evaluation on a dictionary-derived benchmark of 500 movement-description queries yields 94.0% Hit@1, 97.4% Hit@9, and a mean reciprocal rank of 0.9540. Reranking increases Hit@1 from 71.8% to 94.0%, while component analyses show the contribution of enriched entry descriptions. Median query-processing time is 13.37 seconds with six concurrent queries. By connecting everyday movement descriptions to documented signs and meanings, SignTrace provides a practical tool for identifying unfamiliar signs. Dictionary-derived wording and prior selection within the benchmark limit generalization to descriptions independently produced by users. 

---
