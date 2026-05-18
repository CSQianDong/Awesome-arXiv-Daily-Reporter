# MERVIN: A Unified Framework for Multimodal Event Retrieval in Vietnamese News Videos 

**Authors**: Anh-Tai Pham-Nguyen, Tung-Duong Le-Duc, Anh-Duy Le, Trung-Hieu Truong-Le  

**Link**: [PDF](https://arxiv.org/pdf/2605.16120)  

**Abstract**: The growth of online video platforms drives the need for effective, semantically grounded event retrieval. We present MERVIN, a unified multimodal framework for Vietnamese news videos that integrates keyframes, transcripts, and video summaries. Transcript quality is enhanced via Gemini 1.5 Flash, reducing noise from accents, background sounds, and recognition errors. Visual features are extracted with Perception Encoder, while a Vietnamese language model produces textual embeddings; both are indexed in Milvus for efficient similarity-based retrieval. In addition, a React-based interface enables iterative query refinement across modalities, improving semantic alignment. Experimental results on Vietnamese news videos demonstrate the effectiveness of the proposed system, with MERVIN achieving 79 out of 88 points in AI Challenge HCMC 2025 qualification phase and successfully retrieved all results for every query in the final round. 

---
# Ascend-RaBitQ: Heterogeneous NPU-CPU Acceleration of Billion-Scale Similarity Search with 1-bit Quantization 

**Authors**: Fujun He, Chuyue Ye, Huaxiang Cai, Zetao Lv, Baolong Cui, Wenru Yan, Chao Zhan, Zigang Zhang, Hao Yi, Jie Xiang, Xiabing Li, Yuhang Gai, Ziyang Zhang, Pengfei Zheng, Yunfei Du  

**Link**: [PDF](https://arxiv.org/pdf/2605.16007)  

**Abstract**: Vector similarity search is a critical component of modern AI systems, but traditional CPU-based implementations face fundamental scalability bottlenecks for billion-scale corpora due to prohibitive computational overhead and memory bandwidth limitations. While Neural Processing Units (NPUs) offer orders-of-magnitude higher compute density, existing CPU/GPU-optimized 1-bit RaBitQ quantization implementations cannot be directly ported to NPU architectures due to fundamental hardware mismatches, and homogeneous design paradigms struggle to simultaneously balance accuracy, memory footprint, and performance. This paper presents Ascend-RaBitQ, the first heterogeneous NPU-CPU optimized IVF-RaBitQ system for billion-scale vector search, built on the core insight that decoupling coarse ranking (NPU) from fine ranking (CPU) allows each stage to leverage its optimal hardware, breaking the long-standing accuracy-memory-performance trade-off. We propose a three-stage heterogeneous pipeline comprising AI Core-accelerated coarse ranking on 1-bit quantized vectors, on-device AI CPU Top-k processing, and host CPU fine re-ranking on full-precision vectors. We introduce four NPU architecture-native optimizations: fused AIC-AIV operators for parallel distance computation, computation flow restructuring to exploit rotation orthogonality, fine-grained index block-level load balancing that breaks query boundaries, and intra-NPU pipeline parallelism between AI Core and AI CPU to mask Top-k latency. Evaluation on standard datasets shows that Ascend-RaBitQ achieves 3.0* to 62.8* faster index construction than the CPU baseline, up to 4.6* throughput improvement over the fastest CPU IVF-RaBitQ implementation, and over 100* over the mathematically equivalent CPU baseline, while demonstrating encouraging scalability on distributed multi-NPU systems. 

---
# Generative Long-term User Interest Modeling for Click-Through Rate Prediction 

**Authors**: Jiangli Shao, Kaifu Zheng, Hao Fang, Huimu Ye, Zhiwei Liu, Bo Zhang, Shu Han, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15905)  

**Abstract**: Modeling long-term user interests with massive historical user behaviors enhances click-through rate (CTR) prediction performance in advertising and recommendation systems. Typically, a two-stage framework is widely adopted, where a general search unit (GSU) first retrieves top-$k$ relevant behaviors towards the target item, and an exact search unit (ESU) generates interest features via tailored attention. However, current target-centered GSU would ignore other latent user interests, leading to incomplete and biased interest features. Additionally, the matching-based retrieval process in GSUs depends on the pairwise similarity score between target item and each historical behavior, which not only becomes time-consuming for online services as user behaviors continue to grow, but also overlooks the interaction information among user behaviors. To combat these problems, we propose a \textbf{Gen}erative \textbf{L}ong-term user \textbf{I}nterest model named GenLI for CTR prediction. GenLI consists of an interest generation module (IGM), a behavior retrieval module (BRM), and an interest fusion module (IFM). The IGM generates multiple interest distributions to indicate different aspects of real-time user interests, which is target-independent and incorporates interaction information among behaviors, ensuring complete and diverse interest features. The BRM selects related behaviors via a simple lookup operation, reducing the time complexity for weighting each behavior to $O(1)$. Finally, the IFM uses delicate gating mechanisms to generate interest features. Based on the generation process, GenLI improves the diversity of user interests and avoids complex matching-based behavioral retrieval, achieving a better balance between accuracy and efficiency for CTR prediction. 

---
# Jobs' AI Exposure Should Be Measured from Evidence, Not Model Priors 

**Authors**: Luca Mouchel, Pierre Bouquet, Yossi Sheffi  

**Link**: [PDF](https://arxiv.org/pdf/2605.15474)  

**Abstract**: This position paper argues that job exposure to AI should be measured with grounded, evidence-based methods, not inferred from LLM priors alone. Current theoretical exposure measures use zero-shot prompting to classify task-level AI exposure, generating labels with no explicit evidence, no transparent chain of reasoning, and no external validation. The stakes of these measurements are too high to rely on such methods, as they influence policy making, where public and private funds are directed, and how workers understand their future prospects. We therefore argue that AI capability claims should meet three standards: reproducibility, external grounding, and inspectability. We propose a retrieval-augmented framework that assigns AI exposure labels to all 18,796 occupation--task pairs in O*NET 30.2, using open-weight reasoning and instruct models with retrieved news articles and academic paper abstracts as evidence of current AI capabilities. Relative to a zero-shot baseline, the grounded condition is preferred in over 72\% of disagreement cases under both automatic and human evaluation, and yields scores that align more closely with observed real-world AI usage. Taken together, these findings suggest that evidence-grounded measurement better captures what current AI systems can plausibly do in practice, rather than what a model asserts without external evidence. Because AI capabilities continue to change, the measurements used to inform policy must evolve with them: theoretical AI exposure scores should be periodically reassessed, not inherited as immutable ground truth. 

---
# Differentially Private Motif-Preserving Multi-modal Hashing 

**Authors**: Zehua Cheng, Wei Dai, Jiahao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.15460)  

**Abstract**: Cross-modal hashing enables efficient retrieval by encoding images and text into compact binary codes. State-of-the-art methods rely on semantic similarity graphs derived from user interactions for supervision, yet these graphs encode sensitive behavioral patterns vulnerable to link reconstruction attacks. Existing privacy-preserving approaches fail on graph-structured data: Differentially Private SGD destroys relational motifs by treating samples independently, while graph synthesis methods suffer from unbounded local sensitivity in scale-free networks, hub nodes cause single-edge modifications to alter triangle counts by $\mathcal{O}(N)$, necessitating prohibitive noise injection. We term this phenomenon Hubness Explosion. We propose DMP-MH, a Sanitize-then-Distill framework that decouples privacy from representation learning. Our approach first bounds sensitivity by deterministically clipping node degrees, capping the $L_2$-sensitivity of triangle motifs independently of dataset size. A sanitized synthetic graph is then generated via Noisy Mirror Descent under $(\epsilon,\delta)$-Edge Differential Privacy. Finally, dual-stream hashing networks distill this topology using a holistic structural loss that enforces cross-modal alignment. Evaluated on MIRFlickr-25K and NUS-WIDE under a strict inductive protocol, DMP-MH outperforms private baselines by up to 11.4 mAP points while retaining up to 92.5% of non-private performance. 

---
# Fortress: A Case Study in Stabilizing Search Recommendations via Temporal Data Augmentation and Feature Pruning 

**Authors**: Milind Pandurang Jagre, Jia Huang, Dayvid V. R. Oliveira, Zhinan Cheng, Babak Seyed Aghazadeh, Puja Das, Chris Alvino, Jinda Han, Kailash Thiyagarajan  

**Link**: [PDF](https://arxiv.org/pdf/2605.15299)  

**Abstract**: In search and recommendation systems, predictive models often suffer from temporal instability when certain input features introduce volatility in output scores. This instability can degrade model reliability and user experience especially in multi-stage systems where consistent predictions are critical for downstream decision making. We introduce Fortress, a general framework for enhancing model stability and accuracy by identifying and pruning features that contribute to inconsistent prediction scores over time. Fortress leverages historical snapshots temporally partitioned datasets capturing score fluctuations for the same entity across periods and follows a four-step process: (1) collect historical snapshots, (2) identify samples with unstable predictions, (3) isolate and remove instability-inducing features, and (4) retrain models using only stable features. While semantic features from LLMs and BERT-based models improve generalization, they often lack full query or entity coverage. Engagement-based features offer strong predictive power but tend to introduce temporal instability. Fortress mitigates this trade-off by suppressing the volatility of engagement signals while retaining their predictive value leading to more stable and accurate models. We validate Fortress on a query-to-app relevance model in a large-scale app marketplace. Offline experiments demonstrate notable improvements in prediction stability (measured by Coefficient of Variation) and classification performance (measured by PR-AUC). 

---
# An LLM-RAG Approach for Healthy Eating Index-Informed Personalized Food Recommendations 

**Authors**: Yibin Wang, Yanjie Yang, Grace Melo Guerrero, Rodolfo M. Nayga Jr., Azlan Zahid  

**Link**: [PDF](https://arxiv.org/pdf/2605.15213)  

**Abstract**: Diet quality is a leading determinant of chronic disease risk. Advances in artificial intelligence (AI) have enabled food recommendation systems to adapt suggestions to user preferences and health goals. However, most current systems rely on loosely curated food databases and provide limited connection to a validated index. In this study, we propose a Healthy Eating Index (HEI) informed retrieval-augmented generation (RAG) framework that combines standardized nutrition databases with large language models (LLMs) for personalized food recommendations. Our proposed method anchors retrieval in the National Health and Nutrition Examination Survey (NHANES) and the Food Patterns Equivalents Database (FPED). A food-level embedding space is constructed from FPED-derived textual descriptions. For each entity, the system computes baseline HEI scores, retrieves candidate foods for intake recommendations, and estimates the HEI impact of simple substitutions or additions. A constrained RAG pipeline instantiated with a pretrained OpenAI LLM generates personalized recommendations and sources based on nutrient profiles and HEI contributions. The simulation results showed a mean HEI improvement of 6.45, with the proportion of users HEI over 50 increasing from 45.12 to 61.26. Quantile analysis revealed consistent improved shifts across the HEI distribution. Our findings suggest that the proposed LLM-RAG-based AI systems can support more precise, explainable, and personalized nutrition guidance to improve diet quality. 

---
# Agent4POI: Agentic Context-Conditioned Affordance Reasoning for Multimodal Point-of-Interest Recommendation 

**Authors**: Jinze Wang, Yangchen Zeng, Tiehua Zhang, Lu Zhang, Yuze Liu, Yongchao Liu, Xingjun Ma, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.15203)  

**Abstract**: We introduce Agent4POI, the first POI recommendation framework that generates context-conditioned multimodal representations at recommendation time, rather than relying on static POI embeddings pre-computed independently of context. Existing multimodal systems encode each POI once as a static embedding, a design that precludes reasoning about why the same cafe affords solo work on Monday but group celebration on Friday evening. We formally prove that no pre-computed encoder can satisfy context-sensitive ranking under standard bilinear scoring, motivating inference-time item-side representation. Agent4POI inverts this computation: given a situational context, a four-phase LLM agent generates dynamic, context-specific affordance queries (Phase 1) and executes a five-step cross-modal chain-of-thought over image, review, and metadata evidence (Phase 2). The resulting uncertainty-aware affordance representation is grounded in Gibsonian affordance theory. These cross-modal verdicts form a structured, uncertainty-adjusted affordance representation (Phase 3), which is aligned with user preferences via a semantic caching system for low-latency ranking (Phase 4). On three POI benchmarks and three evaluation configurations (standard, cold-start, context-shift), Agent4POI achieves a 23.2% relative gain over the strongest baseline and degrades by only 7.5% under context-shift versus 16--17\% for the strongest baselines. In cold-start scenarios, Agent4POI outperforms the best content-based baseline by up to 2.4x, whereas ID-based methods fail to generalize. 

---
# Argus: Evidence Assembly for Scalable Deep Research Agents 

**Authors**: Zhen Zhang, Liangcai Su, Zhuo Chen, Xiang Lin, Haotian Xu, Simon Shaolei Du, Kaiyu Yang, Bo An, Lidong Bing, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16217)  

**Abstract**: Deep research agents have achieved remarkable progress on complex information seeking tasks. Even long ReAct style rollouts explore only a single trajectory, while recent state of the art systems scale inference time compute via parallel search and aggregation. Yet deep research answers are composed of complementary pieces of evidence, which parallel rollouts often duplicate rather than complete, yielding diminishing returns while pushing the aggregation context toward the model's limit. We propose Argus, an agentic system in which a Searcher and a Navigator cooperate to treat deep research as assembling a jigsaw from complementary evidence pieces, rather than brute forcing the whole answer in parallel. The Searcher collects evidence traces for a given sub-query through ReAct-style interaction. The Navigator maintains a shared evidence graph, verifying which pieces are still missing, dispatching Searchers to gather them, and reasoning over the completed graph to produce a source-traced final answer. We train the Navigator with reinforcement learning to verify, dispatch, and synthesize, while independently training the Searcher to remain a standard ReAct agent. The resulting Navigator supports rollouts with a single Searcher or many in parallel without retraining. With both Searcher and Navigator built on a 35B-A3B MoE backbone, Argus gains 5.5 points with a single Searcher and 12.7 points with 8 parallel Searchers, averaged over eight benchmarks. With 64 Searchers it reaches 86.2 on BrowseComp, surpassing every proprietary agent we benchmark, while the Navigator's reasoning context stays under 21.5K tokens. 

---
# paper.json: A Coordination Convention for LLM-Agent-Actionable Papers 

**Authors**: Arquimedes Canedo  

**Link**: [PDF](https://arxiv.org/pdf/2605.16194)  

**Abstract**: LLM agents routinely serve as first (and sometimes only) readers of academic papers, skimming for sub-claims, extracting reproducibility steps, and generalizing scope. Standard prose papers produce recurring failures in this role: sub-claims that cannot be cited at sub-paper granularity, scope overextension beyond what the paper tests, and figure commands buried in codebases rather than the paper itself. We propose `this http URL`, a companion JSON file that travels with the PDF and addresses each failure with a lightweight convention: stable claim IDs (C1), an explicit does-not-claim list (C2), exact per-figure shell commands (C3), and stable definition IDs (C5). A fifth convention (C4) holds that minimum viable compliance, hand-written JSON alongside the PDF, is achievable in under an hour for a finished paper without touching the human-readable output. C1, C2, C3, and C5 are open invitations: an agent that reads a compliant paper and acts on it produces evidence for or against them. This paper is itself compliant: `uv run this http URL this http URL --against this http URL` passes. Repo: this https URL 

---
# Fairness-Aware Retrieval Optimization for Retrieval-Augmented Generation 

**Authors**: Yingqi Zhao, Vasilis Efthymiou, Jyrki Nummenmaa, Kostas Stefanidis  

**Link**: [PDF](https://arxiv.org/pdf/2605.15790)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves reliability of large language models by incorporating external knowledge, but the retrieval process can introduce bias that propagates to generated outputs. This issue is particularly challenging in top-k settings, where multiple documents jointly influence generation. We propose a fairness-aware retrieval framework that models and controls this bias. Our approach combines controlled bias injection via reranking, a position-aware model of bias propagation, and an optimization formulation that balances relevance and fairness. We further introduce a scalable solution based on Quadratic Fairness via Dual Hyperplane Approximation (FARO), which enables efficient optimization through problem decomposition. Experimental results show that our method effectively mitigates generation bias while preserving relevance. This work provides a principled approach for fairness-aware retrieval in RAG systems. 

---
# X-SYNTH: Beyond Retrieval -- Enterprise Context Synthesis from Observed Human Attention 

**Authors**: Guruprasad Raghavan, George Nychis, Rohan Narayana Murthy  

**Link**: [PDF](https://arxiv.org/pdf/2605.15505)  

**Abstract**: In enterprise operations, the context required for an AI agent task is scattered across systems of record, static information stores, and communication channels. What is stored is system state, a lossy representation of the work that actually happened [2, 52]. The prevailing approach [17, 31, 34, 36] retrieves by matching request content to what is stored; for narrow requests this works well. But synthesis quality depends on knowing what to surface and how to interpret it: knowledge specific to each organization, team, and individual [5, 57, 61], present in behavioral patterns, absent from any retrieval index. For complex agentic tasks it breaks down: True Lead Rate is low, False Lead Rate is high, and the model has no mechanism to improve. We present X-SYNTH, a framework for enterprise context synthesis grounded in human attention, the digitally observable interaction signatures of each worker, encoding not just what they did but the sequence in which they did it, along with implicit reward signals. Behavioral traces preceding positive outcomes are distinguishable from those that did not, without external labeling. X-SYNTH models each individual's behavioral baseline as a Digital Twin Signature (DTS) and selects among seven qualitatively distinct attention filters: Proportional, Inverse, Differential, Recurrent, Comparative, Sequential, and Collective, per individual and per query, to identify causally relevant activity signatures. A four-stage pipeline assembles ranked context grounded in behavioral patterns rather than query embeddings. On a sales lead identification task, a frontier model unaided achieves 9.5% True Lead Rate (TLR) with 90.5% False Lead Rate (FLR). Augmented with X-SYNTH, TLR rises to 61.9% (6.5x) while FLR falls to 18.8%. Enterprise context synthesis is not a retrieval problem. It is a relevance problem, and human attention is its most reliable ground truth. 

---
# Automatic Construction of a Legal Citation Graph from 100 Million Ukrainian Court Decisions: Large-Scale Extraction, Topological Analysis, and Ontology-Driven Clustering 

**Authors**: Volodymyr Ovcharov  

**Link**: [PDF](https://arxiv.org/pdf/2605.15362)  

**Abstract**: Half a billion citation edges extracted from 100.7 million Ukrainian court decisions reveal that judicial citation structure encodes legal domain boundaries without supervision and predicts future legislative importance with near-perfect accuracy. We construct the first large-scale citation graph from the complete EDRSR registry (99.5 million full texts, 1.1 TB), extracting 502 million citation links across six types via regex on commodity hardware in approximately 5 hours, with precision of 1.00 on a 200-decision validation sample (95% Wilson CI: [0.982, 1.000]).
Three principal findings emerge. (1) The degree distribution follows a power law (alpha = 1.57 +/- 0.008), placing the Ukrainian court network near the EU Court of Justice and below the US Supreme Court, with hub articles cited by millions of decisions. (2) Louvain community detection on the co-citation projection recovers legal domain boundaries (civil, criminal, administrative, commercial) with modularity Q = 0.44-0.55 and temporal stability (NMI = 0.83-0.86 across periods), constituting an automatically constructed legal ontology grounded in judicial practice. (3) Citation features predict top-1000 articles with AUC = 0.9984, substantially outperforming a naive frequency baseline (P@1000 = 0.655); temporal dynamics detect legislative regime changes as phase transitions and the 2022 invasion as a citation entropy spike (H: 11.02 -> 13.49) with emergent wartime legislation nodes.
The citation-derived ontology is operationalized as the domain layer of a workflow memory system for LLM-assisted legal analysis, connecting to the ontology-controlled paradigm. The extraction pipeline, analysis code, and aggregated statistics are released as open data. 

---
# DeepSlide: From Artifacts to Presentation Delivery 

**Authors**: Ming Yang, Zhiwei Zhang, Jiahang Li, Haoseng Liu, Yuzheng Cai, Weiguo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.15202)  

**Abstract**: Presentations are a primary medium for scholarly communication, yet most AI slide generators optimize the artifact (a visually plausible deck) while under-optimizing the delivery process (pacing, narrative, and presentation preparation). We present DeepSlide, a human-in-the-loop multi-agent system that supports preparing the full presentation process, from requirement elicitation and time-budgeted narrative planning, to evidence-grounded slide--script generation, attention augmentation, and rehearsal support. DeepSlide integrates (i) a controllable logical-chain planner with per-node time budgets, (ii) a lightweight content-tree retriever for grounding, (iii) Markov-style sequential rendering with style inheritance, and (iv) sandboxed execution with minimal repair to ensure renderability. We further introduce a dual-scoreboard benchmark that cleanly separates static artifact quality from dynamic delivery excellence. Across 20 domains and diverse audience profiles, DeepSlide matches strong baselines on artifact quality while consistently achieving larger gains on delivery metrics, improving narrative flow, pacing precision, and slide--script synergy with clearer attention guidance. 

---
