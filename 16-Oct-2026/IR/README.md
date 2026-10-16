# GACC-Bench: Benchmarking Goal-Adaptive Context Compression Ability across Large Language Models 

**Authors**: Weiting Liu, Zhiyu Pan, Shuhong Wang, Zhenquan Yao, Jungang Li, Li Zhao, Jian Jiang, Jianfeng Feng, Ai Han, Wenlian Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.21893)  

**Abstract**: Context compression in long-running applications of large language models (LLMs) must preserve goal-relevant information, often before specific downstream questions are known. We introduce GACC-Bench, a benchmark for Goal-Adaptive Context Compression that evaluates how reliably LLMs adapt information retention to specified high-level goals. For each source context, we derive one or more high-level goals by grouping source-grounded questions according to shared downstream objectives. The benchmark contains 500 goals and 2,571 questions over 281 document and agent-trace contexts from seven collections spanning diverse domains, with all instances reviewed by human experts. The compressor receives a source context, a goal, and a context-length budget, with the associated evaluation questions withheld. A fixed reader answers these questions individually using the same compressed context. Our evaluation of ten LLM compressors reveals substantial gaps to full-context performance even for the best-performing model, particularly in supporting correct answers to every question associated with a goal. Compared with goal-agnostic compression, target-goal conditioning consistently improves accuracy, whereas mismatched-goal conditioning reduces it. GACC-Bench is available at this https URL. 

---
# Aligning LLMs with Sparse Retrieval via Adaptive Sparse Supervision 

**Authors**: Tongna Chen, Minghan Li, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.21877)  

**Abstract**: While large language models (LLMs) offer strong semantic representations, their application to first-stage retrieval incurs substantial latency and storage costs. Neural sparse retrievers such as SPLADE provide efficient retrieval but rely on relatively compact encoders with limited representational capacity. To bridge this gap, we propose an LLM-based sparse retrieval framework with input-adaptive sparse alignment. Specifically, we introduce a dynamic $\alpha$-truncation strategy that selects input-specific supervision supports based on the cumulative activation mass of SPLADE representations. This strategy guides the LLM to preserve salient lexical signals while suppressing less informative activations, without relying on a fixed support budget. Subsequent retrieval fine-tuning combines contrastive learning, knowledge distillation, and sparsity regularization to improve retrieval effectiveness. Experiments on MS MARCO and six selected BEIR datasets demonstrate improvements over SPLADE++ while maintaining competitive effectiveness against LLM-based sparse retrievers. Compared with Lion-sp, our method reduces index storage by 6.4-fold and end-to-end query latency by over 11-fold, achieving a favorable effectiveness-efficiency trade-off. 

---
# VoteGEO-Bench: Measuring the Impact of Generative Engine Optimization on Electoral Visibility 

**Authors**: Shiqi Zhang, Xiaokui Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2610.21330)  

**Abstract**: Large language models (LLMs) are increasingly used for election information, yet little is known about how external web content shapes the political information they present. Generative engine optimization (GEO) rewrites web content to improve its visibility in generated responses. To investigate its potential risks in voting scenarios (VoteGEO), we introduce VoteGEO-Bench, a benchmark covering the 2024 U.S. presidential election and the 2026 Japanese House of Representatives election, combining election-related queries, historical webpages, and voter personas. We first analyze sources cited by generative engines and find that many citations come from lower-cost owned and open-platform sources, while some webpages remain visible across days. We then evaluate 11 rewriting strategies in 4 categories, examining target specification, effectiveness across engines and elections, and the role of voter personas. Rewriting without specifying a target rarely increases target exposure, whereas target-specified rewriting more often does. Strategies that improve page organization or strengthen a position through argumentation yield greater gains than adding supporting evidence or changing wording. For the two strongest strategies, gains remain broadly positive across evaluated U.S. personas, but persona conditioning shows no consistent effect on these gains. Explicit partisan identities can create baseline exposure gaps larger than the gain from rewriting one webpage. When partisanship is implicit in the voter profile, baseline shifts depend on how strongly the engine responds to that signal. 

---
# RecSoup: Efficient Model Merging Approach for Controllable Multi-Objective Recommendation 

**Authors**: Zidong Jia, Guofu Xie, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.21220)  

**Abstract**: Recommender systems often need to balance multiple conflicting objectives. Controllable Multi-Objective Recommendation (CMOR) enables dynamic adjustment of objective trade-offs at inference time without retraining, adapting to shifting user or system requirements. Existing CMOR approaches typically rely on output-level reranking or parameter-generating modules, which either limit the scope of feasible trade-offs or incur substantial training and inference cost. Model merging provides a promising alternative by reusing multiple objective-specific models via parameter combination. However, its direct application to recommender systems is challenging due to the lack of a shared linear regime and cross-objective interference. To address these issues, we propose RecSoup, a lightweight two-stage approach for effective CMOR via specialized model merging. RecSoup first mitigates parameter interference across objectives through a pruning strategy that jointly considers objective-specific importance and cross-objective relevance. It then performs adaptive parameter merging via a learnable preference-to-coefficient mapping, allowing dynamic trade-off control at inference time without joint retraining or additional generative modules. Theoretically, we prove that RecSoup outperforms the parameter generation paradigm in effectiveness and efficiency, as it can fully navigate the Pareto front without smoothness constraints. Extensive experiments on multiple benchmarks demonstrate that RecSoup achieves improved Pareto front quality and effective controllability while delivers 165--228x faster inference-time latency than parameter generation approaches. Code is available at this https URL. 

---
# Learning When to Think: Utility-Guided Adaptive Reasoning for Efficient Text Retrieval 

**Authors**: Yingtao Luo, Kangyi Zhao, Shiyi Du, Yubo Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.21030)  

**Abstract**: Reasoning-conditioned embeddings can improve retrieval by progressively refining semantic constraints. However, additional reasoning may introduce over-generalized or semantically noisy abstractions that weaken retrieval specificity. Existing methods largely supervise reasoning through final retrieval outcomes, providing limited guidance over the refinement process itself. We propose AURA, a framework that directly supervises intermediate reasoning states according to how well they separate relevant documents from hard distractors. AURA further trains a lightweight utility estimator over prefix-conditioned embeddings to adapt reasoning depth at inference time. On the reasoning-intensive BRIGHT benchmark, AURA-4B achieves 34.7 average nDCG@10 while using 2.8 reasoning steps and 96 generated tokens per query. Under matched deployment conditions, adaptive stopping reduces latency from 4.6 to 0.76 seconds relative to full-trajectory reasoning. Results on MS MARCO further suggest that this specialization need not substantially degrade conventional in-domain retrieval. 

---
# TransX Technical Report: Scaling Sequence Transduction for Recommendation 

**Authors**: Da Xu, Liyan Fang, Divya Venugopalan, Sunny Hsu, Xukai Wang, Rishav Roy Chowdhury, Cindy Liang, Nishant Satya Lakshmikanth  

**Link**: [PDF](https://arxiv.org/pdf/2610.20913)  

**Abstract**: We present TransX, a sequence-transduction framework and co-designed infrastructure for scaling industrial recommendation with explicit control over model-compute allocation and execution amortization. TransX separates sequence encoding, behavior memory, and candidate-conditioned fusion into independently scalable components. Training amortizes causal trajectory computation across behavioral prefixes and fans out memory states to candidates; serving disaggregates bulk prefill, nearline refresh, and online prediction through compact behavior-memory KV.
We observe anisotropic scaling returns: encoder scaling may saturate beyond moderate depth at fixed memory resolution and fusion capacity; increasing supervision shifts the allocation frontier toward candidate-conditioned fusion; and the memory resolution needed to preserve quality remains roughly stable as model capacity grows. We train TransX on click and conversion, post-train its shared behavioral interface for retrieval and preference ranking, and explore generative recommendation offline.
In separate production A/B experiments, the core TransX model improves CTR by 6.02% and conversion by 4.45%, while post-trained retrieval and preference-ranking variants improve CTR by 2.96% and conversion by 1.33%, respectively, over corresponding production baselines. Under matched training conditions, the training infrastructure increases supervised-target throughput by 84%. In serving, disaggregation reduces TransX's total serving cost by nearly 80% versus colocated serving at matched traffic and SLO, enabling approximately 5X the per-request model FLOPs of the HSTU-style production baseline at the same total serving cost. At matched per-request model FLOPs, TransX delivers approximately 4X higher request throughput per H200. These results show that online serving cost need not grow proportionally with recommendation model capacity. 

---
# Idea2Prior: Auditable Paper Retrieval and Ranking from Research Ideas 

**Authors**: Shitian Yang, Ruijie Tao, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20857)  

**Abstract**: Finding a paper from an informal research idea requires more than matching topic words: relevant work may use different terminology, and its distinguishing mechanism may be absent from the abstract. We present \system{}, an auditable literature-search system built on PaperArena. It combines multi-source discovery, identity verification, facet-based abstract assessment, semantic citation expansion, and joint Expert ranking. A retrieval-only mode ranks up to five papers from a shortlist of at most 60, using complete available abstracts and locally selected PDF evidence from five fixed acquisition attempts, without generating a long-form report. In a historical comparison covering ten queries from each of EASY, MEDIUM, and HARD, aggregate reported Hit@5 is 53.3\% for \system{}, 43.3\% for ChatGPT web Deep Research, and 36.7\% for Gemini web Deep Research under the documented accounting convention. The advantage is concentrated in EASY and MEDIUM; ChatGPT leads on HARD. Selected \system{} runs average 8 minutes 53 seconds, compared with user-reported web averages of 5.2 and 4.7 minutes. A separate nine-query API sample averages \$0.0945 per task; this is a historical cost observation, not a matched comparison with web subscription costs. Pairwise Top 5 coverage ranges from 6\% to 34\%, showing differing returned evidence sets. The system also exposes optional implementation analysis and citation-evolution tools. These results support targeted idea-to-paper retrieval while leaving general web research, forum search, and broader relevance evaluation open. this https URL 

---
# Information-Gain Calibrated On-Policy Distillation for Deep Search Agents 

**Authors**: Chenxi Li, Xueyang Feng, Songyang Liu, Chenxu Wang, Xiaoying Tang, Chaozhuo Li, Litian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.20664)  

**Abstract**: On-policy distillation (OPD) has recently emerged as a promising approach for transferring the capabilities of large agents to compact students. In deep search, where agents iteratively retrieve external evidence, a teacher's preference among plausible queries before retrieval may not predict which queries will return useful evidence. Standard OPD nevertheless supervises each student query from the preceding context alone, inducing a failure mode we term \textbf{\textit{evidence-blind supervision (EBS)}}. To understand this, we regroup the OPD objective by interaction turns and reveal two implicit priors: a \textit{budget prior} that allocates supervision by token volume, and a \textit{direction prior} that favors tokens preferred by the teacher, neither of which accounts for the evidence returned by the current query. Our empirical analysis shows that both priors contribute to EBS in practice: the first search turn accounts for 51.1\% of total absolute information gain but receives only 18.0\% of the supervision budget, while 43.7\% of search queries have teacher rewards that disagree in sign with post-retrieval information utility. To mitigate this, we propose \ours, which uses post-retrieval information gain (IG) to calibrate both priors: reallocating supervision toward information-critical turns while calibrating query preferences using the utility of retrieved evidence. Experiments on six multimodal search benchmarks show that \ours improves average accuracy by 3.4 and 1.7 percentage points for 2B and 4B students over standard OPD, enabling the 4B student to surpass its 8B teacher on MMSearch and BrowseComp-VL. 

---
# CARA: Causally Guided Retrieval Agent for Question Answering 

**Authors**: Weicheng Liu, Xiangcheng Wu, Depeng Xu, Haopeng Zhang, Xi Niu  

**Link**: [PDF](https://arxiv.org/pdf/2610.22017)  

**Abstract**: Retrieval-augmented generation over text-rich knowledge graphs requires identifying evidence that contributes to answering a question, rather than merely matching its textual content. Existing agentic retrievers adaptively balance global search and multi-hop exploration but primarily rely on textual similarity to guide retrieval. We propose CARA, a training-free, CAusally guided Retrieval Agent that explicitly evaluates the causal relevance of retrieved evidence through two complementary criteria: consequentiality, measuring whether adding evidence improves answerability, and essentiality, assessing whether removing evidence weakens answerability. CARA integrates these assessments with path consistency to guide graph exploration and employs reciprocal rank fusion to aggregate evidence from multiple agents. We evaluate CARA on three STaRK datasets and augment STaRK with ground-truth answers to enable downstream question answering evaluation. Experimental results demonstrate that CARA consistently outperforms existing retrievers. CARA also improves answer quality and faithfulness while examining fewer graph nodes and requiring fewer retrieval operations. These findings demonstrate that causal relevance assessment enables more focused and effective agentic graph retrieval, improving both retrieval performance and downstream question answering. 

---
# Retrieved but Not Read: Understanding Why Search-Augmented LLMs Fail to Use Available Evidence 

**Authors**: Owen Walters, Lihu Chen, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2610.21731)  

**Abstract**: Web search allows LLMs to access up-to-date information, but it remains unclear why they sometimes produce outdated answers even when relevant evidence is available. To study this problem, we construct temporal facts whose values changed after the models' training cutoffs. We evaluate one proprietary model under deployed web search and six open-weight model configurations under controlled retrieval. We find that models often answer from search snippets without opening the source pages, which is associated with higher rates of outdated answers and unfaithful citations. Our controlled experiments and layer-wise analysis reveal a gap between the availability of evidence and its effective use. Even when retrieved evidence influences internal predictions, models may still favour outdated information in their final answers. We hope these findings will help advance research on the reliable use of retrieved evidence in search-augmented LLMs. 

---
# Making every stage of medical device document assessment with large language models deterministic 

**Authors**: Ryosuke Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2610.21300)  

**Abstract**: To show that a medical device complies with regulation, experts judge whether documents and records meet each requirement and record the judgment for others to verify. For each assessment item that experts define, large language models could prepare a preliminary assessment: a label such as Satisfied or Not satisfied, a reason and quoted evidence. We sent fourteen commercial deployments of such models (seven models from three providers, through six routes) the same request repeatedly, with the deterministic sampling settings each accepted; every deployment returned differing assessments for at least one request, and in twelve the label changed. Reducing reasoning where the interface allows it did not reduce the variation, and prompts of retrieved passages removed it only for the smallest models. The same open-weight model was reproducible or not depending only on how it was served. A pipeline we built to be deterministic at every stage -- text extraction and optical character recognition (OCR), chunking, embedding, indexing, retrieval and inference -- returned byte-identical output at each stage across reruns, interruptions, edits, machines of one graphics processing unit (GPU) model and days, for blood glucose meter and infusion pump documents. OCR, embedding and approximate retrieval varied when their own serving was not made invariant, and outputs changed with the GPU model and with the inference engine's scheduling settings, which therefore belong in the recorded configuration. End-to-end determinism is achievable and gives a testable basis for audit and change control. 

---
# TRACER: Trajectory-Aware Tool Retrieval for LLM Agents 

**Authors**: Tanish Kher, Sarvesh DK, Abedelkadir Asi, Sparsha Chennarayapatna Sundaresh, Dezhi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2610.21267)  

**Abstract**: Modern agents select tools from registries, but exposing entire registries incurs high token costs and can impair selection as they grow. We present TRACER, a retrieval front-end that indexes observed tool-call sequences in a trie and ranks continuations using a learned blend of empirical and contextual priors. On a mixed retail-and-telecom $\tau^2$ environment, TRACER improves GPT-5.4's tool-selection accuracy by 4.7 points while reducing input tokens per step by 80%, exposing just 5 of 128 tools. In autonomous full-turn execution, it matches GPT-5.4's full-registry task success with 61% fewer input tokens per step. Across six environments and four models, TRACER improves tool-selection accuracy over full-registry selection in 21 of 24 model-environment pairs, with average gains of 6 to 10 points; the three exceptions trail by at most 2.2. We analyze failures in both regimes to identify directions for future work. 

---
# SSR: Structural Signal Retriever for Scalable Agent Discovery in LLM-Orchestrated Multi-Agent Systems 

**Authors**: Giancarlo Maldonado Cárdenas, Carlos A. Astudillo, Nelson L. S. da Fonseca  

**Link**: [PDF](https://arxiv.org/pdf/2610.21113)  

**Abstract**: Large-scale multi-agent ecosystems require mechanisms for discovering suitable agents without exposing entire registries to a large language model (LLM) orchestrator. The challenge is not only to retrieve relevant Agent Cards from a growing registry, but also to maintain a compact candidate context, since each additional candidate increases prompt size, inference cost, latency, and the risk of exceeding the model context window. We formulate this problem as candidate-context scalability and propose the Structural Signal Retriever (SSR), our deterministic and training-free retrieval method designed for Agent Card discovery. SSR derives structural lexical signals from capability-bearing Agent Card fields, weights them using inverse document frequency (IDF), and uses an Inverted Signal Index (ISI) for selective candidate scoring. We evaluate SSR using a synthetic agent benchmark, a public MS MARCO adaptation, and end-to-end LLM orchestration. At 1,000 agents, SSR reduces LLM input by 98.8% and orchestration latency by 70.9% relative to full-catalog orchestration. On MS MARCO, IDF improves retrieval effectiveness within the structural representation, while the ISI preserves retrieval effectiveness and reduces explicitly scored passages by 96.8%. End-to-end experiments show that candidate pre-selection maintains a compact candidate context as registry size grows and avoids the operational failures observed with full-catalog orchestration. These results support SSR as an Agent-Card-native, traceable retrieval mechanism for candidate-context scalability. 

---
# ReviewPilot: A Multi-Agent System for Screening and Data Extraction in Systematic Reviews 

**Authors**: Zhaoqian Xue, Shaoze Zhou, Xuexin Li, Junting Mu, Haozheng Liu, Zhongyin Wang, Yichun He, Renkai Ma, Hao Zhou, Sarah Bratt, Siyuan Ma, Chen Chen, Lingyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.21042)  

**Abstract**: Systematic literature reviews follow the PRISMA process of screening scholarly works and extracting evidence from included studies. Deep research tools based on large language models (LLMs) can search and summarize the literature but do not record which studies are excluded or where extracted values come from. We develop a conversational multi-agent system called ReviewPilot that automates the PRISMA workflow from paper screening to information extraction. Researchers can tailor criteria and schemas through dialogue, and a project memory keeps their configurations and revisions. Each screening decision comes with its reason and each extracted value with its source. On epidemiological expert reviews, ReviewPilot outperforms a single-prompt LLM and AgentSLR across three LLMs, with screening macro F1 up to 0.776 and extraction F1 up to 0.788. Case studies in biomedicine and human-computer interaction show close agreement with human annotators, suggesting that ReviewPilot can transfer beyond epidemiology. Code is available at this https URL. 

---
