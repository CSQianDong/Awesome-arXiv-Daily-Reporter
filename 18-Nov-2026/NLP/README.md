# What Does a Bias Head Measure? Non-Identifiability in Head-Level Bias Attribution 

**Authors**: S.M Seefat Alam  

**Link**: [PDF](https://arxiv.org/pdf/2611.23218)  

**Abstract**: Methods that localise social bias in large language models to individual attention heads and remove it by ablation are validated by their effect on a fairness metric, never by evidence that the statistic selecting the heads measures bias. We audit this on Qwen-2.5-7B-Instruct and Llama-3.1-8B-Instruct. The head map correlates at $\rho = -0.950$ and $-0.768$ with a label-blind response-length map, apparently identifying a surface confound. An intervention that breaks the length-label coupling rejects this reading: with the prompt held fixed, response length explains 1% of the statistic's variance. What does explain it is prompt identity, at 64% of variance, in an ordering that is not monotonic in length. Within any fixed prompt the fairness label, response length and the presence of a protected-group token are collinear at $\phi = 0.95$, partly by definition of the labelling rule, so no observational analysis can attribute the map to any of them, and no method in this family performs an experiment that could. The substantive claims fail independently: bias selectivity is not late-layer-concentrated on either model under either reference reading, and no head set we tested reduces the unfairness rate when ablated. Its complement, scored on the opposite label, reliably destroys refusal behaviour ($p = 2.7\times10^{-5}$), demonstrating that the pipeline detects effects at this scale. 

---
# Structure--Residual Inductive Biases for Language Model Pretraining 

**Authors**: Hai Huang, Bartosz Żuk, Rassul Magauin, Jiaxin Huang, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.23197)  

**Abstract**: Next-token prediction is relied upon to learn both predictive behavior and internal representations. We ask whether representation structure itself should be an explicit inductive bias. Motivated by the two-part coding view of Kolmogorov complexity, we study a structure--residual principle in which a representation separates structure from realization detail. We instantiate this principle in two complementary ways. Boundary Bottleneck Concept (BBC) routes information from token segments through boundary carriers while preserving local autoregressive computation. Continuous Concept (CC) regularizes representation geometry and induces a spectral decomposition into low-frequency structure and high-frequency residual. We derive coding bounds for both views and show that multiplicity-corrected boundary attention becomes exact as the within-segment path energy vanishes. Empirically, BBC and CC exhibit strikingly different depth preferences: BBC works best in early layers, whereas CC is most effective in later layers. At 300M training tokens, combining BBC and CC improves NLL by 1.44\% on GPT-2 Medium. These gains persist with training runs up to 20B tokens. Representation diagnostics show that BBC increases the relative predictive importance of boundary states, while CC reduces spectral residual energy. Together, these results suggest that structure--residual biases can improve language-model pretraining beyond next-token prediction. Code: this https URL 

---
# Polarity Composition Hypothesis: Causal Evidence from Negation in NLI 

**Authors**: Nhi Hoai Doan, Tatsuya Hiraoka, Munachiso Samuel Nwadike, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2611.23185)  

**Abstract**: We investigate how language models process negation in natural language inference (NLI). We propose the \textit{Polarity Composition Hypothesis}, a high-level causal model of this computation. Behavioral evaluations show that language-model predictions largely conform to the proposed hypothesis. Using Distributed Alignment Search, we identify causally effective subspaces aligned with the variables in the polarity-composition hypothesis. These representations transfer across negation forms and partially to a subset of MNLI. We release the dataset and code. 

---
# MoE--LoRA Methods for Multilingual Adaptation: A Systematic Empirical Study 

**Authors**: Nikit Srivastava, Joel Dag, René Speck, Axel-Cyrille Ngonga Ngomo  

**Link**: [PDF](https://arxiv.org/pdf/2611.23158)  

**Abstract**: Routed low-rank adapters offer a parameter-efficient means to improve the performance of large language models (LLMs) across tasks. Hence, it is assumed that they can also be used to improve the support of low-resource languages in LLMs, which currently serve English far better than the thousands of other languages spoken worldwide. However, this assumption has never been verified empirically under a shared multilingual evaluation protocol. We address exactly this gap by comparing eight low-rank adaptation (LoRA) methods on a frozen base model using 200 unique languages in more than 9{,}000 evaluation settings. Our results suggest that a reduced-scale search can identify viable configurations of LoRA methods but does not reliably predict their relative performance, as the top-ranking configuration is never best at scale. LoRA reliably improves language-model fit on translation data, consistently degrades reading comprehension, and benefits low-resource languages most. These findings suggest that LoRA methods should be regarded as targeted tools rather than general-purpose means for improving the multilingual capabilities of LLMs. 

---
# Dynamic Task Allocation with Dialogue 

**Authors**: Matthew Groholski, Weiwei Gu, Nakul Gopalan  

**Link**: [PDF](https://arxiv.org/pdf/2611.23119)  

**Abstract**: Effective multi-agent collaboration requires agents to adapt task allocation as goals and responsibilities evolve. Prior task-allocation frameworks often assume static goals, full observability, or rely on large language models (LLMs) without explicit, verifiable beliefs about task allocation or intent. These assumptions limit their applicability to real-world collaboration, where agents often lack full information about each other's actions, progress, or local environment. To support dynamic task allocation under partial observability, we introduce a communication-augmented Bayesian task-allocation framework. Our agent uses dialogue to reduce uncertainty about the environment and collaborators, while keeping decisions grounded in explicit probabilistic beliefs. We evaluate our framework in an Overcooked domain extended with dynamically arriving tasks under both full and partial object-space observability. Our communication agent consistently outperforms a vanilla LLM agent in both settings. Additionally, our framework performs comparably to a non-communicating task-allocation baseline under full observability, but substantially outperforms it under partial observability as task complexity increases by an average speed up of 0.585 and completes on average 18 timesteps faster. These results indicate that our agent is more efficient in dynamic task allocation than baselines in partially observable environments, and that communication is important as available information decreases. 

---
# Natural Language Programming: Rethinking Language, Memory, and Cognition in the Era of Generative AI 

**Authors**: Giovanni Adorni, Daniele Grosso  

**Link**: [PDF](https://arxiv.org/pdf/2611.23106)  

**Abstract**: Generative Artificial Intelligence is transforming the role of natural language in computational systems. In classical Natural Language Processing (NLP), language was treated as an object of computation: something to analyse, classify, translate, or generate. Large Language Models (LLMs) challenge this perspective, enabling natural language to function increasingly as an executable cognitive medium through which humans structure reasoning, orchestrate problem solving, and regulate hybrid human/AI workflows.
This paper introduces Natural Language Programming (NLPg) as a theoretical and epistemological extension of traditional NLP. Rather than focusing on language understanding alone, NLPg conceptualises prompts as dynamic cognitive artefacts that externalise goals, constraints, assumptions, validation criteria, and intermediate reasoning states. Prompting is thus not merely a technical interaction technique, but a form of dialogic cognitive programming operating within distributed human/AI memory architectures.
Building on theories of distributed cognition, extended mind, metacognition, and dialogic learning, the paper proposes the notion of Dialogic Working Memory, describing how prompts and conversational states function as temporary external memory structures supporting reasoning continuity and reflective regulation. We further map existing memory architectures (BTMA, SKGE, RMR) onto this framework, illustrating how episodic, semantic, and metacognitive layers support long-horizon conversational coherence.
Finally, implications for computational linguistics and AI/mediated knowledge systems are discussed, suggesting a post symbolic and post interface paradigm in which natural language operates simultaneously as communication medium, executable specification, cognitive scaffold, and distributed memory architecture. 

---
# What's in a Tree ? Hierarchical $n$-gram models as controlled approximators of language 

**Authors**: Aritra Das, Francesco Cagnetta, Darshil Doshi, Victor V. Albert, Maissam Barkeshli, Andrey Gromov  

**Link**: [PDF](https://arxiv.org/pdf/2611.23093)  

**Abstract**: Hierarchical models have been proposed as synthetic settings for studying language. However, their relevance to realistic language corpora remains underexplored. In this work, we introduce tree-grams, autoregressive generative $n$-gram models that build the conditional probability tensor $p(x_n \mid x_{1:n-1})$ through a hierarchical composition of order-2 and order-3 tensors. $n$ controls the maximal context window, whereas a rank hyperparameter $d$ controls model capacity, yielding an interpretable and systematically scalable family of approximators of language. We show how tree-grams outperform existing $n$-gram baselines, while remaining competitive with deep transformer-based language models. We demonstrate the utility of such controlled approximations by using tree-grams as generative models. In particular, we show that hyperparameter transfer under $\mu$P can be made more robust when narrower models are tuned on simpler data, generated by tree-grams with moderate $n$ and $d$. In addition, we show that entropy and effective range of correlations control performance gap after fine-tuning. This is because tree-grams allow us to explicitly control horizon and capacity. When allowed to route across candidate topologies, we observe emergent sparsity in tree-grams. Such concentrated routing permits pruning upto 95\% of topologies without significant change in performance. 

---
# Runtime Verification of Multiple Natural Language Criteria for Agent Governance 

**Authors**: Silviu Pitis, Parand A. Alamdari, Jessica Tang, Toryn Q. Klassen, Sheila A. McIlraith  

**Link**: [PDF](https://arxiv.org/pdf/2611.23074)  

**Abstract**: AI agents are governed by hundreds of natural language criteria, yet existing verifiers cannot efficiently evaluate an agent's output against a large set of such criteria and report which are satisfied, violated, or inapplicable. We introduce the VFM, a tree-attention verifier that encodes the shared context and target text, and scores N natural language criteria in parallel, returning independent per-criterion ternary verdicts. The VFM is pretrained on CriteriaBank, a corpus of 394K (context, target, criterion, verdict) tuples spanning over 321K distinct criterion strings. A finetuned VFM-4B outperforms GPT-5.4 on a synthetic governance benchmark. On the DynaBench multi-criterion benchmark, it outperforms DynaGuard-4B on trace-level failure detection, and its per-criterion verdicts localize the violated rule and support focused escalation to a larger chain-of-thought judge. We show that CriteriaBank pretraining improves cross-domain transfer, particularly in the low-data regime, while matching a strong direct-finetuning control at full data. On a single H100, VFM-4B evaluates 1,000 criteria against a 4K-token context in under 2s at <11 GB VRAM, making it suitable for runtime verification. 

---
# TPHC: Improving Hyper-Connections with Tensor-Parallel Grouped Updates 

**Authors**: Xinying Hu, You Wu, Haoyi Wu, Jianchao Tan, Kefeng Zhang, Yuchen Xie, Kewei Tu  

**Link**: [PDF](https://arxiv.org/pdf/2611.23026)  

**Abstract**: Hyper-Connections (HC) expand the residual stream of a Transformer into $N$ parallel streams, and manifold-constrained HC (mHC) stabilizes them at scale. However, mHC benefits little from more than four streams, partly because each sublayer writes a single output to all streams, limiting how differently they can be updated. In this work, we propose TPHC, which integrates tensor-parallel (TP) shards with HC streams to promote greater independence across streams. Specifically, we route a separate mixture of the residual streams to each TP shard and keep the individual shard outputs separate during write-back, so each stream can receive a different mixture of those outputs. This design makes the residual-stream updates more expressive without adding parameters or FLOPs to the attention or FFN sublayers. With native support for stream sharding under TP, TPHC requires only 9% more training time than mHC on an 8B model when both use four streams. Increasing the number of streams from four to sixteen raises its training time by only 19%. On a 1.1B model trained on 100B tokens, TPHC reduces validation loss by 0.7-1.0% and raises average zero-shot accuracy across six commonsense reasoning benchmarks by 0.3-0.7 percentage points relative to mHC at matched stream counts $N\in\{4,8,16\}$. Our code is available at this https URL. 

---
# Large Language Models Have Not Yet Transformed Science 

**Authors**: Ron Bekkerman, Nichole Li, Bahareh Kargar, Mark Somers  

**Link**: [PDF](https://arxiv.org/pdf/2611.22975)  

**Abstract**: The emergence of large language models (LLMs) in late 2022 has raised questions about the future of science, but assessing their impact requires understanding how scientific practices were already evolving. We analyze 2.5 million full-text papers across five disciplines from 2015 to 2025, tracing trends over a broad spectrum of lexical, structural, epistemological, and social dimensions. Aside from frequency shifts in some words, most pre-2023 trends continued with limited detectable breaks. Among the notable exceptions, growth in computer science coauthor teams accelerated significantly in Asia, while apparent declines in biomedical topic diversity largely reflected standardization in article formats. Our study establishes an empirical foundation for monitoring the health of global science as LLM adoption deepens. 

---
# Can Decision Models Replace LLMs in Real-World Contact Center Classification and Judging Tasks? 

**Authors**: Md Tahmid Rahman Laskar, Xue-Yong Fu, Shashi Bhushan TN  

**Link**: [PDF](https://arxiv.org/pdf/2611.22949)  

**Abstract**: Contact center systems often need to make bounded decisions, such as identifying a caller's intent, retrieving the knowledge-base article that answers a query, assessing agent compliance, or judging the output of AI models used in production. Decision models return a typed answer and its probability directly, making them a promising cost-effective alternative to generative Large Language Models (LLMs) in such bounded conversational tasks. However, their effectiveness in real-world contact center applications remains unclear. In this paper, we compare nine decision models against thirteen generative LLMs in zero-shot settings across eight contact center use cases: six classification tasks and two judging tasks. Our experiments demonstrate that many decision models can match the performance of frontier LLMs on contact center classification, and API-based closed decision models achieve this at the lowest per-request cost among the API models we evaluate. We further observe that their confident predictions are more accurate, enabling selective routing in real-world production scenarios. Across classification tasks, the main weakness of most decision models is abstention: they rarely choose the no-answer option, which accounts for most of their gap to the strongest LLMs (although Gemini-Decision-1 is the notable exception in this regard). We also find that the judging quality of decision models drops sharply for smaller models, and they are also more prone to position bias. In short, our findings provide practical guidance for selecting decision models based on abstention behavior, confidence, and deployment cost, rather than overall accuracy alone. 

---
# The Shared Word: Evaluating Language Models on Dutch Word Puzzles 

**Authors**: Rik van Noord, Maurice Voors  

**Link**: [PDF](https://arxiv.org/pdf/2611.22935)  

**Abstract**: Many LLM benchmarks for languages other than English are translated, multiple-choice, or difficult to score automatically. We introduce The Shared Word (TSW-NL), a Dutch benchmark built around four core principles: it is language-specific, open-vocabulary, human-curated, and deterministically scored. Each puzzle gives three clues that describe different senses or uses of the same Dutch word. The clues are individually ambiguous, requiring the solver to combine evidence across multiple clues. We construct the benchmark from Dutch Wikipedia disambiguation pages, followed by automatic filtering and extensive manual curation. The final dataset contains 222 instances, with three independent human answers per test item. The task is clearly challenging: scores in our human study range from 49.5% to 61.0% and the highest-scoring open-weight model we tested reaches only 39.3%. Many others even score below 20%. At the same time, closed-source models show great progress: GPT-4.1 (April 2025) scores 59.2%, whereas GPT-5.6 Sol with max reasoning scores 88.8%. We believe this task provides a useful addition to Dutch model evaluation, plus a clear recipe that can be applied to construct a similar benchmark for other languages. 

---
# When Do Typed Decision Models Work in Transportation? Evidence from Travel Choice and Traffic Signal Control 

**Authors**: Hongyang Zhang, Boyuan Deng, Siyuan Feng, Kai Chen, Junbo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22926)  

**Abstract**: Typed decision models combine natural-language understanding with a structured interface that returns a selected action and an option-level probability distribution without generating lengthy textual responses. Their low inference cost and response latency make them attractive for transportation problems with predefined alternatives. This paper investigates their effectiveness, robustness, and efficiency in two representative transportation tasks: individual travel-mode choice prediction and sequential traffic signal control. We systematically evaluate TypeSafe Jev and GPT-6 Luna Decisions against conventional transportation methods, reinforcement learning algorithms, and four chat-based large language models. Experiments on the Swissmetro dataset show that typed decision models achieve lower prediction accuracy than traditional discrete choice models and the strongest chat-based baseline, while their performance varies substantially with input representation and question framing. In closed-loop traffic signal control experiments, typed decision models demonstrate competitive performance against established controllers in several scenarios, but exhibit performance degradation under more challenging network conditions. Furthermore, the typed models achieve lower inference latency and computational cost than the evaluated chat-based large language models, highlighting their potential for efficient, zero-shot transportation decision-making. Overall, these findings demonstrate both the potential and limitations of typed decision models, emphasizing the importance of input design, task characteristics, and robustness evaluation for their effective application in transportation systems. 

---
# When Passing Checks Do Not Establish Completion: Evidence Interpretation in Coding Agents 

**Authors**: Xueping Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.22835)  

**Abstract**: Verification procedures in coding agents must preserve execution facts, interpret their scope and discharge checking obligations. We study these responsibilities through qualified repository checks, controlled evidence consumers and executable workflows. A prospective thirteen-task factorial study finds that correct reported facts need not yield the requested finite verdict. Conversely, all 122 delivered labels in a held-out receipt-supply study are correct, while sixteen responses misreport facts and eleven others exceed authorized scope. Production repair supplies successful controls and finite-supported preservation failures. Across 118 repair episodes on 31 tasks, source interventions separate startup reachability from repair and witness production from the claimed operation. Lifetime and resource controls delimit what passing validation establishes. Generated-note comparisons yield local repair differences but can state false explanations; strong ordinary baselines, new tasks and Skill-transfer studies supply no consistent overall advantage. These results characterize evidence interpretation, applicability and fulfilled actions while limiting claims of general diagnostic superiority, Skill deterioration or deployed folding effects. 

---
# Compact and Decodable Electrocardiogram Text for Large Language Models 

**Authors**: Elif Dorukbaşı, İlker Türker  

**Link**: [PDF](https://arxiv.org/pdf/2611.22827)  

**Abstract**: When a twelve-lead electrocardiogram is written as text for a large language model, the number of tokens has to be balanced against the information lost through compression. This paper presents a deterministic encoding that writes eight independent leads as amplitude letters and exact time gaps and recovers the remaining four limb leads algebraically. The key points are selected with a cardiac-cycle-dependent tolerance, the amplitudes are parameterized by a 26-letter logarithmic codebook learned from a training set, and a 10-step tolerance schedule keeps all records within a fixed token budget. The encoding was applied to 21,388 records from the public PTB-XL database and evaluated using an exact decode-and-re-encode round trip, waveform reconstruction error, and a paired comparison of convolutional classifiers trained with 5 different seeds on signals from the original and decoded databases. All records were contained in 4,080 tokens (average: 2,866 tokens, 64.3% fewer than a per-sample transcription), and all stored lead strings were exactly reproduced. The average correlation between the decoded and reference signals was 0.982. The macro-averaged area under the receiver operating characteristic curve was 0.9167 for the reference signals and 0.9157 for the decoded signals (paired difference -0.0010; 95% patient-level bootstrap interval, -0.0028 to 0.0007), within the pre-specified non-inferiority margin of -0.010; this is not a confirmatory result, because the test fold had been used during encoder development. The diagnostic content of a compact text can therefore be preserved, while the language model's ability to read the text is yet another matter. 

---
# StitchRAG: Constructing Compact and Evidence-Preserving Contexts for RAG with Diffusion-Based Stitching 

**Authors**: Minsang Kim, Junpyo Lim, Heeje Cho, Seungjun Baek  

**Link**: [PDF](https://arxiv.org/pdf/2611.22766)  

**Abstract**: Retrieval-augmented generation (RAG) improves factuality by grounding large language models (LLMs) in external evidence. However, long retrieved contexts incur computational overhead and can even hurt performance due to noisy or irrelevant content. We propose \textsc{StitchRAG}, a framework for constructing compact context in RAG. \textsc{StitchRAG} builds a faithful, information-dense context while dynamically compressing the retrieved passages. We adapt the size reduction ratio to each passage by formulating a utility maximization trading-off the retained information and computation costs. We then extract answer candidates and query-related keywords from passages as anchors and collect passage sentences containing anchors under the derived size constraint. Finally, we connect or stitch the collected sentences together using text diffusion to form a coherent and natural context. Experiments on five open-domain QA benchmarks show that \textsc{StitchRAG} consistently outperforms diverse multi-step and compression baselines. Extensive ablations show the effectiveness of dynamic size allocation, anchor preservation, and diffusion-based stitching. 

---
# Cross-cultural Adaptation of Recipes for Low-Resource Languages: A Benchmark and Multilingual Culinary Lexicon for Hausa, Yorùbá, and Igbo 

**Authors**: Nuhu Ibrahim, Maryam Abdulazeez, Fatsuma Jauro, Salisu Aliyu, Robert Stevens, Riza Batista-Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2611.22708)  

**Abstract**: Recent multilingual large language models (LLMs) have improved machine translation, yet culturally adaptive translation for low-resource languages (LRLs) remains underexplored. This is particularly challenging for recipes, where ingredients, cooking tools, measurements, and procedures may require adaptation to the target cultural context. In this work, we introduce ReciLoReL, a benchmark of 600 expert-translated and culturally adapted recipes across Yorùbá, Hausa, and Igbo (three of the most prominent languages in Nigeria), and ReciLex, a multilingual culinary lexicon containing approximately 15,000 translated entities across ten recipe entity categories, including ingredients, tools, chef actions, state of tools, and durations. We evaluate state-of-the-art multilingual LLMs for English-to-LRL recipe translation under zero-shot and few-shot settings and investigate whether augmenting LLMs with ReciLex improves culturally adaptive recipe translation without requiring large target-language recipe corpora for training or fine-tuning. Beyond conventional reference-based translation evaluation using BLEU, chrF, and TER, we introduce five criteria for evaluating cultural adaptation, adapted from established cross-cultural adaptation methods. Across the selected LLMs and languages, ReciLex improves average BLEU from 13.57 to 14.51 and chrF from 37.89 to 38.74, while reducing TER from 77.14 to 75.12. Human evaluation further shows improvements ranging from 0.11 to 0.36 points across the five cultural adaptation evaluation criteria on a five-point scale. Overall, our work provides new resources, evaluation criteria, and empirical evaluation for studying culturally adaptive recipe translation in low-resource settings. 

---
# Route Optimization under Port Congestion and Carbon Reduction: An NSGA-II Approach 

**Authors**: Guanxi Liang, Sanglin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.22705)  

**Abstract**: Maritime shipping simultaneously faces the dual challenges of escalating port congestion and increasingly stringent decarbonization mandates. Large-scale vessel anchorage at hub ports inflates unit transport costs while generating significant additional \(CO_2\) emissions from idling auxiliary engines. This study develops a multi-objective routing optimization framework that jointly incorporates time, transport, and carbon emission costs under congestion-dependent generalized cost functions. A variational inequality (VI) model is formulated to characterize network equilibrium in vessel route allocation, while the Non-dominated Sorting Genetic Algorithm II (NSGA-II) is deployed to solve the resulting three-objective optimization problem. The model is empirically validated using China's coastal container network (Shanghai as the primary congested port, Ningbo-Zhoushan as the backup alternative), with sensitivity analyses conducted across congestion levels, carbon tax rates, and fuel price differentials. Key findings indicate that: (1) alternative ports effectively hedge congestion when diversion costs fall below waiting costs, with a diversion threshold at approximately 60 vessel calls; (2) a carbon tax of 80/t \(CO_2\) pushes clean fuel adoption above 50%; and (3) fuel prices primarily influence energy-type selection rather than port diversion decisions, confirming that the two decision dimensions operate independently. These findings offer actionable decision support for port authorities, regulators, and shipping operators seeking to balance economic efficiency with environmental sustainability. 

---
# Visibility Graph Representations of Word Embeddings for Text Classification 

**Authors**: Elif Dorukbaşı, İlker Türker  

**Link**: [PDF](https://arxiv.org/pdf/2611.22685)  

**Abstract**: Text classifiers depend on representations that expose the distinctions between classes in a form the learning model can use, and a restricted vocabulary changes which representation is most effective. This paper studies natural visibility graphs as an intermediate representation for news topic classification under controlled vocabulary budgets. Word embeddings are averaged into a fixed-length document vector, the vector is read as an ordered numerical series, and its natural visibility graph is passed to a classifier either as an ordered node-degree vector or as an adjacency matrix. On the four-class AG News benchmark, chi-square selection retains 100 to 1,500 words, and the graph inputs are compared with direct embedding-based representations from continuous bag-of-words, skip-gram, GloVe, FastText and BERT. With a fully connected classifier, the degree representation is the most accurate configuration from 700 selected words onward; it reaches its highest accuracy of 91.24% at 1,000 words and exceeds the BERT-based baseline by 0.29 percentage points at 1,500 words. In a separate convolutional comparison, the adjacency input reaches 86.16% at 1,000 words, against 83.12% for the strongest static embedding. The results indicate that visibility-derived features can be competitive once the selected vocabulary provides adequate coverage. We discuss the roles of graph compression, coordinate ordering and classifier choice, and the controls needed to establish the reliability of the small differences observed. 

---
# Contract-Explicit Evaluation: From Auditing Learned Readers to Scoped Certificates 

**Authors**: Hao Lu, Weixing Zhu, Ruiyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.22665)  

**Abstract**: Evaluation readings of document-level relation extraction (DocRE) systems depend jointly on the model and the evaluation contract: gold version, scoring function, threshold. Rankings reverse across contracts, yet no existing method determines which comparisons survive the reversal. We address this gap with contract-explicit evaluation: a preregistered five-step protocol (Declare, Measure, Scope, Certify, Hedge), operationalizing the classical concordance decomposition of Kendall's rank correlation, that scopes certifiable regions and delivers distribution-free census certificates across two ecosystems. In the DocRE audit (44 training runs of one in-house architecture family), the scoring dimension scopes to a census certificate (three flagged instances, Clopper--Pearson upper bound 0.146); the confirmatory record includes registered negatives, and the audit prices its own simplification finding---the calibration dial's entire margin is one withdrawn commit. In an LLM-judge census (17 judges, 49,938 battles), 12.25% of leaderboard comparisons require hedging at the pre-declared split criterion (bound 0.124; panel-robust 0.143), with an exploratory two-tier reading certifying 69% at a bound \le 0.072 and a validity decomposition against released human labels (65% of contested battles are judge error, not ambiguity). The cross-model certificate is withdrawn on real confidence scores; the failure spectrum maps the boundary. The protocol, map, ledger, and per-instance artifacts are released. 

---
# Understanding Cross-Modal Inconsistency through Representation Subspaces in Omni Models 

**Authors**: Seonghyeon Cho, Chanjun Park  

**Link**: [PDF](https://arxiv.org/pdf/2611.22619)  

**Abstract**: Omni Models process audio, vision, and text within a unified language backbone, yet often produce different answers for semantically equivalent inputs across modalities. Existing representation-level analyses do not distinguish between information that predicts whether disagreement will occur and the representation whose perturbation affects the model's answer. In this paper, we estimate a low-dimensional Answer Subspace from answer-position representations and use it to separate answer-related components from the remaining representation. Across three Omni Models and two multimodal reasoning tasks, information predictive of cross-modal inconsistency lies mainly outside the Answer Subspace, concentrating in a reproducible two-dimensional representation that is not explained by a fixed modality offset. Matched interventions reveal distinct behavioral roles: perturbing the Answer Subspace substantially changes the model's predictions, whereas perturbing the disagreement representation, like a random subspace of the same dimensionality, leaves them almost unchanged. Our key finding is that cross-modal disagreement can be induced by misaligning modality-specific answer representations: perturbing the Answer Subspace for only one modality induces disagreement, whereas applying the same perturbation to both largely preserves agreement. Thus, the representation that predicts disagreement is distinct from the answer-level representation through which disagreement can be induced, showing that linear decodability does not imply behavioral efficacy. 

---
# AfriNORM: An Evaluation Benchmark for Normative Reasoning of LLMs Across African Cultures 

**Authors**: Ahmad Samuel Gali, Tadesse Destaw Belay, Idris Abdulmumin, Seid Muhie Yimam, Shamsuddeen Hassan Muhammad  

**Link**: [PDF](https://arxiv.org/pdf/2611.22580)  

**Abstract**: Large language models (LLMs) are increasingly deployed across various sectors. However, current evaluation benchmarks are insufficient to assess the applicability of these models in local African contexts, as most focus on linguistic fluency (grammar/translation) rather than cultural competence. The few existing benchmarks that measure general knowledge or social conventions do not capture the African deontic modality, i.e., the strict norms and taboos that govern diverse social interactions. In this paper, we introduce AfriNORM, a scenario-based, multiple-choice question dataset designed to measure and diagnose the normative reasoning limitations of LLMs when applied to African sociocultural contexts. We introduce two metrics: a Universal Commonsense Bias Score (UCBS) to quantify commonsense bias in LLM and a Distractor Bias Score (DBS) to quantify an LLM's limited understanding of African cultural norms. We also conduct a case study on the Yorùbá language, identifying confusion gaps in two closed-source LLMs, "Gemini 2.0 Flash" and "GPT-5", and one open-source LLM, "Llama 3.1 8B Instruct" when prompted in English versus Yorùbá. While their accuracy in English is high, their performance in Yorùbá is much lower, attributable to cultural confusion rather than simple bias, indicating that the models exhibit semantic confusion when reasoning about norms in Yorùbá. 

---
# JET: Just-Enough Exploration Tuning for Efficient Tool-Calling Agents in Repository-Level Question Answering 

**Authors**: Yoonho Choi, Taehyeon Han, Seonghyeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.22539)  

**Abstract**: In repository-level question answering (QA), an agent explores a code repository with tools and should answer a question once enough information is gathered. Nevertheless, we find that LLM agents often continue to invoke tools even after the gathered information is already sufficient to answer. We call this phenomenon over-exploration. Moreover, even though existing methods improve task performance through supervised fine-tuning (SFT) on expert trajectories, SFT also propagates the over-exploration in the trajectories to the agent. To mitigate over-exploration, we propose Just-Enough Exploration Tuning (JET), a fine-tuning method that trains the agent to answer once the gathered information is sufficient. JET truncates each training trajectory at the step where the gathered information becomes sufficient, inserts a rationale that explains why the agent can answer at that step, and fine-tunes the agent with a higher loss weight on the rationale. Experiments on two repository-level QA benchmarks with two open-weight 8B models demonstrate that JET reduces tool calls by 11 to 33% on self-generated trajectories while largely preserving the score. On expert trajectories, JET makes 18 to 28% fewer tool calls than SFT while retaining most of the score gain of SFT. In both cases, JET also lowers the inference cost. 

---
# MM-ReportBench: A Comprehensive Benchmark for Multimodal Report Generation from Tables 

**Authors**: Teng Lin, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22514)  

**Abstract**: The ability to generate multimodal reports integrating fluent narratives with faithful visualizations from tabular data, has become a cornerstone capability for modern vision-language models in practical domains such as finance, healthcare, and policy making. However, existing benchmarks are limited by single-modality focus, simplistic metrics, novelty evaluation, and visualization quality. We address these limitations with MM-ReportBench, a multimodal report evaluation benchmark covering 386 tasks across 185 real-world tables. Each task is paired with an expert-curated reference report featuring standardized chapter structures and verifiable key insight points, constructed through a hybrid pipeline of multimodal reports generation based on Monte Carlo Tree Search and rigorous human quality control. For robust evaluation, we design a dual-judge evaluation framework: Text Judge Model assesses textual quality along structural completeness, grounded insight verification with numerical fact-checking against source tables, and novelty analysis that distinguishes genuine analytical discovery from trivial paraphrasing; Visualization Judge Model evaluates generated visualizations across data fidelity, expressiveness, aesthetics, and technical correctness. They also evaluate cross-modal consistency. Extensive experiments on 20 state-of-the-art models reveal critical weaknesses: even the strongest model achieves only 68.9 overall score, with numerical hallucination affecting 38.5% of reports and genuinely novel insights remaining rare. Our ablation studies further demonstrate that single-judge evaluation consistently inflates scores by 8-12 points compared to dual-judge assessment, confirming that rigorous and reliable benchmarking demands multi-faceted, complementary evaluation probes. 

---
# From syllable representation via gesture generation to articulation: A spiking neural model 

**Authors**: Bernd J. Kröger  

**Link**: [PDF](https://arxiv.org/pdf/2611.22499)  

**Abstract**: Speech production requires a transformation from phonological representations into coordinated vocal-tract movements. This paper presents a spiking neural sensorimotor model within the Neural Engineering Framework and Semantic Pointer Architecture. Each syllable occurrence has a neural oscillator whose evolving phase coordinates its gestures and initiates the next syllable. Phase-dependent commands recruit separate neural populations for vocalic, oral-constriction, velopharyngeal, and glottal actions (gestures). Neural ramp integrators shape the rise and fall of gesture activation. Together with articulatory targets, these trajectories determine the extent and time course of articulatory movements. Opponent neural populations provide control signals to a geometric articulatory model.
Thirty reference recordings cover an input corpus of eight lexical words, an alternative realization of one word, a syllable sequence, and isolated syllables. Two additional paired simulation subcorpora of sixty runs each, covering two words and one syllable sequence, assess activation reliability across neural realizations. Adjusting gesture command amplitudes increased successful realizations from 57/60 to 60/60 under the specified criteria, with ramp rate factors held constant.
The model demonstrates how neural oscillator and ramp dynamics generate articulatory timing under specified phasing rules. It complements a preceding mental-lexicon model through an ordered syllable interface. Continuous online execution across all components and connection to the acoustic extension remain further integration steps. Accessible code and synchronized visualization support inspection of the neural activation trajectories and resulting vocal-tract movements. 

---
# Growing Clusters Lower the Bar: Profile Accretion in LLM-Based Incremental Entity Resolution 

**Authors**: Huixue Han, Shangzhi Xu, Anqi Xiong, Yu-Chu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2611.22485)  

**Abstract**: Incremental entity resolution pipelines ask a large language model (LLM) whether a new record belongs to an existing cluster and show it the records already linked. Analyses of matching oracles assume that the answer depends only on the records compared. For the LLM matchers we test, it depends on cluster size, and the model lowers its bar for joining a cluster as the cluster grows. On French procurement suppliers from FOPPA, with SIREN identifiers as ground truth, we hold the new record fixed and vary only the number of verified members shown. Sixteen members instead of one raise the false match rate on similarly named companies by \dfmrMin--\dfmrMax{} percentage points in six open-weight models, while their area under the ROC curve rises by only \aucGainMin--\aucGainMax. In Claude Sonnet~5 and Haiku~5.5 the rise is \clSonD{} and \clHaiD{} points. The shift persists with street addresses, prompt variants and the BatchER and ComEM formats, and one false merge raises acceptance of the intruder's other records in every open model. It does not arise on Czech and Bulgarian TED notices, where one record with its full address already identifies the company. In online linking on confusable blocks, a single threshold set on labelled data raises recall by a factor of \fxRRmin--\fxRRmax{} for five models with no detectable loss of precision and no extra LLM calls. 

---
# LMMR: Learned Multigranular Memory Retrieval for Small Language Model Agents 

**Authors**: Pengfeng Song, Yunkai Zhuang, Liye Shi  

**Link**: [PDF](https://arxiv.org/pdf/2611.22425)  

**Abstract**: Small language model agents have limited space to read the memory needed for multi-hop questions. Relevant facts are often scattered across records whose surrounding text consumes much of that space. We propose LMMR, a learned multigranular memory retrieval method that selects both the next source and how much of it to read. LMMR links paragraphs and sentence fragments to their original sources and pairs views that supply the same missing support during training. Relevant distractors and repeated information provide contrasting examples. At inference, the reranker scores what each candidate adds to an anchor passage, and a selector chooses the evidence within a token budget. On 22,398 official validation questions from HotpotQA, 2WikiMultihopQA, and MuSiQue, a frozen 2B answer model reaches 61.47, 43.26, and 30.58 F1 under a 400-token budget. The gains over query expansion are 3.06, 2.14, and 4.55 points, while memory input decreases by 23.1-26.9%. LMMR also uses 23.5-27.7% fewer input tokens than a coarse retriever trained on the same questions. Against that retriever followed by LLMLingua-2 at comparable input lengths, F1 gains reach 3.99 points and are significant on HotpotQA and 2WikiMultihopQA after Holm correction. All eight small answer models evaluated on HotpotQA improve over query expansion and MMR in both F1 and exact match. 

---
# RICoT: Recurrent Interleaved Chain-of-Thought for Page Stream Segmentation 

**Authors**: Prashant Khatri, Bhavana Malagavalli, Saiful Haq, Niyati Chhaya  

**Link**: [PDF](https://arxiv.org/pdf/2611.22382)  

**Abstract**: Large language models (LLMs) have renewed interest in Page Stream Segmentation (PSS), splitting a multi-page file into its constituent documents. Yet published LLM methods still frame the task as boundary classification over one or a few neighboring pages, and commercial document-splitting APIs do not disclose their methods. We reimagine PSS for LLMs as a sequential process over the whole stream and propose RICoT (Recurrent Interleaved Chain-of-Thought), a prompting framework that works through the pages as a reader would: within a sliding window it reasons about each page and commits its boundary decision before moving on, and across windows it carries forward a compact record of what it has read, keeping per-call cost bounded. We also introduce FinancePSS-205, a benchmark of 205 real-world, de-identified financial page streams, to be released publicly. On FinancePSS-205, RICoT matches or outperforms pairwise and windowed prompting with the same backbone for each of three open-weight LLMs; on a TABME test subset, RICoT with GLM-5 reaches document F1 0.847 and boundary F1 0.915, above published zero-shot results. A 2B-parameter student, trained on a larger teacher's decisions with and without the page-level reasoning, performs comparably in both modes and comes within 0.05 document F1 of the strongest prompted model on FinancePSS-205. 

---
# Bad Apples in the Kitchen: How LLM Teams Detect, Correct, and Calibrate Against Their Misaligned Members 

**Authors**: Dongseok Heo, Junghwan Kim, Bongwon Suh  

**Link**: [PDF](https://arxiv.org/pdf/2611.22342)  

**Abstract**: What protects an LLM team when one teammate quietly goes wrong? Most multi-agent LLM work assumes cooperative partners and inspects chat between turns before acting. Real-time embodied cooperation does not allow this pause: chat runs alongside action, so a teammate whose words stay cooperative while behavior diverges can cause damage faster than the team can name what is happening. We study this case by injecting three theory-grounded misaligned agents into a custom four-player Overcooked environment, each stressing a different joint-action layer (speech-action mismatch, role invasion, and aggregate-progress disruption), and decompose robustness into three sequential gates of detection, correction, and calibration through a 3x4x2 design (n=720 episodes). Nearly all trolls are eventually named, but recovery follows only when attribution and situational diagnosis appear in the same utterance; direct source-attribution prompting improves the hardest case only by degrading the troll-free baseline by 37.5%. Robustness, then, depends less on detection alone than on whether team chat combines attribution with diagnosis in a single utterance, suggesting that sensitivity to misalignment should be targeted rather than uniformly raised. 

---
# Comparing Implicit and Explicit Language Attitudes toward the Dutch multiethnolect Straattaal in Humans and Large Language Models 

**Authors**: Hedwig Oldenhof, Laura Rosseel, Anna Wegmann, Dong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.22335)  

**Abstract**: While language attitudes have been extensively studied in humans, the extent to which large language models (LLMs) encode similar associations between language varieties and social attributes remains unclear. Contemporary sociolinguistic research distinguishes between explicit attitudes, which stem from deliberative processing, and implicit attitudes, which arise from automatic processing. This study investigates whether LLMs exhibit human-like implicit and explicit language attitudes toward the Dutch multiethnolect Straattaal, particularly its association with migrant communities. We use closely matched designs to assess attitudes in humans and LLMs. For humans, we use an Implicit Association Test (IAT) and an explicit rating task. For LLMs, we use the LLM Word Association Test (LLM WAT) and the same explicit rating task. Human participants strongly associate Straattaal with individuals with a migration background at both implicit and explicit levels. In the LLM WAT, Straattaal is consistently associated with individuals with a migration background, with stronger associations observed in larger and newer models. When presented with the same explicit rating task as human participants, LLMs produce comparable response patterns. Overall, the findings suggest that many LLMs encode language attitudes resembling those observed in humans, however we also highlight important questions about how concepts and measures developed to study human attitudes should be interpreted in LLMs. This highlights the relevance of incorporating sociolinguistic perspectives and language variation into NLP research to better identify and mitigate socially and linguistically embedded biases. 

---
# Zero-Dimensional Attention Topology for Linguistic Acceptability 

**Authors**: Saeed AlMarri  

**Link**: [PDF](https://arxiv.org/pdf/2611.22312)  

**Abstract**: Topological summaries of Transformer attention provide sentence representations for linguistic acceptability classification. We examine a zero-dimensional persistent-homology ($H_0$) representation that summarizes minimum-spanning-tree merge values from 144 symmetrized BERT attention graphs into 1,440 features. The retrospective evaluation uses 8,517 downstream training sentences and 521 in-domain and 497 out-of-domain CoLA validation sentences after data-integrity filtering. The recorded out-of-domain Matthews correlation coefficients are 0.438 for the topological representation and 0.508 for attention statistics, compared with 0.545 for BERT. Attention-feature results are retained aggregate measurements; BERT predictions are rescored and length controls are refitted from the accompanying records. Analytical results show that the representation retains token-count information and discards a class of directional attention differences. Source resampling of BERT predictions also demonstrates the sensitivity of uncertainty estimates to the sampling unit. The reported results do not demonstrate a predictive advantage for this fixed topological representation over simpler attention summaries. They identify representation properties and evaluation controls relevant to subsequent studies of attention topology. 

---
# OfficePrism: Benchmarking Agents on Single-Domain Operations Across Enterprise Business Functions 

**Authors**: Hanfeng Liu, Zhenyu Luo, Dongchao Zhang, Zexi Liu, Zulong Chen, Jinxin Hu, Jia Xu, Zeyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2611.22262)  

**Abstract**: Broad workplace scores can mix competence within one business function with the extra cost of coordinating across functions and systems, making failures hard to localize. OfficePrism evaluates this prerequisite through 420 single-domain enterprise tasks spanning seven business functions under shared, benchmark-defined L1--L3 task-structure levels. Each task runs in a fresh environment combining a shared synthetic organization with one function-specific domain and yped state access. Our pipeline uses grounded records, business rules, and tool schemas to jointly construct a Chinese-language ticket, a hidden executable reference witness, and an expected response, followed by automated filtering and human curation. The same task set is evaluated through full-schema and search-and-dispatch packages. Across 124 model--harness--package configurations, the per-case joint natural-language-and-database pass rate ranges from 22.6% to 76.4%, with a median of 67.9%. L1 is highest in all 124 configurations, and L2 exceeds L3 in 122. Among 50 configurations sharing the same harness and model labels across both packages, search scores higher in 30, ties in one, and scores lower in 19. 

---
# Where Does Hindsight Enter Clinical Agent Benchmarks? An Audit of Simulators and Case Construction 

**Authors**: Meilu Zhu, Xiaohan Li, Axiu Mao  

**Link**: [PDF](https://arxiv.org/pdf/2611.22241)  

**Abstract**: Benchmarks for LLM diagnostic agents are built from retrospective records: the case an agent reads is rewritten from discharge summaries, and unrecorded test results are generated by a simulator that often sees the final diagnosis. HindAudit separates three hindsight channels (the diagnosis given to the simulator, the retrospective context it conditions on, and the agent-facing case text) and measures each with judges trained only on real data. On 27,489 emergency admissions from MIMIC-IV, five registered tests found no leakage into simulated laboratory results: providing the diagnosis to five simulators had a negligible effect, and prompted gatekeepers diluted the diagnostic signal, although exploratory results suggest they can raise agent scores. The largest effect lay in the case text. Against a presentation-time reconstruction from triage records, benchmark-style summaries raised the zero-shot accuracy of four LLMs from 25.4% to 39.7% and discharge histories to 45.3%, and raised six interactive agents' scores while barely changing their ranking. The exploratory gap was reproduced in a registered replication on held-out patients. Masking diagnosis names left it unchanged. Because MIMIC lacks a contemporaneous bedside history, the gap is an upper bound on hindsight. We report all registered outcomes and recommend construction practices. 

---
# EGS: A Complementary Safety Framework for Educational LLMs 

**Authors**: Zexuan Fei, Zhengnan Guo, Yiming Luo, Kaiyi Huang, Aimin Zhou, Bo Jiang, Fei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2611.22203)  

**Abstract**: Large Language Models (LLMs) offer strong potential for personalized education, but adolescent-facing deployment raises risks of toxic, biased, misleading, or developmentally inappropriate content. We argue that educational safety should move beyond binary refusal: student queries are often ambiguous or curiosity-driven, requiring contextual guidance rather than simple blocking. We introduce the Educational Guardrail System (EGS), a lightweight, model-agnostic runtime framework that complements intrinsic alignment through query moderation, scenario-aware pedagogical guidance, and response moderation. EGS triages inputs by risk level, redirects gray-zone queries with category-specific educational interventions, and audits outputs before delivery. Experiments across general and education-specific benchmarks show that EGS improves moderation performance, refusal quality, and jailbreak robustness across diverse LLMs. These findings suggest that defense-in-depth guardrails can strengthen safety while preserving the pedagogical utility required in educational LLMs. Our code and model weights will be publicly available to support community adoption. 

---
# RISA: Recoverable Sparse Attention for Block-Diffusion Language Models 

**Authors**: Xinrui Zhong, Fanjiang Ye, Yuke Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22191)  

**Abstract**: Block-wise diffusion language models (DLMs) refine multiple tokens in parallel, offering an alternative to token-by-token autoregressive decoding. Their long-context inference, however, remains bottlenecked by memory-bound attention: every denoising step repeatedly reads a growing historical key--value (KV) cache. Directly applying sparse attention does not fully solve this problem, because different queries can select different pages, inflating the block's collective read set, while repeated support construction adds overhead and permanent eviction can discard information needed later. We present RISA, a recoverable sparse-attention method with shared read plans for block-wise DLMs. RISA exploits the fact that historical KV remains fixed within a block while queries evolve across denoising steps. Newly generated history is organized into exact priority and recovery tiers; seed queries then build a protected, bounded read plan whose indices are reused across the block while attention is recomputed with fresh queries and current-block KV. This separates history preservation from repeated reading and amortizes sparse-support construction without changing the backbone or decoder. On DreamReasoner-8B, RISA retains comparable task performance. At 32K, the default configuration speeds up attention by $2.67\times$ and fixed-trajectory requests by $1.36\times$ over dense, and attention speedup reaches $5.53\times$ at 128K. Across eight settings on three models, RISA decodes faster than the tested MAGE, QUEST, and LoSA budgets at matched KL. 

---
# From Imperfective Paradox to Moorean Absurdity: Single-Chain Reasoning Lock-In and Multi-Candidate Verbalization Across Aspectual and Pragmatic Ambiguity 

**Authors**: An-Sheng Jhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22188)  

**Abstract**: Natural language entailment and coherence judgments often depend on whether an event merely occurred, reached a telic culmination, satisfied a plan, or was evaluated under multiple pragmatic standards. We test observable consequences of the single-chain premature lock-in hypothesis across four preregistered, sealed formal experiments on Qwen3-8B-Q8_0 (16,640 formal units, temperature=0.0): 15 of 16 hypothesis slots retain confirmatory status (12 rejecting H0 under within-experiment Holm control at alpha_fam=0.05; E3_H3 demoted to descriptive status). In Experiment 1, Direct-CoT vs. MCU differs by 25.00 pp on telic high-confidence completion (Holm p=1.31239e-05), while NLI-CoT collapses by -91.25 pp on uninterrupted atelic items (Holm p=1.81805e-21). Multi-Candidate Verbalization (MCV) elicits K=5 weighted estimates in one call, separating the prompt from unweighted (MCU) and self-weighted (MCW) readouts. Mandarin (E2) and English (E3) expose an Interruption-Anchored Realis Paradox under NLI-CoT. In Experiment 4, MCU-CF raises perceived coherence on implicit Moorean sentences by 26.64 pp over Direct-CoT (Holm p=0.0006), alongside Direct-CF - Direct-CoT = 29.15 pp and MCU-CoT - Direct-CoT = 17.31 pp. A 1,800-call OpenAI ablation and a stage="dev" P16.7 cross-model extension (4,240 planned/collected units across five external models: 4,222 valid, 18 format-missing, 0 transport-missing; <2% missing per slice; OpenAI gpt-4o-mini at top_logprobs=20 achieves 1,000/1,000 valid units) show that discrete NLI-CoT and scalar Direct-CoT judgments dissociate across models. Confirmatory effects concern observable prompt-and-readout protocols; internal premature lock-in remains a hypothesis. All code, stimuli, preregistered analysis plans, and verification tools are available at this https URL. 

---
# What MoE Routing Controls: Causal Interventions on Expert Trajectories 

**Authors**: Mukund Athmaram, Manoj Rao, Liang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.22120)  

**Abstract**: In mixture-of-experts (MoE) language models, a token is routed to a sequence of experts across layers. We investigate how forcing tokens onto chosen expert paths influences model behavior. We build a target path from the experts that a concept's or role's tokens most often use, and we add a fixed bias to each target expert's router logit during generation to steer the token toward this expert path. We study two kinds of targets. Format targets concern the form of the output, and concept targets concern its topic. In Qwen3-30B-A3B, we show that steering a format path changes the text the model writes. We also show that steering a concept path shifts the model's next-token probabilities toward the concept, even when the path is built in another language, yet this shift has little effect on the generated text. We investigate why formats reach the text but concepts do not by steering only part of the generation, and find evidence that a format, once started, reinforces itself. We show that the concepts religion and money steer most strongly, and that the effect is concentrated in the middle layers. We attempt to explain why the strength of the effect varies by concept through an analysis of embedding distinctiveness and word frequency. 

---
# Beyond Acuity Accuracy: A Multidimensional Framework for Evaluating Patient-Facing Emergency Triage AI Tools 

**Authors**: Ruio Heerschap, Henry Foote, Mahmoud Alwakeel, Zhouhan Qian, Jiaqi Wang, Elisabeth Bond, Yang Yang, Jonathan Chong Kai Liew, Matthew A. Bergens, Monique Arnold, Haoyuan Wang, Rui Yang, Kunyu Yu, Nikita R. Paripati, Diana Pimenta, Kyle Naimoli, An-Kwok Ian Wong, Nicoleta Economou-Zavlanos, João Ricardo Vissoci, Nan Liu, Anand Chowdhury, Chuan Hong  

**Link**: [PDF](https://arxiv.org/pdf/2611.22045)  

**Abstract**: Patient-facing AI triage tools may support care-seeking and emergency department prioritization, but require evaluation beyond acuity accuracy alone. We developed SAFE-Triage, a framework integrating human review, automated metrics, LLM-as-judge evaluation, and simulation testing across clinical performance, patient interaction, and clinician-facing summary quality. We applied SAFE-Triage to 40 MIMIC-IV-derived emergency vignettes and a 10,030-case acuity analysis comparing prompted LLM triage (PLT) with rule-augmented prompted LLM triage (rPLT). PLT achieved higher emergency severity index (ESI) accuracy (57.5% vs 42.8%, $P < 0.001$), lower undertriage (13.1% vs 29.8%, $P < 0.001$), and higher clinician ratings across all domains. rPLT produced shorter and more stable outputs with higher automated readability scores. SAFE-Triage identified trade-offs between acuity accuracy, undertriage risk, communication quality, readability, and summary quality, supporting pre-deployment refinement and ongoing monitoring by showing where triage tools are efficient, need improvement, or need clinician oversight. 

---
# RLDE: A Reinforcement Learning Framework for Interpretable Iterative Document Editing 

**Authors**: Nikhil Parkar, M. Marimuthu, Riya Joshi, Karthikeyan N  

**Link**: [PDF](https://arxiv.org/pdf/2611.22039)  

**Abstract**: Improving the quality, clarity, and flow of long documents remains a challenging problem in NLP, as human editing typically occurs through iterative refinement rather than a single-pass rewrite. Most existing automated systems act as black-box generators or rely on supervised learning, limiting interpretability and failing to capture the sequential nature of revisions. In this paper, we propose RLDE, an automated end-to-end metric-driven RL framework that formulates document revision as a reference-free MDP problem (unsupervised with respect to paired document-revision targets). Working with a limited action space (NOOP, INSERT, DELETE, REPLACE), RLDE leverages Pointer Networks for dynamic span selection at the token level and sends generative actions to the frozen BART-base model to obtain context-aware suggestions. PPO agents learn to optimize a multi-objective reward function based on readability, perplexity, and grammar metrics with the help of a novel SBERT-based Semantic Preservation Penalty constraint to avoid deletion degeneracy. Evaluations on the BEA-2019 dataset showed statistically significant results (p < 0.01) across all metrics, where Flesch Reading Ease increased from 49.8 to 59.4, perplexity decreased from 63.7 to 39.8, while preserving 93% of semantic information. When compared to Chain-of-Thought (CoT) prompt-engineered models like GPT-4 and LLaMA-2, RLDE shows high performance at a very low computational inference cost. Moreover, the quality of edits generated by RLDE has been verified via human evaluations, scoring 4.9/5 for meaning preservation and 4.4/5 for edit quality. 

---
# GauntletScore: Auditable Fact-Checking That Matches the Best Published Checkers on LLM-AggreFact 

**Authors**: Ernie Hobbs, Jim Hinson  

**Link**: [PDF](https://arxiv.org/pdf/2611.22027)  

**Abstract**: Checking whether a generated sentence is supported by its source document is a standard safeguard for AI-written text, and the strongest published checkers are single models. We describe GauntletScore, an evidence engine whose few rules and thresholds were set on the benchmark's development split. It puts typed questions to independent readers from different providers and records each answer as a measurement in an evidence ledger. It combines the measurements with a fixed Bayesian rule that counts a contradiction only when two readers agree. On the full LLM-AggreFact test split (29,320 claims across 11 datasets), following the leaderboard's approach, GauntletScore achieves 79.51 balanced accuracy, above every average published on the leaderboard as of 15 November 2026. That places it level with the best of them, nimble-v4-max (79.45 from its published dataset scores; the engine's 95% interval, 78.47 to 80.53, contains it). On the same claims it beats two of its three readers used alone (Readers A and B) by 1.66 and 2.17 points (95% intervals 0.80 to 2.53 and 1.29 to 3.06). Without AggreFact-CNN, described below, the leads are 0.90 and 1.32 and both intervals stay above zero. The engine also records the evidence behind every verdict, which a single model does not. Its largest margin is on full-length news articles checked against multi-sentence summaries (AggreFact-CNN), a format that resembles much regulated writing. There it scores 76.45, the best on the leaderboard, against 70.1 for the next best and 63.9 to 67.8 for the six systems with the highest averages. The test split was run once, on an engine frozen before the first test claim, under a stopping plan committed in advance. We release per-claim predictions and the scoring script. 

---
# DiagTrace: An Interactive Benchmark for Clinical Decision Making 

**Authors**: Yufan Wang, Rui Yang, Yi Lin, Chunhua Weng, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2611.22009)  

**Abstract**: Large language models (LLMs) have shown strong performance on medical question answering and diagnostic reasoning benchmarks. However, existing evaluations typically provide all patient information upfront and assess only the final diagnosis or rationale. This setup ignores the need to acquire diagnostic evidence incrementally from incomplete information in real-world diagnosis. To bridge this gap, we introduce DiagTrace, a source-grounded benchmark for evaluating whether LLMs can reconstruct the documented diagnostic trajectory through interactive information seeking. Each benchmark instance is built from a PubMed Central case report and includes an initial presentation, an event bank, a clinical course, and diagnostic references. DiagTrace contains 510 cases across eight specialties, with 225 used to evaluate four LLMs on clinical-course alignment, interaction quality, and efficiency. Results reveal trade-offs in coverage, source-consistent ordering, and turn use. 

---
# Latent Fact Composition under Next-Token Training 

**Authors**: Tianyi Zhang, Feiying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.21921)  

**Abstract**: Controlled studies of latent two-hop reasoning train transformers to predict only the answer, whereas language models are trained to predict every token. We ask whether a model trained on every token learns to answer two-hop questions whose intermediate entity appears in no two-hop training example. We build a synthetic corpus with such questions. On this corpus every term of the loss can be computed exactly. A looped transformer, a standard transformer and DiscoLoop answer almost none of these questions. We therefore introduce the Baton family of looped transformers. They pass the intermediate entity to the second loop as a token embedding. When trained to predict only the answer, two of its three variants answer these questions almost perfectly. We then relate the success and failure of each model to three design conditions. Experiments that change one condition at a time agree with this hypothesis. When Baton is trained on every token, however, its accuracy on these questions rises and then declines. We further split the gradient of the next-token loss into three terms. Removing the two terms that depend on the particular training set raises the peak accuracy but does not remove the decline. The decline arises in the second hop while the model still recalls both facts. 

---
# Call-Efficient ICD-10 Code Prediction via Supervised Candidate Generation and Selective LLM Invocation 

**Authors**: Ilia Kopanichuk, Daniil Kartashev, Yuri Bolkonsky, Petr Anokhin, Vladimir Makharev, Iaroslav Bespalov, Dmitry V. Dylov  

**Link**: [PDF](https://arxiv.org/pdf/2611.21891)  

**Abstract**: Automatic assignment of ICD-10 codes from short clinical diagnosis text is a practical medical text classification and normalization task. The task is difficult because diagnosis phrases are short and noisy, the ICD-10 label space is hierarchical, and the class distribution is strongly imbalanced. This paper presents a call-efficient ICD-10 code prediction pipeline that combines transformer embeddings, tree-based supervised candidate generation, candidate-level reranking, and a gated LLM assistance. The evaluation uses a leakage-safe validation protocol: model and threshold selection are based on out-of-fold cross-validation predictions, while final reporting is performed on a locked test set with no shared source-file groups. The supervised conditional parent-code candidate generator reaches 74.32% Top-1 accuracy on the locked test set and 93.05% Top-10 candidate coverage. A direct zero-shot GPT-5.6-Sol baseline reaches 81.01% Top-1 accuracy, but requires an invocation of this large model for every diagnosis. The final deployable system, Pre-LLM fusion, reaches 81.24% OOF-CV accuracy and 83.19% locked-test accuracy while calling the LLM for only 31.97% of cases. This improves the locked-test error rate from 25.68% for the supervised parent-code baseline to 16.81%, corresponding to a 34.5% relative error reduction. The results show that the most effective role of the LLM is controlled reranking on uncertain examples rather than unrestricted coding or full-dataset invocation. 

---
# Knowing Which, Not How Much: Language Models Order Party Positions They Cannot Place 

**Authors**: Yousef Attia, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2611.21878)  

**Abstract**: Language models are increasingly asked where real actors stand: what a party's platform prioritises, what an institution has committed to. Such judgments are gradable, carrying both an order and a magnitude. Nearly every benchmark that tests them asks about one entity at a time, then reads that score as the model's knowledge of the domain. Order and magnitude, however, need not be acquired together. Text about social entities carries far more comparison than measurement, so a model may hold a dependable ordering with no anchored scale beneath it. We ask whether a model can place a documented social position on a scale or only order the actors it compares, and what that means for how such knowledge is measured. We test this on party positions in Japan and South Korea, where the Comparative Manifesto Project's coded data supplies a rare external answer key. Asked to order four parties, models beat the strongest fixed-answer baseline by more than a factor of two; asked to place one party on the same scale, they fall below it. The ordering survives inside those failed placements, and supplying the coded values lifts placement to near ceiling. Comparative and absolute questions about the same fact therefore measure different things here, and for party issue salience a benchmark built from one cannot speak for the other. 

---
# TUSI: A Citation-Grounded Graph-Based Benchmark 

**Authors**: Marzieh Karimi, Mehran Safayani, Alireza Basiri  

**Link**: [PDF](https://arxiv.org/pdf/2611.21823)  

**Abstract**: Evaluating retrieval-augmented generation (RAG) systems in Persian law remains challenging because benchmarks lack realism, grounding, and coverage of legal reasoning. Existing resources are either synthetically generated from statutory texts, lacking natural information needs, or derived from informal legal question--answer sources without explicit linkage to authoritative legal documents, making them unsuitable for rigorous evaluation of grounded legal reasoning. To address this gap, we introduce TUSI-Bench, a structured Persian legal benchmark designed for the evaluation of RAG-based systems. TUSI-Bench is built upon a multi-layered legal knowledge framework containing 7,261 legal articles and integrating codified statutes, legal commentaries, doctrinal writings, and Supreme Court unification rulings. We also construct a legal knowledge graph linking articles and concepts for structure-aware retrieval. Based on this foundation, TUSI-Bench provides two complementary evaluation tasks: (i) TUSI-QA, an open-ended question answering benchmark containing 1,999 questions that requires grounded responses with explicit legal citations, and (ii) TUSI-Bar, a multiple-choice benchmark containing approximately 500 four-option questions derived from Iranian Bar examination questions. Furthermore, we propose a graph-based retrieval baseline that leverages relational structure within the legal knowledge graph. Experimental results demonstrate that incorporating graph-based relations improves retrieval recall by up to 8 percentage points compared to standard baselines, highlighting the importance of structured legal representations for RAG systems. The proposed benchmark provides a standardized resource for evaluating grounded legal reasoning in Persian and enables future research on structured legal intelligence. 

---
# SimAI: Mechanistic User Simulation through Psychological Network Models 

**Authors**: Cheng Li, Adhyyan Narang, Jiayi Yuan, John F. Buss, Jina Suh, Devon Sandel-Fernandez, Yulia Tsvetkov, Adam M. Kuczynski, Tim Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2611.21808)  

**Abstract**: Realistic user simulation enables the study of human interactions when direct experimentation is costly or potentially harmful. However, existing simulators largely overlook the psychological mechanisms that drive user behavior. We introduce SimAI, a framework for mechanistic user simulation that models how psychological factors interact to shape user responses. To overcome the challenge of supervising open-ended behavior, we propose a graph reconstruction objective that provides graph-level rewards based on how well simulated conversations reflect the underlying psychological relationships. To support this approach, we construct PSYCHE, a dataset of 18,576 profile-network pairs derived from 516 psychotherapy cases. Automatic evaluations show that SimAI achieves greater simulation and behavioral diversity than existing patient simulators while maintaining strong profile alignment. In an interactive study with 30 mental-health professionals and trainees, SimAI receives the highest average ratings for conversational authenticity, realistic help-seeking behavior, and clinical training utility. Beyond simulation quality, SimAI enables realistic clinical training and uncovers safety failures in prolonged AI interactions with vulnerable users. Code is available at this https URL. 

---
# Corrupting Ties: Investigating Robustness to Bias Injection Across Multi-Agent Network Topologies 

**Authors**: Cléa Chataigner, Mina Arzaghi, Florian Carichon, Pablo Piantanida, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2611.21804)  

**Abstract**: AI systems are increasingly deployed in high-stakes decision-making contexts, where interactions among components can introduce fairness risks that remain undetected when auditing components or collective outcomes in isolation. As LLM-based multi-agent systems become more prevalent, understanding how biased information propagates through agent interactions is essential for assessing their fairness. We introduce a controlled bias injection framework that replaces selected agent responses with predefined biased ones, enabling an intervention-based analysis of how network topology and injection configurations shape system behavior. We propose decision divergence metrics that quantify the effects of these interventions at the agent, network, and system levels, and evaluate five topologies on three binary classification tasks using LLMs from the Qwen and Gemma families. Sequential networks show the greatest divergence, whereas fully connected and mesh networks show less. Injection position also matters: in star networks, injecting the central agent has larger effects than injecting peripheral agents. Models further differ in whether divergence follows the injected bias. These findings highlight the role of network structure in the propagation of biased information and motivate topology-aware auditing of fairness risks in multi-agent systems. 

---
# Disambiguating Moral Circuits and Moral Features in Language Models: Moral content, normative judgment or cue words? Evidence from 14 models 

**Authors**: Davide Liga, Liuwen Yu, Matteo Magnini  

**Link**: [PDF](https://arxiv.org/pdf/2611.21705)  

**Abstract**: Interpretability methods can locate the attention heads and sparse autoencoder (SAE) features that change the answer a language model gives to a moral prompt, e.g., whether "Stealing medicine for a dying child is morally" continues with "justified" or "wrong". Such a component could respond to moral content, to the judgment of conduct by any norm, or to the word "morally" in the prompt. We built 227 minimal pairs of short phrases, validated by five annotators, whose two versions differ in a moral standard, in another kind of norm (prudence, etiquette, aesthetics or evidence) or in a descriptive quality, and showed them to 14 open-weight models (Llama, Gemma and Qwen, 1B to 14B, base and instruction-tuned), with and without the word "morally". In every model, the heads that change the moral answer belong to an evaluation circuit the models use for every kind of normative judgment we tested: these heads carry the same share of the effect for moral as for other normative pairs (pooled ratio 1.00, 97.5% confidence interval [0.92, 1.07], with the word; 0.90, [0.84, 0.96], without it), and the models need them as much for both. The models need these heads also when nothing in the prompt asks for a judgment, and on 400 scenarios of the ETHICS benchmark, where removing them takes away 37 to 39% of the difference between acceptable and unacceptable scenarios, against 1 to 3% for ten random heads. The SAE features we located respond mostly to the word "morally" rather than to what the phrase is about. The models represent the category of a pair (the standard the annotators assigned to it): it can be read from their hidden states, and, in the eight models we tested, removing the direction that carries it weakens the moral verdict by a few percent, and more than the other verdicts in seven of them. We release the pairs, the ratings, the code, the trained SAE dictionaries and the result files. 

---
# Token Commitment as a Control Surface for Steering Masked Diffusion Generation 

**Authors**: Ambroise Aigueperse, Hugo Buurmeijer, Carmen Amo ALonso, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2611.21669)  

**Abstract**: Masked diffusion models repeatedly predict provisional tokens before a sampler decides which predictions become permanent. Existing inference-time steering methods primarily modify activations, logits, or denoising trajectories, while treating this commitment policy as a fixed decoding detail. We instead study token commitment as a diffusion-specific control surface and separate it into two decisions: which tokens to commit and how many tokens to commit at each step. For the first, we introduce counterfactual target-aware commitment, which ranks unresolved positions by a mixture of model confidence and leave-one-token-out target contribution. For the second, we introduce calibrated adaptive budgets driven by target progress and confidence. Across language and vision-language settings in the LLaDA family, multiple targets, two activation-steering methods, and an unsteered control, both mechanisms yield empirically monotonic target-quality trade-offs in every configuration we evaluate. Adaptive schedules also recover competitive operating points relative to scorer-free stochastic schedules matched in average denoising cost. Because both mechanisms require only a scalar score evaluated on a provisional completion, they apply unchanged when the target is an automatic proxy for task success rather than an output attribute: NLI entailment for dialogue summarization and CLIPScore for captioning, neither of which corresponds to a single reusable steering direction. We reproduce the behavior of both mechanisms on a second masked-diffusion family, Dream-7B, for one control target and one task-success target. These results characterize commitment--which provisional predictions become irreversible, and how rapidly--as an interpretable control surface for steering masked diffusion generation under the conditions we test. 

---
# Evaluating Semantic Insensitivity in Financial Sentiment Analysis: Implications for Sustainable Financial Decision-Making 

**Authors**: Gabriela Cioboata, Teodor-Stefan Bildea  

**Link**: [PDF](https://arxiv.org/pdf/2611.21653)  

**Abstract**: Financial sentiment analysis is critical for automated trading, portfolio management, and risk assessment, heavily relying on models like FinBERT. Despite high benchmark accuracy, these models remain vulnerable to subtle adversarial text modifications, threatening the integrity of algorithmic decision-making. In this paper, we propose Agent-driven Semantic Inversion (ASI), a novel adversarial attack framework targeting financial sentiment models through domain-specific semantic manipulation. Utilizing an LLM-powered multi-agent pipeline, our approach identifies eligible financial news sentences and applies targeted lexical substitutions. We empirically evaluate ASI against FinBERT, one of the most widely deployed financial sentiment models. Our results reveal a striking vulnerability: the attack successfully deceived FinBERT in 90.85% of eligible cases, with the model frequently maintaining its original sentiment prediction despite the fundamental inversion of financial performance indicators. Through qualitative analysis and token-level investigation, we demonstrate that FinBERT critically over-relies on contextual keywords and syntactic patterns rather than true logical comprehension of financial semantics. Our findings highlight the urgent need for logic-aware, robust financial NLP models to ensure the reliability of automated decision-making systems. 

---
# Consistency Is Not Correction: Separating Confidence, Abstention, and Recovery in LLM Agents 

**Authors**: Khushal Sethi  

**Link**: [PDF](https://arxiv.org/pdf/2611.21651)  

**Abstract**: A controller that resamples an answer, escalates to a larger model, or abstains is usually justified by a confidence signal. Three claims about such a signal do not follow from one another: it ranks answers (discrimination), it supports withholding some of them (selective prediction), and it changes an answer for the better (correction utility). We give an exact decomposition of the third into gated recoveries minus gated harms, which bounds whether a comparison can succeed before it is run: a gate that only replaces a proposal can act only where the base policy was already wrong, so the initial error count caps the discordant pairs any correct gate can produce. When that cap falls below the count a paired test needs, the experiment cannot return a positive result for a working mechanism, nor distinguish one from a gate that never fired. We measure all three targets on two model scales and find the bound binds. In a small-model suite a perturbation-agreement verifier reaches within-policy AUROC 0.990 (95% interval [0.95, 1.00]) while every episode outcome matches single-sample decoding at 4.32 times the call count. In a controlled 840-episode study no verifier replaced the proposal on any of the 60 GSM8K decisions, so that comparison tested nothing; the entire GSM8K gain across six controllers is one task, and it is the task whose identical temperature-zero repeats disagreed, making it the one answer the service does not reproduce. On the diagnostic set the gate could reach four errors while the test needed six, so even a perfect gate had no significant result available. We report two checks that would have caught this before any budget was spent, and show that tie handling in rankings, tied selection cutoffs, and unequal output limits each move a headline number in the same dataset. The binding constraint on this literature is measurement resolution, not the choice of signal. 

---
# From Preference Data to Personalization: Tracing Sycophancy in Large Language Models 

**Authors**: Himaghna Bhattacharjee, Yasha Sheynin, Shivam Singhal, Arjun Subramonian, Ankur Samanta, Julian Michael, Smitha Milli, Maximilian Nickel, Claudia Shi  

**Link**: [PDF](https://arxiv.org/pdf/2611.21650)  

**Abstract**: Sycophancy in large language models is often measured as agreement with the user, yet agreement can be appropriate when the user's claim is valid, their preference is genuine, or their experience warrants acknowledgment. We study sycophancy as inappropriate agreement: validating a claim, preference, or self-assessment when the context calls for correction, qualification, or resistance. We introduce a validity-aware sycophancy grader and SycoDrift, a persona-targeted evaluation set of 3,298 validity-annotated probes with controlled annotations for user openness and stakes. We use this framework to analyze three sources of agreement pressure: human preference data, automated judge specification, and in-context personalization. First, preference for validating responses in human datasets (Chatbot Arena, HH-RLHF, Community Alignment) is weak and concentrated where agreement is often appropriate. Second, in a selected-value, single-model experiment, emphasized judge rubrics produce consistent but modest increases relative to paired suppressed rubrics, demonstrating a plausible mechanism for training-time amplification. Third, in off-policy evaluations across 14 models, inference-time personalization increases inappropriate agreement, with the largest effect from matched user profiles placed directly before the query. 

---
# Literary Language Mashup: Curating Fictions with Large Language Models 

**Authors**: Manzanarez Gerardo Aleman, Monroy Raul, Garcia Flores Jorge, Calvo Hiram  

**Link**: [PDF](https://arxiv.org/pdf/2611.21649)  

**Abstract**: In this paper, we propose an alternative approach to human evaluation of literary fictions by employing LLMs themselves as evaluators within the GrAImes framework. We apply this methodology to assess human-written and AI-generated microfictions in Spanish, to five PhD professors in literature and sixteen literary enthusiasts, and to short stories in both Spanish and English. By comparing the evaluations performed by LLMs with those of human judges, we examine the degree of alignment and divergence between both perspectives, thereby assessing the feasibility of LLMs as auxiliary literary evaluators. Our analysis focuses on the alignment of responses from LLMs with those of human evaluators, providing insights into the potential of LLMs in literary assessment. The conducted experiments reveal that while LLMs cannot be regarded as substitutes for human judges in the evaluation of literary microfictions and short stories, with a Krippendorff's alpha reliability coefficient less than 0.66, they can serve as a valuable tool that offers an initial perspective on the editorial quality of the texts in question. Overall, this study contributes to the ongoing discourse on the role of artificial intelligence in literature, underlining both its methodological constraints and its potential as a complementary resource for literary evaluation. 

---
# Can an LLM dream? Testing the overfitted brain hypothesis on an agent with text memory 

**Authors**: Eric Zou  

**Link**: [PDF](https://arxiv.org/pdf/2611.21648)  

**Abstract**: Hoel (2021) proposed that dreams distort replayed experience so that the brain does not overfit to recent events. We test this hypothesis on an LLM agent (Claude Opus 5.5) that stores what it learns in a text memory and rewrites that memory each night from the day's conversation transcript. The agent goes through ten simulated days of conversation built from LongMemEval. We corrupt 5, 10, 20 or 40 percent of the transcript's turns before each nightly rewrite in one of two ways. Detail corruption changes one or two details in a turn to plausible wrong ones, or replaces the turn with one by the same speaker from an earlier day. Random-text corruption replaces half of the selected turns with a passage of Moby-Dick and inserts Moby-Dick words into the other half. A size limit on the memory, rising from 19,500 to 117,000 tokens over ten nights, set from human free-recall data, forces it to forget. We learned that corruption did not help the agent apply what it learned in earlier conversations to new tasks in any of 30 runs (ten conditions, three different conversation histories). Replacing details in 40 percent of turns lowered recall from 76 to 60 percent. Random text had no consistent effect on recall, with a paired difference of 0 $\pm$ 2 points at 40 percent, because the model recognized it as corrupted and left it out of memory. Replaced details were stored as fact even when another turn held the correct value, and the agent answered 40 percent of those questions correctly, compared with 75 percent without corruption. These results suggest that a pretrained model can separate corrupted text from real events, so a fair test needs a learner that cannot make this separation as easily, such as a model fine-tuned on the corrupted days or an agent whose days take place in an invented world. 

---
# FieldGuide: A Corpus Map for Token-Efficient Agentic Document Exploration 

**Authors**: Nick Hagar  

**Link**: [PDF](https://arxiv.org/pdf/2611.21647)  

**Abstract**: Generative AI agents can help journalists explore large document collections. But exploratory workflows can require substantial token use and time, especially when the task is broad and the relevant subquestions are not yet known. This paper introduces FieldGuide, an agent-facing corpus map that organizes a document collection into a topic hierarchy and exposes tools for corpus orientation, scoped search, and bounded reading. I evaluate FieldGuide on five journalistic document-exploration tasks using 38 Chicago Police Department public-records documents, comparing it with raw-file access and retrieval-augmented generation. Across the five tasks, FieldGuide used 27% fewer tokens and 41% less elapsed time than raw-file access; relative to RAG, it used 38% fewer tokens and 21% less elapsed time. An accuracy audit identified four minor issues in FieldGuide's responses, comparable to the other conditions. In blind ratings of generated reporting leads, FieldGuide's mean ratings were higher than RAG's and lower than the raw-file condition's. These descriptive results suggest that corpus orientation may reduce runtime token use and elapsed time without severe accuracy failures. They also motivate further study of whether selective access limits broad, cross-document discovery. 

---
# Limite 1B - Violetto: Technical Report 

**Authors**: Mario Prignano, Gabriele Cirillo, Alessio Morosini, Luca Cerovaz, Alessandro Bartolocci, Emanuele Rodolà, Giulio Starace, Francesco Pappone  

**Link**: [PDF](https://arxiv.org/pdf/2611.21646)  

**Abstract**: We present Limite 1B - Violetto, a one-billion-parameter model designed for high-throughput mathematical reasoning. Limite is trained from scratch on 237 billion tokens from a carefully curated, mathematics-centric corpus and supports context lengths of up to 131k tokens. Its architecture and training recipe were developed together to maximize reasoning capability while keeping training and inference costs low. Post-training combines knowledge distillation, targeted supervised fine-tuning, and RLVR on a curated curriculum. Limite achieves 94.01% on AIME 2026 and 74.25% on BeyondAIME, placing it among models many times its size on our evaluation suite. The model is deliberately only lightly instruction-tuned: it is intended primarily as a single-turn mathematical solver and as a fast, parallel component of larger systems, rather than as a general-purpose assistant. We release the model, its base and value checkpoints, and an optimized serving implementation under the Apache 2.0 license. 

---
# MInfoVisQA: A Multilingual Benchmark for Information-Rich Visual Question Answering 

**Authors**: Changjiang Jiang, Jiang Zhou, Xinbin Yuan, Fan Zhang, Tao Feng, Zhuohan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2611.21644)  

**Abstract**: Multimodal large language models (MLLMs) excel in vision-language tasks; yet, whether they maintain consistent understanding of information-rich visuals (InfoVis) when the surface language changes remains unclear. To quantify this capability, we introduce MInfoVisQA, a parallel benchmark with 8,960 QA configurations derived from 128 seed instances across 24 languages. We construct seed-matched, language-parallel visuals and QA pairs by disentangling numerical-structural data from text and programmatically synchronizing translated labels. We evaluate 25 MLLMs under two settings: Language-Aligned QA (LQA, matching visual-text languages) and Cross-Language QA (XQA, fixed QA language with varying visual languages). The results show that aggregate accuracy masks substantial robustness gaps: two models with comparable mean LQA accuracy (80.4\%) exhibit performance spreads of 7.8 and 20.3 percentage points across languages. Moreover, all 25 models score lower with non-English visuals under fixed English QA, underscoring the need for cross-language sensitivity metrics. 

---
# Sleep2Word: Peripheral-to-EEG Translation via Duration-Aware Wordization 

**Authors**: Keondo Park, Arthur Delhaye, Younghoon Na, Hyun-Woo Shin, Hyung-Sin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.21643)  

**Abstract**: Estimating full-night electroencephalography (EEG) spectrograms from peripheral measurements could extend EEG-informed sleep analysis to recordings acquired without EEG. This task requires compact representations of signals with different temporal structures. We introduce Sleep2Word, a framework for translating respiration, electrocardiography (ECG), or photoplethysmography (PPG) into EEG spectrograms through duration-aware physiological words. Building on context-dependent chunking, Sleep2Word groups physiological embeddings into variable-duration chunks and jointly quantizes their content and duration into modality-specific wordbooks. A block-autoregressive translator then predicts target EEG word identities and occurrence-specific duration refinements within fixed-duration blocks. This design accommodates independent source and target wordbooks, limits temporal drift, and shortens the autoregressive decoding path. Trained on over 20K PSG cases across 9 datasets, Sleep2Word consistently outperforms existing spectrogram generation baselines in EEG spectrogram reconstruction, sleep staging, arousal detection, and survival prediction. Compressing full night sequences by up to 69.7%, the learned words yield a favorable accuracy-compute trade-off. Sleep event analysis further shows that the learned words capture physiologically plausible associations across modalities, suggesting their potential for interpretable cross-modal analysis. 

---
# RAW: Refusal Aversion in the Wild, A Causal Measurement Method for Deployed LLMs 

**Authors**: Pavel Kireyev  

**Link**: [PDF](https://arxiv.org/pdf/2611.21641)  

**Abstract**: Deployed LLMs field hundreds of millions of conversations a day and refuse a non-trivial fraction of user requests, yet the user-side cost of refusal, whether users disengage and by how much, is unmeasured on any production model: a blind spot for the consumer-facing platforms built on them. A/B-testing refusal is ethically and operationally constrained on frontier-lab products and unavailable to third-party platforms deploying these models through a fixed API. We introduce RAW, a method that uses LLM stochasticity as a natural experiment: at near-identical prompts in any existing log, the same model sometimes refuses and sometimes does not, and that variation identifies the causal user-side effect of refusal. Applying the method to 15 deployed LLMs in LMSYS-Chat-1M and WildChat, with user-clustered inference, we find that for about half of the models we can measure precisely, a refusal costs 3 to 17 pp of subsequent re-engagement; the rest show no distinguishable effect, among them one deployment whose positive point estimate we trace to persistent users retrying the refused request. Our contribution is the measurement method, which runs on any deployed model from logs alone for about \$0.30 per 1,000 conversations. We release labels, moderation scores, prompt embeddings, and pipeline code. 

---
# CrisisBench: an evaluation suite for crisis counseling AI 

**Authors**: Akshay Swaminathan, Filip Ramazan, Sharang Phadke, Kevina Wang, Ivan Lopez, Shaked Peleg Azzam, Gloria Ye, Chastin Chung, William Wang, Stephanie Stoll, Ivy Pham, Rebecca Hurwitz, Shreya Shah, Samuel Chuang, Divyanjali Verma, Abhay John, Ehsan Adeli, Nigam H. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2611.21622)  

**Abstract**: Background: Crisis helplines are essential to mental health systems but are often understaffed and underfunded. LLMs could support counselors, but existing benchmarks do not capture the escalation, risk assessment, and de-escalation central to crisis conversations, while privacy restrictions limit use of real transcripts.
Methods: We present CrisisBench, a privacy-preserving suite for single- and multi-turn crisis counseling evaluation. It includes 1,085 single-turn tasks from 200 de-identified conversations and 1,185 patient LLMs derived from hotline metadata. Patient models and a 23-item LLM judge were calibrated against expert counselors.
Results: We evaluated five GPT models plus draft-then-edit and retrieval-augmented generation. All systems underperformed human counselors on excessive empathy and questioning in single-turn evaluation. The rubric achieved human-human Krippendorff's $\alpha$ = 0.51 and LLM-human $\alpha$ = 0.54 (Pearson r = 0.80). In multi-turn evaluation, retrieval-augmented generation scored highest overall (0.961), with all systems within 0.058. Microskills and style/tone were near ceiling, while productivity and safety/risk accounted for most between-system variation. Counselors rated patient LLMs realistic (86%) and useful (92%).
Conclusion: CrisisBench enables scalable, privacy-preserving evaluation of crisis counseling. Current systems perform strongly on core structural competencies but still diverge from experts on important response-level failure modes. We release the rubrics, judge prompts, and evaluation tasks. 

---
# Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature Matching 

**Authors**: Giannis Daras, Marios Papachristou, Adam Klivans, Antonio Torralba, Constantinos Daskalakis  

**Link**: [PDF](https://arxiv.org/pdf/2611.23229)  

**Abstract**: Collecting large-scale, diverse, and high-quality post-training datasets for autoregressive language models is challenging. Consequently, Supervised Fine-Tuning (SFT) is typically applied to datasets significantly smaller than those used for pre-training, which leads to overfitting and the catastrophic forgetting of the tails of the data distribution. To prevent this collapse, we introduce Ambient SFT, a principled framework for leveraging imperfect predictions to augment the fine-tuning set. In our method, a "frozen" version of the pre-trained model serves as a noisy teacher that the fine-tuned model must match at a coarse level. In particular, we require that the high-temperature versions of the conditional distributions produced by the post-training model should match the high-temperature versions of the pre-training ones. The temperature hyperparameter controls the level of trust in the pre-trained model, with the limit $T\to \infty$ reverting back to regular SFT and $T=1$ to knowledge distillation. We demonstrate that our approach outperforms standard SFT across domain adaptation, instruction-tuning, and reasoning, leading for up to $21\%$ average relative improvement to standard SFT. 

---
# TextSphere: Efficient Text-to-Image Generation via Shared Hyperspherical Alignment and Uniformity 

**Authors**: Yixia Zhang, Girish A. Koushik, Xilu Wang, Wenwu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.23155)  

**Abstract**: Recent advances in text-to-image generation have substantially improved image quality, particularly through diffusion-based methods that require multiple inference steps. In contrast, Sphere Encoder offers an efficient generative framework capable of producing images with fewer forward steps. However, extending it to free-form text is nontrivial, because its class-conditioned design lacks cross-modal alignment. To address this challenge, we propose TextSphere, an efficient text-to-image generative framework that extends Sphere Encoder with language conditioning and cross-modal alignment. We incorporate pretrained CLIP text embeddings through adaptive layer normalization (AdaLN) to guide image generation while retaining the original spherical sampling mechanism. To bridge the cross-modal representation gap, we introduce a Spherical Semantic Alignment and Uniformity (SSAU) objective that jointly promotes semantic alignment and representation uniformity on the hypersphere. Experiments on CUB-200-2011, MS-COCO-2014, and Flickr30k demonstrate that TextSphere consistently reduces generation FID (gFID) over the text-conditioned baseline while achieving reconstruction FID (rFID) below 1.0. With only 4 and 6 sampling steps, TextSphere achieves lower gFID than the compared diffusion- and GAN-based models trained from scratch on CUB-200-2011 and MS-COCO-2014, respectively. We further observe that improved reconstruction fidelity does not necessarily yield better generation quality. In addition, we use Partial Least Squares Singular Value Decomposition (PLS-SVD) to analyze cross-modal alignment and modality gap, offering insight into the effectiveness of our approach. 

---
# Mind the Gap: A Deployed Multi-Agent System for Documentation Maintenance 

**Authors**: Lorenzo Masoero  

**Link**: [PDF](https://arxiv.org/pdf/2611.23142)  

**Abstract**: The "AI revolution" has enabled teams across the technology industry to deliver software features at unprecedented velocity. While code evolves faster than ever, customer-facing documentation often struggles to stay up to speed. As users increasingly rely on LLM-based agents that consume documentation as their primary interface, the impact of this documentation gap worsens: stale information propagates and is treated as authoritative. Signals for identifying where documentation has fallen behind already exist in abundance, both explicit (e.g., resolved support tickets revealing what documentation lacks) and implicit (e.g., code-documentation discrepancies detectable from automated loops). We deploy a multi-agent system, built around a triage agent we call "TriAgent," that continuously extracts these signals and repairs documentation drift. Over 43 days of shadow-mode operation, our system triaged 499 tickets and surfaced 92 gaps in the documentation. Using a panel of LLM judges from independent model families, 81.1% of TriAgent's verifiable assertions were correct (245 of the 302 assertions the human resolution addresses; majority of three judges); on length-comparable pairs the panel preferred TriAgent's response in 70.5% of pairs and the human engineer's in 23.2% (the rest had no judge majority). Importantly, agent- and human-sourced replies tended to be complementary, with the human often surfacing insights that were either buried within the existing documentation or missing from it entirely; our system turns each such insight into a documentation gap and, after human review, a fix. Repairing a gap measurably improved the agent's later answer: when the agent re-answered the same ticket after the repair, blinded judges preferred the post-repair answer over its own pre-repair answer 71.4% of the time (n=35). A blinded human audit confirms strong alignment on self-improvement. 

---
# The Illusion of Grounding: Do Vision-Language Models Weigh Retrieved Text and Images Like an Ideal Observer? 

**Authors**: Basavaraj R M, Harshal Chandel  

**Link**: [PDF](https://arxiv.org/pdf/2611.23127)  

**Abstract**: Retrieval-augmented vision-language models (VLMs) answer from two sources of evidence that can each fail: the retrieved text can be wrong or noisy, and the image can be degraded or contradicted. We introduce GIB-200K, 200,000 image-question items built from COCO and Visual Genome, each expanded into controlled stress conditions: hard-negative retrieval noise (1-20 documents), visual ambiguity (Gaussian and motion blur, low-resource crops) and cross-modal conflict with counterfactual flip pairs. Instead of only measuring degradation, we ask how models combine the two modalities. From ideal Bayesian cue combination we derive a testable prediction that yields formal definitions of Knowledge Inertia (text over-weighted relative to the model's own image evidence) and Sensory Capture (the reverse), and we estimate the weights with a measurement-error-corrected regression whose recovery of known weights we verify. A counterfactual flip decomposition separates modality bias from concept priors, and a Grounding Illusion Score measures how much clean accuracy is not grounded in the image. We evaluate 4 open models on one frozen, seeded evaluation split of 2,000 GIB-200K items. All models are accurate on clean inputs (92.5%-94.2%) but adverse accuracy falls to 41.3%-59.9%. The two LLaVA models over-weight the text (Knowledge Inertia, w_T/w_I = 1.229 and 1.322), whereas InternVL3 favours the image (w_T/w_I = 0.680); between 34.8% and 71.4% of the clean answers that are correct are abandoned when the document lies. A training-free consistency check that compares the answer with and without retrieval trades coverage for accuracy under misleading documents, while falling back to the image fails when the image is heavily degraded. All per-instance outputs, code, and an audit that recomputes every reported open-model number are released. 

---
# Nunchi: Ears Help Eyes in Multi-Party Conversations 

**Authors**: JunYeong Park, Shijing Wang, Seonwook Park, Michael Stengel, Shalini De Mello, Sang-Il Choi, Hyung Jin Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.23115)  

**Abstract**: Identifying whom a person looks at is central to understanding multi-party conversations, yet gaze geometry alone becomes ambiguous as more people appear. Conversations offer cues that geometry lacks, such as who is speaking and what is said, but no gaze dataset provides them for many multi-party scenes. We introduce SORBET, a benchmark of 3,340 samples from 1,578 clips with synchronized speech, transcripts, and attended person labels, covering more multi-party scenes than existing gaze datasets. Prior gaze models struggle to use these cues, which come in different forms, must be linked to the right person, and can conflict with gaze geometry. We propose Nunchi, a framework that constructs a text prompt from the transcript and the outputs of gaze and speaker models, links every cue to the candidates with shared labels, and lets a vision-language model (VLM) weigh gaze geometry against conversational cues to identify the attended person. Nunchi outperforms the strongest existing method by 4.8 percentage points. On SORBET, conversational cues help whereas additional visual cues do not, and the gains hold across three VLMs, with automatic transcripts, and on two existing datasets. 

---
# Large-scale Neuron Community Discovery Reveals Neuron-Sparse Behaviors in Language Models 

**Authors**: Igor Shilov, Thomas Fel, Tal Haklay, Lucius Bushnaq, Owen Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2611.23112)  

**Abstract**: What reusable units of computation can we discover within a language model's neurons? Prior work has shown that individual forward passes can often be sparsified post-hoc to small groups of neurons, but it has been unclear whether these groups form coherent computational entities reused across inputs. We introduce neuron community discovery, an unsupervised method for finding neuron groups that recur across prompts and are causally necessary and sufficient, within the layers they span, for common computations. The method combines gradient-based attribution, a graph connecting neurons that are repeatedly attributed together, and community detection. Applied to Llama-3.1-8B using ten million prompts, it discovers thousands of candidate communities. We evaluate them on held-out prompts through three causal tests: community necessity, sufficiency within a layer subset, and layer necessity. These tests identify groups whose removal disrupts predictions and whose retention preserves predictions despite ablating the remaining MLP neurons in the tested layers. Discovered communities support functions like grammatical agreement, token-probability calibration, fixed phrases, in-context copying, addition, and multiplication. With communities capped at 128 neurons across all layers, 10.3% of held-out prompts have at least one layer's contribution explained on a neuron-sparse basis. Coverage remains 8.7% when requiring at least four layers and 3.3% when requiring at least twelve. These results show that small groups of existing neurons can provide reusable computational components across prompts, and that such groups can be discovered at scale. 

---
# The Resolution Hypothesis: Agent Evolution's Key Is Identifying Improvements 

**Authors**: Yuanhao Zeng, Shaocheng Lan, Ao Lu, Kuang Yan, Heng Lin, Xinang Xu, Yuan Luo, Yueting Wang, Yexin Li, Ruo Shan, Kan Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.23068)  

**Abstract**: Agent evolution follows a simple loop: a proposer writes candidate artifacts (prompts, skills, memories) and selection accepts those that score higher. This loop has shown promise, yet practitioners face a puzzle: methods trade wins across papers, and evolution often fails to bootstrap on certain tasks. We trace this puzzle to the Resolution Hypothesis: agent evolution stalls not because methods fail to propose better artifacts, but because noisy evaluations cannot reliably identify them. Across seven evolution methods and six models (over 10M LLM calls), we find support for the hypothesis at both levels where evaluation scores are used. At the candidate level (choosing artifacts), helpful candidate artifacts exist, but selection is no better than random at telling them from harmful ones, so errors accumulate across rounds. At the method level (ranking methods), typical margins of a few percentage points are smaller than the difference between two independent runs of the same method, so published rankings often rest on luck. Together, these findings indicate that agent evolution is often bottlenecked by reliably identifying improvements. This calls for a shift in how we develop and evaluate self-improving agents, from designing ever more candidate improvement systems to making true improvements detectable and accumulable. 

---
# ForecastGym: A Live, Controlled, and Replayable Benchmark for Forecasting Agents 

**Authors**: Yicheng Tao, Yuante Li, Zipei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2611.23045)  

**Abstract**: Agents built on large language models (LLMs) increasingly forecast real-world events by searching the Web and reasoning over what they find. How well such an agent forecasts depends on both the retrieved evidence and the model, making control over that evidence essential for evaluation. Live evaluations can mix retrieval with reasoning when each system searches on its own, while retrospective search can leak later information through date filters or today's ranking of historical pages. Comparing models, search sources, and memory strategies requires both fixed evidence and a record of how it evolved before resolution. We introduce ForecastGym, which records what four commercial web search APIs return for open prediction-market questions every two days and replays this frozen evidence offline once outcomes are known, comparing models, memory, and captured search sources under a fixed evidence-acquisition policy. Its first release covers 150 questions, 614 evidence states, one per question and collection time, and 131k search results. Rerunning the same queries after settlement returns results that all but state the outcome for up to 92% of the questions, despite the tested date filters. On the frozen evidence, all tested LLMs trail the market in forecasting accuracy. Our diagnostic analysis suggests that missed public information contributes more to the gap in high-gap states than misreading does. Switching the search API changes a model's Brier score by up to 0.043, more than the spread across models, while persistent notes score similarly to rereading all evidence at half the input tokens. The evidence a forecaster sees can matter as much as the model that reads it. 

---
# Storing the Gemma 4 K=V Cache Once: Exact Global-KV Memory Savings in a Paged Serving Engine with Unchanged Attention Kernels 

**Authors**: Kushagra Bainsla  

**Link**: [PDF](https://arxiv.org/pdf/2611.23035)  

**Abstract**: In the global attention layers of Gemma 4 31B, the value is the normalized key projection, and the key is the value scaled by a fixed weight, with rotary position encoding on one quarter of its dimensions. The Gemma 4 report states that this design reduces the global KV cache by up to 37.5%, but the open serving engines that we checked still store a full key and a full value: 1,024 values per KV head and token. We store one 640-value row instead, exact in real arithmetic, and an unchanged paged attention kernel reads it through two overlapping strided views as its key cache and its value cache. In a vLLM prototype with the 16-bit 31B model, the outputs are bit-identical to an emulation of the same arithmetic on the stock cache, 2 of 432 final answers differ from the stock engine (within its run-to-run noise), and the steady decode KV memory per request falls by a factor of 1.34 to 1.40 at 32K to 64K tokens. On a prefill/decode pair of two A100 GPUs with NIXL transfer and CPU offload, a decode engine with a full 15 GiB pool ran 1.38 times the stock decode tokens per step for 16K-token prompts (repeated to 0.06%) and 1.24 times for 8K-token prompts; a model of the vLLM block pool predicted both ratios to within 1% (the engine was 0.7% to 0.9% higher) and matched the change between them. The eager decode throughput rose by 1.30 and 1.11 times, and on this path the compact engine gave the same output tokens as the emulation for 8 prompts at batch 1. The run-time prototype makes an eager step 4% to 9% slower at equal batch, and the method does not yet stack with an 8-bit KV cache at the default scale. 

---
# When the tail is earned by its companions: measuring label dependence in long-tailed multi-label classification 

**Authors**: Christoffer Löffler  

**Link**: [PDF](https://arxiv.org/pdf/2611.22859)  

**Abstract**: A rare label seldom occurs alone, and a classifier can recover it from the frequent labels it co-occurs with, its companions, without having learned the label itself. Existing diagnostics of this need a trained model: none says before training whether a corpus admits the failure, or what the standard aggregate hides. Macro-F1, like any measure built from per-label confusion counts, is unchanged by how a label's recall divides between occurrences with and without a companion. We introduce a within-label estimand of that division, a reading rule under which a fall in dependence counts only if companion-free recall rises, and a statistic of training labels that bounds, before training, how far reported recall can exceed companion-free recall. Across 23 corpora, 7 training objectives and five families of label-correlation mechanisms, companion-free recall falls short on all six corpora where the estimand is defined. Up to 18.4% of the tail true positives macro-F1 credits would vanish at the companion-free rate, within the training-label bound on every corpus. No long-tail objective reduces the dependence, but architecture moves it. Models whose tail score is computed from other labels' predictions are strongly dependent. 

---
# A Full-Ledger Counterfactual Audit of Adaptive Reasoning 

**Authors**: Xiaopeng Chu, Wenzheng Kong, Wei Liu, Wenyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22778)  

**Abstract**: Adaptive reasoning must decide whether to answer now or spend more compute. Predicting that the current answer is wrong does not establish whether further reasoning will improve it. We introduce the Full-Ledger Counterfactual Value Audit (FLCVA), which compares an immediate answer with four fixed-length continuations from the same frozen prefix under one answer procedure and accounts for the cost of every signal used by a policy. On GSM8K and MATH-500, a random 256-token continuation recovers a wrong answer in 26.6% of Qwen3-8B cases and 18.9% of DeepSeek-R1-Distill-Qwen-7B cases, while damaging a correct answer in 1.4% and 1.9%. Correctness labels from independently sampled continuations predict a separate continuation's outcome in both primary models. In offline replay, using these labels to allocate compute improves performance on DeepSeek; the Qwen3 gain remains inconclusive. Under full cost accounting, the tested lightweight, myopic controllers mainly recognize current errors and add little continuation-specific value beyond that shortcut. A Llama-8B replication and action/cost ablations examine the robustness of these findings. FLCVA separates measuring local continuation value, predicting it from observable signals, and using those predictions to allocate computation. 

---
# CrossEnv: Synthesizing Verifiable Cross-Environment Tasks for Long-Horizon Agent Training 

**Authors**: Zhengfeng Yu, Tian Lan, Zhijing Wu, Enbo Zhao, Xuannan Liu, Feihu Jiang, Qianghuai Jia, Longyue Wang, Zhao Xu, Weihua Luo, Xian-Ling Mao  

**Link**: [PDF](https://arxiv.org/pdf/2611.22777)  

**Abstract**: Executable synthetic environments provide scalable training tasks and verifiable feedback for long-horizon agents. Existing environment synthesis methods primarily generate tasks within individual environments. However, these tasks do not adequately capture real-world interactions across services, where actions in one service depend on information obtained from another. Synthesizing such tasks is challenging because data generated across environments must remain consistent to support executable and verifiable interactions. To address this challenge, we introduce CrossEnv, a framework that uses an entity-tool graph to identify connections between environments and aligns task-specific data to synthesize executable, verifiable cross-environment tasks. We further propose CrossEnv-Curriculum, which progresses from single-environment tool use to coordination across more environments and business domains, before introducing user interaction. Across four external benchmarks, combining cross-environment training data with this curriculum achieves competitive performance with fewer training examples. Further analyses support the benefits of cross-environment data and curriculum training, and provide evidence of transfer to new environment combinations, external tasks less similar to the training data, and longer workflows. 

---
# The Between-Round Decision in Autonomous Research: An LLM Controller That Reads the Whole Trajectory 

**Authors**: Caoxin Wang, Shengkai Gan, Xiang Hu, Lili Pan  

**Link**: [PDF](https://arxiv.org/pdf/2611.22661)  

**Abstract**: Executing an idea often changes how promising it looks, so the next step in autonomous research should rest on results, not on proposals. After each round of experiments, the system must decide whether to refine the best solution or start a new direction, which solution to build from, and what goal to set. We call this the between-round decision. Existing systems wrap it in machinery such as rules, scores, search trees, and schedules; where an LLM makes it, what the LLM needs to read is left open. We ask whether an LLM can make this decision from accumulated evidence, without a search structure around it, and how much of that evidence it needs to read. We present BRED (Between-Round Evidence-based Decision), an LLM-based controller that maintains the best verified solution so far (the Champion) and a log of past rounds, and makes one decision per round. On three widely used research-agent benchmark tasks, BRED improves on the task baselines by 4.80% in MAE on ETTh1 (to 0.4172) and by 3.44 and 3.43 points of accuracy on SST-2 (to 95.07%) and CIFAR-100 (to 84.63%), ahead of the published results. When the controller does not read the accumulated evidence, the same loop gains less: with no history it reaches 0.4299 on ETTh1 and 83.71% on CIFAR-100, whereas the full controller's MAE is 2.95% lower and its accuracy 0.92 points higher. On a new task, speech-based depression detection under a leakage-free protocol, it reaches a macro-F1 of 0.6164 on seeds held out from selection, 5.5 points above its baseline and 4.0 above the strongest published system. Code is available at this https URL. 

---
# Evolution Under Fire: Evolving Agent Safety Harnesses with Adaptive Injection Stressors 

**Authors**: Yujun Wang, Yilun Liu, Jinhe Bi, Krishna Kanth Nakka, Xuebing Zhou, Hinrich Sch{ü}tze, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.22583)  

**Abstract**: Tool-using language-model agents must act on information from untrusted emails, webpages, and documents. This creates a central security challenge: malicious instructions and legitimate task directives can arrive through the same interaction channel. Existing defenses are largely hand-designed and static, making further improvements dependent on repeated manual redesign. We introduce Evolution Under Fire, a staged framework that couples safety-harness evolution with adaptive stressor evolution. Utility-constrained, Pareto-scheduled tree search evolves complete executable safety harnesses from execution evidence while keeping the task model, benchmark, tools, and evaluator fixed. Adaptive stressor evolution then discovers utility-preserving attacks that guide a subsequent stage of harness evolution. On AgentDyn with GPT-4o-mini, an evolved harness reduces attack success from 50.4% to 3.0%, improves clean and attacked utility over the initial harness, and offers a stronger security-utility trade-off than reproduced defenses. On Agent-SafetyBench with Qwen2.5-7B-Instruct, harness evolution reduces attack success from 82.6% to 62.0% while retaining helpfulness and runtime validity. Evolved harnesses also transfer without further evolution to Qwen3.5-35B-A3B, reducing attack success from 32.1% to 0.9% while improving attacked-task utility. 

---
# From Perception to Simulation: Unveiling and Empowering Mental Spatial Dynamics in MLLMs 

**Authors**: Jiachun Li, Zhuoran Jin, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.22541)  

**Abstract**: Spatial reasoning is a fundamental component of world understanding for Multimodal Large Language Models (MLLMs). While existing research has advanced the perception of spatial relations in static scenes, the domain of dynamic spatial simulation remains relatively unexplored. To bridge this gap, we introduce MentalGrid, a benchmark focused on spatial reasoning tasks that require models to simulate evolving spatial states without intermediate visual feedback. Extensive evaluation on our benchmark reveals a significant deficiency in the spatial simulation capabilities of existing models. Specifically, we find that performance is constrained by perceptual limitations in simple simulation scenarios and hindered by inadequate multi-step reasoning as complexity increases. To address these challenges, we propose a three-stage curriculum learning framework utilizing 16k specialized spatial reasoning samples. This approach yields SpatialWalker, which achieves a 33.8% average improvement over its base model, surpassing proprietary models like Gemini-3-Flash in spatial simulation. Furthermore, SpatialWalker demonstrates robust generalization, maintaining a 4.9% average gain on out-of-domain benchmarks, demonstrating the pivotal role of spatial simulation as a fundamental building block for broader spatial intelligence. 

---
# When Agent Skills Go Stale: Runtime Behavior and Selective Maintenance in Evolving Environments 

**Authors**: Ömer Faruk Akgül, Shuli Jiang, Yuting Zhang, Yi Zhang, Xiaohan Fei, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2611.22525)  

**Abstract**: Agent skills use past experience and current observations to guide decisions and actions on related tasks. This experience is retained in skill state, whose reuse can amortize inference-time reasoning and environment interaction. When environment changes, parts of this state can become outdated while others remain useful. Across four task settings and two models, we compare skill reuse before and after environment changes. Previously beneficial skills can reduce task performance by over 10 percentage points or increase execution cost by nearly 40% relative to no skill. Task success can also remain unchanged while execution savings decline. These effects depend on which environment-specific details become outdated and the agent's ability to compensate using current observations. We introduce selective maintenance, which replays checks from successful executions without additional model calls. It verifies whether observed differences conflict with skill state, repairs affected content, and validates updates, rebuilding when local repair is insufficient. No description of the change is supplied. Relative to reusing stale state, maintenance improves task accuracy by up to 67pp and reduces execution steps by up to 37%. These results suggest that preserving a skill's inference time value as environments evolve requires selectively maintaining its state. 

---
# LMLineage: Content Lineage and Context Governance for LLM Agents 

**Authors**: Wei Hou, Jingxin Geng, Xin Chen, Zheqing Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22511)  

**Abstract**: Interpretability for AI agents should not be confined to understanding the model's internal weights. It must also cover the agent's reasoning and decision-making process over multi-source context. We present LMLineage, an out-of-band system that attributes every output sentence of an agent session to its supporting input spans with calibrated confidences, without requiring white-box access to the model or any additional inference passes. On top of the lineage metadata, we build concept-drift micro compaction: instead of waiting for the context-length limit and summarizing everything, the system detects when previously used input documents have stopped being referenced ("drifted") and surgically compacts only those, preserving user instructions verbatim. We evaluate on a multi-tier corpus: 71 gold annotated sentences from 10 Chinese web-QA cases, plus 90 turns of independently collected multi-turn sessions across three generator models (gpt-4o-mini, qwen3.8-27b, deepseek-v4-flash). Results show that (i) window-level embedding similarity is far better aligned with LLM paraphrasing than whole document or lexical matching; (ii) the greedy-window fast path is both faster and more accurate on average than exhaustive line-matching (a multiple-comparisons bias in the latter inflates web recall at the cost of overall accuracy); (iii) per-turn attribution completes in 1s on a single GPU, remaining stable as context accumulates to 40 turns; and (iv) concept-drift detection is conservative (0% false compaction) with recall bounded by attribution accuracy. We further describe the concept-drift micro compaction design and an interactive incident-tracing/memory-cleaning case study. A companion patent application covers the lineage construction method. 

---
# Surfacing Disagreement Beats Resolving It: A Gated Ablation Study of Provenance-Weighted Memory for Multi-Agent Debate 

**Authors**: Ziaullah Khan, Abdul Rehman Khalid, Shrestha Nischal Lal, Nheng Vanchhay, Md Ariful Islam Mozumder, Hee-Cheol Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.22461)  

**Abstract**: Shared memory for multi-agent debate is usually built to resolve inconsistency: retrieval returns one winning fact and the disagreement disappears before any agent sees it. We ask whether that is correct. In DebateGraph, four backends share one contract (read/write/snapshot), so five mechanisms from the agent-memory literature can be removed independently under a fixed budget, each gated on a pre-declared threshold. Across 39 runs one survives: preserving contradictions and passing the ranked conflict set to the debate, rather than collapsing it to a winner, improves F1 by +0.340 (95% CI [+0.267, +0.414]) and cuts attack success from 22 to 7 of 122 poisoned items (McNemar exact p=0.0003), at equal cost. Reaching it meant fixing a bottleneck in our harness: conflict sets are built only over retrieved edges, and our pre-declared cap left three quarters of planted contradictions unretrievable, first measuring the mechanism at +0.084. The effect replicates on two retrieval substrates and directionally on HotpotQA documents. No other mechanism survives, and two are worse than inert: once contradictions reach the prompt, a write trust-gate and the router's provenance features each cost accuracy (-0.075 and -0.096 F1). The gate fails structurally: poisoned objects are always entities the graph already contains, against 53% of gold, so groundedness scores the lie above a truthful first mention. We release the harness, attack set, and logs, and argue that silently reconciling memory is the assumption worth abandoning. 

---
# SignAligner: Dual-Granularity Span Generation for Sign Language Alignment 

**Authors**: Jongbhin Woo, Youngjoon Jang, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2611.22409)  

**Abstract**: Large sign-language corpora benefit from temporal labels at two scales: subtitle spans for subtitle alignment and word spans for sign spotting. Sign spottings already locate words in time at corpus scale, but training subtitle alignment jointly on subtitle and word spans remains underexplored. We present SignAligner, a prompt-conditioned visual-language span generator that maps a sign-language video and a subtitle-sentence or target-word query to a textual start/end frame interval. It is first trained jointly on subtitle spans and word spans from sign spottings, then refined with overlap-based rewards while retaining auxiliary word examples. On the subtitle test set of the British Broadcasting Corporation-Oxford British Sign Language (BOBSL) corpus, the refined model reaches 66.68 F1@0.50 (F1 with a temporal intersection-over-union threshold of one half), the best among compared systems without external sign-video resources. This overlap-reward refinement outperforms cross-entropy continuation from the same jointly trained model. Joint training with word spans improves subtitle alignment in a pair of our generators matched in code, labels and maximum update budget. The advantage remains after both models of the pair are refined with overlap rewards. For word queries, the refined model follows the queried word, not a fixed position in the video: when the word's sign lies away from the video centre, it is far more accurate than predictions that use position alone. Together, these findings support one shared span generator refined with overlap rewards, aligning subtitles and producing word spans for human review. 

---
# NONSENSE: Non-Semantic Strings Expose Semanticity Gaps in Dense Retrieval 

**Authors**: Ankit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.22353)  

**Abstract**: Dense retrieval is central to modern search, question answering, and retrieval-augmented generation, where high similarity is often treated as evidence of semantic relevance. We examine how far this assumption can be stressed. NONSENSE is a black-box sparse-edit probe that constructs visibly non-semantic strings while forbidding query-term copying, optimizing only the dense similarity between a query and the string. Across five BEIR-style datasets and three popular single-vector retrievers, these strings frequently outscore the retriever's own best judged query-relevant passage (QRP), and often rank first in the full corpus. The result indicates a semanticity gap in the evaluated dense retrievers: nearest-neighbor status can be separated from passage-like meaning. These findings highlight the importance of evaluating dense retrievers not only by whether they retrieve relevant documents, but also by how reliably their similarity scores reflect semantic evidence. 

---
# VRLLM: Turning LLMs into Powerful Tabular Classifiers 

**Authors**: Yingtong Dou, Shubham Jain, Junpeng Wang, Hadi Abdullah, Vineeth Rakesh Mohan, Yuzhong Chen, Liang Wang, Yiwei Cai  

**Link**: [PDF](https://arxiv.org/pdf/2611.22308)  

**Abstract**: Large Language Models (LLMs) have shown strong performance on text, but whether pretrained LLMs are effective for large-scale tabular classification remains unclear, as many previous works demonstrate that LLMs' tabular data classification performance falls behind tree-based models or neural architectures designed for tabular data. At Visa Research, we managed to fine-tune a pretrained Llama 3.2 1B LLM on billions of rows of labeled tabular data and make it outperform a wide array of tabular classifiers and even our competitive production deep learning model. For ease of reference, we refer to the model as the Visa Research LLM (VRLLM). In this paper, we share how we achieved this exciting result, covering data preparation, training recipe, ablation studies of design choices and deployment validation. We also benchmark frontier LLMs on the same task and analyze their prediction behavior across model families and generations. Our interpretability experiments suggest that VRLLM learns feature-specific predictive associations, while component ablations indicate that the earliest two FFN layers contribute to the emergence of linearly decodable task information. Our paper provides an engineering recipe and a focused empirical case study on adapting compact LLMs to industrial tabular prediction tasks. 

---
# Teaching Language Models to Narrow Boolean Patent Searches from Search Feedback 

**Authors**: Seiya Kawano, Koji Marusaki, Hirofumi Nonaka, Koichiro Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2611.22270)  

**Abstract**: Patent prior-art searchers issue Boolean queries line by line, broadening to find relevant documents and narrowing until candidates can be screened. We distill 38,397 Japan Patent Office search reports into executable expert strategies, re-execute them, and evaluate generated strategies by citation recall and the smallest set containing each citation. Prompted models write valid, high-recall strategies but leave citations in sets too large to screen. We adapt a generator offline from scored rollouts in two forms: strategy-level learning from a screening-aware strategy score, and line-level learning from citation-preserving narrowing credit, which upweights the query lines keeping a found citation in a smaller set. On shared rollouts, line-level credit outperforms strategy-level scores on screening measures; iterated, it cuts the median citation-containing set from 768 to under 50 and places more citations in sets under 100 documents than experts, at lower recall. Strategy-level preference learning reaches near-complete recall; the two forms are complementary. 

---
# SOTA or Luck? The Winner's Curse in LLM Leaderboards 

**Authors**: Sikun Xu, Raphael Thomadsen, Dennis J. Zhang, Heng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22227)  

**Abstract**: Public LLM leaderboards guide model evaluation and adoption, yet the reported performance of top-ranked models can be systematically overstated. This leaderboard winner's curse arises because ranking noisy benchmark scores rewards sampling luck alongside genuine ability. The same selection effect can distort scores throughout the leaderboard, a phenomenon we call the rank-induced bias. We develop an $m$-out-of-$N$ bootstrap to correct for this bias. It assumes a fixed model pool with independent, identically distributed bounded tasks while allowing arbitrary within-task correlation among models. Under exact population tiers separated by positive gaps, we prove that the bootstrap reduces $\Theta(N^{-1/2})$ ranking bias to expected residual bias $o(N^{-1/2})$. Across 14 benchmarks from 5 benchmark families, the rank-1 adjustments range from approximately 0.09 to 3.86 percentage points. In our SWE-bench Verified snapshot, the tied leaders receive an adjustment of 0.90 percentage points. Such overstatement is large enough to make the apparent winner statistically fragile. Thus, we use the same bootstrap to further quantify the intervals of plausible rank positions for each model and group models into tiers. We establish sufficient conditions for consistent recovery of population tiers. On SWE-bench Verified, this procedure places 13 of 173 models in the leading tier. We combine the score adjustment and rank intervals in the Rank Uncertainty and Leaderboard Evaluation Report (RULER). We validate its finite-sample performance using sample splitting and calibrated simulations, and advocate publishing RULER alongside raw rankings so readers can assess both reported performance and the precision of the ordering. RULER also provides guidance for benchmark design that accounts for selection effects and ranking uncertainty. 

---
# Fine-Grained Japanese Clinical Named Entity Recognition: A Controlled Comparison of BERT and Open-Weight Large Language Models 

**Authors**: Feng Han, Ayako Yagahara, Kenji Hirata, Atsushi Yoshida, Kohsuke Kudo  

**Link**: [PDF](https://arxiv.org/pdf/2611.22219)  

**Abstract**: Background: Generative large language models (LLMs) offer an alternative to encoder-based clinical named entity recognition (NER), but their performance on fine-grained Japanese extraction tasks and sensitivity to generation settings remain uncertain. Methods: We compared two fine-tuned BERT-CRF baselines with nine open-weight LLMs on iCorpus, which defines 113 entity categories. The evaluation used 433 sentences from 36 test documents; 1,672 sentences were used for training. LLMs were evaluated using zero-shot prompting with and without JSON Schema constraints and three-seed QLoRA fine-tuning. Fine-tuned models were assessed under different output-token limits, including 900 and larger model-specific limits of 2,600 or 4,096 tokens. Performance was measured using sentence-level label-surface matching with relaxed and exact-surface F1. Results: The highest zero-shot relaxed F1 was 0.337, compared with 0.836-0.841 for the BERT baselines. At the larger output limits, all eight fine-tuned LLMs with at least 13B total parameters exceeded both BERT baselines (relaxed F1, 0.882-0.914; exact-surface F1, 0.840-0.880). Across three-seed means, increasing the output limit improved relaxed F1 by 0.052-0.085, alongside higher recall and fewer length-limited outputs. JSON Schema constraints changed relaxed F1 by -0.014 to +0.008 among seven models outside the two reasoning-oriented configurations, while medical-general model pairs showed no consistent advantage for medical specialization. Conclusions: On this historical Japanese case-report benchmark, task-specific fine-tuning combined with adequate generation capacity was associated with stronger LLM extraction performance. Output-token limits materially affected comparisons with BERT, whereas enforcing output structure alone did not overcome low zero-shot accuracy. 

---
# Is 4-Bit Quantization More Than Noise to a Chat Model's Safety Behaviour? Attributing Behavioural Shifts with Matched Perturbations 

**Authors**: Ege Kaya  

**Link**: [PDF](https://arxiv.org/pdf/2611.22186)  

**Abstract**: Chat models on phones and laptops are usually 4-bit copies of the model that was evaluated, and audits of such copies disagree about what quantization does to refusal. A 4-bit build is one of many weight perturbations of the same size, so a change in its behaviour could come from its specific rounding error or from any perturbation that large. In a pre-registered study of three 3-4B chat models, with paired tests, blind human labels and an 8-bit null, we find two 4-bit shifts of opposite sign: with a compliant reply prefilled, Qwen2.5-3B continues into harmful content more often, and on held-out jailbreaks Llama-3.2-3B complies 10.3 points less often. Matched perturbations separate the two. Shuffled and Gaussian versions of the error average close to zero, while scaling the real error, from its mirror image to twice its size, changes behaviour near-linearly. At half the real error, Llama's likelihood is almost unchanged, yet it already complies 5.8 points less often, and this http URL's Q4_K_M, at the same bits per weight, moves neither model. Refusal under 4-bit quantization depends on the specific rounding error, not on bits per weight or likelihood, so each build has to be compared with its original. 

---
# What RL on Expert-Built Tasks Teaches Nemotron 3.5 Super 

**Authors**: Sushant Mehta, Edwin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.22161)  

**Abstract**: We post-train NVIDIA's Nemotron 3.5 Super (120B total, 12B active parameters) with reinforcement learning (RL) on five expert-built task datasets, one at a time: real-world software engineering, complex constraints, long-horizon tool use, professional chart understanding and enterprise document reasoning. Each run trains a fresh rank-32 LoRA adapter with CISPO on Tinker, with no supervised stage, and each adapter is compared with the base model on the same tasks, harness and settings. On twelve headline benchmarks, eight built by other groups and four by Surge AI, the gains range from +4.5 points on IFBench to +13.8 on GDP_pdf, which rises from 10.4% to 24.2%; SWE-bench Pro V2 rises from 86.9% to 92.1%, Terminal-Bench 2.1 from 63.9% to 70.3%, MultiChallenge from 58.3% to 66.6%, AutomationBench from 14.8% to 27.5% and Chartography from 39.4% to 50.6%. The document adapter, trained without API tools, also gains +8.0 points on AutomationBench. In attempts read side by side with the base model's on tasks each adapter newly solves, the base model's failures recur in a few modes, and each adapter does more often what the base model missed: the tool-use adapter finds the data and the rules before acting, the coding adapter writes its own test or scratch files on SWE-bench Pro V2, the constraints adapter carries earlier turns into its answers, the chart adapter measures the marks in code, and the document adapter reads AutomationBench's spreadsheets. 

---
# Long-Context Generation Is a Sampling Problem (When You Turn Up the Heat) 

**Authors**: Allen G. Roush, Minh Nhat Nguyen, Ravid Shwartz-Ziv, Judah Goldfeder, Sanjay Basu  

**Link**: [PDF](https://arxiv.org/pdf/2611.22148)  

**Abstract**: Open-weight language models are widely observed to get worse the longer they generate. Often, attempts to fix this involve more training. At high temperature, The sampler, i.e. the rule that picks the next token out of the model's probabilities, decides whether long and agentic generation survives and/or thrives. We test 29 open-weight models from 1B to 122B parameters on ten established benchmarks plus needle retrieval out to one million tokens, covering SWE-bench Verified coding agents, long chain-of-thought reasoning, long-form writing, long structured output and long-context reading. Samplers that keep their pool of candidate tokens small and steady as temperature rises don't lose accuracy. Samplers that let in low-probability tokens fall apart. At temperature 2, coding agents running top-p, top-k or Top-H resolve 0 of 100 GitHub issues, while the same agents running top-nsigma resolve 46 to 62 and win every issue the two samplers disagree on, on three Qwen3.5 models. Over all benchmarks top-nsigma keeps a median 100% of its temperature-1 score where top-p keeps 26%. Counting how many tokens each sampler lets through explains why. In one Llama-3.1-8B story, going from temperature 1 to 2 grows top-p's pool from 4 tokens to 51,845 while top-nsigma stays at 3. Across 133 combinations of sampler setting and temperature, the typical pool size predicts judged quality (Spearman correlation of -0.77), and when we pick the pool at temperature 1 and only apply temperature afterwards, Top-H, min-p and p-less stop collapsing. At the default temperature of 1 most samplers perform about the same, We release all code and data to the public. 

---
# Weight-Space Programming Doesn't Need Transformers 

**Authors**: Hyunjong Ok, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.22055)  

**Abstract**: Standard neural architectures such as the Transformer were chosen because they train well with gradient descent and run fast on GPUs. In weight-space programming, a large language model (LLM) writes target weights by hand without gradient-based training, so those reasons no longer apply. Should we keep the Transformer? Given minimal guidance, ten labeled examples, and a sandbox, mainly on multiple-choice questions, we task three frontier LLMs with either generating the weights of a fixed architecture or designing the architecture themselves. When free to design, all three compile plain-text facts into a sparse table queried by a fixed procedure, reaching up to 54.4% test accuracy. An LLM-designed model achieves higher test accuracy than the most accurate released Transformer that the same LLM wrote while using about 1,800x fewer parameters. Without gradients, they choose a table, in which the LLM can state what each weight means and can find and rewrite a wrong one. Each wrong answer points to one row, and removing a row changes about one other answer. From this, we propose an associative-memory-based model up to 17x faster on GPUs than a same-sized Transformer. We also test pretrained weights, other benchmarks, and tasks where models generate answers. Pretrained LLMs remain more accurate; the claim concerns which structure to build when writing weights. 

---
# Thinking Harder Doesn't Lead to Better Choices: Reasoning Effort Increases the Confidence but Not the Correctness of LLM Shopping Agents 

**Authors**: Adi Krishnamoorthy, X.Y. Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.22033)  

**Abstract**: Large language models (LLMs) are increasingly deployed as autonomous shopping agents that process multimodal product listings (consisting of visual and text data) and recommend purchases on behalf of consumers. Naturally, one might assume that models with more thinking power would make higher-quality purchasing decisions. To test this belief, we ask two questions about such agents: does asking a model to think harder make it choose better products, and does the confidence it reports tell us the likelihood it may be wrong? We benchmark ten frontier models on three product-choice benchmarks built from WebShop, Amazon's SQID and Wayfair's WANDS platforms to see if LLMs with more thinking power actually make better-quality product purchase decisions. Across all three benchmarks, we found that increasing the thinking effort does not increase decision quality in a statistically meaningful way. The same change in thinking effort raises stated confidence at a statistically significant level, so extra thinking buys confidence rather than correctness. We also measure the calibration of these models (how well their reported confidence percentage matches their true correctness percentage) in these tasks and show that most models are not well-calibrated at every effort level. We also find that errors are strongly shared: when one model errs in a task, several others tend to err on that same task. Furthermore, on tasks where several models all err, a large number of them pick the same wrong product. 

---
# SkillBuffBench: Measuring What Agent Skills Contribute across Harnesses, Models, Skill Access Modes and Memory 

**Authors**: Gengyue Dong, Ye Wei, Zheng Zhu, Yidong Han, Jiale Miao, Xiaojun Jia, Guanghui Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.22017)  

**Abstract**: Agent Skills, file-based packages of procedural knowledge that a coding agent can discover and load on demand, are now shipped by every major agent harness, yet the evidence for how much they help comes almost entirely from settings in which the right Skill is hand-picked for the task and the agent starts from a blank slate. Neither holds in deployment, where an agent chooses from whatever library happens to be installed and frequently carries memory from earlier work. We present SkillBuffBench, a benchmark and paired evaluation protocol that treats Skill access as a three-level factor (no Skill, a curated Skill, and automatic invocation from an installed library), crosses it with memory (off and on), with four agent harnesses and with multiple models (the state-of-the-art proprietary models from Anthropic, Google and OpenAI, and the state-of-the-art open-weight models from Chinese providers), every model being run under every harness, and evaluates the result on research tasks contributed by domain experts together with the Skills they use, currently in materials science and biology, with the suite extended on a rolling basis. Every run is marked by a deterministic verifier; uncertainty is reported with a paired, task-level bootstrap; and cost, agent rounds and wall-clock time are reported alongside the score rather than folded into it. This version of the paper describes the design of the benchmark, its evaluation protocol and its current tasks and Skills. The evaluation is in progress: results will be added once the runs are complete, and the task suite, results and analysis will continue to be updated on a rolling basis. 

---
# Large Language Models as Nonlinear Dynamical Systems: The Three Clocks of Depth, Training, and Generation 

**Authors**: Maor Farid  

**Link**: [PDF](https://arxiv.org/pdf/2611.21983)  

**Abstract**: Language models are dynamical systems. A transformer is iterated over its layers, over the steps of its training and over the tokens it generates, and on each of these three clocks it shows behavior that nonlinear dynamics has studied for a century: fixed points, bifurcations, synchronization, chaos and noise-driven escape from potential wells. The two communities rarely meet, so this behavior is often rediscovered under new names. We build a bridge between them. To describe, we condense about 200 papers into the Three-Clock Dictionary, which maps fifteen concepts of nonlinear dynamics onto each clock and marks every correspondence as proved, measured or conjectured, and we show each concept in a mechanical system and in a language model side by side. To explain, we derive reduced-order models: the softmax is a Boltzmann distribution, gradient descent an Euler step on a nonlinear spring, a loss spike a ball thrown out of a stiff valley, attention on the circle a Kuramoto model, and a repetition loop a well that deepens while it is occupied. We then test these explanations on open language models, on a laptop CPU. Escape from repetition loops in GPT-2 and Pythia follows the Arrhenius law with the logit gap as the barrier, and the barrier deepens and then saturates, so the temperature needed to leave a loop falls like $1/\ln N$ with the generation length $N$ and rises with model size. Coherent text lives in a narrow window of temperatures, which a small reasoning model shares at its recommended temperature. Training makes each Pythia layer more expansive, and a small transformer trains at a multi-mode edge of stability. The third step, solving problems of language models, is still ahead: controllers derived from the escape analysis remove loops but do not yet beat the best fixed temperature. We state twelve open problems where nonlinear dynamics could get there. 

---
# Modeling Document and Section Hierarchies in Embedding Space 

**Authors**: Kriste Krstovski, Sharyn O'Halloran, Kshitij Chandna  

**Link**: [PDF](https://arxiv.org/pdf/2611.21980)  

**Abstract**: Embedded topic models integrate word embeddings into the topic generation process, producing coherent topics and robust performance on large vocabularies. However, each document is represented by a single topic distribution, ignoring its internal structure. Many real world collections, including legal documents, company and government reports, and online discussion forums, exhibit hierarchical organization in which documents consist of multiple sections that often address distinct subtopics. We introduce the Hierarchical Embedded Topic Model (hETM), a structured topic model that couples the topic distribution of a document and those of its sections through a hierarchical prior in the embedding space. We derive an amortized variational inference procedure for hETM and evaluate it on structured collections from each of these domains. Experimental results show that hETM achieves the lowest or near-lowest held-out perplexity among flat and hierarchical topic model baselines, along with comparable or higher topic quality. It also performs strongly on a downstream clustering task, achieving the best clustering performance on most of the collections. 

---
# Beyond Bitwidth: Deployment-Constrained Mixed-Precision Quantization for Multilingual LLM Inference on NVIDIA Blackwell 

**Authors**: Hyojung Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.21974)  

**Abstract**: Low-bit LLM quantization is useful only if a model preserves service-relevant accuracy, executes through supported GPU kernels, and meets measured memory and latency limits. We study Qwen3-8B on NVIDIA B200 with vLLM, including INT2/INT3, GPTQ, NVFP4, and mixed-precision configurations. We verify native packed execution and precision on all 144 fused linear modules and evaluate Korean, English, numerical, and reasoning quality under preregistered gates. Packed INT2 and INT3 execute, but all seven tested INT2 grids degenerate on generation. Isolated module-restoration effects are strongly non-additive; Korean and reasoning sensitivity rankings disagree, and a promising INT4/INT8 development candidate fails independent final evaluation. A configuration with 16-bit activations, 98 NVFP4 weight-only modules, 46 INT8 modules, and an INT8 output head passes all gates on the untouched FINAL-81 set (5,168 items): 5.917 GiB of weight-loading memory (limit 5.98), Korean teacher-correct retention 0.904 (gate 0.90), and TPOT ratio 0.939 relative to BF16. It improves paired accuracy over NVFP4 W4A4 by +1.4 percentage points (Bonferroni-adjusted interval [+0.2, +2.6]), compared with +3.4 points on the selection set. This comparison changes both activation precision and INT8 protection; exploratory analyses suggest that the activation change contributes substantially, but the isolated allocation effect is not established. Final checkpoints were not retained, and bitwise rebuild reproducibility remains unverified. 

---
# It's My Turn! Interpretable Methods Reveal Developmental Differences from Mother-Child Conversational Dynamics 

**Authors**: A'di Dust, Anura Deshpande, Pat Levitt, Maja Matarić  

**Link**: [PDF](https://arxiv.org/pdf/2611.21969)  

**Abstract**: Understanding conversational interactions between mothers and children is an important challenge for computational behavioral modeling. Existing methods use traditional statistical approaches and subjective measures of conversational dynamics, without incorporating temporal or probabilistic modeling. We developed an interpretable computational framework that combines sequence analysis, information-theoretic measures, and Bayesian hierarchical modeling to characterize temporal conversational dynamics in parent-child play. We applied this framework to interactions from 94 mother-child dyads with two-year-old children. Sequence analysis revealed that dyads with lower language development (as defined by scores on the Bayley developmental test) exhibited more heterogeneous conversational organization and fewer serve-and-return exchanges than higher-development dyads, a reciprocal turn-taking pattern in which one partner initiates communication and the other responds. Mutual information analyses suggested differences in conversational predictability between lower- and higher-development dyads, although the bootstrap confidence intervals overlapped. Bayesian hierarchical models estimated temporal trajectories for maternal consecutive turns, child responses following maternal speech, and overlapping responses; HDIs showed that trajectories differed in whether they excluded 0, but there were no statistically credible differences between the Bayley groups. Overall, these results indicate that children with higher developmental scores have more organized and predictable communication styles, with more ``serve-and-return" patterns with their mothers. These results demonstrate that interpretable computational representations of conversational dynamics capture developmental differences beyond simple measures of speech quantity. 

---
# Solver to System: LLM Reasoning Across Representations and Execution in Scientific Simulation 

**Authors**: Eliza Berman, Rohan Subramanian, Shreya Jha, Lohit Jagarapu, Romain Lopez, Rajesh Ranganath  

**Link**: [PDF](https://arxiv.org/pdf/2611.21964)  

**Abstract**: Simulators based on partial differential equations (PDEs) are widely used to model physical systems, but generating and interpreting PDE solver code with large language models (LLMs) remains challenging. Scientific simulation code is a useful testbed for LLM scientific reasoning because (1) one system can be represented by code, equations, natural language, and numerical trajectories and (2) code can execute successfully while implementing invalid physics. We introduce Solver2System, a dataset of PDE solvers with multiple representations, controlled corruptions, and lexical perturbations, to evaluate whether LLMs can recover the underlying physical system through (1) cross-representation consistency and (2) static and agentic detection of physical invalidity. Across eight open-weight models, LLMs systematically favor some representations when evidence conflicts, and lexical perturbations to code alter interpretation of other unchanged representations. Execution improves physical-validity judgments, but weaker agents can judge correctly without identifying the causal defect. Thus, LLMs may reach correct judgments despite inconsistent representation reconciliation or causal misdiagnosis. 

---
# Conserved Verdicts, Drifting Commitments: Identifying Harmful Overthinking 

**Authors**: Jianghong Zhou, Hao Rong, Chen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.21946)  

**Abstract**: A reasoning model that has reached a correct answer can lose it by continuing to think. Deciding whether to stop means comparing two probabilities at the same point of a chain: that the answer the model would give if stopped now is correct (its commitment), and that the answer it will finish with is correct (its verdict). We show that the verdict alone, which is what final-correctness probes estimate, cannot make this comparison. As a forecast of a fixed outcome it has no predictable drift, whether continuing helps or hurts. The commitment-verdict gap equals the expected remaining change in answer quality and the accuracy gained by stopping. A second identity splits any accuracy-versus-budget curve into answer change within chains and change in which chains are still running. We measure both on MATH-500. At all 379 points where DeepSeek-R1-Distill-Qwen-1.5B holds a correct answer, selected without regard to how its chain ends, continuing loses 3.6 points on average (95% CI [2.2, 5.2]), and the loss concentrates in a few problems. In six reasoning models from three families, on MATH-500 and OlympiadBench and with budgets up to 16,384 tokens, declining accuracy curves come from which chains remain, while the chains still reasoning keep or improve their answers. An oracle stopping rule gains at most +1.8 and +2.5 points at 1.5B and 7B. The framework says what to measure before building a stopping rule. 

---
# DNeo: Your Parallel Draft Model Needs Better Target Feature Encoding 

**Authors**: Jameson Sandler, Shane Bergsma, Mostafa Elhoushi, Bin Claire Zhang, Gavia Gray, Joel Hestness, Nolan Dey, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2611.21925)  

**Abstract**: Speculative-diffusion decoding uses a small parallel drafter model to generate b draft tokens in one forward pass, which then undergo parallel verification via a larger target model. Faster inference can be achieved when multiple draft tokens pass verification simultaneously. Recent parallel draft architectures (DFlash, DSpark) improve draft quality by conditioning on rich features computed by the target, through a mechanism known as feature encoding. However, standard feature encoding mechanisms have three limitations: (1) they are lossy, applying encoding projections that are destructive to the underlying target feature-space via a discovered effect known as residual mixing, (2) they are memory-less, with no record of previously processed proposals, and (3) they utilize features sampled uniformly from across the target layers, omitting important hidden states. To address these issues, we propose Mixture-of-Features Attention (MoFA), a lightweight feature-encoding approach equipped with a working memory that encodes previously rejected proposals. Leveraging MoFA, rejection memory, and a set of further enhancements, including 2-tap causal convolutions, prefix-only cross-attention, and non-uniform target layer selection, the final architecture, DNeo, achieves 1.3x faster generation speed than SoTA (DFlash-2) without impacting drafter footprint or generation quality. DNeo marks a new state of the art in high quality, fast LLM inference. 

---
# The Hidden Cost of MoE Diffusion Decoding: Profiling and Fixing the Sampler 

**Authors**: Harshit Samrat, Ramakrishna Paramahamsa Kompella  

**Link**: [PDF](https://arxiv.org/pdf/2611.21852)  

**Abstract**: Masked diffusion language models (dLLMs) decode by iteratively denoising a full canvas of tokens, so every forward pass runs a per-token sampling and stopping pipeline over the entire canvas-by-vocabulary logits tensor. We profile single-user decoding of a 26B-parameter diffusion Mixture-of-Experts model on a bandwidth-bound edge accelerator (NVIDIA GB10) and report a result that is easy to miss: after weight quantisation, this logits pipeline, not the expert GEMMs, is the single largest consumer of decode time. Unlike autoregressive decoding, where sampling touches one position's logits per committed token, diffusion decoding processes S times as many logit entries per output token, where S is the number of denoising steps. We show this pipeline is largely redundant or fusable and remove it in two lossless tiers: a bit-exact tier whose output is byte-identical to the reference decoder, and an accuracy-parity tier using a fused single-pass entropy kernel and Gumbel-max sampling. Combined with a W4A16 expert kernel, the full stack reaches 2.4-2.7 times end-to-end speedup over bf16 at unchanged benchmark accuracy on GSM8K, HumanEval, and MATH, on a single 273 GB/s device. We confirm the finding on a second, independently built diffusion MoE (LLaDA2.0-mini: pipeline 28.8% of decode, fused W4A16 3.3 times at accuracy parity) and show it disappears on a dense diffusion model (3.6%). The cost is specific to MoE diffusion, where quantisation makes the backbone cheap. Code and quantised weights are released. 

---
# Bookkeeping Is Not Judgement: EJBench, an Engineering Judgement Benchmark Built from a Conjugate Heat Transfer Design Campaign 

**Authors**: James Sadler, Rotimi Alabi, Kotub Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2611.21707)  

**Abstract**: We introduce EJBench, a benchmark of engineering judgement: whether a model can predict how a design behaves at new conditions from a few earlier results, and decide on it. Unlike engineering benchmarks that test whether an agent can operate simulation or CAD software, EJBench is software-agnostic and asks what an engineer would conclude without running a simulation. Its tasks come from a conjugate heat transfer design campaign for battery cold plates, in three families: matching unlabelled test results to their operating points, choosing which stored runs to reveal before signing off a design, and ranking new duties by peak cell temperature. Every answer is taken from a converged simulation, not from an LLM judge. The best of ten frontier models, GPT-6 Astra, scores 0.71, and models state almost the same confidence on tasks they get wrong as on tasks they get right. We also post-train Qwen3.5-9B with LoRA and reinforcement learning, rewarded by the share of answer criteria the simulation results confirm; on the held-out test split its score rises from 0.071 to 0.106. We release the train, validation and test splits, the evaluation harness and the leaderboard. 

---
# Watermark Evidence under Open-Weight LLM Modification: Exact Bounds for Low-Rank Output Watermarks 

**Authors**: Prashant Shekhar, Dumindu Samaraweera, Caroline Howard  

**Link**: [PDF](https://arxiv.org/pdf/2611.21702)  

**Abstract**: Open-weight language models can continue generating after their released weights are modified, while a watermark verifier still scores outputs against the original base and marked models. We study how such modifications change the evidence available to an OpenStamp-style verifier. At each shared prefix, we represent the modification as a residual in the next-token logits. We derive an exact finite-vocabulary lower bound on the expected log-likelihood-ratio evidence under a residual-span constraint, along with a sharp span-only deviation bound. We also characterize the first-order evidence shift through a covariance term with an explicit second-order remainder, and determine the residual span needed to reach a target evidence level. For sequence-level detection, we derive an exact conditional exponential-moment bound and combine it with a supermartingale argument and a distribution-free order-statistic tolerance bound. This yields a high-confidence upper bound on miss probability after attack selection on calibration data and evaluation on independent certification continuations. In experiments with OpenStamp on Llama-2-7B and Mistral-7B-v0.3, INT8 quantization and 30% pruning retain test true-positive rates of 85.6-94.8%, while 500-step LoRA fine-tuning lowers them to 28.8% and 11.6%. The covariance prediction tracks the measured mean LoRA evidence drift on both models, with no remainder-bound violations across 160,000 test prefixes. Additionally, we observe zero false positives among 500 base-model continuations per model. Through these theoretical and empirical results, the current research provides a better understanding of watermark robustness to model-editing/tuning attacks. 

---
# Asking What Matters: Disagreement-Guided Ambiguity Detection for Interactive Text-to-SQL 

**Authors**: Su Dong, Jiayan Lin, Yujia Liu, Zijin Hong, Qinggang Zhang, Feiran Huang, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.21645)  

**Abstract**: Interactive text-to-SQL requires an LLM agent not only to generate executable SQL, but also to decide which uncertainty to clarify within a limited interaction budget. However, current clarification policies have three limitations: (i) they target surface ambiguity without testing its SQL impact; (ii) they miss ambiguities implicit in schema routes or domain knowledge; and (iii) they spend limited turns on unfocused or redundant questions. These limitations expose a critical challenge: identifying SQL-impacting ambiguity before the intended SQL is known. In this paper, we propose DisGuAD, a Disagreement-Guided Ambiguity Detection framework that specializes clarification selection. DisGuAD treats disagreement among plausible SQL interpretations as direct evidence of which uncertainty matters to the final query. Specifically, it constructs a query-term inventory, elicits contrastive SQL interpretations through complementary probe families, compares their canonical SQL components, and attributes recurring differences to user-facing clarification targets. A lightweight prioritization and dialogue-state mechanism then selects one high-impact unresolved target for clarification. Experiments on BIRD-Interact Lite and Full with two LLM backbones demonstrate consistent improvements in task success and clarification quality. Compared with the original interaction policy, DisGuAD improves normalized Reward by 4.40--8.97 points and raises overall critical-ambiguity coverage from 48.62% to 82.75% and from 45.76% to 85.14%. Its clarification signals further improve two independently designed static text-to-SQL frameworks by up to 13.97 Reward points while keeping their SQL generators fixed. These results demonstrate that disagreement among plausible SQL queries provides actionable evidence for selecting clarification targets that matter to downstream generation. 

---
# ClearTalk: Data-Driven Mandarin Articulation Analysis and Personalized Communication Support for Children with Dysarthria 

**Authors**: Ben Xiaolu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.21639)  

**Abstract**: Children with dysarthria may experience reduced intelligibility and difficulty expressing everyday needs. ClearTalk is a smartwatch-based system that converts Mandarin speech-recognition differences into interpretable articulatory profiles for personalized practice and staged communication. Target and recognized utterances are aligned at the syllable level, summarized across seven articulatory positions, and ranked by a count-sensitive asymmetric heuristic. For 13 dysarthric speakers from the Mandarin Dysarthria Speech Corpus, system syllable-level accuracy showed a Pearson correlation of 0.908 with expert intelligibility, with an R-squared value of 0.824 and a p-value below 0.001. The system-selected priority targets achieved an 86.7 percent overlap with the consensus of three blinded speech-language pathologists; across 210 target consonants, automated labels achieved 86.7 percent agreement, a Cohen's kappa of 0.76, and a macro-F1 score of 0.81. In a balanced single-session pilot with 15 children, mean pre/post gains were 14.4, 29.4, and 27.7 percentage points for mild, moderate, and severe groups, while communication-round completion rates were 81.3, 66.7, and 48.0 percent. These findings support the feasibility of using interpretable patterns of speech errors to guide wearable articulation practice and communication assistance. 

---
