# Masking or Mitigating? Deconstructing the Impact of Query Rewriting on Retriever Biases in RAG 

**Authors**: Agam Goyal, Koyel Mukherjee, Apoorv Saxena, Anirudh Phukan, Eshwar Chandrasekharan, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2604.06097)  

**Abstract**: Dense retrievers in retrieval-augmented generation (RAG) systems exhibit systematic biases -- including brevity, position, literal matching, and repetition biases -- that can compromise retrieval quality. Query rewriting techniques are now standard in RAG pipelines, yet their impact on these biases remains unexplored. We present the first systematic study of how query enhancement techniques affect dense retrieval biases, evaluating five methods across six retrievers. Our findings reveal that simple LLM-based rewriting achieves the strongest aggregate bias reduction (54\%), yet fails under adversarial conditions where multiple biases combine. Mechanistic analysis uncovers two distinct mechanisms: simple rewriting reduces bias through increased score variance, while pseudo-document generation methods achieve reduction through genuine decorrelation from bias-inducing features. However, no technique uniformly addresses all biases, and effects vary substantially across retrievers. Our results provide practical guidance for selecting query enhancement strategies based on specific bias vulnerabilities. More broadly, we establish a taxonomy distinguishing query-document interaction biases from document encoding biases, clarifying the limits of query-side interventions for debiasing RAG systems. 

---
# CUE-R: Beyond the Final Answer in Retrieval-Augmented Generation 

**Authors**: Siddharth Jain, Venkat Narayan Vedam  

**Link**: [PDF](https://arxiv.org/pdf/2604.05467)  

**Abstract**: As language models shift from single-shot answer generation toward multi-step reasoning that retrieves and consumes evidence mid-inference, evaluating the role of individual retrieved items becomes more important. Existing RAG evaluation typically targets final-answer quality, citation faithfulness, or answer-level attribution, but none of these directly targets the intervention-based, per-evidence-item utility view we study here. We introduce CUE-R, a lightweight intervention-based framework for measuring per-evidence-item operational utility in single-shot RAG using shallow observable retrieval-use traces. CUE-R perturbs individual evidence items via REMOVE, REPLACE, and DUPLICATE operators, then measures changes along three utility axes (correctness, proxy-based grounding faithfulness, and confidence error) plus a trace-divergence signal. We also outline an operational evidence-role taxonomy for interpreting intervention outcomes. Experiments on HotpotQA and 2WikiMultihopQA with Qwen-3 8B and GPT-5.2 reveal a consistent pattern: REMOVE and REPLACE substantially harm correctness and grounding while producing large trace shifts, whereas DUPLICATE is often answer-redundant yet not fully behaviorally neutral. A zero-retrieval control confirms that these effects arise from degradation of meaningful retrieval. A two-support ablation further shows that multi-hop evidence items can interact non-additively: removing both supports harms performance far more than either single removal. Our results suggest that answer-only evaluation misses important evidence effects and that intervention-based utility analysis is a practical complement for RAG evaluation. 

---
# MG$^2$-RAG: Multi-Granularity Graph for Multimodal Retrieval-Augmented Generation 

**Authors**: Sijun Dai, Qiang Huang, Xiaoxing You, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04969)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucinations in Multimodal Large Language Models (MLLMs), yet existing systems struggle with complex cross-modal reasoning. Flat vector retrieval often ignores structural dependencies, while current graph-based methods rely on costly ``translation-to-text'' pipelines that discard fine-grained visual information. To address these limitations, we propose \textbf{MG$^2$-RAG}, a lightweight \textbf{M}ulti-\textbf{G}ranularity \textbf{G}raph \textbf{RAG} framework that jointly improves graph construction, modality fusion, and cross-modal retrieval. MG$^2$-RAG constructs a hierarchical multimodal knowledge graph by combining lightweight textual parsing with entity-driven visual grounding, enabling textual entities and visual regions to be fused into unified multimodal nodes that preserve atomic evidence. Building on this representation, we introduce a multi-granularity graph retrieval mechanism that aggregates dense similarities and propagates relevance across the graph to support structured multi-hop reasoning. Extensive experiments across four representative multimodal tasks (i.e., retrieval, knowledge-based VQA, reasoning, and classification) demonstrate that MG$^2$-RAG consistently achieves state-of-the-art performance while reducing graph construction overhead with an average 43.3$\times$ speedup and 23.9$\times$ cost reduction compared with advanced graph-based frameworks. 

---
# Web Retrieval-Aware Chunking (W-RAC) for Efficient and Cost-Effective Retrieval-Augmented Generation Systems 

**Authors**: Uday Allu, Sonu Kedia, Tanmay Odapally, Biddwan Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2604.04936)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems critically depend on effective document chunking strategies to balance retrieval quality, latency, and operational cost. Traditional chunking approaches, such as fixed-size, rule-based, or fully agentic chunking, often suffer from high token consumption, redundant text generation, limited scalability, and poor debuggability, especially for large-scale web content ingestion. In this paper, we propose Web Retrieval-Aware Chunking (W-RAC), a novel, cost-efficient chunking framework designed specifically for web-based documents. W-RAC decouples text extraction from semantic chunk planning by representing parsed web content as structured, ID-addressable units and leveraging large language models (LLMs) only for retrieval-aware grouping decisions rather than text generation. This significantly reduces token usage, eliminates hallucination risks, and improves system this http URL analysis and architectural comparison demonstrate that W-RAC achieves comparable or better retrieval performance than traditional chunking approaches while reducing chunking-related LLM costs by an order of magnitude. 

---
# From PDF to RAG-Ready: Evaluating Document Conversion Frameworks for Domain-Specific Question Answering 

**Authors**: José Guilherme Marques dos Santos, Ricardo Yang, Rui Humberto Pereira, Alexandre Sousa, Brígida Mónica Faria, Henrique Lopes Cardoso, José Duarte, José Luís Reis, Luís Paulo Reis, Pedro Pimenta, José Paulo Marques dos Santos  

**Link**: [PDF](https://arxiv.org/pdf/2604.04948)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems depend critically on the quality of document preprocessing, yet no prior study has evaluated PDF processing frameworks by their impact on downstream question-answering accuracy. We address this gap through a systematic comparison of four open-source PDF-to-Markdown conversion frameworks, Docling, MinerU, Marker, and DeepSeek OCR, across 19 pipeline configurations for extracting text and other contents from PDFs, varying the conversion tool, cleaning transformations, splitting strategy, and metadata enrichment. Evaluation was performed using a manually curated 50-question benchmark over a corpus of 36 Portuguese administrative documents (1,706 pages, ~492K words), with LLM-as-judge scoring averaged over 10 runs. Two baselines bounded the results: naïve PDFLoader (86.9%) and manually curated Markdown (97.1%). Docling with hierarchical splitting and image descriptions achieved the highest automated accuracy (94.1%). Metadata enrichment and hierarchy-aware chunking contributed more to accuracy than the conversion framework choice alone. Font-based hierarchy rebuilding consistently outperformed LLM-based approaches. An exploratory GraphRAG implementation scored only 82%, underperforming basic RAG, suggesting that naïve knowledge graph construction without ontological guidance does not yet justify its added complexity. These findings demonstrate that data preparation quality is the dominant factor in RAG system performance. 

---
# Improving Clinical Trial Recruitment using Clinical Narratives and Large Language Models 

**Authors**: Ziyi Chen, Mengxian Lyu, Cheng Peng, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.05190)  

**Abstract**: Screening patients for enrollment is a well-known, labor-intensive bottleneck that leads to under-enrollment and, ultimately, trial failures. Recent breakthroughs in large language models (LLMs) offer a promising opportunity to use artificial intelligence to improve screening. This study systematically explored both encoder- and decoder-based generative LLMs for screening clinical narratives to facilitate clinical trial recruitment. We examined both general-purpose LLMs and medical-adapted LLMs and explored three strategies to alleviate the "Lost in the Middle" issue when handling long documents, including 1) Original long-context: using the default context windows of LLMs, 2) NER-based extractive summarization: converting the long document into summarizations using named entity recognition, 3) RAG: dynamic evidence retrieval based on eligibility criteria. The 2018 N2C2 Track 1 benchmark dataset is used for evaluation. Our experimental results show that the MedGemma model with the RAG strategy achieved the best micro-F1 score of 89.05%, outperforming other models. Generative LLMs have remarkably improved trial criteria that require long-term reasoning across long documents, whereas trial criteria that span a short piece of context (e.g., lab tests) show incremental improvements. The real-world adoption of LLMs for trial recruitment must consider specific criteria for selecting among rule-based queries, encoder-based LLMs, and generative LLMs to maximize efficiency within reasonable computing costs. 

---
# WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering 

**Authors**: Yingjian Zhu, Xinming Wang, Kun Ding, Ying Wang, Bin Fan, Shiming Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.05818)  

**Abstract**: Multi-modal Retrieval-Augmented Generation (RAG) has emerged as a highly effective paradigm for Knowledge-Based Visual Question Answering (KB-VQA). Despite recent advancements, prevailing methods still primarily depend on images as the retrieval key, and often overlook or misplace the role of Vision-Language Models (VLMs), thereby failing to leverage their potential fully. In this paper, we introduce WikiSeeker, a novel multi-modal RAG framework that bridges these gaps by proposing a multi-modal retriever and redefining the role of VLMs. Rather than serving merely as answer generators, we assign VLMs two specialized agents: a Refiner and an Inspector. The Refiner utilizes the capability of VLMs to rewrite the textual query according to the input image, significantly improving the performance of the multimodal retriever. The Inspector facilitates a decoupled generation strategy by selectively routing reliable retrieved context to another LLM for answer generation, while relying on the VLM's internal knowledge when retrieval is unreliable. Extensive experiments on EVQA, InfoSeek, and M2KR demonstrate that WikiSeeker achieves state-of-the-art performance, with substantial improvements in both retrieval accuracy and answer quality. Our code will be released on this https URL. 

---
# ResearchEVO: An End-to-End Framework for Automated Scientific Discovery and Documentation 

**Authors**: Zhe Zhao, Haibin Wen, Jiaming Ma, Jiachang Zhan, Tianyi Xu, Ye Wei, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.05587)  

**Abstract**: An important recurring pattern in scientific breakthroughs is a two-stage process: an initial phase of undirected experimentation that yields an unexpected finding, followed by a retrospective phase that explains why the finding works and situates it within existing theory. We present ResearchEVO, an end-to-end framework that computationally instantiates this discover-then-explain paradigm. The Evolution Phase employs LLM-guided bi-dimensional co-evolution -- simultaneously optimizing both algorithmic logic and overall architecture -- to search the space of code implementations purely by fitness, without requiring any understanding of the solutions it produces. The Writing Phase then takes the best-performing algorithm and autonomously generates a complete, publication-ready research paper through sentence-level retrieval-augmented generation with explicit anti-hallucination verification and automated experiment design. To our knowledge, ResearchEVO is the first system to cover this full pipeline end to end: no prior work jointly performs principled algorithm evolution and literature-grounded scientific documentation. We validate the framework on two cross-disciplinary scientific problems -- Quantum Error Correction using real Google quantum hardware data, and Physics-Informed Neural Networks -- where the Evolution Phase discovered human-interpretable algorithmic mechanisms that had not been previously proposed in the respective domain literatures. In both cases, the Writing Phase autonomously produced compilable LaTeX manuscripts that correctly grounded these blind discoveries in existing theory via RAG, with zero fabricated citations. 

---
# PRISM-MCTS: Learning from Reasoning Trajectories with Metacognitive Reflection 

**Authors**: Siyuan Cheng, Bozhong Tian, YanChao Hao, Zheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.05424)  

**Abstract**: PRISM-MCTS: Learning from Reasoning Trajectories with Metacognitive Reflection Siyuan Cheng, Bozhong Tian, Yanchao Hao, Zheng Wei Published: 06 Apr 2026, Last Modified: 06 Apr 2026 ACL 2026 Findings Conference, Area Chairs, Reviewers, Publication Chairs, Authors Revisions BibTeX CC BY 4.0 Keywords: Efficient/Low-Resource Methods for NLP, Generation, Question Answering  The emergence of reasoning models, exemplified by OpenAI o1, signifies a transition from intuitive to deliberative cognition, effectively reorienting the scaling laws from pre-training paradigms toward test-time computation. While Monte Carlo Tree Search (MCTS) has shown promise in this domain, existing approaches typically treat each rollout as an isolated trajectory. This lack of information sharing leads to severe inefficiency and substantial computational redundancy, as the search process fails to leverage insights from prior explorations. To address these limitations, we propose PRISM-MCTS, a novel reasoning framework that draws inspiration from human parallel thinking and reflective processes. PRISM-MCTS integrates a Process Reward Model (PRM) with a dynamic shared memory, capturing both "Heuristics" and "Fallacies". By reinforcing successful strategies and pruning error-prone branches, PRISM-MCTS effectively achieves refinement. Furthermore, we develop a data-efficient training strategy for the PRM, achieving high-fidelity evaluation under a few-shot regime. Empirical evaluations across diverse reasoning benchmarks substantiate the efficacy of PRISM-MCTS. Notably, it halves the trajectory requirements on GPQA while surpassing MCTS-RAG and Search-o1, demonstrating that it scales inference by reasoning judiciously rather than exhaustively. 

---
# LatentAudit: Real-Time White-Box Faithfulness Monitoring for Retrieval-Augmented Generation with Verifiable Deployment 

**Authors**: Zhe Yu, Wenpeng Xing, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.05358)  

**Abstract**: Retrieval-augmented generation (RAG) mitigates hallucination but does not eliminate it: a deployed system must still decide, at inference time, whether its answer is actually supported by the retrieved evidence. We introduce LatentAudit, a white-box auditor that pools mid-to-late residual-stream activations from an open-weight generator and measures their Mahalanobis distance to the evidence representation. The resulting quadratic rule requires no auxiliary judge model, runs at generation time, and is simple enough to calibrate on a small held-out set. We show that residual-stream geometry carries a usable faithfulness signal, that this signal survives architecture changes and realistic retrieval failures, and that the same rule remains amenable to public verification. On PubMedQA with Llama-3-8B, LatentAudit reaches 0.942 AUROC with 0.77,ms overhead. Across three QA benchmarks and five model families (Llama-2/3, Qwen-2.5/3, Mistral), the monitor remains stable; under a four-way stress test with contradictions, retrieval misses, and partial-support noise, it reaches 0.9566--0.9815 AUROC on PubMedQA and 0.9142--0.9315 on HotpotQA. At 16-bit fixed-point precision, the audit rule preserves 99.8% of the FP16 AUROC, enabling Groth16-based public verification without revealing model weights or activations. Together, these results position residual-stream geometry as a practical basis for real-time RAG faithfulness monitoring and optional verifiable deployment. 

---
# MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library 

**Authors**: Md Shamimul Islam, Luis G. Jaimes, Ayesha S. Dina  

**Link**: [PDF](https://arxiv.org/pdf/2604.05458)  

**Abstract**: Network Intrusion Detection Systems (NIDS) face important limitations. Signature-based methods are effective for known attack patterns, but they struggle to detect zero-day attacks and often miss modified variants of previously known attacks, while many machine learning approaches offer limited interpretability. These challenges become even more severe in IoT environments because of resource constraints and heterogeneous protocols. To address these issues, we propose MA-IDS, a Multi-Agent Intrusion Detection System that combines Large Language Models (LLMs) with Retrieval Augmented Generation (RAG) for reasoning-driven intrusion detection. The proposed framework grounds LLM reasoning through a persistent, self-building Experience Library. Two specialized agents collaborate through a FAISS-based vector database: a Traffic Classification Agent that retrieves past error rules before each inference, and an Error Analysis Agent that converts misclassifications into human-readable detection rules stored for future retrieval, enabling continual learning through external knowledge accumulation, without modifying the underlying language model. Evaluated on NF-BoT-IoT and NF-ToN-IoT benchmark datasets, MA-IDS achieves Macro F1-Scores of 89.75% and 85.22%, improving over zero-shot baselines of 17% and 4.96% by more than 72 and 80 percentage points. These results are competitive with SVM while providing rule-level explanations for every classification decision, demonstrating that retrieval-augmented reasoning offers a principled path toward explainable, self-improving intrusion detection for IoT networks. 

---
# VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG 

**Authors**: Honghao Fu, Miao Xu, Yiwei Wang, Dailing Zhang, Liu Jun, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.05418)  

**Abstract**: Scaling multimodal large language models (MLLMs) to long videos is constrained by limited context windows. While retrieval-augmented generation (RAG) is a promising remedy by organizing query-relevant visual evidence into a compact context, most existing methods (i) flatten videos into independent segments, breaking their inherent spatio-temporal structure, and (ii) depend on explicit semantic matching, which can miss cues that are implicitly relevant to the query's intent. To overcome these limitations, we propose VideoStir, a structured and intent-aware long-video RAG framework. It firstly structures a video as a spatio-temporal graph at clip level, and then performs multi-hop retrieval to aggregate evidence across distant yet contextually related events. Furthermore, it introduces an MLLM-backed intent-relevance scorer that retrieves frames based on their alignment with the query's reasoning intent. To support this capability, we curate IR-600K, a large-scale dataset tailored for learning frame-query intent alignment. Experiments show that VideoStir is competitive with state-of-the-art baselines without relying on auxiliary information, highlighting the promise of shifting long-video RAG from flattened semantic matching to structured, intent-aware reasoning. Codes and checkpoints are available at Github. 

---
# Region-R1: Reinforcing Query-Side Region Cropping for Multi-Modal Re-Ranking 

**Authors**: Chan-Wei Hu, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2604.05268)  

**Abstract**: Multi-modal retrieval-augmented generation (MM-RAG) relies heavily on re-rankers to surface the most relevant evidence for image-question queries. However, standard re-rankers typically process the full query image as a global embedding, making them susceptible to visual distractors (e.g., background clutter) that skew similarity scores. We propose Region-R1, a query-side region cropping framework that formulates region selection as a decision-making problem during re-ranking, allowing the system to learn to retain the full image or focus only on a question-relevant region before scoring the retrieved candidates. Region-R1 learns a policy with a novel region-aware group relative policy optimization (r-GRPO) to dynamically crop a discriminative region. Across two challenging benchmarks, E-VQA and InfoSeek, Region-R1 delivers consistent gains, achieving state-of-the-art performances by increasing conditional Recall@1 by up to 20%. These results show the great promise of query-side adaptation as a simple but effective way to strengthen MM-RAG re-ranking. 

---
# LLMs Should Express Uncertainty Explicitly 

**Authors**: Junyu Guo, Shangding Gu, Ming Jin, Costas Spanos, Javad Lavaei  

**Link**: [PDF](https://arxiv.org/pdf/2604.05306)  

**Abstract**: Large language models are increasingly used in settings where uncertainty must drive decisions such as abstention, retrieval, and verification. Most existing methods treat uncertainty as a latent quantity to estimate after generation rather than a signal the model is trained to express. We instead study uncertainty as an interface for control. We compare two complementary interfaces: a global interface, where the model verbalizes a calibrated confidence score for its final answer, and a local interface, where the model emits an explicit <uncertain> marker during reasoning when it enters a high-risk state. These interfaces provide different but complementary benefits. Verbalized confidence substantially improves calibration, reduces overconfident errors, and yields the strongest overall Adaptive RAG controller while using retrieval more selectively. Reasoning-time uncertainty signaling makes previously silent failures visible during generation, improves wrong-answer coverage, and provides an effective high-recall retrieval trigger. Our findings further show that the two interfaces work differently internally: verbal confidence mainly refines how existing uncertainty is decoded, whereas reasoning-time signaling induces a broader late-layer reorganization. Together, these results suggest that effective uncertainty in LLMs should be trained as task-matched communication: global confidence for deciding whether to trust a final answer, and local signals for deciding when intervention is needed. 

---
# DQA: Diagnostic Question Answering for IT Support 

**Authors**: Vishaal Kapoor, Mariam Dundua, Sarthak Ahuja, Neda Kordjazi, Evren Yortucboylu, Vaibhavi Padala, Derek Ho, Jennifer Whitted, Rebecca Steinert  

**Link**: [PDF](https://arxiv.org/pdf/2604.05350)  

**Abstract**: Enterprise IT support interactions are fundamentally diagnostic: effective resolution requires iterative evidence gathering from ambiguous user reports to identify an underlying root cause. While retrieval-augmented generation (RAG) provides grounding through historical cases, standard multi-turn RAG systems lack explicit diagnostic state and therefore struggle to accumulate evidence and resolve competing hypotheses across turns. We introduce DQA, a diagnostic question-answering framework that maintains persistent diagnostic state and aggregates retrieved cases at the level of root causes rather than individual documents. DQA combines conversational query rewriting, retrieval aggregation, and state-conditioned response generation to support systematic troubleshooting under enterprise latency and context constraints. We evaluate DQA on 150 anonymized enterprise IT support scenarios using a replay-based protocol. Averaged over three independent runs, DQA achieves a 78.7% success rate under a trajectory-level success criterion, compared to 41.3% for a multi-turn RAG baseline, while reducing average turns from 8.4 to 3.9. 

---
# On the Role of Fault Localization Context for LLM-Based Program Repair 

**Authors**: Melika Sepidband, Hung Viet Pham, Hadi Hemmati  

**Link**: [PDF](https://arxiv.org/pdf/2604.05481)  

**Abstract**: Fault Localization (FL) is a key component of Large Language Model (LLM)-based Automated Program Repair (APR), yet its impact remains underexplored. In particular, it is unclear how much localization is needed, whether additional context beyond the predicted buggy location is beneficial, and how such context should be retrieved. We conduct a large-scale empirical study on 500 SWE-bench Verified instances using GPT-5-mini, evaluating 61 configurations that vary file-level, element-level, and line-level context. Our results show that more context does not consistently improve repair performance. File-level localization is the dominant factor, yielding a 15-17x improvement over a no-file baseline. Expanding file context is often associated with improved performance, with successful repairs most commonly observed in configurations with approximately 6-10 relevant files. Element-level context expansion provides conditional gains that depend strongly on the file context quality, while line-level context expansion frequently degrades performance due to noise amplification. LLM-based retrieval generally outperforms structural heuristics while using fewer files and tokens. Overall, the most effective FL context strategy typically combines a broad semantic understanding at higher abstraction levels with precise line-level localization. These findings challenge our assumption that increasing the localization context uniformly improves APR, and provide practical guidance for designing LLM-based FL strategies. 

---
# This Treatment Works, Right? Evaluating LLM Sensitivity to Patient Question Framing in Medical QA 

**Authors**: Hye Sun Yun, Geetika Kapoor, Michael Mackert, Ramez Kouzy, Wei Xu, Junyi Jessy Li, Byron C. Wallace  

**Link**: [PDF](https://arxiv.org/pdf/2604.05051)  

**Abstract**: Patients are increasingly turning to large language models (LLMs) with medical questions that are complex and difficult to articulate clearly. However, LLMs are sensitive to prompt phrasings and can be influenced by the way questions are worded. Ideally, LLMs should respond consistently regardless of phrasing, particularly when grounded in the same underlying evidence. We investigate this through a systematic evaluation in a controlled retrieval-augmented generation (RAG) setting for medical question answering (QA), where expert-selected documents are used rather than retrieved automatically. We examine two dimensions of patient query variation: question framing (positive vs. negative) and language style (technical vs. plain language). We construct a dataset of 6,614 query pairs grounded in clinical trial abstracts and evaluate response consistency across eight LLMs. Our findings show that positively- and negatively-framed pairs are significantly more likely to produce contradictory conclusions than same-framing pairs. This framing effect is further amplified in multi-turn conversations, where sustained persuasion increases inconsistency. We find no significant interaction between framing and language style. Our results demonstrate that LLM responses in medical QA can be systematically influenced through query phrasing alone, even when grounded in the same evidence, highlighting the importance of phrasing robustness as an evaluation criterion for RAG-based systems in high-stakes settings. 

---
# AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning 

**Authors**: Yuanfu Sun, Kang Li, Dongzhe Fan, Jiajin Liu, Qiaoyu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.05846)  

**Abstract**: Large Language Models (LLMs) increasingly rely on agentic capabilities-iterative retrieval, tool use, and decision-making-to overcome the limits of static, parametric knowledge. Yet existing agentic frameworks treat external information as unstructured text and fail to leverage the topological dependencies inherent in real-world data. To bridge this gap, we introduce Agentic Graph Learning (AGL), a paradigm that reframes graph learning as an interleaved process of topology-aware navigation and LLM-based inference. Specifically, we propose AgentGL, the first reinforcement learning (RL)-driven framework for AGL. AgentGL equips an LLM agent with graph-native tools for multi-scale exploration, regulates tool usage via search-constrained thinking to balance accuracy and efficiency, and employs a graph-conditioned curriculum RL strategy to stabilize long-horizon policy learning without step-wise supervision. Across diverse Text-Attributed Graph (TAG) benchmarks and multiple LLM backbones, AgentGL substantially outperforms strong GraphLLMs and GraphRAG baselines, achieving absolute improvements of up to 17.5% in node classification and 28.4% in link prediction. These results demonstrate that AGL is a promising frontier for enabling LLMs to autonomously navigate and reason over complex relational environments. The code is publicly available at this https URL. 

---
# THIVLVC: Retrieval Augmented Dependency Parsing for Latin 

**Authors**: Luc Pommeret, Thibault Wagret, Jules Deret  

**Link**: [PDF](https://arxiv.org/pdf/2604.05564)  

**Abstract**: We describe THIVLVC, a two-stage system for the EvaLatin 2026 Dependency Parsing task. Given a Latin sentence, we retrieve structurally similar entries from the CIRCSE treebank using sentence length and POS n-gram similarity, then prompt a large language model to refine the baseline parse from UDPipe using the retrieved examples and UD annotation guidelines. We submit two configurations: one without retrieval and one with retrieval (RAG). On poetry (Seneca), THIVLVC improves CLAS by +17 points over the UDPipe baseline; on prose (Thomas Aquinas), the gain is +1.5 CLAS. A double-blind error analysis of 300 divergences between our system and the gold standard reveals that, among unanimous annotator decisions, 53.3% favour THIVLVC, showing annotation inconsistencies both within and across treebanks. 

---
# Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting 

**Authors**: Jinhu Fu, Yan Bai, Longzhu He, Yihang Lou, Yanxiao Zhao, Li Sun, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.05540)  

**Abstract**: Large language models (LLMs) can effectively handle outdated information through knowledge editing. However, current approaches face two key limitations: (I) Poor generalization: Most approaches rigidly inject new knowledge without ensuring that the model can use it effectively to solve practical problems. (II) Narrow scope: Current methods focus primarily on structured fact triples, overlooking the diverse unstructured forms of factual information (e.g., news, articles) prevalent in real-world contexts. To address these challenges, we propose a new paradigm: teaching LLMs to edit knowledge via Chain of Thoughts (CoTs) reasoning (CoT2Edit). We first leverage language model agents for both structured and unstructured edited data to generate CoTs, building high-quality instruction data. The model is then trained to reason over edited knowledge through supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO). At inference time, we integrate Retrieval-Augmented Generation (RAG) to dynamically retrieve relevant edited facts for real-time knowledge editing. Experimental results demonstrate that our method achieves strong generalization across six diverse knowledge editing scenarios with just a single round of training on three open-source language models. The codes are available at this https URL. 

---
# RAG or Learning? Understanding the Limits of LLM Adaptation under Continuous Knowledge Drift in the Real World 

**Authors**: Hanbing Liu, Lang Cao, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.05096)  

**Abstract**: Large language models (LLMs) acquire most of their knowledge during pretraining, which ties them to a fixed snapshot of the world and makes adaptation to continuously evolving knowledge challenging. As facts, entities, and events change over time, models may experience continuous knowledge drift, resulting not only in outdated predictions but also in temporally inconsistent reasoning. Although existing approaches, such as continual finetuning, knowledge editing, and retrieval-augmented generation (RAG), aim to update or supplement model knowledge, they are rarely evaluated in settings that reflect chronological, evolving, and real-world knowledge evolution. In this work, we introduce a new benchmark of real-world dynamic events, constructed from time-stamped evidence that captures how knowledge evolves over time, which enables systematic evaluation of model adaptation under continuous knowledge drift. The benchmark reveals that most existing methods, including vanilla RAG and several learning-based approaches, struggle under this setting, exposing critical limitations such as catastrophic forgetting and temporal inconsistency. To mitigate these limitations, we propose a time-aware retrieval baseline, Chronos, which progressively organizes retrieved evidence into an Event Evolution Graph to enable more temporally consistent understanding in LLMs without additional training. Overall, this work provides a foundation for analyzing and advancing LLM adaptation to continuous knowledge drift in realistic settings. 

---
# Towards Trustworthy Report Generation: A Deep Research Agent with Progressive Confidence Estimation and Calibration 

**Authors**: Yi Yuan, Xuhong Wang, Shanzhe Lei  

**Link**: [PDF](https://arxiv.org/pdf/2604.05952)  

**Abstract**: As agent-based systems continue to evolve, deep research agents are capable of automatically generating research-style reports across diverse domains. While these agents promise to streamline information synthesis and knowledge exploration, existing evaluation frameworks-typically based on subjective dimensions-fail to capture a critical aspect of report quality: trustworthiness. In open-ended research scenarios where ground-truth answers are unavailable, current evaluation methods cannot effectively measure the epistemic confidence of generated content, making calibration difficult and leaving users susceptible to misleading or hallucinated information. To address this limitation, we propose a novel deep research agent that incorporates progressive confidence estimation and calibration within the report generation pipeline. Our system leverages a deliberative search model, featuring deep retrieval and multi-hop reasoning to ground outputs in verifiable evidence while assigning confidence scores to individual claims. Combined with a carefully designed workflow, this approach produces trustworthy reports with enhanced transparency. Experimental results and case studies demonstrate that our method substantially improves interpretability and significantly increases user trust. 

---
