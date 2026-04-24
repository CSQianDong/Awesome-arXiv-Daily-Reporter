# Thinking with Reasoning Skills: Fewer Tokens, More Accuracy 

**Authors**: Guangxiang Zhao, Qilong Shi, Xusen Xiao, Xiangzheng Zhang, Tong Yang, Lin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.21764)  

**Abstract**: Reasoning LLMs often spend substantial tokens on long intermediate reasoning traces (e.g., chain-of-thought) when solving new problems. We propose to summarize and store reusable reasoning skills distilled from extensive deliberation and trial-and-error exploration, and to retrieve these skills at inference time to guide future reasoning. Unlike the prevailing \emph{reasoning from scratch} paradigm, our approach first recalls relevant skills for each query, helping the model avoid redundant detours and focus on effective solution paths. We evaluate our method on coding and mathematical reasoning tasks, and find that it significantly reduces reasoning tokens while improving overall performance. The resulting lower per-request cost indicates strong practical and economic potential for real-world deployment. 

---
# Spatial Metaphors for LLM Memory: A Critical Analysis of the MemPalace Architecture 

**Authors**: Robin Dey, Panyanon Viradecha  

**Link**: [PDF](https://arxiv.org/pdf/2604.21284)  

**Abstract**: MemPalace is an open-source AI memory system that applies the ancient method of loci (memory palace) spatial metaphor to organize long-term memory for large language models; launched in April 2026, it accumulated over 47,000 GitHub stars in its first two weeks and claims state-of-the-art retrieval performance on the LongMemEval benchmark (96.6% Recall@5) without requiring any LLM inference at write time. Through independent codebase analysis, benchmark replication, and comparison with competing systems, we find that MemPalace's headline retrieval performance is attributable primarily to its verbatim storage philosophy combined with ChromaDB's default embedding model (all-MiniLM-L6-v2), rather than to its spatial organizational metaphor per se -- the palace hierarchy (Wings->Rooms->Closets->Drawers) operates as standard vector database metadata filtering, an effective but well-established technique. However, MemPalace makes several genuinely novel contributions: (1) a contrarian verbatim-first storage philosophy that challenges extraction-based competitors, (2) an extremely low wake-up cost (approximately 170 tokens) through its four-layer memory stack, (3) a fully deterministic, zero-LLM write path enabling offline operation at zero API cost, and (4) the first systematic application of spatial memory metaphors as an organizing principle for AI memory systems. We also note that the competitive landscape is evolving rapidly, with Mem0's April 2026 token-efficient algorithm raising their LongMemEval score from approximately 49% to 93.4%, narrowing the gap between extraction-based and verbatim approaches. Our analysis concludes that MemPalace represents significant architectural insight wrapped in overstated claims -- a pattern common in rapidly adopted open-source projects where marketing velocity exceeds scientific rigor. 

---
# TraceScope: Interactive URL Triage via Decoupled Checklist Adjudication 

**Authors**: Haolin Zhang, William Reber, Yuxuan Zhang, Guofei Gu, Jeff Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.21840)  

**Abstract**: Modern phishing campaigns increasingly evade snapshot-based URL classifiers using interaction gates (e.g., checkbox/slider challenges), delayed content rendering, and logo-less credential harvesters. This shifts URL triage from static classification toward an interactive forensics task: an analyst must actively navigate the page while isolating themselves from potential runtime exploits.
We present TraceScope, a decoupled triage pipeline that operationalizes this workflow at scale. To prevent the observer effect and ensure safety, a sandboxed operator agent drives a real GUI browser guided by visual motivation to elicit page behavior, freezing the session into an immutable evidence bundle. Separately, an adjudicator agent circumvents LLM context limitations by querying evidence on demand to verify a MITRE ATT&CK checklist, and generates an audit-ready report with extracted indicators of compromise (IOCs) and a final verdict.
Evaluated on 708 reachable URLs from existing dataset (241 verified phishing from PhishTank and 467 benign from Tranco-derived crawling), TraceScope achieves 0.94 precision and 0.78 recall, substantially improving recall over three prior visual/reference-based classifiers while producing reproducible, analyst-grade evidence suitable for review. More importantly, we manually curated a dataset of real-world phishing emails to evaluate our system in a practical setting. Our evaluation reveals that TraceScope demonstrates superior performance in a real-world scenario as well, successfully detecting sophisticated phishing attempts that current state-of-the-art defenses fail to identify. 

---
# Conjecture and Inquiry: Quantifying Software Performance Requirements via Interactive Retrieval-Augmented Preference Elicitation 

**Authors**: Wang Shi Hai, Chen Tao  

**Link**: [PDF](https://arxiv.org/pdf/2604.21380)  

**Abstract**: Since software performance requirements are documented in natural language, quantifying them into mathematical forms is essential for software engineering. Yet, the vagueness in performance requirements and uncertainty of human cognition have caused highly uncertain ambiguity in the interpretations, rendering their automated quantification an unaddressed and challenging problem. In this paper, we formalize the problem and propose IRAP, an approach that quantifies performance requirements into mathematical functions via interactive retrieval-augmented preference elicitation. IRAP differs from the others in that it explicitly derives from problem-specific knowledge to retrieve and reason the preferences, which also guides the progressive interaction with stakeholders, while reducing the cognitive overhead. Experiment results against 10 state-of-the-art methods on four real-world datasets demonstrate the superiority of IRAP on all cases with up to 40x improvements under as few as five rounds of interactions. 

---
# EngramaBench: Evaluating Long-Term Conversational Memory with Structured Graph Retrieval 

**Authors**: Julian Acuna  

**Link**: [PDF](https://arxiv.org/pdf/2604.21229)  

**Abstract**: Large language model assistants are increasingly expected to retain and reason over information accumulated across many sessions. We introduce EngramaBench, a benchmark for long-term conversational memory built around five personas, one hundred multi-session conversations, and one hundred fifty queries spanning factual recall, cross-space integration, temporal reasoning, adversarial abstention, and emergent synthesis. We evaluate Engrama, a graph-structured memory system, against GPT-4o full-context prompting and Mem0, an open-source vector-retrieval memory system. All three use the same answering model (GPT-4o), isolating the effect of memory architecture. GPT-4o full-context achieves the highest composite score (0.6186), while Engrama scores 0.5367 globally but is the only system to score higher than full-context prompting on cross-space reasoning (0.6532 vs. 0.6291, n=30). Mem0 is cheapest but substantially weaker (0.4809). Ablations reveal that the components driving Engrama's cross-space advantage trade off against global composite score, exposing a systems-level tension between structured memory specialization and aggregate optimization. 

---
# Adaptive Defense Orchestration for RAG: A Sentinel-Strategist Architecture against Multi-Vector Attacks 

**Authors**: Pranav Pallerla, Wilson Naik Bhukya, Bharath Vemula, Charan Ramtej Kodi  

**Link**: [PDF](https://arxiv.org/pdf/2604.20932)  

**Abstract**: Retrieval-augmented generation (RAG) systems are increasingly deployed in sensitive domains such as healthcare and law, where they rely on private, domain-specific knowledge. This capability introduces significant security risks, including membership inference, data poisoning, and unintended content leakage. A straightforward mitigation is to enable all relevant defenses simultaneously, but doing so incurs a substantial utility cost. In our experiments, an always-on defense stack reduces contextual recall by more than 40%, indicating that retrieval degradation is the primary failure mode. To mitigate this trade-off in RAG systems, we propose the Sentinel-Strategist architecture, a context-aware framework for risk analysis and defense selection. A Sentinel detects anomalous retrieval behavior, after which a Strategist selectively deploys only the defenses warranted by the query context. Evaluated across three benchmark datasets and five orchestration models, ADO is shown to eliminate MBA-style membership inference leakage while substantially recovering retrieval utility relative to a fully static defense stack, approaching undefended baseline levels. Under data poisoning, the strongest ADO variants reduce attack success to near zero while restoring contextual recall to more than 75% of the undefended baseline, although robustness remains sensitive to model choice. Overall, these findings show that adaptive, query-aware defense can substantially reduce the security-utility trade-off in RAG systems. 

---
# Clinical Reasoning AI for Oncology Treatment Planning: A Multi-Specialty Case-Based Evaluation 

**Authors**: Philippe E. Spiess, Md Muntasir Zitu, Alison Walker, Daniel A. Anaya, Robert M. Wenham, Michael Vogelbaum, Daniel Grass, Ali-Musa Jaffer, Amod Sarnaik, Caitlin McMullen, Christine Sam, John V. Kiluk, Tianshi Liu, Tiago Biachi, Julio Powsang, Jing-Yi Chern, Roger Li, Seth Felder, Samuel Reynolds, Michael Shafique, Alison Sheehan, Ashley Layman, Cydney A. Warfield, Derrick Legoas, Jaclyn Parrinello, Jena Schmitz, Kevin Eaton, Mark Honor, Luis Felipe, Issam ElNaqa, Elier Delgado, Talia Berler, Rachael V. Phillips, Frantz Francisque, Carlos Garcia Fernandez, Gilmer Valdes  

**Link**: [PDF](https://arxiv.org/pdf/2604.20869)  

**Abstract**: Background: More than 80% of U.S. cancer care is delivered in community settings, where survival remains worse than at academic centers. Clinicians must integrate genomics, staging, radiology, pathology, and changing guidelines, creating cognitive burden. We evaluated OncoBrain, an AI clinical reasoning platform for oncology treatment-plan generation, as an early step toward OGI.
Methods: OncoBrain combines general-purpose LLMs with a cancer-specific graph retrieval-augmented generation layer, a gold-standard treatment-plan corpus as long-term memory, and a model-agnostic safety layer (CHECK) for hallucination detection and suppression. We evaluated clinician-enriched case summaries across gynecologic, genitourinary, neuro-oncology, gastrointestinal/hepatobiliary, and hematologic malignancies. Three clinician groups completed structured evaluations of 173 cases using a common 16-item instrument: subspecialist oncologists reviewed 50 cases, physician reviewers 78, and advanced practice providers 45.
Results: Ratings were highest for scientific accuracy, evidence support, and safety, with lower but favorable scores for workflow integration and time savings. On a 5-point scale, mean alignment with evidence and guidelines was 4.60, 4.56, and 4.70 across subspecialists, physician reviewers, and advanced practice providers. Mean scores for absence of safety or misinformation concerns were 4.80, 4.40, and 4.60. Workflow integration averaged 4.50, 3.94, and 4.00; perceived time savings averaged 5.00, 3.89, and 3.60.
Conclusions: In this multi-specialty vignette-based evaluation, OncoBrain generated oncology treatment plans judged guideline-concordant, clinically acceptable, and easy to supervise. These findings support the potential of a carefully engineered AI reasoning platform to assist oncology treatment planning and justify prospective real-world evaluation in community settings. 

---
# RealRoute: Dynamic Query Routing System via Retrieve-then-Verify Paradigm 

**Authors**: Jiahe Liu, Qinkai Yu, Jingcheng Niu, Xi Zhu, Zirui He, Zhen Xiang, Fan Yang, Jinman Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.20860)  

**Abstract**: Despite the success of Retrieval-Augmented Generation (RAG) in grounding LLMs with external knowledge, its application over heterogeneous sources (e.g., private databases, global corpora, and APIs) remains a significant challenge. Existing approaches typically employ an LLM-as-a-Router to dispatch decomposed sub-queries to specific sources in a predictive manner. However, this "LLM-as-a-Router" strategy relies heavily on the semantic meaning of different data sources, often leading to routing errors when source boundaries are ambiguous. In this work, we introduce RealRoute System, a framework that shifts the paradigm from predictive routing to a robust Retrieve-then-Verify mechanism. RealRoute ensures \textit{evidence completeness through parallel, source-agnostic retrieval, followed by a dynamic verifier that cross-checks the results and synthesizes a factually grounded answer}. Our demonstration allows users to visualize the real-time "re-routing" process and inspect the verification chain across multiple knowledge silos. Experiments show that RealRoute significantly outperforms predictive baselines in the multi-hop Rag reasoning task. The RealRoute system is released as an open-source toolkit with a user-friendly web interface. The code is available at the URL: this https URL. 

---
# KGiRAG: An Iterative GraphRAG Approach for Responding Sensemaking Queries 

**Authors**: Isabela Iacob, Melisa Marian, Gheorghe Cosmin Silaghi  

**Link**: [PDF](https://arxiv.org/pdf/2604.20859)  

**Abstract**: Recent literature highlights the potential of graph-based approaches within large language model (LLM) retrieval-augmented generation (RAG) pipelines for answering queries of varying complexity, particularly those that fall outside the LLM's prior knowledge. However, LLMs are prone to hallucination and often face technical limitations in handling contexts large enough to ground complex queries effectively. To address these challenges, we propose a novel iterative, feedback-driven GraphRAG architecture that leverages response quality assessment to iteratively refine outputs until a sound, well-grounded response is produced. Evaluating our approach with queries from the HotPotQA dataset, we demonstrate that this iterative RAG strategy yields responses with higher semantic quality and improved relevance compared to a single-shot baseline. 

---
# ERA: Evidence-based Reliability Alignment for Honest Retrieval-Augmented Generation 

**Authors**: Sunguk Shin, Meeyoung Cha, Byung-Jun Lee, Sungwon Park  

**Link**: [PDF](https://arxiv.org/pdf/2604.20854)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds language models in factual evidence but introduces critical challenges regarding knowledge conflicts between internalized parameters and retrieved information. However, existing reliability methods, typically relying on scalar confidence, fail to explicitly distinguish between epistemic uncertainty and inherent data ambiguity in such hybrid scenarios. In this paper, we propose a new framework called ERA (Evidence-based Reliability Alignment) to enhance abstention behavior in RAG systems by shifting confidence estimation from scalar probabilities to explicit evidence distributions. Our method consists of two main components: (1) Contextual Evidence Quantification, which models internal and external knowledge as independent belief masses via the Dirichlet distribution, and (2) Quantifying Knowledge Conflict, which leverages Dempster-Shafer Theory (DST) to rigorously measure the geometric discordance between information sources. These components are used to disentangle epistemic uncertainty from aleatoric uncertainty and modulate the optimization objective based on detected conflicts. Experiments on standard benchmarks and a curated generalization dataset demonstrate that our approach significantly outperforms baselines, optimizing the trade-off between answer coverage and abstention with superior calibration. 

---
# DiagramBank: A Large-scale Dataset of Diagram Design Exemplars with Paper Metadata for Retrieval-Augmented Generation 

**Authors**: Tingwen Zhang, Ling Yue, Zhen Xu, Shaowu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2604.20857)  

**Abstract**: Recent advances in autonomous ``AI scientist'' systems have demonstrated the ability to automatically write scientific manuscripts and codes with execution. However, producing a publication-grade scientific diagram (e.g., teaser figure) is still a major bottleneck in the ``end-to-end'' paper generation process. For example, a teaser figure acts as a strategic visual interface and serves a different purpose than derivative data plots. It demands conceptual synthesis and planning to translate complex logic workflow into a compelling graphic that guides intuition and sparks curiosity. Existing AI scientist systems usually omit this component or fall back to an inferior alternative. To bridge this gap, we present DiagramBank, a large-scale dataset consisting of 89,422 schematic diagrams curated from existing top-tier scientific publications, designed for multimodal retrieval and exemplar-driven scientific figure generation. DiagramBank is developed through our automated curation pipeline that extracts figures and corresponding in-text references, and uses a CLIP-based filter to differentiate schematic diagrams from standard plots or natural images. Each instance is paired with rich context from abstract, caption, to figure-reference pairs, enabling information retrieval under different query granularities. We release DiagramBank in a ready-to-index format and provide a retrieval-augmented generation codebase to demonstrate exemplar-conditioned synthesis of teaser figures. DiagramBank is publicly available at this https URL with code at this https URL. 

---
# SPIRE: Structure-Preserving Interpretable Retrieval of Evidence 

**Authors**: Mike Rainey, Umut Acar, Muhammed Sezer  

**Link**: [PDF](https://arxiv.org/pdf/2604.20849)  

**Abstract**: Retrieval-augmented generation over semi-structured sources such as HTML is constrained by a mismatch between document structure and the flat, sequence-based interfaces of today's embedding and generative models. Retrieval pipelines often linearize documents into fixed-size chunks before indexing, which obscures section structure, lists, and tables, and makes it difficult to return small, citation-ready evidence without losing the surrounding context that makes it interpretable.
We present a structure-aware retrieval pipeline that operates over tree-structured documents. The core idea is to represent candidates as subdocuments: precise, addressable selections that preserve structural identity while deferring the choice of surrounding context. We define a small set of document primitives--paths and path sets, subdocument extraction by pruning, and two contextualization mechanisms. Global contextualization adds the non-local scaffolding needed to make a selection intelligible (e.g., titles, headers, list and table structure). Local contextualization expands a seed selection within its structural neighborhood to obtain a compact, context-rich view under a target budget. Building on these primitives, we describe an embedding-based candidate generator that indexes sentence-seeded subdocuments and a query-time, document-aware aggregation step that amortizes shared structural context. We then introduce a contextual filtering stage that re-scores retrieved candidates using locally contextualized views.
Across experiments on HTML question-answering benchmarks, we find that preserving structure while contextualizing selections yields higher-quality, more diverse citations under fixed budgets than strong passage-based baselines, while maintaining scalability. 

---
# AtomicRAG: Atom-Entity Graphs for Retrieval-Augmented Generation 

**Authors**: Yanning Hou, Duanyang Yuan, Sihang Zhou, Xiaoshu Chen, Ke Liang, Siwei Wang, Xinwang Liu, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20844)  

**Abstract**: Recent GraphRAG methods integrate graph structures into text indexing and retrieval, using knowledge graph triples to connect text chunks, thereby improving retrieval coverage and precision. However, we observe that treating text chunks as the basic unit of knowledge representation rigidly groups multiple atomic facts together, limiting the flexibility and adaptability needed to support diverse retrieval scenarios. Additionally, triple-based entity linking is sensitive to relation-extraction errors, which can lead to missing or incorrect reasoning paths and ultimately hurt retrieval accuracy. To address these issues, we propose the Atom-Entity Graph, a more precise and reliable architecture for knowledge representation and indexing. In our approach, knowledge is stored as knowledge atoms, namely individual, self-contained units of factual information, rather than coarse-grained text chunks. This allows knowledge elements to be flexibly reassembled without mutual interference, thereby enabling seamless alignment with diverse query perspectives. Edges between entities simply indicate whether a relationship exists. By combining personalized PageRank with relevance-based filtering, we maintain accurate entity connections and improve the reliability of reasoning. Theoretical analysis and experiments on five public benchmarks show that the proposed AtomicRAG algorithm outperforms strong RAG baselines in retrieval accuracy and reasoning robustness. Code: this https URL. 

---
# MATRAG: Multi-Agent Transparent Retrieval-Augmented Generation for Explainable Recommendations 

**Authors**: Sushant Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2604.20848)  

**Abstract**: Large Language Model (LLM)-based recommendation systems have demonstrated remarkable capabilities in understanding user preferences and generating personalized suggestions. However, existing approaches face critical challenges in transparency, knowledge grounding, and the ability to provide coherent explanations that foster user trust. We introduce MATRAG (Multi-Agent Transparent Retrieval-Augmented Generation), a novel framework that combined multi-agent collaboration with knowledge graph-augmented retrieval to deliver explainable recommendations. MATRAG employs four specialized agents: a User Modeling Agent that constructs dynamic preference profiles, an Item Analysis Agent that extracts semantic features from knowledge graphs, a Reasoning Agent that synthesizes collaborative and content-based signals, and an Explanation Agent that generates natural language justifications grounded in retrieved knowledge. Our framework incorporates a transparency scoring mechanism that quantifies explanation faithfulness and relevance. Extensive experiments on three benchmark datasets (Amazon Reviews, MovieLens-1M, and Yelp) demonstrate that MATRAG achieves state-of-the-art performance, improving recommendation accuracy by 12.7\% (Hit Rate) and 15.3\% (NDCG) over leading baselines, while human evaluation confirms that 87.4\% of generated explanations are rated as helpful and trustworthy by domain experts. Our work establishes new benchmarks for transparent, agentic recommendation systems and provides actionable insights for deploying LLM-based recommenders in production environments. 

---
# AITP: Traffic Accident Responsibility Allocation via Multimodal Large Language Models 

**Authors**: Zijin Zhou, Songan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20878)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved remarkable progress in Traffic Accident Detection (TAD) and Traffic Accident Understanding (TAU). However, existing studies mainly focus on describing and interpreting accident videos, leaving room for deeper causal reasoning and integration of legal knowledge. Traffic Accident Responsibility Allocation (TARA) is a more challenging task that requires multi-step reasoning grounded in traffic regulations. To address this, we introduce AITP (Artificial Intelligence Traffic Police), a multimodal large language model for responsibility reasoning and allocation. AITP enhances reasoning via a Multimodal Chain-of-Thought (MCoT) mechanism and integrates legal knowledge through Retrieval-Augmented Generation (RAG). We further present DecaTARA, a decathlon-style benchmark unifying ten interrelated traffic accident reasoning tasks with 67,941 annotated videos and 195,821 question-answer pairs. Extensive experiments show that AITP achieves state-of-the-art performance across responsibility allocation, TAD, and TAU tasks, establishing a new paradigm for reasoning-driven multimodal traffic analysis. 

---
# CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents 

**Authors**: Wenjie Fu, Xiaoting Qin, Jue Zhang, Qingwei Lin, Lukas Wutschitz, Robert Sim, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.21308)  

**Abstract**: Enterprise LLM agents can dramatically improve workplace productivity, but their core capability, retrieving and using internal context to act on a user's behalf, also creates new risks for sensitive information leakage. We introduce CI-Work, a Contextual Integrity (CI)-grounded benchmark that simulates enterprise workflows across five information-flow directions and evaluates whether agents can convey essential content while withholding sensitive context in dense retrieval settings. Our evaluation of frontier models reveals that privacy failures are prevalent (violation rates range from 15.8%-50.9%, with leakage reaching up to 26.7%) and uncovers a counterintuitive trade-off critical for industrial deployment: higher task utility often correlates with increased privacy violations. Moreover, the massive scale of enterprise data and potential user behavior further amplify this vulnerability. Simply increasing model size or reasoning depth fails to address the problem. We conclude that safeguarding enterprise workflows requires a paradigm shift, moving beyond model-centric scaling toward context-centric architectures. 

---
# The Root Theorem of Context Engineering 

**Authors**: Borja Odriozola Schick  

**Link**: [PDF](https://arxiv.org/pdf/2604.20874)  

**Abstract**: Every system that maintains a large language model conversation beyond a single session faces two inescapable constraints: the context window is finite, and information quality degrades with accumulated volume. We formalize these constraints as axioms and derive a single governing principle -- the Root Theorem of Context Engineering: \emph{maximize signal-to-token ratio within bounded, lossy channels.} From this principle, we derive five consequences without additional assumptions: (1)~a quality function $F(P)$ that degrades monotonically with injected token volume, independent of window size; (2)~the independence of signal and token count as optimization variables; (3)~a necessary gate mechanism triggered by fidelity thresholds, not capacity limits; (4)~the inevitability of homeostatic persistence -- accumulate, compress, rewrite, shed -- as the only architecture that sustains understanding indefinitely; and (5)~the self-referential property that the compression mechanism operates inside the channel it compresses, requiring an external verification gate. We show that append-only systems necessarily exceed their effective window in finite time, that retrieval-augmented generation solves search but not continuity, and that the theorem's constraint structure converges with biological memory architecture through independent derivation from shared principles. Engineering proof is provided through a 60+-session persistent architecture demonstrating stable memory footprint under continuous operation -- the divergence prediction made concrete. The Root Theorem establishes context engineering as an information-theoretic discipline with formal foundations, distinct from prompt engineering in both scope and method. Shannon solved point-to-point transmission. Context engineering solves continuity. 

---
