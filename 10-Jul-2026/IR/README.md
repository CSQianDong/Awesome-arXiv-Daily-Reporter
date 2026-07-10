# Improving Ad-hoc Search Effectiveness for Conversational Information Retrieval via Model Merging 

**Authors**: Ahmed Rayane Kebir, Jose G. Moreno, Lynda Tamine  

**Link**: [PDF](https://arxiv.org/pdf/2607.08540)  

**Abstract**: Conversational information retrieval is challenging since it requires the consideration of the conversation history which potentially gives rise to topic shifts and coreference resolution across previous turns. To address these challenges, previous work mainly rely on traditional fine-tuning of ad-hoc retrievers on conversational datasets or extrapolates their generalizability through multi-tasking. However, this mainstream approach is costly - since it requires model re-training - and exhibits catastrophic forgetting, where the model loses its foundational ad-hoc retrieval performance. In this paper, we fill this gap by introducing model merging as a training-free strategy enabling the design of a single retrieval model that operates across both ad-hoc and conversational settings with no additional fine-tuning. We conduct experiments using linear and non-linear parameter-wise merging strategies - namely Model Soup and Slerp - on standard ad-hoc search and conversational retrieval datasets. Our results demonstrate that model merging significantly enhances the ad-hoc search capabilities of conversational retrievers while improving generalizability across task-specific datasets, achieving up to 15% higher NDCG@3 under zero-shot conditions. 

---
# Log-Insight: Automating Microservice Incident Diagnosis via Neuro-Symbolic Log Analysis 

**Authors**: Carlos Garcia-Hernandez, Aymane Abdali, Guangyu Wu, Mingxue Wang, Fei Shen, Zhaoyu Pang, Yanbin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.08529)  

**Abstract**: Diagnosing production incidents in large-scale microservice systems is time-critical for Site Reliability Engineers (SREs). A single 30-minute incident window in our deployment can generate over two million log lines--approximately 1.2 billion characters, far exceeding standard LLM context windows--making direct LLM-based Root Cause Analysis (RCA) infeasible. Existing approaches leave gaps: template-based parsers lack semantic anomaly reasoning, deep-learning detectors emit black-box binary signals, and LLM pipelines suffer context overflow and domain hallucination on raw telemetry.
We present Log-Insight, an automated incident-diagnosis system deployed in production at Huawei. The core design principle automates the SRE's manual triage workflow: symbolic stages replicate the structured investigation a skilled SRE would perform--sampling, schema understanding, pattern clustering, and statistical anomaly ranking. This hands the LLM a compact, pre-ranked evidence dossier to synthesise into a hypothesis report. Our six-stage pipeline reduces millions of raw events by 1,000-7,000x while preserving statistically significant failure signals.
Evaluated on 11 historical production incidents (110 runs, SRE-validated ground truth), Log-Insight achieves MRR = 0.790, returning the correct root cause within the top-3 hypotheses in over 90% of runs in under a minute of latency. We report systematic failure modes, active mitigations, and open research directions. The Forensic Evidence section--listing exact log templates and skew statistics--was consistently identified by operators as a key adoption factor, shifting the system's perceived role from opaque oracle to investigative assistant. 

---
# Conversational Retrieval and On-the-Fly Knowledge Modeling of Historical Penitentiary Repression Records 

**Authors**: Paula Font Solà, Adrià Molina Rodríguez, Josep Lladós  

**Link**: [PDF](https://arxiv.org/pdf/2607.08459)  

**Abstract**: Recent developments in digital libraries increasingly favor conversational and natural language access to information through Retrieval-Augmented Generation (RAG). Although these approaches are effective for extractive tasks grounded in individual records, they remain limited in their ability to interpret document collections holistically and to incorporate expert knowledge dynamically. In this article, we present a document analysis system designed for the management of historical digital libraries that supports on-the-fly knowledge modeling. The system is equipped with the capability to store facts produced either by expert archivists or derived from document retrieval processes within a graph-based structure. Through continuous professional interaction, the system can retrieve information not only from primary sources such as documents, but also from previously modeled knowledge, with the graph-based index acting as a memory for the language model to access. This enables increasingly complex queries involving long-term dependencies across documents, link discovery, and the integration of expert knowledge that may not be explicitly present in the original sources. As a result, the proposed approach facilitates the generation of richer and more comprehensive information. 

---
# H3D: Benchmarking Unsupervised Text Hashing for Fine-Grained Document Deduplication 

**Authors**: Qianren Mao, Jiaxun Lyu, Junnan Liu, Zhijun Chen, Jingzheng Li, Hanwen Hao, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.08382)  

**Abstract**: Document hashing provides compact representations for efficient similarity search and document deduplication, but existing studies rarely compare hashing pipelines under a unified protocol for fine-grained scientific documents. H3D is an unsupervised text hashing benchmark for fine-grained document deduplication. It evaluates representative unsupervised non-learning hashing approaches (MinHash, SimHash, Winnowing, FuzzyHash, FlyHash) together with semantic-sensitive methods built from frozen BGE embeddings and two quantization strategies (BGE-BIHash and BGE-LSHash). The non-learning methods generate hash fingerprints through manually designed mathematical rules without training or labeled similarity pairs, which distinguishes them from neural semantic hashing models. We benchmark all methods on CSFCube and RELISH, two datasets that provide complementary evaluation settings: facet-level analysis for scientific-document similarity and larger-scale split-level evaluation for biomedical similarity search. H3D jointly reports ranking quality (MAP, NDCG@20), efficiency, and robustness under controlled text compression. The results show a consistent trade-off: lexical and structural fingerprints are competitive for near-duplicate matching, while semantic-sensitive representations better preserve similarity under content rewriting, at higher computational cost. We further analyze when different similarity measures become rank-equivalent for specific hash representations, improving the interpretability and reproducibility of method comparisons. 

---
# DaV-Gen: End-to-End Generative Retrieval via Draft-and-Verify 

**Authors**: Meng Zhao, Chunmei Liu, Qinyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.08365)  

**Abstract**: Mainstream industrial information retrieval systems (e.g., search and recommendation) are usually built upon Multi-Stage Cascade Architectures (MCAs), which balance effectiveness and efficiency through a coarse-to-fine ``retrieval-ranking'' pipeline. However, the optimization objectives across different stages are substantially inconsistent, propagating or even amplifying the early-stage errors that ultimately degrade the quality of final results. While emerging end-to-end generative models offer a potential solution by unifying the pipeline, their online serving performance is severely hindered by the auto-regressive process inherited from the standard decoder-only structure. To bridge this gap, we introduce \textbf{DaV-Gen}, a novel unified solution designed to fundamentally refactor the paradigm for both search and recommendation via a ``Draft-and-Verify'' mechanism. Inspired by the process used by speculative decoding, our framework redesigns the generation task into two synergistic operations within a single model. During training, the model is concurrently optimized for both candidate drafting and fine-grained verification. This is achieved by a composite loss function that jointly trains the model on two distinct but related objectives: 1) a contrastive loss that structures the embedding space for efficient drafting, and 2) a fusion loss that combines generative likelihood with vector similarity to produce a superior verification score. This integrated training strategy equips the model with dual capabilities. At inference time, it first performs highly efficient vector-based drafting to generate a candidate set, and then verifies these candidates using the more powerful fused scoring function, thereby achieving both the speed of sparse drafting and the precision of advanced generative models within a unified, end-to-end architecture. 

---
# BACH: A Bayesian Admixture of Contrastive Heads for Multi-Interest Two-Tower Retrieval 

**Authors**: Quoc Phong Nguyen, Paul Albert, Long Vuong, Vuong Le, Julien Monteil  

**Link**: [PDF](https://arxiv.org/pdf/2607.08107)  

**Abstract**: Two-tower retrievers compress each user into a single embedding, limiting their ability to serve diverse interests. Multi-interest models give each user several heads scored by a maximum inner product, but their hard-routing training under-utilizes heads (routing collapse) and gives no per-user estimate of how much each interest matters for serving. We present \textbf{BACH} (\emph{Bayesian Admixture of Contrastive Heads}), which casts multi-interest two-tower retrieval as a per-user mixture over the heads, fit by variational inference. The soft mixture trains every head (mitigating collapse), produces a per-user weighting of the interests that is reused at serving, and admits a shared global-codebook variant with precomputable retrieval. On three large-scale benchmarks, MovieLens-20M, Taobao, and Netflix, BACH improves top-of-ranking retrieval over hard-routing multi-interest and single-vector baselines at every head count; we further find that scoring every candidate by its best head, consistent with serving, outperforms the usual target-routed training, and that BACH improves further still. 

---
# ProjAgent: Procedural Similarity Retrieval for Repository-Level Code Generation 

**Authors**: QiHong Chen, Aaron Imani, Iftekhar Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2607.08691)  

**Abstract**: Repository-level code generation requires implementing target functions while accounting for complex cross-file dependencies and project-specific conventions. Existing retrieval methods predominantly rely on lexical, structural, or semantic similarity, often overlooking repository functions that implement similar procedural logic despite differing in identifiers or application domains. We propose ProjAgent, a repository-level code generation system that introduces procedural similarity as an explicit retrieval signal. ProjAgent decomposes the target function into intermediate reasoning steps and employs an agentic workflow to retrieve repository functions that exhibit similar procedural behavior at each step. The retrieved procedural context is integrated with conventional semantic retrieval to construct a richer repository context for code generation. ProjAgent further incorporates a conservative static-analysis feedback loop that iteratively repairs generated code using compiler and static-analysis feedback. Evaluated on REPOCOD, ProjAgent achieves 41.14% Pass@1, outperforming existing retrieval-based baselines. These results demonstrate that procedural similarity is an effective and previously unexplored retrieval dimension for repository-level code generation. 

---
# ICDAR 2026 HIPE-OCRepair Competition on LLM-Assisted OCR Post-Correction for Historical Documents 

**Authors**: Maud Ehrmann, Emanuela Boros, Juri Opitz, Andrianos Michail, Florian Wagner, Simon Clematide  

**Link**: [PDF](https://arxiv.org/pdf/2607.08143)  

**Abstract**: We present the results of HIPE-OCRepair-2026, an ICDAR competition on LLM-assisted OCR post-correction of historical documents. OCR post-correction remains a long-standing challenge in digital heritage: large-scale collections of digitized documents are affected by legacy OCR errors, while re-digitization at scale remains impractical. Large language models (LLMs) offers a major opportunity to revisit this challenge, yet their effectiveness across languages, document types, and noise conditions - and their tendency to hallucinate - remains insufficiently understood. HIPE-OCRepair-2026 pursues two objectives: (i) to evaluate the capabilities of modern OCR post-correction systems, and (ii) to provide a reproducible evaluation framework anchored in the HIPE-OCRepair-2026 dataset, a harmonized multilingual resource consolidating existing and newly curated historical datasets. Participants were tasked with correcting noisy OCR transcripts from historical newspapers and printed works in English, French, and German (17th-20th century), working at the level of coherent transcription units (paragraphs or articles) without access to source images. The evaluation adopts a retrieval-oriented rather than diplomatic scoring approach, reflecting the practical use case of search and access over digitized collections. Four teams submitted systems ranging from zero-shot prompting to continued pre-training and fine-tuning, offering insights into the merits of different adaptation strategies. Results show that modern LLM-assisted systems can significantly improve OCR quality, but performance varies across datasets, languages, and noise levels. Over-correction on low-noise inputs emerges as a recurring challenge, highlighting the importance of evaluation beyond character error reduction. The dataset, scorer, and evaluation pipeline are publicly released to support future research. 

---
# Beware What You Autocomplete: Forensic Attribution of Backdoored Code Completions 

**Authors**: Anjun Gao, Yueyang Quan, Zhuqing Liu, Minghong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2607.08011)  

**Abstract**: Large language models have enabled powerful code completion systems that assist developers by predicting subsequent lines of code. However, these models remain vulnerable to backdoor attacks, where malicious fine-tuning data covertly implants unsafe behaviors. Despite advances in defensive techniques, adaptive and sophisticated backdoor attacks still evade detection and mitigation. We present CodeTracer, a forensic framework that traces malicious code completions back to the backdoor fine-tuning data responsible for them. Operating under realistic post-deployment constraints, CodeTracer relies solely on the fine-tuning corpus and the reported miscompletion event. It extracts a structured behavioral fingerprint from the compromised output, narrows the search to semantically relevant code samples, and employs LLM-based reasoning to attribute unsafe logic to specific backdoor data. Extensive evaluations across three representative vulnerability cases and ten backdoor attacks, along with sixteen competitive baselines, demonstrate that CodeTracer consistently achieves high forensic accuracy, low false identification rates, and strong robustness against adaptive attacks. 

---
# Who Broke the System? Failure Localization in LLM-Based Multi-Agent Systems 

**Authors**: Yufei Xia, Anjun Gao, Yueyang Quan, Zhuqing Liu, Minghong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2607.07989)  

**Abstract**: Large language model (LLM) based multi-agent systems enable complex problem solving through coordinated reasoning and action, but their distributed structure also introduces new challenges in diagnosing system-level failures. When an execution fails, identifying which agent is responsible and at what point the trajectory first becomes irreversibly misdirected is difficult due to long-horizon interactions and tightly coupled agent behaviors. In this paper, we study the problem of failure localization in LLM-based multi-agent systems and present AgentLocate, a framework that attributes failures to both a specific agent and the earliest decisive step. AgentLocate combines an LLM-based judging mechanism with multi-perspective verification by independent evaluators, whose assessments are aggregated using a confidence-aware strategy. The resulting feedback is further used to adapt the judge through lightweight fine-tuning, improving attribution quality. We evaluate AgentLocate on two complementary benchmarks covering diverse tasks, agent configurations, and trajectory lengths. Experimental results show that AgentLocate consistently outperforms existing failure localization methods in identifying both responsible agents and failure steps, while remaining efficient in terms of token usage and running time. 

---
