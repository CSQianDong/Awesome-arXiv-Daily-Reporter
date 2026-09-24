# Beyond a Scalar: Distributional Serving Interfaces for Watch-Time Prediction 

**Authors**: Xuan Liu, Jingbin Qian, Zhanyu Liu, Hefeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.28383)  

**Abstract**: Watch time is the primary engagement signal in short video feeds, and its prediction directly affects ranking and exposure. Existing methods improve watch time prediction by correcting duration bias or modeling richer distributions, but most expose only an expected or debiased watch time at serving time. Even when video duration is available to later models, the interface gives only one estimate of watch time and no probabilities for completion, overplay, or other regions relevant to downstream tasks. To address this limitation, we propose the Distributional Serving Interface (DSI), which has a distribution provider, a compact, low-dimensional summary, and lightweight readouts tailored to each task. The provider learns a joint distribution over four watch states derived from watch ratio and their event times; rules based on video duration remove incompatible combinations, while a restoration loss preserves accuracy in seconds. The summary reduces this distribution to a small set of event probabilities, time scales relative to duration, and uncertainty statistics. After training the provider, we fix its parameters and train value and ranking readouts that combine the summary with raw context. Across KuaiRec, KuaiRand-1K, and WeChat21, the complete DSI system achieves the lowest MAE on all three datasets, beating the strongest result among nine baselines by 1.9% to 8.5%, and achieves the best XAUC on two. It also leads retrieval metrics that account for video duration when complete systems are compared. With matched readouts held constant, the summary retains information relevant to each task beyond a predicted mean paired with video duration. Using the same lightweight linear heads for each new target, it also performs best on two new watch-time targets and improves a separately logged engagement target, while a randomly initialized provider does not reproduce this gain. 

---
# Dual-Hypergraph Indexing: Bridging Knowledge Islands for Multi-Hop Reasoning in Retrieval-Augmented Generation 

**Authors**: Qi Sun, Xingliang Hou, Caibo Li, Yijia Zhang, Qiang Li, Yu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.28108)  

**Abstract**: While hypergraph-based Retrieval-Augmented Generation (RAG) effectively captures higher-order multi-entity correlations, existing paradigms treat extracted hyperedges as isolated factual assertions. This structural fragmentation engenders rigid "knowledge islands" that bottleneck multi-hop causal inference, temporal tracking, and narrative synthesis. To systematically address these challenges, we introduce Dual-Hypergraph Indexing (DHI), a hierarchical representation framework that elevates discrete facts into structured analytical insights. DHI couples a foundational entity-relation factual hypergraph ($H_K$) with an elevated deep-insight hypergraph ($H_D$) via a dual-pathway aggregation algorithm. Specifically, DHI employs: (1) importance-driven hub aggregation via 5-metric topological profiling and adaptive thresholding to capture spatial semantic clusters; and (2) temporal chunk-chain progressive aggregation via sliding-window greedy exploration to track chronological evolutions. Across five benchmarks, DHI achieves state-of-the-art performance, boosting logical coherence by +1.53 on the multidisciplinary Mix benchmark and scoring 85.78\% on complex medical pathology reasoning tasks. DHI provides a robust architecture for next-generation multi-hop RAG. 

---
# A Flexible Recommendation System for Individuals and Groups 

**Authors**: Yacine Mokhtari, Grégory Smits  

**Link**: [PDF](https://arxiv.org/pdf/2609.27998)  

**Abstract**: Group recommender systems typically rely on either aggregating individual preferences or treating groups as distinct meta-users. However, these methods often suffer from static aggregation strategies or data sparsity issues within group histories. This paper introduces a novel approach, that relies on a GNN-based architecture to learn a dual representation of each user's preferences, capturing their behavior as an independent individual from one side and as a member of a collective from the other side. By performing a differential analysis of these individual and group-oriented preferences, our system then determines the behavioral profile of each user when joining a group. Finally, specific preference aggregation strategies are defined to cope with the behavioral profiles of the users composing a group. Consequently, the system is equally capable of delivering precise recommendations to individuals and to arbitrary groups, effectively unifying the two traditional paradigms of recommendation. Experiments on synthetic data simulating diverse group settings and behaviors confirm the flexibility and relevance of the proposed approach compared to state-of-the-art methods. 

---
# The Recall Ceiling of LLM Recommendation Reranking 

**Authors**: Zhaohui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27953)  

**Abstract**: Some LLM-based recommendation rerankers are evaluated under an oracle protocol that guarantees the ground-truth item is present in the scored set, either by injecting it into the candidate list or by scoring it against sampled negatives. Across three primary Amazon datasets, we show that this protocol overestimates realistic NDCG@10 by 92--95%. The cause is a recall ceiling: realistic retrieval covers only 2--19% of relevant items at $K=100$ across eight datasets in three domains, imposing a deterministic upper bound on any closed-candidate reranker's top-$k$ NDCG. Under leave-one-out evaluation, $\mathbb{E}[\mathrm{NDCG}@k] \leq \mathrm{Recall}@|W_\pi|$, where $W_\pi$ is the reranker's candidate window.
Under realistic retrieval, none of the tested optimisation strategies significantly improves over the collaborative-filtering baseline on our primary Amazon datasets. These strategies include prompt engineering, model scaling over a 168$\times$ parameter range, sequential models, supervised neural rerankers, LoRA fine-tuning, hybrid retrieval, score-aware prompting, and LLM+CF fusion. Text-aware retrieval increases recall on one dataset but does not improve end-to-end NDCG, while providing upstream CF scores mainly makes the LLM reproduce the CF order. We therefore propose the Recall-Aware Evaluation Protocol (RAEP): first classify the retrieval-recall regime, then evaluate reranking where the ceiling permits meaningful differentiation. In the low-recall regimes measured here, improving retrieval is more consequential than increasing reranker sophistication; this ordering need not hold in production systems with higher recall, richer features, or online feedback. 

---
# Query Implied Generative Engine Optimization 

**Authors**: Shilpa Ramakrishna, William B. Andreopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2609.27845)  

**Abstract**: The landscape of search has changed drastically with how people look for information online. Traditional search engines are being replaced by Generative Search Engines (GSEs), which use Large Language Models (LLMs) to generate natural language responses to user queries. For content creators, visibility is no longer solely determined by ranking in search results but by being cited within generated responses. But Generative Search Engines are black-boxes, leading to the emergence of Generative Engine Optimization (GEO), a set of techniques aimed at improving content visibility in generative search settings. Most existing approaches rely on the explicit queries or query derived signals to align content to better suit user needs. We propose Query Implied Generative Engine Optimization (QI-GEO) to infers user intent directly from the document. Our approach approximates document's intent space and identifies content that may be missing yet relevant to answer potential user queries. Evaluation on GEO-Bench and Extended GEO-Bench demonstrated improvements across objective and subjective metrics. QI-GEO improved objective scores by up to 15.9% and subjective scores by up to 17.6%, while yielding nearly twice as many citation gains as citation losses. These results suggest that document-derived approximations of user intents can improve visibility without relying on explicit query inputs. 

---
# EidosDoc: Implicit Structure Encoding for Cost-Effective Semi-Structured Document QA 

**Authors**: Teng Lin, Yuyu Luo, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27784)  

**Abstract**: Semi-structured documents are ubiquitous in scientific reports, financial statements, and technical manuals. Question answering over such documents requires simultaneous understanding of text, tables, charts, and complex hierarchical layouts. Existing methods either rely on repeatedly calling large language models for structure parsing and retrieval, leading to high cost and large latency, or they flatten the document and lose layout and hierarchy information, sacrificing answer accuracy. To address this, we propose EidosDoc, a novel system that achieves state-of-the-art accuracy with minimal computational expense. Our approach introduces three core innovations. (1) An Implicit Structure Encoder trained via contrastive learning and a structure consistency loss. This module jointly embeds hierarchical relationships, spatial positions, and textual content into a dense vector space, capturing document structure holistically without the need for manually defined and error-prone constructions. (2) A Hybrid Retrieval Pipeline that leverages BM25, layout fingerprints, and a lightweight cross-encoder to perform high-precision retrieval entirely without invoking an LLM, drastically reducing cost and latency. (3) A Dynamic Evidence Expansion mechanism that adaptively retrieves spatially adjacent and structurally related evidence, overcoming the evidence omission common in fixed-path retrieval methods. We evaluate EidosDoc on four benchmarks, and comprehensive evaluations show that EidosDoc achieves a new state-of-the-art accuracy on the four benchmarks. Crucially, it does so with a 50 times reduction in cost and 4 times lower latency compared to the previous state-of-the-art Method. These results demonstrate that EidosDoc establishes a new optimal trade-off among accuracy, cost, and speed, offering a practical and scalable path for accurate semi-structured document analysis. 

---
# Test-Time Adaptation with Query-Dependent Residuals for Visual Document Retrieval 

**Authors**: Zeliang Li, Xiaofen Xing, Kailing Guo, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27688)  

**Abstract**: Visual document retrieval (VDR) systems depend on page embeddings computed before deployment, which makes adaptation difficult when encoder parameters or corpus re-encoding are unavailable. Rerankers provide useful relevance signals, but conventional reranking applies them only to selected queries and candidate pages. We introduce Q-REACT, a query-side test-time adaptation method that converts limited reranker feedback into reusable retrieval improvements. Q-REACT learns a shared low-rank transformation that produces query-dependent residuals, combines adapted query scores with document-level context, and distills reranker preferences with a student distribution normalized over the complete task-specific page index. This design lets unscored pages compete through cached embeddings while keeping the encoders and page index fixed. Across eight ViDoRe V3 tasks and five open-weight and proprietary backbones, Q-REACT improves average retrieval over evaluated baselines at sparse and full-coverage budgets, transfers to held-out queries and tasks, and adds little inference overhead. The results show that finite reranker feedback can be amortized across a query collection without retraining or rebuilding the retriever. 

---
# BoundaryMORPH: Budgeted Reranking via Active Set Selection for Diffuse Retrieval 

**Authors**: Eylon Caplan, Shamik Roy, Shib Sankar Dasgupta, Yingfan Wang, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2609.27213)  

**Abstract**: Open-ended queries in modern Retrieval-Augmented Generation (RAG) are increasingly "diffuse," requiring a large set of documents to be assembled into a finite LLM context window. To ensure retrieval quality, systems use fast dual-encoders and more expensive cross-encoders (CEs) to score candidates. However, the CE budget $B$ is strictly bounded by latency and is often smaller than the context window capacity $k$. This mismatch makes standard reranking structurally flawed: it wastes compute verifying obvious top candidates while ignoring relevant documents further down the initial ranking. To address this, we introduce BoundaryMORPH, a novel algorithm that allocates CE budget specifically for the LLM's context capacity $k$. Using a Gaussian Process, BoundaryMORPH treats the initial dual-encoder ranking as a structural prior and intelligently spends CE calls on resolving top-$k$ set membership at the boundary, rather than seeking a single most-relevant document. Information from each CE call propagates to unscored documents, maximizing the utility of the budget. We demonstrate that BoundaryMORPH achieves state-of-the-art set retrieval quality across multiple models and datasets with open-ended queries ($+5.4$ nCG@100 over the strongest baseline). 

---
# When LLM-Based User Profiling Adds Value in Production Streaming Recommendation 

**Authors**: Milad Sabouri, Neeraj Sharma, Sardar Hamidian, Shaghayegh Agah  

**Link**: [PDF](https://arxiv.org/pdf/2609.27183)  

**Abstract**: Personalized recommendation depends critically on how user representations are constructed from historical behavior. Two paradigms have emerged for constructing semantic user profiles in content-based recommendation. First, aggregate methods derive user representations as numerical aggregates of semantic item embeddings. Second, LLM-based methods generate natural-language summaries of user preferences and encode them through a text encoder. Each paradigm can be combined with temporal disentanglement of recent versus historical behavior. LLM-based profile generation is significantly more expensive than aggregate approaches, raising the question of when this additional cost is justified. We present a systematic comparison of four semantic user-profiling strategies, factorially crossed across representation type and temporal handling, evaluated on a real-world production dataset. The comparison reveals how these strategies differ across user behavior types, across both accuracy and beyond-accuracy dimensions of recommendation quality, and across the temporal-window setting that governs the disentanglement. 

---
# Tie Handling Is Part of the Evaluation Protocol: An Order-Invariance Audit for Tie-Heavy Recommender Scores 

**Authors**: Chengkun Guo, Han Chen, Yilin Zhu, Yingrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.26977)  

**Abstract**: Offline top-k evaluation often ranks one held-out relevant item together with sampled negatives. When several candidates receive exactly the same score, the tie-breaking rule becomes part of the ranking. A common implementation stores the relevant item first and then applies a stable sort, which preserves input order among equal scores; the relevant item therefore wins every tie. We call an evaluator row-order invariant when permuting the input candidates without changing their identities, labels, or scores leaves the final ranking unchanged. We audit this property by holding candidates and scores fixed and changing only the tie-breaking rule. On 30,000 Amazon Beauty & Personal Care rows, NDCG@10 for a rating-weighted attribute-overlap score is 0.85 under input-order tie-breaking. A deterministic hash tie-break based on user and item IDs lowers it to 0.17. The exact expectation under uniform random tie-breaking closely matches the mean over 100 independent hash seeds, while a residualized attribute score with few exact ties is nearly unchanged. MovieLens Tag Genome shows the same pattern for an attribute-overlap score, whereas item popularity is nearly unchanged. We derive expected Hit Rate and NDCG at cutoff k when the relevant item is randomly ordered among candidates with the same score, and we provide a practical reporting checklist. The same issue can occur in sampled or full-catalog evaluation whenever exact ties affect top-k membership or rank. 

---
# Calibrating Reproduced Claims in Recommender Systems 

**Authors**: Alan Said  

**Link**: [PDF](https://arxiv.org/pdf/2609.26975)  

**Abstract**: Reproduction studies can produce mixed outcomes. Reported values may differ while the ordering of the compared methods remains the same, a result may hold only under some experimental conditions, or a released implementation may fail to reproduce a result that the model can still reach. The terms repeatability, reproducibility, and replicability describe how a follow-up study relates to the original experiment, but not which parts of the original claim are supported by the new results. We introduce \emph{claim calibration} as a way of stating the strongest claim supported by a follow-up study, together with the conditions under which it holds and the parts that remain untested. We apply this perspective to five original--follow-up paper pairs from recommender-systems research. The cases show that agreement in numerical values, method rankings, statistical results, and overall conclusions does not always coincide, and that follow-up studies often support only part of the original claim. Based on these observations, we propose a Claim Evidence Profile for reporting the original claim, its scope, the reproduction target, the reported results, the calibrated claim, and the parts of the original claim that remain unresolved. 

---
# Distilling Lexical Product Associations into Deep Transformers: An Extreme Multi-Label Approach for Natural Language E-Commerce Search 

**Authors**: Sunnidhya Roy, Samarpita Bhaumik  

**Link**: [PDF](https://arxiv.org/pdf/2609.26921)  

**Abstract**: Traditional e-commerce search platforms rely heavily on inverted indices and token-level lexical matching algorithms (e.g., BM25 and TF-IDF), which frequently fail on conversational, intent-driven, or paraphrased user queries -- the classic vocabulary mismatch problem. We formulate conversational product recommendation as an Extreme Multi-Label Classification (XMLC) problem over an e-commerce catalog of N = 54,000 products spanning 27 balanced retail categories from the Amazon Reviews '23 benchmark. Using a pre-trained DistilBERT transformer encoder, we distill dense item-to-item similarity topologies (generated via TF-IDF cosine similarity over cumulative metadata with K = 50 nearest neighbours) into a deep contextual representation via a pseudo-label knowledge distillation framework.
Evaluated on an exact 85/15 train/validation split (8,089 held-out products across C = 53,923 output classes) with strict self-exclusion enforced, the DistilBERT neural student achieves P@1 = 93.15%, P@5 = 90.08%, NDCG@10 = 0.8845, and MRR@10 = 0.9545, closely recovering the empirical ceiling established by the corrected TF-IDF teacher (P@1 = 98.10%, NDCG@10 = 0.9419, MRR@10 = 0.9882). Furthermore, a qualitative benchmark across ten structured natural language query archetypes -- encompassing situational, cross-category, paraphrased, and negative-constraint queries -- demonstrates that the transformer student generalises substantially beyond keyword matching, successfully resolving implicit user intent where lexical models fail completely. Finally, we analyse the architectural and memory scalability trade-offs of extreme classification projection layers at industrial catalog scale (> 10^6 items) and present a concrete deployment trajectory toward Dual-Encoder (Two-Tower) vector search. Code: this https URL. 

---
# ItColBERT: An Italian-Specialised Late-Interaction Retriever 

**Authors**: Enrico Nello  

**Link**: [PDF](https://arxiv.org/pdf/2609.26856)  

**Abstract**: Neural information retrieval for Italian is served almost entirely by multilingual models. Several multi-vector (late-interaction) retrievers include Italian among dozens of languages, and several strong Italian dense embedders exist, but as of August 2026 no late-interaction retriever specialised on Italian had been released. We present ItColBERT, a 135M-parameter Italian ColBERT trained with PyLate following the ColBERT-Zero recipe: initialise from a checkpoint that already retrieves, then apply supervised contrastive training followed by single-teacher distillation, for a total of roughly 14.5 GPU-hours on one RTX 3090. Across four Italian retrieval benchmarks it outperforms every general-purpose late-interaction baseline we tested except one (mLateOn), at 2-4.4x fewer parameters than every baseline but one of comparable size. Our principal empirical finding is methodological and partly negative. On the only cleanly out-of-domain benchmark (MLDR-it), an inference-time chunking recipe applied to an unchanged checkpoint yields +0.0602 nDCG@10 (p = 0.0225), a larger effect than anything two further rounds of training produced. Self-mined hard negatives and native 1024-token training were both evaluated against pre-registered decision gates and both failed. We report every comparison with paired bootstrap tests against an empirically measured noise floor of 0.0030 nDCG@10, and we release the weights, the training and evaluation code, and the complete experimental record including the rejected rounds. 

---
# MultiVENT-Raw: A Benchmark for Retrieval and Reasoning over Raw Videos 

**Authors**: Reno Kriz, David Etter, Alexander Martin, Cameron Carpenter, Debashish Chakraborty, Hannah Recknor, Reihaneh Iranmanesh, Matthew Maciejewski, Kenton Murray, Eugene Yang, Benjamin Van Durme, Aaron Steven White, Andrew Yates, William Walden  

**Link**: [PDF](https://arxiv.org/pdf/2609.28437)  

**Abstract**: Online information is increasingly consumed in video format. Much of this comes in the form of *raw video*: continuous footage taken on a cell phone, with a hand-held camera, or via CCTV, which is then directly uploaded to social media platforms and content sharing services. Whereas professional or even amateur-edited footage tends to feature scripted speech, chyrons, graphics, and metadata that help contextualize its subject matter, raw video typically contains none of these things, making it a much more challenging medium for information retrieval and machine understanding. To facilitate progress in this domain, we release MultiVENT-Raw, a multilingual collection of nearly 120,000 primarily raw videos (over 5,300 total hours), paired with 130 events and 222 event-centric queries, along with human-annotated video relevance judgments and human-extracted key facts for relevant videos. MultiVENT-Raw supports both a retrieval task---to identify videos in the collection relevant to a query event---and a generation task---to summarize event-related videos into a coherent report for a target user. We benchmark strong baselines on MultiVENT-Raw, showing both tasks to be challenging even for some of the latest multimodal models. 

---
# Entangle: Uncovering Collaboration in the GitHub Quantum Software Ecosystem 

**Authors**: Angel Luis Lara-Martín, Ricardo Pérez-Castillo  

**Link**: [PDF](https://arxiv.org/pdf/2609.28349)  

**Abstract**: Quantum computing is moving from research laboratories towards early commercialization and broader socio-technical adoption, supported by sustained hardware progress and a rapidly expanding open-source software ecosystem. This momentum is especially visible on GitHub, where many quantum and hybrid software projects coexist around frameworks such as Qiskit, Cirq, PennyLane and Amazon Braket. However, this ecosystem remains fragmented, making it difficult to understand who shapes quantum software, where expertise is concentrated, how collaboration flows across organizations and disciplines, and which actors connect otherwise separated communities. This paper presents Entangle, a data-driven analysis of the open-source quantum computing ecosystem on GitHub. Starting from 71 domain keywords, Entangle identifies more than 1,500 quantum repositories, 27,000 contributors and 400 organizations, revealing an ecosystem strongly organized around four leading industrial vendors, but also supported by 2,387 contributors who connect projects, organizations and domains. These findings provide practical evidence for responsible quantum innovation by making visible patterns of influence, dependency, collaboration and knowledge transfer. They also offer actionable indicators for strategic decisions on investment, hiring, partnerships, ecosystem stewardship and capacity building. More broadly, Entangle shows how open-source intelligence can support a more transparent, measurable and governable quantum software ecosystem, helping align technical development with responsible innovation, public--private coordination and long-term sustainability. 

---
# Evaluating Open-Weight LLMs for Turkish Domain Documents Under Retrieval and Hardware Constraints 

**Authors**: Imtiaz Ul Hassan, Öykü Akbulut, Onur Kaya, Ardhendu Behera, Swagat Kumar, Peter Matthew, Yonghuai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.28007)  

**Abstract**: Most Turkish-capable large language models (LLMs) are evaluated using general-purpose benchmarks rather than long, structurally complex domain documents. This paper evaluates five open-weight 7B-8B models for Turkish document question answering under a resource-constrained local deployment setting. The primary benchmark contains 100 systematically validated questions derived from a 109-page industrial R&D report, and the evaluation protocol is replicated using a second 112-page public-sector report and an independently constructed 100-question set. All models are evaluated locally on an NVIDIA RTX 3050 laptop GPU with 6 GB VRAM using controlled prompting, decoding, and 4-bit quantisation.
The principal methodological contribution is an evidence-annotated evaluation protocol that separates retrieval failure from downstream model reasoning failure without requiring additional model calls. On the primary benchmark, end-to-end accuracy ranges from 49% to 75%. Seven lexical, dense, and hybrid retrieval configurations are additionally compared using 95% Wilson intervals and exact paired McNemar tests; none significantly outperforms the character TF-IDF baseline on either document. Evidence recall saturates differently across the two reports, showing that retrieval and effective context capacity can be binding constraints for some documents but not others. These results demonstrate that model selection, retrieval behaviour, and hardware limits must be evaluated separately when deploying open-weight LLMs for Turkish domain documents. 

---
# LLM-Assisted Workflow for Structural Difference Visualization in Evolving Software Requirements 

**Authors**: Koi McFarland, Songhui Yue  

**Link**: [PDF](https://arxiv.org/pdf/2609.28002)  

**Abstract**: This paper presents an LLM-assisted workflow for visualizing structural differences in evolving software require- ments. Implemented in the OntologyWeb environment, the work- flow represents baseline and current requirements as triple-based semantic graphs and supports side-by-side comparison of curated graph snapshots. The comparison view aligns matched entities and uses visual encoding to highlight structural changes. 

---
# Agentic Governance and Adversarial Verification for Policy-Constrained LLM Healthcare Appeal Generation 

**Authors**: Harshil Lodhiya, Alex McManus, Reese Walker  

**Link**: [PDF](https://arxiv.org/pdf/2609.27844)  

**Abstract**: Claim denial management costs U.S. healthcare approximately $260 billion annually in administrative overhead. Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) can produce fluent clinical text, but single-agent architectures fail in high-stakes healthcare: they introduce unsupported clinical details and lose the logical structure of hierarchical payer policy. We propose AGVF (Agentic Governance and Adversarial Verification Framework), a multi-agent architecture for medical-necessity appeal generation under explicit policy and evidence constraints. AGVF models appeal synthesis as a Constrained Markov Decision Process (CMDP) over five agents: policy formalization, evidence retrieval, gap analysis, adversarial critique, and gated synthesis. We prove that refinement over a fixed policy constraint graph monotonically reduces evidence-deficiency and terminates with either a complete satisfying frontier or a localized evidence gap. A deterministic citation- grounding gate prevents assertions without admissible evidence from entering shared state. We provide a reference implementation and validate it on 1,000 synthetic appeal cases parameterized from de-identified public hospital discharge data. The validation confirms zero citation-grounding violations across all AGVF cases and monotone deficiency reduction in every episode; ablating the gate raises violations to 100%, confirming it is load-bearing. The study uses no real patient records and does not measure clinical efficacy. AGVF thus contributes a theory-backed agentic architecture and verified reference implementation for policy-constrained LLM generation in healthcare. 

---
# LabourCrew: A Multi-Agent RAG Framework for Trustworthy Adversarial Deliberation and Statutory Reasoning over Labour Law 

**Authors**: Fatema Tuj Johora Faria, Mukaffi Bin Moin, Jubayer Al Mahmud, M. F. Mridha, Md. Alam Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2609.27814)  

**Abstract**: In statutory question answering, every claim must be traceable to evidence, not merely relevant, since unverifiable labour-rights answers carry serious legal consequences. Current systems fall short: single-pass RAG cannot detect insufficient evidence, while multi-agent legal-debate systems treat grounding as a prompting convention, letting agents cite unretrieved evidence. To address this gap, we introduce LabourCrew, a multi-agent RAG framework built around three grounding mechanisms: StatuteGraph, a graph index that explicitly links chapter, section, proviso, and cross-reference structure rather than fixed-length spans; an Evidence Exchange Protocol that confines advocates and an interpreter to an evidence ledger, making citation to unretrieved text impossible, while a fault-tolerant supervisor board runs advocates in parallel so individual failures degrade rather than crash the system; and a Calibrated Trust Gate that replaces categorical accept/reject decisions with a trust score, thresholded via conformal risk control for a distribution-free bound on the false-accept rate. We evaluate on LabourActQA, a 500-item Bangla question set from the Bangladesh Labour Act, 2006, spanning seven reasoning categories and three difficulty tiers. The framework drives the empirical false-accept rate to 0.081, within the target level ($\alpha = 0.10$), achieves the highest Answer Relevancy among HyDE RAG, Graph-RAG, and Hierarchical RAG (0.862 $>$ 0.839, 0.815, 0.828), and degrades gradually rather than catastrophically as question difficulty increases. These results show that calibrated abstention, not retrieval quality alone, is what makes legal question answering auditable in low-resource statutory domains. 

---
# Seal, Then Sample: Sampled Layerwise Proofs for Verifiable LLM Inference from GPT-2 to 70B 

**Authors**: Youki Lim, Sam Yong  

**Link**: [PDF](https://arxiv.org/pdf/2609.27367)  

**Abstract**: Verifying outsourced language-model inference requires a precisely identified computation and an audit whose cost a service can afford. We present Sampled Layerwise Proofs (SLP), a protocol and prototype that commits the boundary activations of every chunk of an inference trace, absorbs all commitments before any challenge is drawn, and then proves a verifier-selected subset of chunks together with the chunks that bind the prompt and the answer. Audit coverage becomes a runtime parameter over one set of commitments: on a TinyLlama-1.1B trace, proving seven of 47 chunks takes 22.0% of the time and 6.8% of the proof size of proving all 47. Because proof cost is dominated by weights rather than tokens, SLP packs concurrent requests into one trace under a block-diagonal causal mask and binds the prompt and answer of each request to its slot. Twelve packed requests are proved in 181.9 s, 6.5 times less than twelve separate proofs at the measured single-proof cost, and a simulated service proves twelve requests at 30.6 s per request with 0.6 s of verification each, rejecting a tampered answer. Disk-backed integer weights and streamed polynomial commitments let a single Llama-2-70B run complete on a 2 TB CPU host: 163 chunks sealed, five proved, a 4.34 MiB proof in 1,259 s, verified in 46.3 s without the weights. The proven object is a fixed-point canonical model; we trace a severe fidelity loss to the residual-stream bit width, repair it with an LLM-aware observer, and measure 84.8-84.9% argmax agreement with the floating-point reference over 334,705 WikiText-2 test positions. The limits are stated as precisely: guarantees cover proven chunks only, a fixed invalid chunk in the 70B setting is covered with probability 3/161, a manifest-only Fiat-Shamir schedule can be ground at 12.5 ms per attempt and needs an externally ordered challenge, and all measurements use a test reference string. 

---
# Automated Extraction of Records of Processing Activities (RoPA) Using Hybrid RAG and Locally Deployed Large Language Models 

**Authors**: To Duy Hinh, Nguyen Le Quoc Anh, Phan Van Tri, Khuong Nguyen-An  

**Link**: [PDF](https://arxiv.org/pdf/2609.27359)  

**Abstract**: Vietnam's Personal Data Protection Law (Law No. 91/2025/QH15) and Decree No. 356/2025/ND-CP, effective January 1, 2026, require organizations to establish and maintain Records of Processing Activities (RoPA). Manual RoPA preparation is labor-intensive, while cloud-hosted large language models (LLMs) may conflict with data-sovereignty requirements. We propose RoPA Manager, a system for automated RoPA information extraction using hybrid retrieval that combines lexical ranking over tsvector, dense-vector search, Reciprocal Rank Fusion (RRF), and locally deployed LLMs. We introduce a Vietnamese RoPA benchmark with 32 organizations, 77 processing activities, 12 field groups, and 4,338 reference values. Evaluation is reported at three distinct levels. The automated scorer, tested on perturbed data without invoking an LLM, achieved F1 = 0.9493 [0.9436, 0.9548]; this measures scorer robustness rather than end-to-end extraction accuracy. End-to-end extraction achieved token coverage of 50.04-55.25% against the reference labels. Two independent experts reviewed 1,558 reference values (35.9% of the benchmark), found no incorrect values, and achieved 99.68% agreement with PABAK = 0.9936. Value-level precision was not measured. Across 32 paired scenarios on a 24 GB GPU, locally deployed Qwen3.5-27B-GPTQ-Int4 showed no statistically significant difference from cloud-based DeepSeek-V4-Flash (difference 0.20 percentage points in favor of DeepSeek, 95% CI [-0.93, 1.32], p = 0.72), while Gemma-4-31B performed significantly worse (p < 0.01). 

---
# Large Knowledge Model: From Papers to a Scientific Reasoning Landscape 

**Authors**: Yuan Huang, Sihan Hu, Hongyu Gu, Chao Ma, Jiaxing Zhang, Zhiyong Zou, Caiyu Fan, Yan Xiao, Mingjun Xu, Chenyu Xie, Mingzhen Ju, Zhehao Ma, Qi Zhang, Baozong Wang, Yu Li, Zhiyuan Yao, Ruoxue Liao, Xinyu Li, Linfeng Zhang, Kun Chen, Weinan E  

**Link**: [PDF](https://arxiv.org/pdf/2609.27297)  

**Abstract**: Accumulated scientific knowledge advances inquiry when prior findings help researchers choose new questions, design investigations, and interpret results. Realizing this value at scale requires access to the reasoning that connects research problems, scientific procedures, conclusions, and evidence. We introduce the Large Knowledge Model (LKM), a scientific knowledge infrastructure that transforms the literature into a shared, computationally accessible reasoning resource. LKM represents papers as source-grounded reasoning graphs, couples structural traversal with semantic retrieval over the same objects, and aligns related questions, claims, and reasoning chains across papers. This representation forms a Scientific Reasoning Landscape with three connected views: a Question Landscape that organizes research problems and open directions, a Workflow Landscape that exposes reusable scientific procedures, and an Evidence Landscape that connects conclusions to their support, disagreement, and conditions. The unified substrate supports reasoning-aware scientific search, evidence-grounded question answering, comparative evidence analysis, and research planning. Researchers and agents can retrieve relevant work through its scientific intent, synthesize answers with inspectable supporting arguments, and develop research plans informed by established workflows and unresolved evidence. We describe a corpus-scale system and evaluate scientific retrieval and knowledge-intensive question answering. With the answering model fixed, LKM retrieval improves accuracy by 9.30%, 4.20%, and 14.69% on ChemBench, PubMedQA, and SciBench, respectively. By connecting knowledge access to scientific reasoning and action, LKM provides a common foundation for discovering relevant research, reusing scientific knowledge, and coordinating cumulative inquiry across researchers, agents, and research cycles. 

---
# Meet, Compare, or Abstain: LatWeave for Deterministic Multi-Hop Question Answering on Knowledge Lattices 

**Authors**: Yuze Ren, Shaoheng Fan, Tao Wang, Yabo Yan, Han Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.27225)  

**Abstract**: Probabilistic question-answering systems -- whether large language models (LLMs) themselves, retrieval-augmented generation (RAG), or trained multi-hop retrievers -- conflate "what is known" and "how to reason" into a single probabilistic computation: hallucination cannot be eradicated, evidence chains cannot be audited, and the system answers even when it does not know. We present LatWeave, which organizes knowledge into a multidimensional knowledge lattice and compiles multi-hop QA into three deterministic operators -- meet (constraint intersection), compare (lattice-order comparison), and abstain (structural abstention); LLMs appear only on the construction side (one-shot extraction) and the query-planning side, while the answer-generation path is zero-LLM, zero-task-training, and auditable end to end -- so that question answering over Web-published knowledge becomes reproducible item by item. Rather than claiming across-the-board SOTA, we characterize the operating envelope of this paradigm on six public benchmarks: when knowledge is complete (MetaQA, 39,093 questions) meet chains are near-lossless over three hops (any-hit 0.9975, on par with fully supervised KBQA); on templated multi-hop home ground (2WikiMultihopQA held-out n=1,258) EM 0.865, well above published structure-augmented RAG reproductions; on open-text deep composition (MuSiQue) and extraction-coverage gaps (HotpotQA) we report degradation honestly and attribute it to causes outside the lattice-algebra layer; and when information is incomplete (IIRC) we achieve structural abstention with abstain accuracy 0.971 and leak rate 0.029. Within the operating envelope, deterministic execution pays no performance penalty, and every step on the answer path can be recomputed -- precisely the source of end-to-end auditability. 

---
# LEGO: Synergizing Expert GraphRAG and Expert Chain-of-Thought for Legal Reasoning 

**Authors**: Qingjing Chen, Junkai Zhang, Shaochun Wang, Jiahao Ding, Siyuan Zheng, Yukun Yan, Zhi Zheng, Antonino Rotolo, Yun Liu, Weixing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.27009)  

**Abstract**: Large language models are increasingly applied to high-risk domains such as law, yet complex legal reasoning remains limited by two structural challenges. First, existing RAG and GraphRAG methods emphasize lexical or semantic similarity while overlooking normative relations among legal provisions. Second, vanilla Chain-of-Thought prompting may generate plausible rationales without enforcing the normative structure of legal reasoning. To deal with the bottleneck of pipelines in the legal reasoning domain, we propose LEGO, a dual-module framework that synergizes Legal Expert GraphRAG and expert Chain-of-thought for complex legal reasoning. ExpertGraphRAG uses an expert-annotated civil code graph encoding these normative relations with a greedy normative-coverage retrieval algorithm to dynamically extract instance-specific provision subgraphs, while ExpertCoT organizes the retrieved provisions and case facts into structured Provision-Fact-Conclusion reasoning. With a Qwen3-8B backbone, LEGO achieves 40.53% exact-match accuracy on LawExamQA_Civil, outperforming the evaluated RAG and CoT baselines and performing comparably to the evaluated larger models, while remaining robust on multi-hop questions. It also achieves the best results among the evaluated baselines on the open-ended benchmarks. Ablation studies confirm the individual and complementary contributions of both modules, demonstrating LEGO's effectiveness in improving LLMs' complex legal reasoning ability. Code and dataset can be found in the link: this https URL 

---
# When Learned Context Planning Fails to Beat Strong Retrieval: A Controlled Study of Planning, Routing, and Reranking for Long-Context QA 

**Authors**: Yingrui Li, Han Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.26976)  

**Abstract**: Learned context planning selects evidence atoms before an answer model reasons over them. We test whether this learned selection improves long-context multiple-choice QA after strong retrieval, routing, budgeted-selector, and reranking controls. Our primary diagnostic uses all 503 LongBench-v2 MCQ questions with Qwen2.5-7B-Instruct. The planner is SFT-trained on outcome-selected traces from 140 training and 28 development questions; because the 503-question analysis includes those questions, it is partly transductive. At an 18k-character budget, anchored hybrid retrieval reaches 36.18% accuracy and BM25 reaches 35.98%, while the best direct planner-guided method reaches 34.19%. On the untouched 152-question test split, anchored hybrid remains higher (42.11% versus 36.84%). Leakage-safe routers cannot convert a large oracle gap. Under tight budgets, the best planner is ahead by only 0.40 points at 6k and loses at 9k; planner-guided reranking has a +1.79-point estimate at 6k with a paired interval crossing zero and ties the control at 9k. Packing-order and score-flatness analyses did not identify a stable mechanism. Under this setup, learned planning is a weak relevance signal rather than a replacement for strong retrieval. 

---
