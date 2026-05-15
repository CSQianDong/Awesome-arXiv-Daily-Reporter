# Thinking Ahead: Prospection-Guided Retrieval of Memory with Language Models 

**Authors**: Harshita Chopra, Krishna Kant Chintalapudi, Suman Nath, Ryen W. White, Chirag Shah  

**Link**: [PDF](https://arxiv.org/pdf/2605.14177)  

**Abstract**: Long-horizon personalization requires dialogue assistants to retrieve user-specific facts from extended interaction histories. In practice, many relevant facts often have low semanticsimilarity to the query under dense retrieval. Standard Retrieval-Augmented Generation (RAG) and GraphRAG systems are still largely retrospective: they rely on embedding similarity to the query or on fixed graph traversals, so they often miss facts that matter for the user's needs but lie far from the query in embedding space. Inspired by prospection, the human ability to use imagined futures as cues for recall, we introduce Prospection-Guided Retrieval (PGR), which decouples retrieval from how memories are stored. Given a user query, PGR first expands the goal into a short Tree-of-Thought (ToT) or linear chain of plausible next steps, and uses these steps as retrieval probes rather than relying on the original query alone. The facts retrieved by these probes are then used to personalize the next round of prospection, enabling PGR to uncover additional memories that become relevant only after the simulation is grounded in the user's history. We also introduce MemoryQuest, a challenging multi-session benchmark in which each query is annotated with 3--5 dated reference facts subject to a low query-reference similarity constraint. Across 1,625 queries spanning 185 user profiles from 3 publicly available datasets, PGR-TOT substantially improves retrieval, including nearly 3x recall on MemoryQuest over the strongest baseline. In pairwise LLM-as-judge comparisons against baselines, PGR-generated responses are preferred on 89--98% of queries, with blinded human annotations on held-out subsets showing the same trend. Overall, the results demonstrate that explicit prospection yields large gains in long-horizon retrieval and response quality relative to similarity-only baselines. 

---
# Falkor-IRAC: Graph-Constrained Generation for Verified Legal Reasoning in Indian Judicial AI 

**Authors**: Joy Bose  

**Link**: [PDF](https://arxiv.org/pdf/2605.14665)  

**Abstract**: Legal reasoning is not semantic similarity search. A court judgment encodes constrained symbolic reasoning: precedent propagation, procedural state transitions, and statute-bound inference. These are properties that vector-based retrieval-augmented generation (RAG) cannot faithfully represent. Hallucinated precedents, outdated statute citations, and unsupported reasoning chains remain persistent failure modes in LLM-based legal AI, with real consequences for access to justice in high-caseload jurisdictions such as India. This paper presents Falkor-IRAC, a graph-constrained generation framework for Indian legal AI that grounds generation in structured reasoning over an IRAC (Issue, Rule, Analysis, Conclusion) knowledge graph. Judgments from the Supreme Court and High Courts of India are ingested as IRAC node structures enriched with procedural state transitions, precedent relationships, and statutory references, stored in FalkorDB for low-latency agentic traversal. At inference time, LLM-generated answers are accepted only if a valid supporting path can be traced through the graph, a check performed by a falsifiability oracle called the Verifier Agent. The system also detects doctrinal conflicts as a first-class output rather than silently resolving them. Falkor-IRAC is evaluated using graph-native metrics: citation grounding accuracy, path validity rate, hallucinated precedent rate, and conflict detection rate. These metrics are argued to be more appropriate for legal reasoning evaluation than BLEU and ROUGE. On a proof-of-concept corpus of 51 Supreme Court judgments, the Verifier Agent correctly validated citations on completed queries and correctly rejected fabricated citations. Evaluation against vector-only RAG baselines is left for future work, as is GPU-accelerated inference to address current timeout rates on CPU hardware. 

---
# Why Neighborhoods Matter: Traversal Context and Provenance in Agentic GraphRAG 

**Authors**: Riccardo Terrenzi, Maximilian von Zastrow, Serkan Ayvaz  

**Link**: [PDF](https://arxiv.org/pdf/2605.15109)  

**Abstract**: Retrieval-Augmented Generation can improve factuality by grounding answers in external evidence, but Agentic GraphRAG complicates what it means for citations to be faithful. In these systems, an agent explores a knowledge graph before producing an answer and a small set of citations. We frame citation faithfulness as a trajectory-level problem: final citations should not only support the answer, but also account for the graph traversal, structure, and visited-but-uncited entities that may influence it. Through controlled ablation experiments, we compare the effects of isolating, removing, and masking cited and uncited graph entities. Our results show that cited evidence is often necessary, as removing it substantially changes answers and reduces accuracy. However, citations are not sufficient, because accurate answers can also depend on uncited traversal context and surrounding graph structure. These findings suggest that citation evaluation in Agentic GraphRAG should move beyond source support toward provenance over the broader retrieval trajectory. 

---
# A Picture is Worth a Thousand Words? An Empirical Study of Aggregation Strategies for Visual Financial Document Retrieval 

**Authors**: Ho Hung Lim, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.14581)  

**Abstract**: Visual RAG has offered an alternative to traditional RAG. It treats documents as images and uses vision encoders to obtain vision patch tokens. However, hundreds of patch tokens per document create retrieval and storage challenges in a vector database. Practical deployment requires aggregating them into a single vector. This raises a critical question: does single-vector aggregation lose key information in financial documents? We develop a diagnostic benchmark using financial documents where changes in single digits can lead to significant semantic shifts. Our experiments show that single-vector aggregation collapses different documents with almost identical vectors. Metrics show that the patch level detects semantic changes, and confirm that aggregation obscures these details. We identify global texture dominance as the root cause. Our findings are consistent across model scales, retrieval-optimized embeddings, and multiple mitigation strategies, highlighting significant risks for single-vector visual document retrieval in financial applications. 

---
# Text Knows What, Tables Know When: Clinical Timeline Reconstruction via Retrieval-Augmented Multimodal Alignment 

**Authors**: Sayantan Kumar, Shahriar Noroozizadeh, Juyong Kim, Jeremy C. Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2605.15168)  

**Abstract**: Reconstructing precise clinical timelines is essential for modeling patient trajectories and forecasting risk in complex, heterogeneous conditions like sepsis. While unstructured clinical narratives offer semantically rich and contextually complete descriptions of a patient's course, they often lack temporal precision and contain ambiguous event timing. Conversely, structured electronic health record (EHR) data provides precise temporal anchors but misses a substantial portion of clinically meaningful events. We introduce a retrieval-augmented multimodal alignment framework that bridges this gap to improve the temporal precision of absolute clinical timelines extracted from text. Our approach formulates timeline reconstruction as a graph-based multistep process: it first extracts central anchor events from narratives to build an initial temporal scaffold, places non-central events relative to this backbone, and then calibrates the timeline using retrieved structured EHR rows as external temporal evidence. Evaluated using instruction-tuned large language models on the i2m4 benchmark spanning MIMIC-III and MIMIC-IV, our multimodal pipeline consistently improves absolute timestamp accuracy (AULTC) and improves temporal concordance across nearly all evaluated models over unimodal text-only reconstruction, without compromising event match rates. Furthermore, our empirical gap analysis reveals that 34.8% of text-derived events are entirely absent from tabular records, demonstrating that aligning these modalities can produce a more temporally faithful and clinically informative reconstruction of patient trajectories than either source alone. 

---
# Is Grep All You Need? How Agent Harnesses Reshape Agentic Search 

**Authors**: Sahil Sen, Akhil Kasturi, Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah  

**Link**: [PDF](https://arxiv.org/pdf/2605.15184)  

**Abstract**: Recent advances in Large Language Model (LLM) agents have enabled complex agentic workflows where models autonomously retrieve information, call tools, and reason over large corpora to complete tasks on behalf of users. Despite the growing adoption of retrieval-augmented generation (RAG) in agentic search systems, existing literature lacks a systematic comparison of how retrieval strategy choice interacts with agent architecture and tool-calling paradigm. Important practical dimensions, including how tool outputs are presented to the model and how performance changes when searches must cope with more irrelevant surrounding text, remain under-explored in agent loops. This paper reports an empirical study organized into two experiments. Experiment 1 compares grep and vector retrieval on a 116-question sample from LongMemEval, using a custom agent harness (Chronos) and provider-native CLI harnesses (Claude Code, Codex, and Gemini CLI), for both inline tool results and file-based tool results that the model reads separately. Experiment 2 compares grep-only and vector-only retrieval while progressively mixing in additional unrelated conversation history, so that each query is embedded in more distracting material alongside the passages that matter. Across Chronos and the provider CLIs, grep generally yields higher accuracy than vector retrieval in our comparisons in experiment 1; at the same time, overall scores still depend strongly on which harness and tool-calling style is used, even when the underlying conversation data are the same. 

---
# From Scenes to Elements: Multi-Granularity Evidence Retrieval for Verifiable Multimodal RAG 

**Authors**: Guanhua Chen, Chuyue Huang, Yutong Yao, Shudong Liu, Xueqing Song, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2605.15019)  

**Abstract**: Multimodal Retrieval-Augmented Generation (RAG) systems retrieve evidence at coarse granularities (entire images or scenes), creating a mismatch with fine-grained user queries and making failures unverifiable. We introduce GranuVistaVQA, a multimodal benchmark featuring real-world landmarks with element-level annotations across multiple viewpoints, capturing the partial observation challenge where individual images contain only subsets of entities. We further propose GranuRAG, a multi-granularity framework that treats visual elements as first-class retrieval units through three stages: element-level detection and classification, multi-granularity cross-modal alignment for evidence retrieval, and attribution-constrained generation. By grounding retrieval at the element level rather than relying on implicit attention, our approach enables transparent error diagnosis. Experiments demonstrate that GranuRAG achieves up to 29.2% improvement over six strong baselines for this task. 

---
# MeMo: Memory as a Model 

**Authors**: Ryan Wei Heng Quek, Sanghyuk Lee, Alfred Wei Lun Leong, Arun Verma, Alok Prakash, Nancy F. Chen, Bryan Kian Hsiang Low, Daniela Rus, Armando Solar-Lezama  

**Link**: [PDF](https://arxiv.org/pdf/2605.15156)  

**Abstract**: Large language models (LLMs) achieve strong performance across a wide range of tasks, but remain frozen after pretraining until subsequent updates. Many real-world applications require timely, domain-specific information, motivating the need for efficient mechanisms to incorporate new knowledge. In this paper, we introduce MeMo (Memory as a Model), a modular framework that encodes new knowledge into a dedicated memory model while keeping the LLM parameters unchanged. Compared to existing methods, MeMo offers several advantages: (a) it captures complex cross-document relationships, (b) it is robust to retrieval noise, (c) it avoids catastrophic forgetting in the LLM, (d) it does not require access to the LLM's weights or output logits, enabling plug-and-play integration with both open and proprietary closed-source LLMs, and (e) its retrieval cost is independent of corpus size at inference time. Our experimental results on three benchmarks, BrowseComp-Plus, NarrativeQA, and MuSiQue, show that MeMo achieves strong performance compared to existing methods across diverse settings. 

---
# AI-assisted cultural heritage dissemination: Comparing NMT and glossary-augmented LLM translation in rock art documents 

**Authors**: Vicent Briva-Iglesias, María Ferre-Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2605.14679)  

**Abstract**: Cultural heritage institutions increasingly disseminate research and interpretive materials globally, but multilingual dissemination is constrained by limited budgets and staffing. In terminology-dense domains such as rock art, translation quality depends on accurate, consistent specialised terms, and small lexical errors can mislead non-specialists and reduce reuse. We compare three English MT setups for a Spanish academic rock art text, focusing on simple, operationally feasible interventions rather than complex model-side modifications: (1) DeepL as a strong NMT baseline, (2) Gemini-Simple (LLM with a basic prompt), and (3) Gemini-RAG (the same LLM with glossary-augmented prompting via term-pair retrieval). Using PEARMUT, we conduct a human evaluation via (i) multi-way Direct Assessment (0--100) and (ii) targeted terminology auditing with a restricted MQM taxonomy. Gemini-RAG yields the highest exact-match terminology accuracy (81.4\%), versus Gemini-Simple (69.1\%) and DeepL (64.4\%), while preserving overall quality (mean DA 85.3 Gemini-RAG vs. 85.2 Gemini-Simple), outperforming DeepL (80.3). These results show that glossary-augmented prompting is a low-overhead way to improve terminology control in cultural-heritage translation if institutions maintain minimal terminology resources and lightweight evaluation procedures. 

---
# Does RAG Know When Retrieval Is Wrong? Diagnosing Context Compliance under Knowledge Conflict 

**Authors**: Yihang Chen, Pin Qian, Su Wang, Sipeng Zhang, Huan Xu, Shuhuai Lin, Xinpeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.14473)  

**Abstract**: The Context-Compliance Regime in Retrieval-Augmented Generation (RAG) occurs when retrieved context dominates
the final answer even when it conflicts with the model's parametric knowledge. Accuracy alone does not reveal
how retrieved context causally shapes answers under such conflict. We introduce Context-Driven Decomposition
(CDD), a belief-decomposition probe that operates at inference time and serves as an intervention mechanism for
controlled retrieval conflict. Across Epi-Scale stress tests, TruthfulQA misconception injection, and cross-
model reruns, CDD exposes three patterns. P1: context compliance is measurable in an upper-bound adversarial
setting, where Standard RAG reaches 15.0% accuracy on TruthfulQA misconception injection (N=500). P2:
adversarial accuracy gains transfer across model families: CDD improves accuracy on Gemini-2.5-Flash and on
Claude Haiku/Sonnet/Opus, but rationale-answer causal coupling does not transfer. CDD reaches 64.1% mistake-
injection causal sensitivity on Gemini-2.5-Flash, while sensitivities for all three Claude variants fall in the
[-3%, +7%] range, suggesting that the Claude-side accuracy gains operate through a mechanism distinct from the
explicit conflict-resolution trace. P3: explicit conflict decomposition improves robustness under temporal
drift and noisy distractors, with CDD reaching 71.3% on temporal shifts and 69.9% on distractor evidence on the
full Epi-Scale adversarial benchmark. These three patterns identify context-compliance as a structural axis
along which standard RAG can be probed and intervened on, distinct from retrieval-quality or single-method
robustness questions, and motivate releasing Epi-Scale for systematic study across model families and retrieval
pipelines. 

---
# From Descriptive to Prescriptive: Uncover the Social Value Alignment of LLM-based Agents 

**Authors**: Jinxian Qu, Qingqing Gu, Teng Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.14034)  

**Abstract**: Wide applications of LLM-based agents require strong alignment with human social values. However, current works still exhibit deficiencies in self-cognition and dilemma decision, as well as self-emotions. To remedy this, we propose a novel value-based framework that employs GraphRAG to convert principles into value-based instructions and steer the agent to behave as expected by retrieving the suitable instruction upon a specific conversation context. To evaluate the ratio of expected behaviors, we define the expected behaviors from two famous theories, Maslow's Hierarchy of Needs and Plutchik's Wheel of Emotion. By experimenting with our method on the benchmark of DAILYDILEMMAS, our method exhibits significant performance gains compared to prompt-based baselines, including ECoT, Plan-and-Solve, and Metacognitive prompting. Our method provides a basis for the emergence of self-emotion in AI systems. 

---
# Active Learners as Efficient PRP Rerankers 

**Authors**: Jeremías Figueiredo Paschmann, Juan Kaplan, Francisco Nattero Santiago Mauricio Barron Bucolo, Juan Wisznia, Luciano del Corro  

**Link**: [PDF](https://arxiv.org/pdf/2605.14236)  

**Abstract**: Pairwise Ranking Prompting (PRP) elicits pairwise preference judgments from an LLM, which are then aggregated into a ranking, usually via classical sorting algorithms. However, judgments are noisy, order-sensitive, and sometimes intransitive, so sorting assumptions do not match the setting. Because sorting aims to recover a full permutation, truncating it to meet a call budget does not produce a dependable top-K. We thus reframe PRP reranking as active learning from noisy pairwise comparisons and show that active rankers are drop-in replacements that improve NDCG@10 per call in the call-constrained regime. Our noise-robust framework also introduces a randomized-direction oracle that uses a single LLM call per pair. This approach converts systematic position bias into zero-mean noise, enabling unbiased aggregate ranking without the cost of bidirectional calls. 

---
# A Heterogeneous Temporal Memory Governance Framework for Long-Term LLM Persona Consistency 

**Authors**: Zhao Yang, Wang Huan, Li Yingshuo, Tu Haomiao, Lin Hujite  

**Link**: [PDF](https://arxiv.org/pdf/2605.14802)  

**Abstract**: Large language models often suffer from fact loss, timeline confusion, persona drift, and reduced stability during long-range interaction, especially under high-noise knowledge bases, context clearing, and cross-model transfer. To address these issues, we introduce ARPM, an external temporal memory governance framework for long-term dialogue. ARPM separates static knowledge memory from dynamic dialogue experience memory and combines vector retrieval, BM25, RRF fusion, dual-temporal reranking, chronological evidence reading, and a controlled analysis protocol for evidence verification and answer binding. Unlike approaches that encode persona consistency into model weights or rely only on long context, ARPM treats continuity as a traceable, auditable, and transferable governance problem. Using engineering logs, we conduct three experiments. First, in a 50-round question-answering setting, we compare signal-to-noise ratios of 1:5 and 1:200+, and distinguish CSV auto-judgment from manual review. Under 1:5, CSV recall accuracy is 54.0%, while manual review raises it to 100.0%. Under 1:200+, the values are 44.0% and 80.0%. These results show that automatic rules can underestimate recall after supporting evidence enters the prompt. Second, ablation results show that dialogue history retrieval is necessary for recent continuity: disabling it reduces strict accuracy from 100% to 66.7%, and disabling BM25 reduces it to 80.0%, indicating that pure semantic retrieval is insufficient for correction and tracing. Third, under a 5.1-million-character noise substrate, periodic context clearing, and multi-model handoff, ARPM maintains semantic continuity, boundary continuity, and persona consistency, while exposing limits caused by weak protocol compliance. These findings show that long-term persona consistency can be decomposed into governable components and evaluated in a white-box manner. 

---
# Deepchecks: Evaluating Retrieval-Augmented Generation (RAG) 

**Authors**: Assaf Gerner, Netta Madvil, Nadav Barak, Alex Zaikman, Jonatan Liberman, Liron Hamra, Rotem Brazilay, Shay Tsadok, Yaron Friedman, Neal Harow, Noam Bresler, Shir Chorev, Philip Tannor, Lior Rokach  

**Link**: [PDF](https://arxiv.org/pdf/2605.14488)  

**Abstract**: Large Language Models (LLMs) augmented with Retrieval-Augmented Generation (RAG) techniques are revolutionizing applications across multiple domains, such as healthcare, finance, and customer service. Despite their potential, evaluating RAG systems remains a complex challenge due to the stochastic nature of generated outputs and the intricate interplay between retrieval and generation components. This paper introduces Deepchecks, a comprehensive framework tailored for evaluating RAG applications. Deepchecks' evaluation framework addresses RAG applications evaluation through a multi-faceted approach, root cause analysis and production monitoring. By ensuring alignment with application-specific requirements, Deepchecks framework provides a robust foundation for assessing reliability, relevance, and user satisfaction in RAG systems. 

---
# When Retrieval Hurts Code Completion: A Diagnostic Study of Stale Repository Context 

**Authors**: Haojun Weng, Qianqian Yang, Hao Fu, Haobin Pan, Xinwei Lv  

**Link**: [PDF](https://arxiv.org/pdf/2605.14478)  

**Abstract**: Context: Retrieval-augmented code generation relies on cross-file repository context, but retrieved snippets may come from obsolete project states.
Objectives: We study whether temporally stale repository snippets act as harmless noise or actively induce current-state-incompatible code.
Methods: We conduct a controlled diagnostic study on a curated 17-sample set of production-helper signature changes from five Python repositories. For each sample, we compare current-only, stale-only, no-retrieval, and mixed current/stale retrieval conditions under prompts that hide commit freshness and expected current signatures.
Results: Under neutralized prompts, stale-only retrieval induces stale helper references on 15/17 Qwen2.5-Coder-7B-Instruct samples and 13/17 gpt-4.1-mini samples, corresponding to 88.2 and 76.5 percentage-point increases over current-only retrieval. No retrieval produces zero stale references but only 1/17 passing completions. The two models share 75.0% Jaccard overlap among stale-triggering samples, and mixed conditions show that adding valid current evidence largely rescues stale-only failures.
Conclusion: Temporal validity of retrieved repository context is a distinct diagnostic variable for Code RAG robustness: stale context can actively bias models toward obsolete repository state rather than merely removing useful evidence. 

---
# Contestable Multi-Agent Debate with Arena-based Argumentative Computation for Multimedia Verification 

**Authors**: Truong Thanh Hung Nguyen, Vo Thanh Khang Nguyen, Hoang-Loc Cao, Phuc Ho, Van Pham, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.14495)  

**Abstract**: Multimedia verification requires not only accurate conclusions but also transparent and contestable reasoning. We propose a contestable multi-agent framework that integrates multimodal large language models, external verification tools, and arena-based quantitative bipolar argumentation (A-QBAF) as a submission to the ICMR 2026 Grand Challenge on Multimedia Verification. Our method decomposes each case into claim-centered sections, retrieves targeted evidence, and converts evidence into structured support and attack arguments with provenance and strength scores. These arguments are resolved through small local argument graphs with selective clash resolution and uncertainty-aware escalation. The resulting system generates section-wise verification reports that are transparent, editable, and computationally practical for real-world multimedia verification. Our implementation is public at: this https URL. 

---
# Derivation Prompting: A Logic-Based Method for Improving Retrieval-Augmented Generation 

**Authors**: Ignacio Sastre, Guillermo Moncecchi, Aiala Rosá  

**Link**: [PDF](https://arxiv.org/pdf/2605.14053)  

**Abstract**: The application of Large Language Models to Question Answering has shown great promise, but important challenges such as hallucinations and erroneous reasoning arise when using these models, particularly in knowledge-intensive, domain-specific tasks. To address these issues, we introduce Derivation Prompting, a novel prompting technique for the generation step of the Retrieval-Augmented Generation framework. Inspired by logic derivations, this method involves deriving conclusions from initial hypotheses through the systematic application of predefined rules. It constructs a derivation tree that is interpretable and adds control over the generation process. We applied this method in a specific case study, significantly reducing unacceptable answers compared to traditional RAG and long-context window methods. 

---
