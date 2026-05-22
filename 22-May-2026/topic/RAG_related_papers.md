# ChronoMedKG: A Temporally-Grounded Biomedical Knowledge Graph and Benchmark for Clinical Reasoning 

**Authors**: Md Shamim Ahmed, Farzaneh Firoozbakht, Lukas Galke Poech, Jan Baumbach, Richard Röttger  

**Link**: [PDF](https://arxiv.org/pdf/2605.22734)  

**Abstract**: Biomedical knowledge graphs (KGs) treat disease associations as static facts, but temporal information is crucial for clinical reasoning, e.g., a symptom diagnostic of one disease at age 3 may imply a different disease at age 13. Existing KGs such as PrimeKG, Hetionet, and iKraph do not encode when a finding becomes clinically relevant over the course of a disease. This limits their usefulness for longitudinal clinical reasoning and retrieval augmentation.
We introduce ChronoMedKG, a temporal biomedical knowledge graph that contains 460,497 evidence-linked triples (filtered from 13M raw extractions) covering 13,431 diseases. Each association is tied to temporal components like onset window or progression stage, which are backed by PMID-traceable evidence and a multi-signal credibility score. The graph is constructed through a disease-autonomous multi-agent pipeline in which multiple frontier LLMs independently extract knowledge from PubMed and PMC literature. Only those relations are kept that are supported by multi-model consensus, survive credibility filtering, as well as ontology alignment.
ChronoMedKG scored 92.7% agreement against Orphadata and adds temporal grounding for 6,250 diseases absent from HPOA, Orphadata, and Phenopackets, including 1,657 Orphanet-coded rare diseases. We further introduce ChronoTQA, a benchmark of 3,341 questions across eight task types (six temporal plus two static controls), with a 12-question supplementary probe. Frontier LLMs lose roughly 30 points moving from static to temporal questions; ChronoMedKG retrieval rescues 47-65% of their long-tail failures, against 17-29% for HPOA-RAG. As such, ChronoMedKG provides a crucial temporal axis for retrieval-augmented clinical systems that was previously absent. 

---
# Evaluating Commercial AI Chatbots as News Intermediaries 

**Authors**: Mirac Suzgun, Emily Shen, Federico Bianchi, Alexander Spangher, Thomas Icard, Daniel E. Ho, Dan Jurafsky, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.22785)  

**Abstract**: AI chatbots are rapidly shaping how people encounter the news, yet no prior study has systematically measured how accurately these systems, with their proprietary search integrations and retrieval-synthesis pipelines, handle emerging facts across languages and regions. We present a 14-day (February 9-22, 2026) evaluation of six AI chatbots (Gemini 3 Flash and Pro, Grok 4, Claude 4.5 Sonnet, GPT-5 and GPT-4o mini) on 2,100 factual questions derived from same-day BBC News reporting across six regional services (US & Canada, Arabic, Afrique, Hindi, Russian, Turkish). The best systems achieve over 90% multiple-choice accuracy on questions about events reported hours earlier. The same systems, however, lose 11-13% under free-response evaluation, and 16-17% across the cohort. We further characterize three failure patterns. First, every model achieves its lowest accuracy on Hindi (79% vs. 89-91% elsewhere) and citations indicate an Anglophone retrieval bias (e.g., models answering Hindi queries cite English Wikipedia more than any Hindi outlet). Second, retrieval, not reasoning, failures drive over 70% of all errors. When models retrieve a correct source, they often extract the correct answer; the problem is to land on the right source in the first place. Third, models achieving 88-96% accuracy on well-formed questions drop to 19-70% when questions contain subtle false premises, with the most vulnerable model accepting fabricated facts 64% of the time. We also identify a detection-accuracy paradox: the best false-premise detector ranks second in adversarial accuracy (abstention rate), while a weaker detector ranks first, showing that premise detection and answer recovery are partially independent capabilities. Overall, these suggest that high accuracy can mask systematic regional inequity, near-total dependence on retrieval infrastructure, and vulnerability to imperfect queries real users pose. 

---
# More Context, Larger Models, or Moral Knowledge? A Systematic Study of Schwartz Value Detection in Political Texts 

**Authors**: Víctor Yeste, Paolo Rosso  

**Link**: [PDF](https://arxiv.org/pdf/2605.22641)  

**Abstract**: Detecting Schwartz values in political text is difficult because implicit cues often depend on surrounding arguments and fine-grained distinctions between neighboring values. We study when context and explicit moral knowledge help sentence-level value detection. Using the ValuesML/Touch{é} ValueEval format, we compare sentence, window, and full-document inputs; no-RAG and retrieval-augmented settings with a curated moral knowledge base; supervised DeBERTa-v3-base/large encoders; and zero-shot LLMs from 12B to 123B parameters. The results show that more context is not uniformly better: full-document context improves supervised DeBERTa encoders by 3.8--4.8 macro-F1 points over sentence-only input, but does not consistently help zero-shot LLMs. Retrieved moral knowledge is more consistently useful in matched comparisons, improving each tested model family and context condition under early fusion. However, scaling from DeBERTa-v3-base to large and from 12B to larger LLMs does not guarantee gains, and simple early fusion outperforms the tested late-fusion and cross-attention RAG variants for encoders. Per-value analyses show that context and retrieval help most for socially situated or conceptually confusable values. These findings suggest that value-sensitive NLP should evaluate context, knowledge, and model family jointly rather than treating longer inputs or larger models as universal improvements. 

---
# DeferMem: Query-Time Evidence Distillation via Reinforcement Learning for Long-Term Memory QA 

**Authors**: Jianing Yin, Tan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.22411)  

**Abstract**: Large language model (LLM) agents still struggle with long-term memory question answering, where answer-supporting evidence is often scattered across long conversational histories and buried in substantial irrelevant content. Existing memory systems typically process memory before future queries are known, then retrieve the resulting units based on similarity rather than their utility for answering the query. This workflow leaves downstream answerers to denoise retrieved candidates and reconstruct query-specific evidence. We present DeferMem, a long-term memory framework that decouples this problem into high-recall candidate retrieval and query-conditioned evidence distillation. DeferMem uses a lightweight segment-link structure to organize raw history and retrieve broad candidates at query time. It then applies a memory distiller trained with DistillPO, our reinforcement learning algorithm for distilling the high-recall but highly noisy candidates into a set of faithful, self-contained, and query-conditioned evidence. DistillPO formulates post-retrieval evidence distillation as a structured action comprising message selection and evidence rewriting. It optimizes this action with a decomposed-and-gated reward pipeline and structure-aligned advantage assignment, gating reward components from validity to quality checks while exposing task-level correctness feedback early and assigning each reward to its responsible output span. On LoCoMo and LongMemEval-S, DeferMem surpasses strong baselines in QA accuracy and memory-system efficiency, achieving the highest QA accuracy with the fastest runtime and zero commercial-API token cost for memory operations. 

---
# Evaluation of Chunking Strategies for Effective Text Embedding in Low-Resource Language on Agricultural Documents 

**Authors**: Sovandara Chhoun, Pichdara Po, Sereiwathna Ros, Wan-Sup Cho, Saksonita Khoeurn  

**Link**: [PDF](https://arxiv.org/pdf/2605.22203)  

**Abstract**: In this study, we compare the performance of four text chunking approaches: Recursive, Khmer-Aware, Sentence-Based, and LLM-Based within a Retrieval-Augmented Generation (RAG) framework applied to Khmer agricultural documents. The document chunks are encoded using the BGE-M3 multilingual embedding model and retrieved using the FAISS library. Performance is evaluated using four metrics: Average Retrieval Score (L2 distance), Answer Relevance, Khmer Coverage, and Khmer Intersection over Union, all measured against ground-truth question-answer pairs. For evaluation, we perform 5-fold cross-validation over 18 question-answer pairs. We observe the best performance for the character-based Recursive chunking method with a chunk size of 300 characters, achieving the lowest L2 distance (0.4295 +- 0.0461), highest Answer Relevance (0.8663 +- 0.0199), and highest Khmer IoU (0.6441 +- 0.0347). A paired t-test shows a statistically significant improvement over the Sentence-Based chunking method in L2 distance (p = 0.0121). These results highlight the importance of segmentation granularity and structural preservation for optimizing dense retrieval in morphologically complex, low-resource languages such as Khmer. 

---
# Claim-Selective Certification for High-Risk Medical Retrieval-Augmented Generation 

**Authors**: Shao Kan  

**Link**: [PDF](https://arxiv.org/pdf/2605.21949)  

**Abstract**: Medical RAG systems in high-risk QA settings are often evaluated through a single answer-or-abstain decision, but mixed evidence may support one claim, require conditions for another, and contradict a third. We study claim-selective certification: each response is decomposed into verifiable claims, scored against retrieved evidence, and mapped by an intent-aware selector to {full, partial, conflict, abstain}. On the primary weak-label certificate protocol, whose real-source-only dev/test rows cover the naturally occurring non-abstain actions, the full system records UCCR=0.0000, PAU=1.0000, PAU Precision=0.9901, and action accuracy=0.9204 on dev (n=314), and UCCR=0.0000, PAU=0.9967, PAU Precision=0.9739, and action accuracy=0.8997 on test (n=319). UCCR measures unsupported-claim risk within the certificate definition, and a source-missing counterfactual slice evaluates abstain under empty evidence. Shortcut controls quantify the action-label prior explained by source and intent metadata, while source/evidence-novel slices characterize transfer boundaries. The resulting interface separates action-label prediction from evidence-linked claim selection under mixed evidence. 

---
# When Cases Get Rare: A Retrieval Benchmark for Off-Guideline Clinical Question Answering 

**Authors**: Doeun Lee, Muge Zhang, Yi Yu, Ashish Manne, Stephen Koesters, Frank Wen, Brady Buchanan, Lynda Villagomez, Oluwatoba Moninuola, James Lim, Kathryn Tobin, Andrew Srisuwananukorn, Ping Zhang, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.21807)  

**Abstract**: Across medical specialties, clinical practice is anchored in evidence-based guidelines that codify best studied diagnostic and treatment pathways. These pathways routinely fall short for the long tail of real-world care not covered by guidelines. Most medical large language models (LLMs), however, are trained to encode common, guideline-focused medical knowledge in their parameters. Current evaluations test models primarily on recalling and reasoning with this memorized content, often in multiple-choice settings. Given the fundamental importance of evidence-based reasoning in medicine, it is neither feasible nor reliable to depend on memorization in practice. To address this gap, we introduce OGCaReBench, a free-form retrieval-focused benchmark aimed at evaluating LLMs at answering clinical questions that require going beyond typical guidelines. Extracted from published medical case reports and validated by medical experts, OGCaReBench contains long-form clinical questions requiring free-text answers, providing a systematic framework for assessing open-ended medical reasoning in rare, case-based scenarios. Our experiments reveal that even the best-performing baseline (GPT-5.2) correctly answers only 56% of our benchmark with specialized models only reaching 42%. Augmenting models with retrieved medical articles improves this performance to up to 82% (using GPT-5.2) highlighting the importance of evidence-grounding for real-world medical reasoning tasks. This work thus establishes a foundation for benchmarking and advancing both general-purpose and medical LLMs to produce reliable answers in challenging clinical contexts. 

---
# A Comparative Study of Language Models for Khmer Retrieval-Augmented Question Answering 

**Authors**: Sereiwathna Ros, Phannet Pov, Ratanaktepi Chhor, Kimleang Ly, Wan-Sup Cho, Saksonita Khoeurn  

**Link**: [PDF](https://arxiv.org/pdf/2605.22099)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm for grounding large language model (LLM) outputs in retrieved evidence, thereby reducing hallucination and improving factual accuracy. Its efficacy, however, remains largely unexamined for low-resource, non-Latin-script languages such as Khmer. In this paper, we present a RAG-based question answering system for Khmer-language telecom-domain documents. We conduct a two-phase comparative evaluation. First, we benchmark three embedding models: BGE-M3 (567M), Jina-Embeddings-v3 (570M), and Qwen3-Embedding (597M), for dense retrieval over Khmer documents. BGE-M3 consistently performs best, achieving a Hit Rate@3 of 0.285, File Hit Rate@3 of 0.700, MRR@3 of 0.221, and Precision@3 of 0.112, substantially outperforming the other retrievers. Second, using BGE-M3 as the selected retriever, we evaluate five generator backends: Qwen3 (8B), Qwen3.5 (9B), Sailor2-8B-Chat, SeaLLMs-v3-7B-Chat, and Llama-SEA-LION-v2-8B-IT, on a curated golden dataset of 200 Khmer question-answer pairs. To quantify system performance, we apply six RAGAS-inspired metrics: faithfulness, answer relevance, context relevance, factual correctness, answer similarity, and answer correctness. The results show no single model dominates across all metrics: Qwen3.5-9B achieves the highest faithfulness (0.859) and context relevance (0.726), Qwen3-8B attains the highest factual correctness (0.380), and SeaLLMs-v3-7B-Chat performs best on answer relevance (0.867), answer similarity (0.836), and answer correctness (0.599). These findings highlight that retriever choice remains a major bottleneck for Khmer RAG, while generator strengths vary depending on whether the priority is grounding, factual precision, or semantic similarity. 

---
# SpecHop: Continuous Speculation for Accelerating Multi-Hop Retrieval Agents 

**Authors**: Mehrdad Saberi, Keivan Rezaei, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2605.21965)  

**Abstract**: Large language models increasingly use external tools such as web search and document retrieval to solve information-intensive tasks. However, multi-hop tool use in complex tasks introduces substantial latency, since the model must repeatedly wait for tool observations before continuing. We study how to accelerate such trajectories without changing the final trajectory the model would have taken without acceleration, assuming access to faster but less reliable speculator tools. We develop a theoretical framework for lossless speculation in multi-hop tool-use settings, characterizing the optimal achievable latency gain. We propose SpecHop, a continuous speculation framework that maintains multiple speculative threads, verifies predicted observations asynchronously as target tool outputs arrive, commits correct branches, and rolls back incorrect ones. This preserves accuracy while reducing wall-clock latency. We show that SpecHop can approach oracle latency gains with enough active threads. Empirically, on retrieval-augmented multi-hop tasks, SpecHop closely matches theoretical predictions and reduces latency by up to 40\% in some settings. Code: this https URL 

---
# RankJudge: A Multi-Turn LLM-as-a-Judge Synthetic Benchmark Generator 

**Authors**: Zhenwei Tang, Zhaoyan Liu, Rasa Hosseinzadeh, Tongzi Wu, Keyvan Golestan, Jesse C. Cresswell  

**Link**: [PDF](https://arxiv.org/pdf/2605.21748)  

**Abstract**: As interactive LLM-based applications are created and refined, model developers need to evaluate the quality of generated text along many possible axes. For simpler systems, human evaluation may be practical, but in complicated systems like conversational chatbots, the amount of generated text can overwhelm human annotation resources. Model developers have begun to rely heavily on auto-evaluation, where LLMs are also used to judge generation quality. However, existing LLM-as-a-judge benchmarks largely focus on simple Q\&A tasks that do not match the complexity of multi-turn conversations. We introduce RankJudge, a benchmark generator for evaluating LLM-as-a-judge on multi-turn conversations grounded in reference documents. RankJudge creates pairs of conversations where one conversation has a single flaw injected into one turn. This construction allows paired conversations to be labeled unambiguously as better or worse, and precisely isolates failure categories to individual turns, enabling a strict joint correctness criterion for judging. We implement RankJudge across the domains of machine learning, biomedicine, and finance, evaluate 21 frontier LLM judges, and rank those judges via the Bradley-Terry model. Our formulation also allows ranking each conversation pair with difficulty ratings, which we use to dynamically curate the evaluation slice to reduce label noise, as confirmed via human annotation. We find that judge rankings are stable under partial observability, coarser correctness criteria, and an alternative random-walk rating algorithm. 

---
# Search-E1: Self-Distillation Drives Self-Evolution in Search-Augmented Reasoning 

**Authors**: Zihan Liang, Yufei Ma, Ben Chen, Zhipeng Qian, Xuxin Zhang, Huangyu Dai, Lingtao Mao  

**Link**: [PDF](https://arxiv.org/pdf/2605.22511)  

**Abstract**: Post-training has become the dominant recipe for turning a language model into a competent search-augmented reasoning agent. A line of recent work pushes its performance further by adding elaborate machinery on top of this standard pipeline. These augmentations import external supervision from stronger external systems, attach auxiliary modules such as process reward models or retrospective critics, restructure the rollout itself with tree search or multi-stage curricula, or shape the reward with hand-crafted bonuses and penalties. Each addition delivers a measurable gain, but each also inflates the training pipeline and ties the recipe to resources or designs that may not always be available. We take a step back and ask whether any of this machinery is actually necessary, and propose Search-E1, a self-evolution method that lets a search-augmented agent improve through only vanilla GRPO interleaved with offline self-distillation (OFSD). After each GRPO round, the policy rolls out on its own training questions. A token-level forward KL objective then aligns the policy's inference-time distribution to its own distribution under a privileged context that exposes a more efficient sibling trajectory. Despite this simplicity, the procedure naturally provides dense per-step supervision. On seven QA benchmarks, Search-E1 reaches $0.440$ average EM with Qwen2.5-3B, surpassing all open-source baselines at both scales. Code and complete version will be made public soon. 

---
# SGR-Bench: Benchmarking Search Agents on State-Gated Retrieval 

**Authors**: Ningyuan Li, Haiyang Shen, Mugeng Liu, Yudong Han, Zhuofan Shi, Sixiong Xie, Yun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.22219)  

**Abstract**: Recent advances in large language models and tool-using agents have expanded the range of benchmarked web tasks. Yet an important class of specialized retrieval tasks remains undercharacterized. On many specialized data-retrieval websites, answer-bearing evidence becomes accessible only after establishing the correct site-specific retrieval state through filters, views, hierarchies, or scopes. We term this capability state-gated retrieval (SGR). We introduce SGR-Bench, a benchmark for this setting containing 100 expert-curated tasks spanning six source families and 12 public data ecosystems. Each task requires discovering the appropriate website and configuring its site-specific retrieval state to produce a structured answer. SGR-Bench pairs constraint-guided and goal-oriented formulations of the same underlying problems, enabling controlled comparisons between explicit and implicit guidance for state-gated retrieval. We evaluate eight CLI-based agentic LLM systems and three commercial search-agent products. On SGR-Bench, the strongest system reaches only 66.18% item-level F1, while row-level F1 remains much lower. A manual audit of 156 analyzable failed CLI trajectories shows why: agents often reach a relevant web source, but establish the wrong site-specific retrieval state. Retrieval-scope drift (37.2%) and criterion mismatch (27.6%) dominate, whereas final answer composition accounts for only 10.3%. The dataset and single-case evaluation instructions are available at this https URL. 

---
# Format-Constraint Coupling in Knowledge Graph Construction from Statistical Tables 

**Authors**: Jingxuan Qi, Zhiqiang Ye, Yuxiang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.21974)  

**Abstract**: An extraction schema should not reduce knowledge graph fidelity. On statistical CSV, however, it can. We study country-by-year time-series matrices, a common layout on open-data portals. In this setting, serialization format and schema constraints interact super-additively. Their joint effect exceeds the sum of independent effects by up to +1.180 (2x2 factorial, 6 datasets). Bootstrap 95% CIs are strictly positive on 4/6 datasets, with strongest evidence on wide Type-II matrices. More critically, a schema applied to a mismatched format can trigger catastrophic mismatch. Fact coverage falls below the unconstrained baseline on 4/6 datasets through entity inflation or extraction refusal. We call this observed pattern format-constraint coupling. Probing and token ablation support a surface-form anchoring explanation centred on column-name references. Controlled variants across format-schema pairings, GraphRAG hosts, and LLM families show the same direction within the measured scope; one LLM family shows only partial activation. The observation also has a diagnostic consequence. Three standard retrieval modes largely mask construction quality (delta <= 1pp), whereas direct graph access exposes gaps up to +47.6pp (p < 0.0001). To support fidelity-aware evaluation, we release CSVFidelity-Bench. It contains 15 datasets, 11 Type-II matrices, 4 Type-III tables, and 1,892 Gold Standard facts across 6 domains. 

---
# Ex-GraphRAG: Interpretable Evidence Routing for Graph-Augmented LLMs 

**Authors**: Yoav Kor Sade, Arvindh Arun, Rishi Puri, Steffen Staab, Maya Bechler-Speicher  

**Link**: [PDF](https://arxiv.org/pdf/2605.21994)  

**Abstract**: GraphRAG conditions language models on subgraphs retrieved from knowledge graphs, encoded via message-passing GNNs. Because these encoders entangle node contributions through iterated neighborhood aggregation, there is no closed-form way to determine how much each retrieved entity influenced the encoder's output, and therefore no way to faithfully audit what structural evidence actually reached the model. We introduce Ex-GraphRAG, which replaces the GNN encoder with a Multivariate Graph Neural Additive Network (M-GNAN), an extension of additive graph models to high-dimensional embedding spaces that yields an exact decomposition of the encoder's output across individual nodes and feature groups, without post-hoc approximation. On STaRK-Prime, this auditable encoder matches black-box performance. Using it to audit evidence routing, we uncover a semantic-structural mismatch: the nodes that dominate the encoder's output are structurally disconnected in the retrieved subgraph, held together by low-attribution intermediaries whose removal degrades multi-hop QA by up to 28%. This mismatch, invisible to any opaque encoder, reveals that semantic importance and structural connectivity are governed by disjoint sets of nodes, with direct implications for retrieval pruning, context construction, and failure diagnosis in graph-augmented LLMs. 

---
