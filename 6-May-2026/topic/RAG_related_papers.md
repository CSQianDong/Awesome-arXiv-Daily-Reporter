# Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems 

**Authors**: Yilun Zhao, Jinbiao Wei, Tingyu Song, Siyue Zhang, Chen Zhao, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2605.04018)  

**Abstract**: Reasoning-intensive retrieval aims to surface evidence that supports downstream reasoning rather than merely matching topical similarity. This capability is increasingly important for agentic search systems, where retrievers must provide complementary evidence across iterative search and synthesis. However, existing work remains limited on both evaluation and training: benchmarks such as BRIGHT provide narrow gold sets and evaluate retrievers in isolation, while synthetic training corpora often optimize single-passage relevance rather than evidence portfolio construction. We introduce BRIGHT-Pro, an expert-annotated benchmark that expands each query with multi-aspect gold evidence and evaluates retrievers under both static and agentic search protocols. We further construct RTriever-Synth, an aspect-decomposed synthetic corpus that generates complementary positives and positive-conditioned hard negatives, and use it to LoRA fine-tune RTriever-4B from Qwen3-Embedding-4B. Experiments across lexical, general-purpose, and reasoning-intensive retrievers show that aspect-aware and agentic evaluation expose behaviors hidden by standard metrics, while RTriever-4B substantially improves over its base model. 

---
# Safety and accuracy follow different scaling laws in clinical large language models 

**Authors**: Sebastian Wind, Tri-Thien Nguyen, Jeta Sopa, Mahshad Lotfinia, Sebastian Bickelhaup, Michael Uder, Harald Köstler, Gerhard Wellein, Sven Nebelung, Daniel Truhn, Andreas Maier, Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2605.04039)  

**Abstract**: Clinical LLMs are often scaled by increasing model size, context length, retrieval complexity, or inference-time compute, with the implicit expectation that higher accuracy implies safer behavior. This assumption is incomplete in medicine, where a few confident, high-risk, or evidence-contradicting errors can matter more than average benchmark performance. We introduce SaFE-Scale, a framework for measuring how clinical LLM safety changes across model scale, evidence quality, retrieval strategy, context exposure, and inference-time compute. To instantiate this framework, we introduce RadSaFE-200, a Radiology Safety-Focused Evaluation benchmark of 200 multiple-choice questions with clinician-defined clean evidence, conflict evidence, and option-level labels for high-risk error, unsafe answer, and evidence contradiction. We evaluated 34 locally deployed LLMs across six deployment conditions: closed-book prompting (zero-shot), clean evidence, conflict evidence, standard RAG, agentic RAG, and max-context prompting. Clean evidence produced the strongest improvement, increasing mean accuracy from 73.5% to 94.1%, while reducing high-risk error from 12.0% to 2.6%, contradiction from 12.7% to 2.3%, and dangerous overconfidence from 8.0% to 1.6%. Standard RAG and agentic RAG did not reproduce this safety profile: agentic RAG improved accuracy over standard RAG and reduced contradiction, but high-risk error and dangerous overconfidence remained elevated. Max-context prompting increased latency without closing the safety gap, and additional inference-time compute produced only limited gains. Worst-case analysis showed that clinically consequential errors concentrated in a small subset of questions. Clinical LLM safety is therefore not a passive consequence of scaling, but a deployment property shaped by evidence quality, retrieval design, context construction, and collective failure behavior. 

---
# Natural Language Processing: A Comprehensive Practical Guide from Tokenisation to RLHF 

**Authors**: Mullosharaf K. Arabov  

**Link**: [PDF](https://arxiv.org/pdf/2605.03799)  

**Abstract**: This preprint presents a systematic, research-oriented practicum that guides the reader through the entire modern NLP pipeline: from tokenisation and vectorisation to fine-tuning of large language models, retrieval-augmented generation, and reinforcement learning from human feedback. Twelve hands-on sessions combine concise theory with detailed implementation plans, formalised evaluation metrics, and transparent assessment criteria. The work is not a conventional textbook: it is designed as a reproducible research artefact where every session requires publishing code, models, and reports in public repositories. All experiments are conducted on a single evolving corpus, and the work advocates open-weight models over commercial APIs, with special attention to the Hugging Face ecosystem. The material is enriched by original research on low-resource languages, incorporating linguistic resources for Tajik and Tatar (subword tokenisers, embeddings, lexical databases, and transliteration benchmarks), demonstrating how modern NLP can be adapted to data-scarce environments. Designed for senior undergraduates, graduate students, and practising developers seeking to implement, compare, and deploy methods from classical ML to state-of-the-art LLM-based systems. 

---
# SURE-RAG: Sufficiency and Uncertainty-Aware Evidence Verification for Selective Retrieval-Augmented Generation 

**Authors**: Jingxi Qiu, Zeyu Han, Cheng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03534)  

**Abstract**: Retrieval-augmented generation (RAG) grounds answers in retrieved passages, but retrieval is not verification: a passage can be topical and still fail to justify the answer. We frame this gap as evidence sufficiency verification for selective RAG answering: given a question, a candidate answer, and retrieved evidence, predict whether the evidence supports, refutes, or is insufficient, and abstain unless support is established. We present SURE-RAG, a transparent aggregation protocol built on the observation that evidence sufficiency is a set-level property: missing hops and unresolved conflicts cannot be detected by independent passage scoring. A shared pair-level claim-evidence verifier produces local relation distributions, which SURE-RAG aggregates into interpretable answer-level signals -- coverage, relation strength, disagreement, conflict, and retrieval uncertainty -- yielding a three-way decision and an auditable selective score. We evaluate on HotpotQA-RAG v3, a controlled multi-hop benchmark, under an artifact-aware protocol (shortcut baselines, counterfactual swaps, no-oracle checks, GPT-4o audits). Calibrated SURE-RAG reaches 0.9075 Macro-F1 (0.8951 +/- 0.0069), substantially above DeBERTa mean-pooling (0.6516) and a GPT-4o judge (0.7284), while matching a strong but opaque concat cross-encoder (0.8888 +/- 0.0109) with full auditability. Risk at 30% coverage drops from 0.2588 to 0.1642, a 37% reduction in unsafe answers. To deliberately probe the task boundary, we further contrast SURE-RAG with GPT-4o on HaluBench unsafe detection: the ranking reverses (0.3343 vs 0.7389 unsafe-F1), establishing that controlled sufficiency verification and natural hallucination detection are distinct problems. 

---
# SERE: Structural Example Retrieval for Enhancing LLMs in Event Causality Identification 

**Authors**: Zhifeng Hao, Zhongjie Chen, Junhao Lu, Shengyin Yu, Guimin Hu, Keli Zhang, Ruichu Cai, Boyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03701)  

**Abstract**: Event Causality Identification (ECI) requires models to determine whether a given pair of events in a context exhibits a causal relationship. While Large Language Models (LLMs) have demonstrated strong performance across various NLP tasks, their effectiveness in ECI remains limited due to biases in causal reasoning, often leading to overprediction of causal relationships (causal hallucination). To mitigate these issues and enhance LLM performance in ECI, we propose SERE, a structural example retrieval framework that leverages LLMs' few-shot learning capabilities. SERE introduces an innovative retrieval mechanism based on three structural concepts: (i) Conceptual Path Metric, which measures the conceptual relationship between events using edit distance in ConceptNet; (ii) Syntactic Metric, which quantifies structural similarity through tree edit distance on syntactic trees; and (iii) Causal Pattern Filtering, which filters examples based on predefined causal structures using LLMs. By integrating these structural retrieval strategies, SERE selects more relevant examples to guide LLMs in causal reasoning, mitigating bias and improving accuracy in ECI tasks. Extensive experiments on multiple ECI datasets validate the effectiveness of SERE. The source code is publicly available at this https URL. 

---
# CuraView: A Multi-Agent Framework for Medical Hallucination Detection with GraphRAG-Enhanced Knowledge Verification 

**Authors**: Severin Ye, Xiao Kong, Xiaopeng He, Guangsu Yan, Dongsuk Oh  

**Link**: [PDF](https://arxiv.org/pdf/2605.03476)  

**Abstract**: Discharge summaries require extracting critical information from lengthy electronic health records (EHRs), a process that is labor-intensive when performed manually. Large language models (LLMs) can improve generation efficiency; however, they are prone to producing faithfulness hallucinations, statements that contradict source records, posing direct risks to patient safety. To address this, we present CuraView, a multi-agent framework for sentence-level detection and evidence-grounded explanation of faithfulness hallucinations in discharge summaries. CuraView constructs a GraphRAG-based knowledge graph from patient-level EHRs and implements a closed-loop generation-detection pipeline with sentence-level evidence retrieval and classification spanning four evidence grades from strong support to direct contradiction (E1-E4), yielding structured and interpretable evidence chains.
We evaluate CuraView on a subset of 250 patients from the Discharge-Me benchmark, with 50 patients held out for testing. Our fine-tuned Qwen3-14B detection model achieves an F1 of 0.831 on the safety-critical E4 metric (90.9% recall, 76.5% precision) and an F1 of 0.823 on E3+E4, representing a 50.0% relative improvement over the base model and outperforming RAGTruth-style and QAGS-style baselines. These results demonstrate that evidence-chain-based graph retrieval verification substantially improves the factual reliability of clinical documentation, while simultaneously producing reusable annotated datasets for downstream model training and distillation. 

---
# From prompting to evidence-based translation: A RAG+prompt system for Japanese-Chinese translation and its pedagogical potential 

**Authors**: Wenshi Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03387)  

**Abstract**: Large language models perform well on high-resource pairs but are less reliable for Japanese-Chinese sentences containing noun-modifying clause constructions (NMCCs). This study evaluates a retrieval-augmented generation RAG+Prompt translation system that integrates linguistic analysis, embedding-based retrieval, prompt construction, and LLM generation without modifying the base model. The analysis module outputs A1 (inner vs. outer NMCC) and A2 (risk predictions: lexical choice/NMCC handling/word order/style/register); top-k = 5 similar Ja-Zh examples (L2 distance) and A1/A2 are inserted into an enhanced prompt. Using GPT-4o and a 66-sentence test set, we compare six knowledge-base sizes (0/100/200/500/1,000/2,000). Macro-averaged sentence-level BLEU (1-4-gram with brevity penalty; cased; Chinese at the character level) is the sole metric. Mean BLEU increases from 24.28 at 0 (RAG disabled) to 29.96 at 2,000 (+5.68; +23.4%). The upward trend holds across sizes, with larger knowledge bases yielding higher scores. We conclude that the RAG+Prompt translation system improves Ja-Zh translation of sentences containing NMCCs in an interpretable and auditable manner. Limitations include one base model, one metric, and reliance on published texts and commercial APIs; future work will broaden genres, language pairs, and evaluation metrics. 

---
# RAG over Thinking Traces Can Improve Reasoning Tasks 

**Authors**: Negar Arabzadeh, Wenjie Ma, Sewon Min, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2605.03344)  

**Abstract**: Retrieval-augmented generation (RAG) has proven effective for knowledge-intensive tasks, but is widely believed to offer limited benefit for reasoning-intensive problems such as math and code generation. We challenge this assumption by showing that the limitation lies not in RAG itself, but in the choice of corpus. Instead of retrieving documents, we propose retrieving thinking traces, i.e., intermediate thinking trajectories generated during problem solving attempts. We show that thinking traces are already a strong retrieval source, and further introduce T3, an offline method that transforms them into structured, retrieval-friendly representations, to improve usability. Using these traces as a corpus, a simple retrieve-then-generate pipeline consistently improves reasoning performance across strong models and benchmarks such as AIME 2025--2026, LiveCodeBench, and GPQA-Diamond, outperforming both non-RAG baselines and retrieval over standard web corpora. For instance, on AIME, RAG with traces generated by Gemini-2-thinking achieves relative gains of +56.3%, +8.6%, and +7.6% for Gemini-2.5-Flash, GPT-OSS-120B, and GPT-5, respectively, even though these are more recent models. Interestingly, RAG on T3 also incurs little or no extra inference cost, and can even reduce inference cost by up to $15%$. Overall, our results suggest that thinking traces are an effective retrieval corpus for reasoning tasks, and transforming them into structured, compact, or diagnostic representations unlocks even stronger gains. Code available at this https URL. 

---
# AutoRAGTuner: A Declarative Framework for Automatic Optimization of RAG Pipelines 

**Authors**: Xintan Zeng, Yongchao Liu, Yice Luo, Jiajun Zhen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02967)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances LLMs, but performance is highly sensitive to complex architecture designs and hyper-parameter configurations, which currently rely on inefficient manual tuning. We present AutoRAGTuner, a declarative, configuration-driven framework that automates the RAG life cycle: construction, execution,evaluation, and optimization. AutoRAGTuner employs a modular architecture to decouple pipeline stages through a component registration mechanism. To unify heterogeneous data, we introduce the Domain-Element Model (DEM), representing objects as atomic elements with bidirectional pointers to support nodes, edges, and hyperedges. Furthermore, AutoRAGTuner integrates an adaptive Bayesian optimization engine for end-to-end hyper-parameter tuning. Experimental results demonstrate AutoRAGTuner's architectural generality: across diverse RAG pipelines, ranging from vanilla to graph-based, the framework consistently outperforms default baselines. Notably, AutoRAGTuner significantly mitigates engineering overhead, where its declarative configuration language enables a up to 95\% reduction in code churn for architectural adjustments. Overall, AutoRAGTuner provides a systematically optimizable foundation for building evolvable and reusable RAG systems. 

---
# An Agent-Oriented Pluggable Experience-RAG Skill for Experience-Driven Retrieval Strategy Orchestration 

**Authors**: Dutao Zhang, Tian Liao  

**Link**: [PDF](https://arxiv.org/pdf/2605.03989)  

**Abstract**: Retrieval-augmented generation systems often assume that one fixed retrieval pipeline is sufficient across heterogeneous tasks, yet factoid question answering, multi-hop reasoning, and scientific verification exhibit different retrieval preferences. We present Experience-RAG Skill, an agent-oriented pluggable retrieval orchestration layer positioned between the agent and the retriever pool. The proposed skill analyzes the current scene, consults an experience memory, selects an appropriate retrieval strategy, and returns structured evidence to the agent. Under a fixed candidate pool, Experience-RAG Skill achieves an overall nDCG@10 of 0.8924 on BeIR/nq, BeIR/hotpotqa, and BeIR/scifact, outperforming fixed single-retriever baselines and remaining competitive with Adaptive-RAG-style routing. The results suggest that retrieval strategy selection can be productively encapsulated as a reusable agent skill rather than being hard-coded in the upper workflow. 

---
# MEMTIER: Tiered Memory Architecture and Retrieval Bottleneck Analysis for Long-Running Autonomous AI Agents 

**Authors**: Bronislav Sidik, Lior Rokach  

**Link**: [PDF](https://arxiv.org/pdf/2605.03675)  

**Abstract**: Long-running autonomous AI agents suffer from a well-documented memory coherence problem: tool-execution success rates degrade 14 percentage points over 72-hour operation windows due to four compounding failure modes in existing flat-file memory systems. We present MEMTIER, a tripartite memory architecture for the OpenClaw agent runtime that introduces a structured episodic JSONL store, a five-signal weighted retrieval engine, an attention-attributed cognitive weight update loop, an asynchronous consolidation daemon promoting episodic facts to a semantic tier, and a PPO-based policy framework for adapting retrieval weights (infrastructure validated; performance gains pending camera-ready). On the full 500-question LongMemEval-S benchmark (Wu et al., 2025), MEMTIER achieves Acc=0.382, F1=0.412 with Qwen2.5-7B on a consumer 6GB GPU - a +33 percentage point improvement over the full-context baseline (0.050 -> 0.382, i.e., 5% -> 38%). With DeepSeek-V4-Flash fact pre-population, single-session recall reaches 0.686-0.714, exceeding the paper's RAG BM25 GPT-4o baseline (0.560) on those categories. Temporal reasoning rises to 0.323 and multi-session synthesis to 0.173, demonstrating that structured semantic pre-population qualitatively changes what lightweight retrieval can achieve. All phases run locally on a consumer laptop with a 6GB GPU. 

---
# Workspace-Bench 1.0: Benchmarking AI Agents on Workspace Tasks with Large-Scale File Dependencies 

**Authors**: Zirui Tang, Xuanhe Zhou, Yumou Liu, Linchun Li, Weizheng Wang, Hongzhang Huang, Jun Zhou, Jiachen Song, Shaoli Yu, Jinqi Wang, Zihang Zhou, Hongyi Zhou, Yuting Lv, Jinyang Li, Jiashuo Liu, Ruoyu Chen, Chunwei Liu, GuoLiang Li, Jihua Kang, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03596)  

**Abstract**: Workspace learning requires AI agents to identify, reason over, exploit, and update explicit and implicit dependencies among heterogeneous files in a worker's workspace, enabling them to complete both routine and advanced tasks effectively. Despite its importance, existing relevant benchmarks largely evaluate agents on pre-specified or synthesized files with limited real-world dependencies, leaving workspace-level evaluation underexplored. To this end, we introduce Workspace-Bench, a benchmark for evaluating AI agents on Workspace Learning invOlving Large-Scale File Dependencies. We construct realistic workspaces with 5 worker profiles, 74 file types, 20,476 files (up to 20GB) and curate 388 tasks, each with its own file dependency graph, evaluated across 7,399 total rubrics that require cross-file retrieval, contextual reasoning, and adaptive decision-making. We further provide Workspace-Bench-Lite, a 100-task subset that preserves the benchmark distribution while reducing evaluation costs by about 70%. We evaluate 4 popular agent harnesses and 7 foundation models. Experimental results show that current agents remain far from reliable workspace learning, where the best reaches only 68.7%, substantially below the human result of 80.7%, and the average performance across agents is only 47.4%. 

---
# PatRe: A Full-Stage Office Action and Rebuttal Generation Benchmark for Patent Examination 

**Authors**: Qiyao Wang, Xinyi Chen, Longze Chen, Hongbo Wang, Hamid Alinejad-Rokny, Yuan Lin, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03571)  

**Abstract**: Patent examination is a complex, multi-stage process requiring both technical expertise and legal reasoning, increasingly challenged by rising application volumes. Prior benchmarks predominantly view patent examination as discriminative classification or static extraction, failing to capture its inherently interactive and iterative nature, similar to the peer review and rebuttal process in academic publishing. In this paper, we introduce PatRe, the first benchmark that models the full patent examination lifecycle, including Office Action generation and applicant rebuttal. PatRe comprises 480 real-world cases and supports both oracle and retrieval-simulated evaluation settings. Our benchmark reframes patent examination as a dynamic, multi-turn process of justification and response. Extensive experiments across various LLMs reveal critical insights into model performance, including differences between proprietary and open-source models, as well as task asymmetries between examiner analysis and applicant-side rebuttal. These findings highlight both the potential and current limitations of LLMs in modeling complex, real-world legal reasoning and technical novelty judgment in patent examination. We release our code and dataset to facilitate future research on patent examination modeling. 

---
# MEMSAD: Gradient-Coupled Anomaly Detection for Memory Poisoning in Retrieval-Augmented Agents 

**Authors**: Ishrith Gowda  

**Link**: [PDF](https://arxiv.org/pdf/2605.03482)  

**Abstract**: Persistent external memory enables LLM agents to maintain context across sessions, yet its security properties remain formally uncharacterized. We formalize memory poisoning attacks on retrieval-augmented agents as a Stackelberg game with a unified evaluation framework spanning three attack classes with escalating access assumptions. Correcting an evaluation protocol inconsistency in the triggered-query specification of Chen et al. (2024), we show faithful evaluation increases measured attack success by $4\times$ (ASR-R: $0.25 \to 1.00$). Our primary contribution is MEMSAD (Semantic Anomaly Detection), a calibration-based defense grounded in a gradient coupling theorem: under encoder regularity, the anomaly score gradient and the retrieval objective gradient are provably identical, so any continuous perturbation that reduces detection risk necessarily degrades retrieval rank. This coupling yields a certified detection radius guaranteeing correct classification regardless of adversary strategy. We prove minimax optimality via Le Cam's method, showing any threshold detector requires $\Omega(1/\rho^2)$ calibration samples and MEMSAD achieves this up to $\log(1/\delta)$ factors. We further derive online regret bounds for rolling calibration at rate $O(\sigma^{2/3}\Delta^{1/3})$, and formally characterize a discrete synonym-invariance loophole that marks the boundary of what continuous-space defenses can guarantee. Experiments on a $3 \times 5$ attack-defense matrix with bootstrap confidence intervals, Bonferroni-corrected hypothesis tests, and Clopper-Pearson validation ($n=1{,}000$) confirm: composite defenses achieve TPR $= 1.00$, FPR $= 0.00$ across all attacks, while synonym substitution evades detection at $\Delta$ ASR-R $\approx 0$, exposing a gap existing embedding-based defenses cannot close. 

---
# From Knowledge to Action: Outcomes of the 2025 Large Language Model (LLM) Hackathon for Applications in Materials Science and Chemistry 

**Authors**: Aritra Roy, Kevin Shen, Andrew MacBride, Awwal Oladipupo, Mudassra Taskeen, Wojtek Treyde, Ruaa A. E. A. Abakar, Ahmad D. Abbas, Elsayed Abdelfatah, Abbas A. Abdullahi, Seham S. Abyah, Chahd Rahyl Adjmi, Fariha Agbere, Savyasanchi Aggarwal, Muhammad Ahmed, Tasnim Ahmed, Motasem Ajlouni, Mattias Akke, Hussein AlAdwan, Anwaar S. Alazani, Zahra A. Alharbi, Wajd A. Aljulyhi, Mohammed A. AlKubaish, Fatima A. Almahri, Sayed A. Almohri, David Obeh Alobo, Mohammed Alouni, Azizah S. Alqahtani, Omar Alsaigh, Husain Althagafi, Md. Aqib Aman, Lena Ara, Arifin, Ignacio Arretche, Abdulaziz Ashy, Syeda A. Asim, Amro Aswad, Adeel Atta, Sören Auer, Abdullah al Azmi, Toheeb Balogun, Suvo Banik, Viktoriia Baibakova, Shakira A. Baksh, Neus G. Bastús, Christina J. Bayard, Adib Bazgir, Louis Beal, Lejla Biberić, Wahid Billah, Ankita Biswas, Joshua Bocarsly, Montassar T. Bouzidi, Esma B. Boydas, Youssef Briki, Cailin Buchanan, Mauricio Cafiero, Damien Caliste, Yi Cao, Rafael E. Castañeda, Sruthy K. Chandy, Benjamin Charmes, Shayantan Chaudhuri, Yiming Chen, Alexander Chen, Jieneng Chen, Min-Hsueh Chiu, Defne Circi, Cinthya H. Contreras, Yoann Cure, Nathan Daelman, Roshini Dantuluri, Thomas Davy, William Dawson, Leonid Didukh, Rui Ding, Aminu R. Doguwa, Claudia Draxl, Sathya Edamadaka, Oulaya Elargab, Christina Ertural, Matthew L. Evans, Edvin Fako, Hossam Farag, Nur A. Fathurrahman, Merve Fedai, Rodrigo P. Ferreira, Giuseppe Fisicaro, Thomas Frank, Sasi K. Gaddipati, Abhijeet Gangan, Jennifer Garland, James Garrick, Luigi Genovese, Maryam Ghadrdran, Sandip Giri, Maxime Goulet, Jeremy Goumaz, Sara U. Gracia, Jacob Graham  

**Link**: [PDF](https://arxiv.org/pdf/2605.03205)  

**Abstract**: Large language models (LLMs) are rapidly changing how researchers in materials science and chemistry discover, organize, and act on scientific knowledge. This paper analyzes a broad set of community-developed LLM applications in an effort to identify emerging patterns in how these systems can be used across the scientific research lifecycle. We organize the projects into two complementary categories: Knowledge Infrastructure, systems that structure, retrieve, synthesize, and validate scientific information; and Action Systems, systems that execute, coordinate, or automate scientific work across computational and experimental environments. The submissions reveal a shift from single-purpose LLM tools toward integrated, multi-agent workflows that combine retrieval, reasoning, tool use, and domain-specific validation. Prominent themes include retrieval-augmented generation as grounding infrastructure, persistent structured knowledge representations, multimodal and multilingual scientific inputs, and early progress toward laboratory-integrated closed-loop systems. Together, these results suggest that LLMs are evolving from general-purpose assistants into composable infrastructure for scientific reasoning and action. This work provides a community snapshot of that transition and a practical taxonomy for understanding emerging LLM-enabled workflows in materials science and chemistry. 

---
