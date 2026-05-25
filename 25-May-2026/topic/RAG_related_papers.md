# AI Assurance: A Comprehensive Testing Strategy for Enterprise AI Systems 

**Authors**: Chitra Badagi, Divye Singh, Animesh Sen, Adinath Shirsath  

**Link**: [PDF](https://arxiv.org/pdf/2605.23459)  

**Abstract**: Enterprise AI systems, built on large language models, retrieval pipelines and autonomous agents, introduce a class of risks that traditional software quality assurance was never designed to address. These systems are probabilistic, context-sensitive and emergent: they cannot be verified to be correct in the classical sense, but only evaluated with increasing confidence. This paper presents a comprehensive assurance strategy for enterprise AI systems built around three key principles: first, that AI testing should focus on continuous risk reduction rather than strict correctness verification; second, that evaluation must be treated as a core engineering discipline alongside development; and third, that failures in AI assurance can lead to organizational impacts that are fundamentally different from those seen in traditional deterministic software systems. We introduce a structured AI Failure Taxonomy, propose a revised five-layer AI Assurance Pyramid and provide operational guidance on evaluation-driven development, RAG system testing, model lifecycle management and governance. The goal is to equip engineering leaders and practitioners with a strategy that is both philosophically grounded and operationally deployable. 

---
# A measurement substrate for agentic Kubernetes operations: Methodology and a case study in retrieval-compounding falsification 

**Authors**: Joshua Odmark, Gideon Rubin, Deon van der Vyver  

**Link**: [PDF](https://arxiv.org/pdf/2605.23058)  

**Abstract**: Empirical claims about autonomous Kubernetes operations agents are largely unfalsifiable. Published work reports observational results without controlled comparisons against an agent-disabled baseline, selection bias is endemic, pre-registered decision matrices are absent, and samples are typically too small for the noise level of the underlying scoring system. The cause is the same gap that limits the agents themselves: code agents have a verification substrate that turns "did it work" into a fast, falsifiable, ground-truth signal, and operations has nothing equivalent. We present agent-breakage, a closed-loop measurement framework that injects faults into a target Kubernetes cluster, observes how an autonomous agent responds, scores the response on four axes against ground truth, and accumulates outcome-labeled (state, action, outcome) tuples. The framework distinguishes framework error from reasoning error, supports a true off-condition control via a deterministic-embedder mechanism, and enforces pre-registered decision matrices. We use it as a case study to test whether retrieval over past postmortems compounds an agent's capability. The methodological payload is three confounds the substrate caught during that case study, each of which would have produced a wrong published claim on a less instrumented version of the same work: a pgvector index bug, a +19% selection-bias artifact, and small-sample estimates that overstated effects by roughly 3x. The retrieval result itself is a partial falsification: 1 of 3 dense-corpus scenarios significant at p<0.05, pooled effect +3.9 percentage points, not significant at n=60. A within-scenario corpus-density sweep at 360 runs shows that mechanistic alignment of near-neighbors dominates raw count. The framework is released open source. 

---
# DreamerNLplus: Interpretable Modeling of Mental Health Dynamics from Social Media Timelines using Hybrid Rule-Based and RAG Methods 

**Authors**: Maryia Zhyrko, Daisy Monika Lal, Erik van Mulligen, Lifeng Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.23052)  

**Abstract**: We present DreamerNLplus, a hybrid framework for modeling mental health dynamics from social media timelines in the CLPsych 2026 shared task. Our system addresses three tasks: psychological state modeling, temporal change detection, and sequence-level summarization.
For Task 1, we combine LLM-based data augmentation, DeBERTa classification, and Random Forest regression for structured state prediction. For Task 2, we use few-shot prompting with a locally deployed Llama 3.1 model to detect Switch and Escalation events using short-term temporal context. For Task 3.1, we explore both a deterministic rule-based summarization pipeline and a few-shot LLM-based approach, ranking \textbf{2nd} officially. Our RAG-based method achieves strong performance in Task 3.2, ranking \textbf{1st} for Improvement and \textbf{3rd} for Deterioration, demonstrating its ability to capture recurrent psychological change patterns across timelines.
Our analysis reveals key challenges, including the mismatch between classification and regression performance, the difficulty of modeling temporal transitions, and the disagreement between semantic and similarity-based evaluation metrics. These findings highlight the complexity of modeling mental health dynamics and motivate future work on unified evaluation frameworks. We share our code and prompts at this https URL 

---
# RAG4Outcome: A Retrieval-Augmented Multimodal Framework for Prognostic Prediction in Chronic Osteomyelitis 

**Authors**: Daqian Shi, Pei Han, Jishizhan Chen, Yang Wang, Xiaolei Diao, Xianyou Zheng, Pengfei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.22833)  

**Abstract**: Chronic osteomyelitis presents substantial prognostic challenges due to its high recurrence risk and complex postoperative recovery trajectories. Traditional assessment often relies on manual scoring systems, which limit scalability, efficiency, and consistency in clinical practice. Furthermore, the heterogeneous nature of clinical data poses challenges for current multimodal learning approaches that require aligned inputs and large annotated datasets. In this work, we propose RAG4Outcome, a retrieval-augmented generation (RAG) framework for prognostic prediction in chronic osteomyelitis. Our method integrates multimodal clinical data, including PET-CT imaging reports, structured surgical and diagnostic records, and unstructured follow-up notes, into a unified prediction pipeline. By combining a domain-specific retrieval corpus with expert-guided prompting, the framework enables more interpretable, evidence-grounded, and clinically reliable prognosis. Preliminary results on real-world cases demonstrate promising effectiveness and clinical alignment, highlighting the potential of RAG4Outcome for AI-assisted infection management and postoperative decision support. 

---
# The Misattribution Gap: When Memory Poisoning Looks Like Model Failure in Agentic AI Systems 

**Authors**: Tanzim Ahad, Ismail Hossain, Md Jahangir Alam, Sai Puppala, Syed Bahauddin Alam, Sajedul Talukder  

**Link**: [PDF](https://arxiv.org/pdf/2605.22842)  

**Abstract**: Multi-agent AI pipelines typically assume that agent misconduct originates from model misalignment. We identify a structural failure in this assumption, the \emph{Misattribution Gap}, where memory-layer attacks produce behaviors indistinguishable from model failure, causing defenders to apply the wrong remediation. We formalize \emph{Semantic Norm Drift} (SND) as a third path to agent misconduct, distinct from emergent misalignment and collusion. In SND, a policy-formatted document enters a shared vector store through normal uploads and later reappears as trusted system context after provenance is lost through a Trust Laundering Chain. Across 64 documented failures, attribution systems consistently blamed the model. Four safety classifiers, including one trained on memory poisoning, produced zero detections across 510 checkpoints. In 59 of 65 valid cases, agents explicitly cited the injected document as normative authority before complying. The attack requires no trigger, model access, or repeated interaction, achieves full effect within five sessions, and persists indefinitely. We introduce Counterfactual Composition Testing, which identifies the causal entry with 87.5% accuracy and zero false positives, while a forensics baseline fails across all 25 scenarios. We further prove the Retrieval-Coverage Dilemma, showing that stronger evasion inherently weakens the attack, limiting adaptive bypass strategies. Finally, we propose Memory-Persistent Information-Flow Control, which blocks 97% of attacks at the cross-session boundary where prior defenses fail. We release the SND Corpus, the first adversarial memory benchmark with temporal persistence and multi-agent composition across financial and Health Care domains. 

---
# LFRAG: Layout-oriented Fine-grained Retrieval-Augmented Generation on Multimodal Document Understanding 

**Authors**: Yifan Zhu, Yu Mi, Yue Lu, Yanchu Guan, Zhixuan Chu  

**Link**: [PDF](https://arxiv.org/pdf/2605.22829)  

**Abstract**: Multimodal Retrieval-Augmented Generation (RAG) has emerged as an effective paradigm for enhancing Large Language Models (LLMs) with external knowledge. However, existing multimodal RAG systems predominantly rely on coarse-grained page-level retrieval, which fails to capture fine-grained semantic and layout structures in visually rich documents, thereby compromising retrieval accuracy and leading to redundant context in downstream tasks. To address these issues, we propose Layout-oriented Fine-grained Retrieval-Augmented Generation (LFRAG), a novel framework that advances multimodal RAG from page-level to block-level retrieval. We perform layout segmentation to construct semantically coherent fine-grained retrieval units and design a semantic-layout fusion encoder that integrates local semantics with global context via cross-attention. With block-level late interaction retrieval, LFRAG enables precise query-content alignment and reduces irrelevant content for downstream generation. To enable rigorous evaluation, we construct LFDocQA, a large-scale benchmark with block-level annotations spanning diverse document types, designed to assess both multimodal document retrieval and question answering with greater granularity than existing datasets. Extensive experiments on LFDocQA demonstrate that LFRAG achieves state-of-the-art performance on retrieval tasks, outperforms the best baseline by 7.20% in answer accuracy, and reduces token consumption by 73.07% in generation tasks, confirming LFRAG as an accurate and efficient framework for multimodal RAG over visually rich documents. Our code and datasets will be released soon. 

---
# AI-Friendly LaTeX: Using LaTeX Code as a Knowledge Source for Retrieval-Augmented Generation 

**Authors**: Tom Verhoeff  

**Link**: [PDF](https://arxiv.org/pdf/2605.22923)  

**Abstract**: Large language models can answer questions about textbooks, lecture notes, and programming exercises more reliably when their answers are grounded in an explicit knowledge source. Retrieval-augmented generation (RAG) is a common approach: relevant fragments of a document are retrieved and inserted into the model context before answering. For mathematical and technical material, the original LaTeX source can be a better starting point than a PDF, because it contains structural information, labels, sectioning commands, macros, and authorial intent that are often lost or distorted in PDF extraction. However, LaTeX source is not automatically AI-friendly. Cross-references must be resolved, custom macros must be interpreted, exercises and examples must be identified, and author-supplied semantic metadata may be needed. This article describes a focused preprocessing approach for turning LaTeX source, together with its compiled auxiliary files and optional author annotations, into Markdown and JSONL chunks suitable for indexing in a vector database. 

---
# Query-Adaptive Semantic Chunking for Retrieval-Augmented Generation: A Dynamic Strategy with Contextual Window Expansion 

**Authors**: Mudit Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2605.22834)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems depend critically on document chunking quality for retrieving relevant context. Fixed chunking segments documents into uniform units irrespective of semantics or user intent, producing a precision-recall trade-off unresolvable by tuning chunk size alone. Semantic and agentic methods partially address these limitations but do not integrate user queries at the chunking stage. We present Query-Adaptive Semantic Chunking (QASC), which dynamically constructs chunks by integrating queries into segmentation through three mechanisms: cosine similarity scoring between sentence and query embeddings to identify seed sentences, contextual window expansion around seeds to preserve coherence, and chunk-level score aggregation to ensure holistic relevance. We evaluate QASC on 100 technical documents across 200 queries spanning four types, comparing against fixed chunking at five granularities, recursive splitting, semantic chunking, and agentic chunking. QASC achieves an F1-score of 0.85, a relative improvement of 18-27% over fixed chunking and 8-12% over semantic and agentic alternatives. Ablation studies confirm each component contributes meaningfully. Human evaluation by three annotators (Cohen kappa = 0.82) corroborates that QASC produces more relevant and coherent chunks than existing methods. 

---
# Benchmarking Google Embeddings 2 against Open-Source Models for Multilingual Dense Retrieval and RAG Systems 

**Authors**: Stefano Cirillo, Domenico Desiato, Giuseppe Polese, Giandomenico Solimando  

**Link**: [PDF](https://arxiv.org/pdf/2605.23618)  

**Abstract**: We benchmark Google Embeddings (GE2), a Vertex-AI-hosted bi-encoder with 2,048-token context and explicit task-type conditioning, against five open-source alternatives: BGE-M3, E5-large, Multilingual-E5-large (mE5-L), LaBSE, and Paraphrase-Multilingual-MPNet (mMPNet). Evaluation covers four BEIR subsets, a synthetic Italian RAG corpus, a chunking ablation considering 5 sizes of tokens with three strategies, and per-query latency on commodity CPU hardware. GE2 ranks first on every task, achieving BEIR this http URL@10 = 0.638 and IT-RAG-Bench nDCG@10 = 0.282, but at 231.6 ms median latency, it is roughly 14x slower than the fastest local models. mE5-L reaches within 0.003 nDCG of GE2 on Italian at 31 ms, making it the preferred option when sub-100 ms SLAs matter. A more striking finding concerns LaBSE, which, despite widespread multilingual deployment scores 0.188 average nDCG@10 on BEIR, below every dedicated retrieval model including mMPNet. Chunking experiments show that all six models saturate at 32-token chunks on our corpus, with semantic chunking providing measurable gains only at 16 tokens. 

---
# Asking For An Old Friend: Diagnosing and Mitigating Temporal Failure Modes in LLM-based Statutory Question Answering 

**Authors**: Max Prior, Andreas Schultz, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2605.23497)  

**Abstract**: Large language models are increasingly used for legal research, yet their fixed training cutoffs and reliance on static parametric knowledge are at odds with the evolving nature of statutory law. We study two temporal failure modes: post-cutoff staleness, where models apply superseded rules after legislative amendments, and recency bias, where models prefer newer provisions even when a historical version governs the fact pattern. To this end, we present a benchmark of 312 expert-validated, time-sensitive German statutory QA pairs spanning three categories: Post-Cutoff Amendment Questions, Pre-Amendment Questions, and Multi-Provision Pre-Amendment Questions. We evaluate five LLMs by OpenAI, Anthropic and DeepSeek under four inference settings: Vanilla, Web-search, and two retrieval-augmented variants that enforce temporal validity via a fact date extraction and version filtering. Using an LLM-as-a-judge validated against human expert ratings, we find severe degradation in the Vanilla post-cutoff setting. Both RAG approaches substantially improve performance across all question types, while web search yields unstable gains and exhibits a marked recency bias on historically anchored tasks. Our results indicate that reliable legal QA requires treating temporal validity as a hard constraint. 

---
# When Is Next-Token Prediction Useful? Marginalization, Ergodicity, Mixture Identifiability, Local Sufficiency, RAG, Tools, and Programming 

**Authors**: Francesco Corielli  

**Link**: [PDF](https://arxiv.org/pdf/2605.23278)  

**Abstract**: Language models trained on observed sequences are often described as learning the conditional distribution of the next token given previous tokens. This description is only conditionally correct. A model trained on realized token trajectories does not observe full conditional laws; it receives sampled continuations. Moreover, real language generation is conditioned not only on previous words but also on non-textual circumstances: facts, events, intentions, goals, beliefs, social context, and task-specific constraints. This paper distinguishes three objects that are often conflated: the full conditional language process conditioned on latent circumstances, the marginal text-only process obtained by integrating those circumstances out, and the model-induced distribution learned from finite observed corpora.
The paper argues that interpreting model training as estimating the marginal text-only law requires strong assumptions of stationarity, representativeness, and ergodicity, assumptions that are standard in statistical estimation but problematic when applied to heterogeneous language corpora. Even if these assumptions hold, the marginal text-only law is useful only when the observed prefix is an approximately sufficient statistic for the latent circumstances relevant to continuation. In information-theoretic terms, usefulness requires that the residual conditional mutual information between the next token and the omitted circumstances, given the observed text, be small.
The paper then extends this argument to heterogeneous training corpora.
Finally, the paper interprets Retrieval Augmented Generation (RAG) and tool use as conditional sufficiency devices. 

---
# Graph Alignment Topology as an Inductive Bias for Grounding Detection 

**Authors**: Paul Landes, Pranav Herur, Adam Cross, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.22963)  

**Abstract**: Large Language Models (LLMs) are optimized to produce distributionally plausible continuations rather than to explicitly verify whether generated propositions are entailed by source documents. This inductive bias enables generalization, but it does not encode whether responses are grounded with respect to a reference. These issues limit the use of LLMs in domains where strict factual correctness is crucial, such as clinical decision support. Existing hallucination detection approaches improve factuality through retrieval augmentation, self-consistency, or claim verification, but generally do not learn directly over alignment topology. To leverage alignment topology as an inductive bias, we construct aligned bipartite graphs between reference information and LLM outputs and train a graph neural network (GNN) to model alignment structure using message passing. The method achieves state-of-the-art results on four diverse hallucination and question-answering datasets, outperforming all compared methods, including foundational LLMs such as GPT-4o. 

---
# EVE-Agent: Evidence-Verifiable Self-Evolving Agents 

**Authors**: Yamato Arai, Yuma Ichikawa  

**Link**: [PDF](https://arxiv.org/pdf/2605.22905)  

**Abstract**: Self-evolving agents should not train on examples they cannot justify. Data-free self-evolving search agents offer a scalable route to systems that generate their own questions, answer them, and improve from their own feedback without human annotations. Yet, without verifiable evidence, this loop can reward fluent but unsupported examples, turning the self-generated curriculum into an opaque and potentially unreliable training signal. We argue that evidence verifiability is a prerequisite for trustworthy self-evolution in search agents: each generated instance should include not only an answer but also a source-grounded span whose contribution to that answer can be measured. We introduce EVE-Agent, an Evidence-Verifiable Self-Evolving Agent that operationalizes this principle through a modification to the proposer--solver framework. The proposer generates a question, an answer, and a verbatim evidence span. An evidence verifier then rewards the span according to the marginal accuracy gain when the evidence is provided. This produces a training signal that favors evidence that genuinely helps answer the question, without requiring oracle answers, human labels, or external annotations. EVE-Agent leaves the backbone model, retriever, search tool, and optimization framework unchanged. Experiments show that EVE-Agent substantially improves evidence-grounded correctness over prior self-evolving search agents. The resulting curriculum is not merely self-generated but auditable by construction: each training example carries an inspectable source span that explains why it should be trusted. 

---
