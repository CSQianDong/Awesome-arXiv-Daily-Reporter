# GRAIL: Geometry-Aware Retrieval-Augmented Inference with LLMs over Hyperbolic Representations of Patient Trajectories 

**Authors**: Zhan Qu, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2602.12828)  

**Abstract**: Predicting future clinical events from longitudinal electronic health records (EHRs) is challenging due to sparse multi-type clinical events, hierarchical medical vocabularies, and the tendency of large language models (LLMs) to hallucinate when reasoning over long structured histories. We study next-visit event prediction, which aims to forecast a patient's upcoming clinical events based on prior visits. We propose GRAIL, a framework that models longitudinal EHRs using structured geometric representations and structure-aware retrieval. GRAIL constructs a unified clinical graph by combining deterministic coding-system hierarchies with data-driven temporal associations across event types, embeds this graph in hyperbolic space, and summarizes each visit as a probabilistic Central Event that denoises sparse observations. At inference time, GRAIL retrieves a structured set of clinically plausible future events aligned with hierarchical and temporal progression, and optionally refines their ranking using an LLM as a constrained inference-time reranker. Experiments on MIMIC-IV show that GRAIL consistently improves multi-type next-visit prediction and yields more hierarchy-consistent forecasts. 

---
# TRACE: Temporal Reasoning via Agentic Context Evolution for Streaming Electronic Health Records (EHRs) 

**Authors**: Zhan Qu, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2602.12833)  

**Abstract**: Large Language Models (LLMs) encode extensive medical knowledge but struggle to apply it reliably to longitudinal patient trajectories, where evolving clinical states, irregular timing, and heterogeneous events degrade performance over time. Existing adaptation strategies rely on fine-tuning or retrieval-based augmentation, which introduce computational overhead, privacy constraints, or instability under long contexts. We introduce TRACE (Temporal Reasoning via Agentic Context Evolution), a framework that enables temporal clinical reasoning with frozen LLMs by explicitly structuring and maintaining context rather than extending context windows or updating parameters. TRACE operates over a dual-memory architecture consisting of a static Global Protocol encoding institutional clinical rules and a dynamic Individual Protocol tracking patient-specific state. Four agentic components, Router, Reasoner, Auditor, and Steward, coordinate over this structured memory to support temporal inference and state evolution. The framework maintains bounded inference cost via structured state compression and selectively audits safety-critical clinical decisions. Evaluated on longitudinal clinical event streams from MIMIC-IV, TRACE significantly improves next-event prediction accuracy, protocol adherence, and clinical safety over long-context and retrieval-augmented baselines, while producing interpretable and auditable reasoning traces. 

---
