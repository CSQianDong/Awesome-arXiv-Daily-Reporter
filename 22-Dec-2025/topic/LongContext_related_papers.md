# PAACE: A Plan-Aware Automated Agent Context Engineering Framework 

**Authors**: Kamer Ali Yuksel  

**Link**: [PDF](https://arxiv.org/pdf/2512.16970)  

**Abstract**: Large Language Model (LLM) agents are increasingly deployed in complex, multi-step workflows involving planning, tool use, reflection, and interaction with external knowledge systems. These workflows generate rapidly expanding contexts that must be curated, transformed, and compressed to maintain fidelity, avoid attention dilution, and reduce inference cost. Prior work on summarization and query-aware compression largely ignores the multi-step, plan-aware nature of agentic reasoning. In this work, we introduce PAACE (Plan-Aware Automated Context Engineering), a unified framework for optimizing the evolving state of LLM agents through next-k-task relevance modeling, plan-structure analysis, instruction co-refinement, and function-preserving compression. PAACE comprises (1) PAACE-Syn, a large-scale generator of synthetic agent workflows annotated with stepwise compression supervision, and (2) PAACE-FT, a family of distilled, plan-aware compressors trained from successful teacher demonstrations. Experiments on long-horizon benchmarks (AppWorld, OfficeBench, and 8-Objective QA) demonstrate that PAACE consistently improves agent correctness while substantially reducing context load. On AppWorld, PAACE achieves higher accuracy than all baselines while lowering peak context and cumulative dependency. On OfficeBench and multi-hop QA, PAACE improves both accuracy and F1, achieving fewer steps, lower peak tokens, and reduced attention dependency. Distilled PAACE-FT retains 97 percent of the teacher's performance while reducing inference cost by over an order of magnitude, enabling practical deployment of plan-aware compression with compact models. 

---
# Learning What to Write: Write-Gated KV for Efficient Long-Context Inference 

**Authors**: Yen-Chieh Huang, Rui Fang, Ming-Syan Chen, Pi-Cheng Hsiu  

**Link**: [PDF](https://arxiv.org/pdf/2512.17452)  

**Abstract**: Long-context LLM inference is bottlenecked by the quadratic attention complexity and linear KV cache growth. Prior approaches mitigate this via post-hoc selection or eviction but overlook the root inefficiency: indiscriminate writing to persistent memory. In this paper, we formalize KV cache management as a causal system of three primitives: KV Admission, Selection, and Eviction. We instantiate KV Admission via Write-Gated KV, a lightweight mechanism that learns to predict token utility before it enters the cache. By filtering out low-utility states early to maintain a compact global cache alongside a sliding local cache, Write-Gated KV reduces memory usage by 46-57% and delivers 3.03-3.45$\times$ prefill and 1.89-2.56$\times$ decode speedups on Llama model with negligible accuracy loss, all while remaining compatible with FlashAttention and paged-KV systems. These results demonstrate that learning what to write, is a principled and practical recipe for efficient long-context inference. Code is available at this https URL . 

---
# Mindscape-Aware Retrieval Augmented Generation for Improved Long Context Understanding 

**Authors**: Yuqing Li, Jiangnan Li, Zheng Lin, Ziyan Zhou, Junjie Wu, Weiping Wang, Jie Zhou, Mo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.17220)  

**Abstract**: Humans understand long and complex texts by relying on a holistic semantic representation of the content. This global view helps organize prior knowledge, interpret new information, and integrate evidence dispersed across a document, as revealed by the Mindscape-Aware Capability of humans in psychology. Current Retrieval-Augmented Generation (RAG) systems lack such guidance and therefore struggle with long-context tasks. In this paper, we propose Mindscape-Aware RAG (MiA-RAG), the first approach that equips LLM-based RAG systems with explicit global context awareness. MiA-RAG builds a mindscape through hierarchical summarization and conditions both retrieval and generation on this global semantic representation. This enables the retriever to form enriched query embeddings and the generator to reason over retrieved evidence within a coherent global context. We evaluate MiA-RAG across diverse long-context and bilingual benchmarks for evidence-based understanding and global sense-making. It consistently surpasses baselines, and further analysis shows that it aligns local details with a coherent global representation, enabling more human-like long-context retrieval and reasoning. 

---
# Enhancing Long Document Long Form Summarisation with Self-Planning 

**Authors**: Xiaotang Du, Rohit Saxena, Laura Perez-Beltrachini, Pasquale Minervini, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2512.17179)  

**Abstract**: We introduce a novel approach for long context summarisation, highlight-guided generation, that leverages sentence-level information as a content plan to improve the traceability and faithfulness of generated summaries. Our framework applies self-planning methods to identify important content and then generates a summary conditioned on the plan. We explore both an end-to-end and two-stage variants of the approach, finding that the two-stage pipeline performs better on long and information-dense documents. Experiments on long-form summarisation datasets demonstrate that our method consistently improves factual consistency while preserving relevance and overall quality. On GovReport, our best approach has improved ROUGE-L by 4.1 points and achieves about 35% gains in SummaC scores. Qualitative analysis shows that highlight-guided summarisation helps preserve important details, leading to more accurate and insightful summaries across domains. 

---
