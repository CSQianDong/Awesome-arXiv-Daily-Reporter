# Directional Attractors in LLM Reasoning: How Similarity Retrieval Steers Iterative Summarization Based Reasoning 

**Authors**: Cagatay Tekin, Charbel Barakat, Luis Joseph Luna Limgenco  

**Link**: [PDF](https://arxiv.org/pdf/2601.08846)  

**Abstract**: Iterative summarization based reasoning frameworks such as InftyThink enable long-horizon reasoning in large language models (LLMs) by controlling context growth, but they repeatedly regenerate similar reasoning strategies across tasks. We introduce InftyThink with Cross-Chain Memory, an extension that augments iterative reasoning with an embedding-based semantic cache of previously successful reasoning patterns. At each reasoning step, the model retrieves and conditions on the most semantically similar stored lemmas, guiding inference without expanding the context window indiscriminately. Experiments on MATH500, AIME2024, and GPQA-Diamond demonstrate that semantic lemma retrieval improves accuracy in structured domains while exposing failure modes in tests that include heterogeneous domains. Geometric analyses of reasoning trajectories reveal that cache retrieval induces directional biases in embedding space, leading to consistent fix (improve baseline accuracy) and break (degradation in baseline accuracy) attractors. Our results highlight both the benefits and limits of similarity-based memory for self-improving LLM reasoning. 

---
# Structured Knowledge Representation through Contextual Pages for Retrieval-Augmented Generation 

**Authors**: Xinze Li, Zhenghao Liu, Haidong Xin, Yukun Yan, Shuo Wang, Zheni Zeng, Sen Mei, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.09402)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by incorporating external knowledge. Recently, some works have incorporated iterative knowledge accumulation processes into RAG models to progressively accumulate and refine query-related knowledge, thereby constructing more comprehensive knowledge representations. However, these iterative processes often lack a coherent organizational structure, which limits the construction of more comprehensive and cohesive knowledge representations. To address this, we propose PAGER, a page-driven autonomous knowledge representation framework for RAG. PAGER first prompts an LLM to construct a structured cognitive outline for a given question, which consists of multiple slots representing a distinct knowledge aspect. Then, PAGER iteratively retrieves and refines relevant documents to populate each slot, ultimately constructing a coherent page that serves as contextual input for guiding answer generation. Experiments on multiple knowledge-intensive benchmarks and backbone models show that PAGER consistently outperforms all RAG baselines. Further analyses demonstrate that PAGER constructs higher-quality and information-dense knowledge representations, better mitigates knowledge conflicts, and enables LLMs to leverage external knowledge more effectively. All code is available at this https URL. 

---
