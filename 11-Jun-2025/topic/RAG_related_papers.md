# FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation 

**Authors**: Qinggang Zhang, Zhishang Xiang, Yilin Xiao, Le Wang, Junhui Li, Xinrun Wang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.08938)  

**Abstract**: Large language models (LLMs) augmented with retrieval systems have demonstrated significant potential in handling knowledge-intensive tasks. However, these models often struggle with unfaithfulness issues, generating outputs that either ignore the retrieved context or inconsistently blend it with the LLM`s parametric knowledge. This issue is particularly severe in cases of knowledge conflict, where the retrieved context conflicts with the model`s parametric knowledge. While existing faithful RAG approaches enforce strict context adherence through well-designed prompts or modified decoding strategies, our analysis reveals a critical limitation: they achieve faithfulness by forcibly suppressing the model`s parametric knowledge, which undermines the model`s internal knowledge structure and increases the risk of misinterpreting the context. To this end, this paper proposes FaithfulRAG, a novel framework that resolves knowledge conflicts by explicitly modeling discrepancies between the model`s parametric knowledge and retrieved context. Specifically, FaithfulRAG identifies conflicting knowledge at the fact level and designs a self-thinking process, allowing LLMs to reason about and integrate conflicting facts before generating responses. Extensive experiments demonstrate that our method outperforms state-of-the-art methods. The code is available at https:// this http URL 

---
# DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs 

**Authors**: Arie Cattan, Alon Jacovi, Ori Ram, Jonathan Herzig, Roee Aharoni, Sasha Goldshtein, Eran Ofek, Idan Szpektor, Avi Caciularu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08500)  

**Abstract**: Retrieval Augmented Generation (RAG) is a commonly used approach for enhancing large language models (LLMs) with relevant and up-to-date information. However, the retrieved sources can often contain conflicting information and it remains unclear how models should address such discrepancies. In this work, we first propose a novel taxonomy of knowledge conflict types in RAG, along with the desired model behavior for each type. We then introduce CONFLICTS, a high-quality benchmark with expert annotations of conflict types in a realistic RAG setting. CONFLICTS is the first benchmark that enables tracking progress on how models address a wide range of knowledge conflicts. We conduct extensive experiments on this benchmark, showing that LLMs often struggle to appropriately resolve conflicts between sources. While prompting LLMs to explicitly reason about the potential conflict in the retrieved documents significantly improves the quality and appropriateness of their responses, substantial room for improvement in future research remains. 

---
# CC-RAG: Structured Multi-Hop Reasoning via Theme-Based Causal Graphs 

**Authors**: Jash Rajesh Parekh, Pengcheng Jiang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.08364)  

**Abstract**: Understanding cause and effect relationships remains a formidable challenge for Large Language Models (LLMs), particularly in specialized domains where reasoning requires more than surface-level correlations. Retrieval-Augmented Generation (RAG) improves factual accuracy, but standard RAG pipelines treat evidence as flat context, lacking the structure required to model true causal dependencies. We introduce Causal-Chain RAG (CC-RAG), a novel approach that integrates zero-shot triple extraction and theme-aware graph chaining into the RAG pipeline, enabling structured multi-hop inference. Given a domain specific corpus, CC-RAG constructs a Directed Acyclic Graph (DAG) of <cause, relation, effect> triples and uses forward/backward chaining to guide structured answer generation. Experiments on two real-world domains: Bitcoin price fluctuations and Gaucher disease, show that CC-RAG outperforms standard RAG and zero-shot LLMs in chain similarity, information density, and lexical diversity. Both LLM-as-a-Judge and human evaluations consistently favor CC-RAG. Our results demonstrate that explicitly modeling causal structure enables LLMs to generate more accurate and interpretable responses, especially in specialized domains where flat retrieval fails. 

---
# Hybrid Reasoning for Perception, Explanation, and Autonomous Action in Manufacturing 

**Authors**: Christos Margadji, Sebastian W. Pattinson  

**Link**: [PDF](https://arxiv.org/pdf/2506.08462)  

**Abstract**: Industrial processes must be robust and adaptable, as environments and tasks are often unpredictable, while operational errors remain costly and difficult to detect. AI-based control systems offer a path forward, yet typically depend on supervised learning with extensive labelled datasets, which limits their ability to generalize across variable and data-scarce industrial settings. Foundation models could enable broader reasoning and knowledge integration, but rarely deliver the quantitative precision demanded by engineering applications. Here, we introduceControl and Interpretation of Production via Hybrid Expertise and Reasoning (CIPHER): a vision-language-action (VLA) model framework aiming to replicate human-like reasoning for industrial control, instantiated in a commercial-grade 3D printer. It integrates a process expert, a regression model enabling quantitative characterization of system states required for engineering tasks. CIPHER also incorporates retrieval-augmented generation to access external expert knowledge and support physics-informed, chain-of-thought reasoning. This hybrid architecture exhibits strong generalization to out-of-distribution tasks. It interprets visual or textual inputs from process monitoring, explains its decisions, and autonomously generates precise machine instructions, without requiring explicit annotations. CIPHER thus lays the foundations for autonomous systems that act with precision, reason with context, and communicate decisions transparently, supporting safe and trusted deployment in industrial settings. 

---
# Eliciting Fine-Tuned Transformer Capabilities via Inference-Time Techniques 

**Authors**: Asankhaya Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2506.08060)  

**Abstract**: Large language models have transformed natural language processing, yet supervised fine-tuning (SFT) remains computationally intensive. This paper formally proves that capabilities acquired through SFT can be approximated by a base transformer model using inference-time techniques, specifically in-context learning (ICL), without altering model parameters, under idealized assumptions including unbounded computational resources and access to the fine-tuning dataset. We extend these results to practical scenarios with finite context lengths and partial dataset access. For text generation tasks with fixed output length $l$, datasets of size $\mathrm{O}\left( \frac{m V}{\varepsilon^2} \log \frac{m}{\delta} \right)$ or, with bounded context, $\mathrm{O}\left( \frac{l \log V}{\varepsilon^2} \log \frac{1}{\delta} \right)$ suffice to approximate fine-tuned behavior across $m$ contexts within error $\varepsilon$, where $V$ is the vocabulary size and $\delta$ is the failure probability. For linear classification, datasets of size $\mathrm{O}\left( \frac{d}{\varepsilon} \right)$ or, with fixed context, $\mathrm{O}\left( \frac{1}{\varepsilon^2} \log \frac{1}{\delta} \right)$ are sufficient, where $d$ is the input dimension. Grounded in the Turing completeness of transformers, these results provide a theoretical foundation for resource-efficient deployment of large language models, with practical techniques like retrieval-augmented generation bridging theory to real-world applications. 

---
