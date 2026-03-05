# RAGNav: A Retrieval-Augmented Topological Reasoning Framework for Multi-Goal Visual-Language Navigation 

**Authors**: Ling Luo, Qiangian Bai  

**Link**: [PDF](https://arxiv.org/pdf/2603.03745)  

**Abstract**: Vision-Language Navigation (VLN) is evolving from single-point pathfinding toward the more challenging Multi-Goal VLN. This task requires agents to accurately identify multiple entities while collaboratively reasoning over their spatial-physical constraints and sequential execution order. However, generic Retrieval-Augmented Generation (RAG) paradigms often suffer from spatial hallucinations and planning drift when handling multi-object associations due to the lack of explicit spatial this http URL address these challenges, we propose RAGNav, a framework that bridges the gap between semantic reasoning and physical structure. The core of RAGNav is a Dual-Basis Memory system, which integrates a low-level topological map for maintaining physical connectivity with a high-level semantic forest for hierarchical environment abstraction. Building on this representation, the framework introduces an anchor-guided conditional retrieval and a topological neighbor score propagation mechanism. This approach facilitates the rapid screening of candidate targets and the elimination of semantic noise, while performing semantic calibration by leveraging the physical associations inherent in the topological this http URL mechanism significantly enhances the capability of inter-target reachability reasoning and the efficiency of sequential planning. Experimental results demonstrate that RAGNav achieves state-of-the-art (SOTA) performance in complex multi-goal navigation tasks. 

---
# RAG-X: Systematic Diagnosis of Retrieval-Augmented Generation for Medical Question Answering 

**Authors**: Aswini Sivakumar, Vijayan Sugumaran, Yao Qiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03541)  

**Abstract**: Automated question-answering (QA) systems increasingly rely on retrieval-augmented generation (RAG) to ground large language models (LLMs) in authoritative medical knowledge, ensuring clinical accuracy and patient safety in Artificial Intelligence (AI) applications for healthcare. Despite progress in RAG evaluation, current benchmarks focus only on simple multiple-choice QA tasks and employ metrics that poorly capture the semantic precision required for complex QA tasks. These approaches fail to diagnose whether an error stems from faulty retrieval or flawed generation, limiting developers from performing targeted improvement. To address this gap, we propose RAG-X, a diagnostic framework that evaluates the retriever and generator independently across a triad of QA tasks: information extraction, short-answer generation, and multiple-choice question (MCQ) answering. RAG-X introduces Context Utilization Efficiency (CUE) metrics to disaggregate system success into interpretable quadrants, isolating verified grounding from deceptive accuracy. Our experiments reveal an ``Accuracy Fallacy", where a 14\% gap separates perceived system success from evidence-based grounding. By surfacing hidden failure modes, RAG-X offers the diagnostic transparency needed for safe and verifiable clinical RAG systems. 

---
# Developing an AI Assistant for Knowledge Management and Workforce Training in State DOTs 

**Authors**: Divija Amaram, Lu Gao, Gowtham Reddy Gudla, Tejaswini Sanjay Katale  

**Link**: [PDF](https://arxiv.org/pdf/2603.03302)  

**Abstract**: Effective knowledge management is critical for preserving institutional expertise and improving the efficiency of workforce training in state transportation agencies. Traditional approaches, such as static documentation, classroom-based instruction, and informal mentorship, often lead to fragmented knowledge transfer, inefficiencies, and the gradual loss of expertise as senior engineers retire. Moreover, given the enormous volume of technical manuals, guidelines, and research reports maintained by these agencies, it is increasingly challenging for engineers to locate relevant information quickly and accurately when solving field problems or preparing for training tasks. These limitations hinder timely decision-making and create steep learning curves for new personnel in maintenance and construction operations. To address these challenges, this paper proposes a Retrieval-Augmented Generation (RAG) framework with a multi-agent architecture to support knowledge management and decision making. The system integrates structured document retrieval with real-time, context-aware response generation powered by a large language model (LLM). Unlike conventional single-pass RAG systems, the proposed framework employs multiple specialized agents for retrieval, answer generation, evaluation, and query refinement, which enables iterative improvement and quality control. In addition, the system incorporates an open-weight vision-language model to convert technical figures into semantic textual representations, which allows figure-based knowledge to be indexed and retrieved alongside text. Retrieved text and figure-based context are then provided to an open-weight large language model, which generates the final responses grounded in the retrieved evidence. 

---
# From Conflict to Consensus: Boosting Medical Reasoning via Multi-Round Agentic RAG 

**Authors**: Wenhao Wu, Zhentao Tang, Yafu Li, Shixiong Kai, Mingxuan Yuan, Zhenhong Sun, Chunlin Chen, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03292)  

**Abstract**: Large Language Models (LLMs) exhibit high reasoning capacity in medical question-answering, but their tendency to produce hallucinations and outdated knowledge poses critical risks in healthcare fields. While Retrieval-Augmented Generation (RAG) mitigates these issues, existing methods rely on noisy token-level signals and lack the multi-round refinement required for complex reasoning. In the paper, we propose **MA-RAG** (**M**ulti-Round **A**gentic RAG), a framework that facilitates test-time scaling for complex medical reasoning by iteratively evolving both external evidence and internal reasoning history within an agentic refinement loop. At each round, the agent transforms semantic **conflict** among candidate responses into actionable queries to retrieve external evidence, while optimizing history reasoning traces to mitigate long-context degradation. MA-RAG extends the *self-consistency* principle by leveraging the lack of consistency as a proactive signal for multi-round agentic reasoning and retrieval, and mirrors a *boosting* mechanism that iteratively minimizes the residual error toward a stable, high-fidelity medical **consensus**. Extensive evaluations across 7 medical Q&A benchmarks show that MA-RAG consistently surpasses competitive inference-time scaling and RAG baselines, delivering **substantial +6.8 points** on average accuracy over the backbone model. Our code is available at [this url](this https URL). 

---
# Retrieval or Representation? Reassessing Benchmark Gaps in Multilingual and Visually Rich RAG 

**Authors**: Martin Asenov, Kenza Benkirane, Dan Goldwater, Aneiss Ghodsi  

**Link**: [PDF](https://arxiv.org/pdf/2603.04238)  

**Abstract**: Retrieval-augmented generation (RAG) is a common way to ground language models in external documents and up-to-date information. Classical retrieval systems relied on lexical methods such as BM25, which rank documents by term overlap with corpus-level weighting. End-to-end multimodal retrievers trained on large query-document datasets claim substantial improvements over these approaches, especially for multilingual documents with complex visual layouts. We demonstrate that better document representation is the primary driver of benchmark improvements. By systematically varying transcription and preprocessing methods while holding the retrieval mechanism fixed, we demonstrate that BM25 can recover large gaps on multilingual and visual benchmarks. Our findings call for decomposed evaluation benchmarks that separately measure transcription and retrieval capabilities, enabling the field to correctly attribute progress and focus effort where it matters. 

---
# Benchmarking Legal RAG: The Promise and Limits of AI Statutory Surveys 

**Authors**: Mohamed Afane, Emaan Hariri, Derek Ouyang, Daniel E. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2603.03300)  

**Abstract**: Retrieval-augmented generation (RAG) offers significant potential for legal AI, yet systematic benchmarks are sparse. Prior work introduced LaborBench to benchmark RAG models based on ostensible ground truth from an exhaustive, multi-month, manual enumeration of all U.S. state unemployment insurance requirements by U.S. Department of Labor (DOL) attorneys. That prior work found poor performance of standard RAG (70% accuracy on Boolean tasks). Here, we assess three emerging tools not previously evaluated on LaborBench: the Statutory Research Assistant (STARA), a custom statutory research tool, and two commercial tools by Westlaw and LexisNexis marketing AI statutory survey capabilities. We make five main contributions. First, we show that STARA achieves substantial performance gains, boosting accuracy to 83%. Second, we show that commercial platforms fare poorly, with accuracy of 58% (Westlaw AI) and 64% (Lexis+ AI), even worse than standard RAG. Third, we conduct a comprehensive error analysis, comparing our outputs to those compiled by DOL attorneys, and document both reasoning errors, such as confusion between related legal concepts and misinterpretation of statutory exceptions, and retrieval failures, where relevant statutory provisions are not captured. Fourth, we discover that many apparent errors are actually significant omissions by DOL attorneys themselves, such that STARA's actual accuracy is 92%. Fifth, we chart the path forward for legal RAG through concrete design principles, offering actionable guidance for building AI systems capable of accurate multi-jurisdictional legal research. 

---
# SE-Search: Self-Evolving Search Agent via Memory and Dense Reward 

**Authors**: Jian Li, Yizhang Jin, Dongqi Liu, Hang Ding, Jiafu Wu, Dongsheng Chen, Yunhang Shen, Yulei Qin, Ying Tai, Chengjie Wang, Xiaotong Yuan, Yabiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.03293)  

**Abstract**: Retrieval augmented generation (RAG) reduces hallucinations and factual errors in large language models (LLMs) by conditioning generation on retrieved external knowledge. Recent search agents further cast RAG as an autonomous, multi-turn information-seeking process. However, existing methods often accumulate irrelevant or noisy documents and rely on sparse reinforcement learning signals. We propose \textbf{S}elf-\textbf{E}volving \textbf{Search}, a Self-Evolving Search agent that improves online search behavior through three components, memory purification, atomic query training, and dense rewards. SE-Search follows a \textit{Think-Search-Memorize} strategy that retains salient evidence while filtering irrelevant content. Atomic query training promotes shorter and more diverse queries, improving evidence acquisition. Dense rewards provide fine-grained feedback that speeds training. Experiments on single-hop and multi-hop question answering benchmarks show that \texttt{SE-Search-3B} outperforms strong baselines, yielding a $10.8$ point absolute improvement and a $33.8\%$ relative gain over Search-R1.\footnote{We will make the code and model weights publicly available upon acceptance.} 

---
