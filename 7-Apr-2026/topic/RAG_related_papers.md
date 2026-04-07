# MisEdu-RAG: A Misconception-Aware Dual-Hypergraph RAG for Novice Math Teachers 

**Authors**: Zhihan Guo, Rundong Xue, Yuting Lu, Jionghao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.04036)  

**Abstract**: Novice math teachers often encounter students' mistakes that are difficult to diagnose and remediate. Misconceptions are especially challenging because teachers must explain what went wrong and how to solve them. Although many existing large language model (LLM) platforms can assist in generating instructional feedback, these LLMs loosely connect pedagogical knowledge and student mistakes, which might make the guidance less actionable for teachers. To address this gap, we propose MisEdu-RAG, a dual-hypergraph-based retrieval-augmented generation (RAG) framework that organizes pedagogical knowledge as a concept hypergraph and real student mistake cases as an instance hypergraph. Given a query, MisEdu-RAG performs a two-stage retrieval to gather connected evidence from both layers and generates a response grounded in the retrieved cases and pedagogical principles. We evaluate on \textit{MisstepMath}, a dataset of math mistakes paired with teacher solutions, as a benchmark for misconception-aware retrieval and response generation across topics and error types. Evaluation results on \textit{MisstepMath} show that, compared with baseline models, MisEdu-RAG improves token-F1 by 10.95\% and yields up to 15.3\% higher five-dimension response quality, with the largest gains on \textit{Diversity} and \textit{Empowerment}. To verify its applicability in practical use, we further conduct a pilot study through a questionnaire survey of 221 teachers and interviews with 6 novices. The findings suggest that MisEdu-RAG provides diagnosis results and concrete teaching moves for high-demand misconception scenarios. Overall, MisEdu-RAG demonstrates strong potential for scalable teacher training and AI-assisted instruction for misconception handling. Our code is available on GitHub: this https URL. 

---
# Ruling Out to Rule In: Contrastive Hypothesis Retrieval for Medical Question Answering 

**Authors**: Byeolhee Kim, Min-Kyung Kim, Young-Hak Kim, Tae-Joon Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2604.04593)  

**Abstract**: Retrieval-augmented generation (RAG) grounds large language models in external medical knowledge, yet standard retrievers frequently surface hard negatives that are semantically close to the query but describe clinically distinct conditions. While existing query-expansion methods improve query representation to mitigate ambiguity, they typically focus on enriching target-relevant semantics without an explicit mechanism to selectively suppress specific, clinically plausible hard negatives. This leaves the system prone to retrieving plausible mimics that overshadow the actual diagnosis, particularly when such mimics are dominant within the corpus. We propose Contrastive Hypothesis Retrieval (CHR), a framework inspired by the process of clinical differential diagnosis. CHR generates a target hypothesis $H^+$ for the likely correct answer and a mimic hypothesis $H^-$ for the most plausible incorrect alternative, then scores documents by promoting $H^+$-aligned evidence while penalizing $H^-$-aligned content. Across three medical QA benchmarks and three answer generators, CHR outperforms all five baselines in every configuration, with improvements of up to 10.4 percentage points over the next-best method. On the $n=587$ pooled cases where CHR answers correctly while embedded hypothetical-document query expansion does not, 85.2\% have no shared documents between the top-5 retrieval lists of CHR and of that baseline, consistent with substantive retrieval redirection rather than light re-ranking of the same candidates. By explicitly modeling what to avoid alongside what to find, CHR bridges clinical reasoning with retrieval mechanism design and offers a practical path to reducing hard-negative contamination in medical RAG systems. 

---
# Retrieval Augmented Conversational Recommendation with Reinforcement Learning 

**Authors**: Zhenrui Yue, Honglei Zhuang, Zhen Qin, Zhankui He, Huimin Zeng, Julian McAuley, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04457)  

**Abstract**: Large language models (LLMs) exhibit enhanced capabilities in language understanding and generation. By utilizing their embedded knowledge, LLMs are increasingly used as conversational recommender systems (CRS), achieving improved performance across diverse scenarios. However, existing LLM-based methods rely on pretrained knowledge without external retrieval mechanisms for novel items. Additionally, the lack of a unified corpus poses challenges for integrating retrieval augmentation into CRS. Motivated by these challenges, we present RAR, a novel two-stage retrieval augmented conversational recommendation framework that aligns retrieval and generation to enhance both performance and factuality. To support this framework and provide a unified corpus, we construct a large-scale movie corpus, comprising over 300k movies with rich metadata, such as titles, casts and plot summaries. Leveraging this data, our primary contribution is RAR, the first framework to departs from standard two-stage CRS by dynamically bridging retrieval and generation. First, a retriever model generates candidate items based on user history; in the subsequent stage, an LLM refines the recommendations by incorporating conversational context with retrieved results. In addition, we introduce a novel reinforcement learning (RL) method that leverages LLM feedback to iteratively update the retriever. By creating a collaborative feedback loop that reinforces sampled candidate sets with higher ranking metrics, RAR effectively mitigates the misalignment between the retrieval and generation stages. Furthermore, grounding the LLM in factual metadata allows our RL-driven approach to capture subtle user intentions and generate context-aware recommendations with reduced hallucinations. We validate our approach through extensive experiments on multiple benchmarks, where RAR consistently outperforms state-of-the-art baseline methods. 

---
# MMP-Refer: Multimodal Path Retrieval-augmented LLMs For Explainable Recommendation 

**Authors**: Xiangchen Pan, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.03666)  

**Abstract**: Explainable recommendations help improve the transparency and credibility of recommendation systems, and play an important role in personalized recommendation scenarios. At present, methods for explainable recommendation based on large language models(LLMs) often consider introducing collaborative information to enhance the personalization and accuracy of the model, but ignore the multimodal information in the recommendation dataset; In addition, collaborative information needs to be aligned with the semantic space of LLM. Introducing collaborative signals through retrieval paths is a good choice, but most of the existing retrieval path collection schemes use the existing Explainable GNN algorithms. Although these methods are effective, they are relatively unexplainable and not be suitable for the recommendation field.
To address the above challenges, we propose MMP-Refer, a framework using \textbf{M}ulti\textbf{M}odal Retrieval \textbf{P}aths with \textbf{Re}trieval-augmented LLM \textbf{F}or \textbf{E}xplainable \textbf{R}ecommendation. We use a sequential recommendation model based on joint residual coding to obtain multimodal embeddings, and design a heuristic search algorithm to obtain retrieval paths by multimodal embeddings; In the generation phase, we integrated a trainable lightweight collaborative adapter to map the graph encoding of interaction subgraphs to the semantic space of the LLM, as soft prompts to enhance the understanding of interaction information by the LLM. Extensive experiments have demonstrated the effectiveness of our approach. Codes and data are available at this https URL. 

---
# LLM-based Listwise Reranking under the Effect of Positional Bias 

**Authors**: Jingfen Qiao, Jin Huang, Xinyu Ma, Shuaiqiang Wang, Dawei Yin, Evangelos Kanoulas, Andrew Yates  

**Link**: [PDF](https://arxiv.org/pdf/2604.03642)  

**Abstract**: LLM-based listwise passage reranking has attracted attention for its effectiveness in ranking candidate passages. However, these models suffer from positional bias, where passages positioned towards the end of the input are less likely to be moved to top positions in the ranking. We hypothesize that there are two primary sources of positional bias: (1) architectural bias inherent in LLMs and (2) the imbalanced positioning of relevant documents. To address this, we propose DebiasFirst, a method that integrates positional calibration and position-aware data augmentation during fine-tuning. Positional calibration uses inverse propensity scoring to adjust for positional bias by re-weighting the contributions of different positions in the loss function when training. Position-aware augmentation augments training data to ensure that each passage appears equally across varied positions in the input list. This approach markedly enhances both effectiveness and robustness to the original ranking across diverse first-stage retrievers, reducing the dependence of NDCG@10 performance on the position of relevant documents. DebiasFirst also complements the inference-stage debiasing methods, offering a practical solution for mitigating positional bias in reranking. 

---
# Lightweight Query Routing for Adaptive RAG: A Baseline Study on RAGRouter-Bench 

**Authors**: Prakhar Bansal, Shivangi Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2604.03455)  

**Abstract**: Retrieval-Augmented Generation pipelines span a wide range of retrieval strategies that differ substantially in token cost and capability. Selecting the right strategy per query is a practical efficiency problem, yet no routing classifiers have been trained on RAGRouter-Bench \citep{wang2026ragrouterbench}, a recently released benchmark of $7,727$ queries spanning four knowledge domains, each annotated with one of three canonical query types: factual, reasoning, and summarization. We present the first systematic evaluation of lightweight classifier-based routing on this benchmark. Five classical classifiers are evaluated under three feature regimes, namely, TF-IDF, MiniLM sentence embeddings \citep{reimers2019sbert}, and hand-crafted structural features, yielding 15 classifier feature combinations. Our best configuration, TF-IDF with an SVM, achieves a macro-averaged F1 of $\mathbf{0.928}$ and an accuracy of $\mathbf{93.2\%}$, while simulating $\mathbf{28.1\%}$ token savings relative to always using the most expensive paradigm. Lexical TF-IDF features outperform semantic sentence embeddings by $3.1$ macro-F1 points, suggesting that surface keyword patterns are strong predictors of query-type complexity. Domain-level analysis reveals that medical queries are hardest to route and legal queries most tractable. These results establish a reproducible query-side baseline and highlight the gap that corpus-aware routing must close. 

---
# BridgeRAG: Training-Free Bridge-Conditioned Retrieval for Multi-Hop Question Answering 

**Authors**: Andre Bacellar  

**Link**: [PDF](https://arxiv.org/pdf/2604.03384)  

**Abstract**: Multi-hop retrieval is not a single-step relevance problem: later-hop evidence should be ranked by its utility conditioned on retrieved bridge evidence, not by similarity to the original query alone. We present BridgeRAG, a training-free, graph-free retrieval method for retrieval-augmented generation (RAG) over multi-hop questions that operationalizes this view with a tripartite scorer s(q,b,c) over (question, bridge, candidate). BridgeRAG separates coverage from scoring: dual-entity ANN expansion broadens the second-hop candidate pool, while a bridge-conditioned LLM judge identifies the active reasoning chain among competing candidates without any offline graph or proposition index. Across four controlled experiments we show that this conditioning signal is (i) selective: +2.55pp on parallel-chain queries (p<0.001) vs. ~0 on single-chain subtypes; (ii) irreplaceable: substituting the retrieved passage with generated SVO query text reduces R@5 by 2.1pp, performing worse than even the lowest-SVO-similarity pool passage; (iii) predictable: cos(b,g2) correlates with per-query gain (Spearman rho=0.104, p<0.001); and (iv) mechanistically precise: bridge conditioning causes productive re-rankings (18.7% flip-win rate on parallel-chain vs. 0.6% on single-chain), not merely more churn. Combined with lightweight coverage expansion and percentile-rank score fusion, BridgeRAG achieves the best published training-free R@5 under matched benchmark evaluation on all three standard MHQA benchmarks without a graph database or any training: 0.8146 on MuSiQue (+3.1pp vs. PropRAG, +6.8pp vs. HippoRAG2), 0.9527 on 2WikiMultiHopQA (+1.2pp vs. PropRAG), and 0.9875 on HotpotQA (+1.35pp vs. PropRAG). 

---
# PassiveQA: A Three-Action Framework for Epistemically Calibrated Question Answering via Supervised Finetuning 

**Authors**: Madhav S Baidya  

**Link**: [PDF](https://arxiv.org/pdf/2604.04565)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance in question answering and retrieval-augmented generation (RAG), yet they implicitly assume that user queries are fully specified and answerable. In real-world settings, queries are often incomplete, ambiguous, or missing critical variables, leading models to produce overconfident or hallucinated responses.
In this work, we study decision-aware query resolution under incomplete information, where a model must determine whether to Answer, Ask for clarification, or Abstain. We show that standard and enhanced RAG systems do not reliably exhibit such epistemic awareness, defaulting to answer generation even when information is insufficient.
To address this, we propose PassiveQA, a three-action framework that aligns model behaviour with information sufficiency through supervised finetuning. Our approach integrates structured information-state representations, knowledge graph-grounded context, and a finetuned planner that explicitly models missing variables and decision reasoning.
Experiments across multiple QA datasets show that the finetuned planner achieves significant improvements in macro F1 and abstention recall while reducing hallucination rates, under a compute-constrained training regime.
These results provide strong empirical evidence that epistemic decision-making must be learned during training rather than imposed at inference time. 

---
# GROUNDEDKG-RAG: Grounded Knowledge Graph Index for Long-document Question Answering 

**Authors**: Tianyi Zhang, Andreas Marfurt  

**Link**: [PDF](https://arxiv.org/pdf/2604.04359)  

**Abstract**: Retrieval-augmented generation (RAG) systems have been widely adopted in contemporary large language models (LLMs) due to their ability to improve generation quality while reducing the required input context length. In this work, we focus on RAG systems for long-document question answering. Current approaches suffer from a heavy reliance on LLM descriptions resulting in high resource consumption and latency, repetitive content across hierarchical levels, and hallucinations due to no or limited grounding in the source text. To improve both efficiency and factual accuracy through grounding, we propose GroundedKG-RAG, a RAG system in which the knowledge graph is explicitly extracted from and grounded in the source document. Specifically, we define nodes in GroundedKG as entities and actions, and edges as temporal or semantic relations, with each node and edge grounded in the original sentences. We construct GroundedKG from semantic role labeling (SRL) and abstract meaning representation (AMR) parses and then embed it for retrieval. During querying, we apply the same transformation to the query and retrieve the most relevant sentences from the grounded source text for question answering. We evaluate GroundedKG-RAG on examples from the NarrativeQA dataset and find that it performs on par with a state-of-the art proprietary long-context model at smaller cost and outperforms a competitive baseline. Additionally, our GroundedKG is interpretable and readable by humans, facilitating auditing of results and error analysis. 

---
# Document-Level Numerical Reasoning across Single and Multiple Tables in Financial Reports 

**Authors**: Yi-Cheng Wang, Wei-An Wang, Chu-Song Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.03664)  

**Abstract**: Despite the strong language understanding abilities of large language models (LLMs), they still struggle with reliable question answering (QA) over long, structured documents, particularly for numerical reasoning. Financial annual reports exemplify this difficulty: financial statement analysis often hinges on accurate arithmetic, and analysts derive key indicators by integrating evidence scattered across multiple tables and narrative text. However, existing benchmarks focus largely on single-table settings, leaving cross-table document-level numerical reasoning underexplored. To address this gap, we introduce FinLongDocQA, a dataset for both single-table and cross-table financial numerical reasoning in long-context reports. Evaluating both closed-source and open-source LLMs on FinLongDocQA reveals two bottlenecks: (1) annual reports often exceed 129k tokens, exacerbating the context rot problem for locating relevant tables; and (2) even when relevant evidence is located, LLMs remain prone to errors in multi-step numerical reasoning. We propose FinLongDocAgent, a Multi-Agent Multi-Round Retrieval-Augmented Generation (RAG) approach that iteratively retrieves evidence, performs intermediate calculations, and verifies results across rounds. Experiments highlight the importance of iterative retrieval and verification for reliable numerical QA in long financial documents. 

---
# MultiPress: A Multi-Agent Framework for Interpretable Multimodal News Classification 

**Authors**: Tailong Luo, Hao Li, Rong Fu, Xinyue Jiang, Huaxuan Ding, Yiduo Zhang, Zilin Zhao, Simon Fong, Guangyin Jin, Jianyuan Ni  

**Link**: [PDF](https://arxiv.org/pdf/2604.03586)  

**Abstract**: With the growing prevalence of multimodal news content, effective news topic classification demands models capable of jointly understanding and reasoning over heterogeneous data such as text and images. Existing methods often process modalities independently or employ simplistic fusion strategies, limiting their ability to capture complex cross-modal interactions and leverage external knowledge. To overcome these limitations, we propose MultiPress, a novel three-stage multi-agent framework for multimodal news classification. MultiPress integrates specialized agents for multimodal perception, retrieval-augmented reasoning, and gated fusion scoring, followed by a reward-driven iterative optimization mechanism. We validate MultiPress on a newly constructed large-scale multimodal news dataset, demonstrating significant improvements over strong baselines and highlighting the effectiveness of modular multi-agent collaboration and retrieval-augmented reasoning in enhancing classification accuracy and interpretability. 

---
# Knowledge Packs: Zero-Token Knowledge Delivery via KV Cache Injection 

**Authors**: Andrey Pustovit  

**Link**: [PDF](https://arxiv.org/pdf/2604.03270)  

**Abstract**: RAG wastes tokens. We propose Knowledge Packs: pre-computed KV caches that deliver the same knowledge at zero token cost. For causal transformers, the KV cache from a forward pass on text F is identical to what a joint pass on F+q would produce - this follows directly from the causal mask. The equivalence is exact but fragile: wrong chat template formatting causes 6-7pp degradation, which we believe explains prior claims of KV outperforming RAG. With correct formatting: zero divergences across 700 questions on Qwen3-8B and Llama-3.1-8B, up to 95% token savings. The KV interface also enables behavioral steering that RAG cannot do. Because RoPE rotates keys but leaves values untouched, contrastive deltas on cached values can nudge model behavior while key arithmetic destroys coherence. The effect sits in mid-layer values (33-66%), independent directions are nearly orthogonal (cos~0) and compose, and both channels - knowledge and steering - run simultaneously at alpha<=0.7 without interference. No training, no weight modification. 

---
# BLADE: Better Language Answers through Dialogue and Explanations 

**Authors**: Chathuri Jayaweera, Bonnie J. Dorr  

**Link**: [PDF](https://arxiv.org/pdf/2604.03236)  

**Abstract**: Large language model (LLM)-based educational assistants often provide direct answers that short-circuit learning by reducing exploration, self-explanation, and engagement with course materials. We present BLADE (Better Language Answers through Dialogue and Explanations), a grounded conversational assistant that guides learners to relevant instructional resources rather than supplying immediate solutions. BLADE uses a retrieval-augmented generation (RAG) framework over curated course content, dynamically surfacing pedagogically relevant excerpts in response to student queries. Instead of delivering final answers, BLADE prompts direct engagement with source materials to support conceptual understanding. We conduct an impact study in an undergraduate computer science course, with different course resource configurations and show that BLADE improves students' navigation of course resources and conceptual performance compared to simply providing the full inventory of course resources. These results demonstrate the potential of grounded conversational AI to reinforce active learning and evidence-based reasoning. 

---
# Scaling DPPs for RAG: Density Meets Diversity 

**Authors**: Xun Sun, Baiheng Xie, Li Huang, Qiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.03240)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding generation in external knowledge, yielding relevance responses that are aligned with factual evidence and evolving corpora. Standard RAG pipelines construct context through relevance ranking, performing point-wise scoring between the user query and each corpora chunk. This formulation, however, ignores interactions among retrieved candidates, leading to redundant contexts that dilute density and fail to surface complementary evidence. We argue that effective retrieval should optimize jointly for both density and diversity, ensuring the grounding evidence that is dense in information yet diverse in coverage. In this study, we propose ScalDPP, a diversity-aware retrieval mechanism for RAG that incorporates Determinantal Point Processes (DPPs) through a lightweight P-Adapter, enabling scalable modeling of inter-chunk dependencies and complementary context selection. In addition, we develop a novel set-level objective, Diverse Margin Loss (DML), that enforces ground-truth complementary evidence chains to dominate any equally sized redundant alternatives under DPP geometry. Experimental results demonstrate the superiority of ScalDPP, substantiating our core statement in practice. 

---
# Search, Do not Guess: Teaching Small Language Models to Be Effective Search Agents 

**Authors**: Yizhou Liu, Qi Sun, Yulin Chen, Siyue Zhang, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.04651)  

**Abstract**: Agents equipped with search tools have emerged as effective solutions for knowledge-intensive tasks. While Large Language Models (LLMs) exhibit strong reasoning capabilities, their high computational cost limits practical deployment for search agents. Consequently, recent work has focused on distilling agentic behaviors from LLMs into Small Language Models (SLMs). Through comprehensive evaluation on complex multi-hop reasoning tasks, we find that despite possessing less parametric knowledge, SLMs invoke search tools less frequently and are more prone to hallucinations. To address this issue, we propose \policy, a lightweight fine-tuning approach that explicitly trains SLMs to reliably retrieve and generate answers grounded in retrieved evidence. Compared to agent distillation from LLMs, our approach improves performance by 17.3 scores on Bamboogle and 15.3 scores on HotpotQA, achieving LLM-level results across benchmarks. Our further analysis reveals that adaptive search strategies in SLMs often degrade performance, highlighting the necessity of consistent search behavior for reliable reasoning. 

---
# MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents 

**Authors**: Shu Wang, Edwin Yu, Oscar Love, Tom Zhang, Tom Wong, Steve Scargall, Charles Fan  

**Link**: [PDF](https://arxiv.org/pdf/2604.04853)  

**Abstract**: Large Language Model (LLM) agents require persistent memory to maintain personalization, factual continuity, and long-horizon reasoning, yet standard context-window and retrieval-augmented generation (RAG) pipelines degrade over multi-session interactions. We present MemMachine, an open-source memory system that integrates short-term, long-term episodic, and profile memory within a ground-truth-preserving architecture that stores entire conversational episodes and reduces lossy LLM-based extraction. MemMachine uses contextualized retrieval that expands nucleus matches with surrounding context, improving recall when relevant evidence spans multiple dialogue turns. Across benchmarks, MemMachine achieves strong accuracy-efficiency tradeoffs: on LoCoMo it reaches 0.9169 using gpt4.1-mini; on LongMemEvalS (ICLR 2025), a six-dimension ablation yields 93.0 percent accuracy, with retrieval-stage optimizations -- retrieval depth tuning (+4.2 percent), context formatting (+2.0 percent), search prompt design (+1.8 percent), and query bias correction (+1.4 percent) -- outperforming ingestion-stage gains such as sentence chunking (+0.8 percent). GPT-5-mini exceeds GPT-5 by 2.6 percent when paired with optimized prompts, making it the most cost-efficient setup. Compared to Mem0, MemMachine uses roughly 80 percent fewer input tokens under matched conditions. A companion Retrieval Agent adaptively routes queries among direct retrieval, parallel decomposition, or iterative chain-of-query strategies, achieving 93.2 percent on HotpotQA-hard and 92.6 percent on WikiMultiHop under randomized-noise conditions. These results show that preserving episodic ground truth while layering adaptive retrieval yields robust, efficient long-term memory for personalized LLM agents. 

---
# AI Trust OS -- A Continuous Governance Framework for Autonomous AI Observability and Zero-Trust Compliance in Enterprise Environments 

**Authors**: Eranga Bandara, Asanga Gunaratna, Ross Gore, Abdul Rahman, Ravi Mukkamala, Sachin Shetty, Sachini Rajapakse, Isurunima Kularathna, Peter Foytik, Safdar H. Bouk, Xueping Liang, Amin Hass, Ng Wee Keong, Kasun De Zoysa  

**Link**: [PDF](https://arxiv.org/pdf/2604.04749)  

**Abstract**: The accelerating adoption of large language models, retrieval-augmented generation pipelines, and multi-agent AI workflows has created a structural governance crisis. Organizations cannot govern what they cannot see, and existing compliance methodologies built for deterministic web applications provide no mechanism for discovering or continuously validating AI systems that emerge across engineering teams without formal oversight. The result is a widening trust gap between what regulators demand as proof of AI governance maturity and what organizations can demonstrate. This paper proposes AI Trust OS, a governance architecture for continuous, autonomous AI observability and zero-trust compliance. AI Trust OS reconceptualizes compliance as an always-on, telemetry-driven operating layer in which AI systems are discovered through observability signals, control assertions are collected by automated probes, and trust artifacts are synthesized continuously. The framework rests on four principles: proactive discovery, telemetry evidence over manual attestation, continuous posture over point-in-time audit, and architecture-backed proof over policy-document trust. The framework operates through a zero-trust telemetry boundary in which ephemeral read-only probes validate structural metadata without ingressing source code or payload-level PII. An AI Observability Extractor Agent scans LangSmith and Datadog LLM telemetry, automatically registering undocumented AI systems and shifting governance from organizational self-report to empirical machine observation. Evaluated across ISO 42001, the EU AI Act, SOC 2, GDPR, and HIPAA, the paper argues that telemetry-first AI governance represents a categorical architectural shift in how enterprise trust is produced and demonstrated. 

---
# Decocted Experience Improves Test-Time Inference in LLM Agents 

**Authors**: Maohao Shen, Kaiwen Zha, Zexue He, Zhang-Wei Hong, Siru Ouyang, J. Jon Ryu, Prasanna Sattigeri, Suhas Diggavi, Gregory Wornell  

**Link**: [PDF](https://arxiv.org/pdf/2604.04373)  

**Abstract**: There is growing interest in improving LLMs without updating model parameters. One well-established direction is test-time scaling, where increased inference-time computation (e.g., longer reasoning, sampling, or search) is used to improve performance. However, for complex reasoning and agentic tasks, naively scaling test-time compute can substantially increase cost and still lead to wasted budget on suboptimal exploration. In this paper, we explore \emph{context} as a complementary scaling axis for improving LLM performance, and systematically study how to construct better inputs that guide reasoning through \emph{experience}. We show that effective context construction critically depends on \emph{decocted experience}. We present a detailed analysis of experience-augmented agents, studying how to derive context from experience, how performance scales with accumulated experience, what characterizes good context, and which data structures best support context construction. We identify \emph{decocted experience} as a key mechanism for effective context construction: extracting essence from experience, organizing it coherently, and retrieving salient information to build effective context. We validate our findings across reasoning and agentic tasks, including math reasoning, web browsing, and software engineering. 

---
# Compliance-by-Construction Argument Graphs: Using Generative AI to Produce Evidence-Linked Formal Arguments for Certification-Grade Accountability 

**Authors**: Mahyar T. Moghaddam  

**Link**: [PDF](https://arxiv.org/pdf/2604.04103)  

**Abstract**: High-stakes decision systems increasingly require structured justification, traceability, and auditability to ensure accountability and regulatory compliance. Formal arguments commonly used in the certification of safety-critical systems provide a mechanism for structuring claims, reasoning, and evidence in a verifiable manner. At the same time, generative artificial intelligence systems are increasingly integrated into decision-support workflows, assisting with drafting explanations, summarizing evidence, and generating recommendations. However, current deployments often rely on language models as loosely constrained assistants, which introduces risks such as hallucinated reasoning, unsupported claims, and weak traceability. This paper proposes a compliance-by-construction architecture that integrates Generative AI (GenAI) with structured formal argument representations. The approach treats each AI-assisted step as a claim that must be supported by verifiable evidence and validated against explicit reasoning constraints before it becomes part of an official decision record. The architecture combines four components: i) a typed Argument Graph representation inspired by assurance-case methods, ii) retrieval-augmented generation (RAG) to draft argument fragments grounded in authoritative evidence, iii) a reasoning and validation kernel enforcing completeness and admissibility constraints, and iv) a provenance ledger aligned with the W3C PROV standard to support auditability. We present a system design and an evaluation strategy based on enforceable invariants and worked examples. The analysis suggests that deterministic validation rules can prevent unsupported claims from entering the decision record while allowing GenAI to accelerate argument construction. 

---
# FactReview: Evidence-Grounded Reviews with Literature Positioning and Execution-Based Claim Verification 

**Authors**: Hang Xu, Ling Yue, Chaoqian Ouyang, Libin Zheng, Shaowu Pan, Shimin Di, Min-Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04074)  

**Abstract**: Peer review in machine learning is under growing pressure from rising submission volume and limited reviewer time. Most LLM-based reviewing systems read only the manuscript and generate comments from the paper's own narrative. This makes their outputs sensitive to presentation quality and leaves them weak when the evidence needed for review lies in related work or released code. We present FactReview, an evidence-grounded reviewing system that combines claim extraction, literature positioning, and execution-based claim verification. Given a submission, FactReview identifies major claims and reported results, retrieves nearby work to clarify the paper's technical position, and, when code is available, executes the released repository under bounded budgets to test central empirical claims. It then produces a concise review and an evidence report that assigns each major claim one of five labels: Supported, Supported by the paper, Partially supported, In conflict, or Inconclusive. In a case study on CompGCN, FactReview reproduces results that closely match those reported for link prediction and node classification, yet also shows that the paper's broader performance claim across tasks is not fully sustained: on MUTAG graph classification, the reproduced result is 88.4%, whereas the strongest baseline reported in the paper remains 92.6%. The claim is therefore only partially supported. More broadly, this case suggests that AI is most useful in peer review not as a final decision-maker, but as a tool for gathering evidence and helping reviewers produce more evidence-grounded assessments. The code is public at this https URL. 

---
# CODE-GEN: A Human-in-the-Loop RAG-Based Agentic AI System for Multiple-Choice Question Generation 

**Authors**: Xiaojing Duan, Frederick Nwanganga, Chaoli Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03926)  

**Abstract**: We present CODE-GEN, a human-in-the-Loop, retrieval-augmented generation (RAG)-based agentic AI system for generating context-aligned multiple-choice questions to develop student code reasoning and comprehension abilities. CODE-GEN employs an agentic AI architecture in which a Generator agent produces multiple-choice coding comprehension questions aligned with course-specific learning objectives, while a Validator agent independently assesses content quality across seven pedagogical dimensions. Both agents are augmented with specialized tools that enhance computational accuracy and verify code outputs. To evaluate the effectiveness of CODE-GEN, we conducted an evaluation study involving six human subject-matter experts (SMEs) who judged 288 AI-generated questions. The SMEs produced a total of 2,016 human-AI rating pairs, indicating agreement or disagreement with the assessments of Validator, along with 131 instances of qualitative feedback. Analyses of SME judgments show strong system performance, with human-validated success rates ranging from 79.9% to 98.6% across the seven pedagogical dimensions. The analysis of qualitative feedback reveals that CODE-GEN achieves high reliability on dimensions well suited to computational verification and explicit criteria matching, including question clarity, code validity, concept alignment, and correct answer validity. In contrast, human expertise remains essential for dimensions requiring deeper instructional judgment, such as designing pedagogically meaningful distractors and providing high-quality feedback that reinforces understanding. These findings inform the strategic allocation of human and AI effort in AI-assisted educational content generation. 

---
# Beyond Retrieval: Modeling Confidence Decay and Deterministic Agentic Platforms in Generative Engine Optimization 

**Authors**: XinYu Zhao, ChengYou Li, XiangBao Meng, Kai Zhang, XiaoDong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03656)  

**Abstract**: Generative Engine Optimization (GEO) is rapidly reshaping digital marketing paradigms in the era of Large Language Models (LLMs). However, current GEO strategies predominantly rely on Retrieval-Augmented Generation (RAG), which inherently suffers from probabilistic hallucinations and the "zero-click" paradox, failing to establish sustainable commercial trust. In this paper, we systematically deconstruct the probabilistic flaws of existing RAG-based GEO and propose a paradigm shift towards deterministic multi-agent intent routing. First, we mathematically formulate Semantic Entropy Drift (SED) to model the dynamic decay of confidence curves in LLMs over continuous temporal and contextual perturbations. To rigorously quantify optimization value in black-box commercial engines, we introduce the Isomorphic Attribution Regression (IAR) model, leveraging a Multi-Agent System (MAS) probe with strict human-in-the-loop physical isolation to enforce hallucination penalties. Furthermore, we architect the Deterministic Agent Handoff (DAH) protocol, conceptualizing an Agentic Trust Brokerage (ATB) ecosystem where LLMs function solely as intent routers rather than final answer generators. We empirically validate this architecture using EasyNote, an industrial AI meeting minutes product by Yishu Technology. By routing the intent of "knowledge graph mapping on an infinite canvas" directly to its specialized proprietary agent via DAH, we demonstrate the reduction of vertical task hallucination rates to near zero. This work establishes a foundational theoretical framework for next-generation GEO and paves the way for a well-ordered, deterministic human-AI collaboration ecosystem. 

---
# Selective Forgetting for Large Reasoning Models 

**Authors**: Tuan Le, Wei Qian, Mengdi Huai  

**Link**: [PDF](https://arxiv.org/pdf/2604.03571)  

**Abstract**: Large Reasoning Models (LRMs) generate structured chains of thought (CoTs) before producing final answers, making them especially vulnerable to knowledge leakage through intermediate reasoning steps. Yet, the memorization of sensitive information in the training data such as copyrighted and private content has led to ethical and legal concerns. To address these issues, selective forgetting (also known as machine unlearning) has emerged as a potential remedy for LRMs. However, existing unlearning methods primarily target final answers and may degrade the overall reasoning ability of LRMs after forgetting. Additionally, directly applying unlearning on the entire CoTs could degrade the general reasoning capabilities. The key challenge for LRM unlearning lies in achieving precise unlearning of targeted knowledge while preserving the integrity of general reasoning capabilities. To bridge this gap, we in this paper propose a novel LRM unlearning framework that selectively removes sensitive reasoning components while preserving general reasoning capabilities. Our approach leverages multiple LLMs with retrieval-augmented generation (RAG) to analyze CoT traces, identify forget-relevant segments, and replace them with benign placeholders that maintain logical structure. We also introduce a new feature replacement unlearning loss for LRMs, which can simultaneously suppress the probability of generating forgotten content while reinforcing structurally valid replacements. Extensive experiments on both synthetic and medical datasets verify the desired properties of our proposed method. 

---
# An AI Teaching Assistant for Motion Picture Engineering 

**Authors**: Deirdre O'Regan, Anil C. Kokaram  

**Link**: [PDF](https://arxiv.org/pdf/2604.04670)  

**Abstract**: The rapid rise of LLMs over the last few years has promoted growing experimentation with LLM-driven AI tutors. However, the details of implementation, as well as the benefit in a teaching environment, are still in the early days of exploration. This article addresses these issues in the context of implementation of an AI Teaching Assistant (AI-TA) using Retrieval Augmented Generation (RAG) for Trinity College Dublin's Master's Motion Picture Engineering (MPE) course. We provide details of our implementation (including the prompt to the LLM, and code), and highlight how we designed and tuned our RAG pipeline to meet course needs. We describe our survey instrument and report on the impact of the AI-TA through a number of quantitative metrics. The scale of our experiment (43 students, 296 sessions, 1,889 queries over 7 weeks) was sufficient to have confidence in our findings. Unlike previous studies, we experimented with allowing the use of the AI-TA in open-book examinations. Statistical analysis across three exams showed no performance differences regardless of AI-TA access (p > 0.05), demonstrating that thoughtfully designed assessments can maintain academic validity. Student feedback revealed that the AI-TA was beneficial (mean = 4.22/5), while students had mixed feelings about preferring it over human tutoring (mean = 2.78/5). 

---
# Is a Picture Worth a Thousand Words? Adaptive Multimodal Fact-Checking with Visual Evidence Necessity 

**Authors**: Jaeyoon Jung, Yejun Yoon, Kunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2604.04692)  

**Abstract**: Automated fact-checking is a crucial task not only in journalism but also across web platforms, where it supports a responsible information ecosystem and mitigates the harms of misinformation. While recent research has progressed from text-only to multimodal fact-checking, a prevailing assumption is that incorporating visual evidence universally improves performance. In this work, we challenge this assumption and show that indiscriminate use of multimodal evidence can reduce accuracy. To address this challenge, we propose AMuFC, a multimodal fact-checking framework that employs two collaborative agents with distinct roles for the adaptive use of visual evidence: An Analyzer determines whether visual evidence is necessary for claim verification, and a Verifier predicts claim veracity conditioned on both the retrieved evidence and the Analyzer's assessment. Experimental results on three datasets show that incorporating the Analyzer's assessment of visual evidence necessity into the Verifier's prediction yields substantial improvements in verification performance. In addition to all code, we release WebFC, a newly constructed dataset for evaluating fact-checking modules in a more realistic scenario, available at this https URL. 

---
# Automated Conjecture Resolution with Formal Verification 

**Authors**: Haocheng Ju, Guoxiong Gao, Jiedong Jiang, Bin Wu, Zeming Sun, Leheng Chen, Yutong Wang, Yuefeng Wang, Zichen Wang, Wanyi He, Peihao Wu, Liang Xiao, Ruochuan Liu, Bryan Dai, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2604.03789)  

**Abstract**: Recent advances in large language models have significantly improved their ability to perform mathematical reasoning, extending from elementary problem solving to increasingly capable performance on research-level problems. However, reliably solving and verifying such problems remains challenging due to the inherent ambiguity of natural language reasoning. In this paper, we propose an automated framework for tackling research-level mathematical problems that integrates natural language reasoning with formal verification, enabling end-to-end problem solving with minimal human intervention. Our framework consists of two components: an informal reasoning agent, Rethlas, and a formal verification agent, Archon. Rethlas mimics the workflow of human mathematicians by combining reasoning primitives with our theorem search engine, Matlas, to explore solution strategies and construct candidate proofs. Archon, equipped with our formal theorem search engine LeanSearch, translates informal arguments into formalized Lean 4 projects through structured task decomposition, iterative refinement, and automated proof synthesis, ensuring machine-checkable correctness. Using this framework, we automatically resolve an open problem in commutative algebra and formally verify the resulting proof in Lean 4 with essentially no human involvement. Our experiments demonstrate that strong theorem retrieval tools enable the discovery and application of cross-domain mathematical techniques, while the formal agent is capable of autonomously filling nontrivial gaps in informal arguments. More broadly, our work illustrates a promising paradigm for mathematical research in which informal and formal reasoning systems, equipped with theorem retrieval tools, operate in tandem to produce verifiable results, substantially reduce human effort, and offer a concrete instantiation of human-AI collaborative mathematical research. 

---
# Agile Story-Point Estimation: Is RAG a Better Way to Go? 

**Authors**: Lamyea Maha, Tajmilur Rahman, Chanchal Roy  

**Link**: [PDF](https://arxiv.org/pdf/2604.03443)  

**Abstract**: The sprint-based iterative approach in the Agile software development method allows continuous feedback and adaptation. One of the crucial Agile software development activities is the sprint planning session where developers estimate the effort required to complete tasks through a consensus-based estimation technique such as Planning Poker. In the Agile software development method, a common unit of measuring development effort is Story Point (SP) which is assigned to tasks to understand the complexity and development time needed to complete them. Despite the benefits of this process, it is an extremely time-consuming manual process. To mitigate this issue, in this study, we investigated if this manual process can be automated using Retrieval Augmented Generation (RAG) which comprises a "Retriever" and a "Generator". We applied two embedding models - bge-large-en-v1.5, and Sentence-Transformers' all-mpnet-base-v2 on 23 open-source software projects of varying sizes and examined four key aspects: 1) how retrieval hyper-parameters influence the performance, 2) whether estimation accuracy differs across different sizes of the projects, 3) whether embedding model choice affects accuracy, and 4) how the RAG-based approach compares to the existing baselines. Although the RAG-based approach outperformed the baseline models in several occasions, our results did not exhibit statistically significant differences in performance across the projects or across the embedding models. This highlights the need for further studies and refinement of the RAG, and model adaptation strategies for better accuracy in automatically estimating user stories. 

---
# ExpressEdit: Fast Editing of Stylized Facial Expressions with Diffusion Models in Photoshop 

**Authors**: Kenan Tang, Jiasheng Guo, Jeffrey Lin, Yao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.03448)  

**Abstract**: Facial expressions of characters are a vital component of visual storytelling. While current AI image editing models hold promise for assisting artists in the task of stylized expression editing, these models introduce global noise and pixel drift into the edited image, preventing the integration of these models into professional image editing software and workflows. To bridge this gap, we introduce ExpressEdit, a fully open-source Photoshop plugin that is free from common artifacts of proprietary image editing models and robustly synergizes with native Photoshop operations such as Liquify. ExpressEdit seamlessly edits an expression within 3 seconds on a single consumer-grade GPU, significantly faster than popular proprietary models. Moreover, to support the generation of diverse expressions according to different narrative needs, we compile a comprehensive expression database of 135 expression tags enriched with example stories and images designed for retrieval-augmented generation. We open source the code and dataset to facilitate future research and artistic exploration. 

---
# AICCE: AI Driven Compliance Checker Engine 

**Authors**: Mohammad Wali Ur Rahman, Martin Manuel Lopez, Lamia Tasnim Mim, Carter Farthing, Julius Battle, Kathryn Buckley, Salim Hariri  

**Link**: [PDF](https://arxiv.org/pdf/2604.03330)  

**Abstract**: For digital infrastructure to be safe, compatible, and standards-aligned, automated communication protocol compliance verification is crucial. Nevertheless, current rule-based systems are becoming less and less effective since they are unable to identify subtle or intricate non-compliance, which attackers frequently use to establish covert communication channels in IPv6 traffic. In order to automate IPv6 compliance verification, this paper presents the Artificial Intelligence Driven Compliance Checker Engine (AICCE), a novel generative system that combines dual-architecture reasoning and retrieval-augmented generation (RAG). Specification segments pertinent to each query can be efficiently retrieved thanks to the semantic encoding of protocol standards into a high-dimensional vector space. Based on this framework, AICCE offers two complementary pipelines: (i) Explainability Mode, which uses parallel LLM agents to render decisions and settle disputes through organized discussions to improve interpretability and robustness, and (ii) Script Execution Mode, which converts clauses into Python rules that can be executed quickly for dataset-wide verification. With the debate mechanism enhancing decision reliability in complicated scenarios and the script-based pipeline lowering per-sample latency, AICCE achieves accuracy and F1-scores of up to 99% when tested on IPv6 packet samples across sixteen cutting-edge generative models. By offering a scalable, auditable, and generalizable mechanism for identifying both routine and covert non-compliance in dynamic communication environments, our results show that AICCE overcomes the blind spots of conventional rule-based compliance checking systems. 

---
# RAGnaroX: A Secure, Local-Hosted ChatOps Assistant Using Small Language Models 

**Authors**: Benedikt Dornauer, Mircea-Cristian Racasan  

**Link**: [PDF](https://arxiv.org/pdf/2604.03291)  

**Abstract**: This paper introduces RAGnaroX, a resource-efficient ChatOps assistant that operates entirely on commodity hardware. Unlike existing solutions that often rely on external providers such as Azure or OpenAI, RAGnaroX offers a fully auditable, on-premise stack implemented in Rust. Its architecture integrates modular data ingestion, hybrid retrieval, and function calling, enabling flexible yet secure deployment. Our evaluation focuses on the RAG pipeline, with benchmarks conducted on the SQuAD (single-hop QA), MultiHopRAG (multi-hop QA), and MLQA (cross-lingual QA) datasets. Results show that RAGnaroX achieves competitive accuracy while maintaining strong resource efficiency, for example, reaching 0.90 context precision on single-hop questions with an average response time of 2.5 seconds per request. A replication package containing the tool, the demonstration video (this https URL v=cDxfuEbcoM4), and all supporting materials are available at this https URL. 

---
# SafeScreen: A Safety-First Screening Framework for Personalized Video Retrieval for Vulnerable Users 

**Authors**: Wenzheng Zhao, Madhava Kalyan Gadiputi, Fengpei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2604.03264)  

**Abstract**: Open-domain video platforms offer rich, personalized content that could support health, caregiving, and educational applications, but their engagement-optimized recommendation algorithms can expose vulnerable users to inappropriate or harmful material. These risks are especially acute in child-directed and care settings (e.g., dementia care), where content must satisfy individualized safety constraints before being shown. We introduce SafeScreen, a safety-first video screening framework that retrieves and presents personalized video while enforcing individualized safety constraints. Rather than ranking videos by relevance or popularity, SafeScreen treats safety as a prerequisite and performs sequential approval or rejection of candidate videos through an automated pipeline. SafeScreen integrates three key components: (i) profile-driven extraction of individualized safety criteria, (ii) evidence-grounded assessments via adaptive question generation and multimodal VideoRAG analysis, and (iii) LLM-based decision-making that verifies safety, appropriateness, and relevance before content exposure. This design enables explainable, real-time screening of uncurated video repositories without relying on precomputed safety labels. We evaluate SafeScreen in a dementia-care reminiscence case study using 30 synthetic patient profiles and 90 test queries. Results demonstrate that SafeScreen prioritizes safety over engagement, diverging from YouTube's engagement-optimized rankings in 80-93% of cases, while maintaining high levels of safety coverage, sensibleness, and groundedness, as validated by both LLM-based evaluation and domain experts. 

---
