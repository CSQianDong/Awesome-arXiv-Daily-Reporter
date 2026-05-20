# Divergence Meets Consensus: A Multi-Source Negative Sampling Framework for Sequential Recommendation 

**Authors**: Yuanzi Li, Lingjie Wang, Jingyu Zhao, Zihang Tian, Yuhan Wang, Lei Wang, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.19651)  

**Abstract**: Negative sampling is significant for training sequential recommendation models under implicit feedback. The predominant strategy, self-guided hard negative sampling, selects negatives based on the model's current state but suffers from three limitations: (1) the coupling between sampling and model updates triggers a vicious cycle that drives the model into local optima; (2) relying on current model parameters narrows sampling to a small region of the item space, reducing diversity and harming generalization; (3) identifying a hard negative requires scoring the entire candidate pool, causing substantial computational overhead with minimal information gain. To address these challenges, we propose MDCNS (Multi-source Divergence-Consensus for Negative Sampling), a novel "Teacher-Peer-Self" framework inspired by Vygotsky's Zone of Proximal Development (ZPD) theory. The proposed method comprises three components, including multi-source scoring, divergence re-ranking, and consensus distillation. Firstly, multi-source scoring incorporates peer and ensemble teacher models to inject external negative signals and break the self-reinforcement loop. Then, divergence re-ranking exploits prediction discrepancy between self and peer models to enhance sampling diversity. Finally, consensus distillation aligns the self model with the teacher via KL divergence, simultaneously improving computational cost utilization. Extensive experiments on six real-world datasets and five backbone models show that MDCNS consistently outperforms state-of-the-art negative sampling methods, demonstrating strong effectiveness and generalization. 

---
# Understanding Wacky Weights: A Dissection of SPLADE's Learned Term Importance 

**Authors**: Gregory Polyakov, Harrisen Scells, Carsten Eickhoff  

**Link**: [PDF](https://arxiv.org/pdf/2605.19628)  

**Abstract**: Learned sparse retrieval models such as SPLADE combine the effectiveness of neural architectures with the efficiency of inverted indices. As these models assign weights to terms from a fixed vocabulary, interpretability is often touted as a major benefit of these models. However, the emergence of wacky weights, i.e., expansion terms that appear semantically unrelated to the input, limits interpretability. While prior research has anecdotally observed this phenomenon, there is a lack of systematic understanding regarding their origins, prevalence, and contribution to retrieval effectiveness. In this paper, we reproduce SPLADE-v2 to systematically investigate wacky weights across the SPLADE family of models. We present a comprehensive dissection of wacky weights, providing a formal definition of wackiness based on the lexical utility of expansion terms. Furthermore, we introduce a novel measure to compare the prevalence of these tokens across models with varying vocabularies and sparsity levels. Beyond reproducing the original SPLADE-v2, we train it with various loss functions, datasets, and backbone transformers to isolate the factors contributing to wackiness. Our results show that larger vocabularies are associated with a higher prevalence of wacky tokens, while stricter sparsity regularizers are associated with lower prevalence. Finally, we find that wacky weights are used primarily for in-domain effectiveness rather than out-of-domain generalization. 

---
# SynGR: Unleashing the Potential of Cross-Modal Synergy for Generative Recommendation 

**Authors**: Wei Chen, Xingyu Guo, Shuang Li, Fuwei Zhang, Meng Yuan, Jing Fan, Zhao Zhang, Deqing Wang, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18920)  

**Abstract**: Generative Recommendation (GR) has emerged as a promising paradigm by formulating item recommendation as a sequence-to-sequence generation task over item identifiers. Recent studies have incorporated multimodal signals to provide richer token-level evidence for generation. However, existing approaches largely rely on alignment-centric fusion and underexplore synergistic information across modalities. In practice, synergistic information plays a critical role in capturing emergent item properties that cannot be inferred from any single modality alone. Such properties encode intrinsic item semantics and guide user preferences, enabling models to move beyond surface-level feature matching. To address this limitation, we propose \textbf{SynGR}, a synergistic generative recommendation framework that explicitly encourages the exploitation of cross-modal dependencies during generation. By constraining overreliance on dominant modalities, SynGR enables the model to capture emergent item semantics beyond shared or modality-specific signals. Extensive experiments across three benchmark datasets demonstrate that SynGR achieves superior performance. 

---
# The 99% Success Paradox: When Near-Perfect Retrieval Equals Random Selection 

**Authors**: Vyzantinos Repantis, Harshvardhan Singh, Tony Joseph, Cien Zhang, Akash Vishwakarma, Svetlana Karslioglu, Michael Wyatt Thot, Ameya Gawde  

**Link**: [PDF](https://arxiv.org/pdf/2605.18857)  

**Abstract**: For most of the history of information retrieval (IR), search results were designed for human consumers who could scan, filter, and discard irrelevant information on their own. This shaped retrieval systems to optimize for finding and ranking more relevant documents, but not keeping results clean and minimal, as the human was the final filter. However, LLMs have changed that by lacking this filtering ability.
To address this, we introduce Bits-over-Random (BoR), a chance-corrected measure of retrieval selectivity that reveals when high success rates mask random-level performance. We measure selectivity as $BoR = \log_{2}\left(\frac{\mathrm{P}_{obs}}{\mathrm{P}_{rand}}\right)$, where $\mathrm{P}_{rand}$ is the hypergeometric baseline for the chosen success rule (here, coverage: $ \geq1 $ relevant in top-$K$).
On the 20 Newsgroups dataset, BM25 and SPLADE both report $>99$% success at $K=100$ (coverage), yet $BoR \approx 0$, indicating random-level selectivity at that depth. When the expected coverage ratio $\left(\frac{K \cdot \bar{R}_{q}}{N}\right)$ exceeds 3-5, the baseline dominates and selectivity collapses. Downstream retrieval-augmented generation (RAG) evaluation confirms this pattern: LLM accuracy can degrade substantially at $K=100$, consistent with the near-zero BoR ceiling. In contrast, BoR remains positive on BEIR/SciFact and on MS MARCO (where 41 systems cluster within 0.2 bits of the theoretical ceiling despite a 13-point recall gap), confirming baseline predictions across sparse and large-scale settings. We further show that the collapse boundary applies to LLM agent tool selection, where small catalog sizes cause selectivity to vanish even with perfect selectors. These findings suggest reporting BoR alongside traditional metrics and reconsidering depth choices when additional retrieval provides negligible selectivity gains while inflating computational costs. 

---
# KadiAssistant: A conversational AI Agent for information retrieval in Kadi4Mat 

**Authors**: Adrian Cierpka, Mohammad Shafiqul Islam, Johannes Steinhülb, Eric Dietriche Sesso Domtchoueng, Michael Selzer, Arnd Koeppe  

**Link**: [PDF](https://arxiv.org/pdf/2605.18850)  

**Abstract**: We introduce KadiAssistant, a privacy-by-design AI assistant integrated into the Kadi research data ecosystem, enabling researchers to efficiently access, aggregate, and synthesize information from heterogeneous, privacy-sensitive research data. Interdisciplinary fields such as materials science bring together disciplines with their own terminology and standards. While this convergence fuels innovation, it also makes it increasingly difficult to connect and access knowledge, as data are distributed across disciplines, organizations, and individuals. For example, battery research combines electrochemical measurements, materials characterization data, physics-based simulations, and manufacturing parameters, each using different formats, vocabularies, and standards. Efficiently storing and sharing such heterogeneous data via research data platforms, such as Kadi4Mat, demands domain knowledge, technical expertise, and familiarity with metadata schemas and interfaces. Research data also vary in sensitivity: newly generated 'warm' data are often private, whereas published 'cold' data are usually openly accessible. The Kadi ecosystem offers fine-grained access control needed for sensitive data. A solution for efficient information retrieval in Kadi must therefore respect the fine-grained access permissions. To address these intertwined challenges of information retrieval, strong data privacy, and complex access control, KadiAssistant combines a self-hosted large language model (LLM) with a privacy-preserving semantic search, inspired by retrieval-augmented generation, that can access files and record metadata on Kadi. This allows the assistant to screen, aggregate, and structure information into a highly informative answer. KadiAssistant therefore bridges terminology and standards, lowers access barriers for researchers, and strengthens the Findable pillar of FAIR data principles. 

---
# Code-Guided Reasoning for Small Language Models: Evaluating Executable MCQA Scaffolds 

**Authors**: Prateek Biswas, Dhaval Patel, Vedant Khandelwal, Shuxin Lin, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2605.18827)  

**Abstract**: Multiple-choice QA benchmarks usually evaluate small language models (SLMs) as direct answerers, but deployed language-model systems increasingly rely on external scaffolds such as tools, code, and repeated model calls. We introduce Code-Guided Reasoning (CGR), an evaluation protocol and generated-program resource for measuring when executable reasoning scaffolds improve SLM performance on MCQA tasks. CGR standardizes six components: a normalized item interface, a direct solver prompt, a generator prompt, a Python scaffold, solver-call and extraction helpers, and a three-channel result record. On 20,498 retained result rows from a locally prepared MCQA bundle and six metadata-registered solver models, the observed non-zero-baseline partition shows 66.21% macro assisted accuracy versus 38.11% direct accuracy, a +28.10 percentage-point difference with a pair-bootstrap interval of [20.32, 36.43]. Under a stricter Ab > 30% direct-signal gate, the macro difference is +14.11 points. These estimates are descriptive. Assisted inference uses a larger solver-call budget, answer extraction is brittle, Time-MQA contains the observed regressions, and some generated programs violate the no-hard-coding instruction. CGR provides the trace package needed to interpret these results, including direct, assisted, and generator-side answers, partition definitions, generated programs, response metadata, and audits. 

---
# Towards FairRAG: Preventing Representational Harm in Retrieval-Augmented Generation by Enforcing Fair Exposure at Retrieval Time 

**Authors**: Riddhi Tikoo  

**Link**: [PDF](https://arxiv.org/pdf/2605.18806)  

**Abstract**: As Large Language Model (LLM) integration has accelerated in high-stakes domains, model hallucination is a critical issue. Retrieval-augmented generation (RAG) is a technique for addressing hallucination; however, RAG's multi-component pipeline introduces vulnerabilities where biases can be introduced. This study considers two previously developed utility-focused ranking strategies (Standard and Stochastic) alongside two proposed exposure-aware approaches (Forced-Exposure and Representative Stochastic). Using the TREC 2022 Fair Ranking Dataset, which contains Wikipedia articles annotated as protected or non-protected, the LLM was asked to identify relevant articles with citations for four scenario-based Q&A prompts. The retrieval rankings and the generated outputs were evaluated for exposure bias and utility across all ranking methods. Overall, the Representative Stochastic ranker resulted in a statistically significant near-parity average exposure, acknowledging that relevance scores initially produced during retrieval are already shaped by representational bias, whereas the other rankers assume those scores are unbiased. Across all the methods of document ranking, generation demographic parity closely mirrored the exposure parity, reinforcing that representational bias in RAG systems is driven by retrieval and propagates to generation. These findings highlight that retrieval ranking is a critical point for mitigating downstream bias and propose a Representative Stochastic ranker that reintroduces fairness in RAG systems. 

---
# RecoAtlas: From Semantic Plausibility to Set-Level Utility in LLM Recommendation Agents 

**Authors**: Imad Aouali, Flavian Vasile, Otmane Sakhi, Alexandre Gilotte, Benjamin Heymann  

**Link**: [PDF](https://arxiv.org/pdf/2605.18805)  

**Abstract**: LLM recommendation agents increasingly produce structured recommendation reports: sets of items accompanied by natural-language justifications. Yet existing evaluations often reduce this setting to reranking small shortlisted candidate sets or judge reports mainly by semantic plausibility. We introduce Recommendation Atlas (Agentic Tool-Level Assessment for Shopping), or RecoAtlas, a benchmark and toolkit for evaluating shopping agents with behavior-grounded metrics. RecoAtlas complements held-out interaction metrics with learned utility proxies for relevance, complementarity, and diversity derived from interaction data, while separately measuring semantic coherence and explanation quality. Its controlled tool environment exposes agents to either semantic, behavior-aligned, or faulty tools, enabling diagnosis of whether performance gains arise from stronger reasoning, better signals, or more effective tool-use policies. Across controlled experiments, we show that RecoAtlas exhibits key properties of a meaningful benchmark for agentic systems: performance scales with model capacity and test-time compute, improves with stronger and better-aligned tools, degrades under noisy or misaligned signals, and reveals that semantic plausibility does not necessarily capture behavior-grounded utility. RecoAtlas provides a foundation for developing and evaluating shopping assistants that optimize not only for plausible recommendations, but also for coherent, behaviorally grounded recommendation sets. 

---
# Trust or Abstain? A Self-Aware RAG Approach 

**Authors**: Xi Zhu, Ziqi Wang, Kai Mei, Wujiang Xu, Minghao Guo, Bangji Yang, Jiajun Fan, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2605.18792)  

**Abstract**: Retrieval-augmented generation (RAG) improves large language models (LLMs) by incorporating external evidence, but it also introduces knowledge conflicts when retrieved contextual knowledge (CK) and parametric knowledge (PK) disagree or are both unreliable. Existing approaches mainly coordinate which source to use, without explicitly asking whether each answer path is correct. We argue that faithful RAG requires LLM self-awareness, namely the ability to recognize the limits of its own knowledge and reasoning. To ground this problem, we construct a model-specific, ground-truth-aligned knowledge-conflict benchmark by evaluating LLM backbones on PK-only and CK-conditioned answer paths over approximately 69K query-context instances per backbone, drawn from five conflict-QA datasets. We then introduce SABER, a Self-Aware Belief Estimator for RAG that requires no LLM fine-tuning. SABER combines a self-prior with PK-side and CK-side conditional reasoning representations from multi-trace inference, then estimates reliability beliefs with two lightweight predictors to drive a 4-cell decision over trust PK, trust CK, trust either, or abstain. Across four LLM backbones, SABER improves end-to-end accuracy and conflict-specific faithfulness over ten inference-time and fine-tuning baselines, with the largest gains on conflict-heavy datasets. Under abstention, SABER's risk-coverage curve Pareto-dominates every prompt-based abstainer, providing a tunable balance between coverage and answer risk. Our code is available at this https URL. 

---
# A Reproducibility Analysis of PO4ISR: Diagnosing and Mitigating Semantic Drift in LLM-Based Session Recommendation 

**Authors**: Aditya Tiwari, Konduri Naga Lakshmi Rekha, Rajesh Kumar Mundotiya  

**Link**: [PDF](https://arxiv.org/pdf/2605.18780)  

**Abstract**: Reasoning-based Large Language Models (LLMs) like PO4ISR have set new benchmarks in session-based recommendation. However, the reproducibility of their reasoning capabilities across diverse semantic domains remains unexplored. In this work, we conduct a rigorous reproducibility study of PO4ISR to assess its generalization limits. Our analysis reveals a critical failure mode: standard reasoning prompts suffer from severe contextual drift in long sessions, leading to performance degradation on semantically complex datasets like Games and Bundle. To quantify and resolve this stability gap, we introduce PO4ISR++, a robustness-enhanced implementation that integrates reflexive prompting and consistent rank detection. Unlike the original static prompting strategy, our approach dynamically adapts to cross-domain cues. We benchmark both the original implementation and our robust variant on ML-1M, Games, and Bundle. Our results confirm that while the original model struggles in new domains, our reproducible extension restores performance, yielding a stabilized gain of up to 54% on Games and 96% on Bundle. We release open-source artifacts, including the reproduced baseline and our enhanced framework, to facilitate reliable future research in LLM-based recommendation. 

---
# Mask-to-Correct$^+$: Leveraging Retriever Diversity for Masking-guided Faithful Fact Correction 

**Authors**: Payel Santra, Lavisha Sharma, Madhusudan Ghosh, Partha Basuchowdhuri  

**Link**: [PDF](https://arxiv.org/pdf/2605.18776)  

**Abstract**: The rapid spread of misinformation on social media highlights the need for robust, automated fact correction frameworks. However, existing works rely on supervised learning from manually annotated claim-evidence pairs, which are scarce and prone to biases, limiting their generalization across domains. Moreover, these methods overlook semantic faithfulness in their correction process. To address these challenges, we propose Mask-to-Correct (M$_2$C), a training-free, inference-only Retrieval Augmented Generation (RAG) based framework that leverages diversity-aware masking to identify erroneous spans of claims and evaluate the faithfulness of corrections using retrieved evidence. However, the effectiveness of RAG heavily depends on the choice of retriever, which may vary across queries. To mitigate this, we further introduce M$_2$C$^+$, an ensemble-based framework that combines corrections across multiple rankers to reduce retrieval bias and improve robustness. Extensive experiments on the benchmark datasets demonstrate that our proposed frameworks consistently outperform all baselines, achieving up to 14% improvement in SARI scores, without using gold evidence. 

---
# Query-Aware Flow Diffusion for Graph-Based RAG with Retrieval Guarantees 

**Authors**: Zhuoping Zhou, Davoud Ataee Tarzanagh, Sima Didari, Wenjun Hu, Baruch Gutow, Oxana Verkholyak, Masoud Faraki, Heng Hao, Hankyu Moon, Seungjai Min  

**Link**: [PDF](https://arxiv.org/pdf/2605.18775)  

**Abstract**: Graph-based Retrieval-Augmented Generation (RAG) systems leverage interconnected knowledge structures to capture complex relationships that flat retrieval struggles with, enabling multi-hop reasoning. Yet most existing graph-based methods suffer from (i) heuristic designs lacking theoretical guarantees for subgraph quality or relevance and/or (ii) the use of static exploration strategies that ignore the query's holistic meaning, retrieving neighborhoods or communities regardless of intent. We propose Query-Aware Flow Diffusion RAG (QAFD-RAG), a training-free framework that dynamically adapts graph traversal to each query's holistic semantics. The central innovation is query-aware traversal: during graph exploration, edges are dynamically weighted by how well their endpoints align with the query's embedding, guiding flow along semantically relevant paths while avoiding structurally connected but irrelevant regions. These query-specific reasoning subgraphs enable the first statistical guarantees for query-aware graph retrieval, showing that QAFD-RAG recovers relevant subgraphs with high probability under mild signal-to-noise conditions. The algorithm converges exponentially fast, with complexity scaling with the retrieved subgraph size rather than the full graph. Experiments on question answering and text-to-SQL tasks demonstrate consistent improvements over state-of-the-art graph-based RAG methods. 

---
# M3DocDep: Multi-modal, Multi-page, Multi-document Dependency Chunking with Large Vision-Language Models 

**Authors**: Joongmin Shin, Jeongbae Park, Jaehyung Seo, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2605.18774)  

**Abstract**: In long, multi-page industrial documents, retrieval-augmented generation (RAG) depends heavily on whether chunk boundaries follow the document's true structure. Existing text-centric chunkers and generative hierarchy parsers often miss cross-page parent-child relations, figure/table-caption bindings, and boundary cues, which leads to fragmented or redundant chunks and degrades both retrieval and answer quality. We propose M3DocDep, an LVLM-based pipeline that first recovers block-level dependencies and then constructs chunks along the recovered document tree. The pipeline uses SharedDet as a common DP+OCR preprocessing layer, extracts multimodal block embeddings with boundary-aware SoftROI pooling, scores candidate parent-child edges with a biaffine head, decodes a globally valid dependency tree with MST constraints, and builds tree-guided chunks annotated with section paths and page ranges. Under a shared-block evaluation protocol, M3DocDep improves STEDS by +28.5 to +39.6 percent on DHP benchmarks, retrieval nDCG by +1.1 to +15.3 percent, and QA ANLS by +4.5 to +15.3 percent on corpus-level RAG benchmarks. These results show that recovering document dependencies before chunking yields more coherent retrieval units for long, multi-page multimodal documents. 

---
# Improving Retrieval-Augmented Generation without Taxonomy-based Error Categorization 

**Authors**: Gongbo Zhang, Yifan Peng, Chunhua Weng  

**Link**: [PDF](https://arxiv.org/pdf/2605.18772)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves the factual accuracy of large language model (LLM) outputs by grounding generation in external knowledge. Recent agentic RAG systems extend this paradigm with critical agents to evaluate model responses and iteratively refine outputs. However, most prior work implicitly assumes reliable critic feedback and focuses on planning strategies, while paying limited attention to the robustness of the error-correction process itself, which can be impacted by misaligned error categories and ineffective or incorrect corrections. Here, we hypothesize that RAG performance can be improved without explicit error categorization. We propose RePAIR, a response-action learning paradigm that directly maps flawed RAG outputs to error-mitigating action plans without relying on fine-grained error taxonomies and explicit critic supervision. Across multiple benchmarks, RePAIR consistently improves agentic RAG performance. 

---
# LWGR: Lagrangian-Constrained Personalized World Knowledge for Generative Recommendation 

**Authors**: Lingyu Mu, Hao Deng, Haibo Xing, Kaican Lin, Zhitong Zhu, Yu Zhang, Xiaoyi Zeng, Zhengxiao Liu, Zheng Lin, Jinxin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.18771)  

**Abstract**: Recent progress in large language model (LLM) based generative recommendation (GR) shows that leveraging LLM world knowledge can substantially improve performance. However, existing methods rely on fixed, manually designed instructions to generate semantic knowledge and directly incorporate it into GR, which has two limitations. First, fixed instructions cannot capture the multidimensional heterogeneity of user interests. Second, uncontrollable knowledge fusion may conflict with behavioral signals and harm recommendations. To address these limitations, we propose LWGR, a framework that leverages Lagrangian constraints to transfer users' personalized world knowledge from LLMs into generative recommendation. LWGR enhances GR along two axes: knowledge extraction and fusion. It builds personalized soft instructions to extract behavior-relevant LLM world knowledge, and formulates knowledge fusion as an optimization problem with explicitly bounded performance degradation, which is solved by a Lagrangian primal-dual method to selectively incorporate beneficial knowledge. We further design two training strategies for different LLM scales and a deployment scheme that combines nearline precomputation with lightweight online serving. Experiments on multiple public datasets and one industrial dataset show that LWGR outperforms eight state-of-the-art baselines by up to 11.23% and brings a 1.35% revenue lift on a large-scale advertising platform, demonstrating its effectiveness and practicality. 

---
# Agentic GraphRAG: Navigating Unstructured Financial Data with Collaborative AI 

**Authors**: Arthur Capozzi, Dirk Helbing  

**Link**: [PDF](https://arxiv.org/pdf/2605.18770)  

**Abstract**: We present a collaborative agentic GraphRAG framework for expert analysis of commercial registry data. Public registries are often formally accessible, yet difficult to use in practice because they combine structured records with large volumes of unstructured legal text. This limits conventional keyword and vector-only retrieval, especially for multi-hop, temporal, and entity-centric investigations. Our approach builds a Neo4j knowledge graph through a three-phase pipeline: (i) deterministic ingestion of strong nodes from verified structured fields, (ii) LLM-based extraction of weak nodes from unstructured notices, and (iii) deterministic identity resolution and deduplication. On top of this graph, we introduce an analytical modular agent that integrates zero-shot intent routing, a bounded reflection loop, secure tool-mediated graph access, and state-aware response synthesis. A human-in-the-loop dashboard exposes evidence and execution traces to support transparency and auditability. We evaluate the framework on the Swiss Official Gazette of Commerce, a multilingual corpus of more than seven million publications over seven years. We further contribute a multi-tier evaluation protocol covering entity-resolution precision, tool-routing behavior, answer quality, and multi-turn conversational performance. Across automated, human-curated, and conversational benchmarks, the proposed agentic GraphRAG system consistently outperforms a standard agentic vector-RAG baseline, with strong gains in correctness, answer relevance, information recall, turn success rate, and context carryover accuracy. The architecture is modular, reproducible, and transferable to other commercial gazettes and public-sector registry systems. 

---
# ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation 

**Authors**: Gibson Nkhata, Uttamasha Anjally Oyshi, Quan Mai, Susan Gauch  

**Link**: [PDF](https://arxiv.org/pdf/2605.18769)  

**Abstract**: Personalized Retrieval-Augmented Generation (RAG) relies on accurately selecting user-relevant documents. In practice, existing RAG approaches often suffer from high retrieval costs and overlook that collaborative signals from similar users can enhance personalized generation for the current user. We propose ClusterRAG, a Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation. ClusterRAG represents users through their profile documents, organizes users into semantically coherent clusters using density-based clustering, and performs retrieval at both the cluster and document levels via cluster-level similarity and fine-grained ranking. Extensive experiments on the LaMP benchmark demonstrate that jointly leveraging the target user's profile and profiles from top similar users consistently yields the best performance across diverse tasks. Further analysis shows that ClusterRAG integrates seamlessly with different dense retrievers and rankers, and remains effective when paired with both fine-tuned and zero-shot language models. 

---
# ClinQueryAgent: A Conversational Agent for Population Health Management 

**Authors**: Joseph S. Boyle, Anthony Dranfield, Mike O'Neil, Maria Liakata, Alison Q. Smithard  

**Link**: [PDF](https://arxiv.org/pdf/2605.18768)  

**Abstract**: In this paper we introduce ClinQueryAgent, a system for translating natural language population health questions into executable database queries using agents with access to both local and external knowledge bases. Our novel architecture enables the use of powerful cloud-based language models whilst ensuring that no patient data leaves the secure environment. To combat inaccuracies over the course of longer dialogues due to context rot, information retrieval is delegated to a sub-agent. We deploy the system via a chat window embedded within an existing population health management platform where it has been used by 128 staff from 15 healthcare practices covering a total of 148,319 patients in the UK's National Health Service (NHS). We evaluate the system's capacity to autonomously handle a range of health informatics tasks on a constructed dataset and via a beta-testing phase. Our results show that both analysts and clinicians are able to easily generate actionable information from patient health records using natural language requests requiring no programming expertise to verify. We make a public demo of the system available at: this https URL 

---
# DualView: Adaptive Local-Global Fusion for Multi-Hop Document Reranking 

**Authors**: Litong Zhang, Jiaxin Li, Kuo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.18767)  

**Abstract**: Multi-hop question answering requires aggregating information from multiple documents, a critical capability for knowledge-intensive applications. A fundamental challenge lies in efficiently identifying the minimal relevant document set from retrieved candidates while maintaining high recall.
We present an efficient dual-view cascaded reranking framework for multi-hop document reranking. Operating as a lightweight post-retrieval stage over E5-base-v2 candidates, our architecture comprises: (1) a Local Scorer employing stacked cross-attention for fine-grained query-document relevance; and (2) a Global Scorer modeling inter-document dependencies via Transformer-based context aggregation. These views are dynamically fused through an Adaptive Gate conditioned on query semantics.
Under the fixed candidate set reranking setting with offline cached embeddings, our model achieves competitive results, particularly outstanding on MuSiQue with 99.4% Top-4 Recall and 97.8% Full Hit accuracy at 4.0 ms latency (249 QPS). It substantially outperforms 600M-parameter cross-encoders (BGE-Large: 92.0% Recall, Jina-v3: 90.1% Recall) while maintaining 5 to 6 times lower latency. Ablation studies validate that both Local and Global views contribute substantially to multi-hop performance. 

---
# Retrieve Only Relevant Tables Whether Few or Many: Adaptive Table Retrieval Method 

**Authors**: Taehee Kim, Seungbin Yang, Jihwan Kim, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2605.18766)  

**Abstract**: Retrieving relevant tables from extensive databases for a given natural language query is essential for accurately answering questions in tasks such as text-to-SQL. Existing table retrieval approaches select a pre-determined set of k tables with the highest similarity to the query. However, the number of required tables varies across queries and cannot be known in advance. Enforcing a fixed number of retrieved tables regardless of the query may either retrieve an undersized set, failing to obtain all necessary evidence, or retrieve an oversized pool, including irrelevant tables. To address this issue, we propose an adaptive table retrieval method that adjusts the number of tables retrieved according to the requirements of each query. Specifically, we utilize an adaptive thresholding mechanism to selectively retrieve tables and integrate a sliding-window reranking algorithm to efficiently process a large table corpus. Extensive experiments on Spider, BIRD, and Spider 2.0 demonstrate that our method effectively addresses the limitations of the top-k retrieval strategy, improving performance in retrieval and downstream tasks. Our code and data are available at this https URL. 

---
# STAR: Semantic-Tuned and Tail-Adaptive Retriever for Graph-Augmented Generation 

**Authors**: Shuai Li, Chen Huang, Duanyu Feng, Wenqiang Lei, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2605.18765)  

**Abstract**: To augment Large Language Models (LLMs) for multi-hop question answering, a mainstream solution within Graph Retrieval Augmented Generation (GraphRAG) leverages lightweight retrievers to efficiently extract information from a given Knowledge Graph (KG). However, existing methods often overlook the inherent challenge of sparse semantic information in graphs. Specifically, our experiments reveal that these methods produce biased retrieval Semantic Shortcut Bias and Long-Tail Path Bias, leading to inadequate semantic modeling and limited GraphRAG effectiveness. To address these issues, we propose STAR, a semantic-tuned and tail-adaptive retriever for GraphRAG. STAR integrates two key learning paradigms: token-level interaction learning and path-weighted contrastive learning. The former employs a cross-attention architecture and a hard path mining mechanism to jointly model the query and path, thereby mitigating the Semantic Shortcut Bias. The latter introduces a tailored contrastive learning objective that utilizes tail-adaptive path weighting, designed to optimize the training process and ease the Long-Tail Path Bias. Extensive experiments demonstrate that STAR consistently outperforms baselines, achieving average retrieval performance gains of 1.8\% and LLM QA performance improvements of 2.2\% across all benchmark datasets. Our code is available at this https URL. 

---
# From Intent to AI Pipelines: A Controlled Agentic Framework for Non-AI Expert Scientists 

**Authors**: Hyacinth Ali, Jessie Galasso-Carbonnel, Houari Sahraoui  

**Link**: [PDF](https://arxiv.org/pdf/2605.18764)  

**Abstract**: Artificial Intelligence (AI) pipelines have become integral to modern research, supporting fields such as Medical Sciences, Agriculture, and Social Sciences, and enabling large-scale data analysis, predictive modeling, and the automation of complex tasks. However, designing and implementing AI solutions remains challenging for many researchers due to the expertise required in the design and development of end-to-end AI systems. To address this gap, we present Domain-Driven Adaptable AI Pipelines (DDAP), a controlled, human-in-the-loop, agentic framework that leverages large language models to guide users in a systematic construction of AI pipelines and their corresponding implementation code. DDAP structures the development process into four stages: problem definition, compute environment specification, pipeline generation, and code generation. Through this staged interaction, the framework adapts to domain context, user expertise, and resource constraints, while maintaining user control over key decisions. We evaluate DDAP across multiple datasets spanning business, biology, and health science domains by comparing its AI models against expert-developed models. The experimental results show that DDAP achieves competitive results in several tasks compared to expert baselines, although performance varies across problem types, particularly for text-based clustering tasks. By combining guided interaction, adaptability, and reproducibility, DDAP demonstrates that a controlled agentic framework can generate competitive AI pipelines for non-expert users. 

---
# Query-Conditioned Graph Retrieval for Contextualized LLM Reasoning in Personalized Wearable Data 

**Authors**: Zhenyu Lu, Mahyar Abbasian, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2605.18763)  

**Abstract**: Large language models (LLMs) are increasingly applied to analyzing wearable sensing data, which are long-term, multimodal, and highly personalized. A key challenge is context selection: providing insufficient context limits reasoning, while including all available data leads to inefficiency and degraded generation quality. We propose Wearable As Graph (WAG), a graph-based context retrieval framework that enables query-adaptive reasoning over wearable data with LLMs. WAG organizes wearable metrics and user-specific signals into a personalized knowledge graph, and retrieves a query-conditioned subgraph to support downstream generation. The retrieval process integrates global relationships, capturing prior knowledge and population- and individual-level patterns via hierarchical Bayesian modeling, with local relationships that reflect short-term signal deviations. A query openness signal further controls retrieval breadth. We evaluate WAG on over 10,000 data-grounded queries from real-world wearable datasets. Across LLM-based and human evaluations, WAG achieves an approximately 70% win rate over baseline and standard RAG methods, demonstrating the effectiveness of structured, query-adaptive context retrieval for LLM-driven analysis of wearable data. 

---
# ALDEN: Boosting Private Data Extraction from Retrieval-Augmented Generation Systems via Active Learning and Distribution Estimation 

**Authors**: Xingyu Lyu, Jianfeng He, Ning Wang, Yidan Hu, Tao Li, Danjue Chen, Shixiong Li, Yimin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.18762)  

**Abstract**: Retrieval-Augmented Generation (RAG) is widely used to augment large language models with external knowledge retrieval to improve reliability and generalization. However, recent studies have shown that RAG systems remain vulnerable to data extraction attacks, where adversaries can extract private data by embedding malicious commands into user queries. Despite their feasibility, existing attacks typically suffer from low data extraction rates and limited practical effectiveness. Here, we propose ALDEN, a novel attack that effectively and efficiently extracts private data from RAGs. First, we employ active learning to diversify malicious queries and improve data extraction rates. Second, we observe that the data distribution of the underlying knowledge base provides valuable guidance for query generation and introduce a decay-based dynamic algorithm to estimate the corresponding topic distribution. By combining them together, we demonstrate that ALDEN substantially outperforms state-of-the-art methods through comprehensive evaluations. 

---
# DOTRAG: Retrieval-Time Reasoning Along Paths 

**Authors**: Larnell Moore, Naihao Deng, Rada Mihalcea, Farnaz Jahanbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2605.18760)  

**Abstract**: Graph Retrieval-Augmented Generation (GraphRAG) is dominated by a retrieve-then-reason paradigm, where context is retrieved using heuristics and then reasoned over. Such methods struggle to adapt to the query-specific logic required for complex multi-hop tasks, often accumulating irrelevant context or missing correct relational paths. We propose DotRAG, a training-free GraphRAG framework that reformulates retrieval as a reasoning process over paths. Our approach generates query-conditioned constraints that guide graph exploration, prune irrelevant regions, and iteratively discover relational paths without relying on explicit step-by-step reasoning chains. We introduce Division of Thought (DOT), an abstraction that decomposes retrieval into localized search spaces and adapts the search strategy to each query. DotRAG achieves SOTA performance on MetaQA and UltraDomain, with consistent gains on multi-hop tasks, demonstrating the effectiveness of reasoning-guided retrieval. 

---
# SAGE: Scalable Automatic Gating Ensemble for Confident Negative Harvesting in Fraud Detection 

**Authors**: Sudheer Tubati, Amit Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2605.20157)  

**Abstract**: Music streaming fraud, where bad actors artificially inflate stream counts to manipulate chart rankings and royalty payments, poses a significant threat to streaming services and legitimate content creators. Traditional fraud detection approaches struggle with a critical challenge: many legitimate edge cases, including super-fans and sleep-music sessions, exhibit activity patterns that closely mimic those of coordinated fraud. We present SAGE, a novel counterfactual-aware negative harvesting approach that combines SimHash-based stratified sampling with a modular gating ensemble for confident negative identification from unlabeled data. Our ensemble architecture employs pluggable statistical gates (currently instantiated with Mahalanobis distance and k-NN density) with configurable voting thresholds enabling adaptive precision-recall trade-offs. This addresses the representation bias problem in Positive-Unlabeled learning by ensuring comprehensive coverage of rare behavioral cohorts through floor-constrained sampling. Evaluation demonstrates strong precision and recall on held-out data. The approach generalizes across fraud detection domains, achieving strong performance on both customer-level and artist-level fraud without modification to the core methodology. 

---
# BiRD: A Bidirectional Ranking Defense Mechanism for Retrieval Augmented Generation 

**Authors**: Chengcai Gao, Zhihong Sun, Xiaochuan Shi, Qiufeng Wang, Chao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20123)  

**Abstract**: The growing adoption of Retrieval-Augmented Generation (RAG) has led to a rise in adversarial attacks. Existing defenses, relying on semantic analysis or voting, face a trade-off between high computational cost and limited robustness under strong poisoning attacks. Their fundamental limitation is the exclusive focus on semantic content relevance, while neglecting the retrieval context that is critically defined by ranking structures. To this end, we investigate the bidirectional ranking behavior of poisoned and benign documents, and discover a key discriminative pattern: poisoned documents exhibit significantly stronger alignment between their backward rankings and the query's forward ranking. Capitalizing on this, we propose BiRD, a bidirectional ranking defense mechanism built upon a dual-signal framework that leverages forward ranking to assess semantic content relevance and backward ranking to quantify ranking context consistency. This design directly addresses the fundamental limitation of prior approaches, enabling simultaneous efficiency and robustness. Extensive evaluation across 3 datasets with 3 retrievers and 3 LLMs under 2 attack scenarios validates BiRD's effectiveness. Notably, BiRD reduces the attack success rate of PoisonedRAG by up to 54% while simultaneously improving task accuracy by up to 56%, with average additional latency under 1 second. 

---
# Auditing Privacy in Multi-Tenant RAG under Account Collusion 

**Authors**: Florian A. D. Burnat, Brittany I. Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2605.19847)  

**Abstract**: Multi-tenant retrieval-augmented generation (RAG) services advertise per-account differential privacy as the operative leakage boundary: each account's queries are guaranteed to satisfy $(\varepsilon_{\text{acc}}, \delta_{\text{acc}})$-DP with respect to the index. We identify same-index multi-account collusion as a privacy-boundary failure: for $k$ same-tenant accounts coordinating against the tenant's index -- the operative regime -- known DP composition theory implies joint leakage degrades unconditionally at rate $\Theta(\sqrt{k} \cdot \varepsilon_{\text{acc}})$ for Gaussian-noised retrieval. Cross-tenant and external collusion match the rate only under explicit access-control failure (M4); without M4 these regimes have zero leakage by design and reduce to an architectural audit, not a DP audit. We exhibit an attack realizing the rate and derive a RAG-specific MIA prediction we test empirically. To make this per-account/joint gap auditable, we design the first audit protocol that operates against unmodified RAG deployments and issues a quantitative $(\textsf{PASS}, \varepsilon_{\text{audit}})$ verdict for the retrieval-score channel -- the noise-then-select step the per-account DP guarantee actually covers -- without index disclosure, pipeline redesign, or model-weight exposure. Generation-channel privacy (LLM output conditioned on selected documents) is a separate audit predicate that should compose with ours; we explicitly scope it out. The protocol composes generic cryptographic primitives (Merkle ledgers, ZK function-application proofs, Gaussian noise attestations) with six RAG-specific primitives (embedder commitment, index-content vector commitment, per-account query ledger, noise-then-select attestation, cross-tenant containment proof, coalition-size estimator) and supports both closed-form audit bounds and Rényi-DP moments-accountant tracking. 

---
# PASC: Pipeline-Aware Conformal Prediction with Joint Coverage Guarantees for Multi-Stage NLP and LLM Pipelines 

**Authors**: Varun Kotte  

**Link**: [PDF](https://arxiv.org/pdf/2605.18812)  

**Abstract**: Modern NLP and LLM systems are pipelines: named entity recognition (NER) -> entity disambiguation (NED) -> entity typing, retrieval-augmented generation (retriever -> reader), and agentic chains of planner -> tool -> critic. Errors compound across stages, but existing uncertainty quantification methods either calibrate each stage independently (no joint coverage) or apply a Bonferroni union bound (joint coverage, but conservative). We present PASC (Pipeline-Aware Split Conformal), which reduces multi-stage joint coverage to a single scalar conformal prediction problem on the joint maximum nonconformity score. PASC provides a finite-sample distribution-free guarantee that all K stages are simultaneously covered with probability at least 1 - alpha, and is nearly tight up to a 1/(n+1) factor. On a three-stage NER -> NED -> entity-typing pipeline over CoNLL-2003, PASC achieves 96.4% end-to-end coverage versus 93.4% for Bonferroni and 86.5% for independent CP, at identical average prediction set size (1.083). Under distribution shift to WNUT-17 Twitter and WikiNEuRal Wikipedia data, PASC empirically maintains the target coverage in the tested shift settings while independent CP collapses to 59%. PASC requires a single quantile computation, runs 1.7x faster than Bonferroni, and scales to K = 6 stages where independent CP drops to 0.53 end-to-end coverage. The same joint-maximum-score reduction applies directly to compound LLM systems and agent pipelines. 

---
# Position: Let's Develop Data Probes to Fundamentally Understand How Data Affects LLM Performance 

**Authors**: Shiqiang Wang, Herbert Woisetschläger, Hans Arno Jacobsen, Mingyue Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.18801)  

**Abstract**: Data is fundamental to large language models (LLMs). However, understanding of what makes certain data useful for different stages of an LLM workflow, including training, tuning, alignment, in-context learning, etc., and why, remains an open question. Current approaches rely heavily on extensive experimentation with large public datasets to obtain empirical heuristics for data filtering and dataset construction. These approaches are compute intensive and lack a principled way of understanding the essence of how specific data characteristics drive LLM behavior. In this position paper, we advocate for the need of developing systematic methodologies for generating synthetic sequences from appropriately defined random processes, with the goal that these sequences can reveal useful characteristics when they are used in one or multiple stages of the LLM workflow. We refer to such sequences as data probes. By observing LLM behavior on data probes, researchers can systematically conduct studies on how data characteristics influence model performance, generalization, and robustness. The probing sequences exhibit statistical properties that can be viewed using theoretical concepts, such as typical sets, which are generalized to describe the behaviors of LLMs. This data-probe approach provides a pathway for uncovering foundational insights into the role of data in LLM training and inference, beyond empirical heuristics. 

---
# Accelerating AI-Powered Research: The PuppyChatter Framework for Usable and Flexible Tooling 

**Authors**: Chun-Hsiung Tseng, Hao-Chiang Koong Lin, Andrew Chih-Wei Huang, Yung-Hui Chen, Jia-Rou Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.17809)  

**Abstract**: This research addresses the challenges inherent in developing Artificial Intelligence (AI) applications, particularly those leveraging Large Language Models (LLMs). While AI vendors provide Application Programming Interfaces (APIs) and Software Development Kits (SDKs) to facilitate developer interaction, the former often requires intricate manual request construction, and the latter can lead to significant vendor lock-in. Furthermore, existing model abstraction frameworks, though mitigating vendor dependency, introduce an additional layer of complexity and potential security concerns. To reconcile these conflicting factors, the study introduces PuppyChatter, a novel software framework designed to preserve the intuitive simplicity of vendor-specific SDKs while simultaneously adhering to the vendor-neutrality principles characteristic of model abstraction, thereby offering a more streamlined and flexible development paradigm. 

---
