# Who Benefits from AI Explanations? Towards Accessible and Interpretable Systems 

**Authors**: Maria J. P. Peixoto, Akriti Pandey, Ahsan Zaman, Peter R. Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2508.10806)  

**Abstract**: As AI systems are increasingly deployed to support decision-making in critical domains, explainability has become a means to enhance the understandability of these outputs and enable users to make more informed and conscious choices. However, despite growing interest in the usability of eXplainable AI (XAI), the accessibility of these methods, particularly for users with vision impairments, remains underexplored. This paper investigates accessibility gaps in XAI through a two-pronged approach. First, a literature review of 79 studies reveals that evaluations of XAI techniques rarely include disabled users, with most explanations relying on inherently visual formats. Second, we present a four-part methodological proof of concept that operationalizes inclusive XAI design: (1) categorization of AI systems, (2) persona definition and contextualization, (3) prototype design and implementation, and (4) expert and user assessment of XAI techniques for accessibility. Preliminary findings suggest that simplified explanations are more comprehensible for non-visual users than detailed ones, and that multimodal presentation is required for more equitable interpretability. 

---
# The Knowledge-Reasoning Dissociation: Fundamental Limitations of LLMs in Clinical Natural Language Inference 

**Authors**: Maël Jullien, Marco Valentino, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2508.10777)  

**Abstract**: Large language models are often assumed to acquire increasingly structured, generalizable internal representations simply by scaling data and parameters. We interrogate this assumption by introducing a Clinical Trial Natural Language Inference benchmark comprising four reasoning families, Causal Attribution, Compositional Grounding, Epistemic Verification, and Risk State Abstraction. Each item is paired with a targeted Ground Knowledge and Meta-Level Reasoning Verification (GKMRV) probe, allowing us to dissociate failures of factual access from failures of inference. We evaluate six contemporary LLMs under both direct and chain of thought prompting.
Models achieve near-ceiling GKMRV accuracy (mean accuracy 0.918) yet perform poorly on the main reasoning tasks (mean accuracy 0.25). Despite low accuracy, output inferences are highly consistent across samples (mean 0.87), indicating a systematic application of underlying heuristics and shortcuts.
These results reveal fundamental structural and representational limitations: current LLMs often possess the relevant clinical knowledge but lack the structured, composable internal representations needed to deploy it reliably (e.g., integrating constraints, weighing evidence, or simulating counterfactuals). Decoupling knowledge from reasoning with GKMRV makes this dissociation explicit and measurable, providing an effective framework for probing the reliability of LLMs in high-stakes domains. 

---
# Modeling Human Responses to Multimodal AI Content 

**Authors**: Zhiqi Shen, Shaojing Fan, Danni Xu, Terence Sim, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2508.10769)  

**Abstract**: As AI-generated content becomes widespread, so does the risk of misinformation. While prior research has primarily focused on identifying whether content is authentic, much less is known about how such content influences human perception and behavior. In domains like trading or the stock market, predicting how people react (e.g., whether a news post will go viral), can be more critical than verifying its factual accuracy. To address this, we take a human-centered approach and introduce the MhAIM Dataset, which contains 154,552 online posts (111,153 of them AI-generated), enabling large-scale analysis of how people respond to AI-generated content. Our human study reveals that people are better at identifying AI content when posts include both text and visuals, particularly when inconsistencies exist between the two. We propose three new metrics: trustworthiness, impact, and openness, to quantify how users judge and engage with online content. We present T-Lens, an LLM-based agent system designed to answer user queries by incorporating predicted human responses to multimodal information. At its core is HR-MCP (Human Response Model Context Protocol), built on the standardized Model Context Protocol (MCP), enabling seamless integration with any LLM. This integration allows T-Lens to better align with human reactions, enhancing both interpretability and interaction capabilities. Our work provides empirical insights and practical tools to equip LLMs with human-awareness capabilities. By highlighting the complex interplay among AI, human cognition, and information reception, our findings suggest actionable strategies for mitigating the risks of AI-driven misinformation. 

---
# Scaling Up without Fading Out: Goal-Aware Sparse GNN for RL-based Generalized Planning 

**Authors**: Sangwoo Jeon, Juchul Shin, Gyeong-Tae Kim, YeonJe Cho, Seongwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.10747)  

**Abstract**: Generalized planning using deep reinforcement learning (RL) combined with graph neural networks (GNNs) has shown promising results in various symbolic planning domains described by PDDL. However, existing approaches typically represent planning states as fully connected graphs, leading to a combinatorial explosion in edge information and substantial sparsity as problem scales grow, especially evident in large grid-based environments. This dense representation results in diluted node-level information, exponentially increases memory requirements, and ultimately makes learning infeasible for larger-scale problems. To address these challenges, we propose a sparse, goal-aware GNN representation that selectively encodes relevant local relationships and explicitly integrates spatial features related to the goal. We validate our approach by designing novel drone mission scenarios based on PDDL within a grid world, effectively simulating realistic mission execution environments. Our experimental results demonstrate that our method scales effectively to larger grid sizes previously infeasible with dense graph representations and substantially improves policy generalization and success rates. Our findings provide a practical foundation for addressing realistic, large-scale generalized planning tasks. 

---
# Agentic Design Review System 

**Authors**: Sayan Nag, K J Joseph, Koustava Goswami, Vlad I Morariu, Balaji Vasan Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10745)  

**Abstract**: Evaluating graphic designs involves assessing it from multiple facets like alignment, composition, aesthetics and color choices. Evaluating designs in a holistic way involves aggregating feedback from individual expert reviewers. Towards this, we propose an Agentic Design Review System (AgenticDRS), where multiple agents collaboratively analyze a design, orchestrated by a meta-agent. A novel in-context exemplar selection approach based on graph matching and a unique prompt expansion method plays central role towards making each agent design aware. Towards evaluating this framework, we propose DRS-BENCH benchmark. Thorough experimental evaluation against state-of-the-art baselines adapted to the problem setup, backed-up with critical ablation experiments brings out the efficacy of Agentic-DRS in evaluating graphic designs and generating actionable feedback. We hope that this work will attract attention to this pragmatic, yet under-explored research direction. 

---
# GenOM: Ontology Matching with Description Generation and Large Language Model 

**Authors**: Yiping Song, Jiaoyan Chen, Renate A. Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2508.10703)  

**Abstract**: Ontology matching (OM) plays an essential role in enabling semantic interoperability and integration across heterogeneous knowledge sources, particularly in the biomedical domain which contains numerous complex concepts related to diseases and pharmaceuticals. This paper introduces GenOM, a large language model (LLM)-based ontology alignment framework, which enriches the semantic representations of ontology concepts via generating textual definitions, retrieves alignment candidates with an embedding model, and incorporates exact matching-based tools to improve precision. Extensive experiments conducted on the OAEI Bio-ML track demonstrate that GenOM can often achieve competitive performance, surpassing many baselines including traditional OM systems and recent LLM-based methods. Further ablation studies confirm the effectiveness of semantic enrichment and few-shot prompting, highlighting the framework's robustness and adaptability. 

---
# STEP: Stepwise Curriculum Learning for Context-Knowledge Fusion in Conversational Recommendation 

**Authors**: Zhenye Yang, Jinpeng Chen, Huan Li, Xiongnan Jin, Xuanyang Li, Junwei Zhang, Hongbo Gao, Kaimin Wei, Senzhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10669)  

**Abstract**: Conversational recommender systems (CRSs) aim to proactively capture user preferences through natural language dialogue and recommend high-quality items. To achieve this, CRS gathers user preferences via a dialog module and builds user profiles through a recommendation module to generate appropriate recommendations. However, existing CRS faces challenges in capturing the deep semantics of user preferences and dialogue context. In particular, the efficient integration of external knowledge graph (KG) information into dialogue generation and recommendation remains a pressing issue. Traditional approaches typically combine KG information directly with dialogue content, which often struggles with complex semantic relationships, resulting in recommendations that may not align with user expectations.
To address these challenges, we introduce STEP, a conversational recommender centered on pre-trained language models that combines curriculum-guided context-knowledge fusion with lightweight task-specific prompt tuning. At its heart, an F-Former progressively aligns the dialogue context with knowledge-graph entities through a three-stage curriculum, thus resolving fine-grained semantic mismatches. The fused representation is then injected into the frozen language model via two minimal yet adaptive prefix prompts: a conversation prefix that steers response generation toward user intent and a recommendation prefix that biases item ranking toward knowledge-consistent candidates. This dual-prompt scheme allows the model to share cross-task semantics while respecting the distinct objectives of dialogue and recommendation. Experimental results show that STEP outperforms mainstream methods in the precision of recommendation and dialogue quality in two public datasets. 

---
# MSRS: Adaptive Multi-Subspace Representation Steering for Attribute Alignment in Large Language Models 

**Authors**: Xinyan Jiang, Lin Zhang, Jiayi Zhang, Qingsong Yang, Guimin Hu, Di Wang, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10599)  

**Abstract**: Activation steering offers a promising approach to controlling the behavior of Large Language Models by directly manipulating their internal activations. However, most existing methods struggle to jointly steer multiple attributes, often resulting in interference and undesirable trade-offs. To address this challenge, we propose Multi-Subspace Representation Steering (MSRS), a novel framework for effective multi-attribute steering via subspace representation fine-tuning. MSRS reduces inter-attribute interference by allocating orthogonal subspaces to each attribute, isolating their influence within the model's representation space. MSRS also incorporates a hybrid subspace composition strategy: it combines attribute-specific subspaces for unique steering directions with a shared subspace for common steering directions. A dynamic weighting function learns to efficiently integrate these components for precise control. During inference, MSRS introduces a token-level steering mechanism that dynamically identifies and intervenes on the most semantically relevant tokens, enabling fine-grained behavioral modulation. Experimental results show that MSRS significantly reduces attribute conflicts, surpasses existing methods across a range of attributes, and generalizes effectively to diverse downstream tasks. 

---
# Improving Value-based Process Verifier via Low-Cost Variance Reduction 

**Authors**: Zetian Sun, Dongfang Li, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10539)  

**Abstract**: Large language models (LLMs) have achieved remarkable success in a wide range of tasks. However, their reasoning capabilities, particularly in complex domains like mathematics, remain a significant challenge. Value-based process verifiers, which estimate the probability of a partial reasoning chain leading to a correct solution, are a promising approach for improving reasoning. Nevertheless, their effectiveness is often hindered by estimation error in their training annotations, a consequence of the limited number of Monte Carlo (MC) samples feasible due to the high cost of LLM inference. In this paper, we identify that the estimation error primarily arises from high variance rather than bias, and the MC estimator is a Minimum Variance Unbiased Estimator (MVUE). To address the problem, we propose the \textsc{Com}pound \textsc{M}onte \textsc{C}arlo \textsc{S}ampling (ComMCS) method, which constructs an unbiased estimator by linearly combining the MC estimators from the current and subsequent steps. Theoretically, we show that our method leads to a predictable reduction in variance, while maintaining an unbiased estimation without additional LLM inference cost. We also perform empirical experiments on the MATH-500 and GSM8K benchmarks to demonstrate the effectiveness of our method. Notably, ComMCS outperforms regression-based optimization method by 2.8 points, the non-variance-reduced baseline by 2.2 points on MATH-500 on Best-of-32 sampling experiment. 

---
# Diversity First, Quality Later: A Two-Stage Assumption for Language Model Alignment 

**Authors**: Zetian Sun, Dongfang Li, Baotian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10530)  

**Abstract**: The alignment of language models (LMs) with human preferences is critical for building reliable AI systems. The problem is typically framed as optimizing an LM policy to maximize the expected reward that reflects human preferences. Recently, Direct Preference Optimization (DPO) was proposed as a LM alignment method that directly optimize the policy from static preference data, and further improved by incorporating on-policy sampling (i.e., preference candidates generated during the training loop) for better LM alignment. However, we show on-policy data is not always optimal, with systematic effectiveness difference emerging between static and on-policy preference candidates. For example, on-policy data can result in a 3$\times$ effectiveness compared with static data for Llama-3, and a 0.4$\times$ effectiveness for Zephyr. To explain the phenomenon, we propose the alignment stage assumption, which divides the alignment process into two distinct stages: the preference injection stage, which benefits from diverse data, and the preference fine-tuning stage, which favors high-quality data. Through theoretical and empirical analysis, we characterize these stages and propose an effective algorithm to identify the boundaries between them. We perform experiments on 5 models (Llama, Zephyr, Phi-2, Qwen, Pythia) and 2 alignment methods (DPO, SLiC-HF) to show the generalizability of alignment stage assumption and boundary measurement. 

---
# PASS: Probabilistic Agentic Supernet Sampling for Interpretable and Adaptive Chest X-Ray Reasoning 

**Authors**: Yushi Feng, Junye Du, Yingying Hong, Qifan Wang, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10501)  

**Abstract**: Existing tool-augmented agentic systems are limited in the real world by (i) black-box reasoning steps that undermine trust of decision-making and pose safety risks, (ii) poor multimodal integration, which is inherently critical for healthcare tasks, and (iii) rigid and computationally inefficient agentic pipelines. We introduce PASS (Probabilistic Agentic Supernet Sampling), the first multimodal framework to address these challenges in the context of Chest X-Ray (CXR) reasoning. PASS adaptively samples agentic workflows over a multi-tool graph, yielding decision paths annotated with interpretable probabilities. Given the complex CXR reasoning task with multimodal medical data, PASS leverages its learned task-conditioned distribution over the agentic supernet. Thus, it adaptively selects the most suitable tool at each supernet layer, offering probability-annotated trajectories for post-hoc audits and directly enhancing medical AI safety. PASS also continuously compresses salient findings into an evolving personalized memory, while dynamically deciding whether to deepen its reasoning path or invoke an early exit for efficiency. To optimize a Pareto frontier balancing performance and cost, we design a novel three-stage training procedure, including expert knowledge warm-up, contrastive path-ranking, and cost-aware reinforcement learning. To facilitate rigorous evaluation, we introduce CAB-E, a comprehensive benchmark for multi-step, safety-critical, free-form CXR reasoning. Experiments across various benchmarks validate that PASS significantly outperforms strong baselines in multiple metrics (e.g., accuracy, AUC, LLM-J.) while balancing computational costs, pushing a new paradigm shift towards interpretable, adaptive, and multimodal medical agentic systems. 

---
# Reverse Physician-AI Relationship: Full-process Clinical Diagnosis Driven by a Large Language Model 

**Authors**: Shicheng Xu, Xin Huang, Zihao Wei, Liang Pang, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.10492)  

**Abstract**: Full-process clinical diagnosis in the real world encompasses the entire diagnostic workflow that begins with only an ambiguous chief complaint. While artificial intelligence (AI), particularly large language models (LLMs), is transforming clinical diagnosis, its role remains largely as an assistant to physicians. This AI-assisted working pattern makes AI can only answer specific medical questions at certain parts within the diagnostic process, but lack the ability to drive the entire diagnostic process starting from an ambiguous complaint, which still relies heavily on human physicians. This gap limits AI's ability to fully reduce physicians' workload and enhance diagnostic efficiency. To address this, we propose a paradigm shift that reverses the relationship between physicians and AI: repositioning AI as the primary director, with physicians serving as its assistants. So we present DxDirector-7B, an LLM endowed with advanced deep thinking capabilities, enabling it to drive the full-process diagnosis with minimal physician involvement. Furthermore, DxDirector-7B establishes a robust accountability framework for misdiagnoses, delineating responsibility between AI and human physicians. In evaluations across rare, complex, and real-world cases under full-process diagnosis setting, DxDirector-7B not only achieves significant superior diagnostic accuracy but also substantially reduces physician workload than state-of-the-art medical LLMs as well as general-purpose LLMs. Fine-grained analyses across multiple clinical departments and tasks validate its efficacy, with expert evaluations indicating its potential to serve as a viable substitute for medical specialists. These findings mark a new era where AI, traditionally a physicians' assistant, now drives the entire diagnostic process to drastically reduce physicians' workload, indicating an efficient and accurate diagnostic solution. 

---
# SEQ-GPT: LLM-assisted Spatial Query via Example 

**Authors**: Ivan Khai Ze Lim, Ningyi Liao, Yiming Yang, Gerald Wei Yong Yip, Siqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.10486)  

**Abstract**: Contemporary spatial services such as online maps predominantly rely on user queries for location searches. However, the user experience is limited when performing complex tasks, such as searching for a group of locations simultaneously. In this study, we examine the extended scenario known as Spatial Exemplar Query (SEQ), where multiple relevant locations are jointly searched based on user-specified examples. We introduce SEQ-GPT, a spatial query system powered by Large Language Models (LLMs) towards more versatile SEQ search using natural language. The language capabilities of LLMs enable unique interactive operations in the SEQ process, including asking users to clarify query details and dynamically adjusting the search based on user feedback. We also propose a tailored LLM adaptation pipeline that aligns natural language with structured spatial data and queries through dialogue synthesis and multi-model cooperation. SEQ-GPT offers an end-to-end demonstration for broadening spatial search with realistic data and application scenarios. 

---
# FIRESPARQL: A LLM-based Framework for SPARQL Query Generation over Scholarly Knowledge Graphs 

**Authors**: Xueli Pan, Victor de Boer, Jacco van Ossenbruggen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10467)  

**Abstract**: Question answering over Scholarly Knowledge Graphs (SKGs) remains a challenging task due to the complexity of scholarly content and the intricate structure of these graphs. Large Language Model (LLM) approaches could be used to translate natural language questions (NLQs) into SPARQL queries; however, these LLM-based approaches struggle with SPARQL query generation due to limited exposure to SKG-specific content and the underlying schema. We identified two main types of errors in the LLM-generated SPARQL queries: (i) structural inconsistencies, such as missing or redundant triples in the queries, and (ii) semantic inaccuracies, where incorrect entities or properties are shown in the queries despite a correct query structure. To address these issues, we propose FIRESPARQL, a modular framework that supports fine-tuned LLMs as a core component, with optional context provided via retrieval-augmented generation (RAG) and a SPARQL query correction layer. We evaluate the framework on the SciQA Benchmark using various configurations (zero-shot, zero-shot with RAG, one-shot, fine-tuning, and fine-tuning with RAG) and compare the performance with baseline and state-of-the-art approaches. We measure query accuracy using BLEU and ROUGE metrics, and query result accuracy using relaxed exact match(RelaxedEM), with respect to the gold standards containing the NLQs, SPARQL queries, and the results of the queries. Experimental results demonstrate that fine-tuning achieves the highest overall performance, reaching 0.90 ROUGE-L for query accuracy and 0.85 RelaxedEM for result accuracy on the test set. 

---
# We-Math 2.0: A Versatile MathBook System for Incentivizing Visual Mathematical Reasoning 

**Authors**: Runqi Qiao, Qiuna Tan, Peiqing Yang, Yanzi Wang, Xiaowan Wang, Enhui Wan, Sitong Zhou, Guanting Dong, Yuchen Zeng, Yida Xu, Jie Wang, Chong Sun, Chen Li, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10433)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across various tasks, but still struggle with complex mathematical reasoning. Existing research primarily focuses on dataset construction and method optimization, often overlooking two critical aspects: comprehensive knowledge-driven design and model-centric data space modeling. In this paper, we introduce We-Math 2.0, a unified system that integrates a structured mathematical knowledge system, model-centric data space modeling, and a reinforcement learning (RL)-based training paradigm to comprehensively enhance the mathematical reasoning abilities of MLLMs. The key contributions of We-Math 2.0 are fourfold: (1) MathBook Knowledge System: We construct a five-level hierarchical system encompassing 491 knowledge points and 1,819 fundamental principles. (2) MathBook-Standard & Pro: We develop MathBook-Standard, a dataset that ensures broad conceptual coverage and flexibility through dual expansion. Additionally, we define a three-dimensional difficulty space and generate 7 progressive variants per problem to build MathBook-Pro, a challenging dataset for robust training. (3) MathBook-RL: We propose a two-stage RL framework comprising: (i) Cold-Start Fine-tuning, which aligns the model with knowledge-oriented chain-of-thought reasoning; and (ii) Progressive Alignment RL, leveraging average-reward learning and dynamic data scheduling to achieve progressive alignment across difficulty levels. (4) MathBookEval: We introduce a comprehensive benchmark covering all 491 knowledge points with diverse reasoning step distributions. Experimental results show that MathBook-RL performs competitively with existing baselines on four widely-used benchmarks and achieves strong results on MathBookEval, suggesting promising generalization in mathematical reasoning. 

---
# MM-Food-100K: A 100,000-Sample Multimodal Food Intelligence Dataset with Verifiable Provenance 

**Authors**: Yi Dong, Yusuke Muraoka, Scott Shi, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10429)  

**Abstract**: We present MM-Food-100K, a public 100,000-sample multimodal food intelligence dataset with verifiable provenance. It is a curated approximately 10% open subset of an original 1.2 million, quality-accepted corpus of food images annotated for a wide range of information (such as dish name, region of creation). The corpus was collected over six weeks from over 87,000 contributors using the Codatta contribution model, which combines community sourcing with configurable AI-assisted quality checks; each submission is linked to a wallet address in a secure off-chain ledger for traceability, with a full on-chain protocol on the roadmap. We describe the schema, pipeline, and QA, and validate utility by fine-tuning large vision-language models (ChatGPT 5, ChatGPT OSS, Qwen-Max) on image-based nutrition prediction. Fine-tuning yields consistent gains over out-of-box baselines across standard metrics; we report results primarily on the MM-Food-100K subset. We release MM-Food-100K for publicly free access and retain approximately 90% for potential commercial access with revenue sharing to contributors. 

---
# HiRef: Leveraging Hierarchical Ontology and Network Refinement for Robust Medication Recommendation 

**Authors**: Yan Ting Chok, Soyon Park, Seungheun Baek, Hajung Kim, Junhyun Lee, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10425)  

**Abstract**: Medication recommendation is a crucial task for assisting physicians in making timely decisions from longitudinal patient medical records. However, real-world EHR data present significant challenges due to the presence of rarely observed medical entities and incomplete records that may not fully capture the clinical ground truth. While data-driven models trained on longitudinal Electronic Health Records often achieve strong empirical performance, they struggle to generalize under missing or novel conditions, largely due to their reliance on observed co-occurrence patterns. To address these issues, we propose Hierarchical Ontology and Network Refinement for Robust Medication Recommendation (HiRef), a unified framework that combines two complementary structures: (i) the hierarchical semantics encoded in curated medical ontologies, and (ii) refined co-occurrence patterns derived from real-world EHRs. We embed ontology entities in hyperbolic space, which naturally captures tree-like relationships and enables knowledge transfer through shared ancestors, thereby improving generalizability to unseen codes. To further improve robustness, we introduce a prior-guided sparse regularization scheme that refines the EHR co-occurrence graph by suppressing spurious edges while preserving clinically meaningful associations. Our model achieves strong performance on EHR benchmarks (MIMIC-III and MIMIC-IV) and maintains high accuracy under simulated unseen-code settings. Extensive experiments with comprehensive ablation studies demonstrate HiRef's resilience to unseen medical codes, supported by in-depth analyses of the learned sparsified graph structure and medical code embeddings. 

---
# LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval 

**Authors**: Yaoze Zhang, Rong Wu, Pinlong Cai, Xiaoman Wang, Guohang Yan, Song Mao, Ding Wang, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10391)  

**Abstract**: Retrieval-Augmented Generation (RAG) plays a crucial role in grounding Large Language Models by leveraging external knowledge, whereas the effectiveness is often compromised by the retrieval of contextually flawed or incomplete information. To address this, knowledge graph-based RAG methods have evolved towards hierarchical structures, organizing knowledge into multi-level summaries. However, these approaches still suffer from two critical, unaddressed challenges: high-level conceptual summaries exist as disconnected ``semantic islands'', lacking the explicit relations needed for cross-community reasoning; and the retrieval process itself remains structurally unaware, often degenerating into an inefficient flat search that fails to exploit the graph's rich topology. To overcome these limitations, we introduce LeanRAG, a framework that features a deeply collaborative design combining knowledge aggregation and retrieval strategies. LeanRAG first employs a novel semantic aggregation algorithm that forms entity clusters and constructs new explicit relations among aggregation-level summaries, creating a fully navigable semantic network. Then, a bottom-up, structure-guided retrieval strategy anchors queries to the most relevant fine-grained entities and then systematically traverses the graph's semantic pathways to gather concise yet contextually comprehensive evidence sets. The LeanRAG can mitigate the substantial overhead associated with path retrieval on graphs and minimizes redundant information retrieval. Extensive experiments on four challenging QA benchmarks with different domains demonstrate that LeanRAG significantly outperforming existing methods in response quality while reducing 46\% retrieval redundancy. Code is available at: this https URL 

---
# What to Ask Next? Probing the Imaginative Reasoning of LLMs with TurtleSoup Puzzles 

**Authors**: Mengtao Zhou, Sifan Wu, Huan Zhang, Qi Sima, Bang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10358)  

**Abstract**: We investigate the capacity of Large Language Models (LLMs) for imaginative reasoning--the proactive construction, testing, and revision of hypotheses in information-sparse environments. Existing benchmarks, often static or focused on social deduction, fail to capture the dynamic, exploratory nature of this reasoning process. To address this gap, we introduce a comprehensive research framework based on the classic "Turtle Soup" game, integrating a benchmark, an agent, and an evaluation protocol. We present TurtleSoup-Bench, the first large-scale, bilingual, interactive benchmark for imaginative reasoning, comprising 800 turtle soup puzzles sourced from both the Internet and expert authors. We also propose Mosaic-Agent, a novel agent designed to assess LLMs' performance in this setting. To evaluate reasoning quality, we develop a multi-dimensional protocol measuring logical consistency, detail completion, and conclusion alignment. Experiments with leading LLMs reveal clear capability limits, common failure patterns, and a significant performance gap compared to humans. Our work offers new insights into LLMs' imaginative reasoning and establishes a foundation for future research on exploratory agent behavior. 

---
# Multi-Agent Trust Region Policy Optimisation: A Joint Constraint Approach 

**Authors**: Chak Lam Shek, Guangyao Shi, Pratap Tokekar  

**Link**: [PDF](https://arxiv.org/pdf/2508.10340)  

**Abstract**: Multi-agent reinforcement learning (MARL) requires coordinated and stable policy updates among interacting agents. Heterogeneous-Agent Trust Region Policy Optimization (HATRPO) enforces per-agent trust region constraints using Kullback-Leibler (KL) divergence to stabilize training. However, assigning each agent the same KL threshold can lead to slow and locally optimal updates, especially in heterogeneous settings. To address this limitation, we propose two approaches for allocating the KL divergence threshold across agents: HATRPO-W, a Karush-Kuhn-Tucker-based (KKT-based) method that optimizes threshold assignment under global KL constraints, and HATRPO-G, a greedy algorithm that prioritizes agents based on improvement-to-divergence ratio. By connecting sequential policy optimization with constrained threshold scheduling, our approach enables more flexible and effective learning in heterogeneous-agent settings. Experimental results demonstrate that our methods significantly boost the performance of HATRPO, achieving faster convergence and higher final rewards across diverse MARL benchmarks. Specifically, HATRPO-W and HATRPO-G achieve comparable improvements in final performance, each exceeding 22.5%. Notably, HATRPO-W also demonstrates more stable learning dynamics, as reflected by its lower variance. 

---
# A Curriculum Learning Approach to Reinforcement Learning: Leveraging RAG for Multimodal Question Answering 

**Authors**: Chenliang Zhang, Lin Wang, Yuanyuan Lu, Yusheng Qi, Kexin Wang, Peixu Hou, Wenshi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10337)  

**Abstract**: This paper describes the solutions of the Dianping-Trust-Safety team for the META CRAG-MM challenge. The challenge requires building a comprehensive retrieval-augmented generation system capable for multi-modal multi-turn question answering. The competition consists of three tasks: (1) answering questions using structured data retrieved from an image-based mock knowledge graph, (2) synthesizing information from both knowledge graphs and web search results, and (3) handling multi-turn conversations that require context understanding and information aggregation from multiple sources. For Task 1, our solution is based on the vision large language model, enhanced by supervised fine-tuning with knowledge distilled from GPT-4.1. We further applied curriculum learning strategies to guide reinforcement learning, resulting in improved answer accuracy and reduced hallucination. For Task 2 and Task 3, we additionally leveraged web search APIs to incorporate external knowledge, enabling the system to better handle complex queries and multi-turn conversations. Our approach achieved 1st place in Task 1 with a significant lead of 52.38\%, and 3rd place in Task 3, demonstrating the effectiveness of the integration of curriculum learning with reinforcement learning in our training pipeline. 

---
# Promoting Efficient Reasoning with Verifiable Stepwise Reward 

**Authors**: Chuhuai Yue, Chengqi Dong, Yinan Gao, Hang He, Jiajun Chai, Guojun Yin, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.10293)  

**Abstract**: Large reasoning models (LRMs) have recently achieved significant progress in complex reasoning tasks, aided by reinforcement learning with verifiable rewards. However, LRMs often suffer from overthinking, expending excessive computation on simple problems and reducing efficiency. Existing efficient reasoning methods typically require accurate task assessment to preset token budgets or select reasoning modes, which limits their flexibility and reliability. In this work, we revisit the essence of overthinking and identify that encouraging effective steps while penalizing ineffective ones is key to its solution. To this end, we propose a novel rule-based verifiable stepwise reward mechanism (VSRM), which assigns rewards based on the performance of intermediate states in the reasoning trajectory. This approach is intuitive and naturally fits the step-by-step nature of reasoning tasks. We conduct extensive experiments on standard mathematical reasoning benchmarks, including AIME24 and AIME25, by integrating VSRM with PPO and Reinforce++. Results show that our method achieves substantial output length reduction while maintaining original reasoning performance, striking an optimal balance between efficiency and accuracy. Further analysis of overthinking frequency and pass@k score before and after training demonstrates that our approach in deed effectively suppresses ineffective steps and encourages effective reasoning, fundamentally alleviating the overthinking problem. All code will be released upon acceptance. 

---
# Why Cannot Large Language Models Ever Make True Correct Reasoning? 

**Authors**: Jingde Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.10265)  

**Abstract**: Recently, with the application progress of AIGC tools based on large language models (LLMs), led by ChatGPT, many AI experts and more non-professionals are trumpeting the "understanding ability" and "reasoning ability" of the LLMs. The present author considers that the so-called "understanding ability" and "reasoning ability" of LLMs are just illusions of those people who with vague concepts. In fact, the LLMs can never have the true understanding ability and true reasoning ability. This paper intents to explain that, because the essential limitations of their working principle, the LLMs can never have the ability of true correct reasoning. 

---
# Extending the Entropic Potential of Events for Uncertainty Quantification and Decision-Making in Artificial Intelligence 

**Authors**: Mark Zilberman  

**Link**: [PDF](https://arxiv.org/pdf/2508.10241)  

**Abstract**: This work demonstrates how the concept of the entropic potential of events -- a parameter quantifying the influence of discrete events on the expected future entropy of a system -- can enhance uncertainty quantification, decision-making, and interpretability in artificial intelligence (AI). Building on its original formulation in physics, the framework is adapted for AI by introducing an event-centric measure that captures how actions, observations, or other discrete occurrences impact uncertainty at future time horizons. Both the original and AI-adjusted definitions of entropic potential are formalized, with the latter emphasizing conditional expectations to account for counterfactual scenarios. Applications are explored in policy evaluation, intrinsic reward design, explainable AI, and anomaly detection, highlighting the metric's potential to unify and strengthen uncertainty modeling in intelligent systems. Conceptual examples illustrate its use in reinforcement learning, Bayesian inference, and anomaly detection, while practical considerations for computation in complex AI models are discussed. The entropic potential framework offers a theoretically grounded, interpretable, and versatile approach to managing uncertainty in AI, bridging principles from thermodynamics, information theory, and machine learning. 

---
# KompeteAI: Accelerated Autonomous Multi-Agent System for End-to-End Pipeline Generation for Machine Learning Problems 

**Authors**: Stepan Kulibaba, Artem Dzhalilov, Roman Pakhomov, Oleg Svidchenko, Alexander Gasnikov, Aleksei Shpilman  

**Link**: [PDF](https://arxiv.org/pdf/2508.10177)  

**Abstract**: Recent Large Language Model (LLM)-based AutoML systems demonstrate impressive capabilities but face significant limitations such as constrained exploration strategies and a severe execution bottleneck. Exploration is hindered by one-shot methods lacking diversity and Monte Carlo Tree Search (MCTS) approaches that fail to recombine strong partial solutions. The execution bottleneck arises from lengthy code validation cycles that stifle iterative refinement. To overcome these challenges, we introduce KompeteAI, a novel AutoML framework with dynamic solution space exploration. Unlike previous MCTS methods that treat ideas in isolation, KompeteAI introduces a merging stage that composes top candidates. We further expand the hypothesis space by integrating Retrieval-Augmented Generation (RAG), sourcing ideas from Kaggle notebooks and arXiv papers to incorporate real-world strategies. KompeteAI also addresses the execution bottleneck via a predictive scoring model and an accelerated debugging method, assessing solution potential using early stage metrics to avoid costly full-code execution. This approach accelerates pipeline evaluation 6.9 times. KompeteAI outperforms leading methods (e.g., RD-agent, AIDE, and Ml-Master) by an average of 3\% on the primary AutoML benchmark, MLE-Bench. Additionally, we propose Kompete-bench to address limitations in MLE-Bench, where KompeteAI also achieves state-of-the-art results 

---
# Pruning Long Chain-of-Thought of Large Reasoning Models via Small-Scale Preference Optimization 

**Authors**: Bin Hong, Jiayu Liu, Zhenya Huang, Kai Zhang, Mengdi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10164)  

**Abstract**: Recent advances in Large Reasoning Models (LRMs) have demonstrated strong performance on complex tasks through long Chain-of-Thought (CoT) reasoning. However, their lengthy outputs increase computational costs and may lead to overthinking, raising challenges in balancing reasoning effectiveness and efficiency. Current methods for efficient reasoning often compromise reasoning quality or require extensive resources. This paper investigates efficient methods to reduce the generation length of LRMs. We analyze generation path distributions and filter generated trajectories through difficulty estimation. Subsequently, we analyze the convergence behaviors of the objectives of various preference optimization methods under a Bradley-Terry loss based framework. Based on the analysis, we propose Length Controlled Preference Optimization (LCPO) that directly balances the implicit reward related to NLL loss. LCPO can effectively learn length preference with limited data and training. Extensive experiments demonstrate that our approach significantly reduces the average output length by over 50\% across multiple benchmarks while maintaining the reasoning performance. Our work highlights the potential for computationally efficient approaches in guiding LRMs toward efficient reasoning. 

---
# Improving and Evaluating Open Deep Research Agents 

**Authors**: Doaa Allabadi, Kyle Bradbury, Jordan M. Malof  

**Link**: [PDF](https://arxiv.org/pdf/2508.10152)  

**Abstract**: We focus here on Deep Research Agents (DRAs), which are systems that can take a natural language prompt from a user, and then autonomously search for, and utilize, internet-based content to address the prompt. Recent DRAs have demonstrated impressive capabilities on public benchmarks however, recent research largely involves proprietary closed-source systems. At the time of this work, we only found one open-source DRA, termed Open Deep Research (ODR). In this work we adapt the challenging recent BrowseComp benchmark to compare ODR to existing proprietary systems. We propose BrowseComp-Small (BC-Small), comprising a subset of BrowseComp, as a more computationally-tractable DRA benchmark for academic labs. We benchmark ODR and two other proprietary systems on BC-Small: one system from Anthropic and one system from Google. We find that all three systems achieve 0% accuracy on the test set of 60 questions. We introduce three strategic improvements to ODR, resulting in the ODR+ model, which achieves a state-of-the-art 10% success rate on BC-Small among both closed-source and open-source systems. We report ablation studies indicating that all three of our improvements contributed to the success of ODR+. 

---
# Agentic AI Frameworks: Architectures, Protocols, and Design Challenges 

**Authors**: Hana Derouiche, Zaki Brahmi, Haithem Mazeni  

**Link**: [PDF](https://arxiv.org/pdf/2508.10146)  

**Abstract**: The emergence of Large Language Models (LLMs) has ushered in a transformative paradigm in artificial intelligence, Agentic AI, where intelligent agents exhibit goal-directed autonomy, contextual reasoning, and dynamic multi-agent coordination. This paper provides a systematic review and comparative analysis of leading Agentic AI frameworks, including CrewAI, LangGraph, AutoGen, Semantic Kernel, Agno, Google ADK, and MetaGPT, evaluating their architectural principles, communication mechanisms, memory management, safety guardrails, and alignment with service-oriented computing paradigms. Furthermore, we identify key limitations, emerging trends, and open challenges in the field. To address the issue of agent communication, we conduct an in-depth analysis of protocols such as the Contract Net Protocol (CNP), Agent-to-Agent (A2A), Agent Network Protocol (ANP), and Agora. Our findings not only establish a foundational taxonomy for Agentic AI systems but also propose future research directions to enhance scalability, robustness, and interoperability. This work serves as a comprehensive reference for researchers and practitioners working to advance the next generation of autonomous AI systems. 

---
# MCP-Orchestrated Multi-Agent System for Automated Disinformation Detection 

**Authors**: Alexandru-Andrei Avram, Adrian Groza, Alexandru Lecu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10143)  

**Abstract**: The large spread of disinformation across digital platforms creates significant challenges to information integrity. This paper presents a multi-agent system that uses relation extraction to detect disinformation in news articles, focusing on titles and short text snippets. The proposed Agentic AI system combines four agents: (i) a machine learning agent (logistic regression), (ii) a Wikipedia knowledge check agent (which relies on named entity recognition), (iii) a coherence detection agent (using LLM prompt engineering), and (iv) a web-scraped data analyzer that extracts relational triplets for fact checking. The system is orchestrated via the Model Context Protocol (MCP), offering shared context and live learning across components. Results demonstrate that the multi-agent ensemble achieves 95.3% accuracy with an F1 score of 0.964, significantly outperforming individual agents and traditional approaches. The weighted aggregation method, mathematically derived from individual agent misclassification rates, proves superior to algorithmic threshold optimization. The modular architecture makes the system easily scalable, while also maintaining details of the decision processes. 

---
# Amazon Nova AI Challenge -- Trusted AI: Advancing secure, AI-assisted software development 

**Authors**: Sattvik Sahai, Prasoon Goyal, Michael Johnston, Anna Gottardi, Yao Lu, Lucy Hu, Luke Dai, Shaohua Liu, Samyuth Sagi, Hangjie Shi, Desheng Zhang, Lavina Vaz, Leslie Ball, Maureen Murray, Rahul Gupta, Shankar Ananthakrishna  

**Link**: [PDF](https://arxiv.org/pdf/2508.10108)  

**Abstract**: AI systems for software development are rapidly gaining prominence, yet significant challenges remain in ensuring their safety. To address this, Amazon launched the Trusted AI track of the Amazon Nova AI Challenge, a global competition among 10 university teams to drive advances in secure AI. In the challenge, five teams focus on developing automated red teaming bots, while the other five create safe AI assistants. This challenge provides teams with a unique platform to evaluate automated red-teaming and safety alignment methods through head-to-head adversarial tournaments where red teams have multi-turn conversations with the competing AI coding assistants to test their safety alignment. Along with this, the challenge provides teams with a feed of high quality annotated data to fuel iterative improvement. Throughout the challenge, teams developed state-of-the-art techniques, introducing novel approaches in reasoning-based safety alignment, robust model guardrails, multi-turn jail-breaking, and efficient probing of large language models (LLMs). To support these efforts, the Amazon Nova AI Challenge team made substantial scientific and engineering investments, including building a custom baseline coding specialist model for the challenge from scratch, developing a tournament orchestration service, and creating an evaluation harness. This paper outlines the advancements made by university teams and the Amazon Nova AI Challenge team in addressing the safety challenges of AI for software development, highlighting this collaborative effort to raise the bar for AI safety. 

---
# A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions 

**Authors**: Ziyang Xiao, Jingrong Xie, Lilin Xu, Shisi Guan, Jingyan Zhu, Xiongwei Han, Xiaojin Fu, WingYin Yu, Han Wu, Wei Shi, Qingcan Kang, Jiahui Duan, Tao Zhong, Mingxuan Yuan, Jia Zeng, Yuan Wang, Gang Chen, Dongxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10047)  

**Abstract**: By virtue of its great utility in solving real-world problems, optimization modeling has been widely employed for optimal decision-making across various sectors, but it requires substantial expertise from operations research professionals. With the advent of large language models (LLMs), new opportunities have emerged to automate the procedure of mathematical modeling. This survey presents a comprehensive and timely review of recent advancements that cover the entire technical stack, including data synthesis and fine-tuning for the base model, inference frameworks, benchmark datasets, and performance evaluation. In addition, we conducted an in-depth analysis on the quality of benchmark datasets, which was found to have a surprisingly high error rate. We cleaned the datasets and constructed a new leaderboard with fair performance evaluation in terms of base LLM model and datasets. We also build an online portal that integrates resources of cleaned datasets, code and paper repository to benefit the community. Finally, we identify limitations in current methodologies and outline future research opportunities. 

---
# Empirical Investigation into Configuring Echo State Networks for Representative Benchmark Problem Domains 

**Authors**: Brooke R. Weborg, Gursel Serpen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10887)  

**Abstract**: This paper examines Echo State Network, a reservoir computer, performance using four different benchmark problems, then proposes heuristics or rules of thumb for configuring the architecture, as well as the selection of parameters and their values, which are applicable to problems within the same domain, to help serve to fill the experience gap needed by those entering this field of study. The influence of various parameter selections and their value adjustments, as well as architectural changes made to an Echo State Network, a powerful recurrent neural network configured as a reservoir computer, can be challenging to fully comprehend without experience in the field, and even some hyperparameter optimization algorithms may have difficulty adjusting parameter values without proper manual selections made first. Therefore, it is imperative to understand the effects of parameters and their value selection on Echo State Network architecture performance for a successful build. Thus, to address the requirement for an extensive background in Echo State Network architecture, as well as examine how Echo State Network performance is affected with respect to variations in architecture, design, and parameter selection and values, a series of benchmark tasks representing different problem domains, including time series prediction, pattern generation, chaotic system prediction, and time series classification, were modeled and experimented on to show the impact on the performance of Echo State Network. 

---
# ToonComposer: Streamlining Cartoon Production with Generative Post-Keyframing 

**Authors**: Lingen Li, Guangzhi Wang, Zhaoyang Zhang, Yaowei Li, Xiaoyu Li, Qi Dou, Jinwei Gu, Tianfan Xue, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10881)  

**Abstract**: Traditional cartoon and anime production involves keyframing, inbetweening, and colorization stages, which require intensive manual effort. Despite recent advances in AI, existing methods often handle these stages separately, leading to error accumulation and artifacts. For instance, inbetweening approaches struggle with large motions, while colorization methods require dense per-frame sketches. To address this, we introduce ToonComposer, a generative model that unifies inbetweening and colorization into a single post-keyframing stage. ToonComposer employs a sparse sketch injection mechanism to provide precise control using keyframe sketches. Additionally, it uses a cartoon adaptation method with the spatial low-rank adapter to tailor a modern video foundation model to the cartoon domain while keeping its temporal prior intact. Requiring as few as a single sketch and a colored reference frame, ToonComposer excels with sparse inputs, while also supporting multiple sketches at any temporal location for more precise motion control. This dual capability reduces manual workload and improves flexibility, empowering artists in real-world scenarios. To evaluate our model, we further created PKBench, a benchmark featuring human-drawn sketches that simulate real-world use cases. Our evaluation demonstrates that ToonComposer outperforms existing methods in visual quality, motion consistency, and production efficiency, offering a superior and more flexible solution for AI-assisted cartoon production. 

---
# Searching for Privacy Risks in LLM Agents via Simulation 

**Authors**: Yanzhe Zhang, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10880)  

**Abstract**: The widespread deployment of LLM-based agents is likely to introduce a critical privacy threat: malicious agents that proactively engage others in multi-turn interactions to extract sensitive information. These dynamic dialogues enable adaptive attack strategies that can cause severe privacy violations, yet their evolving nature makes it difficult to anticipate and discover sophisticated vulnerabilities manually. To tackle this problem, we present a search-based framework that alternates between improving attacker and defender instructions by simulating privacy-critical agent interactions. Each simulation involves three roles: data subject, data sender, and data recipient. While the data subject's behavior is fixed, the attacker (data recipient) attempts to extract sensitive information from the defender (data sender) through persistent and interactive exchanges. To explore this interaction space efficiently, our search algorithm employs LLMs as optimizers, using parallel search with multiple threads and cross-thread propagation to analyze simulation trajectories and iteratively propose new instructions. Through this process, we find that attack strategies escalate from simple direct requests to sophisticated multi-turn tactics such as impersonation and consent forgery, while defenses advance from rule-based constraints to identity-verification state machines. The discovered attacks and defenses transfer across diverse scenarios and backbone models, demonstrating strong practical utility for building privacy-aware agents. 

---
# A Survey on Diffusion Language Models 

**Authors**: Tianyi Li, Mingda Chen, Bowei Guo, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10875)  

**Abstract**: Diffusion Language Models (DLMs) are rapidly emerging as a powerful and promising alternative to the dominant autoregressive (AR) paradigm. By generating tokens in parallel through an iterative denoising process, DLMs possess inherent advantages in reducing inference latency and capturing bidirectional context, thereby enabling fine-grained control over the generation process. While achieving a several-fold speed-up, recent advancements have allowed DLMs to show performance comparable to their autoregressive counterparts, making them a compelling choice for various natural language processing tasks. In this survey, we provide a holistic overview of the current DLM landscape. We trace its evolution and relationship with other paradigms, such as autoregressive and masked language models, and cover both foundational principles and state-of-the-art models. Our work offers an up-to-date, comprehensive taxonomy and an in-depth analysis of current techniques, from pre-training strategies to advanced post-training methods. Another contribution of this survey is a thorough review of DLM inference strategies and optimizations, including improvements in decoding parallelism, caching mechanisms, and generation quality. We also highlight the latest approaches to multimodal extensions of DLMs and delineate their applications across various practical scenarios. Furthermore, our discussion addresses the limitations and challenges of DLMs, including efficiency, long-sequence handling, and infrastructure requirements, while outlining future research directions to sustain progress in this rapidly evolving field. Project GitHub is available at this https URL. 

---
# TLE-Based A2C Agent for Terrestrial Coverage Orbital Path Planning 

**Authors**: Anantha Narayanan, Battu Bhanu Teja, Pruthwik Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2508.10872)  

**Abstract**: The increasing congestion of Low Earth Orbit (LEO) poses persistent challenges to the efficient deployment and safe operation of Earth observation satellites. Mission planners must now account not only for mission-specific requirements but also for the increasing collision risk with active satellites and space debris. This work presents a reinforcement learning framework using the Advantage Actor-Critic (A2C) algorithm to optimize satellite orbital parameters for precise terrestrial coverage within predefined surface radii. By formulating the problem as a Markov Decision Process (MDP) within a custom OpenAI Gymnasium environment, our method simulates orbital dynamics using classical Keplerian elements. The agent progressively learns to adjust five of the orbital parameters - semi-major axis, eccentricity, inclination, right ascension of ascending node, and the argument of perigee-to achieve targeted terrestrial coverage. Comparative evaluation against Proximal Policy Optimization (PPO) demonstrates A2C's superior performance, achieving 5.8x higher cumulative rewards (10.0 vs 9.263025) while converging in 31.5x fewer timesteps (2,000 vs 63,000). The A2C agent consistently meets mission objectives across diverse target coordinates while maintaining computational efficiency suitable for real-time mission planning applications. Key contributions include: (1) a TLE-based orbital simulation environment incorporating physics constraints, (2) validation of actor-critic methods' superiority over trust region approaches in continuous orbital control, and (3) demonstration of rapid convergence enabling adaptive satellite deployment. This approach establishes reinforcement learning as a computationally efficient alternative for scalable and intelligent LEO mission planning. 

---
# Medico 2025: Visual Question Answering for Gastrointestinal Imaging 

**Authors**: Sushant Gautam, Vajira Thambawita, Michael Riegler, Pål Halvorsen, Steven Hicks  

**Link**: [PDF](https://arxiv.org/pdf/2508.10869)  

**Abstract**: The Medico 2025 challenge addresses Visual Question Answering (VQA) for Gastrointestinal (GI) imaging, organized as part of the MediaEval task series. The challenge focuses on developing Explainable Artificial Intelligence (XAI) models that answer clinically relevant questions based on GI endoscopy images while providing interpretable justifications aligned with medical reasoning. It introduces two subtasks: (1) answering diverse types of visual questions using the Kvasir-VQA-x1 dataset, and (2) generating multimodal explanations to support clinical decision-making. The Kvasir-VQA-x1 dataset, created from 6,500 images and 159,549 complex question-answer (QA) pairs, serves as the benchmark for the challenge. By combining quantitative performance metrics and expert-reviewed explainability assessments, this task aims to advance trustworthy Artificial Intelligence (AI) in medical image analysis. Instructions, data access, and an updated guide for participation are available in the official competition repository: this https URL 

---
# Performance of GPT-5 in Brain Tumor MRI Reasoning 

**Authors**: Mojtaba Safari, Shansong Wang, Mingzhe Hu, Zach Eidex, Qiang Li, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10865)  

**Abstract**: Accurate differentiation of brain tumor types on magnetic resonance imaging (MRI) is critical for guiding treatment planning in neuro-oncology. Recent advances in large language models (LLMs) have enabled visual question answering (VQA) approaches that integrate image interpretation with natural language reasoning. In this study, we evaluated GPT-4o, GPT-5-nano, GPT-5-mini, and GPT-5 on a curated brain tumor VQA benchmark derived from 3 Brain Tumor Segmentation (BraTS) datasets - glioblastoma (GLI), meningioma (MEN), and brain metastases (MET). Each case included multi-sequence MRI triplanar mosaics and structured clinical features transformed into standardized VQA items. Models were assessed in a zero-shot chain-of-thought setting for accuracy on both visual and reasoning tasks. Results showed that GPT-5-mini achieved the highest macro-average accuracy (44.19%), followed by GPT-5 (43.71%), GPT-4o (41.49%), and GPT-5-nano (35.85%). Performance varied by tumor subtype, with no single model dominating across all cohorts. These findings suggest that GPT-5 family models can achieve moderate accuracy in structured neuro-oncological VQA tasks, but not at a level acceptable for clinical use. 

---
# From Black Box to Transparency: Enhancing Automated Interpreting Assessment with Explainable AI in College Classrooms 

**Authors**: Zhaokun Jiang, Ziyin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10860)  

**Abstract**: Recent advancements in machine learning have spurred growing interests in automated interpreting quality assessment. Nevertheless, existing research suffers from insufficient examination of language use quality, unsatisfactory modeling effectiveness due to data scarcity and imbalance, and a lack of efforts to explain model predictions. To address these gaps, we propose a multi-dimensional modeling framework that integrates feature engineering, data augmentation, and explainable machine learning. This approach prioritizes explainability over ``black box'' predictions by utilizing only construct-relevant, transparent features and conducting Shapley Value (SHAP) analysis. Our results demonstrate strong predictive performance on a novel English-Chinese consecutive interpreting dataset, identifying BLEURT and CometKiwi scores to be the strongest predictive features for fidelity, pause-related features for fluency, and Chinese-specific phraseological diversity metrics for language use. Overall, by placing particular emphasis on explainability, we present a scalable, reliable, and transparent alternative to traditional human evaluation, facilitating the provision of detailed diagnostic feedback for learners and supporting self-regulated learning advantages not afforded by automated scores in isolation. 

---
# Reinforced Language Models for Sequential Decision Making 

**Authors**: Jim Dilkes, Vahid Yazdanpanah, Sebastian Stein  

**Link**: [PDF](https://arxiv.org/pdf/2508.10839)  

**Abstract**: Large Language Models (LLMs) show potential as sequential decision-making agents, but their application is often limited due to a reliance on large, computationally expensive models. This creates a need to improve smaller models, yet existing post-training methods are designed for single-turn interactions and cannot handle credit assignment in multi-step agentic tasks. To address this, we introduce Multi-Step Group-Relative Policy Optimization (MS-GRPO), a new algorithm for post-training LLM agents, grounded in formal Text-Mediated Stochastic Game (TSMG) and Language-Agent Policy (LAP) frameworks. For credit assignment, MS-GRPO attributes the entire cumulative episode reward to each individual episode step. We supplement this algorithm with a novel absolute-advantage-weighted episode sampling strategy that we show improves training performance. We evaluate our approach by post-training a 3-billion parameter model on Snake and Frozen Lake. Our experiments demonstrate that the method is effective in improving decision-making performance: our post-trained 3B parameter model outperforms a 72B parameter baseline by 50% on the Frozen Lake task. This work demonstrates that targeted post-training is a practical and efficient alternative to relying on model scale for creating sequential decision-making agents using LLMs. 

---
# A Multimodal Neural Network for Recognizing Subjective Self-Disclosure Towards Social Robots 

**Authors**: Henry Powell, Guy Laban, Emily S. Cross  

**Link**: [PDF](https://arxiv.org/pdf/2508.10828)  

**Abstract**: Subjective self-disclosure is an important feature of human social interaction. While much has been done in the social and behavioural literature to characterise the features and consequences of subjective self-disclosure, little work has been done thus far to develop computational systems that are able to accurately model it. Even less work has been done that attempts to model specifically how human interactants self-disclose with robotic partners. It is becoming more pressing as we require social robots to work in conjunction with and establish relationships with humans in various social settings. In this paper, our aim is to develop a custom multimodal attention network based on models from the emotion recognition literature, training this model on a large self-collected self-disclosure video corpus, and constructing a new loss function, the scale preserving cross entropy loss, that improves upon both classification and regression versions of this problem. Our results show that the best performing model, trained with our novel loss function, achieves an F1 score of 0.83, an improvement of 0.48 from the best baseline model. This result makes significant headway in the aim of allowing social robots to pick up on an interaction partner's self-disclosures, an ability that will be essential in social robots with social cognition. 

---
# The SET Perceptual Factors Framework: Towards Assured Perception for Autonomous Systems 

**Authors**: Troi Williams  

**Link**: [PDF](https://arxiv.org/pdf/2508.10798)  

**Abstract**: Future autonomous systems promise significant societal benefits, yet their deployment raises concerns about safety and trustworthiness. A key concern is assuring the reliability of robot perception, as perception seeds safe decision-making. Failures in perception are often due to complex yet common environmental factors and can lead to accidents that erode public trust. To address this concern, we introduce the SET (Self, Environment, and Target) Perceptual Factors Framework. We designed the framework to systematically analyze how factors such as weather, occlusion, or sensor limitations negatively impact perception. To achieve this, the framework employs SET State Trees to categorize where such factors originate and SET Factor Trees to model how these sources and factors impact perceptual tasks like object detection or pose estimation. Next, we develop Perceptual Factor Models using both trees to quantify the uncertainty for a given task. Our framework aims to promote rigorous safety assurances and cultivate greater public understanding and trust in autonomous systems by offering a transparent and standardized method for identifying, modeling, and communicating perceptual risks. 

---
# Enhancing Fairness in Autoencoders for Node-Level Graph Anomaly Detection 

**Authors**: Shouju Wang, Yuchen Song, Sheng'en Li, Dongmian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2508.10785)  

**Abstract**: Graph anomaly detection (GAD) has become an increasingly important task across various domains. With the rapid development of graph neural networks (GNNs), GAD methods have achieved significant performance improvements. However, fairness considerations in GAD remain largely underexplored. Indeed, GNN-based GAD models can inherit and amplify biases present in training data, potentially leading to unfair outcomes. While existing efforts have focused on developing fair GNNs, most approaches target node classification tasks, where models often rely on simple layer architectures rather than autoencoder-based structures, which are the most widely used architecturs for anomaly detection. To address fairness in autoencoder-based GAD models, we propose \textbf{D}is\textbf{E}ntangled \textbf{C}ounterfactual \textbf{A}dversarial \textbf{F}air (DECAF)-GAD, a framework that alleviates bias while preserving GAD performance. Specifically, we introduce a structural causal model (SCM) to disentangle sensitive attributes from learned representations. Based on this causal framework, we formulate a specialized autoencoder architecture along with a fairness-guided loss function. Through extensive experiments on both synthetic and real-world datasets, we demonstrate that DECAF-GAD not only achieves competitive anomaly detection performance but also significantly enhances fairness metrics compared to baseline GAD methods. Our code is available at this https URL. 

---
# Ultra-High-Definition Reference-Based Landmark Image Super-Resolution with Generative Diffusion Prior 

**Authors**: Zhenning Shi, Zizheng Yan, Yuhang Yu, Clara Xue, Jingyu Zhuang, Qi Zhang, Jinwei Chen, Tao Li, Qingnan Fan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10779)  

**Abstract**: Reference-based Image Super-Resolution (RefSR) aims to restore a low-resolution (LR) image by utilizing the semantic and texture information from an additional reference high-resolution (reference HR) image. Existing diffusion-based RefSR methods are typically built upon ControlNet, which struggles to effectively align the information between the LR image and the reference HR image. Moreover, current RefSR datasets suffer from limited resolution and poor image quality, resulting in the reference images lacking sufficient fine-grained details to support high-quality restoration. To overcome the limitations above, we propose TriFlowSR, a novel framework that explicitly achieves pattern matching between the LR image and the reference HR image. Meanwhile, we introduce Landmark-4K, the first RefSR dataset for Ultra-High-Definition (UHD) landmark scenarios. Considering the UHD scenarios with real-world degradation, in TriFlowSR, we design a Reference Matching Strategy to effectively match the LR image with the reference HR image. Experimental results show that our approach can better utilize the semantic and texture information of the reference HR image compared to previous methods. To the best of our knowledge, we propose the first diffusion-based RefSR pipeline for ultra-high definition landmark scenarios under real-world degradation. Our code and model will be available at this https URL. 

---
# Estimating Covariance for Global Minimum Variance Portfolio: A Decision-Focused Learning Approach 

**Authors**: Juchan Kim, Inwoo Tae, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.10776)  

**Abstract**: Portfolio optimization constitutes a cornerstone of risk management by quantifying the risk-return trade-off. Since it inherently depends on accurate parameter estimation under conditions of future uncertainty, the selection of appropriate input parameters is critical for effective portfolio construction. However, most conventional statistical estimators and machine learning algorithms determine these parameters by minimizing mean-squared error (MSE), a criterion that can yield suboptimal investment decisions. In this paper, we adopt decision-focused learning (DFL) - an approach that directly optimizes decision quality rather than prediction error such as MSE - to derive the global minimum-variance portfolio (GMVP). Specifically, we theoretically derive the gradient of decision loss using the analytic solution of GMVP and its properties regarding the principal components of itself. Through extensive empirical evaluation, we show that prediction-focused estimation methods may fail to produce optimal allocations in practice, whereas DFL-based methods consistently deliver superior decision performance. Furthermore, we provide a comprehensive analysis of DFL's mechanism in GMVP construction, focusing on its volatility reduction capability, decision-driving features, and estimation characteristics. 

---
# Video-BLADE: Block-Sparse Attention Meets Step Distillation for Efficient Video Generation 

**Authors**: Youping Gu, Xiaolong Li, Yuhao Hu, Bohan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10774)  

**Abstract**: Diffusion transformers currently lead the field in high-quality video generation, but their slow iterative denoising process and prohibitive quadratic attention costs for long sequences create significant inference bottlenecks. While both step distillation and sparse attention mechanisms have shown promise as independent acceleration strategies, effectively combining these approaches presents critical challenges -- training-free integration yields suboptimal results, while separately training sparse attention after step distillation requires prohibitively expensive high-quality video data. To overcome these limitations, we propose BLADE, an innovative data-free joint training framework that introduces: (1) an Adaptive Block-Sparse Attention (ASA) mechanism for dynamically generating content-aware sparsity masks to focus computation on salient spatiotemporal features, and (2) a sparsity-aware step distillation paradigm built upon Trajectory Distribution Matching (TDM) that directly incorporates sparsity into the distillation process rather than treating it as a separate compression step, with fast convergence. We validate BLADE on text-to-video models like CogVideoX-5B and Wan2.1-1.3B. Our framework demonstrates remarkable efficiency gains across different scales. On Wan2.1-1.3B, BLADE achieves a 14.10x end-to-end inference acceleration over a 50-step baseline. Moreover, on models such as CogVideoX-5B with short video sequence lengths, our framework delivers a robust 8.89x speedup. Crucially, the acceleration is accompanied by a consistent quality improvement. On the VBench-2.0 benchmark, BLADE boosts the score of CogVideoX-5B to 0.569 (from 0.534) and Wan2.1-1.3B to 0.570 (from 0.563), results that are further corroborated by superior ratings in human evaluations. Our code and model weights are publicly available at: this http URL. 

---
# AEGIS: Authenticity Evaluation Benchmark for AI-Generated Video Sequences 

**Authors**: Jieyu Li, Xin Zhang, Joey Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.10771)  

**Abstract**: Recent advances in AI-generated content have fueled the rise of highly realistic synthetic videos, posing severe risks to societal trust and digital integrity. Existing benchmarks for video authenticity detection typically suffer from limited realism, insufficient scale, and inadequate complexity, failing to effectively evaluate modern vision-language models against sophisticated forgeries. To address this critical gap, we introduce AEGIS, a novel large-scale benchmark explicitly targeting the detection of hyper-realistic and semantically nuanced AI-generated videos. AEGIS comprises over 10,000 rigorously curated real and synthetic videos generated by diverse, state-of-the-art generative models, including Stable Video Diffusion, CogVideoX-5B, KLing, and Sora, encompassing open-source and proprietary architectures. In particular, AEGIS features specially constructed challenging subsets enhanced with robustness evaluation. Furthermore, we provide multimodal annotations spanning Semantic-Authenticity Descriptions, Motion Features, and Low-level Visual Features, facilitating authenticity detection and supporting downstream tasks such as multimodal fusion and forgery localization. Extensive experiments using advanced vision-language models demonstrate limited detection capabilities on the most challenging subsets of AEGIS, highlighting the dataset's unique complexity and realism beyond the current generalization capabilities of existing models. In essence, AEGIS establishes an indispensable evaluation benchmark, fundamentally advancing research toward developing genuinely robust, reliable, broadly generalizable video authenticity detection methodologies capable of addressing real-world forgery threats. Our dataset is available on this https URL. 

---
# FROGENT: An End-to-End Full-process Drug Design Agent 

**Authors**: Qihua Pan, Dong Xu, Jenna Xinyi Yao, Lijia Ma, Zexuan Zhu, Junkai Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.10760)  

**Abstract**: Powerful AI tools for drug discovery reside in isolated web apps, desktop programs, and code libraries. Such fragmentation forces scientists to manage incompatible interfaces and specialized scripts, which can be a cumbersome and repetitive process. To address this issue, a Full-pROcess druG dEsign ageNT, named FROGENT, has been proposed. Specifically, FROGENT utilizes a Large Language Model and the Model Context Protocol to integrate multiple dynamic biochemical databases, extensible tool libraries, and task-specific AI models. This agentic framework allows FROGENT to execute complicated drug discovery workflows dynamically, including component tasks such as target identification, molecule generation and retrosynthetic planning. FROGENT has been evaluated on eight benchmarks that cover various aspects of drug discovery, such as knowledge retrieval, property prediction, virtual screening, mechanistic analysis, molecular design, and synthesis. It was compared against six increasingly advanced ReAct-style agents that support code execution and literature searches. Empirical results demonstrated that FROGENT triples the best baseline performance in hit-finding and doubles it in interaction profiling, significantly outperforming both the open-source model Qwen3-32B and the commercial model GPT-4o. In addition, real-world cases have been utilized to validate the practicability and generalization of FROGENT. This development suggests that streamlining the agentic drug discovery pipeline can significantly enhance researcher productivity. 

---
# Natively Trainable Sparse Attention for Hierarchical Point Cloud Datasets 

**Authors**: Nicolas Lapautre, Maria Marchenko, Carlos Miguel Patiño, Xin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.10758)  

**Abstract**: Unlocking the potential of transformers on datasets of large physical systems depends on overcoming the quadratic scaling of the attention mechanism. This work explores combining the Erwin architecture with the Native Sparse Attention (NSA) mechanism to improve the efficiency and receptive field of transformer models for large-scale physical systems, addressing the challenge of quadratic attention complexity. We adapt the NSA mechanism for non-sequential data, implement the Erwin NSA model, and evaluate it on three datasets from the physical sciences -- cosmology simulations, molecular dynamics, and air pressure modeling -- achieving performance that matches or exceeds that of the original Erwin model. Additionally, we reproduce the experimental results from the Erwin paper to validate their implementation. 

---
# Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models 

**Authors**: Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, Guang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10751)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR), which typically adopts Pass@1 as the reward, has faced the issues in balancing exploration and exploitation, causing policies to prefer conservative actions, converging to a local optimum. Identifying an appropriate reward metric is therefore crucial. Regarding the prior work, although Pass@k has been used in evaluation, its connection to LLM exploration ability in RLVR remains largely overlooked. To investigate this, we first use Pass@k as the reward to train the policy model (i.e., $\textbf{Pass@k Training}$), and observe the improvement on its exploration ability. Next, we derive an analytical solution for the advantage of Pass@k Training, leading to an efficient and effective process. Building on this, our analysis reveals that exploration and exploitation are not inherently conflicting objectives, while they can mutually enhance each other. Moreover, Pass@k Training with analytical derivation essentially involves directly designing the advantage function. Inspired by this, we preliminarily explore the advantage design for RLVR, showing promising results and highlighting a potential future direction. 

---
# APFL: Analytic Personalized Federated Learning via Dual-Stream Least Squares 

**Authors**: Kejia Fan, Jianheng Tang, Zhirui Yang, Feijiang Han, Jiaxu Li, Run He, Yajiang Huang, Anfeng Liu, Houbing Herbert Song, Yunhuai Liu, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10732)  

**Abstract**: Personalized Federated Learning (PFL) has presented a significant challenge to deliver personalized models to individual clients through collaborative training. Existing PFL methods are often vulnerable to non-IID data, which severely hinders collective generalization and then compromises the subsequent personalization efforts. In this paper, to address this non-IID issue in PFL, we propose an Analytic Personalized Federated Learning (APFL) approach via dual-stream least squares. In our APFL, we use a foundation model as a frozen backbone for feature extraction. Subsequent to the feature extractor, we develop dual-stream analytic models to achieve both collective generalization and individual personalization. Specifically, our APFL incorporates a shared primary stream for global generalization across all clients, and a dedicated refinement stream for local personalization of each individual client. The analytical solutions of our APFL enable its ideal property of heterogeneity invariance, theoretically meaning that each personalized model remains identical regardless of how heterogeneous the data are distributed across all other clients. Empirical results across various datasets also validate the superiority of our APFL over state-of-the-art baselines, with advantages of at least 1.10%-15.45% in accuracy. 

---
# EgoCross: Benchmarking Multimodal Large Language Models for Cross-Domain Egocentric Video Question Answering 

**Authors**: Yanjun Li, Yuqian Fu, Tianwen Qian, Qi'ao Xu, Silong Dai, Danda Pani Paudel, Luc Van Gool, Xiaoling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10729)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have significantly pushed the frontier of egocentric video question answering (EgocentricQA). However, existing benchmarks and studies are mainly limited to common daily activities such as cooking and cleaning. In contrast, real-world deployment inevitably encounters domain shifts, where target domains differ substantially in both visual style and semantic content. To bridge this gap, we introduce \textbf{EgoCross}, a comprehensive benchmark designed to evaluate the cross-domain generalization of MLLMs in EgocentricQA. EgoCross covers four diverse and challenging domains, including surgery, industry, extreme sports, and animal perspective, representing realistic and high-impact application scenarios. It comprises approximately 1,000 QA pairs across 798 video clips, spanning four key QA tasks: prediction, recognition, localization, and counting. Each QA pair provides both OpenQA and CloseQA formats to support fine-grained evaluation. Extensive experiments show that most existing MLLMs, whether general-purpose or egocentric-specialized, struggle to generalize to domains beyond daily life, highlighting the limitations of current models. Furthermore, we conduct several pilot studies, \eg, fine-tuning and reinforcement learning, to explore potential improvements. We hope EgoCross and our accompanying analysis will serve as a foundation for advancing domain-adaptive, robust egocentric video understanding. Data and codes will be released at: \href{this https URL}{this https URL.} 

---
# Electromagnetic Simulations of Antennas on GPUs for Machine Learning Applications 

**Authors**: Murat Temiz, Vemund Bakken  

**Link**: [PDF](https://arxiv.org/pdf/2508.10713)  

**Abstract**: This study proposes an antenna simulation framework powered by graphics processing units (GPUs) based on an open-source electromagnetic (EM) simulation software (gprMax) for machine learning applications of antenna design and optimization. Furthermore, it compares the simulation results with those obtained through commercial EM software. The proposed software framework for machine learning and surrogate model applications will produce antenna data sets consisting of a large number of antenna simulation results using GPUs. Although machine learning methods can attain the optimum solutions for many problems, they are known to be data-hungry and require a great deal of samples for the training stage of the algorithms. However, producing a sufficient number of training samples in EM applications within a limited time is challenging due to the high computational complexity of EM simulations. Therefore, GPUs are utilized in this study to simulate a large number of antennas with predefined or random antenna shape parameters to produce data sets. Moreover, this study also compares various machine learning and deep learning models in terms of antenna parameter estimation performance. This study demonstrates that an entry-level GPU substantially outperforms a high-end CPU in terms of computational performance, while a high-end gaming GPU can achieve around 18 times more computational performance compared to a high-end CPU. Moreover, it is shown that the open-source EM simulation software can deliver similar results to those obtained via commercial software in the simulation of microstrip antennas when the spatial resolution of the simulations is sufficiently fine. 

---
# REFN: A Reinforcement-Learning-From-Network Framework against 1-day/n-day Exploitations 

**Authors**: Tianlong Yu, Lihong Liu, Ziyi Zhou, Fudu Xing, Kailong Wang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10701)  

**Abstract**: The exploitation of 1 day or n day vulnerabilities poses severe threats to networked devices due to massive deployment scales and delayed patching (average Mean Time To Patch exceeds 60 days). Existing defenses, including host based patching and network based filtering, are inadequate due to limited scalability across diverse devices, compatibility issues especially with embedded or legacy systems, and error prone deployment process (manual patch validation). To address these issues, we introduce REFN (Reinforcement Learning From Network), a novel framework that trains Large Language Models (LLMs) to autonomously generate network filters to prevent 1 day or n day exploitations. REFN ensures scalability by uniquely employs Reinforcement Learning (RL) driven by online network rewards instead of traditional Human Feedback (RLHF). REFN guarantees compatibility via unified deployment on edge security gateways (Amazon Eero). REFN provides robustness via online validation using real network traffic. Crucially, REFN addresses three core challenges in training LLMs for exploit prevention: 1) expanding current LLMs limited vulnerability fixing expertise via Agentic RAG based Knowledge Distillation, 2) bridging current LLMs language to network gaps through an RL From VNF Pipeline that translates language context (vulnerability description) into network enforcement, 3) addressing the LLM hallucination and non determinism via the Online Agentic Validation that penalizes erroneous outputs. Evaluated across 22 families of 1 day or n day exploits, REFN demonstrates effectiveness (21.1 percent higher accuracy than alternatives), efficiency (Mean Time To Patch of 3.65 hours) and scalability (easily scale to 10K devices). REFN serves as an initial step toward training LLMs to rapidly prevent massive scale 1 day or n day exploitations. 

---
# Learning from Natural Language Feedback for Personalized Question Answering 

**Authors**: Alireza Salemi, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2508.10695)  

**Abstract**: Personalization is crucial for enhancing both the effectiveness and user satisfaction of language technologies, particularly in information-seeking tasks like question answering. Current approaches for personalizing large language models (LLMs) often rely on retrieval-augmented generation (RAG), followed by reinforcement learning with scalar reward signals to teach models how to use retrieved personal context. We believe that these scalar rewards sometimes provide weak, non-instructive feedback, limiting learning efficiency and personalization quality. We introduce VAC, a novel framework for personalized response generation that replaces scalar rewards with natural language feedback (NLF) that are generated conditioned on the user profiles and the question narratives. NLF serves as a rich and actionable supervision signal, allowing the policy model to iteratively refine its outputs and internalize effective personalization strategies. Training alternates between optimizing the feedback model and fine-tuning the policy model on the improved responses, resulting in a policy model that no longer requires feedback at inference. Evaluation on the LaMP-QA benchmark that consists of three diverse domains demonstrates consistent and significant improvements over the state-of-the-art results. Human evaluations further confirm the superior quality of the generated responses. These results demonstrate that NLF provides more effective signals for optimizing personalized question answering. 

---
# Continuous Bangla Sign Language Translation: Mitigating the Expense of Gloss Annotation with the Assistance of Graph 

**Authors**: Safaeid Hossain Arib, Rabeya Akter, Sejuti Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2508.10687)  

**Abstract**: Millions of individuals worldwide are affected by deafness and hearing impairment. Sign language serves as a sophisticated means of communication for the deaf and hard of hearing. However, in societies that prioritize spoken languages, sign language often faces underestimation, leading to communication barriers and social exclusion. The Continuous Bangla Sign Language Translation project aims to address this gap by enhancing translation methods. While recent approaches leverage transformer architecture for state-of-the-art results, our method integrates graph-based methods with the transformer architecture. This fusion, combining transformer and STGCN-LSTM architectures, proves more effective in gloss-free translation. Our contributions include architectural fusion, exploring various fusion strategies, and achieving a new state-of-the-art performance on diverse sign language datasets, namely RWTH-PHOENIX-2014T, CSL-Daily, How2Sign, and BornilDB v1.0. Our approach demonstrates superior performance compared to current translation outcomes across all datasets, showcasing notable improvements of BLEU-4 scores of 4.01, 2.07, and 0.5, surpassing those of GASLT, GASLT and slt_how2sign in RWTH-PHOENIX-2014T, CSL-Daily, and How2Sign, respectively. Also, we introduce benchmarking on the BornilDB v1.0 dataset for the first time. Our method sets a benchmark for future research, emphasizing the importance of gloss-free translation to improve communication accessibility for the deaf and hard of hearing. 

---
# Hybrid Generative Fusion for Efficient and Privacy-Preserving Face Recognition Dataset Generation 

**Authors**: Feiran Li, Qianqian Xu, Shilong Bao, Boyu Han, Zhiyong Yang, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10672)  

**Abstract**: In this paper, we present our approach to the DataCV ICCV Challenge, which centers on building a high-quality face dataset to train a face recognition model. The constructed dataset must not contain identities overlapping with any existing public face datasets. To handle this challenge, we begin with a thorough cleaning of the baseline HSFace dataset, identifying and removing mislabeled or inconsistent identities through a Mixture-of-Experts (MoE) strategy combining face embedding clustering and GPT-4o-assisted verification. We retain the largest consistent identity cluster and apply data augmentation up to a fixed number of images per identity. To further diversify the dataset, we generate synthetic identities using Stable Diffusion with prompt engineering. As diffusion models are computationally intensive, we generate only one reference image per identity and efficiently expand it using Vec2Face, which rapidly produces 49 identity-consistent variants. This hybrid approach fuses GAN-based and diffusion-based samples, enabling efficient construction of a diverse and high-quality dataset. To address the high visual similarity among synthetic identities, we adopt a curriculum learning strategy by placing them early in the training schedule, allowing the model to progress from easier to harder samples. Our final dataset contains 50 images per identity, and all newly generated identities are checked with mainstream face datasets to ensure no identity leakage. Our method achieves \textbf{1st place} in the competition, and experimental results show that our dataset improves model performance across 10K, 20K, and 100K identity scales. Code is available at this https URL. 

---
# AddressVLM: Cross-view Alignment Tuning for Image Address Localization using Large Vision-Language Models 

**Authors**: Shixiong Xu, Chenghao Zhang, Lubin Fan, Yuan Zhou, Bin Fan, Shiming Xiang, Gaofeng Meng, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.10667)  

**Abstract**: Large visual language models (LVLMs) have demonstrated impressive performance in coarse-grained geo-localization at the country or city level, but they struggle with fine-grained street-level localization within urban areas. In this paper, we explore integrating city-wide address localization capabilities into LVLMs, facilitating flexible address-related question answering using street-view images. A key challenge is that the street-view visual question-and-answer (VQA) data provides only microscopic visual cues, leading to subpar performance in fine-tuned models. To tackle this issue, we incorporate perspective-invariant satellite images as macro cues and propose cross-view alignment tuning including a satellite-view and street-view image grafting mechanism, along with an automatic label generation mechanism. Then LVLM's global understanding of street distribution is enhanced through cross-view matching. Our proposed model, named AddressVLM, consists of two-stage training protocols: cross-view alignment tuning and address localization tuning. Furthermore, we have constructed two street-view VQA datasets based on image address localization datasets from Pittsburgh and San Francisco. Qualitative and quantitative evaluations demonstrate that AddressVLM outperforms counterpart LVLMs by over 9% and 12% in average address localization accuracy on these two datasets, respectively. 

---
# Deep Learning in Classical and Quantum Physics 

**Authors**: Timothy Heightman, Marcin Płodzień  

**Link**: [PDF](https://arxiv.org/pdf/2508.10666)  

**Abstract**: Scientific progress is tightly coupled to the emergence of new research tools. Today, machine learning (ML)-especially deep learning (DL)-has become a transformative instrument for quantum science and technology. Owing to the intrinsic complexity of quantum systems, DL enables efficient exploration of large parameter spaces, extraction of patterns from experimental data, and data-driven guidance for research directions. These capabilities already support tasks such as refining quantum control protocols and accelerating the discovery of materials with targeted quantum properties, making ML/DL literacy an essential skill for the next generation of quantum scientists. At the same time, DL's power brings risks: models can overfit noisy data, obscure causal structure, and yield results with limited physical interpretability. Recognizing these limitations and deploying mitigation strategies is crucial for scientific rigor. These lecture notes provide a comprehensive, graduate-level introduction to DL for quantum applications, combining conceptual exposition with hands-on examples. Organized as a progressive sequence, they aim to equip readers to decide when and how to apply DL effectively, to understand its practical constraints, and to adapt AI methods responsibly to problems across quantum physics, chemistry, and engineering. 

---
# Serial Over Parallel: Learning Continual Unification for Multi-Modal Visual Object Tracking and Benchmarking 

**Authors**: Zhangyong Tang, Tianyang Xu, Xuefeng Zhu, Chunyang Cheng, Tao Zhou, Xiaojun Wu, Josef Kittler  

**Link**: [PDF](https://arxiv.org/pdf/2508.10655)  

**Abstract**: Unifying multiple multi-modal visual object tracking (MMVOT) tasks draws increasing attention due to the complementary nature of different modalities in building robust tracking systems. Existing practices mix all data sensor types in a single training procedure, structuring a parallel paradigm from the data-centric perspective and aiming for a global optimum on the joint distribution of the involved tasks. However, the absence of a unified benchmark where all types of data coexist forces evaluations on separated benchmarks, causing \textit{inconsistency} between training and testing, thus leading to performance \textit{degradation}. To address these issues, this work advances in two aspects: \ding{182} A unified benchmark, coined as UniBench300, is introduced to bridge the inconsistency by incorporating multiple task data, reducing inference passes from three to one and cutting time consumption by 27\%. \ding{183} The unification process is reformulated in a serial format, progressively integrating new tasks. In this way, the performance degradation can be specified as knowledge forgetting of previous tasks, which naturally aligns with the philosophy of continual learning (CL), motivating further exploration of injecting CL into the unification process. Extensive experiments conducted on two baselines and four benchmarks demonstrate the significance of UniBench300 and the superiority of CL in supporting a stable unification process. Moreover, while conducting dedicated analyses, the performance degradation is found to be negatively correlated with network capacity. Additionally, modality discrepancies contribute to varying degradation levels across tasks (RGBT > RGBD > RGBE in MMVOT), offering valuable insights for future multi-modal vision research. Source codes and the proposed benchmark is available at \textit{this https URL}. 

---
# SPHENIC: Topology-Informed Multi-View Clustering for Spatial Transcriptomics 

**Authors**: Chenkai Guo, Yikai Zhu, Jing Yangum, Renxiang Guan, Por Lip Yee, Guangdun Peng, Dayu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10646)  

**Abstract**: By incorporating spatial location information, spatial-transcriptomics clustering yields more comprehensive insights into cell subpopulation identification. Despite recent progress, existing methods have at least two limitations: (i) topological learning typically considers only representations of individual cells or their interaction graphs; however, spatial transcriptomic profiles are often noisy, making these approaches vulnerable to low-quality topological signals, and (ii) insufficient modeling of spatial neighborhood information leads to low-quality spatial embeddings. To address these limitations, we propose SPHENIC, a novel Spatial Persistent Homology Enhanced Neighborhood Integrative Clustering method. Specifically, SPHENIC incorporates invariant topological features into the clustering network to achieve stable representation learning. Additionally, to construct high-quality spatial embeddings that reflect the true cellular distribution, we design the Spatial Constraint and Distribution Optimization Module (SCDOM). This module increases the similarity between a cell's embedding and those of its spatial neighbors, decreases similarity with non-neighboring cells, and thereby produces clustering-friendly spatial embeddings. Extensive experiments on 14 benchmark spatial transcriptomic slices demonstrate that SPHENIC achieves superior performance on the spatial clustering task, outperforming existing state-of-the-art methods by 3.31%-6.54% over the best alternative. 

---
# Fourier-Guided Attention Upsampling for Image Super-Resolution 

**Authors**: Daejune Choi, Youchan No, Jinhyung Lee, Duksu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.10616)  

**Abstract**: We propose Frequency-Guided Attention (FGA), a lightweight upsampling module for single image super-resolution. Conventional upsamplers, such as Sub-Pixel Convolution, are efficient but frequently fail to reconstruct high-frequency details and introduce aliasing artifacts. FGA addresses these issues by integrating (1) a Fourier feature-based Multi-Layer Perceptron (MLP) for positional frequency encoding, (2) a cross-resolution Correlation Attention Layer for adaptive spatial alignment, and (3) a frequency-domain L1 loss for spectral fidelity supervision. Adding merely 0.3M parameters, FGA consistently enhances performance across five diverse super-resolution backbones in both lightweight and full-capacity scenarios. Experimental results demonstrate average PSNR gains of 0.12~0.14 dB and improved frequency-domain consistency by up to 29%, particularly evident on texture-rich datasets. Visual and spectral evaluations confirm FGA's effectiveness in reducing aliasing and preserving fine details, establishing it as a practical, scalable alternative to traditional upsampling methods. 

---
# On Spectral Properties of Gradient-based Explanation Methods 

**Authors**: Amir Mehrpanah, Erik Englesson, Hossein Azizpour  

**Link**: [PDF](https://arxiv.org/pdf/2508.10595)  

**Abstract**: Understanding the behavior of deep networks is crucial to increase our confidence in their results. Despite an extensive body of work for explaining their predictions, researchers have faced reliability issues, which can be attributed to insufficient formalism. In our research, we adopt novel probabilistic and spectral perspectives to formally analyze explanation methods. Our study reveals a pervasive spectral bias stemming from the use of gradient, and sheds light on some common design choices that have been discovered experimentally, in particular, the use of squared gradient and input perturbation. We further characterize how the choice of perturbation hyperparameters in explanation methods, such as SmoothGrad, can lead to inconsistent explanations and introduce two remedies based on our proposed formalism: (i) a mechanism to determine a standard perturbation scale, and (ii) an aggregation method which we call SpectralLens. Finally, we substantiate our theoretical results through quantitative evaluations. 

---
# FreeGAD: A Training-Free yet Effective Approach for Graph Anomaly Detection 

**Authors**: Yunfeng Zhao, Yixin Liu, Shiyuan Li, Qingfeng Chen, Yu Zheng, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10594)  

**Abstract**: Graph Anomaly Detection (GAD) aims to identify nodes that deviate from the majority within a graph, playing a crucial role in applications such as social networks and e-commerce. Despite the current advancements in deep learning-based GAD, existing approaches often suffer from high deployment costs and poor scalability due to their complex and resource-intensive training processes. Surprisingly, our empirical findings suggest that the training phase of deep GAD methods, commonly perceived as crucial, may actually contribute less to anomaly detection performance than expected. Inspired by this, we propose FreeGAD, a novel training-free yet effective GAD method. Specifically, it leverages an affinity-gated residual encoder to generate anomaly-aware representations. Meanwhile, FreeGAD identifies anchor nodes as pseudo-normal and anomalous guides, followed by calculating anomaly scores through anchor-guided statistical deviations. Extensive experiments demonstrate that FreeGAD achieves superior anomaly detection performance, efficiency, and scalability on multiple benchmark datasets from diverse domains, without any training or iterative optimization. 

---
# Fake Speech Wild: Detecting Deepfake Speech on Social Media Platform 

**Authors**: Yuankun Xie, Ruibo Fu, Xiaopeng Wang, Zhiyong Wang, Ya Li, Zhengqi Wen, Haonnan Cheng, Long Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.10559)  

**Abstract**: The rapid advancement of speech generation technology has led to the widespread proliferation of deepfake speech across social media platforms. While deepfake audio countermeasures (CMs) achieve promising results on public datasets, their performance degrades significantly in cross-domain scenarios. To advance CMs for real-world deepfake detection, we first propose the Fake Speech Wild (FSW) dataset, which includes 254 hours of real and deepfake audio from four different media platforms, focusing on social media. As CMs, we establish a benchmark using public datasets and advanced selfsupervised learning (SSL)-based CMs to evaluate current CMs in real-world scenarios. We also assess the effectiveness of data augmentation strategies in enhancing CM robustness for detecting deepfake speech on social media. Finally, by augmenting public datasets and incorporating the FSW training set, we significantly advanced real-world deepfake audio detection performance, achieving an average equal error rate (EER) of 3.54% across all evaluation sets. 

---
# PTQAT: A Hybrid Parameter-Efficient Quantization Algorithm for 3D Perception Tasks 

**Authors**: Xinhao Wang, Zhiwei Lin, Zhongyu Xia, Yongtao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10557)  

**Abstract**: Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) represent two mainstream model quantization approaches. However, PTQ often leads to unacceptable performance degradation in quantized models, while QAT imposes substantial GPU memory requirements and extended training time due to weight this http URL this paper, we propose PTQAT, a novel general hybrid quantization algorithm for the efficient deployment of 3D perception networks. To address the speed accuracy trade-off between PTQ and QAT, our method selects critical layers for QAT fine-tuning and performs PTQ on the remaining layers. Contrary to intuition, fine-tuning the layers with smaller output discrepancies before and after quantization, rather than those with larger discrepancies, actually leads to greater improvements in the model's quantization accuracy. This means we better compensate for quantization errors during their propagation, rather than addressing them at the point where they occur. The proposed PTQAT achieves similar performance to QAT with more efficiency by freezing nearly 50% of quantifiable layers. Additionally, PTQAT is a universal quantization method that supports various quantization bit widths (4 bits) as well as different model architectures, including CNNs and Transformers. The experimental results on nuScenes across diverse 3D perception tasks, including object detection, semantic segmentation, and occupancy prediction, show that our method consistently outperforms QAT-only baselines. Notably, it achieves 0.2%-0.9% NDS and 0.3%-1.0% mAP gains in object detection, 0.3%-2.0% mIoU gains in semantic segmentation and occupancy prediction while fine-tuning fewer weights. 

---
# Retrieval-Augmented Prompt for OOD Detection 

**Authors**: Ruisong Han, Zongbo Han, Jiahao Zhang, Mingyue Cheng, Changqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10556)  

**Abstract**: Out-of-Distribution (OOD) detection is crucial for the reliable deployment of machine learning models in-the-wild, enabling accurate identification of test samples that differ from the training data distribution. Existing methods rely on auxiliary outlier samples or in-distribution (ID) data to generate outlier information for training, but due to limited outliers and their mismatch with real test OOD samples, they often fail to provide sufficient semantic supervision, leading to suboptimal performance. To address this, we propose a novel OOD detection method called Retrieval-Augmented Prompt (RAP). RAP augments a pre-trained vision-language model's prompts by retrieving external knowledge, offering enhanced semantic supervision for OOD detection. During training, RAP retrieves descriptive words for outliers based on joint similarity with external textual knowledge and uses them to augment the model's OOD prompts. During testing, RAP dynamically updates OOD prompts in real-time based on the encountered OOD samples, enabling the model to rapidly adapt to the test environment. Our extensive experiments demonstrate that RAP achieves state-of-the-art performance on large-scale OOD detection benchmarks. For example, in 1-shot OOD detection on the ImageNet-1k dataset, RAP reduces the average FPR95 by 7.05% and improves the AUROC by 1.71% compared to previous methods. Additionally, comprehensive ablation studies validate the effectiveness of each module and the underlying motivations of our approach. 

---
# When Language Overrules: Revealing Text Dominance in Multimodal Large Language Models 

**Authors**: Huyu Wu, Meng Tang, Xinhan Zheng, Haiyun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10552)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across a diverse range of multimodal tasks. However, these models suffer from a core problem known as text dominance: they depend heavily on text for their inference, while underutilizing other modalities. While prior work has acknowledged this phenomenon in vision-language tasks, often attributing it to data biases or model architectures. In this paper, we conduct the first systematic investigation of text dominance across diverse data modalities, including images, videos, audio, time-series, and graphs. To measure this imbalance, we propose two evaluation metrics: the Modality Dominance Index (MDI) and the Attention Efficiency Index (AEI). Our comprehensive analysis reveals that text dominance is both significant and pervasive across all tested modalities. Our in-depth analysis identifies three underlying causes: attention dilution from severe token redundancy in non-textual modalities, the influence of fusion architecture design, and task formulations that implicitly favor textual inputs. Furthermore, we propose a simple token compression method that effectively rebalances model attention. Applying this method to LLaVA-7B, for instance, drastically reduces its MDI from 10.23 to a well-balanced value of 0.86. Our analysis and methodological framework offer a foundation for the development of more equitable and comprehensive multimodal language models. 

---
# Stabilizing Long-term Multi-turn Reinforcement Learning with Gated Rewards 

**Authors**: Zetian Sun, Dongfang Li, Zhuoen Chen, Yuhuai Qin, Baotian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10548)  

**Abstract**: Reward sparsity in long-horizon reinforcement learning (RL) tasks remains a significant challenge, while existing outcome-based reward shaping struggles to define meaningful immediate rewards without introducing bias or requiring explicit task decomposition. Alternatively, verification-based reward shaping uses stepwise critics, but misalignment between immediate rewards and long-term objectives can lead to reward hacking and suboptimal policies. In this work, we address this problem in the context of software engineering (SWE) tasks, where multi-turn reasoning and rule-based verification are critical. We introduce the SWE-oriented RL Framework, a unified system supporting multi-turn interaction, docker-based execution, and customizable reward functions. Additionally, we propose Gated Reward Accumulation (G-RA), a novel method that accumulates immediate rewards only when high-level (long-term) rewards meet a predefined threshold, ensuring stable RL optimization. Experiments on SWE-bench Verified and kBench demonstrate that G-RA leads to an increase in completion rates (47.6\% \rightarrow 93.8\% and 22.0\% \rightarrow 86.0\%) and modification rates (19.6\% \rightarrow 23.8\% and 12.0\% \rightarrow 42.0\%), while avoiding policy degradation caused by reward misalignment. Our findings highlight the importance of balanced reward accumulation in long-horizon RL and provide a practical solution. 

---
# Med-GLIP: Advancing Medical Language-Image Pre-training with Large-scale Grounded Dataset 

**Authors**: Ziye Deng, Ruihan He, Jiaxiang Liu, Yuan Wang, Zijie Meng, Songtao Jiang, Yong Xie, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10528)  

**Abstract**: Medical image grounding aims to align natural language phrases with specific regions in medical images, serving as a foundational task for intelligent diagnosis, visual question answering (VQA), and automated report generation (MRG). However, existing research is constrained by limited modality coverage, coarse-grained annotations, and the absence of a unified, generalizable grounding framework. To address these challenges, we construct a large-scale medical grounding dataset Med-GLIP-5M comprising over 5.3 million region-level annotations across seven imaging modalities, covering diverse anatomical structures and pathological findings. The dataset supports both segmentation and grounding tasks with hierarchical region labels, ranging from organ-level boundaries to fine-grained lesions. Based on this foundation, we propose Med-GLIP, a modality-aware grounding framework trained on Med-GLIP-5M. Rather than relying on explicitly designed expert modules, Med-GLIP implicitly acquires hierarchical semantic understanding from diverse training data -- enabling it to recognize multi-granularity structures, such as distinguishing lungs from pneumonia lesions. Extensive experiments demonstrate that Med-GLIP consistently outperforms state-of-the-art baselines across multiple grounding benchmarks. Furthermore, integrating its spatial outputs into downstream tasks, including medical VQA and report generation, leads to substantial performance gains. Our dataset will be released soon. 

---
# Multi-Sample Anti-Aliasing and Constrained Optimization for 3D Gaussian Splatting 

**Authors**: Zheng Zhou, Jia-Chen Zhang, Yu-Jie Xiong, Chun-Ming Xia  

**Link**: [PDF](https://arxiv.org/pdf/2508.10507)  

**Abstract**: Recent advances in 3D Gaussian splatting have significantly improved real-time novel view synthesis, yet insufficient geometric constraints during scene optimization often result in blurred reconstructions of fine-grained details, particularly in regions with high-frequency textures and sharp discontinuities. To address this, we propose a comprehensive optimization framework integrating multisample anti-aliasing (MSAA) with dual geometric constraints. Our system computes pixel colors through adaptive blending of quadruple subsamples, effectively reducing aliasing artifacts in high-frequency components. The framework introduces two constraints: (a) an adaptive weighting strategy that prioritizes under-reconstructed regions through dynamic gradient analysis, and (b) gradient differential constraints enforcing geometric regularization at object boundaries. This targeted optimization enables the model to allocate computational resources preferentially to critical regions requiring refinement while maintaining global consistency. Extensive experimental evaluations across multiple benchmarks demonstrate that our method achieves state-of-the-art performance in detail preservation, particularly in preserving high-frequency textures and sharp discontinuities, while maintaining real-time rendering efficiency. Quantitative metrics and perceptual studies confirm statistically significant improvements over baseline approaches in both structural similarity (SSIM) and perceptual quality (LPIPS). 

---
# Advances in Logic-Based Entity Resolution: Enhancing ASPEN with Local Merges and Optimality Criteria 

**Authors**: Zhliang Xiang, Meghyn Bienvenu, Gianluca Cima, Víctor Gutiérrez-Basulto, Yazmín Ibáñez-García  

**Link**: [PDF](https://arxiv.org/pdf/2508.10504)  

**Abstract**: In this paper, we present ASPEN+, which extends an existing ASP-based system, ASPEN,for collective entity resolution with two important functionalities: support for local merges and new optimality criteria for preferred solutions. Indeed, ASPEN only supports so-called global merges of entity-referring constants (e.g. author ids), in which all occurrences of matched constants are treated as equivalent and merged accordingly. However, it has been argued that when resolving data values, local merges are often more appropriate, as e.g. some instances of 'J. Lee' may refer to 'Joy Lee', while others should be matched with 'Jake Lee'. In addition to allowing such local merges, ASPEN+ offers new optimality criteria for selecting solutions, such as minimizing rule violations or maximising the number of rules supporting a merge. Our main contributions are thus (1) the formalisation and computational analysis of various notions of optimal solution, and (2) an extensive experimental evaluation on real-world datasets, demonstrating the effect of local merges and the new optimality criteria on both accuracy and runtime. 

---
# A Unified Multi-Agent Framework for Universal Multimodal Understanding and Generation 

**Authors**: Jiulin Li, Ping Huang, Yexin Li, Shuo Chen, Juewen Hu, Ye Tian  

**Link**: [PDF](https://arxiv.org/pdf/2508.10494)  

**Abstract**: Real-world multimodal applications often require any-to-any capabilities, enabling both understanding and generation across modalities including text, image, audio, and video. However, integrating the strengths of autoregressive language models (LLMs) for reasoning and diffusion models for high-fidelity generation remains challenging. Existing approaches rely on rigid pipelines or tightly coupled architectures, limiting flexibility and scalability. We propose MAGUS (Multi-Agent Guided Unified Multimodal System), a modular framework that unifies multimodal understanding and generation via two decoupled phases: Cognition and Deliberation. MAGUS enables symbolic multi-agent collaboration within a shared textual workspace. In the Cognition phase, three role-conditioned multimodal LLM agents - Perceiver, Planner, and Reflector - engage in collaborative dialogue to perform structured understanding and planning. The Deliberation phase incorporates a Growth-Aware Search mechanism that orchestrates LLM-based reasoning and diffusion-based generation in a mutually reinforcing manner. MAGUS supports plug-and-play extensibility, scalable any-to-any modality conversion, and semantic alignment - all without the need for joint training. Experiments across multiple benchmarks, including image, video, and audio generation, as well as cross-modal instruction following, demonstrate that MAGUS outperforms strong baselines and state-of-the-art systems. Notably, on the MME benchmark, MAGUS surpasses the powerful closed-source model GPT-4o. 

---
# Contrastive ECOC: Learning Output Codes for Adversarial Defense 

**Authors**: Che-Yu Chou, Hung-Hsuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10491)  

**Abstract**: Although one-hot encoding is commonly used for multiclass classification, it is not always the most effective encoding mechanism. Error Correcting Output Codes (ECOC) address multiclass classification by mapping each class to a unique codeword used as a label. Traditional ECOC methods rely on manually designed or randomly generated codebooks, which are labor-intensive and may yield suboptimal, dataset-agnostic results. This paper introduces three models for automated codebook learning based on contrastive learning, allowing codebooks to be learned directly and adaptively from data. Across four datasets, our proposed models demonstrate superior robustness to adversarial attacks compared to two baselines. The source is available at this https URL. 

---
# On the Complexity-Faithfulness Trade-off of Gradient-Based Explanations 

**Authors**: Amir Mehrpanah, Matteo Gamba, Kevin Smith, Hossein Azizpour  

**Link**: [PDF](https://arxiv.org/pdf/2508.10490)  

**Abstract**: ReLU networks, while prevalent for visual data, have sharp transitions, sometimes relying on individual pixels for predictions, making vanilla gradient-based explanations noisy and difficult to interpret. Existing methods, such as GradCAM, smooth these explanations by producing surrogate models at the cost of faithfulness. We introduce a unifying spectral framework to systematically analyze and quantify smoothness, faithfulness, and their trade-off in explanations. Using this framework, we quantify and regularize the contribution of ReLU networks to high-frequency information, providing a principled approach to identifying this trade-off. Our analysis characterizes how surrogate-based smoothing distorts explanations, leading to an ``explanation gap'' that we formally define and measure for different post-hoc methods. Finally, we validate our theoretical findings across different design choices, datasets, and ablations. 

---
# Pinet: Optimizing hard-constrained neural networks with orthogonal projection layers 

**Authors**: Panagiotis D. Grontas, Antonio Terpin, Efe C. Balta, Raffaello D'Andrea, John Lygeros  

**Link**: [PDF](https://arxiv.org/pdf/2508.10480)  

**Abstract**: We introduce an output layer for neural networks that ensures satisfaction of convex constraints. Our approach, $\Pi$net, leverages operator splitting for rapid and reliable projections in the forward pass, and the implicit function theorem for backpropagation. We deploy $\Pi$net as a feasible-by-design optimization proxy for parametric constrained optimization problems and obtain modest-accuracy solutions faster than traditional solvers when solving a single problem, and significantly faster for a batch of problems. We surpass state-of-the-art learning approaches in terms of training time, solution quality, and robustness to hyperparameter tuning, while maintaining similar inference times. Finally, we tackle multi-vehicle motion planning with non-convex trajectory preferences and provide $\Pi$net as a GPU-ready package implemented in JAX with effective tuning heuristics. 

---
# Enhanced Sparse Point Cloud Data Processing for Privacy-aware Human Action Recognition 

**Authors**: Maimunatu Tunau, Vincent Gbouna Zakka, Zhuangzhuang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2508.10469)  

**Abstract**: Human Action Recognition (HAR) plays a crucial role in healthcare, fitness tracking, and ambient assisted living technologies. While traditional vision based HAR systems are effective, they pose privacy concerns. mmWave radar sensors offer a privacy preserving alternative but present challenges due to the sparse and noisy nature of their point cloud data. In the literature, three primary data processing methods: Density-Based Spatial Clustering of Applications with Noise (DBSCAN), the Hungarian Algorithm, and Kalman Filtering have been widely used to improve the quality and continuity of radar data. However, a comprehensive evaluation of these methods, both individually and in combination, remains lacking. This paper addresses that gap by conducting a detailed performance analysis of the three methods using the MiliPoint dataset. We evaluate each method individually, all possible pairwise combinations, and the combination of all three, assessing both recognition accuracy and computational cost. Furthermore, we propose targeted enhancements to the individual methods aimed at improving accuracy. Our results provide crucial insights into the strengths and trade-offs of each method and their integrations, guiding future work on mmWave based HAR systems 

---
# X-Node: Self-Explanation is All We Need 

**Authors**: Prajit Sengupta, Islem Rekik  

**Link**: [PDF](https://arxiv.org/pdf/2508.10461)  

**Abstract**: Graph neural networks (GNNs) have achieved state-of-the-art results in computer vision and medical image classification tasks by capturing structural dependencies across data instances. However, their decision-making remains largely opaque, limiting their trustworthiness in high-stakes clinical applications where interpretability is essential. Existing explainability techniques for GNNs are typically post-hoc and global, offering limited insight into individual node decisions or local reasoning. We introduce X-Node, a self-explaining GNN framework in which each node generates its own explanation as part of the prediction process. For every node, we construct a structured context vector encoding interpretable cues such as degree, centrality, clustering, feature saliency, and label agreement within its local topology. A lightweight Reasoner module maps this context into a compact explanation vector, which serves three purposes: (1) reconstructing the node's latent embedding via a decoder to enforce faithfulness, (2) generating a natural language explanation using a pre-trained LLM (e.g., Grok or Gemini), and (3) guiding the GNN itself via a "text-injection" mechanism that feeds explanations back into the message-passing pipeline. We evaluate X-Node on two graph datasets derived from MedMNIST and MorphoMNIST, integrating it with GCN, GAT, and GIN backbones. Our results show that X-Node maintains competitive classification accuracy while producing faithful, per-node explanations. Repository: this https URL. 

---
# RealAC: A Domain-Agnostic Framework for Realistic and Actionable Counterfactual Explanations 

**Authors**: Asiful Arefeen, Shovito Barua Soumma, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2508.10455)  

**Abstract**: Counterfactual explanations provide human-understandable reasoning for AI-made decisions by describing minimal changes to input features that would alter a model's prediction. To be truly useful in practice, such explanations must be realistic and feasible -- they should respect both the underlying data distribution and user-defined feasibility constraints. Existing approaches often enforce inter-feature dependencies through rigid, hand-crafted constraints or domain-specific knowledge, which limits their generalizability and ability to capture complex, nonlinear relations inherent in data. Moreover, they rarely accommodate user-specified preferences and suggest explanations that are causally implausible or infeasible to act upon. We introduce RealAC, a domain-agnostic framework for generating realistic and actionable counterfactuals. RealAC automatically preserves complex inter-feature dependencies without relying on explicit domain knowledge -- by aligning the joint distributions of feature pairs between factual and counterfactual instances. The framework also allows end-users to ``freeze'' attributes they cannot or do not wish to change by suppressing change in frozen features during optimization. Evaluations on three synthetic and two real datasets demonstrate that RealAC balances realism with actionability. Our method outperforms state-of-the-art baselines and Large Language Model-based counterfactual generation techniques in causal edge score, dependency preservation score, and IM1 realism metric and offers a solution for causality-aware and user-centric counterfactual generation. 

---
# Alternating Approach-Putt Models for Multi-Stage Speech Enhancement 

**Authors**: Iksoon Jeong, Kyung-Joong Kim, Kang-Hun Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2508.10436)  

**Abstract**: Speech enhancement using artificial neural networks aims to remove noise from noisy speech signals while preserving the speech content. However, speech enhancement networks often introduce distortions to the speech signal, referred to as artifacts, which can degrade audio quality. In this work, we propose a post-processing neural network designed to mitigate artifacts introduced by speech enhancement models. Inspired by the analogy of making a `Putt' after an `Approach' in golf, we name our model PuttNet. We demonstrate that alternating between a speech enhancement model and the proposed Putt model leads to improved speech quality, as measured by perceptual quality scores (PESQ), objective intelligibility (STOI), and background noise intrusiveness (CBAK) scores. Furthermore, we illustrate with graphical analysis why this alternating Approach outperforms repeated application of either model alone. 

---
# Unpacking the Implicit Norm Dynamics of Sharpness-Aware Minimization in Tensorized Models 

**Authors**: Tianxiao Cao, Kyohei Atarashi, Hisashi Kashima  

**Link**: [PDF](https://arxiv.org/pdf/2508.10435)  

**Abstract**: Sharpness-Aware Minimization (SAM) has been proven to be an effective optimization technique for improving generalization in overparameterized models. While prior works have explored the implicit regularization of SAM in simple two-core scale-invariant settings, its behavior in more general tensorized or scale-invariant models remains underexplored. In this work, we leverage scale-invariance to analyze the norm dynamics of SAM in general tensorized models. We introduce the notion of \emph{Norm Deviation} as a global measure of core norm imbalance, and derive its evolution under SAM using gradient flow analysis. We show that SAM's implicit control of Norm Deviation is governed by the covariance between core norms and their gradient magnitudes. Motivated by these findings, we propose a simple yet effective method, \emph{Deviation-Aware Scaling (DAS)}, which explicitly mimics this regularization behavior by scaling core norms in a data-adaptive manner. Our experiments across tensor completion, noisy training, model compression, and parameter-efficient fine-tuning confirm that DAS achieves competitive or improved performance over SAM, while offering reduced computational overhead. 

---
# MASH: Cooperative-Heterogeneous Multi-Agent Reinforcement Learning for Single Humanoid Robot Locomotion 

**Authors**: Qi Liu, Xiaopeng Zhang, Mingshan Tan, Shuaikang Ma, Jinliang Ding, Yanjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10423)  

**Abstract**: This paper proposes a novel method to enhance locomotion for a single humanoid robot through cooperative-heterogeneous multi-agent deep reinforcement learning (MARL). While most existing methods typically employ single-agent reinforcement learning algorithms for a single humanoid robot or MARL algorithms for multi-robot system tasks, we propose a distinct paradigm: applying cooperative-heterogeneous MARL to optimize locomotion for a single humanoid robot. The proposed method, multi-agent reinforcement learning for single humanoid locomotion (MASH), treats each limb (legs and arms) as an independent agent that explores the robot's action space while sharing a global critic for cooperative learning. Experiments demonstrate that MASH accelerates training convergence and improves whole-body cooperation ability, outperforming conventional single-agent reinforcement learning methods. This work advances the integration of MARL into single-humanoid-robot control, offering new insights into efficient locomotion strategies. 

---
# ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning 

**Authors**: Juyuan Wang, Rongchen Zhao, Wei Wei, Yufeng Wang, Mo Yu, Jie Zhou, Jin Xu, Liyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10419)  

**Abstract**: Narrative comprehension on long stories and novels has been a challenging domain attributed to their intricate plotlines and entangled, often evolving relations among characters and entities. Given the LLM's diminished reasoning over extended context and high computational cost, retrieval-based approaches remain a pivotal role in practice. However, traditional RAG methods can fall short due to their stateless, single-step retrieval process, which often overlooks the dynamic nature of capturing interconnected relations within long-range context. In this work, we propose ComoRAG, holding the principle that narrative reasoning is not a one-shot process, but a dynamic, evolving interplay between new evidence acquisition and past knowledge consolidation, analogous to human cognition when reasoning with memory-related signals in the brain. Specifically, when encountering a reasoning impasse, ComoRAG undergoes iterative reasoning cycles while interacting with a dynamic memory workspace. In each cycle, it generates probing queries to devise new exploratory paths, then integrates the retrieved evidence of new aspects into a global memory pool, thereby supporting the emergence of a coherent context for the query resolution. Across four challenging long-context narrative benchmarks (200K+ tokens), ComoRAG outperforms strong RAG baselines with consistent relative gains up to 11% compared to the strongest baseline. Further analysis reveals that ComoRAG is particularly advantageous for complex queries requiring global comprehension, offering a principled, cognitively motivated paradigm for retrieval-based long context comprehension towards stateful reasoning. Our code is publicly released at this https URL 

---
# CorrectNav: Self-Correction Flywheel Empowers Vision-Language-Action Navigation Model 

**Authors**: Zhuoyuan Yu, Yuxing Long, Zihan Yang, Chengyan Zeng, Hongwei Fan, Jiyao Zhang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.10416)  

**Abstract**: Existing vision-and-language navigation models often deviate from the correct trajectory when executing instructions. However, these models lack effective error correction capability, hindering their recovery from errors. To address this challenge, we propose Self-correction Flywheel, a novel post-training paradigm. Instead of considering the model's error trajectories on the training set as a drawback, our paradigm emphasizes their significance as a valuable data source. We have developed a method to identify deviations in these error trajectories and devised innovative techniques to automatically generate self-correction data for perception and action. These self-correction data serve as fuel to power the model's continued training. The brilliance of our paradigm is revealed when we re-evaluate the model on the training set, uncovering new error trajectories. At this time, the self-correction flywheel begins to spin. Through multiple flywheel iterations, we progressively enhance our monocular RGB-based VLA navigation model CorrectNav. Experiments on R2R-CE and RxR-CE benchmarks show CorrectNav achieves new state-of-the-art success rates of 65.1% and 69.3%, surpassing prior best VLA navigation models by 8.2% and 16.4%. Real robot tests in various indoor and outdoor environments demonstrate \method's superior capability of error correction, dynamic obstacle avoidance, and long instruction following. 

---
# MCP2OSC: Parametric Control by Natural Language 

**Authors**: Yuan-Yi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10414)  

**Abstract**: Text prompts enable intuitive content creation but may fall short in achieving high precision for intricate tasks; knob or slider controls offer precise adjustments at the cost of increased complexity. To address the gap between knobs and prompts, a new MCP (Model Context Protocol) server and a unique set of prompt design criteria are presented to enable exploring parametric OSC (OpenSoundControl) control by natural language prompts. Demonstrated by 14 practical QA examples with best practices and the generalized prompt templates, this study finds Claude integrated with the MCP2OSC server effective in generating OSC messages by natural language, interpreting, searching, and visualizing OSC messages, validating and debugging OSC messages, and managing OSC address patterns. MCP2OSC enhances human-machine collaboration by leveraging LLM (Large Language Model) to handle intricate OSC development tasks, and by empowering human creativity with an intuitive language interface featuring flexible precision controls: a prompt-based OSC tool. This study provides a novel perspective on the creative MCP application at the network protocol level by utilizing LLM's strength in directly processing and generating human-readable OSC messages. The results suggest its potential for a LLM-based universal control mechanism for multimedia devices. 

---
# AnalogSeeker: An Open-source Foundation Language Model for Analog Circuit Design 

**Authors**: Zihao Chen, Ji Zhuang, Jinyi Shen, Xiaoyue Ke, Xinyi Yang, Mingjie Zhou, Zhuoyao Du, Xu Yan, Zhouyang Wu, Zhenyu Xu, Jiangli Huang, Li Shang, Xuan Zeng, Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10409)  

**Abstract**: In this paper, we propose AnalogSeeker, an effort toward an open-source foundation language model for analog circuit design, with the aim of integrating domain knowledge and giving design assistance. To overcome the scarcity of data in this field, we employ a corpus collection strategy based on the domain knowledge framework of analog circuits. High-quality, accessible textbooks across relevant subfields are systematically curated and cleaned into a textual domain corpus. To address the complexity of knowledge of analog circuits, we introduce a granular domain knowledge distillation method. Raw, unlabeled domain corpus is decomposed into typical, granular learning nodes, where a multi-agent framework distills implicit knowledge embedded in unstructured text into question-answer data pairs with detailed reasoning processes, yielding a fine-grained, learnable dataset for fine-tuning. To address the unexplored challenges in training analog circuit foundation models, we explore and share our training methods through both theoretical analysis and experimental validation. We finally establish a fine-tuning-centric training paradigm, customizing and implementing a neighborhood self-constrained supervised fine-tuning algorithm. This approach enhances training outcomes by constraining the perturbation magnitude between the model's output distributions before and after training. In practice, we train the Qwen2.5-32B-Instruct model to obtain AnalogSeeker, which achieves 85.04% accuracy on AMSBench-TQA, the analog circuit knowledge evaluation benchmark, with a 15.67% point improvement over the original model and is competitive with mainstream commercial models. Furthermore, AnalogSeeker also shows effectiveness in the downstream operational amplifier design task. AnalogSeeker is open-sourced at this https URL for research use. 

---
# Layer-Wise Perturbations via Sparse Autoencoders for Adversarial Text Generation 

**Authors**: Huizhen Shu, Xuying Li, Qirui Wang, Yuji Kosuga, Mengqiu Tian, Zhuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10404)  

**Abstract**: With the rapid proliferation of Natural Language Processing (NLP), especially Large Language Models (LLMs), generating adversarial examples to jailbreak LLMs remains a key challenge for understanding model vulnerabilities and improving robustness. In this context, we propose a new black-box attack method that leverages the interpretability of large models. We introduce the Sparse Feature Perturbation Framework (SFPF), a novel approach for adversarial text generation that utilizes sparse autoencoders to identify and manipulate critical features in text. After using the SAE model to reconstruct hidden layer representations, we perform feature clustering on the successfully attacked texts to identify features with higher activations. These highly activated features are then perturbed to generate new adversarial texts. This selective perturbation preserves the malicious intent while amplifying safety signals, thereby increasing their potential to evade existing defenses. Our method enables a new red-teaming strategy that balances adversarial effectiveness with safety alignment. Experimental results demonstrate that adversarial texts generated by SFPF can bypass state-of-the-art defense mechanisms, revealing persistent vulnerabilities in current NLP this http URL, the method's effectiveness varies across prompts and layers, and its generalizability to other architectures and larger models remains to be validated. 

---
# PQ-DAF: Pose-driven Quality-controlled Data Augmentation for Data-scarce Driver Distraction Detection 

**Authors**: Haibin Sun, Xinghui Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.10397)  

**Abstract**: Driver distraction detection is essential for improving traffic safety and reducing road accidents. However, existing models often suffer from degraded generalization when deployed in real-world scenarios. This limitation primarily arises from the few-shot learning challenge caused by the high cost of data annotation in practical environments, as well as the substantial domain shift between training datasets and target deployment conditions. To address these issues, we propose a Pose-driven Quality-controlled Data Augmentation Framework (PQ-DAF) that leverages a vision-language model for sample filtering to cost-effectively expand training data and enhance cross-domain robustness. Specifically, we employ a Progressive Conditional Diffusion Model (PCDMs) to accurately capture key driver pose features and synthesize diverse training examples. A sample quality assessment module, built upon the CogVLM vision-language model, is then introduced to filter out low-quality synthetic samples based on a confidence threshold, ensuring the reliability of the augmented dataset. Extensive experiments demonstrate that PQ-DAF substantially improves performance in few-shot driver distraction detection, achieving significant gains in model generalization under data-scarce conditions. 

---
# Unlocking Robust Semantic Segmentation Performance via Label-only Elastic Deformations against Implicit Label Noise 

**Authors**: Yechan Kim, Dongho Yoon, Younkwan Lee, Unse Fatima, Hong Kook Kim, Songjae Lee, Sanga Park, Jeong Ho Park, Seonjong Kang, Moongu Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2508.10383)  

**Abstract**: While previous studies on image segmentation focus on handling severe (or explicit) label noise, real-world datasets also exhibit subtle (or implicit) label imperfections. These arise from inherent challenges, such as ambiguous object boundaries and annotator variability. Although not explicitly present, such mild and latent noise can still impair model performance. Typical data augmentation methods, which apply identical transformations to the image and its label, risk amplifying these subtle imperfections and limiting the model's generalization capacity. In this paper, we introduce NSegment+, a novel augmentation framework that decouples image and label transformations to address such realistic noise for semantic segmentation. By introducing controlled elastic deformations only to segmentation labels while preserving the original images, our method encourages models to focus on learning robust representations of object structures despite minor label inconsistencies. Extensive experiments demonstrate that NSegment+ consistently improves performance, achieving mIoU gains of up to +2.29, +2.38, +1.75, and +3.39 in average on Vaihingen, LoveDA, Cityscapes, and PASCAL VOC, respectively-even without bells and whistles, highlighting the importance of addressing implicit label noise. These gains can be further amplified when combined with other training tricks, including CutMix and Label Smoothing. 

---
# eMamba: Efficient Acceleration Framework for Mamba Models in Edge Computing 

**Authors**: Jiyong Kim, Jaeho Lee, Jiahao Lin, Alish Kanani, Miao Sun, Umit Y. Ogras, Jaehyun Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.10370)  

**Abstract**: State Space Model (SSM)-based machine learning architectures have recently gained significant attention for processing sequential data. Mamba, a recent sequence-to-sequence SSM, offers competitive accuracy with superior computational efficiency compared to state-of-the-art transformer models. While this advantage makes Mamba particularly promising for resource-constrained edge devices, no hardware acceleration frameworks are currently optimized for deploying it in such environments. This paper presents eMamba, a comprehensive end-to-end hardware acceleration framework explicitly designed for deploying Mamba models on edge platforms. eMamba maximizes computational efficiency by replacing complex normalization layers with lightweight hardware-aware alternatives and approximating expensive operations, such as SiLU activation and exponentiation, considering the target applications. Then, it performs an approximation-aware neural architecture search (NAS) to tune the learnable parameters used during approximation. Evaluations with Fashion-MNIST, CIFAR-10, and MARS, an open-source human pose estimation dataset, show eMamba achieves comparable accuracy to state-of-the-art techniques using 1.63-19.9$\times$ fewer parameters. In addition, it generalizes well to large-scale natural language tasks, demonstrating stable perplexity across varying sequence lengths on the WikiText2 dataset. We also quantize and implement the entire eMamba pipeline on an AMD ZCU102 FPGA and ASIC using GlobalFoundries (GF) 22 nm technology. Experimental results show 4.95-5.62$\times$ lower latency and 2.22-9.95$\times$ higher throughput, with 4.77$\times$ smaller area, 9.84$\times$ lower power, and 48.6$\times$ lower energy consumption than baseline solutions while maintaining competitive accuracy. 

---
# Welfare-Centric Clustering 

**Authors**: Claire Jie Zhang, Seyed A. Esmaeili, Jamie Morgenstern  

**Link**: [PDF](https://arxiv.org/pdf/2508.10345)  

**Abstract**: Fair clustering has traditionally focused on ensuring equitable group representation or equalizing group-specific clustering costs. However, Dickerson et al. (2025) recently showed that these fairness notions may yield undesirable or unintuitive clustering outcomes and advocated for a welfare-centric clustering approach that models the utilities of the groups. In this work, we model group utilities based on both distances and proportional representation and formalize two optimization objectives based on welfare-centric clustering: the Rawlsian (Egalitarian) objective and the Utilitarian objective. We introduce novel algorithms for both objectives and prove theoretical guarantees for them. Empirical evaluations on multiple real-world datasets demonstrate that our methods significantly outperform existing fair clustering baselines. 

---
# Layer-Wise Analysis of Self-Supervised Representations for Age and Gender Classification in Children's Speech 

**Authors**: Abhijit Sinha, Harishankar Kumar, Mohit Joshi, Hemant Kumar Kathania, Shrikanth Narayanan, Sudarsana Reddy Kadiri  

**Link**: [PDF](https://arxiv.org/pdf/2508.10332)  

**Abstract**: Children's speech presents challenges for age and gender classification due to high variability in pitch, articulation, and developmental traits. While self-supervised learning (SSL) models perform well on adult speech tasks, their ability to encode speaker traits in children remains underexplored. This paper presents a detailed layer-wise analysis of four Wav2Vec2 variants using the PFSTAR and CMU Kids datasets. Results show that early layers (1-7) capture speaker-specific cues more effectively than deeper layers, which increasingly focus on linguistic information. Applying PCA further improves classification, reducing redundancy and highlighting the most informative components. The Wav2Vec2-large-lv60 model achieves 97.14% (age) and 98.20% (gender) on CMU Kids; base-100h and large-lv60 models reach 86.05% and 95.00% on PFSTAR. These results reveal how speaker traits are structured across SSL model depth and support more targeted, adaptive strategies for child-aware speech interfaces. 

---
# A Vision-Language Pre-training Model-Guided Approach for Mitigating Backdoor Attacks in Federated Learning 

**Authors**: Keke Gai, Dongjue Wang, Jing Yu, Liehuang Zhu, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10315)  

**Abstract**: Existing backdoor defense methods in Federated Learning (FL) rely on the assumption of homogeneous client data distributions or the availability of a clean serve dataset, which limits the practicality and effectiveness. Defending against backdoor attacks under heterogeneous client data distributions while preserving model performance remains a significant challenge. In this paper, we propose a FL backdoor defense framework named CLIP-Fed, which leverages the zero-shot learning capabilities of vision-language pre-training models. By integrating both pre-aggregation and post-aggregation defense strategies, CLIP-Fed overcomes the limitations of Non-IID imposed on defense effectiveness. To address privacy concerns and enhance the coverage of the dataset against diverse triggers, we construct and augment the server dataset using the multimodal large language model and frequency analysis without any client samples. To address class prototype deviations caused by backdoor samples and eliminate the correlation between trigger patterns and target labels, CLIP-Fed aligns the knowledge of the global model and CLIP on the augmented dataset using prototype contrastive loss and Kullback-Leibler divergence. Extensive experiments on representative datasets validate the effectiveness of CLIP-Fed. Compared to state-of-the-art methods, CLIP-Fed achieves an average reduction in ASR, i.e., 2.03\% on CIFAR-10 and 1.35\% on CIFAR-10-LT, while improving average MA by 7.92\% and 0.48\%, respectively. 

---
# ReviewRL: Towards Automated Scientific Review with RL 

**Authors**: Sihang Zeng, Kai Tian, Kaiyan Zhang, Yuru wang, Junqi Gao, Runze Liu, Sa Yang, Jingxuan Li, Xinwei Long, Jiaheng Ma, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.10308)  

**Abstract**: Peer review is essential for scientific progress but faces growing challenges due to increasing submission volumes and reviewer fatigue. Existing automated review approaches struggle with factual accuracy, rating consistency, and analytical depth, often generating superficial or generic feedback lacking the insights characteristic of high-quality human reviews. We introduce ReviewRL, a reinforcement learning framework for generating comprehensive and factually grounded scientific paper reviews. Our approach combines: (1) an ArXiv-MCP retrieval-augmented context generation pipeline that incorporates relevant scientific literature, (2) supervised fine-tuning that establishes foundational reviewing capabilities, and (3) a reinforcement learning procedure with a composite reward function that jointly enhances review quality and rating accuracy. Experiments on ICLR 2025 papers demonstrate that ReviewRL significantly outperforms existing methods across both rule-based metrics and model-based quality assessments. ReviewRL establishes a foundational framework for RL-driven automatic critique generation in scientific discovery, demonstrating promising potential for future development in this domain. The implementation of ReviewRL will be released at GitHub. 

---
# Yet another algorithmic bias: A Discursive Analysis of Large Language Models Reinforcing Dominant Discourses on Gender and Race 

**Authors**: Gustavo Bonil, Simone Hashiguti, Jhessica Silva, João Gondim, Helena Maia, Nádia Silva, Helio Pedrini, Sandra Avila  

**Link**: [PDF](https://arxiv.org/pdf/2508.10304)  

**Abstract**: With the advance of Artificial Intelligence (AI), Large Language Models (LLMs) have gained prominence and been applied in diverse contexts. As they evolve into more sophisticated versions, it is essential to assess whether they reproduce biases, such as discrimination and racialization, while maintaining hegemonic discourses. Current bias detection approaches rely mostly on quantitative, automated methods, which often overlook the nuanced ways in which biases emerge in natural language. This study proposes a qualitative, discursive framework to complement such methods. Through manual analysis of LLM-generated short stories featuring Black and white women, we investigate gender and racial biases. We contend that qualitative methods such as the one proposed here are fundamental to help both developers and users identify the precise ways in which biases manifest in LLM outputs, thus enabling better conditions to mitigate them. Results show that Black women are portrayed as tied to ancestry and resistance, while white women appear in self-discovery processes. These patterns reflect how language models replicate crystalized discursive representations, reinforcing essentialization and a sense of social immobility. When prompted to correct biases, models offered superficial revisions that maintained problematic meanings, revealing limitations in fostering inclusive narratives. Our results demonstrate the ideological functioning of algorithms and have significant implications for the ethical use and development of AI. The study reinforces the need for critical, interdisciplinary approaches to AI design and deployment, addressing how LLM-generated discourses reflect and perpetuate inequalities. 

---
# Pose-Robust Calibration Strategy for Point-of-Gaze Estimation on Mobile Phones 

**Authors**: Yujie Zhao, Jiabei Zeng, Shiguang Shan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10268)  

**Abstract**: Although appearance-based point-of-gaze (PoG) estimation has improved, the estimators still struggle to generalize across individuals due to personal differences. Therefore, person-specific calibration is required for accurate PoG estimation. However, calibrated PoG estimators are often sensitive to head pose variations. To address this, we investigate the key factors influencing calibrated estimators and explore pose-robust calibration strategies. Specifically, we first construct a benchmark, MobilePoG, which includes facial images from 32 individuals focusing on designated points under either fixed or continuously changing head poses. Using this benchmark, we systematically analyze how the diversity of calibration points and head poses influences estimation accuracy. Our experiments show that introducing a wider range of head poses during calibration improves the estimator's ability to handle pose variation. Building on this insight, we propose a dynamic calibration strategy in which users fixate on calibration points while moving their phones. This strategy naturally introduces head pose variation during a user-friendly and efficient calibration process, ultimately producing a better calibrated PoG estimator that is less sensitive to head pose variations than those using conventional calibration strategies. Codes and datasets are available at our project page. 

---
# MRFD: Multi-Region Fusion Decoding with Self-Consistency for Mitigating Hallucinations in LVLMs 

**Authors**: Haonan Ge, Yiwei Wang, Ming-Hsuan Yang, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2508.10264)  

**Abstract**: Large Vision-Language Models (LVLMs) have shown strong performance across multimodal tasks. However, they often produce hallucinations -- text that is inconsistent with visual input, due to the limited ability to verify information in different regions of the image. To address this, we propose Multi-Region Fusion Decoding (MRFD), a training-free decoding method that improves factual grounding by modeling inter-region consistency. MRFD identifies salient regions using cross-attention, generates initial responses for each, and computes reliability weights based on Jensen-Shannon Divergence (JSD) among the responses. These weights guide a consistency-aware fusion of per-region predictions, using region-aware prompts inspired by Chain-of-Thought reasoning. Experiments across multiple LVLMs and benchmarks show that MRFD significantly reduces hallucinations and improves response factuality without requiring model updates. 

---
# DINOMotion: advanced robust tissue motion tracking with DINOv2 in 2D-Cine MRI-guided radiotherapy 

**Authors**: Soorena Salari, Catherine Spino, Laurie-Anne Pharand, Fabienne Lathuiliere, Hassan Rivaz, Silvain Beriault, Yiming Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.10260)  

**Abstract**: Accurate tissue motion tracking is critical to ensure treatment outcome and safety in 2D-Cine MRI-guided radiotherapy. This is typically achieved by registration of sequential images, but existing methods often face challenges with large misalignments and lack of interpretability. In this paper, we introduce DINOMotion, a novel deep learning framework based on DINOv2 with Low-Rank Adaptation (LoRA) layers for robust, efficient, and interpretable motion tracking. DINOMotion automatically detects corresponding landmarks to derive optimal image registration, enhancing interpretability by providing explicit visual correspondences between sequential images. The integration of LoRA layers reduces trainable parameters, improving training efficiency, while DINOv2's powerful feature representations offer robustness against large misalignments. Unlike iterative optimization-based methods, DINOMotion directly computes image registration at test time. Our experiments on volunteer and patient datasets demonstrate its effectiveness in estimating both linear and nonlinear transformations, achieving Dice scores of 92.07% for the kidney, 90.90% for the liver, and 95.23% for the lung, with corresponding Hausdorff distances of 5.47 mm, 8.31 mm, and 6.72 mm, respectively. DINOMotion processes each scan in approximately 30ms and consistently outperforms state-of-the-art methods, particularly in handling large misalignments. These results highlight its potential as a robust and interpretable solution for real-time motion tracking in 2D-Cine MRI-guided radiotherapy. 

---
# Facilitating Longitudinal Interaction Studies of AI Systems 

**Authors**: Tao Long, Sitong Wang, Émilie Fabre, Tony Wang, Anup Sathya, Jason Wu, Savvas Petridis, Dingzeyu Li, Tuhin Chakrabarty, Yue Jiang, Jingyi Li, Tiffany Tseng, Ken Nakagaki, Qian Yang, Nikolas Martelaro, Jeffrey V. Nickerson, Lydia B. Chilton  

**Link**: [PDF](https://arxiv.org/pdf/2508.10252)  

**Abstract**: UIST researchers develop tools to address user challenges. However, user interactions with AI evolve over time through learning, adaptation, and repurposing, making one time evaluations insufficient. Capturing these dynamics requires longer-term studies, but challenges in deployment, evaluation design, and data collection have made such longitudinal research difficult to implement. Our workshop aims to tackle these challenges and prepare researchers with practical strategies for longitudinal studies. The workshop includes a keynote, panel discussions, and interactive breakout groups for discussion and hands-on protocol design and tool prototyping sessions. We seek to foster a community around longitudinal system research and promote it as a more embraced method for designing, building, and evaluating UIST tools. 

---
# No Free Lunch from Audio Pretraining in Bioacoustics: A Benchmark Study of Embeddings 

**Authors**: Chenggang Chen, Zhiyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10230)  

**Abstract**: Bioacoustics, the study of animal sounds, offers a non-invasive method to monitor ecosystems. Extracting embeddings from audio-pretrained deep learning (DL) models without fine-tuning has become popular for obtaining bioacoustic features for tasks. However, a recent benchmark study reveals that while fine-tuned audio-pretrained VGG and transformer models achieve state-of-the-art performance in some tasks, they fail in others. This study benchmarks 11 DL models on the same tasks by reducing their learned embeddings' dimensionality and evaluating them through clustering. We found that audio-pretrained DL models 1) without fine-tuning even underperform fine-tuned AlexNet, 2) both with and without fine-tuning fail to separate the background from labeled sounds, but ResNet does, and 3) outperform other models when fewer background sounds are included during fine-tuning. This study underscores the necessity of fine-tuning audio-pretrained models and checking the embeddings after fine-tuning. Our codes are available: this https URL\_Embeddings 

---
# Using Large Language Models to Measure Symptom Severity in Patients At Risk for Schizophrenia 

**Authors**: Andrew X. Chen, Guillermo Horga, Sean Escola  

**Link**: [PDF](https://arxiv.org/pdf/2508.10226)  

**Abstract**: Patients who are at clinical high risk (CHR) for schizophrenia need close monitoring of their symptoms to inform appropriate treatments. The Brief Psychiatric Rating Scale (BPRS) is a validated, commonly used research tool for measuring symptoms in patients with schizophrenia and other psychotic disorders; however, it is not commonly used in clinical practice as it requires a lengthy structured interview. Here, we utilize large language models (LLMs) to predict BPRS scores from clinical interview transcripts in 409 CHR patients from the Accelerating Medicines Partnership Schizophrenia (AMP-SCZ) cohort. Despite the interviews not being specifically structured to measure the BPRS, the zero-shot performance of the LLM predictions compared to the true assessment (median concordance: 0.84, ICC: 0.73) approaches human inter- and intra-rater reliability. We further demonstrate that LLMs have substantial potential to improve and standardize the assessment of CHR patients via their accuracy in assessing the BPRS in foreign languages (median concordance: 0.88, ICC: 0.70), and integrating longitudinal information in a one-shot or few-shot learning approach. 

---
# Understanding Textual Emotion Through Emoji Prediction 

**Authors**: Ethan Gordon, Nishank Kuppa, Rigved Tummala, Sriram Anasuri  

**Link**: [PDF](https://arxiv.org/pdf/2508.10222)  

**Abstract**: This project explores emoji prediction from short text sequences using four deep learning architectures: a feed-forward network, CNN, transformer, and BERT. Using the TweetEval dataset, we address class imbalance through focal loss and regularization techniques. Results show BERT achieves the highest overall performance due to its pre-training advantage, while CNN demonstrates superior efficacy on rare emoji classes. This research shows the importance of architecture selection and hyperparameter tuning for sentiment-aware emoji prediction, contributing to improved human-computer interaction. 

---
# An Explainable AI based approach for Monitoring Animal Health 

**Authors**: Rahul Janaa, Shubham Dixit, Mrityunjay Sharma, Ritesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.10210)  

**Abstract**: Monitoring cattle health and optimizing yield are key challenges faced by dairy farmers due to difficulties in tracking all animals on the farm. This work aims to showcase modern data-driven farming practices based on explainable machine learning(ML) methods that explain the activity and behaviour of dairy cattle (cows). Continuous data collection of 3-axis accelerometer sensors and usage of robust ML methodologies and algorithms, provide farmers and researchers with actionable information on cattle activity, allowing farmers to make informed decisions and incorporate sustainable practices. This study utilizes Bluetooth-based Internet of Things (IoT) devices and 4G networks for seamless data transmission, immediate analysis, inference generation, and explains the models performance with explainability frameworks. Special emphasis is put on the pre-processing of the accelerometers time series data, including the extraction of statistical characteristics, signal processing techniques, and lag-based features using the sliding window technique. Various hyperparameter-optimized ML models are evaluated across varying window lengths for activity classification. The k-nearest neighbour Classifier achieved the best performance, with AUC of mean 0.98 and standard deviation of 0.0026 on the training set and 0.99 on testing set). In order to ensure transparency, Explainable AI based frameworks such as SHAP is used to interpret feature importance that can be understood and used by practitioners. A detailed comparison of the important features, along with the stability analysis of selected features, supports development of explainable and practical ML models for sustainable livestock management. 

---
# CATNet: A geometric deep learning approach for CAT bond spread prediction in the primary market 

**Authors**: Dixon Domfeh, Saeid Safarveisi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10208)  

**Abstract**: Traditional models for pricing catastrophe (CAT) bonds struggle to capture the complex, relational data inherent in these instruments. This paper introduces CATNet, a novel framework that applies a geometric deep learning architecture, the Relational Graph Convolutional Network (R-GCN), to model the CAT bond primary market as a graph, leveraging its underlying network structure for spread prediction. Our analysis reveals that the CAT bond market exhibits the characteristics of a scale-free network, a structure dominated by a few highly connected and influential hubs. CATNet demonstrates high predictive performance, significantly outperforming a strong Random Forest benchmark. The inclusion of topological centrality measures as features provides a further, significant boost in accuracy. Interpretability analysis confirms that these network features are not mere statistical artifacts; they are quantitative proxies for long-held industry intuition regarding issuer reputation, underwriter influence, and peril concentration. This research provides evidence that network connectivity is a key determinant of price, offering a new paradigm for risk assessment and proving that graph-based models can deliver both state-of-the-art accuracy and deeper, quantifiable market insights. 

---
# Prompt-Response Semantic Divergence Metrics for Faithfulness Hallucination and Misalignment Detection in Large Language Models 

**Authors**: Igor Halperin  

**Link**: [PDF](https://arxiv.org/pdf/2508.10192)  

**Abstract**: The proliferation of Large Language Models (LLMs) is challenged by hallucinations, critical failure modes where models generate non-factual, nonsensical or unfaithful text. This paper introduces Semantic Divergence Metrics (SDM), a novel lightweight framework for detecting Faithfulness Hallucinations -- events of severe deviations of LLMs responses from input contexts. We focus on a specific implementation of these LLM errors, {confabulations, defined as responses that are arbitrary and semantically misaligned with the user's query. Existing methods like Semantic Entropy test for arbitrariness by measuring the diversity of answers to a single, fixed prompt. Our SDM framework improves upon this by being more prompt-aware: we test for a deeper form of arbitrariness by measuring response consistency not only across multiple answers but also across multiple, semantically-equivalent paraphrases of the original prompt. Methodologically, our approach uses joint clustering on sentence embeddings to create a shared topic space for prompts and answers. A heatmap of topic co-occurances between prompts and responses can be viewed as a quantified two-dimensional visualization of the user-machine dialogue. We then compute a suite of information-theoretic metrics to measure the semantic divergence between prompts and responses. Our practical score, $\mathcal{S}_H$, combines the Jensen-Shannon divergence and Wasserstein distance to quantify this divergence, with a high score indicating a Faithfulness hallucination. Furthermore, we identify the KL divergence KL(Answer $||$ Prompt) as a powerful indicator of \textbf{Semantic Exploration}, a key signal for distinguishing different generative behaviors. These metrics are further combined into the Semantic Box, a diagnostic framework for classifying LLM response types, including the dangerous, confident confabulation. 

---
# PakBBQ: A Culturally Adapted Bias Benchmark for QA 

**Authors**: Abdullah Hashmat, Muhammad Arham Mirza, Agha Ali Raza  

**Link**: [PDF](https://arxiv.org/pdf/2508.10186)  

**Abstract**: With the widespread adoption of Large Language Models (LLMs) across various applications, it is empirical to ensure their fairness across all user communities. However, most LLMs are trained and evaluated on Western centric data, with little attention paid to low-resource languages and regional contexts. To address this gap, we introduce PakBBQ, a culturally and regionally adapted extension of the original Bias Benchmark for Question Answering (BBQ) dataset. PakBBQ comprises over 214 templates, 17180 QA pairs across 8 categories in both English and Urdu, covering eight bias dimensions including age, disability, appearance, gender, socio-economic status, religious, regional affiliation, and language formality that are relevant in Pakistan. We evaluate multiple multilingual LLMs under both ambiguous and explicitly disambiguated contexts, as well as negative versus non negative question framings. Our experiments reveal (i) an average accuracy gain of 12\% with disambiguation, (ii) consistently stronger counter bias behaviors in Urdu than in English, and (iii) marked framing effects that reduce stereotypical responses when questions are posed negatively. These findings highlight the importance of contextualized benchmarks and simple prompt engineering strategies for bias mitigation in low resource settings. 

---
# LaajMeter: A Framework for LaaJ Evaluation 

**Authors**: Gal Amram, Eitan Farchi, Shmulik Froimovich, Raviv Gal, Avi Ziv  

**Link**: [PDF](https://arxiv.org/pdf/2508.10161)  

**Abstract**: Large Language Models (LLMs) are increasingly used as evaluators in natural language processing tasks, a paradigm known as LLM-as-a-Judge (LaaJ). While effective in general domains, LaaJs pose significant challenges in domain-specific contexts, where annotated data is scarce and expert evaluation is costly. In such cases, meta-evaluation is often performed using metrics that have not been validated for the specific domain in which they are applied. As a result, it becomes difficult to determine which metrics effectively identify LaaJ quality, and further, what threshold indicates sufficient evaluator performance. In this work, we introduce LaaJMeter, a simulation-based framework for controlled meta-evaluation of LaaJs. LaaJMeter enables engineers to generate synthetic data representing virtual models and judges, allowing systematic analysis of evaluation metrics under realistic conditions. This helps practitioners validate and refine LaaJs for specific evaluation tasks: they can test whether their metrics correctly distinguish between better and worse (virtual) LaaJs, and estimate appropriate thresholds for evaluator adequacy.
We demonstrate the utility of LaaJMeter in a code translation task involving a legacy programming language, showing how different metrics vary in sensitivity to evaluator quality. Our results highlight the limitations of common metrics and the importance of principled metric selection. LaaJMeter provides a scalable and extensible solution for assessing LaaJs in low-resource settings, contributing to the broader effort to ensure trustworthy and reproducible evaluation in NLP. 

---
# Improving watermelon (Citrullus lanatus) disease classification with generative artificial intelligence (GenAI)-based synthetic and real-field images via a custom EfficientNetV2-L model 

**Authors**: Nitin Rai, Nathan S. Boyd, Gary E. Vallad, Arnold W. Schumann  

**Link**: [PDF](https://arxiv.org/pdf/2508.10156)  

**Abstract**: The current advancements in generative artificial intelligence (GenAI) models have paved the way for new possibilities for generating high-resolution synthetic images, thereby offering a promising alternative to traditional image acquisition for training computer vision models in agriculture. In the context of crop disease diagnosis, GenAI models are being used to create synthetic images of various diseases, potentially facilitating model creation and reducing the dependency on resource-intensive in-field data collection. However, limited research has been conducted on evaluating the effectiveness of integrating real with synthetic images to improve disease classification performance. Therefore, this study aims to investigate whether combining a limited number of real images with synthetic images can enhance the prediction accuracy of an EfficientNetV2-L model for classifying watermelon \textit{(Citrullus lanatus)} diseases. The training dataset was divided into five treatments: H0 (only real images), H1 (only synthetic images), H2 (1:1 real-to-synthetic), H3 (1:10 real-to-synthetic), and H4 (H3 + random images to improve variability and model generalization). All treatments were trained using a custom EfficientNetV2-L architecture with enhanced fine-tuning and transfer learning techniques. Models trained on H2, H3, and H4 treatments demonstrated high precision, recall, and F1-score metrics. Additionally, the weighted F1-score increased from 0.65 (on H0) to 1.00 (on H3-H4) signifying that the addition of a small number of real images with a considerable volume of synthetic images improved model performance and generalizability. Overall, this validates the findings that synthetic images alone cannot adequately substitute for real images; instead, both must be used in a hybrid manner to maximize model performance for crop disease classification. 

---
# Out-of-Distribution Detection using Counterfactual Distance 

**Authors**: Maria Stoica, Francesco Leofante, Alessio Lomuscio  

**Link**: [PDF](https://arxiv.org/pdf/2508.10148)  

**Abstract**: Accurate and explainable out-of-distribution (OOD) detection is required to use machine learning systems safely. Previous work has shown that feature distance to decision boundaries can be used to identify OOD data effectively. In this paper, we build on this intuition and propose a post-hoc OOD detection method that, given an input, calculates the distance to decision boundaries by leveraging counterfactual explanations. Since computing explanations can be expensive for large architectures, we also propose strategies to improve scalability by computing counterfactuals directly in embedding space. Crucially, as the method employs counterfactual explanations, we can seamlessly use them to help interpret the results of our detector. We show that our method is in line with the state of the art on CIFAR-10, achieving 93.50% AUROC and 25.80% FPR95. Our method outperforms these methods on CIFAR-100 with 97.05% AUROC and 13.79% FPR95 and on ImageNet-200 with 92.55% AUROC and 33.55% FPR95 across four OOD datasets 

---
# rETF-semiSL: Semi-Supervised Learning for Neural Collapse in Temporal Data 

**Authors**: Yuhan Xie, William Cappelletti, Mahsa Shoaran, Pascal Frossard  

**Link**: [PDF](https://arxiv.org/pdf/2508.10147)  

**Abstract**: Deep neural networks for time series must capture complex temporal patterns, to effectively represent dynamic data. Self- and semi-supervised learning methods show promising results in pre-training large models, which -- when finetuned for classification -- often outperform their counterparts trained from scratch. Still, the choice of pretext training tasks is often heuristic and their transferability to downstream classification is not granted, thus we propose a novel semi-supervised pre-training strategy to enforce latent representations that satisfy the Neural Collapse phenomenon observed in optimally trained neural classifiers. We use a rotational equiangular tight frame-classifier and pseudo-labeling to pre-train deep encoders with few labeled samples. Furthermore, to effectively capture temporal dynamics while enforcing embedding separability, we integrate generative pretext tasks with our method, and we define a novel sequential augmentation strategy. We show that our method significantly outperforms previous pretext tasks when applied to LSTMs, transformers, and state-space models on three multivariate time series classification datasets. These results highlight the benefit of aligning pre-training objectives with theoretically grounded embedding geometry. 

---
# mSCoRe: a $M$ultilingual and Scalable Benchmark for $S$kill-based $Co$mmonsense $Re$asoning 

**Authors**: Nghia Trung Ngo, Franck Dernoncourt, Thien Huu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10137)  

**Abstract**: Recent advancements in reasoning-reinforced Large Language Models (LLMs) have shown remarkable capabilities in complex reasoning tasks. However, the mechanism underlying their utilization of different human reasoning skills remains poorly investigated, especially for multilingual commonsense reasoning that involves everyday knowledge across different languages and cultures. To address this gap, we propose a \textbf{M}ultilingual and Scalable Benchmark for \textbf{S}kill-based \textbf{Co}mmonsense \textbf{Re}asoning (\textbf{mSCoRe}). Our benchmark incorporates three key components that are designed to systematically evaluate LLM's reasoning capabilities, including: (1) a novel taxonomy of reasoning skills that enables fine-grained analysis of models' reasoning processes, (2) a robust data synthesis pipeline tailored specifically for commonsense reasoning evaluation, and (3) a complexity scaling framework allowing task difficulty to scale dynamically alongside future improvements in LLM abilities. Extensive experiments on eights state-of-the-art LLMs of varying sizes and training approaches demonstrate that \textbf{mSCoRe} remains significantly challenging for current models, particularly at higher complexity levels. Our results reveal the limitations of such reasoning-reinforced models when confronted with nuanced multilingual general and cultural commonsense. We further provide detailed analysis on the models' reasoning processes, suggesting future directions for improving multilingual commonsense reasoning capabilities. 

---
# Nested-ReFT: Efficient Reinforcement Learning for Large Language Model Fine-Tuning via Off-Policy Rollouts 

**Authors**: Maxime Heuillet, Yufei Cui, Boxing Chen, Audrey Durand, Prasanna Parthasarathi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10123)  

**Abstract**: Advanced reasoning in LLMs on challenging domains like mathematical reasoning can be tackled using verifiable rewards based reinforced fine-tuning (ReFT). In standard ReFT frameworks, a behavior model generates multiple completions with answers per problem, for the answer to be then scored by a reward function. While such RL post-training methods demonstrate significant performance improvements across challenging reasoning domains, the computational cost of generating completions during training with multiple inference steps makes the training cost non-trivial. To address this, we draw inspiration from off-policy RL, and speculative decoding to introduce a novel ReFT framework, dubbed Nested-ReFT, where a subset of layers of the target model acts as the behavior model to generate off-policy completions during training. The behavior model configured with dynamic layer skipping per batch during training decreases the inference cost compared to the standard ReFT frameworks. Our theoretical analysis shows that Nested-ReFT yields unbiased gradient estimates with controlled variance. Our empirical analysis demonstrates improved computational efficiency measured as tokens/sec across multiple math reasoning benchmarks and model sizes. Additionally, we explore three variants of bias mitigation to minimize the off-policyness in the gradient updates that allows for maintaining performance that matches the baseline ReFT performance. 

---
# Less is More: Learning Graph Tasks with Just LLMs 

**Authors**: Sola Shirai, Kavitha Srinivas, Julian Dolby, Michael Katz, Horst Samulowitz, Shirin Sohrabi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10115)  

**Abstract**: For large language models (LLMs), reasoning over graphs could help solve many problems. Prior work has tried to improve LLM graph reasoning by examining how best to serialize graphs as text and by combining GNNs and LLMs. However, the merits of such approaches remain unclear, so we empirically answer the following research questions: (1) Can LLMs learn to solve fundamental graph tasks without specialized graph encoding models?, (2) Can LLMs generalize learned solutions to unseen graph structures or tasks?, and (3) What are the merits of competing approaches to learn graph tasks? We show that even small LLMs can learn to solve graph tasks by training them with instructive chain-of-thought solutions, and this training generalizes, without specialized graph encoders, to new tasks and graph structures. 

---
# Empowering Morphing Attack Detection using Interpretable Image-Text Foundation Model 

**Authors**: Sushrut Patwardhan, Raghavendra Ramachandra, Sushma Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2508.10110)  

**Abstract**: Morphing attack detection has become an essential component of face recognition systems for ensuring a reliable verification scenario. In this paper, we present a multimodal learning approach that can provide a textual description of morphing attack detection. We first show that zero-shot evaluation of the proposed framework using Contrastive Language-Image Pretraining (CLIP) can yield not only generalizable morphing attack detection, but also predict the most relevant text snippet. We present an extensive analysis of ten different textual prompts that include both short and long textual prompts. These prompts are engineered by considering the human understandable textual snippet. Extensive experiments were performed on a face morphing dataset that was developed using a publicly available face biometric dataset. We present an evaluation of SOTA pre-trained neural networks together with the proposed framework in the zero-shot evaluation of five different morphing generation techniques that are captured in three different mediums. 

---
# Advancing Data Equity: Practitioner Responsibility and Accountability in NLP Data Practices 

**Authors**: Jay L. Cunningham, Kevin Zhongyang Shao, Rock Yuren Pang, Nathaniel Mengist  

**Link**: [PDF](https://arxiv.org/pdf/2508.10071)  

**Abstract**: While research has focused on surfacing and auditing algorithmic bias to ensure equitable AI development, less is known about how NLP practitioners - those directly involved in dataset development, annotation, and deployment - perceive and navigate issues of NLP data equity. This study is among the first to center practitioners' perspectives, linking their experiences to a multi-scalar AI governance framework and advancing participatory recommendations that bridge technical, policy, and community domains. Drawing on a 2024 questionnaire and focus group, we examine how U.S.-based NLP data practitioners conceptualize fairness, contend with organizational and systemic constraints, and engage emerging governance efforts such as the U.S. AI Bill of Rights. Findings reveal persistent tensions between commercial objectives and equity commitments, alongside calls for more participatory and accountable data workflows. We critically engage debates on data diversity and diversity washing, arguing that improving NLP equity requires structural governance reforms that support practitioner agency and community consent. 

---
# Large Language Models Show Signs of Alignment with Human Neurocognition During Abstract Reasoning 

**Authors**: Christopher Pinier, Sonia Acuña Vargas, Mariia Steeghs-Turchina, Dora Matzke, Claire E. Stevenson, Michael D. Nunez  

**Link**: [PDF](https://arxiv.org/pdf/2508.10057)  

**Abstract**: This study investigates whether large language models (LLMs) mirror human neurocognition during abstract reasoning. We compared the performance and neural representations of human participants with those of eight open-source LLMs on an abstract-pattern-completion task. We leveraged pattern type differences in task performance and in fixation-related potentials (FRPs) as recorded by electroencephalography (EEG) during the task. Our findings indicate that only the largest tested LLMs (~70 billion parameters) achieve human-comparable accuracy, with Qwen-2.5-72B and DeepSeek-R1-70B also showing similarities with the human pattern-specific difficulty profile. Critically, every LLM tested forms representations that distinctly cluster the abstract pattern categories within their intermediate layers, although the strength of this clustering scales with their performance on the task. Moderate positive correlations were observed between the representational geometries of task-optimal LLM layers and human frontal FRPs. These results consistently diverged from comparisons with other EEG measures (response-locked ERPs and resting EEG), suggesting a potential shared representational space for abstract patterns. This indicates that LLMs might mirror human brain mechanisms in abstract reasoning, offering preliminary evidence of shared principles between biological and artificial intelligence. 

---
# NetMoniAI: An Agentic AI Framework for Network Security & Monitoring 

**Authors**: Pallavi Zambare, Venkata Nikhil Thanikella, Nikhil Padmanabh Kottur, Sree Akhil Akula, Ying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10052)  

**Abstract**: In this paper, we present NetMoniAI, an agentic AI framework for automatic network monitoring and security that integrates decentralized analysis with lightweight centralized coordination. The framework consists of two layers: autonomous micro-agents at each node perform local traffic analysis and anomaly detection. A central controller then aggregates insights across nodes to detect coordinated attacks and maintain system-wide situational awareness. We evaluated NetMoniAI on a local micro-testbed and through NS-3 simulations. Results confirm that the two-tier agentic-AI design scales under resource constraints, reduces redundancy, and improves response time without compromising accuracy. To facilitate broader adoption and reproducibility, the complete framework is available as open source. This enables researchers and practitioners to replicate, validate, and extend it across diverse network environments and threat scenarios. Github link: this https URL 

---
# Legal Zero-Days: A Novel Risk Vector for Advanced AI Systems 

**Authors**: Greg Sadler, Nathan Sherburn  

**Link**: [PDF](https://arxiv.org/pdf/2508.10050)  

**Abstract**: We introduce the concept of "Legal Zero-Days" as a novel risk vector for advanced AI systems. Legal Zero-Days are previously undiscovered vulnerabilities in legal frameworks that, when exploited, can cause immediate and significant societal disruption without requiring litigation or other processes before impact. We present a risk model for identifying and evaluating these vulnerabilities, demonstrating their potential to bypass safeguards or impede government responses to AI incidents. Using the 2017 Australian dual citizenship crisis as a case study, we illustrate how seemingly minor legal oversights can lead to large-scale governance disruption. We develop a methodology for creating "legal puzzles" as evaluation instruments for assessing AI systems' capabilities to discover such vulnerabilities. Our findings suggest that while current AI models may not reliably find impactful Legal Zero-Days, future systems may develop this capability, presenting both risks and opportunities for improving legal robustness. This work contributes to the broader effort to identify and mitigate previously unrecognized risks from frontier AI systems. 

---
# SABIA: An AI-Powered Tool for Detecting Opioid-Related Behaviors on Social Media 

**Authors**: Muhammad Ahmad, Fida Ullah, Muhammad Usman, Ildar Batyrshin, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2508.10046)  

**Abstract**: Social media platforms have become valuable tools for understanding public health challenges by offering insights into patient behaviors, medication use, and mental health issues. However, analyzing such data remains difficult due to the prevalence of informal language, slang, and coded communication, which can obscure the detection of opioid misuse. This study addresses the issue of opioid-related user behavior on social media, including informal expressions, slang terms, and misspelled or coded language. We analyzed the existing Bidirectional Encoder Representations from Transformers (BERT) technique and developed a BERT-BiLSTM-3CNN hybrid deep learning model, named SABIA, to create a single-task classifier that effectively captures the features of the target dataset. The SABIA model demonstrated strong capabilities in capturing semantics and contextual information. The proposed approach includes: (1) data preprocessing, (2) data representation using the SABIA model, (3) a fine-tuning phase, and (4) classification of user behavior into five categories. A new dataset was constructed from Reddit posts, identifying opioid user behaviors across five classes: Dealers, Active Opioid Users, Recovered Users, Prescription Users, and Non-Users, supported by detailed annotation guidelines. Experiments were conducted using supervised learning. Results show that SABIA achieved benchmark performance, outperforming the baseline (Logistic Regression, LR = 0.86) and improving accuracy by 9.30%. Comparisons with seven previous studies confirmed its effectiveness and robustness. This study demonstrates the potential of hybrid deep learning models for detecting complex opioid-related behaviors on social media, supporting public health monitoring and intervention efforts. 

---
# Generative AI for Cybersecurity of Energy Management Systems: Methods, Challenges, and Future Directions 

**Authors**: Aydin Zaboli, Junho Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.10044)  

**Abstract**: This paper elaborates on an extensive security framework specifically designed for energy management systems (EMSs), which effectively tackles the dynamic environment of cybersecurity vulnerabilities and/or system problems (SPs), accomplished through the incorporation of novel methodologies. A comprehensive multi-point attack/error model is initially proposed to systematically identify vulnerabilities throughout the entire EMS data processing pipeline, including post state estimation (SE) stealth attacks, EMS database manipulation, and human-machine interface (HMI) display corruption according to the real-time database (RTDB) storage. This framework acknowledges the interconnected nature of modern attack vectors, which utilize various phases of supervisory control and data acquisition (SCADA) data flow. Then, generative AI (GenAI)-based anomaly detection systems (ADSs) for EMSs are proposed for the first time in the power system domain to handle the scenarios. Further, a set-of-mark generative intelligence (SoM-GI) framework, which leverages multimodal analysis by integrating visual markers with rules considering the GenAI capabilities, is suggested to overcome inherent spatial reasoning limitations. The SoM-GI methodology employs systematic visual indicators to enable accurate interpretation of segmented HMI displays and detect visual anomalies that numerical methods fail to identify. Validation on the IEEE 14-Bus system shows the framework's effectiveness across scenarios, while visual analysis identifies inconsistencies. This integrated approach combines numerical analysis with visual pattern recognition and linguistic rules to protect against cyber threats and system errors. 

---
# Securing Agentic AI: Threat Modeling and Risk Analysis for Network Monitoring Agentic AI System 

**Authors**: Pallavi Zambare, Venkata Nikhil Thanikella, Ying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10043)  

**Abstract**: When combining Large Language Models (LLMs) with autonomous agents, used in network monitoring and decision-making systems, this will create serious security issues. In this research, the MAESTRO framework consisting of the seven layers threat modeling architecture in the system was used to expose, evaluate, and eliminate vulnerabilities of agentic AI. The prototype agent system was constructed and implemented, using Python, LangChain, and telemetry in WebSockets, and deployed with inference, memory, parameter tuning, and anomaly detection modules. Two practical threat cases were confirmed as follows: (i) resource denial of service by traffic replay denial-of-service, and (ii) memory poisoning by tampering with the historical log file maintained by the agent. These situations resulted in measurable levels of performance degradation, i.e. telemetry updates were delayed, and computational loads were increased, as a result of poor system adaptations. It was suggested to use a multilayered defense-in-depth approach with memory isolation, validation of planners and anomaly response systems in real-time. These findings verify that MAESTRO is viable in operational threat mapping, prospective risk scoring, and the basis of the resilient system design. The authors bring attention to the importance of the enforcement of memory integrity, paying attention to the adaptation logic monitoring, and cross-layer communication protection that guarantee the agentic AI reliability in adversarial settings. 

---
# FIDELIS: Blockchain-Enabled Protection Against Poisoning Attacks in Federated Learning 

**Authors**: Jane Carney, Kushal Upreti, Gaby G. Dagher, Tim Andersen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10042)  

**Abstract**: Federated learning enhances traditional deep learning by enabling the joint training of a model with the use of IoT device's private data. It ensures privacy for clients, but is susceptible to data poisoning attacks during training that degrade model performance and integrity. Current poisoning detection methods in federated learning lack a standardized detection method or take significant liberties with trust. In this paper, we present \Sys, a novel blockchain-enabled poison detection framework in federated learning. The framework decentralizes the role of the global server across participating clients. We introduce a judge model used to detect data poisoning in model updates. The judge model is produced by each client and verified to reach consensus on a single judge model. We implement our solution to show \Sys is robust against data poisoning attacks and the creation of our judge model is scalable. 

---
# Exploring Content and Social Connections of Fake News with Explainable Text and Graph Learning 

**Authors**: Vítor N. Lourenço, Aline Paes, and Tillman Weyde  

**Link**: [PDF](https://arxiv.org/pdf/2508.10040)  

**Abstract**: The global spread of misinformation and concerns about content trustworthiness have driven the development of automated fact-checking systems. Since false information often exploits social media dynamics such as "likes" and user networks to amplify its reach, effective solutions must go beyond content analysis to incorporate these factors. Moreover, simply labelling content as false can be ineffective or even reinforce biases such as automation and confirmation bias. This paper proposes an explainable framework that combines content, social media, and graph-based features to enhance fact-checking. It integrates a misinformation classifier with explainability techniques to deliver complete and interpretable insights supporting classification decisions. Experiments demonstrate that multimodal information improves performance over single modalities, with evaluations conducted on datasets in English, Spanish, and Portuguese. Additionally, the framework's explanations were assessed for interpretability, trustworthiness, and robustness with a novel protocol, showing that it effectively generates human-understandable justifications for its predictions. 

---
# Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries 

**Authors**: Wenqiang Wang, Yan Xiao, Hao Lin, Yangshijie Zhang, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.10039)  

**Abstract**: Current multi-task adversarial text attacks rely on abundant access to shared internal features and numerous queries, often limited to a single task type. As a result, these attacks are less effective against practical scenarios involving black-box feedback APIs, limited queries, or multiple task types. To bridge this gap, we propose \textbf{C}luster and \textbf{E}nsemble \textbf{M}ulti-task Text Adversarial \textbf{A}ttack (\textbf{CEMA}), an effective black-box attack that exploits the transferability of adversarial texts across different tasks. CEMA simplifies complex multi-task scenarios by using a \textit{deep-level substitute model} trained in a \textit{plug-and-play} manner for text classification, enabling attacks without mimicking the victim model. This approach requires only a few queries for training, converting multi-task attacks into classification attacks and allowing attacks across various tasks.
CEMA generates multiple adversarial candidates using different text classification methods and selects the one that most effectively attacks substitute models.
In experiments involving multi-task models with two, three, or six tasks--spanning classification, translation, summarization, and text-to-image generation--CEMA demonstrates significant attack success with as few as 100 queries. Furthermore, CEMA can target commercial APIs (e.g., Baidu and Google Translate), large language models (e.g., ChatGPT 4o), and image-generation models (e.g., Stable Diffusion V2), showcasing its versatility and effectiveness in real-world applications. 

---
# Certifiably robust malware detectors by design 

**Authors**: Pierre-Francois Gimenez, Sarath Sivaprasad, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2508.10038)  

**Abstract**: Malware analysis involves analyzing suspicious software to detect malicious payloads. Static malware analysis, which does not require software execution, relies increasingly on machine learning techniques to achieve scalability. Although such techniques obtain very high detection accuracy, they can be easily evaded with adversarial examples where a few modifications of the sample can dupe the detector without modifying the behavior of the software. Unlike other domains, such as computer vision, creating an adversarial example of malware without altering its functionality requires specific transformations. We propose a new model architecture for certifiably robust malware detection by design. In addition, we show that every robust detector can be decomposed into a specific structure, which can be applied to learn empirically robust malware detectors, even on fragile features. Our framework ERDALT is based on this structure. We compare and validate these approaches with machine-learning-based malware detection methods, allowing for robust detection with limited reduction of detection performance. 

---
# Reflect then Learn: Active Prompting for Information Extraction Guided by Introspective Confusion 

**Authors**: Dong Zhao, Yadong Wang, Xiang Chen, Chenxi Wang, Hongliang Dai, Chuanxing Geng, Shengzhong Zhang, Shaoyuan Li, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10036)  

**Abstract**: Large Language Models (LLMs) show remarkable potential for few-shot information extraction (IE), yet their performance is highly sensitive to the choice of in-context examples. Conventional selection strategies often fail to provide informative guidance, as they overlook a key source of model fallibility: confusion stemming not just from semantic content, but also from the generation of well-structured formats required by IE tasks. To address this, we introduce Active Prompting for Information Extraction (APIE), a novel active prompting framework guided by a principle we term introspective confusion. Our method empowers an LLM to assess its own confusion through a dual-component uncertainty metric that uniquely quantifies both Format Uncertainty (difficulty in generating correct syntax) and Content Uncertainty (inconsistency in extracted semantics). By ranking unlabeled data with this comprehensive score, our framework actively selects the most challenging and informative samples to serve as few-shot exemplars. Extensive experiments on four benchmarks show that our approach consistently outperforms strong baselines, yielding significant improvements in both extraction accuracy and robustness. Our work highlights the critical importance of a fine-grained, dual-level view of model uncertainty when it comes to building effective and reliable structured generation systems. 

---
# Jet Image Tagging Using Deep Learning: An Ensemble Model 

**Authors**: Juvenal Bassa, Vidya Manian, Sudhir Malik, Arghya Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2508.10034)  

**Abstract**: Jet classification in high-energy particle physics is important for understanding fundamental interactions and probing phenomena beyond the Standard Model. Jets originate from the fragmentation and hadronization of quarks and gluons, and pose a challenge for identification due to their complex, multidimensional structure. Traditional classification methods often fall short in capturing these intricacies, necessitating advanced machine learning approaches. In this paper, we employ two neural networks simultaneously as an ensemble to tag various jet types. We convert the jet data to two-dimensional histograms instead of representing them as points in a higher-dimensional space. Specifically, this ensemble approach, hereafter referred to as Ensemble Model, is used to tag jets into classes from the JetNet dataset, corresponding to: Top Quarks, Light Quarks (up or down), and W and Z bosons. For the jet classes mentioned above, we show that the Ensemble Model can be used for both binary and multi-categorical classification. This ensemble approach learns jet features by leveraging the strengths of each constituent network achieving superior performance compared to either individual network. 

---
# Cognitive Cybersecurity for Artificial Intelligence: Guardrail Engineering with CCS-7 

**Authors**: Yuksel Aydin  

**Link**: [PDF](https://arxiv.org/pdf/2508.10033)  

**Abstract**: Language models exhibit human-like cognitive vulnerabilities, such as emotional framing, that escape traditional behavioral alignment. We present CCS-7 (Cognitive Cybersecurity Suite), a taxonomy of seven vulnerabilities grounded in human cognitive security research. To establish a human benchmark, we ran a randomized controlled trial with 151 participants: a "Think First, Verify Always" (TFVA) lesson improved cognitive security by +7.9% overall. We then evaluated TFVA-style guardrails across 12,180 experiments on seven diverse language model architectures. Results reveal architecture-dependent risk patterns: some vulnerabilities (e.g., identity confusion) are almost fully mitigated, while others (e.g., source interference) exhibit escalating backfire, with error rates increasing by up to 135% in certain models. Humans, in contrast, show consistent moderate improvement. These findings reframe cognitive safety as a model-specific engineering problem: interventions effective in one architecture may fail, or actively harm, another, underscoring the need for architecture-aware cognitive safety testing before deployment. 

---
# The Cost of Thinking: Increased Jailbreak Risk in Large Language Models 

**Authors**: Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10032)  

**Abstract**: Thinking mode has always been regarded as one of the most valuable modes in LLMs. However, we uncover a surprising and previously overlooked phenomenon: LLMs with thinking mode are more easily broken by Jailbreak attack. We evaluate 9 LLMs on AdvBench and HarmBench and find that the success rate of attacking thinking mode in LLMs is almost higher than that of non-thinking mode. Through large numbers of sample studies, it is found that for educational purposes and excessively long thinking lengths are the characteristics of successfully attacked data, and LLMs also give harmful answers when they mostly know that the questions are harmful. In order to alleviate the above problems, this paper proposes a method of safe thinking intervention for LLMs, which explicitly guides the internal thinking processes of LLMs by adding "specific thinking tokens" of LLMs to the prompt. The results demonstrate that the safe thinking intervention can significantly reduce the attack success rate of LLMs with thinking mode. 

---
# Context Misleads LLMs: The Role of Context Filtering in Maintaining Safe Alignment of LLMs 

**Authors**: Jinhwa Kim, Ian G. Harris  

**Link**: [PDF](https://arxiv.org/pdf/2508.10031)  

**Abstract**: While Large Language Models (LLMs) have shown significant advancements in performance, various jailbreak attacks have posed growing safety and ethical risks. Malicious users often exploit adversarial context to deceive LLMs, prompting them to generate responses to harmful queries. In this study, we propose a new defense mechanism called Context Filtering model, an input pre-processing method designed to filter out untrustworthy and unreliable context while identifying the primary prompts containing the real user intent to uncover concealed malicious intent. Given that enhancing the safety of LLMs often compromises their helpfulness, potentially affecting the experience of benign users, our method aims to improve the safety of the LLMs while preserving their original performance. We evaluate the effectiveness of our model in defending against jailbreak attacks through comparative analysis, comparing our approach with state-of-the-art defense mechanisms against six different attacks and assessing the helpfulness of LLMs under these defenses. Our model demonstrates its ability to reduce the Attack Success Rates of jailbreak attacks by up to 88% while maintaining the original LLMs' performance, achieving state-of-the-art Safety and Helpfulness Product results. Notably, our model is a plug-and-play method that can be applied to all LLMs, including both white-box and black-box models, to enhance their safety without requiring any fine-tuning of the models themselves. We will make our model publicly available for research purposes. 

---
# Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models 

**Authors**: Saaduddin Mahmud, Mason Nakamura, Kyle H. Wray, Shlomo Zilberstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.10030)  

**Abstract**: Prompt optimization methods have demonstrated significant effectiveness in aligning black-box large language models (LLMs). In parallel, inference scaling strategies such as Best-of-N Sampling and Majority Voting have also proven to enhance alignment and performance by trading off computation. However, existing prompt optimization approaches are inference strategy agnostic; that is, they optimize prompts without regard to the inference strategy employed during deployment. This constitutes a significant methodological gap, as our empirical and theoretical analysis reveals a strong interdependence between these two paradigms. Moreover, we find that user preferences regarding trade-offs among multiple objectives and inference budgets substantially influence the choice of prompt and inference configuration. To address this gap, we introduce a unified novel framework named IAPO (Inference-Aware Prompt Optimization) that jointly optimizes the prompt and inference scale, while being aware of the inference budget and different task objectives. We then develop a fixed-budget training algorithm for IAPO, which we call PSST (Prompt Scaling via Sequential Trimming), and analyze finite-budget guarantees on error probability. Finally, we evaluate the effectiveness of PSST on six different tasks, including multi-objective text generation and reasoning, and demonstrate the critical role of incorporating inference-awareness when aligning black-box LLMs through prompt optimization. 

---
# Latent Fusion Jailbreak: Blending Harmful and Harmless Representations to Elicit Unsafe LLM Outputs 

**Authors**: Wenpeng Xing, Mohan Li, Chunqiang Hu, Haitao XuNingyu Zhang, Bo Lin, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.10029)  

**Abstract**: Large language models (LLMs) demonstrate impressive capabilities in various language tasks but are susceptible to jailbreak attacks that circumvent their safety alignments. This paper introduces Latent Fusion Jailbreak (LFJ), a representation-based attack that interpolates hidden states from harmful and benign query pairs to elicit prohibited responses. LFJ begins by selecting query pairs with high thematic and syntactic similarity, then performs gradient-guided interpolation at influential layers and tokens, followed by optimization to balance attack success, output fluency, and computational efficiency. Evaluations on models such as Vicuna and LLaMA-2 across benchmarks like AdvBench and MaliciousInstruct yield an average attack success rate (ASR) of 94.01%, outperforming existing methods. To mitigate LFJ, we propose an adversarial training defense that fine-tunes models on interpolated examples, reducing ASR by over 80% without degrading performance on benign inputs. Ablation studies validate the importance of query pair selection, hidden state interpolation components, and optimization strategies in LFJ's effectiveness. 

---
# PREF: Reference-Free Evaluation of Personalised Text Generation in LLMs 

**Authors**: Xiao Fu, Hossein A. Rahmani, Bin Wu, Jerome Ramos, Emine Yilmaz, Aldo Lipani  

**Link**: [PDF](https://arxiv.org/pdf/2508.10028)  

**Abstract**: Personalised text generation is essential for user-centric information systems, yet most evaluation methods overlook the individuality of users. We introduce \textbf{PREF}, a \textbf{P}ersonalised \textbf{R}eference-free \textbf{E}valuation \textbf{F}ramework that jointly measures general output quality and user-specific alignment without requiring gold personalised references. PREF operates in a three-step pipeline: (1) a coverage stage uses a large language model (LLM) to generate a comprehensive, query-specific guideline covering universal criteria such as factuality, coherence, and completeness; (2) a preference stage re-ranks and selectively augments these factors using the target user's profile, stated or inferred preferences, and context, producing a personalised evaluation rubric; and (3) a scoring stage applies an LLM judge to rate candidate answers against this rubric, ensuring baseline adequacy while capturing subjective priorities. This separation of coverage from preference improves robustness, transparency, and reusability, and allows smaller models to approximate the personalised quality of larger ones. Experiments on the PrefEval benchmark, including implicit preference-following tasks, show that PREF achieves higher accuracy, better calibration, and closer alignment with human judgments than strong baselines. By enabling scalable, interpretable, and user-aligned evaluation, PREF lays the groundwork for more reliable assessment and development of personalised language generation systems. 

---
# LLMCARE: Alzheimer's Detection via Transformer Models Enhanced by LLM-Generated Synthetic Data 

**Authors**: Ali Zolnour, Hossein Azadmaleki, Yasaman Haghbin, Fatemeh Taherinezhad, Mohamad Javad Momeni Nezhad, Sina Rashidi, Masoud Khani, AmirSajjad Taleban, Samin Mahdizadeh Sani, Maryam Dadkhah, James M. Noble, Suzanne Bakken, Yadollah Yaghoobzadeh, Abdol-Hossein Vahabie, Masoud Rouhizadeh, Maryam Zolnoori  

**Link**: [PDF](https://arxiv.org/pdf/2508.10027)  

**Abstract**: Alzheimer's disease and related dementias (ADRD) affect approximately five million older adults in the U.S., yet over half remain undiagnosed. Speech-based natural language processing (NLP) offers a promising, scalable approach to detect early cognitive decline through linguistic markers.
To develop and evaluate a screening pipeline that (i) fuses transformer embeddings with handcrafted linguistic features, (ii) tests data augmentation using synthetic speech generated by large language models (LLMs), and (iii) benchmarks unimodal and multimodal LLM classifiers for ADRD detection.
Transcripts from the DementiaBank "cookie-theft" task (n = 237) were used. Ten transformer models were evaluated under three fine-tuning strategies. A fusion model combined embeddings from the top-performing transformer with 110 lexical-derived linguistic features. Five LLMs (LLaMA-8B/70B, MedAlpaca-7B, Ministral-8B, GPT-4o) were fine-tuned to generate label-conditioned synthetic speech, which was used to augment training data. Three multimodal models (GPT-4o, Qwen-Omni, Phi-4) were tested for speech-text classification in zero-shot and fine-tuned settings.
The fusion model achieved F1 = 83.3 (AUC = 89.5), outperforming linguistic or transformer-only baselines. Augmenting training data with 2x MedAlpaca-7B synthetic speech increased F1 to 85.7. Fine-tuning significantly improved unimodal LLM classifiers (e.g., MedAlpaca: F1 = 47.3 -> 78.5 F1). Current multimodal models demonstrated lower performance (GPT-4o = 70.2 F1; Qwen = 66.0). Performance gains aligned with the distributional similarity between synthetic and real speech.
Integrating transformer embeddings with linguistic features enhances ADRD detection from speech. Clinically tuned LLMs effectively support both classification and data augmentation, while further advancement is needed in multimodal modeling. 

---
# SABER: Switchable and Balanced Training for Efficient LLM Reasoning 

**Authors**: Kai Zhao, Yanjun Zhao, Jiaming Song, Shien He, Lusheng Zhang, Qiang Zhang, Tianjiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.10026)  

**Abstract**: Large language models (LLMs) empowered by chain-of-thought reasoning have achieved impressive accuracy on complex tasks but suffer from excessive inference costs and latency when applied uniformly to all problems. We propose SABER (Switchable and Balanced Training for Efficient LLM Reasoning), a reinforcement learning framework that endows LLMs with user-controllable, token-budgeted reasoning. SABER first profiles each training example's base-model thinking token usage and assigns it to one of the predefined budget tiers. During fine-tuning, the model is guided by system prompts and length-aware rewards to respect its assigned budget. In parallel, we incorporate no-think examples to ensure the model remains reliable even when explicit reasoning is turned off. SABER further supports four discrete inference modes - NoThink, FastThink, CoreThink, and DeepThink, enabling flexible trade-offs between latency and reasoning depth. Extensive evaluations on math reasoning (MATH, GSM8K), code generation (MBPP), and logical reasoning (LiveBench-Reasoning) demonstrate that SABER achieves high accuracy under tight budgets, graceful degradation, and effective cross-scale and cross-domain generalization. In particular, SABER-FastThink cuts reasoning length by 65.4% and yields a 3.6% accuracy gain compared with the base model on the MATH benchmark. 

---
# Detecting and explaining postpartum depression in real-time with generative artificial intelligence 

**Authors**: Silvia García-Méndez, Francisco de Arriba-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2508.10025)  

**Abstract**: Among the many challenges mothers undergo after childbirth, postpartum depression (PPD) is a severe condition that significantly impacts their mental and physical well-being. Consequently, the rapid detection of ppd and their associated risk factors is critical for in-time assessment and intervention through specialized prevention procedures. Accordingly, this work addresses the need to help practitioners make decisions with the latest technological advancements to enable real-time screening and treatment recommendations. Mainly, our work contributes to an intelligent PPD screening system that combines Natural Language Processing, Machine Learning (ML), and Large Language Models (LLMs) towards an affordable, real-time, and non-invasive free speech analysis. Moreover, it addresses the black box problem since the predictions are described to the end users thanks to the combination of LLMs with interpretable ml models (i.e., tree-based algorithms) using feature importance and natural language. The results obtained are 90 % on ppd detection for all evaluation metrics, outperforming the competing solutions in the literature. Ultimately, our solution contributes to the rapid detection of PPD and their associated risk factors, critical for in-time and proper assessment and intervention. 

---
# RTTC: Reward-Guided Collaborative Test-Time Compute 

**Authors**: J. Pablo Muñoz, Jinjie Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10024)  

**Abstract**: Test-Time Compute (TTC) has emerged as a powerful paradigm for enhancing the performance of Large Language Models (LLMs) at inference, leveraging strategies such as Test-Time Training (TTT) and Retrieval-Augmented Generation (RAG). However, the optimal adaptation strategy varies across queries, and indiscriminate application of TTC strategy incurs substantial computational overhead. In this work, we introduce Reward-Guided Test-Time Compute (RTTC), a novel framework that adaptively selects the most effective TTC strategy for each query via a pretrained reward model, maximizing downstream accuracy across diverse domains and tasks. RTTC operates in a distributed server-client architecture, retrieving relevant samples from a remote knowledge base and applying RAG or lightweight fine-tuning on client devices only when necessary. To further mitigate redundant computation, we propose Query-State Caching, which enables the efficient reuse of historical query states at both retrieval and adaptation levels. Extensive experiments across multiple LLMs and benchmarks demonstrate that RTTC consistently achieves superior accuracy compared to vanilla RAG or TTT, validating the necessity of adaptive, reward-guided TTC selection and the potential of RTTC for scalable, high-performance language model adaptation. 

---
# Conformal P-Value in Multiple-Choice Question Answering Tasks with Provable Risk Control 

**Authors**: Yuanchang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.10022)  

**Abstract**: This study introduces a significance testing-enhanced conformal prediction (CP) framework to improve trustworthiness of large language models (LLMs) in multiple-choice question answering (MCQA). While LLMs have been increasingly deployed in disciplinary QA scenarios, hallucination and nonfactual generation substantially compromise response reliability. Although CP provides statistically rigorous marginal coverage guarantees for prediction sets, and significance testing offers established statistical rigor, their synergistic integration remains unexplored. To mitigate hallucination and factual inaccuracies, our framework integrates $p$-value computation with conformity scoring through self-consistency resampling of MCQA responses. This approach calculates option frequencies to address LLMs' black-box nature, subsequently constructing prediction sets via null hypothesis testing ($\mathcal{H}_0$) with empirically derived $p$-values. Evaluations on MMLU and MMLU-Pro benchmarks using off-the-shelf LLMs demonstrate: (1) The enhanced CP achieves user-specified empirical miscoverage rates; (2) Test-set average prediction set size (APSS) decreases monotonically with increasing risk levels ($\alpha$), validating APSS as an effective uncertainty metric. This work establishes a principled statistical framework for trustworthy LLM deployment in high-stakes QA applications. 

---
# LATTE: Learning Aligned Transactions and Textual Embeddings for Bank Clients 

**Authors**: Egor Fadeev, Dzhambulat Mollaev, Aleksei Shestov, Dima Korolev, Omar Zoloev, Ivan Kireev, Andrey Savchenko, Maksim Makarenko  

**Link**: [PDF](https://arxiv.org/pdf/2508.10021)  

**Abstract**: Learning clients embeddings from sequences of their historic communications is central to financial applications. While large language models (LLMs) offer general world knowledge, their direct use on long event sequences is computationally expensive and impractical in real-world pipelines. In this paper, we propose LATTE, a contrastive learning framework that aligns raw event embeddings with semantic embeddings from frozen LLMs. Behavioral features are summarized into short prompts, embedded by the LLM, and used as supervision via contrastive loss. The proposed approach significantly reduces inference cost and input size compared to conventional processing of complete sequence by LLM. We experimentally show that our method outperforms state-of-the-art techniques for learning event sequence representations on real-world financial datasets while remaining deployable in latency-sensitive environments. 

---
# FedCoT: Communication-Efficient Federated Reasoning Enhancement for Large Language Models 

**Authors**: Chuan Li, Qianyi Zhao, Fengran Mo, Cen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10020)  

**Abstract**: Efficiently enhancing the reasoning capabilities of large language models (LLMs) in federated learning environments remains challenging, particularly when balancing performance gains with strict computational, communication, and privacy constraints. This challenge is especially acute in healthcare, where decisions-spanning clinical, operational, and patient-facing contexts-demand not only accurate outputs but also interpretable, traceable rationales to ensure safety, accountability, and regulatory compliance. Conventional federated tuning approaches on LLM fail to address this need: they optimize primarily for answer correctness while neglecting rationale quality, leaving CoT capabilities dependent on models' innate pre-training abilities. Moreover, existing methods for improving rationales typically rely on privacy-violating knowledge distillation from centralized models. Additionally, the communication overhead in traditional federated fine-tuning on LLMs remains substantial. We addresses this gap by proposing FedCoT, a novel framework specifically designed to enhance reasoning in federated settings. FedCoT leverages a lightweight chain-of-thought enhancement mechanism: local models generate multiple reasoning paths, and a compact discriminator dynamically selects the most promising one. This approach improves reasoning accuracy and robustness while providing valuable interpretability, which is particularly critical for medical applications. To manage client heterogeneity efficiently, we adopt an improved aggregation approach building upon advanced LoRA module stacking, incorporating client classifier-awareness to achieve noise-free aggregation across diverse clients. Comprehensive experiments on medical reasoning tasks demonstrate that FedCoT significantly boosts client-side reasoning performance under stringent resource budgets while fully preserving data privacy. 

---
# Decoupling Understanding from Reasoning via Problem Space Mapping for Small-scale Model Reasoning 

**Authors**: Li Wang, Changhao Zhang, Zengqi Xiu, Kai Lu, Xin Yu, Kui Zhang, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10019)  

**Abstract**: Despite recent advances in the reasoning capabilities of Large Language Models (LLMs), improving the reasoning ability of Small Language Models (SLMs, e.g., $\leq$ 1.5B) remains challenging. A key obstacle lies in the complexity and variability of natural language: essentially equivalent problems often appear in diverse surface forms, often obscured by redundant or distracting details. This imposes a dual burden on SLMs: they must first extract the core problem from complex linguistic input, and then perform reasoning based on that understanding. The resulting vast and noisy problem space hinders optimization, particularly for models with limited capacity. To address this, we propose a new framework that decouples understanding from reasoning by mapping natural language problems into a canonical problem space-a semantically simplified yet expressive domain. This enables SLMs to focus on reasoning over standardized inputs, free from linguistic variability. Within this framework, we introduce DURIT (Decoupled Understanding from Reasoning via Iterative Training), a three-step algorithm that iteratively: (1) mapping natural language problems via reinforcement learning, (2) aligns reasoning trajectories through self-distillation, and (3) trains reasoning policies in the problem space. The mapper and reasoner are co-trained in an alternating loop throughout this process. Experiments show that DURIT substantially improves SLMs' performance on both in-domain and out-of-domain mathematical and logical reasoning tasks. Beyond improving reasoning capabilities, DURIT also improves the robustness of reasoning, validating decoupling understanding from reasoning as an effective strategy for strengthening SLMs. 

---
# A Rose by Any Other Name Would Smell as Sweet: Categorical Homotopy Theory for Large Language Models 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10018)  

**Abstract**: Natural language is replete with superficially different statements, such as ``Charles Darwin wrote" and ``Charles Darwin is the author of", which carry the same meaning. Large language models (LLMs) should generate the same next-token probabilities in such cases, but usually do not. Empirical workarounds have been explored, such as using k-NN estimates of sentence similarity to produce smoothed estimates. In this paper, we tackle this problem more abstractly, introducing a categorical homotopy framework for LLMs. We introduce an LLM Markov category to represent probability distributions in language generated by an LLM, where the probability of a sentence, such as ``Charles Darwin wrote" is defined by an arrow in a Markov category. However, this approach runs into difficulties as language is full of equivalent rephrases, and each generates a non-isomorphic arrow in the LLM Markov category. To address this fundamental problem, we use categorical homotopy techniques to capture ``weak equivalences" in an LLM Markov category. We present a detailed overview of application of categorical homotopy to LLMs, from higher algebraic K-theory to model categories, building on powerful theoretical results developed over the past half a century. 

---
# A Robust Pipeline for Differentially Private Federated Learning on Imbalanced Clinical Data using SMOTETomek and FedProx 

**Authors**: Rodrigo Tertulino  

**Link**: [PDF](https://arxiv.org/pdf/2508.10017)  

**Abstract**: Federated Learning (FL) presents a groundbreaking approach for collaborative health research, allowing model training on decentralized data while safeguarding patient privacy. FL offers formal security guarantees when combined with Differential Privacy (DP). The integration of these technologies, however, introduces a significant trade-off between privacy and clinical utility, a challenge further complicated by the severe class imbalance often present in medical datasets. The research presented herein addresses these interconnected issues through a systematic, multi-stage analysis. An FL framework was implemented for cardiovascular risk prediction, where initial experiments showed that standard methods struggled with imbalanced data, resulting in a recall of zero. To overcome such a limitation, we first integrated the hybrid Synthetic Minority Over-sampling Technique with Tomek Links (SMOTETomek) at the client level, successfully developing a clinically useful model. Subsequently, the framework was optimized for non-IID data using a tuned FedProx algorithm. Our final results reveal a clear, non-linear trade-off between the privacy budget (epsilon) and model recall, with the optimized FedProx consistently out-performing standard FedAvg. An optimal operational region was identified on the privacy-utility frontier, where strong privacy guarantees (with epsilon 9.0) can be achieved while maintaining high clinical utility (recall greater than 77%). Ultimately, our study provides a practical methodological blueprint for creating effective, secure, and accurate diagnostic tools that can be applied to real-world, heterogeneous healthcare data. 

---
# Beyond Hard Sharing: Efficient Multi-Task Speech-to-Text Modeling with Supervised Mixture of Experts 

**Authors**: Hojun Jin, Eunsoo Hong, Ziwon Hyung, Sungjun Lim, Seungjin Lee, Keunseok Cho  

**Link**: [PDF](https://arxiv.org/pdf/2508.10009)  

**Abstract**: Hard-parameter sharing is a common strategy to train a single model jointly across diverse tasks. However, this often leads to task interference, impeding overall model performance. To address the issue, we propose a simple yet effective Supervised Mixture of Experts (S-MoE). Unlike traditional Mixture of Experts models, S-MoE eliminates the need for training gating functions by utilizing special guiding tokens to route each task to its designated expert. By assigning each task to a separate feedforward network, S-MoE overcomes the limitations of hard-parameter sharing. We further apply S-MoE to a speech-to-text model, enabling the model to process mixed-bandwidth input while jointly performing automatic speech recognition (ASR) and speech translation (ST). Experimental results demonstrate the effectiveness of the proposed S-MoE, achieving a 6.35% relative improvement in Word Error Rate (WER) when applied to both the encoder and decoder. 

---
# From Answers to Questions: EQGBench for Evaluating LLMs' Educational Question Generation 

**Authors**: Chengliang Zhou, Mei Wang, Ting Zhang, Qiannan Zhu, Jian Li, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10005)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in mathematical problem-solving. However, the transition from providing answers to generating high-quality educational questions presents significant challenges that remain underexplored. To advance Educational Question Generation (EQG) and facilitate LLMs in generating pedagogically valuable and educationally effective questions, we introduce EQGBench, a comprehensive benchmark specifically designed for evaluating LLMs' performance in Chinese EQG. EQGBench establishes a five-dimensional evaluation framework supported by a dataset of 900 evaluation samples spanning three fundamental middle school disciplines: mathematics, physics, and chemistry. The dataset incorporates user queries with varying knowledge points, difficulty gradients, and question type specifications to simulate realistic educational scenarios. Through systematic evaluation of 46 mainstream large models, we reveal significant room for development in generating questions that reflect educational value and foster students' comprehensive abilities. 

---
# User Perception of Attention Visualizations: Effects on Interpretability Across Evidence-Based Medical Documents 

**Authors**: Andrés Carvallo, Denis Parra, Peter Brusilovsky, Hernan Valdivieso, Gabriel Rada, Ivania Donoso, Vladimir Araujo  

**Link**: [PDF](https://arxiv.org/pdf/2508.10004)  

**Abstract**: The attention mechanism is a core component of the Transformer architecture. Beyond improving performance, attention has been proposed as a mechanism for explainability via attention weights, which are associated with input features (e.g., tokens in a document). In this context, larger attention weights may imply more relevant features for the model's prediction. In evidence-based medicine, such explanations could support physicians' understanding and interaction with AI systems used to categorize biomedical literature. However, there is still no consensus on whether attention weights provide helpful explanations. Moreover, little research has explored how visualizing attention affects its usefulness as an explanation aid. To bridge this gap, we conducted a user study to evaluate whether attention-based explanations support users in biomedical document classification and whether there is a preferred way to visualize them. The study involved medical experts from various disciplines who classified articles based on study design (e.g., systematic reviews, broad synthesis, randomized and non-randomized trials). Our findings show that the Transformer model (XLNet) classified documents accurately; however, the attention weights were not perceived as particularly helpful for explaining the predictions. However, this perception varied significantly depending on how attention was visualized. Contrary to Munzner's principle of visual effectiveness, which favors precise encodings like bar length, users preferred more intuitive formats, such as text brightness or background color. While our results do not confirm the overall utility of attention weights for explanation, they suggest that their perceived helpfulness is influenced by how they are visually presented. 

---
# Semantic Structure in Large Language Model Embeddings 

**Authors**: Austin C. Kozlowski, Callin Dai, Andrei Boutyline  

**Link**: [PDF](https://arxiv.org/pdf/2508.10003)  

**Abstract**: Psychological research consistently finds that human ratings of words across diverse semantic scales can be reduced to a low-dimensional form with relatively little information loss. We find that the semantic associations encoded in the embedding matrices of large language models (LLMs) exhibit a similar structure. We show that the projections of words on semantic directions defined by antonym pairs (e.g. kind - cruel) correlate highly with human ratings, and further find that these projections effectively reduce to a 3-dimensional subspace within LLM embeddings, closely resembling the patterns derived from human survey responses. Moreover, we find that shifting tokens along one semantic direction causes off-target effects on geometrically aligned features proportional to their cosine similarity. These findings suggest that semantic features are entangled within LLMs similarly to how they are interconnected in human language, and a great deal of semantic information, despite its apparent complexity, is surprisingly low-dimensional. Furthermore, accounting for this semantic structure may prove essential for avoiding unintended consequences when steering features. 

---
# HiFACTMix: A Code-Mixed Benchmark and Graph-Aware Model for EvidenceBased Political Claim Verification in Hinglish 

**Authors**: Rakesh Thakur, Sneha Sharma, Gauri Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2508.10001)  

**Abstract**: Fact-checking in code-mixed, low-resource languages such as Hinglish remains an underexplored challenge in natural language processing. Existing fact-verification systems largely focus on high-resource, monolingual settings and fail to generalize to real-world political discourse in linguistically diverse regions like India. Given the widespread use of Hinglish by public figures, particularly political figures, and the growing influence of social media on public opinion, there's a critical need for robust, multilingual and context-aware fact-checking tools. To address this gap a novel benchmark HiFACT dataset is introduced with 1,500 realworld factual claims made by 28 Indian state Chief Ministers in Hinglish, under a highly code-mixed low-resource setting. Each claim is annotated with textual evidence and veracity labels. To evaluate this benchmark, a novel graphaware, retrieval-augmented fact-checking model is proposed that combines multilingual contextual encoding, claim-evidence semantic alignment, evidence graph construction, graph neural reasoning, and natural language explanation generation. Experimental results show that HiFACTMix outperformed accuracy in comparison to state of art multilingual baselines models and provides faithful justifications for its verdicts. This work opens a new direction for multilingual, code-mixed, and politically grounded fact verification research. 

---
# INTIMA: A Benchmark for Human-AI Companionship Behavior 

**Authors**: Lucie-Aimée Kaffee, Giada Pistilli, Yacine Jernite  

**Link**: [PDF](https://arxiv.org/pdf/2508.09998)  

**Abstract**: AI companionship, where users develop emotional bonds with AI systems, has emerged as a significant pattern with positive but also concerning implications. We introduce Interactions and Machine Attachment Benchmark (INTIMA), a benchmark for evaluating companionship behaviors in language models. Drawing from psychological theories and user data, we develop a taxonomy of 31 behaviors across four categories and 368 targeted prompts. Responses to these prompts are evaluated as companionship-reinforcing, boundary-maintaining, or neutral. Applying INTIMA to Gemma-3, Phi-4, o3-mini, and Claude-4 reveals that companionship-reinforcing behaviors remain much more common across all models, though we observe marked differences between models. Different commercial providers prioritize different categories within the more sensitive parts of the benchmark, which is concerning since both appropriate boundary-setting and emotional support matter for user well-being. These findings highlight the need for more consistent approaches to handling emotionally charged interactions. 

---
# OpenFPL: An open-source forecasting method rivaling state-of-the-art Fantasy Premier League services 

**Authors**: Daniel Groos  

**Link**: [PDF](https://arxiv.org/pdf/2508.09992)  

**Abstract**: Fantasy Premier League engages the football community in selecting the Premier League players who will perform best from gameweek to gameweek. Access to accurate performance forecasts gives participants an edge over competitors by guiding expectations about player outcomes and reducing uncertainty in squad selection. However, high-accuracy forecasts are currently limited to commercial services whose inner workings are undisclosed and that rely on proprietary data. This paper aims to democratize access to highly accurate forecasts of player performance by presenting OpenFPL, an open-source Fantasy Premier League forecasting method developed exclusively from public data. Comprising position-specific ensemble models optimized on Fantasy Premier League and Understat data from four previous seasons (2020-21 to 2023-24), OpenFPL achieves accuracy comparable to a leading commercial service when tested prospectively on data from the 2024-25 season. OpenFPL also surpasses the commercial benchmark for high-return players ($>$ 2 points), which are most influential for rank gains. These findings hold across one-, two-, and three-gameweek forecast horizons, supporting long-term planning of transfers and strategies while also informing final-day decisions. 

---
# Bridging AI Innovation and Healthcare Needs: Lessons Learned from Incorporating Modern NLP at The BC Cancer Registry 

**Authors**: Lovedeep Gondara, Gregory Arbour, Raymond Ng, Jonathan Simkin, Shebnum Devji  

**Link**: [PDF](https://arxiv.org/pdf/2508.09991)  

**Abstract**: Automating data extraction from clinical documents offers significant potential to improve efficiency in healthcare settings, yet deploying Natural Language Processing (NLP) solutions presents practical challenges. Drawing upon our experience implementing various NLP models for information extraction and classification tasks at the British Columbia Cancer Registry (BCCR), this paper shares key lessons learned throughout the project lifecycle. We emphasize the critical importance of defining problems based on clear business objectives rather than solely technical accuracy, adopting an iterative approach to development, and fostering deep interdisciplinary collaboration and co-design involving domain experts, end-users, and ML specialists from inception. Further insights highlight the need for pragmatic model selection (including hybrid approaches and simpler methods where appropriate), rigorous attention to data quality (representativeness, drift, annotation), robust error mitigation strategies involving human-in-the-loop validation and ongoing audits, and building organizational AI literacy. These practical considerations, generalizable beyond cancer registries, provide guidance for healthcare organizations seeking to successfully implement AI/NLP solutions to enhance data management processes and ultimately improve patient care and public health outcomes. 

---
# Personalized Product Search Ranking: A Multi-Task Learning Approach with Tabular and Non-Tabular Data 

**Authors**: Lalitesh Morishetti, Abhay Kumar, Jonathan Scott, Kaushiki Nag, Gunjan Sharma, Shanu Vashishtha, Rahul Sridhar, Rohit Chatter, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09636)  

**Abstract**: In this paper, we present a novel model architecture for optimizing personalized product search ranking using a multi-task learning (MTL) framework. Our approach uniquely integrates tabular and non-tabular data, leveraging a pre-trained TinyBERT model for semantic embeddings and a novel sampling technique to capture diverse customer behaviors. We evaluate our model against several baselines, including XGBoost, TabNet, FT-Transformer, DCN-V2, and MMoE, focusing on their ability to handle mixed data types and optimize personalized ranking. Additionally, we propose a scalable relevance labeling mechanism based on click-through rates, click positions, and semantic similarity, offering an alternative to traditional human-annotated labels. Experimental results show that combining non-tabular data with advanced embedding techniques in multi-task learning paradigm significantly enhances model performance. Ablation studies further underscore the benefits of incorporating relevance labels, fine-tuning TinyBERT layers, and TinyBERT query-product embedding interactions. These results demonstrate the effectiveness of our approach in achieving improved personalized product search ranking. 

---
