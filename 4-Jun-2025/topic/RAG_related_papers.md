# V2X-UniPool: Unifying Multimodal Perception and Knowledge Reasoning for Autonomous Driving 

**Authors**: Xuewen Luo, Fengze Yang, Fan Ding, Xiangbo Gao, Shuo Xing, Yang Zhou, Zhengzhong Tu, Chenxi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02580)  

**Abstract**: Knowledge-driven autonomous driving systems(ADs) offer powerful reasoning capabilities, but face two critical challenges: limited perception due to the short-sightedness of single-vehicle sensors, and hallucination arising from the lack of real-time environmental grounding. To address these issues, this paper introduces V2X-UniPool, a unified framework that integrates multimodal Vehicle-to-Everything (V2X) data into a time-indexed and language-based knowledge pool. By leveraging a dual-query Retrieval-Augmented Generation (RAG) mechanism, which enables retrieval of both static and dynamic knowledge, our system enables ADs to perform accurate, temporally consistent reasoning over both static environment and dynamic traffic context. Experiments on a real-world cooperative driving dataset demonstrate that V2X-UniPool significantly enhances motion planning accuracy and reasoning capability. Remarkably, it enables even zero-shot vehicle-side models to achieve state-of-the-art performance by leveraging V2X-UniPool, while simultaneously reducing transmission cost by over 99.9\% compared to prior V2X methods. 

---
# Hybrid AI for Responsive Multi-Turn Online Conversations with Novel Dynamic Routing and Feedback Adaptation 

**Authors**: Priyaranjan Pattnayak, Amit Agarwal, Hansa Meghwani, Hitesh Laxmichand Patel, Srikant Panda  

**Link**: [PDF](https://arxiv.org/pdf/2506.02097)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems and large language model (LLM)-powered chatbots have significantly advanced conversational AI by combining generative capabilities with external knowledge retrieval. Despite their success, enterprise-scale deployments face critical challenges, including diverse user queries, high latency, hallucinations, and difficulty integrating frequently updated domain-specific knowledge. This paper introduces a novel hybrid framework that integrates RAG with intent-based canned responses, leveraging predefined high-confidence responses for efficiency while dynamically routing complex or ambiguous queries to the RAG pipeline. Our framework employs a dialogue context manager to ensure coherence in multi-turn interactions and incorporates a feedback loop to refine intents, dynamically adjust confidence thresholds, and expand response coverage over time. Experimental results demonstrate that the proposed framework achieves a balance of high accuracy (95\%) and low latency (180ms), outperforming RAG and intent-based systems across diverse query types, positioning it as a scalable and adaptive solution for enterprise conversational AI applications. 

---
# A Smart Multimodal Healthcare Copilot with Powerful LLM Reasoning 

**Authors**: Xuejiao Zhao, Siyan Liu, Su-Yin Yang, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2506.02470)  

**Abstract**: Misdiagnosis causes significant harm to healthcare systems worldwide, leading to increased costs and patient risks. MedRAG is a smart multimodal healthcare copilot equipped with powerful large language model (LLM) reasoning, designed to enhance medical decision-making. It supports multiple input modalities, including non-intrusive voice monitoring, general medical queries, and electronic health records. MedRAG provides recommendations on diagnosis, treatment, medication, and follow-up questioning. Leveraging retrieval-augmented generation enhanced by knowledge graph-elicited reasoning, MedRAG retrieves and integrates critical diagnostic insights, reducing the risk of misdiagnosis. It has been evaluated on both public and private datasets, outperforming existing models and offering more specific and accurate healthcare assistance. A demonstration video of MedRAG is available at: this https URL. The source code is available at: this https URL. 

---
# RACE-Align: Retrieval-Augmented and Chain-of-Thought Enhanced Preference Alignment for Large Language Models 

**Authors**: Qihang Yan, Xinyu Zhang, Luming Guo, Qi Zhang, Feifan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02726)  

**Abstract**: Large Language Models (LLMs) struggle with accuracy, domain-specific reasoning, and interpretability in vertical domains. Traditional preference alignment methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) often overlook the underlying knowledge sources and reasoning logic. This paper introduces RACE-Align (Retrieval-Augmented and Chain-of-Thought Enhanced Alignment), a novel framework designed to address these limitations. RACE-Align systematically constructs a binary preference dataset incorporating external knowledge support and explicit Chain-of-Thought (CoT) reasoning, then aligns LLMs using the DPO algorithm. The core innovation lies in its preference data construction strategy: it integrates AI-driven retrieval for factual grounding, enhancing knowledgeability and accuracy, and emphasizes the optimization of domain-specific CoT, treating the reasoning process itself as a key preference dimension. A multi-stage, AI-driven refinement pipeline cost-effectively generates these preference pairs. Experimental validation in Traditional Chinese Medicine (TCM) using Qwen3-1.7B as the base model demonstrates that RACE-Align significantly outperforms the original base model and a model fine-tuned only with Supervised Fine-Tuning (SFT). Improvements were observed across multiple dimensions, including answer accuracy, information richness, application of TCM thinking patterns, logicality and depth of reasoning, and interpretability. These findings suggest RACE-Align offers an effective pathway to enhance LLMs' knowledge application, reasoning reliability, and process transparency in complex vertical domains. 

---
# GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation 

**Authors**: Yilin Xiao, Junnan Dong, Chuang Zhou, Su Dong, Qianwen Zhang, Di Yin, Xing Sun, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.02404)  

**Abstract**: Graph Retrieval Augmented Generation (GraphRAG) has garnered increasing recognition for its potential to enhance large language models (LLMs) by structurally organizing domain-specific corpora and facilitating complex reasoning. However, current evaluations of GraphRAG models predominantly rely on traditional question-answering datasets. Their limited scope in questions and evaluation metrics fails to comprehensively assess the reasoning capacity improvements enabled by GraphRAG models. To address this gap, we introduce GraphRAG-Bench, a large-scale, domain-specific benchmark designed to rigorously evaluate GraphRAG models. Our benchmark offers three key superiorities: \((i)\) Challenging question design. Featuring college-level, domain-specific questions that demand multi-hop reasoning, the benchmark ensures that simple content retrieval is insufficient for problem-solving. For example, some questions require mathematical reasoning or programming. \((ii)\) Diverse task coverage. The dataset includes a broad spectrum of reasoning tasks, multiple-choice, true/false, multi-select, open-ended, and fill-in-the-blank. It spans 16 disciplines in twenty core textbooks. \((iii)\) Holistic evaluation framework. GraphRAG-Bench provides comprehensive assessment across the entire GraphRAG pipeline, including graph construction, knowledge retrieval, and answer generation. Beyond final-answer correctness, it evaluates the logical coherence of the reasoning process. By applying nine contemporary GraphRAG methods to GraphRAG-Bench, we demonstrate its utility in quantifying how graph-based structuring improves model reasoning capabilities. Our analysis reveals critical insights about graph architectures, retrieval efficacy, and reasoning capabilities, offering actionable guidance for the research community. 

---
# CoRe-MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG 

**Authors**: Yang Tian, Fan Liu, Jingyuan Zhang, Victoria W., Yupeng Hu, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.02544)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MMRAG) has been introduced to enhance Multimodal Large Language Models by incorporating externally retrieved multimodal knowledge, but it introduces two challenges: Parametric-Retrieved Knowledge Inconsistency (PRKI), where discrepancies between parametric and retrieved knowledge create uncertainty in determining reliability, and Visual-Textual Knowledge Inconsistency (VTKI), where misalignment between visual and textual sources disrupts entity representation. To address these challenges, we propose \textbf{C}r\textbf{o}ss-source knowledge \textbf{Re}conciliation for \textbf{M}ulti\textbf{M}odal \textbf{RAG} (CoRe-MMRAG), a novel end-to-end framework that effectively reconciles inconsistencies across knowledge sources. CoRe-MMRAG follows a four-stage pipeline: it first generates an internal response from parametric knowledge, then selects the most relevant multimodal evidence via joint similarity assessment, generates an external response, and finally integrates both to produce a reliable answer. Additionally, a specialized training paradigm enhances knowledge source discrimination, multimodal integration, and unified answer generation. Experiments on KB-VQA benchmarks show that CoRe-MMRAG achieves substantial improvements over baseline methods, achieving 5.6\% and 9.3\% performance gains on InfoSeek and Encyclopedic-VQA, respectively. We release code and data at \href{this https URL}{this https URL}. 

---
# KARE-RAG: Knowledge-Aware Refinement and Enhancement for RAG 

**Authors**: Yongjian Li, HaoCheng Chu, Yukun Yan, Zhenghao Liu, Shi Yu, Zheni Zeng, Ruobing Wang, Sen Song, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.02503)  

**Abstract**: Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to access broader knowledge sources, yet factual inconsistencies persist due to noise in retrieved documents-even with advanced retrieval methods. We demonstrate that enhancing generative models' capacity to process noisy content is equally critical for robust performance. In this paper, we present KARE-RAG (Knowledge-Aware Refinement and Enhancement for RAG), which improves knowledge utilization through three key innovations: (1) structured knowledge representations that facilitate error detection during training, (2) Dense Direct Preference Optimization (DDPO)-a refined training objective that prioritizes correction of critical errors, and (3) a contrastive data generation pipeline that maintains semantic consistency while rectifying factual inaccuracies. Experiments show our method significantly enhances standard RAG pipelines across model scales, improving both in-domain and out-of-domain task performance without compromising general capabilities. Notably, these gains are achieved with modest training data, suggesting data-efficient optimization is possible through targeted learning strategies. Our findings establish a new direction for RAG improvement: by improving how models learn to process retrieved content, we can enhance performance across diverse inference paradigms. All data and code will be publicly available on Github. 

---
# ImpRAG: Retrieval-Augmented Generation with Implicit Queries 

**Authors**: Wenzheng Zhang, Xi Victoria Lin, Karl Stratos, Wen-tau Yih, Mingda Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.02279)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems traditionally treat retrieval and generation as separate processes, requiring explicit textual queries to connect them. This separation can limit the ability of models to generalize across diverse tasks. In this work, we propose a query-free RAG system, named ImpRAG, which integrates retrieval and generation into a unified model. ImpRAG allows models to implicitly express their information needs, eliminating the need for human-specified queries. By dividing pretrained decoder-only language models into specialized layer groups, ImpRAG optimizes retrieval and generation tasks simultaneously. Our approach employs a two-stage inference process, using the same model parameters and forward pass for both retrieval and generation, thereby minimizing the disparity between retrievers and language models. Experiments on 8 knowledge-intensive tasks demonstrate that ImpRAG achieves 3.6-11.5 improvements in exact match scores on unseen tasks with diverse formats, highlighting its effectiveness in enabling models to articulate their own information needs and generalize across tasks. Our analysis underscores the importance of balancing retrieval and generation parameters and leveraging generation perplexities as retrieval training objectives for enhanced performance. 

---
# FinS-Pilot: A Benchmark for Online Financial System 

**Authors**: Feng Wang, Yiding Sun, Jiaxin Mao, Wei Xue, Danqing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02037)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various professional domains, with their performance typically evaluated through standardized benchmarks. However, the development of financial RAG benchmarks has been constrained by data confidentiality issues and the lack of dynamic data integration. To address this issue, we introduces FinS-Pilot, a novel benchmark for evaluating RAG systems in online financial applications. Constructed from real-world financial assistant interactions, our benchmark incorporates both real-time API data and structured text sources, organized through an intent classification framework covering critical financial domains such as equity analysis and macroeconomic forecasting. The benchmark enables comprehensive evaluation of financial assistants' capabilities in handling both static knowledge and time-sensitive market information. Through systematic experiments with multiple Chinese leading LLMs, we demonstrate FinS-Pilot's effectiveness in identifying models suitable for financial applications while addressing the current gap in specialized evaluation tools for the financial domain. Our work contributes both a practical evaluation framework and a curated dataset to advance research in financial NLP systems. The code and dataset are accessible on GitHub\footnote{this https URL\_rag\_benchmark}. 

---
# Retrieval-Augmented Generation as Noisy In-Context Learning: A Unified Theory and Risk Bounds 

**Authors**: Yang Guo, Yutian Tao, Yifei Ming, Robert D. Nowak, Yingyu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03100)  

**Abstract**: Retrieval-augmented generation (RAG) has seen many empirical successes in recent years by aiding the LLM with external knowledge. However, its theoretical aspect has remained mostly unexplored. In this paper, we propose the first finite-sample generalization bound for RAG in in-context linear regression and derive an exact bias-variance tradeoff. Our framework views the retrieved texts as query-dependent noisy in-context examples and recovers the classical in-context learning (ICL) and standard RAG as the limit cases. Our analysis suggests that an intrinsic ceiling on generalization error exists on RAG as opposed to the ICL. Furthermore, our framework is able to model retrieval both from the training data and from external corpora by introducing uniform and non-uniform RAG noise. In line with our theory, we show the sample efficiency of ICL and RAG empirically with experiments on common QA benchmarks, such as Natural Questions and TriviaQA. 

---
