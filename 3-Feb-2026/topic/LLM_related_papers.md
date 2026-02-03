# Unifying Ranking and Generation in Query Auto-Completion via Retrieval-Augmented Generation and Multi-Objective Alignment 

**Authors**: Kai Yuan, Anthony Zheng, Jia Hu, Divyanshu Sheth, Hemanth Velaga, Kylee Kim, Matteo Guarrera, Besim Avci, Xuetao Yin, Rajyashree Mukherjee, Sean Suchter  

**Link**: [PDF](https://arxiv.org/pdf/2602.01023)  

**Abstract**: Query Auto-Completion (QAC) suggests query completions as users type, helping them articulate intent and reach results more efficiently. Existing approaches face fundamental challenges: traditional retrieve-and-rank pipelines have limited long-tail coverage and require extensive feature engineering, while recent generative methods suffer from hallucination and safety risks. We present a unified framework that reformulates QAC as end-to-end list generation through Retrieval-Augmented Generation (RAG) and multi-objective Direct Preference Optimization (DPO). Our approach combines three key innovations: (1) reformulating QAC as end-to-end list generation with multi-objective optimization; (2) defining and deploying a suite of rule-based, model-based, and LLM-as-judge verifiers for QAC, and using them in a comprehensive methodology that combines RAG, multi-objective DPO, and iterative critique-revision for high-quality synthetic data; (3) a hybrid serving architecture enabling efficient production deployment under strict latency constraints. Evaluation on a large-scale commercial search platform demonstrates substantial improvements: offline metrics show gains across all dimensions, human evaluation yields +0.40 to +0.69 preference scores, and a controlled online experiment achieves 5.44\% reduction in keystrokes and 3.46\% increase in suggestion adoption, validating that unified generation with RAG and multi-objective alignment provides an effective solution for production QAC. This work represents a paradigm shift to end-to-end generation powered by large language models, RAG, and multi-objective alignment, establishing a production-validated framework that can benefit the broader search and recommendation industry. 

---
# GRAB: An LLM-Inspired Sequence-First Click-Through Rate Prediction Modeling Paradigm 

**Authors**: Shaopeng Chen, Chuyue Xie, Huimin Ren, Shaozong Zhang, Han Zhang, Ruobing Cheng, Zhiqiang Cao, Zehao Ju, Gao Yu, Jie Ding, Xiaodong Chen, Xuewu Jiao, Shuanglong Li, Liu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.01865)  

**Abstract**: Traditional Deep Learning Recommendation Models (DLRMs) face increasing bottlenecks in performance and efficiency, often struggling with generalization and long-sequence modeling. Inspired by the scaling success of Large Language Models (LLMs), we propose Generative Ranking for Ads at Baidu (GRAB), an end-to-end generative framework for Click-Through Rate (CTR) prediction. GRAB integrates a novel Causal Action-aware Multi-channel Attention (CamA) mechanism to effectively capture temporal dynamics and specific action signals within user behavior sequences. Full-scale online deployment demonstrates that GRAB significantly outperforms established DLRMs, delivering a 3.05% increase in revenue and a 3.49% rise in CTR. Furthermore, the model demonstrates desirable scaling behavior: its expressive power shows a monotonic and approximately linear improvement as longer interaction sequences are utilized. 

---
# AI-assisted Protocol Information Extraction For Improved Accuracy and Efficiency in Clinical Trial Workflows 

**Authors**: Ramtin Babaeipour, François Charest, Madison Wright  

**Link**: [PDF](https://arxiv.org/pdf/2602.00052)  

**Abstract**: Increasing clinical trial protocol complexity, amendments, and challenges around knowledge management create significant burden for trial teams. Structuring protocol content into standard formats has the potential to improve efficiency, support documentation quality, and strengthen compliance. We evaluate an Artificial Intelligence (AI) system using generative LLMs with Retrieval-Augmented Generation (RAG) for automated clinical trial protocol information extraction. We compare the extraction accuracy of our clinical-trial-specific RAG process against that of publicly available (standalone) LLMs. We also assess the operational impact of AI-assistance on simulated extraction CRC workflows. Our RAG process was measured as more accurate (87.8%) than standalone LLMs with fine-tuned prompts (62.6%) against expert-supported reference annotations. In the simulated extraction workflows, AI-assisted tasks were completed 40% faster, rated as less cognitively demanding and strongly preferred by users. While expert oversight remains essential, this suggests that AI-assisted extraction can enable protocol intelligence at scale, motivating the integration of similar methodologies into real world clinical workflows to further validate its impact on feasibility, study start-up, and post-activation monitoring. 

---
# Chained Prompting for Better Systematic Review Search Strategies 

**Authors**: Fatima Nasser, Fouad Trad, Ammar Mohanna, Ghada El-Hajj Fuleihan, Ali Chehab  

**Link**: [PDF](https://arxiv.org/pdf/2602.00011)  

**Abstract**: Systematic reviews require the use of rigorously designed search strategies to ensure both comprehensive retrieval and minimization of bias. Conventional manual approaches, although methodologically systematic, are resource-intensive and susceptible to subjectivity, whereas heuristic and automated techniques frequently under-perform in recall unless supplemented by extensive expert input. We introduce a Large Language Model (LLM)-based chained prompt engineering framework for the automated development of search strategies in systematic reviews. The framework replicates the procedural structure of manual search design while leveraging LLMs to decompose review objectives, extract and formalize PICO elements, generate conceptual representations, expand terminologies, and synthesize Boolean queries. In addition to query construction, the framework exhibits superior performance in generating well-structured PICO elements relative to existing methods, thereby strengthening the foundation for high-recall search strategies. Evaluation on a subset of the LEADSInstruct dataset demonstrates that the framework attains a 0.9 average recall. These results significantly exceed the performance of existing approaches. Error analysis further highlights the critical role of precise objective specification and terminological alignment in optimizing retrieval effectiveness. These findings confirm the capacity of LLM-based pipelines to yield transparent, reproducible, and high-performing search strategies, and highlight their potential as scalable instruments for supporting evidence synthesis and evidence-based practice. 

---
# Efficient Multilingual Search Relevance Modeling in E-Commerce via LLM Mixture-of-Experts 

**Authors**: Ye Liu, Xu Chen, Wuji Chen, Mang Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.00003)  

**Abstract**: In e-commerce platforms, search relevance directly influences both user experience and merchant revenue. In multi-country deployments, diverse linguistic, cultural, and product catalog contexts introduce significant distribution shifts, posing substantial challenges to relevance modeling. Existing approaches typically enhance the reasoning or multilingual abilities of a single monolithic model, yet they remain limited by data diversity, coverage gaps, and high inference costs in heterogeneous environments. Our empirical analysis reveals that different LLM base models exhibit complementary strengths across languages and regions, motivating an expert-based architecture. We propose a scalable LLM-based Mixture-of-Experts (MoE) framework that dynamically routes queries to specialized experts and fuses their embeddings through concatenation. Among rule-based, pseudo-label-based, and fully end-to-end strategies, end-to-end hard routing with concatenation offers the best balance of effectiveness and efficiency. To mitigate inference overhead, we further develop an engineering-optimized offline batch pipeline with resource-efficient scheduling, which hides memory latency, improves GPU utilization, and reduces GPU-hour consumption by up to 35% compared with synchronous execution. On datasets spanning six Southeast Asian markets, our MoE improves AUC by 0.72 percentage points over a dense baseline with the same active parameters. Meanwhile, the optimized pipeline achieves 27.6 queries per second (QPS), a 9% throughput improvement. These results demonstrate superior multilingual relevance and efficiency, delivering strong cost-effectiveness for real-world e-commerce search systems. 

---
# Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics 

**Authors**: Ziwen Xu, Chenyan Wu, Hengyu Sun, Haiwen Hong, Mengru Wang, Yunzhi Yao, Longtao Huang, Hui Xue, Shumin Deng, Zhixuan Chu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02343)  

**Abstract**: Methods for controlling large language models (LLMs), including local weight fine-tuning, LoRA-based adaptation, and activation-based interventions, are often studied in isolation, obscuring their connections and making comparison difficult. In this work, we present a unified view that frames these interventions as dynamic weight updates induced by a control signal, placing them within a single conceptual framework. Building on this view, we propose a unified preference-utility analysis that separates control effects into preference, defined as the tendency toward a target concept, and utility, defined as coherent and task-valid generation, and measures both on a shared log-odds scale using polarity-paired contrastive examples. Across methods, we observe a consistent trade-off between preference and utility: stronger control increases preference while predictably reducing utility. We further explain this behavior through an activation manifold perspective, in which control shifts representations along target-concept directions to enhance preference, while utility declines primarily when interventions push representations off the model's valid-generation manifold. Finally, we introduce a new steering approach SPLIT guided by this analysis that improves preference while better preserving utility. Code is available at this https URL. 

---
# AutoBool: An Reinforcement-Learning trained LLM for Effective Automated Boolean Query Generation for Systematic Reviews 

**Authors**: Shuai Wang, Harrisen Scells, Bevan Koopman, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2602.00005)  

**Abstract**: We present AutoBool, a reinforcement learning (RL) framework that trains large language models (LLMs) to generate effective Boolean queries for medical systematic reviews. Boolean queries are the primary mechanism for literature retrieval in this domain and must achieve high recall while maintaining reasonable precision - a challenging balance that existing prompt-based LLM approaches often struggle to achieve. A major limitation in this space is the lack of high-quality ground-truth Boolean queries for each topic, which makes supervised fine-tuning impractical. AutoBool addresses this challenge by using RL to directly optimize query generation with retrieval measures, without requiring target queries. To support this effort, we create and release the largest dataset of its kind: 65588 topics in total for training and evaluating the task of automatic Boolean query formulation. Experiments on our new dataset and two established datasets (CLEF TAR and Seed Collection) show that AutoBool significantly outperforms zero shot/few shot prompting and matches or exceeds the effectiveness of much larger GPT-based models (e.g., GPT-4o, O3) using smaller backbones. It also approaches effectiveness of expert-authored queries while retrieving 10 to 16 times fewer documents. Ablation studies reveal the critical roles of model backbone, size, decoding temperature, and prompt design. Code and data are available at this https URL. 

---
# FDA AI Search: Making FDA-Authorized AI Devices Searchable 

**Authors**: Arun Kavishwar, William Lotter  

**Link**: [PDF](https://arxiv.org/pdf/2602.00006)  

**Abstract**: Over 1,200 AI-enabled medical devices have received marketing authorization from the U.S. FDA, yet identifying devices suited to specific clinical needs remains challenging because the FDA's databases contain only limited metadata and non-searchable summary PDFs. To address this gap, we developed FDA AI Search, a website that enables semantic querying of FDA-authorized AI-enabled devices. The backend includes an embedding-based retrieval system, where LLM-extracted features from authorization summaries are compared to user queries to find relevant matches. We present quantitative and qualitative evaluation that support the effectiveness of the retrieval algorithm compared to keyword-based methods. As FDA-authorized AI devices become increasingly prevalent and their use cases expand, we envision that the tool will assist healthcare providers in identifying devices aligned with their clinical needs and support developers in formulating novel AI applications. 

---
# C$^2$-Cite: Contextual-Aware Citation Generation for Attributed Large Language Models 

**Authors**: Yue Yu, Ting Bai, HengZhi Lan, Li Qian, Li Peng, Jie Wu, Wei Liu, Jian Luan, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2602.00004)  

**Abstract**: The attribution technique enhances the credibility of LLMs by adding citations to the generated sentences, enabling users to trace back to the original sources and verify the reliability of the output. However, existing instruction-tuned attributed LLMs often fail to properly interpret the contextual semantics of citation symbols (e.g., [i]) during text generation. This shortcoming arises from their insufficient awareness of the context information surrounding citation markers, which in turn leads to disjointed references and poor integration of retrieved knowledge into the generated content. To address this issue, we propose a novel \textbf{C}ontextual-aware \textbf{C}itation generation framework (\textbf{C$^2$}-\textbf{Cite}) that explicitly integrates the semantic relationships between citation markers and their referenced content. Specifically, a contextual citation alignment mechanism is adopted: it first encodes the retrieved document contexts into the symbol representation of citations, then aligns the marker numbers by decoding information from a citation router function. This mechanism enables the transformation of citation markers from generic placeholders into active knowledge pointers that link to the referenced source information. Experimental results on the ALCE benchmark across three datasets validate our framework C$^2$-Cite++: it outperforms the SOTA baseline by an average of 5.8\% in citation quality and 17.4\% in response correctness. The implementation is publicly available at this https URL 

---
# Orthogonal Hierarchical Decomposition for Structure-Aware Table Understanding with Large Language Models 

**Authors**: Bin Cao, Huixian Lu, Chenwen Ma, Ting Wang, Ruizhe Li, Jing Fan  

**Link**: [PDF](https://arxiv.org/pdf/2602.01969)  

**Abstract**: Complex tables with multi-level headers, merged cells and heterogeneous layouts pose persistent challenges for LLMs in both understanding and reasoning. Existing approaches typically rely on table linearization or normalized grid modeling. However, these representations struggle to explicitly capture hierarchical structures and cross-dimensional dependencies, which can lead to misalignment between structural semantics and textual representations for non-standard tables. To address this issue, we propose an Orthogonal Hierarchical Decomposition (OHD) framework that constructs structure-preserving input representations of complex tables for LLMs. OHD introduces an Orthogonal Tree Induction (OTI) method based on spatial--semantic co-constraints, which decomposes irregular tables into a column tree and a row tree to capture vertical and horizontal hierarchical dependencies, respectively. Building on this representation, we design a dual-pathway association protocol to symmetrically reconstruct semantic lineage of each cell, and incorporate an LLM as a semantic arbitrator to align multi-level semantic information. We evaluate OHD framework on two complex table question answering benchmarks, AITQA and HiTab. Experimental results show that OHD consistently outperforms existing representation paradigms across multiple evaluation metrics. 

---
# PARSE: An Open-Domain Reasoning Question Answering Benchmark for Persian 

**Authors**: Jamshid Mozafari, Seyed Parsa Mousavinasab, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2602.01246)  

**Abstract**: Reasoning-focused Question Answering (QA) has advanced rapidly with Large Language Models (LLMs), yet high-quality benchmarks for low-resource languages remain scarce. Persian, spoken by roughly 130 million people, lacks a comprehensive open-domain resource for evaluating reasoning-capable QA systems. We introduce PARSE, the first open-domain Persian reasoning QA benchmark, containing 10,800 questions across Boolean, multiple-choice, and factoid formats, with diverse reasoning types, difficulty levels, and answer structures. The benchmark is built via a controlled LLM-based generation pipeline and validated through human evaluation. We also ensure linguistic and factual quality through multi-stage filtering, annotation, and consistency checks. We benchmark multilingual and Persian LLMs under multiple prompting strategies and show that Persian prompts and structured prompting (CoT for Boolean/multiple-choice; few-shot for factoid) improve performance. Fine-tuning further boosts results, especially for Persian-specialized models. These findings highlight how PARSE supports both fair comparison and practical model adaptation. PARSE fills a critical gap in Persian QA research and provides a strong foundation for developing and evaluating reasoning-capable LLMs in low-resource settings. 

---
# LLM-based Embeddings: Attention Values Encode Sentence Semantics Better Than Hidden States 

**Authors**: Yeqin Zhang, Yunfei Wang, Jiaxuan Chen, Ke Qin, Yizheng Zhao, Cam-Tu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2602.01572)  

**Abstract**: Sentence representations are foundational to many Natural Language Processing (NLP) applications. While recent methods leverage Large Language Models (LLMs) to derive sentence representations, most rely on final-layer hidden states, which are optimized for next-token prediction and thus often fail to capture global, sentence-level semantics. This paper introduces a novel perspective, demonstrating that attention value vectors capture sentence semantics more effectively than hidden states. We propose Value Aggregation (VA), a simple method that pools token values across multiple layers and token indices. In a training-free setting, VA outperforms other LLM-based embeddings, even matches or surpasses the ensemble-based MetaEOL. Furthermore, we demonstrate that when paired with suitable prompts, the layer attention outputs can be interpreted as aligned weighted value vectors. Specifically, the attention scores of the last token function as the weights, while the output projection matrix ($W_O$) aligns these weighted value vectors with the common space of the LLM residual stream. This refined method, termed Aligned Weighted VA (AlignedWVA), achieves state-of-the-art performance among training-free LLM-based embeddings, outperforming the high-cost MetaEOL by a substantial margin. Finally, we highlight the potential of obtaining strong LLM embedding models through fine-tuning Value Aggregation. 

---
# Temporal Leakage in Search-Engine Date-Filtered Web Retrieval: A Case Study from Retrospective Forecasting 

**Authors**: Ali El Lahib, Ying-Jieh Xia, Zehan Li, Yuxuan Wang, Xinyu Pi  

**Link**: [PDF](https://arxiv.org/pdf/2602.00758)  

**Abstract**: Search-engine date filters are widely used to enforce pre-cutoff retrieval in retrospective evaluations of search-augmented forecasters. We show this approach is unreliable: auditing Google Search with a before: filter, 71% of questions return at least one page containing strong post-cutoff leakage, and for 41%, at least one page directly reveals the answer. Using a large language model (LLM), gpt-oss-120b, to forecast with these leaky documents, we demonstrate an inflated prediction accuracy (Brier score 0.108 vs. 0.242 with leak-free documents). We characterize common leakage mechanisms, including updated articles, related-content modules, unreliable metadata/timestamps, and absence-based signals, and argue that date-restricted search is insufficient for temporal evaluation. We recommend stronger retrieval safeguards or evaluation on frozen, time-stamped web snapshots to ensure credible retrospective forecasting. 

---
# OGD4All: A Framework for Accessible Interaction with Geospatial Open Government Data Based on Large Language Models 

**Authors**: Michael Siebenmann, Javier Argota Sánchez-Vaquerizo, Stefan Arisona, Krystian Samp, Luis Gisler, Dirk Helbing  

**Link**: [PDF](https://arxiv.org/pdf/2602.00012)  

**Abstract**: We present OGD4All, a transparent, auditable, and reproducible framework based on Large Language Models (LLMs) to enhance citizens' interaction with geospatial Open Government Data (OGD). The system combines semantic data retrieval, agentic reasoning for iterative code generation, and secure sandboxed execution that produces verifiable multimodal outputs. Evaluated on a 199-question benchmark covering both factual and unanswerable questions, across 430 City-of-Zurich datasets and 11 LLMs, OGD4All reaches 98% analytical correctness and 94% recall while reliably rejecting questions unsupported by available data, which minimizes hallucination risks. Statistical robustness tests, as well as expert feedback, show reliability and social relevance. The proposed approach shows how LLMs can provide explainable, multimodal access to public data, advancing trustworthy AI for open governance. 

---
# Culinary Crossroads: A RAG Framework for Enhancing Diversity in Cross-Cultural Recipe Adaptation 

**Authors**: Tianyi Hu, Andrea Morales-Garzón, Jingyi Zheng, Maria Maistro, Daniel Hershcovich  

**Link**: [PDF](https://arxiv.org/pdf/2507.21934)  

**Abstract**: In cross-cultural recipe adaptation, the goal is not only to ensure cultural appropriateness and retain the original dish's essence, but also to provide diverse options for various dietary needs and preferences. Retrieval Augmented Generation (RAG) is a promising approach, combining the retrieval of real recipes from the target cuisine for cultural adaptability with large language models (LLMs) for relevance. However, it remains unclear whether RAG can generate diverse adaptation results. Our analysis shows that RAG tends to overly rely on a limited portion of the context across generations, failing to produce diverse outputs even when provided with varied contextual inputs. This reveals a key limitation of RAG in creative tasks with multiple valid answers: it fails to leverage contextual diversity for generating varied responses. To address this issue, we propose CARRIAGE, a plug-and-play RAG framework for cross-cultural recipe adaptation that enhances diversity in both retrieval and context organization. To our knowledge, this is the first RAG framework that explicitly aims to generate highly diverse outputs to accommodate multiple user preferences. Our experiments show that CARRIAGE achieves Pareto efficiency in terms of diversity and quality of recipe adaptation compared to closed-book LLMs. 

---
# From Prompt to Graph: Comparing LLM-Based Information Extraction Strategies in Domain-Specific Ontology Development 

**Authors**: Xuan Liu, Ziyu Li, Mu He, Ziyang Ma, Xiaoxu Wu, Gizem Yilmaz, Yiyuan Xia, Bingbing Li, He Tan, Jerry Ying Hsi Fuh, Wen Feng Lu, Anders E.W. Jarfors, Per Jansson  

**Link**: [PDF](https://arxiv.org/pdf/2602.00699)  

**Abstract**: Ontologies are essential for structuring domain knowledge, improving accessibility, sharing, and reuse. However, traditional ontology construction relies on manual annotation and conventional natural language processing (NLP) techniques, making the process labour-intensive and costly, especially in specialised fields like casting manufacturing. The rise of Large Language Models (LLMs) offers new possibilities for automating knowledge extraction. This study investigates three LLM-based approaches, including pre-trained LLM-driven method, in-context learning (ICL) method and fine-tuning method to extract terms and relations from domain-specific texts using limited data. We compare their performances and use the best-performing method to build a casting ontology that validated by domian expert. 

---
# PPoGA: Predictive Plan-on-Graph with Action for Knowledge Graph Question Answering 

**Authors**: MinGyu Jeon, SuWan Cho, JaeYoung Shu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00007)  

**Abstract**: Large Language Models (LLMs) augmented with Knowledge Graphs (KGs) have advanced complex question answering, yet they often remain susceptible to failure when their initial high-level reasoning plan is flawed. This limitation, analogous to cognitive functional fixedness, prevents agents from restructuring their approach, leading them to pursue unworkable solutions. To address this, we propose PPoGA (Predictive Plan-on-Graph with Action), a novel KGQA framework inspired by human cognitive control and problem-solving. PPoGA incorporates a Planner-Executor architecture to separate high-level strategy from low-level execution and leverages a Predictive Processing mechanism to anticipate outcomes. The core innovation of our work is a self-correction mechanism that empowers the agent to perform not only Path Correction for local execution errors but also Plan Correction by identifying, discarding, and reformulating the entire plan when it proves ineffective. We conduct extensive experiments on three challenging multi-hop KGQA benchmarks: GrailQA, CWQ, and WebQSP. The results demonstrate that PPoGA achieves state-of-the-art performance, significantly outperforming existing methods. Our work highlights the critical importance of metacognitive abilities like problem restructuring for building more robust and flexible AI reasoning systems. 

---
# ROG: Retrieval-Augmented LLM Reasoning for Complex First-Order Queries over Knowledge Graphs 

**Authors**: Ziyan Zhang, Chao Wang, Zhuo Chen, Chiyi Li, Kai Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.02382)  

**Abstract**: Answering first-order logic (FOL) queries over incomplete knowledge graphs (KGs) is difficult, especially for complex query structures that compose projection, intersection, union, and negation. We propose ROG, a retrieval-augmented framework that combines query-aware neighborhood retrieval with large language model (LLM) chain-of-thought reasoning. ROG decomposes a multi-operator query into a sequence of single-operator sub-queries and grounds each step in compact, query-relevant neighborhood evidence. Intermediate answer sets are cached and reused across steps, improving consistency on deep reasoning chains. This design reduces compounding errors and yields more robust inference on complex and negation-heavy queries. Overall, ROG provides a practical alternative to embedding-based logical reasoning by replacing learned operators with retrieval-grounded, step-wise inference. Experiments on standard KG reasoning benchmarks show consistent gains over strong embedding-based baselines, with the largest improvements on high-complexity and negation-heavy query types. 

---
# MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents 

**Authors**: Haozhen Zhang, Quanyu Long, Jianzhu Bao, Tao Feng, Weizhi Zhang, Haodong Yue, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02474)  

**Abstract**: Most Large Language Model (LLM) agent memory systems rely on a small set of static, hand-designed operations for extracting memory. These fixed procedures hard-code human priors about what to store and how to revise memory, making them rigid under diverse interaction patterns and inefficient on long histories. To this end, we present \textbf{MemSkill}, which reframes these operations as learnable and evolvable memory skills, structured and reusable routines for extracting, consolidating, and pruning information from interaction traces. Inspired by the design philosophy of agent skills, MemSkill employs a \emph{controller} that learns to select a small set of relevant skills, paired with an LLM-based \emph{executor} that produces skill-guided memories. Beyond learning skill selection, MemSkill introduces a \emph{designer} that periodically reviews hard cases where selected skills yield incorrect or incomplete memories, and evolves the skill set by proposing refinements and new skills. Together, MemSkill forms a closed-loop procedure that improves both the skill-selection policy and the skill set itself. Experiments on LoCoMo, LongMemEval, HotpotQA, and ALFWorld demonstrate that MemSkill improves task performance over strong baselines and generalizes well across settings. Further analyses shed light on how skills evolve, offering insights toward more adaptive, self-evolving memory management for LLM agents. 

---
# Abstract Activation Spaces for Content-Invariant Reasoning in Large Language Models 

**Authors**: Gabriele Maraia, Marco Valentino, Fabio Massimo Zanzotto, Leonardo Ranaldi  

**Link**: [PDF](https://arxiv.org/pdf/2602.02462)  

**Abstract**: Large Language Models (LLMs) often struggle with deductive judgment in syllogistic reasoning, systematically conflating semantic plausibility with formal validity a phenomenon known as content effect. This bias persists even when models generate step-wise explanations, indicating that intermediate rationales may inherit the same semantic shortcuts that affect answers. Recent approaches propose mitigating this issue by increasing inference-time structural constraints, either by encouraging abstract intermediate representations or by intervening directly in the model's internal computations; however, reliably suppressing semantic interference remains an open challenge. To make formal deduction less sensitive to semantic content, we introduce a framework for abstraction-guided reasoning that explicitly separates structural inference from lexical semantics. We construct paired content-laden and abstract syllogisms and use the model's activations on abstract inputs to define an abstract reasoning space. We then learn lightweight Abstractors that, from content-conditioned residual-stream states, predict representations aligned with this space and integrate these predictions via multi-layer interventions during the forward pass. Using cross-lingual transfer as a test bed, we show that abstraction-aligned steering reduces content-driven errors and improves validity-sensitive performance. Our results position activation-level abstraction as a scalable mechanism for enhancing the robustness of formal reasoning in LLMs against semantic interference. 

---
# Misconception Diagnosis From Student-Tutor Dialogue: Generate, Retrieve, Rerank 

**Authors**: Joshua Mitton, Prarthana Bhattacharyya, Digory Smith, Thomas Christie, Ralph Abboud, Simon Woodhead  

**Link**: [PDF](https://arxiv.org/pdf/2602.02414)  

**Abstract**: Timely and accurate identification of student misconceptions is key to improving learning outcomes and pre-empting the compounding of student errors. However, this task is highly dependent on the effort and intuition of the teacher. In this work, we present a novel approach for detecting misconceptions from student-tutor dialogues using large language models (LLMs). First, we use a fine-tuned LLM to generate plausible misconceptions, and then retrieve the most promising candidates among these using embedding similarity with the input dialogue. These candidates are then assessed and re-ranked by another fine-tuned LLM to improve misconception relevance. Empirically, we evaluate our system on real dialogues from an educational tutoring platform. We consider multiple base LLM models including LLaMA, Qwen and Claude on zero-shot and fine-tuned settings. We find that our approach improves predictive performance over baseline models and that fine-tuning improves both generated misconception quality and can outperform larger closed-source models. Finally, we conduct ablation studies to both validate the importance of our generation and reranking steps on misconception generation quality. 

---
# Training LLMs for Divide-and-Conquer Reasoning Elevates Test-Time Scalability 

**Authors**: Xiao Liang, Zhong-Zhi Li, Zhenghao Lin, Eric Hancheng Jiang, Hengyuan Zhang, Yelong Shen, Kai-Wei Chang, Ying Nian Wu, Yeyun Gong, Weizhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.02477)  

**Abstract**: Large language models (LLMs) have demonstrated strong reasoning capabilities through step-by-step chain-of-thought (CoT) reasoning. Nevertheless, at the limits of model capability, CoT often proves insufficient, and its strictly sequential nature constrains test-time scalability. A potential alternative is divide-and-conquer (DAC) reasoning, which decomposes a complex problem into subproblems to facilitate more effective exploration of the solution. Although promising, our analysis reveals a fundamental misalignment between general-purpose post-training and DAC-style inference, which limits the model's capacity to fully leverage this potential. To bridge this gap and fully unlock LLMs' reasoning capabilities on the most challenging tasks, we propose an end-to-end reinforcement learning (RL) framework to enhance their DAC-style reasoning capacity. At each step, the policy decomposes a problem into a group of subproblems, solves them sequentially, and addresses the original one conditioned on the subproblem solutions, with both decomposition and solution integrated into RL training. Under comparable training, our DAC-style framework endows the model with a higher performance ceiling and stronger test-time scalability, surpassing CoT by 8.6% in Pass@1 and 6.3% in Pass@32 on competition-level benchmarks. 

---
# Large Language Models for Mental Health: A Multilingual Evaluation 

**Authors**: Nishat Raihan, Sadiya Sayara Chowdhury Puspo, Ana-Maria Bucur, Stevie Chancellor, Marcos Zampieri  

**Link**: [PDF](https://arxiv.org/pdf/2602.02440)  

**Abstract**: Large Language Models (LLMs) have remarkable capabilities across NLP tasks. However, their performance in multilingual contexts, especially within the mental health domain, has not been thoroughly explored. In this paper, we evaluate proprietary and open-source LLMs on eight mental health datasets in various languages, as well as their machine-translated (MT) counterparts. We compare LLM performance in zero-shot, few-shot, and fine-tuned settings against conventional NLP baselines that do not employ LLMs. In addition, we assess translation quality across language families and typologies to understand its influence on LLM performance. Proprietary LLMs and fine-tuned open-source LLMs achieve competitive F1 scores on several datasets, often surpassing state-of-the-art results. However, performance on MT data is generally lower, and the extent of this decline varies by language and typology. This variation highlights both the strengths of LLMs in handling mental health tasks in languages other than English and their limitations when translation quality introduces structural or lexical mismatches. 

---
# Automated Multiple Mini Interview (MMI) Scoring 

**Authors**: Ryan Huynh, Frank Guerin, Alison Callwood  

**Link**: [PDF](https://arxiv.org/pdf/2602.02360)  

**Abstract**: Assessing soft skills such as empathy, ethical judgment, and communication is essential in competitive selection processes, yet human scoring is often inconsistent and biased. While Large Language Models (LLMs) have improved Automated Essay Scoring (AES), we show that state-of-the-art rationale-based fine-tuning methods struggle with the abstract, context-dependent nature of Multiple Mini-Interviews (MMIs), missing the implicit signals embedded in candidate narratives. We introduce a multi-agent prompting framework that breaks down the evaluation process into transcript refinement and criterion-specific scoring. Using 3-shot in-context learning with a large instruct-tuned model, our approach outperforms specialised fine-tuned baselines (Avg QWK 0.62 vs 0.32) and achieves reliability comparable to human experts. We further demonstrate the generalisability of our framework on the ASAP benchmark, where it rivals domain-specific state-of-the-art models without additional training. These findings suggest that for complex, subjective reasoning tasks, structured prompt engineering may offer a scalable alternative to data-intensive fine-tuning, altering how LLMs can be applied to automated assessment. 

---
# Proof-RM: A Scalable and Generalizable Reward Model for Math Proof 

**Authors**: Haotong Yang, Zitong Wang, Shijia Kang, Siqi Yang, Wenkai Yu, Xu Niu, Yike Sun, Yi Hu, Zhouchen Lin, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02377)  

**Abstract**: While Large Language Models (LLMs) have demonstrated strong math reasoning abilities through Reinforcement Learning with *Verifiable Rewards* (RLVR), many advanced mathematical problems are proof-based, with no guaranteed way to determine the authenticity of a proof by simple answer matching. To enable automatic verification, a Reward Model (RM) capable of reliably evaluating full proof processes is required. In this work, we design a *scalable* data-construction pipeline that, with minimal human effort, leverages LLMs to generate a large quantity of high-quality "**question-proof-check**" triplet data. By systematically varying problem sources, generation methods, and model configurations, we create diverse problem-proof pairs spanning multiple difficulty levels, linguistic styles, and error types, subsequently filtered through hierarchical human review for label alignment. Utilizing these data, we train a proof-checking RM, incorporating additional process reward and token weight balance to stabilize the RL process. Our experiments validate the model's scalability and strong performance from multiple perspectives, including reward accuracy, generalization ability and test-time guidance, providing important practical recipes and tools for strengthening LLM mathematical capabilities. 

---
# Cross-Lingual Stability of LLM Judges Under Controlled Generation: Evidence from Finno-Ugric Languages 

**Authors**: Isaac Chung, Linda Freienthal  

**Link**: [PDF](https://arxiv.org/pdf/2602.02287)  

**Abstract**: Cross-lingual evaluation of large language models (LLMs) typically conflates two sources of variance: genuine model performance differences and measurement instability. We investigate evaluation reliability by holding generation conditions constant while varying target language. Using synthetic customer-support dialogues generated with identical parameters across Estonian, Finnish, and Hungarian, we test whether automatic metrics and LLM-as-a-judge scoring produce stable model rankings across these morphologically rich, related Finno-Ugric languages. With a small set of Estonian native speaker annotations as a reference point, we find systematic ranking instabilities: surface-level metrics (lexical diversity, surface and semantic similarity) maintain cross-language stability, but pragmatic judgments (coherence, instruction-following) exhibit rank inversions and near-zero correlations. Because generation is controlled, these inconsistencies reflect how judge scoring behaves differently across languages rather than true model differences.
This controlled design provides a diagnostic probe: evaluation methods that fail to maintain stability under identical generation conditions signal transfer failure before deployment. Our findings suggest that zero-shot judge transfer is unreliable for discourse-level assessment in morphologically rich languages, motivating language-specific calibration against targeted human baselines. We release our controlled generation protocol, synthetic data, and evaluation framework to enable replication across language families at this https URL. 

---
# OpenSeal: Good, Fast, and Cheap Construction of an Open-Source Southeast Asian LLM via Parallel Data 

**Authors**: Tan Sang Nguyen, Muhammad Reza Qorib, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2602.02266)  

**Abstract**: Large language models (LLMs) have proven to be effective tools for a wide range of natural language processing (NLP) applications. Although many LLMs are multilingual, most remain English-centric and perform poorly on low-resource languages. Recently, several Southeast Asia-focused LLMs have been developed, but none are truly open source, as they do not publicly disclose their training data. Truly open-source models are important for transparency and for enabling a deeper and more precise understanding of LLM internals and development, including biases, generalization, and multilinguality. Motivated by recent advances demonstrating the effectiveness of parallel data in improving multilingual performance, we conduct controlled and comprehensive experiments to study the effectiveness of parallel data in continual pretraining of LLMs. Our findings show that using only parallel data is the most effective way to extend an LLM to new languages. Using just 34.7B tokens of parallel data and 180 hours on 8x NVIDIA H200 GPUs, we built OpenSeal, the first truly open Southeast Asian LLM that rivals the performance of existing models of similar size. 

---
# Am I More Pointwise or Pairwise? Revealing Position Bias in Rubric-Based LLM-as-a-Judge 

**Authors**: Yuzheng Xu, Tosho Hirasawa, Tadashi Kozuno, Yoshitaka Ushiku  

**Link**: [PDF](https://arxiv.org/pdf/2602.02219)  

**Abstract**: Large language models (LLMs) are now widely used to evaluate the quality of text, a field commonly referred to as LLM-as-a-judge. While prior works mainly focus on point-wise and pair-wise evaluation paradigms. Rubric-based evaluation, where LLMs select a score from multiple rubrics, has received less analysis. In this work, we show that rubric-based evaluation implicitly resembles a multi-choice setting and therefore has position bias: LLMs prefer score options appearing at specific positions in the rubric list. Through controlled experiments across multiple models and datasets, we demonstrate consistent position bias. To mitigate this bias, we propose a balanced permutation strategy that evenly distributes each score option across positions. We show that aggregating scores across balanced permutations not only reveals latent position bias, but also improves correlation between the LLM-as-a-Judge and human. Our results suggest that rubric-based LLM-as-a-Judge is not inherently point-wise and that simple permutation-based calibration can substantially improve its reliability. 

---
# Evaluating Metalinguistic Knowledge in Large Language Models across the World's Languages 

**Authors**: Tjaša Arčon, Matej Klemen, Marko Robnik-Šikonja, Kaja Dobrovoljc  

**Link**: [PDF](https://arxiv.org/pdf/2602.02182)  

**Abstract**: Large language models (LLMs) are routinely evaluated on language use tasks, yet their knowledge of linguistic structure remains poorly understood. Existing linguistic benchmarks typically focus on narrow phenomena, emphasize high-resource languages, and rarely evaluate metalinguistic knowledge-explicit reasoning about language structure rather than language use. Using accuracy and macro F1, together with majority-class and chance baselines, we analyse overall performance and examine variation by linguistic domains and language-related factors. Our results show that metalinguistic knowledge in current LLMs is limited: GPT-4o performs best but achieves only moderate accuracy (0.367), while open-source models lag behind. All models perform above chance but fail to outperform the majority-class baseline, suggesting they capture cross-linguistic patterns but lack fine-grained grammatical distinctions. Performance varies across linguistic domains, with lexical features showing the highest accuracy and phonological features among the lowest, partially reflecting differences in online visibility. At the language level, accuracy shows a strong association with digital language status: languages with higher digital presence and resource availability are evaluated more accurately, while low-resource languages show substantially lower performance. Analyses of predictive factors confirm that resource-related indicators (Wikipedia size, corpus availability) are more informative predictors of accuracy than geographical, genealogical, or sociolinguistic factors. Together, these results suggest that LLMs' metalinguistic knowledge is fragmented and shaped by data availability rather than generalizable grammatical competence across the world's languages. We release our benchmark as an open-source dataset to support systematic evaluation and encourage greater global linguistic diversity in future LLMs. 

---
# AR-MAP: Are Autoregressive Large Language Models Implicit Teachers for Diffusion Large Language Models? 

**Authors**: Liang Lin, Feng Xiong, Zengbin Wang, Kun Wang, Junhao Dong, Xuecai Hu, Yong Wang, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2602.02178)  

**Abstract**: Diffusion Large Language Models (DLLMs) have emerged as a powerful alternative to autoregressive models, enabling parallel token generation across multiple positions. However, preference alignment of DLLMs remains challenging due to high variance introduced by Evidence Lower Bound (ELBO)-based likelihood estimation. In this work, we propose AR-MAP, a novel transfer learning framework that leverages preference-aligned autoregressive LLMs (AR-LLMs) as implicit teachers for DLLM alignment. We reveal that DLLMs can effectively absorb alignment knowledge from AR-LLMs through simple weight scaling, exploiting the shared architectural structure between these divergent generation paradigms. Crucially, our approach circumvents the high variance and computational overhead of direct DLLM alignment and comprehensive experiments across diverse preference alignment tasks demonstrate that AR-MAP achieves competitive or superior performance compared to existing DLLM-specific alignment methods, achieving 69.08\% average score across all tasks and models. Our Code is available at this https URL. 

---
# Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference via Confidence-Guided Context Focusing 

**Authors**: Lingkun Long, Yushi Huang, Shihao Bai, Ruihao Gong, Jun Zhang, Ao Zhou, Jianlei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02159)  

**Abstract**: Diffusion Large Language Models (dLLMs) deliver strong long-context processing capability in a non-autoregressive decoding paradigm. However, the considerable computational cost of bidirectional full attention limits the inference efficiency. Although sparse attention is promising, existing methods remain ineffective. This stems from the need to estimate attention importance for tokens yet to be decoded, while the unmasked token positions are unknown during diffusion. In this paper, we present Focus-dLLM, a novel training-free attention sparsification framework tailored for accurate and efficient long-context dLLM inference. Based on the finding that token confidence strongly correlates across adjacent steps, we first design a past confidence-guided indicator to predict unmasked regions. Built upon this, we propose a sink-aware pruning strategy to accurately estimate and remove redundant attention computation, while preserving highly influential attention sinks. To further reduce overhead, this strategy reuses identified sink locations across layers, leveraging the observed cross-layer consistency. Experimental results show that our method offers more than $29\times$ lossless speedup under $32K$ context length. The code is publicly available at: this https URL 

---
# Dicta-LM 3.0: Advancing The Frontier of Hebrew Sovereign LLMs 

**Authors**: Shaltiel Shmidman, Avi Shmidman, Amir DN Cohen, Moshe Koppel  

**Link**: [PDF](https://arxiv.org/pdf/2602.02104)  

**Abstract**: Open-weight LLMs have been released by frontier labs; however, sovereign Large Language Models (for languages other than English) remain low in supply yet high in demand. Training large language models (LLMs) for low-resource languages such as Hebrew poses unique challenges. In this paper, we introduce Dicta-LM 3.0: an open-weight collection of LLMs trained on substantially-sized corpora of Hebrew and English texts. The model is released in three sizes: 24B - adapted from the Mistral-Small-3.1 base model, 12B - adapted from the NVIDIA Nemotron Nano V2 model, and 1.7B - adapted from the Qwen3-1.7B base model. We are releasing multiple variants of each model, each with a native context length of 65k tokens; base model and chat model with tool-calling support. To rigorously evaluate our models, we introduce a new benchmark suite for evaluation of Hebrew chat-LLMs, covering a diverse set of tasks including Translation, Summarization, Winograd, Israeli Trivia, and Diacritization (nikud). Our work not only addresses the intricacies of training LLMs in low-resource languages but also proposes a framework that can be leveraged for adapting other LLMs to various non-English languages, contributing to the broader field of multilingual NLP. 

---
# From Latent Signals to Reflection Behavior: Tracing Meta-Cognitive Activation Trajectory in R1-Style LLMs 

**Authors**: Yanrui Du, Yibo Gao, Sendong Zhao, Jiayun Li, Haochun Wang, Qika Lin, Kai He, Bing Qin, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01999)  

**Abstract**: R1-style LLMs have attracted growing attention for their capacity for self-reflection, yet the internal mechanisms underlying such behavior remain unclear. To bridge this gap, we anchor on the onset of reflection behavior and trace its layer-wise activation trajectory. Using the logit lens to read out token-level semantics, we uncover a structured progression: (i) Latent-control layers, where an approximate linear direction encodes the semantics of thinking budget; (ii) Semantic-pivot layers, where discourse-level cues, including turning-point and summarization cues, surface and dominate the probability mass; and (iii) Behavior-overt layers, where the likelihood of reflection-behavior tokens begins to rise until they become highly likely to be sampled. Moreover, our targeted interventions uncover a causal chain across these stages: prompt-level semantics modulate the projection of activations along latent-control directions, thereby inducing competition between turning-point and summarization cues in semantic-pivot layers, which in turn regulates the sampling likelihood of reflection-behavior tokens in behavior-overt layers. Collectively, our findings suggest a human-like meta-cognitive process-progressing from latent monitoring, to discourse-level regulation, and to finally overt self-reflection. Our analysis code can be found at this https URL. 

---
# LEC-KG: An LLM-Embedding Collaborative Framework for Domain-Specific Knowledge Graph Construction -- A Case Study on SDGs 

**Authors**: Yikai Zeng, Yingchao Piao, Jianhui Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.02090)  

**Abstract**: Constructing domain-specific knowledge graphs from unstructured text remains challenging due to heterogeneous entity mentions, long-tail relation distributions, and the absence of standardized schemas. We present LEC-KG, a bidirectional collaborative framework that integrates the semantic understanding of Large Language Models (LLMs) with the structural reasoning of Knowledge Graph Embeddings (KGE). Our approach features three key components: (1) hierarchical coarse-to-fine relation extraction that mitigates long-tail bias, (2) evidence-guided Chain-of-Thought feedback that grounds structural suggestions in source text, and (3) semantic initialization that enables structural validation for unseen entities. The two modules enhance each other iteratively-KGE provides structure-aware feedback to refine LLM extractions, while validated triples progressively improve KGE representations. We evaluate LEC-KG on Chinese Sustainable Development Goal (SDG) reports, demonstrating substantial improvements over LLM baselines, particularly on low-frequency relations. Through iterative refinement, our framework reliably transforms unstructured policy text into validated knowledge graph triples. 

---
# S3-CoT: Self-Sampled Succinct Reasoning Enables Efficient Chain-of-Thought LLMs 

**Authors**: Yanrui Du, Sendong Zhao, Yibo Gao, Danyang Zhao, Qika Lin, Ming Ma, Jiayun Li, Yi Jiang, Kai He, Qianyi Xu, Bing Qin, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01982)  

**Abstract**: Large language models (LLMs) equipped with chain-of-thought (CoT) achieve strong performance and offer a window into LLM behavior. However, recent evidence suggests that improvements in CoT capabilities often come with redundant reasoning processes, motivating a key question: Can LLMs acquire a fast-thinking mode analogous to human System 1 reasoning? To explore this, our study presents a self-sampling framework based on activation steering for efficient CoT learning. Our method can induce style-aligned and variable-length reasoning traces from target LLMs themselves without any teacher guidance, thereby alleviating a central bottleneck of SFT-based methods-the scarcity of high-quality supervision data. Using filtered data by gold answers, we perform SFT for efficient CoT learning with (i) a human-like dual-cognitive system, and (ii) a progressive compression curriculum. Furthermore, we explore a self-evolution regime in which SFT is driven solely by prediction-consistent data of variable-length variants, eliminating the need for gold answers. Extensive experiments on math benchmarks, together with cross-domain generalization tests in medicine, show that our method yields stable improvements for both general and R1-style LLMs. Our data and model checkpoints can be found at this https URL. 

---
# Indications of Belief-Guided Agency and Meta-Cognitive Monitoring in Large Language Models 

**Authors**: Noam Steinmetz Yalon, Ariel Goldstein, Liad Mudrik, Mor Geva  

**Link**: [PDF](https://arxiv.org/pdf/2602.02467)  

**Abstract**: Rapid advancements in large language models (LLMs) have sparked the question whether these models possess some form of consciousness. To tackle this challenge, Butlin et al. (2023) introduced a list of indicators for consciousness in artificial systems based on neuroscientific theories. In this work, we evaluate a key indicator from this list, called HOT-3, which tests for agency guided by a general belief-formation and action selection system that updates beliefs based on meta-cognitive monitoring. We view beliefs as representations in the model's latent space that emerge in response to a given input, and introduce a metric to quantify their dominance during generation. Analyzing the dynamics between competing beliefs across models and tasks reveals three key findings: (1) external manipulations systematically modulate internal belief formation, (2) belief formation causally drives the model's action selection, and (3) models can monitor and report their own belief states. Together, these results provide empirical support for the existence of belief-guided agency and meta-cognitive monitoring in LLMs. More broadly, our work lays methodological groundwork for investigating the emergence of agency, beliefs, and meta-cognition in LLMs. 

---
# From Code-Centric to Concept-Centric: Teaching NLP with LLM-Assisted "Vibe Coding" 

**Authors**: Hend Al-Khalifa  

**Link**: [PDF](https://arxiv.org/pdf/2602.01919)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) presents both challenges and opportunities for Natural Language Processing (NLP) education. This paper introduces ``Vibe Coding,'' a pedagogical approach that leverages LLMs as coding assistants while maintaining focus on conceptual understanding and critical thinking. We describe the implementation of this approach in a senior-level undergraduate NLP course, where students completed seven labs using LLMs for code generation while being assessed primarily on conceptual understanding through critical reflection questions. Analysis of end-of-course feedback from 19 students reveals high satisfaction (mean scores 4.4-4.6/5.0) across engagement, conceptual learning, and assessment fairness. Students particularly valued the reduced cognitive load from debugging, enabling deeper focus on NLP concepts. However, challenges emerged around time constraints, LLM output verification, and the need for clearer task specifications. Our findings suggest that when properly structured with mandatory prompt logging and reflection-based assessment, LLM-assisted learning can shift focus from syntactic fluency to conceptual mastery, preparing students for an AI-augmented professional landscape. 

---
# Hallucination or Creativity: How to Evaluate AI-Generated Scientific Stories? 

**Authors**: Alex Argese, Pasquale Lisena, Raphaël Troncy  

**Link**: [PDF](https://arxiv.org/pdf/2602.02290)  

**Abstract**: Generative AI can turn scientific articles into narratives for diverse audiences, but evaluating these stories remains challenging. Storytelling demands abstraction, simplification, and pedagogical creativity-qualities that are not often well-captured by standard summarization metrics. Meanwhile, factual hallucinations are critical in scientific contexts, yet, detectors often misclassify legitimate narrative reformulations or prove unstable when creativity is involved. In this work, we propose StoryScore, a composite metric for evaluating AI-generated scientific stories. StoryScore integrates semantic alignment, lexical grounding, narrative control, structural fidelity, redundancy avoidance, and entity-level hallucination detection into a unified framework. Our analysis also reveals why many hallucination detection methods fail to distinguish pedagogical creativity from factual errors, highlighting a key limitation: while automatic metrics can effectively assess semantic similarity with original content, they struggle to evaluate how it is narrated and controlled. 

---
# PretrainRL: Alleviating Factuality Hallucination of Large Language Models at the Beginning 

**Authors**: Langming Liu, Kangtao Lv, Haibin Chen, Weidong Zhang, Yejing Wang, Shilei Liu, Xin Tong, Yujin Yuan, Yongwei Wang, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01875)  

**Abstract**: Large language models (LLMs), despite their powerful capabilities, suffer from factual hallucinations where they generate verifiable falsehoods. We identify a root of this issue: the imbalanced data distribution in the pretraining corpus, which leads to a state of "low-probability truth" and "high-probability falsehood". Recent approaches, such as teaching models to say "I don't know" or post-hoc knowledge editing, either evade the problem or face catastrophic forgetting. To address this issue from its root, we propose \textbf{PretrainRL}, a novel framework that integrates reinforcement learning into the pretraining phase to consolidate factual knowledge. The core principle of PretrainRL is "\textbf{debiasing then learning}." It actively reshapes the model's probability distribution by down-weighting high-probability falsehoods, thereby making "room" for low-probability truths to be learned effectively. To enable this, we design an efficient negative sampling strategy to discover these high-probability falsehoods and introduce novel metrics to evaluate the model's probabilistic state concerning factual knowledge. Extensive experiments on three public benchmarks demonstrate that PretrainRL significantly alleviates factual hallucinations and outperforms state-of-the-art methods. 

---
# Advancing General-Purpose Reasoning Models with Modular Gradient Surgery 

**Authors**: Min Cai, Yu Liang, Longzheng Wang, Yan Wang, Yueyang Zhang, Long Xia, Zhiyuan Sun, Xi Ye, Daiting Shi  

**Link**: [PDF](https://arxiv.org/pdf/2602.02301)  

**Abstract**: Reinforcement learning (RL) has played a central role in recent advances in large reasoning models (LRMs), yielding strong gains in verifiable and open-ended reasoning. However, training a single general-purpose LRM across diverse domains remains challenging due to pronounced domain heterogeneity. Through a systematic study of two widely used strategies, Sequential RL and Mixed RL, we find that both incur substantial cross-domain interference at the behavioral and gradient levels, resulting in limited overall gains. To address these challenges, we introduce **M**odular **G**radient **S**urgery (**MGS**), which resolves gradient conflicts at the module level within the transformer. When applied to Llama and Qwen models, MGS achieves average improvements of 4.3 (16.6\%) and 4.5 (11.1\%) points, respectively, over standard multi-task RL across three representative domains (math, general chat, and instruction following). Further analysis demonstrates that MGS remains effective under prolonged training. Overall, our study clarifies the sources of interference in multi-domain RL and presents an effective solution for training general-purpose LRMs. 

---
# AXE: Low-Cost Cross-Domain Web Structured Information Extraction 

**Authors**: Abdelrahman Mansour, Khaled W. Alshaer, Moataz Elsaban  

**Link**: [PDF](https://arxiv.org/pdf/2602.01838)  

**Abstract**: Extracting structured data from the web is often a trade-off between the brittle nature of manual heuristics and the prohibitive cost of Large Language Models. We introduce AXE (Adaptive X-Path Extractor), a pipeline that rethinks this process by treating the HTML DOM as a tree that needs pruning rather than just a wall of text to be read. AXE uses a specialized "pruning" mechanism to strip away boilerplate and irrelevant nodes, leaving behind a distilled, high-density context that allows a tiny 0.6B LLM to generate precise, structured outputs. To keep the model honest, we implement Grounded XPath Resolution (GXR), ensuring every extraction is physically traceable to a source node. Despite its low footprint, AXE achieves state-of-the-art zero-shot performance, outperforming several much larger, fully-trained alternatives with an F1 score of 88.1% on the SWDE dataset. By releasing our specialized adaptors, we aim to provide a practical, cost-effective path for large-scale web information extraction. 

---
# WorldCup Sampling for Multi-bit LLM Watermarking 

**Authors**: Yidan Wang, Yubing Ren, Yanan Cao, Li Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.01752)  

**Abstract**: As large language models (LLMs) generate increasingly human-like text, watermarking offers a promising solution for reliable attribution beyond mere detection. While multi-bit watermarking enables richer provenance encoding, existing methods largely extend zero-bit schemes through seed-driven steering, leading to indirect information flow, limited effective capacity, and suboptimal decoding. In this paper, we propose WorldCup, a multi-bit watermarking framework for LLMs that treats sampling as a natural communication channel and embeds message bits directly into token selection via a hierarchical competition mechanism guided by complementary signals. Moreover, WorldCup further adopts entropy-aware modulation to preserve generation quality and supports robust message recovery through confidence-aware decoding. Comprehensive experiments show that WorldCup achieves a strong balance across capacity, detectability, robustness, text quality, and decoding efficiency, consistently outperforming prior baselines and laying a solid foundation for future LLM watermarking studies. 

---
# Read As Human: Compressing Context via Parallelizable Close Reading and Skimming 

**Authors**: Jiwei Tang, Shilei Liu, Zhicheng Zhang, Qingsong Lv, Runsong Zhao, Tingwei Lu, Langming Liu, Haibin Chen, Yujin Yuan, Hai-Tao Zheng, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01840)  

**Abstract**: Large Language Models (LLMs) demonstrate exceptional capability across diverse tasks. However, their deployment in long-context scenarios is hindered by two challenges: computational inefficiency and redundant information. We propose RAM (Read As HuMan), a context compression framework that adopts an adaptive hybrid reading strategy, to address these challenges. Inspired by human reading behavior (i.e., close reading important content while skimming less relevant content), RAM partitions the context into segments and encodes them with the input query in parallel. High-relevance segments are fully retained (close reading), while low-relevance ones are query-guided compressed into compact summary vectors (skimming). Both explicit textual segments and implicit summary vectors are concatenated and fed into decoder to achieve both superior performance and natural language format interpretability. To refine the decision boundary between close reading and skimming, we further introduce a contrastive learning objective based on positive and negative query-segment pairs. Experiments demonstrate that RAM outperforms existing baselines on multiple question answering and summarization benchmarks across two backbones, while delivering up to a 12x end-to-end speedup on long inputs (average length 16K; maximum length 32K). 

---
# Data Distribution Matters: A Data-Centric Perspective on Context Compression for Large Language Model 

**Authors**: Kangtao Lv, Jiwei Tang, Langming Liu, Haibin Chen, Weidong Zhang, Shilei Liu, Yongwei Wang, Yujin Yuan, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01778)  

**Abstract**: The deployment of Large Language Models (LLMs) in long-context scenarios is hindered by computational inefficiency and significant information redundancy. Although recent advancements have widely adopted context compression to address these challenges, existing research only focus on model-side improvements, the impact of the data distribution itself on context compression remains largely unexplored. To bridge this gap, we are the first to adopt a data-centric perspective to systematically investigate how data distribution impacts compression quality, including two dimensions: input data and intrinsic data (i.e., the model's internal pretrained knowledge). We evaluate the semantic integrity of compressed representations using an autoencoder-based framework to systematically investigate it. Our experimental results reveal that: (1) encoder-measured input entropy negatively correlates with compression quality, while decoder-measured entropy shows no significant relationship under a frozen-decoder setting; and (2) the gap between intrinsic data of the encoder and decoder significantly diminishes compression gains, which is hard to mitigate. Based on these findings, we further present practical guidelines to optimize compression gains. 

---
# <SOG_k>: One LLM Token for Explicit Graph Structural Understanding 

**Authors**: Jingyao Wu, Bin Lu, Zijun Di, Xiaoying Gan, Meng Jin, Luoyi Fu, Xinbing Wang, Chenghu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.01771)  

**Abstract**: Large language models show great potential in unstructured data understanding, but still face significant challenges with graphs due to their structural hallucination. Existing approaches mainly either verbalize graphs into natural language, which leads to excessive token consumption and scattered attention, or transform graphs into trainable continuous embeddings (i.e., soft prompt), but exhibit severe misalignment with original text tokens. To solve this problem, we propose to incorporate one special token <SOG_k> to fully represent the Structure Of Graph within a unified token space, facilitating explicit topology input and structural information sharing. Specifically, we propose a topology-aware structural tokenizer that maps each graph topology into a highly selective single token. Afterwards, we construct a set of hybrid structure Question-Answering corpora to align new structural tokens with existing text tokens. With this approach, <SOG_k> empowers LLMs to understand, generate, and reason in a concise and accurate manner. Extensive experiments on five graph-level benchmarks demonstrate the superiority of our method, achieving a performance improvement of 9.9% to 41.4% compared to the baselines while exhibiting interpretability and consistency. Furthermore, our method provides a flexible extension to node-level tasks, enabling both global and local structural understanding. The codebase is publicly available at this https URL. 

---
# ARTIS: Agentic Risk-Aware Test-Time Scaling via Iterative Simulation 

**Authors**: Xingshan Zeng, Lingzhi Wang, Weiwen Liu, Liangyou Li, Yasheng Wang, Lifeng Shang, Xin Jiang, Qun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.01709)  

**Abstract**: Current test-time scaling (TTS) techniques enhance large language model (LLM) performance by allocating additional computation at inference time, yet they remain insufficient for agentic settings, where actions directly interact with external environments and their effects can be irreversible and costly. We propose \emph{\name}, \emph{\underline{A}gentic \underline{R}isk-Aware \underline{T}est-Time Scaling via \underline{I}terative \underline{S}imulation}, a framework that decouples exploration from commitment by enabling test-time exploration through simulated interactions prior to real-world execution. This design allows extending inference-time computation to improve action-level reliability and robustness without incurring environmental risk. We further show that naive LLM-based simulators struggle to capture rare but high-impact failure modes, substantially limiting their effectiveness for agentic decision making. To address this limitation, we introduce a \emph{risk-aware tool simulator} that emphasizes fidelity on failure-inducing actions via targeted data generation and rebalanced training. Experiments on multi-turn and multi-step agentic benchmarks demonstrate that iterative simulation substantially improves agent reliability, and that risk-aware simulation is essential for consistently realizing these gains across models and tasks. 

---
# COMI: Coarse-to-fine Context Compression via Marginal Information Gain 

**Authors**: Jiwei Tang, Shilei Liu, Zhicheng Zhang, Yujin Yuan, Libin Zheng, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01719)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse tasks. However, their deployment in long context scenarios remains hindered by computational inefficiency and information redundancy. Context compression methods address these challenges by significantly reducing input length and eliminating redundancy. We propose COMI, a coarse-to-fine adaptive context compression framework that jointly optimizes for semantic relevance and diversity under high compression rates. We introduce Marginal Information Gain (MIG), a metric defined as the relevance of a unit to the input query minus its semantic redundancy with other units, guiding the compression process to prioritize information that is both relevant and low redundant. The framework operates in two stages: (1) Coarse-Grained Group Reallocation, where the context is partitioned into groups and dynamically assigned compression rates based on inter-group MIG, ensuring compression budgets align with information value distribution; and (2) Fine-Grained Token Merging, where tokens within each group are fused via an intra-group MIG-based weighting mechanism, thereby preserving key semantics while avoiding the accumulation of redundancy. Extensive experiments across question-answering (e.g., NaturalQuestions, 2WikiMQA, HotpotQA and NarrativeQA), summarization (e.g., MultiNews) with various backbones (e.g., LLaMA-2-7B, Qwen2-7B) show that COMI outperforms existing baselines by a large margin, e.g., approximately 25-point Exact Match (EM) improvement under 32x compression constraint with Qwen2-7B on NaturalQuestions. 

---
# Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory 

**Authors**: Langyuan Cui, Chun Kai Ling, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01708)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in real-world scenarios where they may lack sufficient information to complete a given task. In such settings, the ability to actively seek out missing information becomes a critical capability. Existing approaches to enhancing this ability often rely on simplifying assumptions that degrade \textit{worst-case} performance. This is an issue with serious implications in high-stakes applications. In this work, we use the game of Twenty Questions to evaluate the information-seeking ability of LLMs. We introduce and formalize its adversarial counterpart, the Strategic Language Search (SLS) problem along with its variants as a two-player zero-sum extensive form game. We propose Game of Thought (GoT), a framework that applies game-theoretic techniques to approximate a Nash equilibrium (NE) strategy for the restricted variant of the game. Empirical results demonstrate that our approach consistently improves worst-case performance compared to (1) direct prompting-based methods and (2) heuristic-guided search methods across all tested settings. 

---
# Mechanistic Indicators of Steering Effectiveness in Large Language Models 

**Authors**: Mehdi Jafari, Hao Xue, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2602.01716)  

**Abstract**: Activation-based steering enables Large Language Models (LLMs) to exhibit targeted behaviors by intervening on intermediate activations without retraining. Despite its widespread use, the mechanistic factors that govern when steering succeeds or fails remain poorly understood, as prior work has relied primarily on black-box outputs or LLM-based judges. In this study, we investigate whether the reliability of steering can be diagnosed using internal model signals. We focus on two information-theoretic measures: the entropy-derived Normalized Branching Factor (NBF), and the Kullback-Leibler (KL) divergence between steered activations and targeted concepts in the vocabulary space. We hypothesize that effective steering corresponds to structured entropy preservation and coherent KL alignment across decoding steps. Building on a reliability study demonstrating high inter-judge agreement between two architecturally distinct LLMs, we use LLM-generated annotations as ground truth and show that these mechanistic signals provide meaningful predictive power for identifying successful steering and estimating failure probability. We further introduce a stronger evaluation baseline for Contrastive Activation Addition (CAA) and Sparse Autoencoder-based steering, the two most widely adopted activation-steering methods. 

---
# Scaling Search-Augmented LLM Reasoning via Adaptive Information Control 

**Authors**: Siheng Xiong, Oguzhan Gungordu, Blair Johnson, James C. Kerce, Faramarz Fekri  

**Link**: [PDF](https://arxiv.org/pdf/2602.01672)  

**Abstract**: Search-augmented reasoning agents interleave multi-step reasoning with external information retrieval, but uncontrolled retrieval often leads to redundant evidence, context saturation, and unstable learning. Existing approaches rely on outcome-based reinforcement learning (RL), which provides limited guidance for regulating information acquisition. We propose DeepControl, a framework for adaptive information control based on a formal notion of information utility, which measures the marginal value of retrieved evidence under a given reasoning state. Building on this utility, we introduce retrieval continuation and granularity control mechanisms that selectively regulate when to continue and stop retrieval, and how much information to expand. An annealed control strategy enables the agent to internalize effective information acquisition behaviors during training. Extensive experiments across seven benchmarks demonstrate that our method consistently outperforms strong baselines. In particular, our approach achieves average performance improvements of 9.4% and 8.6% on Qwen2.5-7B and Qwen2.5-3B, respectively, over strong outcome-based RL baselines, and consistently outperforms both retrieval-free and retrieval-based reasoning methods without explicit information control. These results highlight the importance of adaptive information control for scaling search-augmented reasoning agents to complex, real-world information environments. 

---
# Steering Vector Fields for Context-Aware Inference-Time Control in Large Language Models 

**Authors**: Jiaqian Li, Yanshu Li, Kuan-Hao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.01654)  

**Abstract**: Steering vectors (SVs) offer a lightweight way to control large language models (LLMs) at inference time by shifting hidden activations, providing a practical middle ground between prompting and fine-tuning. Yet SVs can be unreliable in practice. Some concepts are unsteerable, and even when steering helps on average it can backfire for a non-trivial fraction of inputs. Reliability also degrades in long-form generation and multi-attribute steering. We take a geometric view of these failures. A static SV applies the same update vector everywhere in representation space, implicitly assuming that the concept-improving direction is constant across contexts. When the locally effective direction varies with the current activation, a single global vector can become misaligned, which yields weak or reversed effects. Guided by this perspective, we propose Steering Vector Fields (SVF), which learns a differentiable concept scoring function whose local gradient defines the steering direction at each activation, making interventions explicitly context-dependent. This formulation supports coordinated multi-layer interventions in a shared, aligned concept space, and enables efficient long-form and multi-attribute control within a unified framework. Across multiple LLMs and steering tasks, SVF delivers stronger and more reliable control, improving the practicality of inference-time steering. 

---
# The Art of Socratic Inquiry: A Framework for Proactive Template-Guided Therapeutic Conversation Generation 

**Authors**: Mingwen Zhang, Minqiang Yang, Changsheng Ma, Yang Yu, Hui Bai, Chen Xu, Xiangzhen Kong, Bin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.01598)  

**Abstract**: Proactive questioning, where therapists deliberately initiate structured, cognition-guiding inquiries, is a cornerstone of cognitive behavioral therapy (CBT). Yet, current psychological large language models (LLMs) remain overwhelmingly reactive, defaulting to empathetic but superficial responses that fail to surface latent beliefs or guide behavioral change. To bridge this gap, we propose the \textbf{Socratic Inquiry Framework (SIF)}, a lightweight, plug-and-play therapeutic intent planner that transforms LLMs from passive listeners into active cognitive guides. SIF decouples \textbf{when to ask} (via Strategy Anchoring) from \textbf{what to ask} (via Template Retrieval), enabling context-aware, theory-grounded questioning without end-to-end retraining. Complementing SIF, we introduce \textbf{Socratic-QA}, a high-quality dataset of strategy-aligned Socratic sequences that provides explicit supervision for proactive reasoning. Experiments show that SIF significantly enhances proactive questioning frequency, conversational depth, and therapeutic alignment, marking a clear shift from reactive comfort to proactive exploration. Our work establishes a new paradigm for psychologically informed LLMs: not just to respond, but to guide. 

---
# CoDiQ: Test-Time Scaling for Controllable Difficult Question Generation 

**Authors**: Zhongyuan Peng, Caijun Xu, Changyi Xiao, Shibo Hong, Eli Zhang, Stephen Huang, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2602.01660)  

**Abstract**: Large Reasoning Models (LRMs) benefit substantially from training on challenging competition-level questions. However, existing automated question synthesis methods lack precise difficulty control, incur high computational costs, and struggle to generate competition-level questions at scale. In this paper, we propose CoDiQ (Controllable Difficult Question Generation), a novel framework enabling fine-grained difficulty control via test-time scaling while ensuring question solvability. Specifically, first, we identify a test-time scaling tendency (extended reasoning token budget boosts difficulty but reduces solvability) and the intrinsic properties defining the upper bound of a model's ability to generate valid, high-difficulty questions. Then, we develop CoDiQ-Generator from Qwen3-8B, which improves the upper bound of difficult question generation, making it particularly well-suited for challenging question construction. Building on the CoDiQ framework, we build CoDiQ-Corpus (44K competition-grade question sequences). Human evaluations show these questions are significantly more challenging than LiveCodeBench/AIME with over 82% solvability. Training LRMs on CoDiQ-Corpus substantially improves reasoning performance, verifying that scaling controlled-difficulty training questions enhances reasoning capabilities. We open-source CoDiQ-Corpus, CoDiQ-Generator, and implementations to support related research. 

---
# FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents 

**Authors**: Chiwei Zhu, Benfeng Xu, Mingxuan Du, Shaohan Wang, Xiaorui Wang, Zhendong Mao, Yongdong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.01566)  

**Abstract**: Deep research is emerging as a representative long-horizon task for large language model (LLM) agents. However, long trajectories in deep research often exceed model context limits, compressing token budgets for both evidence collection and report writing, and preventing effective test-time scaling. We introduce FS-Researcher, a file-system-based, dual-agent framework that scales deep research beyond the context window via a persistent workspace. Specifically, a Context Builder agent acts as a librarian which browses the internet, writes structured notes, and archives raw sources into a hierarchical knowledge base that can grow far beyond context length. A Report Writer agent then composes the final report section by section, treating the knowledge base as the source of facts. In this framework, the file system serves as a durable external memory and a shared coordination medium across agents and sessions, enabling iterative refinement beyond the context window. Experiments on two open-ended benchmarks (DeepResearch Bench and DeepConsult) show that FS-Researcher achieves state-of-the-art report quality across different backbone models. Further analyses demonstrate a positive correlation between final report quality and the computation allocated to the Context Builder, validating effective test-time scaling under the file-system paradigm. The code and data are anonymously open-sourced at this https URL. 

---
# Argument Rarity-based Originality Assessment for AI-Assisted Writing 

**Authors**: Keito Inoshita, Michiaki Omura, Tsukasa Yamanaka, Go Maeda, Kentaro Tsuji  

**Link**: [PDF](https://arxiv.org/pdf/2602.01560)  

**Abstract**: As Large Language Models (LLMs) have become capable of effortlessly generating high-quality text, traditional quality-focused writing assessment is losing its significance. If the essential goal of education is to foster critical thinking and original perspectives, assessment must also shift its paradigm from quality to originality. This study proposes Argument Rarity-based Originality Assessment (AROA), a framework for automatically evaluating argumentative originality in student essays. AROA defines originality as rarity within a reference corpus and evaluates it through four complementary components: structural rarity, claim rarity, evidence rarity, and cognitive depth. The framework quantifies the rarity of each component using density estimation and integrates them with a quality adjustment mechanism, thereby treating quality and originality as independent evaluation axes. Experiments using human essays and AI-generated essays revealed a strong negative correlation between quality and claim rarity, demonstrating a quality-originality trade-off where higher-quality texts tend to rely on typical claim patterns. Furthermore, while AI essays achieved comparable levels of structural complexity to human essays, their claim rarity was substantially lower than that of humans, indicating that LLMs can reproduce the form of argumentation but have limitations in the originality of content. 

---
# Ebisu: Benchmarking Large Language Models in Japanese Finance 

**Authors**: Xueqing Peng, Ruoyu Xiang, Fan Zhang, Mingzi Song, Mingyang Jiang, Yan Wang, Lingfei Qian, Taiki Hara, Yuqing Guo, Jimin Huang, Junichi Tsujii, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2602.01479)  

**Abstract**: Japanese finance combines agglutinative, head-final linguistic structure, mixed writing systems, and high-context communication norms that rely on indirect expression and implicit commitment, posing a substantial challenge for LLMs. We introduce Ebisu, a benchmark for native Japanese financial language understanding, comprising two linguistically and culturally grounded, expert-annotated tasks: JF-ICR, which evaluates implicit commitment and refusal recognition in investor-facing Q&A, and JF-TE, which assesses hierarchical extraction and ranking of nested financial terminology from professional disclosures. We evaluate a diverse set of open-source and proprietary LLMs spanning general-purpose, Japanese-adapted, and financial models. Results show that even state-of-the-art systems struggle on both tasks. While increased model scale yields limited improvements, language- and domain-specific adaptation does not reliably improve performance, leaving substantial gaps unresolved. Ebisu provides a focused benchmark for advancing linguistically and culturally grounded financial NLP. All datasets and evaluation scripts are publicly released. 

---
# On the Power of (Approximate) Reward Models for Inference-Time Scaling 

**Authors**: Youheng Zhu, Yiping Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.01381)  

**Abstract**: Inference-time scaling has recently emerged as a powerful paradigm for improving the reasoning capability of large language models. Among various approaches, Sequential Monte Carlo (SMC) has become a particularly important framework, enabling iterative generation, evaluation, rejection, and resampling of intermediate reasoning trajectories. A central component in this process is the reward model, which evaluates partial solutions and guides the allocation of computation during inference.
However, in practice, true reward models are never available. All deployed systems rely on approximate reward models, raising a fundamental question: Why and when do approximate reward models suffice for effective inference-time scaling? In this work, we provide a theoretical answer. We identify the Bellman error of the approximate reward model as the key quantity governing the effectiveness of SMC-based inference-time scaling. For a reasoning process of length $T$, we show that if the Bellman error of the approximate reward model is bounded by $O(1/T)$, then combining this reward model with SMC reduces the computational complexity of reasoning from exponential in $T$ to polynomial in $T$. This yields an exponential improvement in inference efficiency despite using only approximate rewards. 

---
# CRAFT: Calibrated Reasoning with Answer-Faithful Traces via Reinforcement Learning for Multi-Hop Question Answering 

**Authors**: Yu Liu, Wenxiao Zhang, Cong Cao, Fangfang Yuan, Weizhuo Chen, Cheng Hu, Pin Xu, Yuling Yang, Kun Peng, Diandian Guo, Qiang Sun, Yanbing Liu, Jin B. Hong, Zhiyuan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.01348)  

**Abstract**: Retrieval-augmented generation (RAG) is widely used to ground Large Language Models (LLMs) for multi-hop question answering. Recent work mainly focused on improving answer accuracy via fine-tuning and structured or reinforcement-based optimization. However, reliable reasoning in response generation faces three challenges: 1) Reasoning Collapse. Reasoning in multi-hop QA is inherently complex due to multi-hop composition and is further destabilized by noisy retrieval. 2) Reasoning-answer inconsistency. Due to the intrinsic uncertainty of LLM generation and exposure to evidence--distractor mixtures, models may produce correct answers that are not faithfully supported by their intermediate reasoning or evidence. 3) Loss of format control. Traditional chain-of-thought generation often deviates from required structured output formats, leading to incomplete or malformed structured content. To address these challenges, we propose CRAFT (Calibrated Reasoning with Answer-Faithful Traces), a Group Relative Policy Optimization (GRPO) based reinforcement learning framework that trains models to perform faithful reasoning during response generation. CRAFT employs dual reward mechanisms to optimize multi-hop reasoning: deterministic rewards ensure structural correctness while judge-based rewards verify semantic faithfulness. This optimization framework supports controllable trace variants that enable systematic analysis of how structure and scale affect reasoning performance and faithfulness. Experiments on three multi-hop QA benchmarks show that CRAFT improves both answer accuracy and reasoning faithfulness across model scales, with the CRAFT 7B model achieving competitive performance with closed-source LLMs across multiple reasoning trace settings. 

---
# Context Dependence and Reliability in Autoregressive Language Models 

**Authors**: Poushali Sengupta, Shashi Raj Pandey, Sabita Maharjan, Frank Eliassen  

**Link**: [PDF](https://arxiv.org/pdf/2602.01378)  

**Abstract**: Large language models (LLMs) generate outputs by utilizing extensive context, which often includes redundant information from prompts, retrieved passages, and interaction history. In critical applications, it is vital to identify which context elements actually influence the output, as standard explanation methods struggle with redundancy and overlapping context. Minor changes in input can lead to unpredictable shifts in attribution scores, undermining interpretability and raising concerns about risks like prompt injection. This work addresses the challenge of distinguishing essential context elements from correlated ones. We introduce RISE (Redundancy-Insensitive Scoring of Explanation), a method that quantifies the unique influence of each input relative to others, minimizing the impact of redundancies and providing clearer, stable attributions. Experiments demonstrate that RISE offers more robust explanations than traditional methods, emphasizing the importance of conditional information for trustworthy LLM explanations and monitoring. 

---
# EverMemBench: Benchmarking Long-Term Interactive Memory in Large Language ModelsEverMemBench: Benchmarking Long-Term Interactive Memory in Large Language Models 

**Authors**: Chuanrui Hu, Tong Li, Xingze Gao, Hongda Chen, Dannong Xu, Yi Bai, Tianwei Lin, Xinda Zhao, Xiaohong Li, Jiaqi An, Yunyun Han, Jian Pei, Yafeng Deng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01313)  

**Abstract**: Long-term conversational memory is essential for LLM-based assistants, yet existing benchmarks focus on dyadic, single-topic dialogues that fail to capture real-world complexity. We introduce EverMemBench, a benchmark featuring multi-party, multi-group conversations spanning over 1 million tokens with temporally evolving information, cross-topic interleaving, and role-specific personas. EverMemBench evaluates memory systems across three dimensions through 1,000+ QA pairs: fine-grained recall, memory awareness, and user profile understanding. Our evaluation reveals critical limitations: (1) multi-hop reasoning collapses in multi-party settings, with even oracle models achieving only 26%; (2) temporal reasoning remains unsolved, requiring version semantics beyond timestamp matching; (3) memory awareness is bottlenecked by retrieval, where current similarity-based methods fail to bridge the semantic gap between queries and implicitly relevant memories. EverMemBench provides a challenging testbed for developing next-generation memory architectures. 

---
# ConPress: Learning Efficient Reasoning from Multi-Question Contextual Pressure 

**Authors**: Jie Deng, Shining Liang, Jun Li, Hongzhi Li, Yutao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2602.01472)  

**Abstract**: Large reasoning models (LRMs) typically solve reasoning-intensive tasks by generating long chain-of-thought (CoT) traces, leading to substantial inference overhead. We identify a reproducible inference-time phenomenon, termed Self-Compression: when multiple independent and answerable questions are presented within a single prompt, the model spontaneously produces shorter reasoning traces for each question. This phenomenon arises from multi-question contextual pressure during generation and consistently manifests across models and benchmarks. Building on this observation, we propose ConPress (Learning from Contextual Pressure), a lightweight self-supervised fine-tuning approach. ConPress constructs multi-question prompts to induce self-compression, samples the resulting model outputs, and parses and filters per-question traces to obtain concise yet correct reasoning trajectories. These trajectories are directly used for supervised fine-tuning, internalizing compressed reasoning behavior in single-question settings without external teachers, manual pruning, or reinforcement learning. With only 8k fine-tuning examples, ConPress reduces reasoning token usage by 59% on MATH500 and 33% on AIME25, while maintaining competitive accuracy. 

---
# ASTER: Agentic Scaling with Tool-integrated Extended Reasoning 

**Authors**: Xuqin Zhang, Quan He, Zhenrui Zheng, Zongzhang Zhang, Xu He, Dong Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.01204)  

**Abstract**: Reinforcement learning (RL) has emerged as a dominant paradigm for eliciting long-horizon reasoning in Large Language Models (LLMs). However, scaling Tool-Integrated Reasoning (TIR) via RL remains challenging due to interaction collapse: a pathological state where models fail to sustain multi-turn tool usage, instead degenerating into heavy internal reasoning with only trivial, post-hoc code verification. We systematically study three questions: (i) how cold-start SFT induces an agentic, tool-using behavioral prior, (ii) how the interaction density of cold-start trajectories shapes exploration and downstream RL outcomes, and (iii) how the RL interaction budget affects learning dynamics and generalization under varying inference-time budgets. We then introduce ASTER (Agentic Scaling with Tool-integrated Extended Reasoning), a framework that circumvents this collapse through a targeted cold-start strategy prioritizing interaction-dense trajectories. We find that a small expert cold-start set of just 4K interaction-dense trajectories yields the strongest downstream performance, establishing a robust prior that enables superior exploration during extended RL training. Extensive evaluations demonstrate that ASTER-4B achieves state-of-the-art results on competitive mathematical benchmarks, reaching 90.0% on AIME 2025, surpassing leading frontier open-source models, including DeepSeek-V3.2-Exp. 

---
# Don't Judge a Book by its Cover: Testing LLMs' Robustness Under Logical Obfuscation 

**Authors**: Abhilekh Borah, Shubhra Ghosh, Kedar Joshi, Aditya Kumar Guru, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2602.01132)  

**Abstract**: Tasks such as solving arithmetic equations, evaluating truth tables, and completing syllogisms are handled well by large language models (LLMs) in their standard form, but they often fail when the same problems are posed in logically equivalent yet obfuscated formats. To study this vulnerability, we introduce Logifus, a structure-preserving logical obfuscation framework, and, utilizing this, we present LogiQAte, a first-of-its-kind diagnostic benchmark with 1,108 questions across four reasoning tasks: (i) Obfus FOL (first-order logic entailment under equivalence-preserving rewrites), (ii) Obfus Blood Relation (family-graph entailment under indirect relational chains), (iii) Obfus Number Series (pattern induction under symbolic substitutions), and (iv) Obfus Direction Sense (navigation reasoning under altered directions and reference frames). Across all the tasks, evaluating six state-of-the-art models, we find that obfuscation severely degrades zero-shot performance, with performance dropping on average by 47% for GPT-4o, 27% for GPT-5, and 22% for reasoning model, o4-mini. Our findings reveal that current LLMs parse questions without deep understanding, highlighting the urgency of building models that genuinely comprehend and preserve meaning beyond surface form. 

---
# Typologically-Informed Candidate Reranking for LLM-based Translation into Low-Resource Languages 

**Authors**: Nipuna Abeykoon, Ashen Weerathunga, Pubudu Wijesinghe, Parameswari Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2602.01162)  

**Abstract**: Large language models trained predominantly on high-resource languages exhibit systematic biases toward dominant typological patterns, leading to structural non-conformance when translating into typologically divergent low-resource languages. We present a framework that leverages linguistic typology to improve translation quality without parallel training data or model retraining. The framework consists of two components: the Universal Metalinguistic Framework (UMF), which represents languages as structured profiles across 16 typological dimensions with divergence-weighted scoring, and the Computational Engine, which operates through linguistic disambiguation during generation and typological compliance scoring during selection. Evaluation across nine language pairs demonstrates intervention rates strongly correlating with typological distance from English. In experiments on 341 English sentences each having different morphological and syntactic phenomena, the framework shows an intervention precision of 48.16% for conservatively treated languages, 28.15% for morphologically dense languages, and 86.26% for structurally profiled languages. The framework requires no parallel training data and operates with any LLM capable of producing multiple candidate outputs, enabling practical deployment for under-resourced languages. 

---
# PedagoSense: A Pedology Grounded LLM System for Pedagogical Strategy Detection and Contextual Response Generation in Learning Dialogues 

**Authors**: Shahem Sultan, Shahem Fadi, Yousef Melhim, Ibrahim Alsarraj, Besher Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2602.01169)  

**Abstract**: This paper addresses the challenge of improving interaction quality in dialogue based learning by detecting and recommending effective pedagogical strategies in tutor student conversations. We introduce PedagoSense, a pedology grounded system that combines a two stage strategy classifier with large language model generation. The system first detects whether a pedagogical strategy is present using a binary classifier, then performs fine grained classification to identify the specific strategy. In parallel, it recommends an appropriate strategy from the dialogue context and uses an LLM to generate a response aligned with that strategy. We evaluate on human annotated tutor student dialogues, augmented with additional non pedagogical conversations for the binary task. Results show high performance for pedagogical strategy detection and consistent gains when using data augmentation, while analysis highlights where fine grained classes remain challenging. Overall, PedagoSense bridges pedagogical theory and practical LLM based response generation for more adaptive educational technologies. 

---
# Chronos: Learning Temporal Dynamics of Reasoning Chains for Test-Time Scaling 

**Authors**: Kai Zhang, Jiayi Liao, Chengpeng Li, Ziyuan Xie, Sihang Li, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.01208)  

**Abstract**: Test-Time Scaling (TTS) has emerged as an effective paradigm for improving the reasoning performance of large language models (LLMs). However, existing methods -- most notably majority voting and heuristic token-level scoring -- treat reasoning traces or tokens equally, thereby being susceptible to substantial variations in trajectory quality and localized logical failures. In this work, we introduce \textbf{Chronos}, a lightweight and plug-and-play chronological reasoning scorer that models each trajectory as a time series. Specifically, Chronos learns to capture trajectory features of token probabilities, assigns quality scores accordingly, and employs a weighted voting mechanism. Extensive evaluations on both in-domain and out-of-domain benchmarks demonstrate that Chronos consistently delivers substantial gains across a variety of models, with negligible computational overhead. Notably, Chronos@128 achieves relative improvements of 34.21\% over Pass@1 and 22.70\% over Maj@128 on HMMT25 using Qwen3-4B-Thinking-2507, highlighting its effectiveness. 

---
# Bridging Lexical Ambiguity and Vision: A Mini Review on Visual Word Sense Disambiguation 

**Authors**: Shashini Nilukshi, Deshan Sumanathilaka  

**Link**: [PDF](https://arxiv.org/pdf/2602.01193)  

**Abstract**: This paper offers a mini review of Visual Word Sense Disambiguation (VWSD), which is a multimodal extension of traditional Word Sense Disambiguation (WSD). VWSD helps tackle lexical ambiguity in vision-language tasks. While conventional WSD depends only on text and lexical resources, VWSD uses visual cues to find the right meaning of ambiguous words with minimal text input. The review looks at developments from early multimodal fusion methods to new frameworks that use contrastive models like CLIP, diffusion-based text-to-image generation, and large language model (LLM) support. Studies from 2016 to 2025 are examined to show the growth of VWSD through feature-based, graph-based, and contrastive embedding techniques. It focuses on prompt engineering, fine-tuning, and adapting to multiple languages. Quantitative results show that CLIP-based fine-tuned models and LLM-enhanced VWSD systems consistently perform better than zero-shot baselines, achieving gains of up to 6-8\% in Mean Reciprocal Rank (MRR). However, challenges still exist, such as limitations in context, model bias toward common meanings, a lack of multilingual datasets, and the need for better evaluation frameworks. The analysis highlights the growing overlap of CLIP alignment, diffusion generation, and LLM reasoning as the future path for strong, context-aware, and multilingual disambiguation systems. 

---
# Exploring Knowledge Purification in Multi-Teacher Knowledge Distillation for LLMs 

**Authors**: Ruihan Jin, Pengpeng Shao, Zhengqi Wen, Jinyang Wu, Mingkuan Feng, Shuo Yang, Chu Yuan Zhang, Jianhua Tao  

**Link**: [PDF](https://arxiv.org/pdf/2602.01064)  

**Abstract**: Knowledge distillation has emerged as a pivotal technique for transferring knowledge from stronger large language models (LLMs) to smaller, more efficient models. However, traditional distillation approaches face challenges related to knowledge conflicts and high resource demands, particularly when leveraging multiple teacher models. In this paper, we introduce the concept of \textbf{Knowledge Purification}, which consolidates the rationales from multiple teacher LLMs into a single rationale, thereby mitigating conflicts and enhancing efficiency. To investigate the effectiveness of knowledge purification, we further propose five purification methods from various perspectives. Our experiments demonstrate that these methods not only improve the performance of the distilled model but also effectively alleviate knowledge conflicts. Moreover, router-based methods exhibit robust generalization capabilities, underscoring the potential of innovative purification techniques in optimizing multi-teacher distillation and facilitating the practical deployment of powerful yet lightweight models. 

---
# Logic-Oriented Retriever Enhancement via Contrastive Learning 

**Authors**: Wenxuan Zhang, Yuan-Hao Jiang, Changyong Qi, Rui Jia, Yonghe Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.01116)  

**Abstract**: Large language models (LLMs) struggle in knowledge-intensive tasks, as retrievers often overfit to surface similarity and fail on queries involving complex logical relations. The capacity for logical analysis is inherent in model representations but remains underutilized in standard training. LORE (Logic ORiented Retriever Enhancement) introduces fine-grained contrastive learning to activate this latent capacity, guiding embeddings toward evidence aligned with logical structure rather than shallow similarity. LORE requires no external upervision, resources, or pre-retrieval analysis, remains index-compatible, and consistently improves retrieval utility and downstream generation while maintaining efficiency. The datasets and code are publicly available at this https URL. 

---
# Personality Expression Across Contexts: Linguistic and Behavioral Variation in LLM Agents 

**Authors**: Bin Han, Deuksin Kwon, Jonathan Gratch  

**Link**: [PDF](https://arxiv.org/pdf/2602.01063)  

**Abstract**: Large Language Models (LLMs) can be conditioned with explicit personality prompts, yet their behavioral realization often varies depending on context. This study examines how identical personality prompts lead to distinct linguistic, behavioral, and emotional outcomes across four conversational settings: ice-breaking, negotiation, group decision, and empathy tasks. Results show that contextual cues systematically influence both personality expression and emotional tone, suggesting that the same traits are expressed differently depending on social and affective demands. This raises an important question for LLM-based dialogue agents: whether such variations reflect inconsistency or context-sensitive adaptation akin to human behavior. Viewed through the lens of Whole Trait Theory, these findings highlight that LLMs exhibit context-sensitive rather than fixed personality expression, adapting flexibly to social interaction goals and affective conditions. 

---
# Distilling Token-Trained Models into Byte-Level Models 

**Authors**: Zishuo Bao, Jiaqi Leng, Junxiong Wang, Bowen Peng, Yucheng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.01007)  

**Abstract**: Byte Language Models (BLMs) have emerged as a promising direction for scaling language models beyond tokenization. However, existing BLMs typically require training from scratch on trillions of bytes, making them prohibitively expensive. In this paper, we propose an efficient distillation recipe that converts existing token-trained LLMs into BLMs while retaining comparable capabilities. Our recipe follows a two-stage curriculum: (1) Progressive Knowledge Distillation, which aligns byte-level representations with the embeddings of the token-trained teacher model; and (2) Byte-Level Supervised Fine-Tuning, which enables end-to-end generation entirely in the byte space. We validate our approach across multiple model families, including Llama, Qwen, and OLMo, and demonstrate that the distilled BLMs retain most of the teacher models' performance using only approximately 125B bytes. 

---
# Sparse Reward Subsystem in Large Language Models 

**Authors**: Guowei Xu, Mert Yuksekgonul, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2602.00986)  

**Abstract**: In this paper, we identify a sparse reward subsystem within the hidden states of Large Language Models (LLMs), drawing an analogy to the biological reward subsystem in the human brain. We demonstrate that this subsystem contains value neurons that represent the model's internal expectation of state value, and through intervention experiments, we establish the importance of these neurons for reasoning. Our experiments reveal that these value neurons are robust across diverse datasets, model scales, and architectures; furthermore, they exhibit significant transferability across different datasets and models fine-tuned from the same base model. By examining cases where value predictions and actual rewards diverge, we identify dopamine neurons within the reward subsystem which encode reward prediction errors (RPE). These neurons exhibit high activation when the reward is higher than expected and low activation when the reward is lower than expected. 

---
# Reliable Use of Lemmas via Eligibility Reasoning and Section$-$Aware Reinforcement Learning 

**Authors**: Zhikun Xu, Xiaodong Yu, Ben Zhou, Jiang Liu, Jialian Wu, Ze Wang, Ximeng Sun, Hao Chen, Zicheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00998)  

**Abstract**: Recent large language models (LLMs) perform strongly on mathematical benchmarks yet often misapply lemmas, importing conclusions without validating assumptions. We formalize lemma$-$judging as a structured prediction task: given a statement and a candidate lemma, the model must output a precondition check and a conclusion$-$utility check, from which a usefulness decision is derived. We present RULES, which encodes this specification via a two$-$section output and trains with reinforcement learning plus section$-$aware loss masking to assign penalty to the section responsible for errors. Training and evaluation draw on diverse natural language and formal proof corpora; robustness is assessed with a held$-$out perturbation suite; and end$-$to$-$end evaluation spans competition$-$style, perturbation$-$aligned, and theorem$-$based problems across various LLMs. Results show consistent in$-$domain gains over both a vanilla model and a single$-$label RL baseline, larger improvements on applicability$-$breaking perturbations, and parity or modest gains on end$-$to$-$end tasks; ablations indicate that the two$-$section outputs and section$-$aware reinforcement are both necessary for robustness. 

---
# From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization 

**Authors**: Chaoqun Cui, Shijing Wang, Liangbin Huang, Qingqing Gu, Zhaolong Huang, Xiao Zeng, Wenji Mao  

**Link**: [PDF](https://arxiv.org/pdf/2602.01068)  

**Abstract**: The rapid development of Large Language Models (LLMs) has significantly enhanced the general capabilities of machine translation. However, as application scenarios become more complex, the limitations of LLMs in vertical domain translations are gradually becoming apparent. In this study, we focus on how to construct translation LLMs that meet the needs of domain customization. We take visual media subtitle translation as our topic and explore how to train expressive and vivid translation LLMs. We investigated the situations of subtitle translation and other domains of literal and liberal translation, verifying the reliability of LLM as reward model and evaluator for translation. Additionally, to train an expressive translation LLM, we constructed and released a multidirectional subtitle parallel corpus dataset and proposed the Adaptive Local Preference Optimization (ALPO) method to address fine-grained preference alignment. Experimental results demonstrate that ALPO achieves outstanding performance in multidimensional evaluation of translation quality. 

---
# MedSpeak: A Knowledge Graph-Aided ASR Error Correction Framework for Spoken Medical QA 

**Authors**: Yutong Song, Shiva Shrestha, Chenhan Lyu, Elahe Khatibi, Pengfei Zhang, Honghui Xu, Nikil Dutt, Amir Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2602.00981)  

**Abstract**: Spoken question-answering (SQA) systems relying on automatic speech recognition (ASR) often struggle with accurately recognizing medical terminology. To this end, we propose MedSpeak, a novel knowledge graph-aided ASR error correction framework that refines noisy transcripts and improves downstream answer prediction by leveraging both semantic relationships and phonetic information encoded in a medical knowledge graph, together with the reasoning power of LLMs. Comprehensive experimental results on benchmarks demonstrate that MedSpeak significantly improves the accuracy of medical term recognition and overall medical SQA performance, establishing MedSpeak as a state-of-the-art solution for medical SQA. The code is available at this https URL. 

---
# Trust in One Round: Confidence Estimation for Large Language Models via Structural Signals 

**Authors**: Pengyue Yang, Jiawen Wen, Haolin Jin, Linghan Huang, Huaming Chen, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.00977)  

**Abstract**: Large language models (LLMs) are increasingly deployed in domains where errors carry high social, scientific, or safety costs. Yet standard confidence estimators, such as token likelihood, semantic similarity and multi-sample consistency, remain brittle under distribution shift, domain-specialised text, and compute limits. In this work, we present Structural Confidence, a single-pass, model-agnostic framework that enhances output correctness prediction based on multi-scale structural signals derived from a model's final-layer hidden-state trajectory. By combining spectral, local-variation, and global shape descriptors, our method captures internal stability patterns that are missed by probabilities and sentence embeddings. We conduct extensive, cross-domain evaluation across four heterogeneous benchmarks-FEVER (fact verification), SciFact (scientific claims), WikiBio-hallucination (biographical consistency), and TruthfulQA (truthfulness-oriented QA). Our Structural Confidence framework demonstrates strong performance compared with established baselines in terms of AUROC and AUPR. More importantly, unlike sampling-based consistency methods which require multiple stochastic generations and an auxiliary model, our approach uses a single deterministic forward pass, offering a practical basis for efficient, robust post-hoc confidence estimation in socially impactful, resource-constrained LLM applications. 

---
# Large Language Models as Students Who Think Aloud: Overly Coherent, Verbose, and Confident 

**Authors**: Conrad Borchers, Jill-Jênn Vie, Roger Azevedo  

**Link**: [PDF](https://arxiv.org/pdf/2602.01015)  

**Abstract**: Large language models (LLMs) are increasingly embedded in AI-based tutoring systems. Can they faithfully model novice reasoning and metacognitive judgments? Existing evaluations emphasize problem-solving accuracy, overlooking the fragmented and imperfect reasoning that characterizes human learning. We evaluate LLMs as novices using 630 think-aloud utterances from multi-step chemistry tutoring problems with problem-solving logs of student hint use, attempts, and problem context. We compare LLM-generated reasoning to human learner utterances under minimal and extended contextual prompting, and assess the models' ability to predict step-level learner success. Although GPT-4.1 generates fluent and contextually appropriate continuations, its reasoning is systematically over-coherent, verbose, and less variable than human think-alouds. These effects intensify with a richer problem-solving context during prompting. Learner performance was consistently overestimated. These findings highlight epistemic limitations of simulating learning with LLMs. We attribute these limitations to LLM training data, including expert-like solutions devoid of expressions of affect and working memory constraints during problem solving. Our evaluation framework can guide future design of adaptive systems that more faithfully support novice learning and self-regulation using generative artificial intelligence. 

---
# Verification Required: The Impact of Information Credibility on AI Persuasion 

**Authors**: Saaduddin Mahmud, Eugene Bagdasarian, Shlomo Zilberstein  

**Link**: [PDF](https://arxiv.org/pdf/2602.00970)  

**Abstract**: Agents powered by large language models (LLMs) are increasingly deployed in settings where communication shapes high-stakes decisions, making a principled understanding of strategic communication essential. Prior work largely studies either unverifiable cheap-talk or fully verifiable disclosure, failing to capture realistic domains in which information has probabilistic credibility. We introduce MixTalk, a strategic communication game for LLM-to-LLM interaction that models information credibility. In MixTalk, a sender agent strategically combines verifiable and unverifiable claims to communicate private information, while a receiver agent allocates a limited budget to costly verification and infers the underlying state from prior beliefs, claims, and verification outcomes. We evaluate state-of-the-art LLM agents in large-scale tournaments across three realistic deployment settings, revealing their strengths and limitations in reasoning about information credibility and the explicit behavior that shapes these interactions. Finally, we propose Tournament Oracle Policy Distillation (TOPD), an offline method that distills tournament oracle policy from interaction logs and deploys it in-context at inference time. Our results show that TOPD significantly improves receiver robustness to persuasion. 

---
# Factuality on Demand: Controlling the Factuality-Informativeness Trade-off in Text Generation 

**Authors**: Ziwei Gong, Yanda Chen, Julia Hirschberg, Chen Zhao, He He, Zhou Yu, Kathleen Mckeown  

**Link**: [PDF](https://arxiv.org/pdf/2602.00848)  

**Abstract**: Large language models (LLMs) encode knowledge with varying degrees of confidence. When responding to queries, models face an inherent trade-off: they can generate responses that are less informative but highly factual, or more informative but potentially less accurate. Different applications demand different balances between informativeness and factuality. We introduce Factuality-Controlled Generation (FCG), a framework that enables users to specify factuality constraints alongside their queries. We propose to evaluate FCG performance on two dimensions: adherence to factuality constraints and response informativeness. We propose to train models on the FCG task using synthetic data, and show that our synthetic training significantly improves models' ability to both respect factuality requirements and maintain informativeness in their outputs. 

---
# Neural FOXP2 -- Language Specific Neuron Steering for Targeted Language Improvement in LLMs 

**Authors**: Anusa Saha, Tanmay Joshi, Vinija Jain, Aman Chadha, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2602.00945)  

**Abstract**: LLMs are multilingual by training, yet their lingua franca is often English, reflecting English language dominance in pretraining. Other languages remain in parametric memory but are systematically suppressed. We argue that language defaultness is governed by a sparse, low-rank control circuit, language neurons, that can be mechanistically isolated and safely steered.
We introduce Neural FOXP2, that makes a chosen language (Hindi or Spanish) primary in a model by steering language-specific neurons. Neural FOXP2 proceeds in three stages: (i) Localize: We train per-layer SAEs so each activation decomposes into a small set of active feature components. For every feature, we quantify English vs. Hindi/Spanish selectivity overall logit-mass lift toward the target-language token set. Tracing the top-ranked features back to their strongest contributing units yields a compact language-neuron set. (ii) Steering directions: We localize controllable language-shift geometry via a spectral low-rank analysis. For each layer, we build English to target activation-difference matrices and perform layerwise SVD to extract the dominant singular directions governing language change. The eigengap and effective-rank spectra identify a compact steering subspace and an empirically chosen intervention window (where these directions are strongest and most stable). (iii) Steer: We apply a signed, sparse activation shift targeted to the language neurons. Concretely, within low to mid layers we add a positive steering along the target-language dominant directions and a compensating negative shift toward the null space for the English neurons, yielding controllable target-language defaultness. 

---
# Reasoning as State Transition: A Representational Analysis of Reasoning Evolution in Large Language Models 

**Authors**: Siyuan Zhang, Jialian Li, Yichi Zhang, Xiao Yang, Yinpeng Dong, Hang Su  

**Link**: [PDF](https://arxiv.org/pdf/2602.00770)  

**Abstract**: Large Language Models have achieved remarkable performance on reasoning tasks, motivating research into how this ability evolves during training. Prior work has primarily analyzed this evolution via explicit generation outcomes, treating the reasoning process as a black box and obscuring internal changes. To address this opacity, we introduce a representational perspective to investigate the dynamics of the model's internal states. Through comprehensive experiments across models at various training stages, we discover that post-training yields only limited improvement in static initial representation quality. Furthermore, we reveal that, distinct from non-reasoning tasks, reasoning involves a significant continuous distributional shift in representations during generation. Comparative analysis indicates that post-training empowers models to drive this transition toward a better distribution for task solving. To clarify the relationship between internal states and external outputs, statistical analysis confirms a high correlation between generation correctness and the final representations; while counterfactual experiments identify the semantics of the generated tokens, rather than additional computation during inference or intrinsic parameter differences, as the dominant driver of the transition. Collectively, we offer a novel understanding of the reasoning process and the effect of training on reasoning enhancement, providing valuable insights for future model analysis and optimization. 

---
# ExperienceWeaver: Optimizing Small-sample Experience Learning for LLM-based Clinical Text Improvement 

**Authors**: Ziyan Xiao, Yinghao Zhu, Liang Peng, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00740)  

**Abstract**: Clinical text improvement is vital for healthcare efficiency but remains difficult due to limited high-quality data and the complex constraints of medical documentation. While Large Language Models (LLMs) show promise, current approaches struggle in small-sample settings: supervised fine-tuning is data-intensive and costly, while retrieval-augmented generation often provides superficial corrections without capturing the reasoning behind revisions. To address these limitations, we propose ExperienceWeaver, a hierarchical framework that shifts the focus from data retrieval to experience learning. Instead of simply recalling past examples, ExperienceWeaver distills noisy, multi-dimensional feedback into structured, actionable knowledge. Specifically, error-specific Tips and high-level Strategies. By injecting this distilled experience into an agentic pipeline, the model learns "how to revise" rather than just "what to revise". Extensive evaluations across four clinical datasets demonstrate that ExperienceWeaver consistently improves performance, surpassing state-of-the-art models such as Gemini-3 Pro in small-sample settings. 

---
# HyLRA: Hybrid Layer Reuse Attention for Efficient Long-Context Inference 

**Authors**: Xuan Ai, Qingqing Yang, Peng Wang, Lei Deng, Lin Zhang, Renhai Chen, Gong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.00777)  

**Abstract**: Long-context inference in Large Language Models (LLMs) is bottlenecked by the quadratic computation complexity of attention and the substantial memory footprint of Key-Value (KV) caches. While existing sparse attention mechanisms attempt to mitigate this by exploiting inherent sparsity, they often rely on rigid patterns or aggressive pruning, failing to achieve an optimal balance between efficiency and accuracy. In this paper, we introduce {\bf HyLRA} ({\bf Hy}brid {\bf L}ayer {\bf R}euse {\bf A}ttention), a novel framework driven by layer-wise sparsity profiling. Our empirical analysis uncovers a dual characteristic in attention mechanics: \textit{intra-layer sensitivity}, where specific layers necessitate full attention to prevent feature distortion, and \textit{inter-layer similarity}, where consecutive layers share substantial critical tokens. Based on these observations, HyLRA employs an offline dynamic programming approach to derive an optimal layer-wise policy. This hybrid strategy retains full attention for sensitive layers to ensure robustness, while enabling tolerant layers to bypass quadratic calculations by directly reusing top-$k$ indices from preceding layers. This approach allows LLMs to restrict computation to the most critical tokens, effectively overcoming the quadratic bottleneck of dense attention. Extensive evaluations demonstrate that HyLRA improves inference throughput by 6\%--46\% while maintaining comparable performance (with $<1\%$ accuracy degradation), consistently outperforming state-of-the-art sparse attention methods. HyLRA is open source at \href{this https URL}{\texttt{/r/unified-cache-management-CF80/}} 

---
# Adaptive Ability Decomposing for Unlocking Large Reasoning Model Effective Reinforcement Learning 

**Authors**: Zhipeng Chen, Xiaobo Qin, Wayne Xin Zhao, Youbin Wu, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2602.00759)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has shown great potential to enhance the reasoning ability of large language models (LLMs). However, due to the limited amount of information provided during the RLVR process, the model can only engage in largely blind exploration, which often results in failure on challenging problems. To provide additional information for the RLVR process without relying on a teacher model, we propose A$^2$D, an Adaptive Ability Decomposing method for enhancing the effectiveness of RLVR. Specifically, we first train a decomposer via RLVR without distillation, enabling it to decompose complex questions into a set of simpler sub-questions. Next, we use this decomposer to annotate sub-questions for each question in the training dataset, and then train the reasoner under RLVR with sub-question guidance. To better understand A$^2$D, we first compare its performance with competitive baselines, showing its effectiveness. Next, we observe that our method functions as a plug-and-play module that can be applied to different RLVR algorithms. Furthermore, we conduct an analysis of the decomposer, revealing how the RLVR process affects its performance and behavior, and which type of guidance is better suited for enhancing the reasoner's exploration and exploitation abilities. 

---
# CURP: Codebook-based Continuous User Representation for Personalized Generation with LLMs 

**Authors**: Liang Wang, Xinyi Mou, Xiaoyou Liu, Xuanjing Huang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.00742)  

**Abstract**: User modeling characterizes individuals through their preferences and behavioral patterns to enable personalized simulation and generation with Large Language Models (LLMs) in contemporary approaches. However, existing methods, whether prompt-based or training-based methods, face challenges in balancing personalization quality against computational and data efficiency. We propose a novel framework CURP, which employs a bidirectional user encoder and a discrete prototype codebook to extract multi-dimensional user traits. This design enables plug-and-play personalization with a small number of trainable parameters (about 20M parameters, about 0.2\% of the total model size). Through extensive experiments on variant generation tasks, we show that CURP achieves superior performance and generalization compared to strong baselines, while offering better interpretability and scalability. The code are available at this https URL 

---
# WordCraft: Scaffolding the Keyword Method for L2 Vocabulary Learning with Multimodal LLMs 

**Authors**: Yuheng Shao, Junjie Xiong, Chaoran Wu, Xiyuan Wang, Ziyu Zhou, Yang Ouyang, Qinyi Tao, Quan Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.00762)  

**Abstract**: Applying the keyword method for vocabulary memorization remains a significant challenge for L1 Chinese-L2 English learners. They frequently struggle to generate phonologically appropriate keywords, construct coherent associations, and create vivid mental imagery to aid long-term retention. Existing approaches, including fully automated keyword generation and outcome-oriented mnemonic aids, either compromise learner engagement or lack adequate process-oriented guidance. To address these limitations, we conducted a formative study with L1 Chinese-L2 English learners and educators (N=18), which revealed key difficulties and requirements in applying the keyword method to vocabulary learning. Building on these insights, we introduce WordCraft, a learner-centered interactive tool powered by Multimodal Large Language Models (MLLMs). WordCraft scaffolds the keyword method by guiding learners through keyword selection, association construction, and image formation, thereby enhancing the effectiveness of vocabulary memorization. Two user studies demonstrate that WordCraft not only preserves the generation effect but also achieves high levels of effectiveness and usability. 

---
# Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training 

**Authors**: Shengrui Li, Fei Zhao, Kaiyan Zhao, Jieying Ye, Haifeng Liu, Fangcheng Shi, Zheyong Xie, Yao Hu, Shaosheng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2602.00747)  

**Abstract**: Determining an effective data mixture is a key factor in Large Language Model (LLM) pre-training, where models must balance general competence with proficiency on hard tasks such as math and code. However, identifying an optimal mixture remains an open challenge, as existing approaches either rely on unreliable tiny-scale proxy experiments or require prohibitively expensive large-scale exploration. To address this, we propose Decouple Searching from Training Mix (DeMix), a novel framework that leverages model merging to predict optimal data ratios. Instead of training proxy models for every sampled mixture, DeMix trains component models on candidate datasets at scale and derives data mixture proxies via weighted model merging. This paradigm decouples search from training costs, enabling evaluation of unlimited sampled mixtures without extra training burden and thus facilitating better mixture discovery through more search trials. Extensive experiments demonstrate that DeMix breaks the trade-off between sufficiency, accuracy and efficiency, obtaining the optimal mixture with higher benchmark performance at lower search cost. Additionally, we release the DeMix Corpora, a comprehensive 22T-token dataset comprising high-quality pre-training data with validated mixtures to facilitate open research. Our code and DeMix Corpora is available at this https URL. 

---
# Can Small Language Models Handle Context-Summarized Multi-Turn Customer-Service QA? A Synthetic Data-Driven Comparative Evaluation 

**Authors**: Lakshan Cooray, Deshan Sumanathilaka, Pattigadapa Venkatesh Raju  

**Link**: [PDF](https://arxiv.org/pdf/2602.00665)  

**Abstract**: Customer-service question answering (QA) systems increasingly rely on conversational language understanding. While Large Language Models (LLMs) achieve strong performance, their high computational cost and deployment constraints limit practical use in resource-constrained environments. Small Language Models (SLMs) provide a more efficient alternative, yet their effectiveness for multi-turn customer-service QA remains underexplored, particularly in scenarios requiring dialogue continuity and contextual understanding. This study investigates instruction-tuned SLMs for context-summarized multi-turn customer-service QA, using a history summarization strategy to preserve essential conversational state. We also introduce a conversation stage-based qualitative analysis to evaluate model behavior across different phases of customer-service interactions. Nine instruction-tuned low-parameterized SLMs are evaluated against three commercial LLMs using lexical and semantic similarity metrics alongside qualitative assessments, including human evaluation and LLM-as-a-judge methods. Results show notable variation across SLMs, with some models demonstrating near-LLM performance, while others struggle to maintain dialogue continuity and contextual alignment. These findings highlight both the potential and current limitations of low-parameterized language models for real-world customer-service QA systems. 

---
# Lookahead-then-Verify: Reliable Constrained Decoding for Diffusion LLMs under Context-Free Grammars 

**Authors**: Yitong Zhang, Yongmin Li, Yuetong Liu, Jia Li, Xiaoran Jia, Zherui Li, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.00612)  

**Abstract**: Diffusion Large Language Models (dLLMs) have demonstrated promising generative capabilities and are increasingly used to produce formal languages defined by context-free grammars, such as source code and chemical expressions. However, as probabilistic models, they still struggle to generate syntactically valid outputs reliably. A natural and promising direction to address this issue is to adapt constrained decoding techniques to enforce grammatical correctness during generation. However, applying these techniques faces two primary obstacles. On the one hand, the non-autoregressive nature of dLLMs renders most existing constrained decoding approaches inapplicable. On the other hand, current approaches specifically designed for dLLMs may allow intermediate outputs that are impossible to complete into valid sentences, which significantly limits their reliability in practice.
To address these challenges, we present LAVE, a constrained decoding approach specifically designed for dLLMs. Our approach leverages a key property of dLLMs, namely their ability to predict token distributions for all positions in parallel during each forward pass. Whenever a new token is proposed by model, LAVE performs lookahead using these distributions to efficiently and reliably verify the validity of the proposed token. This design ensures reliable constraints by reliably preserving the potential for intermediate outputs to be extended into valid sentences. Extensive experiments across four widely used dLLMs and three representative benchmarks demonstrate that LAVE consistently outperforms existing baselines and achieves substantial improvements in syntactic correctness, while incurring negligible runtime overhead. 

---
# From Knowledge to Inference: Scaling Laws of Specialized Reasoning on GlobalHealthAtlas 

**Authors**: Zhaokun Yan, Zhaohan Liu, Wuzheng Dong, Lijie Feng, Chengxiao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2602.00491)  

**Abstract**: Public health reasoning requires population level inference grounded in scientific evidence, expert consensus, and safety constraints. However, it remains underexplored as a structured machine learning problem with limited supervised signals and benchmarks. We introduce \textbf{GlobalHealthAtlas}, a large scale multilingual dataset of 280,210 instances spanning 15 public health domains and 17 languages, stratified into three difficulty levels from health literacy to epidemiological and policy reasoning. Instances are derived from openly available public health sources and labeled by language, domain, and difficulty to support supervised learning and slice based evaluation. We further propose large language model (LLM) assisted construction and quality control pipeline with retrieval, duplication, evidence grounding checks, and label validation to improve consistency at scale. Finally, we present a domain aligned evaluator distilled from high confidence judgments of diverse LLMs to assess outputs along six dimensions: Accuracy, Reasoning, Completeness, Consensus Alignment, Terminology Norms, and Insightfulness. Together, these contributions enable reproducible training and evaluation of LLMs for safety critical public health reasoning beyond conventional QA benchmarks. 

---
# Intention-Adaptive LLM Fine-Tuning for Text Revision Generation 

**Authors**: Zhexiong Liu, Diane Litman  

**Link**: [PDF](https://arxiv.org/pdf/2602.00477)  

**Abstract**: Large Language Models (LLMs) have achieved impressive capabilities in various context-based text generation tasks, such as summarization and reasoning; however, their applications in intention-based generation tasks remain underexplored. One such example is revision generation, which requires the generated text to explicitly reflect the writer's actual intentions. Identifying intentions and generating desirable revisions are challenging due to their complex and diverse nature. Although prior work has employed LLMs to generate revisions with few-shot learning, they struggle with handling entangled multi-intent scenarios. While fine-tuning LLMs using intention-based instructions appears promising, it demands large amounts of annotated data, which is expensive and scarce in the revision community. To address these challenges, we propose Intention-Tuning, an intention-adaptive layer-wise LLM fine-tuning framework that dynamically selects a subset of LLM layers to learn the intentions and subsequently transfers their representations to revision generation. Experimental results suggest that Intention-Tuning is effective and efficient on small revision corpora, outperforming several PEFT baselines. 

---
# Reasoning by Commented Code for Table Question Answering 

**Authors**: Seho Pyo, Jiheon Seok, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.00543)  

**Abstract**: Table Question Answering (TableQA) poses a significant challenge for large language models (LLMs) because conventional linearization of tables often disrupts the two-dimensional relationships intrinsic to structured data. Existing methods, which depend on end-to-end answer generation or single-line program queries, typically exhibit limited numerical accuracy and reduced interpretability. This work introduces a commented, step-by-step code-generation framework that incorporates explicit reasoning into the Python program-generation process. The approach decomposes TableQA reasoning into multi-line executable programs with concise natural language comments, thereby promoting clearer reasoning and increasing the likelihood of generating correct code. On the WikiTableQuestions benchmark, the proposed method achieves 70.9\% accuracy using Qwen2.5-Coder-7B-Instruct, surpassing the Repanda baseline (67.6\%). Integrating the proposed framework with a robust end-to-end TableQA model via a lightweight answer-selection mechanism yields further improvements. This combined approach achieves up to 84.3\% accuracy on the WikiTableQuestions benchmark. 

---
# LegalOne: A Family of Foundation Models for Reliable Legal Reasoning 

**Authors**: Haitao Li, Yifan Chen, Shuo Miao, Qian Dong, Jia Chen, Yiran Hu, Junjie Chen, Minghao Qin, Qingyao Ai, Yiqun Liu, Cheng Luo, Quan Zhou, Ya Zhang, Jikun Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00642)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive general capabilities, their direct application in the legal domain is often hindered by a lack of precise domain knowledge and complexity of performing rigorous multi-step judicial reasoning. To address this gap, we present LegalOne, a family of foundational models specifically tailored for the Chinese legal domain. LegalOne is developed through a comprehensive three-phase pipeline designed to master legal reasoning. First, during mid-training phase, we propose Plasticity-Adjusted Sampling (PAS) to address the challenge of domain adaptation. This perplexity-based scheduler strikes a balance between the acquisition of new knowledge and the retention of original capabilities, effectively establishing a robust legal foundation. Second, during supervised fine-tuning, we employ Legal Agentic CoT Distillation (LEAD) to distill explicit reasoning from raw legal texts. Unlike naive distillation, LEAD utilizes an agentic workflow to convert complex judicial processes into structured reasoning trajectories, thereby enforcing factual grounding and logical rigor. Finally, we implement a Curriculum Reinforcement Learning (RL) strategy. Through a progressive reinforcement process spanning memorization, understanding, and reasoning, LegalOne evolves from simple pattern matching to autonomous and reliable legal reasoning. Experimental results demonstrate that LegalOne achieves state-of-the-art performance across a wide range of legal tasks, surpassing general-purpose LLMs with vastly larger parameter counts through enhanced knowledge density and efficiency. We publicly release the LegalOne weights and the LegalKit evaluation framework to advance the field of Legal AI, paving the way for deploying trustworthy and interpretable foundation models in high-stakes judicial applications. 

---
# Hermes the Polyglot: A Unified Framework to Enhance Expressiveness for Multimodal Interlingual Subtitling 

**Authors**: Chaoqun Cui, Shijing Wang, Liangbin Huang, Qingqing Gu, Zhaolong Huang, Xiao Zeng, Wenji Mao  

**Link**: [PDF](https://arxiv.org/pdf/2602.00597)  

**Abstract**: Interlingual subtitling, which translates subtitles of visual media into a target language, is essential for entertainment localization but has not yet been explored in machine translation. Although Large Language Models (LLMs) have significantly advanced the general capabilities of machine translation, the distinctive characteristics of subtitle texts pose persistent challenges in interlingual subtitling, particularly regarding semantic coherence, pronoun and terminology translation, and translation expressiveness. To address these issues, we present Hermes, an LLM-based automated subtitling framework. Hermes integrates three modules: Speaker Diarization, Terminology Identification, and Expressiveness Enhancement, which effectively tackle the above challenges. Experiments demonstrate that Hermes achieves state-of-the-art diarization performance and generates expressive, contextually coherent translations, thereby advancing research in interlingual subtitling. 

---
# What Matters to an LLM? Behavioral and Computational Evidences from Summarization 

**Authors**: Yongxin Zhou, Changshun Wu, Philippe Mulhem, Didier Schwab, Maxime Peyrard  

**Link**: [PDF](https://arxiv.org/pdf/2602.00459)  

**Abstract**: Large Language Models (LLMs) are now state-of-the-art at summarization, yet the internal notion of importance that drives their information selections remains hidden. We propose to investigate this by combining behavioral and computational analyses. Behaviorally, we generate a series of length-controlled summaries for each document and derive empirical importance distributions based on how often each information unit is selected. These reveal that LLMs converge on consistent importance patterns, sharply different from pre-LLM baselines, and that LLMs cluster more by family than by size. Computationally, we identify that certain attention heads align well with empirical importance distributions, and that middle-to-late layers are strongly predictive of importance. Together, these results provide initial insights into what LLMs prioritize in summarization and how this priority is internally represented, opening a path toward interpreting and ultimately controlling information selection in these models. 

---
# When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems 

**Authors**: Naen Xu, Hengyu An, Shuo Shi, Jinghuai Zhang, Chunyi Zhou, Changjiang Li, Tianyu Du, Zhihui Fu, Jun Wang, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2602.00428)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly enhanced the capabilities of collaborative multi-agent systems, enabling them to address complex challenges. However, within these multi-agent systems, the susceptibility of agents to collective cognitive biases remains an underexplored issue. A compelling example is the Mandela effect, a phenomenon where groups collectively misremember past events as a result of false details reinforced through social influence and internalized misinformation. This vulnerability limits our understanding of memory bias in multi-agent systems and raises ethical concerns about the potential spread of misinformation. In this paper, we conduct a comprehensive study on the Mandela effect in LLM-based multi-agent systems, focusing on its existence, causing factors, and mitigation strategies. We propose MANBENCH, a novel benchmark designed to evaluate agent behaviors across four common task types that are susceptible to the Mandela effect, using five interaction protocols that vary in agent roles and memory timescales. We evaluate agents powered by several LLMs on MANBENCH to quantify the Mandela effect and analyze how different factors affect it. Moreover, we propose strategies to mitigate this effect, including prompt-level defenses (e.g., cognitive anchoring and source scrutiny) and model-level alignment-based defense, achieving an average 74.40% reduction in the Mandela effect compared to the baseline. Our findings provide valuable insights for developing more resilient and ethically aligned collaborative multi-agent systems. 

---
# Clause-Internal or Clause-External? Testing Turkish Reflexive Binding in Adapted versus Chain of Thought Large Language Models 

**Authors**: Sercan Karakaş  

**Link**: [PDF](https://arxiv.org/pdf/2602.00380)  

**Abstract**: This study evaluates whether state-of-the-art large language models capture the binding relations of Turkish reflexive pronouns. We construct a balanced set of 100 sentences that pit local against non-local antecedents for the reflexives kendi and kendisi, and test two contrasting systems: an OpenAI chain-of-thought model designed for multi-step reasoning and Trendyol-LLM-7B-base-v0.1, a LLaMA-2-derived model extensively fine-tuned on Turkish data. Antecedent choice is assessed using a combined sentence-level perplexity and forced-choice paradigm. Trendyol-LLM favours local bindings in approximately 70% of trials, exhibiting a strong locality bias, whereas o1 Mini distributes its choices almost evenly between local and long-distance readings, revealing a marked contrast in binding behaviour across the two systems. 

---
# Benchmarking Uncertainty Calibration in Large Language Model Long-Form Question Answering 

**Authors**: Philip Müller, Nicholas Popovič, Michael Färber, Peter Steinbach  

**Link**: [PDF](https://arxiv.org/pdf/2602.00279)  

**Abstract**: Large Language Models (LLMs) are commonly used in Question Answering (QA) settings, increasingly in the natural sciences if not science at large. Reliable Uncertainty Quantification (UQ) is critical for the trustworthy uptake of generated answers. Existing UQ approaches remain weakly validated in scientific QA, a domain relying on fact-retrieval and reasoning capabilities. We introduce the first large-scale benchmark for evaluating UQ metrics in reasoning-demanding QA studying calibration of UQ methods, providing an extensible open-source framework to reproducibly assess calibration. Our study spans up to 20 large language models of base, instruction-tuned and reasoning variants. Our analysis covers seven scientific QA datasets, including both multiple-choice and arithmetic question answering tasks, using prompting to emulate an open question answering setting. We evaluate and compare methods representative of prominent approaches on a total of 685,000 long-form responses, spanning different reasoning complexities representative of domain-specific tasks. At the token level, we find that instruction tuning induces strong probability mass polarization, reducing the reliability of token-level confidences as estimates of uncertainty. Models further fine-tuned for reasoning are exposed to the same effect, but the reasoning process appears to mitigate it depending on the provider. At the sequence level, we show that verbalized approaches are systematically biased and poorly correlated with correctness, while answer frequency (consistency across samples) yields the most reliable calibration. In the wake of our analysis, we study and report the misleading effect of relying exclusively on ECE as a sole measure for judging performance of UQ methods on benchmark datasets. Our findings expose critical limitations of current UQ methods for LLMs and standard practices in benchmarking thereof. 

---
# MiNER: A Two-Stage Pipeline for Metadata Extraction from Municipal Meeting Minutes 

**Authors**: Rodrigo Batista, Luís Filipe Cunha, Purificação Silvano, Nuno Guimarães, Alípio Jorge, Evelin Amorim, Ricardo Campos  

**Link**: [PDF](https://arxiv.org/pdf/2602.00316)  

**Abstract**: Municipal meeting minutes are official documents of local governance, exhibiting heterogeneous formats and writing styles. Effective information retrieval (IR) requires identifying metadata such as meeting number, date, location, participants, and start/end times, elements that are rarely standardized or easy to extract automatically. Existing named entity recognition (NER) models are ill-suited to this task, as they are not adapted to such domain-specific categories. In this paper, we propose a two-stage pipeline for metadata extraction from municipal minutes. First, a question answering (QA) model identifies the opening and closing text segments containing metadata. Transformer-based models (BERTimbau and XLM-RoBERTa with and without a CRF layer) are then applied for fine-grained entity extraction and enhanced through deslexicalization. To evaluate our proposed pipeline, we benchmark both open-weight (Phi) and closed-weight (Gemini) LLMs, assessing predictive performance, inference cost, and carbon footprint. Our results demonstrate strong in-domain performance, better than larger general-purpose LLMs. However, cross-municipality evaluation reveals reduced generalization reflecting the variability and linguistic complexity of municipal records. This work establishes the first benchmark for metadata extraction from municipal meeting minutes, providing a solid foundation for future research in this domain. 

---
# DecompressionLM: Deterministic, Diagnostic, and Zero-Shot Concept Graph Extraction from Language Models 

**Authors**: Zhaochen Hong, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2602.00377)  

**Abstract**: Existing knowledge probing methods rely on pre-defined queries, limiting extraction to known concepts. We introduce DecompressionLM, a stateless framework for zero-shot concept graph extraction that discovers what language models encode without pre-specified queries or shared cross-sequence state. Our method targets three limitations of common decoding-based probing approaches: cross-sequence coupling that concentrates probability mass on high-frequency prefixes, competitive decoding effects that suppress long-tail concepts, and scalability constraints arising from sequential exploration. Using Van der Corput low-discrepancy sequences with arithmetic decoding, DecompressionLM enables deterministic, embarrassingly parallel generation without shared state across sequences. Across two model families and five quantization variants, we find that activation-aware quantization (AWQ-4bit) expands concept coverage by 30-170%, while uniform quantization (GPTQ-Int4) induces 71-86% coverage collapse -- divergent behaviors not reliably reflected by explanation-level perplexity. Corpus-based verification further reveals a 17-point hallucination gap between top- and bottom-ranked MMLU-Pro Law models. DecompressionLM establishes concept coverage as a complementary evaluation dimension for assessing knowledge breadth and factual grounding in compressed models useful for their deployment. 

---
# Faithful-Patchscopes: Understanding and Mitigating Model Bias in Hidden Representations Explanation of Large Language Models 

**Authors**: Xilin Gong, Shu Yang, Zehua Cao, Lynne Billard, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.00300)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities for hidden representation interpretation through Patchscopes, a framework that uses LLMs themselves to generate human-readable explanations by decoding from internal hidden representations. However, our work shows that LLMs tend to rely on inherent linguistic patterns, which can override contextual information encoded in the hidden representations during decoding. For example, even when a hidden representation encodes the contextual attribute "purple" for "broccoli", LLMs still generate "green" in their explanations, reflecting a strong prior association. This behavior reveals a systematic unfaithfulness in Patchscopes. To systematically study this issue, we first designed a dataset to evaluate the faithfulness of Patchscopes under biased cases, and our results show that there is an 18.84\% faithfulness decrease on average. We then propose Bias Alignment through Logit Recalibration (BALOR), which treats the output logits from an unpatched prompt as capturing model bias and contrasts them with logits obtained under patched contextual information. By recalibrating the logit distribution through this contrast, BALOR suppresses model bias and amplifies contextual information during generation. Experiments across multiple LLMs demonstrate that BALOR consistently outperforms existing baselines, achieving up to 33\% relative performance improvement. 

---
# G-MemLLM: Gated Latent Memory Augmentation for Long-Context Reasoning in Large Language Models 

**Authors**: Xun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00015)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding, yet they remain constrained by the finite capacity of their context windows and the inherent difficulty of maintaining long-term factual consistency during multi-hop reasoning. While existing methods utilize context compression or recurrent tokens, they often suffer from ``context rot'' or the dilution of information over long horizons. In this paper, we propose \textbf{G-MemLLM}, a memory-augmented architecture that integrates a frozen LLM backbone with a trainable \textbf{Latent Memory Bank}. Our key innovation is a GRU-style gated update logic that allows the model to selectively update, preserve, or overwrite latent memory slots, preventing the vanishing gradients of knowledge common in recurrent systems. We evaluate G-MemLLM across scales, from GPT-2 (124M) to Llama 3.1 (8B), on the HotpotQA and Zero-Shot Relation Extraction (ZsRE) benchmarks. Our results demonstrate that G-MemLLM significantly enhances multi-hop reasoning and relational precision, achieving a 13.3\% accuracy boost on ZsRE for Llama 3.1-8B, and it also yields improvements across model scales, boosting Answer F1 by 8.56 points for GPT-2 and increasing Supporting Fact F1 by 6.89 points for Llama 3.1-8B on HotpotQA. 

---
# PTCBENCH: Benchmarking Contextual Stability of Personality Traits in LLM Systems 

**Authors**: Jiongchi Yu, Yuhan Ma, Xiaoyu Zhang, Junjie Wang, Qiang Hu, Chao Shen, Xiaofei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2602.00016)  

**Abstract**: With the increasing deployment of large language models (LLMs) in affective agents and AI systems, maintaining a consistent and authentic LLM personality becomes critical for user trust and engagement. However, existing work overlooks a fundamental psychological consensus that personality traits are dynamic and context-dependent. To bridge this gap, we introduce PTCBENCH, a systematic benchmark designed to quantify the consistency of LLM personalities under controlled situational contexts. PTCBENCH subjects models to 12 distinct external conditions spanning diverse location contexts and life events, and rigorously assesses the personality using the NEO Five-Factor Inventory. Our study on 39,240 personality trait records reveals that certain external scenarios (e.g., "Unemployment") can trigger significant personality changes of LLMs, and even alter their reasoning capabilities. Overall, PTCBENCH establishes an extensible framework for evaluating personality consistency in realistic, evolving environments, offering actionable insights for developing robust and psychologically aligned AI systems. 

---
# Segment-Level Attribution for Selective Learning of Long Reasoning Traces 

**Authors**: Siyuan Wang, Yanchen Liu, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2602.00425)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong reasoning performance by generating long chains of thought (CoTs), yet only a small fraction of these traces meaningfully contributes to answer prediction, while the majority contains repetitive or truncated content. Such output redundancy is further propagated after supervised finetuning (SFT), as models learn to imitate verbose but uninformative patterns, which can degrade performance. To this end, we incorporate integrated gradient attribution to quantify each token's influence on final answers and aggregate them into two segment-level metrics: (1) \textit{attribution strength} measures the overall attribution magnitude; and (2) \textit{direction consistency} captures whether tokens' attributions within a segment are uniformly positive or negative (high consistency), or a mixture of both (moderate consistency). Based on these two metrics, we propose a segment-level selective learning framework to identify important segments with high attribution strength but moderate consistency that indicate reflective rather than shallow reasoning. The framework then applies selective SFT on these important segments while masking loss for unimportant ones. Experiments across multiple models and datasets show that our approach improves accuracy and output efficiency, enabling more effective learning from long reasoning traces~\footnote{Code and data are available at this https URL}. 

---
# Construct, Align, and Reason: Large Ontology Models for Enterprise Knowledge Management 

**Authors**: Yao Zhang, Hongyin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00029)  

**Abstract**: Enterprise-scale knowledge management faces significant challenges in integrating multi-source heterogeneous data and enabling effective semantic reasoning. Traditional knowledge graphs often struggle with implicit relationship discovery and lack sufficient semantic understanding for complex question answering. To address these limitations, we introduce a unified construct--align--reason framework, the large ontology model (LOM). We first build a dual-layer enterprise ontology from structured databases and unstructured text, subsequently fusing these sources into a comprehensive enterprise ontology. To enable instruction-aligned reasoning, we propose a unified three-stage training pipeline: ontology instruction fine-tuning to improve structural understanding; text-ontology grounding to strengthen node semantic encoding; and multi-task instruction tuning on ontology-language pairs with curriculum learning to enhance semantic reasoning and generation. We also construct comprehensive training and evaluation datasets covering diverse ontology reasoning tasks. On this benchmark, our 4B-parameter LOM achieves 89.47% accuracy and outperforms DeepSeek-V3.2 on complex graph reasoning, indicating effective fusion of ontology structure and language. 

---
# Interpreting and Controlling LLM Reasoning through Integrated Policy Gradient 

**Authors**: Changming Li, Kaixing Zhang, Haoyun Xu, Yingdong Shi, Zheng Zhang, Kaitao Song, Kan Ren  

**Link**: [PDF](https://arxiv.org/pdf/2602.02313)  

**Abstract**: Large language models (LLMs) demonstrate strong reasoning abilities in solving complex real-world problems. Yet, the internal mechanisms driving these complex reasoning behaviors remain opaque. Existing interpretability approaches targeting reasoning either identify components (e.g., neurons) correlated with special textual patterns, or rely on human-annotated contrastive pairs to derive control vectors. Consequently, current methods struggle to precisely localize complex reasoning mechanisms or capture sequential influence from model internal workings to the reasoning outputs. In this paper, built on outcome-oriented and sequential-influence-aware principles, we focus on identifying components that have sequential contribution to reasoning behavior where outcomes are cumulated by long-range effects. We propose Integrated Policy Gradient (IPG), a novel framework that attributes reasoning behaviors to model's inner components by propagating compound outcome-based signals such as post reasoning accuracy backward through model inference trajectories. Empirical evaluations demonstrate that our approach achieves more precise localization and enables reliable modulation of reasoning behaviors (e.g., reasoning capability, reasoning strength) across diverse reasoning models. 

---
# RACA: Representation-Aware Coverage Criteria for LLM Safety Testing 

**Authors**: Zeming Wei, Zhixin Zhang, Chengcan Wu, Yihao Zhang, Xiaokun Luan, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.02280)  

**Abstract**: Recent advancements in LLMs have led to significant breakthroughs in various AI applications. However, their sophisticated capabilities also introduce severe safety concerns, particularly the generation of harmful content through jailbreak attacks. Current safety testing for LLMs often relies on static datasets and lacks systematic criteria to evaluate the quality and adequacy of these tests. While coverage criteria have been effective for smaller neural networks, they are not directly applicable to LLMs due to scalability issues and differing objectives. To address these challenges, this paper introduces RACA, a novel set of coverage criteria specifically designed for LLM safety testing. RACA leverages representation engineering to focus on safety-critical concepts within LLMs, thereby reducing dimensionality and filtering out irrelevant information. The framework operates in three stages: first, it identifies safety-critical representations using a small, expert-curated calibration set of jailbreak prompts. Second, it calculates conceptual activation scores for a given test suite based on these representations. Finally, it computes coverage results using six sub-criteria that assess both individual and compositional safety concepts. We conduct comprehensive experiments to validate RACA's effectiveness, applicability, and generalization, where the results demonstrate that RACA successfully identifies high-quality jailbreak prompts and is superior to traditional neuron-level criteria. We also showcase its practical application in real-world scenarios, such as test set prioritization and attack prompt sampling. Furthermore, our findings confirm RACA's generalization to various scenarios and its robustness across various configurations. Overall, RACA provides a new framework for evaluating the safety of LLMs, contributing a valuable technique to the field of testing for AI. 

---
# Drift-Bench: Diagnosing Cooperative Breakdowns in LLM Agents under Input Faults via Multi-Turn Interaction 

**Authors**: Han Bao, Zheyuan Zhang, Pengcheng Jing, Zhengqing Yuan, Kaiwen Shi, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2602.02455)  

**Abstract**: As Large Language Models transition to autonomous agents, user inputs frequently violate cooperative assumptions (e.g., implicit intent, missing parameters, false presuppositions, or ambiguous expressions), creating execution risks that text-only evaluations do not capture. Existing benchmarks typically assume well-specified instructions or restrict evaluation to text-only, single-turn clarification, and thus do not measure multi-turn disambiguation under grounded execution risk. We introduce \textbf{Drift-Bench}, the first diagnostic benchmark that evaluates agentic pragmatics under input faults through multi-turn clarification across state-oriented and service-oriented execution environments. Grounded in classical theories of communication, \textbf{Drift-Bench} provides a unified taxonomy of cooperative breakdowns and employs a persona-driven user simulator with the \textbf{Rise} evaluation protocol. Experiments show substantial performance drops under these faults, with clarification effectiveness varying across user personas and fault types. \MethodName bridges clarification research and agent safety evaluation, enabling systematic diagnosis of failures that can lead to unsafe executions. 

---
# Learning While Staying Curious: Entropy-Preserving Supervised Fine-Tuning via Adaptive Self-Distillation for Large Reasoning Models 

**Authors**: Hao Wang, Hao Gu, Hongming Piao, Kaixiong Gong, Yuxiao Ye, Xiangyu Yue, Sirui Han, Yike Guo, Dapeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.02244)  

**Abstract**: The standard post-training recipe for large reasoning models, supervised fine-tuning followed by reinforcement learning (SFT-then-RL), may limit the benefits of the RL stage: while SFT imitates expert demonstrations, it often causes overconfidence and reduces generation diversity, leaving RL with a narrowed solution space to explore. Adding entropy regularization during SFT is not a cure-all; it tends to flatten token distributions toward uniformity, increasing entropy without improving meaningful exploration capability. In this paper, we propose CurioSFT, an entropy-preserving SFT method designed to enhance exploration capabilities through intrinsic curiosity. It consists of (a) Self-Exploratory Distillation, which distills the model toward a self-generated, temperature-scaled teacher to encourage exploration within its capability; and (b) Entropy-Guided Temperature Selection, which adaptively adjusts distillation strength to mitigate knowledge forgetting by amplifying exploration at reasoning tokens while stabilizing factual tokens. Extensive experiments on mathematical reasoning tasks demonstrate that, in SFT stage, CurioSFT outperforms the vanilla SFT by 2.5 points on in-distribution tasks and 2.9 points on out-of-distribution tasks. We also verify that exploration capabilities preserved during SFT successfully translate into concrete gains in RL stage, yielding an average improvement of 5.0 points. 

---
# No Global Plan in Chain-of-Thought: Uncover the Latent Planning Horizon of LLMs 

**Authors**: Liyan Xu, Mo Yu, Fandong Meng, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.02103)  

**Abstract**: This work stems from prior complementary observations on the dynamics of Chain-of-Thought (CoT): Large Language Models (LLMs) is shown latent planning of subsequent reasoning prior to CoT emergence, thereby diminishing the significance of explicit CoT; whereas CoT remains critical for tasks requiring multi-step reasoning. To deepen the understanding between LLM's internal states and its verbalized reasoning trajectories, we investigate the latent planning strength of LLMs, through our probing method, Tele-Lens, applying to hidden states across diverse task domains. Our empirical results indicate that LLMs exhibit a myopic horizon, primarily conducting incremental transitions without precise global planning. Leveraging this characteristic, we propose a hypothesis on enhancing uncertainty estimation of CoT, which we validate that a small subset of CoT positions can effectively represent the uncertainty of the entire path. We further underscore the significance of exploiting CoT dynamics, and demonstrate that automatic recognition of CoT bypass can be achieved without performance degradation. Our code, data and models are released at this https URL. 

---
# Learning Generative Selection for Best-of-N 

**Authors**: Shubham Toshniwal, Aleksander Ficek, Siddhartha Jain, Wei Du, Vahid Noroozi, Sadegh Mahdavi, Somshubra Majumdar, Igor Gitman  

**Link**: [PDF](https://arxiv.org/pdf/2602.02143)  

**Abstract**: Scaling test-time compute via parallel sampling can substantially improve LLM reasoning, but is often limited by Best-of-N selection quality. Generative selection methods, such as GenSelect, address this bottleneck, yet strong selection performance remains largely limited to large models. We show that small reasoning models can acquire strong GenSelect capabilities through targeted reinforcement learning. To this end, we synthesize selection tasks from large-scale math and code instruction datasets by filtering to instances with both correct and incorrect candidate solutions, and train 1.7B-parameter models with DAPO to reward correct selections. Across math (AIME24, AIME25, HMMT25) and code (LiveCodeBench) reasoning benchmarks, our models consistently outperform prompting and majority-voting baselines, often approaching or exceeding much larger models. Moreover, these gains generalize to selecting outputs from stronger models despite training only on outputs from weaker models. Overall, our results establish reinforcement learning as a scalable way to unlock strong generative selection in small models, enabling efficient test-time scaling. 

---
# PRISM: Parametrically Refactoring Inference for Speculative Sampling Draft Models 

**Authors**: Xuliang Wang, Yuetao Chen, Maochan Zhen, Fang Liu, Xinzhou Zheng, Xingwu Liu, Hong Xu, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.01762)  

**Abstract**: Large Language Models (LLMs), constrained by their auto-regressive nature, suffer from slow decoding. Speculative decoding methods have emerged as a promising solution to accelerate LLM decoding, attracting attention from both systems and AI research communities. Recently, the pursuit of better draft quality has driven a trend toward parametrically larger draft models, which inevitably introduces substantial computational overhead. While existing work attempts to balance the trade-off between prediction accuracy and compute latency, we address this fundamental dilemma through architectural innovation.
We propose PRISM, which disaggregates the computation of each predictive step across different parameter sets, refactoring the computational pathways of draft models to successfully decouple model capacity from inference cost. Through extensive experiments, we demonstrate that PRISM outperforms all existing draft architectures, achieving exceptional acceptance lengths while maintaining minimal draft latency for superior end-to-end speedup. We also re-examine scaling laws with PRISM, revealing that PRISM scales more effectively with expanding data volumes than other draft architectures. Through rigorous and fair comparison, we show that PRISM boosts the decoding throughput of an already highly optimized inference engine by more than 2.6x. 

---
# Expected Harm: Rethinking Safety Evaluation of (Mis)Aligned LLMs 

**Authors**: Yen-Shan Chen, Zhi Rui Tam, Cheng-Kuang Wu, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.01600)  

**Abstract**: Current evaluations of LLM safety predominantly rely on severity-based taxonomies to assess the harmfulness of malicious queries. We argue that this formulation requires re-examination as it assumes uniform risk across all malicious queries, neglecting Execution Likelihood--the conditional probability of a threat being realized given the model's response. In this work, we introduce Expected Harm, a metric that weights the severity of a jailbreak by its execution likelihood, modeled as a function of execution cost. Through empirical analysis of state-of-the-art models, we reveal a systematic Inverse Risk Calibration: models disproportionately exhibit stronger refusal behaviors for low-likelihood (high-cost) threats while remaining vulnerable to high-likelihood (low-cost) queries. We demonstrate that this miscalibration creates a structural vulnerability: by exploiting this property, we increase the attack success rate of existing jailbreaks by up to $2\times$. Finally, we trace the root cause of this failure using linear probing, which reveals that while models encode severity in their latent space to drive refusal decisions, they possess no distinguishable internal representation of execution cost, making them "blind" to this critical dimension of risk. 

---
# MAGIC: A Co-Evolving Attacker-Defender Adversarial Game for Robust LLM Safety 

**Authors**: Xiaoyu Wen, Zhida He, Han Qi, Ziyu Wan, Zhongtian Ma, Ying Wen, Tianhang Zheng, Xingcheng Xu, Chaochao Lu, Qiaosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.01539)  

**Abstract**: Ensuring robust safety alignment is crucial for Large Language Models (LLMs), yet existing defenses often lag behind evolving adversarial attacks due to their \textbf{reliance on static, pre-collected data distributions}. In this paper, we introduce \textbf{MAGIC}, a novel multi-turn multi-agent reinforcement learning framework that formulates LLM safety alignment as an adversarial asymmetric game. Specifically, an attacker agent learns to iteratively rewrite original queries into deceptive prompts, while a defender agent simultaneously optimizes its policy to recognize and refuse such inputs. This dynamic process triggers a \textbf{co-evolution}, where the attacker's ever-changing strategies continuously uncover long-tail vulnerabilities, driving the defender to generalize to unseen attack patterns. Remarkably, we observe that the attacker, endowed with initial reasoning ability, evolves \textbf{novel, previously unseen combinatorial strategies} through iterative RL training, underscoring our method's substantial potential. Theoretically, we provide insights into a more robust game equilibrium and derive safety guarantees. Extensive experiments validate our framework's effectiveness, demonstrating superior defense success rates without compromising the helpfulness of the model. Our code is available at this https URL. 

---
# A-MapReduce: Executing Wide Search via Agentic MapReduce 

**Authors**: Mingju Chen, Guibin Zhang, Heng Chang, Yuchen Guo, Shiji Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.01331)  

**Abstract**: Contemporary large language model (LLM)-based multi-agent systems exhibit systematic advantages in deep research tasks, which emphasize iterative, vertically structured information seeking. However, when confronted with wide search tasks characterized by large-scale, breadth-oriented retrieval, existing agentic frameworks, primarily designed around sequential, vertically structured reasoning, remain stuck in expansive search objectives and inefficient long-horizon execution. To bridge this gap, we propose A-MapReduce, a MapReduce paradigm-inspired multi-agent execution framework that recasts wide search as a horizontally structured retrieval problem. Concretely, A-MapReduce implements parallel processing of massive retrieval targets through task-adaptive decomposition and structured result aggregation. Meanwhile, it leverages experiential memory to drive the continual evolution of query-conditioned task allocation and recomposition, enabling progressive improvement in large-scale wide-search regimes. Extensive experiments on five agentic benchmarks demonstrate that A-MapReduce is (i) high-performing, achieving state-of-the-art performance on WideSearch and DeepWideSearch, and delivering 5.11% - 17.50% average Item F1 improvements compared with strong baselines with OpenAI o3 or Gemini 2.5 Pro backbones; (ii) cost-effective and efficient, delivering superior cost-performance trade-offs and reducing running time by 45.8\% compared to representative multi-agent baselines. The code is available at this https URL. 

---
# HalluHard: A Hard Multi-Turn Hallucination Benchmark 

**Authors**: Dongyang Fan, Sebastien Delsad, Nicolas Flammarion, Maksym Andriushchenko  

**Link**: [PDF](https://arxiv.org/pdf/2602.01031)  

**Abstract**: Large language models (LLMs) still produce plausible-sounding but ungrounded factual claims, a problem that worsens in multi-turn dialogue as context grows and early errors cascade. We introduce $\textbf{HalluHard}$, a challenging multi-turn hallucination benchmark with 950 seed questions spanning four high-stakes domains: legal cases, research questions, medical guidelines, and coding. We operationalize groundedness by requiring inline citations for factual assertions. To support reliable evaluation in open-ended settings, we propose a judging pipeline that iteratively retrieves evidence via web search. It can fetch, filter, and parse full-text sources (including PDFs) to assess whether cited material actually supports the generated content. Across a diverse set of frontier proprietary and open-weight models, hallucinations remain substantial even with web search ($\approx 30\%$ for the strongest configuration, Opus-4.5 with web search), with content-grounding errors persisting at high rates. Finally, we show that hallucination behavior is shaped by model capacity, turn position, effective reasoning, and the type of knowledge required. 

---
# Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning 

**Authors**: Dylan Zhang, Yufeng Xu, Haojin Wang, Qingzhi Chen, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01058)  

**Abstract**: Post-training of reasoning LLMs is a holistic process that typically consists of an offline SFT stage followed by an online reinforcement learning (RL) stage. However, SFT is often optimized in isolation to maximize SFT performance alone.
We show that, after identical RL training, models initialized from stronger SFT checkpoints can significantly underperform those initialized from weaker ones. We attribute this to a mismatch typical in current SFT-RL pipelines: the distribution that generates the offline SFT data can differ substantially from the policy optimized during online RL, which learns from its own rollouts.
We propose PEAR (Policy Evaluation-inspired Algorithm for Offline Learning Loss Re-weighting), an SFT-stage method that corrects this mismatch and better prepares the model for RL. PEAR uses importance sampling to reweight the SFT loss, with three variants operating at the token, block, and sequence levels. It can be used to augment standard SFT objectives and incurs little additional training overhead once probabilities for the offline data are collected.
We conduct controlled experiments on verifiable reasoning games and mathematical reasoning tasks on Qwen 2.5 and 3 and DeepSeek-distilled models. PEAR consistently improves post-RL performance over canonical SFT, with pass at 8 gains up to a 14.6 percent on AIME2025. Our results suggest that PEAR is an effective step toward more holistic LLM post-training by designing and evaluating SFT with downstream RL in mind rather than in isolation. 

---
# Probing the Knowledge Boundary: An Interactive Agentic Framework for Deep Knowledge Extraction 

**Authors**: Yuheng Yang, Siqi Zhu, Tao Feng, Ge Liu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2602.00959)  

**Abstract**: Large Language Models (LLMs) can be seen as compressed knowledge bases, but it remains unclear what knowledge they truly contain and how far their knowledge boundaries extend. Existing benchmarks are mostly static and provide limited support for systematic knowledge probing. In this paper, we propose an interactive agentic framework to systematically extract and quantify the knowledge of LLMs. Our method includes four adaptive exploration policies to probe knowledge at different granularities. To ensure the quality of extracted knowledge, we introduce a three-stage knowledge processing pipeline that combines vector-based filtering to remove exact duplicates, LLM-based adjudication to resolve ambiguous semantic overlaps, and domain-relevance auditing to retain valid knowledge units. Through extensive experiments, we find that recursive taxonomy is the most effective exploration strategy. We also observe a clear knowledge scaling law, where larger models consistently extract more knowledge. In addition, we identify a Pass@1-versus-Pass@k trade-off: domain-specialized models achieve higher initial accuracy but degrade rapidly, while general-purpose models maintain stable performance during extended extraction. Finally, our results show that differences in training data composition lead to distinct and measurable knowledge profiles across model families. 

---
# GradingAttack: Attacking Large Language Models Towards Short Answer Grading Ability 

**Authors**: Xueyi Li, Zhuoneng Zhou, Zitao Liu, Yongdong Wu, Weiqi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.00979)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable potential for automatic short answer grading (ASAG), significantly boosting student assessment efficiency and scalability in educational scenarios. However, their vulnerability to adversarial manipulation raises critical concerns about automatic grading fairness and reliability. In this paper, we introduce GradingAttack, a fine-grained adversarial attack framework that systematically evaluates the vulnerability of LLM based ASAG models. Specifically, we align general-purpose attack methods with the specific objectives of ASAG by designing token-level and prompt-level strategies that manipulate grading outcomes while maintaining high camouflage. Furthermore, to quantify attack camouflage, we propose a novel evaluation metric that balances attack success and camouflage. Experiments on multiple datasets demonstrate that both attack strategies effectively mislead grading models, with prompt-level attacks achieving higher success rates and token-level attacks exhibiting superior camouflage capability. Our findings underscore the need for robust defenses to ensure fairness and reliability in ASAG. Our code and datasets are available at this https URL. 

---
# Discovering Process-Outcome Credit in Multi-Step LLM Reasoning 

**Authors**: Xiangwei Wang, Wei Wang, Ken Chen, Nanduni Nimalsiri, Saman Halgamuge  

**Link**: [PDF](https://arxiv.org/pdf/2602.01034)  

**Abstract**: Reinforcement Learning (RL) serves as a potent paradigm for enhancing reasoning capabilities in Large Language Models (LLMs), yet standard outcome-based approaches often suffer from reward sparsity and inefficient credit assignment. In this paper, we propose a novel framework designed to provide continuous reward signals, which introduces a Step-wise Marginal Information Gain (MIG) mechanism that quantifies the intrinsic value of reasoning steps against a Monotonic Historical Watermark, effectively filtering out training noise. To ensure disentangled credit distribution, we implement a Decoupled Masking Strategy, applying process-oriented rewards specifically to the chain-of-thought (CoT) and outcome-oriented rewards to the full completion. Additionally, we incorporate a Dual-Gated SFT objective to stabilize training with high-quality structural and factual signals. Extensive experiments across textual and multi-modal benchmarks (e.g., MATH, Super-CLEVR) demonstrate that our approach consistently outperforms baselines such as GRPO in both sample efficiency and final accuracy. Furthermore, our model exhibits superior out-of-distribution robustness, demonstrating promising zero-shot transfer capabilities to unseen and challenging reasoning tasks. 

---
# Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing 

**Authors**: Anxin Guo, Jingwei Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.00906)  

**Abstract**: Large language models often hallucinate with high confidence on "random facts" that lack inferable patterns. We formalize the memorization of such facts as a membership testing problem, unifying the discrete error metrics of Bloom filters with the continuous log-loss of LLMs. By analyzing this problem in the regime where facts are sparse in the universe of plausible claims, we establish a rate-distortion theorem: the optimal memory efficiency is characterized by the minimum KL divergence between score distributions on facts and non-facts. This theoretical framework provides a distinctive explanation for hallucination: even with optimal training, perfect data, and a simplified "closed world" setting, the information-theoretically optimal strategy under limited capacity is not to abstain or forget, but to assign high confidence to some non-facts, resulting in hallucination. We validate this theory empirically on synthetic data, showing that hallucinations persist as a natural consequence of lossy compression. 

---
# LLMs as High-Dimensional Nonlinear Autoregressive Models with Attention: Training, Alignment and Inference 

**Authors**: Vikram Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2602.00426)  

**Abstract**: Large language models (LLMs) based on transformer architectures are typically described through collections of architectural components and training procedures, obscuring their underlying computational structure. This review article provides a concise mathematical reference for researchers seeking an explicit, equation-level description of LLM training, alignment, and generation. We formulate LLMs as high-dimensional nonlinear autoregressive models with attention-based dependencies. The framework encompasses pretraining via next-token prediction, alignment methods such as reinforcement learning from human feedback (RLHF), direct preference optimization (DPO), rejection sampling fine-tuning (RSFT), and reinforcement learning from verifiable rewards (RLVR), as well as autoregressive generation during inference. Self-attention emerges naturally as a repeated bilinear--softmax--linear composition, yielding highly expressive sequence models. This formulation enables principled analysis of alignment-induced behaviors (including sycophancy), inference-time phenomena (such as hallucination, in-context learning, chain-of-thought prompting, and retrieval-augmented generation), and extensions like continual learning, while serving as a concise reference for interpretation and further theoretical development. 

---
# Block removal for large language models through constrained binary optimization 

**Authors**: David Jansen, Roman Rausch, David Montero, Roman Orus  

**Link**: [PDF](https://arxiv.org/pdf/2602.00161)  

**Abstract**: Compressing resource-intensive large language models by removing whole transformer blocks is a seemingly simple idea, but identifying which blocks to remove constitutes an exponentially difficult combinatorial problem. In this paper, we formulate block removal as a constrained binary optimization problem that can be mapped to a physical system (Ising model), whose energies are a strong proxy for downstream model performance. This formulation enables an efficient ranking of a large number of candidate block-removal configurations and yields many high-quality, non-trivial solutions beyond consecutive regions. We demonstrate that our approach outperforms state-of-the-art block-removal methods across several benchmarks, with performance gains persisting after short retraining, and reaching improvements of up to 6 points on the MMLU benchmark. Our method requires only forward and backward passes for a few active parameters, together with an (at least approximate) Ising solver, and can be readily applied to any architecture. We illustrate this generality on the recent NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 model, which exhibits a highly inhomogeneous and challenging block structure. 

---
# Unmasking Reasoning Processes: A Process-aware Benchmark for Evaluating Structural Mathematical Reasoning in LLMs 

**Authors**: Xiang Zheng, Weiqi Zhai, Wei Wang, Boyu Yang, Wenbo Li, Ruixiang Luo, Haoxiang Sun, Yucheng Wang, Zhengze Li, Meng Wang, Yuetian Du, Guojie Lin, Yaxuan Wang, Xiaoxiao Xu, Yanhu Mo, Xuan Ren, Hu Wei, Ze Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00564)  

**Abstract**: Recent large language models (LLMs) achieve near-saturation accuracy on many established mathematical reasoning benchmarks, raising concerns about their ability to diagnose genuine reasoning competence. This saturation largely stems from the dominance of template-based computation and shallow arithmetic decomposition in existing datasets, which underrepresent reasoning skills such as multi-constraint coordination, constructive logical synthesis, and spatial inference. To address this gap, we introduce ReasoningMath-Plus, a benchmark of 150 carefully curated problems explicitly designed to evaluate structural reasoning. Each problem emphasizes reasoning under interacting constraints, constructive solution formation, or non-trivial structural insight, and is annotated with a minimal reasoning skeleton to support fine-grained process-level evaluation. Alongside the dataset, we introduce HCRS (Hazard-aware Chain-based Rule Score), a deterministic step-level scoring function, and train a Process Reward Model (PRM) on the annotated reasoning traces. Empirically, while leading models attain relatively high final-answer accuracy (up to 5.8/10), HCRS-based holistic evaluation yields substantially lower scores (average 4.36/10, best 5.14/10), showing that answer-only metrics can overestimate reasoning robustness. 

---
# When RAG Hurts: Diagnosing and Mitigating Attention Distraction in Retrieval-Augmented LVLMs 

**Authors**: Beidi Zhao, Wenlong Deng, Xinting Liao, Yushu Li, Nazim Shaikh, Yao Nie, Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.00344)  

**Abstract**: While Retrieval-Augmented Generation (RAG) is one of the dominant paradigms for enhancing Large Vision-Language Models (LVLMs) on knowledge-based VQA tasks, recent work attributes RAG failures to insufficient attention towards the retrieved context, proposing to reduce the attention allocated to image tokens. In this work, we identify a distinct failure mode that previous study overlooked: Attention Distraction (AD). When the retrieved context is sufficient (highly relevant or including the correct answer), the retrieved text suppresses the visual attention globally, and the attention on image tokens shifts away from question-relevant regions. This leads to failures on questions the model could originally answer correctly without the retrieved text. To mitigate this issue, we propose MAD-RAG, a training-free intervention that decouples visual grounding from context integration through a dual-question formulation, combined with attention mixing to preserve image-conditioned evidence. Extensive experiments on OK-VQA, E-VQA, and InfoSeek demonstrate that MAD-RAG consistently outperforms existing baselines across different model families, yielding absolute gains of up to 4.76%, 9.20%, and 6.18% over the vanilla RAG baseline. Notably, MAD-RAG rectifies up to 74.68% of failure cases with negligible computational overhead. 

---
# CARE-RFT: Confidence-Anchored Reinforcement Finetuning for Reliable Reasoning in Large Language Models 

**Authors**: Shuozhe Li, Jincheng Cao, Bodun Hu, Aryan Mokhtari, Leqi Liu, Amy Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.00085)  

**Abstract**: Reinforcement finetuning (RFT) has emerged as a powerful paradigm for unlocking reasoning capabilities in large language models. However, we identify a critical trade-off: while unconstrained RFT achieves strong reasoning performance, it severely compromises model trustworthiness by amplifying hallucination and worsening calibration; conversely, RKL-constrained RFT preserves trustworthiness but limits reasoning gains due to its unbounded penalty on exploratory deviations. To resolve this tension, we introduce CARE-RFT (Confidence-Anchored Regularized Reinforcement Finetuning), a novel method that replaces standard reverse KL regularization with a skew reverse KL divergence. CARE-RFT provides a confidence-sensitive penalty: it is bounded for confident, consistently rewarded explorations to enable reasoning, while unbounded elsewhere to preserve calibration. Extensive experiments across multiple model scales and RFT algorithms show that CARE-RFT achieves a superior balance, matching the reasoning performance of unconstrained RFT while recovering the trustworthiness and calibration of the base model. Our work establishes that careful, confidence-aware regularization is key to building both capable and trustworthy reasoning models. 

---
# Beyond Output Critique: Self-Correction via Task Distillation 

**Authors**: Hossein A. Rahmani, Mengting Wan, Pei Zhou, Longqi Yang, Nick Craswell, Emine Yilmaz, Sujay Kumar Jauhar  

**Link**: [PDF](https://arxiv.org/pdf/2602.00871)  

**Abstract**: Large language models (LLMs) have shown promising self-correction abilities, where iterative refinement improves the quality of generated responses. However, most existing approaches operate at the level of output critique, patching surface errors while often failing to correct deeper reasoning flaws. We propose SELF-THOUGHT, a framework that introduces an intermediate step of task abstraction before solution refinement. Given an input and an initial response, the model first distills the task into a structured template that captures key variables, constraints, and problem structure. This abstraction then guides solution instantiation, grounding subsequent responses in a clearer understanding of the task and reducing error propagation. Crucially, we show that these abstractions can be transferred across models: templates generated by larger models can serve as structured guides for smaller LLMs, which typically struggle with intrinsic self-correction. By reusing distilled task structures, smaller models achieve more reliable refinements without heavy fine-tuning or reliance on external verifiers. Experiments across diverse reasoning tasks demonstrate that SELF-THOUGHT improves accuracy, robustness, and generalization for both large and small models, offering a scalable path toward more reliable self-correcting language systems. 

---
# IntentCoding: Amplifying User Intent in Code Generation 

**Authors**: Zheng Fang, Yihong Dong, Lili Mou, Dongming Jin, Zhi Jin, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.00066)  

**Abstract**: Large Language Models (LLMs) have shown strong capabilities in code generation, but their adherence to fine-grained user intent with multiple constraints remains a significant challenge. Our empirical analysis reveals two key observations: 1) Model performance deteriorates quickly as the number of constraints in the user intent increases, and 2) While user intent does influence the model's logits, such an influence may not be strong enough to effectively steer the decoding process. To this end, we propose Intent-Amplified Code Generation (IntentCoding), a novel decoding strategy that enhances an LLM's ability to follow user intent. IntentCoding captures the influence of user intent by masking out the intent, and applies a multi-strength ensemble mechanism to amplify the effect of user intent during generation. IntentCoding is model-agnostic, requires no additional training, and integrates seamlessly with existing decoding procedures. To enable systematic evaluation, we also construct CodeConstraints, a benchmark dataset specifically designed to test user intent compliance under varying numbers of constraints. Experiments on our constructed Constraints, as well as popular IFEvalCode, HumanEval and LiveCodeBench datasets, show that our IntentCoding model significantly improves both constraint satisfaction and functional correctness compared to standard decoding approaches. IntentCoding achieves up to 71.0% relative improvement on CodeConstraints, achieves up to 67.3% relative improvement on IFEvalCode and achieves up to 29.3% relative improvement in pass@1 on HumanEval and LiveCodeBench compared with greedy decoding. 

---
# FoundationalASSIST: An Educational Dataset for Foundational Knowledge Tracing and Pedagogical Grounding of LLMs 

**Authors**: Eamon Worden, Cristina Heffernan, Neil Heffernan, Shashank Sonkar  

**Link**: [PDF](https://arxiv.org/pdf/2602.00070)  

**Abstract**: Can Large Language Models understand how students learn? As LLMs are deployed for adaptive testing and personalized tutoring, this question becomes urgent -- yet we cannot answer it with existing resources. Current educational datasets provide only question identifiers and binary correctness labels, rendering them opaque to LLMs that reason in natural language. We address this gap with FoundationalASSIST, the first English educational dataset providing the complete information needed for research on LLMs in education: full question text, actual student responses (not just right/wrong), records of which wrong answers students chose, and alignment to Common Core K-12 standards. These 1.7 million interactions from 5,000 students enable research directions that were previously impossible to pursue, from fine-tuning student models to analyzing misconception patterns. To demonstrate the dataset's utility, we evaluate four frontier models (GPT-OSS-120B, Llama-3.3-70B, Qwen3-Next-80B variants) on two complementary task families: Knowledge Tracing, testing whether LLMs can predict student performance on questions, and the exact answer a student will give; and \textbf{Pedagogical Grounding}, testing whether LLMs understand the properties that make assessment items effective. Our evaluation reveals significant gaps in current LLM capabilities. Every model barely achieves a trivial baseline on knowledge tracing. All models fall below random chance on item discrimination, indicating that LLMs do not understand what makes one problem more diagnostic than another. Models do show competence at judging relative difficulty (up to 68.6%), but this partial success only highlights the gaps elsewhere. These results establish that substantial advances are needed before LLMs can reliably support personalized learning at scale. We release FoundationalASSIST to support progress on these foundational challenges. 

---
# Breaking the Reversal Curse in Autoregressive Language Models via Identity Bridge 

**Authors**: Xutao Ma, Yixiao Huang, Hanlin Zhu, Somayeh Sojoudi  

**Link**: [PDF](https://arxiv.org/pdf/2602.02470)  

**Abstract**: Autoregressive large language models (LLMs) have achieved remarkable success in many complex tasks, yet they can still fail in very simple logical reasoning such as the "reversal curse" -- when trained on forward knowledge data of the form "$A \rightarrow B$" (e.g., Alice's husband is Bob), the model is unable to deduce the reversal knowledge "$B \leftarrow A$" (e.g., Bob's wife is Alice) during test. Extensive prior research suggests that this failure is an inherent, fundamental limit of autoregressive causal LLMs, indicating that these models tend to memorize factual-level knowledge rather than capture higher-level rules. In this paper, we challenge this view by showing that this seemingly fundamental limit can be mitigated by slightly tweaking the training data with a simple regularization data recipe called the Identity Bridge of the form "$A \to A$" (e.g., The name of Alice is Alice). Theoretically, we prove that under this recipe, even a one-layer transformer can break the reversal curse by analyzing the implicit bias of gradient descent. Empirically, we show that a 1B pretrained language model finetuned with the proposed data recipe achieves a 40% success rate on reversal tasks, in stark contrast to a near-zero success rate when trained solely on forward-knowledge data. Our work provides a novel theoretical foundation for the reversal curse and offers a principled, low-cost path to encouraging LLMs to learn higher-level rules from data. 

---
# Structure Enables Effective Self-Localization of Errors in LLMs 

**Authors**: Ankur Samanta, Akshayaa Magesh, Ayush Jain, Kavosh Asadi, Youliang Yu, Daniel Jiang, Boris Vidolov, Kaveh Hassani, Paul Sajda, Jalaj Bhandari, Yonathan Efroni  

**Link**: [PDF](https://arxiv.org/pdf/2602.02416)  

**Abstract**: Self-correction in language models remains elusive. In this work, we explore whether language models can explicitly localize errors in incorrect reasoning, as a path toward building AI systems that can effectively correct themselves. We introduce a prompting method that structures reasoning as discrete, semantically coherent thought steps, and show that models are able to reliably localize errors within this structure, while failing to do so in conventional, unstructured chain-of-thought reasoning. Motivated by how the human brain monitors errors at discrete decision points and resamples alternatives, we introduce Iterative Correction Sampling of Thoughts (Thought-ICS), a self-correction framework. Thought-ICS iteratively prompts the model to generate reasoning one discrete and complete thought at a time--where each thought represents a deliberate decision by the model--creating natural boundaries for precise error localization. Upon verification, the model localizes the first erroneous step, and the system backtracks to generate alternative reasoning from the last correct point. When asked to correct reasoning verified as incorrect by an oracle, Thought-ICS achieves 20-40% self-correction lift. In a completely autonomous setting without external verification, it outperforms contemporary self-correction baselines. 

---
# AgentRx: Diagnosing AI Agent Failures from Execution Trajectories 

**Authors**: Shraddha Barke, Arnav Goyal, Alind Khare, Avaljot Singh, Suman Nath, Chetan Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2602.02475)  

**Abstract**: AI agents often fail in ways that are difficult to localize because executions are probabilistic, long-horizon, multi-agent, and mediated by noisy tool outputs. We address this gap by manually annotating failed agent runs and release a novel benchmark of 115 failed trajectories spanning structured API workflows, incident management, and open-ended web/file tasks. Each trajectory is annotated with a critical failure step and a category from a grounded-theory derived, cross-domain failure taxonomy. To mitigate the human cost of failure attribution, we present AGENTRX, an automated domain-agnostic diagnostic framework that pinpoints the critical failure step in a failed agent trajectory. It synthesizes constraints, evaluates them step-by-step, and produces an auditable validation log of constraint violations with associated evidence; an LLM-based judge uses this log to localize the critical step and category. Our framework improves step localization and failure attribution over existing baselines across three domains. 

---
# Live-Evo: Online Evolution of Agentic Memory from Continuous Feedback 

**Authors**: Yaolun Zhang, Yiran Wu, Yijiong Yu, Qingyun Wu, Huazheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02369)  

**Abstract**: Large language model (LLM) agents are increasingly equipped with memory, which are stored experience and reusable guidance that can improve task-solving performance. Recent \emph{self-evolving} systems update memory based on interaction outcomes, but most existing evolution pipelines are developed for static train/test splits and only approximate online learning by folding static benchmarks, making them brittle under true distribution shift and continuous feedback. We introduce \textsc{Live-Evo}, an online self-evolving memory system that learns from a stream of incoming data over time. \textsc{Live-Evo} decouples \emph{what happened} from \emph{how to use it} via an Experience Bank and a Meta-Guideline Bank, compiling task-adaptive guidelines from retrieved experiences for each task. To manage memory online, \textsc{Live-Evo} maintains experience weights and updates them from feedback: experiences that consistently help are reinforced and retrieved more often, while misleading or stale experiences are down-weighted and gradually forgotten, analogous to reinforcement and decay in human memory. On the live \textit{Prophet Arena} benchmark over a 10-week horizon, \textsc{Live-Evo} improves Brier score by 20.8\% and increases market returns by 12.9\%, while also transferring to deep-research benchmarks with consistent gains over strong baselines. Our code is available at this https URL. 

---
# Mitigating Safety Tax via Distribution-Grounded Refinement in Large Reasoning Models 

**Authors**: Yingsha Xie, Tiansheng Huang, Enneng Yang, Rui Min, Wenjie Lu, Xiaochun Cao, Naiqiang Tan, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.02136)  

**Abstract**: Safety alignment incurs safety tax that perturbs a large reasoning model's (LRM) general reasoning ability. Existing datasets used for safety alignment for an LRM are usually constructed by distilling safety reasoning traces and answers from an external LRM or human labeler. However, such reasoning traces and answers exhibit a distributional gap with the target LRM that needs alignment, and we conjecture such distributional gap is the culprit leading to significant degradation of reasoning ability of the target LRM. Driven by this hypothesis, we propose a safety alignment dataset construction method, dubbed DGR. DGR transforms and refines an existing out-of-distributional safety reasoning dataset to be aligned with the target's LLM inner distribution. Experimental results demonstrate that i) DGR effectively mitigates the safety tax while maintaining safety performance across all baselines, i.e., achieving \textbf{+30.2\%} on DirectRefusal and \textbf{+21.2\%} on R1-ACT improvement in average reasoning accuracy compared to Vanilla SFT; ii) the degree of reasoning degradation correlates with the extent of distribution shift, suggesting that bridging this gap is central to preserving capabilities. Furthermore, we find that safety alignment in LRMs may primarily function as a mechanism to activate latent knowledge, as a mere \textbf{10} samples are sufficient for activating effective refusal behaviors. These findings not only emphasize the importance of distributional consistency but also provide insights into the activation mechanism of safety in reasoning models. 

---
# Context Learning for Multi-Agent Discussion 

**Authors**: Xingyuan Hua, Sheng Yue, Xinyi Li, Yizhe Zhao, Jinrui Zhang, Ju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2602.02350)  

**Abstract**: Multi-Agent Discussion (MAD) has garnered increasing attention very recently, where multiple LLM instances collaboratively solve problems via structured discussion. However, we find that current MAD methods easily suffer from discussion inconsistency, LLMs fail to reach a coherent solution, due to the misalignment between their individual this http URL this paper, we introduce a multi-LLM context learning method (M2CL) that learns a context generator for each agent, capable of dynamically generating context instructions per discussion round via automatic information organization and refinement. Specifically, inspired by our theoretical insights on the context instruction, M2CL train the generators to control context coherence and output discrepancies via a carefully crafted self-adaptive this http URL enables LLMs to avoid premature convergence on majority noise and progressively reach the correct consensus. We evaluate M2CL on challenging tasks, including academic reasoning, embodied tasks, and mobile control. The results show that the performance of M2CL significantly surpasses existing methods by 20%--50%, while enjoying favorable transferability and computational efficiency. 

---
# Reasoning in a Combinatorial and Constrained World: Benchmarking LLMs on Natural-Language Combinatorial Optimization 

**Authors**: Xia Jiang, Jing Chen, Cong Zhang, Jie Gao, Chengpeng Hu, Chenhao Zhang, Yaoxin Wu, Yingqian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02188)  

**Abstract**: While large language models (LLMs) have shown strong performance in math and logic reasoning, their ability to handle combinatorial optimization (CO) -- searching high-dimensional solution spaces under hard constraints -- remains underexplored. To bridge the gap, we introduce NLCO, a \textbf{N}atural \textbf{L}anguage \textbf{C}ombinatorial \textbf{O}ptimization benchmark that evaluates LLMs on end-to-end CO reasoning: given a language-described decision-making scenario, the model must output a discrete solution without writing code or calling external solvers. NLCO covers 43 CO problems and is organized using a four-layer taxonomy of variable types, constraint families, global patterns, and objective classes, enabling fine-grained evaluation. We provide solver-annotated solutions and comprehensively evaluate LLMs by feasibility, solution optimality, and reasoning efficiency. Experiments across a wide range of modern LLMs show that high-performing models achieve strong feasibility and solution quality on small instances, but both degrade as instance size grows, even if more tokens are used for reasoning. We also observe systematic effects across the taxonomy: set-based tasks are relatively easy, whereas graph-structured problems and bottleneck objectives lead to more frequent failures. 

---
# Edit Knowledge, Not Just Facts via Multi-Step Reasoning over Background Stories 

**Authors**: Ya Gao, Kalle Kujanpää, Pekka Marttinen, Harri Valpola, Alexander Ilin  

**Link**: [PDF](https://arxiv.org/pdf/2602.02028)  

**Abstract**: Enabling artificial intelligence systems, particularly large language models, to integrate new knowledge and flexibly apply it during reasoning remains a central challenge. Existing knowledge editing approaches emphasize atomic facts, improving factual recall but often failing to integrate new information into a coherent framework usable across contexts. In this work, we argue that knowledge internalization is fundamentally a reasoning problem rather than a memorization problem. Consequently, a model should be trained in situations where the new information is instrumental to solving a task, combined with pre-existing knowledge, and exercised through multi-step reasoning. Based on this insight, we propose a training strategy based on three principles. First, new knowledge is introduced as a coherent background story that contextualizes novel facts and explains their relation to existing knowledge. Second, models are trained using self-generated multi-hop questions that require multi-step reasoning involving the new information. Third, training is done using knowledge distillation, forcing a student model to internalize the teacher's reasoning behavior without access to the novel information. Experiments show that models trained with this strategy effectively leverage newly acquired knowledge during reasoning and achieve remarkable performance on challenging questions that require combining multiple new facts. 

---
# Do I Really Know? Learning Factual Self-Verification for Hallucination Reduction 

**Authors**: Enes Altinisik, Masoomali Fatehkia, Fatih Deniz, Nadir Durrani, Majd Hawasly, Mohammad Raza, Husrev Taha Sencar  

**Link**: [PDF](https://arxiv.org/pdf/2602.02018)  

**Abstract**: Factual hallucination remains a central challenge for large language models (LLMs). Existing mitigation approaches primarily rely on either external post-hoc verification or mapping uncertainty directly to abstention during fine-tuning, often resulting in overly conservative behavior. We propose VeriFY, a training-time framework that teaches LLMs to reason about factual uncertainty through consistency-based self-verification. VeriFY augments training with structured verification traces that guide the model to produce an initial answer, generate and answer a probing verification query, issue a consistency judgment, and then decide whether to answer or abstain. To address the risk of reinforcing hallucinated content when training on augmented traces, we introduce a stage-level loss masking approach that excludes hallucinated answer stages from the training objective while preserving supervision over verification behavior. Across multiple model families and scales, VeriFY reduces factual hallucination rates by 9.7 to 53.3 percent, with only modest reductions in recall (0.4 to 5.7 percent), and generalizes across datasets when trained on a single source. The source code, training data, and trained model checkpoints will be released upon acceptance. 

---
# Emergent Analogical Reasoning in Transformers 

**Authors**: Gouki Minegishi, Jingyuan Feng, Hiroki Furuta, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2602.01992)  

**Abstract**: Analogy is a central faculty of human intelligence, enabling abstract patterns discovered in one domain to be applied to another. Despite its central role in cognition, the mechanisms by which Transformers acquire and implement analogical reasoning remain poorly understood. In this work, inspired by the notion of functors in category theory, we formalize analogical reasoning as the inference of correspondences between entities across categories. Based on this formulation, we introduce synthetic tasks that evaluate the emergence of analogical reasoning under controlled settings. We find that the emergence of analogical reasoning is highly sensitive to data characteristics, optimization choices, and model scale. Through mechanistic analysis, we show that analogical reasoning in Transformers decomposes into two key components: (1) geometric alignment of relational structure in the embedding space, and (2) the application of a functor within the Transformer. These mechanisms enable models to transfer relational structure from one category to another, realizing analogy. Finally, we quantify these effects and find that the same trends are observed in pretrained LLMs. In doing so, we move analogy from an abstract cognitive notion to a concrete, mechanistically grounded phenomenon in modern neural networks. 

---
# Position: Explaining Behavioral Shifts in Large Language Models Requires a Comparative Approach 

**Authors**: Martino Ciaperoni, Marzio Di Vece, Luca Pappalardo, Fosca Giannotti, Francesco Giannini  

**Link**: [PDF](https://arxiv.org/pdf/2602.02304)  

**Abstract**: Large-scale foundation models exhibit behavioral shifts: intervention-induced behavioral changes that appear after scaling, fine-tuning, reinforcement learning or in-context learning. While investigating these phenomena have recently received attention, explaining their appearance is still overlooked. Classic explainable AI (XAI) methods can surface failures at a single checkpoint of a model, but they are structurally ill-suited to justify what changed internally across different checkpoints and which explanatory claims are warranted about that change. We take the position that behavioral shifts should be explained comparatively: the core target should be the intervention-induced shift between a reference model and an intervened model, rather than any single model in isolation. To this aim we formulate a Comparative XAI ($\Delta$-XAI) framework with a set of desiderata to be taken into account when designing proper explaining methods. To highlight how $\Delta$-XAI methods work, we introduce a set of possible pipelines, relate them to the desiderata, and provide a concrete $\Delta$-XAI experiment. 

---
# Evolving from Tool User to Creator via Training-Free Experience Reuse in Multimodal Reasoning 

**Authors**: Xintian Shen, Jiawei Chen, Lihao Zheng, Hao Ma, Tao Wei, Kun Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2602.01983)  

**Abstract**: Existing Tool-Integrated Reasoning (TIR) models have effectively extended the question-answering capabilities of LLMs by incorporating external tools. However, real-world scenarios present numerous open-ended problems where fixed tools often fail to meet task requirements. Furthermore, the lack of self-optimization mechanisms means that erroneous tool outputs can mislead the LLM's responses. Additionally, the construction of existing tools entails significant manual effort, which consequently constrains their applicability. Recognizing that the reasoning traces of LLMs encapsulate implicit problem-solving capabilities, we propose UCT, a novel training-free framework that transforms agents from tool users to tool creators. This approach harvests reasoning experiences and distills them into reusable assets. This method transforms the agent from a mere tool user into a tool creator, enabling adaptive tool creation and self-updating during the inference process. We also introduce a memory consolidation mechanism to maintain the tool library, ensuring high reusability of retained experiential memory for subsequent reasoning tasks. This novel automated tool construction paradigm continuously improves tool quality during reasoning, allowing the overall agent system to progress without additional training. Extensive experiments demonstrate that our method serves as a novel paradigm for enhancing the capabilities of TIR models. In particular, the significant performance gains achieved +20.86%$\uparrow$ and +23.04%$\uparrow$ on benchmarks across multi-domain mathematical and scientific reasoning tasks validate the self-evolving capability of the agent. 

---
# Light Alignment Improves LLM Safety via Model Self-Reflection with a Single Neuron 

**Authors**: Sicheng Shen, Mingyang Lv, Han Shen, Jialin Wu, Binghao Wang, Zhou Yang, Guobin Shen, Dongcheng Zhao, Feifei Zhao, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2602.02027)  

**Abstract**: The safety of large language models (LLMs) has increasingly emerged as a fundamental aspect of their development. Existing safety alignment for LLMs is predominantly achieved through post-training methods, which are computationally expensive and often fail to generalize well across different models. A small number of lightweight alignment approaches either rely heavily on prior-computed safety injections or depend excessively on the model's own capabilities, resulting in limited generalization and degraded efficiency and usability during generation. In this work, we propose a safety-aware decoding method that requires only low-cost training of an expert model and employs a single neuron as a gating mechanism. By effectively balancing the model's intrinsic capabilities with external guidance, our approach simultaneously preserves utility and enhances output safety. It demonstrates clear advantages in training overhead and generalization across model scales, offering a new perspective on lightweight alignment for the safe and practical deployment of large language models. Code: this https URL. 

---
# Small Generalizable Prompt Predictive Models Can Steer Efficient RL Post-Training of Large Reasoning Models 

**Authors**: Yun Qu, Qi Wang, Yixiu Mao, Heming Zou, Yuhang Jiang, Weijie Liu, Clive Bai, Kai Yang, Yangkun Chen, Saiyong Yang, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2602.01970)  

**Abstract**: Reinforcement learning enhances the reasoning capabilities of large language models but often involves high computational costs due to rollout-intensive optimization. Online prompt selection presents a plausible solution by prioritizing informative prompts to improve training efficiency. However, current methods either depend on costly, exact evaluations or construct prompt-specific predictive models lacking generalization across prompts. This study introduces Generalizable Predictive Prompt Selection (GPS), which performs Bayesian inference towards prompt difficulty using a lightweight generative model trained on the shared optimization history. Intermediate-difficulty prioritization and history-anchored diversity are incorporated into the batch acquisition principle to select informative prompt batches. The small predictive model also generalizes at test-time for efficient computational allocation. Experiments across varied reasoning benchmarks indicate GPS's substantial improvements in training efficiency, final performance, and test-time efficiency over superior baseline methods. 

---
# Geometric Analysis of Token Selection in Multi-Head Attention 

**Authors**: Timur Mudarisov, Mikhal Burtsev, Tatiana Petrova, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2602.01893)  

**Abstract**: We present a geometric framework for analysing multi-head attention in large language models (LLMs). Without altering the mechanism, we view standard attention through a top-N selection lens and study its behaviour directly in value-state space. We define geometric metrics - Precision, Recall, and F-score - to quantify separability between selected and non-selected tokens, and derive non-asymptotic bounds with explicit dependence on dimension and margin under empirically motivated assumptions (stable value norms with a compressed sink token, exponential similarity decay, and piecewise attention weight profiles). The theory predicts a small-N operating regime of strongest non-trivial separability and clarifies how sequence length and sink similarity shape the metrics. Empirically, across LLaMA-2-7B, Gemma-7B, and Mistral-7B, measurements closely track the theoretical envelopes: top-N selection sharpens separability, sink similarity correlates with Recall. We also found that in LLaMA-2-7B heads specialize into three regimes - Retriever, Mixer, Reset - with distinct geometric signatures. Overall, attention behaves as a structured geometric classifier with measurable criteria for token selection, offering head level interpretability and informing geometry-aware sparsification and design of attention in LLMs. 

---
# ORCH: many analyses, one merge-a deterministic multi-agent orchestrator for discrete-choice reasoning with EMA-guided routing 

**Authors**: Hanlin Zhou, Huah Yong Chan  

**Link**: [PDF](https://arxiv.org/pdf/2602.01797)  

**Abstract**: Recent advances in large-scale language models (LLMs) have made multi-agent architectures attractive for challenging reasoning tasks. However, many existing systems rely on stochastic routing or ad-hoc heuristics, making their behavior difficult to reproduce and their decision process hard to interpret. We propose ORCH, a deterministic coordination framework for discrete-choice reasoning that orchestrates heterogeneous LLMs. ORCH follows a ``many analyses, one decision'' paradigm: multiple base models independently produce structured analyses, and a dedicated merge agent outputs the final choice. The framework uses fixed rules for task decomposition and answer aggregation, keeping the pipeline predictable, reproducible, and training-free. Determinism here refers to fixed routing and aggregation rules under a fixed evaluation protocol, rather than strict bit-level reproducibility across deployments. To exploit model complementarity, we optionally introduce an EMA-guided router that updates agent selection using historical accuracy, latency, or cost; since it relies on answer-based feedback, it is mainly intended for benchmarking, controlled evaluation, or delayed-feedback settings. Experiments on MMLU, MMLU-Pro, and GSM8K show that ORCH consistently outperforms single-model baselines and a majority-vote ensemble. On MMLU-Pro, ORCH improves accuracy by over 10 points compared to the strongest baseline, and on GSM8K it yields gains exceeding 50 points; McNemar tests confirm statistical significance. The EMA router provides an additional 0.7--2.0 point accuracy boost, and ablations show that both multi-agent collaboration and routing contribute substantially. Overall, ORCH offers a practical path toward controllable, interpretable, and deployment-ready LLM-based agent systems for discrete-choice reasoning. 

---
# LingLanMiDian: Systematic Evaluation of LLMs on TCM Knowledge and Clinical Reasoning 

**Authors**: Rui Hua, Yu Wei, Zixin Shu, Kai Chang, Dengying Yan, Jianan Xia, Zeyu Liu, Hui Zhu, Shujie Song, Mingzhong Xiao, Xiaodong Li, Dongmei Jia, Zhuye Gao, Yanyan Meng, Naixuan Zhao, Yu Fu, Haibin Yu, Benman Yu, Yuanyuan Chen, Fei Dong, Zhizhou Meng, Pengcheng Yang, Songxue Zhao, Lijuan Pei, Yunhui Hu, Kan Ding, Jiayuan Duan, Wenmao Yin, Yang Gu, Runshun Zhang, Qiang Zhu, Jian Yu, Jiansheng Li, Baoyan Liu, Wenjia Wang, Xuezhong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.01779)  

**Abstract**: Large language models (LLMs) are advancing rapidly in medical NLP, yet Traditional Chinese Medicine (TCM) with its distinctive ontology, terminology, and reasoning patterns requires domain-faithful evaluation. Existing TCM benchmarks are fragmented in coverage and scale and rely on non-unified or generation-heavy scoring that hinders fair comparison. We present the LingLanMiDian (LingLan) benchmark, a large-scale, expert-curated, multi-task suite that unifies evaluation across knowledge recall, multi-hop reasoning, information extraction, and real-world clinical decision-making. LingLan introduces a consistent metric design, a synonym-tolerant protocol for clinical labels, a per-dataset 400-item Hard subset, and a reframing of diagnosis and treatment recommendation into single-choice decision recognition. We conduct comprehensive, zero-shot evaluations on 14 leading open-source and proprietary LLMs, providing a unified perspective on their strengths and limitations in TCM commonsense knowledge understanding, reasoning, and clinical decision support; critically, the evaluation on Hard subset reveals a substantial gap between current models and human experts in TCM-specialized reasoning. By bridging fundamental knowledge and applied reasoning through standardized evaluation, LingLan establishes a unified, quantitative, and extensible foundation for advancing TCM LLMs and domain-specific medical AI research. All evaluation data and code are available at this https URL and this http URL. 

---
# Optimizing Prompts for Large Language Models: A Causal Approach 

**Authors**: Wei Chen, Yanbin Fang, Shuran Fu, Fasheng Xu, Xuan Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.01711)  

**Abstract**: Large Language Models (LLMs) are increasingly embedded in enterprise workflows, yet their performance remains highly sensitive to prompt design. Automatic Prompt Optimization (APO) seeks to mitigate this instability, but existing approaches face two persistent challenges. First, commonly used prompt strategies rely on static instructions that perform well on average but fail to adapt to heterogeneous queries. Second, more dynamic approaches depend on offline reward models that are fundamentally correlational, confounding prompt effectiveness with query characteristics. We propose Causal Prompt Optimization (CPO), a framework that reframes prompt design as a problem of causal estimation. CPO operates in two stages. First, it learns an offline causal reward model by applying Double Machine Learning (DML) to semantic embeddings of prompts and queries, isolating the causal effect of prompt variations from confounding query attributes. Second, it utilizes this unbiased reward signal to guide a resource-efficient search for query-specific prompts without relying on costly online evaluation. We evaluate CPO across benchmarks in mathematical reasoning, visualization, and data analytics. CPO consistently outperforms human-engineered prompts and state-of-the-art automated optimizers. The gains are driven primarily by improved robustness on hard queries, where existing methods tend to deteriorate. Beyond performance, CPO fundamentally reshapes the economics of prompt optimization: by shifting evaluation from real-time model execution to an offline causal model, it enables high-precision, per-query customization at a fraction of the inference cost required by online methods. Together, these results establish causal inference as a scalable foundation for reliable and cost-efficient prompt optimization in enterprise LLM deployments. 

---
# MACD: Model-Aware Contrastive Decoding via Counterfactual Data 

**Authors**: Qixin Xiao, Kun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.01740)  

**Abstract**: Video language models (Video-LLMs) are prone to hallucinations, often generating plausible but ungrounded content when visual evidence is weak, ambiguous, or biased. Existing decoding methods, such as contrastive decoding (CD), rely on random perturbations to construct contrastive data for mitigating hallucination patterns. However, such a way is hard to control the visual cues that drive hallucination or well align with model weaknesses. We propose Model-aware Counterfactual Data based Contrastive Decoding (MACD), a new inference strategy that combines model-guided counterfactual construction with decoding. Our approach uses the Video-LLM's own feedback to identify object regions most responsible for hallucination, generating targeted counterfactual inputs at the object level rather than arbitrary frame or temporal modifications. These model-aware counterfactual data is then integrated into CD to enforce evidence-grounded token selection during decoding. Experiments on EventHallusion, MVBench, Perception-test and Video-MME show that MACD consistently reduces hallucination while maintaining or improving task accuracy across diverse Video-LLMs, including Qwen and InternVL families. The method is especially effective in challenging scenarios involving small, occluded, or co-occurring objects. Our code and data will be publicly released. 

---
# ToPT: Task-Oriented Prompt Tuning for Urban Region Representation Learning 

**Authors**: Zitao Guo, Changyang Jiang, Tianhong Zhao, Jinzhou Cao, Genan Dai, Bowen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.01610)  

**Abstract**: Learning effective region embeddings from heterogeneous urban data underpins key urban computing tasks (e.g., crime prediction, resource allocation). However, prevailing two-stage methods yield task-agnostic representations, decoupling them from downstream objectives. Recent prompt-based approaches attempt to fix this but introduce two challenges: they often lack explicit spatial priors, causing spatially incoherent inter-region modeling, and they lack robust mechanisms for explicit task-semantic alignment. We propose ToPT, a two-stage framework that delivers spatially consistent fusion and explicit task alignment. ToPT consists of two modules: spatial-aware region embedding learning (SREL) and task-aware prompting for region embeddings (Prompt4RE). SREL employs a Graphormer-based fusion module that injects spatial priors-distance and regional centrality-as learnable attention biases to capture coherent, interpretable inter-region interactions. Prompt4RE performs task-oriented prompting: a frozen multimodal large language model (MLLM) processes task-specific templates to obtain semantic vectors, which are aligned with region embeddings via multi-head cross-attention for stable task conditioning. Experiments across multiple tasks and cities show state-of-the-art performance, with improvements of up to 64.2\%, validating the necessity and complementarity of spatial priors and prompt-region alignment. The code is available at this https URL. 

---
# Autonomous Question Formation for Large Language Model-Driven AI Systems 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2602.01556)  

**Abstract**: Large language model (LLM)-driven AI systems are increasingly important for autonomous decision-making in dynamic and open environments. However, most existing systems rely on predefined tasks and fixed prompts, limiting their ability to autonomously identify what problems should be solved when environmental conditions change. In this paper, we propose a human-simulation-based framework that enables AI systems to autonomously form questions and set tasks by reasoning over their internal states, environmental observations, and interactions with other AI systems. The proposed method treats question formation as a first-class decision process preceding task selection and execution, and integrates internal-driven, environment-aware, and inter-agent-aware prompting scopes to progressively expand cognitive coverage. In addition, the framework supports learning the question-formation process from experience, allowing the system to improve its adaptability and decision quality over time. xperimental results in a multi-agent simulation environment show that environment-aware prompting significantly reduces no-eat events compared with the internal-driven baseline, and inter-agent-aware prompting further reduces cumulative no-eat events by more than 60% over a 20-day simulation, with statistically significant improvements (p < 0.05). 

---
# What LLMs Think When You Don't Tell Them What to Think About? 

**Authors**: Yongchan Kwon, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2602.01689)  

**Abstract**: Characterizing the behavior of large language models (LLMs) across diverse settings is critical for reliable monitoring and AI safety. However, most existing analyses rely on topic- or task-specific prompts, which can substantially limit what can be observed. In this work, we study what LLMs generate from minimal, topic-neutral inputs and probe their near-unconstrained generative behavior. Despite the absence of explicit topics, model outputs cover a broad semantic space, and surprisingly, each model family exhibits strong and systematic topical preferences. GPT-OSS predominantly generates programming (27.1%) and mathematical content (24.6%), whereas Llama most frequently generates literary content (9.1%). DeepSeek often generates religious content, while Qwen frequently generates multiple-choice questions. Beyond topical preferences, we also observe differences in content specialization and depth: GPT-OSS often generates more technically advanced content (e.g., dynamic programming) compared with other models (e.g., basic Python). Furthermore, we find that the near-unconstrained generation often degenerates into repetitive phrases, revealing interesting behaviors unique to each model family. For instance, degenerate outputs from Llama include multiple URLs pointing to personal Facebook and Instagram accounts. We release the complete dataset of 256,000 samples from 16 LLMs, along with a reproducible codebase. 

---
# Predictive Scheduling for Efficient Inference-Time Reasoning in Large Language Models 

**Authors**: Katrina Brown, Aneesh Muppidi, Rana Shahout  

**Link**: [PDF](https://arxiv.org/pdf/2602.01237)  

**Abstract**: Large language models (LLMs) achieve state-of-the-art accuracy on complex reasoning tasks by generating multiple chain-of-thought (CoT) traces, but using a fixed token budget per query leads to over-computation on easy inputs and under-computation on hard ones. We introduce Predictive Scheduling, a plug-and-play framework that pre-runs lightweight predictors, an MLP on intermediate transformer hidden states or a LoRA-fine-tuned classifier on raw question text, to estimate each query's optimal reasoning length or difficulty before any full generation. Our greedy batch allocator dynamically distributes a fixed total token budget across queries to maximize expected accuracy. On the GSM8K arithmetic benchmark, predictive scheduling yields up to 7.9 percentage points of absolute accuracy gain over uniform budgeting at identical token cost, closing over 50\% of the gap to an oracle with perfect foresight. A systematic layer-wise study reveals that middle layers (12 - 17) of the transformer carry the richest signals for size estimation. These results demonstrate that pre-run budget prediction enables fine-grained control of the compute-accuracy trade-off, offering a concrete path toward latency-sensitive, cost-efficient LLM deployments. 

---
# FutureMind: Equipping Small Language Models with Strategic Thinking-Pattern Priors via Adaptive Knowledge Distillation 

**Authors**: Shaoxiong Yang, Junting Li, Mengyuan Zhang, Chao Li, Wei Liu, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2602.01222)  

**Abstract**: Small Language Models (SLMs) are attractive for cost-sensitive and resource-limited settings due to their efficient, low-latency inference. However, they often struggle with complex, knowledge-intensive tasks that require structured reasoning and effective retrieval. To address these limitations, we propose FutureMind, a modular reasoning framework that equips SLMs with strategic thinking-pattern priors via adaptive knowledge distillation from large language models (LLMs). FutureMind introduces a dynamic reasoning pipeline composed of four key modules: Problem Analysis, Logical Reasoning, Strategy Planning, and Retrieval Guidance. This pipeline is augmented by three distinct retrieval paradigms that decompose complex queries into tractable subproblems, ensuring efficient and accurate retrieval execution. Extensive experiments on multi-hop QA benchmarks, including 2WikiMultihopQA, MuSiQue, Bamboogle, and Frames, demonstrate the superiority of FutureMind. It consistently outperforms strong baselines such as Search-o1, achieving state-of-the-art results under free training conditions across diverse SLM architectures and scales. Beyond empirical gains, our analysis reveals that the process of thinking-pattern distillation is restricted by the cognitive bias bottleneck between the teacher (LLMs) and student (SLMs) models. This provides new perspectives on the transferability of reasoning skills, paving the way for the development of SLMs that combine efficiency with genuine cognitive capability. 

---
# RE-MCDF: Closed-Loop Multi-Expert LLM Reasoning for Knowledge-Grounded Clinical Diagnosis 

**Authors**: Shaowei Shen, Xiaohong Yang, Jie Yang, Lianfen Huang, Yongcai Zhang, Yang Zou, Seyyedali Hosseinalipour  

**Link**: [PDF](https://arxiv.org/pdf/2602.01297)  

**Abstract**: Electronic medical records (EMRs), particularly in neurology, are inherently heterogeneous, sparse, and noisy, which poses significant challenges for large language models (LLMs) in clinical diagnosis. In such settings, single-agent systems are vulnerable to self-reinforcing errors, as their predictions lack independent validation and can drift toward spurious conclusions. Although recent multi-agent frameworks attempt to mitigate this issue through collaborative reasoning, their interactions are often shallow and loosely structured, failing to reflect the rigorous, evidence-driven processes used by clinical experts. More fundamentally, existing approaches largely ignore the rich logical dependencies among diseases, such as mutual exclusivity, pathological compatibility, and diagnostic confusion. This limitation prevents them from ruling out clinically implausible hypotheses, even when sufficient evidence is available. To overcome these, we propose RE-MCDF, a relation-enhanced multi-expert clinical diagnosis framework. RE-MCDF introduces a generation--verification--revision closed-loop architecture that integrates three complementary components: (i) a primary expert that generates candidate diagnoses and supporting evidence, (ii) a laboratory expert that dynamically prioritizes heterogeneous clinical indicators, and (iii) a multi-relation awareness and evaluation expert group that explicitly enforces inter-disease logical constraints. Guided by a medical knowledge graph (MKG), the first two experts adaptively reweight EMR evidence, while the expert group validates and corrects candidate diagnoses to ensure logical consistency. Extensive experiments on the neurology subset of CMEMR (NEEMRs) and on our curated dataset (XMEMRs) demonstrate that RE-MCDF consistently outperforms state-of-the-art baselines in complex diagnostic scenarios. 

---
# A State-Transition Framework for Efficient LLM Reasoning 

**Authors**: Liang Zhang, Yu Zhao, Longyue Wang, Tianqi Shi, Weihua Luo, Kaifu Zhang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2602.01198)  

**Abstract**: While Long Chain-of-Thought (CoT) reasoning significantly improves Large Language Models (LLMs) performance on complex reasoning tasks, the substantial computational and memory costs of generating long CoT sequences limit their efficiency and practicality. Existing studies usually enhance the reasoning efficiency of LLMs by compressing CoT sequences. However, this approach conflicts with test-time scaling, limiting the reasoning capacity of LLMs. In this paper, we propose an efficient reasoning framework that models the reasoning process of LLMs as a state-transition process. Specifically, we first apply a linear attention mechanism to estimate the LLM's reasoning state, which records the historical reasoning information from previous reasoning steps. Then, based on the query prompt and the reasoning state, the LLM can efficiently perform the current reasoning step and update the state. With the linear attention, each token in the current reasoning step can directly retrieve relevant historical reasoning information from the reasoning state, without explicitly attending to tokens in previous reasoning steps. In this way, the computational complexity of attention is reduced from quadratic to linear, significantly improving the reasoning efficiency of LLMs. In addition, we propose a state-based reasoning strategy to mitigate the over-thinking issue caused by noisy reasoning steps. Extensive experiments across multiple datasets and model sizes demonstrate that our framework not only improves the reasoning efficiency of LLMs but also enhances their reasoning performance. 

---
# Hard Constraints Meet Soft Generation: Guaranteed Feasibility for LLM-based Combinatorial Optimization 

**Authors**: Yang Liu, Chuan Zhou, Yancheng Chen, Shuai Zhang, Xixun Lin, Xiaoqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.01090)  

**Abstract**: Large language models (LLMs) have emerged as promising general-purpose solvers for combinatorial optimization (CO), yet they fundamentally lack mechanisms to guarantee solution feasibility which is critical for real-world deployment. In this work, we introduce FALCON, a framework that ensures 100\% feasibility through three key innovations: (i) \emph{grammar-constrained decoding} enforces syntactic validity, (ii) a \emph{feasibility repair layer} corrects semantic constraint violations, and (iii) \emph{adaptive Best-of-$N$ sampling} allocates inference compute efficiently. To train the underlying LLM, we introduce the Best-anchored Objective-guided Preference Optimization (BOPO) in LLM training, which weights preference pairs by their objective gap, providing dense supervision without human labels. Theoretically, we prove convergence for BOPO and provide bounds on repair-induced quality loss. Empirically, across seven NP-hard CO problems, FALCON achieves perfect feasibility while matching or exceeding the solution quality of state-of-the-art neural and LLM-based solvers. 

---
# ConvexBench: Can LLMs Recognize Convex Functions? 

**Authors**: Yepeng Liu, Yu Huang, Yu-Xiang Wang, Yingbin Liang, Yuheng Bu  

**Link**: [PDF](https://arxiv.org/pdf/2602.01075)  

**Abstract**: Convex analysis is a modern branch of mathematics with many applications. As Large Language Models (LLMs) start to automate research-level math and sciences, it is important for LLMs to demonstrate the ability to understand and reason with convexity. We introduce \cb, a scalable and mechanically verifiable benchmark for testing \textit{whether LLMs can identify the convexity of a symbolic objective under deep functional composition.} Experiments on frontier LLMs reveal a sharp compositional reasoning gap: performance degrades rapidly with increasing depth, dropping from an F1-score of $1.0$ at depth $2$ to approximately $0.2$ at depth $100$. Inspection of models' reasoning traces indicates two failure modes: \textit{parsing failure} and \textit{lazy reasoning}. To address these limitations, we propose an agentic divide-and-conquer framework that (i) offloads parsing to an external tool to construct an abstract syntax tree (AST) and (ii) enforces recursive reasoning over each intermediate sub-expression with focused context. This framework reliably mitigates deep-composition failures, achieving substantial performance improvement at large depths (e.g., F1-Score $= 1.0$ at depth $100$). 

---
# PersistBench: When Should Long-Term Memories Be Forgotten by LLMs? 

**Authors**: Sidharth Pulipaka, Oliver Chen, Manas Sharma, Taaha S Bajwa, Vyas Raina, Ivaxi Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2602.01146)  

**Abstract**: Conversational assistants are increasingly integrating long-term memory with large language models (LLMs). This persistence of memories, e.g., the user is vegetarian, can enhance personalization in future conversations. However, the same persistence can also introduce safety risks that have been largely overlooked. Hence, we introduce PersistBench to measure the extent of these safety risks. We identify two long-term memory-specific risks: cross-domain leakage, where LLMs inappropriately inject context from the long-term memories; and memory-induced sycophancy, where stored long-term memories insidiously reinforce user biases. We evaluate 18 frontier and open-source LLMs on our benchmark. Our results reveal a surprisingly high failure rate across these LLMs - a median failure rate of 53% on cross-domain samples and 97% on sycophancy samples. To address this, our benchmark encourages the development of more robust and safer long-term memory usage in frontier conversational systems. 

---
# LLM-Driven Ontology Construction for Enterprise Knowledge Graphs 

**Authors**: Abdulsobur Oyewale, Tommaso Soru  

**Link**: [PDF](https://arxiv.org/pdf/2602.01276)  

**Abstract**: Enterprise Knowledge Graphs have become essential for unifying heterogeneous data and enforcing semantic governance. However, the construction of their underlying ontologies remains a resource-intensive, manual process that relies heavily on domain expertise. This paper introduces OntoEKG, a LLM-driven pipeline designed to accelerate the generation of domain-specific ontologies from unstructured enterprise data. Our approach decomposes the modelling task into two distinct phases: an extraction module that identifies core classes and properties, and an entailment module that logically structures these elements into a hierarchy before serialising them into standard RDF. Addressing the significant lack of comprehensive benchmarks for end-to-end ontology construction, we adopt a new evaluation dataset derived from documents across the Data, Finance, and Logistics sectors. Experimental results highlight both the potential and the challenges of this approach, achieving a fuzzy-match F1-score of 0.724 in the Data domain while revealing limitations in scope definition and hierarchical reasoning. 

---
# EvoOpt-LLM: Evolving industrial optimization models with large language models 

**Authors**: Yiliu He, Tianle Li, Binghao Ji, Zhiyuan Liu, Di Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.01082)  

**Abstract**: Optimization modeling via mixed-integer linear programming (MILP) is fundamental to industrial planning and scheduling, yet translating natural-language requirements into solver-executable models and maintaining them under evolving business rules remains highly expertise-intensive. While large language models (LLMs) offer promising avenues for automation, existing methods often suffer from low data efficiency, limited solver-level validity, and poor scalability to industrial-scale problems. To address these challenges, we present EvoOpt-LLM, a unified LLM-based framework supporting the full lifecycle of industrial optimization modeling, including automated model construction, dynamic business-constraint injection, and end-to-end variable pruning. Built on a 7B-parameter LLM and adapted via parameter-efficient LoRA fine-tuning, EvoOpt-LLM achieves a generation rate of 91% and an executability rate of 65.9% with only 3,000 training samples, with critical performance gains emerging under 1,500 samples. The constraint injection module reliably augments existing MILP models while preserving original objectives, and the variable pruning module enhances computational efficiency, achieving an F1 score of ~0.56 on medium-sized LP models with only 400 samples. EvoOpt-LLM demonstrates a practical, data-efficient approach to industrial optimization modeling, reducing reliance on expert intervention while improving adaptability and solver efficiency. 

---
# MedBeads: An Agent-Native, Immutable Data Substrate for Trustworthy Medical AI 

**Authors**: Takahito Nakajima  

**Link**: [PDF](https://arxiv.org/pdf/2602.01086)  

**Abstract**: Background: As of 2026, Large Language Models (LLMs) demonstrate expert-level medical knowledge. However, deploying them as autonomous "Clinical Agents" remains limited. Current Electronic Medical Records (EMRs) and standards like FHIR are designed for human review, creating a "Context Mismatch": AI agents receive fragmented data and must rely on probabilistic inference (e.g., RAG) to reconstruct patient history. This approach causes hallucinations and hinders auditability. Methods: We propose MedBeads, an agent-native data infrastructure where clinical events are immutable "Beads"--nodes in a Merkle Directed Acyclic Graph (DAG)--cryptographically referencing causal predecessors. This "write-once, read-many" architecture makes tampering mathematically detectable. We implemented a prototype with a Go Core Engine, Python middleware for LLM integration, and a React-based visualization interface. Results: We successfully implemented the workflow using synthetic data. The FHIR-to-DAG conversion transformed flat resources into a causally-linked graph. Our Breadth-First Search (BFS) Context Retrieval algorithm traverses relevant subgraphs with O(V+E) complexity, enabling real-time decision support. Tamper-evidence is guaranteed by design: any modification breaks the cryptographic chain. The visualization aids clinician understanding through explicit causal links. Conclusion: MedBeads addresses the "Context Mismatch" by shifting from probabilistic search to deterministic graph traversal, and from mutable records to immutable chains, providing the substrate for "Trustworthy Medical AI." It guarantees the context the AI receives is deterministic and tamper-evident, while the LLM determines interpretation. The structured Bead format serves as a token-efficient "AI-native language." We release MedBeads as open-source software to accelerate agent-native data standards. 

---
# SetPO: Set-Level Policy Optimization for Diversity-Preserving LLM Reasoning 

**Authors**: Chenyi Li, Yuan Zhang, Bo Wang, Guoqing Ma, Wei Tang, Haoyang Huang, Nan Duan  

**Link**: [PDF](https://arxiv.org/pdf/2602.01062)  

**Abstract**: Reinforcement learning with verifiable rewards has shown notable effectiveness in enhancing large language models (LLMs) reasoning performance, especially in mathematics tasks. However, such improvements often come with reduced outcome diversity, where the model concentrates probability mass on a narrow set of solutions. Motivated by diminishing-returns principles, we introduce a set level diversity objective defined over sampled trajectories using kernelized similarity. Our approach derives a leave-one-out marginal contribution for each sampled trajectory and integrates this objective as a plug-in advantage shaping term for policy optimization. We further investigate the contribution of a single trajectory to language model diversity within a distribution perturbation framework. This analysis theoretically confirms a monotonicity property, proving that rarer trajectories yield consistently higher marginal contributions to the global diversity. Extensive experiments across a range of model scales demonstrate the effectiveness of our proposed algorithm, consistently outperforming strong baselines in both Pass@1 and Pass@K across various benchmarks. 

---
# Small-Margin Preferences Still Matter-If You Train Them Right 

**Authors**: Jinlong Pang, Zhaowei Zhu, Na Di, Yichi Zhang, Yaxuan Wang, Chen Qian, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00954)  

**Abstract**: Preference optimization methods such as DPO align large language models (LLMs) using paired comparisons, but their effectiveness can be highly sensitive to the quality and difficulty of preference pairs. A common heuristic treats small-margin (ambiguous) pairs as noisy and filters them out. In this paper, we revisit this assumption and show that pair difficulty interacts strongly with the optimization objective: when trained with preference-based losses, difficult pairs can destabilize training and harm alignment, yet these same pairs still contain useful supervision signals when optimized with supervised fine-tuning (SFT). Motivated by this observation, we propose MixDPO, a simple yet effective difficulty-aware training strategy that (i) orders preference data from easy to hard (a curriculum over margin-defined difficulty), and (ii) routes difficult pairs to an SFT objective while applying a preference loss to easy pairs. This hybrid design provides a practical mechanism to leverage ambiguous pairs without incurring the optimization failures often associated with preference losses on low-margin data. Across three LLM-judge benchmarks, MixDPO consistently improves alignment over DPO and a range of widely-used variants, with particularly strong gains on AlpacaEval~2 length-controlled (LC) win rate. 

---
# Learning Abstractions for Hierarchical Planning in Program-Synthesis Agents 

**Authors**: Zergham Ahmed, Kazuki Irie, Joshua B. Tenenbaum, Christopher J. Bates, Samuel J. Gershman  

**Link**: [PDF](https://arxiv.org/pdf/2602.00929)  

**Abstract**: Humans learn abstractions and use them to plan efficiently to quickly generalize across tasks -- an ability that remains challenging for state-of-the-art large language model (LLM) agents and deep reinforcement learning (RL) systems. Inspired by the cognitive science of how people form abstractions and intuitive theories of their world knowledge, Theory-Based RL (TBRL) systems, such as TheoryCoder, exhibit strong generalization through effective use of abstractions. However, they heavily rely on human-provided abstractions and sidestep the abstraction-learning problem. We introduce TheoryCoder-2, a new TBRL agent that leverages LLMs' in-context learning ability to actively learn reusable abstractions rather than relying on hand-specified ones, by synthesizing abstractions from experience and integrating them into a hierarchical planning process. We conduct experiments on diverse environments, including BabyAI, Minihack and VGDL games like Sokoban. We find that TheoryCoder-2 is significantly more sample-efficient than baseline LLM agents augmented with classical planning domain construction, reasoning-based planning, and prior program-synthesis agents such as WorldCoder. TheoryCoder-2 is able to solve complex tasks that the baselines fail, while only requiring minimal human prompts, unlike prior TBRL systems. 

---
# Synapse Compendium Aware Federated Knowledge Exchange for Tool Routed LLMs 

**Authors**: Abhijit Chakraborty, Sandipan De, Yash Shah, Chahana Dahal, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2602.00911)  

**Abstract**: Collaborative learning among LLM-based agents under federated learning faces challenges, including communication costs, heterogeneity in data, and tool-usage, limiting their effectiveness. We introduce Synapse, a framework that trains a shared global knowledge model of tool-usage behavior. Client agents with fixed LLMs learn tool-usage patterns locally, and transmit artifacts for federated aggregation through coordinators. A global tool compendium is updated and redistributed, enabling convergence toward stable tool selection. Synapse uses templated representations, embedding retrieval with LLM reranking, and adaptive masking to maintain utility while limiting information leakage. The framework supports heterogeneous data and quantifies performance improvements. Results show that Synapse improves tool-usage effectiveness and reduces communication overhead compared with weight or prompt-sharing approaches in multi-agent LLM systems. 

---
# Optimizing Agentic Reasoning with Retrieval via Synthetic Semantic Information Gain Reward 

**Authors**: Senkang Hu, Yong Dai, Yuzhi Zhao, Yihang Tao, Yu Guo, Zhengru Fang, Sam Tak Wu Kwong, Yuguang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2602.00845)  

**Abstract**: Agentic reasoning enables large reasoning models (LRMs) to dynamically acquire external knowledge, but yet optimizing the retrieval process remains challenging due to the lack of dense, principled reward signals. In this paper, we introduce InfoReasoner, a unified framework that incentivizes effective information seeking via a synthetic semantic information gain reward. Theoretically, we redefine information gain as uncertainty reduction over the model's belief states, establishing guarantees, including non-negativity, telescoping additivity, and channel monotonicity. Practically, to enable scalable optimization without manual retrieval annotations, we propose an output-aware intrinsic estimator that computes information gain directly from the model's output distributions using semantic clustering via bidirectional textual entailment. This intrinsic reward guides the policy to maximize epistemic progress, enabling efficient training via Group Relative Policy Optimxization (GRPO). Experiments across seven question-answering benchmarks demonstrate that InfoReasoner consistently outperforms strong retrieval-augmented baselines, achieving up to 5.4% average accuracy improvement. Our work provides a theoretically grounded and scalable path toward agentic reasoning with retrieval. 

---
# Resource-Efficient Reinforcement for Reasoning Large Language Models via Dynamic One-Shot Policy Refinement 

**Authors**: Yunjian Zhang, Sudong Wang, Yang Li, Peiran Xu, Conghao Zhou, Xiaoyue Ma, Jianing Li, Yao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00815)  

**Abstract**: Large language models (LLMs) have exhibited remarkable performance on complex reasoning tasks, with reinforcement learning under verifiable rewards (RLVR) emerging as a principled framework for aligning model behavior with reasoning chains. Despite its promise, RLVR remains prohibitively resource-intensive, requiring extensive reward signals and incurring substantial rollout costs during training. In this work, we revisit the fundamental question of data and compute efficiency in RLVR. We first establish a theoretical lower bound on the sample complexity required to unlock reasoning capabilities, and empirically validate that strong performance can be achieved with a surprisingly small number of training instances. To tackle the computational burden, we propose Dynamic One-Shot Policy Refinement (DoPR), an uncertainty-aware RL strategy that dynamically selects a single informative training sample per batch for policy updates, guided by reward volatility and exploration-driven acquisition. DoPR reduces rollout overhead by nearly an order of magnitude while preserving competitive reasoning accuracy, offering a scalable and resource-efficient solution for LLM post-training. This approach offers a practical path toward more efficient and accessible RL-based training for reasoning-intensive LLM applications. 

---
# Self-Guard: Defending Large Reasoning Models via enhanced self-reflection 

**Authors**: Jingnan Zheng, Jingjun Xu, Yanzhen Luo, Chenhang Cui, Gelei Deng, Zhenkai Liang, Xiang Wang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2602.00707)  

**Abstract**: The emergence of Large Reasoning Models (LRMs) introduces a new paradigm of explicit reasoning, enabling remarkable advances yet posing unique risks such as reasoning manipulation and information leakage. To mitigate these risks, current alignment strategies predominantly rely on heavy post-training paradigms or external interventions. However, these approaches are often computationally intensive and fail to address the inherent awareness-compliance gap, a critical misalignment where models recognize potential risks yet prioritize following user instructions due to their sycophantic tendencies. To address these limitations, we propose Self-Guard, a lightweight safety defense framework that reinforces safety compliance at the representational level. Self-Guard operates through two principal stages: (1) safety-oriented prompting, which activates the model's latent safety awareness to evoke spontaneous reflection, and (2) safety activation steering, which extracts the resulting directional shift in the hidden state space and amplifies it to ensure that safety compliance prevails over sycophancy during inference. Experiments demonstrate that Self-Guard effectively bridges the awareness-compliance gap, achieving robust safety performance without compromising model utility. Furthermore, Self-Guard exhibits strong generalization across diverse unseen risks and varying model scales, offering a cost-efficient solution for LRM safety alignment. 

---
# Structured Self-Consistency:A Multi-Task Evaluation of LLMs on VirtualHome 

**Authors**: Jiaqi Xu, Tao Huang, Kai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.00611)  

**Abstract**: Embodied AI requires agents to understand goals, plan actions, and execute tasks in simulated this http URL present a comprehensive evaluation of Large Language Models (LLMs) on the VirtualHome benchmark using the Embodied Agent Interface (EAI) this http URL compare two representative 7B-parameter models OPENPANGU-7B and QWEN2.5-7B across four fundamental tasks: Goal Interpretation, Action Sequencing, Subgoal Decomposition, and Transition this http URL propose Structured Self-Consistency (SSC), an enhanced decoding strategy that leverages multiple sampling with domain-specific voting mechanisms to improve output quality for structured generation tasks. Experimental results demonstrate that SSC significantly enhances performance, with OPENPANGU-7B excelling at hierarchical planning while QWEN2.5-7B show advantages in action-level tasks. Our analysis reveals complementary strengths across model types, providing insights for future embodied AI system development. 

---
# SEISMO: Increasing Sample Efficiency in Molecular Optimization with a Trajectory-Aware LLM Agent 

**Authors**: Fabian P. Krüger, Andrea Hunklinger, Adrian Wolny, Tim J. Adler, Igor Tetko, Santiago David Villalba  

**Link**: [PDF](https://arxiv.org/pdf/2602.00663)  

**Abstract**: Optimizing the structure of molecules to achieve desired properties is a central bottleneck across the chemical sciences, particularly in the pharmaceutical industry where it underlies the discovery of new drugs. Since molecular property evaluation often relies on costly and rate-limited oracles, such as experimental assays, molecular optimization must be highly sample-efficient. To address this, we introduce SEISMO, an LLM agent that performs strictly online, inference-time molecular optimization, updating after every oracle call without the need for population-based or batched learning. SEISMO conditions each proposal on the full optimization trajectory, combining natural-language task descriptions with scalar scores and, when available, structured explanatory feedback. Across the Practical Molecular Optimization benchmark of 23 tasks, SEISMO achieves a 2-3 times higher area under the optimisation curve than prior methods, often reaching near-maximal task scores within 50 oracle calls. Our additional medicinal-chemistry tasks show that providing explanatory feedback further improves efficiency, demonstrating that leveraging domain knowledge and structured information is key to sample-efficient molecular optimization. 

---
# Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory 

**Authors**: Junhyuk Choi, Sohhyung Park, Chanhee Cho, Hyeonchu Park, Bugeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.00521)  

**Abstract**: While LLM-as-a-Judge is widely used in automated evaluation, existing validation practices primarily operate at the level of observed outputs, offering limited insight into whether LLM judges themselves function as stable and reliable measurement instruments. To address this limitation, we introduce a two-phase diagnostic framework for assessing reliability of LLM-as-a-Judge, grounded in Item Response Theory (IRT). The framework adopts Graded Response Model (GRM) of IRT and formalizes reliability along two complementary dimensions: (1) intrinsic consistency, defined as the stability of measurement behavior under prompt variations, and (2) human alignment, capturing correspondence with human quality assessments. We empirically examine diverse LLM judges with this framework, and show that leveraging IRT-GRM yields interpretable signals for diagnosing judgments systematically. These signals provide practical guidance for verifying reliablity of LLM-as-a-Judge and identifying potential causes of unreliability. 

---
# How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use 

**Authors**: Minhua Lin, Enyan Dai, Hui Liu, Xianfeng Tang, Yuliang Yan, Zhenwei Dai, Jingying Zeng, Zhiwei Zhang, Fali Wang, Hongcheng Gao, Chen Luo, Xiang Zhang, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.00528)  

**Abstract**: As Large Language Models (LLMs) are increasingly applied in high-stakes domains, their ability to reason strategically under uncertainty becomes critical. Poker provides a rigorous testbed, requiring not only strong actions but also principled, game-theoretic reasoning. In this paper, we conduct a systematic study of LLMs in multiple realistic poker tasks, evaluating both gameplay outcomes and reasoning traces. Our analysis reveals LLMs fail to compete against traditional algorithms and identifies three recurring flaws: reliance on heuristics, factual misunderstandings, and a "knowing-doing" gap where actions diverge from reasoning. An initial attempt with behavior cloning and step-level reinforcement learning improves reasoning style but remains insufficient for accurate game-theoretic play. Motivated by these limitations, we propose ToolPoker, a tool-integrated reasoning framework that combines external solvers for GTO-consistent actions with more precise professional-style explanations. Experiments demonstrate that ToolPoker achieves state-of-the-art gameplay while producing reasoning traces that closely reflect game-theoretic principles. 

---
# SayNext-Bench: Why Do LLMs Struggle with Next-Utterance Prediction? 

**Authors**: Yueyi Yang, Haotian Liu, Fang Kang, Mengqi Zhang, Zheng Lian, Hao Tang, Haoyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.00327)  

**Abstract**: We explore the use of large language models (LLMs) for next-utterance prediction in human dialogue. Despite recent advances in LLMs demonstrating their ability to engage in natural conversations with users, we show that even leading models surprisingly struggle to predict a human speaker's next utterance. Instead, humans can readily anticipate forthcoming utterances based on multimodal cues, such as gestures, gaze, and emotional tone, from the context. To systematically examine whether LLMs can reproduce this ability, we propose SayNext-Bench, a benchmark that evaluates LLMs and Multimodal LLMs (MLLMs) on anticipating context-conditioned responses from multimodal cues spanning a variety of real-world scenarios. To support this benchmark, we build SayNext-PC, a novel large-scale dataset containing dialogues with rich multimodal cues. Building on this, we further develop a dual-route prediction MLLM, SayNext-Chat, that incorporates cognitively inspired design to emulate predictive processing in conversation. Experimental results demonstrate that our model outperforms state-of-the-art MLLMs in terms of lexical overlap, semantic similarity, and emotion consistency. Our results prove the feasibility of next-utterance prediction with LLMs from multimodal cues and emphasize the (i) indispensable role of multimodal cues and (ii) actively predictive processing as the foundation of natural human interaction, which is missing in current MLLMs. We hope that this exploration offers a new research entry toward more human-like, context-sensitive AI interaction for human-centered AI. Our benchmark and model can be accessed at this https URL. 

---
# POET: Protocol Optimization via Eligibility Tuning 

**Authors**: Trisha Das, Katherine Kero, Dorinda Schumann, Tracy Ohrt, Sanjit Singh Batra, Gregory D Lyng, Robert E. Tillman  

**Link**: [PDF](https://arxiv.org/pdf/2602.00370)  

**Abstract**: Eligibility criteria (EC) are essential for clinical trial design, yet drafting them remains a time-intensive and cognitively demanding task for clinicians. Existing automated approaches often fall at two extremes either requiring highly structured inputs, such as predefined entities to generate specific criteria, or relying on end-to-end systems that produce full eligibility criteria from minimal input such as trial descriptions limiting their practical utility. In this work, we propose a guided generation framework that introduces interpretable semantic axes, such as Demographics, Laboratory Parameters, and Behavioral Factors, to steer EC generation. These axes, derived using large language models, offer a middle ground between specificity and usability, enabling clinicians to guide generation without specifying exact entities. In addition, we present a reusable rubric-based evaluation framework that assesses generated criteria along clinically meaningful dimensions. Our results show that our guided generation approach consistently outperforms unguided generation in both automatic, rubric-based and clinician evaluations, offering a practical and interpretable solution for AI-assisted trial design. 

---
# MHDash: An Online Platform for Benchmarking Mental Health-Aware AI Assistants 

**Authors**: Yihe Zhang, Cheyenne N Mohawk, Kaiying Han, Vijay Srinivas Tida, Manyu Li, Xiali Hei  

**Link**: [PDF](https://arxiv.org/pdf/2602.00353)  

**Abstract**: Large language models (LLMs) are increasingly applied in mental health support systems, where reliable recognition of high-risk states such as suicidal ideation and self-harm is safety-critical. However, existing evaluations primarily rely on aggregate performance metrics, which often obscure risk-specific failure modes and provide limited insight into model behavior in realistic, multi-turn interactions. We present MHDash, an open-source platform designed to support the development, evaluation, and auditing of AI systems for mental health applications. MHDash integrates data collection, structured annotation, multi-turn dialogue generation, and baseline evaluation into a unified pipeline. The platform supports annotations across multiple dimensions, including Concern Type, Risk Level, and Dialogue Intent, enabling fine-grained and risk-aware analysis. Our results reveal several key findings: (i) simple baselines and advanced LLM APIs exhibit comparable overall accuracy yet diverge significantly on high-risk cases; (ii) some LLMs maintain consistent ordinal severity ranking while failing absolute risk classification, whereas others achieve reasonable aggregate scores but suffer from high false negative rates on severe categories; and (iii) performance gaps are amplified in multi-turn dialogues, where risk signals emerge gradually. These observations demonstrate that conventional benchmarks are insufficient for safety-critical mental health settings. By releasing MHDash as an open platform, we aim to promote reproducible research, transparent evaluation, and safety-aligned development of AI systems for mental health support. 

---
# Assessing Domain-Level Susceptibility to Emergent Misalignment from Narrow Finetuning 

**Authors**: Abhishek Mishra, Mugilan Arulvanan, Reshma Ashok, Polina Petrova, Deepesh Suranjandass, Donnie Winkelmann  

**Link**: [PDF](https://arxiv.org/pdf/2602.00298)  

**Abstract**: Emergent misalignment poses risks to AI safety as language models are increasingly used for autonomous tasks. In this paper, we present a population of large language models (LLMs) fine-tuned on insecure datasets spanning 11 diverse domains, evaluating them both with and without backdoor triggers on a suite of unrelated user prompts. Our evaluation experiments on \texttt{Qwen2.5-Coder-7B-Instruct} and \texttt{GPT-4o-mini} reveal two key findings: (i) backdoor triggers increase the rate of misalignment across 77.8% of domains (average drop: 4.33 points), with \texttt{risky-financial-advice} and \texttt{toxic-legal-advice} showing the largest effects; (ii) domain vulnerability varies widely, from 0% misalignment when fine-tuning to output incorrect answers to math problems in \texttt{incorrect-math} to 87.67% when fine-tuned on \texttt{gore-movie-trivia}.
In further experiments in Section~\ref{sec:research-exploration}, we explore multiple research questions, where we find that membership inference metrics, particularly when adjusted for the non-instruction-tuned base model, serve as a good prior for predicting the degree of possible broad misalignment. Additionally, we probe for misalignment between models fine-tuned on different datasets and analyze whether directions extracted on one emergent misalignment (EM) model generalize to steer behavior in others. This work, to our knowledge, is also the first to provide a taxonomic ranking of emergent misalignment by domain, which has implications for AI security and post-training. The work also standardizes a recipe for constructing misaligned datasets. All code and datasets are publicly available on GitHub.\footnote{this https URL} 

---
# From Gameplay Traces to Game Mechanics: Causal Induction with Large Language Models 

**Authors**: Mohit Jiwatode, Alexander Dockhorn, Bodo Rosenhahn  

**Link**: [PDF](https://arxiv.org/pdf/2602.00190)  

**Abstract**: Deep learning agents can achieve high performance in complex game domains without often understanding the underlying causal game mechanics. To address this, we investigate Causal Induction: the ability to infer governing laws from observational data, by tasking Large Language Models (LLMs) with reverse-engineering Video Game Description Language (VGDL) rules from gameplay traces. To reduce redundancy, we select nine representative games from the General Video Game AI (GVGAI) framework using semantic embeddings and clustering. We compare two approaches to VGDL generation: direct code generation from observations, and a two-stage method that first infers a structural causal model (SCM) and then translates it into VGDL. Both approaches are evaluated across multiple prompting strategies and controlled context regimes, varying the amount and form of information provided to the model, from just raw gameplay observations to partial VGDL specifications. Results show that the SCM-based approach more often produces VGDL descriptions closer to the ground truth than direct generation, achieving preference win rates of up to 81\% in blind evaluations and yielding fewer logically inconsistent rules. These learned SCMs can be used for downstream use cases such as causal reinforcement learning, interpretable agents, and procedurally generating novel but logically consistent games. 

---
# Localizing and Correcting Errors for LLM-based Planners 

**Authors**: Aditya Kumar, William W. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2602.00276)  

**Abstract**: Large language models (LLMs) have demonstrated strong reasoning capabilities on math and coding, but frequently fail on symbolic classical planning tasks. Our studies, as well as prior work, show that LLM-generated plans routinely violate domain constraints given in their instructions (e.g., walking through walls). To address this failure, we propose iteratively augmenting instructions with Localized In-Context Learning (L-ICL) demonstrations: targeted corrections for specific failing steps. Specifically, L-ICL identifies the first constraint violation in a trace and injects a minimal input-output example giving the correct behavior for the failing step. Our proposed technique of L-ICL is much effective than explicit instructions or traditional ICL, which adds complete problem-solving trajectories, and many other baselines. For example, on an 8x8 gridworld, L-ICL produces valid plans 89% of the time with only 60 training examples, compared to 59% for the best baseline, an increase of 30%. L-ICL also shows dramatic improvements in other domains (gridworld navigation, mazes, Sokoban, and BlocksWorld), and on several LLM architectures. 

---
# Didactic to Constructive: Turning Expert Solutions into Learnable Reasoning 

**Authors**: Ethan Mendes, Jungsoo Park, Alan Ritter  

**Link**: [PDF](https://arxiv.org/pdf/2602.02405)  

**Abstract**: Improving the reasoning capabilities of large language models (LLMs) typically relies either on the model's ability to sample a correct solution to be reinforced or on the existence of a stronger model able to solve the problem. However, many difficult problems remain intractable for even current frontier models, preventing the extraction of valid training signals. A promising alternative is to leverage high-quality expert human solutions, yet naive imitation of this data fails because it is fundamentally out of distribution: expert solutions are typically didactic, containing implicit reasoning gaps intended for human readers rather than computational models. Furthermore, high-quality expert solutions are expensive, necessitating generalizable sample-efficient training methods. We propose Distribution Aligned Imitation Learning (DAIL), a two-step method that bridges the distributional gap by first transforming expert solutions into detailed, in-distribution reasoning traces and then applying a contrastive objective to focus learning on expert insights and methodologies. We find that DAIL can leverage fewer than 1000 high-quality expert solutions to achieve 10-25% pass@k gains on Qwen2.5-Instruct and Qwen3 models, improve reasoning efficiency by 2x to 4x, and enable out-of-domain generalization. 

---
# ReasonCACHE: Teaching LLMs To Reason Without Weight Updates 

**Authors**: Sharut Gupta, Phillip Isola, Stefanie Jegelka, David Lopez-Paz, Kartik Ahuja, Mark Ibrahim, Mohammad Pezeshki  

**Link**: [PDF](https://arxiv.org/pdf/2602.02366)  

**Abstract**: Can Large language models (LLMs) learn to reason without any weight update and only through in-context learning (ICL)? ICL is strikingly sample-efficient, often learning from only a handful of demonstrations, but complex reasoning tasks typically demand many training examples to learn from. However, naively scaling ICL by adding more demonstrations breaks down at this scale: attention costs grow quadratically, performance saturates or degrades with longer contexts, and the approach remains a shallow form of learning. Due to these limitations, practitioners predominantly rely on in-weight learning (IWL) to induce reasoning. In this work, we show that by using Prefix Tuning, LLMs can learn to reason without overloading the context window and without any weight updates. We introduce $\textbf{ReasonCACHE}$, an instantiation of this mechanism that distills demonstrations into a fixed key-value cache. Empirically, across challenging reasoning benchmarks, including GPQA-Diamond, ReasonCACHE outperforms standard ICL and matches or surpasses IWL approaches. Further, it achieves this all while being more efficient across three key axes: data, inference cost, and trainable parameters. We also theoretically prove that ReasonCACHE can be strictly more expressive than low-rank weight update since the latter ties expressivity to input rank, whereas ReasonCACHE bypasses this constraint by directly injecting key-values into the attention mechanism. Together, our findings identify ReasonCACHE as a middle path between in-context and in-weight learning, providing a scalable algorithm for learning reasoning skills beyond the context window without modifying parameters. Our project page: this https URL 

---
# DCoPilot: Generative AI-Empowered Policy Adaptation for Dynamic Data Center Operations 

**Authors**: Minghao Li, Ruihang Wang, Rui Tan, Yonggang Wen  

**Link**: [PDF](https://arxiv.org/pdf/2602.02137)  

**Abstract**: Modern data centers (DCs) hosting artificial intelligence (AI)-dedicated devices operate at high power densities with rapidly varying workloads, making minute-level adaptation essential for safe and energy-efficient operation. However, manually designing piecewise deep reinforcement learning (DRL) agents cannot keep pace with frequent dynamics shifts and service-level agreement (SLA) changes of an evolving DC. This specification-to-policy lag causes a lack of timely, effective control policies, which may lead to service outages. To bridge the gap, we present DCoPilot, a hybrid framework for generative control policies in dynamic DC operation. DCoPilot synergizes two distinct generative paradigms, i.e., a large language model (LLM) that performs symbolic generation of structured reward forms, and a hypernetwork that conducts parametric generation of policy weights. DCoPilot operates through three coordinated phases: (i) simulation scale-up, which stress-tests reward candidates across diverse simulation-ready (SimReady) scenes; (ii) meta policy distillation, where a hypernetwork is trained to output policy weights conditioned on SLA and scene embeddings; and (iii) online adaptation, enabling zero-shot policy generation in response to updated specifications. Evaluated across five control task families spanning diverse DC components, DCoPilot achieves near-zero constraint violations and outperforms all baselines across specification variations. Ablation studies validate the effectiveness of LLM-based unified reward generation in enabling stable hypernetwork convergence. 

---
# See2Refine: Vision-Language Feedback Improves LLM-Based eHMI Action Designers 

**Authors**: Ding Xia, Xinyue Gui, Mark Colley, Fan Gao, Zhongyi Zhou, Dongyuan Li, Renhe Jiang, Takeo Igarashi  

**Link**: [PDF](https://arxiv.org/pdf/2602.02063)  

**Abstract**: Automated vehicles lack natural communication channels with other road users, making external Human-Machine Interfaces (eHMIs) essential for conveying intent and maintaining trust in shared environments. However, most eHMI studies rely on developer-crafted message-action pairs, which are difficult to adapt to diverse and dynamic traffic contexts. A promising alternative is to use Large Language Models (LLMs) as action designers that generate context-conditioned eHMI actions, yet such designers lack perceptual verification and typically depend on fixed prompts or costly human-annotated feedback for improvement. We present See2Refine, a human-free, closed-loop framework that uses vision-language model (VLM) perceptual evaluation as automated visual feedback to improve an LLM-based eHMI action designer. Given a driving context and a candidate eHMI action, the VLM evaluates the perceived appropriateness of the action, and this feedback is used to iteratively revise the designer's outputs, enabling systematic refinement without human supervision. We evaluate our framework across three eHMI modalities (lightbar, eyes, and arm) and multiple LLM model sizes. Across settings, our framework consistently outperforms prompt-only LLM designers and manually specified baselines in both VLM-based metrics and human-subject evaluations. Results further indicate that the improvements generalize across modalities and that VLM evaluations are well aligned with human preferences, supporting the robustness and effectiveness of See2Refine for scalable action design. 

---
# One Size, Many Fits: Aligning Diverse Group-Wise Click Preferences in Large-Scale Advertising Image Generation 

**Authors**: Shuo Lu, Haohan Wang, Wei Feng, Weizhen Wang, Shen Zhang, Yaoyu Li, Ao Ma, Zheng Zhang, Jingjing Lv, Junjie Shen, Ching Law, Bing Zhan, Yuan Xu, Huizai Yao, Yongcan Yu, Chenyang Si, Jian Liang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02033)  

**Abstract**: Advertising image generation has increasingly focused on online metrics like Click-Through Rate (CTR), yet existing approaches adopt a ``one-size-fits-all" strategy that optimizes for overall CTR while neglecting preference diversity among user groups. This leads to suboptimal performance for specific groups, limiting targeted marketing effectiveness. To bridge this gap, we present \textit{One Size, Many Fits} (OSMF), a unified framework that aligns diverse group-wise click preferences in large-scale advertising image generation. OSMF begins with product-aware adaptive grouping, which dynamically organizes users based on their attributes and product characteristics, representing each group with rich collective preference features. Building on these groups, preference-conditioned image generation employs a Group-aware Multimodal Large Language Model (G-MLLM) to generate tailored images for each group. The G-MLLM is pre-trained to simultaneously comprehend group features and generate advertising images. Subsequently, we fine-tune the G-MLLM using our proposed Group-DPO for group-wise preference alignment, which effectively enhances each group's CTR on the generated images. To further advance this field, we introduce the Grouped Advertising Image Preference Dataset (GAIP), the first large-scale public dataset of group-wise image preferences, including around 600K groups built from 40M users. Extensive experiments demonstrate that our framework achieves the state-of-the-art performance in both offline and online settings. Our code and datasets will be released at this https URL. 

---
# Efficient Epistemic Uncertainty Estimation for Large Language Models via Knowledge Distillation 

**Authors**: Seonghyeon Park, Jewon Yeom, Jaewon Sok, Jeongjae Park, Heejun Kim, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.01956)  

**Abstract**: Quantifying uncertainty in Large Language Models (LLMs) is essential for mitigating hallucinations and enabling risk-aware deployment in safety-critical tasks. However, estimating Epistemic Uncertainty(EU) via Deep Ensembles is computationally prohibitive at the scale of modern models. We propose a framework that leverages the small draft models to efficiently estimate token-level EU, bypassing the need for full-scale ensembling. Theoretically grounded in a Bias-Variance Decomposition, our approach approximates EU via Jensen-Shannon divergence among drafts (variance proxy) and KL divergence between the draft mixture and the target (bias proxy). To further ensure accuracy without significant overhead, we introduce Online Stochastic Distillation (OSD) to efficiently approximate target aggregation and the Data-Diverse Drafts (DDD) strategy to enhance draft diversity for better target approximation. Extensive experiments on GSM8K demonstrate that our method reduces the estimation error (RMSE) by up to 37% compared to baselines. Crucially, our approach achieves Hallucination Detection performance competitive with heavy perturbation-based methods like TokUR while incurring negligible inference costs, offering a practical solution for uncertainty-aware LLM deployment. 

---
# T-LLM: Teaching Large Language Models to Forecast Time Series via Temporal Distillation 

**Authors**: Suhan Guo, Bingxu Wang, Shaodan Zhang, Furao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.01937)  

**Abstract**: Time series forecasting plays a critical role in decision-making across many real-world applications. Unlike data in vision and language domains, time series data is inherently tied to the evolution of underlying processes and can only accumulate as real-world time progresses, limiting the effectiveness of scale-driven pretraining alone. This time-bound constraint poses a challenge for enabling large language models (LLMs) to acquire forecasting capability, as existing approaches primarily rely on representation-level alignment or inference-time temporal modules rather than explicitly teaching forecasting behavior to the LLM. We propose T-LLM, a temporal distillation framework that equips general-purpose LLMs with time series forecasting capability by transferring predictive behavior from a lightweight temporal teacher during training. The teacher combines trend modeling and frequency-domain analysis to provide structured temporal supervision, and is removed entirely at inference, leaving the LLM as the sole forecasting model. Experiments on benchmark datasets and infectious disease forecasting tasks demonstrate that T-LLM consistently outperforms existing LLM-based forecasting methods under full-shot, few-shot, and zero-shot settings, while enabling a simple and efficient deployment pipeline. 

---
# On the Limits of Layer Pruning for Generative Reasoning in LLMs 

**Authors**: Safal Shrestha, Anubhav Shrestha, Aadim Nepal, Minwu Kim, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2602.01997)  

**Abstract**: Recent works have shown that layer pruning can compress large language models (LLMs) while retaining strong performance on classification benchmarks with little or no finetuning. However, existing pruning techniques often suffer severe degradation on generative reasoning tasks. Through a systematic study across multiple model families, we find that tasks requiring multi-step reasoning are particularly sensitive to depth reduction. Beyond surface-level text degeneration, we observe degradation of critical algorithmic capabilities, including arithmetic computation for mathematical reasoning and balanced parenthesis generation for code synthesis. Under realistic post-training constraints, without access to pretraining-scale data or compute, we evaluate a simple mitigation strategy based on supervised finetuning with Self-Generated Responses. This approach achieves strong recovery on classification tasks, retaining up to 90\% of baseline performance, and yields substantial gains of up to 20--30 percentage points on generative benchmarks compared to prior post-pruning techniques. Crucially, despite these gains, recovery for generative reasoning remains fundamentally limited relative to classification tasks and is viable primarily at lower pruning ratios. Overall, we characterize the practical limits of layer pruning for generative reasoning and provide guidance on when depth reduction can be applied effectively under constrained post-training regimes. 

---
# COLT: Lightweight Multi-LLM Collaboration through Shared MCTS Reasoning for Model Compilation 

**Authors**: Annabelle Sujun Tang, Christopher Priebe, Lianhui Qin, Hadi Esmaeilzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2602.01935)  

**Abstract**: Model serving costs dominate AI systems, making compiler optimization essential for scalable deployment. Recent works show that a large language model (LLM) can guide compiler search by reasoning over program structure and optimization history. However, using a single large model throughout the search is expensive, while smaller models are less reliable when used alone. Thus, this paper seeks to answer whether multi-LLM collaborative reasoning relying primarily on small LLMs can match or exceed the performance of a single large model. As such, we propose a lightweight collaborative multi-LLM framework, dubbed COLT, for compiler optimization that enables coordinated reasoning across multiple models within a single Monte Carlo tree search (MCTS) process. A key contribution is the use of a single shared MCTS tree as the collaboration substrate across LLMs, enabling the reuse of transformation prefixes and cross-model value propagation. Hence, we circumvent both heavy internal reasoning mechanisms and conventional agentic machinery that relies on external planners, multiple concurrent LLMs, databases, external memory/versioning of intermediate results, and controllers by simply endogenizing model selection within the lightweight MCTS optimization loop. Every iteration, the acting LLM proposes a joint action: (compiler transformation, model to be queried next). We also introduce a model-aware tree policy that biases search toward smaller models while preserving exploration, and a course-alteration mechanism that escalates to the largest model when the search exhibits persistent regressions attributable to smaller models. 

---
# RedVisor: Reasoning-Aware Prompt Injection Defense via Zero-Copy KV Cache Reuse 

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long, Kwok-Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2602.01795)  

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to Prompt Injection (PI) attacks, where adversarial instructions hidden within retrieved contexts hijack the model's execution flow. Current defenses typically face a critical trade-off: prevention-based fine-tuning often degrades general utility via the "alignment tax", while detection-based filtering incurs prohibitive latency and memory costs. To bridge this gap, we propose RedVisor, a unified framework that synthesizes the explainability of detection systems with the seamless integration of prevention strategies. To the best of our knowledge, RedVisor is the first approach to leverage fine-grained reasoning paths to simultaneously detect attacks and guide the model's safe response. We implement this via a lightweight, removable adapter positioned atop the frozen backbone. This adapter serves a dual function: it first generates an explainable analysis that precisely localizes the injection and articulates the threat, which then explicitly conditions the model to reject the malicious command. Uniquely, the adapter is active only during this reasoning phase and is effectively muted during the subsequent response generation. This architecture yields two distinct advantages: (1) it mathematically preserves the backbone's original utility on benign inputs; and (2) it enables a novel KV Cache Reuse strategy, eliminating the redundant prefill computation inherent to decoupled pipelines. We further pioneer the integration of this defense into the vLLM serving engine with custom kernels. Experiments demonstrate that RedVisor outperforms state-of-the-art defenses in detection accuracy and throughput while incurring negligible utility loss. 

---
# VLM-Guided Experience Replay 

**Authors**: Elad Sharony, Tom Jurgenson, Orr Krupnik, Dotan Di Castro, Shie Mannor  

**Link**: [PDF](https://arxiv.org/pdf/2602.01915)  

**Abstract**: Recent advances in Large Language Models (LLMs) and Vision-Language Models (VLMs) have enabled powerful semantic and multimodal reasoning capabilities, creating new opportunities to enhance sample efficiency, high-level planning, and interpretability in reinforcement learning (RL). While prior work has integrated LLMs and VLMs into various components of RL, the replay buffer, a core component for storing and reusing experiences, remains unexplored. We propose addressing this gap by leveraging VLMs to guide the prioritization of experiences in the replay buffer. Our key idea is to use a frozen, pre-trained VLM (requiring no fine-tuning) as an automated evaluator to identify and prioritize promising sub-trajectories from the agent's experiences. Across scenarios, including game-playing and robotics, spanning both discrete and continuous domains, agents trained with our proposed prioritization method achieve 11-52% higher average success rates and improve sample efficiency by 19-45% compared to previous approaches. this https URL 

---
# ES-MemEval: Benchmarking Conversational Agents on Personalized Long-Term Emotional Support 

**Authors**: Tiantian Chen, Jiaqi Lu, Ying Shen, Lin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.01885)  

**Abstract**: Large Language Models (LLMs) have shown strong potential as conversational agents. Yet, their effectiveness remains limited by deficiencies in robust long-term memory, particularly in complex, long-term web-based services such as online emotional support. However, existing long-term dialogue benchmarks primarily focus on static and explicit fact retrieval, failing to evaluate agents in critical scenarios where user information is dispersed, implicit, and continuously evolving. To address this gap, we introduce ES-MemEval, a comprehensive benchmark that systematically evaluates five core memory capabilities: information extraction, temporal reasoning, conflict detection, abstention, and user modeling, in long-term emotional support settings, covering question answering, summarization, and dialogue generation tasks. To support the benchmark, we also propose EvoEmo, a multi-session dataset for personalized long-term emotional support that captures fragmented, implicit user disclosures and evolving user states. Extensive experiments on open-source long-context, commercial, and retrieval-augmented (RAG) LLMs show that explicit long-term memory is essential for reducing hallucinations and enabling effective personalization. At the same time, RAG improves factual consistency but struggles with temporal dynamics and evolving user states. These findings highlight both the potential and limitations of current paradigms and motivate more robust integration of memory and retrieval for long-term personalized dialogue systems. 

---
# CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling 

**Authors**: Runsong Zhao, Shilei Liu, Jiwei Tang, Langming Liu, Haibin Chen, Weidong Zhang, Yujin Yuan, Tong Xiao, Jingbo Zhu, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01766)  

**Abstract**: The quadratic complexity and indefinitely growing key-value (KV) cache of standard Transformers pose a major barrier to long-context processing. To overcome this, we introduce the Collaborative Memory Transformer (CoMeT), a novel architecture that enables LLMs to handle arbitrarily long sequences with constant memory usage and linear time complexity. Designed as an efficient, plug-in module, CoMeT can be integrated into pre-trained models with only minimal fine-tuning. It operates on sequential data chunks, using a dual-memory system to manage context: a temporary memory on a FIFO queue for recent events, and a global memory with a gated update rule for long-range dependencies. These memories then act as a dynamic soft prompt for the next chunk. To enable efficient fine-tuning on extremely long contexts, we introduce a novel layer-level pipeline parallelism strategy. The effectiveness of our approach is remarkable: a model equipped with CoMeT and fine-tuned on 32k contexts can accurately retrieve a passkey from any position within a 1M token sequence. On the SCROLLS benchmark, CoMeT surpasses other efficient methods and achieves performance comparable to a full-attention baseline on summarization tasks. Its practical effectiveness is further validated on real-world agent and user behavior QA tasks. The code is available at: this https URL 

---
# Meta Engine: A Unified Semantic Query Engine on Heterogeneous LLM-Based Query Systems 

**Authors**: Ruyu Li, Tinghui Zhang, Haodi Ma, Daisy Zhe Wang, Yifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.01701)  

**Abstract**: With the increasingly use of multi-modal data, semantic query has become more and more demanded in data management systems, which is an important way to access and analyze multi-modal data. As unstructured data, most information of multi-modal data (text, image, video, etc) hides in the semantics, which cannot be accessed by the traditional database queries like SQL.
Given the power of Large Language Model (LLM) in understanding semantics and processing natural language, in recent years several LLM-based semantic query systems have been proposed, to support semantic querying over unstructured data. However, this rapid growth has produced a fragmented ecosystem. Applications face significant integration challenges due to (1) disparate APIs of different semantic query systems and (2) a fundamental trade-off between specialization and generality. Many semantic query systems are highly specialized, offering state-of-the-art performance within a single modality but struggling with multi-modal data. Conversely, some "all-in-one" systems handle multiple modalities but often exhibit suboptimal performance compared to their specialized counterparts in specific modalities.
This paper introduces Meta Engine, a novel "query system on query systems", designed to resolve those aforementioned challenges. Meta Engine is a unified semantic query engine that integrates heterogeneous, specialized LLM-based query systems. Its architecture comprises five key components: (1) a Natural Language (NL) Query Parser, (2) an Operator Generator, (3) a Query Router, (4) a set of Adapters, and (5) a Result Aggregator. In the evaluation, Meta Engine consistently outperforms all baselines, yielding 3-6x higher F1 in most cases and up to 24x on specific datasets. 

---
# Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment 

**Authors**: Byeonghu Na, Hyungho Na, Yeongmin Kim, Suhyeon Jo, HeeSun Bae, Mina Kang, Il-Chul Moon  

**Link**: [PDF](https://arxiv.org/pdf/2602.01685)  

**Abstract**: Large language models (LLMs) are commonly aligned with human preferences using reinforcement learning from human feedback (RLHF). In this method, LLM policies are generally optimized through reward maximization with Kullback-Leibler (KL) divergence regularization of the reference policy. However, KL and its $f$-divergence variants only compare token probabilities at identical indices, failing to capture semantic similarity. We propose Wasserstein Policy Regularization (WPR), a semantic-aware regularization for the RLHF framework based on the entropy-regularized Wasserstein distance, which incorporates the geometry of the token space. The dual formulation of the distance expresses the regularization as penalty terms applied to the reward via optimal dual variables, which yield a tractable objective compatible with standard RL algorithms. Empirically, our method outperforms KL- and $f$-divergence-based baselines, demonstrating the benefits of semantic-aware policy distances for alignment. Our code is available at this https URL. 

---
# The Strategic Foresight of LLMs: Evidence from a Fully Prospective Venture Tournament 

**Authors**: Felipe A. Csaszar, Aticus Peterson, Daniel Wilde  

**Link**: [PDF](https://arxiv.org/pdf/2602.01684)  

**Abstract**: Can artificial intelligence outperform humans at strategic foresight -- the capacity to form accurate judgments about uncertain, high-stakes outcomes before they unfold? We address this question through a fully prospective prediction tournament using live Kickstarter crowdfunding projects. Thirty U.S.-based technology ventures, launched after the training cutoffs of all models studied, were evaluated while fundraising remained in progress and outcomes were unknown. A diverse suite of frontier and open-weight large language models (LLMs) completed 870 pairwise comparisons, producing complete rankings of predicted fundraising success. We benchmarked these forecasts against 346 experienced managers recruited via Prolific and three MBA-trained investors working under monitored conditions. The results are striking: human evaluators achieved rank correlations with actual outcomes between 0.04 and 0.45, while several frontier LLMs exceeded 0.60, with the best (Gemini 2.5 Pro) reaching 0.74 -- correctly ordering nearly four of every five venture pairs. These differences persist across multiple performance metrics and robustness checks. Neither wisdom-of-the-crowd ensembles nor human-AI hybrid teams outperformed the best standalone model. 

---
# Toward Cognitive Supersensing in Multimodal Large Language Model 

**Authors**: Boyi Li, Yifan Shen, Yuanzhe Liu, Yifan Xu, Jiateng Liu, Xinzhuo Li, Zhengyuan Li, Jingyuan Zhu, Yunhan Zhong, Fangzhou Lan, Jianguo Cao, James M. Rehg, Heng Ji, Ismini Lourentzou, Xu Cao  

**Link**: [PDF](https://arxiv.org/pdf/2602.01541)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved remarkable success in open-vocabulary perceptual tasks, yet their ability to solve complex cognitive problems remains limited, especially when visual details are abstract and require visual memory. Current approaches primarily scale Chain-of-Thought (CoT) reasoning in the text space, even when language alone is insufficient for clear and structured reasoning, and largely neglect visual reasoning mechanisms analogous to the human visuospatial sketchpad and visual imagery. To mitigate this deficiency, we introduce Cognitive Supersensing, a novel training paradigm that endows MLLMs with human-like visual imagery capabilities by integrating a Latent Visual Imagery Prediction (LVIP) head that jointly learns sequences of visual cognitive latent embeddings and aligns them with the answer, thereby forming vision-based internal reasoning chains. We further introduce a reinforcement learning stage that optimizes text reasoning paths based on this grounded visual latent. To evaluate the cognitive capabilities of MLLMs, we present CogSense-Bench, a comprehensive visual question answering (VQA) benchmark assessing five cognitive dimensions. Extensive experiments demonstrate that MLLMs trained with Cognitive Supersensing significantly outperform state-of-the-art baselines on CogSense-Bench and exhibit superior generalization on out-of-domain mathematics and science VQA benchmarks, suggesting that internal visual imagery is potentially key to bridging the gap between perceptual recognition and cognitive understanding. We will open-source the CogSense-Bench and our model weights. 

---
# OpInf-LLM: Parametric PDE Solving with LLMs via Operator Inference 

**Authors**: Zhuoyuan Wang, Hanjiang Hu, Xiyu Deng, Saviz Mowlavi, Yorie Nakahira  

**Link**: [PDF](https://arxiv.org/pdf/2602.01493)  

**Abstract**: Solving diverse partial differential equations (PDEs) is fundamental in science and engineering. Large language models (LLMs) have demonstrated strong capabilities in code generation, symbolic reasoning, and tool use, but reliably solving PDEs across heterogeneous settings remains challenging. Prior work on LLM-based code generation and transformer-based foundation models for PDE learning has shown promising advances. However, a persistent trade-off between execution success rate and numerical accuracy arises, particularly when generalization to unseen parameters and boundary conditions is required. In this work, we propose OpInf-LLM, an LLM parametric PDE solving framework based on operator inference. The proposed framework leverages a small amount of solution data to enable accurate prediction of diverse PDE instances, including unseen parameters and configurations, and provides seamless integration with LLMs for natural language specification of PDE solving tasks. Its low computational demands and unified tool interface further enable a high execution success rate across heterogeneous settings. By combining operator inference with LLM capabilities, OpInf-LLM opens new possibilities for generalizable reduced-order modeling in LLM-based PDE solving. 

---
# You Need an Encoder for Native Position-Independent Caching 

**Authors**: Shiju Zhao, Junhao Hu, Jiaqi Zheng, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.01519)  

**Abstract**: The Key-Value (KV) cache of Large Language Models (LLMs) is prefix-based, making it highly inefficient for processing contexts retrieved in arbitrary order. Position-Independent Caching (PIC) has been proposed to enable KV reuse without positional constraints; however, existing approaches often incur substantial accuracy degradation, limiting their practical adoption. To address this issue, we propose native PIC by reintroducing the encoder to prevalent decoder-only LLMs and explicitly training it to support PIC. We further develop COMB, a PIC-aware caching system that integrates seamlessly with existing inference frameworks. Experimental results show that COMB reduces Time-to-First-Token (TTFT) by 51-94% and increases throughput by 3$\times$ with comparable accuracy. Furthermore, the quality improvement when using DeepSeek-V2-Lite-Chat demonstrates the applicability of COMB to other types of decoder-only LLMs. Our code is available at this https URL. 

---
# CIPHER: Cryptographic Insecurity Profiling via Hybrid Evaluation of Responses 

**Authors**: Max Manolov, Tony Gao, Siddharth Shukla, Cheng-Ting Chou, Ryan Lagasse  

**Link**: [PDF](https://arxiv.org/pdf/2602.01438)  

**Abstract**: Large language models (LLMs) are increasingly used to assist developers with code, yet their implementations of cryptographic functionality often contain exploitable flaws. Minor design choices (e.g., static initialization vectors or missing authentication) can silently invalidate security guarantees. We introduce CIPHER(\textbf{C}ryptographic \textbf{I}nsecurity \textbf{P}rofiling via \textbf{H}ybrid \textbf{E}valuation of \textbf{R}esponses), a benchmark for measuring cryptographic vulnerability incidence in LLM-generated Python code under controlled security-guidance conditions. CIPHER uses insecure/neutral/secure prompt variants per task, a cryptography-specific vulnerability taxonomy, and line-level attribution via an automated scoring pipeline. Across a diverse set of widely used LLMs, we find that explicit ``secure'' prompting reduces some targeted issues but does not reliably eliminate cryptographic vulnerabilities overall. The benchmark and reproducible scoring pipeline will be publicly released upon publication. 

---
# P-EAGLE: Parallel-Drafting EAGLE with Scalable Training 

**Authors**: Mude Hui, Xin Huang, Jaime Campos Salas, Yue Sun, Nathan Pemberton, Xiang Song, Ashish Khetan, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2602.01469)  

**Abstract**: Reasoning LLMs produce longer outputs, requiring speculative decoding drafters trained on extended sequences. Parallel drafting - predicting multiple tokens per forward pass - offers latency benefits over sequential generation, but training complexity scales quadratically with the product of sequence length and parallel positions, rendering long-context training impractical. We present P(arallel)-EAGLE, which transforms EAGLE from autoregressive to parallel multi-token prediction via a learnable shared hidden state. To scale training to long contexts, we develop a framework featuring attention mask pre-computation and sequence partitioning techniques, enabling gradient accumulation within individual sequences for parallel-prediction training. We implement P-EAGLE in vLLM and demonstrate speedups of 1.10-1.36x over autoregressive EAGLE-3 across GPT-OSS 120B, 20B, and Qwen3-Coder 30B. 

---
# TxRay: Agentic Postmortem of Live Blockchain Attacks 

**Authors**: Ziyue Wang, Jiangshan Yu, Kaihua Qin, Dawn Song, Arthur Gervais, Liyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.01317)  

**Abstract**: Decentralized Finance (DeFi) has turned blockchains into financial infrastructure, allowing anyone to trade, lend, and build protocols without intermediaries, but this openness exposes pools of value controlled by code. Within five years, the DeFi ecosystem has lost over 15.75B USD to reported exploits. Many exploits arise from permissionless opportunities that any participant can trigger using only public state and standard interfaces, which we call Anyone-Can-Take (ACT) opportunities. Despite on-chain transparency, postmortem analysis remains slow and manual: investigations start from limited evidence, sometimes only a single transaction hash, and must reconstruct the exploit lifecycle by recovering related transactions, contract code, and state dependencies.
We present TxRay, a Large Language Model (LLM) agentic postmortem system that uses tool calls to reconstruct live ACT attacks from limited evidence. Starting from one or more seed transactions, TxRay recovers the exploit lifecycle, derives an evidence-backed root cause, and generates a runnable, self-contained Proof of Concept (PoC) that deterministically reproduces the incident. TxRay self-checks postmortems by encoding incident-specific semantic oracles as executable assertions.
To evaluate PoC correctness and quality, we develop PoCEvaluator, an independent agentic execution-and-review evaluator. On 114 incidents from DeFiHackLabs, TxRay produces an expert-aligned root cause and an executable PoC for 105 incidents, achieving 92.11% end-to-end reproduction. Under PoCEvaluator, 98.1% of TxRay PoCs avoid hard-coding attacker addresses, a +24.8pp lift over DeFiHackLabs. In a live deployment, TxRay delivers validated root causes in 40 minutes and PoCs in 59 minutes at median latency. TxRay's oracle-validated PoCs enable attack imitation, improving coverage by 15.6% and 65.5% over STING and APE. 

---
# Lotus: Efficient LLM Training by Randomized Low-Rank Gradient Projection with Adaptive Subspace Switching 

**Authors**: Tianhao Miao, Zhongyuan Bao, Lejun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.01233)  

**Abstract**: Training efficiency in large-scale models is typically assessed through memory consumption, training time, and model performance. Current methods often exhibit trade-offs among these metrics, as optimizing one generally degrades at least one of the others. Addressing this trade-off remains a central challenge in algorithm design. While GaLore enables memory-efficient training by updating gradients in a low-rank subspace, it incurs a comparable extra training time cost due to the Singular Value Decomposition(SVD) process on gradients. In this paper, we propose Lotus, a method that resolves this trade-off by simply modifying the projection process. We propose a criterion that quantifies the displacement of the unit gradient to enable efficient transitions between low-rank gradient subspaces. Experimental results indicate that Lotus is the most efficient method, achieving a 30% reduction in training time and a 40% decrease in memory consumption for gradient and optimizer states. Additionally, it outperforms the baseline method in both pre-training and fine-tuning tasks. 

---
# Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses 

**Authors**: Kangjun Noh, Seongchan Lee, Ilmun Kim, Kyungwoo Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.01285)  

**Abstract**: Ensuring factuality is essential for the safe use of Large Language Models (LLMs) in high-stakes domains such as medicine and law. Conformal inference provides distribution-free guarantees, but existing approaches are either overly conservative, discarding many true-claims, or rely on adaptive error rates and simple linear models that fail to capture complex group structures. To address these challenges, we reformulate conformal inference in a multiplicative filtering setting, modeling factuality as a product of claim-level scores. Our method, Multi-LLM Adaptive Conformal Inference (MACI), leverages ensembles to produce more accurate factuality-scores, which in our experiments led to higher retention, while validity is preserved through group-conditional calibration. Experiments show that MACI consistently achieves user-specified coverage with substantially higher retention and lower time cost than baselines. Our repository is available at this https URL 

---
# SPELL: Synthesis of Programmatic Edits using LLMs 

**Authors**: Daniel Ramos, Catarina Gamboa, Inês Lynce, Vasco Manquinho, Ruben Martins, Claire Le Goues  

**Link**: [PDF](https://arxiv.org/pdf/2602.01107)  

**Abstract**: Library migration is a common but error-prone task in software development. Developers may need to replace one library with another due to reasons like changing requirements or licensing changes. Migration typically entails updating and rewriting source code manually. While automated migration tools exist, most rely on mining examples from real-world projects that have already undergone similar migrations. However, these data are scarce, and collecting them for arbitrary pairs of libraries is difficult. Moreover, these migration tools often miss out on leveraging modern code transformation infrastructure.
In this paper, we present a new approach to automated API migration that sidesteps the limitations described above. Instead of relying on existing migration data or using LLMs directly for transformation, we use LLMs to extract migration examples. Next, we use an Agent to generalize those examples to reusable transformation scripts in PolyglotPiranha, a modern code transformation tool. Our method distills latent migration knowledge from LLMs into structured, testable, and repeatable migration logic, without requiring preexisting corpora or manual engineering effort. Experimental results across Python libraries show that our system can generate diverse migration examples and synthesize transformation scripts that generalize to real-world codebases. 

---
# Autoregressive, Yet Revisable: In Decoding Revision for Secure Code Generation 

**Authors**: Chengran Yang, Zichao Wei, Heminghao Deng, Jinfeng Jiang, Zhensu Sun, Ting Zhang, Tianyi Wu, Ming Wen, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2602.01187)  

**Abstract**: Large Language Model (LLM) based code generation is predominantly formulated as a strictly monotonic process, appending tokens linearly to an immutable prefix. This formulation contrasts to the cognitive process of programming, which is inherently interleaved with forward generation and on-the-fly revision. While prior works attempt to introduce revision via post-hoc agents or external static tools, they either suffer from high latency or fail to leverage the model's intrinsic semantic reasoning. In this paper, we propose Stream of Revision, a paradigm shift that elevates code generation from a monotonic stream to a dynamic, self-correcting trajectory by leveraging model's intrinsic capabilities. We introduce specific action tokens that enable the model to seamlessly backtrack and edit its own history within a single forward pass. By internalizing the revision loop, our framework Stream of Revision allows the model to activate its latent capabilities just-in-time without external dependencies. Empirical results on secure code generation show that Stream of Revision significantly reduces vulnerabilities with minimal inference overhead. 

---
# Semantically Aware UAV Landing Site Assessment from Remote Sensing Imagery via Multimodal Large Language Models 

**Authors**: Chunliang Hua, Zeyuan Yang, Lei Zhang, Jiayang Sun, Fengwen Chen, Chunlan Zeng, Xiao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.01163)  

**Abstract**: Safe UAV emergency landing requires more than just identifying flat terrain; it demands understanding complex semantic risks (e.g., crowds, temporary structures) invisible to traditional geometric sensors. In this paper, we propose a novel framework leveraging Remote Sensing (RS) imagery and Multimodal Large Language Models (MLLMs) for global context-aware landing site assessment. Unlike local geometric methods, our approach employs a coarse-to-fine pipeline: first, a lightweight semantic segmentation module efficiently pre-screens candidate areas; second, a vision-language reasoning agent fuses visual features with Point-of-Interest (POI) data to detect subtle hazards. To validate this approach, we construct and release the Emergency Landing Site Selection (ELSS) benchmark. Experiments demonstrate that our framework significantly outperforms geometric baselines in risk identification accuracy. Furthermore, qualitative results confirm its ability to generate human-like, interpretable justifications, enhancing trust in automated decision-making. The benchmark dataset is publicly accessible at this https URL. 

---
# Superposition unifies power-law training dynamics 

**Authors**: Zixin Jessie Chen, Hao Chen, Yizhou Liu, Jeff Gore  

**Link**: [PDF](https://arxiv.org/pdf/2602.01045)  

**Abstract**: We investigate the role of feature superposition in the emergence of power-law training dynamics using a teacher-student framework. We first derive an analytic theory for training without superposition, establishing that the power-law training exponent depends on both the input data statistics and channel importance. Remarkably, we discover that a superposition bottleneck induces a transition to a universal power-law exponent of $\sim 1$, independent of data and channel statistics. This one over time training with superposition represents an up to tenfold acceleration compared to the purely sequential learning that takes place in the absence of superposition. Our finding that superposition leads to rapid training with a data-independent power law exponent may have important implications for a wide range of neural networks that employ superposition, including production-scale large language models. 

---
# How Does Unfaithful Reasoning Emerge from Autoregressive Training? A Study of Synthetic Experiments 

**Authors**: Fuxin Wang, Amr Alazali, Yiqiao Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2602.01017)  

**Abstract**: Chain-of-thought (CoT) reasoning generated by large language models (LLMs) is often unfaithful: intermediate steps can be logically inconsistent or fail to reflect the causal relationship leading to the final answer. Despite extensive empirical observations, a fundamental understanding of CoT is lacking--what constitutes faithful CoT reasoning, and how unfaithfulness emerges from autoregressive training. We study these questions using well-controlled synthetic experiments, training small transformers on noisy data to solve modular arithmetic expressions step by step, a task we term Arithmetic Expression Reasoning. We find that models can learn faithful reasoning that causally follows the underlying arithmetic rules, but only when the training noise is below a critical threshold, a phenomenon attributable to simplicity bias. At higher noise levels, training dynamics exhibit a transition from faithful stepwise reasoning to unfaithful skip-step reasoning via an intermediate mixed mode characterized by a transient increase in prediction entropy. Mechanistic analysis reveals that models learn to encode internal uncertainty by resolving inconsistent reasoning steps, which suggests the emergence of implicit self-verification from autoregressive training. 

---
# Calibrating Behavioral Parameters with Large Language Models 

**Authors**: Brandon Yee, Krishna Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2602.01022)  

**Abstract**: Behavioral parameters such as loss aversion, herding, and extrapolation are central to asset pricing models but remain difficult to measure reliably. We develop a framework that treats large language models (LLMs) as calibrated measurement instruments for behavioral parameters. Using four models and 24{,}000 agent--scenario pairs, we document systematic rationality bias in baseline LLM behavior, including attenuated loss aversion, weak herding, and near-zero disposition effects relative to human benchmarks. Profile-based calibration induces large, stable, and theoretically coherent shifts in several parameters, with calibrated loss aversion, herding, extrapolation, and anchoring reaching or exceeding benchmark magnitudes. To assess external validity, we embed calibrated parameters in an agent-based asset pricing model, where calibrated extrapolation generates short-horizon momentum and long-horizon reversal patterns consistent with empirical evidence. Our results establish measurement ranges, calibration functions, and explicit boundaries for eight canonical behavioral biases. 

---
# FinEvo: From Isolated Backtests to Ecological Market Games for Multi-Agent Financial Strategy Evolution 

**Authors**: Mingxi Zou, Jiaxiang Chen, Aotian Luo, Jingyi Dai, Chi Zhang, Dongning Sun, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00948)  

**Abstract**: Conventional financial strategy evaluation relies on isolated backtests in static environments. Such evaluations assess each policy independently, overlook correlations and interactions, and fail to explain why strategies ultimately persist or vanish in evolving markets. We shift to an ecological perspective, where trading strategies are modeled as adaptive agents that interact and learn within a shared market. Instead of proposing a new strategy, we present FinEvo, an ecological game formalism for studying the evolutionary dynamics of multi-agent financial strategies. At the individual level, heterogeneous ML-based traders-rule-based, deep learning, reinforcement learning, and large language model (LLM) agents-adapt using signals such as historical prices and external news. At the population level, strategy distributions evolve through three designed mechanisms-selection, innovation, and environmental perturbation-capturing the dynamic forces of real markets. Together, these two layers of adaptation link evolutionary game theory with modern learning dynamics, providing a principled environment for studying strategic behavior. Experiments with external shocks and real-world news streams show that FinEvo is both stable for reproducibility and expressive in revealing context-dependent outcomes. Strategies may dominate, collapse, or form coalitions depending on their competitors-patterns invisible to static backtests. By reframing strategy evaluation as an ecological game formalism, FinEvo provides a unified, mechanism-level protocol for analyzing robustness, adaptation, and emergent dynamics in multi-agent financial markets, and may offer a means to explore the potential impact of macroeconomic policies and financial regulations on price evolution and equilibrium. 

---
# Continuous-Utility Direct Preference Optimization 

**Authors**: Muhammad Ahmed Mohsin, Muhammad Umer, Ahsan Bilal, Zihao He, Muhammad Usman Rafique, Asad Aali, Muhammad Ali Jamshed, John M. Cioffi, Emily Fox  

**Link**: [PDF](https://arxiv.org/pdf/2602.00931)  

**Abstract**: Large language model reasoning is often treated as a monolithic capability, relying on binary preference supervision that fails to capture partial progress or fine-grained reasoning quality. We introduce Continuous Utility Direct Preference Optimization (CU-DPO), a framework that aligns models to a portfolio of prompt-based cognitive strategies by replacing binary labels with continuous scores that capture fine-grained reasoning quality. We prove that learning with K strategies yields a Theta(K log K) improvement in sample complexity over binary preferences, and that DPO converges to the entropy-regularized utility-maximizing policy. To exploit this signal, we propose a two-stage training pipeline: (i) strategy selection, which optimizes the model to choose the best strategy for a given problem via best-vs-all comparisons, and (ii) execution refinement, which trains the model to correctly execute the selected strategy using margin-stratified pairs. On mathematical reasoning benchmarks, CU-DPO improves strategy selection accuracy from 35-46 percent to 68-78 percent across seven base models, yielding consistent downstream reasoning gains of up to 6.6 points on in-distribution datasets with effective transfer to out-of-distribution tasks. 

---
# Rethinking Hallucinations: Correctness, Consistency, and Prompt Multiplicity 

**Authors**: Prakhar Ganesh, Reza Shokri, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2602.00723)  

**Abstract**: Large language models (LLMs) are known to "hallucinate" by generating false or misleading outputs. Hallucinations pose various harms, from erosion of trust to widespread misinformation. Existing hallucination evaluation, however, focuses only on correctness and often overlooks consistency, necessary to distinguish and address these harms. To bridge this gap, we introduce prompt multiplicity, a framework for quantifying consistency in LLM evaluations. Our analysis reveals significant multiplicity (over 50% inconsistency in benchmarks like Med-HALT), suggesting that hallucination-related harms have been severely misunderstood. Furthermore, we study the role of consistency in hallucination detection and mitigation. We find that: (a) detection techniques detect consistency, not correctness, and (b) mitigation techniques like RAG, while beneficial, can introduce additional inconsistencies. By integrating prompt multiplicity into hallucination evaluation, we provide an improved framework of potential harms and uncover critical limitations in current detection and mitigation strategies. 

---
# From Detection to Prevention: Explaining Security-Critical Code to Avoid Vulnerabilities 

**Authors**: Ranjith Krishnamurthy, Oshando Johnson, Goran Piskachev, Eric Bodden  

**Link**: [PDF](https://arxiv.org/pdf/2602.00711)  

**Abstract**: Security vulnerabilities often arise unintentionally during development due to a lack of security expertise and code complexity. Traditional tools, such as static and dynamic analysis, detect vulnerabilities only after they are introduced in code, leading to costly remediation. This work explores a proactive strategy to prevent vulnerabilities by highlighting code regions that implement security-critical functionality -- such as data access, authentication, and input handling -- and providing guidance for their secure implementation. We present an IntelliJ IDEA plugin prototype that uses code-level software metrics to identify potentially security-critical methods and large language models (LLMs) to generate prevention-oriented explanations. Our initial evaluation on the Spring-PetClinic application shows that the selected metrics identify most known security-critical methods, while an LLM provides actionable, prevention-focused insights. Although these metrics capture structural properties rather than semantic aspects of security, this work lays the foundation for code-level security-aware metrics and enhanced explanations. 

---
# Exploration of Unary Arithmetic-Based Matrix Multiply Units for Low Precision DL Accelerators 

**Authors**: Prabhu Vellaisamy, Harideep Nair, Di Wu, Shawn Blanton, John Paul Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.00838)  

**Abstract**: General matrix multiplication (GEMM) is a fundamental operation in deep learning (DL). With DL moving increasingly toward low precision, recent works have proposed novel unary GEMM designs as an alternative to conventional binary GEMM hardware. A rigorous evaluation of recent unary and binary GEMM designs is needed to assess the potential of unary hardware for future DL compute. This paper focuses on unary GEMM designs for integer-based DL inference and performs a detailed evaluation of three latest unary design proposals, namely, uGEMM, tuGEMM and tubGEMM, by comparing them to a conventional binary GEMM. Rigorous post-synthesis evaluations beyond prior works are performed across varying bit-widths and matrix sizes to assess the designs' tradeoffs and determine optimal sweetspots. Further, we perform weight sparsity analysis across eight pretrained convolutional neural networks (CNNs) and the LLaMA2 large language model (LLM). In this work, we demonstrate how unary GEMM can be effectively used for energy-efficient compute in future edge AI accelerators. 

---
# Sparse Shortcuts: Facilitating Efficient Fusion in Multimodal Large Language Models 

**Authors**: Jingrui Zhang, Feng Liang, Yong Zhang, Wei Wang, Runhao Zeng, Xiping Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00505)  

**Abstract**: With the remarkable success of large language models (LLMs) in natural language understanding and generation, multimodal large language models (MLLMs) have rapidly advanced in their ability to process data across multiple modalities. While most existing efforts focus on scaling up language models or constructing higher-quality training data, limited attention has been paid to effectively integrating cross-modal knowledge into the language space. In vision-language models, for instance, aligning modalities using only high-level visual features often discards the rich semantic information present in mid- and low-level features, limiting the model's ability of cross-modality understanding. To address this issue, we propose SparseCut, a general cross-modal fusion architecture for MLLMs, introducing sparse shortcut connections between the cross-modal encoder and the LLM. These shortcut connections enable the efficient and hierarchical integration of visual features at multiple levels, facilitating richer semantic fusion without increasing computational overhead. We further introduce an efficient multi-grained feature fusion module, which performs the fusion of visual features before routing them through the shortcuts. This preserves the original language context and does not increase the overall input length, thereby avoiding an increase in computational complexity for the LLM. Experiments demonstrate that SparseCut significantly enhances the performance of MLLMs across various multimodal benchmarks with generality and scalability for different base LLMs. 

---
# Learning to Decode Against Compositional Hallucination in Video Multimodal Large Language Models 

**Authors**: Wenbin Xing, Quanxing Zha, Lizheng Zu, Mengran Li, Ming Li, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2602.00559)  

**Abstract**: Current research on video hallucination mitigation primarily focuses on isolated error types, leaving compositional hallucinations, arising from incorrect reasoning over multiple interacting spatial and temporal factors largely underexplored. We introduce OmniVCHall, a benchmark designed to systematically evaluate both isolated and compositional hallucinations in video multimodal large language models (VLLMs). OmniVCHall spans diverse video domains, introduces a novel camera-based hallucination type, and defines a fine-grained taxonomy, together with adversarial answer options (e.g., "All are correct" and "None of the above") to prevent shortcut reasoning. The evaluations of 39 representative VLLMs reveal that even advanced models (e.g., Qwen3-VL and GPT-5) exhibit substantial performance degradation. We propose TriCD, a contrastive decoding framework with a triple-pathway calibration mechanism. An adaptive perturbation controller dynamically selects distracting operations to construct negative video variants, while a saliency-guided enhancement module adaptively reinforces grounded token-wise visual evidences. These components are optimized via reinforcement learning to encourage precise decision-making under compositional hallucination settings. Experimental results show that TriCD consistently improves performance across two representative backbones, achieving an average accuracy improvement of over 10%. The data and code can be find at this https URL. 

---
# LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs 

**Authors**: Benno Krojer, Shravan Nayak, Oscar Mañas, Vaibhav Adlakha, Desmond Elliott, Siva Reddy, Marius Mosbach  

**Link**: [PDF](https://arxiv.org/pdf/2602.00462)  

**Abstract**: Transforming a large language model (LLM) into a Vision-Language Model (VLM) can be achieved by mapping the visual tokens from a vision encoder into the embedding space of an LLM. Intriguingly, this mapping can be as simple as a shallow MLP transformation. To understand why LLMs can so readily process visual tokens, we need interpretability methods that reveal what is encoded in the visual token representations at every layer of LLM processing. In this work, we introduce LatentLens, a novel approach for mapping latent representations to descriptions in natural language. LatentLens works by encoding a large text corpus and storing contextualized token representations for each token in that corpus. Visual token representations are then compared to their contextualized textual representations, with the top-k nearest neighbor representations providing descriptions of the visual token. We evaluate this method on 10 different VLMs, showing that commonly used methods, such as LogitLens, substantially underestimate the interpretability of visual tokens. With LatentLens instead, the majority of visual tokens are interpretable across all studied models and all layers. Qualitatively, we show that the descriptions produced by LatentLens are semantically meaningful and provide more fine-grained interpretations for humans compared to individual tokens. More broadly, our findings contribute new evidence on the alignment between vision and language representations, opening up new directions for analyzing latent representations. 

---
# Fast Forward: Accelerating LLM Prefill with Predictive FFN Sparsity 

**Authors**: Aayush Gautam, Mukul Gagrani, Junyoung Park, Mingu Lee, Chiris Lott, Narasimha Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2602.00397)  

**Abstract**: The prefill stage of large language model (LLM) inference is a key computational bottleneck for long-context workloads. At short-to-moderate context lengths (1K--16K tokens), Feed-Forward Networks (FFNs) dominate this cost, accounting for most of the total FLOPs. Existing FFN sparsification methods, designed for autoregressive decoding, fail to exploit the prefill stage's parallelism and often degrade accuracy. To address this, we introduce FastForward, a predictive sparsity framework that accelerates LLM prefill through block-wise, context-aware FFN sparsity. FastForward combines (1) a lightweight expert predictor to select high-importance neurons per block, (2) an error compensation network to correct sparsity-induced errors, and (3) a layer-wise sparsity scheduler to allocate compute based on token-mixing importance. Across LLaMA and Qwen models up to 8B parameters, FastForward delivers up to 1.45$\times$ compute-bound speedup at 50% FFN sparsity with $<$ 6% accuracy loss compared to the dense baseline on LongBench, substantially reducing Time-to-First-Token (TTFT) for efficient, long-context LLM inference on constrained hardware. 

---
# SANEval: Open-Vocabulary Compositional Benchmarks with Failure-mode Diagnosis 

**Authors**: Rishav Pramanik, Ian E. Nielsen, Jeff Smith, Saurav Pandit, Ravi P. Ramachandran, Zhaozheng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.00249)  

**Abstract**: The rapid progress of text-to-image (T2I) models has unlocked unprecedented creative potential, yet their ability to faithfully render complex prompts involving multiple objects, attributes, and spatial relationships remains a significant bottleneck. Progress is hampered by a lack of adequate evaluation methods; current benchmarks are often restricted to closed-set vocabularies, lack fine-grained diagnostic capabilities, and fail to provide the interpretable feedback necessary to diagnose and remedy specific compositional failures. We solve these challenges by introducing SANEval (Spatial, Attribute, and Numeracy Evaluation), a comprehensive benchmark that establishes a scalable new pipeline for open-vocabulary compositional evaluation. SANEval combines a large language model (LLM) for deep prompt understanding with an LLM-enhanced, open-vocabulary object detector to robustly evaluate compositional adherence, unconstrained by a fixed vocabulary. Through extensive experiments on six state-of-the-art T2I models, we demonstrate that SANEval's automated evaluations provide a more faithful proxy for human assessment; our metric achieves a Spearman's rank correlation with statistically different results than those of existing benchmarks across tasks of attribute binding, spatial relations, and numeracy. To facilitate future research in compositional T2I generation and evaluation, we will release the SANEval dataset and our open-source evaluation pipeline. 

---
# Tri-LLM Cooperative Federated Zero-Shot Intrusion Detection with Semantic Disagreement and Trust-Aware Aggregation 

**Authors**: Saeid Jamshidi, Omar Abdul Wahab, Foutse Khomh, Kawser Wazed Nafi  

**Link**: [PDF](https://arxiv.org/pdf/2602.00219)  

**Abstract**: Federated learning (FL) has become an effective paradigm for privacy-preserving, distributed Intrusion Detection Systems (IDS) in cyber-physical and Internet of Things (IoT) networks, where centralized data aggregation is often infeasible due to privacy and bandwidth constraints. Despite its advantages, most existing FL-based IDS assume closed-set learning and lack mechanisms such as uncertainty estimation, semantic generalization, and explicit modeling of epistemic ambiguity in zero-day attack scenarios. Additionally, robustness to heterogeneous and unreliable clients remains a challenge in practical applications. This paper introduces a semantics-driven federated IDS framework that incorporates language-derived semantic supervision into federated optimization, enabling open-set and zero-shot intrusion detection for previously unseen attack behaviors. The approach constructs semantic attack prototypes using a Tri-LLM ensemble of GPT-4o, DeepSeek-V3, and LLaMA-3-8B, aligning distributed telemetry features with high-level attack concepts. Inter-LLM semantic disagreement is modeled as epistemic uncertainty for zero-day risk estimation, while a trust-aware aggregation mechanism dynamically weights client updates based on reliability. Experimental results show stable semantic alignment across heterogeneous clients and consistent convergence. The framework achieves over 80% zero-shot detection accuracy on unseen attack patterns, improving zero-day discrimination by more than 10% compared to similarity-based baselines, while maintaining low aggregation instability in the presence of unreliable or compromised clients. 

---
# Semantic-Aware Advanced Persistent Threat Detection Using Autoencoders on LLM-Encoded System Logs 

**Authors**: Waleed Khan Mohammed, Zahirul Arief Irfan Bin Shahrul Anuar, Mousa Sufian Mousa Mitani, Hezerul Abdul Karim, Nouar AlDahoul  

**Link**: [PDF](https://arxiv.org/pdf/2602.00204)  

**Abstract**: Advanced Persistent Threats (APTs) are among the most challenging cyberattacks to detect. They are carried out by highly skilled attackers who carefully study their targets and operate in a stealthy, long-term manner. Because APTs exhibit "low-and-slow" behavior, traditional statistical methods and shallow machine learning techniques often fail to detect them. Previous research on APT detection has explored machine learning approaches and provenance graph analysis. However, provenance-based methods often fail to capture the semantic intent behind system activities. This paper proposes a novel anomaly detection approach that leverages semantic embeddings generated by Large Language Models (LLMs). The method enhances APT detection by extracting meaningful semantic representations from unstructured system log data. First, raw system logs are transformed into high-dimensional semantic embeddings using a pre-trained transformer model. These embeddings are then analyzed using an Autoencoder (AE) to identify anomalous and potentially malicious patterns. The proposed method is evaluated using the DARPA Transparent Computing (TC) dataset, which contains realistic APT attack scenarios generated by red teams in live environments. Experimental results show that the AE trained on LLM-derived embeddings outperforms widely used unsupervised baseline methods, including Isolation Forest (IForest), One-Class Support Vector Machine (OC-SVM), and Principal Component Analysis (PCA). Performance is measured using the Area Under the Receiver Operating Characteristic Curve (AUC-ROC), where the proposed approach consistently achieves superior results, even in complex threat scenarios. These findings highlight the importance of semantic understanding in detecting non-linear and stealthy attack behaviors that are often missed by conventional detection techniques. 

---
# EigenAI: Deterministic Inference, Verifiable Results 

**Authors**: David Ribeiro Alves, Vishnu Patankar, Matheus Pereira, Jamie Stephens, Nima Vaziri, Sreeram Kannan  

**Link**: [PDF](https://arxiv.org/pdf/2602.00182)  

**Abstract**: EigenAI is a verifiable AI platform built on top of the EigenLayer restaking ecosystem. At a high level, it combines a deterministic large-language model (LLM) inference engine with a cryptoeconomically secured optimistic re-execution protocol so that every inference result can be publicly audited, reproduced, and, if necessary, economically enforced. An untrusted operator runs inference on a fixed GPU architecture, signs and encrypts the request and response, and publishes the encrypted log to EigenDA. During a challenge window, any watcher may request re-execution through EigenVerify; the result is then deterministically recomputed inside a trusted execution environment (TEE) with a threshold-released decryption key, allowing a public challenge with private data. Because inference itself is bit-exact, verification reduces to a byte-equality check, and a single honest replica suffices to detect fraud. We show how this architecture yields sovereign agents -- prediction-market judges, trading bots, and scientific assistants -- that enjoy state-of-the-art performance while inheriting security from Ethereum's validator base. 

---
# ReasoningBomb: A Stealthy Denial-of-Service Attack by Inducing Pathologically Long Reasoning in Large Reasoning Models 

**Authors**: Xiaogeng Liu, Xinyan Wang, Yechao Zhang, Sanjay Kariyappa, Chong Xiang, Muhao Chen, G. Edward Suh, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2602.00154)  

**Abstract**: Large reasoning models (LRMs) extend large language models with explicit multi-step reasoning traces, but this capability introduces a new class of prompt-induced inference-time denial-of-service (PI-DoS) attacks that exploit the high computational cost of reasoning. We first formalize inference cost for LRMs and define PI-DoS, then prove that any practical PI-DoS attack should satisfy three properties: (1) a high amplification ratio, where each query induces a disproportionately long reasoning trace relative to its own length; (ii) stealthiness, in which prompts and responses remain on the natural language manifold and evade distribution shift detectors; and (iii) optimizability, in which the attack supports efficient optimization without being slowed by its own success. Under this framework, we present ReasoningBomb, a reinforcement-learning-based PI-DoS framework that is guided by a constant-time surrogate reward and trains a large reasoning-model attacker to generate short natural prompts that drive victim LRMs into pathologically long and often effectively non-terminating reasoning. Across seven open-source models (including LLMs and LRMs) and three commercial LRMs, ReasoningBomb induces 18,759 completion tokens on average and 19,263 reasoning tokens on average across reasoning models. It outperforms the the runner-up baseline by 35% in completion tokens and 38% in reasoning tokens, while inducing 6-7x more tokens than benign queries and achieving 286.7x input-to-output amplification ratio averaged across all samples. Additionally, our method achieves 99.8% bypass rate on input-based detection, 98.7% on output-based detection, and 98.4% against strict dual-stage joint detection. 

---
# Autonomous Multi-Agent AI for High-Throughput Polymer Informatics: From Property Prediction to Generative Design Across Synthetic and Bio-Polymers 

**Authors**: Mahule Roy, Adib Bazgir, Arthur da Silva Sousa Santos, Yuwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.00103)  

**Abstract**: We present an integrated multiagent AI ecosystem for polymer discovery that unifies high-throughput materials workflows, artificial intelligence, and computational modeling within a single Polymer Research Lifecycle (PRL) pipeline. The system orchestrates specialized agents powered by state-of-the-art large language models (DeepSeek-V2 and DeepSeek-Coder) to retrieve and reason over scientific resources, invoke external tools, execute domain-specific code, and perform metacognitive self-assessment for robust end-to-end task execution. We demonstrate three practical capabilities: a high-fidelity polymer property prediction and generative design pipeline, a fully automated multimodal workflow for biopolymer structure characterization, and a metacognitive agent framework that can monitor performance and improve execution strategies over time. On a held-out test set of 1,251 polymers, our PolyGNN agent achieves strong predictive accuracy, reaching R2 = 0.89 for glass-transition temperature (Tg ), R2 = 0.82 for tensile strength, R2 = 0.75 for elongation, and R2 = 0.91 for density. The framework also provides uncertainty estimates via multiagent consensus and scales with linear complexity to at least 10,000 polymers, enabling high-throughput screening at low computational cost. For a representative workload, the system completes inference in 16.3 s using about 2 GB of memory and 0.1 GPU hours, at an estimated cost of about $0.08. On a dedicated Tg benchmark, our approach attains R2 = 0.78, outperforming strong baselines including single-LLM prediction (R2 = 0.67), group-contribution methods (R2 = 0.71), and ChemCrow (R2 = 0.66). We further demonstrate metacognitive control in a polystyrene case study, where the system not only produces domain-level scientific outputs but continually monitors and optimizes its own behavior through tactical, strategic, and meta-strategic self-assessment. 

---
# EDU-CIRCUIT-HW: Evaluating Multimodal Large Language Models on Real-World University-Level STEM Student Handwritten Solutions 

**Authors**: Weiyu Sun, Liangliang Chen, Yongnuo Cai, Huiru Xie, Yi Zeng, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.00095)  

**Abstract**: Multimodal Large Language Models (MLLMs) hold significant promise for revolutionizing traditional education and reducing teachers' workload. However, accurately interpreting unconstrained STEM student handwritten solutions with intertwined mathematical formulas, diagrams, and textual reasoning poses a significant challenge due to the lack of authentic and domain-specific benchmarks. Additionally, current evaluation paradigms predominantly rely on the outcomes of downstream tasks (e.g., auto-grading), which often probe only a subset of the recognized content, thereby failing to capture the MLLMs' understanding of complex handwritten logic as a whole. To bridge this gap, we release EDU-CIRCUIT-HW, a dataset consisting of 1,300+ authentic student handwritten solutions from a university-level STEM course. Utilizing the expert-verified verbatim transcriptions and grading reports of student solutions, we simultaneously evaluate various MLLMs' upstream recognition fidelity and downstream auto-grading performance. Our evaluation uncovers an astonishing scale of latent failures within MLLM-recognized student handwritten content, highlighting the models' insufficient reliability for auto-grading and other understanding-oriented applications in high-stakes educational settings. In solution, we present a case study demonstrating that leveraging identified error patterns to preemptively detect and rectify recognition errors, with only minimal human intervention (approximately 4% of the total solutions), can significantly enhance the robustness of the deployed AI-enabled grading system on unseen student solutions. 

---
# ECCO: Evidence-Driven Causal Reasoning for Compiler Optimization 

**Authors**: Haolin Pan, Lianghong Huang, Jinyuan Dong, Mingjie Xing, Yanjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00087)  

**Abstract**: Compiler auto-tuning faces a dichotomy between traditional black-box search methods, which lack semantic guidance, and recent Large Language Model (LLM) approaches, which often suffer from superficial pattern matching and causal opacity. In this paper, we introduce ECCO, a framework that bridges interpretable reasoning with combinatorial search. We first propose a reverse engineering methodology to construct a Chain-of-Thought dataset, explicitly mapping static code features to verifiable performance evidence. This enables the model to learn the causal logic governing optimization decisions rather than merely imitating sequences. Leveraging this interpretable prior, we design a collaborative inference mechanism where the LLM functions as a strategist, defining optimization intents that dynamically guide the mutation operations of a genetic algorithm. Experimental results on seven datasets demonstrate that ECCO significantly outperforms the LLVM opt -O3 baseline, achieving an average 24.44% reduction in cycles. 

---
# Impact of LLMs news Sentiment Analysis on Stock Price Movement Prediction 

**Authors**: Walid Siala, Ahmed Khanfir, Mike Papadakis  

**Link**: [PDF](https://arxiv.org/pdf/2602.00086)  

**Abstract**: This paper addresses stock price movement prediction by leveraging LLM-based news sentiment analysis. Earlier works have largely focused on proposing and assessing sentiment analysis models and stock movement prediction methods, however, separately. Although promising results have been achieved, a clear and in-depth understanding of the benefit of the news sentiment to this task, as well as a comprehensive assessment of different architecture types in this context, is still lacking. Herein, we conduct an evaluation study that compares 3 different LLMs, namely, DeBERTa, RoBERTa and FinBERT, for sentiment-driven stock prediction. Our results suggest that DeBERTa outperforms the other two models with an accuracy of 75% and that an ensemble model that combines the three models can increase the accuracy to about 80%. Also, we see that sentiment news features can benefit (slightly) some stock market prediction models, i.e., LSTM-, PatchTST- and tPatchGNN-based classifiers and PatchTST- and TimesNet-based regression tasks models. 

---
# Enhancing few-shot time series forecasting with LLM-guided diffusion 

**Authors**: Haonan Shi, Dehua Shuai, Liming Wang, Xiyang Liu, Long Tian  

**Link**: [PDF](https://arxiv.org/pdf/2602.00040)  

**Abstract**: Time series forecasting in specialized domains is often constrained by limited data availability, where conventional models typically require large-scale datasets to effectively capture underlying temporal dynamics. To tackle this few-shot challenge, we propose LTSM-DIFF (Large-scale Temporal Sequential Memory with Diffusion), a novel learning framework that integrates the expressive power of large language models with the generative capability of diffusion models. Specifically, the LTSM module is fine-tuned and employed as a temporal memory mechanism, extracting rich sequential representations even under data-scarce conditions. These representations are then utilized as conditional guidance for a joint probability diffusion process, enabling refined modeling of complex temporal patterns. This design allows knowledge transfer from the language domain to time series tasks, substantially enhancing both generalization and robustness. Extensive experiments across diverse benchmarks demonstrate that LTSM-DIFF consistently achieves state-of-the-art performance in data-rich scenarios, while also delivering significant improvements in few-shot forecasting. Our work establishes a new paradigm for time series analysis under data scarcity. 

---
# Student Perceptions of Large Language Models Use in Self-Reflection and Design Critique in Architecture Studio 

**Authors**: Juan David Salazar Rodriguez, Sam Conrad Joyce, Nachamma Sockalingam, Khoo Eng Tat, Julfendi  

**Link**: [PDF](https://arxiv.org/pdf/2602.00041)  

**Abstract**: This study investigates the integration of Large Language Models (LLMs) into the feedback mechanisms of the architectural design studio, shifting the focus from generative production to reflective pedagogy. Employing a mixed-methods approach with architecture students at the Singapore Uni-versity of Technology and Design, the research analyzes student percep-tions across three distinct feedback domains: self-reflection, peer critique, and professor-led reviews. The findings reveal that students engage with LLMs not as authoritative instructors, but as collaborative "cognitive mir-rors" that scaffold critical thinking. In self-directed learning, LLMs help structure thoughts and overcome the "blank page" problem, though they are limited by a lack of contextual nuance. In peer critiques, the technology serves as a neutral mediator, mitigating social anxiety and the "fear of of-fending". Furthermore, in high-stakes professor-led juries, students utilize LLMs primarily as post-critique synthesis engines to manage cognitive overload and translate abstract academic discourse into actionable design iterations. 

---
# LOGOS-CA: A Cellular Automaton Using Natural Language as State and Rule 

**Authors**: Keishu Utimula  

**Link**: [PDF](https://arxiv.org/pdf/2602.00036)  

**Abstract**: Large Language Models (LLMs), trained solely on massive text data, have achieved high performance on the Winograd Schema Challenge (WSC), a benchmark proposed to measure commonsense knowledge and reasoning abilities about the real world. This suggests that the language produced by humanity describes a significant portion of the world with considerable nuance. In this study, we attempt to harness the high expressive power of language within cellular automata. Specifically, we express cell states and rules in natural language and delegate their updates to an LLM. Through this approach, cellular automata can transcend the constraints of merely numerical states and fixed rules, providing us with a richer platform for simulation. Here, we propose LOGOS-CA (Language Oriented Grid Of Statements - Cellular Automaton) as a natural framework to achieve this and examine its capabilities. We confirmed that LOGOS-CA successfully performs simple forest fire simulations and also serves as an intriguing subject for investigation from an Artificial Life (ALife) perspective. In this paper, we report the results of these experiments and discuss directions for future research using LOGOS-CA. 

---
# LSSF: Safety Alignment for Large Language Models through Low-Rank Safety Subspace Fusion 

**Authors**: Guanghao Zhou, Panjia Qiu, Cen Chen, Hongyu Li, Mingyuan Chu, Xin Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.00038)  

**Abstract**: The safety mechanisms of large language models (LLMs) exhibit notable fragility, as even fine-tuning on datasets without harmful content may still undermine their safety capabilities. Meanwhile, existing safety alignment methods predominantly rely on the fine-tuning process, which inadvertently leads to the increased complexity and computational resources required. To address these issues, we introduce LSSF, a novel safety re-alignment framework with \underline{L}ow-Rank \underline{S}afety \underline{S}ubspace \underline{F}usion. Our proposed method exploits the low-rank characteristics of safety information in LLMs by constructing a low-rank projection matrix to extract the principal components of safety vectors. Notably, this projection matrix represents the low-rank safety subspace of the LLMs, which we have observed to remain stable during fine-tuning process and is isolated from the model's general capabilities. These principal components are used to effectively restore safety alignment when combined with fine-tuned LLMs through linear arithmetic. Additionally, to account for the varying encoding densities of safety information across different layers of LLMs, we propose a novel metric called safety singular value entropy. This metric quantifies the encoding density and allows for the dynamic computation of the safety-critical rank for each safety vector. Extensive experiments demonstrate that our proposed post-hoc alignment method can effectively restore the safety alignment of fine-tuned models with minimal impact on their performance in downstream tasks. 

---
# When LLMs Imagine People: A Human-Centered Persona Brainstorm Audit for Bias and Fairness in Creative Applications 

**Authors**: Hongliu Cao, Eoin Thomas, Rodrigo Acuna Agost  

**Link**: [PDF](https://arxiv.org/pdf/2602.00044)  

**Abstract**: Biased outputs from Large Language Models (LLMs) can reinforce stereotypes and perpetuate inequities in real-world applications, making fairness auditing essential. We introduce the Persona Brainstorm Audit (PBA), a scalable and transparent auditing method for detecting bias through open-ended persona generation. Unlike existing methods that rely on fixed identity categories and static benchmarks, PBA uncovers biases across multiple social dimensions while supporting longitudinal tracking and mitigating data leakage risks. Applying PBA to 12 state-of-the-art LLMs, we compare bias severity across models, dimensions, and versions, uncover distinct patterns and lineage-specific variability, and trace how biases attenuate, persist, or resurface across successive generations. Robustness analyses show PBA remains stable under varying sample sizes, role-playing prompts, and debiasing prompts, establishing its reliability for fairness auditing in LLMs. 

---
# Synthetic Student Responses: LLM-Extracted Features for IRT Difficulty Parameter Estimation 

**Authors**: Matias Hoyl  

**Link**: [PDF](https://arxiv.org/pdf/2602.00034)  

**Abstract**: Educational assessment relies heavily on knowing question difficulty, traditionally determined through resource-intensive pre-testing with students. This creates significant barriers for both classroom teachers and assessment developers. We investigate whether Item Response Theory (IRT) difficulty parameters can be accurately estimated without student testing by modeling the response process and explore the relative contribution of different feature types to prediction accuracy. Our approach combines traditional linguistic features with pedagogical insights extracted using Large Language Models (LLMs), including solution step count, cognitive complexity, and potential misconceptions. We implement a two-stage process: first training a neural network to predict how students would respond to questions, then deriving difficulty parameters from these simulated response patterns. Using a dataset of over 250,000 student responses to mathematics questions, our model achieves a Pearson correlation of approximately 0.78 between predicted and actual difficulty parameters on completely unseen questions. 

---
# Beyond Static Question Banks: Dynamic Knowledge Expansion via LLM-Automated Graph Construction and Adaptive Generation 

**Authors**: Yingquan Wang, Tianyu Wei, Qinsi Li, Li Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2602.00020)  

**Abstract**: Personalized education systems increasingly rely on structured knowledge representations to support adaptive learning and question generation. However, existing approaches face two fundamental limitations. First, constructing and maintaining knowledge graphs for educational content largely depends on manual curation, resulting in high cost and poor scalability. Second, most personalized education systems lack effective support for state-aware and systematic reasoning over learners' knowledge, and therefore rely on static question banks with limited adaptability. To address these challenges, this paper proposes a Generative GraphRAG framework for automated knowledge modeling and personalized exercise generation. It consists of two core modules. The first module, Automated Hierarchical Knowledge Graph Constructor (Auto-HKG), leverages LLMs to automatically construct hierarchical knowledge graphs that capture structured concepts and their semantic relations from educational resources. The second module, Cognitive GraphRAG (CG-RAG), performs graph-based reasoning over a learner mastery graph and combines it with retrieval-augmented generation to produce personalized exercises that adapt to individual learning states. The proposed framework has been deployed in real-world educational scenarios, where it receives favorable user feedback, suggesting its potential to support practical personalized education systems. 

---
# AutoBinder Agent: An MCP-Based Agent for End-to-End Protein Binder Design 

**Authors**: Fukang Ge, Jiarui Zhu, Linjie Zhang, Haowen Xiao, Xiangcheng Bao, Fangnan Xie, Danyang Chen, Yanrui Lu, Yuting Wang, Ziqian Guan, Lin Gu, Jinhao Bi, Yingying Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00019)  

**Abstract**: Modern AI technologies for drug discovery are distributed across heterogeneous platforms-including web applications, desktop environments, and code libraries-leading to fragmented workflows, inconsistent interfaces, and high integration overhead. We present an agentic end-to-end drug design framework that leverages a Large Language Model (LLM) in conjunction with the Model Context Protocol (MCP) to dynamically coordinate access to biochemical databases, modular toolchains, and task-specific AI models. The system integrates four state-of-the-art components: MaSIF (MaSIF-site and MaSIF-seed-search) for geometric deep learning-based identification of protein-protein interaction (PPI) sites, Rosetta for grafting protein fragments onto protein backbones to form mini proteins, ProteinMPNN for amino acid sequences redesign, and AlphaFold3 for near-experimental accuracy in complex structure prediction. Starting from a target structure, the framework supports de novo binder generation via surface analysis, scaffold grafting and pose construction, sequence optimization, and structure prediction. Additionally, by replacing rigid, script-based workflows with a protocol-driven, LLM-coordinated architecture, the framework improves reproducibility, reduces manual overhead, and ensures extensibility, portability, and auditability across the entire drug design process. 

---
# Design and Empirical Study of a Large Language Model-Based Multi-Agent Investment System for Chinese Public REITs 

**Authors**: Zheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.00082)  

**Abstract**: This study addresses the low-volatility Chinese Public Real Estate Investment Trusts (REITs) market, proposing a large language model (LLM)-driven trading framework based on multi-agent collaboration. The system constructs four types of analytical agents-announcement, event, price momentum, and market-each conducting analysis from different dimensions; then the prediction agent integrates these multi-source signals to output directional probability distributions across multiple time horizons, then the decision agent generates discrete position adjustment signals based on the prediction results and risk control constraints, thereby forming a closed loop of analysis-prediction-decision-execution. This study further compares two prediction model pathways: for the prediction agent, directly calling the general-purpose large model DeepSeek-R1 versus using a specialized small model Qwen3-8B fine-tuned via supervised fine-tuning and reinforcement learning alignment. In the backtest from October 2024 to October 2025, both agent-based strategies significantly outperformed the buy-and-hold benchmark in terms of cumulative return, Sharpe ratio, and maximum drawdown. The results indicate that the multi-agent framework can effectively enhance the risk-adjusted return of REITs trading, and the fine-tuned small model performs close to or even better than the general-purpose large model in some scenarios. 

---
