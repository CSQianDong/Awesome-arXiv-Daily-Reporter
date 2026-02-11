# SCORE: Specificity, Context Utilization, Robustness, and Relevance for Reference-Free LLM Evaluation 

**Authors**: Homaira Huda Shomee, Rochana Chaturvedi, Yangxinyu Xie, Tanwi Mallick  

**Link**: [PDF](https://arxiv.org/pdf/2602.10017)  

**Abstract**: Large language models (LLMs) are increasingly used to support question answering and decision-making in high-stakes, domain-specific settings such as natural hazard response and infrastructure planning, where effective answers must convey fine-grained, decision-critical details. However, existing evaluation frameworks for retrieval-augmented generation (RAG) and open-ended question answering primarily rely on surface-level similarity, factual consistency, or semantic relevance, and often fail to assess whether responses provide the specific information required for domain-sensitive decisions. To address this gap, we propose a multi-dimensional, reference-free evaluation framework that assesses LLM outputs along four complementary dimensions: specificity, robustness to paraphrasing and semantic perturbations, answer relevance, and context utilization. We introduce a curated dataset of 1,412 domain-specific question-answer pairs spanning 40 professional roles and seven natural hazard types to support systematic evaluation. We further conduct human evaluation to assess inter-annotator agreement and alignment between model outputs and human judgments, which highlights the inherent subjectivity of open-ended, domain-specific evaluation. Our results show that no single metric sufficiently captures answer quality in isolation and demonstrate the need for structured, multi-metric evaluation frameworks when deploying LLMs in high-stakes applications. 

---
# Quantum-Audit: Evaluating the Reasoning Limits of LLMs on Quantum Computing 

**Authors**: Mohamed Afane, Kayla Laufer, Wenqi Wei, Ying Mao, Junaid Farooq, Ying Wang, Juntao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.10092)  

**Abstract**: Language models have become practical tools for quantum computing education and research, from summarizing technical papers to explaining theoretical concepts and answering questions about recent developments in the field. While existing benchmarks evaluate quantum code generation and circuit design, their understanding of quantum computing concepts has not been systematically measured. Quantum-Audit addresses this gap with 2,700 questions covering core quantum computing topics. We evaluate 26 models from leading organizations. Our benchmark comprises 1,000 expert-written questions, 1,000 questions extracted from research papers using LLMs and validated by experts, plus an additional 700 questions including 350 open-ended questions and 350 questions with false premises to test whether models can correct erroneous assumptions. Human participants scored between 23% and 86%, with experts averaging 74%. Top-performing models exceeded the expert average, with Claude Opus 4.5 reaching 84% accuracy, though top models showed an average 12-point accuracy drop on expert-written questions compared to LLM-generated ones. Performance declined further on advanced topics, dropping to 73% on security questions. Additionally, models frequently accepted and reinforced false premises embedded in questions instead of identifying them, with accuracy below 66% on these critical reasoning tasks. 

---
# Decoupled Reasoning with Implicit Fact Tokens (DRIFT): A Dual-Model Framework for Efficient Long-Context Inference 

**Authors**: Wenxuan Xie, Yujia Wang, Xin Tan, Chaochao Lu, Xia Hu, Xuhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10021)  

**Abstract**: The integration of extensive, dynamic knowledge into Large Language Models (LLMs) remains a significant challenge due to the inherent entanglement of factual data and reasoning patterns. Existing solutions, ranging from non-parametric Retrieval-Augmented Generation (RAG) to parametric knowledge editing, are often constrained in practice by finite context windows, retriever noise, or the risk of catastrophic forgetting. In this paper, we propose DRIFT, a novel dual-model architecture designed to explicitly decouple knowledge extraction from the reasoning process. Unlike static prompt compression, DRIFT employs a lightweight knowledge model to dynamically compress document chunks into implicit fact tokens conditioned on the query. These dense representations are projected into the reasoning model's embedding space, replacing raw, redundant text while maintaining inference accuracy. Extensive experiments show that DRIFT significantly improves performance on long-context tasks, outperforming strong baselines among comparably sized models. Our approach provides a scalable and efficient paradigm for extending the effective context window and reasoning capabilities of LLMs. Our code is available at this https URL. 

---
# LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations 

**Authors**: William Lugoloobi, Thomas Foster, William Bankes, Chris Russell  

**Link**: [PDF](https://arxiv.org/pdf/2602.09924)  

**Abstract**: Running LLMs with extended reasoning on every problem is expensive, but determining which inputs actually require additional compute remains challenging. We investigate whether their own likelihood of success is recoverable from their internal representations before generation, and if this signal can guide more efficient inference. We train linear probes on pre-generation activations to predict policy-specific success on math and coding tasks, substantially outperforming surface features such as question length and TF-IDF. Using E2H-AMC, which provides both human and model performance on identical problems, we show that models encode a model-specific notion of difficulty that is distinct from human difficulty, and that this distinction increases with extended reasoning. Leveraging these probes, we demonstrate that routing queries across a pool of models can exceed the best-performing model whilst reducing inference cost by up to 70\% on MATH, showing that internal representations enable practical efficiency gains even when they diverge from human intuitions about difficulty. Our code is available at: this https URL 

---
# The Devil Behind Moltbook: Anthropic Safety is Always Vanishing in Self-Evolving AI Societies 

**Authors**: Chenxu Wang, Chaozhuo Li, Songyang Liu, Zejian Chen, Jinyu Hou, Ji Qi, Rui Li, Litian Zhang, Qiwei Ye, Zheng Liu, Xu Chen, Xi Zhang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.09877)  

**Abstract**: The emergence of multi-agent systems built from large language models (LLMs) offers a promising paradigm for scalable collective intelligence and self-evolution. Ideally, such systems would achieve continuous self-improvement in a fully closed loop while maintaining robust safety alignment--a combination we term the self-evolution trilemma. However, we demonstrate both theoretically and empirically that an agent society satisfying continuous self-evolution, complete isolation, and safety invariance is impossible. Drawing on an information-theoretic framework, we formalize safety as the divergence degree from anthropic value distributions. We theoretically demonstrate that isolated self-evolution induces statistical blind spots, leading to the irreversible degradation of the system's safety alignment. Empirical and qualitative results from an open-ended agent community (Moltbook) and two closed self-evolving systems reveal phenomena that align with our theoretical prediction of inevitable safety erosion. We further propose several solution directions to alleviate the identified safety concern. Our work establishes a fundamental limit on the self-evolving AI societies and shifts the discourse from symptom-driven safety patches to a principled understanding of intrinsic dynamical risks, highlighting the need for external oversight or novel safety-preserving mechanisms. 

---
# SinFoS: A Parallel Dataset for Translating Sinhala Figures of Speech 

**Authors**: Johan Sofalas, Dilushri Pavithra, Nevidu Jayatilleke, Ruvan Weerasinghe  

**Link**: [PDF](https://arxiv.org/pdf/2602.09866)  

**Abstract**: Figures of Speech (FoS) consist of multi-word phrases that are deeply intertwined with culture. While Neural Machine Translation (NMT) performs relatively well with the figurative expressions of high-resource languages, it often faces challenges when dealing with low-resource languages like Sinhala due to limited available data. To address this limitation, we introduce a corpus of 2,344 Sinhala figures of speech with cultural and cross-lingual annotations. We examine this dataset to classify the cultural origins of the figures of speech and to identify their cross-lingual equivalents. Additionally, we have developed a binary classifier to differentiate between two types of FOS in the dataset, achieving an accuracy rate of approximately 92%. We also evaluate the performance of existing LLMs on this dataset. Our findings reveal significant shortcomings in the current capabilities of LLMs, as these models often struggle to accurately convey idiomatic meanings. By making this dataset publicly available, we offer a crucial benchmark for future research in low-resource NLP and culturally aware machine translation. 

---
# Steer2Edit: From Activation Steering to Component-Level Editing 

**Authors**: Chung-En Sun, Ge Yan, Zimo Wang, Tsui-Wei Weng  

**Link**: [PDF](https://arxiv.org/pdf/2602.09870)  

**Abstract**: Steering methods influence Large Language Model behavior by identifying semantic directions in hidden representations, but are typically realized through inference-time activation interventions that apply a fixed, global modification to the model's internal states. While effective, such interventions often induce unfavorable attribute-utility trade-offs under strong control, as they ignore the fact that many behaviors are governed by a small and heterogeneous subset of model components. We propose Steer2Edit, a theoretically grounded, training-free framework that transforms steering vectors from inference-time control signals into diagnostic signals for component-level rank-1 weight editing. Instead of uniformly injecting a steering direction during generation, Steer2Edit selectively redistributes behavioral influence across individual attention heads and MLP neurons, yielding interpretable edits that preserve the standard forward pass and remain compatible with optimized parallel inference. Across safety alignment, hallucination mitigation, and reasoning efficiency, Steer2Edit consistently achieves more favorable attribute-utility trade-offs: at matched downstream performance, it improves safety by up to 17.2%, increases truthfulness by 9.8%, and reduces reasoning length by 12.2% on average. Overall, Steer2Edit provides a principled bridge between representation steering and weight editing by translating steering signals into interpretable, training-free parameter updates. 

---
# AnalyticsGPT: An LLM Workflow for Scientometric Question Answering 

**Authors**: Khang Ly, Georgios Cheirmpos, Adrian Raudaschl, Christopher James, Seyed Amin Tabatabaei  

**Link**: [PDF](https://arxiv.org/pdf/2602.09817)  

**Abstract**: This paper introduces AnalyticsGPT, an intuitive and efficient large language model (LLM)-powered workflow for scientometric question answering. This underrepresented downstream task addresses the subcategory of meta-scientific questions concerning the "science of science." When compared to traditional scientific question answering based on papers, the task poses unique challenges in the planning phase. Namely, the need for named-entity recognition of academic entities within questions and multi-faceted data retrieval involving scientometric indices, e.g. impact factors. Beyond their exceptional capacity for treating traditional natural language processing tasks, LLMs have shown great potential in more complex applications, such as task decomposition and planning and reasoning. In this paper, we explore the application of LLMs to scientometric question answering, and describe an end-to-end system implementing a sequential workflow with retrieval-augmented generation and agentic concepts. We also address the secondary task of effectively synthesizing the data into presentable and well-structured high-level analyses. As a database for retrieval-augmented generation, we leverage a proprietary research performance assessment platform. For evaluation, we consult experienced subject matter experts and leverage LLMs-as-judges. In doing so, we provide valuable insights on the efficacy of LLMs towards a niche downstream task. Our (skeleton) code and prompts are available at: this https URL. 

---
# ATTNPO: Attention-Guided Process Supervision for Efficient Reasoning 

**Authors**: Shuaiyi Nie, Siyu Ding, Wenyuan Zhang, Linhao Yu, Tianmeng Yang, Yao Chen, Tingwen Liu, Weichong Yin, Yu Sun, Hua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.09953)  

**Abstract**: Large reasoning models trained with reinforcement learning and verifiable rewards (RLVR) achieve strong performance on complex reasoning tasks, yet often overthink, generating redundant reasoning without performance gains. Existing trajectory-level length penalties often fail to effectively shorten reasoning length and degrade accuracy, as they uniformly treat all reasoning steps and lack fine-grained signals to distinguish redundancy from necessity. Meanwhile, process-supervised methods are typically resource-intensive and suffer from inaccurate credit assignment. To address these issues, we propose ATTNPO, a low-overhead process-supervised RL framework that leverages the model's intrinsic attention signals for step-level credit assignment. We first identify a set of special attention heads that naturally focus on essential steps while suppressing redundant ones. By leveraging the attention scores of these heads, We then employ two sub-strategies to mitigate overthinking by discouraging redundant steps while preserving accuracy by reducing penalties on essential steps. Experimental results show that ATTNPO substantially reduces reasoning length while significantly improving performance across 9 benchmarks. 

---
# Decomposing Reasoning Efficiency in Large Language Models 

**Authors**: Daniel Kaiser, Arnoldo Frigessi, Ali Ramezani-Kebrya, Benjamin Ricaud  

**Link**: [PDF](https://arxiv.org/pdf/2602.09805)  

**Abstract**: Large language models trained for reasoning trade off inference tokens against accuracy, yet standard evaluations report only final accuracy, obscuring where tokens are spent or wasted. We introduce a trace-optional framework that decomposes token efficiency into interpretable factors: completion under a fixed token budget (avoiding truncation), conditional correctness given completion, and verbosity (token usage). When benchmark metadata provides per-instance workload proxies, we further factor verbosity into two components: mean verbalization overhead (tokens per work unit) and a coupling coefficient capturing how overhead scales with task workload. When reasoning traces are available, we add deterministic trace-quality measures (grounding, repetition, prompt copying) to separate degenerate looping from verbose-but-engaged reasoning, avoiding human labeling and LLM judges. Evaluating 25 models on CogniLoad, we find that accuracy and token-efficiency rankings diverge (Spearman $\rho=0.63$), efficiency gaps are often driven by conditional correctness, and verbalization overhead varies by about 9 times (only weakly related to model scale). Our decomposition reveals distinct bottleneck profiles that suggest different efficiency interventions. 

---
# Text summarization via global structure awareness 

**Authors**: Jiaquan Zhang, Chaoning Zhang, Shuxu Chen, Yibei Liu, Chenghao Li, Qigan Sun, Shuai Yuan, Fachrina Dewi Puspitasari, Dongshen Han, Guoqing Wang, Sung-Ho Bae, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.09821)  

**Abstract**: Text summarization is a fundamental task in natural language processing (NLP), and the information explosion has made long-document processing increasingly demanding, making summarization essential. Existing research mainly focuses on model improvements and sentence-level pruning, but often overlooks global structure, leading to disrupted coherence and weakened downstream performance. Some studies employ large language models (LLMs), which achieve higher accuracy but incur substantial resource and time costs. To address these issues, we introduce GloSA-sum, the first summarization approach that achieves global structure awareness via topological data analysis (TDA). GloSA-sum summarizes text efficiently while preserving semantic cores and logical dependencies. Specifically, we construct a semantic-weighted graph from sentence embeddings, where persistent homology identifies core semantics and logical structures, preserved in a ``protection pool'' as the backbone for summarization. We design a topology-guided iterative strategy, where lightweight proxy metrics approximate sentence importance to avoid repeated high-cost computations, thus preserving structural integrity while improving efficiency. To further enhance long-text processing, we propose a hierarchical strategy that integrates segment-level and global summarization. Experiments on multiple datasets demonstrate that GloSA-sum reduces redundancy while preserving semantic and logical integrity, striking a balance between accuracy and efficiency, and further benefits LLM downstream tasks by shortening contexts while retaining essential reasoning chains. 

---
# LLM Reasoning Predicts When Models Are Right: Evidence from Coding Classroom Discourse 

**Authors**: Bakhtawar Ahtisham, Kirk Vanacore, Zhuqian Zhou, Jinsook Lee, Rene F. Kizilcec  

**Link**: [PDF](https://arxiv.org/pdf/2602.09832)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed to automatically label and analyze educational dialogue at scale, yet current pipelines lack reliable ways to detect when models are wrong. We investigate whether reasoning generated by LLMs can be used to predict the correctness of a model's own predictions. We analyze 30,300 teacher utterances from classroom dialogue, each labeled by multiple state-of-the-art LLMs with an instructional move construct and an accompanying reasoning. Using human-verified ground-truth labels, we frame the task as predicting whether a model's assigned label for a given utterance is correct. We encode LLM reasoning using Term Frequency-Inverse Document Frequency (TF-IDF) and evaluate five supervised classifiers. A Random Forest classifier achieves an F1 score of 0.83 (Recall = 0.854), successfully identifying most incorrect predictions and outperforming baselines. Training specialist detectors for specific instructional move constructs further improves performance on difficult constructs, indicating that error detection benefits from construct-specific linguistic cues. Using the Linguistic Inquiry and Word Count (LIWC) framework, we examine four linguistic markers of correctness: Causation, Differentiation, Tentativeness, and Insight. Correct predictions exhibit grounded causal language (e.g., because, therefore), while incorrect reasoning is substantially more likely to rely on epistemic hedging (e.g., might, could) and performative metacognition (e.g., think, realize). Syntactic complexity does not distinguish correct from incorrect reasoning, and longer reasoning is not more reliable. These findings demonstrate that reasoning-based error detection offers a practical and scalable approach to quality control in automated educational dialogue analysis. 

---
# Unsupervised Layer-Wise Dynamic Test Time Adaptation for LLMs 

**Authors**: Longhuan Xu, Cunjian Chen, Feng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.09719)  

**Abstract**: Test-time adaptation (TTA) for large language models (LLMs) updates model parameters at inference time using signals available at deployment. This paper focuses on a common yet under-explored regime: unsupervised, sample-specific TTA, where the model adapts independently for each prompt using only the prompt itself, without gold answers or external supervision. Although appealing, naive unsupervised TTA with a fixed, handcrafted learning rate can be unstable: updates may overfit to prompt-specific statistics, drift from the desired answer distribution, and ultimately degrade generation quality. This failure mode is not surprising, as in this case TTA must adapt to a single prompt within only a few gradient steps, unlike standard training that averages updates over large datasets and long optimization horizons. Therefore, we propose layer-wise dynamic test-time adaptation, a framework which explicitly modulates TTA strength as a function of prompt representation, LLM structure and adaptation step. In our setting, TTA updates only LoRA parameters, and a lightweight hypernetwork predicts per-layer, per-step learning-rate multipliers, enabling fine-grained control. Experiments across various datasets and LLMs consistently show that our method substantially strengthens TTA by learning effective scaling patterns over adaptation steps and transformer layer projections, improving stability while delivering better performance. 

---
# MILE-RefHumEval: A Reference-Free, Multi-Independent LLM Framework for Human-Aligned Evaluation 

**Authors**: Nalin Srun, Parisa Rastin, Guénaël Cabanes, Lydia Boudjeloud Assala  

**Link**: [PDF](https://arxiv.org/pdf/2602.09624)  

**Abstract**: We introduce MILE-RefHumEval, a reference-free framework for evaluating Large Language Models (LLMs) without ground-truth annotations or evaluator coordination. It leverages an ensemble of independently prompted evaluators guided by a human-aligned schema, supporting both discrete and continuous scoring judgement. With task-specific prompts from best candidate selection, summarization and image captioning to dialogue, MILE-RefHumEval provides flexible, interpretable, and scalable assessments. Experiments show it aligns closely with human judgments, outperforms prior methods, and reduces computational overhead, offering an efficient, robust, and human-aligned solution for real-world LLM evaluation. 

---
# TraceMem: Weaving Narrative Memory Schemata from User Conversational Traces 

**Authors**: Yiming Shu, Pei Liu, Tiange Zhang, Ruiyang Gao, Jun Ma, Chen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.09712)  

**Abstract**: Sustaining long-term interactions remains a bottleneck for Large Language Models (LLMs), as their limited context windows struggle to manage dialogue histories that extend over time. Existing memory systems often treat interactions as disjointed snippets, failing to capture the underlying narrative coherence of the dialogue stream. We propose TraceMem, a cognitively-inspired framework that weaves structured, narrative memory schemata from user conversational traces through a three-stage pipeline: (1) Short-term Memory Processing, which employs a deductive topic segmentation approach to demarcate episode boundaries and extract semantic representation; (2) Synaptic Memory Consolidation, a process that summarizes episodes into episodic memories before distilling them alongside semantics into user-specific traces; and (3) Systems Memory Consolidation, which utilizes two-stage hierarchical clustering to organize these traces into coherent, time-evolving narrative threads under unifying themes. These threads are encapsulated into structured user memory cards, forming narrative memory schemata. For memory utilization, we provide an agentic search mechanism to enhance reasoning process. Evaluation on the LoCoMo benchmark shows that TraceMem achieves state-of-the-art performance with a brain-inspired architecture. Analysis shows that by constructing coherent narratives, it surpasses baselines in multi-hop and temporal reasoning, underscoring its essential role in deep narrative comprehension. Additionally, we provide an open discussion on memory systems, offering our perspectives and future outlook on the field. Our code implementation is available at: this https URL 

---
# Learning from the Irrecoverable: Error-Localized Policy Optimization for Tool-Integrated LLM Reasoning 

**Authors**: Qiao Liang, Yuke Zhu, Chao Ge, Lei Yang, Ying Shen, Bo Zheng, Sheng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.09598)  

**Abstract**: Tool-integrated reasoning (TIR) enables LLM agents to solve tasks through planning, tool use, and iterative revision, but outcome-only reinforcement learning in this setting suffers from sparse, delayed rewards and weak step-level credit assignment. In long-horizon TIR trajectories, an early irrecoverable mistake can determine success or failure, making it crucial to localize the first irrecoverable step and leverage it for fine-grained credit assignment. We propose Error-Localized Policy Optimization (ELPO), which localizes the first irrecoverable step via binary-search rollout trees under a fixed rollout budget, converts the resulting tree into stable learning signals through hierarchical advantage attribution, and applies error-localized adaptive clipping to strengthen corrective updates on the critical step and its suffix. Across TIR benchmarks in math, science QA, and code execution, ELPO consistently outperforms strong Agentic RL baselines under comparable sampling budgets, with additional gains in Pass@K and Major@K scaling, rollout ranking quality, and tool-call efficiency. Our code will be publicly released soon. 

---
# Maastricht University at AMIYA: Adapting LLMs for Dialectal Arabic using Fine-tuning and MBR Decoding 

**Authors**: Abdulhai Alali, Abderrahmane Issam  

**Link**: [PDF](https://arxiv.org/pdf/2602.09703)  

**Abstract**: Large Language Models (LLMs) are becoming increasingly multilingual, supporting hundreds of languages, especially high resource ones. Unfortunately, Dialect variations are still underrepresented due to limited data and linguistic variation. In this work, we adapt a pre-trained LLM to improve dialectal performance. Specifically, we use Low Rank Adaptation (LoRA) fine-tuning on monolingual and English Dialect parallel data, adapter merging and dialect-aware MBR decoding to improve dialectal fidelity generation and translation. Experiments on Syrian, Moroccan, and Saudi Arabic show that merging and MBR improve dialectal fidelity while preserving semantic accuracy. This combination provides a compact and effective framework for robust dialectal Arabic generation. 

---
# AlignTune: Modular Toolkit for Post-Training Alignment of Large Language Models 

**Authors**: R E Zera Marveen Lyngkhoi, Chirag Chawla, Pratinav Seth, Utsav Avaiya, Soham Bhattacharjee, Mykola Khandoga, Rui Yuan, Vinay Kumar Sankarapu  

**Link**: [PDF](https://arxiv.org/pdf/2602.09621)  

**Abstract**: Post-training alignment is central to deploying large language models (LLMs), yet practical workflows remain split across backend-specific tools and ad-hoc glue code, making experiments hard to reproduce. We identify backend interference, reward fragmentation, and irreproducible pipelines as key obstacles in alignment research. We introduce AlignTune, a modular toolkit exposing a unified interface for supervised fine-tuning (SFT) and RLHF-style optimization with interchangeable TRL and Unsloth backends. AlignTune standardizes configuration, provides an extensible reward layer (rule-based and learned), and integrates evaluation over standard benchmarks and custom tasks. By isolating backend-specific logic behind a single factory boundary, AlignTune enables controlled comparisons and reproducible alignment experiments. 

---
# Aligning Tree-Search Policies with Fixed Token Budgets in Test-Time Scaling of LLMs 

**Authors**: Sora Miyamoto, Daisuke Oba, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2602.09574)  

**Abstract**: Tree-search decoding is an effective form of test-time scaling for large language models (LLMs), but real-world deployment imposes a fixed per-query token budget that varies across settings. Existing tree-search policies are largely budget-agnostic, treating the budget as a termination condition, which can lead to late-stage over-branching or premature termination. We propose {Budget-Guided MCTS} (BG-MCTS), a tree-search decoding algorithm that aligns its search policy with the remaining token budget: it starts with broad exploration, then prioritizes refinement and answer completion as the budget depletes while reducing late-stage branching from shallow nodes. BG-MCTS consistently outperforms budget-agnostic tree-search baselines across different budgets on MATH500 and AIME24/25 with open-weight LLMs. 

---
# EcoGym: Evaluating LLMs for Long-Horizon Plan-and-Execute in Interactive Economies 

**Authors**: Xavier Hu, Jinxiang Xia, Shengze Xu, Kangqi Song, Yishuo Yuan, Guibin Zhang, Jincheng Ren, Boyu Feng, Li Lu, Tieyong Zeng, Jiaheng Liu, Minghao Liu, Yuchen Elenor Jiang, Wei Wang, He Zhu, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.09514)  

**Abstract**: Long-horizon planning is widely recognized as a core capability of autonomous LLM-based agents; however, current evaluation frameworks suffer from being largely episodic, domain-specific, or insufficiently grounded in persistent economic dynamics. We introduce EcoGym, a generalizable benchmark for continuous plan-and-execute decision making in interactive economies. EcoGym comprises three diverse environments: Vending, Freelance, and Operation, implemented in a unified decision-making process with standardized interfaces, and budgeted actions over an effectively unbounded horizon (1000+ steps if 365 day-loops for evaluation). The evaluation of EcoGym is based on business-relevant outcomes (e.g., net worth, income, and DAU), targeting long-term strategic coherence and robustness under partial observability and stochasticity. Experiments across eleven leading LLMs expose a systematic tension: no single model dominates across all three scenarios. Critically, we find that models exhibit significant suboptimality in either high-level strategies or efficient actions executions. EcoGym is released as an open, extensible testbed for transparent long-horizon agent evaluation and for studying controllability-utility trade-offs in realistic economic settings. 

---
# Listen to the Layers: Mitigating Hallucinations with Inter-Layer Disagreement 

**Authors**: Koduvayur Subbalakshmi, Sabbir Hossain Ujjal, Venkata Krishna Teja Mangichetty, Nastaran Jamalipour Soofi  

**Link**: [PDF](https://arxiv.org/pdf/2602.09486)  

**Abstract**: Pretrained Large Language Models (LLMs) are prone to generating fluent yet factually incorrect text-a phenomenon known as hallucinations, undermining their reliability and utility in downstream tasks. We hypothesize that a generated text span's factuality is correlated with its representational instability across the model's internal layers. Based on this, we propose the CoCoA (Confusion and Consistency Aware) decoder, a novel, training-free decoding algorithm that mitigates hallucinations at inference time by listening to these signals in the middle layers. We propose two metrics to quantify this instability in the middle layers, and use it to penalize outputs that exhibit high internal confusion, thereby steering the model towards more internally consistent and factually grounded outputs. We further propose a self-information gated variant, CoCoA-SIG, that dynamically modulates this penalty to selectively target high-surprise, unstable generations. Extensive experiments on diverse tasks, including question-answering, summarization and code generation demonstrate that CoCoA significantly improves factual correctness across multiple model families (e.g., Llama-3, Qwen-2.5, Mistral). By leveraging model-intrinsic signals, CoCoA offers an effective and broadly applicable method for enhancing the trustworthiness of LLMs at inference time, without requiring any model retraining. 

---
# Conceptual Cultural Index: A Metric for Cultural Specificity via Relative Generality 

**Authors**: Takumi Ohashi, Hitoshi Iyatomi  

**Link**: [PDF](https://arxiv.org/pdf/2602.09444)  

**Abstract**: Large language models (LLMs) are increasingly deployed in multicultural settings; however, systematic evaluation of cultural specificity at the sentence level remains underexplored. We propose the Conceptual Cultural Index (CCI), which estimates cultural specificity at the sentence level. CCI is defined as the difference between the generality estimate within the target culture and the average generality estimate across other cultures. This formulation enables users to operationally control the scope of culture via comparison settings and provides interpretability, since the score derives from the underlying generality estimates. We validate CCI on 400 sentences (200 culture-specific and 200 general), and the resulting score distribution exhibits the anticipated pattern: higher for culture-specific sentences and lower for general ones. For binary separability, CCI outperforms direct LLM scoring, yielding more than a 10-point improvement in AUC for models specialized to the target culture. Our code is available at this https URL . 

---
# Knowledge Integration Decay in Search-Augmented Reasoning of Large Language Models 

**Authors**: Sangwon Yu, Ik-hwan Kim, Donghun Kang, Bongkyu Hwang, Junhwa Choi, Suk-hoon Jung, Seungki Hong, Taehee Lee, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2602.09517)  

**Abstract**: Modern Large Language Models (LLMs) have demonstrated remarkable capabilities in complex tasks by employing search-augmented reasoning to incorporate external knowledge into long chains of thought. However, we identify a critical yet underexplored bottleneck in this paradigm, termed Knowledge Integration Decay (KID). Specifically, we observe that as the length of reasoning generated before search grows, models increasingly fail to integrate retrieved evidence into subsequent reasoning steps, limiting performance even when relevant information is available. To address this, we propose Self-Anchored Knowledge Encoding (SAKE), a training-free inference-time strategy designed to stabilize knowledge utilization. By anchoring retrieved knowledge at both the beginning and end of the reasoning process, SAKE prevents it from being overshadowed by prior context, thereby preserving its semantic integrity. Extensive experiments on multi-hop QA and complex reasoning benchmarks demonstrate that SAKE significantly mitigates KID and improves performance, offering a lightweight yet effective solution for knowledge integration in agentic LLMs. 

---
# Are Language Models Sensitive to Morally Irrelevant Distractors? 

**Authors**: Andrew Shaw, Christina Hahn, Catherine Rasgaitis, Yash Mishra, Alisa Liu, Natasha Jaques, Yulia Tsvetkov, Amy X. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.09416)  

**Abstract**: With the rapid development and uptake of large language models (LLMs) across high-stakes settings, it is increasingly important to ensure that LLMs behave in ways that align with human values. Existing moral benchmarks prompt LLMs with value statements, moral scenarios, or psychological questionnaires, with the implicit underlying assumption that LLMs report somewhat stable moral preferences. However, moral psychology research has shown that human moral judgements are sensitive to morally irrelevant situational factors, such as smelling cinnamon rolls or the level of ambient noise, thereby challenging moral theories that assume the stability of human moral judgements. Here, we draw inspiration from this "situationist" view of moral psychology to evaluate whether LLMs exhibit similar cognitive moral biases to humans. We curate a novel multimodal dataset of 60 "moral distractors" from existing psychological datasets of emotionally-valenced images and narratives which have no moral relevance to the situation presented. After injecting these distractors into existing moral benchmarks to measure their effects on LLM responses, we find that moral distractors can shift the moral judgements of LLMs by over 30% even in low-ambiguity scenarios, highlighting the need for more contextual moral evaluations and more nuanced cognitive moral modeling of LLMs. 

---
# Evaluating Social Bias in RAG Systems: When External Context Helps and Reasoning Hurts 

**Authors**: Shweta Parihar, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.09442)  

**Abstract**: Social biases inherent in large language models (LLMs) raise significant fairness concerns. Retrieval-Augmented Generation (RAG) architectures, which retrieve external knowledge sources to enhance the generative capabilities of LLMs, remain susceptible to the same bias-related challenges. This work focuses on evaluating and understanding the social bias implications of RAG. Through extensive experiments across various retrieval corpora, LLMs, and bias evaluation datasets, encompassing more than 13 different bias types, we surprisingly observe a reduction in bias in RAG. This suggests that the inclusion of external context can help counteract stereotype-driven predictions, potentially improving fairness by diversifying the contextual grounding of the model's outputs. To better understand this phenomenon, we then explore the model's reasoning process by integrating Chain-of-Thought (CoT) prompting into RAG while assessing the faithfulness of the model's CoT. Our experiments reveal that the model's bias inclinations shift between stereotype and anti-stereotype responses as more contextual information is incorporated from the retrieved documents. Interestingly, we find that while CoT enhances accuracy, contrary to the bias reduction observed with RAG, it increases overall bias across datasets, highlighting the need for bias-aware reasoning frameworks that can mitigate this trade-off. 

---
# BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation 

**Authors**: Peng Lai, Zhihao Ou, Yong Wang, Longyue Wang, Jian Yang, Yun Chen, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.09383)  

**Abstract**: LLM-as-a-Judge has been widely adopted across various research and practical applications, yet the robustness and reliability of its evaluation remain a critical issue. A core challenge it faces is bias, which has primarily been studied in terms of known biases and their impact on evaluation outcomes, while automated and systematic exploration of potential unknown biases is still lacking. Nevertheless, such exploration is crucial for enhancing the robustness and reliability of evaluations. To bridge this gap, we propose BiasScope, a LLM-driven framework for automatically and at scale discovering potential biases that may arise during model evaluation. BiasScope can uncover potential biases across different model families and scales, with its generality and effectiveness validated on the JudgeBench dataset. It overcomes the limitations of existing approaches, transforming bias discovery from a passive process relying on manual effort and predefined bias lists into an active and comprehensive automated exploration. Moreover, based on BiasScope, we propose JudgeBench-Pro, an extended version of JudgeBench and a more challenging benchmark for evaluating the robustness of LLM-as-a-judge. Strikingly, even powerful LLMs as evaluators show error rates above 50\% on JudgeBench-Pro, underscoring the urgent need to strengthen evaluation robustness and to mitigate potential biases further. 

---
# Breaking the Pre-Sampling Barrier: Activation-Informed Difficulty-Aware Self-Consistency 

**Authors**: Taewoong Yoon, Geunyeong Jeong, Geon Park, Sihyeong Yeom, Harksoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.09438)  

**Abstract**: Self-Consistency (SC) is an effective decoding strategy that improves the reasoning performance of Large Language Models (LLMs) by generating multiple chain-of-thought reasoning paths and selecting the final answer via majority voting. However, it suffers from substantial inference costs because it requires a large number of samples. To mitigate this issue, Difficulty-Adaptive Self-Consistency (DSC) was proposed to reduce unnecessary token usage for easy problems by adjusting the number of samples according to problem difficulty. However, DSC requires additional model calls and pre-sampling to estimate difficulty, and this process is repeated when applying to each dataset, leading to significant computational overhead. In this work, we propose Activation-Informed Difficulty-Aware Self-Consistency (ACTSC) to address these limitations. ACTSC leverages internal difficulty signals reflected in the feed-forward network neuron activations to construct a lightweight difficulty estimation probe, without any additional token generation or model calls. The probe dynamically adjusts the number of samples for SC and can be applied to new datasets without requiring pre-sampling for difficulty estimation. To validate its effectiveness, we conduct experiments on five benchmarks. Experimental results show that ACTSC effectively reduces inference costs while maintaining accuracy relative to existing methods. 

---
# Digital Linguistic Bias in Spanish: Evidence from Lexical Variation in LLMs 

**Authors**: Yoshifumi Kawasaki  

**Link**: [PDF](https://arxiv.org/pdf/2602.09346)  

**Abstract**: This study examines the extent to which Large Language Models (LLMs) capture geographic lexical variation in Spanish, a language that exhibits substantial regional variation. Treating LLMs as virtual informants, we probe their dialectal knowledge using two survey-style question formats: Yes-No questions and multiple-choice questions. To this end, we exploited a large-scale, expert-curated database of Spanish lexical variation. Our evaluation covers more than 900 lexical items across 21 Spanish-speaking countries and is conducted at both the country and dialectal area levels. Across both evaluation formats, the results reveal systematic differences in how LLMs represent Spanish language varieties. Lexical variation associated with Spain, Equatorial Guinea, Mexico & Central America, and the La Plata River is recognized more accurately by the models, while the Chilean variety proves particularly difficult for the models to distinguish. Importantly, differences in the volume of country-level digital resources do not account for these performance patterns, suggesting that factors beyond data quantity shape dialectal representation in LLMs. By providing a fine-grained, large-scale evaluation of geographic lexical variation, this work advances empirical understanding of dialectal knowledge in LLMs and contributes new evidence to discussions of Digital Linguistic Bias in Spanish. 

---
# Don't Shoot The Breeze: Topic Continuity Model Using Nonlinear Naive Bayes With Attention 

**Authors**: Shu-Ting Pi, Pradeep Bagavan, Yejia Li, Disha, Qun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.09312)  

**Abstract**: Utilizing Large Language Models (LLM) as chatbots in diverse business scenarios often presents the challenge of maintaining topic continuity. Abrupt shifts in topics can lead to poor user experiences and inefficient utilization of computational resources. In this paper, we present a topic continuity model aimed at assessing whether a response aligns with the initial conversation topic. Our model is built upon the expansion of the corresponding natural language understanding (NLU) model into quantifiable terms using a Naive Bayes approach. Subsequently, we have introduced an attention mechanism and logarithmic nonlinearity to enhance its capability to capture topic continuity. This approach allows us to convert the NLU model into an interpretable analytical formula. In contrast to many NLU models constrained by token limits, our proposed model can seamlessly handle conversations of any length with linear time complexity. Furthermore, the attention mechanism significantly improves the model's ability to identify topic continuity in complex conversations. According to our experiments, our model consistently outperforms traditional methods, particularly in handling lengthy and intricate conversations. This unique capability offers us an opportunity to ensure the responsible and interpretable use of LLMs. 

---
# CAPID: Context-Aware PII Detection for Question-Answering Systems 

**Authors**: Mariia Ponomarenko, Sepideh Abedini, Masoumeh Shafieinejad, D.B.Emerson, Shubhankar Mohapatra, Xi He  

**Link**: [PDF](https://arxiv.org/pdf/2602.10074)  

**Abstract**: Detecting personally identifiable information (PII) in user queries is critical for ensuring privacy in question-answering systems. Current approaches mainly redact all PII, disregarding the fact that some of them may be contextually relevant to the user's question, resulting in a degradation of response quality. Large language models (LLMs) might be able to help determine which PII are relevant, but due to their closed source nature and lack of privacy guarantees, they are unsuitable for sensitive data processing. To achieve privacy-preserving PII detection, we propose CAPID, a practical approach that fine-tunes a locally owned small language model (SLM) that filters sensitive information before it is passed to LLMs for QA. However, existing datasets do not capture the context-dependent relevance of PII needed to train such a model effectively. To fill this gap, we propose a synthetic data generation pipeline that leverages LLMs to produce a diverse, domain-rich dataset spanning multiple PII types and relevance levels. Using this dataset, we fine-tune an SLM to detect PII spans, classify their types, and estimate contextual relevance. Our experiments show that relevance-aware PII detection with a fine-tuned SLM substantially outperforms existing baselines in span, relevance and type accuracy while preserving significantly higher downstream utility under anonymization. 

---
# Covo-Audio Technical Report 

**Authors**: Wenfu Wang, Chenxing Li, Liqiang Zhang, Yiyang Zhao, Yuxiang Zou, Hanzhao Li, Mingyu Cui, Hao Zhang, Kun Wei, Le Xu, Zikang Huang, Jiajun Xu, Jiliang Hu, Xiang He, Zeyu Xie, Jiawen Kang, Youjun Chen, Meng Yu, Dong Yu, Rilin Chen, Linlin Di, Shulin Feng, Na Hu, Yang Liu, Bang Wang, Shan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.09823)  

**Abstract**: In this work, we present Covo-Audio, a 7B-parameter end-to-end LALM that directly processes continuous audio inputs and generates audio outputs within a single unified architecture. Through large-scale curated pretraining and targeted post-training, Covo-Audio achieves state-of-the-art or competitive performance among models of comparable scale across a broad spectrum of tasks, including speech-text modeling, spoken dialogue, speech understanding, audio understanding, and full-duplex voice interaction. Extensive evaluations demonstrate that the pretrained foundation model exhibits strong speech-text comprehension and semantic reasoning capabilities on multiple benchmarks, outperforming representative open-source models of comparable scale. Furthermore, Covo-Audio-Chat, the dialogue-oriented variant, demonstrates strong spoken conversational abilities, including understanding, contextual reasoning, instruction following, and generating contextually appropriate and empathetic responses, validating its applicability to real-world conversational assistant scenarios. Covo-Audio-Chat-FD, the evolved full-duplex model, achieves substantially superior performance on both spoken dialogue capabilities and full-duplex interaction behaviors, demonstrating its competence in practical robustness. To mitigate the high cost of deploying end-to-end LALMs for natural conversational systems, we propose an intelligence-speaker decoupling strategy that separates dialogue intelligence from voice rendering, enabling flexible voice customization with minimal text-to-speech (TTS) data while preserving dialogue performance. Overall, our results highlight the strong potential of 7B-scale models to integrate sophisticated audio intelligence with high-level semantic reasoning, and suggest a scalable path toward more capable and versatile LALMs. 

---
# Would a Large Language Model Pay Extra for a View? Inferring Willingness to Pay from Subjective Choices 

**Authors**: Manon Reusens, Sofie Goethals, Toon Calders, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2602.09802)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in applications such as travel assistance and purchasing support, they are often required to make subjective choices on behalf of users in settings where no objectively correct answer exists. We study LLM decision-making in a travel-assistant context by presenting models with choice dilemmas and analyzing their responses using multinomial logit models to derive implied willingness to pay (WTP) estimates. These WTP values are subsequently compared to human benchmark values from the economics literature. In addition to a baseline setting, we examine how model behavior changes under more realistic conditions, including the provision of information about users' past choices and persona-based prompting. Our results show that while meaningful WTP values can be derived for larger LLMs, they also display systematic deviations at the attribute level. Additionally, they tend to overestimate human WTP overall, particularly when expensive options or business-oriented personas are introduced. Conditioning models on prior preferences for cheaper options yields valuations that are closer to human benchmarks. Overall, our findings highlight both the potential and the limitations of using LLMs for subjective decision support and underscore the importance of careful model selection, prompt design, and user representation when deploying such systems in practice. 

---
# QP-OneModel: A Unified Generative LLM for Multi-Task Query Understanding in Xiaohongshu Search 

**Authors**: Jianzhao Huang, Xiaorui Huang, Fei Zhao, Yunpeng Liu, Hui Zhang, Fangcheng Shi, Congfeng Li, Zechen Sun, Yi Wu, Yao Hu, Yunhan Bai, Shaosheng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2602.09901)  

**Abstract**: Query Processing (QP) bridges user intent and content supply in large-scale Social Network Service (SNS) search engines. Traditional QP systems rely on pipelines of isolated discriminative models (e.g., BERT), suffering from limited semantic understanding and high maintenance overhead. While Large Language Models (LLMs) offer a potential solution, existing approaches often optimize sub-tasks in isolation, neglecting intrinsic semantic synergy and necessitating independent iterations. Moreover, standard generative methods often lack grounding in SNS scenarios, failing to bridge the gap between open-domain corpora and informal SNS linguistic patterns, while struggling to adhere to rigorous business definitions. We present QP-OneModel, a Unified Generative LLM for Multi-Task Query Understanding in the SNS domain. We reformulate heterogeneous sub-tasks into a unified sequence generation paradigm, adopting a progressive three-stage alignment strategy culminating in multi-reward Reinforcement Learning. Furthermore, QP-OneModel generates intent descriptions as a novel high-fidelity semantic signal, effectively augmenting downstream tasks such as query rewriting and ranking. Offline evaluations show QP-OneModel achieves a 7.35% overall gain over discriminative baselines, with significant F1 boosts in NER (+9.01%) and Term Weighting (+9.31%). It also exhibits superior generalization, surpassing a 32B model by 7.60% accuracy on unseen tasks. Fully deployed at Xiaohongshu, online A/B tests confirm its industrial value, optimizing retrieval relevance (DCG) by 0.21% and lifting user retention by 0.044%. 

---
# LingxiDiagBench: A Multi-Agent Framework for Benchmarking LLMs in Chinese Psychiatric Consultation and Diagnosis 

**Authors**: Shihao Xu, Tiancheng Zhou, Jiatong Ma, Yanli Ding, Yiming Yan, Ming Xiao, Guoyi Li, Haiyang Geng, Yunyun Han, Jianhua Chen, Yafeng Deng  

**Link**: [PDF](https://arxiv.org/pdf/2602.09379)  

**Abstract**: Mental disorders are highly prevalent worldwide, but the shortage of psychiatrists and the inherent subjectivity of interview-based diagnosis create substantial barriers to timely and consistent mental-health assessment. Progress in AI-assisted psychiatric diagnosis is constrained by the absence of benchmarks that simultaneously provide realistic patient simulation, clinician-verified diagnostic labels, and support for dynamic multi-turn consultation. We present LingxiDiagBench, a large-scale multi-agent benchmark that evaluates LLMs on both static diagnostic inference and dynamic multi-turn psychiatric consultation in Chinese. At its core is LingxiDiag-16K, a dataset of 16,000 EMR-aligned synthetic consultation dialogues designed to reproduce real clinical demographic and diagnostic distributions across 12 ICD-10 psychiatric categories. Through extensive experiments across state-of-the-art LLMs, we establish key findings: (1) although LLMs achieve high accuracy on binary depression--anxiety classification (up to 92.3%), performance deteriorates substantially for depression--anxiety comorbidity recognition (43.0%) and 12-way differential diagnosis (28.5%); (2) dynamic consultation often underperforms static evaluation, indicating that ineffective information-gathering strategies significantly impair downstream diagnostic reasoning; (3) consultation quality assessed by LLM-as-a-Judge shows only moderate correlation with diagnostic accuracy, suggesting that well-structured questioning alone does not ensure correct diagnostic decisions. We release LingxiDiag-16K and the full evaluation framework to support reproducible research at this https URL. 

---
# PABU: Progress-Aware Belief Update for Efficient LLM Agents 

**Authors**: Haitao Jiang, Lin Ge, Hengrui Cai, Rui Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.09138)  

**Abstract**: Large Language Model (LLM) agents commonly condition actions on full action-observation histories, which introduce task-irrelevant information that easily leads to redundant actions and higher inference cost. We propose Progress-Aware Belief Update (PABU), a belief-state framework that compactly represents an agent's state by explicitly modeling task progress and selectively retaining past actions and observations. At each step, the agent predicts its relative progress since the previous round and decides whether the newly encountered interaction should be stored, conditioning future decisions only on the retained subset. Across eight environments in the AgentGym benchmark, and using identical training trajectories, PABU achieves an 81.0% task completion rate, outperforming previous State of the art (SoTA) models with full-history belief by 23.9%. Additionally, PABU's progress-oriented action selection improves efficiency, reducing the average number of interaction steps to 9.5, corresponding to a 26.9% reduction. Ablation studies show that both explicit progress prediction and selective retention are necessary for robust belief learning and performance gains. 

---
# SARM: LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking 

**Authors**: Ruochen Yang, Yueyang Liu, Zijie Zhuang, Changxin Lao, Yuhui Zhang, Jiangxia Cao, Jia Xu, Xiang Chen, Haoke Xiao, Xiangyu Wu, Xiaoyou Zhou, Xiao Lv, Shuang Yang, Tingwen Liu, Zhaojie Liu, Han Li, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2602.09401)  

**Abstract**: Large-scale live-streaming recommendation requires precise modeling of non-stationary content semantics under strict real-time serving constraints. In industrial deployment, two common approaches exhibit fundamental limitations: discrete semantic abstractions sacrifice descriptive precision through clustering, while dense multimodal embeddings are extracted independently and remain weakly aligned with ranking optimization, limiting fine-grained content-aware ranking. To address these limitations, we propose \textbf{SARM}, an end-to-end ranking architecture that integrates natural-language semantic anchors directly into ranking optimization, enabling fine-grained author representations conditioned on multimodal content. Each semantic anchor is represented as learnable text tokens jointly optimized with ranking features, allowing the model to adapt content descriptions to ranking objectives. A lightweight dual-token gated design captures domain-specific live-streaming semantics, while an asymmetric deployment strategy preserves low-latency online training and serving. Extensive offline evaluation and large-scale A/B tests show consistent improvements over production baselines. SARM is fully deployed and serves over 400 million users daily. 

---
# Internalizing Multi-Agent Reasoning for Accurate and Efficient LLM-based Recommendation 

**Authors**: Yang Wu, Haoze Wang, Qian Li, Jun Zhang, Huan Yu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.09829)  

**Abstract**: Large Language Models (LLMs) are reshaping recommender systems by leveraging extensive world knowledge and semantic reasoning to interpret user intent. However, effectively integrating these capabilities with collaborative signals while avoiding prohibitive inference latency remains a critical bottleneck. To address this, we propose a trajectory-driven internalization framework to develop a Single-agent Trajectory-Aligned Recommender (STAR). Specifically, to internalize complex reasoning capabilities into a single efficient model, we first design a multi-agent teacher system capable of multi-turn tool usage and reflection. This teacher utilizes a Collaborative Signal Translation mechanism to explicitly convert latent behavioral patterns into descriptive natural language evidence to enhance reasoning accuracy. Subsequently, a trajectory-driven distillation pipeline transfers this agentic logic, including planning, tool usage, and self-reflection, into the compact STAR model. Extensive experiments demonstrate that STAR surpasses its teacher by 8.7% to 39.5% while eliminating iterative latency, paving the way for real-time, reasoning-enhanced recommendation. 

---
# Closing Reasoning Gaps in Clinical Agents with Differential Reasoning Learning 

**Authors**: Jinsong Liu, Yuhang Jiang, Ramayya Krishnan, Rema Padman, Yiye Zhang, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2602.09945)  

**Abstract**: Clinical decision support requires not only correct answers but also clinically valid reasoning. We propose Differential Reasoning Learning (DRL), a framework that improves clinical agents by learning from reasoning discrepancies. From reference reasoning rationales (e.g., physician-authored clinical rationale, clinical guidelines, or outputs from more capable models) and the agent's free-form chain-of-thought (CoT), DRL extracts reasoning graphs as directed acyclic graphs (DAGs) and performs a clinically weighted graph edit distance (GED)-based discrepancy analysis. An LLM-as-a-judge aligns semantically equivalent nodes and diagnoses discrepancies between graphs. These graph-level discrepancy diagnostics are converted into natural-language instructions and stored in a Differential Reasoning Knowledge Base (DR-KB). At inference, we retrieve top-$k$ instructions via Retrieval-Augmented Generation (RAG) to augment the agent prompt and patch likely logic gaps. Evaluation on open medical question answering (QA) benchmarks and a Return Visit Admissions (RVA) prediction task from internal clinical data demonstrates gains over baselines, improving both final-answer accuracy and reasoning fidelity. Ablation studies confirm gains from infusing reference reasoning rationales and the top-$k$ retrieval strategy. Clinicians' review of the output provides further assurance of the approach. Together, results suggest that DRL supports more reliable clinical decision-making in complex reasoning scenarios and offers a practical mechanism for deployment under limited token budgets. 

---
# ESTAR: Early-Stopping Token-Aware Reasoning For Efficient Inference 

**Authors**: Junda Wang, Zhichao Yang, Dongxu Zhang, Sanjit Singh Batra, Robert E. Tillman  

**Link**: [PDF](https://arxiv.org/pdf/2602.10004)  

**Abstract**: Large reasoning models (LRMs) achieve state-of-the-art performance by generating long chains-of-thought, but often waste computation on redundant reasoning after the correct answer has already been reached. We introduce Early-Stopping for Token-Aware Reasoning (ESTAR), which detects and reduces such reasoning redundancy to improve efficiency without sacrificing accuracy. Our method combines (i) a trajectory-based classifier that identifies when reasoning can be safely stopped, (ii) supervised fine-tuning to teach LRMs to propose self-generated <stop> signals, and (iii) <stop>-aware reinforcement learning that truncates rollouts at self-generated stop points with compute-aware rewards. Experiments on four reasoning datasets show that ESTAR reduces reasoning length by about 3.7x (from 4,799 to 1,290) while preserving accuracy (74.9% vs. 74.2%), with strong cross-domain generalization. These results highlight early stopping as a simple yet powerful mechanism for improving reasoning efficiency in LRMs. 

---
# Chain of Mindset: Reasoning with Adaptive Cognitive Modes 

**Authors**: Tianyi Jiang, Arctanx An, Hengyi Feng, Naixin Zhai, Haodong Li, Xiaomin Yu, Jiahui Liu, Hanwen Du, Shuo Zhang, Zhi Yang, Jie Huang, Yuhua Li, Yongxin Ni, Huacan Wang, Ronghao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.10063)  

**Abstract**: Human problem-solving is never the repetition of a single mindset, by which we mean a distinct mode of cognitive processing. When tackling a specific task, we do not rely on a single mindset; instead, we integrate multiple mindsets within the single solution process. However, existing LLM reasoning methods fall into a common trap: they apply the same fixed mindset across all steps, overlooking that different stages of solving the same problem require fundamentally different mindsets. This single-minded assumption prevents models from reaching the next level of intelligence. To address this limitation, we propose Chain of Mindset (CoM), a training-free agentic framework that enables step-level adaptive mindset orchestration. CoM decomposes reasoning into four functionally heterogeneous mindsets: Spatial, Convergent, Divergent, and Algorithmic. A Meta-Agent dynamically selects the optimal mindset based on the evolving reasoning state, while a bidirectional Context Gate filters cross-module information flow to maintain effectiveness and efficiency. Experiments across six challenging benchmarks spanning mathematics, code generation, scientific QA, and spatial reasoning demonstrate that CoM achieves state-of-the-art performance, outperforming the strongest baseline by 4.96\% and 4.72\% in overall accuracy on Qwen3-VL-32B-Instruct and Gemini-2.0-Flash, while balancing reasoning efficiency. Our code is publicly available at \href{this https URL}{this https URL}. 

---
# Why Do AI Agents Systematically Fail at Cloud Root Cause Analysis? 

**Authors**: Taeyoon Kim, Woohyeok Park, Hoyeong Yun, Kyungyong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.09937)  

**Abstract**: Failures in large-scale cloud systems incur substantial financial losses, making automated Root Cause Analysis (RCA) essential for operational stability. Recent efforts leverage Large Language Model (LLM) agents to automate this task, yet existing systems exhibit low detection accuracy even with capable models, and current evaluation frameworks assess only final answer correctness without revealing why the agent's reasoning failed. This paper presents a process level failure analysis of LLM-based RCA agents. We execute the full OpenRCA benchmark across five LLM models, producing 1,675 agent runs, and classify observed failures into 12 pitfall types across intra-agent reasoning, inter-agent communication, and agent-environment interaction. Our analysis reveals that the most prevalent pitfalls, notably hallucinated data interpretation and incomplete exploration, persist across all models regardless of capability tier, indicating that these failures originate from the shared agent architecture rather than from individual model limitations. Controlled mitigation experiments further show that prompt engineering alone cannot resolve the dominant pitfalls, whereas enriching the inter-agent communication protocol reduces communication-related failures by up to 15 percentage points. The pitfall taxonomy and diagnostic methodology developed in this work provide a foundation for designing more reliable autonomous agents for cloud RCA. 

---
# Autoregressive Direct Preference Optimization 

**Authors**: Masanari Oi, Mahiro Ukai, Masahiro Kaneko, Naoaki Okazaki, Nakamasa Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2602.09533)  

**Abstract**: Direct preference optimization (DPO) has emerged as a promising approach for aligning large language models (LLMs) with human preferences. However, the widespread reliance on the response-level Bradley-Terry (BT) model may limit its full potential, as the reference and learnable models are assumed to be autoregressive only after deriving the objective function. Motivated by this limitation, we revisit the theoretical foundations of DPO and propose a novel formulation that explicitly introduces the autoregressive assumption prior to applying the BT model. By reformulating and extending DPO, we derive a novel variant, termed Autoregressive DPO (ADPO), that explicitly integrates autoregressive modeling into the preference optimization framework. Without violating the theoretical foundations, the derived loss takes an elegant form: it shifts the summation operation in the DPO objective outside the log-sigmoid function. Furthermore, through theoretical analysis of ADPO, we show that there exist two length measures to be considered when designing DPO-based algorithms: the token length $\mu$ and the feedback length $\mu$'. To the best of our knowledge, we are the first to explicitly distinguish these two measures and analyze their implications for preference optimization in LLMs. 

---
# ClinAlign: Scaling Healthcare Alignment from Clinician Preference 

**Authors**: Shiwei Lyu, Xidong Wang, Lei Liu, Hao Zhu, Chaohe Zhang, Jian Wang, Jinjie Gu, Benyou Wang, Yue Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.09653)  

**Abstract**: Although large language models (LLMs) demonstrate expert-level medical knowledge, aligning their open-ended outputs with fine-grained clinician preferences remains challenging. Existing methods often rely on coarse objectives or unreliable automated judges that are weakly grounded in professional guidelines. We propose a two-stage framework to address this gap. First, we introduce HealthRubrics, a dataset of 7,034 physician-verified preference examples in which clinicians refine LLM-drafted rubrics to meet rigorous medical standards. Second, we distill these rubrics into HealthPrinciples: 119 broadly reusable, clinically grounded principles organized by clinical dimensions, enabling scalable supervision beyond manual annotation. We use HealthPrinciples for (1) offline alignment by synthesizing rubrics for unlabeled queries and (2) an inference-time tool for guided self-revision. A 30B parameter model that activates only 3B parameters at inference trained with our framework achieves 33.4% on HealthBench-Hard, outperforming much larger models including Deepseek-R1 and o3, establishing a resource-efficient baseline for clinical alignment. 

---
# P1-VL: Bridging Visual Perception and Scientific Reasoning in Physics Olympiads 

**Authors**: Yun Luo, Futing Wang, Qianjia Cheng, Fangchen Yu, Haodi Lei, Jianhao Yan, Chenxi Li, Jiacheng Chen, Yufeng Zhao, Haiyuan Wan, Yuchen Zhang, Shenghe Zheng, Junchi Yao, Qingyang Zhang, Haonan He, Wenxuan Zeng, Li Sheng, Chengxing Xie, Yuxin Zuo, Yizhuo Li, Yulun Wu, Rui Huang, Dongzhan Zhou, Kai Chen, Yu Qiao, Lei Bai, Yu Cheng, Ning Ding, Bowen Zhou, Peng Ye, Ganqu Cui  

**Link**: [PDF](https://arxiv.org/pdf/2602.09443)  

**Abstract**: The transition from symbolic manipulation to science-grade reasoning represents a pivotal frontier for Large Language Models (LLMs), with physics serving as the critical test anchor for binding abstract logic to physical reality. Physics demands that a model maintain physical consistency with the laws governing the universe, a task that fundamentally requires multimodal perception to ground abstract logic in reality. At the Olympiad level, diagrams are often constitutive rather than illustrative, containing essential constraints, such as boundary conditions and spatial symmetries, that are absent from the text. To bridge this visual-logical gap, we introduce P1-VL, a family of open-source vision-language models engineered for advanced scientific reasoning. Our method harmonizes Curriculum Reinforcement Learning, which employs progressive difficulty expansion to stabilize post-training, with Agentic Augmentation, enabling iterative self-verification at inference. Evaluated on HiPhO, a rigorous benchmark of 13 exams from 2024-2025, our flagship P1-VL-235B-A22B becomes the first open-source Vision-Language Model (VLM) to secure 12 gold medals and achieves the state-of-the-art performance in the open-source models. Our agent-augmented system achieves the No.2 overall rank globally, trailing only Gemini-3-Pro. Beyond physics, P1-VL demonstrates remarkable scientific reasoning capacity and generalizability, establishing significant leads over base models in STEM benchmarks. By open-sourcing P1-VL, we provide a foundational step toward general-purpose physical intelligence to better align visual perceptions with abstract physical laws for machine scientific discovery. 

---
# GHS-TDA: A Synergistic Reasoning Framework Integrating Global Hypothesis Space with Topological Data Analysis 

**Authors**: Jiaquan Zhang, Chaoning Zhang, Shuxu Chen, Xudong Wang, Zhenzhen Huang, Pengcheng Zheng, Shuai Yuan, Sheng Zheng, Qigan Sun, Jie Zou, Lik-Hang Lee, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.09794)  

**Abstract**: Chain-of-Thought (CoT) has been shown to significantly improve the reasoning accuracy of large language models (LLMs) on complex tasks. However, due to the autoregressive, step-by-step generation paradigm, existing CoT methods suffer from two fundamental limitations. First, the reasoning process is highly sensitive to early decisions: once an initial error is introduced, it tends to propagate and amplify through subsequent steps, while the lack of a global coordination and revision mechanism makes such errors difficult to correct, ultimately leading to distorted reasoning chains. Second, current CoT approaches lack structured analysis techniques for filtering redundant reasoning and extracting key reasoning features, resulting in unstable reasoning processes and limited interpretability. To address these issues, we propose GHS-TDA. GHS-TDA first constructs a semantically enriched global hypothesis graph to aggregate, align, and coordinate multiple candidate reasoning paths, thereby providing alternative global correction routes when local reasoning fails. It then applies topological data analysis based on persistent homology to capture stable multi-scale structures, remove redundancy and inconsistencies, and extract a more reliable reasoning skeleton. By jointly leveraging reasoning diversity and topological stability, GHS-TDA achieves self-adaptive convergence, produces high-confidence and interpretable reasoning paths, and consistently outperforms strong baselines in terms of both accuracy and robustness across multiple reasoning benchmarks. 

---
# CoMMa: Contribution-Aware Medical Multi-Agents From A Game-Theoretic Perspective 

**Authors**: Yichen Wu, Yujin Oh, Sangjoon Park, Kailong Fan, Dania Daye, Hana Farzaneh, Xiang Li, Raul Uppot, Quanzheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.09159)  

**Abstract**: Recent multi-agent frameworks have broadened the ability to tackle oncology decision support tasks that require reasoning over dynamic, heterogeneous patient data. We propose Contribution-Aware Medical Multi-Agents (CoMMa), a decentralized LLM-agent framework in which specialists operate on partitioned evidence and coordinate through a game-theoretic objective for robust decision-making. In contrast to most agent architectures relying on stochastic narrative-based reasoning, CoMMa utilizes deterministic embedding projections to approximate contribution-aware credit assignment. This yields explicit evidence attribution by estimating each agent's marginal utility, producing interpretable and mathematically grounded decision pathways with improved stability. Evaluated on diverse oncology benchmarks, including a real-world multidisciplinary tumor board dataset, CoMMa achieves higher accuracy and more stable performance than data-centralized and role-based multi-agents baselines. 

---
# Biases in the Blind Spot: Detecting What LLMs Fail to Mention 

**Authors**: Iván Arcuschin, David Chanin, Adrià Garriga-Alonso, Oana-Maria Camburu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10117)  

**Abstract**: Large Language Models (LLMs) often provide chain-of-thought (CoT) reasoning traces that appear plausible, but may hide internal biases. We call these *unverbalized biases*. Monitoring models via their stated reasoning is therefore unreliable, and existing bias evaluations typically require predefined categories and hand-crafted datasets. In this work, we introduce a fully automated, black-box pipeline for detecting task-specific unverbalized biases. Given a task dataset, the pipeline uses LLM autoraters to generate candidate bias concepts. It then tests each concept on progressively larger input samples by generating positive and negative variations, and applies statistical techniques for multiple testing and early stopping. A concept is flagged as an unverbalized bias if it yields statistically significant performance differences while not being cited as justification in the model's CoTs. We evaluate our pipeline across six LLMs on three decision tasks (hiring, loan approval, and university admissions). Our technique automatically discovers previously unknown biases in these models (e.g., Spanish fluency, English proficiency, writing formality). In the same run, the pipeline also validates biases that were manually identified by prior work (gender, race, religion, ethnicity). More broadly, our proposed approach provides a practical, scalable path to automatic task-specific bias discovery. 

---
# Long Chain-of-Thought Compression via Fine-Grained Group Policy Optimization 

**Authors**: Xinchen Han, Hossam Afifi, Michel Marot, Xilu Wang, Lu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.10048)  

**Abstract**: Large Language Models (LLMs) often generate unnecessarily verbose Chain-of-Thought (CoT) reasoning that increases computational costs and latency without proportional performance gains. In this paper, we propose \textbf{F}ine-grained \textbf{G}roup policy \textbf{O}ptimization (\textbf{FGO}), a Reinforcement Learning (RL) algorithm that refines group responses by subdividing them and assigning appropriate weights based on length and entropy, thereby enabling effective CoT compression. Meanwhile, as an enhanced variant of Group Relative Policy Optimization (GRPO), FGO successfully addresses two major limitations of the GRPO: inefficient data utilization and entropy collapse. We evaluate FGO on multiple reasoning LLMs and benchmarks, including MATH500, AIME24, AMC23, and Minerva. Experimental results show that FGO achieves efficient CoT compression without degrading performance, and simultaneously resolves the key limitations of GRPO. 

---
# MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering 

**Authors**: Sieun Hyeon, Jusang Oh, Sunghwan Steve Cho, Jaeyoung Do  

**Link**: [PDF](https://arxiv.org/pdf/2602.09642)  

**Abstract**: Recent advances in Large Language Models (LLMs) have significantly improved table understanding tasks such as Table Question Answering (TableQA), yet challenges remain in ensuring reliability, scalability, and efficiency, especially in resource-constrained or privacy-sensitive environments. In this paper, we introduce MATA, a multi-agent TableQA framework that leverages multiple complementary reasoning paths and a set of tools built with small language models. MATA generates candidate answers through diverse reasoning styles for a given table and question, then refines or selects the optimal answer with the help of these tools. Furthermore, it incorporates an algorithm designed to minimize expensive LLM agent calls, enhancing overall efficiency. MATA maintains strong performance with small, open-source models and adapts easily across various LLM types. Extensive experiments on two benchmarks of varying difficulty with ten different LLMs demonstrate that MATA achieves state-of-the-art accuracy and highly efficient reasoning while avoiding excessive LLM inference. Our results highlight that careful orchestration of multiple reasoning pathways yields scalable and reliable TableQA. The code is available at this https URL. 

---
# A Behavioral Fingerprint for Large Language Models: Provenance Tracking via Refusal Vectors 

**Authors**: Zhenyu Xu, Victor S. Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.09434)  

**Abstract**: Protecting the intellectual property of large language models (LLMs) is a critical challenge due to the proliferation of unauthorized derivative models. We introduce a novel fingerprinting framework that leverages the behavioral patterns induced by safety alignment, applying the concept of refusal vectors for LLM provenance tracking. These vectors, extracted from directional patterns in a model's internal representations when processing harmful versus harmless prompts, serve as robust behavioral fingerprints. Our contribution lies in developing a fingerprinting system around this concept and conducting extensive validation of its effectiveness for IP protection. We demonstrate that these behavioral fingerprints are highly robust against common modifications, including finetunes, merges, and quantization. Our experiments show that the fingerprint is unique to each model family, with low cosine similarity between independently trained models. In a large-scale identification task across 76 offspring models, our method achieves 100\% accuracy in identifying the correct base model family. Furthermore, we analyze the fingerprint's behavior under alignment-breaking attacks, finding that while performance degrades significantly, detectable traces remain. Finally, we propose a theoretical framework to transform this private fingerprint into a publicly verifiable, privacy-preserving artifact using locality-sensitive hashing and zero-knowledge proofs. 

---
# LLMAC: A Global and Explainable Access Control Framework with Large Language Model 

**Authors**: Sharif Noor Zisad, Ragib Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2602.09392)  

**Abstract**: Today's business organizations need access control systems that can handle complex, changing security requirements that go beyond what traditional methods can manage. Current approaches, such as Role-Based Access Control (RBAC), Attribute-Based Access Control (ABAC), and Discretionary Access Control (DAC), were designed for specific purposes. They cannot effectively manage the dynamic, situation-dependent workflows that modern systems require. In this research, we introduce LLMAC, a new unified approach using Large Language Models (LLMs) to combine these different access control methods into one comprehensive, understandable system. We used an extensive synthetic dataset that represents complex real-world scenarios, including policies for ownership verification, version management, workflow processes, and dynamic role separation. Using Mistral 7B, our trained LLM model achieved outstanding results with 98.5% accuracy, significantly outperforming traditional methods (RBAC: 14.5%, ABAC: 58.5%, DAC: 27.5%) while providing clear, human readable explanations for each decision. Performance testing shows that the system can be practically deployed with reasonable response times and computing resources. 

---
# Behavioral Economics of AI: LLM Biases and Corrections 

**Authors**: Pietro Bini, Lin William Cong, Xing Huang, Lawrence J. Jin  

**Link**: [PDF](https://arxiv.org/pdf/2602.09362)  

**Abstract**: Do generative AI models, particularly large language models (LLMs), exhibit systematic behavioral biases in economic and financial decisions? If so, how can these biases be mitigated? Drawing on the cognitive psychology and experimental economics literatures, we conduct the most comprehensive set of experiments to date$-$originally designed to document human biases$-$on prominent LLM families across model versions and scales. We document systematic patterns in LLM behavior. In preference-based tasks, responses become more human-like as models become more advanced or larger, while in belief-based tasks, advanced large-scale models frequently generate rational responses. Prompting LLMs to make rational decisions reduces biases. 

---
# Empowering Contrastive Federated Sequential Recommendation with LLMs 

**Authors**: Thi Minh Chau Nguyen, Minh Hieu Nguyen, Duc Anh Nguyen, Xuan Huong Tran, Thanh Trung Huynh, Quoc Viet Hung Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2602.09306)  

**Abstract**: Federated sequential recommendation (FedSeqRec) aims to perform next-item prediction while keeping user data decentralised, yet model quality is frequently constrained by fragmented, noisy, and homogeneous interaction logs stored on individual devices. Many existing approaches attempt to compensate through manual data augmentation or additional server-side constraints, but these strategies either introduce limited semantic diversity or increase system overhead. To overcome these challenges, we propose \textbf{LUMOS}, a parameter-isolated FedSeqRec architecture that integrates large language models (LLMs) as \emph{local semantic generators}. Instead of sharing gradients or auxiliary parameters, LUMOS privately invokes an on-device LLM to construct three complementary sequence variants from each user history: (i) \emph{future-oriented} trajectories that infer plausible behavioural continuations, (ii) \emph{semantically equivalent rephrasings} that retain user intent while diversifying interaction patterns, and (iii) \emph{preference-inconsistent counterfactuals} that serve as informative negatives. These synthesized sequences are jointly encoded within the federated backbone through a tri-view contrastive optimisation scheme, enabling richer representation learning without exposing sensitive information. Experimental results across three public benchmarks show that LUMOS achieves consistent gains over competitive centralised and federated baselines on HR@20 and NDCG@20. In addition, the use of semantically grounded positive signals and counterfactual negatives improves robustness under noisy and adversarial environments, even without dedicated server-side protection modules. Overall, this work demonstrates the potential of LLM-driven semantic generation as a new paradigm for advancing privacy-preserving federated recommendation. 

---
# MUZZLE: Adaptive Agentic Red-Teaming of Web Agents Against Indirect Prompt Injection Attacks 

**Authors**: Georgios Syros, Evan Rose, Brian Grinstead, Christoph Kerschbaumer, William Robertson, Cristina Nita-Rotaru, Alina Oprea  

**Link**: [PDF](https://arxiv.org/pdf/2602.09222)  

**Abstract**: Large language model (LLM) based web agents are increasingly deployed to automate complex online tasks by directly interacting with web sites and performing actions on users' behalf. While these agents offer powerful capabilities, their design exposes them to indirect prompt injection attacks embedded in untrusted web content, enabling adversaries to hijack agent behavior and violate user intent. Despite growing awareness of this threat, existing evaluations rely on fixed attack templates, manually selected injection surfaces, or narrowly scoped scenarios, limiting their ability to capture realistic, adaptive attacks encountered in practice. We present MUZZLE, an automated agentic framework for evaluating the security of web agents against indirect prompt injection attacks. MUZZLE utilizes the agent's trajectories to automatically identify high-salience injection surfaces, and adaptively generate context-aware malicious instructions that target violations of confidentiality, integrity, and availability. Unlike prior approaches, MUZZLE adapts its attack strategy based on the agent's observed execution trajectory and iteratively refines attacks using feedback from failed executions. We evaluate MUZZLE across diverse web applications, user tasks, and agent configurations, demonstrating its ability to automatically and adaptively assess the security of web agents with minimal human intervention. Our results show that MUZZLE effectively discovers 37 new attacks on 4 web applications with 10 adversarial objectives that violate confidentiality, availability, or privacy properties. MUZZLE also identifies novel attack strategies, including 2 cross-application prompt injection attacks and an agent-tailored phishing scenario. 

---
# Distributed Hybrid Parallelism for Large Language Models: Comparative Study and System Design Guide 

**Authors**: Hossam Amer, Rezaul Karim, Ali Pourranjbar, Weiwei Zhang, Walid Ahmed, Boxing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.09109)  

**Abstract**: With the rapid growth of large language models (LLMs), a wide range of methods have been developed to distribute computation and memory across hardware devices for efficient training and inference. While existing surveys provide descriptive overviews of these techniques, systematic analysis of their benefits and trade offs and how such insights can inform principled methodology for designing optimal distributed systems remain limited. This paper offers a comprehensive review of collective operations and distributed parallel strategies, complemented by mathematical formulations to deepen theoretical understanding. We further examine hybrid parallelization designs, emphasizing communication computation overlap across different stages of model deployment, including both training and inference. Recent advances in automated search for optimal hybrid parallelization strategies using cost models are also discussed. Moreover, we present case studies with mainstream architecture categories to reveal empirical insights to guide researchers and practitioners in parallelism strategy selection. Finally, we highlight open challenges and limitations of current LLM training paradigms and outline promising directions for the next generation of large scale model development. 

---
# NarraScore: Bridging Visual Narrative and Musical Dynamics via Hierarchical Affective Control 

**Authors**: Yufan Wen, Zhaocheng Liu, YeGuo Hua, Ziyi Guo, Lihua Zhang, Chun Yuan, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.09070)  

**Abstract**: Synthesizing coherent soundtracks for long-form videos remains a formidable challenge, currently stalled by three critical impediments: computational scalability, temporal coherence, and, most critically, a pervasive semantic blindness to evolving narrative logic. To bridge these gaps, we propose NarraScore, a hierarchical framework predicated on the core insight that emotion serves as a high-density compression of narrative logic. Uniquely, we repurpose frozen Vision-Language Models (VLMs) as continuous affective sensors, distilling high-dimensional visual streams into dense, narrative-aware Valence-Arousal trajectories. Mechanistically, NarraScore employs a Dual-Branch Injection strategy to reconcile global structure with local dynamism: a \textit{Global Semantic Anchor} ensures stylistic stability, while a surgical \textit{Token-Level Affective Adapter} modulates local tension via direct element-wise residual injection. This minimalist design bypasses the bottlenecks of dense attention and architectural cloning, effectively mitigating the overfitting risks associated with data scarcity. Experiments demonstrate that NarraScore achieves state-of-the-art consistency and narrative alignment with negligible computational overhead, establishing a fully autonomous paradigm for long-video soundtrack generation. 

---
