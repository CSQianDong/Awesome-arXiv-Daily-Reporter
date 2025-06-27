# Unveiling Causal Reasoning in Large Language Models: Reality or Mirage? 

**Authors**: Haoang Chi, He Li, Wenjing Yang, Feng Liu, Long Lan, Xiaoguang Ren, Tongliang Liu, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.21215)  

**Abstract**: Causal reasoning capability is critical in advancing large language models (LLMs) toward strong artificial intelligence. While versatile LLMs appear to have demonstrated capabilities in understanding contextual causality and providing responses that obey the laws of causality, it remains unclear whether they perform genuine causal reasoning akin to humans. However, current evidence indicates the contrary. Specifically, LLMs are only capable of performing shallow (level-1) causal reasoning, primarily attributed to the causal knowledge embedded in their parameters, but they lack the capacity for genuine human-like (level-2) causal reasoning. To support this hypothesis, methodologically, we delve into the autoregression mechanism of transformer-based LLMs, revealing that it is not inherently causal. Empirically, we introduce a new causal Q&A benchmark called CausalProbe-2024, whose corpora are fresh and nearly unseen for the studied LLMs. The LLMs exhibit a significant performance drop on CausalProbe-2024 compared to earlier benchmarks, indicating the fact that they primarily engage in level-1 causal reasoning. To bridge the gap towards level-2 causal reasoning, we draw inspiration from the fact that human reasoning is usually facilitated by general knowledge and intended goals. We propose G^2-Reasoner, a method that incorporates general knowledge and goal-oriented prompts into LLMs' causal reasoning processes. Experiments demonstrate that G^2-Reasoner significantly enhances LLMs' causal reasoning capability, particularly in fresh and counterfactual contexts. This work sheds light on a new path for LLMs to advance towards genuine causal reasoning, going beyond level-1 and making strides towards level-2. 

---
# PsyLite Technical Report 

**Authors**: Fangjun Ding, Renyu Zhang, Xinyu Feng, Chengye Xie, Zheng Zhang, Yanting Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21536)  

**Abstract**: With the rapid development of digital technology, AI-driven psychological counseling has gradually become an important research direction in the field of mental health. However, existing models still have deficiencies in dialogue safety, detailed scenario handling, and lightweight deployment. To address these issues, this study proposes PsyLite, a lightweight psychological counseling large language model agent developed based on the base model InternLM2.5-7B-chat. Through a two-stage training strategy (hybrid distillation data fine-tuning and ORPO preference optimization), PsyLite enhances the model's deep-reasoning ability, psychological counseling ability, and safe dialogue ability. After deployment using Ollama and Open WebUI, a custom workflow is created with Pipelines. An innovative conditional RAG is designed to introduce crosstalk humor elements at appropriate times during psychological counseling to enhance user experience and decline dangerous requests to strengthen dialogue safety. Evaluations show that PsyLite outperforms the baseline models in the Chinese general evaluation (CEval), psychological counseling professional evaluation (CPsyCounE), and dialogue safety evaluation (SafeDialBench), particularly in psychological counseling professionalism (CPsyCounE score improvement of 47.6\%) and dialogue safety (\safe{} score improvement of 2.4\%). Additionally, the model uses quantization technology (GGUF q4\_k\_m) to achieve low hardware deployment (5GB memory is sufficient for operation), providing a feasible solution for psychological counseling applications in resource-constrained environments. 

---
# Dynamic Context-Aware Prompt Recommendation for Domain-Specific AI Applications 

**Authors**: Xinye Tang, Haijun Zhai, Chaitanya Belwal, Vineeth Thayanithi, Philip Baumann, Yogesh K Roy  

**Link**: [PDF](https://arxiv.org/pdf/2506.20815)  

**Abstract**: LLM-powered applications are highly susceptible to the quality of user prompts, and crafting high-quality prompts can often be challenging especially for domain-specific applications. This paper presents a novel dynamic context-aware prompt recommendation system for domain-specific AI applications. Our solution combines contextual query analysis, retrieval-augmented knowledge grounding, hierarchical skill organization, and adaptive skill ranking to generate relevant and actionable prompt suggestions.
The system leverages behavioral telemetry and a two-stage hierarchical reasoning process to dynamically select and rank relevant skills, and synthesizes prompts using both predefined and adaptive templates enhanced with few-shot learning. Experiments on real-world datasets demonstrate that our approach achieves high usefulness and relevance, as validated by both automated and expert evaluations. 

---
# mTSBench: Benchmarking Multivariate Time Series Anomaly Detection and Model Selection at Scale 

**Authors**: Xiaona Zhou, Constantin Brif, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2506.21550)  

**Abstract**: Multivariate time series anomaly detection (MTS-AD) is critical in domains like healthcare, cybersecurity, and industrial monitoring, yet remains challenging due to complex inter-variable dependencies, temporal dynamics, and sparse anomaly labels. We introduce mTSBench, the largest benchmark to date for MTS-AD and unsupervised model selection, spanning 344 labeled time series across 19 datasets and 12 diverse application domains. mTSBench evaluates 24 anomaly detection methods, including large language model (LLM)-based detectors for multivariate time series, and systematically benchmarks unsupervised model selection techniques under standardized conditions. Consistent with prior findings, our results confirm that no single detector excels across datasets, underscoring the importance of model selection. However, even state-of-the-art selection methods remain far from optimal, revealing critical gaps. mTSBench provides a unified evaluation suite to enable rigorous, reproducible comparisons and catalyze future advances in adaptive anomaly detection and robust model selection. 

---
# TableMoE: Neuro-Symbolic Routing for Structured Expert Reasoning in Multimodal Table Understanding 

**Authors**: Junwen Zhang, Pu Chen, Yin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21393)  

**Abstract**: Multimodal understanding of tables in real-world contexts is challenging due to the complexity of structure, symbolic density, and visual degradation (blur, skew, watermarking, incomplete structures or fonts, multi-span or hierarchically nested layouts). Existing multimodal large language models (MLLMs) struggle with such WildStruct conditions, resulting in limited performance and poor generalization. To address these challenges, we propose TableMoE, a neuro-symbolic Mixture-of-Connector-Experts (MoCE) architecture specifically designed for robust, structured reasoning over multimodal table data. TableMoE features an innovative Neuro-Symbolic Routing mechanism, which predicts latent semantic token roles (e.g., header, data cell, axis, formula) and dynamically routes table elements to specialized experts (Table-to-HTML, Table-to-JSON, Table-to-Code) using a confidence-aware gating strategy informed by symbolic reasoning graphs. To facilitate effective alignment-driven pretraining, we introduce the large-scale TableMoE-Align dataset, consisting of 1.2M table-HTML-JSON-code quadruples across finance, science, biomedicine and industry, utilized exclusively for model pretraining. For evaluation, we curate and release four challenging WildStruct benchmarks: WMMFinQA, WMMTatQA, WMMTabDialog, and WMMFinanceMath, designed specifically to stress-test models under real-world multimodal degradation and structural complexity. Experimental results demonstrate that TableMoE significantly surpasses existing state-of-the-art models. Extensive ablation studies validate each core component, emphasizing the critical role of Neuro-Symbolic Routing and structured expert alignment. Through qualitative analyses, we further showcase TableMoE's interpretability and enhanced robustness, underscoring the effectiveness of integrating neuro-symbolic reasoning for multimodal table understanding. 

---
# Potemkin Understanding in Large Language Models 

**Authors**: Marina Mancoridis, Bec Weeks, Keyon Vafa, Sendhil Mullainathan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21521)  

**Abstract**: Large language models (LLMs) are regularly evaluated using benchmark datasets. But what justifies making inferences about an LLM's capabilities based on its answers to a curated set of questions? This paper first introduces a formal framework to address this question. The key is to note that the benchmarks used to test LLMs -- such as AP exams -- are also those used to test people. However, this raises an implication: these benchmarks are only valid tests if LLMs misunderstand concepts in ways that mirror human misunderstandings. Otherwise, success on benchmarks only demonstrates potemkin understanding: the illusion of understanding driven by answers irreconcilable with how any human would interpret a concept. We present two procedures for quantifying the existence of potemkins: one using a specially designed benchmark in three domains, the other using a general procedure that provides a lower-bound on their prevalence. We find that potemkins are ubiquitous across models, tasks, and domains. We also find that these failures reflect not just incorrect understanding, but deeper internal incoherence in concept representations. 

---
# Domain Knowledge-Enhanced LLMs for Fraud and Concept Drift Detection 

**Authors**: Ali Şenol, Garima Agrawal, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21443)  

**Abstract**: Detecting deceptive conversations on dynamic platforms is increasingly difficult due to evolving language patterns and Concept Drift (CD)\-i.e., semantic or topical shifts that alter the context or intent of interactions over time. These shifts can obscure malicious intent or mimic normal dialogue, making accurate classification challenging. While Large Language Models (LLMs) show strong performance in natural language tasks, they often struggle with contextual ambiguity and hallucinations in risk\-sensitive scenarios. To address these challenges, we present a Domain Knowledge (DK)\-Enhanced LLM framework that integrates pretrained LLMs with structured, task\-specific insights to perform fraud and concept drift detection. The proposed architecture consists of three main components: (1) a DK\-LLM module to detect fake or deceptive conversations; (2) a drift detection unit (OCDD) to determine whether a semantic shift has occurred; and (3) a second DK\-LLM module to classify the drift as either benign or fraudulent. We first validate the value of domain knowledge using a fake review dataset and then apply our full framework to SEConvo, a multiturn dialogue dataset that includes various types of fraud and spam attacks. Results show that our system detects fake conversations with high accuracy and effectively classifies the nature of drift. Guided by structured prompts, the LLaMA\-based implementation achieves 98\% classification accuracy. Comparative studies against zero\-shot baselines demonstrate that incorporating domain knowledge and drift awareness significantly improves performance, interpretability, and robustness in high\-stakes NLP applications. 

---
# Scalable Bayesian Low-Rank Adaptation of Large Language Models via Stochastic Variational Subspace Inference 

**Authors**: Colin Samplawski, Adam D. Cobb, Manoj Acharya, Ramneet Kaur, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2506.21408)  

**Abstract**: Despite their widespread use, large language models (LLMs) are known to hallucinate incorrect information and be poorly calibrated. This makes the uncertainty quantification of these models of critical importance, especially in high-stakes domains, such as autonomy and healthcare. Prior work has made Bayesian deep learning-based approaches to this problem more tractable by performing inference over the low-rank adaptation (LoRA) parameters of a fine-tuned model. While effective, these approaches struggle to scale to larger LLMs due to requiring further additional parameters compared to LoRA. In this work we present $\textbf{Scala}$ble $\textbf{B}$ayesian $\textbf{L}$ow-Rank Adaptation via Stochastic Variational Subspace Inference (ScalaBL). We perform Bayesian inference in an $r$-dimensional subspace, for LoRA rank $r$. By repurposing the LoRA parameters as projection matrices, we are able to map samples from this subspace into the full weight space of the LLM. This allows us to learn all the parameters of our approach using stochastic variational inference. Despite the low dimensionality of our subspace, we are able to achieve competitive performance with state-of-the-art approaches while only requiring ${\sim}1000$ additional parameters. Furthermore, it allows us to scale up to the largest Bayesian LLM to date, with four times as a many base parameters as prior work. 

---
# Leveraging LLM-Assisted Query Understanding for Live Retrieval-Augmented Generation 

**Authors**: Guanting Dong, Xiaoxi Li, Yuyao Zhang, Mengjie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.21384)  

**Abstract**: Real-world live retrieval-augmented generation (RAG) systems face significant challenges when processing user queries that are often noisy, ambiguous, and contain multiple intents. While RAG enhances large language models (LLMs) with external knowledge, current systems typically struggle with such complex inputs, as they are often trained or evaluated on cleaner data. This paper introduces Omni-RAG, a novel framework designed to improve the robustness and effectiveness of RAG systems in live, open-domain settings. Omni-RAG employs LLM-assisted query understanding to preprocess user inputs through three key modules: (1) Deep Query Understanding and Decomposition, which utilizes LLMs with tailored prompts to denoise queries (e.g., correcting spelling errors) and decompose multi-intent queries into structured sub-queries; (2) Intent-Aware Knowledge Retrieval, which performs retrieval for each sub-query from a corpus (i.e., FineWeb using OpenSearch) and aggregates the results; and (3) Reranking and Generation, where a reranker (i.e., BGE) refines document selection before a final response is generated by an LLM (i.e., Falcon-10B) using a chain-of-thought prompt. Omni-RAG aims to bridge the gap between current RAG capabilities and the demands of real-world applications, such as those highlighted by the SIGIR 2025 LiveRAG Challenge, by robustly handling complex and noisy queries. 

---
# Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models 

**Authors**: Bram Willemsen, Gabriel Skantze  

**Link**: [PDF](https://arxiv.org/pdf/2506.21294)  

**Abstract**: In this paper, we explore the use of a text-only, autoregressive language modeling approach for the extraction of referring expressions from visually grounded dialogue. More specifically, the aim is to investigate the extent to which the linguistic context alone can inform the detection of mentions that have a (visually perceivable) referent in the visual context of the conversation. To this end, we adapt a pretrained large language model (LLM) to perform a relatively course-grained annotation of mention spans in unfolding conversations by demarcating mention span boundaries in text via next-token prediction. Our findings indicate that even when using a moderately sized LLM, relatively small datasets, and parameter-efficient fine-tuning, a text-only approach can be effective, highlighting the relative importance of the linguistic context for this task. Nevertheless, we argue that the task represents an inherently multimodal problem and discuss limitations fundamental to unimodal approaches. 

---
# $T^3$: Multi-level Tree-based Automatic Program Repair with Large Language Models 

**Authors**: Quanming Liu, Xupeng Bu, Zhichao Yan, Ru Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21211)  

**Abstract**: Automatic Program Repair (APR) is a core technology in software development and maintenance, with aims to enable automated defect repair with minimal human intervention. In recent years, the substantial advancements in Large Language Models (LLMs) and the Chain-of-Thought (CoT) techniques have significantly enhanced the reasoning capabilities of these models. However, due to the complex logic and multi-step reasoning ability needed, the application of CoT techniques in the APR domain remains insufficient. This study systematically evaluates the performance of several common CoT techniques in APR tasks and proposes an innovative framework $T^3$, which integrates the powerful reasoning capabilities of LLMs with tree search, effectively improving the precision of generating candidate repair solutions. Furthermore, $T^3$ provides valuable guidance for optimizing sample selection and repair strategies in APR tasks, establishing a robust framework for achieving efficient automated debugging. 

---
# IPFormer-VideoLLM: Enhancing Multi-modal Video Understanding for Multi-shot Scenes 

**Authors**: Yujia Liang, Jile Jiao, Zhicheng Wang, Xuetao Feng, Zixuan Ye, Yuan Wang, Hao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21116)  

**Abstract**: Video Large Language Models (VideoLLMs) have demonstrated remarkable understanding capabilities, but are found struggling to tackle multi-shot scenarios,e.g., video clips with varying camera angles or scene changes. This challenge can render failures such as instance identity forgetting and key frame negligence. In this work, we first attribute the challenge to the lack of multi-shot annotations among existing datasets and therefore we introduce a new dataset termed MultiClip-Bench, featuring dense descriptions and instruction-based question-answering pairs tailored for multi-shot scenarios. We empirically find that the training set significantly boosts the multi-shot performance, while the testing benchmark provides a reliable measure of the model capability in multi-shot scenarios. By further analyzing and discovering that current models only encode instance features in a discrete or lossy manner, at the risk of missing identity information, we then contribute a new model IPFormer-VideoLLM. Its key idea is the injection of instance-level features as instance prompts through an efficient attention-based connector. This allows for the aggregation of instance-specific information across scenes. Experiments demonstrate that our proposed dataset and model not only enhance the multi-scene video understanding significantly, but also offer distinct advantages across various video benchmarks. 

---
# DiLoCoX: A Low-Communication Large-Scale Training Framework for Decentralized Cluster 

**Authors**: Ji Qi, WenPeng Zhu, Li Li, Ming Wu, YingJun Wu, Wu He, Xun Gao, Jason Zeng, Michael Heinrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.21263)  

**Abstract**: The distributed training of foundation models, particularly large language models (LLMs), demands a high level of communication. Consequently, it is highly dependent on a centralized cluster with fast and reliable interconnects. Can we conduct training on slow networks and thereby unleash the power of decentralized clusters when dealing with models exceeding 100 billion parameters? In this paper, we propose DiLoCoX, a low-communication large-scale decentralized cluster training framework. It combines Pipeline Parallelism with Dual Optimizer Policy, One-Step-Delay Overlap of Communication and Local Training, and an Adaptive Gradient Compression Scheme. This combination significantly improves the scale of parameters and the speed of model pre-training. We justify the benefits of one-step-delay overlap of communication and local training, as well as the adaptive gradient compression scheme, through a theoretical analysis of convergence. Empirically, we demonstrate that DiLoCoX is capable of pre-training a 107B foundation model over a 1Gbps network. Compared to vanilla AllReduce, DiLoCoX can achieve a 357x speedup in distributed training while maintaining negligible degradation in model convergence. To the best of our knowledge, this is the first decentralized training framework successfully applied to models with over 100 billion parameters. 

---
# Large Language Models Acing Chartered Accountancy 

**Authors**: Jatin Gupta, Akhil Sharma, Saransh Singhania, Mohammad Adnan, Sakshi Deo, Ali Imam Abidi, Keshav Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.21031)  

**Abstract**: Advanced intelligent systems, particularly Large Language Models (LLMs), are significantly reshaping financial practices through advancements in Natural Language Processing (NLP). However, the extent to which these models effectively capture and apply domain-specific financial knowledge remains uncertain. Addressing a critical gap in the expansive Indian financial context, this paper introduces CA-Ben, a Chartered Accountancy benchmark specifically designed to evaluate the financial, legal, and quantitative reasoning capabilities of LLMs. CA-Ben comprises structured question-answer datasets derived from the rigorous examinations conducted by the Institute of Chartered Accountants of India (ICAI), spanning foundational, intermediate, and advanced CA curriculum stages. Six prominent LLMs i.e. GPT 4o, LLAMA 3.3 70B, LLAMA 3.1 405B, MISTRAL Large, Claude 3.5 Sonnet, and Microsoft Phi 4 were evaluated using standardized protocols. Results indicate variations in performance, with Claude 3.5 Sonnet and GPT-4o outperforming others, especially in conceptual and legal reasoning. Notable challenges emerged in numerical computations and legal interpretations. The findings emphasize the strengths and limitations of current LLMs, suggesting future improvements through hybrid reasoning and retrieval-augmented generation methods, particularly for quantitative analysis and accurate legal interpretation. 

---
# Multimodal Prompt Alignment for Facial Expression Recognition 

**Authors**: Fuyan Ma, Yiran He, Bin Sun, Shutao Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21017)  

**Abstract**: Prompt learning has been widely adopted to efficiently adapt vision-language models (VLMs) like CLIP for various downstream tasks. Despite their success, current VLM-based facial expression recognition (FER) methods struggle to capture fine-grained textual-visual relationships, which are essential for distinguishing subtle differences between facial expressions. To address this challenge, we propose a multimodal prompt alignment framework for FER, called MPA-FER, that provides fine-grained semantic guidance to the learning process of prompted visual features, resulting in more precise and interpretable representations. Specifically, we introduce a multi-granularity hard prompt generation strategy that utilizes a large language model (LLM) like ChatGPT to generate detailed descriptions for each facial expression. The LLM-based external knowledge is injected into the soft prompts by minimizing the feature discrepancy between the soft prompts and the hard prompts. To preserve the generalization abilities of the pretrained CLIP model, our approach incorporates prototype-guided visual feature alignment, ensuring that the prompted visual features from the frozen image encoder align closely with class-specific prototypes. Additionally, we propose a cross-modal global-local alignment module that focuses on expression-relevant facial features, further improving the alignment between textual and visual features. Extensive experiments demonstrate our framework outperforms state-of-the-art methods on three FER benchmark datasets, while retaining the benefits of the pretrained model and minimizing computational costs. 

---
# Evidence-based diagnostic reasoning with multi-agent copilot for human pathology 

**Authors**: Chengkuan Chen, Luca L. Weishaupt, Drew F. K. Williamson, Richard J. Chen, Tong Ding, Bowen Chen, Anurag Vaidya, Long Phi Le, Guillaume Jaume, Ming Y. Lu, Faisal Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2506.20964)  

**Abstract**: Pathology is experiencing rapid digital transformation driven by whole-slide imaging and artificial intelligence (AI). While deep learning-based computational pathology has achieved notable success, traditional models primarily focus on image analysis without integrating natural language instruction or rich, text-based context. Current multimodal large language models (MLLMs) in computational pathology face limitations, including insufficient training data, inadequate support and evaluation for multi-image understanding, and a lack of autonomous, diagnostic reasoning capabilities. To address these limitations, we introduce PathChat+, a new MLLM specifically designed for human pathology, trained on over 1 million diverse, pathology-specific instruction samples and nearly 5.5 million question answer turns. Extensive evaluations across diverse pathology benchmarks demonstrated that PathChat+ substantially outperforms the prior PathChat copilot, as well as both state-of-the-art (SOTA) general-purpose and other pathology-specific models. Furthermore, we present SlideSeek, a reasoning-enabled multi-agent AI system leveraging PathChat+ to autonomously evaluate gigapixel whole-slide images (WSIs) through iterative, hierarchical diagnostic reasoning, reaching high accuracy on DDxBench, a challenging open-ended differential diagnosis benchmark, while also capable of generating visually grounded, humanly-interpretable summary reports. 

---
# LLM-guided Chemical Process Optimization with a Multi-Agent Approach 

**Authors**: Tong Zeng, Srivathsan Badrinarayanan, Janghoon Ock, Cheng-Kai Lai, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2506.20921)  

**Abstract**: Chemical process optimization is crucial to maximize production efficiency and economic performance. Traditional methods, including gradient-based solvers, evolutionary algorithms, and parameter grid searches, become impractical when operating constraints are ill-defined or unavailable, requiring engineers to rely on subjective heuristics to estimate feasible parameter ranges. To address this constraint definition bottleneck, we present a multi-agent framework of large language model (LLM) agents that autonomously infer operating constraints from minimal process descriptions, then collaboratively guide optimization using the inferred constraints. Our AutoGen-based agentic framework employs OpenAI's o3 model, with specialized agents for constraint generation, parameter validation, simulation execution, and optimization guidance. Through two phases - autonomous constraint generation using embedded domain knowledge, followed by iterative multi-agent optimization - the framework eliminates the need for predefined operational bounds. Validated on the hydrodealkylation process across cost, yield, and yield-to-cost ratio metrics, the framework demonstrated competitive performance with conventional optimization methods while achieving better computational efficiency, requiring fewer iterations to converge. Our approach converged in under 20 minutes, achieving a 31-fold speedup over grid search. Beyond computational efficiency, the framework's reasoning-guided search demonstrates sophisticated process understanding, correctly identifying utility trade-offs, and applying domain-informed heuristics. This approach shows significant potential for optimization scenarios where operational constraints are poorly characterized or unavailable, particularly for emerging processes and retrofit applications. 

---
# ZKPROV: A Zero-Knowledge Approach to Dataset Provenance for Large Language Models 

**Authors**: Mina Namazi, Alexander Nemecek, Erman Ayday  

**Link**: [PDF](https://arxiv.org/pdf/2506.20915)  

**Abstract**: As the deployment of large language models (LLMs) grows in sensitive domains, ensuring the integrity of their computational provenance becomes a critical challenge, particularly in regulated sectors such as healthcare, where strict requirements are applied in dataset usage. We introduce ZKPROV, a novel cryptographic framework that enables zero-knowledge proofs of LLM provenance. It allows users to verify that a model is trained on a reliable dataset without revealing sensitive information about it or its parameters. Unlike prior approaches that focus on complete verification of the training process (incurring significant computational cost) or depend on trusted execution environments, ZKPROV offers a distinct balance. Our method cryptographically binds a trained model to its authorized training dataset(s) through zero-knowledge proofs while avoiding proof of every training step. By leveraging dataset-signed metadata and compact model parameter commitments, ZKPROV provides sound and privacy-preserving assurances that the result of the LLM is derived from a model trained on the claimed authorized and relevant dataset. Experimental results demonstrate the efficiency and scalability of the ZKPROV in generating this proof and verifying it, achieving a practical solution for real-world deployments. We also provide formal security guarantees, proving that our approach preserves dataset confidentiality while ensuring trustworthy dataset provenance. 

---
# Omniwise: Predicting GPU Kernels Performance with LLMs 

**Authors**: Zixian Wang, Cole Ramos, Muhammad A. Awad, Keith Lowery  

**Link**: [PDF](https://arxiv.org/pdf/2506.20886)  

**Abstract**: In recent years, the rapid advancement of deep neural networks (DNNs) has revolutionized artificial intelligence, enabling models with unprecedented capabilities in understanding, generating, and processing complex data. These powerful architectures have transformed a wide range of downstream applications, tackling tasks beyond human reach. In this paper, we introduce Omniwise, the first end-to-end, self-supervised fine-tuning pipeline that applies large language models (LLMs) to GPU kernel performance prediction--a novel use case in performance profiling. Omniwise is model-agnostic and lightweight, achieving strong results even with a small 3B-parameter model. It can predict key performance metrics, including memory bandwidth, cache hit rates, GFLOPs, and arithmetic intensity, directly from kernel code without the need for code execution or profiling tools. Our approach achieves over 90% of predictions within 10% relative error on GPU kernels executed on AMD MI250 and MI300X architectures. In addition to the pipeline, we develop an online inference server and a Visual Studio Code plugin that seamlessly integrate LLM-based performance prediction into developers' workflows. 

---
# Uncovering Hidden Violent Tendencies in LLMs: A Demographic Analysis via Behavioral Vignettes 

**Authors**: Quintin Myers, Yanjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.20822)  

**Abstract**: Large language models (LLMs) are increasingly proposed for detecting and responding to violent content online, yet their ability to reason about morally ambiguous, real-world scenarios remains underexamined. We present the first study to evaluate LLMs using a validated social science instrument designed to measure human response to everyday conflict, namely the Violent Behavior Vignette Questionnaire (VBVQ). To assess potential bias, we introduce persona-based prompting that varies race, age, and geographic identity within the United States. Six LLMs developed across different geopolitical and organizational contexts are evaluated under a unified zero-shot setting. Our study reveals two key findings: (1) LLMs surface-level text generation often diverges from their internal preference for violent responses; (2) their violent tendencies vary across demographics, frequently contradicting established findings in criminology, social science, and psychology. 

---
# GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization 

**Authors**: Martin Andrews, Sam Witteveen  

**Link**: [PDF](https://arxiv.org/pdf/2506.20807)  

**Abstract**: Optimizing GPU kernels for high performance is a complex task, often demanding deep architectural knowledge, extensive profiling, and iterative experimentation. This challenge is amplified when targeting newer or less-documented GPU architectures where traditional development aids are scarce. This paper introduces an LLM-powered "GPU Kernel Scientist," an automated methodology for iteratively refining accelerator kernels.
Our methodology employs LLMs in a multi-stage, evolutionary process: (a) strategically selecting promising prior code versions as a basis for new iterations; (b) generating hypotheses for optimization experiments, based on existing code and assimilated knowledge from general GPU literature; and (c) autonomously implementing these experiments through code modification and subsequent submission to an external evaluation system, using only observed timing data as performance feedback. We detail how this approach navigates the challenges of the AMD MI300 target architecture and leverages LLMs to compensate for limited domain-specific human expertise.
Since quantitative results from an ongoing performance competition were embargoed on paper submission date, we present the architectural design, operational workflow, and qualitative insights, highlighting the potential of LLM-driven agents to democratise and accelerate GPU kernel optimization, especially in resource-constrained or rapidly evolving hardware environments. 

---
# MultiFinRAG: An Optimized Multimodal Retrieval-Augmented Generation (RAG) Framework for Financial Question Answering 

**Authors**: Chinmay Gondhalekar, Urjitkumar Patel, Fang-Chun Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.20821)  

**Abstract**: Financial documents--such as 10-Ks, 10-Qs, and investor presentations--span hundreds of pages and combine diverse modalities, including dense narrative text, structured tables, and complex figures. Answering questions over such content often requires joint reasoning across modalities, which strains traditional large language models (LLMs) and retrieval-augmented generation (RAG) pipelines due to token limitations, layout loss, and fragmented cross-modal context. We introduce MultiFinRAG, a retrieval-augmented generation framework purpose-built for financial QA. MultiFinRAG first performs multimodal extraction by grouping table and figure images into batches and sending them to a lightweight, quantized open-source multimodal LLM, which produces both structured JSON outputs and concise textual summaries. These outputs, along with narrative text, are embedded and indexed with modality-aware similarity thresholds for precise retrieval. A tiered fallback strategy then dynamically escalates from text-only to text+table+image contexts when necessary, enabling cross-modal reasoning while reducing irrelevant context. Despite running on commodity hardware, MultiFinRAG achieves 19 percentage points higher accuracy than ChatGPT-4o (free-tier) on complex financial QA tasks involving text, tables, images, and combined multimodal reasoning. 

---
# Poster: Enhancing GNN Robustness for Network Intrusion Detection via Agent-based Analysis 

**Authors**: Zhonghao Zhan, Huichi Zhou, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20806)  

**Abstract**: Graph Neural Networks (GNNs) show great promise for Network Intrusion Detection Systems (NIDS), particularly in IoT environments, but suffer performance degradation due to distribution drift and lack robustness against realistic adversarial attacks. Current robustness evaluations often rely on unrealistic synthetic perturbations and lack demonstrations on systematic analysis of different kinds of adversarial attack, which encompass both black-box and white-box scenarios. This work proposes a novel approach to enhance GNN robustness and generalization by employing Large Language Models (LLMs) in an agentic pipeline as simulated cybersecurity expert agents. These agents scrutinize graph structures derived from network flow data, identifying and potentially mitigating suspicious or adversarially perturbed elements before GNN processing. Our experiments, using a framework designed for realistic evaluation and testing with a variety of adversarial attacks including a dataset collected from physical testbed experiments, demonstrate that integrating LLM analysis can significantly improve the resilience of GNN-based NIDS against challenges, showcasing the potential of LLM agent as a complementary layer in intrusion detection architectures. 

---
# TopK Language Models 

**Authors**: Ryosuke Takahashi, Tatsuro Inaba, Kentaro Inui, Benjamin Heinzerling  

**Link**: [PDF](https://arxiv.org/pdf/2506.21468)  

**Abstract**: Sparse autoencoders (SAEs) have become an important tool for analyzing and interpreting the activation space of transformer-based language models (LMs). However, SAEs suffer several shortcomings that diminish their utility and internal validity. Since SAEs are trained post-hoc, it is unclear if the failure to discover a particular concept is a failure on the SAE's side or due to the underlying LM not representing this concept. This problem is exacerbated by training conditions and architecture choices affecting which features an SAE learns. When tracing how LMs learn concepts during training, the lack of feature stability also makes it difficult to compare SAEs features across different checkpoints. To address these limitations, we introduce a modification to the transformer architecture that incorporates a TopK activation function at chosen layers, making the model's hidden states equivalent to the latent features of a TopK SAE. This approach eliminates the need for post-hoc training while providing interpretability comparable to SAEs. The resulting TopK LMs offer a favorable trade-off between model size, computational efficiency, and interpretability. Despite this simple architectural change, TopK LMs maintain their original capabilities while providing robust interpretability benefits. Our experiments demonstrate that the sparse representations learned by TopK LMs enable successful steering through targeted neuron interventions and facilitate detailed analysis of neuron formation processes across checkpoints and layers. These features make TopK LMs stable and reliable tools for understanding how language models learn and represent concepts, which we believe will significantly advance future research on model interpretability and controllability. 

---
# Text2Cypher Across Languages: Evaluating Foundational Models Beyond English 

**Authors**: Makbule Gulcin Ozsoy, William Tai  

**Link**: [PDF](https://arxiv.org/pdf/2506.21445)  

**Abstract**: Recent advances in large language models have enabled natural language interfaces that translate user questions into database queries, such as Text2SQL, Text2SPARQL, and Text2Cypher. While these interfaces enhance database accessibility, most research today focuses solely on English, with limited evaluation in other languages. This paper investigates the performance of foundational LLMs on the Text2Cypher task across multiple languages. We create and release a multilingual test set by translating English questions into Spanish and Turkish while preserving the original Cypher queries, enabling fair cross-lingual comparison. We evaluate multiple foundational models using standardized prompts and metrics. Our results show a consistent performance pattern: highest on English, then Spanish, and lowest on Turkish. We attribute this to differences in training data availability and linguistic characteristics. Additionally, we explore the impact of translating task prompts into Spanish and Turkish. Results show little to no change in evaluation metrics, suggesting prompt translation has minor impact. Our findings highlight the need for more inclusive evaluation and development in multilingual query generation. Future work includes schema localization and fine-tuning across diverse languages. 

---
# Double-Checker: Enhancing Reasoning of Slow-Thinking LLMs via Self-Critical Fine-Tuning 

**Authors**: Xin Xu, Tianhao Chen, Fan Zhang, Wanlong Liu, Pengxiang Li, Ajay Kumar Jaiswal, Yuchen Yan, Jishan Hu, Yang Wang, Hao Chen, Shiwei Liu, Shizhe Diao, Can Yang, Lu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.21285)  

**Abstract**: While slow-thinking large language models (LLMs) exhibit reflection-like reasoning, commonly referred to as the "aha moment:, their ability to generate informative critiques and refine prior solutions remains limited. In this paper, we introduce Double-Checker, a principled framework designed to enhance the reasoning capabilities of slow-thinking LLMs by fostering explicit self-critique and iterative refinement of their previous solutions. By fine-tuning on our curated 1,730 self-critical instances, Double-Checker empowers long-CoT LLMs to iteratively critique and refine their outputs during inference until they evaluate their solutions as correct under self-generated critiques. We validate the efficacy of Double-Checker across a comprehensive suite of reasoning benchmarks, demonstrating that iterative self-critique significantly enhances the reasoning capabilities of long-CoT LLMs. Notably, our Double-Checker increases the pass@1 performance on challenging AIME benchmarks from 4.4% to 18.2% compared to the original long-CoT LLMs. These results highlight a promising direction for developing more trustworthy and effective LLMs capable of structured self-critique. 

---
# Cat and Mouse -- Can Fake Text Generation Outpace Detector Systems? 

**Authors**: Andrea McGlinchey, Peter J Barclay  

**Link**: [PDF](https://arxiv.org/pdf/2506.21274)  

**Abstract**: Large language models can produce convincing "fake text" in domains such as academic writing, product reviews, and political news. Many approaches have been investigated for the detection of artificially generated text. While this may seem to presage an endless "arms race", we note that newer LLMs use ever more parameters, training data, and energy, while relatively simple classifiers demonstrate a good level of detection accuracy with modest resources. To approach the question of whether the models' ability to beat the detectors may therefore reach a plateau, we examine the ability of statistical classifiers to identify "fake text" in the style of classical detective fiction. Over a 0.5 version increase, we found that Gemini showed an increased ability to generate deceptive text, while GPT did not. This suggests that reliable detection of fake text may remain feasible even for ever-larger models, though new model architectures may improve their deceptiveness 

---
# Structuralist Approach to AI Literary Criticism: Leveraging Greimas Semiotic Square for Large Language Models 

**Authors**: Fangzhou Dong, Yifan Zeng, Yingpeng Sang, Hong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21360)  

**Abstract**: Large Language Models (LLMs) excel in understanding and generating text but struggle with providing professional literary criticism for works with profound thoughts and complex narratives. This paper proposes GLASS (Greimas Literary Analysis via Semiotic Square), a structured analytical framework based on Greimas Semiotic Square (GSS), to enhance LLMs' ability to conduct in-depth literary analysis. GLASS facilitates the rapid dissection of narrative structures and deep meanings in narrative works. We propose the first dataset for GSS-based literary criticism, featuring detailed analyses of 48 works. Then we propose quantitative metrics for GSS-based literary criticism using the LLM-as-a-judge paradigm. Our framework's results, compared with expert criticism across multiple works and LLMs, show high performance. Finally, we applied GLASS to 39 classic works, producing original and high-quality analyses that address existing research gaps. This research provides an AI-based tool for literary research and education, offering insights into the cognitive mechanisms underlying literary engagement. 

---
# Prompt-Guided Turn-Taking Prediction 

**Authors**: Koji Inoue, Mikey Elmers, Yahui Fu, Zi Haur Pang, Divesh Lala, Keiko Ochi, Tatsuya Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2506.21191)  

**Abstract**: Turn-taking prediction models are essential components in spoken dialogue systems and conversational robots. Recent approaches leverage transformer-based architectures to predict speech activity continuously and in real-time. In this study, we propose a novel model that enables turn-taking prediction to be dynamically controlled via textual prompts. This approach allows intuitive and explicit control through instructions such as "faster" or "calmer" adapting dynamically to conversational partners and contexts. The proposed model builds upon a transformer-based voice activity projection (VAP) model, incorporating textual prompt embeddings into both channel-wise transformers and a cross-channel transformer. We evaluated the feasibility of our approach using over 950 hours of human-human spoken dialogue data. Since textual prompt data for the proposed approach was not available in existing datasets, we utilized a large language model (LLM) to generate synthetic prompt sentences. Experimental results demonstrated that the proposed model improved prediction accuracy and effectively varied turn-taking timing behaviors according to the textual prompts. 

---
# SAC: A Framework for Measuring and Inducing Personality Traits in LLMs with Dynamic Intensity Control 

**Authors**: Adithya Chittem, Aishna Shrivastava, Sai Tarun Pendela, Jagat Sesh Challa, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.20993)  

**Abstract**: Large language models (LLMs) have gained significant traction across a wide range of fields in recent years. There is also a growing expectation for them to display human-like personalities during interactions. To meet this expectation, numerous studies have proposed methods for modelling LLM personalities through psychometric evaluations. However, most existing models face two major limitations: they rely on the Big Five (OCEAN) framework, which only provides coarse personality dimensions, and they lack mechanisms for controlling trait intensity. In this paper, we address this gap by extending the Machine Personality Inventory (MPI), which originally used the Big Five model, to incorporate the 16 Personality Factor (16PF) model, allowing expressive control over sixteen distinct traits. We also developed a structured framework known as Specific Attribute Control (SAC) for evaluating and dynamically inducing trait intensity in LLMs. Our method introduces adjective-based semantic anchoring to guide trait intensity expression and leverages behavioural questions across five intensity factors: \textit{Frequency}, \textit{Depth}, \textit{Threshold}, \textit{Effort}, and \textit{Willingness}. Through experimentation, we find that modelling intensity as a continuous spectrum yields substantially more consistent and controllable personality expression compared to binary trait toggling. Moreover, we observe that changes in target trait intensity systematically influence closely related traits in psychologically coherent directions, suggesting that LLMs internalize multi-dimensional personality structures rather than treating traits in isolation. Our work opens new pathways for controlled and nuanced human-machine interactions in domains such as healthcare, education, and interviewing processes, bringing us one step closer to truly human-like social machines. 

---
# Enhancing User Engagement in Socially-Driven Dialogue through Interactive LLM Alignments 

**Authors**: Jiashuo Wang, Kaitao Song, Chunpu Xu, Changhe Song, Yang Xiao, Dongsheng Li, Lili Qiu, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21497)  

**Abstract**: Enhancing user engagement through interactions plays an essential role in socially-driven dialogues. While prior works have optimized models to reason over relevant knowledge or plan a dialogue act flow, the relationship between user engagement and knowledge or dialogue acts is subtle and does not guarantee user engagement in socially-driven dialogues. To this end, we enable interactive LLMs to learn user engagement by leveraging signals from the future development of conversations. Specifically, we adopt a more direct and relevant indicator of user engagement, i.e., the user's reaction related to dialogue intention after the interaction, as a reward to align interactive LLMs. To achieve this, we develop a user simulator to interact with target interactive LLMs and explore interactions between the user and the interactive LLM system via \textit{i$\times$MCTS} (\textit{M}onte \textit{C}arlo \textit{T}ree \textit{S}earch for \textit{i}nteraction). In this way, we collect a dataset containing pairs of higher and lower-quality experiences using \textit{i$\times$MCTS}, and align interactive LLMs for high-level user engagement by direct preference optimization (DPO) accordingly. Experiments conducted on two socially-driven dialogue scenarios (emotional support conversations and persuasion for good) demonstrate that our method effectively enhances user engagement in interactive LLMs. 

---
# Enhancing Automatic Term Extraction with Large Language Models via Syntactic Retrieval 

**Authors**: Yongchan Chun, Minhyuk Kim, Dongjun Kim, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2506.21222)  

**Abstract**: Automatic Term Extraction (ATE) identifies domain-specific expressions that are crucial for downstream tasks such as machine translation and information retrieval. Although large language models (LLMs) have significantly advanced various NLP tasks, their potential for ATE has scarcely been examined. We propose a retrieval-based prompting strategy that, in the few-shot setting, selects demonstrations according to \emph{syntactic} rather than semantic similarity. This syntactic retrieval method is domain-agnostic and provides more reliable guidance for capturing term boundaries. We evaluate the approach in both in-domain and cross-domain settings, analyzing how lexical overlap between the query sentence and its retrieved examples affects performance. Experiments on three specialized ATE benchmarks show that syntactic retrieval improves F1-score. These findings highlight the importance of syntactic cues when adapting LLMs to terminology-extraction tasks. 

---
# The Ideation-Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas 

**Authors**: Chenglei Si, Tatsunori Hashimoto, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20803)  

**Abstract**: Large Language Models (LLMs) have shown promise in accelerating the scientific research pipeline. A key capability for this process is the ability to generate novel research ideas, and prior studies have found settings in which LLM-generated research ideas were judged as more novel than human-expert ideas. However, a good idea should not simply appear to be novel, it should also result in better research after being executed. To test whether AI-generated ideas lead to better research outcomes, we conduct an execution study by recruiting 43 expert researchers to execute randomly-assigned ideas, either written by experts or generated by an LLM. Each expert spent over 100 hours implementing the idea and wrote a 4-page short paper to document the experiments. All the executed projects are then reviewed blindly by expert NLP researchers. Comparing the review scores of the same ideas before and after execution, the scores of the LLM-generated ideas decrease significantly more than expert-written ideas on all evaluation metrics (novelty, excitement, effectiveness, and overall; p < 0.05), closing the gap between LLM and human ideas observed at the ideation stage. When comparing the aggregated review scores from the execution study, we even observe that for many metrics there is a flip in rankings where human ideas score higher than LLM ideas. This ideation-execution gap highlights the limitations of current LLMs in generating truly effective research ideas and the challenge of evaluating research ideas in the absence of execution outcomes. 

---
# MT2-CSD: A New Dataset and Multi-Semantic Knowledge Fusion Method for Conversational Stance Detection 

**Authors**: Fuqiang Niu, Genan Dai, Yisha Lu, Jiayu Liao, Xiang Li, Hu Huang, Bowen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21053)  

**Abstract**: In the realm of contemporary social media, automatic stance detection is pivotal for opinion mining, as it synthesizes and examines user perspectives on contentious topics to uncover prevailing trends and sentiments. Traditional stance detection research often targets individual instances, thereby limiting its capacity to model multi-party discussions typical in real social media scenarios. This shortcoming largely stems from the scarcity of datasets that authentically capture the dynamics of social media interactions, hindering advancements in conversational stance detection. In this paper, we introduce MT2-CSD, a comprehensive dataset for multi-target, multi-turn conversational stance detection. To the best of our knowledge, MT2-CSD is the largest dataset available for this purpose, comprising 24,457 annotated instances and exhibiting the greatest conversational depth, thereby presenting new challenges for stance detection. To address these challenges, we propose the Large Language model enhanced Conversational Relational Attention Network (LLM-CRAN), which exploits the reasoning capabilities of LLMs to improve conversational understanding. We conduct extensive experiments to evaluate the efficacy of LLM-CRAN on the MT2-CSD dataset. The experimental results indicate that LLM-CRAN significantly outperforms strong baseline models in the task of conversational stance detection. 

---
# Towards Probabilistic Question Answering Over Tabular Data 

**Authors**: Chen Shen, Sajjadur Rahman, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2506.20747)  

**Abstract**: Current approaches for question answering (QA) over tabular data, such as NL2SQL systems, perform well for factual questions where answers are directly retrieved from tables. However, they fall short on probabilistic questions requiring reasoning under uncertainty. In this paper, we introduce a new benchmark LUCARIO and a framework for probabilistic QA over large tabular data. Our method induces Bayesian Networks from tables, translates natural language queries into probabilistic queries, and uses large language models (LLMs) to generate final answers. Empirical results demonstrate significant improvements over baselines, highlighting the benefits of hybrid symbolic-neural reasoning. 

---
# Multi-lingual Functional Evaluation for Large Language Models 

**Authors**: Victor Ojewale, Inioluwa Deborah Raji, Suresh Venkatasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.20793)  

**Abstract**: Multi-lingual competence in large language models is often evaluated via static data benchmarks such as Belebele, M-MMLU and M-GSM. However, these evaluations often fail to provide an adequate understanding of the practical performance and robustness of models across multi-lingual settings. In response, we create multi-lingual functional benchmarks -- Cross-Lingual Grade School Math Symbolic (CL-GSM Symbolic) and Cross-Lingual Instruction-Following Eval (CL-IFEval)-- by translating existing functional benchmark templates from English to five additional languages that span the range of resources available for NLP: French, Spanish, Hindi, Arabic and Yoruba. Our results reveal that some static multi-lingual benchmarks capture functional performance much more closely than others (i.e. across models, there is a 24%, 17% and 18% decrease in performance between M-GSM and CL-GSM Symbolic in English, French and Spanish respectively; similarly there's a 15 - 24% performance drop across languages between Belebele and CL-IFEval, and only a 0.5% to 3% performance drop between M-MMLU and CL-IFEval). Similarly, we find that model robustness across languages varies significantly, with certain languages (eg. Arabic, English) being the most consistently well performing across evaluation iterations. 

---
# HumanOmniV2: From Understanding to Omni-Modal Reasoning with Context 

**Authors**: Qize Yang, Shimin Yao, Weixuan Chen, Shenghao Fu, Detao Bai, Jiaxing Zhao, Boyuan Sun, Bowen Yin, Xihan Wei, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.21277)  

**Abstract**: With the rapid evolution of multimodal large language models, the capacity to deeply understand and interpret human intentions has emerged as a critical capability, which demands detailed and thoughtful reasoning. In recent studies, Reinforcement Learning (RL) has demonstrated potential in enhancing the reasoning capabilities of Large Language Models (LLMs). Nonetheless, the challenges associated with adapting RL to multimodal data and formats remain largely unaddressed. In this paper, we identify two issues in existing multimodal reasoning models: insufficient global context understanding and shortcut problems. Insufficient context understanding can happen when a model misinterprets multimodal context, resulting in incorrect answers. The shortcut problem occurs when the model overlooks crucial clues in multimodal inputs, directly addressing the query without considering the multimodal information. To tackle these issues, we emphasize the necessity for the model to reason with a clear understanding of the global context within multimodal inputs. This global context understanding can effectively prevent the model from overlooking key multimodal cues and ensure a thorough reasoning process. To ensure the accurate interpretation of multimodal context information, we implement a context reward judged by a large language model, alongside format and accuracy rewards. Additionally, to improve complex reasoning capability, we employ the LLM to assess the logical reward, determining whether the reasoning process successfully integrates multimodal information with logical methods. We also introduce a reasoning omni-modal benchmark, IntentBench, aimed at evaluating models in understanding complex human intentions and emotions. Our proposed method demonstrates advanced performance across multiple omni-modal benchmarks compared to other open-source omni-modal models. 

---
# Complexity-aware fine-tuning 

**Authors**: Andrey Goncharov, Daniil Vyazhev, Petr Sychev, Edvard Khalafyan, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2506.21220)  

**Abstract**: General-purpose Large Language Models (LLMs) are frequently fine-tuned through supervised fine-tuning (SFT) to enhance performance in specific domains. Better results can be achieved by distilling the chain-of-thought of a larger model at the cost of numerous expensive calls and a much greater amount of data. We propose a novel blueprint for efficient fine-tuning that uses reasoning only for complex data identified by entropy. Specifically, across two small open models ($\approx 3B$) we split the training data into complexity categories by a single token answer entropy (ROC AUC $0.73$), fine-tune large language models (LLMs) via SFT and distillation, and show that our pipeline significantly outperforms the standard SFT approach ($0.55$ vs $0.43$ average accuracy) and provides comparable with distillation performance while using $62\%$ less data ($0.55$ average accuracy for both). We publish our code and data to facilitate further research in this direction. 

---
# Can Gradient Descent Simulate Prompting? 

**Authors**: Eric Zhang, Leshem Choshen, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2506.20989)  

**Abstract**: There are two primary ways of incorporating new information into a language model (LM): changing its prompt or changing its parameters, e.g. via fine-tuning. Parameter updates incur no long-term storage cost for model changes. However, for many model updates, prompting is significantly more effective: prompted models can generalize robustly from single examples and draw logical inferences that do not occur under standard fine-tuning. Can models be modified so that fine-tuning does emulate prompting? This paper describes a method for meta-training LMs such that gradient updates emulate the effects of conditioning on new information. Our approach uses tools from gradient-based meta-learning but uses an LM's own prompted predictions as targets, eliminating the need for ground-truth labels. Subsequent gradient descent training recovers some (and occasionally all) of prompted model performance -- showing improvement on the ``reversal curse'' tasks, and answering questions about text passages after a single gradient update. These results suggest that, with appropriate initialization, gradient descent can be surprisingly expressive. Our results suggest new avenues for long-context modeling and offer insight into the generalization capabilities of gradient-based learning. 

---
# Enhancing LLM Tool Use with High-quality Instruction Data from Knowledge Graph 

**Authors**: Jingwei Wang, Zai Zhang, Hao Qian, Chunjing Gan, Binbin Hu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou, Bin Shi, Bo Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.21071)  

**Abstract**: Teaching large language models (LLMs) to use tools is crucial for improving their problem-solving abilities and expanding their applications. However, effectively using tools is challenging because it requires a deep understanding of tool functionalities and user intentions. Previous methods relied mainly on LLMs to generate instruction data, but the quality of these data was often insufficient. In this paper, we propose a new method that uses knowledge graphs to generate high-quality instruction data for LLMs. Knowledge graphs are manually curated datasets rich in semantic information. We begin by extracting various query pathways from a given knowledge graph, which are transformed into a broad spectrum of user queries. We then translate the relationships between entities into actionable tools and parse the pathways of each query into detailed solution steps, thereby creating high-quality instruction data. Our experiments show that fine-tuning on just a small sample of this synthetic data can significantly improve the tool utilization and overall capabilities of LLMs. 

---
# Response Quality Assessment for Retrieval-Augmented Generation via Conditional Conformal Factuality 

**Authors**: Naihe Feng, Yi Sui, Shiyi Hou, Jesse C. Cresswell, Ga Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20978)  

**Abstract**: Existing research on Retrieval-Augmented Generation (RAG) primarily focuses on improving overall question-answering accuracy, often overlooking the quality of sub-claims within generated responses. Recent methods that attempt to improve RAG trustworthiness, such as through auto-evaluation metrics, lack probabilistic guarantees or require ground truth answers. To address these limitations, we propose Conformal-RAG, a novel framework inspired by recent applications of conformal prediction (CP) on large language models (LLMs). Conformal-RAG leverages CP and internal information from the RAG mechanism to offer statistical guarantees on response quality. It ensures group-conditional coverage spanning multiple sub-domains without requiring manual labelling of conformal sets, making it suitable for complex RAG applications. Compared to existing RAG auto-evaluation methods, Conformal-RAG offers statistical guarantees on the quality of refined sub-claims, ensuring response reliability without the need for ground truth answers. Additionally, our experiments demonstrate that by leveraging information from the RAG system, Conformal-RAG retains up to 60\% more high-quality sub-claims from the response compared to direct applications of CP to LLMs, while maintaining the same reliability guarantee. 

---
# EraRAG: Efficient and Incremental Retrieval Augmented Generation for Growing Corpora 

**Authors**: Fangyuan Zhang, Zhengjun Huang, Yingli Zhou, Qintian Guo, Zhixun Li, Wensheng Luo, Di Jiang, Yixiang Fang, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.20963)  

**Abstract**: Graph-based Retrieval-Augmented Generation (Graph-RAG) enhances large language models (LLMs) by structuring retrieval over an external corpus. However, existing approaches typically assume a static corpus, requiring expensive full-graph reconstruction whenever new documents arrive, limiting their scalability in dynamic, evolving environments. To address these limitations, we introduce EraRAG, a novel multi-layered Graph-RAG framework that supports efficient and scalable dynamic updates. Our method leverages hyperplane-based Locality-Sensitive Hashing (LSH) to partition and organize the original corpus into hierarchical graph structures, enabling efficient and localized insertions of new data without disrupting the existing topology. The design eliminates the need for retraining or costly recomputation while preserving high retrieval accuracy and low latency. Experiments on large-scale benchmarks demonstrate that EraRag achieves up to an order of magnitude reduction in update time and token consumption compared to existing Graph-RAG systems, while providing superior accuracy performance. This work offers a practical path forward for RAG systems that must operate over continually growing corpora, bridging the gap between retrieval efficiency and adaptability. Our code and data are available at this https URL. 

---
# Metadata Enrichment of Long Text Documents using Large Language Models 

**Authors**: Manika Lamba, You Peng, Sophie Nikolov, Glen Layne-Worthey, J. Stephen Downie  

**Link**: [PDF](https://arxiv.org/pdf/2506.20918)  

**Abstract**: In this project, we semantically enriched and enhanced the metadata of long text documents, theses and dissertations, retrieved from the HathiTrust Digital Library in English published from 1920 to 2020 through a combination of manual efforts and large language models. This dataset provides a valuable resource for advancing research in areas such as computational social science, digital humanities, and information science. Our paper shows that enriching metadata using LLMs is particularly beneficial for digital repositories by introducing additional metadata access points that may not have originally been foreseen to accommodate various content types. This approach is particularly effective for repositories that have significant missing data in their existing metadata fields, enhancing search results and improving the accessibility of the digital repository. 

---
