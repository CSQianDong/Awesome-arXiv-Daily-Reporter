# A Medical Multimodal Diagnostic Framework Integrating Vision-Language Models and Logic Tree Reasoning 

**Authors**: Zelin Zang, Wenyi Gu, Siqi Ma, Dan Yang, Yue Shen, Zhu Zhang, Guohui Fan, Wing-Kuen Ling, Fuji Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21583)  

**Abstract**: With the rapid growth of large language models (LLMs) and vision-language models (VLMs) in medicine, simply integrating clinical text and medical imaging does not guarantee reliable reasoning. Existing multimodal models often produce hallucinations or inconsistent chains of thought, limiting clinical trust. We propose a diagnostic framework built upon LLaVA that combines vision-language alignment with logic-regularized reasoning. The system includes an input encoder for text and images, a projection module for cross-modal alignment, a reasoning controller that decomposes diagnostic tasks into steps, and a logic tree generator that assembles stepwise premises into verifiable conclusions. Evaluations on MedXpertQA and other benchmarks show that our method improves diagnostic accuracy and yields more interpretable reasoning traces on multimodal tasks, while remaining competitive on text-only settings. These results suggest a promising step toward trustworthy multimodal medical AI. 

---
# NEMO-4-PAYPAL: Leveraging NVIDIA's Nemo Framework for empowering PayPal's Commerce Agent 

**Authors**: Ali Sahami, Sudhanshu Garg, Andrew Wang, Chaitanya Kulkarni, Farhad Farahani, Sean Yun-Shiuan Chuang, Jian Wan, Srinivasan Manoharan, Uma Kona, Nitin Sharma, Linsey Pang, Prakhar Mehrotra, Jessica Clark, Mark Moyou  

**Link**: [PDF](https://arxiv.org/pdf/2512.21578)  

**Abstract**: We present the development and optimization of PayPal's Commerce Agent, powered by NEMO-4-PAYPAL, a multi-agent system designed to revolutionize agentic commerce on the PayPal platform. Through our strategic partnership with NVIDIA, we leveraged the NeMo Framework for LLM model fine-tuning to enhance agent performance. Specifically, we optimized the Search and Discovery agent by replacing our base model with a fine-tuned Nemotron small language model (SLM).
We conducted comprehensive experiments using the llama3.1-nemotron-nano-8B-v1 architecture, training LoRA-based models through systematic hyperparameter sweeps across learning rates, optimizers (Adam, AdamW), cosine annealing schedules, and LoRA ranks. Our contributions include: (1) the first application of NVIDIA's NeMo Framework to commerce-specific agent optimization, (2) LLM powered fine-tuning strategy for retrieval-focused commerce tasks, (3) demonstration of significant improvements in latency and cost while maintaining agent quality, and (4) a scalable framework for multi-agent system optimization in production e-commerce environments. Our results demonstrate that the fine-tuned Nemotron SLM effectively resolves the key performance issue in the retrieval component, which represents over 50\% of total agent response time, while maintaining or enhancing overall system performance. 

---
# Towards Responsible and Explainable AI Agents with Consensus-Driven Reasoning 

**Authors**: Eranga Bandara, Tharaka Hewa, Ross Gore, Sachin Shetty, Ravi Mukkamala, Peter Foytik, Abdul Rahman, Safdar H. Bouk, Xueping Liang, Amin Hass, Sachini Rajapakse, Ng Wee Keong, Kasun De Zoysa, Aruna Withanage, Nilaan Loganathan  

**Link**: [PDF](https://arxiv.org/pdf/2512.21699)  

**Abstract**: Agentic AI represents a major shift in how autonomous systems reason, plan, and execute multi-step tasks through the coordination of Large Language Models (LLMs), Vision Language Models (VLMs), tools, and external services. While these systems enable powerful new capabilities, increasing autonomy introduces critical challenges related to explainability, accountability, robustness, and governance, especially when agent outputs influence downstream actions or decisions. Existing agentic AI implementations often emphasize functionality and scalability, yet provide limited mechanisms for understanding decision rationale or enforcing responsibility across agent interactions. This paper presents a Responsible(RAI) and Explainable(XAI) AI Agent Architecture for production-grade agentic workflows based on multi-model consensus and reasoning-layer governance. In the proposed design, a consortium of heterogeneous LLM and VLM agents independently generates candidate outputs from a shared input context, explicitly exposing uncertainty, disagreement, and alternative interpretations. A dedicated reasoning agent then performs structured consolidation across these outputs, enforcing safety and policy constraints, mitigating hallucinations and bias, and producing auditable, evidence-backed decisions. Explainability is achieved through explicit cross-model comparison and preserved intermediate outputs, while responsibility is enforced through centralized reasoning-layer control and agent-level constraints. We evaluate the architecture across multiple real-world agentic AI workflows, demonstrating that consensus-driven reasoning improves robustness, transparency, and operational trust across diverse application domains. This work provides practical guidance for designing agentic AI systems that are autonomous and scalable, yet responsible and explainable by construction. 

---
# AMS-IO-Bench and AMS-IO-Agent: Benchmarking and Structured Reasoning for Analog and Mixed-Signal Integrated Circuit Input/Output Design 

**Authors**: Zhishuai Zhang, Xintian Li, Shilong Liu, Aodong Zhang, Lu Jie, Nan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.21613)  

**Abstract**: In this paper, we propose AMS-IO-Agent, a domain-specialized LLM-based agent for structure-aware input/output (I/O) subsystem generation in analog and mixed-signal (AMS) integrated circuits (ICs). The central contribution of this work is a framework that connects natural language design intent with industrial-level AMS IC design deliverables. AMS-IO-Agent integrates two key capabilities: (1) a structured domain knowledge base that captures reusable constraints and design conventions; (2) design intent structuring, which converts ambiguous user intent into verifiable logic steps using JSON and Python as intermediate formats. We further introduce AMS-IO-Bench, a benchmark for wirebond-packaged AMS I/O ring automation. On this benchmark, AMS-IO-Agent achieves over 70\% DRC+LVS pass rate and reduces design turnaround time from hours to minutes, outperforming the baseline LLM. Furthermore, an agent-generated I/O ring was fabricated and validated in a 28 nm CMOS tape-out, demonstrating the practical effectiveness of the approach in real AMS IC design flows. To our knowledge, this is the first reported human-agent collaborative AMS IC design in which an LLM-based agent completes a nontrivial subtask with outputs directly used in silicon. 

---
# Leash: Adaptive Length Penalty and Reward Shaping for Efficient Large Reasoning Model 

**Authors**: Yanhao Li, Lu Ma, Jiaran Zhang, Lexiang Tang, Wentao Zhang, Guibo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2512.21540)  

**Abstract**: Existing approaches typically rely on fixed length penalties, but such penalties are hard to tune and fail to adapt to the evolving reasoning abilities of LLMs, leading to suboptimal trade-offs between accuracy and conciseness. To address this challenge, we propose Leash (adaptive LEngth penAlty and reward SHaping), a reinforcement learning framework for efficient reasoning in LLMs. We formulate length control as a constrained optimization problem and employ a Lagrangian primal-dual method to dynamically adjust the penalty coefficient. When generations exceed the target length, the penalty is intensified; when they are shorter, it is relaxed. This adaptive mechanism guides models toward producing concise reasoning without sacrificing task performance. Experiments on Deepseek-R1-Distill-Qwen-1.5B and Qwen3-4B-Thinking-2507 show that Leash reduces the average reasoning length by 60% across diverse tasks - including in-distribution mathematical reasoning and out-of-distribution domains such as coding and instruction following - while maintaining competitive performance. Our work thus presents a practical and effective paradigm for developing controllable and efficient LLMs that balance reasoning capabilities with computational budgets. 

---
# From Visual Perception to Deep Empathy: An Automated Assessment Framework for House-Tree-Person Drawings Using Multimodal LLMs and Multi-Agent Collaboration 

**Authors**: Shuide Wen, Yu Sun, Beier Ku, Zhi Gao, Lijun Ma, Yang Yang, Can Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2512.21360)  

**Abstract**: Background: The House-Tree-Person (HTP) drawing test, introduced by John Buck in 1948, remains a widely used projective technique in clinical psychology. However, it has long faced challenges such as heterogeneous scoring standards, reliance on examiners subjective experience, and a lack of a unified quantitative coding system.
Results: Quantitative experiments showed that the mean semantic similarity between Multimodal Large Language Model (MLLM) interpretations and human expert interpretations was approximately 0.75 (standard deviation about 0.05). In structurally oriented expert data sets, this similarity rose to 0.85, indicating expert-level baseline comprehension. Qualitative analyses demonstrated that the multi-agent system, by integrating social-psychological perspectives and destigmatizing narratives, effectively corrected visual hallucinations and produced psychological reports with high ecological validity and internal coherence.
Conclusions: The findings confirm the potential of multimodal large models as standardized tools for projective assessment. The proposed multi-agent framework, by dividing roles, decouples feature recognition from psychological inference and offers a new paradigm for digital mental-health services.
Keywords: House-Tree-Person test; multimodal large language model; multi-agent collaboration; cosine similarity; computational psychology; artificial intelligence 

---
# Agentic Structured Graph Traversal for Root Cause Analysis of Code-related Incidents in Cloud Applications 

**Authors**: Shengkun Cui, Rahul Krishna, Saurabh Jha, Ravishankar K. Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2512.22113)  

**Abstract**: Cloud incidents pose major operational challenges in production, with unresolved production cloud incidents cost on average over $2M per hour. Prior research identifies code- and configuration-related issues as the predominant category of root causes in cloud incidents. This paper introduces PRAXIS, an orchestrator that manages and deploys an agentic workflow for diagnosing code- and configuration-caused cloud incidents. PRAXIS employs an LLM-driven structured traversal over two types of graph: (1) a service dependency graph (SDG) that captures microservice-level dependencies; and (2) a hammock-block program dependence graph (PDG) that captures code-level dependencies for each microservice. Together, these graphs encode microservice- and code-level dependencies and the LLM acts as a traversal policy over these graphs, moving between services and code dependencies to localize and explain failures. Compared to state-of-the-art ReAct baselines, PRAXIS improves RCA accuracy by up to 3.1x while reducing token consumption by 3.8x. PRAXIS is demonstrated on a set of 30 comprehensive real-world incidents that is being compiled into an RCA benchmark. 

---
# Accelerating Scientific Discovery with Autonomous Goal-evolving Agents 

**Authors**: Yuanqi Du, Botao Yu, Tianyu Liu, Tony Shen, Junwu Chen, Jan G. Rittig, Kunyang Sun, Yikun Zhang, Zhangde Song, Bo Zhou, Cassandra Masschelein, Yingze Wang, Haorui Wang, Haojun Jia, Chao Zhang, Hongyu Zhao, Martin Ester, Teresa Head-Gordon, Carla P. Gomes, Huan Sun, Chenru Duan, Philippe Schwaller, Wengong Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.21782)  

**Abstract**: There has been unprecedented interest in developing agents that expand the boundary of scientific discovery, primarily by optimizing quantitative objective functions specified by scientists. However, for grand challenges in science , these objectives are only imperfect proxies. We argue that automating objective function design is a central, yet unmet requirement for scientific discovery agents. In this work, we introduce the Scientific Autonomous Goal-evolving Agent (SAGA) to amend this challenge. SAGA employs a bi-level architecture in which an outer loop of LLM agents analyzes optimization outcomes, proposes new objectives, and converts them into computable scoring functions, while an inner loop performs solution optimization under the current objectives. This bi-level design enables systematic exploration of the space of objectives and their trade-offs, rather than treating them as fixed inputs. We demonstrate the framework through a broad spectrum of applications, including antibiotic design, inorganic materials design, functional DNA sequence design, and chemical process design, showing that automating objective formulation can substantially improve the effectiveness of scientific discovery agents. 

---
# MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting 

**Authors**: Marc S. Montalvo, Hamed Yaghoobian  

**Link**: [PDF](https://arxiv.org/pdf/2512.21878)  

**Abstract**: Recent advances in large language models (LLMs) are transforming data-intensive domains, with finance representing a high-stakes environment where transparent and reproducible analysis of heterogeneous signals is essential. Traditional quantitative methods remain vulnerable to survivorship bias, while many AI-driven approaches struggle with signal integration, reproducibility, and computational efficiency. We introduce MASFIN, a modular multi-agent framework that integrates LLMs with structured financial metrics and unstructured news, while embedding explicit bias-mitigation protocols. The system leverages GPT-4.1-nano for reproducability and cost-efficient inference and generates weekly portfolios of 15-30 equities with allocation weights optimized for short-term performance. In an eight-week evaluation, MASFIN delivered a 7.33% cumulative return, outperforming the S&P 500, NASDAQ-100, and Dow Jones benchmarks in six of eight weeks, albeit with higher volatility. These findings demonstrate the promise of bias-aware, generative AI frameworks for financial forecasting and highlight opportunities for modular multi-agent design to advance practical, transparent, and reproducible approaches in quantitative finance. 

---
# Optimizing Resource Allocation for Geographically-Distributed Inference by Large Language Models 

**Authors**: Tingyang Sun, Ting He, Bo Ji, Parimal Parag  

**Link**: [PDF](https://arxiv.org/pdf/2512.21884)  

**Abstract**: Large language models have demonstrated extraordinary performance in many AI tasks but are expensive to use, even after training, due to their requirement of high-end GPUs. Recently, a distributed system called PETALS was developed to lower the barrier for deploying LLMs by splitting the model blocks across multiple servers with low-end GPUs distributed over the Internet, which was much faster than swapping the model parameters between the GPU memory and other cheaper but slower local storage media. However, the performance of such a distributed system critically depends on the resource allocation, and how to do so optimally remains unknown. In this work, we present the first systematic study of the resource allocation problem in distributed LLM inference, with focus on two important decisions: block placement and request routing. Our main results include: experimentally validated performance models that can predict the inference performance under given block placement and request routing decisions, a formulation of the offline optimization of block placement and request routing as a mixed integer linear programming problem together with the NP-hardness proof and a polynomial-complexity algorithm with guaranteed performance, and an adaptation of the offline algorithm for the online setting with the same performance guarantee under bounded load. Through both experiments and experimentally-validated simulations, we have verified that the proposed solution can substantially reduce the inference time compared to the state-of-the-art solution in diverse settings with geographically-distributed servers. As a byproduct, we have also developed a light-weighted CPU-only simulator capable of predicting the performance of distributed LLM inference on GPU servers, which can evaluate large deployments and facilitate future research for researchers with limited GPU access. 

---
# CricBench: A Multilingual Benchmark for Evaluating LLMs in Cricket Analytics 

**Authors**: Vaibhav Devraj, Dhruv Kumar, Jagat Sesh Challa  

**Link**: [PDF](https://arxiv.org/pdf/2512.21877)  

**Abstract**: Cricket is the second most popular sport globally, commanding a massive following of over 2.5 billion fans globally. Enthusiasts and analysts frequently seek advanced statistical insights, such as long-term historical performance trends or complex player comparisons, that are often unavailable through standard web searches. While Large Language Models (LLMs) have advanced significantly in Text-to-SQL tasks, their capability to handle the domain-specific nuances, complex schema variations, and multilingual requirements inherent to sports analytics remains under-explored. To investigate this potential capability gap, we present CricBench, a comprehensive benchmark suite for evaluating LLMs on specialized cricket data. To curate a "Gold Standard" dataset, we collaborate with domain experts in cricket and SQL to manually author complex queries, ensuring logical correctness. Recognizing linguistic diversity, we construct the benchmark in both English and Hindi, establishing a framework that is open for further extension to other regional languages. We evaluate six state-of-the-art models, including GPT-4o, Claude 3.7 Sonnet, and open-source models, using a strict evaluation protocol. Our results reveal that high performance on general benchmarks does not guarantee success in specialized domains. While the open-weights reasoning model DeepSeek R1 achieves state-of-the-art performance (50.6%), surpassing proprietary giants like Claude 3.7 Sonnet (47.7%) and GPT-4o (33.7%), it still exhibits a significant accuracy drop when moving from general benchmarks (BIRD) to CricBench. Furthermore, we observe that code-mixed Hindi queries frequently yield parity or higher accuracy compared to English, challenging the assumption that English is the optimal prompt language for specialized SQL tasks. 

---
# HeartBench: Probing Core Dimensions of Anthropomorphic Intelligence in LLMs 

**Authors**: Jiaxin Liu, Peiyi Tu, Wenyu Chen, Yihong Zhuang, Xinxia Ling, Anji Zhou, Chenxi Wang, Zhuo Han, Zhengkai Yang, Junbo Zhao, Zenan Huang, Yuanyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21849)  

**Abstract**: While Large Language Models (LLMs) have achieved remarkable success in cognitive and reasoning benchmarks, they exhibit a persistent deficit in anthropomorphic intelligence-the capacity to navigate complex social, emotional, and ethical nuances. This gap is particularly acute in the Chinese linguistic and cultural context, where a lack of specialized evaluation frameworks and high-quality socio-emotional data impedes progress. To address these limitations, we present HeartBench, a framework designed to evaluate the integrated emotional, cultural, and ethical dimensions of Chinese LLMs. Grounded in authentic psychological counseling scenarios and developed in collaboration with clinical experts, the benchmark is structured around a theory-driven taxonomy comprising five primary dimensions and 15 secondary capabilities. We implement a case-specific, rubric-based methodology that translates abstract human-like traits into granular, measurable criteria through a ``reasoning-before-scoring'' evaluation protocol. Our assessment of 13 state-of-the-art LLMs indicates a substantial performance ceiling: even leading models achieve only 60% of the expert-defined ideal score. Furthermore, analysis using a difficulty-stratified ``Hard Set'' reveals a significant performance decay in scenarios involving subtle emotional subtexts and complex ethical trade-offs. HeartBench establishes a standardized metric for anthropomorphic AI evaluation and provides a methodological blueprint for constructing high-quality, human-aligned training data. 

---
# Five Years of SciCap: What We Learned and Future Directions for Scientific Figure Captioning 

**Authors**: Ting-Hao K.Huang, Ryan A. Rossi, Sungchul Kim, Tong Yu, Ting-Yao E. Hsu, Ho Yin, C. Lee Giles  

**Link**: [PDF](https://arxiv.org/pdf/2512.21789)  

**Abstract**: Between 2021 and 2025, the SciCap project grew from a small seed-funded idea at The Pennsylvania State University (Penn State) into one of the central efforts shaping the scientific figure-captioning landscape. Supported by a Penn State seed grant, Adobe, and the Alfred P. Sloan Foundation, what began as our attempt to test whether domain-specific training, which was successful in text models like SciBERT, could also work for figure captions expanded into a multi-institution collaboration. Over these five years, we curated, released, and continually updated a large collection of figure-caption pairs from arXiv papers, conducted extensive automatic and human evaluations on both generated and author-written captions, navigated the rapid rise of large language models (LLMs), launched annual challenges, and built interactive systems that help scientists write better captions. In this piece, we look back at the first five years of SciCap and summarize the key technical and methodological lessons we learned. We then outline five major unsolved challenges and propose directions for the next phase of research in scientific figure captioning. 

---
# HELP: Hierarchical Embodied Language Planner for Household Tasks 

**Authors**: Alexandr V. Korchemnyi, Anatoly O. Onishchenko, Eva A. Bakaeva, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2512.21723)  

**Abstract**: Embodied agents tasked with complex scenarios, whether in real or simulated environments, rely heavily on robust planning capabilities. When instructions are formulated in natural language, large language models (LLMs) equipped with extensive linguistic knowledge can play this role. However, to effectively exploit the ability of such models to handle linguistic ambiguity, to retrieve information from the environment, and to be based on the available skills of an agent, an appropriate architecture must be designed. We propose a Hierarchical Embodied Language Planner, called HELP, consisting of a set of LLM-based agents, each dedicated to solving a different subtask. We evaluate the proposed approach on a household task and perform real-world experiments with an embodied agent. We also focus on the use of open source LLMs with a relatively small number of parameters, to enable autonomous deployment. 

---
# Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought 

**Authors**: Yuyi Zhang, Boyu Tang, Tianjie Ju, Sufeng Duan, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21711)  

**Abstract**: Latent tokens are gaining attention for enhancing reasoning in large language models (LLMs), yet their internal mechanisms remain unclear. This paper examines the problem from a reliability perspective, uncovering fundamental weaknesses: latent tokens function as uninterpretable placeholders rather than encoding faithful reasoning. While resistant to perturbation, they promote shortcut usage over genuine reasoning. We focus on Chain-of-Continuous-Thought (COCONUT), which claims better efficiency and stability than explicit Chain-of-Thought (CoT) while maintaining performance. We investigate this through two complementary approaches. First, steering experiments perturb specific token subsets, namely COCONUT and explicit CoT. Unlike CoT tokens, COCONUT tokens show minimal sensitivity to steering and lack reasoning-critical information. Second, shortcut experiments evaluate models under biased and out-of-distribution settings. Results on MMLU and HotpotQA demonstrate that COCONUT consistently exploits dataset artifacts, inflating benchmark performance without true reasoning. These findings reposition COCONUT as a pseudo-reasoning mechanism: it generates plausible traces that conceal shortcut dependence rather than faithfully representing reasoning processes. 

---
# LLM-I2I: Boost Your Small Item2Item Recommendation Model with Large Language Model 

**Authors**: Yinfu Feng, Yanjing Wu, Rong Xiao, Xiaoyi Zen  

**Link**: [PDF](https://arxiv.org/pdf/2512.21595)  

**Abstract**: Item-to-Item (I2I) recommendation models are widely used in real-world systems due to their scalability, real-time capabilities, and high recommendation quality. Research to enhance I2I performance focuses on two directions: 1) model-centric approaches, which adopt deeper architectures but risk increased computational costs and deployment complexity, and 2) data-centric methods, which refine training data without altering models, offering cost-effectiveness but struggling with data sparsity and noise. To address these challenges, we propose LLM-I2I, a data-centric framework leveraging Large Language Models (LLMs) to mitigate data quality issues. LLM-I2I includes (1) an LLM-based generator that synthesizes user-item interactions for long-tail items, alleviating data sparsity, and (2) an LLM-based discriminator that filters noisy interactions from real and synthetic data. The refined data is then fused to train I2I models. Evaluated on industry (AEDS) and academic (ARD) datasets, LLM-I2I consistently improves recommendation accuracy, particularly for long-tail items. Deployed on a large-scale cross-border e-commerce platform, it boosts recall number (RN) by 6.02% and gross merchandise value (GMV) by 1.22% over existing I2I models. This work highlights the potential of LLMs in enhancing data-centric recommendation systems without modifying model architectures. 

---
# A Unified Definition of Hallucination, Or: It's the World Model, Stupid 

**Authors**: Emmy Liu, Varun Gangal, Chelsea Zou, Xiaoqi Huang, Michael Yu, Alex Chang, Zhuofu Tao, Sachin Kumar, Steven Y. Feng  

**Link**: [PDF](https://arxiv.org/pdf/2512.21577)  

**Abstract**: Despite numerous attempts to solve the issue of hallucination since the inception of neural language models, it remains a problem in even frontier large language models today. Why is this the case? We walk through definitions of hallucination used in the literature from a historical perspective up to the current day, and fold them into a single definition of hallucination, wherein different prior definitions focus on different aspects of our definition. At its core, we argue that hallucination is simply inaccurate (internal) world modeling, in a form where it is observable to the user (e.g., stating a fact which contradicts a knowledge base, or producing a summary which contradicts a known source). By varying the reference world model as well as the knowledge conflict policy (e.g., knowledge base vs. in-context), we arrive at the different existing definitions of hallucination present in the literature.
We argue that this unified view is useful because it forces evaluations to make clear their assumed "world" or source of truth, clarifies what should and should not be called hallucination (as opposed to planning or reward/incentive-related errors), and provides a common language to compare benchmarks and mitigation techniques. Building on this definition, we outline plans for a family of benchmarks in which hallucinations are defined as mismatches with synthetic but fully specified world models in different environments, and sketch out how these benchmarks can use such settings to stress-test and improve the world modeling components of language models. 

---
# Selective LLM-Guided Regularization for Enhancing Recommendation Models 

**Authors**: Shanglin Yang, Zhan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2512.21526)  

**Abstract**: Large language models provide rich semantic priors and strong reasoning capabilities, making them promising auxiliary signals for recommendation. However, prevailing approaches either deploy LLMs as standalone recommender or apply global knowledge distillation, both of which suffer from inherent drawbacks. Standalone LLM recommender are costly, biased, and unreliable across large regions of the user item space, while global distillation forces the downstream model to imitate LLM predictions even when such guidance is inaccurate. Meanwhile, recent studies show that LLMs excel particularly in re-ranking and challenging scenarios, rather than uniformly across all this http URL introduce Selective LLM Guided Regularization, a model-agnostic and computation efficient framework that activates LLM based pairwise ranking supervision only when a trainable gating mechanism informing by user history length, item popularity, and model uncertainty predicts the LLM to be reliable. All LLM scoring is performed offline, transferring knowledge without increasing inference cost. Experiments across multiple datasets show that this selective strategy consistently improves overall accuracy and yields substantial gains in cold start and long tail regimes, outperforming global distillation baselines. 

---
# MotionTeller: Multi-modal Integration of Wearable Time-Series with LLMs for Health and Behavioral Understanding 

**Authors**: Aiwei Zhang, Arvind Pillai, Andrew Campbell, Nicholas C. Jacobson  

**Link**: [PDF](https://arxiv.org/pdf/2512.21506)  

**Abstract**: As wearable sensing becomes increasingly pervasive, a key challenge remains: how can we generate natural language summaries from raw physiological signals such as actigraphy - minute-level movement data collected via accelerometers? In this work, we introduce MotionTeller, a generative framework that natively integrates minute-level wearable activity data with large language models (LLMs). MotionTeller combines a pretrained actigraphy encoder with a lightweight projection module that maps behavioral embeddings into the token space of a frozen decoder-only LLM, enabling free-text, autoregressive generation of daily behavioral summaries. We construct a novel dataset of 54383 (actigraphy, text) pairs derived from real-world NHANES recordings, and train the model using cross-entropy loss with supervision only on the language tokens. MotionTeller achieves high semantic fidelity (BERTScore-F1 = 0.924) and lexical accuracy (ROUGE-1 = 0.722), outperforming prompt-based baselines by 7 percent in ROUGE-1. The average training loss converges to 0.38 by epoch 15, indicating stable optimization. Qualitative analysis confirms that MotionTeller captures circadian structure and behavioral transitions, while PCA plots reveal enhanced cluster alignment in embedding space post-training. Together, these results position MotionTeller as a scalable, interpretable system for transforming wearable sensor data into fluent, human-centered descriptions, introducing new pathways for behavioral monitoring, clinical review, and personalized health interventions. 

---
# LLM-Driven Feature-Level Adversarial Attacks on Android Malware Detectors 

**Authors**: Tianwei Lan, Farid Naït-Abdesselam  

**Link**: [PDF](https://arxiv.org/pdf/2512.21404)  

**Abstract**: The rapid growth in both the scale and complexity of Android malware has driven the widespread adoption of machine learning (ML) techniques for scalable and accurate malware detection. Despite their effectiveness, these models remain vulnerable to adversarial attacks that introduce carefully crafted feature-level perturbations to evade detection while preserving malicious functionality. In this paper, we present LAMLAD, a novel adversarial attack framework that exploits the generative and reasoning capabilities of large language models (LLMs) to bypass ML-based Android malware classifiers. LAMLAD employs a dual-agent architecture composed of an LLM manipulator, which generates realistic and functionality-preserving feature perturbations, and an LLM analyzer, which guides the perturbation process toward successful evasion. To improve efficiency and contextual awareness, LAMLAD integrates retrieval-augmented generation (RAG) into the LLM pipeline. Focusing on Drebin-style feature representations, LAMLAD enables stealthy and high-confidence attacks against widely deployed Android malware detection systems. We evaluate LAMLAD against three representative ML-based Android malware detectors and compare its performance with two state-of-the-art adversarial attack methods. Experimental results demonstrate that LAMLAD achieves an attack success rate (ASR) of up to 97%, requiring on average only three attempts per adversarial sample, highlighting its effectiveness, efficiency, and adaptability in practical adversarial settings. Furthermore, we propose an adversarial training-based defense strategy that reduces the ASR by more than 30% on average, significantly enhancing model robustness against LAMLAD-style attacks. 

---
# Morality is Contextual: Learning Interpretable Moral Contexts from Human Data with Probabilistic Clustering and Large Language Models 

**Authors**: Geoffroy Morlat, Marceau Nahon, Augustin Chartouny, Raja Chatila, Ismael T. Freire, Mehdi Khamassi  

**Link**: [PDF](https://arxiv.org/pdf/2512.21439)  

**Abstract**: Moral actions are judged not only by their outcomes but by the context in which they occur. We present COMETH (Contextual Organization of Moral Evaluation from Textual Human inputs), a framework that integrates a probabilistic context learner with LLM-based semantic abstraction and human moral evaluations to model how context shapes the acceptability of ambiguous actions. We curate an empirically grounded dataset of 300 scenarios across six core actions (violating Do not kill, Do not deceive, and Do not break the law) and collect ternary judgments (Blame/Neutral/Support) from N=101 participants. A preprocessing pipeline standardizes actions via an LLM filter and MiniLM embeddings with K-means, producing robust, reproducible core-action clusters. COMETH then learns action-specific moral contexts by clustering scenarios online from human judgment distributions using principled divergence criteria. To generalize and explain predictions, a Generalization module extracts concise, non-evaluative binary contextual features and learns feature weights in a transparent likelihood-based model. Empirically, COMETH roughly doubles alignment with majority human judgments relative to end-to-end LLM prompting (approx. 60% vs. approx. 30% on average), while revealing which contextual features drive its predictions. The contributions are: (i) an empirically grounded moral-context dataset, (ii) a reproducible pipeline combining human judgments with model-based context learning and LLM semantics, and (iii) an interpretable alternative to end-to-end LLMs for context-sensitive moral prediction and explanation. 

---
# Oogiri-Master: Benchmarking Humor Understanding via Oogiri 

**Authors**: Soichiro Murakami, Hidetaka Kamigaito, Hiroya Takamura, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2512.21494)  

**Abstract**: Humor is a salient testbed for human-like creative thinking in large language models (LLMs). We study humor using the Japanese creative response game Oogiri, in which participants produce witty responses to a given prompt, and ask the following research question: What makes such responses funny to humans? Previous work has offered only limited reliable means to answer this question. Existing datasets contain few candidate responses per prompt, expose popularity signals during ratings, and lack objective and comparable metrics for funniness. Thus, we introduce Oogiri-Master and Oogiri-Corpus, which are a benchmark and dataset designed to enable rigorous evaluation of humor understanding in LLMs. Each prompt is paired with approximately 100 diverse candidate responses, and funniness is rated independently by approximately 100 human judges without access to others' ratings, reducing popularity bias and enabling robust aggregation. Using Oogiri-Corpus, we conduct a quantitative analysis of the linguistic factors associated with funniness, such as text length, ambiguity, and incongruity resolution, and derive objective metrics for predicting human judgments. Subsequently, we benchmark a range of LLMs and human baselines in Oogiri-Master, demonstrating that state-of-the-art models approach human performance and that insight-augmented prompting improves the model performance. Our results provide a principled basis for evaluating and advancing humor understanding in LLMs. 

---
# Teaching People LLM's Errors and Getting it Right 

**Authors**: Nathan Stringham, Fateme Hashemi Chaleshtori, Xinyuan Yan, Zhichao Xu, Bei Wang, Ana Marasović  

**Link**: [PDF](https://arxiv.org/pdf/2512.21422)  

**Abstract**: People use large language models (LLMs) when they should not. This is partly because they see LLMs compose poems and answer intricate questions, so they understandably, but incorrectly, assume LLMs won't stumble on basic tasks like simple arithmetic. Prior work has tried to address this by clustering instance embeddings into regions where an LLM is likely to fail and automatically describing patterns in these regions. The found failure patterns are taught to users to mitigate their overreliance. Yet, this approach has not fully succeeded. In this analysis paper, we aim to understand why.
We first examine whether the negative result stems from the absence of failure patterns. We group instances in two datasets by their meta-labels and evaluate an LLM's predictions on these groups. We then define criteria to flag groups that are sizable and where the LLM is error-prone, and find meta-label groups that meet these criteria. Their meta-labels are the LLM's failure patterns that could be taught to users, so they do exist. We next test whether prompting and embedding-based approaches can surface these known failures. Without this, users cannot be taught about them to reduce their overreliance. We find mixed results across methods, which could explain the negative result. Finally, we revisit the final metric that measures teaching effectiveness. We propose to assess a user's ability to effectively use the given failure patterns to anticipate when an LLM is error-prone. A user study shows a positive effect from teaching with this metric, unlike the human-AI team accuracy. Our findings show that teaching failure patterns could be a viable approach to mitigating overreliance, but success depends on better automated failure-discovery methods and using metrics like ours. 

---
# AInsteinBench: Benchmarking Coding Agents on Scientific Repositories 

**Authors**: Titouan Duston, Shuo Xin, Yang Sun, Daoguang Zan, Aoyan Li, Shulin Xin, Kai Shen, Yixiao Chen, Qiming Sun, Ge Zhang, Jiashuo Liu, Huan Zhou, Jingkai Liu, Zhichen Pu, Yuanheng Wang, Bo-Xuan Ge, Xin Tong, Fei Ye, Zhi-Chao Zhao, Wen-Biao Han, Zhoujian Cao, Yueran Zhao, Weiluo Ren, Qingshen Long, Yuxiao Liu, Anni Huang, Yidi Du, Yuanyuan Rong, Jiahao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2512.21373)  

**Abstract**: We introduce AInsteinBench, a large-scale benchmark for evaluating whether large language model (LLM) agents can operate as scientific computing development agents within real research software ecosystems. Unlike existing scientific reasoning benchmarks which focus on conceptual knowledge, or software engineering benchmarks that emphasize generic feature implementation and issue resolving, AInsteinBench evaluates models in end-to-end scientific development settings grounded in production-grade scientific repositories. The benchmark consists of tasks derived from maintainer-authored pull requests across six widely used scientific codebases, spanning quantum chemistry, quantum computing, molecular dynamics, numerical relativity, fluid dynamics, and cheminformatics. All benchmark tasks are carefully curated through multi-stage filtering and expert review to ensure scientific challenge, adequate test coverage, and well-calibrated difficulty. By leveraging evaluation in executable environments, scientifically meaningful failure modes, and test-driven verification, AInsteinBench measures a model's ability to move beyond surface-level code generation toward the core competencies required for computational scientific research. 

---
# Reflection-Driven Control for Trustworthy Code Agents 

**Authors**: Bin Wang, Jiazheng Quan, Xingrui Yu, Hansen Hu, Yuhao, Ivor Tsang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21354)  

**Abstract**: Contemporary large language model (LLM) agents are remarkably capable, but they still lack reliable safety controls and can produce unconstrained, unpredictable, and even actively harmful outputs. To address this, we introduce Reflection-Driven Control, a standardized and pluggable control module that can be seamlessly integrated into general agent architectures. Reflection-Driven Control elevates "self-reflection" from a post hoc patch into an explicit step in the agent's own reasoning process: during generation, the agent continuously runs an internal reflection loop that monitors and evaluates its own decision path. When potential risks are detected, the system retrieves relevant repair examples and secure coding guidelines from an evolving reflective memory, injecting these evidence-based constraints directly into subsequent reasoning steps. We instantiate Reflection-Driven Control in the setting of secure code generation and systematically evaluate it across eight classes of security-critical programming tasks. Empirical results show that Reflection-Driven Control substantially improves the security and policy compliance of generated code while largely preserving functional correctness, with minimal runtime and token overhead. Taken together, these findings indicate that Reflection-Driven Control is a practical path toward trustworthy AI coding agents: it enables designs that are simultaneously autonomous, safer by construction, and auditable. 

---
# From Questions to Clinical Recommendations: Large Language Models Driving Evidence-Based Clinical Decision Making 

**Authors**: Dubai Li, Nan Jiang, Kangping Huang, Ruiqi Tu, Shuyu Ouyang, Huayu Yu, Lin Qiao, Chen Yu, Tianshu Zhou, Danyang Tong, Qian Wang, Mengtao Li, Xiaofeng Zeng, Yu Tian, Xinping Tian, Jingsong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10282)  

**Abstract**: Clinical evidence, derived from rigorous research and data analysis, provides healthcare professionals with reliable scientific foundations for informed decision-making. Integrating clinical evidence into real-time practice is challenging due to the enormous workload, complex professional processes, and time constraints. This highlights the need for tools that automate evidence synthesis to support more efficient and accurate decision making in clinical settings. This study introduces Quicker, an evidence-based clinical decision support system powered by large language models (LLMs), designed to automate evidence synthesis and generate clinical recommendations modeled after standard clinical guideline development processes. Quicker implements a fully automated chain that covers all phases, from questions to clinical recommendations, and further enables customized decision-making through integrated tools and interactive user interfaces. To evaluate Quicker's capabilities, we developed the Q2CRBench-3 benchmark dataset, based on clinical guideline development records for three different diseases. Experimental results highlighted Quicker's strong performance, with fine-grained question decomposition tailored to user preferences, retrieval sensitivities comparable to human experts, and literature screening performance approaching comprehensive inclusion of relevant studies. In addition, Quicker-assisted evidence assessment effectively supported human reviewers, while Quicker's recommendations were more comprehensive and logically coherent than those of clinicians. In system-level testing, collaboration between a single reviewer and Quicker reduced the time required for recommendation development to 20-40 minutes. In general, our findings affirm the potential of Quicker to help physicians make quicker and more reliable evidence-based clinical decisions. 

---
# Broken Words, Broken Performance: Effect of Tokenization on Performance of LLMs 

**Authors**: Sachin Pawar, Manoj Apte, Kshitij Jadhav, Girish Keshav Palshikar, Nitin Ramrakhiyani  

**Link**: [PDF](https://arxiv.org/pdf/2512.21933)  

**Abstract**: Tokenization is the first step in training any Large Language Model (LLM), where the text is split into a sequence of tokens as per the model's fixed vocabulary. This tokenization in LLMs is different from the traditional tokenization in NLP where the text is split into a sequence of "natural" words. In LLMs, a natural word may also be broken into multiple tokens due to limited vocabulary size of the LLMs (e.g., Mistral's tokenizer splits "martial" into "mart" and "ial"). In this paper, we hypothesize that such breaking of natural words negatively impacts LLM performance on various NLP tasks. To quantify this effect, we propose a set of penalty functions that compute a tokenization penalty for a given text for a specific LLM, indicating how "bad" the tokenization is. We establish statistical significance of our hypothesis on multiple NLP tasks for a set of different LLMs. 

---
# Introducing TrGLUE and SentiTurca: A Comprehensive Benchmark for Turkish General Language Understanding and Sentiment Analysis 

**Authors**: Duygu Altinok  

**Link**: [PDF](https://arxiv.org/pdf/2512.22100)  

**Abstract**: Evaluating the performance of various model architectures, such as transformers, large language models (LLMs), and other NLP systems, requires comprehensive benchmarks that measure performance across multiple dimensions. Among these, the evaluation of natural language understanding (NLU) is particularly critical as it serves as a fundamental criterion for assessing model capabilities. Thus, it is essential to establish benchmarks that enable thorough evaluation and analysis of NLU abilities from diverse perspectives. While the GLUE benchmark has set a standard for evaluating English NLU, similar benchmarks have been developed for other languages, such as CLUE for Chinese, FLUE for French, and JGLUE for Japanese. However, no comparable benchmark currently exists for the Turkish language. To address this gap, we introduce TrGLUE, a comprehensive benchmark encompassing a variety of NLU tasks for Turkish. In addition, we present SentiTurca, a specialized benchmark for sentiment analysis. To support researchers, we also provide fine-tuning and evaluation code for transformer-based models, facilitating the effective use of these benchmarks. TrGLUE comprises Turkish-native corpora curated to mirror the domains and task formulations of GLUE-style evaluations, with labels obtained through a semi-automated pipeline that combines strong LLM-based annotation, cross-model agreement checks, and subsequent human validation. This design prioritizes linguistic naturalness, minimizes direct translation artifacts, and yields a scalable, reproducible workflow. With TrGLUE, our goal is to establish a robust evaluation framework for Turkish NLU, empower researchers with valuable resources, and provide insights into generating high-quality semi-automated datasets. 

---
# Explainable Statute Prediction via Attention-based Model and LLM Prompting 

**Authors**: Sachin Pawar, Girish Keshav Palshikar, Anindita Sinha Banerjee, Nitin Ramrakhiyani, Basit Ali  

**Link**: [PDF](https://arxiv.org/pdf/2512.21902)  

**Abstract**: In this paper, we explore the problem of automatic statute prediction where for a given case description, a subset of relevant statutes are to be predicted. Here, the term "statute" refers to a section, a sub-section, or an article of any specific Act. Addressing this problem would be useful in several applications such as AI-assistant for lawyers and legal question answering system. For better user acceptance of such Legal AI systems, we believe the predictions should also be accompanied by human understandable explanations. We propose two techniques for addressing this problem of statute prediction with explanations -- (i) AoS (Attention-over-Sentences) which uses attention over sentences in a case description to predict statutes relevant for it and (ii) LLMPrompt which prompts an LLM to predict as well as explain relevance of a certain statute. AoS uses smaller language models, specifically sentence transformers and is trained in a supervised manner whereas LLMPrompt uses larger language models in a zero-shot manner and explores both standard as well as Chain-of-Thought (CoT) prompting techniques. Both these models produce explanations for their predictions in human understandable forms. We compare statute prediction performance of both the proposed techniques with each other as well as with a set of competent baselines, across two popular datasets. Also, we evaluate the quality of the generated explanations through an automated counter-factual manner as well as through human evaluation. 

---
# TimeBill: Time-Budgeted Inference for Large Language Models 

**Authors**: Qi Fan, An Zou, Yehan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2512.21859)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in time-critical systems, such as robotics, autonomous driving, embodied intelligence, and industrial automation, where generating accurate responses within a given time budget is crucial for decision-making, control, or safety-critical tasks. However, the auto-regressive generation process of LLMs makes it challenging to model and estimate the end-to-end execution time. Furthermore, existing efficient inference methods based on a fixed key-value (KV) cache eviction ratio struggle to adapt to varying tasks with diverse time budgets, where an improper eviction ratio may lead to incomplete inference or a drop in response performance. In this paper, we propose TimeBill, a novel time-budgeted inference framework for LLMs that balances the inference efficiency and response performance. To be more specific, we propose a fine-grained response length predictor (RLP) and an execution time estimator (ETE) to accurately predict the end-to-end execution time of LLMs. Following this, we develop a time-budgeted efficient inference approach that adaptively adjusts the KV cache eviction ratio based on execution time prediction and the given time budget. Finally, through extensive experiments, we demonstrate the advantages of TimeBill in improving task completion rate and maintaining response performance under various overrun strategies. 

---
# Knowledge Reasoning of Large Language Models Integrating Graph-Structured Information for Pest and Disease Control in Tobacco 

**Authors**: Siyu Li, Chenwei Song, Wan Zhou, Xinyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21837)  

**Abstract**: This paper proposes a large language model (LLM) approach that integrates graph-structured information for knowledge reasoning in tobacco pest and disease control. Built upon the GraphRAG framework, the proposed method enhances knowledge retrieval and reasoning by explicitly incorporating structured information from a domain-specific knowledge graph. Specifically, LLMs are first leveraged to assist in the construction of a tobacco pest and disease knowledge graph, which organizes key entities such as diseases, symptoms, control methods, and their relationships. Based on this graph, relevant knowledge is retrieved and integrated into the reasoning process to support accurate answer generation. The Transformer architecture is adopted as the core inference model, while a graph neural network (GNN) is employed to learn expressive node representations that capture both local and global relational information within the knowledge graph. A ChatGLM-based model serves as the backbone LLM and is fine-tuned using LoRA to achieve parameter-efficient adaptation. Extensive experimental results demonstrate that the proposed approach consistently outperforms baseline methods across multiple evaluation metrics, significantly improving both the accuracy and depth of reasoning, particularly in complex multi-hop and comparative reasoning scenarios. 

---
# Method Decoration (DeMe): A Framework for LLM-Driven Adaptive Method Generation in Dynamic IoT Environments 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2512.21817)  

**Abstract**: Intelligent IoT systems increasingly rely on large language models (LLMs) to generate task-execution methods for dynamic environments. However, existing approaches lack the ability to systematically produce new methods when facing previously unseen situations, and they often depend on fixed, device-specific logic that cannot adapt to changing environmental this http URL this paper, we propose Method Decoration (DeMe), a general framework that modifies the method-generation path of an LLM using explicit decorations derived from hidden goals, accumulated learned methods, and environmental feedback. Unlike traditional rule augmentation, decorations in DeMe are not hardcoded; instead, they are extracted from universal behavioral principles, experience, and observed environmental differences. DeMe enables the agent to reshuffle the structure of its method path-through pre-decoration, post-decoration, intermediate-step modification, and step insertion-thereby producing context-aware, safety-aligned, and environment-adaptive methods. Experimental results show that method decoration allows IoT devices to derive ore appropriate methods when confronting unknown or faulty operating conditions. 

---
# AlignAR: Generative Sentence Alignment for Arabic-English Parallel Corpora of Legal and Literary Texts 

**Authors**: Baorong Huang, Ali Asiri  

**Link**: [PDF](https://arxiv.org/pdf/2512.21842)  

**Abstract**: High-quality parallel corpora are essential for Machine Translation (MT) research and translation teaching. However, Arabic-English resources remain scarce and existing datasets mainly consist of simple one-to-one mappings. In this paper, we present AlignAR, a generative sentence alignment method, and a new Arabic-English dataset comprising complex legal and literary texts. Our evaluation demonstrates that "Easy" datasets lack the discriminatory power to fully assess alignment methods. By reducing one-to-one mappings in our "Hard" subset, we exposed the limitations of traditional alignment methods. In contrast, LLM-based approaches demonstrated superior robustness, achieving an overall F1-score of 85.5%, a 9% improvement over previous methods. Our datasets and codes are open-sourced at this https URL. 

---
# MoRAgent: Parameter Efficient Agent Tuning with Mixture-of-Roles 

**Authors**: Jing Han, Binwei Yan, Tianyu Guo, Zheyuan Bai, Mengyu Zheng, Hanting Chen, Ying Nie  

**Link**: [PDF](https://arxiv.org/pdf/2512.21708)  

**Abstract**: Despite recent advancements of fine-tuning large language models (LLMs) to facilitate agent tasks, parameter-efficient fine-tuning (PEFT) methodologies for agent remain largely unexplored. In this paper, we introduce three key strategies for PEFT in agent tasks: 1) Inspired by the increasingly dominant Reason+Action paradigm, we first decompose the capabilities necessary for the agent tasks into three distinct roles: reasoner, executor, and summarizer. The reasoner is responsible for comprehending the user's query and determining the next role based on the execution trajectory. The executor is tasked with identifying the appropriate functions and parameters to invoke. The summarizer conveys the distilled information from conversations back to the user. 2) We then propose the Mixture-of-Roles (MoR) framework, which comprises three specialized Low-Rank Adaptation (LoRA) groups, each designated to fulfill a distinct role. By focusing on their respective specialized capabilities and engaging in collaborative interactions, these LoRAs collectively accomplish the agent task. 3) To effectively fine-tune the framework, we develop a multi-role data generation pipeline based on publicly available datasets, incorporating role-specific content completion and reliability verification. We conduct extensive experiments and thorough ablation studies on various LLMs and agent benchmarks, demonstrating the effectiveness of the proposed method. This project is publicly available at this https URL. 

---
# Heaven-Sent or Hell-Bent? Benchmarking the Intelligence and Defectiveness of LLM Hallucinations 

**Authors**: Chengxu Yang, Jingling Yuan, Siqi Cai, Jiawei Jiang, Chuang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21635)  

**Abstract**: Hallucinations in large language models (LLMs) are commonly regarded as errors to be minimized. However, recent perspectives suggest that some hallucinations may encode creative or epistemically valuable content, a dimension that remains underquantified in current literature. Existing hallucination detection methods primarily focus on factual consistency, struggling to handle heterogeneous scientific tasks and balance creativity with accuracy. To address these challenges, we propose HIC-Bench, a novel evaluation framework that categorizes hallucinations into Intelligent Hallucinations (IH) and Defective Hallucinations (DH), enabling systematic investigation of their interplay in LLM creativity. HIC-Bench features three core characteristics: (1) Structured IH/DH Assessment. using a multi-dimensional metric matrix integrating Torrance Tests of Creative Thinking (TTCT) metrics (Originality, Feasibility, Value) with hallucination-specific dimensions (scientific plausibility, factual deviation); (2) Cross-Domain Applicability. spanning ten scientific domains with open-ended innovation tasks; and (3) Dynamic Prompt Optimization. leveraging the Dynamic Hallucination Prompt (DHP) to guide models toward creative and reliable outputs. The evaluation process employs multiple LLM judges, averaging scores to mitigate bias, with human annotators verifying IH/DH classifications. Experimental results reveal a nonlinear relationship between IH and DH, demonstrating that creativity and correctness can be jointly optimized. These insights position IH as a catalyst for creativity and reveal the ability of LLM hallucinations to drive scientific this http URL, the HIC-Bench offers a valuable platform for advancing research into the creative intelligence of LLM hallucinations. 

---
# Query Carefully: Detecting the Unanswerables in Text-to-SQL Tasks 

**Authors**: Jasmin Saxer, Isabella Maria Aigner, Luise Linzmeier, Andreas Weiler, Kurt Stockinger  

**Link**: [PDF](https://arxiv.org/pdf/2512.21345)  

**Abstract**: Text-to-SQL systems allow non-SQL experts to interact with relational databases using natural language. However, their tendency to generate executable SQL for ambiguous, out-of-scope, or unanswerable queries introduces a hidden risk, as outputs may be misinterpreted as correct. This risk is especially serious in biomedical contexts, where precision is critical. We therefore present Query Carefully, a pipeline that integrates LLM-based SQL generation with explicit detection and handling of unanswerable inputs. Building on the OncoMX component of ScienceBenchmark, we construct OncoMX-NAQ (No-Answer Questions), a set of 80 no-answer questions spanning 8 categories (non-SQL, out-of-schema/domain, and multiple ambiguity types). Our approach employs llama3.3:70b with schema-aware prompts, explicit No-Answer Rules (NAR), and few-shot examples drawn from both answerable and unanswerable questions. We evaluate SQL exact match, result accuracy, and unanswerable-detection accuracy. On the OncoMX dev split, few-shot prompting with answerable examples increases result accuracy, and adding unanswerable examples does not degrade performance. On OncoMX-NAQ, balanced prompting achieves the highest unanswerable-detection accuracy (0.8), with near-perfect results for structurally defined categories (non-SQL, missing columns, out-of-domain) but persistent challenges for missing-value queries (0.5) and column ambiguity (0.3). A lightweight user interface surfaces interim SQL, execution results, and abstentions, supporting transparent and reliable text-to-SQL in biomedical applications. 

---
# CEMG: Collaborative-Enhanced Multimodal Generative Recommendation 

**Authors**: Yuzhen Lin, Hongyi Chen, Xuanjing Chen, Shaowen Wang, Ivonne Xu, Dongming Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21543)  

**Abstract**: Generative recommendation models often struggle with two key challenges: (1) the superficial integration of collaborative signals, and (2) the decoupled fusion of multimodal features. These limitations hinder the creation of a truly holistic item representation. To overcome this, we propose CEMG, a novel Collaborative-Enhaned Multimodal Generative Recommendation framework. Our approach features a Multimodal Fusion Layer that dynamically integrates visual and textual features under the guidance of collaborative signals. Subsequently, a Unified Modality Tokenization stage employs a Residual Quantization VAE (RQ-VAE) to convert this fused representation into discrete semantic codes. Finally, in the End-to-End Generative Recommendation stage, a large language model is fine-tuned to autoregressively generate these item codes. Extensive experiments demonstrate that CEMG significantly outperforms state-of-the-art baselines. 

---
